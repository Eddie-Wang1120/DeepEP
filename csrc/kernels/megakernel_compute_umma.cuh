// megakernel_compute_umma.cuh — S4.4 (route B2): Blackwell UMMA (tcgen05) + TMEM
// fused gate+up+SwiGLU compute for the megakernel, with 2CTA weight multicast TMA.
//
// Built on the VERIFIED standalone kernel compute_ref/umma_swiglu_2cta.cu (stage 3').
// Per-expert 2D TMA atoms (one [I,d] multicast atom per expert) avoid any 3D
// expert-dim TMA — each atom is exactly the 2D [N,K] multicast TMA proven in
// tutorial 04. The device kernel selects atoms by expert_id from arrays.
//
// Ref: MEGAKERNEL_COMPUTE_DESIGN.md I.9 (tile schedule), I.9.10 (route B2).
// Assumes hidden == intermediate == 4096 (I.9.0).
//
// USAGE (in megakernel.cu, an nvcc TU):
//   - MegaKernelState holds a `ComputeTmaAtoms* compute_tma;` device pointer.
//   - Host: build_compute_tma_atoms(host_struct, W_gate, W_up, E, I, d); upload.
//   - Device (compute_worker stage1, per 2-CTA cluster): call
//       umma_up_swiglu_tile(state->compute_tma, expert_id, i_tile, A_smem_ptr,
//                           route_w_ptr, act_out_ptr, ...);

#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <cuda_bf16.h>

namespace deep_ep {
namespace megakernel {
namespace umma {

using namespace cute;

#ifdef MK_PERF_TRACE
// Full per-tile UMMA timing breakdown (ns), accumulated by the group leader thread.
// Every internal phase of umma_up_swiglu_tile / umma_down_proj_tile is captured so a
// future perf run can pinpoint exactly which sub-step dominates p3a/p4a — no blind spot.
struct UmmaPerf {
    int64_t setup_ns;          // entry -> just before TMEM alloc/sync (fragment/partition setup)
    int64_t tmem_alloc_ns;     // TMEM allocate + its cluster_sync/umma_sync128 (first call only)
    int64_t prologue_ns;       // partition/TMA-setup + barrier init + cluster_sync before mainloop
    int64_t tma_wait_ns;       // accumulated wait_barrier(tma_barrier)
    int64_t mma_issue_ns;      // gemm() issue + umma_arrive (between tma wait and mma wait)
    int64_t mma_wait_ns;       // accumulated wait_barrier(mma_barrier)
    int64_t loop_other_ns;     // mainloop time not in tma_wait/mma_issue/mma_wait (copy t2r, loop overhead)
    int64_t cluster_sync_ns;   // accumulated cute::cluster_sync() (whole-CTA, 800 threads)
    int64_t epilogue_ns;       // SwiGLU/cast epilogue + final stores + umma_sync128
};
#endif

// Named barrier over exactly the 128 threads (4 warps) that run the UMMA kernel
// inside an 800-thread megakernel block. Plain __syncthreads() would wait for all
// 800 threads and deadlock, since only thread_id<128 enter this code path.
//
// IMPORTANT: barrier ids 0/1/2 are ALREADY in use by the dispatch/combine roles
// of this megakernel (megakernel.cu: `barrier.sync 0/1/2`). Reusing id 1 (the old
// value) aliased the combine forwarder's `barrier.sync 1`, corrupting the
// cluster-scoped tcgen05 alloc handshake and tripping
//   __cuda_sm10x_tcgen05_guardrail_trap_phase_invalid_during_alloc.
// Use id 8 (unused by any role) so the 128 UMMA threads sync in isolation.
CUTE_DEVICE void umma_sync128() {
    asm volatile("barrier.sync 8, 128;" ::: "memory");
}

using ElemAB  = cutlass::bfloat16_t;
using ElemAcc = float;

static constexpr int kTileM = 256;     // one 256-token batch (M)
static constexpr int kTileN = 256;     // one intermediate i-tile (N)
static constexpr int kAtomK = 16;
static constexpr int kKStep = 64;      // 4 x K16 per K tile

// ---- MMA / cluster / layout types (deduced once) ----
using MmaAtom_t = SM100_MMA_F16BF16_2x1SM_SS<ElemAB, ElemAB, ElemAcc, kTileM, kTileN,
                                             UMMA::Major::K, UMMA::Major::K>;
using TiledMMA_t = decltype(make_tiled_mma(MmaAtom_t{}));
using ClusterShape_t = decltype(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));

// mma_tiler = (256, 256, 64)
CUTE_HOST_DEVICE auto make_mma_tiler() {
    return make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kKStep>{});
}

// Deduce the swizzled SMEM B layout type for a [N=256, K=64] tile.
CUTE_HOST_DEVICE auto make_sB_layout() {
    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
    return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<ElemAB>{}, mma_shape_B);
}
CUTE_HOST_DEVICE auto make_sA_layout() {
    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
    return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<ElemAB>{}, mma_shape_A);
}

using SBLayout_t = decltype(make_sB_layout());

// One per-expert weight TMA atom for a [I, d] (= [N, K]) 2D tensor.
// We build it from a single-expert GMEM view; the TMA descriptor is bound to
// that expert's base pointer (W + e*I*d). This is exactly the 2D multicast atom
// from tutorial 04 / stage 3', so no new (3D) TMA path is introduced.
inline auto make_weight_tma_atom(const __nv_bfloat16* w_e_ptr, int I, int d) {
    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
    Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                              make_tile(typename TiledMMA_t::AtomThrID{}));
    auto sB_layout = make_sB_layout();
    auto mW = make_tensor(make_gmem_ptr(reinterpret_cast<const ElemAB*>(w_e_ptr)),
                          make_layout(make_shape(I, d), make_stride(d, Int<1>{})));
    return make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mW, sB_layout,
                                 mma_tiler, tiled_mma, cluster_layout_vmnk);
}

using WeightTmaAtom_t = decltype(make_weight_tma_atom((const __nv_bfloat16*)nullptr, 1, 1));

// Holds per-expert TMA atoms for W_gate and W_up. Stored in device memory and
// pointed to by MegaKernelState. Sized for up to kMaxLocalExperts.
static constexpr int kMaxLocalExperts = 64;

struct ComputeTmaAtoms {
    int num_experts;
    int I;
    int d;
    WeightTmaAtom_t wgate[kMaxLocalExperts];
    WeightTmaAtom_t wup[kMaxLocalExperts];
};

// ---- Host builder: fill ComputeTmaAtoms (call on host, then cudaMemcpy to device) ----
inline void build_compute_tma_atoms(ComputeTmaAtoms& atoms,
                                     const __nv_bfloat16* W_gate, const __nv_bfloat16* W_up,
                                     int E, int I, int d) {
    EP_HOST_ASSERT(E <= kMaxLocalExperts);
    atoms.num_experts = E;
    atoms.I = I;
    atoms.d = d;
    for (int e = 0; e < E; ++e) {
        const __nv_bfloat16* wg_e = W_gate + (size_t)e * I * d;
        const __nv_bfloat16* wu_e = W_up   + (size_t)e * I * d;
        atoms.wgate[e] = make_weight_tma_atom(wg_e, I, d);
        atoms.wup[e]   = make_weight_tma_atom(wu_e, I, d);
    }
}

// A-side TMA atom over a per-group contiguous input_buf [M=256, d]. Built on host
// (input_buf address is fixed after gemm_workspace allocation). One per group.
inline auto make_input_tma_atom(const __nv_bfloat16* input_buf_ptr, int M, int d) {
    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
    Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                              make_tile(typename TiledMMA_t::AtomThrID{}));
    auto sA_layout = make_sA_layout();
    auto mA = make_tensor(make_gmem_ptr(reinterpret_cast<const ElemAB*>(input_buf_ptr)),
                          make_layout(make_shape(M, d), make_stride(d, Int<1>{})));
    return make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA_layout,
                                 mma_tiler, tiled_mma, cluster_layout_vmnk);
}
using InputTmaAtom_t = decltype(make_input_tma_atom((const __nv_bfloat16*)nullptr, 1, 1));

// ---------------------------------------------------------------------------
// Device kernel: one 2-CTA cluster computes one [256,256] act tile for a given
// (expert, i_tile). Adapted directly from the verified standalone
// compute_ref/umma_swiglu_2cta.cu (stage 3'); the only differences are:
//   - A/Wg/Wu TMA atoms are passed in (A from input_buf, W per-expert).
//   - output writes act[:, i_tile*256:+256] to a per-group GMEM workspace.
//   - TMEM alloc/free are done per call (persistent-loop safe, R3).
//
// SMEM layout reuses the standalone kernel's SharedStorage shape.
// ---------------------------------------------------------------------------

template <class TypeAB, class ASmemLayout, class BSmemLayout>
struct ClusterSharedStorage {
    alignas(128) cute::ArrayEngine<TypeAB, cute::cosize_v<ASmemLayout>> A;
    alignas(128) cute::ArrayEngine<TypeAB, cute::cosize_v<BSmemLayout>> Bg;
    alignas(128) cute::ArrayEngine<TypeAB, cute::cosize_v<BSmemLayout>> Bu;
    alignas(16) cute::uint64_t mma_barrier;
    alignas(16) cute::uint64_t tma_barrier;
    alignas(16) cute::uint32_t tmem_base_ptr;
    CUTE_DEVICE constexpr auto tensor_sA()  { return make_tensor(make_smem_ptr(A.begin()),  ASmemLayout{}); }
    CUTE_DEVICE constexpr auto tensor_sBg() { return make_tensor(make_smem_ptr(Bg.begin()), BSmemLayout{}); }
    CUTE_DEVICE constexpr auto tensor_sBu() { return make_tensor(make_smem_ptr(Bu.begin()), BSmemLayout{}); }
};

// Per-cluster compute of one act tile. cluster_smem must point to enough dynamic
// SMEM for ClusterSharedStorage. block_rank_in_cluster()/cluster_sync used inside.
//   tma_A : input_buf TMA atom (this group)
//   tma_Bg/tma_Bu : weight TMA atoms (this expert)
//   i_tile : which 256-col N-tile of intermediate (0..15)
//   act_out : group act workspace base [M, I]; this writes [:, i_tile*256:+256]
//   route_w : [M] route weights (already gathered for this batch)
//   M, I, d : dims (M=256, I=d=4096)
// Standalone dealloc — call once after the persistent loop exits (all UMMA work done).
// Both CTAs of the cluster must call this together.
__device__ void umma_dealloc(char* cluster_smem) {
    using TmemAllocator = cute::TMEM::Allocator2Sm;
    TmemAllocator tmem_allocator{};
    auto sA_layout = make_sA_layout();
    auto sB_layout = make_sB_layout();
    using SMEM = ClusterSharedStorage<ElemAB, decltype(sA_layout), decltype(sB_layout)>;
    SMEM& smem = *reinterpret_cast<SMEM*>(cluster_smem);
    const bool active = (threadIdx.x < 128);
    if (active) umma_sync128();
    if (active && (threadIdx.x / 32 == 0)) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
    cute::cluster_sync();
}

template <class TmaA, class TmaB>
__device__ void umma_up_swiglu_tile(
    const TmaA& tma_A, const TmaB& tma_Bg, const TmaB& tma_Bu,
    int i_tile, __nv_bfloat16* act_out, const float* route_w,
    int M, int I, int d, char* cluster_smem, bool& tmem_allocated
#ifdef MK_PERF_TRACE
    , UmmaPerf* perf = nullptr
#endif
    ) {
#ifdef MK_PERF_TRACE
    // Only the block's thread 0 records (it is the compute group leader / perf_leader).
    const bool _perf_rec = (perf != nullptr && threadIdx.x == 0);
    int64_t _t_prev = _perf_rec ? globaltimer_ns() : 0;
    auto _perf_mark = [&](int64_t* slot) {
        if (!_perf_rec) return;
        int64_t now = globaltimer_ns();
        *slot += now - _t_prev;
        _t_prev = now;
    };
#endif

    // ALL 800 threads of the megakernel block enter here (the compute_worker no
    // longer gates on thread_id<128) so that cute::cluster_sync() — which needs
    // whole-CTA arrival — does not deadlock. Only the first 128 threads of each
    // CTA do real work (TMEM alloc, TMA/MMA, TMEM-load, GMEM write); the other
    // 672 threads only participate in the all-thread cluster_sync() calls and
    // must touch neither TMEM nor the output. umma_sync128() (barrier.sync 1,128)
    // syncs exactly those 128 active threads.
    const bool active = (threadIdx.x < 128);

    // Earliest diagnostic: confirm we entered umma_up_swiglu_tile
    // if (threadIdx.x == 0) {
    //     printf("[UMMA-ENTER-V0] block=%d active=%d\n",
    //            (int)blockIdx.x, (int)active);
    // }

    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
    Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                              make_tile(typename TiledMMA_t::AtomThrID{}));

    auto sA_layout = make_sA_layout();
    auto sB_layout = make_sB_layout();
    using SMEM = ClusterSharedStorage<ElemAB, decltype(sA_layout), decltype(sB_layout)>;
    SMEM& smem = *reinterpret_cast<SMEM*>(cluster_smem);

    // if (threadIdx.x == 0) {
    //     printf("[UMMA-STEP1] block=%d smem_ptr=%p sizeof_SMEM=%d\n",
    //            (int)blockIdx.x, (void*)cluster_smem, (int)sizeof(SMEM));
    // }

    // GMEM tensors: A = input_buf [M, d]; Wg/Wu via the tma atoms' tensors.
    // act_out tile view: D = act_out as [M, I], take N-tile = i_tile.
    auto mD = make_tensor(make_gmem_ptr(reinterpret_cast<ElemAB*>(act_out)),
                          make_layout(make_shape(M, I), make_stride(I, Int<1>{})));
    auto mR = make_tensor(make_gmem_ptr(route_w), make_layout(make_shape(M), make_stride(Int<1>{})));

    // mma_coord selects M-tile=0 (single 256 batch), N-tile=i_tile, K iterated.
    auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                                     0, i_tile, _);
    auto mma_coord = select<1,2,3>(mma_coord_vmnk);

    Tensor gD  = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});
    auto mma_v = get<0>(mma_coord_vmnk);
    ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
    Tensor tCgD = cta_mma.partition_C(gD);

    Tensor tCsA  = smem.tensor_sA();
    Tensor tCsBg = smem.tensor_sBg();
    Tensor tCsBu = smem.tensor_sBu();

    // On SM100 UMMA 2x1SM, make_fragment_A/B/C internally compute TMEM
    // descriptors using threadIdx. Only threads 0-127 map to valid TMEM
    // addresses; threads >= 128 would produce corrupt descriptors and may
    // silently damage HW state. Therefore these (and elect_one_sync) must
    // be guarded by `active`.
    //
    // We declare the tensors outside the `if` so the compiler can see the type
    // in both branches, but only the `active` path actually initializes them.
    using FragA_t  = decltype(cta_mma.make_fragment_A(tCsA));
    using FragB_t  = decltype(cta_mma.make_fragment_B(tCsBg));
    using FragC_t  = decltype(cta_mma.make_fragment_C(tCgD));
    FragA_t tCrA;
    FragB_t tCrBg;
    FragB_t tCrBu;
    FragC_t tCtAcc;
    uint32_t elect_one_thr  = 0;
    uint32_t elect_one_warp = 0;

    if (active) {
        tCrA   = cta_mma.make_fragment_A(tCsA);
        tCrBg  = cta_mma.make_fragment_B(tCsBg);
        tCrBu  = cta_mma.make_fragment_B(tCsBu);
        tCtAcc = cta_mma.make_fragment_C(tCgD);
        // if (threadIdx.x == 0) {
        //     printf("[UMMA-STEP2B] block=%d pre-elect\n", (int)blockIdx.x);
        // }
        elect_one_thr  = cute::elect_one_sync();
        elect_one_warp = (threadIdx.x / 32 == 0);
    }

    // if (threadIdx.x == 0) {
    //     printf("[UMMA-STEP3] block=%d post-elect\n", (int)blockIdx.x);
    // }

    using TmemAllocator = cute::TMEM::Allocator2Sm;
    TmemAllocator tmem_allocator{};
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->setup_ns);
#endif
    // Only allocate TMEM on first invocation; subsequent calls reuse the allocation.
    if (!tmem_allocated) {
        cute::cluster_sync();
        if (active && elect_one_warp) tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
        if (active) umma_sync128();
        tmem_allocated = true;
    } else {
        // Already allocated — just sync cluster so both CTAs are aligned.
        cute::cluster_sync();
    }
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->tmem_alloc_ns);
#endif
    if (active) tCtAcc.data() = smem.tmem_base_ptr;
    // if (threadIdx.x == 0) {
    //     printf("[UMMA-STEP5] block=%d\n", (int)blockIdx.x);
    // }

    auto cta_in_cluster = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
    auto elect_one_cta = get<0>(cta_in_cluster) == Int<0>{};

    // A/Wg/Wu TMA tensors and partitions (A from input_buf, Wg/Wu per-expert).
    Tensor mA_tma  = tma_A.get_tma_tensor(make_shape(M, d));
    Tensor mBg_tma = tma_Bg.get_tma_tensor(make_shape(I, d));
    Tensor mBu_tma = tma_Bu.get_tma_tensor(make_shape(I, d));
    Tensor gA  = local_tile(mA_tma,  mma_tiler, mma_coord, Step<_1, X,_1>{});
    Tensor gBg = local_tile(mBg_tma, mma_tiler, mma_coord, Step< X,_1,_1>{});
    Tensor gBu = local_tile(mBu_tma, mma_tiler, mma_coord, Step< X,_1,_1>{});
    Tensor tCgA  = cta_mma.partition_A(gA);
    Tensor tCgBg = cta_mma.partition_B(gBg);
    Tensor tCgBu = cta_mma.partition_B(gBu);

    auto [tAgA,  tAsA]  = tma_partition(tma_A,  get<2>(cta_in_cluster),
                                        make_layout(size<2>(cluster_layout_vmnk)),
                                        group_modes<0,3>(tCsA),  group_modes<0,3>(tCgA));
    auto [tBggBg, tBgsBg] = tma_partition(tma_Bg, get<1>(cta_in_cluster),
                                          make_layout(size<1>(cluster_layout_vmnk)),
                                          group_modes<0,3>(tCsBg), group_modes<0,3>(tCgBg));
    auto [tBugBu, tBusBu] = tma_partition(tma_Bu, get<1>(cta_in_cluster),
                                          make_layout(size<1>(cluster_layout_vmnk)),
                                          group_modes<0,3>(tCsBu), group_modes<0,3>(tCgBu));

    uint16_t mcast_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster);
    uint16_t mcast_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster);
    uint16_t mcast_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster) |
                       create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster);
    int txbytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))
                + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBgsBg));

    if (active && elect_one_warp && elect_one_thr) {
        int np = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
        cute::initialize_barrier(smem.mma_barrier, np);
        cute::initialize_barrier(smem.tma_barrier, 1);
    }
    int mma_phase = 0, tma_phase = 0;
    cute::cluster_sync();   // ALL 800 threads — whole-CTA arrival required

    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
    ThrCopy   thr_t2r = t2r.get_slice(threadIdx.x);
    Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
    Tensor tDgD   = thr_t2r.partition_D(tCgD);
    using AccT = typename decltype(tCtAcc)::value_type;
    Tensor tDrG = make_tensor<AccT>(shape(tDgD));
    Tensor tDrU = make_tensor<AccT>(shape(tDgD));

    int nKt = size<3>(tCgA);

    // One-shot diagnostic per block (persistent kernel — use __shared__ guard)
    __shared__ int s_umma_diag_done;
    if (threadIdx.x == 0) s_umma_diag_done = 0;
    cute::cluster_sync();  // reuse the upcoming barrier; safe since all 800 threads here
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->prologue_ns);
#endif

    for (int pass = 0; pass < 2; ++pass) {
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        if (active) {
            for (int k_tile = 0; k_tile < nKt; ++k_tile) {
                if (elect_one_warp && elect_one_thr) {
                    if (elect_one_cta) cute::set_barrier_transaction_bytes(smem.tma_barrier, txbytes);
                    copy(tma_A.with(smem.tma_barrier, mcast_a), tAgA(_,k_tile), tAsA);
                    if (pass == 0) copy(tma_Bg.with(smem.tma_barrier, mcast_b), tBggBg(_,k_tile), tBgsBg);
                    else           copy(tma_Bu.with(smem.tma_barrier, mcast_b), tBugBu(_,k_tile), tBusBu);
                }
#ifdef MK_PERF_TRACE
                if (_perf_rec) _perf_mark(&perf->loop_other_ns);
#endif
                if (elect_one_cta) {
                    cute::wait_barrier(smem.tma_barrier, tma_phase); tma_phase ^= 1;
#ifdef MK_PERF_TRACE
                    if (_perf_rec) _perf_mark(&perf->tma_wait_ns);
#endif
                    if (elect_one_warp) {
                        auto& tCrB = (pass == 0) ? tCrBg : tCrBu;
                        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
                            gemm(tiled_mma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
                            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                        }
                        cutlass::arch::umma_arrive_multicast_2x1SM(&smem.mma_barrier, mcast_c);
                    }
#ifdef MK_PERF_TRACE
                    if (_perf_rec) _perf_mark(&perf->mma_issue_ns);
#endif
                }
                cute::wait_barrier(smem.mma_barrier, mma_phase); mma_phase ^= 1;
#ifdef MK_PERF_TRACE
                if (_perf_rec) _perf_mark(&perf->mma_wait_ns);
#endif
            }
            if (pass == 0) copy(t2r, tDtAcc, tDrG);
            else           copy(t2r, tDtAcc, tDrU);
        }
#ifdef MK_PERF_TRACE
        if (_perf_rec) _perf_mark(&perf->loop_other_ns);
#endif
        cute::cluster_sync();   // ALL 800 threads — whole-CTA arrival required
#ifdef MK_PERF_TRACE
        if (_perf_rec) _perf_mark(&perf->cluster_sync_ns);
#endif
    }

    // if (threadIdx.x == 0) {
    //     printf("[UMMA-STEP7B] block=%d post-MMA-loop\n", (int)blockIdx.x);
    // }

    // SwiGLU epilogue: act = silu(gate)*up*route_w[m]. Only the 128 active
    // threads own TMEM fragments / output partitions; the other 672 skip it.
    if (active) {
        Tensor cD  = make_identity_tensor(make_shape(M, I));
        Tensor gcD = local_tile(cD, mma_tiler, mma_coord, Step<_1,_1, X>{});
        Tensor tCgcD = cta_mma.partition_C(gcD);
        Tensor tDcD  = thr_t2r.partition_D(tCgcD);
        Tensor tDrAct = make_tensor<ElemAB>(shape(tDgD));
        CUTE_UNROLL
        for (int i = 0; i < size(tDrG); ++i) {
            int m = get<0>(tDcD(i));
            float g = static_cast<float>(tDrG(i));
            float u = static_cast<float>(tDrU(i));
            float silu_g = g * (1.0f / (1.0f + ::expf(-g)));
            tDrAct(i) = static_cast<ElemAB>(silu_g * u * static_cast<float>(mR(m)));
        }
        copy(tDrAct, tDgD);

        umma_sync128();  // 128-thread barrier (not __syncthreads); see umma_sync128
    }
    // No dealloc here — TMEM stays allocated for reuse across iterations.
    // Dealloc is done once via umma_dealloc() when the persistent loop exits.
    // Final whole-CTA sync so the 672 non-active threads do not race ahead and
    // the compute_group_sync() that follows in compute_worker sees a converged block.
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->epilogue_ns);
#endif
    cute::cluster_sync();   // ALL 800 threads
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->cluster_sync_ns);
#endif
}

// ---- W_down TMA atom: shape [hidden, intermediate] = [N, K] K-major ----
// Same tile shape (256, 256, 64) as up-proj, just N=hidden and K=intermediate.
inline auto make_weight_down_tma_atom(const __nv_bfloat16* wd_e_ptr, int hidden, int intermediate) {
    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
    Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                              make_tile(typename TiledMMA_t::AtomThrID{}));
    auto sB_layout = make_sB_layout();
    // W_down for one expert: [hidden, intermediate], stride=(intermediate, 1) → K-major
    auto mW = make_tensor(make_gmem_ptr(reinterpret_cast<const ElemAB*>(wd_e_ptr)),
                          make_layout(make_shape(hidden, intermediate), make_stride(intermediate, Int<1>{})));
    return make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mW, sB_layout,
                                 mma_tiler, tiled_mma, cluster_layout_vmnk);
}

using WeightDownTmaAtom_t = decltype(make_weight_down_tma_atom((const __nv_bfloat16*)nullptr, 1, 1));

// ---- A-side TMA for down-proj: act workspace [M=256, intermediate] ----
inline auto make_act_tma_atom(const __nv_bfloat16* act_ptr, int M, int intermediate) {
    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
    Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                              make_tile(typename TiledMMA_t::AtomThrID{}));
    auto sA_layout = make_sA_layout();
    auto mA = make_tensor(make_gmem_ptr(reinterpret_cast<const ElemAB*>(act_ptr)),
                          make_layout(make_shape(M, intermediate), make_stride(intermediate, Int<1>{})));
    return make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA_layout,
                                 mma_tiler, tiled_mma, cluster_layout_vmnk);
}
using ActTmaAtom_t = decltype(make_act_tma_atom((const __nv_bfloat16*)nullptr, 1, 1));

// Holds per-expert W_down TMA atoms. Separate struct so S4.5 can be added independently.
struct ComputeDownTmaAtoms {
    int num_experts;
    int hidden;
    int intermediate;
    WeightDownTmaAtom_t wdown[kMaxLocalExperts];
};

inline void build_compute_down_tma_atoms(ComputeDownTmaAtoms& atoms,
                                          const __nv_bfloat16* W_down,
                                          int E, int hidden, int intermediate) {
    EP_HOST_ASSERT(E <= kMaxLocalExperts);
    atoms.num_experts = E;
    atoms.hidden = hidden;
    atoms.intermediate = intermediate;
    for (int e = 0; e < E; ++e) {
        const __nv_bfloat16* wd_e = W_down + (size_t)e * hidden * intermediate;
        atoms.wdown[e] = make_weight_down_tma_atom(wd_e, hidden, intermediate);
    }
}

// ---------------------------------------------------------------------------
// Device kernel: one 2-CTA cluster computes one [256, 256] output tile of the
// down-proj GEMM: D = act @ W_down^T. This is a pure GEMM with BF16 output
// (no SwiGLU). Follows the same pattern as umma_up_swiglu_tile but simpler:
// single-pass mainloop, direct TMEM→RMEM→cast→GMEM epilogue.
//
//   tma_A  : act workspace [M=256, intermediate] TMA atom (per-group)
//   tma_Wd : W_down[expert] [hidden, intermediate] TMA atom
//   d_tile : which 256-col tile of hidden (0..hidden/256-1)
//   out    : output buffer base [M, hidden]; writes [:, d_tile*256:+256]
//   M, hidden, intermediate : dims
// ---------------------------------------------------------------------------
template <class TmaA, class TmaB>
__device__ void umma_down_proj_tile(
    const TmaA& tma_A, const TmaB& tma_Wd,
    int d_tile, __nv_bfloat16* out,
    int M, int hidden, int intermediate, char* cluster_smem, bool& tmem_allocated
#ifdef MK_PERF_TRACE
    , UmmaPerf* perf = nullptr
#endif
    ) {

    const bool active = (threadIdx.x < 128);
#ifdef MK_PERF_TRACE
    const bool _perf_rec = (perf != nullptr && threadIdx.x == 0);
    int64_t _t_prev = _perf_rec ? globaltimer_ns() : 0;
    auto _perf_mark = [&](int64_t* slot) {
        if (!_perf_rec) return;
        int64_t now = globaltimer_ns();
        *slot += now - _t_prev;
        _t_prev = now;
    };
#endif

    TiledMMA_t tiled_mma = make_tiled_mma(MmaAtom_t{});
    auto mma_tiler = make_mma_tiler();
    auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
    Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                              make_tile(typename TiledMMA_t::AtomThrID{}));

    auto sA_layout = make_sA_layout();
    auto sB_layout = make_sB_layout();
    using SMEM = ClusterSharedStorage<ElemAB, decltype(sA_layout), decltype(sB_layout)>;
    SMEM& smem = *reinterpret_cast<SMEM*>(cluster_smem);

    // Output tensor: [M, hidden], N-major (row-major with stride=hidden)
    auto mD = make_tensor(make_gmem_ptr(reinterpret_cast<ElemAB*>(out)),
                          make_layout(make_shape(M, hidden), make_stride(hidden, Int<1>{})));

    // mma_coord: M-tile=0, N-tile=d_tile, K iterated
    auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                                     0, d_tile, _);
    auto mma_coord = select<1,2,3>(mma_coord_vmnk);

    Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});
    auto mma_v = get<0>(mma_coord_vmnk);
    ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
    Tensor tCgD = cta_mma.partition_C(gD);

    Tensor tCsA  = smem.tensor_sA();
    Tensor tCsBg = smem.tensor_sBg();  // reuse Bg slot for W_down

    using FragA_t = decltype(cta_mma.make_fragment_A(tCsA));
    using FragB_t = decltype(cta_mma.make_fragment_B(tCsBg));
    using FragC_t = decltype(cta_mma.make_fragment_C(tCgD));
    FragA_t tCrA;
    FragB_t tCrB;
    FragC_t tCtAcc;
    uint32_t elect_one_thr = 0;
    uint32_t elect_one_warp = 0;

    if (active) {
        tCrA   = cta_mma.make_fragment_A(tCsA);
        tCrB   = cta_mma.make_fragment_B(tCsBg);
        tCtAcc = cta_mma.make_fragment_C(tCgD);
        elect_one_thr  = cute::elect_one_sync();
        elect_one_warp = (threadIdx.x / 32 == 0);
    }

    using TmemAllocator = cute::TMEM::Allocator2Sm;
    TmemAllocator tmem_allocator{};
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->setup_ns);
#endif
    if (!tmem_allocated) {
        cute::cluster_sync();
        if (active && elect_one_warp) tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
        if (active) umma_sync128();
        tmem_allocated = true;
    } else {
        cute::cluster_sync();
    }
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->tmem_alloc_ns);
#endif
    if (active) tCtAcc.data() = smem.tmem_base_ptr;

    auto cta_in_cluster = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
    auto elect_one_cta = get<0>(cta_in_cluster) == Int<0>{};

    // TMA tensors: A=[M, intermediate], Wd=[hidden, intermediate]
    Tensor mA_tma  = tma_A.get_tma_tensor(make_shape(M, intermediate));
    Tensor mB_tma  = tma_Wd.get_tma_tensor(make_shape(hidden, intermediate));
    Tensor gA  = local_tile(mA_tma, mma_tiler, mma_coord, Step<_1, X,_1>{});
    Tensor gB  = local_tile(mB_tma, mma_tiler, mma_coord, Step< X,_1,_1>{});
    Tensor tCgA = cta_mma.partition_A(gA);
    Tensor tCgB = cta_mma.partition_B(gB);

    auto [tAgA, tAsA] = tma_partition(tma_A, get<2>(cta_in_cluster),
                                      make_layout(size<2>(cluster_layout_vmnk)),
                                      group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
    auto [tBgB, tBsB] = tma_partition(tma_Wd, get<1>(cta_in_cluster),
                                      make_layout(size<1>(cluster_layout_vmnk)),
                                      group_modes<0,3>(tCsBg), group_modes<0,3>(tCgB));

    uint16_t mcast_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster);
    uint16_t mcast_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster);
    uint16_t mcast_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster) |
                       create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster);
    int txbytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))
                + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));

    if (active && elect_one_warp && elect_one_thr) {
        int np = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
        cute::initialize_barrier(smem.mma_barrier, np);
        cute::initialize_barrier(smem.tma_barrier, 1);
    }
    int mma_phase = 0, tma_phase = 0;
    cute::cluster_sync();
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->prologue_ns);
#endif

    // Main GEMM loop
    int nKt = size<3>(tCgA);
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (active) {
        for (int k_tile = 0; k_tile < nKt; ++k_tile) {
            if (elect_one_warp && elect_one_thr) {
                if (elect_one_cta) cute::set_barrier_transaction_bytes(smem.tma_barrier, txbytes);
                copy(tma_A.with(smem.tma_barrier, mcast_a), tAgA(_,k_tile), tAsA);
                copy(tma_Wd.with(smem.tma_barrier, mcast_b), tBgB(_,k_tile), tBsB);
            }
#ifdef MK_PERF_TRACE
            if (_perf_rec) _perf_mark(&perf->loop_other_ns);
#endif
            if (elect_one_cta) {
                cute::wait_barrier(smem.tma_barrier, tma_phase); tma_phase ^= 1;
#ifdef MK_PERF_TRACE
                if (_perf_rec) _perf_mark(&perf->tma_wait_ns);
#endif
                if (elect_one_warp) {
                    for (int kb = 0; kb < size<2>(tCrA); ++kb) {
                        gemm(tiled_mma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
                        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                    }
                    cutlass::arch::umma_arrive_multicast_2x1SM(&smem.mma_barrier, mcast_c);
                }
#ifdef MK_PERF_TRACE
                if (_perf_rec) _perf_mark(&perf->mma_issue_ns);
#endif
            }
            cute::wait_barrier(smem.mma_barrier, mma_phase); mma_phase ^= 1;
#ifdef MK_PERF_TRACE
            if (_perf_rec) _perf_mark(&perf->mma_wait_ns);
#endif
        }
    }
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->loop_other_ns);
#endif

    // Epilogue: TMEM -> RMEM -> cast FP32->BF16 -> GMEM
    if (active) {
        TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
        ThrCopy thr_t2r = t2r.get_slice(threadIdx.x);
        Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
        Tensor tDgD   = thr_t2r.partition_D(tCgD);
        using AccT = typename decltype(tCtAcc)::value_type;
        Tensor tDrAcc = make_tensor<AccT>(shape(tDgD));
        copy(t2r, tDtAcc, tDrAcc);

        Tensor tDrD = make_tensor<ElemAB>(shape(tDgD));
        CUTE_UNROLL
        for (int i = 0; i < size(tDrAcc); ++i)
            tDrD(i) = static_cast<ElemAB>(tDrAcc(i));
        copy(tDrD, tDgD);

        umma_sync128();
    }
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->epilogue_ns);
#endif
    cute::cluster_sync();
#ifdef MK_PERF_TRACE
    if (_perf_rec) _perf_mark(&perf->cluster_sync_ns);
#endif
}

}  // namespace umma
}  // namespace megakernel
}  // namespace deep_ep
