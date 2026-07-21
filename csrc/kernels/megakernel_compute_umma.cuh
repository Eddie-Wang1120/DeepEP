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
// USAGE (in megakernel_forward_backward.cu, an nvcc TU):
//   - MegaKernelState holds a `ComputeTmaAtoms* compute_tma;` device pointer.
//   - Host: build_compute_tma_atoms(host_struct, W_gateup, E, I, d); upload.
//   - Device (compute_worker stage1, per 1-CTA/2-CTA cluster): call
//       umma_gateup_interleaved_persistent(...).

#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/transform.cuh>

// sm100_store_swiglu_from_gate lives only in the standalone reference (it is the
// SwiGLU-fused store epilogue). Pull it in directly so the migrated GEMM body
// can reuse the verified store path verbatim.
#include "../../compute_ref/sm100_bf16_gemm_dg_copy.cuh"

#include <cuda_bf16.h>

namespace deep_ep {
namespace megakernel {
namespace umma {

using namespace cute;

// Down-proj row-scatter epilogue used by the migrated megakernel path.
struct DownScatterParams {
    int4* combine_input_i4;
    int4* compute_output_slot_i4;
    const int* recv_token_idx;
    const unsigned char* is_single;
    int slot_base;
    int valid_rows;
    int hidden_int4;
};

// Named barrier over exactly the 128 threads (4 warps) that run the UMMA kernel
// inside an 800-thread megakernel block. Plain __syncthreads() would wait for all
// 800 threads and deadlock, since only thread_id<128 enter this code path.
//
// IMPORTANT: barrier ids 0/1/2 are ALREADY in use by the dispatch/combine roles
// of this megakernel (`barrier.sync 0/1/2`). Reusing id 1 (the old
// value) aliased the combine forwarder's `barrier.sync 1`, corrupting the
// cluster-scoped tcgen05 alloc handshake and tripping
//   __cuda_sm10x_tcgen05_guardrail_trap_phase_invalid_during_alloc.
// Use id 8 (unused by any role) so the 128 UMMA threads sync in isolation.
CUTE_DEVICE void umma_sync128() {
    asm volatile("barrier.sync 8, 128;" ::: "memory");
}

using ElemAB  = cutlass::bfloat16_t;
using ElemAcc = float;

// DeepGEMM reference config from compute_ref/sm100_bf16_gemm_dg_copy.cuh:
// BLOCK_M=128, BLOCK_N=128, BLOCK_K=64. A compute group covers the full
// M=256 batch by having each 2-CTA cluster iterate over multiple 128x128 tiles.
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
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

// ===========================================================================
// DeepGEMM-verbatim port (compute_ref/umma_swiglu_ws_dg.cu + sm100_bf16_gemm_dg_copy.cuh)
//
// We drop the CuTe `make_tma_atom_*` atoms in favor of raw CUtensorMap TMA
// descriptors built exactly like umma_swiglu_ws_dg.cu's make_a_desc/make_b_desc/
// make_cd_desc, and DeepGEMM's own deep_gemm::tma::copy + make_umma_desc::fma
// device path. This keeps the verified single-CTA (kNumMulticast=1) numeric
// path; the megakernel block still launches as a 2-CTA cluster but the GEMM
// helper runs the single-CTA DeepGEMM kernel body on the leader-CTA threads.
// ===========================================================================

// DeepGEMM config. kDgRunMulticast selects the 1-CTA vs 2-CTA path; the TMA
// descriptor smem box (LOAD_BLOCK_M/N) must match the device-side LOAD_BLOCK
// used in dg_gemm_tile, which divides by the multicast factor.
static constexpr uint32_t kDgBlockM     = 128;
#ifndef MK_DG_BLOCK_N
#define MK_DG_BLOCK_N 128
#endif
#ifndef MK_DG_BLOCK_K
#define MK_DG_BLOCK_K 64
#endif
static constexpr uint32_t kDgBlockN     = MK_DG_BLOCK_N;
static constexpr uint32_t kDgBlockK     = MK_DG_BLOCK_K;
static constexpr uint32_t kDgSwizzleA   = 128;
static constexpr uint32_t kDgSwizzleB   = 128;
static constexpr uint32_t kDgSwizzleCD  = 128;
static constexpr bool     kDgMcastOnA   = false;
static constexpr uint32_t kDgKAlign     = 128;
// MK_COMPUTE_KERNEL: 1 selects 1-CTA UMMA; 2 selects 2-CTA UMMA.
// Mode 0 (WMMA) still compiles this header, so keep the UMMA constants valid.
#ifndef MK_COMPUTE_KERNEL
#define MK_COMPUTE_KERNEL 2
#endif
static_assert(MK_COMPUTE_KERNEL >= 0 && MK_COMPUTE_KERNEL <= 2,
              "MK_COMPUTE_KERNEL must be 0 (WMMA), 1 (1-CTA UMMA), or 2 (2-CTA UMMA)");
static constexpr uint32_t kDgRunMulticast = (MK_COMPUTE_KERNEL == 2 ? 2 : 1);
static constexpr uint32_t kDgLoadBlockM = kDgBlockM / (kDgMcastOnA ? kDgRunMulticast : 1);
static constexpr uint32_t kDgLoadBlockN = kDgBlockN / (kDgMcastOnA ? 1 : kDgRunMulticast);
static constexpr uint32_t kDgStoreBlockM = (kDgBlockM < 128 ? kDgBlockM : 128);              // 128
static constexpr uint32_t kDgStoreBlockN = kDgSwizzleCD / sizeof(cutlass::bfloat16_t);       // 64

// Warp-specialization constants (mirror sm100_bf16_gemm.cuh hardcoded values).
// kNumStages: pipeline depth (matching DeepGEMM default kNumStages_=4 before merge).
// kNumEpilogueStages: double-buffered TMEM accumulator pipeline depth = 2.
// kNumTMAStoreStages: TMA store pipeline depth = 2.
// kNumNonEpilogueThreads: TMA + MMA + alloc warps = 4 warps * 32 = 128.
// kNumEpilogueThreads: epilogue store warps = 4 warps * 32 = 128 (= STORE_BLOCK_M).
// kNumTmemCols: get_num_aligned_tmem_cols<kNumEpilogueStages * UMMA_N>
//   = get_num_aligned_tmem_cols<2 * 128> = get_num_aligned_tmem_cols<256> = 256.
#ifndef MK_DG_NUM_STAGES
#define MK_DG_NUM_STAGES 4
#endif
#ifndef MK_DG_EPI_STAGES
#define MK_DG_EPI_STAGES 2
#endif
#ifndef MK_DG_TMA_STORE_STAGES
#define MK_DG_TMA_STORE_STAGES 2
#endif
static constexpr int kDgWsNumStages         = MK_DG_NUM_STAGES;
static constexpr int kDgWsNumEpilogueStages = MK_DG_EPI_STAGES;
static constexpr int kDgWsNumTmaStoreStages = MK_DG_TMA_STORE_STAGES;
static constexpr int kDgWsNonEpiThreads     = 128;   // 4 warps: warp0=TMA, warp1=MMA, warp2=alloc, warp3=idle
static constexpr int kDgWsEpiThreads        = 128;   // 4 warps: warps[4..7] = epilogue
// align_tmem(kNumEpilogueStages * BLOCK_N); depends on BLOCK_N so it tracks tuning.
static constexpr int kDgWsNumTmemCols       = (kDgWsNumEpilogueStages * kDgBlockN + 31) / 32 * 32;

// Build a 2D TMA descriptor exactly like umma_swiglu_ws_dg.cu::make_tma_2d.
inline CUtensorMap dg_make_tma_2d(const void* ptr, CUtensorMapDataType dtype,
                                  int gmem_inner, int gmem_outer,
                                  int smem_inner, int smem_outer,
                                  int gmem_outer_stride_elems, int elem_size,
                                  int swizzle_bytes) {
    CUtensorMap tm;
    int si = smem_inner;
    if (swizzle_bytes != 0) si = swizzle_bytes / elem_size;
    const cuuint64_t gdims[2] = { (cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer };
    const cuuint32_t sdims[2] = { (cuuint32_t)si, (cuuint32_t)smem_outer };
    const cuuint64_t gstr[1]  = { (cuuint64_t)gmem_outer_stride_elems * elem_size };
    const cuuint32_t estr[2]  = { 1, 1 };
    CUtensorMapSwizzle sw =
        swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult r = cuTensorMapEncodeTiled(
        &tm, dtype, 2, (void*)ptr, gdims, gstr, sdims, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    EP_HOST_ASSERT(r == CUDA_SUCCESS);
    return tm;
}

// A: [M,K] K-major. inner=K outer=M. smem inner=BLOCK_K outer=LOAD_BLOCK_M.
inline CUtensorMap dg_make_a_desc(const __nv_bfloat16* a, int M, int K) {
    return dg_make_tma_2d(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, M,
                          kDgBlockK, kDgLoadBlockM, K, sizeof(__nv_bfloat16), kDgSwizzleA);
}
// B: [N,K] K-major. inner=K outer=N. smem inner=BLOCK_K outer=LOAD_BLOCK_N.
inline CUtensorMap dg_make_b_desc(const __nv_bfloat16* b, int N, int K) {
    return dg_make_tma_2d(b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, N,
                          kDgBlockK, kDgLoadBlockN, K, sizeof(__nv_bfloat16), kDgSwizzleB);
}
// B: [K,N] MN-major. inner=N outer=K. smem inner=LOAD_BLOCK_N outer=BLOCK_K.
inline CUtensorMap dg_make_b_mn_desc(const __nv_bfloat16* b, int N, int K) {
    return dg_make_tma_2d(b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, N, K,
                          kDgLoadBlockN, kDgBlockK, N, sizeof(__nv_bfloat16), kDgSwizzleB);
}
// CD: [M,N] row-major BF16. inner=N outer=M, smem inner=STORE_BLOCK_N outer=STORE_BLOCK_M.
inline CUtensorMap dg_make_cd_desc(const __nv_bfloat16* d, int M, int N) {
    return dg_make_tma_2d(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, N, M,
                          kDgStoreBlockN, kDgStoreBlockM, N, sizeof(__nv_bfloat16), kDgSwizzleCD);
}

static constexpr int kMaxLocalExperts = 64;

// ---------------------------------------------------------------------------
// MegaTileScheduler — megakernel-adapted persistent tile scheduler.
//
// Functionally identical to DeepGEMM sched::Scheduler<Normal>, but:
//   - the "block index space" is the cluster index inside a compute group
//     (cluster_idx in [0, num_clusters)) instead of grid-global blockIdx.x;
//   - num_clusters (=16) replaces kNumSMs.
// It reproduces DeepGEMM's `next_block_idx = (++current_iter)*kNumSMs + blockIdx.x`
// persistent block assignment and the L2-swizzled (m,n) mapping
// (get_swizzled_block_idx). Each warp role (TMA/MMA/epilogue) constructs its own
// instance with the SAME (M, N, cluster_idx, num_clusters): because the sequence
// is a deterministic integer progression, all three roles enumerate the IDENTICAL
// tile order without any cross-warp communication; data hand-off is via mbarriers.
//
// BLOCK_M/BLOCK_N are compile-time (kDgBlockM/kDgBlockN). kIsMulticastOnA=false
// (multicast on B/N) → swizzle groups on M, exactly mirroring DeepGEMM.
struct MegaTileScheduler {
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;
    uint32_t num_blocks;
    uint32_t num_blocks_in_group;     // set by get_swizzled_block_idx (unused on SM100)
    int cluster_idx;                  // 0..num_clusters-1 (= cluster_in_group)
    int num_clusters;                 // 16 (= COMPUTE_GROUP_SIZE/2)
    int current_iter;                 // mirrors DeepGEMM scheduler.current_iter

    static constexpr uint32_t BLOCK_M = kDgBlockM;
    static constexpr uint32_t BLOCK_N = kDgBlockN;
    static constexpr bool kIsMulticastOnA = kDgMcastOnA;          // false
    // DeepGEMM picks 8 or 16 by usage; for our M-small/N-large shape mirror its
    // default selection at compile time. group-on-M (kIsMulticastOnA=false).
    static constexpr uint32_t kNum1DBlocksPerGroup = 16;

    __device__ MegaTileScheduler(uint32_t shape_m, uint32_t shape_n,
                                 int cluster_idx_, int num_clusters_)
        : cluster_idx(cluster_idx_), num_clusters(num_clusters_), current_iter(-1) {
        num_m_blocks = (shape_m + BLOCK_M - 1) / BLOCK_M;
        num_n_blocks = (shape_n + BLOCK_N - 1) / BLOCK_N;
        num_blocks = num_m_blocks * num_n_blocks;
    }

    // Verbatim DeepGEMM get_swizzled_block_idx (SM100 path, no SM90 multicast fix).
    __device__ void get_swizzled_block_idx(uint32_t block_idx,
                                           uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // Swizzle for better L2 usage.
        const auto primary_num_blocks   = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    // Returns false when this cluster has no more tiles. Mirrors DeepGEMM Normal path.
    __device__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t next_block_idx =
            static_cast<uint32_t>(++current_iter) * static_cast<uint32_t>(num_clusters)
            + static_cast<uint32_t>(cluster_idx);
        if (next_block_idx >= num_blocks)
            return false;
        get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        return true;
    }
};

// Per-expert raw TMA descriptors for interleaved gate/up weights.
// wgateup is [2I,d] = [g0,u0,g1,u1,...], for the single-GEMM gate/up +
// in-TMEM SwiGLU epilogue.
struct ComputeTmaAtoms {
    int num_experts;
    int I;
    int d;
    CUtensorMap wgate[kMaxLocalExperts];      // legacy, unused by interleaved path
    CUtensorMap wup[kMaxLocalExperts];        // legacy, unused by interleaved path
    CUtensorMap wgateup[kMaxLocalExperts];    // [2I, d] K-major interleaved(g0,u0,...)
};

inline void build_compute_tma_atoms(ComputeTmaAtoms& atoms,
                                     const __nv_bfloat16* W_gateup,
                                     int E, int I, int d) {
    EP_HOST_ASSERT(E <= kMaxLocalExperts);
    atoms.num_experts = E;
    atoms.I = I;
    atoms.d = d;
    for (int e = 0; e < E; ++e) {
        const __nv_bfloat16* wgu_e = W_gateup + (size_t)e * (2 * I) * d;
        atoms.wgateup[e] = dg_make_b_desc(wgu_e, 2 * I, d);
    }
}

// Per-group A(input_buf [M,d]) and workspace descriptors.
// act_cd is the direct output of the interleaved SwiGLU epilogue; gu_cd is kept
// for legacy tile helpers and reserved scratch layout compatibility.
struct InputTmaAtom_t {
    CUtensorMap a;        // input_buf [M, d]   K-major A
    CUtensorMap gate_cd;  // gate_buf  [M, I]   row-major CD
    CUtensorMap act_cd;   // up_buf    [M, I]   row-major CD
    CUtensorMap act_a;    // up_buf    [M, I]   K-major A (down-proj A operand)
    CUtensorMap down_cd;  // down_buf  [M, hidden] row-major CD (down-proj output)
    CUtensorMap gu_cd;    // gate_buf  [M, 2I]  row-major CD (reserved GU scratch / legacy helpers)
    CUtensorMap gu_a;     // gate_buf  [M, 2I]  K-major A (backward grad_x A operand)
};

inline InputTmaAtom_t make_input_tma_atom(const __nv_bfloat16* input_buf_ptr, int M, int d) {
    // Kept signature for the host call site; gate/act descriptors are filled by
    // make_input_group_atoms below (this builds only A so callers compile).
    InputTmaAtom_t out{};
    out.a = dg_make_a_desc(input_buf_ptr, M, d);
    return out;
}

// Full per-group builder: A from input_buf, gate/act CD into the workspace.
// Also builds the down-proj operands: act_a (up_buf as K-major A) and
// down_cd (down_buf as row-major CD).
inline InputTmaAtom_t make_input_group_atoms(const __nv_bfloat16* input_buf_ptr,
                                             const __nv_bfloat16* gate_buf_ptr,
                                             const __nv_bfloat16* act_buf_ptr,
                                             const __nv_bfloat16* down_buf_ptr,
                                             int M, int d, int I, int hidden) {
    InputTmaAtom_t out{};
    out.a       = dg_make_a_desc(input_buf_ptr, M, d);
    out.gate_cd = dg_make_cd_desc(gate_buf_ptr, M, I);
    out.act_cd  = dg_make_cd_desc(act_buf_ptr, M, I);
    // down-proj: A = act_buf [M, I] (K=I), CD = down_buf [M, hidden].
    out.act_a   = dg_make_a_desc(act_buf_ptr, M, I);
    out.down_cd = dg_make_cd_desc(down_buf_ptr, M, hidden);
    // Reserved GU scratch descriptor for legacy helpers; the interleaved path stores
    // act directly through act_cd and does not write this region.
    out.gu_cd   = dg_make_cd_desc(gate_buf_ptr, M, 2 * I);
    out.gu_a    = dg_make_a_desc(gate_buf_ptr, M, 2 * I);
    return out;
}

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
// ---------------------------------------------------------------------------
// DeepGEMM-verbatim single-tile GEMM (+ optional SwiGLU epilogue).
//
// This is the warp-specialized DeepGEMM kernel body from
// compute_ref/sm100_bf16_gemm_dg_copy.cuh, inlined as a device routine and
// driven by a fixed (m_block, n_block) tile instead of DeepGEMM's grid
// scheduler. Runs the single-CTA (kNumMulticast=1) numeric path proven in
// compute_ref/umma_swiglu_ws_dg.cu.
//
// Warp roles within the first kDgWsNonEpiThreads+kDgWsEpiThreads threads:
//   warp 0        : TMA load
//   warp 1        : MMA issue (make_umma_desc + SM100_MMA_F16BF16_SS::fma)
//   warp 2        : TMEM alloc/dealloc
//   warps [4,8)   : epilogue store (sm100_store_cd / swiglu_from_gate)
// The remaining threads of the 800-thread megakernel block sit idle (no
// cluster_sync inside — single-CTA path uses __syncthreads-free barriers).
//
//   desc_a   : input_buf [M,d]   K-major raw CUtensorMap
//   desc_b   : W [I,d] = [N,K]   K-major raw CUtensorMap (gate or up)
//   desc_cd  : output [M,I]      row-major BF16 CUtensorMap
//   m_block/n_block : tile coordinates (M/BLOCK_M, I/BLOCK_N)
//   gate_ptr/route_ptr/stride_n : only used when fuse_swiglu (up pass)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DgSmemLayout — shared SMEM/barrier layout for the DeepGEMM persistent GEMM.
// Used by dg_init_barriers_tmem / dg_dealloc_tmem / dg_gemm_persistent so the
// init-once, run-many, free-once split keeps the SAME smem offsets everywhere.
// ---------------------------------------------------------------------------
template <uint32_t kNumMulticast>
struct DgSmemLayout {
    static constexpr uint32_t BLOCK_M = kDgBlockM, BLOCK_N = kDgBlockN, BLOCK_K = kDgBlockK;
    static constexpr bool kIsMulticastOnA = kDgMcastOnA;
    static constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    static constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    static constexpr uint32_t STORE_BLOCK_M = kDgStoreBlockM, STORE_BLOCK_N = kDgStoreBlockN;
    static constexpr uint32_t kNumStages = kDgWsNumStages;
    static constexpr uint32_t kNumEpilogueStages = kDgWsNumEpilogueStages;
    static constexpr uint32_t kNumTMAStoreStages = kDgWsNumTmaStoreStages;
    static constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t UMMA_N = BLOCK_N;
    static constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    static constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    __device__ static Barrier* barrier_start(char* cluster_smem) {
        uint8_t* sb = reinterpret_cast<uint8_t*>(cluster_smem);
        return reinterpret_cast<Barrier*>(sb + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    }
    __device__ static uint32_t* tmem_ptr_in_smem(char* cluster_smem) {
        return reinterpret_cast<uint32_t*>(barrier_start(cluster_smem) + kNumStages * 3 + kNumEpilogueStages * 2 + 1);
    }
};

// dg_init_barriers_tmem — init mbarriers + allocate TMEM ONCE per task (4a).
// Called by all GEMM warps (0..7) before the persistent loops.
template <uint32_t kNumMulticast>
__device__ void dg_init_barriers_tmem(char* cluster_smem) {
    using namespace deep_gemm;
    using L = DgSmemLayout<kNumMulticast>;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<L::kNumAccumTmemCols>();
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    auto bar = L::barrier_start(cluster_smem);
    auto full_b  = utils::PatternVisitor([=](const uint32_t& i) { return bar + i; });
    auto empty_b = utils::PatternVisitor([=](const uint32_t& i) { return bar + (L::kNumStages + i); });
    auto tf_b    = utils::PatternVisitor([=](const uint32_t& i) { return bar + (L::kNumStages * 2 + i); });
    auto te_b    = utils::PatternVisitor([=](const uint32_t& i) { return bar + (L::kNumStages * 2 + L::kNumEpilogueStages + i); });

    if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < L::kNumStages; ++ i) { full_b[i]->init(kNumMulticast); empty_b[i]->init(1); }
        #pragma unroll
        for (uint32_t i = 0; i < L::kNumEpilogueStages; ++ i) {
            tf_b[i]->init(1);
            te_b[i]->init(kNumMulticast * L::kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, L::tmem_ptr_in_smem(cluster_smem));
    }
    if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
    else __syncthreads();
}

// dg_reinit_barriers — re-init mbarriers WITHOUT re-allocating TMEM.
// Use when TMEM is persistent across multiple dg_gemm_persistent calls
// (e.g., gate/up then down within the same task). Avoids the TMEM alloc/free
// overhead while resetting barrier phase state for the next GEMM pass.
template <uint32_t kNumMulticast>
__device__ void dg_reinit_barriers(char* cluster_smem) {
    using namespace deep_gemm;
    using L = DgSmemLayout<kNumMulticast>;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    auto bar = L::barrier_start(cluster_smem);
    auto full_b  = utils::PatternVisitor([=](const uint32_t& i) { return bar + i; });
    auto empty_b = utils::PatternVisitor([=](const uint32_t& i) { return bar + (L::kNumStages + i); });
    auto tf_b    = utils::PatternVisitor([=](const uint32_t& i) { return bar + (L::kNumStages * 2 + i); });
    auto te_b    = utils::PatternVisitor([=](const uint32_t& i) { return bar + (L::kNumStages * 2 + L::kNumEpilogueStages + i); });

    if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < L::kNumStages; ++ i) { full_b[i]->init(kNumMulticast); empty_b[i]->init(1); }
        #pragma unroll
        for (uint32_t i = 0; i < L::kNumEpilogueStages; ++ i) {
            tf_b[i]->init(1);
            te_b[i]->init(kNumMulticast * L::kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
    else __syncthreads();
}

// dg_dealloc_tmem — free TMEM ONCE per task (4a).
template <uint32_t kNumMulticast>
__device__ void dg_dealloc_tmem(char* cluster_smem) {
    using namespace deep_gemm;
    using L = DgSmemLayout<kNumMulticast>;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<L::kNumAccumTmemCols>();
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
    else __syncthreads();
    if (warp_idx == 0)
        Allocator().free(*L::tmem_ptr_in_smem(cluster_smem), kNumTmemCols);
    if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
    else __syncthreads();
}

template <bool kFuseSwiGLU, uint32_t kNumMulticast = 1, bool kFuseSwiGLUInterleaved = false,
          cute::UMMA::Major kMajorB = cute::UMMA::Major::K>
__device__ void dg_gemm_tile(
    const CUtensorMap* desc_a, const CUtensorMap* desc_b, const CUtensorMap* desc_cd,
    int m_block, int n_block,
    int shape_m, int shape_n, int shape_k,
    char* cluster_smem, bool& tmem_allocated, uint32_t& accum_iter,
    const cutlass::bfloat16_t* gate_ptr, const float* route_ptr, uint32_t stride_n,
    cutlass::bfloat16_t* preact_ptr = nullptr,
    const int* preact_recv_idx = nullptr,
    const int* preact_topk_idx = nullptr,
    uint32_t num_topk = 0,
    uint32_t preact_stride = 0,
    uint32_t valid_rows = 0xffffffffu) {

    using namespace deep_gemm;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t BLOCK_M = kDgBlockM, BLOCK_N = kDgBlockN, BLOCK_K = kDgBlockK;
    constexpr bool kIsMulticastOnA = kDgMcastOnA;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = kDgStoreBlockM, STORE_BLOCK_N = kDgStoreBlockN;
    constexpr uint32_t kNumStages = kDgWsNumStages;
    constexpr uint32_t kNumEpilogueStages = kDgWsNumEpilogueStages;
    constexpr uint32_t kNumTMAStoreStages = kDgWsNumTmaStoreStages;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;   // 128 for BLOCK_M=128

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    // Mirror DeepGEMM: kNumAccumTmemCols = kNumEpilogueStages * UMMA_N, then align.
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    const bool is_leader_cta = (kNumMulticast == 1) ? true : (cute::block_rank_in_cluster() == 0);

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Hang-debug tracing: trace BOTH CTAs of the first compute cluster (blk 50 leader,
    // blk 51 follower). cluster_sync needs all threads of both CTAs to arrive; if only
    // one CTA reaches the closing sync the other CTA's last line reveals where it stuck.
    // Prints rank-in-cluster so leader/follower divergence is visible.
    // Disabled by default; define MK_DG_HANG_DEBUG (e.g. via -DMK_DG_HANG_DEBUG) to enable.
    constexpr int MK_DG_HANG_BLK0 = 50;
    constexpr int MK_DG_HANG_BLK1 = 51;
    (void)MK_DG_HANG_BLK0; (void)MK_DG_HANG_BLK1;
#ifdef MK_DG_STANDALONE_DBG
    // Standalone microkernel debug: single-block launch, trace block 0 only,
    // lane 0 of each warp, every phase. Used by compute_ref/mega_vs_wmma_gate.cu.
    #define MK_DG_DBG(fmt, ...) do { \
        if (lane_idx == 0 && blockIdx.x == 0) \
            printf("[MK-DG][blk=%d w=%d sw=%d ms=%u mc=%u] " fmt "\n", \
                   (int)blockIdx.x, (int)warp_idx, (int)kFuseSwiGLU, \
                   accum_iter, (unsigned)kNumMulticast, ##__VA_ARGS__); \
    } while (0)
#else
    #define MK_DG_DBG(fmt, ...) do {} while (0)
#endif

    MK_DG_DBG("enter dg_gemm_tile m_block=%d n_block=%d shape_k=%d tmem_alloc=%d",
              m_block, n_block, shape_k, (int)tmem_allocated);

    // cluster-scoped sync helper: 2-CTA path needs cluster barriers, 1-CTA uses block.
    auto block_or_cluster_sync = [&]() {
        if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
        else __syncthreads();
    };

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(desc_a);
        cute::prefetch_tma_descriptor(desc_b);
        cute::prefetch_tma_descriptor(desc_cd);
    }

    // ---- SMEM partition over the shared cluster_smem buffer (mirror DeepGEMM) ----
    uint8_t* smem_buffer = reinterpret_cast<uint8_t*>(cluster_smem);
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers= utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    // tensor_core_full_barrier occupies the slot at [kNumStages*3 + kNumEpilogueStages*2],
    // exactly as in DeepGEMM sm100_bf16_gemm.cuh — we do NOT use it (kTensorCoreUtilControl=100),
    // but the slot must exist so tmem_ptr_in_smem lands at +1 past it (matching DeepGEMM layout).
    auto tensor_core_full_barrier = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2;
    (void)tensor_core_full_barrier;   // unused; kept for smem layout alignment with DeepGEMM
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + 1);

    // ---- Barrier + TMEM init (once per persistent loop) ----
    if (!tmem_allocated) {
        // 2-CTA TMEM allocation requires a cluster sync before allocate (DeepGEMM).
        MK_DG_DBG("init: pre-alloc cluster_sync BEGIN");
        if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
        MK_DG_DBG("init: pre-alloc cluster_sync END");
        if (warp_idx == 1 and cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                full_barriers[i]->init(kNumMulticast);   // arrive only at leader CTA
                empty_barriers[i]->init(1);              // arrive at all CTAs
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                tmem_full_barriers[i]->init(1);
                tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
            }
            cutlass::arch::fence_barrier_init();
            MK_DG_DBG("init: barriers initialized");
        } else if (warp_idx == 2) {
            Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
            MK_DG_DBG("init: tmem allocated cols=%u", kNumTmemCols);
        }
        MK_DG_DBG("init: post-init sync BEGIN");
        block_or_cluster_sync();
        MK_DG_DBG("init: post-init sync END");
        tmem_allocated = true;
    } else {
        MK_DG_DBG("reuse: entry sync BEGIN");
        block_or_cluster_sync();
        MK_DG_DBG("reuse: entry sync END");
    }

    // ---- accumulator (epilogue) pipeline stage/phase, persisted across calls ----
    // This is the key fix: gate/up are two consecutive DeepGEMM "iterations" that
    // share the TMEM accumulator pipeline. Without persisting accum_iter, the
    // second (up) pass would wait on a stale tmem_empty phase. We mirror
    // DeepGEMM's `scheduler.current_iter % kNumEpilogueStages`.
    const uint32_t accum_stage_idx = accum_iter % kNumEpilogueStages;
    const uint32_t accum_phase_idx = (accum_iter / kNumEpilogueStages) & 1;

    // Pipeline phases (mainloop)
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= (stage_idx == 0);
    };

    uint32_t m_idx0 = m_block * BLOCK_M;
    uint32_t n_idx0 = n_block * BLOCK_N;
    const auto num_total_k_blocks = math::ceil_div<uint32_t>(shape_k, BLOCK_K);

    // 2-CTA load offsets (split A on M or B on N across the cluster's 2 CTAs).
    const uint32_t cta_rank = (kNumMulticast > 1) ? cute::block_rank_in_cluster() : 0;
    const uint32_t load_m_idx = m_idx0 + (kIsMulticastOnA ? cta_rank * LOAD_BLOCK_M : 0);
    const uint32_t load_n_idx = n_idx0 + (kIsMulticastOnA ? 0 : cta_rank * LOAD_BLOCK_N);

    if (warp_idx == 0 and cute::elect_one_sync()) {
        // ---- TMA load warp ----
        MK_DG_DBG("TMA: loop start num_k=%u load_m_idx=%u load_n_idx=%u LBM=%u LBN=%u BK=%u",
                  num_total_k_blocks, load_m_idx, load_n_idx,
                  (unsigned)LOAD_BLOCK_M, (unsigned)LOAD_BLOCK_N, (unsigned)BLOCK_K);
        MK_DG_DBG("TMA: smem_a0=%p smem_b0=%p bar0=%p SMEM_CD=%u A_per=%u B_per=%u",
                  (void*)smem_a[0], (void*)smem_b[0], (void*)full_barriers[0],
                  (unsigned)SMEM_CD_SIZE, (unsigned)SMEM_A_SIZE_PER_STAGE, (unsigned)SMEM_B_SIZE_PER_STAGE);
        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
            MK_DG_DBG("TMA: k=%u wait empty stage=%u phase=%u", k_block_idx, stage_idx, phase ^ 1);
            empty_barriers[stage_idx]->wait(phase ^ 1);
            const uint32_t k_idx = k_block_idx * BLOCK_K;
            MK_DG_DBG("TMA: k=%u before copy A k_idx=%u m_idx=%u smem=%p", k_block_idx, k_idx, load_m_idx, (void*)smem_a[stage_idx]);
            tma::copy<BLOCK_K, LOAD_BLOCK_M, kDgSwizzleA, cutlass::bfloat16_t>(
                desc_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, load_m_idx, kNumMulticast);
            MK_DG_DBG("TMA: k=%u before copy B k_idx=%u n_idx=%u smem=%p", k_block_idx, k_idx, load_n_idx, (void*)smem_b[stage_idx]);
            if constexpr (kMajorB == cute::UMMA::Major::K) {
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kDgSwizzleB, cutlass::bfloat16_t>(
                    desc_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, load_n_idx, kNumMulticast);
            } else {
                tma::copy<LOAD_BLOCK_N, BLOCK_K, kDgSwizzleB, cutlass::bfloat16_t>(
                    desc_b, full_barriers[stage_idx], smem_b[stage_idx], load_n_idx, k_idx, kNumMulticast);
            }
            MK_DG_DBG("TMA: k=%u after copy AB, arrive_tx", k_block_idx);
            constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
            if (is_leader_cta)
                full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
            else
                full_barriers[stage_idx]->arrive(0u);
            MK_DG_DBG("TMA: k=%u arrived", k_block_idx);
        }
        MK_DG_DBG("TMA: loop done");
    } else if (warp_idx == 1 and is_leader_cta) {
        // ---- MMA issue warp (leader CTA only) ----
        // kDoMergeStages: mirror DeepGEMM's stage-merge optimization (reduces umma_arrive overhead).
        // Condition: kNumStages >= 8 and Normal GEMM and K-major A and B. For our
        // kNumStages=4 this is always false, so kNumStagesPerMerge=1 and BLOCK_ATOM_K=BLOCK_K.
        // We keep the full structure so future kNumStages_ >= 8 configs work correctly.
        constexpr bool kDoMergeStages = (kNumStages >= 8 && kMajorB == cute::UMMA::Major::K);
        constexpr uint32_t kNumMinStages = 8;
        constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages / kNumMinStages : 1;
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, kMajorB>();
        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_ATOM_K, kDgSwizzleA>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kDgSwizzleB>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
        ptx::tcgen05_after_thread_sync();
        MK_DG_DBG("MMA: tmem_empty acquired stage=%u phase=%u", accum_stage_idx, accum_phase_idx ^ 1);

        // UMMA / empty-barrier arrive alias: single- vs multi-cast.
        auto umma_arrive = [](const uint64_t* barrier) {
            if constexpr (kNumMulticast == 1) {
                cutlass::arch::umma_arrive(barrier);
            } else {
                constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
            }
        };
        // Mirrors DeepGEMM's empty_barrier_arrive lambda: arrives on empty_barriers[stage_idx]
        // and optionally on tmem_full_barriers[accum_stage_idx] at the last K block.
        auto empty_barrier_arrive = [&](const bool do_tmem_full_arrive) {
            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
            if (do_tmem_full_arrive)
                umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
            __syncwarp();
        };

        const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
            // MK_DG_DBG("MMA: wait full stage=%u phase=%u k=%u", stage_idx, phase, k_block_idx);
            full_barriers[stage_idx]->wait(phase);
            ptx::tcgen05_after_thread_sync();

            const auto a_base = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
            const auto b_base = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
            if (cute::elect_one_sync()) {
                using mma_t = cute::conditional_t<kNumMulticast == 1,
                                  ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                auto issue_umma = [&]<uint32_t kUMMAKIdx>() {
                    // Align with DeepGEMM: kAtomKIdx selects which merged stage atom,
                    // kInnerKIdx is the offset within that atom. When kNumStagesPerMerge=1
                    // (our case): kAtomKIdx=0, kInnerKIdx=kUMMAKIdx*UMMA_K — same as before
                    // but now structurally correct for future merge cases too.
                    constexpr uint32_t kAtomKIdx  = kUMMAKIdx * UMMA_K / BLOCK_ATOM_K;
                    constexpr uint32_t kInnerKIdx = kUMMAKIdx * UMMA_K % BLOCK_ATOM_K;
                    a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kDgSwizzleA, cutlass::bfloat16_t>(
                                    a_base, kAtomKIdx * LOAD_BLOCK_M * BLOCK_ATOM_K, kInnerKIdx);
                    b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kDgSwizzleB, cutlass::bfloat16_t>(
                                    b_base, kAtomKIdx * LOAD_BLOCK_N * BLOCK_ATOM_K, kInnerKIdx);
                    mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                               kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
                };
                utils::for_each_static_until<BLOCK_K / UMMA_K>(
                    std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), issue_umma);
            }
            __syncwarp();
            // Use empty_barrier_arrive lambda (mirrors DeepGEMM) instead of inline arrives.
            empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
        }
        MK_DG_DBG("MMA: loop done");
        // NOTE: DeepGEMM has a final `tmem_empty_barriers[...]->wait(accum_phase)` drain
        // here, but that relies on the persistent-kernel TERMINAL state (all blocks done,
        // epilogue has arrived on every tmem_empty stage). In our PER-TILE call model that
        // drain dead-locks: this tile's epilogue only arrives tmem_empty ONCE (to release
        // the MMA warp's next-iter `wait(phase ^ 1)`), so waiting on the positive phase here
        // never completes — the MMA warp stalls before the closing cluster_sync, hanging the
        // whole 2-CTA cluster. The accum_iter handoff across tiles already keeps the phase
        // ring correct, so no per-tile drain is needed.
    } else if (warp_idx >= kDgWsNonEpiThreads / 32 and warp_idx < (kDgWsNonEpiThreads + kDgWsEpiThreads) / 32) {
        // ---- Epilogue store warps ----
        const auto epilogue_warp_idx = warp_idx - (kDgWsNonEpiThreads / 32);
        uint32_t tma_stage_idx = 0;
        const cute::TmaDescriptor& tensor_map_cd = *desc_cd;

        MK_DG_DBG("EPI[%d]: wait tmem_full stage=%u phase=%u", (int)epilogue_warp_idx, accum_stage_idx, accum_phase_idx);
        tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
        ptx::tcgen05_after_thread_sync();
        const auto tmem_base_addr = accum_stage_idx * UMMA_N;
        MK_DG_DBG("EPI[%d]: store begin fuse=%d interleaved=%d", (int)epilogue_warp_idx,
                  (int)kFuseSwiGLU, (int)kFuseSwiGLUInterleaved);

        if constexpr (kFuseSwiGLUInterleaved) {
            deep_gemm::sm100_store_swiglu_interleaved<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                GemmType::Normal, false,
                cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>
            (smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0, 0,
             epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
             tensor_map_cd, route_ptr, preact_ptr, preact_recv_idx, preact_topk_idx,
             num_topk, preact_stride, valid_rows);
        } else if constexpr (kFuseSwiGLU) {
            deep_gemm::sm100_store_swiglu_from_gate<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                GemmType::Normal, false,
                cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>
            (smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0, 0,
             epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
             tensor_map_cd, gate_ptr, route_ptr, stride_n);
        } else {
            epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                GemmType::Normal, false,
                cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>
            (smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0, 0,
             epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
             tensor_map_cd);
        }

        // P0-a: drain all outstanding TMA stores for this pass before the closing
        // sync. DeepGEMM relies on the kernel-exit boundary to flush the gate
        // GMEM write; since we collapsed the two GEMMs into back-to-back device
        // calls there is no kernel boundary, so the gate_buf store must be made
        // globally visible here or the next (up) pass's SwiGLU would read stale
        // gate values. tma_store_arrive was already issued inside the epilogue.
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<0>();
        MK_DG_DBG("EPI[%d]: store done", (int)epilogue_warp_idx);
    }
    // Advance the persistent accumulator iteration so the next pass (or next tile)
    // waits on the correct tmem phase.
    ++accum_iter;
    // System-scope fence so the BF16 gate_buf TMA-store is visible to the up
    // pass's SwiGLU GMEM reads across the whole cluster, then sync all CTAs.
    if constexpr (kFuseSwiGLU == false)
        __threadfence();
    MK_DG_DBG("exit: closing sync BEGIN");
    block_or_cluster_sync();
    MK_DG_DBG("exit: closing sync END");
#undef MK_DG_DBG
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void sm100_store_cd_row_scatter(
    const deep_gemm::utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
    const uint32_t& tmem_base_addr,
    const uint32_t& base_m_idx, const uint32_t& base_n_idx,
    const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
    const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
    const DownScatterParams& scatter) {

    using namespace deep_gemm;

    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    static_assert(kSwizzleCDMode == 128, "This down-scatter path assumes 128B CD swizzle");
    static_assert(kNumElemsPerBankGroup == 8, "BF16 down scatter expects 8 BF16 per segment");
    static_assert(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Only BF16 output is supported");
    static_assert(STORE_BLOCK_M == 128 && STORE_BLOCK_N == 64 && BLOCK_N == 128,
                  "Expected current DeepGEMM tile shape");

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    constexpr uint32_t kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    constexpr uint32_t kSegmentsPerStoreRow = STORE_BLOCK_N / kNumElemsPerBankGroup;
    constexpr uint32_t kScatterItems = STORE_BLOCK_M * kSegmentsPerStoreRow;

    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);
            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_type_t::template apply_index_n<STORE_BLOCK_N>(base_n_idx + s * STORE_BLOCK_N);

            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                auto smem_ptr = smem_base_ptr +
                                epilogue_warp_idx * 32 * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                uint32_t values[kNumElemsPerBankGroup];
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                    values[0], values[1], values[2], values[3],
                    values[4], values[5], values[6], values[7]);
                cutlass::arch::fence_view_async_tmem_load();
                ptx::st_shared(
                    smem_ptr,
                    math::cast_into_bf16_and_pack(values[0], values[1]),
                    math::cast_into_bf16_and_pack(values[2], values[3]),
                    math::cast_into_bf16_and_pack(values[4], values[5]),
                    math::cast_into_bf16_and_pack(values[6], values[7]));
            }

            if (w == kNumMWaves - 1 && s == kNumStores - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const uint32_t epi_tid = epilogue_warp_idx * 32 + lane_idx;
            for (uint32_t item = epi_tid; item < kScatterItems; item += kNumUMMAStoreThreads) {
                const uint32_t row_in_store = item / kSegmentsPerStoreRow;
                const uint32_t seg = item - row_in_store * kSegmentsPerStoreRow;
                const uint32_t global_row = m_idx + row_in_store;
                if (global_row >= static_cast<uint32_t>(scatter.valid_rows)) continue;

                const uint32_t row_in_warp = row_in_store & 31u;
                const uint32_t warp_row_group = row_in_store >> 5;
                const uint32_t bank_group = seg ^ (row_in_warp & 7u);
                const auto smem_ptr = smem_base_ptr +
                    warp_row_group * 32 * kSwizzleCDMode +
                    row_in_warp * (kNumBankGroupBytes * 8) +
                    bank_group * kNumBankGroupBytes;
                const int4 packed = *reinterpret_cast<const int4*>(smem_ptr);

                const int recv_token = scatter.recv_token_idx[global_row];
                const uint32_t hidden_i4 = (n_idx >> 3) + seg;
                if (scatter.is_single[global_row]) {
                    scatter.combine_input_i4[(int64_t)recv_token * scatter.hidden_int4 + hidden_i4] = packed;
                } else {
                    scatter.compute_output_slot_i4[(int64_t)(scatter.slot_base + global_row) * scatter.hidden_int4 + hidden_i4] = packed;
                }
            }

            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            __syncwarp();
        }
    }
}

// ===========================================================================
// dg_gemm_persistent — DeepGEMM persistent warp-specialized GEMM over a TILE
// SEQUENCE (not a single tile). barriers/TMEM are init/alloc'd ONCE by the
// caller (dg_init_barriers_tmem); the three warp roles each run their own
// MegaTileScheduler and loop over all tiles this cluster owns, with ZERO
// cluster sync between tiles — TMA pipeline rolls continuously across tiles.
//
// This mirrors compute_ref/sm100_bf16_gemm_dg_copy.cuh's `while(get_next_block)`
// structure, the difference being the scheduler enumerates THIS cluster's tiles
// (cluster_idx/num_clusters) instead of grid-global blocks, and there is no
// `cudaGridDependencySynchronize` (device-function context).
//
//   accum_iter : persists the TMEM accumulator phase across tiles AND across the
//                gate->up passes (caller threads it through both calls).
//   A closing block_or_cluster_sync() ensures all warps finish before return.
// ===========================================================================
template <bool kFuseSwiGLU, uint32_t kNumMulticast, bool kFuseSwiGLUInterleaved = false,
          cute::UMMA::Major kMajorB = cute::UMMA::Major::K>
__device__ void dg_gemm_persistent(
    const CUtensorMap* desc_a, const CUtensorMap* desc_b, const CUtensorMap* desc_cd,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter,
    const cutlass::bfloat16_t* gate_ptr, const float* route_ptr, uint32_t stride_n,
    cutlass::bfloat16_t* preact_ptr = nullptr,
    const int* preact_recv_idx = nullptr,
    const int* preact_topk_idx = nullptr,
    uint32_t num_topk = 0,
    uint32_t preact_stride = 0,
    uint32_t valid_rows = 0xffffffffu,
    // A-operand row base offset into desc_a's global tensor. Default 0 keeps the
    // legacy behavior (desc_a points at a per-group buffer with M origin at row 0).
    // When A is a shared buffer (e.g. recv_tokens), pass expert_slot_base+start_slot.
    uint32_t m_base = 0,
    const DownScatterParams* scatter_params = nullptr) {

    using namespace deep_gemm;
    using L = DgSmemLayout<kNumMulticast>;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t BLOCK_M = L::BLOCK_M, BLOCK_N = L::BLOCK_N, BLOCK_K = L::BLOCK_K;
    constexpr bool kIsMulticastOnA = L::kIsMulticastOnA;
    constexpr uint32_t LOAD_BLOCK_M = L::LOAD_BLOCK_M, LOAD_BLOCK_N = L::LOAD_BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = L::STORE_BLOCK_M, STORE_BLOCK_N = L::STORE_BLOCK_N;
    constexpr uint32_t kNumStages = L::kNumStages;
    constexpr uint32_t kNumEpilogueStages = L::kNumEpilogueStages;
    constexpr uint32_t kNumTMAStoreStages = L::kNumTMAStoreStages;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t kNumUMMAStoreThreads = L::kNumUMMAStoreThreads;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = L::SMEM_A_SIZE_PER_STAGE;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = L::SMEM_B_SIZE_PER_STAGE;

    const bool is_leader_cta = (kNumMulticast == 1) ? true : (cute::block_rank_in_cluster() == 0);
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    auto block_or_cluster_sync = [&]() {
        if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
        else __syncthreads();
    };

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(desc_a);
        cute::prefetch_tma_descriptor(desc_b);
        cute::prefetch_tma_descriptor(desc_cd);
    }

    // ---- SMEM partition (same layout as dg_gemm_tile, via DgSmemLayout) ----
    uint8_t* smem_buffer = reinterpret_cast<uint8_t*>(cluster_smem);
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + i * L::SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + L::SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + L::SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto bar = L::barrier_start(cluster_smem);
    auto full_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return bar + i; });
    auto empty_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return bar + (kNumStages + i); });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return bar + (kNumStages * 2 + i); });
    auto tmem_empty_barriers= utils::PatternVisitor([=](const uint32_t& i) { return bar + (kNumStages * 2 + kNumEpilogueStages + i); });

    const auto num_total_k_blocks = math::ceil_div<uint32_t>(shape_k, BLOCK_K);
    const uint32_t cta_rank = (kNumMulticast > 1) ? cute::block_rank_in_cluster() : 0;

    // Mainloop pipeline phases — roll CONTINUOUSLY across tiles (the key win).
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= (stage_idx == 0);
    };

    if (warp_idx == 0 and cute::elect_one_sync()) {
        // ================= TMA load warp =================
        MegaTileScheduler sched(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t m_block, n_block;
        while (sched.get_next_block(m_block, n_block)) {
            const uint32_t m_idx0 = m_block * BLOCK_M;
            const uint32_t n_idx0 = n_block * BLOCK_N;
            const uint32_t load_m_idx = m_base + m_idx0 + (kIsMulticastOnA ? cta_rank * LOAD_BLOCK_M : 0);
            const uint32_t load_n_idx = n_idx0 + (kIsMulticastOnA ? 0 : cta_rank * LOAD_BLOCK_N);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kDgSwizzleA, cutlass::bfloat16_t>(
                    desc_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, load_m_idx, kNumMulticast);
                if constexpr (kMajorB == cute::UMMA::Major::K) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kDgSwizzleB, cutlass::bfloat16_t>(
                        desc_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, load_n_idx, kNumMulticast);
                } else {
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kDgSwizzleB, cutlass::bfloat16_t>(
                        desc_b, full_barriers[stage_idx], smem_b[stage_idx], load_n_idx, k_idx, kNumMulticast);
                }
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta)
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                else
                    full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // ================= MMA issue warp (leader CTA) =================
        constexpr bool kDoMergeStages = (kNumStages >= 8 && kMajorB == cute::UMMA::Major::K);
        constexpr uint32_t kNumMinStages = 8;
        constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages / kNumMinStages : 1;
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, kMajorB>();
        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_ATOM_K, kDgSwizzleA>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kDgSwizzleB>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        auto umma_arrive = [](const uint64_t* barrier) {
            if constexpr (kNumMulticast == 1) cutlass::arch::umma_arrive(barrier);
            else { constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                   cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask); }
        };

        MegaTileScheduler sched(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t m_block, n_block;
        uint32_t local_accum = accum_iter;   // MMA advances its own copy
        while (sched.get_next_block(m_block, n_block)) {
            const uint32_t accum_stage_idx = local_accum % kNumEpilogueStages;
            const uint32_t accum_phase_idx = (local_accum / kNumEpilogueStages) & 1;

            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();
                const auto a_base = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_base = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    using mma_t = cute::conditional_t<kNumMulticast == 1,
                                      ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                    auto issue_umma = [&]<uint32_t kUMMAKIdx>() {
                        constexpr uint32_t kAtomKIdx  = kUMMAKIdx * UMMA_K / BLOCK_ATOM_K;
                        constexpr uint32_t kInnerKIdx = kUMMAKIdx * UMMA_K % BLOCK_ATOM_K;
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kDgSwizzleA, cutlass::bfloat16_t>(
                                        a_base, kAtomKIdx * LOAD_BLOCK_M * BLOCK_ATOM_K, kInnerKIdx);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kDgSwizzleB, cutlass::bfloat16_t>(
                                        b_base, kAtomKIdx * LOAD_BLOCK_N * BLOCK_ATOM_K, kInnerKIdx);
                        mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                   kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
                    };
                    utils::for_each_static_until<BLOCK_K / UMMA_K>(
                        std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), issue_umma);
                }
                __syncwarp();
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (k_block_idx == num_total_k_blocks - 1)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            }
            ++ local_accum;   // advance TMEM accumulator phase per tile (local copy)
        }
        // Terminal drain: after ALL tiles, wait the last tmem_empty so barriers are
        // safe to reuse / deconstruct. This is DeepGEMM's post-loop drain — valid
        // here because this is the END of the persistent loop (not per-tile).
        if (kNumMulticast > 1 and local_accum > accum_iter) {
            const uint32_t last = local_accum - 1;
            tmem_empty_barriers[last % kNumEpilogueStages]->wait((last / kNumEpilogueStages) & 1);
        }
    } else if (warp_idx >= kDgWsNonEpiThreads / 32 and warp_idx < (kDgWsNonEpiThreads + kDgWsEpiThreads) / 32) {
        // ================= Epilogue store warps =================
        const auto epilogue_warp_idx = warp_idx - (kDgWsNonEpiThreads / 32);
        const cute::TmaDescriptor& tensor_map_cd = *desc_cd;
        uint32_t tma_stage_idx = 0;

        MegaTileScheduler sched(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t m_block, n_block;
        uint32_t local_accum = accum_iter;   // epilogue advances its own copy
        while (sched.get_next_block(m_block, n_block)) {
            const uint32_t accum_stage_idx = local_accum % kNumEpilogueStages;
            const uint32_t accum_phase_idx = (local_accum / kNumEpilogueStages) & 1;
            const uint32_t m_idx0 = m_block * BLOCK_M;
            const uint32_t n_idx0 = n_block * BLOCK_N;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;

            if constexpr (kFuseSwiGLUInterleaved) {
                deep_gemm::sm100_store_swiglu_interleaved<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    GemmType::Normal, false, cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0, 0,
                 epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd, route_ptr, preact_ptr, preact_recv_idx, preact_topk_idx,
                 num_topk, preact_stride, valid_rows);
            } else if constexpr (kFuseSwiGLU) {
                deep_gemm::sm100_store_swiglu_from_gate<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    GemmType::Normal, false, cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0, 0,
                 epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd, gate_ptr, route_ptr, stride_n);
            } else if (scatter_params != nullptr) {
                sm100_store_cd_row_scatter<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0,
                 epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
                 *scatter_params);
            } else {
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    GemmType::Normal, false, cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0, 0,
                 epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
            ++ local_accum;
        }
        // Drain all outstanding TMA stores before the closing sync (gate_buf must be
        // globally visible before the up pass's SwiGLU reads it).
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<0>();
    }

    // Sync accum_iter across the caller: all roles must agree on the post-loop
    // value so the NEXT persistent call (up pass / down) starts at the right phase.
    // Every tile advanced accum_iter once; the MMA warp already did that in its
    // local copy. Compute the deterministic total here for the shared reference.
    {
        MegaTileScheduler probe(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t mb, nb, tiles_done = 0;
        while (probe.get_next_block(mb, nb)) ++ tiles_done;
        accum_iter += tiles_done;
    }

    // System-scope fence so a non-fused (gate) pass's BF16 GMEM store is visible
    // to the next pass's SwiGLU GMEM reads across the cluster, then sync.
    if constexpr (kFuseSwiGLU == false)
        __threadfence();
    block_or_cluster_sync();
}


// TMEM dealloc (call once after persistent loop). Allocator must match the run path.
__device__ inline void umma_dealloc(char* cluster_smem) {
    using Allocator = cute::conditional_t<kDgRunMulticast == 1,
                                          cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    constexpr uint32_t SMEM_CD_SIZE = kDgStoreBlockM * kDgStoreBlockN * sizeof(cutlass::bfloat16_t) * kDgWsNumTmaStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = kDgLoadBlockM * kDgBlockK * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = kDgLoadBlockN * kDgBlockK * sizeof(cutlass::bfloat16_t);
    uint8_t* smem_buffer = reinterpret_cast<uint8_t*>(cluster_smem);
    auto barrier_start_ptr = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(
        smem_buffer + SMEM_CD_SIZE + kDgWsNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + kDgWsNumStages * 3 + kDgWsNumEpilogueStages * 2 + 1);
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    if constexpr (kDgRunMulticast > 1) deep_gemm::comm::cluster_sync_with_relaxed_arrive();
    else __syncthreads();
    if (warp_idx == 0)
        Allocator().free(*tmem_ptr_in_smem, kDgWsNumTmemCols);
    if constexpr (kDgRunMulticast > 1) deep_gemm::comm::cluster_sync_with_relaxed_arrive();
    else __syncthreads();
}

// DeepGEMM gate/up + SwiGLU for one (m_block,n_block) tile. Mirrors
// compute_ref/umma_swiglu_ws_dg.cu launch_swiglu_dg_copy: gate GEMM materializes
// BF16 gate, then up GEMM runs with the SwiGLU-from-gate epilogue.
__device__ inline void umma_up_swiglu_tile(
    const CUtensorMap* desc_a, const CUtensorMap* desc_gate_cd, const CUtensorMap* desc_act_cd,
    const CUtensorMap* desc_wgate, const CUtensorMap* desc_wup,
    int i_tile, __nv_bfloat16* gate_out, const float* route_w,
    int M, int I, int d, char* cluster_smem, bool& tmem_allocated, uint32_t& accum_iter) {
    const int n_tiles = (I + kDgBlockN - 1) / kDgBlockN;
    const int m_block = i_tile / n_tiles;
    const int n_block = i_tile - m_block * n_tiles;

    // Pass 1: gate = A @ Wg^T -> gate_buf (BF16), no SwiGLU.
    dg_gemm_tile<false, kDgRunMulticast>(desc_a, desc_wgate, desc_gate_cd,
                        m_block, n_block, M, I, d,
                        cluster_smem, tmem_allocated, accum_iter,
                        nullptr, nullptr, 0);

    // Pass 2: act = silu(gate)*(A @ Wu^T)*route -> act_buf, fused SwiGLU epilogue
    // reads the BF16 gate_buf just written. accum_iter persists the TMEM
    // accumulator pipeline phase across the two passes (DeepGEMM-equivalent).
    dg_gemm_tile<true, kDgRunMulticast>(desc_a, desc_wup, desc_act_cd,
                       m_block, n_block, M, I, d,
                       cluster_smem, tmem_allocated, accum_iter,
                       reinterpret_cast<const cutlass::bfloat16_t*>(gate_out), route_w, (uint32_t)I);
}

// ===========================================================================
// umma_up_swiglu_persistent — task-level persistent gate+up+SwiGLU for THIS
// cluster. Replaces the per-tile umma_up_swiglu_tile loop in compute_worker.
//
// Caller must:
//   - call dg_init_barriers_tmem<kNumMulticast>(cluster_smem) ONCE before this
//   - call dg_dealloc_tmem<kNumMulticast>(cluster_smem) ONCE after this
//
// Structure (route 2, A1 precision path):
//   gate persistent loop (all this cluster's tiles -> gate_buf)  [no SwiGLU]
//   -> dg_gemm_persistent's closing fence+sync makes gate_buf visible
//   up   persistent loop (all this cluster's tiles -> act_buf, SwiGLU reads gate_buf)
//   accum_iter threads through both so the TMEM phase ring stays continuous.
//
//   cluster_idx  : cluster_in_group (0..num_clusters-1)
//   num_clusters : COMPUTE_GROUP_SIZE/2 (16) for the 16-cluster sharing
// ===========================================================================
__device__ inline void umma_up_swiglu_persistent(
    const CUtensorMap* desc_a, const CUtensorMap* desc_gate_cd, const CUtensorMap* desc_act_cd,
    const CUtensorMap* desc_wgate, const CUtensorMap* desc_wup,
    __nv_bfloat16* gate_out, const float* route_w,
    int M, int I, int d,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter) {

    // Pass 1: gate = A @ Wg^T -> gate_buf (BF16), persistent over all tiles.
    dg_gemm_persistent<false, kDgRunMulticast>(
        desc_a, desc_wgate, desc_gate_cd,
        (uint32_t)M, (uint32_t)I, (uint32_t)d,
        cluster_idx, num_clusters, cluster_smem, accum_iter,
        nullptr, nullptr, 0);

    // Pass 2: act = silu(gate)*(A @ Wu^T)*route -> act_buf, SwiGLU reads gate_buf.
    dg_gemm_persistent<true, kDgRunMulticast>(
        desc_a, desc_wup, desc_act_cd,
        (uint32_t)M, (uint32_t)I, (uint32_t)d,
        cluster_idx, num_clusters, cluster_smem, accum_iter,
        reinterpret_cast<const cutlass::bfloat16_t*>(gate_out), route_w, (uint32_t)I);
}

// ===========================================================================
// umma_gateup_interleaved_persistent — SINGLE gate/up GEMM + in-TMEM SwiGLU.
// Wgu rows are element-interleaved [g0,u0,g1,u1,...]. The GEMM accumulates
// GU[M,2I] in TMEM, then the epilogue collapses every (gate,up) pair to act[M,I]
// and stores directly to desc_act_cd. No GU round-trip through GMEM is needed.
// ===========================================================================
__device__ inline void umma_gateup_interleaved_persistent(
    const CUtensorMap* desc_a, const CUtensorMap* desc_wgateup, const CUtensorMap* desc_act_cd,
    const float* route_w, int M, int I, int d,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter,
    cutlass::bfloat16_t* preact_ptr = nullptr,
    const int* preact_recv_idx = nullptr,
    const int* preact_topk_idx = nullptr,
    uint32_t num_topk = 0,
    uint32_t preact_stride = 0,
    uint32_t valid_rows = 0xffffffffu,
    uint32_t a_m_base = 0) {

    dg_gemm_persistent<false, kDgRunMulticast, true>(
        desc_a, desc_wgateup, desc_act_cd,
        (uint32_t)M, (uint32_t)(2 * I), (uint32_t)d,
        cluster_idx, num_clusters, cluster_smem, accum_iter,
        nullptr, route_w, 0, preact_ptr, preact_recv_idx, preact_topk_idx,
        num_topk, preact_stride, valid_rows, a_m_base);
}


// ===========================================================================
// umma_down_persistent — task-level persistent down-proj GEMM for THIS cluster.
// D = act @ W_down^T. Pure GEMM (no SwiGLU). Mirrors umma_up_swiglu_persistent
// but a single pass. Caller must bracket with dg_init_barriers_tmem /
// dg_dealloc_tmem exactly like gate/up.
//
//   desc_act_a  : up_buf [M, I] K-major A (in.act_a)
//   desc_wdown  : W_down[expert] [hidden, I] = [N, K] K-major
//   desc_down_cd: down_buf [M, hidden] row-major CD (in.down_cd)
//   K == intermediate (I), N == hidden.
//   accum_iter starts fresh at 0 (down runs after its own init_barriers_tmem).
// ===========================================================================
__device__ inline void umma_down_persistent(
    const CUtensorMap* desc_act_a, const CUtensorMap* desc_wdown, const CUtensorMap* desc_down_cd,
    int M, int hidden, int intermediate,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter) {

    dg_gemm_persistent<false, kDgRunMulticast>(
        desc_act_a, desc_wdown, desc_down_cd,
        (uint32_t)M, (uint32_t)hidden, (uint32_t)intermediate,
        cluster_idx, num_clusters, cluster_smem, accum_iter,
        nullptr, nullptr, 0);
}

__device__ inline void umma_down_scatter_persistent(
    const CUtensorMap* desc_act_a, const CUtensorMap* desc_wdown, const CUtensorMap* desc_down_cd,
    int M, int hidden, int intermediate,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter,
    const DownScatterParams& scatter) {

    dg_gemm_persistent<false, kDgRunMulticast>(
        desc_act_a, desc_wdown, desc_down_cd,
        (uint32_t)M, (uint32_t)hidden, (uint32_t)intermediate,
        cluster_idx, num_clusters, cluster_smem, accum_iter,
        nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0, 0, 0xffffffffu, 0,
        &scatter);
}

// Backward dgrad consumes original weights in [K,N] row-major form and asks the
// DeepGEMM mainloop to treat B as MN-major instead of materializing W^T.
__device__ inline void umma_dgrad_mn_persistent(
    const CUtensorMap* desc_a, const CUtensorMap* desc_b_mn, const CUtensorMap* desc_cd,
    int M, int N, int K,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter,
    uint32_t m_base = 0) {

    dg_gemm_persistent<false, kDgRunMulticast, false, cute::UMMA::Major::MN>(
        desc_a, desc_b_mn, desc_cd,
        (uint32_t)M, (uint32_t)N, (uint32_t)K,
        cluster_idx, num_clusters, cluster_smem, accum_iter,
        nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0, 0, 0xffffffffu, m_base);
}

// DeepGEMM down-proj for one (m_block, n_block) tile: D = act @ W_down^T.
// Pure GEMM (no SwiGLU); reuses dg_gemm_tile<false>. Shares the same TMEM
// accum pipeline / accum_iter as the gate/up passes so the single
// `tmem_allocated` flag and allocator stay consistent (fixes the old IMA where
// the CuTe down path used a different TMEM handle location).
//   desc_act_a : up_buf [M, I] K-major A descriptor (in.act_a)
//   desc_wdown : W_down[expert] [hidden, I] = [N, K] K-major
//   desc_down_cd : down_buf [M, hidden] row-major CD (in.down_cd)
//   d_tile : linear output tile id over [M_tiles, hidden_tiles]
__device__ inline void umma_down_proj_tile_dg(
    const CUtensorMap* desc_act_a, const CUtensorMap* desc_wdown, const CUtensorMap* desc_down_cd,
    int d_tile, int M, int hidden, int intermediate,
    char* cluster_smem, bool& tmem_allocated, uint32_t& accum_iter) {
    const int n_tiles = (hidden + kDgBlockN - 1) / kDgBlockN;
    const int m_block = d_tile / n_tiles;
    const int n_block = d_tile - m_block * n_tiles;
    dg_gemm_tile<false, kDgRunMulticast>(desc_act_a, desc_wdown, desc_down_cd,
                       m_block, n_block, M, hidden, intermediate,
                       cluster_smem, tmem_allocated, accum_iter,
                       nullptr, nullptr, 0);
}

// Per-expert raw W_down descriptors ([hidden, I] = [N, K] K-major).
struct ComputeDownTmaAtoms {
    int num_experts;
    int hidden;
    int intermediate;
    CUtensorMap wdown[kMaxLocalExperts];
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
        atoms.wdown[e] = dg_make_b_desc(wd_e, hidden, intermediate);   // [N=hidden, K=intermediate]
    }
}

// Per-expert backward dgrad descriptors over the original BF16 weights. B is
// passed as MN-major [K,N], so no physical W_down_T/W_gateup_T copy is needed.
struct ComputeBackwardTmaAtoms {
    int num_experts;
    int hidden;
    int intermediate;
    CUtensorMap wdown[kMaxLocalExperts];
    CUtensorMap wgateup[kMaxLocalExperts];
};

inline void build_compute_backward_tma_atoms(ComputeBackwardTmaAtoms& atoms,
                                             const __nv_bfloat16* W_down,
                                             const __nv_bfloat16* W_gateup,
                                             int E, int hidden, int intermediate) {
    EP_HOST_ASSERT(E <= kMaxLocalExperts);
    atoms.num_experts = E;
    atoms.hidden = hidden;
    atoms.intermediate = intermediate;
    for (int e = 0; e < E; ++e) {
        const __nv_bfloat16* wd_e = W_down + (size_t)e * hidden * intermediate;              // [K=hidden, N=I]
        const __nv_bfloat16* wgu_e = W_gateup + (size_t)e * (2 * intermediate) * hidden;     // [K=2I, N=hidden]
        atoms.wdown[e] = dg_make_b_mn_desc(wd_e, intermediate, hidden);
        atoms.wgateup[e] = dg_make_b_mn_desc(wgu_e, hidden, 2 * intermediate);
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
// down-proj now uses the DeepGEMM dg_gemm_tile path (umma_down_proj_tile_dg,
// defined above). The old CuTe ClusterSharedStorage + Allocator2Sm down kernel
// was removed: it stored the TMEM handle in a different smem location and shared
// the single `umma_tmem_allocated` flag with gate/up, which caused a TMEM
// handle mismatch / IMA once gate/up switched to the DeepGEMM allocator.
// ---------------------------------------------------------------------------

}  // namespace umma
}  // namespace megakernel
}  // namespace deep_ep
