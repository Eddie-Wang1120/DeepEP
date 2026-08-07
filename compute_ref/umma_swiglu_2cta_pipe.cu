// umma_swiglu_2cta_pipe.cu — Stage (3'-pipe): 2CTA (2x1SM) UMMA fused gate+up
// GEMM with in-register SwiGLU epilogue, M_tile=256, plus a 2-stage software
// pipeline that overlaps TMA load of K-tile (k+1) with the UMMA of K-tile (k).
//
// Difference vs umma_swiglu_2cta.cu (serial):
//   - SMEM A/Bg/Bu are 2-stage ring buffers (kStages=2).
//   - Per-stage pair of mbarriers: tma_full[s] (TMA producer -> MMA consumer),
//     mma_empty[s] (MMA consumer -> TMA producer, "stage free to reload").
//   - Prologue issues the load for k=0; the mainloop issues load for k+1 then
//     waits/MMAs k, so TMA(k+1) overlaps MMA(k).
//
//   act[m,n] = silu(A@Wg^T) * (A@Wu^T) * route_w[m]      (M=256)
//
// Correctness: vs host fused-SwiGLU reference (same as serial version).
// Performance: pipelined 2CTA vs serial 2CTA vs WMMA baseline.
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -arch=sm_103a -O3 \
//        -I../cutlass_ref/include -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        umma_swiglu_2cta_pipe.cu -o umma_swiglu_2cta_pipe
//   ./umma_swiglu_2cta_pipe            # default M=256 K=4096 N=4096

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

using namespace cute;

#define CHECK_CUDA(call)                                                         \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                               \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":"      \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

static constexpr int kStages = 2;

// A/Bg/Bu are 2-stage ring buffers. Each stage has its own pair of mbarriers.
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout> * kStages> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout> * kStages> Bg;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout> * kStages> Bu;
  alignas(16) cute::uint64_t tma_full[kStages];   // TMA arrives (producer -> consumer)
  alignas(16) cute::uint64_t mma_empty[kStages];  // MMA arrives (consumer -> producer)
  alignas(16) cute::uint32_t tmem_base_ptr;

  // Per-stage tensor views. ASmemLayout is the single-stage layout; we index the
  // stage by offsetting the base pointer by cosize * stage.
  CUTE_DEVICE auto tensor_sA(int s, ASmemLayout l)  { return make_tensor(make_smem_ptr(A.begin()  + cute::cosize_v<ASmemLayout> * s), l); }
  CUTE_DEVICE auto tensor_sBg(int s, BSmemLayout l) { return make_tensor(make_smem_ptr(Bg.begin() + cute::cosize_v<BSmemLayout> * s), l); }
  CUTE_DEVICE auto tensor_sBu(int s, BSmemLayout l) { return make_tensor(make_smem_ptr(Bu.begin() + cute::cosize_v<BSmemLayout> * s), l); }
};

template <class SharedStorage, class ASmemLayout, class BSmemLayout,
          class ATensor, class BgTensor, class BuTensor, class DTensor, class RTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomBg, class TmaAtomBu>
__global__ static void
umma_swiglu_2cta_pipe_device(ATensor mA, BgTensor mBg, BuTensor mBu, DTensor mD, RTensor mRoute,
                             ASmemLayout sA_layout, BSmemLayout sB_layout,
                             MmaTiler_MNK mma_tiler, TiledMMA tiled_mma, ClusterShape_MNK cluster_shape,
                             CUTE_GRID_CONSTANT TmaAtomA  const tma_atom_A,
                             CUTE_GRID_CONSTANT TmaAtomBg const tma_atom_Bg,
                             CUTE_GRID_CONSTANT TmaAtomBu const tma_atom_Bu) {
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                                   blockIdx.x / size<0>(cluster_layout_vmnk),
                                   blockIdx.y, _);
  auto mma_coord = select<1,2,3>(mma_coord_vmnk);
  Tensor gA  = local_tile(mA,  mma_tiler, mma_coord, Step<_1, X,_1>{});
  Tensor gBg = local_tile(mBg, mma_tiler, mma_coord, Step< X,_1,_1>{});
  Tensor gBu = local_tile(mBu, mma_tiler, mma_coord, Step< X,_1,_1>{});
  Tensor gD  = local_tile(mD,  mma_tiler, mma_coord, Step<_1,_1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA  = cta_mma.partition_A(gA);
  Tensor tCgBg = cta_mma.partition_B(gBg);
  Tensor tCgBu = cta_mma.partition_B(gBu);
  Tensor tCgD  = cta_mma.partition_C(gD);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);   // single TMEM acc, reused per pass

  // Per-stage SMEM fragments. The single-stage layout is sA_layout/sB_layout.
  // Build fragment_A/B for each stage by viewing that stage's SMEM tile.
  Tensor tCsA0  = smem.tensor_sA(0, sA_layout);
  Tensor tCsA1  = smem.tensor_sA(1, sA_layout);
  Tensor tCsBg0 = smem.tensor_sBg(0, sB_layout);
  Tensor tCsBg1 = smem.tensor_sBg(1, sB_layout);
  Tensor tCsBu0 = smem.tensor_sBu(0, sB_layout);
  Tensor tCsBu1 = smem.tensor_sBu(1, sB_layout);

  auto tCrA0  = cta_mma.make_fragment_A(tCsA0);
  auto tCrA1  = cta_mma.make_fragment_A(tCsA1);
  auto tCrBg0 = cta_mma.make_fragment_B(tCsBg0);
  auto tCrBg1 = cta_mma.make_fragment_B(tCsBg1);
  auto tCrBu0 = cta_mma.make_fragment_B(tCsBu0);
  auto tCrBu1 = cta_mma.make_fragment_B(tCsBu1);

  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
  __syncthreads();
  tCtAcc.data() = smem.tmem_base_ptr;

  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};

  // TMA partitions, per stage (SMEM destination differs by stage).
  auto tma_part_A = [&](int s) {
    auto sAv = (s == 0) ? tCsA0 : tCsA1;
    return tma_partition(tma_atom_A, get<2>(cta_in_cluster_coord_vmnk),
                         make_layout(size<2>(cluster_layout_vmnk)),
                         group_modes<0,3>(sAv), group_modes<0,3>(tCgA));
  };
  auto tma_part_Bg = [&](int s) {
    auto sBv = (s == 0) ? tCsBg0 : tCsBg1;
    return tma_partition(tma_atom_Bg, get<1>(cta_in_cluster_coord_vmnk),
                         make_layout(size<1>(cluster_layout_vmnk)),
                         group_modes<0,3>(sBv), group_modes<0,3>(tCgBg));
  };
  auto tma_part_Bu = [&](int s) {
    auto sBv = (s == 0) ? tCsBu0 : tCsBu1;
    return tma_partition(tma_atom_Bu, get<1>(cta_in_cluster_coord_vmnk),
                         make_layout(size<1>(cluster_layout_vmnk)),
                         group_modes<0,3>(sBv), group_modes<0,3>(tCgBu));
  };

  uint16_t mcast_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t mcast_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t mcast_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                     create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  (void)mcast_c;  // pipe kernel releases empty barrier via umma_arrive_2x1SM_sm0, not multicast

  // Transaction bytes for one K-tile of A + one K-tile of B.
  auto [tAgA0, tAsA0] = tma_part_A(0);
  auto [tBg0, tBgs0]  = tma_part_Bg(0);
  int txbytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA0))
              + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBgs0));

  // Initialize per-stage barriers.
  //   tma_full[s]: producer = 1 thread issuing TMA (arrival via expect_tx + TMA bytes).
  //   mma_empty[s]: consumer = num mcast participants (UMMA arrives on completion).
  // Barrier arrival counts follow CUTLASS PipelineTmaUmmaAsync::init_barriers:
  //   producer (tma_full): producer_arv_cnt = 1 (one thread issues expect_tx + TMA).
  //   consumer (mma_empty): multicast_consumer_arrival_count =
  //       (size<0>(cluster)/atom0 + size<1>(cluster)/atom1 - 1).
  //   For cluster=(2,1,1) with 2x1SM AtomThrShape=(2,1,1): (2/2 + 1/1 - 1) = 1.
  // The empty barrier is released by umma_arrive_2x1SM_sm0 (single SM0 arrival),
  // NOT by umma_arrive_multicast_2x1SM (which is the FULL-barrier MMA-completion path).
  constexpr int producer_arv_cnt = 1;
  const int consumer_arv_cnt = (size<0>(cluster_layout_vmnk) /* /atom0=2/2=1 */)
                             + (size<1>(cluster_layout_vmnk) /* /atom1=1 */) - 1;
  if (elect_one_warp && elect_one_thr) {
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(smem.tma_full[s],  producer_arv_cnt);
      cute::initialize_barrier(smem.mma_empty[s], consumer_arv_cnt);
    }
  }
  cute::cluster_sync();

  // Drain helper bound to acc.
  TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r = t2r.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
  Tensor tDgD   = thr_t2r.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrG = make_tensor<AccType>(shape(tDgD));
  Tensor tDrU = make_tensor<AccType>(shape(tDgD));

  const int nK = size<3>(tCgA);

  // Helper: issue TMA load for k-tile `k` into ring stage `s` for current pass.
  // Only the leader CTA's elected thread arms the FULL barrier with the expected
  // transaction bytes; the TMA copy itself is also issued by that thread and the
  // hardware multicasts data into BOTH CTAs' SMEM stage and signals the FULL
  // barrier on SM0 (the cluster-shared barrier).
  auto issue_load = [&](int pass, int k, int s) {
    if (!(elect_one_cta && elect_one_warp && elect_one_thr)) return;
    cute::set_barrier_transaction_bytes(smem.tma_full[s], txbytes);
    auto [tAgA, tAsA] = tma_part_A(s);
    copy(tma_atom_A.with(smem.tma_full[s], mcast_a), tAgA(_,k), tAsA);
    if (pass == 0) {
      auto [tBg, tBgs] = tma_part_Bg(s);
      copy(tma_atom_Bg.with(smem.tma_full[s], mcast_b), tBg(_,k), tBgs);
    } else {
      auto [tBu, tBus] = tma_part_Bu(s);
      copy(tma_atom_Bu.with(smem.tma_full[s], mcast_b), tBu(_,k), tBus);
    }
  };

  // Two passes: pass 0 = gate (Wg), pass 1 = up (Wu). Single TMEM acc reused.
  //
  // Pipeline phase model (mirrors CUTLASS PipelineState):
  //   - A stage's full barrier (tma_full) starts at phase 0; the producer's TMA
  //     arrival flips it to 1 on first fill, so the consumer waits phase 0 first
  //     time, phase 1 on the wrap-around revisit, etc.
  //   - A stage's empty barrier (mma_empty) starts at phase 0 meaning "free".
  //     The producer must wait phase 0 the FIRST time it wants to *reload* a
  //     stage (i.e. on the wrap-around), because the consumer's umma_arrive flips
  //     it to 1 after consuming. So the producer's empty-wait phase starts at 0
  //     and flips each reload.
  //   Per-stage parity bits below capture exactly this for kStages=2.
  for (int pass = 0; pass < 2; ++pass) {
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    // Pipeline structure (standard producer-ahead):
    //   prologue: producer issues loads for k=0..kStages-1 (fills the ring).
    //   steady:   for k=0..nK-1:
    //               consumer waits tma_full[s], issues MMA, releases mma_empty[s];
    //               producer (for the NEXT-but-kStages tile) waits mma_empty of
    //               that tile's stage, then issues its TMA.
    // The reload of a stage is separated from the MMA that consumed it by a full
    // kStages iterations, so the async tcgen05 MMA has drained its SMEM reads
    // (its umma_arrive release is the gate) before TMA overwrites the stage.
    int full_phase[kStages]  = {0, 0};
    int empty_phase[kStages] = {0, 0};

    // ---- Prologue: fill all ring stages. ----
    int prologue = (nK < kStages) ? nK : kStages;
    for (int s = 0; s < prologue; ++s) {
      issue_load(pass, s, s);
    }

    for (int k = 0; k < nK; ++k) {
      int s = k % kStages;

      // Consumer: leader waits this stage's FULL fill and issues the MMA.
      // (FULL wait is leader-only: only the leader consumes via tcgen05 MMA.)
      if (elect_one_cta) {
#ifdef PIPE_DEBUG
        if (threadIdx.x == 0) printf("[DBG] blk=%d k=%d s=%d L1 wait tma_full ph=%d\n",(int)blockIdx.x,k,s,full_phase[s]);
#endif
        cute::wait_barrier(smem.tma_full[s], full_phase[s]);
        full_phase[s] ^= 1;
#ifdef PIPE_DEBUG
        if (threadIdx.x == 0) printf("[DBG] blk=%d k=%d s=%d L2 tma_full done, MMA\n",(int)blockIdx.x,k,s);
#endif
        if (elect_one_warp) {
          if (s == 0) {
            auto& rA = tCrA0;
            auto& rB = (pass == 0) ? tCrBg0 : tCrBu0;
            for (int kb = 0; kb < size<2>(rA); ++kb) {
              gemm(tiled_mma, rA(_,_,kb), rB(_,_,kb), tCtAcc);
              tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
          } else {
            auto& rA = tCrA1;
            auto& rB = (pass == 0) ? tCrBg1 : tCrBu1;
            for (int kb = 0; kb < size<2>(rA); ++kb) {
              gemm(tiled_mma, rA(_,_,kb), rB(_,_,kb), tCtAcc);
              tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
          }
          // Release stage s on the cluster-shared EMPTY barrier (SM0). The
          // enqueued tcgen05 MMA's SMEM reads are ordered before this arrival.
          cutlass::arch::umma_arrive_2x1SM_sm0(&smem.mma_empty[s]);
        }
      }
      // Producer: refill stage s for its next user (k + kStages). The EMPTY wait
      // must be done by BOTH CTAs (not gated by elect_one_cta) so the whole
      // cluster observes the phase flip before the leader overwrites the stage.
      // This mirrors the serial reference where the mma_barrier wait is run by
      // all CTAs while the tma_barrier wait is leader-only.
      int k_refill = k + kStages;
      if (k_refill < nK) {
#ifdef PIPE_DEBUG
        if (threadIdx.x == 0) printf("[DBG] blk=%d k=%d s=%d L3 wait mma_empty ph=%d\n",(int)blockIdx.x,k,s,empty_phase[s]);
#endif
        cute::wait_barrier(smem.mma_empty[s], empty_phase[s]);
        empty_phase[s] ^= 1;
#ifdef PIPE_DEBUG
        if (threadIdx.x == 0) printf("[DBG] blk=%d k=%d s=%d L4 mma_empty done, load %d\n",(int)blockIdx.x,k,s,k_refill);
#endif
        issue_load(pass, k_refill, s);   // issue_load itself is leader-only inside
      }
    }

    // Drain: both CTAs wait the final EMPTY releases before reading out TMEM.
    int drain = (nK < kStages) ? nK : kStages;
    for (int s = 0; s < drain; ++s) {
#ifdef PIPE_DEBUG
      if (threadIdx.x == 0) printf("[DBG] blk=%d DRAIN s=%d wait mma_empty ph=%d\n",(int)blockIdx.x,s,empty_phase[s]);
#endif
      cute::wait_barrier(smem.mma_empty[s], empty_phase[s]);
      empty_phase[s] ^= 1;
    }

    if (pass == 0) copy(t2r, tDtAcc, tDrG);
    else           copy(t2r, tDtAcc, tDrU);
    cute::cluster_sync();   // ensure acc drained before reusing TMEM
  }

  // SwiGLU epilogue: act = silu(gate)*up*route_w[m]
  Tensor cD  = make_identity_tensor(shape(mD));
  Tensor gcD = local_tile(cD, mma_tiler, mma_coord, Step<_1,_1, X>{});
  Tensor tCgcD = cta_mma.partition_C(gcD);
  Tensor tDcD  = thr_t2r.partition_D(tCgcD);

  using OutType = typename DTensor::value_type;
  Tensor tDrAct = make_tensor<OutType>(shape(tDgD));
  CUTE_UNROLL
  for (int i = 0; i < size(tDrG); ++i) {
    int m = get<0>(tDcD(i));
    float g = static_cast<float>(tDrG(i));
    float u = static_cast<float>(tDrU(i));
    float silu_g = g * (1.0f / (1.0f + ::expf(-g)));
    tDrAct(i) = static_cast<OutType>(silu_g * u * static_cast<float>(mRoute(m)));
  }
  copy(tDrAct, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeAB>
void launch_umma_swiglu_2cta_pipe(TypeAB const* dA, TypeAB const* dWg, TypeAB const* dWu,
                                  TypeAB* dAct, float const* dRoute,
                                  int M, int N, int K, cudaStream_t stream) {
  using TypeC = float;
  Layout layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  Layout layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  Layout layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  Layout layout_R = make_layout(make_shape(M), make_stride(Int<1>{}));
  Tensor mA  = make_tensor(make_gmem_ptr(dA),  layout_A);
  Tensor mBg = make_tensor(make_gmem_ptr(dWg), layout_B);
  Tensor mBu = make_tensor(make_gmem_ptr(dWu), layout_B);
  Tensor mD  = make_tensor(make_gmem_ptr(dAct), layout_D);
  Tensor mR  = make_tensor(make_gmem_ptr(dRoute), layout_R);

  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TypeAB, TypeAB, TypeC, 256, 256,
                                 UMMA::Major::K, UMMA::Major::K>{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeAB>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeAB>{}, mma_shape_B);
  using SMEMStorage = SharedStorage<TypeAB, TypeAB, decltype(sA_layout), decltype(sB_layout)>;

  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  Copy_Atom tma_atom_A  = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA,  sA_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_atom_Bg = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mBg, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_atom_Bu = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mBu, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mA_tma  = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor mBg_tma = tma_atom_Bg.get_tma_tensor(shape(mBg));
  Tensor mBu_tma = tma_atom_Bu.get_tma_tensor(shape(mBu));

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(size(ceil_div(M, bM * size<1>(cluster_layout_vmnk))) * dimCluster.x,
               size(ceil_div(N, bN * size<2>(cluster_layout_vmnk))) * dimCluster.y);
  int smemBytes = sizeof(SMEMStorage);

  auto* kernel_ptr = &umma_swiglu_2cta_pipe_device<SMEMStorage,
                          decltype(sA_layout), decltype(sB_layout),
                          decltype(mA_tma), decltype(mBg_tma), decltype(mBu_tma), decltype(mD), decltype(mR),
                          decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                          decltype(tma_atom_A), decltype(tma_atom_Bg), decltype(tma_atom_Bu)>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*) kernel_ptr, mA_tma, mBg_tma, mBu_tma, mD, mR,
      sA_layout, sB_layout, mma_tiler, tiled_mma, cluster_shape, tma_atom_A, tma_atom_Bg, tma_atom_Bu);
  if (status != cutlass::Status::kSuccess) { std::cerr << "2CTA SwiGLU pipe launch failed\n"; std::exit(1); }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ============================================================================
// Serial 2CTA reference (copy of umma_swiglu_2cta.cu kernel) for A/B comparison.
// ============================================================================
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SerialSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> Bg;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> Bu;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;
  CUTE_DEVICE constexpr auto tensor_sA()  { return make_tensor(make_smem_ptr(A.begin()),  ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sBg() { return make_tensor(make_smem_ptr(Bg.begin()), BSmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sBu() { return make_tensor(make_smem_ptr(Bu.begin()), BSmemLayout{}); }
};

template <class SharedStorage,
          class ATensor, class BgTensor, class BuTensor, class DTensor, class RTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomBg, class TmaAtomBu>
__global__ static void
umma_swiglu_2cta_serial_device(ATensor mA, BgTensor mBg, BuTensor mBu, DTensor mD, RTensor mRoute,
                        MmaTiler_MNK mma_tiler, TiledMMA tiled_mma, ClusterShape_MNK cluster_shape,
                        CUTE_GRID_CONSTANT TmaAtomA  const tma_atom_A,
                        CUTE_GRID_CONSTANT TmaAtomBg const tma_atom_Bg,
                        CUTE_GRID_CONSTANT TmaAtomBu const tma_atom_Bu) {
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                                   blockIdx.x / size<0>(cluster_layout_vmnk),
                                   blockIdx.y, _);
  auto mma_coord = select<1,2,3>(mma_coord_vmnk);
  Tensor gA  = local_tile(mA,  mma_tiler, mma_coord, Step<_1, X,_1>{});
  Tensor gBg = local_tile(mBg, mma_tiler, mma_coord, Step< X,_1,_1>{});
  Tensor gBu = local_tile(mBu, mma_tiler, mma_coord, Step< X,_1,_1>{});
  Tensor gD  = local_tile(mD,  mma_tiler, mma_coord, Step<_1,_1, X>{});
  extern __shared__ char shared_memory[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor tCsA  = smem.tensor_sA();
  Tensor tCsBg = smem.tensor_sBg();
  Tensor tCsBu = smem.tensor_sBu();
  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA  = cta_mma.partition_A(gA);
  Tensor tCgBg = cta_mma.partition_B(gBg);
  Tensor tCgBu = cta_mma.partition_B(gBu);
  Tensor tCgD  = cta_mma.partition_C(gD);
  Tensor tCrA  = cta_mma.make_fragment_A(tCsA);
  Tensor tCrBg = cta_mma.make_fragment_B(tCsBg);
  Tensor tCrBu = cta_mma.make_fragment_B(tCsBu);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);
  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);
  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
  __syncthreads();
  tCtAcc.data() = smem.tmem_base_ptr;
  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};
  auto [tAgA,  tAsA]  = tma_partition(tma_atom_A,  get<2>(cta_in_cluster_coord_vmnk),
                                      make_layout(size<2>(cluster_layout_vmnk)),
                                      group_modes<0,3>(tCsA),  group_modes<0,3>(tCgA));
  auto [tBggBg, tBgsBg] = tma_partition(tma_atom_Bg, get<1>(cta_in_cluster_coord_vmnk),
                                        make_layout(size<1>(cluster_layout_vmnk)),
                                        group_modes<0,3>(tCsBg), group_modes<0,3>(tCgBg));
  auto [tBugBu, tBusBu] = tma_partition(tma_atom_Bu, get<1>(cta_in_cluster_coord_vmnk),
                                        make_layout(size<1>(cluster_layout_vmnk)),
                                        group_modes<0,3>(tCsBu), group_modes<0,3>(tCgBu));
  uint16_t mcast_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t mcast_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t mcast_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                     create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  int txbytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))
              + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBgsBg));
  if (elect_one_warp && elect_one_thr) {
    int np = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(smem.mma_barrier, np);
    cute::initialize_barrier(smem.tma_barrier, 1);
  }
  int mma_phase = 0, tma_phase = 0;
  cute::cluster_sync();
  TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r = t2r.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
  Tensor tDgD   = thr_t2r.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrG = make_tensor<AccType>(shape(tDgD));
  Tensor tDrU = make_tensor<AccType>(shape(tDgD));
  for (int pass = 0; pass < 2; ++pass) {
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
      if (elect_one_warp && elect_one_thr) {
        if (elect_one_cta) cute::set_barrier_transaction_bytes(smem.tma_barrier, txbytes);
        copy(tma_atom_A.with(smem.tma_barrier, mcast_a), tAgA(_,k_tile), tAsA);
        if (pass == 0) copy(tma_atom_Bg.with(smem.tma_barrier, mcast_b), tBggBg(_,k_tile), tBgsBg);
        else           copy(tma_atom_Bu.with(smem.tma_barrier, mcast_b), tBugBu(_,k_tile), tBusBu);
      }
      if (elect_one_cta) {
        cute::wait_barrier(smem.tma_barrier, tma_phase); tma_phase ^= 1;
        if (elect_one_warp) {
          auto& tCrB = (pass == 0) ? tCrBg : tCrBu;
          for (int kb = 0; kb < size<2>(tCrA); ++kb) {
            gemm(tiled_mma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive_multicast_2x1SM(&smem.mma_barrier, mcast_c);
        }
      }
      cute::wait_barrier(smem.mma_barrier, mma_phase); mma_phase ^= 1;
    }
    if (pass == 0) copy(t2r, tDtAcc, tDrG);
    else           copy(t2r, tDtAcc, tDrU);
    cute::cluster_sync();
  }
  Tensor cD  = make_identity_tensor(shape(mD));
  Tensor gcD = local_tile(cD, mma_tiler, mma_coord, Step<_1,_1, X>{});
  Tensor tCgcD = cta_mma.partition_C(gcD);
  Tensor tDcD  = thr_t2r.partition_D(tCgcD);
  using OutType = typename DTensor::value_type;
  Tensor tDrAct = make_tensor<OutType>(shape(tDgD));
  CUTE_UNROLL
  for (int i = 0; i < size(tDrG); ++i) {
    int m = get<0>(tDcD(i));
    float g = static_cast<float>(tDrG(i));
    float u = static_cast<float>(tDrU(i));
    float silu_g = g * (1.0f / (1.0f + ::expf(-g)));
    tDrAct(i) = static_cast<OutType>(silu_g * u * static_cast<float>(mRoute(m)));
  }
  copy(tDrAct, tDgD);
  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeAB>
void launch_umma_swiglu_2cta_serial(TypeAB const* dA, TypeAB const* dWg, TypeAB const* dWu,
                                    TypeAB* dAct, float const* dRoute,
                                    int M, int N, int K, cudaStream_t stream) {
  using TypeC = float;
  Layout layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  Layout layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  Layout layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  Layout layout_R = make_layout(make_shape(M), make_stride(Int<1>{}));
  Tensor mA  = make_tensor(make_gmem_ptr(dA),  layout_A);
  Tensor mBg = make_tensor(make_gmem_ptr(dWg), layout_B);
  Tensor mBu = make_tensor(make_gmem_ptr(dWu), layout_B);
  Tensor mD  = make_tensor(make_gmem_ptr(dAct), layout_D);
  Tensor mR  = make_tensor(make_gmem_ptr(dRoute), layout_R);
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TypeAB, TypeAB, TypeC, 256, 256,
                                 UMMA::Major::K, UMMA::Major::K>{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeAB>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeAB>{}, mma_shape_B);
  using SMEMStorage = SerialSharedStorage<TypeAB, TypeAB, decltype(sA_layout), decltype(sB_layout)>;
  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  Copy_Atom tma_atom_A  = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA,  sA_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_atom_Bg = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mBg, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_atom_Bu = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mBu, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mA_tma  = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor mBg_tma = tma_atom_Bg.get_tma_tensor(shape(mBg));
  Tensor mBu_tma = tma_atom_Bu.get_tma_tensor(shape(mBu));
  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(size(ceil_div(M, bM * size<1>(cluster_layout_vmnk))) * dimCluster.x,
               size(ceil_div(N, bN * size<2>(cluster_layout_vmnk))) * dimCluster.y);
  int smemBytes = sizeof(SMEMStorage);
  auto* kernel_ptr = &umma_swiglu_2cta_serial_device<SMEMStorage,
                          decltype(mA_tma), decltype(mBg_tma), decltype(mBu_tma), decltype(mD), decltype(mR),
                          decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                          decltype(tma_atom_A), decltype(tma_atom_Bg), decltype(tma_atom_Bu)>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*) kernel_ptr, mA_tma, mBg_tma, mBu_tma, mD, mR,
      mma_tiler, tiled_mma, cluster_shape, tma_atom_A, tma_atom_Bg, tma_atom_Bu);
  if (status != cutlass::Status::kSuccess) { std::cerr << "2CTA SwiGLU serial launch failed\n"; std::exit(1); }
}
#endif

// ============================================================================
static float relative_error(const std::vector<float>& t,const std::vector<float>& r){
  double num=0,den=0; for(size_t i=0;i<r.size();++i){double d=t[i]-r[i];num+=d*d;den+=double(r[i])*r[i];}
  return float(std::sqrt(num)/(std::sqrt(den)+1e-12));
}
template<class F> float time_kernel(F fn,int warm,int it,cudaStream_t s){
  for(int i=0;i<warm;++i) fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  cudaEvent_t b,e; CHECK_CUDA(cudaEventCreate(&b)); CHECK_CUDA(cudaEventCreate(&e));
  std::vector<float> ts(it);
  for(int i=0;i<it;++i){CHECK_CUDA(cudaEventRecord(b,s)); fn(); CHECK_CUDA(cudaEventRecord(e,s));
    CHECK_CUDA(cudaEventSynchronize(e)); CHECK_CUDA(cudaEventElapsedTime(&ts[i],b,e));}
  std::sort(ts.begin(),ts.end()); cudaEventDestroy(b); cudaEventDestroy(e); return ts[it/2];
}

int main(int argc,char**argv){
  int M=256,K=4096,N=4096;
  if(argc>=2)M=atoi(argv[1]); if(argc>=3)K=atoi(argv[2]); if(argc>=4)N=atoi(argv[3]);
  std::cout<<"2CTA fused SwiGLU PIPE vs SERIAL  M="<<M<<" K="<<K<<" N="<<N<<"\n";
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout<<"SM100 not supported in build.\n"; return 0;
#else
  std::mt19937 gen(1234); std::uniform_real_distribution<float> dist(-1.f,1.f);
  auto bf=[](float v){return __bfloat162float(__float2bfloat16(v));};
  std::vector<float> hA(M*K),hWg(N*K),hWu(N*K),hR(M);
  for(auto&x:hA)x=dist(gen); for(auto&x:hWg)x=dist(gen); for(auto&x:hWu)x=dist(gen);
  for(auto&x:hR)x=0.5f+0.5f*dist(gen);

  std::vector<float> Ab(M*K),Wgb(N*K),Wub(N*K);
  for(int i=0;i<M*K;++i)Ab[i]=bf(hA[i]);
  for(int i=0;i<N*K;++i){Wgb[i]=bf(hWg[i]);Wub[i]=bf(hWu[i]);}
  std::vector<float> ref(M*N);
  for(int m=0;m<M;++m)for(int n=0;n<N;++n){
    double g=0,u=0; for(int k=0;k<K;++k){g+=double(Ab[m*K+k])*Wgb[n*K+k];u+=double(Ab[m*K+k])*Wub[n*K+k];}
    float gf=float(g),uf=float(u),silu=gf*(1.f/(1.f+std::exp(-gf)));
    ref[m*N+n]=bf(silu*uf*hR[m]);
  }

  std::vector<__nv_bfloat16> A16(M*K),Wg16(N*K),Wu16(N*K);
  for(int i=0;i<M*K;++i)A16[i]=__float2bfloat16(hA[i]);
  for(int i=0;i<N*K;++i){Wg16[i]=__float2bfloat16(hWg[i]);Wu16[i]=__float2bfloat16(hWu[i]);}
  __nv_bfloat16 *dA,*dWg,*dWu,*dActP,*dActS; float* dR;
  CHECK_CUDA(cudaMalloc(&dA,M*K*2)); CHECK_CUDA(cudaMalloc(&dWg,N*K*2)); CHECK_CUDA(cudaMalloc(&dWu,N*K*2));
  CHECK_CUDA(cudaMalloc(&dActP,M*N*2)); CHECK_CUDA(cudaMalloc(&dActS,M*N*2)); CHECK_CUDA(cudaMalloc(&dR,M*4));
  CHECK_CUDA(cudaMemcpy(dA,A16.data(),M*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWg,Wg16.data(),N*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWu,Wu16.data(),N*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dR,hR.data(),M*4,cudaMemcpyHostToDevice));

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  using BF=cutlass::bfloat16_t;
  auto pipe_fn=[&]{ launch_umma_swiglu_2cta_pipe(reinterpret_cast<const BF*>(dA),reinterpret_cast<const BF*>(dWg),
                      reinterpret_cast<const BF*>(dWu),reinterpret_cast<BF*>(dActP),dR,M,N,K,s); };
  auto serial_fn=[&]{ launch_umma_swiglu_2cta_serial(reinterpret_cast<const BF*>(dA),reinterpret_cast<const BF*>(dWg),
                      reinterpret_cast<const BF*>(dWu),reinterpret_cast<BF*>(dActS),dR,M,N,K,s); };

  pipe_fn(); serial_fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  std::vector<__nv_bfloat16> hP(M*N),hS(M*N);
  CHECK_CUDA(cudaMemcpy(hP.data(),dActP,M*N*2,cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hS.data(),dActS,M*N*2,cudaMemcpyDeviceToHost));
  std::vector<float> fP(M*N),fS(M*N);
  for(int i=0;i<M*N;++i){fP[i]=__bfloat162float(hP[i]);fS[i]=__bfloat162float(hS[i]);}
  std::cout<<"Rel err vs fused-SwiGLU ref:  PIPE="<<relative_error(fP,ref)
           <<"   SERIAL="<<relative_error(fS,ref)<<"\n";
  std::cout<<"Rel err PIPE vs SERIAL: "<<relative_error(fP,fS)<<"\n";
  bool ok=(relative_error(fP,ref)<5e-2f)&&(relative_error(fS,ref)<5e-2f);
  std::cout<<"Correctness: "<<(ok?"PASS":"FAIL")<<"\n";

  const int warm=20,it=100;
  float tp=time_kernel(pipe_fn,warm,it,s), tser=time_kernel(serial_fn,warm,it,s);
  double flops=2.0*2*M*N*K;  // gate+up two GEMMs
  std::cout<<"PIPE:   "<<tp<<" ms  "<<(flops/(tp*1e-3)/1e12)<<" TFLOPS\n";
  std::cout<<"SERIAL: "<<tser<<" ms  "<<(flops/(tser*1e-3)/1e12)<<" TFLOPS\n";
  std::cout<<"Speedup (SERIAL/PIPE time): "<<(tser/tp)<<"x\n";

  cudaFree(dA);cudaFree(dWg);cudaFree(dWu);cudaFree(dActP);cudaFree(dActS);cudaFree(dR);
  cudaStreamDestroy(s);
  return 0;
#endif
}
