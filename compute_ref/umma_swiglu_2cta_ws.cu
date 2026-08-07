// umma_swiglu_2cta_ws.cu — Stage (3'-ws): 2CTA (2x1SM) UMMA fused gate+up GEMM
// with in-register SwiGLU epilogue, M=256, using WARP-SPECIALIZED software
// pipeline via CUTLASS PipelineTmaUmmaAsync.
//
// Difference vs the serial umma_swiglu_2cta.cu and the failed hand-written
// pipe: here the TMA producer and the UMMA consumer run on SEPARATE warps, so
// the producer can run ahead filling pipeline stages while the consumer issues
// MMAs — this is the configuration the CUTLASS pipeline class is designed for
// and the only way to actually overlap TMA with tcgen05 MMA.
//
// Warp roles (within the 128-thread / 4-warp UMMA block):
//   warp 0  -> TMA producer (issues A + Wg/Wu multicast loads)
//   warp 1  -> UMMA consumer (issues tcgen05 gemm, releases stages)
//   warp 0..3 (all 128 threads) -> epilogue (TMEM->reg->SwiGLU->GMEM)
//
//   act[m,n] = silu(A@Wg^T) * (A@Wu^T) * route_w[m]      (M=256)
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -arch=sm_103a -O3 \
//        -I../cutlass_ref/include -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        umma_swiglu_2cta_ws.cu -o umma_swiglu_2cta_ws
//   ./umma_swiglu_2cta_ws            # default M=256 K=4096 N=4096

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>

using namespace cute;

#define CHECK_CUDA(call)                                                         \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                               \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":"      \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

static constexpr int kStages = 2;

// PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShape_MNK>.
// CRITICAL: the THIRD template arg (AtomThrShape_MNK) drives is_2sm_mma. For a
// 2x1SM MMA it MUST be Shape<_2,_1,_1>; otherwise the pipeline defaults it to
// Shape<_1,_1,_1>, is_2sm_mma=false, and producer_commit/consumer_release emit
// tcgen05 cta_group::1 ops that conflict with the cta_group::2 MMA (ptxas error
// "uses single CTA(.cta_group::1) and CTA pair granularity(.cta_group::2)").
using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
    kStages,
    cute::Shape<cute::_2,cute::_1,cute::_1>,   // ClusterShape
    cute::Shape<cute::_2,cute::_1,cute::_1>>;  // AtomThrShape_MNK (2SM!)
using PipelineState    = typename MainloopPipeline::PipelineState;

// SMEM: A/Bg/Bu ring buffers (kStages) + the pipeline's barrier storage +
// tmem base pointer. The pipeline owns its full/empty barriers internally.
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout> * kStages> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout> * kStages> Bg;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout> * kStages> Bu;
  alignas(16)  typename MainloopPipeline::SharedStorage pipeline;
  alignas(16)  cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA(int s, ASmemLayout l)  { return make_tensor(make_smem_ptr(A.begin()  + cute::cosize_v<ASmemLayout> * s), l); }
  CUTE_DEVICE auto tensor_sBg(int s, BSmemLayout l) { return make_tensor(make_smem_ptr(Bg.begin() + cute::cosize_v<BSmemLayout> * s), l); }
  CUTE_DEVICE auto tensor_sBu(int s, BSmemLayout l) { return make_tensor(make_smem_ptr(Bu.begin() + cute::cosize_v<BSmemLayout> * s), l); }
};

template <class SharedStorage, class ASmemLayout, class BSmemLayout,
          class ATensor, class BgTensor, class BuTensor, class DTensor, class RTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomBg, class TmaAtomBu>
__global__ static void
umma_swiglu_2cta_ws_device(ATensor mA, BgTensor mBg, BuTensor mBu, DTensor mD, RTensor mRoute,
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
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);

  // Per-stage SMEM views + MMA fragments.
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

  int warp_idx = cutlass::canonical_warp_idx_sync();
  uint32_t lane_predicate = cute::elect_one_sync();
  const bool active = (threadIdx.x < 128);

  auto cta_in_cluster = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto elect_one_cta = get<0>(cta_in_cluster) == Int<0>{};

  // ---- Pipeline params ----
  // Producer = warp 0 (TMA), Consumer = warp 1 (UMMA). is_leader: the producer
  // thread that issues expect_tx, only on the leader CTA's producer warp lane0.
  typename MainloopPipeline::Params params;
  if (warp_idx == 0) params.role = MainloopPipeline::ThreadCategory::Producer;
  else               params.role = MainloopPipeline::ThreadCategory::Consumer;
  // transaction bytes: one K-tile of A + one K-tile of B (both CTAs' share via mcast).
  // computed below from tma partition; set after we know txbytes.

  // TMA partitions per stage.
  auto tma_part_A = [&](int s) {
    auto sAv = (s == 0) ? tCsA0 : tCsA1;
    return tma_partition(tma_atom_A, get<2>(cta_in_cluster),
                         make_layout(size<2>(cluster_layout_vmnk)),
                         group_modes<0,3>(sAv), group_modes<0,3>(tCgA));
  };
  auto tma_part_Bg = [&](int s) {
    auto sBv = (s == 0) ? tCsBg0 : tCsBg1;
    return tma_partition(tma_atom_Bg, get<1>(cta_in_cluster),
                         make_layout(size<1>(cluster_layout_vmnk)),
                         group_modes<0,3>(sBv), group_modes<0,3>(tCgBg));
  };
  auto tma_part_Bu = [&](int s) {
    auto sBv = (s == 0) ? tCsBu0 : tCsBu1;
    return tma_partition(tma_atom_Bu, get<1>(cta_in_cluster),
                         make_layout(size<1>(cluster_layout_vmnk)),
                         group_modes<0,3>(sBv), group_modes<0,3>(tCgBu));
  };

  uint16_t mcast_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster);
  uint16_t mcast_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster);

  auto [tAgA0_, tAsA0_] = tma_part_A(0);
  auto [tBg0_, tBgs0_]  = tma_part_Bg(0);
  uint32_t txbytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA0_))
                   + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBgs0_));
  params.is_leader = (warp_idx == 0) && lane_predicate && elect_one_cta;
  params.num_consumers = 32;          // one consumer warp
  params.transaction_bytes = txbytes;
  params.initializing_warp = 0;

  // Construct the pipeline (all 128 active threads participate; init_barriers
  // gates to initializing_warp internally; fence_barrier_init synchronizes).
  MainloopPipeline pipeline(smem.pipeline, params, cluster_shape);

  // TMEM allocate (consumer warp 1 owns it; mirror official: MMA warp allocates).
  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (warp_idx == 1 && lane_predicate) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
  }
  // Named barrier so all 128 UMMA threads see the tmem allocation before use.
  cutlass::arch::NamedBarrier(128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier).arrive_and_wait();
  cute::cluster_sync();   // cross-CTA: barriers + tmem visible cluster-wide
  tCtAcc.data() = smem.tmem_base_ptr;

  const int nK = size<3>(tCgA);

  // ====================== PASS LOOP (gate, up) ======================
  // Single TMEM accumulator reused per pass. Each pass runs the full warp-
  // specialized pipeline over nK K-tiles. tDrG / tDrU hold drained results.
  TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r = t2r.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
  Tensor tDgD   = thr_t2r.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrG = make_tensor<AccType>(shape(tDgD));
  Tensor tDrU = make_tensor<AccType>(shape(tDgD));

  for (int pass = 0; pass < 2; ++pass) {
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    if (warp_idx == 0) {
      // ---------- PRODUCER WARP: issue TMA loads for all K-tiles ----------
      PipelineState prod = cutlass::make_producer_start_state<MainloopPipeline>();
      auto tok = pipeline.producer_try_acquire(prod);
      for (int k = 0; k < nK; ++k) {
        pipeline.producer_acquire(prod, tok);
        int s = prod.index();
        if (lane_predicate) {
          auto* bar = pipeline.producer_get_barrier(prod);
          auto [tAgA, tAsA] = tma_part_A(s);
          copy(tma_atom_A.with(*bar, mcast_a), tAgA(_,k), tAsA);
          if (pass == 0) {
            auto [tBg, tBgs] = tma_part_Bg(s);
            copy(tma_atom_Bg.with(*bar, mcast_b), tBg(_,k), tBgs);
          } else {
            auto [tBu, tBus] = tma_part_Bu(s);
            copy(tma_atom_Bu.with(*bar, mcast_b), tBu(_,k), tBus);
          }
        }
        // TMA-based producer_commit is a NOP (the TMA hardware arrives on FULL
        // via the expect_tx set inside producer_acquire).
        pipeline.producer_commit(prod, txbytes);
        ++prod;
        tok = pipeline.producer_try_acquire(prod);
      }
      // Keep producer alive until consumer drained all stages.
      pipeline.producer_tail(prod);
    } else if (warp_idx == 1) {
      // ---------- CONSUMER WARP: wait stage, gemm, release ----------
      PipelineState cons;   // default: index 0, phase 0
      auto tok = pipeline.consumer_try_wait(cons);
      for (int k = 0; k < nK; ++k) {
        pipeline.consumer_wait(cons, tok);
        int s = cons.index();
        if (lane_predicate) {
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
        }
        // consumer_release ties the MMA completion to the EMPTY barrier via
        // umma_arrive_multicast_2x1SM (cluster-aware).
        pipeline.consumer_release(cons);
        ++cons;
        tok = pipeline.consumer_try_wait(cons);
      }
    }

    // All UMMA warps converge before draining the accumulator. The MMA result
    // for this pass is now committed to TMEM.
    cutlass::arch::NamedBarrier(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier).arrive_and_wait();

    // Drain TMEM -> reg (all 128 threads participate in t2r copy).
    if (active) {
      if (pass == 0) copy(t2r, tDtAcc, tDrG);
      else           copy(t2r, tDtAcc, tDrU);
    }
    cute::cluster_sync();
  }

  // ====================== SwiGLU EPILOGUE (all 128 threads) ======================
  if (active) {
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
  }

  cutlass::arch::NamedBarrier(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier).arrive_and_wait();
  if (warp_idx == 1 && lane_predicate) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
  cute::cluster_sync();
}

template <class TypeAB>
void launch_umma_swiglu_2cta_ws(TypeAB const* dA, TypeAB const* dWg, TypeAB const* dWu,
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

  auto* kernel_ptr = &umma_swiglu_2cta_ws_device<SMEMStorage,
                          decltype(sA_layout), decltype(sB_layout),
                          decltype(mA_tma), decltype(mBg_tma), decltype(mBu_tma), decltype(mD), decltype(mR),
                          decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                          decltype(tma_atom_A), decltype(tma_atom_Bg), decltype(tma_atom_Bu)>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams lp = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      lp, (void const*) kernel_ptr, mA_tma, mBg_tma, mBu_tma, mD, mR,
      sA_layout, sB_layout, mma_tiler, tiled_mma, cluster_shape, tma_atom_A, tma_atom_Bg, tma_atom_Bu);
  if (status != cutlass::Status::kSuccess) { std::cerr << "2CTA WS launch failed\n"; std::exit(1); }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

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
  std::cout<<"2CTA fused SwiGLU WARP-SPECIALIZED  M="<<M<<" K="<<K<<" N="<<N<<"\n";
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
  __nv_bfloat16 *dA,*dWg,*dWu,*dAct; float* dR;
  CHECK_CUDA(cudaMalloc(&dA,M*K*2)); CHECK_CUDA(cudaMalloc(&dWg,N*K*2)); CHECK_CUDA(cudaMalloc(&dWu,N*K*2));
  CHECK_CUDA(cudaMalloc(&dAct,M*N*2)); CHECK_CUDA(cudaMalloc(&dR,M*4));
  CHECK_CUDA(cudaMemcpy(dA,A16.data(),M*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWg,Wg16.data(),N*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWu,Wu16.data(),N*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dR,hR.data(),M*4,cudaMemcpyHostToDevice));

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  using BF=cutlass::bfloat16_t;
  auto ws_fn=[&]{ launch_umma_swiglu_2cta_ws(reinterpret_cast<const BF*>(dA),reinterpret_cast<const BF*>(dWg),
                      reinterpret_cast<const BF*>(dWu),reinterpret_cast<BF*>(dAct),dR,M,N,K,s); };

  ws_fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  std::vector<__nv_bfloat16> hO(M*N);
  CHECK_CUDA(cudaMemcpy(hO.data(),dAct,M*N*2,cudaMemcpyDeviceToHost));
  std::vector<float> fO(M*N);
  for(int i=0;i<M*N;++i)fO[i]=__bfloat162float(hO[i]);
  float rel = relative_error(fO, ref);
  std::cout<<"Rel err vs fused-SwiGLU ref: "<<rel<<"\n";
  std::cout<<"Correctness: "<<((rel<5e-2f)?"PASS":"FAIL")<<"\n";

  const int warm=20,it=100;
  float tw=time_kernel(ws_fn,warm,it,s);
  double flops=2.0*2*M*N*K;
  std::cout<<"WS: "<<tw<<" ms  "<<(flops/(tw*1e-3)/1e12)<<" TFLOPS\n";

  cudaFree(dA);cudaFree(dWg);cudaFree(dWu);cudaFree(dAct);cudaFree(dR);
  cudaStreamDestroy(s);
  return 0;
#endif
}
