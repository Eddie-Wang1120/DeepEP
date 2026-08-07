// umma_swiglu_2cta.cu — Stage (3'): 2CTA (2x1SM) UMMA fused gate+up GEMM with
// in-register SwiGLU epilogue, M_tile=256. Combines:
//   - tutorial 04's 2SM tcgen05.mma + 2SM multicast TMA (weight multicast)
//   - stage 2b's fused gate/up two-pass + SwiGLU epilogue
//
//   act[m,n] = silu(A@Wg^T) * (A@Wu^T) * route_w[m]      (M=256)
//
// Correctness: vs host fused-SwiGLU reference (megakernel WMMA fused math).
// Performance: 2CTA fused-SwiGLU vs a WMMA fused-SwiGLU baseline (megakernel
//   device_gemm_swiglu_fused style: 32 blocks x 25 warps, M=256).
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -arch=sm_103a -O3 \
//        -I../cutlass_ref/include -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        umma_swiglu_2cta.cu -o umma_swiglu_2cta
//   ./umma_swiglu_2cta            # default M=256 K=4096 N=4096

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

// A shared; B has two buffers (Wgate / Wup), each loaded per-pass via multicast TMA.
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
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
umma_swiglu_2cta_device(ATensor mA, BgTensor mBg, BuTensor mBu, DTensor mD, RTensor mRoute,
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
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);   // single TMEM acc, reused per pass

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
    cute::initialize_barrier(smem.mma_barrier, /* num_ctas */ np);
    cute::initialize_barrier(smem.tma_barrier, /* num_threads */ 1);
  }
  int mma_phase = 0, tma_phase = 0;
  cute::cluster_sync();

  // Drain helper bound to acc
  TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r = t2r.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
  Tensor tDgD   = thr_t2r.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrG = make_tensor<AccType>(shape(tDgD));
  Tensor tDrU = make_tensor<AccType>(shape(tDgD));

  // Two passes: pass 0 = gate (Wg), pass 1 = up (Wu). Single TMEM acc reused.
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
void launch_umma_swiglu_2cta(TypeAB const* dA, TypeAB const* dWg, TypeAB const* dWu,
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

  auto* kernel_ptr = &umma_swiglu_2cta_device<SMEMStorage,
                          decltype(mA_tma), decltype(mBg_tma), decltype(mBu_tma), decltype(mD), decltype(mR),
                          decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                          decltype(tma_atom_A), decltype(tma_atom_Bg), decltype(tma_atom_Bu)>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*) kernel_ptr, mA_tma, mBg_tma, mBu_tma, mD, mR,
      mma_tiler, tiled_mma, cluster_shape, tma_atom_A, tma_atom_Bg, tma_atom_Bu);
  if (status != cutlass::Status::kSuccess) { std::cerr << "2CTA SwiGLU launch failed\n"; std::exit(1); }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ============================================================================
// WMMA fused-SwiGLU baseline (megakernel device_gemm_swiglu_fused style).
// 32 blocks x 25 warps; each warp strides over 16x16 N-tiles of gate&up,
// in-register SwiGLU, write act. Single GEMM (gate+up share A-tile load).
// ============================================================================
constexpr int WM=16, WN=16, WK=16;
__global__ void wmma_swiglu_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ Wg,
                                   const __nv_bfloat16* __restrict__ Wu,
                                   const float* __restrict__ route,
                                   __nv_bfloat16* __restrict__ act,
                                   int M, int K, int N) {
  using namespace nvcuda;
  extern __shared__ float smem[];
  const int warp = (blockIdx.x*blockDim.x + threadIdx.x)/32;
  const int nwarp = (gridDim.x*blockDim.x)/32;
  const int sw = threadIdx.x/32;
  const int tiles_m=(M+WM-1)/WM, tiles_n=(N+WN-1)/WN, total=tiles_m*tiles_n;
  for (int t=warp; t<total; t+=nwarp) {
    int ro=(t/tiles_n)*WM, co=(t%tiles_n)*WN;
    if (ro>=M||co>=N) continue;
    wmma::fragment<wmma::matrix_a,WM,WN,WK,__nv_bfloat16,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,WM,WN,WK,__nv_bfloat16,wmma::col_major> bg,bu;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> cg,cu;
    wmma::fill_fragment(cg,0.f); wmma::fill_fragment(cu,0.f);
    for (int k=0;k<K;k+=WK){
      wmma::load_matrix_sync(a, A+ro*K+k, K);
      wmma::load_matrix_sync(bg, Wg+co*K+k, K);
      wmma::load_matrix_sync(bu, Wu+co*K+k, K);
      wmma::mma_sync(cg,a,bg,cg);
      wmma::mma_sync(cu,a,bu,cu);
    }
    float* gb=smem + sw*2*WM*WN; float* ub=gb+WM*WN;
    wmma::store_matrix_sync(gb,cg,WN,wmma::mem_row_major);
    wmma::store_matrix_sync(ub,cu,WN,wmma::mem_row_major);
    __syncwarp();
    int lane=threadIdx.x%32;
    for (int i=lane;i<WM*WN;i+=32){
      int r=ro+i/WN, c=co+i%WN;
      if (r>=M||c>=N) continue;
      float g=gb[i], u=ub[i];
      float silu=g*(1.f/(1.f+__expf(-g)));
      act[r*N+c]=__float2bfloat16(silu*u*route[r]);
    }
  }
}
void launch_wmma_swiglu(const __nv_bfloat16* dA,const __nv_bfloat16* dWg,const __nv_bfloat16* dWu,
                        const float* dRoute,__nv_bfloat16* dAct,int M,int N,int K,cudaStream_t s){
  const int threads=800, blocks=32, smem=(threads/32)*2*WM*WN*sizeof(float);
  CHECK_CUDA(cudaFuncSetAttribute(wmma_swiglu_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  wmma_swiglu_kernel<<<blocks,threads,smem,s>>>(dA,dWg,dWu,dRoute,dAct,M,K,N);
}

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
  std::cout<<"2CTA fused SwiGLU  M="<<M<<" K="<<K<<" N="<<N<<"\n";
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
  __nv_bfloat16 *dA,*dWg,*dWu,*dActU,*dActW; float* dR;
  CHECK_CUDA(cudaMalloc(&dA,M*K*2)); CHECK_CUDA(cudaMalloc(&dWg,N*K*2)); CHECK_CUDA(cudaMalloc(&dWu,N*K*2));
  CHECK_CUDA(cudaMalloc(&dActU,M*N*2)); CHECK_CUDA(cudaMalloc(&dActW,M*N*2)); CHECK_CUDA(cudaMalloc(&dR,M*4));
  CHECK_CUDA(cudaMemcpy(dA,A16.data(),M*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWg,Wg16.data(),N*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWu,Wu16.data(),N*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dR,hR.data(),M*4,cudaMemcpyHostToDevice));

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  using BF=cutlass::bfloat16_t;
  auto umma_fn=[&]{ launch_umma_swiglu_2cta(reinterpret_cast<const BF*>(dA),reinterpret_cast<const BF*>(dWg),
                      reinterpret_cast<const BF*>(dWu),reinterpret_cast<BF*>(dActU),dR,M,N,K,s); };
  auto wmma_fn=[&]{ launch_wmma_swiglu(dA,dWg,dWu,dR,dActW,M,N,K,s); };

  umma_fn(); wmma_fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  std::vector<__nv_bfloat16> hU(M*N),hW(M*N);
  CHECK_CUDA(cudaMemcpy(hU.data(),dActU,M*N*2,cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hW.data(),dActW,M*N*2,cudaMemcpyDeviceToHost));
  std::vector<float> fU(M*N),fW(M*N);
  for(int i=0;i<M*N;++i){fU[i]=__bfloat162float(hU[i]);fW[i]=__bfloat162float(hW[i]);}
  std::cout<<"Rel err vs fused-SwiGLU ref:  2CTA-UMMA="<<relative_error(fU,ref)
           <<"   WMMA="<<relative_error(fW,ref)<<"\n";
  bool ok=(relative_error(fU,ref)<5e-2f)&&(relative_error(fW,ref)<5e-2f);
  std::cout<<"Correctness: "<<(ok?"PASS":"FAIL")<<"\n";

  const int warm=20,it=100;
  float tu=time_kernel(umma_fn,warm,it,s), tw=time_kernel(wmma_fn,warm,it,s);
  double flops=2.0*2*M*N*K;  // gate+up two GEMMs
  std::cout<<"2CTA-UMMA: "<<tu<<" ms  "<<(flops/(tu*1e-3)/1e12)<<" TFLOPS\n";
  std::cout<<"WMMA:      "<<tw<<" ms  "<<(flops/(tw*1e-3)/1e12)<<" TFLOPS\n";
  std::cout<<"Speedup (WMMA/UMMA time): "<<(tw/tu)<<"x\n";

  cudaFree(dA);cudaFree(dWg);cudaFree(dWu);cudaFree(dActU);cudaFree(dActW);cudaFree(dR);
  cudaStreamDestroy(s);
  return 0;
#endif
}
