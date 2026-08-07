// umma_gemm_2cta.cu — Stage (3): standalone 2CTA (2x1SM) Blackwell UMMA BF16 GEMM.
//
// Pure GEMM D = A @ B^T (no SwiGLU yet). Adapted from cutlass tutorial
// 04_mma_tma_2sm_sm100.cu: 2SM tcgen05.mma + 2SM multicast TMA. M_tile=256 so a
// megakernel batch of 256 tokens fills it without padding (user confirmed
// COMPUTE_BATCH_SIZE can be raised to 256).
//
// cluster = (2,1,1): a pair of CTAs cooperate on one M=256 tile; B (weight) tile
// is multicast-shared across the pair -> the weight-IO saving we care about.
//
// Shape (megakernel real config, M raised to 256): M=256 K=4096 N=4096.
//   A:[M,K] K-major, B:[N,K] K-major (A@B^T), D:[M,N] N-major. BF16/FP32acc/BF16out.
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -arch=sm_103a -O3 \
//        -I../cutlass_ref/include -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        umma_gemm_2cta.cu -o umma_gemm_2cta
//   ./umma_gemm_2cta            # default M=256 K=4096 N=4096

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

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

using namespace cute;

#define CHECK_CUDA(call)                                                         \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                               \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":"      \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};

template <class SharedStorage,
          class ATensor, class BTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB>
__global__ static void
gemm_2cta_device(ATensor mA, BTensor mB, DTensor mD,
                 MmaTiler_MNK mma_tiler, TiledMMA tiled_mma, ClusterShape_MNK cluster_shape,
                 CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                 CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                                   blockIdx.x / size<0>(cluster_layout_vmnk),
                                   blockIdx.y, _);
  auto mma_coord = select<1,2,3>(mma_coord_vmnk);
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X,_1,_1>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor tCsA = smem.tensor_sA();
  Tensor tCsB = smem.tensor_sB();

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgD = cta_mma.partition_C(gD);
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);

  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = smem.tmem_base_ptr;

  // 2SM multicast TMA partitioning
  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};

  auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                    get<2>(cta_in_cluster_coord_vmnk),
                                    make_layout(size<2>(cluster_layout_vmnk)),
                                    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
  auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                    get<1>(cta_in_cluster_coord_vmnk),
                                    make_layout(size<1>(cluster_layout_vmnk)),
                                    group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));

  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                              create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

  int tma_transaction_bytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))
                            + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));

  if (elect_one_warp && elect_one_thr) {
    int num_mcast_participants = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(smem.mma_barrier, /* num_ctas */ num_mcast_participants);
    cute::initialize_barrier(smem.tma_barrier, /* num_threads */ 1);
  }
  int mma_phase = 0, tma_phase = 0;
  cute::cluster_sync();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    if (elect_one_warp && elect_one_thr) {
      if (elect_one_cta)
        cute::set_barrier_transaction_bytes(smem.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_A.with(smem.tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile), tAsA);
      copy(tma_atom_B.with(smem.tma_barrier, tma_mcast_mask_b), tBgB(_,k_tile), tBsB);
    }
    if (elect_one_cta) {
      cute::wait_barrier(smem.tma_barrier, tma_phase);
      tma_phase ^= 1;
      if (elect_one_warp) {
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
          gemm(tiled_mma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&smem.mma_barrier, mma_mcast_mask_c);
      }
    }
    cute::wait_barrier(smem.mma_barrier, mma_phase);
    mma_phase ^= 1;
  }

  // Epilogue: TMEM -> RMEM -> GMEM (cast FP32->BF16)
  TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r = t2r.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
  Tensor tDgD   = thr_t2r.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));
  copy(t2r, tDtAcc, tDrAcc);

  using OutType = typename DTensor::value_type;
  Tensor tDrD = make_tensor<OutType>(shape(tDgD));
  CUTE_UNROLL
  for (int i = 0; i < size(tDrAcc); ++i) tDrD(i) = static_cast<OutType>(tDrAcc(i));
  copy(tDrD, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeAB>
void launch_gemm_2cta(TypeAB const* dA, TypeAB const* dB, TypeAB* dD,
                      int M, int N, int K, cudaStream_t stream) {
  using TypeC = float;
  Layout layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  Layout layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  Layout layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  Tensor mA = make_tensor(make_gmem_ptr(dA), layout_A);
  Tensor mB = make_tensor(make_gmem_ptr(dB), layout_B);
  Tensor mD = make_tensor(make_gmem_ptr(dD), layout_D);

  // 2x1SM atom: M=256, N=256 tile.
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

  // cluster = (2,1,1): 2 CTAs cooperate on one M=256 tile.
  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  Copy_Atom tma_atom_A = make_tma_atom_A_sm100(
      SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));
  Copy_Atom tma_atom_B = make_tma_atom_B_sm100(
      SM100_TMA_2SM_LOAD_MULTICAST{}, mB, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(size(ceil_div(M, bM * size<1>(cluster_layout_vmnk))) * dimCluster.x,
               size(ceil_div(N, bN * size<2>(cluster_layout_vmnk))) * dimCluster.y);
  int smemBytes = sizeof(SMEMStorage);

  auto* kernel_ptr = &gemm_2cta_device<SMEMStorage,
                                        decltype(mA_tma), decltype(mB_tma), decltype(mD),
                                        decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                                        decltype(tma_atom_A), decltype(tma_atom_B)>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*) kernel_ptr, mA_tma, mB_tma, mD,
      mma_tiler, tiled_mma, cluster_shape, tma_atom_A, tma_atom_B);
  if (status != cutlass::Status::kSuccess) { std::cerr << "2CTA launch failed\n"; std::exit(1); }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

static float relative_error(const std::vector<float>& t, const std::vector<float>& r) {
  double num=0, den=0;
  for (size_t i=0;i<r.size();++i){ double d=t[i]-r[i]; num+=d*d; den+=double(r[i])*r[i]; }
  return float(std::sqrt(num)/(std::sqrt(den)+1e-12));
}

int main(int argc, char** argv) {
  int M=256, K=4096, N=4096;
  if (argc>=2) M=atoi(argv[1]);
  if (argc>=3) K=atoi(argv[2]);
  if (argc>=4) N=atoi(argv[3]);
  std::cout << "2CTA GEMM D = A@B^T  M="<<M<<" K="<<K<<" N="<<N<<" (BF16/FP32acc/BF16out)\n";
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n"; return 0;
#else
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-1.f,1.f);
  auto bf=[](float v){ return __bfloat162float(__float2bfloat16(v)); };

  std::vector<float> hA(M*K), hB(N*K);
  for (auto& x:hA) x=dist(gen);
  for (auto& x:hB) x=dist(gen);
  std::vector<float> hAb(M*K), hBb(N*K);
  for (int i=0;i<M*K;++i) hAb[i]=bf(hA[i]);
  for (int i=0;i<N*K;++i) hBb[i]=bf(hB[i]);
  std::vector<float> hRef(M*N);
  for (int m=0;m<M;++m) for (int n=0;n<N;++n){
    double acc=0; for (int k=0;k<K;++k) acc+=double(hAb[m*K+k])*hBb[n*K+k];
    hRef[m*N+n]=float(acc);
  }

  std::vector<__nv_bfloat16> hA16(M*K), hB16(N*K);
  for (int i=0;i<M*K;++i) hA16[i]=__float2bfloat16(hA[i]);
  for (int i=0;i<N*K;++i) hB16[i]=__float2bfloat16(hB[i]);
  __nv_bfloat16 *dA,*dB,*dD;
  CHECK_CUDA(cudaMalloc(&dA,M*K*sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dB,N*K*sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dD,M*N*sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemcpy(dA,hA16.data(),M*K*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,hB16.data(),N*K*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));

  cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));
  using BF = cutlass::bfloat16_t;
  launch_gemm_2cta(reinterpret_cast<const BF*>(dA), reinterpret_cast<const BF*>(dB),
                   reinterpret_cast<BF*>(dD), M, N, K, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> hD(M*N);
  CHECK_CUDA(cudaMemcpy(hD.data(),dD,M*N*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
  std::vector<float> fD(M*N);
  for (int i=0;i<M*N;++i) fD[i]=__bfloat162float(hD[i]);
  float rel = relative_error(fD, hRef);
  std::cout << "Relative error vs FP32 ref: " << rel << "\n";
  std::cout << "Correctness: " << ((rel<5e-2f)?"PASS":"FAIL") << "\n";

  cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaStreamDestroy(stream);
  return 0;
#endif
}
