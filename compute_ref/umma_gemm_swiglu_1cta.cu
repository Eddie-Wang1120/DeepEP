// umma_gemm_swiglu_1cta.cu — Stage (2b): 1CTA UMMA fused gate+up GEMM with
// in-register SwiGLU epilogue. Computes, for one output tile:
//
//   act[m,n] = silu(gate[m,n]) * up[m,n] * route_w[m]
//   gate = A @ W_gate^T      (A:[M,K], W_gate:[N,K])   N = intermediate
//   up   = A @ W_up^T        (A:[M,K], W_up:[N,K])
//
// This mirrors megakernel's device_gemm_swiglu_fused (WMMA baseline) but on
// tcgen05 + TMEM. gate/up use TWO separate TMEM accumulators (two mainloops over
// the same A but different weights); the SwiGLU epilogue reads both back to
// registers and writes only `act` ([M,N]) to GMEM — no gate/up GMEM round-trip.
//
// Correctness baseline: a host reference that reproduces the WMMA fused-SwiGLU
// math (silu(gate)*up*route_w, route applied before nothing here — pure act out,
// matching the megakernel position: SwiGLU after, W_down before).
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -arch=sm_103a -O3 \
//        -I../cutlass_ref/include -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        umma_gemm_swiglu_1cta.cu -o umma_gemm_swiglu_1cta
//   ./umma_gemm_swiglu_1cta            # default M=128 K=4096 N=4096

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
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

using namespace cute;

#define CHECK_CUDA(call)                                                         \
  do {                                                                           \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
      std::cerr << "CUDA error " << cudaGetErrorString(_e) << " at "             \
                << __FILE__ << ":" << __LINE__ << std::endl;                     \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// SMEM: A is shared by both gate and up GEMMs; B has two buffers (Wgate / Wup).
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> Bg;  // W_gate tile
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> Bu;  // W_up   tile
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA()  { return make_tensor(make_smem_ptr(A.begin()),  ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sBg() { return make_tensor(make_smem_ptr(Bg.begin()), BSmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sBu() { return make_tensor(make_smem_ptr(Bu.begin()), BSmemLayout{}); }
};

// Run one K-mainloop accumulating A@Bw^T into the given TMEM accumulator.
template <class TiledMMA, class FragA, class FragB, class FragAcc>
CUTE_DEVICE void mainloop_into(TiledMMA tiled_mma, FragA const& tCrA, FragB const& tCrB,
                               FragAcc& tCtAcc, cute::uint64_t& bar, int& phase,
                               uint32_t elect_one_warp, int num_k_tiles,
                               // loaders: copy GMEM k-tile -> SMEM (done by caller before call)
                               // here we only do MMA over already-loaded SMEM fragments per k_tile
                               int /*unused*/) {
  // NOTE: this helper is intentionally not used for the staged-load version below;
  // kept minimal. See kernel body for the actual interleaved load+mma loop.
}

template <class SharedStorage,
          class ATensor, class BgTensor, class BuTensor, class DTensor, class RTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK>
__global__ static void
umma_swiglu_device(ATensor  mA,                      // (M, K)
                   BgTensor mBg,                     // (N, K)  W_gate
                   BuTensor mBu,                     // (N, K)  W_up
                   DTensor  mD,                      // (M, N)  act out
                   RTensor  mRoute,                  // (M,)    route weight per row
                   MmaTiler_MNK mma_tiler,
                   TiledMMA tiled_mma,
                   ClusterShape_MNK cluster_shape) {
  // --- Prologue ---
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

  // Single TMEM accumulator, reused for gate then up. We read gate back to
  // registers before clearing+computing up, so both live in RMEM for SwiGLU.
  // (Avoids manual TMEM column-offset arithmetic, which is 2D-encoded and was
  //  the cause of a misaligned-address fault.)
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);

  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = smem.tmem_base_ptr;

  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(smem.mma_barrier, /* num_ctas */ 1);
  }
  int phase = 0;
  __syncthreads();

  // Load all K-tiles of A, Wgate, Wup into SMEM once (K=4096 fits since tiles
  // are streamed per k_tile below). We stage per-k_tile inside each pass.

  // Helper: run a full K-mainloop of A@Bw^T into tCtAcc, then return.
  // Pass 0 = gate (Wg), pass 1 = up (Wu).
  TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r = t2r.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
  Tensor tDgD   = thr_t2r.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrG = make_tensor<AccType>(shape(tDgD));   // gate result in RMEM
  Tensor tDrU = make_tensor<AccType>(shape(tDgD));   // up   result in RMEM

  for (int pass = 0; pass < 2; ++pass) {
    auto mma = tiled_mma;
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
      cooperative_copy<128>(threadIdx.x, tCgA(_,_,_,k_tile), tCsA);
      if (pass == 0) cooperative_copy<128>(threadIdx.x, tCgBg(_,_,_,k_tile), tCsBg);
      else           cooperative_copy<128>(threadIdx.x, tCgBu(_,_,_,k_tile), tCsBu);
      __syncthreads();

      if (elect_one_warp) {
        auto& tCrB = (pass == 0) ? tCrBg : tCrBu;
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
          gemm(mma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
          mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&smem.mma_barrier);
      }
      cute::wait_barrier(smem.mma_barrier, phase);
      phase ^= 1;
    }
    // Drain this pass's accumulator to RMEM before reusing TMEM for next pass.
    if (pass == 0) copy(t2r, tDtAcc, tDrG);
    else           copy(t2r, tDtAcc, tDrU);
    __syncthreads();
  }

  // Map each epilogue element to its global (m,n) to fetch route_w[m].
  // Build an identity tensor over the SAME tile (gD) and partition it the same
  // way the accumulator is partitioned to GMEM, so coordinates line up 1:1.
  Tensor cD  = make_identity_tensor(shape(mD));                       // (M, N) -> (m,n)
  Tensor gcD = local_tile(cD, mma_tiler, mma_coord, Step<_1,_1, X>{}); // tile-local identity
  Tensor tCgcD = cta_mma.partition_C(gcD);                            // same partition as tCgD
  Tensor tDcD  = thr_t2r.partition_D(tCgcD);                          // per-thread (m,n) coords

  using OutType = typename DTensor::value_type;
  Tensor tDrAct = make_tensor<OutType>(shape(tDgD));
  CUTE_UNROLL
  for (int i = 0; i < size(tDrG); ++i) {
    auto coord = tDcD(i);                 // (m_global, n_global)
    int  m = get<0>(coord);
    float g = static_cast<float>(tDrG(i));
    float u = static_cast<float>(tDrU(i));
    float silu_g = g * (1.0f / (1.0f + ::expf(-g)));
    float rw = static_cast<float>(mRoute(m));
    tDrAct(i) = static_cast<OutType>(silu_g * u * rw);
  }
  copy(tDrAct, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeAB>
void launch_umma_swiglu(TypeAB const* dA, TypeAB const* dWg, TypeAB const* dWu,
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

  // Use N tile 128 (not 256) so the two accumulators fit TMEM together
  // (gate+up each 128xNtile FP32; 256 cols total budget is 512).
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeAB, TypeAB, TypeC, 128, 128,
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

  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  dim3 dimBlock(128);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(size(ceil_div(M, bM * size<1>(cluster_layout_vmnk))) * dimCluster.x,
               size(ceil_div(N, bN * size<2>(cluster_layout_vmnk))) * dimCluster.y);
  int smemBytes = sizeof(SMEMStorage);

  auto* kernel_ptr = &umma_swiglu_device<SMEMStorage,
                                          decltype(mA), decltype(mBg), decltype(mBu),
                                          decltype(mD), decltype(mR),
                                          decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape)>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*) kernel_ptr, mA, mBg, mBu, mD, mR, mma_tiler, tiled_mma, cluster_shape);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "UMMA SwiGLU kernel launch failed" << std::endl;
    std::exit(1);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ============================================================================
// Harness: host reference (reproduces megakernel WMMA fused-SwiGLU math) + compare.
// ============================================================================

static float relative_error(const std::vector<float>& t, const std::vector<float>& r) {
  double num = 0, den = 0;
  for (size_t i = 0; i < r.size(); ++i) { double d = t[i]-r[i]; num += d*d; den += double(r[i])*r[i]; }
  return float(std::sqrt(num) / (std::sqrt(den) + 1e-12));
}

int main(int argc, char** argv) {
  int M = 128, K = 4096, N = 4096;
  if (argc >= 2) M = atoi(argv[1]);
  if (argc >= 3) K = atoi(argv[2]);
  if (argc >= 4) N = atoi(argv[3]);
  std::cout << "Fused SwiGLU: act = silu(A@Wg^T)*(A@Wu^T)*route_w   M=" << M
            << " K=" << K << " N=" << N << "\n";

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined — build with -arch=sm_103a.\n";
  return 0;
#else
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  auto bf = [](float v){ return __bfloat162float(__float2bfloat16(v)); };

  std::vector<float> hA(M*K), hWg(N*K), hWu(N*K), hRoute(M);
  for (auto& x : hA)  x = dist(gen);
  for (auto& x : hWg) x = dist(gen);
  for (auto& x : hWu) x = dist(gen);
  for (auto& x : hRoute) x = 0.5f + 0.5f*dist(gen);  // positive-ish route weights

  // Host reference (bf16-rounded inputs, fp32 accumulate, matches kernel math)
  std::vector<float> hAb(M*K), hWgb(N*K), hWub(N*K);
  for (int i=0;i<M*K;++i) hAb[i]=bf(hA[i]);
  for (int i=0;i<N*K;++i){ hWgb[i]=bf(hWg[i]); hWub[i]=bf(hWu[i]); }
  std::vector<float> hRef(M*N);
  for (int m=0;m<M;++m) for (int n=0;n<N;++n) {
    double g=0, u=0;
    for (int k=0;k<K;++k){ g += double(hAb[m*K+k])*hWgb[n*K+k]; u += double(hAb[m*K+k])*hWub[n*K+k]; }
    float gf=float(g), uf=float(u);
    float silu = gf * (1.f/(1.f+std::exp(-gf)));
    hRef[m*N+n] = bf(silu * uf * hRoute[m]);  // round like bf16 store
  }

  // Device buffers
  std::vector<__nv_bfloat16> hA16(M*K), hWg16(N*K), hWu16(N*K);
  for (int i=0;i<M*K;++i) hA16[i]=__float2bfloat16(hA[i]);
  for (int i=0;i<N*K;++i){ hWg16[i]=__float2bfloat16(hWg[i]); hWu16[i]=__float2bfloat16(hWu[i]); }

  __nv_bfloat16 *dA,*dWg,*dWu,*dAct; float* dRoute;
  CHECK_CUDA(cudaMalloc(&dA,  M*K*sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dWg, N*K*sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dWu, N*K*sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dAct,M*N*sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dRoute, M*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA,  hA16.data(),  M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWg, hWg16.data(), N*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWu, hWu16.data(), N*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dRoute, hRoute.data(), M*sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));
  using BF = cutlass::bfloat16_t;
  launch_umma_swiglu(reinterpret_cast<const BF*>(dA), reinterpret_cast<const BF*>(dWg),
                     reinterpret_cast<const BF*>(dWu), reinterpret_cast<BF*>(dAct),
                     dRoute, M, N, K, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> hAct(M*N);
  CHECK_CUDA(cudaMemcpy(hAct.data(), dAct, M*N*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> fAct(M*N);
  for (int i=0;i<M*N;++i) fAct[i]=__bfloat162float(hAct[i]);

  float rel = relative_error(fAct, hRef);
  std::cout << "Relative error vs fused-SwiGLU ref: " << rel << "\n";
  std::cout << "Correctness: " << ((rel < 5e-2f) ? "PASS" : "FAIL") << "\n";

  cudaFree(dA); cudaFree(dWg); cudaFree(dWu); cudaFree(dAct); cudaFree(dRoute);
  cudaStreamDestroy(stream);
  return 0;
#endif
}
