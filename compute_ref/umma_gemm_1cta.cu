// umma_gemm_1cta.cu — Stage (2a): standalone 1CTA Blackwell UMMA BF16 GEMM microkernel.
//
// Goal: validate, OUTSIDE the megakernel, that a single-CTA tcgen05.mma + TMEM
// GEMM compiles on sm_103a, is numerically correct, and is faster than the
// current WMMA path. This is the foundation for replacing megakernel's
// device_gemm_bf16 / device_gemm_swiglu_fused (see MEGAKERNEL_COMPUTE_DESIGN.md
// section I, stage 2a). No SwiGLU here yet — pure GEMM D = A @ B^T.
//
// Problem shape (megakernel real config): M=128, K=hidden=4096, N=intermediate=4096.
//   A: [M, K] row-major (K-major)         — activations
//   B: [N, K] row-major over (N,K)         — expert weight, accessed as A@B^T
//   D: [M, N] row-major                    — output
//   BF16 inputs, FP32 accumulate, BF16 output (matches megakernel).
//
// Comparison harness (this file):
//   1. UMMA microkernel  -> D_umma   (this kernel)
//   2. WMMA reference     -> D_wmma  (extracted from megakernel device_gemm_bf16)
//   3. FP32 CPU reference -> D_ref
//   Correctness: relative error of D_umma and D_wmma vs D_ref.
//   Performance: CUDA-event timing -> TFLOPS for UMMA vs WMMA.
//
// Build (user runs; B30Z = cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -arch=sm_103a -O3 \
//        -I../cutlass_ref/include -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr \
//        umma_gemm_1cta.cu -o umma_gemm_1cta
//   ./umma_gemm_1cta            # default M=128 K=4096 N=4096
//   ./umma_gemm_1cta 128 4096 4096

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

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

// ============================================================================
// Part 1: UMMA (tcgen05) 1CTA GEMM device kernel.
//   Adapted from cutlass_ref/examples/cute/tutorial/blackwell/01_mma_sm100.cu,
//   changed to BF16 inputs and beta=0 (pure D = A @ B^T, no C).
// ============================================================================

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <class TypeA, class TypeB,
          class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};

template <class SharedStorage,
          class ATensor, class BTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK>
__global__ static void
umma_gemm_device(ATensor mA,                     // (M, K)
                 BTensor mB,                      // (N, K)
                 DTensor mD,                      // (M, N)
                 MmaTiler_MNK mma_tiler,
                 TiledMMA tiled_mma,
                 ClusterShape_MNK cluster_shape) {
  // --- Prologue: tile partition ---
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                                   blockIdx.x / size<0>(cluster_layout_vmnk),
                                   blockIdx.y, _);
  auto mma_coord = select<1,2,3>(mma_coord_vmnk);
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X,_1>{});  // (MmaTile_M, MmaTile_K, Tiles_K)
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X,_1,_1>{});  // (MmaTile_N, MmaTile_K, Tiles_K)
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});  // (MmaTile_M, MmaTile_N)

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
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);   // TMEM accumulator

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
  int mma_barrier_phase_bit = 0;
  __syncthreads();

  // --- Mainloop ---
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;     // first MMA clears TMEM
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    cooperative_copy<128>(threadIdx.x, tCgA(_,_,_,k_tile), tCsA);
    cooperative_copy<128>(threadIdx.x, tCgB(_,_,_,k_tile), tCsB);
    __syncthreads();

    if (elect_one_warp) {
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&smem.mma_barrier);
    }
    cute::wait_barrier(smem.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  // --- Epilogue: TMEM -> RMEM -> GMEM (cast FP32 acc to BF16) ---
  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD   = thr_t2r_copy.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  // Cast FP32 accumulator -> BF16 output (this is where SwiGLU will go in 2b).
  using OutType = typename DTensor::value_type;
  Tensor tDrD = make_tensor<OutType>(shape(tDgD));
  CUTE_UNROLL
  for (int i = 0; i < size(tDrAcc); ++i) {
    tDrD(i) = static_cast<OutType>(tDrAcc(i));
  }
  copy(tDrD, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

// Host launcher for the UMMA kernel. A:[M,K] K-major, B:[N,K] K-major, D:[M,N] N-major.
template <class TypeAB>
void launch_umma_gemm(TypeAB const* dA, TypeAB const* dB, TypeAB* dD,
                      int M, int N, int K, cudaStream_t stream) {
  using TypeC = float;  // accumulator

  Layout layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  Layout layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  Layout layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

  Tensor mA = make_tensor(make_gmem_ptr(dA), layout_A);
  Tensor mB = make_tensor(make_gmem_ptr(dB), layout_B);
  Tensor mD = make_tensor(make_gmem_ptr(dD), layout_D);

  // 1CTA MMA atom: 128x256x16 (BF16). N rounded to multiple of 256; M to 128.
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeAB, TypeAB, TypeC, 128, 256,
                           UMMA::Major::K, UMMA::Major::K>{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};   // 4 MMAs per K tile (K16 each)
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

  auto* kernel_ptr = &umma_gemm_device<SMEMStorage,
                                        decltype(mA), decltype(mB), decltype(mD),
                                        decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape)>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*) kernel_ptr, mA, mB, mD, mma_tiler, tiled_mma, cluster_shape);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "UMMA kernel launch failed" << std::endl;
    std::exit(1);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ============================================================================
// Part 2: WMMA reference kernel (extracted from megakernel device_gemm_bf16).
//   Same semantics: C[M,N] = A[M,K] @ B[N,K]^T, BF16 in, FP32 acc, BF16 out.
//   Single block, warps stride over 16x16 output tiles (mirrors megakernel).
// ============================================================================

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void wmma_gemm_kernel(const __nv_bfloat16* __restrict__ A,
                                 const __nv_bfloat16* __restrict__ B,
                                 __nv_bfloat16* __restrict__ C,
                                 int M, int K, int N) {
  using namespace nvcuda;
  extern __shared__ float smem_buf[];
  const int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int num_warps = (gridDim.x * blockDim.x) / 32;
  const int smem_warp = threadIdx.x / 32;

  const int tiles_m = (M + WMMA_M - 1) / WMMA_M;
  const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
  const int total   = tiles_m * tiles_n;

  for (int tile = warp_id; tile < total; tile += num_warps) {
    int row_off = (tile / tiles_n) * WMMA_M;
    int col_off = (tile % tiles_n) * WMMA_N;
    if (row_off >= M || col_off >= N) continue;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
      wmma::load_matrix_sync(a_frag, A + row_off * K + k, K);
      wmma::load_matrix_sync(b_frag, B + col_off * K + k, K);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* cb = smem_buf + smem_warp * WMMA_M * WMMA_N;
    wmma::store_matrix_sync(cb, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();
    int lane = threadIdx.x % 32;
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
      int r = row_off + i / WMMA_N, c = col_off + i % WMMA_N;
      if (r < M && c < N) C[r * N + c] = __float2bfloat16(cb[i]);
    }
  }
}

void launch_wmma_gemm(const __nv_bfloat16* dA, const __nv_bfloat16* dB, __nv_bfloat16* dC,
                      int M, int N, int K, cudaStream_t stream) {
  // Mirror megakernel: 32 SMs (blocks) cooperate, 25 warps/block (800 threads).
  const int threads = 800;
  const int blocks  = 32;
  const int smem    = (threads / 32) * WMMA_M * WMMA_N * sizeof(float);
  CHECK_CUDA(cudaFuncSetAttribute(wmma_gemm_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  wmma_gemm_kernel<<<blocks, threads, smem, stream>>>(dA, dB, dC, M, K, N);
}

// ============================================================================
// Part 3: harness — init, FP32 CPU reference, correctness, timing.
// ============================================================================

static float relative_error(const std::vector<float>& test, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = double(test[i]) - double(ref[i]);
    num += d * d;
    den += double(ref[i]) * double(ref[i]);
  }
  return float(std::sqrt(num) / (std::sqrt(den) + 1e-12));
}

template <class LaunchFn>
float time_kernel(LaunchFn fn, int warmup, int iters, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) fn();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaEvent_t beg, end;
  CHECK_CUDA(cudaEventCreate(&beg));
  CHECK_CUDA(cudaEventCreate(&end));
  std::vector<float> times(iters);
  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(cudaEventRecord(beg, stream));
    fn();
    CHECK_CUDA(cudaEventRecord(end, stream));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaEventElapsedTime(&times[i], beg, end));
  }
  std::sort(times.begin(), times.end());
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  return times[iters / 2];  // median (ms)
}

int main(int argc, char** argv) {
  int M = 128, K = 4096, N = 4096;
  if (argc >= 2) M = atoi(argv[1]);
  if (argc >= 3) K = atoi(argv[2]);
  if (argc >= 4) N = atoi(argv[3]);
  std::cout << "GEMM D = A @ B^T  M=" << M << " K=" << K << " N=" << N
            << "  (BF16 in, FP32 acc, BF16 out)\n";

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined — build with -arch=sm_100a/sm_103a.\n";
  return 0;
#else
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> hA(M * K), hB(N * K);
  for (auto& x : hA) x = dist(gen);
  for (auto& x : hB) x = dist(gen);

  // FP32 CPU reference: D_ref[m,n] = sum_k A[m,k]*B[n,k]  (with values rounded to bf16 first)
  auto to_bf16_f = [](float v) { return __bfloat162float(__float2bfloat16(v)); };
  std::vector<float> hA_bf(M * K), hB_bf(N * K);
  for (int i = 0; i < M * K; ++i) hA_bf[i] = to_bf16_f(hA[i]);
  for (int i = 0; i < N * K; ++i) hB_bf[i] = to_bf16_f(hB[i]);
  std::vector<float> hRef(M * N, 0.f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += double(hA_bf[m * K + k]) * double(hB_bf[n * K + k]);
      hRef[m * N + n] = float(acc);
    }

  // Device BF16 inputs
  std::vector<__nv_bfloat16> hA16(M * K), hB16(N * K);
  for (int i = 0; i < M * K; ++i) hA16[i] = __float2bfloat16(hA[i]);
  for (int i = 0; i < N * K; ++i) hB16[i] = __float2bfloat16(hB[i]);

  __nv_bfloat16 *dA, *dB, *dD_umma, *dD_wmma;
  CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dB, N * K * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dD_umma, M * N * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dD_wmma, M * N * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemcpy(dA, hA16.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB16.data(), N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  using BF = cutlass::bfloat16_t;  // bit-compatible with __nv_bfloat16
  auto umma_fn = [&] {
    launch_umma_gemm(reinterpret_cast<const BF*>(dA), reinterpret_cast<const BF*>(dB),
                     reinterpret_cast<BF*>(dD_umma), M, N, K, stream);
  };
  auto wmma_fn = [&] { launch_wmma_gemm(dA, dB, dD_wmma, M, N, K, stream); };

  // Correctness
  umma_fn();
  wmma_fn();
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> hD_umma(M * N), hD_wmma(M * N);
  CHECK_CUDA(cudaMemcpy(hD_umma.data(), dD_umma, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hD_wmma.data(), dD_wmma, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> fU(M * N), fW(M * N);
  for (int i = 0; i < M * N; ++i) { fU[i] = __bfloat162float(hD_umma[i]); fW[i] = __bfloat162float(hD_wmma[i]); }

  float relU = relative_error(fU, hRef);
  float relW = relative_error(fW, hRef);
  std::cout << "Relative error vs FP32 ref:  UMMA=" << relU << "   WMMA=" << relW << "\n";
  // BF16 GEMM with K=4096 typically lands ~1e-2; flag if clearly wrong.
  bool ok = (relU < 5e-2f) && (relW < 5e-2f);
  std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << "\n";

  // Performance
  const int warmup = 20, iters = 100;
  float t_umma = time_kernel(umma_fn, warmup, iters, stream);
  float t_wmma = time_kernel(wmma_fn, warmup, iters, stream);
  double flops = 2.0 * M * N * K;
  std::cout << "UMMA: " << t_umma << " ms   " << (flops / (t_umma * 1e-3) / 1e12) << " TFLOPS\n";
  std::cout << "WMMA: " << t_wmma << " ms   " << (flops / (t_wmma * 1e-3) / 1e12) << " TFLOPS\n";
  std::cout << "Speedup (WMMA/UMMA time): " << (t_wmma / t_umma) << "x\n";

  cudaFree(dA); cudaFree(dB); cudaFree(dD_umma); cudaFree(dD_wmma);
  cudaStreamDestroy(stream);
  return 0;
#endif
}
