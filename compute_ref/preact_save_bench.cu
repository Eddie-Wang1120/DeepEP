// preact_save_bench.cu — isolate the forward PreAct save store cost.
//
// The megakernel forward SwiGLU epilogue saves gate/up preactivation to
// bwd_preact[(recv_token,topk_slot), 2I] with 8 scalar BF16 stores per 4-column
// group (sm100_bf16_gemm_dg_copy.cuh:675). This microbench isolates JUST the
// store cost of that save, comparing:
//   - scalar : 8 x BF16 stores per 4-col group (current)
//   - int4   : one 16B vectorized store per 4-col group (proposed S1)
// under two address patterns:
//   - contiguous : dst row = m (slot-major, TMA-friendly)
//   - scatter    : dst row = perm[m] (mimics (recv_token,topk_slot) scatter)
//
// This measures store-instruction / address-generation cost in isolation. It
// does NOT model epilogue<->MMA overlap or TMEM-release stalls; those need the
// full fused-GEMM harness (S3). Use it as first-order signal for S1.
//
// Build:
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        preact_save_bench.cu -o preact_save_bench -lcuda
// Run:
//   ./preact_save_bench --m 1024 --n 3072 --warmup 100 --iters 500 --scatter 1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  std::exit(1); } } while (0)

__device__ __forceinline__ uint32_t pack_gu(float g, float u) {
  __nv_bfloat162 b = __float22bfloat162_rn({g, u});
  return *reinterpret_cast<uint32_t*>(&b);
}

// One CTA per token-row; block threads split the I logical channels in groups
// of 4 (mirrors the epilogue's 4-(gate,up)-pairs-per-bank-group granularity).
__global__ void save_scalar_kernel(const __nv_bfloat16* __restrict__ gu_src,
                                   __nv_bfloat16* __restrict__ preact_dst,
                                   const int* __restrict__ dst_row,
                                   int M, int I) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int I4 = I / 4;
  const size_t twoI = (size_t)2 * I;
  const int4* gu4 = reinterpret_cast<const int4*>(gu_src) + (size_t)m * I4;
  __nv_bfloat16* row = preact_dst + (size_t)dst_row[m] * twoI;
  for (int q = threadIdx.x; q < I4; q += blockDim.x) {
    const int4 v = gu4[q];
    auto bf = [](uint32_t x, int l) { return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&x)[l]); };
    const int base = q * 4;  // logical channel base
    float g0 = bf((uint32_t)v.x, 0), u0 = bf((uint32_t)v.x, 1);
    float g1 = bf((uint32_t)v.y, 0), u1 = bf((uint32_t)v.y, 1);
    float g2 = bf((uint32_t)v.z, 0), u2 = bf((uint32_t)v.z, 1);
    float g3 = bf((uint32_t)v.w, 0), u3 = bf((uint32_t)v.w, 1);
    row[2 * (base + 0)] = __float2bfloat16(g0); row[2 * (base + 0) + 1] = __float2bfloat16(u0);
    row[2 * (base + 1)] = __float2bfloat16(g1); row[2 * (base + 1) + 1] = __float2bfloat16(u1);
    row[2 * (base + 2)] = __float2bfloat16(g2); row[2 * (base + 2) + 1] = __float2bfloat16(u2);
    row[2 * (base + 3)] = __float2bfloat16(g3); row[2 * (base + 3) + 1] = __float2bfloat16(u3);
  }
}

__global__ void save_int4_kernel(const __nv_bfloat16* __restrict__ gu_src,
                                 __nv_bfloat16* __restrict__ preact_dst,
                                 const int* __restrict__ dst_row,
                                 int M, int I) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int I4 = I / 4;
  const size_t twoI = (size_t)2 * I;
  const int4* gu4 = reinterpret_cast<const int4*>(gu_src) + (size_t)m * I4;
  __nv_bfloat16* row = preact_dst + (size_t)dst_row[m] * twoI;
  for (int q = threadIdx.x; q < I4; q += blockDim.x) {
    const int4 v = gu4[q];
    auto bf = [](uint32_t x, int l) { return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&x)[l]); };
    const int base = q * 4;
    float g0 = bf((uint32_t)v.x, 0), u0 = bf((uint32_t)v.x, 1);
    float g1 = bf((uint32_t)v.y, 0), u1 = bf((uint32_t)v.y, 1);
    float g2 = bf((uint32_t)v.z, 0), u2 = bf((uint32_t)v.z, 1);
    float g3 = bf((uint32_t)v.w, 0), u3 = bf((uint32_t)v.w, 1);
    uint4 packed = make_uint4(pack_gu(g0, u0), pack_gu(g1, u1), pack_gu(g2, u2), pack_gu(g3, u3));
    *reinterpret_cast<uint4*>(&row[2 * base]) = packed;
  }
}

int main(int argc, char** argv) {
  int M = 1024, I = 3072, warmup = 100, iters = 500, scatter = 1, threads = 128;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char*) -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "missing value\n"); std::exit(1); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--m")) M = std::atoi(need("--m"));
    else if (!std::strcmp(argv[i], "--n")) I = std::atoi(need("--n"));
    else if (!std::strcmp(argv[i], "--warmup")) warmup = std::atoi(need("--warmup"));
    else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(need("--iters"));
    else if (!std::strcmp(argv[i], "--scatter")) scatter = std::atoi(need("--scatter"));
    else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(need("--threads"));
    else { fprintf(stderr, "unknown %s\n", argv[i]); std::exit(1); }
  }
  const size_t gu_elems = (size_t)M * 2 * I;
  const size_t gu_bytes = gu_elems * sizeof(__nv_bfloat16);

  std::vector<__nv_bfloat16> hgu(gu_elems);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : hgu) x = __float2bfloat16(dist(rng));

  std::vector<int> hrow(M);
  for (int m = 0; m < M; ++m) hrow[m] = m;
  if (scatter) std::shuffle(hrow.begin(), hrow.end(), rng);

  __nv_bfloat16 *dgu, *dscalar, *dint4;
  int* drow;
  CHECK_CUDA(cudaMalloc(&dgu, gu_bytes));
  CHECK_CUDA(cudaMalloc(&dscalar, gu_bytes));
  CHECK_CUDA(cudaMalloc(&dint4, gu_bytes));
  CHECK_CUDA(cudaMalloc(&drow, M * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dgu, hgu.data(), gu_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(drow, hrow.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dscalar, 0, gu_bytes));
  CHECK_CUDA(cudaMemset(dint4, 0, gu_bytes));

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  cudaEvent_t e0, e1; CHECK_CUDA(cudaEventCreate(&e0)); CHECK_CUDA(cudaEventCreate(&e1));
  auto dt = [&](auto&& fn) {
    CHECK_CUDA(cudaEventRecord(e0, s)); fn(); CHECK_CUDA(cudaEventRecord(e1, s));
    CHECK_CUDA(cudaEventSynchronize(e1));
    float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1)); return ms;
  };

  auto run_scalar = [&]() { save_scalar_kernel<<<M, threads, 0, s>>>(dgu, dscalar, drow, M, I); };
  auto run_int4 = [&]() { save_int4_kernel<<<M, threads, 0, s>>>(dgu, dint4, drow, M, I); };

  for (int w = 0; w < warmup; ++w) { run_scalar(); run_int4(); }
  CHECK_CUDA(cudaStreamSynchronize(s));

  double ts = 0, ti = 0;
  for (int it = 0; it < iters; ++it) ts += dt(run_scalar);
  for (int it = 0; it < iters; ++it) ti += dt(run_int4);
  ts /= iters; ti /= iters;

  // Correctness: both kernels must produce identical bytes.
  std::vector<__nv_bfloat16> a(gu_elems), b(gu_elems);
  CHECK_CUDA(cudaMemcpy(a.data(), dscalar, gu_bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(b.data(), dint4, gu_bytes, cudaMemcpyDeviceToHost));
  size_t mismatch = 0;
  for (size_t i = 0; i < gu_elems; ++i)
    if (*reinterpret_cast<uint16_t*>(&a[i]) != *reinterpret_cast<uint16_t*>(&b[i])) ++mismatch;

  printf("preact_save M=%d I=%d scatter=%d threads=%d\n", M, I, scatter, threads);
  printf("  scalar 8xBF16 : %.4f ms\n", ts);
  printf("  int4 16B      : %.4f ms\n", ti);
  printf("  speedup       : %.2fx\n", ts / ti);
  printf("  byte-mismatch : %zu / %zu\n", mismatch, gu_elems);

  CHECK_CUDA(cudaFree(dgu)); CHECK_CUDA(cudaFree(dscalar)); CHECK_CUDA(cudaFree(dint4));
  CHECK_CUDA(cudaFree(drow));
  CHECK_CUDA(cudaEventDestroy(e0)); CHECK_CUDA(cudaEventDestroy(e1));
  CHECK_CUDA(cudaStreamDestroy(s));
  return 0;
}
