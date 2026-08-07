// umma_dgrad_direct_layout.cu - BF16 dgrad direct-layout microkernel.
//
// Compares the current megakernel dgrad layout against a DeepGEMM-style
// direct-original-weight layout:
//   current: A[M,K] @ W_T[N,K]^T, B major = K
//   direct:  A[M,K] @ W[K,N],     B major = MN
//
// Build:
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../cutlass_ref/include \
//        -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        umma_dgrad_direct_layout.cu -o umma_dgrad_direct_layout -lcuda
//
// Run:
//   ./umma_dgrad_direct_layout --m 256 --hidden 4096 --intermediate 4096 --iters 20

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/simd_sm100.hpp>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include "sm100_bf16_gemm_dg_copy.cuh"

#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(_e) << " at " \
              << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while (0)

#define CHECK_CU(call) do { \
  CUresult _e = (call); \
  if (_e != CUDA_SUCCESS) { \
    const char* _s = nullptr; \
    cuGetErrorString(_e, &_s); \
    std::cerr << "CU error " << (_s ? _s : "<unknown>") << " at " \
              << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while (0)

static constexpr uint32_t CFG_BLOCK_M = 128;
static constexpr uint32_t CFG_BLOCK_N = 128;
static constexpr uint32_t CFG_BF16_BLOCK_K = 64;
static constexpr uint32_t CFG_NUM_GROUPS = 1;
static constexpr uint32_t CFG_SWZ_A = 128;
static constexpr uint32_t CFG_SWZ_B = 128;
static constexpr uint32_t CFG_SWZ_CD = 128;
static constexpr uint32_t CFG_NUM_STAGES = 4;
static constexpr uint32_t CFG_NON_EPI_THREADS = 128;
static constexpr uint32_t CFG_EPI_THREADS = 128;
static constexpr uint32_t CFG_PHYSICAL_THREADS = 800;
static constexpr uint32_t CFG_NUM_MULTICAST = 1;
static constexpr bool CFG_MCAST_ON_A = false;
static constexpr uint32_t CFG_K_ALIGNMENT = 128;
static constexpr uint64_t CFG_TC_UTIL = 100;
static constexpr int CFG_NUM_SMS = 48;

struct Options {
  int M = 256;
  int hidden = 4096;
  int intermediate = 4096;
  int sms = CFG_NUM_SMS;
  int warmup = 5;
  int iters = 20;
  int experts = 4;
  int device = 0;
};

static void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--m M] [--hidden H] [--intermediate I]"
            << " [--sms 48] [--warmup W] [--iters I] [--experts E] [--device D]\n";
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        usage(argv[0]);
        std::exit(1);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--m")) opt.M = std::atoi(need("--m"));
    else if (!std::strcmp(argv[i], "--hidden") || !std::strcmp(argv[i], "--k")) opt.hidden = std::atoi(need(argv[i]));
    else if (!std::strcmp(argv[i], "--intermediate") || !std::strcmp(argv[i], "--n") || !std::strcmp(argv[i], "--i")) opt.intermediate = std::atoi(need(argv[i]));
    else if (!std::strcmp(argv[i], "--sms")) opt.sms = std::atoi(need("--sms"));
    else if (!std::strcmp(argv[i], "--warmup")) opt.warmup = std::atoi(need("--warmup"));
    else if (!std::strcmp(argv[i], "--iters")) opt.iters = std::atoi(need("--iters"));
    else if (!std::strcmp(argv[i], "--experts")) opt.experts = std::atoi(need("--experts"));
    else if (!std::strcmp(argv[i], "--device")) opt.device = std::atoi(need("--device"));
    else if (!std::strcmp(argv[i], "--help")) { usage(argv[0]); std::exit(0); }
    else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (opt.sms != CFG_NUM_SMS) {
    std::cerr << "This microkernel is intentionally fixed to --sms 48 to match megakernel.\n";
    std::exit(1);
  }
  if (opt.M <= 0 || opt.hidden <= 0 || opt.intermediate <= 0 || opt.warmup < 0 || opt.iters <= 0 || opt.experts <= 0) {
    std::cerr << "Invalid non-positive option\n";
    std::exit(1);
  }
  if (opt.M % 8 || opt.hidden % 128 || opt.intermediate % 128) {
    std::cerr << "Expected M multiple of 8 and hidden/intermediate multiples of 128\n";
    std::exit(1);
  }
  return opt;
}

static CUtensorMap make_tma_2d(const void* ptr, CUtensorMapDataType dtype,
                               int gmem_inner, int gmem_outer,
                               int smem_inner, int smem_outer,
                               int gmem_outer_stride_elems,
                               int elem_size, int swizzle_bytes) {
  CUtensorMap tm;
  int swizzled_inner = swizzle_bytes ? swizzle_bytes / elem_size : smem_inner;
  const cuuint64_t global_dims[2] = {
      static_cast<cuuint64_t>(gmem_inner),
      static_cast<cuuint64_t>(gmem_outer)};
  const cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride_elems) * static_cast<cuuint64_t>(elem_size)};
  const cuuint32_t box_dims[2] = {
      static_cast<cuuint32_t>(swizzled_inner),
      static_cast<cuuint32_t>(smem_outer)};
  const cuuint32_t elem_strides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  if (swizzle_bytes == 128) swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  else if (swizzle_bytes == 64) swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  else if (swizzle_bytes == 32) swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  CHECK_CU(cuTensorMapEncodeTiled(&tm, dtype, 2, const_cast<void*>(ptr),
                                  global_dims, global_strides, box_dims, elem_strides,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}

static CUtensorMap make_bf16_a_desc(const __nv_bfloat16* a, int M, int K) {
  constexpr uint32_t load_block_m = CFG_BLOCK_M / (CFG_MCAST_ON_A ? CFG_NUM_MULTICAST : 1);
  return make_tma_2d(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                     K, M, CFG_BF16_BLOCK_K, load_block_m,
                     K, sizeof(__nv_bfloat16), CFG_SWZ_A);
}

static CUtensorMap make_bf16_b_kmajor_desc(const __nv_bfloat16* b_nk, int N, int K) {
  constexpr uint32_t load_block_n = CFG_BLOCK_N / (CFG_MCAST_ON_A ? 1 : CFG_NUM_MULTICAST);
  return make_tma_2d(b_nk, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                     K, N, CFG_BF16_BLOCK_K, load_block_n,
                     K, sizeof(__nv_bfloat16), CFG_SWZ_B);
}

static CUtensorMap make_bf16_b_mnmajor_desc(const __nv_bfloat16* b_kn, int N, int K) {
  constexpr uint32_t load_block_n = CFG_BLOCK_N / (CFG_MCAST_ON_A ? 1 : CFG_NUM_MULTICAST);
  return make_tma_2d(b_kn, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                     N, K, load_block_n, CFG_BF16_BLOCK_K,
                     N, sizeof(__nv_bfloat16), CFG_SWZ_B);
}

static CUtensorMap make_bf16_cd_desc(__nv_bfloat16* d, int M, int N) {
  return make_tma_2d(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                     N, M, CFG_SWZ_CD / sizeof(__nv_bfloat16), CFG_BLOCK_M,
                     N, sizeof(__nv_bfloat16), CFG_SWZ_CD);
}

template <cute::UMMA::Major MajorB>
static void launch_bf16_gemm_48(const __nv_bfloat16* dA,
                                const __nv_bfloat16* dB,
                                __nv_bfloat16* dD,
                                int M, int N, int K,
                                cudaStream_t stream) {
  auto dmap_a = make_bf16_a_desc(dA, M, K);
  auto dmap_b = [&]() {
    if constexpr (MajorB == cute::UMMA::Major::K) {
      return make_bf16_b_kmajor_desc(dB, N, K);
    } else {
      return make_bf16_b_mnmajor_desc(dB, N, K);
    }
  }();
  auto dmap_cd = make_bf16_cd_desc(dD, M, N);

  auto kernel = &deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K, MajorB,
      0u, 0u, 0u,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_BF16_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      CFG_NUM_SMS,
      CFG_K_ALIGNMENT,
      false, true,
      deep_gemm::GemmType::Normal, false, cutlass::bfloat16_t,
      CFG_TC_UTIL,
      false, false,
      CFG_PHYSICAL_THREADS>;

  size_t smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CFG_NUM_SMS, 1, 1);
  cfg.blockDim = dim3(CFG_PHYSICAL_THREADS, 1, 1);
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = CFG_NUM_MULTICAST;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel,
      static_cast<int*>(nullptr),
      static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K),
      dmap_a, dmap_b, dmap_cd,
      static_cast<const cutlass::bfloat16_t*>(nullptr),
      static_cast<const float*>(nullptr), 0u,
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<float*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<const int*>(nullptr),
      static_cast<const int*>(nullptr),
      0u, 0u));
  CHECK_CUDA(cudaGetLastError());
}

static void fill_random_bf16(std::vector<__nv_bfloat16>& v, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = __float2bfloat16(dist(rng));
}

static void transpose_rowmajor(const std::vector<__nv_bfloat16>& src,
                               std::vector<__nv_bfloat16>& dst,
                               int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
    }
  }
}

struct ErrorStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  double rmse = 0.0;
  double nrmse = 0.0;
  int nz = 0;
};

static ErrorStats compare_bf16(const std::vector<__nv_bfloat16>& ref,
                               const std::vector<__nv_bfloat16>& test) {
  ErrorStats st;
  double ss = 0.0;
  double rs = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    float r = __bfloat162float(ref[i]);
    float t = __bfloat162float(test[i]);
    float ar = std::fabs(r);
    float d = std::fabs(r - t);
    st.max_abs = std::max(st.max_abs, d);
    if (ar > 1e-6f) {
      st.max_rel = std::max(st.max_rel, d / ar);
      ++st.nz;
    }
    ss += static_cast<double>(d) * d;
    rs += static_cast<double>(ar) * ar;
  }
  st.rmse = std::sqrt(ss / std::max<size_t>(ref.size(), 1));
  st.nrmse = rs > 0.0 ? std::sqrt(ss / rs) : st.rmse;
  return st;
}

static void print_error(const char* label, const ErrorStats& st, size_t count) {
  std::printf("%s max_abs=%.6g max_rel=%.6g rmse=%.6g nrmse=%.6g nz=%d/%zu\n",
              label, st.max_abs, st.max_rel, st.rmse, st.nrmse, st.nz, count);
}

template <typename Fn>
static double bench_ms(cudaStream_t stream, cudaEvent_t ev0, cudaEvent_t ev1,
                       int warmup, int iters, Fn&& fn) {
  for (int i = 0; i < warmup; ++i) fn();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  double total = 0.0;
  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(cudaEventRecord(ev0, stream));
    fn();
    CHECK_CUDA(cudaEventRecord(ev1, stream));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    total += ms;
  }
  return total / iters;
}

static double tflops(double flop, double ms) {
  return flop / (ms * 1e-3) / 1e12;
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  CHECK_CU(cuInit(0));
  CHECK_CUDA(cudaSetDevice(opt.device));

  const int M = opt.M;
  const int H = opt.hidden;
  const int I = opt.intermediate;
  const int twoI = 2 * I;

  std::printf("=== umma_dgrad_direct_layout M=%d hidden=%d intermediate=%d sms=%d threads=%u ===\n",
              M, H, I, opt.sms, CFG_PHYSICAL_THREADS);
  std::printf("Layouts:\n");
  std::printf("  down current: grad_down[M,H] x W_down_T[I,H]^T -> grad_act[M,I]\n");
  std::printf("  down direct : grad_down[M,H] x W_down[H,I]     -> grad_act[M,I]\n");
  std::printf("  gate current: dgu[M,2I]     x W_gateup_T[H,2I]^T -> dx[M,H]\n");
  std::printf("  gate direct : dgu[M,2I]     x W_gateup[2I,H]     -> dx[M,H]\n\n");

  const size_t grad_down_elems = static_cast<size_t>(M) * H;
  const size_t dgu_elems = static_cast<size_t>(M) * twoI;
  const size_t down_out_elems = static_cast<size_t>(M) * I;
  const size_t dx_elems = static_cast<size_t>(M) * H;
  const size_t wdown_elems = static_cast<size_t>(H) * I;
  const size_t wgu_elems = static_cast<size_t>(twoI) * H;
  const size_t bf16_size = sizeof(__nv_bfloat16);

  std::vector<__nv_bfloat16> h_grad_down(grad_down_elems);
  std::vector<__nv_bfloat16> h_dgu(dgu_elems);
  std::vector<__nv_bfloat16> h_wdown(wdown_elems);
  std::vector<__nv_bfloat16> h_wdown_t(wdown_elems);
  std::vector<__nv_bfloat16> h_wgu(wgu_elems);
  std::vector<__nv_bfloat16> h_wgu_t(wgu_elems);
  std::mt19937 rng(42);
  fill_random_bf16(h_grad_down, rng);
  fill_random_bf16(h_dgu, rng);
  fill_random_bf16(h_wdown, rng);
  fill_random_bf16(h_wgu, rng);
  transpose_rowmajor(h_wdown, h_wdown_t, H, I);
  transpose_rowmajor(h_wgu, h_wgu_t, twoI, H);

  __nv_bfloat16 *d_grad_down = nullptr, *d_dgu = nullptr;
  __nv_bfloat16 *d_wdown = nullptr, *d_wdown_t = nullptr;
  __nv_bfloat16 *d_wgu = nullptr, *d_wgu_t = nullptr;
  __nv_bfloat16 *d_down_current = nullptr, *d_down_direct = nullptr;
  __nv_bfloat16 *d_gate_current = nullptr, *d_gate_direct = nullptr;

  CHECK_CUDA(cudaMalloc(&d_grad_down, grad_down_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_dgu, dgu_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_wdown, wdown_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_wdown_t, wdown_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_wgu, wgu_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_wgu_t, wgu_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_down_current, down_out_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_down_direct, down_out_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_gate_current, dx_elems * bf16_size));
  CHECK_CUDA(cudaMalloc(&d_gate_direct, dx_elems * bf16_size));

  CHECK_CUDA(cudaMemcpy(d_grad_down, h_grad_down.data(), grad_down_elems * bf16_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_dgu, h_dgu.data(), dgu_elems * bf16_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_wdown, h_wdown.data(), wdown_elems * bf16_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_wdown_t, h_wdown_t.data(), wdown_elems * bf16_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_wgu, h_wgu.data(), wgu_elems * bf16_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_wgu_t, h_wgu_t.data(), wgu_elems * bf16_size, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaEvent_t ev0, ev1;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&ev0));
  CHECK_CUDA(cudaEventCreate(&ev1));

  auto down_current = [&]() {
    launch_bf16_gemm_48<cute::UMMA::Major::K>(d_grad_down, d_wdown_t, d_down_current, M, I, H, stream);
  };
  auto down_direct = [&]() {
    launch_bf16_gemm_48<cute::UMMA::Major::MN>(d_grad_down, d_wdown, d_down_direct, M, I, H, stream);
  };
  auto gate_current = [&]() {
    launch_bf16_gemm_48<cute::UMMA::Major::K>(d_dgu, d_wgu_t, d_gate_current, M, H, twoI, stream);
  };
  auto gate_direct = [&]() {
    launch_bf16_gemm_48<cute::UMMA::Major::MN>(d_dgu, d_wgu, d_gate_direct, M, H, twoI, stream);
  };
  auto full_current = [&]() {
    down_current();
    gate_current();
  };
  auto full_direct = [&]() {
    down_direct();
    gate_direct();
  };

  down_current();
  down_direct();
  gate_current();
  gate_direct();
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> h_down_current(down_out_elems), h_down_direct(down_out_elems);
  std::vector<__nv_bfloat16> h_gate_current(dx_elems), h_gate_direct(dx_elems);
  CHECK_CUDA(cudaMemcpy(h_down_current.data(), d_down_current, down_out_elems * bf16_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_down_direct.data(), d_down_direct, down_out_elems * bf16_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_gate_current.data(), d_gate_current, dx_elems * bf16_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_gate_direct.data(), d_gate_direct, dx_elems * bf16_size, cudaMemcpyDeviceToHost));

  print_error("down_direct_vs_current", compare_bf16(h_down_current, h_down_direct), down_out_elems);
  print_error("gate_direct_vs_current", compare_bf16(h_gate_current, h_gate_direct), dx_elems);
  std::printf("\n");

  double ms_down_current = bench_ms(stream, ev0, ev1, opt.warmup, opt.iters, down_current);
  double ms_down_direct = bench_ms(stream, ev0, ev1, opt.warmup, opt.iters, down_direct);
  double ms_gate_current = bench_ms(stream, ev0, ev1, opt.warmup, opt.iters, gate_current);
  double ms_gate_direct = bench_ms(stream, ev0, ev1, opt.warmup, opt.iters, gate_direct);
  double ms_full_current = bench_ms(stream, ev0, ev1, opt.warmup, opt.iters, full_current);
  double ms_full_direct = bench_ms(stream, ev0, ev1, opt.warmup, opt.iters, full_direct);

  const double flops_down = 2.0 * M * H * I;
  const double flops_gate = 2.0 * M * twoI * H;
  const double flops_full = flops_down + flops_gate;

  std::printf("Timing, transpose materialization excluded from current path:\n");
  std::printf("  down_current_Kmajor     %.4f ms  %.3f TFLOPS\n", ms_down_current, tflops(flops_down, ms_down_current));
  std::printf("  down_direct_MNmajor     %.4f ms  %.3f TFLOPS  speedup %.3fx\n", ms_down_direct, tflops(flops_down, ms_down_direct), ms_down_current / ms_down_direct);
  std::printf("  gate_current_Kmajor     %.4f ms  %.3f TFLOPS\n", ms_gate_current, tflops(flops_gate, ms_gate_current));
  std::printf("  gate_direct_MNmajor     %.4f ms  %.3f TFLOPS  speedup %.3fx\n", ms_gate_direct, tflops(flops_gate, ms_gate_direct), ms_gate_current / ms_gate_direct);
  std::printf("  both_current_Kmajor     %.4f ms  %.3f TFLOPS\n", ms_full_current, tflops(flops_full, ms_full_current));
  std::printf("  both_direct_MNmajor     %.4f ms  %.3f TFLOPS  speedup %.3fx\n\n", ms_full_direct, tflops(flops_full, ms_full_direct), ms_full_current / ms_full_direct);

  const double transposed_per_expert_mib = static_cast<double>((wdown_elems + wgu_elems) * bf16_size) / 1024.0 / 1024.0;
  std::printf("Transposed-weight storage avoided: %.2f MiB/expert, %.2f MiB for %d experts\n",
              transposed_per_expert_mib, transposed_per_expert_mib * opt.experts, opt.experts);

  CHECK_CUDA(cudaFree(d_grad_down));
  CHECK_CUDA(cudaFree(d_dgu));
  CHECK_CUDA(cudaFree(d_wdown));
  CHECK_CUDA(cudaFree(d_wdown_t));
  CHECK_CUDA(cudaFree(d_wgu));
  CHECK_CUDA(cudaFree(d_wgu_t));
  CHECK_CUDA(cudaFree(d_down_current));
  CHECK_CUDA(cudaFree(d_down_direct));
  CHECK_CUDA(cudaFree(d_gate_current));
  CHECK_CUDA(cudaFree(d_gate_direct));
  CHECK_CUDA(cudaEventDestroy(ev0));
  CHECK_CUDA(cudaEventDestroy(ev1));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
