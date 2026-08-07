// umma_swiglu_ws_dg.cu — DeepGEMM BF16/FP8 standalone microkernel harness.
//
// This file compares the current BF16 DeepGEMM UMMA path with the SM100 FP8
// 1D1D path copied into compute_ref.  Gate/up uses the same interleaved single
// GEMM shape as the megakernel baseline: Wgu rows are [g0,u0,g1,u1,...].  The
// BF16 path uses the interleaved SwiGLU epilogue; the FP8 paths use one
// interleaved gateup GEMM, then apply no-quant or quantized SwiGLU before down.
//
// Build (B30Z compute_103a; must use -gencode):
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../cutlass_ref/include \
//        -I../cutlass_ref/tools/util/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 \
//        umma_swiglu_ws_dg.cu -o umma_swiglu_ws_dg -lcuda
//
// Example:
//   ./umma_swiglu_ws_dg --mode both --m 256 --n 4096 --k 4096 --sms 132

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include "sm100_bf16_gemm_dg_copy.cuh"
#include "sm100_fp8_fp4_gemm_1d1d_copy.cuh"

#define CHECK_CUDA(call)                                                         \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                               \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":"      \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call)                                                           \
  do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s=nullptr;           \
    cuGetErrorString(_e,&s);                                                     \
    std::cerr<<"CU error "<<(s?s:"<unknown>")<<" at "<<__FILE__<<":"            \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)

// ======================= Config =======================
static constexpr uint32_t CFG_SHAPE_M = 0;
static constexpr uint32_t CFG_SHAPE_N = 0;
static constexpr uint32_t CFG_SHAPE_K = 0;
static constexpr uint32_t CFG_BLOCK_M = 128;
static constexpr uint32_t CFG_BLOCK_N = 128;
static constexpr uint32_t CFG_BF16_BLOCK_K = 64;
static constexpr uint32_t CFG_FP8_BLOCK_K = 128;
static constexpr uint32_t CFG_FP8_FUSED_BLOCK_N = 256;
static constexpr uint32_t CFG_FP8_FUSED_NUM_STAGES = 2;
static constexpr uint32_t CFG_NUM_GROUPS = 1;
static constexpr uint32_t CFG_SWZ_A = 128;
static constexpr uint32_t CFG_SWZ_B = 128;
static constexpr uint32_t CFG_SWZ_CD = 128;
static constexpr uint32_t CFG_NUM_STAGES = 4;
static constexpr uint32_t CFG_NON_EPI_THREADS = 128;
static constexpr uint32_t CFG_EPI_THREADS = 128;
static constexpr uint32_t CFG_PHYSICAL_THREADS = 800;  // mirror megakernel compute CTA
static constexpr uint32_t CFG_NUM_MULTICAST = 1;        // align with current 1-CTA megakernel baseline
static constexpr bool CFG_MCAST_ON_A = false;           // megakernel multicasts B/N when multicast is enabled
static constexpr uint32_t CFG_K_ALIGNMENT = 128;
static constexpr uint64_t CFG_TC_UTIL = 100;
static constexpr uint32_t CFG_FP8_GRAN_K_A = 128;
static constexpr uint32_t CFG_FP8_GRAN_K_B = 128;

struct Options {
  int M = 256;
  int N = 4096;
  int K = 4096;
  int sms = 132;
  int warmup = 5;
  int iters = 20;
  bool check_swapab = false;
  std::string mode = "both";
};

static void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--mode bf16|fp8|both] [--m M] [--n N] [--k K]"
            << " [--sms 16|32|64|96|120|132] [--warmup W] [--iters I]"
            << " [--check-swapab]\n";
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--mode")) opt.mode = need("--mode");
    else if (!std::strcmp(argv[i], "--m")) opt.M = std::atoi(need("--m"));
    else if (!std::strcmp(argv[i], "--n")) opt.N = std::atoi(need("--n"));
    else if (!std::strcmp(argv[i], "--k")) opt.K = std::atoi(need("--k"));
    else if (!std::strcmp(argv[i], "--sms")) opt.sms = std::atoi(need("--sms"));
    else if (!std::strcmp(argv[i], "--warmup")) opt.warmup = std::atoi(need("--warmup"));
    else if (!std::strcmp(argv[i], "--iters")) opt.iters = std::atoi(need("--iters"));
    else if (!std::strcmp(argv[i], "--check-swapab")) opt.check_swapab = true;
    else if (!std::strcmp(argv[i], "--help")) { usage(argv[0]); std::exit(0); }
    else { std::cerr << "Unknown arg: " << argv[i] << "\n"; usage(argv[0]); std::exit(1); }
  }
  if (opt.mode != "bf16" && opt.mode != "fp8" && opt.mode != "both") {
    std::cerr << "Invalid --mode: " << opt.mode << "\n";
    std::exit(1);
  }
  if (opt.M <= 0 || opt.N <= 0 || opt.K <= 0 || opt.warmup < 0 || opt.iters <= 0) {
    std::cerr << "Invalid non-positive shape/iteration option\n";
    std::exit(1);
  }
  if (opt.M % 8 || opt.N % 128 || opt.K % 128) {
    std::cerr << "Expected M multiple of 8 and N,K multiples of 128 for this harness\n";
    std::exit(1);
  }
  return opt;
}

// ======================= TMA descriptor builders =======================
static CUtensorMap make_tma_2d(const void* ptr, CUtensorMapDataType dtype,
                               int gmem_inner, int gmem_outer,
                               int smem_inner, int smem_outer,
                               int gmem_outer_stride_elems, int elem_size,
                               int swizzle_bytes) {
  CUtensorMap tm;
  int si = smem_inner;
  if (swizzle_bytes != 0) si = swizzle_bytes / elem_size;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint64_t strides[1] = {(cuuint64_t)gmem_outer_stride_elems * (cuuint64_t)elem_size};
  const cuuint32_t box[2] = {(cuuint32_t)si, (cuuint32_t)smem_outer};
  const cuuint32_t elem_stride[2] = {1, 1};
  CUtensorMapSwizzle swz = CU_TENSOR_MAP_SWIZZLE_NONE;
  if (swizzle_bytes == 128) swz = CU_TENSOR_MAP_SWIZZLE_128B;
  else if (swizzle_bytes == 64) swz = CU_TENSOR_MAP_SWIZZLE_64B;
  else if (swizzle_bytes == 32) swz = CU_TENSOR_MAP_SWIZZLE_32B;
  CHECK_CU(cuTensorMapEncodeTiled(&tm, dtype, 2, (void*)ptr, gdims, strides,
                                  box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  swz, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}

static CUtensorMap make_bf16_a_desc(const __nv_bfloat16* a, int M, int K) {
  constexpr uint32_t load_block_m = CFG_BLOCK_M / (CFG_MCAST_ON_A ? CFG_NUM_MULTICAST : 1);
  return make_tma_2d(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, M,
                     CFG_BF16_BLOCK_K, load_block_m, K, sizeof(__nv_bfloat16), CFG_SWZ_A);
}
static CUtensorMap make_bf16_b_desc(const __nv_bfloat16* b, int N, int K) {
  constexpr uint32_t load_block_n = CFG_BLOCK_N / (CFG_MCAST_ON_A ? 1 : CFG_NUM_MULTICAST);
  return make_tma_2d(b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, N,
                     CFG_BF16_BLOCK_K, load_block_n, K, sizeof(__nv_bfloat16), CFG_SWZ_B);
}
static CUtensorMap make_cd_desc(__nv_bfloat16* d, int M, int N) {
  return make_tma_2d(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, N, M,
                     CFG_SWZ_CD / sizeof(__nv_bfloat16), CFG_BLOCK_M, N, sizeof(__nv_bfloat16), CFG_SWZ_CD);
}
static CUtensorMap make_cd_desc_swapab(__nv_bfloat16* d, int M, int N) {
  constexpr uint32_t swapab_store_block_m = 16;
  return make_tma_2d(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, N, M,
                     CFG_SWZ_CD / sizeof(__nv_bfloat16), swapab_store_block_m, N, sizeof(__nv_bfloat16), CFG_SWZ_CD);
}
static CUtensorMap make_fp8_a_desc(const cutlass::float_e4m3_t* a, int M, int K) {
  constexpr uint32_t load_block_m = CFG_BLOCK_M / (CFG_MCAST_ON_A ? CFG_NUM_MULTICAST : 1);
  return make_tma_2d(a, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, M,
                     CFG_FP8_BLOCK_K, load_block_m, K, sizeof(cutlass::float_e4m3_t), CFG_SWZ_A);
}
static CUtensorMap make_fp8_b_desc(const cutlass::float_e4m3_t* b, int N, int K) {
  constexpr uint32_t load_block_n = CFG_BLOCK_N / (CFG_MCAST_ON_A ? 1 : CFG_NUM_MULTICAST);
  return make_tma_2d(b, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, N,
                     CFG_FP8_BLOCK_K, load_block_n, K, sizeof(cutlass::float_e4m3_t), CFG_SWZ_B);
}
static CUtensorMap make_sf_desc(const uint32_t* sf, int mn, int sf_k_packed) {
  return make_tma_2d(sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, mn, sf_k_packed,
                     CFG_BLOCK_M, 1, mn, sizeof(uint32_t), 0);
}

// ======================= Launch helpers =======================
template <int NumSms, bool FuseSwiGLU>
void launch_bf16_gemm_sms(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                          __nv_bfloat16* dD, int M, int N, int K,
                          cudaStream_t stream,
                          const __nv_bfloat16* gate = nullptr,
                          const float* route = nullptr) {
  auto dmap_a = make_bf16_a_desc(dA, M, K);
  auto dmap_b = make_bf16_b_desc(dB, N, K);
  auto dmap_cd = make_cd_desc(dD, M, N);

  auto kernel = &deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      CFG_SHAPE_M, CFG_SHAPE_N, CFG_SHAPE_K,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_BF16_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      NumSms,
      CFG_K_ALIGNMENT,
      false, true,
      deep_gemm::GemmType::Normal, false, cutlass::bfloat16_t,
      CFG_TC_UTIL,
      FuseSwiGLU, false,
      CFG_PHYSICAL_THREADS>;

  size_t smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NumSms, 1, 1);
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

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, nullptr, (uint32_t)M, (uint32_t)N, (uint32_t)K,
                                dmap_a, dmap_b, dmap_cd,
                                reinterpret_cast<const cutlass::bfloat16_t*>(gate), route, (uint32_t)N));
  CHECK_CUDA(cudaGetLastError());
}

template <int NumSms>
void launch_bf16_interleaved_swiglu_sms(const __nv_bfloat16* dA, const __nv_bfloat16* dWgu,
                                        __nv_bfloat16* dAct, const float* dRoute,
                                        int M, int I, int K, cudaStream_t stream) {
  const int N2 = 2 * I;
  auto dmap_a = make_bf16_a_desc(dA, M, K);
  auto dmap_b = make_bf16_b_desc(dWgu, N2, K);
  auto dmap_cd = make_cd_desc(dAct, M, I);

  auto kernel = &deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      CFG_SHAPE_M, CFG_SHAPE_N, CFG_SHAPE_K,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_BF16_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      NumSms,
      CFG_K_ALIGNMENT,
      false, true,
      deep_gemm::GemmType::Normal, false, cutlass::bfloat16_t,
      CFG_TC_UTIL,
      false, true,
      CFG_PHYSICAL_THREADS>;

  size_t smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NumSms, 1, 1);
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

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, nullptr, (uint32_t)M, (uint32_t)N2, (uint32_t)K,
                                dmap_a, dmap_b, dmap_cd,
                                static_cast<const cutlass::bfloat16_t*>(nullptr), dRoute, (uint32_t)I));
  CHECK_CUDA(cudaGetLastError());
}

template <int NumSms, uint32_t GranKA = CFG_FP8_GRAN_K_A, uint32_t GranKB = CFG_FP8_GRAN_K_B>
void launch_fp8_gemm_sms(const cutlass::float_e4m3_t* dA, const cutlass::float_e4m3_t* dB,
                         const uint32_t* dSfa, const uint32_t* dSfb,
                         __nv_bfloat16* dD, int M, int N, int K,
                         cudaStream_t stream) {
  auto dmap_a = make_fp8_a_desc(dA, M, K);
  auto dmap_b = make_fp8_b_desc(dB, N, K);
  int sf_k_packed_a = (K + (int)GranKA * 4 - 1) / ((int)GranKA * 4);
  int sf_k_packed_b = (K + (int)GranKB * 4 - 1) / ((int)GranKB * 4);
  auto dmap_sfa = make_sf_desc(dSfa, M, sf_k_packed_a);
  auto dmap_sfb = make_sf_desc(dSfb, N, sf_k_packed_b);
  auto dmap_cd = make_cd_desc(dD, M, N);

  auto kernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      GranKA, GranKB, CFG_K_ALIGNMENT,
      CFG_SHAPE_M, CFG_SHAPE_N, CFG_SHAPE_K,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_FP8_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      NumSms,
      false, true,
      deep_gemm::GemmType::Normal, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity,
      CFG_PHYSICAL_THREADS>;

  size_t smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NumSms, 1, 1);
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

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, nullptr, (uint32_t)M, (uint32_t)N, (uint32_t)K,
                                dmap_a, dmap_b, dmap_sfa, dmap_sfb, dmap_cd,
                                nullptr, nullptr, nullptr, 0u, dmap_cd));
  CHECK_CUDA(cudaGetLastError());
}

template <int NumSms>
void launch_fp8_gemm_swapab_sms(const cutlass::float_e4m3_t* dA, const cutlass::float_e4m3_t* dB,
                                const uint32_t* dSfa, const uint32_t* dSfb,
                                __nv_bfloat16* dD, int M, int N, int K,
                                cudaStream_t stream) {
  auto dmap_a = make_fp8_a_desc(dA, M, K);
  auto dmap_b = make_fp8_b_desc(dB, N, K);
  int sf_k_packed = (K + (int)CFG_FP8_GRAN_K_A * 4 - 1) / ((int)CFG_FP8_GRAN_K_A * 4);
  auto dmap_sfa = make_sf_desc(dSfa, M, sf_k_packed);
  auto dmap_sfb = make_sf_desc(dSfb, N, sf_k_packed);
  auto dmap_cd = make_cd_desc_swapab(dD, M, N);

  auto kernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      CFG_FP8_GRAN_K_A, CFG_FP8_GRAN_K_B, CFG_K_ALIGNMENT,
      CFG_SHAPE_M, CFG_SHAPE_N, CFG_SHAPE_K,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_FP8_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      NumSms,
      true, true,
      deep_gemm::GemmType::Normal, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity,
      CFG_PHYSICAL_THREADS>;

  size_t smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NumSms, 1, 1);
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

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, nullptr, (uint32_t)M, (uint32_t)N, (uint32_t)K,
                                dmap_a, dmap_b, dmap_sfa, dmap_sfb, dmap_cd,
                                nullptr, nullptr, nullptr, 0u, dmap_cd));
  CHECK_CUDA(cudaGetLastError());
}

template <int NumSms>
void launch_fp8_interleaved_swiglu_quant_fused_swapab_half_sms(const cutlass::float_e4m3_t* dA,
                                                               const cutlass::float_e4m3_t* dWgu,
                                                               const uint32_t* dSfa,
                                                               const uint32_t* dSfwGateUp,
                                                               const float* dRoute,
                                                               cutlass::float_e4m3_t* dAct8,
                                                               uint32_t* dActSf,
                                                               __nv_bfloat16* dScratch,
                                                               int M, int I, int K,
                                                               cudaStream_t stream) {
  const int N2 = 2 * I;
  auto dmap_a = make_fp8_a_desc(dA, M, K);
  auto dmap_b = make_fp8_b_desc(dWgu, N2, K);
  int sf_k_packed = (K + (int)CFG_FP8_GRAN_K_A * 4 - 1) / ((int)CFG_FP8_GRAN_K_A * 4);
  auto dmap_sfa = make_sf_desc(dSfa, M, sf_k_packed);
  auto dmap_sfb = make_sf_desc(dSfwGateUp, N2, sf_k_packed);
  auto dmap_cd = make_cd_desc_swapab(dScratch, M, N2);
  auto dmap_act = make_tma_2d(dAct8, CU_TENSOR_MAP_DATA_TYPE_UINT8, I, M,
                              64, 16, I, sizeof(cutlass::float_e4m3_t), 64);

  auto kernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      CFG_FP8_GRAN_K_A, CFG_FP8_GRAN_K_B, CFG_K_ALIGNMENT,
      CFG_SHAPE_M, CFG_SHAPE_N, CFG_SHAPE_K,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_FP8_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      NumSms,
      true, true,
      deep_gemm::GemmType::Normal, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity,
      CFG_PHYSICAL_THREADS,
      true>;

  size_t smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NumSms, 1, 1);
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

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, nullptr, (uint32_t)M, (uint32_t)N2, (uint32_t)K,
                                dmap_a, dmap_b, dmap_sfa, dmap_sfb, dmap_cd,
                                dAct8, dActSf, dRoute, (uint32_t)I, dmap_act));
  CHECK_CUDA(cudaGetLastError());
}

template <int NumSms, bool Group32 = false>
void launch_fp8_interleaved_swiglu_quant_fused_sms(const cutlass::float_e4m3_t* dA,
                                                   const cutlass::float_e4m3_t* dWgu,
                                                   const uint32_t* dSfa,
                                                   const uint32_t* dSfwGateUp,
                                                   const float* dRoute,
                                                   cutlass::float_e4m3_t* dAct8,
                                                   uint32_t* dActSf,
                                                   __nv_bfloat16* dScratch,
                                                   int M, int I, int K,
                                                   cudaStream_t stream) {
  const int N2 = 2 * I;
  auto dmap_a = make_fp8_a_desc(dA, M, K);
  auto dmap_b = make_tma_2d(dWgu, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, N2,
                            CFG_FP8_BLOCK_K, CFG_FP8_FUSED_BLOCK_N,
                            K, sizeof(cutlass::float_e4m3_t), CFG_SWZ_B);
  int sf_k_packed = (K + (int)CFG_FP8_GRAN_K_A * 4 - 1) / ((int)CFG_FP8_GRAN_K_A * 4);
  auto dmap_sfa = make_sf_desc(dSfa, M, sf_k_packed);
  auto dmap_sfb = make_tma_2d(dSfwGateUp, CU_TENSOR_MAP_DATA_TYPE_UINT32, N2, sf_k_packed,
                              CFG_FP8_FUSED_BLOCK_N, 1, N2, sizeof(uint32_t), 0);
  auto dmap_cd = make_tma_2d(dAct8, CU_TENSOR_MAP_DATA_TYPE_UINT8, I, M,
                             CFG_FP8_FUSED_BLOCK_N / 2, CFG_BLOCK_M, I, sizeof(cutlass::float_e4m3_t), CFG_SWZ_CD);

  auto kernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      CFG_FP8_GRAN_K_A, CFG_FP8_GRAN_K_B, CFG_K_ALIGNMENT,
      CFG_SHAPE_M, CFG_SHAPE_N, CFG_SHAPE_K,
      CFG_BLOCK_M, CFG_FP8_FUSED_BLOCK_N, CFG_FP8_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_FP8_FUSED_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      NumSms,
      false, true,
      deep_gemm::GemmType::Normal, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity,
      CFG_PHYSICAL_THREADS,
      true,
      Group32>;

  size_t smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NumSms, 1, 1);
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

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, nullptr, (uint32_t)M, (uint32_t)N2, (uint32_t)K,
                                dmap_a, dmap_b, dmap_sfa, dmap_sfb, dmap_cd,
                                dAct8, dActSf, dRoute, (uint32_t)I, dmap_cd));
  CHECK_CUDA(cudaGetLastError());
}

template <bool FuseSwiGLU>
void launch_bf16_gemm(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                      __nv_bfloat16* dD, int M, int N, int K, int sms,
                      cudaStream_t stream,
                      const __nv_bfloat16* gate = nullptr,
                      const float* route = nullptr) {
  switch (sms) {
    case 16: launch_bf16_gemm_sms<16, FuseSwiGLU>(dA, dB, dD, M, N, K, stream, gate, route); break;
    case 32: launch_bf16_gemm_sms<32, FuseSwiGLU>(dA, dB, dD, M, N, K, stream, gate, route); break;
    case 64: launch_bf16_gemm_sms<64, FuseSwiGLU>(dA, dB, dD, M, N, K, stream, gate, route); break;
    case 96: launch_bf16_gemm_sms<96, FuseSwiGLU>(dA, dB, dD, M, N, K, stream, gate, route); break;
    case 120: launch_bf16_gemm_sms<120, FuseSwiGLU>(dA, dB, dD, M, N, K, stream, gate, route); break;
    case 132: launch_bf16_gemm_sms<132, FuseSwiGLU>(dA, dB, dD, M, N, K, stream, gate, route); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

void launch_bf16_interleaved_swiglu(const __nv_bfloat16* dA, const __nv_bfloat16* dWgu,
                                     __nv_bfloat16* dAct, const float* dRoute,
                                     int M, int I, int K, int sms, cudaStream_t stream) {
  switch (sms) {
    case 16: launch_bf16_interleaved_swiglu_sms<16>(dA, dWgu, dAct, dRoute, M, I, K, stream); break;
    case 32: launch_bf16_interleaved_swiglu_sms<32>(dA, dWgu, dAct, dRoute, M, I, K, stream); break;
    case 64: launch_bf16_interleaved_swiglu_sms<64>(dA, dWgu, dAct, dRoute, M, I, K, stream); break;
    case 96: launch_bf16_interleaved_swiglu_sms<96>(dA, dWgu, dAct, dRoute, M, I, K, stream); break;
    case 120: launch_bf16_interleaved_swiglu_sms<120>(dA, dWgu, dAct, dRoute, M, I, K, stream); break;
    case 132: launch_bf16_interleaved_swiglu_sms<132>(dA, dWgu, dAct, dRoute, M, I, K, stream); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

void launch_fp8_gemm(const cutlass::float_e4m3_t* dA, const cutlass::float_e4m3_t* dB,
                     const uint32_t* dSfa, const uint32_t* dSfb,
                     __nv_bfloat16* dD, int M, int N, int K, int sms,
                     cudaStream_t stream) {
  switch (sms) {
    case 16: launch_fp8_gemm_sms<16>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 32: launch_fp8_gemm_sms<32>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 64: launch_fp8_gemm_sms<64>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 96: launch_fp8_gemm_sms<96>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 120: launch_fp8_gemm_sms<120>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 132: launch_fp8_gemm_sms<132>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

void launch_fp8_gemm_a32(const cutlass::float_e4m3_t* dA, const cutlass::float_e4m3_t* dB,
                         const uint32_t* dSfa, const uint32_t* dSfb,
                         __nv_bfloat16* dD, int M, int N, int K, int sms,
                         cudaStream_t stream) {
  switch (sms) {
    case 16: launch_fp8_gemm_sms<16, 32, CFG_FP8_GRAN_K_B>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 32: launch_fp8_gemm_sms<32, 32, CFG_FP8_GRAN_K_B>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 64: launch_fp8_gemm_sms<64, 32, CFG_FP8_GRAN_K_B>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 96: launch_fp8_gemm_sms<96, 32, CFG_FP8_GRAN_K_B>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 120: launch_fp8_gemm_sms<120, 32, CFG_FP8_GRAN_K_B>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 132: launch_fp8_gemm_sms<132, 32, CFG_FP8_GRAN_K_B>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

void launch_fp8_gemm_swapab(const cutlass::float_e4m3_t* dA, const cutlass::float_e4m3_t* dB,
                            const uint32_t* dSfa, const uint32_t* dSfb,
                            __nv_bfloat16* dD, int M, int N, int K, int sms,
                            cudaStream_t stream) {
  switch (sms) {
    case 16: launch_fp8_gemm_swapab_sms<16>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 32: launch_fp8_gemm_swapab_sms<32>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 64: launch_fp8_gemm_swapab_sms<64>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 96: launch_fp8_gemm_swapab_sms<96>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 120: launch_fp8_gemm_swapab_sms<120>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    case 132: launch_fp8_gemm_swapab_sms<132>(dA, dB, dSfa, dSfb, dD, M, N, K, stream); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

void launch_fp8_interleaved_swiglu_quant_fused_swapab_half(const cutlass::float_e4m3_t* dA,
                                                           const cutlass::float_e4m3_t* dWgu,
                                                           const uint32_t* dSfa,
                                                           const uint32_t* dSfwGateUp,
                                                           const float* dRoute,
                                                           cutlass::float_e4m3_t* dAct8,
                                                           uint32_t* dActSf,
                                                           __nv_bfloat16* dScratch,
                                                           int M, int I, int K, int sms,
                                                           cudaStream_t stream) {
  switch (sms) {
    case 16: launch_fp8_interleaved_swiglu_quant_fused_swapab_half_sms<16>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 32: launch_fp8_interleaved_swiglu_quant_fused_swapab_half_sms<32>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 64: launch_fp8_interleaved_swiglu_quant_fused_swapab_half_sms<64>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 96: launch_fp8_interleaved_swiglu_quant_fused_swapab_half_sms<96>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 120: launch_fp8_interleaved_swiglu_quant_fused_swapab_half_sms<120>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 132: launch_fp8_interleaved_swiglu_quant_fused_swapab_half_sms<132>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

void launch_fp8_interleaved_swiglu_quant_fused(const cutlass::float_e4m3_t* dA,
                                               const cutlass::float_e4m3_t* dWgu,
                                               const uint32_t* dSfa,
                                               const uint32_t* dSfwGateUp,
                                               const float* dRoute,
                                               cutlass::float_e4m3_t* dAct8,
                                               uint32_t* dActSf,
                                               __nv_bfloat16* dScratch,
                                               int M, int I, int K, int sms,
                                               cudaStream_t stream) {
  switch (sms) {
    case 16: launch_fp8_interleaved_swiglu_quant_fused_sms<16>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 32: launch_fp8_interleaved_swiglu_quant_fused_sms<32>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 64: launch_fp8_interleaved_swiglu_quant_fused_sms<64>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 96: launch_fp8_interleaved_swiglu_quant_fused_sms<96>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 120: launch_fp8_interleaved_swiglu_quant_fused_sms<120>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 132: launch_fp8_interleaved_swiglu_quant_fused_sms<132>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

void launch_fp8_interleaved_swiglu_quant_fused_group32(const cutlass::float_e4m3_t* dA,
                                                       const cutlass::float_e4m3_t* dWgu,
                                                       const uint32_t* dSfa,
                                                       const uint32_t* dSfwGateUp,
                                                       const float* dRoute,
                                                       cutlass::float_e4m3_t* dAct8,
                                                       uint32_t* dActSf,
                                                       __nv_bfloat16* dScratch,
                                                       int M, int I, int K, int sms,
                                                       cudaStream_t stream) {
  switch (sms) {
    case 16: launch_fp8_interleaved_swiglu_quant_fused_sms<16, true>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 32: launch_fp8_interleaved_swiglu_quant_fused_sms<32, true>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 64: launch_fp8_interleaved_swiglu_quant_fused_sms<64, true>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 96: launch_fp8_interleaved_swiglu_quant_fused_sms<96, true>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 120: launch_fp8_interleaved_swiglu_quant_fused_sms<120, true>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    case 132: launch_fp8_interleaved_swiglu_quant_fused_sms<132, true>(dA, dWgu, dSfa, dSfwGateUp, dRoute, dAct8, dActSf, dScratch, M, I, K, stream); break;
    default: std::cerr << "Unsupported --sms " << sms << "\n"; std::exit(1);
  }
}

__device__ __forceinline__ int fast_log2_ceil_device(float x) {
  if (!(x > 0.0f)) return -127;
  uint32_t bits = *reinterpret_cast<uint32_t*>(&x);
  int exp = static_cast<int>(bits >> 23) - 127;
  uint32_t man = bits & ((1u << 23) - 1u);
  return exp + (man != 0);
}

__device__ __forceinline__ float fast_pow2_device(int x) {
  uint32_t bits = static_cast<uint32_t>(x + 127) << 23;
  return *reinterpret_cast<float*>(&bits);
}

__device__ __forceinline__ float fast_rcp_device(float x) {
  float ret;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}

__device__ __forceinline__ uint8_t ue8m0_code_from_scale(float sf) {
  uint32_t bits = *reinterpret_cast<uint32_t*>(&sf);
  return static_cast<uint8_t>(bits >> 23);
}

__global__ void swiglu_interleaved_kernel(const __nv_bfloat16* gu,
                                          const float* route, __nv_bfloat16* out,
                                          int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx < total) {
    int m = idx / N;
    int n = idx - m * N;
    int gu_idx = m * (2 * N) + 2 * n;
    float g = __bfloat162float(gu[gu_idx]);
    float u = __bfloat162float(gu[gu_idx + 1]);
    float y = (g * fast_rcp_device(1.0f + __expf(-g))) * u * route[m];
    out[idx] = __float2bfloat16(y);
  }
}

__global__ void bf16_to_fp8_unit_kernel(const __nv_bfloat16* in,
                                        cutlass::float_e4m3_t* out,
                                        int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) out[idx] = cutlass::float_e4m3_t(__bfloat162float(in[idx]));
}

__device__ __forceinline__ float warp_reduce_max_device(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

template <int kGroupElems>
__global__ void swiglu_interleaved_quant_grouped_kernel(const __nv_bfloat16* gu,
                                                        const float* route, cutlass::float_e4m3_t* out_fp8,
                                                        uint32_t* out_sf, __nv_bfloat16* out_bf16,
                                                        int M, int N) {
  constexpr int kGroupsPerPack = 4;
  int pack = blockIdx.x;
  int m = blockIdx.y;
  int tid = threadIdx.x;

  __shared__ float s_act[kGroupsPerPack][kGroupElems];
  __shared__ float s_amax[kGroupsPerPack][kGroupElems];
  __shared__ float s_sf_inv[kGroupsPerPack];
  __shared__ uint32_t s_sf_code[kGroupsPerPack];

  float route_m = m < M ? route[m] : 0.0f;
  #pragma unroll
  for (int gidx = 0; gidx < kGroupsPerPack; ++gidx) {
    int group = pack * kGroupsPerPack + gidx;
    int n = group * kGroupElems + tid;
    float y = 0.0f;
    if (tid < kGroupElems && m < M && n < N) {
      int gu_idx = m * (2 * N) + 2 * n;
      float gate_v = __bfloat162float(gu[gu_idx]);
      float up_v = __bfloat162float(gu[gu_idx + 1]);
      y = (gate_v * fast_rcp_device(1.0f + __expf(-gate_v))) * up_v * route_m;
    }
    s_act[gidx][tid] = y;
    s_amax[gidx][tid] = fabsf(y);
  }
  __syncthreads();

  for (int stride = kGroupElems / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int gidx = 0; gidx < kGroupsPerPack; ++gidx) {
        s_amax[gidx][tid] = fmaxf(s_amax[gidx][tid], s_amax[gidx][tid + stride]);
      }
    }
    __syncthreads();
  }

  if (tid < kGroupsPerPack) {
    float amax = s_amax[tid][0];
    float sf = 1.0f, sf_inv = 1.0f;
    if (amax > 0.0f) {
      int exp = fast_log2_ceil_device(amax * (1.0f / 448.0f));
      sf = fast_pow2_device(exp);
      sf_inv = fast_pow2_device(-exp);
    }
    s_sf_inv[tid] = sf_inv;
    s_sf_code[tid] = pack * kGroupsPerPack * kGroupElems + tid * kGroupElems < N ? static_cast<uint32_t>(ue8m0_code_from_scale(sf)) : 0u;
  }
  __syncthreads();

  #pragma unroll
  for (int gidx = 0; gidx < kGroupsPerPack; ++gidx) {
    int group = pack * kGroupsPerPack + gidx;
    int n = group * kGroupElems + tid;
    if (tid < kGroupElems && m < M && n < N) {
      int idx = m * N + n;
      float y = s_act[gidx][tid];
      if (out_bf16) out_bf16[idx] = __float2bfloat16(y);
      out_fp8[idx] = cutlass::float_e4m3_t(y * s_sf_inv[gidx]);
    }
  }

  if (tid == 0 && m < M) {
    uint32_t sf_pack = s_sf_code[0] | (s_sf_code[1] << 8) | (s_sf_code[2] << 16) | (s_sf_code[3] << 24);
    out_sf[(size_t)pack * M + m] = sf_pack;
  }
}

__global__ void swiglu_interleaved_quant_group128_warp_kernel(const __nv_bfloat16* gu,
                                                              const float* route,
                                                              cutlass::float_e4m3_t* out_fp8,
                                                              uint32_t* out_sf,
                                                              __nv_bfloat16* out_bf16,
                                                              int M, int N) {
  constexpr int kGroupElems = 128;
  constexpr int kGroupsPerPack = 4;
  constexpr int kValsPerLane = kGroupElems / 32;
  int pack = blockIdx.x;
  int m = blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int group = pack * kGroupsPerPack + warp_id;
  int base_n = group * kGroupElems;

  __shared__ uint32_t s_sf_code[kGroupsPerPack];

  float route_m = m < M ? route[m] : 0.0f;
  float y[kValsPerLane];
  float local_amax = 0.0f;
  #pragma unroll
  for (int i = 0; i < kValsPerLane; ++i) {
    int n = base_n + lane + i * 32;
    float v = 0.0f;
    if (m < M && n < N) {
      int gu_idx = m * (2 * N) + 2 * n;
      float gate_v = __bfloat162float(gu[gu_idx]);
      float up_v = __bfloat162float(gu[gu_idx + 1]);
      v = (gate_v * fast_rcp_device(1.0f + __expf(-gate_v))) * up_v * route_m;
      local_amax = fmaxf(local_amax, fabsf(v));
    }
    y[i] = v;
  }

  float amax = warp_reduce_max_device(local_amax);
  float sf = 1.0f;
  float sf_inv = 1.0f;
  if (amax > 0.0f) {
    int exp = fast_log2_ceil_device(amax * (1.0f / 448.0f));
    sf = fast_pow2_device(exp);
    sf_inv = fast_pow2_device(-exp);
  }
  if (lane == 0) {
    s_sf_code[warp_id] = base_n < N ? static_cast<uint32_t>(ue8m0_code_from_scale(sf)) : 0u;
  }

  #pragma unroll
  for (int i = 0; i < kValsPerLane; ++i) {
    int n = base_n + lane + i * 32;
    if (m < M && n < N) {
      int idx = m * N + n;
      float v = y[i];
      if (out_bf16) out_bf16[idx] = __float2bfloat16(v);
      out_fp8[idx] = cutlass::float_e4m3_t(v * sf_inv);
    }
  }

  __syncthreads();
  if (tid == 0 && m < M) {
    uint32_t sf_pack = s_sf_code[0] | (s_sf_code[1] << 8) | (s_sf_code[2] << 16) | (s_sf_code[3] << 24);
    out_sf[(size_t)pack * M + m] = sf_pack;
  }
}


void launch_swiglu_interleaved_epilogue(const __nv_bfloat16* dGU,
                                        const float* dRoute, __nv_bfloat16* dAct,
                                        int M, int N, cudaStream_t stream) {
  int total = M * N;
  swiglu_interleaved_kernel<<<(total + 255) / 256, 256, 0, stream>>>(dGU, dRoute, dAct, M, N);
  CHECK_CUDA(cudaGetLastError());
}

void launch_bf16_to_fp8_unit(const __nv_bfloat16* dIn, cutlass::float_e4m3_t* dOut,
                             int total, cudaStream_t stream) {
  bf16_to_fp8_unit_kernel<<<(total + 255) / 256, 256, 0, stream>>>(dIn, dOut, total);
  CHECK_CUDA(cudaGetLastError());
}

void launch_swiglu_interleaved_quant_epilogue(const __nv_bfloat16* dGU,
                                              const float* dRoute, cutlass::float_e4m3_t* dAct8,
                                              uint32_t* dActSf, __nv_bfloat16* dAct16,
                                              int M, int N, cudaStream_t stream) {
  dim3 grid((N + 511) / 512, M, 1);
  swiglu_interleaved_quant_grouped_kernel<128><<<grid, 128, 0, stream>>>(dGU, dRoute, dAct8, dActSf, dAct16, M, N);
  CHECK_CUDA(cudaGetLastError());
}

void launch_swiglu_interleaved_quant_warp_epilogue(const __nv_bfloat16* dGU,
                                                   const float* dRoute, cutlass::float_e4m3_t* dAct8,
                                                   uint32_t* dActSf, __nv_bfloat16* dAct16,
                                                   int M, int N, cudaStream_t stream) {
  dim3 grid((N + 511) / 512, M, 1);
  swiglu_interleaved_quant_group128_warp_kernel<<<grid, 128, 0, stream>>>(dGU, dRoute, dAct8, dActSf, dAct16, M, N);
  CHECK_CUDA(cudaGetLastError());
}

void launch_swiglu_interleaved_quant_half64_epilogue(const __nv_bfloat16* dGU,
                                                     const float* dRoute, cutlass::float_e4m3_t* dAct8,
                                                     uint32_t* dActSf, __nv_bfloat16* dAct16,
                                                     int M, int N, cudaStream_t stream) {
  dim3 grid((N + 255) / 256, M, 1);
  swiglu_interleaved_quant_grouped_kernel<64><<<grid, 64, 0, stream>>>(dGU, dRoute, dAct8, dActSf, dAct16, M, N);
  CHECK_CUDA(cudaGetLastError());
}

void launch_swiglu_interleaved_quant_group32_epilogue(const __nv_bfloat16* dGU,
                                                      const float* dRoute, cutlass::float_e4m3_t* dAct8,
                                                      uint32_t* dActSf, __nv_bfloat16* dAct16,
                                                      int M, int N, cudaStream_t stream) {
  dim3 grid((N + 127) / 128, M, 1);
  swiglu_interleaved_quant_grouped_kernel<32><<<grid, 32, 0, stream>>>(dGU, dRoute, dAct8, dActSf, dAct16, M, N);
  CHECK_CUDA(cudaGetLastError());
}

void launch_bf16_interleaved_swiglu_down(const __nv_bfloat16* dA,
                                          const __nv_bfloat16* dWgu,
                                          const __nv_bfloat16* dWd,
                                          const float* dRoute,
                                          __nv_bfloat16* dAct,
                                          __nv_bfloat16* dOut,
                                          int M, int N, int K, int sms, cudaStream_t stream) {
  launch_bf16_interleaved_swiglu(dA, dWgu, dAct, dRoute, M, N, K, sms, stream);
  launch_bf16_gemm<false>(dAct, dWd, dOut, M, K, N, sms, stream);
}

void launch_fp8_interleaved_swiglu_noquant_down(const cutlass::float_e4m3_t* dA,
                                                const cutlass::float_e4m3_t* dWgu,
                                                const cutlass::float_e4m3_t* dWd,
                                                const uint32_t* dSfa,
                                                const uint32_t* dSfwGateUp,
                                                const uint32_t* dActUnitSf,
                                                const uint32_t* dSfwDown,
                                                const float* dRoute,
                                                __nv_bfloat16* dGU,
                                                __nv_bfloat16* dAct,
                                                cutlass::float_e4m3_t* dAct8,
                                                __nv_bfloat16* dOut,
                                                int M, int N, int K, int sms, cudaStream_t stream) {
  launch_fp8_gemm(dA, dWgu, dSfa, dSfwGateUp, dGU, M, 2 * N, K, sms, stream);
  launch_swiglu_interleaved_epilogue(dGU, dRoute, dAct, M, N, stream);
  launch_bf16_to_fp8_unit(dAct, dAct8, M * N, stream);
  launch_fp8_gemm(dAct8, dWd, dActUnitSf, dSfwDown, dOut, M, K, N, sms, stream);
}

void launch_fp8_interleaved_swiglu_quant_down(const cutlass::float_e4m3_t* dA,
                                              const cutlass::float_e4m3_t* dWgu,
                                              const cutlass::float_e4m3_t* dWd,
                                              const uint32_t* dSfa,
                                              const uint32_t* dSfwGateUp,
                                              const uint32_t* dSfwDown,
                                              const float* dRoute,
                                              __nv_bfloat16* dGU,
                                              cutlass::float_e4m3_t* dAct8,
                                              uint32_t* dActSf,
                                              __nv_bfloat16* dOut,
                                              int M, int N, int K, int sms, cudaStream_t stream) {
  launch_fp8_gemm(dA, dWgu, dSfa, dSfwGateUp, dGU, M, 2 * N, K, sms, stream);
  launch_swiglu_interleaved_quant_epilogue(dGU, dRoute, dAct8, dActSf, nullptr, M, N, stream);
  launch_fp8_gemm(dAct8, dWd, dActSf, dSfwDown, dOut, M, K, N, sms, stream);
}

void launch_fp8_interleaved_swiglu_quant_warp_down(const cutlass::float_e4m3_t* dA,
                                                   const cutlass::float_e4m3_t* dWgu,
                                                   const cutlass::float_e4m3_t* dWd,
                                                   const uint32_t* dSfa,
                                                   const uint32_t* dSfwGateUp,
                                                   const uint32_t* dSfwDown,
                                                   const float* dRoute,
                                                   __nv_bfloat16* dGU,
                                                   cutlass::float_e4m3_t* dAct8,
                                                   uint32_t* dActSf,
                                                   __nv_bfloat16* dOut,
                                                   int M, int N, int K, int sms, cudaStream_t stream) {
  launch_fp8_gemm(dA, dWgu, dSfa, dSfwGateUp, dGU, M, 2 * N, K, sms, stream);
  launch_swiglu_interleaved_quant_warp_epilogue(dGU, dRoute, dAct8, dActSf, nullptr, M, N, stream);
  launch_fp8_gemm(dAct8, dWd, dActSf, dSfwDown, dOut, M, K, N, sms, stream);
}

void launch_fp8_interleaved_swiglu_quant_group32_down(const cutlass::float_e4m3_t* dA,
                                                      const cutlass::float_e4m3_t* dWgu,
                                                      const cutlass::float_e4m3_t* dWd,
                                                      const uint32_t* dSfa,
                                                      const uint32_t* dSfwGateUp,
                                                      const uint32_t* dSfwDown,
                                                      const float* dRoute,
                                                      __nv_bfloat16* dGU,
                                                      cutlass::float_e4m3_t* dAct8,
                                                      uint32_t* dActSfGroup32,
                                                      __nv_bfloat16* dOut,
                                                      int M, int N, int K, int sms, cudaStream_t stream) {
  launch_fp8_gemm(dA, dWgu, dSfa, dSfwGateUp, dGU, M, 2 * N, K, sms, stream);
  launch_swiglu_interleaved_quant_group32_epilogue(dGU, dRoute, dAct8, dActSfGroup32, nullptr, M, N, stream);
  launch_fp8_gemm_a32(dAct8, dWd, dActSfGroup32, dSfwDown, dOut, M, K, N, sms, stream);
}

void launch_fp8_interleaved_swiglu_quant_fused_down(const cutlass::float_e4m3_t* dA,
                                                    const cutlass::float_e4m3_t* dWgu,
                                                    const cutlass::float_e4m3_t* dWd,
                                                    const uint32_t* dSfa,
                                                    const uint32_t* dSfwGateUp,
                                                    const uint32_t* dSfwDown,
                                                    const float* dRoute,
                                                    cutlass::float_e4m3_t* dAct8,
                                                    uint32_t* dActSf,
                                                    __nv_bfloat16* dScratch,
                                                    __nv_bfloat16* dOut,
                                                    int M, int N, int K, int sms, cudaStream_t stream) {
  CHECK_CUDA(cudaMemsetAsync(dActSf, 0, sizeof(uint32_t) * (size_t)M * ((N + 511) / 512), stream));
  launch_fp8_interleaved_swiglu_quant_fused(dA, dWgu, dSfa, dSfwGateUp, dRoute,
                                            dAct8, dActSf, dScratch, M, N, K, sms, stream);
  launch_fp8_gemm(dAct8, dWd, dActSf, dSfwDown, dOut, M, K, N, sms, stream);
}

void launch_fp8_interleaved_swiglu_quant_fused_swapab_half_down(const cutlass::float_e4m3_t* dA,
                                                               const cutlass::float_e4m3_t* dWgu,
                                                               const cutlass::float_e4m3_t* dWd,
                                                               const uint32_t* dSfa,
                                                               const uint32_t* dSfwGateUp,
                                                               const uint32_t* dSfwDown,
                                                               const float* dRoute,
                                                               cutlass::float_e4m3_t* dAct8,
                                                               uint32_t* dActSfGroup32,
                                                               __nv_bfloat16* dScratch,
                                                               __nv_bfloat16* dOut,
                                                               int M, int N, int K, int sms,
                                                               cudaStream_t stream) {
  CHECK_CUDA(cudaMemsetAsync(dActSfGroup32, 0, sizeof(uint32_t) * (size_t)M * ((N + 127) / 128), stream));
  launch_fp8_interleaved_swiglu_quant_fused_swapab_half(dA, dWgu, dSfa, dSfwGateUp, dRoute,
                                                        dAct8, dActSfGroup32, dScratch,
                                                        M, N, K, sms, stream);
  launch_fp8_gemm_a32(dAct8, dWd, dActSfGroup32, dSfwDown, dOut, M, K, N, sms, stream);
}

void launch_fp8_interleaved_swiglu_quant_fused_group32_down(const cutlass::float_e4m3_t* dA,
                                                           const cutlass::float_e4m3_t* dWgu,
                                                           const cutlass::float_e4m3_t* dWd,
                                                           const uint32_t* dSfa,
                                                           const uint32_t* dSfwGateUp,
                                                           const uint32_t* dSfwDown,
                                                           const float* dRoute,
                                                           cutlass::float_e4m3_t* dAct8,
                                                           uint32_t* dActSfGroup32,
                                                           __nv_bfloat16* dScratch,
                                                           __nv_bfloat16* dOut,
                                                           int M, int N, int K, int sms,
                                                           cudaStream_t stream) {
  CHECK_CUDA(cudaMemsetAsync(dActSfGroup32, 0, sizeof(uint32_t) * (size_t)M * ((N + 127) / 128), stream));
  launch_fp8_interleaved_swiglu_quant_fused_group32(dA, dWgu, dSfa, dSfwGateUp, dRoute,
                                                    dAct8, dActSfGroup32, dScratch,
                                                    M, N, K, sms, stream);
  launch_fp8_gemm_a32(dAct8, dWd, dActSfGroup32, dSfwDown, dOut, M, K, N, sms, stream);
}

// ======================= Host helpers =======================
static float median_ms(cudaStream_t s, int warm, int it, const std::function<void()>& fn) {
  for (int i = 0; i < warm; ++i) fn();
  CHECK_CUDA(cudaStreamSynchronize(s));
  cudaEvent_t b, e;
  CHECK_CUDA(cudaEventCreate(&b));
  CHECK_CUDA(cudaEventCreate(&e));
  std::vector<float> ts(it);
  for (int i = 0; i < it; ++i) {
    CHECK_CUDA(cudaEventRecord(b, s));
    fn();
    CHECK_CUDA(cudaEventRecord(e, s));
    CHECK_CUDA(cudaEventSynchronize(e));
    CHECK_CUDA(cudaEventElapsedTime(&ts[i], b, e));
  }
  std::sort(ts.begin(), ts.end());
  CHECK_CUDA(cudaEventDestroy(b));
  CHECK_CUDA(cudaEventDestroy(e));
  return ts[it / 2];
}

static float bf_round(float v) {
  __nv_bfloat16 b = __float2bfloat16(v);
  return __bfloat162float(b);
}

static uint32_t unit_ue8m0_pack() {
  // pack_ue8m0_to_int([1,1,1,1]) -> four exponent bytes 127.
  return 0x7f7f7f7fu;
}

static int host_fast_log2_ceil(float x) {
  if (!(x > 0.0f)) return -127;
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  int exp = static_cast<int>(bits >> 23) - 127;
  uint32_t man = bits & ((1u << 23) - 1u);
  return exp + (man != 0);
}

static float host_fast_pow2(int x) {
  x = std::max(-127, std::min(127, x));
  uint32_t bits = static_cast<uint32_t>(x + 127) << 23;
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

static uint8_t host_ue8m0_code_from_scale(float sf) {
  uint32_t bits;
  std::memcpy(&bits, &sf, sizeof(bits));
  return static_cast<uint8_t>(bits >> 23);
}

static float host_scale_from_ue8m0_code(uint8_t code) {
  uint32_t bits = static_cast<uint32_t>(code) << 23;
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

static void fill_unit_sf(std::vector<uint32_t>& sf, int mn, int K) {
  int sf_k_packed = (K + (int)CFG_FP8_GRAN_K_A * 4 - 1) / ((int)CFG_FP8_GRAN_K_A * 4);
  sf.assign((size_t)mn * sf_k_packed, unit_ue8m0_pack());
}

static cutlass::float_e4m3_t to_fp8(float x) {
  return cutlass::float_e4m3_t(x);
}

static float from_fp8(cutlass::float_e4m3_t x) {
  return static_cast<float>(x);
}

struct DiffStats {
  float max_abs = 0.f;
  float max_rel = 0.f;
  double mean_abs = 0.0;
};

static DiffStats diff_stats(const std::vector<__nv_bfloat16>& got, const std::vector<float>& ref) {
  DiffStats s;
  for (size_t i = 0; i < got.size(); ++i) {
    float g = __bfloat162float(got[i]);
    float a = std::fabs(g - ref[i]);
    float r = a / (std::fabs(ref[i]) + 1e-6f);
    s.max_abs = std::max(s.max_abs, a);
    s.max_rel = std::max(s.max_rel, r);
    s.mean_abs += a;
  }
  s.mean_abs /= std::max<size_t>(got.size(), 1);
  return s;
}

static DiffStats diff_stats_gu_interleaved_range(const std::vector<__nv_bfloat16>& got,
                                                 const std::vector<float>& gate,
                                                 const std::vector<float>& up,
                                                 int M, int N, int m_begin, int m_end) {
  DiffStats s;
  m_begin = std::max(0, m_begin);
  m_end = std::min(M, m_end);
  for (int m = m_begin; m < m_end; ++m) {
    for (int n = 0; n < N; ++n) {
      const size_t gu = (size_t)m * (2 * N) + 2 * n;
      const float g = __bfloat162float(got[gu]);
      const float u = __bfloat162float(got[gu + 1]);
      const float ref_g = bf_round(gate[(size_t)m * N + n]);
      const float ref_u = bf_round(up[(size_t)m * N + n]);
      const float ag = std::fabs(g - ref_g);
      const float au = std::fabs(u - ref_u);
      s.max_abs = std::max(s.max_abs, std::max(ag, au));
      s.max_rel = std::max(s.max_rel, ag / (std::fabs(ref_g) + 1e-6f));
      s.max_rel = std::max(s.max_rel, au / (std::fabs(ref_u) + 1e-6f));
      s.mean_abs += ag + au;
    }
  }
  s.mean_abs /= std::max<size_t>((size_t)(m_end - m_begin) * 2 * N, 1);
  return s;
}

static DiffStats diff_stats_gu_interleaved(const std::vector<__nv_bfloat16>& got,
                                           const std::vector<float>& gate,
                                           const std::vector<float>& up,
                                           int M, int N) {
  return diff_stats_gu_interleaved_range(got, gate, up, M, N, 0, M);
}

static DiffStats diff_stats_fp8(const std::vector<cutlass::float_e4m3_t>& got,
                                const std::vector<cutlass::float_e4m3_t>& ref) {
  DiffStats s;
  for (size_t i = 0; i < got.size(); ++i) {
    float g = from_fp8(got[i]);
    float rref = from_fp8(ref[i]);
    float a = std::fabs(g - rref);
    float r = a / (std::fabs(rref) + 1e-6f);
    s.max_abs = std::max(s.max_abs, a);
    s.max_rel = std::max(s.max_rel, r);
    s.mean_abs += a;
  }
  s.mean_abs /= std::max<size_t>(got.size(), 1);
  return s;
}

static DiffStats diff_stats_fp8_dequant_grouped(const std::vector<cutlass::float_e4m3_t>& got,
                                                const std::vector<uint32_t>& got_sf,
                                                const std::vector<float>& ref,
                                                int M, int N, int group_cols) {
  DiffStats s;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int group = n / group_cols;
      uint32_t pack = got_sf[(size_t)(group / 4) * M + m];
      uint8_t code = static_cast<uint8_t>((pack >> ((group % 4) * 8)) & 0xffu);
      float sf = host_scale_from_ue8m0_code(code);
      size_t idx = (size_t)m * N + n;
      float g = from_fp8(got[idx]) * sf;
      float a = std::fabs(g - ref[idx]);
      float r = a / (std::fabs(ref[idx]) + 1e-6f);
      s.max_abs = std::max(s.max_abs, a);
      s.max_rel = std::max(s.max_rel, r);
      s.mean_abs += a;
    }
  }
  s.mean_abs /= std::max<size_t>((size_t)M * N, 1);
  return s;
}

static DiffStats diff_stats_fp8_dequant(const std::vector<cutlass::float_e4m3_t>& got,
                                        const std::vector<uint32_t>& got_sf,
                                        const std::vector<float>& ref,
                                        int M, int N) {
  return diff_stats_fp8_dequant_grouped(got, got_sf, ref, M, N, 128);
}

static void dump_fp8_mismatches(const char* name,
                                const std::vector<cutlass::float_e4m3_t>& got,
                                const std::vector<cutlass::float_e4m3_t>& ref,
                                const std::vector<uint32_t>& got_sf,
                                const std::vector<uint32_t>& ref_sf,
                                int M, int N, int group_cols, int limit = 8) {
  int printed = 0;
  for (int m = 0; m < M && printed < limit; ++m) {
    for (int n = 0; n < N && printed < limit; ++n) {
      const size_t idx = (size_t)m * N + n;
      if (got[idx].storage == ref[idx].storage) continue;
      const int group = n / group_cols;
      const uint32_t got_pack = got_sf[(size_t)(group / 4) * M + m];
      const uint32_t ref_pack = ref_sf[(size_t)(group / 4) * M + m];
      const uint8_t got_code = static_cast<uint8_t>((got_pack >> ((group % 4) * 8)) & 0xffu);
      const uint8_t ref_code = static_cast<uint8_t>((ref_pack >> ((group % 4) * 8)) & 0xffu);
      std::printf("[%s mismatch] m=%d n=%d got=%g ref=%g got_byte=0x%02x ref_byte=0x%02x got_sf=0x%02x ref_sf=0x%02x\n",
                  name, m, n, from_fp8(got[idx]), from_fp8(ref[idx]),
                  (unsigned)got[idx].storage, (unsigned)ref[idx].storage,
                  (unsigned)got_code, (unsigned)ref_code);
      ++printed;
    }
  }
}

static void print_diff(const char* name, const DiffStats& s) {
  std::printf("[%s] max_abs=%.6g max_rel=%.6g mean_abs=%.6g\n",
              name, s.max_abs, s.max_rel, s.mean_abs);
}

static void host_swiglu_ref(const std::vector<float>& A, const std::vector<float>& Wg,
                            const std::vector<float>& Wu, const std::vector<float>& route,
                            int M, int N, int K,
                            std::vector<float>& gate, std::vector<float>& up,
                            std::vector<float>& act) {
  gate.assign((size_t)M * N, 0.f);
  up.assign((size_t)M * N, 0.f);
  act.assign((size_t)M * N, 0.f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double sg = 0.0, su = 0.0;
      for (int k = 0; k < K; ++k) {
        sg += double(A[(size_t)m * K + k]) * double(Wg[(size_t)n * K + k]);
        su += double(A[(size_t)m * K + k]) * double(Wu[(size_t)n * K + k]);
      }
      float g = bf_round((float)sg);
      float u = bf_round((float)su);
      gate[(size_t)m * N + n] = (float)sg;
      up[(size_t)m * N + n] = (float)su;
      act[(size_t)m * N + n] = bf_round((g * (1.0f / (1.0f + std::exp(-g)))) * u * route[m]);
    }
  }
}

static void host_swiglu_quant_input_from_gate_up(const std::vector<float>& gate,
                                                 const std::vector<float>& up,
                                                 const std::vector<float>& route,
                                                 int M, int N,
                                                 std::vector<float>& act) {
  act.assign((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      size_t idx = (size_t)m * N + n;
      float g = bf_round(gate[idx]);
      float u = bf_round(up[idx]);
      act[idx] = (g * (1.0f / (1.0f + std::exp(-g)))) * u * route[m];
    }
  }
}

static void host_unit_fp8_activation_ref(const std::vector<float>& act,
                                         std::vector<cutlass::float_e4m3_t>& act8,
                                         std::vector<float>& act8_dequant) {
  act8.assign(act.size(), cutlass::float_e4m3_t(0.0f));
  act8_dequant.assign(act.size(), 0.0f);
  for (size_t i = 0; i < act.size(); ++i) {
    act8[i] = cutlass::float_e4m3_t(act[i]);
    act8_dequant[i] = from_fp8(act8[i]);
  }
}

static void host_quantize_activation_grouped_ref(const std::vector<float>& act,
                                                 int M, int N, int group_cols,
                                                 std::vector<cutlass::float_e4m3_t>& act8,
                                                 std::vector<float>& act8_dequant,
                                                 std::vector<uint32_t>& act_sf) {
  const int sf_k_packed = (N + group_cols * 4 - 1) / (group_cols * 4);
  act8.assign((size_t)M * N, cutlass::float_e4m3_t(0.0f));
  act8_dequant.assign((size_t)M * N, 0.0f);
  act_sf.assign((size_t)M * sf_k_packed, 0u);
  for (int m = 0; m < M; ++m) {
    for (int group = 0; group < (N + group_cols - 1) / group_cols; ++group) {
      int n0 = group * group_cols;
      float amax = 0.0f;
      for (int n = n0; n < std::min(n0 + group_cols, N); ++n)
        amax = std::max(amax, std::fabs(act[(size_t)m * N + n]));
      float sf = 1.0f;
      float sf_inv = 1.0f;
      if (amax > 0.0f) {
        int exp = host_fast_log2_ceil(amax * (1.0f / 448.0f));
        sf = host_fast_pow2(exp);
        sf_inv = host_fast_pow2(-exp);
      }
      int pack_idx = group / 4;
      int byte_idx = group % 4;
      uint32_t code = static_cast<uint32_t>(host_ue8m0_code_from_scale(sf));
      act_sf[(size_t)pack_idx * M + m] |= code << (byte_idx * 8);
      for (int n = n0; n < std::min(n0 + group_cols, N); ++n) {
        auto v = cutlass::float_e4m3_t(act[(size_t)m * N + n] * sf_inv);
        act8[(size_t)m * N + n] = v;
        act8_dequant[(size_t)m * N + n] = from_fp8(v) * sf;
      }
    }
  }
}

static void host_quantize_activation_half64_ref(const std::vector<float>& act,
                                                int M, int N,
                                                std::vector<cutlass::float_e4m3_t>& act8,
                                                std::vector<float>& act8_dequant,
                                                std::vector<uint32_t>& act_sf) {
  host_quantize_activation_grouped_ref(act, M, N, 64, act8, act8_dequant, act_sf);
}

static void host_quantize_activation_32_ref(const std::vector<float>& act,
                                            int M, int N,
                                            std::vector<cutlass::float_e4m3_t>& act8,
                                            std::vector<float>& act8_dequant,
                                            std::vector<uint32_t>& act_sf) {
  host_quantize_activation_grouped_ref(act, M, N, 32, act8, act8_dequant, act_sf);
}


static void host_quantize_activation_ref(const std::vector<float>& act,
                                         int M, int N,
                                         std::vector<cutlass::float_e4m3_t>& act8,
                                         std::vector<float>& act8_dequant,
                                         std::vector<uint32_t>& act_sf) {
  const int sf_k_packed = (N + 511) / 512;
  act8.assign((size_t)M * N, cutlass::float_e4m3_t(0.0f));
  act8_dequant.assign((size_t)M * N, 0.0f);
  act_sf.assign((size_t)M * sf_k_packed, 0u);
  for (int m = 0; m < M; ++m) {
    for (int group = 0; group < (N + 127) / 128; ++group) {
      int n0 = group * 128;
      float amax = 0.0f;
      for (int i = 0; i < 128 && n0 + i < N; ++i) {
        amax = std::max(amax, std::fabs(act[(size_t)m * N + n0 + i]));
      }
      float sf = 1.0f;
      float sf_inv = 1.0f;
      if (amax > 0.0f) {
        int exp = host_fast_log2_ceil(amax * (1.0f / 448.0f));
        sf = host_fast_pow2(exp);
        sf_inv = host_fast_pow2(-exp);
      }
      int pack_idx = group / 4;
      int byte_idx = group % 4;
      uint32_t code = static_cast<uint32_t>(host_ue8m0_code_from_scale(sf));
      act_sf[(size_t)pack_idx * M + m] |= code << (byte_idx * 8);
      float stored_sf = host_scale_from_ue8m0_code(static_cast<uint8_t>(code));
      for (int i = 0; i < 128 && n0 + i < N; ++i) {
        size_t idx = (size_t)m * N + n0 + i;
        act8[idx] = cutlass::float_e4m3_t(act[idx] * sf_inv);
        act8_dequant[idx] = from_fp8(act8[idx]) * stored_sf;
      }
    }
  }
}

static void host_down_ref(const std::vector<float>& act, const std::vector<float>& Wd,
                          int M, int K, int N, std::vector<float>& out) {
  out.assign((size_t)M * K, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      double sum = 0.0;
      for (int n = 0; n < N; ++n) {
        sum += double(act[(size_t)m * N + n]) * double(Wd[(size_t)k * N + n]);
      }
      out[(size_t)m * K + k] = (float)sum;
    }
  }
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  const bool run_bf16 = opt.mode == "bf16" || opt.mode == "both";
  const bool run_fp8 = opt.mode == "fp8" || opt.mode == "both";
  std::printf("DeepGEMM microkernel BF16/FP8 SwiGLU interleaved-gateup 1CTA M=%d N=%d K=%d sms=%d threads=%u warmup=%d iters=%d mode=%s\n",
              opt.M, opt.N, opt.K, opt.sms, CFG_PHYSICAL_THREADS, opt.warmup, opt.iters, opt.mode.c_str());
  CHECK_CU(cuInit(0));

  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> rdist(0.5f, 1.0f);

  std::vector<float> hA((size_t)opt.M * opt.K), hWg((size_t)opt.N * opt.K), hWu((size_t)opt.N * opt.K), hWd((size_t)opt.K * opt.N), hRoute(opt.M);
  for (auto& v : hA) v = dist(gen);
  for (auto& v : hWg) v = dist(gen);
  for (auto& v : hWu) v = dist(gen);
  for (auto& v : hWd) v = dist(gen);
  for (auto& v : hRoute) v = rdist(gen);

  std::fprintf(stderr, "[host] references starting\n");

  std::vector<__nv_bfloat16> A16(hA.size()), Wg16(hWg.size()), Wu16(hWu.size()), Wd16(hWd.size());
  std::vector<float> A16_ref(hA.size()), Wg16_ref(hWg.size()), Wu16_ref(hWu.size()), Wd16_ref(hWd.size());
  for (size_t i = 0; i < hA.size(); ++i) { A16[i] = __float2bfloat16(hA[i]); A16_ref[i] = __bfloat162float(A16[i]); }
  for (size_t i = 0; i < hWg.size(); ++i) { Wg16[i] = __float2bfloat16(hWg[i]); Wg16_ref[i] = __bfloat162float(Wg16[i]); }
  for (size_t i = 0; i < hWu.size(); ++i) { Wu16[i] = __float2bfloat16(hWu[i]); Wu16_ref[i] = __bfloat162float(Wu16[i]); }
  for (size_t i = 0; i < hWd.size(); ++i) { Wd16[i] = __float2bfloat16(hWd[i]); Wd16_ref[i] = __bfloat162float(Wd16[i]); }

  std::vector<cutlass::float_e4m3_t> A8(hA.size()), Wg8(hWg.size()), Wu8(hWu.size()), Wd8(hWd.size());
  std::vector<float> A8_ref(hA.size()), Wg8_ref(hWg.size()), Wu8_ref(hWu.size()), Wd8_ref(hWd.size());
  for (size_t i = 0; i < hA.size(); ++i) { A8[i] = to_fp8(hA[i]); A8_ref[i] = from_fp8(A8[i]); }
  for (size_t i = 0; i < hWg.size(); ++i) { Wg8[i] = to_fp8(hWg[i]); Wg8_ref[i] = from_fp8(Wg8[i]); }
  for (size_t i = 0; i < hWu.size(); ++i) { Wu8[i] = to_fp8(hWu[i]); Wu8_ref[i] = from_fp8(Wu8[i]); }
  for (size_t i = 0; i < hWd.size(); ++i) { Wd8[i] = to_fp8(hWd[i]); Wd8_ref[i] = from_fp8(Wd8[i]); }

  std::vector<__nv_bfloat16> Wgu16((size_t)2 * opt.N * opt.K);
  std::vector<cutlass::float_e4m3_t> Wgu8((size_t)2 * opt.N * opt.K);
  for (int n = 0; n < opt.N; ++n) {
    for (int k = 0; k < opt.K; ++k) {
      size_t src = (size_t)n * opt.K + k;
      Wgu16[((size_t)2 * n) * opt.K + k] = Wg16[src];
      Wgu16[((size_t)2 * n + 1) * opt.K + k] = Wu16[src];
      Wgu8[((size_t)2 * n) * opt.K + k] = Wg8[src];
      Wgu8[((size_t)2 * n + 1) * opt.K + k] = Wu8[src];
    }
  }
  std::vector<float> ref_bf16_gate, ref_bf16_up, ref_bf16_act, ref_bf16_down;
  host_swiglu_ref(A16_ref, Wg16_ref, Wu16_ref, hRoute, opt.M, opt.N, opt.K,
                  ref_bf16_gate, ref_bf16_up, ref_bf16_act);
  host_down_ref(ref_bf16_act, Wd16_ref, opt.M, opt.K, opt.N, ref_bf16_down);

  std::vector<float> ref8_gate, ref8_up, ref8_act, ref8_noquant_act_dequant, ref8_noquant_down;
  std::vector<float> ref8_quant_input, ref8_act_dequant, ref8_half64_act_dequant, ref8_group32_act_dequant;
  std::vector<float> ref8_down, ref8_group32_down;
  std::vector<cutlass::float_e4m3_t> ref8_noquant_act8, ref8_act8, ref8_half64_act8, ref8_group32_act8;
  std::vector<uint32_t> ref8_act_sf, ref8_half64_act_sf, ref8_group32_act_sf;
  host_swiglu_ref(A8_ref, Wg8_ref, Wu8_ref, hRoute, opt.M, opt.N, opt.K, ref8_gate, ref8_up, ref8_act);
  host_unit_fp8_activation_ref(ref8_act, ref8_noquant_act8, ref8_noquant_act_dequant);
  host_down_ref(ref8_noquant_act_dequant, Wd8_ref, opt.M, opt.K, opt.N, ref8_noquant_down);
  host_swiglu_quant_input_from_gate_up(ref8_gate, ref8_up, hRoute, opt.M, opt.N, ref8_quant_input);
  host_quantize_activation_ref(ref8_quant_input, opt.M, opt.N, ref8_act8, ref8_act_dequant, ref8_act_sf);
  host_quantize_activation_half64_ref(ref8_quant_input, opt.M, opt.N, ref8_half64_act8, ref8_half64_act_dequant, ref8_half64_act_sf);
  host_quantize_activation_32_ref(ref8_quant_input, opt.M, opt.N, ref8_group32_act8, ref8_group32_act_dequant, ref8_group32_act_sf);
  host_down_ref(ref8_act_dequant, Wd8_ref, opt.M, opt.K, opt.N, ref8_down);
  host_down_ref(ref8_group32_act_dequant, Wd8_ref, opt.M, opt.K, opt.N, ref8_group32_down);

  std::vector<uint32_t> hSfa, hSfwGateUp, hSfwDown;
  fill_unit_sf(hSfa, opt.M, opt.K);
  fill_unit_sf(hSfwGateUp, 2 * opt.N, opt.K);
  fill_unit_sf(hSfwDown, opt.K, opt.N);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  __nv_bfloat16 *dA16 = nullptr, *dWgu16 = nullptr, *dWd16 = nullptr;
  cutlass::float_e4m3_t *dA8 = nullptr, *dWgu8 = nullptr, *dWd8 = nullptr, *dAct8 = nullptr;
  uint32_t *dSfa = nullptr, *dSfwGateUp = nullptr, *dSfwDown = nullptr, *dActSf = nullptr, *dActSfHalf64 = nullptr, *dActSfGroup32 = nullptr, *dActUnitSf = nullptr;
  float* dRoute = nullptr;
  __nv_bfloat16 *dGU = nullptr, *dAct = nullptr, *dDown = nullptr;
  const size_t bytes_a16 = sizeof(__nv_bfloat16) * A16.size();
  const size_t bytes_wgu16 = sizeof(__nv_bfloat16) * Wgu16.size();
  const size_t bytes_wd16 = sizeof(__nv_bfloat16) * Wd16.size();
  const size_t bytes_a8 = sizeof(cutlass::float_e4m3_t) * A8.size();
  const size_t bytes_wgu8 = sizeof(cutlass::float_e4m3_t) * Wgu8.size();
  const size_t bytes_wd8 = sizeof(cutlass::float_e4m3_t) * Wd8.size();
  const size_t bytes_act8 = sizeof(cutlass::float_e4m3_t) * (size_t)opt.M * opt.N;
  const size_t bytes_act = sizeof(__nv_bfloat16) * (size_t)opt.M * opt.N;
  const size_t bytes_gu = sizeof(__nv_bfloat16) * (size_t)opt.M * (2 * opt.N);
  const size_t bytes_down = sizeof(__nv_bfloat16) * (size_t)opt.M * opt.K;
  CHECK_CUDA(cudaMalloc(&dA16, bytes_a16));
  CHECK_CUDA(cudaMalloc(&dWgu16, bytes_wgu16));
  CHECK_CUDA(cudaMalloc(&dWd16, bytes_wd16));
  CHECK_CUDA(cudaMalloc(&dA8, bytes_a8));
  CHECK_CUDA(cudaMalloc(&dWgu8, bytes_wgu8));
  CHECK_CUDA(cudaMalloc(&dWd8, bytes_wd8));
  CHECK_CUDA(cudaMalloc(&dAct8, bytes_act8));
  CHECK_CUDA(cudaMalloc(&dSfa, sizeof(uint32_t) * hSfa.size()));
  CHECK_CUDA(cudaMalloc(&dSfwGateUp, sizeof(uint32_t) * hSfwGateUp.size()));
  CHECK_CUDA(cudaMalloc(&dSfwDown, sizeof(uint32_t) * hSfwDown.size()));
  CHECK_CUDA(cudaMalloc(&dActSf, sizeof(uint32_t) * (size_t)opt.M * ((opt.N + 511) / 512)));
  CHECK_CUDA(cudaMalloc(&dActSfHalf64, sizeof(uint32_t) * (size_t)opt.M * ((opt.N + 255) / 256)));
  CHECK_CUDA(cudaMalloc(&dActSfGroup32, sizeof(uint32_t) * (size_t)opt.M * ((opt.N + 127) / 128)));
  CHECK_CUDA(cudaMalloc(&dActUnitSf, sizeof(uint32_t) * (size_t)opt.M * ((opt.N + 511) / 512)));
  CHECK_CUDA(cudaMalloc(&dRoute, sizeof(float) * hRoute.size()));
  CHECK_CUDA(cudaMalloc(&dGU, bytes_gu));
  CHECK_CUDA(cudaMalloc(&dAct, bytes_act));
  CHECK_CUDA(cudaMalloc(&dDown, bytes_down));

  CHECK_CUDA(cudaMemcpy(dA16, A16.data(), bytes_a16, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWgu16, Wgu16.data(), bytes_wgu16, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWd16, Wd16.data(), bytes_wd16, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA8, A8.data(), bytes_a8, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWgu8, Wgu8.data(), bytes_wgu8, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWd8, Wd8.data(), bytes_wd8, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dSfa, hSfa.data(), sizeof(uint32_t) * hSfa.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dSfwGateUp, hSfwGateUp.data(), sizeof(uint32_t) * hSfwGateUp.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dSfwDown, hSfwDown.data(), sizeof(uint32_t) * hSfwDown.size(), cudaMemcpyHostToDevice));
  std::vector<uint32_t> hActUnitSf((size_t)opt.M * ((opt.N + 511) / 512), unit_ue8m0_pack());
  CHECK_CUDA(cudaMemcpy(dActUnitSf, hActUnitSf.data(), sizeof(uint32_t) * hActUnitSf.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dRoute, hRoute.data(), sizeof(float) * hRoute.size(), cudaMemcpyHostToDevice));

  std::vector<__nv_bfloat16> hAct((size_t)opt.M * opt.N), hDown((size_t)opt.M * opt.K);
  std::vector<cutlass::float_e4m3_t> hAct8((size_t)opt.M * opt.N);
  std::vector<uint32_t> hActSf((size_t)opt.M * ((opt.N + 511) / 512));
  std::vector<uint32_t> hActSfHalf64((size_t)opt.M * ((opt.N + 255) / 256));
  std::vector<uint32_t> hActSfGroup32((size_t)opt.M * ((opt.N + 127) / 128));

  if (run_bf16) {
    launch_bf16_interleaved_swiglu_down(dA16, dWgu16, dWd16, dRoute,
                                         dAct, dDown, opt.M, opt.N, opt.K, opt.sms, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
    print_diff("bf16 full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
    float ms = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_bf16_interleaved_swiglu_down(dA16, dWgu16, dWd16, dRoute,
                                           dAct, dDown, opt.M, opt.N, opt.K, opt.sms, stream);
    });
    std::printf("[perf] bf16_full_down median_ms=%.6f\n", ms);
  }

  if (run_fp8) {
    if (opt.check_swapab) {
      launch_fp8_gemm_swapab(dA8, dWgu8, dSfa, dSfwGateUp, dGU,
                             opt.M, 2 * opt.N, opt.K, opt.sms, stream);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      std::vector<__nv_bfloat16> hGU((size_t)opt.M * (2 * opt.N));
      CHECK_CUDA(cudaMemcpy(hGU.data(), dGU, bytes_gu, cudaMemcpyDeviceToHost));
      print_diff("fp8 gateup swapab GU vs fp8-ref", diff_stats_gu_interleaved(hGU, ref8_gate, ref8_up, opt.M, opt.N));
      if (opt.M > (int)CFG_BLOCK_M) {
        print_diff("fp8 gateup swapab GU first M-block", diff_stats_gu_interleaved_range(hGU, ref8_gate, ref8_up, opt.M, opt.N, 0, CFG_BLOCK_M));
        print_diff("fp8 gateup swapab GU later M-blocks", diff_stats_gu_interleaved_range(hGU, ref8_gate, ref8_up, opt.M, opt.N, CFG_BLOCK_M, opt.M));
      }

      CHECK_CUDA(cudaMemsetAsync(dAct8, 0, bytes_act8, stream));
      CHECK_CUDA(cudaMemsetAsync(dActSfGroup32, 0, sizeof(uint32_t) * hActSfGroup32.size(), stream));
      launch_fp8_interleaved_swiglu_quant_fused_swapab_half(dA8, dWgu8, dSfa, dSfwGateUp, dRoute,
                                                            dAct8, dActSfGroup32, dGU,
                                                            opt.M, opt.N, opt.K, opt.sms, stream);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaMemcpy(hAct8.data(), dAct8, bytes_act8, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(hActSfGroup32.data(), dActSfGroup32, sizeof(uint32_t) * hActSfGroup32.size(), cudaMemcpyDeviceToHost));
      DiffStats swapab_half_raw = diff_stats_fp8(hAct8, ref8_group32_act8);
      print_diff("fp8 swapab half fused act raw vs group32-ref", swapab_half_raw);
      print_diff("fp8 swapab half fused act dequant vs group32-input",
                 diff_stats_fp8_dequant_grouped(hAct8, hActSfGroup32, ref8_quant_input, opt.M, opt.N, 32));
      bool swapab_half_sf_match = hActSfGroup32 == ref8_group32_act_sf;
      std::printf("[fp8 swapab half fused act sf32] match_ref=%s packs=%zu\n",
                  swapab_half_sf_match ? "true" : "false", hActSfGroup32.size());
      if (swapab_half_raw.max_abs != 0.0f || !swapab_half_sf_match)
        dump_fp8_mismatches("fp8 swapab half fused", hAct8, ref8_group32_act8,
                            hActSfGroup32, ref8_group32_act_sf, opt.M, opt.N, 32);

      CHECK_CUDA(cudaMemsetAsync(dDown, 0, bytes_down, stream));
      launch_fp8_gemm_a32(dAct8, dWd8, dActSfGroup32, dSfwDown, dDown,
                          opt.M, opt.K, opt.N, opt.sms, stream);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
      print_diff("fp8 swapab half fused full down vs group32-ref", diff_stats(hDown, ref8_group32_down));
      print_diff("fp8 swapab half fused full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
      float ms_swapab_half_fused = median_ms(stream, opt.warmup, opt.iters, [&]() {
        launch_fp8_interleaved_swiglu_quant_fused_swapab_half_down(
            dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown, dRoute,
            dAct8, dActSfGroup32, dGU, dDown, opt.M, opt.N, opt.K, opt.sms, stream);
      });
      std::printf("[perf] fp8_swapab_half_fused_full_down median_ms=%.6f\n", ms_swapab_half_fused);

      CHECK_CUDA(cudaMemsetAsync(dAct8, 0, bytes_act8, stream));
      CHECK_CUDA(cudaMemsetAsync(dActSfHalf64, 0, sizeof(uint32_t) * hActSfHalf64.size(), stream));
      launch_swiglu_interleaved_quant_half64_epilogue(dGU, dRoute, dAct8, dActSfHalf64, nullptr,
                                                      opt.M, opt.N, stream);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaMemcpy(hAct8.data(), dAct8, bytes_act8, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(hActSfHalf64.data(), dActSfHalf64, sizeof(uint32_t) * hActSfHalf64.size(), cudaMemcpyDeviceToHost));
      print_diff("fp8 swapab scratch half64 act raw vs half64-ref", diff_stats_fp8(hAct8, ref8_half64_act8));
      print_diff("fp8 swapab scratch half64 act dequant vs half64-input",
                 diff_stats_fp8_dequant_grouped(hAct8, hActSfHalf64, ref8_quant_input, opt.M, opt.N, 64));
      std::printf("[fp8 swapab scratch half64 act sf] match_ref=%s packs=%zu\n",
                  (hActSfHalf64 == ref8_half64_act_sf) ? "true" : "false", hActSfHalf64.size());
    }

    launch_fp8_interleaved_swiglu_noquant_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dActUnitSf, dSfwDown,
                                                dRoute, dGU, dAct, dAct8, dDown,
                                                opt.M, opt.N, opt.K, opt.sms, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
    print_diff("fp8 noquant full down vs fp8-noquant-ref", diff_stats(hDown, ref8_noquant_down));
    print_diff("fp8 noquant full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
    float ms_noquant = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_fp8_interleaved_swiglu_noquant_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dActUnitSf, dSfwDown,
                                                  dRoute, dGU, dAct, dAct8, dDown,
                                                  opt.M, opt.N, opt.K, opt.sms, stream);
    });
    std::printf("[perf] fp8_noquant_full_down median_ms=%.6f\n", ms_noquant);

    launch_fp8_interleaved_swiglu_quant_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                              dRoute, dGU, dAct8, dActSf, dDown,
                                              opt.M, opt.N, opt.K, opt.sms, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(hAct8.data(), dAct8, bytes_act8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hActSf.data(), dActSf, sizeof(uint32_t) * hActSf.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
    print_diff("fp8 quant act raw vs fp8-quant-ref", diff_stats_fp8(hAct8, ref8_act8));
    print_diff("fp8 quant act dequant vs quant-input", diff_stats_fp8_dequant(hAct8, hActSf, ref8_quant_input, opt.M, opt.N));
    print_diff("fp8 quant full down vs fp8-quant-ref", diff_stats(hDown, ref8_down));
    print_diff("fp8 quant full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
    bool sf_match = hActSf == ref8_act_sf;
    std::printf("[fp8 act sf] match_ref=%s packs=%zu\n", sf_match ? "true" : "false", hActSf.size());
    float ms_quant_only = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_swiglu_interleaved_quant_epilogue(dGU, dRoute, dAct8, dActSf, nullptr,
                                               opt.M, opt.N, stream);
    });
    std::printf("[perf] fp8_quant_only median_ms=%.6f\n", ms_quant_only);
    float ms_quant = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_fp8_interleaved_swiglu_quant_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                dRoute, dGU, dAct8, dActSf, dDown,
                                                opt.M, opt.N, opt.K, opt.sms, stream);
    });
    std::printf("[perf] fp8_quant_full_down median_ms=%.6f\n", ms_quant);

    launch_fp8_interleaved_swiglu_quant_warp_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                   dRoute, dGU, dAct8, dActSf, dDown,
                                                   opt.M, opt.N, opt.K, opt.sms, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(hAct8.data(), dAct8, bytes_act8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hActSf.data(), dActSf, sizeof(uint32_t) * hActSf.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
    print_diff("fp8 quant warp act raw vs fp8-quant-ref", diff_stats_fp8(hAct8, ref8_act8));
    print_diff("fp8 quant warp act dequant vs quant-input", diff_stats_fp8_dequant(hAct8, hActSf, ref8_quant_input, opt.M, opt.N));
    print_diff("fp8 quant warp full down vs fp8-quant-ref", diff_stats(hDown, ref8_down));
    print_diff("fp8 quant warp full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
    bool warp_sf_match = hActSf == ref8_act_sf;
    std::printf("[fp8 warp act sf] match_ref=%s packs=%zu\n", warp_sf_match ? "true" : "false", hActSf.size());
    float ms_quant_warp_only = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_swiglu_interleaved_quant_warp_epilogue(dGU, dRoute, dAct8, dActSf, nullptr,
                                                    opt.M, opt.N, stream);
    });
    std::printf("[perf] fp8_quant_warp_only median_ms=%.6f\n", ms_quant_warp_only);
    float ms_quant_warp = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_fp8_interleaved_swiglu_quant_warp_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                     dRoute, dGU, dAct8, dActSf, dDown,
                                                     opt.M, opt.N, opt.K, opt.sms, stream);
    });
    std::printf("[perf] fp8_quant_warp_full_down median_ms=%.6f\n", ms_quant_warp);

    launch_fp8_interleaved_swiglu_quant_group32_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                      dRoute, dGU, dAct8, dActSfGroup32, dDown,
                                                      opt.M, opt.N, opt.K, opt.sms, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(hAct8.data(), dAct8, bytes_act8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hActSfGroup32.data(), dActSfGroup32, sizeof(uint32_t) * hActSfGroup32.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
    print_diff("fp8 quant group32 act raw vs group32-ref", diff_stats_fp8(hAct8, ref8_group32_act8));
    print_diff("fp8 quant group32 act dequant vs group32-input",
               diff_stats_fp8_dequant_grouped(hAct8, hActSfGroup32, ref8_quant_input, opt.M, opt.N, 32));
    print_diff("fp8 quant group32 full down vs group32-ref", diff_stats(hDown, ref8_group32_down));
    print_diff("fp8 quant group32 full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
    bool group32_sf_match = hActSfGroup32 == ref8_group32_act_sf;
    std::printf("[fp8 group32 act sf] match_ref=%s packs=%zu\n",
                group32_sf_match ? "true" : "false", hActSfGroup32.size());
    float ms_quant_group32_only = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_swiglu_interleaved_quant_group32_epilogue(dGU, dRoute, dAct8, dActSfGroup32, nullptr,
                                                       opt.M, opt.N, stream);
    });
    std::printf("[perf] fp8_quant_group32_only median_ms=%.6f\n", ms_quant_group32_only);
    float ms_quant_group32 = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_fp8_interleaved_swiglu_quant_group32_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                        dRoute, dGU, dAct8, dActSfGroup32, dDown,
                                                        opt.M, opt.N, opt.K, opt.sms, stream);
    });
    std::printf("[perf] fp8_quant_group32_full_down median_ms=%.6f\n", ms_quant_group32);

    launch_fp8_interleaved_swiglu_quant_fused_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                    dRoute, dAct8, dActSf, dGU, dDown,
                                                    opt.M, opt.N, opt.K, opt.sms, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(hAct8.data(), dAct8, bytes_act8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hActSf.data(), dActSf, sizeof(uint32_t) * hActSf.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
    print_diff("fp8 quant fused act raw vs fp8-quant-ref", diff_stats_fp8(hAct8, ref8_act8));
    print_diff("fp8 quant fused act dequant vs quant-input",
               diff_stats_fp8_dequant(hAct8, hActSf, ref8_quant_input, opt.M, opt.N));
    print_diff("fp8 quant fused full down vs fp8-quant-ref", diff_stats(hDown, ref8_down));
    print_diff("fp8 quant fused full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
    bool fused_sf_match = hActSf == ref8_act_sf;
    std::printf("[fp8 fused act sf] match_ref=%s packs=%zu\n", fused_sf_match ? "true" : "false", hActSf.size());
    float ms_quant_fused = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_fp8_interleaved_swiglu_quant_fused_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                      dRoute, dAct8, dActSf, dGU, dDown,
                                                      opt.M, opt.N, opt.K, opt.sms, stream);
    });
    std::printf("[perf] fp8_quant_fused_full_down median_ms=%.6f\n", ms_quant_fused);

    launch_fp8_interleaved_swiglu_quant_fused_group32_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                            dRoute, dAct8, dActSfGroup32, dGU, dDown,
                                                            opt.M, opt.N, opt.K, opt.sms, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(hAct8.data(), dAct8, bytes_act8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hActSfGroup32.data(), dActSfGroup32, sizeof(uint32_t) * hActSfGroup32.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDown.data(), dDown, bytes_down, cudaMemcpyDeviceToHost));
    print_diff("fp8 quant fused group32 act raw vs group32-ref", diff_stats_fp8(hAct8, ref8_group32_act8));
    print_diff("fp8 quant fused group32 act dequant vs group32-input",
               diff_stats_fp8_dequant_grouped(hAct8, hActSfGroup32, ref8_quant_input, opt.M, opt.N, 32));
    print_diff("fp8 quant fused group32 full down vs group32-ref", diff_stats(hDown, ref8_group32_down));
    print_diff("fp8 quant fused group32 full down vs bf16-ref", diff_stats(hDown, ref_bf16_down));
    bool fused_group32_sf_match = hActSfGroup32 == ref8_group32_act_sf;
    std::printf("[fp8 fused group32 act sf] match_ref=%s packs=%zu\n",
                fused_group32_sf_match ? "true" : "false", hActSfGroup32.size());
    float ms_quant_fused_group32 = median_ms(stream, opt.warmup, opt.iters, [&]() {
      launch_fp8_interleaved_swiglu_quant_fused_group32_down(dA8, dWgu8, dWd8, dSfa, dSfwGateUp, dSfwDown,
                                                              dRoute, dAct8, dActSfGroup32, dGU, dDown,
                                                              opt.M, opt.N, opt.K, opt.sms, stream);
    });
    std::printf("[perf] fp8_quant_fused_group32_full_down median_ms=%.6f\n", ms_quant_fused_group32);

  }

  CHECK_CUDA(cudaFree(dA16));
  CHECK_CUDA(cudaFree(dWgu16));
  CHECK_CUDA(cudaFree(dWd16));
  CHECK_CUDA(cudaFree(dA8));
  CHECK_CUDA(cudaFree(dWgu8));
  CHECK_CUDA(cudaFree(dWd8));
  CHECK_CUDA(cudaFree(dAct8));
  CHECK_CUDA(cudaFree(dSfa));
  CHECK_CUDA(cudaFree(dSfwGateUp));
  CHECK_CUDA(cudaFree(dSfwDown));
  CHECK_CUDA(cudaFree(dActSf));
  CHECK_CUDA(cudaFree(dActSfHalf64));
  CHECK_CUDA(cudaFree(dActSfGroup32));
  CHECK_CUDA(cudaFree(dActUnitSf));
  CHECK_CUDA(cudaFree(dRoute));
  CHECK_CUDA(cudaFree(dGU));
  CHECK_CUDA(cudaFree(dAct));
  CHECK_CUDA(cudaFree(dDown));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
