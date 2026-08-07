// bench_gateup_gather.cu — compare two gate/up implementations at the megakernel's
// full-batch scale. Both M (batch_size) and SM count are now runtime-configurable:
//
//   GEMM shape: M=<arg> (COMPUTE_BATCH_SIZE), K=2048 (hidden), N=6144 (2*intermediate)
//   Config A: int4-vectorized gather (combine_input -> contiguous input_buf) + 2-CTA GEMM
//   Config B: hardware gather4 fused into A load + 1-CTA GEMM
//
// "Include input prep": A's timing = gather-to-input_buf kernel + 2-CTA GEMM.
//                        B's gather is fused into the GEMM (no separate prep).
// A launches clusterDim=2 (so SMS must be even for A), B clusterDim=1
// (1-CTA gather4 cannot run under a clusterDim=2 launch — verified separately).
//
// Usage:
//   ./bench_gateup_gather [M] [SMS] [mode]
//     M    : batch size (rows), default 256
//     SMS  : SM / grid size (one of 1,2,4,8,16,32,64,132,148), default 32
//     mode : 'A' = only config A, 'B' = only config B, else both (default)
//   Examples:
//     ./bench_gateup_gather               # M=256 SMS=32 both
//     ./bench_gateup_gather 2048          # M=2048 SMS=32 both
//     ./bench_gateup_gather 2048 64       # M=2048 SMS=64 both
//     ./bench_gateup_gather 2048 64 B     # M=2048 SMS=64 config B only
//
// Build:
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        bench_gateup_gather.cu -o bench_gateup_gather -lcuda

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/types.cuh>
#include "sm100_bf16_gemm_dg_gather.cuh"   // config A (nullptr idx = plain 2-CTA) + config B (gather4 1-CTA)

#define CHECK_CUDA(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" @ "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  std::cerr<<"CU error "<<s<<" @ "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(1);} } while(0)

// ---- megakernel-aligned gate/up shape. K/N fixed; M is runtime. ----
static constexpr int KK = 2048;   // hidden
static constexpr int NN = 6144;   // 2 * intermediate (concat gate+up)

static constexpr uint32_t BLK_M = 128, BLK_N = 128, BLK_K = 64;
static constexpr uint32_t SWZ = 128, NSTAGES = 4;
static constexpr uint32_t NON_EPI = 128, EPI = 128, PHYS = 800;
static constexpr uint32_t KALIGN = 128;

static CUtensorMap make_tma_2d(const void* ptr, int gi, int go, int si_in, int so,
                               int gostr, int esz, int swz) {
  CUtensorMap tm; int si = swz ? swz/esz : si_in;
  const cuuint64_t gd[2]={(cuuint64_t)gi,(cuuint64_t)go};
  const cuuint32_t sd[2]={(cuuint32_t)si,(cuuint32_t)so};
  const cuuint64_t gs[1]={(cuuint64_t)gostr*esz};
  const cuuint32_t es[2]={1,1};
  CUtensorMapSwizzle sw = swz==128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE;
  CHECK_CU(cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gd,gs,sd,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,sw,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}

// int4-vectorized gather: input_buf[m] = combine_input[a_idx[m]] (M rows x K).
__global__ void gather_int4_kernel(const __nv_bfloat16* __restrict__ src,
                                   const int* __restrict__ a_idx,
                                   __nv_bfloat16* __restrict__ dst, int M, int K) {
  int row = blockIdx.x;
  if (row >= M) return;
  int src_row = a_idx[row];
  const int4* s = reinterpret_cast<const int4*>(src + (size_t)src_row * K);
  int4* d = reinterpret_cast<int4*>(dst + (size_t)row * K);
  int vecK = K / 8;  // 8 bf16 per int4
  for (int i = threadIdx.x; i < vecK; i += blockDim.x) d[i] = s[i];
}

// kNumSMs is a compile-time template parameter of the kernel, so it is
// parametrized by NSMS here and dispatched from main() based on the runtime arg.
template <uint32_t MCAST, uint32_t NSMS>
auto get_kernel() {
  constexpr bool MCAST_ON_A = false;
  return &deep_gemm::sm100_bf16_gemm_gather_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K, 0u,0u,0u,
      BLK_M, BLK_N, BLK_K, 1u, SWZ, SWZ, SWZ, NSTAGES,
      NON_EPI, EPI, MCAST, MCAST_ON_A, NSMS, KALIGN,
      false, false, deep_gemm::GemmType::Normal, false, cutlass::bfloat16_t,
      100ul, false, PHYS>;
}

template <uint32_t MCAST, class K_t>
void launch(K_t kernel, int cluster_dim, int M, int nsms,
            const __nv_bfloat16* dA, const __nv_bfloat16* dB,
            __nv_bfloat16* dD, const int* a_idx, const CUtensorMap& a, const CUtensorMap& a_row,
            const CUtensorMap& b, const CUtensorMap& cd, cudaStream_t s) {
  int smem = 227*1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(nsms,1,1); cfg.blockDim=dim3(PHYS,1,1);
  cfg.dynamicSmemBytes=smem; cfg.stream=s;
  cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x=cluster_dim; at[0].val.clusterDim.y=1; at[0].val.clusterDim.z=1;
  cfg.attrs=at; cfg.numAttrs=1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, (int*)nullptr,(uint32_t)M,(uint32_t)NN,(uint32_t)KK,
      a,b,cd,(const cutlass::bfloat16_t*)nullptr,(const float*)nullptr,(uint32_t)NN, a_idx, a_row));
}

template<class F> float median_ms(F fn, int warm, int it, cudaStream_t s) {
  for(int i=0;i<warm;++i) fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  std::vector<float> ts(it); cudaEvent_t b,e; cudaEventCreate(&b); cudaEventCreate(&e);
  for(int i=0;i<it;++i){cudaEventRecord(b,s); fn(); cudaEventRecord(e,s);
    cudaEventSynchronize(e); cudaEventElapsedTime(&ts[i],b,e);}
  std::sort(ts.begin(),ts.end()); cudaEventDestroy(b); cudaEventDestroy(e); return ts[it/2];
}

// Run config A and/or B for a given (compile-time) SM count and runtime M.
template <uint32_t NSMS>
void run_configs(int M, char mode, cudaStream_t s,
                 const __nv_bfloat16* dAscat, __nv_bfloat16* dInbuf,
                 const __nv_bfloat16* dB, __nv_bfloat16* dD, const int* dIdx,
                 const CUtensorMap& a_in, const CUtensorMap& a_row,
                 const CUtensorMap& b_mc2, const CUtensorMap& b_mc1,
                 const CUtensorMap& cd_d, double flop) {
  // Config A: int4 gather to input_buf (clusterDim=2 GEMM). Timing includes gather.
  auto k2 = get_kernel<2, NSMS>();
  auto fnA = [&]{
    gather_int4_kernel<<<M, 256, 0, s>>>(dAscat, dIdx, dInbuf, M, KK);
    launch<2>(k2, 2, M, (int)NSMS, dInbuf, dB, dD, nullptr, a_in, a_row, b_mc2, cd_d, s);
  };
  // Config B: gather4 fused into A-load, 1-CTA GEMM (clusterDim=1). No prep.
  auto k1 = get_kernel<1, NSMS>();
  auto fnB = [&]{
    launch<1>(k1, 1, M, (int)NSMS, dAscat, dB, dD, dIdx, a_in, a_row, b_mc1, cd_d, s);
  };

  // Fewer iters when profiling under ncu (mode A/B); full timing when running both.
  int warm = (mode == '*') ? 20 : 3;
  int iter = (mode == '*') ? 100 : 5;
  if (mode != 'B') {
    if ((NSMS & 1u) != 0)
      fprintf(stderr,"[A] WARNING: SMS=%u is odd; clusterDim=2 requires an even grid.\n", NSMS);
    float ta = median_ms(fnA, warm, iter, s);
    fprintf(stderr,"[A] int4-gather + 2-CTA : %.4f ms  %.1f TFLOPS\n", ta, flop/(ta*1e-3)/1e12);
  }
  if (mode != 'A') {
    float tb = median_ms(fnB, warm, iter, s);
    fprintf(stderr,"[B] gather4 + 1-CTA     : %.4f ms  %.1f TFLOPS\n", tb, flop/(tb*1e-3)/1e12);
  }
}

int main(int argc, char** argv) {
  // args: [M] [SMS] [mode]
  int M    = (argc > 1) ? atoi(argv[1]) : 256;
  int nsms = (argc > 2) ? atoi(argv[2]) : 32;
  char mode = (argc > 3) ? argv[3][0] : '*';
  if (M <= 0) { fprintf(stderr,"invalid M=%d\n", M); return 1; }

  CHECK_CU(cuInit(0));
  fprintf(stderr,"gate/up bench: M=%d K=%d N=%d SMs=%d mode=%c\n", M, KK, NN, nsms, mode);

  std::mt19937 g(0); std::uniform_real_distribution<float> d(-1,1);
  std::vector<__nv_bfloat16> hA((size_t)M*KK), hB((size_t)NN*KK);
  for(auto&x:hA)x=__float2bfloat16(d(g)); for(auto&x:hB)x=__float2bfloat16(d(g));
  std::vector<int> perm(M); std::iota(perm.begin(),perm.end(),0); std::shuffle(perm.begin(),perm.end(),g);
  std::vector<__nv_bfloat16> hAscat((size_t)M*KK);
  for(int m=0;m<M;++m)for(int k=0;k<KK;++k)hAscat[(size_t)perm[m]*KK+k]=hA[(size_t)m*KK+k];

  __nv_bfloat16 *dAscat,*dInbuf,*dB,*dD; int* dIdx;
  CHECK_CUDA(cudaMalloc(&dAscat,(size_t)M*KK*2)); CHECK_CUDA(cudaMalloc(&dInbuf,(size_t)M*KK*2));
  CHECK_CUDA(cudaMalloc(&dB,(size_t)NN*KK*2)); CHECK_CUDA(cudaMalloc(&dD,(size_t)M*NN*2));
  CHECK_CUDA(cudaMalloc(&dIdx,(size_t)M*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dAscat,hAscat.data(),(size_t)M*KK*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,hB.data(),(size_t)NN*KK*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dIdx,perm.data(),(size_t)M*sizeof(int),cudaMemcpyHostToDevice));

  // Descriptors. A (both configs): box outer = BLOCK_M (A never multicast).
  // B: multicast-on-N. Config A (2-CTA) loads BLOCK_N/2 per CTA; config B (1-CTA)
  // loads BLOCK_N. Build B descriptors per config with the correct smem box.
  auto a_in    = make_tma_2d(dInbuf, KK, M, BLK_K, BLK_M, KK, 2, SWZ);
  auto a_row   = make_tma_2d(dAscat, KK, M, BLK_K, 1,     KK, 2, SWZ);
  auto b_mc2   = make_tma_2d(dB, KK, NN, BLK_K, BLK_N/2, KK, 2, SWZ);  // 2-CTA: LOAD_BLOCK_N=64
  auto b_mc1   = make_tma_2d(dB, KK, NN, BLK_K, BLK_N,   KK, 2, SWZ);  // 1-CTA: LOAD_BLOCK_N=128
  auto cd_d    = make_tma_2d(dD, NN, M, SWZ/2, 128,      NN, 2, SWZ);

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  double flop = 2.0*M*NN*KK;

  // Dispatch on the runtime SM count into a compile-time NSMS instantiation.
  switch (nsms) {
    case 1:   run_configs<1>  (M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 2:   run_configs<2>  (M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 4:   run_configs<4>  (M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 8:   run_configs<8>  (M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 16:  run_configs<16> (M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 32:  run_configs<32> (M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 64:  run_configs<64> (M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 132: run_configs<132>(M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    case 148: run_configs<148>(M,mode,s,dAscat,dInbuf,dB,dD,dIdx,a_in,a_row,b_mc2,b_mc1,cd_d,flop); break;
    default:
      fprintf(stderr,"unsupported SMS=%d (supported: 1,2,4,8,16,32,64,132,148)\n", nsms);
      return 1;
  }
  return 0;
}
