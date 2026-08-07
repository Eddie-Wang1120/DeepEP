// bench_gateup_swiglu.cu — perf comparison of gate/up + SwiGLU across 4 configs:
//   [1] 1-CTA concat      : GEMM(N=2I) plain store GU[M,2I] + separate SwiGLU pass
//   [2] 2-CTA concat      : same, 2-CTA multicast GEMM
//   [3] 1-CTA interleaved : GEMM(N=2I) fused SwiGLU epilogue -> act[M,I] (quack-style)
//   [4] 2-CTA interleaved : same, 2-CTA multicast GEMM
//
// concat's timing INCLUDES the separate elementwise SwiGLU pass (GU[M,2I]->act[M,I]),
// because interleaved fuses that work into the GEMM epilogue — so this is the fair
// end-to-end gate/up+SwiGLU cost for each layout.
//
// Both M (num_tokens) and SM count are runtime-configurable, mirroring
// bench_gateup_gather.cu (NSMS is a compile-time kernel template, dispatched from
// the runtime arg).
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        bench_gateup_swiglu.cu -o bench_gateup_swiglu -lcuda
//
// Usage:
//   ./bench_gateup_swiglu [M] [SMS] [mode]
//     M    : num_tokens (rows), default 256
//     SMS  : SM / grid size (1,2,4,8,16,32,64,132,148), default 32
//     mode : 'c' = concat only, 'i' = interleaved only, else all four (default)

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
#include "sm100_bf16_gemm_dg_copy.cuh"   // sm100_bf16_gemm_impl (+ kFuseSwiGLUInterleaved)

#define CHECK_CUDA(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" @ "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  std::cerr<<"CU error "<<s<<" @ "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(1);} } while(0)

// ---- megakernel-aligned gate/up shape. K/N fixed; M runtime. ----
static constexpr int KK = 2048;   // hidden
static constexpr int II = 3072;   // intermediate
static constexpr int NN = 2 * II; // 6144 (concat / interleaved GU width)

static constexpr uint32_t BLK_M = 128, BLK_N = 128, BLK_K = 64;
static constexpr uint32_t SWZ = 128, NSTAGES = 4;
static constexpr uint32_t NON_EPI = 128, EPI = 128, PHYS = 800;
static constexpr uint32_t KALIGN = 128;
static constexpr uint32_t STORE_BLK_N = SWZ / sizeof(cutlass::bfloat16_t); // 64

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

// Separate SwiGLU pass for the CONCAT layout: GU cols [0,I)=gate, [I,2I)=up.
//   act[m,i] = silu(GU[m,i]) * GU[m,I+i] * route[m]
// int4(8-bf16)-vectorized, matching the megakernel's elementwise fold.
__global__ void swiglu_concat_kernel(const __nv_bfloat16* __restrict__ GU,
                                     __nv_bfloat16* __restrict__ act,
                                     const float* __restrict__ route, int M, int I) {
  int vpr = I >> 3;                       // int4 chunks per row
  int total = M * vpr;
  for (int v = blockIdx.x*blockDim.x + threadIdx.x; v < total; v += gridDim.x*blockDim.x) {
    int row = v / vpr, cv = v - row*vpr, col = cv << 3;
    const int64_t gu = (int64_t)row * (2*I);
    const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&GU[gu + col]);
    const __nv_bfloat162* up = reinterpret_cast<const __nv_bfloat162*>(&GU[gu + I + col]);
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&act[(int64_t)row*I + col]);
    float rw = route[row];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 g2 = __bfloat1622float2(gp[j]);
      float2 u2 = __bfloat1622float2(up[j]);
      float a0 = (g2.x*(1.0f/(1.0f+__expf(-g2.x))))*u2.x*rw;
      float a1 = (g2.y*(1.0f/(1.0f+__expf(-g2.y))))*u2.y*rw;
      op[j] = __float22bfloat162_rn(make_float2(a0, a1));
    }
  }
}

// kNumSMs is a compile-time template param; parametrize by NSMS and dispatch from main().
template <uint32_t MCAST, uint32_t NSMS, bool INTERLEAVE>
auto get_kernel() {
  constexpr bool MCAST_ON_A = false;
  return &deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K, 0u,0u,0u,
      BLK_M, BLK_N, BLK_K, 1u, SWZ, SWZ, SWZ, NSTAGES,
      NON_EPI, EPI, MCAST, MCAST_ON_A, NSMS, KALIGN,
      false, false, deep_gemm::GemmType::Normal, false, cutlass::bfloat16_t,
      100ul, /*kFuseSwiGLU=*/false, /*kFuseSwiGLUInterleaved=*/INTERLEAVE, PHYS>;
}

template <class K_t>
void launch(K_t kernel, int cluster_dim, int M, int nsms,
            const __nv_bfloat16* dA, const CUtensorMap& a,
            const CUtensorMap& b, const CUtensorMap& cd,
            const float* route, uint32_t stride_n, cudaStream_t s) {
  int smem = 227*1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(nsms,1,1); cfg.blockDim=dim3(PHYS,1,1);
  cfg.dynamicSmemBytes=smem; cfg.stream=s;
  cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x=cluster_dim; at[0].val.clusterDim.y=1; at[0].val.clusterDim.z=1;
  cfg.attrs=at; cfg.numAttrs=1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, (int*)nullptr,(uint32_t)M,(uint32_t)NN,(uint32_t)KK,
      a,b,cd,(const cutlass::bfloat16_t*)nullptr, route, stride_n));
}

template<class F> float median_ms(F fn, int warm, int it, cudaStream_t s) {
  for(int i=0;i<warm;++i) fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  std::vector<float> ts(it); cudaEvent_t b,e; cudaEventCreate(&b); cudaEventCreate(&e);
  for(int i=0;i<it;++i){cudaEventRecord(b,s); fn(); cudaEventRecord(e,s);
    cudaEventSynchronize(e); cudaEventElapsedTime(&ts[i],b,e);}
  std::sort(ts.begin(),ts.end()); cudaEventDestroy(b); cudaEventDestroy(e); return ts[it/2];
}

template <uint32_t NSMS>
void run_configs(int M, char mode, cudaStream_t s,
                 const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                 __nv_bfloat16* dGU, __nv_bfloat16* dAct, const float* dRoute,
                 const CUtensorMap& a_in,
                 const CUtensorMap& b_mc1, const CUtensorMap& b_mc2,
                 const CUtensorMap& cd_gu, const CUtensorMap& cd_act, double flop) {
  // kernels
  auto k1_concat = get_kernel<1, NSMS, false>();
  auto k2_concat = get_kernel<2, NSMS, false>();
  auto k1_intlv  = get_kernel<1, NSMS, true>();
  auto k2_intlv  = get_kernel<2, NSMS, true>();

  int sw_threads = 256;
  int sw_blocks  = NSMS;   // saturate the SMs we're benchmarking against
  auto swiglu_pass = [&]{
    swiglu_concat_kernel<<<sw_blocks, sw_threads, 0, s>>>(dGU, dAct, dRoute, M, II);
  };

  // concat = plain GEMM (store GU[M,2I]) + separate SwiGLU pass.
  auto fn1c = [&]{ launch(k1_concat, 1, M, (int)NSMS, dA, a_in, b_mc1, cd_gu, dRoute, (uint32_t)NN, s); swiglu_pass(); };
  auto fn2c = [&]{ launch(k2_concat, 2, M, (int)NSMS, dA, a_in, b_mc2, cd_gu, dRoute, (uint32_t)NN, s); swiglu_pass(); };
  // interleaved = GEMM with fused SwiGLU epilogue -> act[M,I]. stride_n=II (unused by interleaved store but passed).
  auto fn1i = [&]{ launch(k1_intlv, 1, M, (int)NSMS, dA, a_in, b_mc1, cd_act, dRoute, (uint32_t)II, s); };
  auto fn2i = [&]{ launch(k2_intlv, 2, M, (int)NSMS, dA, a_in, b_mc2, cd_act, dRoute, (uint32_t)II, s); };

  int warm = 20, iter = 100;
  auto report = [&](const char* tag, float t){
    fprintf(stderr, "  %-22s : %.4f ms  %.1f TFLOPS\n", tag, t, flop/(t*1e-3)/1e12);
  };
  bool even = ((NSMS & 1u) == 0);
  if (mode != 'i') {
    report("[1] 1-CTA concat",      median_ms(fn1c, warm, iter, s));
    if (even) report("[2] 2-CTA concat", median_ms(fn2c, warm, iter, s));
    else fprintf(stderr,"  [2] 2-CTA concat       : skipped (SMS odd, needs even grid)\n");
  }
  if (mode != 'c') {
    report("[3] 1-CTA interleaved", median_ms(fn1i, warm, iter, s));
    if (even) report("[4] 2-CTA interleaved", median_ms(fn2i, warm, iter, s));
    else fprintf(stderr,"  [4] 2-CTA interleaved  : skipped (SMS odd, needs even grid)\n");
  }
}

int main(int argc, char** argv) {
  int M    = (argc > 1) ? atoi(argv[1]) : 256;
  int nsms = (argc > 2) ? atoi(argv[2]) : 32;
  char mode = (argc > 3) ? argv[3][0] : '*';
  if (M <= 0) { fprintf(stderr,"invalid M=%d\n", M); return 1; }

  CHECK_CU(cuInit(0));
  fprintf(stderr,"gate/up+SwiGLU bench: M=%d K=%d I=%d N(2I)=%d SMs=%d mode=%c\n",
          M, KK, II, NN, nsms, mode);
  fprintf(stderr,"  (concat timing includes the separate SwiGLU elementwise pass)\n");

  std::mt19937 g(0); std::uniform_real_distribution<float> d(-1,1);
  std::vector<__nv_bfloat16> hA((size_t)M*KK), hB((size_t)NN*KK);
  for(auto&x:hA)x=__float2bfloat16(d(g)); for(auto&x:hB)x=__float2bfloat16(d(g));
  std::vector<float> hRoute(M); for(auto&x:hRoute)x=d(g);

  __nv_bfloat16 *dA,*dB,*dGU,*dAct; float* dRoute;
  CHECK_CUDA(cudaMalloc(&dA,(size_t)M*KK*2));
  CHECK_CUDA(cudaMalloc(&dB,(size_t)NN*KK*2));
  CHECK_CUDA(cudaMalloc(&dGU,(size_t)M*NN*2));   // concat GU[M,2I]
  CHECK_CUDA(cudaMalloc(&dAct,(size_t)M*II*2));  // act[M,I]
  CHECK_CUDA(cudaMalloc(&dRoute,(size_t)M*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA,hA.data(),(size_t)M*KK*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,hB.data(),(size_t)NN*KK*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dRoute,hRoute.data(),(size_t)M*sizeof(float),cudaMemcpyHostToDevice));

  // A: box outer=BLOCK_M (A never multicast).
  auto a_in  = make_tma_2d(dA, KK, M, BLK_K, BLK_M, KK, 2, SWZ);
  // B multicast-on-N: 1-CTA loads BLOCK_N, 2-CTA loads BLOCK_N/2 per CTA.
  auto b_mc1 = make_tma_2d(dB, KK, NN, BLK_K, BLK_N,   KK, 2, SWZ);
  auto b_mc2 = make_tma_2d(dB, KK, NN, BLK_K, BLK_N/2, KK, 2, SWZ);
  // CD: concat store GU[M,2I]; interleaved store act[M,I]. Both STORE_BLOCK_N=64, box_m=128.
  auto cd_gu  = make_tma_2d(dGU,  NN, M, STORE_BLK_N, 128, NN, 2, SWZ);
  auto cd_act = make_tma_2d(dAct, II, M, STORE_BLK_N, 128, II, 2, SWZ);

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  double flop = 2.0*M*NN*KK;   // gate+up GEMM flops (N=2I)

  switch (nsms) {
    case 1:   run_configs<1>  (M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 2:   run_configs<2>  (M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 4:   run_configs<4>  (M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 8:   run_configs<8>  (M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 16:  run_configs<16> (M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 32:  run_configs<32> (M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 64:  run_configs<64> (M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 132: run_configs<132>(M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    case 148: run_configs<148>(M,mode,s,dA,dB,dGU,dAct,dRoute,a_in,b_mc1,b_mc2,cd_gu,cd_act,flop); break;
    default:
      fprintf(stderr,"unsupported SMS=%d (supported: 1,2,4,8,16,32,64,132,148)\n", nsms);
      return 1;
  }
  return 0;
}
