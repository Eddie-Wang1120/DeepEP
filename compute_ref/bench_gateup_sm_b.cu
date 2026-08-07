// bench_gateup_sm_b.cu — SM-scaling study for config B (gather4 + 1-CTA).
//
//   GEMM shape: M=256, K=2048, N=6144. Config B: hardware gather4 fused A load,
//   1-CTA MMA (kNumMulticast=1), clusterDim=1 launch. Sweep SMs = 8/16/32.
//   Reports median time, TFLOPS, correctness. Tensor-core活跃度 via ncu (arg = sm).
//
// Usage: ./bench_gateup_sm_b        -> all 8/16/32
//        ./bench_gateup_sm_b 8/16/32 -> single (few iters, for ncu)
//
// Build:
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 -lineinfo \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        bench_gateup_sm_b.cu -o bench_gateup_sm_b -lcuda

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
#include "sm100_bf16_gemm_dg_gather.cuh"

#define CHECK_CUDA(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" @ "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  std::cerr<<"CU error "<<s<<" @ "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(1);} } while(0)

static constexpr int MM = 256, KK = 2048, NN = 6144;
static constexpr uint32_t BLK_M = 128, BLK_N = 128, BLK_K = 64;
static constexpr uint32_t SWZ = 128, NSTAGES = 4;
static constexpr uint32_t NON_EPI = 128, EPI = 128, PHYS = 800, KALIGN = 128;

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

// 1-CTA gather4 GEMM kernel template, parametrized by SM count.
template <uint32_t NSMS>
auto get_kernel() {
  return &deep_gemm::sm100_bf16_gemm_gather_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K, 0u,0u,0u,
      BLK_M, BLK_N, BLK_K, 1u, SWZ, SWZ, SWZ, NSTAGES,
      NON_EPI, EPI, /*MCAST=*/1u, /*MCAST_ON_A=*/false, NSMS, KALIGN,
      false, false, deep_gemm::GemmType::Normal, false, cutlass::bfloat16_t,
      100ul, false, PHYS>;
}

template <class K_t>
void launch(K_t kernel, int nsms, const __nv_bfloat16* dAscat, const __nv_bfloat16* dB,
            __nv_bfloat16* dD, const int* dIdx, const CUtensorMap& a, const CUtensorMap& a_row,
            const CUtensorMap& b, const CUtensorMap& cd, cudaStream_t s) {
  int smem = 227*1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(nsms,1,1); cfg.blockDim=dim3(PHYS,1,1);
  cfg.dynamicSmemBytes=smem; cfg.stream=s;
  cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x=1; at[0].val.clusterDim.y=1; at[0].val.clusterDim.z=1;   // 1-CTA
  cfg.attrs=at; cfg.numAttrs=1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, (int*)nullptr,(uint32_t)MM,(uint32_t)NN,(uint32_t)KK,
      a,b,cd,(const cutlass::bfloat16_t*)nullptr,(const float*)nullptr,(uint32_t)NN, dIdx, a_row));
}

template<class F> float median_ms(F fn, int warm, int it, cudaStream_t s) {
  for(int i=0;i<warm;++i) fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  std::vector<float> ts(it); cudaEvent_t b,e; cudaEventCreate(&b); cudaEventCreate(&e);
  for(int i=0;i<it;++i){cudaEventRecord(b,s); fn(); cudaEventRecord(e,s);
    cudaEventSynchronize(e); cudaEventElapsedTime(&ts[i],b,e);}
  std::sort(ts.begin(),ts.end()); cudaEventDestroy(b); cudaEventDestroy(e); return ts[it/2];
}

static __nv_bfloat16 *dAscat,*dB,*dD; static int* dIdx;
static CUtensorMap a_in,a_row,b_mc1,cd_d;
static double g_flop; static std::vector<float> g_ref;

template <uint32_t NSMS>
void run_one(int nsms, char mode, cudaStream_t s) {
  auto k = get_kernel<NSMS>();
  auto fn = [&]{ launch(k, nsms, dAscat, dB, dD, dIdx, a_in, a_row, b_mc1, cd_d, s); };
  int warm=(mode=='*')?20:3, iter=(mode=='*')?100:5;
  CHECK_CUDA(cudaMemset(dD,0,(size_t)MM*NN*2));
  fn(); CHECK_CUDA(cudaStreamSynchronize(s));
  std::vector<__nv_bfloat16> hb(MM*NN);
  CHECK_CUDA(cudaMemcpy(hb.data(),dD,(size_t)MM*NN*2,cudaMemcpyDeviceToHost));
  double num=0,den=0;
  for(int i=0;i<MM*NN;++i){float v=__bfloat162float(hb[i]); double dd=v-g_ref[i]; num+=dd*dd; den+=(double)g_ref[i]*g_ref[i];}
  float rel=(float)(std::sqrt(num)/(std::sqrt(den)+1e-12));
  float t=median_ms(fn,warm,iter,s);
  fprintf(stderr,"[B SMS=%2d] %.4f ms  %.1f TFLOPS  rel_err=%.6g  %s\n",
          nsms, t, g_flop/(t*1e-3)/1e12, rel, rel<5e-2f?"PASS":"FAIL");
}

int main(int argc,char**argv){
  int only=(argc>1)?atoi(argv[1]):0; char mode=(only==0)?'*':'x';
  CHECK_CU(cuInit(0));
  fprintf(stderr,"gate/up SM-scaling (config B: gather4 + 1-CTA): M=%d K=%d N=%d\n",MM,KK,NN);
  std::mt19937 g(0); std::uniform_real_distribution<float> d(-1,1);
  std::vector<__nv_bfloat16> hA(MM*KK),hB(NN*KK); std::vector<float> Ab(MM*KK),Bb(NN*KK);
  auto bf=[](float v){return __bfloat162float(__float2bfloat16(v));};
  for(int i=0;i<MM*KK;++i){float v=d(g);hA[i]=__float2bfloat16(v);Ab[i]=bf(v);}
  for(int i=0;i<NN*KK;++i){float v=d(g);hB[i]=__float2bfloat16(v);Bb[i]=bf(v);}
  std::vector<int> perm(MM); std::iota(perm.begin(),perm.end(),0); std::shuffle(perm.begin(),perm.end(),g);
  std::vector<__nv_bfloat16> hAscat(MM*KK);
  for(int m=0;m<MM;++m)for(int k=0;k<KK;++k)hAscat[(size_t)perm[m]*KK+k]=hA[(size_t)m*KK+k];
  g_ref.assign((size_t)MM*NN,0.f);
  for(int m=0;m<MM;++m)for(int n=0;n<NN;++n){double sg=0;for(int k=0;k<KK;++k)sg+=(double)Ab[m*KK+k]*Bb[n*KK+k];g_ref[m*NN+n]=(float)sg;}
  g_flop=2.0*MM*NN*KK;

  CHECK_CUDA(cudaMalloc(&dAscat,(size_t)MM*KK*2)); CHECK_CUDA(cudaMalloc(&dB,(size_t)NN*KK*2));
  CHECK_CUDA(cudaMalloc(&dD,(size_t)MM*NN*2)); CHECK_CUDA(cudaMalloc(&dIdx,(size_t)MM*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dAscat,hAscat.data(),(size_t)MM*KK*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,hB.data(),(size_t)NN*KK*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dIdx,perm.data(),(size_t)MM*sizeof(int),cudaMemcpyHostToDevice));

  a_in  = make_tma_2d(dAscat, KK, MM, BLK_K, BLK_M, KK, 2, SWZ);
  a_row = make_tma_2d(dAscat, KK, MM, BLK_K, 1,     KK, 2, SWZ);
  b_mc1 = make_tma_2d(dB,     KK, NN, BLK_K, BLK_N,  KK, 2, SWZ);  // 1-CTA: LOAD_BLOCK_N=128
  cd_d  = make_tma_2d(dD,     NN, MM, SWZ/2, 128,   NN, 2, SWZ);

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  if(only==0||only==8)  run_one<8>(8,mode,s);
  if(only==0||only==16) run_one<16>(16,mode,s);
  if(only==0||only==32) run_one<32>(32,mode,s);
  return 0;
}
