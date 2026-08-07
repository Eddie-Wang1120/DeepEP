// umma_bwd_dswiglu.cu — Backward dSwiGLU microkernel: baseline vs optimized  
//  
// PATH A (baseline): current megakernel backward compute pattern with saved PreAct
//     input: saved forward gate/up preactivation in gu_buf [M,2I]
//     GEMM 2: dY @ W_down_T^T -> up_buf [M,I]     (UMMA via DeepGEMM)
//     KERNEL: scalar dSwiGLU reads up_buf and writes dGU/A'/dS side outputs
//     GEMM 3: grad_gu @ W_gateup_T^T -> dx [M,K]  (UMMA via DeepGEMM)
//
// PATH B (optimized): Sonic-MoE-aligned dH kernel decomposition
//     input: saved forward gate/up preactivation in gu_buf [M,2I]
//     GEMM 2 epilogue computes dSwiGLU in-register from accumulator + gu_buf
//     epilogue writes dGU, A'/y1s, and dS/route_grad partials
//     GEMM 3 consumes dGU directly; up_buf and duplicate dGU side copy are skipped.
//  
// Shape: M=256, hidden=4096, intermediate=4096 (DeepSeek-V3 MoE)  
// Config: 48/132 SMs, 800 threads/block, 1-CTA  
//  
// Build: same as umma_swiglu_ws_dg  
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \  
//        -I../DeepGEMM/deep_gemm/include -I../cutlass_ref/include \  
//        -I../cutlass_ref/tools/util/include \  
//        --expt-relaxed-constexpr -diag-suppress 20281 \  
//        umma_bwd_dswiglu.cu -o umma_bwd_dswiglu -lcuda  
// Run: ./umma_bwd_dswiglu --phase1-only --m 256 --k 4096 --n 4096 --sms 48 --iters 20  

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/simd_sm100.hpp>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include "sm100_bf16_gemm_dg_copy.cuh"

#define CHECK_CUDA(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":" \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
    const char* s=nullptr; cuGetErrorString(_e,&s); \
    std::cerr<<"CU error "<<(s?s:"<unknown>")<<" at "<<__FILE__<<":" \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)

// ======================= Config =======================
static constexpr uint32_t CFG_BLOCK_M = 128, CFG_BLOCK_N = 128, CFG_BF16_BLOCK_K = 64;
static constexpr uint32_t CFG_NUM_GROUPS = 1;
static constexpr uint32_t CFG_SWZ_A = 128, CFG_SWZ_B = 128, CFG_SWZ_CD = 128;
#ifndef CFG_NUM_STAGES
#define CFG_NUM_STAGES 4
#endif
static constexpr uint32_t CFG_NON_EPI_THREADS = 128, CFG_EPI_THREADS = 128;
static constexpr uint32_t CFG_PHYSICAL_THREADS = 800;
static constexpr uint32_t CFG_NUM_MULTICAST = 1;
static constexpr bool CFG_MCAST_ON_A = false;
static constexpr uint32_t CFG_K_ALIGNMENT = 128;
static constexpr uint64_t CFG_TC_UTIL = 100;

struct Options {
  int M = 256, K = 4096, N = 4096, sms = 48, warmup = 5, iters = 20;
  bool phase1_only = false;
  bool megakernel_like = false;
  std::string profile;
};

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char*)->const char*{
      if(i+1>=argc){std::cerr<<"Missing value for "<<argv[i]<<"\n";std::exit(1);}
      return argv[++i];
    };
    if(!std::strcmp(argv[i],"--m")) opt.M=std::atoi(need("--m"));
    else if(!std::strcmp(argv[i],"--k")) opt.K=std::atoi(need("--k"));
    else if(!std::strcmp(argv[i],"--n")) opt.N=std::atoi(need("--n"));
    else if(!std::strcmp(argv[i],"--sms")) opt.sms=std::atoi(need("--sms"));
    else if(!std::strcmp(argv[i],"--warmup")) opt.warmup=std::atoi(need("--warmup"));
    else if(!std::strcmp(argv[i],"--iters")) opt.iters=std::atoi(need("--iters"));
    else if(!std::strcmp(argv[i],"--phase1-only")) opt.phase1_only=true;
    else if(!std::strcmp(argv[i],"--megakernel-like")) opt.megakernel_like=true;
    else if(!std::strcmp(argv[i],"--profile")) opt.profile=need("--profile");
    else {std::cerr<<"Unknown: "<<argv[i]<<"\n";std::exit(1);}
  }
  return opt;
}

// ---- TMA descriptor builders (identical to umma_swiglu_ws_dg) ----
static CUtensorMap make_tma_2d(const void* ptr, CUtensorMapDataType dtype,
    int gmem_inner, int gmem_outer, int smem_inner, int smem_outer,
    int gmem_outer_stride_elems, int elem_size, int swizzle_bytes) {
  CUtensorMap tm;
  int si=swizzle_bytes?swizzle_bytes/elem_size:smem_inner;
  cuuint64_t gd[2]={(cuuint64_t)gmem_inner,(cuuint64_t)gmem_outer};
  cuuint64_t gs[1]={(cuuint64_t)gmem_outer_stride_elems*elem_size};
  cuuint32_t bd[2]={(cuuint32_t)si,(cuuint32_t)smem_outer}, es[2]={1,1};
  CUtensorMapSwizzle swz=CU_TENSOR_MAP_SWIZZLE_NONE;
  if(swizzle_bytes==128)swz=CU_TENSOR_MAP_SWIZZLE_128B;
  else if(swizzle_bytes==64)swz=CU_TENSOR_MAP_SWIZZLE_64B;
  else if(swizzle_bytes==32)swz=CU_TENSOR_MAP_SWIZZLE_32B;
  CHECK_CU(cuTensorMapEncodeTiled(&tm,dtype,2,(void*)ptr,gd,gs,bd,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,swz,CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}
static CUtensorMap make_bf16_a_desc(const __nv_bfloat16* a, int M, int K) {
  constexpr uint32_t lbm=CFG_BLOCK_M/(CFG_MCAST_ON_A?CFG_NUM_MULTICAST:1);
  return make_tma_2d(a,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,K,M,CFG_BF16_BLOCK_K,lbm,K,sizeof(__nv_bfloat16),CFG_SWZ_A);
}
static CUtensorMap make_bf16_b_desc(const __nv_bfloat16* b, int N, int K) {
  constexpr uint32_t lbn=CFG_BLOCK_N/(CFG_MCAST_ON_A?1:CFG_NUM_MULTICAST);
  return make_tma_2d(b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,K,N,CFG_BF16_BLOCK_K,lbn,K,sizeof(__nv_bfloat16),CFG_SWZ_B);
}
static CUtensorMap make_bf16_b_mn_desc(const __nv_bfloat16* b, int N, int K) {
  constexpr uint32_t lbn=CFG_BLOCK_N/(CFG_MCAST_ON_A?1:CFG_NUM_MULTICAST);
  return make_tma_2d(b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,N,K,lbn,CFG_BF16_BLOCK_K,N,sizeof(__nv_bfloat16),CFG_SWZ_B);
}
static CUtensorMap make_cd_desc(__nv_bfloat16* d, int M, int N) {
  return make_tma_2d(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,N,M,CFG_SWZ_CD/sizeof(__nv_bfloat16),CFG_BLOCK_M,N,sizeof(__nv_bfloat16),CFG_SWZ_CD);
}
static CUtensorMap make_cd_packed_desc(__nv_bfloat16* d, int M, int I) {
  return make_tma_2d(d,CU_TENSOR_MAP_DATA_TYPE_UINT32,I,M,CFG_SWZ_CD/sizeof(uint32_t),CFG_BLOCK_M,I,sizeof(uint32_t),CFG_SWZ_CD);
}

// ---- Launch GEMM via DeepGEMM sm100_bf16_gemm_impl (runtime sms dispatch) ----
template <int NumSms>
static void launch_bf16_gemm_inner(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    __nv_bfloat16* dD, int M, int N, int K, cudaStream_t stream) {
  auto dmap_a=make_bf16_a_desc(dA,M,K);
  auto dmap_b=make_bf16_b_desc(dB,N,K);
  auto dmap_cd=make_cd_desc(dD,M,N);
  auto kernel=&deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K,cute::UMMA::Major::K,
      0u,0u,0u, CFG_BLOCK_M,CFG_BLOCK_N,CFG_BF16_BLOCK_K, CFG_NUM_GROUPS,
      CFG_SWZ_A,CFG_SWZ_B,CFG_SWZ_CD, CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS,CFG_EPI_THREADS, CFG_NUM_MULTICAST,CFG_MCAST_ON_A,
      NumSms, CFG_K_ALIGNMENT, false,true,
      deep_gemm::GemmType::Normal,false,cutlass::bfloat16_t,
      CFG_TC_UTIL, false,false, CFG_PHYSICAL_THREADS>;
  size_t smem=227*1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(NumSms,1,1); cfg.blockDim=dim3(CFG_PHYSICAL_THREADS,1,1);
  cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x=CFG_NUM_MULTICAST; attrs[0].val.clusterDim.y=1; attrs[0].val.clusterDim.z=1;
  cfg.attrs=attrs; cfg.numAttrs=1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg,kernel,(int*)nullptr,(uint32_t)M,(uint32_t)N,(uint32_t)K,
      dmap_a,dmap_b,dmap_cd,
      static_cast<const cutlass::bfloat16_t*>(nullptr),
      static_cast<const float*>(nullptr),0u,
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<float*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<const int*>(nullptr),
      static_cast<const int*>(nullptr),
      0u, 0u));
  CHECK_CUDA(cudaGetLastError());
}
static void launch_bf16_gemm(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    __nv_bfloat16* dD, int M, int N, int K, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_inner<16>(dA,dB,dD,M,N,K,stream);break;
    case 32: launch_bf16_gemm_inner<32>(dA,dB,dD,M,N,K,stream);break;
    case 48: launch_bf16_gemm_inner<48>(dA,dB,dD,M,N,K,stream);break;
    case 64: launch_bf16_gemm_inner<64>(dA,dB,dD,M,N,K,stream);break;
    case 96: launch_bf16_gemm_inner<96>(dA,dB,dD,M,N,K,stream);break;
    case 120:launch_bf16_gemm_inner<120>(dA,dB,dD,M,N,K,stream);break;
    case 132:launch_bf16_gemm_inner<132>(dA,dB,dD,M,N,K,stream);break;
    case 148:launch_bf16_gemm_inner<148>(dA,dB,dD,M,N,K,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

template <int NumSms>
static void launch_bf16_gemm_mn_inner(const __nv_bfloat16* dA, const __nv_bfloat16* dBmn,
    __nv_bfloat16* dD, int M, int N, int K, cudaStream_t stream) {
  auto dmap_a=make_bf16_a_desc(dA,M,K);
  auto dmap_b=make_bf16_b_mn_desc(dBmn,N,K);
  auto dmap_cd=make_cd_desc(dD,M,N);
  auto kernel=&deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K,cute::UMMA::Major::MN,
      0u,0u,0u, CFG_BLOCK_M,CFG_BLOCK_N,CFG_BF16_BLOCK_K, CFG_NUM_GROUPS,
      CFG_SWZ_A,CFG_SWZ_B,CFG_SWZ_CD, CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS,CFG_EPI_THREADS, CFG_NUM_MULTICAST,CFG_MCAST_ON_A,
      NumSms, CFG_K_ALIGNMENT, false,true,
      deep_gemm::GemmType::Normal,false,cutlass::bfloat16_t,
      CFG_TC_UTIL, false,false, CFG_PHYSICAL_THREADS>;
  size_t smem=227*1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(NumSms,1,1); cfg.blockDim=dim3(CFG_PHYSICAL_THREADS,1,1);
  cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x=CFG_NUM_MULTICAST; attrs[0].val.clusterDim.y=1; attrs[0].val.clusterDim.z=1;
  cfg.attrs=attrs; cfg.numAttrs=1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg,kernel,(int*)nullptr,(uint32_t)M,(uint32_t)N,(uint32_t)K,
      dmap_a,dmap_b,dmap_cd,
      static_cast<const cutlass::bfloat16_t*>(nullptr),
      static_cast<const float*>(nullptr),0u,
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<float*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<const int*>(nullptr),
      static_cast<const int*>(nullptr),
      0u, 0u));
  CHECK_CUDA(cudaGetLastError());
}
static void launch_bf16_gemm_mn(const __nv_bfloat16* dA, const __nv_bfloat16* dBmn,
    __nv_bfloat16* dD, int M, int N, int K, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_mn_inner<16>(dA,dBmn,dD,M,N,K,stream);break;
    case 32: launch_bf16_gemm_mn_inner<32>(dA,dBmn,dD,M,N,K,stream);break;
    case 48: launch_bf16_gemm_mn_inner<48>(dA,dBmn,dD,M,N,K,stream);break;
    case 64: launch_bf16_gemm_mn_inner<64>(dA,dBmn,dD,M,N,K,stream);break;
    case 96: launch_bf16_gemm_mn_inner<96>(dA,dBmn,dD,M,N,K,stream);break;
    case 120:launch_bf16_gemm_mn_inner<120>(dA,dBmn,dD,M,N,K,stream);break;
    case 132:launch_bf16_gemm_mn_inner<132>(dA,dBmn,dD,M,N,K,stream);break;
    case 148:launch_bf16_gemm_mn_inner<148>(dA,dBmn,dD,M,N,K,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

template <int NumSms>
static void launch_bf16_gemm_dswiglu_inner(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, __nv_bfloat16* dGuOut, int M, int I, int K,
    int twoI, cudaStream_t stream) {
  auto dmap_a=make_bf16_a_desc(dA,M,K);
  auto dmap_b=make_bf16_b_desc(dB,I,K);
  auto dmap_cd=make_cd_desc(dGuOut,M,twoI);
  auto kernel=&deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K,cute::UMMA::Major::K,
      0u,0u,0u, CFG_BLOCK_M,CFG_BLOCK_N,CFG_BF16_BLOCK_K, CFG_NUM_GROUPS,
      CFG_SWZ_A,CFG_SWZ_B,CFG_SWZ_CD, CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS,CFG_EPI_THREADS, CFG_NUM_MULTICAST,CFG_MCAST_ON_A,
      NumSms, CFG_K_ALIGNMENT, false,true,
      deep_gemm::GemmType::Normal,false,cutlass::bfloat16_t,
      CFG_TC_UTIL, false,false, CFG_PHYSICAL_THREADS, true>;
  size_t smem=227*1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(NumSms,1,1); cfg.blockDim=dim3(CFG_PHYSICAL_THREADS,1,1);
  cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x=CFG_NUM_MULTICAST; attrs[0].val.clusterDim.y=1; attrs[0].val.clusterDim.z=1;
  cfg.attrs=attrs; cfg.numAttrs=1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg,kernel,(int*)nullptr,(uint32_t)M,(uint32_t)I,(uint32_t)K,
      dmap_a,dmap_b,dmap_cd,
      reinterpret_cast<const cutlass::bfloat16_t*>(dGuIn),
      static_cast<const float*>(nullptr),(uint32_t)twoI,
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<float*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<const int*>(nullptr),
      static_cast<const int*>(nullptr),
      0u, 0u));
  CHECK_CUDA(cudaGetLastError());
}
static void launch_bf16_gemm_dswiglu(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, __nv_bfloat16* dGuOut, int M, int I, int K,
    int twoI, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_dswiglu_inner<16>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    case 32: launch_bf16_gemm_dswiglu_inner<32>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    case 48: launch_bf16_gemm_dswiglu_inner<48>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    case 64: launch_bf16_gemm_dswiglu_inner<64>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    case 96: launch_bf16_gemm_dswiglu_inner<96>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    case 120:launch_bf16_gemm_dswiglu_inner<120>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    case 132:launch_bf16_gemm_dswiglu_inner<132>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    case 148:launch_bf16_gemm_dswiglu_inner<148>(dA,dB,dGuIn,dGuOut,M,I,K,twoI,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

template <int NumSms, bool EmitSideOutputs = false, bool RouteGradOnly = false, bool RouteGradPartial = false>
static void launch_bf16_gemm_dswiglu_packed_inner(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, const float* dRoute, __nv_bfloat16* dGuOut,
    __nv_bfloat16* dWgradAct, __nv_bfloat16* dWgradDgu, float* dRouteGrad,
    int M, int I, int K, int twoI, cudaStream_t stream) {
  auto dmap_a=make_bf16_a_desc(dA,M,K);
  auto dmap_b=make_bf16_b_desc(dB,I,K);
  auto dmap_cd=make_cd_packed_desc(dGuOut,M,I);
  auto kernel=&deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K,cute::UMMA::Major::K,
      0u,0u,0u, CFG_BLOCK_M,CFG_BLOCK_N,CFG_BF16_BLOCK_K, CFG_NUM_GROUPS,
      CFG_SWZ_A,CFG_SWZ_B,CFG_SWZ_CD, CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS,CFG_EPI_THREADS, CFG_NUM_MULTICAST,CFG_MCAST_ON_A,
      NumSms, CFG_K_ALIGNMENT, false,true,
      deep_gemm::GemmType::Normal,false,float,
      CFG_TC_UTIL, false,false, CFG_PHYSICAL_THREADS, false, true, EmitSideOutputs, RouteGradOnly, RouteGradPartial>;
  size_t smem=227*1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(NumSms,1,1); cfg.blockDim=dim3(CFG_PHYSICAL_THREADS,1,1);
  cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x=CFG_NUM_MULTICAST; attrs[0].val.clusterDim.y=1; attrs[0].val.clusterDim.z=1;
  cfg.attrs=attrs; cfg.numAttrs=1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg,kernel,(int*)nullptr,(uint32_t)M,(uint32_t)I,(uint32_t)K,
      dmap_a,dmap_b,dmap_cd,
      reinterpret_cast<const cutlass::bfloat16_t*>(dGuIn),
      dRoute,(uint32_t)twoI,
      reinterpret_cast<cutlass::bfloat16_t*>(dWgradAct),
      reinterpret_cast<cutlass::bfloat16_t*>(dWgradDgu),
      dRouteGrad,
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<const int*>(nullptr),
      static_cast<const int*>(nullptr),
      0u, 0u));
  CHECK_CUDA(cudaGetLastError());
}
static void launch_bf16_gemm_dswiglu_packed(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, const float* dRoute, __nv_bfloat16* dGuOut, int M, int I, int K,
    int twoI, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_dswiglu_packed_inner<16,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    case 32: launch_bf16_gemm_dswiglu_packed_inner<32,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    case 48: launch_bf16_gemm_dswiglu_packed_inner<48,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    case 64: launch_bf16_gemm_dswiglu_packed_inner<64,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    case 96: launch_bf16_gemm_dswiglu_packed_inner<96,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    case 120:launch_bf16_gemm_dswiglu_packed_inner<120,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    case 132:launch_bf16_gemm_dswiglu_packed_inner<132,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    case 148:launch_bf16_gemm_dswiglu_packed_inner<148,false>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,nullptr,M,I,K,twoI,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

static void launch_bf16_gemm_dswiglu_packed_side_outputs(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, const float* dRoute, __nv_bfloat16* dGuOut,
    __nv_bfloat16* dWgradAct, __nv_bfloat16* dWgradDgu, float* dRouteGrad,
    int M, int I, int K, int twoI, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_dswiglu_packed_inner<16,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    case 32: launch_bf16_gemm_dswiglu_packed_inner<32,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    case 48: launch_bf16_gemm_dswiglu_packed_inner<48,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    case 64: launch_bf16_gemm_dswiglu_packed_inner<64,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    case 96: launch_bf16_gemm_dswiglu_packed_inner<96,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    case 120:launch_bf16_gemm_dswiglu_packed_inner<120,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    case 132:launch_bf16_gemm_dswiglu_packed_inner<132,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    case 148:launch_bf16_gemm_dswiglu_packed_inner<148,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

static void launch_bf16_gemm_dswiglu_packed_side_outputs_partial(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, const float* dRoute, __nv_bfloat16* dGuOut,
    __nv_bfloat16* dWgradAct, __nv_bfloat16* dWgradDgu, float* dRouteGradPartial,
    int M, int I, int K, int twoI, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_dswiglu_packed_inner<16,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 32: launch_bf16_gemm_dswiglu_packed_inner<32,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 48: launch_bf16_gemm_dswiglu_packed_inner<48,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 64: launch_bf16_gemm_dswiglu_packed_inner<64,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 96: launch_bf16_gemm_dswiglu_packed_inner<96,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 120:launch_bf16_gemm_dswiglu_packed_inner<120,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 132:launch_bf16_gemm_dswiglu_packed_inner<132,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 148:launch_bf16_gemm_dswiglu_packed_inner<148,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

static void launch_bf16_gemm_dswiglu_sonic_outputs_partial(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, const float* dRoute, __nv_bfloat16* dGuOut,
    __nv_bfloat16* dY1s, float* dRouteGradPartial,
    int M, int I, int K, int twoI, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_dswiglu_packed_inner<16,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 32: launch_bf16_gemm_dswiglu_packed_inner<32,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 48: launch_bf16_gemm_dswiglu_packed_inner<48,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 64: launch_bf16_gemm_dswiglu_packed_inner<64,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 96: launch_bf16_gemm_dswiglu_packed_inner<96,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 120:launch_bf16_gemm_dswiglu_packed_inner<120,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 132:launch_bf16_gemm_dswiglu_packed_inner<132,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 148:launch_bf16_gemm_dswiglu_packed_inner<148,true,false,true>(dA,dB,dGuIn,dRoute,dGuOut,dY1s,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

static void launch_bf16_gemm_dswiglu_packed_route_grad(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, const float* dRoute, __nv_bfloat16* dGuOut, float* dRouteGrad,
    int M, int I, int K, int twoI, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_dswiglu_packed_inner<16,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    case 32: launch_bf16_gemm_dswiglu_packed_inner<32,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    case 48: launch_bf16_gemm_dswiglu_packed_inner<48,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    case 64: launch_bf16_gemm_dswiglu_packed_inner<64,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    case 96: launch_bf16_gemm_dswiglu_packed_inner<96,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    case 120:launch_bf16_gemm_dswiglu_packed_inner<120,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    case 132:launch_bf16_gemm_dswiglu_packed_inner<132,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    case 148:launch_bf16_gemm_dswiglu_packed_inner<148,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGrad,M,I,K,twoI,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

static void launch_bf16_gemm_dswiglu_packed_route_grad_partial(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
    const __nv_bfloat16* dGuIn, const float* dRoute, __nv_bfloat16* dGuOut, float* dRouteGradPartial,
    int M, int I, int K, int twoI, int sms, cudaStream_t stream) {
  switch(sms) {
    case 16: launch_bf16_gemm_dswiglu_packed_inner<16,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 32: launch_bf16_gemm_dswiglu_packed_inner<32,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 48: launch_bf16_gemm_dswiglu_packed_inner<48,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 64: launch_bf16_gemm_dswiglu_packed_inner<64,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 96: launch_bf16_gemm_dswiglu_packed_inner<96,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 120:launch_bf16_gemm_dswiglu_packed_inner<120,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 132:launch_bf16_gemm_dswiglu_packed_inner<132,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    case 148:launch_bf16_gemm_dswiglu_packed_inner<148,true,true,true>(dA,dB,dGuIn,dRoute,dGuOut,nullptr,nullptr,dRouteGradPartial,M,I,K,twoI,stream);break;
    default: std::cerr<<"Unsupported --sms "<<sms<<"\n";std::exit(1);
  }
}

__device__ __forceinline__ uint32_t pack_bf16_pair(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  union { __nv_bfloat16 h[2]; uint32_t u; } cvt;
  cvt.h[0]=lo; cvt.h[1]=hi;
  return cvt.u;
}

// ===== PATH A: Baseline element-wise dSwiGLU =====
__global__ void baseline_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    __nv_bfloat16* __restrict__ gu_buf_out,
    int M, int twoI, int I) {
  int m=blockIdx.x; if(m>=M)return;
  int tid=threadIdx.x, stride=blockDim.x;
  for(int i=tid;i<I;i+=stride){
    float g=__bfloat162float(gu_buf[m*twoI+2*i]);
    float u=__bfloat162float(gu_buf[m*twoI+2*i+1]);
    float ga=__bfloat162float(up_buf[m*I+i]);
    float sig=1.0f/(1.0f+__expf(-g));
    float silu=g*sig;
    float silu_ga=silu*ga;
    float gg=((sig+(-silu)*sig)*ga+silu_ga)*u;
    float gu=__bfloat162float(gu_buf[m*twoI+2*i+1]);
    gu_buf_out[m*twoI+2*i]=__float2bfloat16(gg);
    gu_buf_out[m*twoI+2*i+1]=__float2bfloat16(silu_ga);
  }
}

// ===== PATH B: Fused vectorized dSwiGLU (SIMT int4 I/O) =====
__global__ void megakernel_like_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    const float* __restrict__ route,
    __nv_bfloat16* __restrict__ gu_buf_out,
    __nv_bfloat16* __restrict__ wgrad_act,
    __nv_bfloat16* __restrict__ wgrad_dgu,
    float* __restrict__ route_grad,
    int M, int twoI, int I) {
  int m=blockIdx.x; if(m>=M)return;
  int tid=threadIdx.x, stride=blockDim.x;
  float rg=0.0f;
  float r=route[m];
  for(int i=tid;i<I;i+=stride){
    float g=__bfloat162float(gu_buf[(size_t)m*twoI+2*i]);
    float u=__bfloat162float(gu_buf[(size_t)m*twoI+2*i+1]);
    float ga=__bfloat162float(up_buf[(size_t)m*I+i]);
    float sig=1.0f/(1.0f+__expf(-g));
    float silu=g*sig;
    float activation=silu*u;
    float g_pre=ga*r;
    float silu_ga=silu*g_pre;
    float dgate=((sig+(-silu)*sig)*g_pre+silu_ga)*u;
    uint32_t out=pack_bf16_pair(__float2bfloat16(dgate),__float2bfloat16(silu_ga));
    reinterpret_cast<uint32_t*>(gu_buf_out)[(size_t)m*I+i]=out;
    reinterpret_cast<uint32_t*>(wgrad_dgu)[(size_t)m*I+i]=out;
    wgrad_act[(size_t)m*I+i]=__float2bfloat16(r*activation);
    rg += ga*activation;
  }
  for(int offset=16;offset>0;offset>>=1) rg += __shfl_down_sync(0xffffffff,rg,offset);
  if((threadIdx.x&31)==0) atomicAdd(&route_grad[m],rg);
}

__global__ void megakernel_like_side_outputs_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    const float* __restrict__ route,
    const __nv_bfloat16* __restrict__ gu_buf_out,
    __nv_bfloat16* __restrict__ wgrad_act,
    __nv_bfloat16* __restrict__ wgrad_dgu,
    float* __restrict__ route_grad,
    int M, int twoI, int I) {
  int m=blockIdx.x; if(m>=M)return;
  int tid=threadIdx.x, stride=blockDim.x;
  float rg=0.0f;
  float r=route[m];
  for(int i=tid;i<I;i+=stride){
    float g=__bfloat162float(gu_buf[(size_t)m*twoI+2*i]);
    float u=__bfloat162float(gu_buf[(size_t)m*twoI+2*i+1]);
    float ga=__bfloat162float(up_buf[(size_t)m*I+i]);
    float sig=1.0f/(1.0f+__expf(-g));
    float silu=g*sig;
    float activation=silu*u;
    reinterpret_cast<uint32_t*>(wgrad_dgu)[(size_t)m*I+i]=reinterpret_cast<const uint32_t*>(gu_buf_out)[(size_t)m*I+i];
    wgrad_act[(size_t)m*I+i]=__float2bfloat16(r*activation);
    rg += ga*activation;
  }
  for(int offset=16;offset>0;offset>>=1) rg += __shfl_down_sync(0xffffffff,rg,offset);
  if((threadIdx.x&31)==0) atomicAdd(&route_grad[m],rg);
}

__global__ void megakernel_like_side_outputs_from_dgu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ gu_buf_out,
    const float* __restrict__ route,
    __nv_bfloat16* __restrict__ wgrad_act,
    __nv_bfloat16* __restrict__ wgrad_dgu,
    float* __restrict__ route_grad,
    int M, int twoI, int I) {
  int m=blockIdx.x; if(m>=M)return;
  int tid=threadIdx.x, stride=blockDim.x;
  float rg=0.0f;
  float r=route[m];
  const uint32_t* packed_dgu = reinterpret_cast<const uint32_t*>(gu_buf_out);
  uint32_t* packed_wgrad_dgu = reinterpret_cast<uint32_t*>(wgrad_dgu);
  auto bf=[](uint32_t v,int lane)->float{return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v)[lane]);};
  for(int i=tid;i<I;i+=stride){
    float g=__bfloat162float(gu_buf[(size_t)m*twoI+2*i]);
    float u=__bfloat162float(gu_buf[(size_t)m*twoI+2*i+1]);
    float sig=deep_gemm::dswiglu_fast_sigmoid(g);
    float silu=g*sig;
    float activation=silu*u;
    uint32_t dgu=packed_dgu[(size_t)m*I+i];
    packed_wgrad_dgu[(size_t)m*I+i]=dgu;
    wgrad_act[(size_t)m*I+i]=__float2bfloat16(r*activation);
    if(route_grad != nullptr){
      float g_up=bf(dgu,1);
      rg += fabsf(r)>1e-20f ? (g_up/r)*u : 0.0f;
    }
  }
  if(route_grad != nullptr){
    for(int offset=16;offset>0;offset>>=1) rg += __shfl_down_sync(0xffffffff,rg,offset);
    if((threadIdx.x&31)==0) atomicAdd(&route_grad[m],rg);
  }
}

__global__ void megakernel_like_side_outputs_act_only_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const float* __restrict__ route,
    __nv_bfloat16* __restrict__ wgrad_act,
    int M, int twoI, int I) {
  int m=blockIdx.x; if(m>=M)return;
  int tid=threadIdx.x, stride=blockDim.x;
  float r=route[m];
  for(int i=tid;i<I;i+=stride){
    float g=__bfloat162float(gu_buf[(size_t)m*twoI+2*i]);
    float u=__bfloat162float(gu_buf[(size_t)m*twoI+2*i+1]);
    float sig=deep_gemm::dswiglu_fast_sigmoid(g);
    float activation=g*sig*u;
    wgrad_act[(size_t)m*I+i]=__float2bfloat16(r*activation);
  }
}

__global__ void reduce_route_grad_partials_kernel(
    const float* __restrict__ route_grad_partial,
    float* __restrict__ route_grad,
    int M, int num_n_tiles) {
  int m = blockIdx.x;
  if (m >= M) return;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  float sum = 0.0f;
  for (int i = tid; i < num_n_tiles; i += blockDim.x)
    sum += route_grad_partial[(size_t)m * num_n_tiles + i];
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ float warp_sums[32];
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  int num_warps = (blockDim.x + 31) >> 5;
  sum = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) route_grad[m] = sum;
  }
}

__global__ void fused_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    __nv_bfloat16* __restrict__ gu_buf_out,
    int M, int twoI, int I) {
  int m=blockIdx.x; if(m>=M)return;
  int tid=threadIdx.x, stride=blockDim.x;
  int I4=I/4;
  size_t gu_row_int4 = (size_t)twoI / 8;
  size_t up_row_int2 = (size_t)I / 4;
  for(int iv=tid;iv<I4;iv+=stride){
    int4 gu_vec=reinterpret_cast<const int4*>(gu_buf)[(size_t)m*gu_row_int4+iv];
    int2 up_vec=reinterpret_cast<const int2*>(up_buf)[(size_t)m*up_row_int2+iv];
    auto bf=[](uint32_t v,int i)->float{return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v)[i]);};
    auto pack=[](__nv_bfloat16 lo,__nv_bfloat16 hi)->uint32_t{
      union { __nv_bfloat16 h[2]; uint32_t u; } cvt;
      cvt.h[0]=lo; cvt.h[1]=hi; return cvt.u;
    };
    float g0=bf((uint32_t)gu_vec.x,0),u0=bf((uint32_t)gu_vec.x,1);
    float g1=bf((uint32_t)gu_vec.y,0),u1=bf((uint32_t)gu_vec.y,1);
    float g2=bf((uint32_t)gu_vec.z,0),u2=bf((uint32_t)gu_vec.z,1);
    float g3=bf((uint32_t)gu_vec.w,0),u3=bf((uint32_t)gu_vec.w,1);
    float a0=bf((uint32_t)up_vec.x,0),a1=bf((uint32_t)up_vec.x,1);
    float a2=bf((uint32_t)up_vec.y,0),a3=bf((uint32_t)up_vec.y,1);
    auto ds=[&](float g,float u,float ga)->__nv_bfloat162{
      float sig=1.0f/(1.0f+__expf(-g));
      float silu=g*sig, silu_ga=silu*ga;
      return {__float2bfloat16(((sig+(-silu)*sig)*ga+silu_ga)*u), __float2bfloat16(silu_ga)};
    };
    auto r0=ds(g0,u0,a0),r1=ds(g1,u1,a1),r2=ds(g2,u2,a2),r3=ds(g3,u3,a3);
    int4 out;
    out.x=(int)pack(r0.x,r0.y); out.y=(int)pack(r1.x,r1.y);
    out.z=(int)pack(r2.x,r2.y); out.w=(int)pack(r3.x,r3.y);
    reinterpret_cast<int4*>(gu_buf_out)[(size_t)m*gu_row_int4+iv]=out;
  }
  int tail_start=I4*4;
  for(int i=tail_start+tid;i<I;i+=stride){
    float g=__bfloat162float(gu_buf[m*twoI+2*i]);
    float u=__bfloat162float(gu_buf[m*twoI+2*i+1]);
    float ga=__bfloat162float(up_buf[m*I+i]);
    float sig=1.0f/(1.0f+__expf(-g)), silu=g*sig, silu_ga=silu*ga;
    gu_buf_out[m*twoI+2*i]=__float2bfloat16(((sig+(-silu)*sig)*ga+silu_ga)*u);
    gu_buf_out[m*twoI+2*i+1]=__float2bfloat16(silu_ga);
  }
}

__device__ __forceinline__ float bf16_from_u32(uint32_t v, int lane) {
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v)[lane]);
}

__device__ __forceinline__ uint32_t dswiglu_pair(float gate, float up, float grad_act) {
  float sig=1.0f/(1.0f+__expf(-gate));
  float silu=gate*sig;
  float silu_ga=silu*grad_act;
  float dgate=((sig+(-silu)*sig)*grad_act+silu_ga)*up;
  return pack_bf16_pair(__float2bfloat16(dgate), __float2bfloat16(silu_ga));
}

__device__ __forceinline__ uint32_t cvt_f32x2_bf16x2(float lo, float hi) {
  uint32_t out;
  asm volatile("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;\n"
               : "=r"(out) : "f"(hi), "f"(lo));
  return out;
}

__device__ __forceinline__ void dswiglu_pair2_f32x2(
    uint32_t gu0, uint32_t gu1, uint32_t grad01,
    uint32_t& out0, uint32_t& out1) {
  float2 gate = {bf16_from_u32(gu0, 0), bf16_from_u32(gu1, 0)};
  float2 up = {bf16_from_u32(gu0, 1), bf16_from_u32(gu1, 1)};
  float2 grad = {bf16_from_u32(grad01, 0), bf16_from_u32(grad01, 1)};
  float2 sig = {
      1.0f / (1.0f + __expf(-gate.x)),
      1.0f / (1.0f + __expf(-gate.y))};
  float2 silu, silu_grad, sig_minus_silu_sig, d_silu_grad, dgate;
  cute::mul(silu, gate, sig);
  cute::mul(silu_grad, silu, grad);
  cute::fma(sig_minus_silu_sig, silu, {-sig.x, -sig.y}, sig);
  cute::fma(d_silu_grad, sig_minus_silu_sig, grad, silu_grad);
  cute::mul(dgate, d_silu_grad, up);
  out0 = cvt_f32x2_bf16x2(dgate.x, silu_grad.x);
  out1 = cvt_f32x2_bf16x2(dgate.y, silu_grad.y);
}

__device__ __forceinline__ void dswiglu_pair2_side_f32x2(
    uint32_t gu0, uint32_t gu1, uint32_t grad01, float route,
    uint32_t& out0, uint32_t& out1, uint32_t& act01, float& route_grad) {
  float2 gate = {bf16_from_u32(gu0, 0), bf16_from_u32(gu1, 0)};
  float2 up = {bf16_from_u32(gu0, 1), bf16_from_u32(gu1, 1)};
  float2 grad_raw = {bf16_from_u32(grad01, 0), bf16_from_u32(grad01, 1)};
  float2 grad = {grad_raw.x * route, grad_raw.y * route};
  float2 sig = {
      1.0f / (1.0f + __expf(-gate.x)),
      1.0f / (1.0f + __expf(-gate.y))};
  float2 silu, activation, silu_grad, sig_minus_silu_sig, d_silu_grad, dgate;
  cute::mul(silu, gate, sig);
  cute::mul(activation, silu, up);
  cute::mul(silu_grad, silu, grad);
  cute::fma(sig_minus_silu_sig, silu, {-sig.x, -sig.y}, sig);
  cute::fma(d_silu_grad, sig_minus_silu_sig, grad, silu_grad);
  cute::mul(dgate, d_silu_grad, up);
  out0 = cvt_f32x2_bf16x2(dgate.x, silu_grad.x);
  out1 = cvt_f32x2_bf16x2(dgate.y, silu_grad.y);
  act01 = cvt_f32x2_bf16x2(route * activation.x, route * activation.y);
  route_grad += grad_raw.x * activation.x + grad_raw.y * activation.y;
}

// Row-owned packed dSwiGLU keeps the SIMT work off the GEMM epilogue. dGuOut is
// also the weight-gradient dGU input, so callers can alias those two consumers.
template <bool WriteDuplicateDgu>
__global__ void packed4_f32x2_side_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ grad_act,
    const float* __restrict__ route,
    __nv_bfloat16* __restrict__ dgu_out,
    __nv_bfloat16* __restrict__ wgrad_act,
    __nv_bfloat16* __restrict__ wgrad_dgu,
    float* __restrict__ route_grad,
    int M, int I) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int I4 = I / 4;
  const float r = route[m];
  const int4* gu4 = reinterpret_cast<const int4*>(gu_buf) + (size_t)m * I4;
  const int2* ga4 = reinterpret_cast<const int2*>(grad_act) + (size_t)m * I4;
  int4* out4 = reinterpret_cast<int4*>(dgu_out) + (size_t)m * I4;
  int4* wgrad4 = WriteDuplicateDgu
      ? reinterpret_cast<int4*>(wgrad_dgu) + (size_t)m * I4 : nullptr;
  int2* act4 = reinterpret_cast<int2*>(wgrad_act) + (size_t)m * I4;
  float rg = 0.0f;
  for (int q = tid; q < I4; q += blockDim.x) {
    const int4 gu = gu4[q];
    const int2 ga = ga4[q];
    uint32_t o0, o1, o2, o3, a01, a23;
    dswiglu_pair2_side_f32x2((uint32_t)gu.x, (uint32_t)gu.y, (uint32_t)ga.x,
                             r, o0, o1, a01, rg);
    dswiglu_pair2_side_f32x2((uint32_t)gu.z, (uint32_t)gu.w, (uint32_t)ga.y,
                             r, o2, o3, a23, rg);
    const int4 out = {static_cast<int>(o0), static_cast<int>(o1),
                      static_cast<int>(o2), static_cast<int>(o3)};
    out4[q] = out;
    if constexpr (WriteDuplicateDgu) wgrad4[q] = out;
    act4[q] = {static_cast<int>(a01), static_cast<int>(a23)};
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    rg += __shfl_down_sync(0xffffffff, rg, offset);
  __shared__ float warp_sums[32];
  if (lane == 0) warp_sums[warp] = rg;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) / 32;
  rg = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1)
      rg += __shfl_down_sync(0xffffffff, rg, offset);
    if (lane == 0) route_grad[m] = rg;
  }
}

// Megakernel-shaped variant: one persistent CTA per compute SM, 800 physical
// threads, with each CTA owning complete rows so route_grad stays block-local.
template <bool WriteDuplicateDgu>
__global__ void packed4_f32x2_side_persistent_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ grad_act,
    const float* __restrict__ route,
    __nv_bfloat16* __restrict__ dgu_out,
    __nv_bfloat16* __restrict__ wgrad_act,
    __nv_bfloat16* __restrict__ wgrad_dgu,
    float* __restrict__ route_grad,
    int M, int I) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (blockDim.x + 31) / 32;
  const int I4 = I / 4;
  __shared__ float warp_sums[32];
  for (int m = blockIdx.x; m < M; m += gridDim.x) {
    const float r = route[m];
    const int4* gu4 = reinterpret_cast<const int4*>(gu_buf) + (size_t)m * I4;
    const int2* ga4 = reinterpret_cast<const int2*>(grad_act) + (size_t)m * I4;
    int4* out4 = reinterpret_cast<int4*>(dgu_out) + (size_t)m * I4;
    int4* wgrad4 = WriteDuplicateDgu
        ? reinterpret_cast<int4*>(wgrad_dgu) + (size_t)m * I4 : nullptr;
    int2* act4 = reinterpret_cast<int2*>(wgrad_act) + (size_t)m * I4;
    float rg = 0.0f;
    for (int q = tid; q < I4; q += blockDim.x) {
      const int4 gu = gu4[q];
      const int2 ga = ga4[q];
      uint32_t o0, o1, o2, o3, a01, a23;
      dswiglu_pair2_side_f32x2((uint32_t)gu.x, (uint32_t)gu.y, (uint32_t)ga.x,
                               r, o0, o1, a01, rg);
      dswiglu_pair2_side_f32x2((uint32_t)gu.z, (uint32_t)gu.w, (uint32_t)ga.y,
                               r, o2, o3, a23, rg);
      const int4 out = {static_cast<int>(o0), static_cast<int>(o1),
                        static_cast<int>(o2), static_cast<int>(o3)};
      out4[q] = out;
      if constexpr (WriteDuplicateDgu) wgrad4[q] = out;
      act4[q] = {static_cast<int>(a01), static_cast<int>(a23)};
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      rg += __shfl_down_sync(0xffffffff, rg, offset);
    if (lane == 0) warp_sums[warp] = rg;
    __syncthreads();
    rg = tid < num_warps ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
      for (int offset = 16; offset > 0; offset >>= 1)
        rg += __shfl_down_sync(0xffffffff, rg, offset);
      if (lane == 0) route_grad[m] = rg;
    }
    __syncthreads();
  }
}

// 48x800 megakernel-shaped scheduler with one complete row per warp. This keeps
// route_grad warp-local while exposing 48 * 25 independent rows concurrently.
__global__ void packed4_f32x2_side_warp_rows_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ grad_act,
    const float* __restrict__ route,
    __nv_bfloat16* __restrict__ dgu_out,
    __nv_bfloat16* __restrict__ wgrad_act,
    float* __restrict__ route_grad,
    int M, int I) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int global_warp = blockIdx.x * warps_per_block + warp;
  const int total_warps = gridDim.x * warps_per_block;
  const int I4 = I / 4;
  for (int m = global_warp; m < M; m += total_warps) {
    const float r = route[m];
    const int4* gu4 = reinterpret_cast<const int4*>(gu_buf) + (size_t)m * I4;
    const int2* ga4 = reinterpret_cast<const int2*>(grad_act) + (size_t)m * I4;
    int4* out4 = reinterpret_cast<int4*>(dgu_out) + (size_t)m * I4;
    int2* act4 = reinterpret_cast<int2*>(wgrad_act) + (size_t)m * I4;
    float rg = 0.0f;
    for (int q = lane; q < I4; q += 32) {
      const int4 gu = gu4[q];
      const int2 ga = ga4[q];
      uint32_t o0, o1, o2, o3, a01, a23;
      dswiglu_pair2_side_f32x2((uint32_t)gu.x, (uint32_t)gu.y, (uint32_t)ga.x,
                               r, o0, o1, a01, rg);
      dswiglu_pair2_side_f32x2((uint32_t)gu.z, (uint32_t)gu.w, (uint32_t)ga.y,
                               r, o2, o3, a23, rg);
      out4[q] = {static_cast<int>(o0), static_cast<int>(o1),
                 static_cast<int>(o2), static_cast<int>(o3)};
      act4[q] = {static_cast<int>(a01), static_cast<int>(a23)};
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      rg += __shfl_down_sync(0xffffffff, rg, offset);
    if (lane == 0) route_grad[m] = rg;
  }
}

// Phase 1: same scalar math as baseline, but scheduled as SM-count persistent CTAs.
__global__ void phase1_scalar_persistent_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    __nv_bfloat16* __restrict__ gu_buf_out,
    int M, int twoI, int I) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int total = M * I;
  for (int idx = tid; idx < total; idx += stride) {
    int m = idx / I, i = idx - m * I;
    float g=__bfloat162float(gu_buf[(size_t)m*twoI+2*i]);
    float u=__bfloat162float(gu_buf[(size_t)m*twoI+2*i+1]);
    float ga=__bfloat162float(up_buf[(size_t)m*I+i]);
    uint32_t out=dswiglu_pair(g,u,ga);
    reinterpret_cast<uint32_t*>(gu_buf_out)[(size_t)m*I+i]=out;
  }
}

// Phase 1: Sonic-MoE-style packed PreAct/Out view.  GU and dGU are physical
// bf16[M,2I], but this kernel treats them as uint32[M,I] packed (gate,up) and
// (dgate,dup), avoiding I->2I indexing in the hot loop.
__global__ void phase1_packed_persistent_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    __nv_bfloat16* __restrict__ gu_buf_out,
    int M, int I) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  const uint32_t* __restrict__ gu_packed = reinterpret_cast<const uint32_t*>(gu_buf);
  uint32_t* __restrict__ out_packed = reinterpret_cast<uint32_t*>(gu_buf_out);
  const uint32_t* __restrict__ up_packed = reinterpret_cast<const uint32_t*>(up_buf);
  int I2 = I / 2;
  int total2 = M * I2;
  for (int idx2 = tid; idx2 < total2; idx2 += stride) {
    int m = idx2 / I2;
    int i0 = (idx2 - m * I2) * 2;
    uint32_t gu0 = gu_packed[(size_t)m*I+i0];
    uint32_t gu1 = gu_packed[(size_t)m*I+i0+1];
    uint32_t ga01 = up_packed[(size_t)m*I2+(i0/2)];
    out_packed[(size_t)m*I+i0] = dswiglu_pair(bf16_from_u32(gu0,0), bf16_from_u32(gu0,1), bf16_from_u32(ga01,0));
    out_packed[(size_t)m*I+i0+1] = dswiglu_pair(bf16_from_u32(gu1,0), bf16_from_u32(gu1,1), bf16_from_u32(ga01,1));
  }
}

// Same packed view, but each thread processes a 16B GU vector: four packed
// (gate,up) slots plus one 8B grad_act vector. This better matches the memory
// granularity we want in the future GEMM epilogue.
__global__ void phase1_packed4_persistent_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    __nv_bfloat16* __restrict__ gu_buf_out,
    int M, int I) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int I4 = I / 4;
  int total4 = M * I4;
  const int4* __restrict__ gu_vec = reinterpret_cast<const int4*>(gu_buf);
  const int2* __restrict__ up_vec = reinterpret_cast<const int2*>(up_buf);
  int4* __restrict__ out_vec = reinterpret_cast<int4*>(gu_buf_out);
  for (int idx4 = tid; idx4 < total4; idx4 += stride) {
    int m = idx4 / I4;
    int q = idx4 - m * I4;
    int4 gu = gu_vec[(size_t)m*I4+q];
    int2 ga = up_vec[(size_t)m*I4+q];
    int4 out;
    out.x = static_cast<int>(dswiglu_pair(bf16_from_u32((uint32_t)gu.x,0), bf16_from_u32((uint32_t)gu.x,1), bf16_from_u32((uint32_t)ga.x,0)));
    out.y = static_cast<int>(dswiglu_pair(bf16_from_u32((uint32_t)gu.y,0), bf16_from_u32((uint32_t)gu.y,1), bf16_from_u32((uint32_t)ga.x,1)));
    out.z = static_cast<int>(dswiglu_pair(bf16_from_u32((uint32_t)gu.z,0), bf16_from_u32((uint32_t)gu.z,1), bf16_from_u32((uint32_t)ga.y,0)));
    out.w = static_cast<int>(dswiglu_pair(bf16_from_u32((uint32_t)gu.w,0), bf16_from_u32((uint32_t)gu.w,1), bf16_from_u32((uint32_t)ga.y,1)));
    out_vec[(size_t)m*I4+q] = out;
  }
}

__global__ void phase1_packed4_f32x2_persistent_dswiglu_kernel(
    const __nv_bfloat16* __restrict__ gu_buf,
    const __nv_bfloat16* __restrict__ up_buf,
    __nv_bfloat16* __restrict__ gu_buf_out,
    int M, int I) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int I4 = I / 4;
  int total4 = M * I4;
  const int4* __restrict__ gu_vec = reinterpret_cast<const int4*>(gu_buf);
  const int2* __restrict__ up_vec = reinterpret_cast<const int2*>(up_buf);
  int4* __restrict__ out_vec = reinterpret_cast<int4*>(gu_buf_out);
  for (int idx4 = tid; idx4 < total4; idx4 += stride) {
    int m = idx4 / I4;
    int q = idx4 - m * I4;
    int4 gu = gu_vec[(size_t)m*I4+q];
    int2 ga = up_vec[(size_t)m*I4+q];
    uint32_t ox, oy, oz, ow;
    dswiglu_pair2_f32x2((uint32_t)gu.x, (uint32_t)gu.y, (uint32_t)ga.x, ox, oy);
    dswiglu_pair2_f32x2((uint32_t)gu.z, (uint32_t)gu.w, (uint32_t)ga.y, oz, ow);
    out_vec[(size_t)m*I4+q] = {static_cast<int>(ox), static_cast<int>(oy), static_cast<int>(oz), static_cast<int>(ow)};
  }
}

// ---- CPU references ----
static void cpu_gemm_bf16(const std::vector<__nv_bfloat16>& A, const std::vector<__nv_bfloat16>& B,
    std::vector<__nv_bfloat16>& C, int M, int K, int N) {
  for(int m=0;m<M;++m)for(int n=0;n<N;++n){
    float s=0; for(int k=0;k<K;++k)s+=__bfloat162float(A[m*K+k])*__bfloat162float(B[n*K+k]);
    C[m*N+n]=__float2bfloat16(s);
  }
}
static void cpu_dswiglu(const std::vector<__nv_bfloat16>& gu, const std::vector<__nv_bfloat16>& up,
    std::vector<__nv_bfloat16>& out, int M, int twoI, int I) {
  for(int m=0;m<M;++m)for(int i=0;i<I;++i){
    float g=__bfloat162float(gu[m*twoI+2*i]),u=__bfloat162float(gu[m*twoI+2*i+1]);
    float ga=__bfloat162float(up[m*I+i]);
    float sig=1.0f/(1.0f+std::exp(-g)),silu=g*sig,silu_ga=silu*ga;
    out[m*twoI+2*i]=__float2bfloat16(((sig+(-silu)*sig)*ga+silu_ga)*u);
    out[m*twoI+2*i+1]=__float2bfloat16(silu_ga);
  }
}
static void cpu_megakernel_like_dswiglu(const std::vector<__nv_bfloat16>& gu,
    const std::vector<__nv_bfloat16>& up, const std::vector<float>& route,
    std::vector<__nv_bfloat16>& out, std::vector<__nv_bfloat16>& wgrad_act,
    std::vector<__nv_bfloat16>& wgrad_dgu, std::vector<float>& route_grad,
    int M, int twoI, int I) {
  std::fill(route_grad.begin(), route_grad.end(), 0.0f);
  for(int m=0;m<M;++m)for(int i=0;i<I;++i){
    float g=__bfloat162float(gu[m*twoI+2*i]),u=__bfloat162float(gu[m*twoI+2*i+1]);
    float ga=__bfloat162float(up[m*I+i]);
    float sig=1.0f/(1.0f+std::exp(-g));
    float silu=g*sig;
    float activation=silu*u;
    float g_pre=ga*route[m];
    float silu_ga=silu*g_pre;
    float dgate=((sig+(-silu)*sig)*g_pre+silu_ga)*u;
    out[m*twoI+2*i]=__float2bfloat16(dgate);
    out[m*twoI+2*i+1]=__float2bfloat16(silu_ga);
    wgrad_dgu[m*twoI+2*i]=__float2bfloat16(dgate);
    wgrad_dgu[m*twoI+2*i+1]=__float2bfloat16(silu_ga);
    wgrad_act[m*I+i]=__float2bfloat16(route[m]*activation);
    route_grad[m]+=ga*activation;
  }
}

static void print_error(const char* label, const std::vector<__nv_bfloat16>& ref,
    const std::vector<__nv_bfloat16>& test, int count) {
  float mr=0; double ss=0,rs=0; int nz=0;
  for(int i=0;i<count;++i){
    float r=__bfloat162float(ref[i]), t=__bfloat162float(test[i]);
    float ar=fabsf(r), d=fabsf(r-t);
    if(ar>1e-6f){mr=std::max(mr,d/ar);++nz;}
    ss+=d*d; rs+=ar*ar;
  }
  printf("%s max_rel=%.6g rmse=%.6g nrmse=%.6g nz=%d/%d\n",label,mr,sqrt(ss/count),
         rs>0?sqrt(ss/rs):sqrt(ss/count),nz,count);
}
static void print_error_f32(const char* label, const std::vector<float>& ref,
    const std::vector<float>& test, int count) {
  float mr=0; double ss=0,rs=0; int nz=0;
  for(int i=0;i<count;++i){
    float r=ref[i], t=test[i];
    float ar=fabsf(r), d=fabsf(r-t);
    if(ar>1e-6f){mr=std::max(mr,d/ar);++nz;}
    ss+=d*d; rs+=ar*ar;
  }
  printf("%s max_rel=%.6g rmse=%.6g nrmse=%.6g nz=%d/%d\n",label,mr,sqrt(ss/count),
         rs>0?sqrt(ss/rs):sqrt(ss/count),nz,count);
}

static void run_phase1_only(const Options& opt) {
  int M=opt.M, I=opt.N, twoI=2*I;
  if (opt.sms != 48) {
    std::cerr << "Phase1 is fixed to --sms 48 for the megakernel target\n";
    std::exit(1);
  }
  if ((I & 3) != 0) {
    std::cerr << "Phase1 packed4 path requires --n/intermediate size divisible by 4\n";
    std::exit(1);
  }
  printf("=== Phase1 dSwiGLU-only M=%d I=%d sms=%d threads/CTA=%u ===\n\n",
         M,I,opt.sms,CFG_PHYSICAL_THREADS);

  int gu_elems = M * twoI;
  size_t gub=(size_t)gu_elems*sizeof(__nv_bfloat16), upb=(size_t)M*I*sizeof(__nv_bfloat16);
  __nv_bfloat16 *dGuBuf,*dUpBuf,*dGuOut;
  CHECK_CUDA(cudaMalloc(&dGuBuf,gub));
  CHECK_CUDA(cudaMalloc(&dUpBuf,upb));
  CHECK_CUDA(cudaMalloc(&dGuOut,gub));

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f,1.f);
  auto rb=[&](){return __float2bfloat16(dist(rng));};
  std::vector<__nv_bfloat16> hGu(M*twoI), hGa(M*I), ref(M*twoI), out(M*twoI);
  for (int i=0;i<M*twoI;++i) hGu[i]=rb();
  for (int i=0;i<M*I;++i) hGa[i]=rb();
  cpu_dswiglu(hGu,hGa,ref,M,twoI,I);
  CHECK_CUDA(cudaMemcpy(dGuBuf,hGu.data(),gub,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dUpBuf,hGa.data(),upb,cudaMemcpyHostToDevice));

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  cudaEvent_t ev0,ev1; CHECK_CUDA(cudaEventCreate(&ev0)); CHECK_CUDA(cudaEventCreate(&ev1));
  auto dt=[&](auto&& fn){
    CHECK_CUDA(cudaEventRecord(ev0,s)); fn();
    CHECK_CUDA(cudaEventRecord(ev1,s)); CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,ev0,ev1)); return ms;
  };

  dim3 sm48(48), t800(CFG_PHYSICAL_THREADS), row_grid(M), row_block(256);
  for (int w=0; w<opt.warmup; ++w) {
    baseline_dswiglu_kernel<<<row_grid,row_block,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
    phase1_scalar_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
    phase1_packed_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,I);
    phase1_packed4_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,I);
    phase1_packed4_f32x2_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,I);
  }
  CHECK_CUDA(cudaStreamSynchronize(s));

  auto bench=[&](const char* name, auto&& fn){
    double total=0;
    for(int it=0; it<opt.iters; ++it) total += dt(fn);
    total /= opt.iters;
    CHECK_CUDA(cudaMemcpy(out.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    printf("%s: %.3f us\n", name, total*1000.0);
    print_error(name, ref, out, M*twoI);
    return total;
  };

  double t_row = bench("row256_scalar", [&](){
    baseline_dswiglu_kernel<<<row_grid,row_block,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
  });
  double t_scalar48 = bench("sm48x800_scalar", [&](){
    phase1_scalar_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
  });
  double t_packed48 = bench("sm48x800_packed2", [&](){
    phase1_packed_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,I);
  });
  double t_packed4_48 = bench("sm48x800_packed4", [&](){
    phase1_packed4_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,I);
  });
  double t_packed4_f32x2_48 = bench("sm48x800_packed4_f32x2", [&](){
    phase1_packed4_f32x2_persistent_dswiglu_kernel<<<sm48,t800,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,I);
  });
  printf("\nSpeedup vs row256 scalar: persistent %.2fx, packed2 %.2fx, packed4 %.2fx, packed4_f32x2 %.2fx\n",
         t_row/t_scalar48, t_row/t_packed48, t_row/t_packed4_48, t_row/t_packed4_f32x2_48);
  printf("Speedup vs sm48 scalar: packed2 %.2fx, packed4 %.2fx, packed4_f32x2 %.2fx\n",
         t_scalar48/t_packed48, t_scalar48/t_packed4_48, t_scalar48/t_packed4_f32x2_48);

  CHECK_CUDA(cudaFree(dGuBuf)); CHECK_CUDA(cudaFree(dUpBuf)); CHECK_CUDA(cudaFree(dGuOut));
  CHECK_CUDA(cudaEventDestroy(ev0)); CHECK_CUDA(cudaEventDestroy(ev1));
  CHECK_CUDA(cudaStreamDestroy(s));
}

// Standalone int4 PreAct gather (mirrors megakernel backward PreAct load into
// gu_buf). Bounds the backward PreAct-load cost: one CTA per row, 16B loads.
__global__ void preact_gather_int4_kernel(const __nv_bfloat16* __restrict__ src,
                                          __nv_bfloat16* __restrict__ dst,
                                          int M, int twoI) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const int i4 = twoI * (int)sizeof(__nv_bfloat16) / (int)sizeof(int4);
  const int4* s = reinterpret_cast<const int4*>(src) + (size_t)row * i4;
  int4* d = reinterpret_cast<int4*>(dst) + (size_t)row * i4;
  for (int q = threadIdx.x; q < i4; q += blockDim.x) d[q] = s[q];
}

static void run_profile_only(const Options& opt) {
  int M=opt.M, K=opt.K, I=opt.N, twoI=2*I;
  int num_n_tiles=(I + (int)CFG_BLOCK_N - 1) / (int)CFG_BLOCK_N;
  size_t xb=(size_t)M*K*sizeof(__nv_bfloat16), wgub=(size_t)twoI*K*sizeof(__nv_bfloat16);
  size_t wguTb=(size_t)K*twoI*sizeof(__nv_bfloat16), wdTb=(size_t)I*K*sizeof(__nv_bfloat16);
  size_t gub=(size_t)M*twoI*sizeof(__nv_bfloat16), upb=(size_t)M*I*sizeof(__nv_bfloat16);
  size_t routeb=(size_t)M*sizeof(float), route_partial_b=(size_t)M*num_n_tiles*sizeof(float);
  __nv_bfloat16 *dX,*dWgu,*dWguT,*dY,*dWdT,*dWdMN,*dGuBuf,*dUpBuf,*dGuOut,*dDxPerm,*dWgradAct,*dWgradDgu;
  float *dRoute,*dRouteGrad,*dRouteGradPartial;
  CHECK_CUDA(cudaMalloc(&dX,xb)); CHECK_CUDA(cudaMalloc(&dWgu,wgub));
  CHECK_CUDA(cudaMalloc(&dWguT,wguTb)); CHECK_CUDA(cudaMalloc(&dY,xb));
  CHECK_CUDA(cudaMalloc(&dWdT,wdTb)); CHECK_CUDA(cudaMalloc(&dWdMN,wdTb)); CHECK_CUDA(cudaMalloc(&dGuBuf,gub));
  CHECK_CUDA(cudaMalloc(&dUpBuf,upb)); CHECK_CUDA(cudaMalloc(&dGuOut,gub));
  CHECK_CUDA(cudaMalloc(&dDxPerm,xb)); CHECK_CUDA(cudaMalloc(&dWgradAct,upb));
  CHECK_CUDA(cudaMalloc(&dWgradDgu,gub)); CHECK_CUDA(cudaMalloc(&dRoute,routeb));
  CHECK_CUDA(cudaMalloc(&dRouteGrad,routeb)); CHECK_CUDA(cudaMalloc(&dRouteGradPartial,route_partial_b));
  CHECK_CUDA(cudaMemset(dX,1,xb)); CHECK_CUDA(cudaMemset(dWgu,2,wgub));
  CHECK_CUDA(cudaMemset(dWguT,3,wguTb)); CHECK_CUDA(cudaMemset(dY,4,xb));
  CHECK_CUDA(cudaMemset(dWdT,5,wdTb)); CHECK_CUDA(cudaMemset(dWdMN,6,wdTb));
  CHECK_CUDA(cudaMemset(dGuOut,0,gub)); CHECK_CUDA(cudaMemset(dRoute,0x3f,routeb));
  CHECK_CUDA(cudaMemset(dRouteGrad,0,routeb)); CHECK_CUDA(cudaMemset(dRouteGradPartial,0,route_partial_b));
  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  cudaEvent_t ev0,ev1; CHECK_CUDA(cudaEventCreate(&ev0)); CHECK_CUDA(cudaEventCreate(&ev1));
  auto dt=[&](auto&& fn){
    CHECK_CUDA(cudaEventRecord(ev0,s)); fn();
    CHECK_CUDA(cudaEventRecord(ev1,s)); CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,ev0,ev1)); return ms;
  };
  launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
  launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
  CHECK_CUDA(cudaStreamSynchronize(s));
  auto run_one=[&]() {
    if (opt.profile == "gemm2") {
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
    } else if (opt.profile == "gemm2-mn") {
      launch_bf16_gemm_mn(dY,dWdMN,dUpBuf,M,I,K,opt.sms,s);
    } else if (opt.profile == "gemm3") {
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "preact-gather") {
      preact_gather_int4_kernel<<<M,256,0,s>>>(dGuOut,dGuBuf,M,twoI);
    } else if (opt.profile == "gemm3-mn") {
      launch_bf16_gemm_mn(dGuOut,dWgu,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "packed") {
      launch_bf16_gemm_dswiglu_packed(dY,dWdT,dGuBuf,dRoute,dGuOut,M,I,K,twoI,opt.sms,s);
    } else if (opt.profile == "packed-route") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
    } else if (opt.profile == "packed-partial") {
      launch_bf16_gemm_dswiglu_packed_route_grad_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
    } else if (opt.profile == "side-act") {
      megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
    } else if (opt.profile == "side-from-dgu") {
      megakernel_like_side_outputs_from_dgu_kernel<<<M,256,0,s>>>(dGuBuf,dGuOut,dRoute,dWgradAct,dWgradDgu,nullptr,M,twoI,I);
    } else if (opt.profile == "scalar-side") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      megakernel_like_dswiglu_kernel<<<M,256,0,s>>>(
          dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,twoI,I);
    } else if (opt.profile == "packed4-side" || opt.profile == "packed4-side-128" ||
               opt.profile == "packed4-side-512" || opt.profile == "packed4-side-copy") {
      const int threads = opt.profile == "packed4-side-128" ? 128 :
                          opt.profile == "packed4-side-512" ? 512 : 256;
      if (opt.profile == "packed4-side-copy")
        packed4_f32x2_side_dswiglu_kernel<true><<<M,threads,0,s>>>(
            dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I);
      else
        packed4_f32x2_side_dswiglu_kernel<false><<<M,threads,0,s>>>(
            dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,nullptr,dRouteGrad,M,I);
    } else if (opt.profile == "packed4-full" || opt.profile == "packed4-full-128" ||
               opt.profile == "packed4-full-512") {
      const int threads = opt.profile == "packed4-full-128" ? 128 :
                          opt.profile == "packed4-full-512" ? 512 : 256;
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
      packed4_f32x2_side_dswiglu_kernel<false><<<M,threads,0,s>>>(
          dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,nullptr,dRouteGrad,M,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sm48x800-side") {
      packed4_f32x2_side_persistent_dswiglu_kernel<false><<<48,800,0,s>>>(
          dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,nullptr,dRouteGrad,M,I);
    } else if (opt.profile == "sm48x800-full") {
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
      packed4_f32x2_side_persistent_dswiglu_kernel<false><<<48,800,0,s>>>(
          dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,nullptr,dRouteGrad,M,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sm48x800-warp-side") {
      packed4_f32x2_side_warp_rows_dswiglu_kernel<<<48,800,0,s>>>(
          dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dRouteGrad,M,I);
    } else if (opt.profile == "sm48x800-warp-full") {
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
      packed4_f32x2_side_warp_rows_dswiglu_kernel<<<48,800,0,s>>>(
          dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dRouteGrad,M,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "legacy-recompute-full") {
      launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      megakernel_like_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "baseline-full") {
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      megakernel_like_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sonic-full") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_side_outputs(dY,dWdT,dGuBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,opt.sms,s);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sonic-partial-full") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_side_outputs_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
      reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sonic-original-full") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_sonic_outputs_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dWgradAct,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
      reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sonic-alias-full") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sonic-dgu-act-full") {
      launch_bf16_gemm_dswiglu_packed(dY,dWdT,dGuBuf,dRoute,dGuOut,M,I,K,twoI,opt.sms,s);
      megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "sonic-split-full") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      megakernel_like_side_outputs_from_dgu_kernel<<<M,256,0,s>>>(dGuBuf,dGuOut,dRoute,dWgradAct,dWgradDgu,nullptr,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "packed-full") {
      launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      megakernel_like_side_outputs_from_dgu_kernel<<<M,256,0,s>>>(dGuBuf,dGuOut,dRoute,dWgradAct,dWgradDgu,nullptr,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else if (opt.profile == "saved-preact") {
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    } else {
      std::cerr << "Unknown --profile " << opt.profile << "\n";
      std::exit(1);
    }
  };
  for(int w=0; w<opt.warmup; ++w) run_one();
  CHECK_CUDA(cudaStreamSynchronize(s));
  double total=0.0;
  for(int it=0; it<opt.iters; ++it) total += dt(run_one);
  printf("PROFILE %s avg=%.4f ms M=%d K=%d I=%d sms=%d iters=%d\n",
         opt.profile.c_str(), total/opt.iters, M, K, I, opt.sms, opt.iters);
  CHECK_CUDA(cudaFree(dX)); CHECK_CUDA(cudaFree(dWgu)); CHECK_CUDA(cudaFree(dWguT));
  CHECK_CUDA(cudaFree(dY)); CHECK_CUDA(cudaFree(dWdT)); CHECK_CUDA(cudaFree(dWdMN)); CHECK_CUDA(cudaFree(dGuBuf));
  CHECK_CUDA(cudaFree(dUpBuf)); CHECK_CUDA(cudaFree(dGuOut)); CHECK_CUDA(cudaFree(dDxPerm));
  CHECK_CUDA(cudaFree(dWgradAct)); CHECK_CUDA(cudaFree(dWgradDgu)); CHECK_CUDA(cudaFree(dRoute));
  CHECK_CUDA(cudaFree(dRouteGrad)); CHECK_CUDA(cudaFree(dRouteGradPartial));
  CHECK_CUDA(cudaEventDestroy(ev0)); CHECK_CUDA(cudaEventDestroy(ev1)); CHECK_CUDA(cudaStreamDestroy(s));
}

int main(int argc, char** argv) {
  Options opt=parse_args(argc,argv);
  if (!opt.profile.empty()) {
    run_profile_only(opt);
    return 0;
  }
  if (opt.phase1_only) {
    run_phase1_only(opt);
    return 0;
  }
  int M=opt.M, K=opt.K, I=opt.N, twoI=2*I;
  printf("=== umma_bwd_dswiglu M=%d K=%d I=%d sms=%d ===\n\n",M,K,I,opt.sms);

  // Alloc
  size_t xb=(size_t)M*K*2, wgub=(size_t)twoI*K*2, wguTb=(size_t)K*twoI*2;
  size_t wdTb=(size_t)I*K*2, gub=(size_t)M*twoI*2, upb=(size_t)M*I*2;
  int num_n_tiles=(I + (int)CFG_BLOCK_N - 1) / (int)CFG_BLOCK_N;
  size_t routeb=(size_t)M*sizeof(float), route_partial_b=(size_t)M*num_n_tiles*sizeof(float);
  __nv_bfloat16 *dX,*dWgu,*dWguT,*dY,*dWdT,*dWdMN,*dGuBuf,*dUpBuf,*dGuOut,*dDxPerm;
  __nv_bfloat16 *dWgradAct=nullptr,*dWgradDgu=nullptr;
  float *dRoute=nullptr,*dRouteGrad=nullptr,*dRouteGradPartial=nullptr;
  CHECK_CUDA(cudaMalloc(&dX,xb)); CHECK_CUDA(cudaMalloc(&dWgu,wgub));
  CHECK_CUDA(cudaMalloc(&dWguT,wguTb)); CHECK_CUDA(cudaMalloc(&dY,xb));
  CHECK_CUDA(cudaMalloc(&dWdT,wdTb)); CHECK_CUDA(cudaMalloc(&dWdMN,wdTb)); CHECK_CUDA(cudaMalloc(&dGuBuf,gub));
  CHECK_CUDA(cudaMalloc(&dUpBuf,upb)); CHECK_CUDA(cudaMalloc(&dGuOut,gub));
  CHECK_CUDA(cudaMalloc(&dDxPerm,xb));
  if(opt.megakernel_like){
    CHECK_CUDA(cudaMalloc(&dRoute,routeb));
    CHECK_CUDA(cudaMalloc(&dRouteGrad,routeb));
    CHECK_CUDA(cudaMalloc(&dRouteGradPartial,route_partial_b));
    CHECK_CUDA(cudaMalloc(&dWgradAct,upb));
    CHECK_CUDA(cudaMalloc(&dWgradDgu,gub));
  }

  // Random data
  std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.f,1.f);
  std::uniform_real_distribution<float> route_dist(0.1f,1.0f);
  auto rb=[&](){return __float2bfloat16(dist(rng));};
  std::vector<__nv_bfloat16> hX(M*K),hY(M*K),hWgu(twoI*K),hWguT(K*twoI),hWdT(I*K),hWdMN(K*I);
  std::vector<float> hRoute(M);
  for(int i=0;i<M*K;++i){hX[i]=rb();hY[i]=rb();}
  for(int i=0;i<twoI*K;++i)hWgu[i]=rb();
  for(int i=0;i<I*K;++i)hWdT[i]=rb();
  for(int i=0;i<M;++i)hRoute[i]=route_dist(rng);
  for(int k=0;k<K;++k)for(int j=0;j<twoI;++j)hWguT[k*twoI+j]=hWgu[j*K+k];
  for(int k=0;k<K;++k)for(int i=0;i<I;++i)hWdMN[k*I+i]=hWdT[i*K+k];
  CHECK_CUDA(cudaMemcpy(dX,hX.data(),xb,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWgu,hWgu.data(),wgub,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWguT,hWguT.data(),wguTb,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY,hY.data(),xb,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWdT,hWdT.data(),wdTb,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWdMN,hWdMN.data(),wdTb,cudaMemcpyHostToDevice));
  if(opt.megakernel_like) CHECK_CUDA(cudaMemcpy(dRoute,hRoute.data(),routeb,cudaMemcpyHostToDevice));

  // CPU ref
  std::vector<__nv_bfloat16> ref_gu(M*twoI),ref_ga(M*I),ref_dswiglu(M*twoI),ref_dx(M*K);
  std::vector<__nv_bfloat16> ref_wgrad_act(M*I),ref_wgrad_dgu(M*twoI);
  std::vector<float> ref_route_grad(M),hRouteGrad(M);
  cpu_gemm_bf16(hX,hWgu,ref_gu,M,K,twoI);
  cpu_gemm_bf16(hY,hWdT,ref_ga,M,K,I);
  if(opt.megakernel_like)
    cpu_megakernel_like_dswiglu(ref_gu,ref_ga,hRoute,ref_dswiglu,ref_wgrad_act,ref_wgrad_dgu,ref_route_grad,M,twoI,I);
  else
    cpu_dswiglu(ref_gu,ref_ga,ref_dswiglu,M,twoI,I);
  cpu_gemm_bf16(ref_dswiglu,hWguT,ref_dx,M,twoI,K);
  printf("[cpu] reference computed\n");
  if(opt.megakernel_like)
    CHECK_CUDA(cudaMemcpy(dGuBuf,ref_gu.data(),gub,cudaMemcpyHostToDevice));

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  cudaEvent_t ev0,ev1; CHECK_CUDA(cudaEventCreate(&ev0)); CHECK_CUDA(cudaEventCreate(&ev1));
  auto dt=[&](auto&& fn){
    CHECK_CUDA(cudaEventRecord(ev0,s)); fn();
    CHECK_CUDA(cudaEventRecord(ev1,s)); CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,ev0,ev1)); return ms;
  };

  // ===== PATH A: Baseline =====
  printf("--- PATH A: Baseline ---\n");
  for(int w=0;w<opt.warmup;++w){
    if(!opt.megakernel_like)
      launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
    launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
    if(opt.megakernel_like){
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      megakernel_like_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,twoI,I);
    } else {
      baseline_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
    }
    launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
  }
  CHECK_CUDA(cudaStreamSynchronize(s));
  double totA=0;
  for(int it=0;it<opt.iters;++it)
    totA+=dt([&](){
      if(!opt.megakernel_like)
        launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
      if(opt.megakernel_like){
        CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
        megakernel_like_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,twoI,I);
      } else {
        baseline_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
      }
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    });
  double msA=totA/opt.iters;
  std::vector<__nv_bfloat16> hDx(M*K), hGuCheck(M*twoI);
  CHECK_CUDA(cudaMemcpy(hGuCheck.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
  print_error("Baseline_dGU",ref_dswiglu,hGuCheck,M*twoI);
  print_error("Baseline_dx",ref_dx,hDx,M*K);
  if(opt.megakernel_like){
    std::vector<__nv_bfloat16> hWgradAct(M*I), hWgradDgu(M*twoI);
    CHECK_CUDA(cudaMemcpy(hWgradAct.data(),dWgradAct,upb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hWgradDgu.data(),dWgradDgu,gub,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRouteGrad.data(),dRouteGrad,routeb,cudaMemcpyDeviceToHost));
    print_error("Baseline_wgrad_act",ref_wgrad_act,hWgradAct,M*I);
    print_error("Baseline_wgrad_dgu",ref_wgrad_dgu,hWgradDgu,M*twoI);
    print_error_f32("Baseline_route_grad",ref_route_grad,hRouteGrad,M);
  }
  const double fl_recompute = ((double)M*K*twoI*2+(double)M*K*I*2+(double)M*twoI*K*2);
  const double fl_saved = ((double)M*K*I*2+(double)M*twoI*K*2);
  const double flA = opt.megakernel_like ? fl_saved : fl_recompute;
  printf("Baseline  avg=%.4f ms  GEMM-TFLOPS=%.3f\n\n",msA,flA/(msA*1e-3)/1e12);

  if(opt.megakernel_like){
    printf("--- PATH A2: packed4 f32x2 standalone dSwiGLU ---\n");
    for(int w=0;w<opt.warmup;++w){
      launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
      packed4_f32x2_side_dswiglu_kernel<false><<<M,256,0,s>>>(
          dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,nullptr,dRouteGrad,M,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    double total=0.0;
    for(int it=0;it<opt.iters;++it)
      total+=dt([&](){
        launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);
        packed4_f32x2_side_dswiglu_kernel<false><<<M,256,0,s>>>(
            dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,nullptr,dRouteGrad,M,I);
        launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
      });
    const double optimized_ms=total/opt.iters;
    std::vector<__nv_bfloat16> hWgradAct(M*I);
    CHECK_CUDA(cudaMemcpy(hGuCheck.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hWgradAct.data(),dWgradAct,upb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRouteGrad.data(),dRouteGrad,routeb,cudaMemcpyDeviceToHost));
    print_error("Packed4Side_dGU_and_wgrad_dgu",ref_wgrad_dgu,hGuCheck,M*twoI);
    print_error("Packed4Side_dx",ref_dx,hDx,M*K);
    print_error("Packed4Side_wgrad_act",ref_wgrad_act,hWgradAct,M*I);
    print_error_f32("Packed4Side_route_grad",ref_route_grad,hRouteGrad,M);
    printf("Packed4Side avg=%.4f ms  speedup=%.2fx\n\n",optimized_ms,msA/optimized_ms);

    packed4_f32x2_side_persistent_dswiglu_kernel<false><<<48,800,0,s>>>(
        dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,nullptr,dRouteGrad,M,I);
    launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    CHECK_CUDA(cudaStreamSynchronize(s));
    CHECK_CUDA(cudaMemcpy(hGuCheck.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hWgradAct.data(),dWgradAct,upb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRouteGrad.data(),dRouteGrad,routeb,cudaMemcpyDeviceToHost));
    print_error("SM48x800_dGU",ref_wgrad_dgu,hGuCheck,M*twoI);
    print_error("SM48x800_dx",ref_dx,hDx,M*K);
    print_error("SM48x800_wgrad_act",ref_wgrad_act,hWgradAct,M*I);
    print_error_f32("SM48x800_route_grad",ref_route_grad,hRouteGrad,M);

    packed4_f32x2_side_warp_rows_dswiglu_kernel<<<48,800,0,s>>>(
        dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dRouteGrad,M,I);
    launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    CHECK_CUDA(cudaStreamSynchronize(s));
    CHECK_CUDA(cudaMemcpy(hGuCheck.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hWgradAct.data(),dWgradAct,upb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRouteGrad.data(),dRouteGrad,routeb,cudaMemcpyDeviceToHost));
    print_error("SM48x800Warp_dGU",ref_wgrad_dgu,hGuCheck,M*twoI);
    print_error("SM48x800Warp_dx",ref_dx,hDx,M*K);
    print_error("SM48x800Warp_wgrad_act",ref_wgrad_act,hWgradAct,M*I);
    print_error_f32("SM48x800Warp_route_grad",ref_route_grad,hRouteGrad,M);
  }

  // ===== PATH B: GEMM2 epilogue-fused dSwiGLU =====
  double msB=0.0;
  if(!opt.megakernel_like){
    printf("--- PATH B: GEMM2 epilogue dSwiGLU ---\n");
    for(int w=0;w<opt.warmup;++w){
      launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
      launch_bf16_gemm_dswiglu(dY,dWdT,dGuBuf,dGuOut,M,I,K,twoI,opt.sms,s);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    double totB=0;
    for(int it=0;it<opt.iters;++it)
      totB+=dt([&](){
        launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
        launch_bf16_gemm_dswiglu(dY,dWdT,dGuBuf,dGuOut,M,I,K,twoI,opt.sms,s);
        launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
      });
    msB=totB/opt.iters;
    CHECK_CUDA(cudaMemcpy(hGuCheck.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
    print_error("EpilogueFused_dGU",ref_dswiglu,hGuCheck,M*twoI);
    print_error("EpilogueFused_dx",ref_dx,hDx,M*K);
    printf("EpiFused avg=%.4f ms  GEMM-TFLOPS=%.3f  speedup=%.2fx\n\n",msB,flA/(msB*1e-3)/1e12,msA/msB);
  } else {
    printf("--- PATH B: GEMM2 epilogue dSwiGLU ---\n");
    printf("Skipped in --megakernel-like mode: this older expanded path is route-less.\n\n");
  }

  // ===== PATH C: Sonic-MoE-aligned GEMM2 packed epilogue-fused dSwiGLU =====
  printf("--- PATH C: Sonic-original GEMM2 epilogue dSwiGLU ---\n");
  for(int w=0;w<opt.warmup;++w){
    if(!opt.megakernel_like)
      launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
    if(opt.megakernel_like){
      launch_bf16_gemm_dswiglu_sonic_outputs_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dWgradAct,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
      reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
    } else {
      launch_bf16_gemm_dswiglu_packed(dY,dWdT,dGuBuf,nullptr,dGuOut,M,I,K,twoI,opt.sms,s);
    }
    launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
  }
  CHECK_CUDA(cudaStreamSynchronize(s));
  double totC=0;
  for(int it=0;it<opt.iters;++it)
    totC+=dt([&](){
      if(!opt.megakernel_like)
        launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
      if(opt.megakernel_like){
        launch_bf16_gemm_dswiglu_sonic_outputs_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dWgradAct,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
        reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
      } else {
        launch_bf16_gemm_dswiglu_packed(dY,dWdT,dGuBuf,nullptr,dGuOut,M,I,K,twoI,opt.sms,s);
      }
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    });
  double msC=totC/opt.iters;
  CHECK_CUDA(cudaMemcpy(hGuCheck.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
  print_error(opt.megakernel_like ? "SonicOriginal_dGU" : "EpiloguePacked_dGU",ref_dswiglu,hGuCheck,M*twoI);
  print_error(opt.megakernel_like ? "SonicOriginal_dx" : "EpiloguePacked_dx",ref_dx,hDx,M*K);
  if(opt.megakernel_like){
    std::vector<__nv_bfloat16> hWgradAct(M*I);
    CHECK_CUDA(cudaMemcpy(hWgradAct.data(),dWgradAct,upb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRouteGrad.data(),dRouteGrad,routeb,cudaMemcpyDeviceToHost));
    print_error("SonicOriginal_y1s",ref_wgrad_act,hWgradAct,M*I);
    print_error_f32("SonicOriginal_route_grad",ref_route_grad,hRouteGrad,M);
  }
  printf("SonicOriginal avg=%.4f ms  GEMM-TFLOPS=%.3f  speedup=%.2fx\n",msC,flA/(msC*1e-3)/1e12,msA/msC);
  if(opt.megakernel_like){
    for(int w=0;w<opt.warmup;++w){
      launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    double totAlias=0;
    for(int it=0;it<opt.iters;++it)
      totAlias+=dt([&](){
        launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
        CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
        launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
        megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
        launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
      });
    double msAlias=totAlias/opt.iters;
    std::vector<__nv_bfloat16> hAliasWgradAct(M*I), hAliasDx(M*K), hAliasDgu(M*twoI);
    CHECK_CUDA(cudaMemcpy(hAliasWgradAct.data(),dWgradAct,upb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hAliasDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hAliasDgu.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    print_error("LegacyRecomputeAlias_wgrad_act",ref_wgrad_act,hAliasWgradAct,M*I);
    print_error("LegacyRecomputeAlias_dGU_as_wgrad_dgu",ref_wgrad_dgu,hAliasDgu,M*twoI);
    print_error("LegacyRecomputeAlias_dx",ref_dx,hAliasDx,M*K);
    printf("Legacy recompute alias-dGu avg=%.4f ms  GEMM-TFLOPS=%.3f  speedup=%.2fx\n",msAlias,flA/(msAlias*1e-3)/1e12,msA/msAlias);

    for(int w=0;w<opt.warmup;++w){
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
      launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    double totSavedPreact=0;
    for(int it=0;it<opt.iters;++it)
      totSavedPreact+=dt([&](){
        CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
        launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
        megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
        launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);
      });
    double msSavedPreact=totSavedPreact/opt.iters;
    CHECK_CUDA(cudaMemcpy(hAliasWgradAct.data(),dWgradAct,upb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hAliasDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hAliasDgu.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRouteGrad.data(),dRouteGrad,routeb,cudaMemcpyDeviceToHost));
    print_error("SavedPreAct_wgrad_act",ref_wgrad_act,hAliasWgradAct,M*I);
    print_error("SavedPreAct_dGU",ref_wgrad_dgu,hAliasDgu,M*twoI);
    print_error("SavedPreAct_dx",ref_dx,hAliasDx,M*K);
    print_error_f32("SavedPreAct_route_grad",ref_route_grad,hRouteGrad,M);
    printf("SavedPreAct upper-bound avg=%.4f ms  remaining-GEMM-TFLOPS=%.3f  speedup-vs-baseline=%.2fx\n",
           msSavedPreact,((double)M*K*I*2+(double)M*twoI*K*2)/(msSavedPreact*1e-3)/1e12,msA/msSavedPreact);
  }
  printf("\n");

  // ===== Direct-layout weight experiment (BWD-12) =====
  if(opt.megakernel_like){
    printf("--- Direct-layout weight experiment ---\n");
    CHECK_CUDA(cudaMemcpy(dGuOut,ref_dswiglu.data(),gub,cudaMemcpyHostToDevice));
    double t_gemm2_mn=0.0,t_gemm3_mn=0.0;
    for(int w=0;w<opt.warmup;++w){
      launch_bf16_gemm_mn(dY,dWdMN,dUpBuf,M,I,K,opt.sms,s);
      launch_bf16_gemm_mn(dGuOut,dWgu,dDxPerm,M,K,twoI,opt.sms,s);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    for(int it=0;it<opt.iters;++it){
      t_gemm2_mn+=dt([&](){launch_bf16_gemm_mn(dY,dWdMN,dUpBuf,M,I,K,opt.sms,s);});
      t_gemm3_mn+=dt([&](){launch_bf16_gemm_mn(dGuOut,dWgu,dDxPerm,M,K,twoI,opt.sms,s);});
    }
    t_gemm2_mn/=opt.iters; t_gemm3_mn/=opt.iters;
    CHECK_CUDA(cudaMemcpy(hGuCheck.data(),dUpBuf,upb,cudaMemcpyDeviceToHost));
    print_error("DirectMN_GEMM2_up",ref_ga,hGuCheck,M*I);
    CHECK_CUDA(cudaMemcpy(hDx.data(),dDxPerm,xb,cudaMemcpyDeviceToHost));
    print_error("DirectMN_GEMM3_dx",ref_dx,hDx,M*K);
    printf("DirectMN pieces: GEMM2 %.4f ms, GEMM3 %.4f ms\n\n",t_gemm2_mn,t_gemm3_mn);
  }

  // ===== Stage breakdown =====
  printf("--- Stage breakdown ---\n");
  double t_gemm1=0,t_gemm2=0,t_dsw=0,t_gemm3=0,t_gemm2_epi=0,t_gemm2_epi_packed=0,t_side_from_dgu=0,t_side_act_only=0,t_gemm2_epi_packed_side=0;
  double t_gemm2_epi_packed_partial=0,t_route_grad_reduce=0,t_gemm2_epi_packed_partial_total=0;
  for(int it=0;it<opt.iters;++it){
    t_gemm1+=dt([&](){launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);});
    t_gemm2+=dt([&](){launch_bf16_gemm(dY,dWdT,dUpBuf,M,I,K,opt.sms,s);});
    if(opt.megakernel_like){
      t_dsw+=dt([&](){
        CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
        megakernel_like_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,twoI,I);
      });
    } else {
      t_dsw+=dt([&](){baseline_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);});
    }
    t_gemm3+=dt([&](){launch_bf16_gemm(dGuOut,dWguT,dDxPerm,M,K,twoI,opt.sms,s);});
    if(!opt.megakernel_like)
      t_gemm2_epi+=dt([&](){launch_bf16_gemm_dswiglu(dY,dWdT,dGuBuf,dGuOut,M,I,K,twoI,opt.sms,s);});
    t_gemm2_epi_packed+=dt([&](){
      if(opt.megakernel_like){
        CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
        launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      } else {
        launch_bf16_gemm_dswiglu_packed(dY,dWdT,dGuBuf,nullptr,dGuOut,M,I,K,twoI,opt.sms,s);
      }
    });
    if(opt.megakernel_like){
      t_gemm2_epi_packed_partial+=dt([&](){
        launch_bf16_gemm_dswiglu_packed_route_grad_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
      });
      t_route_grad_reduce+=dt([&](){
        reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
      });
      t_gemm2_epi_packed_partial_total+=dt([&](){
        launch_bf16_gemm_dswiglu_packed_route_grad_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
        reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
      });
    }
    if(opt.megakernel_like){
      t_side_from_dgu+=dt([&](){
        megakernel_like_side_outputs_from_dgu_kernel<<<M,256,0,s>>>(dGuBuf,dGuOut,dRoute,dWgradAct,dWgradDgu,nullptr,M,twoI,I);
      });
      t_side_act_only+=dt([&](){
        megakernel_like_side_outputs_act_only_kernel<<<M,256,0,s>>>(dGuBuf,dRoute,dWgradAct,M,twoI,I);
      });
      t_gemm2_epi_packed_side+=dt([&](){
        CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
        launch_bf16_gemm_dswiglu_packed_side_outputs(dY,dWdT,dGuBuf,dRoute,dGuOut,dWgradAct,dWgradDgu,dRouteGrad,M,I,K,twoI,opt.sms,s);
      });
    }

  }
  t_gemm1/=opt.iters; t_gemm2/=opt.iters; t_dsw/=opt.iters; t_gemm3/=opt.iters; t_gemm2_epi/=opt.iters; t_gemm2_epi_packed/=opt.iters;
  t_side_from_dgu/=opt.iters; t_side_act_only/=opt.iters; t_gemm2_epi_packed_side/=opt.iters;
  t_gemm2_epi_packed_partial/=opt.iters; t_route_grad_reduce/=opt.iters; t_gemm2_epi_packed_partial_total/=opt.iters;
  if(opt.megakernel_like){
    printf("SavedPreAct baseline pieces: GEMM2 %.4f ms, scalar dSwiGLU+side %.4f ms, GEMM3 %.4f ms\n",t_gemm2,t_dsw,t_gemm3);
    printf("Legacy recompute GEMM1 cost for reference: %.4f ms\n",t_gemm1);
    printf("Legacy split epilogue: GEMM2+dSwiGLU %.4f ms, side-from-dGu %.4f ms, GEMM3 %.4f ms\n",t_gemm2_epi_packed,t_side_from_dgu,t_gemm3);
    printf("Legacy alias-dGu split: GEMM2+dSwiGLU %.4f ms, side-act-only %.4f ms, GEMM3 %.4f ms\n",t_gemm2_epi_packed,t_side_act_only,t_gemm3);
    printf("Sonic partial route_grad: GEMM2+dSwiGLU+side %.4f ms, reduce %.4f ms, total %.4f ms\n",t_gemm2_epi_packed_partial,t_route_grad_reduce,t_gemm2_epi_packed_partial_total);
    printf("Sonic monolithic: GEMM2+dSwiGLU+side %.4f ms\n\n",t_gemm2_epi_packed_side);

    printf("--- Route-grad timing isolated ---\n");
    launch_bf16_gemm(dX,dWgu,dGuBuf,M,twoI,K,opt.sms,s);
    for(int w=0; w<opt.warmup; ++w){
      CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
      launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      launch_bf16_gemm_dswiglu_packed_route_grad_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
      reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    double t_route_atomic_iso=0,t_route_partial_epi_iso=0,t_route_partial_reduce_iso=0,t_route_partial_total_iso=0;
    for(int it=0; it<opt.iters; ++it)
      t_route_atomic_iso+=dt([&](){
        CHECK_CUDA(cudaMemsetAsync(dRouteGrad,0,routeb,s));
        launch_bf16_gemm_dswiglu_packed_route_grad(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGrad,M,I,K,twoI,opt.sms,s);
      });
    for(int it=0; it<opt.iters; ++it)
      t_route_partial_epi_iso+=dt([&](){
        launch_bf16_gemm_dswiglu_packed_route_grad_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
      });
    launch_bf16_gemm_dswiglu_packed_route_grad_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
    CHECK_CUDA(cudaStreamSynchronize(s));
    for(int it=0; it<opt.iters; ++it)
      t_route_partial_reduce_iso+=dt([&](){
        reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
      });
    for(int it=0; it<opt.iters; ++it)
      t_route_partial_total_iso+=dt([&](){
        launch_bf16_gemm_dswiglu_packed_route_grad_partial(dY,dWdT,dGuBuf,dRoute,dGuOut,dRouteGradPartial,M,I,K,twoI,opt.sms,s);
        reduce_route_grad_partials_kernel<<<M,64,0,s>>>(dRouteGradPartial,dRouteGrad,M,num_n_tiles);
      });
    t_route_atomic_iso/=opt.iters;
    t_route_partial_epi_iso/=opt.iters;
    t_route_partial_reduce_iso/=opt.iters;
    t_route_partial_total_iso/=opt.iters;
    printf("Atomic route_grad epilogue %.4f ms\n",t_route_atomic_iso);
    printf("Partial route_grad epilogue %.4f ms, reduce %.4f ms, total %.4f ms\n\n",t_route_partial_epi_iso,t_route_partial_reduce_iso,t_route_partial_total_iso);
  } else {
    printf("Baseline pieces: GEMM1 %.4f ms, GEMM2 %.4f ms, dSwiGLU %.4f ms, GEMM3 %.4f ms\n",t_gemm1,t_gemm2,t_dsw,t_gemm3);
    printf("EpiFused pieces: GEMM1 %.4f ms, GEMM2+dSwiGLU %.4f ms, GEMM3 %.4f ms\n",t_gemm1,t_gemm2_epi,t_gemm3);
    printf("EpiPacked pieces: GEMM1 %.4f ms, GEMM2+dSwiGLU %.4f ms, GEMM3 %.4f ms\n\n",t_gemm1,t_gemm2_epi_packed,t_gemm3);
  }

  // ===== dSwiGLU-only microbench =====
  if(!opt.megakernel_like){
    printf("--- dSwiGLU-only comparison ---\n");
    CHECK_CUDA(cudaMemcpy(dGuBuf,ref_gu.data(),gub,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dUpBuf,ref_ga.data(),upb,cudaMemcpyHostToDevice));
    for(int w=0;w<opt.warmup;++w){
      baseline_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
      fused_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    double bds=0,fds=0;
    for(int it=0;it<opt.iters;++it){
      bds+=dt([&](){baseline_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);});
      fds+=dt([&](){fused_dswiglu_kernel<<<M,256,0,s>>>(dGuBuf,dUpBuf,dGuOut,M,twoI,I);});
    }
    bds/=opt.iters; fds/=opt.iters;
    printf("Baseline dSwiGLU: %.2f us\n",bds*1000);
    printf("Fused    dSwiGLU: %.2f us\n",fds*1000);
    printf("dSwiGLU speedup: %.2fx\n",bds/fds);
    std::vector<__nv_bfloat16> hGuOut(M*twoI);
    CHECK_CUDA(cudaMemcpy(hGuOut.data(),dGuOut,gub,cudaMemcpyDeviceToHost));
    print_error("dSwiGLU_out",ref_dswiglu,hGuOut,M*twoI);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(dX)); CHECK_CUDA(cudaFree(dWgu)); CHECK_CUDA(cudaFree(dWguT));
  CHECK_CUDA(cudaFree(dY)); CHECK_CUDA(cudaFree(dWdT)); CHECK_CUDA(cudaFree(dWdMN));
  CHECK_CUDA(cudaFree(dGuBuf)); CHECK_CUDA(cudaFree(dUpBuf));
  CHECK_CUDA(cudaFree(dDxPerm)); CHECK_CUDA(cudaFree(dGuOut));
  if(opt.megakernel_like){
    CHECK_CUDA(cudaFree(dRoute));
    CHECK_CUDA(cudaFree(dRouteGrad));
    CHECK_CUDA(cudaFree(dRouteGradPartial));
    CHECK_CUDA(cudaFree(dWgradAct));
    CHECK_CUDA(cudaFree(dWgradDgu));
  }
  CHECK_CUDA(cudaEventDestroy(ev0)); CHECK_CUDA(cudaEventDestroy(ev1));
  CHECK_CUDA(cudaStreamDestroy(s));
  printf("\nDone.\n");
  return 0;
}
