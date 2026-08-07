// umma_gather_ws_dg.cu — verify index-based A-row gather fused into the DeepGEMM
// warp-specialized GEMM's TMA load, WITHOUT pre-permuting A into a contiguous
// buffer. This models the megakernel optimization: instead of gathering
// combine_input[recv_token_idx] -> input_buf and then TMA-loading input_buf,
// issue one single-row TMA per M row directly from the scattered buffer using
// A_idx (== s_recv_token_idx).
//
// Correctness idea:
//   A_full[m]            : logical GEMM input row m (reference order)
//   A_scattered[perm[m]] = A_full[m]   (scattered storage)
//   A_idx[m] = perm[m]                 (gather index)
//   gather load reads A_scattered[A_idx[m]] == A_full[m]  => GEMM(A_full) result
//
// We run TWO cases:
//   (1) A_idx = identity, A_scattered = A_full   -> must match plain GEMM ref
//   (2) A_idx = random permutation               -> must ALSO match the SAME ref
//
// Build (B30Z compute_103a):
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        umma_gather_ws_dg.cu -o umma_gather_ws_dg -lcuda

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <unistd.h>
#include <sys/syscall.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/types.cuh>
#include "sm100_bf16_gemm_dg_gather.cuh"

#define CHECK_CUDA(call)                                                         \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                               \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":"      \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call)                                                           \
  do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s;                  \
    cuGetErrorString(_e,&s);                                                     \
    std::cerr<<"CU error "<<s<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;       \
    std::exit(1);} } while(0)

// Config: match umma_swiglu_ws_dg.cu, but BLOCK_K must equal swizzle atom for
// the single-row gather path (BLOCK_K*2 == SWZ_A). BLOCK_K=64 bf16 = 128B = SWZ_A.
static constexpr uint32_t CFG_BLOCK_M = 128;
static constexpr uint32_t CFG_BLOCK_N = 128;
static constexpr uint32_t CFG_BLOCK_K = 64;
static constexpr uint32_t CFG_NUM_GROUPS = 1;
static constexpr uint32_t CFG_SWZ_A = 128;
static constexpr uint32_t CFG_SWZ_B = 128;
static constexpr uint32_t CFG_SWZ_CD = 128;
static constexpr uint32_t CFG_NUM_STAGES = 4;
static constexpr uint32_t CFG_NON_EPI_THREADS = 128;
static constexpr uint32_t CFG_EPI_THREADS = 128;
static constexpr uint32_t CFG_PHYSICAL_THREADS = 800;
// Template runs the GEMM logically as 1-CTA (kNumMulticast=1: 1-SM MMA,
// non-multicast gather4, 1x barrier accounting). The megakernel however LAUNCHES
// the whole grid with clusterDim=2 (fixed, S4.2). This must match that env: launch
// with clusterDim=2 but run 1-CTA GEMM logic. So keep CFG_NUM_MULTICAST=1 for the
// template, and separately control the LAUNCH cluster dim.
static constexpr uint32_t CFG_NUM_MULTICAST = 1;
static constexpr bool     CFG_MCAST_ON_A = false;
// Physical cluster dim used at launch (independent of the MMA multicast factor).
// Set to 2 to reproduce the megakernel's fixed clusterDim=2 launch while the GEMM
// runs 1-CTA logic — the key scenario for "1-CTA gate/up inside a 2-CTA launch".
static int g_launch_cluster_dim = 1;
static constexpr uint32_t CFG_KALIGN = 128;
static constexpr bool     CFG_SWAP_AB = false;
static constexpr bool     CFG_ENSURE_ZERO_PAD = false;
static constexpr bool     CFG_WITH_ACCUM = false;
static constexpr uint64_t CFG_TC_UTIL = 100;
static constexpr uint32_t CFG_NUM_SMS = 132;

static constexpr uint32_t LOAD_BLOCK_M = CFG_BLOCK_M / (CFG_MCAST_ON_A ? CFG_NUM_MULTICAST : 1);
static constexpr uint32_t LOAD_BLOCK_N = CFG_BLOCK_N / (CFG_MCAST_ON_A ? 1 : CFG_NUM_MULTICAST);
static constexpr uint32_t STORE_BLOCK_M = (CFG_BLOCK_M < 128 ? CFG_BLOCK_M : 128);
static constexpr uint32_t STORE_BLOCK_N = CFG_SWZ_CD / sizeof(cutlass::bfloat16_t);

static CUtensorMap make_tma_2d(const void* ptr, CUtensorMapDataType dtype,
                               int gmem_inner, int gmem_outer,
                               int smem_inner, int smem_outer,
                               int gmem_outer_stride_elems, int elem_size,
                               int swizzle_bytes) {
  CUtensorMap tm;
  int si = smem_inner;
  if (swizzle_bytes != 0) si = swizzle_bytes / elem_size;
  const cuuint64_t gdims[2]   = { (cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer };
  const cuuint32_t sdims[2]   = { (cuuint32_t)si, (cuuint32_t)smem_outer };
  const cuuint64_t gstr[1]    = { (cuuint64_t)gmem_outer_stride_elems * elem_size };
  const cuuint32_t estr[2]    = { 1, 1 };
  CUtensorMapSwizzle sw =
      swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
      swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
      swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  : CU_TENSOR_MAP_SWIZZLE_NONE;
  CHECK_CU(cuTensorMapEncodeTiled(
      &tm, dtype, 2, (void*)ptr, gdims, gstr, sdims, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}

// A: [M,K] K-major. smem box outer = LOAD_BLOCK_M (normal block load).
static CUtensorMap make_a_desc(const __nv_bfloat16* a, int M, int K) {
  return make_tma_2d(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, M,
                     CFG_BLOCK_K, LOAD_BLOCK_M, K, sizeof(__nv_bfloat16), CFG_SWZ_A);
}
// A single-row: identical to make_a_desc but smem box outer = 1, so each TMA
// loads exactly one M row of BLOCK_K. Used for the index-gather path.
static CUtensorMap make_a_row_desc(const __nv_bfloat16* a, int M, int K) {
  return make_tma_2d(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, M,
                     CFG_BLOCK_K, 1, K, sizeof(__nv_bfloat16), CFG_SWZ_A);
}
static CUtensorMap make_b_desc(const __nv_bfloat16* b, int N, int K) {
  return make_tma_2d(b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, N,
                     CFG_BLOCK_K, LOAD_BLOCK_N, K, sizeof(__nv_bfloat16), CFG_SWZ_B);
}
static CUtensorMap make_cd_desc(const __nv_bfloat16* d, int M, int N) {
  return make_tma_2d(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, N, M,
                     STORE_BLOCK_N, STORE_BLOCK_M, N, sizeof(__nv_bfloat16), CFG_SWZ_CD);
}

// Plain GEMM (no gather): baseline path (a_row_idx=nullptr).
void launch_gemm_plain(const __nv_bfloat16* dA, const __nv_bfloat16* dB, __nv_bfloat16* dD,
                       int M, int N, int K, cudaStream_t stream) {
  auto dmap_a  = make_a_desc(dA, M, K);
  auto dmap_b  = make_b_desc(dB, N, K);
  auto dmap_cd = make_cd_desc(dD, M, N);
  auto dmap_a_row = make_a_row_desc(dA, M, K);
  auto kernel = &deep_gemm::sm100_bf16_gemm_gather_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      0u, 0u, 0u,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_BLOCK_K,
      CFG_NUM_GROUPS, CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD, CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS, CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      CFG_NUM_SMS, CFG_KALIGN, CFG_SWAP_AB, CFG_ENSURE_ZERO_PAD,
      deep_gemm::GemmType::Normal, CFG_WITH_ACCUM, cutlass::bfloat16_t,
      CFG_TC_UTIL, false, CFG_PHYSICAL_THREADS>;
  int smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CFG_NUM_SMS,1,1); cfg.blockDim = dim3(CFG_PHYSICAL_THREADS,1,1);
  cfg.dynamicSmemBytes = smem_bytes; cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = g_launch_cluster_dim; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs; cfg.numAttrs = 1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel,
      (int*)nullptr, (uint32_t)M, (uint32_t)N, (uint32_t)K,
      dmap_a, dmap_b, dmap_cd,
      (const cutlass::bfloat16_t*)nullptr, (const float*)nullptr, (uint32_t)N,
      (const int*)nullptr, dmap_a_row));
}

// Gather GEMM: A rows read via a_row_idx from scattered dA.
void launch_gemm_gather(const __nv_bfloat16* dA_scattered, const __nv_bfloat16* dB, __nv_bfloat16* dD,
                        const int* dA_idx, int M, int N, int K, cudaStream_t stream) {
  auto dmap_a  = make_a_desc(dA_scattered, M, K);       // unused by gather path but required arg
  auto dmap_b  = make_b_desc(dB, N, K);
  auto dmap_cd = make_cd_desc(dD, M, N);
  auto dmap_a_row = make_a_row_desc(dA_scattered, M, K);
  auto kernel = &deep_gemm::sm100_bf16_gemm_gather_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      0u, 0u, 0u,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_BLOCK_K,
      CFG_NUM_GROUPS, CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD, CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS, CFG_NUM_MULTICAST, CFG_MCAST_ON_A,
      CFG_NUM_SMS, CFG_KALIGN, CFG_SWAP_AB, CFG_ENSURE_ZERO_PAD,
      deep_gemm::GemmType::Normal, CFG_WITH_ACCUM, cutlass::bfloat16_t,
      CFG_TC_UTIL, false, CFG_PHYSICAL_THREADS>;
  int smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CFG_NUM_SMS,1,1); cfg.blockDim = dim3(CFG_PHYSICAL_THREADS,1,1);
  cfg.dynamicSmemBytes = smem_bytes; cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = g_launch_cluster_dim; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs; cfg.numAttrs = 1;
  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel,
      (int*)nullptr, (uint32_t)M, (uint32_t)N, (uint32_t)K,
      dmap_a, dmap_b, dmap_cd,
      (const cutlass::bfloat16_t*)nullptr, (const float*)nullptr, (uint32_t)N,
      dA_idx, dmap_a_row));
}

static float relative_error(const std::vector<float>& t,const std::vector<float>& r){
  double num=0,den=0; for(size_t i=0;i<r.size();++i){double d=t[i]-r[i];num+=d*d;den+=double(r[i])*r[i];}
  return float(std::sqrt(num)/(std::sqrt(den)+1e-12));
}

int main(int argc,char**argv){
  int M=256,K=4096,N=4096;
  fprintf(stderr,"Gather-A DeepGEMM verify M=%d K=%d N=%d\n",M,K,N); fflush(stderr);
  CHECK_CU(cuInit(0));
  std::mt19937 gen(1234); std::uniform_real_distribution<float> dist(-1.f,1.f);
  auto bf=[](float v){return __bfloat162float(__float2bfloat16(v));};

  std::vector<float> hA(M*K),hWg(N*K);
  for(auto&x:hA)x=dist(gen); for(auto&x:hWg)x=dist(gen);
  std::vector<float> Ab(M*K),Wgb(N*K);
  for(int i=0;i<M*K;++i)Ab[i]=bf(hA[i]);
  for(int i=0;i<N*K;++i)Wgb[i]=bf(hWg[i]);

  // Reference: gate[m,n] = sum_k A_full[m,k] * Wg[n,k]
  std::vector<float> ref_gate(M*N);
  for(int m=0;m<M;++m)for(int n=0;n<N;++n){
    double sg=0; for(int k=0;k<K;++k) sg+=double(Ab[m*K+k])*Wgb[n*K+k];
    ref_gate[m*N+n]=float(sg);
  }

  // Random permutation perm: A_full[m] stored at A_scattered[perm[m]].
  std::vector<int> perm(M); std::iota(perm.begin(),perm.end(),0);
  std::shuffle(perm.begin(),perm.end(),gen);

  std::vector<__nv_bfloat16> A16(M*K), Ascat16(M*K), Wg16(N*K);
  for(int i=0;i<M*K;++i)A16[i]=__float2bfloat16(hA[i]);
  for(int i=0;i<N*K;++i)Wg16[i]=__float2bfloat16(hWg[i]);
  // A_scattered[perm[m]] = A_full[m]  => A_idx[m] = perm[m]
  for(int m=0;m<M;++m) for(int k=0;k<K;++k) Ascat16[(size_t)perm[m]*K+k]=A16[(size_t)m*K+k];
  std::vector<int> Aidx(M); for(int m=0;m<M;++m) Aidx[m]=perm[m];
  std::vector<int> Aid_identity(M); std::iota(Aid_identity.begin(),Aid_identity.end(),0);

  __nv_bfloat16 *dA,*dAscat,*dWg,*dGate; int* dAidx; int* dAidI;
  CHECK_CUDA(cudaMalloc(&dA,(size_t)M*K*2)); CHECK_CUDA(cudaMalloc(&dAscat,(size_t)M*K*2));
  CHECK_CUDA(cudaMalloc(&dWg,(size_t)N*K*2)); CHECK_CUDA(cudaMalloc(&dGate,(size_t)M*N*2));
  CHECK_CUDA(cudaMalloc(&dAidx,(size_t)M*sizeof(int))); CHECK_CUDA(cudaMalloc(&dAidI,(size_t)M*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dA,A16.data(),(size_t)M*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAscat,Ascat16.data(),(size_t)M*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWg,Wg16.data(),(size_t)N*K*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAidx,Aidx.data(),(size_t)M*sizeof(int),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAidI,Aid_identity.data(),(size_t)M*sizeof(int),cudaMemcpyHostToDevice));

  cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));
  // KEY SCENARIO: launch grid with clusterDim=2 (matches the megakernel's fixed
  // S4.2 clusterDim=2 launch) while the GEMM runs 1-CTA logic (kNumMulticast=1).
  // This validates "1-CTA gate/up gather4 inside a 2-CTA cluster launch".
  g_launch_cluster_dim = 2;
  fprintf(stderr,"[cfg] launch clusterDim=%d, GEMM kNumMulticast=%u (1-CTA logic)\n",
          g_launch_cluster_dim, CFG_NUM_MULTICAST);
  auto readback=[&](const char* tag)->float{
    std::vector<__nv_bfloat16> hb(M*N); std::vector<float> hf(M*N);
    CHECK_CUDA(cudaMemcpy(hb.data(),dGate,(size_t)M*N*2,cudaMemcpyDeviceToHost));
    for(int i=0;i<M*N;++i)hf[i]=__bfloat162float(hb[i]);
    float e=relative_error(hf,ref_gate);
    fprintf(stderr,"  [%s] rel err vs ref = %.8g\n",tag,e); return e;
  };

  // Case 0: plain GEMM on A_full (sanity: header still correct).
  CHECK_CUDA(cudaMemset(dGate,0,(size_t)M*N*2));
  launch_gemm_plain(dA,dWg,dGate,M,N,K,s); CHECK_CUDA(cudaStreamSynchronize(s));
  float e_plain = readback("plain A_full");

  // Case 1: gather with identity index on A_full (must equal plain).
  CHECK_CUDA(cudaMemset(dGate,0,(size_t)M*N*2));
  launch_gemm_gather(dA,dWg,dGate,dAidI,M,N,K,s); CHECK_CUDA(cudaStreamSynchronize(s));
  float e_ident = readback("gather identity");

  // Case 2: gather with random permutation on A_scattered (must equal plain).
  CHECK_CUDA(cudaMemset(dGate,0,(size_t)M*N*2));
  launch_gemm_gather(dAscat,dWg,dGate,dAidx,M,N,K,s); CHECK_CUDA(cudaStreamSynchronize(s));
  float e_perm = readback("gather permuted");

  int status = (e_plain<5e-2f && e_ident<5e-2f && e_perm<5e-2f) ? 0 : 1;
  fprintf(stderr,"Correctness: %s\n", status==0 ? "PASS" : "FAIL");
  syscall(SYS_exit_group, status);
}
