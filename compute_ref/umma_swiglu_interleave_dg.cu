// umma_swiglu_interleave_dg.cu — quack-style interleaved gate/up -> SwiGLU fused
// epilogue, validated on the DeepGEMM sm100_bf16_gemm_impl kernel for BOTH the
// 1-CTA (num_multicast=1) and 2-CTA (num_multicast=2) paths.
//
// Weight layout (element interleave, gran=1), Wgu = [2I, K] K-major:
//   row 2j   = Wg[j]   (gate for logical channel j)
//   row 2j+1 = Wu[j]   (up   for logical channel j)
// GEMM: GU = A @ Wgu^T -> [M, 2I], TMEM col 2j=gate_j, 2j+1=up_j.
// Fused epilogue (sm100_store_swiglu_interleaved) reads gate/up straight from
// TMEM, computes act[m,j] = silu(gate)*up*route[m], writes act[M, I] (half width).
// NOTHING round-trips through GMEM — this is the quack GemmGatedMixin scheme.
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        umma_swiglu_interleave_dg.cu -o umma_swiglu_interleave_dg -lcuda
//   ./umma_swiglu_interleave_dg            # default M=256 K=4096 I=4096

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <unistd.h>
#include <sys/syscall.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/types.cuh>
#include "sm100_bf16_gemm_dg_copy.cuh"

#define CHECK_CUDA(call)                                                         \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                               \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":"      \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call)                                                           \
  do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s;                  \
    cuGetErrorString(_e,&s);                                                     \
    std::cerr<<"CU error "<<s<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;       \
    std::exit(1);} } while(0)

// ======================= GEMM config =======================
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
static constexpr uint32_t CFG_PHYSICAL_THREADS = 800;   // mirror megakernel compute CTA
static constexpr bool     CFG_MCAST_ON_A = false;       // multicast on B/N
static constexpr uint32_t CFG_KALIGN = 128;
static constexpr bool     CFG_SWAP_AB = false;
static constexpr bool     CFG_ENSURE_ZERO_PAD = false;
static constexpr bool     CFG_WITH_ACCUM = false;
static constexpr uint64_t CFG_TC_UTIL = 100;
static constexpr uint32_t CFG_NUM_SMS = 132;

static constexpr uint32_t STORE_BLOCK_N = CFG_SWZ_CD / sizeof(cutlass::bfloat16_t);  // 64

// ======================= TMA descriptor builders =======================
template <uint32_t NUM_MULTICAST>
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

// A: [M,K] K-major. LOAD_BLOCK_M = BLOCK_M/(mcastA?MC:1).
template <uint32_t MC>
static CUtensorMap make_a_desc(const __nv_bfloat16* a, int M, int K) {
  constexpr uint32_t LOAD_BLOCK_M = CFG_BLOCK_M / (CFG_MCAST_ON_A ? MC : 1);
  return make_tma_2d<MC>(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, M,
                         CFG_BLOCK_K, LOAD_BLOCK_M, K, sizeof(__nv_bfloat16), CFG_SWZ_A);
}
// B: [N,K] K-major. LOAD_BLOCK_N = BLOCK_N/(mcastA?1:MC).
template <uint32_t MC>
static CUtensorMap make_b_desc(const __nv_bfloat16* b, int N, int K) {
  constexpr uint32_t LOAD_BLOCK_N = CFG_BLOCK_N / (CFG_MCAST_ON_A ? 1 : MC);
  return make_tma_2d<MC>(b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, K, N,
                         CFG_BLOCK_K, LOAD_BLOCK_N, K, sizeof(__nv_bfloat16), CFG_SWZ_B);
}
// CD: act [M, I] row-major bf16. STORE_BLOCK_N=64, STORE_BLOCK_M=128.
template <uint32_t MC>
static CUtensorMap make_cd_desc(const __nv_bfloat16* d, int M, int I) {
  return make_tma_2d<MC>(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, I, M,
                         STORE_BLOCK_N, 128, I, sizeof(__nv_bfloat16), CFG_SWZ_CD);
}

// ======================= Launch (templated on multicast) =======================
template <uint32_t NUM_MULTICAST>
void launch_interleave_swiglu(const __nv_bfloat16* dA, const __nv_bfloat16* dWgu,
                              __nv_bfloat16* dAct, const float* dRoute,
                              int M, int I, int K, cudaStream_t stream,
                              __nv_bfloat16* dPreact = nullptr,
                              const int* dRecvIdx = nullptr,
                              const int* dTopkIdx = nullptr,
                              int preactNumTopk = 0,
                              int preactStride = 0) {
  const int N2 = 2 * I;   // GEMM N dimension is the interleaved GU width (2I)
  auto dmap_a  = make_a_desc<NUM_MULTICAST>(dA, M, K);
  auto dmap_b  = make_b_desc<NUM_MULTICAST>(dWgu, N2, K);
  auto dmap_cd = make_cd_desc<NUM_MULTICAST>(dAct, M, I);

  auto kernel = &deep_gemm::sm100_bf16_gemm_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      0u, 0u, 0u,
      CFG_BLOCK_M, CFG_BLOCK_N, CFG_BLOCK_K,
      CFG_NUM_GROUPS,
      CFG_SWZ_A, CFG_SWZ_B, CFG_SWZ_CD,
      CFG_NUM_STAGES,
      CFG_NON_EPI_THREADS, CFG_EPI_THREADS,
      NUM_MULTICAST, CFG_MCAST_ON_A,
      CFG_NUM_SMS,
      CFG_KALIGN,
      CFG_SWAP_AB, CFG_ENSURE_ZERO_PAD,
      deep_gemm::GemmType::Normal, CFG_WITH_ACCUM, cutlass::bfloat16_t,
      CFG_TC_UTIL,
      /*kFuseSwiGLU=*/false, /*kFuseSwiGLUInterleaved=*/true,
      CFG_PHYSICAL_THREADS>;

  int smem_bytes = 227 * 1024;
  CHECK_CUDA(cudaFuncSetAttribute((const void*)kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim  = dim3(CFG_NUM_SMS, 1, 1);
  cfg.blockDim = dim3(CFG_PHYSICAL_THREADS, 1, 1);
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = NUM_MULTICAST;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel,
      (int*)nullptr, (uint32_t)M, (uint32_t)N2, (uint32_t)K,
      dmap_a, dmap_b, dmap_cd,
      /*gate_ptr=*/(const cutlass::bfloat16_t*)nullptr, dRoute, (uint32_t)I,
      /*wgrad_act_ptr=*/(cutlass::bfloat16_t*)nullptr,
      /*wgrad_dgu_ptr=*/(cutlass::bfloat16_t*)nullptr,
      /*route_grad_ptr=*/(float*)nullptr,
      reinterpret_cast<cutlass::bfloat16_t*>(dPreact), dRecvIdx, dTopkIdx,
      (uint32_t)preactNumTopk, (uint32_t)preactStride));
}

// ======================= WMMA fused reference =======================
using namespace nvcuda;
static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void swiglu_wmma_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ W_gate,
                                   const __nv_bfloat16* __restrict__ W_up,
                                   __nv_bfloat16* __restrict__ act,
                                   const float* __restrict__ route_w,
                                   int valid_rows, int M, int K, int N) {
    extern __shared__ float smem_buf[];
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tiles_m = (M + WMMA_M - 1) / WMMA_M;
    const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
    const int total_tiles = tiles_m * tiles_n;
    float* gate_s = smem_buf + warp_id * (2 * WMMA_M * WMMA_N);
    float* up_s   = gate_s + WMMA_M * WMMA_N;
    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
        int tr = tile_idx / tiles_n, tc = tile_idx % tiles_n;
        int row_off = tr * WMMA_M, col_off = tc * WMMA_N;
        if (row_off >= M || col_off >= N) continue;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bg_frag, bu_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cg_frag, cu_frag;
        wmma::fill_fragment(cg_frag, 0.0f);
        wmma::fill_fragment(cu_frag, 0.0f);
        for (int k = 0; k < K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A + row_off * K + k, K);
            wmma::load_matrix_sync(bg_frag, W_gate + col_off * K + k, K);
            wmma::load_matrix_sync(bu_frag, W_up   + col_off * K + k, K);
            wmma::mma_sync(cg_frag, a_frag, bg_frag, cg_frag);
            wmma::mma_sync(cu_frag, a_frag, bu_frag, cu_frag);
        }
        wmma::store_matrix_sync(gate_s, cg_frag, WMMA_N, wmma::mem_row_major);
        wmma::store_matrix_sync(up_s,   cu_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N, col = i % WMMA_N;
            int orow = row_off + row, ocol = col_off + col;
            if (orow >= M || ocol >= N) continue;
            if (orow < valid_rows) {
                float g = gate_s[i], u = up_s[i];
                float silu = g * (1.0f / (1.0f + __expf(-g)));
                act[orow * N + ocol] = __float2bfloat16(silu * u * route_w[orow]);
            } else {
                act[orow * N + ocol] = __float2bfloat16(0.0f);
            }
        }
    }
}

static float relative_error(const std::vector<float>& t, const std::vector<float>& r) {
    double num = 0, den = 0;
    for (size_t i = 0; i < r.size(); ++i) { double d = t[i]-r[i]; num += d*d; den += double(r[i])*r[i]; }
    return float(std::sqrt(num) / (std::sqrt(den) + 1e-12));
}

int main(int argc, char** argv) {
    int M = 256, K = 4096, I = 4096;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); I = atoi(argv[3]); }
    const int N2 = 2 * I;
    fprintf(stderr, "[interleave SwiGLU DG] M=%d K=%d I=%d (gran=1 element interleave, fused epilogue)\n", M, K, I);
    CHECK_CU(cuInit(0));

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto bf = [](float v){ return __bfloat162float(__float2bfloat16(v)); };

    std::vector<float> hA(M*K), hWg(I*K), hWu(I*K), hRoute(M);
    for (auto& x : hA) x = dist(gen);
    for (auto& x : hWg) x = dist(gen);
    for (auto& x : hWu) x = dist(gen);
    for (auto& x : hRoute) x = dist(gen);

    // FP32 ref: silu(bf16(gate))*bf16(up)*route.
    std::vector<float> Ab(M*K), Wgb(I*K), Wub(I*K);
    for (int i = 0; i < M*K; ++i) Ab[i] = bf(hA[i]);
    for (int i = 0; i < I*K; ++i) { Wgb[i] = bf(hWg[i]); Wub[i] = bf(hWu[i]); }
    std::vector<float> ref_act(M*I);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < I; ++n) {
            double sg = 0, su = 0;
            for (int k = 0; k < K; ++k) { sg += double(Ab[m*K+k])*Wgb[n*K+k]; su += double(Ab[m*K+k])*Wub[n*K+k]; }
            float g = bf(float(sg)), u = bf(float(su));
            ref_act[m*I+n] = bf((g*(1.0f/(1.0f+std::exp(-g))))*u*hRoute[m]);
        }
    fprintf(stderr, "[host] SwiGLU FP32 ref done\n");

    // Element-interleave (gran=1) concat weight Wgu [2I, K]:
    //   row 2j   = Wg[j], row 2j+1 = Wu[j].
    std::vector<__nv_bfloat16> A16(M*K), Wg16(I*K), Wu16(I*K), Wgu16((size_t)N2*K);
    for (int i = 0; i < M*K; ++i) A16[i] = __float2bfloat16(hA[i]);
    for (int i = 0; i < I*K; ++i) { Wg16[i] = __float2bfloat16(hWg[i]); Wu16[i] = __float2bfloat16(hWu[i]); }
    for (int j = 0; j < I; ++j)
        for (int k = 0; k < K; ++k) {
            Wgu16[(size_t)(2*j)   * K + k] = Wg16[(size_t)j * K + k];
            Wgu16[(size_t)(2*j+1) * K + k] = Wu16[(size_t)j * K + k];
        }

    __nv_bfloat16 *dA, *dWgu, *dWg, *dWu, *dAct1, *dAct2, *dAct_wmma;
    float* dRoute;
    CHECK_CUDA(cudaMalloc(&dA, (size_t)M*K*2));
    CHECK_CUDA(cudaMalloc(&dWgu, (size_t)N2*K*2));
    CHECK_CUDA(cudaMalloc(&dWg, (size_t)I*K*2));
    CHECK_CUDA(cudaMalloc(&dWu, (size_t)I*K*2));
    CHECK_CUDA(cudaMalloc(&dAct1, (size_t)M*I*2));
    CHECK_CUDA(cudaMalloc(&dAct2, (size_t)M*I*2));
    CHECK_CUDA(cudaMalloc(&dAct_wmma, (size_t)M*I*2));
    CHECK_CUDA(cudaMalloc(&dRoute, (size_t)M*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A16.data(), (size_t)M*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWgu, Wgu16.data(), (size_t)N2*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWg, Wg16.data(), (size_t)I*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWu, Wu16.data(), (size_t)I*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRoute, hRoute.data(), (size_t)M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dAct1, 0, (size_t)M*I*2));
    CHECK_CUDA(cudaMemset(dAct2, 0, (size_t)M*I*2));
    CHECK_CUDA(cudaMemset(dAct_wmma, 0, (size_t)M*I*2));

    cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));

    // 1-CTA path
    launch_interleave_swiglu<1>(dA, dWgu, dAct1, dRoute, M, I, K, s);
    CHECK_CUDA(cudaStreamSynchronize(s));
    // 2-CTA path
    launch_interleave_swiglu<2>(dA, dWgu, dAct2, dRoute, M, I, K, s);
    CHECK_CUDA(cudaStreamSynchronize(s));

    // WMMA reference
    int wmma_smem = (CFG_PHYSICAL_THREADS/32) * (2 * WMMA_M * WMMA_N) * sizeof(float);
    CHECK_CUDA(cudaFuncSetAttribute((const void*)swiglu_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_smem));
    swiglu_wmma_kernel<<<1, CFG_PHYSICAL_THREADS, wmma_smem, s>>>(dA, dWg, dWu, dAct_wmma, dRoute, M, M, K, I);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(s));

    std::vector<__nv_bfloat16> h1(M*I), h2(M*I), hW(M*I);
    CHECK_CUDA(cudaMemcpy(h1.data(), dAct1, (size_t)M*I*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h2.data(), dAct2, (size_t)M*I*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hW.data(), dAct_wmma, (size_t)M*I*2, cudaMemcpyDeviceToHost));
    std::vector<float> f1(M*I), f2(M*I), fW(M*I);
    for (int i = 0; i < M*I; ++i) { f1[i]=__bfloat162float(h1[i]); f2[i]=__bfloat162float(h2[i]); fW[i]=__bfloat162float(hW[i]); }

    float rel1_ref = relative_error(f1, ref_act);
    float rel2_ref = relative_error(f2, ref_act);
    float relW_ref = relative_error(fW, ref_act);
    float rel1_wm  = relative_error(f1, fW);
    float rel2_wm  = relative_error(f2, fW);
    fprintf(stderr, "Rel err 1-CTA interleave-fused vs FP32 ref : %.8g\n", rel1_ref);
    fprintf(stderr, "Rel err 2-CTA interleave-fused vs FP32 ref : %.8g\n", rel2_ref);
    fprintf(stderr, "Rel err WMMA                   vs FP32 ref : %.8g\n", relW_ref);
    fprintf(stderr, "Rel err 1-CTA vs WMMA : %.8g\n", rel1_wm);
    fprintf(stderr, "Rel err 2-CTA vs WMMA : %.8g\n", rel2_wm);

    int status = (rel1_ref < 5e-2f && rel2_ref < 5e-2f) ? 0 : 1;
    fprintf(stderr, "Correctness: %s\n", status == 0 ? "PASS" : "FAIL");

    // ---- Forward PreAct save-cost benchmark: nosave vs int4-save (slot-major) ----
    // Establishes the CEILING of any PreAct-save optimization (TMA included): the
    // gap (int4-save - nosave) is the epilogue-embedded save cost. If it is tiny,
    // the epilogue already hides the int4 store and TMA save cannot help.
    {
        int warmup = 100, iters = 500;
        if (argc >= 5) warmup = atoi(argv[4]);
        if (argc >= 6) iters = atoi(argv[5]);
        __nv_bfloat16* dPreact;
        int *dRecvIdx, *dTopkIdx;
        CHECK_CUDA(cudaMalloc(&dPreact, (size_t)M * N2 * 2));
        CHECK_CUDA(cudaMalloc(&dRecvIdx, (size_t)M * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&dTopkIdx, (size_t)M * sizeof(int)));
        std::vector<int> hrecv(M), htopk(M, 0);
        for (int m = 0; m < M; ++m) hrecv[m] = m;   // slot-major identity mapping
        CHECK_CUDA(cudaMemcpy(dRecvIdx, hrecv.data(), (size_t)M*sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dTopkIdx, htopk.data(), (size_t)M*sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dPreact, 0, (size_t)M * N2 * 2));

        cudaEvent_t e0, e1; CHECK_CUDA(cudaEventCreate(&e0)); CHECK_CUDA(cudaEventCreate(&e1));
        auto dt = [&](auto&& fn){ CHECK_CUDA(cudaEventRecord(e0,s)); fn();
            CHECK_CUDA(cudaEventRecord(e1,s)); CHECK_CUDA(cudaEventSynchronize(e1));
            float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,e0,e1)); return ms; };
        auto run_nosave = [&](){ launch_interleave_swiglu<1>(dA, dWgu, dAct1, dRoute, M, I, K, s); };
        auto run_int4   = [&](){ launch_interleave_swiglu<1>(dA, dWgu, dAct1, dRoute, M, I, K, s,
                                                             dPreact, dRecvIdx, dTopkIdx, 1, N2); };
        for (int w=0; w<warmup; ++w) { run_nosave(); run_int4(); }
        CHECK_CUDA(cudaStreamSynchronize(s));
        double tn=0, ti=0;
        for (int it=0; it<iters; ++it) tn += dt(run_nosave);
        for (int it=0; it<iters; ++it) ti += dt(run_int4);
        tn/=iters; ti/=iters;
        fprintf(stderr, "\n[fwd save-cost 1-CTA] M=%d K=%d I=%d warmup=%d iters=%d\n", M, K, I, warmup, iters);
        fprintf(stderr, "  gate/up GEMM + SwiGLU, no PreAct save : %.4f ms\n", tn);
        fprintf(stderr, "  + int4 PreAct save (slot-major)       : %.4f ms\n", ti);
        fprintf(stderr, "  save overhead (int4)                  : %.4f ms (%.1f%%)\n",
                ti-tn, tn > 0 ? 100.0*(ti-tn)/tn : 0.0);
        CHECK_CUDA(cudaFree(dPreact)); CHECK_CUDA(cudaFree(dRecvIdx)); CHECK_CUDA(cudaFree(dTopkIdx));
        CHECK_CUDA(cudaEventDestroy(e0)); CHECK_CUDA(cudaEventDestroy(e1));
    }

    syscall(SYS_exit_group, status);
}
