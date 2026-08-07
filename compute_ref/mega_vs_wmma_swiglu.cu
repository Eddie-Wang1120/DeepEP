// mega_vs_wmma_swiglu.cu — precision cross-check of the megakernel's DeepGEMM
// gate+up+SwiGLU path (umma_up_swiglu_tile, 2-CTA, exactly as compute_worker
// calls it) against the megakernel's WMMA reference (device_gemm_swiglu_fused).
//
//   act[m,n] = silu(bf16(A@Wg^T)) * bf16(A@Wu^T) * route[m]
//
// Both use 800-thread CTA shape; DeepGEMM uses a 2-CTA cluster + per-tile model
// with accum_iter handoff (gate pass -> up+SwiGLU pass), reading gate_buf from
// GMEM in the SwiGLU epilogue. This isolates whether the SwiGLU epilogue +
// per-tile two-pass model is numerically correct vs the trusted WMMA fused path.
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        mega_vs_wmma_swiglu.cu -o mega_vs_wmma_swiglu -lcuda
//   ./mega_vs_wmma_swiglu          # default M=256 K=4096 N=4096

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#define EP_HOST_ASSERT(cond) do { if(!(cond)){ \
    std::cerr<<"EP_HOST_ASSERT failed: "#cond" at "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(1);} } while(0)
#include "../csrc/kernels/megakernel_compute_umma.cuh"

#define CHECK_CUDA(call)                                                         \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                               \
    std::cerr<<"CUDA error "<<cudaGetErrorString(_e)<<" at "<<__FILE__<<":"      \
             <<__LINE__<<std::endl; std::exit(1);} } while(0)
#define CHECK_CU(call)                                                           \
  do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s;                  \
    cuGetErrorString(_e,&s);                                                     \
    std::cerr<<"CU error "<<s<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;       \
    std::exit(1);} } while(0)

using namespace deep_ep::megakernel::umma;
static constexpr int CFG_THREADS = 800;

// ---------------------------------------------------------------------------
// 2-CTA DeepGEMM gate+up+SwiGLU kernel: one 2-CTA cluster runs all tiles via
// umma_up_swiglu_tile (gate pass -> up+SwiGLU pass), exactly compute_worker.
//   desc_a   : input A [M,K] K-major (cuh dg_make_a_desc)
//   desc_wg  : W_gate [N,K] K-major   desc_wu : W_up [N,K] K-major
//   desc_gate_cd : gate_buf [M,N] row-major CD   desc_act_cd : act_buf [M,N] CD
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(CFG_THREADS, 1)
swiglu_dg_kernel(__nv_bfloat16* gate_buf, const float* route_w,
                 const __grid_constant__ CUtensorMap desc_a,
                 const __grid_constant__ CUtensorMap desc_wg,
                 const __grid_constant__ CUtensorMap desc_wu,
                 const __grid_constant__ CUtensorMap desc_gate_cd,
                 const __grid_constant__ CUtensorMap desc_act_cd,
                 int M, int N, int K) {
    extern __shared__ __align__(1024) char cluster_smem[];
    const int n_tiles = (N + kDgBlockN - 1) / kDgBlockN;
    const int m_tiles = (M + kDgBlockM - 1) / kDgBlockM;
    const int total_tiles = m_tiles * n_tiles;

    bool tmem_allocated = false;
    uint32_t accum_iter = 0;

    for (int tile = 0; tile < total_tiles; ++tile) {
        umma_up_swiglu_tile(
            &desc_a, &desc_gate_cd, &desc_act_cd, &desc_wg, &desc_wu,
            tile, gate_buf, route_w,
            M, N, K, cluster_smem, tmem_allocated, accum_iter);
    }
    if (tmem_allocated)
        umma_dealloc(cluster_smem);
}

// ---------------------------------------------------------------------------
// WMMA reference: device_gemm_swiglu_fused from the retired standalone forward kernel,
// run by one CTA's warps. A:[M,K] row-major, Wg/Wu:[N,K] col_major access.
// ---------------------------------------------------------------------------
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
    for (size_t i = 0; i < r.size(); ++i) { double d = t[i] - r[i]; num += d*d; den += double(r[i])*r[i]; }
    return float(std::sqrt(num) / (std::sqrt(den) + 1e-12));
}

int main(int argc, char** argv) {
    int M = 256, K = 4096, N = 4096;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    fprintf(stderr, "[mega-vs-wmma SwiGLU] M=%d K=%d N=%d (2-CTA DeepGEMM)\n", M, K, N);
    CHECK_CU(cuInit(0));

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto bf = [](float v){ return __bfloat162float(__float2bfloat16(v)); };

    std::vector<float> hA(M*K), hWg(N*K), hWu(N*K), hRoute(M);
    for (auto& x : hA) x = dist(gen);
    for (auto& x : hWg) x = dist(gen);
    for (auto& x : hWu) x = dist(gen);
    for (auto& x : hRoute) x = dist(gen);

    // FP32 (bf16-rounded inputs) reference: silu(bf16(gate))*bf16(up)*route.
    std::vector<float> Ab(M*K), Wgb(N*K), Wub(N*K);
    for (int i = 0; i < M*K; ++i) Ab[i] = bf(hA[i]);
    for (int i = 0; i < N*K; ++i) { Wgb[i] = bf(hWg[i]); Wub[i] = bf(hWu[i]); }
    std::vector<float> ref_act(M*N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double sg = 0, su = 0;
            for (int k = 0; k < K; ++k) { sg += double(Ab[m*K+k])*Wgb[n*K+k]; su += double(Ab[m*K+k])*Wub[n*K+k]; }
            float g = bf(float(sg)), u = bf(float(su));
            ref_act[m*N+n] = bf((g*(1.0f/(1.0f+std::exp(-g))))*u*hRoute[m]);
        }
    fprintf(stderr, "[host] SwiGLU ref done\n");

    std::vector<__nv_bfloat16> A16(M*K), Wg16(N*K), Wu16(N*K);
    for (int i = 0; i < M*K; ++i) A16[i] = __float2bfloat16(hA[i]);
    for (int i = 0; i < N*K; ++i) { Wg16[i] = __float2bfloat16(hWg[i]); Wu16[i] = __float2bfloat16(hWu[i]); }

    __nv_bfloat16 *dA, *dWg, *dWu, *dGate, *dAct_dg, *dAct_wmma;
    float* dRoute;
    CHECK_CUDA(cudaMalloc(&dA, (size_t)M*K*2));
    CHECK_CUDA(cudaMalloc(&dWg, (size_t)N*K*2));
    CHECK_CUDA(cudaMalloc(&dWu, (size_t)N*K*2));
    CHECK_CUDA(cudaMalloc(&dGate, (size_t)M*N*2));
    CHECK_CUDA(cudaMalloc(&dAct_dg, (size_t)M*N*2));
    CHECK_CUDA(cudaMalloc(&dAct_wmma, (size_t)M*N*2));
    CHECK_CUDA(cudaMalloc(&dRoute, (size_t)M*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A16.data(), (size_t)M*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWg, Wg16.data(), (size_t)N*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWu, Wu16.data(), (size_t)N*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRoute, hRoute.data(), (size_t)M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dGate, 0, (size_t)M*N*2));
    CHECK_CUDA(cudaMemset(dAct_dg, 0, (size_t)M*N*2));
    CHECK_CUDA(cudaMemset(dAct_wmma, 0, (size_t)M*N*2));

    cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));

    // ---- DeepGEMM 2-CTA gate+up+SwiGLU ----
    CUtensorMap desc_a       = dg_make_a_desc(dA, M, K);
    CUtensorMap desc_wg      = dg_make_b_desc(dWg, N, K);
    CUtensorMap desc_wu      = dg_make_b_desc(dWu, N, K);
    CUtensorMap desc_gate_cd = dg_make_cd_desc(dGate, M, N);
    CUtensorMap desc_act_cd  = dg_make_cd_desc(dAct_dg, M, N);

    int smem_bytes = 227 * 1024;
    CHECK_CUDA(cudaFuncSetAttribute((const void*)swiglu_dg_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(CFG_THREADS, 1, 1);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = s;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs; cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, swiglu_dg_kernel,
        dGate, dRoute, desc_a, desc_wg, desc_wu, desc_gate_cd, desc_act_cd, M, N, K));
    CHECK_CUDA(cudaStreamSynchronize(s));

    // ---- WMMA fused reference ----
    int wmma_smem = (CFG_THREADS/32) * (2 * WMMA_M * WMMA_N) * sizeof(float);
    CHECK_CUDA(cudaFuncSetAttribute((const void*)swiglu_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_smem));
    swiglu_wmma_kernel<<<1, CFG_THREADS, wmma_smem, s>>>(
        dA, dWg, dWu, dAct_wmma, dRoute, /*valid_rows=*/M, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(s));

    // ---- compare ----
    std::vector<__nv_bfloat16> hDg(M*N), hWm(M*N);
    CHECK_CUDA(cudaMemcpy(hDg.data(), dAct_dg, (size_t)M*N*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hWm.data(), dAct_wmma, (size_t)M*N*2, cudaMemcpyDeviceToHost));
    std::vector<float> fDg(M*N), fWm(M*N);
    for (int i = 0; i < M*N; ++i) { fDg[i] = __bfloat162float(hDg[i]); fWm[i] = __bfloat162float(hWm[i]); }

    float rel_dg_ref  = relative_error(fDg, ref_act);
    float rel_wm_ref  = relative_error(fWm, ref_act);
    float rel_dg_wm   = relative_error(fDg, fWm);
    fprintf(stderr, "Rel err DeepGEMM-SwiGLU vs FP32 ref : %.8g\n", rel_dg_ref);
    fprintf(stderr, "Rel err WMMA-SwiGLU    vs FP32 ref : %.8g\n", rel_wm_ref);
    fprintf(stderr, "Rel err DeepGEMM       vs WMMA     : %.8g\n", rel_dg_wm);

    int printed = 0;
    for (int i = 0; i < M*N && printed < 8; ++i) {
        if (std::fabs(fDg[i] - fWm[i]) > 0.5f * (std::fabs(fWm[i]) + 1e-3f)) {
            int m = i / N, n = i % N;
            fprintf(stderr, "  mismatch [m=%d n=%d] dg=%.5f wmma=%.5f ref=%.5f\n",
                    m, n, fDg[i], fWm[i], ref_act[i]);
            ++printed;
        }
    }
    if (printed == 0) fprintf(stderr, "  (no large elementwise dg/wmma mismatches)\n");

    int status = (rel_dg_ref < 5e-2f && rel_dg_wm < 5e-2f) ? 0 : 1;
    fprintf(stderr, "Correctness: %s\n", status == 0 ? "PASS" : "FAIL");
    return status;
}
