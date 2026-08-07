// mega_vs_wmma_gateup_interleave.cu — validate block-interleave-32 gate/up weight
// layout + single GEMM + block-paired SwiGLU (step 1 of the "fuse SwiGLU into
// gateup epilogue" plan; here SwiGLU is still a separate elementwise pass, but
// the weight layout and gate/up pairing match TE's fc1_weight / ScaledSwiGLU
// (glu_interleave_size=32) used by the Megatron/TE baseline helpers).
//
// Weight layout (N = 2I, block-interleave 32):
//   Wgu = [ g[0:32], u[0:32], g[32:64], u[32:64], ..., g[I-32:I], u[I-32:I] ]
// GEMM: GU = A @ Wgu^T -> [M, 2I] in that column order.
// SwiGLU (logical i in [0,I)):
//   blk = i/32, off = i%32
//   gate_col(i) = blk*64 + off
//   up_col(i)   = blk*64 + 32 + off
//   act[m,i] = silu(GU[m,gate_col]) * GU[m,up_col] * route[m]
//
// Compared against the trusted WMMA fused reference (device_gemm_swiglu_fused
// semantics: separate Wg/Wu, in-register SwiGLU).
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        mega_vs_wmma_gateup_interleave.cu -o mega_vs_wmma_gateup_interleave -lcuda
//   ./mega_vs_wmma_gateup_interleave      # default M=256 K=4096 I=4096

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
static constexpr int GLU_BLK = 32;   // glu_interleave_size, matches TE ScaledSwiGLU

// Single concat-weight GEMM (N=2I) via dg_gemm_tile, NO SwiGLU. Same as the
// concat microkernel; the block-interleave is purely in how the host lays out
// Wgu and how the SwiGLU kernel pairs columns.
__global__ void __launch_bounds__(CFG_THREADS, 1)
gateup_kernel(const __grid_constant__ CUtensorMap desc_a,
              const __grid_constant__ CUtensorMap desc_wgu,
              const __grid_constant__ CUtensorMap desc_gu_cd,
              int M, int N2, int K) {
    extern __shared__ __align__(1024) char cluster_smem[];
    const int n_tiles = (N2 + kDgBlockN - 1) / kDgBlockN;
    const int m_tiles = (M + kDgBlockM - 1) / kDgBlockM;
    const int total_tiles = m_tiles * n_tiles;
    bool tmem_allocated = false;
    uint32_t accum_iter = 0;
    for (int tile = 0; tile < total_tiles; ++tile) {
        int m_block = tile / n_tiles;
        int n_block = tile - m_block * n_tiles;
        dg_gemm_tile<false, kDgRunMulticast>(
            &desc_a, &desc_wgu, &desc_gu_cd,
            m_block, n_block, M, N2, K,
            cluster_smem, tmem_allocated, accum_iter,
            nullptr, nullptr, 0);
    }
    if (tmem_allocated)
        umma_dealloc(cluster_smem);
}

// Block-interleave-32 SwiGLU: act[m,i] = silu(GU[m,gate_col]) * GU[m,up_col] * route.
__global__ void swiglu_blkinterleave_kernel(const __nv_bfloat16* __restrict__ GU,
                                            __nv_bfloat16* __restrict__ act,
                                            const float* __restrict__ route_w,
                                            int M, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * I) return;
    int m = idx / I;
    int i = idx - m * I;
    int blk = i / GLU_BLK;
    int off = i - blk * GLU_BLK;
    int gate_col = blk * (2 * GLU_BLK) + off;
    int up_col   = gate_col + GLU_BLK;
    float g = __bfloat162float(GU[(int64_t)m * (2 * I) + gate_col]);
    float u = __bfloat162float(GU[(int64_t)m * (2 * I) + up_col]);
    float silu = g * (1.0f / (1.0f + __expf(-g)));
    act[(int64_t)m * I + i] = __float2bfloat16(silu * u * route_w[m]);
}

// WMMA fused reference (device_gemm_swiglu_fused semantics).
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
    EP_HOST_ASSERT(I % GLU_BLK == 0);
    const int N2 = 2 * I;
    fprintf(stderr, "[mega interleave gate/up] M=%d K=%d I=%d (block-interleave-%d, single GEMM + block-paired SwiGLU)\n",
            M, K, I, GLU_BLK);
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
    fprintf(stderr, "[host] SwiGLU ref done\n");

    // Block-interleave-32 concat weight Wgu [2I, K]:
    //   rows [blk*64 .. blk*64+32) = Wg[blk*32 .. blk*32+32)
    //   rows [blk*64+32 .. blk*64+64) = Wu[blk*32 .. blk*32+32)
    std::vector<__nv_bfloat16> A16(M*K), Wg16(I*K), Wu16(I*K), Wgu16((size_t)N2*K);
    for (int i = 0; i < M*K; ++i) A16[i] = __float2bfloat16(hA[i]);
    for (int i = 0; i < I*K; ++i) { Wg16[i] = __float2bfloat16(hWg[i]); Wu16[i] = __float2bfloat16(hWu[i]); }
    const int nblk = I / GLU_BLK;
    for (int blk = 0; blk < nblk; ++blk) {
        for (int r = 0; r < GLU_BLK; ++r) {
            int src_row = blk * GLU_BLK + r;
            int gate_dst = (blk * 2 * GLU_BLK + r);
            int up_dst   = (blk * 2 * GLU_BLK + GLU_BLK + r);
            for (int k = 0; k < K; ++k) {
                Wgu16[(size_t)gate_dst * K + k] = Wg16[(size_t)src_row * K + k];
                Wgu16[(size_t)up_dst   * K + k] = Wu16[(size_t)src_row * K + k];
            }
        }
    }

    __nv_bfloat16 *dA, *dWgu, *dWg, *dWu, *dGU, *dAct_dg, *dAct_wmma;
    float* dRoute;
    CHECK_CUDA(cudaMalloc(&dA, (size_t)M*K*2));
    CHECK_CUDA(cudaMalloc(&dWgu, (size_t)N2*K*2));
    CHECK_CUDA(cudaMalloc(&dWg, (size_t)I*K*2));
    CHECK_CUDA(cudaMalloc(&dWu, (size_t)I*K*2));
    CHECK_CUDA(cudaMalloc(&dGU, (size_t)M*N2*2));
    CHECK_CUDA(cudaMalloc(&dAct_dg, (size_t)M*I*2));
    CHECK_CUDA(cudaMalloc(&dAct_wmma, (size_t)M*I*2));
    CHECK_CUDA(cudaMalloc(&dRoute, (size_t)M*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A16.data(), (size_t)M*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWgu, Wgu16.data(), (size_t)N2*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWg, Wg16.data(), (size_t)I*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWu, Wu16.data(), (size_t)I*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRoute, hRoute.data(), (size_t)M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dGU, 0, (size_t)M*N2*2));
    CHECK_CUDA(cudaMemset(dAct_dg, 0, (size_t)M*I*2));
    CHECK_CUDA(cudaMemset(dAct_wmma, 0, (size_t)M*I*2));

    cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));

    CUtensorMap desc_a     = dg_make_a_desc(dA, M, K);
    CUtensorMap desc_wgu   = dg_make_b_desc(dWgu, N2, K);
    CUtensorMap desc_gu_cd = dg_make_cd_desc(dGU, M, N2);

    int smem_bytes = 227 * 1024;
    CHECK_CUDA(cudaFuncSetAttribute((const void*)gateup_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(CFG_THREADS, 1, 1);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = s;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs; cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, gateup_kernel, desc_a, desc_wgu, desc_gu_cd, M, N2, K));
    CHECK_CUDA(cudaStreamSynchronize(s));

    {
        int total = M * I, threads = 256, blocks = (total + threads - 1) / threads;
        swiglu_blkinterleave_kernel<<<blocks, threads, 0, s>>>(dGU, dAct_dg, dRoute, M, I);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s));
    }

    int wmma_smem = (CFG_THREADS/32) * (2 * WMMA_M * WMMA_N) * sizeof(float);
    CHECK_CUDA(cudaFuncSetAttribute((const void*)swiglu_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_smem));
    swiglu_wmma_kernel<<<1, CFG_THREADS, wmma_smem, s>>>(dA, dWg, dWu, dAct_wmma, dRoute, M, M, K, I);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(s));

    std::vector<__nv_bfloat16> hDg(M*I), hWm(M*I);
    CHECK_CUDA(cudaMemcpy(hDg.data(), dAct_dg, (size_t)M*I*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hWm.data(), dAct_wmma, (size_t)M*I*2, cudaMemcpyDeviceToHost));
    std::vector<float> fDg(M*I), fWm(M*I);
    for (int i = 0; i < M*I; ++i) { fDg[i] = __bfloat162float(hDg[i]); fWm[i] = __bfloat162float(hWm[i]); }

    float rel_dg_ref = relative_error(fDg, ref_act);
    float rel_wm_ref = relative_error(fWm, ref_act);
    float rel_dg_wm  = relative_error(fDg, fWm);
    fprintf(stderr, "Rel err interleave-GEMM+SwiGLU vs FP32 ref : %.8g\n", rel_dg_ref);
    fprintf(stderr, "Rel err WMMA-SwiGLU            vs FP32 ref : %.8g\n", rel_wm_ref);
    fprintf(stderr, "Rel err interleave            vs WMMA      : %.8g\n", rel_dg_wm);

    int printed = 0;
    for (int i = 0; i < M*I && printed < 8; ++i) {
        if (std::fabs(fDg[i] - fWm[i]) > 0.5f * (std::fabs(fWm[i]) + 1e-3f)) {
            int m = i / I, n = i % I;
            fprintf(stderr, "  mismatch [m=%d n=%d] intlv=%.5f wmma=%.5f ref=%.5f\n", m, n, fDg[i], fWm[i], ref_act[i]);
            ++printed;
        }
    }
    if (printed == 0) fprintf(stderr, "  (no large elementwise interleave/wmma mismatches)\n");

    int status = (rel_dg_ref < 5e-2f && rel_dg_wm < 5e-2f) ? 0 : 1;
    fprintf(stderr, "Correctness: %s\n", status == 0 ? "PASS" : "FAIL");
    return status;
}
