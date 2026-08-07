// mega_vs_wmma_gate.cu — precision cross-check of the megakernel's DeepGEMM
// compute path (per-tile dg_gemm_tile, exactly as compute_worker calls it)
// against the megakernel's WMMA reference (device_gemm_bf16), for a single
// pure GATE GEMM: C[m,n] = sum_k A[m,k] * Wg[n,k]  (B = A @ Wg^T).
//
// STEP B: 1-CTA path (kNumMulticast=1) first — isolates whether the migrated
// GEMM ALGORITHM is numerically correct, avoiding the 2-CTA tcgen05 cluster
// alloc handshake. Single CTA, 800 threads (compute_worker shape), no cluster.
//
// Build (B30Z cc10.3, CUDA 13.2):
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        mega_vs_wmma_gate.cu -o mega_vs_wmma_gate -lcuda
//   ./mega_vs_wmma_gate            # default M=256 K=4096 N=4096

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

// Pull in the megakernel's DeepGEMM device path verbatim. The header lives in
// csrc/kernels and uses deep_ep::megakernel::umma::dg_gemm_tile + the raw
// CUtensorMap builders. It includes sm100_bf16_gemm_dg_copy.cuh internally.
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

static constexpr int CFG_THREADS = 800;   // match compute_worker CTA shape

// ---------------------------------------------------------------------------
// 1-CTA TMA descriptor builders. The cuh's dg_make_*_desc are sized for the
// 2-CTA path (kDgLoadBlockN = BLOCK_N/2 = 64). For the 1-CTA numeric check we
// need LOAD_BLOCK_M = BLOCK_M = 128 and LOAD_BLOCK_N = BLOCK_N = 128, so build
// local 1-CTA descriptors here (mirror umma_swiglu_ws_dg.cu make_*_desc).
// ---------------------------------------------------------------------------
static constexpr uint32_t C1_BLOCK_K = 64, C1_SWZ = 128;
static constexpr uint32_t C1_LOAD_BLOCK_M = 128;   // 1-CTA: full BLOCK_M
static constexpr uint32_t C1_LOAD_BLOCK_N = 128;   // 1-CTA: full BLOCK_N
static constexpr uint32_t C1_STORE_BLOCK_M = 128;
static constexpr uint32_t C1_STORE_BLOCK_N = C1_SWZ / sizeof(cutlass::bfloat16_t); // 64

static CUtensorMap c1_tma_2d(const void* ptr, int gmem_inner, int gmem_outer,
                             int smem_inner, int smem_outer,
                             int gmem_outer_stride_elems, int swizzle_bytes) {
    CUtensorMap tm;
    int si = (swizzle_bytes != 0) ? swizzle_bytes / (int)sizeof(__nv_bfloat16) : smem_inner;
    const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    const cuuint32_t sdims[2] = {(cuuint32_t)si, (cuuint32_t)smem_outer};
    const cuuint64_t gstr[1]  = {(cuuint64_t)gmem_outer_stride_elems * sizeof(__nv_bfloat16)};
    const cuuint32_t estr[2]  = {1, 1};
    CUtensorMapSwizzle sw = swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
    CHECK_CU(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr,
        gdims, gstr, sdims, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tm;
}
static CUtensorMap c1_a_desc(const __nv_bfloat16* a, int M, int K) {
    return c1_tma_2d(a, K, M, C1_BLOCK_K, C1_LOAD_BLOCK_M, K, C1_SWZ);
}
static CUtensorMap c1_b_desc(const __nv_bfloat16* b, int N, int K) {
    return c1_tma_2d(b, K, N, C1_BLOCK_K, C1_LOAD_BLOCK_N, K, C1_SWZ);
}
static CUtensorMap c1_cd_desc(const __nv_bfloat16* d, int M, int N) {
    return c1_tma_2d(d, N, M, C1_STORE_BLOCK_N, C1_STORE_BLOCK_M, N, C1_SWZ);
}

// ---------------------------------------------------------------------------
// DeepGEMM-path kernel: ONE CTA (1-CTA, kNumMulticast=1) runs all gate tiles
// via per-tile dg_gemm_tile<false, 1> calls — same per-tile call model as
// compute_worker, but the 1-CTA numeric path (no cluster alloc handshake).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(CFG_THREADS, 1)
gate_dg_kernel(const __nv_bfloat16* A, const __nv_bfloat16* Wg, __nv_bfloat16* C,
               const __grid_constant__ CUtensorMap desc_a,
               const __grid_constant__ CUtensorMap desc_b,
               const __grid_constant__ CUtensorMap desc_cd,
               int M, int N, int K) {
    extern __shared__ __align__(1024) char cluster_smem[];
    const int m_tiles = (M + kDgBlockM - 1) / kDgBlockM;
    const int n_tiles = (N + kDgBlockN - 1) / kDgBlockN;
    const int total_tiles = m_tiles * n_tiles;

    bool tmem_allocated = false;
    uint32_t accum_iter = 0;

    for (int tile = 0; tile < total_tiles; ++tile) {
        const int m_block = tile / n_tiles;
        const int n_block = tile - m_block * n_tiles;
        dg_gemm_tile<false, 1>(
            &desc_a, &desc_b, &desc_cd,
            m_block, n_block, M, N, K,
            cluster_smem, tmem_allocated, accum_iter,
            nullptr, nullptr, 0);
    }
    // Free TMEM before kernel exit (1-CTA Allocator1Sm). The cuh's umma_dealloc
    // assumes the 2-CTA layout (kDgRunMulticast), so do a 1-CTA free inline here:
    // recompute the tmem_ptr_in_smem slot with 1-CTA LOAD_BLOCK_N=128.
    if (tmem_allocated) {
        constexpr uint32_t SMEM_CD = 128 * 64 * 2 * 2;       // STORE_BLOCK_M*N*bf16*stages
        constexpr uint32_t SMEM_A_PER = 128 * 64 * 2;        // LOAD_BLOCK_M(128)*BLOCK_K*bf16
        constexpr uint32_t SMEM_B_PER = 128 * 64 * 2;        // 1-CTA LOAD_BLOCK_N(128)*BLOCK_K*bf16
        uint8_t* sb = reinterpret_cast<uint8_t*>(cluster_smem);
        auto bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(
            sb + SMEM_CD + 4 * (SMEM_A_PER + SMEM_B_PER));
        auto tmem_ptr = reinterpret_cast<uint32_t*>(bar + 4 * 3 + 2 * 2 + 1);
        __syncthreads();
        if (cutlass::canonical_warp_idx_sync() == 0)
            cute::TMEM::Allocator1Sm().free(*tmem_ptr, 256);
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// 2-CTA DeepGEMM-path kernel: a 2-CTA cluster runs all gate tiles via per-tile
// dg_gemm_tile<false, 2> calls — EXACTLY compute_worker's 2-CTA gate model.
// Uses the cuh's dg_make_*_desc (LOAD_BLOCK_N=64). Both CTAs of the cluster
// iterate the same tile sequence (single cluster, no group striding here).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(CFG_THREADS, 1)
gate_dg_kernel_2cta(const __nv_bfloat16* A, const __nv_bfloat16* Wg, __nv_bfloat16* C,
                    const __grid_constant__ CUtensorMap desc_a,
                    const __grid_constant__ CUtensorMap desc_b,
                    const __grid_constant__ CUtensorMap desc_cd,
                    int M, int N, int K) {
    extern __shared__ __align__(1024) char cluster_smem[];
    const int m_tiles = (M + kDgBlockM - 1) / kDgBlockM;
    const int n_tiles = (N + kDgBlockN - 1) / kDgBlockN;
    const int total_tiles = m_tiles * n_tiles;

    bool tmem_allocated = false;
    uint32_t accum_iter = 0;

    for (int tile = 0; tile < total_tiles; ++tile) {
        const int m_block = tile / n_tiles;
        const int n_block = tile - m_block * n_tiles;
        dg_gemm_tile<false, 2>(
            &desc_a, &desc_b, &desc_cd,
            m_block, n_block, M, N, K,
            cluster_smem, tmem_allocated, accum_iter,
            nullptr, nullptr, 0);
    }
    // 2-CTA TMEM free (Allocator2Sm) — matches the cuh umma_dealloc layout.
    if (tmem_allocated)
        umma_dealloc(cluster_smem);
}

// ---------------------------------------------------------------------------
// WMMA reference from the retired standalone forward megakernel, run by a
// single CTA's warps (group cooperative form, but one CTA). Pure numeric ref.
// A:[M,K] row-major, Wg:[N,K] row-major (col_major access => B=A@B^T), C:[M,N].
// ---------------------------------------------------------------------------
using namespace nvcuda;
static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void gate_wmma_kernel(const __nv_bfloat16* __restrict__ A,
                                 const __nv_bfloat16* __restrict__ B,
                                 __nv_bfloat16* __restrict__ C,
                                 int M, int K, int N) {
    extern __shared__ float smem_buf[];
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;
    const int tiles_m = (M + WMMA_M - 1) / WMMA_M;
    const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
    const int total_tiles = tiles_m * tiles_n;

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
        int tile_row = tile_idx / tiles_n;
        int tile_col = tile_idx % tiles_n;
        int row_offset = tile_row * WMMA_M;
        int col_offset = tile_col * WMMA_N;
        if (row_offset >= M || col_offset >= N) continue;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A + row_offset * K + k, K);
            wmma::load_matrix_sync(b_frag, B + col_offset * K + k, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* c_buf = smem_buf + warp_id * WMMA_M * WMMA_N;
        wmma::store_matrix_sync(c_buf, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        int lane_id = threadIdx.x % 32;
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N, col = i % WMMA_N;
            int out_row = row_offset + row, out_col = col_offset + col;
            if (out_row < M && out_col < N)
                C[out_row * N + out_col] = __float2bfloat16(c_buf[i]);
        }
    }
}

// ---------------------------------------------------------------------------
static float relative_error(const std::vector<float>& t, const std::vector<float>& r) {
    double num = 0, den = 0;
    for (size_t i = 0; i < r.size(); ++i) { double d = t[i] - r[i]; num += d*d; den += double(r[i])*r[i]; }
    return float(std::sqrt(num) / (std::sqrt(den) + 1e-12));
}

int main(int argc, char** argv) {
    int M = 256, K = 4096, N = 4096;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    fprintf(stderr, "[mega-vs-wmma gate GEMM] M=%d K=%d N=%d\n", M, K, N);
    CHECK_CU(cuInit(0));

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto bf = [](float v){ return __bfloat162float(__float2bfloat16(v)); };

    std::vector<float> hA(M*K), hWg(N*K);
    for (auto& x : hA) x = dist(gen);
    for (auto& x : hWg) x = dist(gen);

    // FP32 (bf16-rounded inputs) reference.
    std::vector<float> Ab(M*K), Wgb(N*K);
    for (int i = 0; i < M*K; ++i) Ab[i] = bf(hA[i]);
    for (int i = 0; i < N*K; ++i) Wgb[i] = bf(hWg[i]);
    std::vector<float> ref_gate(M*N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += double(Ab[m*K+k]) * Wgb[n*K+k];
            ref_gate[m*N+n] = float(s);
        }
    fprintf(stderr, "[host] FP32 gate ref done\n");

    std::vector<__nv_bfloat16> A16(M*K), Wg16(N*K);
    for (int i = 0; i < M*K; ++i) A16[i] = __float2bfloat16(hA[i]);
    for (int i = 0; i < N*K; ++i) Wg16[i] = __float2bfloat16(hWg[i]);

    __nv_bfloat16 *dA, *dWg, *dC_dg, *dC_wmma;
    CHECK_CUDA(cudaMalloc(&dA, (size_t)M*K*2));
    CHECK_CUDA(cudaMalloc(&dWg, (size_t)N*K*2));
    CHECK_CUDA(cudaMalloc(&dC_dg, (size_t)M*N*2));
    CHECK_CUDA(cudaMalloc(&dC_wmma, (size_t)M*N*2));
    CHECK_CUDA(cudaMemcpy(dA, A16.data(), (size_t)M*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dWg, Wg16.data(), (size_t)N*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_dg, 0, (size_t)M*N*2));
    CHECK_CUDA(cudaMemset(dC_wmma, 0, (size_t)M*N*2));

    cudaStream_t s; CHECK_CUDA(cudaStreamCreate(&s));

#ifdef USE_2CTA
    // ===== 2-CTA path: cluster_dim=2, dg_gemm_tile<false,2>, cuh descriptors =====
    CUtensorMap desc_a  = dg_make_a_desc(dA, M, K);
    CUtensorMap desc_b  = dg_make_b_desc(dWg, N, K);
    CUtensorMap desc_cd = dg_make_cd_desc(dC_dg, M, N);

    int smem_bytes = 227 * 1024;
    CHECK_CUDA(cudaFuncSetAttribute((const void*)gate_dg_kernel_2cta,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2, 1, 1);              // one 2-CTA cluster
    cfg.blockDim = dim3(CFG_THREADS, 1, 1);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = s;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs; cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, gate_dg_kernel_2cta,
        dA, dWg, dC_dg, desc_a, desc_b, desc_cd, M, N, K));
    CHECK_CUDA(cudaStreamSynchronize(s));
#else
    // ---- DeepGEMM path: build 1-CTA raw CUtensorMaps ----
    CUtensorMap desc_a  = c1_a_desc(dA, M, K);
    CUtensorMap desc_b  = c1_b_desc(dWg, N, K);
    CUtensorMap desc_cd = c1_cd_desc(dC_dg, M, N);

    int smem_bytes = 227 * 1024;
    CHECK_CUDA(cudaFuncSetAttribute((const void*)gate_dg_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // 1-CTA launch: single block, NO cluster attribute (kNumMulticast=1 path).
    gate_dg_kernel<<<1, CFG_THREADS, smem_bytes, s>>>(
        dA, dWg, dC_dg, desc_a, desc_b, desc_cd, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(s));
#endif

    // ---- WMMA reference path ----
    int wmma_smem = (CFG_THREADS/32) * WMMA_M * WMMA_N * sizeof(float);
    CHECK_CUDA(cudaFuncSetAttribute((const void*)gate_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_smem));
    gate_wmma_kernel<<<1, CFG_THREADS, wmma_smem, s>>>(dA, dWg, dC_wmma, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(s));

    // ---- compare ----
    std::vector<__nv_bfloat16> hC_dg_b(M*N), hC_wmma_b(M*N);
    CHECK_CUDA(cudaMemcpy(hC_dg_b.data(), dC_dg, (size_t)M*N*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_wmma_b.data(), dC_wmma, (size_t)M*N*2, cudaMemcpyDeviceToHost));
    std::vector<float> hC_dg(M*N), hC_wmma(M*N);
    for (int i = 0; i < M*N; ++i) { hC_dg[i] = __bfloat162float(hC_dg_b[i]); hC_wmma[i] = __bfloat162float(hC_wmma_b[i]); }

    float rel_dg_ref   = relative_error(hC_dg, ref_gate);
    float rel_wmma_ref = relative_error(hC_wmma, ref_gate);
    float rel_dg_wmma  = relative_error(hC_dg, hC_wmma);

    fprintf(stderr, "Rel err DeepGEMM vs FP32 ref : %.8g\n", rel_dg_ref);
    fprintf(stderr, "Rel err WMMA    vs FP32 ref : %.8g\n", rel_wmma_ref);
    fprintf(stderr, "Rel err DeepGEMM vs WMMA    : %.8g\n", rel_dg_wmma);

    // Print a few sample mismatches to eyeball the failure mode.
    int printed = 0;
    for (int i = 0; i < M*N && printed < 8; ++i) {
        if (std::fabs(hC_dg[i] - hC_wmma[i]) > 0.5f * (std::fabs(hC_wmma[i]) + 1e-3f)) {
            int m = i / N, n = i % N;
            fprintf(stderr, "  mismatch [m=%d n=%d] dg=%.5f wmma=%.5f ref=%.5f\n",
                    m, n, hC_dg[i], hC_wmma[i], ref_gate[i]);
            ++printed;
        }
    }
    if (printed == 0) fprintf(stderr, "  (no large elementwise dg/wmma mismatches)\n");

    int status = (rel_dg_ref < 5e-2f && rel_dg_wmma < 5e-2f) ? 0 : 1;
    fprintf(stderr, "Correctness: %s\n", status == 0 ? "PASS" : "FAIL");
    return status;
}
