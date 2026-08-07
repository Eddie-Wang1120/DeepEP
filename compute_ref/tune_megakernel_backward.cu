// tune_megakernel_backward.cu
//
// Single-GPU, compute-only tuner for the MegaKernel BACKWARD compute GEMMs:
//   grad_act = grad_down @ W_down^T   (dgrad, MN-major B)   [M, I] = [M,hidden]x[hidden,I]
//   grad_x   = grad_gu   @ W_gateup^T (dgrad, MN-major B)   [M, hidden] = [M,2I]x[2I,hidden]
//
// Mirrors tune_megakernel_compute.cu but for the backward dgrad path
// (umma_dgrad_mn_persistent). The in-between dSwiGLU is NOT a GEMM and is
// excluded — this isolates the two dgrad GEMMs for config tuning.
//
// Build (from compute_ref/, B30Z/SM103a, 1-CTA):
//   nvcc -std=c++17 -O3 -gencode=arch=compute_103a,code=sm_103a -DMK_COMPUTE_KERNEL=1 \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        tune_megakernel_backward.cu -o tune_megakernel_backward -lcuda

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define EP_HOST_ASSERT(cond) do {                                                   \
    if (!(cond)) {                                                                  \
        std::cerr << "EP_HOST_ASSERT failed: " #cond " at " << __FILE__ << ":"    \
                  << __LINE__ << std::endl;                                         \
        std::exit(1);                                                               \
    }                                                                               \
} while (0)

#include "../csrc/kernels/megakernel_compute_umma.cuh"

#define CHECK_CUDA(call) do {                                                       \
    cudaError_t err__ = (call);                                                     \
    if (err__ != cudaSuccess) {                                                     \
        std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at "          \
                  << __FILE__ << ":" << __LINE__ << std::endl;                    \
        std::exit(1);                                                               \
    }                                                                               \
} while (0)

#define CHECK_CU(call) do {                                                         \
    CUresult err__ = (call);                                                        \
    if (err__ != CUDA_SUCCESS) {                                                    \
        const char* msg__ = nullptr;                                                \
        cuGetErrorString(err__, &msg__);                                            \
        std::cerr << "CU error " << (msg__ ? msg__ : "unknown") << " at "         \
                  << __FILE__ << ":" << __LINE__ << std::endl;                    \
        std::exit(1);                                                               \
    }                                                                               \
} while (0)

namespace umma = deep_ep::megakernel::umma;

static constexpr int kThreads = 800;
static constexpr int kClusterDim = umma::kDgRunMulticast;
static constexpr int kSmemBytes = 227 * 1024;

struct GridBarrier { int* counter; int* phase; int expected_blocks; };

__device__ __forceinline__ int volatile_load(const int* ptr) {
    return *reinterpret_cast<const volatile int*>(ptr);
}
__device__ __forceinline__ void grid_barrier(GridBarrier barrier) {
    __syncthreads(); __threadfence();
    if (threadIdx.x == 0) {
        const int observed_phase = volatile_load(barrier.phase);
        const int arrived = atomicAdd(barrier.counter, 1) + 1;
        if (arrived == barrier.expected_blocks) {
            __threadfence(); atomicExch(barrier.counter, 0);
            __threadfence(); atomicExch(barrier.phase, observed_phase + 1);
        }
        while (volatile_load(barrier.phase) == observed_phase) __nanosleep(64);
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kThreads, 1)
megakernel_backward_only_kernel(
    const __grid_constant__ CUtensorMap desc_graddown_a,  // A grad_act: [M, hidden] K-major
    const __grid_constant__ CUtensorMap desc_wdown_mn,    // B grad_act: W_down MN-major [N=I, K=hidden]
    const __grid_constant__ CUtensorMap desc_gradact_cd,  // CD grad_act: [M, I]
    const __grid_constant__ CUtensorMap desc_dgu_a,       // A grad_x: [M, 2I] K-major
    const __grid_constant__ CUtensorMap desc_wgateup_mn,  // B grad_x: W_gateup MN-major [N=hidden, K=2I]
    const __grid_constant__ CUtensorMap desc_gradx_cd,    // CD grad_x: [M, hidden]
    int m, int hidden, int intermediate,
    GridBarrier barrier) {

    extern __shared__ __align__(1024) char cluster_smem[];
    const int cluster_idx = blockIdx.x / kClusterDim;
    const int num_clusters = gridDim.x / kClusterDim;

    // grad_act = grad_down @ W_down^T : M, N=intermediate, K=hidden
    uint32_t iter0 = 0;
    umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
    umma::umma_dgrad_mn_persistent(
        &desc_graddown_a, &desc_wdown_mn, &desc_gradact_cd,
        m, intermediate, hidden,
        cluster_idx, num_clusters, cluster_smem, iter0);
    umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);

    grid_barrier(barrier);

    // grad_x = grad_gu @ W_gateup^T : M, N=hidden, K=2*intermediate
    uint32_t iter1 = 0;
    umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
    umma::umma_dgrad_mn_persistent(
        &desc_dgu_a, &desc_wgateup_mn, &desc_gradx_cd,
        m, hidden, 2 * intermediate,
        cluster_idx, num_clusters, cluster_smem, iter1);
    umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);
}

struct Args {
    std::string m_values = "4096";
    std::string sm_values = "48";
    int k = 2048; int n = 3072;
    int warmup = 10; int iters = 40; int device = 0;
};

static std::vector<int> parse_int_values(const std::string& spec) {
    std::vector<int> values; std::stringstream ss(spec); std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const size_t colon = part.find(':');
        if (colon == std::string::npos) { values.push_back(std::stoi(part)); continue; }
        std::vector<int> f; std::stringstream rs(part); std::string fld;
        while (std::getline(rs, fld, ':')) f.push_back(std::stoi(fld));
        const int step = f.size() == 3 ? f[2] : 1;
        for (int v = f[0]; v <= f[1]; v += step) values.push_back(v);
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto nv = [&](const char* n){ if (i+1>=argc){std::cerr<<"missing "<<n<<"\n";std::exit(2);} return argv[++i]; };
        if (!std::strcmp(argv[i], "--m-values")) a.m_values = nv("m");
        else if (!std::strcmp(argv[i], "--sm-values")) a.sm_values = nv("sm");
        else if (!std::strcmp(argv[i], "--k")) a.k = std::atoi(nv("k"));
        else if (!std::strcmp(argv[i], "--n")) a.n = std::atoi(nv("n"));
        else if (!std::strcmp(argv[i], "--warmup")) a.warmup = std::atoi(nv("w"));
        else if (!std::strcmp(argv[i], "--iters")) a.iters = std::atoi(nv("i"));
        else if (!std::strcmp(argv[i], "--device")) a.device = std::atoi(nv("d"));
    }
    return a;
}

static void fill_bf16(std::vector<__nv_bfloat16>& d, uint32_t s) {
    std::mt19937 g(s); std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : d) x = __float2bfloat16(dist(g));
}

static double median_ms(std::vector<float>& t) {
    std::sort(t.begin(), t.end()); const size_t n=t.size();
    return n%2 ? t[n/2] : 0.5*(double(t[n/2-1])+double(t[n/2]));
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    auto m_values = parse_int_values(args.m_values);
    auto sm_values = parse_int_values(args.sm_values);
    CHECK_CU(cuInit(0)); CHECK_CUDA(cudaSetDevice(args.device));
    cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, args.device));

    const int max_m = *std::max_element(m_values.begin(), m_values.end());
    const int hidden = args.k, intermediate = args.n;
    std::cerr << "[bwd cfg] " << prop.name << " K(hidden)=" << hidden << " N(inter)=" << intermediate
              << " threads=" << kThreads << " clusterDim=" << kClusterDim << std::endl;

    std::vector<__nv_bfloat16> h_graddown((size_t)max_m*hidden), h_wdown((size_t)hidden*intermediate),
        h_gradact((size_t)max_m*intermediate), h_dgu((size_t)max_m*2*intermediate),
        h_wgateup((size_t)2*intermediate*hidden), h_gradx((size_t)max_m*hidden);
    fill_bf16(h_graddown, 21); fill_bf16(h_wdown, 22); fill_bf16(h_dgu, 23); fill_bf16(h_wgateup, 24);

    __nv_bfloat16 *d_graddown,*d_wdown,*d_gradact,*d_dgu,*d_wgateup,*d_gradx;
    int *d_bc,*d_bp;
    CHECK_CUDA(cudaMalloc(&d_graddown, h_graddown.size()*2));
    CHECK_CUDA(cudaMalloc(&d_wdown, h_wdown.size()*2));
    CHECK_CUDA(cudaMalloc(&d_gradact, h_gradact.size()*2));
    CHECK_CUDA(cudaMalloc(&d_dgu, h_dgu.size()*2));
    CHECK_CUDA(cudaMalloc(&d_wgateup, h_wgateup.size()*2));
    CHECK_CUDA(cudaMalloc(&d_gradx, h_gradx.size()*2));
    CHECK_CUDA(cudaMalloc(&d_bc, sizeof(int))); CHECK_CUDA(cudaMalloc(&d_bp, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_graddown, h_graddown.data(), h_graddown.size()*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wdown, h_wdown.data(), h_wdown.size()*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dgu, h_dgu.data(), h_dgu.size()*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wgateup, h_wgateup.data(), h_wgateup.size()*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_gradact, 0, h_gradact.size()*2));
    CHECK_CUDA(cudaMemset(d_gradx, 0, h_gradx.size()*2));

    cudaStream_t stream=nullptr; CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaFuncSetAttribute((const void*)megakernel_backward_only_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

    for (int sms : sm_values) for (int m : m_values) {
        CUtensorMap desc_graddown_a = umma::dg_make_a_desc(d_graddown, m, hidden);
        CUtensorMap desc_wdown_mn   = umma::dg_make_b_mn_desc(d_wdown, intermediate, hidden);
        CUtensorMap desc_gradact_cd = umma::dg_make_cd_desc(d_gradact, m, intermediate);
        CUtensorMap desc_dgu_a      = umma::dg_make_a_desc(d_dgu, m, 2*intermediate);
        CUtensorMap desc_wgateup_mn = umma::dg_make_b_mn_desc(d_wgateup, hidden, 2*intermediate);
        CUtensorMap desc_gradx_cd   = umma::dg_make_cd_desc(d_gradx, m, hidden);
        GridBarrier barrier{d_bc, d_bp, sms};

        auto launch = [&](){
            cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(sms,1,1); cfg.blockDim=dim3(kThreads,1,1);
            cfg.dynamicSmemBytes=kSmemBytes; cfg.stream=stream;
            cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeClusterDimension;
            at[0].val.clusterDim.x=kClusterDim; at[0].val.clusterDim.y=1; at[0].val.clusterDim.z=1;
            cfg.attrs=at; cfg.numAttrs=1;
            CHECK_CUDA(cudaLaunchKernelEx(&cfg, megakernel_backward_only_kernel,
                desc_graddown_a, desc_wdown_mn, desc_gradact_cd,
                desc_dgu_a, desc_wgateup_mn, desc_gradx_cd, m, hidden, intermediate, barrier));
        };
        for (int i=0;i<args.warmup;++i){ CHECK_CUDA(cudaMemsetAsync(d_bc,0,sizeof(int),stream)); CHECK_CUDA(cudaMemsetAsync(d_bp,0,sizeof(int),stream)); launch(); }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> times;
        for (int i=0;i<args.iters;++i){
            cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
            CHECK_CUDA(cudaMemsetAsync(d_bc,0,sizeof(int),stream)); CHECK_CUDA(cudaMemsetAsync(d_bp,0,sizeof(int),stream));
            CHECK_CUDA(cudaEventRecord(s,stream)); launch(); CHECK_CUDA(cudaEventRecord(e,stream));
            CHECK_CUDA(cudaEventSynchronize(e)); float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));
            CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e)); times.push_back(ms);
        }
        const double ms=median_ms(times);
        const double flops = 6.0*double(m)*double(hidden)*double(intermediate); // grad_act 2mhi + grad_x 4mhi
        std::cout << "case m=" << std::setw(6) << m << " sms=" << std::setw(4) << sms
                  << " median_ms=" << std::fixed << std::setprecision(4) << std::setw(9) << ms
                  << " tflops=" << std::setprecision(2) << std::setw(9) << (flops/(ms*1e-3)/1e12) << std::endl;
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
