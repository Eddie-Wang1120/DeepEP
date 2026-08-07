// tune_megakernel_compute.cu
//
// Single-GPU, compute-only tuner for the MegaKernel compute task path:
//   gate/up GEMM + in-TMEM SwiGLU -> down GEMM
//
// This bypasses DeepEP Buffer/NVSHMEM/dispatch/combine and directly reuses the
// UMMA compute helpers used by the active megakernel forward/backward path.
//
// Build, from compute_ref/ on B30Z/SM103a:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_103a,code=sm_103a \
//        -I../DeepGEMM/deep_gemm/include \
//        -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        tune_megakernel_compute.cu -o tune_megakernel_compute -lcuda
//
// Example:
//   ./tune_megakernel_compute --m-values 128:4096:128 --sm-values 2,4,8,16,32,64,96,128

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct GridBarrier {
    int* counter;
    int* phase;
    int expected_blocks;
};

__device__ __forceinline__ int volatile_load(const int* ptr) {
    return *reinterpret_cast<const volatile int*>(ptr);
}

__device__ __forceinline__ void grid_barrier(GridBarrier barrier) {
    __syncthreads();
    __threadfence();

    if (threadIdx.x == 0) {
        const int observed_phase = volatile_load(barrier.phase);
        const int arrived = atomicAdd(barrier.counter, 1) + 1;
        if (arrived == barrier.expected_blocks) {
            __threadfence();
            atomicExch(barrier.counter, 0);
            __threadfence();
            atomicExch(barrier.phase, observed_phase + 1);
        }
        while (volatile_load(barrier.phase) == observed_phase) {
            __nanosleep(64);
        }
    }

    __syncthreads();
}

__global__ void __launch_bounds__(kThreads, 1)
megakernel_compute_only_kernel(
    const __grid_constant__ CUtensorMap desc_a,
    const __grid_constant__ CUtensorMap desc_wgateup,
    const __grid_constant__ CUtensorMap desc_act_cd,
    const __grid_constant__ CUtensorMap desc_act_a,
    const __grid_constant__ CUtensorMap desc_wdown,
    const __grid_constant__ CUtensorMap desc_down_cd,
    const float* __restrict__ route_w,
    int m, int hidden, int intermediate,
    cutlass::bfloat16_t* __restrict__ preact,
    const int* __restrict__ recv_idx,
    const int* __restrict__ topk_slot,
    GridBarrier barrier) {

    extern __shared__ __align__(1024) char cluster_smem[];

    const int cluster_idx = blockIdx.x / kClusterDim;
    const int num_clusters = gridDim.x / kClusterDim;

    // Route/preact metadata source: default GMEM (metadata moved OUT of smem).
    const float* route_src = route_w;
#if defined(MK_TUNE_PREACT) && defined(MK_TUNE_META_SMEM)
    // Baseline where metadata stays IN smem (like the current megakernel): stage
    // route_w / recv_idx / topk_slot into a smem region after the GEMM scratch and
    // read from there. Lets us measure the smem-vs-GMEM metadata cost at fixed config.
    using L = umma::DgSmemLayout<umma::kDgRunMulticast>;
    constexpr size_t kScratch =
        L::SMEM_CD_SIZE + L::kNumStages * (L::SMEM_A_SIZE_PER_STAGE + L::SMEM_B_SIZE_PER_STAGE) +
        (L::kNumStages * 3 + L::kNumEpilogueStages * 2 + 1) *
            sizeof(cutlass::arch::ClusterTransactionBarrier) + sizeof(uint32_t);
    constexpr size_t kMetaOff = (kScratch + 1024 - 1) & ~(size_t)(1024 - 1);
    float* s_route = reinterpret_cast<float*>(cluster_smem + kMetaOff);
    int* s_recv = reinterpret_cast<int*>(s_route + m);
    int* s_topk = s_recv + m;
    for (int i = threadIdx.x; i < m; i += blockDim.x) { s_route[i] = route_w[i]; s_recv[i] = recv_idx[i]; s_topk[i] = topk_slot[i]; }
    __syncthreads();
    route_src = s_route;
    const int* recv_src = s_recv; const int* topk_src = s_topk;
#else
    const int* recv_src = recv_idx; const int* topk_src = topk_slot;
#endif

    uint32_t gateup_accum_iter = 0;
    umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
    umma::umma_gateup_interleaved_persistent(
        &desc_a, &desc_wgateup, &desc_act_cd,
        route_src, m, intermediate, hidden,
        cluster_idx, num_clusters,
        cluster_smem, gateup_accum_iter
#ifdef MK_TUNE_PREACT
        // Validate metadata-from-GMEM: enable preact save reading recv_idx/topk_slot
        // per-row from GMEM (mirrors moving s_recv_token_idx/s_topk_slot out of smem).
        , preact, recv_src, topk_src, /*num_topk*/0, /*preact_stride*/2 * intermediate
#endif
    );
    umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);

    grid_barrier(barrier);

    uint32_t down_accum_iter = 0;
    umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
    umma::umma_down_persistent(
        &desc_act_a, &desc_wdown, &desc_down_cd,
        m, hidden, intermediate,
        cluster_idx, num_clusters,
        cluster_smem, down_accum_iter);
    umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);
}

struct Args {
    std::string m_values = "128:4096:128";
    std::string sm_values = "2,4,8,16,32,64,96,128";
    int k = 2048;
    int n = 3072;
    int threads = kThreads;
    int warmup = 10;
    int iters = 50;
    int device = 0;
    std::string csv_path;
};

struct Result {
    int m;
    int sms;
    double median_ms;
    double tflops;
    double tflops_per_sm;
};

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --m-values <list|start:stop[:step]>   default 128:4096:128\n"
        << "  --sm-values <list|start:stop[:step]>  default 2,4,8,16,32,64,96,128\n"
        << "  --k <hidden>                          default 2048\n"
        << "  --n <intermediate>                    default 3072\n"
        << "  --threads <threads>                   default 800, must stay 800\n"
        << "  --warmup <iters>                      default 10\n"
        << "  --iters <iters>                       default 50\n"
        << "  --device <cuda device>                default 0\n"
        << "  --csv <path>                          optional CSV output\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--m-values") == 0) args.m_values = need_value(argv[i]);
        else if (std::strcmp(argv[i], "--sm-values") == 0) args.sm_values = need_value(argv[i]);
        else if (std::strcmp(argv[i], "--k") == 0) args.k = std::atoi(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--n") == 0) args.n = std::atoi(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--threads") == 0) args.threads = std::atoi(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--warmup") == 0) args.warmup = std::atoi(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--iters") == 0) args.iters = std::atoi(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--device") == 0) args.device = std::atoi(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--csv") == 0) args.csv_path = need_value(argv[i]);
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            usage(argv[0]);
            std::exit(2);
        }
    }
    return args;
}

static std::vector<int> parse_int_values(const std::string& spec, const char* name) {
    std::vector<int> values;
    std::stringstream ss(spec);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const size_t colon = part.find(':');
        if (colon == std::string::npos) {
            values.push_back(std::stoi(part));
            continue;
        }

        std::vector<int> fields;
        std::stringstream rs(part);
        std::string field;
        while (std::getline(rs, field, ':')) fields.push_back(std::stoi(field));
        if (fields.size() < 2 || fields.size() > 3) {
            std::cerr << name << " range must be start:stop[:step], got " << part << std::endl;
            std::exit(2);
        }
        const int start = fields[0];
        const int stop = fields[1];
        const int step = fields.size() == 3 ? fields[2] : 1;
        if (step <= 0) {
            std::cerr << name << " range step must be positive, got " << part << std::endl;
            std::exit(2);
        }
        for (int v = start; v <= stop; v += step) values.push_back(v);
    }

    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    if (values.empty()) {
        std::cerr << name << " must contain at least one value" << std::endl;
        std::exit(2);
    }
    return values;
}

static void validate_args(const Args& args, const std::vector<int>& m_values,
                          const std::vector<int>& sm_values, int physical_sms) {
    if (args.threads != kThreads) {
        std::cerr << "threads is fixed by the megakernel compute CTA shape; expected 800, got "
                  << args.threads << std::endl;
        std::exit(2);
    }
    if (args.k <= 0 || args.n <= 0) {
        std::cerr << "K and N must be positive" << std::endl;
        std::exit(2);
    }
    if (args.k % umma::kDgKAlign != 0) {
        std::cerr << "K must be aligned to " << umma::kDgKAlign << ", got " << args.k << std::endl;
        std::exit(2);
    }
    if (args.n % 128 != 0) {
        std::cerr << "N should be a multiple of 128 for this UMMA tile path, got " << args.n << std::endl;
        std::exit(2);
    }
    if (args.warmup < 0 || args.iters <= 0) {
        std::cerr << "warmup must be >= 0 and iters must be > 0" << std::endl;
        std::exit(2);
    }
    for (int m : m_values) {
        if (m <= 0) {
            std::cerr << "M must be positive, got " << m << std::endl;
            std::exit(2);
        }
    }
    for (int sms : sm_values) {
        if (sms <= 0 || sms % kClusterDim != 0) {
            std::cerr << "SM count must be positive and divisible by clusterDim=" << kClusterDim
                      << ", got " << sms << std::endl;
            std::exit(2);
        }
        if (sms > physical_sms) {
            std::cerr << "SM count " << sms << " exceeds device SM count " << physical_sms << std::endl;
            std::exit(2);
        }
    }
}

static void fill_bf16(std::vector<__nv_bfloat16>& dst, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : dst) x = __float2bfloat16(dist(gen));
}

static void fill_float(std::vector<float>& dst, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.0f);
    for (auto& x : dst) x = dist(gen);
}

static void reset_barrier(GridBarrier barrier, cudaStream_t stream) {
    CHECK_CUDA(cudaMemsetAsync(barrier.counter, 0, sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(barrier.phase, 0, sizeof(int), stream));
}

static void launch_compute_only(int sms, int m, int hidden, int intermediate,
                                const CUtensorMap& desc_a,
                                const CUtensorMap& desc_wgateup,
                                const CUtensorMap& desc_act_cd,
                                const CUtensorMap& desc_act_a,
                                const CUtensorMap& desc_wdown,
                                const CUtensorMap& desc_down_cd,
                                const float* route_w,
                                cutlass::bfloat16_t* preact,
                                const int* recv_idx,
                                const int* topk_slot,
                                GridBarrier barrier,
                                cudaStream_t stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(sms, 1, 1);
    cfg.blockDim = dim3(kThreads, 1, 1);
    cfg.dynamicSmemBytes = kSmemBytes;
    cfg.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kClusterDim;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    CHECK_CUDA(cudaLaunchKernelEx(&cfg, megakernel_compute_only_kernel,
        desc_a, desc_wgateup, desc_act_cd, desc_act_a, desc_wdown, desc_down_cd,
        route_w, m, hidden, intermediate, preact, recv_idx, topk_slot, barrier));
}

static double median_ms(std::vector<float>& times) {
    std::sort(times.begin(), times.end());
    const size_t n = times.size();
    if (n % 2 == 1) return times[n / 2];
    return 0.5 * (double(times[n / 2 - 1]) + double(times[n / 2]));
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::vector<int> m_values = parse_int_values(args.m_values, "M values");
    std::vector<int> sm_values = parse_int_values(args.sm_values, "SM values");

    CHECK_CU(cuInit(0));
    CHECK_CUDA(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, args.device));
    validate_args(args, m_values, sm_values, prop.multiProcessorCount);

    const int max_m = *std::max_element(m_values.begin(), m_values.end());
    const int hidden = args.k;
    const int intermediate = args.n;

    std::cerr << "[cfg] device=" << args.device << " " << prop.name
              << " sms=" << prop.multiProcessorCount
              << " K=" << hidden << " N=" << intermediate
              << " threads=" << kThreads << " clusterDim=" << kClusterDim
              << " warmup=" << args.warmup << " iters=" << args.iters << std::endl;

    std::vector<__nv_bfloat16> h_a((size_t)max_m * hidden);
    std::vector<__nv_bfloat16> h_wgateup((size_t)2 * intermediate * hidden);
    std::vector<__nv_bfloat16> h_wdown((size_t)hidden * intermediate);
    std::vector<float> h_route((size_t)max_m);
    fill_bf16(h_a, 1001);
    fill_bf16(h_wgateup, 1002);
    fill_bf16(h_wdown, 1003);
    fill_float(h_route, 1004);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_wgateup = nullptr;
    __nv_bfloat16* d_act = nullptr;
    __nv_bfloat16* d_wdown = nullptr;
    __nv_bfloat16* d_down = nullptr;
    float* d_route = nullptr;
    int* d_barrier_counter = nullptr;
    int* d_barrier_phase = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, (size_t)max_m * hidden * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_wgateup, (size_t)2 * intermediate * hidden * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_act, (size_t)max_m * intermediate * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_wdown, (size_t)hidden * intermediate * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_down, (size_t)max_m * hidden * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_route, (size_t)max_m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_barrier_counter, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_barrier_phase, sizeof(int)));

    // Preact-save validation buffers (metadata read from GMEM per-row).
    cutlass::bfloat16_t* d_preact = nullptr;
    int* d_recv_idx = nullptr;
    int* d_topk_slot = nullptr;
    CHECK_CUDA(cudaMalloc(&d_preact, (size_t)max_m * 2 * intermediate * sizeof(cutlass::bfloat16_t)));
    CHECK_CUDA(cudaMalloc(&d_recv_idx, (size_t)max_m * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_topk_slot, (size_t)max_m * sizeof(int)));
    {
        std::vector<int> h_idx(max_m);
        for (int i = 0; i < max_m; ++i) h_idx[i] = i;
        CHECK_CUDA(cudaMemcpy(d_recv_idx, h_idx.data(), max_m * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_topk_slot, h_idx.data(), max_m * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_preact, 0, (size_t)max_m * 2 * intermediate * sizeof(cutlass::bfloat16_t)));
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wgateup, h_wgateup.data(), h_wgateup.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wdown, h_wdown.data(), h_wdown.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_route, h_route.data(), h_route.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_act, 0, (size_t)max_m * intermediate * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemset(d_down, 0, (size_t)max_m * hidden * sizeof(__nv_bfloat16)));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaFuncSetAttribute((const void*)megakernel_compute_only_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

    std::vector<Result> results;
    for (int sms : sm_values) {
        for (int m : m_values) {
            CUtensorMap desc_a = umma::dg_make_a_desc(d_a, m, hidden);
            CUtensorMap desc_wgateup = umma::dg_make_b_desc(d_wgateup, 2 * intermediate, hidden);
            CUtensorMap desc_act_cd = umma::dg_make_cd_desc(d_act, m, intermediate);
            CUtensorMap desc_act_a = umma::dg_make_a_desc(d_act, m, intermediate);
            CUtensorMap desc_wdown = umma::dg_make_b_desc(d_wdown, hidden, intermediate);
            CUtensorMap desc_down_cd = umma::dg_make_cd_desc(d_down, m, hidden);
            GridBarrier barrier{d_barrier_counter, d_barrier_phase, sms};

            for (int i = 0; i < args.warmup; ++i) {
                reset_barrier(barrier, stream);
                launch_compute_only(sms, m, hidden, intermediate,
                                    desc_a, desc_wgateup, desc_act_cd, desc_act_a,
                                    desc_wdown, desc_down_cd, d_route,
                                    d_preact, d_recv_idx, d_topk_slot, barrier, stream);
            }
            CHECK_CUDA(cudaStreamSynchronize(stream));

            std::vector<float> times;
            times.reserve(args.iters);
            for (int i = 0; i < args.iters; ++i) {
                cudaEvent_t start = nullptr;
                cudaEvent_t stop = nullptr;
                CHECK_CUDA(cudaEventCreate(&start));
                CHECK_CUDA(cudaEventCreate(&stop));
                reset_barrier(barrier, stream);
                CHECK_CUDA(cudaEventRecord(start, stream));
                launch_compute_only(sms, m, hidden, intermediate,
                                    desc_a, desc_wgateup, desc_act_cd, desc_act_a,
                                    desc_wdown, desc_down_cd, d_route,
                                    d_preact, d_recv_idx, d_topk_slot, barrier, stream);
                CHECK_CUDA(cudaEventRecord(stop, stream));
                CHECK_CUDA(cudaEventSynchronize(stop));
                float elapsed = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
                CHECK_CUDA(cudaEventDestroy(start));
                CHECK_CUDA(cudaEventDestroy(stop));
                times.push_back(elapsed);
            }

            const double ms = median_ms(times);
            const double flops = 6.0 * double(m) * double(hidden) * double(intermediate);
            const double tflops = flops / (ms * 1.0e-3) / 1.0e12;
            results.push_back(Result{m, sms, ms, tflops, tflops / double(sms)});

            std::cout << "case m=" << std::setw(6) << m
                      << " sms=" << std::setw(4) << sms
                      << " median_ms=" << std::fixed << std::setprecision(4) << std::setw(9) << ms
                      << " tflops=" << std::setprecision(2) << std::setw(9) << tflops
                      << " tflops_per_sm=" << std::setprecision(4) << std::setw(9) << (tflops / double(sms))
                      << std::endl;
        }
    }

    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
        if (a.tflops_per_sm != b.tflops_per_sm) return a.tflops_per_sm > b.tflops_per_sm;
        return a.median_ms < b.median_ms;
    });

    std::cout << "\nranked_by_tflops_per_sm\n";
    std::cout << std::setw(6) << "rank" << std::setw(8) << "M" << std::setw(8) << "SMs"
              << std::setw(14) << "median_ms" << std::setw(14) << "TFLOPS"
              << std::setw(18) << "TFLOPS/SM" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        std::cout << std::setw(6) << (i + 1)
                  << std::setw(8) << r.m
                  << std::setw(8) << r.sms
                  << std::fixed << std::setprecision(4) << std::setw(14) << r.median_ms
                  << std::setprecision(2) << std::setw(14) << r.tflops
                  << std::setprecision(4) << std::setw(18) << r.tflops_per_sm
                  << std::endl;
    }

    if (!args.csv_path.empty()) {
        FILE* fp = std::fopen(args.csv_path.c_str(), "w");
        if (!fp) {
            std::cerr << "Failed to open CSV output: " << args.csv_path << std::endl;
            std::exit(1);
        }
        std::fprintf(fp, "rank,m,sms,median_ms,tflops,tflops_per_sm\n");
        for (size_t i = 0; i < results.size(); ++i) {
            const Result& r = results[i];
            std::fprintf(fp, "%zu,%d,%d,%.6f,%.6f,%.9f\n",
                         i + 1, r.m, r.sms, r.median_ms, r.tflops, r.tflops_per_sm);
        }
        std::fclose(fp);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_wgateup));
    CHECK_CUDA(cudaFree(d_act));
    CHECK_CUDA(cudaFree(d_wdown));
    CHECK_CUDA(cudaFree(d_down));
    CHECK_CUDA(cudaFree(d_route));
    CHECK_CUDA(cudaFree(d_barrier_counter));
    CHECK_CUDA(cudaFree(d_barrier_phase));
    return 0;
}
