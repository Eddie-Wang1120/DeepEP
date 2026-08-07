// bench_l2_reuse.cu
//
// Microbenchmark: measure the effect of L2 cache weight reuse for megakernel
// expert compute (gate/up + down GEMM).
//
// Scenario: 8 experts, each expert has ~24576 tokens (192 batches of 128).
// We compare:
//   (A) L2-hot:  process all batches of ONE expert, then move to the next
//   (B) L2-cold: interleave experts (expert0-batch0, expert1-batch0, ..., expert0-batch1, ...)
//
// For the given case:
//   num_tokens=32768, hidden=2048, intermediate=3072, experts_per_rank=8, num_topk=6
//   Total expert-token pairs = 32768*6 = 196608, per expert = 24576 tokens
//   COMPUTE_BATCH_SIZE = 128, so 192 batches per expert
//
// Build (from compute_ref/ on B30Z):
//   nvcc -std=c++17 -O3 -gencode=arch=compute_103a,code=sm_103a \
//        -I../DeepGEMM/deep_gemm/include -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        bench_l2_reuse.cu -o bench_l2_reuse -lcuda

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define EP_HOST_ASSERT(cond) do {                                              \
    if (!(cond)) {                                                             \
        std::cerr << "ASSERT failed: " #cond " at " << __FILE__ << ":"        \
                  << __LINE__ << std::endl;                                    \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

#include "../csrc/kernels/megakernel_compute_umma.cuh"

#define CHECK_CUDA(call) do {                                                  \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
        std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at "     \
                  << __FILE__ << ":" << __LINE__ << std::endl;               \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

#define CHECK_CU(call) do {                                                    \
    CUresult err__ = (call);                                                   \
    if (err__ != CUDA_SUCCESS) {                                               \
        const char* msg__ = nullptr;                                           \
        cuGetErrorString(err__, &msg__);                                       \
        std::cerr << "CU error " << (msg__ ? msg__ : "unknown") << " at "    \
                  << __FILE__ << ":" << __LINE__ << std::endl;               \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

namespace umma = deep_ep::megakernel::umma;

static constexpr int kThreads = 800;
static constexpr int kClusterDim = umma::kDgRunMulticast;
static constexpr int kSmemBytes = 227 * 1024;

struct GridBarrier { int* counter; int* phase; int expected_blocks; };

__device__ __forceinline__ int volatile_load(const int* ptr) {
    return *reinterpret_cast<const volatile int*>(ptr);
}
__device__ __forceinline__ void grid_barrier_fn(GridBarrier barrier) {
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

// Kernel: run a sequence of expert compute tasks.
// task_expert_ids[i] tells which expert to use for i-th task.
// Each task does: gate/up GEMM (M=batch_m, N=intermediate, K=hidden) + down GEMM (M=batch_m, N=hidden, K=intermediate)
// reading from expert-specific weight pointers.
__global__ void __launch_bounds__(kThreads, 1)
bench_expert_sequence_kernel(
    // Per-expert TMA descriptors (gate/up B, down B), stored in arrays
    const CUtensorMap* __restrict__ descs_wgateup,  // [num_experts]
    const CUtensorMap* __restrict__ descs_wdown,     // [num_experts]
    // Shared A/CD descriptors (input/output, same layout for all experts)
    const __grid_constant__ CUtensorMap desc_a,
    const __grid_constant__ CUtensorMap desc_act_cd,
    const __grid_constant__ CUtensorMap desc_act_a,
    const __grid_constant__ CUtensorMap desc_down_cd,
    // Task schedule
    const int* __restrict__ task_expert_ids,
    int num_tasks,
    // Dimensions
    int batch_m, int hidden, int intermediate,
    // Route weights (dummy)
    const float* __restrict__ route_w,
    GridBarrier barrier) {

    extern __shared__ __align__(1024) char cluster_smem[];
    const int cluster_idx = blockIdx.x / kClusterDim;
    const int num_clusters = gridDim.x / kClusterDim;

    for (int task = 0; task < num_tasks; ++task) {
        int expert_id = task_expert_ids[task];

        // Gate/Up GEMM
        uint32_t gateup_iter = 0;
        umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
        umma::umma_gateup_interleaved_persistent(
            &desc_a, &descs_wgateup[expert_id], &desc_act_cd,
            route_w, batch_m, intermediate, hidden,
            cluster_idx, num_clusters, cluster_smem, gateup_iter);
        umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);

        grid_barrier_fn(barrier);

        // Down GEMM
        uint32_t down_iter = 0;
        umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
        umma::umma_down_persistent(
            &desc_act_a, &descs_wdown[expert_id], &desc_down_cd,
            batch_m, hidden, intermediate,
            cluster_idx, num_clusters, cluster_smem, down_iter);
        umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);

        // Sync between tasks
        if (task < num_tasks - 1)
            grid_barrier_fn(barrier);
    }
}

static void fill_bf16(std::vector<__nv_bfloat16>& dst, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : dst) x = __float2bfloat16(dist(gen));
}

static double median_ms(std::vector<float>& t) {
    std::sort(t.begin(), t.end());
    size_t n = t.size();
    return n % 2 ? t[n/2] : 0.5 * (double(t[n/2-1]) + double(t[n/2]));
}

int main(int argc, char** argv) {
    // Config matching the test case
    const int num_experts = 8;
    const int hidden = 2048;
    const int intermediate = 3072;
    const int batch_m = 128;  // COMPUTE_BATCH_SIZE
    const int batches_per_expert = 192;  // 24576 / 128
    const int total_tasks = num_experts * batches_per_expert;  // 1536

    // For the benchmark we only run a subset to keep iteration time manageable
    // Run 8 batches per expert (64 total tasks) to measure the steady-state effect
    const int bench_batches_per_expert = 8;
    const int bench_total_tasks = num_experts * bench_batches_per_expert;  // 64

    int sms = 48;  // Use 48 SMs (typical compute allocation in megakernel)
    int warmup = 5;
    int iters = 20;
    int device = 0;

    // Parse optional args
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--sms") && i+1 < argc) sms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batches") && i+1 < argc) {
            // override bench_batches_per_expert — not used, keep simple
        }
    }

    CHECK_CU(cuInit(0));
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cerr << "[bench_l2_reuse] " << prop.name << " sms=" << prop.multiProcessorCount
              << " using=" << sms << " hidden=" << hidden << " inter=" << intermediate
              << " batch_m=" << batch_m << " experts=" << num_experts
              << " bench_batches/expert=" << bench_batches_per_expert << std::endl;

    if (sms % kClusterDim != 0) {
        std::cerr << "sms must be divisible by clusterDim=" << kClusterDim << std::endl;
        return 1;
    }

    // Allocate per-expert weights
    size_t wgateup_size = (size_t)2 * intermediate * hidden;  // per expert
    size_t wdown_size = (size_t)hidden * intermediate;         // per expert
    std::vector<__nv_bfloat16*> d_wgateup(num_experts);
    std::vector<__nv_bfloat16*> d_wdown(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        CHECK_CUDA(cudaMalloc(&d_wgateup[e], wgateup_size * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_wdown[e], wdown_size * sizeof(__nv_bfloat16)));
        // Fill with random data (different seed per expert to ensure different data)
        std::vector<__nv_bfloat16> h_wgu(wgateup_size), h_wd(wdown_size);
        fill_bf16(h_wgu, 100 + e); fill_bf16(h_wd, 200 + e);
        CHECK_CUDA(cudaMemcpy(d_wgateup[e], h_wgu.data(), wgateup_size * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_wdown[e], h_wd.data(), wdown_size * 2, cudaMemcpyHostToDevice));
    }

    // Shared input/output buffers
    __nv_bfloat16 *d_input, *d_act, *d_output;
    float *d_route;
    CHECK_CUDA(cudaMalloc(&d_input, (size_t)batch_m * hidden * 2));
    CHECK_CUDA(cudaMalloc(&d_act, (size_t)batch_m * intermediate * 2));
    CHECK_CUDA(cudaMalloc(&d_output, (size_t)batch_m * hidden * 2));
    CHECK_CUDA(cudaMalloc(&d_route, (size_t)batch_m * sizeof(float)));
    {
        std::vector<__nv_bfloat16> h_in((size_t)batch_m * hidden);
        std::vector<float> h_route(batch_m, 1.0f);
        fill_bf16(h_in, 42);
        CHECK_CUDA(cudaMemcpy(d_input, h_in.data(), h_in.size() * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_route, h_route.data(), h_route.size() * 4, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMemset(d_act, 0, (size_t)batch_m * intermediate * 2));
    CHECK_CUDA(cudaMemset(d_output, 0, (size_t)batch_m * hidden * 2));

    // Build TMA descriptors
    CUtensorMap desc_a = umma::dg_make_a_desc(d_input, batch_m, hidden);
    CUtensorMap desc_act_cd = umma::dg_make_cd_desc(d_act, batch_m, intermediate);
    CUtensorMap desc_act_a = umma::dg_make_a_desc(d_act, batch_m, intermediate);
    CUtensorMap desc_down_cd = umma::dg_make_cd_desc(d_output, batch_m, hidden);

    // Per-expert weight descriptors
    std::vector<CUtensorMap> h_descs_wgateup(num_experts);
    std::vector<CUtensorMap> h_descs_wdown(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        h_descs_wgateup[e] = umma::dg_make_b_desc(d_wgateup[e], 2 * intermediate, hidden);
        h_descs_wdown[e] = umma::dg_make_b_desc(d_wdown[e], hidden, intermediate);
    }
    CUtensorMap *d_descs_wgateup, *d_descs_wdown;
    CHECK_CUDA(cudaMalloc(&d_descs_wgateup, num_experts * sizeof(CUtensorMap)));
    CHECK_CUDA(cudaMalloc(&d_descs_wdown, num_experts * sizeof(CUtensorMap)));
    CHECK_CUDA(cudaMemcpy(d_descs_wgateup, h_descs_wgateup.data(), num_experts * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_descs_wdown, h_descs_wdown.data(), num_experts * sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // Build task schedules
    // (A) L2-hot: expert0 x bench_batches, expert1 x bench_batches, ...
    std::vector<int> schedule_hot(bench_total_tasks);
    for (int e = 0; e < num_experts; ++e)
        for (int b = 0; b < bench_batches_per_expert; ++b)
            schedule_hot[e * bench_batches_per_expert + b] = e;

    // (B) L2-cold: round-robin across experts
    std::vector<int> schedule_cold(bench_total_tasks);
    for (int b = 0; b < bench_batches_per_expert; ++b)
        for (int e = 0; e < num_experts; ++e)
            schedule_cold[b * num_experts + e] = e;

    int *d_schedule_hot, *d_schedule_cold;
    CHECK_CUDA(cudaMalloc(&d_schedule_hot, bench_total_tasks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_schedule_cold, bench_total_tasks * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_schedule_hot, schedule_hot.data(), bench_total_tasks * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_schedule_cold, schedule_cold.data(), bench_total_tasks * 4, cudaMemcpyHostToDevice));

    // Barrier
    int *d_bc, *d_bp;
    CHECK_CUDA(cudaMalloc(&d_bc, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_bp, sizeof(int)));
    GridBarrier barrier{d_bc, d_bp, sms};

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaFuncSetAttribute((const void*)bench_expert_sequence_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

    auto run_bench = [&](const char* label, int* d_schedule) {
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            CHECK_CUDA(cudaMemsetAsync(d_bc, 0, 4, stream));
            CHECK_CUDA(cudaMemsetAsync(d_bp, 0, 4, stream));
            cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(sms,1,1); cfg.blockDim = dim3(kThreads,1,1);
            cfg.dynamicSmemBytes = kSmemBytes; cfg.stream = stream;
            cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
            at[0].val.clusterDim.x = kClusterDim; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
            cfg.attrs = at; cfg.numAttrs = 1;
            CHECK_CUDA(cudaLaunchKernelEx(&cfg, bench_expert_sequence_kernel,
                d_descs_wgateup, d_descs_wdown, desc_a, desc_act_cd, desc_act_a, desc_down_cd,
                d_schedule, bench_total_tasks, batch_m, hidden, intermediate, d_route, barrier));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Timed
        std::vector<float> times;
        for (int i = 0; i < iters; ++i) {
            cudaEvent_t s, e;
            CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
            CHECK_CUDA(cudaMemsetAsync(d_bc, 0, 4, stream));
            CHECK_CUDA(cudaMemsetAsync(d_bp, 0, 4, stream));
            CHECK_CUDA(cudaEventRecord(s, stream));
            cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(sms,1,1); cfg.blockDim = dim3(kThreads,1,1);
            cfg.dynamicSmemBytes = kSmemBytes; cfg.stream = stream;
            cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
            at[0].val.clusterDim.x = kClusterDim; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
            cfg.attrs = at; cfg.numAttrs = 1;
            CHECK_CUDA(cudaLaunchKernelEx(&cfg, bench_expert_sequence_kernel,
                d_descs_wgateup, d_descs_wdown, desc_a, desc_act_cd, desc_act_a, desc_down_cd,
                d_schedule, bench_total_tasks, batch_m, hidden, intermediate, d_route, barrier));
            CHECK_CUDA(cudaEventRecord(e, stream));
            CHECK_CUDA(cudaEventSynchronize(e));
            float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
            CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
            times.push_back(ms);
        }

        double med = median_ms(times);
        // FLOPs: each task = gateup (2*M*2I*H) + down (2*M*H*I) = 2M*H*I*(2+1)*2 = 6*M*H*I per task? No:
        // gateup: 2 * M * (2*intermediate) * hidden = 4*M*I*H
        // down:   2 * M * hidden * intermediate     = 2*M*H*I
        // total per task = 6*M*H*I
        double flops_per_task = 6.0 * batch_m * hidden * intermediate;
        double total_flops = flops_per_task * bench_total_tasks;
        double tflops = total_flops / (med * 1e-3) / 1e12;
        double ms_per_task = med / bench_total_tasks;

        std::cout << std::setw(12) << label
                  << "  total_ms=" << std::fixed << std::setprecision(3) << med
                  << "  ms/task=" << std::setprecision(4) << ms_per_task
                  << "  TFLOPS=" << std::setprecision(2) << tflops
                  << std::endl;
        return med;
    };

    std::cout << "\n=== L2 Cache Weight Reuse Benchmark ===" << std::endl;
    std::cout << "Config: " << num_experts << " experts, hidden=" << hidden
              << ", inter=" << intermediate << ", batch_m=" << batch_m
              << ", tasks=" << bench_total_tasks << ", sms=" << sms << std::endl;
    std::cout << "Expert weight size: gate/up=" << (wgateup_size*2/1024/1024) << "MB, down="
              << (wdown_size*2/1024/1024) << "MB, total=" << ((wgateup_size+wdown_size)*2/1024/1024) << "MB"
              << std::endl;
    std::cout << std::endl;

    double hot_ms = run_bench("L2-hot", d_schedule_hot);
    double cold_ms = run_bench("L2-cold", d_schedule_cold);

    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "L2-hot  (expert-major): " << std::fixed << std::setprecision(3) << hot_ms << " ms" << std::endl;
    std::cout << "L2-cold (round-robin):  " << std::fixed << std::setprecision(3) << cold_ms << " ms" << std::endl;
    std::cout << "Speedup (cold/hot):     " << std::setprecision(2) << (cold_ms / hot_ms) << "x" << std::endl;
    std::cout << "Overhead of L2 miss:    " << std::setprecision(1) << ((cold_ms - hot_ms) / hot_ms * 100.0) << "%" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream));
    for (int e = 0; e < num_experts; ++e) {
        CHECK_CUDA(cudaFree(d_wgateup[e])); CHECK_CUDA(cudaFree(d_wdown[e]));
    }
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_act)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_route));
    CHECK_CUDA(cudaFree(d_descs_wgateup)); CHECK_CUDA(cudaFree(d_descs_wdown));
    CHECK_CUDA(cudaFree(d_schedule_hot)); CHECK_CUDA(cudaFree(d_schedule_cold));
    CHECK_CUDA(cudaFree(d_bc)); CHECK_CUDA(cudaFree(d_bp));
    return 0;
}
