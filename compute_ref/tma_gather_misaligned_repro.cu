// tma_gather_misaligned_repro.cu
//
// Standalone repro for the megakernel combine single-hit TMA experiment.
// It isolates only: GMEM int4 row -> TMA load -> SMEM, mbarrier wait, optional
// warp publish to a packet buffer. It does not touch the main DeepEP extension.
//
// Build, for B30Z / Blackwell:
//   nvcc -std=c++17 -arch=sm_103a -O2 tma_gather_misaligned_repro.cu -o tma_gather_misaligned_repro
//   # If sm_103a is unavailable, try -arch=sm_100a.
//
// Run examples:
//   ./tma_gather_misaligned_repro 2048 8 0 0 2 10000  # mode 0: direct TMA into packet buffer
//   ./tma_gather_misaligned_repro 2048 8 1 0 2 10000  # mode 1: TMA into aligned load buffer
//   ./tma_gather_misaligned_repro 2048 8 2 0 2 10000  # mode 2: warp global load into packet buffer
//   ./tma_gather_misaligned_repro 2048 8 1 0 3 10000  # reproduce bad 3-barrier load buffer layout
//   ./tma_gather_misaligned_repro 2048 8 0 4 2 10000  # intentionally misalign SMEM destination by 4 bytes
//
// Args:
//   hidden_int4      number of int4 values in hidden row; default 2048 for hidden=16384 bf16
//   num_topk         packet metadata topk weights count; default 8
//   mode             0=direct_to_packet, 1=chunked_load_buffer_then_warp_copy, 2=warp_global_copy
//   smem_dst_offset  extra bytes added to TMA destination pointer; default 0
//   barrier_words    number of uint64_t barrier slots before load buffer; default 2
//   iters            timed loop iterations; default 1

#include <cuda_runtime.h>

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error " << cudaGetErrorString(err) << " at "     \
                      << __FILE__ << ":" << __LINE__ << std::endl;              \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

__host__ __device__ __forceinline__ int align_up_int(int value, int align) {
    return (value + align - 1) / align * align;
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" : : "r"(arrive_count), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "LAB_WAIT_REPRO: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE_REPRO; \n\t"
        "bra LAB_WAIT_REPRO; \n\t"
        "DONE_REPRO: \n\t"
        "}" : : "r"(mbar_int_ptr), "r"(phase), "r"(0x989680));
    phase ^= 1;
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" : : "r"(num_bytes), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void tma_load_1d(const void* smem_ptr, const void* gmem_ptr,
                                            uint64_t* mbar_ptr, int num_bytes) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    constexpr uint64_t kEvictNormal = 0x1000000000000000;
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;\n" : :
        "r"(smem_int_ptr), "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr), "l"(kEvictNormal) : "memory");
}

__global__ void tma_gather_repro_kernel(const int4* input, int4* output, int hidden_int4,
                                        int packet_bytes, int mode, int smem_dst_offset,
                                        int barrier_words, int iters, int* debug) {
    constexpr int kGatherChunkInt4 = 128;
    constexpr int kGatherNumStages = 2;
    extern __shared__ __align__(1024) uint8_t smem[];

    const int lane_id = threadIdx.x & 31;
    uint8_t* packet_buffer = smem;
    uint8_t* tma_dst = packet_buffer + smem_dst_offset;
    auto single_mbarrier = reinterpret_cast<uint64_t*>(packet_buffer + packet_bytes);
    auto stage_mbarrier = [=](int stage) {
        return reinterpret_cast<uint64_t*>(packet_buffer + packet_bytes + sizeof(uint64_t) + stage * sizeof(uint64_t));
    };
    auto load_buffer = packet_buffer + packet_bytes + barrier_words * static_cast<int>(sizeof(uint64_t));

    uint32_t single_phase = 0;
    if (lane_id == 0)
        mbarrier_init(single_mbarrier, 1);
    if (lane_id < kGatherNumStages)
        mbarrier_init(stage_mbarrier(lane_id), 1);
    if (lane_id == 0) {
        fence_barrier_init();
        debug[0] = packet_bytes;
        debug[1] = smem_dst_offset;
        debug[2] = static_cast<int>(reinterpret_cast<uintptr_t>(tma_dst) & 127);
        debug[3] = static_cast<int>(reinterpret_cast<uintptr_t>(single_mbarrier) & 15);
        debug[4] = static_cast<int>(reinterpret_cast<uintptr_t>(load_buffer) & 127);
    }
    __syncwarp();

    int4* packet_i4 = reinterpret_cast<int4*>(packet_buffer);
    int4* load_i4 = reinterpret_cast<int4*>(load_buffer);
    for (int iter = 0; iter < iters; ++iter) {
        if (mode == 0) {
            if (lane_id == 0) {
                tma_load_1d(tma_dst, input, single_mbarrier, hidden_int4 * static_cast<int>(sizeof(int4)));
                mbarrier_arrive_and_expect_tx(single_mbarrier, hidden_int4 * static_cast<int>(sizeof(int4)));
            }
            __syncwarp();
            mbarrier_wait(single_mbarrier, single_phase);
        } else if (mode == 1) {
            for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kGatherChunkInt4) {
                const int chunk_int4 = min(kGatherChunkInt4, hidden_int4 - chunk_base);
                const int chunk_bytes = chunk_int4 * static_cast<int>(sizeof(int4));
                if (lane_id == 0) {
                    tma_load_1d(load_buffer, input + chunk_base, single_mbarrier, chunk_bytes);
                    mbarrier_arrive_and_expect_tx(single_mbarrier, chunk_bytes);
                }
                __syncwarp();
                mbarrier_wait(single_mbarrier, single_phase);
                #pragma unroll
                for (int j = 0; j < kGatherChunkInt4 / 32; ++j) {
                    const int local_vi = lane_id + j * 32;
                    if (local_vi < chunk_int4)
                        packet_i4[chunk_base + local_vi] = load_i4[local_vi];
                }
            }
        } else {
            for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kGatherChunkInt4) {
                const int chunk_int4 = min(kGatherChunkInt4, hidden_int4 - chunk_base);
                #pragma unroll
                for (int j = 0; j < kGatherChunkInt4 / 32; ++j) {
                    const int local_vi = lane_id + j * 32;
                    if (local_vi < chunk_int4)
                        packet_i4[chunk_base + local_vi] = input[chunk_base + local_vi];
                }
            }
        }
        __syncwarp();
    }

    for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kGatherChunkInt4) {
        const int chunk_int4 = min(kGatherChunkInt4, hidden_int4 - chunk_base);
        #pragma unroll
        for (int j = 0; j < kGatherChunkInt4 / 32; ++j) {
            const int local_vi = lane_id + j * 32;
            if (local_vi < chunk_int4)
                output[chunk_base + local_vi] = packet_i4[chunk_base + local_vi];
        }
    }
    if (lane_id == 0) {
        debug[5] = mode;
        debug[6] = barrier_words;
        debug[7] = iters;
    }
}

int main(int argc, char** argv) {
    int hidden_int4 = argc > 1 ? std::atoi(argv[1]) : 2048;
    int num_topk = argc > 2 ? std::atoi(argv[2]) : 8;
    int mode = argc > 3 ? std::atoi(argv[3]) : 0;
    int smem_dst_offset = argc > 4 ? std::atoi(argv[4]) : 0;
    int barrier_words = argc > 5 ? std::atoi(argv[5]) : 2;
    int iters = argc > 6 ? std::atoi(argv[6]) : 1;

    const int hidden_bytes = hidden_int4 * static_cast<int>(sizeof(int4));
    const int packet_bytes = align_up_int(hidden_bytes + 8 + num_topk * static_cast<int>(sizeof(float)),
                                          static_cast<int>(sizeof(int4)));
    const int smem_bytes = packet_bytes + std::max(3, barrier_words) * static_cast<int>(sizeof(uint64_t)) +
                           2 * 128 * static_cast<int>(sizeof(int4)) + 1024;

    std::vector<int4> host_input(hidden_int4);
    for (int i = 0; i < hidden_int4; ++i)
        host_input[i] = make_int4(i, i + 1, i + 2, i + 3);

    int4* input = nullptr;
    int4* output = nullptr;
    int* debug = nullptr;
    CHECK_CUDA(cudaMalloc(&input, hidden_int4 * sizeof(int4)));
    CHECK_CUDA(cudaMalloc(&output, hidden_int4 * sizeof(int4)));
    CHECK_CUDA(cudaMalloc(&debug, 8 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(input, host_input.data(), hidden_int4 * sizeof(int4), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(output, 0, hidden_int4 * sizeof(int4)));
    CHECK_CUDA(cudaMemset(debug, 0, 8 * sizeof(int)));

    std::cout << "hidden_int4=" << hidden_int4
              << " hidden_bytes=" << hidden_bytes
              << " packet_bytes=" << packet_bytes
              << " smem_bytes=" << smem_bytes
              << " mode=" << mode
              << " smem_dst_offset=" << smem_dst_offset
              << " barrier_words=" << barrier_words
              << " iters=" << iters << std::endl;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    tma_gather_repro_kernel<<<1, 32, smem_bytes>>>(input, output, hidden_int4, packet_bytes,
                                                   mode, smem_dst_offset, barrier_words, iters, debug);
    CHECK_CUDA(cudaEventRecord(stop));
    cudaError_t sync_err = cudaDeviceSynchronize();
    float elapsed_ms = 0.0f;
    if (sync_err == cudaSuccess)
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::vector<int> host_debug(8, 0);
    cudaMemcpy(host_debug.data(), debug, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "debug packet_bytes=" << host_debug[0]
              << " smem_dst_offset=" << host_debug[1]
              << " tma_dst_mod128=" << host_debug[2]
              << " mbarrier_mod16=" << host_debug[3]
              << " load_buffer_mod128=" << host_debug[4]
              << " mode=" << host_debug[5]
              << " barrier_words=" << host_debug[6]
              << " iters=" << host_debug[7] << std::endl;
    if (sync_err != cudaSuccess) {
        std::cerr << "kernel failed: " << cudaGetErrorString(sync_err) << std::endl;
        return 1;
    }
    const double moved_gib = static_cast<double>(hidden_bytes) * static_cast<double>(iters) / (1024.0 * 1024.0 * 1024.0);
    const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
    std::cout << "elapsed_ms=" << elapsed_ms
              << " payload_GiB=" << moved_gib
              << " payload_GiB_per_s=" << (moved_gib / seconds)
              << " per_iter_us=" << (elapsed_ms * 1000.0 / static_cast<double>(iters))
              << std::endl;

    std::vector<int4> host_output(hidden_int4);
    CHECK_CUDA(cudaMemcpy(host_output.data(), output, hidden_int4 * sizeof(int4), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < hidden_int4; ++i) {
        if (host_output[i].x != host_input[i].x || host_output[i].y != host_input[i].y ||
            host_output[i].z != host_input[i].z || host_output[i].w != host_input[i].w) {
            std::cerr << "mismatch at int4 " << i << std::endl;
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(output);
    cudaFree(debug);
    return ok ? 0 : 2;
}
