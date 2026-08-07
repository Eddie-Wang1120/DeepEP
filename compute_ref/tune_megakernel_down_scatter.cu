// tune_megakernel_down_scatter.cu
//
// Standalone operator-level harness for the PUBLISH_OFFLOAD_PLAN.md variant A:
//   gate/up UMMA + in-TMEM SwiGLU -> down UMMA epilogue row-scatter
//
// This file does not modify compute_ref/sm100_bf16_gemm_dg_copy.cuh or active
// megakernel headers. It includes the active UMMA helper for the old/reference
// path and defines a local experimental down-scatter persistent GEMM path.
//
// Build from compute_ref/ on B30Z/SM103a:
//   nvcc -std=c++17 -O3 -DMK_COMPUTE_KERNEL=1 \
//        -gencode=arch=compute_103a,code=sm_103a \
//        -I../DeepGEMM/deep_gemm/include \
//        -I../DeepGEMM/third-party/cutlass/include \
//        --expt-relaxed-constexpr -diag-suppress 20281 -diag-suppress 2361 \
//        tune_megakernel_down_scatter.cu -o tune_megakernel_down_scatter -lcuda
//
// Example:
//   ./tune_megakernel_down_scatter --m 128 --batch-size 117 --hidden 2048 --intermediate 3072 --sms 16

#include <algorithm>
#include <cmath>
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
static_assert(umma::kDgRunMulticast == 1, "This harness validates the current 1-CTA path; compile with -DMK_COMPUTE_KERNEL=1");

struct GridBarrier {
    int* counter;
    int* phase;
    int expected_blocks;
};

__device__ __forceinline__ int volatile_load_int(const int* ptr) {
    return *reinterpret_cast<const volatile int*>(ptr);
}

__device__ __forceinline__ void grid_barrier(GridBarrier barrier) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        const int observed_phase = volatile_load_int(barrier.phase);
        const int arrived = atomicAdd(barrier.counter, 1) + 1;
        if (arrived == barrier.expected_blocks) {
            __threadfence();
            atomicExch(barrier.counter, 0);
            __threadfence();
            atomicExch(barrier.phase, observed_phase + 1);
        }
        while (volatile_load_int(barrier.phase) == observed_phase) __nanosleep(64);
    }
    __syncthreads();
}

namespace down_scatter_exp {

using namespace cute;
using namespace deep_gemm;

struct ScatterParams {
    int4* combine_input_i4;
    int4* compute_output_slot_i4;
    const int* recv_token_idx;
    const int* is_single;
    int slot_base;
    int batch_size;
    int hidden_int4;
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void sm100_store_cd_row_scatter(
    const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
    const uint32_t& tmem_base_addr,
    const uint32_t& base_m_idx, const uint32_t& base_n_idx,
    const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
    const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
    ScatterParams scatter) {

    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    static_assert(kSwizzleCDMode == 128, "This row-scatter harness assumes 128B CD swizzle");
    static_assert(kNumElemsPerBankGroup == 8, "BF16 float4 row scatter expects 8 BF16 per segment");
    static_assert(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Only BF16 output is supported");
    static_assert(STORE_BLOCK_M == 128 && STORE_BLOCK_N == 64 && BLOCK_N == 128, "Expected current DeepGEMM tile shape");

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    constexpr uint32_t kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    constexpr uint32_t kSegmentsPerStoreRow = STORE_BLOCK_N / kNumElemsPerBankGroup;  // 8 float4 segments
    constexpr uint32_t kScatterItems = STORE_BLOCK_M * kSegmentsPerStoreRow;

    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);
            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_type_t::template apply_index_n<STORE_BLOCK_N>(base_n_idx + s * STORE_BLOCK_N);

            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                auto smem_ptr = smem_base_ptr +
                                epilogue_warp_idx * 32 * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                uint32_t values[kNumElemsPerBankGroup];
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                    values[0], values[1], values[2], values[3],
                    values[4], values[5], values[6], values[7]);
                cutlass::arch::fence_view_async_tmem_load();
                ptx::st_shared(
                    smem_ptr,
                    math::cast_into_bf16_and_pack(values[0], values[1]),
                    math::cast_into_bf16_and_pack(values[2], values[3]),
                    math::cast_into_bf16_and_pack(values[4], values[5]),
                    math::cast_into_bf16_and_pack(values[6], values[7]));
            }

            if (w == kNumMWaves - 1 && s == kNumStores - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const uint32_t epi_tid = epilogue_warp_idx * 32 + lane_idx;
            for (uint32_t item = epi_tid; item < kScatterItems; item += kNumUMMAStoreThreads) {
                const uint32_t row_in_store = item / kSegmentsPerStoreRow;
                const uint32_t seg = item - row_in_store * kSegmentsPerStoreRow;
                const uint32_t global_row = m_idx + row_in_store;
                if (global_row >= static_cast<uint32_t>(scatter.batch_size)) continue;

                const uint32_t row_in_warp = row_in_store & 31u;
                const uint32_t warp_row_group = row_in_store >> 5;
                const uint32_t bank_group = seg ^ (row_in_warp & 7u);
                const auto smem_ptr = smem_base_ptr +
                    warp_row_group * 32 * kSwizzleCDMode +
                    row_in_warp * (kNumBankGroupBytes * 8) +
                    bank_group * kNumBankGroupBytes;
                const int4 packed = *reinterpret_cast<const int4*>(smem_ptr);

                const int recv_token = scatter.recv_token_idx[global_row];
                const uint32_t hidden_i4 = (n_idx >> 3) + seg;
                if (scatter.is_single[global_row]) {
                    scatter.combine_input_i4[(int64_t)recv_token * scatter.hidden_int4 + hidden_i4] = packed;
                } else {
                    scatter.compute_output_slot_i4[(int64_t)(scatter.slot_base + global_row) * scatter.hidden_int4 + hidden_i4] = packed;
                }
            }

            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            __syncwarp();
        }
    }
}

template <uint32_t kNumMulticast>
__device__ void dg_gemm_persistent_scatter(
    const CUtensorMap* desc_a, const CUtensorMap* desc_b,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter,
    ScatterParams scatter) {

    using L = umma::DgSmemLayout<kNumMulticast>;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t BLOCK_M = L::BLOCK_M, BLOCK_N = L::BLOCK_N, BLOCK_K = L::BLOCK_K;
    constexpr bool kIsMulticastOnA = L::kIsMulticastOnA;
    constexpr uint32_t LOAD_BLOCK_M = L::LOAD_BLOCK_M, LOAD_BLOCK_N = L::LOAD_BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = L::STORE_BLOCK_M, STORE_BLOCK_N = L::STORE_BLOCK_N;
    constexpr uint32_t kNumStages = L::kNumStages;
    constexpr uint32_t kNumEpilogueStages = L::kNumEpilogueStages;
    constexpr uint32_t kNumTMAStoreStages = L::kNumTMAStoreStages;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t kNumUMMAStoreThreads = L::kNumUMMAStoreThreads;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = L::SMEM_A_SIZE_PER_STAGE;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = L::SMEM_B_SIZE_PER_STAGE;

    const bool is_leader_cta = (kNumMulticast == 1) ? true : (cute::block_rank_in_cluster() == 0);
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    auto block_or_cluster_sync = [&]() {
        if constexpr (kNumMulticast > 1) comm::cluster_sync_with_relaxed_arrive();
        else __syncthreads();
    };

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(desc_a);
        cute::prefetch_tma_descriptor(desc_b);
    }

    uint8_t* smem_buffer = reinterpret_cast<uint8_t*>(cluster_smem);
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + i * L::SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + L::SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + L::SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto bar = L::barrier_start(cluster_smem);
    auto full_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return bar + i; });
    auto empty_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return bar + (kNumStages + i); });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return bar + (kNumStages * 2 + i); });
    auto tmem_empty_barriers= utils::PatternVisitor([=](const uint32_t& i) { return bar + (kNumStages * 2 + kNumEpilogueStages + i); });

    const auto num_total_k_blocks = math::ceil_div<uint32_t>(shape_k, BLOCK_K);
    const uint32_t cta_rank = (kNumMulticast > 1) ? cute::block_rank_in_cluster() : 0;
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= (stage_idx == 0);
    };

    if (warp_idx == 0 and cute::elect_one_sync()) {
        umma::MegaTileScheduler sched(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t m_block, n_block;
        while (sched.get_next_block(m_block, n_block)) {
            const uint32_t m_idx0 = m_block * BLOCK_M;
            const uint32_t n_idx0 = n_block * BLOCK_N;
            const uint32_t load_m_idx = m_idx0 + (kIsMulticastOnA ? cta_rank * LOAD_BLOCK_M : 0);
            const uint32_t load_n_idx = n_idx0 + (kIsMulticastOnA ? 0 : cta_rank * LOAD_BLOCK_N);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, umma::kDgSwizzleA, cutlass::bfloat16_t>(
                    desc_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, load_m_idx, kNumMulticast);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, umma::kDgSwizzleB, cutlass::bfloat16_t>(
                    desc_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, load_n_idx, kNumMulticast);
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta)
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                else
                    full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        constexpr bool kDoMergeStages = (kNumStages >= 8);
        constexpr uint32_t kNumMinStages = 8;
        constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages / kNumMinStages : 1;
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_ATOM_K, umma::kDgSwizzleA>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_ATOM_K, umma::kDgSwizzleB>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        auto umma_arrive = [](const uint64_t* barrier) {
            if constexpr (kNumMulticast == 1) cutlass::arch::umma_arrive(barrier);
            else { constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                   cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask); }
        };

        umma::MegaTileScheduler sched(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t m_block, n_block;
        uint32_t local_accum = accum_iter;
        while (sched.get_next_block(m_block, n_block)) {
            const uint32_t accum_stage_idx = local_accum % kNumEpilogueStages;
            const uint32_t accum_phase_idx = (local_accum / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();
                const auto a_base = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_base = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    using mma_t = cute::conditional_t<kNumMulticast == 1,
                                      ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                    auto issue_umma = [&]<uint32_t kUMMAKIdx>() {
                        constexpr uint32_t kAtomKIdx  = kUMMAKIdx * UMMA_K / BLOCK_ATOM_K;
                        constexpr uint32_t kInnerKIdx = kUMMAKIdx * UMMA_K % BLOCK_ATOM_K;
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, umma::kDgSwizzleA, cutlass::bfloat16_t>(
                                        a_base, kAtomKIdx * LOAD_BLOCK_M * BLOCK_ATOM_K, kInnerKIdx);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, umma::kDgSwizzleB, cutlass::bfloat16_t>(
                                        b_base, kAtomKIdx * LOAD_BLOCK_N * BLOCK_ATOM_K, kInnerKIdx);
                        mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                   kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
                    };
                    utils::for_each_static_until<BLOCK_K / UMMA_K>(
                        std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), issue_umma);
                }
                __syncwarp();
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (k_block_idx == num_total_k_blocks - 1)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            }
            ++local_accum;
        }
    } else if (warp_idx >= umma::kDgWsNonEpiThreads / 32 &&
               warp_idx < (umma::kDgWsNonEpiThreads + umma::kDgWsEpiThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - (umma::kDgWsNonEpiThreads / 32);
        uint32_t tma_stage_idx = 0;
        umma::MegaTileScheduler sched(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t m_block, n_block;
        uint32_t local_accum = accum_iter;
        while (sched.get_next_block(m_block, n_block)) {
            const uint32_t accum_stage_idx = local_accum % kNumEpilogueStages;
            const uint32_t accum_phase_idx = (local_accum / kNumEpilogueStages) & 1;
            const uint32_t m_idx0 = m_block * BLOCK_M;
            const uint32_t n_idx0 = n_block * BLOCK_N;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;

            sm100_store_cd_row_scatter<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                umma::kDgSwizzleCD, kNumTMAStoreStages, kNumUMMAStoreThreads,
                cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity>(
                smem_cd, tma_stage_idx, tmem_base_addr, m_idx0, n_idx0,
                epilogue_warp_idx, lane_idx,
                tmem_empty_barriers[accum_stage_idx], scatter);
            ++local_accum;
        }
    }

    {
        umma::MegaTileScheduler probe(shape_m, shape_n, cluster_idx, num_clusters);
        uint32_t mb, nb, tiles_done = 0;
        while (probe.get_next_block(mb, nb)) ++tiles_done;
        accum_iter += tiles_done;
    }
    __threadfence();
    block_or_cluster_sync();
}

__device__ inline void umma_down_scatter_persistent(
    const CUtensorMap* desc_act_a, const CUtensorMap* desc_wdown,
    int M, int hidden, int intermediate,
    int cluster_idx, int num_clusters,
    char* cluster_smem, uint32_t& accum_iter,
    ScatterParams scatter) {
    dg_gemm_persistent_scatter<umma::kDgRunMulticast>(
        desc_act_a, desc_wdown,
        (uint32_t)M, (uint32_t)hidden, (uint32_t)intermediate,
        cluster_idx, num_clusters, cluster_smem, accum_iter, scatter);
}

}  // namespace down_scatter_exp

__global__ void __launch_bounds__(kThreads, 1)
old_downbuf_then_scatter_kernel_ptr(
    const __grid_constant__ CUtensorMap desc_a,
    const __grid_constant__ CUtensorMap desc_wgateup,
    const __grid_constant__ CUtensorMap desc_act_cd,
    const __grid_constant__ CUtensorMap desc_act_a,
    const __grid_constant__ CUtensorMap desc_wdown,
    const __grid_constant__ CUtensorMap desc_down_cd,
    const float* __restrict__ route_w,
    const __nv_bfloat16* down_buf,
    int4* combine_input_i4,
    int4* compute_output_slot_i4,
    const int* recv_token_idx,
    const int* is_single,
    int slot_base,
    int m, int batch_size, int hidden, int intermediate, int hidden_int4,
    GridBarrier barrier) {

    extern __shared__ __align__(1024) char cluster_smem[];
    const int cluster_idx = blockIdx.x / kClusterDim;
    const int num_clusters = gridDim.x / kClusterDim;

    uint32_t accum = 0;
    umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
    umma::umma_gateup_interleaved_persistent(&desc_a, &desc_wgateup, &desc_act_cd,
        route_w, m, intermediate, hidden, cluster_idx, num_clusters, cluster_smem, accum,
        nullptr, nullptr, nullptr, 0, 0, batch_size);
    grid_barrier(barrier);
    umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
    accum = 0;
    umma::umma_down_persistent(&desc_act_a, &desc_wdown, &desc_down_cd,
        m, hidden, intermediate, cluster_idx, num_clusters, cluster_smem, accum);
    umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);

    grid_barrier(barrier);

    const int4* down_i4 = reinterpret_cast<const int4*>(down_buf);
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = global_tid; idx < batch_size * hidden_int4; idx += stride) {
        const int row = idx / hidden_int4;
        const int v = idx - row * hidden_int4;
        const int recv = recv_token_idx[row];
        if (is_single[row])
            combine_input_i4[(int64_t)recv * hidden_int4 + v] = down_i4[(int64_t)row * hidden_int4 + v];
        else
            compute_output_slot_i4[(int64_t)(slot_base + row) * hidden_int4 + v] = down_i4[(int64_t)row * hidden_int4 + v];
    }
    __threadfence();
}

__global__ void __launch_bounds__(kThreads, 1)
down_epilogue_scatter_kernel(
    const __grid_constant__ CUtensorMap desc_a,
    const __grid_constant__ CUtensorMap desc_wgateup,
    const __grid_constant__ CUtensorMap desc_act_cd,
    const __grid_constant__ CUtensorMap desc_act_a,
    const __grid_constant__ CUtensorMap desc_wdown,
    const float* __restrict__ route_w,
    int4* combine_input_i4,
    int4* compute_output_slot_i4,
    const int* recv_token_idx,
    const int* is_single,
    int slot_base,
    int m, int batch_size, int hidden, int intermediate, int hidden_int4,
    GridBarrier barrier) {

    extern __shared__ __align__(1024) char cluster_smem[];
    const int cluster_idx = blockIdx.x / kClusterDim;
    const int num_clusters = gridDim.x / kClusterDim;

    uint32_t accum = 0;
    umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
    umma::umma_gateup_interleaved_persistent(&desc_a, &desc_wgateup, &desc_act_cd,
        route_w, m, intermediate, hidden, cluster_idx, num_clusters, cluster_smem, accum,
        nullptr, nullptr, nullptr, 0, 0, batch_size);
    grid_barrier(barrier);
    umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
    accum = 0;

    down_scatter_exp::ScatterParams scatter{
        combine_input_i4, compute_output_slot_i4, recv_token_idx, is_single,
        slot_base, batch_size, hidden_int4};
    down_scatter_exp::umma_down_scatter_persistent(&desc_act_a, &desc_wdown,
        m, hidden, intermediate, cluster_idx, num_clusters, cluster_smem, accum, scatter);
    umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);
    __threadfence();
}

struct Args {
    int m = 128;
    int batch_size = 117;
    int hidden = 2048;
    int intermediate = 3072;
    int sms = 16;
    int warmup = 5;
    int iters = 20;
    int device = 0;
    std::string pattern = "mix";
};

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--m M] [--batch-size B] [--hidden K] [--intermediate N]"
              << " [--sms SMS] [--warmup W] [--iters I] [--device D]"
              << " [--pattern single|multi|mix]\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; usage(argv[0]); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--m")) args.m = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--batch-size")) args.batch_size = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--hidden")) args.hidden = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--intermediate")) args.intermediate = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--sms")) args.sms = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--warmup")) args.warmup = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--iters")) args.iters = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--device")) args.device = std::atoi(need(argv[i]));
        else if (!std::strcmp(argv[i], "--pattern")) args.pattern = need(argv[i]);
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown option " << argv[i] << "\n"; usage(argv[0]); std::exit(2); }
    }
    return args;
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

static void fill_routing(std::vector<int>& recv, std::vector<int>& single, int batch_size, const std::string& pattern) {
    recv.resize(batch_size);
    single.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        recv[i] = batch_size - 1 - i;
        if (pattern == "single") single[i] = 1;
        else if (pattern == "multi") single[i] = 0;
        else single[i] = (i % 3) != 1;
    }
}

static void reset_barrier(GridBarrier barrier, cudaStream_t stream) {
    CHECK_CUDA(cudaMemsetAsync(barrier.counter, 0, sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(barrier.phase, 0, sizeof(int), stream));
}

static void launch_old(int sms, int m, int batch_size, int hidden, int intermediate,
                       const CUtensorMap& desc_a, const CUtensorMap& desc_wgateup,
                       const CUtensorMap& desc_act_cd, const CUtensorMap& desc_act_a,
                       const CUtensorMap& desc_wdown, const CUtensorMap& desc_down_cd,
                       const float* route, const __nv_bfloat16* down_buf,
                       int4* combine_i4, int4* slot_i4, const int* recv, const int* single,
                       int slot_base, int hidden_int4, GridBarrier barrier, cudaStream_t stream) {
    cudaLaunchConfig_t cfg{};
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
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, old_downbuf_then_scatter_kernel_ptr,
        desc_a, desc_wgateup, desc_act_cd, desc_act_a, desc_wdown, desc_down_cd,
        route, down_buf, combine_i4, slot_i4, recv, single, slot_base,
        m, batch_size, hidden, intermediate, hidden_int4, barrier));
}

static void launch_new(int sms, int m, int batch_size, int hidden, int intermediate,
                       const CUtensorMap& desc_a, const CUtensorMap& desc_wgateup,
                       const CUtensorMap& desc_act_cd, const CUtensorMap& desc_act_a,
                       const CUtensorMap& desc_wdown,
                       const float* route, int4* combine_i4, int4* slot_i4,
                       const int* recv, const int* single,
                       int slot_base, int hidden_int4, GridBarrier barrier, cudaStream_t stream) {
    (void)barrier;
    cudaLaunchConfig_t cfg{};
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
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, down_epilogue_scatter_kernel,
        desc_a, desc_wgateup, desc_act_cd, desc_act_a, desc_wdown,
        route, combine_i4, slot_i4, recv, single, slot_base,
        m, batch_size, hidden, intermediate, hidden_int4, barrier));
}

static double median_ms(std::vector<float>& times) {
    std::sort(times.begin(), times.end());
    const size_t n = times.size();
    return n & 1 ? times[n / 2] : 0.5 * (double(times[n / 2 - 1]) + double(times[n / 2]));
}

static void compare_outputs(const std::vector<int4>& ref, const std::vector<int4>& got, const char* name) {
    if (ref.size() != got.size()) {
        std::cerr << name << " size mismatch\n";
        std::exit(1);
    }
    size_t mismatches = 0;
    size_t first = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const uint4& a = reinterpret_cast<const uint4&>(ref[i]);
        const uint4& b = reinterpret_cast<const uint4&>(got[i]);
        if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
            if (mismatches++ == 0) first = i;
        }
    }
    if (mismatches) {
        const uint4& a = reinterpret_cast<const uint4&>(ref[first]);
        const uint4& b = reinterpret_cast<const uint4&>(got[first]);
        std::cerr << name << " FAIL mismatches=" << mismatches << " first_i=" << first
                  << " ref=(" << a.x << "," << a.y << "," << a.z << "," << a.w << ")"
                  << " got=(" << b.x << "," << b.y << "," << b.z << "," << b.w << ")\n";
        std::exit(1);
    }
    std::cout << name << " PASS bytewise\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    CHECK_CU(cuInit(0));
    CHECK_CUDA(cudaSetDevice(args.device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, args.device));

    if (args.m <= 0 || args.batch_size <= 0 || args.batch_size > args.m) {
        std::cerr << "Require 0 < batch-size <= m\n";
        return 2;
    }
    if (args.hidden % 128 != 0 || args.intermediate % 128 != 0 || args.intermediate % umma::kDgKAlign != 0) {
        std::cerr << "Require hidden/intermediate multiples of 128\n";
        return 2;
    }
    if (args.sms <= 0 || args.sms % kClusterDim != 0 || args.sms > prop.multiProcessorCount) {
        std::cerr << "Invalid sms=" << args.sms << " device_sms=" << prop.multiProcessorCount << "\n";
        return 2;
    }
    if (args.pattern != "single" && args.pattern != "multi" && args.pattern != "mix") {
        std::cerr << "Invalid pattern=" << args.pattern << " expected single|multi|mix\n";
        return 2;
    }
    const int hidden_int4 = args.hidden / 8;
    const int slot_base = 0;

    std::cout << "[cfg] device=" << prop.name << " sms=" << args.sms
              << " m=" << args.m << " batch=" << args.batch_size
              << " hidden=" << args.hidden << " intermediate=" << args.intermediate
              << " warmup=" << args.warmup << " iters=" << args.iters
              << " pattern=" << args.pattern << "\n";

    std::vector<__nv_bfloat16> h_a((size_t)args.m * args.hidden);
    std::vector<__nv_bfloat16> h_wgateup((size_t)2 * args.intermediate * args.hidden);
    std::vector<__nv_bfloat16> h_wdown((size_t)args.hidden * args.intermediate);
    std::vector<float> h_route((size_t)args.m);
    std::vector<int> h_recv, h_single;
    fill_bf16(h_a, 2001);
    fill_bf16(h_wgateup, 2002);
    fill_bf16(h_wdown, 2003);
    fill_float(h_route, 2004);
    fill_routing(h_recv, h_single, args.batch_size, args.pattern);

    __nv_bfloat16 *d_a = nullptr, *d_wgateup = nullptr, *d_act_old = nullptr, *d_act_new = nullptr;
    __nv_bfloat16 *d_wdown = nullptr, *d_down = nullptr;
    float* d_route = nullptr;
    int *d_recv = nullptr, *d_single = nullptr, *d_barrier_counter = nullptr, *d_barrier_phase = nullptr;
    int4 *d_ref_combine = nullptr, *d_ref_slot = nullptr, *d_new_combine = nullptr, *d_new_slot = nullptr;

    const size_t a_bytes = h_a.size() * sizeof(__nv_bfloat16);
    const size_t wgu_bytes = h_wgateup.size() * sizeof(__nv_bfloat16);
    const size_t wd_bytes = h_wdown.size() * sizeof(__nv_bfloat16);
    const size_t act_bytes = (size_t)args.m * args.intermediate * sizeof(__nv_bfloat16);
    const size_t down_bytes = (size_t)args.m * args.hidden * sizeof(__nv_bfloat16);
    const size_t out_i4_count = (size_t)args.batch_size * hidden_int4;
    const size_t out_bytes = out_i4_count * sizeof(int4);

    CHECK_CUDA(cudaMalloc(&d_a, a_bytes));
    CHECK_CUDA(cudaMalloc(&d_wgateup, wgu_bytes));
    CHECK_CUDA(cudaMalloc(&d_act_old, act_bytes));
    CHECK_CUDA(cudaMalloc(&d_act_new, act_bytes));
    CHECK_CUDA(cudaMalloc(&d_wdown, wd_bytes));
    CHECK_CUDA(cudaMalloc(&d_down, down_bytes));
    CHECK_CUDA(cudaMalloc(&d_route, h_route.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_recv, h_recv.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_single, h_single.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_barrier_counter, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_barrier_phase, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ref_combine, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_ref_slot, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_new_combine, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_new_slot, out_bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wgateup, h_wgateup.data(), wgu_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wdown, h_wdown.data(), wd_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_route, h_route.data(), h_route.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_recv, h_recv.data(), h_recv.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_single, h_single.data(), h_single.size() * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaFuncSetAttribute((const void*)old_downbuf_then_scatter_kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    CHECK_CUDA(cudaFuncSetAttribute((const void*)down_epilogue_scatter_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

    CUtensorMap desc_a = umma::dg_make_a_desc(d_a, args.m, args.hidden);
    CUtensorMap desc_wgateup = umma::dg_make_b_desc(d_wgateup, 2 * args.intermediate, args.hidden);
    CUtensorMap desc_act_old_cd = umma::dg_make_cd_desc(d_act_old, args.m, args.intermediate);
    CUtensorMap desc_act_old_a = umma::dg_make_a_desc(d_act_old, args.m, args.intermediate);
    CUtensorMap desc_act_new_cd = umma::dg_make_cd_desc(d_act_new, args.m, args.intermediate);
    CUtensorMap desc_act_new_a = umma::dg_make_a_desc(d_act_new, args.m, args.intermediate);
    CUtensorMap desc_wdown = umma::dg_make_b_desc(d_wdown, args.hidden, args.intermediate);
    CUtensorMap desc_down_cd = umma::dg_make_cd_desc(d_down, args.m, args.hidden);
    GridBarrier barrier{d_barrier_counter, d_barrier_phase, args.sms};

    auto clear_outputs = [&]() {
        CHECK_CUDA(cudaMemsetAsync(d_act_old, 0, act_bytes, stream));
        CHECK_CUDA(cudaMemsetAsync(d_act_new, 0, act_bytes, stream));
        CHECK_CUDA(cudaMemsetAsync(d_down, 0, down_bytes, stream));
        CHECK_CUDA(cudaMemsetAsync(d_ref_combine, 0xA5, out_bytes, stream));
        CHECK_CUDA(cudaMemsetAsync(d_ref_slot, 0x5A, out_bytes, stream));
        CHECK_CUDA(cudaMemsetAsync(d_new_combine, 0xA5, out_bytes, stream));
        CHECK_CUDA(cudaMemsetAsync(d_new_slot, 0x5A, out_bytes, stream));
    };

    clear_outputs();
    reset_barrier(barrier, stream);
    launch_old(args.sms, args.m, args.batch_size, args.hidden, args.intermediate,
               desc_a, desc_wgateup, desc_act_old_cd, desc_act_old_a, desc_wdown, desc_down_cd,
               d_route, d_down, d_ref_combine, d_ref_slot, d_recv, d_single,
               slot_base, hidden_int4, barrier, stream);
    launch_new(args.sms, args.m, args.batch_size, args.hidden, args.intermediate,
               desc_a, desc_wgateup, desc_act_new_cd, desc_act_new_a, desc_wdown,
               d_route, d_new_combine, d_new_slot, d_recv, d_single,
               slot_base, hidden_int4, barrier, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<int4> h_ref_combine(out_i4_count), h_ref_slot(out_i4_count);
    std::vector<int4> h_new_combine(out_i4_count), h_new_slot(out_i4_count);
    CHECK_CUDA(cudaMemcpy(h_ref_combine.data(), d_ref_combine, out_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ref_slot.data(), d_ref_slot, out_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_new_combine.data(), d_new_combine, out_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_new_slot.data(), d_new_slot, out_bytes, cudaMemcpyDeviceToHost));
    compare_outputs(h_ref_combine, h_new_combine, "combine_input");
    compare_outputs(h_ref_slot, h_new_slot, "compute_output_slot");

    auto time_old = [&]() -> double {
        for (int i = 0; i < args.warmup; ++i) {
            clear_outputs();
            reset_barrier(barrier, stream);
            launch_old(args.sms, args.m, args.batch_size, args.hidden, args.intermediate,
                       desc_a, desc_wgateup, desc_act_old_cd, desc_act_old_a, desc_wdown, desc_down_cd,
                       d_route, d_down, d_ref_combine, d_ref_slot, d_recv, d_single,
                       slot_base, hidden_int4, barrier, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> times;
        for (int i = 0; i < args.iters; ++i) {
            clear_outputs();
            reset_barrier(barrier, stream);
            cudaEvent_t start = nullptr, stop = nullptr;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start, stream));
            launch_old(args.sms, args.m, args.batch_size, args.hidden, args.intermediate,
                       desc_a, desc_wgateup, desc_act_old_cd, desc_act_old_a, desc_wdown, desc_down_cd,
                       d_route, d_down, d_ref_combine, d_ref_slot, d_recv, d_single,
                       slot_base, hidden_int4, barrier, stream);
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            times.push_back(ms);
        }
        return median_ms(times);
    };

    auto time_new = [&]() -> double {
        for (int i = 0; i < args.warmup; ++i) {
            clear_outputs();
            launch_new(args.sms, args.m, args.batch_size, args.hidden, args.intermediate,
                       desc_a, desc_wgateup, desc_act_new_cd, desc_act_new_a, desc_wdown,
                       d_route, d_new_combine, d_new_slot, d_recv, d_single,
                       slot_base, hidden_int4, barrier, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> times;
        for (int i = 0; i < args.iters; ++i) {
            clear_outputs();
            cudaEvent_t start = nullptr, stop = nullptr;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start, stream));
            launch_new(args.sms, args.m, args.batch_size, args.hidden, args.intermediate,
                       desc_a, desc_wgateup, desc_act_new_cd, desc_act_new_a, desc_wdown,
                       d_route, d_new_combine, d_new_slot, d_recv, d_single,
                       slot_base, hidden_int4, barrier, stream);
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            times.push_back(ms);
        }
        return median_ms(times);
    };

    const double old_ms = time_old();
    const double new_ms = time_new();
    std::cout << std::fixed << std::setprecision(4)
              << "old_downbuf_scatter_ms=" << old_ms
              << " new_epilogue_scatter_ms=" << new_ms
              << " speedup=" << (old_ms / new_ms) << "x\n";

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_wgateup));
    CHECK_CUDA(cudaFree(d_act_old));
    CHECK_CUDA(cudaFree(d_act_new));
    CHECK_CUDA(cudaFree(d_wdown));
    CHECK_CUDA(cudaFree(d_down));
    CHECK_CUDA(cudaFree(d_route));
    CHECK_CUDA(cudaFree(d_recv));
    CHECK_CUDA(cudaFree(d_single));
    CHECK_CUDA(cudaFree(d_barrier_counter));
    CHECK_CUDA(cudaFree(d_barrier_phase));
    CHECK_CUDA(cudaFree(d_ref_combine));
    CHECK_CUDA(cudaFree(d_ref_slot));
    CHECK_CUDA(cudaFree(d_new_combine));
    CHECK_CUDA(cudaFree(d_new_slot));
    return 0;
}
