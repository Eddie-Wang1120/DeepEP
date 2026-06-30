/**
 * megakernel.cu — MK-v7: Fused Dispatch(DeepEP RDMA) + Compute(GEMM+SwiGLU) + Combine
 *
 * Architecture:
 *   Single persistent kernel with overlapping phases:
 *   - Dispatch SMs: Full DeepEP dispatch (even SM = forwarder, odd SM = sender)
 *     Uses even/odd SM pairing from internode.cu with 5 warp roles:
 *     kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder,
 *     kForwarderCoordinator, kNVLReceivers
 *   - Compute SMs: Poll expert_recv_count, batch >= 128 -> GEMM+SwiGLU
 *   - Combine: Serial after all compute done
 *
 * Overlap: Dispatch (NIC) and Compute (TensorCore) run concurrently on different SMs.
 *
 * Key design:
 *   - Dispatch completely reuses DeepEP internode.cu dispatch logic as __device__
 *   - NVLReceivers additionally route tokens to expert storage + atomicAdd expert_recv_count
 *   - Compute polls counter, triggers GEMM at threshold (COMPUTE_BATCH_SIZE=128)
 *   - Flush: dispatch_done flag -> compute processes tail < 128
 */

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "internode_common.cuh"
#include "launch.cuh"
#include "utils.cuh"
// S4.4 (route B2): Blackwell UMMA + 2CTA multicast TMA compute (CuTe). Isolated
// header; only included here (nvcc TU), never by deep_ep.cpp (g++).
#include "megakernel_compute_umma.cuh"

#include <cuda_bf16.h>
#include <mma.h>
#include <limits>
#ifdef MK_PERF_TRACE
#include <vector>
#include <cstdio>
#endif

namespace deep_ep {
namespace megakernel {

// ============================================================================
// Configuration
// ============================================================================

constexpr int COMPUTE_BATCH_SIZE = 256;  // Tokens per expert batch before triggering GEMM (S4.1: 128->256 for 2x1SM M_tile=256)
constexpr int COMPUTE_GROUP_SIZE = 32;   // SMs cooperating on one expert batch
constexpr int COMPUTE_SCHEDULER_SMS = 2; // Scheduler region (S4.2: 2 so launched grid is even for cluster_dim=2; only #0 works, #1 idles)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int MK_TIMEOUT_LOG_BUDGET = 8;

enum TimeoutLogSite {
    kTimeoutLogComputeRoundFlush = 0,
    kTimeoutLogComputeReady = 1,
    kTimeoutLogCombineRdmaReceiver = 2,
    kTimeoutLogCombineForwarderNvl = 3,
    kTimeoutLogDispatchRound = 4,
    kTimeoutLogDispatchChannel = 5,
    kTimeoutLogCombineRdmaCheck = 6,
    kTimeoutLogCombineNvlCheck = 7,
    kTimeoutLogCount = 8,
};

// Dispatch constants (matching internode.cu)
constexpr int MK_NUM_RDMA_RANKS = 2;            // 2 nodes

// Combine constants (matching DeepEP internode.cu combine kernel)
// For kNumRDMARanks=2, kNumCombineForwarderWarps=24:
//   kNumWarpsPerForwarder = 24/2 = 12
//   kNumForwarders = 2*12 = 24
//   kNumRDMAReceivers = 24 - 8 = 16
// Sender SM: 8 NVL senders + 16 RDMA receivers + 1 coordinator = 25 warps (800 threads)
// Forwarder SM: 24 forwarders + 1 coordinator = 25 warps (800 threads)
constexpr int kNumCombineForwarderWarps = 24;
constexpr int kNumCombineWarpsPerForwarder = kNumCombineForwarderWarps / MK_NUM_RDMA_RANKS;  // 12
constexpr int kNumCombineForwarders = MK_NUM_RDMA_RANKS * kNumCombineWarpsPerForwarder;      // 24
constexpr int kNumCombineRDMAReceivers = kNumCombineForwarders - NUM_MAX_NVL_PEERS;          // 16
constexpr int kNumCombineTMABytesPerSenderWarp = 16384;
// Per forwarder warp: 2 stages * (sizeof(int4)*32 * (NUM_MAX_NVL_PEERS+1) + 16)
constexpr int kNumCombineTMABytesPerForwarderWarp = 9248;
constexpr int kNumTopkCombineRDMARanks = MK_NUM_RDMA_RANKS;  // get_num_topk_rdma_ranks(2)=2
constexpr int kMegaKernelNumThreads = (kNumCombineForwarders + 1) * 32;

// SM Role assignment (configured at launch time)
enum class SmRole {
    kDispatch,      // Runs full DeepEP dispatch (even SM = forwarder, odd SM = sender)
    kCombine,       // DeepEP combine (even SM = NVLSender+RDMAReceiver, odd SM = Forwarder)
    kScheduler,     // Enqueues expert compute batches for dynamic compute groups
    kCompute        // Pops compute tasks, does GEMM+SwiGLU
};

struct ComputeTask {
    int expert_id;
    int start_slot;
    int num_tokens;
};

// ============================================================================
// MegaKernel State (device-side, passed as kernel arg)
// ============================================================================

struct MegaKernelState {
    int* timeout_log_counters;        // [kTimeoutLogCount] per-site bounded logging budget
    // --- DeepEP NVSHMEM infrastructure (from Buffer object) ---
    void* rdma_buffer_ptr;            // Symmetric RDMA buffer base (for SymBuffer construction)
    void** buffer_ptrs;               // NVL buffer pointer array [NUM_MAX_NVL_PEERS]

    // --- Dispatch input data ---
    const int4* x;                    // [num_tokens, hidden_int4] input token data
    const float* x_scales;            // [num_tokens, num_scales] scales (if FP8)
    const topk_idx_t* topk_idx;       // [num_tokens, num_topk] expert indices (global)
    const float* topk_weights;        // [num_tokens, num_topk] routing weights
    const bool* is_token_in_rank;     // [num_tokens, num_ranks] routing bitmap

    // --- Dispatch metadata for RDMA ---
    const int* rdma_channel_prefix_matrix;   // Per-rank cumulative token counts per logical channel
    const int* recv_rdma_rank_prefix_sum;    // Prefix sums for forwarder
    const int* gbl_channel_prefix_matrix;    // Global NVL-level prefix matrix per logical channel
    const int* recv_gbl_rank_prefix_sum;     // Global rank prefix sums
    int* send_rdma_head;             // [num_logical_channels, num_tokens, kNumRDMARanks] for combine
    int* send_nvl_head;              // [num_logical_channels, num_rdma_recv_tokens_ub, NUM_MAX_NVL_PEERS] for combine NVL tracking
    int* recv_rdma_channel_prefix_matrix;   // Written by forwarder
    int* recv_gbl_channel_prefix_matrix;    // Written by NVL receiver
    int* recv_rdma_channel_token_count;     // [num_rdma_ranks * num_logical_channels] non-cumulative logical-channel count
    int* recv_gbl_channel_token_count;      // [num_ranks * num_logical_channels] non-cumulative logical-channel count

    // --- Dispatch dimensions ---
    int num_tokens;                   // Total tokens to dispatch
    int hidden_int4;                  // hidden_dim / 4 (int4 units)
    int num_scales;                   // Number of scales per token (0 for BF16)
    int num_topk;                     // Top-k experts per token
    int num_experts;                  // Total experts globally
    int scale_token_stride;           // Stride for x_scales
    int scale_hidden_stride;          // Stride for x_scales hidden dim

    // --- RDMA buffer sizing ---
    int num_max_rdma_chunked_send_tokens;   // Max tokens per RDMA put batch
    int num_max_rdma_chunked_recv_tokens;   // Recv buffer capacity per rank
    int num_max_nvl_chunked_send_tokens;    // Max tokens per NVL send batch
    int num_max_nvl_chunked_recv_tokens;    // NVL recv buffer capacity

    // --- Topology ---
    int rank;                         // Global rank (rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank)
    int num_ranks;                    // Total ranks

    // --- Receive-side signaling (written by Forwarder, read by Compute) ---
    int* expert_recv_count;           // [num_local_experts] continuous ready count (advanced by scheduler)
    int* expert_slot_ready;           // [num_local_experts * max_tokens_per_expert] per-slot ready flag
    int* dispatch_done;               // Flag: set to 1 when all dispatch+forward is finished
    int* dispatch_done_count;         // Atomic: how many NVL receiver warps have finished
    int expected_dispatch_done_count; // Expected number of NVL receiver warp completions

    // --- Per-expert receive storage (filled by NVL receiver) ---
    __nv_bfloat16* recv_tokens;       // [num_local_experts * max_tokens_per_expert, hidden]
    int* expert_token_offsets;        // [num_local_experts] — atomic write offset
    int* recv_token_source_info;      // [max_total_recv_tokens, 2] — (recv_token_idx, topk_slot)
    float* recv_token_route_weights;  // [max_total_recv_tokens] — route weight for this compute slot
    internode::SourceMeta* recv_src_meta; // [max_total_recv_tokens] — DeepEP SourceMeta for combine routing

    // --- Compute signaling (per-token completion gate for combine) ---
    int* token_compute_expected;        // [max_total_recv_tokens] how many local experts must compute this token
    int* token_compute_done;            // [max_total_recv_tokens] how many local experts have finished
    int* combine_token_ready;           // [max_total_recv_tokens] set to 1 when all local experts done
    float* compute_output_f;            // [max_total_recv_tokens, hidden] float accumulator for multi-expert reduce
    int* compute_group_barrier;         // [num_compute_groups] reusable global barrier counters
    int* compute_group_phase;           // [num_compute_groups] reusable global barrier phase flags
    ComputeTask* compute_tasks;         // [max_compute_tasks] dynamic compute task queue
    int max_compute_tasks;
    int* compute_task_head;             // CAS pop cursor
    int* compute_task_tail;             // single scheduler publish cursor
    int* compute_enqueue_done;          // set by scheduler after tail tasks are published
    int* expert_enqueue_cursor;         // [num_local_experts] how many slots have been enqueued
    int* compute_group_task_idx;        // [num_compute_groups] broadcast popped task idx to group SMs

    // --- Compute state ---
    int* compute_done_count;          // Atomic: how many experts have finished compute
    int* expert_compute_cursor;       // [num_local_experts] — how many tokens already computed

    // --- Expert weights ---
    const __nv_bfloat16* W_gate;      // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_up;        // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_down;      // [num_local_experts, hidden, intermediate]

    // --- S4.4 (route B2): UMMA compute TMA atoms (device-resident) ---
    // Per-expert 2D multicast TMA atoms for W_gate/W_up, and per-group A(input_buf)
    // TMA atoms. Built on host (setup_compute_tma_v7), copied to device. nullptr
    // when UMMA compute is disabled (falls back to WMMA path).
    umma::ComputeTmaAtoms* compute_tma;      // device ptr; wgate[e]/wup[e]
    umma::ComputeDownTmaAtoms* compute_down_tma;  // device ptr; wdown[e]
    umma::InputTmaAtom_t* group_input_tma;   // device array [num_compute_groups]
    int num_compute_groups;                  // for indexing group_input_tma / barriers

    // --- Compute output buffer ---
    __nv_bfloat16* compute_output;    // [max_total_recv_tokens, hidden]
    __nv_bfloat16* combine_input;     // [max_total_recv_tokens, hidden] DeepEP compact recv-token namespace
    float* combine_input_topk_weights; // [max_total_recv_tokens, num_topk] DeepEP compact recv-token namespace
    internode::SourceMeta* combine_input_src_meta; // [max_total_recv_tokens] DeepEP compact recv-token namespace
    int* combine_notify_done;         // Atomic flag: combine head metadata has been normalized
    int* combine_rdma_head_work;      // [num_combined_tokens, num_rdma_ranks] normalized combine RDMA heads
    int* combine_nvl_head_work;       // [num_tokens upper bound, NUM_MAX_NVL_PEERS] normalized combine NVL heads
    __nv_bfloat16* gemm_workspace;    // Scratch for gate/up intermediate results

    // --- Combine output ---
    float* output_accum;              // [num_tokens, hidden] float accumulator

    // --- Combine infrastructure (DeepEP combine kernel inputs) ---
    void* combine_rdma_buffer_ptr;            // Symmetric RDMA buffer for combine
    void** combine_buffer_ptrs;               // NVL buffer ptrs for combine [NUM_MAX_NVL_PEERS]
    int4* combined_x;                         // [num_combined_tokens, hidden_int4] final output
    float* combined_topk_weights;             // [num_combined_tokens, num_topk] final topk weights
    const bool* is_combined_token_in_rank;    // [num_combined_tokens, num_ranks]
    const int4* combine_x;                    // Input to combine = compute_output cast to int4*
    const float* combine_topk_weights;        // topk_weights for combine
    const int4* combine_bias_0;               // bias (nullptr for MoE)
    const int4* combine_bias_1;               // bias (nullptr for MoE)
    const int* combined_rdma_head;            // [num_logical_channels, num_combined_tokens, kNumRDMARanks]
    int* combined_nvl_head;                   // [num_logical_channels, num_rdma_recv_tokens_ub, NUM_MAX_NVL_PEERS]
    const void* combine_src_meta;             // SourceMeta array
    const int* combine_rdma_channel_prefix_matrix;
    const int* combine_rdma_rank_prefix_sum;
    const int* combine_gbl_channel_prefix_matrix;
    const int* combine_gbl_channel_token_count;      // [num_ranks * num_logical_channels] non-cumulative logical-channel count
    const int* combine_rdma_channel_token_count;     // [num_rdma_ranks * num_logical_channels] non-cumulative logical-channel count
    int combine_num_tokens;                   // num tokens for combine (= tokens received by this rank)
    int combine_num_combined_tokens;          // num combined tokens (= original dispatch num_tokens)
    int combine_rdma_head_stride;             // num_combined_tokens * kNumRDMARanks per logical channel
    int combine_nvl_head_stride;              // num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS per logical channel
    int combine_hidden;                       // hidden dim in dtype units
    int num_max_combine_rdma_chunked_send_tokens;
    int num_max_combine_rdma_chunked_recv_tokens;
    int num_max_combine_nvl_chunked_send_tokens;
    int num_max_combine_nvl_chunked_recv_tokens;
    int num_combine_sms;                      // Must be even (even/odd SM pairing)
    int num_combine_channels;                 // = num_combine_sms / 2
    int num_logical_channels;                 // Logical channel count for dispatch/compute/combine overlap

    // --- Combine signaling (per-expert completion from compute) ---
    int* expert_compute_done;                 // [num_local_experts] per-expert done flag

    // --- Compute dimensions ---
    int hidden_dim;
    int intermediate_dim;
    int num_local_experts;
    int max_tokens_per_expert;
    int max_total_recv_tokens;

    // --- SM allocation ---
    int num_dispatch_sms;
    int num_forwarder_sms;
    int num_combine_sms_total;        // Total combine SMs (alias for num_combine_sms above)
    int num_compute_sms;

    // --- Per-logical-channel overlap signaling ---
    int* channel_dispatch_done;       // [num_logical_channels] atomicAdd counter, reaches NUM_MAX_NVL_PEERS when logical channel done
    int* channel_normalized;          // [num_logical_channels] set to 1 after head normalization completes for that logical channel
    int* dispatch_channel_barrier;    // [num_logical_channels] counts dispatch SMs done with a logical channel
    int* dispatch_round_barrier;      // [num_dispatch_rounds] counts physical channels done with a dispatch round
    int* combine_channel_barrier;     // [num_logical_channels] counts combine SMs done with a logical channel

    // --- Dispatch SM config ---
    int num_dispatch_channels;        // = num_dispatch_sms / 2 (physical even/odd SM pairing)

#ifdef MK_PERF_TRACE
    // Per-logical-channel timing. Each logical channel has sender and forwarder rows
    // for both dispatch and combine so channel-level overlap is visible in Perfetto.
    // Dispatch phases: 0=enter, 1=channel_barrier_start, 2=round_barrier_start, 3=exit
    // Combine phases:  0=enter, 1=dispatch_done_acquired, 2=head_norm_done, 3=protocol_start, 4=exit
    static constexpr int MK_PERF_NUM_LCH_PHASES = 5;
    int64_t* perf_dispatch_lch_ts;     // [num_logical_channels * 2 * MK_PERF_NUM_LCH_PHASES]
    int64_t* perf_combine_lch_ts;      // [num_logical_channels * 2 * MK_PERF_NUM_LCH_PHASES]
    // Accumulated semaphore-interaction times (ns), indexed [logical_ch * 2 + role].
    // Rendered as args on existing dispatch/combine rows (no new rows added).
    int64_t* perf_disp_wait_nvl_ns;        // NVL data-wait spin in dispatch NVL receiver
    int64_t* perf_disp_publish_ns;         // publish expected/slot/source_info to compute
    int64_t* perf_disp_wait_recvcount_ns;  // expert_recv_count ordered-advance spin
    // publish breakdown (subset of perf_disp_publish_ns), all [logical_ch * 2 + role]
    int64_t* perf_disp_pub_scan_ns;        // first topk scan + write topk weights
    int64_t* perf_disp_pub_atomic_ns;      // atomicAdd(expected) + atomicAdd(expert_token_offsets)
    int64_t* perf_disp_pub_fence_ns;       // __threadfence()
    int64_t* perf_disp_pub_store_ns;       // st_release writes of source_info
    int64_t* perf_comb_wait_ready_ns;      // combine wait on combine_token_ready (compute gate)
    // Per-compute-task timing buffer (own Perfetto rows per compute group).
    // Fields 0..7  : start, end, sm_id, group_id, expert_id, batch_size, hidden, intermediate
    // Fields 8..13 : coarse phase boundary timestamps (ns) captured by group leader:
    //   8  = ts after token metadata load   (gather meta done)
    //   9  = ts after input_buf gather + sync
    //   10 = ts after gate+up GEMM + SwiGLU + sync (up phase done)
    //   11 = ts after down GEMM + sync           (down phase done)
    //   12 = ts after output write/reduce + sync  (output phase done)
    //   13 = ts after completion signaling        (== end, signal done)
    // Fields 14..21 : finer breakpoints to separate compute vs group-barrier wait
    //                 and to split the signaling phase. All captured by leader.
    //   14 = ts after up-proj GEMM body, BEFORE its group barrier  (up compute end)
    //   15 = ts after down-proj GEMM body, BEFORE its group barrier (down compute end)
    //   16 = ts after output write loop, BEFORE the first signaling barrier (output compute end)
    //   17 = ts after done-count atomic phase + sync
    //   18 = ts after fp32->bf16 finalize phase + sync
    //   19 = ts after the post-finalize threadfence_system + sync
    //   20 = ts after combine_token_ready publish + sync (== signal done)
    //   21 = reserved
    static constexpr int MK_PERF_NUM_COMPUTE_FIELDS = 22;
    int64_t* perf_compute_task;        // [max_compute_tasks * MK_PERF_NUM_COMPUTE_FIELDS]
    int* perf_compute_task_count;      // atomic write cursor into perf_compute_task
    // Root-cause diagnostics, parallel arrays indexed by the same compute-task slot.
    // Rendered as args on each compute X-event (no extra COMPUTE_FIELDS).
    // Full UMMA per-phase breakdown (ns) for up/down GEMM, so a single perf run
    // pinpoints exactly which UMMA sub-step (setup/tmem_alloc/prologue/tma_wait/
    // mma_issue/mma_wait/loop_other/cluster_sync/epilogue) dominates p3a/p4a.
    int64_t* perf_up_setup;        int64_t* perf_up_tmem_alloc;   int64_t* perf_up_prologue;
    int64_t* perf_up_tma_wait;     int64_t* perf_up_mma_issue;    int64_t* perf_up_mma_wait;
    int64_t* perf_up_loop_other;   int64_t* perf_up_cluster_sync; int64_t* perf_up_epilogue;
    int64_t* perf_down_setup;      int64_t* perf_down_tmem_alloc; int64_t* perf_down_prologue;
    int64_t* perf_down_tma_wait;   int64_t* perf_down_mma_issue;  int64_t* perf_down_mma_wait;
    int64_t* perf_down_loop_other; int64_t* perf_down_cluster_sync; int64_t* perf_down_epilogue;
    int* perf_compute_multi_expert_rows;  // [max_compute_tasks]
#endif
};

// ============================================================================
// Device GEMM using wmma (for compute phase)
// ============================================================================

using namespace nvcuda;

__device__ void device_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (transposed access)
    __nv_bfloat16* __restrict__ C,        // [M, N] row-major
    int M, int K, int N,
    int tile_warp_id, int num_tile_warps,
    int smem_warp_id,
    float* smem_buf
) {
    const int tiles_m = (M + WMMA_M - 1) / WMMA_M;
    const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
    const int total_tiles = tiles_m * tiles_n;

    for (int tile_idx = tile_warp_id; tile_idx < total_tiles; tile_idx += num_tile_warps) {
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

        float* c_buf = smem_buf + smem_warp_id * WMMA_M * WMMA_N;
        wmma::store_matrix_sync(c_buf, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        int lane_id = threadIdx.x % 32;
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int out_row = row_offset + row;
            int out_col = col_offset + col;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = __float2bfloat16(c_buf[i]);
            }
        }
    }
}

// ============================================================================
// Fused gate+up GEMM with in-register SwiGLU epilogue.
//
// Computes act = silu(gate) * up * route_weight in a single pass:
//   gate = A @ W_gate^T   (A:[M,K] row_major, W_gate:[N,K] col_major)
//   up   = A @ W_up^T     (same shapes, N = intermediate)
// For each N-tile, both gate and up accumulators stay in registers; SwiGLU is
// applied before any store. Only the activation `act` ([M,N]) is written to GMEM,
// eliminating the two GMEM round-trips for gate_buf/up_buf and the standalone
// SwiGLU read-modify-write loop.
//
// route_w[row] is the per-token route weight (already gathered by caller);
// rows >= valid_rows are written as 0 so the downstream W_down GEMM is unaffected.
// ============================================================================
__device__ void device_gemm_swiglu_fused(
    const __nv_bfloat16* __restrict__ A,       // [M, K] row_major
    const __nv_bfloat16* __restrict__ W_gate,  // [N, K] col_major (B = A@B^T)
    const __nv_bfloat16* __restrict__ W_up,    // [N, K] col_major
    __nv_bfloat16* __restrict__ act,           // [M, N] row_major output
    const float* __restrict__ route_w,         // [M] per-row route weight
    int valid_rows,                            // rows < valid_rows are real tokens
    int M, int K, int N,
    int tile_warp_id, int num_tile_warps,
    int smem_warp_id,
    float* smem_buf
) {
    const int tiles_m = (M + WMMA_M - 1) / WMMA_M;
    const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
    const int total_tiles = tiles_m * tiles_n;
    const int lane_id = threadIdx.x % 32;

    // Two separate SMEM scratch regions per warp (gate / up) to avoid races.
    float* gate_buf = smem_buf + smem_warp_id * (2 * WMMA_M * WMMA_N);
    float* up_buf   = gate_buf + WMMA_M * WMMA_N;

    for (int tile_idx = tile_warp_id; tile_idx < total_tiles; tile_idx += num_tile_warps) {
        int tile_row = tile_idx / tiles_n;
        int tile_col = tile_idx % tiles_n;
        int row_offset = tile_row * WMMA_M;
        int col_offset = tile_col * WMMA_N;

        if (row_offset >= M || col_offset >= N) continue;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bg_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bu_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cg_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cu_frag;

        wmma::fill_fragment(cg_frag, 0.0f);
        wmma::fill_fragment(cu_frag, 0.0f);

        // A tile is shared between gate and up GEMM (loaded once per K step).
        for (int k = 0; k < K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A + row_offset * K + k, K);
            wmma::load_matrix_sync(bg_frag, W_gate + col_offset * K + k, K);
            wmma::load_matrix_sync(bu_frag, W_up   + col_offset * K + k, K);
            wmma::mma_sync(cg_frag, a_frag, bg_frag, cg_frag);
            wmma::mma_sync(cu_frag, a_frag, bu_frag, cu_frag);
        }

        wmma::store_matrix_sync(gate_buf, cg_frag, WMMA_N, wmma::mem_row_major);
        wmma::store_matrix_sync(up_buf,   cu_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        // In-register (SMEM-staged) SwiGLU epilogue: silu(gate) * up * route_w.
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int out_row = row_offset + row;
            int out_col = col_offset + col;
            if (out_row >= M || out_col >= N) continue;
            if (out_row < valid_rows) {
                float g = gate_buf[i];
                float u = up_buf[i];
                float silu_g = g * (1.0f / (1.0f + __expf(-g)));
                act[out_row * N + out_col] = __float2bfloat16(silu_g * u * route_w[out_row]);
            } else {
                act[out_row * N + out_col] = __float2bfloat16(0.0f);
            }
        }
    }
}

// ============================================================================
// Dispatch Worker v2: Complete copy of DeepEP internode.cu dispatch function
// as a __device__ function. Uses even/odd SM pattern from DeepEP:
//   - Even SM (is_forwarder): kRDMAAndNVLForwarder + kForwarderCoordinator warps
//   - Odd SM (!is_forwarder): kRDMASender + kRDMASenderCoordinator + kNVLReceivers
//
// Template params instantiated for MK-v7:
//   kLowLatencyMode=false, kNumRDMARanks=2, kCachedMode=false,
//   kNumTMABytesPerWarp=16384, kNumDispatchRDMASenderWarps=7
//
// Only difference from DeepEP: In NVLReceivers section, after copying token
// data to recv_x, we also route tokens to expert storage + signal compute SMs.
// ============================================================================

// Instantiated template constants
constexpr int kNumRDMARanks = MK_NUM_RDMA_RANKS;  // 2
constexpr int kNumDispatchRDMASenderWarps = 7;
constexpr int kNumTopkRDMARanks = 2;
constexpr int kNumTMABytesPerWarp = 16384;
constexpr bool kLowLatencyMode = false;
constexpr bool kCachedMode = false;

__device__ void dispatch_worker_v2(
    int sm_id,
    int dispatch_sm_idx,  // 0-based index among all dispatch SMs
    MegaKernelState* state
) {
    using namespace internode;
    const auto num_sms = state->num_dispatch_sms;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = state->num_dispatch_channels, channel_id = sm_id / 2;
    const auto num_logical_channels = state->num_logical_channels;
    const bool is_forwarder = dispatch_sm_idx % 2 == 0;
    const auto rdma_rank = state->rank / NUM_MAX_NVL_PEERS, nvl_rank = state->rank % NUM_MAX_NVL_PEERS;
    const auto num_ranks = state->num_ranks;

    if (threadIdx.x == 0 && dispatch_sm_idx == 0) {
        printf("rank: %d, sm_id: %d, dispatch_sm_idx: %d num_channels: %d num_logical_channels: %d \n", state->rank, sm_id, dispatch_sm_idx, num_channels, num_logical_channels);
    }

    EP_DEVICE_ASSERT(num_warps >= kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);
    if (warp_id >= kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS)
        return;

    // Warp role assignment (same as DeepEP internode.cu L494-508)
    enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };
    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (is_forwarder) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
            }
        } else if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASenderCoordinator, -1};
        } else {
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
        }
    }();
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second;
    // Data dimensions
    const int hidden_int4 = state->hidden_int4;
    const int num_scales = state->num_scales;
    const int num_topk = state->num_topk;
    const int num_tokens = state->num_tokens;
    const int num_experts = state->num_experts;
    const int scale_token_stride = state->scale_token_stride;
    const int scale_hidden_stride = state->scale_hidden_stride;
    EP_DEVICE_ASSERT(num_topk <= 32);

    auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto scale_bytes = num_scales * sizeof(float);
    const int num_max_rdma_chunked_send_tokens = state->num_max_rdma_chunked_send_tokens;
    const int num_max_rdma_chunked_recv_tokens = state->num_max_rdma_chunked_recv_tokens;
    const int num_max_nvl_chunked_send_tokens = state->num_max_nvl_chunked_send_tokens;
    const int num_max_nvl_chunked_recv_tokens = state->num_max_nvl_chunked_recv_tokens;

    // Input pointers
    const int4* x = state->x;
    const float* x_scales = state->x_scales;
    const topk_idx_t* topk_idx = state->topk_idx;
    const float* topk_weights = state->topk_weights;
    const bool* is_token_in_rank = state->is_token_in_rank;
    const int* rdma_channel_prefix_matrix = state->rdma_channel_prefix_matrix;
    const int* recv_rdma_rank_prefix_sum = state->recv_rdma_rank_prefix_sum;
    const int* gbl_channel_prefix_matrix = state->gbl_channel_prefix_matrix;
    const int* recv_gbl_rank_prefix_sum = state->recv_gbl_rank_prefix_sum;
    int* send_rdma_head = state->send_rdma_head;
    int* send_nvl_head_base = state->send_nvl_head;
    int* recv_rdma_channel_prefix_matrix = state->recv_rdma_channel_prefix_matrix;
    int* recv_gbl_channel_prefix_matrix = state->recv_gbl_channel_prefix_matrix;

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    void* rdma_buffer_ptr = state->rdma_buffer_ptr;

    // NVL buffer layouts
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        // printf("enter v0 role\n");
        rs_wr_buffer_ptr = state->buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = state->buffer_ptrs[target_rank],
        rs_wr_rank = nvl_rank, ws_rr_rank = target_rank;
    if (warp_role == WarpRole::kNVLReceivers)
        // printf("enter v1 role\n");
        rs_wr_buffer_ptr = state->buffer_ptrs[target_rank], ws_rr_buffer_ptr = state->buffer_ptrs[nvl_rank],
        rs_wr_rank = target_rank, ws_rr_rank = nvl_rank;

    // RDMA sender warp synchronization
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // TMA stuffs
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
    auto tma_buffer = smem_tma_buffer + target_rank * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
    uint32_t tma_phase = 0;
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and elect_one_sync()) {
        // printf("enter v2 role\n");
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    int sender_cached_rdma_channel_head = 0, sender_global_rdma_tail_idx = 0;
    int coordinator_last_issued_tail = 0;
    int forwarder_cached_rdma_channel_head = 0, forwarder_cached_rdma_channel_tail = 0;
    int forwarder_cached_nvl_channel_head = 0, forwarder_cached_nvl_channel_tail = 0;
    int forwarder_rdma_nvl_token_idx = 0;
    int receiver_cached_channel_head_idx = 0, receiver_cached_channel_tail_idx = 0;

    for (int logical_channel_id = channel_id; logical_channel_id < num_logical_channels; logical_channel_id += num_channels) {
        auto rdma_channel_data = SymBuffer<uint8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token,
            kNumRDMARanks, logical_channel_id, num_logical_channels);
        auto rdma_channel_meta = SymBuffer<int>(
            rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2,
            kNumRDMARanks, logical_channel_id, num_logical_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(
            rdma_buffer_ptr, 1, kNumRDMARanks, logical_channel_id, num_logical_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(
            rdma_buffer_ptr, 1, kNumRDMARanks, logical_channel_id, num_logical_channels);

        auto nvl_channel_x = AsymBuffer<uint8_t>(
            ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
            NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
        auto nvl_channel_prefix_start = AsymBuffer<int>(
            ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
        auto nvl_channel_prefix_end = AsymBuffer<int>(
            ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(
            rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, ws_rr_rank)
            .advance_also(ws_rr_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(
            ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);

        sender_cached_rdma_channel_head = 0;
        sender_global_rdma_tail_idx = 0;
        coordinator_last_issued_tail = 0;
        forwarder_cached_rdma_channel_head = 0;
        forwarder_cached_rdma_channel_tail = 0;
        forwarder_cached_nvl_channel_head = 0;
        forwarder_cached_nvl_channel_tail = 0;
        forwarder_rdma_nvl_token_idx = 0;
        receiver_cached_channel_head_idx = 0;
        receiver_cached_channel_tail_idx = 0;
#ifdef MK_PERF_TRACE
        if (thread_id == 0) {
            int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
            int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
            state->perf_dispatch_lch_ts[trace_idx + 0] = globaltimer_ns();
        }
#endif
#ifdef MK_TOKEN_TRACE
        if (lane_id == 0 && logical_channel_id >= num_channels) {
            printf("[MK-DIAG][DISPATCH-LCH-ENTRY] rank=%d physical_ch=%d logical_ch=%d role=%d target=%d warp=%d sender_cached_head=%d sender_global_tail=%d coord_last_tail=%d fwd_rdma_head=%d fwd_rdma_tail=%d fwd_nvl_head=%d fwd_nvl_tail=%d fwd_nvl_token=%d recv_head=%d recv_tail=%d\n",
                   state->rank, channel_id, logical_channel_id, static_cast<int>(warp_role), target_rank, warp_id,
                   sender_cached_rdma_channel_head, sender_global_rdma_tail_idx, coordinator_last_issued_tail,
                   forwarder_cached_rdma_channel_head, forwarder_cached_rdma_channel_tail,
                   forwarder_cached_nvl_channel_head, forwarder_cached_nvl_channel_tail,
                   forwarder_rdma_nvl_token_idx, receiver_cached_channel_head_idx,
                   receiver_cached_channel_tail_idx);
        }
#endif
    // ========== kRDMASender ==========
    if (warp_role == WarpRole::kRDMASender) {
        // printf("enter v3 role\n");
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_logical_channels, logical_channel_id, token_start_idx, token_end_idx);

        // Send channel prefix metadata
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                // Prefix values keep DeepEP's cumulative token namespace for recv indexing.
                // Head arrays are channel-private, so this cumulative prefix is no longer used
                // as a shared head storage namespace.
                int prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_logical_channels + logical_channel_id;
                int prefix_start = logical_channel_id == 0 ? 0 : gbl_channel_prefix_matrix[prefix_idx - 1];
                dst_ptr[lane_id] = -prefix_start - 1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                int src_nvl_rank = lane_id - NUM_MAX_NVL_PEERS;
                int prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_logical_channels + logical_channel_id;
                dst_ptr[lane_id] = -gbl_channel_prefix_matrix[prefix_idx] - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                int prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
                int prefix_start = logical_channel_id == 0 ? 0 : rdma_channel_prefix_matrix[prefix_idx - 1];
                dst_ptr[lane_id] = -prefix_start - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                int prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[prefix_idx] - 1;
            }
            __syncwarp();

            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
            }
        }
        sync_rdma_sender_smem();

        // Iterate tokens
        int64_t token_idx;
        auto& cached_rdma_channel_head = sender_cached_rdma_channel_head;
        auto& global_rdma_tail_idx = sender_global_rdma_tail_idx;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            // printf("sender: token_idx: %ld\n", token_idx);
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS));
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            // Wait buffer release
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine in this logical channel's independent namespace.
            int* logical_send_rdma_head = send_rdma_head + logical_channel_id * num_tokens * kNumRDMARanks;
            if (lane_id < kNumRDMARanks)
                logical_send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

            // Broadcast tails
            SourceMeta src_meta;
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
                if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                    slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                    topk_ranks[num_topk_ranks] = i;
                    auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                    if (lane_id == num_topk_ranks)
                        src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
                    dst_send_buffers[num_topk_ranks++] =
                        reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;
                }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

// #ifdef MK_TOKEN_TRACE
//             if (lane_id == 0) {
//                 int gbl_expert = (int)ld_nc_global(topk_idx + token_idx * num_topk);
//                 int lcl_expert = gbl_expert - state->rank * (num_experts / num_ranks);
//                 printf("[MK-TOKEN][DISPATCH-SEND] rank=%d token=%lld num_topk_ranks=%d ch=%d rdma=%d nvl=%d expert=%d local_expert=%d topk0_w=%f h0=%f\n",
//                        state->rank, token_idx, num_topk_ranks, channel_id, rdma_rank, nvl_rank,
//                        gbl_expert, lcl_expert,
//                        ld_nc_global(topk_weights + token_idx * num_topk + 0),
//                        __bfloat162float(reinterpret_cast<const nv_bfloat16*>(x + token_idx * hidden_int4)[0]));
//             }
// #endif

            // Copy x
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

            // Copy x_scales
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                auto value = ld_nc_global(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

            // Copy source metadata
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy topk_idx and topk_weights
            #pragma unroll
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release transaction window
            if (is_token_in_rank_uint64 != 0) {
                acquire_lock(rdma_send_channel_lock + lane_id);
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;
                }
                rdma_send_channel_window[lane_id] = window;
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }

    // ========== kRDMASenderCoordinator ==========
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // printf("enter v4 role\n");
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

        // Each logical channel owns an independent RDMA queue, so reset CTA-local
        // sender state before issuing this channel's chunks.
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;
        sync_rdma_sender_smem();

        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_logical_channels + logical_channel_id];
            if (logical_channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_logical_channels + logical_channel_id - 1];
        }

        // printf("RdmaSenderCoordinator: num_tokens_to_send: %lld\n", num_tokens_to_send);

        auto& last_issued_tail = coordinator_last_issued_tail;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("MK RDMA coordinator timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, tail: %d, remaining: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, last_issued_tail, num_tokens_to_send);
                trap();
            }

            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                auto processed_tail =
                    __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;

                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
                    const auto dst_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    const auto src_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                      translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                      channel_id, lane_id, 0);
                } else {
                    memory_fence();
                }
                __syncwarp();

                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                    num_tokens_to_issue,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                    channel_id,
                                                    dst_rdma_rank == rdma_rank);
                }
                __syncwarp();
            }
        }

    // ========== kRDMAAndNVLForwarder ==========
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // printf("enter v5 role\n");
        const auto dst_nvl_rank = target_rank;

        // Wait counters to arrive
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    // if (blockIdx.x == 16 && lane_id == 0) {
                    // printf("lane_id: %d, src_rdma_channel_prefix: %d, src_rdma_channel_prefix_1: %d, num_tokens_to_recv_from_rdma: %d num_channels: %d channel_id: %d\n", lane_id, src_rdma_channel_prefix, src_rdma_channel_prefix_1, num_tokens_to_recv_from_rdma, num_channels, channel_id);
                    // }
                    recv_rdma_channel_prefix_matrix[lane_id * num_logical_channels + logical_channel_id] = src_rdma_channel_prefix_1;
                    // Save per-logical-channel token count (non-cumulative) for diagnostics and bounds checks.
                    state->recv_rdma_channel_token_count[lane_id * num_logical_channels + logical_channel_id] = num_tokens_to_recv_from_rdma;
                    __threadfence_system();  // Ensure prefix_matrix and token_count visible before NVL Receiver signals done
                    // Match original DeepEP's combine-head namespace inside this logical channel:
                    // rank prefix + cumulative channel prefix, with the outer allocation already sliced by logical_channel_id.
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch forwarder timeout (RDMA meta), rank=%d block=%d thread=%d channel=%d logical_ch=%d RDMA=%d nvl=%d src RDMA lane=%d dst NVL=%d meta=(%d,%d,%d,%d)\n",
                           state->rank, static_cast<int>(blockIdx.x), thread_id, channel_id, logical_channel_id,
                           rdma_rank, nvl_rank, lane_id, dst_nvl_rank,
                           meta_0, meta_1, meta_2, meta_3);
                    trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head inside this logical channel's independent NVL head namespace.
        int* send_nvl_head = send_nvl_head_base + logical_channel_id * state->combine_nvl_head_stride +
            src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        int src_rdma_rank = dispatch_sm_idx % kNumRDMARanks;
        auto& cached_rdma_channel_head = forwarder_cached_rdma_channel_head;
        auto& cached_rdma_channel_tail = forwarder_cached_rdma_channel_tail;
        auto& cached_nvl_channel_head = forwarder_cached_nvl_channel_head;
        auto& cached_nvl_channel_tail = forwarder_cached_nvl_channel_tail;
        auto& rdma_nvl_token_idx = forwarder_rdma_nvl_token_idx;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Check NVL destination queue
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;
                cached_nvl_channel_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank,
                           ld_volatile_global(nvl_channel_head.buffer()), cached_nvl_channel_tail);
                    trap();
                }
            }

            // Find next source RDMA rank
            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("MK dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id);
                    trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            // Iterate tokens
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);

                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    int rdma_nvl_token_idx_before = rdma_nvl_token_idx;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    int* nvl_head_ptr = send_nvl_head + i * NUM_MAX_NVL_PEERS;
                    *nvl_head_ptr = cached_head;

                }
                if (not is_in_dst_nvl_rank)
                    continue;

                // Get empty slot
                int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                // TMA copy
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, num_bytes_per_token);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (elect_one_sync())
                    tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token);
                __syncwarp();

                if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;

                tma_store_wait<0>();
                __syncwarp();
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            // Move tail index
            __syncwarp();
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        // Retired
        __syncwarp();
        if (elect_one_sync()) {
            forward_channel_retired[dst_nvl_rank] = true;
        }

    // ========== kForwarderCoordinator ==========
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        // printf("enter v6 role\n");

        if (target_rank == 0) {
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
        if (lane_id < NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;
        sync_forwarder_smem();

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        while (true) {
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                if (not forward_channel_retired[i])
                    min_head = min(min_head, forward_channel_head[i][target_rdma]);
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;

            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and
                lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                min_head - last_head,
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank),
                                                channel_id + num_channels,
                                                lane_id == rdma_rank);
                last_head = min_head;
            }

            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
        }

    // ========== kNVLReceivers ==========
    } else {
        // printf("enter v7 role\n");
        int src_nvl_rank = target_rank, total_offset = 0;
        const int local_expert_begin = state->rank * (num_experts / num_ranks);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                total_offset += start_offset;
                break;
            }

            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                int raw_start_now = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
                int raw_end_now = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
                printf("MK dispatch NVL receiver timeout, rank=%d block=%d thread=%d channel=%d logical_ch=%d RDMA=%d nvl=%d src RDMA=%d src nvl=%d prefix=(%d,%d) cached_head=%d cached_tail=%d prefix_ptr=(%p,%p) raw_now=(%d,%d) target=%d warp=%d rs_wr=%d ws_rr=%d\n",
                       state->rank, static_cast<int>(blockIdx.x), thread_id, channel_id, logical_channel_id,
                       rdma_rank, nvl_rank, lane_id, src_nvl_rank, start_offset, end_offset,
                       receiver_cached_channel_head_idx, receiver_cached_channel_tail_idx,
                       nvl_channel_prefix_start.buffer() + lane_id,
                       nvl_channel_prefix_end.buffer() + lane_id,
                       raw_start_now, raw_end_now,
                       target_rank, warp_id, rs_wr_rank, ws_rr_rank);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage
        if (lane_id < kNumRDMARanks) {
            int idx = (lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_logical_channels + logical_channel_id;
            recv_gbl_channel_prefix_matrix[idx] = total_offset;
        }
        // Save per-channel token count (non-cumulative) for combine NVL Sender
        if (lane_id < kNumRDMARanks) {
            int idx = (lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_logical_channels + logical_channel_id;
            int count = end_offset - start_offset;
            state->recv_gbl_channel_token_count[idx] = count;
        }
        __syncwarp();

        // printf("NVLReceivers: num_tokens_to_recv: %d\n", num_tokens_to_recv);

        auto& cached_channel_head_idx = receiver_cached_channel_head_idx;
        auto& cached_channel_tail_idx = receiver_cached_channel_tail_idx;
        while (num_tokens_to_recv > 0) {
            // Wait for data
            start_time = clock64();
#ifdef MK_PERF_TRACE
            int64_t wait_nvl_start = (lane_id == 0) ? globaltimer_ns() : 0;
#endif
            while (true) {
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                cached_channel_tail_idx = __shfl_sync(0xffffffff, ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch NVL receiver timeout (data), channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank, cached_channel_head_idx, cached_channel_tail_idx);
                    trap();
                }
            }
#ifdef MK_PERF_TRACE
            // Only one NVL receiver warp (target_rank==0) records timing, so the
            // accumulated value reflects a single execution stream's real occupancy
            // and is directly comparable to dispatch_sender_work wall time.
            if (lane_id == 0 && target_rank == 0) {
                int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
                int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
                state->perf_disp_wait_nvl_ns[acc_idx] += globaltimer_ns() - wait_nvl_start;
            }
#endif

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
                (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;

                // TMA copy to recv_x (DeepEP standard path)
                bool scale_aligned = (scale_bytes % 16 == 0);
                auto tma_load_bytes = hidden_bytes + (scale_aligned ? scale_bytes : 0);

                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, tma_load_bytes);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, tma_load_bytes);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);

                // === MK-v7: Write directly to combine input namespace (no expert-local layout) ===
                auto topk_data_ptr = reinterpret_cast<int*>(shifted + hidden_bytes + scale_bytes + sizeof(SourceMeta));
                auto weight_data_ptr = reinterpret_cast<float*>(topk_data_ptr + num_topk);
                auto* src_data = reinterpret_cast<const __nv_bfloat16*>(tma_buffer);

                // Copy token data to DeepEP compact combine-input namespace.
                for (int h = lane_id; h < state->hidden_dim; h += 32)
                    state->combine_input[recv_token_idx * state->hidden_dim + h] = src_data[h];


                // Fill topk weights/src_meta, publish this token's local expert count, then
                // publish expert slots. expected must be final before any expert_recv_count
                // advances, otherwise compute can mark multi-local-expert tokens ready early.
                const int local_expert_end = local_expert_begin + state->num_local_experts;
                if (lane_id == 0) {
#ifdef MK_PERF_TRACE
                    int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
                    int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
                    bool record_perf = (target_rank == 0);  // only one receiver warp records
                    int64_t publish_start = globaltimer_ns();
                    int64_t wait_recvcount_acc = 0;
#endif
                    int local_hits = 0;
                    for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
                        int expert_id = ld_nc_global(topk_data_ptr + topk_slot);
                        if (expert_id < local_expert_begin || expert_id >= local_expert_end)
                            continue;
                        float route_w = ld_nc_global(weight_data_ptr + topk_slot);
                        state->combine_input_topk_weights[recv_token_idx * num_topk + topk_slot] = route_w;
                        local_hits += 1;
                    }
                    if (local_hits > 0) {
                        state->combine_input_src_meta[recv_token_idx] = meta;
                        atomicAdd(&state->token_compute_expected[recv_token_idx], local_hits);
                    }

                    // Pass 1: allocate slots and write source_info for all local hits.
                    int hit_local_expert[32];
                    int hit_slot[32];
                    int num_hits = 0;
                    for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
                        int expert_id = ld_nc_global(topk_data_ptr + topk_slot);
                        if (expert_id < local_expert_begin || expert_id >= local_expert_end)
                            continue;
                        int local_expert_id = expert_id - local_expert_begin;
                        int slot = atomicAdd(&state->expert_token_offsets[local_expert_id], 1);
                        if (slot >= state->max_tokens_per_expert)
                            continue;  // overflow guard: drop tokens beyond capacity
                        int dest_offset = local_expert_id * state->max_tokens_per_expert + slot;
                        int* dst_ptr = &state->recv_token_source_info[dest_offset * 2];
                        // Plain stores: ordering vs slot_ready is enforced by the single
                        // __threadfence() below. All readers are this GPU's compute workers,
                        // so device-scope visibility is sufficient.
                        st_na_global(dst_ptr, static_cast<int>(recv_token_idx));
                        st_na_global(dst_ptr + 1, topk_slot);
                        hit_local_expert[num_hits] = local_expert_id;
                        hit_slot[num_hits] = slot;
                        num_hits += 1;
                    }

                    // Single device-scope fence per token: orders the plain source_info
                    // stores before the slot_ready release stores. The whole signal chain
                    // (slot_ready -> scheduler -> expert_recv_count -> compute) stays in
                    // device scope because no remote GPU reads these buffers.
                    if (num_hits > 0)
                        __threadfence();

                    // Pass 2: mark each slot ready (unordered, no spin-wait). Scheduler
                    // scans the bitmap and advances expert_recv_count.
                    for (int h = 0; h < num_hits; ++h) {
                        int local_expert_id = hit_local_expert[h];
                        int slot = hit_slot[h];
                        st_na_release(&state->expert_slot_ready[local_expert_id * state->max_tokens_per_expert + slot], 1);
                    }
#ifdef MK_PERF_TRACE
                    int64_t publish_total = globaltimer_ns() - publish_start;
                    if (record_perf) {
                        state->perf_disp_wait_recvcount_ns[acc_idx] += wait_recvcount_acc;
                        state->perf_disp_publish_ns[acc_idx] += publish_total - wait_recvcount_acc;
                    }
#endif
                }
                __syncwarp();

                // Wait TMA to be finished
                tma_store_wait<0>();
                __syncwarp();

            }

            // Move queue
            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }

        // Signal dispatch done after all NVL receiver warps on this rank have finished.
        // This is the correct location: NVL Receiver writes gbl_channel_prefix_matrix and
        // gbl_channel_token_count (L929-934), so it must be the one to signal channel_dispatch_done.
        __syncwarp();
        if (lane_id == 0) {
            __threadfence_system();  // Ensure prefix/count writes visible before signaling
            atomicAdd(&state->channel_dispatch_done[logical_channel_id], 1);
            int done_count = atomicAdd(state->dispatch_done_count, 1) + 1;
            if (done_count == state->expected_dispatch_done_count)
                st_release_sys_global(state->dispatch_done, 1);
        }
    }

    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
#ifdef MK_PERF_TRACE
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        state->perf_dispatch_lch_ts[trace_idx + 1] = globaltimer_ns();
    }
#endif
    if (thread_id == 0)
        atomicAdd(&state->dispatch_channel_barrier[logical_channel_id], 1);
    if (thread_id == 0) {
        auto start_time = clock64();
        while (ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]) < 2) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch logical-channel barrier timeout, physical_ch=%d logical_ch=%d count=%d\n",
                       channel_id, logical_channel_id, ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]));
                trap();
            }
            __nanosleep(32);
        }
    }
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
#ifdef MK_PERF_TRACE
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        state->perf_dispatch_lch_ts[trace_idx + 2] = globaltimer_ns();
    }
#endif

    if (thread_id == 0 && dispatch_sm_idx % 2 == 0) {
        const int round_idx = logical_channel_id / num_channels;
        __threadfence_system();
        atomicAdd(&state->dispatch_round_barrier[round_idx], 1);
    }
    if (thread_id == 0) {
        const int round_idx = logical_channel_id / num_channels;
        auto start_time = clock64();
        while (ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]) < num_channels) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch round barrier timeout, physical_ch=%d logical_ch=%d round=%d count=%d need=%d\n",
                       channel_id, logical_channel_id, round_idx,
                       ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]), num_channels);
                trap();
            }
            __nanosleep(32);
        }
    }
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
#ifdef MK_PERF_TRACE
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        state->perf_dispatch_lch_ts[trace_idx + 3] = globaltimer_ns();
    }
#endif

    }
}

// ============================================================================
// Compute Scheduler + Worker: scheduler enqueues expert batches, compute groups run GEMM+SwiGLU
// ============================================================================

__device__ __forceinline__ void compute_group_sync(MegaKernelState* state, int group_id, int group_size) {
    __syncthreads();
    if (threadIdx.x == 0) {
        int phase = ld_acquire_sys_global(&state->compute_group_phase[group_id]);
        int arrived = atomicAdd(&state->compute_group_barrier[group_id], 1) + 1;
        if (arrived == group_size) {
            st_release_sys_global(&state->compute_group_barrier[group_id], 0);
            st_release_sys_global(&state->compute_group_phase[group_id], phase + 1);
        } else {
            while (ld_acquire_sys_global(&state->compute_group_phase[group_id]) == phase)
                __nanosleep(64);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ bool timeout_log_once(MegaKernelState* state, int site_id) {
    if (site_id < 0 || site_id >= kTimeoutLogCount)
        return false;
    if (state->timeout_log_counters == nullptr)
        return false;
    int ticket = atomicAdd(&state->timeout_log_counters[site_id], 1);
    return ticket < MK_TIMEOUT_LOG_BUDGET;
}

__device__ __forceinline__ void scheduler_publish_task(MegaKernelState* state, int expert_id, int start_slot, int num_tokens) {
    int tail = ld_acquire_sys_global(state->compute_task_tail);
    if (tail >= state->max_compute_tasks) {
        printf("MK compute task queue overflow, rank=%d tail=%d max=%d\n", state->rank, tail, state->max_compute_tasks);
        trap();
    }
    state->compute_tasks[tail] = ComputeTask{expert_id, start_slot, num_tokens};
    __threadfence_system();
    st_release_sys_global(state->compute_task_tail, tail + 1);
}

__device__ void compute_scheduler_worker(MegaKernelState* state) {
    if (threadIdx.x != 0)
        return;

    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;
    bool tail_enqueued = false;

    while (true) {
        // Scan per-slot ready bitmap and advance expert_recv_count for each expert.
        for (int expert_id = 0; expert_id < num_local_experts; ++expert_id) {
            int old_count = ld_acquire_global(&state->expert_recv_count[expert_id]);
            int count = old_count;
            while (count < max_tpe &&
                   ld_acquire_global(&state->expert_slot_ready[expert_id * max_tpe + count]) == 1) {
                ++count;
            }
            if (count != old_count) {
                // Fence: source_info stores (made visible to us via slot_ready acquire)
                // must be visible to compute workers that read them after observing the
                // expert_recv_count advance. Device scope suffices (no remote reader).
                __threadfence();
                st_na_release(&state->expert_recv_count[expert_id], count);
            }

            // Enqueue full batches.
            int cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
            while (count - cursor >= COMPUTE_BATCH_SIZE) {
                scheduler_publish_task(state, expert_id, cursor, COMPUTE_BATCH_SIZE);
                cursor += COMPUTE_BATCH_SIZE;
                st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
            }
        }

        int dispatch_done_count = ld_acquire_sys_global(state->dispatch_done_count);
        bool dispatch_done = (dispatch_done_count == state->expected_dispatch_done_count);
        if (dispatch_done && !tail_enqueued) {
            // Final scan after dispatch is done to catch the last (< BATCH) slots.
            for (int expert_id = 0; expert_id < num_local_experts; ++expert_id) {
                int count = ld_acquire_global(&state->expert_recv_count[expert_id]);
                while (count < max_tpe &&
                       ld_acquire_global(&state->expert_slot_ready[expert_id * max_tpe + count]) == 1) {
                    ++count;
                }
                __threadfence();
                st_na_release(&state->expert_recv_count[expert_id], count);

                int cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
                // Enqueue full batches first, then a final (< BATCH) tail batch.
                // Every published task must have num_tokens <= COMPUTE_BATCH_SIZE,
                // otherwise compute_worker overruns its [COMPUTE_BATCH_SIZE] shared
                // arrays (s_recv_token_idx/...) and triggers an illegal memory access.
                while (count - cursor >= COMPUTE_BATCH_SIZE) {
                    scheduler_publish_task(state, expert_id, cursor, COMPUTE_BATCH_SIZE);
                    cursor += COMPUTE_BATCH_SIZE;
                    st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
                }
                if (count > cursor) {
                    scheduler_publish_task(state, expert_id, cursor, count - cursor);
                    st_na_release(&state->expert_enqueue_cursor[expert_id], count);
                }
            }
            __threadfence_system();
            st_release_sys_global(state->compute_enqueue_done, 1);
            tail_enqueued = true;
        }

        if (tail_enqueued)
            break;

        __nanosleep(64);
    }
}

__device__ void compute_worker(
    int sm_id,
    int compute_sm_idx,  // 0-based index among compute SMs
    int num_compute_sms,
    MegaKernelState* state,
    float* smem_wmma_buf
) {
    const int thread_id = threadIdx.x;
    const int local_warp_id = thread_id / 32;
    const int group_sm_idx = compute_sm_idx % COMPUTE_GROUP_SIZE;
    const int group_id = compute_sm_idx / COMPUTE_GROUP_SIZE;
    const int num_compute_groups = num_compute_sms / COMPUTE_GROUP_SIZE;
    if (num_compute_groups == 0 || group_id >= num_compute_groups)
        return;
    const int num_warps_per_sm = blockDim.x / 32;
    const int group_warp_id = group_sm_idx * num_warps_per_sm + local_warp_id;
    const int group_num_warps = COMPUTE_GROUP_SIZE * num_warps_per_sm;
    const int group_thread_id = group_sm_idx * blockDim.x + thread_id;
    const int group_num_threads = COMPUTE_GROUP_SIZE * blockDim.x;
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;
    const int hidden = state->hidden_dim;
    const int intermediate = state->intermediate_dim;
    const int num_topk = state->num_topk;

    // Per-SM global-memory workspace for batched GEMM intermediates.
    // Full batches use M=128. Tail batches use the same path with rows [batch_size,128)
    // zero-filled so WMMA M tiles never read past valid token rows.
    const int padded_m = COMPUTE_BATCH_SIZE;
    const int input_stride = padded_m * hidden;
    const int gate_stride = padded_m * intermediate;
    const int up_stride = padded_m * intermediate;
    const int down_stride = padded_m * hidden;
    const int gemm_stride = input_stride + gate_stride + up_stride + down_stride;
    __nv_bfloat16* input_buf = state->gemm_workspace + group_id * gemm_stride;
    __nv_bfloat16* gate_buf = input_buf + input_stride;   // unused after gate+up SwiGLU fusion (kept for workspace layout)
    __nv_bfloat16* up_buf   = gate_buf + gate_stride;     // holds fused activation = silu(gate)*up*route_w
    __nv_bfloat16* down_buf = up_buf + up_stride;
    (void)gate_buf;

    __shared__ int s_recv_token_idx[COMPUTE_BATCH_SIZE];
    __shared__ int s_topk_slot[COMPUTE_BATCH_SIZE];
    __shared__ int s_expected[COMPUTE_BATCH_SIZE];
    __shared__ int s_is_last[COMPUTE_BATCH_SIZE];
    // Per-row route weight, gathered once and consumed inside the fused
    // gate+up SwiGLU epilogue (act = silu(gate) * up * route_w).
    __shared__ float s_route_w[COMPUTE_BATCH_SIZE];

    // TMEM alloc-once flag: first UMMA call allocates, subsequent calls reuse.
    bool umma_tmem_allocated = false;

    while (true) {
        if (group_sm_idx == 0 && thread_id == 0) {
            int task_idx = -1;
            while (true) {
                int head = ld_acquire_sys_global(state->compute_task_head);
                int tail = ld_acquire_sys_global(state->compute_task_tail);
                if (head >= tail) {
                    if (ld_acquire_sys_global(state->compute_enqueue_done))
                        task_idx = -2;
                    break;
                }
                if (atomicCAS(state->compute_task_head, head, head + 1) == head) {
                    task_idx = head;
                    break;
                }
            }
            st_release_sys_global(&state->compute_group_task_idx[group_id], task_idx);
        }
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);

        int task_idx = ld_acquire_sys_global(&state->compute_group_task_idx[group_id]);
        if (task_idx == -2) {
            // Dealloc TMEM before exiting the persistent loop (if we ever allocated).
            if (umma_tmem_allocated) {
                char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
                umma::umma_dealloc(cluster_smem);
            }
            break;
        }
        if (task_idx < 0) {
            if (group_sm_idx == 0 && thread_id == 0)
                __nanosleep(128);
            compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
            continue;
        }

        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
        ComputeTask task = state->compute_tasks[task_idx];
        int expert_id = task.expert_id;
        int start_slot = task.start_slot;
        int batch_size = task.num_tokens;
#ifdef MK_PERF_TRACE
        const bool perf_leader = (group_sm_idx == 0 && thread_id == 0);
        int64_t compute_task_start_ns = perf_leader ? globaltimer_ns() : 0;
        int64_t perf_ph_meta_ns = 0, perf_ph_input_ns = 0, perf_ph_upgemm_ns = 0;
        int64_t perf_ph_downgemm_ns = 0, perf_ph_output_ns = 0, perf_ph_signal_ns = 0;
        // Finer breakpoints: GEMM body end (before barrier) and signaling sub-phases.
        int64_t perf_up_body_ns = 0, perf_down_body_ns = 0, perf_out_body_ns = 0;
        int64_t perf_sig_donecount_ns = 0, perf_sig_finalize_ns = 0;
        int64_t perf_sig_fence_ns = 0, perf_sig_publish_ns = 0;
        // Root-cause diagnostics (rendered as args, no new COMPUTE_FIELDS):
        //   full UMMA per-phase breakdown for up/down GEMM (leader thread only),
        //   plus multi-expert finalize row count.
        __shared__ umma::UmmaPerf s_perf_up, s_perf_down;
        __shared__ int s_perf_multi_expert_rows;
        if (perf_leader) {
            s_perf_up = umma::UmmaPerf{};
            s_perf_down = umma::UmmaPerf{};
            s_perf_multi_expert_rows = 0;
        }
        __syncthreads();
#endif

        for (int i = thread_id; i < batch_size; i += blockDim.x) {
            int base_offset = expert_id * max_tpe + start_slot + i;
            s_recv_token_idx[i] = ld_acquire_global(&state->recv_token_source_info[base_offset * 2]);
            s_topk_slot[i] = ld_acquire_global(&state->recv_token_source_info[base_offset * 2 + 1]);
            s_expected[i] = ld_acquire_global(&state->token_compute_expected[s_recv_token_idx[i]]);
            s_route_w[i] = ld_nc_global(&state->combine_input_topk_weights[s_recv_token_idx[i] * num_topk + s_topk_slot[i]]);
#ifdef MK_TOKEN_TRACE
            printf("[MK-TOKEN][COMPUTE] rank=%d sm=%d expert=%d row=%d recv_token=%d topk_slot=%d\n",
                   state->rank, sm_id, expert_id, i, s_recv_token_idx[i], s_topk_slot[i]);
#endif
        }
        __syncthreads();
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_meta_ns = globaltimer_ns();
#endif

        for (int idx = group_thread_id; idx < input_stride; idx += group_num_threads) {
            int row = idx / hidden;
            int h = idx - row * hidden;
            input_buf[idx] = (row < batch_size)
                ? state->combine_input[(int64_t)s_recv_token_idx[row] * hidden + h]
                : __float2bfloat16(0.0f);
        }
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_input_ns = globaltimer_ns();
#endif

        // Expert weight slices
        const __nv_bfloat16* w_gate = &state->W_gate[expert_id * intermediate * hidden];
        const __nv_bfloat16* w_up   = &state->W_up[expert_id * intermediate * hidden];
        const __nv_bfloat16* w_down = &state->W_down[expert_id * hidden * intermediate];

        // DeepGEMM-style gate/up path from compute_ref/umma_swiglu_ws_dg.cu:
        // first materialize gate BF16, then run up GEMM with a SwiGLU epilogue
        // that reads gate_buf. This preserves the verified BF16 precision path.
        // umma_accum_iter tracks TMEM accumulator pipeline phase (tmem_full/tmem_empty
        // barrier ring) across ALL three GEMMs (gate, up, down). Must NOT be reset
        // between gate/up and down-proj — the barrier ring is initialized once and
        // must stay in phase. Declared here so it spans both if-blocks below.
        uint32_t umma_accum_iter = 0;

        // MK_FORCE_WMMA: step-(1) isolation — force gate/up + down to the trusted
        // WMMA path (device_gemm_swiglu_fused / device_gemm_bf16), bypassing the
        // DeepGEMM UMMA path entirely. If end-to-end PASSes under this flag, the
        // data preparation (input_buf gather / W layout / route_w) is correct and
        // any precision bug is isolated to the UMMA outer layer (descriptors /
        // tile schedule / cluster mapping).
#ifdef MK_FORCE_WMMA
        constexpr bool kUseUmmaGateUp = false;
#else
        constexpr bool kUseUmmaGateUp = true;
#endif
        // MK_UMMA_SINGLE_CLUSTER: step-(2) isolation — only ONE 2-CTA cluster
        // (group_sm_idx 0/1) runs ALL tiles (num_clusters=1), reproducing the
        // standalone PASS config inside the real megakernel. The other 15 clusters
        // skip the UMMA work. This verifies the 2-CTA DeepGEMM gate/up is correct
        // on real megakernel data, isolated from the 16-cluster tile sharing.
        // down-proj falls back to WMMA in this mode (step 2 isolates gate/up only).
#ifdef MK_UMMA_SINGLE_CLUSTER
        constexpr bool kUmmaSingleCluster = true;
#else
        constexpr bool kUmmaSingleCluster = false;
#endif
        if (kUseUmmaGateUp && state->compute_tma != nullptr && batch_size == COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / 2;   // 0..15 2-CTA clusters
            // step-2: single cluster (#0) does all tiles; step-3/default: 16 clusters share.
            const int num_clusters = kUmmaSingleCluster ? 1 : (COMPUTE_GROUP_SIZE / 2);
            const bool cluster_active = kUmmaSingleCluster ? (cluster_in_group == 0) : true;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];

            if (cluster_active) {
                // Route-2 persistent: init barriers+TMEM once, run gate+up persistent
                // (tile loop INSIDE the three warp roles, zero cluster sync between
                // tiles), then dealloc once. accum_iter threads through gate->up.
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma::umma_up_swiglu_persistent(
                    &in_atom.a, &in_atom.gate_cd, &in_atom.act_cd,
                    &state->compute_tma->wgate[expert_id],
                    &state->compute_tma->wup[expert_id],
                    gate_buf, s_route_w,
                    batch_size, intermediate, hidden,
                    cluster_in_group, num_clusters,
                    cluster_smem, umma_accum_iter);
                umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = false;   // freed each task (4a)
            }
#ifdef MK_PERF_TRACE
            if (perf_leader) perf_up_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
        } else {
            device_gemm_swiglu_fused(input_buf, w_gate, w_up, up_buf, s_route_w,
                                     batch_size, batch_size, hidden, intermediate,
                                     group_warp_id, group_num_warps, local_warp_id, smem_wmma_buf);
#ifdef MK_PERF_TRACE
            if (perf_leader) perf_up_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
        }
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_upgemm_ns = globaltimer_ns();
#endif

        // GEMM 3: down-proj D = act @ W_down^T.
        // NOTE: gate/up are now task-level PERSISTENT (init/alloc TMEM once, free
        // once per task). The old per-tile UMMA down path (umma_down_proj_tile_dg
        // via dg_gemm_tile + umma_tmem_allocated) is incompatible with that
        // lifecycle, so down-proj uses the WMMA fallback for now (down persistent
        // is a later step). Forced off by `false &&`.
        if (false && kUseUmmaGateUp && !kUmmaSingleCluster &&
            state->compute_down_tma != nullptr && batch_size == COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / 2;
            const int m_tiles = (batch_size + umma::kDgBlockM - 1) / umma::kDgBlockM;
            const int n_tiles = hidden / umma::kDgBlockN;
            const int total_tiles = m_tiles * n_tiles;
            const int num_clusters = COMPUTE_GROUP_SIZE / 2;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];
            // Reuse umma_accum_iter (declared above) — must continue from wherever
            // gate+up passes left off so tmem_full/tmem_empty barrier ring stays in phase.

            for (int tile = cluster_in_group; tile < total_tiles; tile += num_clusters) {
                umma::umma_down_proj_tile_dg(
                    &in_atom.act_a,
                    &state->compute_down_tma->wdown[expert_id],
                    &in_atom.down_cd,
                    tile, batch_size, hidden, intermediate,
                    cluster_smem, umma_tmem_allocated, umma_accum_iter);
            }
#ifdef MK_PERF_TRACE
            if (perf_leader) perf_down_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
        } else {
            device_gemm_bf16(up_buf, w_down, down_buf, batch_size, intermediate, hidden,
                              group_warp_id, group_num_warps, local_warp_id, smem_wmma_buf);
#ifdef MK_PERF_TRACE
            if (perf_leader) perf_down_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
        }
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_downgemm_ns = globaltimer_ns();
#endif

        // expected==1 can store directly; multi-local-expert tokens reduce through float atomics.
        for (int idx = group_thread_id; idx < batch_size * hidden; idx += group_num_threads) {
            int row = idx / hidden;
            int h = idx - row * hidden;
            int recv_token_idx = s_recv_token_idx[row];
            if (s_expected[row] == 1) {
                state->compute_output[(int64_t)recv_token_idx * hidden + h] = down_buf[idx];
            } else {
                atomicAdd(&state->compute_output_f[(int64_t)recv_token_idx * hidden + h],
                          __bfloat162float(down_buf[idx]));
            }
        }
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_out_body_ns = globaltimer_ns();
#endif
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
        __threadfence_system();  // ensure output/reduce writes are visible before done counters
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_output_ns = globaltimer_ns();
#endif

        // Per-token completion signal. One SM per group owns counters/ready publication.
        if (group_sm_idx == 0 && thread_id < batch_size) {
            int recv_token_idx = s_recv_token_idx[thread_id];
            if (s_expected[thread_id] == 1) {
                s_is_last[thread_id] = 1;
            } else {
                int done_cnt = atomicAdd(&state->token_compute_done[recv_token_idx], 1) + 1;
                s_is_last[thread_id] = (done_cnt == s_expected[thread_id]);
            }
        }
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) {
            perf_sig_donecount_ns = globaltimer_ns();
            // Count rows that actually do the fp32->bf16 finalize (multi-local-expert tokens).
            int multi_rows = 0;
            for (int row = 0; row < batch_size; ++row)
                if (s_is_last[row] && s_expected[row] > 1) ++multi_rows;
            s_perf_multi_expert_rows = multi_rows;
        }
#endif

        if (group_sm_idx == 0) {
            for (int idx = thread_id; idx < batch_size * hidden; idx += blockDim.x) {
                int row = idx / hidden;
                int h = idx - row * hidden;
                if (s_is_last[row] && s_expected[row] > 1) {
                    int recv_token_idx = s_recv_token_idx[row];
                    state->compute_output[(int64_t)recv_token_idx * hidden + h] =
                        __float2bfloat16(state->compute_output_f[(int64_t)recv_token_idx * hidden + h]);
                }
            }
        }
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_sig_finalize_ns = globaltimer_ns();
#endif
        __threadfence_system();
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_sig_fence_ns = globaltimer_ns();
#endif

        if (group_sm_idx == 0 && thread_id < batch_size && s_is_last[thread_id]) {
            st_release_sys_global(&state->combine_token_ready[s_recv_token_idx[thread_id]], 1);
        }
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_sig_publish_ns = globaltimer_ns();
#endif

#ifdef MK_PERF_TRACE
        if (perf_leader) {
            perf_ph_signal_ns = perf_sig_publish_ns;
            int slot = atomicAdd(state->perf_compute_task_count, 1);
            if (slot < state->max_compute_tasks) {
                int64_t* rec = state->perf_compute_task + (int64_t)slot * MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS;
                rec[0] = compute_task_start_ns;
                rec[1] = perf_ph_signal_ns;
                rec[2] = sm_id;
                rec[3] = group_id;
                rec[4] = expert_id;
                rec[5] = batch_size;
                rec[6] = hidden;
                rec[7] = intermediate;
                rec[8]  = perf_ph_meta_ns;
                rec[9]  = perf_ph_input_ns;
                rec[10] = perf_ph_upgemm_ns;
                rec[11] = perf_ph_downgemm_ns;
                rec[12] = perf_ph_output_ns;
                rec[13] = perf_ph_signal_ns;
                rec[14] = perf_up_body_ns;
                rec[15] = perf_down_body_ns;
                rec[16] = perf_out_body_ns;
                rec[17] = perf_sig_donecount_ns;
                rec[18] = perf_sig_finalize_ns;
                rec[19] = perf_sig_fence_ns;
                rec[20] = perf_sig_publish_ns;
                rec[21] = 0;
                // Root-cause diagnostics (parallel arrays, same slot): full UMMA breakdown.
                state->perf_up_setup[slot]        = s_perf_up.setup_ns;
                state->perf_up_tmem_alloc[slot]   = s_perf_up.tmem_alloc_ns;
                state->perf_up_prologue[slot]     = s_perf_up.prologue_ns;
                state->perf_up_tma_wait[slot]     = s_perf_up.tma_wait_ns;
                state->perf_up_mma_issue[slot]    = s_perf_up.mma_issue_ns;
                state->perf_up_mma_wait[slot]     = s_perf_up.mma_wait_ns;
                state->perf_up_loop_other[slot]   = s_perf_up.loop_other_ns;
                state->perf_up_cluster_sync[slot] = s_perf_up.cluster_sync_ns;
                state->perf_up_epilogue[slot]     = s_perf_up.epilogue_ns;
                state->perf_down_setup[slot]        = s_perf_down.setup_ns;
                state->perf_down_tmem_alloc[slot]   = s_perf_down.tmem_alloc_ns;
                state->perf_down_prologue[slot]     = s_perf_down.prologue_ns;
                state->perf_down_tma_wait[slot]     = s_perf_down.tma_wait_ns;
                state->perf_down_mma_issue[slot]    = s_perf_down.mma_issue_ns;
                state->perf_down_mma_wait[slot]     = s_perf_down.mma_wait_ns;
                state->perf_down_loop_other[slot]   = s_perf_down.loop_other_ns;
                state->perf_down_cluster_sync[slot] = s_perf_down.cluster_sync_ns;
                state->perf_down_epilogue[slot]     = s_perf_down.epilogue_ns;
                state->perf_compute_multi_expert_rows[slot] = s_perf_multi_expert_rows;
            }
        }
#endif
    }

}

// ============================================================================
// Combine Worker v2: Full DeepEP internode.cu combine ported as __device__
// Polls per-expert compute_done signals, then runs the full combine protocol.
// SM pairing: even SM = NVLSender + RDMAReceiver + Coordinator
//             odd SM  = NVLAndRDMAForwarder + Coordinator
// Template constants: kNumRDMARanks=2, kNumCombineForwarderWarps=24
// ============================================================================

__device__ void combine_worker_v2(
    int combine_sm_idx,       // 0-based index among combine SMs
    MegaKernelState* state
) {
    using namespace internode;
    using dtype_t = nv_bfloat16;

    const int num_tokens = state->combine_num_tokens;
    const int num_combined_tokens = state->combine_num_combined_tokens;
    const int num_channels = state->num_combine_channels;
    const int num_logical_channels = state->num_logical_channels;
    EP_DEVICE_ASSERT(num_channels == state->num_dispatch_channels);  // physical channel counts must match for queue reuse
    const int num_ranks = state->num_ranks;
    constexpr int kNumRDMARanks_C = MK_NUM_RDMA_RANKS;
    const int rdma_rank = state->rank / NUM_MAX_NVL_PEERS;

    if (threadIdx.x == 0 && combine_sm_idx == 0) {
        printf("rank: %d, combine_sm_idx: %d num_channels: %d num_logical_channels: %d \n", state->rank, combine_sm_idx, num_channels, num_logical_channels);
    }

    // ---- Overlap design (v2: per-channel compute-combine overlap) ----
    // Per-channel pipeline: dispatch(ch) -> normalize(ch) -> combine(ch)
    // Sender SM (even): warp0 does head normalization after channel's dispatch is done,
    //                   then signals channel_normalized. All warps then proceed.
    // Forwarder SM (odd): waits channel_normalized, then uses normalized heads.

    // DEBUG: very first entry point (unconditional, one per SM)
    // if (threadIdx.x == 0)
    //     printf("[MK-DEBUG][COMBINE-ENTRY] rank=%d block=%d combine_sm=%d is_forwarder_sm=%d\n",
    //            state->rank, blockIdx.x, combine_sm_idx, combine_sm_idx % 2 == 1);

    // Compute is currently a placeholder; combine consumes dispatch-filled compact input.

    // --- DeepEP combine kernel logic begins (direct port from internode.cu L1741-2269) ---
    enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };

    constexpr int kNumForwarders_C = kNumCombineForwarders;           // 24
    constexpr int kNumWarpsPerForwarder_C = kNumCombineWarpsPerForwarder;  // 12
    constexpr int kNumRDMAReceivers_C = kNumCombineRDMAReceivers;    // 16
    constexpr int kNumTopkRDMARanks_C = kNumTopkCombineRDMARanks;    // 2

    const auto sm_id = combine_sm_idx;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    const int num_topk = state->num_topk;
    const int hidden = state->combine_hidden;
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const auto hidden_bytes = hidden_int4 * sizeof(int4);
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

    const auto nvl_rank = state->rank % NUM_MAX_NVL_PEERS;

    // Role assignment (same as DeepEP combine kernel)
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (not is_forwarder_sm) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = (warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id};
            } else if (warp_id < kNumForwarders_C) {
                return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        } else {
            if (warp_id < kNumForwarders_C) {
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders_C;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    const int num_max_rdma_chunked_send_tokens = state->num_max_combine_rdma_chunked_send_tokens;
    const int num_max_rdma_chunked_recv_tokens = state->num_max_combine_rdma_chunked_recv_tokens;
    const int num_max_nvl_chunked_send_tokens = state->num_max_combine_nvl_chunked_send_tokens;
    const int num_max_nvl_chunked_recv_tokens = state->num_max_combine_nvl_chunked_recv_tokens;
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks_C;

    // Combine input/output pointers
    int4* combined_x = state->combined_x;
    float* combined_topk_weights = state->combined_topk_weights;
    const int4* x = reinterpret_cast<const int4*>(state->combine_x);
    const float* topk_weights = state->combine_topk_weights;
    const int* combined_rdma_head_base = state->combined_rdma_head;     // Will be normalized in-place
    int* combined_nvl_head_global_base = state->combined_nvl_head;      // Will be normalized in-place
    const SourceMeta* src_meta = reinterpret_cast<const SourceMeta*>(state->combine_src_meta);
    const int* rdma_channel_prefix_matrix = state->combine_rdma_channel_prefix_matrix;
    const int* rdma_rank_prefix_sum = state->combine_rdma_rank_prefix_sum;
    const int* gbl_channel_prefix_matrix = state->combine_gbl_channel_prefix_matrix;
    void* rdma_buffer_ptr = state->combine_rdma_buffer_ptr;
    void** buffer_ptrs = state->combine_buffer_ptrs;

    constexpr int kCombineNumTMAStages = 2;

    for (int logical_channel_id = channel_id; logical_channel_id < num_logical_channels; logical_channel_id += num_channels) {
    const int* combined_rdma_head = combined_rdma_head_base + logical_channel_id * state->combine_rdma_head_stride;
    int* combined_nvl_head_base = combined_nvl_head_global_base + logical_channel_id * state->combine_nvl_head_stride;
    int combine_nvl_sender_cached_channel_head_idx = 0;
    int combine_nvl_sender_cached_channel_tail_idx = 0;
    int combine_forwarder_cached_nvl_channel_tail_idx = 0;
    int combine_coordinator_last_rdma_head = 0;
    int combine_coordinator_last_nvl_head[kNumRDMARanks_C] = {0};
    uint32_t combine_forwarder_tma_phase[kCombineNumTMAStages] = {0};
#ifdef MK_PERF_TRACE
    if (thread_id == 0) {
        int combine_lch_role = is_forwarder_sm ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + combine_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        state->perf_combine_lch_ts[trace_idx + 0] = globaltimer_ns();
    }
#endif
    // if (thread_id == 0)
    //     printf("[MK-DBG][COMBINE][logical-loop-enter] rank=%d block=%d sm=%d combine_sm=%d physical_ch=%d logical_ch=%d is_forwarder_sm=%d dispatch_done=%d normalized=%d barrier=%d\n",
    //            state->rank, static_cast<int>(blockIdx.x), sm_id, combine_sm_idx, channel_id, logical_channel_id,
    //            is_forwarder_sm, ld_acquire_sys_global(&state->channel_dispatch_done[logical_channel_id]),
    //            ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]),
    //            ld_acquire_sys_global(&state->combine_channel_barrier[logical_channel_id]));

    // --- Per-logical-channel head normalization (done by Coordinator on Sender SM) ---

    // All warps on this SM wait for channel_normalized[channel_id] before proceeding
    // except the Coordinator on the Sender SM which performs the normalization.
    if (!is_forwarder_sm && warp_role == WarpRole::kCoordinator) {
        // Step 1: Wait for this channel's dispatch to complete
        if (lane_id == 0) {
            // printf("[MK-DBG][COMBINE][normalizer-dispatch-wait] rank=%d block=%d physical_ch=%d logical_ch=%d channel_done=%d need=%d\n",
            //        state->rank, static_cast<int>(blockIdx.x), channel_id, logical_channel_id,
            //        ld_acquire_sys_global(&state->channel_dispatch_done[logical_channel_id]), NUM_MAX_NVL_PEERS);
            auto start_time = clock64();
            while (ld_acquire_sys_global(&state->channel_dispatch_done[logical_channel_id]) < NUM_MAX_NVL_PEERS) {
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK combine normalizer timeout waiting for channel_dispatch_done[%d] (physical_ch=%d), got %d\n",
                           logical_channel_id, channel_id, ld_acquire_sys_global(&state->channel_dispatch_done[logical_channel_id]));
                    trap();
                }
                __nanosleep(32);
            }
#ifdef MK_PERF_TRACE
            int trace_idx = (logical_channel_id * 2) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
            state->perf_combine_lch_ts[trace_idx + 1] = globaltimer_ns();
#endif
        }
        __syncwarp();

        // Step 2: Normalize combined_rdma_head for this channel's token range.
        // This mirrors original DeepEP: token_idx remains the global/original token index,
        // but the head array itself is private to logical_channel_id.
        {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_logical_channels, logical_channel_id, token_start_idx, token_end_idx);
            if (lane_id < kNumRDMARanks_C) {
                int rdma_prefix_idx = lane_id * num_logical_channels + logical_channel_id;
                int rdma_ch_count = state->combine_rdma_channel_token_count[rdma_prefix_idx];
                int last_head = 1 << 25;
                for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                    auto current_head = ld_acquire_sys_global(
                        const_cast<int*>(combined_rdma_head) + token_idx * kNumRDMARanks_C + lane_id);
                    bool is_in_src_rdma = current_head >= 0;
                    int last_before = last_head;
                    int normalized_head = current_head;
                    if (current_head < 0) {
                        normalized_head = -last_head - 1;
                        const_cast<int*>(combined_rdma_head)[token_idx * kNumRDMARanks_C + lane_id] = normalized_head;
                    } else {
                        last_head = current_head;
                    }
#ifdef MK_TOKEN_TRACE
                    if (current_head >= 0 || rdma_ch_count > 0 || logical_channel_id >= num_channels) {
                        int stored_head = ld_acquire_sys_global(
                            const_cast<int*>(combined_rdma_head) + token_idx * kNumRDMARanks_C + lane_id);
                        printf("[MK-DIAG][COMBINE-RDMA-HEAD-NORM] rank=%d rdma_rank=%d nvl_rank=%d physical_ch=%d logical_ch=%d src_rdma_lane=%d global_token=%d in_src_rdma=%d raw=%d norm=%d stored=%d last_before=%d last_after=%d rdma_prefix_idx=%d rdma_ch_count=%d token_range=[%d,%d) head_ptr=%p\n",
                               state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id,
                               lane_id, token_idx, static_cast<int>(is_in_src_rdma), current_head,
                               normalized_head, stored_head, last_before, last_head, rdma_prefix_idx,
                               rdma_ch_count, token_start_idx, token_end_idx,
                               const_cast<int*>(combined_rdma_head) + token_idx * kNumRDMARanks_C + lane_id);
                    }
#endif
                }
            }
        }
        __syncwarp();

        // Step 3: Normalize combined_nvl_head for the RDMA compact token span.
        // Dispatch forwarders already populated this head array in the original
        // DeepEP namespace: rank prefix + channel-local RDMA prefix. Do not rebuild
        // it from alternate token prefixes here; combine forwarders consume the
        // compact RDMA span below.
        {
            for (int dst_rdma_rank = 0; dst_rdma_rank < kNumRDMARanks_C; ++dst_rdma_rank) {
                int rdma_prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
                int channel_end = ld_acquire_sys_global(rdma_channel_prefix_matrix + rdma_prefix_idx);
                int channel_count = ld_acquire_sys_global(state->combine_rdma_channel_token_count + rdma_prefix_idx);
                int channel_start = channel_end - channel_count;
                int rank_shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                int rdma_token_start = rank_shift + channel_start;
                int rdma_token_end = rank_shift + channel_end;
                int nvl_head_capacity = state->combine_nvl_head_stride / NUM_MAX_NVL_PEERS;
                EP_DEVICE_ASSERT(channel_count >= 0 and channel_end >= channel_start);
                EP_DEVICE_ASSERT(rdma_token_start >= 0 and rdma_token_end >= rdma_token_start and rdma_token_end <= nvl_head_capacity);

                if (lane_id < NUM_MAX_NVL_PEERS) {
                    int last_head = 1 << 25;
                    for (int token_idx = rdma_token_end - 1; token_idx >= rdma_token_start; --token_idx) {
                        auto current_head = ld_acquire_sys_global(
                            combined_nvl_head_base + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                        if (current_head < 0) {
                            combined_nvl_head_base[token_idx * NUM_MAX_NVL_PEERS + lane_id] = -last_head - 1;
                        } else {
                            last_head = current_head;
                        }
                    }
                }
                __syncwarp();
            }
        }

        // Step 4: Signal normalization complete
        __threadfence_system();
        if (lane_id == 0) {
            st_release_sys_global(&state->channel_normalized[logical_channel_id], 1);
#ifdef MK_PERF_TRACE
            int trace_idx = (logical_channel_id * 2) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
            state->perf_combine_lch_ts[trace_idx + 2] = globaltimer_ns();
#endif
            // printf("[MK-DBG][COMBINE][normalizer-done] rank=%d block=%d physical_ch=%d logical_ch=%d rdma_prefix0=%d gbl_prefix0=%d\n",
            //        state->rank, static_cast<int>(blockIdx.x), channel_id, logical_channel_id,
            //        rdma_channel_prefix_matrix[logical_channel_id],
            //        gbl_channel_prefix_matrix[logical_channel_id]);
        }
    }

    // All warps (except the sender-SM coordinator that just signaled) wait for normalization
    if (warp_role != WarpRole::kCoordinator || is_forwarder_sm) {
        if (lane_id == 0) {
            auto start_time = clock64();
            while (ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]) == 0) {
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK combine warp timeout waiting for channel_normalized[%d] (physical_ch=%d), role=%d\n",
                           logical_channel_id, channel_id, (int)warp_role);
                    trap();
                }
                __nanosleep(32);
            }
#ifdef MK_PERF_TRACE
            if (is_forwarder_sm && thread_id == 0) {
                int trace_idx = (logical_channel_id * 2 + 1) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
                state->perf_combine_lch_ts[trace_idx + 2] = globaltimer_ns();
            }
#endif
        }
        __syncwarp();
    }

#ifdef MK_PERF_TRACE
    if (thread_id == 0) {
        int combine_lch_role = is_forwarder_sm ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + combine_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        if (state->perf_combine_lch_ts[trace_idx + 1] == 0)
            state->perf_combine_lch_ts[trace_idx + 1] = state->perf_combine_lch_ts[trace_idx + 2];
        state->perf_combine_lch_ts[trace_idx + 3] = globaltimer_ns();
    }
#endif

    if (warp_role == WarpRole::kNVLSender) {
        // ========== NVL Sender (direct port from internode.cu L1784-1922) ==========
        const auto dst_nvl_rank = warp_id;

        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr,
                                                 num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                                 NUM_MAX_NVL_PEERS,
                                                 logical_channel_id,
                                                 num_logical_channels,
                                                 nvl_rank)
                                 .advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, dst_nvl_rank)
                                    .advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, nvl_rank)
                                    .advance_also(local_buffer_ptr);

        // TMA stuffs
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumCombineTMABytesPerSenderWarp;
        auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
        uint32_t tma_phase = 0;
        if (elect_one_sync()) {
            mbarrier_init(tma_mbarrier, 1);
            fence_barrier_init();
            EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumCombineTMABytesPerSenderWarp);
        }
        __syncwarp();

        // Get tasks for each RDMA lane in the DeepEP compact recv-token namespace.
        // In the fused kernel, the next prefix entry may belong to a later logical
        // channel that has not run yet, so use the current channel's explicit count.
        int token_start_idx = 0, token_end_idx = 0;
        int nvl_sender_prefix_idx = -1;
        int nvl_sender_count = 0;
        if (lane_id < kNumRDMARanks_C) {
            nvl_sender_prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_logical_channels + logical_channel_id;
            token_start_idx = ld_acquire_sys_global(gbl_channel_prefix_matrix + nvl_sender_prefix_idx);
            nvl_sender_count = ld_acquire_sys_global(state->combine_gbl_channel_token_count + nvl_sender_prefix_idx);
            token_end_idx = token_start_idx + nvl_sender_count;
            EP_DEVICE_ASSERT(token_start_idx >= 0 and nvl_sender_count >= 0 and token_end_idx <= num_tokens);
        }
        __syncwarp();

        auto& cached_channel_head_idx = combine_nvl_sender_cached_channel_head_idx;
        auto& cached_channel_tail_idx = combine_nvl_sender_cached_channel_tail_idx;

        // DEBUG: NVL sender task range (only for ch=0, use warp-safe approach)
        {
            int my_range = (lane_id < kNumRDMARanks_C) ? (token_end_idx - token_start_idx) : 0;
            // Warp-reduce to get total tasks
            for (int offset = 16; offset > 0; offset >>= 1)
                my_range += __shfl_down_sync(0xffffffff, my_range, offset);
            // if (lane_id == 0 && logical_channel_id == 0)
            //     printf("MK combine NVL sender: sm=%d, ch=%d, dst_nvl=%d, total_tasks=%d\n",
            //            sm_id, channel_id, dst_nvl_rank, my_range);
        }
        __syncwarp();

        // Iterate over all tokens and send by chunks
        int current_rdma_idx = channel_id % kNumRDMARanks_C;
        while (true) {
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;

            bool is_lane_ready = false;
            auto start_time = clock64();
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                is_lane_ready = lane_id < kNumRDMARanks_C and token_start_idx < token_end_idx and
                    num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
                if (__any_sync(0xffffffff, is_lane_ready))
                    break;

                if (lane_id < kNumRDMARanks_C and token_start_idx < token_end_idx)
                    cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks_C) {
                    int live_head = ld_volatile_global(nvl_channel_head.buffer() + lane_id);
                    int live_tail = ld_acquire_sys_global(nvl_channel_tail.buffer() + lane_id);
                    int used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                    int free_slots = num_max_nvl_chunked_recv_tokens_per_rdma - used_slots;
                    int blocked_head_token = cached_channel_head_idx < cached_channel_tail_idx ?
                        gbl_channel_prefix_matrix[nvl_sender_prefix_idx] + cached_channel_head_idx : -1;
                    int next_send_token = token_start_idx < token_end_idx ? token_start_idx : -1;
                    int last_sent_token = cached_channel_tail_idx > 0 ?
                        gbl_channel_prefix_matrix[nvl_sender_prefix_idx] + cached_channel_tail_idx - 1 : -1;
                    printf("MK combine NVL sender timeout, ch: %d, logical_ch: %d, RDMA: %d, nvl: %d, dst NVL: %d, lane: %d, "
                           "head=%d, live_head=%d, tail=%d, live_tail=%d, used=%d, free=%d, capacity=%d, slots_needed=%d, "
                           "prefix_idx=%d, sender_base=%d, sender_count=%d, range=[%d,%d), "
                           "waiting_next_token=%d, blocked_head_token=%d, last_sent_token=%d, "
                           "head_ptr=%p, tail_ptr=%p, channel_normalized=%d\n",
                           channel_id, logical_channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id,
                           cached_channel_head_idx, live_head, cached_channel_tail_idx, live_tail,
                           used_slots, free_slots, num_max_nvl_chunked_recv_tokens_per_rdma, num_max_nvl_chunked_send_tokens,
                           nvl_sender_prefix_idx, gbl_channel_prefix_matrix[nvl_sender_prefix_idx], nvl_sender_count,
                           gbl_channel_prefix_matrix[nvl_sender_prefix_idx], token_end_idx,
                           next_send_token, blocked_head_token, last_sent_token,
                           (void*)(nvl_channel_head.buffer() + lane_id), (void*)(nvl_channel_tail.buffer() + lane_id),
                           ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]));
                    trap();
                }
            }

            for (int i = 0; i < kNumRDMARanks_C; ++i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks_C;
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
                int producer_token_end_idx = __shfl_sync(0xffffffff, token_end_idx, current_rdma_idx);
                int num_tokens_in_chunk = min(num_max_nvl_chunked_send_tokens, producer_token_end_idx - static_cast<int>(token_idx));

                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    // Per-token-ready gate: wait for compute to finish before reading compute_output.
                    // expected==0 is safe here because combine reaches this protocol only after
                    // dispatch_done/head normalization, so token_compute_expected is final.
                    if (elect_one_sync()) {
                        auto wait_start = clock64();
#ifdef MK_PERF_TRACE
                        int64_t ready_wait_start_ns = globaltimer_ns();
#endif
                        while (true) {
                            if (ld_acquire_sys_global(&state->combine_token_ready[token_idx]) == 1)
                                break;
                            if (ld_acquire_sys_global(&state->token_compute_expected[token_idx]) == 0)
                                break;
                            if (clock64() - wait_start > NUM_TIMEOUT_CYCLES) {
                                if (timeout_log_once(state, kTimeoutLogComputeReady)) {
                                    printf("MK combine per-token-ready timeout, rank=%d token=%lld expected=%d done=%d\n",
                                           state->rank, (long long)token_idx,
                                           ld_acquire_sys_global(&state->token_compute_expected[token_idx]),
                                           ld_acquire_sys_global(&state->token_compute_done[token_idx]));
                                }
                                trap();
                            }
                            __nanosleep(32);
                        }
#ifdef MK_PERF_TRACE
                        // Only one NVL sender warp (warp_id==0) records, so the value is a
                        // single execution stream's real occupancy, comparable to the wall time.
                        if (warp_id == 0)
                            state->perf_comb_wait_ready_ns[logical_channel_id * 2 + 0] += globaltimer_ns() - ready_wait_start_ns;
#endif
                    }
                    __syncwarp();
                    // NOTE: DeepEP's combine NVL sender forwards every token in the
                    // [token_start_idx, token_end_idx) range unconditionally. The range from
                    // gbl_channel_prefix_matrix already encodes exactly which tokens belong to
                    // (src_rdma_lane, dst_nvl). The dispatch-time meta NVL bits describe a
                    // different (dispatch) routing and must NOT be used to filter combine sends;
                    // doing so drops legitimately-assigned tokens and stalls the NVL queue tail,
                    // deadlocking the destination combine forwarder.
                    int dst_slot_idx = 0;
                    int queue_tail_before = __shfl_sync(0xffffffff, cached_channel_tail_idx, current_rdma_idx);
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;
                    auto shifted_x = x + token_idx * hidden_int4;
                    // if (lane_id == 0) {
                    //     printf("[MK-TRACE][COMBINE][NVL-SENDER][SEND] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d dst_nvl_rank=%d current_rdma_idx=%d token=%lld dst_slot=%d shifted_x=%p shifted_buffer=%p src_meta_addr=%p topk_weights_addr=%p tma_buffer=%p hidden_int4=%d num_topk=%d\n",
                    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, dst_nvl_rank, current_rdma_idx,
                    //            (long long)token_idx, dst_slot_idx, shifted_x, shifted_x_buffers, src_meta + token_idx,
                    //            topk_weights + token_idx * num_topk, tma_buffer, (int)hidden_int4, num_topk);
                    // }
                    tma_store_wait<0>();
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_x, tma_mbarrier, hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
                    }
                    __syncwarp();
                    mbarrier_wait(tma_mbarrier, tma_phase);

                    if (lane_id == num_topk)
                        *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);

                    if (lane_id < num_topk)
                        *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) =
                            ld_nc_global(topk_weights + token_idx * num_topk + lane_id);

#ifdef MK_TOKEN_TRACE
                    if (lane_id == 0) {
                        auto* hptr = reinterpret_cast<nv_bfloat16*>(const_cast<int4*>(x + token_idx * hidden_int4));
                        SourceMeta send_meta = ld_nc_global(src_meta + token_idx);
                        int send_prefix_idx = (current_rdma_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_logical_channels + logical_channel_id;
                        int send_base = gbl_channel_prefix_matrix[send_prefix_idx];
                        printf("[MK-TOKEN][COMBINE-NVL-SEND] rank=%d token=%lld dst_nvl=%d src_rdma=%d ch=%d logical_ch=%d sender_prefix_idx=%d sender_base=%d queue_tail_before=%d sender_queue_token=%d dst_slot=%d dst_lane_slot=%d meta=(%d,0x%x) tail_ptr=%p x_ptr=%p dst_ptr=%p topk_w0=%f topk_w1=%f h0=%f\n",
                               state->rank, (long long)token_idx, dst_nvl_rank, current_rdma_idx, channel_id, logical_channel_id,
                               send_prefix_idx, send_base, queue_tail_before, send_base + queue_tail_before,
                               dst_slot_idx, dst_slot_idx % num_max_nvl_chunked_recv_tokens_per_rdma,
                               send_meta.src_rdma_rank, send_meta.is_token_in_nvl_rank_bits,
                               (void*)(nvl_channel_tail.buffer() + current_rdma_idx), (void*)shifted_x,
                               (void*)shifted_x_buffers,
                               ld_nc_global(topk_weights + token_idx * num_topk),
                               num_topk > 1 ? ld_nc_global(topk_weights + token_idx * num_topk + 1) : 0.0f,
                               __bfloat162float(hptr[0]));
                    }
#endif

                    tma_store_fence();
                    __syncwarp();
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
            }

            tma_store_wait<0>();
            __syncwarp();
            if (lane_id < kNumRDMARanks_C and is_lane_ready) {
#ifdef MK_TOKEN_TRACE
                printf("[MK-DIAG][COMBINE-NVL-SEND-TAIL] rank=%d physical_ch=%d logical_ch=%d dst_nvl=%d src_rdma_lane=%d tail_val=%d range_now=[%d,%d) count=%d tail_ptr=%p\n",
                       state->rank, channel_id, logical_channel_id, dst_nvl_rank, lane_id,
                       cached_channel_tail_idx, token_start_idx, token_end_idx, nvl_sender_count,
                       (void*)(nvl_channel_tail.buffer() + lane_id));
#endif
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
            }
        }
    } else {
        // if (threadIdx.x == 0) {
        //     printf("[MK-DEBUG][FWD-ENTER] rank=%d nvl_rank=%d block=%d combine_sm=%d\n",
        //                state->rank, nvl_rank, blockIdx.x, sm_id);
        // }

        // ========== Forwarder SM: NVLAndRDMAForwarder + RDMAReceiver + Coordinator ==========
        // (direct port from internode.cu L1923-2269)

        // RDMA symmetric layout
        auto rdma_channel_data = SymBuffer<int8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks_C, logical_channel_id, num_logical_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks_C, logical_channel_id, num_logical_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks_C, logical_channel_id, num_logical_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x =
            AsymBuffer<uint8_t>(
                local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels)
                .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head =
            AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, nvl_rank)
                .advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels)
                                    .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

        // Shared memory for warp synchronization
        __shared__ volatile int forwarder_nvl_head[kNumForwarders_C][NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders_C];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers_C][kNumRDMARanks_C];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers_C];
        auto sync_forwarder_smem = [=]() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumForwarders_C + 1) * 32)); };
        auto sync_rdma_receiver_smem = [=]() { asm volatile("barrier.sync 1, %0;" ::"r"((kNumRDMAReceivers_C + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // ========== NVL+RDMA Forwarder (internode.cu L1955-2144) ==========
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder_C;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder_C;
            auto send_buffer =
                dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder_C == 1) {
                    __syncwarp();
                } else {
                    asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder_C * 32));
                }
            };

            // TMA stuffs
            constexpr int kNumStages = kCombineNumTMAStages;
            constexpr int kNumTMALoadBytes = sizeof(int4) * 32;
            constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
            constexpr int kNumTMABytesPerForwarderWarp = kNumStages * kNumTMABufferBytesPerStage;
            EP_STATIC_ASSERT(kNumTMABytesPerForwarderWarp <= kNumCombineTMABytesPerForwarderWarp,
                             "combine forwarder TMA buffer is not large enough");

            extern __shared__ __align__(1024) uint8_t smem_buffer[];
            auto smem_ptr = smem_buffer + warp_id * kNumCombineTMABytesPerForwarderWarp;
            auto tma_mbarrier = [=](const int& i) {
                return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
            };
            auto& tma_phase = combine_forwarder_tma_phase;
            // Logical channels reuse the same per-warp TMA scratch space. Make sure
            // the previous channel has no outstanding TMA store before reinitializing
            // the mbarriers for this channel.
            tma_store_wait<0>();
            if (lane_id < kNumStages) {
                mbarrier_init(tma_mbarrier(lane_id), 32);
                fence_barrier_init();
            }
            __syncwarp();

            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);
            nvl_channel_head.advance(dst_rdma_rank);
            nvl_channel_tail.advance(dst_rdma_rank);

            lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            sync_forwarder_smem();

            auto& cached_nvl_channel_tail_idx = combine_forwarder_cached_nvl_channel_tail_idx;
            // RDMA prefix entries for later logical channels may not be available yet.
            // Use this channel's explicit count to recover its local start.
            int rdma_prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
            int channel_end = ld_acquire_sys_global(rdma_channel_prefix_matrix + rdma_prefix_idx);
            int num_tokens_to_combine = ld_acquire_sys_global(state->combine_rdma_channel_token_count + rdma_prefix_idx);
            int channel_start = channel_end - num_tokens_to_combine;
            int num_tokens_prefix = channel_start + (dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1]);
            int* logical_combined_nvl_head = combined_nvl_head_base + num_tokens_prefix * NUM_MAX_NVL_PEERS;
#ifdef MK_TOKEN_TRACE
            if (lane_id == 0 && sub_warp_id == 0 && num_tokens_to_combine > 0) {
                printf("[MK-DIAG][COMBINE-FWD-ENTRY] rank=%d rdma_rank=%d nvl_rank=%d physical_ch=%d logical_ch=%d dst_rdma=%d tokens=%d rank_prefix=%d global_range=[%d,%d) nvl_head_base=%p combined_nvl_base=%p nvl_tail_base=%p nvl_x_base=%p rdma_tail_src_ptr=%p\n",
                       state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id, dst_rdma_rank,
                       num_tokens_to_combine, num_tokens_prefix, num_tokens_prefix, num_tokens_prefix + num_tokens_to_combine,
                       (void*)logical_combined_nvl_head, (void*)combined_nvl_head_base,
                       (void*)nvl_channel_tail.buffer(), (void*)nvl_channel_x.buffer(),
                       (void*)rdma_channel_tail.buffer(rdma_rank));
            }
#endif

            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        int cur_head = ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                        printf("MK combine forwarder (RDMA check) timeout, ch: %d, dst_rdma: %d, "
                               "logical_token_start=%d, rdma_token_start=%d, cur_head=%d, capacity=%d, needed=%d\n",
                               channel_id, dst_rdma_rank,
                               token_start_idx, token_start_idx, cur_head, num_max_rdma_chunked_recv_tokens, num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder_C) {
                    const int global_token_idx = num_tokens_prefix + token_idx;
                    // Read normalized head (original DeepEP logic)
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS) {
                        int lane_raw_head = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                        expected_head = lane_raw_head;
                        // Normalized semantics: negative = -(next_valid_head)-1, positive = actual head
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    start_time = clock64();
                    // Wait for NVL tail to advance past expected_head (original DeepEP logic)
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
                            int head0 = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS);
                            int head1 = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 1);
                            int lane_raw_head = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                            int local_head = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + nvl_rank);
                            int tail0 = ld_acquire_sys_global(nvl_channel_tail.buffer(0));
                            int tail1 = ld_acquire_sys_global(nvl_channel_tail.buffer(1));
                            int global_token_idx = num_tokens_prefix + token_idx;
                            int sender_prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_logical_channels + logical_channel_id;
                            int sender_count = state->combine_gbl_channel_token_count[sender_prefix_idx];
                            int sender_base = gbl_channel_prefix_matrix[sender_prefix_idx];
                            int sender_local_head = expected_head >= 0 ? expected_head : -1;
                            int sender_token = sender_local_head >= 0 ? sender_base + sender_local_head : -1;
                            int local_dst_prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) * num_logical_channels + logical_channel_id;
                            int local_dst_count = state->combine_gbl_channel_token_count[local_dst_prefix_idx];
                            int local_dst_base = gbl_channel_prefix_matrix[local_dst_prefix_idx];
                            int local_dst_token = sender_local_head >= 0 ? local_dst_base + sender_local_head : -1;
                            printf("MK combine forwarder (NVL check) timeout, rank=%d rdma_rank=%d nvl_rank=%d ch=%d logical_ch=%d dst_rdma=%d token=%d global_token=%d lane=%d lane_raw_head=%d local_head=%d expected_head=%d cached_tail=%d head0=%d head1=%d tail0=%d tail1=%d sender_prefix_idx=%d sender_token=%d sender_range=[%d,%d) sender_count=%d local_dst_prefix_idx=%d local_dst_token=%d local_dst_range=[%d,%d) local_dst_count=%d nvl_tail_ptr=%p nvl_x_lane_base=%p sub_warp=%d rank_prefix=%d tokens=%d\n",
                                   state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id,
                                   dst_rdma_rank, token_idx, global_token_idx, lane_id,
                                   lane_raw_head, local_head, expected_head,
                                   cached_nvl_channel_tail_idx, head0, head1, tail0, tail1,
                                   sender_prefix_idx, sender_token, sender_base, sender_base + sender_count, sender_count,
                                   local_dst_prefix_idx, local_dst_token, local_dst_base, local_dst_base + local_dst_count, local_dst_count,
                                   (void*)nvl_channel_tail.buffer(lane_id), (void*)nvl_channel_x.buffer(lane_id), sub_warp_id,
                                   num_tokens_prefix, num_tokens_to_combine);
                            trap();
                        }
                    }

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;
                    // if (lane_id == 0) {
                    //     int global_token_idx = num_tokens_prefix + token_idx;
                    //     printf("[MK-TRACE][COMBINE][FWD][COMBINE-START] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d sub_warp=%d local_token=%d global_token=%d rdma_slot=%d shifted=%p send_buffer=%p num_bytes_per_token=%d hidden_int4=%d hidden_bytes=%d num_topk=%d\n",
                    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
                    //            dst_rdma_rank, sub_warp_id, token_idx, global_token_idx, (int)rdma_slot_idx,
                    //            shifted, send_buffer, (int)num_bytes_per_token, (int)hidden_int4, (int)hidden_bytes, num_topk);
                    // }
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) +
                            hidden_int4_idx;
                    };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                                     hidden_bytes + sizeof(SourceMeta)) +
                                            topk_idx);
                    };
#ifdef MK_TOKEN_TRACE
                    if (lane_id == 0) {
                        int heads[NUM_MAX_NVL_PEERS];
#pragma unroll
                        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                            heads[i] = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + i);
                        int slot0 = heads[0] >= 0 ? heads[0] % num_max_nvl_chunked_recv_tokens_per_rdma : 0;
                        int slot1 = heads[1] >= 0 ? heads[1] % num_max_nvl_chunked_recv_tokens_per_rdma : 0;
                        int local_slot = heads[nvl_rank] >= 0 ? heads[nvl_rank] % num_max_nvl_chunked_recv_tokens_per_rdma : 0;
                        int anomalous_head = 0;
#pragma unroll
                        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                            anomalous_head |= static_cast<int>(heads[i] < -1000000);
                        auto* hptr = reinterpret_cast<nv_bfloat16*>(const_cast<int4*>(get_addr_fn(0, slot0, 0)));
                        auto* local_hptr = reinterpret_cast<nv_bfloat16*>(const_cast<int4*>(get_addr_fn(nvl_rank, local_slot, 0)));
                        printf("[MK-TOKEN][COMBINE-NVL-FWD] rank=%d token=%d global_token=%d dst_rdma=%d head=%d ch=%d logical_ch=%d rank_prefix=%d tokens=%d slot0=%d head0=%d head1=%d slot1=%d local_head=%d local_slot=%d topk0_w=%f h0=%f local_topk0_w=%f local_h0=%f tail0=%d tail1=%d tail_local=%d normalized=%d anomalous=%d head_ptr=%p local_head_ptr=%p\n",
                               state->rank, token_idx, num_tokens_prefix + token_idx, dst_rdma_rank, expected_head, channel_id, logical_channel_id,
                               num_tokens_prefix, num_tokens_to_combine, slot0, heads[0], heads[1], slot1,
                               heads[nvl_rank], local_slot, recv_tw_fn(0, slot0, 0), expected_head >= 0 ? __bfloat162float(hptr[0]) : 0.f,
                               recv_tw_fn(nvl_rank, local_slot, 0), heads[nvl_rank] >= 0 ? __bfloat162float(local_hptr[0]) : 0.f,
                               static_cast<int>(ld_acquire_sys_global(nvl_channel_tail.buffer(0))),
                               static_cast<int>(ld_acquire_sys_global(nvl_channel_tail.buffer(1))),
                               static_cast<int>(ld_acquire_sys_global(nvl_channel_tail.buffer(nvl_rank))),
                               ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]),
                               anomalous_head,
                               (void*)(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS),
                               (void*)(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + nvl_rank));
                    }
#endif

#ifdef MK_TOKEN_TRACE
                    bool has_nvl_contribution = __any_sync(0xffffffff, lane_id < NUM_MAX_NVL_PEERS && expected_head >= 0);
                    if (lane_id == 0) {
                        int head0 = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS);
                        int head1 = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 1);
                        int head2 = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 2);
                        int head3 = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 3);
                        int local_head = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + nvl_rank);
                        int anomalous_head = static_cast<int>(head0 < -1000000 || head1 < -1000000 || head2 < -1000000 || head3 < -1000000 || local_head < -1000000);
                        printf("[MK-DIAG][COMBINE-NVL-FWD-CHECK] rank=%d physical_ch=%d logical_ch=%d dst_rdma=%d sub_warp=%d warp_id=%d token=%d global_token=%d has_nvl_contribution=%d lane0_head=%d local_head=%d heads=[%d,%d,%d,%d] token_chunk=[%d,%d) global_chunk=[%d,%d) smem=%p mbar=[%p,%p] tma_phase=[%u,%u] hidden_int4=%d num_topk=%d tma_bytes=%d normalized=%d anomalous=%d head_base=%p\n",
                               state->rank, channel_id, logical_channel_id, dst_rdma_rank, sub_warp_id,
                               warp_id, token_idx, num_tokens_prefix + token_idx, static_cast<int>(has_nvl_contribution),
                               expected_head, local_head, head0, head1, head2, head3, token_start_idx, token_end_idx,
                               num_tokens_prefix + token_start_idx, num_tokens_prefix + token_end_idx,
                               smem_ptr, tma_mbarrier(0), tma_mbarrier(1), tma_phase[0], tma_phase[1], hidden_int4, num_topk,
                               kNumTMALoadBytes, ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]),
                               anomalous_head,
                               (void*)(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS));
                        if (num_tokens_prefix + token_idx == 2) {
                            int slot0 = head0 >= 0 ? head0 % num_max_nvl_chunked_recv_tokens_per_rdma : 0;
                            int slot1 = head1 >= 0 ? head1 % num_max_nvl_chunked_recv_tokens_per_rdma : 0;
                            auto* src0 = reinterpret_cast<nv_bfloat16*>(const_cast<int4*>(get_addr_fn(0, slot0, 0)));
                            auto* src1 = reinterpret_cast<nv_bfloat16*>(const_cast<int4*>(get_addr_fn(1, slot1, 0)));
                            printf("[MK-DIAG][TOKEN2-COMBINE-NVL-FWD-CHECK] rank=%d rdma_rank=%d nvl_rank=%d physical_ch=%d logical_ch=%d dst_rdma=%d sub_warp=%d warp_id=%d token=%d global_token=%d has_nvl_contribution=%d expected_lane0=%d local_head=%d heads=[%d,%d,%d,%d] slots=[%d,%d] topk0=[%f,%f] h0=[%f,%f] token_chunk=[%d,%d) global_chunk=[%d,%d) normalized=%d head_base=%p\n",
                                   state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id,
                                   dst_rdma_rank, sub_warp_id, warp_id, token_idx, num_tokens_prefix + token_idx,
                                   static_cast<int>(has_nvl_contribution), expected_head, local_head,
                                   head0, head1, head2, head3, slot0, slot1,
                                   recv_tw_fn(0, slot0, 0), recv_tw_fn(1, slot1, 0),
                                   head0 >= 0 ? __bfloat162float(src0[0]) : 0.0f,
                                   head1 >= 0 ? __bfloat162float(src1[0]) : 0.0f,
                                   token_start_idx, token_end_idx,
                                   num_tokens_prefix + token_start_idx, num_tokens_prefix + token_end_idx,
                                   ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]),
                                   (void*)(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS));
                        }
                    }
#endif
#ifdef MK_TOKEN_TRACE
                    if (lane_id == 0) {
                        int heads[NUM_MAX_NVL_PEERS];
                        int slots[NUM_MAX_NVL_PEERS];
                        float src_w0[NUM_MAX_NVL_PEERS];
                        float src_h0[NUM_MAX_NVL_PEERS];
#pragma unroll
                        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) {
                            heads[i] = ld_acquire_sys_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + i);
                            slots[i] = heads[i] >= 0 ? heads[i] % num_max_nvl_chunked_recv_tokens_per_rdma : 0;
                            auto* src_hptr = reinterpret_cast<nv_bfloat16*>(const_cast<int4*>(get_addr_fn(i, slots[i], 0)));
                            src_w0[i] = heads[i] >= 0 ? recv_tw_fn(i, slots[i], 0) : 0.0f;
                            src_h0[i] = heads[i] >= 0 ? __bfloat162float(src_hptr[0]) : 0.0f;
                        }
                        auto* dst_hptr = reinterpret_cast<nv_bfloat16*>(shifted);
                        float dst_w0_before = ld_nc_global(reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)));
                        printf("[MK-DIAG][COMBINE-NVL-FWD-COMBINE-BEGIN] rank=%d physical_ch=%d logical_ch=%d dst_rdma=%d sub_warp=%d warp_id=%d token=%d global_token=%d expected_head=%d rdma_slot=%d shifted=%p dst_w0_before=%f dst_h0_before=%f heads=[%d,%d,%d,%d] slots=[%d,%d,%d,%d] src_w0=[%f,%f,%f,%f] src_h0=[%f,%f,%f,%f] phase=[%u,%u]\n",
                               state->rank, channel_id, logical_channel_id, dst_rdma_rank, sub_warp_id,
                               warp_id, token_idx, num_tokens_prefix + token_idx, expected_head,
                               static_cast<int>(rdma_slot_idx), shifted, dst_w0_before, __bfloat162float(dst_hptr[0]),
                               heads[0], heads[1], heads[2], heads[3], slots[0], slots[1], slots[2], slots[3],
                               src_w0[0], src_w0[1], src_w0[2], src_w0[3], src_h0[0], src_h0[1], src_h0[2], src_h0[3],
                               tma_phase[0], tma_phase[1]);
                    }
#endif
                    // [IMPORTANT]
                    // when hidden_size < 1024, using tma combine_token may cause hang
                    // make sure you combine_token correctly when using high parallelism
                    combine_token<NUM_MAX_NVL_PEERS, false, dtype_t, NUM_MAX_NVL_PEERS, true, kNumStages, kNumTMALoadBytes>(
                        expected_head >= 0,
                        expected_head,
                        lane_id,
                        hidden_int4,
                        num_topk,
                        static_cast<int4*>(shifted),
                        reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                        nullptr,
                        nullptr,
                        num_max_nvl_chunked_recv_tokens_per_rdma,
                        get_addr_fn,
                        recv_tw_fn,
                        smem_ptr,
                        tma_phase);

                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();

                // Issue RDMA send
#ifdef MK_TOKEN_TRACE
                if (lane_id == 0 && sub_warp_id == 0) {
                    printf("[MK-DIAG][COMBINE-FWD-RDMA-ISSUE-READY] rank=%d physical_ch=%d logical_ch=%d dst_rdma=%d token_chunk=[%d,%d) global_chunk=[%d,%d)\n",
                           state->rank, channel_id, logical_channel_id, dst_rdma_rank,
                           token_start_idx, token_end_idx, num_tokens_prefix + token_start_idx,
                           num_tokens_prefix + token_end_idx);
                }
#endif
                    if (sub_warp_id == kNumWarpsPerForwarder_C - 1) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                        const auto dst_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        const auto src_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        auto* rdma_tail_ptr = rdma_channel_tail.buffer(rdma_rank);

                        if (dst_rdma_rank != rdma_rank) {
                            nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
                                                              src_ptr,
                                                              num_bytes_per_msg,
                                                              translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                              logical_channel_id + num_logical_channels,
                                                              lane_id,
                                                              0);
                        } else {
                            rdma_tail_ptr = rdma_channel_tail.buffer(dst_rdma_rank);
                            memory_fence();
                        }

                        __syncwarp();
                        if (elect_one_sync()) {
                            nvshmemi_ibgda_amo_nonfetch_add(rdma_tail_ptr,
                                                            num_chunked_tokens,
                                                            translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                            logical_channel_id + num_logical_channels,
                                                            dst_rdma_rank == rdma_rank);
                        }
                    }

            }

            __syncwarp();
            // Set INT_MAX before retiring so Coordinator won't be blocked by this warp
            if (lane_id < NUM_MAX_NVL_PEERS)
                forwarder_nvl_head[warp_id][lane_id] = std::numeric_limits<int>::max();
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;

        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // ========== RDMA Receiver (internode.cu L2145-2222) ==========
            lane_id < kNumRDMARanks_C ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_logical_channels, logical_channel_id, token_start_idx, token_end_idx);
#ifdef MK_TOKEN_TRACE
            if (lane_id == 0 && warp_id == 0 && token_end_idx > token_start_idx) {
                printf("[MK-DIAG][RDMA-RECV-RANGE] rank=%d rdma_rank=%d nvl_rank=%d physical_ch=%d logical_ch=%d token_range=[%d,%d) num_combined_tokens=%d tail0=%d tail1=%d head0=%d head1=%d\n",
                       state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id,
                       token_start_idx, token_end_idx, num_combined_tokens,
                       static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(0))),
                       static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(1))),
                       static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(0))),
                       static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(1))));
            }
#endif

            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers_C) {
                int expected_head = -1;
                if (lane_id < kNumRDMARanks_C) {
                    expected_head = ld_acquire_sys_global(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
                    // Normalized semantics: negative = -(next_valid_head)-1, positive = actual wait head.
                    int normalized_wait_head = expected_head < 0 ? -expected_head - 1 : expected_head;
                    rdma_receiver_rdma_head[warp_id][lane_id] = normalized_wait_head;

#ifdef MK_TOKEN_TRACE
                    int rdma_prefix_idx = lane_id * num_logical_channels + logical_channel_id;
                    int rdma_ch_count = state->combine_rdma_channel_token_count[rdma_prefix_idx];
                    printf("[MK-DIAG][RDMA-RECV-HEAD] rank=%d rdma_rank=%d nvl_rank=%d physical_ch=%d logical_ch=%d warp=%d token=%lld src_rdma_lane=%d raw_expected=%d wait_head=%d slot=%d rdma_prefix_idx=%d rdma_ch_count=%d token_range=[%d,%d) cached_tail=%d tail_snapshot=%d tail_ptr=%p combined_rdma_head_addr=%p\n",
                           state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id,
                           warp_id, (long long)token_idx, lane_id, expected_head, normalized_wait_head,
                           expected_head >= 0 ? expected_head % num_max_rdma_chunked_recv_tokens : -1,
                           rdma_prefix_idx, rdma_ch_count, token_start_idx, token_end_idx,
                           cached_channel_tail_idx,
                           static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))),
                           rdma_channel_tail.buffer(lane_id),
                           combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
#endif
                }

                auto start_time = clock64();
                // Wait for RDMA tail (normalized heads: always wait, negative heads skip via large value)
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        if (timeout_log_once(state, kTimeoutLogCombineRdmaReceiver)) {
                            int tail0 = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(0)));
                            int tail1 = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(1)));
                            int head0 = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(0)));
                            int head1 = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(1)));
                            int rdma_ch_count = state->combine_rdma_channel_token_count[lane_id * num_logical_channels + logical_channel_id];
                            printf("MK combine RDMA receiver timeout, rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d ch=%d logical_ch=%d warp=%d token=%lld token_range=[%d,%d) expected_head=%d cached_tail=%d tail_snapshot=%d rdma_tail_ptr=%p lane=%d tail0=%d tail1=%d head0=%d head1=%d rdma_ch_count=%d tail0_ptr=%p tail1_ptr=%p head0_ptr=%p head1_ptr=%p rdma_recv_buffer_lane=%p combined_rdma_head_addr=%p\n",
                                   state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, logical_channel_id,
                                   warp_id, (long long)token_idx,
                                   token_start_idx, token_end_idx, expected_head, cached_channel_tail_idx,
                                   static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))),
                                   (void*)rdma_channel_tail.buffer(lane_id), lane_id, tail0, tail1, head0, head1,
                                   rdma_ch_count,
                                   rdma_channel_tail.buffer(0), rdma_channel_tail.buffer(1),
                                   rdma_channel_head.buffer(0), rdma_channel_head.buffer(1),
                                   rdma_channel_data.recv_buffer(lane_id),
                                   combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
                        }
                        trap();
                    }

                }
                __syncwarp();

                auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                    return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) +
                        hidden_int4_idx;
                };
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
                    return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                                       slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) +
                                        topk_idx);
                };
                uint32_t dummy_tma_phases[2];
                combine_token<kNumRDMARanks_C, true, dtype_t, kNumTopkRDMARanks_C, false, 2>(
                    expected_head >= 0,
                    expected_head,
                    lane_id,
                    hidden_int4,
                    num_topk,
                    combined_x + token_idx * hidden_int4,
                    combined_topk_weights + token_idx * num_topk,
                    nullptr,
                    nullptr,
                    num_max_rdma_chunked_recv_tokens,
                    get_addr_fn,
                    recv_tw_fn,
                    nullptr,
                    dummy_tma_phases);
#ifdef MK_TOKEN_TRACE
                if (lane_id == 0) {
                    printf("[MK-TOKEN][COMBINE-RDMA-RECV] rank=%d token=%lld head=%d ch=%d logical_ch=%d rdma=%d nvl=%d topk0_w=%f h0=%f head0=%d head1=%d normalized=%d combined_x=%p topk_ptr=%p\n",
                           state->rank, (long long)token_idx, expected_head, channel_id, logical_channel_id, rdma_rank, nvl_rank,
                           ld_nc_global(combined_topk_weights + token_idx * num_topk),
                           __bfloat162float(reinterpret_cast<nv_bfloat16*>(combined_x + token_idx * hidden_int4)[0]),
                           ld_acquire_sys_global(combined_rdma_head + token_idx * kNumRDMARanks_C),
                           ld_acquire_sys_global(combined_rdma_head + token_idx * kNumRDMARanks_C + 1),
                           ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]),
                           (void*)(combined_x + token_idx * hidden_int4),
                           (void*)(combined_topk_weights + token_idx * num_topk));
                }
#endif
            }

            __syncwarp();
            // Set INT_MAX before retiring so Coordinator won't be blocked by this warp
            if (lane_id < kNumRDMARanks_C)
                rdma_receiver_rdma_head[warp_id][lane_id] = std::numeric_limits<int>::max();
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;

        } else {
            // ========== Coordinator (internode.cu L2223-2269) ==========
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders_C / kNumRDMARanks_C;

            auto& last_rdma_head = combine_coordinator_last_rdma_head;
            auto& last_nvl_head = combine_coordinator_last_nvl_head;
            int dst_rdma_rank = lane_id < kNumRDMARanks_C ? lane_id : 0;
            int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;

            while (true) {
                if (not is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers_C or rdma_receiver_retired[lane_id]))
                    break;
                if (is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders_C or forwarder_retired[lane_id]))
                    break;

                if (not is_forwarder_sm) {
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers_C; ++i)
                        if (not rdma_receiver_retired[i])
                            min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                        lane_id < kNumRDMARanks_C) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                        min_head - last_rdma_head,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        logical_channel_id + num_logical_channels,
                                                        dst_rdma_rank == rdma_rank);
                        last_rdma_head = min_head;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < kNumRDMARanks_C; ++i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++j)
                            if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);

                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS) {
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
                        }
                    }
                }

                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
    }

    __syncthreads();
    if (thread_id == 0) {
        atomicAdd(&state->combine_channel_barrier[logical_channel_id], 1);
    }
    if (thread_id == 0) {
        auto start_time = clock64();
        while (ld_acquire_sys_global(&state->combine_channel_barrier[logical_channel_id]) < 2) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK combine logical-channel barrier timeout, rank=%d combine_sm_idx=%d physical_ch=%d logical_ch=%d is_forwarder_sm=%d count=%d\n",
                       state->rank, combine_sm_idx, channel_id, logical_channel_id,
                       static_cast<int>(is_forwarder_sm), ld_acquire_sys_global(&state->combine_channel_barrier[logical_channel_id]));
                trap();
            }
            __nanosleep(32);
        }
    }
    __syncthreads();
#ifdef MK_PERF_TRACE
    if (thread_id == 0) {
        int combine_lch_role = is_forwarder_sm ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + combine_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        state->perf_combine_lch_ts[trace_idx + 4] = globaltimer_ns();
    }
#endif
    // if (thread_id == 0)
    //     printf("[MK-DBG][COMBINE][after-logical-barrier] rank=%d block=%d physical_ch=%d logical_ch=%d barrier=%d\n",
    //            state->rank, static_cast<int>(blockIdx.x), channel_id, logical_channel_id,
    //            ld_acquire_sys_global(&state->combine_channel_barrier[logical_channel_id]));

    }
}

// ============================================================================
// Main MegaKernel Entry Point
// ============================================================================

__global__ void __launch_bounds__(kMegaKernelNumThreads, 1) moe_megakernel_v7(
    MegaKernelState* state
) {
    const int sm_id = blockIdx.x;
    const int num_dispatch_sms = state->num_dispatch_sms;
    const int num_combine_sms = state->num_combine_sms;
    const int num_compute_sms = state->num_compute_sms;

    // if (threadIdx.x == 0) {
    //     int rdma_rank = state->rank / NUM_MAX_NVL_PEERS;
    //     int nvl_rank = state->rank % NUM_MAX_NVL_PEERS;
    //     printf("[MK-TRACE][KERNEL][ENTRY] block=%d blocks=%d rank=%d rdma_rank=%d nvl_rank=%d dispatch_sms=%d combine_sms=%d compute_sms=%d num_tokens=%d num_topk=%d hidden_dim=%d intermediate_dim=%d num_experts=%d num_local_experts=%d max_tokens_per_expert=%d expected_dispatch_done=%d state=%p rdma_buffer=%p combine_rdma_buffer=%p buffer_ptrs=%p combine_buffer_ptrs=%p\n",
    //            sm_id, gridDim.x, state->rank, rdma_rank, nvl_rank, num_dispatch_sms, num_combine_sms, num_compute_sms,
    //            state->num_tokens, state->num_topk, state->hidden_dim, state->intermediate_dim,
    //            state->num_experts, state->num_local_experts, state->max_tokens_per_expert,
    //            state->expected_dispatch_done_count, state, state->rdma_buffer_ptr, state->combine_rdma_buffer_ptr,
    //            state->buffer_ptrs, state->combine_buffer_ptrs);
    // }

    // Reuse dynamic shared memory for WMMA output on compute SMs.
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_wmma_buf = reinterpret_cast<float*>(smem_buffer);

    // Determine SM role based on blockIdx.x
    // Layout: [Dispatch] [Combine] [Scheduler] [Compute groups]
    SmRole role;
    int role_idx;

    if (sm_id < num_dispatch_sms) {
        role = SmRole::kDispatch;
        role_idx = sm_id;
    } else if (sm_id < num_dispatch_sms + num_combine_sms) {
        role = SmRole::kCombine;
        role_idx = sm_id - num_dispatch_sms;
    } else if (sm_id < num_dispatch_sms + num_combine_sms + COMPUTE_SCHEDULER_SMS) {
        role = SmRole::kScheduler;
        role_idx = sm_id - num_dispatch_sms - num_combine_sms;
    } else {
        role = SmRole::kCompute;
        role_idx = sm_id - num_dispatch_sms - num_combine_sms - COMPUTE_SCHEDULER_SMS;
    }

    switch (role) {
        case SmRole::kDispatch:
            dispatch_worker_v2(sm_id, role_idx, state);
            break;

        case SmRole::kCombine:
            combine_worker_v2(role_idx, state);
            break;

        case SmRole::kScheduler:
            // S4.2: scheduler region has 2 SMs to keep the grid even (cluster_dim=2),
            // but only scheduler #0 enqueues tasks; #1 idles to avoid double-enqueue.
            if (role_idx == 0)
                compute_scheduler_worker(state);
            break;

        case SmRole::kCompute:
            compute_worker(sm_id, role_idx, num_compute_sms, state, smem_wmma_buf);
            break;

        default:
            break;
    }
}

// ============================================================================
// Host-Side Launch
// ============================================================================

#ifdef MK_PERF_TRACE
static void dump_perf_trace_perfetto(MegaKernelState* device_state, int total_sms);
#endif

void launch_megakernel_v7(
    MegaKernelState* device_state,
    int total_sms,
    int smem_size,
    cudaStream_t stream
) {
    printf("[MK-HOST][LAUNCH] device_state=%p total_sms=%d block_threads=%d smem_size=%d stream=%p\n",
           device_state, total_sms, kMegaKernelNumThreads, smem_size, stream);
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(moe_megakernel_v7,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        printf("[MK-HOST][LAUNCH] set dynamic smem attribute=%d\n", smem_size);
    }

#ifndef DISABLE_SM90_FEATURES
    // S4.2: enable cluster launch (cluster_dim=2) for the whole grid. Compute will
    // later use 2-CTA clusters (2x1SM UMMA); dispatch/combine do not call any cluster
    // API and are unaffected (see MEGAKERNEL_COMPUTE_DESIGN.md I.5/I.9.7-S4.2).
    // cluster_dim divides gridDim only if total_sms is even — assert instead of
    // silently falling back, so an odd SM layout is caught at launch.
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = total_sms;
    cfg.blockDim = kMegaKernelNumThreads;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = stream;

    EP_HOST_ASSERT(total_sms % 2 == 0 && "S4.2: total_sms must be even for cluster_dim=2");

    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = 2;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_megakernel_v7, device_state));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 同步后才能看到 printf 输出
#else
    moe_megakernel_v7<<<total_sms, kMegaKernelNumThreads, smem_size, stream>>>(device_state);
    CUDA_CHECK(cudaGetLastError());

#endif

#ifdef MK_PERF_TRACE
    CUDA_CHECK(cudaStreamSynchronize(stream));
    dump_perf_trace_perfetto(device_state, total_sms);
#endif
}

#ifdef MK_PERF_TRACE
static void dump_perf_trace_perfetto(MegaKernelState* device_state, int total_sms) {
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));

    int num_logical_channels = host_state.num_logical_channels;
    constexpr int NLP = MegaKernelState::MK_PERF_NUM_LCH_PHASES;

    std::vector<int64_t> dispatch_lch_ts(num_logical_channels * 2 * NLP);
    std::vector<int64_t> combine_lch_ts(num_logical_channels * 2 * NLP);
    CUDA_CHECK(cudaMemcpy(dispatch_lch_ts.data(), host_state.perf_dispatch_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(combine_lch_ts.data(), host_state.perf_combine_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Accumulated semaphore-interaction timers (rendered as args on existing rows).
    std::vector<int64_t> disp_wait_nvl(num_logical_channels * 2);
    std::vector<int64_t> disp_publish(num_logical_channels * 2);
    std::vector<int64_t> disp_wait_recvcount(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready(num_logical_channels * 2);
    CUDA_CHECK(cudaMemcpy(disp_wait_nvl.data(), host_state.perf_disp_wait_nvl_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_publish.data(), host_state.perf_disp_publish_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_wait_recvcount.data(), host_state.perf_disp_wait_recvcount_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready.data(), host_state.perf_comb_wait_ready_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // publish breakdown timers.
    std::vector<int64_t> disp_pub_scan(num_logical_channels * 2);
    std::vector<int64_t> disp_pub_atomic(num_logical_channels * 2);
    std::vector<int64_t> disp_pub_fence(num_logical_channels * 2);
    std::vector<int64_t> disp_pub_store(num_logical_channels * 2);
    CUDA_CHECK(cudaMemcpy(disp_pub_scan.data(), host_state.perf_disp_pub_scan_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_pub_atomic.data(), host_state.perf_disp_pub_atomic_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_pub_fence.data(), host_state.perf_disp_pub_fence_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_pub_store.data(), host_state.perf_disp_pub_store_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Per-compute-task records.
    constexpr int NCF = MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS;
    int compute_task_count = 0;
    CUDA_CHECK(cudaMemcpy(&compute_task_count, host_state.perf_compute_task_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (compute_task_count > host_state.max_compute_tasks)
        compute_task_count = host_state.max_compute_tasks;
    std::vector<int64_t> compute_task(compute_task_count > 0 ? (size_t)compute_task_count * NCF : 1);
    if (compute_task_count > 0)
        CUDA_CHECK(cudaMemcpy(compute_task.data(), host_state.perf_compute_task, (size_t)compute_task_count * NCF * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Root-cause diagnostic parallel arrays: full UMMA breakdown + multi-expert rows.
    const int diag_n = compute_task_count > 0 ? compute_task_count : 1;
    std::vector<int64_t> d_up_setup(diag_n), d_up_tmem(diag_n), d_up_prologue(diag_n),
        d_up_tma(diag_n), d_up_mma_issue(diag_n), d_up_mma_wait(diag_n),
        d_up_loop(diag_n), d_up_csync(diag_n), d_up_epi(diag_n);
    std::vector<int64_t> d_dn_setup(diag_n), d_dn_tmem(diag_n), d_dn_prologue(diag_n),
        d_dn_tma(diag_n), d_dn_mma_issue(diag_n), d_dn_mma_wait(diag_n),
        d_dn_loop(diag_n), d_dn_csync(diag_n), d_dn_epi(diag_n);
    std::vector<int> diag_multi_rows(diag_n);
    if (compute_task_count > 0) {
        const size_t b = (size_t)compute_task_count * sizeof(int64_t);
        auto cp = [&](std::vector<int64_t>& v, int64_t* src) {
            CUDA_CHECK(cudaMemcpy(v.data(), src, b, cudaMemcpyDeviceToHost));
        };
        cp(d_up_setup, host_state.perf_up_setup);     cp(d_up_tmem, host_state.perf_up_tmem_alloc);
        cp(d_up_prologue, host_state.perf_up_prologue); cp(d_up_tma, host_state.perf_up_tma_wait);
        cp(d_up_mma_issue, host_state.perf_up_mma_issue); cp(d_up_mma_wait, host_state.perf_up_mma_wait);
        cp(d_up_loop, host_state.perf_up_loop_other); cp(d_up_csync, host_state.perf_up_cluster_sync);
        cp(d_up_epi, host_state.perf_up_epilogue);
        cp(d_dn_setup, host_state.perf_down_setup);   cp(d_dn_tmem, host_state.perf_down_tmem_alloc);
        cp(d_dn_prologue, host_state.perf_down_prologue); cp(d_dn_tma, host_state.perf_down_tma_wait);
        cp(d_dn_mma_issue, host_state.perf_down_mma_issue); cp(d_dn_mma_wait, host_state.perf_down_mma_wait);
        cp(d_dn_loop, host_state.perf_down_loop_other); cp(d_dn_csync, host_state.perf_down_cluster_sync);
        cp(d_dn_epi, host_state.perf_down_epilogue);
        CUDA_CHECK(cudaMemcpy(diag_multi_rows.data(), host_state.perf_compute_multi_expert_rows, (size_t)compute_task_count * sizeof(int), cudaMemcpyDeviceToHost));
    }

    int64_t base_ts = std::numeric_limits<int64_t>::max();
    for (int i = 0; i < num_logical_channels * 2 * NLP; ++i) {
        int64_t trace_ts = dispatch_lch_ts[i];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    for (int i = 0; i < num_logical_channels * 2 * NLP; ++i) {
        int64_t trace_ts = combine_lch_ts[i];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    if (base_ts == std::numeric_limits<int64_t>::max()) base_ts = 0;

    // globaltimer_ns() already returns nanoseconds; convert to us for Perfetto.
    // Use per-rank min as base so trace starts near 0; for cross-rank comparison,
    // use aggregate_mk_perf_traces.py which aligns rank baselines.

    char filename[256];
    snprintf(filename, sizeof(filename), "mk_perf_trace_rank%d.json", host_state.rank);
    FILE* f = fopen(filename, "w");
    if (!f) { printf("[MK-PERF] Failed to open %s\n", filename); return; }

    // Also write raw base_ts so post-processing can re-align ranks
    printf("[MK-PERF] rank=%d base_ts_ns=%lld\n", host_state.rank, (long long)base_ts);

    fprintf(f, "[\n");
    bool first = true;
    auto emit_comma = [&]() {
        if (!first) fprintf(f, ",\n");
        first = false;
    };
    auto emit_event = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid) {
        if (start == 0 || end == 0 || end <= start) return;
        // Output in nanoseconds (Perfetto supports "displayTimeUnit":"ns" but default is us)
        // We use microseconds with fractional part via integer division
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid);
    };
    // Same as emit_event, but attaches semaphore-interaction breakdown (us) as args.
    auto emit_event_sema = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid,
                               int64_t wait_a_ns, const char* key_a,
                               int64_t wait_b_ns, const char* key_b,
                               int64_t wait_c_ns, const char* key_c) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"%s_us\":%.3f,\"%s_us\":%.3f,\"%s_us\":%.3f}}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid,
                key_a, wait_a_ns / 1000.0, key_b, wait_b_ns / 1000.0, key_c, wait_c_ns / 1000.0);
    };
    // Dispatch sender/forwarder work event with full publish breakdown as args (us).
    auto emit_event_publish = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid,
                                  int64_t wait_nvl_ns, int64_t publish_ns, int64_t wait_recvcount_ns,
                                  int64_t scan_ns, int64_t atomic_ns, int64_t fence_ns, int64_t store_ns) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"wait_nvl_us\":%.3f,\"publish_us\":%.3f,\"wait_recvcount_us\":%.3f,"
                   "\"pub_scan_us\":%.3f,\"pub_atomic_us\":%.3f,\"pub_fence_us\":%.3f,\"pub_store_us\":%.3f}}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid,
                wait_nvl_ns / 1000.0, publish_ns / 1000.0, wait_recvcount_ns / 1000.0,
                scan_ns / 1000.0, atomic_ns / 1000.0, fence_ns / 1000.0, store_ns / 1000.0);
    };

    // Process/thread metadata
    emit_comma();
    fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":%d,\"args\":{\"name\":\"rank %d\"}}",
            host_state.rank, host_state.rank);
    // Embed base_ts_ns for cross-rank alignment by aggregate script
    emit_comma();
    fprintf(f, "{\"name\":\"mk_base_ts_ns\",\"ph\":\"M\",\"pid\":%d,\"args\":{\"base_ts_ns\":%lld}}",
            host_state.rank, (long long)base_ts);

    int lch_tid_base = 0;
    for (int logical_channel_id = 0; logical_channel_id < num_logical_channels; ++logical_channel_id) {
        int dispatch_sender_tid = lch_tid_base + logical_channel_id * 4;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"dispatch_sender_lch_%d\"}}",
                host_state.rank, dispatch_sender_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, dispatch_sender_tid, dispatch_sender_tid);

        int dispatch_forwarder_tid = dispatch_sender_tid + 1;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"dispatch_forwarder_lch_%d\"}}",
                host_state.rank, dispatch_forwarder_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, dispatch_forwarder_tid, dispatch_forwarder_tid);

        int combine_sender_tid = dispatch_sender_tid + 2;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"combine_sender_lch_%d\"}}",
                host_state.rank, combine_sender_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, combine_sender_tid, combine_sender_tid);

        int combine_forwarder_tid = dispatch_sender_tid + 3;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"combine_forwarder_lch_%d\"}}",
                host_state.rank, combine_forwarder_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, combine_forwarder_tid, combine_forwarder_tid);
    }

    int pid = host_state.rank;
    for (int logical_channel_id = 0; logical_channel_id < num_logical_channels; ++logical_channel_id) {
        int dispatch_sender_tid = lch_tid_base + logical_channel_id * 4;
        int64_t* dsp = &dispatch_lch_ts[(logical_channel_id * 2) * NLP];
        emit_event_publish("dispatch_sender_work", "dispatch_sender_lch", dsp[0], dsp[1], pid, dispatch_sender_tid,
                           disp_wait_nvl[logical_channel_id * 2 + 0], disp_publish[logical_channel_id * 2 + 0],
                           disp_wait_recvcount[logical_channel_id * 2 + 0],
                           disp_pub_scan[logical_channel_id * 2 + 0], disp_pub_atomic[logical_channel_id * 2 + 0],
                           disp_pub_fence[logical_channel_id * 2 + 0], disp_pub_store[logical_channel_id * 2 + 0]);
        emit_event("dispatch_sender_channel_barrier", "dispatch_sender_lch", dsp[1], dsp[2], pid, dispatch_sender_tid);
        emit_event("dispatch_sender_round_barrier", "dispatch_sender_lch", dsp[2], dsp[3], pid, dispatch_sender_tid);

        int dispatch_forwarder_tid = dispatch_sender_tid + 1;
        int64_t* dfp = &dispatch_lch_ts[(logical_channel_id * 2 + 1) * NLP];
        emit_event_publish("dispatch_forwarder_work", "dispatch_forwarder_lch", dfp[0], dfp[1], pid, dispatch_forwarder_tid,
                           disp_wait_nvl[logical_channel_id * 2 + 1], disp_publish[logical_channel_id * 2 + 1],
                           disp_wait_recvcount[logical_channel_id * 2 + 1],
                           disp_pub_scan[logical_channel_id * 2 + 1], disp_pub_atomic[logical_channel_id * 2 + 1],
                           disp_pub_fence[logical_channel_id * 2 + 1], disp_pub_store[logical_channel_id * 2 + 1]);
        emit_event("dispatch_forwarder_channel_barrier", "dispatch_forwarder_lch", dfp[1], dfp[2], pid, dispatch_forwarder_tid);
        emit_event("dispatch_forwarder_round_barrier", "dispatch_forwarder_lch", dfp[2], dfp[3], pid, dispatch_forwarder_tid);

        int combine_sender_tid = dispatch_sender_tid + 2;
        int64_t* csp = &combine_lch_ts[(logical_channel_id * 2) * NLP];
        emit_event("sender_wait_dispatch_done", "combine_sender_lch", csp[0], csp[1], pid, combine_sender_tid);
        emit_event("sender_head_normalize", "combine_sender_lch", csp[1], csp[2], pid, combine_sender_tid);
        emit_event_sema("sender_nvl_send_rdma_recv", "combine_sender_lch", csp[3], csp[4], pid, combine_sender_tid,
                        comb_wait_ready[logical_channel_id * 2 + 0], "wait_compute_ready",
                        0, "unused_b",
                        0, "unused_c");

        int combine_forwarder_tid = dispatch_sender_tid + 3;
        int64_t* cfp = &combine_lch_ts[(logical_channel_id * 2 + 1) * NLP];
        emit_event("forwarder_wait_normalized", "combine_forwarder_lch", cfp[0], cfp[2], pid, combine_forwarder_tid);
        emit_event("forwarder_nvl_to_rdma", "combine_forwarder_lch", cfp[3], cfp[4], pid, combine_forwarder_tid);
    }

    // Compute task rows: one Perfetto row per compute group, one X-event per task batch.
    int compute_tid_base = num_logical_channels * 4 + 100;
    int num_compute_groups = host_state.num_compute_sms / COMPUTE_GROUP_SIZE;
    for (int g = 0; g < num_compute_groups; ++g) {
        int tid = compute_tid_base + g;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"compute_group_%d\"}}",
                host_state.rank, tid, g);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, tid, tid);
    }
    for (int t = 0; t < compute_task_count; ++t) {
        int64_t* rec = &compute_task[(size_t)t * NCF];
        int64_t start = rec[0];
        int64_t end = rec[1];
        if (start == 0 || end == 0 || end <= start) continue;
        int sm_id = static_cast<int>(rec[2]);
        int group_id = static_cast<int>(rec[3]);
        int expert_id = static_cast<int>(rec[4]);
        int batch_size = static_cast<int>(rec[5]);
        int hidden = static_cast<int>(rec[6]);
        int intermediate = static_cast<int>(rec[7]);
        int tid = compute_tid_base + group_id;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        // Phase boundary timestamps (0 if not captured); derive per-phase durations (us).
        int64_t ph_meta = rec[8], ph_input = rec[9], ph_upgemm = rec[10];
        int64_t ph_downgemm = rec[11], ph_output = rec[12], ph_signal = rec[13];
        // Finer breakpoints to separate GEMM compute vs group-barrier wait, and split signaling.
        int64_t up_body = rec[14], down_body = rec[15], out_body = rec[16];
        int64_t sig_donecount = rec[17], sig_finalize = rec[18];
        int64_t sig_fence = rec[19], sig_publish = rec[20];
        auto phase_us = [](int64_t a, int64_t b) -> double {
            if (a == 0 || b == 0 || b <= a) return 0.0;
            return (b - a) / 1000.0;
        };
        double meta_us     = phase_us(start, ph_meta);
        double input_us    = phase_us(ph_meta, ph_input);
        double upgemm_us   = phase_us(ph_input, ph_upgemm);
        double downgemm_us = phase_us(ph_upgemm, ph_downgemm);
        double output_us   = phase_us(ph_downgemm, ph_output);
        double signal_us   = phase_us(ph_output, ph_signal);
        // p3 split: actual up GEMM body vs barrier wait after it.
        double up_compute_us  = phase_us(ph_input, up_body);
        double up_barrier_us   = phase_us(up_body, ph_upgemm);
        // p4 split: actual down GEMM body vs barrier wait after it.
        double down_compute_us = phase_us(ph_upgemm, down_body);
        double down_barrier_us = phase_us(down_body, ph_downgemm);
        // p5 (output) split: write loop vs barrier+fence after it.
        double out_compute_us  = phase_us(ph_downgemm, out_body);
        double out_barrier_us  = phase_us(out_body, ph_output);
        // p6 split: done-count, fp32 finalize, fence, ready-publish sub-phases.
        double sig_donecount_us = phase_us(ph_output, sig_donecount);
        double sig_finalize_us  = phase_us(sig_donecount, sig_finalize);
        double sig_fence_us     = phase_us(sig_finalize, sig_fence);
        double sig_publish_us   = phase_us(sig_fence, sig_publish);
        // Root-cause diagnostics: FULL UMMA per-phase breakdown (us) for up/down GEMM,
        // so the dominant sub-step inside p3a/p4a is never a blind spot.
        auto ns_us = [](int64_t v) -> double { return v > 0 ? v / 1000.0 : 0.0; };
        int multi_expert_rows = diag_multi_rows[t];
        emit_comma();
        fprintf(f, "{\"name\":\"compute_e%d\",\"cat\":\"compute_group\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,"
                   "\"pid\":%d,\"tid\":%d,\"args\":{\"expert_id\":%d,\"sm_id\":%d,\"group_id\":%d,\"batch_size\":%d,"
                   "\"hidden_size\":%d,\"intermediate_size\":%d,"
                   "\"p1_meta_us\":%.3f,\"p2_input_load_us\":%.3f,\"p3_gateup_gemm_us\":%.3f,"
                   "\"p4_down_gemm_us\":%.3f,\"p5_output_us\":%.3f,\"p6_signal_us\":%.3f,"
                   "\"p3a_up_compute_us\":%.3f,\"p3b_up_barrier_us\":%.3f,"
                   "\"p4a_down_compute_us\":%.3f,\"p4b_down_barrier_us\":%.3f,"
                   "\"p5a_out_compute_us\":%.3f,\"p5b_out_barrier_us\":%.3f,"
                   "\"p6a_donecount_us\":%.3f,\"p6b_fp32finalize_us\":%.3f,"
                   "\"p6c_fence_us\":%.3f,\"p6d_publish_us\":%.3f,"
                   "\"up_setup_us\":%.3f,\"up_tmem_alloc_us\":%.3f,\"up_prologue_us\":%.3f,"
                   "\"up_tma_wait_us\":%.3f,\"up_mma_issue_us\":%.3f,\"up_mma_wait_us\":%.3f,"
                   "\"up_loop_other_us\":%.3f,\"up_cluster_sync_us\":%.3f,\"up_epilogue_us\":%.3f,"
                   "\"down_setup_us\":%.3f,\"down_tmem_alloc_us\":%.3f,\"down_prologue_us\":%.3f,"
                   "\"down_tma_wait_us\":%.3f,\"down_mma_issue_us\":%.3f,\"down_mma_wait_us\":%.3f,"
                   "\"down_loop_other_us\":%.3f,\"down_cluster_sync_us\":%.3f,\"down_epilogue_us\":%.3f,"
                   "\"p6b_multi_expert_rows\":%d}}",
                expert_id, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid, expert_id, sm_id, group_id, batch_size,
                hidden, intermediate,
                meta_us, input_us, upgemm_us, downgemm_us, output_us, signal_us,
                up_compute_us, up_barrier_us, down_compute_us, down_barrier_us,
                out_compute_us, out_barrier_us,
                sig_donecount_us, sig_finalize_us, sig_fence_us, sig_publish_us,
                ns_us(d_up_setup[t]), ns_us(d_up_tmem[t]), ns_us(d_up_prologue[t]),
                ns_us(d_up_tma[t]), ns_us(d_up_mma_issue[t]), ns_us(d_up_mma_wait[t]),
                ns_us(d_up_loop[t]), ns_us(d_up_csync[t]), ns_us(d_up_epi[t]),
                ns_us(d_dn_setup[t]), ns_us(d_dn_tmem[t]), ns_us(d_dn_prologue[t]),
                ns_us(d_dn_tma[t]), ns_us(d_dn_mma_issue[t]), ns_us(d_dn_mma_wait[t]),
                ns_us(d_dn_loop[t]), ns_us(d_dn_csync[t]), ns_us(d_dn_epi[t]),
                multi_expert_rows);
    }

    fprintf(f, "\n]\n");
    fclose(f);
    printf("[MK-PERF] Perfetto trace written to %s (%d logical channels)\n", filename, num_logical_channels);
}
#endif

MegaKernelState* allocate_megakernel_state_v7(
    // --- Dispatch input data (from PyTorch tensors) ---
    const int4* x,
    const float* x_scales,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    // --- Dispatch prefix matrices (from notify_dispatch) ---
    const int* rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* recv_gbl_rank_prefix_sum,
    // --- Buffer infrastructure ---
    void* rdma_buffer_ptr,
    void** buffer_ptrs,
    void** combine_buffer_ptrs,
    // --- Dimensions ---
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_scales,
    int num_topk,
    int num_experts,
    int num_local_experts,
    int num_ranks,
    int rank,
    int scale_token_stride,
    int scale_hidden_stride,
    // --- Buffer sizing ---
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    // --- Expert weights ---
    const __nv_bfloat16* W_gate,
    const __nv_bfloat16* W_up,
    const __nv_bfloat16* W_down,
    // --- SM config ---
    int num_dispatch_sms,
    int num_forwarder_sms,
    int num_compute_sms,
    int num_combine_sms,
    int num_logical_channels,
    // --- Max token budget ---
    int max_tokens_per_expert,
    int max_total_recv_tokens,
    // --- Buffer sizes for combine mirror ---
    int64_t num_rdma_bytes,
    int64_t num_nvl_bytes
) {
    // Allocate workspace buffers on device
    int* expert_recv_count;
    int* dispatch_done;
    int* dispatch_done_count;
    int* timeout_log_counters;
    __nv_bfloat16* recv_tokens;
    int* expert_token_offsets;
    int* recv_token_source_info;
    float* recv_token_route_weights;
    internode::SourceMeta* recv_src_meta;
    int* compute_done_count;
    int* expert_compute_cursor;
    int* compute_group_barrier;
    int* compute_group_phase;
    ComputeTask* compute_tasks;
    int* compute_task_head;
    int* compute_task_tail;
    int* compute_enqueue_done;
    int* expert_enqueue_cursor;
    int* compute_group_task_idx;
    __nv_bfloat16* compute_output;
    __nv_bfloat16* combine_input;
    float* combine_input_topk_weights;
    internode::SourceMeta* combine_input_src_meta;
    int* combine_notify_done;
    int* combine_rdma_head_work;
    int* combine_nvl_head_work;
    __nv_bfloat16* gemm_workspace;
    float* output_accum;
    int* send_rdma_head;
    int* send_nvl_head;
    int* recv_rdma_channel_prefix_matrix;
    int* recv_gbl_channel_prefix_matrix;

    const int hidden_int4 = hidden_dim * sizeof(__nv_bfloat16) / sizeof(int4);

    // Mirror DeepEP host-side launch invariants before allocating state. These
    // protect the producer/consumer queue geometry used by dispatch and combine.
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());
    EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    EP_HOST_ASSERT(num_rdma_ranks == MK_NUM_RDMA_RANKS);

    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
    EP_HOST_ASSERT(num_rdma_ranks <= kNumCombineForwarderWarps);
    EP_HOST_ASSERT(num_forwarder_warps > NUM_MAX_NVL_PEERS and num_forwarder_warps % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks >
                   std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks - num_warps_per_forwarder >= num_max_nvl_chunked_send_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);

    // Signaling
    CUDA_CHECK(cudaMalloc(&expert_recv_count, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_recv_count, 0, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&timeout_log_counters, kTimeoutLogCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(timeout_log_counters, 0, kTimeoutLogCount * sizeof(int)));

    // Receive storage — indexed as [local_expert_id * max_tokens_per_expert + slot]
    const size_t total_expert_slots = (size_t)num_local_experts * max_tokens_per_expert;

    int* expert_slot_ready;
    CUDA_CHECK(cudaMalloc(&expert_slot_ready, total_expert_slots * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_slot_ready, 0, total_expert_slots * sizeof(int)));
    size_t recv_tokens_bytes = total_expert_slots * hidden_dim * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&recv_tokens, recv_tokens_bytes));

    CUDA_CHECK(cudaMalloc(&expert_token_offsets, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_token_offsets, 0, num_local_experts * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&recv_token_source_info, total_expert_slots * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_token_source_info, 0xff, total_expert_slots * 2 * sizeof(int)));  // Init to -1
    CUDA_CHECK(cudaMalloc(&recv_token_route_weights, total_expert_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_src_meta, total_expert_slots * sizeof(internode::SourceMeta)));

    // Compute state
    CUDA_CHECK(cudaMalloc(&compute_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_compute_cursor, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_compute_cursor, 0, num_local_experts * sizeof(int)));
    int num_compute_groups = num_compute_sms / COMPUTE_GROUP_SIZE;
    EP_HOST_ASSERT(num_compute_groups > 0);
    EP_HOST_ASSERT(num_compute_sms == num_compute_groups * COMPUTE_GROUP_SIZE);
    CUDA_CHECK(cudaMalloc(&compute_group_barrier, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_barrier, 0, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_group_phase, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_phase, 0, num_compute_groups * sizeof(int)));

    int max_compute_tasks = num_local_experts * (max_tokens_per_expert / COMPUTE_BATCH_SIZE + 2);
    CUDA_CHECK(cudaMalloc(&compute_tasks, (size_t)max_compute_tasks * sizeof(ComputeTask)));
    CUDA_CHECK(cudaMalloc(&compute_task_head, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_task_head, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_task_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_task_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_enqueue_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_enqueue_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_enqueue_cursor, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_enqueue_cursor, 0, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_group_task_idx, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_task_idx, 0xff, num_compute_groups * sizeof(int)));

    // Combine per-expert completion signals
    int* expert_compute_done;
    CUDA_CHECK(cudaMalloc(&expert_compute_done, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_compute_done, 0, num_local_experts * sizeof(int)));

    // Compute output: same shape as recv_tokens
    CUDA_CHECK(cudaMalloc(&compute_output, recv_tokens_bytes));
    CUDA_CHECK(cudaMemset(compute_output, 0, recv_tokens_bytes));  // Zero for tokens with no local expert
    CUDA_CHECK(cudaMalloc(&combine_input, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(combine_input, 0, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&combine_input_topk_weights, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(combine_input_topk_weights, 0, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&combine_input_src_meta, (size_t)max_total_recv_tokens * sizeof(internode::SourceMeta)));
    CUDA_CHECK(cudaMalloc(&combine_notify_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_notify_done, 0, sizeof(int)));

    // Per-token compute signaling
    int* token_compute_expected;
    int* token_compute_done;
    int* combine_token_ready;
    float* compute_output_f;
    CUDA_CHECK(cudaMalloc(&token_compute_expected, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_compute_expected, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&token_compute_done, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_compute_done, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_token_ready, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_token_ready, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_output_f, (size_t)max_total_recv_tokens * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(compute_output_f, 0, (size_t)max_total_recv_tokens * hidden_dim * sizeof(float)));

    // GEMM workspace: per-compute-group batched intermediates for M=128 compute batches.
    size_t per_group_elems = (size_t)COMPUTE_BATCH_SIZE * (2 * hidden_dim + 2 * intermediate_dim);
    size_t workspace_bytes = num_compute_groups * per_group_elems * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&gemm_workspace, workspace_bytes));

    // --- S4.4 (route B2): build UMMA compute TMA atoms on host, upload to device ---
    // Only when hidden==intermediate (first version, I.9.0) and weights present.
    umma::ComputeTmaAtoms* d_compute_tma = nullptr;
    umma::InputTmaAtom_t* d_group_input_tma = nullptr;
    umma::ComputeDownTmaAtoms* d_compute_down_tma = nullptr;
    if (W_gate != nullptr && W_up != nullptr && hidden_dim == intermediate_dim &&
        num_local_experts <= umma::kMaxLocalExperts) {
        // Per-expert weight atoms.
        umma::ComputeTmaAtoms h_atoms;
        umma::build_compute_tma_atoms(h_atoms, W_gate, W_up, num_local_experts,
                                      intermediate_dim, hidden_dim);
        CUDA_CHECK(cudaMalloc(&d_compute_tma, sizeof(umma::ComputeTmaAtoms)));
        CUDA_CHECK(cudaMemcpy(d_compute_tma, &h_atoms, sizeof(umma::ComputeTmaAtoms), cudaMemcpyHostToDevice));

        // Per-group A(input_buf) + gate/act/down workspace atoms. Layout per group:
        //   [input_buf (M*hidden)] [gate_buf (M*I)] [up_buf/act (M*I)] [down_buf (M*hidden)]
        std::vector<umma::InputTmaAtom_t> h_in;
        h_in.reserve(num_compute_groups);
        for (int g = 0; g < num_compute_groups; ++g) {
            const __nv_bfloat16* in_g = gemm_workspace + (size_t)g * per_group_elems;
            const __nv_bfloat16* gate_g = in_g + (size_t)COMPUTE_BATCH_SIZE * hidden_dim;
            const __nv_bfloat16* act_g  = gate_g + (size_t)COMPUTE_BATCH_SIZE * intermediate_dim;
            const __nv_bfloat16* down_g = act_g + (size_t)COMPUTE_BATCH_SIZE * intermediate_dim;
            h_in.push_back(umma::make_input_group_atoms(in_g, gate_g, act_g, down_g,
                                                        COMPUTE_BATCH_SIZE, hidden_dim,
                                                        intermediate_dim, hidden_dim));
        }
        CUDA_CHECK(cudaMalloc(&d_group_input_tma, num_compute_groups * sizeof(umma::InputTmaAtom_t)));
        CUDA_CHECK(cudaMemcpy(d_group_input_tma, h_in.data(),
                              num_compute_groups * sizeof(umma::InputTmaAtom_t), cudaMemcpyHostToDevice));

        // Per-expert W_down raw TMA descriptors (DeepGEMM path; A/CD live in InputTmaAtom_t).
        umma::ComputeDownTmaAtoms h_down;
        umma::build_compute_down_tma_atoms(h_down, W_down, num_local_experts, hidden_dim, intermediate_dim);
        CUDA_CHECK(cudaMalloc(&d_compute_down_tma, sizeof(umma::ComputeDownTmaAtoms)));
        CUDA_CHECK(cudaMemcpy(d_compute_down_tma, &h_down, sizeof(umma::ComputeDownTmaAtoms), cudaMemcpyHostToDevice));
    }

    // Output accumulator [num_tokens, hidden_dim] in float32
    CUDA_CHECK(cudaMalloc(&output_accum, (size_t)num_tokens * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(output_accum, 0, (size_t)num_tokens * hidden_dim * sizeof(float)));

    // Dispatch tracking heads. Each logical channel owns an independent DeepEP-shaped head space.
    const int combine_rdma_head_stride = num_tokens * num_rdma_ranks;
    // In megakernel we don't have num_rdma_recv_tokens at alloc time, use num_tokens * num_topk as upper bound.
    int num_rdma_recv_tokens_ub = num_tokens * num_topk;
    const int combine_nvl_head_stride = num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS;
    CUDA_CHECK(cudaMalloc(&send_rdma_head, (size_t)num_logical_channels * combine_rdma_head_stride * sizeof(int)));
    CUDA_CHECK(cudaMemset(send_rdma_head, 0xFF, (size_t)num_logical_channels * combine_rdma_head_stride * sizeof(int)));  // Init to -1
    CUDA_CHECK(cudaMalloc(&send_nvl_head, (size_t)num_logical_channels * combine_nvl_head_stride * sizeof(int)));
    CUDA_CHECK(cudaMemset(send_nvl_head, 0xFF, (size_t)num_logical_channels * combine_nvl_head_stride * sizeof(int)));  // Init to -1
    CUDA_CHECK(cudaMalloc(&combine_rdma_head_work, (size_t)num_logical_channels * combine_rdma_head_stride * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_rdma_head_work, 0, (size_t)num_logical_channels * combine_rdma_head_stride * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_nvl_head_work, (size_t)num_logical_channels * combine_nvl_head_stride * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_nvl_head_work, 0, (size_t)num_logical_channels * combine_nvl_head_stride * sizeof(int)));

    // Recv logical-channel prefix matrices (written by forwarder)
    int num_physical_channels = num_dispatch_sms / 2;  // even/odd pairing
    int num_combine_channels = num_combine_sms / 2;
    EP_HOST_ASSERT(num_combine_channels == num_physical_channels);
    EP_HOST_ASSERT(num_logical_channels >= num_physical_channels);

    printf("num_tokens: %d, um_rdma_ranks: %d, num_physical_channels: %d, num_logical_channels: %d\n",
           num_tokens, num_rdma_ranks, num_physical_channels, num_logical_channels);

    // Per-logical-channel overlap signaling
    int* channel_dispatch_done;
    int* channel_normalized;
    int* dispatch_channel_barrier;
    int* dispatch_round_barrier;
    int* combine_channel_barrier;
    CUDA_CHECK(cudaMalloc(&channel_dispatch_done, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(channel_dispatch_done, 0, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&channel_normalized, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(channel_normalized, 0, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_channel_barrier, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_channel_barrier, 0, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_round_barrier, ((num_logical_channels + num_physical_channels - 1) / num_physical_channels) * sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_round_barrier, 0, ((num_logical_channels + num_physical_channels - 1) / num_physical_channels) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_channel_barrier, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_channel_barrier, 0, num_logical_channels * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&recv_rdma_channel_prefix_matrix, num_rdma_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_rdma_channel_prefix_matrix, 0, num_rdma_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_gbl_channel_prefix_matrix, num_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_gbl_channel_prefix_matrix, 0, num_ranks * num_logical_channels * sizeof(int)));

    // Per-logical-channel token counts (non-cumulative) for overlap
    int* recv_rdma_channel_token_count;
    int* recv_gbl_channel_token_count;
    CUDA_CHECK(cudaMalloc(&recv_rdma_channel_token_count, num_rdma_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_rdma_channel_token_count, 0, num_rdma_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_gbl_channel_token_count, num_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_gbl_channel_token_count, 0, num_ranks * num_logical_channels * sizeof(int)));

    // Build host-side state and copy to device
    MegaKernelState host_state;
    memset(&host_state, 0, sizeof(host_state));

    // NVSHMEM infra
    host_state.rdma_buffer_ptr = rdma_buffer_ptr;
    host_state.buffer_ptrs = buffer_ptrs;

    // Dispatch input
    host_state.x = x;
    host_state.x_scales = x_scales;
    host_state.topk_idx = topk_idx;
    host_state.topk_weights = topk_weights;
    host_state.is_token_in_rank = is_token_in_rank;

    // Dispatch metadata
    host_state.rdma_channel_prefix_matrix = rdma_channel_prefix_matrix;
    host_state.recv_rdma_rank_prefix_sum = recv_rdma_rank_prefix_sum;
    host_state.gbl_channel_prefix_matrix = gbl_channel_prefix_matrix;
    host_state.recv_gbl_rank_prefix_sum = recv_gbl_rank_prefix_sum;
    host_state.send_rdma_head = send_rdma_head;
    host_state.send_nvl_head = send_nvl_head;
    host_state.recv_rdma_channel_prefix_matrix = recv_rdma_channel_prefix_matrix;
    host_state.recv_gbl_channel_prefix_matrix = recv_gbl_channel_prefix_matrix;
    host_state.recv_rdma_channel_token_count = recv_rdma_channel_token_count;
    host_state.recv_gbl_channel_token_count = recv_gbl_channel_token_count;

    // Dimensions
    host_state.num_tokens = num_tokens;
    host_state.hidden_int4 = hidden_int4;
    host_state.num_scales = num_scales;
    host_state.num_topk = num_topk;
    host_state.num_experts = num_experts;
    host_state.scale_token_stride = scale_token_stride;
    host_state.scale_hidden_stride = scale_hidden_stride;

    // Buffer sizing
    host_state.num_max_rdma_chunked_send_tokens = num_max_rdma_chunked_send_tokens;
    host_state.num_max_rdma_chunked_recv_tokens = num_max_rdma_chunked_recv_tokens;
    host_state.num_max_nvl_chunked_send_tokens = num_max_nvl_chunked_send_tokens;
    host_state.num_max_nvl_chunked_recv_tokens = num_max_nvl_chunked_recv_tokens;

    // Topology
    host_state.rank = rank;
    host_state.num_ranks = num_ranks;

    // Receive-side signaling
    host_state.expert_recv_count = expert_recv_count;
    host_state.expert_slot_ready = expert_slot_ready;
    host_state.dispatch_done = dispatch_done;
    host_state.dispatch_done_count = dispatch_done_count;
    host_state.timeout_log_counters = timeout_log_counters;
    host_state.expected_dispatch_done_count = num_logical_channels * NUM_MAX_NVL_PEERS;

    // Per-expert receive storage
    host_state.recv_tokens = recv_tokens;
    host_state.expert_token_offsets = expert_token_offsets;
    host_state.recv_token_source_info = recv_token_source_info;
    host_state.recv_token_route_weights = recv_token_route_weights;
    host_state.recv_src_meta = recv_src_meta;

    // Compute state
    host_state.compute_done_count = compute_done_count;
    host_state.expert_compute_cursor = expert_compute_cursor;
    host_state.compute_group_barrier = compute_group_barrier;
    host_state.compute_group_phase = compute_group_phase;
    host_state.compute_tasks = compute_tasks;
    host_state.max_compute_tasks = max_compute_tasks;
    host_state.compute_task_head = compute_task_head;
    host_state.compute_task_tail = compute_task_tail;
    host_state.compute_enqueue_done = compute_enqueue_done;
    host_state.expert_enqueue_cursor = expert_enqueue_cursor;
    host_state.compute_group_task_idx = compute_group_task_idx;

    // Expert weights
    host_state.W_gate = W_gate;
    host_state.W_up = W_up;
    host_state.W_down = W_down;

    // Compute output
    host_state.compute_output = compute_output;
    host_state.combine_input = combine_input;
    host_state.combine_input_topk_weights = combine_input_topk_weights;
    host_state.combine_input_src_meta = combine_input_src_meta;
    host_state.combine_notify_done = combine_notify_done;
    host_state.combine_rdma_head_work = combine_rdma_head_work;
    host_state.combine_nvl_head_work = combine_nvl_head_work;
    host_state.gemm_workspace = gemm_workspace;
    host_state.output_accum = output_accum;
    // S4.4 (route B2): UMMA compute TMA atoms (nullptr if disabled -> WMMA fallback).
    host_state.compute_tma = d_compute_tma;
    host_state.compute_down_tma = d_compute_down_tma;
    host_state.group_input_tma = d_group_input_tma;
    host_state.num_compute_groups = num_compute_groups;

    // Compute dimensions
    host_state.hidden_dim = hidden_dim;
    host_state.intermediate_dim = intermediate_dim;
    host_state.num_local_experts = num_local_experts;
    host_state.max_tokens_per_expert = max_tokens_per_expert;
    host_state.max_total_recv_tokens = max_total_recv_tokens;

    // SM allocation
    host_state.num_dispatch_sms = num_dispatch_sms;
    host_state.num_forwarder_sms = num_forwarder_sms;
    host_state.num_compute_sms = num_compute_sms;
    host_state.num_dispatch_channels = num_physical_channels;  // even/odd SM pairing
    host_state.channel_dispatch_done = channel_dispatch_done;
    host_state.channel_normalized = channel_normalized;
    host_state.dispatch_channel_barrier = dispatch_channel_barrier;
    host_state.dispatch_round_barrier = dispatch_round_barrier;
    host_state.combine_channel_barrier = combine_channel_barrier;

    // Combine state
    host_state.num_combine_sms = num_combine_sms;
    host_state.num_combine_channels = num_combine_channels;
    host_state.num_logical_channels = num_logical_channels;
    host_state.expert_compute_done = expert_compute_done;
    host_state.token_compute_expected = token_compute_expected;
    host_state.token_compute_done = token_compute_done;
    host_state.combine_token_ready = combine_token_ready;
    host_state.compute_output_f = compute_output_f;

    // Combine infrastructure
    void* combine_rdma_ptr = static_cast<uint8_t*>(rdma_buffer_ptr) + num_rdma_bytes;

#ifdef MK_PERF_TRACE
    int64_t* perf_dispatch_lch_ts;
    int64_t* perf_combine_lch_ts;
    constexpr int NLP = MegaKernelState::MK_PERF_NUM_LCH_PHASES;
    CUDA_CHECK(cudaMalloc(&perf_dispatch_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(perf_dispatch_lch_ts, 0, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&perf_combine_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(perf_combine_lch_ts, 0, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    host_state.perf_dispatch_lch_ts = perf_dispatch_lch_ts;
    host_state.perf_combine_lch_ts = perf_combine_lch_ts;

    // Accumulated dispatch/combine semaphore-interaction timers (rendered as args on existing rows).
    int64_t* perf_disp_wait_nvl_ns;
    int64_t* perf_disp_publish_ns;
    int64_t* perf_disp_wait_recvcount_ns;
    int64_t* perf_comb_wait_ready_ns;
    const size_t acc_bytes = (size_t)num_logical_channels * 2 * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&perf_disp_wait_nvl_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_wait_nvl_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_disp_publish_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_publish_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_disp_wait_recvcount_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_wait_recvcount_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_ns, 0, acc_bytes));
    host_state.perf_disp_wait_nvl_ns = perf_disp_wait_nvl_ns;
    host_state.perf_disp_publish_ns = perf_disp_publish_ns;
    host_state.perf_disp_wait_recvcount_ns = perf_disp_wait_recvcount_ns;
    host_state.perf_comb_wait_ready_ns = perf_comb_wait_ready_ns;

    // publish breakdown timers.
    int64_t* perf_disp_pub_scan_ns;
    int64_t* perf_disp_pub_atomic_ns;
    int64_t* perf_disp_pub_fence_ns;
    int64_t* perf_disp_pub_store_ns;
    CUDA_CHECK(cudaMalloc(&perf_disp_pub_scan_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_pub_scan_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_disp_pub_atomic_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_pub_atomic_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_disp_pub_fence_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_pub_fence_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_disp_pub_store_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_pub_store_ns, 0, acc_bytes));
    host_state.perf_disp_pub_scan_ns = perf_disp_pub_scan_ns;
    host_state.perf_disp_pub_atomic_ns = perf_disp_pub_atomic_ns;
    host_state.perf_disp_pub_fence_ns = perf_disp_pub_fence_ns;
    host_state.perf_disp_pub_store_ns = perf_disp_pub_store_ns;

    // Per-compute-task timing buffer.
    int64_t* perf_compute_task;
    int* perf_compute_task_count;
    const size_t compute_task_bytes = (size_t)max_compute_tasks * MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&perf_compute_task, compute_task_bytes));
    CUDA_CHECK(cudaMemset(perf_compute_task, 0, compute_task_bytes));
    CUDA_CHECK(cudaMalloc(&perf_compute_task_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(perf_compute_task_count, 0, sizeof(int)));
    host_state.perf_compute_task = perf_compute_task;
    host_state.perf_compute_task_count = perf_compute_task_count;

    // Root-cause diagnostic parallel arrays (one entry per compute-task slot).
    const size_t diag_i64_bytes = (size_t)max_compute_tasks * sizeof(int64_t);
    const size_t diag_i32_bytes = (size_t)max_compute_tasks * sizeof(int);
    auto alloc_diag_i64 = [&](int64_t** p) {
        CUDA_CHECK(cudaMalloc(p, diag_i64_bytes));
        CUDA_CHECK(cudaMemset(*p, 0, diag_i64_bytes));
    };
    alloc_diag_i64(&host_state.perf_up_setup);
    alloc_diag_i64(&host_state.perf_up_tmem_alloc);
    alloc_diag_i64(&host_state.perf_up_prologue);
    alloc_diag_i64(&host_state.perf_up_tma_wait);
    alloc_diag_i64(&host_state.perf_up_mma_issue);
    alloc_diag_i64(&host_state.perf_up_mma_wait);
    alloc_diag_i64(&host_state.perf_up_loop_other);
    alloc_diag_i64(&host_state.perf_up_cluster_sync);
    alloc_diag_i64(&host_state.perf_up_epilogue);
    alloc_diag_i64(&host_state.perf_down_setup);
    alloc_diag_i64(&host_state.perf_down_tmem_alloc);
    alloc_diag_i64(&host_state.perf_down_prologue);
    alloc_diag_i64(&host_state.perf_down_tma_wait);
    alloc_diag_i64(&host_state.perf_down_mma_issue);
    alloc_diag_i64(&host_state.perf_down_mma_wait);
    alloc_diag_i64(&host_state.perf_down_loop_other);
    alloc_diag_i64(&host_state.perf_down_cluster_sync);
    alloc_diag_i64(&host_state.perf_down_epilogue);
    CUDA_CHECK(cudaMalloc(&host_state.perf_compute_multi_expert_rows, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_compute_multi_expert_rows, 0, diag_i32_bytes));
#endif

    host_state.combine_rdma_buffer_ptr = combine_rdma_ptr;
    host_state.combine_buffer_ptrs = combine_buffer_ptrs;
    host_state.combine_x = reinterpret_cast<const int4*>(compute_output);  // Combine reads FFN output from compute
    host_state.combine_topk_weights = combine_input_topk_weights;
    host_state.is_combined_token_in_rank = is_token_in_rank;
    host_state.combined_rdma_head = send_rdma_head;  // dispatch output, combine reads back
    host_state.combined_nvl_head = send_nvl_head;
    host_state.combine_src_meta = combine_input_src_meta;  // SourceMeta in DeepEP compact combine-input namespace
    host_state.combine_rdma_channel_prefix_matrix = recv_rdma_channel_prefix_matrix;
    host_state.combine_rdma_rank_prefix_sum = recv_rdma_rank_prefix_sum;
    host_state.combine_gbl_channel_prefix_matrix = recv_gbl_channel_prefix_matrix;
    host_state.combine_gbl_channel_token_count = recv_gbl_channel_token_count;
    host_state.combine_rdma_channel_token_count = recv_rdma_channel_token_count;
    host_state.combine_num_tokens = max_total_recv_tokens;
    host_state.combine_num_combined_tokens = num_tokens;
    host_state.combine_rdma_head_stride = combine_rdma_head_stride;
    host_state.combine_nvl_head_stride = combine_nvl_head_stride;
    host_state.combine_hidden = hidden_dim;  // in dtype units (bf16), not int4
    host_state.num_max_combine_rdma_chunked_send_tokens = num_max_rdma_chunked_send_tokens;
    host_state.num_max_combine_rdma_chunked_recv_tokens = num_max_rdma_chunked_recv_tokens;
    host_state.num_max_combine_nvl_chunked_send_tokens = num_max_nvl_chunked_send_tokens;
    host_state.num_max_combine_nvl_chunked_recv_tokens = num_max_nvl_chunked_recv_tokens;
    host_state.combine_bias_0 = nullptr;
    host_state.combine_bias_1 = nullptr;

    // Combine output buffers
    int4* combined_x;
    CUDA_CHECK(cudaMalloc(&combined_x, num_tokens * hidden_int4 * sizeof(int4)));
    CUDA_CHECK(cudaMemset(combined_x, 0, num_tokens * hidden_int4 * sizeof(int4)));
    host_state.combined_x = combined_x;

    float* combined_topk_weights;
    CUDA_CHECK(cudaMalloc(&combined_topk_weights, num_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(combined_topk_weights, 0, num_tokens * num_topk * sizeof(float)));
    host_state.combined_topk_weights = combined_topk_weights;

    // Copy to device
    MegaKernelState* device_state;
    CUDA_CHECK(cudaMalloc(&device_state, sizeof(MegaKernelState)));
    CUDA_CHECK(cudaMemcpy(device_state, &host_state, sizeof(MegaKernelState), cudaMemcpyHostToDevice));

    return device_state;
}

void free_megakernel_state_v7(MegaKernelState* device_state) {
    // Copy back to read pointers for freeing
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(host_state.expert_recv_count));
    CUDA_CHECK(cudaFree(host_state.expert_slot_ready));
    CUDA_CHECK(cudaFree(host_state.dispatch_done));
    CUDA_CHECK(cudaFree(host_state.dispatch_done_count));
    CUDA_CHECK(cudaFree(host_state.timeout_log_counters));
    CUDA_CHECK(cudaFree(host_state.recv_tokens));
    CUDA_CHECK(cudaFree(host_state.expert_token_offsets));
    CUDA_CHECK(cudaFree(host_state.recv_token_source_info));
    CUDA_CHECK(cudaFree(host_state.recv_token_route_weights));
    CUDA_CHECK(cudaFree(host_state.recv_src_meta));
    CUDA_CHECK(cudaFree(host_state.compute_done_count));
    CUDA_CHECK(cudaFree(host_state.expert_compute_cursor));
    CUDA_CHECK(cudaFree(host_state.expert_compute_done));
    CUDA_CHECK(cudaFree(host_state.token_compute_expected));
    CUDA_CHECK(cudaFree(host_state.token_compute_done));
    CUDA_CHECK(cudaFree(host_state.combine_token_ready));
    CUDA_CHECK(cudaFree(host_state.compute_output_f));
    CUDA_CHECK(cudaFree(host_state.compute_group_barrier));
    CUDA_CHECK(cudaFree(host_state.compute_group_phase));
    CUDA_CHECK(cudaFree(host_state.compute_tasks));
    CUDA_CHECK(cudaFree(host_state.compute_task_head));
    CUDA_CHECK(cudaFree(host_state.compute_task_tail));
    CUDA_CHECK(cudaFree(host_state.compute_enqueue_done));
    CUDA_CHECK(cudaFree(host_state.expert_enqueue_cursor));
    CUDA_CHECK(cudaFree(host_state.compute_group_task_idx));
    CUDA_CHECK(cudaFree(host_state.combined_x));
    CUDA_CHECK(cudaFree(host_state.combined_topk_weights));
    CUDA_CHECK(cudaFree(host_state.compute_output));
    CUDA_CHECK(cudaFree(host_state.combine_input));
    CUDA_CHECK(cudaFree(host_state.combine_input_topk_weights));
    CUDA_CHECK(cudaFree(host_state.combine_input_src_meta));
    CUDA_CHECK(cudaFree(host_state.combine_notify_done));
    CUDA_CHECK(cudaFree(host_state.combine_rdma_head_work));
    CUDA_CHECK(cudaFree(host_state.combine_nvl_head_work));
    CUDA_CHECK(cudaFree(host_state.gemm_workspace));
    CUDA_CHECK(cudaFree(host_state.output_accum));
    CUDA_CHECK(cudaFree(host_state.send_rdma_head));
    CUDA_CHECK(cudaFree(host_state.send_nvl_head));
    CUDA_CHECK(cudaFree(host_state.recv_rdma_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(host_state.recv_gbl_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(host_state.recv_rdma_channel_token_count));
    CUDA_CHECK(cudaFree(host_state.recv_gbl_channel_token_count));
    CUDA_CHECK(cudaFree(host_state.channel_dispatch_done));
    CUDA_CHECK(cudaFree(host_state.channel_normalized));
    CUDA_CHECK(cudaFree(host_state.dispatch_channel_barrier));
    CUDA_CHECK(cudaFree(host_state.dispatch_round_barrier));
    CUDA_CHECK(cudaFree(host_state.combine_channel_barrier));
#ifdef MK_PERF_TRACE
    CUDA_CHECK(cudaFree(host_state.perf_dispatch_lch_ts));
    CUDA_CHECK(cudaFree(host_state.perf_combine_lch_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_wait_nvl_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_publish_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_wait_recvcount_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_pub_scan_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_pub_atomic_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_pub_fence_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_pub_store_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_ns));
    CUDA_CHECK(cudaFree(host_state.perf_compute_task));
    CUDA_CHECK(cudaFree(host_state.perf_compute_task_count));
    CUDA_CHECK(cudaFree(host_state.perf_up_setup));
    CUDA_CHECK(cudaFree(host_state.perf_up_tmem_alloc));
    CUDA_CHECK(cudaFree(host_state.perf_up_prologue));
    CUDA_CHECK(cudaFree(host_state.perf_up_tma_wait));
    CUDA_CHECK(cudaFree(host_state.perf_up_mma_issue));
    CUDA_CHECK(cudaFree(host_state.perf_up_mma_wait));
    CUDA_CHECK(cudaFree(host_state.perf_up_loop_other));
    CUDA_CHECK(cudaFree(host_state.perf_up_cluster_sync));
    CUDA_CHECK(cudaFree(host_state.perf_up_epilogue));
    CUDA_CHECK(cudaFree(host_state.perf_down_setup));
    CUDA_CHECK(cudaFree(host_state.perf_down_tmem_alloc));
    CUDA_CHECK(cudaFree(host_state.perf_down_prologue));
    CUDA_CHECK(cudaFree(host_state.perf_down_tma_wait));
    CUDA_CHECK(cudaFree(host_state.perf_down_mma_issue));
    CUDA_CHECK(cudaFree(host_state.perf_down_mma_wait));
    CUDA_CHECK(cudaFree(host_state.perf_down_loop_other));
    CUDA_CHECK(cudaFree(host_state.perf_down_cluster_sync));
    CUDA_CHECK(cudaFree(host_state.perf_down_epilogue));
    CUDA_CHECK(cudaFree(host_state.perf_compute_multi_expert_rows));
#endif
    CUDA_CHECK(cudaFree(device_state));
}

float* get_output_accum_ptr(MegaKernelState* device_state) {
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    return host_state.output_accum;
}

void* get_combined_x_ptr(MegaKernelState* device_state) {
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    return host_state.combined_x;
}

}  // namespace megakernel
}  // namespace deep_ep
