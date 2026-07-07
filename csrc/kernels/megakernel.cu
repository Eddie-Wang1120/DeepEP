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

constexpr int COMPUTE_BATCH_SIZE = 256;  // Tokens per expert batch before triggering GEMM; UMMA paths use fixed padded M=256.
constexpr int COMPUTE_GROUP_SIZE = 32;   // SMs cooperating on one expert batch
constexpr int COMPUTE_SCHEDULER_SMS = 2; // Scheduler region; only #0 does scheduler work today, #1 idles.
constexpr int MK_COMPUTE_CLUSTER_DIM = (MK_COMPUTE_KERNEL == 2 ? 2 : 1);
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int MK_TIMEOUT_LOG_BUDGET = 8;
constexpr int MK_PRIORITY_SCAN_WINDOW_TOKENS = 128;
constexpr int MK_PRIORITY_MAX_ENQUEUE_PER_LOOP = 2;
constexpr int MK_DISPATCH_ROLE_COUNT = 5;
// Publish-offload (dispatch->compute bridge). NVL receiver enqueues recv_token_idx
// into a per-receiver-warp SPSC ring; a dedicated publisher warp consumes it and
// does topk scan / slot alloc / ready publish off the receive critical path.
// Stage 1 only allocates the backing state; no logic consumes it yet.
constexpr int PUB_RING_DEPTH = 64;

enum TimeoutLogSite {
    kTimeoutLogComputeRoundFlush = 0,
    kTimeoutLogComputeReady = 1,
    kTimeoutLogCombineRdmaReceiver = 2,
    kTimeoutLogCombineForwarderNvl = 3,
    kTimeoutLogDispatchRound = 4,
    kTimeoutLogDispatchChannel = 5,
    kTimeoutLogCombineRdmaCheck = 6,
    kTimeoutLogCombineNvlCheck = 7,
    kTimeoutLogCombineNvlSender = 8,
    kTimeoutLogCombineForwarderRdma = 9,
    kTimeoutLogCombineBarrier = 10,
    kTimeoutLogCount = 11,
};

// Dispatch/combine constants. Original DeepEP computes kNumRDMARanks as
// num_ranks / NUM_MAX_NVL_PEERS, then uses SWITCH_RDMA_RANKS to select the
// compile-time kNumRDMARanks specialization.
constexpr int kNumCombineForwarderWarps = 24;
constexpr int kNumCombineTMABytesPerSenderWarp = 16384;
// Per forwarder warp: 2 stages * (sizeof(int4)*32 * (NUM_MAX_NVL_PEERS+1) + 16)
constexpr int kNumCombineTMABytesPerForwarderWarp = 9248;

template <int kNumRDMARanks>
struct MegaKernelRdmaConfig {
    static constexpr int kNumCombineWarpsPerForwarder =
        (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1;
    static constexpr int kNumCombineForwarders = kNumRDMARanks * kNumCombineWarpsPerForwarder;
    static constexpr int kNumCombineRDMAReceivers = kNumCombineForwarders - NUM_MAX_NVL_PEERS;
    static constexpr int kNumTopkCombineRDMARanks = internode::get_num_topk_rdma_ranks(kNumRDMARanks);
    static constexpr int kMegaKernelNumThreads = (kNumCombineForwarders + 1) * 32;
};

// SM Role assignment (configured at launch time)
enum class SmRole {
    kDispatch,      // Runs full DeepEP dispatch (even SM = forwarder, odd SM = sender)
    kCombine,       // DeepEP combine (even SM = NVLSender+RDMAReceiver, odd SM = Forwarder)
    kScheduler,     // Enqueues expert compute batches for dynamic compute groups
    kCompute,       // Pops compute tasks, does GEMM+SwiGLU
    kGather         // Dedicated gather SM placeholder (no-op in this experiment)
};

struct ComputeTask {
    int expert_id;
    int start_slot;
    int num_tokens;
    int is_flush;
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
    int* recv_rdma_channel_token_count;     // [kNumRDMARanks * num_logical_channels] non-cumulative logical-channel count
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

    // --- Compute signaling / per-slot output path (MEGAKERNEL_COMPUTE_DESIGN section III) ---
    int* token_compute_expected;        // [max_total_recv_tokens] how many local experts must compute this token
    __nv_bfloat16* compute_output_slot; // [num_local_experts * max_tokens_per_expert, hidden] per-slot output
    int* compute_slot_ready;            // [num_local_experts * max_tokens_per_expert] per-slot ready flag
    int* compute_slot_from_flush;       // [num_local_experts * max_tokens_per_expert] whether ready came from final flush task
    int64_t* compute_slot_ready_ts;     // [num_local_experts * max_tokens_per_expert] ready publish timestamp (perf/debug)
    int* token_nhits;                   // [max_total_recv_tokens] #local-expert hits for this recv token
    int* token_slot_list;               // [max_total_recv_tokens * num_topk] absolute slot ids per hit
    int* priority_token_cursor;         // scheduler combine-order cursor for token priority scan
    int* expert_batch_enqueued;         // [num_local_experts * max_batches_per_expert] enqueue de-dup bitmap
    int max_batches_per_expert;
    int* compute_group_barrier;         // [num_compute_groups] reusable global barrier counters
    int* compute_group_phase;           // [num_compute_groups] reusable global barrier phase flags
    ComputeTask* compute_tasks;         // [max_compute_tasks] dynamic compute task queue
    int max_compute_tasks;
    int* compute_task_head;             // CAS pop cursor
    int* compute_task_tail;             // visible publish cursor consumed by workers
    int* compute_task_reserve_tail;     // atomic reservation cursor used by scheduler lanes
    int* compute_enqueue_done;          // set after all scheduler lanes publish tail tasks
    int* scheduler_done_count;          // how many scheduler lanes finished final tail publish
    int* expert_enqueue_cursor;         // [num_local_experts] how many slots have been enqueued
    int* compute_group_task_idx;        // [num_compute_groups] broadcast popped task idx to group SMs

    // --- Dedicated Gather SM state (semaphore-only experiment) ---
    int* token_done_count;              // [max_total_recv_tokens] atomicAdd by compute worker per slot completion
    int* gather_claimed;                // [max_total_recv_tokens] CAS flag: 0=unclaimed, 1=claimed by a gather SM
    int* combine_token_ready;           // [max_total_recv_tokens] set by gather SM when all local slots are ready
    int num_gather_sms;                 // number of dedicated gather SMs (fixed 2 in this experiment)
    int* combine_done_count;            // atomic: how many combine SMs have fully finished
    int* combine_all_done;              // flag: 1 once all combine SMs finished; gather SMs poll this to exit

    // --- Compute state ---
    int* compute_done_count;          // Atomic: how many experts have finished compute
    int* expert_compute_cursor;       // [num_local_experts] — how many tokens already computed

    // --- Expert weights ---
    const __nv_bfloat16* W_gateup;    // [num_local_experts, 2 * intermediate, hidden], rows [g0,u0,...]
    const __nv_bfloat16* W_down;      // [num_local_experts, hidden, intermediate]

    // --- S4.4 (route B2): UMMA compute TMA atoms (device-resident) ---
    // Per-expert 2D multicast TMA atoms for W_gateup, and per-group A(input_buf)
    // TMA atoms. Built on host (setup_compute_tma_v7), copied to device. nullptr
    // when UMMA compute is disabled (falls back to WMMA path).
    umma::ComputeTmaAtoms* compute_tma;      // device ptr; wgateup[e]
    umma::ComputeDownTmaAtoms* compute_down_tma;  // device ptr; wdown[e]
    umma::InputTmaAtom_t* group_input_tma;   // device array [num_compute_groups]
    int num_compute_groups;                  // for indexing group_input_tma / barriers

    // --- Compute output buffer ---
    __nv_bfloat16* combine_input;     // [max_total_recv_tokens, hidden] DeepEP compact recv-token namespace
    float* combine_input_topk_weights; // [max_total_recv_tokens, num_topk] DeepEP compact recv-token namespace
    internode::SourceMeta* combine_input_src_meta; // [max_total_recv_tokens] DeepEP compact recv-token namespace
    int* combine_notify_done;         // Atomic flag: combine head metadata has been normalized
    int* combine_rdma_head_work;      // [num_combined_tokens, kNumRDMARanks] normalized combine RDMA heads
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
    const int* combine_rdma_channel_token_count;     // [kNumRDMARanks * num_logical_channels] non-cumulative logical-channel count
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

    // --- Publish offload (dispatch->compute bridge), Stage 1: backing state only ---
    // receiver copies token data + stashes topk/meta into pending_* (indexed by
    // recv_token_idx, overwrite-safe), then pushes recv_token_idx into its SPSC ring.
    // publisher warp consumes, does the slot alloc / metadata writes / ready publish.
    int* token_publish_done;          // [max_total_recv_tokens] set by publisher when a token is fully published
    int* pending_topk_idx;            // [max_total_recv_tokens * num_topk] receiver-stashed expert ids
    float* pending_topk_weights;      // [max_total_recv_tokens * num_topk] receiver-stashed routing weights
    internode::SourceMeta* pending_meta; // [max_total_recv_tokens] receiver-stashed SourceMeta
    int* pub_ring;                    // [num_pub_warps_total * PUB_RING_DEPTH] recv_token_idx queue
    int* pub_ring_head;               // [num_pub_warps_total] consumer cursor (publisher)
    int* pub_ring_tail;               // [num_pub_warps_total] producer cursor (receiver)
    int* recv_warp_done;              // [num_pub_warps_total] receiver warp finished producing
    int* publish_done_count;          // atomic: how many publisher warps have drained
    int* publish_all_done;            // flag: 1 once all publishers drained (scheduler/gather use)
    int num_pub_warps_total;          // = (num_dispatch_sms / 2) * NUM_MAX_NVL_PEERS

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
    int64_t* perf_disp_cta_barrier_ns;     // first CTA barrier after dispatch role work
    int64_t* perf_disp_channel_barrier_ns; // logical-channel dispatch barrier wait
    int64_t* perf_disp_round_barrier_ns;   // dispatch round barrier wait
    int64_t* perf_disp_tokens;             // tokens processed by measured NVL receiver warp
    int64_t* perf_disp_local_hit_tokens;   // tokens with local expert hits
    int64_t* perf_disp_local_hits;         // total local expert hits
    int64_t* perf_disp_cta_release_ts;     // [lch * 2] timestamp after first CTA barrier releases
    int64_t* perf_disp_role_arrive_ts;     // [lch * 2 * role_count * NUM_MAX_NVL_PEERS] CTA barrier arrival timestamps
    int64_t* perf_disp_role_work_ns;       // [lch * 2 * role_count * NUM_MAX_NVL_PEERS] role work time before first CTA barrier
    int64_t* perf_disp_allrecv_wait_nvl_ns; // [lch * 2 * NUM_MAX_NVL_PEERS] all NVL receiver wait data
    int64_t* perf_disp_allrecv_prefix_wait_ns;
    int64_t* perf_disp_allrecv_prefix_wait_start_ts;
    int64_t* perf_disp_allrecv_prefix_observe_ts;
    int64_t* perf_disp_allrecv_prefix_done_ts;
    int64_t* perf_disp_allrecv_prefix_slowest_rdma;
    int64_t* perf_disp_allrecv_prefix_src_nvl;
    int64_t* perf_disp_allrecv_prefix_raw_start;
    int64_t* perf_disp_allrecv_prefix_raw_end;
    int64_t* perf_disp_allrecv_token_loop_ns;
    int64_t* perf_disp_allrecv_retire_ns;
    int64_t* perf_disp_prefix_store_begin_ts; // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer before prefix stores
    int64_t* perf_disp_prefix_publish_ts;   // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer after prefix stores
    int64_t* perf_disp_prefix_fence_done_ts;
    int64_t* perf_disp_prefix_store_to_fence_ns;
    int64_t* perf_disp_prefix_meta_wait_ns; // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer wait for RDMA meta
    int64_t* perf_disp_prefix_tokens;       // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer token count
    int64_t* perf_disp_prefix_producer_rank;
    int64_t* perf_disp_prefix_producer_nvl;
    int64_t* perf_disp_prefix_producer_dst_nvl;
    int64_t* perf_disp_prefix_producer_src_rdma;
    int64_t* perf_disp_allrecv_publish_ns;  // [lch * 2 * NUM_MAX_NVL_PEERS] all NVL receiver publish time
    int64_t* perf_disp_allrecv_tokens;      // [lch * 2 * NUM_MAX_NVL_PEERS] all NVL receiver tokens
    int64_t* perf_disp_allrecv_local_hits;  // [lch * 2 * NUM_MAX_NVL_PEERS] all NVL receiver local hits
    int64_t* perf_comb_tma_wait_ns;        // sum: combine sender tma_store_wait before reusing smem buffer
    int64_t* perf_comb_wait_ready_ns;      // sum: combine sender wait on per-slot compute_slot_ready
    int64_t* perf_comb_wait_ready_single_ns; // sum: ready wait for nh==1 tokens
    int64_t* perf_comb_wait_ready_multi_ns;  // sum: ready wait for nh>1 tokens
    int64_t* perf_comb_wait_ready_flush_ns;  // sum: per-slot ready wait for slots computed by final flush tasks
    int64_t* perf_comb_wait_ready_full_ns;   // sum: per-slot ready wait for slots computed by full-batch tasks
    int64_t* perf_comb_wait_ready_flush_count; // number of waited slots from final flush tasks
    int64_t* perf_comb_wait_ready_full_count;  // number of waited slots from full-batch tasks
    int64_t* perf_comb_gather_reduce_ns;   // sum: gather per-slot outputs + fp32 reduce + bf16 pack
    int64_t* perf_comb_gather_single_ns;   // sum: nh==1 gather/copy path
    int64_t* perf_comb_gather_multi_ns;    // sum: nh>1 gather/reduce path
    int64_t* perf_comb_pack_meta_ns;       // sum: write SourceMeta/topk_weights/padding into TMA packet
    int64_t* perf_comb_pack_meta_work_ns;  // sum: metadata writes without trailing syncwarp
    int64_t* perf_comb_pack_meta_sync_ns;  // sum: trailing syncwarp after metadata writes
    int64_t* perf_comb_tma_store_ns;       // sum: tma_store_fence + tma_store_1d issue
    int64_t* perf_comb_tma_wait_max_ns;        // max per sender warp/token phase, comparable to wall time
    int64_t* perf_comb_wait_ready_max_ns;
    int64_t* perf_comb_wait_ready_single_max_ns;
    int64_t* perf_comb_wait_ready_multi_max_ns;
    int64_t* perf_comb_wait_ready_flush_max_ns;
    int64_t* perf_comb_wait_ready_full_max_ns;
    int64_t* perf_comb_wait_top_ns;
    int64_t* perf_comb_wait_top_token;
    int64_t* perf_comb_wait_top_slot;
    int64_t* perf_comb_wait_top_expert;
    int64_t* perf_comb_wait_top_from_flush;
    int64_t* perf_comb_gather_reduce_max_ns;
    int64_t* perf_comb_gather_single_max_ns;
    int64_t* perf_comb_gather_multi_max_ns;
    int64_t* perf_comb_pack_meta_max_ns;
    int64_t* perf_comb_pack_meta_work_max_ns;
    int64_t* perf_comb_pack_meta_sync_max_ns;
    int64_t* perf_comb_tma_store_max_ns;
    int64_t* perf_comb_nhit_sum;           // sum of token_nhits processed by combine sender
    int64_t* perf_comb_token_count;        // number of tokens processed by combine sender
    int64_t* perf_comb_single_token_count; // number of nh==1 tokens processed by combine sender
    int64_t* perf_comb_multi_token_count;  // number of nh>1 tokens processed by combine sender
    // Scheduler bridge timing: our expert_slot_ready -> expert_recv_count -> compute task queue layer.
    int64_t* perf_sched_ts;                // [2] scheduler start/end
    int64_t* perf_sched_scan_ns;           // ready bitmap scan + expert_recv_count publish
    int64_t* perf_sched_enqueue_ns;        // compute task queue publish
    int64_t* perf_sched_idle_ns;           // poll sleep while waiting for more ready slots
    int64_t* perf_sched_priority_ns;       // priority token window scan + enqueue attempts
    int64_t* perf_sched_normal_ns;         // normal full-batch enqueue path
    int64_t* perf_sched_tail_flush_ns;     // final dispatch_done full/tail enqueue path
    int64_t* perf_sched_publish_total_ns;  // scheduler_publish_task total time
    int64_t* perf_sched_publish_wait_ns;   // waiting for compute_task_tail ordered publish
    int64_t* perf_sched_publish_wait_max_ns;
    int64_t* perf_sched_publish_wait_max_tail;
    int64_t* perf_sched_publish_wait_max_visible_tail;
    int64_t* perf_sched_priority_scan_tokens;
    int64_t* perf_sched_priority_ready_tokens;
    int64_t* perf_sched_priority_full_batch_hits;
    int64_t* perf_sched_priority_batch_already_enqueued;
    int64_t* perf_sched_priority_not_full;
    int64_t* perf_sched_normal_full_batch_enqueues;
    int64_t* perf_sched_flush_tail_enqueues;
    int64_t* perf_sched_queue_empty_count;
    int64_t* perf_sched_queue_empty_after_dispatch_count;
    int64_t* perf_sched_max_ready_tail_gap;
    int64_t* perf_sched_stall_expert;
    int64_t* perf_sched_stall_recv_count;
    int64_t* perf_sched_stall_alloc_count;
    int64_t* perf_sched_stall_enqueue_cursor;
    int64_t* perf_sched_stall_first_unready_slot;
    int64_t* perf_sched_stall_first_unready_ready;
    int64_t* perf_sched_stall_dispatch_done;
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
    //   20 = ts after compute_slot_ready publish + sync (== signal done)
    //   21 = compute task queue index
    //   22 = expert-local start_slot
    //   23 = expert-local end_slot (exclusive)
    //   24 = absolute slot base (= expert_id * max_tokens_per_expert + start_slot)
    //   25 = is_flush task flag
    static constexpr int MK_PERF_NUM_COMPUTE_FIELDS = 26;
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
    int* perf_compute_task_has_multi;     // [max_compute_tasks] output-phase s_has_multi_finalize (1=slow path)
    // Queue handoff diagnostics, indexed by compute task queue index.
    int64_t* perf_task_publish_ts;        // scheduler published task tail
    int64_t* perf_task_pop_start_ts;      // compute group leader started dequeue loop
    int64_t* perf_task_pop_done_ts;       // compute group leader acquired this task
    int64_t* perf_task_bcast_done_ts;     // task_idx broadcast sync done for the group
    int64_t* perf_task_start_ts;          // compute task body timing start
    int64_t* perf_task_prev_gap_ns;       // this group's task_start - previous task end
    int* perf_task_pop_attempts;          // number of queue polls/CAS attempts for this task
    int* perf_task_cas_failures;          // failed CAS attempts before acquiring this task
    int* perf_task_group_id;              // group that popped this task
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
//   gate = A @ W_gateup[0::2]^T   (A:[M,K], W_gateup:[2N,K])
//   up   = A @ W_gateup[1::2]^T   (same shapes, N = intermediate)
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
    const __nv_bfloat16* __restrict__ W_gateup, // [2N, K] rows [g0,u0,g1,u1,...]
    __nv_bfloat16* __restrict__ act,            // [M, N] row_major output
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
            wmma::load_matrix_sync(bg_frag, W_gateup + (2 * col_offset) * K + k, 2 * K);
            wmma::load_matrix_sync(bu_frag, W_gateup + (2 * col_offset + 1) * K + k, 2 * K);
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
//   kLowLatencyMode=false, kCachedMode=false,
//   kNumTMABytesPerWarp=16384, kNumDispatchRDMASenderWarps=7.
//   kNumRDMARanks is selected at launch by SWITCH_RDMA_RANKS.
//
// Only difference from DeepEP: In NVLReceivers section, after copying token
// data to recv_x, we also route tokens to expert storage + signal compute SMs.
// ============================================================================

// Instantiated template constants
constexpr int kNumDispatchRDMASenderWarps = 7;
constexpr int kNumTMABytesPerWarp = 16384;
constexpr bool kLowLatencyMode = false;
constexpr bool kCachedMode = false;

template <int kNumRDMARanks, int kStage>
__device__ void dispatch_worker_v2(
    int sm_id,
    int dispatch_sm_idx,  // 0-based index among all dispatch SMs
    MegaKernelState* state
) {
    using namespace internode;
    constexpr int kNumTopkRDMARanks = internode::get_num_topk_rdma_ranks(kNumRDMARanks);
    const auto num_sms = state->num_dispatch_sms;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = state->num_dispatch_channels, channel_id = sm_id / 2;
    constexpr int num_logical_channels_per_physical = kStage;
    const int num_logical_channels = num_channels * num_logical_channels_per_physical;
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
    int dispatch_role_id = 0;
    if (warp_role == WarpRole::kRDMASender)
        dispatch_role_id = 0;
    else if (warp_role == WarpRole::kRDMASenderCoordinator)
        dispatch_role_id = 1;
    else if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        dispatch_role_id = 2;
    else if (warp_role == WarpRole::kForwarderCoordinator)
        dispatch_role_id = 3;
    else
        dispatch_role_id = 4;
    int dispatch_role_slot = target_rank >= 0 ? target_rank : 0;
    if (warp_role == WarpRole::kRDMASender)
        dispatch_role_slot = warp_id;
    if (dispatch_role_slot >= NUM_MAX_NVL_PEERS)
        dispatch_role_slot = NUM_MAX_NVL_PEERS - 1;
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
#ifdef MK_PERF_TRACE
    int64_t recv_retire_start_ns = 0;
#endif
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

    for (int logical_stage = 0; logical_stage < num_logical_channels_per_physical; ++logical_stage) {
        const int logical_channel_id = channel_id * num_logical_channels_per_physical + logical_stage;
#ifdef MK_PERF_TRACE
        int dispatch_lch_role_for_perf = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int dispatch_acc_idx_for_perf = logical_channel_id * 2 + dispatch_lch_role_for_perf;
        int64_t dispatch_role_work_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
#endif
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
#ifdef MK_PERF_TRACE
        int64_t fwd_prefix_meta_wait_start_ns = (lane_id < kNumRDMARanks) ? globaltimer_ns() : 0;
#endif
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
#ifdef MK_PERF_TRACE
                    int prefix_prod_idx = (dispatch_acc_idx_for_perf * NUM_MAX_NVL_PEERS + dst_nvl_rank) * kNumRDMARanks + lane_id;
                    int64_t prefix_store_begin_ts = globaltimer_ns();
#endif
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
#ifdef MK_PERF_TRACE
                    int64_t prefix_publish_ts = globaltimer_ns();
                    state->perf_disp_prefix_store_begin_ts[prefix_prod_idx] = prefix_store_begin_ts;
                    state->perf_disp_prefix_publish_ts[prefix_prod_idx] = prefix_publish_ts;
                    state->perf_disp_prefix_meta_wait_ns[prefix_prod_idx] += prefix_store_begin_ts - fwd_prefix_meta_wait_start_ns;
                    state->perf_disp_prefix_tokens[prefix_prod_idx] = num_tokens_to_recv_from_rdma;
                    state->perf_disp_prefix_producer_rank[prefix_prod_idx] = state->rank;
                    state->perf_disp_prefix_producer_nvl[prefix_prod_idx] = nvl_rank;
                    state->perf_disp_prefix_producer_dst_nvl[prefix_prod_idx] = dst_nvl_rank;
                    state->perf_disp_prefix_producer_src_rdma[prefix_prod_idx] = lane_id;
#endif
                    // if (blockIdx.x == 16 && lane_id == 0) {
                    // printf("lane_id: %d, src_rdma_channel_prefix: %d, src_rdma_channel_prefix_1: %d, num_tokens_to_recv_from_rdma: %d num_channels: %d channel_id: %d\n", lane_id, src_rdma_channel_prefix, src_rdma_channel_prefix_1, num_tokens_to_recv_from_rdma, num_channels, channel_id);
                    // }
                    recv_rdma_channel_prefix_matrix[lane_id * num_logical_channels + logical_channel_id] = src_rdma_channel_prefix_1;
                    // Save per-logical-channel token count (non-cumulative) for diagnostics and bounds checks.
                    state->recv_rdma_channel_token_count[lane_id * num_logical_channels + logical_channel_id] = num_tokens_to_recv_from_rdma;
                    // __threadfence_system();
#ifdef MK_PERF_TRACE
                    {
                        int64_t prefix_fence_done_ts = globaltimer_ns();
                        state->perf_disp_prefix_fence_done_ts[prefix_prod_idx] = prefix_fence_done_ts;
                        state->perf_disp_prefix_store_to_fence_ns[prefix_prod_idx] += prefix_fence_done_ts - prefix_publish_ts;
                    }
#endif
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
#ifdef MK_PERF_TRACE
        int recv_perf_idx_for_role = dispatch_acc_idx_for_perf * NUM_MAX_NVL_PEERS + src_nvl_rank;
        int64_t recv_prefix_wait_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
        int64_t recv_token_loop_start_ns = 0;
#endif
        const int local_expert_begin = state->rank * (num_experts / num_ranks);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
#ifdef MK_PERF_TRACE
        int64_t recv_prefix_lane_start_ns = (lane_id < kNumRDMARanks) ? globaltimer_ns() : 0;
        int64_t recv_prefix_lane_observe_ts = 0;
        int64_t recv_prefix_lane_wait_ns = 0;
        int recv_prefix_lane_src_rdma = -1;
        int recv_prefix_lane_raw_start = 0;
        int recv_prefix_lane_raw_end = 0;
#endif
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (start_offset < 0 and end_offset < 0) {
#ifdef MK_PERF_TRACE
                recv_prefix_lane_observe_ts = globaltimer_ns();
                recv_prefix_lane_wait_ns = recv_prefix_lane_observe_ts - recv_prefix_lane_start_ns;
                recv_prefix_lane_src_rdma = lane_id;
                recv_prefix_lane_raw_start = start_offset;
                recv_prefix_lane_raw_end = end_offset;
#endif
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
#ifdef MK_PERF_TRACE
        int64_t prefix_slowest_wait_ns = recv_prefix_lane_wait_ns;
        int64_t prefix_slowest_start_ts = recv_prefix_lane_start_ns;
        int64_t prefix_slowest_observe_ts = recv_prefix_lane_observe_ts;
        int prefix_slowest_src_rdma = recv_prefix_lane_src_rdma;
        int prefix_slowest_raw_start = recv_prefix_lane_raw_start;
        int prefix_slowest_raw_end = recv_prefix_lane_raw_end;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            int64_t other_wait = __shfl_down_sync(0xffffffff, prefix_slowest_wait_ns, offset);
            int64_t other_start_ts = __shfl_down_sync(0xffffffff, prefix_slowest_start_ts, offset);
            int64_t other_observe_ts = __shfl_down_sync(0xffffffff, prefix_slowest_observe_ts, offset);
            int other_src = __shfl_down_sync(0xffffffff, prefix_slowest_src_rdma, offset);
            int other_raw_start = __shfl_down_sync(0xffffffff, prefix_slowest_raw_start, offset);
            int other_raw_end = __shfl_down_sync(0xffffffff, prefix_slowest_raw_end, offset);
            if (other_wait > prefix_slowest_wait_ns) {
                prefix_slowest_wait_ns = other_wait;
                prefix_slowest_start_ts = other_start_ts;
                prefix_slowest_observe_ts = other_observe_ts;
                prefix_slowest_src_rdma = other_src;
                prefix_slowest_raw_start = other_raw_start;
                prefix_slowest_raw_end = other_raw_end;
            }
        }
        if (lane_id == 0) {
            int64_t now = globaltimer_ns();
            state->perf_disp_allrecv_prefix_wait_ns[recv_perf_idx_for_role] += now - recv_prefix_wait_start_ns;
            state->perf_disp_allrecv_prefix_wait_start_ts[recv_perf_idx_for_role] = prefix_slowest_start_ts;
            state->perf_disp_allrecv_prefix_observe_ts[recv_perf_idx_for_role] = prefix_slowest_observe_ts;
            state->perf_disp_allrecv_prefix_done_ts[recv_perf_idx_for_role] = now;
            state->perf_disp_allrecv_prefix_slowest_rdma[recv_perf_idx_for_role] = prefix_slowest_src_rdma;
            state->perf_disp_allrecv_prefix_src_nvl[recv_perf_idx_for_role] = src_nvl_rank;
            state->perf_disp_allrecv_prefix_raw_start[recv_perf_idx_for_role] = prefix_slowest_raw_start;
            state->perf_disp_allrecv_prefix_raw_end[recv_perf_idx_for_role] = prefix_slowest_raw_end;
            recv_token_loop_start_ns = now;
        }
#endif

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
            if (lane_id == 0) {
                int64_t wait_nvl_ns = globaltimer_ns() - wait_nvl_start;
                int recv_idx = dispatch_acc_idx_for_perf * NUM_MAX_NVL_PEERS + src_nvl_rank;
                state->perf_disp_allrecv_wait_nvl_ns[recv_idx] += wait_nvl_ns;
                if (target_rank == 0)
                    state->perf_disp_wait_nvl_ns[dispatch_acc_idx_for_perf] += wait_nvl_ns;
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
                const int hidden_int4 = state->hidden_dim * sizeof(__nv_bfloat16) / sizeof(int4);
                const int4* src_i4 = reinterpret_cast<const int4*>(src_data);
                int4* dst_i4 = reinterpret_cast<int4*>(state->combine_input) +
                    (int64_t)recv_token_idx * hidden_int4;
                for (int v = lane_id; v < hidden_int4; v += 32)
                    dst_i4[v] = src_i4[v];


                // Fill topk weights/src_meta, publish this token's local expert count, then
                // publish expert slots. expected must be final before any expert_recv_count
                // advances, otherwise compute can mark multi-local-expert tokens ready early.
                const int local_expert_end = local_expert_begin + state->num_local_experts;
                if (lane_id == 0) {
#ifdef MK_PERF_TRACE
                    int acc_idx = dispatch_acc_idx_for_perf;
                    int recv_perf_idx = acc_idx * NUM_MAX_NVL_PEERS + src_nvl_rank;
                    bool record_perf = (target_rank == 0);  // keep legacy single-receiver aggregate
                    int64_t publish_start = globaltimer_ns();
                    int64_t wait_recvcount_acc = 0;
#endif
#ifdef MK_PERF_TRACE
                    int64_t pub_scan_start = globaltimer_ns();
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
#ifdef MK_PERF_TRACE
                    int64_t pub_scan_ns = globaltimer_ns() - pub_scan_start;
                    int64_t pub_atomic_start = globaltimer_ns();
#endif
                    if (local_hits > 0)
                        state->combine_input_src_meta[recv_token_idx] = meta;

                    // Pass 1: allocate slots and write source_info for all local hits.
                    int hit_local_expert[32];
                    int hit_slot[32];
                    int hit_abs_slot[32];
                    int num_hits = 0;
                    for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
                        int expert_id = ld_nc_global(topk_data_ptr + topk_slot);
                        if (expert_id < local_expert_begin || expert_id >= local_expert_end)
                            continue;
                        int local_expert_id = expert_id - local_expert_begin;
                        int slot = atomicAdd(&state->expert_token_offsets[local_expert_id], 1);
                        if (slot >= state->max_tokens_per_expert) {
                            printf("MK dispatch expert slot overflow, rank=%d recv_token=%lld expert=%d slot=%d max_tpe=%d\n",
                                   state->rank, (long long)recv_token_idx, expert_id, slot, state->max_tokens_per_expert);
                            trap();
                        }
                        int dest_offset = local_expert_id * state->max_tokens_per_expert + slot;
                        int* dst_ptr = &state->recv_token_source_info[dest_offset * 2];
                        // Plain stores: ordering vs slot_ready is enforced by the single
                        // __threadfence() below. All readers are this GPU's compute workers,
                        // so device-scope visibility is sufficient.
                        st_na_global(dst_ptr, static_cast<int>(recv_token_idx));
                        st_na_global(dst_ptr + 1, topk_slot);
                        hit_local_expert[num_hits] = local_expert_id;
                        hit_slot[num_hits] = slot;
                        hit_abs_slot[num_hits] = local_expert_id * state->max_tokens_per_expert + slot;
                        num_hits += 1;
                    }
                    if (num_hits > 0) {
                        // Multiple dispatch receivers may contribute to the same recv_token_idx.
                        // Reserve this writer's slice in token_slot_list atomically, mirroring
                        // the old token_compute_expected atomic accumulation.
                        int hit_base = atomicAdd(&state->token_nhits[recv_token_idx], num_hits);
                        if (hit_base + num_hits > num_topk) {
                            printf("MK dispatch token hit overflow, rank=%d recv_token=%lld hit_base=%d num_hits=%d num_topk=%d\n",
                                   state->rank, (long long)recv_token_idx, hit_base, num_hits, num_topk);
                            trap();
                        }
                        for (int h = 0; h < num_hits; ++h)
                            state->token_slot_list[recv_token_idx * num_topk + hit_base + h] = hit_abs_slot[h];
                        atomicAdd(&state->token_compute_expected[recv_token_idx], num_hits);
                    }
#ifdef MK_PERF_TRACE
                    int64_t pub_atomic_ns = globaltimer_ns() - pub_atomic_start;
                    int64_t pub_fence_start = globaltimer_ns();
#endif

                    // Single device-scope fence per token: orders the plain source_info and
                    // token_slot_list stores before the slot_ready releases. The whole signal
                    // chain stays in device scope because no remote GPU reads these buffers.
                    if (num_hits > 0)
                        __threadfence();
#ifdef MK_PERF_TRACE
                    int64_t pub_fence_ns = globaltimer_ns() - pub_fence_start;
                    int64_t pub_store_start = globaltimer_ns();
#endif

                    // Pass 2: mark each slot ready (unordered, no spin-wait). Scheduler
                    // scans the bitmap and advances expert_recv_count.
                    for (int h = 0; h < num_hits; ++h) {
                        int local_expert_id = hit_local_expert[h];
                        int slot = hit_slot[h];
                        st_na_release(&state->expert_slot_ready[local_expert_id * state->max_tokens_per_expert + slot], 1);
                    }
#ifdef MK_PERF_TRACE
                    int64_t pub_store_ns = globaltimer_ns() - pub_store_start;
                    int64_t publish_total = globaltimer_ns() - publish_start;
                    state->perf_disp_allrecv_publish_ns[recv_perf_idx] += publish_total - wait_recvcount_acc;
                    state->perf_disp_allrecv_tokens[recv_perf_idx] += 1;
                    state->perf_disp_allrecv_local_hits[recv_perf_idx] += num_hits;
                    if (record_perf) {
                        state->perf_disp_wait_recvcount_ns[acc_idx] += wait_recvcount_acc;
                        state->perf_disp_publish_ns[acc_idx] += publish_total - wait_recvcount_acc;
                        state->perf_disp_pub_scan_ns[acc_idx] += pub_scan_ns;
                        state->perf_disp_pub_atomic_ns[acc_idx] += pub_atomic_ns;
                        state->perf_disp_pub_fence_ns[acc_idx] += pub_fence_ns;
                        state->perf_disp_pub_store_ns[acc_idx] += pub_store_ns;
                        state->perf_disp_tokens[acc_idx] += 1;
                        if (num_hits > 0)
                            state->perf_disp_local_hit_tokens[acc_idx] += 1;
                        state->perf_disp_local_hits[acc_idx] += num_hits;
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

#ifdef MK_PERF_TRACE
        if (lane_id == 0) {
            int64_t now = globaltimer_ns();
            state->perf_disp_allrecv_token_loop_ns[recv_perf_idx_for_role] += now - recv_token_loop_start_ns;
            recv_retire_start_ns = now;
        }
#endif

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

#ifdef MK_PERF_TRACE
    int64_t dispatch_cta_barrier_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
    if (lane_id == 0 && dispatch_role_id == 4) {
        int recv_idx = dispatch_acc_idx_for_perf * NUM_MAX_NVL_PEERS + dispatch_role_slot;
        state->perf_disp_allrecv_retire_ns[recv_idx] += dispatch_cta_barrier_start_ns - recv_retire_start_ns;
    }
    if (lane_id == 0) {
        int role_idx = (((dispatch_acc_idx_for_perf * MK_DISPATCH_ROLE_COUNT + dispatch_role_id) * NUM_MAX_NVL_PEERS) + dispatch_role_slot);
        state->perf_disp_role_arrive_ts[role_idx] = dispatch_cta_barrier_start_ns;
        state->perf_disp_role_work_ns[role_idx] = dispatch_cta_barrier_start_ns - dispatch_role_work_start_ns;
    }
#endif
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
#ifdef MK_PERF_TRACE
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
        int64_t cta_release_ns = globaltimer_ns();
        state->perf_disp_cta_barrier_ns[acc_idx] += cta_release_ns - dispatch_cta_barrier_start_ns;
        state->perf_disp_cta_release_ts[acc_idx] = cta_release_ns;
    }
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
#ifdef MK_PERF_TRACE
        int64_t barrier_start_ns = globaltimer_ns();
#endif
        while (ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]) < 2) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch logical-channel barrier timeout, physical_ch=%d logical_ch=%d count=%d\n",
                       channel_id, logical_channel_id, ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]));
                trap();
            }
            __nanosleep(32);
        }
#ifdef MK_PERF_TRACE
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
        state->perf_disp_channel_barrier_ns[acc_idx] += globaltimer_ns() - barrier_start_ns;
#endif
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
#ifdef MK_PERF_TRACE
        int64_t barrier_start_ns = globaltimer_ns();
#endif
        while (ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]) < num_channels) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch round barrier timeout, physical_ch=%d logical_ch=%d round=%d count=%d need=%d\n",
                       channel_id, logical_channel_id, round_idx,
                       ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]), num_channels);
                trap();
            }
            __nanosleep(32);
        }
#ifdef MK_PERF_TRACE
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
        state->perf_disp_round_barrier_ns[acc_idx] += globaltimer_ns() - barrier_start_ns;
#endif
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

__device__ __forceinline__ void scheduler_publish_task(MegaKernelState* state, int expert_id, int start_slot, int num_tokens, int is_flush) {
#ifdef MK_PERF_TRACE
    int64_t publish_start_ns = globaltimer_ns();
#endif
    int tail = atomicAdd(state->compute_task_reserve_tail, 1);
    if (tail >= state->max_compute_tasks) {
        printf("MK compute task queue overflow, rank=%d tail=%d max=%d\n", state->rank, tail, state->max_compute_tasks);
        trap();
    }
    state->compute_tasks[tail] = ComputeTask{expert_id, start_slot, num_tokens, is_flush};
    __threadfence_system();
#ifdef MK_PERF_TRACE
    int64_t wait_start_ns = globaltimer_ns();
    int visible_tail = ld_acquire_sys_global(state->compute_task_tail);
    int wait_start_visible_tail = visible_tail;
    while (visible_tail != tail) {
        __nanosleep(32);
        visible_tail = ld_acquire_sys_global(state->compute_task_tail);
    }
    int64_t publish_ready_ns = globaltimer_ns();
    int64_t wait_ns = publish_ready_ns - wait_start_ns;
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_publish_wait_ns),
              static_cast<unsigned long long>(wait_ns));
    unsigned long long old_max = atomicMax(reinterpret_cast<unsigned long long*>(state->perf_sched_publish_wait_max_ns),
                                           static_cast<unsigned long long>(wait_ns));
    if (static_cast<unsigned long long>(wait_ns) > old_max) {
        *state->perf_sched_publish_wait_max_tail = static_cast<int64_t>(tail);
        *state->perf_sched_publish_wait_max_visible_tail = static_cast<int64_t>(wait_start_visible_tail);
    }
    state->perf_task_publish_ts[tail] = publish_ready_ns;
#else
    while (ld_acquire_sys_global(state->compute_task_tail) != tail)
        __nanosleep(32);
#endif
    st_release_sys_global(state->compute_task_tail, tail + 1);
#ifdef MK_PERF_TRACE
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_publish_total_ns),
              static_cast<unsigned long long>(globaltimer_ns() - publish_start_ns));
#endif
}

__device__ __forceinline__ bool scheduler_try_enqueue_batch(
    MegaKernelState* state, int expert_id, int batch_id, int start_slot, int num_tokens, int is_flush) {
    if (batch_id < 0 || batch_id >= state->max_batches_per_expert) {
        printf("MK scheduler batch id overflow, rank=%d expert=%d batch=%d max=%d\n",
               state->rank, expert_id, batch_id, state->max_batches_per_expert);
        trap();
    }
    int idx = expert_id * state->max_batches_per_expert + batch_id;
    if (atomicCAS(&state->expert_batch_enqueued[idx], 0, 1) != 0)
        return false;
    scheduler_publish_task(state, expert_id, start_slot, num_tokens, is_flush);
    return true;
}

__device__ void compute_scheduler_worker(MegaKernelState* state, int scheduler_id, int num_schedulers) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;
    bool tail_enqueued = false;

    // Shared memory for cooperative priority batch-ready check
    __shared__ int s_priority_expert_id;
    __shared__ int s_priority_batch_start;
    __shared__ int s_priority_batch_end;
    __shared__ int s_priority_ready_base;
    __shared__ int s_priority_alloc_count;
    __shared__ int s_priority_batch_ready;  // result: 1 = ready, 0 = not ready
    __shared__ int s_priority_enqueued;
    __shared__ int s_priority_cursor_token;
    __shared__ int s_priority_new_cursor;
    __shared__ int s_priority_scan_end;
    __shared__ int s_priority_token;
    __shared__ int s_priority_nh;
    __shared__ int s_priority_hit_idx;
    __shared__ int s_priority_token_has_not_full;
    __shared__ int s_priority_done;  // signals priority loop exit
    __shared__ int s_dispatch_done;

#ifdef MK_PERF_TRACE
    if (tid == 0 && scheduler_id == 0)
        state->perf_sched_ts[0] = globaltimer_ns();
    int64_t sched_scan_acc = 0;
    int64_t sched_enqueue_acc = 0;
    int64_t sched_idle_acc = 0;
    int64_t sched_priority_acc = 0;
    int64_t sched_normal_acc = 0;
    int64_t sched_tail_flush_acc = 0;
    int64_t priority_scan_tokens = 0;
    int64_t priority_ready_tokens = 0;
    int64_t priority_full_batch_hits = 0;
    int64_t priority_batch_already_enqueued = 0;
    int64_t priority_not_full = 0;
    int64_t normal_full_batch_enqueues = 0;
    int64_t flush_tail_enqueues = 0;
    int64_t queue_empty_count = 0;
    int64_t queue_empty_after_dispatch_count = 0;
    int64_t max_ready_tail_gap = 0;
    int64_t stall_expert = -1;
    int64_t stall_recv_count = 0;
    int64_t stall_alloc_count = 0;
    int64_t stall_enqueue_cursor = 0;
    int64_t stall_first_unready_slot = -1;
    int64_t stall_first_unready_ready = 0;
    int64_t stall_dispatch_done = 0;
#endif

    while (true) {
        // Thread 0 loads dispatch_done and broadcasts via shared memory
        if (tid == 0) {
            int dispatch_done_count = ld_acquire_sys_global(state->dispatch_done_count);
            s_dispatch_done = (dispatch_done_count == state->expected_dispatch_done_count) ? 1 : 0;
        }
        __syncthreads();
        bool dispatch_done = (s_dispatch_done != 0);

#ifdef MK_PERF_TRACE
        if (tid == 0 && scheduler_id == 0) {
            int task_head = ld_acquire_sys_global(state->compute_task_head);
            int task_tail = ld_acquire_sys_global(state->compute_task_tail);
            if (task_head >= task_tail) {
                queue_empty_count += 1;
                if (dispatch_done)
                    queue_empty_after_dispatch_count += 1;
            }
        }
        int64_t sched_scan_start = globaltimer_ns();
#endif

        // === Multi-threaded ready prefix scan ===
        // Each thread scans a chunk of the slot-ready array for each expert.
        // Thread 0 reads the current recv_count, then all threads cooperatively
        // scan forward from that point in parallel chunks.
        for (int expert_id = scheduler_id; expert_id < num_local_experts; expert_id += num_schedulers) {
            int old_count;
            if (tid == 0) {
                old_count = ld_acquire_global(&state->expert_recv_count[expert_id]);
                s_priority_alloc_count = old_count;  // reuse shared var to broadcast old_count
            }
            __syncthreads();
            old_count = s_priority_alloc_count;

            // Cooperative scan: each thread checks a contiguous chunk of slots starting from old_count.
            // We scan in waves of num_threads slots at a time, finding the contiguous prefix.
            int count = old_count;
            while (count < max_tpe) {
                int my_slot = count + tid;
                int my_ready = 0;
                if (my_slot < max_tpe) {
                    my_ready = (ld_acquire_global(&state->expert_slot_ready[expert_id * max_tpe + my_slot]) == 1) ? 1 : 0;
                }
                // All threads report: if all slots in this wave are ready, advance
                int all_ready = __syncthreads_and(my_ready || (my_slot >= max_tpe));
                if (!all_ready) {
                    // Find the first non-ready slot in this wave via warp vote
                    // Thread 0 does a sequential scan of just this wave's portion
                    if (tid == 0) {
                        for (int s = count; s < count + num_threads && s < max_tpe; ++s) {
                            if (ld_acquire_global(&state->expert_slot_ready[expert_id * max_tpe + s]) == 1) {
                                count = s + 1;
                            } else {
                                break;
                            }
                        }
                        s_priority_alloc_count = count;  // broadcast final count
                    }
                    __syncthreads();
                    count = s_priority_alloc_count;
                    break;
                }
                count += num_threads;
                if (count > max_tpe) count = max_tpe;
            }

            if (tid == 0 && count != old_count) {
                __threadfence();
                st_na_release(&state->expert_recv_count[expert_id], count);
            }
#ifdef MK_PERF_TRACE
            if (tid == 0) {
                int alloc_count = ld_acquire_global(&state->expert_token_offsets[expert_id]);
                int gap = alloc_count - count;
                if (gap > max_ready_tail_gap) {
                    max_ready_tail_gap = gap;
                    stall_expert = expert_id;
                    stall_recv_count = count;
                    stall_alloc_count = alloc_count;
                    stall_enqueue_cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
                    stall_first_unready_slot = count;
                    stall_first_unready_ready = (count < max_tpe) ? ld_acquire_global(&state->expert_slot_ready[expert_id * max_tpe + count]) : -1;
                    stall_dispatch_done = dispatch_done ? 1 : 0;
                }
            }
#endif
            __syncthreads();
        }
#ifdef MK_PERF_TRACE
        sched_scan_acc += globaltimer_ns() - sched_scan_start;
        int64_t sched_enqueue_start = globaltimer_ns();
#endif

        // === Multi-threaded priority batch-ready check ===
        // Thread 0 drives the token/hit iteration; all threads cooperatively check 256 slots.
#ifdef MK_PERF_TRACE
        int64_t sched_priority_start = globaltimer_ns();
#endif
        if (scheduler_id == 0) {
            if (tid == 0) {
                s_priority_cursor_token = ld_acquire_global(state->priority_token_cursor);
                s_priority_scan_end = s_priority_cursor_token + MK_PRIORITY_SCAN_WINDOW_TOKENS;
                if (s_priority_scan_end > state->max_total_recv_tokens)
                    s_priority_scan_end = state->max_total_recv_tokens;
                s_priority_new_cursor = s_priority_cursor_token;
                s_priority_enqueued = 0;
                s_priority_done = 0;
                s_priority_token = s_priority_cursor_token;
                s_priority_hit_idx = 0;
                s_priority_token_has_not_full = 0;
            }
            __syncthreads();

            // Cooperative priority loop
            while (true) {
                // Thread 0 advances to next valid (token, hit) pair
                if (tid == 0) {
                    while (true) {
                        if (s_priority_enqueued >= MK_PRIORITY_MAX_ENQUEUE_PER_LOOP ||
                            s_priority_token >= s_priority_scan_end) {
                            s_priority_done = 1;
                            s_priority_expert_id = -1;
                            break;
                        }
                        int token = s_priority_token;
                        int h = s_priority_hit_idx;
                        int nh = ld_acquire_global(&state->token_nhits[token]);
#ifdef MK_PERF_TRACE
                        if (h == 0) {
                            priority_scan_tokens += 1;
                            priority_ready_tokens += 1;
                        }
#endif
                        if (h >= nh) {
                            // Move to next token
                            if (!s_priority_token_has_not_full)
                                s_priority_new_cursor = token + 1;
                            s_priority_token = token + 1;
                            s_priority_hit_idx = 0;
                            s_priority_token_has_not_full = 0;
                            continue;
                        }
                        int slot = ld_nc_global(&state->token_slot_list[token * state->num_topk + h]);
                        s_priority_hit_idx = h + 1;
                        if (slot < 0)
                            continue;
                        int expert_id = slot / max_tpe;
                        int expert_local_slot = slot - expert_id * max_tpe;
                        int batch_id = expert_local_slot / COMPUTE_BATCH_SIZE;
                        int batch_start = batch_id * COMPUTE_BATCH_SIZE;
                        int batch_end = batch_start + COMPUTE_BATCH_SIZE;
                        int alloc_count = ld_acquire_global(&state->expert_token_offsets[expert_id]);
                        if (alloc_count < batch_end) {
                            // Not enough tokens allocated yet
                            s_priority_token_has_not_full = 1;
#ifdef MK_PERF_TRACE
                            priority_not_full += 1;
#endif
                            continue;
                        }
                        // Broadcast batch info for cooperative check
                        s_priority_expert_id = expert_id;
                        s_priority_batch_start = batch_start;
                        s_priority_batch_end = batch_end;
                        s_priority_ready_base = expert_id * max_tpe + batch_start;
                        s_priority_alloc_count = alloc_count;
                        break;
                    }
                }
                __syncthreads();

                if (s_priority_done)
                    break;
                if (s_priority_expert_id < 0) {
                    __syncthreads();
                    continue;
                }

                // All threads cooperatively check COMPUTE_BATCH_SIZE slots
                int ready_base = s_priority_ready_base;
                int my_ready = 1;
                for (int s = tid; s < COMPUTE_BATCH_SIZE; s += num_threads) {
                    if (ld_acquire_global(&state->expert_slot_ready[ready_base + s]) == 0) {
                        my_ready = 0;
                        break;
                    }
                }
                int batch_ready = __syncthreads_and(my_ready);

                // Thread 0 processes the result
                if (tid == 0) {
                    if (!batch_ready) {
                        s_priority_token_has_not_full = 1;
#ifdef MK_PERF_TRACE
                        priority_not_full += 1;
#endif
                    } else {
                        int expert_id = s_priority_expert_id;
                        int batch_id = s_priority_batch_start / COMPUTE_BATCH_SIZE;
                        bool enq = scheduler_try_enqueue_batch(state, expert_id, batch_id, s_priority_batch_start, COMPUTE_BATCH_SIZE, 0);
                        if (enq) {
                            s_priority_enqueued += 1;
#ifdef MK_PERF_TRACE
                            priority_full_batch_hits += 1;
#endif
                        } else {
#ifdef MK_PERF_TRACE
                            priority_batch_already_enqueued += 1;
#endif
                        }
                    }
                }
                __syncthreads();
            }

            // Advance cursor
            if (tid == 0) {
                if (dispatch_done && s_priority_new_cursor != s_priority_cursor_token)
                    st_na_release(state->priority_token_cursor, s_priority_new_cursor);
            }
        }
        __syncthreads();
        int priority_enqueued_local = (scheduler_id == 0) ? s_priority_enqueued : 0;

#ifdef MK_PERF_TRACE
        sched_priority_acc += globaltimer_ns() - sched_priority_start;
        int64_t sched_normal_start = globaltimer_ns();
#endif

        // === Normal enqueue (thread 0 only) ===
        if (tid == 0) {
            int task_head_for_fill = ld_acquire_sys_global(state->compute_task_head);
            int task_tail_for_fill = ld_acquire_sys_global(state->compute_task_tail);
            int queue_depth_for_fill = task_tail_for_fill - task_head_for_fill;
            int queue_low_watermark = state->num_compute_groups * 2;
            if (priority_enqueued_local == 0 || queue_depth_for_fill < queue_low_watermark) {
                for (int expert_id = scheduler_id; expert_id < num_local_experts; expert_id += num_schedulers) {
                    int count = ld_acquire_global(&state->expert_recv_count[expert_id]);
                    int cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
                    while (count - cursor >= COMPUTE_BATCH_SIZE) {
                        int batch_id = cursor / COMPUTE_BATCH_SIZE;
                        bool enq = scheduler_try_enqueue_batch(state, expert_id, batch_id, cursor, COMPUTE_BATCH_SIZE, 0);
                        cursor += COMPUTE_BATCH_SIZE;
                        st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
#ifdef MK_PERF_TRACE
                        if (enq)
                            normal_full_batch_enqueues += 1;
#endif
                    }
                }
            }
        }
        __syncthreads();
#ifdef MK_PERF_TRACE
        sched_normal_acc += globaltimer_ns() - sched_normal_start;
        sched_enqueue_acc += globaltimer_ns() - sched_enqueue_start;
#endif

        // === Tail flush (thread 0 only, with cooperative ready scan) ===
        if (dispatch_done && !tail_enqueued) {
#ifdef MK_PERF_TRACE
            int64_t sched_tail_flush_start = globaltimer_ns();
            sched_scan_start = globaltimer_ns();
#endif
            // Final cooperative scan after dispatch is done
            for (int expert_id = scheduler_id; expert_id < num_local_experts; expert_id += num_schedulers) {
                int old_count;
                if (tid == 0) {
                    old_count = ld_acquire_global(&state->expert_recv_count[expert_id]);
                    s_priority_alloc_count = old_count;
                }
                __syncthreads();
                old_count = s_priority_alloc_count;

                // Cooperative scan (same pattern as above)
                int count = old_count;
                while (count < max_tpe) {
                    int my_slot = count + tid;
                    int my_ready = 0;
                    if (my_slot < max_tpe) {
                        my_ready = (ld_acquire_global(&state->expert_slot_ready[expert_id * max_tpe + my_slot]) == 1) ? 1 : 0;
                    }
                    int all_ready = __syncthreads_and(my_ready || (my_slot >= max_tpe));
                    if (!all_ready) {
                        if (tid == 0) {
                            for (int s = count; s < count + num_threads && s < max_tpe; ++s) {
                                if (ld_acquire_global(&state->expert_slot_ready[expert_id * max_tpe + s]) == 1) {
                                    count = s + 1;
                                } else {
                                    break;
                                }
                            }
                            s_priority_alloc_count = count;
                        }
                        __syncthreads();
                        count = s_priority_alloc_count;
                        break;
                    }
                    count += num_threads;
                    if (count > max_tpe) count = max_tpe;
                }

                if (tid == 0) {
                    __threadfence();
                    st_na_release(&state->expert_recv_count[expert_id], count);

                    int cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
#ifdef MK_PERF_TRACE
                    sched_scan_acc += globaltimer_ns() - sched_scan_start;
                    sched_enqueue_start = globaltimer_ns();
#endif
                    while (count - cursor >= COMPUTE_BATCH_SIZE) {
                        int batch_id = cursor / COMPUTE_BATCH_SIZE;
                        scheduler_try_enqueue_batch(state, expert_id, batch_id, cursor, COMPUTE_BATCH_SIZE, 0);
                        cursor += COMPUTE_BATCH_SIZE;
                        st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
                    }
                    if (count > cursor) {
                        int batch_id = cursor / COMPUTE_BATCH_SIZE;
                        bool enq = scheduler_try_enqueue_batch(state, expert_id, batch_id, cursor, count - cursor, 1);
                        st_na_release(&state->expert_enqueue_cursor[expert_id], count);
#ifdef MK_PERF_TRACE
                        if (enq)
                            flush_tail_enqueues += 1;
#endif
                    }
#ifdef MK_PERF_TRACE
                    sched_enqueue_acc += globaltimer_ns() - sched_enqueue_start;
                    sched_scan_start = globaltimer_ns();
#endif
                }
                __syncthreads();
            }
#ifdef MK_PERF_TRACE
            int64_t sched_done_publish_start = globaltimer_ns();
#endif
            if (tid == 0) {
                __threadfence_system();
                int finished = atomicAdd(state->scheduler_done_count, 1) + 1;
                if (finished == num_schedulers) {
#ifdef MK_PERF_TRACE
                    state->perf_sched_ts[1] = globaltimer_ns();
#endif
                    st_release_sys_global(state->compute_enqueue_done, 1);
                }
            }
#ifdef MK_PERF_TRACE
            sched_enqueue_acc += globaltimer_ns() - sched_done_publish_start;
            sched_tail_flush_acc += globaltimer_ns() - sched_tail_flush_start;
#endif
            tail_enqueued = true;
        }

        if (tail_enqueued)
            break;

        __syncthreads();
#ifdef MK_PERF_TRACE
        int64_t sched_idle_start = globaltimer_ns();
#endif
        __nanosleep(64);
#ifdef MK_PERF_TRACE
        sched_idle_acc += globaltimer_ns() - sched_idle_start;
#endif
    }
#ifdef MK_PERF_TRACE
    if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_scan_ns),
              static_cast<unsigned long long>(sched_scan_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_enqueue_ns),
              static_cast<unsigned long long>(sched_enqueue_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_idle_ns),
              static_cast<unsigned long long>(sched_idle_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_ns),
              static_cast<unsigned long long>(sched_priority_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_normal_ns),
              static_cast<unsigned long long>(sched_normal_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_tail_flush_ns),
              static_cast<unsigned long long>(sched_tail_flush_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_scan_tokens),
              static_cast<unsigned long long>(priority_scan_tokens));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_ready_tokens),
              static_cast<unsigned long long>(priority_ready_tokens));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_full_batch_hits),
              static_cast<unsigned long long>(priority_full_batch_hits));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_batch_already_enqueued),
              static_cast<unsigned long long>(priority_batch_already_enqueued));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_not_full),
              static_cast<unsigned long long>(priority_not_full));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_normal_full_batch_enqueues),
              static_cast<unsigned long long>(normal_full_batch_enqueues));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_flush_tail_enqueues),
              static_cast<unsigned long long>(flush_tail_enqueues));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_queue_empty_count),
              static_cast<unsigned long long>(queue_empty_count));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_queue_empty_after_dispatch_count),
              static_cast<unsigned long long>(queue_empty_after_dispatch_count));
    unsigned long long old_gap = atomicMax(reinterpret_cast<unsigned long long*>(state->perf_sched_max_ready_tail_gap),
                                           static_cast<unsigned long long>(max_ready_tail_gap));
    if (static_cast<unsigned long long>(max_ready_tail_gap) > old_gap && max_ready_tail_gap > 0) {
        st_na_global(state->perf_sched_stall_expert, stall_expert);
        st_na_global(state->perf_sched_stall_recv_count, stall_recv_count);
        st_na_global(state->perf_sched_stall_alloc_count, stall_alloc_count);
        st_na_global(state->perf_sched_stall_enqueue_cursor, stall_enqueue_cursor);
        st_na_global(state->perf_sched_stall_first_unready_slot, stall_first_unready_slot);
        st_na_global(state->perf_sched_stall_first_unready_ready, stall_first_unready_ready);
        st_na_global(state->perf_sched_stall_dispatch_done, stall_dispatch_done);
    }
    }  // tid == 0
#endif
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
    const int gu_stride   = padded_m * (2 * intermediate);   // reserved GU scratch [M,2I]
    const int act_stride  = padded_m * intermediate;         // interleaved SwiGLU epilogue result (down A)
    const int down_stride = padded_m * hidden;
    const int gemm_stride = input_stride + gu_stride + act_stride + down_stride;
    __nv_bfloat16* input_buf = state->gemm_workspace + group_id * gemm_stride;
    __nv_bfloat16* gu_buf   = input_buf + input_stride;   // reserved GU scratch [M,2I]
    __nv_bfloat16* up_buf   = gu_buf + gu_stride;         // act = silu(gate)*up*route_w (down-proj A operand)
    __nv_bfloat16* down_buf = up_buf + act_stride;

    __shared__ int s_recv_token_idx[COMPUTE_BATCH_SIZE];
    __shared__ int s_topk_slot[COMPUTE_BATCH_SIZE];
    __shared__ int s_expected[COMPUTE_BATCH_SIZE];
    __shared__ int s_is_last[COMPUTE_BATCH_SIZE];
    __shared__ int s_has_multi_finalize;
    // Per-row route weight, gathered once and consumed inside the fused
    // gate+up SwiGLU epilogue (act = silu(gate) * up * route_w).
    __shared__ float s_route_w[COMPUTE_BATCH_SIZE];

    // TMEM alloc-once flag: first UMMA call allocates, subsequent calls reuse.
    bool umma_tmem_allocated = false;
#ifdef MK_PERF_TRACE
    int64_t last_task_end_ns = 0;
#endif

    while (true) {
#ifdef MK_PERF_TRACE
        int64_t pop_start_ns = 0;
        int64_t pop_done_ns = 0;
        int pop_attempts = 0;
        int cas_failures = 0;
#endif
        if (group_sm_idx == 0 && thread_id == 0) {
            int task_idx = -1;
#ifdef MK_PERF_TRACE
            pop_start_ns = globaltimer_ns();
#endif
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
#ifdef MK_PERF_TRACE
                    if (task_idx >= 0 && task_idx < state->max_compute_tasks) {
                        state->perf_task_pop_start_ts[task_idx] = pop_start_ns;
                        state->perf_task_pop_done_ts[task_idx] = pop_done_ns;
                        state->perf_task_pop_attempts[task_idx] = pop_attempts;
                        state->perf_task_cas_failures[task_idx] = cas_failures;
                        state->perf_task_group_id[task_idx] = group_id;
                    }
#endif
                    break;
                }
#ifdef MK_PERF_TRACE
                cas_failures += 1;
#endif
            }
            st_release_sys_global(&state->compute_group_task_idx[group_id], task_idx);
        }
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);

        int task_idx = ld_acquire_sys_global(&state->compute_group_task_idx[group_id]);
#ifdef MK_PERF_TRACE
        if (group_sm_idx == 0 && thread_id == 0 && task_idx >= 0 && task_idx < state->max_compute_tasks)
            state->perf_task_bcast_done_ts[task_idx] = globaltimer_ns();
#endif
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

        // MK_COMPUTE_KERNEL selects the compute implementation at compile time:
        //   0 = WMMA gate/up + WMMA down
        //   1 = 1-CTA UMMA gate/up + 1-CTA UMMA down
        //   2 = 2-CTA UMMA gate/up + 2-CTA UMMA down
        constexpr bool kUseUmmaCompute = (MK_COMPUTE_KERNEL != 0);
        constexpr int kUmmaClusterDim = (MK_COMPUTE_KERNEL == 2 ? 2 : 1);
        constexpr int kUmmaClustersPerGroup = COMPUTE_GROUP_SIZE / kUmmaClusterDim;

        // DeepGEMM mega_moe prefetches TMA descriptors before the main data movement.
        // Do the same after task decode so descriptor fetch can overlap input gather.
        if constexpr (kUseUmmaCompute) {
            if (state->compute_tma != nullptr && state->compute_down_tma != nullptr &&
                batch_size <= COMPUTE_BATCH_SIZE) {
                const umma::InputTmaAtom_t& prefetch_atom = state->group_input_tma[group_id];
                if (local_warp_id == 0) {
                    cute::prefetch_tma_descriptor(&prefetch_atom.a);
                    cute::prefetch_tma_descriptor(&state->compute_tma->wgateup[expert_id]);
                    cute::prefetch_tma_descriptor(&prefetch_atom.act_cd);
                    cute::prefetch_tma_descriptor(&prefetch_atom.act_a);
                    cute::prefetch_tma_descriptor(&state->compute_down_tma->wdown[expert_id]);
                    cute::prefetch_tma_descriptor(&prefetch_atom.down_cd);
                }
            }
        }
#ifdef MK_PERF_TRACE
        const bool perf_leader = (group_sm_idx == 0 && thread_id == 0);
        int64_t compute_task_start_ns = perf_leader ? globaltimer_ns() : 0;
        if (perf_leader && task_idx >= 0 && task_idx < state->max_compute_tasks) {
            state->perf_task_start_ts[task_idx] = compute_task_start_ns;
            state->perf_task_prev_gap_ns[task_idx] = last_task_end_ns == 0 ? 0 : compute_task_start_ns - last_task_end_ns;
        }
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
        // Output-phase task-has-multi decision (before slow path recomputes
        // s_has_multi_finalize against group_is_last). 1 = slow path was taken.
        __shared__ int s_perf_task_has_multi;
        if (perf_leader) {
            s_perf_up = umma::UmmaPerf{};
            s_perf_down = umma::UmmaPerf{};
            s_perf_multi_expert_rows = 0;
            s_perf_task_has_multi = 0;
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
        // Zero-init padding rows [batch_size, COMPUTE_BATCH_SIZE) so the UMMA path
        // can run at fixed M=256 for tail batches: padded input_buf rows are 0, and
        // route_w must be defined (SwiGLU on padding is 0 anyway, but avoid reading
        // uninitialized shared memory). Output/reduce/signal all mask by batch_size,
        // so padding rows never leave the kernel.
        for (int i = batch_size + thread_id; i < COMPUTE_BATCH_SIZE; i += blockDim.x) {
            s_route_w[i] = 0.0f;
        }
        __syncthreads();
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_meta_ns = globaltimer_ns();
#endif

        const int hidden_int4 = hidden * sizeof(__nv_bfloat16) / sizeof(int4);
        const int input_vec_stride = COMPUTE_BATCH_SIZE * hidden_int4;
        const int4* combine_input_i4 = reinterpret_cast<const int4*>(state->combine_input);
        int4* input_buf_i4 = reinterpret_cast<int4*>(input_buf);
        for (int idx = group_thread_id; idx < input_vec_stride; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            input_buf_i4[idx] = (row < batch_size)
                ? combine_input_i4[(int64_t)s_recv_token_idx[row] * hidden_int4 + v]
                : make_int4(0, 0, 0, 0);
        }
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_input_ns = globaltimer_ns();
#endif

        // Expert weight slices
        const __nv_bfloat16* w_gateup = &state->W_gateup[expert_id * 2 * intermediate * hidden];
        const __nv_bfloat16* w_down = &state->W_down[expert_id * hidden * intermediate];

        // Gate/up compute always consumes pairwise interleaved W_gateup rows
        // [g0,u0,g1,u1,...]. UMMA folds adjacent gate/up columns in its epilogue;
        // WMMA fallback reads the same layout with a 2*K B-matrix stride.
        // umma_accum_iter tracks TMEM accumulator pipeline phase (tmem_full/tmem_empty
        // barrier ring) across ALL three GEMMs (gate, up, down). Must NOT be reset
        // between gate/up and down-proj — the barrier ring is initialized once and
        // must stay in phase. Declared here so it spans both if-blocks below.
        uint32_t umma_accum_iter = 0;

        // Tail batches (batch_size < COMPUTE_BATCH_SIZE) also run the UMMA path at
        // FIXED M=256: input_buf rows [batch_size,256) are zero-padded above, so the
        // GEMM computes 256 rows (padding rows -> 0, harmless) but SwiGLU/output/
        // reduce/signal all mask by batch_size, so padding never leaves the kernel.
        // The 2-CTA UMMA M-tile is 256 regardless, so padding costs no extra time.
        if (kUseUmmaCompute && state->compute_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];

            // Interleaved gate/up fusion: ONE persistent GEMM computes GU with
            // Wgu rows [g0,u0,g1,u1,...], then the epilogue folds each adjacent
            // gate/up pair directly from TMEM into act_buf (up_buf). This is the
            // microkernel path moved into the megakernel for both 1-CTA and 2-CTA.
            umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
            umma::umma_gateup_interleaved_persistent(
                &in_atom.a,
                &state->compute_tma->wgateup[expert_id],
                &in_atom.act_cd,
                s_route_w,
                COMPUTE_BATCH_SIZE, intermediate, hidden,
                cluster_in_group, num_clusters,
                cluster_smem, umma_accum_iter);
            umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);
            umma_tmem_allocated = false;   // freed each task (4a)
#ifdef MK_PERF_TRACE
            if (perf_leader) perf_up_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
        } else {
            device_gemm_swiglu_fused(input_buf, w_gateup, up_buf, s_route_w,
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
        // Same task-level PERSISTENT lifecycle as gate/up: init barriers+TMEM
        // once, run the persistent down GEMM (tile loop inside the three warp
        // roles, zero cluster sync between tiles), dealloc once. A fresh accum
        // counter is used because gate/up already freed TMEM at their dealloc.
        // Tail batches run at FIXED M=256: act rows [batch_size,256) hold the
        // SwiGLU of zero-padded gate/up (== 0), so down output rows [batch_size,256)
        // are 0 and are masked off by the batch_size-bounded output/reduce below.
        if (kUseUmmaCompute &&
            state->compute_down_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];

            uint32_t down_accum_iter = 0;
            umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
            umma::umma_down_persistent(
                &in_atom.act_a,
                &state->compute_down_tma->wdown[expert_id],
                &in_atom.down_cd,
                COMPUTE_BATCH_SIZE, hidden, intermediate,
                cluster_in_group, num_clusters,
                cluster_smem, down_accum_iter);
            umma::dg_dealloc_tmem<umma::kDgRunMulticast>(cluster_smem);
            umma_tmem_allocated = false;
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

        // Per-slot output (change A): every (recv_token, local-expert-hit) writes its OWN
        // expert-sorted slot row in compute_output_slot. Slots are unique across tasks, so
        // no write conflict and no atomic. Same-rank multi-expert reduce is deferred to the
        // combine sender (change C), which gathers a token's nh slots and fp32-sums them.
#ifdef MK_PERF_TRACE
        if (thread_id == 0) {
            int multi_rows = 0;
            for (int row = 0; row < batch_size; ++row)
                if (s_expected[row] > 1) ++multi_rows;
            if (perf_leader) s_perf_task_has_multi = (multi_rows != 0);
        }
        __syncthreads();
#endif

        const int4* down_i4 = reinterpret_cast<const int4*>(down_buf);
        int4* slot_out_i4 = reinterpret_cast<int4*>(state->compute_output_slot);
        const int slot_base = expert_id * max_tpe + start_slot;
        for (int idx = group_thread_id; idx < batch_size * hidden_int4; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            int slot = slot_base + row;
            slot_out_i4[(int64_t)slot * hidden_int4 + v] = down_i4[idx];
        }
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_out_body_ns = globaltimer_ns();
#endif
        // device-scope fence: combine_worker reads compute_output_slot on the same GPU
        // (different SM, same kernel launch), so device-scope visibility is sufficient.
        // Each thread fences its own per-slot writes before the group sync lets any SM
        // publish ready flags.
        __threadfence();
        compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
        if (perf_leader) perf_ph_output_ns = globaltimer_ns();
#endif

        // ==== Signal: per-slot ready publish (change B) ====
        // compute wrote per-slot rows into compute_output_slot. The same-rank multi-expert
        // reduce is now done by the combine sender (change C), which gathers a token's nh
        // slots and fp32-sums them before sending. So compute only needs to publish a
        // per-slot ready flag; NO donecount, NO group_is_last broadcast, NO fp32 finalize.
        {
#ifdef MK_PERF_TRACE
            if (perf_leader) {
                perf_sig_donecount_ns = globaltimer_ns();
                perf_sig_finalize_ns = perf_sig_donecount_ns;
                perf_sig_fence_ns = perf_sig_donecount_ns;
                int mr = 0;
                for (int row = 0; row < batch_size; ++row)
                    if (s_expected[row] > 1) ++mr;
                s_perf_multi_expert_rows = mr;
            }
#endif
            const int slot_base_sig = expert_id * max_tpe + start_slot;
            const int task_is_flush = task.is_flush;
#ifdef MK_PERF_TRACE
            const int64_t slot_ready_ts = globaltimer_ns();
#endif
            for (int row = group_thread_id; row < batch_size; row += group_num_threads) {
                const int slot = slot_base_sig + row;
                st_na_global(&state->compute_slot_from_flush[slot], task_is_flush);
#ifdef MK_PERF_TRACE
                st_na_global(&state->compute_slot_ready_ts[slot], slot_ready_ts);
#endif
                st_na_release(&state->compute_slot_ready[slot], 1);
                const int recv_token = s_recv_token_idx[row];
                atomicAdd(&state->token_done_count[recv_token], 1);
            }
            compute_group_sync(state, group_id, COMPUTE_GROUP_SIZE);
#ifdef MK_PERF_TRACE
            if (perf_leader) perf_sig_publish_ns = globaltimer_ns();
#endif
        }

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
                rec[21] = task_idx;
                rec[22] = start_slot;
                rec[23] = start_slot + batch_size;
                rec[24] = static_cast<int64_t>(expert_id) * max_tpe + start_slot;
                rec[25] = task.is_flush;
                last_task_end_ns = perf_ph_signal_ns;
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
                state->perf_compute_task_has_multi[slot] = s_perf_task_has_multi;
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
// Template constants: kNumRDMARanks is selected at launch by SWITCH_RDMA_RANKS.
// ============================================================================

template <int kNumRDMARanks, int kStage>
__device__ void combine_worker_v2(
    int combine_sm_idx,       // 0-based index among combine SMs
    MegaKernelState* state
) {
    using namespace internode;
    using dtype_t = nv_bfloat16;

    const int num_tokens = state->combine_num_tokens;
    const int num_combined_tokens = state->combine_num_combined_tokens;
    const int num_channels = state->num_combine_channels;
    constexpr int num_logical_channels_per_physical = kStage;
    const int num_logical_channels = num_channels * num_logical_channels_per_physical;
    EP_DEVICE_ASSERT(num_channels == state->num_dispatch_channels);  // physical channel counts must match for queue reuse
    const int num_ranks = state->num_ranks;
    constexpr int kNumRDMARanks_C = kNumRDMARanks;
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

    using RdmaCfg = MegaKernelRdmaConfig<kNumRDMARanks>;
    constexpr int kNumForwarders_C = RdmaCfg::kNumCombineForwarders;
    constexpr int kNumWarpsPerForwarder_C = RdmaCfg::kNumCombineWarpsPerForwarder;
    constexpr int kNumRDMAReceivers_C = RdmaCfg::kNumCombineRDMAReceivers;
    constexpr int kNumTopkRDMARanks_C = RdmaCfg::kNumTopkCombineRDMARanks;

    const auto sm_id = combine_sm_idx;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    const int num_topk = state->num_topk;
    const int hidden = state->combine_hidden;
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const int hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const int hidden_bytes = hidden_int4 * sizeof(int4);
    const int num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

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

    for (int logical_stage = 0; logical_stage < num_logical_channels_per_physical; ++logical_stage) {
        const int logical_channel_id = channel_id * num_logical_channels_per_physical + logical_stage;
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
        constexpr int kGatherChunkInt4 = 128;
        constexpr int kGatherChunkBytes = kGatherChunkInt4 * sizeof(int4);
        constexpr int kGatherNumStages = 2;
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumCombineTMABytesPerSenderWarp;
        auto gather_mbarrier = [=](int stage) {
            return reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token +
                                               stage * sizeof(uint64_t));
        };
        auto gather_load_buffer = [=](int stage) {
            return tma_buffer + num_bytes_per_token + kGatherNumStages * sizeof(uint64_t) +
                   stage * kGatherChunkBytes;
        };
        uint32_t gather_tma_phase[kGatherNumStages] = {0, 0};
        if (lane_id < kGatherNumStages)
            mbarrier_init(gather_mbarrier(lane_id), 1);
        if (elect_one_sync()) {
            fence_barrier_init();
            EP_DEVICE_ASSERT(num_bytes_per_token + kGatherNumStages * sizeof(uint64_t) +
                             kGatherNumStages * kGatherChunkBytes <= kNumCombineTMABytesPerSenderWarp);
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
                    // Change B/C: the per-slot ready wait + gather-reduce below replaces the
                    // old per-token ready gate. compute now publishes per-slot flags
                    // (compute_slot_ready), waited inside the gather block. nh==0 tokens
                    // (no local expert) send zeros.
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
                    // if (lane_id == 0) {
                    //     printf("[MK-TRACE][COMBINE][NVL-SENDER][SEND] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d dst_nvl_rank=%d current_rdma_idx=%d token=%lld dst_slot=%d shifted_buffer=%p src_meta_addr=%p topk_weights_addr=%p tma_buffer=%p hidden_int4=%d num_topk=%d\n",
                    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, dst_nvl_rank, current_rdma_idx,
                    //            (long long)token_idx, dst_slot_idx, shifted_x_buffers, src_meta + token_idx,
                    //            topk_weights + token_idx * num_topk, tma_buffer, hidden_int4, num_topk);
                    // }
#ifdef MK_PERF_TRACE
                    int64_t comb_tma_wait_ns = 0;
                    int64_t comb_wait_ready_ns = 0;
                    int64_t comb_wait_ready_single_ns = 0;
                    int64_t comb_wait_ready_multi_ns = 0;
                    int64_t comb_wait_ready_flush_ns = 0;
                    int64_t comb_wait_ready_full_ns = 0;
                    int64_t comb_wait_ready_flush_count = 0;
                    int64_t comb_wait_ready_full_count = 0;
                    int64_t comb_wait_top_ns = 0;
                    int64_t comb_wait_top_slot = -1;
                    int64_t comb_wait_top_from_flush = -1;
                    int64_t comb_gather_reduce_ns = 0;
                    int64_t comb_gather_single_ns = 0;
                    int64_t comb_gather_multi_ns = 0;
                    int64_t comb_pack_meta_ns = 0;
                    int64_t comb_pack_meta_work_ns = 0;
                    int64_t comb_pack_meta_sync_ns = 0;
                    int64_t comb_tma_store_ns = 0;
                    int64_t phase_start_ns = lane_id == 0 ? globaltimer_ns() : 0;
#endif
                    tma_store_wait<0>();
#ifdef MK_PERF_TRACE
                    if (lane_id == 0) {
                        int64_t now = globaltimer_ns();
                        comb_tma_wait_ns = now - phase_start_ns;
                        phase_start_ns = now;
                    }
#endif
                    // Change C: gather this token's nh local-expert slots and fp32-reduce
                    // them directly into tma_buffer (per-lane registers). compute now writes
                    // per-slot rows to compute_output_slot; the same-rank reduce happens here.
                    const int nh = state->token_nhits[token_idx];
                    const bool is_single_hit = (nh == 1);
                    // All tokens pass through gather now. Gather does no work for nhits=0,
                    // only signals for nhits=1, and performs local reduce for nhits>1.
                    auto gather_wait_start = clock64();
                    while (ld_acquire_global(&state->combine_token_ready[token_idx]) != 1) {
                        if (clock64() - gather_wait_start > NUM_TIMEOUT_CYCLES) {
                            if (timeout_log_once(state, kTimeoutLogComputeReady))
                                printf("MK combine gather-semaphore timeout, rank=%d token=%lld nh=%d\n",
                                       state->rank, (long long)token_idx, nh);
                            trap();
                        }
                        __nanosleep(32);
                    }
                    {
                        constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(__nv_bfloat16);
                        int4* tma_i4 = reinterpret_cast<int4*>(tma_buffer);
                        const int4* slot_base_i4 = reinterpret_cast<const int4*>(state->compute_output_slot);
                        // Per-slot ready wait (change B): each lane waits on one slot's ready
                        // flag; nh <= num_topk <= 32 so a single warp covers all slots.
#ifdef MK_PERF_TRACE
                        int64_t lane_slot_wait_ns = 0;
                        int lane_wait_slot = -1;
                        int lane_wait_from_flush = 0;
#endif
                        // Per-slot ready wait: only for nh<=1. For nh>1, gather SM
                        // already did ld_acquire per slot before reducing.
                        if (nh <= 1 && lane_id < nh) {
                            int slot = state->token_slot_list[token_idx * num_topk + lane_id];
                            auto wait_start = clock64();
#ifdef MK_PERF_TRACE
                            int64_t wait_start_ns = globaltimer_ns();
#endif
                            while (ld_acquire_global(&state->compute_slot_ready[slot]) != 1) {
                                if (clock64() - wait_start > NUM_TIMEOUT_CYCLES) {
                                    if (timeout_log_once(state, kTimeoutLogComputeReady))
                                        printf("MK combine per-slot-ready timeout, rank=%d token=%lld slot=%d nh=%d\n",
                                               state->rank, (long long)token_idx, slot, nh);
                                    trap();
                                }
                                __nanosleep(32);
                            }
#ifdef MK_PERF_TRACE
                            lane_slot_wait_ns = globaltimer_ns() - wait_start_ns;
                            lane_wait_slot = slot;
                            lane_wait_from_flush = ld_nc_global(&state->compute_slot_from_flush[slot]);
#endif
                        }
                        __syncwarp();
#ifdef MK_PERF_TRACE
                        int64_t wait_ready_flush_sum = 0;
                        int64_t wait_ready_full_sum = 0;
                        int64_t wait_ready_flush_slots = 0;
                        int64_t wait_ready_full_slots = 0;
                        int64_t wait_top_ns = 0;
                        int wait_top_slot = -1;
                        int wait_top_from_flush = -1;
                        #pragma unroll
                        for (int l = 0; l < 32; ++l) {
                            int64_t w = __shfl_sync(0xffffffff, lane_slot_wait_ns, l);
                            int slot = __shfl_sync(0xffffffff, lane_wait_slot, l);
                            int from_flush = __shfl_sync(0xffffffff, lane_wait_from_flush, l);
                            if (l < nh) {
                                if (from_flush) {
                                    wait_ready_flush_sum += w;
                                    wait_ready_flush_slots += 1;
                                } else {
                                    wait_ready_full_sum += w;
                                    wait_ready_full_slots += 1;
                                }
                                if (w > wait_top_ns) {
                                    wait_top_ns = w;
                                    wait_top_slot = slot;
                                    wait_top_from_flush = from_flush;
                                }
                            }
                        }
                        if (lane_id == 0) {
                            int64_t now = globaltimer_ns();
                            comb_wait_ready_ns = now - phase_start_ns;
                            if (is_single_hit)
                                comb_wait_ready_single_ns = comb_wait_ready_ns;
                            else
                                comb_wait_ready_multi_ns = comb_wait_ready_ns;
                            comb_wait_ready_flush_ns = wait_ready_flush_sum;
                            comb_wait_ready_full_ns = wait_ready_full_sum;
                            comb_wait_ready_flush_count = wait_ready_flush_slots;
                            comb_wait_ready_full_count = wait_ready_full_slots;
                            comb_wait_top_ns = wait_top_ns;
                            comb_wait_top_slot = wait_top_slot;
                            comb_wait_top_from_flush = wait_top_from_flush;
                            phase_start_ns = now;
                        }
#endif
                        constexpr int kVecsPerLane = kGatherChunkInt4 / 32;
                        if (true) {
                            const int slot = (nh > 0) ? state->token_slot_list[token_idx * num_topk] : 0;
#ifdef COMBINE_TMA_LOAD
                            if (nh > 0) {
                                if (lane_id == 0) {
                                    tma_load_1d(tma_buffer,
                                                slot_base_i4 + (int64_t)slot * hidden_int4,
                                                gather_mbarrier(0), hidden_bytes, false);
                                    mbarrier_arrive_and_expect_tx(gather_mbarrier(0), hidden_bytes);
                                }
                                __syncwarp();
                                mbarrier_wait(gather_mbarrier(0), gather_tma_phase[0]);
                            } else {
                                for (int vi = lane_id; vi < hidden_int4; vi += 32)
                                    tma_i4[vi] = make_int4(0, 0, 0, 0);
                            }
#else
                            for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kGatherChunkInt4) {
                                const int chunk_end = min(chunk_base + kGatherChunkInt4, hidden_int4);
                                #pragma unroll
                                for (int j = 0; j < kVecsPerLane; ++j) {
                                    const int vi = chunk_base + lane_id + j * 32;
                                    if (vi < chunk_end) {
                                        if (nh > 1)
                                            // gather SM wrote reduced data to first_slot; use coherent load
                                            tma_i4[vi] = slot_base_i4[(int64_t)slot * hidden_int4 + vi];
                                        else
                                            tma_i4[vi] = (nh > 0) ? ld_nc_global(slot_base_i4 + (int64_t)slot * hidden_int4 + vi) : make_int4(0, 0, 0, 0);
                                    }
                                }
                            }
#endif
                        } else {
                            // Chunked smem-load reduce: use the sender warp's spare shared
                            // memory to TMA-load each slot chunk, then reduce from smem in
                            // registers. This keeps the current expert-slot-major layout while
                            // moving the nh>1 path closer to mega_moe's combine pipeline.
                            for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kGatherChunkInt4) {
                                const int chunk_end = min(chunk_base + kGatherChunkInt4, hidden_int4);
                                const int chunk_int4 = chunk_end - chunk_base;
                                const int chunk_bytes = chunk_int4 * static_cast<int>(sizeof(int4));
                                constexpr int kBfloat162PerInt4 = kElemsPerInt4 / 2;
                                float2 acc[kVecsPerLane][kBfloat162PerInt4];
                                #pragma unroll
                                for (int j = 0; j < kVecsPerLane; ++j) {
                                    #pragma unroll
                                    for (int p = 0; p < kBfloat162PerInt4; ++p)
                                        acc[j][p] = make_float2(0.0f, 0.0f);
                                }
                                auto issue_slot_chunk = [&](int stage_idx, int hit_idx) {
                                    int slot = state->token_slot_list[token_idx * num_topk + hit_idx];
                                    tma_load_1d(gather_load_buffer(stage_idx),
                                                slot_base_i4 + (int64_t)slot * hidden_int4 + chunk_base,
                                                gather_mbarrier(stage_idx), chunk_bytes, false);
                                    mbarrier_arrive_and_expect_tx(gather_mbarrier(stage_idx), chunk_bytes);
                                };
                                int stage = 0;
                                if (lane_id == 0)
                                    issue_slot_chunk(stage, 0);
                                for (int k = 0; k < nh; ++k) {
                                    const int cur_stage = stage;
                                    const int next_stage = stage ^ 1;
                                    mbarrier_wait(gather_mbarrier(cur_stage), gather_tma_phase[cur_stage]);
                                    if (k + 1 < nh && lane_id == 0)
                                        issue_slot_chunk(next_stage, k + 1);
                                    const int4* smem_i4 = reinterpret_cast<const int4*>(gather_load_buffer(cur_stage));
                                    #pragma unroll
                                    for (int j = 0; j < kVecsPerLane; ++j) {
                                        const int local_vi = lane_id + j * 32;
                                        if (local_vi < chunk_int4) {
                                            int4 raw = smem_i4[local_vi];
                                            const __nv_bfloat162* bv2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                                            #pragma unroll
                                            for (int p = 0; p < kBfloat162PerInt4; ++p) {
                                                float2 v = __bfloat1622float2(bv2[p]);
                                                acc[j][p].x += v.x;
                                                acc[j][p].y += v.y;
                                            }
                                        }
                                    }
                                    stage = next_stage;
                                }
                                #pragma unroll
                                for (int j = 0; j < kVecsPerLane; ++j) {
                                    const int vi = chunk_base + lane_id + j * 32;
                                    if (vi < chunk_end) {
                                        int4 packed;
                                        __nv_bfloat162* pv2 = reinterpret_cast<__nv_bfloat162*>(&packed);
                                        #pragma unroll
                                        for (int p = 0; p < kBfloat162PerInt4; ++p)
                                            pv2[p] = __float22bfloat162_rn(acc[j][p]);
                                        tma_i4[vi] = packed;
                                    }
                                }
                            }
                        }
                    }
                    __syncwarp();
#ifdef MK_PERF_TRACE
                    if (lane_id == 0) {
                        int64_t now = globaltimer_ns();
                        comb_gather_reduce_ns = now - phase_start_ns;
                        if (is_single_hit)
                            comb_gather_single_ns = comb_gather_reduce_ns;
                        else
                            comb_gather_multi_ns = comb_gather_reduce_ns;
                        phase_start_ns = now;
                    }
#endif

                    if (lane_id == 0)
                        *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);

                    if (lane_id < num_topk)
                        *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) =
                            ld_nc_global(topk_weights + token_idx * num_topk + lane_id);
                    const int meta_end = hidden_bytes + sizeof(SourceMeta) + num_topk * sizeof(float);
                    if (lane_id == 0) {
                        for (int byte_idx = meta_end; byte_idx < num_bytes_per_token; ++byte_idx)
                            tma_buffer[byte_idx] = 0;
                    }
#ifdef MK_PERF_TRACE
                    if (lane_id == 0) {
                        int64_t now = globaltimer_ns();
                        comb_pack_meta_work_ns = now - phase_start_ns;
                        phase_start_ns = now;
                    }
#endif
                    __syncwarp();
#ifdef MK_PERF_TRACE
                    if (lane_id == 0) {
                        int64_t now = globaltimer_ns();
                        comb_pack_meta_sync_ns = now - phase_start_ns;
                        comb_pack_meta_ns = comb_pack_meta_work_ns + comb_pack_meta_sync_ns;
                        phase_start_ns = now;
                    }
#endif

#ifdef MK_TOKEN_TRACE
                    if (lane_id == 0) {
                        auto* hptr = reinterpret_cast<nv_bfloat16*>(tma_buffer);
                        SourceMeta send_meta = ld_nc_global(src_meta + token_idx);
                        int send_prefix_idx = (current_rdma_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_logical_channels + logical_channel_id;
                        int send_base = gbl_channel_prefix_matrix[send_prefix_idx];
                        printf("[MK-TOKEN][COMBINE-NVL-SEND] rank=%d token=%lld dst_nvl=%d src_rdma=%d ch=%d logical_ch=%d sender_prefix_idx=%d sender_base=%d queue_tail_before=%d sender_queue_token=%d dst_slot=%d dst_lane_slot=%d meta=(%d,0x%x) tail_ptr=%p dst_ptr=%p topk_w0=%f topk_w1=%f h0=%f\n",
                               state->rank, (long long)token_idx, dst_nvl_rank, current_rdma_idx, channel_id, logical_channel_id,
                               send_prefix_idx, send_base, queue_tail_before, send_base + queue_tail_before,
                               dst_slot_idx, dst_slot_idx % num_max_nvl_chunked_recv_tokens_per_rdma,
                               send_meta.src_rdma_rank, send_meta.is_token_in_nvl_rank_bits,
                               (void*)(nvl_channel_tail.buffer() + current_rdma_idx),
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
#ifdef MK_PERF_TRACE
                    if (lane_id == 0) {
                        int64_t now = globaltimer_ns();
                        comb_tma_store_ns = now - phase_start_ns;
                        int acc_idx = logical_channel_id * 2;
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_tma_wait_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_tma_wait_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_single_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_single_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_multi_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_multi_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_flush_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_flush_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_full_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_full_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_flush_count[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_flush_count));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_full_count[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_full_count));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_gather_reduce_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_gather_reduce_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_gather_single_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_gather_single_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_gather_multi_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_gather_multi_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_pack_meta_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_pack_meta_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_pack_meta_work_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_pack_meta_work_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_pack_meta_sync_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_pack_meta_sync_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_tma_store_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_tma_store_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_tma_wait_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_tma_wait_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_single_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_single_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_multi_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_multi_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_flush_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_flush_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_ready_full_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_wait_ready_full_ns));
                        unsigned long long prev_top = atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_wait_top_ns[acc_idx]),
                                                                static_cast<unsigned long long>(comb_wait_top_ns));
                        if (static_cast<unsigned long long>(comb_wait_top_ns) > prev_top) {
                            state->perf_comb_wait_top_token[acc_idx] = token_idx;
                            state->perf_comb_wait_top_slot[acc_idx] = comb_wait_top_slot;
                            state->perf_comb_wait_top_expert[acc_idx] = comb_wait_top_slot >= 0 ? comb_wait_top_slot / state->max_tokens_per_expert : -1;
                            state->perf_comb_wait_top_from_flush[acc_idx] = comb_wait_top_from_flush;
                        }
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_gather_reduce_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_gather_reduce_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_gather_single_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_gather_single_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_gather_multi_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_gather_multi_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_pack_meta_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_pack_meta_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_pack_meta_work_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_pack_meta_work_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_pack_meta_sync_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_pack_meta_sync_ns));
                        atomicMax(reinterpret_cast<unsigned long long*>(&state->perf_comb_tma_store_max_ns[acc_idx]),
                                  static_cast<unsigned long long>(comb_tma_store_ns));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_nhit_sum[acc_idx]),
                                  static_cast<unsigned long long>(nh));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_token_count[acc_idx]),
                                  static_cast<unsigned long long>(1));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_single_token_count[acc_idx]),
                                  static_cast<unsigned long long>(is_single_hit ? 1 : 0));
                        atomicAdd(reinterpret_cast<unsigned long long*>(&state->perf_comb_multi_token_count[acc_idx]),
                                  static_cast<unsigned long long>(is_single_hit ? 0 : 1));
                    }
#endif
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

    __syncthreads();
    if (thread_id == 0) {
        int finished = atomicAdd(state->combine_done_count, 1) + 1;
        if (finished == state->num_combine_sms) {
            __threadfence();
            atomicExch(state->combine_all_done, 1);
        }
    }
}

// ============================================================================
// Dedicated Gather Worker
// Releases combine_token_ready for nhits <= 1 tokens and performs local
// same-rank reduce for nhits > 1 tokens into the token's first_slot.
// ============================================================================

__device__ void gather_worker(MegaKernelState* state, int gather_sm_idx) {
    const int tid = threadIdx.x;
    const int total_tokens = state->combine_num_tokens;
    if (total_tokens == 0) return;

    int cursor = 0;
    __shared__ int s_should_exit;
    while (true) {
        if (tid == 0)
            s_should_exit = (ld_acquire_global(state->combine_all_done) != 0) ? 1 : 0;
        __syncthreads();
        if (s_should_exit) break;

        int token_idx = cursor;
        cursor++;
        if (cursor >= total_tokens) cursor = 0;

        if (ld_nc_global(&state->gather_claimed[token_idx]) != 0) continue;

        // Must ensure dispatch has finished writing ALL hits for this token.
        // Race: multiple dispatch SMs may atomicAdd token_nhits/token_compute_expected
        // at different times. We need both to be equal AND token_done_count to match,
        // guaranteeing all dispatches + all computes are done for this token.
        int nhits = ld_acquire_global(&state->token_nhits[token_idx]);
        if (nhits == 0) continue;
        int expected = ld_acquire_global(&state->token_compute_expected[token_idx]);
        if (nhits != expected) continue;  // dispatch still in progress

        int completed = 0;
        if (tid == 0)
            completed = atomicAdd(&state->token_done_count[token_idx], 0);
        __shared__ int s_completed;
        if (tid == 0) s_completed = completed;
        __syncthreads();
        completed = s_completed;
        if (completed < nhits) continue;

        int claimed_success = 0;
        if (tid == 0)
            claimed_success = (atomicCAS(&state->gather_claimed[token_idx], 0, 1) == 0) ? 1 : 0;
        __shared__ int s_claimed;
        if (tid == 0) s_claimed = claimed_success;
        __syncthreads();
        if (!s_claimed) continue;

        // Per-slot acquire: form release-acquire pair with compute's
        // st_na_release(compute_slot_ready). This ensures slot data is visible.
        {
            const int num_topk = state->num_topk;
            for (int k = tid; k < nhits; k += blockDim.x) {
                int slot = state->token_slot_list[token_idx * num_topk + k];
                while (ld_acquire_global(&state->compute_slot_ready[slot]) != 1) {
                    __nanosleep(32);
                }
            }
            __syncthreads();
        }

        if (nhits > 1) {
            constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(__nv_bfloat16);
            constexpr int kBfloat162PerInt4 = kElemsPerInt4 / 2;
            constexpr int kGatherChunkInt4_g = 128;
            constexpr int kVecsPerLane_g = kGatherChunkInt4_g / 32;
            const int hidden = state->combine_hidden;
            const int hidden_int4 = hidden / kElemsPerInt4;
            const int num_topk = state->num_topk;
            const int first_slot = state->token_slot_list[token_idx * num_topk];
            int4* slot_base_i4 = reinterpret_cast<int4*>(state->compute_output_slot);
            const int lane_id_g = tid % 32;

            // Only warp 0 does the reduce (matches original combine pattern exactly)
            if (tid < 32) {
                for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kGatherChunkInt4_g) {
                    const int chunk_end = min(chunk_base + kGatherChunkInt4_g, hidden_int4);
                    const int chunk_int4 = chunk_end - chunk_base;
                    float2 acc[kVecsPerLane_g][kBfloat162PerInt4];
                    #pragma unroll
                    for (int j = 0; j < kVecsPerLane_g; ++j) {
                        #pragma unroll
                        for (int p = 0; p < kBfloat162PerInt4; ++p)
                            acc[j][p] = make_float2(0.0f, 0.0f);
                    }

                    for (int k = 0; k < nhits; ++k) {
                        int slot = state->token_slot_list[token_idx * num_topk + k];
                        #pragma unroll
                        for (int j = 0; j < kVecsPerLane_g; ++j) {
                            const int local_vi = lane_id_g + j * 32;
                            if (local_vi < chunk_int4) {
                                int4 raw;
                                asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                                    : "l"(slot_base_i4 + (int64_t)slot * hidden_int4 + chunk_base + local_vi));
                                const __nv_bfloat162* bv2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                                #pragma unroll
                                for (int p = 0; p < kBfloat162PerInt4; ++p) {
                                    float2 v = __bfloat1622float2(bv2[p]);
                                    acc[j][p].x += v.x;
                                    acc[j][p].y += v.y;
                                }
                            }
                        }
                    }

                    #pragma unroll
                    for (int j = 0; j < kVecsPerLane_g; ++j) {
                        const int vi = chunk_base + lane_id_g + j * 32;
                        if (vi < chunk_end) {
                            int4 packed;
                            __nv_bfloat162* pv2 = reinterpret_cast<__nv_bfloat162*>(&packed);
                            #pragma unroll
                            for (int p = 0; p < kBfloat162PerInt4; ++p)
                                pv2[p] = __float22bfloat162_rn(acc[j][p]);
                            asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};"
                                :: "l"(slot_base_i4 + (int64_t)first_slot * hidden_int4 + vi),
                                   "r"(packed.x), "r"(packed.y), "r"(packed.z), "r"(packed.w));
                        }
                    }
                }
            }
        }

        // All threads sync first, then single fence ensures all writes visible
        __syncthreads();
        __threadfence();
        if (tid == 0) {
            atomicExch(&state->combine_token_ready[token_idx], 1);
        }
        __syncthreads();
    }
}

// ============================================================================
// Main MegaKernel Entry Point
// ============================================================================

template <int kNumRDMARanks, int kStage>
__global__ void __launch_bounds__(MegaKernelRdmaConfig<kNumRDMARanks>::kMegaKernelNumThreads, 1) moe_megakernel_v7(
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
    // Layout: [Dispatch] [Combine] [Scheduler] [Compute groups] [Gather]
    SmRole role;
    int role_idx;

    const int compute_begin = num_dispatch_sms + num_combine_sms + COMPUTE_SCHEDULER_SMS;
    const int gather_begin = compute_begin + num_compute_sms;
    if (sm_id < num_dispatch_sms) {
        role = SmRole::kDispatch;
        role_idx = sm_id;
    } else if (sm_id < num_dispatch_sms + num_combine_sms) {
        role = SmRole::kCombine;
        role_idx = sm_id - num_dispatch_sms;
    } else if (sm_id < compute_begin) {
        role = SmRole::kScheduler;
        role_idx = sm_id - num_dispatch_sms - num_combine_sms;
    } else if (sm_id < gather_begin) {
        role = SmRole::kCompute;
        role_idx = sm_id - compute_begin;
    } else {
        role = SmRole::kGather;
        role_idx = sm_id - gather_begin;
    }

    switch (role) {
        case SmRole::kDispatch:
            dispatch_worker_v2<kNumRDMARanks, kStage>(sm_id, role_idx, state);
            break;

        case SmRole::kCombine:
            combine_worker_v2<kNumRDMARanks, kStage>(role_idx, state);
            break;

        case SmRole::kScheduler:
            compute_scheduler_worker(state, role_idx, COMPUTE_SCHEDULER_SMS);
            break;

        case SmRole::kCompute:
            compute_worker(sm_id, role_idx, num_compute_sms, state, smem_wmma_buf);
            break;

        case SmRole::kGather:
            gather_worker(state, role_idx);
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

template <int kNumRDMARanks, int kStage>
static void launch_megakernel_v7_case(
    MegaKernelState* device_state,
    const MegaKernelState& host_state,
    int total_sms,
    int smem_size,
    cudaStream_t stream
) {
    using RdmaCfg = MegaKernelRdmaConfig<kNumRDMARanks>;
    constexpr int kThreads = RdmaCfg::kMegaKernelNumThreads;
    const int num_ranks = host_state.num_ranks;

    printf("[MK-HOST][LAUNCH] device_state=%p total_sms=%d num_ranks=%d kNumRDMARanks=%d stage=%d block_threads=%d smem_size=%d stream=%p\n",
           device_state, total_sms, num_ranks, kNumRDMARanks, kStage, kThreads, smem_size, stream);
    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(moe_megakernel_v7<kNumRDMARanks, kStage>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
        printf("[MK-HOST][LAUNCH] set dynamic smem attribute=%d\n", smem_size);
    }

    constexpr int num_gather_sms = 2;
    const int launch_total_sms = total_sms + num_gather_sms;
    EP_HOST_ASSERT(MK_COMPUTE_CLUSTER_DIM == 1 || (launch_total_sms % 2 == 0 && "launch_total_sms must be even for MK_COMPUTE_KERNEL=2 cluster_dim=2"));
    EP_HOST_ASSERT(host_state.num_combine_sms % 2 == 0);
    EP_HOST_ASSERT(host_state.num_combine_sms > 0);
    EP_HOST_ASSERT(kThreads >= (RdmaCfg::kNumCombineForwarders + 1) * 32);

#ifndef DISABLE_SM90_FEATURES
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = launch_total_sms;
    cfg.blockDim = kThreads;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = stream;

    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = MK_COMPUTE_CLUSTER_DIM;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_megakernel_v7<kNumRDMARanks, kStage>, device_state));
#else
    moe_megakernel_v7<kNumRDMARanks, kStage><<<launch_total_sms, kThreads, smem_size, stream>>>(device_state);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void launch_megakernel_v7(
    MegaKernelState* device_state,
    int total_sms,
    int smem_size,
    int stage,
    cudaStream_t stream
) {
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    const int num_ranks = host_state.num_ranks;
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);

#define MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, kStage) \
    launch_megakernel_v7_case<kNumRDMARanks, kStage>(device_state, host_state, total_sms, smem_size, stream); \
    break

#define MEGAKERNEL_LAUNCH_CASE(kNumRDMARanks) \
    switch (stage) { \
        case 1: MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, 1); \
        case 2: MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, 2); \
        default: EP_HOST_ASSERT(false && "Unsupported megakernel stage"); \
    } \
    break

    SWITCH_RDMA_RANKS(MEGAKERNEL_LAUNCH_CASE);

#undef MEGAKERNEL_LAUNCH_CASE
#undef MEGAKERNEL_LAUNCH_STAGE_CASE

#ifdef MK_PERF_TRACE
    CUDA_CHECK(cudaStreamSynchronize(stream));
    dump_perf_trace_perfetto(device_state, total_sms);
#endif
}

#ifdef MK_PERF_TRACE
static void dump_perf_trace_perfetto(MegaKernelState* device_state, int total_sms) {
    constexpr bool emit_perf_args = MK_PERF_TRACE >= 2;
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
    std::vector<int64_t> comb_tma_wait(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_single(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_multi(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_flush(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_full(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_flush_count(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_full_count(num_logical_channels * 2);
    std::vector<int64_t> comb_gather_reduce(num_logical_channels * 2);
    std::vector<int64_t> comb_gather_single(num_logical_channels * 2);
    std::vector<int64_t> comb_gather_multi(num_logical_channels * 2);
    std::vector<int64_t> comb_pack_meta(num_logical_channels * 2);
    std::vector<int64_t> comb_pack_meta_work(num_logical_channels * 2);
    std::vector<int64_t> comb_pack_meta_sync(num_logical_channels * 2);
    std::vector<int64_t> comb_tma_store(num_logical_channels * 2);
    std::vector<int64_t> comb_tma_wait_max(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_max(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_single_max(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_multi_max(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_flush_max(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_ready_full_max(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_top(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_top_token(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_top_slot(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_top_expert(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_top_from_flush(num_logical_channels * 2);
    std::vector<int64_t> comb_gather_reduce_max(num_logical_channels * 2);
    std::vector<int64_t> comb_gather_single_max(num_logical_channels * 2);
    std::vector<int64_t> comb_gather_multi_max(num_logical_channels * 2);
    std::vector<int64_t> comb_pack_meta_max(num_logical_channels * 2);
    std::vector<int64_t> comb_pack_meta_work_max(num_logical_channels * 2);
    std::vector<int64_t> comb_pack_meta_sync_max(num_logical_channels * 2);
    std::vector<int64_t> comb_tma_store_max(num_logical_channels * 2);
    std::vector<int64_t> comb_nhit_sum(num_logical_channels * 2);
    std::vector<int64_t> comb_token_count(num_logical_channels * 2);
    std::vector<int64_t> comb_single_token_count(num_logical_channels * 2);
    std::vector<int64_t> comb_multi_token_count(num_logical_channels * 2);
    CUDA_CHECK(cudaMemcpy(disp_wait_nvl.data(), host_state.perf_disp_wait_nvl_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_publish.data(), host_state.perf_disp_publish_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_wait_recvcount.data(), host_state.perf_disp_wait_recvcount_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_tma_wait.data(), host_state.perf_comb_tma_wait_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready.data(), host_state.perf_comb_wait_ready_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_single.data(), host_state.perf_comb_wait_ready_single_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_multi.data(), host_state.perf_comb_wait_ready_multi_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_flush.data(), host_state.perf_comb_wait_ready_flush_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_full.data(), host_state.perf_comb_wait_ready_full_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_flush_count.data(), host_state.perf_comb_wait_ready_flush_count, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_full_count.data(), host_state.perf_comb_wait_ready_full_count, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_gather_reduce.data(), host_state.perf_comb_gather_reduce_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_gather_single.data(), host_state.perf_comb_gather_single_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_gather_multi.data(), host_state.perf_comb_gather_multi_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_pack_meta.data(), host_state.perf_comb_pack_meta_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_pack_meta_work.data(), host_state.perf_comb_pack_meta_work_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_pack_meta_sync.data(), host_state.perf_comb_pack_meta_sync_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_tma_store.data(), host_state.perf_comb_tma_store_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_tma_wait_max.data(), host_state.perf_comb_tma_wait_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_max.data(), host_state.perf_comb_wait_ready_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_single_max.data(), host_state.perf_comb_wait_ready_single_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_multi_max.data(), host_state.perf_comb_wait_ready_multi_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_flush_max.data(), host_state.perf_comb_wait_ready_flush_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_ready_full_max.data(), host_state.perf_comb_wait_ready_full_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_top.data(), host_state.perf_comb_wait_top_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_top_token.data(), host_state.perf_comb_wait_top_token, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_top_slot.data(), host_state.perf_comb_wait_top_slot, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_top_expert.data(), host_state.perf_comb_wait_top_expert, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_top_from_flush.data(), host_state.perf_comb_wait_top_from_flush, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_gather_reduce_max.data(), host_state.perf_comb_gather_reduce_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_gather_single_max.data(), host_state.perf_comb_gather_single_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_gather_multi_max.data(), host_state.perf_comb_gather_multi_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_pack_meta_max.data(), host_state.perf_comb_pack_meta_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_pack_meta_work_max.data(), host_state.perf_comb_pack_meta_work_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_pack_meta_sync_max.data(), host_state.perf_comb_pack_meta_sync_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_tma_store_max.data(), host_state.perf_comb_tma_store_max_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_nhit_sum.data(), host_state.perf_comb_nhit_sum, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_token_count.data(), host_state.perf_comb_token_count, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_single_token_count.data(), host_state.perf_comb_single_token_count, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_multi_token_count.data(), host_state.perf_comb_multi_token_count, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // publish breakdown timers.
    std::vector<int64_t> disp_pub_scan(num_logical_channels * 2);
    std::vector<int64_t> disp_pub_atomic(num_logical_channels * 2);
    std::vector<int64_t> disp_pub_fence(num_logical_channels * 2);
    std::vector<int64_t> disp_pub_store(num_logical_channels * 2);
    std::vector<int64_t> disp_cta_barrier(num_logical_channels * 2);
    std::vector<int64_t> disp_channel_barrier(num_logical_channels * 2);
    std::vector<int64_t> disp_round_barrier(num_logical_channels * 2);
    std::vector<int64_t> disp_tokens(num_logical_channels * 2);
    std::vector<int64_t> disp_local_hit_tokens(num_logical_channels * 2);
    std::vector<int64_t> disp_local_hits(num_logical_channels * 2);
    const int disp_role_n = num_logical_channels * 2 * MK_DISPATCH_ROLE_COUNT * NUM_MAX_NVL_PEERS;
    const int disp_recv_n = num_logical_channels * 2 * NUM_MAX_NVL_PEERS;
    const int num_rdma_ranks = host_state.num_ranks / NUM_MAX_NVL_PEERS;
    std::vector<int64_t> disp_cta_release(num_logical_channels * 2);
    std::vector<int64_t> disp_role_arrive(disp_role_n);
    std::vector<int64_t> disp_role_work(disp_role_n);
    std::vector<int64_t> disp_allrecv_wait_nvl(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_wait(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_wait_start(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_observe(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_done(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_slowest_rdma(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_src_nvl(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_raw_start(disp_recv_n);
    std::vector<int64_t> disp_allrecv_prefix_raw_end(disp_recv_n);
    const int disp_prefix_prod_n = disp_recv_n * num_rdma_ranks;
    std::vector<int64_t> disp_prefix_store_begin(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_publish(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_fence_done(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_store_to_fence(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_meta_wait(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_tokens(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_producer_rank(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_producer_nvl(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_producer_dst_nvl(disp_prefix_prod_n);
    std::vector<int64_t> disp_prefix_producer_src_rdma(disp_prefix_prod_n);
    std::vector<int64_t> disp_allrecv_token_loop(disp_recv_n);
    std::vector<int64_t> disp_allrecv_retire(disp_recv_n);
    std::vector<int64_t> disp_allrecv_publish(disp_recv_n);
    std::vector<int64_t> disp_allrecv_tokens(disp_recv_n);
    std::vector<int64_t> disp_allrecv_local_hits(disp_recv_n);
    CUDA_CHECK(cudaMemcpy(disp_pub_scan.data(), host_state.perf_disp_pub_scan_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_pub_atomic.data(), host_state.perf_disp_pub_atomic_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_pub_fence.data(), host_state.perf_disp_pub_fence_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_pub_store.data(), host_state.perf_disp_pub_store_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_cta_barrier.data(), host_state.perf_disp_cta_barrier_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_channel_barrier.data(), host_state.perf_disp_channel_barrier_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_round_barrier.data(), host_state.perf_disp_round_barrier_ns, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_tokens.data(), host_state.perf_disp_tokens, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_local_hit_tokens.data(), host_state.perf_disp_local_hit_tokens, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_local_hits.data(), host_state.perf_disp_local_hits, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_cta_release.data(), host_state.perf_disp_cta_release_ts, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_role_arrive.data(), host_state.perf_disp_role_arrive_ts, (size_t)disp_role_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_role_work.data(), host_state.perf_disp_role_work_ns, (size_t)disp_role_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_wait_nvl.data(), host_state.perf_disp_allrecv_wait_nvl_ns, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_wait.data(), host_state.perf_disp_allrecv_prefix_wait_ns, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_wait_start.data(), host_state.perf_disp_allrecv_prefix_wait_start_ts, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_observe.data(), host_state.perf_disp_allrecv_prefix_observe_ts, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_done.data(), host_state.perf_disp_allrecv_prefix_done_ts, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_slowest_rdma.data(), host_state.perf_disp_allrecv_prefix_slowest_rdma, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_src_nvl.data(), host_state.perf_disp_allrecv_prefix_src_nvl, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_raw_start.data(), host_state.perf_disp_allrecv_prefix_raw_start, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_prefix_raw_end.data(), host_state.perf_disp_allrecv_prefix_raw_end, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_store_begin.data(), host_state.perf_disp_prefix_store_begin_ts, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_publish.data(), host_state.perf_disp_prefix_publish_ts, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_fence_done.data(), host_state.perf_disp_prefix_fence_done_ts, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_store_to_fence.data(), host_state.perf_disp_prefix_store_to_fence_ns, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_meta_wait.data(), host_state.perf_disp_prefix_meta_wait_ns, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_tokens.data(), host_state.perf_disp_prefix_tokens, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_producer_rank.data(), host_state.perf_disp_prefix_producer_rank, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_producer_nvl.data(), host_state.perf_disp_prefix_producer_nvl, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_producer_dst_nvl.data(), host_state.perf_disp_prefix_producer_dst_nvl, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_prefix_producer_src_rdma.data(), host_state.perf_disp_prefix_producer_src_rdma, (size_t)disp_prefix_prod_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_token_loop.data(), host_state.perf_disp_allrecv_token_loop_ns, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_retire.data(), host_state.perf_disp_allrecv_retire_ns, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_publish.data(), host_state.perf_disp_allrecv_publish_ns, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_tokens.data(), host_state.perf_disp_allrecv_tokens, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp_allrecv_local_hits.data(), host_state.perf_disp_allrecv_local_hits, (size_t)disp_recv_n * sizeof(int64_t), cudaMemcpyDeviceToHost));

    std::vector<int64_t> sched_ts(2);
    int64_t sched_scan_ns = 0, sched_enqueue_ns = 0, sched_idle_ns = 0;
    int64_t sched_priority_ns = 0, sched_normal_ns = 0, sched_tail_flush_ns = 0;
    int64_t sched_publish_total_ns = 0, sched_publish_wait_ns = 0, sched_publish_wait_max_ns = 0;
    int64_t sched_publish_wait_max_tail = -1, sched_publish_wait_max_visible_tail = -1;
    int64_t sched_priority_scan_tokens = 0, sched_priority_ready_tokens = 0;
    int64_t sched_priority_full_batch_hits = 0, sched_priority_batch_already_enqueued = 0;
    int64_t sched_priority_not_full = 0, sched_normal_full_batch_enqueues = 0, sched_flush_tail_enqueues = 0;
    int64_t sched_queue_empty_count = 0, sched_queue_empty_after_dispatch_count = 0;
    int64_t sched_max_ready_tail_gap = 0, sched_stall_expert = -1, sched_stall_recv_count = 0;
    int64_t sched_stall_alloc_count = 0, sched_stall_enqueue_cursor = 0;
    int64_t sched_stall_first_unready_slot = -1, sched_stall_first_unready_ready = 0, sched_stall_dispatch_done = 0;
    CUDA_CHECK(cudaMemcpy(sched_ts.data(), host_state.perf_sched_ts, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_scan_ns, host_state.perf_sched_scan_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_enqueue_ns, host_state.perf_sched_enqueue_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_idle_ns, host_state.perf_sched_idle_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_ns, host_state.perf_sched_priority_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_normal_ns, host_state.perf_sched_normal_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_tail_flush_ns, host_state.perf_sched_tail_flush_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_publish_total_ns, host_state.perf_sched_publish_total_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_publish_wait_ns, host_state.perf_sched_publish_wait_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_publish_wait_max_ns, host_state.perf_sched_publish_wait_max_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_publish_wait_max_tail, host_state.perf_sched_publish_wait_max_tail, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_publish_wait_max_visible_tail, host_state.perf_sched_publish_wait_max_visible_tail, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_scan_tokens, host_state.perf_sched_priority_scan_tokens, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_ready_tokens, host_state.perf_sched_priority_ready_tokens, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_full_batch_hits, host_state.perf_sched_priority_full_batch_hits, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_batch_already_enqueued, host_state.perf_sched_priority_batch_already_enqueued, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_not_full, host_state.perf_sched_priority_not_full, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_normal_full_batch_enqueues, host_state.perf_sched_normal_full_batch_enqueues, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_flush_tail_enqueues, host_state.perf_sched_flush_tail_enqueues, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_queue_empty_count, host_state.perf_sched_queue_empty_count, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_queue_empty_after_dispatch_count, host_state.perf_sched_queue_empty_after_dispatch_count, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_max_ready_tail_gap, host_state.perf_sched_max_ready_tail_gap, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_stall_expert, host_state.perf_sched_stall_expert, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_stall_recv_count, host_state.perf_sched_stall_recv_count, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_stall_alloc_count, host_state.perf_sched_stall_alloc_count, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_stall_enqueue_cursor, host_state.perf_sched_stall_enqueue_cursor, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_stall_first_unready_slot, host_state.perf_sched_stall_first_unready_slot, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_stall_first_unready_ready, host_state.perf_sched_stall_first_unready_ready, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_stall_dispatch_done, host_state.perf_sched_stall_dispatch_done, sizeof(int64_t), cudaMemcpyDeviceToHost));

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
    std::vector<int> diag_task_has_multi(diag_n);
    const int queue_diag_n = host_state.max_compute_tasks > 0 ? host_state.max_compute_tasks : 1;
    std::vector<int64_t> q_publish(queue_diag_n), q_pop_start(queue_diag_n), q_pop_done(queue_diag_n),
        q_bcast_done(queue_diag_n), q_task_start(queue_diag_n), q_prev_gap(queue_diag_n);
    std::vector<int> q_pop_attempts(queue_diag_n), q_cas_failures(queue_diag_n), q_group_id(queue_diag_n);
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
        CUDA_CHECK(cudaMemcpy(diag_task_has_multi.data(), host_state.perf_compute_task_has_multi, (size_t)compute_task_count * sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (host_state.max_compute_tasks > 0) {
        const size_t qb64 = (size_t)host_state.max_compute_tasks * sizeof(int64_t);
        const size_t qb32 = (size_t)host_state.max_compute_tasks * sizeof(int);
        CUDA_CHECK(cudaMemcpy(q_publish.data(), host_state.perf_task_publish_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_pop_start.data(), host_state.perf_task_pop_start_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_pop_done.data(), host_state.perf_task_pop_done_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_bcast_done.data(), host_state.perf_task_bcast_done_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_task_start.data(), host_state.perf_task_start_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_prev_gap.data(), host_state.perf_task_prev_gap_ns, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_pop_attempts.data(), host_state.perf_task_pop_attempts, qb32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_cas_failures.data(), host_state.perf_task_cas_failures, qb32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_group_id.data(), host_state.perf_task_group_id, qb32, cudaMemcpyDeviceToHost));
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
    for (int i = 0; i < 2; ++i) {
        int64_t trace_ts = sched_ts[i];
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
        if (!emit_perf_args) {
            emit_event(name, cat, start, end, pid, tid);
            return;
        }
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"%s_us\":%.3f,\"%s_us\":%.3f,\"%s_us\":%.3f}}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid,
                key_a, wait_a_ns / 1000.0, key_b, wait_b_ns / 1000.0, key_c, wait_c_ns / 1000.0);
    };
    auto emit_event_combine_sender = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid,
                                         int64_t tma_wait_sum_ns, int64_t wait_ready_sum_ns,
                                         int64_t wait_ready_single_sum_ns, int64_t wait_ready_multi_sum_ns,
                                         int64_t wait_ready_flush_sum_ns, int64_t wait_ready_full_sum_ns,
                                         int64_t wait_ready_flush_count, int64_t wait_ready_full_count,
                                         int64_t gather_reduce_sum_ns,
                                         int64_t gather_single_sum_ns, int64_t gather_multi_sum_ns,
                                         int64_t pack_meta_sum_ns, int64_t pack_meta_work_sum_ns,
                                         int64_t pack_meta_sync_sum_ns, int64_t tma_store_sum_ns,
                                         int64_t tma_wait_max_ns, int64_t wait_ready_max_ns,
                                         int64_t wait_ready_single_max_ns, int64_t wait_ready_multi_max_ns,
                                         int64_t wait_ready_flush_max_ns, int64_t wait_ready_full_max_ns,
                                         int64_t wait_top_ns, int64_t wait_top_token, int64_t wait_top_slot,
                                         int64_t wait_top_expert, int64_t wait_top_from_flush,
                                         int64_t gather_reduce_max_ns,
                                         int64_t gather_single_max_ns, int64_t gather_multi_max_ns,
                                         int64_t pack_meta_max_ns, int64_t pack_meta_work_max_ns,
                                         int64_t pack_meta_sync_max_ns, int64_t tma_store_max_ns,
                                         int64_t nhit_sum, int64_t token_count,
                                         int64_t single_token_count, int64_t multi_token_count) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        if (!emit_perf_args) {
            emit_event(name, cat, start, end, pid, tid);
            return;
        }
        double avg_nhits = token_count > 0 ? static_cast<double>(nhit_sum) / static_cast<double>(token_count) : 0.0;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"tma_wait_sum_us\":%.3f,\"wait_slot_ready_sum_us\":%.3f,"
                   "\"wait_slot_ready_single_sum_us\":%.3f,\"wait_slot_ready_multi_sum_us\":%.3f,"
                   "\"wait_slot_ready_flush_sum_us\":%.3f,\"wait_slot_ready_full_sum_us\":%.3f,"
                   "\"wait_slot_ready_flush_count\":%lld,\"wait_slot_ready_full_count\":%lld,"
                   "\"gather_reduce_sum_us\":%.3f,\"gather_single_sum_us\":%.3f,\"gather_multi_sum_us\":%.3f,"
                   "\"pack_meta_sum_us\":%.3f,\"pack_meta_work_sum_us\":%.3f,\"pack_meta_sync_sum_us\":%.3f,"
                   "\"tma_store_issue_sum_us\":%.3f,"
                   "\"tma_wait_max_us\":%.3f,\"wait_slot_ready_max_us\":%.3f,"
                   "\"wait_slot_ready_single_max_us\":%.3f,\"wait_slot_ready_multi_max_us\":%.3f,"
                   "\"wait_slot_ready_flush_max_us\":%.3f,\"wait_slot_ready_full_max_us\":%.3f,"
                   "\"top_wait_us\":%.3f,\"top_wait_token\":%lld,\"top_wait_slot\":%lld,"
                   "\"top_wait_expert\":%lld,\"top_wait_from_flush\":%lld,"
                   "\"gather_reduce_max_us\":%.3f,\"gather_single_max_us\":%.3f,\"gather_multi_max_us\":%.3f,"
                   "\"pack_meta_max_us\":%.3f,\"pack_meta_work_max_us\":%.3f,\"pack_meta_sync_max_us\":%.3f,"
                   "\"tma_store_issue_max_us\":%.3f,"
                   "\"tokens\":%lld,\"single_tokens\":%lld,\"multi_tokens\":%lld,"
                   "\"nhit_sum\":%lld,\"avg_nhits\":%.3f}}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid,
                tma_wait_sum_ns / 1000.0, wait_ready_sum_ns / 1000.0,
                wait_ready_single_sum_ns / 1000.0, wait_ready_multi_sum_ns / 1000.0,
                wait_ready_flush_sum_ns / 1000.0, wait_ready_full_sum_ns / 1000.0,
                static_cast<long long>(wait_ready_flush_count), static_cast<long long>(wait_ready_full_count),
                gather_reduce_sum_ns / 1000.0, gather_single_sum_ns / 1000.0, gather_multi_sum_ns / 1000.0,
                pack_meta_sum_ns / 1000.0, pack_meta_work_sum_ns / 1000.0, pack_meta_sync_sum_ns / 1000.0,
                tma_store_sum_ns / 1000.0,
                tma_wait_max_ns / 1000.0, wait_ready_max_ns / 1000.0,
                wait_ready_single_max_ns / 1000.0, wait_ready_multi_max_ns / 1000.0,
                wait_ready_flush_max_ns / 1000.0, wait_ready_full_max_ns / 1000.0,
                wait_top_ns / 1000.0, static_cast<long long>(wait_top_token), static_cast<long long>(wait_top_slot),
                static_cast<long long>(wait_top_expert), static_cast<long long>(wait_top_from_flush),
                gather_reduce_max_ns / 1000.0, gather_single_max_ns / 1000.0, gather_multi_max_ns / 1000.0,
                pack_meta_max_ns / 1000.0, pack_meta_work_max_ns / 1000.0, pack_meta_sync_max_ns / 1000.0,
                tma_store_max_ns / 1000.0,
                static_cast<long long>(token_count), static_cast<long long>(single_token_count),
                static_cast<long long>(multi_token_count), static_cast<long long>(nhit_sum), avg_nhits);
    };
    auto emit_event_scheduler = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid,
                                    int64_t scan_ns, int64_t enqueue_ns, int64_t idle_ns,
                                    int64_t priority_ns, int64_t normal_ns, int64_t tail_flush_ns,
                                    int64_t publish_total_ns, int64_t publish_wait_ns,
                                    int64_t publish_wait_max_ns, int64_t publish_wait_max_tail,
                                    int64_t publish_wait_max_visible_tail,
                                    int64_t priority_scan_tokens, int64_t priority_ready_tokens,
                                    int64_t priority_full_batch_hits, int64_t priority_batch_already_enqueued,
                                    int64_t priority_not_full, int64_t normal_full_batch_enqueues,
                                    int64_t flush_tail_enqueues, int64_t queue_empty_count,
                                    int64_t queue_empty_after_dispatch_count, int64_t max_ready_tail_gap,
                                    int64_t stall_expert, int64_t stall_recv_count, int64_t stall_alloc_count,
                                    int64_t stall_enqueue_cursor, int64_t stall_first_unready_slot,
                                    int64_t stall_first_unready_ready, int64_t stall_dispatch_done) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        if (!emit_perf_args) {
            emit_event(name, cat, start, end, pid, tid);
            return;
        }
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"scheduler_lanes\":%d,\"scan_ready_us\":%.3f,\"enqueue_task_us\":%.3f,\"idle_sleep_us\":%.3f,"
                   "\"priority_path_us\":%.3f,\"normal_path_us\":%.3f,\"tail_flush_path_us\":%.3f,"
                   "\"publish_total_us\":%.3f,\"publish_wait_tail_us\":%.3f,\"publish_wait_tail_max_us\":%.3f,"
                   "\"publish_wait_tail_max_tail\":%lld,\"publish_wait_tail_max_visible_tail\":%lld,"
                   "\"priority_scan_tokens\":%lld,\"priority_ready_tokens\":%lld,"
                   "\"priority_full_batch_hits\":%lld,\"priority_batch_already_enqueued\":%lld,"
                   "\"priority_not_full\":%lld,\"normal_full_batch_enqueues\":%lld,\"flush_tail_enqueues\":%lld,"
                   "\"queue_empty_count\":%lld,\"queue_empty_after_dispatch_count\":%lld,"
                   "\"max_ready_tail_gap\":%lld,\"stall_expert\":%lld,\"stall_recv_count\":%lld,"
                   "\"stall_alloc_count\":%lld,\"stall_enqueue_cursor\":%lld,"
                   "\"stall_first_unready_slot\":%lld,\"stall_first_unready_ready\":%lld,"
                   "\"stall_dispatch_done\":%lld}}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid,
                COMPUTE_SCHEDULER_SMS, scan_ns / 1000.0, enqueue_ns / 1000.0, idle_ns / 1000.0,
                priority_ns / 1000.0, normal_ns / 1000.0, tail_flush_ns / 1000.0,
                publish_total_ns / 1000.0, publish_wait_ns / 1000.0, publish_wait_max_ns / 1000.0,
                static_cast<long long>(publish_wait_max_tail), static_cast<long long>(publish_wait_max_visible_tail),
                static_cast<long long>(priority_scan_tokens), static_cast<long long>(priority_ready_tokens),
                static_cast<long long>(priority_full_batch_hits), static_cast<long long>(priority_batch_already_enqueued),
                static_cast<long long>(priority_not_full), static_cast<long long>(normal_full_batch_enqueues),
                static_cast<long long>(flush_tail_enqueues), static_cast<long long>(queue_empty_count),
                static_cast<long long>(queue_empty_after_dispatch_count), static_cast<long long>(max_ready_tail_gap),
                static_cast<long long>(stall_expert), static_cast<long long>(stall_recv_count),
                static_cast<long long>(stall_alloc_count), static_cast<long long>(stall_enqueue_cursor),
                static_cast<long long>(stall_first_unready_slot), static_cast<long long>(stall_first_unready_ready),
                static_cast<long long>(stall_dispatch_done));
    };
    // Dispatch sender/forwarder work event with full publish breakdown as args (us).
    auto emit_event_publish = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid,
                                  int64_t wait_nvl_ns, int64_t publish_ns, int64_t wait_recvcount_ns,
                                  int64_t scan_ns, int64_t atomic_ns, int64_t fence_ns, int64_t store_ns,
                                  int64_t cta_barrier_ns, int64_t channel_barrier_ns, int64_t round_barrier_ns,
                                  int64_t tokens, int64_t local_hit_tokens, int64_t local_hits,
                                  int64_t cta_last_role, int64_t cta_last_slot, int64_t cta_last_work_ns,
                                  int64_t cta_last_arrive_to_release_ns,
                                  int64_t last_recv_prefix_wait_ns, int64_t last_recv_prefix_src_nvl,
                                  int64_t last_recv_prefix_src_rdma,
                                  int64_t last_recv_prefix_publish_to_done_ns,
                                  int64_t last_recv_prefix_store_to_fence_ns,
                                  int64_t last_recv_prefix_fence_done_to_done_ns,
                                  int64_t last_recv_prefix_producer_meta_wait_ns,
                                  int64_t last_recv_prefix_producer_tokens,
                                  int64_t last_recv_prefix_producer_rank,
                                  int64_t last_recv_prefix_producer_nvl,
                                  int64_t last_recv_prefix_producer_dst_nvl,
                                  int64_t last_recv_prefix_producer_src_rdma,
                                  int64_t last_recv_prefix_wait_start_to_store_begin_ns,
                                  int64_t last_recv_prefix_store_begin_to_observe_ns,
                                  int64_t last_recv_prefix_publish_to_observe_ns,
                                  int64_t last_recv_prefix_observe_to_done_ns,
                                  int64_t last_recv_prefix_raw_start,
                                  int64_t last_recv_prefix_raw_end,
                                  int64_t last_recv_wait_nvl_ns,
                                  int64_t last_recv_token_loop_ns, int64_t last_recv_publish_ns,
                                  int64_t last_recv_retire_ns, int64_t last_recv_tokens,
                                  int64_t last_recv_local_hits,
                                  int64_t slowest_recv_rank, int64_t slowest_recv_prefix_wait_ns,
                                  int64_t slowest_recv_prefix_src_nvl,
                                  int64_t slowest_recv_prefix_src_rdma,
                                  int64_t slowest_recv_prefix_publish_to_done_ns,
                                  int64_t slowest_recv_prefix_store_to_fence_ns,
                                  int64_t slowest_recv_prefix_fence_done_to_done_ns,
                                  int64_t slowest_recv_prefix_producer_meta_wait_ns,
                                  int64_t slowest_recv_prefix_producer_tokens,
                                  int64_t slowest_recv_prefix_producer_rank,
                                  int64_t slowest_recv_prefix_producer_nvl,
                                  int64_t slowest_recv_prefix_producer_dst_nvl,
                                  int64_t slowest_recv_prefix_producer_src_rdma,
                                  int64_t slowest_recv_prefix_wait_start_to_store_begin_ns,
                                  int64_t slowest_recv_prefix_store_begin_to_observe_ns,
                                  int64_t slowest_recv_prefix_publish_to_observe_ns,
                                  int64_t slowest_recv_prefix_observe_to_done_ns,
                                  int64_t slowest_recv_prefix_raw_start,
                                  int64_t slowest_recv_prefix_raw_end,
                                  int64_t slowest_recv_wait_nvl_ns, int64_t slowest_recv_token_loop_ns,
                                  int64_t slowest_recv_publish_ns, int64_t slowest_recv_retire_ns,
                                  int64_t slowest_recv_tokens, int64_t slowest_recv_local_hits) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        if (!emit_perf_args) {
            emit_event(name, cat, start, end, pid, tid);
            return;
        }
        int64_t attributed_ns = wait_nvl_ns + publish_ns + cta_barrier_ns + channel_barrier_ns + round_barrier_ns;
        int64_t unattributed_ns = dur_ns > attributed_ns ? dur_ns - attributed_ns : 0;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"wait_nvl_us\":%.3f,\"publish_us\":%.3f,\"wait_recvcount_us\":%.3f,"
                   "\"pub_scan_us\":%.3f,\"pub_atomic_us\":%.3f,\"pub_fence_us\":%.3f,\"pub_store_us\":%.3f,"
                   "\"cta_barrier_us\":%.3f,\"channel_barrier_us\":%.3f,\"round_barrier_us\":%.3f,\"unattributed_us\":%.3f,"
                   "\"tokens\":%lld,\"local_hit_tokens\":%lld,\"local_hits\":%lld,"
                   "\"cta_last_role\":%lld,\"cta_last_slot\":%lld,\"cta_last_work_us\":%.3f,"
                   "\"cta_last_arrive_to_release_us\":%.3f,"
                   "\"last_recv_prefix_wait_us\":%.3f,\"last_recv_prefix_src_nvl\":%lld,"
                   "\"last_recv_prefix_src_rdma\":%lld,"
                   "\"last_recv_prefix_publish_to_done_us\":%.3f,\"last_recv_prefix_store_to_fence_us\":%.3f,"
                   "\"last_recv_prefix_fence_done_to_done_us\":%.3f,\"last_recv_prefix_producer_meta_wait_us\":%.3f,"
                   "\"last_recv_prefix_producer_tokens\":%lld,\"last_recv_prefix_producer_rank\":%lld,"
                   "\"last_recv_prefix_producer_nvl\":%lld,\"last_recv_prefix_producer_dst_nvl\":%lld,"
                   "\"last_recv_prefix_producer_src_rdma\":%lld,"
                   "\"last_recv_prefix_wait_start_to_store_begin_us\":%.3f,"
                   "\"last_recv_prefix_store_begin_to_observe_us\":%.3f,"
                   "\"last_recv_prefix_publish_to_observe_us\":%.3f,"
                   "\"last_recv_prefix_observe_to_done_us\":%.3f,"
                   "\"last_recv_prefix_raw_start\":%lld,\"last_recv_prefix_raw_end\":%lld,"
                   "\"last_recv_wait_nvl_us\":%.3f,"
                   "\"last_recv_token_loop_us\":%.3f,\"last_recv_publish_us\":%.3f,"
                   "\"last_recv_retire_us\":%.3f,\"last_recv_tokens\":%lld,"
                   "\"last_recv_local_hits\":%lld,"
                   "\"slowest_recv_rank\":%lld,\"slowest_recv_prefix_wait_us\":%.3f,"
                   "\"slowest_recv_prefix_src_nvl\":%lld,\"slowest_recv_prefix_src_rdma\":%lld,"
                   "\"slowest_recv_prefix_publish_to_done_us\":%.3f,"
                   "\"slowest_recv_prefix_store_to_fence_us\":%.3f,\"slowest_recv_prefix_fence_done_to_done_us\":%.3f,"
                   "\"slowest_recv_prefix_producer_meta_wait_us\":%.3f,\"slowest_recv_prefix_producer_tokens\":%lld,"
                   "\"slowest_recv_prefix_producer_rank\":%lld,\"slowest_recv_prefix_producer_nvl\":%lld,"
                   "\"slowest_recv_prefix_producer_dst_nvl\":%lld,\"slowest_recv_prefix_producer_src_rdma\":%lld,"
                   "\"slowest_recv_prefix_wait_start_to_store_begin_us\":%.3f,"
                   "\"slowest_recv_prefix_store_begin_to_observe_us\":%.3f,"
                   "\"slowest_recv_prefix_publish_to_observe_us\":%.3f,"
                   "\"slowest_recv_prefix_observe_to_done_us\":%.3f,"
                   "\"slowest_recv_prefix_raw_start\":%lld,\"slowest_recv_prefix_raw_end\":%lld,"
                   "\"slowest_recv_wait_nvl_us\":%.3f,\"slowest_recv_token_loop_us\":%.3f,"
                   "\"slowest_recv_publish_us\":%.3f,\"slowest_recv_retire_us\":%.3f,"
                   "\"slowest_recv_tokens\":%lld,\"slowest_recv_local_hits\":%lld}}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid,
                wait_nvl_ns / 1000.0, publish_ns / 1000.0, wait_recvcount_ns / 1000.0,
                scan_ns / 1000.0, atomic_ns / 1000.0, fence_ns / 1000.0, store_ns / 1000.0,
                cta_barrier_ns / 1000.0, channel_barrier_ns / 1000.0, round_barrier_ns / 1000.0, unattributed_ns / 1000.0,
                static_cast<long long>(tokens), static_cast<long long>(local_hit_tokens),
                static_cast<long long>(local_hits), static_cast<long long>(cta_last_role),
                static_cast<long long>(cta_last_slot), cta_last_work_ns / 1000.0,
                cta_last_arrive_to_release_ns / 1000.0,
                last_recv_prefix_wait_ns / 1000.0, static_cast<long long>(last_recv_prefix_src_nvl),
                static_cast<long long>(last_recv_prefix_src_rdma),
                last_recv_prefix_publish_to_done_ns / 1000.0,
                last_recv_prefix_store_to_fence_ns / 1000.0,
                last_recv_prefix_fence_done_to_done_ns / 1000.0,
                last_recv_prefix_producer_meta_wait_ns / 1000.0,
                static_cast<long long>(last_recv_prefix_producer_tokens),
                static_cast<long long>(last_recv_prefix_producer_rank),
                static_cast<long long>(last_recv_prefix_producer_nvl),
                static_cast<long long>(last_recv_prefix_producer_dst_nvl),
                static_cast<long long>(last_recv_prefix_producer_src_rdma),
                last_recv_prefix_wait_start_to_store_begin_ns / 1000.0,
                last_recv_prefix_store_begin_to_observe_ns / 1000.0,
                last_recv_prefix_publish_to_observe_ns / 1000.0,
                last_recv_prefix_observe_to_done_ns / 1000.0,
                static_cast<long long>(last_recv_prefix_raw_start),
                static_cast<long long>(last_recv_prefix_raw_end),
                last_recv_wait_nvl_ns / 1000.0,
                last_recv_token_loop_ns / 1000.0, last_recv_publish_ns / 1000.0,
                last_recv_retire_ns / 1000.0, static_cast<long long>(last_recv_tokens),
                static_cast<long long>(last_recv_local_hits),
                static_cast<long long>(slowest_recv_rank), slowest_recv_prefix_wait_ns / 1000.0,
                static_cast<long long>(slowest_recv_prefix_src_nvl),
                static_cast<long long>(slowest_recv_prefix_src_rdma),
                slowest_recv_prefix_publish_to_done_ns / 1000.0,
                slowest_recv_prefix_store_to_fence_ns / 1000.0,
                slowest_recv_prefix_fence_done_to_done_ns / 1000.0,
                slowest_recv_prefix_producer_meta_wait_ns / 1000.0,
                static_cast<long long>(slowest_recv_prefix_producer_tokens),
                static_cast<long long>(slowest_recv_prefix_producer_rank),
                static_cast<long long>(slowest_recv_prefix_producer_nvl),
                static_cast<long long>(slowest_recv_prefix_producer_dst_nvl),
                static_cast<long long>(slowest_recv_prefix_producer_src_rdma),
                slowest_recv_prefix_wait_start_to_store_begin_ns / 1000.0,
                slowest_recv_prefix_store_begin_to_observe_ns / 1000.0,
                slowest_recv_prefix_publish_to_observe_ns / 1000.0,
                slowest_recv_prefix_observe_to_done_ns / 1000.0,
                static_cast<long long>(slowest_recv_prefix_raw_start),
                static_cast<long long>(slowest_recv_prefix_raw_end),
                slowest_recv_wait_nvl_ns / 1000.0, slowest_recv_token_loop_ns / 1000.0,
                slowest_recv_publish_ns / 1000.0, slowest_recv_retire_ns / 1000.0,
                static_cast<long long>(slowest_recv_tokens), static_cast<long long>(slowest_recv_local_hits));
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

    int scheduler_tid = num_logical_channels * 4 + 50;
    emit_comma();
    fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
               "\"args\":{\"name\":\"scheduler_bridge\"}}",
            host_state.rank, scheduler_tid);
    emit_comma();
    fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
               "\"args\":{\"sort_index\":%d}}",
            host_state.rank, scheduler_tid, scheduler_tid);

    int pid = host_state.rank;
    auto get_dispatch_role_diag = [&](int acc_idx, int64_t& last_role, int64_t& last_slot,
                                      int64_t& last_work_ns, int64_t& last_arrive_to_release_ns,
                                      int64_t& last_recv_prefix_wait_ns, int64_t& last_recv_prefix_src_nvl,
                                      int64_t& last_recv_prefix_src_rdma,
                                      int64_t& last_recv_prefix_publish_to_done_ns,
                                      int64_t& last_recv_prefix_store_to_fence_ns,
                                      int64_t& last_recv_prefix_fence_done_to_done_ns,
                                      int64_t& last_recv_prefix_producer_meta_wait_ns,
                                      int64_t& last_recv_prefix_producer_tokens,
                                      int64_t& last_recv_prefix_producer_rank,
                                      int64_t& last_recv_prefix_producer_nvl,
                                      int64_t& last_recv_prefix_producer_dst_nvl,
                                      int64_t& last_recv_prefix_producer_src_rdma,
                                      int64_t& last_recv_prefix_wait_start_to_store_begin_ns,
                                      int64_t& last_recv_prefix_store_begin_to_observe_ns,
                                      int64_t& last_recv_prefix_publish_to_observe_ns,
                                      int64_t& last_recv_prefix_observe_to_done_ns,
                                      int64_t& last_recv_prefix_raw_start,
                                      int64_t& last_recv_prefix_raw_end,
                                      int64_t& last_recv_wait_ns,
                                      int64_t& last_recv_token_loop_ns, int64_t& last_recv_publish_ns,
                                      int64_t& last_recv_retire_ns, int64_t& last_recv_tokens,
                                      int64_t& last_recv_local_hits,
                                      int64_t& slow_recv_rank, int64_t& slow_recv_prefix_wait_ns,
                                      int64_t& slow_recv_prefix_src_nvl,
                                      int64_t& slow_recv_prefix_src_rdma,
                                      int64_t& slow_recv_prefix_publish_to_done_ns,
                                      int64_t& slow_recv_prefix_store_to_fence_ns,
                                      int64_t& slow_recv_prefix_fence_done_to_done_ns,
                                      int64_t& slow_recv_prefix_producer_meta_wait_ns,
                                      int64_t& slow_recv_prefix_producer_tokens,
                                      int64_t& slow_recv_prefix_producer_rank,
                                      int64_t& slow_recv_prefix_producer_nvl,
                                      int64_t& slow_recv_prefix_producer_dst_nvl,
                                      int64_t& slow_recv_prefix_producer_src_rdma,
                                      int64_t& slow_recv_prefix_wait_start_to_store_begin_ns,
                                      int64_t& slow_recv_prefix_store_begin_to_observe_ns,
                                      int64_t& slow_recv_prefix_publish_to_observe_ns,
                                      int64_t& slow_recv_prefix_observe_to_done_ns,
                                      int64_t& slow_recv_prefix_raw_start,
                                      int64_t& slow_recv_prefix_raw_end,
                                      int64_t& slow_recv_wait_ns, int64_t& slow_recv_token_loop_ns,
                                      int64_t& slow_recv_publish_ns, int64_t& slow_recv_retire_ns,
                                      int64_t& slow_recv_tokens, int64_t& slow_recv_local_hits) {
        last_role = -1;
        last_slot = -1;
        last_work_ns = 0;
        last_arrive_to_release_ns = 0;
        last_recv_prefix_wait_ns = 0;
        last_recv_prefix_src_nvl = -1;
        last_recv_prefix_src_rdma = -1;
        last_recv_prefix_publish_to_done_ns = 0;
        last_recv_prefix_store_to_fence_ns = 0;
        last_recv_prefix_fence_done_to_done_ns = 0;
        last_recv_prefix_producer_meta_wait_ns = 0;
        last_recv_prefix_producer_tokens = 0;
        last_recv_prefix_producer_rank = -1;
        last_recv_prefix_producer_nvl = -1;
        last_recv_prefix_producer_dst_nvl = -1;
        last_recv_prefix_producer_src_rdma = -1;
        last_recv_prefix_wait_start_to_store_begin_ns = 0;
        last_recv_prefix_store_begin_to_observe_ns = 0;
        last_recv_prefix_publish_to_observe_ns = 0;
        last_recv_prefix_observe_to_done_ns = 0;
        last_recv_prefix_raw_start = 0;
        last_recv_prefix_raw_end = 0;
        last_recv_wait_ns = 0;
        last_recv_token_loop_ns = 0;
        last_recv_publish_ns = 0;
        last_recv_retire_ns = 0;
        last_recv_tokens = 0;
        last_recv_local_hits = 0;
        slow_recv_rank = -1;
        slow_recv_prefix_wait_ns = 0;
        slow_recv_prefix_src_nvl = -1;
        slow_recv_prefix_src_rdma = -1;
        slow_recv_prefix_publish_to_done_ns = 0;
        slow_recv_prefix_store_to_fence_ns = 0;
        slow_recv_prefix_fence_done_to_done_ns = 0;
        slow_recv_prefix_producer_meta_wait_ns = 0;
        slow_recv_prefix_producer_tokens = 0;
        slow_recv_prefix_producer_rank = -1;
        slow_recv_prefix_producer_nvl = -1;
        slow_recv_prefix_producer_dst_nvl = -1;
        slow_recv_prefix_producer_src_rdma = -1;
        slow_recv_prefix_wait_start_to_store_begin_ns = 0;
        slow_recv_prefix_store_begin_to_observe_ns = 0;
        slow_recv_prefix_publish_to_observe_ns = 0;
        slow_recv_prefix_observe_to_done_ns = 0;
        slow_recv_prefix_raw_start = 0;
        slow_recv_prefix_raw_end = 0;
        slow_recv_wait_ns = 0;
        slow_recv_token_loop_ns = 0;
        slow_recv_publish_ns = 0;
        slow_recv_retire_ns = 0;
        slow_recv_tokens = 0;
        slow_recv_local_hits = 0;
        int64_t last_arrive = 0;
        for (int r = 0; r < MK_DISPATCH_ROLE_COUNT; ++r) {
            for (int s = 0; s < NUM_MAX_NVL_PEERS; ++s) {
                int idx = ((acc_idx * MK_DISPATCH_ROLE_COUNT + r) * NUM_MAX_NVL_PEERS) + s;
                int64_t arrive = disp_role_arrive[idx];
                if (arrive > last_arrive) {
                    last_arrive = arrive;
                    last_role = r;
                    last_slot = s;
                    last_work_ns = disp_role_work[idx];
                }
            }
        }
        if (disp_cta_release[acc_idx] > last_arrive && last_arrive > 0)
            last_arrive_to_release_ns = disp_cta_release[acc_idx] - last_arrive;
        if (last_role == 4 && last_slot >= 0) {
            int idx = acc_idx * NUM_MAX_NVL_PEERS + last_slot;
            last_recv_prefix_wait_ns = disp_allrecv_prefix_wait[idx];
            last_recv_prefix_src_nvl = disp_allrecv_prefix_src_nvl[idx];
            last_recv_prefix_src_rdma = disp_allrecv_prefix_slowest_rdma[idx];
            if (last_recv_prefix_src_rdma >= 0) {
                int producer_acc_idx = (acc_idx / 2) * 2 + 1;
                int prod_idx = (producer_acc_idx * NUM_MAX_NVL_PEERS + last_slot) * num_rdma_ranks + static_cast<int>(last_recv_prefix_src_rdma);
                int64_t store_begin_ts = disp_prefix_store_begin[prod_idx];
                int64_t publish_ts = disp_prefix_publish[prod_idx];
                int64_t observe_ts = disp_allrecv_prefix_observe[idx];
                if (disp_allrecv_prefix_done[idx] > publish_ts && publish_ts > 0)
                    last_recv_prefix_publish_to_done_ns = disp_allrecv_prefix_done[idx] - publish_ts;
                last_recv_prefix_store_to_fence_ns = disp_prefix_store_to_fence[prod_idx];
                if (disp_allrecv_prefix_done[idx] > disp_prefix_fence_done[prod_idx] && disp_prefix_fence_done[prod_idx] > 0)
                    last_recv_prefix_fence_done_to_done_ns = disp_allrecv_prefix_done[idx] - disp_prefix_fence_done[prod_idx];
                if (store_begin_ts > disp_allrecv_prefix_wait_start[idx] && disp_allrecv_prefix_wait_start[idx] > 0)
                    last_recv_prefix_wait_start_to_store_begin_ns = store_begin_ts - disp_allrecv_prefix_wait_start[idx];
                if (observe_ts > store_begin_ts && store_begin_ts > 0)
                    last_recv_prefix_store_begin_to_observe_ns = observe_ts - store_begin_ts;
                if (observe_ts > publish_ts && publish_ts > 0)
                    last_recv_prefix_publish_to_observe_ns = observe_ts - publish_ts;
                if (disp_allrecv_prefix_done[idx] > observe_ts && observe_ts > 0)
                    last_recv_prefix_observe_to_done_ns = disp_allrecv_prefix_done[idx] - observe_ts;
                last_recv_prefix_raw_start = disp_allrecv_prefix_raw_start[idx];
                last_recv_prefix_raw_end = disp_allrecv_prefix_raw_end[idx];
                last_recv_prefix_producer_meta_wait_ns = disp_prefix_meta_wait[prod_idx];
                last_recv_prefix_producer_tokens = disp_prefix_tokens[prod_idx];
                last_recv_prefix_producer_rank = disp_prefix_producer_rank[prod_idx];
                last_recv_prefix_producer_nvl = disp_prefix_producer_nvl[prod_idx];
                last_recv_prefix_producer_dst_nvl = disp_prefix_producer_dst_nvl[prod_idx];
                last_recv_prefix_producer_src_rdma = disp_prefix_producer_src_rdma[prod_idx];
            }
            last_recv_wait_ns = disp_allrecv_wait_nvl[idx];
            last_recv_token_loop_ns = disp_allrecv_token_loop[idx];
            last_recv_publish_ns = disp_allrecv_publish[idx];
            last_recv_retire_ns = disp_allrecv_retire[idx];
            last_recv_tokens = disp_allrecv_tokens[idx];
            last_recv_local_hits = disp_allrecv_local_hits[idx];
        }
        int64_t slow_recv_total = 0;
        for (int s = 0; s < NUM_MAX_NVL_PEERS; ++s) {
            int idx = acc_idx * NUM_MAX_NVL_PEERS + s;
            int64_t total = disp_allrecv_prefix_wait[idx] + disp_allrecv_wait_nvl[idx] +
                disp_allrecv_token_loop[idx] + disp_allrecv_publish[idx] + disp_allrecv_retire[idx];
            if (total > slow_recv_total) {
                slow_recv_total = total;
                slow_recv_rank = s;
                slow_recv_prefix_wait_ns = disp_allrecv_prefix_wait[idx];
                slow_recv_prefix_src_nvl = disp_allrecv_prefix_src_nvl[idx];
                slow_recv_prefix_src_rdma = disp_allrecv_prefix_slowest_rdma[idx];
                slow_recv_prefix_publish_to_done_ns = 0;
                slow_recv_prefix_store_to_fence_ns = 0;
                slow_recv_prefix_fence_done_to_done_ns = 0;
                slow_recv_prefix_producer_meta_wait_ns = 0;
                slow_recv_prefix_producer_tokens = 0;
                slow_recv_prefix_producer_rank = -1;
                slow_recv_prefix_producer_nvl = -1;
                slow_recv_prefix_producer_dst_nvl = -1;
                slow_recv_prefix_producer_src_rdma = -1;
                slow_recv_prefix_wait_start_to_store_begin_ns = 0;
                slow_recv_prefix_store_begin_to_observe_ns = 0;
                slow_recv_prefix_publish_to_observe_ns = 0;
                slow_recv_prefix_observe_to_done_ns = 0;
                slow_recv_prefix_raw_start = 0;
                slow_recv_prefix_raw_end = 0;
                if (slow_recv_prefix_src_rdma >= 0) {
                    int producer_acc_idx = (acc_idx / 2) * 2 + 1;
                    int prod_idx = (producer_acc_idx * NUM_MAX_NVL_PEERS + s) * num_rdma_ranks + static_cast<int>(slow_recv_prefix_src_rdma);
                    int64_t store_begin_ts = disp_prefix_store_begin[prod_idx];
                    int64_t publish_ts = disp_prefix_publish[prod_idx];
                    int64_t observe_ts = disp_allrecv_prefix_observe[idx];
                    if (disp_allrecv_prefix_done[idx] > publish_ts && publish_ts > 0)
                        slow_recv_prefix_publish_to_done_ns = disp_allrecv_prefix_done[idx] - publish_ts;
                    slow_recv_prefix_store_to_fence_ns = disp_prefix_store_to_fence[prod_idx];
                    if (disp_allrecv_prefix_done[idx] > disp_prefix_fence_done[prod_idx] && disp_prefix_fence_done[prod_idx] > 0)
                        slow_recv_prefix_fence_done_to_done_ns = disp_allrecv_prefix_done[idx] - disp_prefix_fence_done[prod_idx];
                    if (store_begin_ts > disp_allrecv_prefix_wait_start[idx] && disp_allrecv_prefix_wait_start[idx] > 0)
                        slow_recv_prefix_wait_start_to_store_begin_ns = store_begin_ts - disp_allrecv_prefix_wait_start[idx];
                    if (observe_ts > store_begin_ts && store_begin_ts > 0)
                        slow_recv_prefix_store_begin_to_observe_ns = observe_ts - store_begin_ts;
                    if (observe_ts > publish_ts && publish_ts > 0)
                        slow_recv_prefix_publish_to_observe_ns = observe_ts - publish_ts;
                    if (disp_allrecv_prefix_done[idx] > observe_ts && observe_ts > 0)
                        slow_recv_prefix_observe_to_done_ns = disp_allrecv_prefix_done[idx] - observe_ts;
                    slow_recv_prefix_raw_start = disp_allrecv_prefix_raw_start[idx];
                    slow_recv_prefix_raw_end = disp_allrecv_prefix_raw_end[idx];
                    slow_recv_prefix_producer_meta_wait_ns = disp_prefix_meta_wait[prod_idx];
                    slow_recv_prefix_producer_tokens = disp_prefix_tokens[prod_idx];
                    slow_recv_prefix_producer_rank = disp_prefix_producer_rank[prod_idx];
                    slow_recv_prefix_producer_nvl = disp_prefix_producer_nvl[prod_idx];
                    slow_recv_prefix_producer_dst_nvl = disp_prefix_producer_dst_nvl[prod_idx];
                    slow_recv_prefix_producer_src_rdma = disp_prefix_producer_src_rdma[prod_idx];
                }
                slow_recv_wait_ns = disp_allrecv_wait_nvl[idx];
                slow_recv_token_loop_ns = disp_allrecv_token_loop[idx];
                slow_recv_publish_ns = disp_allrecv_publish[idx];
                slow_recv_retire_ns = disp_allrecv_retire[idx];
                slow_recv_tokens = disp_allrecv_tokens[idx];
                slow_recv_local_hits = disp_allrecv_local_hits[idx];
            }
        }
    };

    for (int logical_channel_id = 0; logical_channel_id < num_logical_channels; ++logical_channel_id) {
        int dispatch_sender_tid = lch_tid_base + logical_channel_id * 4;
        int64_t* dsp = &dispatch_lch_ts[(logical_channel_id * 2) * NLP];
        int64_t last_role = -1, last_slot = -1, last_work_ns = 0, last_arrive_to_release_ns = 0;
        int64_t last_recv_prefix_wait_ns = 0, last_recv_prefix_src_nvl = -1, last_recv_prefix_src_rdma = -1;
        int64_t last_recv_prefix_publish_to_done_ns = 0, last_recv_prefix_store_to_fence_ns = 0;
        int64_t last_recv_prefix_fence_done_to_done_ns = 0, last_recv_prefix_producer_meta_wait_ns = 0;
        int64_t last_recv_prefix_producer_tokens = 0, last_recv_prefix_producer_rank = -1;
        int64_t last_recv_prefix_producer_nvl = -1, last_recv_prefix_producer_dst_nvl = -1;
        int64_t last_recv_prefix_producer_src_rdma = -1;
        int64_t last_recv_prefix_wait_start_to_store_begin_ns = 0, last_recv_prefix_store_begin_to_observe_ns = 0;
        int64_t last_recv_prefix_publish_to_observe_ns = 0, last_recv_prefix_observe_to_done_ns = 0;
        int64_t last_recv_prefix_raw_start = 0, last_recv_prefix_raw_end = 0;
        int64_t last_recv_wait_ns = 0, last_recv_token_loop_ns = 0;
        int64_t last_recv_publish_ns = 0, last_recv_retire_ns = 0, last_recv_tokens = 0, last_recv_local_hits = 0;
        int64_t slow_recv_rank = -1, slow_recv_prefix_wait_ns = 0, slow_recv_prefix_src_nvl = -1, slow_recv_prefix_src_rdma = -1;
        int64_t slow_recv_prefix_publish_to_done_ns = 0, slow_recv_prefix_store_to_fence_ns = 0;
        int64_t slow_recv_prefix_fence_done_to_done_ns = 0, slow_recv_prefix_producer_meta_wait_ns = 0;
        int64_t slow_recv_prefix_producer_tokens = 0, slow_recv_prefix_producer_rank = -1;
        int64_t slow_recv_prefix_producer_nvl = -1, slow_recv_prefix_producer_dst_nvl = -1;
        int64_t slow_recv_prefix_producer_src_rdma = -1;
        int64_t slow_recv_prefix_wait_start_to_store_begin_ns = 0, slow_recv_prefix_store_begin_to_observe_ns = 0;
        int64_t slow_recv_prefix_publish_to_observe_ns = 0, slow_recv_prefix_observe_to_done_ns = 0;
        int64_t slow_recv_prefix_raw_start = 0, slow_recv_prefix_raw_end = 0;
        int64_t slow_recv_wait_ns = 0;
        int64_t slow_recv_token_loop_ns = 0, slow_recv_publish_ns = 0, slow_recv_retire_ns = 0;
        int64_t slow_recv_tokens = 0, slow_recv_local_hits = 0;
        get_dispatch_role_diag(logical_channel_id * 2 + 0, last_role, last_slot, last_work_ns,
                               last_arrive_to_release_ns,
                               last_recv_prefix_wait_ns, last_recv_prefix_src_nvl, last_recv_prefix_src_rdma,
                               last_recv_prefix_publish_to_done_ns,
                               last_recv_prefix_store_to_fence_ns,
                               last_recv_prefix_fence_done_to_done_ns,
                               last_recv_prefix_producer_meta_wait_ns,
                               last_recv_prefix_producer_tokens,
                               last_recv_prefix_producer_rank, last_recv_prefix_producer_nvl,
                               last_recv_prefix_producer_dst_nvl, last_recv_prefix_producer_src_rdma,
                               last_recv_prefix_wait_start_to_store_begin_ns,
                               last_recv_prefix_store_begin_to_observe_ns,
                               last_recv_prefix_publish_to_observe_ns,
                               last_recv_prefix_observe_to_done_ns,
                               last_recv_prefix_raw_start, last_recv_prefix_raw_end,
                               last_recv_wait_ns, last_recv_token_loop_ns,
                               last_recv_publish_ns, last_recv_retire_ns, last_recv_tokens,
                               last_recv_local_hits,
                               slow_recv_rank, slow_recv_prefix_wait_ns, slow_recv_prefix_src_nvl, slow_recv_prefix_src_rdma,
                               slow_recv_prefix_publish_to_done_ns,
                               slow_recv_prefix_store_to_fence_ns,
                               slow_recv_prefix_fence_done_to_done_ns,
                               slow_recv_prefix_producer_meta_wait_ns,
                               slow_recv_prefix_producer_tokens,
                               slow_recv_prefix_producer_rank, slow_recv_prefix_producer_nvl,
                               slow_recv_prefix_producer_dst_nvl, slow_recv_prefix_producer_src_rdma,
                               slow_recv_prefix_wait_start_to_store_begin_ns,
                               slow_recv_prefix_store_begin_to_observe_ns,
                               slow_recv_prefix_publish_to_observe_ns,
                               slow_recv_prefix_observe_to_done_ns,
                               slow_recv_prefix_raw_start, slow_recv_prefix_raw_end,
                               slow_recv_wait_ns, slow_recv_token_loop_ns,
                               slow_recv_publish_ns, slow_recv_retire_ns,
                               slow_recv_tokens, slow_recv_local_hits);
        emit_event_publish("dispatch_sender_work", "dispatch_sender_lch", dsp[0], dsp[1], pid, dispatch_sender_tid,
                           disp_wait_nvl[logical_channel_id * 2 + 0], disp_publish[logical_channel_id * 2 + 0],
                           disp_wait_recvcount[logical_channel_id * 2 + 0],
                           disp_pub_scan[logical_channel_id * 2 + 0], disp_pub_atomic[logical_channel_id * 2 + 0],
                           disp_pub_fence[logical_channel_id * 2 + 0], disp_pub_store[logical_channel_id * 2 + 0],
                           disp_cta_barrier[logical_channel_id * 2 + 0],
                           disp_channel_barrier[logical_channel_id * 2 + 0],
                           disp_round_barrier[logical_channel_id * 2 + 0],
                           disp_tokens[logical_channel_id * 2 + 0],
                           disp_local_hit_tokens[logical_channel_id * 2 + 0],
                           disp_local_hits[logical_channel_id * 2 + 0],
                           last_role, last_slot, last_work_ns, last_arrive_to_release_ns,
                           last_recv_prefix_wait_ns, last_recv_prefix_src_nvl, last_recv_prefix_src_rdma,
                           last_recv_prefix_publish_to_done_ns,
                           last_recv_prefix_store_to_fence_ns,
                           last_recv_prefix_fence_done_to_done_ns,
                           last_recv_prefix_producer_meta_wait_ns,
                           last_recv_prefix_producer_tokens,
                           last_recv_prefix_producer_rank, last_recv_prefix_producer_nvl,
                           last_recv_prefix_producer_dst_nvl, last_recv_prefix_producer_src_rdma,
                           last_recv_prefix_wait_start_to_store_begin_ns,
                           last_recv_prefix_store_begin_to_observe_ns,
                           last_recv_prefix_publish_to_observe_ns,
                           last_recv_prefix_observe_to_done_ns,
                           last_recv_prefix_raw_start, last_recv_prefix_raw_end,
                           last_recv_wait_ns, last_recv_token_loop_ns,
                           last_recv_publish_ns, last_recv_retire_ns, last_recv_tokens,
                           last_recv_local_hits,
                           slow_recv_rank, slow_recv_prefix_wait_ns, slow_recv_prefix_src_nvl, slow_recv_prefix_src_rdma,
                           slow_recv_prefix_publish_to_done_ns,
                           slow_recv_prefix_store_to_fence_ns,
                           slow_recv_prefix_fence_done_to_done_ns,
                           slow_recv_prefix_producer_meta_wait_ns,
                           slow_recv_prefix_producer_tokens,
                           slow_recv_prefix_producer_rank, slow_recv_prefix_producer_nvl,
                           slow_recv_prefix_producer_dst_nvl, slow_recv_prefix_producer_src_rdma,
                           slow_recv_prefix_wait_start_to_store_begin_ns,
                           slow_recv_prefix_store_begin_to_observe_ns,
                           slow_recv_prefix_publish_to_observe_ns,
                           slow_recv_prefix_observe_to_done_ns,
                           slow_recv_prefix_raw_start, slow_recv_prefix_raw_end,
                           slow_recv_wait_ns, slow_recv_token_loop_ns,
                           slow_recv_publish_ns, slow_recv_retire_ns,
                           slow_recv_tokens, slow_recv_local_hits);
        emit_event("dispatch_sender_channel_barrier", "dispatch_sender_lch", dsp[1], dsp[2], pid, dispatch_sender_tid);
        emit_event("dispatch_sender_round_barrier", "dispatch_sender_lch", dsp[2], dsp[3], pid, dispatch_sender_tid);

        int dispatch_forwarder_tid = dispatch_sender_tid + 1;
        int64_t* dfp = &dispatch_lch_ts[(logical_channel_id * 2 + 1) * NLP];
        get_dispatch_role_diag(logical_channel_id * 2 + 1, last_role, last_slot, last_work_ns,
                               last_arrive_to_release_ns,
                               last_recv_prefix_wait_ns, last_recv_prefix_src_nvl, last_recv_prefix_src_rdma,
                               last_recv_prefix_publish_to_done_ns,
                               last_recv_prefix_store_to_fence_ns,
                               last_recv_prefix_fence_done_to_done_ns,
                               last_recv_prefix_producer_meta_wait_ns,
                               last_recv_prefix_producer_tokens,
                               last_recv_prefix_producer_rank, last_recv_prefix_producer_nvl,
                               last_recv_prefix_producer_dst_nvl, last_recv_prefix_producer_src_rdma,
                               last_recv_prefix_wait_start_to_store_begin_ns,
                               last_recv_prefix_store_begin_to_observe_ns,
                               last_recv_prefix_publish_to_observe_ns,
                               last_recv_prefix_observe_to_done_ns,
                               last_recv_prefix_raw_start, last_recv_prefix_raw_end,
                               last_recv_wait_ns, last_recv_token_loop_ns,
                               last_recv_publish_ns, last_recv_retire_ns, last_recv_tokens,
                               last_recv_local_hits,
                               slow_recv_rank, slow_recv_prefix_wait_ns, slow_recv_prefix_src_nvl, slow_recv_prefix_src_rdma,
                               slow_recv_prefix_publish_to_done_ns,
                               slow_recv_prefix_store_to_fence_ns,
                               slow_recv_prefix_fence_done_to_done_ns,
                               slow_recv_prefix_producer_meta_wait_ns,
                               slow_recv_prefix_producer_tokens,
                               slow_recv_prefix_producer_rank, slow_recv_prefix_producer_nvl,
                               slow_recv_prefix_producer_dst_nvl, slow_recv_prefix_producer_src_rdma,
                               slow_recv_prefix_wait_start_to_store_begin_ns,
                               slow_recv_prefix_store_begin_to_observe_ns,
                               slow_recv_prefix_publish_to_observe_ns,
                               slow_recv_prefix_observe_to_done_ns,
                               slow_recv_prefix_raw_start, slow_recv_prefix_raw_end,
                               slow_recv_wait_ns, slow_recv_token_loop_ns,
                               slow_recv_publish_ns, slow_recv_retire_ns,
                               slow_recv_tokens, slow_recv_local_hits);
        emit_event_publish("dispatch_forwarder_work", "dispatch_forwarder_lch", dfp[0], dfp[1], pid, dispatch_forwarder_tid,
                           disp_wait_nvl[logical_channel_id * 2 + 1], disp_publish[logical_channel_id * 2 + 1],
                           disp_wait_recvcount[logical_channel_id * 2 + 1],
                           disp_pub_scan[logical_channel_id * 2 + 1], disp_pub_atomic[logical_channel_id * 2 + 1],
                           disp_pub_fence[logical_channel_id * 2 + 1], disp_pub_store[logical_channel_id * 2 + 1],
                           disp_cta_barrier[logical_channel_id * 2 + 1],
                           disp_channel_barrier[logical_channel_id * 2 + 1],
                           disp_round_barrier[logical_channel_id * 2 + 1],
                           disp_tokens[logical_channel_id * 2 + 1],
                           disp_local_hit_tokens[logical_channel_id * 2 + 1],
                           disp_local_hits[logical_channel_id * 2 + 1],
                           last_role, last_slot, last_work_ns, last_arrive_to_release_ns,
                           last_recv_prefix_wait_ns, last_recv_prefix_src_nvl, last_recv_prefix_src_rdma,
                           last_recv_prefix_publish_to_done_ns,
                           last_recv_prefix_store_to_fence_ns,
                           last_recv_prefix_fence_done_to_done_ns,
                           last_recv_prefix_producer_meta_wait_ns,
                           last_recv_prefix_producer_tokens,
                           last_recv_prefix_producer_rank, last_recv_prefix_producer_nvl,
                           last_recv_prefix_producer_dst_nvl, last_recv_prefix_producer_src_rdma,
                           last_recv_prefix_wait_start_to_store_begin_ns,
                           last_recv_prefix_store_begin_to_observe_ns,
                           last_recv_prefix_publish_to_observe_ns,
                           last_recv_prefix_observe_to_done_ns,
                           last_recv_prefix_raw_start, last_recv_prefix_raw_end,
                           last_recv_wait_ns, last_recv_token_loop_ns,
                           last_recv_publish_ns, last_recv_retire_ns, last_recv_tokens,
                           last_recv_local_hits,
                           slow_recv_rank, slow_recv_prefix_wait_ns, slow_recv_prefix_src_nvl, slow_recv_prefix_src_rdma,
                           slow_recv_prefix_publish_to_done_ns,
                           slow_recv_prefix_store_to_fence_ns,
                           slow_recv_prefix_fence_done_to_done_ns,
                           slow_recv_prefix_producer_meta_wait_ns,
                           slow_recv_prefix_producer_tokens,
                           slow_recv_prefix_producer_rank, slow_recv_prefix_producer_nvl,
                           slow_recv_prefix_producer_dst_nvl, slow_recv_prefix_producer_src_rdma,
                           slow_recv_prefix_wait_start_to_store_begin_ns,
                           slow_recv_prefix_store_begin_to_observe_ns,
                           slow_recv_prefix_publish_to_observe_ns,
                           slow_recv_prefix_observe_to_done_ns,
                           slow_recv_prefix_raw_start, slow_recv_prefix_raw_end,
                           slow_recv_wait_ns, slow_recv_token_loop_ns,
                           slow_recv_publish_ns, slow_recv_retire_ns,
                           slow_recv_tokens, slow_recv_local_hits);
        emit_event("dispatch_forwarder_channel_barrier", "dispatch_forwarder_lch", dfp[1], dfp[2], pid, dispatch_forwarder_tid);
        emit_event("dispatch_forwarder_round_barrier", "dispatch_forwarder_lch", dfp[2], dfp[3], pid, dispatch_forwarder_tid);

        int combine_sender_tid = dispatch_sender_tid + 2;
        int64_t* csp = &combine_lch_ts[(logical_channel_id * 2) * NLP];
        emit_event("sender_wait_dispatch_done", "combine_sender_lch", csp[0], csp[1], pid, combine_sender_tid);
        emit_event("sender_head_normalize", "combine_sender_lch", csp[1], csp[2], pid, combine_sender_tid);
        emit_event_combine_sender("sender_nvl_send_rdma_recv", "combine_sender_lch", csp[3], csp[4], pid, combine_sender_tid,
                                  comb_tma_wait[logical_channel_id * 2 + 0],
                                  comb_wait_ready[logical_channel_id * 2 + 0],
                                  comb_wait_ready_single[logical_channel_id * 2 + 0],
                                  comb_wait_ready_multi[logical_channel_id * 2 + 0],
                                  comb_wait_ready_flush[logical_channel_id * 2 + 0],
                                  comb_wait_ready_full[logical_channel_id * 2 + 0],
                                  comb_wait_ready_flush_count[logical_channel_id * 2 + 0],
                                  comb_wait_ready_full_count[logical_channel_id * 2 + 0],
                                  comb_gather_reduce[logical_channel_id * 2 + 0],
                                  comb_gather_single[logical_channel_id * 2 + 0],
                                  comb_gather_multi[logical_channel_id * 2 + 0],
                                  comb_pack_meta[logical_channel_id * 2 + 0],
                                  comb_pack_meta_work[logical_channel_id * 2 + 0],
                                  comb_pack_meta_sync[logical_channel_id * 2 + 0],
                                  comb_tma_store[logical_channel_id * 2 + 0],
                                  comb_tma_wait_max[logical_channel_id * 2 + 0],
                                  comb_wait_ready_max[logical_channel_id * 2 + 0],
                                  comb_wait_ready_single_max[logical_channel_id * 2 + 0],
                                  comb_wait_ready_multi_max[logical_channel_id * 2 + 0],
                                  comb_wait_ready_flush_max[logical_channel_id * 2 + 0],
                                  comb_wait_ready_full_max[logical_channel_id * 2 + 0],
                                  comb_wait_top[logical_channel_id * 2 + 0],
                                  comb_wait_top_token[logical_channel_id * 2 + 0],
                                  comb_wait_top_slot[logical_channel_id * 2 + 0],
                                  comb_wait_top_expert[logical_channel_id * 2 + 0],
                                  comb_wait_top_from_flush[logical_channel_id * 2 + 0],
                                  comb_gather_reduce_max[logical_channel_id * 2 + 0],
                                  comb_gather_single_max[logical_channel_id * 2 + 0],
                                  comb_gather_multi_max[logical_channel_id * 2 + 0],
                                  comb_pack_meta_max[logical_channel_id * 2 + 0],
                                  comb_pack_meta_work_max[logical_channel_id * 2 + 0],
                                  comb_pack_meta_sync_max[logical_channel_id * 2 + 0],
                                  comb_tma_store_max[logical_channel_id * 2 + 0],
                                  comb_nhit_sum[logical_channel_id * 2 + 0],
                                  comb_token_count[logical_channel_id * 2 + 0],
                                  comb_single_token_count[logical_channel_id * 2 + 0],
                                  comb_multi_token_count[logical_channel_id * 2 + 0]);

        int combine_forwarder_tid = dispatch_sender_tid + 3;
        int64_t* cfp = &combine_lch_ts[(logical_channel_id * 2 + 1) * NLP];
        emit_event("forwarder_wait_normalized", "combine_forwarder_lch", cfp[0], cfp[2], pid, combine_forwarder_tid);
        emit_event("forwarder_nvl_to_rdma", "combine_forwarder_lch", cfp[3], cfp[4], pid, combine_forwarder_tid);
    }

    emit_event_scheduler("scheduler_bridge", "scheduler", sched_ts[0], sched_ts[1], pid, scheduler_tid,
                         sched_scan_ns, sched_enqueue_ns, sched_idle_ns,
                         sched_priority_ns, sched_normal_ns, sched_tail_flush_ns,
                         sched_publish_total_ns, sched_publish_wait_ns, sched_publish_wait_max_ns,
                         sched_publish_wait_max_tail, sched_publish_wait_max_visible_tail,
                         sched_priority_scan_tokens, sched_priority_ready_tokens,
                         sched_priority_full_batch_hits, sched_priority_batch_already_enqueued,
                         sched_priority_not_full, sched_normal_full_batch_enqueues,
                         sched_flush_tail_enqueues, sched_queue_empty_count,
                         sched_queue_empty_after_dispatch_count, sched_max_ready_tail_gap,
                         sched_stall_expert, sched_stall_recv_count, sched_stall_alloc_count,
                         sched_stall_enqueue_cursor, sched_stall_first_unready_slot,
                         sched_stall_first_unready_ready, sched_stall_dispatch_done);

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
        int queue_task_idx = static_cast<int>(rec[21]);
        int start_slot = static_cast<int>(rec[22]);
        int end_slot = static_cast<int>(rec[23]);
        int64_t abs_slot_base = rec[24];
        int is_flush_task = static_cast<int>(rec[25]);
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
        int task_has_multi = diag_task_has_multi[t];
        bool have_queue_diag = queue_task_idx >= 0 && queue_task_idx < queue_diag_n;
        int64_t task_publish = have_queue_diag ? q_publish[queue_task_idx] : 0;
        int64_t task_pop_start = have_queue_diag ? q_pop_start[queue_task_idx] : 0;
        int64_t task_pop_done = have_queue_diag ? q_pop_done[queue_task_idx] : 0;
        int64_t task_bcast_done = have_queue_diag ? q_bcast_done[queue_task_idx] : 0;
        int64_t task_start_diag = have_queue_diag ? q_task_start[queue_task_idx] : 0;
        int64_t task_prev_gap = have_queue_diag ? q_prev_gap[queue_task_idx] : 0;
        int task_pop_attempts = have_queue_diag ? q_pop_attempts[queue_task_idx] : 0;
        int task_cas_failures = have_queue_diag ? q_cas_failures[queue_task_idx] : 0;
        int task_group_diag = have_queue_diag ? q_group_id[queue_task_idx] : -1;
        double publish_to_pop_us = phase_us(task_publish, task_pop_done);
        double pop_wait_us = phase_us(task_pop_start, task_pop_done);
        double pop_to_bcast_us = phase_us(task_pop_done, task_bcast_done);
        double bcast_to_start_us = phase_us(task_bcast_done, task_start_diag);
        double prev_task_gap_us = ns_us(task_prev_gap);
        if (!emit_perf_args) {
            emit_event("compute_task", "compute_group", start, end, pid, tid);
            continue;
        }
        emit_comma();
        fprintf(f, "{\"name\":\"compute_e%d\",\"cat\":\"compute_group\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,"
                   "\"pid\":%d,\"tid\":%d,\"args\":{\"expert_id\":%d,\"sm_id\":%d,\"group_id\":%d,\"batch_size\":%d,"
                   "\"start_slot\":%d,\"end_slot\":%d,\"abs_slot_base\":%lld,\"is_flush_task\":%d,"
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
                   "\"queue_task_idx\":%d,\"queue_group_id\":%d,\"prev_task_gap_us\":%.3f,"
                   "\"publish_to_pop_us\":%.3f,\"pop_wait_us\":%.3f,\"pop_to_bcast_us\":%.3f,"
                   "\"bcast_to_start_us\":%.3f,\"pop_attempts\":%d,\"cas_failures\":%d,"
                   "\"p6b_multi_expert_rows\":%d,\"p6_task_has_multi\":%d}}",
                expert_id, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid, expert_id, sm_id, group_id, batch_size,
                start_slot, end_slot, static_cast<long long>(abs_slot_base), is_flush_task,
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
                queue_task_idx, task_group_diag, prev_task_gap_us,
                publish_to_pop_us, pop_wait_us, pop_to_bcast_us,
                bcast_to_start_us, task_pop_attempts, task_cas_failures,
                multi_expert_rows, task_has_multi);
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
    int dispatch_num_max_rdma_chunked_send_tokens,
    int dispatch_num_max_rdma_chunked_recv_tokens,
    int dispatch_num_max_nvl_chunked_send_tokens,
    int dispatch_num_max_nvl_chunked_recv_tokens,
    int combine_num_max_rdma_chunked_send_tokens,
    int combine_num_max_rdma_chunked_recv_tokens,
    int combine_num_max_nvl_chunked_send_tokens,
    int combine_num_max_nvl_chunked_recv_tokens,
    // --- Expert weights ---
    const __nv_bfloat16* W_gateup,
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
    int* compute_task_reserve_tail;
    int* compute_enqueue_done;
    int* scheduler_done_count;
    int* expert_enqueue_cursor;
    int* compute_group_task_idx;
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
    EP_HOST_ASSERT(hidden_dim * static_cast<int>(sizeof(__nv_bfloat16)) % static_cast<int>(sizeof(int4)) == 0);
    EP_HOST_ASSERT(num_topk <= 32);
    EP_HOST_ASSERT(num_experts > 0 and num_local_experts > 0);
    EP_HOST_ASSERT(num_ranks > 0 and num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_logical_channels * 2 > 3);
    int kNumRDMARanks = num_ranks / NUM_MAX_NVL_PEERS;

    EP_HOST_ASSERT(dispatch_num_max_rdma_chunked_send_tokens > 0 and dispatch_num_max_rdma_chunked_recv_tokens > 0);
    EP_HOST_ASSERT(dispatch_num_max_nvl_chunked_send_tokens > 0 and dispatch_num_max_nvl_chunked_recv_tokens > 0);
    EP_HOST_ASSERT(dispatch_num_max_rdma_chunked_recv_tokens % dispatch_num_max_rdma_chunked_send_tokens == 0);
    EP_HOST_ASSERT(dispatch_num_max_nvl_chunked_send_tokens < dispatch_num_max_nvl_chunked_recv_tokens);

    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / kNumRDMARanks, 1);
    int num_forwarder_warps = kNumRDMARanks * num_warps_per_forwarder;
    EP_HOST_ASSERT(kNumRDMARanks <= kNumCombineForwarderWarps);
    EP_HOST_ASSERT(num_forwarder_warps > NUM_MAX_NVL_PEERS and num_forwarder_warps % kNumRDMARanks == 0);
    EP_HOST_ASSERT(combine_num_max_nvl_chunked_recv_tokens % kNumRDMARanks == 0);
    EP_HOST_ASSERT(combine_num_max_nvl_chunked_recv_tokens / kNumRDMARanks >
                   std::max(combine_num_max_rdma_chunked_send_tokens, combine_num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(combine_num_max_nvl_chunked_recv_tokens / kNumRDMARanks - num_warps_per_forwarder >= combine_num_max_nvl_chunked_send_tokens);
    EP_HOST_ASSERT(combine_num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);

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
    CUDA_CHECK(cudaMalloc(&compute_task_reserve_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_task_reserve_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_enqueue_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_enqueue_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scheduler_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(scheduler_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_enqueue_cursor, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_enqueue_cursor, 0, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_group_task_idx, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_task_idx, 0xff, num_compute_groups * sizeof(int)));

    // Dedicated gather SM semaphore state (no compute/copy in gather for this experiment).
    int* token_done_count;
    int* gather_claimed;
    int* combine_token_ready;
    CUDA_CHECK(cudaMalloc(&token_done_count, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_done_count, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_claimed, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_claimed, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_token_ready, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_token_ready, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    int* combine_done_count;
    int* combine_all_done;
    CUDA_CHECK(cudaMalloc(&combine_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_all_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_all_done, 0, sizeof(int)));
    // Combine per-expert completion signals
    int* expert_compute_done;
    CUDA_CHECK(cudaMalloc(&expert_compute_done, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_compute_done, 0, num_local_experts * sizeof(int)));

    // Publish-offload backing state (Stage 1: allocate + zero only, no consumer yet).
    // num_pub_warps_total = one publisher per NVL receiver warp per receiver SM.
    const int num_pub_warps_total = (num_dispatch_sms / 2) * NUM_MAX_NVL_PEERS;
    int* token_publish_done;
    int* pending_topk_idx;
    float* pending_topk_weights;
    internode::SourceMeta* pending_meta;
    int* pub_ring;
    int* pub_ring_head;
    int* pub_ring_tail;
    int* recv_warp_done;
    int* publish_done_count;
    int* publish_all_done;
    CUDA_CHECK(cudaMalloc(&token_publish_done, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_publish_done, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pending_topk_idx, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMemset(pending_topk_idx, 0xff, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pending_topk_weights, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(pending_topk_weights, 0, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pending_meta, (size_t)max_total_recv_tokens * sizeof(internode::SourceMeta)));
    CUDA_CHECK(cudaMalloc(&pub_ring, (size_t)num_pub_warps_total * PUB_RING_DEPTH * sizeof(int)));
    CUDA_CHECK(cudaMemset(pub_ring, 0, (size_t)num_pub_warps_total * PUB_RING_DEPTH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pub_ring_head, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(pub_ring_head, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pub_ring_tail, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(pub_ring_tail, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_warp_done, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_warp_done, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&publish_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&publish_all_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_all_done, 0, sizeof(int)));

    // Combine input namespace from dispatch receive.
    CUDA_CHECK(cudaMalloc(&combine_input, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(combine_input, 0, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&combine_input_topk_weights, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(combine_input_topk_weights, 0, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&combine_input_src_meta, (size_t)max_total_recv_tokens * sizeof(internode::SourceMeta)));
    CUDA_CHECK(cudaMalloc(&combine_notify_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_notify_done, 0, sizeof(int)));

    // Per-token compute signaling
    int* token_compute_expected;
    CUDA_CHECK(cudaMalloc(&token_compute_expected, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_compute_expected, 0, (size_t)max_total_recv_tokens * sizeof(int)));

    // Per-slot output path scratch + reverse map + per-slot ready (MEGAKERNEL_COMPUTE_DESIGN III).
    __nv_bfloat16* compute_output_slot;
    int* compute_slot_ready;
    int* compute_slot_from_flush;
    int64_t* compute_slot_ready_ts;
    int* token_nhits;
    int* token_slot_list;
    int* priority_token_cursor;
    int* expert_batch_enqueued;
    CUDA_CHECK(cudaMalloc(&compute_output_slot, recv_tokens_bytes));
    CUDA_CHECK(cudaMemset(compute_output_slot, 0, recv_tokens_bytes));
    CUDA_CHECK(cudaMalloc(&compute_slot_ready, total_expert_slots * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_slot_ready, 0, total_expert_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_slot_from_flush, total_expert_slots * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_slot_from_flush, 0, total_expert_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_slot_ready_ts, total_expert_slots * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(compute_slot_ready_ts, 0, total_expert_slots * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&token_nhits, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_nhits, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&token_slot_list, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_slot_list, 0xff, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&priority_token_cursor, sizeof(int)));
    CUDA_CHECK(cudaMemset(priority_token_cursor, 0, sizeof(int)));
    const int max_batches_per_expert = (max_tokens_per_expert + COMPUTE_BATCH_SIZE - 1) / COMPUTE_BATCH_SIZE;
    CUDA_CHECK(cudaMalloc(&expert_batch_enqueued, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_batch_enqueued, 0, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));

    // GEMM workspace: per-compute-group batched intermediates for M=128 compute batches.
    // Layout: [input(M*hidden)][GU scratch(M*2I)][act(M*I)][down(M*hidden)].
    // The interleaved gate/up path writes act directly from the GEMM epilogue; the
    // GU scratch region remains reserved so existing descriptors/helpers stay valid.
    size_t per_group_elems = (size_t)COMPUTE_BATCH_SIZE * (2 * hidden_dim + 3 * intermediate_dim);
    size_t workspace_bytes = num_compute_groups * per_group_elems * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&gemm_workspace, workspace_bytes));

    // --- S4.4 (route B2): build UMMA compute TMA atoms on host, upload to device ---
    // Gate/up uses A[M, hidden] x interleaved Wgu[2 * intermediate, hidden]^T,
    // with the SwiGLU epilogue storing act[M, intermediate] directly.
    // Down uses act[M, intermediate] x W_down[hidden, intermediate]^T.
    umma::ComputeTmaAtoms* d_compute_tma = nullptr;
    umma::InputTmaAtom_t* d_group_input_tma = nullptr;
    umma::ComputeDownTmaAtoms* d_compute_down_tma = nullptr;
    if (W_gateup != nullptr && W_down != nullptr &&
        num_local_experts <= umma::kMaxLocalExperts) {
        // Per-expert weight atoms. W_gateup is already interleaved by the Python test
        // as [E,2I,d] with rows [2j]=Wg[j], [2j+1]=Wu[j].
        umma::ComputeTmaAtoms h_atoms;
        umma::build_compute_tma_atoms(h_atoms, W_gateup, num_local_experts,
                                      intermediate_dim, hidden_dim);
        CUDA_CHECK(cudaMalloc(&d_compute_tma, sizeof(umma::ComputeTmaAtoms)));
        CUDA_CHECK(cudaMemcpy(d_compute_tma, &h_atoms, sizeof(umma::ComputeTmaAtoms), cudaMemcpyHostToDevice));

        // Per-group A(input_buf) + GU/act/down workspace atoms. Layout per group:
        //   [input_buf (M*hidden)] [GU scratch (M*2I)] [act (M*I)] [down_buf (M*hidden)]
        std::vector<umma::InputTmaAtom_t> h_in;
        h_in.reserve(num_compute_groups);
        for (int g = 0; g < num_compute_groups; ++g) {
            const __nv_bfloat16* in_g = gemm_workspace + (size_t)g * per_group_elems;
            const __nv_bfloat16* gu_g   = in_g + (size_t)COMPUTE_BATCH_SIZE * hidden_dim;
            const __nv_bfloat16* act_g  = gu_g + (size_t)COMPUTE_BATCH_SIZE * (2 * intermediate_dim);
            const __nv_bfloat16* down_g = act_g + (size_t)COMPUTE_BATCH_SIZE * intermediate_dim;
            // gate_buf_ptr keeps the reserved GU scratch descriptor; act_buf_ptr is the interleaved epilogue output and down A.
            h_in.push_back(umma::make_input_group_atoms(in_g, gu_g, act_g, down_g,
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
    const int combine_rdma_head_stride = num_tokens * kNumRDMARanks;
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
           num_tokens, kNumRDMARanks, num_physical_channels, num_logical_channels);

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

    CUDA_CHECK(cudaMalloc(&recv_rdma_channel_prefix_matrix, kNumRDMARanks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_rdma_channel_prefix_matrix, 0, kNumRDMARanks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_gbl_channel_prefix_matrix, num_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_gbl_channel_prefix_matrix, 0, num_ranks * num_logical_channels * sizeof(int)));

    // Per-logical-channel token counts (non-cumulative) for overlap
    int* recv_rdma_channel_token_count;
    int* recv_gbl_channel_token_count;
    CUDA_CHECK(cudaMalloc(&recv_rdma_channel_token_count, kNumRDMARanks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_rdma_channel_token_count, 0, kNumRDMARanks * num_logical_channels * sizeof(int)));
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
    host_state.num_max_rdma_chunked_send_tokens = dispatch_num_max_rdma_chunked_send_tokens;
    host_state.num_max_rdma_chunked_recv_tokens = dispatch_num_max_rdma_chunked_recv_tokens;
    host_state.num_max_nvl_chunked_send_tokens = dispatch_num_max_nvl_chunked_send_tokens;
    host_state.num_max_nvl_chunked_recv_tokens = dispatch_num_max_nvl_chunked_recv_tokens;

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
    host_state.compute_task_reserve_tail = compute_task_reserve_tail;
    host_state.compute_enqueue_done = compute_enqueue_done;
    host_state.scheduler_done_count = scheduler_done_count;
    host_state.expert_enqueue_cursor = expert_enqueue_cursor;
    host_state.compute_group_task_idx = compute_group_task_idx;

    // Dedicated gather SM semaphore state
    host_state.token_done_count = token_done_count;
    host_state.gather_claimed = gather_claimed;
    host_state.combine_token_ready = combine_token_ready;
    host_state.combine_done_count = combine_done_count;
    host_state.combine_all_done = combine_all_done;
    host_state.num_gather_sms = 2;

    // Publish-offload backing state (Stage 1)
    host_state.token_publish_done = token_publish_done;
    host_state.pending_topk_idx = pending_topk_idx;
    host_state.pending_topk_weights = pending_topk_weights;
    host_state.pending_meta = pending_meta;
    host_state.pub_ring = pub_ring;
    host_state.pub_ring_head = pub_ring_head;
    host_state.pub_ring_tail = pub_ring_tail;
    host_state.recv_warp_done = recv_warp_done;
    host_state.publish_done_count = publish_done_count;
    host_state.publish_all_done = publish_all_done;
    host_state.num_pub_warps_total = num_pub_warps_total;


    // Expert weights
    host_state.W_gateup = W_gateup;
    host_state.W_down = W_down;

    // Compute output
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
    host_state.compute_output_slot = compute_output_slot;
    host_state.compute_slot_ready = compute_slot_ready;
    host_state.compute_slot_from_flush = compute_slot_from_flush;
    host_state.compute_slot_ready_ts = compute_slot_ready_ts;
    host_state.token_nhits = token_nhits;
    host_state.token_slot_list = token_slot_list;
    host_state.priority_token_cursor = priority_token_cursor;
    host_state.expert_batch_enqueued = expert_batch_enqueued;
    host_state.max_batches_per_expert = max_batches_per_expert;

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
    int64_t* perf_comb_tma_wait_ns;
    int64_t* perf_comb_wait_ready_ns;
    int64_t* perf_comb_wait_ready_single_ns;
    int64_t* perf_comb_wait_ready_multi_ns;
    int64_t* perf_comb_wait_ready_flush_ns;
    int64_t* perf_comb_wait_ready_full_ns;
    int64_t* perf_comb_wait_ready_flush_count;
    int64_t* perf_comb_wait_ready_full_count;
    int64_t* perf_comb_gather_reduce_ns;
    int64_t* perf_comb_gather_single_ns;
    int64_t* perf_comb_gather_multi_ns;
    int64_t* perf_comb_pack_meta_ns;
    int64_t* perf_comb_pack_meta_work_ns;
    int64_t* perf_comb_pack_meta_sync_ns;
    int64_t* perf_comb_tma_store_ns;
    int64_t* perf_comb_tma_wait_max_ns;
    int64_t* perf_comb_wait_ready_max_ns;
    int64_t* perf_comb_wait_ready_single_max_ns;
    int64_t* perf_comb_wait_ready_multi_max_ns;
    int64_t* perf_comb_wait_ready_flush_max_ns;
    int64_t* perf_comb_wait_ready_full_max_ns;
    int64_t* perf_comb_wait_top_ns;
    int64_t* perf_comb_wait_top_token;
    int64_t* perf_comb_wait_top_slot;
    int64_t* perf_comb_wait_top_expert;
    int64_t* perf_comb_wait_top_from_flush;
    int64_t* perf_comb_gather_reduce_max_ns;
    int64_t* perf_comb_gather_single_max_ns;
    int64_t* perf_comb_gather_multi_max_ns;
    int64_t* perf_comb_pack_meta_max_ns;
    int64_t* perf_comb_pack_meta_work_max_ns;
    int64_t* perf_comb_pack_meta_sync_max_ns;
    int64_t* perf_comb_tma_store_max_ns;
    int64_t* perf_comb_nhit_sum;
    int64_t* perf_comb_token_count;
    int64_t* perf_comb_single_token_count;
    int64_t* perf_comb_multi_token_count;
    const size_t acc_bytes = (size_t)num_logical_channels * 2 * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&perf_disp_wait_nvl_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_wait_nvl_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_disp_publish_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_publish_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_disp_wait_recvcount_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_disp_wait_recvcount_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_tma_wait_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_tma_wait_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_single_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_single_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_multi_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_multi_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_flush_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_flush_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_full_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_full_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_flush_count, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_flush_count, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_full_count, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_full_count, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_gather_reduce_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_gather_reduce_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_gather_single_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_gather_single_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_gather_multi_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_gather_multi_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_pack_meta_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_pack_meta_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_pack_meta_work_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_pack_meta_work_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_pack_meta_sync_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_pack_meta_sync_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_tma_store_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_tma_store_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_tma_wait_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_tma_wait_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_single_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_single_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_multi_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_multi_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_flush_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_flush_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_ready_full_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_ready_full_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_top_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_top_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_top_token, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_top_token, 0xff, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_top_slot, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_top_slot, 0xff, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_top_expert, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_top_expert, 0xff, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_top_from_flush, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_top_from_flush, 0xff, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_gather_reduce_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_gather_reduce_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_gather_single_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_gather_single_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_gather_multi_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_gather_multi_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_pack_meta_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_pack_meta_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_pack_meta_work_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_pack_meta_work_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_pack_meta_sync_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_pack_meta_sync_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_tma_store_max_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_tma_store_max_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_nhit_sum, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_nhit_sum, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_token_count, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_token_count, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_single_token_count, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_single_token_count, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_multi_token_count, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_multi_token_count, 0, acc_bytes));
    host_state.perf_disp_wait_nvl_ns = perf_disp_wait_nvl_ns;
    host_state.perf_disp_publish_ns = perf_disp_publish_ns;
    host_state.perf_disp_wait_recvcount_ns = perf_disp_wait_recvcount_ns;
    host_state.perf_comb_tma_wait_ns = perf_comb_tma_wait_ns;
    host_state.perf_comb_wait_ready_ns = perf_comb_wait_ready_ns;
    host_state.perf_comb_wait_ready_single_ns = perf_comb_wait_ready_single_ns;
    host_state.perf_comb_wait_ready_multi_ns = perf_comb_wait_ready_multi_ns;
    host_state.perf_comb_wait_ready_flush_ns = perf_comb_wait_ready_flush_ns;
    host_state.perf_comb_wait_ready_full_ns = perf_comb_wait_ready_full_ns;
    host_state.perf_comb_wait_ready_flush_count = perf_comb_wait_ready_flush_count;
    host_state.perf_comb_wait_ready_full_count = perf_comb_wait_ready_full_count;
    host_state.perf_comb_gather_reduce_ns = perf_comb_gather_reduce_ns;
    host_state.perf_comb_gather_single_ns = perf_comb_gather_single_ns;
    host_state.perf_comb_gather_multi_ns = perf_comb_gather_multi_ns;
    host_state.perf_comb_pack_meta_ns = perf_comb_pack_meta_ns;
    host_state.perf_comb_pack_meta_work_ns = perf_comb_pack_meta_work_ns;
    host_state.perf_comb_pack_meta_sync_ns = perf_comb_pack_meta_sync_ns;
    host_state.perf_comb_tma_store_ns = perf_comb_tma_store_ns;
    host_state.perf_comb_tma_wait_max_ns = perf_comb_tma_wait_max_ns;
    host_state.perf_comb_wait_ready_max_ns = perf_comb_wait_ready_max_ns;
    host_state.perf_comb_wait_ready_single_max_ns = perf_comb_wait_ready_single_max_ns;
    host_state.perf_comb_wait_ready_multi_max_ns = perf_comb_wait_ready_multi_max_ns;
    host_state.perf_comb_wait_ready_flush_max_ns = perf_comb_wait_ready_flush_max_ns;
    host_state.perf_comb_wait_ready_full_max_ns = perf_comb_wait_ready_full_max_ns;
    host_state.perf_comb_wait_top_ns = perf_comb_wait_top_ns;
    host_state.perf_comb_wait_top_token = perf_comb_wait_top_token;
    host_state.perf_comb_wait_top_slot = perf_comb_wait_top_slot;
    host_state.perf_comb_wait_top_expert = perf_comb_wait_top_expert;
    host_state.perf_comb_wait_top_from_flush = perf_comb_wait_top_from_flush;
    host_state.perf_comb_gather_reduce_max_ns = perf_comb_gather_reduce_max_ns;
    host_state.perf_comb_gather_single_max_ns = perf_comb_gather_single_max_ns;
    host_state.perf_comb_gather_multi_max_ns = perf_comb_gather_multi_max_ns;
    host_state.perf_comb_pack_meta_max_ns = perf_comb_pack_meta_max_ns;
    host_state.perf_comb_pack_meta_work_max_ns = perf_comb_pack_meta_work_max_ns;
    host_state.perf_comb_pack_meta_sync_max_ns = perf_comb_pack_meta_sync_max_ns;
    host_state.perf_comb_tma_store_max_ns = perf_comb_tma_store_max_ns;
    host_state.perf_comb_nhit_sum = perf_comb_nhit_sum;
    host_state.perf_comb_token_count = perf_comb_token_count;
    host_state.perf_comb_single_token_count = perf_comb_single_token_count;
    host_state.perf_comb_multi_token_count = perf_comb_multi_token_count;

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
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_cta_barrier_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_cta_barrier_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_channel_barrier_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_channel_barrier_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_round_barrier_ns, acc_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_round_barrier_ns, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_tokens, acc_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_tokens, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_local_hit_tokens, acc_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_local_hit_tokens, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_local_hits, acc_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_local_hits, 0, acc_bytes));
    const size_t disp_role_bytes = (size_t)num_logical_channels * 2 * MK_DISPATCH_ROLE_COUNT * NUM_MAX_NVL_PEERS * sizeof(int64_t);
    const size_t disp_recv_bytes = (size_t)num_logical_channels * 2 * NUM_MAX_NVL_PEERS * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_cta_release_ts, acc_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_cta_release_ts, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_role_arrive_ts, disp_role_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_role_arrive_ts, 0, disp_role_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_role_work_ns, disp_role_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_role_work_ns, 0, disp_role_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_wait_nvl_ns, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_wait_nvl_ns, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_wait_ns, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_wait_ns, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_wait_start_ts, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_wait_start_ts, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_observe_ts, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_observe_ts, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_done_ts, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_done_ts, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_slowest_rdma, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_slowest_rdma, 0xff, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_src_nvl, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_src_nvl, 0xff, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_raw_start, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_raw_start, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_raw_end, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_raw_end, 0, disp_recv_bytes));
    const size_t disp_prefix_prod_bytes = disp_recv_bytes * (host_state.num_ranks / NUM_MAX_NVL_PEERS);
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_store_begin_ts, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_store_begin_ts, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_publish_ts, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_publish_ts, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_fence_done_ts, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_fence_done_ts, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_store_to_fence_ns, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_store_to_fence_ns, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_wait_ns, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_wait_ns, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_tokens, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_tokens, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_producer_rank, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_producer_rank, 0xff, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_producer_nvl, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_producer_nvl, 0xff, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_producer_dst_nvl, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_producer_dst_nvl, 0xff, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_producer_src_rdma, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_producer_src_rdma, 0xff, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_token_loop_ns, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_token_loop_ns, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_retire_ns, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_retire_ns, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_publish_ns, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_publish_ns, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_tokens, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_tokens, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_local_hits, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_local_hits, 0, disp_recv_bytes));
    host_state.perf_disp_pub_scan_ns = perf_disp_pub_scan_ns;
    host_state.perf_disp_pub_atomic_ns = perf_disp_pub_atomic_ns;
    host_state.perf_disp_pub_fence_ns = perf_disp_pub_fence_ns;
    host_state.perf_disp_pub_store_ns = perf_disp_pub_store_ns;

    // Scheduler bridge timers.
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_ts, 2 * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_ts, 0, 2 * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_scan_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_scan_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_enqueue_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_enqueue_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_idle_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_idle_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_normal_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_normal_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_tail_flush_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_tail_flush_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_publish_total_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_publish_total_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_publish_wait_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_publish_wait_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_publish_wait_max_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_publish_wait_max_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_publish_wait_max_tail, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_publish_wait_max_tail, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_publish_wait_max_visible_tail, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_publish_wait_max_visible_tail, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_scan_tokens, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_scan_tokens, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_ready_tokens, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_ready_tokens, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_full_batch_hits, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_full_batch_hits, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_batch_already_enqueued, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_batch_already_enqueued, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_not_full, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_not_full, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_normal_full_batch_enqueues, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_normal_full_batch_enqueues, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_flush_tail_enqueues, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_flush_tail_enqueues, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_queue_empty_count, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_queue_empty_count, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_queue_empty_after_dispatch_count, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_queue_empty_after_dispatch_count, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_max_ready_tail_gap, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_max_ready_tail_gap, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_stall_expert, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_stall_expert, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_stall_recv_count, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_stall_recv_count, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_stall_alloc_count, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_stall_alloc_count, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_stall_enqueue_cursor, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_stall_enqueue_cursor, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_stall_first_unready_slot, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_stall_first_unready_slot, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_stall_first_unready_ready, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_stall_first_unready_ready, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_stall_dispatch_done, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_stall_dispatch_done, 0, sizeof(int64_t)));

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
    CUDA_CHECK(cudaMalloc(&host_state.perf_compute_task_has_multi, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_compute_task_has_multi, 0, diag_i32_bytes));
    alloc_diag_i64(&host_state.perf_task_publish_ts);
    alloc_diag_i64(&host_state.perf_task_pop_start_ts);
    alloc_diag_i64(&host_state.perf_task_pop_done_ts);
    alloc_diag_i64(&host_state.perf_task_bcast_done_ts);
    alloc_diag_i64(&host_state.perf_task_start_ts);
    alloc_diag_i64(&host_state.perf_task_prev_gap_ns);
    CUDA_CHECK(cudaMalloc(&host_state.perf_task_pop_attempts, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_task_pop_attempts, 0, diag_i32_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_task_cas_failures, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_task_cas_failures, 0, diag_i32_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_task_group_id, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_task_group_id, 0xff, diag_i32_bytes));
#endif

    host_state.combine_rdma_buffer_ptr = combine_rdma_ptr;
    host_state.combine_buffer_ptrs = combine_buffer_ptrs;
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
    host_state.num_max_combine_rdma_chunked_send_tokens = combine_num_max_rdma_chunked_send_tokens;
    host_state.num_max_combine_rdma_chunked_recv_tokens = combine_num_max_rdma_chunked_recv_tokens;
    host_state.num_max_combine_nvl_chunked_send_tokens = combine_num_max_nvl_chunked_send_tokens;
    host_state.num_max_combine_nvl_chunked_recv_tokens = combine_num_max_nvl_chunked_recv_tokens;
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
    CUDA_CHECK(cudaFree(host_state.compute_output_slot));
    CUDA_CHECK(cudaFree(host_state.compute_slot_ready));
    CUDA_CHECK(cudaFree(host_state.compute_slot_from_flush));
    CUDA_CHECK(cudaFree(host_state.compute_slot_ready_ts));
    CUDA_CHECK(cudaFree(host_state.token_nhits));
    CUDA_CHECK(cudaFree(host_state.token_slot_list));
    CUDA_CHECK(cudaFree(host_state.priority_token_cursor));
    CUDA_CHECK(cudaFree(host_state.expert_batch_enqueued));
    CUDA_CHECK(cudaFree(host_state.compute_group_barrier));
    CUDA_CHECK(cudaFree(host_state.compute_group_phase));
    CUDA_CHECK(cudaFree(host_state.compute_tasks));
    CUDA_CHECK(cudaFree(host_state.compute_task_head));
    CUDA_CHECK(cudaFree(host_state.compute_task_tail));
    CUDA_CHECK(cudaFree(host_state.compute_task_reserve_tail));
    CUDA_CHECK(cudaFree(host_state.compute_enqueue_done));
    CUDA_CHECK(cudaFree(host_state.scheduler_done_count));
    CUDA_CHECK(cudaFree(host_state.expert_enqueue_cursor));
    CUDA_CHECK(cudaFree(host_state.compute_group_task_idx));
    CUDA_CHECK(cudaFree(host_state.token_done_count));
    CUDA_CHECK(cudaFree(host_state.gather_claimed));
    CUDA_CHECK(cudaFree(host_state.combine_token_ready));
    CUDA_CHECK(cudaFree(host_state.combine_done_count));
    CUDA_CHECK(cudaFree(host_state.combine_all_done));
    CUDA_CHECK(cudaFree(host_state.token_publish_done));
    CUDA_CHECK(cudaFree(host_state.pending_topk_idx));
    CUDA_CHECK(cudaFree(host_state.pending_topk_weights));
    CUDA_CHECK(cudaFree(host_state.pending_meta));
    CUDA_CHECK(cudaFree(host_state.pub_ring));
    CUDA_CHECK(cudaFree(host_state.pub_ring_head));
    CUDA_CHECK(cudaFree(host_state.pub_ring_tail));
    CUDA_CHECK(cudaFree(host_state.recv_warp_done));
    CUDA_CHECK(cudaFree(host_state.publish_done_count));
    CUDA_CHECK(cudaFree(host_state.publish_all_done));
    CUDA_CHECK(cudaFree(host_state.combined_x));
    CUDA_CHECK(cudaFree(host_state.combined_topk_weights));
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
    CUDA_CHECK(cudaFree(host_state.perf_disp_cta_barrier_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_channel_barrier_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_round_barrier_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_disp_local_hit_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_disp_local_hits));
    CUDA_CHECK(cudaFree(host_state.perf_disp_cta_release_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_role_arrive_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_role_work_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_wait_nvl_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_wait_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_wait_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_observe_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_slowest_rdma));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_src_nvl));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_raw_start));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_raw_end));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_store_begin_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_publish_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_fence_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_store_to_fence_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_wait_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_producer_rank));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_producer_nvl));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_producer_dst_nvl));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_producer_src_rdma));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_token_loop_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_retire_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_publish_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_local_hits));
    CUDA_CHECK(cudaFree(host_state.perf_comb_tma_wait_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_single_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_multi_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_flush_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_full_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_flush_count));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_full_count));
    CUDA_CHECK(cudaFree(host_state.perf_comb_gather_reduce_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_gather_single_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_gather_multi_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_pack_meta_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_pack_meta_work_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_pack_meta_sync_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_tma_store_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_tma_wait_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_single_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_multi_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_flush_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_ready_full_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_top_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_top_token));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_top_slot));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_top_expert));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_top_from_flush));
    CUDA_CHECK(cudaFree(host_state.perf_comb_gather_reduce_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_gather_single_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_gather_multi_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_pack_meta_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_pack_meta_work_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_pack_meta_sync_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_tma_store_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_comb_nhit_sum));
    CUDA_CHECK(cudaFree(host_state.perf_comb_token_count));
    CUDA_CHECK(cudaFree(host_state.perf_comb_single_token_count));
    CUDA_CHECK(cudaFree(host_state.perf_comb_multi_token_count));
    CUDA_CHECK(cudaFree(host_state.perf_sched_ts));
    CUDA_CHECK(cudaFree(host_state.perf_sched_scan_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_enqueue_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_idle_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_normal_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_tail_flush_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_publish_total_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_publish_wait_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_publish_wait_max_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_publish_wait_max_tail));
    CUDA_CHECK(cudaFree(host_state.perf_sched_publish_wait_max_visible_tail));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_scan_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_ready_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_full_batch_hits));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_batch_already_enqueued));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_not_full));
    CUDA_CHECK(cudaFree(host_state.perf_sched_normal_full_batch_enqueues));
    CUDA_CHECK(cudaFree(host_state.perf_sched_flush_tail_enqueues));
    CUDA_CHECK(cudaFree(host_state.perf_sched_queue_empty_count));
    CUDA_CHECK(cudaFree(host_state.perf_sched_queue_empty_after_dispatch_count));
    CUDA_CHECK(cudaFree(host_state.perf_sched_max_ready_tail_gap));
    CUDA_CHECK(cudaFree(host_state.perf_sched_stall_expert));
    CUDA_CHECK(cudaFree(host_state.perf_sched_stall_recv_count));
    CUDA_CHECK(cudaFree(host_state.perf_sched_stall_alloc_count));
    CUDA_CHECK(cudaFree(host_state.perf_sched_stall_enqueue_cursor));
    CUDA_CHECK(cudaFree(host_state.perf_sched_stall_first_unready_slot));
    CUDA_CHECK(cudaFree(host_state.perf_sched_stall_first_unready_ready));
    CUDA_CHECK(cudaFree(host_state.perf_sched_stall_dispatch_done));
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
    CUDA_CHECK(cudaFree(host_state.perf_compute_task_has_multi));
    CUDA_CHECK(cudaFree(host_state.perf_task_publish_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_pop_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_pop_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_bcast_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_prev_gap_ns));
    CUDA_CHECK(cudaFree(host_state.perf_task_pop_attempts));
    CUDA_CHECK(cudaFree(host_state.perf_task_cas_failures));
    CUDA_CHECK(cudaFree(host_state.perf_task_group_id));
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
