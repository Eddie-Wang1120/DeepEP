/**
 * megakernel_forward_backward.cu: Fused dispatch, compute, combine, and backward debug path
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

#include "../config.hpp"
#include "api.cuh"
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
#include "megakernel_compute_umma_fp8.cuh"

#include <cute/arch/simd_sm100.hpp>

#include <ATen/cuda/CUDABlas.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <limits>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#if MK_PERF_TRACE_ENABLED
#include <cstdio>
#endif

namespace deep_ep {
namespace megakernel_debug {

using ComputeDType = megakernel::ComputeDType;
// The compute headers define these helper namespaces under deep_ep::megakernel.
// Alias them so the copied body keeps using unqualified umma::/umma_fp8::.
namespace umma = ::deep_ep::megakernel::umma;
namespace umma_fp8 = ::deep_ep::megakernel::umma_fp8;

// ============================================================================
// Configuration
// ============================================================================

constexpr int COMPUTE_BATCH_SIZE = megakernel_config::kComputeBatchSize;
constexpr int COMPUTE_GROUP_SIZE = megakernel_config::kComputeGroupSize;
constexpr int COMPUTE_SCHEDULER_SMS = megakernel_config::kComputeSchedulerSms;
constexpr int GATHER_SMS = megakernel_config::kGatherSms;
constexpr int PRIORITY_SCHED_TID_BEGIN = megakernel_config::kPrioritySchedTidBegin;
constexpr int GATHER_SCHED_TID_BEGIN = megakernel_config::kGatherSchedTidBegin;
constexpr int NORMAL_SCHED_THREADS = megakernel_config::kNormalSchedThreads;
constexpr int GATHER_SCHED_MAX_WARPS = megakernel_config::kGatherSchedMaxWarps;
constexpr int MK_COMPUTE_CLUSTER_DIM = megakernel_config::kComputeClusterDim;
constexpr int COMBINE_START_HEAD_PERCENT = megakernel_config::kCombineStartHeadPercent;
constexpr int WMMA_M = megakernel_config::kWmmaM;
constexpr int WMMA_N = megakernel_config::kWmmaN;
constexpr int WMMA_K = megakernel_config::kWmmaK;
constexpr int MK_TIMEOUT_LOG_BUDGET = megakernel_config::kTimeoutLogBudget;
constexpr int MK_PRIORITY_SCAN_WINDOW_TOKENS = megakernel_config::kPriorityScanWindowTokens;
constexpr int MK_PRIORITY_MAX_ENQUEUE_PER_LOOP = megakernel_config::kPriorityMaxEnqueuePerLoop;
constexpr int MK_PRIORITY_ALREADY_SKIP_EPOCHS = megakernel_config::kPriorityAlreadySkipEpochs;
constexpr int MK_PRIORITY_NOT_READY_RETRY_EPOCHS = megakernel_config::kPriorityNotReadyRetryEpochs;
constexpr int MK_DISPATCH_ROLE_COUNT = megakernel_config::kDispatchRoleCount;
constexpr int PUB_RING_DEPTH = megakernel_config::kPubRingDepth;
constexpr int PUB_CONSUME_BATCH = megakernel_config::kPubConsumeBatch;
constexpr int PUB_PRODUCE_BATCH = megakernel_config::kPubProduceBatch;
#ifndef MK_ASYNC_PUBLISH
#define MK_ASYNC_PUBLISH 1
#endif
#ifndef MK_PRIORITY_ENABLE
#define MK_PRIORITY_ENABLE 0
#endif
#ifndef MK_UMMA_SAVE_PREACT
#define MK_UMMA_SAVE_PREACT 1
#endif
#ifndef MK_UMMA_GATEUP
#define MK_UMMA_GATEUP 1
#endif
#ifndef MK_UMMA_DOWN
#define MK_UMMA_DOWN 1
#endif

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
    kGather         // Dedicated gather SM worker for nhits>1 local reduce
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

struct MegaKernelBackwardState;

// Headroom for the smaller arena chunk granularity below: with an 8 MiB default
// chunk, large activation buffers each take their own exact-sized chunk and the
// remaining small buffers pack into a few chunks, so the chunk count grows but
// stays well under this cap for realistic cases.
constexpr int kMegakernelArenaChunkCap = 128;

struct MegaKernelState {
    int* timeout_log_counters;        // [kTimeoutLogCount] per-site bounded logging budget
    // --- DeepEP NVSHMEM infrastructure (from Buffer object) ---
    void* rdma_buffer_ptr;            // Symmetric RDMA buffer base (for SymBuffer construction)
    void** buffer_ptrs;               // NVL buffer pointer array [NUM_MAX_NVL_PEERS]
    void** allocator_combine_buffer_ptrs; // External Buffer infrastructure used to rebuild state

    // --- Dispatch input data ---
    const int4* x;                    // [num_tokens, hidden_int4] input token data
    const uint32_t* x_scales;         // [num_tokens, num_scales] packed UE8M0 scales (if FP8)
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
    int allocator_num_dispatch_sms;
    int allocator_num_forwarder_sms;
    int allocator_num_compute_sms;
    int allocator_num_combine_sms;
    int allocator_num_logical_channels;
    int allocator_max_tokens_per_expert;
    int allocator_max_total_recv_tokens;
    int64_t allocator_num_rdma_bytes;
    int64_t allocator_num_nvl_bytes;

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
    // --- Compact per-expert slot layout (P0 memory optimization scaffolding) ---
    // Phase 0: placeholder values (expert_slot_base[le]=le*max_tokens_per_expert,
    // expert_count[le]=max_tokens_per_expert) so addressing is byte-identical to the
    // legacy `le*max_tpe+slot` scheme. Not read by any kernel yet. Later phases switch
    // these to real exclusive-prefix-sum bases / real per-expert counts to compact the
    // per-expert-slot buffers from num_local_experts*max_total_recv_tokens down to Σ count.
    int* expert_slot_base;            // [num_local_experts] base offset into per-expert-slot buffers
    int* expert_count;                // [num_local_experts] received token count per local expert
    int* recv_token_source_info;      // [max_total_recv_tokens, 2] — (recv_token_idx, topk_slot)
    float* recv_token_route_weights;  // [max_total_recv_tokens] — route weight for this compute slot
    internode::SourceMeta* recv_src_meta; // [max_total_recv_tokens] — DeepEP SourceMeta for combine routing

    // --- Compute signaling / per-slot output path (MEGAKERNEL_COMPUTE_DESIGN section III) ---
    int* token_compute_expected;        // [max_total_recv_tokens] how many local experts must compute this token
    __nv_bfloat16* compute_output_slot; // [num_local_experts * max_tokens_per_expert, hidden] per-slot output
    // --- Backward activation save (forward writes; backward reads) ---
    // fc1 input (permuted token X) saved by RECV_TOKEN index (compact, prefix-driven,
    // deterministic across a dispatch re-run) rather than by expert slot (which is
    // assigned via a non-deterministic atomicAdd). The backward megakernel re-runs
    // dispatch with x=grad_output, so combine_input[recv_token] becomes grad_down;
    // the original X is recovered here by the same recv_token index. A token that hits
    // multiple local experts writes the same X (idempotent).
    __nv_bfloat16* bwd_fc1_input;       // [max_total_recv_tokens, hidden] fc1 input (permuted X) by recv_token
    __nv_bfloat16* bwd_preact;          // [max_total_recv_tokens, num_topk, 2 * intermediate] saved gate/up by recv_token/topk_slot
    bool owns_bwd_fc1_input;
    bool owns_bwd_preact;
    // Phase 3 (A2): compact preact storage. bwd_preact is keyed by the compact forward slot
    // (like recv_tokens), and fwd_slot_map translates the cross-pass-stable key
    // (recv_token, topk_slot) -> forward slot so the backward pass can find it (slot ids are
    // non-deterministic across the two dispatch runs, but recv_token/topk_slot are stable).
    int* fwd_slot_map;                  // [max_total_recv_tokens * num_topk] (recv_token,topk_slot) -> forward slot, -1 if none
    bool owns_fwd_slot_map;
    int* token_nhits;                   // [max_total_recv_tokens] #local-expert hits for this recv token
    int* token_slot_list;               // [max_total_recv_tokens * num_topk] absolute slot ids per hit
    int* priority_token_cursor;         // scheduler combine-order cursor for token priority scan
    int* expert_batch_enqueued;         // [num_local_experts * max_batches_per_expert] enqueue source: 0=none, 1=normal, 2=priority, 3=tail
#if MK_PERF_TRACE_ARGS
    int64_t* expert_batch_enqueue_ts;   // [num_local_experts * max_batches_per_expert] publish timestamp for source attribution
    int* token_priority_dep_count;      // [max_total_recv_tokens] number of token hit batches first enqueued by priority
#endif
    int* priority_batch_skip_epoch;     // [num_local_experts * max_batches_per_expert] last priority epoch that saw an already-enqueued batch
    int* priority_batch_retry_epoch;    // [num_local_experts * max_batches_per_expert] next priority epoch to retry not-full/not-ready batch
    int max_batches_per_expert;
    int* compute_group_barrier;         // [num_compute_groups] reusable global barrier counters
    int* compute_group_phase;           // [num_compute_groups] reusable global barrier phase flags
    ComputeTask* compute_tasks;         // [max_compute_tasks] dynamic compute task queue
    int max_compute_tasks;
    int* compute_task_head;             // CAS pop cursor
    int* compute_task_tail;             // visible publish cursor consumed by workers
    int* compute_task_reserve_tail;     // atomic reservation cursor used by scheduler lanes
    int* compute_enqueue_done;          // set after all scheduler lanes publish tail tasks
    int* scheduler_done_count;          // final-flush arrival barrier, then scheduler completion count
    int* priority_scheduler_done;       // 0=running/barrier closed, 1=priority stopped, 2=final-flush barrier released
    int* expert_enqueue_cursor;         // [num_local_experts] how many slots have been enqueued
    int* compute_group_task_idx;        // [num_compute_groups] broadcast popped task idx to group SMs

    // --- Dedicated Gather SM state ---
    int* token_done_count;              // [max_total_recv_tokens] atomicAdd by compute worker per slot completion
    int* gather_claimed;                // [max_total_recv_tokens] CAS flag: scheduler has batched this token for gather
    int* combine_token_ready;           // [max_total_recv_tokens] set when token is ready for combine
    int* gather_ready_queue;            // [max_total_recv_tokens] token storage for scheduler-built gather tasks
    int* gather_ready_head;             // task queue consumer cursor; gather SMs CAS-pop one task at a time
    int* gather_ready_tail;             // ordered visible task tail published by scheduler lanes
    int* gather_ready_reserve_tail;     // token-storage reservation cursor into gather_ready_queue
    int* gather_scan_cursor;            // [COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS] next token for each gather scheduler warp
    int* gather_task_count;             // task metadata reservation cursor
    int* gather_task_tokens;            // [max_total_recv_tokens] task_idx -> token_base in gather_ready_queue
    int* gather_task_nhits;             // [max_total_recv_tokens] task_idx -> number of tokens in the task
    int* combine_done_count;            // atomic: how many combine SMs have fully finished
    int* combine_all_done;              // flag: 1 once all combine SMs finished; gather SMs poll this to exit

    // --- Expert weights ---
    const __nv_bfloat16* W_gateup;    // [num_local_experts, 2 * intermediate, hidden], rows [g0,u0,...]
    const __nv_bfloat16* W_down;      // [num_local_experts, hidden, intermediate]

    // --- FP8 compute inputs (state plumbing only; BF16 compute remains active) ---
    ComputeDType compute_dtype;
    const umma_fp8::ElemAB* W_gateup_fp8;        // [num_local_experts, 2 * intermediate, hidden]
    const umma_fp8::ElemAB* W_down_fp8;          // [num_local_experts, hidden, intermediate]
    const umma_fp8::ScalePack* W_gateup_fp8_sf;  // [num_local_experts, 2 * intermediate, ceil(hidden / (128 * 4))]
    const umma_fp8::ScalePack* W_down_fp8_sf;    // [num_local_experts, hidden, ceil(intermediate / (128 * 4))]
    umma_fp8::ElemAB* recv_tokens_fp8;           // [num_local_experts * max_tokens_per_expert, hidden]
    umma_fp8::ScalePack* recv_tokens_fp8_sf;     // [num_local_experts * max_tokens_per_expert, ceil(hidden / (128 * 4))]
    umma_fp8::ElemAB* input_fp8_workspace;       // [num_compute_groups, COMPUTE_BATCH_SIZE, hidden]
    umma_fp8::ScalePack* input_fp8_sf_workspace; // [num_compute_groups, COMPUTE_BATCH_SIZE, ceil(hidden / (128 * 4))]
    umma_fp8::ElemAB* act_fp8_workspace;         // [num_compute_groups, COMPUTE_BATCH_SIZE, intermediate]
    umma_fp8::ScalePack* act_fp8_sf_workspace;   // [num_compute_groups, COMPUTE_BATCH_SIZE, ceil(intermediate / (128 * 4))]
    umma_fp8::ComputeFp8TmaAtoms* compute_fp8_tma;
    umma_fp8::ComputeFp8DownTmaAtoms* compute_fp8_down_tma;
    umma_fp8::InputFp8TmaAtom_t* group_input_fp8_tma;
    int fp8_hidden_scale_k_packed;
    int fp8_intermediate_scale_k_packed;

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
    int* combine_rdma_head_work;      // [num_combined_tokens, kNumRDMARanks] normalized combine RDMA heads
    int* combine_nvl_head_work;       // [num_tokens upper bound, NUM_MAX_NVL_PEERS] normalized combine NVL heads
    __nv_bfloat16* gemm_workspace;    // Scratch for gate/up intermediate results

    // --- Combine output ---
    float* output_accum;              // [num_tokens, hidden] float accumulator

    // --- Combine infrastructure (DeepEP combine kernel inputs) ---
    void* combine_rdma_buffer_ptr;            // Symmetric RDMA buffer for combine
    void** combine_buffer_ptrs;               // NVL buffer ptrs for combine [NUM_MAX_NVL_PEERS]
    int64_t num_rdma_bytes;                   // Capacity of each dispatch/combine RDMA region
    int64_t num_nvl_bytes;                    // Capacity of each dispatch/combine NVL region
    int4* combined_x;                         // [num_combined_tokens, hidden_int4] final output
    float* combined_topk_weights;             // [num_combined_tokens, num_topk] final topk weights
    bool owns_combined_x;
    bool owns_combined_topk_weights;
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

    // --- Compute dimensions ---
    int hidden_dim;
    int intermediate_dim;
    int num_local_experts;
    int max_tokens_per_expert;
    int total_expert_slots;           // Σ expert_count[le] = size of the per-expert-slot buffers
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

    // --- RDMA buffer reuse (dispatch <-> combine) ---
    // Symmetric mailboxes indexed by rdma_rank (size num_rdma_ranks). Peers publish per-phase
    // "done" via IBGDA into the local copy; the combine prelude polls the local copy.
    int* rdma_reuse_dispatch_quiet_done;  // borrowed (Buffer-owned symmetric memory)
    int* rdma_reuse_combine_clear_done;   // borrowed (Buffer-owned symmetric memory)
    int* rdma_reuse_prelude_done;         // state-owned [1] gate: leader sets, others wait
    int rdma_reuse_prelude_enable;        // 1 = run combine RDMA-reuse prelude (forward only)

    // --- Publish offload (dispatch->compute bridge), Stage 1: backing state only ---
    // receiver copies token data + stashes topk/meta into pending_* (indexed by
    // recv_token_idx, overwrite-safe), then pushes recv_token_idx into its SPSC ring.
    // publisher warp consumes, does the slot alloc / metadata writes / ready publish.
    int* pending_topk_idx;            // [max_total_recv_tokens * num_topk] receiver-stashed expert ids
    float* pending_topk_weights;      // [max_total_recv_tokens * num_topk] receiver-stashed routing weights
    internode::SourceMeta* pending_meta; // [max_total_recv_tokens] receiver-stashed SourceMeta
    int* pub_ring;                    // [num_pub_warps_total * PUB_RING_DEPTH] recv_token_idx queue
    int* pub_ring_head;               // [num_pub_warps_total] consumer cursor (publisher)
    int* pub_ring_tail;               // [num_pub_warps_total] producer cursor (receiver)
    int* recv_warp_done;              // [num_pub_warps_total] receiver warp finished producing
    int* publish_warp_done;           // [num_pub_warps_total] publisher warp drained and released metadata
    int* publish_done_count;          // atomic: how many publisher warps have drained
    int* publish_all_done;            // flag: 1 once all publishers drained (scheduler/gather use)
    int num_pub_warps_total;          // = (num_dispatch_sms / 2) * NUM_MAX_NVL_PEERS

    // Backing store for the fused buffer-init descriptors (see fused_fill_kernel). Freed with the state.
    void* fused_fill_desc_buf;

    // --- Owned arena bookkeeping ---
    void* persistent_arena_chunks[kMegakernelArenaChunkCap];
    int persistent_arena_chunk_count;
    void* transient_arena_chunks[kMegakernelArenaChunkCap];
    int transient_arena_chunk_count;

#if MK_PERF_TRACE_ENABLED
    // Per-logical-channel timing. Each logical channel has sender and forwarder rows
    // for both dispatch and combine so channel-level overlap is visible in Perfetto.
    // Dispatch phases: 0=enter, 1=channel_barrier_start, 2=round_barrier_start, 3=exit
    // Combine phases:  0=enter, 1=dispatch_done_acquired, 2=head_norm_done, 3=protocol_start, 4=exit
    static constexpr int MK_PERF_NUM_LCH_PHASES = 5;
    int64_t* perf_dispatch_lch_ts;     // [num_logical_channels * 2 * MK_PERF_NUM_LCH_PHASES]
    int64_t* perf_combine_lch_ts;      // [num_logical_channels * 2 * MK_PERF_NUM_LCH_PHASES]
    int64_t* perf_async_pub_start_ts;  // [num_pub_warps_total] publisher warp first active timestamp
    int64_t* perf_async_pub_end_ts;    // [num_pub_warps_total] publisher warp exit timestamp
    int64_t* perf_async_publish_all_done_ts; // [1] timestamp when publish_all_done is released
    int64_t* perf_sched_ts;            // [2] scheduler start/end
#endif
#if MK_PERF_TRACE_ARGS
    struct DispatchRoundTrace {
        int64_t sender_work_begin_ns, sender_work_end_ns;
        int64_t forwarder_wait_begin_ns, forwarder_wait_end_ns;
        int64_t forwarder_work_begin_ns, forwarder_work_end_ns;
        int64_t channel_barrier_arrival_ns, round_barrier_arrival_ns, async_publish_ns;
        int64_t actual_token_count, rdma_packet_slot_count, polling_iteration_count;
        int64_t ready_wait_ns, tail_head_wait_ns;
    };
    // Fixed round dimension: num_ranks / NUM_MAX_NVL_PEERS.
    DispatchRoundTrace* perf_dispatch_round_trace;

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
    int64_t* perf_disp_allrecv_prefix_start_first_negative_ts;
    int64_t* perf_disp_allrecv_prefix_end_first_negative_ts;
    int64_t* perf_disp_allrecv_prefix_pair_ready_ts;
    int64_t* perf_disp_allrecv_prefix_poll_count;
    int64_t* perf_disp_allrecv_prefix_mixed_poll_count;
    int64_t* perf_disp_allrecv_prefix_slowest_rdma;
    int64_t* perf_disp_allrecv_prefix_src_nvl;
    int64_t* perf_disp_allrecv_prefix_raw_start;
    int64_t* perf_disp_allrecv_prefix_raw_end;
    int64_t* perf_disp_allrecv_token_loop_ns;
    int64_t* perf_disp_allrecv_retire_ns;
    int64_t* perf_disp_sender_meta_build_start_ts; // [lch * 2 * kNumRDMARanks]
    int64_t* perf_disp_sender_meta_ready_ts;
    int64_t* perf_disp_sender_meta_put_begin_ts;
    int64_t* perf_disp_sender_meta_put_end_ts;
    int64_t* perf_disp_sender_meta_raw_2;
    int64_t* perf_disp_sender_meta_raw_3;
    int64_t* perf_disp_prefix_meta_wait_start_ts; // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks]
    int64_t* perf_disp_prefix_meta_ready_ts;
    int64_t* perf_disp_prefix_meta_first_negative_ts; // [... * 4], one timestamp per metadata word
    int64_t* perf_disp_prefix_meta_mixed_poll_count;
    int64_t* perf_disp_prefix_meta_poll_count;
    int64_t* perf_disp_prefix_meta_raw_0;
    int64_t* perf_disp_prefix_meta_raw_1;
    int64_t* perf_disp_prefix_meta_raw_2;
    int64_t* perf_disp_prefix_meta_raw_3;
    int64_t* perf_disp_prefix_store_start_done_ts;
    int64_t* perf_disp_prefix_store_begin_ts; // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer before prefix stores
    int64_t* perf_disp_prefix_publish_ts;   // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer after prefix stores
    int64_t* perf_disp_prefix_fence_done_ts;
    int64_t* perf_disp_prefix_store_to_fence_ns;
    int64_t* perf_disp_prefix_meta_wait_ns; // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer wait for RDMA meta
    int64_t* perf_disp_prefix_raw_start;
    int64_t* perf_disp_prefix_raw_end;
    int64_t* perf_disp_prefix_tokens;       // [lch * 2 * NUM_MAX_NVL_PEERS * kNumRDMARanks] producer token count
    int64_t* perf_disp_prefix_producer_rank;
    int64_t* perf_disp_prefix_producer_nvl;
    int64_t* perf_disp_prefix_producer_dst_nvl;
    int64_t* perf_disp_prefix_producer_src_rdma;
    int64_t* perf_disp_allrecv_publish_ns;  // [lch * 2 * NUM_MAX_NVL_PEERS] all NVL receiver publish time
    int64_t* perf_disp_allrecv_tokens;      // [lch * 2 * NUM_MAX_NVL_PEERS] all NVL receiver tokens
    int64_t* perf_disp_allrecv_local_hits;  // [lch * 2 * NUM_MAX_NVL_PEERS] all NVL receiver local hits
    int64_t* perf_async_pub_wait_ring_ns;   // [num_pub_warps_total] empty-ring sleep time
    int64_t* perf_async_pub_poll_ns;        // [num_pub_warps_total] ring tail/done polling overhead
    int64_t* perf_async_pub_gap_ns;         // [num_pub_warps_total] non-work gap between consumed tokens
    int64_t* perf_async_pub_start_gap_ns;   // [num_pub_warps_total] worker-entry to first loop gap
    int64_t* perf_async_pub_empty_gap_ns;   // [num_pub_warps_total] empty-loop sleep end to next loop gap
    int64_t* perf_async_pub_done_recheck_ns; // [num_pub_warps_total] final done-path tail recheck time
    int64_t* perf_async_pub_ring_load_ns;   // [num_pub_warps_total] pub_ring entry load time
    int64_t* perf_async_pub_head_release_ns; // [num_pub_warps_total] pub_ring_head release time
    int64_t* perf_async_pub_syncwarp_ns;    // [num_pub_warps_total] post-token syncwarp time
    int64_t* perf_async_pub_batch_wall_ns;  // [num_pub_warps_total] non-empty batch wall time
    int64_t* perf_async_pub_batch_accounted_ns; // [num_pub_warps_total] accounted time inside non-empty batches
    int64_t* perf_async_pub_batch_unattributed_ns; // [num_pub_warps_total] residual time inside non-empty batches
    int64_t* perf_async_pub_token_gap_ns;   // [num_pub_warps_total] gap between tokens inside non-empty batches
    int64_t* perf_async_pub_helper_unattributed_ns; // [num_pub_warps_total] publish helper wall minus phase timers
    int64_t* perf_async_pub_finish_ns;      // [num_pub_warps_total] final done-count/publish_all_done overhead
    int64_t* perf_async_pub_work_ns;        // [num_pub_warps_total] total publish token work time
    int64_t* perf_async_pub_scan_ns;        // [num_pub_warps_total] pending topk scan + weight publish
    int64_t* perf_async_pub_atomic_ns;      // [num_pub_warps_total] token/expert atomic slot allocation
    int64_t* perf_async_pub_fence_ns;       // [num_pub_warps_total] publish threadfence time
    int64_t* perf_async_pub_store_ns;       // [num_pub_warps_total] source-info/ready stores
    int64_t* perf_async_pub_drain_ns;       // [num_pub_warps_total] time after receiver-done observed
    int64_t* perf_async_pub_tokens;         // [num_pub_warps_total] tokens consumed by publisher
    int64_t* perf_async_pub_batch_count;    // [num_pub_warps_total] non-empty consume batches
    int64_t* perf_async_pub_head_release_count; // [num_pub_warps_total] pub_ring_head release count
    int64_t* perf_async_pub_max_batch;      // [num_pub_warps_total] max tokens consumed in one batch
    int64_t* perf_async_pub_local_hit_tokens; // [num_pub_warps_total] tokens with local expert hits
    int64_t* perf_async_pub_local_hits;     // [num_pub_warps_total] total local expert hits
    int64_t* perf_comb_tma_wait_ns;        // sum: combine sender tma_store_wait before reusing smem buffer
    int64_t* perf_comb_wait_ready_ns;      // sum: combine sender wait on per-token combine_token_ready
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
    int64_t* perf_comb_wait_top_nhits;
    int64_t* perf_comb_wait_top_priority_deps;
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
    int64_t* perf_sched_priority_already_normal;
    int64_t* perf_sched_priority_already_priority;
    int64_t* perf_sched_priority_already_tail;
    int64_t* perf_sched_normal_after_priority_ns;
    int64_t* perf_sched_normal_after_priority_count;
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
    int64_t* perf_sched_first_done_seen_ts;
    int64_t* perf_sched_first_recv_count_advance_ts;
    int64_t* perf_sched_first_recv_count_advance_expert;
    int64_t* perf_sched_first_recv_count_advance_old;
    int64_t* perf_sched_first_recv_count_advance_new;
    int64_t* perf_sched_first_normal_enqueue_attempt_ts;
    int64_t* perf_sched_first_normal_enqueue_success_ts;
    int64_t* perf_sched_first_task_publish_ts;
    int64_t* perf_sched_first_task_source;
    int64_t* perf_sched_first_task_expert;
    int64_t* perf_sched_first_task_batch;
    int64_t* perf_sched_first_task_start_slot;
    int64_t* perf_sched_first_task_num_tokens;
#endif
#if MK_PERF_TRACE_ENABLED
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
    //   20 = ts after token-ready publish + sync (== signal done)
    //   21 = compute task queue index
    //   22 = expert-local start_slot
    //   23 = expert-local end_slot (exclusive)
    //   24 = absolute slot base (= expert_id * max_tokens_per_expert + start_slot)
    //   25 = is_flush task flag
    //   26 = valid record flag
    static constexpr int MK_PERF_NUM_COMPUTE_FIELDS = 27;
    int64_t* perf_compute_task;        // [max_compute_tasks * MK_PERF_NUM_COMPUTE_FIELDS], indexed by queue task_idx
    int* perf_compute_task_count;      // legacy counter; trace export scans valid records by task_idx
#endif
#if MK_PERF_TRACE_ARGS
    // Root-cause diagnostics, parallel arrays indexed by the same compute-task slot.
    // Rendered as args on each compute X-event (no extra COMPUTE_FIELDS).
    int* perf_compute_multi_expert_rows;  // [max_compute_tasks]
    int* perf_compute_task_has_multi;     // [max_compute_tasks] output-phase multi-expert flag
#endif
#if MK_PERF_TRACE_ENABLED
    // Per-gather-batch timing buffer. Fields:
    // 0=start, 1=end, 2=gather_sm_idx, 3=batch_size, 4=first_token, 5=last_token,
    // 6=nhit_sum, 7=scan_done_ts, 8=reduce_done_ts, 9=signal_done_ts
    static constexpr int MK_PERF_NUM_GATHER_FIELDS = 10;
    int64_t* perf_gather_task;            // [max_compute_tasks * MK_PERF_NUM_GATHER_FIELDS]
    int* perf_gather_task_count;          // atomic write cursor into perf_gather_task
#endif
#if MK_PERF_TRACE_ARGS
    // Queue handoff diagnostics, indexed by compute task queue index.
    int64_t* perf_task_publish_ts;        // scheduler published task tail
    int* perf_task_source;                // [max_compute_tasks] 1=normal, 2=priority, 3=tail
    int64_t* perf_task_pop_start_ts;      // compute group leader started dequeue loop
    int64_t* perf_task_pop_done_ts;       // compute group leader acquired this task
    int64_t* perf_task_bcast_done_ts;     // task_idx broadcast sync done for the group
    int64_t* perf_task_start_ts;          // compute task body timing start
    int64_t* perf_task_prev_end_ts;       // previous task end timestamp in the same compute group
    int64_t* perf_task_prev_gap_ns;       // this group's task_start - previous task end
    int* perf_task_pop_attempts;          // number of queue polls/CAS attempts for this task
    int* perf_task_cas_failures;          // failed CAS attempts before acquiring this task
    int* perf_task_group_id;              // group that popped this task
#endif
};

__device__ __forceinline__ bool mk_debug_bad_float(float v) {
    return v != v || v > 3.402823466e38f || v < -3.402823466e38f;
}

__device__ __forceinline__ bool mk_debug_check_bf16_matrix(
    const char* stage,
    const __nv_bfloat16* buf,
    int rows,
    int cols,
    int stride,
    int rank,
    int sm_id,
    int block_id,
    int group_id,
    int group_sm_idx,
    int task_idx,
    int expert_id) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float v = __bfloat162float(buf[(int64_t)row * stride + col]);
            if (mk_debug_bad_float(v)) {
                printf("[MK-NAN][%s] rank=%d block=%d sm=%d group=%d gsm=%d task=%d expert=%d row=%d col=%d v=%f\n",
                       stage, rank, block_id, sm_id, group_id, group_sm_idx, task_idx, expert_id, row, col, v);
                return true;
            }
        }
    }
    return false;
}

__device__ __forceinline__ bool mk_debug_check_float_vector(
    const char* stage,
    const float* buf,
    int count,
    int rank,
    int sm_id,
    int block_id,
    int group_id,
    int group_sm_idx,
    int task_idx,
    int expert_id) {
    for (int i = 0; i < count; ++i) {
        float v = buf[i];
        if (mk_debug_bad_float(v)) {
            printf("[MK-NAN][%s] rank=%d block=%d sm=%d group=%d gsm=%d task=%d expert=%d idx=%d v=%f\n",
                   stage, rank, block_id, sm_id, group_id, group_sm_idx, task_idx, expert_id, i, v);
            return true;
        }
    }
    return false;
}

// ============================================================================
// FP8 routing helpers
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

__device__ void device_gemm_bf16_mn(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [K, N] row-major
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
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A + row_offset * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + col_offset, N);
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
    float* smem_buf,
    __nv_bfloat16* __restrict__ preact = nullptr,
    const int* __restrict__ preact_recv_idx = nullptr,
    const int* __restrict__ preact_topk_idx = nullptr,
    int num_topk = 0,
    int preact_stride = 0
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
                if (preact != nullptr && preact_recv_idx != nullptr && preact_topk_idx != nullptr) {
                    const int recv_idx = preact_recv_idx[out_row];
                    const int topk_idx = preact_topk_idx[out_row];
                    if (recv_idx >= 0 && topk_idx >= 0) {
                        __nv_bfloat16* preact_row = preact + ((int64_t)recv_idx * num_topk + topk_idx) * preact_stride;
                        preact_row[2 * out_col] = __float2bfloat16(g);
                        preact_row[2 * out_col + 1] = __float2bfloat16(u);
                    }
                }
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

// Map an absolute per-expert slot index back to its local expert id via the
// expert_slot_base / expert_count arrays. Replaces the legacy `slot / max_tpe`
// division so it stays correct once the layout switches from fixed stride
// (base[e]=e*max_tpe) to a real exclusive-prefix-sum (variable stride).
// num_local_experts is small (e.g. 16), so a linear scan is cheap and only runs
// in scheduler warps, not the per-element compute loops.
__device__ __forceinline__ int mk_slot_to_local_expert(const MegaKernelState* state, int slot) {
    const int ne = state->num_local_experts;
    for (int e = 0; e < ne; ++e) {
        const int base = state->expert_slot_base[e];
        if (slot >= base && slot < base + state->expert_count[e])
            return e;
    }
    return ne - 1;  // fallback; should not happen for a valid allocated slot
}

// Instantiated template constants
constexpr int kNumDispatchRDMASenderWarps = 7;
constexpr int kNumTMABytesPerWarp = 16384;
constexpr bool kLowLatencyMode = false;
constexpr bool kCachedMode = false;

#if MK_ASYNC_PUBLISH
__device__ __forceinline__ int get_publish_warp_index(int dispatch_sm_idx, int src_nvl_rank) {
    return (dispatch_sm_idx / 2) * NUM_MAX_NVL_PEERS + src_nvl_rank;
}

__device__ __forceinline__ int publish_recv_token_from_pending(
    MegaKernelState* state,
    int recv_token_idx,
    int local_expert_begin,
    int num_topk,
    int lane_id
#if MK_PERF_TRACE_ARGS
    , int64_t* scan_ns,
    int64_t* atomic_ns,
    int64_t* fence_ns,
    int64_t* store_ns
#endif
) {
#if MK_PERF_TRACE_ARGS
    int64_t phase_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
#endif
    const int local_expert_end = local_expert_begin + state->num_local_experts;
    int expert_id = -1;
    float route_w = 0.0f;
    bool is_local_hit = false;
    if (lane_id < num_topk) {
        expert_id = ld_nc_global(&state->pending_topk_idx[recv_token_idx * num_topk + lane_id]);
        is_local_hit = (expert_id >= local_expert_begin && expert_id < local_expert_end);
        if (is_local_hit) {
            route_w = ld_nc_global(&state->pending_topk_weights[recv_token_idx * num_topk + lane_id]);
            state->combine_input_topk_weights[recv_token_idx * num_topk + lane_id] = route_w;
        }
    }

    const unsigned hit_mask = __ballot_sync(0xffffffff, is_local_hit);
    const int num_hits = __popc(hit_mask);
#if MK_PERF_TRACE_ARGS
    if (lane_id == 0) {
        int64_t now = globaltimer_ns();
        *scan_ns += now - phase_start_ns;
        phase_start_ns = now;
    }
#endif
    int hit_base = 0;
    int hit_abs_slot = -1;
    int hit_slot = -1;
    int hit_rank = is_local_hit ? __popc(hit_mask & ((1u << lane_id) - 1)) : -1;

    if (num_hits > 0) {
        if (lane_id == 0) {
            state->combine_input_src_meta[recv_token_idx] = state->pending_meta[recv_token_idx];
            if (num_hits > num_topk) {
                printf("MK publish token hit overflow, rank=%d recv_token=%d num_hits=%d num_topk=%d\n",
                       state->rank, recv_token_idx, num_hits, num_topk);
                __threadfence_system(); trap();
            }
            st_na_global(&state->token_nhits[recv_token_idx], num_hits);
            st_na_global(&state->token_compute_expected[recv_token_idx], num_hits);
        }

        if (is_local_hit) {
            int local_expert_id = expert_id - local_expert_begin;
            unsigned same_expert_mask = __match_any_sync(hit_mask, local_expert_id) & hit_mask;
            int leader_lane = __ffs(same_expert_mask) - 1;
            int group_count = __popc(same_expert_mask);
            int rank_in_expert = __popc(same_expert_mask & ((1u << lane_id) - 1));

            int group_base_slot = 0;
            if (lane_id == leader_lane) {
                group_base_slot = atomicAdd(&state->expert_token_offsets[local_expert_id], group_count);
                if (group_base_slot + group_count > state->expert_count[local_expert_id]) {
                    printf("MK publish expert slot overflow, rank=%d recv_token=%d expert=%d slot=%d count=%d max_tpe=%d\n",
                           state->rank, recv_token_idx, expert_id, group_base_slot, group_count, state->max_tokens_per_expert);
                    __threadfence_system(); trap();
                }
            }
            group_base_slot = __shfl_sync(same_expert_mask, group_base_slot, leader_lane);
            hit_slot = group_base_slot + rank_in_expert;
            hit_abs_slot = state->expert_slot_base[local_expert_id] + hit_slot;
        }

#if MK_PERF_TRACE_ARGS
        if (lane_id == 0) {
            int64_t now = globaltimer_ns();
            *atomic_ns += now - phase_start_ns;
            phase_start_ns = now;
        }
#endif

        if (is_local_hit) {
            int* dst_ptr = &state->recv_token_source_info[hit_abs_slot * 2];
            st_na_global(dst_ptr, recv_token_idx);
            st_na_global(dst_ptr + 1, lane_id);
            state->token_slot_list[recv_token_idx * num_topk + hit_base + hit_rank] = hit_abs_slot;
        }

#if MK_PERF_TRACE_ARGS
        if (lane_id == 0) {
            int64_t now = globaltimer_ns();
            *store_ns += now - phase_start_ns;
            phase_start_ns = now;
        }
#endif
        __threadfence();
#if MK_PERF_TRACE_ARGS
        if (lane_id == 0) {
            int64_t now = globaltimer_ns();
            *fence_ns += now - phase_start_ns;
            phase_start_ns = now;
        }
#endif
        if (is_local_hit) {
            int local_expert_id = expert_id - local_expert_begin;
            st_na_release(&state->expert_slot_ready[state->expert_slot_base[local_expert_id] + hit_slot], 1);
        }
#if MK_PERF_TRACE_ARGS
        if (lane_id == 0)
            *store_ns += globaltimer_ns() - phase_start_ns;
#endif
    }
    __syncwarp();
    return num_hits;
}

__device__ void publish_worker_v2(int dispatch_sm_idx, int src_nvl_rank, MegaKernelState* state) {
    const int lane_id = get_lane_id();
    const int pw = get_publish_warp_index(dispatch_sm_idx, src_nvl_rank);
    const int local_expert_begin = state->rank * state->num_local_experts;
    const int num_topk = state->num_topk;
    int head = 0;
#if MK_PERF_TRACE_ARGS
    int64_t worker_start_ns = 0;
#endif
#if MK_PERF_TRACE_ENABLED
    if (lane_id == 0) {
        int64_t start_ns = globaltimer_ns();
        state->perf_async_pub_start_ts[pw] = start_ns;
#if MK_PERF_TRACE_ARGS
        worker_start_ns = start_ns;
#endif
    }
#endif
#if MK_PERF_TRACE_ARGS
    int64_t recv_done_observe_ns = 0;
    int64_t last_token_done_ns = 0;
    int64_t last_empty_sleep_end_ns = 0;
#endif

    while (true) {
#if MK_PERF_TRACE_ARGS
        int64_t loop_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
        if (lane_id == 0) {
            if (worker_start_ns != 0) {
                state->perf_async_pub_start_gap_ns[pw] += loop_start_ns - worker_start_ns;
                worker_start_ns = 0;
            }
            if (last_empty_sleep_end_ns != 0 && loop_start_ns > last_empty_sleep_end_ns) {
                state->perf_async_pub_empty_gap_ns[pw] += loop_start_ns - last_empty_sleep_end_ns;
                last_empty_sleep_end_ns = 0;
            }
        }
        int recv_done_snapshot = ld_acquire_global(&state->recv_warp_done[pw]);
        if (lane_id == 0 && recv_done_observe_ns == 0 && recv_done_snapshot != 0)
            recv_done_observe_ns = loop_start_ns;
#else
        int recv_done_snapshot = ld_acquire_global(&state->recv_warp_done[pw]);
#endif
        int tail = ld_acquire_global(&state->pub_ring_tail[pw]);
        if (head == tail) {
            if (recv_done_snapshot != 0) {
                tail = ld_acquire_global(&state->pub_ring_tail[pw]);
#if MK_PERF_TRACE_ARGS
                if (lane_id == 0) {
                    int64_t recheck_done_ns = globaltimer_ns();
                    state->perf_async_pub_done_recheck_ns[pw] += recheck_done_ns - loop_start_ns;
                }
#endif
                if (head == tail)
                    break;
            }
#if MK_PERF_TRACE_ARGS
            int64_t sleep_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
#endif
            __nanosleep(32);
#if MK_PERF_TRACE_ARGS
            if (lane_id == 0) {
                int64_t sleep_end_ns = globaltimer_ns();
                state->perf_async_pub_poll_ns[pw] += sleep_start_ns - loop_start_ns;
                state->perf_async_pub_wait_ring_ns[pw] += sleep_end_ns - sleep_start_ns;
                last_empty_sleep_end_ns = sleep_end_ns;
            }
#endif
            continue;
        }

        int batch_count = tail - head;
        if (batch_count > PUB_CONSUME_BATCH)
            batch_count = PUB_CONSUME_BATCH;

#if MK_PERF_TRACE_ARGS
        int64_t batch_start_ns = 0;
        int64_t batch_poll_ns = 0;
        int64_t batch_ring_load_ns = 0;
        int64_t batch_work_ns = 0;
        int64_t batch_token_gap_ns = 0;
        int64_t last_helper_done_ns = 0;
        int64_t batch_head_release_ns = 0;
        int64_t batch_syncwarp_ns = 0;
        if (lane_id == 0) {
            batch_start_ns = globaltimer_ns();
            if (last_token_done_ns != 0 && loop_start_ns > last_token_done_ns)
                state->perf_async_pub_gap_ns[pw] += loop_start_ns - last_token_done_ns;
            batch_poll_ns = batch_start_ns - loop_start_ns;
            state->perf_async_pub_poll_ns[pw] += batch_poll_ns;
            state->perf_async_pub_batch_count[pw] += 1;
            if (batch_count > state->perf_async_pub_max_batch[pw])
                state->perf_async_pub_max_batch[pw] = batch_count;
        }
#endif

        for (int i = 0; i < batch_count; ++i) {
#if MK_PERF_TRACE_ARGS
            int64_t token_load_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
            if (lane_id == 0 && last_helper_done_ns != 0 && token_load_start_ns > last_helper_done_ns)
                batch_token_gap_ns += token_load_start_ns - last_helper_done_ns;
#endif
            int recv_token_idx = ld_acquire_global(&state->pub_ring[pw * PUB_RING_DEPTH + ((head + i) % PUB_RING_DEPTH)]);
#if MK_PERF_TRACE_ARGS
            int64_t work_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
            if (lane_id == 0) {
                int64_t ring_load_ns = work_start_ns - token_load_start_ns;
                state->perf_async_pub_ring_load_ns[pw] += ring_load_ns;
                batch_ring_load_ns += ring_load_ns;
            }
            int64_t scan_ns = 0, atomic_ns = 0, fence_ns = 0, store_ns = 0;
#endif
            int num_hits = publish_recv_token_from_pending(state, recv_token_idx, local_expert_begin, num_topk, lane_id
#if MK_PERF_TRACE_ARGS
                                                           , &scan_ns, &atomic_ns, &fence_ns, &store_ns
#endif
            );
            if (lane_id == 0) {
#if MK_PERF_TRACE_ARGS
                int64_t helper_done_ns = globaltimer_ns();
                int64_t work_ns = helper_done_ns - work_start_ns;
                int64_t helper_accounted_ns = scan_ns + atomic_ns + fence_ns + store_ns;
                int64_t helper_unattributed_ns = work_ns > helper_accounted_ns ? work_ns - helper_accounted_ns : 0;
                state->perf_async_pub_work_ns[pw] += work_ns;
                state->perf_async_pub_helper_unattributed_ns[pw] += helper_unattributed_ns;
                batch_work_ns += work_ns;
                last_helper_done_ns = helper_done_ns;
                state->perf_async_pub_scan_ns[pw] += scan_ns;
                state->perf_async_pub_atomic_ns[pw] += atomic_ns;
                state->perf_async_pub_fence_ns[pw] += fence_ns;
                state->perf_async_pub_store_ns[pw] += store_ns;
                state->perf_async_pub_tokens[pw] += 1;
                state->perf_async_pub_local_hit_tokens[pw] += (num_hits > 0 ? 1 : 0);
                state->perf_async_pub_local_hits[pw] += num_hits;
#endif
            }
        }

        head += batch_count;
        if (lane_id == 0) {
#if MK_PERF_TRACE_ARGS
            last_token_done_ns = globaltimer_ns();
#endif
            st_na_release(&state->pub_ring_head[pw], head);
#if MK_PERF_TRACE_ARGS
            int64_t release_done_ns = globaltimer_ns();
            int64_t head_release_ns = release_done_ns - last_token_done_ns;
            state->perf_async_pub_head_release_ns[pw] += head_release_ns;
            batch_head_release_ns += head_release_ns;
            state->perf_async_pub_head_release_count[pw] += 1;
            last_token_done_ns = release_done_ns;
#endif
        }
#if MK_PERF_TRACE_ARGS
        int64_t sync_start_ns = (lane_id == 0) ? globaltimer_ns() : 0;
#endif
        __syncwarp();
#if MK_PERF_TRACE_ARGS
        if (lane_id == 0) {
            int64_t sync_done_ns = globaltimer_ns();
            int64_t syncwarp_ns = sync_done_ns - sync_start_ns;
            state->perf_async_pub_syncwarp_ns[pw] += syncwarp_ns;
            batch_syncwarp_ns += syncwarp_ns;
            int64_t batch_wall_ns = sync_done_ns - batch_start_ns;
            int64_t batch_accounted_ns = batch_poll_ns + batch_ring_load_ns + batch_work_ns +
                batch_token_gap_ns + batch_head_release_ns + batch_syncwarp_ns;
            state->perf_async_pub_token_gap_ns[pw] += batch_token_gap_ns;
            state->perf_async_pub_batch_wall_ns[pw] += batch_wall_ns;
            state->perf_async_pub_batch_accounted_ns[pw] += batch_accounted_ns;
            if (batch_wall_ns > batch_accounted_ns)
                state->perf_async_pub_batch_unattributed_ns[pw] += batch_wall_ns - batch_accounted_ns;
            last_token_done_ns = sync_done_ns;
        }
#endif
    }

    if (lane_id == 0) {
#if MK_PERF_TRACE_ARGS
        int64_t finish_start_ns = globaltimer_ns();
#endif
        __threadfence();
        st_na_release(&state->publish_warp_done[pw], 1);
        int done = atomicAdd(state->publish_done_count, 1) + 1;
        if (done == state->num_pub_warps_total) {
            __threadfence();
#if MK_PERF_TRACE_ENABLED
            int64_t all_done_ts = globaltimer_ns();
            *state->perf_async_publish_all_done_ts = all_done_ts;
#endif
            st_na_release(state->publish_all_done, 1);
        }
#if MK_PERF_TRACE_ENABLED
        int64_t end_ns = globaltimer_ns();
        state->perf_async_pub_end_ts[pw] = end_ns;
#endif
#if MK_PERF_TRACE_ARGS
        state->perf_async_pub_finish_ns[pw] = end_ns - finish_start_ns;
        if (recv_done_observe_ns != 0 && end_ns > recv_done_observe_ns)
            state->perf_async_pub_drain_ns[pw] = end_ns - recv_done_observe_ns;
#endif
    }
}
#endif

__device__ __forceinline__ void compute_group_sync(MegaKernelState* state, int group_id, int group_size) {
    EP_DEVICE_ASSERT(group_size > 0 && group_size <= COMPUTE_GROUP_SIZE);
    EP_DEVICE_ASSERT(group_id >= 0 && group_id < state->num_compute_groups);
    __syncthreads();
    memory_fence_gpu();
    if (threadIdx.x == 0) {
        int phase = ld_acquire_global(&state->compute_group_phase[group_id]);
        int arrived = atomicAdd(&state->compute_group_barrier[group_id], 1) + 1;
        if (arrived == group_size) {
            st_release_gpu_global(&state->compute_group_barrier[group_id], 0);
            st_release_gpu_global(&state->compute_group_phase[group_id], phase + 1);
        } else {
            while (ld_acquire_global(&state->compute_group_phase[group_id]) == phase)
                __nanosleep(64);
        }
    }
    __syncthreads();
}

template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_worker(
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    MegaKernelState* state,
    uint8_t* smem_buffer
);

template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    uint8_t* smem_buffer);

template <ComputeDType kComputeDType>
__device__ __forceinline__ void combine_precompute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int combine_sm_idx,
    int num_combine_sms,
    uint8_t* smem_buffer);

#define MK_FORWARD_COMPUTE_WORKER(COMPUTE_DTYPE, SM_ID, COMPUTE_SM_IDX, NUM_COMPUTE_SMS, STATE, SMEM_BUFFER) \
    compute_worker<COMPUTE_DTYPE>((SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (STATE), (SMEM_BUFFER))

#define MK_BACKWARD_COMPUTE_WORKER(COMPUTE_DTYPE, BS, SM_ID, COMPUTE_SM_IDX, NUM_COMPUTE_SMS, SMEM_BUFFER) \
    compute_backward_worker<COMPUTE_DTYPE>((BS), (SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (SMEM_BUFFER))

#define MK_DISPATCH_REUSED_COMPUTE(IS_BACKWARD, COMPUTE_DTYPE, BS, SM_ID, COMPUTE_SM_IDX, NUM_COMPUTE_SMS, STATE, SMEM_BUFFER) \
    if constexpr (IS_BACKWARD) { \
        EP_DEVICE_ASSERT((BS) != nullptr); \
        MK_BACKWARD_COMPUTE_WORKER(COMPUTE_DTYPE, (BS), (SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (SMEM_BUFFER)); \
    } else { \
        MK_FORWARD_COMPUTE_WORKER(COMPUTE_DTYPE, (SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (STATE), (SMEM_BUFFER)); \
    }

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType, bool kDispatchBackwardCompute = false>
__device__ void dispatch_worker_v2(
    int sm_id,
    int dispatch_sm_idx,  // 0-based index among all dispatch SMs
    MegaKernelState* state,
    MegaKernelBackwardState* backward_state = nullptr
) {
    using namespace internode;
    constexpr int kNumTopkRDMARanks = internode::get_num_topk_rdma_ranks(kNumRDMARanks);
    const auto num_sms = state->num_dispatch_sms;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
#ifdef MK_TOKEN_TRACE
    if (thread_id == 0)
        printf("[MK-DISPATCH] worker entry rank=%d cta=%d dispatch_idx=%d stage=%d\n",
               state->rank, static_cast<int>(blockIdx.x), dispatch_sm_idx, kStage);
#endif
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    const auto num_channels = state->num_dispatch_channels, channel_id = sm_id / 2;
    // Logical channels per physical CTA pair = kStage. Must match host num_logical_channels =
    // num_physical_channels * stage and the combine worker's num_logical_channels_per_physical,
    // so the dispatch receiver writes and the combine sender reads the gbl-channel arrays with
    // the same (logical) stride. Restores logical!=physical support.
    constexpr int num_logical_channels_per_physical = kStage;
    const int num_logical_channels = num_channels * num_logical_channels_per_physical;
    const bool is_forwarder = dispatch_sm_idx % 2 == 0;
    const auto rdma_rank = state->rank / NUM_MAX_NVL_PEERS, nvl_rank = state->rank % NUM_MAX_NVL_PEERS;
    const auto num_ranks = state->num_ranks;

    // if (threadIdx.x == 0 && dispatch_sm_idx == 0) {
    //     printf("rank: %d, sm_id: %d, dispatch_sm_idx: %d num_channels: %d num_logical_channels: %d \n", state->rank, sm_id, dispatch_sm_idx, num_channels, num_logical_channels);
    // }

    constexpr int kDispatchWorkerWarps = kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS;
    EP_DEVICE_ASSERT(num_warps >= kDispatchWorkerWarps);
#if MK_ASYNC_PUBLISH
    EP_DEVICE_ASSERT(num_warps >= kDispatchWorkerWarps + NUM_MAX_NVL_PEERS);

    if (!is_forwarder && warp_id >= kDispatchWorkerWarps &&
        warp_id < kDispatchWorkerWarps + NUM_MAX_NVL_PEERS) {
        const int publisher_slot = warp_id - kDispatchWorkerWarps;
        const int paired_receiver_warp = kNumDispatchRDMASenderWarps + 1 + publisher_slot;
        const int src_nvl_rank = (paired_receiver_warp + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS;
        publish_worker_v2(dispatch_sm_idx, src_nvl_rank, state);
    }
#endif
    const bool dispatch_thread_active = warp_id < kDispatchWorkerWarps;
    if (dispatch_thread_active) {

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
    const uint32_t* x_scales = state->x_scales;
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
    auto tma_buffer = smem_buffer + target_rank * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
    uint32_t tma_phase = 0;
#if MK_PERF_TRACE_ARGS
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
#ifdef MK_TOKEN_TRACE
        if (thread_id == 0)
            printf("[MK-DISPATCH] channel begin rank=%d cta=%d channel=%d round=%d forwarder=%d\n",
                   state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage, is_forwarder);
#endif
#if MK_PERF_TRACE_ARGS
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
#if MK_PERF_TRACE_ENABLED
        if (thread_id == 0) {
            int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
            int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
            int64_t now_ns = globaltimer_ns();
            state->perf_dispatch_lch_ts[trace_idx + 0] = now_ns;
#if MK_PERF_TRACE_ARGS
            int round_idx = target_rank / NUM_MAX_NVL_PEERS;
            auto* round_trace = &state->perf_dispatch_round_trace[
                logical_channel_id * kNumRDMARanks + round_idx];
            if (dispatch_lch_role == 0)
                round_trace->sender_work_begin_ns = now_ns;
            else
                round_trace->forwarder_wait_begin_ns = now_ns;
#endif
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
#if MK_PERF_TRACE_ARGS
            int sender_meta_idx = dispatch_acc_idx_for_perf * kNumRDMARanks + dst_rdma_rank;
            int64_t sender_meta_build_start_ts = (lane_id == 0) ? globaltimer_ns() : 0;
#endif
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
#if MK_PERF_TRACE_ARGS
            if (lane_id == 0) {
                state->perf_disp_sender_meta_build_start_ts[sender_meta_idx] = sender_meta_build_start_ts;
                state->perf_disp_sender_meta_ready_ts[sender_meta_idx] = globaltimer_ns();
                state->perf_disp_sender_meta_raw_2[sender_meta_idx] = dst_ptr[NUM_MAX_NVL_PEERS * 2];
                state->perf_disp_sender_meta_raw_3[sender_meta_idx] = dst_ptr[NUM_MAX_NVL_PEERS * 2 + 1];
            }
            __syncwarp();
            if (lane_id == 0)
                state->perf_disp_sender_meta_put_begin_ts[sender_meta_idx] = globaltimer_ns();
#endif

            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
            }
#if MK_PERF_TRACE_ARGS
            __syncwarp();
            if (lane_id == 0)
                state->perf_disp_sender_meta_put_end_ts[sender_meta_idx] = globaltimer_ns();
#endif
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
#if MK_PERF_TRACE_ARGS
            if (is_token_in_rank_uint64 != 0 && lane_id < kNumRDMARanks) {
                auto* lane_round_trace = &state->perf_dispatch_round_trace[
                    logical_channel_id * kNumRDMARanks + lane_id];
                atomicAdd(reinterpret_cast<unsigned long long*>(&lane_round_trace->actual_token_count), 1ULL);
                atomicAdd(reinterpret_cast<unsigned long long*>(&lane_round_trace->rdma_packet_slot_count), 1ULL);
            }
#endif

            // Wait buffer release
            auto start_time = clock64();
#if MK_PERF_TRACE_ARGS
            int64_t head_wait_begin_ns = globaltimer_ns();
            int64_t head_poll_count = 0;
#endif
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
#if MK_PERF_TRACE_ARGS
                ++head_poll_count;
#endif
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    __threadfence_system(); trap();
                }
            }
#if MK_PERF_TRACE_ARGS
            if (lane_id == 0 && head_poll_count != 0) {
                int round_idx = target_rank / NUM_MAX_NVL_PEERS;
                auto* round_trace = &state->perf_dispatch_round_trace[
                    logical_channel_id * kNumRDMARanks + round_idx];
                round_trace->polling_iteration_count += head_poll_count;
                round_trace->tail_head_wait_ns += globaltimer_ns() - head_wait_begin_ns;
            }
#endif
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

            // Copy packed UE8M0 x scales. Layout stays 4 bytes per scale pack.
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                auto value = ld_nc_global(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<uint32_t*>(dst_send_buffers[j]) + i, value);
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<uint32_t*>(dst_send_buffers[i]) + num_scales;

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
                __threadfence_system(); trap();
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
#if MK_PERF_TRACE_ARGS
        int64_t fwd_prefix_meta_wait_start_ns = (lane_id < kNumRDMARanks) ? globaltimer_ns() : 0;
        int64_t fwd_prefix_meta_first_negative_ts[4] = {0, 0, 0, 0};
        int64_t fwd_prefix_meta_mixed_poll_count = 0;
        int64_t fwd_prefix_meta_poll_count = 0;
#endif
        if (lane_id < kNumRDMARanks) {
            while (true) {
#if MK_PERF_TRACE_ARGS
                ++fwd_prefix_meta_poll_count;
#endif
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
#if MK_PERF_TRACE_ARGS
                int meta_ready_mask = (meta_0 < 0 ? 1 : 0) | (meta_1 < 0 ? 2 : 0) |
                                      (meta_2 < 0 ? 4 : 0) | (meta_3 < 0 ? 8 : 0);
                if (meta_ready_mask != 0) {
                    int64_t meta_observe_ts = globaltimer_ns();
                    if ((meta_ready_mask & 1) && fwd_prefix_meta_first_negative_ts[0] == 0)
                        fwd_prefix_meta_first_negative_ts[0] = meta_observe_ts;
                    if ((meta_ready_mask & 2) && fwd_prefix_meta_first_negative_ts[1] == 0)
                        fwd_prefix_meta_first_negative_ts[1] = meta_observe_ts;
                    if ((meta_ready_mask & 4) && fwd_prefix_meta_first_negative_ts[2] == 0)
                        fwd_prefix_meta_first_negative_ts[2] = meta_observe_ts;
                    if ((meta_ready_mask & 8) && fwd_prefix_meta_first_negative_ts[3] == 0)
                        fwd_prefix_meta_first_negative_ts[3] = meta_observe_ts;
                    if (meta_ready_mask != 0xf)
                        ++fwd_prefix_meta_mixed_poll_count;
                }
#endif
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
#if MK_PERF_TRACE_ARGS
                    int prefix_prod_idx = (dispatch_acc_idx_for_perf * NUM_MAX_NVL_PEERS + dst_nvl_rank) * kNumRDMARanks + lane_id;
                    int64_t prefix_meta_ready_ts = globaltimer_ns();
                    int64_t prefix_store_begin_ts = globaltimer_ns();
#endif
                    // The receiver treats end as the publication flag for the prefix pair.
                    // Publish start first, then release end at system scope so a matching
                    // acquire observes both values without relying on a later unrelated fence.
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
#if MK_PERF_TRACE_ARGS
                    int64_t prefix_store_start_done_ts = globaltimer_ns();
#endif
                    st_release_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
#if MK_PERF_TRACE_ARGS
                    int64_t prefix_publish_ts = globaltimer_ns();
                    state->perf_disp_prefix_meta_wait_start_ts[prefix_prod_idx] = fwd_prefix_meta_wait_start_ns;
                    state->perf_disp_prefix_meta_ready_ts[prefix_prod_idx] = prefix_meta_ready_ts;
                    #pragma unroll
                    for (int meta_word = 0; meta_word < 4; ++meta_word)
                        state->perf_disp_prefix_meta_first_negative_ts[prefix_prod_idx * 4 + meta_word] =
                            fwd_prefix_meta_first_negative_ts[meta_word];
                    state->perf_disp_prefix_meta_mixed_poll_count[prefix_prod_idx] = fwd_prefix_meta_mixed_poll_count;
                    state->perf_disp_prefix_meta_poll_count[prefix_prod_idx] = fwd_prefix_meta_poll_count;
                    state->perf_disp_prefix_meta_raw_0[prefix_prod_idx] = meta_0;
                    state->perf_disp_prefix_meta_raw_1[prefix_prod_idx] = meta_1;
                    state->perf_disp_prefix_meta_raw_2[prefix_prod_idx] = meta_2;
                    state->perf_disp_prefix_meta_raw_3[prefix_prod_idx] = meta_3;
                    state->perf_disp_prefix_store_start_done_ts[prefix_prod_idx] = prefix_store_start_done_ts;
                    state->perf_disp_prefix_store_begin_ts[prefix_prod_idx] = prefix_store_begin_ts;
                    state->perf_disp_prefix_publish_ts[prefix_prod_idx] = prefix_publish_ts;
                    state->perf_disp_prefix_meta_wait_ns[prefix_prod_idx] += prefix_meta_ready_ts - fwd_prefix_meta_wait_start_ns;
                    state->perf_disp_prefix_raw_start[prefix_prod_idx] = -start_sum - 1;
                    state->perf_disp_prefix_raw_end[prefix_prod_idx] = -end_sum - 1;
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
#if MK_PERF_TRACE_ARGS
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
#ifdef MK_TOKEN_TRACE
                    // Producer side: what this forwarder announces to the NVL receiver
                    // (nvl_prefix_cnt = end_sum-start_sum, per (src_rdma=lane, dst_nvl)) vs what it
                    // will actually forward from RDMA (num_tokens_to_recv_from_rdma, per src_rdma).
                    // If they disagree the receiver waits forever -> dispatch NVL data timeout.
                    printf("[MK-DISPATCH][XCHECK-FWD-META] rank=%d cta=%d channel=%d logical_ch=%d dst_nvl=%d src_rdma=%d meta=(%d,%d,%d,%d) start_sum=%d end_sum=%d nvl_prefix_cnt=%d num_tokens_to_recv_from_rdma=%d\n",
                           state->rank, static_cast<int>(blockIdx.x), channel_id, logical_channel_id,
                           dst_nvl_rank, lane_id, meta_0, meta_1, meta_2, meta_3,
                           start_sum, end_sum, end_sum - start_sum, num_tokens_to_recv_from_rdma);
#endif
                    break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch forwarder timeout (RDMA meta), rank=%d block=%d thread=%d channel=%d logical_ch=%d RDMA=%d nvl=%d src RDMA lane=%d dst NVL=%d meta=(%d,%d,%d,%d)\n",
                           state->rank, static_cast<int>(blockIdx.x), thread_id, channel_id, logical_channel_id,
                           rdma_rank, nvl_rank, lane_id, dst_nvl_rank,
                           meta_0, meta_1, meta_2, meta_3);
                    __threadfence_system(); trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head inside this logical channel's independent NVL head namespace.
        int* send_nvl_head = send_nvl_head_base + logical_channel_id * state->combine_nvl_head_stride +
            src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
#ifdef MK_TOKEN_TRACE
        if (thread_id == 0)
            printf("[MK-DISPATCH] channel barrier before rank=%d cta=%d channel=%d round=%d\n",
                   state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage);
#endif
        sync_forwarder_smem();
#ifdef MK_TOKEN_TRACE
        if (thread_id == 0)
            printf("[MK-DISPATCH] channel barrier after rank=%d cta=%d channel=%d round=%d\n",
                   state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage);
#endif

#if MK_PERF_TRACE_ENABLED
        // [EXPERIMENT 3b] Batch-flush the prefixes published by this forwarder's warps
        // ONCE (system scope) before starting bulk token forwarding. v320 showed the
        // relaxed st.sys prefix writes are not visible to peers until the producer hits
        // its next system fence (the round barrier, ~1.5ms later), so consumers spin
        // for milliseconds. A single per-physical-channel fence here (24x for stage=1,
        // not 384x like the per-store fence in v315) should make prefixes visible to
        // peers at ~publish time. Not a named barrier, so no deadlock risk.
        // __threadfence_system();
#endif

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
#if MK_PERF_TRACE_ARGS
            int64_t nvl_head_wait_start_ns = globaltimer_ns();
            int64_t nvl_head_poll_count = 0;
#endif
            while (true) {
#if MK_PERF_TRACE_ARGS
                ++nvl_head_poll_count;
#endif
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;
                cached_nvl_channel_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank,
                           ld_volatile_global(nvl_channel_head.buffer()), cached_nvl_channel_tail);
                    __threadfence_system(); trap();
                }
            }
#if MK_PERF_TRACE_ARGS
            if (elect_one_sync() && nvl_head_poll_count > 1) {
                auto* round_trace = &state->perf_dispatch_round_trace[
                    logical_channel_id * kNumRDMARanks + src_rdma_rank];
                round_trace->polling_iteration_count += nvl_head_poll_count - 1;
                round_trace->tail_head_wait_ns += globaltimer_ns() - nvl_head_wait_start_ns;
            }
#endif

            // Find next source RDMA rank
            start_time = clock64();
#if MK_PERF_TRACE_ARGS
            int64_t ready_wait_start_ns = globaltimer_ns();
            int64_t ready_poll_count = 0;
#endif
            while (true) {
#if MK_PERF_TRACE_ARGS
                ++ready_poll_count;
#endif
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
                    __threadfence_system(); trap();
                }
            }
#if MK_PERF_TRACE_ARGS
            if (elect_one_sync() && ready_poll_count > 1) {
                auto* round_trace = &state->perf_dispatch_round_trace[
                    logical_channel_id * kNumRDMARanks + src_rdma_rank];
                round_trace->polling_iteration_count += ready_poll_count - 1;
                round_trace->ready_wait_ns += globaltimer_ns() - ready_wait_start_ns;
            }
#endif
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
#if MK_PERF_TRACE_ARGS
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
#if MK_PERF_TRACE_ARGS
        int64_t recv_prefix_lane_start_ns = (lane_id < kNumRDMARanks) ? globaltimer_ns() : 0;
        int64_t recv_prefix_lane_observe_ts = 0;
        int64_t recv_prefix_lane_start_first_negative_ts = 0;
        int64_t recv_prefix_lane_end_first_negative_ts = 0;
        int64_t recv_prefix_lane_pair_ready_ts = 0;
        int64_t recv_prefix_lane_poll_count = 0;
        int64_t recv_prefix_lane_mixed_poll_count = 0;
        int64_t recv_prefix_lane_wait_ns = 0;
        int recv_prefix_lane_src_rdma = -1;
        int recv_prefix_lane_raw_start = 0;
        int recv_prefix_lane_raw_end = 0;
#endif
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            end_offset = ld_acquire_sys_global(nvl_channel_prefix_end.buffer() + lane_id);
            start_offset = end_offset < 0
                ? ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id)
                : 0;
#if MK_PERF_TRACE_ARGS
            bool recv_prefix_lane_start_negative = start_offset < 0;
            bool recv_prefix_lane_end_negative = end_offset < 0;
            ++recv_prefix_lane_poll_count;
            int64_t recv_prefix_lane_poll_ts = 0;
            if (recv_prefix_lane_start_negative || recv_prefix_lane_end_negative)
                recv_prefix_lane_poll_ts = globaltimer_ns();
            if (recv_prefix_lane_start_negative && recv_prefix_lane_start_first_negative_ts == 0)
                recv_prefix_lane_start_first_negative_ts = recv_prefix_lane_poll_ts;
            if (recv_prefix_lane_end_negative && recv_prefix_lane_end_first_negative_ts == 0)
                recv_prefix_lane_end_first_negative_ts = recv_prefix_lane_poll_ts;
            if (recv_prefix_lane_start_negative != recv_prefix_lane_end_negative)
                ++recv_prefix_lane_mixed_poll_count;
#endif
            if (start_offset < 0 and end_offset < 0) {
#if MK_PERF_TRACE_ARGS
                recv_prefix_lane_pair_ready_ts = recv_prefix_lane_poll_ts;
                recv_prefix_lane_observe_ts = recv_prefix_lane_poll_ts;
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
                __threadfence_system(); trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);
#ifdef MK_TOKEN_TRACE
        // Consumer side: how many NVL tokens this receiver expects on (src_nvl, channel).
        // Per-lane lane_cnt (end-start) must match the producer's nvl_prefix_cnt above for the
        // same (dst_nvl==this rank's src_nvl, src_rdma==lane). total is what receiver will wait for.
        if (lane_id < kNumRDMARanks)
            printf("[MK-DISPATCH][XCHECK-RECV-CNT] rank=%d cta=%d channel=%d logical_ch=%d src_nvl=%d src_rdma=%d start=%d end=%d lane_cnt=%d total_num_tokens_to_recv=%d\n",
                   state->rank, static_cast<int>(blockIdx.x), channel_id, logical_channel_id,
                   src_nvl_rank, lane_id, start_offset, end_offset, end_offset - start_offset, num_tokens_to_recv);
#endif
#if MK_PERF_TRACE_ARGS
        int64_t prefix_slowest_wait_ns = recv_prefix_lane_wait_ns;
        int64_t prefix_slowest_start_ts = recv_prefix_lane_start_ns;
        int64_t prefix_slowest_observe_ts = recv_prefix_lane_observe_ts;
        int64_t prefix_slowest_start_first_negative_ts = recv_prefix_lane_start_first_negative_ts;
        int64_t prefix_slowest_end_first_negative_ts = recv_prefix_lane_end_first_negative_ts;
        int64_t prefix_slowest_pair_ready_ts = recv_prefix_lane_pair_ready_ts;
        int64_t prefix_slowest_poll_count = recv_prefix_lane_poll_count;
        int64_t prefix_slowest_mixed_poll_count = recv_prefix_lane_mixed_poll_count;
        int prefix_slowest_src_rdma = recv_prefix_lane_src_rdma;
        int prefix_slowest_raw_start = recv_prefix_lane_raw_start;
        int prefix_slowest_raw_end = recv_prefix_lane_raw_end;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            int64_t other_wait = __shfl_down_sync(0xffffffff, prefix_slowest_wait_ns, offset);
            int64_t other_start_ts = __shfl_down_sync(0xffffffff, prefix_slowest_start_ts, offset);
            int64_t other_observe_ts = __shfl_down_sync(0xffffffff, prefix_slowest_observe_ts, offset);
            int64_t other_start_first_negative_ts = __shfl_down_sync(0xffffffff, prefix_slowest_start_first_negative_ts, offset);
            int64_t other_end_first_negative_ts = __shfl_down_sync(0xffffffff, prefix_slowest_end_first_negative_ts, offset);
            int64_t other_pair_ready_ts = __shfl_down_sync(0xffffffff, prefix_slowest_pair_ready_ts, offset);
            int64_t other_poll_count = __shfl_down_sync(0xffffffff, prefix_slowest_poll_count, offset);
            int64_t other_mixed_poll_count = __shfl_down_sync(0xffffffff, prefix_slowest_mixed_poll_count, offset);
            int other_src = __shfl_down_sync(0xffffffff, prefix_slowest_src_rdma, offset);
            int other_raw_start = __shfl_down_sync(0xffffffff, prefix_slowest_raw_start, offset);
            int other_raw_end = __shfl_down_sync(0xffffffff, prefix_slowest_raw_end, offset);
            if (other_wait > prefix_slowest_wait_ns) {
                prefix_slowest_wait_ns = other_wait;
                prefix_slowest_start_ts = other_start_ts;
                prefix_slowest_observe_ts = other_observe_ts;
                prefix_slowest_start_first_negative_ts = other_start_first_negative_ts;
                prefix_slowest_end_first_negative_ts = other_end_first_negative_ts;
                prefix_slowest_pair_ready_ts = other_pair_ready_ts;
                prefix_slowest_poll_count = other_poll_count;
                prefix_slowest_mixed_poll_count = other_mixed_poll_count;
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
            state->perf_disp_allrecv_prefix_start_first_negative_ts[recv_perf_idx_for_role] = prefix_slowest_start_first_negative_ts;
            state->perf_disp_allrecv_prefix_end_first_negative_ts[recv_perf_idx_for_role] = prefix_slowest_end_first_negative_ts;
            state->perf_disp_allrecv_prefix_pair_ready_ts[recv_perf_idx_for_role] = prefix_slowest_pair_ready_ts;
            state->perf_disp_allrecv_prefix_poll_count[recv_perf_idx_for_role] = prefix_slowest_poll_count;
            state->perf_disp_allrecv_prefix_mixed_poll_count[recv_perf_idx_for_role] = prefix_slowest_mixed_poll_count;
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
#if MK_ASYNC_PUBLISH
        const int pub_warp_idx = get_publish_warp_index(dispatch_sm_idx, src_nvl_rank);
        int producer_tail = 0;
        int producer_batch_count = 0;
        if (lane_id == 0)
            producer_tail = ld_acquire_global(&state->pub_ring_tail[pub_warp_idx]);
        producer_tail = __shfl_sync(0xffffffff, producer_tail, 0);
#endif
        while (num_tokens_to_recv > 0) {
            // Wait for data
            start_time = clock64();
#if MK_PERF_TRACE_ARGS
            int64_t wait_nvl_start = (lane_id == 0) ? globaltimer_ns() : 0;
#endif
            while (true) {
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                cached_channel_tail_idx = __shfl_sync(0xffffffff, ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch NVL receiver timeout (data), channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank, cached_channel_head_idx, cached_channel_tail_idx);
                    __threadfence_system(); trap();
                }
            }
#if MK_PERF_TRACE_ARGS
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
#if MK_ASYNC_PUBLISH
                for (int topk_slot = lane_id; topk_slot < num_topk; topk_slot += 32) {
                    state->pending_topk_idx[recv_token_idx * num_topk + topk_slot] = ld_nc_global(topk_data_ptr + topk_slot);
                    state->pending_topk_weights[recv_token_idx * num_topk + topk_slot] = ld_nc_global(weight_data_ptr + topk_slot);
                }
                if (lane_id == 0)
                    state->pending_meta[recv_token_idx] = meta;
                __syncwarp();
                if (lane_id == 0) {
                    while (producer_tail - ld_acquire_global(&state->pub_ring_head[pub_warp_idx]) >= PUB_RING_DEPTH)
                        __nanosleep(32);
                    st_na_global(&state->pub_ring[pub_warp_idx * PUB_RING_DEPTH + (producer_tail % PUB_RING_DEPTH)], static_cast<int>(recv_token_idx));
                    ++producer_tail;
                    ++producer_batch_count;
                    if (producer_batch_count >= PUB_PRODUCE_BATCH) {
                        __threadfence();
                        st_na_release(&state->pub_ring_tail[pub_warp_idx], producer_tail);
                        producer_batch_count = 0;
                    }
                }
#else
                const int local_expert_end = local_expert_begin + state->num_local_experts;
                if (lane_id == 0) {
#if MK_PERF_TRACE_ARGS
                    int acc_idx = dispatch_acc_idx_for_perf;
                    int recv_perf_idx = acc_idx * NUM_MAX_NVL_PEERS + src_nvl_rank;
                    bool record_perf = (target_rank == 0);  // keep legacy single-receiver aggregate
                    int64_t publish_start = globaltimer_ns();
                    int64_t wait_recvcount_acc = 0;
#endif
#if MK_PERF_TRACE_ARGS
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
#if MK_PERF_TRACE_ARGS
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
                        if (slot >= state->expert_count[local_expert_id]) {
                            printf("MK dispatch expert slot overflow, rank=%d recv_token=%lld expert=%d slot=%d max_tpe=%d\n",
                                   state->rank, (long long)recv_token_idx, expert_id, slot, state->max_tokens_per_expert);
                            __threadfence_system(); trap();
                        }
                        int dest_offset = state->expert_slot_base[local_expert_id] + slot;
                        int* dst_ptr = &state->recv_token_source_info[dest_offset * 2];
                        // Plain stores: ordering vs slot_ready is enforced by the single
                        // __threadfence() below. All readers are this GPU's compute workers,
                        // so device-scope visibility is sufficient.
                        st_na_global(dst_ptr, static_cast<int>(recv_token_idx));
                        st_na_global(dst_ptr + 1, topk_slot);
                        hit_local_expert[num_hits] = local_expert_id;
                        hit_slot[num_hits] = slot;
                        hit_abs_slot[num_hits] = state->expert_slot_base[local_expert_id] + slot;
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
                            __threadfence_system(); trap();
                        }
                        for (int h = 0; h < num_hits; ++h)
                            state->token_slot_list[recv_token_idx * num_topk + hit_base + h] = hit_abs_slot[h];
                        atomicAdd(&state->token_compute_expected[recv_token_idx], num_hits);
                    }
#if MK_PERF_TRACE_ARGS
                    int64_t pub_atomic_ns = globaltimer_ns() - pub_atomic_start;
                    int64_t pub_fence_start = globaltimer_ns();
#endif

                    // Single device-scope fence per token: orders the plain source_info and
                    // token_slot_list stores before the slot_ready releases. The whole signal
                    // chain stays in device scope because no remote GPU reads these buffers.
                    if (num_hits > 0)
                        __threadfence();
#if MK_PERF_TRACE_ARGS
                    int64_t pub_fence_ns = globaltimer_ns() - pub_fence_start;
                    int64_t pub_store_start = globaltimer_ns();
#endif

                    // Pass 2: mark each slot ready (unordered, no spin-wait). Scheduler
                    // scans the bitmap and advances expert_recv_count.
                    for (int h = 0; h < num_hits; ++h) {
                        int local_expert_id = hit_local_expert[h];
                        int slot = hit_slot[h];
                        st_na_release(&state->expert_slot_ready[state->expert_slot_base[local_expert_id] + slot], 1);
                    }
#if MK_PERF_TRACE_ARGS
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
#endif
                __syncwarp();

                // Wait TMA to be finished
                tma_store_wait<0>();
                __syncwarp();

            }

            // Move queue
            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }

#if MK_PERF_TRACE_ARGS
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
#if MK_ASYNC_PUBLISH
#ifdef MK_TOKEN_TRACE
            printf("[MK-DISPATCH] async publish before rank=%d cta=%d channel=%d round=%d pub_warp=%d count=%d\n",
                   state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage,
                   pub_warp_idx, producer_batch_count);
#endif
            if (producer_batch_count > 0) {
                __threadfence();
                st_na_release(&state->pub_ring_tail[pub_warp_idx], producer_tail);
#ifdef MK_TOKEN_TRACE
                printf("[MK-DISPATCH] async publish after rank=%d cta=%d channel=%d round=%d pub_warp=%d tail=%llu\n",
                       state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage,
                       pub_warp_idx, static_cast<unsigned long long>(producer_tail));
#endif
#if MK_PERF_TRACE_ARGS
                int round_idx = target_rank / NUM_MAX_NVL_PEERS;
                state->perf_dispatch_round_trace[
                    logical_channel_id * kNumRDMARanks + round_idx].async_publish_ns = globaltimer_ns();
#endif
                producer_batch_count = 0;
            }
            __threadfence();
            // A publisher warp is paired with a physical receiver warp and drains all logical stages
            // for that receiver. Signal done only after the final logical stage; otherwise stage > 1
            // can make the publisher exit before later logical channels enqueue their tokens.
            if (logical_stage == num_logical_channels_per_physical - 1)
                st_na_release(&state->recv_warp_done[pub_warp_idx], 1);
#endif
            __threadfence_system();  // Ensure prefix/count writes visible before signaling
            atomicAdd(&state->channel_dispatch_done[logical_channel_id], 1);
            int done_count = atomicAdd(state->dispatch_done_count, 1) + 1;
            if (done_count == state->expected_dispatch_done_count)
                st_na_release(state->dispatch_done, 1);
        }
    }

#if MK_PERF_TRACE_ARGS
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
#if MK_PERF_TRACE_ARGS
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
        int64_t cta_release_ns = globaltimer_ns();
        state->perf_disp_cta_barrier_ns[acc_idx] += cta_release_ns - dispatch_cta_barrier_start_ns;
        state->perf_disp_cta_release_ts[acc_idx] = cta_release_ns;
    }
#endif
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
#ifdef MK_TOKEN_TRACE
    if (thread_id == 0) {
        printf("[MK-DISPATCH] channel barrier after rank=%d cta=%d channel=%d round=%d\n",
               state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage);
        printf("[MK-DISPATCH] round barrier before rank=%d cta=%d channel=%d round=%d\n",
               state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage);
    }
#endif
#if MK_PERF_TRACE_ENABLED
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        state->perf_dispatch_lch_ts[trace_idx + 1] = globaltimer_ns();
    }
#endif
#ifdef MK_TOKEN_TRACE
    if (thread_id == 0) {
        printf("[MK-DISPATCH] prefix done rank=%d cta=%d channel=%d round=%d role=%s\n",
               state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage,
               is_forwarder ? "forwarder" : "sender");
        printf("[MK-DISPATCH] %s work done rank=%d cta=%d channel=%d round=%d\n",
               is_forwarder ? "forwarder" : "sender", state->rank, static_cast<int>(blockIdx.x),
               logical_channel_id, logical_stage);
    }
    if (thread_id == 0)
        printf("[MK-DISPATCH] channel barrier before rank=%d cta=%d channel=%d round=%d\n",
               state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage);
#endif
    if (thread_id == 0)
        atomicAdd(&state->dispatch_channel_barrier[logical_channel_id], 1);
    if (thread_id == 0) {
        auto start_time = clock64();
#if MK_PERF_TRACE_ARGS
        int64_t barrier_start_ns = globaltimer_ns();
#endif
        uint64_t wait_polls = 0;
        while (ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]) < 2) {
#ifdef MK_TOKEN_TRACE
            ++wait_polls;
            if ((wait_polls & ((1ull << 28) - 1)) == 0)
                printf("[MK-DISPATCH][WAIT] rank=%d cta=%d channel=%d round=%d role=%s ptr=%p value=%d expected=2 polls=%llu\n",
                       state->rank, static_cast<int>(blockIdx.x), logical_channel_id, logical_stage,
                       is_forwarder ? "forwarder" : "sender",
                       state->dispatch_channel_barrier + logical_channel_id,
                       ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]),
                       static_cast<unsigned long long>(wait_polls));
#endif
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch logical-channel barrier timeout, physical_ch=%d logical_ch=%d count=%d\n",
                       channel_id, logical_channel_id, ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]));
                __threadfence_system(); trap();
            }
            __nanosleep(32);
        }
#if MK_PERF_TRACE_ARGS
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
        state->perf_disp_channel_barrier_ns[acc_idx] += globaltimer_ns() - barrier_start_ns;
#endif
    }
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
#if MK_PERF_TRACE_ENABLED
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        state->perf_dispatch_lch_ts[trace_idx + 2] = globaltimer_ns();
    }
#endif

    if (thread_id == 0 && dispatch_sm_idx % 2 == 0) {
        const int round_idx = logical_stage;
        __threadfence_system();
        atomicAdd(&state->dispatch_round_barrier[round_idx], 1);
    }
    if (thread_id == 0) {
        const int round_idx = logical_stage;
        auto start_time = clock64();
#if MK_PERF_TRACE_ARGS
        int64_t barrier_start_ns = globaltimer_ns();
#endif
        uint64_t wait_polls = 0;
        while (ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]) < num_channels) {
#ifdef MK_TOKEN_TRACE
            ++wait_polls;
            if ((wait_polls & ((1ull << 28) - 1)) == 0)
                printf("[MK-DISPATCH][WAIT] rank=%d cta=%d channel=%d round=%d role=%s ptr=%p value=%d expected=%d polls=%llu\n",
                       state->rank, static_cast<int>(blockIdx.x), logical_channel_id, round_idx,
                       is_forwarder ? "forwarder" : "sender", state->dispatch_round_barrier + round_idx,
                       ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]), num_channels,
                       static_cast<unsigned long long>(wait_polls));
#endif
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch round barrier timeout, physical_ch=%d logical_ch=%d round=%d count=%d need=%d\n",
                       channel_id, logical_channel_id, round_idx,
                       ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]), num_channels);
                __threadfence_system(); trap();
            }
            __nanosleep(32);
        }
#if MK_PERF_TRACE_ARGS
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int acc_idx = logical_channel_id * 2 + dispatch_lch_role;
        state->perf_disp_round_barrier_ns[acc_idx] += globaltimer_ns() - barrier_start_ns;
#endif
    }
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
#if MK_PERF_TRACE_ENABLED
    if (thread_id == 0) {
        int dispatch_lch_role = (dispatch_sm_idx % 2 == 0) ? 1 : 0;
        int trace_idx = (logical_channel_id * 2 + dispatch_lch_role) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
        int64_t now_ns = globaltimer_ns();
        state->perf_dispatch_lch_ts[trace_idx + 3] = now_ns;
#if MK_PERF_TRACE_ARGS
        int round_idx = target_rank / NUM_MAX_NVL_PEERS;
        auto* round_trace = &state->perf_dispatch_round_trace[
            logical_channel_id * kNumRDMARanks + round_idx];
        round_trace->channel_barrier_arrival_ns = dispatch_cta_barrier_start_ns;
        round_trace->round_barrier_arrival_ns = now_ns;
        if (dispatch_lch_role == 0)
            round_trace->sender_work_end_ns = dispatch_cta_barrier_start_ns;
        else {
            round_trace->forwarder_wait_end_ns = dispatch_cta_barrier_start_ns;
            round_trace->forwarder_work_begin_ns = round_trace->forwarder_wait_end_ns;
            round_trace->forwarder_work_end_ns = now_ns;
        }
#endif
    }
#endif

    }

    }

#ifdef MK_TOKEN_TRACE
    if (thread_id == 0)
        printf("[MK-DISPATCH] worker exit rank=%d cta=%d dispatch_idx=%d\n",
               state->rank, static_cast<int>(blockIdx.x), dispatch_sm_idx);
#endif

    // Extra dispatch CTA warps can reach this point while the active dispatch warps
    // are still using named barriers 0/1/2. Use a separate barrier id for the
    // whole-CTA handoff into compute so the two protocols never share a live barrier.
    asm volatile("barrier.sync 15, %0;" :: "r"(num_threads));

#ifdef MK_TOKEN_TRACE
    if (thread_id == 0)
        printf("[MK-DISPATCH] worker really exit rank=%d cta=%d dispatch_idx=%d\n",
               state->rank, static_cast<int>(blockIdx.x), dispatch_sm_idx);
#endif

    const int reused_compute_sm_idx = state->num_compute_sms + dispatch_sm_idx;
    const int total_compute_sms_after_dispatch = state->num_compute_sms + state->num_dispatch_sms;
    MK_DISPATCH_REUSED_COMPUTE(
        kDispatchBackwardCompute, kComputeDType, backward_state, sm_id,
        reused_compute_sm_idx, total_compute_sms_after_dispatch, state, smem_buffer);
    return;
}


// ============================================================================
// Compute Scheduler + Worker: scheduler enqueues expert batches, compute groups run GEMM+SwiGLU
// ============================================================================

__device__ __forceinline__ bool timeout_log_once(MegaKernelState* state, int site_id) {
    if (site_id < 0 || site_id >= kTimeoutLogCount)
        return false;
    if (state->timeout_log_counters == nullptr)
        return false;
    int ticket = atomicAdd(&state->timeout_log_counters[site_id], 1);
    return ticket < MK_TIMEOUT_LOG_BUDGET;
}

__device__ __forceinline__ void scheduler_compute_sync(int num_threads) {
    asm volatile("barrier.sync 3, %0;" :: "r"(num_threads));
}

#if MK_PERF_TRACE_ARGS
__device__ __forceinline__ bool mk_perf_record_first_i64(int64_t* slot, int64_t value) {
    return atomicCAS(reinterpret_cast<unsigned long long*>(slot), 0ULL,
                     static_cast<unsigned long long>(value)) == 0ULL;
}
#endif

__device__ __forceinline__ int scheduler_compute_all(int predicate, int* shared_result, int num_threads) {
    if (threadIdx.x == 0)
        *shared_result = 1;
    scheduler_compute_sync(num_threads);
    if (!predicate)
        atomicExch(shared_result, 0);
    scheduler_compute_sync(num_threads);
    int result = *shared_result;
    scheduler_compute_sync(num_threads);
    return result;
}

#if MK_ASYNC_PUBLISH
__device__ __forceinline__ int scheduler_compute_all_until_publish_done(
    int predicate, int* shared_result, int num_threads, MegaKernelState* state, int* dispatch_done) {
    if (threadIdx.x == 0) {
        if (*dispatch_done == 0 && ld_acquire_global(state->publish_all_done) != 0) {
            *dispatch_done = 1;
#if MK_PERF_TRACE_ARGS
            mk_perf_record_first_i64(state->perf_sched_first_done_seen_ts, globaltimer_ns());
#endif
        }
        *shared_result = 1;
    }
    // Reuse scheduler_compute_all's first barrier to broadcast dispatch_done.
    scheduler_compute_sync(num_threads);
    if (*dispatch_done != 0) {
        if (threadIdx.x == 0)
            *shared_result = 0;
    } else if (!predicate) {
        atomicExch(shared_result, 0);
    }
    scheduler_compute_sync(num_threads);
    int result = *shared_result;
    scheduler_compute_sync(num_threads);
    return result;
}
#endif

__device__ __forceinline__ void scheduler_publish_task(MegaKernelState* state, int expert_id, int start_slot, int num_tokens, int is_flush, int source) {
#if MK_PERF_TRACE_ARGS
    int64_t publish_start_ns = globaltimer_ns();
#endif
    int tail = atomicAdd(state->compute_task_reserve_tail, 1);
    if (tail >= state->max_compute_tasks) {
        printf("MK compute task queue overflow, rank=%d tail=%d max=%d\n", state->rank, tail, state->max_compute_tasks);
        __threadfence_system(); trap();
    }
    state->compute_tasks[tail] = ComputeTask{expert_id, start_slot, num_tokens, is_flush};
    __threadfence();
#if MK_PERF_TRACE_ARGS
    int64_t wait_start_ns = globaltimer_ns();
    int visible_tail = ld_acquire_global(state->compute_task_tail);
    int wait_start_visible_tail = visible_tail;
    while (visible_tail != tail) {
        __nanosleep(32);
        visible_tail = ld_acquire_global(state->compute_task_tail);
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
    state->perf_task_source[tail] = source;
#else
    while (ld_acquire_global(state->compute_task_tail) != tail)
        __nanosleep(32);
#endif
    st_na_release(state->compute_task_tail, tail + 1);
#if MK_PERF_TRACE_ARGS
    int64_t publish_visible_ns = globaltimer_ns();
    if (mk_perf_record_first_i64(state->perf_sched_first_task_publish_ts, publish_visible_ns)) {
        st_na_global(state->perf_sched_first_task_source, static_cast<int64_t>(source));
        st_na_global(state->perf_sched_first_task_expert, static_cast<int64_t>(expert_id));
        st_na_global(state->perf_sched_first_task_batch, static_cast<int64_t>(start_slot / COMPUTE_BATCH_SIZE));
        st_na_global(state->perf_sched_first_task_start_slot, static_cast<int64_t>(start_slot));
        st_na_global(state->perf_sched_first_task_num_tokens, static_cast<int64_t>(num_tokens));
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_publish_total_ns),
              static_cast<unsigned long long>(publish_visible_ns - publish_start_ns));
#endif
}

// source: 1=normal scheduler, 2=priority scheduler, 3=tail flush.
__device__ __forceinline__ bool scheduler_try_enqueue_batch(
    MegaKernelState* state, int expert_id, int batch_id, int start_slot, int num_tokens, int is_flush, int source) {
    if (batch_id < 0 || batch_id >= state->max_batches_per_expert) {
        printf("MK scheduler batch id overflow, rank=%d expert=%d batch=%d max=%d\n",
               state->rank, expert_id, batch_id, state->max_batches_per_expert);
        __threadfence_system(); trap();
    }
    int idx = expert_id * state->max_batches_per_expert + batch_id;
#if MK_PERF_TRACE_ARGS
    if (source == 1)
        mk_perf_record_first_i64(state->perf_sched_first_normal_enqueue_attempt_ts, globaltimer_ns());
#endif
    int old_source = atomicCAS(&state->expert_batch_enqueued[idx], 0, source);
#if MK_PERF_TRACE_ARGS
    if (old_source == 2 && source == 1) {
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_normal_after_priority_count), 1ULL);
        int64_t ts = *reinterpret_cast<volatile int64_t*>(&state->expert_batch_enqueue_ts[idx]);
        if (ts > 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_normal_after_priority_ns),
                      static_cast<unsigned long long>(globaltimer_ns() - ts));
        }
    }
#endif
    if (old_source != 0)
        return false;
#if MK_PERF_TRACE_ARGS
    int64_t enqueue_success_ns = globaltimer_ns();
    if (source == 1)
        mk_perf_record_first_i64(state->perf_sched_first_normal_enqueue_success_ts, enqueue_success_ns);
    state->expert_batch_enqueue_ts[idx] = enqueue_success_ns;
#endif
    scheduler_publish_task(state, expert_id, start_slot, num_tokens, is_flush, source);
    return true;
}

__device__ __forceinline__ void scheduler_priority_warp_worker(MegaKernelState* state, int scheduler_id) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    if (scheduler_id != 0)
        return;

    const int max_tpe = state->max_tokens_per_expert;
    const int num_topk = state->num_topk;
#if MK_PERF_TRACE_ARGS
    int64_t sched_priority_acc = 0;
    int64_t priority_scan_tokens = 0;
    int64_t priority_ready_tokens = 0;
    int64_t priority_full_batch_hits = 0;
    int64_t priority_batch_already_enqueued = 0;
    int64_t priority_already_normal = 0;
    int64_t priority_already_priority = 0;
    int64_t priority_already_tail = 0;
    int64_t priority_not_full = 0;
#endif
    int priority_epoch = 1;

    while (ld_acquire_global(state->compute_enqueue_done) == 0) {
#if MK_ASYNC_PUBLISH
        if (ld_acquire_global(state->publish_all_done) != 0)
            break;
#else
        if (ld_acquire_global(state->dispatch_done_count) == state->expected_dispatch_done_count)
            break;
#endif
#if MK_PERF_TRACE_ARGS
        int64_t priority_start = globaltimer_ns();
#endif
        ++priority_epoch;
        if (priority_epoch > (1 << 28))
            priority_epoch = 1;
        int cursor = 0;
        int scan_end = 0;
        if (lane == 0) {
            cursor = ld_acquire_global(state->priority_token_cursor);
            if (cursor < 0 || cursor >= state->max_total_recv_tokens)
                cursor = 0;
            scan_end = min(cursor + MK_PRIORITY_SCAN_WINDOW_TOKENS, state->max_total_recv_tokens);
            if (scan_end <= cursor) {
                cursor = 0;
                scan_end = min(MK_PRIORITY_SCAN_WINDOW_TOKENS, state->max_total_recv_tokens);
            }
        }
        cursor = __shfl_sync(0xffffffff, cursor, 0);
        scan_end = __shfl_sync(0xffffffff, scan_end, 0);

        int new_cursor = cursor;
        int enqueued = 0;
        for (int token = cursor; token < scan_end && enqueued < MK_PRIORITY_MAX_ENQUEUE_PER_LOOP; ++token) {
            int stop_priority = 0;
            if (lane == 0) {
#if MK_ASYNC_PUBLISH
                stop_priority = ld_acquire_global(state->publish_all_done);
#else
                stop_priority =
                    (ld_acquire_global(state->dispatch_done_count) == state->expected_dispatch_done_count) ? 1 : 0;
#endif
            }
            stop_priority = __shfl_sync(0xffffffff, stop_priority, 0);
            if (stop_priority != 0)
                break;
            int nh = 0;
            if (lane == 0)
                nh = ld_acquire_global(&state->token_nhits[token]);
            nh = __shfl_sync(0xffffffff, nh, 0);
#if MK_PERF_TRACE_ARGS
            if (lane == 0) {
                priority_scan_tokens += 1;
                if (nh > 0)
                    priority_ready_tokens += 1;
            }
#endif
            bool token_blocked = false;
            for (int h = 0; h < nh && enqueued < MK_PRIORITY_MAX_ENQUEUE_PER_LOOP; ++h) {
                int slot = -1;
                int expert_id = -1;
                int batch_start = 0;
                int batch_end = 0;
                if (lane == 0) {
                    slot = ld_nc_global(&state->token_slot_list[token * num_topk + h]);
                    if (slot >= 0) {
                        expert_id = mk_slot_to_local_expert(state, slot);
                        int expert_local_slot = slot - state->expert_slot_base[expert_id];
                        int batch_id = expert_local_slot / COMPUTE_BATCH_SIZE;
                        batch_start = batch_id * COMPUTE_BATCH_SIZE;
                        batch_end = batch_start + COMPUTE_BATCH_SIZE;
                        int alloc_count = ld_acquire_global(&state->expert_token_offsets[expert_id]);
                        if (alloc_count < batch_end)
                            token_blocked = true;
                    }
                }
                slot = __shfl_sync(0xffffffff, slot, 0);
                expert_id = __shfl_sync(0xffffffff, expert_id, 0);
                batch_start = __shfl_sync(0xffffffff, batch_start, 0);
                batch_end = __shfl_sync(0xffffffff, batch_end, 0);
                token_blocked = __shfl_sync(0xffffffff, static_cast<int>(token_blocked), 0) != 0;
                if (slot < 0)
                    continue;

                const int batch_id = batch_start / COMPUTE_BATCH_SIZE;
                const int batch_idx = expert_id * state->max_batches_per_expert + batch_id;
                bool skip_batch = false;
                if (lane == 0) {
                    int skip_until = ld_nc_global(&state->priority_batch_skip_epoch[batch_idx]);
                    int retry_after = ld_nc_global(&state->priority_batch_retry_epoch[batch_idx]);
                    if (priority_epoch < skip_until || priority_epoch < retry_after) {
                        skip_batch = true;
                    } else {
                        int source = ld_acquire_global(&state->expert_batch_enqueued[batch_idx]);
                        if (source != 0) {
                            st_na_release(&state->priority_batch_skip_epoch[batch_idx], priority_epoch + MK_PRIORITY_ALREADY_SKIP_EPOCHS);
                            skip_batch = true;
#if MK_PERF_TRACE_ARGS
                            priority_batch_already_enqueued += 1;
                            if (source == 1)
                                priority_already_normal += 1;
                            else if (source == 2)
                                priority_already_priority += 1;
                            else if (source == 3)
                                priority_already_tail += 1;
#endif
                        }
                    }
                }
                skip_batch = __shfl_sync(0xffffffff, static_cast<int>(skip_batch), 0) != 0;
                if (skip_batch)
                    continue;
                if (token_blocked) {
#if MK_PERF_TRACE_ARGS
                    if (lane == 0)
                        priority_not_full += 1;
#endif
                    if (lane == 0)
                        st_na_release(&state->priority_batch_retry_epoch[batch_idx], priority_epoch + MK_PRIORITY_NOT_READY_RETRY_EPOCHS);
                    continue;
                }

                const int ready_base = state->expert_slot_base[expert_id] + batch_start;
                bool lane_ready = true;
                for (int s = lane; s < COMPUTE_BATCH_SIZE; s += 32) {
                    if (ld_acquire_global(&state->expert_slot_ready[ready_base + s]) == 0) {
                        lane_ready = false;
                        break;
                    }
                }
                unsigned ready_mask = __ballot_sync(0xffffffff, lane_ready);
                if (ready_mask == 0xffffffffu) {
                    bool did_enqueue = false;
                    if (lane == 0)
                        did_enqueue = scheduler_try_enqueue_batch(state, expert_id, batch_id, batch_start, COMPUTE_BATCH_SIZE, 0, 2);
                    did_enqueue = __shfl_sync(0xffffffff, static_cast<int>(did_enqueue), 0) != 0;
                    if (did_enqueue) {
                        ++enqueued;
#if MK_PERF_TRACE_ARGS
                        if (lane == 0) {
                            priority_full_batch_hits += 1;
                            atomicAdd(&state->token_priority_dep_count[token], 1);
                        }
#endif
                    } else {
#if MK_PERF_TRACE_ARGS
                        if (lane == 0) {
                            priority_batch_already_enqueued += 1;
                            int source = ld_acquire_global(&state->expert_batch_enqueued[batch_idx]);
                            if (source == 1)
                                priority_already_normal += 1;
                            else if (source == 2)
                                priority_already_priority += 1;
                            else if (source == 3)
                                priority_already_tail += 1;
                        }
#endif
                    }
                } else {
                    token_blocked = true;
#if MK_PERF_TRACE_ARGS
                    if (lane == 0)
                        priority_not_full += 1;
#endif
                    if (lane == 0)
                        st_na_release(&state->priority_batch_retry_epoch[batch_idx], priority_epoch + MK_PRIORITY_NOT_READY_RETRY_EPOCHS);
                }
            }
            // Priority is a low-latency scanner: a not-full batch should not pin
            // the global cursor and starve later combine-order tokens.
            new_cursor = token + 1;
        }
        if (new_cursor >= state->max_total_recv_tokens)
            new_cursor = 0;
        if (lane == 0 && new_cursor != cursor)
            st_na_release(state->priority_token_cursor, new_cursor);
#if MK_PERF_TRACE_ARGS
        sched_priority_acc += globaltimer_ns() - priority_start;
#endif
        __nanosleep(enqueued > 0 ? 16 : 64);
    }

    if (lane == 0)
        st_na_release(state->priority_scheduler_done, 1);

#if MK_PERF_TRACE_ARGS
    if (lane == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_ns),
                  static_cast<unsigned long long>(sched_priority_acc));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_scan_tokens),
                  static_cast<unsigned long long>(priority_scan_tokens));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_ready_tokens),
                  static_cast<unsigned long long>(priority_ready_tokens));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_full_batch_hits),
                  static_cast<unsigned long long>(priority_full_batch_hits));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_batch_already_enqueued),
                  static_cast<unsigned long long>(priority_batch_already_enqueued));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_already_normal),
                  static_cast<unsigned long long>(priority_already_normal));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_already_priority),
                  static_cast<unsigned long long>(priority_already_priority));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_already_tail),
                  static_cast<unsigned long long>(priority_already_tail));
        atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_priority_not_full),
                  static_cast<unsigned long long>(priority_not_full));
    }
#endif
}

__device__ __forceinline__ void scheduler_scan_gather_tokens(MegaKernelState* state, int scheduler_id, int num_schedulers) {
    const int tid = threadIdx.x;
    if (tid < GATHER_SCHED_TID_BEGIN)
        return;

    constexpr int kGatherBatchTokens = 64;
    constexpr int kGatherTailBatchTokens = 8;
    const int gather_tid = tid - GATHER_SCHED_TID_BEGIN;
    const int gather_threads = blockDim.x - GATHER_SCHED_TID_BEGIN;
    if (gather_threads <= 0)
        return;

    const int gather_lane = gather_tid & 31;
    const int gather_warp_idx = gather_tid >> 5;
    const int gather_num_warps = (gather_threads + 31) / 32;
    if (gather_warp_idx >= GATHER_SCHED_MAX_WARPS)
        return;

    __shared__ int s_gather_batch_tokens[GATHER_SCHED_MAX_WARPS][kGatherBatchTokens];
    __shared__ int s_gather_batch_count[GATHER_SCHED_MAX_WARPS];

    if (gather_lane == 0)
        s_gather_batch_count[gather_warp_idx] = 0;
    __syncwarp();

    const int total_tokens = state->combine_num_tokens;
    const int cursor_idx = scheduler_id * GATHER_SCHED_MAX_WARPS + gather_warp_idx;
    const int cursor_start = scheduler_id + gather_warp_idx * num_schedulers * 32;
    const int warp_stride = gather_num_warps * num_schedulers * 32;
    const int lane_stride = num_schedulers;
    int cursor = ld_nc_global(&state->gather_scan_cursor[cursor_idx]);
    if (cursor < cursor_start || cursor >= total_tokens)
        cursor = cursor_start;

    const int scan_groups = (total_tokens > cursor_start) ? ((total_tokens - cursor_start + warp_stride - 1) / warp_stride) : 0;
    const bool dispatch_finished = ld_acquire_global(state->publish_all_done) != 0;
    const int target_batch_tokens = dispatch_finished ? kGatherTailBatchTokens : kGatherBatchTokens;
    const int max_scan_groups = scan_groups;
    int scan_steps = 0;

    // Interleave by token index so scheduler SM0 scans token_idx % 2 == 0 and
    // scheduler SM1 scans token_idx % 2 == 1. All lanes in a gather scheduler
    // warp scan in parallel and compact ready multi-hit tokens into one task.
    while (scan_steps < max_scan_groups && s_gather_batch_count[gather_warp_idx] < target_batch_tokens) {
        const int token = cursor + gather_lane * lane_stride;
        bool ready = false;
        if (token < total_tokens && ld_nc_global(&state->gather_claimed[token]) == 0) {
            int nhits = ld_acquire_global(&state->token_nhits[token]);
            if (nhits > 1) {
                int expected = ld_acquire_global(&state->token_compute_expected[token]);
                int done = ld_acquire_global(&state->token_done_count[token]);
                ready = (expected == nhits && done >= nhits);
            }
        }

        unsigned ready_mask = __ballot_sync(0xffffffff, ready);
        const int ready_count = __popc(ready_mask);
        int base = 0;
        int space = 0;
        if (gather_lane == 0) {
            base = s_gather_batch_count[gather_warp_idx];
            space = max(target_batch_tokens - base, 0);
        }
        base = __shfl_sync(0xffffffff, base, 0);
        space = __shfl_sync(0xffffffff, space, 0);

        const int ready_rank = __popc(ready_mask & ((1u << gather_lane) - 1));
        bool claimed = ready && ready_rank < space && atomicCAS(&state->gather_claimed[token], 0, 1) == 0;
        unsigned claimed_mask = __ballot_sync(0xffffffff, claimed);
        const int claimed_count = __popc(claimed_mask);
        const int claimed_rank = __popc(claimed_mask & ((1u << gather_lane) - 1));
        if (claimed)
            s_gather_batch_tokens[gather_warp_idx][base + claimed_rank] = token;
        if (gather_lane == 0)
            s_gather_batch_count[gather_warp_idx] = base + claimed_count;

        cursor += warp_stride;
        if (cursor >= total_tokens)
            cursor = cursor_start;
        ++scan_steps;
    }
    __syncwarp();

    int batch_count = s_gather_batch_count[gather_warp_idx];
    if (batch_count > kGatherBatchTokens)
        batch_count = kGatherBatchTokens;
    if (gather_lane == 0)
        state->gather_scan_cursor[cursor_idx] = cursor;
    if (batch_count == 0 || gather_lane != 0)
        return;

    int token_base = atomicAdd(state->gather_ready_reserve_tail, batch_count);
    if (token_base + batch_count > state->max_total_recv_tokens) {
        printf("MK gather task token queue overflow, rank=%d base=%d count=%d max=%d\n",
               state->rank, token_base, batch_count, state->max_total_recv_tokens);
        __threadfence_system(); trap();
    }
    for (int i = 0; i < batch_count; ++i)
        state->gather_ready_queue[token_base + i] = s_gather_batch_tokens[gather_warp_idx][i];

    int task_idx = atomicAdd(state->gather_task_count, 1);
    if (task_idx >= state->max_total_recv_tokens) {
        printf("MK gather task queue overflow, rank=%d task=%d max=%d\n",
               state->rank, task_idx, state->max_total_recv_tokens);
        __threadfence_system(); trap();
    }
    state->gather_task_tokens[task_idx] = token_base;
    state->gather_task_nhits[task_idx] = batch_count;
    __threadfence();
    while (atomicCAS(state->gather_ready_tail, task_idx, task_idx + 1) != task_idx) {
        if (ld_acquire_global(state->combine_all_done) != 0)
            break;
        __nanosleep(32);
    }
}

__device__ void compute_scheduler_worker(MegaKernelState* state, int scheduler_id, int num_schedulers) {
    const int tid = threadIdx.x;
    const int num_threads = min(NORMAL_SCHED_THREADS, static_cast<int>(blockDim.x));
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;

    if (tid >= GATHER_SCHED_TID_BEGIN) {
        while (ld_acquire_global(state->combine_all_done) == 0) {
            scheduler_scan_gather_tokens(state, scheduler_id, num_schedulers);
            __nanosleep(64);
        }
        return;
    }

    if (tid >= PRIORITY_SCHED_TID_BEGIN) {
#if MK_PRIORITY_ENABLE
        scheduler_priority_warp_worker(state, scheduler_id);
#endif
        return;
    }

    bool tail_enqueued = false;

    __shared__ int s_priority_alloc_count;
    __shared__ int s_priority_new_cursor;
    __shared__ int s_normal_enqueued_count;
    __shared__ int s_dispatch_done;
    __shared__ int s_compute_all_result;


#if MK_PERF_TRACE_ENABLED
    if (tid == 0 && scheduler_id == 0)
        state->perf_sched_ts[0] = globaltimer_ns();
#endif
#if MK_PERF_TRACE_ARGS
    int64_t sched_scan_acc = 0;
    int64_t sched_enqueue_acc = 0;
    int64_t sched_idle_acc = 0;
    int64_t sched_normal_acc = 0;
    int64_t sched_tail_flush_acc = 0;
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
#if MK_ASYNC_PUBLISH
            s_dispatch_done = (ld_acquire_global(state->publish_all_done) != 0) ? 1 : 0;
#else
            int dispatch_done_count = ld_acquire_global(state->dispatch_done_count);
            s_dispatch_done = (dispatch_done_count == state->expected_dispatch_done_count) ? 1 : 0;
#endif
        }
        scheduler_compute_sync(num_threads);
        bool dispatch_done = (s_dispatch_done != 0);
#if MK_PERF_TRACE_ARGS
        if (tid == 0 && dispatch_done)
            mk_perf_record_first_i64(state->perf_sched_first_done_seen_ts, globaltimer_ns());
#endif

        if (tail_enqueued)
            break;

#if MK_PERF_TRACE_ARGS
        if (tid == 0 && scheduler_id == 0) {
            int task_head = ld_acquire_global(state->compute_task_head);
            int task_tail = ld_acquire_global(state->compute_task_tail);
            if (task_head >= task_tail) {
                queue_empty_count += 1;
                if (dispatch_done)
                    queue_empty_after_dispatch_count += 1;
            }
        }
        int64_t sched_scan_start = 0;
        int64_t sched_enqueue_start = 0;
#endif

        if (!dispatch_done) {
#if MK_PERF_TRACE_ARGS
            sched_scan_start = globaltimer_ns();
#endif

            // === Unified multi-thread cooperative scan + immediate enqueue ===
            // All threads cooperatively scan one expert at a time (parallel ld_acquire).
            // all_ready → advance full wave.  !all_ready → atomicMin finds prefix boundary.
            // tid0 immediately enqueues full batches after each scan advance.
            // Single producer (tid0) — no CAS contention, no spin-wait deadlock.
            for (int expert_id = scheduler_id; expert_id < num_local_experts; expert_id += num_schedulers) {
#if MK_ASYNC_PUBLISH
                if (tid == 0 && ld_acquire_global(state->publish_all_done) != 0) {
                    s_dispatch_done = 1;
#if MK_PERF_TRACE_ARGS
                    mk_perf_record_first_i64(state->perf_sched_first_done_seen_ts, globaltimer_ns());
#endif
                }
                scheduler_compute_sync(num_threads);
                if (s_dispatch_done != 0)
                    break;
#endif
                int old_count;
                if (tid == 0) {
                    old_count = ld_acquire_global(&state->expert_recv_count[expert_id]);
                    s_priority_alloc_count = old_count;
                }
                scheduler_compute_sync(num_threads);
                old_count = s_priority_alloc_count;

                // Read current enqueue cursor
                int cursor = old_count;
                if (tid == 0)
                    cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
                if (tid == 0)
                    s_priority_new_cursor = cursor;
                scheduler_compute_sync(num_threads);
                cursor = s_priority_new_cursor;

                int count = old_count;
                while (count < state->expert_count[expert_id]) {
                    int my_slot = count + tid;
                    int my_ready = 0;
                    if (my_slot < state->expert_count[expert_id])
                        my_ready = (ld_acquire_global(&state->expert_slot_ready[state->expert_slot_base[expert_id] + my_slot]) == 1) ? 1 : 0;

#if MK_ASYNC_PUBLISH
                    int all_ready = scheduler_compute_all_until_publish_done(
                        my_ready || (my_slot >= state->expert_count[expert_id]),
                        &s_compute_all_result, num_threads, state, &s_dispatch_done);
                    if (s_dispatch_done != 0)
                        break;
#else
                    int all_ready = scheduler_compute_all(
                        my_ready || (my_slot >= state->expert_count[expert_id]),
                        &s_compute_all_result, num_threads);
#endif
                    if (!all_ready) {
                        // atomicMin reduction: find the first not-ready tid offset.
                        if (tid == 0)
                            s_priority_alloc_count = num_threads;
                        scheduler_compute_sync(num_threads);
                        if (my_slot < state->expert_count[expert_id] && !my_ready)
                            atomicMin(&s_priority_alloc_count, tid);
                        scheduler_compute_sync(num_threads);
                        count += s_priority_alloc_count;
                        break;
                    }
                    count += num_threads;
                    if (count > state->expert_count[expert_id])
                        count = state->expert_count[expert_id];

                    // Publish recv_count and immediately enqueue full batches
                    if (tid == 0) {
                        __threadfence();
                        st_na_release(&state->expert_recv_count[expert_id], count);
#if MK_PERF_TRACE_ARGS
                        sched_scan_acc += globaltimer_ns() - sched_scan_start;
                        sched_enqueue_start = globaltimer_ns();
#endif
                    }
                    scheduler_compute_sync(num_threads);
                    // Enqueue full batches in [cursor, count) — multi-thread parallel
                    int full_batch_count = (count - cursor) / COMPUTE_BATCH_SIZE;
                    for (int b = tid; b < full_batch_count; b += num_threads) {
                        int start = cursor + b * COMPUTE_BATCH_SIZE;
                        int batch_id = start / COMPUTE_BATCH_SIZE;
                        scheduler_try_enqueue_batch(state, expert_id, batch_id, start, COMPUTE_BATCH_SIZE, 0, 1);
                    }
                    scheduler_compute_sync(num_threads);
                    if (tid == 0) {
                        cursor += full_batch_count * COMPUTE_BATCH_SIZE;
                        st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
                        s_priority_new_cursor = cursor;
#if MK_PERF_TRACE_ARGS
                        normal_full_batch_enqueues += full_batch_count;
                        sched_enqueue_acc += globaltimer_ns() - sched_enqueue_start;
                        sched_scan_start = globaltimer_ns();
#endif
                    }
                    scheduler_compute_sync(num_threads);
                    cursor = s_priority_new_cursor;
                }
#if MK_ASYNC_PUBLISH
                if (s_dispatch_done != 0)
                    break;
#endif

                // Final enqueue for any remaining full batches after scan completes for this expert
                if (tid == 0 && count != old_count) {
                    __threadfence();
                    st_na_release(&state->expert_recv_count[expert_id], count);
                }
                scheduler_compute_sync(num_threads);
                int final_full_batch_count = (count - cursor) / COMPUTE_BATCH_SIZE;
                for (int b = tid; b < final_full_batch_count; b += num_threads) {
                    int start = cursor + b * COMPUTE_BATCH_SIZE;
                    int batch_id = start / COMPUTE_BATCH_SIZE;
                    scheduler_try_enqueue_batch(state, expert_id, batch_id, start, COMPUTE_BATCH_SIZE, 0, 1);
                }
                scheduler_compute_sync(num_threads);
                if (tid == 0) {
                    cursor += final_full_batch_count * COMPUTE_BATCH_SIZE;
                    st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
#if MK_PERF_TRACE_ARGS
                    normal_full_batch_enqueues += final_full_batch_count;
#endif
                }
#if MK_PERF_TRACE_ARGS
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
                        stall_first_unready_ready = (count < state->expert_count[expert_id]) ? ld_acquire_global(&state->expert_slot_ready[state->expert_slot_base[expert_id] + count]) : -1;
                        stall_dispatch_done = dispatch_done ? 1 : 0;
                    }
                }
#endif
                scheduler_compute_sync(num_threads);
            }
#if MK_ASYNC_PUBLISH
            dispatch_done = (s_dispatch_done != 0);
#endif
#if MK_PERF_TRACE_ARGS
            sched_scan_acc += globaltimer_ns() - sched_scan_start;
            sched_enqueue_start = globaltimer_ns();
            sched_enqueue_acc += 0;
            sched_normal_acc += 0;
#endif
        }

        // === Tail flush ===
        if (dispatch_done && !tail_enqueued) {
#if MK_PERF_TRACE_ARGS
            int64_t sched_tail_flush_start = globaltimer_ns();
            sched_enqueue_start = globaltimer_ns();
#endif
            // Stop every compute-task producer before reserving final-flush slots.
            // Priority-specific tail pre-publish is only useful when the priority
            // scanner is enabled; keep the cross-scheduler barrier either way so
            // final-flush tasks cannot race ahead of still-running normal producers.
            scheduler_compute_sync(num_threads);
            if (tid == 0) {
                atomicAdd(state->scheduler_done_count, 1);
                if (scheduler_id == 0) {
                    while (ld_acquire_global(state->scheduler_done_count) < num_schedulers)
                        __nanosleep(32);
                    st_na_release(state->scheduler_done_count, 0);
                    st_na_release(state->priority_scheduler_done, 2);
                } else {
                    while (ld_acquire_global(state->priority_scheduler_done) < 2)
                        __nanosleep(32);
                }
            }
            scheduler_compute_sync(num_threads);
#if MK_ASYNC_PUBLISH
            if (tid == 0) {
                for (int pw = 0; pw < state->num_pub_warps_total; ++pw) {
                    while (ld_acquire_global(&state->publish_warp_done[pw]) == 0)
                        __nanosleep(32);
                }
                for (int expert_id = scheduler_id; expert_id < num_local_experts; expert_id += num_schedulers) {
                    int count = ld_acquire_global(&state->expert_token_offsets[expert_id]);
                    int cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
                    if (count < cursor)
                        count = cursor;
                    st_na_release(&state->expert_recv_count[expert_id], count);

                    int full_batch_count = (count - cursor) / COMPUTE_BATCH_SIZE;
                    for (int b = 0; b < full_batch_count; ++b) {
                        int start = cursor + b * COMPUTE_BATCH_SIZE;
                        int batch_id = start / COMPUTE_BATCH_SIZE;
                        bool enq = scheduler_try_enqueue_batch(state, expert_id, batch_id, start, COMPUTE_BATCH_SIZE, 0, 3);
#if MK_PERF_TRACE_ARGS
                        if (enq)
                            normal_full_batch_enqueues += 1;
#endif
                    }
                    cursor += full_batch_count * COMPUTE_BATCH_SIZE;
                    st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);

                    if (count > cursor) {
                        int batch_id = cursor / COMPUTE_BATCH_SIZE;
                        bool enq = scheduler_try_enqueue_batch(state, expert_id, batch_id, cursor, count - cursor, 1, 3);
                        st_na_release(&state->expert_enqueue_cursor[expert_id], count);
#if MK_PERF_TRACE_ARGS
                        if (enq)
                            flush_tail_enqueues += 1;
#endif
                    }
                }
#if MK_PERF_TRACE_ARGS
                int64_t sched_done_publish_start = globaltimer_ns();
                sched_enqueue_acc += sched_done_publish_start - sched_enqueue_start;
#endif
                __threadfence();
                int finished = atomicAdd(state->scheduler_done_count, 1) + 1;
                if (finished == num_schedulers) {
#if MK_PERF_TRACE_ENABLED
                    state->perf_sched_ts[1] = globaltimer_ns();
#endif
                    st_na_release(state->compute_enqueue_done, 1);
                }
#if MK_PERF_TRACE_ARGS
                sched_enqueue_acc += globaltimer_ns() - sched_done_publish_start;
#endif
            }
#else
#if MK_PERF_TRACE_ARGS
            sched_scan_start = globaltimer_ns();
#endif
            for (int expert_id = scheduler_id; expert_id < num_local_experts; expert_id += num_schedulers) {
                int old_count;
                if (tid == 0) {
                    old_count = ld_acquire_global(&state->expert_recv_count[expert_id]);
                    s_priority_alloc_count = old_count;
                }
                scheduler_compute_sync(num_threads);
                old_count = s_priority_alloc_count;

                int count = old_count;
                int cursor = old_count;
                if (tid == 0)
                    cursor = ld_acquire_global(&state->expert_enqueue_cursor[expert_id]);
                scheduler_compute_sync(num_threads);
                if (tid == 0)
                    s_priority_new_cursor = cursor;
                scheduler_compute_sync(num_threads);
                cursor = s_priority_new_cursor;

                while (count < state->expert_count[expert_id]) {
                    int my_slot = (tid < num_threads) ? count + tid : state->expert_count[expert_id];
                    int my_ready = 0;
                    if (my_slot < state->expert_count[expert_id])
                        my_ready = (ld_acquire_global(&state->expert_slot_ready[state->expert_slot_base[expert_id] + my_slot]) == 1) ? 1 : 0;
                    int all_ready = scheduler_compute_all(my_ready || (my_slot >= state->expert_count[expert_id]), &s_compute_all_result, num_threads);
                    if (!all_ready) {
                        if (tid == 0) {
                            for (int s = count; s < count + num_threads && s < state->expert_count[expert_id]; ++s) {
                                if (ld_acquire_global(&state->expert_slot_ready[state->expert_slot_base[expert_id] + s]) == 1)
                                    count = s + 1;
                                else
                                    break;
                            }
                            s_priority_alloc_count = count;
                        }
                        scheduler_compute_sync(num_threads);
                        count = s_priority_alloc_count;
                        break;
                    }
                    count += num_threads;
                    if (count > state->expert_count[expert_id]) count = state->expert_count[expert_id];

                    if (tid == 0) {
                        __threadfence();
                        st_na_release(&state->expert_recv_count[expert_id], count);
#if MK_PERF_TRACE_ARGS
                        sched_scan_acc += globaltimer_ns() - sched_scan_start;
                        sched_enqueue_start = globaltimer_ns();
#endif
                    }
                    scheduler_compute_sync(num_threads);
                    int full_batch_count = (count - cursor) / COMPUTE_BATCH_SIZE;
                    for (int b = tid; b < full_batch_count; b += num_threads) {
                        int start = cursor + b * COMPUTE_BATCH_SIZE;
                        int batch_id = start / COMPUTE_BATCH_SIZE;
                        scheduler_try_enqueue_batch(state, expert_id, batch_id, start, COMPUTE_BATCH_SIZE, 0, 3);
                    }
                    scheduler_compute_sync(num_threads);
                    if (tid == 0) {
                        cursor += full_batch_count * COMPUTE_BATCH_SIZE;
                        st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
                        s_priority_new_cursor = cursor;
#if MK_PERF_TRACE_ARGS
                        normal_full_batch_enqueues += full_batch_count;
                        sched_enqueue_acc += globaltimer_ns() - sched_enqueue_start;
                        sched_scan_start = globaltimer_ns();
#endif
                    }
                    scheduler_compute_sync(num_threads);
                    cursor = s_priority_new_cursor;
                }

                if (tid == 0) {
                    __threadfence();
                    st_na_release(&state->expert_recv_count[expert_id], count);
#if MK_PERF_TRACE_ARGS
                    sched_scan_acc += globaltimer_ns() - sched_scan_start;
                    sched_enqueue_start = globaltimer_ns();
#endif
                }
                scheduler_compute_sync(num_threads);
                int final_full_batch_count = (count - cursor) / COMPUTE_BATCH_SIZE;
                for (int b = tid; b < final_full_batch_count; b += num_threads) {
                    int start = cursor + b * COMPUTE_BATCH_SIZE;
                    int batch_id = start / COMPUTE_BATCH_SIZE;
                    scheduler_try_enqueue_batch(state, expert_id, batch_id, start, COMPUTE_BATCH_SIZE, 0, 3);
                }
                scheduler_compute_sync(num_threads);
                if (tid == 0) {
                    cursor += final_full_batch_count * COMPUTE_BATCH_SIZE;
                    st_na_release(&state->expert_enqueue_cursor[expert_id], cursor);
                    if (count > cursor) {
                        int batch_id = cursor / COMPUTE_BATCH_SIZE;
                        bool enq = scheduler_try_enqueue_batch(state, expert_id, batch_id, cursor, count - cursor, 1, 3);
                        st_na_release(&state->expert_enqueue_cursor[expert_id], count);
#if MK_PERF_TRACE_ARGS
                        if (enq)
                            flush_tail_enqueues += 1;
#endif
                    }
#if MK_PERF_TRACE_ARGS
                    normal_full_batch_enqueues += final_full_batch_count;
                    sched_enqueue_acc += globaltimer_ns() - sched_enqueue_start;
                    sched_scan_start = globaltimer_ns();
#endif
                }
                scheduler_compute_sync(num_threads);
            }
#if MK_PERF_TRACE_ARGS
            int64_t sched_done_publish_start = globaltimer_ns();
#endif
            if (tid == 0) {
                __threadfence();
                int finished = atomicAdd(state->scheduler_done_count, 1) + 1;
                if (finished == num_schedulers) {
#if MK_PERF_TRACE_ENABLED
                    state->perf_sched_ts[1] = globaltimer_ns();
#endif
                    st_na_release(state->compute_enqueue_done, 1);
                }
            }
#if MK_PERF_TRACE_ARGS
            sched_enqueue_acc += globaltimer_ns() - sched_done_publish_start;
#endif
#endif
#if MK_PERF_TRACE_ARGS
            sched_tail_flush_acc += globaltimer_ns() - sched_tail_flush_start;
#endif
            tail_enqueued = true;
        }

        if (tail_enqueued)
            break;

#if MK_PERF_TRACE_ARGS
        int64_t sched_idle_start = globaltimer_ns();
#endif
        __nanosleep(64);
#if MK_PERF_TRACE_ARGS
        sched_idle_acc += globaltimer_ns() - sched_idle_start;
#endif
    }
#if MK_PERF_TRACE_ARGS
    if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_scan_ns),
              static_cast<unsigned long long>(sched_scan_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_enqueue_ns),
              static_cast<unsigned long long>(sched_enqueue_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_idle_ns),
              static_cast<unsigned long long>(sched_idle_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_normal_ns),
              static_cast<unsigned long long>(sched_normal_acc));
    atomicAdd(reinterpret_cast<unsigned long long*>(state->perf_sched_tail_flush_ns),
              static_cast<unsigned long long>(sched_tail_flush_acc));
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

template <ComputeDType kComputeDType, bool kStopAtDispatchDone>
__device__ __forceinline__ void compute_worker_core(
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    MegaKernelState* state,
    int group_id_base,
    uint8_t* smem_buffer
) {

    if constexpr (kComputeDType == ComputeDType::kFP8E4M3) {
        if (threadIdx.x == 0 && compute_sm_idx == 0)
            printf("MK FP8 compute worker is instantiated but FP8 UMMA mainloop is not wired yet.\n");
        __threadfence_system(); trap();
        return;
    }
    const int thread_id = threadIdx.x;
    const int local_warp_id = thread_id / 32;
    if (num_compute_sms <= 0 || compute_sm_idx < 0 || compute_sm_idx >= num_compute_sms)
        return;
    const int local_group_id = compute_sm_idx / COMPUTE_GROUP_SIZE;
    const int group_first_sm_idx = local_group_id * COMPUTE_GROUP_SIZE;
    const int group_size = min(COMPUTE_GROUP_SIZE, num_compute_sms - group_first_sm_idx);
    const int group_id = group_id_base + local_group_id;
    const int num_compute_groups = state->num_compute_groups;
    if (group_size <= 0 || group_id >= num_compute_groups)
        return;
    const int group_sm_idx = compute_sm_idx - group_first_sm_idx;
    const int num_warps_per_sm = blockDim.x / 32;
    const int group_warp_id = group_sm_idx * num_warps_per_sm + local_warp_id;
    const int group_num_warps = group_size * num_warps_per_sm;
    const int group_thread_id = group_sm_idx * blockDim.x + thread_id;
    const int group_num_threads = group_size * blockDim.x;
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

    auto smem_wmma_buf = reinterpret_cast<float*>(smem_buffer);

    using ComputeUmmaSmemLayout = umma::DgSmemLayout<umma::kDgRunMulticast>;
    constexpr size_t kComputeUmmaBarrierBytes =
        (ComputeUmmaSmemLayout::kNumStages * 3 + ComputeUmmaSmemLayout::kNumEpilogueStages * 2 + 1) *
        sizeof(cutlass::arch::ClusterTransactionBarrier) + sizeof(uint32_t);
    constexpr size_t kComputeUmmaScratchBytes =
        ComputeUmmaSmemLayout::SMEM_CD_SIZE +
        ComputeUmmaSmemLayout::kNumStages *
            (ComputeUmmaSmemLayout::SMEM_A_SIZE_PER_STAGE + ComputeUmmaSmemLayout::SMEM_B_SIZE_PER_STAGE) +
        kComputeUmmaBarrierBytes;
    constexpr size_t kComputeWmmaScratchBytes =
        (kNumCombineForwarderWarps + 1) * 2 * WMMA_M * WMMA_N * sizeof(float);
    constexpr size_t kComputeScratchBytes =
        kComputeWmmaScratchBytes > kComputeUmmaScratchBytes ? kComputeWmmaScratchBytes : kComputeUmmaScratchBytes;
    constexpr size_t kComputeMetaOffset = (kComputeScratchBytes + alignof(int) - 1) & ~(size_t)(alignof(int) - 1);

    constexpr size_t kRecvTokenIdxBytes = COMPUTE_BATCH_SIZE * sizeof(int);
    constexpr size_t kTopkSlotBytes = COMPUTE_BATCH_SIZE * sizeof(int);
    constexpr size_t kIsSingleBytes = COMPUTE_BATCH_SIZE * sizeof(unsigned char);
    constexpr size_t kRouteWAlignPad = alignof(float) - 1;
    constexpr size_t kRouteWBytes = COMPUTE_BATCH_SIZE * sizeof(float);
    constexpr size_t kComputeBatchMetaBytes = kRecvTokenIdxBytes + kTopkSlotBytes + kIsSingleBytes + kRouteWAlignPad + kRouteWBytes;
    static_assert(kComputeMetaOffset + kComputeBatchMetaBytes <=
                  kNumCombineTMABytesPerForwarderWarp * kNumCombineForwarderWarps,
                  "compute dynamic smem metadata must fit after compute scratch");

    uint8_t* compute_smem = smem_buffer + kComputeMetaOffset;
    int* s_recv_token_idx = reinterpret_cast<int*>(compute_smem);
    compute_smem += kRecvTokenIdxBytes;
    int* s_topk_slot = reinterpret_cast<int*>(compute_smem);
    compute_smem += kTopkSlotBytes;
    unsigned char* s_is_single = reinterpret_cast<unsigned char*>(compute_smem);
    constexpr size_t kRouteWOffset =
        (kComputeMetaOffset + kRecvTokenIdxBytes + kTopkSlotBytes + kIsSingleBytes + alignof(float) - 1) & ~(size_t)(alignof(float) - 1);
    // Per-row route weight, gathered once and consumed inside the fused
    // gate+up SwiGLU epilogue (act = silu(gate) * up * route_w).
    float* s_route_w = reinterpret_cast<float*>(smem_buffer + kRouteWOffset);

    // TMEM persistent across tasks: allocate on first UMMA use, keep alive
    // until the compute worker exits the persistent loop. This eliminates
    // per-task init/dealloc overhead (2x cluster_sync + barrier init each).
    bool umma_tmem_allocated = false;
#if MK_PERF_TRACE_ARGS
    int64_t last_task_end_ns = 0;
#endif

    while (true) {
#if MK_PERF_TRACE_ARGS
        int64_t pop_start_ns = 0;
        int64_t pop_done_ns = 0;
        int pop_attempts = 0;
        int cas_failures = 0;
#endif
        if (group_sm_idx == 0 && thread_id == 0) {
            int task_idx = -1;
#if MK_PERF_TRACE_ARGS
            pop_start_ns = globaltimer_ns();
#endif
            while (true) {
                if constexpr (kStopAtDispatchDone) {
                    int enqueue_done = ld_acquire_global(state->compute_enqueue_done);
                    if (enqueue_done) {
                        int tail = ld_acquire_global(state->compute_task_tail);
                        int head = ld_acquire_global(state->compute_task_head);
                        if (tail == 0 || head * 100 >= tail * COMBINE_START_HEAD_PERCENT) {
                            task_idx = -3;
                            break;
                        }
                    }
                }
                int head = ld_acquire_global(state->compute_task_head);
                int tail = ld_acquire_global(state->compute_task_tail);
                if (head >= tail) {
                    if constexpr (!kStopAtDispatchDone) {
                        if (ld_acquire_global(state->compute_enqueue_done))
                            task_idx = -2;
                    }
                    break;
                }
                if (atomicCAS(state->compute_task_head, head, head + 1) == head) {
                    task_idx = head;
#if MK_PERF_TRACE_ARGS
                    pop_done_ns = globaltimer_ns();
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
#if MK_PERF_TRACE_ARGS
                cas_failures += 1;
#endif
            }
            st_release_gpu_global(&state->compute_group_task_idx[group_id], task_idx);
        }
        compute_group_sync(state, group_id, group_size);

        int task_idx = ld_acquire_global(&state->compute_group_task_idx[group_id]);
#if MK_PERF_TRACE_ARGS
        if (group_sm_idx == 0 && thread_id == 0 && task_idx >= 0 && task_idx < state->max_compute_tasks)
            state->perf_task_bcast_done_ts[task_idx] = globaltimer_ns();
#endif
        if (task_idx == -2 || task_idx == -3) {
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
            compute_group_sync(state, group_id, group_size);
            continue;
        }

        ComputeTask task = state->compute_tasks[task_idx];
        int expert_id = task.expert_id;
        int start_slot = task.start_slot;
        int batch_size = task.num_tokens;

        // MK_COMPUTE_KERNEL selects the compute implementation at compile time:
        //   0 = WMMA gate/up + WMMA down
        //   1 = 1-CTA UMMA gate/up + 1-CTA UMMA down
        //   2 = 2-CTA UMMA gate/up + 2-CTA UMMA down
        constexpr bool kUseUmmaCompute = (MK_COMPUTE_KERNEL != 0);
        constexpr bool kUseUmmaGateup = kUseUmmaCompute && (MK_UMMA_GATEUP != 0);
        constexpr bool kUseUmmaDown = kUseUmmaCompute && (MK_UMMA_DOWN != 0);
        const bool use_umma_gateup_for_group = kUseUmmaGateup && group_size == COMPUTE_GROUP_SIZE;
        const bool use_umma_down_for_group = kUseUmmaDown && group_size == COMPUTE_GROUP_SIZE;
        constexpr int kUmmaClusterDim = (MK_COMPUTE_KERNEL == 2 ? 2 : 1);
        constexpr int kUmmaClustersPerGroup = COMPUTE_GROUP_SIZE / kUmmaClusterDim;

        // Tail-batch M packing (1-CTA only): round the real token count up to the
        // UMMA M-tile (128) instead of always padding to COMPUTE_BATCH_SIZE (1024).
        // This drops scheduled M-tiles from ceil(1024/128)=8 to ceil(batch_size/128),
        // so a 10-token tail runs 1 M-tile instead of 8. Rows [batch_size, gemm_m)
        // are still zero-padded and masked by valid_rows/batch_size.
        constexpr int kGemmMAlign = 128;   // UMMA M-tile granularity (kDgBlockM)
        const int gemm_m = (kUmmaClusterDim == 1)
            ? min(COMPUTE_BATCH_SIZE,
                  (batch_size + kGemmMAlign - 1) / kGemmMAlign * kGemmMAlign)
            : COMPUTE_BATCH_SIZE;

        // DeepGEMM mega_moe prefetches TMA descriptors before the main data movement.
        // Keep gate/up and down independent so diagnostic switches can isolate each UMMA path.
        if constexpr (kUseUmmaCompute) {
            if (batch_size <= COMPUTE_BATCH_SIZE) {
                const umma::InputTmaAtom_t& prefetch_atom = state->group_input_tma[group_id];
                if (local_warp_id == 0) {
                    if (use_umma_gateup_for_group && state->compute_tma != nullptr) {
                        cute::prefetch_tma_descriptor(&prefetch_atom.a);
                        cute::prefetch_tma_descriptor(&state->compute_tma->wgateup[expert_id]);
                        cute::prefetch_tma_descriptor(&prefetch_atom.act_cd);
                    }
                    if (use_umma_down_for_group && state->compute_down_tma != nullptr) {
                        cute::prefetch_tma_descriptor(&prefetch_atom.act_a);
                        cute::prefetch_tma_descriptor(&state->compute_down_tma->wdown[expert_id]);
                        cute::prefetch_tma_descriptor(&prefetch_atom.down_cd);
                    }
                }
            }
        }
#if MK_PERF_TRACE_ENABLED
        const bool perf_leader = (group_sm_idx == 0 && thread_id == 0);
        int64_t compute_task_start_ns = perf_leader ? globaltimer_ns() : 0;
        int64_t compute_task_end_ns = 0;
#endif
#if MK_PERF_TRACE_ARGS
        int64_t perf_ph_meta_ns = 0, perf_ph_input_ns = 0, perf_ph_upgemm_ns = 0;
        int64_t perf_ph_downgemm_ns = 0, perf_ph_output_ns = 0;
        // Finer breakpoints: GEMM body end (before barrier) and signaling sub-phases.
        int64_t perf_up_body_ns = 0, perf_down_body_ns = 0, perf_out_body_ns = 0;
        int64_t perf_sig_donecount_ns = 0, perf_sig_finalize_ns = 0;
        int64_t perf_sig_fence_ns = 0, perf_sig_publish_ns = 0;
        if (perf_leader && task_idx >= 0 && task_idx < state->max_compute_tasks) {
            state->perf_task_start_ts[task_idx] = compute_task_start_ns;
            state->perf_task_prev_end_ts[task_idx] = last_task_end_ns;
            state->perf_task_prev_gap_ns[task_idx] = last_task_end_ns == 0 ? 0 : compute_task_start_ns - last_task_end_ns;
        }
        // Root-cause diagnostics rendered as args on compute X-events.
        __shared__ int s_perf_multi_expert_rows;
        __shared__ int s_perf_task_has_multi;
        if (perf_leader) {
            s_perf_multi_expert_rows = 0;
            s_perf_task_has_multi = 0;
        }
        __syncthreads();
#endif

        for (int i = thread_id; i < batch_size; i += blockDim.x) {
            int base_offset = state->expert_slot_base[expert_id] + start_slot + i;
            int recv_token = ld_acquire_global(&state->recv_token_source_info[base_offset * 2]);
            int topk_slot = ld_acquire_global(&state->recv_token_source_info[base_offset * 2 + 1]);
            int expected = ld_acquire_global(&state->token_compute_expected[recv_token]);
            s_recv_token_idx[i] = recv_token;
            s_topk_slot[i] = topk_slot;
            s_is_single[i] = static_cast<unsigned char>(expected == 1);
            s_route_w[i] = ld_nc_global(&state->combine_input_topk_weights[recv_token * num_topk + topk_slot]);
            // Phase 3 (Step 3.3a): record (recv_token, topk_slot) -> compact forward slot,
            // and repurpose s_topk_slot to carry that compact slot so the preact epilogue
            // (called with num_topk=0) writes bwd_preact by slot: (recv*0 + slot)*stride = slot*stride.
            if (recv_token >= 0 && topk_slot >= 0)
                state->fwd_slot_map[recv_token * num_topk + topk_slot] = base_offset;
            s_topk_slot[i] = base_offset;
// #ifdef MK_TOKEN_TRACE
//             printf("[MK-TOKEN][COMPUTE] rank=%d sm=%d expert=%d row=%d recv_token=%d topk_slot=%d\n",
//                    state->rank, sm_id, expert_id, i, recv_token, topk_slot);
// #endif
        }
        // Zero-init padding rows [batch_size, COMPUTE_BATCH_SIZE) so the UMMA path
        // can run over the fixed task extent: padded input_buf rows are 0, and
        // route_w must be defined (SwiGLU on padding is 0 anyway, but avoid reading
        // uninitialized shared memory). Output/reduce/signal all mask by batch_size,
        // so padding rows never leave the kernel.
        for (int i = batch_size + thread_id; i < gemm_m; i += blockDim.x) {
            s_route_w[i] = 0.0f;
            s_recv_token_idx[i] = -1;
            s_topk_slot[i] = -1;
        }
        __syncthreads();
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_meta_ns = globaltimer_ns();
#endif

        const int hidden_int4 = hidden * sizeof(__nv_bfloat16) / sizeof(int4);
        const int input_vec_stride = gemm_m * hidden_int4;
        const int4* combine_input_i4 = reinterpret_cast<const int4*>(state->combine_input);
        int4* input_buf_i4 = reinterpret_cast<int4*>(input_buf);
        for (int idx = group_thread_id; idx < input_vec_stride; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            input_buf_i4[idx] = (row < batch_size)
                ? combine_input_i4[(int64_t)s_recv_token_idx[row] * hidden_int4 + v]
                : make_int4(0, 0, 0, 0);
        }
        compute_group_sync(state, group_id, group_size);
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_input_ns = globaltimer_ns();
#endif

        // Expert weight slices
        const __nv_bfloat16* w_gateup = &state->W_gateup[expert_id * 2 * intermediate * hidden];
        const __nv_bfloat16* w_down = &state->W_down[expert_id * hidden * intermediate];

        // Gate/up compute always consumes pairwise interleaved W_gateup rows
        // [g0,u0,g1,u1,...]. UMMA folds adjacent gate/up columns in its epilogue;
        // WMMA fallback reads the same layout with a 2*K B-matrix stride.
        // umma_accum_iter is reset to 0 at the start of each GEMM pass (gate/up
        // and down) since barriers are re-initialized. TMEM itself persists across
        // tasks — only allocated once and freed on worker exit.
        uint32_t umma_accum_iter = 0;

        // Tail batches (batch_size < COMPUTE_BATCH_SIZE) also run the UMMA path over
        // the fixed task extent: input_buf rows [batch_size, COMPUTE_BATCH_SIZE) are
        // zero-padded above, so padded GEMM rows are harmless and SwiGLU/output/
        // reduce/signal all mask by batch_size, so padding never leaves the kernel.
        if (use_umma_gateup_for_group && state->compute_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];

            // Interleaved gate/up fusion: ONE persistent GEMM computes GU with
            // Wgu rows [g0,u0,g1,u1,...], then the epilogue folds each adjacent
            // gate/up pair directly from TMEM into act_buf (up_buf). This is the
            // microkernel path moved into the megakernel for both 1-CTA and 2-CTA.
            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                // TMEM already allocated; just re-init barriers for the new pass.
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            umma_accum_iter = 0;  // Reset accumulator phase for fresh barriers.
            umma::umma_gateup_interleaved_persistent(
                &in_atom.a,
                &state->compute_tma->wgateup[expert_id],
                &in_atom.act_cd,
                s_route_w,
                gemm_m, intermediate, hidden,
                cluster_in_group, num_clusters,
                cluster_smem, umma_accum_iter,
                (MK_UMMA_SAVE_PREACT != 0) ? reinterpret_cast<cutlass::bfloat16_t*>(state->bwd_preact) : nullptr,
                s_recv_token_idx, s_topk_slot,
                /* preact num_topk = 0: s_topk_slot carries the compact forward slot (Step 3.3a) */
                0, 2 * intermediate, batch_size);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_up_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        } else {
            device_gemm_swiglu_fused(input_buf, w_gateup, up_buf, s_route_w,
                                     batch_size, batch_size, hidden, intermediate,
                                     group_warp_id, group_num_warps, local_warp_id, smem_wmma_buf,
                                     state->bwd_preact, s_recv_token_idx, s_topk_slot,
                                     /* preact num_topk = 0: s_topk_slot carries compact slot */ 0, 2 * intermediate);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_up_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        }
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_upgemm_ns = globaltimer_ns();
#endif

        // GEMM 3: down-proj D = act @ W_down^T.
        // TMEM is shared with gate/up — no separate init/dealloc. accum_iter
        // continues from gate/up so the TMEM phase ring stays correct.
        // Tail batches run over the fixed task extent: padded act rows hold the
        // SwiGLU of zero-padded gate/up (== 0), so padded down output rows are 0
        // and are masked off by the batch_size-bounded output/reduce below.
        if (use_umma_down_for_group &&
            state->compute_down_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];

            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                // TMEM already allocated; just re-init barriers for the new pass.
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            umma_accum_iter = 0;  // Reset accumulator phase for fresh barriers.
            umma::umma_down_persistent(
                &in_atom.act_a,
                &state->compute_down_tma->wdown[expert_id],
                &in_atom.down_cd,
                gemm_m, hidden, intermediate,
                cluster_in_group, num_clusters,
                cluster_smem, umma_accum_iter);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_down_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        } else {
            device_gemm_bf16(up_buf, w_down, down_buf, batch_size, intermediate, hidden,
                              group_warp_id, group_num_warps, local_warp_id, smem_wmma_buf);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_down_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        }
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_downgemm_ns = globaltimer_ns();
#endif

        // Per-slot output (change A): every (recv_token, local-expert-hit) writes its OWN
        // expert-sorted slot row in compute_output_slot. Slots are unique across tasks, so
        // no write conflict and no atomic. Same-rank multi-expert reduce is deferred to the
        // combine sender (change C), which gathers a token's nh slots and fp32-sums them.
#if MK_PERF_TRACE_ARGS
        if (thread_id == 0) {
            int multi_rows = 0;
            for (int row = 0; row < batch_size; ++row)
                if (!s_is_single[row]) ++multi_rows;
            if (perf_leader) s_perf_task_has_multi = (multi_rows != 0);
        }
        __syncthreads();
#endif

        const int4* down_i4 = reinterpret_cast<const int4*>(down_buf);
        int4* slot_out_i4 = reinterpret_cast<int4*>(state->compute_output_slot);
        int4* token_out_i4 = reinterpret_cast<int4*>(state->combine_input);
        const int slot_base = state->expert_slot_base[expert_id] + start_slot;
        for (int idx = group_thread_id; idx < batch_size * hidden_int4; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            int slot = slot_base + row;
            if (s_is_single[row])
                token_out_i4[(int64_t)s_recv_token_idx[row] * hidden_int4 + v] = down_i4[idx];
            else
                slot_out_i4[(int64_t)slot * hidden_int4 + v] = down_i4[idx];
        }

        // ==== Backward activation save ====
        // Save fc1 input (permuted X) by recv_token for the backward pass. input_buf still
        // holds X here (the scatter above only overwrote combine_input, not input_buf).
        // Multi-hit tokens write the same X, so the by-recv_token store is idempotent.
        if (state->bwd_fc1_input != nullptr) {
            const int4* bwd_in_src_i4 = reinterpret_cast<const int4*>(input_buf);
            int4* bwd_in_i4 = reinterpret_cast<int4*>(state->bwd_fc1_input);
            for (int idx = group_thread_id; idx < batch_size * hidden_int4; idx += group_num_threads) {
                int row = idx / hidden_int4;
                int v = idx - row * hidden_int4;
                bwd_in_i4[(int64_t)s_recv_token_idx[row] * hidden_int4 + v] = bwd_in_src_i4[idx];
            }
        }
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_out_body_ns = globaltimer_ns();
#endif
        // device-scope fence: combine_worker reads compute_output_slot on the same GPU
        // (different SM, same kernel launch), so device-scope visibility is sufficient.
        // Each thread fences its own per-slot writes before the group sync lets any SM
        // publish ready flags.
        __threadfence();
        compute_group_sync(state, group_id, group_size);
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_output_ns = globaltimer_ns();
#endif

        // ==== Signal: per-token ready publish ====
        // compute wrote per-slot rows into compute_output_slot. The same-rank multi-expert
        // reduce is now done by the combine sender / gather worker, which gather a token's nh
        // slots and fp32-sum them before sending. Compute only advances token_done_count and
        // publishes nhits==1 tokens directly; nhits>1 tokens are claimed/enqueued by gather
        // scheduler lanes (tid >= GATHER_SCHED_TID_BEGIN) once all local expert slots are done.
        {
#if MK_PERF_TRACE_ARGS
            if (perf_leader) {
                perf_sig_donecount_ns = globaltimer_ns();
                perf_sig_finalize_ns = perf_sig_donecount_ns;
                perf_sig_fence_ns = perf_sig_donecount_ns;
            }
            if (perf_leader) {
                int mr = 0;
                for (int row = 0; row < batch_size; ++row)
                    if (!s_is_single[row]) ++mr;
                s_perf_multi_expert_rows = mr;
            }
#endif
            for (int row = group_thread_id; row < batch_size; row += group_num_threads) {
                const int recv_token = s_recv_token_idx[row];
                int done = atomicAdd(&state->token_done_count[recv_token], 1) + 1;
                if (s_is_single[row] && done >= 1) {
                    __threadfence();
                    atomicExch(&state->combine_token_ready[recv_token], 1);
                }
            }
            compute_group_sync(state, group_id, group_size);
#if MK_PERF_TRACE_ENABLED
            if (perf_leader) compute_task_end_ns = globaltimer_ns();
#endif
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_sig_publish_ns = compute_task_end_ns;
#endif
        }

#if MK_PERF_TRACE_ENABLED
        if (perf_leader) {
#if MK_PERF_TRACE_ARGS
            last_task_end_ns = compute_task_end_ns;
#endif
            int slot = task_idx;
            if (slot >= 0 && slot < state->max_compute_tasks) {
                int64_t* rec = state->perf_compute_task + (int64_t)slot * MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS;
                rec[0] = compute_task_start_ns;
                rec[1] = compute_task_end_ns;
                rec[2] = sm_id;
                rec[3] = group_id;
                rec[5] = batch_size;
                rec[25] = task.is_flush;
                rec[26] = 1;
#if MK_PERF_TRACE_ARGS
                rec[4] = expert_id;
                rec[6] = hidden;
                rec[7] = intermediate;
                rec[8]  = perf_ph_meta_ns;
                rec[9]  = perf_ph_input_ns;
                rec[10] = perf_ph_upgemm_ns;
                rec[11] = perf_ph_downgemm_ns;
                rec[12] = perf_ph_output_ns;
                rec[13] = compute_task_end_ns;
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
                rec[24] = static_cast<int64_t>(state->expert_slot_base[expert_id]) + start_slot;
                state->perf_compute_multi_expert_rows[slot] = s_perf_multi_expert_rows;
                state->perf_compute_task_has_multi[slot] = s_perf_task_has_multi;
#endif
            }
        }
#endif

    }
}

// Forward compute worker wrapper. The core above is shared with combine-precompute
// so the persistent kernel keeps one implementation body and only the entry policy changes.
template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_worker(
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    MegaKernelState* state,
    uint8_t* smem_buffer
) {
    compute_worker_core<kComputeDType, false>(
        sm_id, compute_sm_idx, num_compute_sms, state, 0, smem_buffer);
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

    // if (threadIdx.x == 0 && combine_sm_idx == 0) {
    //     printf("rank: %d, combine_sm_idx: %d num_channels: %d num_logical_channels: %d \n", state->rank, combine_sm_idx, num_channels, num_logical_channels);
    // }

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

    // ===== RDMA buffer reuse prelude (phase A + metadata clean + phase B) =====
    // Runs once per GPU before combine begins. Leader = first warp of combine SM 0 (32 lanes
    // cooperate); all other combine threads wait on state->rdma_reuse_prelude_done.
    //   Phase A : wait this GPU's dispatch send side is drained, quiet dispatch QPs, cross-rank
    //             barrier over same-nvl peers via the dispatch_quiet_done mailbox.
    //   Clean   : zero the combine RDMA metadata (data-after head/tail region) of the reused
    //             combine RDMA buffer, re-initializing head/tail for combine.
    //   Phase B : cross-rank barrier over same-nvl peers via the combine_clear_done mailbox, so no
    //             peer starts combine-RDMA sends into our buffer before our metadata is cleared.
    // Work is spread across the leader warp's lanes. Quiet tasks are partitioned so no two lanes
    // ever quiet the same (dst_pe, qp_id) (ibgda_poll_cq is not thread-safe); flag posts use qp 0
    // to distinct dst_pe per lane, and lane 0 owns the local flag stores / fence.
    if (state->rdma_reuse_prelude_enable) {
        const int num_rdma_ranks_local = num_ranks / NUM_MAX_NVL_PEERS;
        if (combine_sm_idx == 0 && thread_id < 32) {
            const int ndc = state->num_dispatch_channels;

            // 1. Dispatch send finished (dispatch_channel_barrier[lc] reaches 2 only after the RDMA
            //    sender coordinator + forwarder passed the post-send CTA barrier). Lane-distributed.
            for (int lc = lane_id; lc < num_logical_channels; lc += 32) {
                while (ld_acquire_sys_global(&state->dispatch_channel_barrier[lc]) < 2)
                    __nanosleep(64);
            }
            __syncwarp();

            // 2. Quiet dispatch QPs (sender ch + forwarder ch+ndc) to same-nvl remote peers.
            //    Task t = ((dr * ndc) + ch) * 2 + is_fwd -> unique (dst_pe, qp), one lane each.
            const int qp_tasks = num_rdma_ranks_local * ndc * 2;
            for (int t = lane_id; t < qp_tasks; t += 32) {
                const int dr = t / (ndc * 2);
                const int rem = t % (ndc * 2);
                const int ch = rem >> 1;
                const int is_fwd = rem & 1;
                if (dr == rdma_rank) continue;
                const int dst_pe = translate_dst_rdma_rank<kLowLatencyMode>(dr, nvl_rank);
                nvshmemi_ibgda_quiet(dst_pe, is_fwd ? (ch + ndc) : ch);
            }
            __syncwarp();

            // 3. Publish dispatch-quiet-done into every same-nvl peer's mailbox (and locally).
            if (lane_id == 0)
                st_release_sys_global(&state->rdma_reuse_dispatch_quiet_done[rdma_rank], 1);
            for (int dr = lane_id; dr < num_rdma_ranks_local; dr += 32) {
                if (dr == rdma_rank) continue;
                const int dst_pe = translate_dst_rdma_rank<kLowLatencyMode>(dr, nvl_rank);
                nvshmemi_ibgda_rma_p(&state->rdma_reuse_dispatch_quiet_done[rdma_rank], 1, dst_pe, 0);
            }
            __syncwarp();

            // 4. Wait until all same-nvl RDMA peers finished dispatch quiet (lane-distributed).
            for (int src = lane_id; src < num_rdma_ranks_local; src += 32) {
                while (ld_acquire_sys_global(&state->rdma_reuse_dispatch_quiet_done[src]) == 0)
                    __nanosleep(64);
            }
            __syncwarp();

            // 5. Clean this GPU's combine RDMA metadata (data-after head/tail/meta int region).
            //    Mirrors get_rdma_clean_meta(combine_hidden_int4, 0, 0, num_topk, ...) used by the
            //    host combine cached_notify. num_bytes_per_token already == combine layout bytes.
            {
                const int recv_tokens = num_max_rdma_chunked_recv_tokens;  // combine recv capacity
                const long long clean_offset =
                    (long long)num_bytes_per_token * recv_tokens * num_rdma_ranks_local * 2 * num_logical_channels
                    / (long long)sizeof(int);
                const int clean_count =
                    (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks_local * 2 * num_logical_channels;
                int* clean_p = static_cast<int*>(state->combine_rdma_buffer_ptr);
                for (int i = lane_id; i < clean_count; i += 32)
                    clean_p[clean_offset + i] = 0;
            }
            __syncwarp();
            if (lane_id == 0)
                __threadfence_system();  // make the metadata clean visible before publishing clear-done
            __syncwarp();

            // 6. Publish combine-clear-done into every same-nvl peer's mailbox (and locally).
            if (lane_id == 0)
                st_release_sys_global(&state->rdma_reuse_combine_clear_done[rdma_rank], 1);
            for (int dr = lane_id; dr < num_rdma_ranks_local; dr += 32) {
                if (dr == rdma_rank) continue;
                const int dst_pe = translate_dst_rdma_rank<kLowLatencyMode>(dr, nvl_rank);
                nvshmemi_ibgda_rma_p(&state->rdma_reuse_combine_clear_done[rdma_rank], 1, dst_pe, 0);
            }
            __syncwarp();

            // 7. Wait until all same-nvl RDMA peers finished the combine metadata clean.
            for (int src = lane_id; src < num_rdma_ranks_local; src += 32) {
                while (ld_acquire_sys_global(&state->rdma_reuse_combine_clear_done[src]) == 0)
                    __nanosleep(64);
            }
            __syncwarp();

            // Release the rest of the combine SMs.
            if (lane_id == 0)
                st_release_sys_global(state->rdma_reuse_prelude_done, 1);
        }
        while (ld_acquire_sys_global(state->rdma_reuse_prelude_done) == 0)
            __nanosleep(64);
        __syncthreads();
    }

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
#if MK_PERF_TRACE_ENABLED
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
                    __threadfence_system(); trap();
                }
                __nanosleep(32);
            }
#if MK_PERF_TRACE_ENABLED
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
                    auto current_head = __ldg(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
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
                        int stored_head = __ldg(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
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
        // Match original DeepEP's TMA batch normalize instead of per-token system-acquire loads.
        {
            constexpr int tma_batch_size = kNumCombineTMABytesPerSenderWarp - static_cast<int>(sizeof(uint64_t));
            constexpr int num_head_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS;
            constexpr int num_tokens_per_batch = tma_batch_size / num_head_bytes_per_token;
            EP_STATIC_ASSERT(num_head_bytes_per_token % 16 == 0, "num_head_bytes_per_token should be divisible by 16");

            extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
            auto tma_buffer = smem_tma_buffer;
            auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);
            uint32_t tma_phase = 0;
            if (elect_one_sync()) {
                tma_store_wait<0>();
                mbarrier_init(tma_mbarrier, 1);
                fence_barrier_init();
                EP_DEVICE_ASSERT(tma_batch_size + static_cast<int>(sizeof(uint64_t)) <= kNumCombineTMABytesPerSenderWarp);
            }
            __syncwarp();

            for (int dst_rdma_rank = 0; dst_rdma_rank < kNumRDMARanks_C; ++dst_rdma_rank) {
                int rdma_prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
                int channel_end = ld_nc_global(rdma_channel_prefix_matrix + rdma_prefix_idx);
                int channel_count = ld_nc_global(state->combine_rdma_channel_token_count + rdma_prefix_idx);
                int channel_start = channel_end - channel_count;
                int rank_shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                int rdma_token_start = rank_shift + channel_start;
                int rdma_token_end = rank_shift + channel_end;
                int nvl_head_capacity = state->combine_nvl_head_stride / NUM_MAX_NVL_PEERS;
                EP_DEVICE_ASSERT(channel_count >= 0 and channel_end >= channel_start);
                EP_DEVICE_ASSERT(rdma_token_start >= 0 and rdma_token_end >= rdma_token_start and rdma_token_end <= nvl_head_capacity);

                int last_head = 1 << 25;
                for (int batch_end_idx = rdma_token_end; batch_end_idx > rdma_token_start; batch_end_idx -= num_tokens_per_batch) {
                    int batch_start_idx = max(rdma_token_start, batch_end_idx - num_tokens_per_batch);
                    int batch_bytes = (batch_end_idx - batch_start_idx) * num_head_bytes_per_token;

                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer,
                                    combined_nvl_head_base + batch_start_idx * NUM_MAX_NVL_PEERS,
                                    tma_mbarrier,
                                    batch_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, batch_bytes);
                    }
                    mbarrier_wait(tma_mbarrier, tma_phase);
                    __syncwarp();

                    for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
                        if (lane_id < NUM_MAX_NVL_PEERS) {
                            auto current_head = reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id];
                            int normalized_head = current_head;
                            if (current_head < 0) {
                                normalized_head = -last_head - 1;
                                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id] = normalized_head;
                            } else {
                                last_head = current_head;
                            }
#ifdef MK_TOKEN_TRACE
                            printf("[MK-DIAG][COMBINE-NVL-HEAD-NORM] rank=%d rdma_rank=%d nvl_rank=%d physical_ch=%d logical_ch=%d dst_rdma=%d token=%d lane_nvl=%d raw=%d norm=%d stored=%d token_range=[%d,%d) batch_range=[%d,%d) head_ptr=%p\n",
                                   state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id, dst_rdma_rank,
                                   token_idx, lane_id, current_head, normalized_head,
                                   reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id],
                                   rdma_token_start, rdma_token_end, batch_start_idx, batch_end_idx,
                                   combined_nvl_head_base + token_idx * NUM_MAX_NVL_PEERS + lane_id);
#endif
                        }
                    }
                    tma_store_fence();
                    __syncwarp();

                    if (elect_one_sync())
                        tma_store_1d(tma_buffer,
                                     combined_nvl_head_base + batch_start_idx * NUM_MAX_NVL_PEERS,
                                     batch_bytes);
                    tma_store_wait<0>();
                    __syncwarp();
                }
            }
        }

        // Step 4: Signal normalization complete
        tma_store_wait<0>();
        __threadfence_system();
        if (lane_id == 0) {
            st_release_sys_global(&state->channel_normalized[logical_channel_id], 1);
#if MK_PERF_TRACE_ENABLED
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
                    __threadfence_system(); trap();
                }
                __nanosleep(32);
            }
#if MK_PERF_TRACE_ENABLED
            if (is_forwarder_sm && thread_id == 0) {
                int trace_idx = (logical_channel_id * 2 + 1) * MegaKernelState::MK_PERF_NUM_LCH_PHASES;
                state->perf_combine_lch_ts[trace_idx + 2] = globaltimer_ns();
            }
#endif
        }
        __syncwarp();
    }

#if MK_PERF_TRACE_ENABLED
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
            token_start_idx = ld_nc_global(gbl_channel_prefix_matrix + nvl_sender_prefix_idx);
            nvl_sender_count = ld_nc_global(state->combine_gbl_channel_token_count + nvl_sender_prefix_idx);
            token_end_idx = token_start_idx + nvl_sender_count;
            EP_DEVICE_ASSERT(token_start_idx >= 0 and nvl_sender_count >= 0 and token_end_idx <= num_tokens);
        }
        __syncwarp();

        auto& cached_channel_head_idx = combine_nvl_sender_cached_channel_head_idx;
        auto& cached_channel_tail_idx = combine_nvl_sender_cached_channel_tail_idx;

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
                    __threadfence_system(); trap();
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
                    // combine_token_ready: nhits==1 is published by compute, nhits>1 by gather
                    // after reduce. Every combine token is expected to have at least one local hit.
                    __syncwarp();
                    // NOTE: DeepEP's combine NVL sender forwards every token in the
                    // [token_start_idx, token_end_idx) range unconditionally. The range from
                    // gbl_channel_prefix_matrix already encodes exactly which tokens belong to
                    // (src_rdma_lane, dst_nvl). The dispatch-time meta NVL bits describe a
                    // different (dispatch) routing and must NOT be used to filter combine sends;
                    // doing so drops legitimately-assigned tokens and stalls the NVL queue tail,
                    // deadlocking the destination combine forwarder.
                    int dst_slot_idx = 0;
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
#if MK_PERF_TRACE_ARGS
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
#if MK_PERF_TRACE_ARGS
                    if (lane_id == 0) {
                        int64_t now = globaltimer_ns();
                        comb_tma_wait_ns = now - phase_start_ns;
                        phase_start_ns = now;
                    }
#endif
                    // Gather-ready means combine_input already contains the same-rank output:
                    // compute writes nhits==1 tokens directly, gather_worker reduces nhits>1 tokens.
#if MK_PERF_TRACE_ARGS
                    const int nh = ld_acquire_global(&state->token_nhits[token_idx]);
                    const bool is_single_hit = (nh == 1);
#endif
                    auto gather_wait_start = clock64();
                    int ready = 0;
                    while (true) {
                        if (lane_id == 0)
                            ready = ld_acquire_global(&state->combine_token_ready[token_idx]);
                        ready = __shfl_sync(0xffffffff, ready, 0);
                        if (ready == 1)
                            break;

                        if (lane_id == 0 && clock64() - gather_wait_start > NUM_TIMEOUT_CYCLES) {
                            if (timeout_log_once(state, kTimeoutLogComputeReady)) {
                                int nh = ld_acquire_global(&state->token_nhits[token_idx]);
                                int expected = ld_acquire_global(&state->token_compute_expected[token_idx]);
                                int done = ld_acquire_global(&state->token_done_count[token_idx]);
                                int ready_snapshot = ld_acquire_global(&state->combine_token_ready[token_idx]);
                                printf("MK combine gather-semaphore timeout, rank=%d token=%lld nh=%d expected=%d done=%d ready=%d\n",
                                       state->rank, (long long)token_idx, nh, expected, done, ready_snapshot);
                            }
                            __threadfence_system(); trap();
                        }
                        __nanosleep(32);
                    }
                    {
                        const int4* token_out_i4 = reinterpret_cast<const int4*>(state->combine_input);
                        // No per-slot wait here. combine_token_ready is the only readiness signal.
                        __syncwarp();
                        if (state->rank == 0 && token_idx == 0 && lane_id == 0) {
                            const __nv_bfloat16* combine_values =
                                state->combine_input + (int64_t)token_idx * hidden;
                            float sum_abs = 0.0f;
                            float max_abs = 0.0f;
                            int nonzero = 0;
                            const int sample = min(hidden, 256);
                            for (int i = 0; i < sample; ++i) {
                                float value = fabsf(__bfloat162float(combine_values[i]));
                                sum_abs += value;
                                max_abs = max(max_abs, value);
                                nonzero += value != 0.0f;
                            }
                            // printf("[MK-BWD-TRACE][COMBINE-READ] token=%lld sample=%d sum_abs=%e max_abs=%e nonzero=%d\n",
                            //        (long long)token_idx, sample, sum_abs, max_abs, nonzero);
                        }
#if MK_PERF_TRACE_ARGS
                        if (lane_id == 0) {
                            int64_t now = globaltimer_ns();
                            comb_wait_ready_ns = now - phase_start_ns;
                            if (is_single_hit)
                                comb_wait_ready_single_ns = comb_wait_ready_ns;
                            else
                                comb_wait_ready_multi_ns = comb_wait_ready_ns;
                            comb_wait_top_ns = comb_wait_ready_ns;
                            comb_wait_top_slot = ld_nc_global(&state->token_slot_list[token_idx * state->num_topk]);
                            comb_wait_top_from_flush = -1;
                            phase_start_ns = now;
                        }
#endif
                        if (lane_id == 0) {
                            tma_load_1d(tma_buffer,
                                        token_out_i4 + token_idx * hidden_int4,
                                        tma_mbarrier, hidden_bytes, false);
                            mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
                        }
                        __syncwarp();
                        mbarrier_wait(tma_mbarrier, tma_phase);
                    }
                    __syncwarp();
#if MK_PERF_TRACE_ARGS
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

                    tma_store_fence();
                    __syncwarp();
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);
#if MK_PERF_TRACE_ARGS
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
                            state->perf_comb_wait_top_nhits[acc_idx] = ld_acquire_global(&state->token_nhits[token_idx]);
                            state->perf_comb_wait_top_priority_deps[acc_idx] = ld_acquire_global(&state->token_priority_dep_count[token_idx]);
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
            int channel_end = ld_nc_global(rdma_channel_prefix_matrix + rdma_prefix_idx);
            int num_tokens_to_combine = ld_nc_global(state->combine_rdma_channel_token_count + rdma_prefix_idx);
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
                        __threadfence_system(); trap();
                    }
                }
                sync_large_warp();

                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder_C) {
                    // Read normalized head (original DeepEP logic)
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS) {
                        int lane_raw_head = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
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
                            int head0 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS);
                            int head1 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 1);
                            int lane_raw_head = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                            int local_head = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + nvl_rank);
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
                            __threadfence_system(); trap();
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
                            heads[i] = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + i);
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
                        int head0 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS);
                        int head1 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 1);
                        int head2 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 2);
                        int head3 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 3);
                        int local_head = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + nvl_rank);
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
                            heads[i] = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + i);
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
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
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
                        __threadfence_system(); trap();
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
                           ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks_C),
                           ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks_C + 1),
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
                __threadfence_system(); trap();
            }
            __nanosleep(32);
        }
    }
    __syncthreads();
#if MK_PERF_TRACE_ENABLED
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
// Pops scheduler-built gather tasks. Each task owns a compact batch of ready
// nhits > 1 tokens in gather_ready_queue; nhits == 1 is signaled by compute.
// ============================================================================


template <ComputeDType kComputeDType>
__device__ void combine_precompute_worker(
    int sm_id,
    int combine_sm_idx,
    int num_combine_sms,
    MegaKernelState* state,
    uint8_t* smem_buffer
) {
    const int post_group_count =
        (state->num_compute_sms + state->num_dispatch_sms + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    compute_worker_core<kComputeDType, true>(
        sm_id, combine_sm_idx, num_combine_sms, state, post_group_count, smem_buffer);
}

__device__ __forceinline__ void gather_accum_bf162(float2& acc, __nv_bfloat162 value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t packed = *reinterpret_cast<uint32_t*>(&value);
    uint16_t lo = static_cast<uint16_t>(packed & 0xffffu);
    uint16_t hi = static_cast<uint16_t>(packed >> 16);
    asm volatile("add.rn.f32.bf16 %0, %1, %0;" : "+f"(acc.x) : "h"(lo));
    asm volatile("add.rn.f32.bf16 %0, %1, %0;" : "+f"(acc.y) : "h"(hi));
#else
    float2 v = __bfloat1622float2(value);
    acc.x += v.x;
    acc.y += v.y;
#endif
}

__device__ __forceinline__ void gather_accum_int4(float2* acc, int4 raw) {
    const __nv_bfloat162* bv2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int p = 0; p < 4; ++p)
        gather_accum_bf162(acc[p], bv2[p]);
}

__device__ void gather_worker(MegaKernelState* state, int gather_sm_idx) {
    const int tid = threadIdx.x;
    const int total_tokens = state->combine_num_tokens;
    if (total_tokens == 0) return;

    __shared__ int s_token_base;
    __shared__ int s_batch_count;
    __shared__ int s_has_task;

    while (true) {
#if MK_PERF_TRACE_ENABLED
        int64_t perf_start_ns = globaltimer_ns();
        int64_t perf_end_ns = 0;
#endif
#if MK_PERF_TRACE_ARGS
        int64_t perf_scan_done_ns = 0;
        int64_t perf_reduce_done_ns = 0;
#endif
        if (tid == 0) {
            s_has_task = 0;
            s_token_base = 0;
            s_batch_count = 0;
            while (ld_acquire_global(state->combine_all_done) == 0) {
                int head = ld_acquire_global(state->gather_ready_head);
                int tail = ld_acquire_global(state->gather_ready_tail);
                if (head < tail && atomicCAS(state->gather_ready_head, head, head + 1) == head) {
                    s_token_base = ld_acquire_global(&state->gather_task_tokens[head]);
                    s_batch_count = ld_acquire_global(&state->gather_task_nhits[head]);
                    s_has_task = (s_batch_count > 0) ? 1 : 0;
                    break;
                }
                __nanosleep(64);
            }
        }
        __syncthreads();
        if (!s_has_task)
            break;
#if MK_PERF_TRACE_ARGS
        if (tid == 0)
            perf_scan_done_ns = globaltimer_ns();
#endif

        const int token_base = s_token_base;
        const int batch_count = s_batch_count;
        constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(__nv_bfloat16);
        constexpr int kBfloat162PerInt4 = kElemsPerInt4 / 2;
        const int hidden = state->combine_hidden;
        const int hidden_int4 = hidden / kElemsPerInt4;
        const int num_topk = state->num_topk;
        int4* slot_base_i4 = reinterpret_cast<int4*>(state->compute_output_slot);
        int4* token_out_i4 = reinterpret_cast<int4*>(state->combine_input);
#if MK_PERF_TRACE_ARGS
        int first_token = -1;
        int last_token = -1;
        int nhit_sum = 0;
        if (tid == 0) {
            for (int batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
                const int token_idx = state->gather_ready_queue[token_base + batch_idx];
                const int nhits = ld_acquire_global(&state->token_nhits[token_idx]);
                if (batch_idx == 0)
                    first_token = token_idx;
                last_token = token_idx;
                nhit_sum += nhits;
            }
        }
#endif

        constexpr int kGatherChunkInt4 = 128;
        constexpr int kGatherVecsPerLane = kGatherChunkInt4 / 32;
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        const int num_warps = (blockDim.x + 31) >> 5;
        const int chunks_per_token = (hidden_int4 + kGatherChunkInt4 - 1) / kGatherChunkInt4;
        const int total_work = batch_count * chunks_per_token;

        for (int work = warp_id; work < total_work; work += num_warps) {
            const int batch_idx = work / chunks_per_token;
            const int chunk = work - batch_idx * chunks_per_token;
            const int token_idx = state->gather_ready_queue[token_base + batch_idx];
            const int nhits = ld_acquire_global(&state->token_nhits[token_idx]);
            const int chunk_base = chunk * kGatherChunkInt4;
            const int chunk_end = min(chunk_base + kGatherChunkInt4, hidden_int4);
            float2 acc[kGatherVecsPerLane][kBfloat162PerInt4];

            #pragma unroll
            for (int j = 0; j < kGatherVecsPerLane; ++j) {
                #pragma unroll
                for (int p = 0; p < kBfloat162PerInt4; ++p)
                    acc[j][p] = make_float2(0.0f, 0.0f);
            }

            for (int k = 0; k < nhits; ++k) {
                int slot = state->token_slot_list[token_idx * num_topk + k];
                #pragma unroll
                for (int j = 0; j < kGatherVecsPerLane; ++j) {
                    const int vi = chunk_base + lane_id + j * 32;
                    if (vi < chunk_end) {
                        int4 raw;
                        asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                            : "l"(slot_base_i4 + (int64_t)slot * hidden_int4 + vi));
                        gather_accum_int4(acc[j], raw);
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < kGatherVecsPerLane; ++j) {
                const int vi = chunk_base + lane_id + j * 32;
                if (vi < chunk_end) {
                    int4 packed;
                    __nv_bfloat162* pv2 = reinterpret_cast<__nv_bfloat162*>(&packed);
                    #pragma unroll
                    for (int p = 0; p < kBfloat162PerInt4; ++p)
                        pv2[p] = __float22bfloat162_rn(acc[j][p]);
                    token_out_i4[(int64_t)token_idx * hidden_int4 + vi] = packed;
                }
            }
        }

        __syncthreads();
        __threadfence();
        __syncthreads();
        if (state->rank == 0 && gather_sm_idx == 0 && tid == 0 && batch_count > 0) {
            const int token_idx = state->gather_ready_queue[token_base];
            const __nv_bfloat16* gathered = state->combine_input + (int64_t)token_idx * hidden;
            float sum_abs = 0.0f;
            float max_abs = 0.0f;
            int nonzero = 0;
            const int sample = min(hidden, 256);
            for (int i = 0; i < sample; ++i) {
                float value = fabsf(__bfloat162float(gathered[i]));
                sum_abs += value;
                max_abs = max(max_abs, value);
                nonzero += value != 0.0f;
            }
            // printf("[MK-BWD-TRACE][GATHER-OUT] token=%d nhits=%d sample=%d sum_abs=%e max_abs=%e nonzero=%d\n",
            //        token_idx, state->token_nhits[token_idx], sample, sum_abs, max_abs, nonzero);
        }
        for (int batch_idx = tid; batch_idx < batch_count; batch_idx += blockDim.x) {
            const int token_idx = state->gather_ready_queue[token_base + batch_idx];
            atomicExch(&state->combine_token_ready[token_idx], 1);
        }
        __syncthreads();
#if MK_PERF_TRACE_ENABLED
        if (tid == 0) {
            perf_end_ns = globaltimer_ns();
            int slot = atomicAdd(state->perf_gather_task_count, 1);
            if (slot < state->max_compute_tasks) {
                int64_t* rec = state->perf_gather_task + (int64_t)slot * MegaKernelState::MK_PERF_NUM_GATHER_FIELDS;
                rec[0] = perf_start_ns;
                rec[1] = perf_end_ns;
                rec[2] = gather_sm_idx;
                rec[3] = batch_count;
#if MK_PERF_TRACE_ARGS
                perf_reduce_done_ns = perf_end_ns;
                rec[4] = first_token;
                rec[5] = last_token;
                rec[6] = nhit_sum;
                rec[7] = perf_scan_done_ns;
                rec[8] = perf_reduce_done_ns;
                rec[9] = perf_reduce_done_ns;
#endif
            }
        }
#endif
    }
}

// ============================================================================
// Main MegaKernel Entry Point
// ============================================================================

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
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

    // Role-specific workers reuse the single dynamic shared-memory allocation.
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Determine SM role based on blockIdx.x
    // Layout: [Dispatch] [Combine] [Scheduler] [Compute groups] [Gather]
    SmRole role;
    int role_idx;

    const int compute_begin = num_dispatch_sms + num_combine_sms + COMPUTE_SCHEDULER_SMS;
    const int gather_begin = compute_begin + num_compute_sms;
    const int total_compute_sms_after_dispatch = num_compute_sms + num_dispatch_sms;
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
            dispatch_worker_v2<kNumRDMARanks, kStage, kComputeDType, false>(sm_id, role_idx, state);
            break;

        case SmRole::kCombine:
            combine_precompute_worker<kComputeDType>(sm_id, role_idx, num_combine_sms, state, smem_buffer);
            combine_worker_v2<kNumRDMARanks, kStage>(role_idx, state);
            break;

        case SmRole::kScheduler:
            compute_scheduler_worker(state, role_idx, COMPUTE_SCHEDULER_SMS);
            break;

        case SmRole::kCompute:
            MK_FORWARD_COMPUTE_WORKER(
                kComputeDType, sm_id, role_idx,
                total_compute_sms_after_dispatch, state, smem_buffer);
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

#if MK_PERF_TRACE_ENABLED
static void dump_perf_trace_perfetto(const MegaKernelState& host_state, int total_sms, const char* trace_phase);
#endif

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
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

    // printf("[MK-HOST][LAUNCH] device_state=%p total_sms=%d num_ranks=%d kNumRDMARanks=%d stage=%d compute_dtype=%d block_threads=%d smem_size=%d stream=%p\n",
        //    device_state, total_sms, num_ranks, kNumRDMARanks, kStage, static_cast<int>(kComputeDType), kThreads, smem_size, stream);
    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(moe_megakernel_v7<kNumRDMARanks, kStage, kComputeDType>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
        // printf("[MK-HOST][LAUNCH] set dynamic smem attribute=%d\n", smem_size);
    }

    constexpr int num_gather_sms = GATHER_SMS;
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
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_megakernel_v7<kNumRDMARanks, kStage, kComputeDType>, device_state));
#else
    moe_megakernel_v7<kNumRDMARanks, kStage, kComputeDType><<<launch_total_sms, kThreads, smem_size, stream>>>(device_state);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void launch_megakernel_v7(
    MegaKernelState* device_state,
    const MegaKernelState* host_state,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream
) {
    MegaKernelState copied_host_state;
    const MegaKernelState* launcher_host_state = host_state;
    if (launcher_host_state == nullptr) {
        CUDA_CHECK(cudaMemcpy(&copied_host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
        launcher_host_state = &copied_host_state;
    }
    const int num_ranks = launcher_host_state->num_ranks;
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);

#define MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, kStage, kComputeDType) \
    launch_megakernel_v7_case<kNumRDMARanks, kStage, kComputeDType>(device_state, *launcher_host_state, total_sms, smem_size, stream); \
    break

#define MEGAKERNEL_LAUNCH_CASE_WITH_DTYPE(kNumRDMARanks, kComputeDType) \
    switch (stage) { \
        case 1: MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, 1, kComputeDType); \
        case 2: MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, 2, kComputeDType); \
        default: EP_HOST_ASSERT(false && "Unsupported megakernel stage"); \
    } \
    break

#define MEGAKERNEL_LAUNCH_CASE(kNumRDMARanks) \
    switch (compute_dtype) { \
        case ComputeDType::kBF16: MEGAKERNEL_LAUNCH_CASE_WITH_DTYPE(kNumRDMARanks, ComputeDType::kBF16); \
        case ComputeDType::kFP8E4M3: MEGAKERNEL_LAUNCH_CASE_WITH_DTYPE(kNumRDMARanks, ComputeDType::kFP8E4M3); \
        default: EP_HOST_ASSERT(false && "Unsupported megakernel compute dtype"); \
    } \
    break

    SWITCH_RDMA_RANKS(MEGAKERNEL_LAUNCH_CASE);

#undef MEGAKERNEL_LAUNCH_CASE
#undef MEGAKERNEL_LAUNCH_CASE_WITH_DTYPE
#undef MEGAKERNEL_LAUNCH_STAGE_CASE

#if MK_PERF_TRACE_ENABLED
    dump_perf_trace_perfetto(*launcher_host_state, total_sms, "forward");
#endif
}

#if MK_PERF_TRACE_ENABLED
static void dump_perf_trace_perfetto(const MegaKernelState& host_state, int total_sms, const char* trace_phase) {

    static int forward_trace_iter = 0;
    static int backward_trace_iter = 0;
    const bool is_backward_trace = trace_phase != nullptr && trace_phase[0] == 'b';
    const int trace_iter = is_backward_trace ? backward_trace_iter++ : forward_trace_iter++;
    const char* trace_retention = std::getenv("MK_PERF_TRACE_RETENTION");
    const bool keep_all_trace_iters = trace_retention != nullptr && trace_retention[0] == 'a';

    int num_logical_channels = host_state.num_logical_channels;
    constexpr int NLP = MegaKernelState::MK_PERF_NUM_LCH_PHASES;

    std::vector<int64_t> dispatch_lch_ts(num_logical_channels * 2 * NLP);
    std::vector<int64_t> combine_lch_ts(num_logical_channels * 2 * NLP);
    CUDA_CHECK(cudaMemcpy(dispatch_lch_ts.data(), host_state.perf_dispatch_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(combine_lch_ts.data(), host_state.perf_combine_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t), cudaMemcpyDeviceToHost));

    constexpr int NCF = MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS;
    int compute_task_count = host_state.max_compute_tasks;
    std::vector<int64_t> compute_task(compute_task_count > 0 ? (size_t)compute_task_count * NCF : 1);
    if (compute_task_count > 0)
        CUDA_CHECK(cudaMemcpy(compute_task.data(), host_state.perf_compute_task, (size_t)compute_task_count * NCF * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int gather_task_count = 0;
    constexpr int NGF = MegaKernelState::MK_PERF_NUM_GATHER_FIELDS;
    CUDA_CHECK(cudaMemcpy(&gather_task_count, host_state.perf_gather_task_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (gather_task_count > host_state.max_compute_tasks)
        gather_task_count = host_state.max_compute_tasks;
    std::vector<int64_t> gather_task(gather_task_count > 0 ? (size_t)gather_task_count * NGF : 1);
    if (gather_task_count > 0)
        CUDA_CHECK(cudaMemcpy(gather_task.data(), host_state.perf_gather_task, (size_t)gather_task_count * NGF * sizeof(int64_t), cudaMemcpyDeviceToHost));

    const int async_pub_n = host_state.num_pub_warps_total;
    std::vector<int64_t> async_pub_start(async_pub_n);
    std::vector<int64_t> async_pub_end(async_pub_n);
    int64_t async_publish_all_done_ts = 0;
    if (async_pub_n > 0) {
        const size_t async_pub_bytes = (size_t)async_pub_n * sizeof(int64_t);
        CUDA_CHECK(cudaMemcpy(async_pub_start.data(), host_state.perf_async_pub_start_ts, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_end.data(), host_state.perf_async_pub_end_ts, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&async_publish_all_done_ts, host_state.perf_async_publish_all_done_ts, sizeof(int64_t), cudaMemcpyDeviceToHost));
    }
    const int logical_channels_per_physical = host_state.num_dispatch_channels > 0 ?
        num_logical_channels / host_state.num_dispatch_channels : 1;

#if !MK_PERF_TRACE_ARGS
    std::vector<int64_t> sched_ts(2);
    CUDA_CHECK(cudaMemcpy(sched_ts.data(), host_state.perf_sched_ts, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int64_t base_ts = std::numeric_limits<int64_t>::max();
    for (int i = 0; i < num_logical_channels * 2 * NLP; ++i) {
        int64_t trace_ts = dispatch_lch_ts[i];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    for (int i = 0; i < num_logical_channels * 2 * NLP; ++i) {
        int64_t trace_ts = combine_lch_ts[i];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    for (int t = 0; t < compute_task_count; ++t) {
        int64_t* rec = &compute_task[(size_t)t * NCF];
        if (rec[26] == 0) continue;
        int64_t trace_ts = rec[0];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    for (int t = 0; t < gather_task_count; ++t) {
        int64_t trace_ts = gather_task[(size_t)t * NGF];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    for (int i = 0; i < async_pub_n; ++i) {
        int64_t trace_ts = async_pub_start[i];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    for (int i = 0; i < 2; ++i) {
        int64_t trace_ts = sched_ts[i];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    if (base_ts == std::numeric_limits<int64_t>::max()) base_ts = 0;

    char filename[256];
    if (keep_all_trace_iters) {
        snprintf(filename, sizeof(filename), "mk_perf_trace_rank%d_%s_iter%d.json", host_state.rank, trace_phase, trace_iter);
    } else {
        snprintf(filename, sizeof(filename), "mk_perf_trace_rank%d_%s.json", host_state.rank, trace_phase);
    }
    FILE* f = fopen(filename, "w");

    if (!f) { printf("[MK-PERF] Failed to open %s\n", filename); return; }
    // printf("[MK-PERF] rank=%d base_ts_ns=%lld\n", host_state.rank, (long long)base_ts);

    fprintf(f, "[\n");
    bool first = true;
    auto emit_comma = [&]() {
        if (!first) fprintf(f, ",\n");
        first = false;
    };
    auto emit_event = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid);
    };
    auto emit_colored_event = [&](const char* name, const char* cat, const char* color,
                                  int64_t start, int64_t end, int pid, int tid) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"cname\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d}",
                name, cat, color, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid);
    };

    emit_comma();
    fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":%d,\"args\":{\"name\":\"rank %d\"}}",
            host_state.rank, host_state.rank);
    emit_comma();
    fprintf(f, "{\"name\":\"mk_base_ts_ns\",\"ph\":\"M\",\"pid\":%d,\"args\":{\"base_ts_ns\":%lld}}",
            host_state.rank, (long long)base_ts);

    int lch_tid_base = 0;
    int compute_tid_base = num_logical_channels * 5;
    const int num_compute_groups = host_state.num_compute_groups;
    for (int group_id = 0; group_id < num_compute_groups; ++group_id) {
        int tid = compute_tid_base + group_id;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"compute_group_%d\"}}",
                host_state.rank, tid, group_id);
    }
    {
        int tid = compute_tid_base + num_compute_groups;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"gather_task\"}}",
                host_state.rank, tid);
    }
    int scheduler_tid = num_logical_channels * 5 + 50;
    emit_comma();
    fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
               "\"args\":{\"name\":\"scheduler_bridge\"}}",
            host_state.rank, scheduler_tid);

    for (int logical_channel_id = 0; logical_channel_id < num_logical_channels; ++logical_channel_id) {
        int dispatch_sender_tid = lch_tid_base + logical_channel_id * 5;
        int dispatch_forwarder_tid = dispatch_sender_tid + 1;
        int async_publish_tid = dispatch_sender_tid + 2;
        int combine_sender_tid = dispatch_sender_tid + 3;
        int combine_forwarder_tid = dispatch_sender_tid + 4;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"dispatch_sender_lch_%d\"}}",
                host_state.rank, dispatch_sender_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"dispatch_forwarder_lch_%d\"}}",
                host_state.rank, dispatch_forwarder_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"dispatch_async_publish_lch_%d\"}}",
                host_state.rank, async_publish_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"combine_sender_lch_%d\"}}",
                host_state.rank, combine_sender_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"combine_forwarder_lch_%d\"}}",
                host_state.rank, combine_forwarder_tid, logical_channel_id);

        int64_t* dsp = &dispatch_lch_ts[(logical_channel_id * 2) * NLP];
        emit_event("dispatch_sender_work", "dispatch_sender_lch", dsp[0], dsp[1], host_state.rank, dispatch_sender_tid);
        emit_event("dispatch_sender_channel_barrier", "dispatch_sender_lch", dsp[1], dsp[2], host_state.rank, dispatch_sender_tid);
        emit_event("dispatch_sender_round_barrier", "dispatch_sender_lch", dsp[2], dsp[3], host_state.rank, dispatch_sender_tid);

        int64_t* dfp = &dispatch_lch_ts[(logical_channel_id * 2 + 1) * NLP];
        emit_event("dispatch_forwarder_work", "dispatch_forwarder_lch", dfp[0], dfp[1], host_state.rank, dispatch_forwarder_tid);
        emit_event("dispatch_forwarder_channel_barrier", "dispatch_forwarder_lch", dfp[1], dfp[2], host_state.rank, dispatch_forwarder_tid);
        emit_event("dispatch_forwarder_round_barrier", "dispatch_forwarder_lch", dfp[2], dfp[3], host_state.rank, dispatch_forwarder_tid);

        int physical_channel_id = logical_channels_per_physical > 0 ?
            logical_channel_id / logical_channels_per_physical : logical_channel_id;
        int max_pw = -1;
        int64_t max_pub_duration_ns = -1;
        for (int src_nvl_rank = 0; src_nvl_rank < NUM_MAX_NVL_PEERS; ++src_nvl_rank) {
            int pw = physical_channel_id * NUM_MAX_NVL_PEERS + src_nvl_rank;
            if (pw >= async_pub_n || async_pub_start[pw] == 0 || async_pub_end[pw] == 0)
                continue;
            int64_t start = async_pub_start[pw];
            int64_t end = async_pub_end[pw];
            if (end <= start)
                end = start + 1;
            int64_t gate_end = end;
            if (async_publish_all_done_ts > start && async_publish_all_done_ts < end)
                gate_end = async_publish_all_done_ts;
            int64_t duration_ns = gate_end - start;
            if (duration_ns > max_pub_duration_ns) {
                max_pub_duration_ns = duration_ns;
                max_pw = pw;
            }
        }
        if (max_pw >= 0) {
            int64_t start = async_pub_start[max_pw];
            int64_t end = async_pub_end[max_pw];
            if (end <= start)
                end = start + 1;
            int64_t gate_end = end;
            if (async_publish_all_done_ts > start && async_publish_all_done_ts < end)
                gate_end = async_publish_all_done_ts;
            emit_event("dispatch_async_publish", "dispatch_async_publish", start, gate_end, host_state.rank, async_publish_tid);
        }

        int64_t* csp = &combine_lch_ts[(logical_channel_id * 2) * NLP];
        emit_event("sender_wait_dispatch_done", "combine_sender_lch", csp[0], csp[1], host_state.rank, combine_sender_tid);
        emit_event("sender_head_normalize", "combine_sender_lch", csp[1], csp[2], host_state.rank, combine_sender_tid);
        emit_event("sender_nvl_send_rdma_recv", "combine_sender_lch", csp[3], csp[4], host_state.rank, combine_sender_tid);

        int64_t* cfp = &combine_lch_ts[(logical_channel_id * 2 + 1) * NLP];
        emit_event("forwarder_wait_normalized", "combine_forwarder_lch", cfp[0], cfp[2], host_state.rank, combine_forwarder_tid);
        emit_event("forwarder_nvl_to_rdma", "combine_forwarder_lch", cfp[3], cfp[4], host_state.rank, combine_forwarder_tid);
    }

    emit_event("scheduler_bridge", "scheduler", sched_ts[0], sched_ts[1], host_state.rank, scheduler_tid);

    for (int t = 0; t < gather_task_count; ++t) {
        int64_t* rec = &gather_task[(size_t)t * NGF];
        int tid = compute_tid_base + num_compute_groups;
        emit_event("gather_task", "gather_sm", rec[0], rec[1], host_state.rank, tid);
    }
    for (int t = 0; t < compute_task_count; ++t) {
        int64_t* rec = &compute_task[(size_t)t * NCF];
        if (rec[26] == 0) continue;
        int group_id = static_cast<int>(rec[3]);
        int batch_size = static_cast<int>(rec[5]);
        int tid = compute_tid_base + group_id;
        const bool is_partial_batch = batch_size < COMPUTE_BATCH_SIZE;
        const char* task_color = is_partial_batch ? "terrible" : "good";
        const char* task_name = is_partial_batch ? "compute_task_partial" : "compute_task_full";
        emit_colored_event(task_name, "compute_group", task_color,
                           rec[0], rec[1], host_state.rank, tid);
    }

    fprintf(f, "\n]\n");
    fclose(f);
    // printf("[MK-PERF] Perfetto trace written to %s (%d logical channels, block events only)\n", filename, num_logical_channels);
    return;
#else
    constexpr bool emit_perf_args = MK_PERF_TRACE_ARGS;

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
    std::vector<int64_t> comb_wait_top_nhits(num_logical_channels * 2);
    std::vector<int64_t> comb_wait_top_priority_deps(num_logical_channels * 2);
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
    CUDA_CHECK(cudaMemcpy(comb_wait_top_nhits.data(), host_state.perf_comb_wait_top_nhits, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(comb_wait_top_priority_deps.data(), host_state.perf_comb_wait_top_priority_deps, num_logical_channels * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
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
    std::vector<int64_t> async_pub_wait_ring(async_pub_n);
    std::vector<int64_t> async_pub_poll(async_pub_n);
    std::vector<int64_t> async_pub_gap(async_pub_n);
    std::vector<int64_t> async_pub_start_gap(async_pub_n);
    std::vector<int64_t> async_pub_empty_gap(async_pub_n);
    std::vector<int64_t> async_pub_done_recheck(async_pub_n);
    std::vector<int64_t> async_pub_ring_load(async_pub_n);
    std::vector<int64_t> async_pub_head_release(async_pub_n);
    std::vector<int64_t> async_pub_syncwarp(async_pub_n);
    std::vector<int64_t> async_pub_batch_wall(async_pub_n);
    std::vector<int64_t> async_pub_batch_accounted(async_pub_n);
    std::vector<int64_t> async_pub_batch_unattributed(async_pub_n);
    std::vector<int64_t> async_pub_token_gap(async_pub_n);
    std::vector<int64_t> async_pub_helper_unattributed(async_pub_n);
    std::vector<int64_t> async_pub_finish(async_pub_n);
    std::vector<int64_t> async_pub_work(async_pub_n);
    std::vector<int64_t> async_pub_scan(async_pub_n);
    std::vector<int64_t> async_pub_atomic(async_pub_n);
    std::vector<int64_t> async_pub_fence(async_pub_n);
    std::vector<int64_t> async_pub_store(async_pub_n);
    std::vector<int64_t> async_pub_drain(async_pub_n);
    std::vector<int64_t> async_pub_tokens(async_pub_n);
    std::vector<int64_t> async_pub_batch_count(async_pub_n);
    std::vector<int64_t> async_pub_head_release_count(async_pub_n);
    std::vector<int64_t> async_pub_max_batch(async_pub_n);
    std::vector<int64_t> async_pub_local_hit_tokens(async_pub_n);
    std::vector<int64_t> async_pub_local_hits(async_pub_n);
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
    if (async_pub_n > 0) {
        const size_t async_pub_bytes = (size_t)async_pub_n * sizeof(int64_t);
        CUDA_CHECK(cudaMemcpy(async_pub_wait_ring.data(), host_state.perf_async_pub_wait_ring_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_poll.data(), host_state.perf_async_pub_poll_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_gap.data(), host_state.perf_async_pub_gap_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_start_gap.data(), host_state.perf_async_pub_start_gap_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_empty_gap.data(), host_state.perf_async_pub_empty_gap_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_done_recheck.data(), host_state.perf_async_pub_done_recheck_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_ring_load.data(), host_state.perf_async_pub_ring_load_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_head_release.data(), host_state.perf_async_pub_head_release_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_syncwarp.data(), host_state.perf_async_pub_syncwarp_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_batch_wall.data(), host_state.perf_async_pub_batch_wall_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_batch_accounted.data(), host_state.perf_async_pub_batch_accounted_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_batch_unattributed.data(), host_state.perf_async_pub_batch_unattributed_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_token_gap.data(), host_state.perf_async_pub_token_gap_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_helper_unattributed.data(), host_state.perf_async_pub_helper_unattributed_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_finish.data(), host_state.perf_async_pub_finish_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_work.data(), host_state.perf_async_pub_work_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_scan.data(), host_state.perf_async_pub_scan_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_atomic.data(), host_state.perf_async_pub_atomic_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_fence.data(), host_state.perf_async_pub_fence_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_store.data(), host_state.perf_async_pub_store_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_drain.data(), host_state.perf_async_pub_drain_ns, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_tokens.data(), host_state.perf_async_pub_tokens, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_batch_count.data(), host_state.perf_async_pub_batch_count, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_head_release_count.data(), host_state.perf_async_pub_head_release_count, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_max_batch.data(), host_state.perf_async_pub_max_batch, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_local_hit_tokens.data(), host_state.perf_async_pub_local_hit_tokens, async_pub_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(async_pub_local_hits.data(), host_state.perf_async_pub_local_hits, async_pub_bytes, cudaMemcpyDeviceToHost));
    }

    std::vector<int64_t> sched_ts(2);
    int64_t sched_scan_ns = 0, sched_enqueue_ns = 0, sched_idle_ns = 0;
    int64_t sched_priority_ns = 0, sched_normal_ns = 0, sched_tail_flush_ns = 0;
    int64_t sched_publish_total_ns = 0, sched_publish_wait_ns = 0, sched_publish_wait_max_ns = 0;
    int64_t sched_publish_wait_max_tail = -1, sched_publish_wait_max_visible_tail = -1;
    int64_t sched_priority_scan_tokens = 0, sched_priority_ready_tokens = 0;
    int64_t sched_priority_full_batch_hits = 0, sched_priority_batch_already_enqueued = 0;
    int64_t sched_priority_already_normal = 0, sched_priority_already_priority = 0, sched_priority_already_tail = 0;
    int64_t sched_normal_after_priority_ns = 0, sched_normal_after_priority_count = 0;
    int64_t sched_priority_not_full = 0, sched_normal_full_batch_enqueues = 0, sched_flush_tail_enqueues = 0;
    int64_t sched_queue_empty_count = 0, sched_queue_empty_after_dispatch_count = 0;
    int64_t sched_max_ready_tail_gap = 0, sched_stall_expert = -1, sched_stall_recv_count = 0;
    int64_t sched_stall_alloc_count = 0, sched_stall_enqueue_cursor = 0;
    int64_t sched_stall_first_unready_slot = -1, sched_stall_first_unready_ready = 0, sched_stall_dispatch_done = 0;
    int64_t sched_first_done_seen_ts = 0, sched_first_recv_count_advance_ts = 0;
    int64_t sched_first_recv_count_advance_expert = -1, sched_first_recv_count_advance_old = -1;
    int64_t sched_first_recv_count_advance_new = -1;
    int64_t sched_first_normal_enqueue_attempt_ts = 0, sched_first_normal_enqueue_success_ts = 0;
    int64_t sched_first_task_publish_ts = 0, sched_first_task_source = 0;
    int64_t sched_first_task_expert = -1, sched_first_task_batch = -1;
    int64_t sched_first_task_start_slot = -1, sched_first_task_num_tokens = 0;
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
    CUDA_CHECK(cudaMemcpy(&sched_priority_already_normal, host_state.perf_sched_priority_already_normal, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_already_priority, host_state.perf_sched_priority_already_priority, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_priority_already_tail, host_state.perf_sched_priority_already_tail, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_normal_after_priority_ns, host_state.perf_sched_normal_after_priority_ns, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_normal_after_priority_count, host_state.perf_sched_normal_after_priority_count, sizeof(int64_t), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaMemcpy(&sched_first_done_seen_ts, host_state.perf_sched_first_done_seen_ts, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_recv_count_advance_ts, host_state.perf_sched_first_recv_count_advance_ts, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_recv_count_advance_expert, host_state.perf_sched_first_recv_count_advance_expert, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_recv_count_advance_old, host_state.perf_sched_first_recv_count_advance_old, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_recv_count_advance_new, host_state.perf_sched_first_recv_count_advance_new, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_normal_enqueue_attempt_ts, host_state.perf_sched_first_normal_enqueue_attempt_ts, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_normal_enqueue_success_ts, host_state.perf_sched_first_normal_enqueue_success_ts, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_task_publish_ts, host_state.perf_sched_first_task_publish_ts, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_task_source, host_state.perf_sched_first_task_source, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_task_expert, host_state.perf_sched_first_task_expert, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_task_batch, host_state.perf_sched_first_task_batch, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_task_start_slot, host_state.perf_sched_first_task_start_slot, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sched_first_task_num_tokens, host_state.perf_sched_first_task_num_tokens, sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Root-cause diagnostic parallel arrays: multi-expert rows.
    const int diag_n = compute_task_count > 0 ? compute_task_count : 1;
    std::vector<int> diag_multi_rows(diag_n);
    std::vector<int> diag_task_has_multi(diag_n);
    const int queue_diag_n = host_state.max_compute_tasks > 0 ? host_state.max_compute_tasks : 1;
    std::vector<int64_t> q_publish(queue_diag_n), q_pop_start(queue_diag_n), q_pop_done(queue_diag_n),
        q_bcast_done(queue_diag_n), q_task_start(queue_diag_n), q_prev_end(queue_diag_n), q_prev_gap(queue_diag_n);
    std::vector<int> q_source(queue_diag_n), q_pop_attempts(queue_diag_n), q_cas_failures(queue_diag_n), q_group_id(queue_diag_n);
    if (compute_task_count > 0) {
        CUDA_CHECK(cudaMemcpy(diag_multi_rows.data(), host_state.perf_compute_multi_expert_rows, (size_t)compute_task_count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(diag_task_has_multi.data(), host_state.perf_compute_task_has_multi, (size_t)compute_task_count * sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (host_state.max_compute_tasks > 0) {
        const size_t qb64 = (size_t)host_state.max_compute_tasks * sizeof(int64_t);
        const size_t qb32 = (size_t)host_state.max_compute_tasks * sizeof(int);
        CUDA_CHECK(cudaMemcpy(q_publish.data(), host_state.perf_task_publish_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_source.data(), host_state.perf_task_source, qb32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_pop_start.data(), host_state.perf_task_pop_start_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_pop_done.data(), host_state.perf_task_pop_done_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_bcast_done.data(), host_state.perf_task_bcast_done_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_task_start.data(), host_state.perf_task_start_ts, qb64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_prev_end.data(), host_state.perf_task_prev_end_ts, qb64, cudaMemcpyDeviceToHost));
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
    for (int t = 0; t < compute_task_count; ++t) {
        int64_t* rec = &compute_task[(size_t)t * NCF];
        if (rec[26] == 0) continue;
        int64_t trace_ts = rec[0];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    for (int t = 0; t < gather_task_count; ++t) {
        int64_t trace_ts = gather_task[(size_t)t * NGF];
        if (trace_ts != 0 && trace_ts < base_ts) base_ts = trace_ts;
    }
    if (base_ts == std::numeric_limits<int64_t>::max()) base_ts = 0;

    // globaltimer_ns() already returns nanoseconds; convert to us for Perfetto.
    // Use per-rank min as base so trace starts near 0; for cross-rank comparison,
    // use aggregate_mk_perf_traces.py which aligns rank baselines.

    char filename[256];
    if (keep_all_trace_iters) {
        snprintf(filename, sizeof(filename), "mk_perf_trace_rank%d_%s_iter%d.json", host_state.rank, trace_phase, trace_iter);
    } else {
        snprintf(filename, sizeof(filename), "mk_perf_trace_rank%d_%s.json", host_state.rank, trace_phase);
    }
    FILE* f = fopen(filename, "w");

    if (!f) { printf("[MK-PERF] Failed to open %s\n", filename); return; }

    // Also write raw base_ts so post-processing can re-align ranks
    // printf("[MK-PERF] rank=%d base_ts_ns=%lld\n", host_state.rank, (long long)base_ts);

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
    auto emit_colored_event = [&](const char* name, const char* cat, const char* color,
                                  int64_t start, int64_t end, int pid, int tid) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        int64_t dur_ns = end - start;
        if (dur_ns <= 0) dur_ns = 1;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"cname\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d}",
                name, cat, color, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid);
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
                                         int64_t wait_top_nhits, int64_t wait_top_priority_deps,
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
                   "\"top_wait_nhits\":%lld,\"top_wait_priority_deps\":%lld,"
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
                static_cast<long long>(wait_top_nhits), static_cast<long long>(wait_top_priority_deps),
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
                                    int64_t priority_already_normal, int64_t priority_already_priority, int64_t priority_already_tail,
                                    int64_t normal_after_priority_ns, int64_t normal_after_priority_count,
                                    int64_t priority_not_full, int64_t normal_full_batch_enqueues,
                                    int64_t flush_tail_enqueues, int64_t queue_empty_count,
                                    int64_t queue_empty_after_dispatch_count, int64_t max_ready_tail_gap,
                                    int64_t stall_expert, int64_t stall_recv_count, int64_t stall_alloc_count,
                                    int64_t stall_enqueue_cursor, int64_t stall_first_unready_slot,
                                    int64_t stall_first_unready_ready, int64_t stall_dispatch_done) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_ns = start - base_ts;
        auto trace_ts_us = [&](int64_t value) -> double {
            return value > 0 ? (value - base_ts) / 1000.0 : -1.0;
        };
        auto sched_delta_us = [&](int64_t value) -> double {
            return value > 0 && sched_ts[0] > 0 ? (value - sched_ts[0]) / 1000.0 : -1.0;
        };
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
                   "\"priority_already_normal\":%lld,\"priority_already_priority\":%lld,\"priority_already_tail\":%lld,"
                   "\"normal_after_priority_us\":%.3f,\"normal_after_priority_count\":%lld,"
                   "\"priority_not_full\":%lld,\"normal_full_batch_enqueues\":%lld,\"flush_tail_enqueues\":%lld,"
                   "\"queue_empty_count\":%lld,\"queue_empty_after_dispatch_count\":%lld,"
                   "\"max_ready_tail_gap\":%lld,\"stall_expert\":%lld,\"stall_recv_count\":%lld,"
                   "\"stall_alloc_count\":%lld,\"stall_enqueue_cursor\":%lld,"
                   "\"stall_first_unready_slot\":%lld,\"stall_first_unready_ready\":%lld,"
                   "\"stall_dispatch_done\":%lld,"
                   "\"async_publish_all_done_ts_us\":%.3f,\"sched_start_to_publish_all_done_us\":%.3f,"
                   "\"first_done_seen_ts_us\":%.3f,\"sched_start_to_first_done_seen_us\":%.3f,"
                   "\"first_recv_count_advance_ts_us\":%.3f,\"sched_start_to_first_recv_count_advance_us\":%.3f,"
                   "\"first_recv_count_advance_expert\":%lld,\"first_recv_count_advance_old\":%lld,\"first_recv_count_advance_new\":%lld,"
                   "\"first_normal_enqueue_attempt_ts_us\":%.3f,\"sched_start_to_first_normal_enqueue_attempt_us\":%.3f,"
                   "\"first_normal_enqueue_success_ts_us\":%.3f,\"sched_start_to_first_normal_enqueue_success_us\":%.3f,"
                   "\"first_task_publish_ts_us\":%.3f,\"sched_start_to_first_task_publish_us\":%.3f,"
                   "\"first_task_source\":%lld,\"first_task_expert\":%lld,\"first_task_batch\":%lld,"
                   "\"first_task_start_slot\":%lld,\"first_task_num_tokens\":%lld}}",
                name, cat, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid,
                COMPUTE_SCHEDULER_SMS, scan_ns / 1000.0, enqueue_ns / 1000.0, idle_ns / 1000.0,
                priority_ns / 1000.0, normal_ns / 1000.0, tail_flush_ns / 1000.0,
                publish_total_ns / 1000.0, publish_wait_ns / 1000.0, publish_wait_max_ns / 1000.0,
                static_cast<long long>(publish_wait_max_tail), static_cast<long long>(publish_wait_max_visible_tail),
                static_cast<long long>(priority_scan_tokens), static_cast<long long>(priority_ready_tokens),
                static_cast<long long>(priority_full_batch_hits), static_cast<long long>(priority_batch_already_enqueued),
                static_cast<long long>(priority_already_normal), static_cast<long long>(priority_already_priority),
                static_cast<long long>(priority_already_tail), normal_after_priority_ns / 1000.0,
                static_cast<long long>(normal_after_priority_count), static_cast<long long>(priority_not_full),
                static_cast<long long>(normal_full_batch_enqueues),
                static_cast<long long>(flush_tail_enqueues), static_cast<long long>(queue_empty_count),
                static_cast<long long>(queue_empty_after_dispatch_count), static_cast<long long>(max_ready_tail_gap),
                static_cast<long long>(stall_expert), static_cast<long long>(stall_recv_count),
                static_cast<long long>(stall_alloc_count), static_cast<long long>(stall_enqueue_cursor),
                static_cast<long long>(stall_first_unready_slot), static_cast<long long>(stall_first_unready_ready),
                static_cast<long long>(stall_dispatch_done),
                trace_ts_us(async_publish_all_done_ts), sched_delta_us(async_publish_all_done_ts),
                trace_ts_us(sched_first_done_seen_ts), sched_delta_us(sched_first_done_seen_ts),
                trace_ts_us(sched_first_recv_count_advance_ts), sched_delta_us(sched_first_recv_count_advance_ts),
                static_cast<long long>(sched_first_recv_count_advance_expert),
                static_cast<long long>(sched_first_recv_count_advance_old),
                static_cast<long long>(sched_first_recv_count_advance_new),
                trace_ts_us(sched_first_normal_enqueue_attempt_ts), sched_delta_us(sched_first_normal_enqueue_attempt_ts),
                trace_ts_us(sched_first_normal_enqueue_success_ts), sched_delta_us(sched_first_normal_enqueue_success_ts),
                trace_ts_us(sched_first_task_publish_ts), sched_delta_us(sched_first_task_publish_ts),
                static_cast<long long>(sched_first_task_source), static_cast<long long>(sched_first_task_expert),
                static_cast<long long>(sched_first_task_batch), static_cast<long long>(sched_first_task_start_slot),
                static_cast<long long>(sched_first_task_num_tokens));
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

    const int args_lch_tid_base = 0;
    const int args_block_compute_tid_base = num_logical_channels * 4;
    const int args_block_num_compute_groups = host_state.num_compute_groups;
    for (int group_id = 0; group_id < args_block_num_compute_groups; ++group_id) {
        int tid = args_block_compute_tid_base + group_id;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"compute_group_%d\"}}",
                host_state.rank, tid, group_id);
    }
    {
        int tid = args_block_compute_tid_base + args_block_num_compute_groups;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":\"gather_task\"}}",
                host_state.rank, tid);
    }

    for (int logical_channel_id = 0; logical_channel_id < num_logical_channels; ++logical_channel_id) {
        int dispatch_sender_tid = args_lch_tid_base + logical_channel_id * 5;
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

        int async_publish_tid = dispatch_sender_tid + 2;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"dispatch_async_publish_lch_%d\"}}",
                host_state.rank, async_publish_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, async_publish_tid, async_publish_tid);

        int combine_sender_tid = dispatch_sender_tid + 3;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"combine_sender_lch_%d\"}}",
                host_state.rank, combine_sender_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, combine_sender_tid, combine_sender_tid);

        int combine_forwarder_tid = dispatch_sender_tid + 4;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"combine_forwarder_lch_%d\"}}",
                host_state.rank, combine_forwarder_tid, logical_channel_id);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, combine_forwarder_tid, combine_forwarder_tid);
    }

    int scheduler_tid = num_logical_channels * 5 + 50;
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
        int dispatch_sender_tid = args_lch_tid_base + logical_channel_id * 5;
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

        int async_publish_tid = dispatch_sender_tid + 2;
        int physical_channel_id = logical_channels_per_physical > 0 ? logical_channel_id / logical_channels_per_physical : logical_channel_id;
        int max_pw = -1;
        int max_src_nvl_rank = -1;
        int64_t max_pub_duration_ns = -1;
        for (int src_nvl_rank = 0; src_nvl_rank < NUM_MAX_NVL_PEERS; ++src_nvl_rank) {
            int pw = physical_channel_id * NUM_MAX_NVL_PEERS + src_nvl_rank;
            if (pw >= async_pub_n || async_pub_start[pw] == 0 || async_pub_end[pw] == 0)
                continue;
            int64_t start = async_pub_start[pw];
            int64_t end = async_pub_end[pw];
            if (end <= start)
                end = start + 1;
            int64_t gate_end = end;
            if (async_publish_all_done_ts > start && async_publish_all_done_ts < end)
                gate_end = async_publish_all_done_ts;
            int64_t duration_ns = gate_end - start;
            if (duration_ns > max_pub_duration_ns) {
                max_pub_duration_ns = duration_ns;
                max_pw = pw;
                max_src_nvl_rank = src_nvl_rank;
            }
        }
        if (max_pw >= 0) {
            int64_t start = async_pub_start[max_pw];
            int64_t worker_end = async_pub_end[max_pw];
            if (worker_end <= start)
                worker_end = start + 1;
            int64_t gate_end = worker_end;
            if (async_publish_all_done_ts > start && async_publish_all_done_ts < worker_end)
                gate_end = async_publish_all_done_ts;
            if (!emit_perf_args) {
                emit_event("dispatch_async_publish", "dispatch_async_publish", start, gate_end, pid, async_publish_tid);
            } else {
                int64_t worker_dur_ns = worker_end - start;
                int64_t gate_dur_ns = gate_end - start;
                int64_t post_gate_ns = worker_end > gate_end ? worker_end - gate_end : 0;
                int64_t attributed_ns = async_pub_wait_ring[max_pw] + async_pub_poll[max_pw] +
                    async_pub_gap[max_pw] + async_pub_start_gap[max_pw] + async_pub_empty_gap[max_pw] +
                    async_pub_done_recheck[max_pw] + async_pub_ring_load[max_pw] +
                    async_pub_head_release[max_pw] + async_pub_syncwarp[max_pw] +
                    async_pub_finish[max_pw] + async_pub_work[max_pw];
                int64_t worker_unattributed_ns = worker_dur_ns > attributed_ns ? worker_dur_ns - attributed_ns : 0;
                int64_t gate_unattributed_ns = gate_dur_ns > attributed_ns ? gate_dur_ns - attributed_ns : 0;
                double avg_batch_tokens = async_pub_batch_count[max_pw] > 0 ?
                    static_cast<double>(async_pub_tokens[max_pw]) / static_cast<double>(async_pub_batch_count[max_pw]) : 0.0;
                double tokens_per_release = async_pub_head_release_count[max_pw] > 0 ?
                    static_cast<double>(async_pub_tokens[max_pw]) / static_cast<double>(async_pub_head_release_count[max_pw]) : 0.0;
                emit_comma();
                fprintf(f, "{\"name\":\"dispatch_async_publish\",\"cat\":\"dispatch_async_publish\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,"
                           "\"args\":{\"logical_channel\":%d,\"physical_channel\":%d,\"src_nvl_rank\":%d,\"publisher_warp\":%d,"
                           "\"worker_dur_us\":%.3f,\"gate_dur_us\":%.3f,\"post_gate_us\":%.3f,"
                           "\"wait_ring_us\":%.3f,\"poll_us\":%.3f,\"gap_us\":%.3f,"
                           "\"start_gap_us\":%.3f,\"empty_gap_us\":%.3f,\"done_recheck_us\":%.3f,"
                           "\"ring_load_us\":%.3f,\"head_release_us\":%.3f,\"syncwarp_us\":%.3f,"
                           "\"batch_wall_us\":%.3f,\"batch_accounted_us\":%.3f,\"batch_unattributed_us\":%.3f,"
                           "\"token_gap_us\":%.3f,\"helper_unattributed_us\":%.3f,"
                           "\"finish_us\":%.3f,\"work_us\":%.3f,\"scan_us\":%.3f,\"atomic_us\":%.3f,"
                           "\"fence_us\":%.3f,\"store_us\":%.3f,\"drain_us\":%.3f,"
                           "\"worker_unattributed_us\":%.3f,\"gate_unattributed_us\":%.3f,\"unattributed_us\":%.3f,"
                           "\"batch_count\":%lld,\"max_batch\":%lld,\"avg_batch_tokens\":%.3f,"
                           "\"head_release_count\":%lld,\"tokens_per_release\":%.3f,"
                           "\"tokens\":%lld,\"local_hit_tokens\":%lld,\"local_hits\":%lld}}",
                        (start - base_ts) / 1000.0, gate_dur_ns / 1000.0, pid, async_publish_tid,
                        logical_channel_id, physical_channel_id, max_src_nvl_rank, max_pw,
                        worker_dur_ns / 1000.0, gate_dur_ns / 1000.0, post_gate_ns / 1000.0,
                        async_pub_wait_ring[max_pw] / 1000.0, async_pub_poll[max_pw] / 1000.0,
                        async_pub_gap[max_pw] / 1000.0, async_pub_start_gap[max_pw] / 1000.0,
                        async_pub_empty_gap[max_pw] / 1000.0, async_pub_done_recheck[max_pw] / 1000.0,
                        async_pub_ring_load[max_pw] / 1000.0, async_pub_head_release[max_pw] / 1000.0,
                        async_pub_syncwarp[max_pw] / 1000.0,
                        async_pub_batch_wall[max_pw] / 1000.0,
                        async_pub_batch_accounted[max_pw] / 1000.0,
                        async_pub_batch_unattributed[max_pw] / 1000.0,
                        async_pub_token_gap[max_pw] / 1000.0,
                        async_pub_helper_unattributed[max_pw] / 1000.0,
                        async_pub_finish[max_pw] / 1000.0, async_pub_work[max_pw] / 1000.0,
                        async_pub_scan[max_pw] / 1000.0, async_pub_atomic[max_pw] / 1000.0,
                        async_pub_fence[max_pw] / 1000.0, async_pub_store[max_pw] / 1000.0,
                        async_pub_drain[max_pw] / 1000.0,
                        worker_unattributed_ns / 1000.0, gate_unattributed_ns / 1000.0,
                        gate_unattributed_ns / 1000.0,
                        static_cast<long long>(async_pub_batch_count[max_pw]),
                        static_cast<long long>(async_pub_max_batch[max_pw]), avg_batch_tokens,
                        static_cast<long long>(async_pub_head_release_count[max_pw]), tokens_per_release,
                        static_cast<long long>(async_pub_tokens[max_pw]),
                        static_cast<long long>(async_pub_local_hit_tokens[max_pw]),
                        static_cast<long long>(async_pub_local_hits[max_pw]));
            }
        }

        int combine_sender_tid = dispatch_sender_tid + 3;
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
                                  comb_wait_top_nhits[logical_channel_id * 2 + 0],
                                  comb_wait_top_priority_deps[logical_channel_id * 2 + 0],
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

        int combine_forwarder_tid = dispatch_sender_tid + 4;
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
                         sched_priority_already_normal, sched_priority_already_priority, sched_priority_already_tail,
                         sched_normal_after_priority_ns, sched_normal_after_priority_count,
                         sched_priority_not_full, sched_normal_full_batch_enqueues,
                         sched_flush_tail_enqueues, sched_queue_empty_count,
                         sched_queue_empty_after_dispatch_count, sched_max_ready_tail_gap,
                         sched_stall_expert, sched_stall_recv_count, sched_stall_alloc_count,
                         sched_stall_enqueue_cursor, sched_stall_first_unready_slot,
                         sched_stall_first_unready_ready, sched_stall_dispatch_done);

    // Compute task rows: one Perfetto row per compute group, one X-event per task batch.
    int compute_tid_base = num_logical_channels * 5 + 100;
    int num_compute_groups = host_state.num_compute_groups;
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
    {
        int tid = compute_tid_base + num_compute_groups;
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"gather_task\"}}",
                host_state.rank, tid);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, tid, tid);
    }
    auto gather_phase_us = [](int64_t a, int64_t b) -> double {
        if (a == 0 || b == 0 || b <= a) return 0.0;
        return (b - a) / 1000.0;
    };
    for (int t = 0; t < gather_task_count; ++t) {
        int64_t* rec = &gather_task[(size_t)t * NGF];
        int64_t start = rec[0], end = rec[1];
        if (start == 0 || end == 0 || end <= start) continue;
        int gather_sm = static_cast<int>(rec[2]);
        int batch_size = static_cast<int>(rec[3]);
        int first_token = static_cast<int>(rec[4]);
        int last_token = static_cast<int>(rec[5]);
        int nhit_sum = static_cast<int>(rec[6]);
        double scan_us = gather_phase_us(start, rec[7]);
        double reduce_us = gather_phase_us(rec[7], rec[8]);
        double signal_us = gather_phase_us(rec[8], rec[9]);
        int tid = compute_tid_base + num_compute_groups;
        if (!emit_perf_args) {
            emit_event("gather_task", "gather_sm", start, end, pid, tid);
            continue;
        }
        emit_comma();
        fprintf(f, "{\"name\":\"gather_task\",\"cat\":\"gather_sm\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,"
                   "\"pid\":%d,\"tid\":%d,\"args\":{\"gather_sm_idx\":%d,\"batch_size\":%d,"
                   "\"first_token\":%d,\"last_token\":%d,\"nhit_sum\":%d,"
                   "\"scan_us\":%.3f,\"reduce_us\":%.3f,\"signal_us\":%.3f}}",
                (start - base_ts) / 1000.0, (end - start) / 1000.0,
                pid, tid, gather_sm, batch_size, first_token, last_token, nhit_sum,
                scan_us, reduce_us, signal_us);
    }
    for (int t = 0; t < compute_task_count; ++t) {
        int64_t* rec = &compute_task[(size_t)t * NCF];
        if (rec[26] == 0) continue;
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
        const bool is_partial_batch = batch_size < COMPUTE_BATCH_SIZE;
        const char* task_color = is_partial_batch ? "terrible" : "good";
        const char* task_kind = is_partial_batch ? "partial" : "full";
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
        int64_t task_prev_end = (task_start_diag != 0 && task_prev_gap != 0) ? task_start_diag - task_prev_gap : 0;
        int task_pop_attempts = have_queue_diag ? q_pop_attempts[queue_task_idx] : 0;
        int task_cas_failures = have_queue_diag ? q_cas_failures[queue_task_idx] : 0;
        int task_group_diag = have_queue_diag ? q_group_id[queue_task_idx] : -1;
        int task_source = have_queue_diag ? q_source[queue_task_idx] : 0;
        double publish_to_pop_us = phase_us(task_publish, task_pop_done);
        double pop_wait_us = phase_us(task_pop_start, task_pop_done);
        double pop_to_bcast_us = phase_us(task_pop_done, task_bcast_done);
        double bcast_to_start_us = phase_us(task_bcast_done, task_start_diag);
        double prev_end_to_pop_start_us = phase_us(task_prev_end, task_pop_start);
        double pop_done_to_bcast_us = pop_to_bcast_us;
        double bcast_done_to_start_us = bcast_to_start_us;
        double prev_task_gap_us = ns_us(task_prev_gap);
        double boundary_unattributed_us = prev_task_gap_us - prev_end_to_pop_start_us - pop_wait_us - pop_done_to_bcast_us - bcast_done_to_start_us;
        if (boundary_unattributed_us < 0.0 && boundary_unattributed_us > -0.001)
            boundary_unattributed_us = 0.0;
        if (!emit_perf_args) {
            const char* task_name = is_partial_batch ? "compute_task_partial" : "compute_task_full";
            emit_colored_event(task_name, "compute_group", task_color,
                               start, end, pid, tid);
            continue;
        }
        emit_comma();
        fprintf(f, "{\"name\":\"compute_e%d_%s\",\"cat\":\"compute_group\",\"cname\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,"
                   "\"pid\":%d,\"tid\":%d,\"args\":{\"expert_id\":%d,\"sm_id\":%d,\"group_id\":%d,\"batch_size\":%d,\"batch_kind\":\"%s\","
                   "\"start_slot\":%d,\"end_slot\":%d,\"abs_slot_base\":%lld,\"is_flush_task\":%d,"
                   "\"hidden_size\":%d,\"intermediate_size\":%d,"
                   "\"p1_meta_us\":%.3f,\"p2_input_load_us\":%.3f,\"p3_gateup_gemm_us\":%.3f,"
                   "\"p4_down_gemm_us\":%.3f,\"p5_output_us\":%.3f,\"p6_signal_us\":%.3f,"
                   "\"p3a_up_compute_us\":%.3f,\"p3b_up_barrier_us\":%.3f,"
                   "\"p4a_down_compute_us\":%.3f,\"p4b_down_barrier_us\":%.3f,"
                   "\"p5a_out_compute_us\":%.3f,\"p5b_out_barrier_us\":%.3f,"
                   "\"p6a_donecount_us\":%.3f,\"p6b_fp32finalize_us\":%.3f,"
                   "\"p6c_fence_us\":%.3f,\"p6d_publish_us\":%.3f,"
                   "\"queue_task_idx\":%d,\"queue_group_id\":%d,\"queue_source\":%d,\"prev_task_gap_us\":%.3f,"
                   "\"publish_to_pop_us\":%.3f,\"pop_wait_us\":%.3f,\"pop_to_bcast_us\":%.3f,"
                   "\"bcast_to_start_us\":%.3f,\"prev_end_to_pop_start_us\":%.3f,"
                   "\"pop_done_to_bcast_us\":%.3f,\"bcast_done_to_start_us\":%.3f,"
                   "\"boundary_unattributed_us\":%.3f,\"pop_attempts\":%d,\"cas_failures\":%d,"
                   "\"p6b_multi_expert_rows\":%d,\"p6_task_has_multi\":%d}}",
                expert_id, task_kind, task_color, ts_ns / 1000.0, dur_ns / 1000.0, pid, tid, expert_id, sm_id, group_id, batch_size, task_kind,
                start_slot, end_slot, static_cast<long long>(abs_slot_base), is_flush_task,
                hidden, intermediate,
                meta_us, input_us, upgemm_us, downgemm_us, output_us, signal_us,
                up_compute_us, up_barrier_us, down_compute_us, down_barrier_us,
                out_compute_us, out_barrier_us,
                sig_donecount_us, sig_finalize_us, sig_fence_us, sig_publish_us,
                queue_task_idx, task_group_diag, task_source, prev_task_gap_us,
                publish_to_pop_us, pop_wait_us, pop_to_bcast_us,
                bcast_to_start_us, prev_end_to_pop_start_us,
                pop_done_to_bcast_us, bcast_done_to_start_us, boundary_unattributed_us,
                task_pop_attempts, task_cas_failures,
                multi_expert_rows, task_has_multi);
    }

    fprintf(f, "\n]\n");
    fclose(f);
    // printf("[MK-PERF] Perfetto trace written to %s (%d logical channels)\n", filename, num_logical_channels);
#endif
}
#endif

// Step 1 optimization from .v0: use PyTorch's CUDA caching allocator for long-lived
// megakernel state buffers, and batch host-side memset initialization into one
// stream-ordered kernel. This should not change math or kernel scheduling.
static inline cudaError_t mk_caching_alloc(void** pp, size_t nbytes) {
    *pp = (nbytes == 0) ? nullptr : c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);
    return cudaSuccess;
}

static inline cudaError_t mk_caching_free(void* ptr) {
    if (ptr != nullptr)
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    return cudaSuccess;
}

struct FusedFillDesc {
    void* ptr;
    size_t bytes;
    uint32_t word;
};

// Simple parallel fill: one block per desc, all threads stride over words.
// Host splits large buffers into <=CHUNK_SIZE descs so blocks are balanced.
// When expert_count_mapped is provided, block 0 also builds the compact expert
// layout in the same launch so we avoid a separate H2D or a separate kernel.
__global__ void fused_fill_kernel(
    const FusedFillDesc* descs,
    int ndescs,
    const int* expert_count_mapped,
    int* expert_slot_base,
    int* expert_count,
    int num_local_experts) {
    if (expert_count_mapped != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
        int slot_base = 0;
        for (int le = 0; le < num_local_experts; ++le) {
            int cnt = expert_count_mapped[le];
            if (cnt < 0)
                cnt = 0;
            expert_count[le] = cnt;
            expert_slot_base[le] = slot_base;
            slot_base += cnt;
        }
    }

    const int desc_idx = blockIdx.x;
    if (desc_idx >= ndescs)
        return;
    const FusedFillDesc desc = descs[desc_idx];
    auto* words = reinterpret_cast<uint32_t*>(desc.ptr);
    const size_t nwords = desc.bytes / sizeof(uint32_t);
    for (size_t i = threadIdx.x; i < nwords; i += blockDim.x)
        words[i] = desc.word;
}

static void initialize_megakernel_launch_state(const MegaKernelState& state, cudaStream_t stream);

// --- TMA descriptor cache (persistent across iterations; the cache is the SOLE owner
// of these device buffers — no MegaKernelState ever frees them). A small bounded,
// round-robin set of entries lets the forward key and backward key coexist so both
// paths hit across iterations. Buffers are only freed when an entry slot is evicted,
// which is safe because the kernel that used them has already completed (synchronous
// launch) before the slot can be reused. ---
namespace {
struct TmaCacheKey {
    const void* w_gateup;
    const void* w_down;
    const void* gemm_ws;
    const void* w_gateup_fp8;
    const void* w_down_fp8;
    int num_local_experts;
    int hidden_dim;
    int intermediate_dim;
    int num_compute_groups;
};
struct TmaCacheEntry {
    TmaCacheKey key{};
    bool valid = false;
    umma::ComputeTmaAtoms* compute_tma = nullptr;
    umma::InputTmaAtom_t* group_input_tma = nullptr;
    umma::ComputeDownTmaAtoms* compute_down_tma = nullptr;
    umma_fp8::ComputeFp8TmaAtoms* compute_fp8_tma = nullptr;
    umma_fp8::InputFp8TmaAtom_t* group_input_fp8_tma = nullptr;
    umma_fp8::ComputeFp8DownTmaAtoms* compute_fp8_down_tma = nullptr;
};
constexpr int kTmaCacheCap = 8;
static thread_local TmaCacheEntry s_tma_cache[kTmaCacheCap];
static thread_local int s_tma_cache_next = 0;

// Default arena chunk granularity. A new chunk is sized max(nbytes, this), so any
// single allocation >= this value gets its own exact-sized chunk (zero tail waste),
// while allocations smaller than this pack together. Kept small (8 MiB) so the large
// bf16 activation/scratch buffers (bwd_preact, bwd_fc1_input, recv_tokens,
// compute_output_slot, combine_input, gemm_workspace, ...) are each right-sized
// instead of rounding up to a 256 MiB chunk; only the small int side-tables pay the
// <=8 MiB packing tail. This removes the ~256 MiB-granularity internal fragmentation
// that inflated the forward activation retained / peak reserved.
constexpr size_t kMegakernelArenaChunkBytes = 8ull * 1024ull * 1024ull;

struct MegakernelArenaAllocator {
    std::vector<void*> chunks;
    void* current_chunk = nullptr;
    size_t current_chunk_bytes = 0;
    size_t offset = 0;

    cudaError_t alloc(void** pp, size_t nbytes) {
        if (nbytes == 0) {
            *pp = nullptr;
            return cudaSuccess;
        }
        constexpr size_t kAlign = NUM_BUFFER_ALIGNMENT_BYTES;
        auto align_up = [](size_t v, size_t a) {
            return (v + a - 1) / a * a;
        };
        nbytes = align_up(nbytes, kAlign);
        size_t aligned_offset = align_up(offset, kAlign);
        if (current_chunk == nullptr || aligned_offset + nbytes > current_chunk_bytes) {
            if (chunks.size() >= kMegakernelArenaChunkCap)
                return cudaErrorMemoryAllocation;
            const size_t chunk_bytes = align_up(std::max(nbytes, kMegakernelArenaChunkBytes), kAlign);
            void* chunk = nullptr;
            cudaError_t err = mk_caching_alloc(&chunk, chunk_bytes);
            if (err != cudaSuccess)
                return err;
            chunks.push_back(chunk);
            current_chunk = chunk;
            current_chunk_bytes = chunk_bytes;
            offset = 0;
            aligned_offset = 0;
        } else {
            offset = aligned_offset;
        }
        *pp = static_cast<void*>(static_cast<char*>(current_chunk) + aligned_offset);
        offset = aligned_offset + nbytes;
        return cudaSuccess;
    }
};

static inline void store_arena_chunks(const MegakernelArenaAllocator& arena, void** dst, int* count) {
    EP_HOST_ASSERT(arena.chunks.size() <= kMegakernelArenaChunkCap);
    *count = static_cast<int>(arena.chunks.size());
    for (int i = 0; i < *count; ++i)
        dst[i] = arena.chunks[i];
}

static inline cudaError_t free_arena_chunks(void** chunks, int count) {
    for (int i = 0; i < count; ++i) {
        if (chunks[i] != nullptr) {
            cudaError_t err = mk_caching_free(chunks[i]);
            if (err != cudaSuccess)
                return err;
            chunks[i] = nullptr;
        }
    }
    return cudaSuccess;
}

}  // anonymous namespace

MegaKernelState* allocate_megakernel_state_v7(
    // --- Dispatch input data (from PyTorch tensors) ---
    const int4* x,
    const uint32_t* x_scales,
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
    int hidden_int4,
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
    ComputeDType compute_dtype,
    const void* W_gateup_fp8,
    const void* W_down_fp8,
    const uint32_t* W_gateup_fp8_sf,
    const uint32_t* W_down_fp8_sf,
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
    int64_t num_nvl_bytes,
    const int* host_expert_count,
    const int* device_expert_count_mapped,
    // Optional caller-owned buffers. Non-null pointers are borrowed by the state.
    __nv_bfloat16* external_bwd_fc1_input,
    __nv_bfloat16* external_bwd_preact,
    int* external_fwd_slot_map,
    int4* external_combined_x,
    float* external_combined_topk_weights,
    MegaKernelState* host_state_out,
    int* rdma_reuse_dispatch_quiet_done,
    int* rdma_reuse_combine_clear_done,
    int rdma_reuse_prelude_enable
) {
    MegakernelArenaAllocator persistent_arena;
    MegakernelArenaAllocator transient_arena;
    MegakernelArenaAllocator* arena = &persistent_arena;
#define cudaMalloc(pp, n) arena->alloc(reinterpret_cast<void**>(pp), (n))
    auto mk_cache_alloc = [](auto** pp, size_t nbytes) -> cudaError_t {
        return mk_caching_alloc(reinterpret_cast<void**>(pp), nbytes);
    };
    struct MkFillRec { void* ptr; int byte_value; size_t bytes; };
    std::vector<MkFillRec> mk_fill_recs;
    auto mk_record_fill = [&](void* ptr, int byte_value, size_t bytes) -> cudaError_t {
        if (bytes != 0)
            mk_fill_recs.push_back({ptr, byte_value, bytes});
        return cudaSuccess;
    };
#define cudaMemset(p, v, n) mk_record_fill((p), (v), (n))

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
    int* compute_group_barrier;
    int* compute_group_phase;
    ComputeTask* compute_tasks;
    int* compute_task_head;
    int* compute_task_tail;
    int* compute_task_reserve_tail;
    int* compute_enqueue_done;
    int* scheduler_done_count;
    int* priority_scheduler_done;
    int* expert_enqueue_cursor;
    int* compute_group_task_idx;
    __nv_bfloat16* combine_input;
    float* combine_input_topk_weights;
    internode::SourceMeta* combine_input_src_meta;
    int* combine_rdma_head_work;
    int* combine_nvl_head_work;
    __nv_bfloat16* gemm_workspace;
    umma_fp8::ElemAB* recv_tokens_fp8;
    umma_fp8::ScalePack* recv_tokens_fp8_sf;
    umma_fp8::ElemAB* input_fp8_workspace;
    umma_fp8::ScalePack* input_fp8_sf_workspace;
    umma_fp8::ElemAB* act_fp8_workspace;
    umma_fp8::ScalePack* act_fp8_sf_workspace;
    float* output_accum;
    int* send_rdma_head;
    int* send_nvl_head;
    int* recv_rdma_channel_prefix_matrix;
    int* recv_gbl_channel_prefix_matrix;

    // Mirror DeepEP host-side launch invariants before allocating state. These
    // protect the producer/consumer queue geometry used by dispatch and combine.
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());
    EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_HOST_ASSERT(hidden_int4 > 0);
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

    // Per-expert slot layout — indexed as [expert_slot_base[le] + slot].
    // Compact packing: when host_expert_count is provided (forward path), each expert's
    // region is sized by its real received-token count and regions are packed contiguously
    // via an exclusive prefix sum (total = Σ count). When null (backward / legacy path),
    // falls back to the fixed le*max_tokens_per_expert layout (total = num_local_experts*max_tpe).
    // The forward path also passes device_expert_count_mapped so fused_fill_kernel can
    // materialize expert_count / expert_slot_base in the same launch that clears buffers.
    int* expert_slot_base;
    int* expert_count;
    CUDA_CHECK(cudaMalloc(&expert_slot_base, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_count, num_local_experts * sizeof(int)));
    size_t total_expert_slots = 0;
    if (host_expert_count != nullptr) {
        for (int le = 0; le < num_local_experts; ++le) {
            const int cnt = host_expert_count[le];
            EP_HOST_ASSERT(cnt >= 0 && cnt <= max_tokens_per_expert);
            total_expert_slots += static_cast<size_t>(cnt);
        }
    } else {
        for (int le = 0; le < num_local_experts; ++le) {
            total_expert_slots += static_cast<size_t>(max_tokens_per_expert);
        }
    }
    if (total_expert_slots == 0) total_expert_slots = 1;  // avoid zero-size allocations
    if (device_expert_count_mapped == nullptr) {
        std::vector<int> h_expert_slot_base(num_local_experts);
        std::vector<int> h_expert_count(num_local_experts);
        size_t slot_base = 0;
        for (int le = 0; le < num_local_experts; ++le) {
            const int cnt = (host_expert_count != nullptr) ? host_expert_count[le] : max_tokens_per_expert;
            h_expert_slot_base[le] = static_cast<int>(slot_base);
            h_expert_count[le] = cnt;
            slot_base += static_cast<size_t>(cnt);
        }
        cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
        CUDA_CHECK(cudaMemcpyAsync(expert_slot_base, h_expert_slot_base.data(),
                              num_local_experts * sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(expert_count, h_expert_count.data(),
                              num_local_experts * sizeof(int), cudaMemcpyHostToDevice, s));
    }

    int* expert_slot_ready;
    CUDA_CHECK(cudaMalloc(&expert_slot_ready, total_expert_slots * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_slot_ready, 0, total_expert_slots * sizeof(int)));
    size_t recv_tokens_bytes = total_expert_slots * hidden_dim * sizeof(__nv_bfloat16);
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&recv_tokens, recv_tokens_bytes));
    arena = &persistent_arena;

    CUDA_CHECK(cudaMalloc(&expert_token_offsets, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_token_offsets, 0, num_local_experts * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&recv_token_source_info, total_expert_slots * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_token_source_info, 0xff, total_expert_slots * 2 * sizeof(int)));  // Init to -1
    CUDA_CHECK(cudaMalloc(&recv_token_route_weights, total_expert_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_src_meta, total_expert_slots * sizeof(internode::SourceMeta)));

    // Compute state
    int base_compute_groups = num_compute_sms / COMPUTE_GROUP_SIZE;
    int total_compute_sms_after_dispatch = num_compute_sms + num_dispatch_sms;
    int post_group_count = (total_compute_sms_after_dispatch + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    int num_combine_compute_groups = (num_combine_sms + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    int num_compute_groups = post_group_count + num_combine_compute_groups;
    EP_HOST_ASSERT(base_compute_groups > 0);
    EP_HOST_ASSERT(num_compute_groups > 0);
    EP_HOST_ASSERT(num_compute_sms == base_compute_groups * COMPUTE_GROUP_SIZE);
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
    CUDA_CHECK(cudaMalloc(&priority_scheduler_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(priority_scheduler_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_enqueue_cursor, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_enqueue_cursor, 0, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_group_task_idx, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_task_idx, 0xff, num_compute_groups * sizeof(int)));

    // Dedicated gather SM task queue state.
    int* token_done_count;
    int* gather_claimed;
    int* combine_token_ready;
    int* gather_ready_queue;
    int* gather_ready_head;
    int* gather_ready_tail;
    int* gather_ready_reserve_tail;
    int* gather_scan_cursor;
    int* gather_task_count;
    int* gather_task_tokens;
    int* gather_task_nhits;
    CUDA_CHECK(cudaMalloc(&token_done_count, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_done_count, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_claimed, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_claimed, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_token_ready, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_token_ready, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_queue, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_queue, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_head, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_head, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_reserve_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_reserve_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_scan_cursor, COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_scan_cursor, 0, COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_task_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_task_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_task_tokens, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_task_tokens, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_task_nhits, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_task_nhits, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    int* combine_done_count;
    int* combine_all_done;
    CUDA_CHECK(cudaMalloc(&combine_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_all_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_all_done, 0, sizeof(int)));
    // Publish-offload backing state (Stage 1: allocate + zero only, no consumer yet).
    // num_pub_warps_total = one publisher per NVL receiver warp per receiver SM.
    const int num_pub_warps_total = (num_dispatch_sms / 2) * NUM_MAX_NVL_PEERS;
    int* pending_topk_idx;
    float* pending_topk_weights;
    internode::SourceMeta* pending_meta;
    int* pub_ring;
    int* pub_ring_head;
    int* pub_ring_tail;
    int* recv_warp_done;
    int* publish_warp_done;
    int* publish_done_count;
    int* publish_all_done;
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
    CUDA_CHECK(cudaMalloc(&publish_warp_done, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_warp_done, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&publish_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&publish_all_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_all_done, 0, sizeof(int)));

    // Combine input namespace from dispatch receive.
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&combine_input, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(combine_input, 0, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&combine_input_topk_weights, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(combine_input_topk_weights, 0, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&combine_input_src_meta, (size_t)max_total_recv_tokens * sizeof(internode::SourceMeta)));
    arena = &persistent_arena;

    // Per-token compute signaling
    int* token_compute_expected;
    CUDA_CHECK(cudaMalloc(&token_compute_expected, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_compute_expected, 0, (size_t)max_total_recv_tokens * sizeof(int)));

    // Per-slot output path scratch + reverse map (MEGAKERNEL_COMPUTE_DESIGN III).
    __nv_bfloat16* compute_output_slot;
    int* token_nhits;
    int* token_slot_list;
    int* priority_token_cursor;
    int* expert_batch_enqueued;
#if MK_PERF_TRACE_ARGS
    int64_t* expert_batch_enqueue_ts;
    int* token_priority_dep_count;
#endif
    int* priority_batch_skip_epoch;
    int* priority_batch_retry_epoch;
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&compute_output_slot, recv_tokens_bytes));
    CUDA_CHECK(cudaMemset(compute_output_slot, 0, recv_tokens_bytes));
    arena = &persistent_arena;
    // Backward activation save: original fc1 input X plus gate/up preact by recv_token.
    // A backward replay borrows the forward state's saved buffers directly, avoiding a
    // transient allocate/free cycle for hundreds of MiB.
    __nv_bfloat16* bwd_fc1_input = external_bwd_fc1_input;
    const bool owns_bwd_fc1_input = bwd_fc1_input == nullptr;
    const size_t bwd_fc1_input_bytes = (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16);
    if (owns_bwd_fc1_input) {
        CUDA_CHECK(cudaMalloc(&bwd_fc1_input, bwd_fc1_input_bytes));
        CUDA_CHECK(cudaMemset(bwd_fc1_input, 0, bwd_fc1_input_bytes));
    }
    // Phase 3 (Step 3.3b): compact preact storage. bwd_preact is now indexed by the compact
    // forward slot (Step 3.3a), so it only needs total_expert_slots (= Σ expert_count = S) rows
    // instead of the max_total_recv_tokens * num_topk (= R*K) sparse upper bound. Written per-row
    // by slot (no batched TMA tile over preact), so no padding is required.
    __nv_bfloat16* bwd_preact = external_bwd_preact;
    const bool owns_bwd_preact = bwd_preact == nullptr;
    const size_t bwd_preact_bytes = (size_t)total_expert_slots * 2 * intermediate_dim * sizeof(__nv_bfloat16);
    if (owns_bwd_preact) {
        CUDA_CHECK(cudaMalloc(&bwd_preact, bwd_preact_bytes));
        CUDA_CHECK(cudaMemset(bwd_preact, 0, bwd_preact_bytes));
    }
    // Phase 3 (A2) scaffolding: (recv_token, topk_slot) -> forward slot translation table.
    // Init to -1; populated by the forward compute worker (Step 3.2), consumed by backward
    // (Step 3.3a). Not read/written yet at this step.
    int* fwd_slot_map = external_fwd_slot_map;
    const bool owns_fwd_slot_map = fwd_slot_map == nullptr;
    const size_t fwd_slot_map_bytes = (size_t)max_total_recv_tokens * num_topk * sizeof(int);
    if (owns_fwd_slot_map) {
        CUDA_CHECK(cudaMalloc(&fwd_slot_map, fwd_slot_map_bytes));
        CUDA_CHECK(cudaMemset(fwd_slot_map, 0xff, fwd_slot_map_bytes));
    }
    CUDA_CHECK(cudaMalloc(&token_nhits, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_nhits, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&token_slot_list, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_slot_list, 0xff, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&priority_token_cursor, sizeof(int)));
    CUDA_CHECK(cudaMemset(priority_token_cursor, 0, sizeof(int)));
    const int max_batches_per_expert = (max_tokens_per_expert + COMPUTE_BATCH_SIZE - 1) / COMPUTE_BATCH_SIZE;
    CUDA_CHECK(cudaMalloc(&expert_batch_enqueued, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_batch_enqueued, 0, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
#if MK_PERF_TRACE_ARGS
    CUDA_CHECK(cudaMalloc(&expert_batch_enqueue_ts, (size_t)num_local_experts * max_batches_per_expert * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(expert_batch_enqueue_ts, 0, (size_t)num_local_experts * max_batches_per_expert * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&token_priority_dep_count, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_priority_dep_count, 0, (size_t)max_total_recv_tokens * sizeof(int)));
#endif
    CUDA_CHECK(cudaMalloc(&priority_batch_skip_epoch, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMemset(priority_batch_skip_epoch, 0, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&priority_batch_retry_epoch, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMemset(priority_batch_retry_epoch, 0, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));

    // GEMM workspace: per-compute-group batched intermediates for M=128 compute batches.
    // Layout: [input(M*hidden)][GU scratch(M*2I)][act(M*I)][down(M*hidden)].
    // The interleaved gate/up path writes act directly from the GEMM epilogue; the
    // GU scratch region remains reserved so existing descriptors/helpers stay valid.
    size_t per_group_elems = (size_t)COMPUTE_BATCH_SIZE * (2 * hidden_dim + 3 * intermediate_dim);
    size_t workspace_bytes = num_compute_groups * per_group_elems * sizeof(__nv_bfloat16);
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&gemm_workspace, workspace_bytes));
    arena = &persistent_arena;

    const auto* W_gateup_fp8_typed = reinterpret_cast<const umma_fp8::ElemAB*>(W_gateup_fp8);
    const auto* W_down_fp8_typed = reinterpret_cast<const umma_fp8::ElemAB*>(W_down_fp8);
    const auto* W_gateup_fp8_sf_typed = reinterpret_cast<const umma_fp8::ScalePack*>(W_gateup_fp8_sf);
    const auto* W_down_fp8_sf_typed = reinterpret_cast<const umma_fp8::ScalePack*>(W_down_fp8_sf);
    const int fp8_hidden_scale_k_packed = umma_fp8::dg_fp8_scale_k_packed(hidden_dim, umma_fp8::kDgFp8GranKA);
    const int fp8_intermediate_scale_k_packed = umma_fp8::dg_fp8_scale_k_packed(intermediate_dim, umma_fp8::kDgFp8GranKA);
    const bool build_fp8_compute = compute_dtype == ComputeDType::kFP8E4M3 &&
        W_gateup_fp8_typed != nullptr && W_down_fp8_typed != nullptr &&
        W_gateup_fp8_sf_typed != nullptr && W_down_fp8_sf_typed != nullptr &&
        num_local_experts <= umma_fp8::kMaxLocalExperts;

    recv_tokens_fp8 = nullptr;
    recv_tokens_fp8_sf = nullptr;
    input_fp8_workspace = nullptr;
    input_fp8_sf_workspace = nullptr;
    act_fp8_workspace = nullptr;
    act_fp8_sf_workspace = nullptr;
    if (build_fp8_compute) {
        CUDA_CHECK(cudaMalloc(&recv_tokens_fp8, total_expert_slots * hidden_dim * sizeof(umma_fp8::ElemAB)));
        CUDA_CHECK(cudaMalloc(&recv_tokens_fp8_sf, total_expert_slots * fp8_hidden_scale_k_packed * sizeof(umma_fp8::ScalePack)));
        CUDA_CHECK(cudaMalloc(&input_fp8_workspace, (size_t)num_compute_groups * COMPUTE_BATCH_SIZE * hidden_dim * sizeof(umma_fp8::ElemAB)));
        CUDA_CHECK(cudaMalloc(&input_fp8_sf_workspace, (size_t)num_compute_groups * COMPUTE_BATCH_SIZE * fp8_hidden_scale_k_packed * sizeof(umma_fp8::ScalePack)));
        CUDA_CHECK(cudaMalloc(&act_fp8_workspace, (size_t)num_compute_groups * COMPUTE_BATCH_SIZE * intermediate_dim * sizeof(umma_fp8::ElemAB)));
        CUDA_CHECK(cudaMalloc(&act_fp8_sf_workspace, (size_t)num_compute_groups * COMPUTE_BATCH_SIZE * fp8_intermediate_scale_k_packed * sizeof(umma_fp8::ScalePack)));
    }

    // --- S4.4 (route B2): build UMMA compute TMA atoms on host, upload to device ---
    // Gate/up uses A[M, hidden] x interleaved Wgu[2 * intermediate, hidden]^T,
    // with the SwiGLU epilogue storing act[M, intermediate] directly.
    // Down uses act[M, intermediate] x W_down[hidden, intermediate]^T.
    //
    // TMA descriptors only depend on weight pointers, workspace pointer, and shape.
    // Look up a bounded, persistent cache so repeated iterations skip the H2D copies.

    TmaCacheKey cur_key{W_gateup, W_down, gemm_workspace, W_gateup_fp8,
                        W_down_fp8, num_local_experts, hidden_dim, intermediate_dim, num_compute_groups};

    umma::ComputeTmaAtoms* d_compute_tma = nullptr;
    umma::InputTmaAtom_t* d_group_input_tma = nullptr;
    umma::ComputeDownTmaAtoms* d_compute_down_tma = nullptr;
    umma_fp8::ComputeFp8TmaAtoms* d_compute_fp8_tma = nullptr;
    umma_fp8::InputFp8TmaAtom_t* d_group_input_fp8_tma = nullptr;
    umma_fp8::ComputeFp8DownTmaAtoms* d_compute_fp8_down_tma = nullptr;

    int hit_idx = -1;
    for (int i = 0; i < kTmaCacheCap; ++i) {
        if (s_tma_cache[i].valid && std::memcmp(&s_tma_cache[i].key, &cur_key, sizeof(TmaCacheKey)) == 0) {
            hit_idx = i;
            break;
        }
    }

    if (hit_idx >= 0) {
        // Reuse cached device TMA descriptors — no H2D needed.
        const TmaCacheEntry& e = s_tma_cache[hit_idx];
        d_compute_tma = e.compute_tma;
        d_group_input_tma = e.group_input_tma;
        d_compute_down_tma = e.compute_down_tma;
        d_compute_fp8_tma = e.compute_fp8_tma;
        d_group_input_fp8_tma = e.group_input_fp8_tma;
        d_compute_fp8_down_tma = e.compute_fp8_down_tma;
    } else {

    if (W_gateup != nullptr && W_down != nullptr &&
        num_local_experts <= umma::kMaxLocalExperts) {
        umma::ComputeTmaAtoms h_atoms;
        umma::build_compute_tma_atoms(h_atoms, W_gateup, num_local_experts,
                                      intermediate_dim, hidden_dim);
        CUDA_CHECK(mk_cache_alloc(&d_compute_tma, sizeof(umma::ComputeTmaAtoms)));
        CUDA_CHECK(cudaMemcpy(d_compute_tma, &h_atoms, sizeof(umma::ComputeTmaAtoms), cudaMemcpyHostToDevice));

        std::vector<umma::InputTmaAtom_t> h_in;
        h_in.reserve(num_compute_groups);
        for (int g = 0; g < num_compute_groups; ++g) {
            const __nv_bfloat16* in_g = gemm_workspace + (size_t)g * per_group_elems;
            const __nv_bfloat16* gu_g   = in_g + (size_t)COMPUTE_BATCH_SIZE * hidden_dim;
            const __nv_bfloat16* act_g  = gu_g + (size_t)COMPUTE_BATCH_SIZE * (2 * intermediate_dim);
            const __nv_bfloat16* down_g = act_g + (size_t)COMPUTE_BATCH_SIZE * intermediate_dim;
            h_in.push_back(umma::make_input_group_atoms(in_g, gu_g, act_g, down_g,
                                                        COMPUTE_BATCH_SIZE, hidden_dim,
                                                        intermediate_dim, hidden_dim));
        }
        CUDA_CHECK(mk_cache_alloc(&d_group_input_tma, num_compute_groups * sizeof(umma::InputTmaAtom_t)));
        CUDA_CHECK(cudaMemcpy(d_group_input_tma, h_in.data(),
                              num_compute_groups * sizeof(umma::InputTmaAtom_t), cudaMemcpyHostToDevice));

        umma::ComputeDownTmaAtoms h_down;
        umma::build_compute_down_tma_atoms(h_down, W_down, num_local_experts, hidden_dim, intermediate_dim);
        CUDA_CHECK(mk_cache_alloc(&d_compute_down_tma, sizeof(umma::ComputeDownTmaAtoms)));
        CUDA_CHECK(cudaMemcpy(d_compute_down_tma, &h_down, sizeof(umma::ComputeDownTmaAtoms), cudaMemcpyHostToDevice));
    }

    if (build_fp8_compute) {
        umma_fp8::ComputeFp8TmaAtoms h_fp8_atoms;
        umma_fp8::build_compute_fp8_tma_atoms(h_fp8_atoms, W_gateup_fp8_typed, W_gateup_fp8_sf_typed,
                                              num_local_experts, intermediate_dim, hidden_dim);
        CUDA_CHECK(mk_cache_alloc(&d_compute_fp8_tma, sizeof(umma_fp8::ComputeFp8TmaAtoms)));
        CUDA_CHECK(cudaMemcpy(d_compute_fp8_tma, &h_fp8_atoms,
                              sizeof(umma_fp8::ComputeFp8TmaAtoms), cudaMemcpyHostToDevice));

        umma_fp8::ComputeFp8DownTmaAtoms h_fp8_down;
        umma_fp8::build_compute_fp8_down_tma_atoms(h_fp8_down, W_down_fp8_typed, W_down_fp8_sf_typed,
                                                   num_local_experts, hidden_dim, intermediate_dim);
        CUDA_CHECK(mk_cache_alloc(&d_compute_fp8_down_tma, sizeof(umma_fp8::ComputeFp8DownTmaAtoms)));
        CUDA_CHECK(cudaMemcpy(d_compute_fp8_down_tma, &h_fp8_down,
                              sizeof(umma_fp8::ComputeFp8DownTmaAtoms), cudaMemcpyHostToDevice));

        std::vector<umma_fp8::InputFp8TmaAtom_t> h_fp8_in;
        h_fp8_in.reserve(num_compute_groups);
        for (int g = 0; g < num_compute_groups; ++g) {
            const __nv_bfloat16* in_g = gemm_workspace + (size_t)g * per_group_elems;
            const __nv_bfloat16* gu_g = in_g + (size_t)COMPUTE_BATCH_SIZE * hidden_dim;
            const __nv_bfloat16* act_g = gu_g + (size_t)COMPUTE_BATCH_SIZE * (2 * intermediate_dim);
            const __nv_bfloat16* down_g = act_g + (size_t)COMPUTE_BATCH_SIZE * intermediate_dim;
            const umma_fp8::ElemAB* in_fp8_g = input_fp8_workspace + (size_t)g * COMPUTE_BATCH_SIZE * hidden_dim;
            const umma_fp8::ScalePack* in_sf_g = input_fp8_sf_workspace + (size_t)g * COMPUTE_BATCH_SIZE * fp8_hidden_scale_k_packed;
            const umma_fp8::ElemAB* act_fp8_g = act_fp8_workspace + (size_t)g * COMPUTE_BATCH_SIZE * intermediate_dim;
            const umma_fp8::ScalePack* act_sf_g = act_fp8_sf_workspace + (size_t)g * COMPUTE_BATCH_SIZE * fp8_intermediate_scale_k_packed;
            h_fp8_in.push_back(umma_fp8::make_input_fp8_group_atoms(
                in_fp8_g, in_sf_g, act_fp8_g, act_sf_g, act_g, down_g,
                COMPUTE_BATCH_SIZE, hidden_dim, intermediate_dim));
        }
        CUDA_CHECK(cudaMalloc(&d_group_input_fp8_tma, num_compute_groups * sizeof(umma_fp8::InputFp8TmaAtom_t)));
        CUDA_CHECK(cudaMemcpy(d_group_input_fp8_tma, h_fp8_in.data(),
                              num_compute_groups * sizeof(umma_fp8::InputFp8TmaAtom_t), cudaMemcpyHostToDevice));
    }

        // Insert into cache (round-robin). Evicting a slot frees its buffers; safe because
        // the kernel that used them completed before this slot can be reused.
        TmaCacheEntry& slot = s_tma_cache[s_tma_cache_next];
        if (slot.valid) {
            if (slot.compute_tma) mk_caching_free(slot.compute_tma);
            if (slot.group_input_tma) mk_caching_free(slot.group_input_tma);
            if (slot.compute_down_tma) mk_caching_free(slot.compute_down_tma);
            if (slot.compute_fp8_tma) mk_caching_free(slot.compute_fp8_tma);
            if (slot.group_input_fp8_tma) mk_caching_free(slot.group_input_fp8_tma);
            if (slot.compute_fp8_down_tma) mk_caching_free(slot.compute_fp8_down_tma);
        }
        slot.key = cur_key;
        slot.valid = true;
        slot.compute_tma = d_compute_tma;
        slot.group_input_tma = d_group_input_tma;
        slot.compute_down_tma = d_compute_down_tma;
        slot.compute_fp8_tma = d_compute_fp8_tma;
        slot.group_input_fp8_tma = d_group_input_fp8_tma;
        slot.compute_fp8_down_tma = d_compute_fp8_down_tma;
        s_tma_cache_next = (s_tma_cache_next + 1) % kTmaCacheCap;
    } // end cache miss

    // Output accumulator [num_tokens, hidden_dim] in float32
    arena = &transient_arena;
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
    arena = &persistent_arena;

    // Recv logical-channel prefix matrices (written by forwarder)
    int num_physical_channels = num_dispatch_sms / 2;  // even/odd pairing
    int num_combine_channels = num_combine_sms / 2;
    EP_HOST_ASSERT(num_combine_channels == num_physical_channels);
    EP_HOST_ASSERT(num_logical_channels >= num_physical_channels);

    // printf("num_tokens: %d, um_rdma_ranks: %d, num_physical_channels: %d, num_logical_channels: %d\n",
    //        num_tokens, kNumRDMARanks, num_physical_channels, num_logical_channels);

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
    host_state.allocator_combine_buffer_ptrs = combine_buffer_ptrs;

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
    host_state.allocator_num_dispatch_sms = num_dispatch_sms;
    host_state.allocator_num_forwarder_sms = num_forwarder_sms;
    host_state.allocator_num_compute_sms = num_compute_sms;
    host_state.allocator_num_combine_sms = num_combine_sms;
    host_state.allocator_num_logical_channels = num_logical_channels;
    host_state.allocator_max_tokens_per_expert = max_tokens_per_expert;
    host_state.allocator_max_total_recv_tokens = max_total_recv_tokens;
    host_state.allocator_num_rdma_bytes = num_rdma_bytes;
    host_state.allocator_num_nvl_bytes = num_nvl_bytes;

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
    host_state.expert_slot_base = expert_slot_base;
    host_state.expert_count = expert_count;
    host_state.recv_token_source_info = recv_token_source_info;
    host_state.recv_token_route_weights = recv_token_route_weights;
    host_state.recv_src_meta = recv_src_meta;

    // Compute state
    host_state.compute_group_barrier = compute_group_barrier;
    host_state.compute_group_phase = compute_group_phase;
    host_state.compute_tasks = compute_tasks;
    host_state.max_compute_tasks = max_compute_tasks;
    host_state.compute_task_head = compute_task_head;
    host_state.compute_task_tail = compute_task_tail;
    host_state.compute_task_reserve_tail = compute_task_reserve_tail;
    host_state.compute_enqueue_done = compute_enqueue_done;
    host_state.scheduler_done_count = scheduler_done_count;
    host_state.priority_scheduler_done = priority_scheduler_done;
    host_state.expert_enqueue_cursor = expert_enqueue_cursor;
    host_state.compute_group_task_idx = compute_group_task_idx;

    // Dedicated gather SM task queue state
    host_state.token_done_count = token_done_count;
    host_state.gather_claimed = gather_claimed;
    host_state.combine_token_ready = combine_token_ready;
    host_state.gather_ready_queue = gather_ready_queue;
    host_state.gather_ready_head = gather_ready_head;
    host_state.gather_ready_tail = gather_ready_tail;
    host_state.gather_ready_reserve_tail = gather_ready_reserve_tail;
    host_state.gather_scan_cursor = gather_scan_cursor;
    host_state.gather_task_count = gather_task_count;
    host_state.gather_task_tokens = gather_task_tokens;
    host_state.gather_task_nhits = gather_task_nhits;
    host_state.combine_done_count = combine_done_count;
    host_state.combine_all_done = combine_all_done;

    // Publish-offload backing state (Stage 1)
    host_state.pending_topk_idx = pending_topk_idx;
    host_state.pending_topk_weights = pending_topk_weights;
    host_state.pending_meta = pending_meta;
    host_state.pub_ring = pub_ring;
    host_state.pub_ring_head = pub_ring_head;
    host_state.pub_ring_tail = pub_ring_tail;
    host_state.recv_warp_done = recv_warp_done;
    host_state.publish_warp_done = publish_warp_done;
    host_state.publish_done_count = publish_done_count;
    host_state.publish_all_done = publish_all_done;
    host_state.num_pub_warps_total = num_pub_warps_total;


    // Expert weights
    host_state.W_gateup = W_gateup;
    host_state.W_down = W_down;
    host_state.compute_dtype = compute_dtype;
    host_state.W_gateup_fp8 = W_gateup_fp8_typed;
    host_state.W_down_fp8 = W_down_fp8_typed;
    host_state.W_gateup_fp8_sf = W_gateup_fp8_sf_typed;
    host_state.W_down_fp8_sf = W_down_fp8_sf_typed;
    host_state.recv_tokens_fp8 = recv_tokens_fp8;
    host_state.recv_tokens_fp8_sf = recv_tokens_fp8_sf;
    host_state.input_fp8_workspace = input_fp8_workspace;
    host_state.input_fp8_sf_workspace = input_fp8_sf_workspace;
    host_state.act_fp8_workspace = act_fp8_workspace;
    host_state.act_fp8_sf_workspace = act_fp8_sf_workspace;
    host_state.compute_fp8_tma = d_compute_fp8_tma;
    host_state.compute_fp8_down_tma = d_compute_fp8_down_tma;
    host_state.group_input_fp8_tma = d_group_input_fp8_tma;
    host_state.fp8_hidden_scale_k_packed = fp8_hidden_scale_k_packed;
    host_state.fp8_intermediate_scale_k_packed = fp8_intermediate_scale_k_packed;

    // Compute output
    host_state.combine_input = combine_input;
    host_state.combine_input_topk_weights = combine_input_topk_weights;
    host_state.combine_input_src_meta = combine_input_src_meta;
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
    host_state.total_expert_slots = static_cast<int>(total_expert_slots);
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

    // RDMA-buffer-reuse plumbing (borrowed symmetric mailboxes + state-owned gate flag)
    host_state.rdma_reuse_dispatch_quiet_done = rdma_reuse_dispatch_quiet_done;
    host_state.rdma_reuse_combine_clear_done = rdma_reuse_combine_clear_done;
    host_state.rdma_reuse_prelude_enable = rdma_reuse_prelude_enable;
    int* rdma_reuse_prelude_done = nullptr;
    CUDA_CHECK(cudaMalloc(&rdma_reuse_prelude_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(rdma_reuse_prelude_done, 0, sizeof(int)));
    host_state.rdma_reuse_prelude_done = rdma_reuse_prelude_done;

    // Combine state
    host_state.num_combine_sms = num_combine_sms;
    host_state.num_combine_channels = num_combine_channels;
    host_state.num_logical_channels = num_logical_channels;
    host_state.token_compute_expected = token_compute_expected;
    host_state.compute_output_slot = compute_output_slot;
    host_state.bwd_fc1_input = bwd_fc1_input;
    host_state.bwd_preact = bwd_preact;
    host_state.owns_bwd_fc1_input = owns_bwd_fc1_input;
    host_state.owns_bwd_preact = owns_bwd_preact;
    host_state.fwd_slot_map = fwd_slot_map;
    host_state.owns_fwd_slot_map = owns_fwd_slot_map;
    host_state.token_nhits = token_nhits;
    host_state.token_slot_list = token_slot_list;
    host_state.priority_token_cursor = priority_token_cursor;
    host_state.expert_batch_enqueued = expert_batch_enqueued;
#if MK_PERF_TRACE_ARGS
    host_state.expert_batch_enqueue_ts = expert_batch_enqueue_ts;
    host_state.token_priority_dep_count = token_priority_dep_count;
#endif
    host_state.priority_batch_skip_epoch = priority_batch_skip_epoch;
    host_state.priority_batch_retry_epoch = priority_batch_retry_epoch;
    host_state.max_batches_per_expert = max_batches_per_expert;

    // Combine infrastructure.
    // RDMA-buffer reuse: combine reuses the single dispatch RDMA region. Both forward and backward
    // enable the in-kernel combine prelude, which guarantees all cross-machine dispatch has fully
    // drained and the combine metadata is re-cleared (with a cross-rank barrier) before combine
    // touches the region. The old separate "combine half" has been removed (allocation halved), so
    // reuse is mandatory; the prelude MUST be enabled whenever this state drives a combine.
    EP_HOST_ASSERT(rdma_reuse_prelude_enable && "combine RDMA reuse requires the prelude enabled");
    void* combine_rdma_ptr = rdma_buffer_ptr;

#if MK_PERF_TRACE_ENABLED
    int64_t* perf_dispatch_lch_ts;
    int64_t* perf_combine_lch_ts;
    constexpr int NLP = MegaKernelState::MK_PERF_NUM_LCH_PHASES;
    CUDA_CHECK(cudaMalloc(&perf_dispatch_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(perf_dispatch_lch_ts, 0, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&perf_combine_lch_ts, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(perf_combine_lch_ts, 0, num_logical_channels * 2 * NLP * sizeof(int64_t)));
    host_state.perf_dispatch_lch_ts = perf_dispatch_lch_ts;
    host_state.perf_combine_lch_ts = perf_combine_lch_ts;

    int64_t* perf_compute_task;
    int* perf_compute_task_count;
    const size_t compute_task_bytes = (size_t)max_compute_tasks * MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&perf_compute_task, compute_task_bytes));
    CUDA_CHECK(cudaMemset(perf_compute_task, 0, compute_task_bytes));
    CUDA_CHECK(cudaMalloc(&perf_compute_task_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(perf_compute_task_count, 0, sizeof(int)));
    host_state.perf_compute_task = perf_compute_task;
    host_state.perf_compute_task_count = perf_compute_task_count;

    const size_t gather_task_bytes = (size_t)max_compute_tasks * MegaKernelState::MK_PERF_NUM_GATHER_FIELDS * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&host_state.perf_gather_task, gather_task_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_gather_task, 0, gather_task_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_gather_task_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(host_state.perf_gather_task_count, 0, sizeof(int)));

    const size_t async_pub_bytes = (size_t)num_pub_warps_total * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_start_ts, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_start_ts, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_end_ts, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_end_ts, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_publish_all_done_ts, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_async_publish_all_done_ts, 0, sizeof(int64_t)));

    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_ts, 2 * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_ts, 0, 2 * sizeof(int64_t)));
#endif

#if MK_PERF_TRACE_ARGS
    const size_t dispatch_round_trace_count = static_cast<size_t>(num_logical_channels) *
                                              (num_ranks / NUM_MAX_NVL_PEERS);
    CUDA_CHECK(cudaMalloc(&host_state.perf_dispatch_round_trace,
                          dispatch_round_trace_count * sizeof(MegaKernelState::DispatchRoundTrace)));
    CUDA_CHECK(cudaMemset(host_state.perf_dispatch_round_trace, 0,
                          dispatch_round_trace_count * sizeof(MegaKernelState::DispatchRoundTrace)));

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
    int64_t* perf_comb_wait_top_nhits;
    int64_t* perf_comb_wait_top_priority_deps;
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
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_top_nhits, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_top_nhits, 0, acc_bytes));
    CUDA_CHECK(cudaMalloc(&perf_comb_wait_top_priority_deps, acc_bytes));
    CUDA_CHECK(cudaMemset(perf_comb_wait_top_priority_deps, 0, acc_bytes));
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
    host_state.perf_comb_wait_top_nhits = perf_comb_wait_top_nhits;
    host_state.perf_comb_wait_top_priority_deps = perf_comb_wait_top_priority_deps;
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
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_start_first_negative_ts, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_start_first_negative_ts, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_end_first_negative_ts, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_end_first_negative_ts, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_pair_ready_ts, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_pair_ready_ts, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_poll_count, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_poll_count, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_mixed_poll_count, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_mixed_poll_count, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_slowest_rdma, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_slowest_rdma, 0xff, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_src_nvl, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_src_nvl, 0xff, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_raw_start, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_raw_start, 0, disp_recv_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_allrecv_prefix_raw_end, disp_recv_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_allrecv_prefix_raw_end, 0, disp_recv_bytes));
    const size_t disp_sender_meta_bytes = (size_t)num_logical_channels * 2 * (host_state.num_ranks / NUM_MAX_NVL_PEERS) * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_sender_meta_build_start_ts, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_sender_meta_build_start_ts, 0, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_sender_meta_ready_ts, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_sender_meta_ready_ts, 0, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_sender_meta_put_begin_ts, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_sender_meta_put_begin_ts, 0, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_sender_meta_put_end_ts, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_sender_meta_put_end_ts, 0, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_sender_meta_raw_2, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_sender_meta_raw_2, 0, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_sender_meta_raw_3, disp_sender_meta_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_sender_meta_raw_3, 0, disp_sender_meta_bytes));
    const size_t disp_prefix_prod_bytes = disp_recv_bytes * (host_state.num_ranks / NUM_MAX_NVL_PEERS);
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_wait_start_ts, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_wait_start_ts, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_ready_ts, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_ready_ts, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_first_negative_ts, disp_prefix_prod_bytes * 4));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_first_negative_ts, 0, disp_prefix_prod_bytes * 4));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_mixed_poll_count, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_mixed_poll_count, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_poll_count, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_poll_count, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_raw_0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_raw_0, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_raw_1, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_raw_1, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_raw_2, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_raw_2, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_meta_raw_3, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_meta_raw_3, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_store_start_done_ts, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_store_start_done_ts, 0, disp_prefix_prod_bytes));
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
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_raw_start, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_raw_start, 0, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_disp_prefix_raw_end, disp_prefix_prod_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_disp_prefix_raw_end, 0, disp_prefix_prod_bytes));
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
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_wait_ring_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_wait_ring_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_poll_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_poll_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_gap_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_gap_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_start_gap_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_start_gap_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_empty_gap_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_empty_gap_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_done_recheck_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_done_recheck_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_ring_load_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_ring_load_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_head_release_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_head_release_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_syncwarp_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_syncwarp_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_batch_wall_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_batch_wall_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_batch_accounted_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_batch_accounted_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_batch_unattributed_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_batch_unattributed_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_token_gap_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_token_gap_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_helper_unattributed_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_helper_unattributed_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_finish_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_finish_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_work_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_work_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_scan_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_scan_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_atomic_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_atomic_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_fence_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_fence_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_store_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_store_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_drain_ns, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_drain_ns, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_tokens, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_tokens, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_batch_count, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_batch_count, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_head_release_count, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_head_release_count, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_max_batch, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_max_batch, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_local_hit_tokens, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_local_hit_tokens, 0, async_pub_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_async_pub_local_hits, async_pub_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_async_pub_local_hits, 0, async_pub_bytes));
    host_state.perf_disp_pub_scan_ns = perf_disp_pub_scan_ns;
    host_state.perf_disp_pub_atomic_ns = perf_disp_pub_atomic_ns;
    host_state.perf_disp_pub_fence_ns = perf_disp_pub_fence_ns;
    host_state.perf_disp_pub_store_ns = perf_disp_pub_store_ns;

    // Scheduler bridge detailed timers.
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
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_already_normal, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_already_normal, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_already_priority, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_already_priority, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_priority_already_tail, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_priority_already_tail, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_normal_after_priority_ns, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_normal_after_priority_ns, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_normal_after_priority_count, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_normal_after_priority_count, 0, sizeof(int64_t)));
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
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_done_seen_ts, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_done_seen_ts, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_recv_count_advance_ts, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_recv_count_advance_ts, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_recv_count_advance_expert, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_recv_count_advance_expert, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_recv_count_advance_old, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_recv_count_advance_old, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_recv_count_advance_new, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_recv_count_advance_new, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_normal_enqueue_attempt_ts, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_normal_enqueue_attempt_ts, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_normal_enqueue_success_ts, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_normal_enqueue_success_ts, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_task_publish_ts, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_task_publish_ts, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_task_source, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_task_source, 0, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_task_expert, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_task_expert, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_task_batch, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_task_batch, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_task_start_slot, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_task_start_slot, 0xff, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&host_state.perf_sched_first_task_num_tokens, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(host_state.perf_sched_first_task_num_tokens, 0, sizeof(int64_t)));

    // Root-cause diagnostic parallel arrays (one entry per compute-task slot).
    const size_t diag_i64_bytes = (size_t)max_compute_tasks * sizeof(int64_t);
    const size_t diag_i32_bytes = (size_t)max_compute_tasks * sizeof(int);
    auto alloc_diag_i64 = [&](int64_t** p) {
        CUDA_CHECK(cudaMalloc(p, diag_i64_bytes));
        CUDA_CHECK(cudaMemset(*p, 0, diag_i64_bytes));
    };
    CUDA_CHECK(cudaMalloc(&host_state.perf_compute_multi_expert_rows, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_compute_multi_expert_rows, 0, diag_i32_bytes));
    CUDA_CHECK(cudaMalloc(&host_state.perf_compute_task_has_multi, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_compute_task_has_multi, 0, diag_i32_bytes));
    alloc_diag_i64(&host_state.perf_task_publish_ts);
    CUDA_CHECK(cudaMalloc(&host_state.perf_task_source, diag_i32_bytes));
    CUDA_CHECK(cudaMemset(host_state.perf_task_source, 0, diag_i32_bytes));
    alloc_diag_i64(&host_state.perf_task_pop_start_ts);
    alloc_diag_i64(&host_state.perf_task_pop_done_ts);
    alloc_diag_i64(&host_state.perf_task_bcast_done_ts);
    alloc_diag_i64(&host_state.perf_task_start_ts);
    alloc_diag_i64(&host_state.perf_task_prev_end_ts);
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
    host_state.num_rdma_bytes = num_rdma_bytes;
    host_state.num_nvl_bytes = num_nvl_bytes;
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

    // Combine writes directly into caller-owned Torch outputs when provided.
    int4* combined_x = external_combined_x;
    const bool owns_combined_x = combined_x == nullptr;
    if (owns_combined_x) {
        arena = &transient_arena;
        CUDA_CHECK(cudaMalloc(&combined_x, num_tokens * hidden_int4 * sizeof(int4)));
        CUDA_CHECK(cudaMemset(combined_x, 0, num_tokens * hidden_int4 * sizeof(int4)));
        arena = &persistent_arena;
    }
    host_state.combined_x = combined_x;
    host_state.owns_combined_x = owns_combined_x;

    float* combined_topk_weights = external_combined_topk_weights;
    const bool owns_combined_topk_weights = combined_topk_weights == nullptr;
    if (owns_combined_topk_weights) {
        arena = &transient_arena;
        CUDA_CHECK(cudaMalloc(&combined_topk_weights, num_tokens * num_topk * sizeof(float)));
        CUDA_CHECK(cudaMemset(combined_topk_weights, 0, num_tokens * num_topk * sizeof(float)));
        arena = &persistent_arena;
    }
    host_state.combined_topk_weights = combined_topk_weights;
    host_state.owns_combined_topk_weights = owns_combined_topk_weights;

    const int mk_num_fills = static_cast<int>(mk_fill_recs.size());
    void* fill_desc_buf = nullptr;
    int mk_num_descs = 0;  // after chunking
    std::vector<FusedFillDesc> host_descs;
    if (mk_num_fills > 0) {
        // Split large buffers into chunks so each block has roughly equal work.
        // Target: each chunk is at most CHUNK_WORDS uint32s (~256KB).
        constexpr size_t CHUNK_WORDS = 64 * 1024;  // 256KB per chunk
        host_descs.reserve(mk_num_fills * 2);
        for (int i = 0; i < mk_num_fills; ++i) {
            const uint32_t bval = static_cast<uint32_t>(mk_fill_recs[i].byte_value & 0xff);
            const uint32_t word = bval * 0x01010101u;
            size_t total_bytes = mk_fill_recs[i].bytes;
            uint8_t* base = reinterpret_cast<uint8_t*>(mk_fill_recs[i].ptr);
            size_t offset = 0;
            while (offset < total_bytes) {
                size_t chunk_bytes = min(total_bytes - offset, CHUNK_WORDS * sizeof(uint32_t));
                // Align chunk_bytes down to uint32 boundary (all allocs are 128B aligned).
                chunk_bytes = (chunk_bytes / sizeof(uint32_t)) * sizeof(uint32_t);
                if (chunk_bytes == 0) chunk_bytes = total_bytes - offset;
                host_descs.push_back(FusedFillDesc{base + offset, chunk_bytes, word});
                offset += chunk_bytes;
            }
        }
        mk_num_descs = static_cast<int>(host_descs.size());
        CUDA_CHECK(cudaMalloc(&fill_desc_buf, static_cast<size_t>(mk_num_descs) * sizeof(FusedFillDesc)));
    }
    host_state.fused_fill_desc_buf = fill_desc_buf;

    // Copy state + fill descs to device in one async batch on the current stream.
    cudaStream_t init_stream = c10::cuda::getCurrentCUDAStream().stream();
    MegaKernelState* device_state;
    CUDA_CHECK(cudaMalloc(&device_state, sizeof(MegaKernelState)));
    store_arena_chunks(persistent_arena, host_state.persistent_arena_chunks, &host_state.persistent_arena_chunk_count);
    store_arena_chunks(transient_arena, host_state.transient_arena_chunks, &host_state.transient_arena_chunk_count);
    if (host_state_out != nullptr)
        *host_state_out = host_state;
    CUDA_CHECK(cudaMemcpyAsync(device_state, &host_state, sizeof(MegaKernelState), cudaMemcpyHostToDevice, init_stream));
    if (mk_num_descs > 0) {
        CUDA_CHECK(cudaMemcpyAsync(fill_desc_buf, host_descs.data(),
                              static_cast<size_t>(mk_num_descs) * sizeof(FusedFillDesc),
                              cudaMemcpyHostToDevice, init_stream));
    }
    if (mk_num_descs > 0 || device_expert_count_mapped != nullptr) {
        const int fill_blocks = std::max(1, mk_num_descs);
        fused_fill_kernel<<<fill_blocks, 512, 0, init_stream>>>(
            static_cast<const FusedFillDesc*>(fill_desc_buf), mk_num_descs,
            device_expert_count_mapped,
            expert_slot_base,
            expert_count,
            num_local_experts);
        CUDA_CHECK(cudaGetLastError());
    }
    // Caller is responsible for ensuring host_state (stack) and host_descs (vector)
    // remain valid until the stream drains. When host_state_out != nullptr the caller
    // receives the snapshot and can control synchronization; otherwise we sync here to
    // protect the stack-local lifetime.
    CUDA_CHECK(cudaStreamSynchronize(init_stream));
#undef cudaMemset
#undef cudaMalloc

    return device_state;
}

void free_megakernel_state_v7(MegaKernelState* device_state, const MegaKernelState* cached_host_state) {
#define cudaFree(p) mk_caching_free(p)
    MegaKernelState host_state_copy;
    MegaKernelState* hs = &host_state_copy;
    if (cached_host_state != nullptr) {
        *hs = *cached_host_state;
    } else {
        CUDA_CHECK(cudaMemcpy(hs, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    }
#define host_state (*hs)

    if (host_state.persistent_arena_chunk_count > 0 || host_state.transient_arena_chunk_count > 0) {
        CUDA_CHECK(free_arena_chunks(host_state.transient_arena_chunks, host_state.transient_arena_chunk_count));
        CUDA_CHECK(free_arena_chunks(host_state.persistent_arena_chunks, host_state.persistent_arena_chunk_count));
        return;
    }

    CUDA_CHECK(cudaFree(host_state.expert_recv_count));
    CUDA_CHECK(cudaFree(host_state.expert_slot_ready));
    CUDA_CHECK(cudaFree(host_state.dispatch_done));
    CUDA_CHECK(cudaFree(host_state.dispatch_done_count));
    CUDA_CHECK(cudaFree(host_state.timeout_log_counters));
    CUDA_CHECK(cudaFree(host_state.recv_tokens));
    CUDA_CHECK(cudaFree(host_state.expert_token_offsets));
    CUDA_CHECK(cudaFree(host_state.expert_slot_base));
    CUDA_CHECK(cudaFree(host_state.expert_count));
    CUDA_CHECK(cudaFree(host_state.recv_token_source_info));
    CUDA_CHECK(cudaFree(host_state.recv_token_route_weights));
    CUDA_CHECK(cudaFree(host_state.recv_src_meta));
    CUDA_CHECK(cudaFree(host_state.token_compute_expected));
    CUDA_CHECK(cudaFree(host_state.compute_output_slot));
    if (host_state.owns_bwd_fc1_input) CUDA_CHECK(cudaFree(host_state.bwd_fc1_input));
    if (host_state.owns_bwd_preact) CUDA_CHECK(cudaFree(host_state.bwd_preact));
    if (host_state.owns_fwd_slot_map) CUDA_CHECK(cudaFree(host_state.fwd_slot_map));
    CUDA_CHECK(cudaFree(host_state.token_nhits));
    CUDA_CHECK(cudaFree(host_state.token_slot_list));
    CUDA_CHECK(cudaFree(host_state.priority_token_cursor));
    CUDA_CHECK(cudaFree(host_state.expert_batch_enqueued));
#if MK_PERF_TRACE_ARGS
    CUDA_CHECK(cudaFree(host_state.expert_batch_enqueue_ts));
    CUDA_CHECK(cudaFree(host_state.token_priority_dep_count));
#endif
    CUDA_CHECK(cudaFree(host_state.priority_batch_skip_epoch));
    CUDA_CHECK(cudaFree(host_state.priority_batch_retry_epoch));
    CUDA_CHECK(cudaFree(host_state.compute_group_barrier));
    CUDA_CHECK(cudaFree(host_state.compute_group_phase));
    CUDA_CHECK(cudaFree(host_state.compute_tasks));
    CUDA_CHECK(cudaFree(host_state.compute_task_head));
    CUDA_CHECK(cudaFree(host_state.compute_task_tail));
    CUDA_CHECK(cudaFree(host_state.compute_task_reserve_tail));
    CUDA_CHECK(cudaFree(host_state.compute_enqueue_done));
    CUDA_CHECK(cudaFree(host_state.scheduler_done_count));
    CUDA_CHECK(cudaFree(host_state.priority_scheduler_done));
    CUDA_CHECK(cudaFree(host_state.expert_enqueue_cursor));
    CUDA_CHECK(cudaFree(host_state.compute_group_task_idx));
    CUDA_CHECK(cudaFree(host_state.token_done_count));
    CUDA_CHECK(cudaFree(host_state.gather_claimed));
    CUDA_CHECK(cudaFree(host_state.combine_token_ready));
    CUDA_CHECK(cudaFree(host_state.gather_ready_queue));
    CUDA_CHECK(cudaFree(host_state.gather_ready_head));
    CUDA_CHECK(cudaFree(host_state.gather_ready_tail));
    CUDA_CHECK(cudaFree(host_state.gather_ready_reserve_tail));
    CUDA_CHECK(cudaFree(host_state.gather_scan_cursor));
    CUDA_CHECK(cudaFree(host_state.gather_task_count));
    CUDA_CHECK(cudaFree(host_state.gather_task_tokens));
    CUDA_CHECK(cudaFree(host_state.gather_task_nhits));
    CUDA_CHECK(cudaFree(host_state.combine_done_count));
    CUDA_CHECK(cudaFree(host_state.combine_all_done));
    CUDA_CHECK(cudaFree(host_state.pending_topk_idx));
    CUDA_CHECK(cudaFree(host_state.pending_topk_weights));
    CUDA_CHECK(cudaFree(host_state.pending_meta));
    CUDA_CHECK(cudaFree(host_state.pub_ring));
    CUDA_CHECK(cudaFree(host_state.pub_ring_head));
    CUDA_CHECK(cudaFree(host_state.pub_ring_tail));
    CUDA_CHECK(cudaFree(host_state.recv_warp_done));
    CUDA_CHECK(cudaFree(host_state.publish_warp_done));
    CUDA_CHECK(cudaFree(host_state.publish_done_count));
    CUDA_CHECK(cudaFree(host_state.publish_all_done));
    if (host_state.owns_combined_x) CUDA_CHECK(cudaFree(host_state.combined_x));
    if (host_state.owns_combined_topk_weights) CUDA_CHECK(cudaFree(host_state.combined_topk_weights));
    CUDA_CHECK(cudaFree(host_state.combine_input));
    CUDA_CHECK(cudaFree(host_state.combine_input_topk_weights));
    CUDA_CHECK(cudaFree(host_state.combine_input_src_meta));
    CUDA_CHECK(cudaFree(host_state.combine_rdma_head_work));
    CUDA_CHECK(cudaFree(host_state.combine_nvl_head_work));
    CUDA_CHECK(cudaFree(host_state.gemm_workspace));
    // TMA descriptors are owned by the persistent thread-local cache (see
    // allocate_megakernel_state_v7). The state only holds borrowed pointers, so it
    // must NOT free them here — the cache frees them on eviction.
    CUDA_CHECK(cudaFree(host_state.recv_tokens_fp8));
    CUDA_CHECK(cudaFree(host_state.recv_tokens_fp8_sf));
    CUDA_CHECK(cudaFree(host_state.input_fp8_workspace));
    CUDA_CHECK(cudaFree(host_state.input_fp8_sf_workspace));
    CUDA_CHECK(cudaFree(host_state.act_fp8_workspace));
    CUDA_CHECK(cudaFree(host_state.act_fp8_sf_workspace));
    // FP8 TMA descriptors are also owned by the persistent cache; do not free here.
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
    CUDA_CHECK(cudaFree(host_state.rdma_reuse_prelude_done));
    CUDA_CHECK(cudaFree(host_state.dispatch_round_barrier));
    CUDA_CHECK(cudaFree(host_state.combine_channel_barrier));
    CUDA_CHECK(cudaFree(host_state.fused_fill_desc_buf));
#if MK_PERF_TRACE_ENABLED
    CUDA_CHECK(cudaFree(host_state.perf_dispatch_lch_ts));
    CUDA_CHECK(cudaFree(host_state.perf_combine_lch_ts));
    CUDA_CHECK(cudaFree(host_state.perf_compute_task));
    CUDA_CHECK(cudaFree(host_state.perf_compute_task_count));
    CUDA_CHECK(cudaFree(host_state.perf_gather_task));
    CUDA_CHECK(cudaFree(host_state.perf_gather_task_count));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_end_ts));
    CUDA_CHECK(cudaFree(host_state.perf_async_publish_all_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_sched_ts));
#endif
#if MK_PERF_TRACE_ARGS
    CUDA_CHECK(cudaFree(host_state.perf_dispatch_round_trace));
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
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_start_first_negative_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_end_first_negative_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_pair_ready_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_poll_count));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_mixed_poll_count));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_slowest_rdma));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_src_nvl));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_raw_start));
    CUDA_CHECK(cudaFree(host_state.perf_disp_allrecv_prefix_raw_end));
    CUDA_CHECK(cudaFree(host_state.perf_disp_sender_meta_build_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_sender_meta_ready_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_sender_meta_put_begin_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_sender_meta_put_end_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_sender_meta_raw_2));
    CUDA_CHECK(cudaFree(host_state.perf_disp_sender_meta_raw_3));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_wait_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_ready_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_first_negative_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_mixed_poll_count));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_poll_count));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_raw_0));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_raw_1));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_raw_2));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_raw_3));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_store_start_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_store_begin_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_publish_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_fence_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_store_to_fence_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_meta_wait_ns));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_raw_start));
    CUDA_CHECK(cudaFree(host_state.perf_disp_prefix_raw_end));
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
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_wait_ring_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_poll_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_gap_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_start_gap_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_empty_gap_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_done_recheck_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_ring_load_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_head_release_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_syncwarp_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_batch_wall_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_batch_accounted_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_batch_unattributed_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_token_gap_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_helper_unattributed_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_finish_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_work_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_scan_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_atomic_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_fence_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_store_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_drain_ns));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_batch_count));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_head_release_count));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_max_batch));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_local_hit_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_async_pub_local_hits));
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
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_top_nhits));
    CUDA_CHECK(cudaFree(host_state.perf_comb_wait_top_priority_deps));
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
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_already_normal));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_already_priority));
    CUDA_CHECK(cudaFree(host_state.perf_sched_priority_already_tail));
    CUDA_CHECK(cudaFree(host_state.perf_sched_normal_after_priority_ns));
    CUDA_CHECK(cudaFree(host_state.perf_sched_normal_after_priority_count));
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
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_done_seen_ts));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_recv_count_advance_ts));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_recv_count_advance_expert));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_recv_count_advance_old));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_recv_count_advance_new));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_normal_enqueue_attempt_ts));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_normal_enqueue_success_ts));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_task_publish_ts));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_task_source));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_task_expert));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_task_batch));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_task_start_slot));
    CUDA_CHECK(cudaFree(host_state.perf_sched_first_task_num_tokens));
    CUDA_CHECK(cudaFree(host_state.perf_compute_multi_expert_rows));
    CUDA_CHECK(cudaFree(host_state.perf_compute_task_has_multi));
    CUDA_CHECK(cudaFree(host_state.perf_task_publish_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_source));
    CUDA_CHECK(cudaFree(host_state.perf_task_pop_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_pop_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_bcast_done_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_start_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_prev_end_ts));
    CUDA_CHECK(cudaFree(host_state.perf_task_prev_gap_ns));
    CUDA_CHECK(cudaFree(host_state.perf_task_pop_attempts));
    CUDA_CHECK(cudaFree(host_state.perf_task_cas_failures));
    CUDA_CHECK(cudaFree(host_state.perf_task_group_id));
#endif
    CUDA_CHECK(cudaFree(device_state));
#undef host_state
#undef cudaFree
}

// Release the forward-only working buffers after the kernel has written the caller-owned
// output. The backward pass allocates a fresh v7 state and only reuses the saved-activation
// buffers (bwd_fc1_input / bwd_preact / fwd_slot_map) plus expert_count from this forward
// state (see allocate_megakernel_backward_state). None of the buffers freed here are read
// by the backward, so releasing them now removes them from the forward->backward resident
// set. Each freed pointer is nulled so the eventual free_megakernel_state_v7 skips it
// (mk_caching_free is null-safe), avoiding a double free.
void free_megakernel_forward_transient(MegaKernelState* device_state) {
    if (device_state == nullptr)
        return;
    MegaKernelState hs;
    CUDA_CHECK(cudaMemcpy(&hs, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    if (hs.transient_arena_chunk_count > 0) {
        CUDA_CHECK(free_arena_chunks(hs.transient_arena_chunks, hs.transient_arena_chunk_count));
        hs.transient_arena_chunk_count = 0;
        hs.recv_tokens = nullptr;
        hs.compute_output_slot = nullptr;
        hs.combine_input = nullptr;
        hs.combine_input_topk_weights = nullptr;
        hs.combine_input_src_meta = nullptr;
        hs.combine_rdma_head_work = nullptr;
        hs.combine_nvl_head_work = nullptr;
        hs.gemm_workspace = nullptr;
        hs.output_accum = nullptr;
        hs.send_rdma_head = nullptr;
        hs.send_nvl_head = nullptr;
        if (hs.owns_combined_x)
            hs.combined_x = nullptr;
        if (hs.owns_combined_topk_weights)
            hs.combined_topk_weights = nullptr;
        CUDA_CHECK(cudaMemcpy(device_state, &hs, sizeof(MegaKernelState), cudaMemcpyHostToDevice));
        return;
    }
    auto free_and_null = [](auto*& ptr) {
        CUDA_CHECK(mk_caching_free(static_cast<void*>(ptr)));
        ptr = nullptr;
    };
    free_and_null(hs.recv_tokens);
    free_and_null(hs.compute_output_slot);
    free_and_null(hs.combine_input);
    free_and_null(hs.combine_input_topk_weights);
    free_and_null(hs.combine_input_src_meta);
    free_and_null(hs.combine_rdma_head_work);
    free_and_null(hs.combine_nvl_head_work);
    free_and_null(hs.gemm_workspace);
    free_and_null(hs.output_accum);
    free_and_null(hs.send_rdma_head);
    free_and_null(hs.send_nvl_head);
    if (hs.owns_combined_x)
        free_and_null(hs.combined_x);
    if (hs.owns_combined_topk_weights)
        free_and_null(hs.combined_topk_weights);
    CUDA_CHECK(cudaMemcpy(device_state, &hs, sizeof(MegaKernelState), cudaMemcpyHostToDevice));
}

// Host-only version: uses the cached host_state snapshot directly, avoiding
// D2H + H2D cudaMemcpy entirely. The device_state is NOT updated (the backward
// re-creates its own v7 state from the host cache, and eventual free_megakernel_state_v7
// handles the remaining persistent arena chunks via the device copy). This path removes
// three memcpy operations from the forward critical path.
void free_megakernel_forward_transient_from_host(MegaKernelState* host_state) {
    if (host_state == nullptr)
        return;
    if (host_state->transient_arena_chunk_count > 0) {
        CUDA_CHECK(free_arena_chunks(host_state->transient_arena_chunks, host_state->transient_arena_chunk_count));
        host_state->transient_arena_chunk_count = 0;
        return;
    }
    // Fallback: individually free each transient pointer from host snapshot.
    auto free_ptr = [](auto*& ptr) {
        if (ptr != nullptr) {
            CUDA_CHECK(mk_caching_free(static_cast<void*>(ptr)));
            ptr = nullptr;
        }
    };
    free_ptr(host_state->recv_tokens);
    free_ptr(host_state->compute_output_slot);
    free_ptr(host_state->combine_input);
    free_ptr(host_state->combine_input_topk_weights);
    free_ptr(host_state->combine_input_src_meta);
    free_ptr(host_state->combine_rdma_head_work);
    free_ptr(host_state->combine_nvl_head_work);
    free_ptr(host_state->gemm_workspace);
    free_ptr(host_state->output_accum);
    free_ptr(host_state->send_rdma_head);
    free_ptr(host_state->send_nvl_head);
    if (host_state->owns_combined_x)
        free_ptr(host_state->combined_x);
    if (host_state->owns_combined_topk_weights)
        free_ptr(host_state->combined_topk_weights);
}

int get_megakernel_compute_batch_size() {
    return COMPUTE_BATCH_SIZE;
}

void get_megakernel_expert_counts(
    MegaKernelState* device_state,
    int* expert_counts,
    int num_local_experts
) {
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expert_counts, host_state.expert_count,
                          (size_t)num_local_experts * sizeof(int), cudaMemcpyDeviceToHost));
}

void get_megakernel_backward_dimensions(
    MegaKernelState* device_state,
    int* num_tokens,
    int* hidden,
    int* intermediate,
    int* num_topk,
    int* num_local_experts
) {
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    *num_tokens = host_state.num_tokens;
    *hidden = host_state.hidden_dim;
    *intermediate = host_state.intermediate_dim;
    *num_topk = host_state.num_topk;
    *num_local_experts = host_state.num_local_experts;
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

// Debug entry points intentionally live beside the copied forward implementation.
// Keeping this wrapper thin guarantees that debug-forward and MK-v7 execute the
// same kernel specialization while the forward implementation is still moving.
void launch_megakernel_debug_forward(
    MegaKernelState* device_state,
    const MegaKernelState* host_state,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream
) {
    launch_megakernel_v7(
        device_state, host_state, total_sms, smem_size, stage, compute_dtype, stream);
}

// ============================================================================
// Backward megakernel (MK-v7 debug) — full-fused, reuses the forward comm.
//
// The backward re-runs the SAME fused megakernel (dispatch + scheduler + gather + combine)
// on a patched copy of the forward MegaKernelState, swapping ONLY the compute role for a
// backward compute worker:
//   x            := grad_output   (so dispatch scatters grad_output into combine_input =: grad_down)
//   combined_x   := grad_input    (so combine reduces grad_xperm over a token's hits =: dX)
//   compute      := compute_backward_worker (reads grad_down + saved X, writes grad_xperm)
//
// Correctness: dispatch/combine are driven by routing that is deterministic per RECV_TOKEN
// (prefix-sum indices), while the expert SLOT index is assigned by a non-deterministic
// atomicAdd and therefore differs run-to-run. Hence the forward saved fc1 input by recv_token
// (fwd.bwd_fc1_input[recv_token]); the backward gathers grad_down and X by the same recv_token
// index, so everything stays aligned regardless of the re-run's slot permutation.
//
// Phase 2 math (per compute task; only grad_input / dX is produced — no dW / dprobs):
//   grad_down = combine_input row for this recv_token (filled by the re-run dispatch)
//   X_perm    = fwd.bwd_fc1_input[recv_token]
//   GU        = X_perm @ W_gateup^T           [M, 2I]   gate=GU[:,0::2], up=GU[:,1::2]
//   grad_act  = grad_down @ W_down            [M, I]
//   SwiGLU bwd (route folded in fwd act = silu(gate)*up*route):
//     g_pre=grad_act*route ; g_up=g_pre*silu(gate) ; g_gate=g_pre*up*silu'(gate)
//   grad_gu   = interleave(g_gate, g_up)      [M, 2I]
//   grad_xperm= grad_gu @ W_gateup            [M, hidden]
//   -> written to compute_output_slot/combine_input exactly like the forward down output,
//      then the reused combine reduces it into combined_x = grad_input.
//
// UMMA dgrad consumes original weights as MN-major B operands, matching DeepGEMM's
// layout handling, so no physical W_gateup_T/W_down_T materialization is needed.
// ============================================================================

struct MegaKernelBackwardState {
    MegaKernelState* fwd;                 // original forward state (buffers reused, freed by caller)
    MegaKernelState* bwd_device_state;    // patched device copy: x=grad_output, combined_x=grad_input

    const __nv_bfloat16* bwd_fc1_input;   // fwd.bwd_fc1_input [max_total_recv_tokens, hidden] (X by recv_token)
    const __nv_bfloat16* bwd_preact;      // fwd.bwd_preact [max_total_recv_tokens, num_topk, 2I]
    umma::ComputeBackwardTmaAtoms* compute_bwd_tma;
    CUtensorMap* wgrad_dgu_a_tma;         // [num_local_experts * max_batches_per_expert] batch-start A descs for wgrad_dgu_slot

    const __nv_bfloat16* grad_output;     // [num_combined_tokens, hidden] dY (== bwd_device_state->x)
    __nv_bfloat16* grad_input;            // [num_combined_tokens, hidden] dX (== bwd_device_state->combined_x)
    __nv_bfloat16* grad_w_gateup;         // [num_local_experts, 2 * intermediate, hidden]
    __nv_bfloat16* grad_w_down;           // [num_local_experts, hidden, intermediate]
    float* grad_topk_weights;             // [num_tokens, num_topk]
    __nv_bfloat16* wgrad_x_slot;           // [expert_slots, hidden]
    __nv_bfloat16* wgrad_act_slot;         // [expert_slots, intermediate], route-weighted
    __nv_bfloat16* wgrad_dz_slot;          // [expert_slots, hidden]
    __nv_bfloat16* wgrad_dgu_slot;         // [expert_slots, 2 * intermediate]
};

struct MegaKernelBackwardHostContext {
    MegaKernelBackwardState backward_state;
    MegaKernelState bwd_state;
    umma::ComputeBackwardTmaAtoms compute_bwd_tma_atoms;
    std::vector<CUtensorMap> wgrad_dgu_a_tma;
};

__device__ __forceinline__ void trace_backward_values(
    const char* stage,
    const __nv_bfloat16* values,
    int num_values,
    int rank,
    int task_idx,
    int expert_id,
    int batch_size
) {
    if (rank != 0 || task_idx != 0 || threadIdx.x != 0)
        return;
    const int sample = min(num_values, 256);
    float sum_abs = 0.0f;
    float max_abs = 0.0f;
    int nonzero = 0;
    for (int i = 0; i < sample; ++i) {
        float value = fabsf(__bfloat162float(values[i]));
        sum_abs += value;
        max_abs = max(max_abs, value);
        nonzero += value != 0.0f;
    }
    // printf("[MK-BWD-TRACE][%s] task=%d expert=%d batch=%d sample=%d sum_abs=%e max_abs=%e nonzero=%d\n",
    //        stage, task_idx, expert_id, batch_size, sample, sum_abs, max_abs, nonzero);
}

__global__ void trace_backward_boundary_kernel(
    const MegaKernelBackwardState* bs,
    int phase
) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const MegaKernelState* state = bs->bwd_device_state;
    if (state == nullptr || state->rank != 0)
        return;
    const __nv_bfloat16* values = phase == 0 ? bs->grad_output : bs->grad_input;
    trace_backward_values(
        phase == 0 ? "GRAD-OUTPUT" : "GRAD-INPUT", values,
        state->num_tokens * state->hidden_dim, state->rank, 0, -1, state->num_tokens);
    // printf("[MK-BWD-TRACE][STATE-%s] task_head=%d task_tail=%d enqueue_done=%d "
    //        "gather_head=%d gather_tail=%d dispatch_done=%d combine_done=%d\n",
    //        phase == 0 ? "BEFORE" : "AFTER",
    //        *state->compute_task_head, *state->compute_task_tail, *state->compute_enqueue_done,
    //        *state->gather_ready_head, *state->gather_ready_tail,
    //        *state->dispatch_done, *state->combine_all_done);
}

__device__ __forceinline__ float mk_bwd_bf16_from_u32(uint32_t v, int lane) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v)[lane]);
}

__device__ __forceinline__ uint32_t mk_bwd_cvt_f32x2_bf16x2(float lo, float hi) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;\n"
                 : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

__device__ __forceinline__ void mk_bwd_dswiglu_pair2_side_f32x2(
    uint32_t gu0, uint32_t gu1, uint32_t grad01, float route,
    uint32_t& out0, uint32_t& out1, uint32_t& act01, float& route_grad
) {
    float2 gate = {mk_bwd_bf16_from_u32(gu0, 0), mk_bwd_bf16_from_u32(gu1, 0)};
    float2 up = {mk_bwd_bf16_from_u32(gu0, 1), mk_bwd_bf16_from_u32(gu1, 1)};
    float2 grad_raw = {mk_bwd_bf16_from_u32(grad01, 0), mk_bwd_bf16_from_u32(grad01, 1)};
    float2 grad = {grad_raw.x * route, grad_raw.y * route};
    float2 sig = {
        1.0f / (1.0f + __expf(-gate.x)),
        1.0f / (1.0f + __expf(-gate.y))};
    float2 silu, activation, silu_grad, sig_minus_silu_sig, d_silu_grad, dgate;
    cute::mul(silu, gate, sig);
    cute::mul(activation, silu, up);
    cute::mul(silu_grad, silu, grad);
    cute::fma(sig_minus_silu_sig, silu, {-sig.x, -sig.y}, sig);
    cute::fma(d_silu_grad, sig_minus_silu_sig, grad, silu_grad);
    cute::mul(dgate, d_silu_grad, up);
    out0 = mk_bwd_cvt_f32x2_bf16x2(dgate.x, silu_grad.x);
    out1 = mk_bwd_cvt_f32x2_bf16x2(dgate.y, silu_grad.y);
    act01 = mk_bwd_cvt_f32x2_bf16x2(route * activation.x, route * activation.y);
    route_grad += grad_raw.x * activation.x + grad_raw.y * activation.y;
}

// Expert-compute backward for one compute SM. Mirrors the forward compute_worker's
// task-queue / gather / output-scatter / signaling protocol EXACTLY (so the reused
// combine handshake works), swapping only the GEMM math for the backward pass.
// state == bs->bwd_device_state: combine_input holds grad_down (filled by the re-run
// dispatch of grad_output), compute_output_slot/combine_input receive grad_xperm.
template <ComputeDType kComputeDType, bool kStopAtDispatchDone>
__device__ __forceinline__ void compute_backward_worker_core(
    MegaKernelBackwardState* bs,
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    int group_id_base,
    uint8_t* smem_buffer
) {
    if constexpr (kComputeDType == ComputeDType::kFP8E4M3) {
        if (threadIdx.x == 0 && compute_sm_idx == 0)
            printf("MK backward FP8 path is not implemented.\n");
        __threadfence_system(); trap();
        return;
    }
    MegaKernelState* state = bs->bwd_device_state;
    const int thread_id = threadIdx.x;
    const int local_warp_id = thread_id / 32;
    if (num_compute_sms <= 0 || compute_sm_idx < 0 || compute_sm_idx >= num_compute_sms)
        return;
    const int local_group_id = compute_sm_idx / COMPUTE_GROUP_SIZE;
    const int group_first_sm_idx = local_group_id * COMPUTE_GROUP_SIZE;
    const int group_size = min(COMPUTE_GROUP_SIZE, num_compute_sms - group_first_sm_idx);
    const int group_id = group_id_base + local_group_id;
    const int num_compute_groups = state->num_compute_groups;
    if (group_size <= 0 || group_id >= num_compute_groups)
        return;
    const int group_sm_idx = compute_sm_idx - group_first_sm_idx;
    const int num_warps_per_sm = blockDim.x / 32;
    const int group_warp_id = group_sm_idx * num_warps_per_sm + local_warp_id;
    const int group_num_warps = group_size * num_warps_per_sm;
    const int group_thread_id = group_sm_idx * blockDim.x + thread_id;
    const int group_num_threads = group_size * blockDim.x;
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;
    const int hidden = state->hidden_dim;
    const int intermediate = state->intermediate_dim;
    const int twoI = 2 * intermediate;
    const int num_topk = state->num_topk;
    const int hidden_int4 = hidden * sizeof(__nv_bfloat16) / sizeof(int4);

    // Per-group global workspace, same layout as the forward compute worker:
    //   input_buf[M,hidden] = grad_down (gathered)   gu_buf[M,2I] = GU scratch
    //   up_buf[M,I]         = grad_act                down_buf[M,hidden] = grad_xperm
    const int input_stride = COMPUTE_BATCH_SIZE * hidden;
    const int gu_stride    = COMPUTE_BATCH_SIZE * twoI;
    const int act_stride   = COMPUTE_BATCH_SIZE * intermediate;
    const int down_stride  = COMPUTE_BATCH_SIZE * hidden;
    const int gemm_stride  = input_stride + gu_stride + act_stride + down_stride;
    __nv_bfloat16* input_buf = state->gemm_workspace + (size_t)group_id * gemm_stride;
    __nv_bfloat16* gu_buf    = input_buf + input_stride;
    __nv_bfloat16* up_buf    = gu_buf + gu_stride;
    __nv_bfloat16* down_buf  = up_buf + act_stride;

    float* smem_wmma_buf = reinterpret_cast<float*>(smem_buffer);

    using ComputeUmmaSmemLayout = umma::DgSmemLayout<umma::kDgRunMulticast>;
    constexpr size_t kComputeUmmaBarrierBytes =
        (ComputeUmmaSmemLayout::kNumStages * 3 + ComputeUmmaSmemLayout::kNumEpilogueStages * 2 + 1) *
        sizeof(cutlass::arch::ClusterTransactionBarrier) + sizeof(uint32_t);
    constexpr size_t kComputeUmmaScratchBytes =
        ComputeUmmaSmemLayout::SMEM_CD_SIZE +
        ComputeUmmaSmemLayout::kNumStages *
            (ComputeUmmaSmemLayout::SMEM_A_SIZE_PER_STAGE + ComputeUmmaSmemLayout::SMEM_B_SIZE_PER_STAGE) +
        kComputeUmmaBarrierBytes;
    constexpr size_t kComputeWmmaScratchBytes =
        (kNumCombineForwarderWarps + 1) * 2 * WMMA_M * WMMA_N * sizeof(float);
    constexpr size_t kComputeScratchBytes =
        kComputeWmmaScratchBytes > kComputeUmmaScratchBytes ? kComputeWmmaScratchBytes : kComputeUmmaScratchBytes;
    constexpr size_t kComputeMetaOffset = (kComputeScratchBytes + alignof(int) - 1) & ~(size_t)(alignof(int) - 1);

    constexpr size_t kRecvTokenIdxBytes = COMPUTE_BATCH_SIZE * sizeof(int);
    constexpr size_t kTopkSlotBytes = COMPUTE_BATCH_SIZE * sizeof(int);
    constexpr size_t kIsSingleBytes = COMPUTE_BATCH_SIZE * sizeof(unsigned char);
    constexpr size_t kRouteWAlignPad = alignof(float) - 1;
    constexpr size_t kRouteWBytes = COMPUTE_BATCH_SIZE * sizeof(float);
    constexpr size_t kComputeBatchMetaBytes = kRecvTokenIdxBytes + kTopkSlotBytes + kIsSingleBytes + kRouteWAlignPad + kRouteWBytes;
    static_assert(kComputeMetaOffset + kComputeBatchMetaBytes <=
                  kNumCombineTMABytesPerForwarderWarp * kNumCombineForwarderWarps,
                  "backward compute dynamic smem metadata must fit after compute scratch");

    uint8_t* compute_smem = smem_buffer + kComputeMetaOffset;
    int* s_recv_token_idx = reinterpret_cast<int*>(compute_smem);
    compute_smem += kRecvTokenIdxBytes;
    int* s_topk_slot = reinterpret_cast<int*>(compute_smem);
    compute_smem += kTopkSlotBytes;
    unsigned char* s_is_single = reinterpret_cast<unsigned char*>(compute_smem);
    constexpr size_t kRouteWOffset =
        (kComputeMetaOffset + kRecvTokenIdxBytes + kTopkSlotBytes + kIsSingleBytes + alignof(float) - 1) & ~(size_t)(alignof(float) - 1);
    float* s_route_w = reinterpret_cast<float*>(smem_buffer + kRouteWOffset);

    // TMEM persistent across tasks (same optimization as forward compute_worker_core).
    bool umma_tmem_allocated = false;
#if MK_PERF_TRACE_ARGS
    int64_t last_task_end_ns = 0;
#endif

    while (true) {
#if MK_PERF_TRACE_ARGS
        int64_t pop_start_ns = 0;
        int64_t pop_done_ns = 0;
        int pop_attempts = 0;
        int cas_failures = 0;
#endif
        // ---- pop a compute task (group leader) and broadcast to the group ----
        if (group_sm_idx == 0 && thread_id == 0) {
            int task_idx = -1;
#if MK_PERF_TRACE_ARGS
            pop_start_ns = globaltimer_ns();
#endif
            while (true) {
                if constexpr (kStopAtDispatchDone) {
                    // Check if compute progress has reached the threshold to switch to combine.
                    int enqueue_done = ld_acquire_global(state->compute_enqueue_done);
                    if (enqueue_done) {
                        int tail = ld_acquire_global(state->compute_task_tail);
                        int head = ld_acquire_global(state->compute_task_head);
                        if (tail == 0 || head * 100 >= tail * COMBINE_START_HEAD_PERCENT) {
                            task_idx = -3;
                            break;
                        }
                    }
                }
                int head = ld_acquire_global(state->compute_task_head);
                int tail = ld_acquire_global(state->compute_task_tail);
                if (head >= tail) {
                    if constexpr (!kStopAtDispatchDone) {
                        if (ld_acquire_global(state->compute_enqueue_done))
                            task_idx = -2;
                    }
                    break;
                }
                if (atomicCAS(state->compute_task_head, head, head + 1) == head) {
                    task_idx = head;
#if MK_PERF_TRACE_ARGS
                    pop_done_ns = globaltimer_ns();
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
#if MK_PERF_TRACE_ARGS
                cas_failures += 1;
#endif
            }
            st_release_gpu_global(&state->compute_group_task_idx[group_id], task_idx);
        }
        compute_group_sync(state, group_id, group_size);

        int task_idx = ld_acquire_global(&state->compute_group_task_idx[group_id]);
#if MK_PERF_TRACE_ARGS
        if (group_sm_idx == 0 && thread_id == 0 && task_idx >= 0 && task_idx < state->max_compute_tasks)
            state->perf_task_bcast_done_ts[task_idx] = globaltimer_ns();
#endif
        if (task_idx == -2 || task_idx == -3) {
            // Dealloc TMEM before exiting the backward persistent loop.
            if (umma_tmem_allocated) {
                char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
                umma::umma_dealloc(cluster_smem);
            }
            break;
        }
        if (task_idx < 0) {
            if (group_sm_idx == 0 && thread_id == 0)
                __nanosleep(128);
            compute_group_sync(state, group_id, group_size);
            continue;
        }

        ComputeTask task = state->compute_tasks[task_idx];
        int expert_id = task.expert_id;
        int start_slot = task.start_slot;
        int batch_size = task.num_tokens;

#if MK_PERF_TRACE_ENABLED
        const bool perf_leader = (group_sm_idx == 0 && thread_id == 0);
        int64_t compute_task_start_ns = perf_leader ? globaltimer_ns() : 0;
        int64_t compute_task_end_ns = 0;
#endif
#if MK_PERF_TRACE_ARGS
        int64_t perf_ph_meta_ns = 0, perf_ph_input_ns = 0, perf_ph_upgemm_ns = 0;
        int64_t perf_ph_downgemm_ns = 0, perf_ph_output_ns = 0;
        int64_t perf_up_body_ns = 0, perf_down_body_ns = 0, perf_out_body_ns = 0;
        int64_t perf_sig_donecount_ns = 0, perf_sig_finalize_ns = 0;
        int64_t perf_sig_fence_ns = 0, perf_sig_publish_ns = 0;
        if (perf_leader && task_idx >= 0 && task_idx < state->max_compute_tasks) {
            state->perf_task_start_ts[task_idx] = compute_task_start_ns;
            state->perf_task_prev_end_ts[task_idx] = last_task_end_ns;
            state->perf_task_prev_gap_ns[task_idx] = last_task_end_ns == 0 ? 0 : compute_task_start_ns - last_task_end_ns;
        }
        __shared__ int s_perf_multi_expert_rows;
        __shared__ int s_perf_task_has_multi;
        if (perf_leader) {
            s_perf_multi_expert_rows = 0;
            s_perf_task_has_multi = 0;
        }
        __syncthreads();
#endif

        constexpr bool kUseUmmaCompute = (MK_COMPUTE_KERNEL != 0);
        constexpr bool kUseUmmaBwdGemm = kUseUmmaCompute && (MK_UMMA_DOWN != 0);
        const bool use_umma_bwd_for_group =
            kUseUmmaBwdGemm && group_size == COMPUTE_GROUP_SIZE &&
            state->group_input_tma != nullptr && bs->compute_bwd_tma != nullptr &&
            bs->wgrad_dgu_a_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE;
        constexpr int kUmmaClusterDim = (MK_COMPUTE_KERNEL == 2 ? 2 : 1);
        constexpr int kUmmaClustersPerGroup = COMPUTE_GROUP_SIZE / kUmmaClusterDim;

        if constexpr (kUseUmmaBwdGemm) {
            if (use_umma_bwd_for_group && local_warp_id == 0) {
                const umma::InputTmaAtom_t& prefetch_atom = state->group_input_tma[group_id];
                cute::prefetch_tma_descriptor(&prefetch_atom.a);
                cute::prefetch_tma_descriptor(&prefetch_atom.act_cd);
                cute::prefetch_tma_descriptor(&prefetch_atom.gu_a);
                cute::prefetch_tma_descriptor(&prefetch_atom.down_cd);
                cute::prefetch_tma_descriptor(&bs->compute_bwd_tma->wdown[expert_id]);
                cute::prefetch_tma_descriptor(&bs->compute_bwd_tma->wgateup[expert_id]);
            }
        }

        // ---- gather per-row (recv_token, single-hit flag, route weight) ----
        for (int i = thread_id; i < batch_size; i += blockDim.x) {
            int base_offset = state->expert_slot_base[expert_id] + start_slot + i;
            int recv_token = ld_acquire_global(&state->recv_token_source_info[base_offset * 2]);
            int topk_slot = ld_acquire_global(&state->recv_token_source_info[base_offset * 2 + 1]);
            int expected = ld_acquire_global(&state->token_compute_expected[recv_token]);
            s_recv_token_idx[i] = recv_token;
            s_topk_slot[i] = topk_slot;
            s_is_single[i] = static_cast<unsigned char>(expected == 1);
            s_route_w[i] = ld_nc_global(&state->combine_input_topk_weights[recv_token * num_topk + topk_slot]);
        }
        for (int i = batch_size + thread_id; i < COMPUTE_BATCH_SIZE; i += blockDim.x) {
            s_recv_token_idx[i] = -1;
            s_topk_slot[i] = -1;
            s_is_single[i] = 0;
            s_route_w[i] = 0.0f;
        }
        __syncthreads();
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_meta_ns = globaltimer_ns();
#endif

        // ---- gather grad_down (input_buf) and cache X in wgrad_x_slot by recv_token ----
        const int64_t slot_base64 = (int64_t)state->expert_slot_base[expert_id] + start_slot;
        const int input_vec_stride = COMPUTE_BATCH_SIZE * hidden_int4;
        const int4* combine_input_i4 = reinterpret_cast<const int4*>(state->combine_input);   // = grad_down
        const int4* bwd_x_i4 = reinterpret_cast<const int4*>(bs->bwd_fc1_input);               // = X
        int4* grad_down_i4 = reinterpret_cast<int4*>(input_buf);
        for (int idx = group_thread_id; idx < input_vec_stride; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            if (row < batch_size) {
                int rt = s_recv_token_idx[row];
                const int64_t slot = slot_base64 + row;
                const int4 grad_vec = combine_input_i4[(int64_t)rt * hidden_int4 + v];
                const int4 x_vec = bwd_x_i4[(int64_t)rt * hidden_int4 + v];
                grad_down_i4[idx] = grad_vec;
                reinterpret_cast<int4*>(bs->wgrad_dz_slot)[slot * hidden_int4 + v] = grad_vec;
                reinterpret_cast<int4*>(bs->wgrad_x_slot)[slot * hidden_int4 + v] = x_vec;
            } else {
                grad_down_i4[idx] = make_int4(0, 0, 0, 0);
            }
        }
        compute_group_sync(state, group_id, group_size);
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_input_ns = globaltimer_ns();
#endif
        if (group_sm_idx == 0) {
            trace_backward_values("GRAD-DOWN", input_buf, batch_size * hidden,
                                  state->rank, task_idx, expert_id, batch_size);
            trace_backward_values("ACTIVATION", bs->wgrad_x_slot + (size_t)slot_base64 * hidden,
                                  batch_size * hidden, state->rank, task_idx, expert_id, batch_size);
        }

        const __nv_bfloat16* Wgu_e = &state->W_gateup[(size_t)expert_id * twoI * hidden];       // [2I,hidden]
        const __nv_bfloat16* Wd_e  = &state->W_down[(size_t)expert_id * hidden * intermediate]; // [hidden,I]

        // Saved-PreAct path skips the PreAct->GU gather: dSwiGLU below reads gate/up
        // directly from bs->bwd_preact by (recv_token, topk_slot) and writes dGU
        // straight to wgrad_dgu_slot. Fallback (no saved PreAct) recomputes gate/up
        // into gu_buf via WMMA first.
        const bool preact_direct = (bs->bwd_preact != nullptr);
        __nv_bfloat16* dgu_dst = bs->wgrad_dgu_slot + (size_t)slot_base64 * twoI;
        if (!preact_direct) {
            device_gemm_bf16(down_buf, Wgu_e, gu_buf, batch_size, hidden, twoI,
                             group_warp_id, group_num_warps, local_warp_id, smem_wmma_buf);
        }
        compute_group_sync(state, group_id, group_size);

        // grad_act = grad_down @ W_down -> up_buf [M,I]. Prefer the verified
        // DeepGEMM/UMMA saved-preact baseline; keep WMMA as descriptor fallback.
        if (use_umma_bwd_for_group) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];
            uint32_t grad_act_accum_iter = 0;
            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            umma::umma_dgrad_mn_persistent(
                &in_atom.a,
                &bs->compute_bwd_tma->wdown[expert_id],
                &in_atom.act_cd,
                COMPUTE_BATCH_SIZE, intermediate, hidden,
                cluster_in_group, num_clusters,
                cluster_smem, grad_act_accum_iter);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_up_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        } else {
            device_gemm_bf16_mn(input_buf, Wd_e, up_buf, batch_size, hidden, intermediate,
                                group_warp_id, group_num_warps, local_warp_id, smem_wmma_buf);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_up_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        }
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_upgemm_ns = globaltimer_ns();
#endif
        if (group_sm_idx == 0) {
            trace_backward_values("GRAD-ACT", up_buf, batch_size * intermediate,
                                  state->rank, task_idx, expert_id, batch_size);
        }

        // SwiGLU backward and route-probability gradient. One warp owns each route row,
        // so dTopKWeight needs only a warp reduction and one scalar store (no atomics).
        const int lane_id = get_lane_id();
        const bool dswiglu_packed4 = ((intermediate & 3) == 0);
        const int intermediate_i4 = intermediate >> 2;
        for (int m = group_warp_id; m < batch_size; m += group_num_warps) {
            const float route = s_route_w[m];
            const int slot = state->expert_slot_base[expert_id] + start_slot + m;
            const int recv_token_m = s_recv_token_idx[m];
            const int topk_slot_m = s_topk_slot[m];
            // No PreAct gather: read gate/up straight from bwd_preact (slot-major by
            // (recv_token, topk_slot)) when available, else from the recomputed gu_buf.
            // Phase 3 (Step 3.3a): translate the cross-pass-stable (recv_token, topk_slot) to
            // forward's compact preact slot via fwd_slot_map, then read bwd_preact by slot.
            int fwd_preact_slot = preact_direct
                ? state->fwd_slot_map[(int64_t)recv_token_m * num_topk + topk_slot_m] : -1;
            if (fwd_preact_slot < 0) fwd_preact_slot = 0;  // safety net; should not occur for a real hit
            const __nv_bfloat16* gu_src = preact_direct
                ? bs->bwd_preact + (int64_t)fwd_preact_slot * twoI
                : gu_buf + (size_t)m * twoI;
            float route_grad = 0.0f;
            if (dswiglu_packed4) {
                const int4* gu4 = reinterpret_cast<const int4*>(gu_src);
                const int2* ga4 = reinterpret_cast<const int2*>(up_buf + (size_t)m * intermediate);
                int4* dgu4 = reinterpret_cast<int4*>(dgu_dst + (size_t)m * twoI);
                int2* wgrad_act4 = reinterpret_cast<int2*>(bs->wgrad_act_slot + (int64_t)slot * intermediate);
                for (int q = lane_id; q < intermediate_i4; q += 32) {
                    const int4 gu = gu4[q];
                    const int2 ga = ga4[q];
                    uint32_t o0, o1, o2, o3, a01, a23;
                    mk_bwd_dswiglu_pair2_side_f32x2(
                        static_cast<uint32_t>(gu.x), static_cast<uint32_t>(gu.y),
                        static_cast<uint32_t>(ga.x), route, o0, o1, a01, route_grad);
                    mk_bwd_dswiglu_pair2_side_f32x2(
                        static_cast<uint32_t>(gu.z), static_cast<uint32_t>(gu.w),
                        static_cast<uint32_t>(ga.y), route, o2, o3, a23, route_grad);
                    const int4 out = make_int4(static_cast<int>(o0), static_cast<int>(o1),
                                               static_cast<int>(o2), static_cast<int>(o3));
                    dgu4[q] = out;
                    wgrad_act4[q] = make_int2(static_cast<int>(a01), static_cast<int>(a23));
                }
            } else {
                for (int i = lane_id; i < intermediate; i += 32) {
                    float gate = __bfloat162float(gu_src[2 * i]);
                    float up = __bfloat162float(gu_src[2 * i + 1]);
                    float ga = __bfloat162float(up_buf[m * intermediate + i]);
                    float sig = 1.0f / (1.0f + __expf(-gate));
                    float silu = gate * sig;
                    float activation = silu * up;
                    float g_pre = ga * route;
                    float g_up = g_pre * silu;
                    float dsilu = sig * (1.0f + gate * (1.0f - sig));
                    float g_gate = g_pre * up * dsilu;
                    route_grad += ga * activation;
                    bs->wgrad_act_slot[(int64_t)slot * intermediate + i] =
                        __float2bfloat16(route * activation);
                    dgu_dst[(size_t)m * twoI + 2 * i] = __float2bfloat16(g_gate);
                    dgu_dst[(size_t)m * twoI + 2 * i + 1] = __float2bfloat16(g_up);
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                route_grad += __shfl_down_sync(0xffffffff, route_grad, offset);
            if (lane_id == 0) {
                state->combine_input_topk_weights[recv_token_m * num_topk + topk_slot_m] = route_grad;
            }
        }
        compute_group_sync(state, group_id, group_size);
        if (group_sm_idx == 0) {
            trace_backward_values("GRAD-GU", dgu_dst, batch_size * twoI,
                                  state->rank, task_idx, expert_id, batch_size);
        }

        // grad_xperm = grad_gu @ W_gateup -> down_buf [M,hidden].
        if (use_umma_bwd_for_group) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];
            const int dgu_batch_id = start_slot / COMPUTE_BATCH_SIZE;
            const CUtensorMap* dgu_a_tma =
                &bs->wgrad_dgu_a_tma[(int64_t)expert_id * state->max_batches_per_expert + dgu_batch_id];
            uint32_t grad_x_accum_iter = 0;
            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            umma::umma_dgrad_mn_persistent(
                dgu_a_tma,
                &bs->compute_bwd_tma->wgateup[expert_id],
                &in_atom.down_cd,
                batch_size, hidden, twoI,
                cluster_in_group, num_clusters,
                cluster_smem, grad_x_accum_iter);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_down_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        } else {
            device_gemm_bf16_mn(dgu_dst, Wgu_e, down_buf, batch_size, twoI, hidden,
                                group_warp_id, group_num_warps, local_warp_id, smem_wmma_buf);
#if MK_PERF_TRACE_ARGS
            if (perf_leader) perf_down_body_ns = globaltimer_ns();
#endif
            compute_group_sync(state, group_id, group_size);
        }
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_downgemm_ns = globaltimer_ns();
#endif
        if (group_sm_idx == 0) {
            trace_backward_values("GRAD-XPERM", down_buf, batch_size * hidden,
                                  state->rank, task_idx, expert_id, batch_size);
        }

        // ---- per-slot output scatter (identical to forward down output) ----
        const int4* down_i4 = reinterpret_cast<const int4*>(down_buf);
        int4* slot_out_i4 = reinterpret_cast<int4*>(state->compute_output_slot);
        int4* token_out_i4 = reinterpret_cast<int4*>(state->combine_input);
        const int slot_base = state->expert_slot_base[expert_id] + start_slot;
        for (int idx = group_thread_id; idx < batch_size * hidden_int4; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            int slot = slot_base + row;
            if (s_is_single[row])
                token_out_i4[(int64_t)s_recv_token_idx[row] * hidden_int4 + v] = down_i4[idx];
            else
                slot_out_i4[(int64_t)slot * hidden_int4 + v] = down_i4[idx];
        }
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_out_body_ns = globaltimer_ns();
#endif
        __threadfence();
        compute_group_sync(state, group_id, group_size);
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_ph_output_ns = globaltimer_ns();
#endif
        if (group_sm_idx == 0 && batch_size > 0) {
            const int recv_token = s_recv_token_idx[0];
            const __nv_bfloat16* scatter_values = s_is_single[0]
                ? state->combine_input + (int64_t)recv_token * hidden
                : state->compute_output_slot + (int64_t)slot_base * hidden;
            trace_backward_values(
                s_is_single[0] ? "SCATTER-SINGLE" : "SCATTER-SLOT",
                scatter_values, hidden, state->rank, task_idx, expert_id, batch_size);
        }

        // ---- signal per-token completion (identical to forward) ----
#if MK_PERF_TRACE_ARGS
        if (perf_leader) {
            perf_sig_donecount_ns = globaltimer_ns();
            perf_sig_finalize_ns = perf_sig_donecount_ns;
            perf_sig_fence_ns = perf_sig_donecount_ns;
        }
        if (perf_leader) {
            int mr = 0;
            for (int row = 0; row < batch_size; ++row)
                if (!s_is_single[row]) ++mr;
            s_perf_multi_expert_rows = mr;
            s_perf_task_has_multi = (mr != 0);
        }
#endif
        for (int row = group_thread_id; row < batch_size; row += group_num_threads) {
            const int recv_token = s_recv_token_idx[row];
            int done = atomicAdd(&state->token_done_count[recv_token], 1) + 1;
            if (s_is_single[row] && done >= 1) {
                __threadfence();
                atomicExch(&state->combine_token_ready[recv_token], 1);
            }
        }
        compute_group_sync(state, group_id, group_size);
#if MK_PERF_TRACE_ENABLED
        if (perf_leader) compute_task_end_ns = globaltimer_ns();
#endif
#if MK_PERF_TRACE_ARGS
        if (perf_leader) perf_sig_publish_ns = compute_task_end_ns;
#endif

#if MK_PERF_TRACE_ENABLED
        if (perf_leader) {
#if MK_PERF_TRACE_ARGS
            last_task_end_ns = compute_task_end_ns;
#endif
            int slot = task_idx;
            if (slot >= 0 && slot < state->max_compute_tasks) {
                int64_t* rec = state->perf_compute_task + (int64_t)slot * MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS;
                rec[0] = compute_task_start_ns;
                rec[1] = compute_task_end_ns;
                rec[2] = sm_id;
                rec[3] = group_id;
                rec[5] = batch_size;
                rec[25] = task.is_flush;
                rec[26] = 1;
#if MK_PERF_TRACE_ARGS
                rec[4] = expert_id;
                rec[6] = hidden;
                rec[7] = intermediate;
                rec[8]  = perf_ph_meta_ns;
                rec[9]  = perf_ph_input_ns;
                rec[10] = perf_ph_upgemm_ns;
                rec[11] = perf_ph_downgemm_ns;
                rec[12] = perf_ph_output_ns;
                rec[13] = compute_task_end_ns;
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
                rec[24] = static_cast<int64_t>(state->expert_slot_base[expert_id]) + start_slot;
                state->perf_compute_multi_expert_rows[slot] = s_perf_multi_expert_rows;
                state->perf_compute_task_has_multi[slot] = s_perf_task_has_multi;
#endif
            }
        }
#endif
    }
    if constexpr (kStopAtDispatchDone) {
        asm volatile("barrier.sync 15, %0;" :: "r"(static_cast<int>(blockDim.x)) : "memory");
    }
}

template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    uint8_t* smem_buffer
) {
    compute_backward_worker_core<kComputeDType, false>(
        bs, sm_id, compute_sm_idx, num_compute_sms, 0, smem_buffer);
}

template <ComputeDType kComputeDType>
__device__ __forceinline__ void combine_precompute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int combine_sm_idx,
    int num_combine_sms,
    uint8_t* smem_buffer
) {
    MegaKernelState* state = bs->bwd_device_state;
    const int post_group_count =
        (state->num_compute_sms + state->num_dispatch_sms + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    compute_backward_worker_core<kComputeDType, true>(
        bs, sm_id, combine_sm_idx, num_combine_sms, post_group_count, smem_buffer);
}

// Backward megakernel: same role layout as moe_megakernel_v7 (dispatch / combine /
// scheduler / gather all reused verbatim on the patched bwd state); only the compute
// role is swapped for compute_backward_worker.
template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
__global__ void __launch_bounds__(MegaKernelRdmaConfig<kNumRDMARanks>::kMegaKernelNumThreads, 1) moe_megakernel_v7_backward(
    MegaKernelBackwardState* bs
) {
    MegaKernelState* state = bs->bwd_device_state;
    const int sm_id = blockIdx.x;
    const int num_dispatch_sms = state->num_dispatch_sms;
    const int num_combine_sms = state->num_combine_sms;
    const int num_compute_sms = state->num_compute_sms;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    SmRole role;
    int role_idx;
    const int compute_begin = num_dispatch_sms + num_combine_sms + COMPUTE_SCHEDULER_SMS;
    const int gather_begin = compute_begin + num_compute_sms;
    const int total_compute_sms_after_dispatch = num_compute_sms + num_dispatch_sms;
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
            dispatch_worker_v2<kNumRDMARanks, kStage, kComputeDType, true>(sm_id, role_idx, state, bs);
            break;
        case SmRole::kCombine:
            combine_precompute_backward_worker<kComputeDType>(bs, sm_id, role_idx, num_combine_sms, smem_buffer);
            combine_worker_v2<kNumRDMARanks, kStage>(role_idx, state);
            break;
        case SmRole::kScheduler:
            compute_scheduler_worker(state, role_idx, COMPUTE_SCHEDULER_SMS);
            break;
        case SmRole::kCompute:
            MK_BACKWARD_COMPUTE_WORKER(
                kComputeDType, bs, sm_id, role_idx,
                total_compute_sms_after_dispatch, smem_buffer);
            break;
        case SmRole::kGather:
            gather_worker(state, role_idx);
            break;
        default:
            break;
    }
}

// Build the backward state with an independent v7 allocation. Only the saved forward
// activation is shared; routing inputs and Buffer-owned transport pointers are reused as
// allocator inputs, while all derived routing, workspace, FIFO, and counter storage is new.
MegaKernelBackwardState* allocate_megakernel_backward_state(
    MegaKernelState* fwd_device_state,
    const void* grad_output,
    void* grad_input,
    void* grad_w_gateup,
    void* grad_w_down,
    void* grad_topk_weights,
    void* wgrad_x_slot,
    void* wgrad_act_slot,
    void* wgrad_dz_slot,
    void* wgrad_dgu_slot,
    const int* host_expert_count,
    int total_sms,
    MegaKernelBackwardHostContext** host_context,
    cudaStream_t stream,
    const MegaKernelState* cached_fwd_host_state
) {
    MegaKernelState fs;
    if (cached_fwd_host_state != nullptr) {
        // Fast path: use the host-cached forward state snapshot directly,
        // avoiding a synchronous D2H cudaMemcpy that stalls the pipeline.
        fs = *cached_fwd_host_state;
    } else {
        // Fallback: synchronous D2H (legacy path, should not be hit in training).
        CUDA_CHECK(cudaMemcpy(&fs, fwd_device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    }

    const int hidden = fs.hidden_dim;
    const int intermediate = fs.intermediate_dim;
    const int twoI = 2 * intermediate;
    const int num_local_experts = fs.num_local_experts;
    EP_HOST_ASSERT(fs.num_compute_groups > 0);
    EP_HOST_ASSERT(total_sms >= fs.num_compute_groups * COMPUTE_GROUP_SIZE);

    EP_HOST_ASSERT(host_expert_count != nullptr);
    EP_HOST_ASSERT(host_context != nullptr);
    std::vector<int> h_bwd_expert_count(host_expert_count, host_expert_count + num_local_experts);
    std::vector<int> h_bwd_expert_slot_base(num_local_experts);
    size_t total_bwd_slots = 0;
    for (int e = 0; e < num_local_experts; ++e) {
        h_bwd_expert_slot_base[e] = static_cast<int>(total_bwd_slots);
        total_bwd_slots += static_cast<size_t>(h_bwd_expert_count[e]);
    }
    if (total_bwd_slots == 0) total_bwd_slots = 1;

    auto* host_ctx = new MegaKernelBackwardHostContext{};
    umma::ComputeBackwardTmaAtoms* d_compute_bwd_tma = nullptr;
    if (num_local_experts <= umma::kMaxLocalExperts) {
        umma::build_compute_backward_tma_atoms(
            host_ctx->compute_bwd_tma_atoms, fs.W_down, fs.W_gateup,
            num_local_experts, hidden, intermediate);
        CUDA_CHECK(mk_caching_alloc(
            reinterpret_cast<void**>(&d_compute_bwd_tma), sizeof(umma::ComputeBackwardTmaAtoms)));
        CUDA_CHECK(cudaMemcpyAsync(d_compute_bwd_tma, &host_ctx->compute_bwd_tma_atoms,
                                   sizeof(umma::ComputeBackwardTmaAtoms),
                                   cudaMemcpyHostToDevice, stream));
    }

    MegaKernelState* bwd_device_state = ::deep_ep::megakernel_debug::allocate_megakernel_state_v7(
        reinterpret_cast<const int4*>(grad_output), nullptr,
        fs.topk_idx, fs.topk_weights, fs.is_token_in_rank,
        fs.rdma_channel_prefix_matrix, fs.recv_rdma_rank_prefix_sum,
        fs.gbl_channel_prefix_matrix, fs.recv_gbl_rank_prefix_sum,
        fs.rdma_buffer_ptr, fs.buffer_ptrs, fs.allocator_combine_buffer_ptrs,
        fs.num_tokens, fs.hidden_dim, fs.hidden_int4, fs.intermediate_dim,
        0, fs.num_topk, fs.num_experts, fs.num_local_experts,
        fs.num_ranks, fs.rank, fs.scale_token_stride, fs.scale_hidden_stride,
        fs.num_max_rdma_chunked_send_tokens, fs.num_max_rdma_chunked_recv_tokens,
        fs.num_max_nvl_chunked_send_tokens, fs.num_max_nvl_chunked_recv_tokens,
        fs.num_max_combine_rdma_chunked_send_tokens,
        fs.num_max_combine_rdma_chunked_recv_tokens,
        fs.num_max_combine_nvl_chunked_send_tokens,
        fs.num_max_combine_nvl_chunked_recv_tokens,
        fs.W_gateup, fs.W_down, fs.compute_dtype,
        fs.W_gateup_fp8, fs.W_down_fp8,
        fs.W_gateup_fp8_sf, fs.W_down_fp8_sf,
        fs.allocator_num_dispatch_sms, fs.allocator_num_forwarder_sms,
        fs.allocator_num_compute_sms, fs.allocator_num_combine_sms,
        fs.allocator_num_logical_channels,
        fs.allocator_max_tokens_per_expert, fs.allocator_max_total_recv_tokens,
        fs.allocator_num_rdma_bytes, fs.allocator_num_nvl_bytes,
        h_bwd_expert_count.data(),
        nullptr,
        const_cast<__nv_bfloat16*>(fs.bwd_fc1_input),
        const_cast<__nv_bfloat16*>(fs.bwd_preact),
        fs.fwd_slot_map,
        reinterpret_cast<int4*>(grad_input),
        reinterpret_cast<float*>(grad_topk_weights),
        &host_ctx->bwd_state,
        fs.rdma_reuse_dispatch_quiet_done,
        fs.rdma_reuse_combine_clear_done,
        1 /* rdma_reuse_prelude_enable: backward reuses the dispatch RDMA region too.
             Backward shares dispatch_worker_v2 / combine_worker_v2, so the same
             dispatch_channel_barrier==2 gate + quiet + phase-B protocol applies. */);

    MegaKernelBackwardState& hs = host_ctx->backward_state;
    hs.fwd = fwd_device_state;
    hs.bwd_device_state = bwd_device_state;
    hs.bwd_fc1_input = fs.bwd_fc1_input;
    hs.bwd_preact = fs.bwd_preact;
    hs.compute_bwd_tma = d_compute_bwd_tma;
    hs.grad_output = reinterpret_cast<const __nv_bfloat16*>(grad_output);
    hs.grad_input = reinterpret_cast<__nv_bfloat16*>(grad_input);
    hs.grad_w_gateup = reinterpret_cast<__nv_bfloat16*>(grad_w_gateup);
    hs.grad_w_down = reinterpret_cast<__nv_bfloat16*>(grad_w_down);
    hs.grad_topk_weights = reinterpret_cast<float*>(grad_topk_weights);
    // Compact Family B (wgrad) scratch: Σ count, same per-expert slot layout as the backward
    // state. wgrad_dgu_slot gets one extra COMPUTE_BATCH_SIZE of padding so the last batch's
    // CBS-row TMA descriptor tile stays within the allocation.
    const size_t num_dgu_batch_tmas =
        (size_t)fs.num_local_experts * fs.max_batches_per_expert;
    EP_HOST_ASSERT(wgrad_x_slot != nullptr && wgrad_act_slot != nullptr);
    EP_HOST_ASSERT(wgrad_dz_slot != nullptr && wgrad_dgu_slot != nullptr);
    hs.wgrad_x_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_x_slot);
    hs.wgrad_act_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_act_slot);
    hs.wgrad_dz_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_dz_slot);
    hs.wgrad_dgu_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_dgu_slot);
    host_ctx->wgrad_dgu_a_tma.resize(num_dgu_batch_tmas);
    for (int expert = 0; expert < fs.num_local_experts; ++expert) {
        const int ebase = h_bwd_expert_slot_base[expert];
        const int ecnt = h_bwd_expert_count[expert];
        for (int batch = 0; batch < fs.max_batches_per_expert; ++batch) {
            // Descriptors for batches past this expert's real slots are never consumed by
            // the compute worker; keep them in-bounds by clamping to the expert base.
            const int row = (batch * COMPUTE_BATCH_SIZE < ecnt)
                ? (ebase + batch * COMPUTE_BATCH_SIZE) : ebase;
            const __nv_bfloat16* dgu_batch = hs.wgrad_dgu_slot + (size_t)row * twoI;
            host_ctx->wgrad_dgu_a_tma[(size_t)expert * fs.max_batches_per_expert + batch] =
                umma::dg_make_a_desc(dgu_batch, COMPUTE_BATCH_SIZE, twoI);
        }
    }
    CUtensorMap* d_wgrad_dgu_a_tma = nullptr;
    CUDA_CHECK(mk_caching_alloc(
        reinterpret_cast<void**>(&d_wgrad_dgu_a_tma),
        num_dgu_batch_tmas * sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpyAsync(d_wgrad_dgu_a_tma, host_ctx->wgrad_dgu_a_tma.data(),
                               num_dgu_batch_tmas * sizeof(CUtensorMap),
                               cudaMemcpyHostToDevice, stream));
    hs.wgrad_dgu_a_tma = d_wgrad_dgu_a_tma;

    MegaKernelBackwardState* device_bs;
    CUDA_CHECK(mk_caching_alloc(
        reinterpret_cast<void**>(&device_bs), sizeof(MegaKernelBackwardState)));
    CUDA_CHECK(cudaMemcpyAsync(device_bs, &hs, sizeof(MegaKernelBackwardState),
                               cudaMemcpyHostToDevice, stream));

    // Append the backward-specific fills (send_rdma_head=0xff, send_nvl_head=0xff,
    // grad_input=0x00) to the same stream so they execute in the fused_fill_kernel
    // already launched by allocate_megakernel_state_v7 above — or, since that kernel
    // has already been enqueued, launch a second merged fill covering all three entries
    // in one kernel. This merges what was previously a separate fused_fill in
    // launch_megakernel_debug_backward into the allocate path, eliminating one kernel
    // launch + one H2D from the critical path between fused_fill and cached_notify.
    {
        const MegaKernelState& bwd_hs = host_ctx->bwd_state;
        const int kRDMA = bwd_hs.num_ranks / NUM_MAX_NVL_PEERS;
        const int NLC = bwd_hs.num_logical_channels;
        const int TK = bwd_hs.num_topk;
        const int combine_rdma_head_stride = bwd_hs.num_tokens * kRDMA;
        const int combine_nvl_head_stride = (bwd_hs.num_tokens * TK) * NUM_MAX_NVL_PEERS;
        const size_t rdma_head_bytes = (size_t)NLC * combine_rdma_head_stride * sizeof(int);
        const size_t nvl_head_bytes = (size_t)NLC * combine_nvl_head_stride * sizeof(int);
        const size_t grad_input_bytes = (size_t)bwd_hs.num_tokens * bwd_hs.hidden_dim * sizeof(__nv_bfloat16);

        constexpr size_t CHUNK_WORDS = 64 * 1024;
        struct BwdFillEntry { void* ptr; size_t bytes; uint32_t word; };
        BwdFillEntry entries[] = {
            {bwd_hs.send_rdma_head, rdma_head_bytes, 0xffffffffu},
            {bwd_hs.send_nvl_head,  nvl_head_bytes,  0xffffffffu},
            {hs.grad_input,         grad_input_bytes, 0x00000000u},
        };
        std::vector<FusedFillDesc> bwd_descs;
        for (auto& e : entries) {
            uint8_t* base = reinterpret_cast<uint8_t*>(e.ptr);
            size_t offset = 0;
            while (offset < e.bytes) {
                size_t chunk = std::min(e.bytes - offset, CHUNK_WORDS * sizeof(uint32_t));
                chunk = (chunk / sizeof(uint32_t)) * sizeof(uint32_t);
                if (chunk == 0) chunk = e.bytes - offset;
                bwd_descs.push_back(FusedFillDesc{base + offset, chunk, e.word});
                offset += chunk;
            }
        }
        const int ndescs = static_cast<int>(bwd_descs.size());
        if (ndescs > 0) {
            void* bwd_fill_buf = nullptr;
            CUDA_CHECK(mk_caching_alloc(&bwd_fill_buf, (size_t)ndescs * sizeof(FusedFillDesc)));
            CUDA_CHECK(cudaMemcpyAsync(bwd_fill_buf, bwd_descs.data(),
                                       (size_t)ndescs * sizeof(FusedFillDesc),
                                       cudaMemcpyHostToDevice, stream));
            fused_fill_kernel<<<ndescs, 512, 0, stream>>>(
                static_cast<const FusedFillDesc*>(bwd_fill_buf), ndescs,
                nullptr, nullptr, nullptr, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(mk_caching_free(bwd_fill_buf));
        }
    }

    // Single sync covers all prior async H2Ds (compute_bwd_tma, allocate_state_v7's
    // state+fill_descs, wgrad_dgu_a_tma, device_bs, backward fill descs) plus both
    // fused_fill_kernel launches. This mirrors the forward pattern: batch all async
    // ops, sync once at the end.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EP_HOST_ASSERT(host_context != nullptr);
    *host_context = host_ctx;
    return device_bs;
}

void free_megakernel_backward_state(
    MegaKernelBackwardState* device_bs,
    const MegaKernelBackwardHostContext* host_context
) {
    if (device_bs == nullptr)
        return;

    EP_HOST_ASSERT(device_bs != nullptr);
    MegaKernelBackwardState hs;
    if (host_context != nullptr) {
        hs = host_context->backward_state;
    } else {
        CUDA_CHECK(cudaMemcpy(&hs, device_bs, sizeof(MegaKernelBackwardState), cudaMemcpyDeviceToHost));
    }
    // wgrad_* buffers are borrowed from caller-owned Torch tensors.
    CUDA_CHECK(mk_caching_free(hs.compute_bwd_tma));
    CUDA_CHECK(mk_caching_free(hs.wgrad_dgu_a_tma));
    free_megakernel_state_v7(
        hs.bwd_device_state,
        host_context != nullptr ? &host_context->bwd_state : nullptr);
    CUDA_CHECK(mk_caching_free(device_bs));
}

void free_megakernel_backward_host_context(MegaKernelBackwardHostContext* host_context) {
    delete host_context;
}

// Establish the post-allocation state for every launch. Routing inputs, weights and the
// saved forward activation are immutable across replay and are intentionally preserved.
static void prepare_megakernel_communication_replay_host(
    const MegaKernelState& hs,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream
) {
    const int combine_hidden_int4 = hs.combine_hidden / (sizeof(int4) / sizeof(__nv_bfloat16));

    // Use the megakernel notify entry so ordinary DeepEP keeps the stock cached_notify path.
    // Besides the cross-rank barriers, this owns the same RDMA/NVL metadata layout and cleanup sizes.
    internode::mk_cached_notfy(
        hs.hidden_int4, hs.num_scales, hs.num_topk + 1, hs.num_topk,
        hs.num_ranks, hs.num_logical_channels, 0, nullptr, nullptr, nullptr, nullptr,
        hs.rdma_buffer_ptr, hs.num_max_rdma_chunked_recv_tokens, hs.buffer_ptrs,
        hs.num_max_nvl_chunked_recv_tokens, dispatch_barrier_signal_ptrs, hs.rank,
        stream, hs.num_rdma_bytes, hs.num_nvl_bytes, true, false);
    internode::mk_cached_notfy(
        combine_hidden_int4, 0, 0, hs.num_topk,
        hs.num_ranks, hs.num_logical_channels, hs.num_tokens,
        hs.send_rdma_head, hs.combine_rdma_channel_prefix_matrix,
        hs.combine_rdma_rank_prefix_sum, hs.send_nvl_head,
        hs.combine_rdma_buffer_ptr, hs.num_max_combine_rdma_chunked_recv_tokens,
        hs.combine_buffer_ptrs, hs.num_max_combine_nvl_chunked_recv_tokens,
        combine_barrier_signal_ptrs, hs.rank, stream, hs.num_rdma_bytes,
        hs.num_nvl_bytes, false, false);
}

void prepare_megakernel_communication_replay(
    MegaKernelState* device_state,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream
) {
    MegaKernelState hs;
    CUDA_CHECK(cudaMemcpy(&hs, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    prepare_megakernel_communication_replay_host(
        hs, dispatch_barrier_signal_ptrs, combine_barrier_signal_ptrs, stream);
}

#if MK_PERF_TRACE_ENABLED
static void reset_megakernel_perf_trace_state(const MegaKernelState& hs, cudaStream_t stream) {
    auto z = [&](void* p, size_t bytes) {
        if (bytes > 0) CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, stream));
    };
    auto f = [&](void* p, size_t bytes) {
        if (bytes > 0) CUDA_CHECK(cudaMemsetAsync(p, 0xff, bytes, stream));
    };

    const int NLC = hs.num_logical_channels;
    const int NPUB = hs.num_pub_warps_total;
    const size_t lch_ts_bytes = (size_t)NLC * 2 * MegaKernelState::MK_PERF_NUM_LCH_PHASES * sizeof(int64_t);
    const size_t compute_task_bytes = (size_t)hs.max_compute_tasks * MegaKernelState::MK_PERF_NUM_COMPUTE_FIELDS * sizeof(int64_t);
    const size_t gather_task_bytes = (size_t)hs.max_compute_tasks * MegaKernelState::MK_PERF_NUM_GATHER_FIELDS * sizeof(int64_t);
    const size_t async_pub_bytes = (size_t)NPUB * sizeof(int64_t);

    z(hs.perf_dispatch_lch_ts, lch_ts_bytes);
    z(hs.perf_combine_lch_ts, lch_ts_bytes);
    z(hs.perf_compute_task, compute_task_bytes);
    z(hs.perf_compute_task_count, sizeof(int));
    z(hs.perf_gather_task, gather_task_bytes);
    z(hs.perf_gather_task_count, sizeof(int));
    z(hs.perf_async_pub_start_ts, async_pub_bytes);
    z(hs.perf_async_pub_end_ts, async_pub_bytes);
    z(hs.perf_async_publish_all_done_ts, sizeof(int64_t));
    z(hs.perf_sched_ts, 2 * sizeof(int64_t));

#if MK_PERF_TRACE_ARGS
    const size_t acc_bytes = (size_t)NLC * 2 * sizeof(int64_t);
    const size_t disp_role_bytes = (size_t)NLC * 2 * MK_DISPATCH_ROLE_COUNT * NUM_MAX_NVL_PEERS * sizeof(int64_t);
    const size_t disp_recv_bytes = (size_t)NLC * 2 * NUM_MAX_NVL_PEERS * sizeof(int64_t);
    const size_t disp_prefix_prod_bytes = disp_recv_bytes * (hs.num_ranks / NUM_MAX_NVL_PEERS);
    const size_t diag_i64_bytes = (size_t)hs.max_compute_tasks * sizeof(int64_t);
    const size_t diag_i32_bytes = (size_t)hs.max_compute_tasks * sizeof(int);
    const size_t expert_batch_bytes = (size_t)hs.num_local_experts * hs.max_batches_per_expert;

    z(hs.expert_batch_enqueue_ts, expert_batch_bytes * sizeof(int64_t));
    z(hs.token_priority_dep_count, (size_t)hs.max_total_recv_tokens * sizeof(int));

    int64_t* const acc_zero[] = {
        hs.perf_disp_wait_nvl_ns, hs.perf_disp_publish_ns, hs.perf_disp_wait_recvcount_ns,
        hs.perf_comb_tma_wait_ns, hs.perf_comb_wait_ready_ns, hs.perf_comb_wait_ready_single_ns,
        hs.perf_comb_wait_ready_multi_ns, hs.perf_comb_wait_ready_flush_ns,
        hs.perf_comb_wait_ready_full_ns, hs.perf_comb_wait_ready_flush_count,
        hs.perf_comb_wait_ready_full_count, hs.perf_comb_gather_reduce_ns,
        hs.perf_comb_gather_single_ns, hs.perf_comb_gather_multi_ns,
        hs.perf_comb_pack_meta_ns, hs.perf_comb_pack_meta_work_ns,
        hs.perf_comb_pack_meta_sync_ns, hs.perf_comb_tma_store_ns,
        hs.perf_comb_tma_wait_max_ns, hs.perf_comb_wait_ready_max_ns,
        hs.perf_comb_wait_ready_single_max_ns, hs.perf_comb_wait_ready_multi_max_ns,
        hs.perf_comb_wait_ready_flush_max_ns, hs.perf_comb_wait_ready_full_max_ns,
        hs.perf_comb_wait_top_ns, hs.perf_comb_wait_top_nhits,
        hs.perf_comb_wait_top_priority_deps, hs.perf_comb_gather_reduce_max_ns,
        hs.perf_comb_gather_single_max_ns, hs.perf_comb_gather_multi_max_ns,
        hs.perf_comb_pack_meta_max_ns, hs.perf_comb_pack_meta_work_max_ns,
        hs.perf_comb_pack_meta_sync_max_ns, hs.perf_comb_tma_store_max_ns,
        hs.perf_comb_nhit_sum, hs.perf_comb_token_count,
        hs.perf_comb_single_token_count, hs.perf_comb_multi_token_count,
        hs.perf_disp_pub_scan_ns, hs.perf_disp_pub_atomic_ns,
        hs.perf_disp_pub_fence_ns, hs.perf_disp_pub_store_ns,
        hs.perf_disp_cta_barrier_ns, hs.perf_disp_channel_barrier_ns,
        hs.perf_disp_round_barrier_ns, hs.perf_disp_tokens,
        hs.perf_disp_local_hit_tokens, hs.perf_disp_local_hits,
        hs.perf_disp_cta_release_ts,
    };
    for (int64_t* p : acc_zero) z(p, acc_bytes);

    int64_t* const acc_ff[] = {
        hs.perf_comb_wait_top_token, hs.perf_comb_wait_top_slot,
        hs.perf_comb_wait_top_expert, hs.perf_comb_wait_top_from_flush,
    };
    for (int64_t* p : acc_ff) f(p, acc_bytes);

    z(hs.perf_disp_role_arrive_ts, disp_role_bytes);
    z(hs.perf_disp_role_work_ns, disp_role_bytes);

    int64_t* const disp_recv_zero[] = {
        hs.perf_disp_allrecv_wait_nvl_ns, hs.perf_disp_allrecv_prefix_wait_ns,
        hs.perf_disp_allrecv_prefix_wait_start_ts, hs.perf_disp_allrecv_prefix_observe_ts,
        hs.perf_disp_allrecv_prefix_done_ts, hs.perf_disp_allrecv_prefix_raw_start,
        hs.perf_disp_allrecv_prefix_raw_end, hs.perf_disp_allrecv_token_loop_ns,
        hs.perf_disp_allrecv_retire_ns, hs.perf_disp_allrecv_publish_ns,
        hs.perf_disp_allrecv_tokens, hs.perf_disp_allrecv_local_hits,
    };
    for (int64_t* p : disp_recv_zero) z(p, disp_recv_bytes);
    f(hs.perf_disp_allrecv_prefix_slowest_rdma, disp_recv_bytes);
    f(hs.perf_disp_allrecv_prefix_src_nvl, disp_recv_bytes);

    int64_t* const disp_prefix_zero[] = {
        hs.perf_disp_prefix_store_begin_ts, hs.perf_disp_prefix_publish_ts,
        hs.perf_disp_prefix_fence_done_ts, hs.perf_disp_prefix_store_to_fence_ns,
        hs.perf_disp_prefix_meta_wait_ns, hs.perf_disp_prefix_tokens,
    };
    for (int64_t* p : disp_prefix_zero) z(p, disp_prefix_prod_bytes);
    f(hs.perf_disp_prefix_producer_rank, disp_prefix_prod_bytes);
    f(hs.perf_disp_prefix_producer_nvl, disp_prefix_prod_bytes);
    f(hs.perf_disp_prefix_producer_dst_nvl, disp_prefix_prod_bytes);
    f(hs.perf_disp_prefix_producer_src_rdma, disp_prefix_prod_bytes);

    int64_t* const async_pub_zero[] = {
        hs.perf_async_pub_wait_ring_ns, hs.perf_async_pub_poll_ns,
        hs.perf_async_pub_gap_ns, hs.perf_async_pub_start_gap_ns,
        hs.perf_async_pub_empty_gap_ns, hs.perf_async_pub_done_recheck_ns,
        hs.perf_async_pub_ring_load_ns, hs.perf_async_pub_head_release_ns,
        hs.perf_async_pub_syncwarp_ns, hs.perf_async_pub_batch_wall_ns,
        hs.perf_async_pub_batch_accounted_ns, hs.perf_async_pub_batch_unattributed_ns,
        hs.perf_async_pub_token_gap_ns, hs.perf_async_pub_helper_unattributed_ns,
        hs.perf_async_pub_finish_ns, hs.perf_async_pub_work_ns,
        hs.perf_async_pub_scan_ns, hs.perf_async_pub_atomic_ns,
        hs.perf_async_pub_fence_ns, hs.perf_async_pub_store_ns,
        hs.perf_async_pub_drain_ns, hs.perf_async_pub_tokens,
        hs.perf_async_pub_batch_count, hs.perf_async_pub_head_release_count,
        hs.perf_async_pub_max_batch, hs.perf_async_pub_local_hit_tokens,
        hs.perf_async_pub_local_hits,
    };
    for (int64_t* p : async_pub_zero) z(p, async_pub_bytes);

    int64_t* const sched_zero[] = {
        hs.perf_sched_scan_ns, hs.perf_sched_enqueue_ns, hs.perf_sched_idle_ns,
        hs.perf_sched_priority_ns, hs.perf_sched_normal_ns, hs.perf_sched_tail_flush_ns,
        hs.perf_sched_publish_total_ns, hs.perf_sched_publish_wait_ns,
        hs.perf_sched_publish_wait_max_ns, hs.perf_sched_priority_scan_tokens,
        hs.perf_sched_priority_ready_tokens, hs.perf_sched_priority_full_batch_hits,
        hs.perf_sched_priority_batch_already_enqueued, hs.perf_sched_priority_already_normal,
        hs.perf_sched_priority_already_priority, hs.perf_sched_priority_already_tail,
        hs.perf_sched_normal_after_priority_ns, hs.perf_sched_normal_after_priority_count,
        hs.perf_sched_priority_not_full, hs.perf_sched_normal_full_batch_enqueues,
        hs.perf_sched_flush_tail_enqueues, hs.perf_sched_queue_empty_count,
        hs.perf_sched_queue_empty_after_dispatch_count, hs.perf_sched_max_ready_tail_gap,
        hs.perf_sched_stall_recv_count, hs.perf_sched_stall_alloc_count,
        hs.perf_sched_stall_enqueue_cursor, hs.perf_sched_stall_first_unready_ready,
        hs.perf_sched_stall_dispatch_done,
        hs.perf_sched_first_done_seen_ts, hs.perf_sched_first_recv_count_advance_ts,
        hs.perf_sched_first_normal_enqueue_attempt_ts, hs.perf_sched_first_normal_enqueue_success_ts,
        hs.perf_sched_first_task_publish_ts, hs.perf_sched_first_task_source,
        hs.perf_sched_first_task_num_tokens,
    };
    for (int64_t* p : sched_zero) z(p, sizeof(int64_t));
    f(hs.perf_sched_publish_wait_max_tail, sizeof(int64_t));
    f(hs.perf_sched_publish_wait_max_visible_tail, sizeof(int64_t));
    f(hs.perf_sched_stall_expert, sizeof(int64_t));
    f(hs.perf_sched_stall_first_unready_slot, sizeof(int64_t));
    f(hs.perf_sched_first_recv_count_advance_expert, sizeof(int64_t));
    f(hs.perf_sched_first_recv_count_advance_old, sizeof(int64_t));
    f(hs.perf_sched_first_recv_count_advance_new, sizeof(int64_t));
    f(hs.perf_sched_first_task_expert, sizeof(int64_t));
    f(hs.perf_sched_first_task_batch, sizeof(int64_t));
    f(hs.perf_sched_first_task_start_slot, sizeof(int64_t));

    int64_t* const diag_zero[] = {
        hs.perf_task_publish_ts, hs.perf_task_pop_start_ts, hs.perf_task_pop_done_ts,
        hs.perf_task_bcast_done_ts, hs.perf_task_start_ts, hs.perf_task_prev_end_ts,
        hs.perf_task_prev_gap_ns,
    };
    for (int64_t* p : diag_zero) z(p, diag_i64_bytes);
    z(hs.perf_compute_multi_expert_rows, diag_i32_bytes);
    z(hs.perf_compute_task_has_multi, diag_i32_bytes);
    z(hs.perf_task_source, diag_i32_bytes);
    z(hs.perf_task_pop_attempts, diag_i32_bytes);
    z(hs.perf_task_cas_failures, diag_i32_bytes);
    f(hs.perf_task_group_id, diag_i32_bytes);
#endif
}
#endif

static void initialize_megakernel_launch_state(const MegaKernelState& hs, cudaStream_t stream) {
    const int E = hs.num_local_experts;
    const size_t MTR = (size_t)hs.max_total_recv_tokens;
    const int TK = hs.num_topk;
    const int NG = hs.num_compute_groups;
    const int NR = hs.num_ranks;
    const int kRDMA = hs.num_ranks / NUM_MAX_NVL_PEERS;
    const int NLC = hs.num_logical_channels;
    const int NPC = hs.num_dispatch_channels;
    const int NPUB = hs.num_pub_warps_total;
    const int MBE = (hs.max_tokens_per_expert + COMPUTE_BATCH_SIZE - 1) / COMPUTE_BATCH_SIZE;
    // Per-expert-slot buffers are sized by the compact Σ expert_count (see allocator);
    // reset must cover exactly that, not the legacy num_local_experts*max_tpe.
    const size_t total_slots = (size_t)hs.total_expert_slots;
    const int round_slots = (NLC + NPC - 1) / NPC;
    const int combine_rdma_head_stride = hs.num_tokens * kRDMA;
    const int combine_nvl_head_stride = (hs.num_tokens * TK) * NUM_MAX_NVL_PEERS;

    auto z = [&](void* p, size_t bytes) { CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, stream)); };
    auto f = [&](void* p, size_t bytes) { CUDA_CHECK(cudaMemsetAsync(p, 0xff, bytes, stream)); };

    z(hs.expert_recv_count, (size_t)E * sizeof(int));
    z(hs.expert_slot_ready, total_slots * sizeof(int));
    z(hs.dispatch_done, sizeof(int));
    z(hs.dispatch_done_count, sizeof(int));
    z(hs.timeout_log_counters, kTimeoutLogCount * sizeof(int));
    z(hs.expert_token_offsets, (size_t)E * sizeof(int));
    f(hs.recv_token_source_info, total_slots * 2 * sizeof(int));
    z(hs.compute_group_barrier, (size_t)NG * sizeof(int));
    z(hs.compute_group_phase, (size_t)NG * sizeof(int));
    z(hs.compute_task_head, sizeof(int));
    z(hs.compute_task_tail, sizeof(int));
    z(hs.compute_task_reserve_tail, sizeof(int));
    z(hs.compute_enqueue_done, sizeof(int));
    z(hs.scheduler_done_count, sizeof(int));
    z(hs.priority_scheduler_done, sizeof(int));
    z(hs.expert_enqueue_cursor, (size_t)E * sizeof(int));
    f(hs.compute_group_task_idx, (size_t)NG * sizeof(int));
    z(hs.token_done_count, MTR * sizeof(int));
    z(hs.gather_claimed, MTR * sizeof(int));
    z(hs.combine_token_ready, MTR * sizeof(int));
    z(hs.gather_ready_queue, MTR * sizeof(int));
    z(hs.gather_ready_head, sizeof(int));
    z(hs.gather_ready_tail, sizeof(int));
    z(hs.gather_ready_reserve_tail, sizeof(int));
    z(hs.gather_scan_cursor, (size_t)COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS * sizeof(int));
    z(hs.gather_task_count, sizeof(int));
    z(hs.gather_task_tokens, MTR * sizeof(int));
    z(hs.gather_task_nhits, MTR * sizeof(int));
    z(hs.combine_done_count, sizeof(int));
    z(hs.combine_all_done, sizeof(int));
    f(hs.pending_topk_idx, MTR * TK * sizeof(int));
    z(hs.pending_topk_weights, MTR * TK * sizeof(float));
    z(hs.pub_ring, (size_t)NPUB * PUB_RING_DEPTH * sizeof(int));
    z(hs.pub_ring_head, (size_t)NPUB * sizeof(int));
    z(hs.pub_ring_tail, (size_t)NPUB * sizeof(int));
    z(hs.recv_warp_done, (size_t)NPUB * sizeof(int));
    z(hs.publish_warp_done, (size_t)NPUB * sizeof(int));
    z(hs.publish_done_count, sizeof(int));
    z(hs.publish_all_done, sizeof(int));
    z(hs.token_compute_expected, MTR * sizeof(int));
    z(hs.compute_output_slot, total_slots * hs.hidden_dim * sizeof(__nv_bfloat16));
    z(hs.token_nhits, MTR * sizeof(int));
    f(hs.token_slot_list, MTR * TK * sizeof(int));
    z(hs.priority_token_cursor, sizeof(int));
    z(hs.expert_batch_enqueued, (size_t)E * MBE * sizeof(int));
    z(hs.priority_batch_skip_epoch, (size_t)E * MBE * sizeof(int));
    z(hs.priority_batch_retry_epoch, (size_t)E * MBE * sizeof(int));
    z(hs.combine_input, MTR * hs.hidden_dim * sizeof(__nv_bfloat16));
    z(hs.combine_input_topk_weights, MTR * TK * sizeof(float));
    z(hs.output_accum, (size_t)hs.num_tokens * hs.hidden_dim * sizeof(float));
    f(hs.send_rdma_head, (size_t)NLC * combine_rdma_head_stride * sizeof(int));
    f(hs.send_nvl_head, (size_t)NLC * combine_nvl_head_stride * sizeof(int));
    z(hs.combine_rdma_head_work, (size_t)NLC * combine_rdma_head_stride * sizeof(int));
    z(hs.combine_nvl_head_work, (size_t)NLC * combine_nvl_head_stride * sizeof(int));
    z(hs.channel_dispatch_done, (size_t)NLC * sizeof(int));
    z(hs.channel_normalized, (size_t)NLC * sizeof(int));
    z(hs.dispatch_channel_barrier, (size_t)NLC * sizeof(int));
    z(hs.rdma_reuse_prelude_done, sizeof(int));
    z(hs.dispatch_round_barrier, (size_t)round_slots * sizeof(int));
    z(hs.combine_channel_barrier, (size_t)NLC * sizeof(int));
    z(hs.recv_rdma_channel_prefix_matrix, (size_t)kRDMA * NLC * sizeof(int));
    z(hs.recv_gbl_channel_prefix_matrix, (size_t)NR * NLC * sizeof(int));
    z(hs.recv_rdma_channel_token_count, (size_t)kRDMA * NLC * sizeof(int));
    z(hs.recv_gbl_channel_token_count, (size_t)NR * NLC * sizeof(int));
#if MK_PERF_TRACE_ENABLED
    reset_megakernel_perf_trace_state(hs, stream);
#endif
}

static void reset_megakernel_post_notify_state(const MegaKernelState& hs, cudaStream_t stream) {
    const int kRDMA = hs.num_ranks / NUM_MAX_NVL_PEERS;
    const int NLC = hs.num_logical_channels;
    const int TK = hs.num_topk;
    const int combine_rdma_head_stride = hs.num_tokens * kRDMA;
    const int combine_nvl_head_stride = (hs.num_tokens * TK) * NUM_MAX_NVL_PEERS;
    auto f = [&](void* p, size_t bytes) { CUDA_CHECK(cudaMemsetAsync(p, 0xff, bytes, stream)); };

    // Backward allocates a fresh state whose ordinary queues/barriers/slot buffers were already
    // initialized by allocate_megakernel_state_v7. The cached_notify replay can write the combine
    // send heads, so restore only those before launching the persistent kernel.
    f(hs.send_rdma_head, (size_t)NLC * combine_rdma_head_stride * sizeof(int));
    f(hs.send_nvl_head, (size_t)NLC * combine_nvl_head_stride * sizeof(int));
}

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
static void launch_megakernel_v7_backward_case(
    MegaKernelBackwardState* device_bs,
    int total_sms,
    int smem_size,
    cudaStream_t stream
) {
    using RdmaCfg = MegaKernelRdmaConfig<kNumRDMARanks>;
    constexpr int kThreads = RdmaCfg::kMegaKernelNumThreads;
    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(moe_megakernel_v7_backward<kNumRDMARanks, kStage, kComputeDType>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    constexpr int num_gather_sms = GATHER_SMS;
    const int launch_total_sms = total_sms + num_gather_sms;
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
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_megakernel_v7_backward<kNumRDMARanks, kStage, kComputeDType>, device_bs));
#else
    moe_megakernel_v7_backward<kNumRDMARanks, kStage, kComputeDType><<<launch_total_sms, kThreads, smem_size, stream>>>(device_bs);
#endif
    CUDA_CHECK(cudaGetLastError());
}

void prepare_megakernel_backward_communication_replay(
    const MegaKernelBackwardHostContext* host_context,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream
) {
    EP_HOST_ASSERT(host_context != nullptr);
    EP_HOST_ASSERT(host_context->backward_state.bwd_device_state != nullptr);
    prepare_megakernel_communication_replay_host(
        host_context->bwd_state, dispatch_barrier_signal_ptrs,
        combine_barrier_signal_ptrs, stream);
}

void launch_megakernel_debug_backward(
    MegaKernelBackwardState* backward_state,
    const MegaKernelBackwardHostContext* host_context,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream
) {
    EP_HOST_ASSERT(backward_state != nullptr);
    EP_HOST_ASSERT(host_context != nullptr);
    EP_HOST_ASSERT(compute_dtype == ComputeDType::kBF16 &&
                   "megakernel debug backward currently supports BF16 only");

    // The backward-specific fills (send_rdma_head, send_nvl_head, grad_input) are now
    // merged into allocate_megakernel_backward_state and executed before the sync there.
    // No separate fused_fill_kernel launch is needed here.

    const MegaKernelBackwardState& hbs = host_context->backward_state;
    const MegaKernelState& hstate = host_context->bwd_state;

    // trace_backward_boundary_kernel<<<1, 1, 0, stream>>>(backward_state, 0);
    CUDA_CHECK(cudaGetLastError());

    const int num_ranks = hstate.num_ranks;
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);

    // The public API receives the physical GPU SM count, while the forward launch uses
    // only its configured role SMs and adds two gather blocks internally. Reconstruct
    // that same active count here; adding gather blocks to the physical SM count makes
    // a cooperative grid larger than the device residency limit.
    const int active_total_sms = hstate.num_dispatch_sms + hstate.num_combine_sms +
        COMPUTE_SCHEDULER_SMS + hstate.num_compute_sms;
    EP_HOST_ASSERT(active_total_sms > 0 && active_total_sms <= total_sms);

#define MEGAKERNEL_BWD_STAGE_CASE(kNumRDMARanks, kStage, kComputeDType) \
    launch_megakernel_v7_backward_case<kNumRDMARanks, kStage, kComputeDType>(backward_state, active_total_sms, smem_size, stream); \
    break

#define MEGAKERNEL_BWD_CASE_WITH_DTYPE(kNumRDMARanks, kComputeDType) \
    switch (stage) { \
        case 1: MEGAKERNEL_BWD_STAGE_CASE(kNumRDMARanks, 1, kComputeDType); \
        case 2: MEGAKERNEL_BWD_STAGE_CASE(kNumRDMARanks, 2, kComputeDType); \
        default: EP_HOST_ASSERT(false && "Unsupported megakernel stage"); \
    } \
    break

#define MEGAKERNEL_BWD_CASE(kNumRDMARanks) \
    MEGAKERNEL_BWD_CASE_WITH_DTYPE(kNumRDMARanks, ComputeDType::kBF16)

    SWITCH_RDMA_RANKS(MEGAKERNEL_BWD_CASE);

    // trace_backward_boundary_kernel<<<1, 1, 0, stream>>>(backward_state, 1);
    CUDA_CHECK(cudaGetLastError());

#if MK_PERF_TRACE_ENABLED
    CUDA_CHECK(cudaStreamSynchronize(stream));
    dump_perf_trace_perfetto(hstate, active_total_sms, "backward");
#endif

    // Weight-gradient operands are consumed by the host-side QuACK grouped GEMMs.
    // Keep this CUDA entry point limited to the megakernel backward itself.

#undef MEGAKERNEL_BWD_CASE
#undef MEGAKERNEL_BWD_CASE_WITH_DTYPE
#undef MEGAKERNEL_BWD_STAGE_CASE
}

} // namespace megakernel_debug

namespace megakernel_state_cache {

megakernel_debug::MegaKernelState* alloc_host() {
    return new megakernel_debug::MegaKernelState{};
}

void copy_host(megakernel_debug::MegaKernelState* dst, const megakernel_debug::MegaKernelState* src) {
    *dst = *src;
}

void free_host(megakernel_debug::MegaKernelState* ptr) {
    delete ptr;
}

}  // namespace megakernel_state_cache

} // namespace deep_ep

