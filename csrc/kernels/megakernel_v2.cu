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

constexpr int COMPUTE_BATCH_SIZE = 128;  // Tokens per expert batch before triggering GEMM
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

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
    kCompute        // Polls expert_recv_count, does GEMM+SwiGLU
};

// ============================================================================
// MegaKernel State (device-side, passed as kernel arg)
// ============================================================================

struct MegaKernelState {
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
    const int* rdma_channel_prefix_matrix;   // Per-rank token counts per channel
    const int* recv_rdma_rank_prefix_sum;    // Prefix sums for forwarder
    const int* gbl_channel_prefix_matrix;    // Global NVL-level prefix matrix
    const int* recv_gbl_rank_prefix_sum;     // Global rank prefix sums
    int* send_rdma_head;             // [num_tokens * kNumRDMARanks] for combine
    int* send_nvl_head;              // For combine NVL tracking
    int* recv_rdma_channel_prefix_matrix;   // Written by forwarder
    int* recv_gbl_channel_prefix_matrix;    // Written by NVL receiver

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
    int* expert_recv_count;           // [num_local_experts] atomic counter
    int* dispatch_done;               // Flag: set to 1 when all dispatch+forward is finished
    int* dispatch_done_count;         // Atomic: how many NVL receiver warps have finished
    int expected_dispatch_done_count; // Expected number of NVL receiver warp completions

    // --- Per-expert receive storage (filled by NVL receiver) ---
    __nv_bfloat16* recv_tokens;       // [num_local_experts * max_tokens_per_expert, hidden]
    int* expert_token_offsets;        // [num_local_experts] — atomic write offset
    int* recv_token_source_info;      // [max_total_recv_tokens, 2] — (recv_token_idx, topk_slot)
    float* recv_token_route_weights;  // [max_total_recv_tokens] — route weight for this compute slot
    internode::SourceMeta* recv_src_meta; // [max_total_recv_tokens] — DeepEP SourceMeta for combine routing

    // --- Compute state ---
    int* compute_done_count;          // Atomic: how many experts have finished compute
    int* expert_compute_cursor;       // [num_local_experts] — how many tokens already computed

    // --- Expert weights ---
    const __nv_bfloat16* W_gate;      // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_up;        // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_down;      // [num_local_experts, hidden, intermediate]

    // --- Compute output buffer ---
    __nv_bfloat16* compute_output;    // [max_total_recv_tokens, hidden]
    __nv_bfloat16* scatter_output;    // [max_total_recv_tokens, hidden] compute output scattered back to recv_token_idx namespace
    float* scatter_topk_weights;      // [max_total_recv_tokens, num_topk] topk weights scattered back to recv_token_idx namespace
    internode::SourceMeta* scatter_src_meta; // [max_total_recv_tokens] SourceMeta scattered back to recv_token_idx namespace
    int* scatter_done;                // Atomic flag: compute outputs have been scattered for combine
    int* scatter_token_ready;         // [max_total_recv_tokens] per-token ready flag for combine-compute overlap
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
    const int* combined_rdma_head;            // [num_combined_tokens * kNumRDMARanks]
    int* combined_nvl_head;                   // [num_combined_tokens * NUM_MAX_NVL_PEERS]
    const void* combine_src_meta;             // SourceMeta array
    const int* combine_rdma_channel_prefix_matrix;
    const int* combine_rdma_rank_prefix_sum;
    const int* combine_gbl_channel_prefix_matrix;
    int combine_num_tokens;                   // num tokens for combine (= tokens received by this rank)
    int combine_num_combined_tokens;          // num combined tokens (= original dispatch num_tokens)
    int combine_hidden;                       // hidden dim in dtype units
    int num_max_combine_rdma_chunked_send_tokens;
    int num_max_combine_rdma_chunked_recv_tokens;
    int num_max_combine_nvl_chunked_send_tokens;
    int num_max_combine_nvl_chunked_recv_tokens;
    int num_combine_sms;                      // Must be even (even/odd SM pairing)
    int num_combine_channels;                 // = num_combine_sms / 2

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

    // --- Dispatch SM config ---
    int num_dispatch_channels;        // = num_dispatch_sms / 2 (even/odd SM pairing)

#ifdef MK_PERF_TRACE
    // Per-SM phase timing for fine-grained overlap visualization
    // Layout: perf_phase_ts[sm_id * MK_PERF_NUM_PHASES + phase_id] = clock64()
    // Phases per role:
    //   Dispatch: 0=enter, 1=exit
    //   Compute:  0=enter, 1=first_batch_start, 2=exit
    //   Combine:  0=enter, 1=dispatch_done_acquired, 2=head_norm_done, 3=combine_protocol_start, 4=exit
    static constexpr int MK_PERF_NUM_PHASES = 5;
    int64_t* perf_phase_ts;
    int perf_total_sms;
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
    int warp_id, int num_warps,
    float* smem_buf
) {
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

__device__ void device_swiglu(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    int total_elements,
    int thread_id, int num_threads
) {
    for (int i = thread_id; i < total_elements; i += num_threads) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float silu_g = g / (1.0f + expf(-g));
        output[i] = __float2bfloat16(silu_g * u);
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

    // printf("enter dispatch worker v2\n");
    // if (threadIdx.x == 0) {
    //     printf("sm_id: %d, dispatch_sm_idx: %d\n", sm_id, dispatch_sm_idx);
    // }

    const auto num_sms = state->num_dispatch_sms;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
    const bool is_forwarder = dispatch_sm_idx % 2 == 0;
    const auto rdma_rank = state->rank / NUM_MAX_NVL_PEERS, nvl_rank = state->rank % NUM_MAX_NVL_PEERS;
    const auto num_ranks = state->num_ranks;

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
    int* send_nvl_head = state->send_nvl_head;
    int* recv_rdma_channel_prefix_matrix = state->recv_rdma_channel_prefix_matrix;
    int* recv_gbl_channel_prefix_matrix = state->recv_gbl_channel_prefix_matrix;

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    void* rdma_buffer_ptr = state->rdma_buffer_ptr;
    auto rdma_channel_data = SymBuffer<uint8_t>(
        rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

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

    // Allocate NVL buffers
    auto nvl_channel_x = AsymBuffer<uint8_t>(ws_rr_buffer_ptr,
                                             num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                             NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                             .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_start =
        AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end =
        AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

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

    // ========== kRDMASender ==========
    if (warp_role == WarpRole::kRDMASender) {
        // printf("enter v3 role\n");
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Send channel prefix metadata
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] =
                    -(channel_id == 0
                          ? 0
                          : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) - 1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] =
                    -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels + channel_id] - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
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
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
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

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks)
                send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

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

        // Clean shared memory
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;
        sync_rdma_sender_smem();

        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
        }

        // printf("RdmaSenderCoordinator: num_tokens_to_send: %lld\n", num_tokens_to_send);

        int last_issued_tail = 0;
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
                    recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, dst_nvl_rank);
                    trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head
        // printf("src_rdma_channel_prefix: %d, dst_nvl_rank: %d\n", src_rdma_channel_prefix, dst_nvl_rank);
        send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        int src_rdma_rank = dispatch_sm_idx % kNumRDMARanks;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
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
// #ifdef MK_TOKEN_TRACE
//                 if (lane_id == 0 && is_in_dst_nvl_rank) {
//                     auto fwd_topk_idx_ptr = reinterpret_cast<int*>(shifted + hidden_bytes + scale_bytes + sizeof(SourceMeta));
//                     auto fwd_topk_w_ptr = reinterpret_cast<float*>(fwd_topk_idx_ptr + num_topk);
//                     int gbl_expert = ld_nc_global(fwd_topk_idx_ptr);
//                     int lcl_expert = gbl_expert - state->rank * (num_experts / num_ranks);
//                     printf("[MK-TOKEN][DISPATCH-FWD] rank=%d src_rdma=%d dst_nvl=%d slot=%d ch=%d expert=%d local_expert=%d topk0_w=%f h0=%f\n",
//                            state->rank, src_rdma_rank, dst_nvl_rank, rdma_slot_idx, channel_id,
//                            gbl_expert, lcl_expert, ld_nc_global(fwd_topk_w_ptr),
//                            __bfloat162float(reinterpret_cast<nv_bfloat16*>(shifted)[0]));
//                 }
// #endif
                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
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
        if (elect_one_sync())
            forward_channel_retired[dst_nvl_rank] = true;

    // ========== kForwarderCoordinator ==========
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        if (target_rank > 0)
            return;
        // printf("enter v6 role\n");

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
                printf("MK dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage
        if (lane_id < kNumRDMARanks)
            recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        __syncwarp();

        // printf("NVLReceivers: num_tokens_to_recv: %d\n", num_tokens_to_recv);

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // Wait for data
            start_time = clock64();
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

                // Copy token data to scatter_output[recv_token_idx]
                for (int h = lane_id; h < state->hidden_dim; h += 32)
                    state->scatter_output[recv_token_idx * state->hidden_dim + h] = src_data[h];

                // Fill topk weights and src_meta, and record mapping for compute
                const int local_expert_end = local_expert_begin + state->num_local_experts;
                for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
                    int expert_id = ld_nc_global(topk_data_ptr + topk_slot);
                    if (expert_id < local_expert_begin || expert_id >= local_expert_end)
                        continue;
                    int local_expert_id = expert_id - local_expert_begin;
                    float route_w = ld_nc_global(weight_data_ptr + topk_slot);
                    if (lane_id == 0) {
                        state->scatter_topk_weights[recv_token_idx * num_topk + topk_slot] = route_w;
                        state->scatter_src_meta[recv_token_idx] = meta;
                        // Record mapping: expert slot -> recv_token_idx (for compute to signal)
                        int slot = atomicAdd(&state->expert_token_offsets[local_expert_id], 1);
                        int dest_offset = local_expert_id * state->max_tokens_per_expert + slot;
                        state->recv_token_source_info[dest_offset * 2] = static_cast<int>(recv_token_idx);
                        state->recv_token_source_info[dest_offset * 2 + 1] = topk_slot;
                        atomicAdd(&state->expert_recv_count[local_expert_id], 1);
                    }
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
        __syncwarp();
        if (lane_id == 0) {
            int done_count = atomicAdd(state->dispatch_done_count, 1) + 1;
            if (done_count == state->expected_dispatch_done_count)
                atomicMax(state->dispatch_done, 1);
        }
    }
}

// ============================================================================
// Compute Worker: polls expert_recv_count, does GEMM+SwiGLU in batches
// ============================================================================

__device__ void compute_worker(
    int sm_id,
    int compute_sm_idx,  // 0-based index among compute SMs
    int num_compute_sms,
    MegaKernelState* state,
    float* smem_wmma_buf
) {
    const int thread_id = threadIdx.x;
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;
    constexpr int COMPUTE_BATCH_SIZE = 10;

    // Each compute SM handles a subset of experts (round-robin)
    for (int expert_id = compute_sm_idx; expert_id < num_local_experts; expert_id += num_compute_sms) {
        int computed_so_far = 0;

        while (true) {
            int arrived = atomicAdd(&state->expert_recv_count[expert_id], 0);
            int ready_tokens = arrived - computed_so_far;

            bool should_compute = (ready_tokens >= COMPUTE_BATCH_SIZE);
            bool done = false;

            if (!should_compute) {
                int dispatch_done_count = atomicAdd(state->dispatch_done_count, 0);
                done = (dispatch_done_count == state->expected_dispatch_done_count);
                if (done && ready_tokens > 0)
                    should_compute = true;
                if (done && ready_tokens == 0)
                    break;
            }

            if (!should_compute) {
                if (thread_id == 0) __nanosleep(64);
                __syncthreads();
                continue;
            }

            int batch_size = min(ready_tokens, COMPUTE_BATCH_SIZE);

#ifdef MK_PERF_TRACE
            if (computed_so_far == 0 && expert_id == compute_sm_idx && thread_id == 0 && sm_id < state->perf_total_sms)
                state->perf_phase_ts[sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 1] = clock64();
#endif

            // Simulate compute with sleep (~100us per batch)
            if (thread_id == 0) {
                for (int i = 0; i < 100; ++i)
                    __nanosleep(1000);
            }
            __syncthreads();

            // Signal per-token ready via mapping table
            int base_offset = expert_id * max_tpe + computed_so_far;
            for (int i = thread_id; i < batch_size; i += blockDim.x) {
                int recv_token_idx = state->recv_token_source_info[(base_offset + i) * 2];
                if (recv_token_idx >= 0)
                    st_release_sys_global(&state->scatter_token_ready[recv_token_idx], 1);
            }
            if (thread_id == 0) __threadfence();
            __syncthreads();

            computed_so_far += batch_size;
        }
        __syncthreads();
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
    const int num_ranks = state->num_ranks;
    constexpr int kNumRDMARanks_C = MK_NUM_RDMA_RANKS;
    const int rdma_rank = state->rank / NUM_MAX_NVL_PEERS;

    // Phase 1: Head normalization can start as soon as dispatch is done (does not depend on compute)
    // Wait for dispatch_done first
    if (threadIdx.x == 0) {
        auto start_time = clock64();
        while (ld_acquire_sys_global(state->dispatch_done) == 0) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK combine dispatch_done wait timeout, rank=%d block=%d combine_sm=%d\n",
                       state->rank, blockIdx.x, combine_sm_idx);
                trap();
            }
            __nanosleep(100);
        }
    }
    __syncthreads();

#ifdef MK_PERF_TRACE
    {
        int perf_sm_id = state->num_dispatch_sms + combine_sm_idx;
        if (threadIdx.x == 0 && perf_sm_id < state->perf_total_sms)
            state->perf_phase_ts[perf_sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 1] = clock64();
    }
#endif

    // Head normalization (only depends on dispatch outputs, not compute)
    if (combine_sm_idx == 0) {
        const int* combined_rdma_head_src = state->combined_rdma_head;
        const int* combined_nvl_head_src = state->combined_nvl_head;
        int* combined_rdma_head_mut = state->combine_rdma_head_work;
        int* combined_nvl_head_mut = state->combine_nvl_head_work;
        const int* rdma_channel_prefix_matrix = state->combine_rdma_channel_prefix_matrix;
        const int* rdma_rank_prefix_sum = state->combine_rdma_rank_prefix_sum;

        for (int lane = threadIdx.x; lane < kNumRDMARanks_C * num_channels; lane += blockDim.x) {
            int rdma_lane = lane / num_channels;
            int channel = lane % num_channels;
            int token_start_idx = 0, token_end_idx = 0;
            get_channel_task_range(num_combined_tokens, num_channels, channel, token_start_idx, token_end_idx);

            int last_head = 1 << 25;
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                const int* src_head_ptr = combined_rdma_head_src + token_idx * kNumRDMARanks_C + rdma_lane;
                int* dst_head_ptr = combined_rdma_head_mut + token_idx * kNumRDMARanks_C + rdma_lane;
                int current_head = ld_nc_global(src_head_ptr);
                int normalized_head = current_head < 0 ? -last_head - 1 : current_head;
                st_na_global(dst_head_ptr, normalized_head);
                if (current_head >= 0)
                    last_head = current_head;
            }
        }

        for (int linear = threadIdx.x; linear < kNumRDMARanks_C * num_channels * NUM_MAX_NVL_PEERS; linear += blockDim.x) {
            int dst_nvl_rank = linear % NUM_MAX_NVL_PEERS;
            int channel = (linear / NUM_MAX_NVL_PEERS) % num_channels;
            int dst_rdma_rank = linear / (NUM_MAX_NVL_PEERS * num_channels);
            int token_start_idx = channel == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel - 1];
            int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel];
            int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            token_start_idx += shift;
            token_end_idx += shift;

            int last_head = 1 << 25;
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                const int* src_head_ptr = combined_nvl_head_src + token_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank;
                int* dst_head_ptr = combined_nvl_head_mut + token_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank;
                int current_head = ld_nc_global(src_head_ptr);
                int normalized_head = current_head < 0 ? -last_head - 1 : current_head;
                st_na_global(dst_head_ptr, normalized_head);
                if (current_head >= 0)
                    last_head = current_head;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
            st_release_sys_global(state->combine_notify_done, 1);
    } else if (threadIdx.x == 0) {
        auto start_time = clock64();
        while (ld_acquire_sys_global(state->combine_notify_done) == 0) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK combine notify wait timeout, rank=%d block=%d combine_sm=%d notify_done=%d\n",
                       state->rank, blockIdx.x, combine_sm_idx, ld_volatile_global(state->combine_notify_done));
                trap();
            }
            __nanosleep(100);
        }
    }
    __syncthreads();

#ifdef MK_PERF_TRACE
    {
        int perf_sm_id = state->num_dispatch_sms + combine_sm_idx;
        if (threadIdx.x == 0 && perf_sm_id < state->perf_total_sms)
            state->perf_phase_ts[perf_sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 2] = clock64();
    }
#endif

    // Per-token readiness is now checked by NVL sender inline; no need for global barrier.
    // Signal scatter_done for compatibility
    if (combine_sm_idx == 0 && threadIdx.x == 0)
        st_release_sys_global(state->scatter_done, 1);
    __syncthreads();

#ifdef MK_PERF_TRACE
    {
        int perf_sm_id = state->num_dispatch_sms + combine_sm_idx;
        if (threadIdx.x == 0 && perf_sm_id < state->perf_total_sms)
            state->perf_phase_ts[perf_sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 3] = clock64();
    }
#endif

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

    // if (threadIdx.x == 0) {
    //     printf("jinheng debug: combine hidden: %d, hidden_int4: %d, hidden_bytes: %d, num_bytes_per_token: %d\n", hidden, (int)hidden_int4, (int)hidden_bytes, (int)num_bytes_per_token);
    // }

    const auto nvl_rank = state->rank % NUM_MAX_NVL_PEERS;

    // DEBUG: Entry log (once per SM, thread 0 only)
    // if (thread_id == 0) {
    //     printf("[MK-DIAG][COMBINE][SM-ENTRY] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d is_forwarder_sm=%d num_tokens=%d num_combined_tokens=%d num_channels=%d hidden=%d hidden_int4=%d hidden_bytes=%d num_topk=%d num_bytes_per_token=%d num_dispatch_sms=%d num_combine_sms=%d num_compute_sms=%d\n",
    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, is_forwarder_sm,
    //            state->combine_num_tokens, state->combine_num_combined_tokens, num_channels, hidden,
    //            (int)hidden_int4, (int)hidden_bytes, num_topk, (int)num_bytes_per_token,
    //            state->num_dispatch_sms, state->num_combine_sms, state->num_compute_sms);
    //     if (sm_id == 0) {
    //         printf("MK combine BUFS: rdma_buffer_ptr=%p, buffer_ptrs=%p, buffer_ptrs[0]=%p\n",
    //                state->combine_rdma_buffer_ptr, state->combine_buffer_ptrs,
    //                state->combine_buffer_ptrs ? state->combine_buffer_ptrs[0] : nullptr);
    //         printf("MK combine SIZING: rdma_send=%d, rdma_recv=%d, nvl_send=%d, nvl_recv=%d\n",
    //                state->num_max_combine_rdma_chunked_send_tokens,
    //                state->num_max_combine_rdma_chunked_recv_tokens,
    //                state->num_max_combine_nvl_chunked_send_tokens,
    //                state->num_max_combine_nvl_chunked_recv_tokens);
    //     }
    // }

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

    // if (channel_id == 4 && lane_id == 0 && warp_role == WarpRole::kNVLAndRDMAForwarder) {
    //     const int physical_warp_id = thread_id / 32;
    //     const int dst_rdma_rank_dbg = warp_id / kNumWarpsPerForwarder_C;
    //     const int sub_warp_id_dbg = warp_id % kNumWarpsPerForwarder_C;
    //     printf("[MK-TRACE][COMBINE][FWD-ROLE-CH4] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d physical_warp=%d logical_warp=%d dst_rdma_rank=%d sub_warp=%d\n",
    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, physical_warp_id, warp_id,
    //            dst_rdma_rank_dbg, sub_warp_id_dbg);
    // }

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
    const int* combined_rdma_head = state->combine_rdma_head_work;
    int* combined_nvl_head = state->combine_nvl_head_work;
    const SourceMeta* src_meta = reinterpret_cast<const SourceMeta*>(state->combine_src_meta);
    const int* rdma_channel_prefix_matrix = state->combine_rdma_channel_prefix_matrix;
    const int* rdma_rank_prefix_sum = state->combine_rdma_rank_prefix_sum;
    const int* gbl_channel_prefix_matrix = state->combine_gbl_channel_prefix_matrix;
    void* rdma_buffer_ptr = state->combine_rdma_buffer_ptr;
    void** buffer_ptrs = state->combine_buffer_ptrs;

    if (warp_role == WarpRole::kNVLSender) {
        // ========== NVL Sender (direct port from internode.cu L1784-1922) ==========
        const auto dst_nvl_rank = warp_id;

        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr,
                                                 num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                                 NUM_MAX_NVL_PEERS,
                                                 channel_id,
                                                 num_channels,
                                                 nvl_rank)
                                 .advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank)
                                    .advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
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

        // Get tasks for each RDMA lane
        int token_start_idx = 0, token_end_idx = 0;
        if (lane_id < kNumRDMARanks_C) {
            int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
            // printf("[MK-DIAG][COMBINE][NVL-SENDER][LANE-RANGE] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d dst_nvl_rank=%d lane=%d src_rdma_lane=%d prefix_idx=%d token_range=[%d,%d) count=%d gbl_prefix_addr=%p nvl_x_base=%p nvl_head_ptr=%p nvl_tail_ptr=%p\n",
            //        state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, dst_nvl_rank,
            //        lane_id, lane_id, prefix_idx, token_start_idx, token_end_idx, token_end_idx - token_start_idx,
            //        gbl_channel_prefix_matrix + prefix_idx, nvl_channel_x.buffer(), nvl_channel_head.buffer() + lane_id,
            //        nvl_channel_tail.buffer() + lane_id);
        }
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;

        // DEBUG: NVL sender task range (only for ch=0, use warp-safe approach)
        {
            int my_range = (lane_id < kNumRDMARanks_C) ? (token_end_idx - token_start_idx) : 0;
            // Warp-reduce to get total tasks
            for (int offset = 16; offset > 0; offset >>= 1)
                my_range += __shfl_down_sync(0xffffffff, my_range, offset);
            // if (lane_id == 0 && channel_id == 0)
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
                    printf("MK combine NVL sender timeout, ch: %d, RDMA: %d, nvl: %d, dst NVL: %d, lane: %d, "
                           "head=%d, tail=%d, start=%d, end=%d, slots_needed=%d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id,
                           cached_channel_head_idx, cached_channel_tail_idx,
                           token_start_idx, token_end_idx, num_max_nvl_chunked_send_tokens);
                    trap();
                }
            }

            for (int i = 0; i < kNumRDMARanks_C; ++i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks_C;
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
                int num_tokens_in_chunk =
                    __shfl_sync(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);
                    // if (lane_id == 0) {
                    //     int producer_start = __shfl_sync(0xffffffff, token_start_idx, current_rdma_idx);
                    //     int producer_end = __shfl_sync(0xffffffff, token_end_idx, current_rdma_idx);
                    //     int producer_ready = __shfl_sync(0xffffffff, is_lane_ready, current_rdma_idx);
                    //     int producer_head = __shfl_sync(0xffffffff, cached_channel_head_idx, current_rdma_idx);
                    //     int producer_tail = __shfl_sync(0xffffffff, cached_channel_tail_idx, current_rdma_idx);
                    //     printf("[MK-TRACE][COMBINE][NVL-SENDER][CHUNK] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d dst_nvl_rank=%d current_rdma_idx=%d token_start=%lld token_range=[%d,%d) num_tokens_in_chunk=%d producer_ready=%d producer_head=%d producer_tail=%d nvl_tail_ptr=%p hidden_int4=%d hidden_bytes=%d num_bytes_per_token=%d nvl_channel_x_base=%p\n",
                    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, dst_nvl_rank, current_rdma_idx,
                    //            (long long)token_idx, producer_start, producer_end, num_tokens_in_chunk, producer_ready, producer_head,
                    //            producer_tail, nvl_channel_tail.buffer() + current_rdma_idx,
                    //            (int)hidden_int4, (int)hidden_bytes, (int)num_bytes_per_token, nvl_channel_x.buffer());
                    // }

                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    // Poll: wait until this token's scatter is complete
                    if (elect_one_sync()) {
                        auto wait_start = clock64();
                        while (ld_acquire_sys_global(&state->scatter_token_ready[token_idx]) == 0) {
                            if (clock64() - wait_start > NUM_TIMEOUT_CYCLES) {
                                printf("MK combine NVL sender token_ready timeout, ch: %d, dst_nvl: %d, token: %lld\n",
                                       channel_id, dst_nvl_rank, (long long)token_idx);
                                trap();
                            }
                            __nanosleep(32);
                        }
                    }
                    __syncwarp();

                    int dst_slot_idx = 0;
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
                        printf("[MK-TOKEN][COMBINE-NVL-SEND] rank=%d token=%lld dst_nvl=%d src_rdma=%d ch=%d topk0_w=%f h0=%f\n",
                               state->rank, (long long)token_idx, dst_nvl_rank, current_rdma_idx, channel_id,
                               ld_nc_global(topk_weights + token_idx * num_topk),
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
            if (lane_id < kNumRDMARanks_C and is_lane_ready)
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
        }
    } else {
        // ========== Forwarder SM: NVLAndRDMAForwarder + RDMAReceiver + Coordinator ==========
        // (direct port from internode.cu L1923-2269)

        // RDMA symmetric layout
        auto rdma_channel_data = SymBuffer<int8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks_C, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks_C, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks_C, channel_id, num_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x =
            AsymBuffer<uint8_t>(
                local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels)
                .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head =
            AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                .advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, channel_id, num_channels)
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
            constexpr int kNumStages = 2;
            constexpr int kNumTMALoadBytes = sizeof(int4) * 32;
            constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;

            extern __shared__ __align__(1024) uint8_t smem_buffer[];
            auto smem_ptr = smem_buffer + warp_id * kNumStages * kNumTMABufferBytesPerStage;
            auto tma_mbarrier = [=](const int& i) {
                return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
            };
            uint32_t tma_phase[kNumStages] = {0};
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

            int cached_nvl_channel_tail_idx = 0;
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            num_tokens_to_combine -= num_tokens_prefix;
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

            // DEBUG: Forwarder entry
            // if (lane_id == 0 && sub_warp_id == 0 && channel_id == 0)
            //     printf("MK combine FWD: sm=%d, warp=%d, dst_rdma=%d, num_tokens_to_combine=%d, "
            //            "num_tokens_prefix=%d, rdma_head_buf=%p, rdma_tail_buf=%p\n",
            //            sm_id, warp_id, dst_rdma_rank, num_tokens_to_combine, num_tokens_prefix,
            //            (void*)rdma_channel_head.buffer(dst_rdma_rank),
            //            (void*)rdma_channel_tail.buffer(rdma_rank));
            // if (lane_id == 0 && channel_id == 4) {
            //     int prefix_end = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            //     int prefix_begin = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            //     int rank_shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            //     printf("[MK-DIAG][COMBINE][FWD-ENTRY-CH4] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d sub_warp=%d local_prefix=[%d,%d) rank_shift=%d global_prefix=[%d,%d) num_tokens_to_combine=%d send_buffer=%p rdma_head_dst_ptr=%p rdma_tail_local_ptr=%p rdma_tail0_ptr=%p rdma_tail1_ptr=%p nvl_tail0_ptr=%p nvl_tail1_ptr=%p\n",
            //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
            //            dst_rdma_rank, sub_warp_id, prefix_begin, prefix_end, rank_shift,
            //            num_tokens_prefix, num_tokens_prefix + num_tokens_to_combine, num_tokens_to_combine,
            //            send_buffer, rdma_channel_head.buffer(dst_rdma_rank), rdma_channel_tail.buffer(rdma_rank),
            //            rdma_channel_tail.buffer(0), rdma_channel_tail.buffer(1), nvl_channel_tail.buffer(0),
            //            nvl_channel_tail.buffer(1));
            // }

            // if (lane_id == 0) {
            //     int prefix_end = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            //     int prefix_begin = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            //     int rank_shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            //     printf("[MK-DIAG][COMBINE][FWD][ENTRY] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d sub_warp=%d num_tokens_to_combine=%d local_prefix=[%d,%d) rank_shift=%d global_prefix=[%d,%d) rdma_prefix_addr=%p rdma_rank_prefix_addr=%p send_buffer=%p rdma_head_dst_ptr=%p rdma_tail_local_ptr=%p rdma_tail_lane0_ptr=%p rdma_tail_lane1_ptr=%p nvl_x_base=%p nvl_head_base=%p nvl_tail_base=%p\n",
            //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
            //            dst_rdma_rank, sub_warp_id, num_tokens_to_combine, prefix_begin, prefix_end,
            //            rank_shift, num_tokens_prefix, num_tokens_prefix + num_tokens_to_combine,
            //            rdma_channel_prefix_matrix + dst_rdma_rank * num_channels + channel_id,
            //            dst_rdma_rank == 0 ? nullptr : rdma_rank_prefix_sum + dst_rdma_rank - 1,
            //            send_buffer, rdma_channel_head.buffer(dst_rdma_rank), rdma_channel_tail.buffer(rdma_rank),
            //            rdma_channel_tail.buffer(0), rdma_channel_tail.buffer(1), nvl_channel_x.buffer(),
            //            nvl_channel_head.buffer_by(0), nvl_channel_tail.buffer());
            // }


            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                // if (lane_id == 0 && channel_id == 4) {
                //     int rdma_head_snapshot = ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                //     printf("[MK-DIAG][COMBINE][FWD-CHUNK-CH4] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d sub_warp=%d token_chunk=[%d,%d) global_token_chunk=[%d,%d) num_chunked_tokens=%d rdma_slot_start=%d rdma_head_snapshot=%d rdma_head_ptr=%p send_buffer=%p\n",
                //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
                //            dst_rdma_rank, sub_warp_id, token_start_idx, token_end_idx,
                //            num_tokens_prefix + token_start_idx, num_tokens_prefix + token_end_idx,
                //            num_chunked_tokens, token_start_idx % num_max_rdma_chunked_recv_tokens,
                //            rdma_head_snapshot, rdma_channel_head.buffer(dst_rdma_rank), send_buffer);
                // }
                // if (lane_id == 0) {
                //     int rdma_head_snapshot = ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                //     printf("[MK-DIAG][COMBINE][FWD][CHUNK-PLAN] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d sub_warp=%d token_chunk=[%d,%d) global_token_chunk=[%d,%d) num_chunked_tokens=%d rdma_slot_start=%d rdma_head_snapshot=%d rdma_head_ptr=%p rdma_tail_local_ptr=%p send_buffer=%p\n",
                //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
                //            dst_rdma_rank, sub_warp_id, token_start_idx, token_end_idx,
                //            num_tokens_prefix + token_start_idx, num_tokens_prefix + token_end_idx,
                //            num_chunked_tokens, token_start_idx % num_max_rdma_chunked_recv_tokens,
                //            rdma_head_snapshot, rdma_channel_head.buffer(dst_rdma_rank),
                //            rdma_channel_tail.buffer(rdma_rank), send_buffer);
                // }
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        int cur_head = ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                        printf("MK combine forwarder (RDMA check) timeout, ch: %d, dst_rdma: %d, "
                               "token_start=%d, cur_head=%d, capacity=%d, needed=%d\n",
                               channel_id, dst_rdma_rank,
                               token_start_idx, cur_head, num_max_rdma_chunked_recv_tokens, num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder_C) {
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS) {
                        expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    // if (channel_id == 4 && lane_id < NUM_MAX_NVL_PEERS && (dst_rdma_rank == 0 || (num_tokens_prefix + token_idx) >= 88 && (num_tokens_prefix + token_idx) <= 110)) {
                    //     int global_token_idx = num_tokens_prefix + token_idx;
                    //     int encoded_progress = expected_head < 0 ? -expected_head - 1 : expected_head;
                    //     printf("[MK-TRACE][COMBINE][FWD-NVL-PLAN-CH4] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d sub_warp=%d lane=%d src_nvl_rank=%d local_token=%d global_token=%d expected_head=%d encoded_progress=%d cached_tail=%d nvl_head_addr=%p nvl_tail_addr=%p nvl_x_base=%p send_buffer=%p\n",
                    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
                    //            dst_rdma_rank, sub_warp_id, lane_id, lane_id, token_idx, global_token_idx,
                    //            expected_head, encoded_progress, cached_nvl_channel_tail_idx,
                    //            combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id,
                    //            nvl_channel_tail.buffer(lane_id), nvl_channel_x.buffer(lane_id), send_buffer);
                    // }
                    start_time = clock64();
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
                            printf("MK combine forwarder (NVL check) timeout, ch: %d, dst_rdma: %d, token: %d, "
                                   "expected_head=%d, cached_tail=%d, nvl_tail_ptr=%p, sub_warp=%d\n",
                                   channel_id, dst_rdma_rank, token_idx,
                                   expected_head, cached_nvl_channel_tail_idx,
                                   (void*)nvl_channel_tail.buffer(lane_id), sub_warp_id);
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
// #ifdef MK_TOKEN_TRACE
//                     if (lane_id == 0) {
//                         auto* hptr = reinterpret_cast<nv_bfloat16*>(const_cast<int4*>(get_addr_fn(0, expected_head >= 0 ? expected_head % num_max_nvl_chunked_recv_tokens_per_rdma : 0, 0)));
//                         printf("[MK-TOKEN][COMBINE-NVL-FWD] rank=%d token=%d dst_rdma=%d head=%d ch=%d topk0_w=%f h0=%f\n",
//                                state->rank, token_idx, dst_rdma_rank, expected_head, channel_id,
//                                recv_tw_fn(0, expected_head >= 0 ? expected_head % num_max_nvl_chunked_recv_tokens_per_rdma : 0, 0),
//                                expected_head >= 0 ? __bfloat162float(hptr[0]) : 0.f);
//                     }
// #endif
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
                if (sub_warp_id == kNumWarpsPerForwarder_C - 1) {
                    auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                    const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                    const auto dst_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                    const auto src_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                    // if (lane_id == 0) {
                    //     printf("[MK-DIAG][COMBINE][FWD][RDMA-ISSUE] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d translated_dst_rank=%d local_copy=%d token_chunk=[%d,%d) global_token_chunk=[%d,%d) rdma_slot_start=%d num_chunked_tokens=%d num_bytes_per_msg=%llu src_ptr=0x%llx dst_ptr=0x%llx rdma_tail_local_ptr=%p tail_add=%d\n",
                    //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
                    //            dst_rdma_rank, translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), dst_rdma_rank == rdma_rank,
                    //            token_start_idx, token_end_idx, num_tokens_prefix + token_start_idx,
                    //            num_tokens_prefix + token_end_idx, (int)rdma_slot_idx, num_chunked_tokens,
                    //            (unsigned long long)num_bytes_per_msg, (unsigned long long)src_ptr, (unsigned long long)dst_ptr, rdma_channel_tail.buffer(rdma_rank),
                    //            num_chunked_tokens);
                    // }
                    if (dst_rdma_rank != rdma_rank) {
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
                                                          src_ptr,
                                                          num_bytes_per_msg,
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                          channel_id,
                                                          lane_id,
                                                          0);
                    } else {
                        memory_fence();
                    }

                    __syncwarp();
                    if (elect_one_sync()) {
                        // printf("[MK-DIAG][COMBINE][FWD][RDMA-TAIL-ADD] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d dst_rdma_rank=%d translated_dst_rank=%d local_copy=%d tail_ptr=%p add=%d token_chunk=[%d,%d)\n",
                        //        state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id,
                        //        dst_rdma_rank, translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), dst_rdma_rank == rdma_rank,
                        //        rdma_channel_tail.buffer(rdma_rank), num_chunked_tokens, token_start_idx, token_end_idx);
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                        num_chunked_tokens,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        channel_id,
                                                        dst_rdma_rank == rdma_rank);
                    }
                }
            }

            __syncwarp();
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;

        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // ========== RDMA Receiver (internode.cu L2145-2222) ==========
            lane_id < kNumRDMARanks_C ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);
            // printf("combine receiver: sm_id: %d num_tokens: %d num_channels: %d channel_id: %d token_start_idx: %d, token_end_idx: %d\n", sm_id, num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // DEBUG: RDMA receiver entry
            // if (lane_id < kNumRDMARanks_C)
            //     printf("[MK-DIAG][COMBINE][RDMA-RECEIVER][ENTRY] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d lane=%d token_range=[%d,%d) num_combined_tokens=%d rdma_tail_ptr=%p rdma_head_ptr=%p rdma_recv_buffer=%p combined_rdma_head_base=%p combined_x_base=%p combined_topk_weights_base=%p\n",
            //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id, lane_id,
            //            token_start_idx, token_end_idx, num_combined_tokens, rdma_channel_tail.buffer(lane_id),
            //            rdma_channel_head.buffer(lane_id), rdma_channel_data.recv_buffer(lane_id), combined_rdma_head,
            //            combined_x, combined_topk_weights);

            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers_C) {
                int expected_head = -1;
                if (lane_id < kNumRDMARanks_C) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
                    // int encoded_progress = expected_head < 0 ? -expected_head - 1 : expected_head;
                    // int tail_snapshot = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));
                    // printf("[MK-TRACE][COMBINE][RDMA-RECEIVER][PLAN] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d lane=%d token=%lld token_range=[%d,%d) expected_head=%d encoded_progress=%d cached_tail=%d tail_snapshot=%d ready=%d combined_rdma_head_addr=%p combined_x=%p combined_topk_weights=%p rdma_tail_addr=%p rdma_recv_buffer=%p\n",
                    //        state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id, lane_id, (long long)token_idx,
                    //        token_start_idx, token_end_idx, expected_head, encoded_progress, cached_channel_tail_idx,
                    //        tail_snapshot, tail_snapshot > expected_head,
                    //        combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id,
                    //        combined_x + token_idx * hidden_int4,
                    //        combined_topk_weights + token_idx * num_topk,
                    //        rdma_channel_tail.buffer(lane_id), rdma_channel_data.recv_buffer(lane_id));
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                                        : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        int tail0 = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(0)));
                        int tail1 = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(1)));
                        int head0 = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(0)));
                        int head1 = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(1)));
                        printf("MK combine RDMA receiver timeout, rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d ch=%d warp=%d token=%lld token_range=[%d,%d) expected_head=%d cached_tail=%d tail_snapshot=%d rdma_tail_ptr=%p lane=%d tail0=%d tail1=%d head0=%d head1=%d tail0_ptr=%p tail1_ptr=%p head0_ptr=%p head1_ptr=%p rdma_recv_buffer_lane=%p combined_rdma_head_addr=%p\n",
                               state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id, (long long)token_idx,
                               token_start_idx, token_end_idx, expected_head, cached_channel_tail_idx,
                               static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))),
                               (void*)rdma_channel_tail.buffer(lane_id), lane_id, tail0, tail1, head0, head1,
                               rdma_channel_tail.buffer(0), rdma_channel_tail.buffer(1),
                               rdma_channel_head.buffer(0), rdma_channel_head.buffer(1),
                               rdma_channel_data.recv_buffer(lane_id),
                               combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
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
                // if (lane_id == 0) {
                //     printf("[MK-TRACE][COMBINE][RDMA-RECEIVER][COMBINE-DONE] rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d channel=%d warp=%d token=%lld output_combined_x=%p output_topk_weights=%p\n",
                //            state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, warp_id, (long long)token_idx,
                //            combined_x + token_idx * hidden_int4, combined_topk_weights + token_idx * num_topk);
                // }
#ifdef MK_TOKEN_TRACE
                // __syncthreads();
                // __syncwarp();
                // if (lane_id == 0 && warp_id == 0) {
                //     int num_combined = state->combine_num_combined_tokens;
                //     int ntopk = state->num_topk;
                //     int hdim = state->hidden_dim;
                //     printf("[MK-TOKEN][COMBINE-FINAL] rank=%d num_combined_tokens=%d\n", state->rank, num_combined);
                //     for (int t = 0; t < num_combined; ++t) {
                //         auto* xptr = reinterpret_cast<nv_bfloat16*>(combined_x + t * hidden_int4);
                //         // Print hidden in chunks of 8 per line to keep each printf atomic
                //         for (int h = 0; h < hdim; h += 8) {
                //             int end = min(h + 8, hdim);
                //             if (h == 0)
                //                 printf("[MK-TOKEN][COMBINE-FINAL] rank=%d token=%d h[%d:%d]=%f,%f,%f,%f,%f,%f,%f,%f\n",
                //                        state->rank, t, h, end,
                //                        __bfloat162float(xptr[h+0]), h+1<end ? __bfloat162float(xptr[h+1]) : 0.f,
                //                        h+2<end ? __bfloat162float(xptr[h+2]) : 0.f, h+3<end ? __bfloat162float(xptr[h+3]) : 0.f,
                //                        h+4<end ? __bfloat162float(xptr[h+4]) : 0.f, h+5<end ? __bfloat162float(xptr[h+5]) : 0.f,
                //                        h+6<end ? __bfloat162float(xptr[h+6]) : 0.f, h+7<end ? __bfloat162float(xptr[h+7]) : 0.f);
                //             else
                //                 printf("[MK-TOKEN][COMBINE-FINAL] rank=%d token=%d h[%d:%d]=%f,%f,%f,%f,%f,%f,%f,%f\n",
                //                        state->rank, t, h, end,
                //                        __bfloat162float(xptr[h+0]), h+1<end ? __bfloat162float(xptr[h+1]) : 0.f,
                //                        h+2<end ? __bfloat162float(xptr[h+2]) : 0.f, h+3<end ? __bfloat162float(xptr[h+3]) : 0.f,
                //                        h+4<end ? __bfloat162float(xptr[h+4]) : 0.f, h+5<end ? __bfloat162float(xptr[h+5]) : 0.f,
                //                        h+6<end ? __bfloat162float(xptr[h+6]) : 0.f, h+7<end ? __bfloat162float(xptr[h+7]) : 0.f);
                //         }
                //         // topk_weights in one line
                //         if (ntopk == 2)
                //             printf("[MK-TOKEN][COMBINE-FINAL] rank=%d token=%d topk_w=[%f,%f]\n",
                //                    state->rank, t,
                //                    ld_nc_global(combined_topk_weights + t * ntopk),
                //                    ld_nc_global(combined_topk_weights + t * ntopk + 1));
                //         else
                //             printf("[MK-TOKEN][COMBINE-FINAL] rank=%d token=%d topk_w=[%f]\n",
                //                    state->rank, t,
                //                    ld_nc_global(combined_topk_weights + t * ntopk));
                //     }
                // }
                // __syncthreads();

                // if (lane_id == 0) {
                //     printf("[MK-TOKEN][COMBINE-RDMA-RECV] rank=%d token=%lld head=%d ch=%d rdma=%d nvl=%d topk0_w=%f h0=%f\n",
                //            state->rank, (long long)token_idx, expected_head, channel_id, rdma_rank, nvl_rank,
                //            ld_nc_global(combined_topk_weights + token_idx * num_topk),
                //            __bfloat162float(reinterpret_cast<nv_bfloat16*>(combined_x + token_idx * hidden_int4)[0]));
                // }
#endif
            }

            __syncwarp();
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;

        } else {
            // ========== Coordinator (internode.cu L2223-2269) ==========
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders_C / kNumRDMARanks_C;

            int last_rdma_head = 0;
            int last_nvl_head[kNumRDMARanks_C] = {0};
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
                                                        channel_id + num_channels,
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
                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS)
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
                    }
                }

                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
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
    // Layout: [Dispatch 0..D-1] [Combine D..D+C-1] [Compute D+C..total-1]
    SmRole role;
    int role_idx;

    if (sm_id < num_dispatch_sms) {
        role = SmRole::kDispatch;
        role_idx = sm_id;
    } else if (sm_id < num_dispatch_sms + num_combine_sms) {
        role = SmRole::kCombine;
        role_idx = sm_id - num_dispatch_sms;
    } else {
        role = SmRole::kCompute;
        role_idx = sm_id - num_dispatch_sms - num_combine_sms;
    }

    // if (threadIdx.x == 0) {
    //     printf("[MK-TRACE][KERNEL][ROLE] rank=%d block=%d role=%d role_idx=%d dispatch_sms=%d combine_sms=%d compute_sms=%d layout=[dispatch:0-%d combine:%d-%d compute:%d-%d] dispatch_done=%d dispatch_done_count=%d compute_done_count=%d scatter_done=%d\n",
    //            state->rank, sm_id, static_cast<int>(role), role_idx, num_dispatch_sms, num_combine_sms, num_compute_sms,
    //            num_dispatch_sms - 1, num_dispatch_sms, num_dispatch_sms + num_combine_sms - 1,
    //            num_dispatch_sms + num_combine_sms, gridDim.x - 1,
    //            ld_volatile_global(state->dispatch_done), ld_volatile_global(state->dispatch_done_count),
    //            ld_volatile_global(state->compute_done_count), ld_volatile_global(state->scatter_done));
    // }

    switch (role) {
        case SmRole::kDispatch:
#ifdef MK_PERF_TRACE
            if (threadIdx.x == 0 && sm_id < state->perf_total_sms) state->perf_phase_ts[sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 0] = clock64();
#endif
            dispatch_worker_v2(sm_id, role_idx, state);
#ifdef MK_PERF_TRACE
            if (threadIdx.x == 0 && sm_id < state->perf_total_sms) state->perf_phase_ts[sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 1] = clock64();
#endif
            break;

        case SmRole::kCombine:
#ifdef MK_PERF_TRACE
            if (threadIdx.x == 0 && sm_id < state->perf_total_sms) state->perf_phase_ts[sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 0] = clock64();
#endif
            combine_worker_v2(role_idx, state);
#ifdef MK_PERF_TRACE
            if (threadIdx.x == 0 && sm_id < state->perf_total_sms) state->perf_phase_ts[sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 4] = clock64();
#endif
            break;

        case SmRole::kCompute:
#ifdef MK_PERF_TRACE
            if (threadIdx.x == 0 && sm_id < state->perf_total_sms) state->perf_phase_ts[sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 0] = clock64();
#endif
            compute_worker(sm_id, role_idx, num_compute_sms, state, smem_wmma_buf);
#ifdef MK_PERF_TRACE
            if (threadIdx.x == 0 && sm_id < state->perf_total_sms) state->perf_phase_ts[sm_id * MegaKernelState::MK_PERF_NUM_PHASES + 2] = clock64();
#endif
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
    cudaLaunchConfig_t cfg = {0};
    cfg.gridDim = total_sms;
    cfg.blockDim = kMegaKernelNumThreads;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = stream;

    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = (total_sms % 2 == 0 ? 2 : 1);
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;

    printf("jinheng debug: enter v0 startup v2\n");

    // CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_megakernel_v7, device_state));
    moe_megakernel_v7<<<total_sms, kMegaKernelNumThreads, smem_size, stream>>>(device_state);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 同步后才能看到 printf 输出

    printf("jinheng debug: after v0 startup v2\n");

#else
    printf("jinheng debug: enter v1 startup v3\n");

    moe_megakernel_v7<<<total_sms, kMegaKernelNumThreads, smem_size, stream>>>(device_state);
    CUDA_CHECK(cudaGetLastError());

    printf("jinheng debug: after v1 startup\n");

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

    int num_dispatch_sms = host_state.num_dispatch_sms;
    int num_combine_sms = host_state.num_combine_sms;
    int perf_total_sms = host_state.perf_total_sms;
    if (perf_total_sms != total_sms) {
        printf("[MK-PERF] perf_total_sms=%d differs from launch total_sms=%d, dump clipped to allocated size\n",
               perf_total_sms, total_sms);
    }
    int dump_sms = min(total_sms, perf_total_sms);
    constexpr int NP = MegaKernelState::MK_PERF_NUM_PHASES;

    std::vector<int64_t> ts(dump_sms * NP);
    CUDA_CHECK(cudaMemcpy(ts.data(), host_state.perf_phase_ts, dump_sms * NP * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int64_t base_ts = std::numeric_limits<int64_t>::max();
    for (int i = 0; i < dump_sms; ++i) {
        int64_t enter_ts = ts[i * NP + 0];
        if (enter_ts != 0 && enter_ts < base_ts) base_ts = enter_ts;
    }
    if (base_ts == std::numeric_limits<int64_t>::max()) base_ts = 0;

    char filename[256];
    snprintf(filename, sizeof(filename), "mk_perf_trace_rank%d.json", host_state.rank);
    FILE* f = fopen(filename, "w");
    if (!f) { printf("[MK-PERF] Failed to open %s\n", filename); return; }

    fprintf(f, "[\n");
    bool first = true;
    auto emit_comma = [&]() {
        if (!first) fprintf(f, ",\n");
        first = false;
    };
    auto emit_event = [&](const char* name, const char* cat, int64_t start, int64_t end, int pid, int tid) {
        if (start == 0 || end == 0 || end <= start) return;
        int64_t ts_us = (start - base_ts) / 1000;
        int64_t dur_us = (end - start) / 1000;
        if (dur_us <= 0) dur_us = 1;
        emit_comma();
        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%lld,\"dur\":%lld,\"pid\":%d,\"tid\":%d}",
                name, cat, (long long)ts_us, (long long)dur_us, pid, tid);
    };

    // Process/thread metadata
    emit_comma();
    fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":%d,\"args\":{\"name\":\"rank %d\"}}",
            host_state.rank, host_state.rank);

    for (int i = 0; i < dump_sms; ++i) {
        const char* role;
        int role_idx;
        if (i < num_dispatch_sms) { role = "dispatch"; role_idx = i; }
        else if (i < num_dispatch_sms + num_combine_sms) { role = "combine"; role_idx = i - num_dispatch_sms; }
        else { role = "compute"; role_idx = i - num_dispatch_sms - num_combine_sms; }
        emit_comma();
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"%s_%d\"}}",
                host_state.rank, i, role, role_idx);
        emit_comma();
        fprintf(f, "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"sort_index\":%d}}",
                host_state.rank, i, i);
    }

    // Emit phase events per SM
    int pid = host_state.rank;
    for (int i = 0; i < dump_sms; ++i) {
        int64_t* p = &ts[i * NP];

        if (i < num_dispatch_sms) {
            // Dispatch: phase[0]=enter, phase[1]=exit
            emit_event("dispatch", "dispatch", p[0], p[1], pid, i);
        } else if (i < num_dispatch_sms + num_combine_sms) {
            // Combine phases:
            // [0]=enter, [1]=dispatch_done_acquired, [2]=head_norm_done, [3]=combine_protocol_start, [4]=exit
            emit_event("wait_dispatch", "combine", p[0], p[1], pid, i);
            emit_event("head_norm", "combine", p[1], p[2], pid, i);
            emit_event("combine_protocol", "combine", p[3], p[4], pid, i);
        } else {
            // Compute: [0]=enter, [1]=first_batch_start, [2]=exit
            emit_event("poll_wait", "compute", p[0], p[1], pid, i);
            emit_event("compute", "compute", p[1], p[2], pid, i);
        }
    }

    fprintf(f, "\n]\n");
    fclose(f);
    printf("[MK-PERF] Perfetto trace written to %s (%d/%d SMs)\n", filename, dump_sms, total_sms);
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
    __nv_bfloat16* recv_tokens;
    int* expert_token_offsets;
    int* recv_token_source_info;
    float* recv_token_route_weights;
    internode::SourceMeta* recv_src_meta;
    int* compute_done_count;
    int* expert_compute_cursor;
    __nv_bfloat16* compute_output;
    __nv_bfloat16* scatter_output;
    float* scatter_topk_weights;
    internode::SourceMeta* scatter_src_meta;
    int* scatter_done;
    int* scatter_token_ready;
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

    // Signaling
    CUDA_CHECK(cudaMalloc(&expert_recv_count, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_recv_count, 0, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_done_count, 0, sizeof(int)));

    // Receive storage — indexed as [local_expert_id * max_tokens_per_expert + slot]
    const size_t total_expert_slots = (size_t)num_local_experts * max_tokens_per_expert;
    size_t recv_tokens_bytes = total_expert_slots * hidden_dim * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&recv_tokens, recv_tokens_bytes));

    CUDA_CHECK(cudaMalloc(&expert_token_offsets, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_token_offsets, 0, num_local_experts * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&recv_token_source_info, total_expert_slots * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_token_route_weights, total_expert_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_src_meta, total_expert_slots * sizeof(internode::SourceMeta)));

    // Compute state
    CUDA_CHECK(cudaMalloc(&compute_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_compute_cursor, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_compute_cursor, 0, num_local_experts * sizeof(int)));

    // Combine per-expert completion signals
    int* expert_compute_done;
    CUDA_CHECK(cudaMalloc(&expert_compute_done, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_compute_done, 0, num_local_experts * sizeof(int)));

    // Compute output: same shape as recv_tokens
    CUDA_CHECK(cudaMalloc(&compute_output, recv_tokens_bytes));
    CUDA_CHECK(cudaMalloc(&scatter_output, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(scatter_output, 0, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&scatter_topk_weights, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(scatter_topk_weights, 0, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scatter_src_meta, (size_t)max_total_recv_tokens * sizeof(internode::SourceMeta)));
    CUDA_CHECK(cudaMalloc(&scatter_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(scatter_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scatter_token_ready, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(scatter_token_ready, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_notify_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_notify_done, 0, sizeof(int)));

    // GEMM workspace: intermediate results for gate/up [total_expert_slots, 2, intermediate_dim]
    size_t workspace_bytes = total_expert_slots * intermediate_dim * 2 * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&gemm_workspace, workspace_bytes));

    // Output accumulator [num_tokens, hidden_dim] in float32
    CUDA_CHECK(cudaMalloc(&output_accum, (size_t)num_tokens * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(output_accum, 0, (size_t)num_tokens * hidden_dim * sizeof(float)));

    // Dispatch tracking heads
    int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    // send_rdma_head: same as DeepEP {num_tokens, num_rdma_ranks}
    CUDA_CHECK(cudaMalloc(&send_rdma_head, num_tokens * num_rdma_ranks * sizeof(int)));
    CUDA_CHECK(cudaMemset(send_rdma_head, 0, num_tokens * num_rdma_ranks * sizeof(int)));
    // send_nvl_head: same as DeepEP {num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}
    // In megakernel we don't have num_rdma_recv_tokens at alloc time, use num_tokens * num_topk as upper bound
    int num_rdma_recv_tokens_ub = num_tokens * num_topk;
    CUDA_CHECK(cudaMalloc(&send_nvl_head, num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS * sizeof(int)));
    CUDA_CHECK(cudaMemset(send_nvl_head, 0, num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_rdma_head_work, num_tokens * num_rdma_ranks * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_rdma_head_work, 0, num_tokens * num_rdma_ranks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_nvl_head_work, num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_nvl_head_work, 0, num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS * sizeof(int)));

    // Recv channel prefix matrices (written by forwarder)
    int num_channels = num_dispatch_sms / 2;  // even/odd pairing

    printf("num_tokens: %d, um_rdma_ranks: %d, num_channels: %d\n", num_tokens, num_rdma_ranks, num_channels);

    CUDA_CHECK(cudaMalloc(&recv_rdma_channel_prefix_matrix, num_rdma_ranks * num_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_rdma_channel_prefix_matrix, 0, num_rdma_ranks * num_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_gbl_channel_prefix_matrix, num_ranks * num_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_gbl_channel_prefix_matrix, 0, num_ranks * num_channels * sizeof(int)));

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
    host_state.dispatch_done = dispatch_done;
    host_state.dispatch_done_count = dispatch_done_count;
    host_state.expected_dispatch_done_count = (num_dispatch_sms / 2) * NUM_MAX_NVL_PEERS;

    // Per-expert receive storage
    host_state.recv_tokens = recv_tokens;
    host_state.expert_token_offsets = expert_token_offsets;
    host_state.recv_token_source_info = recv_token_source_info;
    host_state.recv_token_route_weights = recv_token_route_weights;
    host_state.recv_src_meta = recv_src_meta;

    // Compute state
    host_state.compute_done_count = compute_done_count;
    host_state.expert_compute_cursor = expert_compute_cursor;

    // Expert weights
    host_state.W_gate = W_gate;
    host_state.W_up = W_up;
    host_state.W_down = W_down;

    // Compute output
    host_state.compute_output = compute_output;
    host_state.scatter_output = scatter_output;
    host_state.scatter_topk_weights = scatter_topk_weights;
    host_state.scatter_src_meta = scatter_src_meta;
    host_state.scatter_done = scatter_done;
    host_state.scatter_token_ready = scatter_token_ready;
    host_state.combine_notify_done = combine_notify_done;
    host_state.combine_rdma_head_work = combine_rdma_head_work;
    host_state.combine_nvl_head_work = combine_nvl_head_work;
    host_state.gemm_workspace = gemm_workspace;
    host_state.output_accum = output_accum;

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
    host_state.num_dispatch_channels = num_dispatch_sms / 2;  // even/odd SM pairing

    // Combine state
    host_state.num_combine_sms = num_combine_sms;
    host_state.num_combine_channels = num_combine_sms / 2;
    host_state.expert_compute_done = expert_compute_done;

    // Combine infrastructure
    void* combine_rdma_ptr = static_cast<uint8_t*>(rdma_buffer_ptr) + num_rdma_bytes;

#ifdef MK_PERF_TRACE
    int64_t* perf_phase_ts;
    int perf_total_sms = num_dispatch_sms + num_combine_sms + num_compute_sms;
    constexpr int NP = MegaKernelState::MK_PERF_NUM_PHASES;
    CUDA_CHECK(cudaMalloc(&perf_phase_ts, perf_total_sms * NP * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(perf_phase_ts, 0, perf_total_sms * NP * sizeof(int64_t)));
    host_state.perf_phase_ts = perf_phase_ts;
    host_state.perf_total_sms = perf_total_sms;
#endif

    host_state.combine_rdma_buffer_ptr = combine_rdma_ptr;
    host_state.combine_buffer_ptrs = combine_buffer_ptrs;
    host_state.combine_x = reinterpret_cast<const int4*>(scatter_output);  // combine sends compute results in recv_token_idx namespace
    host_state.combine_topk_weights = scatter_topk_weights;
    host_state.is_combined_token_in_rank = is_token_in_rank;
    host_state.combined_rdma_head = send_rdma_head;  // dispatch output, combine reads back
    host_state.combined_nvl_head = send_nvl_head;
    host_state.combine_src_meta = scatter_src_meta;  // SourceMeta in recv_token_idx namespace
    host_state.combine_rdma_channel_prefix_matrix = recv_rdma_channel_prefix_matrix;
    host_state.combine_rdma_rank_prefix_sum = recv_rdma_rank_prefix_sum;
    host_state.combine_gbl_channel_prefix_matrix = recv_gbl_channel_prefix_matrix;
    host_state.combine_num_tokens = max_total_recv_tokens;
    host_state.combine_num_combined_tokens = num_tokens;
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
    CUDA_CHECK(cudaFree(host_state.dispatch_done));
    CUDA_CHECK(cudaFree(host_state.dispatch_done_count));
    CUDA_CHECK(cudaFree(host_state.recv_tokens));
    CUDA_CHECK(cudaFree(host_state.expert_token_offsets));
    CUDA_CHECK(cudaFree(host_state.recv_token_source_info));
    CUDA_CHECK(cudaFree(host_state.recv_token_route_weights));
    CUDA_CHECK(cudaFree(host_state.recv_src_meta));
    CUDA_CHECK(cudaFree(host_state.compute_done_count));
    CUDA_CHECK(cudaFree(host_state.expert_compute_cursor));
    CUDA_CHECK(cudaFree(host_state.expert_compute_done));
    CUDA_CHECK(cudaFree(host_state.combined_x));
    CUDA_CHECK(cudaFree(host_state.combined_topk_weights));
    CUDA_CHECK(cudaFree(host_state.compute_output));
    CUDA_CHECK(cudaFree(host_state.scatter_output));
    CUDA_CHECK(cudaFree(host_state.scatter_topk_weights));
    CUDA_CHECK(cudaFree(host_state.scatter_src_meta));
    CUDA_CHECK(cudaFree(host_state.scatter_done));
    CUDA_CHECK(cudaFree(host_state.scatter_token_ready));
    CUDA_CHECK(cudaFree(host_state.combine_notify_done));
    CUDA_CHECK(cudaFree(host_state.combine_rdma_head_work));
    CUDA_CHECK(cudaFree(host_state.combine_nvl_head_work));
    CUDA_CHECK(cudaFree(host_state.gemm_workspace));
    CUDA_CHECK(cudaFree(host_state.output_accum));
    CUDA_CHECK(cudaFree(host_state.send_rdma_head));
    CUDA_CHECK(cudaFree(host_state.send_nvl_head));
    CUDA_CHECK(cudaFree(host_state.recv_rdma_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(host_state.recv_gbl_channel_prefix_matrix));
#ifdef MK_PERF_TRACE
    CUDA_CHECK(cudaFree(host_state.perf_phase_ts));
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
