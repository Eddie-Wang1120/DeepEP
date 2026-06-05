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
// For kNumRDMARanks=2, kNumCombineForwarderWarps=14:
//   kNumWarpsPerForwarder = 14/2 = 7
//   kNumForwarders = 2*7 = 14
//   kNumRDMAReceivers = 14 - 8 = 6
// Sender SM: 8 NVL senders + 6 RDMA receivers + 1 coordinator = 15 warps (480 threads)
// Forwarder SM: 14 forwarders + 1 coordinator = 15 warps (480 threads)
// Both fit in blockDim=512
constexpr int kNumCombineForwarderWarps = 14;
constexpr int kNumCombineWarpsPerForwarder = kNumCombineForwarderWarps / MK_NUM_RDMA_RANKS;  // 7
constexpr int kNumCombineForwarders = MK_NUM_RDMA_RANKS * kNumCombineWarpsPerForwarder;      // 14
constexpr int kNumCombineRDMAReceivers = kNumCombineForwarders - NUM_MAX_NVL_PEERS;          // 6
constexpr int kNumCombineTMABytesPerSenderWarp = 16384;
// Per forwarder warp: 2 stages * (sizeof(int4)*32 * (NUM_MAX_NVL_PEERS+1) + 16)
constexpr int kNumCombineTMABytesPerForwarderWarp = 9248;
constexpr int kNumTopkCombineRDMARanks = MK_NUM_RDMA_RANKS;  // get_num_topk_rdma_ranks(2)=2

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

    // --- Per-expert receive storage (filled by NVL receiver) ---
    __nv_bfloat16* recv_tokens;       // [num_local_experts * max_tokens_per_expert, hidden]
    int* expert_token_offsets;        // [num_local_experts] — atomic write offset
    int* recv_token_source_info;      // [max_total_recv_tokens, 2] — (orig_token_idx, topk_slot)

    // --- Compute state ---
    int* compute_done_count;          // Atomic: how many experts have finished compute
    int* expert_compute_cursor;       // [num_local_experts] — how many tokens already computed

    // --- Expert weights ---
    const __nv_bfloat16* W_gate;      // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_up;        // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_down;      // [num_local_experts, hidden, intermediate]

    // --- Compute output buffer ---
    __nv_bfloat16* compute_output;    // [max_total_recv_tokens, hidden]
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
    if (threadIdx.x == 0) {
        printf("sm_id: %d, dispatch_sm_idx: %d\n", sm_id, dispatch_sm_idx);
    }

    const auto num_sms = state->num_dispatch_sms;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
    const bool is_forwarder = dispatch_sm_idx % 2 == 0;
    const auto rdma_rank = state->rank / NUM_MAX_NVL_PEERS, nvl_rank = state->rank % NUM_MAX_NVL_PEERS;
    const auto num_ranks = state->num_ranks;

    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);

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

    // === TARGETED DIAG for block 16 (write OOB) — all variables/buffers initialized ===
    if (blockIdx.x == 16 && threadIdx.x == 0) {
        printf("[DIAG-BLK16-T0] sm=%d dispatch_sm_idx=%d is_fwd=%d ch_id=%d nch=%d\n",
               sm_id, dispatch_sm_idx, (int)is_forwarder, channel_id, num_channels);
        printf("[DIAG-BLK16-T0] rdma_buffer_ptr=%p rs_wr=%p ws_rr=%p nvl_rank=%d\n",
               rdma_buffer_ptr, rs_wr_buffer_ptr, ws_rr_buffer_ptr, nvl_rank);
        printf("[DIAG-BLK16-T0] send_nvl_head=%p recv_rdma_cpm=%p recv_gbl_cpm=%p\n",
               (void*)send_nvl_head, (void*)recv_rdma_channel_prefix_matrix, (void*)recv_gbl_channel_prefix_matrix);
        printf("[DIAG-BLK16-T0] nvl_chan_head=%p nvl_chan_tail=%p nvl_chan_x=%p nvl_prefix_start=%p nvl_prefix_end=%p\n",
               (void*)nvl_channel_head.buffer(), (void*)nvl_channel_tail.buffer(),
               (void*)nvl_channel_x.buffer(), (void*)nvl_channel_prefix_start.buffer(), (void*)nvl_channel_prefix_end.buffer());
        printf("[DIAG-BLK16-T0] rdma_chan_data=%p rdma_chan_meta=%p rdma_chan_head=%p rdma_chan_tail=%p\n",
               (void*)rdma_channel_data.send_buffer(0), (void*)rdma_channel_meta.send_buffer(0),
               (void*)rdma_channel_head.buffer(), (void*)rdma_channel_tail.buffer());
    }
    if (blockIdx.x == 16 && (threadIdx.x == 65 || threadIdx.x == 193)) {
        int w_id = threadIdx.x / 32, l_id = threadIdx.x % 32;
        int tgt = (w_id + channel_id) % NUM_MAX_NVL_PEERS;
        printf("[DIAG-BLK16-T%d] warp=%d lane=%d target_rank=%d nvl_rank=%d rdma_rank=%d\n",
               threadIdx.x, w_id, l_id, tgt, nvl_rank, rdma_rank);
        printf("[DIAG-BLK16-T%d] rs_wr=%p ws_rr=%p nvl_chan_head=%p nvl_chan_tail=%p\n",
               threadIdx.x, rs_wr_buffer_ptr, ws_rr_buffer_ptr,
               (void*)nvl_channel_head.buffer(), (void*)nvl_channel_tail.buffer());
        printf("[DIAG-BLK16-T%d] nvl_prefix_start=%p nvl_prefix_end=%p\n",
               threadIdx.x, (void*)nvl_channel_prefix_start.buffer(), (void*)nvl_channel_prefix_end.buffer());
    }

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
        // printf("sender: sm_id: %d num_tokens: %d num_channels: %d channel_id: %d token_start_idx: %d, token_end_idx: %d\n", sm_id, num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

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
        if (blockIdx.x == 17 && lane_id == 0) {
            printf("[DIAG-SENDER] blk=%d sm=%d ch=%d num_bytes_per_token=%d max_recv_tok=%d send_buf=%p\n",
                   blockIdx.x, sm_id, channel_id, (int)num_bytes_per_token, num_max_rdma_chunked_recv_tokens, (void*)send_buffer);
        }

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

        if (blockIdx.x == 16 && lane_id == 0) {
            printf("[DIAG-FWD-ENTRY] sm=%d warp=%d target=%d ch=%d nch=%d rdma_r=%d nvl_r=%d\n",
                   sm_id, warp_id, dst_nvl_rank, channel_id, num_channels, rdma_rank, nvl_rank);
            printf("[DIAG-FWD-ENTRY] rdma_chan_meta.recv=%p rs_wr=%p ws_rr=%p\n",
                   (void*)rdma_channel_meta.recv_buffer(0),
                   rs_wr_buffer_ptr, ws_rr_buffer_ptr);
            printf("[DIAG-FWD-ENTRY] nvl_prefix_start.buf=%p nvl_prefix_end.buf=%p\n",
                   (void*)nvl_channel_prefix_start.buffer(), (void*)nvl_channel_prefix_end.buffer());
            printf("[DIAG-FWD-ENTRY] recv_rdma_cpm=%p recv_rdma_rank_prefix_sum=%p num_channels=%d\n",
                   (void*)recv_rdma_channel_prefix_matrix, (void*)recv_rdma_rank_prefix_sum, num_channels);
        }

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

        if (lane_id == 0) {
            printf("[DIAG-FWD] sm=%d ch=%d dst_nvl=%d prefix=%d num_to_recv=%d base=%p\n",
                   sm_id, channel_id, dst_nvl_rank, src_rdma_channel_prefix,
                   num_tokens_to_recv_from_rdma, (void*)send_nvl_head);
        }

        // Shift cached head
        // printf("src_rdma_channel_prefix: %d, dst_nvl_rank: %d\n", src_rdma_channel_prefix, dst_nvl_rank);
        send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;
        if (lane_id == 0) {
            printf("[DIAG-FWD] sm=%d shifted=%p off=%d max=%d\n",
                   sm_id, (void*)send_nvl_head,
                   src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + (int)dst_nvl_rank,
                   num_tokens * (int)NUM_MAX_NVL_PEERS);
        }

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
                if (blockIdx.x == 16 && lane_id == 0) {
                    printf("[DIAG-FWD-TMA] i=%d rdma_slot=%d src=%p dst_slot=%d dst=%p nbytes=%d\n",
                           i, (int)rdma_slot_idx, (void*)shifted, dst_slot_idx, (void*)dst_shifted, (int)num_bytes_per_token);
                }
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
        if (blockIdx.x == 17 && lane_id == 0) {
            printf("[DIAG-NVL-RECV] blk=%d sm=%d src_nvl=%d num_tokens_to_recv=%d total_offset=%d local_expert_begin=%d\n",
                   blockIdx.x, sm_id, src_nvl_rank, num_tokens_to_recv, total_offset, local_expert_begin);
            printf("[DIAG-NVL-RECV] recv_tokens=%p expert_token_offsets=%p recv_token_source_info=%p\n",
                   (void*)state->recv_tokens, (void*)state->expert_token_offsets, (void*)state->recv_token_source_info);
            printf("[DIAG-NVL-RECV] max_tokens_per_expert=%d num_local_experts=%d hidden_dim=%d\n",
                   state->max_tokens_per_expert, state->num_local_experts, state->hidden_dim);
        }

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

                // === MK-v7 ADDITION: Route token to expert storage + signal compute ===
                // Parse topk_idx from the buffer
                auto topk_data_ptr = reinterpret_cast<int*>(shifted + hidden_bytes + scale_bytes + sizeof(SourceMeta));
                auto weight_data_ptr = reinterpret_cast<float*>(topk_data_ptr + num_topk);
                auto* src_data = reinterpret_cast<const __nv_bfloat16*>(tma_buffer);
                const int local_expert_end = local_expert_begin + state->num_local_experts;

                for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
                    int expert_id = ld_nc_global(topk_data_ptr + topk_slot);
                    if (expert_id < local_expert_begin or expert_id >= local_expert_end)
                        continue;

                    int local_expert_id = expert_id - local_expert_begin;
                    int slot = -1;
                    if (lane_id == 0) {
                        printf("[DIAG-ATOMIC-GUARD] blk=%d tid=%d expert_id=%d local_expert_id=%d ptr=%p addr=%p\n",
                               blockIdx.x, threadIdx.x, expert_id, local_expert_id,
                               (void*)state->expert_token_offsets,
                               (void*)&state->expert_token_offsets[local_expert_id]);
                        slot = atomicAdd(&state->expert_token_offsets[local_expert_id], 1);
                    }
                    slot = __shfl_sync(0xffffffff, slot, 0);

                    int dest_offset = local_expert_id * state->max_tokens_per_expert + slot;
                    if (blockIdx.x == 17 && lane_id == 0) {
                        printf("[DIAG-NVL-WRITE] expert_id=%d local_expert_id=%d slot=%d dest_offset=%d max_tok=%d hidden=%d\n",
                               expert_id, local_expert_id, slot, dest_offset, state->max_tokens_per_expert, state->hidden_dim);
                    }
                    auto* dst_data = state->recv_tokens + dest_offset * state->hidden_dim;
                    for (int h = lane_id; h < state->hidden_dim; h += 32)
                        dst_data[h] = src_data[h];
                    if (lane_id == 0) {
                        state->recv_token_source_info[dest_offset * 2] = static_cast<int>(recv_token_idx);
                        state->recv_token_source_info[dest_offset * 2 + 1] = topk_slot;
                        printf("[DIAG-ATOMIC-GUARD2] blk=%d dest_offset=%d recv_count_ptr=%p addr=%p\n",
                               blockIdx.x, dest_offset,
                               (void*)state->expert_recv_count,
                               (void*)&state->expert_recv_count[local_expert_id]);
                        atomicAdd(&state->expert_recv_count[local_expert_id], 1);
                    }
                    __syncwarp();
                }

                // Wait TMA to be finished
                tma_store_wait<0>();
                __syncwarp();
            }

            // Move queue
            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }

        // Signal dispatch done (only one NVL receiver needs to set this)
        __syncwarp();
        if (lane_id == 0)
            atomicMax(state->dispatch_done, 1);
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
    // printf("enter compute worker\n");
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = thread_id / 32;
    const int num_warps = num_threads / 32;

    const int hidden = state->hidden_dim;
    const int intermediate = state->intermediate_dim;
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;

    if (blockIdx.x == 34 && thread_id == 0) {
        printf("[DIAG-CMP-ENTRY] sm=%d compute_idx=%d num_compute=%d block=%d\n",
               sm_id, compute_sm_idx, num_compute_sms, blockIdx.x);
        printf("[DIAG-CMP-ENTRY] state=%p dispatch_done=%p\n",
               (void*)state, (void*)state->dispatch_done);
        printf("[DIAG-CMP-ENTRY] expert_recv_count=%p expert_token_offsets=%p\n",
               (void*)state->expert_recv_count, (void*)state->expert_token_offsets);
        printf("[DIAG-CMP-ENTRY] recv_tokens=%p recv_token_source_info=%p\n",
               (void*)state->recv_tokens, (void*)state->recv_token_source_info);
        printf("[DIAG-CMP-ENTRY] num_local_experts=%d max_tpe=%d hidden=%d\n",
               num_local_experts, max_tpe, hidden);
    }

    // Each compute SM is responsible for a subset of experts (round-robin)
    for (int expert_id = compute_sm_idx; expert_id < num_local_experts; expert_id += num_compute_sms) {
        int computed_so_far = 0;

        while (true) {
            // Poll: how many tokens have arrived for this expert?
            if (blockIdx.x == 34 && thread_id == 0 && expert_id == compute_sm_idx) {
                printf("[DIAG-CMP-POLL] sm=%d expert=%d expert_recv_count=%p dispatch_done=%p\n",
                       sm_id, expert_id, (void*)&state->expert_recv_count[expert_id],
                       (void*)state->dispatch_done);
            }
            int arrived = atomicAdd(&state->expert_recv_count[expert_id], 0);  // atomic read

            int ready_tokens = arrived - computed_so_far;

            // Check if we can compute a batch
            bool should_compute = (ready_tokens >= COMPUTE_BATCH_SIZE);

            // Check flush condition: dispatch done + remaining tokens
            if (!should_compute) {
                int done = atomicAdd(state->dispatch_done, 0);
                if (done && ready_tokens > 0) {
                    should_compute = true;  // Flush tail
                }
                if (done && ready_tokens == 0) {
                    break;  // This expert is completely done
                }
            }

            if (!should_compute) {
                __nanosleep(64);  // Yield before re-polling
                continue;
            }

            // Determine batch size (min of ready_tokens and COMPUTE_BATCH_SIZE)
            int batch_size = min(ready_tokens, COMPUTE_BATCH_SIZE);

            // Compute: GEMM+SwiGLU for this batch
            int base_offset = expert_id * max_tpe + computed_so_far;

            if (thread_id == 0) {
                size_t max_bytes = (size_t)state->num_local_experts * state->max_tokens_per_expert * state->hidden_dim;
                size_t used_bytes = (size_t)(base_offset + batch_size) * state->hidden_dim;
                if (used_bytes > max_bytes) {
                    printf("OOB: expert=%d, max_tpe=%d, computed_so_far=%d, batch=%d, base_offset=%d, used=%zu, max=%zu\n",
                        expert_id, max_tpe, computed_so_far, batch_size, base_offset, used_bytes * sizeof(__nv_bfloat16), max_bytes * sizeof(__nv_bfloat16));
                    trap();
                }
            }

            const __nv_bfloat16* input = state->recv_tokens + base_offset * hidden;
            __nv_bfloat16* gate_out = state->gemm_workspace + base_offset * intermediate * 2;
            __nv_bfloat16* up_out = gate_out + batch_size * intermediate;
            __nv_bfloat16* output = state->compute_output + base_offset * hidden;

            const __nv_bfloat16* w_gate = state->W_gate + expert_id * intermediate * hidden;
            const __nv_bfloat16* w_up = state->W_up + expert_id * intermediate * hidden;
            const __nv_bfloat16* w_down = state->W_down + expert_id * hidden * intermediate;

            __syncthreads();

            // GEMM1: gate = input @ W_gate^T
            device_gemm_bf16(input, w_gate, gate_out,
                             batch_size, hidden, intermediate,
                             warp_id, num_warps, smem_wmma_buf);
            __syncthreads();

            // GEMM1': up = input @ W_up^T
            device_gemm_bf16(input, w_up, up_out,
                             batch_size, hidden, intermediate,
                             warp_id, num_warps, smem_wmma_buf);
            __syncthreads();

            // SwiGLU
            device_swiglu(gate_out, up_out, gate_out,
                          batch_size * intermediate,
                          thread_id, num_threads);
            __syncthreads();

            // GEMM2: output = activated @ W_down^T
            device_gemm_bf16(gate_out, w_down, output,
                             batch_size, intermediate, hidden,
                             warp_id, num_warps, smem_wmma_buf);
            __syncthreads();

            computed_so_far += batch_size;
        }

        // Signal that this expert's compute is done (per-expert flag for combine overlap)
        if (thread_id == 0) {
            atomicAdd(state->compute_done_count, 1);
            st_release_sys_global(&state->expert_compute_done[expert_id], 1);
        }
        __syncthreads();
    }
}

// ============================================================================
// Combine Worker v2: Full DeepEP internode.cu combine ported as __device__
// Polls per-expert compute_done signals, then runs the full combine protocol.
// SM pairing: even SM = NVLSender + RDMAReceiver + Coordinator
//             odd SM  = NVLAndRDMAForwarder + Coordinator
// Template constants: kNumRDMARanks=2, kNumCombineForwarderWarps=14
// ============================================================================

__device__ void combine_worker_v2(
    int combine_sm_idx,       // 0-based index among combine SMs
    MegaKernelState* state
) {
    using namespace internode;
    using dtype_t = nv_bfloat16;

    // Wait until ALL experts have finished compute (combine needs all data ready)
    if (threadIdx.x == 0) {
        for (int e = 0; e < state->num_local_experts; ++e) {
            while (ld_acquire_sys_global(&state->expert_compute_done[e]) == 0) {
                __nanosleep(100);
            }
        }
    }
    __syncthreads();

    // --- DeepEP combine kernel logic begins (direct port from internode.cu L1741-2269) ---
    enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };

    constexpr int kNumRDMARanks_C = MK_NUM_RDMA_RANKS;
    constexpr int kNumForwarders_C = kNumCombineForwarders;           // 14
    constexpr int kNumWarpsPerForwarder_C = kNumCombineWarpsPerForwarder;  // 7
    constexpr int kNumRDMAReceivers_C = kNumCombineRDMAReceivers;    // 6
    constexpr int kNumTopkRDMARanks_C = kNumTopkCombineRDMARanks;    // 2

    const auto sm_id = combine_sm_idx;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto num_channels = state->num_combine_channels, channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    const int num_topk = state->num_topk;
    const int hidden = state->combine_hidden;
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const auto hidden_bytes = hidden_int4 * sizeof(int4);
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

    const auto rdma_rank = state->rank / NUM_MAX_NVL_PEERS, nvl_rank = state->rank % NUM_MAX_NVL_PEERS;
    const int num_ranks = state->num_ranks;

    // DEBUG: Entry log (once per SM, thread 0 only)
    if (thread_id == 0 && sm_id == 0) {
        printf("MK combine ENTER: rank=%d, rdma_rank=%d, nvl_rank=%d, num_tokens=%d, num_combined_tokens=%d, "
               "num_channels=%d, hidden=%d, num_topk=%d, num_bytes_per_token=%d\n",
               state->rank, rdma_rank, nvl_rank,
               state->combine_num_tokens, state->combine_num_combined_tokens,
               num_channels, hidden, num_topk, (int)num_bytes_per_token);
        printf("MK combine BUFS: rdma_buffer_ptr=%p, buffer_ptrs=%p, buffer_ptrs[0]=%p\n",
               state->combine_rdma_buffer_ptr, state->combine_buffer_ptrs,
               state->combine_buffer_ptrs ? state->combine_buffer_ptrs[0] : nullptr);
        printf("MK combine SIZING: rdma_send=%d, rdma_recv=%d, nvl_send=%d, nvl_recv=%d\n",
               state->num_max_combine_rdma_chunked_send_tokens,
               state->num_max_combine_rdma_chunked_recv_tokens,
               state->num_max_combine_nvl_chunked_send_tokens,
               state->num_max_combine_nvl_chunked_recv_tokens);
    }

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

    const int num_tokens = state->combine_num_tokens;
    const int num_combined_tokens = state->combine_num_combined_tokens;
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
    const int* combined_rdma_head = state->combined_rdma_head;
    int* combined_nvl_head = state->combined_nvl_head;
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
        }
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;

        // DEBUG: NVL sender task range (only for ch=0, use warp-safe approach)
        {
            int my_range = (lane_id < kNumRDMARanks_C) ? (token_end_idx - token_start_idx) : 0;
            // Warp-reduce to get total tasks
            for (int offset = 16; offset > 0; offset >>= 1)
                my_range += __shfl_down_sync(0xffffffff, my_range, offset);
            if (lane_id == 0 && channel_id == 0)
                printf("MK combine NVL sender: sm=%d, ch=%d, dst_nvl=%d, total_tasks=%d\n",
                       sm_id, channel_id, dst_nvl_rank, my_range);
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

                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    int dst_slot_idx = 0;
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;
                    auto shifted_x = x + token_idx * hidden_int4;
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
            if (lane_id == 0 && sub_warp_id == 0 && channel_id == 0)
                printf("MK combine FWD: sm=%d, warp=%d, dst_rdma=%d, num_tokens_to_combine=%d, "
                       "num_tokens_prefix=%d, rdma_head_buf=%p, rdma_tail_buf=%p\n",
                       sm_id, warp_id, dst_rdma_rank, num_tokens_to_combine, num_tokens_prefix,
                       (void*)rdma_channel_head.buffer(dst_rdma_rank),
                       (void*)rdma_channel_tail.buffer(rdma_rank));

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
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) +
                            hidden_int4_idx;
                    };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                                     hidden_bytes + sizeof(SourceMeta)) +
                                            topk_idx);
                    };
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
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                        const auto dst_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        const auto src_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
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

            // DEBUG: RDMA receiver entry
            if (lane_id == 0 && channel_id == 0)
                printf("MK combine RDMA recv: sm=%d, warp=%d, ch=%d, token_range=[%d,%d), "
                       "rdma_tail_ptr=%p, num_combined_tokens=%d\n",
                       sm_id, warp_id, channel_id, token_start_idx, token_end_idx,
                       (void*)rdma_channel_tail.buffer(0), num_combined_tokens);

            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers_C) {
                int expected_head = -1;
                if (lane_id < kNumRDMARanks_C) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                                        : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("MK combine RDMA receiver timeout, ch: %d, token: %ld, "
                               "expected_head=%d, cached_tail=%d, rdma_tail_ptr=%p, lane=%d\n",
                               channel_id, token_idx,
                               expected_head, cached_channel_tail_idx,
                               (void*)rdma_channel_tail.buffer(lane_id), lane_id);
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

__global__ void __launch_bounds__(512, 1) moe_megakernel_v7(
    MegaKernelState* state
) {
    const int sm_id = blockIdx.x;
    const int num_dispatch_sms = state->num_dispatch_sms;
    const int num_combine_sms = state->num_combine_sms;
    const int num_compute_sms = state->num_compute_sms;

    if (sm_id == 0 && threadIdx.x == 0)
        printf("MK-v7 kernel entered: blocks=%d, rank=%d, dispatch=%d, combine=%d, compute=%d\n",
               gridDim.x, state->rank, num_dispatch_sms, num_combine_sms, num_compute_sms);

    // Shared memory for WMMA output (used by compute SMs only)
    __shared__ float smem_wmma_buf[16 * WMMA_M * WMMA_N];

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

    switch (role) {
        case SmRole::kDispatch:
            dispatch_worker_v2(sm_id, role_idx, state);
            break;

        case SmRole::kCombine:
            combine_worker_v2(role_idx, state);
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

void launch_megakernel_v7(
    MegaKernelState* device_state,
    int total_sms,
    int smem_size,
    cudaStream_t stream
) {
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(moe_megakernel_v7,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

#ifndef DISABLE_SM90_FEATURES
    cudaLaunchConfig_t cfg = {0};
    cfg.gridDim = total_sms;
    cfg.blockDim = 512;
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

    printf("jinheng debug: enter v0 startup");

    // CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_megakernel_v7, device_state));
    moe_megakernel_v7<<<total_sms, 512, smem_size, stream>>>(device_state);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 同步后才能看到 printf 输出

    printf("jinheng debug: after v0 startup");

#else
    printf("jinheng debug: enter v1 startup");

    moe_megakernel_v7<<<total_sms, 512, smem_size, stream>>>(device_state);

    printf("jinheng debug: after v1 startup");

#endif
}

// ============================================================================
// Host-Side State Allocation and Full Launch (v7)
// ============================================================================

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
    __nv_bfloat16* recv_tokens;
    int* expert_token_offsets;
    int* recv_token_source_info;
    int* compute_done_count;
    int* expert_compute_cursor;
    __nv_bfloat16* compute_output;
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

    // Receive storage — indexed as [local_expert_id * max_tokens_per_expert + slot]
    const size_t total_expert_slots = (size_t)num_local_experts * max_tokens_per_expert;
    size_t recv_tokens_bytes = total_expert_slots * hidden_dim * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&recv_tokens, recv_tokens_bytes));

    CUDA_CHECK(cudaMalloc(&expert_token_offsets, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_token_offsets, 0, num_local_experts * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&recv_token_source_info, total_expert_slots * 2 * sizeof(int)));

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

    // Per-expert receive storage
    host_state.recv_tokens = recv_tokens;
    host_state.expert_token_offsets = expert_token_offsets;
    host_state.recv_token_source_info = recv_token_source_info;

    // Compute state
    host_state.compute_done_count = compute_done_count;
    host_state.expert_compute_cursor = expert_compute_cursor;

    // Expert weights
    host_state.W_gate = W_gate;
    host_state.W_up = W_up;
    host_state.W_down = W_down;

    // Compute output
    host_state.compute_output = compute_output;
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

    // Combine infrastructure — mirror buffers using second half of NVSHMEM/NVL allocations
    // TODO: For dispatch/combine overlap, need in-kernel barrier + counter reset instead
    void* combine_rdma_ptr = static_cast<uint8_t*>(rdma_buffer_ptr) + num_rdma_bytes;

    // Read buffer_ptrs from device, compute offset'd pointers for combine
    void* buffer_ptrs_host[NUM_MAX_NVL_PEERS];
    CUDA_CHECK(cudaMemcpy(buffer_ptrs_host, buffer_ptrs, NUM_MAX_NVL_PEERS * sizeof(void*), cudaMemcpyDeviceToHost));
    void* combine_buffer_ptrs_host[NUM_MAX_NVL_PEERS];
    for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
        combine_buffer_ptrs_host[i] = static_cast<uint8_t*>(buffer_ptrs_host[i]) + num_nvl_bytes / 2;
    void** combine_buffer_ptrs_gpu;
    CUDA_CHECK(cudaMalloc(&combine_buffer_ptrs_gpu, NUM_MAX_NVL_PEERS * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(combine_buffer_ptrs_gpu, combine_buffer_ptrs_host, NUM_MAX_NVL_PEERS * sizeof(void*), cudaMemcpyHostToDevice));

    host_state.combine_rdma_buffer_ptr = combine_rdma_ptr;
    host_state.combine_buffer_ptrs = combine_buffer_ptrs_gpu;
    host_state.combine_x = reinterpret_cast<const int4*>(compute_output);  // combine sends compute results
    host_state.combine_topk_weights = topk_weights;
    host_state.is_combined_token_in_rank = is_token_in_rank;
    host_state.combined_rdma_head = send_rdma_head;  // dispatch output, combine reads back
    host_state.combined_nvl_head = send_nvl_head;
    host_state.combine_src_meta = recv_token_source_info;  // dispatch wrote per-token source meta
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
    CUDA_CHECK(cudaFree(host_state.recv_tokens));
    CUDA_CHECK(cudaFree(host_state.expert_token_offsets));
    CUDA_CHECK(cudaFree(host_state.recv_token_source_info));
    CUDA_CHECK(cudaFree(host_state.compute_done_count));
    CUDA_CHECK(cudaFree(host_state.expert_compute_cursor));
    CUDA_CHECK(cudaFree(host_state.expert_compute_done));
    CUDA_CHECK(cudaFree(host_state.combined_x));
    CUDA_CHECK(cudaFree(host_state.combined_topk_weights));
    CUDA_CHECK(cudaFree(host_state.combine_buffer_ptrs));
    CUDA_CHECK(cudaFree(host_state.compute_output));
    CUDA_CHECK(cudaFree(host_state.gemm_workspace));
    CUDA_CHECK(cudaFree(host_state.output_accum));
    CUDA_CHECK(cudaFree(host_state.send_rdma_head));
    CUDA_CHECK(cudaFree(host_state.send_nvl_head));
    CUDA_CHECK(cudaFree(host_state.recv_rdma_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(host_state.recv_gbl_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(device_state));
}

float* get_output_accum_ptr(MegaKernelState* device_state) {
    MegaKernelState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost));
    return host_state.output_accum;
}

}  // namespace megakernel
}  // namespace deep_ep
