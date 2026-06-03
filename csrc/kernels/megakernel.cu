/**
 * megakernel.cu — Fused Dispatch + TensorCore GEMM+SwiGLU + Combine MegaKernel
 *
 * Architecture:
 *   Single persistent kernel with 3 phases (sequential for Phase 1 correctness):
 *   Phase 1 (Dispatch): Copy input tokens to per-expert receive buffers
 *   Phase 2 (Compute): BF16 GEMM + SwiGLU for each expert's tokens
 *   Phase 3 (Combine): Weighted accumulation back to output
 *
 * Phase barriers use dedicated atomic arrive-wait counters for grid-wide sync.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

namespace deep_ep {
namespace megakernel {

// ============================================================================
// Configuration Constants
// ============================================================================

// GEMM tile sizes for wmma (16×16×16 BF16→FP32)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// State Structure
// ============================================================================

struct MegaKernelState {
    // --- Per-expert receive buffers (partitioned) ---
    __nv_bfloat16* recv_tokens;      // [num_local_experts * max_tokens_per_expert, hidden]
    int* recv_source_info;           // [num_local_experts * max_tokens_per_expert, 2] (token_idx, topk_slot)

    // --- Per-expert dispatch count ---
    int* expert_recv_count;          // [num_local_experts] - atomically incremented

    // --- Work-stealing counter for dispatch ---
    int* dispatch_task_counter;      // atomic counter for work-stealing

    // --- Grid barriers (dedicated counters) ---
    int* barrier_phase1_done;        // barrier after dispatch: each SM increments once
    int* barrier_phase2_done;        // barrier after compute: each SM increments once

    // --- Compute output buffer (per-expert partitioned) ---
    __nv_bfloat16* compute_output;   // [num_local_experts * max_tokens_per_expert, hidden]

    // --- Expert weights ---
    const __nv_bfloat16* W_gate;     // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_up;       // [num_local_experts, intermediate, hidden]
    const __nv_bfloat16* W_down;     // [num_local_experts, hidden, intermediate]

    // --- GEMM workspace ---
    __nv_bfloat16* gemm_workspace;   // [num_local_experts * max_tokens_per_expert, intermediate * 2]

    // --- Dimensions ---
    int hidden_dim;
    int intermediate_dim;
    int num_local_experts;
    int num_ranks;
    int rank;
    int max_tokens_per_expert;

    // --- Task counts ---
    int total_dispatch_tasks;

    // --- Output ---
    float* output_accum;             // [num_tokens, hidden] - float accumulator
    const float* topk_weights;       // [num_tokens, topk]
    int num_tokens;
    int num_topk;
};

// ============================================================================
// Grid-wide barrier using atomic arrive-wait
// ============================================================================

__device__ void grid_barrier(int* barrier_counter, int expected_count, int thread_id) {
    __syncthreads();
    __threadfence();

    if (thread_id == 0) {
        // Arrive: signal this SM is done
        atomicAdd(barrier_counter, 1);

        // Wait: spin until all SMs have arrived
        while (atomicAdd(barrier_counter, 0) < expected_count) {
            // Yield hint for sm_103
            __nanosleep(32);
        }
    }

    __syncthreads();
}

// ============================================================================
// Device GEMM using wmma
// ============================================================================

using namespace nvcuda;

__device__ void device_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (will be loaded as col-major = transposed)
    __nv_bfloat16* __restrict__ C,        // [M, N] row-major
    int M, int K, int N,
    int warp_id, int num_warps,
    float* smem_buf                       // [num_warps * WMMA_M * WMMA_N]
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
            // A[row_offset:, k:] with ldm=K
            wmma::load_matrix_sync(a_frag, A + row_offset * K + k, K);
            // B[col_offset:, k:] col-major with ldm=K means B^T[k:, col_offset:]
            wmma::load_matrix_sync(b_frag, B + col_offset * K + k, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store accumulator to shared memory
        float* c_buf = smem_buf + warp_id * WMMA_M * WMMA_N;
        wmma::store_matrix_sync(c_buf, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        // Convert and write to global output
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

/**
 * SwiGLU: output = silu(gate) * up
 */
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
// Main MegaKernel
// ============================================================================

__global__ void __launch_bounds__(512, 1) moe_megakernel(
    const __nv_bfloat16* __restrict__ input_tokens,    // [num_tokens, hidden]
    const int* __restrict__ expert_assignments,        // [num_tokens, topk]
    int num_tokens,
    int num_topk,
    MegaKernelState* state,
    void* rdma_buffer_ptr,
    int num_dispatch_sms,
    int num_combine_sms
) {
    const int sm_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = thread_id / 32;
    const int num_warps = num_threads / 32;

    const int hidden = state->hidden_dim;
    const int intermediate = state->intermediate_dim;
    const int num_local_experts = state->num_local_experts;
    const int num_ranks = state->num_ranks;
    const int rank = state->rank;
    const int max_tpe = state->max_tokens_per_expert;
    const int total_dispatch_tasks = state->total_dispatch_tasks;
    const int experts_per_rank = num_local_experts;

    // Shared memory for wmma output and work-stealing
    __shared__ float smem_wmma_buf[16 * WMMA_M * WMMA_N];  // 16KB
    __shared__ int shared_task_idx;
    __shared__ int shared_slot;

    // ========================================================================
    // PHASE 1: DISPATCH — all SMs participate via work-stealing
    // ========================================================================
    while (true) {
        if (thread_id == 0) {
            shared_task_idx = atomicAdd(state->dispatch_task_counter, 1);
        }
        __syncthreads();
        int task_idx = shared_task_idx;
        if (task_idx >= total_dispatch_tasks) break;

        int token_idx = task_idx / num_topk;
        int topk_slot = task_idx % num_topk;

        int expert_id = expert_assignments[token_idx * num_topk + topk_slot];

        // Uniform branch: all threads see same expert_id
        if (expert_id < 0) {
            __syncthreads();
            continue;
        }

        int dest_rank = expert_id / experts_per_rank;
        int local_expert_id = expert_id % experts_per_rank;

        if (dest_rank != rank) {
            __syncthreads();
            continue;
        }

        // Claim a slot in this expert's partition
        if (thread_id == 0) {
            shared_slot = atomicAdd(&state->expert_recv_count[local_expert_id], 1);
        }
        __syncthreads();
        int slot = shared_slot;

        if (slot >= max_tpe) {
            __syncthreads();
            continue;
        }

        // Destination in partitioned buffer: expert's region + slot
        int recv_offset = local_expert_id * max_tpe + slot;
        __nv_bfloat16* dst = state->recv_tokens + recv_offset * hidden;
        const __nv_bfloat16* src = input_tokens + token_idx * hidden;

        // Copy token data (element-wise, avoids alignment issues)
        for (int i = thread_id; i < hidden; i += num_threads) {
            dst[i] = src[i];
        }

        // Store source metadata for combine phase
        if (thread_id == 0) {
            state->recv_source_info[recv_offset * 2] = token_idx;
            state->recv_source_info[recv_offset * 2 + 1] = topk_slot;
        }
        __syncthreads();
    }

    // ---- Grid barrier: all SMs done with dispatch ----
    grid_barrier(state->barrier_phase1_done, gridDim.x, thread_id);

    // ========================================================================
    // PHASE 2: COMPUTE — work-steal by expert
    // ========================================================================
    for (int expert_id = sm_id; expert_id < num_local_experts; expert_id += gridDim.x) {
        int num_expert_tokens = state->expert_recv_count[expert_id];
        if (num_expert_tokens <= 0) continue;

        // Pointers into partitioned buffers
        int base_offset = expert_id * max_tpe;
        const __nv_bfloat16* input = state->recv_tokens + base_offset * hidden;
        __nv_bfloat16* gate_out = state->gemm_workspace + base_offset * intermediate * 2;
        __nv_bfloat16* up_out = gate_out + num_expert_tokens * intermediate;
        __nv_bfloat16* output = state->compute_output + base_offset * hidden;

        // Expert weight pointers
        const __nv_bfloat16* w_gate = state->W_gate + expert_id * intermediate * hidden;
        const __nv_bfloat16* w_up = state->W_up + expert_id * intermediate * hidden;
        const __nv_bfloat16* w_down = state->W_down + expert_id * hidden * intermediate;

        __syncthreads();

        // GEMM1: gate = input[M,K] @ W_gate[K,N]^T -> [M, intermediate]
        device_gemm_bf16(input, w_gate, gate_out,
                         num_expert_tokens, hidden, intermediate,
                         warp_id, num_warps, smem_wmma_buf);
        __syncthreads();

        // GEMM1': up = input[M,K] @ W_up[K,N]^T -> [M, intermediate]
        device_gemm_bf16(input, w_up, up_out,
                         num_expert_tokens, hidden, intermediate,
                         warp_id, num_warps, smem_wmma_buf);
        __syncthreads();

        // SwiGLU: gate_out = silu(gate_out) * up_out
        device_swiglu(gate_out, up_out, gate_out,
                      num_expert_tokens * intermediate,
                      thread_id, num_threads);
        __syncthreads();

        // GEMM2: output = gate_out[M,intermediate] @ W_down[intermediate,hidden]^T -> [M, hidden]
        device_gemm_bf16(gate_out, w_down, output,
                         num_expert_tokens, intermediate, hidden,
                         warp_id, num_warps, smem_wmma_buf);
        __syncthreads();
    }

    // ---- Grid barrier: all SMs done with compute ----
    grid_barrier(state->barrier_phase2_done, gridDim.x, thread_id);

    // ========================================================================
    // PHASE 3: COMBINE — weighted accumulation back to output
    // ========================================================================
    for (int expert_id = 0; expert_id < num_local_experts; expert_id++) {
        int num_expert_tokens = state->expert_recv_count[expert_id];
        int base_offset = expert_id * max_tpe;

        // Work-steal tokens within this expert across all SMs
        for (int token_local = sm_id; token_local < num_expert_tokens; token_local += gridDim.x) {
            int recv_offset = base_offset + token_local;

            // Read source info
            int src_token_idx = state->recv_source_info[recv_offset * 2];
            int src_topk_slot = state->recv_source_info[recv_offset * 2 + 1];

            // Get topk weight
            float weight = state->topk_weights[src_token_idx * state->num_topk + src_topk_slot];

            // Compute output pointer
            const __nv_bfloat16* computed = state->compute_output + recv_offset * hidden;
            float* out_accum = state->output_accum + src_token_idx * hidden;

            // Weighted accumulation (use atomicAdd for correctness with multiple topk)
            for (int i = thread_id; i < hidden; i += num_threads) {
                float val = __bfloat162float(computed[i]) * weight;
                atomicAdd(&out_accum[i], val);
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Host-Side Launch Function
// ============================================================================

void launch_megakernel(
    const __nv_bfloat16* input_tokens,
    const int* expert_assignments,
    int num_tokens,
    int num_topk,
    MegaKernelState* state,
    void* rdma_buffer_ptr,
    int num_dispatch_sms,
    int num_combine_sms,
    int total_sms,
    cudaStream_t stream
) {
    const int num_threads = 512;
    moe_megakernel<<<total_sms, num_threads, 0, stream>>>(
        input_tokens,
        expert_assignments,
        num_tokens,
        num_topk,
        state,
        rdma_buffer_ptr,
        num_dispatch_sms,
        num_combine_sms
    );
}

// ============================================================================
// Host-Side State Allocation
// ============================================================================

MegaKernelState* allocate_megakernel_state(
    int max_recv_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_local_experts,
    int num_ranks,
    int rank,
    const __nv_bfloat16* W_gate,
    const __nv_bfloat16* W_up,
    const __nv_bfloat16* W_down,
    int total_dispatch_tasks,
    int total_combine_tasks,
    int num_tokens,
    int num_topk,
    __nv_bfloat16* output_ptr,
    const float* topk_weights_ptr,
    float* output_accum_ptr
) {
    // max_tokens_per_expert: use 4x average as cap to save memory
    // (average = max_recv_tokens / num_local_experts, cap at max_recv_tokens for safety)
    int avg_tokens_per_expert = (max_recv_tokens + num_local_experts - 1) / num_local_experts;
    int max_tokens_per_expert = (avg_tokens_per_expert * 4 < max_recv_tokens) ?
                                avg_tokens_per_expert * 4 : max_recv_tokens;

    MegaKernelState host_state;
    host_state.hidden_dim = hidden_dim;
    host_state.intermediate_dim = intermediate_dim;
    host_state.num_local_experts = num_local_experts;
    host_state.num_ranks = num_ranks;
    host_state.rank = rank;
    host_state.max_tokens_per_expert = max_tokens_per_expert;
    host_state.total_dispatch_tasks = total_dispatch_tasks;
    host_state.W_gate = W_gate;
    host_state.W_up = W_up;
    host_state.W_down = W_down;
    host_state.output_accum = output_accum_ptr;
    host_state.topk_weights = topk_weights_ptr;
    host_state.num_tokens = num_tokens;
    host_state.num_topk = num_topk;

    // Allocate device buffers
    int total_recv_slots = num_local_experts * max_tokens_per_expert;

    cudaMalloc(&host_state.recv_tokens, total_recv_slots * hidden_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&host_state.recv_source_info, total_recv_slots * 2 * sizeof(int));
    cudaMalloc(&host_state.expert_recv_count, num_local_experts * sizeof(int));
    cudaMalloc(&host_state.dispatch_task_counter, sizeof(int));
    cudaMalloc(&host_state.barrier_phase1_done, sizeof(int));
    cudaMalloc(&host_state.barrier_phase2_done, sizeof(int));
    cudaMalloc(&host_state.compute_output, total_recv_slots * hidden_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&host_state.gemm_workspace, total_recv_slots * intermediate_dim * 2 * sizeof(__nv_bfloat16));

    // Zero all counters
    cudaMemset(host_state.expert_recv_count, 0, num_local_experts * sizeof(int));
    cudaMemset(host_state.dispatch_task_counter, 0, sizeof(int));
    cudaMemset(host_state.barrier_phase1_done, 0, sizeof(int));
    cudaMemset(host_state.barrier_phase2_done, 0, sizeof(int));

    // Copy state struct to device
    MegaKernelState* device_state;
    cudaMalloc(&device_state, sizeof(MegaKernelState));
    cudaMemcpy(device_state, &host_state, sizeof(MegaKernelState), cudaMemcpyHostToDevice);

    return device_state;
}

void free_megakernel_state(MegaKernelState* device_state) {
    MegaKernelState host_state;
    cudaMemcpy(&host_state, device_state, sizeof(MegaKernelState), cudaMemcpyDeviceToHost);

    cudaFree(host_state.recv_tokens);
    cudaFree(host_state.recv_source_info);
    cudaFree(host_state.expert_recv_count);
    cudaFree(host_state.dispatch_task_counter);
    cudaFree(host_state.barrier_phase1_done);
    cudaFree(host_state.barrier_phase2_done);
    cudaFree(host_state.compute_output);
    cudaFree(host_state.gemm_workspace);
    cudaFree(device_state);
}

}  // namespace megakernel
}  // namespace deep_ep
