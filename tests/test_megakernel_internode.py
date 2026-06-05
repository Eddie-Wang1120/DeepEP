"""
MK-v6: MegaKernel with real DeepEP internode communication.
Tests: internode_dispatch -> expert compute (PyTorch) -> internode_combine

Validates that the full pipeline produces correct results when tokens
are routed across nodes via RDMA+NVLink.

Usage (2 nodes x 8 GPUs):
    torchrun --nproc_per_node=8 --nnodes=2 \
             --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
             test_megakernel_internode.py
"""
import argparse
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import deep_ep
from utils import init_dist, calc_diff, create_grouped_scores, inplace_unique


def pytorch_moe_reference_distributed(x, topk_idx, topk_weights, W_gate, W_up, W_down,
                                       num_experts, rank, num_ranks, group):
    """
    Distributed reference MoE forward.
    Each rank computes its local experts' contribution, then allreduce.
    """
    num_tokens, hidden = x.shape
    num_topk = topk_idx.shape[1]
    experts_per_rank = num_experts // num_ranks

    output = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)

    # Flatten token-topk pairs
    flat_expert_ids = topk_idx.reshape(-1).long()
    flat_weights = topk_weights.reshape(-1)
    flat_token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, num_topk).reshape(-1)

    # Filter to local experts only (skip -1 entries)
    local_expert_start = rank * experts_per_rank
    local_expert_end = local_expert_start + experts_per_rank
    local_mask = (flat_expert_ids >= local_expert_start) & (flat_expert_ids < local_expert_end)

    local_expert_ids = flat_expert_ids[local_mask]
    local_weights = flat_weights[local_mask]
    local_token_indices = flat_token_indices[local_mask]
    local_expert_ids_local = local_expert_ids - local_expert_start

    # Process each local expert
    for e in range(experts_per_rank):
        expert_mask = (local_expert_ids_local == e)
        if not expert_mask.any():
            continue

        token_ids = local_token_indices[expert_mask]
        weights = local_weights[expert_mask]

        tokens = x[token_ids].float()

        # GEMM + SwiGLU + GEMM
        gate = torch.matmul(tokens, W_gate[e].float().T)
        up = torch.matmul(tokens, W_up[e].float().T)
        swiglu_out = F.silu(gate) * up
        down = torch.matmul(swiglu_out, W_down[e].float().T)

        # Weighted accumulate
        weighted_down = down * weights.unsqueeze(1)
        output.scatter_add_(0, token_ids.unsqueeze(1).expand_as(weighted_down).long(), weighted_down)

    # Allreduce across ranks to sum contributions from all experts
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)

    return output.to(torch.bfloat16)


def moe_compute_on_recv(recv_x, recv_topk_idx, recv_topk_weights, W_gate, W_up, W_down, experts_per_rank):
    """
    Compute MoE expert forward on received tokens, applying topk_weights.
    recv_x is de-duplicated per rank: each unique token appears ONCE.
    recv_topk_idx [num_recv, num_topk] contains LOCAL expert IDs (0..experts_per_rank-1)
    for experts on this rank, -1 for others.
    combine() does plain sum, so we apply weights here and accumulate per-token.
    """
    expert_out = torch.zeros(recv_x.shape[0], recv_x.shape[1], dtype=torch.float32, device=recv_x.device)

    for expert_id in range(experts_per_rank):
        # Find tokens assigned to this expert
        mask = (recv_topk_idx == expert_id)  # [num_recv, num_topk]
        row_mask = mask.any(dim=1)  # [num_recv]
        if not row_mask.any():
            continue

        token_indices = row_mask.nonzero(as_tuple=True)[0]
        tokens = recv_x[token_indices].float()

        # GEMM + SwiGLU + GEMM
        gate = torch.matmul(tokens, W_gate[expert_id].float().T)
        up = torch.matmul(tokens, W_up[expert_id].float().T)
        swiglu_out = F.silu(gate) * up
        down = torch.matmul(swiglu_out, W_down[expert_id].float().T)

        # Extract weight for this expert
        weights = (recv_topk_weights[token_indices] * mask[token_indices].float()).sum(dim=1, keepdim=True)

        # Accumulate weighted output (token may go to multiple local experts)
        expert_out[token_indices] += down * weights

    return expert_out.to(torch.bfloat16)


def test_main(local_rank, num_local_ranks, rank, num_ranks, buffer, group):
    """Test internode dispatch -> compute -> combine pipeline."""
    torch.manual_seed(42 + rank)

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    num_nodes = num_ranks // num_local_ranks

    # Configuration
    num_tokens = 4096
    hidden = 2048
    intermediate = 2048
    experts_per_rank = 32
    num_experts = num_ranks * experts_per_rank
    num_topk = 8
    num_topk_groups = num_nodes

    if local_rank == 0:
        print(f'[Rank {rank}] Config: num_tokens={num_tokens}, hidden={hidden}, '
              f'intermediate={intermediate}, experts_per_rank={experts_per_rank}, '
              f'topk={num_topk}, num_ranks={num_ranks}, num_nodes={num_nodes}',
              flush=True)

    # Generate test data (same pattern as test_internode.py)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda') * 0.1

    # Expert routing using grouped topk (ensures cross-node routing)
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn(num_tokens, num_topk, dtype=torch.float32, device='cuda').abs()
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    # Expert weights: each rank has its own experts
    # Use rank-seeded random so each rank's weights are deterministic and consistent
    torch.manual_seed(1000 + rank)
    W_gate = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_up = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_down = torch.randn(experts_per_rank, hidden, intermediate, dtype=torch.bfloat16, device='cuda') * 0.02

    # For the distributed reference, we need ALL expert weights on ALL ranks
    # Gather expert weights from all ranks
    all_W_gate = [torch.empty_like(W_gate) for _ in range(num_ranks)]
    all_W_up = [torch.empty_like(W_up) for _ in range(num_ranks)]
    all_W_down = [torch.empty_like(W_down) for _ in range(num_ranks)]
    dist.all_gather(all_W_gate, W_gate, group=group)
    dist.all_gather(all_W_up, W_up, group=group)
    dist.all_gather(all_W_down, W_down, group=group)
    all_W_gate = torch.cat(all_W_gate, dim=0)  # [num_experts, intermediate, hidden]
    all_W_up = torch.cat(all_W_up, dim=0)
    all_W_down = torch.cat(all_W_down, dim=0)

    # --- Step 1: Compute reference (local, using all weights) ---
    if local_rank == 0:
        print(f'[Rank {rank}] Computing reference...', flush=True)

    # Reference: compute all experts locally (no communication needed since we have all weights)
    num_tokens_local = x.shape[0]
    ref_output = torch.zeros(num_tokens_local, hidden, dtype=torch.float32, device=x.device)
    flat_expert_ids = topk_idx.reshape(-1).long()
    flat_weights = topk_weights.reshape(-1)
    flat_token_indices = torch.arange(num_tokens_local, device=x.device).unsqueeze(1).expand(-1, num_topk).reshape(-1)
    valid_mask = flat_expert_ids >= 0

    for e in range(num_experts):
        expert_mask = (flat_expert_ids == e) & valid_mask
        if not expert_mask.any():
            continue
        token_ids = flat_token_indices[expert_mask]
        weights = flat_weights[expert_mask]
        tokens = x[token_ids].float()

        gate = torch.matmul(tokens, all_W_gate[e].float().T)
        up = torch.matmul(tokens, all_W_up[e].float().T)
        swiglu_out = F.silu(gate) * up
        down = torch.matmul(swiglu_out, all_W_down[e].float().T)

        weighted_down = down * weights.unsqueeze(1)
        ref_output.scatter_add_(0, token_ids.unsqueeze(1).expand_as(weighted_down).long(), weighted_down)
    ref_output = ref_output.to(torch.bfloat16)

    # --- Step 2: DeepEP pipeline: dispatch -> compute -> combine ---
    if local_rank == 0:
        print(f'[Rank {rank}] Running DeepEP dispatch -> compute -> combine...', flush=True)

    # Get dispatch layout
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    # Config for dispatch/combine
    rdma_buffer_size = 128
    nvl_buffer_size = 512
    # Use num_sms=56 for Config to keep cached_notify smem within hardware limit:
    # num_channels = 56/2 = 28, smem = 8192*28 = 229,376 <= 232,448 bytes
    # Buffer's num_qps_per_rank (=148) >= 56, so dispatch assertion still passes.
    config_num_sms = 56
    config = deep_ep.Config(config_num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Dispatch
    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = \
        buffer.dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            config=config
        )

    if local_rank == 0:
        num_recv = recv_x.shape[0]
        print(f'[Rank {rank}] Dispatch done: recv_x shape={recv_x.shape}, '
              f'tokens_per_expert={recv_num_tokens_per_expert_list}', flush=True)
        # Debug: check recv_topk_idx format (should be LOCAL expert indices 0..experts_per_rank-1)
        if recv_topk_idx is not None:
            print(f'[Rank {rank}] recv_topk_idx shape={recv_topk_idx.shape} dtype={recv_topk_idx.dtype} '
                  f'row0={recv_topk_idx[0].tolist()} range=[0,{experts_per_rank-1}]',
                  flush=True)
        else:
            print(f'[Rank {rank}] recv_topk_idx is None!', flush=True)

    # Compute: per-expert GEMM+SwiGLU on received tokens, weighted by topk_weights
    expert_out = moe_compute_on_recv(recv_x, recv_topk_idx, recv_topk_weights,
                                      W_gate, W_up, W_down, experts_per_rank)

    # Combine: send results back with weighted reduction
    combined_x, combined_topk_weights, event = buffer.combine(
        x=expert_out,
        handle=handle,
        topk_weights=recv_topk_weights,
        config=config
    )

    if local_rank == 0:
        print(f'[Rank {rank}] Combine done: combined_x shape={combined_x.shape}', flush=True)

    # --- Step 3: Compare ---
    # combined_x should match ref_output
    diff = calc_diff(ref_output, combined_x)
    max_abs_diff = (ref_output.float() - combined_x.float()).abs().max().item()
    cos_sim = F.cosine_similarity(
        ref_output.float().flatten().unsqueeze(0),
        combined_x.float().flatten().unsqueeze(0)
    ).item()

    if local_rank == 0:
        print(f'[Rank {rank}] MK-v6 Internode correctness test:')
        print(f'  calc_diff (lower=better): {diff:.6e}')
        print(f'  max_abs_diff: {max_abs_diff:.6e}')
        print(f'  cosine_similarity: {cos_sim:.6f}')
        print(f'  ref_output norm: {ref_output.float().norm().item():.4f}')
        print(f'  combined_x norm: {combined_x.float().norm().item():.4f}')

        if diff < 1e-2 and cos_sim > 0.99:
            print('  PASSED')
        else:
            print('  FAILED - output mismatch')
            print(f'  ref[:5]:  {ref_output[0, :5].float().tolist()}')
            print(f'  mega[:5]: {combined_x[0, :5].float().tolist()}')

    return diff, max_abs_diff, cos_sim


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    buffer = deep_ep.Buffer(group,
                            int(2e9),
                            int(1e9),
                            low_latency_mode=False,
                            num_qps_per_rank=num_sms,
                            explicitly_destroy=True)

    if local_rank == 0:
        print(f'[Rank {rank}] Buffer initialized, num_ranks={num_ranks}', flush=True)

    diff, max_abs_diff, cos_sim = test_main(
        local_rank, num_local_ranks, rank, num_ranks, buffer, group)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MK-v6 internode MoE pipeline')
    parser.add_argument('--num-processes', type=int, default=8)
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
