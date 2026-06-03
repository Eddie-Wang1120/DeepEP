"""
Correctness test for MoE MegaKernel.
Tests the fused dispatch→compute→combine pipeline against a PyTorch reference.
All experts are routed locally (single-node, no cross-rank RDMA).

Usage:
    python test_megakernel.py [--num-processes 8]
"""
import argparse
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, calc_diff


def pytorch_moe_reference(x, topk_idx, topk_weights, W_gate, W_up, W_down,
                          num_experts, rank, num_ranks):
    """
    Batched reference MoE forward in PyTorch (efficient for large token counts).
    Groups tokens by expert and does batched matmul.
    """
    num_tokens, hidden = x.shape
    num_topk = topk_idx.shape[1]
    experts_per_rank = num_experts // num_ranks

    output = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)

    # Flatten token-topk pairs
    flat_expert_ids = topk_idx.reshape(-1)  # [num_tokens * topk]
    flat_weights = topk_weights.reshape(-1)  # [num_tokens * topk]
    flat_token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, num_topk).reshape(-1)

    # Filter to local experts only
    local_expert_start = rank * experts_per_rank
    local_expert_end = local_expert_start + experts_per_rank
    local_mask = (flat_expert_ids >= local_expert_start) & (flat_expert_ids < local_expert_end)

    local_expert_ids = flat_expert_ids[local_mask]
    local_weights = flat_weights[local_mask]
    local_token_indices = flat_token_indices[local_mask]
    local_expert_ids_local = local_expert_ids - local_expert_start

    # Process each expert in a batch
    for e in range(experts_per_rank):
        expert_mask = (local_expert_ids_local == e)
        if not expert_mask.any():
            continue

        token_ids = local_token_indices[expert_mask]
        weights = local_weights[expert_mask]  # [num_tokens_for_expert]

        # Gather tokens for this expert: [batch, hidden]
        tokens = x[token_ids].float()

        # GEMM1: gate = tokens @ W_gate[e].T -> [batch, intermediate]
        gate = torch.matmul(tokens, W_gate[e].float().T)
        # GEMM1': up = tokens @ W_up[e].T -> [batch, intermediate]
        up = torch.matmul(tokens, W_up[e].float().T)

        # SwiGLU
        swiglu_out = torch.nn.functional.silu(gate) * up

        # GEMM2: down = swiglu_out @ W_down[e].T -> [batch, hidden]
        down = torch.matmul(swiglu_out, W_down[e].float().T)

        # Weighted accumulate (scatter_add equivalent)
        weighted_down = down * weights.unsqueeze(1)
        output.scatter_add_(0, token_ids.unsqueeze(1).expand_as(weighted_down).long(), weighted_down)

    return output.to(torch.bfloat16)


def test_main(local_rank, num_local_ranks, rank, num_ranks, buffer, group):
    """Run megakernel correctness test on a single rank."""
    torch.manual_seed(42 + rank)

    # Large-scale test configuration
    num_tokens = 8192
    hidden = 2048       # Must be multiple of 16 for wmma
    intermediate = 2048  # Must be multiple of 16 for wmma
    experts_per_rank = 32
    num_experts = num_ranks * experts_per_rank
    num_topk = 8

    # Generate test data
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda') * 0.1
    # Route all tokens to local experts only
    local_expert_start = rank * experts_per_rank
    topk_idx = torch.randint(local_expert_start, local_expert_start + experts_per_rank,
                             (num_tokens, num_topk), dtype=torch.int32, device='cuda')
    topk_weights = torch.randn(num_tokens, num_topk, dtype=torch.float32, device='cuda').abs()
    # Normalize weights per token
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    # Expert weights: [num_local_experts, intermediate/hidden, hidden/intermediate]
    W_gate = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_up = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_down = torch.randn(experts_per_rank, hidden, intermediate, dtype=torch.bfloat16, device='cuda') * 0.02

    if local_rank == 0:
        print(f'[Rank {rank}] Config: num_tokens={num_tokens}, hidden={hidden}, '
              f'intermediate={intermediate}, experts_per_rank={experts_per_rank}, topk={num_topk}',
              flush=True)

    # PyTorch reference (batched, efficient)
    ref_output = pytorch_moe_reference(x, topk_idx, topk_weights,
                                       W_gate, W_up, W_down,
                                       num_experts, rank, num_ranks)

    if local_rank == 0:
        print(f'[Rank {rank}] Reference computed, launching megakernel...', flush=True)

    # MegaKernel output — total_sms must not exceed actual GPU SM count
    actual_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    mega_output = buffer.megakernel_forward(
        x, topk_idx, topk_weights,
        W_gate, W_up, W_down,
        num_experts,
        num_dispatch_sms=24,
        num_combine_sms=24,
        total_sms=actual_sms
    )

    # Compare
    diff = calc_diff(ref_output, mega_output)
    max_abs_diff = (ref_output.float() - mega_output.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.float().flatten().unsqueeze(0),
        mega_output.float().flatten().unsqueeze(0)
    ).item()

    if local_rank == 0:
        print(f'[Rank {rank}] MegaKernel correctness test:')
        print(f'  calc_diff (lower=better): {diff:.6e}')
        print(f'  max_abs_diff: {max_abs_diff:.6e}')
        print(f'  cosine_similarity: {cos_sim:.6f}')
        print(f'  ref_output norm: {ref_output.float().norm().item():.4f}')
        print(f'  mega_output norm: {mega_output.float().norm().item():.4f}')

        if diff < 1e-2 and cos_sim > 0.99:
            print('  PASSED')
        else:
            print('  FAILED - output mismatch')
            # Print first few values for debugging
            print(f'  ref[:5]:  {ref_output[0, :5].float().tolist()}')
            print(f'  mega[:5]: {mega_output[0, :5].float().tolist()}')

    return diff, max_abs_diff, cos_sim


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    buffer = deep_ep.Buffer(group,
                            int(2e9),
                            int(1e9),
                            low_latency_mode=False,
                            num_qps_per_rank=24,
                            explicitly_destroy=True)

    if local_rank == 0:
        print(f'[Rank {rank}] Buffer initialized, num_ranks={num_ranks}', flush=True)

    diff, max_abs_diff, cos_sim = test_main(
        local_rank, num_local_ranks, rank, num_ranks, buffer, group)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MoE MegaKernel correctness')
    parser.add_argument('--num-processes', type=int, default=8)
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
