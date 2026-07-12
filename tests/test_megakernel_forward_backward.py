"""Megakernel forward/backward alignment against Megatron's fused TE MoE path.

The baseline follows Megatron's high-performance routed-expert path:
DeepEP dispatch -> fused permutation -> TE GroupedLinear/ScaledSwiGLU/GroupedLinear
-> fused unpermutation -> DeepEP combine.

The current megakernel backward only produces dX, so this test compares forward output
and input gradients. Router-probability and expert-weight gradients are intentionally out
of scope until the CUDA backward implements them.
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from test_megakernel_v7 import (
    build_te_grouped_experts,
    make_megatron_router_inputs,
    moe_compute_on_recv_te,
)
from utils import calc_diff, init_dist

import deep_ep

import argparse
import contextlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging import version

@dataclass(frozen=True)
class TestCase:
    num_tokens: int = 16
    hidden: int = 256
    intermediate: int = 256
    experts_per_rank: int = 8
    num_topk: int = 2
    num_topk_groups: Optional[int] = None

def test(**kwargs):
    return TestCase(**kwargs)

TEST_CASES = [
    # test(num_tokens=16, hidden=2048, intermediate=2048, experts_per_rank=16, num_topk=8),
    # test(num_tokens=4096, hidden=2048, intermediate=4096, experts_per_rank=8, num_topk=4),
    test(num_tokens=16, hidden=2048, intermediate=2048, experts_per_rank=16, num_topk=8),
    # test(num_tokens=8192, hidden=4096, intermediate=4096, experts_per_rank=16, num_topk=8),
    # test(num_tokens=8192, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6),
    # Add more cases here, for example:
    # test(num_tokens=8192, hidden=256, intermediate=256, experts_per_rank=8, num_topk=2),
]


class MegatronFusedDispatch(torch.autograd.Function):
    """Megatron FusedDispatch semantics using this test's DeepEP buffer."""

    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, buffer):
        layout = buffer.get_dispatch_layout(token_indices, num_experts)
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = layout
        recv_x, recv_indices, recv_probs, tokens_per_expert, handle, _ = buffer.dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=token_indices,
            topk_weights=token_probs,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        return recv_x, recv_indices, recv_probs, torch.tensor(tokens_per_expert), handle

    @staticmethod
    def backward(ctx, grad_x, _grad_indices, _grad_probs, _grad_tokens_per_expert, _grad_handle):
        combined_x, _, _ = ctx.buffer.combine(grad_x.contiguous(), ctx.handle)
        return combined_x, None, None, None, None


class MegatronFusedCombine(torch.autograd.Function):
    """Megatron FusedCombine semantics using this test's DeepEP buffer."""

    @staticmethod
    def forward(ctx, x, buffer, handle):
        output, _, _ = buffer.combine(x=x, handle=handle)
        ctx.buffer = buffer
        ctx.handle = handle
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, _, _, _, _, _ = ctx.buffer.dispatch(
            grad_output.contiguous(), handle=ctx.handle)
        return grad_x, None, None


def run_megatron_fused_baseline(x, topk_idx, topk_weights, num_experts,
                                 experts_per_rank, buffer, te_experts):
    recv_x, recv_idx, recv_probs, tokens_per_expert, handle = MegatronFusedDispatch.apply(
        x, topk_idx, topk_weights, num_experts, buffer)
    local_output = moe_compute_on_recv_te(
        recv_x, recv_idx, recv_probs, tokens_per_expert,
        te_experts, {}, experts_per_rank, use_fp8=False)
    return MegatronFusedCombine.apply(local_output, buffer, handle)


def compare_tensor(name, baseline, actual, rank, max_abs_tol, calc_diff_tol, cos_tol):
    diff = calc_diff(baseline, actual)
    max_abs_diff = (baseline.float() - actual.float()).abs().max().item()
    cos_sim = F.cosine_similarity(
        baseline.float().flatten().unsqueeze(0),
        actual.float().flatten().unsqueeze(0),
    ).item()
    passed = max_abs_diff <= max_abs_tol and diff <= calc_diff_tol and cos_sim >= cos_tol

    if rank == 0:
        print(f'[Rank {rank}] === Megakernel {name} vs Baseline Precision Alignment ===', flush=True)
        print(f'  calc_diff (lower=better): {diff:.6e} (tol={calc_diff_tol:.1e})', flush=True)
        print(f'  max_abs_diff: {max_abs_diff:.6e} (tol={max_abs_tol:.1e})', flush=True)
        print(f'  cosine_similarity: {cos_sim:.6f} (tol={cos_tol:.6f})', flush=True)
        print(f'  baseline norm: {baseline.float().norm().item():.6e}', flush=True)
        print(f'  megakernel norm: {actual.float().norm().item():.6e}', flush=True)

    if not passed:
        raise AssertionError(
            f'{name} mismatch on rank {rank}: calc_diff={diff}, '
            f'max_abs_diff={max_abs_diff}, cosine_similarity={cos_sim}'
        )


def run_case(local_rank, num_local_ranks, rank, num_ranks, buffer, group, args, case, case_idx):
    if args.stage not in (1, 2):
        raise ValueError('megakernel debug backward currently supports stage 1 or 2')

    torch.manual_seed(42 + rank + case_idx * 1000003)
    num_nodes = max(1, num_ranks // num_local_ranks)
    num_experts = num_ranks * case.experts_per_rank
    router_num_groups = args.router_num_groups or num_nodes
    router_group_topk = args.router_group_topk or (
        num_nodes if case.num_topk_groups is None else case.num_topk_groups
    )

    x = torch.randn(
        case.num_tokens, case.hidden, dtype=torch.bfloat16, device='cuda'
    ) * 0.1
    topk_weights, topk_idx = make_megatron_router_inputs(
        case.num_tokens,
        num_experts,
        case.num_topk,
        router_num_groups,
        router_group_topk,
        args.router_score_function,
        'cuda',
    )

    torch.manual_seed(1000 + rank)
    W_gate = torch.randn(
        case.experts_per_rank, case.intermediate, case.hidden,
        dtype=torch.bfloat16, device='cuda') * 0.02
    W_up = torch.randn_like(W_gate) * 0.02
    W_down = torch.randn(
        case.experts_per_rank, case.hidden, case.intermediate,
        dtype=torch.bfloat16, device='cuda') * 0.02
    W_gateup = torch.empty(
        case.experts_per_rank, 2 * case.intermediate, case.hidden,
        dtype=torch.bfloat16, device='cuda')
    W_gateup[:, 0::2, :] = W_gate
    W_gateup[:, 1::2, :] = W_up
    W_gateup = W_gateup.contiguous()

    te_experts = build_te_grouped_experts(
        W_gate.detach(), W_up.detach(), W_down.detach(), case.experts_per_rank)
    for parameter in te_experts.parameters():
        parameter.requires_grad_(False)
    buffer.set_num_sms(args.baseline_sms)

    if rank == 0:
        print('', flush=True)
        print(f'=== Forward/backward case {case_idx + 1} ===', flush=True)
        print(
            f'  tokens={case.num_tokens}, hidden={case.hidden}, intermediate={case.intermediate}, '
            f'experts_per_rank={case.experts_per_rank}, topk={case.num_topk}, ranks={num_ranks}',
            flush=True,
        )

    # Megatron high-performance baseline. Only x requires grad so TE computes dgrad
    # without retaining parameter gradients that the current megakernel cannot return.
    baseline_x = x.detach().clone().requires_grad_(True)
    baseline_output = run_megatron_fused_baseline(
        baseline_x, topk_idx, topk_weights.detach(), num_experts,
        case.experts_per_rank, buffer, te_experts,
    )
    if not baseline_output.requires_grad:
        raise AssertionError('Megatron fused baseline output is not connected to autograd')
    
    torch.manual_seed(2000 + rank + case_idx * 1000003)
    grad_output = torch.randn_like(baseline_output)
    baseline_output.backward(grad_output)
    baseline_grad_x = baseline_x.grad.detach().clone()

    # Fused debug megakernel path connected through torch.autograd.Function.
    megakernel_x = x.detach().clone().requires_grad_(True)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    megakernel_output = buffer.megakernel_debug_autograd(
        megakernel_x, topk_idx, topk_weights.detach(), W_gateup.detach(), W_down.detach(),
        num_experts,
        num_dispatch_sms=args.megakernel_comm_sms,
        num_combine_sms=args.megakernel_comm_sms,
        total_sms=num_sms,
        stage=args.stage,
    )

    print("Megakernel forward complete.")

    megakernel_output.backward(grad_output)

    print("Megakernel backward complete.")

    megakernel_grad_x = megakernel_x.grad.detach()

    compare_tensor(
        'forward', baseline_output.detach(), megakernel_output.detach(), rank,
        args.forward_max_abs_tol, args.forward_calc_diff_tol, args.forward_cos_tol,
    )
    compare_tensor(
        'Backward dX', baseline_grad_x, megakernel_grad_x, rank,
        args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
    )

    if topk_weights.grad is not None or W_gateup.grad is not None or W_down.grad is not None:
        raise AssertionError('Current megakernel autograd contract must only produce dX')

    dist.barrier(group=group)
    torch.cuda.synchronize()


def run_worker(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    buffer = deep_ep.Buffer(
        group, int(2e9), int(1e9), low_latency_mode=False,
        num_qps_per_rank=num_sms, explicitly_destroy=True,
    )
    try:
        for case_idx, case in enumerate(TEST_CASES[:args.num_cases]):
            run_case(
                local_rank, num_local_ranks, rank, num_ranks,
                buffer, group, args, case, case_idx,
            )
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def run_mpirun(args):
    global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_world_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 8))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29501')
    os.environ['RANK'] = str(global_rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='nccl')
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(0)
    group = dist.new_group(list(range(world_size)))
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    buffer = deep_ep.Buffer(
        group, int(2e9), int(1e9), low_latency_mode=False,
        num_qps_per_rank=num_sms, explicitly_destroy=True,
    )
    try:
        for case_idx, case in enumerate(TEST_CASES[:args.num_cases]):
            run_case(
                local_rank, local_world_size, global_rank, world_size,
                buffer, group, args, case, case_idx,
            )
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare megakernel forward/dX with Megatron fused TE MoE')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-cases', type=int, default=1)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--baseline-sms', type=int, default=24)
    parser.add_argument('--megakernel-comm-sms', type=int, default=48)
    parser.add_argument(
        '--router-score-function', choices=['sigmoid', 'softmax', 'sqrtsoftplus'],
        default='sigmoid')
    parser.add_argument('--router-num-groups', type=int, default=0)
    parser.add_argument('--router-group-topk', type=int, default=0)
    parser.add_argument('--forward-max-abs-tol', type=float, default=1e-1)
    parser.add_argument('--forward-calc-diff-tol', type=float, default=1e-5)
    parser.add_argument('--forward-cos-tol', type=float, default=0.85)
    parser.add_argument('--backward-max-abs-tol', type=float, default=2e-1)
    parser.add_argument('--backward-calc-diff-tol', type=float, default=1e-4)
    parser.add_argument('--backward-cos-tol', type=float, default=0.80)
    parser.add_argument('--mpirun', action='store_true')
    args = parser.parse_args()

    if te_ops_unavailable():
        raise RuntimeError(
            'This test requires Transformer Engine ops for the Megatron high-performance baseline')
    if args.num_cases <= 0 or args.num_cases > len(TEST_CASES):
        raise ValueError(f'--num-cases must be in [1, {len(TEST_CASES)}]')
    return args


def te_ops_unavailable():
    try:
        import transformer_engine.pytorch.ops  # noqa: F401
        from transformer_engine.pytorch import moe_permute_with_probs, moe_unpermute  # noqa: F401
    except ImportError:
        return True
    return False


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.mpirun:
        run_mpirun(parsed_args)
    else:
        torch.multiprocessing.spawn(
            run_worker,
            args=(parsed_args.num_processes, parsed_args),
            nprocs=parsed_args.num_processes,
        )
