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

# Under mpirun each process must bind its own GPU BEFORE anything creates a CUDA
# context. Module-level imports below (torch, deep_ep, transformer_engine, and
# megakernel_test_utils which calls torch.cuda.is_available() at import) can
# initialize CUDA on physical device 0 for every local rank, which later makes
# NCCL raise "Duplicate GPU detected". Pin CUDA_VISIBLE_DEVICES from the MPI
# local-rank here, before those imports run.
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    os.environ.setdefault(
        'CUDA_VISIBLE_DEVICES', os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import calc_diff, init_dist

# The Megatron baseline path needs deep_ep + TE-backed helpers, but the standalone
# QuACK wgrad self-test (--quack-wgrad-selftest) does not. Keep these imports
# tolerant so the self-test can run in environments where the prebuilt deep_ep_cpp
# is ABI-incompatible with the active torch.
try:
    from megakernel_test_utils import (
        build_te_grouped_experts,
        make_megatron_router_inputs,
        moe_compute_on_recv_te,
    )
    import deep_ep
    _DEEPEP_IMPORT_ERROR = None
except Exception as _dep_exc:  # pragma: no cover - environment dependent
    build_te_grouped_experts = None
    make_megatron_router_inputs = None
    moe_compute_on_recv_te = None
    deep_ep = None
    _DEEPEP_IMPORT_ERROR = _dep_exc

# QuACK import is deferred: importing quack.gemm_interface pulls in CUTLASS DSL /
# cuda-python and initializes a CUDA context at import time. Under mpirun that would
# bind every local rank to physical device 0 before run_mpirun sets
# CUDA_VISIBLE_DEVICES, causing NCCL "Duplicate GPU detected". Load it lazily instead.
_quack_gemm = None
_QUACK_IMPORT_ERROR = None
_QUACK_LOADED = False


def _load_quack_gemm():
    global _quack_gemm, _QUACK_IMPORT_ERROR, _QUACK_LOADED
    if not _QUACK_LOADED:
        _QUACK_LOADED = True
        try:
            from quack.gemm_interface import gemm as gemm_fn
            _quack_gemm = gemm_fn
        except Exception as exc:  # pragma: no cover - environment dependent
            _quack_gemm = None
            _QUACK_IMPORT_ERROR = exc
    return _quack_gemm


def quack_available():
    return _load_quack_gemm() is not None



def quack_grouped_wgrad(A_src, B_packed, cu_seqlens_k, A_idx, *, out=None):
    """Grouped/varlen-K weight-gradient GEMM via QuACK.

    Computes, per expert e:
        out[e] = A_src[:, A_idx[cu[e]:cu[e+1]]] @ B_packed[cu[e]:cu[e+1], :]

    QuACK varlen-K layout constraints (verified on SM100):
      - A_src must be m-major   (A_src.stride(-2) == 1), e.g. a ``(M, total_K)``
        transpose view of a ``(total_K, M)`` contiguous tensor.
      - B_packed must be n-major (B_packed.stride(-1) == 1), i.e. a contiguous
        ``(total_K, N)`` buffer packed in expert order.
      - cu_seqlens_k is the int32 ``[E + 1]`` prefix sum of per-expert token counts.
      - A_idx is the int32 ``[total_K]`` gather index mapping packed B rows to
        A_src columns (identity when A_src is already packed in the same order).

    For dW_gateup: A_src = dGU^T (2I, total_K), B_packed = X   (total_K, H) -> (E, 2I, H)
    For dW_down  : A_src = dZ^T  (H,  total_K), B_packed = Act (total_K, I) -> (E, H,  I)
    """
    quack_gemm = _load_quack_gemm()
    if quack_gemm is None:
        raise RuntimeError(f'QuACK is not importable: {_QUACK_IMPORT_ERROR!r}')
    assert A_src.stride(-2) == 1, 'QuACK varlen-K requires A to be m-major (stride(-2)==1)'
    assert B_packed.stride(-1) == 1, 'QuACK varlen-K requires B to be n-major (stride(-1)==1)'
    return quack_gemm(
        A_src, B_packed,
        out=out,
        cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        A_idx=A_idx.to(torch.int32),
        tuned=False,
    )


def quack_wgrad_selftest(seed=0):
    """Standalone precision check of the QuACK grouped-wgrad path.

    Mirrors the megakernel dW operands (interleaved gate/up dGU, per-expert
    variable token counts incl. empty/tail experts) and validates both dW GEMMs
    against a float32 torch reference. Requires no megakernel / distributed run.
    """
    if not quack_available():
        raise RuntimeError(f'QuACK is not importable: {_QUACK_IMPORT_ERROR!r}')
    torch.manual_seed(seed)
    device = torch.device('cuda')
    E, hidden, intermediate = 4, 128, 96
    two_i = 2 * intermediate
    counts = [40, 0, 24, 8]
    total = sum(counts)
    cu = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()],
                      dtype=torch.int32, device=device)
    A_idx = torch.arange(total, dtype=torch.int32, device=device)

    X = 0.1 * torch.randn(total, hidden, dtype=torch.bfloat16, device=device)
    dGU = 0.1 * torch.randn(total, two_i, dtype=torch.bfloat16, device=device)
    dZ = 0.1 * torch.randn(total, hidden, dtype=torch.bfloat16, device=device)
    Act = 0.1 * torch.randn(total, intermediate, dtype=torch.bfloat16, device=device)

    dw_gateup = quack_grouped_wgrad(dGU.transpose(0, 1), X, cu, A_idx)
    dw_down = quack_grouped_wgrad(dZ.transpose(0, 1), Act, cu, A_idx)
    torch.cuda.synchronize()

    ref_gateup = torch.zeros(E, two_i, hidden, dtype=torch.bfloat16, device=device)
    ref_down = torch.zeros(E, hidden, intermediate, dtype=torch.bfloat16, device=device)
    for expert in range(E):
        start, end = int(cu[expert]), int(cu[expert + 1])
        if end > start:
            ref_gateup[expert] = (dGU[start:end].float().T @ X[start:end].float()).to(torch.bfloat16)
            ref_down[expert] = (dZ[start:end].float().T @ Act[start:end].float()).to(torch.bfloat16)

    gateup_diff = (dw_gateup.float() - ref_gateup.float()).abs().max().item()
    down_diff = (dw_down.float() - ref_down.float()).abs().max().item()
    print(f'[quack-wgrad-selftest] dW_gateup {tuple(dw_gateup.shape)} max_abs_diff={gateup_diff:.3e}',
          flush=True)
    print(f'[quack-wgrad-selftest] dW_down   {tuple(dw_down.shape)} max_abs_diff={down_diff:.3e}',
          flush=True)
    assert gateup_diff == 0.0 and down_diff == 0.0, (
        f'QuACK grouped wgrad mismatch: gateup={gateup_diff}, down={down_diff}')
    print('[quack-wgrad-selftest] PASS', flush=True)

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
    test(num_tokens=4096, hidden=2048, intermediate=2048, experts_per_rank=16, num_topk=8),
    # test(num_tokens=8192, hidden=4096, intermediate=4096, experts_per_rank=16, num_topk=8),
    # test(num_tokens=8192, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6),
    # Add more cases here, for example:
    # test(num_tokens=8192, hidden=256, intermediate=256, experts_per_rank=8, num_topk=2),
]


TRACE_RE = re.compile(
    r"(mk|deepep)_perf_trace_rank(\d+)(?:_(forward|backward|dispatch|combine|notify))?(?:_iter(\d+))?\.json$"
)


def parse_trace_name(path):
    match = TRACE_RE.search(path.name)
    if not match:
        return None
    source = match.group(1)
    rank = int(match.group(2))
    phase = match.group(3) or source
    iteration = int(match.group(4)) if match.group(4) is not None else -1
    return source, rank, phase, iteration


def canonical_trace_name(source, rank, phase):
    phase_suffix = '' if phase == source else f'_{phase}'
    return f'{source}_perf_trace_rank{rank}{phase_suffix}.json'


def keep_last_perf_trace_iter(trace_dir):
    trace_dir = Path(trace_dir)
    files_by_group = {}
    for path in trace_dir.glob('*perf_trace_rank*.json'):
        parsed = parse_trace_name(path)
        if parsed is None:
            continue
        source, rank, phase, iteration = parsed
        files_by_group.setdefault((source, rank, phase), []).append((iteration, path))

    kept = 0
    removed = 0
    for (source, rank, phase), entries in files_by_group.items():
        tagged_entries = [(iteration, path) for iteration, path in entries if iteration >= 0]
        canonical_path = trace_dir / canonical_trace_name(source, rank, phase)
        canonical_exists = any(path == canonical_path for _iteration, path in entries)

        if source == 'mk' and canonical_exists:
            for _iteration, path in tagged_entries:
                path.unlink(missing_ok=True)
                removed += 1
            kept += 1
            continue

        if not tagged_entries:
            continue
        latest_iter, latest_path = max(tagged_entries, key=lambda item: item[0])

        for iteration, path in entries:
            if path == latest_path:
                continue
            path.unlink(missing_ok=True)
            removed += 1

        if latest_path != canonical_path:
            canonical_path.unlink(missing_ok=True)
            latest_path.rename(canonical_path)
        kept += 1
    return kept, removed


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
    def backward(ctx, grad_x, _grad_indices, grad_probs, _grad_tokens_per_expert, _grad_handle):
        combined_x, combined_probs, _ = ctx.buffer.combine(
            grad_x.contiguous(), ctx.handle,
            topk_weights=None if grad_probs is None else grad_probs.float(),
        )
        return combined_x, None, combined_probs, None, None


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
    baseline = baseline.to(device=actual.device)
    diff = calc_diff(baseline, actual)
    max_abs_diff = (baseline.float() - actual.float()).abs().max().item()
    cos_sim = F.cosine_similarity(
        baseline.float().flatten().unsqueeze(0),
        actual.float().flatten().unsqueeze(0),
    ).item()
    passed = max_abs_diff <= max_abs_tol and diff <= calc_diff_tol and cos_sim >= cos_tol

    # if rank == 0:
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


def clear_parameter_grads(module):
    for parameter in module.parameters():
        parameter.grad = None


def start_memory_measurement():
    """Synchronize and start a fresh CUDA allocator peak measurement."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def record_forward_memory(start_allocated):
    """Measure allocations retained by the forward graph before backward starts."""
    torch.cuda.synchronize()
    return max(0, torch.cuda.memory_allocated() - start_allocated)


def finish_memory_measurement(start_allocated, activation_retained, phase):
    """Report peak allocator usage for one complete forward/backward pair."""
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    print(
        f'  [{phase} memory] forward activation retained: '
        f'{activation_retained / 1024 ** 2:.2f} MiB, '
        f'peak allocated (fwd+bwd): {peak_allocated / 1024 ** 2:.2f} MiB, '
        f'peak reserved: {peak_reserved / 1024 ** 2:.2f} MiB, '
        f'allocated before forward: {start_allocated / 1024 ** 2:.2f} MiB',
        flush=True,
    )
    return activation_retained, peak_allocated, peak_reserved


def report_memory_comparison(baseline_memory, megakernel_memory):
    """Print megakernel minus DeepEP+TE memory for the same test case."""
    names = ('forward activation retained', 'peak allocated (fwd+bwd)', 'peak reserved')
    print('  [Memory comparison] Megakernel - DeepEP + TE:', flush=True)
    for name, baseline_value, megakernel_value in zip(
        names, baseline_memory, megakernel_memory):
        delta = megakernel_value - baseline_value
        ratio = 100.0 * delta / baseline_value if baseline_value else float('nan')
        print(
            f'    {name}: {delta / 1024 ** 2:+.2f} MiB ({ratio:+.2f}%)',
            flush=True,
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
        parameter.requires_grad_(True)
    buffer.set_num_sms(args.baseline_sms)

    if rank == 0:
        print('', flush=True)
        print(f'=== Forward/backward case {case_idx + 1} ===', flush=True)
        print(
            f'  tokens={case.num_tokens}, hidden={case.hidden}, intermediate={case.intermediate}, '
            f'experts_per_rank={case.experts_per_rank}, topk={case.num_topk}, ranks={num_ranks}',
            flush=True,
        )

    torch.manual_seed(2000 + rank + case_idx * 1000003)
    grad_output = torch.randn_like(x)

    # Megatron high-performance baseline with the complete training gradients used by
    # the real fused MoE path: dX, expert weight gradients, and route-prob gradients.
    for w in range(args.warmup):
        if local_rank == 0:
            print(f'[Rank {rank}] Baseline warmup {w + 1}/{args.warmup}', flush=True)
        clear_parameter_grads(te_experts)
        warmup_x = x.detach().clone().requires_grad_(True)
        warmup_topk_weights = topk_weights.detach().clone().requires_grad_(True)
        warmup_output = run_megatron_fused_baseline(
            warmup_x, topk_idx, warmup_topk_weights, num_experts,
            case.experts_per_rank, buffer, te_experts,
        )
        if not warmup_output.requires_grad:
            raise AssertionError('Megatron fused baseline output is not connected to autograd')
        warmup_output.backward(grad_output)
    if args.warmup > 0:
        clear_parameter_grads(te_experts)
        dist.barrier(group=group)
        torch.cuda.synchronize()

    clear_parameter_grads(te_experts)
    dist.barrier(group=group)
    baseline_mem_start = start_memory_measurement()
    baseline_x = x.detach().clone().requires_grad_(True)
    baseline_topk_weights = topk_weights.detach().clone().requires_grad_(True)
    baseline_output = run_megatron_fused_baseline(
        baseline_x, topk_idx, baseline_topk_weights, num_experts,
        case.experts_per_rank, buffer, te_experts,
    )
    if not baseline_output.requires_grad:
        raise AssertionError('Megatron fused baseline output is not connected to autograd')
    baseline_activation_retained = record_forward_memory(baseline_mem_start)
    baseline_output.backward(grad_output)
    baseline_memory = finish_memory_measurement(
        baseline_mem_start, baseline_activation_retained, 'DeepEP + TE')
    baseline_grad_x = baseline_x.grad.detach()
    baseline_grad_topk_weights = baseline_topk_weights.grad.detach()

    expert_weight_grads = [
        parameter.grad.detach()
        for parameter in te_experts.parameters()
        if parameter.grad is not None
    ]
    if len(expert_weight_grads) != 2:
        raise AssertionError(
            f'Expected TE FC1/FC2 weight gradients, got {len(expert_weight_grads)} tensors')
    baseline_fc1_grad, baseline_grad_w_down = expert_weight_grads
    expected_fc1_shape = (case.experts_per_rank, 2 * case.intermediate, case.hidden)
    expected_fc2_shape = (case.experts_per_rank, case.hidden, case.intermediate)
    if tuple(baseline_fc1_grad.shape) != expected_fc1_shape:
        raise AssertionError(
            f'Unexpected TE FC1 grad shape {tuple(baseline_fc1_grad.shape)}, '
            f'expected {expected_fc1_shape}')
    if tuple(baseline_grad_w_down.shape) != expected_fc2_shape:
        raise AssertionError(
            f'Unexpected TE FC2 grad shape {tuple(baseline_grad_w_down.shape)}, '
            f'expected {expected_fc2_shape}')

    baseline_grad_w_gate = torch.empty_like(W_gate)
    baseline_grad_w_up = torch.empty_like(W_up)
    chunk_base = 0
    for start in range(0, case.intermediate, 32):
        rows = min(32, case.intermediate - start)
        baseline_grad_w_gate[:, start:start + rows, :] = baseline_fc1_grad[
            :, chunk_base:chunk_base + rows, :]
        baseline_grad_w_up[:, start:start + rows, :] = baseline_fc1_grad[
            :, chunk_base + rows:chunk_base + 2 * rows, :]
        chunk_base += 2 * rows
    baseline_grad_w_gateup = torch.empty_like(W_gateup)
    baseline_grad_w_gateup[:, 0::2, :] = baseline_grad_w_gate
    baseline_grad_w_gateup[:, 1::2, :] = baseline_grad_w_up
    baseline_grad_w_gateup = baseline_grad_w_gateup.contiguous().cpu()
    baseline_output_reference = baseline_output.detach().cpu()
    baseline_grad_x = baseline_grad_x.cpu()
    baseline_grad_topk_weights = baseline_grad_topk_weights.cpu()
    baseline_grad_w_down = baseline_grad_w_down.cpu()
    clear_parameter_grads(te_experts)
    del baseline_x, baseline_topk_weights, baseline_output
    torch.cuda.empty_cache()

    # Fused debug megakernel path connected through torch.autograd.Function.
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    for w in range(args.warmup):
        if local_rank == 0:
            print(f'[Rank {rank}] MegaKernel warmup {w + 1}/{args.warmup}', flush=True)
        warmup_mk_x = x.detach().clone().requires_grad_(True)
        warmup_mk_topk_weights = topk_weights.detach().clone().requires_grad_(True)
        warmup_mk_w_gateup = W_gateup.detach().clone().requires_grad_(True)
        warmup_mk_w_down = W_down.detach().clone().requires_grad_(True)
        warmup_mk_output = buffer.megakernel_debug_autograd(
            warmup_mk_x, topk_idx, warmup_mk_topk_weights,
            warmup_mk_w_gateup, warmup_mk_w_down, num_experts,
            num_dispatch_sms=args.megakernel_comm_sms,
            num_combine_sms=args.megakernel_comm_sms,
            total_sms=num_sms,
            stage=args.stage,
        )
        warmup_mk_output.backward(grad_output)
    if args.warmup > 0:
        dist.barrier(group=group)
        torch.cuda.synchronize()

    megakernel_x = x.detach().clone().requires_grad_(True)
    megakernel_topk_weights = topk_weights.detach().clone().requires_grad_(True)
    megakernel_w_gateup = W_gateup.detach().clone().requires_grad_(True)
    megakernel_w_down = W_down.detach().clone().requires_grad_(True)
    # Align ranks before resetting the allocator peak so the window covers one
    # complete distributed forward/backward execution on every rank.
    dist.barrier(group=group)
    megakernel_mem_start = start_memory_measurement()
    megakernel_output = buffer.megakernel_debug_autograd(
        megakernel_x, topk_idx, megakernel_topk_weights, megakernel_w_gateup, megakernel_w_down,
        num_experts,
        num_dispatch_sms=args.megakernel_comm_sms,
        num_combine_sms=args.megakernel_comm_sms,
        total_sms=num_sms,
        stage=args.stage,
    )

    megakernel_activation_retained = record_forward_memory(megakernel_mem_start)
    print("Megakernel forward complete.")

    megakernel_output.backward(grad_output)

    megakernel_memory = finish_memory_measurement(
        megakernel_mem_start, megakernel_activation_retained, 'Megakernel')
    report_memory_comparison(baseline_memory, megakernel_memory)
    print("Megakernel backward complete.")

    megakernel_grad_x = megakernel_x.grad.detach()
    if megakernel_w_gateup.grad is None or megakernel_w_down.grad is None:
        raise AssertionError('Megakernel backward did not return expert weight gradients')
    if megakernel_topk_weights.grad is None:
        raise AssertionError('Megakernel backward did not return dTopKWeights')

    compare_tensor(
        'forward', baseline_output_reference, megakernel_output.detach(), rank,
        args.forward_max_abs_tol, args.forward_calc_diff_tol, args.forward_cos_tol,
    )
    compare_tensor(
        'Backward dX', baseline_grad_x, megakernel_grad_x, rank,
        args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
    )
    compare_tensor(
        'Backward dW_gateup', baseline_grad_w_gateup, megakernel_w_gateup.grad.detach(), rank,
        args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
    )
    compare_tensor(
        'Backward dW_down', baseline_grad_w_down, megakernel_w_down.grad.detach(), rank,
        args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
    )
    compare_tensor(
        'Backward dTopKWeights', baseline_grad_topk_weights,
        megakernel_topk_weights.grad.detach(), rank,
        args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
    )

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
            if args.perf_trace_retention == 'last':
                if rank == 0:
                    kept, removed = keep_last_perf_trace_iter(args.perf_trace_dir)
                    print(
                        f'[perf-trace] retention=last kept={kept} removed={removed} dir={args.perf_trace_dir}',
                        flush=True,
                    )
                dist.barrier(group=group)
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def run_mpirun(args):
    global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_world_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 8))

    # CUDA_VISIBLE_DEVICES is already pinned to this local rank at module import
    # (see top of file) so the CUDA context binds to the correct physical GPU.
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
            if args.perf_trace_retention == 'last':
                if global_rank == 0:
                    kept, removed = keep_last_perf_trace_iter(args.perf_trace_dir)
                    print(
                        f'[perf-trace] retention=last kept={kept} removed={removed} dir={args.perf_trace_dir}',
                        flush=True,
                    )
                dist.barrier(group=group)
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare megakernel forward/dX with Megatron fused TE MoE')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-cases', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=20,
                        help='Number of warmup iterations for both baseline and megakernel before the measured run')
    parser.add_argument('--perf-trace-retention', choices=['last', 'all'], default='last',
                        help='With MK_PERF_TRACE, keep only the last iter traces by default; use all to keep every iter')
    parser.add_argument('--perf-trace-dir', default='.',
                        help='Directory where *perf_trace_rank*.json files are emitted')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--baseline-sms', type=int, default=48)
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
    parser.add_argument('--quack-wgrad-selftest', action='store_true',
                        help='Run the standalone QuACK grouped-wgrad precision check and exit '
                             '(no megakernel / distributed run)')
    args = parser.parse_args()

    args.perf_trace_dir = os.path.abspath(args.perf_trace_dir)
    os.environ['MK_PERF_TRACE_RETENTION'] = args.perf_trace_retention

    if args.quack_wgrad_selftest:
        # Standalone QuACK precision check does not need the TE baseline.
        return args
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
    if parsed_args.quack_wgrad_selftest:
        quack_wgrad_selftest()
    elif parsed_args.mpirun:
        run_mpirun(parsed_args)
    else:
        torch.multiprocessing.spawn(
            run_worker,
            args=(parsed_args.num_processes, parsed_args),
            nprocs=parsed_args.num_processes,
        )
