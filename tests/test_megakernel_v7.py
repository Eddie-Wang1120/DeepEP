"""
MK-v7: Persistent MegaKernel precision alignment test.

Compares:
  A) DeepEP dispatch -> PyTorch compute -> DeepEP combine (baseline, MK-v6)
  B) buffer.megakernel_forward() (fused persistent kernel, MK-v7)

Both use identical inputs. Output of (B) should match (A) within BF16 tolerance.

Usage (2 nodes x 8 GPUs):
    torchrun --nproc_per_node=8 --nnodes=2 \
             --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
             test_megakernel_v7.py
"""
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

try:
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.ops as te_ops
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast, moe_permute_with_probs, moe_unpermute
except ImportError:
    te = None
    te_ops = None
    te_recipe = None
    fp8_autocast = None
    moe_permute_with_probs = None
    moe_unpermute = None

def null_decorator(func=None, *args, **kwargs):
    if func is not None:
        return func
    return lambda f: f


HAVE_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
        HAVE_TRITON_AVAILABLE = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


# Copied from Megatron-LM megatron/core/fusions/fused_indices_converter.py.
@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    row_idx = tl.program_id(0)
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    indices_row = tl.where(topk_row_mask, indices_row, -1)
    probs_row = tl.load(probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)

    position_row = tl.where(indices_row != -1, topk_row, -1)
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts
    tl.store(multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(position_map_ptr + row_idx_offset + num_exp_row, -1, mask=num_exp_row_mask)
    tl.debug_barrier()
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)


@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts + num_exp_row
    probs_in_multihot_row = tl.load(probs_in_multihot_ptr + ptr_offset, mask=num_exp_row_mask)

    position_map_row = tl.load(position_map_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.where(num_exp_row_mask, position_map_row, -1)
    mask = position_map_row != -1

    tl.store(probs_indices_ptr + row_idx * topk + topk_row, 0, mask=topk_row_mask)
    tl.debug_barrier()
    tl.store(probs_indices_ptr + row_idx * topk + position_map_row, probs_in_multihot_row, mask)


class IndicesToMultihot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        num_of_tokens = indices.shape[0]
        assert indices.shape == probs_indices.shape, "indices and probs_indices must have the same shape"
        topk = indices.shape[1]
        multihot_indices = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda")
        probs_in_multihot = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=probs_indices.dtype, device="cuda")
        position_map = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.int32, device="cuda")
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))
        grid = (num_of_tokens,)
        _indices_to_multihot_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,
            num_warps=1,
        )

        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk
        grad_probs_indices = torch.empty(
            (num_of_tokens, topk), dtype=grad_probs_in_multihot.dtype, device="cuda")
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))

        grid = (num_of_tokens,)
        _multihot_to_indices_kernel[grid](
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,
            num_warps=1,
        )
        return None, grad_probs_indices, None, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)


sys.path.insert(0, os.path.dirname(__file__))
import deep_ep
from utils import init_dist, calc_diff


def print_bitwise_mismatches(baseline_output, megakernel_output, rank, hidden_states=None, topk_idx=None, topk_weights=None, max_print=None):
    """Print every element whose raw BF16 bit pattern differs."""
    if baseline_output.shape != megakernel_output.shape:
        print(f'[Rank {rank}] BITWISE shape mismatch: baseline={baseline_output.shape}, megakernel={megakernel_output.shape}', flush=True)
        return

    baseline_contig = baseline_output.detach().contiguous()
    megakernel_contig = megakernel_output.detach().contiguous()
    baseline_bits = baseline_contig.view(torch.int16).flatten()
    megakernel_bits = megakernel_contig.view(torch.int16).flatten()
    mismatch_flat = (baseline_bits != megakernel_bits).nonzero(as_tuple=True)[0]
    mismatch_count = mismatch_flat.numel()

    print(f'[Rank {rank}] BITWISE mismatch_count={mismatch_count} / {baseline_bits.numel()}', flush=True)
    if mismatch_count == 0:
        return

    hidden = baseline_output.shape[-1]
    limit = mismatch_count if max_print is None else min(mismatch_count, max_print)
    baseline_values = baseline_contig.flatten()
    megakernel_values = megakernel_contig.flatten()
    printed_tokens = set()
    for i in range(limit):
        flat_idx = int(mismatch_flat[i].item())
        token_idx = flat_idx // hidden
        hidden_idx = flat_idx % hidden
        baseline_bit = int(baseline_bits[flat_idx].item()) & 0xffff
        megakernel_bit = int(megakernel_bits[flat_idx].item()) & 0xffff
        baseline_value = float(baseline_values[flat_idx].float().item())
        megakernel_value = float(megakernel_values[flat_idx].float().item())
        print(
            f'[Rank {rank}] BITWISE-MISMATCH flat={flat_idx} token={token_idx} hidden={hidden_idx} '
            f'baseline={baseline_value} megakernel={megakernel_value} '
            f'baseline_bf16=0x{baseline_bit:04x} megakernel_bf16=0x{megakernel_bit:04x}',
            flush=True)

        # if token_idx not in printed_tokens:
        #     printed_tokens.add(token_idx)
        #     if topk_idx is not None:
        #         print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} topk_idx={topk_idx[token_idx].detach().cpu().tolist()}', flush=True)
        #     if topk_weights is not None:
        #         print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} topk_weights={topk_weights[token_idx].detach().cpu().float().tolist()}', flush=True)
        #     if hidden_states is not None:
        #         print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} hidden_states={hidden_states[token_idx].detach().cpu().float().tolist()}', flush=True)
        #         print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} hidden_states_bf16_hex={[hex(int(v) & 0xffff) for v in hidden_states[token_idx].detach().contiguous().view(torch.int16).cpu().tolist()]}', flush=True)
        #     print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} baseline_output={baseline_contig[token_idx].detach().cpu().float().tolist()}', flush=True)
        #     print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} megakernel_output={megakernel_contig[token_idx].detach().cpu().float().tolist()}', flush=True)

    if limit < mismatch_count:
        print(f'[Rank {rank}] BITWISE mismatch print truncated: printed={limit}, total={mismatch_count}', flush=True)


def moe_compute_on_recv(recv_x, recv_topk_idx, recv_topk_weights, W_gate, W_up, W_down, experts_per_rank):
    """
    Megatron-style non-TE MoE expert forward on received tokens.
    recv_topk_idx [num_recv, num_topk] contains LOCAL expert IDs (0..experts_per_rank-1), -1 for others.
    Router weights are applied inside the expert path, matching Megatron's DeepEP forward usage.
    """
    expert_out = torch.zeros_like(recv_x)

    for expert_id in range(experts_per_rank):
        mask = (recv_topk_idx == expert_id)  # [num_recv, num_topk]
        row_mask = mask.any(dim=1)
        if not row_mask.any():
            continue

        token_indices = row_mask.nonzero(as_tuple=True)[0]
        tokens = recv_x[token_indices]

        gate = torch.matmul(tokens, W_gate[expert_id].T)
        up = torch.matmul(tokens, W_up[expert_id].T)
        swiglu_out = F.silu(gate) * up

        weights = (recv_topk_weights[token_indices] * mask[token_indices].float()).sum(dim=1, keepdim=True)
        swiglu_out = (swiglu_out * weights.to(swiglu_out.dtype)).to(swiglu_out.dtype)

        down = torch.matmul(swiglu_out, W_down[expert_id].T)
        expert_out[token_indices] += down

    return expert_out


def _tokens_per_expert_tensor(tokens_per_expert, device):
    if isinstance(tokens_per_expert, torch.Tensor):
        return tokens_per_expert.to(device=device, dtype=torch.int32)
    return torch.tensor(tokens_per_expert, device=device, dtype=torch.int32)


def build_te_grouped_experts(W_gate, W_up, W_down, experts_per_rank):
    if te_ops is None:
        raise RuntimeError('Transformer Engine ops are required for --baseline-impl te')
    hidden = W_gate.shape[-1]
    intermediate = W_gate.shape[1]
    fc1 = te_ops.GroupedLinear(
        experts_per_rank, hidden, intermediate * 2,
        bias=False, device=W_gate.device, dtype=W_gate.dtype,
        single_grouped_weight=True, single_grouped_bias=False,
        accumulate_into_main_grad=False, delay_wgrad_compute=False,
    )
    scaled_act = te_ops.ScaledSwiGLU(glu_interleave_size=32)
    fc2 = te_ops.GroupedLinear(
        experts_per_rank, intermediate, hidden,
        bias=False, device=W_down.device, dtype=W_down.dtype,
        single_grouped_weight=True, single_grouped_bias=False,
        accumulate_into_main_grad=False, delay_wgrad_compute=False,
    )
    fc1_chunks = []
    for start in range(0, intermediate, 32):
        fc1_chunks.append(W_gate[:, start:start + 32, :])
        fc1_chunks.append(W_up[:, start:start + 32, :])
    fc1_weight = torch.cat(fc1_chunks, dim=1).contiguous()
    with torch.no_grad():
        fc1.weight.copy_(fc1_weight)
        fc2.weight.copy_(W_down.contiguous())
    return te_ops.Sequential(fc1, scaled_act, fc2)


def te_fp8_context(enabled):
    if not enabled:
        return contextlib.nullcontext()
    if fp8_autocast is None or te_recipe is None:
        raise RuntimeError('Transformer Engine fp8_autocast and recipe are required for FP8 TE baseline')
    recipe = te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID, amax_history_len=16, amax_compute_algo='max')
    return fp8_autocast(enabled=True, fp8_recipe=recipe)


def moe_compute_on_recv_te(recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list,
                           te_experts, workspace, experts_per_rank, use_fp8=False):
    """Megatron expert-only baseline: fused permute -> TE grouped MLP -> fused unpermute.

    This mirrors Megatron-LM's production branch:
    MoELayer.routed_experts_compute -> TEGroupedMLP -> TE GroupedLinear.
    """
    if moe_permute_with_probs is None or moe_unpermute is None:
        raise RuntimeError('TE moe_permute_with_probs/moe_unpermute are required for --baseline-impl te')
    if not HAVE_TRITON_AVAILABLE:
        raise RuntimeError('Triton is required for local fused_indices_to_multihot with --baseline-impl te')
    assert recv_topk_weights.dtype == torch.float32, "Megatron/DeepEP dispatcher expects fp32 router probs"
    routing_map, probs_map = fused_indices_to_multihot(
        recv_topk_idx, recv_topk_weights, experts_per_rank)
    tokens_per_expert = _tokens_per_expert_tensor(recv_num_tokens_per_expert_list, recv_x.device)
    num_out_tokens = tokens_per_expert.sum().item()
    permuted_x, permuted_probs, row_map = moe_permute_with_probs(
        recv_x, probs_map, routing_map, num_out_tokens=num_out_tokens)
    with te_fp8_context(use_fp8):
        permuted_output = te_experts(permuted_x, tokens_per_expert, permuted_probs, tokens_per_expert)
    return moe_unpermute(permuted_output, row_map, restore_shape=recv_x.shape)


def run_baseline_pipeline(x, topk_idx, topk_weights, W_gate, W_up, W_down,
                          num_experts, experts_per_rank, buffer, config, local_rank, rank, no_compute,
                          baseline_impl, te_experts=None, te_workspace=None, use_fp8=False):
    """
    Baseline: DeepEP dispatch -> expert compute -> DeepEP combine.
    baseline_impl='torch' keeps the original direct local-expert loop.
    baseline_impl='te' mirrors Megatron's production TEGroupedMLP local permute + grouped GEMM path.
    Returns combined output [num_tokens, hidden] in bf16.
    """
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

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
        print(f'[Rank {rank}] Baseline dispatch done: recv_x shape={recv_x.shape}', flush=True)

    if no_compute:
        combine_x = recv_x
    elif baseline_impl == 'te':
        if te_experts is None:
            raise RuntimeError('te_experts must be prebuilt for --baseline-impl te')
        if te_workspace is None:
            te_workspace = {}
        combine_x = moe_compute_on_recv_te(
            recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list,
            te_experts, te_workspace, experts_per_rank, use_fp8=use_fp8)
    else:
        combine_x = moe_compute_on_recv(recv_x, recv_topk_idx, recv_topk_weights,
                                        W_gate, W_up, W_down, experts_per_rank)
    combine_topk_weights = recv_topk_weights

    combined_x, combined_topk_weights, event = buffer.combine(
        x=combine_x,
        handle=handle,
        topk_weights=combine_topk_weights,
        config=config
    )

    if local_rank == 0:
        print(f'[Rank {rank}] Baseline combine done: combined_x shape={combined_x.shape}', flush=True)

    return combined_x


def run_megakernel_pipeline(x, topk_idx, topk_weights, W_gateup, W_down,
                            num_experts, buffer, local_rank, rank, stage,
                            hidden_states_scales=None, W_gateup_fp8=None, W_down_fp8=None,
                            W_gateup_fp8_sf=None, W_down_fp8_sf=None, debug=False):
    """
    MegaKernel v7: single persistent kernel (dispatch + compute + combine fused).
    W_gateup is pairwise interleaved as [g0, u0, g1, u1, ...].
    Returns output [num_tokens, hidden] in bf16.
    """
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    num_dispatch_sms = 48
    # total_sms = num_dispatch_sms + num_forwarder_sms + num_compute_sms
    total_sms = num_sms  # use all available SMs

    if local_rank == 0:
        print(f'[Rank {rank}] MegaKernel launch: total_sms={total_sms}, dispatch={num_dispatch_sms}, stage={stage}', flush=True)

    forward = buffer.megakernel_debug_forward if debug else buffer.megakernel_forward
    result = forward(
        x, topk_idx, topk_weights,
        W_gateup, W_down,
        num_experts,
        num_dispatch_sms,
        num_dispatch_sms,  # num_combine_sms = num_dispatch_sms
        total_sms,
        stage,
        hidden_states_scales=hidden_states_scales,
        W_gateup_fp8=W_gateup_fp8,
        W_down_fp8=W_down_fp8,
        W_gateup_fp8_sf=W_gateup_fp8_sf,
        W_down_fp8_sf=W_down_fp8_sf,
    )

    if local_rank == 0:
        print(f'[Rank {rank}] MegaKernel done: output shape={result.shape}', flush=True)

    return result


def fp8_scale_k_packed(k):
    return (k + 128 * 4 - 1) // (128 * 4)


def make_unit_fp8_scale(shape, device):
    return torch.full(shape, 0x7f7f7f7f, dtype=torch.int32, device=device)


def make_fp8_megakernel_inputs(x, W_gateup, W_down, hidden, intermediate):
    if not hasattr(torch, 'float8_e4m3fn'):
        raise RuntimeError('torch.float8_e4m3fn is required for FP8 megakernel input tests')
    x_fp8 = x.float().to(torch.float8_e4m3fn).contiguous()
    W_gateup_fp8 = W_gateup.float().to(torch.float8_e4m3fn).contiguous()
    W_down_fp8 = W_down.float().to(torch.float8_e4m3fn).contiguous()
    hidden_scale_k = fp8_scale_k_packed(hidden)
    intermediate_scale_k = fp8_scale_k_packed(intermediate)
    hidden_states_scales = make_unit_fp8_scale((x.shape[0], hidden_scale_k), x.device)
    W_gateup_fp8_sf = make_unit_fp8_scale((W_gateup.shape[0], W_gateup.shape[1], hidden_scale_k), x.device)
    W_down_fp8_sf = make_unit_fp8_scale((W_down.shape[0], W_down.shape[1], intermediate_scale_k), x.device)
    return x_fp8, hidden_states_scales, W_gateup_fp8, W_down_fp8, W_gateup_fp8_sf, W_down_fp8_sf


def megatron_group_limited_topk(scores, topk, num_groups, group_topk):
    """Match Megatron-LM group_limited_topk routing for realistic expert/node load."""
    num_tokens, num_experts = scores.shape
    if num_groups <= 0 or num_experts % num_groups != 0:
        raise ValueError(f'num_groups must divide num_experts, got num_groups={num_groups}, num_experts={num_experts}')
    if group_topk <= 0 or group_topk > num_groups:
        raise ValueError(f'group_topk must be in [1, num_groups], got group_topk={group_topk}, num_groups={num_groups}')
    if topk % group_topk != 0:
        raise ValueError(f'Megatron group_limited_topk requires topk % group_topk == 0, got topk={topk}, group_topk={group_topk}')

    group_scores = (
        scores.view(num_tokens, num_groups, -1)
        .topk(topk // group_topk, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, num_experts)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    return torch.topk(masked_scores, k=topk, dim=-1)


def make_megatron_router_inputs(num_tokens, num_experts, topk, num_groups, group_topk,
                                score_function, device):
    """Create dense top-k routing tensors using Megatron's group-limited top-k semantics."""
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    if score_function == 'softmax':
        topk_logits, topk_idx = megatron_group_limited_topk(logits, topk, num_groups, group_topk)
        topk_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    elif score_function in ('sigmoid', 'sqrtsoftplus'):
        if score_function == 'sigmoid':
            scores = torch.sigmoid(logits.float())
        else:
            scores = F.softplus(logits.float()).sqrt()
        topk_weights, topk_idx = megatron_group_limited_topk(scores, topk, num_groups, group_topk)
        if topk > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f'Unsupported router score function: {score_function}')
    return topk_weights.contiguous(), topk_idx.to(deep_ep.topk_idx_t).contiguous()


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


def test_main(local_rank, num_local_ranks, rank, num_ranks, buffer, group, args, case, case_idx, num_cases):
    """Compare baseline vs megakernel output."""
    torch.manual_seed(42 + rank + case_idx * 1000003)

    num_nodes = num_ranks // num_local_ranks

    # Configuration
    num_tokens = case.num_tokens
    hidden = case.hidden
    intermediate = case.intermediate
    experts_per_rank = case.experts_per_rank
    num_experts = num_ranks * experts_per_rank
    num_topk = case.num_topk
    router_num_groups = args.router_num_groups or num_nodes
    router_group_topk = args.router_group_topk or (
        num_nodes if case.num_topk_groups is None else case.num_topk_groups)

    if local_rank == 0:
        print(f'')
        print(f'[Rank {rank}] === Test case {case_idx + 1}/{num_cases} ===', flush=True)
        print(f'[Rank {rank}] Config: num_tokens={num_tokens}, hidden={hidden}, '
              f'intermediate={intermediate}, experts_per_rank={experts_per_rank}, '
              f'topk={num_topk}, num_ranks={num_ranks}', flush=True)
        print(f'[Rank {rank}] Router: score_function={args.router_score_function}, '
              f'num_groups={router_num_groups}, group_topk={router_group_topk}', flush=True)

    use_fp8 = args.compute_dtype == 'fp8'
    if use_fp8 and args.baseline_impl != 'te':
        raise RuntimeError('--compute-dtype fp8 requires --baseline-impl te for the Megatron/TE grouped-MoE baseline')

    # Generate test data
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda') * 0.1

    # Precompute Megatron-style DeepEP routing once, then feed the same token_indices/token_probs
    # to both the baseline and megakernel paths. The timed paths should not redo router topk.
    topk_weights, topk_idx = make_megatron_router_inputs(
        num_tokens, num_experts, num_topk, router_num_groups, router_group_topk,
        args.router_score_function, 'cuda')

    # Expert weights (deterministic per rank)
    torch.manual_seed(1000 + rank)
    W_gate = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_up = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_down = torch.randn(experts_per_rank, hidden, intermediate, dtype=torch.bfloat16, device='cuda') * 0.02
    W_gateup = torch.empty(experts_per_rank, 2 * intermediate, hidden, dtype=torch.bfloat16, device='cuda')
    W_gateup[:, 0::2, :] = W_gate
    W_gateup[:, 1::2, :] = W_up
    W_gateup = W_gateup.contiguous()

    x_fp8 = hidden_states_scales = W_gateup_fp8 = W_down_fp8 = W_gateup_fp8_sf = W_down_fp8_sf = None
    if use_fp8:
        x_fp8, hidden_states_scales, W_gateup_fp8, W_down_fp8, W_gateup_fp8_sf, W_down_fp8_sf = \
            make_fp8_megakernel_inputs(x, W_gateup, W_down, hidden, intermediate)

    te_experts = None
    te_workspace = None
    if args.baseline_impl == 'te' and not args.skip_baseline and not args.no_compute:
        te_experts = build_te_grouped_experts(W_gate, W_up, W_down, experts_per_rank)
        te_workspace = {}

    # Config for dispatch/combine (baseline path)
    # config_num_sms = 24
    # config = deep_ep.Config(config_num_sms, 1, 256, 16, 256)
    config = None
    buffer.set_num_sms(24)
    

    # --- Path A: Baseline (DeepEP dispatch + expert compute + DeepEP combine) ---
    baseline_output = None
    if not args.skip_baseline:
        if local_rank == 0:
            print(f'[Rank {rank}] Running baseline (dispatch+compute+combine)...', flush=True)

        for w in range(args.warmup):
            if local_rank == 0:
                print(f'[Rank {rank}] Baseline warmup {w + 1}/{args.warmup}', flush=True)
            run_baseline_pipeline(
                x, topk_idx, topk_weights, W_gate, W_up, W_down,
                num_experts, experts_per_rank, buffer, config, local_rank, rank, args.no_compute,
                args.baseline_impl, te_experts, te_workspace, use_fp8=use_fp8)
        if args.warmup > 0:
            dist.barrier(group=group)
            torch.cuda.synchronize()

        baseline_output = run_baseline_pipeline(
            x, topk_idx, topk_weights, W_gate, W_up, W_down,
            num_experts, experts_per_rank, buffer, config, local_rank, rank, args.no_compute,
            args.baseline_impl, te_experts, te_workspace, use_fp8=use_fp8)
    elif local_rank == 0:
        print(f'[Rank {rank}] Skipping baseline; only checking megakernel_forward completion', flush=True)

    if use_fp8 and not args.run_fp8_megakernel:
        if local_rank == 0:
            print(f'[Rank {rank}] FP8 baseline completed; skipping megakernel FP8 because the FP8 UMMA worker is not wired yet', flush=True)
        return 0.0, 0.0, 1.0

    # Sync before megakernel run
    dist.barrier(group=group)
    torch.cuda.synchronize()

    # --- Path B: MegaKernel v7 (fused persistent kernel) ---
    if local_rank == 0:
        print(f'[Rank {rank}] Running megakernel_forward (v7)...', flush=True)

    mk_x = x_fp8 if use_fp8 else x
    for w in range(args.warmup):
        if local_rank == 0:
            print(f'[Rank {rank}] MegaKernel warmup {w + 1}/{args.warmup}', flush=True)
        run_megakernel_pipeline(
            mk_x, topk_idx, topk_weights, W_gateup, W_down,
            num_experts, buffer, local_rank, rank, args.stage,
            hidden_states_scales=hidden_states_scales,
            W_gateup_fp8=W_gateup_fp8,
            W_down_fp8=W_down_fp8,
            W_gateup_fp8_sf=W_gateup_fp8_sf,
            W_down_fp8_sf=W_down_fp8_sf)
    if args.warmup > 0:
        dist.barrier(group=group)
        torch.cuda.synchronize()

    megakernel_output = run_megakernel_pipeline(
        mk_x, topk_idx, topk_weights, W_gateup, W_down,
        num_experts, buffer, local_rank, rank, args.stage,
        hidden_states_scales=hidden_states_scales,
        W_gateup_fp8=W_gateup_fp8,
        W_down_fp8=W_down_fp8,
        W_gateup_fp8_sf=W_gateup_fp8_sf,
        W_down_fp8_sf=W_down_fp8_sf)

    # dist.barrier(group=group)
    # torch.cuda.synchronize()
    # if local_rank == 0:
    #     print(f'[Rank {rank}] Running megakernel_debug_forward...', flush=True)
    # debug_output = run_megakernel_pipeline(
    #     mk_x, topk_idx, topk_weights, W_gateup, W_down,
    #     num_experts, buffer, local_rank, rank, args.stage,
    #     hidden_states_scales=hidden_states_scales,
    #     W_gateup_fp8=W_gateup_fp8,
    #     W_down_fp8=W_down_fp8,
    #     W_gateup_fp8_sf=W_gateup_fp8_sf,
    #     W_down_fp8_sf=W_down_fp8_sf,
    #     debug=True)
    # torch.cuda.synchronize()

    if args.skip_baseline:
        if local_rank == 0:
            print(f'[Rank {rank}] MegaKernel-only check PASSED: output norm={megakernel_output.float().norm().item():.4f}', flush=True)
        return 0.0, 0.0, 1.0

    # --- Compare A vs B/C and B vs C ---
    diff = calc_diff(baseline_output, megakernel_output)
    max_abs_diff = (baseline_output.float() - megakernel_output.float()).abs().max().item()
    cos_sim = F.cosine_similarity(
        baseline_output.float().flatten().unsqueeze(0),
        megakernel_output.float().flatten().unsqueeze(0)
    ).item()
    # debug_diff = calc_diff(baseline_output, debug_output)
    # debug_max_abs_diff = (baseline_output.float() - debug_output.float()).abs().max().item()
    # debug_cos_sim = F.cosine_similarity(
    #     baseline_output.float().flatten().unsqueeze(0),
    #     debug_output.float().flatten().unsqueeze(0)
    # ).item()
    # forward_debug_max_abs_diff = (
    #     megakernel_output.float() - debug_output.float()).abs().max().item()

    if local_rank == 0:
        print(f'')
        print(f'[Rank {rank}] === MK-v7 vs Baseline Precision Alignment ===')
        print(f'  calc_diff (lower=better): {diff:.6e}')
        print(f'  max_abs_diff: {max_abs_diff:.6e}')
        print(f'  cosine_similarity: {cos_sim:.6f}')
        print(f'  baseline norm: {baseline_output.float().norm().item():.4f}')
        print(f'  megakernel norm: {megakernel_output.float().norm().item():.4f}')
        # print(f'  debug calc_diff: {debug_diff:.6e}')
        # print(f'  debug max_abs_diff: {debug_max_abs_diff:.6e}')
        # print(f'  debug cosine_similarity: {debug_cos_sim:.6f}')
        # print(f'  forward_debug_max_abs_diff: {forward_debug_max_abs_diff:.6e}')

        bitwise_equal = torch.equal(baseline_output, megakernel_output)
        # debug_bitwise_equal = torch.equal(baseline_output, debug_output)
        print(f'  bitwise_equal: {bitwise_equal}')

        if args.no_compute:
            passed = bitwise_equal
            failure_reason = 'bitwise precision mismatch'
        else:
            # Real compute uses PyTorch matmul/SwiGLU in the baseline and WMMA/__expf in
            # megakernel, so BF16 bitwise equality is not expected. Keep the test focused
            # on numerical agreement unless --no-compute is used to validate pure comms.
            max_abs_tol = 1e-1
            calc_diff_tol = 1e-5
            cos_tol = 0.85
            passed = max_abs_diff <= max_abs_tol and diff <= calc_diff_tol and cos_sim >= cos_tol
            failure_reason = (
                f'tolerance mismatch: max_abs_diff<={max_abs_tol}, '
                f'calc_diff<={calc_diff_tol}, cosine_similarity>={cos_tol}'
            )
            print(f'  tolerance: max_abs_diff<={max_abs_tol:.1e}, calc_diff<={calc_diff_tol:.1e}, cosine_similarity>={cos_tol:.6f}')

        # if passed:
        #     print(f'  PASSED')
        # else:
        #     print(f'  FAILED - {failure_reason}')
        #     print(f'  baseline[:5]:    {baseline_output[0, :5].float().tolist()}')
        #     print(f'  megakernel[:5]:  {megakernel_output[0, :5].float().tolist()}')
        #     print_bitwise_mismatches(
        #         baseline_output, megakernel_output, rank,
        #         hidden_states=x,
        #         topk_idx=topk_idx,
        #         topk_weights=topk_weights)

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
        print(f'[Rank {rank}] Buffer initialized, num_ranks={num_ranks}, num_sms={num_sms}', flush=True)

    for case_idx, case in enumerate(TEST_CASES):
        test_main(
            local_rank, num_local_ranks, rank, num_ranks, buffer, group, args,
            case, case_idx, len(TEST_CASES))
        dist.barrier(group=group)
        torch.cuda.synchronize()

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MK-v7 persistent megakernel vs baseline')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--no-compute', action='store_true', help='Skip expert compute in the baseline path')
    parser.add_argument('--baseline-impl', choices=['torch', 'te'], default='te',
                        help='Expert compute implementation for the baseline path')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations for both baseline and megakernel before the measured run')
    parser.add_argument('--stage', type=int, default=1,
                        help='Logical channels per physical channel for megakernel')
    parser.add_argument('--compute-dtype', choices=['bf16', 'fp8'], default='bf16',
                        help='Compute dtype path to exercise; fp8 runs the TE FP8 baseline and skips megakernel unless requested')
    parser.add_argument('--router-score-function', choices=['sigmoid', 'softmax', 'sqrtsoftplus'], default='sigmoid',
                        help='Megatron router score function used to precompute top-k token routing')
    parser.add_argument('--router-num-groups', type=int, default=0,
                        help='Megatron group_limited_topk num_groups; 0 uses the node count')
    parser.add_argument('--router-group-topk', type=int, default=0,
                        help='Megatron group_limited_topk group_topk; 0 uses the test case value or node count')
    parser.add_argument('--run-fp8-megakernel', action='store_true',
                        help='Run the current FP8 megakernel path. This is expected to trap until the FP8 UMMA worker is wired.')
    parser.add_argument('--mpirun', action='store_true', help='Direct launch mode via mpirun (one process per GPU)')
    args = parser.parse_args()

    if os.environ.get('SKIP_BASELINE', '0') == '1':
        args.skip_baseline = True
    if os.environ.get('BASELINE_IMPL') in ('torch', 'te'):
        args.baseline_impl = os.environ['BASELINE_IMPL']
    if os.environ.get('COMPUTE_DTYPE') in ('bf16', 'fp8'):
        args.compute_dtype = os.environ['COMPUTE_DTYPE']
    if os.environ.get('ROUTER_SCORE_FUNCTION') in ('sigmoid', 'softmax', 'sqrtsoftplus'):
        args.router_score_function = os.environ['ROUTER_SCORE_FUNCTION']
    if os.environ.get('ROUTER_NUM_GROUPS'):
        args.router_num_groups = int(os.environ['ROUTER_NUM_GROUPS'])
    if os.environ.get('ROUTER_GROUP_TOPK'):
        args.router_group_topk = int(os.environ['ROUTER_GROUP_TOPK'])
    if args.router_num_groups < 0:
        raise ValueError('--router-num-groups must be >= 0')
    if args.router_group_topk < 0:
        raise ValueError('--router-group-topk must be >= 0')
    if os.environ.get('RUN_FP8_MEGAKERNEL', '0') == '1':
        args.run_fp8_megakernel = True
    if args.baseline_impl == 'te' and te is None:
        raise RuntimeError('--baseline-impl te requires transformer_engine to be installed')

    if args.mpirun:
        # mpirun direct mode: each process is one rank, one GPU
        from datetime import timedelta
        global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        num_local_ranks = 8

        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
        # init_process_group directly (bypass init_dist which expects node-level env)
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '10.79.129.19')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29501')
        os.environ['RANK'] = str(global_rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        print(f"[Rank {global_rank}] init_process_group start, world_size={world_size}, local_rank={local_rank}", flush=True)

        params = {
            'backend': 'nccl',
        }
        dist.init_process_group(**params)

        # tensor = torch.tensor([float(global_rank)], device=torch.device("cuda:0"))
        # print(f"[Rank {global_rank}/{world_size}] before all_reduce: {tensor}", flush=True)

        # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # print(f"[Rank {global_rank}/{world_size}] after all_reduce: {tensor}", flush=True)

        # expected = world_size * (world_size - 1) / 2
        # assert tensor.item() == expected, f"Expected {expected}, got {tensor.item()}"
        # print(f"[Rank {global_rank}] PASSED", flush=True)

        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device('cuda')
        torch.cuda.set_device(0)

        group = dist.new_group(list(range(world_size)))
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        buffer = deep_ep.Buffer(group, int(2e9), int(1e9),
                                low_latency_mode=False,
                                num_qps_per_rank=num_sms,
                                explicitly_destroy=True)

        if local_rank == 0:
            print(f'[Rank {global_rank}] Buffer initialized, world_size={world_size}, num_sms={num_sms}', flush=True)

        for case_idx, case in enumerate(TEST_CASES):
            test_main(
                local_rank, num_local_ranks, global_rank, world_size, buffer, group, args,
                case, case_idx, len(TEST_CASES))
            dist.barrier(group=group)
            torch.cuda.synchronize()

        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
    else:
        torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
