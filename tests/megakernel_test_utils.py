import contextlib
import math
import os
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
from packaging import version

import deep_ep

os.environ.setdefault('NVTE_CUTEDSL_FUSED_GROUPED_MLP', '1')
os.environ.setdefault('NVTE_GROUPED_LINEAR_SINGLE_PARAM', '1')

try:
    import transformer_engine.pytorch.ops as te_ops
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast, moe_permute_with_probs, moe_unpermute
except ImportError:
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
        assert indices.shape == probs_indices.shape
        topk = indices.shape[1]
        multihot_indices = torch.empty((num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda")
        probs_in_multihot = torch.empty((num_of_tokens, num_of_local_experts), dtype=probs_indices.dtype, device="cuda")
        position_map = torch.empty((num_of_tokens, num_of_local_experts), dtype=torch.int32, device="cuda")
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))
        _indices_to_multihot_kernel[(num_of_tokens,)](
            indices, probs_indices, multihot_indices, probs_in_multihot, position_map,
            num_of_local_experts, num_of_local_experts_next_power_of_2, topk,
            topk_next_power_of_2, BLOCK_SIZE=32, num_warps=1)
        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        position_map = ctx.saved_tensors[0]
        grad_probs_indices = torch.empty((ctx.num_of_tokens, ctx.topk), dtype=grad_probs_in_multihot.dtype, device="cuda")
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(ctx.topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(ctx.num_of_local_experts)))
        _multihot_to_indices_kernel[(ctx.num_of_tokens,)](
            grad_probs_in_multihot.contiguous(), position_map, grad_probs_indices,
            ctx.num_of_local_experts, num_of_local_experts_next_power_of_2, ctx.topk,
            topk_next_power_of_2, BLOCK_SIZE=32, num_warps=1)
        return None, grad_probs_indices, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)


def _tokens_per_expert_tensor(tokens_per_expert, device):
    if isinstance(tokens_per_expert, torch.Tensor):
        return tokens_per_expert.to(device=device, dtype=torch.int32)
    return torch.tensor(tokens_per_expert, device=device, dtype=torch.int32)


def build_te_grouped_experts(W_gate, W_up, W_down, experts_per_rank):
    """Build TE baseline using raw te_ops.Sequential(GroupedLinear, ScaledSwiGLU, GroupedLinear)."""
    if te_ops is None:
        raise RuntimeError('Transformer Engine ops are required for the TE baseline')

    hidden = W_gate.shape[-1]
    intermediate = W_gate.shape[1]
    device = W_gate.device
    dtype = W_gate.dtype

    fc1 = te_ops.GroupedLinear(
        experts_per_rank, hidden, 2 * intermediate, bias=False,
        device=device, dtype=dtype,
        single_grouped_weight=True, single_grouped_bias=False,
        accumulate_into_main_grad=False, delay_wgrad_compute=False,
    )
    act = te_ops.ScaledSwiGLU(glu_interleave_size=32)
    fc2 = te_ops.GroupedLinear(
        experts_per_rank, intermediate, hidden, bias=False,
        device=device, dtype=dtype,
        single_grouped_weight=True, single_grouped_bias=False,
        accumulate_into_main_grad=False, delay_wgrad_compute=False,
    )

    # Pack weights: interleave gate/up in blocks of 32
    fc1_chunks = []
    for start in range(0, intermediate, 32):
        fc1_chunks.append(W_gate[:, start:start + 32, :])
        fc1_chunks.append(W_up[:, start:start + 32, :])
    fc1_weight = torch.cat(fc1_chunks, dim=1).contiguous()

    with torch.no_grad():
        if hasattr(fc1, 'weight'):
            fc1.weight.copy_(fc1_weight)
            fc2.weight.copy_(W_down.contiguous())
        else:
            for e in range(experts_per_rank):
                getattr(fc1, f'weight{e}').copy_(fc1_weight[e])
                getattr(fc2, f'weight{e}').copy_(W_down[e].contiguous())

    return te_ops.Sequential(fc1, act, fc2)


def get_te_grouped_expert_weight_grads(te_experts, experts_per_rank):
    """Extract fc1 and fc2 weight gradients from a te_ops.Sequential baseline."""
    fc1_op = te_experts[0]
    fc2_op = te_experts[2]

    if hasattr(fc1_op, 'weight'):
        # single_grouped_weight=True: one tensor per layer
        fc1_grad = fc1_op.weight.grad.detach()
        fc2_grad = fc2_op.weight.grad.detach()
    else:
        # single_grouped_weight=False: per-expert weight0, weight1, ...
        fc1_grads = [getattr(fc1_op, f'weight{e}').grad.detach() for e in range(experts_per_rank)]
        fc2_grads = [getattr(fc2_op, f'weight{e}').grad.detach() for e in range(experts_per_rank)]
        fc1_grad = torch.stack(fc1_grads, dim=0)
        fc2_grad = torch.stack(fc2_grads, dim=0)

    return fc1_grad, fc2_grad


def te_fp8_context(enabled):
    if not enabled:
        return contextlib.nullcontext()
    if fp8_autocast is None or te_recipe is None:
        raise RuntimeError('Transformer Engine FP8 support is required for the Megatron baseline')
    recipe = te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID,
                                     amax_history_len=16, amax_compute_algo='max')
    return fp8_autocast(enabled=True, fp8_recipe=recipe)


def moe_compute_on_recv_te(recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list,
                           te_experts, workspace, experts_per_rank, use_fp8=False):
    if moe_permute_with_probs is None or moe_unpermute is None:
        raise RuntimeError('Transformer Engine MoE operators are required for the Megatron baseline')
    if not HAVE_TRITON_AVAILABLE:
        raise RuntimeError('Triton is required for the Megatron baseline')
    assert recv_topk_weights.dtype == torch.float32
    routing_map, probs_map = fused_indices_to_multihot(recv_topk_idx, recv_topk_weights, experts_per_rank)
    tokens_per_expert = _tokens_per_expert_tensor(recv_num_tokens_per_expert_list, recv_x.device)
    permuted_x, permuted_probs, row_map = moe_permute_with_probs(
        recv_x, probs_map, routing_map, num_out_tokens=tokens_per_expert.sum().item())
    with te_fp8_context(use_fp8):
        permuted_output = te_experts(permuted_x, tokens_per_expert, permuted_probs, tokens_per_expert)
    return moe_unpermute(permuted_output, row_map, restore_shape=recv_x.shape)


def megatron_group_limited_topk(scores, topk, num_groups, group_topk):
    num_tokens, num_experts = scores.shape
    if num_groups <= 0 or num_experts % num_groups != 0:
        raise ValueError(f'num_groups must divide num_experts, got num_groups={num_groups}, num_experts={num_experts}')
    if group_topk <= 0 or group_topk > num_groups:
        raise ValueError(f'group_topk must be in [1, num_groups], got group_topk={group_topk}, num_groups={num_groups}')
    if topk % group_topk != 0:
        raise ValueError(f'Megatron group_limited_topk requires topk % group_topk == 0, got topk={topk}, group_topk={group_topk}')
    group_scores = scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        num_tokens, num_groups, num_experts // num_groups).reshape(num_tokens, num_experts)
    return torch.topk(scores.masked_fill(~score_mask.bool(), float('-inf')), k=topk, dim=-1)


def make_megatron_router_inputs(num_tokens, num_experts, topk, num_groups, group_topk, score_function, device):
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    if score_function == 'softmax':
        topk_logits, topk_idx = megatron_group_limited_topk(logits, topk, num_groups, group_topk)
        topk_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    elif score_function in ('sigmoid', 'sqrtsoftplus'):
        scores = torch.sigmoid(logits.float()) if score_function == 'sigmoid' else F.softplus(logits.float()).sqrt()
        topk_weights, topk_idx = megatron_group_limited_topk(scores, topk, num_groups, group_topk)
        if topk > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f'Unsupported router score function: {score_function}')
    return topk_weights.contiguous(), topk_idx.to(deep_ep.topk_idx_t).contiguous()
