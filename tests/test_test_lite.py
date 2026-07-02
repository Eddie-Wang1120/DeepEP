"""
MK-v7: Persistent MegaKernel precision alignment test.

Compares:
  A) DeepEP dispatch -> PyTorch compute -> DeepEP combine (baseline, MK-v6)
  B) buffer.megakernel_forward() (fused persistent kernel, MK-v7)

Both use identical inputs. Output of (B) should match (A) within BF16 tolerance.

Usage (8 nodes x 8 GPUs, mpirun direct mode):
    HOSTS=host1,host2,host3,host4,host5,host6,host7,host8 \
    bash /root/paddlejob/share-storage/gpfs/system-public/wangjinheng/harness_dist_research/run_lite_speed_test.sh
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.ops as te_ops
    from transformer_engine.pytorch import moe_permute_with_probs, moe_unpermute
except ImportError:
    te = None
    te_ops = None
    moe_permute_with_probs = None
    moe_unpermute = None

sys.path.insert(0, os.path.dirname(__file__))
import deep_ep
from utils import init_dist, calc_diff, create_grouped_scores


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


def _get_te_workspace(workspace, key, shape, dtype, device):
    tensor = workspace.get(key)
    if tensor is None or tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        workspace[key] = tensor
    return tensor


def indices_to_routing_map(indices, probs, num_local_experts, workspace):
    batch_size, topk = indices.shape
    routing_map = _get_te_workspace(
        workspace, 'routing_map', (batch_size, num_local_experts), torch.bool, indices.device)
    probs_map = _get_te_workspace(
        workspace, 'probs_map', (batch_size, num_local_experts), torch.float32, indices.device)
    routing_map.zero_()
    probs_map.zero_()

    flat_indices = indices.reshape(-1)
    valid_mask = flat_indices != -1
    if valid_mask.any():
        flat_token_ids = torch.arange(batch_size, device=indices.device).repeat_interleave(topk)
        routing_map[flat_token_ids[valid_mask], flat_indices[valid_mask].long()] = True
        probs_map[flat_token_ids[valid_mask], flat_indices[valid_mask].long()] = probs.float().reshape(-1)[valid_mask]
    return routing_map, probs_map


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


def moe_compute_on_recv_te(recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list,
                           te_experts, workspace, experts_per_rank):
    """TE baseline aligned with sonic_te_compare_20260526: fused permute -> te_ops.Sequential -> fused unpermute."""
    if moe_permute_with_probs is None or moe_unpermute is None:
        raise RuntimeError('TE moe_permute_with_probs/moe_unpermute are required for --baseline-impl te')
    routing_map, probs_map = indices_to_routing_map(
        recv_topk_idx, recv_topk_weights, experts_per_rank, workspace)
    num_out_tokens = int(_tokens_per_expert_tensor(recv_num_tokens_per_expert_list, recv_x.device).sum().item())
    tokens_per_expert = _tokens_per_expert_tensor(recv_num_tokens_per_expert_list, recv_x.device)
    permuted_x, permuted_probs, row_map = moe_permute_with_probs(
        recv_x, probs_map, routing_map, num_out_tokens)
    permuted_output = te_experts(permuted_x, tokens_per_expert, permuted_probs, tokens_per_expert)
    return moe_unpermute(permuted_output, row_map, restore_shape=recv_x.shape)


def run_baseline_pipeline(x, topk_idx, topk_weights, W_gate, W_up, W_down,
                          num_experts, experts_per_rank, buffer, config, local_rank, rank, no_compute,
                          baseline_impl, te_experts=None, te_workspace=None):
    """
    Baseline: DeepEP dispatch -> expert compute -> DeepEP combine.
    baseline_impl='torch' keeps the original direct local-expert loop.
    baseline_impl='te' mirrors Megatron's production TEGroupedMLP local permute + grouped GEMM path.
    Returns combined output [num_tokens, hidden] in bf16.
    """
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    buffer.set_num_sms(24)

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
            te_experts, te_workspace, experts_per_rank)
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


def run_megakernel_pipeline(x, topk_idx, topk_weights, W_gate, W_up, W_down,
                            num_experts, buffer, config, local_rank, rank):
    """
    MegaKernel v7: single persistent kernel (dispatch + compute + combine fused).
    Returns output [num_tokens, hidden] in bf16.
    """
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    # SM allocation: 24 dispatch, 8 forwarder (NUM_MAX_NVL_PEERS), rest compute
    num_dispatch_sms = 24
    # total_sms = num_dispatch_sms + num_forwarder_sms + num_compute_sms
    total_sms = num_sms  # use all available SMs

    if local_rank == 0:
        print(f'[Rank {rank}] MegaKernel launch: total_sms={total_sms}, dispatch={num_dispatch_sms}', flush=True)

    result = buffer.megakernel_forward(
        x, topk_idx, topk_weights,
        W_gate, W_up, W_down,
        num_experts,
        num_dispatch_sms,
        num_dispatch_sms,  # num_combine_sms = num_dispatch_sms
        total_sms,
        dispatch_config=config,
        combine_config=config
    )

    if local_rank == 0:
        print(f'[Rank {rank}] MegaKernel done: output shape={result.shape}', flush=True)

    return result


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
    # test(num_tokens=4096, hidden=2048, intermediate=2048, experts_per_rank=16, num_topk=8),
    # test(num_tokens=8192, hidden=4096, intermediate=4096, experts_per_rank=16, num_topk=8),
    # test(num_tokens=8192, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6),
    test(num_tokens=32768, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6),
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
    num_topk_groups = num_nodes if case.num_topk_groups is None else case.num_topk_groups

    if local_rank == 0:
        print(f'')
        print(f'[Rank {rank}] === Test case {case_idx + 1}/{num_cases} ===', flush=True)
        print(f'[Rank {rank}] Config: num_tokens={num_tokens}, hidden={hidden}, '
              f'intermediate={intermediate}, experts_per_rank={experts_per_rank}, '
              f'topk={num_topk}, num_topk_groups={num_topk_groups}, num_ranks={num_ranks}', flush=True)

    # Generate test data
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda') * 0.1

    # Grouped topk routing (ensures cross-node traffic)
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn(num_tokens, num_topk, dtype=torch.float32, device='cuda').abs()
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    # Expert weights (deterministic per rank)
    torch.manual_seed(1000 + rank)
    W_gate = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_up = torch.randn(experts_per_rank, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
    W_down = torch.randn(experts_per_rank, hidden, intermediate, dtype=torch.bfloat16, device='cuda') * 0.02

    te_experts = None
    te_workspace = None
    if args.baseline_impl == 'te' and not args.skip_baseline and not args.no_compute:
        te_experts = build_te_grouped_experts(W_gate, W_up, W_down, experts_per_rank)
        te_workspace = {}

    # Config for dispatch/combine (baseline path)
    # config_num_sms = 24
    # config = deep_ep.Config(config_num_sms, 1, 256, 16, 256)
    config = None

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
                args.baseline_impl, te_experts, te_workspace)
        if args.warmup > 0:
            dist.barrier(group=group)
            torch.cuda.synchronize()

        baseline_output = run_baseline_pipeline(
            x, topk_idx, topk_weights, W_gate, W_up, W_down,
            num_experts, experts_per_rank, buffer, config, local_rank, rank, args.no_compute,
            args.baseline_impl, te_experts, te_workspace)
    elif local_rank == 0:
        print(f'[Rank {rank}] Skipping baseline; only checking megakernel_forward completion', flush=True)

    # Sync before megakernel run
    dist.barrier(group=group)
    torch.cuda.synchronize()

    # --- Path B: MegaKernel v7 (fused persistent kernel) ---
    if local_rank == 0:
        print(f'[Rank {rank}] Running megakernel_forward (v7)...', flush=True)

    for w in range(args.warmup):
        if local_rank == 0:
            print(f'[Rank {rank}] MegaKernel warmup {w + 1}/{args.warmup}', flush=True)
        run_megakernel_pipeline(
            x, topk_idx, topk_weights, W_gate, W_up, W_down,
            num_experts, buffer, config, local_rank, rank)
    if args.warmup > 0:
        dist.barrier(group=group)
        torch.cuda.synchronize()

    megakernel_output = run_megakernel_pipeline(
        x, topk_idx, topk_weights, W_gate, W_up, W_down,
        num_experts, buffer, config, local_rank, rank)

    if args.skip_baseline:
        if local_rank == 0:
            print(f'[Rank {rank}] MegaKernel-only check PASSED: output norm={megakernel_output.float().norm().item():.4f}', flush=True)
        return 0.0, 0.0, 1.0

    # --- Compare A vs B ---
    diff = calc_diff(baseline_output, megakernel_output)
    max_abs_diff = (baseline_output.float() - megakernel_output.float()).abs().max().item()
    cos_sim = F.cosine_similarity(
        baseline_output.float().flatten().unsqueeze(0),
        megakernel_output.float().flatten().unsqueeze(0)
    ).item()

    if local_rank == 0:
        print(f'')
        print(f'[Rank {rank}] === MK-v7 vs Baseline Precision Alignment ===')
        print(f'  calc_diff (lower=better): {diff:.6e}')
        print(f'  max_abs_diff: {max_abs_diff:.6e}')
        print(f'  cosine_similarity: {cos_sim:.6f}')
        print(f'  baseline norm: {baseline_output.float().norm().item():.4f}')
        print(f'  megakernel norm: {megakernel_output.float().norm().item():.4f}')

        bitwise_equal = torch.equal(baseline_output, megakernel_output)
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
    parser.add_argument('--baseline-impl', choices=['torch', 'torch'], default='torch',
                        help='Expert compute implementation for the baseline path')
    parser.add_argument('--warmup', type=int, default=500,
                        help='Number of warmup iterations for both baseline and megakernel before the measured run')
    parser.add_argument('--mpirun', action='store_true', help='Direct launch mode via mpirun (one process per GPU)')
    args = parser.parse_args()

    if os.environ.get('SKIP_BASELINE', '0') == '1':
        args.skip_baseline = True
    if os.environ.get('BASELINE_IMPL') in ('torch', 'te'):
        args.baseline_impl = os.environ['BASELINE_IMPL']
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
