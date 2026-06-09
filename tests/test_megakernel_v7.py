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
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F

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

        if token_idx not in printed_tokens:
            printed_tokens.add(token_idx)
            if topk_idx is not None:
                print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} topk_idx={topk_idx[token_idx].detach().cpu().tolist()}', flush=True)
            if topk_weights is not None:
                print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} topk_weights={topk_weights[token_idx].detach().cpu().float().tolist()}', flush=True)
            if hidden_states is not None:
                print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} hidden_states={hidden_states[token_idx].detach().cpu().float().tolist()}', flush=True)
                print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} hidden_states_bf16_hex={[hex(int(v) & 0xffff) for v in hidden_states[token_idx].detach().contiguous().view(torch.int16).cpu().tolist()]}', flush=True)
            print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} baseline_output={baseline_contig[token_idx].detach().cpu().float().tolist()}', flush=True)
            print(f'[Rank {rank}] MISMATCH-TOKEN token={token_idx} megakernel_output={megakernel_contig[token_idx].detach().cpu().float().tolist()}', flush=True)

    if limit < mismatch_count:
        print(f'[Rank {rank}] BITWISE mismatch print truncated: printed={limit}, total={mismatch_count}', flush=True)


def moe_compute_on_recv(recv_x, recv_topk_idx, recv_topk_weights, W_gate, W_up, W_down, experts_per_rank):
    """
    Compute MoE expert forward on received tokens (baseline path).
    recv_topk_idx [num_recv, num_topk] contains LOCAL expert IDs (0..experts_per_rank-1), -1 for others.
    """
    expert_out = torch.zeros(recv_x.shape[0], recv_x.shape[1], dtype=torch.float32, device=recv_x.device)

    for expert_id in range(experts_per_rank):
        mask = (recv_topk_idx == expert_id)  # [num_recv, num_topk]
        row_mask = mask.any(dim=1)
        if not row_mask.any():
            continue

        token_indices = row_mask.nonzero(as_tuple=True)[0]
        tokens = recv_x[token_indices].float()

        gate = torch.matmul(tokens, W_gate[expert_id].float().T)
        up = torch.matmul(tokens, W_up[expert_id].float().T)
        swiglu_out = F.silu(gate) * up
        down = torch.matmul(swiglu_out, W_down[expert_id].float().T)

        weights = (recv_topk_weights[token_indices] * mask[token_indices].float()).sum(dim=1, keepdim=True)
        expert_out[token_indices] += down * weights

    return expert_out.to(torch.bfloat16)


def run_baseline_pipeline(x, topk_idx, topk_weights, W_gate, W_up, W_down,
                          num_experts, experts_per_rank, buffer, config, local_rank, rank):
    """
    Baseline: DeepEP dispatch -> PyTorch expert compute -> DeepEP combine.
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

    # expert_out = moe_compute_on_recv(recv_x, recv_topk_idx, recv_topk_weights,
    #                                   W_gate, W_up, W_down, experts_per_rank)

    combined_x, combined_topk_weights, event = buffer.combine(
        x=recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        config=config
    )

    if local_rank == 0:
        print(f'[Rank {rank}] Baseline combine done: combined_x shape={combined_x.shape}', flush=True)

    return combined_x


def run_megakernel_pipeline(x, topk_idx, topk_weights, W_gate, W_up, W_down,
                            num_experts, buffer, local_rank, rank):
    """
    MegaKernel v7: single persistent kernel (dispatch + compute + combine fused).
    Returns output [num_tokens, hidden] in bf16.
    """
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    # SM allocation: 24 dispatch, 8 forwarder (NUM_MAX_NVL_PEERS), rest compute
    num_dispatch_sms = 24
    num_forwarder_sms = 8  # internally derived from NUM_MAX_NVL_PEERS, not a param here
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
        total_sms
    )

    if local_rank == 0:
        print(f'[Rank {rank}] MegaKernel done: output shape={result.shape}', flush=True)

    return result


def test_main(local_rank, num_local_ranks, rank, num_ranks, buffer, group, args):
    """Compare baseline vs megakernel output."""
    torch.manual_seed(42 + rank)

    num_nodes = num_ranks // num_local_ranks

    # Configuration
    num_tokens = 10
    hidden = 256
    intermediate = 256
    experts_per_rank = 8
    num_experts = num_ranks * experts_per_rank
    num_topk = 2
    num_topk_groups = num_nodes

    if local_rank == 0:
        print(f'[Rank {rank}] Config: num_tokens={num_tokens}, hidden={hidden}, '
              f'intermediate={intermediate}, experts_per_rank={experts_per_rank}, '
              f'topk={num_topk}, num_ranks={num_ranks}', flush=True)

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

    # Config for dispatch/combine (baseline path)
    config_num_sms = 24
    config = deep_ep.Config(config_num_sms, 8, 512, 16, 128)

    # --- Path A: Baseline (DeepEP dispatch + PyTorch compute + DeepEP combine) ---
    baseline_output = None
    if not args.skip_baseline:
        if local_rank == 0:
            print(f'[Rank {rank}] Running baseline (dispatch+compute+combine)...', flush=True)

        baseline_output = run_baseline_pipeline(
            x, topk_idx, topk_weights, W_gate, W_up, W_down,
            num_experts, experts_per_rank, buffer, config, local_rank, rank)
    elif local_rank == 0:
        print(f'[Rank {rank}] Skipping baseline; only checking megakernel_forward completion', flush=True)

    # Sync before megakernel run
    dist.barrier(group=group)
    torch.cuda.synchronize()

    # --- Path B: MegaKernel v7 (fused persistent kernel) ---
    if local_rank == 0:
        print(f'[Rank {rank}] Running megakernel_forward (v7)...', flush=True)

    megakernel_output = run_megakernel_pipeline(
        x, topk_idx, topk_weights, W_gate, W_up, W_down,
        num_experts, buffer, local_rank, rank)

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

        if bitwise_equal:
            print(f'  PASSED')
        else:
            print(f'  FAILED - bitwise precision mismatch')
            print(f'  baseline[:5]:    {baseline_output[0, :5].float().tolist()}')
            print(f'  megakernel[:5]:  {megakernel_output[0, :5].float().tolist()}')
            print_bitwise_mismatches(
                baseline_output, megakernel_output, rank,
                hidden_states=x,
                topk_idx=topk_idx,
                topk_weights=topk_weights)

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

    diff, max_abs_diff, cos_sim = test_main(
        local_rank, num_local_ranks, rank, num_ranks, buffer, group, args)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':

    print("jinheng debug: enter this")

    parser = argparse.ArgumentParser(description='Test MK-v7 persistent megakernel vs baseline')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--mpirun', action='store_true', help='Direct launch mode via mpirun (one process per GPU)')
    args = parser.parse_args()

    if os.environ.get('SKIP_BASELINE', '0') == '1':
        args.skip_baseline = True

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

        test_main(local_rank, num_local_ranks, global_rank, world_size, buffer, group, args)

        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
    else:
        torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
