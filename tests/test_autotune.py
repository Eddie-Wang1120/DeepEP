"""Autotune test: grid-search compute_batch_size x combine_start_head_percent.

Runs the megakernel autotune on a specified test case and prints a summary table.

Launch examples:
    # Single-node 8-GPU via torch.multiprocessing.spawn:
    python tests/test_autotune.py --num-processes 8

    # Via mpirun:
    mpirun -np 8 python tests/test_autotune.py --mpirun
"""

import argparse
import os
import sys
from pathlib import Path

# Under mpirun: pin CUDA_VISIBLE_DEVICES before any CUDA context is created.
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    _local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = _local_rank

import torch
import torch.distributed as dist

from utils import init_dist

import deep_ep
from deep_ep.autotune import (
    autotune_megakernel,
    COMPUTE_BATCH_SIZES,
    COMBINE_START_HEAD_PERCENTS,
)

try:
    from megakernel_test_utils import make_megatron_router_inputs
except Exception as _exc:
    make_megatron_router_inputs = None
    _IMPORT_ERROR = _exc


# ============================================================================
# Test case configuration — edit here to change the workloads being autotuned
# ============================================================================
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TestCase:
    num_tokens: int = 4096
    hidden: int = 2048
    intermediate: int = 2048
    experts_per_rank: int = 16
    num_topk: int = 8


def test(**kwargs):
    return TestCase(**kwargs)


TEST_CASES = [
    test(num_tokens=4096, hidden=2048, intermediate=2048, experts_per_rank=16, num_topk=8),
    # test(num_tokens=4096, hidden=2048, intermediate=4096, experts_per_rank=8, num_topk=4),
    # test(num_tokens=8192, hidden=4096, intermediate=4096, experts_per_rank=16, num_topk=8),
    # test(num_tokens=32768, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6),
    # Add more cases here
]


def print_results_table(all_results, best_result, case, case_idx, rank):
    """Print a formatted table of all autotune results on rank 0."""
    if rank != 0:
        return

    print('\n' + '=' * 72)
    print(f'  AUTOTUNE RESULTS — Case {case_idx + 1}')
    print('=' * 72)
    print(f'  Test case: tokens={case.num_tokens}, hidden={case.hidden}, '
          f'intermediate={case.intermediate}, experts_per_rank={case.experts_per_rank}, topk={case.num_topk}')
    print('-' * 72)

    # Table header
    header = f'{"batch_size":>12} | {"percent":>9} | {"avg_time_ms":>12} | {"status":>8}'
    print(header)
    print('-' * 72)

    # Table rows
    for batch_size, percent, avg_ms in all_results:
        is_best = (batch_size == best_result.compute_batch_size and
                   percent == best_result.combine_start_head_percent)
        status = '<-- BEST' if is_best else ''
        print(f'{batch_size:>12} | {percent:>8}% | {avg_ms:>11.4f} | {status:>8}')

    print('-' * 72)
    print(f'  BEST CONFIG: compute_batch_size={best_result.compute_batch_size}, '
          f'combine_start_head_percent={best_result.combine_start_head_percent}%')
    print(f'  BEST TIME:   {best_result.time_ms:.4f} ms/iter')
    print('=' * 72)

    # Pivot table: rows = batch_size, cols = percent
    print('\n  Pivot Table (ms/iter):')
    print(f'{"":>12}', end='')
    for pct in COMBINE_START_HEAD_PERCENTS:
        print(f' | {pct:>6}%', end='')
    print()
    print('-' * (14 + 10 * len(COMBINE_START_HEAD_PERCENTS)))

    time_map = {(bs, pct): t for bs, pct, t in all_results}
    for bs in COMPUTE_BATCH_SIZES:
        print(f'{bs:>12}', end='')
        for pct in COMBINE_START_HEAD_PERCENTS:
            t = time_map.get((bs, pct), float('nan'))
            is_best = (bs == best_result.compute_batch_size and
                       pct == best_result.combine_start_head_percent)
            marker = '*' if is_best else ' '
            print(f' | {t:>6.4f}{marker}', end='')
        print()
    print(f'\n  (* = best configuration)\n')


def run_autotune(local_rank, num_local_ranks, rank, num_ranks, buffer, group, args):
    if make_megatron_router_inputs is None:
        raise RuntimeError(f'Cannot import megakernel_test_utils: {_IMPORT_ERROR!r}')

    num_nodes = max(1, num_ranks // num_local_ranks)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    for case_idx, case in enumerate(TEST_CASES[:args.num_cases]):
        torch.manual_seed(42 + rank + case_idx * 1000003)
        num_experts = num_ranks * case.experts_per_rank

        # Generate inputs
        x = torch.randn(case.num_tokens, case.hidden, dtype=torch.bfloat16, device='cuda') * 0.1
        topk_weights, topk_idx = make_megatron_router_inputs(
            case.num_tokens, num_experts, case.num_topk,
            num_nodes, num_nodes,
            'sigmoid', 'cuda',
        )

        torch.manual_seed(1000 + rank)
        W_gate = torch.randn(case.experts_per_rank, case.intermediate, case.hidden,
                             dtype=torch.bfloat16, device='cuda') * 0.02
        W_up = torch.randn_like(W_gate) * 0.02
        W_down = torch.randn(case.experts_per_rank, case.hidden, case.intermediate,
                             dtype=torch.bfloat16, device='cuda') * 0.02
        W_gateup = torch.empty(case.experts_per_rank, 2 * case.intermediate, case.hidden,
                               dtype=torch.bfloat16, device='cuda')
        W_gateup[:, 0::2, :] = W_gate
        W_gateup[:, 1::2, :] = W_up
        W_gateup = W_gateup.contiguous()

        if rank == 0:
            print(f'\n[Autotune] Case {case_idx + 1}/{min(args.num_cases, len(TEST_CASES))}')
            print(f'  tokens={case.num_tokens}, hidden={case.hidden}, intermediate={case.intermediate}, '
                  f'experts_per_rank={case.experts_per_rank}, topk={case.num_topk}')
            print(f'  compute_batch_size candidates: {COMPUTE_BATCH_SIZES}')
            print(f'  combine_start_head_percent candidates: {COMBINE_START_HEAD_PERCENTS}')
            print(f'  num_iters={args.num_iters}, warmup_iters={args.warmup_iters}')
            print(f'  total_sms={num_sms}, comm_sms={args.megakernel_comm_sms}, stage={args.stage}')
            print()

        dist.barrier(group=group)

        best_result, all_results = autotune_megakernel(
            buffer, x, topk_idx, topk_weights, W_gateup, W_down,
            num_experts=num_experts,
            num_dispatch_sms=args.megakernel_comm_sms,
            num_combine_sms=args.megakernel_comm_sms,
            total_sms=num_sms,
            stage=args.stage,
            num_iters=args.num_iters,
            warmup_iters=args.warmup_iters,
            verbose=(rank == 0),
            group=group,
        )

        dist.barrier(group=group)
        print_results_table(all_results, best_result, case, case_idx, rank)

        # Free memory for next case
        del x, topk_weights, topk_idx, W_gate, W_up, W_down, W_gateup
        torch.cuda.empty_cache()


def run_worker(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    buffer = deep_ep.Buffer(
        group, int(2e9), int(1e9), low_latency_mode=False,
        num_qps_per_rank=num_sms, explicitly_destroy=True,
    )
    try:
        run_autotune(local_rank, num_local_ranks, rank, num_ranks, buffer, group, args)
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def run_mpirun(args):
    global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_world_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 8))

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29502')
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
        run_autotune(local_rank, local_world_size, global_rank, world_size, buffer, group, args)
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Autotune megakernel compute_batch_size and combine_start_head_percent')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-cases', type=int, default=len(TEST_CASES),
                        help='Number of test cases to autotune (from TEST_CASES list)')
    parser.add_argument('--num-iters', type=int, default=1000,
                        help='Number of timed iterations per configuration')
    parser.add_argument('--warmup-iters', type=int, default=10,
                        help='Warmup iterations before timing each configuration')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--megakernel-comm-sms', type=int, default=48)
    parser.add_argument('--mpirun', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.mpirun:
        run_mpirun(parsed_args)
    else:
        torch.multiprocessing.spawn(
            run_worker,
            args=(parsed_args.num_processes, parsed_args),
            nprocs=parsed_args.num_processes,
            join=True,
        )
