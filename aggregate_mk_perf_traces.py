#!/usr/bin/env python3
"""Aggregate megakernel and DeepEP perf trace JSON files into one Perfetto trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TRACE_RE = re.compile(r"(mk|deepep)_perf_trace_rank(\d+)(?:_(forward|backward|dispatch|combine|notify))?(?:_iter(\d+))?\.json$")


def parse_trace_name(path: Path) -> tuple[str, int, str, int]:
    match = TRACE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse trace name from {path}")
    source = match.group(1)
    rank = int(match.group(2))
    phase = match.group(3) or source
    # iteration index: -1 means "untagged" (legacy single-run files)
    iteration = int(match.group(4)) if match.group(4) is not None else -1
    return source, rank, phase, iteration


def parse_rank(path: Path) -> int:
    return parse_trace_name(path)[1]


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON event array")
    return data


def aggregate(input_dir: Path, output: Path, normalize_ts: bool, source_filter: str = "all",
              phase_filter: str | None = None, iter_filter: int | None = None) -> tuple[int, int]:
    files = []
    for path in input_dir.glob("*perf_trace_rank*.json"):
        if not TRACE_RE.search(path.name):
            continue
        source, _rank, phase, iteration = parse_trace_name(path)
        if source_filter != "all" and source != source_filter:
            continue
        if phase_filter is not None and phase != phase_filter:
            continue
        if iter_filter is not None and iteration != iter_filter:
            continue
        files.append(path)
    files = sorted(files, key=lambda path: parse_trace_name(path))
    if not files:
        phase_desc = f" {phase_filter}" if phase_filter is not None else ""
        raise FileNotFoundError(f"no {source_filter}{phase_desc} *perf_trace_rank*.json found under {input_dir}")

    # Pass 1: extract per-file base_ts_ns from metadata events for cross-rank alignment.
    # DeepEP can emit dispatch/combine as separate files for the same rank, so keep files distinct.
    file_base_ts: dict[Path, int] = {}
    file_events: dict[Path, tuple[str, int, str, list[dict[str, Any]]]] = {}

    for path in files:
        source, rank, phase, _iteration = parse_trace_name(path)
        events = load_events(path)
        file_events[path] = (source, rank, phase, events)
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("ph") != "M":
                continue
            name = event.get("name")
            if name not in ("mk_base_ts_ns", "deepep_base_ts_ns"):
                continue
            args = event.get("args") or {}
            base = args.get("base_ts_ns")
            if isinstance(base, (int, float)):
                file_base_ts[path] = int(base)
                break

    # Each rank trace already uses a local zero-based timestamp. Do not offset by base_ts_ns:
    # CPU steady_clock and GPU globaltimer are different clock domains, and their epochs may
    # also differ across nodes. Offsetting by base_ts_ns would spread ranks far apart.
    file_offset_us: dict[Path, float] = {path: 0.0 for path in file_events}

    all_events: list[dict[str, Any]] = []
    min_ts: float | None = None

    for path, (source, rank, phase, events) in sorted(file_events.items(), key=lambda item: parse_trace_name(item[0])):
        offset = file_offset_us.get(path, 0.0)
        tid_offset = 10000 if phase in ("dispatch", "deepep") else 20000 if phase == "combine" else 30000 if phase == "backward" else 0
        is_deepep_trace = phase in ("dispatch", "combine", "deepep")
        is_megakernel_phase = source == "mk" and phase in ("forward", "backward")
        for event in events:
            if not isinstance(event, dict):
                continue
            event = dict(event)
            args = dict(event.get("args") or {})
            args.setdefault("rank", rank)
            args.setdefault("trace_source", "deepep" if is_deepep_trace else "megakernel")
            if is_deepep_trace:
                args.setdefault("deepep_phase", "combined" if phase == "deepep" else phase)
            if is_megakernel_phase:
                args.setdefault("megakernel_phase", phase)
            event["args"] = args
            event["pid"] = rank
            tid = event.get("tid")
            if isinstance(tid, int):
                event["tid"] = tid + tid_offset
            # Apply cross-rank/file offset to timing events
            ts = event.get("ts")
            if isinstance(ts, (int, float)) and event.get("ph") not in ("M",):
                event["ts"] = ts + offset
            all_events.append(event)
            ts_val = event.get("ts")
            if isinstance(ts_val, (int, float)) and event.get("ph") not in ("M",):
                min_ts = ts_val if min_ts is None else min(min_ts, ts_val)

    if normalize_ts and min_ts is not None:
        for event in all_events:
            ts = event.get("ts")
            if isinstance(ts, (int, float)) and event.get("ph") not in ("M",):
                event["ts"] = ts - min_ts

    all_events.sort(key=lambda event: (event.get("ts", 0), event.get("pid", 0), event.get("tid", 0)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(all_events, file, separators=(",", ":"))
        file.write("\n")

    return len(files), len(all_events)


def direction_output_path(output: Path, direction: str) -> Path:
    stem = output.stem
    if stem.startswith("mk_perf_trace_all"):
        stem = "mk_perf_trace_" + direction + stem[len("mk_perf_trace_all"):]
    else:
        stem = f"{stem}_{direction}"
    return output.with_name(stem + output.suffix)


def run_aggregate(input_dir: Path, output: Path, normalize_ts: bool, source: str,
                  split_sources: bool, split_directions: bool, iter_filter: int | None) -> None:
    num_files, num_events = aggregate(input_dir, output, normalize_ts=normalize_ts,
                                      source_filter=source, iter_filter=iter_filter)
    print(f"Aggregated {num_events} events from {num_files} files -> {output}")

    if split_sources and source == "all":
        mk_output = output.with_name(output.stem + "_megakernel_only" + output.suffix)
        deepep_output = output.with_name(output.stem + "_deepep_only" + output.suffix)
        for src, source_output in (("mk", mk_output), ("deepep", deepep_output)):
            try:
                num_files, num_events = aggregate(input_dir, source_output, normalize_ts=normalize_ts,
                                                  source_filter=src, iter_filter=iter_filter)
            except FileNotFoundError as exc:
                print(f"Skipped {src}: {exc}")
                continue
            print(f"Aggregated {src} {num_events} events from {num_files} files -> {source_output}")

    if split_directions and source == "all":
        for direction in ("forward", "backward"):
            direction_output = direction_output_path(output, direction)
            try:
                num_files, num_events = aggregate(
                    input_dir, direction_output, normalize_ts=normalize_ts,
                    source_filter="mk", phase_filter=direction, iter_filter=iter_filter,
                )
            except FileNotFoundError as exc:
                print(f"Skipped mk {direction}: {exc}")
                continue
            print(f"Aggregated mk {direction} {num_events} events from {num_files} files -> {direction_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate megakernel and DeepEP perf traces into one Perfetto trace")
    parser.add_argument("--input-dir", type=Path, default=Path.cwd(), help="directory containing *perf_trace_rank*.json")
    parser.add_argument("--output", type=Path, default=None, help="output JSON path")
    parser.add_argument("--source", choices=("all", "mk", "deepep"), default="all", help="trace source to aggregate")
    parser.add_argument("--split-sources", action="store_true", default=True, help="also write source-specific JSON files")
    parser.add_argument("--no-split-sources", action="store_false", dest="split_sources", help="do not write source-specific JSON files")
    parser.add_argument("--split-directions", action="store_true", default=True, help="also write mk forward/backward JSON files")
    parser.add_argument("--no-split-directions", action="store_false", dest="split_directions", help="do not write mk forward/backward JSON files")
    parser.add_argument("--no-normalize-ts", action="store_true", help="keep original timestamps instead of shifting min ts to 0")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output = args.output.resolve() if args.output else input_dir / "mk_perf_trace_all.json"
    normalize_ts = not args.no_normalize_ts

    # Discover iteration indices present in the directory. Files written by a single run
    # carry `_iter{N}`; legacy untagged files map to iteration -1.
    iterations = sorted({
        parse_trace_name(path)[3]
        for path in input_dir.glob("*perf_trace_rank*.json")
        if TRACE_RE.search(path.name)
    })
    tagged_iters = [i for i in iterations if i >= 0]

    if tagged_iters:
        # One aggregated output per run: mk_perf_trace_all_iter{N}.json
        for i in tagged_iters:
            iter_output = output.with_name(f"{output.stem}_iter{i}{output.suffix}")
            run_aggregate(
                input_dir, iter_output, normalize_ts, args.source,
                args.split_sources, args.split_directions, iter_filter=i,
            )
    else:
        # Legacy single-run behavior.
        run_aggregate(
            input_dir, output, normalize_ts, args.source,
            args.split_sources, args.split_directions, iter_filter=None,
        )


if __name__ == "__main__":
    main()
