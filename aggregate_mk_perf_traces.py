#!/usr/bin/env python3
"""Aggregate megakernel and DeepEP perf trace JSON files into one Perfetto trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TRACE_RE = re.compile(r"(mk|deepep)_perf_trace_rank(\d+)(?:_(dispatch|combine))?\.json$")


def parse_trace_name(path: Path) -> tuple[str, int, str]:
    match = TRACE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse trace name from {path}")
    source = match.group(1)
    rank = int(match.group(2))
    phase = match.group(3) or source
    return source, rank, phase


def parse_rank(path: Path) -> int:
    return parse_trace_name(path)[1]


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON event array")
    return data


def aggregate(input_dir: Path, output: Path, normalize_ts: bool, source_filter: str = "all") -> tuple[int, int]:
    files = sorted(
        [
            path for path in input_dir.glob("*perf_trace_rank*.json")
            if TRACE_RE.search(path.name) and (source_filter == "all" or parse_trace_name(path)[0] == source_filter)
        ],
        key=lambda path: parse_trace_name(path),
    )
    if not files:
        raise FileNotFoundError(f"no {source_filter} *perf_trace_rank*.json found under {input_dir}")

    # Pass 1: extract per-file base_ts_ns from metadata events for cross-rank alignment.
    # DeepEP can emit dispatch/combine as separate files for the same rank, so keep files distinct.
    file_base_ts: dict[Path, int] = {}
    file_events: dict[Path, tuple[str, int, str, list[dict[str, Any]]]] = {}

    for path in files:
        source, rank, phase = parse_trace_name(path)
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
        tid_offset = 10000 if phase in ("dispatch", "deepep") else 20000 if phase == "combine" else 0
        is_deepep_trace = phase in ("dispatch", "combine", "deepep")
        for event in events:
            if not isinstance(event, dict):
                continue
            event = dict(event)
            args = dict(event.get("args") or {})
            args.setdefault("rank", rank)
            args.setdefault("trace_source", "deepep" if is_deepep_trace else "megakernel")
            if is_deepep_trace:
                args.setdefault("deepep_phase", "combined" if phase == "deepep" else phase)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate megakernel and DeepEP perf traces into one Perfetto trace")
    parser.add_argument("--input-dir", type=Path, default=Path.cwd(), help="directory containing *perf_trace_rank*.json")
    parser.add_argument("--output", type=Path, default=None, help="output JSON path")
    parser.add_argument("--source", choices=("all", "mk", "deepep"), default="all", help="trace source to aggregate")
    parser.add_argument("--split-sources", action="store_true", default=True, help="also write source-specific JSON files")
    parser.add_argument("--no-split-sources", action="store_false", dest="split_sources", help="only write the requested output")
    parser.add_argument("--no-normalize-ts", action="store_true", help="keep original timestamps instead of shifting min ts to 0")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output = args.output.resolve() if args.output else input_dir / "mk_perf_trace_all.json"
    num_files, num_events = aggregate(input_dir, output, normalize_ts=not args.no_normalize_ts, source_filter=args.source)
    print(f"Aggregated {num_events} events from {num_files} files -> {output}")

    if args.split_sources and args.source == "all":
        mk_output = output.with_name(output.stem + "_megakernel_only" + output.suffix)
        deepep_output = output.with_name(output.stem + "_deepep_only" + output.suffix)
        for source, source_output in (("mk", mk_output), ("deepep", deepep_output)):
            try:
                num_files, num_events = aggregate(input_dir, source_output, normalize_ts=not args.no_normalize_ts, source_filter=source)
            except FileNotFoundError as exc:
                print(f"Skipped {source}: {exc}")
                continue
            print(f"Aggregated {source} {num_events} events from {num_files} files -> {source_output}")


if __name__ == "__main__":
    main()
