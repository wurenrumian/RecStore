#!/usr/bin/env python3
"""
Analyze embupdate stage metrics from local report events.

Supports:
1) glog text logs containing lines like:
   REPORT_LOCAL_EVENT {"table_name":"embupdate_stages", ...}
2) pure JSONL where each line is the event JSON

Usage:
  python3 src/test/scripts/analyze_embupdate_stages.py --input /path/to/log
  python3 src/test/scripts/analyze_embupdate_stages.py --input /path/to/a --input /path/to/b --top 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from typing import Dict, List, Tuple


GLOG_EVENT_PREFIX = "REPORT_LOCAL_EVENT "
STAGE_TABLE = "embupdate_stages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze embupdate stage metrics.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input log/JSONL file path. Can be used multiple times.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top slow traces to print (default: 10).",
    )
    parser.add_argument(
        "--trace-prefix",
        default="",
        help="Only include traces whose unique_id starts with this prefix.",
    )
    return parser.parse_args()


def read_events(paths: List[str]) -> List[dict]:
    events: List[dict] = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                event = parse_event_line(line)
                if event is not None:
                    events.append(event)
    return events


def parse_event_line(line: str) -> dict | None:
    if line.startswith("{"):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    idx = line.find(GLOG_EVENT_PREFIX)
    if idx >= 0:
        payload = line[idx + len(GLOG_EVENT_PREFIX) :].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    w = rank - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def summarize_metric(values: List[float]) -> str:
    if not values:
        return "count=0"
    return (
        f"count={len(values)} "
        f"mean={statistics.fmean(values):.2f}us "
        f"p50={percentile(values, 50):.2f}us "
        f"p95={percentile(values, 95):.2f}us "
        f"p99={percentile(values, 99):.2f}us "
        f"max={max(values):.2f}us"
    )


def build_trace_map(events: List[dict], trace_prefix: str) -> Dict[str, Dict[str, float]]:
    by_trace: Dict[str, Dict[str, float]] = {}
    for e in events:
        if e.get("table_name") != STAGE_TABLE:
            continue
        unique_id = str(e.get("unique_id", ""))
        if trace_prefix and not unique_id.startswith(trace_prefix):
            continue
        metric = str(e.get("metric_name", ""))
        try:
            value = float(e.get("metric_value", 0.0))
        except (TypeError, ValueError):
            continue
        trace = by_trace.setdefault(unique_id, {})
        trace[metric] = value
    return by_trace


def print_overall(by_trace: Dict[str, Dict[str, float]]) -> None:
    all_metrics: Dict[str, List[float]] = {}
    for metrics in by_trace.values():
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

    print(f"Traces: {len(by_trace)}")
    print("")
    print("Stage Metrics Summary:")
    for metric in sorted(all_metrics.keys()):
        if metric.endswith("_us"):
            print(f"  - {metric}: {summarize_metric(all_metrics[metric])}")
    print("")


def print_breakdown_ratios(by_trace: Dict[str, Dict[str, float]]) -> None:
    network_like: List[float] = []
    backend_like: List[float] = []
    serialize_like: List[float] = []

    for metrics in by_trace.values():
        client_rpc = metrics.get("client_rpc_us")
        server_total = metrics.get("server_total_us")
        backend = metrics.get("server_backend_update_us")
        serialize = metrics.get("client_serialize_us")
        if serialize is not None:
            serialize_like.append(serialize)
        if backend is not None:
            backend_like.append(backend)
        if client_rpc is not None and server_total is not None:
            network_like.append(max(0.0, client_rpc - server_total))

    print("Approx Breakdown:")
    print(f"  - serialize_us: {summarize_metric(serialize_like)}")
    print(f"  - backend_update_us: {summarize_metric(backend_like)}")
    print(f"  - rpc_minus_server_us (network/framework approx): {summarize_metric(network_like)}")
    print("")


def print_top_slow(by_trace: Dict[str, Dict[str, float]], top_n: int) -> None:
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for trace, metrics in by_trace.items():
        score = metrics.get("op_total_us")
        if score is None:
            score = metrics.get("client_total_us")
        if score is None:
            score = metrics.get("server_total_us")
        if score is None:
            continue
        ranked.append((trace, score, metrics))
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"Top {min(top_n, len(ranked))} Slow Traces:")
    for idx, (trace, score, metrics) in enumerate(ranked[:top_n], 1):
        serialize = metrics.get("client_serialize_us", float("nan"))
        rpc = metrics.get("client_rpc_us", float("nan"))
        server = metrics.get("server_total_us", float("nan"))
        backend = metrics.get("server_backend_update_us", float("nan"))
        op_total = metrics.get("op_total_us", float("nan"))
        print(
            f"  {idx:2d}. {trace} total={score:.2f}us "
            f"(op={op_total:.2f}, ser={serialize:.2f}, rpc={rpc:.2f}, server={server:.2f}, backend={backend:.2f})"
        )


def main() -> None:
    args = parse_args()
    events = read_events(args.input)
    by_trace = build_trace_map(events, args.trace_prefix)
    if not by_trace:
        print("No embupdate stage events found. Check input paths and logging mode.")
        return
    print_overall(by_trace)
    print_breakdown_ratios(by_trace)
    print_top_slow(by_trace, args.top)


if __name__ == "__main__":
    main()

