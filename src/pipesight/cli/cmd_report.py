"""`pipesight report <trace.json>` -- analyze a captured trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipesight.analysis.idle import gpu_idle_from_samples, gpu_idle_from_spans
from pipesight.analysis.recommend import build_recommendations
from pipesight.analysis.stats import stage_stats
from pipesight.cli.render import render_text_report
from pipesight.trace import io as trace_io


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("report", help="Analyze a captured trace and print recommendations")
    p.add_argument("trace", nargs="?", help="Path to a trace JSON file")
    p.add_argument(
        "--merge-dir",
        help="Directory of per-worker trace JSON files to merge and analyze together",
    )
    p.add_argument("--html", help="Also write a self-contained HTML timeline to this path")
    p.add_argument(
        "--physical-cores",
        type=int,
        help="Override detected physical core count used for the worker-count recommendation",
    )


def handle(args: argparse.Namespace) -> int:
    if args.merge_dir:
        paths = sorted(Path(args.merge_dir).glob("*.json"))
        if not paths:
            print(f"no .json trace files found in {args.merge_dir}")
            return 1
        trace = trace_io.merge(trace_io.load_many(paths))
    elif args.trace:
        trace = trace_io.load(args.trace)
    else:
        print("provide a trace file or --merge-dir")
        return 1

    stats = stage_stats(trace.spans)
    idle = gpu_idle_from_spans(trace.spans) if trace.spans else gpu_idle_from_samples(trace.samples)
    recs = build_recommendations(trace, physical_cores=args.physical_cores)

    print(render_text_report(stats=stats, idle=idle, recommendations=recs))

    if args.html:
        from pipesight.viz.html import render_trace_html

        Path(args.html).write_text(render_trace_html(trace))
        print(f"\nHTML timeline written to {args.html}")

    return 0
