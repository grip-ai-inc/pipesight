"""`pipesight compare <a.json> <b.json>` -- before/after A/B comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipesight.analysis.idle import gpu_idle_from_samples, gpu_idle_from_spans
from pipesight.analysis.stats import stage_stats, wall_clock_span
from pipesight.cli.render import fmt_ns
from pipesight.trace import io as trace_io
from pipesight.trace.schema import Trace


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "compare", help="Compare two traces (e.g. before/after a Pipeline retrofit)"
    )
    p.add_argument("trace_a", help="Baseline trace JSON")
    p.add_argument("trace_b", help="Comparison trace JSON")
    p.add_argument("--html", help="Also write a self-contained HTML side-by-side diff to this path")


def _summarize(trace: Trace) -> dict[str, Any]:
    spans = trace.spans
    wall_ns = wall_clock_span(spans) if spans else 0
    idle = gpu_idle_from_spans(spans) if spans else gpu_idle_from_samples(trace.samples)
    return {"wall_ns": wall_ns, "idle_pct": idle.idle_pct, "stats": stage_stats(spans)}


def render_comparison(name_a: str, a: dict[str, Any], name_b: str, b: dict[str, Any]) -> str:
    lines = [f"{'':20} {name_a:>18} {name_b:>18} {'delta':>12}", "-" * 72]

    wall_delta_pct = 100.0 * (b["wall_ns"] - a["wall_ns"]) / a["wall_ns"] if a["wall_ns"] else 0.0
    lines.append(
        f"{'wall-clock':20} {fmt_ns(a['wall_ns']):>18} {fmt_ns(b['wall_ns']):>18} "
        f"{wall_delta_pct:>+11.1f}%"
    )
    lines.append(
        f"{'GPU idle %':20} {a['idle_pct']:>17.1f}% {b['idle_pct']:>17.1f}% "
        f"{b['idle_pct'] - a['idle_pct']:>+11.1f}pp"
    )

    all_names = sorted(set(a["stats"]) | set(b["stats"]))
    if all_names:
        lines.append("")
        lines.append("per-stage total time:")
        for name in all_names:
            sa = a["stats"].get(name)
            sb = b["stats"].get(name)
            ta = fmt_ns(sa.total_ns) if sa else "-"
            tb = fmt_ns(sb.total_ns) if sb else "-"
            lines.append(f"  {name:18} {ta:>18} {tb:>18}")

    return "\n".join(lines)


def handle(args: argparse.Namespace) -> int:
    trace_a = trace_io.load(args.trace_a)
    trace_b = trace_io.load(args.trace_b)
    a = _summarize(trace_a)
    b = _summarize(trace_b)

    print(render_comparison(Path(args.trace_a).name, a, Path(args.trace_b).name, b))

    if args.html:
        from pipesight.viz.html import render_compare_html

        Path(args.html).write_text(render_compare_html(trace_a, trace_b))
        print(f"\nHTML comparison written to {args.html}")

    return 0
