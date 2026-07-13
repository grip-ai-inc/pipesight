"""Text rendering for `pipesight report`."""

from __future__ import annotations

from pipesight.analysis.idle import IdleReport
from pipesight.analysis.recommend import Recommendation
from pipesight.analysis.stats import StageStats


def fmt_ns(ns: float) -> str:
    seconds = ns / 1e9
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    return f"{ns / 1e6:.1f}ms"


def render_stats_table(stats: dict[str, StageStats]) -> str:
    if not stats:
        return "  (no named stage spans -- only quick-look samples, no marker data)"
    header = (
        f"  {'stage':<20} {'device':<6} {'count':>6} {'total':>10} "
        f"{'avg':>10} {'p95':>10} {'wall%':>7}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for s in sorted(stats.values(), key=lambda s: s.total_ns, reverse=True):
        lines.append(
            f"  {s.name:<20} {s.device:<6} {s.count:>6} "
            f"{fmt_ns(s.total_ns):>10} {fmt_ns(s.avg_ns):>10} {fmt_ns(s.p95_ns):>10} "
            f"{s.wall_share_pct:>6.1f}%"
        )
    return "\n".join(lines)


def render_idle(idle: IdleReport) -> str:
    return (
        f"  GPU idle: {idle.idle_pct:.1f}% of {fmt_ns(idle.window_ns)} "
        f"({fmt_ns(idle.idle_ns)} idle) -- computed from {idle.source}"
    )


def render_recommendations(recs: list[Recommendation]) -> str:
    if not recs:
        return "  No recommendations -- GPU utilization looks reasonable, or not enough data."
    blocks = []
    for i, r in enumerate(recs, 1):
        blocks.append(
            f"  [{i}] {r.title}  (est. savings: {fmt_ns(r.estimated_savings_ns)})\n"
            f"      {r.detail}"
        )
    return "\n\n".join(blocks)


def render_text_report(
    *,
    stats: dict[str, StageStats],
    idle: IdleReport,
    recommendations: list[Recommendation],
) -> str:
    parts = [
        "== Stage stats ==",
        render_stats_table(stats),
        "",
        "== GPU idle ==",
        render_idle(idle),
        "",
        "== Recommendations ==",
        render_recommendations(recommendations),
    ]
    return "\n".join(parts)
