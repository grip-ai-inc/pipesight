"""Text rendering for `pipesight report`."""

from __future__ import annotations

from pipesight.analysis.idle import IdleReport
from pipesight.analysis.memory import MemoryReport
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


def fmt_mib(mb: float) -> str:
    """Adaptive MiB/GiB so small footprints don't render as a misleading '0.0 GiB'."""
    return f"{mb / 1024:.1f} GiB" if mb >= 1024 else f"{mb:.0f} MiB"


def render_memory(mem: MemoryReport) -> str:
    if not mem.has_data or mem.peak_used_pct is None:
        return "  (no host-memory samples in this trace)"
    line = (
        f"  Host RAM peak: {mem.peak_used_pct:.0f}% of {fmt_mib(mem.sys_total_mb or 0)} "
        f"({fmt_mib(mem.headroom_mb or 0)} free at peak)"
    )
    if mem.peak_proc_rss_mb is not None:
        line += (
            f"\n  Target process tree peak: {fmt_mib(mem.peak_proc_rss_mb)} RSS "
            f"across up to {mem.peak_proc_count} process(es)"
        )
    if mem.near_ceiling:
        line += (
            "\n  WARNING: host RAM approached the ceiling -- a dataloader/worker OOM "
            "('worker killed by signal: Killed') is likely here. Run `pipesight diagnose` "
            "for specifics."
        )
    return line


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
    memory: MemoryReport | None = None,
) -> str:
    parts = [
        "== Stage stats ==",
        render_stats_table(stats),
        "",
        "== GPU idle ==",
        render_idle(idle),
    ]
    if memory is not None and memory.has_data:
        parts += ["", "== Memory ==", render_memory(memory)]
    parts += [
        "",
        "== Recommendations ==",
        render_recommendations(recommendations),
    ]
    return "\n".join(parts)
