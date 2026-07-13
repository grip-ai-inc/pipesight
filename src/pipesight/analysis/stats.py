"""Per-stage-name aggregate statistics."""

from __future__ import annotations

from dataclasses import dataclass

from pipesight.trace.schema import Device, Span


@dataclass
class StageStats:
    name: str
    device: Device
    count: int
    total_ns: int
    avg_ns: float
    p50_ns: float
    p95_ns: float
    wall_share_pct: float


def _percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def wall_clock_span(spans: list[Span]) -> int:
    """Total wall-clock duration covered by `spans` (max end - min start)."""
    if not spans:
        return 0
    return max(s.end_ns for s in spans) - min(s.start_ns for s in spans)


def stage_stats(spans: list[Span]) -> dict[str, StageStats]:
    """One `StageStats` per distinct stage name. Spans sharing a name are
    assumed to share a device; if they don't, the first-seen device wins."""
    by_name: dict[str, list[Span]] = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)

    wall_ns = wall_clock_span(spans)
    result: dict[str, StageStats] = {}
    for name, group in by_name.items():
        durations = sorted(s.duration_ns for s in group)
        total = sum(durations)
        result[name] = StageStats(
            name=name,
            device=group[0].device,
            count=len(group),
            total_ns=total,
            avg_ns=total / len(group),
            p50_ns=_percentile(durations, 0.50),
            p95_ns=_percentile(durations, 0.95),
            wall_share_pct=(100.0 * total / wall_ns) if wall_ns else 0.0,
        )
    return result
