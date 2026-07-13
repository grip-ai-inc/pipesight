"""GPU idle-gap detection, from named spans (marker mode) or coarse
utilization samples (zero-touch quick-look mode, which has no span names)."""

from __future__ import annotations

from dataclasses import dataclass

from pipesight.trace.schema import Sample, Span


@dataclass
class Interval:
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv.start_ns)
    merged = [ordered[0]]
    for iv in ordered[1:]:
        last = merged[-1]
        if iv.start_ns <= last.end_ns:
            merged[-1] = Interval(last.start_ns, max(last.end_ns, iv.end_ns))
        else:
            merged.append(iv)
    return merged


def gpu_busy_intervals(spans: list[Span]) -> list[Interval]:
    gpu_spans = [Interval(s.start_ns, s.end_ns) for s in spans if s.device == "gpu"]
    return _merge_intervals(gpu_spans)


def gpu_idle_gaps(spans: list[Span]) -> list[Interval]:
    """Idle gaps within the overall wall-clock window covered by `spans`."""
    if not spans:
        return []
    window_start = min(s.start_ns for s in spans)
    window_end = max(s.end_ns for s in spans)
    busy = gpu_busy_intervals(spans)
    if not busy:
        return [Interval(window_start, window_end)]

    gaps: list[Interval] = []
    cursor = window_start
    for iv in busy:
        if iv.start_ns > cursor:
            gaps.append(Interval(cursor, iv.start_ns))
        cursor = max(cursor, iv.end_ns)
    if cursor < window_end:
        gaps.append(Interval(cursor, window_end))
    return gaps


@dataclass
class IdleReport:
    idle_pct: float
    idle_ns: int
    window_ns: int
    source: str  # "spans" | "samples"


def gpu_idle_from_spans(spans: list[Span]) -> IdleReport:
    if not spans:
        return IdleReport(idle_pct=0.0, idle_ns=0, window_ns=0, source="spans")
    window_ns = max(s.end_ns for s in spans) - min(s.start_ns for s in spans)
    idle_ns = sum(g.duration_ns for g in gpu_idle_gaps(spans))
    idle_pct = (100.0 * idle_ns / window_ns) if window_ns else 0.0
    return IdleReport(idle_pct=idle_pct, idle_ns=idle_ns, window_ns=window_ns, source="spans")


def gpu_idle_from_samples(samples: list[Sample], threshold_pct: float = 5.0) -> IdleReport:
    """For zero-touch traces, which have no named spans: idle % is the
    fraction of polled samples below `threshold_pct` GPU utilization."""
    gpu_samples = [s for s in samples if s.gpu_util_pct is not None]
    if not gpu_samples:
        return IdleReport(idle_pct=0.0, idle_ns=0, window_ns=0, source="samples")
    idle_count = sum(1 for s in gpu_samples if s.gpu_util_pct < threshold_pct)
    idle_pct = 100.0 * idle_count / len(gpu_samples)
    window_ns = max(s.ts_ns for s in gpu_samples) - min(s.ts_ns for s in gpu_samples)
    idle_ns = round(window_ns * idle_pct / 100.0)
    return IdleReport(idle_pct=idle_pct, idle_ns=idle_ns, window_ns=window_ns, source="samples")
