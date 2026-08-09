"""Host-memory pressure analysis from quick-look samples.

This is the signal a CPU/GPU *profiler* was missing: GPU idle tells you the
GPU is starved, but not *why*. A dataloader that OOMs the host -- the classic
"DataLoader worker (pid N) is killed by signal: Killed" -- shows up here as
system RAM marching toward the ceiling and `proc_count` fanning out across
worker subprocesses. It also keeps the worker-count recommendation honest:
"add more workers" is wrong advice if there's no RAM headroom for them.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipesight.trace.schema import Sample

# At/above this fraction of total RAM in use, we call it "near the ceiling" --
# close enough that adding memory-hungry workers risks the OOM killer.
NEAR_CEILING_FRAC = 0.90


@dataclass
class MemoryReport:
    peak_sys_used_mb: float | None
    sys_total_mb: float | None
    peak_proc_rss_mb: float | None
    peak_proc_count: int | None
    samples_with_mem: int

    @property
    def has_data(self) -> bool:
        return self.samples_with_mem > 0 and self.sys_total_mb is not None

    @property
    def peak_used_frac(self) -> float | None:
        if not self.sys_total_mb or self.peak_sys_used_mb is None:
            return None
        return self.peak_sys_used_mb / self.sys_total_mb

    @property
    def peak_used_pct(self) -> float | None:
        frac = self.peak_used_frac
        return frac * 100.0 if frac is not None else None

    @property
    def headroom_mb(self) -> float | None:
        if self.sys_total_mb is None or self.peak_sys_used_mb is None:
            return None
        return max(0.0, self.sys_total_mb - self.peak_sys_used_mb)

    @property
    def near_ceiling(self) -> bool:
        frac = self.peak_used_frac
        return frac is not None and frac >= NEAR_CEILING_FRAC

    @property
    def rss_per_proc_mb(self) -> float | None:
        """Rough per-process RSS, for estimating how many more workers fit in
        the remaining headroom. Upper-bound-ish (shared pages over-count) and
        undefined until we've actually observed worker processes."""
        if not self.peak_proc_rss_mb or not self.peak_proc_count:
            return None
        return self.peak_proc_rss_mb / self.peak_proc_count


def memory_from_samples(samples: list[Sample]) -> MemoryReport:
    sys_used = [s.sys_mem_used_mb for s in samples if s.sys_mem_used_mb is not None]
    totals = [s.sys_mem_total_mb for s in samples if s.sys_mem_total_mb is not None]
    proc_rss = [s.proc_rss_mb for s in samples if s.proc_rss_mb is not None]
    proc_counts = [s.proc_count for s in samples if s.proc_count is not None]

    return MemoryReport(
        peak_sys_used_mb=max(sys_used) if sys_used else None,
        # Total RAM is fixed over a run; any observed value is representative.
        sys_total_mb=max(totals) if totals else None,
        peak_proc_rss_mb=max(proc_rss) if proc_rss else None,
        peak_proc_count=max(proc_counts) if proc_counts else None,
        samples_with_mem=len(sys_used),
    )
