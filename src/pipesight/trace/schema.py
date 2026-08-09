"""In-memory representation of a captured CPU/GPU pipeline trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Device = Literal["cpu", "gpu", "other"]
TraceSource = Literal["marker", "quicklook", "torch_profiler"]


@dataclass(frozen=True)
class Span:
    """One stage execution: a named, timed unit of work on a CPU or GPU."""

    name: str
    device: Device
    start_ns: int
    end_ns: int
    proc_id: int
    thread_id: int
    item_id: str | int | None = None
    worker_id: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class Sample:
    """One coarse CPU/GPU utilization sample from zero-touch quick-look polling.

    Memory fields are populated when the sampler is asked to track it (the
    default for `pipesight run`/`diagnose`). `sys_mem_*` is host-wide RAM;
    `proc_rss_mb`/`proc_count` cover the *target command's whole process
    tree* (root + recursive children), so DataLoader/worker subprocesses are
    included -- that's what makes an approaching-OOM kill visible before it
    happens.
    """

    ts_ns: int
    cpu_percent: list[float]
    gpu_util_pct: float | None = None
    gpu_mem_used_mb: float | None = None
    proc_id: int | None = None
    sys_mem_used_mb: float | None = None
    sys_mem_total_mb: float | None = None
    proc_rss_mb: float | None = None
    proc_count: int | None = None


@dataclass
class TraceMeta:
    schema_version: int = 1
    source: TraceSource = "marker"
    hostname: str = ""
    gpu_name: str | None = None
    cpu_count_logical: int = 0
    cpu_count_physical: int = 0
    command: str | None = None
    wall_start_epoch_s: float = 0.0
    perf_counter_offset_ns: int = 0
    # Populated for `pipesight run`/`diagnose` traces: how the target command
    # ended. `term_signal` is the positive signal number when the process was
    # killed by a signal (e.g. 9 = SIGKILL, the OOM killer's weapon of
    # choice), else None. `stderr_tail` holds a bounded tail of the captured
    # stderr (only when stderr was teed, i.e. `diagnose -- ...`).
    exit_code: int | None = None
    term_signal: int | None = None
    stderr_tail: str | None = None


@dataclass
class Trace:
    meta: TraceMeta
    spans: list[Span] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
