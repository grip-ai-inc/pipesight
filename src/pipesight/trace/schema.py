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
    """One coarse CPU/GPU utilization sample from zero-touch quick-look polling."""

    ts_ns: int
    cpu_percent: list[float]
    gpu_util_pct: float | None = None
    gpu_mem_used_mb: float | None = None
    proc_id: int | None = None


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


@dataclass
class Trace:
    meta: TraceMeta
    spans: list[Span] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
