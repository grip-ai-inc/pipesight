"""Opt-in precise stage-marker profiling: `with profiler.stage(name, device=...): ...`."""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from pipesight.profiling.cpu import logical_cpu_count, physical_cpu_count
from pipesight.profiling.gpu import GpuTimer, GpuTimingMode, warmup_cuda
from pipesight.trace import io as trace_io
from pipesight.trace.schema import Device, Span, Trace, TraceMeta


class _StageSpan(contextlib.ContextDecorator):
    """Returned by `Profiler.stage()`. Usable as a context manager or, via
    ContextDecorator, directly as a function decorator."""

    def __init__(
        self, profiler: Profiler, name: str, device: Device, item_id: str | int | None
    ) -> None:
        self._profiler = profiler
        self._name = name
        self._device = device
        self._item_id = item_id
        self._gpu_timer: GpuTimer | None = None
        self._start_ns = 0

    def __enter__(self) -> _StageSpan:
        self._start_ns = time.perf_counter_ns()
        if self._device == "gpu" and self._profiler.gpu_timing != "none":
            self._gpu_timer = GpuTimer(mode=self._profiler.gpu_timing)
            self._gpu_timer.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        end_ns = time.perf_counter_ns()
        args: dict[str, Any] = {}
        if self._gpu_timer is not None:
            self._gpu_timer.__exit__(exc_type, exc, tb)
            if self._gpu_timer.gpu_busy_ns is not None:
                args["gpu_busy_ns"] = self._gpu_timer.gpu_busy_ns
            args["timing_method"] = self._gpu_timer.timing_method
        if exc_type is not None:
            args["error"] = f"{exc_type.__name__}: {exc}"

        span = Span(
            name=self._name,
            device=self._device,
            start_ns=self._start_ns,
            end_ns=end_ns,
            proc_id=self._profiler.proc_id,
            thread_id=threading.get_ident(),
            item_id=self._item_id,
            worker_id=self._profiler.worker_id,
            args=args,
        )
        self._profiler._record(span)
        return False  # never suppress exceptions raised in the `with` block


class Profiler:
    """Records `Span`s for a single process.

    Not shared across a `ProcessPoolExecutor`: construct one per worker
    process (mirroring the existing `_init_worker`-style pattern of building
    per-process backends once and registering `atexit.register(profiler.close)`),
    each writing its own `out_path`, then combine with
    `pipesight.trace.io.merge()` / `pipesight report --merge-dir`.
    """

    def __init__(
        self,
        out_path: str | Path | None = None,
        *,
        gpu_timing: GpuTimingMode = "auto",
        worker_id: str | None = None,
        flush_on_close: bool = True,
    ) -> None:
        self.out_path = Path(out_path) if out_path is not None else None
        self.gpu_timing = gpu_timing
        self.worker_id = worker_id
        self.flush_on_close = flush_on_close
        self.proc_id = os.getpid()

        self._spans: list[Span] = []
        self._lock = threading.Lock()
        self._closed = False
        self._wall_start_epoch_s = time.time()
        self._perf_counter_offset_ns = time.perf_counter_ns()

        if self.gpu_timing != "none":
            # Pay CUDA context-init cost now, not inside the first timed
            # device="gpu" stage -- see gpu.warmup_cuda()'s docstring.
            warmup_cuda()

    def stage(
        self, name: str, *, device: Device = "cpu", item_id: str | int | None = None
    ) -> _StageSpan:
        return _StageSpan(self, name, device, item_id)

    def mark(self, name: str, **args: Any) -> None:
        """Record an instantaneous (zero-duration) event, e.g. for a
        one-off milestone that isn't a timed stage."""
        ts = time.perf_counter_ns()
        span = Span(
            name=name,
            device="other",
            start_ns=ts,
            end_ns=ts,
            proc_id=self.proc_id,
            thread_id=threading.get_ident(),
            worker_id=self.worker_id,
            args=dict(args),
        )
        self._record(span)

    def _record(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)

    def get_trace(self) -> Trace:
        with self._lock:
            spans = list(self._spans)
        meta = TraceMeta(
            source="marker",
            hostname=socket.gethostname(),
            cpu_count_logical=logical_cpu_count(),
            cpu_count_physical=physical_cpu_count(),
            wall_start_epoch_s=self._wall_start_epoch_s,
            perf_counter_offset_ns=self._perf_counter_offset_ns,
        )
        return Trace(meta=meta, spans=spans)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.flush_on_close and self.out_path is not None:
            trace_io.save(self.get_trace(), self.out_path)

    def __enter__(self) -> Profiler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
