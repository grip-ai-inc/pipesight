"""Background thread polling CPU+GPU utilization at a fixed interval, for
zero-touch quick-look profiling (no code changes to the target program)."""

from __future__ import annotations

import threading
import time

from pipesight.profiling.cpu import sample_cpu_percent
from pipesight.profiling.gpu import sample_gpu
from pipesight.trace.schema import Sample


class SamplerThread:
    def __init__(self, interval_s: float = 0.2, use_gpu: bool = True) -> None:
        self.interval_s = interval_s
        self.use_gpu = use_gpu
        self._samples: list[Sample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # psutil.cpu_percent(interval=None) reports usage *since the last
        # call* -- the very first call has no baseline and returns a
        # meaningless 0.0/[0.0, ...]. Prime it here so the first sample
        # collected in _run() is real.
        sample_cpu_percent()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ts = time.perf_counter_ns()
            cpu = sample_cpu_percent()
            gpu = sample_gpu() if self.use_gpu else None
            util, mem = gpu if gpu is not None else (None, None)
            self._samples.append(
                Sample(ts_ns=ts, cpu_percent=cpu, gpu_util_pct=util, gpu_mem_used_mb=mem)
            )
            self._stop_event.wait(self.interval_s)

    def stop(self) -> list[Sample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self._samples
