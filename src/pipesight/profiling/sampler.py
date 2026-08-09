"""Background thread polling CPU+GPU utilization (and, by default, memory) at
a fixed interval, for zero-touch quick-look profiling (no code changes to the
target program)."""

from __future__ import annotations

import threading
import time

from pipesight.profiling.cpu import (
    sample_cpu_percent,
    sample_process_tree_rss_mb,
    sample_system_memory_mb,
)
from pipesight.profiling.gpu import sample_gpu
from pipesight.trace.schema import Sample


class SamplerThread:
    def __init__(
        self,
        interval_s: float = 0.2,
        use_gpu: bool = True,
        sample_memory: bool = True,
    ) -> None:
        self.interval_s = interval_s
        self.use_gpu = use_gpu
        self.sample_memory = sample_memory
        # The pid whose process tree (root + children) we attribute RSS to.
        # The caller wires this in *after* start() -- in zero-touch mode the
        # target process doesn't exist yet when the sampler spins up (it's
        # launched right after), so the loop reads this each iteration and
        # simply records no proc RSS until it's set. Assignment of a single
        # int is atomic under CPython's GIL, so no lock is needed.
        self.root_pid: int | None = None
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

            sys_used = sys_total = proc_rss = None
            proc_count = None
            if self.sample_memory:
                sys_used, sys_total = sample_system_memory_mb()
                root_pid = self.root_pid  # read once -- may be set concurrently
                if root_pid is not None:
                    tree = sample_process_tree_rss_mb(root_pid)
                    if tree is not None:
                        proc_rss, proc_count = tree

            self._samples.append(
                Sample(
                    ts_ns=ts,
                    cpu_percent=cpu,
                    gpu_util_pct=util,
                    gpu_mem_used_mb=mem,
                    sys_mem_used_mb=sys_used,
                    sys_mem_total_mb=sys_total,
                    proc_rss_mb=proc_rss,
                    proc_count=proc_count,
                )
            )
            self._stop_event.wait(self.interval_s)

    def stop(self) -> list[Sample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self._samples
