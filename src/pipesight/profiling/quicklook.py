"""Zero-touch quick-look profiling: wrap an arbitrary command with no code
changes, producing a coarse CPU/GPU utilization timeline. Has no stage
names, so it only feeds `analysis.idle`'s sample-based idle %, not
`analysis.overlap`'s cross-iteration detection -- that needs the opt-in
`Profiler` marker API's named, item_id-correlated spans instead.
"""

from __future__ import annotations

import shlex
import socket
import subprocess
import time
from pathlib import Path

from pipesight.profiling.cpu import logical_cpu_count, physical_cpu_count
from pipesight.profiling.gpu import gpu_name as get_gpu_name
from pipesight.profiling.sampler import SamplerThread
from pipesight.trace import io as trace_io
from pipesight.trace.schema import Trace, TraceMeta


def run_quicklook(
    argv: list[str],
    *,
    interval_s: float = 0.2,
    out_path: str | Path,
    use_gpu: bool = True,
) -> Trace:
    wall_start_epoch_s = time.time()
    perf_counter_offset_ns = time.perf_counter_ns()

    sampler = SamplerThread(interval_s=interval_s, use_gpu=use_gpu)
    sampler.start()
    try:
        proc = subprocess.run(argv)
    finally:
        samples = sampler.stop()

    if proc.returncode != 0:
        print(f"note: command exited with code {proc.returncode} (trace saved anyway)")

    meta = TraceMeta(
        source="quicklook",
        hostname=socket.gethostname(),
        gpu_name=get_gpu_name() if use_gpu else None,
        cpu_count_logical=logical_cpu_count(),
        cpu_count_physical=physical_cpu_count(),
        command=shlex.join(argv),
        wall_start_epoch_s=wall_start_epoch_s,
        perf_counter_offset_ns=perf_counter_offset_ns,
    )
    trace = Trace(meta=meta, samples=samples)
    trace_io.save(trace, out_path)
    return trace
