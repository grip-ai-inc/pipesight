"""Zero-touch quick-look profiling: wrap an arbitrary command with no code
changes, producing a coarse CPU/GPU/memory utilization timeline. Has no stage
names, so it only feeds `analysis.idle`'s sample-based idle %, not
`analysis.overlap`'s cross-iteration detection -- that needs the opt-in
`Profiler` marker API's named, item_id-correlated spans instead.

`run_and_profile` also records *how the command ended* (exit code / terminating
signal) and, when asked to capture stderr, a bounded tail of it -- the raw
material `pipesight diagnose` turns into a failure explanation.
"""

from __future__ import annotations

import shlex
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from pipesight.profiling.cpu import logical_cpu_count, physical_cpu_count
from pipesight.profiling.gpu import gpu_name as get_gpu_name
from pipesight.profiling.sampler import SamplerThread
from pipesight.trace import io as trace_io
from pipesight.trace.schema import Trace, TraceMeta


@dataclass
class RunResult:
    trace: Trace
    exit_code: int | None  # normal exit status (>=0), or None if signal-killed
    term_signal: int | None  # positive signal number if killed by a signal
    stderr_tail: str | None  # bounded tail of stderr, only if capture_stderr


def _tee_stderr(pipe, buf: deque[str]) -> None:
    """Drain a child's stderr pipe: echo each line to our own stderr (so the
    run stays visually transparent) while keeping the last N lines in `buf`."""
    for line in pipe:
        sys.stderr.write(line)
        buf.append(line)
    sys.stderr.flush()


def run_and_profile(
    argv: list[str],
    *,
    interval_s: float = 0.2,
    out_path: str | Path | None,
    use_gpu: bool = True,
    sample_memory: bool = True,
    capture_stderr: bool = False,
    capture_lines: int = 200,
) -> RunResult:
    wall_start_epoch_s = time.time()
    perf_counter_offset_ns = time.perf_counter_ns()

    sampler = SamplerThread(interval_s=interval_s, use_gpu=use_gpu, sample_memory=sample_memory)
    sampler.start()

    stderr_buf: deque[str] = deque(maxlen=capture_lines)
    reader: threading.Thread | None = None
    # Only redirect stderr through a pipe when we actually intend to read it.
    # Teeing turns the child's stderr from a TTY into a pipe, which makes
    # progress bars (tqdm et al.) fall back to their non-interactive format --
    # an acceptable trade in `diagnose`, but not something to impose on a plain
    # profiling `run`, which inherits stderr untouched.
    stderr_dest = subprocess.PIPE if capture_stderr else None
    try:
        proc = subprocess.Popen(
            argv,
            stderr=stderr_dest,
            text=True if capture_stderr else None,
            bufsize=1 if capture_stderr else -1,
            errors="replace" if capture_stderr else None,
        )
        sampler.root_pid = proc.pid
        if capture_stderr and proc.stderr is not None:
            reader = threading.Thread(
                target=_tee_stderr, args=(proc.stderr, stderr_buf), daemon=True
            )
            reader.start()
        returncode = proc.wait()
    finally:
        if reader is not None:
            reader.join(timeout=5.0)
        samples = sampler.stop()

    # POSIX: a negative returncode means killed by signal -N.
    if returncode < 0:
        exit_code: int | None = None
        term_signal: int | None = -returncode
    else:
        exit_code = returncode
        term_signal = None
    stderr_tail = "".join(stderr_buf) if capture_stderr else None

    meta = TraceMeta(
        source="quicklook",
        hostname=socket.gethostname(),
        gpu_name=get_gpu_name() if use_gpu else None,
        cpu_count_logical=logical_cpu_count(),
        cpu_count_physical=physical_cpu_count(),
        command=shlex.join(argv),
        wall_start_epoch_s=wall_start_epoch_s,
        perf_counter_offset_ns=perf_counter_offset_ns,
        exit_code=exit_code,
        term_signal=term_signal,
        stderr_tail=stderr_tail,
    )
    trace = Trace(meta=meta, samples=samples)
    if out_path is not None:
        trace_io.save(trace, out_path)
    return RunResult(
        trace=trace, exit_code=exit_code, term_signal=term_signal, stderr_tail=stderr_tail
    )


def run_quicklook(
    argv: list[str],
    *,
    interval_s: float = 0.2,
    out_path: str | Path,
    use_gpu: bool = True,
) -> Trace:
    result = run_and_profile(
        argv, interval_s=interval_s, out_path=out_path, use_gpu=use_gpu, capture_stderr=False
    )
    if result.term_signal is not None:
        print(f"note: command was killed by signal {result.term_signal} (trace saved anyway)")
    elif result.exit_code:
        print(f"note: command exited with code {result.exit_code} (trace saved anyway)")
    return result.trace
