"""GPU timing and utilization sampling.

`torch` and NVML (`pynvml`/`nvidia-ml-py`) are both optional, lazily-imported
dependencies. Nothing in pipesight's core marker API, analysis engine, or
Pipeline library requires either -- this module only adds precision upgrades
(CUDA-event stage timing, NVML utilization sampling) when they happen to be
installed and a CUDA GPU is actually present. A non-PyTorch or non-NVIDIA GPU
workload still gets correct wall-clock stage timing and correct "gpu" device
semantics for the analysis engine.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Literal

GpuTimingMode = Literal["auto", "cuda_event", "wall_clock", "none"]


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


_cuda_warmed_up = False


def warmup_cuda() -> None:
    """Force CUDA context initialization now, if torch+CUDA are available.

    The first-ever `torch.cuda` call in a process (e.g. `Event().record()`)
    lazily initializes the CUDA context, which can cost 1-3+ seconds. If
    that happens inside `GpuTimer`'s first use, it silently inflates that
    span's measured duration by however long context init took. Call this
    once eagerly (`Profiler.__init__` does, when `gpu_timing != "none"`) so
    that cost lands during startup, not inside the first timed GPU stage.
    """
    global _cuda_warmed_up
    if _cuda_warmed_up:
        return
    _cuda_warmed_up = True
    if not _torch_cuda_available():
        return
    import torch

    torch.cuda.init()
    # torch.cuda.init() alone still leaves ~200ms of lazy setup cost on the
    # first real Event()/record()/synchronize() cycle (observed empirically)
    # -- exercise that exact path once here so it's fully paid for upfront.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()


class GpuTimer:
    """Times one GPU stage span.

    mode="cuda_event" (or "auto" with torch+CUDA present): records
    torch.cuda.Event()s around the block and synchronizes on exit to capture
    actual device-busy time as `gpu_busy_ns`. Otherwise falls back to
    wall-clock perf_counter_ns() timing (still correct, just without the
    device-busy-vs-wall-time distinction CUDA events give).
    """

    def __init__(self, mode: GpuTimingMode = "auto") -> None:
        self._requested_mode = mode
        self._use_cuda_event = False
        self._start_event = None
        self._end_event = None
        self.start_ns = 0
        self.end_ns = 0
        self.gpu_busy_ns: int | None = None
        self.timing_method = "wall_clock"

    def __enter__(self) -> GpuTimer:
        self.start_ns = time.perf_counter_ns()
        if self._requested_mode in ("auto", "cuda_event") and _torch_cuda_available():
            import torch

            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
            self._use_cuda_event = True
        elif self._requested_mode == "cuda_event":
            raise RuntimeError("gpu_timing='cuda_event' requested but torch+CUDA is unavailable")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end_ns = time.perf_counter_ns()
        if self._use_cuda_event and exc_type is None:
            import torch

            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
            self.gpu_busy_ns = round(elapsed_ms * 1_000_000)
            self.timing_method = "cuda_event"
        else:
            self.timing_method = "wall_clock"


# ---------- utilization sampling (zero-touch quick-look mode) ----------

_nvml_handle = None
_nvml_init_failed = False


def _nvml_sample() -> tuple[float, float] | None:
    global _nvml_handle, _nvml_init_failed
    if _nvml_init_failed:
        return None
    try:
        import pynvml
    except ImportError:
        _nvml_init_failed = True
        return None
    try:
        if _nvml_handle is None:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        return float(util.gpu), mem.used / (1024 * 1024)
    except Exception:
        _nvml_init_failed = True
        return None


_smi_checked = False
_smi_available = False


def _nvidia_smi_sample() -> tuple[float, float] | None:
    global _smi_checked, _smi_available
    if not _smi_checked:
        _smi_available = shutil.which("nvidia-smi") is not None
        _smi_checked = True
    if not _smi_available:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        util_str, mem_str = out.stdout.strip().splitlines()[0].split(",")
        return float(util_str.strip()), float(mem_str.strip())
    except Exception:
        return None


def sample_gpu() -> tuple[float, float] | None:
    """Returns (util_pct, mem_used_mb), preferring NVML (fast, ~microseconds)
    over shelling out to nvidia-smi (~10-20ms/call). Returns None if neither
    is usable (no GPU, no driver access, headless box without permissions)."""
    return _nvml_sample() or _nvidia_smi_sample()


def gpu_name() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        return name.decode() if isinstance(name, bytes) else name
    except Exception:
        pass
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
    except Exception:
        return None
