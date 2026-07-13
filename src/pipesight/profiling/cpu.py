"""CPU-side helpers backed by psutil."""

from __future__ import annotations

import psutil


def physical_cpu_count() -> int:
    return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1


def logical_cpu_count() -> int:
    return psutil.cpu_count(logical=True) or 1


def sample_cpu_percent(percpu: bool = True) -> list[float]:
    """Non-blocking (interval=None) percent-since-last-call sample.

    Must be primed with a throwaway call before the first meaningful sample
    -- see profiling/sampler.py, which owns the priming call.
    """
    result = psutil.cpu_percent(interval=None, percpu=percpu)
    return result if isinstance(result, list) else [result]
