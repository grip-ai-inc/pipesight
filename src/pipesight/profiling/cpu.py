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


_MIB = 1024 * 1024  # match gpu.py's MB-means-MiB convention


def sample_system_memory_mb() -> tuple[float, float]:
    """(used, total) host RAM in MiB. `used` here is total - available, i.e.
    what psutil considers genuinely in use (excludes reclaimable cache), which
    is the number that actually matters for how close you are to the OOM
    killer."""
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / _MIB, vm.total / _MIB


def sample_process_tree_rss_mb(root_pid: int) -> tuple[float, int] | None:
    """(summed RSS in MiB, process count) over `root_pid` and all of its
    recursive children. Returns None if the root process is already gone.

    Summing RSS across a process tree over-counts shared pages (each child
    that forked from the parent double-counts shared code/read-only data), so
    treat this as an upper bound on real memory pressure, not an exact figure
    -- but it's the right shape for spotting worker fan-out (`proc_count`
    climbing) and a tree marching toward the RAM ceiling.
    """
    try:
        root = psutil.Process(root_pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return None

    total_rss = 0.0
    count = 0
    for p in procs:
        try:
            total_rss += p.memory_info().rss
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue  # process vanished or unreadable mid-walk -- skip it
    if count == 0:
        return None
    return total_rss / _MIB, count
