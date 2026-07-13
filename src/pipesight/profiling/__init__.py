"""Profiling: the opt-in marker API (`Profiler`) and zero-touch quick-look sampling."""

from pipesight.profiling.profiler import Profiler
from pipesight.trace.schema import Device

_default_profiler: Profiler | None = None


def _default() -> Profiler:
    global _default_profiler
    if _default_profiler is None:
        _default_profiler = Profiler()
    return _default_profiler


def stage(name: str, *, device: Device = "cpu", item_id: str | int | None = None):
    """Convenience wrapper around a lazily-created module-level `Profiler`,
    for quick scripts. Prefer constructing your own `Profiler` explicitly in
    anything that spans multiple processes (e.g. `ProcessPoolExecutor`
    workers) -- this default instance is per-process already (module state
    isn't shared across processes), but sharing *this* singleton across
    callers that expect independent `out_path`s is a footgun; each worker
    should build its own named `Profiler` instead.
    """
    return _default().stage(name, device=device, item_id=item_id)


__all__ = ["Profiler", "stage"]
