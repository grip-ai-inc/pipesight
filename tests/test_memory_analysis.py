from __future__ import annotations

from pipesight.analysis.memory import memory_from_samples
from pipesight.trace.schema import Sample


def _mem_sample(ts, used, total, rss=None, count=None):
    return Sample(
        ts_ns=ts,
        cpu_percent=[0.0],
        sys_mem_used_mb=used,
        sys_mem_total_mb=total,
        proc_rss_mb=rss,
        proc_count=count,
    )


def test_memory_from_samples_takes_peaks():
    samples = [
        _mem_sample(0, used=1000.0, total=16000.0, rss=500.0, count=2),
        _mem_sample(1, used=3000.0, total=16000.0, rss=1500.0, count=5),
        _mem_sample(2, used=2000.0, total=16000.0, rss=900.0, count=3),
    ]
    mem = memory_from_samples(samples)
    assert mem.has_data
    assert mem.peak_sys_used_mb == 3000.0
    assert mem.sys_total_mb == 16000.0
    assert mem.peak_proc_rss_mb == 1500.0
    assert mem.peak_proc_count == 5
    assert abs(mem.peak_used_pct - 18.75) < 1e-6
    assert mem.headroom_mb == 13000.0
    assert not mem.near_ceiling


def test_near_ceiling_flags_when_over_threshold():
    samples = [_mem_sample(0, used=15400.0, total=16000.0, rss=14000.0, count=8)]
    mem = memory_from_samples(samples)
    assert mem.near_ceiling  # 96.25% >= 90%
    assert mem.peak_used_pct > 90.0
    # rough per-process estimate used for "how many more workers fit"
    assert abs(mem.rss_per_proc_mb - 14000.0 / 8) < 1e-6


def test_no_memory_samples_reports_no_data():
    samples = [Sample(ts_ns=0, cpu_percent=[0.0], gpu_util_pct=10.0)]
    mem = memory_from_samples(samples)
    assert not mem.has_data
    assert mem.peak_used_pct is None
    assert mem.headroom_mb is None
    assert not mem.near_ceiling
    assert mem.rss_per_proc_mb is None
