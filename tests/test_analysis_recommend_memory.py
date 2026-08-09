from __future__ import annotations

from pipesight.analysis.memory import memory_from_samples
from pipesight.analysis.recommend import recommend_worker_count
from pipesight.trace.schema import Sample


def _cpu_bound_gpu_idle_spans(span_factory, n=5):
    spans = []
    t = 0
    for i in range(n):
        spans.append(span_factory("decode", device="cpu", start_ns=t, dur_ns=30, item_id=i))
        t += 30
        spans.append(span_factory("gpu", device="gpu", start_ns=t, dur_ns=50, item_id=i))
        t += 50
        spans.append(span_factory("post", device="cpu", start_ns=t, dur_ns=80, item_id=i))
        t += 80
    return spans


def _mem(used, total, rss=None, count=None):
    return memory_from_samples(
        [
            Sample(
                ts_ns=0,
                cpu_percent=[0.0],
                sys_mem_used_mb=used,
                sys_mem_total_mb=total,
                proc_rss_mb=rss,
                proc_count=count,
            )
        ]
    )


def test_worker_rec_warns_when_memory_near_ceiling(span_factory):
    spans = _cpu_bound_gpu_idle_spans(span_factory)
    mem = _mem(used=15400.0, total=16000.0, rss=14000.0, count=8)
    rec = recommend_worker_count(spans, physical_cores=8, memory=mem)

    assert rec is not None
    assert "cautiously" in rec.title.lower()
    assert "OOM" in rec.detail
    assert rec.evidence.get("peak_mem_pct") is not None


def test_worker_rec_notes_headroom_when_memory_comfortable(span_factory):
    spans = _cpu_bound_gpu_idle_spans(span_factory)
    mem = _mem(used=4000.0, total=32000.0, rss=2000.0, count=4)
    rec = recommend_worker_count(spans, physical_cores=8, memory=mem)

    assert rec is not None
    assert "cautiously" not in rec.title.lower()
    assert "headroom" in rec.detail.lower()


def test_worker_rec_unchanged_without_memory(span_factory):
    spans = _cpu_bound_gpu_idle_spans(span_factory)
    rec = recommend_worker_count(spans, physical_cores=8, memory=None)
    assert rec is not None
    assert rec.title.startswith("Increase parallel workers")
    assert "peak_mem_pct" not in rec.evidence
