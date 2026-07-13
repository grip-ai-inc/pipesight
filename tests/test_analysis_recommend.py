from pipesight.analysis.recommend import build_recommendations, rank, recommend_worker_count
from pipesight.trace.schema import Trace, TraceMeta


def _sequential_trace(span_factory, n=5):
    spans = []
    t = 0
    for i in range(n):
        spans.append(span_factory("decode", device="cpu", start_ns=t, dur_ns=30, item_id=i))
        t += 30
        spans.append(span_factory("gpu", device="gpu", start_ns=t, dur_ns=50, item_id=i))
        t += 50
        spans.append(span_factory("postprocess", device="cpu", start_ns=t, dur_ns=80, item_id=i))
        t += 80
    return spans


def test_recommend_worker_count_fires_when_cpu_bound_and_gpu_idle(span_factory):
    spans = _sequential_trace(span_factory)
    rec = recommend_worker_count(spans, physical_cores=4)
    assert rec is not None
    assert rec.kind == "increase_workers"
    assert "4" in rec.title


def test_recommend_worker_count_none_with_single_core(span_factory):
    spans = _sequential_trace(span_factory)
    assert recommend_worker_count(spans, physical_cores=1) is None


def test_build_recommendations_ranked_by_savings(span_factory):
    spans = _sequential_trace(span_factory)
    trace = Trace(meta=TraceMeta(cpu_count_physical=4), spans=spans)
    recs = build_recommendations(trace)
    assert len(recs) >= 1
    savings = [r.estimated_savings_ns for r in recs]
    assert savings == sorted(savings, reverse=True)


def test_rank_sorts_descending():
    from pipesight.analysis.recommend import Recommendation

    a = Recommendation(kind="gpu_underutilized", title="a", detail="", estimated_savings_ns=10)
    b = Recommendation(kind="gpu_underutilized", title="b", detail="", estimated_savings_ns=100)
    assert rank([a, b]) == [b, a]


def test_no_recommendations_for_well_utilized_trace(span_factory):
    # GPU busy almost the whole window -> no idle-driven recommendation
    spans = [
        span_factory("gpu", device="gpu", start_ns=0, dur_ns=1000, item_id=0),
    ]
    trace = Trace(meta=TraceMeta(cpu_count_physical=4), spans=spans)
    recs = build_recommendations(trace)
    assert recs == []
