from pipesight.analysis.idle import (
    gpu_busy_intervals,
    gpu_idle_from_samples,
    gpu_idle_from_spans,
    gpu_idle_gaps,
)
from pipesight.trace.schema import Sample


def test_gpu_busy_intervals_merges_overlapping(span_factory):
    spans = [
        span_factory("a", device="gpu", start_ns=0, dur_ns=100),
        span_factory("b", device="gpu", start_ns=50, dur_ns=100),  # overlaps a
    ]
    busy = gpu_busy_intervals(spans)
    assert len(busy) == 1
    assert busy[0].start_ns == 0
    assert busy[0].end_ns == 150


def test_gpu_idle_gaps_fully_idle_no_gpu_spans(span_factory):
    spans = [span_factory("decode", device="cpu", start_ns=0, dur_ns=100)]
    gaps = gpu_idle_gaps(spans)
    assert len(gaps) == 1
    assert gaps[0].duration_ns == 100


def test_gpu_idle_from_spans_matches_known_ratio(span_factory):
    # 100ns window, GPU busy for 40ns -> 60% idle
    spans = [
        span_factory("cpu_stage", device="cpu", start_ns=0, dur_ns=60),
        span_factory("gpu_stage", device="gpu", start_ns=60, dur_ns=40),
    ]
    report = gpu_idle_from_spans(spans)
    assert abs(report.idle_pct - 60.0) < 1e-6
    assert report.idle_ns == 60


def test_gpu_idle_from_samples_threshold():
    samples = [
        Sample(ts_ns=0, cpu_percent=[], gpu_util_pct=0.0),
        Sample(ts_ns=1, cpu_percent=[], gpu_util_pct=90.0),
        Sample(ts_ns=2, cpu_percent=[], gpu_util_pct=1.0),
        Sample(ts_ns=3, cpu_percent=[], gpu_util_pct=95.0),
    ]
    report = gpu_idle_from_samples(samples, threshold_pct=5.0)
    assert report.idle_pct == 50.0
    assert report.source == "samples"


def test_gpu_idle_from_samples_no_gpu_data():
    report = gpu_idle_from_samples([Sample(ts_ns=0, cpu_percent=[10.0], gpu_util_pct=None)])
    assert report.idle_pct == 0.0
