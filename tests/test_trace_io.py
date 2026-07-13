from pipesight.trace import io as trace_io
from pipesight.trace.schema import Sample, TraceMeta


def test_save_load_roundtrip(tmp_path, span_factory, trace_factory):
    spans = [
        span_factory("decode", device="cpu", start_ns=0, dur_ns=50_000_000, item_id=0),
        span_factory(
            "slam", device="gpu", start_ns=50_000_000, dur_ns=200_000_000, item_id=0, note="x"
        ),
    ]
    trace = trace_factory(spans, source="marker", hostname="h1")
    trace.samples.append(Sample(ts_ns=0, cpu_percent=[10.0, 20.0], gpu_util_pct=5.0))

    path = tmp_path / "trace.json"
    trace_io.save(trace, path)
    loaded = trace_io.load(path)

    assert len(loaded.spans) == 2
    assert loaded.spans[0].name == "decode"
    assert loaded.spans[0].duration_ns == 50_000_000
    assert loaded.spans[1].args["note"] == "x"
    assert loaded.spans[1].item_id == 0
    assert loaded.meta.hostname == "h1"
    assert len(loaded.samples) == 1
    assert loaded.samples[0].gpu_util_pct == 5.0


def test_instant_event_roundtrip(tmp_path, span_factory, trace_factory):
    mark = span_factory("checkpoint", device="other", start_ns=42, dur_ns=0)
    trace = trace_factory([mark])
    path = tmp_path / "trace.json"
    trace_io.save(trace, path)
    loaded = trace_io.load(path)
    assert loaded.spans[0].start_ns == loaded.spans[0].end_ns == 42


def test_valid_chrome_trace_format(tmp_path, span_factory, trace_factory):
    trace = trace_factory([span_factory("decode")])
    path = tmp_path / "trace.json"
    trace_io.save(trace, path)
    import json

    data = json.loads(path.read_text())
    assert "traceEvents" in data
    event = data["traceEvents"][0]
    assert event["ph"] == "X"
    assert "ts" in event and "dur" in event


def test_merge_two_processes(span_factory):
    # process A: wall_start_epoch_s=100.0, perf_counter offset=0 -> span at t=0 is wall time 100.0s
    meta_a = TraceMeta(wall_start_epoch_s=100.0, perf_counter_offset_ns=0)
    span_a = span_factory("decode", start_ns=0, dur_ns=1_000_000_000, proc_id=1)
    from pipesight.trace.schema import Trace

    trace_a = Trace(meta=meta_a, spans=[span_a])

    # process B started 0.5s later in wall time, with its own unrelated perf_counter origin
    meta_b = TraceMeta(wall_start_epoch_s=100.5, perf_counter_offset_ns=500_000)
    span_b = span_factory("upload", start_ns=500_000, dur_ns=1_000_000_000, proc_id=2)
    trace_b = Trace(meta=meta_b, spans=[span_b])

    merged = trace_io.merge([trace_a, trace_b])
    assert len(merged.spans) == 2
    a = next(s for s in merged.spans if s.name == "decode")
    b = next(s for s in merged.spans if s.name == "upload")
    # b started 0.5s (5e8 ns) after a in wall-clock time
    assert abs((b.start_ns - a.start_ns) - 500_000_000) < 1_000
