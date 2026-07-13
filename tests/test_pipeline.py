from __future__ import annotations

import threading
import time

import pytest

from pipesight.pipeline import Pipeline, PipelineItemError, StageSpec


def _identity(x):
    return x


def test_ordered_results_match_input_order():
    stages = [StageSpec("double", lambda x: x * 2, workers=3)]
    with Pipeline(stages) as p:
        results = list(p.run(range(20)))
    assert results == [x * 2 for x in range(20)]


def test_unordered_mode_returns_all_results_regardless_of_order():
    def maybe_slow(x):
        if x == 0:
            time.sleep(0.05)  # first item finishes last
        return x

    stages = [StageSpec("f", maybe_slow, workers=4)]
    with Pipeline(stages, ordered=False) as p:
        results = list(p.run(range(10)))
    assert sorted(results) == list(range(10))


def test_multi_stage_overlap_faster_than_sequential():
    def slow(x):
        time.sleep(0.02)
        return x

    stages = [
        StageSpec("a", slow, workers=2),
        StageSpec("b", slow, workers=2),
        StageSpec("c", slow, workers=2),
    ]
    t0 = time.perf_counter()
    with Pipeline(stages) as p:
        results = list(p.run(range(12)))
    elapsed = time.perf_counter() - t0

    assert results == list(range(12))
    sequential_estimate = 12 * 3 * 0.02
    assert elapsed < sequential_estimate * 0.7  # meaningfully faster than fully sequential


def test_exception_raises_in_caller_iteration():
    def boom(x):
        if x == 3:
            raise ValueError("bad item")
        return x

    stages = [StageSpec("f", boom, workers=1)]
    collected = []
    with pytest.raises(PipelineItemError) as exc_info:
        with Pipeline(stages, on_error="raise") as p:
            for r in p.run(range(6)):
                collected.append(r)
    assert collected == [0, 1, 2]  # got everything before the failing item, in order
    assert exc_info.value.seq_id == 3
    assert exc_info.value.stage_name == "f"


def test_on_error_skip_drops_failed_items_and_continues():
    def boom(x):
        if x in (2, 4):
            raise ValueError("bad item")
        return x

    stages = [StageSpec("f", boom, workers=1)]
    with Pipeline(stages, on_error="skip") as p:
        results = list(p.run(range(6)))
    assert results == [0, 1, 3, 5]


def test_backpressure_bounds_in_flight_items():
    max_concurrent = 0
    current = 0
    lock = threading.Lock()

    def track(x):
        nonlocal max_concurrent, current
        with lock:
            current += 1
            max_concurrent = max(max_concurrent, current)
        time.sleep(0.02)
        with lock:
            current -= 1
        return x

    stages = [StageSpec("f", track, workers=2, max_queue=2)]
    with Pipeline(stages) as p:
        list(p.run(range(20)))

    # workers=2 caps true concurrency at 2; bounded queue keeps the feeder
    # from racing far ahead, so max_concurrent should stay small, not spike
    # to anywhere near 20.
    assert max_concurrent <= 4


def test_empty_input_yields_nothing():
    stages = [StageSpec("f", _identity)]
    with Pipeline(stages) as p:
        assert list(p.run([])) == []


def test_close_leaves_no_thread_leak():
    baseline = threading.active_count()

    def slow(x):
        time.sleep(0.005)
        return x

    stages = [StageSpec("a", slow, workers=2), StageSpec("b", slow, workers=3)]
    pipeline = Pipeline(stages)
    list(pipeline.run(range(10)))
    pipeline.close()

    deadline = time.time() + 2.0
    while threading.active_count() > baseline and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() == baseline


def test_close_is_idempotent():
    stages = [StageSpec("f", _identity)]
    pipeline = Pipeline(stages)
    list(pipeline.run(range(3)))
    pipeline.close()
    pipeline.close()  # must not raise


def test_run_after_close_raises():
    from pipesight.pipeline import PipelineClosedError

    stages = [StageSpec("f", _identity)]
    pipeline = Pipeline(stages)
    pipeline.close()
    with pytest.raises(PipelineClosedError):
        list(pipeline.run(range(3)))


def test_profiler_integration_records_spans_with_item_id():
    from pipesight import Profiler

    prof = Profiler()
    stages = [
        StageSpec("a", _identity, device="cpu", workers=2),
        StageSpec("b", _identity, device="gpu", workers=1),
    ]
    with Pipeline(stages, profiler=prof) as p:
        list(p.run(range(5)))

    trace = prof.get_trace()
    assert len(trace.spans) == 10  # 2 stages x 5 items
    names = {s.name for s in trace.spans}
    assert names == {"a", "b"}
    item_ids = {s.item_id for s in trace.spans if s.name == "a"}
    assert item_ids == set(range(5))


def test_no_stages_raises():
    with pytest.raises(ValueError):
        Pipeline([])
