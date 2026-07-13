import threading
import time

from pipesight.profiling.profiler import Profiler


def test_stage_records_span():
    prof = Profiler()
    with prof.stage("decode", device="cpu", item_id="seg1"):
        time.sleep(0.01)
    trace = prof.get_trace()
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.name == "decode"
    assert span.device == "cpu"
    assert span.item_id == "seg1"
    assert span.duration_ns >= 10_000_000 * 0.5  # sleep(0.01) elapsed, generously bounded


def test_stage_as_decorator():
    prof = Profiler()

    @prof.stage("hand_pose", device="cpu")
    def run(x):
        return x * 2

    assert run(3) == 6
    trace = prof.get_trace()
    assert trace.spans[0].name == "hand_pose"


def test_exception_still_records_span_and_propagates():
    prof = Profiler()
    try:
        with prof.stage("broken", device="cpu"):
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("exception should propagate")
    trace = prof.get_trace()
    assert trace.spans[0].args["error"].startswith("ValueError")


def test_thread_safety_concurrent_stages():
    prof = Profiler()

    def worker(i):
        with prof.stage("work", device="cpu", item_id=i):
            time.sleep(0.005)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    trace = prof.get_trace()
    assert len(trace.spans) == 20
    assert {s.item_id for s in trace.spans} == set(range(20))


def test_close_flushes_to_out_path(tmp_path):
    out_path = tmp_path / "trace.json"
    prof = Profiler(out_path=out_path)
    with prof.stage("decode"):
        pass
    prof.close()
    assert out_path.exists()

    # idempotent
    prof.close()


def test_mark_records_instant_event():
    prof = Profiler()
    prof.mark("checkpoint", note="halfway")
    trace = prof.get_trace()
    assert trace.spans[0].start_ns == trace.spans[0].end_ns
    assert trace.spans[0].args["note"] == "halfway"
