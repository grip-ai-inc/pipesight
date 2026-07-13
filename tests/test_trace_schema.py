from pipesight.trace.schema import Span


def test_span_duration_ns():
    span = Span(
        name="decode", device="cpu", start_ns=1_000, end_ns=6_000, proc_id=1, thread_id=1
    )
    assert span.duration_ns == 5_000


def test_span_is_frozen():
    span = Span(name="decode", device="cpu", start_ns=0, end_ns=1, proc_id=1, thread_id=1)
    try:
        span.name = "other"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("Span should be immutable")
