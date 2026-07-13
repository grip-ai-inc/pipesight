from __future__ import annotations

import pytest

from pipesight.trace.schema import Span, Trace, TraceMeta


def make_span(
    name: str,
    device: str = "cpu",
    start_ns: int = 0,
    dur_ns: int = 1_000_000,
    item_id: str | int | None = None,
    proc_id: int = 1000,
    thread_id: int = 1,
    **args,
) -> Span:
    return Span(
        name=name,
        device=device,
        start_ns=start_ns,
        end_ns=start_ns + dur_ns,
        proc_id=proc_id,
        thread_id=thread_id,
        item_id=item_id,
        args=dict(args),
    )


def make_trace(spans: list[Span], **meta_kwargs) -> Trace:
    return Trace(meta=TraceMeta(**meta_kwargs), spans=spans)


@pytest.fixture
def span_factory():
    return make_span


@pytest.fixture
def trace_factory():
    return make_trace
