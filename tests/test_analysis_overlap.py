from pipesight.analysis.overlap import detect_cross_iteration_overlap


def _sequential_trace(span_factory, n=5):
    """decode -> gpu -> postprocess per item, fully sequential across items,
    mirroring the naive grip-ai-shaped baseline."""
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


def test_detects_sequential_overlap_opportunity(span_factory):
    spans = _sequential_trace(span_factory)
    opps = detect_cross_iteration_overlap(spans)
    assert len(opps) == 1
    opp = opps[0]
    assert opp.tail_stage == "postprocess"
    assert opp.head_stage == "decode"
    assert opp.evidence_fraction == 1.0
    assert opp.estimated_savings_ns > 0


def test_no_opportunity_when_spans_already_overlap(span_factory):
    # next item's decode starts before this item's postprocess ends
    spans = [
        span_factory("decode", device="cpu", start_ns=0, dur_ns=30, item_id=0),
        span_factory("gpu", device="gpu", start_ns=30, dur_ns=50, item_id=0),
        span_factory("postprocess", device="cpu", start_ns=80, dur_ns=80, item_id=0),
        span_factory("decode", device="cpu", start_ns=90, dur_ns=30, item_id=1),  # overlaps item 0
        span_factory("gpu", device="gpu", start_ns=200, dur_ns=50, item_id=1),
        span_factory("postprocess", device="cpu", start_ns=250, dur_ns=80, item_id=1),
    ]
    opps = detect_cross_iteration_overlap(spans)
    assert opps == []


def test_no_item_id_returns_empty(span_factory):
    spans = [
        span_factory("decode", start_ns=0, dur_ns=10),
        span_factory("decode", start_ns=10, dur_ns=10),
    ]
    assert detect_cross_iteration_overlap(spans) == []


def test_gpu_gpu_pair_downgraded_to_note(span_factory):
    spans = []
    t = 0
    for i in range(4):
        spans.append(span_factory("gpu_a", device="gpu", start_ns=t, dur_ns=50, item_id=i))
        t += 50
        spans.append(span_factory("gpu_b", device="gpu", start_ns=t, dur_ns=50, item_id=i))
        t += 50
    opps = detect_cross_iteration_overlap(spans)
    assert len(opps) == 1
    assert opps[0].note != ""


def test_min_fraction_threshold(span_factory):
    # Only sequential in 1 of 3 pairs (33%) -- below default 80% threshold
    spans = [
        span_factory("a", device="cpu", start_ns=0, dur_ns=10, item_id=0),
        span_factory("a", device="cpu", start_ns=5, dur_ns=10, item_id=1),  # overlaps item 0
        span_factory("a", device="cpu", start_ns=6, dur_ns=10, item_id=2),  # overlaps item 1
        span_factory("a", device="cpu", start_ns=100, dur_ns=10, item_id=3),  # sequential, item 2
    ]
    assert detect_cross_iteration_overlap(spans) == []
