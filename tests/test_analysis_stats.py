from pipesight.analysis.stats import stage_stats, wall_clock_span


def test_stage_stats_counts_and_totals(span_factory):
    spans = [
        span_factory("decode", start_ns=0, dur_ns=10_000_000, item_id=0),
        span_factory("decode", start_ns=20_000_000, dur_ns=20_000_000, item_id=1),
    ]
    stats = stage_stats(spans)
    assert stats["decode"].count == 2
    assert stats["decode"].total_ns == 30_000_000
    assert stats["decode"].avg_ns == 15_000_000


def test_wall_share_pct_sums_close_to_100_for_sequential_spans(span_factory):
    spans = [
        span_factory("a", start_ns=0, dur_ns=10, item_id=0),
        span_factory("b", start_ns=10, dur_ns=10, item_id=0),
    ]
    stats = stage_stats(spans)
    assert abs(stats["a"].wall_share_pct + stats["b"].wall_share_pct - 100.0) < 1e-6


def test_wall_clock_span_empty():
    assert wall_clock_span([]) == 0


def test_p95_single_value(span_factory):
    stats = stage_stats([span_factory("a", dur_ns=100)])
    assert stats["a"].p95_ns == 100
