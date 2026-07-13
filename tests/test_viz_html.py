from pipesight.trace.schema import Sample, Trace, TraceMeta
from pipesight.viz.html import render_compare_html, render_trace_html


def _trace_with_spans(span_factory):
    spans = [
        span_factory("decode", device="cpu", start_ns=0, dur_ns=30, item_id=0),
        span_factory("gpu", device="gpu", start_ns=30, dur_ns=50, item_id=0),
        span_factory("postprocess", device="cpu", start_ns=80, dur_ns=80, item_id=0),
    ]
    return Trace(meta=TraceMeta(cpu_count_physical=4), spans=spans)


def test_render_trace_html_contains_expected_pieces(span_factory):
    html = render_trace_html(_trace_with_spans(span_factory), title="my-trace")
    assert "<!doctype html>" in html.lower()
    assert "my-trace" in html
    assert "decode" in html  # embedded in the JSON blob
    assert "</html>" in html
    assert html.count("<script>") == html.count("</script>")


def test_render_trace_html_empty_spans_still_renders():
    trace = Trace(meta=TraceMeta(), spans=[], samples=[Sample(ts_ns=0, cpu_percent=[1.0])])
    html = render_trace_html(trace)
    assert "<!doctype html>" in html.lower()


def test_render_compare_html_has_two_panels_and_summary(span_factory):
    a = _trace_with_spans(span_factory)
    b = _trace_with_spans(span_factory)
    html = render_compare_html(a, b, title_a="seq", title_b="piped")
    assert '"title": "seq"' in html
    assert '"title": "piped"' in html
    assert "Wall-clock change" in html
    assert "GPU idle change" in html


def test_html_is_valid_looking_no_unclosed_tags(span_factory):
    import re

    html = render_trace_html(_trace_with_spans(span_factory))
    for tag in ["html", "head", "body", "style", "div", "script"]:
        opens = len(re.findall(rf"<{tag}[ >]", html))
        closes = html.count(f"</{tag}>")
        assert opens == closes, f"<{tag}> open/close mismatch: {opens} opens vs {closes} closes"
