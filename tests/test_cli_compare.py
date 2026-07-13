from pipesight.cli.main import main
from pipesight.trace import io as trace_io
from pipesight.trace.schema import Trace, TraceMeta


def _write_trace(path, span_factory, postprocess_dur=80):
    spans = []
    t = 0
    for i in range(4):
        spans.append(span_factory("decode", device="cpu", start_ns=t, dur_ns=30, item_id=i))
        t += 30
        spans.append(span_factory("gpu", device="gpu", start_ns=t, dur_ns=50, item_id=i))
        t += 50
        spans.append(
            span_factory("postprocess", device="cpu", start_ns=t, dur_ns=postprocess_dur, item_id=i)
        )
        t += postprocess_dur
    trace_io.save(Trace(meta=TraceMeta(cpu_count_physical=4), spans=spans), path)


def test_compare_prints_deltas(tmp_path, span_factory, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_trace(a, span_factory, postprocess_dur=80)
    _write_trace(b, span_factory, postprocess_dur=20)  # "improved" run

    rc = main(["compare", str(a), str(b)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wall-clock" in out
    assert "GPU idle" in out
    assert "postprocess" in out


def test_cli_top_level_help(capsys):
    rc = main([])
    assert rc == 1
    assert "pipesight" in capsys.readouterr().out


def test_cli_version(capsys):
    rc = main(["version"])
    assert rc == 0
    assert "pipesight" in capsys.readouterr().out
