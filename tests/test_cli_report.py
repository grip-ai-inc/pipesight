from pipesight.cli.main import main
from pipesight.trace import io as trace_io
from pipesight.trace.schema import Trace, TraceMeta


def _write_trace(path, span_factory):
    spans = []
    t = 0
    for i in range(4):
        spans.append(span_factory("decode", device="cpu", start_ns=t, dur_ns=30, item_id=i))
        t += 30
        spans.append(span_factory("gpu", device="gpu", start_ns=t, dur_ns=50, item_id=i))
        t += 50
        spans.append(span_factory("postprocess", device="cpu", start_ns=t, dur_ns=80, item_id=i))
        t += 80
    trace = Trace(meta=TraceMeta(cpu_count_physical=4), spans=spans)
    trace_io.save(trace, path)
    return trace


def test_report_prints_output(tmp_path, span_factory, capsys):
    path = tmp_path / "trace.json"
    _write_trace(path, span_factory)

    rc = main(["report", str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Stage stats" in out
    assert "Recommendations" in out
    assert "decode" in out


def test_report_no_trace_errors(capsys):
    rc = main(["report"])
    assert rc == 1
    assert "provide a trace file" in capsys.readouterr().out


def test_report_merge_dir(tmp_path, span_factory, capsys):
    merge_dir = tmp_path / "traces"
    merge_dir.mkdir()
    _write_trace(merge_dir / "worker_1.json", span_factory)
    _write_trace(merge_dir / "worker_2.json", span_factory)

    rc = main(["report", "--merge-dir", str(merge_dir)])
    assert rc == 0
    assert "Stage stats" in capsys.readouterr().out


def test_report_missing_merge_dir_files(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    rc = main(["report", "--merge-dir", str(empty_dir)])
    assert rc == 1
    assert "no .json trace files found" in capsys.readouterr().out
