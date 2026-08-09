from __future__ import annotations

import sys

import pytest

from pipesight.cli.main import main


def test_diagnose_no_input_errors(capsys):
    rc = main(["diagnose"])
    assert rc == 1
    assert "provide something to diagnose" in capsys.readouterr().out


def test_diagnose_log_file(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text(
        "Epoch 0\n"
        "RuntimeError: DataLoader worker (pid 4242) is killed by signal: Killed.\n"
    )
    rc = main(["diagnose", "--log", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OOM" in out
    assert "num_workers" in out


def test_diagnose_log_cuda_fork(tmp_path, capsys):
    log = tmp_path / "err.log"
    log.write_text("RuntimeError: Cannot re-initialize CUDA in forked subprocess.\n")
    rc = main(["diagnose", "--log", str(log)])
    assert rc == 0
    assert "spawn" in capsys.readouterr().out


@pytest.mark.slow
def test_diagnose_live_run_captures_and_diagnoses(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "diag.json"
    # Child prints the classic worker-killed message to stderr and exits 1.
    script = (
        "import sys; "
        "sys.stderr.write('RuntimeError: DataLoader worker (pid 999) "
        "is killed by signal: Killed.\\n'); "
        "sys.exit(1)"
    )
    rc = main(
        [
            "diagnose",
            "--interval",
            "0.05",
            "--no-gpu",
            "--out",
            str(out),
            "--",
            sys.executable,
            "-c",
            script,
        ]
    )
    # exit code mirrors the failed child
    assert rc == 1
    printed = capsys.readouterr().out
    assert "OOM" in printed
    assert out.exists()

    # The saved trace is independently re-diagnosable.
    capsys.readouterr()
    rc2 = main(["diagnose", str(out)])
    assert rc2 == 0
    assert "OOM" in capsys.readouterr().out
