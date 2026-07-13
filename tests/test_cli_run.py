from __future__ import annotations

import sys

import pytest

from pipesight.cli.main import main


def test_run_requires_passthrough_command(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--out", "trace.json"])
    assert "requires a command" in capsys.readouterr().err


@pytest.mark.slow
def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "trace.json"
    rc = main(
        [
            "run",
            "--interval",
            "0.05",
            "--out",
            str(out),
            "--no-gpu",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(0.2)",
        ]
    )
    assert rc == 0
    assert out.exists()
