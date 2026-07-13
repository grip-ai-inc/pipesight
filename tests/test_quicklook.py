from __future__ import annotations

import sys
import time

import pytest

from pipesight.profiling.sampler import SamplerThread
from pipesight.trace import io as trace_io


def test_sampler_collects_samples_no_gpu():
    sampler = SamplerThread(interval_s=0.05, use_gpu=False)
    sampler.start()
    time.sleep(0.25)
    samples = sampler.stop()

    assert len(samples) >= 2
    for s in samples:
        assert isinstance(s.cpu_percent, list)
        assert len(s.cpu_percent) >= 1
        assert s.gpu_util_pct is None
        assert s.gpu_mem_used_mb is None


def test_sampler_stop_is_safe_without_start():
    sampler = SamplerThread(interval_s=0.05, use_gpu=False)
    assert sampler.stop() == []


@pytest.mark.slow
def test_run_quicklook_writes_trace(tmp_path):
    from pipesight.profiling.quicklook import run_quicklook

    out_path = tmp_path / "trace.json"
    argv = [sys.executable, "-c", "import time; time.sleep(0.3)"]

    trace = run_quicklook(argv, interval_s=0.05, out_path=out_path, use_gpu=False)

    assert out_path.exists()
    assert len(trace.samples) >= 3
    assert trace.meta.source == "quicklook"
    assert "time.sleep" in trace.meta.command

    reloaded = trace_io.load(out_path)
    assert len(reloaded.samples) == len(trace.samples)


@pytest.mark.slow
def test_run_quicklook_survives_nonzero_exit(tmp_path, capsys):
    from pipesight.profiling.quicklook import run_quicklook

    out_path = tmp_path / "trace.json"
    argv = [sys.executable, "-c", "import sys; sys.exit(1)"]

    run_quicklook(argv, interval_s=0.05, out_path=out_path, use_gpu=False)

    assert out_path.exists()
    assert "exited with code 1" in capsys.readouterr().out
