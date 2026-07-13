"""End-to-end regression gate: runs the real synthetic_pipeline example
scripts as subprocesses and asserts the Pipeline retrofit actually improves
wall-clock time and GPU idle %, exactly per examples/synthetic_pipeline/README.md's
documented validation story. Slow (spins up real subprocesses + real CUDA
work) and skipped without a GPU, since the whole point is measuring genuine
GPU idle time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pipesight.analysis.idle import gpu_idle_from_spans
from pipesight.analysis.stats import wall_clock_span
from pipesight.trace import io as trace_io

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "synthetic_pipeline"


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not _cuda_available(), reason="requires a CUDA GPU"),
]


def test_pipeline_retrofit_improves_wall_time_and_idle(tmp_path):
    seq_out = tmp_path / "sequential_trace.json"
    pipe_out = tmp_path / "pipeline_trace.json"

    subprocess.run(
        [sys.executable, "run_sequential.py", "--n", "8", "--out", str(seq_out)],
        cwd=EXAMPLE_DIR,
        check=True,
    )
    subprocess.run(
        [sys.executable, "run_pipeline.py", "--n", "8", "--out", str(pipe_out)],
        cwd=EXAMPLE_DIR,
        check=True,
    )

    seq_trace = trace_io.load(seq_out)
    pipe_trace = trace_io.load(pipe_out)

    seq_wall = wall_clock_span(seq_trace.spans)
    pipe_wall = wall_clock_span(pipe_trace.spans)
    seq_idle = gpu_idle_from_spans(seq_trace.spans)
    pipe_idle = gpu_idle_from_spans(pipe_trace.spans)

    wall_improvement = (seq_wall - pipe_wall) / seq_wall
    assert wall_improvement >= 0.20, (
        f"expected >=20% wall-time improvement, got {wall_improvement:.1%} "
        f"(sequential={seq_wall / 1e9:.2f}s, pipeline={pipe_wall / 1e9:.2f}s)"
    )
    assert pipe_idle.idle_pct < seq_idle.idle_pct, (
        f"expected pipelined idle%% to drop, got {seq_idle.idle_pct:.1f}% -> "
        f"{pipe_idle.idle_pct:.1f}%"
    )
