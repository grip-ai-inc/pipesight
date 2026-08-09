"""`pipesight diagnose` -- explain why a pipeline run failed.

Three ways in:
    pipesight diagnose -- python train.py     # run it (tees stderr), then diagnose
    pipesight diagnose run_trace.json         # diagnose a saved run/diagnose trace
    pipesight diagnose --log train.log        # diagnose an existing stderr/log file

The live-run form is the strongest: it captures the terminating signal, a tail
of stderr, AND pipesight's memory timeline, so an OOM can be corroborated rather
than guessed. `--log` is text-only unless paired with `--trace`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipesight.analysis.memory import memory_from_samples
from pipesight.diagnose.diagnose import DiagnoseContext, context_from_trace, diagnose
from pipesight.diagnose.render import render_diagnoses
from pipesight.trace import io as trace_io


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "diagnose",
        help="Explain why a pipeline run failed (dataloader/OOM/shm/fork triage)",
        usage=(
            "pipesight diagnose [-h] [--out OUT] [--interval INTERVAL] [--no-gpu]\n"
            "                          [--log LOG] [--trace TRACE] [trace] -- <command...>"
        ),
        description=(
            "Explain why a pipeline run failed. Provide one of:\n"
            "  -- <command...>   run the command (stderr is teed) and diagnose it live\n"
            "  <trace.json>      a trace saved by a previous `pipesight diagnose`/`run`\n"
            "  --log <file>      an existing stderr/training log to pattern-match\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("trace", nargs="?", help="Path to a saved trace JSON to diagnose")
    p.add_argument("--log", help="Path to an existing stderr/log file to diagnose (text-only)")
    p.add_argument(
        "--trace",
        dest="trace_for_log",
        help="Pair a --log with this trace's memory samples for OOM corroboration",
    )
    p.add_argument(
        "--out",
        help="For the live `-- <command>` form: also save the captured trace here",
    )
    p.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds")
    p.add_argument("--no-gpu", action="store_true", help="Skip GPU utilization sampling")


def _diagnose_live(args: argparse.Namespace, passthrough: list[str]) -> int:
    from pipesight.profiling.quicklook import run_and_profile

    result = run_and_profile(
        passthrough,
        interval_s=args.interval,
        out_path=args.out,
        use_gpu=not args.no_gpu,
        capture_stderr=True,
    )
    ctx = DiagnoseContext(
        stderr=result.stderr_tail or "",
        exit_code=result.exit_code,
        term_signal=result.term_signal,
        memory=memory_from_samples(result.trace.samples),
    )
    print()
    print(render_diagnoses(diagnose(ctx), failed=ctx.failed))
    if args.out:
        print(f"\ntrace written to {args.out}")
    # Mirror the child's failure in our own exit code so this composes in CI.
    return 1 if ctx.failed else 0


def _diagnose_trace(path: str) -> int:
    trace = trace_io.load(path)
    ctx = context_from_trace(trace)
    print(render_diagnoses(diagnose(ctx), failed=ctx.failed))
    return 0


def _diagnose_log(args: argparse.Namespace) -> int:
    text = Path(args.log).read_text(errors="replace")
    memory = None
    if args.trace_for_log:
        memory = memory_from_samples(trace_io.load(args.trace_for_log).samples)
    # A raw log has no exit-code/signal metadata; treat presence of a log as a
    # failed run so the fallback message doesn't wrongly claim success.
    ctx = DiagnoseContext(stderr=text, exit_code=1, term_signal=None, memory=memory)
    print(render_diagnoses(diagnose(ctx), failed=True))
    return 0


def handle(args: argparse.Namespace, passthrough: list[str] | None) -> int:
    if passthrough:
        return _diagnose_live(args, passthrough)
    if args.log:
        return _diagnose_log(args)
    if args.trace:
        return _diagnose_trace(args.trace)
    print(
        "provide something to diagnose: `-- <command>`, a trace JSON path, or `--log <file>`.\n"
        "See `pipesight diagnose --help`."
    )
    return 1
