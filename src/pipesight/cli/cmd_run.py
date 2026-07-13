"""`pipesight run -- <command...>` -- zero-touch quick-look profiling."""

from __future__ import annotations

import argparse


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    # The `-- <command...>` passthrough is split off from argv by
    # cli.main._split_run_passthrough() before argparse ever sees it (see
    # that function's docstring), so argparse itself doesn't know about it
    # and won't render it in the auto-generated usage line -- spell it out
    # explicitly here instead, or `--help` looks like `run` takes no
    # positional command at all.
    p = subparsers.add_parser(
        "run",
        help="Profile an arbitrary command with no code changes (zero-touch)",
        usage="pipesight run [-h] [--interval INTERVAL] [--out OUT] [--no-gpu] -- <command...>",
        description=(
            "Profile an arbitrary command with no code changes (zero-touch).\n\n"
            "Everything after a literal `--` is run as the target command, e.g.:\n"
            "    pipesight run --out trace.json -- python my_script.py --arg1 val1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds")
    p.add_argument("--out", default="trace.json", help="Output trace JSON path")
    p.add_argument("--no-gpu", action="store_true", help="Skip GPU utilization sampling")


def handle(args: argparse.Namespace, passthrough: list[str]) -> int:
    from pipesight.profiling.quicklook import run_quicklook

    run_quicklook(passthrough, interval_s=args.interval, out_path=args.out, use_gpu=not args.no_gpu)
    print(f"trace written to {args.out}")
    return 0
