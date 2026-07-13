"""`pipesight run -- <command...>` -- zero-touch quick-look profiling.

Implementation lands with the sampler in `pipesight.profiling.quicklook`
(Phase 4); this module only owns argument parsing so `pipesight --help`
lists a complete, stable CLI surface from early on.
"""

from __future__ import annotations

import argparse


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run", help="Profile an arbitrary command with no code changes (zero-touch)"
    )
    p.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds")
    p.add_argument("--out", default="trace.json", help="Output trace JSON path")
    p.add_argument("--no-gpu", action="store_true", help="Skip GPU utilization sampling")


def handle(args: argparse.Namespace, passthrough: list[str]) -> int:
    from pipesight.profiling.quicklook import run_quicklook

    run_quicklook(passthrough, interval_s=args.interval, out_path=args.out, use_gpu=not args.no_gpu)
    print(f"trace written to {args.out}")
    return 0
