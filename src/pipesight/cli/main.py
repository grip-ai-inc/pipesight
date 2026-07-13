"""pipesight CLI entry point: `pipesight run|report|compare`."""

from __future__ import annotations

import argparse
import sys

from pipesight._version import __version__
from pipesight.cli import cmd_compare, cmd_report, cmd_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipesight")
    parser.add_argument("--version", action="version", version=f"pipesight {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    cmd_run.build_parser(subparsers)
    cmd_report.build_parser(subparsers)
    cmd_compare.build_parser(subparsers)
    subparsers.add_parser("version", help="Print the pipesight version")

    return parser


def _split_run_passthrough(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """`pipesight run -- <command...>`: argparse doesn't cleanly support
    "everything after a literal --" alongside a subcommand's own flags, so
    split it off manually before argparse ever sees it."""
    if "run" not in argv:
        return argv, None
    run_idx = argv.index("run")
    if "--" not in argv[run_idx:]:
        return argv, None
    sep_idx = argv.index("--", run_idx)
    return argv[:sep_idx], argv[sep_idx + 1 :]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    argv, passthrough = _split_run_passthrough(argv)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "version":
        print(f"pipesight {__version__}")
        return 0
    if args.command == "run":
        if not passthrough:
            parser.error(
                "pipesight run requires a command after '--', "
                "e.g. `pipesight run -- python script.py`"
            )
        return cmd_run.handle(args, passthrough)
    if args.command == "report":
        return cmd_report.handle(args)
    if args.command == "compare":
        return cmd_compare.handle(args)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
