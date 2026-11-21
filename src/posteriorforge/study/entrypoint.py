"""Add study commands while preserving the sealed historical CLI source bytes."""
from __future__ import annotations

import argparse
import json
import sys


def _parser():
    from ..cli import _parser as historical_parser
    from .cli import configure_parser

    parser = historical_parser()
    # argparse exposes no public way to recover an existing subcommand action.
    # Keep this compatibility bridge here rather than altering a sealed runner.
    commands = next(action for action in parser._actions
                    if isinstance(action, argparse._SubParsersAction))
    configure_parser(commands)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["study"]:
        from .cli import run
        args = vars(_parser().parse_args(argv))
        args.pop("operation")
        print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False))
        return
    if argv in (["--help"], ["-h"]):
        _parser().parse_args(argv)
        return
    from ..cli import main as historical_main
    return historical_main(argv)


if __name__ == "__main__":
    main()
