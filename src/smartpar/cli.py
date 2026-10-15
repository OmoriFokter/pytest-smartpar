"""Command-line tool ``smartpar``: maintenance of the method's data files.

Subcommands: ``merge`` (step 5, fold run files into history.json),
``report`` (step 5, metrics) and ``show`` (human-readable summary of groups,
polluters and the recommended number of workers).
"""

from __future__ import annotations

import argparse
import sys

from smartpar import __version__

_SUBCOMMANDS = {
    "merge": ("Merge runs/*.json into history.json (EMA).", 2),
    "report": ("Print metrics of recorded runs.", 5),
    "show": ("Show groups, polluters and the recommended number of workers.", 4),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartpar", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--dir", default=".smartpar", help="smartpar data directory")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _phase) in _SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _help, phase = _SUBCOMMANDS[args.command]
    print(
        f"smartpar {args.command}: not implemented yet (planned in phase {phase}).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
