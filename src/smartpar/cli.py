"""Command-line tool ``smartpar``: maintenance of the method's data files.

Subcommands: ``merge`` (step 5, fold run files into history.json),
``report`` (step 5, metrics) and ``show`` (human-readable summary of groups,
polluters and the recommended number of workers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smartpar import __version__, history
from smartpar.models import SchemaError

# Subcommand -> (help, implementation phase) for those that do not work yet.
_UNIMPLEMENTED = {
    "report": ("Print metrics of recorded runs.", 5),
    "show": ("Show groups, polluters and the recommended number of workers.", 4),
}


def _alpha(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("alpha must be in (0, 1]")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartpar", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--dir", default=".smartpar", help="smartpar data directory")
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="Merge runs/*.json into history.json (EMA).")
    merge.add_argument(
        "--alpha",
        type=_alpha,
        default=history.DEFAULT_ALPHA,
        help=f"EMA smoothing factor (default: {history.DEFAULT_ALPHA})",
    )
    for name, (help_text, _phase) in _UNIMPLEMENTED.items():
        sub.add_parser(name, help=help_text)
    return parser


def cmd_merge(args: argparse.Namespace) -> int:
    try:
        result = history.merge(Path(args.dir), alpha=args.alpha)
    except SchemaError as exc:
        print(f"smartpar merge: {exc}", file=sys.stderr)
        return 1
    print(
        f"smartpar merge: merged {len(result.merged)} run(s), "
        f"moved {len(result.already_merged)} previously merged, "
        f"skipped {len(result.invalid)} invalid"
    )
    for path in result.invalid:
        print(f"  invalid run file left in place: {path}", file=sys.stderr)
    return 1 if result.invalid else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        return cmd_merge(args)
    _help, phase = _UNIMPLEMENTED[args.command]
    print(
        f"smartpar {args.command}: not implemented yet (planned in phase {phase}).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
