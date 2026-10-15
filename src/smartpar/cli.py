"""Command-line tool ``smartpar``: maintenance of the method's data files.

Subcommands: ``merge`` (step 5, fold run files into history.json),
``report`` (step 5, metrics) and ``show`` (human-readable summary of groups,
polluters and the recommended number of workers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smartpar import __version__, history, scheduler
from smartpar.models import SchemaError
from smartpar.profiler import CACHE_GROWTH, load_dependencies

# Subcommand -> (help, implementation phase) for those that do not work yet.
_UNIMPLEMENTED = {
    "report": ("Print metrics of recorded runs.", 5),
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
    show = sub.add_parser(
        "show", help="Show groups, polluters and the recommended number of workers."
    )
    show.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="upper bound for the recommended m (default: available CPUs)",
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


def _short(nodeid: str, width: int = 70) -> str:
    return nodeid if len(nodeid) <= width else "..." + nodeid[-(width - 3) :]


def cmd_show(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    try:
        hist = history.load_history(directory)
        flaky = history.load_flaky(directory)
    except SchemaError as exc:
        print(f"smartpar show: {exc}", file=sys.stderr)
        return 1
    deps = load_dependencies(directory)
    tests = set(hist.tests) | (set(deps.file_access) | set(deps.state_changes) if deps else set())
    if not tests:
        print(f"smartpar show: no data in {directory}; run pytest --smartpar-profile first")
        return 1
    max_workers = args.max_workers or scheduler.available_cpus()
    a = scheduler.analyze(tests, hist, deps, flaky, max_workers)
    est, graph = a.estimates, a.graph

    timed = len(tests) - len(est.estimated) - len(est.new_tests)
    print(
        f"tests: {len(tests)}  (timed: {timed}, estimated with median {est.median_s:.3f}s: "
        f"{len(est.estimated)}, new: {len(est.new_tests)})"
    )
    if not est.has_timings:
        print(
            "  no measured durations yet: all estimates are equal; "
            "the first `pytest -n auto --smartpar` run collects them"
        )
    print(f"file conflicts: {len(graph.conflicts)}")
    print(f"order edges (confirmed): {len(graph.order_edges)}")

    print(f"\ngroups ({len(graph.groups)}):")
    for g in sorted(graph.groups, key=lambda g: (-g.est_duration, g.id)):
        flag = "  OVERSIZED" if g.id in a.schedule.oversized_groups else ""
        print(f"  {g.id}: {len(g.tests)} tests, {g.est_duration:.3f}s, {'+'.join(g.reasons)}{flag}")
        for nodeid in g.order:
            print(f"      {_short(nodeid)}")

    print(f"\nisolated tail ({len(graph.tail)}):")
    for nodeid in graph.tail:
        mark = " (order cycle)" if nodeid in graph.cyclic else ""
        print(f"  {_short(nodeid)}{mark}")

    print("\npolluters:")
    confirmed = sorted({e.after for e in graph.order_edges if e.reason == "victim-before-polluter"})
    print(
        f"  confirmed by detector: {len(confirmed)}"
        if flaky
        else "  confirmed by detector: - (run pytest --smartpar-detect)"
    )
    for nodeid in confirmed:
        print(f"    {_short(nodeid)}")
    changes = deps.state_changes if deps else {}
    strong = {t: k for t, k in changes.items() if any(not x.startswith(CACHE_GROWTH) for x in k)}
    cache_only = len(changes) - len(strong)
    print(
        f"  candidates (unrestored state changes): {len(strong)}, only cache growth: {cache_only}"
    )
    for nodeid, keys in sorted(strong.items()):
        shown = [k for k in keys if not k.startswith(CACHE_GROWTH)]
        print(f"    {_short(nodeid)}: {', '.join(shown)}")

    model = hist.overhead
    fit = f"fitted on m={model.fit_points}" if model.fit_points else "no measurements yet"
    print(f"\npredicted T(m), s (overhead {model.a:.3f} + {model.b:.3f}*m, {fit}):")
    for m, t in a.curve.items():
        mark = "  <- recommended" if m == a.recommended_workers else ""
        print(f"  m={m}: {t:.3f}{mark}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        return cmd_merge(args)
    if args.command == "show":
        return cmd_show(args)
    _help, phase = _UNIMPLEMENTED[args.command]
    print(
        f"smartpar {args.command}: not implemented yet (planned in phase {phase}).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
