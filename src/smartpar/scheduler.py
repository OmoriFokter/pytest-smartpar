"""Method step 3 (scheduling): LPT assignment of groups to m workers.

Graham's (1969) Longest Processing Time rule: units (groups; an independent
test is a group of one) are sorted by estimated duration, descending, and each
goes to the currently least loaded worker. The rule is deterministic: equal
durations are ordered by the unit's first nodeid, equally loaded workers by
worker number.

The isolated tail runs after all workers have finished, sequentially, so

    T(m) = LPT_makespan(m) + tail + overhead(m),   overhead(m) = worker_startup_s * m

(linear start-up model: one measured start-up cost per worker). The
recommended m is the smallest m with T(m) <= 1.05 * min_k T(k).
"""

from __future__ import annotations

import heapq
import logging
import os
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from smartpar.graph import GraphResult
from smartpar.graph import build as build_graph
from smartpar.models import Dependencies, Flaky, History

log = logging.getLogger("smartpar.scheduler")

SINGLE = "single"
# Duration assumed for every test when no test has a measured duration yet.
DEFAULT_DURATION_S = 1.0
RECOMMEND_TOLERANCE = 1.05


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def worker_name(index: int) -> str:
    return f"gw{index}"


# --- durations ----------------------------------------------------------------------


@dataclass
class Estimates:
    durations: dict[str, float]
    new_tests: list[str]  # never seen in history: unknown file access, isolate
    estimated: list[str]  # seen but never timed: duration = median of known
    median_s: float
    has_timings: bool


def estimate_durations(tests: Iterable[str], history: History) -> Estimates:
    known = {nodeid: h.duration_ema for nodeid, h in history.tests.items() if h.timed_runs > 0}
    median = statistics.median(known.values()) if known else DEFAULT_DURATION_S
    durations: dict[str, float] = {}
    new: list[str] = []
    estimated: list[str] = []
    for nodeid in sorted(set(tests)):
        if nodeid in known:
            durations[nodeid] = known[nodeid]
            continue
        durations[nodeid] = median
        (estimated if nodeid in history.tests else new).append(nodeid)
    return Estimates(durations, new, estimated, median, bool(known))


# --- LPT ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """What LPT places: a group (id g<N>) or a single test (id "single")."""

    id: str
    tests: tuple[str, ...]  # execution order
    duration: float

    @property
    def sort_key(self) -> tuple[float, str]:
        return (-self.duration, min(self.tests))


def units_from_graph(graph: GraphResult, durations: Mapping[str, float]) -> list[Unit]:
    units = [Unit(g.id, tuple(g.order), g.est_duration) for g in graph.groups]
    units += [Unit(SINGLE, (t,), durations.get(t, 0.0)) for t in graph.singles]
    return units


@dataclass
class LptResult:
    assignments: list[list[Unit]]
    loads: list[float]

    @property
    def makespan(self) -> float:
        return max(self.loads, default=0.0)


def lpt(units: Iterable[Unit], workers: int) -> LptResult:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    assignments: list[list[Unit]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    heap = [(0.0, i) for i in range(workers)]
    for unit in sorted(units, key=lambda u: u.sort_key):
        load, i = heapq.heappop(heap)
        assignments[i].append(unit)
        loads[i] = load + unit.duration
        heapq.heappush(heap, (loads[i], i))
    return LptResult(assignments, loads)


# --- plan ---------------------------------------------------------------------------


@dataclass
class Schedule:
    workers: int
    assignments: dict[str, list[Unit]]
    loads: dict[str, float]
    lpt_makespan_s: float
    tail: list[str]
    tail_s: float
    overhead_s: float
    predicted_makespan_s: float  # lpt + tail + overhead
    oversized_groups: list[str] = field(default_factory=list)


def overhead(workers: int, worker_startup_s: float) -> float:
    return worker_startup_s * workers


def schedule(
    units: list[Unit],
    workers: int,
    tail: list[str],
    durations: Mapping[str, float],
    worker_startup_s: float = 0.0,
) -> Schedule:
    result = lpt(units, workers)
    tail_s = sum(durations.get(t, 0.0) for t in tail)
    over = overhead(workers, worker_startup_s)
    makespan = result.makespan + tail_s + over
    # A group longer than half the makespan dominates it; it stays whole
    # (correctness first) but is reported.
    oversized = sorted(
        u.id for u in units if u.id != SINGLE and u.duration > result.makespan / 2 and workers > 1
    )
    for gid in oversized:
        log.warning(
            "smartpar: group %s takes more than half of the predicted makespan with %d workers",
            gid,
            workers,
        )
    return Schedule(
        workers=workers,
        assignments={worker_name(i): a for i, a in enumerate(result.assignments)},
        loads={worker_name(i): load for i, load in enumerate(result.loads)},
        lpt_makespan_s=result.makespan,
        tail=list(tail),
        tail_s=tail_s,
        overhead_s=over,
        predicted_makespan_s=makespan,
        oversized_groups=oversized,
    )


def predicted_curve(
    units: list[Unit], tail_s: float, worker_startup_s: float, max_workers: int
) -> dict[int, float]:
    """T(m) for m = 1..max_workers."""
    return {
        m: lpt(units, m).makespan + tail_s + overhead(m, worker_startup_s)
        for m in range(1, max(1, max_workers) + 1)
    }


def recommend_workers(curve: Mapping[int, float]) -> int:
    """Smallest m whose T(m) is within 5% of the best."""
    best = min(curve.values())
    return min(m for m, t in curve.items() if t <= RECOMMEND_TOLERANCE * best)


# --- whole pipeline: estimates -> graph -> LPT -> recommended m ----------------------


@dataclass
class Analysis:
    estimates: Estimates
    graph: GraphResult
    units: list[Unit]
    curve: dict[int, float]
    recommended_workers: int
    schedule: Schedule


def analyze(
    tests: Iterable[str],
    history: History,
    deps: Dependencies | None,
    flaky: Flaky | None,
    max_workers: int,
    workers: int | None = None,
) -> Analysis:
    """Plan ``tests`` for ``workers`` (default: the recommended number).

    New tests (never seen in history) and NOD tests go to the isolated tail.
    """
    tests = sorted(set(tests))
    est = estimate_durations(tests, history)
    isolate = set(est.new_tests) | set(flaky.nod_tests() if flaky else ())
    graph = build_graph(
        tests,
        deps.file_access if deps else {},
        flaky.order_edges() if flaky else [],
        est.durations,
        isolate=isolate,
    )
    units = units_from_graph(graph, est.durations)
    tail_s = sum(est.durations[t] for t in graph.tail)
    startup = history.overhead.get("worker_startup_s", 0.0)
    curve = predicted_curve(units, tail_s, startup, max_workers)
    recommended = recommend_workers(curve)
    plan = schedule(units, workers or recommended, graph.tail, est.durations, startup)
    return Analysis(est, graph, units, curve, recommended, plan)
