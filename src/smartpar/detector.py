"""Method step 1 (profiling, continued): detection of order-dependent tests.

1. The suite runs ``runs`` times in random orders (pytest-randomly, seeds
   ``base_seed + i``), every run a fresh ``python -m pytest`` process.
   Runs in a given order keep pytest-randomly's re-seeding with their own
   seeds (``base_seed + runs + j``), so repeated isolated runs differ.
   Earlier recorded runs (runs/ and runs/merged/: profiling, smartpar runs)
   are added as observations, one per process with its tests in start order.
   This closes a gap of random orders: a test that fails only when it is the
   first of its module, or with a combination of tests that a random order
   rarely produces, is seen once a plan has produced it - the analysis below
   is the same for it.
2. Every test that failed at least once is run alone ``k`` times:

   * fails alone sometimes, but not always - NOD (not order-dependent);
   * fails alone always but passed in some suite order - brittle candidate;
   * passes alone always - victim candidate.

3. Victim: in an order where it failed, ``[t1..tk, V]``, the polluter is
   searched among ``t1..tk``: first pairs ``[P, V]`` for candidates with
   unrestored state changes (cache growth last), then delta debugging (ddmin)
   of the prefix. Brittle: the same in an order where it passed, looking for
   the state setter.
4. Every pair is confirmed by fresh processes - ``[P, V]`` fails and ``[V]``
   passes (brittle: ``[S, V]`` passes, ``[V]`` fails); only confirmed pairs
   are written to flaky.json.

Isolation: runs never share the project directory. Each runs in its own fresh
copy (without .git, .smartpar, virtualenvs and caches), so file conflicts of
concurrent runs cannot create false dependencies. If the project cannot be
copied, runs are sequential in the project directory.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from smartpar.models import (
    ACCESS_KINDS,
    FAILED_OUTCOMES,
    SOURCE_DETECT,
    FileAccess,
    Flaky,
    RunRecord,
)

log = logging.getLogger("smartpar.detector")

ISOLATION_RUNS = 3  # k: runs of a failing test alone
EXTRA_ISOLATION_RUNS = 5  # before declaring a victim without polluter unresolved
CONFIRM_RUNS = 2  # each half of a confirmation ([P, V] and [V])
# Defaults of the ini options smartpar_max_pair_checks / smartpar_max_orders_per_victim.
MAX_PAIR_CHECKS = 10  # direct [P, V] checks before delta debugging
MAX_ORDERS_PER_TEST = 3  # failing (passing) orders examined per victim (brittle)
CACHE_GROWTH = "cache_growth:"

# Never copied into a sandbox.
SANDBOX_IGNORE = (
    ".git",
    ".hg",
    ".svn",
    ".smartpar",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".nox",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".eggs",
    "node_modules",
    "*.egg-info",
)


class BudgetExceeded(Exception):
    """The --smartpar-detect-budget is used up."""


@dataclass
class RunResult:
    outcomes: dict[str, str]
    order: list[str]  # executed order (by start time)
    returncode: int

    def failed(self, nodeid: str) -> bool:
        return self.outcomes.get(nodeid) in FAILED_OUTCOMES

    def passed(self, nodeid: str) -> bool:
        return self.outcomes.get(nodeid) == "passed"


def observed_runs(records: Iterable[RunRecord], nodeids: Iterable[str]) -> list[RunResult]:
    """Recorded runs as observations: one result per process (worker), its tests
    in start order, restricted to ``nodeids`` (the suite being analysed)."""
    wanted = set(nodeids)
    results = []
    for record in records:
        by_worker: dict[str, list[Any]] = {}
        for test in record.tests:
            if test.nodeid in wanted:
                by_worker.setdefault(test.worker, []).append(test)
        for _worker, tests in sorted(by_worker.items()):
            tests.sort(key=lambda t: (t.start, t.nodeid))
            outcomes = {t.nodeid: t.outcome for t in tests}
            failed = any(o in FAILED_OUTCOMES for o in outcomes.values())
            results.append(RunResult(outcomes, [t.nodeid for t in tests], int(failed)))
    return results


class Runner(Protocol):
    def random_runs(self, seeds: Sequence[int]) -> list[RunResult]: ...

    def ordered_runs(self, orders: Sequence[Sequence[str]]) -> list[RunResult]: ...

    def confirm_runs(self, orders: Sequence[Sequence[str]]) -> list[RunResult]:
        """Like ordered_runs, but with less parallelism (timing-sensitive tests)."""
        ...


# --- delta debugging ------------------------------------------------------------------


def ddmin(items: list[str], test: Callable[[list[str]], bool]) -> list[str]:
    """Zeller's ddmin: a 1-minimal sub-sequence (order kept) for which ``test`` holds.

    Precondition: ``test(items)`` is true.
    """
    n = 2
    while len(items) >= 2:
        size = len(items)
        chunks = [items[i * size // n : (i + 1) * size // n] for i in range(n)]
        chunks = [c for c in chunks if c]
        for chunk in chunks:
            if test(chunk):
                items, n = chunk, 2
                break
        else:
            for i in range(len(chunks)):
                complement = [x for j, c in enumerate(chunks) if j != i for x in c]
                if test(complement):
                    items, n = complement, max(n - 1, 2)
                    break
            else:
                if n >= len(items):
                    break
                n = min(len(items), 2 * n)
    return items


def pair_candidates(
    prefix: Sequence[str], state_changes: dict[str, list[str]], limit: int = MAX_PAIR_CHECKS
) -> list[str]:
    """Tests worth a direct [P, V] check: unrestored state changes first, cache growth
    last; within a tier the closest to V first."""
    strong, cache_only = [], []
    for nodeid in reversed(prefix):
        keys = state_changes.get(nodeid, [])
        if any(not k.startswith(CACHE_GROWTH) for k in keys):
            strong.append(nodeid)
        elif keys:
            cache_only.append(nodeid)
    return (strong + cache_only)[:limit]


# --- the algorithm --------------------------------------------------------------------


def _rate(count: int, total: int) -> str:
    return f"{count}/{total}"


@dataclass
class DetectionResult:
    flaky: Flaky
    child_runs: int


class Detector:
    def __init__(
        self,
        runner: Runner,
        state_changes: dict[str, list[str]],
        isolation_runs: int = ISOLATION_RUNS,
        max_pair_checks: int = MAX_PAIR_CHECKS,
        max_orders: int = MAX_ORDERS_PER_TEST,
    ) -> None:
        self.runner = runner
        self.state_changes = state_changes
        self.k = isolation_runs
        self.max_pair_checks = max_pair_checks
        self.max_orders = max_orders
        # Tests whose search hit a limit: nodeid -> names of the limits.
        self.truncated: dict[str, set[str]] = {}
        self._cache: dict[tuple[str, ...], RunResult] = {}

    # Ordered runs are deterministic for order-dependent tests, so results
    # are cached by order (confirmations always use fresh runs).
    def _ordered(self, order: Sequence[str]) -> RunResult:
        key = tuple(order)
        if key not in self._cache:
            self._cache[key] = self.runner.ordered_runs([list(order)])[0]
        return self._cache[key]

    def detect(self, seeds: Sequence[int], observed: Sequence[RunResult] = ()) -> Flaky:
        """``observed``: earlier recorded runs (``observed_runs``), analysed like
        the random runs; fail rates count both."""
        runs = [*self.runner.random_runs(seeds), *observed]
        total = len(runs)
        failing = sorted({t for r in runs for t in r.outcomes if r.failed(t)})
        fails = {t: sum(r.failed(t) for r in runs) for t in failing}
        flaky = Flaky(runs=len(seeds), seeds=list(seeds))

        try:
            alone = self.runner.ordered_runs([[t] for t in failing for _ in range(self.k)])
        except BudgetExceeded:
            flaky.unresolved = [{"test": t, "reason": "budget"} for t in failing]
            return flaky
        alone_fails = {
            t: sum(r.failed(t) for r in alone[i * self.k : (i + 1) * self.k])
            for i, t in enumerate(failing)
        }

        for t in failing:
            rate = _rate(fails[t], total)
            a = alone_fails[t]
            try:
                if 0 < a < self.k:
                    flaky.nod.append(self._nod(t, rate, a, self.k))
                elif a == self.k:
                    if fails[t] == total:
                        flaky.unresolved.append(
                            {"test": t, "reason": "fails in every run and alone"}
                        )
                        continue
                    self._brittle(t, rate, runs, flaky)
                else:
                    self._victim(t, rate, runs, flaky)
            except BudgetExceeded:
                flaky.unresolved.append({"test": t, "reason": "budget"})
        return flaky

    def _nod(self, t: str, rate: str, alone_failed: int, alone_runs: int) -> dict[str, Any]:
        return {
            "test": t,
            "fail_rate": rate,
            "isolation_fail_rate": _rate(alone_failed, alone_runs),
            "note": "fails in isolation too - not order-dependent",
        }

    def _victim(self, v: str, rate: str, runs: list[RunResult], flaky: Flaky) -> None:
        polluters: list[str] = []
        examples: list[list[str]] = []
        orders = [r.order for r in runs if r.failed(v) and v in r.order]
        for order in self._limited_orders(v, orders, lambda prefix: polluters):
            prefix = order[: order.index(v)]
            culprits = self._search(prefix, v, lambda r: r.failed(v))
            if culprits and self._confirm(culprits, v, victim=True):
                polluters.extend(c for c in culprits if c not in polluters)
                examples.append([*culprits, v])
        if polluters:
            flaky.od_victims.append(
                {
                    "victim": v,
                    "polluters": polluters,
                    "failing_order_example": examples[0],
                    "fail_rate": rate,
                    "state_hint": self._hint(polluters),
                }
            )
            return
        # No confirmed polluter: maybe NOD after all (k isolated passes by chance).
        extra = self.runner.ordered_runs([[v]] * EXTRA_ISOLATION_RUNS)
        extra_fails = sum(r.failed(v) for r in extra)
        if extra_fails:
            flaky.nod.append(self._nod(v, rate, extra_fails, self.k + EXTRA_ISOLATION_RUNS))
        else:
            flaky.unresolved.append({"test": v, "reason": "no confirmed polluter found"})

    def _brittle(self, b: str, rate: str, runs: list[RunResult], flaky: Flaky) -> None:
        setters: list[str] = []
        examples: list[list[str]] = []
        orders = [r.order for r in runs if r.passed(b) and b in r.order]
        for order in self._limited_orders(b, orders, lambda prefix: setters):
            prefix = order[: order.index(b)]
            culprits = self._search(prefix, b, lambda r: r.passed(b))
            if culprits and self._confirm(culprits, b, victim=False):
                setters.extend(c for c in culprits if c not in setters)
                examples.append([*culprits, b])
        if setters:
            flaky.od_brittle.append(
                {
                    "test": b,
                    "state_setters": setters,
                    "passing_order_example": examples[0],
                    "fail_rate": rate,
                    "state_hint": self._hint(setters),
                }
            )
        else:
            flaky.unresolved.append({"test": b, "reason": "no confirmed state setter found"})

    def _limited_orders(
        self, target: str, orders: list[list[str]], found: Callable[[list[str]], list[str]]
    ) -> Iterable[list[str]]:
        """Orders not explained by culprits found so far, at most ``max_orders``."""
        examined = 0
        for order in orders:
            prefix = order[: order.index(target)]
            if any(c in prefix for c in found(prefix)):
                continue
            if examined == self.max_orders:
                self.truncated.setdefault(target, set()).add("max_orders_per_victim")
                return
            examined += 1
            yield order

    def _search(
        self, prefix: list[str], target: str, holds: Callable[[RunResult], bool]
    ) -> list[str] | None:
        """Minimal tests from ``prefix`` that, run before ``target``, make ``holds`` true."""
        if not prefix or not holds(self._ordered([*prefix, target])):
            return None  # the suite order does not reproduce it
        candidates = pair_candidates(prefix, self.state_changes, limit=len(prefix))
        if len(candidates) > self.max_pair_checks:
            self.truncated.setdefault(target, set()).add("max_pair_checks")
        for candidate in candidates[: self.max_pair_checks]:
            if holds(self._ordered([candidate, target])):
                return [candidate]
        return ddmin(prefix, lambda subset: holds(self._ordered([*subset, target])))

    def _confirm(self, culprits: list[str], target: str, victim: bool) -> bool:
        """Fresh processes: with the culprits the outcome flips, alone it does not."""
        orders = [[*culprits, target]] * CONFIRM_RUNS + [[target]] * CONFIRM_RUNS
        results = self.runner.confirm_runs(orders)
        with_c, alone = results[:CONFIRM_RUNS], results[CONFIRM_RUNS:]
        if victim:
            return all(r.failed(target) for r in with_c) and all(r.passed(target) for r in alone)
        return all(r.passed(target) for r in with_c) and all(r.failed(target) for r in alone)

    def truncated_report(self) -> list[dict[str, Any]]:
        return [{"test": t, "limits": sorted(v)} for t, v in sorted(self.truncated.items())]

    def _hint(self, tests: Iterable[str]) -> list[str]:
        return sorted({k for t in tests for k in self.state_changes.get(t, [])})


# --- running pytest in sandboxes --------------------------------------------------------


class Sandbox:
    """Fresh copies of the project for child runs (or the project itself)."""

    def __init__(self, root: Path, extra_ignore: Iterable[Path] = ()) -> None:
        self.root = root
        self.extra_ignore = {Path(p).resolve() for p in extra_ignore}
        self.base = Path(tempfile.mkdtemp(prefix="smartpar-detect-"))
        self.in_place = False
        self._counter = 0
        try:
            probe = self.create()
            shutil.rmtree(probe, ignore_errors=True)
        except OSError as exc:
            log.warning("smartpar: cannot copy the project (%s); runs will be sequential", exc)
            self.in_place = True

    def _ignore(self, directory: str, names: list[str]) -> set[str]:
        ignored = {n for n in names if any(fnmatch.fnmatch(n, p) for p in SANDBOX_IGNORE)}
        for name in names:
            if (Path(directory) / name).resolve() in self.extra_ignore:
                ignored.add(name)
        return ignored

    def create(self) -> Path:
        self._counter += 1
        dest = self.base / f"copy{self._counter}"
        shutil.copytree(self.root, dest, ignore=self._ignore, symlinks=True)
        return dest

    def cleanup(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)


@dataclass
class ProcessRunner:
    """Every run: a fresh ``python -m pytest`` process in a fresh project copy."""

    root: Path
    nodeids: list[str]  # the whole suite, for random runs
    sandbox: Sandbox
    inipath: Path | None
    workers: int
    deadline: float | None = None
    # Ordered runs keep pytest-randomly active (it re-seeds random & co. before
    # every test) with --randomly-dont-reorganize, each run with its own seed
    # ordered_seed_start, +1, ... in call order: isolated runs of a NOD test
    # differ, yet every run is reproducible.
    ordered_seed_start: int = 0
    ordered_seeds_used: int = 0
    child_runs: int = 0
    # File accesses seen in any child run (audit hook only, no snapshots):
    # a random order can trigger accesses the profiling order did not.
    file_access: dict[str, FileAccess] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sandbox.in_place:
            self.workers = 1  # concurrent runs would share files

    def random_runs(self, seeds: Sequence[int]) -> list[RunResult]:
        return self._map([["-p", "randomly", f"--randomly-seed={s}"] for s in seeds], None)

    def ordered_runs(self, orders: Sequence[Sequence[str]]) -> list[RunResult]:
        return self._map(self._ordered_opts(len(orders)), [list(o) for o in orders])

    def confirm_runs(self, orders: Sequence[Sequence[str]]) -> list[RunResult]:
        """At most half the workers: CPU load must not fail timing-sensitive tests."""
        return self._map(
            self._ordered_opts(len(orders)),
            [list(o) for o in orders],
            workers=max(1, self.workers // 2),
        )

    def _ordered_opts(self, n: int) -> list[list[str]]:
        with self._lock:
            first = self.ordered_seed_start + self.ordered_seeds_used
            self.ordered_seeds_used += n
        return [
            ["-p", "randomly", f"--randomly-seed={first + i}", "--randomly-dont-reorganize"]
            for i in range(n)
        ]

    def _map(
        self,
        opts: list[list[str]],
        orders: list[list[str]] | None,
        workers: int | None = None,
    ) -> list[RunResult]:
        jobs = list(zip(opts, orders or [self.nodeids] * len(opts), strict=True))
        workers = min(self.workers, workers or self.workers)
        if not jobs:
            return []
        if workers <= 1 or len(jobs) == 1:
            return [self._run(o, ids) for o, ids in jobs]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda job: self._run(*job), jobs))

    def _run(self, opts: list[str], nodeids: list[str]) -> RunResult:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise BudgetExceeded
        project = self.root if self.sandbox.in_place else self.sandbox.create()
        workdir = Path(tempfile.mkdtemp(prefix="run-", dir=self.sandbox.base))
        try:
            report = workdir / "report.json"
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:xdist",
                "-p",
                "no:cacheprovider",
                "-q",
                "--rootdir",
                str(project),
                f"--smartpar-child-report={report}",
                "--smartpar-child-audit",
                f"--basetemp={workdir / 'basetemp'}",
                *opts,
            ]
            ini = self._ini_in(project)
            if ini is not None:
                cmd += ["-c", str(ini)]
            argfile = workdir / "args.txt"
            argfile.write_text("\n".join(nodeids) + "\n", encoding="utf-8")
            cmd.append(f"@{argfile}")
            timeout = None
            if self.deadline is not None:
                timeout = max(1.0, self.deadline - time.monotonic())
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=project,
                    env=self._env(project),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise BudgetExceeded from exc
            with self._lock:
                self.child_runs += 1
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                record = RunRecord.from_dict(data)
            except (OSError, ValueError):
                return RunResult({}, [], proc.returncode)  # crashed: nothing ran
            self._collect_access(data.get("file_access", {}))
            ordered = sorted(record.tests, key=lambda t: t.start)
            return RunResult(
                {t.nodeid: t.outcome for t in record.tests},
                [t.nodeid for t in ordered],
                proc.returncode,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            if not self.sandbox.in_place:
                shutil.rmtree(project, ignore_errors=True)

    def _collect_access(self, raw: dict[str, dict[str, list[str]]]) -> None:
        with self._lock:
            for nodeid, kinds in raw.items():
                access = self.file_access.setdefault(nodeid, FileAccess())
                for kind in ACCESS_KINDS:
                    for resource in kinds.get(kind, []):
                        access.add(kind, resource, SOURCE_DETECT)

    def _ini_in(self, project: Path) -> Path | None:
        if self.inipath is None:
            return None
        try:
            return project / self.inipath.relative_to(self.root)
        except ValueError:
            return self.inipath  # ini outside the project

    def _env(self, project: Path) -> dict[str, str]:
        """Imports must come from the copy, not from the original tree (e.g. an
        editable install of a src/ layout), so map project paths on sys.path."""
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("PYTEST_XDIST_") and k != "PYTEST_CURRENT_TEST"
        }
        mapped: list[str] = []
        for entry in [str(self.root / "src"), str(self.root), *sys.path]:
            path = Path(entry or ".").resolve()
            try:
                rel = path.relative_to(self.root.resolve())
            except ValueError:
                continue
            target = str(project / rel)
            if Path(target).is_dir() and target not in mapped:
                mapped.append(target)
        if env.get("PYTHONPATH"):
            mapped.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(mapped)
        return env


def choose_seeds(base: int | None, runs: int) -> tuple[int, list[int]]:
    """All seeds of a detection run; the base is random only if not given, and recorded."""
    if base is None:
        base = random.SystemRandom().randrange(1, 2**31 - runs)
    return base, [base + i for i in range(runs)]
