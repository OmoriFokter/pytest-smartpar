"""Phase 6: detector algorithm against a simulated suite, ddmin, sandboxes."""

from __future__ import annotations

import random
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from smartpar import detector
from smartpar.detector import (
    BudgetExceeded,
    Detector,
    ProcessRunner,
    RunResult,
    Sandbox,
    choose_seeds,
    ddmin,
    pair_candidates,
)


class FakeSuite:
    """Order-dependent semantics of a small suite, without processes.

    * victims: fail if any of their polluters ran earlier in the same process
      (``combo`` victims need all polluters);
    * brittle: pass only if their setter ran earlier;
    * nod: fails on every other invocation (deterministic "randomness");
    * broken: always fails.
    """

    def __init__(
        self,
        tests: list[str],
        victims: dict[str, list[str]] | None = None,
        combo: dict[str, list[str]] | None = None,
        brittle: dict[str, str] | None = None,
        nod: Sequence[str] = (),
        broken: Sequence[str] = (),
        budget_after: int | None = None,
    ) -> None:
        self.tests = tests
        self.victims = victims or {}
        self.combo = combo or {}
        self.brittle = brittle or {}
        self.nod = set(nod)
        self.broken = set(broken)
        self.nod_calls = 0
        self.ordered_log: list[list[str]] = []
        self.budget_after = budget_after
        self.child_runs = 0

    def _run(self, order: list[str]) -> RunResult:
        self.child_runs += 1
        if self.budget_after is not None and self.child_runs > self.budget_after:
            raise BudgetExceeded
        outcomes: dict[str, str] = {}
        seen: set[str] = set()
        for t in order:
            fail = False
            if t in self.victims:
                fail = any(p in seen for p in self.victims[t])
            elif t in self.combo:
                fail = all(p in seen for p in self.combo[t])
            elif t in self.brittle:
                fail = self.brittle[t] not in seen
            elif t in self.nod:
                self.nod_calls += 1
                fail = self.nod_calls % 2 == 0
            elif t in self.broken:
                fail = True
            outcomes[t] = "failed" if fail else "passed"
            seen.add(t)
        return RunResult(outcomes, list(order), int(any(o == "failed" for o in outcomes.values())))

    def random_runs(self, seeds: Sequence[int]) -> list[RunResult]:
        results = []
        for seed in seeds:
            order = list(self.tests)
            random.Random(seed).shuffle(order)
            results.append(self._run(order))
        return results

    def ordered_runs(self, orders: Sequence[Sequence[str]]) -> list[RunResult]:
        results = []
        for order in orders:
            self.ordered_log.append(list(order))
            results.append(self._run(list(order)))
        return results

    def confirm_runs(self, orders: Sequence[Sequence[str]]) -> list[RunResult]:
        return self.ordered_runs(orders)


TESTS = [f"t{i:02d}" for i in range(12)]


def detect(suite: FakeSuite, state_changes: dict | None = None, runs: int = 20):
    return Detector(suite, state_changes or {}).detect(list(range(100, 100 + runs)))


def test_classifies_victim_brittle_nod_broken() -> None:
    suite = FakeSuite(
        [*TESTS, "polluter", "victim", "setter", "brittle", "nod", "broken"],
        victims={"victim": ["polluter"]},
        brittle={"brittle": "setter"},
        nod=["nod"],
        broken=["broken"],
    )
    flaky = detect(suite, {"polluter": ["m.X"]})
    assert [(v["victim"], v["polluters"]) for v in flaky.od_victims] == [("victim", ["polluter"])]
    assert flaky.od_victims[0]["state_hint"] == ["m.X"]
    assert flaky.od_victims[0]["failing_order_example"] == ["polluter", "victim"]
    assert [(b["test"], b["state_setters"]) for b in flaky.od_brittle] == [("brittle", ["setter"])]
    assert [n["test"] for n in flaky.nod] == ["nod"]
    # Classified by the first k isolated runs, not by the later fallback check.
    assert flaky.nod[0]["isolation_fail_rate"].endswith(f"/{detector.ISOLATION_RUNS}")
    assert flaky.unresolved == [{"test": "broken", "reason": "fails in every run and alone"}]
    assert flaky.seeds == list(range(100, 120)) and flaky.runs == 20
    assert flaky.od_victims[0]["fail_rate"].endswith("/20")


def test_nod_missed_by_isolation_is_caught_by_extra_runs() -> None:
    class LuckyIsolation(FakeSuite):
        """The first k isolated runs of the NOD test happen to pass."""

        def _run(self, order: list[str]) -> RunResult:
            if order == ["nod"] and self.lucky > 0:
                self.lucky -= 1
                return RunResult({"nod": "passed"}, order, 0)
            return super()._run(order)

    suite = LuckyIsolation([*TESTS, "nod"], nod=["nod"])
    suite.lucky = detector.ISOLATION_RUNS
    flaky = detect(suite)
    assert [n["test"] for n in flaky.nod] == ["nod"]
    total = detector.ISOLATION_RUNS + detector.EXTRA_ISOLATION_RUNS
    assert flaky.nod[0]["isolation_fail_rate"].endswith(f"/{total}")


def test_state_change_candidates_are_checked_first_cache_growth_last() -> None:
    suite = FakeSuite([*TESTS, "victim"], victims={"victim": ["t03"]})
    state = {"t07": ["cache_growth:m.f"], "t05": ["m.A"], "t03": ["m.B"]}
    flaky = detect(suite, state)
    assert flaky.od_victims[0]["polluters"] == ["t03"]
    pair_checks = [o for o in suite.ordered_log if len(o) == 2 and o[1] == "victim"]
    assert pair_checks[0][0] in {"t05", "t03"}  # unrestored state changes first
    assert all(o[0] != "t07" for o in pair_checks[: pair_checks.index(["t03", "victim"])])


def test_delta_debugging_finds_polluter_without_state_hints() -> None:
    suite = FakeSuite([*TESTS, "victim"], victims={"victim": ["t09"]})
    flaky = detect(suite, {})
    assert flaky.od_victims[0]["polluters"] == ["t09"]
    assert flaky.od_victims[0]["state_hint"] == []


def test_combination_of_polluters_is_found() -> None:
    suite = FakeSuite([*TESTS, "victim"], combo={"victim": ["t02", "t08"]})
    flaky = detect(suite, {}, runs=40)
    (entry,) = flaky.od_victims
    assert sorted(entry["polluters"]) == ["t02", "t08"]


def test_every_pair_is_confirmed_in_fresh_runs() -> None:
    suite = FakeSuite([*TESTS, "polluter", "victim"], victims={"victim": ["polluter"]})
    detect(suite, {"polluter": ["m.X"]})
    log = suite.ordered_log
    # Cached search runs are not reused for confirmation: the pair and the
    # victim alone are each run CONFIRM_RUNS more times.
    assert log.count(["polluter", "victim"]) >= 1 + detector.CONFIRM_RUNS
    assert log.count(["victim"]) >= detector.ISOLATION_RUNS + detector.CONFIRM_RUNS


def test_unconfirmed_candidate_is_not_reported() -> None:
    class Flipping(FakeSuite):
        """The pair fails during the search but not on confirmation."""

        def ordered_runs(self, orders):  # type: ignore[override]
            results = super().ordered_runs(orders)
            return results

        def confirm_runs(self, orders):  # type: ignore[override]
            return [RunResult({o[-1]: "passed"}, list(o), 0) for o in orders]

    suite = Flipping([*TESTS, "polluter", "victim"], victims={"victim": ["polluter"]})
    flaky = detect(suite, {"polluter": ["m.X"]})
    assert flaky.od_victims == []
    assert flaky.unresolved == [{"test": "victim", "reason": "no confirmed polluter found"}]


def test_budget_exhaustion_is_reported_as_unresolved() -> None:
    suite = FakeSuite(
        [*TESTS, "polluter", "victim"], victims={"victim": ["polluter"]}, budget_after=25
    )
    flaky = detect(suite, {})
    assert flaky.od_victims == []
    assert {"test": "victim", "reason": "budget"} in flaky.unresolved


def test_nothing_flaky() -> None:
    flaky = detect(FakeSuite(TESTS))
    assert (flaky.od_victims, flaky.od_brittle, flaky.nod, flaky.unresolved) == ([], [], [], [])


# --- building blocks -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("culprits", "n"),
    [({"c"}, 10), ({"a", "j"}, 10), ({"x05"}, 50), ({"x01", "x30", "x49"}, 50)],
)
def test_ddmin_finds_minimal_subsequence(culprits: set[str], n: int) -> None:
    items = [chr(ord("a") + i) for i in range(n)] if n <= 10 else [f"x{i:02d}" for i in range(n)]
    calls = []

    def test(subset: list[str]) -> bool:
        calls.append(subset)
        return culprits <= set(subset)

    result = ddmin(items, test)
    assert set(result) == culprits
    assert result == [x for x in items if x in culprits]  # order kept
    unique = {tuple(c) for c in calls}  # the detector caches repeated orders
    if len(culprits) == 1:
        assert len(unique) <= 4 * n.bit_length()  # logarithmic for one culprit


def test_pair_candidates_priority() -> None:
    prefix = ["a", "b", "c", "d", "e"]
    state = {"a": ["m.X"], "b": ["cache_growth:m.f"], "d": ["env:Y", "cache_growth:m.g"]}
    assert pair_candidates(prefix, state) == ["d", "a", "b"]
    assert pair_candidates(prefix, state, limit=1) == ["d"]


def test_choose_seeds() -> None:
    assert choose_seeds(7, 3) == (7, [7, 8, 9])
    base, seeds = choose_seeds(None, 4)
    assert seeds == [base, base + 1, base + 2, base + 3]


def test_sandbox_copy_excludes_vcs_data_envs_and_caches(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    for d in [
        ".git/objects",
        ".smartpar/runs",
        ".venv/lib",
        "venv",
        "pkg/__pycache__",
        "pkg.egg-info",
        ".pytest_cache",
        "custom_env",
        "src/pkg",
        "tests",
    ]:
        (root / d).mkdir(parents=True)
    (root / "src/pkg/__init__.py").write_text("")
    (root / "tests/test_x.py").write_text("")
    (root / "custom_env/marker").write_text("")
    sandbox = Sandbox(root, extra_ignore=[root / "custom_env"])
    try:
        copy = sandbox.create()
        present = {p.relative_to(copy).as_posix() for p in copy.rglob("*")}
        assert {"src/pkg/__init__.py", "tests/test_x.py", "pkg"} <= present
        for excluded in [
            ".git",
            ".smartpar",
            ".venv",
            "venv",
            "pkg/__pycache__",
            "pkg.egg-info",
            ".pytest_cache",
            "custom_env",
        ]:
            assert excluded not in present
        assert sandbox.create() != copy  # every run gets its own copy
    finally:
        sandbox.cleanup()


def test_sandbox_falls_back_to_sequential_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_copytree(*args: object, **kwargs: object) -> None:
        raise OSError("no space left")

    monkeypatch.setattr(detector.shutil, "copytree", broken_copytree)
    sandbox = Sandbox(tmp_path)
    try:
        assert sandbox.in_place
        runner = ProcessRunner(tmp_path, [], sandbox, None, workers=8)
        assert runner.workers == 1  # concurrent runs would share the directory
    finally:
        monkeypatch.setattr(detector.shutil, "copytree", shutil.copytree)
        sandbox.cleanup()


def test_child_imports_come_from_the_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    copy = tmp_path / "copy"
    (copy / "src").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(root / "src"), "/usr/lib/python3"])
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    sandbox = Sandbox(root)
    try:
        runner = ProcessRunner(root, [], sandbox, None, workers=1)
        env = runner._env(copy)
    finally:
        sandbox.cleanup()
    parts = env["PYTHONPATH"].split(":")
    assert parts[0] == str(copy / "src")
    assert str(root / "src") not in parts
    assert parts[-1] == "/elsewhere"


def test_ordered_runs_get_distinct_reproducible_randomly_seeds(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path)
    try:
        runner = ProcessRunner(tmp_path, [], sandbox, None, workers=4, ordered_seed_start=120)
        first = runner._ordered_opts(3)
        second = runner._ordered_opts(2)
    finally:
        sandbox.cleanup()
    seeds = [o[2] for o in first + second]
    assert seeds == [f"--randomly-seed={s}" for s in range(120, 125)]  # all different
    assert all(
        o[:2] == ["-p", "randomly"] and o[3] == "--randomly-dont-reorganize" for o in first + second
    )
    assert runner.ordered_seeds_used == 5


@pytest.mark.parametrize(("workers", "expected"), [(8, 4), (3, 1), (1, 1)])
def test_confirmation_uses_at_most_half_the_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, expected: int
) -> None:
    sandbox = Sandbox(tmp_path)
    used: list[int] = []

    def fake_map(opts, orders, workers=None):  # type: ignore[no-untyped-def]
        used.append(workers)
        return []

    try:
        runner = ProcessRunner(tmp_path, [], sandbox, None, workers=workers)
        monkeypatch.setattr(runner, "_map", fake_map)
        runner.confirm_runs([["a", "b"], ["b"]])
    finally:
        sandbox.cleanup()
    assert used == [expected]


def test_limits_are_applied_and_truncation_reported() -> None:
    suite = FakeSuite([*TESTS, "victim"], victims={"victim": ["t09"]})
    state = {t: ["m.X"] for t in TESTS}  # every test is a pair candidate
    det = Detector(suite, state, max_pair_checks=2, max_orders=1)
    flaky = det.detect(list(range(100, 120)))
    assert flaky.od_victims[0]["polluters"] == ["t09"]  # ddmin still finds it
    pair_checks = [o for o in suite.ordered_log if len(o) == 2 and o[1] == "victim"]
    assert len({tuple(o) for o in pair_checks} - {("t09", "victim")}) <= 2
    assert det.truncated_report() == [{"test": "victim", "limits": ["max_pair_checks"]}]


def test_max_orders_truncation() -> None:
    # Two independent polluters: with max_orders=1 the second may stay unexplored.
    suite = FakeSuite([*TESTS, "victim"], victims={"victim": ["t02", "t08"]})
    det = Detector(suite, {}, max_orders=1)
    flaky = det.detect(list(range(100, 140)))
    assert len(flaky.od_victims[0]["polluters"]) == 1
    assert det.truncated_report() == [{"test": "victim", "limits": ["max_orders_per_victim"]}]
