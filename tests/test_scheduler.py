"""Phase 4: duration estimates, deterministic LPT, T(m) and the recommended m."""

from __future__ import annotations

import logging
import random

import pytest

from smartpar import cli, history, scheduler
from smartpar.models import (
    Dependencies,
    FileAccess,
    Flaky,
    History,
    Overhead,
    TestHistory,
)
from smartpar.scheduler import SINGLE, Unit


def th(duration: float, timed: int = 1) -> TestHistory:
    return TestHistory(
        duration, runs=max(timed, 1), last_outcome="passed", fail_count=0, timed_runs=timed
    )


def singles(durations: dict[str, float]) -> list[Unit]:
    return [Unit(SINGLE, (t,), d) for t, d in durations.items()]


# --- estimates ---------------------------------------------------------------------


def test_estimate_durations() -> None:
    hist = History(tests={"a": th(1.0), "b": th(3.0), "c": th(2.0), "p": th(0.0, timed=0)})
    est = scheduler.estimate_durations(["a", "b", "c", "p", "new"], hist)
    assert est.median_s == 2.0
    assert est.durations == {"a": 1.0, "b": 3.0, "c": 2.0, "p": 2.0, "new": 2.0}
    assert est.estimated == ["p"]  # profiled only: median
    assert est.new_tests == ["new"]  # never seen
    assert est.has_timings


def test_estimate_without_any_timing() -> None:
    est = scheduler.estimate_durations(["a", "b"], History(tests={"a": th(0.0, timed=0)}))
    assert est.durations == {"a": 1.0, "b": 1.0}
    assert not est.has_timings


# --- LPT ---------------------------------------------------------------------------


def test_lpt_graham_example() -> None:
    # Classic worst case for m=4: LPT gives 15, the optimum is 12.
    durations = [7, 7, 6, 6, 5, 5, 4, 4, 4]
    units = singles({f"t{i}": float(d) for i, d in enumerate(durations)})
    result = scheduler.lpt(units, 4)
    assert result.makespan == 15
    assert [[u.tests[0] for u in a] for a in result.assignments] == [
        ["t0", "t6", "t8"],
        ["t1", "t7"],
        ["t2", "t4"],
        ["t3", "t5"],
    ]


def test_lpt_ties_by_nodeid_and_worker_number() -> None:
    units = singles({"d": 1.0, "b": 1.0, "c": 1.0, "a": 1.0})
    result = scheduler.lpt(units, 2)
    assert [[u.tests[0] for u in a] for a in result.assignments] == [["a", "c"], ["b", "d"]]


def test_lpt_is_deterministic_under_permutation() -> None:
    rng = random.Random(3)
    units = singles({f"t{i:03d}": float(rng.choice([0.1, 0.5, 1.0, 2.0])) for i in range(200)})
    units.append(Unit("g0", ("x2", "x1"), 3.0))
    expected = scheduler.lpt(units, 4)
    for _ in range(5):
        assert scheduler.lpt(rng.sample(units, len(units)), 4) == expected


def test_group_is_placed_whole_in_order() -> None:
    units = [Unit("g0", ("b", "a", "c"), 3.0), *singles({"x": 1.0, "y": 1.0})]
    result = scheduler.lpt(units, 2)
    assert result.assignments[0] == [units[0]]
    assert result.makespan == 3.0


def test_lpt_rejects_zero_workers() -> None:
    with pytest.raises(ValueError):
        scheduler.lpt([], 0)


# --- T(m) and recommended m -------------------------------------------------------


def test_schedule_adds_tail_and_overhead(caplog: pytest.LogCaptureFixture) -> None:
    units = [Unit("g0", ("a", "b"), 6.0), *singles({"c": 1.0, "d": 1.0})]
    with caplog.at_level(logging.WARNING, logger="smartpar.scheduler"):
        plan = scheduler.schedule(
            units, 2, tail=["t"], durations={"t": 2.0}, overhead=Overhead(a=0.2, b=0.4)
        )
    assert plan.lpt_makespan_s == 6.0
    assert plan.tail_s == 2.0
    assert plan.overhead_s == 1.0
    assert plan.predicted_makespan_s == 9.0
    assert plan.oversized_groups == ["g0"]  # 6 > 6/2: stays whole, reported
    assert "g0" in caplog.text
    assert list(plan.assignments) == ["gw0", "gw1"]


def test_workers_run_their_units_in_collection_order() -> None:
    # LPT places by duration; each worker then runs its units in the project's
    # collection order, which does not change its load.
    durations = {"m1::a": 1.0, "m2::b": 5.0, "m1::c": 2.0, "m2::d": 3.0, "new": 1.0}
    index = {"m1::a": 0, "m1::c": 1, "m2::b": 2, "m2::d": 3}
    units = [*singles(durations), Unit("g0", ("m2::x", "m1::y"), 4.0)]
    index.update({"m1::y": 4, "m2::x": 5})
    plain = scheduler.schedule(units, 2, tail=[], durations=durations)
    ordered = scheduler.schedule(units, 2, tail=[], durations=durations, collection_index=index)
    assert ordered.loads == plain.loads
    for worker, assigned in ordered.assignments.items():
        assert sorted(assigned, key=lambda u: u.tests) == sorted(
            plain.assignments[worker], key=lambda u: u.tests
        )
        firsts = [min(index.get(t, len(index)) for t in u.tests) for u in assigned]
        assert firsts == sorted(firsts)
    # A test without a collection position ("new") runs after the others.
    (worker,) = [w for w, a in ordered.assignments.items() if any("new" in u.tests for u in a)]
    assert ordered.assignments[worker][-1].tests == ("new",)


def test_curve_and_recommendation_with_startup_cost() -> None:
    units = singles({f"t{i}": 1.0 for i in range(8)})
    curve = scheduler.predicted_curve(units, 0.0, Overhead(a=0.0, b=0.5), max_workers=8)
    assert curve[1] == 8.5 and curve[2] == 5.0 and curve[4] == 4.0 and curve[8] == 5.0
    assert scheduler.recommend_workers(curve) == 4


def test_recommendation_prefers_fewer_workers_within_5_percent() -> None:
    assert scheduler.recommend_workers({1: 10.0, 2: 5.2, 3: 5.0, 4: 5.0}) == 2
    assert scheduler.recommend_workers({1: 10.0, 2: 5.3, 3: 5.0}) == 3


def test_available_cpus_positive() -> None:
    assert scheduler.available_cpus() >= 1


def test_analyze_isolates_nod_tests_only() -> None:
    hist = History(
        tests={t: th(d) for t, d in {"a": 2.0, "b": 1.0, "c": 1.0, "nod": 0.5}.items()},
        overhead=Overhead(a=0.1, b=0.05),
    )
    deps = Dependencies(file_access={"a": FileAccess(W=["f"]), "b": FileAccess(R=["f"])})
    flaky = Flaky(nod=[{"test": "nod", "fail_rate": "2/20"}])
    a = scheduler.analyze(["a", "b", "c", "nod", "new"], hist, deps, flaky, max_workers=3)
    assert a.graph.tail == ["nod"]  # NOD only; the new test is planned normally
    assert a.estimates.new_tests == ["new"]
    assert a.estimates.durations["new"] == a.estimates.median_s
    assert [g.tests for g in a.graph.groups] == [["a", "b"]]
    assert a.recommended_workers in a.curve
    assert set(a.curve) == {1, 2, 3}
    planned = [t for units in a.schedule.assignments.values() for u in units for t in u.tests]
    assert sorted(planned + a.schedule.tail) == ["a", "b", "c", "new", "nod"]
    fixed = scheduler.analyze(["a", "b", "c"], hist, deps, None, max_workers=3, workers=2)
    assert fixed.schedule.workers == 2


# --- smartpar show -----------------------------------------------------------------


def test_cli_show(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--dir", str(tmp_path), "show"]) == 1
    hist = History(tests={"t::a": th(2.0), "t::b": th(1.0), "t::c": th(1.0)})
    history.save_history(tmp_path, hist)
    deps = Dependencies(
        file_access={"t::a": FileAccess(W=["<project>/f"]), "t::b": FileAccess(R=["<project>/f"])},
        state_changes={"t::c": ["mod.X"], "t::b": ["cache_growth:mod.f"]},
    )
    history.atomic_write_json(tmp_path / "dependencies.json", deps.to_dict())
    capsys.readouterr()
    assert cli.main(["--dir", str(tmp_path), "show", "--max-workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "g0: 2 tests, 3.000s, file" in out
    assert "candidates (unrestored state changes): 1, only cache growth: 1" in out
    assert "t::c: mod.X" in out
    assert "m=2: 3.000  <- recommended" in out
