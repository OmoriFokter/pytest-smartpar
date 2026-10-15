"""Phase 5: ``pytest -n m --smartpar`` end to end with real xdist workers.

Data files (history, dependencies, flaky) are written directly so that every
scenario is deterministic; each run is a subprocess with a timeout, so a hang
fails the test instead of blocking the suite.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from smartpar import history
from smartpar.models import Dependencies, FileAccess, Flaky, History, TestHistory
from smartpar.scheduler import available_cpus
from smartpar.xdist_scheduling import child_command

TIMEOUT = 120


def th(duration: float) -> TestHistory:
    return TestHistory(duration, runs=3, last_outcome="passed", fail_count=0, timed_runs=3)


def write_data(
    root: Path,
    durations: dict[str, float],
    file_access: dict[str, FileAccess] | None = None,
    collection: list[str] | None = None,
    flaky: Flaky | None = None,
) -> Path:
    smartpar_dir = root / ".smartpar"
    history.save_history(smartpar_dir, History(tests={k: th(v) for k, v in durations.items()}))
    deps = Dependencies(
        file_access=file_access or {},
        collection_index={n: i for i, n in enumerate(collection or [])},
    )
    history.atomic_write_json(smartpar_dir / "dependencies.json", deps.to_dict())
    if flaky is not None:
        history.atomic_write_json(smartpar_dir / "flaky.json", flaky.to_dict())
    return smartpar_dir


def last_run(smartpar_dir: Path) -> dict:
    runs = sorted((smartpar_dir / "runs").glob("run_*.json"))
    return json.loads(runs[-1].read_text())


def report(smartpar_dir: Path) -> list[dict[str, str]]:
    with open(smartpar_dir / "report.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def by_name(run: dict) -> dict[str, dict]:
    return {t["nodeid"].split("::")[-1]: t for t in run["tests"]}


def junit_cases(path: Path) -> dict[str, ET.Element]:
    root = ET.parse(path).getroot()
    return {c.get("name"): c for c in root.iter("testcase")}


def props(case: ET.Element) -> dict[str, str]:
    return {p.get("name"): p.get("value") for p in case.iter("property")}


def nod_flaky(nodeid: str) -> Flaky:
    return Flaky(runs=20, seeds=[1], nod=[{"test": nodeid, "fail_rate": "2/20", "note": "-"}])


SLEEPY = """
import time
def test_long():  time.sleep(0.6)
def test_mid1():  time.sleep(0.3)
def test_mid2():  time.sleep(0.3)
def test_s1():    time.sleep(0.1)
def test_s2():    time.sleep(0.1)
def test_s3():    time.sleep(0.1)
def test_nod():   time.sleep(0.05)
def test_cycle_a(): time.sleep(0.05)
def test_cycle_b(): time.sleep(0.05)
"""
SLEEPY_D = {
    "test_long": 0.6,
    "test_mid1": 0.3,
    "test_mid2": 0.3,
    "test_s1": 0.1,
    "test_s2": 0.1,
    "test_s3": 0.1,
    "test_nod": 0.05,
    "test_cycle_a": 0.05,
    "test_cycle_b": 0.05,
}


def sleepy_project(pytester: pytest.Pytester, flaky: Flaky | None = None) -> Path:
    pytester.makepyfile(test_sleepy=SLEEPY)
    ids = {n: f"test_sleepy.py::{n}" for n in SLEEPY_D}
    flaky = flaky or Flaky(
        runs=20,
        seeds=[1],
        nod=[{"test": ids["test_nod"], "fail_rate": "2/20", "note": "-"}],
        # An unresolvable order cycle: both go to the tail as well.
        od_victims=[
            {"victim": ids["test_cycle_a"], "polluters": [ids["test_cycle_b"]]},
            {"victim": ids["test_cycle_b"], "polluters": [ids["test_cycle_a"]]},
        ],
    )
    return write_data(
        pytester.path,
        {ids[n]: d for n, d in SLEEPY_D.items()},
        collection=list(ids.values()),
        flaky=flaky,
    )


def test_tail_starts_after_all_other_tests_and_does_not_hang(pytester: pytest.Pytester) -> None:
    smartpar_dir = sleepy_project(pytester)
    result = pytester.runpytest_subprocess("-n", "3", "--smartpar", timeout=TIMEOUT)
    result.assert_outcomes(passed=9)
    plan = json.loads((smartpar_dir / "plan.json").read_text())
    # Topological with collection order; the a<->b cycle is broken by nodeid.
    assert [t.split("::")[1] for t in plan["isolated_tail"]] == [
        "test_nod",
        "test_cycle_a",
        "test_cycle_b",
    ]
    tests = by_name(last_run(smartpar_dir))
    tail = {n: tests[n] for n in ("test_nod", "test_cycle_a", "test_cycle_b")}
    others = [t for n, t in tests.items() if n not in tail]
    last_other_end = max(t["end"] for t in others)
    for t in tail.values():
        assert t["start"] >= last_other_end
        assert t["worker"] == plan["tail_worker"]
    assert last_run(smartpar_dir)["tail_mode"] == "barrier"


def test_custom_scheduler_is_used_normally(pytester: pytest.Pytester) -> None:
    smartpar_dir = sleepy_project(pytester)
    result = pytester.runpytest_subprocess("-n", "2", "--smartpar", "-v", timeout=TIMEOUT)
    result.stdout.fnmatch_lines(["*scheduling tests via SmartparScheduling*"])
    rows = report(smartpar_dir)
    assert {r["mode"] for r in rows} == {"smartpar"}
    assert {r["tail_mode"] for r in rows} == {"barrier"}
    total = next(r for r in rows if r["worker_id"] == "ALL")
    assert float(total["predicted_wall_s"]) > 0
    assert total["prediction_error_pct"] != ""
    plan = json.loads((smartpar_dir / "plan.json").read_text())
    assert plan["enforced"] is True and plan["mode"] == "smartpar"
    run = last_run(smartpar_dir)
    assert run["overhead_s"] is not None and run["overhead_s"] >= 0
    assert run["randomly_reorder_disabled"] is True


def test_forced_fallback_runs_tail_as_separate_pass(pytester: pytest.Pytester) -> None:
    smartpar_dir = sleepy_project(pytester)
    result = pytester.runpytest_subprocess(
        "-n", "2", "--smartpar", "--smartpar-fallback", "-v", timeout=TIMEOUT
    )
    result.stdout.fnmatch_lines(["*scheduling tests via LoadGroupScheduling*"])
    assert result.ret == 0
    rows = report(smartpar_dir)
    assert {r["mode"] for r in rows} == {"smartpar-fallback"}
    assert {r["tail_mode"] for r in rows} == {"separate-pass"}
    assert "post:tail" in {r["worker_id"] for r in rows}
    tests = by_name(last_run(smartpar_dir))
    assert len(tests) == 9
    tail = [tests[n] for n in ("test_nod", "test_cycle_a", "test_cycle_b")]
    assert all(t["worker"] == "post:tail" for t in tail)
    others = [t for n, t in tests.items() if t["worker"] != "post:tail"]
    assert min(t["start"] for t in tail) >= max(t["end"] for t in others)
    cases = junit_cases(smartpar_dir / "results.xml")
    assert props(cases["test_nod"]) == {"smartpar_tail": "separate-pass"}
    assert "@" not in "".join(t["nodeid"] for t in last_run(smartpar_dir)["tests"])


GROUPED = """
import time
def test_a(): time.sleep(0.05)
def test_b(): time.sleep(0.05)
def test_c(): time.sleep(0.05)
def test_d(): time.sleep(0.05)
def test_x(): time.sleep(0.3)
def test_y(): time.sleep(0.3)
"""


def test_group_runs_on_one_worker_in_plan_order(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_grouped=GROUPED)
    ids = {n: f"test_grouped.py::test_{n}" for n in "abcdxy"}
    shared = FileAccess(W=["<project>/shared"])
    smartpar_dir = write_data(
        pytester.path,
        {i: (0.3 if n in "xy" else 0.05) for n, i in ids.items()},
        file_access={ids[n]: shared for n in "abcd"},
        collection=[ids[n] for n in "abcdxy"],
        # Confirmed pair: victim c must run before polluter a.
        flaky=Flaky(runs=20, seeds=[1], od_victims=[{"victim": ids["c"], "polluters": [ids["a"]]}]),
    )
    # randomly stays active: its reordering is switched off on the workers too.
    result = pytester.runpytest_subprocess("-n", "3", "--smartpar", timeout=TIMEOUT)
    result.assert_outcomes(passed=6)
    plan = json.loads((smartpar_dir / "plan.json").read_text())
    (worker, units), *_ = [
        (w, [u for u in us if u[0] != "single"])
        for w, us in plan["assignments"].items()
        if any(u[0] != "single" for u in us)
    ]
    (group,) = units
    planned = group[1]
    # Collection order a, b, c, d - but c must precede a: b, c, a, d.
    assert [t.split("::")[1] for t in planned] == ["test_b", "test_c", "test_a", "test_d"]
    tests = by_name(last_run(smartpar_dir))
    ran = sorted((tests[t.split("::")[1]] for t in planned), key=lambda t: t["start"])
    assert [t["nodeid"] for t in ran] == planned
    assert {t["worker"] for t in ran} == {worker}


NEW_TESTS = """
import os
STATE = {}
def test_polluter():
    STATE["x"] = 1
def test_new_victim():
    assert "x" not in STATE  # fails after test_polluter in the same process
def test_new_broken():
    assert False
"""


def test_failed_new_tests_are_rerun_alone(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_new=NEW_TESTS)
    polluter, victim = "test_new.py::test_polluter", "test_new.py::test_new_victim"
    broken = "test_new.py::test_new_broken"
    shared = FileAccess(W=["<project>/f"])
    smartpar_dir = write_data(
        pytester.path,
        {polluter: 0.01},  # the other two are new: not in history
        file_access={polluter: shared, victim: shared},  # same group: polluter first
        collection=[polluter, victim, broken],
    )
    result = pytester.runpytest_subprocess("-n", "2", "--smartpar", timeout=TIMEOUT)
    result.stdout.fnmatch_lines(
        [
            "*re-run of new test test_new.py::test_new_broken: failed again*",
            "*re-run of new test test_new.py::test_new_victim: passed alone: suspected dependency*",
        ]
    )
    assert result.ret == 1  # test_new_broken still fails
    run = last_run(smartpar_dir)
    reruns = {r["nodeid"]: r for r in run["reruns"]}
    assert reruns[victim]["first_outcome"] == "failed"
    assert reruns[victim]["rerun_outcome"] == "passed"
    assert reruns[victim]["suspected_dependency"] is True
    assert reruns[broken]["suspected_dependency"] is False
    total = next(r for r in report(smartpar_dir) if r["worker_id"] == "ALL")
    assert (total["reruns"], total["suspected_dependency"], total["failures"]) == ("2", "1", "1")
    cases = junit_cases(smartpar_dir / "results.xml")
    assert props(cases["test_new_victim"]) == {
        "smartpar_rerun": "passed",
        "smartpar_suspected_dependency": "true",
        "smartpar_first_outcome": "failed",
    }
    assert cases["test_new_victim"].find("failure") is None
    assert cases["test_new_broken"].find("failure") is not None
    root = ET.parse(smartpar_dir / "results.xml").getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite.get("failures") == "1"


def test_rerun_that_passes_makes_session_green(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_new="\n".join(NEW_TESTS.splitlines()[:-2]))
    polluter, victim = "test_new.py::test_polluter", "test_new.py::test_new_victim"
    shared = FileAccess(W=["<project>/f"])
    write_data(
        pytester.path,
        {polluter: 0.01},
        file_access={polluter: shared, victim: shared},
        collection=[polluter, victim],
    )
    result = pytester.runpytest_subprocess("-n", "2", "--smartpar", timeout=TIMEOUT)
    assert result.ret == 0  # the only failure passed alone: reported, not hidden
    result.stdout.fnmatch_lines(["*suspected dependency*"])


def test_bootstrap_without_data_uses_smartpar_scheduler(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_plain=SLEEPY)
    result = pytester.runpytest_subprocess("-n", "2", "--smartpar", "-v", timeout=TIMEOUT)
    result.assert_outcomes(passed=9)
    result.stdout.fnmatch_lines(
        [
            "*no data yet; run `pytest --smartpar-profile`*",
            "*no measured durations yet: every test is estimated equally*",
            "*scheduling tests via SmartparScheduling*",
        ]
    )
    smartpar_dir = pytester.path / ".smartpar"
    run = last_run(smartpar_dir)
    assert run["mode"] == "smartpar-bootstrap"
    assert all(t["duration"] > 0 for t in run["tests"])
    plan = json.loads((smartpar_dir / "plan.json").read_text())
    assert plan["mode"] == "smartpar-bootstrap" and plan["overhead_model"] == "default"
    history.merge(smartpar_dir)
    assert all(h.timed_runs == 1 for h in history.load_history(smartpar_dir).tests.values())


def test_bootstrap_keeps_groups_estimated_by_test_count(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_plain=SLEEPY)
    ids = [f"test_plain.py::{n}" for n in ("test_s1", "test_s2", "test_s3")]
    smartpar_dir = pytester.path / ".smartpar"
    deps = Dependencies(file_access=dict.fromkeys(ids, FileAccess(W=["<project>/f"])))
    history.atomic_write_json(smartpar_dir / "dependencies.json", deps.to_dict())
    result = pytester.runpytest_subprocess("-n", "2", "--smartpar", timeout=TIMEOUT)
    result.assert_outcomes(passed=9)
    plan = json.loads((smartpar_dir / "plan.json").read_text())
    groups = [u for units in plan["assignments"].values() for u in units if u[0] != "single"]
    assert [sorted(g[1]) for g in groups] == [ids]
    tests = by_name(last_run(smartpar_dir))
    assert len({tests[n.split("::")[1]]["worker"] for n in ids}) == 1


def test_smartpar_requires_workers(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest_subprocess("--smartpar")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--smartpar needs pytest-xdist workers*"])


@pytest.mark.skipif(available_cpus() < 2, reason="needs 2 CPUs")
def test_auto_workers_uses_recommendation(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("import time\ndef test_a(): time.sleep(0.2)\ndef test_b(): time.sleep(0.2)")
    smartpar_dir = write_data(
        pytester.path,
        {
            "test_auto_workers_uses_recommendation.py::test_a": 1.0,
            "test_auto_workers_uses_recommendation.py::test_b": 1.0,
        },
    )
    # T(1) = 2, T(2) = 1, T(m>2) = 1: the smallest m within 5% of the best is 2.
    result = pytester.runpytest_subprocess("-n", "auto", "--smartpar", timeout=TIMEOUT)
    result.assert_outcomes(passed=2)
    plan = json.loads((smartpar_dir / "plan.json").read_text())
    assert plan["workers"] == 2 and plan["recommended_workers"] == 2


CRASHY = """
import os, time
def test_1(): time.sleep(0.05)
def test_2(): os._exit(1)  # kills its worker
def test_3(): time.sleep(0.05)
def test_4(): time.sleep(0.05)
def test_5(): time.sleep(0.05)
def test_6(): time.sleep(0.05)
"""


def test_worker_crash_is_survived(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_crashy=CRASHY)
    ids = [f"test_crashy.py::test_{i}" for i in range(1, 7)]
    shared = FileAccess(W=["<project>/f"])
    # 1-4 in one group (one worker, in order): 3 and 4 are left when 2 crashes.
    smartpar_dir = write_data(
        pytester.path,
        dict.fromkeys(ids, 0.05),
        file_access={i: shared for i in ids[:4]},
        collection=ids,
        flaky=nod_flaky(ids[5]),
    )
    result = pytester.runpytest_subprocess("-n", "2", "--smartpar", timeout=TIMEOUT)
    result.assert_outcomes(passed=5, failed=1)
    run = last_run(smartpar_dir)
    assert {t["nodeid"] for t in run["tests"]} == set(ids)


TWO_MODULES = {
    "test_m1": "def test_a(): pass\ndef test_b(): pass\n",
    "test_m2": "def test_c(): pass\n",
}


def test_child_run_keeps_argument_order_across_modules(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    pytester.makepyfile(**TWO_MODULES)
    order = ["test_m2.py::test_c", "test_m1.py::test_b", "test_m1.py::test_a"]
    config = pytester.parseconfig()
    cmd = child_command(config, order, tmp_path)
    proc = pytester.run(*cmd, timeout=TIMEOUT)
    assert proc.ret == 0
    record = json.loads((tmp_path / "report.json").read_text())
    ran = [t["nodeid"] for t in sorted(record["tests"], key=lambda t: t["start"])]
    assert ran == order


def test_strict_fails_on_suspected_dependency(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_new="\n".join(NEW_TESTS.splitlines()[:-2]))
    polluter, victim = "test_new.py::test_polluter", "test_new.py::test_new_victim"
    shared = FileAccess(W=["<project>/f"])
    smartpar_dir = write_data(
        pytester.path,
        {polluter: 0.01},
        file_access={polluter: shared, victim: shared},
        collection=[polluter, victim],
    )
    result = pytester.runpytest_subprocess(
        "-n", "2", "--smartpar", "--smartpar-strict", timeout=TIMEOUT
    )
    assert result.ret == 1
    result.stdout.fnmatch_lines(
        [
            "*- smartpar (smartpar) -*",
            "*failures: 1 before re-runs, 0 after; tail: none*",
            "*--smartpar-strict: suspected dependencies fail the session*",
        ]
    )
    run = last_run(smartpar_dir)
    assert (run["failures_before_rerun"], run["failures_after_rerun"]) == (1, 0)
    total = next(r for r in report(smartpar_dir) if r["worker_id"] == "ALL")
    assert (total["failures_before_rerun"], total["failures_after_rerun"]) == ("1", "0")


def test_fallback_tail_failure_fails_session(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_tailfail="def test_ok(): pass\ndef test_nod(): assert False\n")
    ok, nod = "test_tailfail.py::test_ok", "test_tailfail.py::test_nod"
    smartpar_dir = write_data(
        pytester.path, {ok: 0.01, nod: 0.01}, collection=[ok, nod], flaky=nod_flaky(nod)
    )
    result = pytester.runpytest_subprocess(
        "-n", "2", "--smartpar", "--smartpar-fallback", timeout=TIMEOUT
    )
    result.assert_outcomes(passed=1)  # the main session itself is green
    assert result.ret == 1  # ... but the separate tail pass failed
    result.stdout.fnmatch_lines(
        [
            "*- smartpar (smartpar-fallback) -*",
            "*isolated tail: 1 tests in a separate sequential pass (failed)*",
            "*failures: 1 before re-runs, 1 after; tail: separate-pass*",
        ]
    )
    cases = junit_cases(smartpar_dir / "results.xml")
    assert cases["test_nod"].find("failure") is not None
