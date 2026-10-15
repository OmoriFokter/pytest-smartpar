"""Phase 5: report.csv rows, results.xml merging and on-the-fly re-planning."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from smartpar import plugin
from smartpar.models import History, RunRecord, TestHistory, TestRecord
from smartpar.report import REPORT_COLUMNS, append_report_csv, merge_junit, report_rows


def run_record(**extra: object) -> RunRecord:
    tests = [
        TestRecord("t::a", "passed", 2.0, "gw0", 0, 2),
        TestRecord("t::b", "failed", 1.0, "gw1", 0, 1),
        TestRecord("t::c", "passed", 0.5, "gw1", 1, 1.5),
    ]
    fields: dict = dict(
        run_id="r1",
        mode="smartpar",
        workers=2,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:03+00:00",
        wall_time_s=2.5,
        tests=tests,
        predicted_wall_s=2.0,
        tail_mode="barrier",
    )
    fields.update(extra)
    return RunRecord(**fields)


def th(d: float) -> TestHistory:
    return TestHistory(d, 1, "passed", 0, 1)


def test_report_rows() -> None:
    hist = History(tests={"t::a": th(2.0), "t::b": th(1.0), "t::c": th(2.0)})
    rows = {r["worker_id"]: r for r in report_rows(run_record(), hist)}
    assert list(rows) == ["gw0", "gw1", "ALL"]
    assert rows["gw0"]["worker_busy_s"] == "2.0000"
    assert rows["gw0"]["worker_idle_s"] == "0.5000"
    assert rows["gw1"]["failures"] == "1"
    total = rows["ALL"]
    assert total["wall_time_s"] == "2.5000"
    assert total["predicted_wall_s"] == "2.0000"
    assert total["prediction_error_pct"] == "25.00"  # (2.5 - 2.0) / 2.0
    assert total["sum_test_time_s"] == "3.5000"
    assert total["speedup_vs_history_serial"] == "2.000"  # (2 + 1 + 2) / 2.5
    assert (total["worker_tests"], total["failures"], total["tail_mode"]) == ("3", "1", "barrier")


def test_report_rows_with_reruns() -> None:
    reruns = [
        {
            "nodeid": "t::b",
            "first_outcome": "failed",
            "rerun_outcome": "passed",
            "duration": 0.7,
            "suspected_dependency": True,
        },
    ]
    rows = {r["worker_id"]: r for r in report_rows(run_record(reruns=reruns), History())}
    assert rows["post:rerun"]["worker_tests"] == "1"
    assert rows["post:rerun"]["failures"] == "0"
    assert rows["ALL"]["failures"] == "0"  # b passed alone
    assert (rows["ALL"]["reruns"], rows["ALL"]["suspected_dependency"]) == ("1", "1")
    assert rows["ALL"]["speedup_vs_history_serial"] == ""  # no history


def test_report_csv_appends_with_one_header(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    rows = report_rows(run_record(), History())
    append_report_csv(path, rows)
    append_report_csv(path, rows)
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(REPORT_COLUMNS)
    assert sum(line.startswith("run_id") for line in lines) == 1
    with open(path, newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2 * len(rows)


MAIN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="0" tests="2" time="1">
<testcase classname="t" name="a" time="0.1"/>
<testcase classname="t" name="b" time="0.2"><failure message="x">boom</failure></testcase>
</testsuite></testsuites>"""
RERUN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1" time="1">
<testcase classname="t" name="b" time="0.3"/></testsuite></testsuites>"""
TAIL = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="1" tests="1" time="1">
<testcase classname="t" name="z" time="0.3"><skipped message="s"/></testcase>
</testsuite></testsuites>"""


def test_merge_junit_replace_and_append(tmp_path: Path) -> None:
    main, rerun, tail = tmp_path / "main.xml", tmp_path / "rerun.xml", tmp_path / "tail.xml"
    main.write_text(MAIN)
    rerun.write_text(RERUN)
    tail.write_text(TAIL)
    merge_junit(main, rerun, {"smartpar_rerun": "passed"}, replace_existing=True)
    merge_junit(main, tail, {"smartpar_tail": "separate-pass"}, replace_existing=False)
    suite = ET.parse(main).getroot().find("testsuite")
    cases = suite.findall("testcase")
    assert [c.get("name") for c in cases] == ["a", "b", "z"]  # b replaced in place
    b_props = {p.get("name"): p.get("value") for p in cases[1].iter("property")}
    assert b_props == {"smartpar_rerun": "passed", "smartpar_first_outcome": "failed"}
    assert cases[1].find("failure") is None
    assert (suite.get("tests"), suite.get("failures"), suite.get("skipped")) == ("3", "0", "1")


def test_replanning_for_a_different_worker_count(pytester: pytest.Pytester) -> None:
    config = pytester.parseconfig("-p", "no:randomly")
    config.stash[plugin.RANDOMLY_REORDER_DISABLED] = False
    hist = History(tests={f"t::{i}": th(float(d)) for i, d in enumerate([5, 4, 3, 3, 2, 2, 1])})
    collector = plugin.RunCollector(plugin.MODE_SMARTPAR, 4, False)
    run = plugin.SmartparRun(
        config, pytester.path / ".smartpar", plugin.MODE_SMARTPAR, 4, (hist, None, None), collector
    )
    collection = sorted(hist.tests)
    planned = run.planner(collection, ["gw0", "gw1", "gw2"])  # 3 instead of 4
    assert len(planned.assignments) == 3
    assert sorted(t for a in planned.assignments for t in a) == collection
    plan = json.loads((pytester.path / ".smartpar" / "plan.json").read_text())
    assert (plan["workers"], plan["requested_workers"], plan["replanned"]) == (3, 4, True)
    # LPT for 3 workers: 5 | 4 + 2 + 1 ... deterministic loads.
    assert sorted(plan["predicted_loads"].values()) == [6.0, 7.0, 7.0]
    assert plan["predicted_lpt_makespan_s"] == 7.0
