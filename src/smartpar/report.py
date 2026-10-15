"""Method step 5 (recording): run records, plan.json, report.csv and metrics.

``RunCollector`` turns pytest's per-phase reports into one ``TestRecord`` per
test (duration = setup + call + teardown) on the process that receives the
reports: the only process in a serial session, the controller under xdist.

Also: plan.json, report.csv rows and merging child-run results into the
JUnit file (results.xml).
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from smartpar import history
from smartpar.history import atomic_write_bytes, new_run_id
from smartpar.models import FAILED_OUTCOMES, History, RunRecord, TestRecord, envelope
from smartpar.scheduler import Analysis, worker_name
from smartpar.xdist_scheduling import strip_group_suffix

log = logging.getLogger("smartpar.report")

# Worker id used for serial sessions (same as xdist's ``worker_id`` fixture).
SERIAL_WORKER = "master"


@dataclass
class _Acc:
    outcome: str
    duration: float
    worker: str
    start: float
    end: float


def _phase_outcome(report: pytest.TestReport) -> str:
    """Outcome of the test as decided by its setup or call phase."""
    xfail = hasattr(report, "wasxfail")
    if report.when == "setup" and report.failed:
        return "error"
    if report.passed:
        return "xpassed" if xfail else "passed"
    if report.skipped:
        return "xfailed" if xfail else "skipped"
    return "failed"


class RunCollector:
    """pytest plugin object collecting the raw data of one run.

    The clock starts when the collector is created (pytest_configure), so the
    wall time includes starting and collecting on xdist workers.
    """

    def __init__(self, mode: str, workers: int, randomly_reorder_disabled: bool) -> None:
        self.mode = mode
        self.workers = workers
        self.randomly_reorder_disabled = randomly_reorder_disabled
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self.tests: dict[str, _Acc] = {}

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        node = getattr(report, "node", None)  # set by xdist on the controller
        worker = node.gateway.id if node is not None else SERIAL_WORKER
        nodeid = strip_group_suffix(report.nodeid)
        acc = self.tests.get(nodeid)
        if acc is None:
            acc = self.tests[nodeid] = _Acc("passed", 0.0, worker, report.start, report.stop)
        acc.duration += report.duration
        acc.start = min(acc.start, report.start)
        acc.end = max(acc.end, report.stop)
        if report.when in ("setup", "call"):  # call only runs after a passed setup
            acc.outcome = _phase_outcome(report)
        elif report.failed and acc.outcome not in ("failed", "error"):
            acc.outcome = "error"  # teardown failure

    def add(self, t: TestRecord) -> None:
        """A test that ran outside this session (separate tail pass)."""
        self.tests[t.nodeid] = _Acc(t.outcome, t.duration, t.worker, t.start, t.end)

    def record(self, **extra: object) -> RunRecord:
        finished = datetime.now(timezone.utc)
        return RunRecord(
            run_id=new_run_id(now=self.started_at),
            mode=self.mode,
            workers=self.workers,
            started_at=self.started_at.isoformat(),
            finished_at=finished.isoformat(),
            wall_time_s=self.elapsed(),
            tests=[
                TestRecord(nodeid, a.outcome, a.duration, a.worker, a.start, a.end)
                for nodeid, a in self.tests.items()
            ],
            randomly_reorder_disabled=self.randomly_reorder_disabled,
            **extra,  # type: ignore[arg-type]
        )


class ProfileRunWriter:
    """--smartpar-profile: write the run file and merge it (counters only)."""

    def __init__(self, collector: RunCollector, smartpar_dir: Path, alpha: float) -> None:
        self.collector = collector
        self.smartpar_dir = smartpar_dir
        self.alpha = alpha

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if not self.collector.tests:
            return
        path = history.write_run(self.smartpar_dir, self.collector.record())
        log.info("smartpar: run written to %s", path)
        history.merge(self.smartpar_dir, alpha=self.alpha)


class ChildReportWriter:
    """In a child pytest process (re-run, tail pass): dump the record as JSON."""

    def __init__(self, collector: RunCollector, path: Path) -> None:
        self.collector = collector
        self.path = path

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        history.atomic_write_json(self.path, self.collector.record().to_dict())


# --- plan.json ----------------------------------------------------------------------


def plan_to_dict(
    analysis: Analysis,
    worker_ids: list[str],
    requested_workers: int,
    mode: str,
    tail_worker: str | None,
    randomly_reorder_disabled: bool,
    token: str = "",
) -> dict[str, Any]:
    """plan.json (spec 3.4) for the actual workers ``worker_ids`` (plan order)."""
    plan = analysis.schedule
    return {
        **envelope(),
        "mode": mode,
        "token": token,
        "workers": len(worker_ids),
        "requested_workers": requested_workers,
        "replanned": len(worker_ids) != requested_workers,
        "recommended_workers": analysis.recommended_workers,
        "predicted_makespan_s": plan.predicted_makespan_s,
        "predicted_lpt_makespan_s": plan.lpt_makespan_s,
        "predicted_tail_s": plan.tail_s,
        "predicted_overhead_s": plan.overhead_s,
        "overhead_model": analysis.overhead_model,
        "predicted_curve": {str(m): t for m, t in analysis.curve.items()},
        "assignments": {
            wid: [[u.id, list(u.tests)] for u in plan.assignments[worker_name(i)]]
            for i, wid in enumerate(worker_ids)
        },
        "predicted_loads": {wid: plan.loads[worker_name(i)] for i, wid in enumerate(worker_ids)},
        "isolated_tail": list(plan.tail),
        "tail_worker": tail_worker,
        "oversized_groups": plan.oversized_groups,
        "new_tests": analysis.estimates.new_tests,
        "randomly_reorder_disabled": randomly_reorder_disabled,
    }


# --- report.csv ---------------------------------------------------------------------

REPORT_COLUMNS = [
    "run_id",
    "mode",
    "workers",
    "wall_time_s",
    "predicted_wall_s",
    "prediction_error_pct",
    "sum_test_time_s",
    "speedup_vs_history_serial",
    "worker_id",
    "worker_busy_s",
    "worker_idle_s",
    "worker_tests",
    "failures",
    "failures_before_rerun",
    "failures_after_rerun",
    "reruns",
    "suspected_dependency",
    "tail_mode",
]
ALL_WORKERS = "ALL"
RERUN_WORKER = "post:rerun"


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def report_rows(run: RunRecord, history_before: History) -> list[dict[str, str]]:
    """One row per worker (incl. post-session pseudo-workers) plus ``ALL``.

    ``failures`` per worker counts first attempts; on the ALL row it counts
    final failures (a new test that passed on its re-run is not one).
    ``failures_before_rerun`` / ``failures_after_rerun`` are run totals on
    every row (for comparisons with plain xdist, which has no re-runs).
    ``speedup_vs_history_serial`` = sum of history EMAs of the run's tests
    (known ones) / wall time.
    """
    wall = run.wall_time_s
    serial = sum(
        history_before.tests[t.nodeid].duration_ema
        for t in run.tests
        if t.nodeid in history_before.tests and history_before.tests[t.nodeid].timed_runs > 0
    )
    predicted = run.predicted_wall_s
    error = (wall - predicted) / predicted * 100 if predicted else None
    suspected = sum(1 for r in run.reruns if r.get("suspected_dependency"))
    common = {
        "run_id": run.run_id,
        "mode": run.mode,
        "workers": str(run.workers),
        "wall_time_s": _fmt(wall),
        "predicted_wall_s": _fmt(predicted),
        "prediction_error_pct": _fmt(error, 2),
        "sum_test_time_s": _fmt(sum(t.duration for t in run.tests)),
        "speedup_vs_history_serial": _fmt(serial / wall if serial and wall else None, 3),
        "failures_before_rerun": str(run.failures_before_rerun),
        "failures_after_rerun": str(run.failures_after_rerun),
        "reruns": str(len(run.reruns)),
        "suspected_dependency": str(suspected),
        "tail_mode": run.tail_mode,
    }
    per_worker: dict[str, list[TestRecord]] = {}
    for t in run.tests:
        per_worker.setdefault(t.worker, []).append(t)
    rows = []
    busy_total = idle_total = 0.0
    for worker in sorted(per_worker, key=_worker_order):
        tests = per_worker[worker]
        busy = sum(t.duration for t in tests)
        idle = max(0.0, wall - busy)
        busy_total += busy
        idle_total += idle
        rows.append(
            {
                **common,
                "worker_id": worker,
                "worker_busy_s": _fmt(busy),
                "worker_idle_s": _fmt(idle),
                "worker_tests": str(len(tests)),
                "failures": str(sum(t.outcome in FAILED_OUTCOMES for t in tests)),
            }
        )
    if run.reruns:
        busy = sum(float(r.get("duration", 0.0)) for r in run.reruns)
        rows.append(
            {
                **common,
                "worker_id": RERUN_WORKER,
                "worker_busy_s": _fmt(busy),
                "worker_idle_s": "",
                "worker_tests": str(len(run.reruns)),
                "failures": str(sum(r.get("rerun_outcome") in FAILED_OUTCOMES for r in run.reruns)),
            }
        )
    first_failures = {t.nodeid for t in run.tests if t.outcome in FAILED_OUTCOMES}
    passed_on_rerun = {r["nodeid"] for r in run.reruns if r.get("suspected_dependency")}
    rows.append(
        {
            **common,
            "worker_id": ALL_WORKERS,
            "worker_busy_s": _fmt(busy_total),
            "worker_idle_s": _fmt(idle_total),
            "worker_tests": str(len(run.tests)),
            "failures": str(len(first_failures - passed_on_rerun)),
        }
    )
    return rows


def _worker_order(worker: str) -> tuple[int, int, str]:
    digits = re.sub(r"\D", "", worker)
    return (worker.startswith("post:"), int(digits) if digits else 0, worker)


def append_report_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Append rows (header when the file is new) with a single O_APPEND write."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REPORT_COLUMNS, lineterminator="\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if new:
        writer.writeheader()
    writer.writerows(rows)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, buf.getvalue().encode())
    finally:
        os.close(fd)


# --- results.xml (JUnit) ------------------------------------------------------------


def _testcases(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """(testsuite, testcase) pairs."""
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return [(suite, case) for suite in suites for case in list(suite.findall("testcase"))]


def _add_properties(case: ET.Element, properties: dict[str, str]) -> None:
    props = case.find("properties")
    if props is None:
        props = ET.Element("properties")
        case.insert(0, props)
    for name, value in properties.items():
        ET.SubElement(props, "property", name=name, value=value)


def _recount(root: ET.Element) -> None:
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        cases = suite.findall("testcase")
        suite.set("tests", str(len(cases)))
        suite.set("failures", str(sum(c.find("failure") is not None for c in cases)))
        suite.set("errors", str(sum(c.find("error") is not None for c in cases)))
        suite.set("skipped", str(sum(c.find("skipped") is not None for c in cases)))


def merge_junit(
    main: Path, child: Path, properties: dict[str, str], replace_existing: bool
) -> None:
    """Merge the testcases of a child run's JUnit file into ``main``.

    ``replace_existing``: a testcase with the same classname and name in
    ``main`` (the first attempt of a re-run test) is replaced in place, its
    first outcome recorded as a property; otherwise testcases are appended.
    Every merged testcase gets ``properties``, so nothing is hidden.
    """
    main_root = ET.parse(main).getroot()
    child_root = ET.parse(child).getroot()
    pairs = _testcases(main_root)
    suites = [main_root] if main_root.tag == "testsuite" else list(main_root.iter("testsuite"))
    if not suites:
        raise ValueError(f"{main}: no testsuite element")
    for _suite, case in _testcases(child_root):
        props = dict(properties)
        target = None
        if replace_existing:
            key = (case.get("classname"), case.get("name"))
            for suite, existing in pairs:
                if (existing.get("classname"), existing.get("name")) == key:
                    target = (suite, existing)
                    break
        if target is not None:
            suite, existing = target
            props["smartpar_first_outcome"] = _junit_outcome(existing)
            suite.insert(list(suite).index(existing), case)
            suite.remove(existing)
        else:
            suites[0].append(case)
        _add_properties(case, props)
    _recount(main_root)
    atomic_write_bytes(main, ET.tostring(main_root, encoding="utf-8", xml_declaration=True))


def _junit_outcome(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        if case.find(tag) is not None:
            return "failed" if tag == "failure" else tag
    return "passed"
