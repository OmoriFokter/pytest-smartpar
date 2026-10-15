"""Method step 5 (recording): run records, plan.json, report.csv and metrics.

``RunCollector`` turns pytest's per-phase reports into one ``TestRecord`` per
test (duration = setup + call + teardown) on the process that receives the
reports: the only process in a serial session, the controller under xdist.

plan.json and report.csv are added in phase 5.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smartpar import history
from smartpar.history import new_run_id
from smartpar.models import RunRecord, TestRecord

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
    """pytest plugin object collecting the raw data of one run."""

    def __init__(
        self,
        mode: str,
        workers: int,
        randomly_reorder_disabled: bool,
        smartpar_dir: Path,
        merge_alpha: float | None = None,
    ) -> None:
        """``merge_alpha`` set: merge into history.json right after writing the run file."""
        self.mode = mode
        self.smartpar_dir = smartpar_dir
        self.merge_alpha = merge_alpha
        self.workers = workers
        self.randomly_reorder_disabled = randomly_reorder_disabled
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self._tests: dict[str, _Acc] = {}

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        node = getattr(report, "node", None)  # set by xdist on the controller
        worker = node.gateway.id if node is not None else SERIAL_WORKER
        acc = self._tests.get(report.nodeid)
        if acc is None:
            acc = self._tests[report.nodeid] = _Acc(
                "passed", 0.0, worker, report.start, report.stop
            )
        acc.duration += report.duration
        acc.start = min(acc.start, report.start)
        acc.end = max(acc.end, report.stop)
        if report.when in ("setup", "call"):  # call only runs after a passed setup
            acc.outcome = _phase_outcome(report)
        elif report.failed and acc.outcome not in ("failed", "error"):
            acc.outcome = "error"  # teardown failure

    def record(self) -> RunRecord:
        finished = datetime.now(timezone.utc)
        return RunRecord(
            run_id=new_run_id(now=self.started_at),
            mode=self.mode,
            workers=self.workers,
            started_at=self.started_at.isoformat(),
            finished_at=finished.isoformat(),
            wall_time_s=time.perf_counter() - self._t0,
            tests=[
                TestRecord(nodeid, a.outcome, a.duration, a.worker, a.start, a.end)
                for nodeid, a in self._tests.items()
            ],
            randomly_reorder_disabled=self.randomly_reorder_disabled,
        )

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if not self._tests:
            return
        path = history.write_run(self.smartpar_dir, self.record())
        log.info("smartpar: run written to %s", path)
        if self.merge_alpha is not None:
            history.merge(self.smartpar_dir, alpha=self.merge_alpha)
