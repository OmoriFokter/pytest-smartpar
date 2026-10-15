"""Timing plugin loaded into every benchmark configuration (``-p bench.timing_plugin``).

Writes one JSON line per test to the file named by the environment variable
``SMARTPAR_BENCH_TIMING``: ``nodeid, worker, start, end, duration, outcome``.
The same plugin in serial, xdist and smartpar runs makes Gantt charts and
worker utilisation comparable across configurations.

Only the process that receives all reports writes (the serial session or the
xdist controller), so there is a single writer per file. Tests run by
smartpar's post-session child processes (re-runs, separate tail pass) are not
in this file; they are in smartpar's own run file.
"""

from __future__ import annotations

import json
import os
import re

import pytest

ENV_VAR = "SMARTPAR_BENCH_TIMING"
_GROUP_SUFFIX = re.compile(r"@[^\]@]+$")  # xdist_group suffix (loadgroup)


class TimingRecorder:
    def __init__(self, path: str) -> None:
        self.path = path
        self.tests: dict[str, dict] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        node = getattr(report, "node", None)
        worker = node.gateway.id if node is not None else "master"
        nodeid = _GROUP_SUFFIX.sub("", report.nodeid)
        rec = self.tests.setdefault(
            nodeid,
            {
                "nodeid": nodeid,
                "worker": worker,
                "start": report.start,
                "end": report.stop,
                "duration": 0.0,
                "outcome": "passed",
            },
        )
        rec["start"] = min(rec["start"], report.start)
        rec["end"] = max(rec["end"], report.stop)
        rec["duration"] += report.duration
        xfail = hasattr(report, "wasxfail")
        if report.when == "setup" and report.failed:
            rec["outcome"] = "error"
        elif report.when in ("setup", "call"):
            if report.passed:
                rec["outcome"] = "xpassed" if xfail else "passed"
            elif report.skipped:
                rec["outcome"] = "xfailed" if xfail else "skipped"
            else:
                rec["outcome"] = "failed"
        elif report.failed and rec["outcome"] not in ("failed", "error"):
            rec["outcome"] = "error"

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for rec in self.tests.values():
                fh.write(json.dumps(rec) + "\n")


def pytest_configure(config: pytest.Config) -> None:
    path = os.environ.get(ENV_VAR)
    if path and not hasattr(config, "workerinput"):
        config.pluginmanager.register(TimingRecorder(path), "bench-timing")
