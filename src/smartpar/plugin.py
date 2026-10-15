"""Integration of the method into pytest: command-line options and mode dispatch.

Registered as the ``smartpar`` plugin through the ``pytest11`` entry point.
Modes: ``--smartpar-profile`` (step 1, profiling), ``--smartpar-detect``
(step 1, order-dependency detection) and ``--smartpar`` (steps 2-5: grouping,
scheduling, execution and recording).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from smartpar import history as hist_io
from smartpar.detector import (
    CONFIRM_RUNS,
    MAX_ORDERS_PER_TEST,
    MAX_PAIR_CHECKS,
    BudgetExceeded,
    Detector,
    ProcessRunner,
    Sandbox,
    choose_seeds,
)
from smartpar.graph import build as build_graph
from smartpar.models import FAILED_OUTCOMES, Dependencies, Flaky, History, SchemaError
from smartpar.profiler import (
    DEFAULT_IGNORE_STATE,
    DEPENDENCIES_FILE,
    ProfilerPlugin,
    load_dependencies,
)
from smartpar.report import (
    ChildReportWriter,
    ProfileRunWriter,
    RunCollector,
    append_report_csv,
    merge_junit,
    plan_to_dict,
    report_rows,
)
from smartpar.scheduler import (
    Analysis,
    analyze,
    available_cpus,
    estimate_durations,
    recommend_workers,
    worker_name,
)
from smartpar.xdist_scheduling import (
    REQUIRED_NODE_API,
    FallbackItems,
    PlannedRun,
    SmartparScheduling,
    parse_tx_spec_config,
    run_child_pytest,
)

log = logging.getLogger("smartpar.plugin")

PLAN_FILE = "plan.json"
REPORT_FILE = "report.csv"
RESULTS_FILE = "results.xml"

# pytest-xdist versions whose internal scheduler interface is covered by the
# tests (3.3.1, 3.4.0, 3.5.0, 3.6.1, 3.7.0, 3.8.0; see README and pyproject.toml);
# outside this range the loadgroup fallback is used.
SUPPORTED_XDIST = ((3, 3), (3, 9))  # [min, max)

MODE_SMARTPAR = "smartpar"
MODE_FALLBACK = "smartpar-fallback"
MODE_BOOTSTRAP = "smartpar-bootstrap"
# Modes executed by SmartparScheduling (bootstrap: no durations yet, every
# test estimated equally, so a group's estimate is its number of tests).
SCHEDULED_MODES = frozenset({MODE_SMARTPAR, MODE_BOOTSTRAP})

# True when smartpar switched off pytest-randomly's reordering for this session.
# Written to plan.json / report.csv as ``randomly_reorder_disabled``.
RANDOMLY_REORDER_DISABLED = pytest.StashKey[bool]()

RANDOMLY_REORDER_WARNING = (
    "smartpar: pytest-randomly test reordering disabled (--randomly-dont-reorganize) "
    "to keep the planned/profiled order; random seeding stays active"
)


PROFILER = pytest.StashKey[ProfilerPlugin]()
RUN = pytest.StashKey["SmartparRun"]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "smartpar_max_pair_checks",
        type="string",
        default=str(MAX_PAIR_CHECKS),
        help="--smartpar-detect: direct [P, V] checks per search before delta debugging",
    )
    parser.addini(
        "smartpar_max_orders_per_victim",
        type="string",
        default=str(MAX_ORDERS_PER_TEST),
        help="--smartpar-detect: failing (passing) orders examined per victim (brittle test)",
    )
    parser.addini(
        "smartpar_ignore_state",
        type="linelist",
        default=list(DEFAULT_IGNORE_STATE),
        help="fnmatch patterns of state keys ignored by --smartpar-profile "
        "(replaces the default list: " + ", ".join(DEFAULT_IGNORE_STATE) + ")",
    )
    group = parser.getgroup("smartpar", "dependency-aware parallel test execution")
    group.addoption(
        "--smartpar",
        action="store_true",
        default=False,
        help="Run tests according to the smartpar plan (use with -n).",
    )
    group.addoption(
        "--smartpar-profile",
        action="store_true",
        default=False,
        help="Profile tests: durations, file R/W sets and state changes.",
    )
    group.addoption(
        "--smartpar-detect",
        action="store_true",
        default=False,
        help="Detect order-dependent tests with randomized runs in child processes.",
    )
    group.addoption(
        "--smartpar-runs",
        type=int,
        default=20,
        help="Number of random-order runs for --smartpar-detect (default: 20).",
    )
    group.addoption(
        "--smartpar-seed",
        type=int,
        default=None,
        help="Base seed for --smartpar-detect (default: chosen and recorded).",
    )
    group.addoption(
        "--smartpar-detect-budget",
        type=float,
        default=None,
        help="Time budget in seconds for --smartpar-detect.",
    )
    group.addoption(
        "--smartpar-detect-workers",
        type=int,
        default=None,
        help="Parallel child runs for --smartpar-detect, each in its own project copy "
        "(default: available CPUs).",
    )
    group.addoption(
        "--smartpar-fallback",
        action="store_true",
        default=False,
        help="Use the --dist loadgroup fallback instead of the smartpar scheduler "
        "(diagnostics; chosen automatically for unsupported pytest-xdist versions).",
    )
    group.addoption(
        "--smartpar-strict",
        action="store_true",
        default=False,
        help="Fail the session if a new test failed but passed when re-run alone "
        "(suspected dependency); by default such a test counts as passed.",
    )
    group.addoption(
        "--smartpar-child-report",
        default=None,
        help="Internal: write the run record of this (child) session to this JSON file.",
    )
    group.addoption(
        "--smartpar-dir",
        default=".smartpar",
        help="Directory for smartpar data files (default: .smartpar).",
    )
    group.addoption(
        "--smartpar-alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor for test durations (default: 0.3).",
    )


def _randomly_active(config: pytest.Config) -> bool:
    return config.pluginmanager.hasplugin("randomly")


def apply_randomly_policy(config: pytest.Config) -> None:
    """Reconcile pytest-randomly with the selected smartpar mode.

    --smartpar-detect needs random orders, so pytest-randomly is mandatory.
    --smartpar and --smartpar-profile need a controlled order, so only the
    reordering of pytest-randomly is switched off (seeding of ``random`` etc.
    is left as configured by the user).
    """
    if config.getoption("smartpar_detect") and not _randomly_active(config):
        raise pytest.UsageError(
            "smartpar: --smartpar-detect requires pytest-randomly to shuffle test order, "
            "but it is not installed or disabled (-p no:randomly). "
            "Install it with: pip install pytest-randomly"
        )
    disabled = False
    if (
        (config.getoption("smartpar") or config.getoption("smartpar_profile"))
        and _randomly_active(config)
        and config.getoption("randomly_reorganize")
    ):
        config.option.randomly_reorganize = False
        disabled = True
    config.stash[RANDOMLY_REORDER_DISABLED] = disabled


# trylast: the terminal reporter is registered by pytest's own pytest_configure.
@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    apply_randomly_policy(config)
    is_controller = not hasattr(config, "workerinput")
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if config.stash[RANDOMLY_REORDER_DISABLED] and is_controller and terminal is not None:
        terminal.write_line(RANDOMLY_REORDER_WARNING, yellow=True)

    if config.getoption("smartpar_child_report"):
        collector = RunCollector("child", 1, config.stash[RANDOMLY_REORDER_DISABLED])
        config.pluginmanager.register(collector, "smartpar-run-collector")
        config.pluginmanager.register(
            ChildReportWriter(collector, Path(config.getoption("smartpar_child_report"))),
            "smartpar-child-report",
        )
    if config.getoption("smartpar_profile"):
        _start_profiling(config)
    elif config.getoption("smartpar"):
        _start_smartpar(config)
    elif config.getoption("smartpar_detect"):
        if is_xdist_controller(config):
            raise pytest.UsageError(
                "smartpar: --smartpar-detect runs its own child processes; do not use -n"
            )
        config.pluginmanager.register(
            DetectSession(config, _smartpar_dir(config)), "smartpar-detect"
        )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """--smartpar always produces results.xml: enable --junitxml if not given."""
    if config.getoption("smartpar") and not config.getoption("xmlpath"):
        config.option.xmlpath = str(_smartpar_dir(config) / RESULTS_FILE)


# --- xdist hooks (optional: the plugin also works with -p no:xdist) -----------------


# tryfirst: xdist's own implementation answers too (firstresult); which one
# pluggy calls first otherwise depends on the plugin load (install) order.
@pytest.hookimpl(optionalhook=True, tryfirst=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """-n auto with --smartpar: the recommended m, capped by available CPUs."""
    if not config.getoption("smartpar"):
        return None
    try:
        hist, deps, flaky = _load_data(_smartpar_dir(config))
    except SchemaError:
        return None
    tests = set(hist.tests) | (set(deps.file_access) if deps else set())
    if not tests or not any(h.timed_runs for h in hist.tests.values()):
        return None  # no timings: xdist's default
    cpus = available_cpus()
    a = analyze(tests, hist, deps, flaky, max_workers=cpus)
    return min(recommend_workers(a.curve), cpus)


@pytest.hookimpl(optionalhook=True, tryfirst=True)
def pytest_xdist_make_scheduler(config: pytest.Config, log: Any) -> Any:
    run = config.stash.get(RUN, None)
    if run is None or run.mode not in SCHEDULED_MODES:
        return None  # fallback: xdist's loadgroup
    run.scheduler = SmartparScheduling(config, log, run.planner)
    return run.scheduler


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    config = node.config
    profiler = config.stash.get(PROFILER, None)
    if profiler is not None:
        profiler.configure_node(node)
    run = config.stash.get(RUN, None)
    if run is not None:
        node.workerinput["smartpar_mode"] = run.mode
        node.workerinput["smartpar_token"] = run.token


# --- helpers -------------------------------------------------------------------------


def _smartpar_dir(config: pytest.Config) -> Path:
    path = Path(config.getoption("smartpar_dir"))
    return path if path.is_absolute() else Path(config.invocation_params.dir) / path


def is_xdist_controller(config: pytest.Config) -> bool:
    """Same predicate xdist uses to create its DSession (registration order of
    the two trylast pytest_configure hooks is not guaranteed)."""
    if hasattr(config, "workerinput") or config.getoption("collectonly"):
        return False
    return config.getoption("dist", default="no") != "no" and bool(
        config.getoption("tx", default=None)
    )


def _xdist_workers(config: pytest.Config) -> int:
    if not is_xdist_controller(config):
        return 1
    return len(parse_tx_spec_config(config)) or 1


def _load_data(smartpar_dir: Path) -> tuple[History, Dependencies | None, Flaky | None]:
    return (
        hist_io.load_history(smartpar_dir),
        load_dependencies(smartpar_dir),
        hist_io.load_flaky(smartpar_dir),
    )


def custom_scheduler_problem() -> str | None:
    """Why the smartpar scheduler cannot be used with the installed xdist, if so."""
    try:
        import xdist
        from xdist.workermanage import WorkerController
    except ImportError as exc:  # pragma: no cover
        return f"pytest-xdist not importable: {exc}"
    version = tuple(int(x) for x in xdist.__version__.split(".")[:2] if x.isdigit())
    low, high = SUPPORTED_XDIST
    if not low <= version < high:
        return f"pytest-xdist {xdist.__version__} is outside the tested range"
    missing = [name for name in REQUIRED_NODE_API if not hasattr(WorkerController, name)]
    missing = [m for m in missing if m != "gateway"]  # instance attribute
    if missing:
        return f"pytest-xdist WorkerController lacks {', '.join(missing)}"
    return None


# --- profiling -----------------------------------------------------------------------


def _start_profiling(config: pytest.Config) -> None:
    smartpar_dir = _smartpar_dir(config)
    profiler = ProfilerPlugin(config, smartpar_dir, is_xdist_controller(config))
    config.pluginmanager.register(profiler, "smartpar-profiler")
    config.stash[PROFILER] = profiler
    if hasattr(config, "workerinput"):
        return
    collector = RunCollector(
        mode="profile",
        workers=_xdist_workers(config),
        randomly_reorder_disabled=config.stash[RANDOMLY_REORDER_DISABLED],
    )
    config.pluginmanager.register(collector, "smartpar-run-collector")
    config.pluginmanager.register(
        ProfileRunWriter(collector, smartpar_dir, config.getoption("smartpar_alpha")),
        "smartpar-profile-writer",
    )


# --- --smartpar ------------------------------------------------------------------------


def _start_smartpar(config: pytest.Config) -> None:
    smartpar_dir = _smartpar_dir(config)
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        if workerinput.get("smartpar_mode") == MODE_FALLBACK:
            _start_fallback_worker(config, smartpar_dir, workerinput)
        return
    if not is_xdist_controller(config):
        raise pytest.UsageError("smartpar: --smartpar needs pytest-xdist workers, e.g. -n 4")
    try:
        data = _load_data(smartpar_dir)
    except SchemaError as exc:
        raise pytest.UsageError(f"smartpar: unreadable data file: {exc}") from exc
    hist, deps, _flaky = data
    terminal = config.pluginmanager.get_plugin("terminalreporter")

    def say(line: str) -> None:
        if terminal is not None:
            terminal.write_line(line, yellow=True)

    if deps is None and not hist.tests:
        say("smartpar: no data yet; run `pytest --smartpar-profile` to find dependencies")
    has_timings = any(h.timed_runs for h in hist.tests.values())
    if config.getoption("smartpar_fallback"):
        mode = MODE_FALLBACK
        say("smartpar: fallback requested: --dist loadgroup, tail as a separate pass")
    elif (problem := custom_scheduler_problem()) is not None:
        mode = MODE_FALLBACK
        say(f"smartpar: {problem}; using the loadgroup fallback")
    elif not has_timings:
        mode = MODE_BOOTSTRAP
        say(
            "smartpar: no measured durations yet: every test is estimated equally "
            "(conflicts and confirmed order pairs still apply); this run collects them"
        )
    else:
        mode = MODE_SMARTPAR
    if mode == MODE_FALLBACK:
        config.option.dist = "loadgroup"

    requested = _xdist_workers(config)
    collector = RunCollector(mode, requested, config.stash[RANDOMLY_REORDER_DISABLED])
    config.pluginmanager.register(collector, "smartpar-run-collector")
    run = SmartparRun(config, smartpar_dir, mode, requested, data, collector)
    config.pluginmanager.register(run, "smartpar-run")
    config.stash[RUN] = run


def _start_fallback_worker(
    config: pytest.Config, smartpar_dir: Path, workerinput: dict[str, Any]
) -> None:
    # xdist sets option.loadgroup from the worker's own arguments, which do
    # not contain --dist loadgroup when the controller chose the fallback.
    config.option.dist = "loadgroup"
    config.option.loadgroup = True
    hist, deps, flaky = _load_data(smartpar_dir)
    workers = int(workerinput.get("workercount", 1))

    def analysis_for(collection: list[str]) -> Analysis:
        a = analyze(
            collection,
            hist,
            deps,
            flaky,
            max_workers=available_cpus(),
            workers=workers,
            collection_index={n: i for i, n in enumerate(collection)},
        )
        if workerinput.get("workerid") == "gw0":
            plan = plan_to_dict(
                a,
                [worker_name(i) for i in range(workers)],
                workers,
                MODE_FALLBACK,
                None,
                config.stash[RANDOMLY_REORDER_DISABLED],
                token=workerinput.get("smartpar_token", ""),
            )
            plan["enforced"] = False  # loadgroup decides the actual placement
            hist_io.atomic_write_json(smartpar_dir / PLAN_FILE, plan)
        return a

    config.pluginmanager.register(FallbackItems(config, analysis_for), "smartpar-fallback")


class SmartparRun:
    """Controller side of ``-n m --smartpar``: plan, post-session passes, records."""

    def __init__(
        self,
        config: pytest.Config,
        smartpar_dir: Path,
        mode: str,
        requested_workers: int,
        data: tuple[History, Dependencies | None, Flaky | None],
        collector: RunCollector,
    ) -> None:
        self.config = config
        self.smartpar_dir = smartpar_dir
        self.mode = mode
        self.requested_workers = requested_workers
        self.history, self.deps, self.flaky = data
        self.collector = collector
        self.token = uuid.uuid4().hex[:12]
        self.analysis: Analysis | None = None
        self.scheduler: SmartparScheduling | None = None
        self.fallback_tail: list[str] = []
        self.tail_worker: str | None = None

    # Called by SmartparScheduling.schedule() once all workers have collected.
    def planner(self, collection: list[str], worker_ids: list[str]) -> PlannedRun:
        n = len(worker_ids)
        a = self.analysis = analyze(
            collection,
            self.history,
            self.deps,
            self.flaky,
            max_workers=available_cpus(),
            workers=n,
            collection_index={nodeid: i for i, nodeid in enumerate(collection)},
        )
        if n != self.requested_workers:
            log.warning(
                "smartpar: %d workers instead of %d; re-planned with LPT",
                n,
                self.requested_workers,
            )
        names = [worker_name(i) for i in range(n)]
        assignments = [[t for u in a.schedule.assignments[w] for t in u.tests] for w in names]
        tail_index = min(range(n), key=lambda i: (a.schedule.loads[names[i]], i))
        self.tail_worker = worker_ids[tail_index] if a.graph.tail else None
        plan = plan_to_dict(
            a,
            worker_ids,
            self.requested_workers,
            self.mode,
            self.tail_worker,
            self.config.stash[RANDOMLY_REORDER_DISABLED],
            token=self.token,
        )
        plan["enforced"] = True
        hist_io.atomic_write_json(self.smartpar_dir / PLAN_FILE, plan)
        return PlannedRun(assignments, list(a.graph.tail), tail_index)

    def pytest_testnodedown(self, node: Any, error: object) -> None:
        output = getattr(node, "workeroutput", None) or {}
        if output.get("smartpar_tail"):
            self.fallback_tail = list(output["smartpar_tail"])

    @pytest.hookimpl(trylast=True)  # after junitxml has written results.xml
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        main_wall = self.collector.elapsed()
        junit = Path(self.config.option.xmlpath) if self.config.option.xmlpath else None
        lines: list[str] = []
        tail_mode = "none"
        reruns: list[dict[str, Any]] = []
        tail_failed = False
        workdir = Path(tempfile.mkdtemp(prefix="smartpar-children-"))
        try:
            if self.mode in SCHEDULED_MODES and self.analysis and self.analysis.graph.tail:
                tail_mode = "barrier"
            if self.mode == MODE_FALLBACK and self.fallback_tail:
                tail_mode = "separate-pass"
                tail_failed = self._tail_pass(workdir / "tail", junit, lines)
            reruns = self._rerun_new_failures(workdir, junit, lines)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        failed_first = {
            nodeid for nodeid, acc in self.collector.tests.items() if acc.outcome in FAILED_OUTCOMES
        }
        resolved = {r["nodeid"] for r in reruns if r["suspected_dependency"]}
        record = self.collector.record(
            overhead_s=self._measured_overhead(main_wall, reruns, tail_mode),
            predicted_wall_s=self._predicted_wall(),
            tail_mode=tail_mode,
            reruns=reruns,
            failures_before_rerun=len(failed_first),
            failures_after_rerun=len(failed_first - resolved),
        )
        hist_io.write_run(self.smartpar_dir, record)
        append_report_csv(self.smartpar_dir / REPORT_FILE, report_rows(record, self.history))
        self._adjust_exitstatus(session, failed_first, resolved, tail_failed)
        self.summary_lines = self._summary(record, lines)

    # --- post-session passes -------------------------------------------------------

    def _tail_pass(self, workdir: Path, junit: Path | None, lines: list[str]) -> bool:
        child = run_child_pytest(self.config, self.fallback_tail, workdir)
        rec = child.record()
        failed = child.returncode != 0
        for t in rec.tests if rec else []:
            t.worker = "post:tail"
            self.collector.add(t)
        if junit and junit.exists() and child.junit.exists():
            merge_junit(junit, child.junit, {"smartpar_tail": "separate-pass"}, False)
        lines.append(
            f"isolated tail: {len(self.fallback_tail)} tests in a separate sequential pass "
            f"({'failed' if failed else 'passed'})"
        )
        return failed

    def _rerun_new_failures(
        self, workdir: Path, junit: Path | None, lines: list[str]
    ) -> list[dict[str, Any]]:
        new = set(self.analysis.estimates.new_tests) if self.analysis else self._new_tests()
        failed_new = sorted(
            nodeid
            for nodeid, acc in self.collector.tests.items()
            if nodeid in new and acc.outcome in FAILED_OUTCOMES
        )
        reruns = []
        for i, nodeid in enumerate(failed_new):
            child = run_child_pytest(self.config, [nodeid], workdir / f"rerun{i}")
            rec = child.record()
            outcome = rec.tests[0].outcome if rec and rec.tests else "error"
            duration = rec.tests[0].duration if rec and rec.tests else 0.0
            suspected = outcome not in FAILED_OUTCOMES
            reruns.append(
                {
                    "nodeid": nodeid,
                    "first_outcome": self.collector.tests[nodeid].outcome,
                    "rerun_outcome": outcome,
                    "duration": duration,
                    "suspected_dependency": suspected,
                }
            )
            if junit and junit.exists() and child.junit.exists():
                merge_junit(
                    junit,
                    child.junit,
                    {
                        "smartpar_rerun": outcome,
                        "smartpar_suspected_dependency": str(suspected).lower(),
                    },
                    True,
                )
            verdict = "passed alone: suspected dependency" if suspected else "failed again"
            lines.append(f"re-run of new test {nodeid}: {verdict}")
        return reruns

    def _new_tests(self) -> set[str]:
        return {nodeid for nodeid in self.collector.tests if nodeid not in self.history.tests}

    # --- numbers -------------------------------------------------------------------

    def _predicted_wall(self) -> float | None:
        if self.mode in SCHEDULED_MODES and self.analysis is not None:
            return self.analysis.schedule.predicted_makespan_s
        if self.mode == MODE_FALLBACK:
            try:
                plan = json.loads((self.smartpar_dir / PLAN_FILE).read_text("utf-8"))
            except (OSError, ValueError):
                return None
            if plan.get("token") == self.token:
                return float(plan["predicted_makespan_s"])
        return None

    def _measured_overhead(
        self, main_wall: float, reruns: list[dict[str, Any]], tail_mode: str
    ) -> float | None:
        """Wall time not explained by the busiest worker and the tail.

        Only for clean runs of the smartpar scheduler (no crash, no re-run),
        where the model T(m) = max load + tail + overhead(m) applies.
        """
        if self.mode not in SCHEDULED_MODES or reruns or self.analysis is None:
            return None
        if self.scheduler is None or self.scheduler.events.crashed_workers:
            return None
        tail = set(self.analysis.graph.tail)
        busy: dict[str, float] = {}
        tail_busy = 0.0
        for nodeid, acc in self.collector.tests.items():
            if nodeid in tail:
                tail_busy += acc.duration
            else:
                busy[acc.worker] = busy.get(acc.worker, 0.0) + acc.duration
        return max(0.0, main_wall - max(busy.values(), default=0.0) - tail_busy)

    def _adjust_exitstatus(
        self,
        session: pytest.Session,
        failed_first: set[str],
        resolved: set[str],
        tail_failed: bool,
    ) -> None:
        """A failure that passed when re-run alone counts as passed (exit code 0),
        unless --smartpar-strict; a failing separate tail pass fails the session."""
        strict = self.config.getoption("smartpar_strict")
        if (
            not strict
            and session.exitstatus == pytest.ExitCode.TESTS_FAILED
            and failed_first
            and failed_first <= resolved
        ):
            session.exitstatus = pytest.ExitCode.OK
        if (tail_failed or (strict and resolved)) and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _summary(self, record: Any, lines: list[str]) -> list[str]:
        out = []
        if record.predicted_wall_s is not None:
            out.append(f"wall {record.wall_time_s:.2f}s, predicted {record.predicted_wall_s:.2f}s")
        out += lines
        out.append(
            f"failures: {record.failures_before_rerun} before re-runs, "
            f"{record.failures_after_rerun} after; tail: {record.tail_mode}"
        )
        if self.config.getoption("smartpar_strict") and any(
            r["suspected_dependency"] for r in record.reruns
        ):
            out.append("--smartpar-strict: suspected dependencies fail the session")
        out.append(f"report: {self.smartpar_dir / REPORT_FILE}")
        return out

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        lines = getattr(self, "summary_lines", None)
        if not lines:
            return
        terminalreporter.write_sep("-", f"smartpar ({self.mode})")
        for line in lines:
            terminalreporter.write_line(line)


# --- --smartpar-detect ------------------------------------------------------------------


class DetectSession:
    """Collects the tests, then runs the detector instead of the tests."""

    def __init__(self, config: pytest.Config, smartpar_dir: Path) -> None:
        self.config = config
        self.smartpar_dir = smartpar_dir
        self.flaky: Flaky | None = None
        self.meta: dict[str, Any] = {}

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session: pytest.Session) -> bool:
        if session.testsfailed and not self.config.option.continue_on_collection_errors:
            raise session.Interrupted("smartpar: collection errors, detection not started")
        nodeids = [item.nodeid for item in session.items]
        self.flaky, self.meta = run_detection(self.config, self.smartpar_dir, nodeids)
        return True  # the tests themselves ran in child processes

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.flaky is None:
            return
        f, m = self.flaky, self.meta
        terminalreporter.write_sep("-", "smartpar detect")
        terminalreporter.write_line(
            f"{f.runs} random runs (seeds {f.seeds[0]}..{f.seeds[-1]}), "
            f"{m['child_runs']} child runs, {m['sandbox']}, {m['elapsed_s']:.1f}s"
        )
        terminalreporter.write_line(
            f"victims: {len(f.od_victims)}, brittle: {len(f.od_brittle)}, "
            f"NOD: {len(f.nod)}, unresolved: {len(f.unresolved)}"
        )
        for v in f.od_victims:
            terminalreporter.write_line(f"  {v['victim']} <- {', '.join(v['polluters'])}")
        for b in f.od_brittle:
            terminalreporter.write_line(f"  {b['test']} needs {', '.join(b['state_setters'])}")
        terminalreporter.write_line(f"written: {self.smartpar_dir / hist_io.FLAKY_FILE}")


def run_detection(
    config: pytest.Config, smartpar_dir: Path, nodeids: list[str]
) -> tuple[Flaky, dict[str, Any]]:
    """Detector over ``nodeids``; writes flaky.json and updates dependencies.json."""
    t0 = time.monotonic()
    runs = config.getoption("smartpar_runs")
    base_seed, seeds = choose_seeds(config.getoption("smartpar_seed"), runs)
    budget = config.getoption("smartpar_detect_budget")
    root = Path(config.rootpath)
    sandbox = Sandbox(root, extra_ignore=[smartpar_dir, Path(sys.prefix)])
    runner = ProcessRunner(
        root=root,
        nodeids=nodeids,
        sandbox=sandbox,
        inipath=Path(config.inipath) if config.inipath else None,
        workers=config.getoption("smartpar_detect_workers") or available_cpus(),
        deadline=None if budget is None else t0 + budget,
        ordered_seed_start=seeds[-1] + 1,
    )
    limits = {
        "max_pair_checks": _ini_int(config, "smartpar_max_pair_checks"),
        "max_orders_per_victim": _ini_int(config, "smartpar_max_orders_per_victim"),
    }
    deps = load_dependencies(smartpar_dir)
    detector = Detector(
        runner,
        deps.state_changes if deps else {},
        max_pair_checks=limits["max_pair_checks"],
        max_orders=limits["max_orders_per_victim"],
    )
    try:
        try:
            flaky = detector.detect(seeds)
        except BudgetExceeded:  # during the random runs themselves
            flaky = Flaky(runs=0, seeds=seeds, unresolved=[{"test": "*", "reason": "budget"}])
    finally:
        sandbox.cleanup()
    meta = {
        "base_seed": base_seed,
        "isolation_runs": detector.k,
        "limits": limits,
        "truncated_victims": len(detector.truncated),
        "truncated": detector.truncated_report(),
        "ordered_seed_start": runner.ordered_seed_start,
        "ordered_runs": runner.ordered_seeds_used,
        "confirm_parallel_runs": max(1, runner.workers // 2),
        "confirm_runs": CONFIRM_RUNS,
        "budget_s": budget,
        "elapsed_s": time.monotonic() - t0,
        "child_runs": runner.child_runs,
        "parallel_runs": runner.workers,
        "sandbox": "sequential in the project directory"
        if sandbox.in_place
        else "a fresh project copy per run",
    }
    flaky.detector = meta
    hist_io.atomic_write_json(smartpar_dir / hist_io.FLAKY_FILE, flaky.to_dict())
    _update_dependencies(smartpar_dir, flaky, nodeids)
    return flaky, meta


def _ini_int(config: pytest.Config, name: str) -> int:
    raw = config.getini(name)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise pytest.UsageError(f"smartpar: ini option {name} must be an integer") from exc
    if value < 0:
        raise pytest.UsageError(f"smartpar: ini option {name} must be >= 0")
    return value


def _update_dependencies(smartpar_dir: Path, flaky: Flaky, nodeids: list[str]) -> None:
    """Mirror the confirmed pairs into dependencies.json and rebuild groups."""
    deps = load_dependencies(smartpar_dir) or Dependencies()
    hist = hist_io.load_history(smartpar_dir)
    tests = set(nodeids) | set(deps.file_access) | set(deps.state_changes)
    durations = estimate_durations(tests, hist).durations
    graph = build_graph(
        tests,
        deps.file_access,
        flaky.order_edges(),
        durations,
        isolate=flaky.nod_tests(),
        collection_index=deps.collection_index or {n: i for i, n in enumerate(nodeids)},
    )
    deps.conflicts, deps.order_edges, deps.groups = graph.conflicts, graph.order_edges, graph.groups
    hist_io.atomic_write_json(smartpar_dir / DEPENDENCIES_FILE, deps.to_dict())
