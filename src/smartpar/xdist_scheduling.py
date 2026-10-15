"""Method step 4 (execution): pytest-xdist scheduler that executes the plan.

``SmartparScheduling`` implements xdist's (internal) scheduler interface, the
same one ``LoadScheduling`` implements: ``nodes``, ``collection_is_completed``,
``tests_finished``, ``has_pending``, ``add_node``, ``add_node_collection``,
``mark_test_complete``, ``mark_test_pending``, ``remove_pending_tests_from_node``,
``remove_node``, ``schedule``. Tested with the pytest-xdist versions pinned in
pyproject.toml.

Static distribution: once all workers have collected, the plan is computed
for the actual number of workers (so a worker count different from the
requested one is re-planned with the same LPT) and every worker receives its
whole list at once, in plan order; an xdist worker runs tests in the order it
receives them.

Barrier for the isolated tail. An xdist worker always holds back the last test
it received until the next test or "shutdown" arrives (it needs ``nextitem``
for fixture teardown), so "every worker has 0 pending tests" never happens
while a worker still waits for work. Therefore:

* workers that will not run the tail get "shutdown" right after their list:
  they run everything, including the held test, and finish;
* the tail worker (the one with the smallest planned load) gets the tail when
  it has only its held test left AND every other worker has finished
  (``remove_node`` was called for it), followed by "shutdown". Its held test
  runs first, then the tail sequentially - after all other tests.

If a worker crashes, its remaining tests (in order) go to the tail worker
before the tail; if the tail worker itself is gone, to the next worker xdist
starts as a replacement.

Fallback (``--dist loadgroup``): groups become ``xdist_group`` marks, items
are reordered by LPT priority and the tail is deselected on the workers and
run afterwards by a separate sequential pass (no barrier exists there).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from smartpar.models import RunRecord

log = logging.getLogger("smartpar.xdist")

try:  # internal xdist API, present in all supported versions
    from xdist.workermanage import parse_tx_spec_config
except ImportError:  # pragma: no cover

    def parse_tx_spec_config(config: pytest.Config) -> list[str]:
        return list(config.getoption("tx") or [])


FALLBACK_GROUP_PREFIX = "smartpar_"
_GROUP_SUFFIX = re.compile(r"@" + FALLBACK_GROUP_PREFIX + r"g\d+$")

# Members of xdist's WorkerController this module relies on.
REQUIRED_NODE_API = ("send_runtest_some", "shutdown", "shutting_down", "gateway")


def strip_group_suffix(nodeid: str) -> str:
    """Remove the ``@smartpar_g<N>`` suffix xdist adds to grouped items."""
    return _GROUP_SUFFIX.sub("", nodeid)


def worker_sort_key(node: Any) -> tuple[int, str]:
    """gw2 before gw10."""
    ident = str(node.gateway.id)
    digits = re.sub(r"\D", "", ident)
    return (int(digits) if digits else 0, ident)


@dataclass
class PlannedRun:
    """What the scheduler needs from a plan, by plan worker index."""

    assignments: list[list[str]]  # nodeids per worker, in execution order
    tail: list[str]
    tail_worker: int


Planner = Callable[[list[str], list[str]], PlannedRun]
"""(collection in order, actual worker ids) -> plan for exactly that many workers."""


@dataclass
class SchedulerEvents:
    """Observable facts for the run record and tests."""

    tail_released_at_pending: int | None = None
    crashed_workers: list[str] = field(default_factory=list)
    replacement_workers: list[str] = field(default_factory=list)


class SmartparScheduling:
    def __init__(
        self, config: pytest.Config, log: Any, planner: Planner, numnodes: int | None = None
    ) -> None:
        self.config = config
        self.numnodes = numnodes if numnodes is not None else len(parse_tx_spec_config(config))
        self.planner = planner
        self.node2collection: dict[Any, list[str]] = {}
        self.node2pending: dict[Any, list[int]] = {}
        self.collection: list[str] | None = None
        self.orphans: list[int] = []  # tests of crashed workers, in order
        self.tail: list[int] = []
        self.tail_node: Any | None = None
        self.tail_sent = False
        self.scheduled = False
        self._initial_nodes: set[Any] = set()
        self.events = SchedulerEvents()

    # --- interface properties ----------------------------------------------------

    @property
    def nodes(self) -> list[Any]:
        return list(self.node2pending)

    @property
    def collection_is_completed(self) -> bool:
        return len(self.node2collection) >= self.numnodes

    @property
    def tests_finished(self) -> bool:
        if not (self.collection_is_completed and self.scheduled):
            return False
        if self.orphans or (self.tail and not self.tail_sent):
            return False
        # The last test of every worker is held back until shutdown.
        return all(len(pending) <= 1 for pending in self.node2pending.values())

    @property
    def has_pending(self) -> bool:
        if self.orphans or (self.tail and not self.tail_sent):
            return True
        return any(self.node2pending.values())

    # --- interface methods --------------------------------------------------------

    def add_node(self, node: Any) -> None:
        assert node not in self.node2pending
        self.node2pending[node] = []

    def add_node_collection(self, node: Any, collection: Sequence[str]) -> None:
        assert node in self.node2pending
        if self.collection is not None and list(collection) != self.collection:
            log.error("smartpar: worker %s collected different tests; not used", node.gateway.id)
            return
        self.node2collection[node] = list(collection)

    def mark_test_complete(self, node: Any, item_index: int, duration: float = 0) -> None:
        self.node2pending[node].remove(item_index)
        self._release_tail_if_ready()

    def mark_test_pending(self, item: str) -> None:
        assert self.collection is not None
        self.orphans.insert(0, self.collection.index(item))
        self._dispatch_orphans()

    def remove_pending_tests_from_node(self, node: Any, indices: Sequence[int]) -> None:
        raise NotImplementedError("smartpar never steals tests")

    def remove_node(self, node: Any) -> str | None:
        # node2collection keeps the entry (as in LoadScheduling):
        # collection_is_completed must stay true after workers finish.
        pending = self.node2pending.pop(node)
        if node is self.tail_node:
            self.tail_node = None
        crashitem = None
        if pending:
            assert self.collection is not None
            crashitem = self.collection[pending[0]]
            self.events.crashed_workers.append(str(node.gateway.id))
            self.orphans.extend(pending[1:])
            log.warning(
                "smartpar: worker %s went down with %d pending tests; "
                "they are moved to the tail worker",
                node.gateway.id,
                len(pending),
            )
            self._dispatch_orphans()
        self._release_tail_if_ready()
        return crashitem

    def schedule(self) -> None:
        assert self.collection_is_completed
        if self.scheduled:
            # A replacement worker joined after a crash.
            for node in self.nodes:
                if node not in self._initial_nodes:
                    self._adopt_replacement(node)
            return

        collections = list(self.node2collection.values())
        if any(c != collections[0] for c in collections[1:]):
            raise RuntimeError("smartpar: workers collected different tests")
        self.collection = collections[0]
        index = {nodeid: i for i, nodeid in enumerate(self.collection)}
        nodes = sorted(self.nodes, key=worker_sort_key)
        self._initial_nodes = set(nodes)
        plan = self.planner(list(self.collection), [str(n.gateway.id) for n in nodes])
        assert len(plan.assignments) == len(nodes)

        self.tail = [index[t] for t in plan.tail]
        self.tail_node = nodes[plan.tail_worker]
        for i, node in enumerate(nodes):
            indices = [index[t] for t in plan.assignments[i]]
            if indices:
                self.node2pending[node].extend(indices)
                node.send_runtest_some(indices)
            if node is not self.tail_node or not self.tail:
                node.shutdown()
        self.scheduled = True
        self._release_tail_if_ready()

    # --- internals -----------------------------------------------------------------

    def _tail_target(self) -> Any | None:
        node = self.tail_node
        if node is None or node not in self.node2pending or node.shutting_down:
            return None
        return node

    def _dispatch_orphans(self) -> None:
        target = self._tail_target()
        if target is None or not self.orphans:
            return
        self.node2pending[target].extend(self.orphans)
        target.send_runtest_some(list(self.orphans))
        self.orphans = []

    def _adopt_replacement(self, node: Any) -> None:
        self._initial_nodes.add(node)
        self.events.replacement_workers.append(str(node.gateway.id))
        needs_home = self.orphans or (self.tail and not self.tail_sent)
        if node in self.node2collection and needs_home and self._tail_target() is None:
            self.tail_node = node
            self._dispatch_orphans()
            if self.tail and not self.tail_sent:
                self._release_tail_if_ready()  # sends "shutdown" with the tail
            else:
                node.shutdown()
        else:
            node.shutdown()

    def _release_tail_if_ready(self) -> None:
        if not self.scheduled or self.tail_sent or not self.tail:
            return
        target = self._tail_target()
        if target is None or self.orphans:
            return
        others_running = any(n is not target for n in self.node2pending)
        held = len(self.node2pending[target])
        if others_running or held > 1:
            return
        self.events.tail_released_at_pending = held
        self.node2pending[target].extend(self.tail)
        target.send_runtest_some(list(self.tail))
        target.shutdown()
        self.tail_sent = True


# --- fallback: --dist loadgroup -------------------------------------------------------


class FallbackItems:
    """Worker-side half of the fallback (``--dist loadgroup``).

    Every worker computes the same analysis from the same data files, so
    marks, order and the deselected tail are identical on all workers (xdist
    requires identical collections). ``analysis_for`` maps the collected
    nodeids (in collection order) to a ``scheduler.Analysis``.
    """

    def __init__(self, config: pytest.Config, analysis_for: Callable[[list[str]], Any]) -> None:
        self.config = config
        self.analysis_for = analysis_for
        self.analysis: Any | None = None

    @pytest.hookimpl(tryfirst=True)  # before xdist turns the marks into @suffixes
    def pytest_collection_modifyitems(
        self, session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        a = self.analysis = self.analysis_for([item.nodeid for item in items])
        by_id = {item.nodeid: item for item in items}
        ordered: list[pytest.Item] = []
        # LoadScope-style scheduling hands out scopes in item order to the
        # first free worker: ordering units by LPT priority makes that greedy.
        for unit in sorted(a.units, key=lambda u: u.sort_key):
            for nodeid in unit.tests:
                item = by_id.pop(nodeid)
                if len(unit.tests) > 1:
                    item.add_marker(pytest.mark.xdist_group(FALLBACK_GROUP_PREFIX + unit.id))
                ordered.append(item)
        tail = [by_id.pop(nodeid) for nodeid in a.graph.tail]
        ordered.extend(by_id.values())  # nothing should be left; keep it safe
        if tail:
            config.hook.pytest_deselected(items=tail)
        items[:] = ordered
        workeroutput = getattr(config, "workeroutput", None)
        if workeroutput is not None:
            workeroutput["smartpar_tail"] = list(a.graph.tail)


# --- child pytest processes (re-runs, separate tail pass) ----------------------------


@dataclass
class ChildRun:
    nodeids: list[str]
    command: list[str]
    returncode: int
    wall_s: float
    junit: Path
    report: Path

    def record(self) -> RunRecord | None:
        try:
            return RunRecord.from_dict(json.loads(self.report.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None


# pytest reads arguments from "@file" since 8.2.
_ARGFILE_MIN_PYTEST = (8, 2)
_ARGFILE_THRESHOLD = 100


def child_command(config: pytest.Config, nodeids: list[str], workdir: Path) -> list[str]:
    """pytest in a fresh process: explicit order, no xdist, no shuffling."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:xdist",
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
        "-q",
        "--rootdir",
        str(config.rootpath),
        f"--junitxml={workdir / 'junit.xml'}",
        "-o",
        "junit_duration_report=total",
        f"--smartpar-child-report={workdir / 'report.json'}",
        f"--basetemp={workdir / 'basetemp'}",
    ]
    if config.inipath is not None:
        cmd += ["-c", str(config.inipath)]
    version = tuple(int(x) for x in pytest.__version__.split(".")[:2] if x.isdigit())
    if len(nodeids) > _ARGFILE_THRESHOLD and version >= _ARGFILE_MIN_PYTEST:
        argfile = workdir / "args.txt"
        argfile.write_text("\n".join(nodeids) + "\n", encoding="utf-8")
        cmd.append(f"@{argfile}")
    else:
        cmd += nodeids
    return cmd


def run_child_pytest(config: pytest.Config, nodeids: list[str], workdir: Path) -> ChildRun:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = child_command(config, nodeids, workdir)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("PYTEST_XDIST_") and k != "PYTEST_CURRENT_TEST"
    }
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=config.invocation_params.dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    (workdir / "output.txt").write_text(proc.stdout, encoding="utf-8")
    return ChildRun(
        nodeids, cmd, proc.returncode, wall, workdir / "junit.xml", workdir / "report.json"
    )
