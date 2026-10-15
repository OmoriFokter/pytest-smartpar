"""Phase 5: SmartparScheduling against a simulator with xdist's worker semantics.

The simulated worker behaves like ``xdist.remote.WorkerInteractor``: it can
only finish a test once the next test or "shutdown" has arrived (it holds the
last received test back for ``nextitem``), runs tests in the order received,
and reports "workerfinished" (``remove_node``) after shutdown.
"""

from __future__ import annotations

import pytest

from smartpar.xdist_scheduling import PlannedRun, SmartparScheduling, strip_group_suffix


class FakeGateway:
    def __init__(self, ident: str) -> None:
        self.id = ident


class FakeNode:
    def __init__(self, ident: str) -> None:
        self.gateway = FakeGateway(ident)
        self.queue: list[int] = []
        self.shutdown_received = False
        self.finished = False

    def send_runtest_some(self, indices: list[int]) -> None:
        assert not self.shutdown_received, "tests sent after shutdown would never run"
        self.queue.extend(indices)

    def shutdown(self) -> None:
        self.shutdown_received = True

    @property
    def shutting_down(self) -> bool:
        return self.shutdown_received or self.finished

    def __repr__(self) -> str:
        return self.gateway.id


def simulate(
    sched: SmartparScheduling,
    nodes: list[FakeNode],
    crash: tuple[str, int] | None = None,
    replacement: FakeNode | None = None,
    collection: list[str] | None = None,
    max_steps: int = 10_000,
) -> list[tuple[str, int]]:
    """Run until every worker finished; return the executed (worker, index) sequence.

    ``crash=(worker, n)``: that worker dies after finishing n tests.
    """
    log: list[tuple[str, int]] = []
    done_count: dict[str, int] = {}
    alive = list(nodes)
    for _ in range(max_steps):
        progressed = False
        for node in list(alive):
            ident = node.gateway.id
            if crash and ident == crash[0] and done_count.get(ident, 0) == crash[1]:
                alive.remove(node)
                node.finished = True
                crashitem = sched.remove_node(node)
                assert crashitem is not None
                if replacement is not None:
                    sched.add_node(replacement)
                    sched.add_node_collection(replacement, collection or [])
                    sched.schedule()
                    alive.append(replacement)
                crash = None
                progressed = True
                break
            if node.queue and (len(node.queue) >= 2 or node.shutdown_received):
                index = node.queue.pop(0)
                log.append((ident, index))
                done_count[ident] = done_count.get(ident, 0) + 1
                sched.mark_test_complete(node, index)
                progressed = True
            elif not node.queue and node.shutdown_received:
                alive.remove(node)
                node.finished = True
                assert sched.remove_node(node) is None
                progressed = True
        if not alive:
            return log
        if not progressed:
            raise AssertionError(f"hang: {[(n, n.queue, n.shutdown_received) for n in alive]}")
    raise AssertionError("did not finish")


def make(
    assignments: list[list[str]], tail: list[str], tail_worker: int, n_nodes: int | None = None
) -> tuple[SmartparScheduling, list[FakeNode], list[str]]:
    collection = sorted({t for a in assignments for t in a} | set(tail))
    plans: list[tuple[list[str], list[str]]] = []

    def planner(coll: list[str], worker_ids: list[str]) -> PlannedRun:
        plans.append((coll, worker_ids))
        return PlannedRun(assignments, tail, tail_worker)

    n = n_nodes or len(assignments)
    sched = SmartparScheduling(None, None, planner, numnodes=n)  # type: ignore[arg-type]
    nodes = [FakeNode(f"gw{i}") for i in range(n)]
    for node in nodes:
        sched.add_node(node)
    for node in nodes:
        assert not sched.collection_is_completed
        sched.add_node_collection(node, collection)
    assert sched.collection_is_completed
    sched.schedule()
    sched.plans = plans  # type: ignore[attr-defined]
    return sched, nodes, collection


def names(log: list[tuple[str, int]], collection: list[str]) -> list[tuple[str, str]]:
    return [(w, collection[i]) for w, i in log]


@pytest.mark.parametrize(
    ("assignments", "tail_worker"),
    [
        ([["a1", "a2", "a3"], ["b1"], ["c1", "c2"]], 1),  # tail worker holds one test
        ([["a1", "a2"], [], ["c1"]], 1),  # tail worker has no main tests
        ([["a1", "a2", "a3", "a4"], ["b1", "b2", "b3"]], 0),  # tail worker is the busiest
    ],
)
def test_tail_runs_after_every_other_test_without_hang(
    assignments: list[list[str]], tail_worker: int
) -> None:
    sched, nodes, collection = make(assignments, ["t1", "t2"], tail_worker)
    log = names(simulate(sched, nodes), collection)
    executed = [t for _w, t in log]
    assert sorted(executed) == collection  # everything ran exactly once
    assert executed[-2:] == ["t1", "t2"]  # tail last, in order
    assert {w for w, t in log if t.startswith("t")} == {f"gw{tail_worker}"}
    assert sched.tests_finished
    # Released only when the tail worker held at most its last main test.
    assert sched.events.tail_released_at_pending in (0, 1)


def test_plan_order_per_worker_is_kept() -> None:
    sched, nodes, collection = make([["g3", "g1", "g2"], ["x"]], [], 0)
    log = names(simulate(sched, nodes), collection)
    assert [t for w, t in log if w == "gw0"] == ["g3", "g1", "g2"]


def test_without_tail_everyone_is_shut_down_upfront() -> None:
    sched, nodes, _collection = make([["a"], ["b", "c"]], [], 0)
    assert all(n.shutdown_received for n in nodes)
    simulate(sched, nodes)
    assert sched.tests_finished and not sched.has_pending


def test_tests_finished_false_until_tail_sent() -> None:
    sched, nodes, _collection = make([["a1"], ["b1", "b2"]], ["t"], 0)
    assert not nodes[0].shutdown_received  # the tail worker waits
    assert nodes[1].shutdown_received
    assert not sched.tests_finished  # otherwise xdist would shut the tail worker down
    assert sched.has_pending


def test_planner_gets_actual_workers() -> None:
    sched, _nodes, collection = make([["a"], ["b"], ["c"]], [], 0)
    coll, worker_ids = sched.plans[0]  # type: ignore[attr-defined]
    assert worker_ids == ["gw0", "gw1", "gw2"]
    assert coll == collection


def test_crash_of_other_worker_moves_its_tests_to_tail_worker() -> None:
    sched, nodes, collection = make([["a1", "a2", "a3", "a4"], ["b1"]], ["t"], 1)
    log = names(simulate(sched, nodes, crash=("gw0", 1)), collection)
    executed = [t for _w, t in log]
    # a1 done, a2 was running when gw0 died (crash item), a3/a4 moved to gw1.
    assert "a2" not in executed
    assert [t for w, t in log if w == "gw1"] == ["b1", "a3", "a4", "t"]
    assert sched.events.crashed_workers == ["gw0"]


def test_crash_of_tail_worker_uses_replacement_worker() -> None:
    sched, nodes, collection = make([["a1"], ["b1", "b2", "b3"]], ["t"], 1)
    spare = FakeNode("gw2")
    log = names(
        simulate(sched, nodes, crash=("gw1", 1), replacement=spare, collection=collection),
        collection,
    )
    assert [t for w, t in log if w == "gw2"] == ["b3", "t"]
    assert sched.events.replacement_workers == ["gw2"]
    assert sched.tests_finished


def test_replacement_without_work_is_shut_down() -> None:
    sched, nodes, collection = make([["a1", "a2"], ["b1"]], [], 0)
    spare = FakeNode("gw2")
    simulate(sched, nodes, crash=("gw0", 1), replacement=spare, collection=collection)
    assert spare.shutdown_received


def test_mark_test_pending_reschedules_on_tail_worker() -> None:
    sched, nodes, collection = make([["a1", "a2"], ["b1"]], ["t"], 1)
    sched.mark_test_pending("a1")
    assert nodes[1].queue[-1] == collection.index("a1")


def test_different_collections_are_rejected() -> None:
    def planner(coll: list[str], ids: list[str]) -> PlannedRun:
        raise AssertionError("not reached")

    sched = SmartparScheduling(None, None, planner, numnodes=2)  # type: ignore[arg-type]
    a, b = FakeNode("gw0"), FakeNode("gw1")
    sched.add_node(a)
    sched.add_node(b)
    sched.add_node_collection(a, ["x", "y"])
    sched.add_node_collection(b, ["y", "x"])
    with pytest.raises(RuntimeError, match="different tests"):
        sched.schedule()


def test_steal_is_not_supported() -> None:
    sched, nodes, _ = make([["a"]], [], 0)
    with pytest.raises(NotImplementedError):
        sched.remove_pending_tests_from_node(nodes[0], [0])


def test_strip_group_suffix() -> None:
    assert strip_group_suffix("t.py::x@smartpar_g12") == "t.py::x"
    assert strip_group_suffix("t.py::x[a@b]") == "t.py::x[a@b]"
    assert strip_group_suffix("t.py::x@other") == "t.py::x@other"
