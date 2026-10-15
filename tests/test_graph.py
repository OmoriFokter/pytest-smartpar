"""Phase 4: Bernstein conflicts, union-find groups, order inside groups, isolated tail."""

from __future__ import annotations

import random

import pytest

from smartpar import graph
from smartpar.models import Conflict, FileAccess, OrderEdge

V_P = "victim-before-polluter"
S_B = "setter-before-brittle"


def fa(r: list[str] | None = None, w: list[str] | None = None) -> FileAccess:
    return FileAccess(R=r or [], W=w or [])


# --- conflicts ---------------------------------------------------------------------


def test_bernstein_conflict_kinds() -> None:
    access = {
        "a": fa(w=["f1"]),
        "b": fa(r=["f1"]),
        "c": fa(r=["f2"]),
        "d": fa(w=["f2"]),
        "e": fa(w=["f3"]),
        "f": fa(w=["f3"]),
    }
    assert graph.find_conflicts(access, access) == [
        Conflict("a", "b", "f1", "W-R"),
        Conflict("c", "d", "f2", "R-W"),
        Conflict("e", "f", "f3", "W-W"),
    ]


def test_no_conflict_for_shared_reads_or_self() -> None:
    access = {"a": fa(r=["f"]), "b": fa(r=["f"]), "c": fa(r=["g"], w=["g"])}
    assert graph.find_conflicts(access, access) == []


def test_reader_writer_both_ways() -> None:
    # a reads and writes f, b only reads: one W-R entry from a's side.
    access = {"a": fa(r=["f"], w=["f"]), "b": fa(r=["f"])}
    assert graph.find_conflicts(access, access) == [Conflict("a", "b", "f", "W-R")]


def test_conflict_per_resource_and_universe_filter() -> None:
    access = {"a": fa(w=["f", "g"]), "b": fa(r=["f", "g"]), "z": fa(w=["f"])}
    assert graph.find_conflicts(access, ["a", "b"]) == [
        Conflict("a", "b", "f", "W-R"),
        Conflict("a", "b", "g", "W-R"),
    ]


def test_inverted_index_scales_with_shared_resources_only() -> None:
    # 20k tests with private files: no pair is ever compared.
    access = {f"t{i}": fa(r=[f"in{i}"], w=[f"out{i}"]) for i in range(20_000)}
    access["x"] = fa(w=["in7"])
    assert graph.find_conflicts(access, access) == [Conflict("t7", "x", "in7", "R-W")]


# --- groups ------------------------------------------------------------------------


def test_union_find_groups_transitively_with_reasons() -> None:
    access = {"a": fa(w=["f"]), "b": fa(r=["f"])}
    edges = [OrderEdge("b", "c", V_P)]
    result = graph.build(["a", "b", "c", "d"], access, edges, {})
    (group,) = result.groups
    assert group.id == "g0"
    assert group.tests == ["a", "b", "c"]
    assert group.reasons == ["file", "od"]
    assert result.singles == ["d"]
    assert result.tail == []


def test_order_inside_group_topological_then_duration_then_nodeid() -> None:
    access = {t: fa(w=["shared"]) for t in "abcde"}
    edges = [OrderEdge("e", "a", V_P)]  # e must precede a
    durations = {"a": 5.0, "b": 1.0, "c": 3.0, "d": 3.0, "e": 0.1}
    (group,) = graph.build("abcde", access, edges, durations).groups
    # Ready first: c, d (3.0, by nodeid), b (1.0), e (0.1); a only after e.
    assert group.order == ["c", "d", "b", "e", "a"]
    assert group.est_duration == pytest.approx(12.1)


def test_cycle_and_its_dependents_go_to_tail() -> None:
    edges = [
        OrderEdge("a", "b", V_P),
        OrderEdge("b", "a", V_P),  # cycle a <-> b
        OrderEdge("b", "c", S_B),  # c must follow b: tail too
        OrderEdge("d", "a", V_P),  # d precedes a: fine before the tail
    ]
    result = graph.build(["a", "b", "c", "d", "e"], {}, edges, {})
    assert result.cyclic == ["a", "b"]
    assert set(result.tail) == {"a", "b", "c"}
    assert result.tail.index("b") < result.tail.index("c")
    assert result.groups == []
    assert result.singles == ["d", "e"]


def test_isolated_test_pulls_its_dependents_into_tail() -> None:
    edges = [OrderEdge("setter", "brittle", S_B), OrderEdge("victim", "setter", V_P)]
    result = graph.build(["setter", "brittle", "victim"], {}, edges, {}, isolate=["setter"])
    assert result.tail == ["setter", "brittle"]
    assert result.singles == ["victim"]
    assert result.cyclic == []


def test_edges_to_unknown_tests_and_self_loops_dropped() -> None:
    edges = [OrderEdge("a", "gone", V_P), OrderEdge("a", "a", V_P)]
    result = graph.build(["a", "b"], {"gone": fa(w=["f"]), "a": fa(r=["f"])}, edges, {})
    assert result.order_edges == [] and result.conflicts == []
    assert result.singles == ["a", "b"]


def test_build_is_deterministic_under_input_permutation() -> None:
    tests = [f"t{i:02d}" for i in range(30)]
    access = {
        t: fa(w=[f"f{i % 7}"] if i % 3 == 0 else [], r=[f"f{i % 5}"]) for i, t in enumerate(tests)
    }
    edges = [OrderEdge(tests[i], tests[i + 4], V_P) for i in range(0, 24, 5)]
    durations = {t: float(i % 4) for i, t in enumerate(tests)}
    expected = graph.build(tests, access, edges, durations, isolate=["t29"])
    rng = random.Random(7)
    for _ in range(5):
        shuffled_tests = rng.sample(tests, len(tests))
        shuffled_access = dict(rng.sample(sorted(access.items()), len(access)))
        shuffled_edges = rng.sample(edges, len(edges))
        assert (
            graph.build(shuffled_tests, shuffled_access, shuffled_edges, durations, ["t29"])
            == expected
        )


def test_topo_order_breaks_cycles_deterministically() -> None:
    edges = [OrderEdge("b", "a", V_P), OrderEdge("a", "b", V_P), OrderEdge("a", "c", V_P)]
    assert graph.topo_order(["a", "b", "c"], edges, {}) == ["a", "b", "c"]
