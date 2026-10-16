"""Phase 2: schemas of history.json and run files."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from smartpar import __version__
from smartpar.models import (
    DEFAULT_OVERHEAD_A,
    DEFAULT_OVERHEAD_B,
    SCHEMA_VERSION,
    History,
    Overhead,
    RunRecord,
    SchemaError,
    TestHistory,
    TestRecord,
)


def make_run(**overrides: object) -> RunRecord:
    fields: dict = {
        "run_id": "run_20260101T000000000000Z_1",
        "mode": "smartpar",
        "workers": 2,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:05+00:00",
        "wall_time_s": 5.0,
        "tests": [TestRecord("t.py::a", "passed", 0.5, "gw0", 10.0, 10.5)],
        "randomly_reorder_disabled": True,
        "overhead_s": 0.4,
    }
    fields.update(overrides)
    return RunRecord(**fields)


def roundtrip(data: dict) -> dict:
    return json.loads(json.dumps(data))


def test_envelope_present() -> None:
    data = History().to_dict()
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["tool_version"] == __version__
    assert datetime.fromisoformat(data["generated_at"]).tzinfo is not None


def test_history_roundtrip() -> None:
    history = History(
        tests={"t.py::a": TestHistory(0.0123, 14, "passed", 0, 13)},
        overhead=Overhead(
            samples_s={1: 0.4, 2: 0.6}, counts={1: 3, 2: 4}, a=0.2, b=0.2, fit_points=[1, 2]
        ),
        last_merged_runs=["r1"],
    )
    assert History.from_dict(roundtrip(history.to_dict())) == history


def test_run_roundtrip() -> None:
    run = make_run()
    assert RunRecord.from_dict(roundtrip(run.to_dict())) == run
    no_startup = make_run(overhead_s=None)
    assert RunRecord.from_dict(roundtrip(no_startup.to_dict())) == no_startup


@pytest.mark.parametrize("version", [None, 0, 2, "1"])
def test_wrong_schema_version_rejected(version: object) -> None:
    data = History().to_dict()
    data["schema_version"] = version
    with pytest.raises(SchemaError, match="schema_version"):
        History.from_dict(data)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("workers",), True, "bool"),
        (("workers",), "2", "workers"),
        (("tests", 0, "outcome"), "exploded", "unknown outcome"),
        (("tests", 0, "duration"), -1, "negative"),
        (("started_at",), "yesterday", "ISO 8601"),
        (("randomly_reorder_disabled",), 1, "randomly_reorder_disabled"),
        (("overhead_s",), "fast", "overhead_s"),
    ],
)
def test_bad_run_fields_rejected(path: tuple, value: object, match: str) -> None:
    data = roundtrip(make_run().to_dict())
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(SchemaError, match=match):
        RunRecord.from_dict(data)


def test_missing_field_rejected() -> None:
    data = roundtrip(make_run().to_dict())
    del data["tests"][0]["worker"]
    with pytest.raises(SchemaError, match="missing field 'worker'"):
        RunRecord.from_dict(data)


def test_not_an_object_rejected() -> None:
    with pytest.raises(SchemaError):
        History.from_dict([])


def test_dependencies_roundtrip_and_validation() -> None:
    from smartpar.models import Conflict, Dependencies, FileAccess, Group, OrderEdge

    deps = Dependencies(
        file_access={"t::a": FileAccess(R=["<project>/x"], W=[])},
        state_changes={"t::a": ["env:X"]},
        conflicts=[Conflict("t::a", "t::b", "<project>/x", "W-R")],
        order_edges=[OrderEdge("a", "b", "r")],
        groups=[Group("g0", ["t::a", "t::b"], ["t::b", "t::a"], 1.5, ["file"])],
        profile_stats={"errors": 0},
    )
    assert Dependencies.from_dict(roundtrip(deps.to_dict())) == deps
    bad = roundtrip(deps.to_dict())
    bad["profile_stats"] = [1]
    with pytest.raises(SchemaError, match="profile_stats"):
        Dependencies.from_dict(bad)
    bad = roundtrip(deps.to_dict())
    bad["file_access"]["t::a"]["W"] = "x"
    with pytest.raises(SchemaError, match="list of strings"):
        Dependencies.from_dict(bad)


def test_flaky_order_edges_and_validation() -> None:
    from smartpar.models import Flaky, OrderEdge

    data = {
        "schema_version": 1,
        "runs": 20,
        "seeds": [101, 102],
        "od_victims": [{"victim": "v", "polluters": ["p1", "p2"], "fail_rate": "4/20"}],
        "od_brittle": [{"test": "b", "state_setters": ["s"], "fail_rate": "6/20"}],
        "nod": [{"test": "n", "fail_rate": "2/20", "note": "x"}],
        "unresolved": [],
    }
    flaky = Flaky.from_dict(data)
    assert flaky.order_edges() == [
        OrderEdge("s", "b", "setter-before-brittle"),
        OrderEdge("v", "p1", "victim-before-polluter"),
        OrderEdge("v", "p2", "victim-before-polluter"),
    ]
    assert flaky.nod_tests() == ["n"]
    assert Flaky.from_dict(roundtrip(flaky.to_dict())) == flaky
    data["od_victims"][0]["polluters"] = "p1"
    with pytest.raises(SchemaError):
        Flaky.from_dict(data)


@pytest.mark.parametrize(
    ("samples", "a", "b", "points"),
    [
        ({}, 0.0, 0.0, []),
        ({2: 0.8}, 0.8, 0.0, [2]),  # one point: constant
        ({1: 0.5, 2: 0.7, 4: 1.1}, 0.3, 0.2, [1, 2, 4]),  # exact line
        ({1: 0.5, 3: 9.0, 8: 9.0}, 0.5, 0.0, [1]),  # only m in (1, 2, 4) calibrates
    ],
)
def test_overhead_least_squares(samples: dict, a: float, b: float, points: list) -> None:
    model = Overhead(samples_s=dict(samples), counts=dict.fromkeys(samples, 3))
    model.refit()
    assert (model.a, model.b, model.fit_points) == (pytest.approx(a), pytest.approx(b), points)


def test_overhead_least_squares_noisy_and_clamped() -> None:
    model = Overhead(samples_s={1: 1.0, 2: 0.9, 4: 0.4}, counts={1: 5, 2: 3, 4: 9})
    model.refit()
    assert model.kind == "fitted"
    # OLS by hand: mean m = 7/3, mean y = 2.3/3; b = sxy / sxx.
    mean_m, mean_y = 7 / 3, 2.3 / 3
    sxy = sum((m - mean_m) * (y - mean_y) for m, y in model.samples_s.items())
    sxx = sum((m - mean_m) ** 2 for m in model.samples_s)
    assert model.b == pytest.approx(sxy / sxx)
    assert model.a == pytest.approx(mean_y - model.b * mean_m)
    assert model.predict(100) == 0.0  # a negative extrapolation is clamped


def test_overhead_needs_three_clean_runs_per_point() -> None:
    model = Overhead(samples_s={1: 0.5, 2: 0.7, 4: 1.1}, counts={1: 3, 2: 2, 4: 1})
    model.refit()
    assert (model.fit_points, model.kind, model.a, model.b) == ([1], "constant", 0.5, 0.0)
    model.counts = {1: 2, 2: 2, 4: 2}
    model.refit()
    assert (model.fit_points, model.kind) == ([], "default")
    assert (model.a, model.b) == (DEFAULT_OVERHEAD_A, DEFAULT_OVERHEAD_B)


def test_file_access_sources() -> None:
    from smartpar.models import FileAccess

    access = FileAccess()
    access.add("R", "<project>/a", "profile")
    access.add("R", "<project>/b", "detect")
    access.add("R", "<project>/b", "profile")  # first source wins
    access.add("W_tree", "<project>/d", "detect")
    assert access.R == ["<project>/a", "<project>/b"]
    assert (access.source("R", "<project>/a"), access.source("R", "<project>/b")) == (
        "profile",
        "detect",
    )
    assert access.only({"profile"}) == FileAccess(R=["<project>/a"])
    assert access.only({"detect"}).W_tree == ["<project>/d"]
    data = access.to_dict()
    assert data["source"]["R"] == {"<project>/a": "profile", "<project>/b": "detect"}
    from smartpar.models import Dependencies

    deps = Dependencies(file_access={"t": access})
    assert Dependencies.from_dict(roundtrip(deps.to_dict())).file_access["t"] == access


def test_file_access_directory_creations() -> None:
    from smartpar.models import Dependencies, FileAccess

    access = FileAccess(W=["<project>/out/f"])
    access.add("C", "<project>/out", "detect")
    assert access.only({"detect"}).C == ["<project>/out"]
    assert access.only({"profile"}) == FileAccess(W=["<project>/out/f"])
    data = Dependencies(file_access={"t": access}).to_dict()
    assert data["file_access"]["t"]["C"] == ["<project>/out"]
    assert Dependencies.from_dict(roundtrip(data)).file_access["t"] == access
    # A file written before directory creations were recorded has no "C".
    del data["file_access"]["t"]["C"]
    del data["file_access"]["t"]["source"]["C"]
    assert Dependencies.from_dict(roundtrip(data)).file_access["t"].C == []
