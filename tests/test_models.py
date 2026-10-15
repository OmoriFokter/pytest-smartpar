"""Phase 2: schemas of history.json and run files."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from smartpar import __version__
from smartpar.models import (
    SCHEMA_VERSION,
    History,
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
        "worker_startup_s": 0.4,
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
        overhead={"worker_startup_s": 0.41},
        last_merged_runs=["r1"],
    )
    assert History.from_dict(roundtrip(history.to_dict())) == history


def test_run_roundtrip() -> None:
    run = make_run()
    assert RunRecord.from_dict(roundtrip(run.to_dict())) == run
    no_startup = make_run(worker_startup_s=None)
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
        (("worker_startup_s",), "fast", "worker_startup_s"),
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
    from smartpar.models import Dependencies, FileAccess

    deps = Dependencies(
        file_access={"t::a": FileAccess(R=["<project>/x"], W=[])},
        state_changes={"t::a": ["env:X"]},
        order_edges=[{"before": "a", "after": "b", "reason": "r"}],
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
