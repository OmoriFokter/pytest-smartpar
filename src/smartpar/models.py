"""Data model of the method: dataclasses for the output files in .smartpar/.

Every JSON file carries the common envelope ``schema_version``,
``generated_at`` (ISO 8601, UTC) and ``tool_version``. ``from_dict`` methods
are the schema: they check the version and the types of the fields they read
and raise ``SchemaError`` on anything unexpected, so a damaged or foreign file
is never silently interpreted.

This module covers history.json and run files runs/run_*.json (step 5) and
dependencies.json (steps 1-2). The models of the other files are added with
the phases that produce them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from smartpar import __version__

SCHEMA_VERSION = 1

# Outcomes that count as a failure of the test in history.json.
FAILED_OUTCOMES = frozenset({"failed", "error"})
OUTCOMES = frozenset({"passed", "failed", "error", "skipped", "xfailed", "xpassed"})
# Run modes whose timings are distorted by instrumentation: such runs update
# the counters in history.json but not duration_ema.
UNTIMED_MODES = frozenset({"profile"})


class SchemaError(ValueError):
    """A data file does not match the expected schema."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def envelope() -> dict[str, Any]:
    """Common header of every smartpar JSON file."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "tool_version": __version__,
    }


def check_version(data: Any, what: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaError(f"{what}: expected a JSON object")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(f"{what}: unsupported schema_version {version!r}")
    return data


def _get(data: dict[str, Any], key: str, types: type | tuple[type, ...], what: str) -> Any:
    if key not in data:
        raise SchemaError(f"{what}: missing field {key!r}")
    value = data[key]
    # bool is a subclass of int: accept it only where bool is asked for.
    if not isinstance(value, types) or (isinstance(value, bool) and types is not bool):
        raise SchemaError(f"{what}: field {key!r} has type {type(value).__name__}")
    return value


# --- history.json -------------------------------------------------------------


@dataclass
class TestHistory:
    __test__ = False  # not a pytest test class

    duration_ema: float
    runs: int
    last_outcome: str
    fail_count: int
    # Passed runs, the only ones that contribute to duration_ema.
    timed_runs: int

    @classmethod
    def from_dict(cls, data: Any, what: str) -> TestHistory:
        if not isinstance(data, dict):
            raise SchemaError(f"{what}: expected a JSON object")
        return cls(
            duration_ema=float(_get(data, "duration_ema", (int, float), what)),
            runs=_get(data, "runs", int, what),
            last_outcome=_get(data, "last_outcome", str, what),
            fail_count=_get(data, "fail_count", int, what),
            timed_runs=_get(data, "timed_runs", int, what),
        )


@dataclass
class History:
    tests: dict[str, TestHistory] = field(default_factory=dict)
    # Measured overheads, e.g. {"worker_startup_s": 0.41}.
    overhead: dict[str, float] = field(default_factory=dict)
    # Run ids folded into this history whose files may still be in runs/.
    # Non-empty only if a merge was interrupted between writing history.json
    # and moving the run files; lets the next merge avoid counting them twice.
    last_merged_runs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **envelope(),
            "tests": {k: asdict(v) for k, v in sorted(self.tests.items())},
            "overhead": dict(sorted(self.overhead.items())),
            "last_merged_runs": sorted(self.last_merged_runs),
        }

    @classmethod
    def from_dict(cls, data: Any) -> History:
        what = "history.json"
        data = check_version(data, what)
        tests = _get(data, "tests", dict, what)
        overhead = _get(data, "overhead", dict, what)
        for key, value in overhead.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchemaError(f"{what}: overhead {key!r} is not a number")
        merged = data.get("last_merged_runs", [])
        if not isinstance(merged, list) or not all(isinstance(r, str) for r in merged):
            raise SchemaError(f"{what}: last_merged_runs must be a list of strings")
        return cls(
            tests={k: TestHistory.from_dict(v, f"{what} test {k!r}") for k, v in tests.items()},
            overhead={k: float(v) for k, v in overhead.items()},
            last_merged_runs=list(merged),
        )


# --- runs/run_<utc>_<pid>.json ---------------------------------------------------


@dataclass
class TestRecord:
    """One test in one run.

    ``duration`` is setup + call + teardown in seconds; ``start``/``end`` are
    ``time.time()`` timestamps.
    """

    __test__ = False

    nodeid: str
    outcome: str
    duration: float
    worker: str
    start: float
    end: float

    @classmethod
    def from_dict(cls, data: Any, what: str) -> TestRecord:
        if not isinstance(data, dict):
            raise SchemaError(f"{what}: expected a JSON object")
        outcome = _get(data, "outcome", str, what)
        if outcome not in OUTCOMES:
            raise SchemaError(f"{what}: unknown outcome {outcome!r}")
        duration = float(_get(data, "duration", (int, float), what))
        if duration < 0:
            raise SchemaError(f"{what}: negative duration")
        return cls(
            nodeid=_get(data, "nodeid", str, what),
            outcome=outcome,
            duration=duration,
            worker=_get(data, "worker", str, what),
            start=float(_get(data, "start", (int, float), what)),
            end=float(_get(data, "end", (int, float), what)),
        )


@dataclass
class RunRecord:
    """Raw data of one pytest run (spec 3.7)."""

    run_id: str
    mode: str
    workers: int
    started_at: str
    finished_at: str
    wall_time_s: float
    tests: list[TestRecord] = field(default_factory=list)
    randomly_reorder_disabled: bool = False
    # Measured worker start-up time of this run, if known.
    worker_startup_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {**envelope(), **data}

    @classmethod
    def from_dict(cls, data: Any) -> RunRecord:
        what = "run file"
        data = check_version(data, what)
        run_id = _get(data, "run_id", str, what)
        what = f"run file {run_id!r}"
        tests = _get(data, "tests", list, what)
        startup = data.get("worker_startup_s")
        if startup is not None and (
            isinstance(startup, bool) or not isinstance(startup, (int, float))
        ):
            raise SchemaError(f"{what}: worker_startup_s is not a number")
        started_at = _get(data, "started_at", str, what)
        try:
            datetime.fromisoformat(started_at)
        except ValueError as exc:
            raise SchemaError(f"{what}: started_at is not ISO 8601") from exc
        return cls(
            run_id=run_id,
            mode=_get(data, "mode", str, what),
            workers=_get(data, "workers", int, what),
            started_at=started_at,
            finished_at=_get(data, "finished_at", str, what),
            wall_time_s=float(_get(data, "wall_time_s", (int, float), what)),
            tests=[TestRecord.from_dict(t, f"{what} test #{i}") for i, t in enumerate(tests)],
            randomly_reorder_disabled=_get(data, "randomly_reorder_disabled", bool, what),
            worker_startup_s=None if startup is None else float(startup),
        )


# --- dependencies.json -----------------------------------------------------------


@dataclass
class FileAccess:
    R: list[str] = field(default_factory=list)
    W: list[str] = field(default_factory=list)


def _str_list(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SchemaError(f"{what}: expected a list of strings")
    return list(value)


def _obj_list(value: Any, what: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise SchemaError(f"{what}: expected a list of objects")
    return list(value)


CONFLICT_KINDS = frozenset({"W-R", "R-W", "W-W"})


@dataclass(frozen=True, order=True)
class Conflict:
    """Tests ``a`` < ``b`` violate Bernstein's conditions on ``resource``.

    ``kind`` is from a's side: "W-R" a writes and b reads, "R-W" the reverse,
    "W-W" both write.
    """

    a: str
    b: str
    resource: str
    kind: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], what: str) -> Conflict:
        kind = _get(data, "kind", str, what)
        if kind not in CONFLICT_KINDS:
            raise SchemaError(f"{what}: unknown conflict kind {kind!r}")
        return cls(
            _get(data, "a", str, what),
            _get(data, "b", str, what),
            _get(data, "resource", str, what),
            kind,
        )


@dataclass(frozen=True, order=True)
class OrderEdge:
    """``before`` must run before ``after`` when both run in one process."""

    before: str
    after: str
    reason: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], what: str) -> OrderEdge:
        return cls(
            _get(data, "before", str, what),
            _get(data, "after", str, what),
            _get(data, "reason", str, what),
        )


@dataclass
class Group:
    """Tests linked by conflicts or order edges: one worker, ``order`` in sequence."""

    id: str
    tests: list[str]
    order: list[str]
    est_duration: float
    reasons: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any], what: str) -> Group:
        return cls(
            id=_get(data, "id", str, what),
            tests=_str_list(_get(data, "tests", list, what), f"{what} tests"),
            order=_str_list(_get(data, "order", list, what), f"{what} order"),
            est_duration=float(_get(data, "est_duration", (int, float), what)),
            reasons=_str_list(_get(data, "reasons", list, what), f"{what} reasons"),
        )


@dataclass
class Dependencies:
    """dependencies.json (spec 3.2).

    Paths are resources: ``<project>/relative/path`` inside the project root,
    absolute paths outside it. ``conflicts`` and ``groups`` are derived by the
    graph step; ``order_edges`` mirror the confirmed pairs of flaky.json.
    """

    file_access: dict[str, FileAccess] = field(default_factory=dict)
    state_changes: dict[str, list[str]] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    order_edges: list[OrderEdge] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    # Diagnostics of the profiling run: skipped/unresolved audit events,
    # modules snapshotted in cheap mode, etc.
    profile_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **envelope(),
            "file_access": {
                k: {"R": sorted(v.R), "W": sorted(v.W)} for k, v in sorted(self.file_access.items())
            },
            "state_changes": {k: sorted(v) for k, v in sorted(self.state_changes.items())},
            "conflicts": [asdict(c) for c in sorted(self.conflicts)],
            "order_edges": [asdict(e) for e in sorted(self.order_edges)],
            "groups": [asdict(g) for g in self.groups],
            "profile_stats": self.profile_stats,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Dependencies:
        what = "dependencies.json"
        data = check_version(data, what)
        file_access = {}
        for nodeid, access in _get(data, "file_access", dict, what).items():
            if not isinstance(access, dict):
                raise SchemaError(f"{what}: file_access {nodeid!r} is not an object")
            file_access[nodeid] = FileAccess(
                R=_str_list(access.get("R"), f"{what}: file_access {nodeid!r} R"),
                W=_str_list(access.get("W"), f"{what}: file_access {nodeid!r} W"),
            )
        state_changes = {
            k: _str_list(v, f"{what}: state_changes {k!r}")
            for k, v in _get(data, "state_changes", dict, what).items()
        }
        # Optional: absent when the file was created by the detector alone.
        stats = _get(data, "profile_stats", dict, what) if "profile_stats" in data else {}
        return cls(
            file_access=file_access,
            state_changes=state_changes,
            conflicts=[
                Conflict.from_dict(c, f"{what}: conflict")
                for c in _obj_list(_get(data, "conflicts", list, what), f"{what}: conflicts")
            ],
            order_edges=[
                OrderEdge.from_dict(e, f"{what}: order edge")
                for e in _obj_list(_get(data, "order_edges", list, what), f"{what}: order_edges")
            ],
            groups=[
                Group.from_dict(g, f"{what}: group")
                for g in _obj_list(_get(data, "groups", list, what), f"{what}: groups")
            ],
            profile_stats=stats,
        )


# --- flaky.json --------------------------------------------------------------------

VICTIM_BEFORE_POLLUTER = "victim-before-polluter"
SETTER_BEFORE_BRITTLE = "setter-before-brittle"


@dataclass
class Flaky:
    """flaky.json (spec 3.3); only confirmed pairs are ever written here.

    Entries are kept as plain objects (the detector, phase 6, owns their
    details); ``order_edges`` and ``nod_tests`` are what scheduling needs.
    """

    runs: int = 0
    seeds: list[int] = field(default_factory=list)
    od_victims: list[dict[str, Any]] = field(default_factory=list)
    od_brittle: list[dict[str, Any]] = field(default_factory=list)
    nod: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[Any] = field(default_factory=list)

    def order_edges(self) -> list[OrderEdge]:
        """Victim before each of its polluters; each state setter before its brittle test."""
        edges = {
            OrderEdge(v["victim"], polluter, VICTIM_BEFORE_POLLUTER)
            for v in self.od_victims
            for polluter in v["polluters"]
        }
        edges |= {
            OrderEdge(setter, b["test"], SETTER_BEFORE_BRITTLE)
            for b in self.od_brittle
            for setter in b["state_setters"]
        }
        return sorted(edges)

    def nod_tests(self) -> list[str]:
        return sorted(n["test"] for n in self.nod)

    def to_dict(self) -> dict[str, Any]:
        return {**envelope(), **asdict(self)}

    @classmethod
    def from_dict(cls, data: Any) -> Flaky:
        what = "flaky.json"
        data = check_version(data, what)
        victims = _obj_list(_get(data, "od_victims", list, what), f"{what}: od_victims")
        for v in victims:
            _get(v, "victim", str, f"{what}: od_victims")
            _str_list(_get(v, "polluters", list, f"{what}: od_victims"), f"{what}: polluters")
        brittle = _obj_list(_get(data, "od_brittle", list, what), f"{what}: od_brittle")
        for b in brittle:
            _get(b, "test", str, f"{what}: od_brittle")
            _str_list(_get(b, "state_setters", list, f"{what}: od_brittle"), f"{what}: setters")
        nod = _obj_list(_get(data, "nod", list, what), f"{what}: nod")
        for n in nod:
            _get(n, "test", str, f"{what}: nod")
        seeds = _get(data, "seeds", list, what)
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in seeds):
            raise SchemaError(f"{what}: seeds must be integers")
        return cls(
            runs=_get(data, "runs", int, what),
            seeds=list(seeds),
            od_victims=victims,
            od_brittle=brittle,
            nod=nod,
            unresolved=list(_get(data, "unresolved", list, what)),
        )
