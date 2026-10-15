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


# Numbers of workers whose measured overhead calibrates the model.
CALIBRATION_WORKERS = (1, 2, 4)


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{what}: expected a number")
    return float(value)


# A calibration point is used only after this many clean runs at that m.
MIN_CLEAN_RUNS = 3
# Model used until enough clean runs exist ("default"): no overhead assumed,
# so T(m) is the LPT makespan plus the tail.
DEFAULT_OVERHEAD_A = 0.0
DEFAULT_OVERHEAD_B = 0.0


@dataclass
class Overhead:
    """Parallel overhead model ``overhead(m) = a + b*m`` (a hypothesis to test).

    ``samples_s`` holds the EMA of the measured overhead per number of
    workers, ``counts`` the number of clean runs behind it. After every merge
    ``a`` and ``b`` are refitted by least squares over the samples at m in
    CALIBRATION_WORKERS that have at least MIN_CLEAN_RUNS runs:
    two or more such points - "fitted"; one - "constant" (``a`` only);
    none - "default" (DEFAULT_OVERHEAD_A/B).
    """

    samples_s: dict[int, float] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    a: float = DEFAULT_OVERHEAD_A
    b: float = DEFAULT_OVERHEAD_B
    fit_points: list[int] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return {0: "default", 1: "constant"}.get(len(self.fit_points), "fitted")

    def predict(self, workers: int) -> float:
        return max(0.0, self.a + self.b * workers)

    def refit(self) -> None:
        points = sorted(
            (m, y)
            for m, y in self.samples_s.items()
            if m in CALIBRATION_WORKERS and self.counts.get(m, 0) >= MIN_CLEAN_RUNS
        )
        self.fit_points = [m for m, _y in points]
        if not points:
            self.a, self.b = DEFAULT_OVERHEAD_A, DEFAULT_OVERHEAD_B
        elif len(points) == 1:
            self.a, self.b = points[0][1], 0.0
        else:
            n = len(points)
            mean_m = sum(m for m, _ in points) / n
            mean_y = sum(y for _, y in points) / n
            sxx = sum((m - mean_m) ** 2 for m, _ in points)
            sxy = sum((m - mean_m) * (y - mean_y) for m, y in points)
            self.b = sxy / sxx
            self.a = mean_y - self.b * mean_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_s": {str(m): y for m, y in sorted(self.samples_s.items())},
            "counts": {str(m): c for m, c in sorted(self.counts.items())},
            "a": self.a,
            "b": self.b,
            "fit_points": self.fit_points,
            "model": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Any, what: str) -> Overhead:
        if not isinstance(data, dict):
            raise SchemaError(f"{what}: expected a JSON object")
        try:
            samples = {
                int(m): _number(y, f"{what} sample {m!r}")
                for m, y in _get(data, "samples_s", dict, what).items()
            }
            counts = {int(m): int(c) for m, c in data.get("counts", {}).items()}
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"{what}: keys must be worker counts") from exc
        points = _get(data, "fit_points", list, what)
        if not all(isinstance(m, int) and not isinstance(m, bool) for m in points):
            raise SchemaError(f"{what}: fit_points must be integers")
        return cls(
            samples_s=samples,
            counts=counts,
            a=_number(_get(data, "a", (int, float), what), what),
            b=_number(_get(data, "b", (int, float), what), what),
            fit_points=list(points),
        )


@dataclass
class History:
    tests: dict[str, TestHistory] = field(default_factory=dict)
    overhead: Overhead = field(default_factory=Overhead)
    # Run ids folded into this history whose files may still be in runs/.
    # Non-empty only if a merge was interrupted between writing history.json
    # and moving the run files; lets the next merge avoid counting them twice.
    last_merged_runs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **envelope(),
            "tests": {k: asdict(v) for k, v in sorted(self.tests.items())},
            "overhead": self.overhead.to_dict(),
            "last_merged_runs": sorted(self.last_merged_runs),
        }

    @classmethod
    def from_dict(cls, data: Any) -> History:
        what = "history.json"
        data = check_version(data, what)
        tests = _get(data, "tests", dict, what)
        merged = data.get("last_merged_runs", [])
        if not isinstance(merged, list) or not all(isinstance(r, str) for r in merged):
            raise SchemaError(f"{what}: last_merged_runs must be a list of strings")
        return cls(
            tests={k: TestHistory.from_dict(v, f"{what} test {k!r}") for k, v in tests.items()},
            overhead=Overhead.from_dict(_get(data, "overhead", dict, what), f"{what} overhead"),
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
    # Measured parallel overhead of this run (wall time not explained by
    # test durations), if known; calibrates the Overhead model.
    overhead_s: float | None = None
    # Prediction of the plan for this run (predicted_makespan_s), if any.
    predicted_wall_s: float | None = None
    # How the isolated tail ran: "barrier", "separate-pass" or "none".
    tail_mode: str = "none"
    # New tests that failed and were re-run alone in a child process:
    # {"nodeid", "first_outcome", "rerun_outcome", "duration", "suspected_dependency"}.
    reruns: list[dict[str, Any]] = field(default_factory=list)
    # Failed/error tests before and after re-running failed new tests alone
    # (a test that passed alone no longer counts after).
    failures_before_rerun: int = 0
    failures_after_rerun: int = 0

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
        overhead = data.get("overhead_s")
        if overhead is not None:
            overhead = _number(overhead, f"{what}: overhead_s")
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
            overhead_s=overhead,
            predicted_wall_s=(
                None
                if data.get("predicted_wall_s") is None
                else _number(data["predicted_wall_s"], f"{what}: predicted_wall_s")
            ),
            tail_mode=data.get("tail_mode", "none"),
            reruns=_obj_list(data.get("reruns", []), f"{what}: reruns"),
            failures_before_rerun=int(data.get("failures_before_rerun", 0)),
            failures_after_rerun=int(data.get("failures_after_rerun", 0)),
        )


# --- dependencies.json -----------------------------------------------------------


@dataclass
class FileAccess:
    R: list[str] = field(default_factory=list)
    W: list[str] = field(default_factory=list)
    # Directories whose whole subtree is written (rmtree, rmdir, rename/move of
    # a directory): conflicts with any access at or below the path.
    W_tree: list[str] = field(default_factory=list)


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
    # Position of each test in pytest's collection order at profiling time;
    # breaks ties of the order inside a group.
    collection_index: dict[str, int] = field(default_factory=dict)
    # Diagnostics of the profiling run: skipped/unresolved audit events,
    # modules snapshotted in cheap mode, etc.
    profile_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **envelope(),
            "file_access": {
                k: {"R": sorted(v.R), "W": sorted(v.W), "W_tree": sorted(v.W_tree)}
                for k, v in sorted(self.file_access.items())
            },
            "collection_index": dict(sorted(self.collection_index.items())),
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
                W_tree=_str_list(
                    access.get("W_tree", []), f"{what}: file_access {nodeid!r} W_tree"
                ),
            )
        state_changes = {
            k: _str_list(v, f"{what}: state_changes {k!r}")
            for k, v in _get(data, "state_changes", dict, what).items()
        }
        index = data.get("collection_index", {})
        if not isinstance(index, dict) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in index.values()
        ):
            raise SchemaError(f"{what}: collection_index must map nodeids to integers")
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
            collection_index=dict(index),
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
    # How the detection ran: base seed, isolation runs, budget, sandbox, ...
    detector: dict[str, Any] = field(default_factory=dict)

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
            detector=_get(data, "detector", dict, what) if "detector" in data else {},
        )
