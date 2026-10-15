"""Phase 2: atomic writes, run files, EMA merge and concurrent writers."""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smartpar import cli, history
from smartpar.models import History, RunRecord, SchemaError, TestRecord

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_run(
    run_id: str,
    started: datetime,
    durations: dict[str, float],
    outcome: str = "passed",
    startup: float | None = None,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        mode="smartpar",
        workers=1,
        started_at=started.isoformat(),
        finished_at=(started + timedelta(seconds=1)).isoformat(),
        wall_time_s=1.0,
        tests=[TestRecord(n, outcome, d, "gw0", 0.0, d) for n, d in durations.items()],
        overhead_s=startup,
    )


def no_temp_files(directory: Path) -> bool:
    return not any(p.name.endswith(".tmp") for p in directory.rglob("*"))


# --- atomic writes ------------------------------------------------------------------


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "f.json"
    history.atomic_write_json(target, {"a": 1})
    history.atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert no_temp_files(tmp_path)


def test_atomic_write_unserializable_keeps_original(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    history.atomic_write_json(target, {"a": 1})
    with pytest.raises(TypeError):
        history.atomic_write_json(target, {"a": object()})
    assert json.loads(target.read_text()) == {"a": 1}
    assert no_temp_files(tmp_path)


def test_atomic_write_failed_replace_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "f.json"
    history.atomic_write_json(target, {"a": 1})

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(history.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk on fire"):
        history.atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}
    assert no_temp_files(tmp_path)


def test_atomic_create_refuses_existing(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    history.atomic_create_json(target, {"a": 1})
    with pytest.raises(FileExistsError):
        history.atomic_create_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}
    assert no_temp_files(tmp_path)


def test_write_run_resolves_name_collision(tmp_path: Path) -> None:
    first = history.write_run(tmp_path, make_run("run_x_1", T0, {"a": 1.0}))
    second_record = make_run("run_x_1", T0, {"a": 2.0})
    second = history.write_run(tmp_path, second_record)
    assert first.name == "run_x_1.json"
    assert second.name == "run_x_1_1.json"
    assert second_record.run_id == "run_x_1_1"
    assert RunRecord.from_dict(json.loads(second.read_text())).run_id == "run_x_1_1"


def test_new_run_id_format() -> None:
    now = datetime(2026, 10, 15, 12, 30, 1, 123456, tzinfo=timezone.utc)
    assert history.new_run_id(pid=42, now=now) == "run_20261015T123001123456Z_42"


def test_load_history(tmp_path: Path) -> None:
    assert history.load_history(tmp_path) == History()
    (tmp_path / "history.json").write_text("{not json")
    with pytest.raises(SchemaError, match="invalid JSON"):
        history.load_history(tmp_path)


# --- EMA ----------------------------------------------------------------------------


def test_apply_run_ema_and_counters() -> None:
    h = History()
    history.apply_run(h, make_run("r1", T0, {"a": 1.0}, startup=0.5), alpha=0.3)
    assert h.tests["a"].duration_ema == 1.0  # first passed run initializes
    history.apply_run(h, make_run("r2", T0, {"a": 2.0}, startup=1.0), alpha=0.3)
    assert h.tests["a"].duration_ema == pytest.approx(0.3 * 2.0 + 0.7 * 1.0)
    # Both runs have workers=1: one calibration point, EMA of the samples.
    assert h.overhead.samples_s == {1: pytest.approx(0.3 * 1.0 + 0.7 * 0.5)}
    assert (h.overhead.a, h.overhead.b) == (pytest.approx(0.65), 0.0)
    entry = h.tests["a"]
    assert (entry.runs, entry.timed_runs, entry.fail_count, entry.last_outcome) == (
        2,
        2,
        0,
        "passed",
    )


@pytest.mark.parametrize("outcome", ["failed", "error", "skipped", "xfailed", "xpassed"])
def test_apply_run_only_passed_updates_duration(outcome: str) -> None:
    h = History()
    history.apply_run(h, make_run("r1", T0, {"a": 9.0}, outcome=outcome), alpha=0.3)
    assert (h.tests["a"].runs, h.tests["a"].timed_runs, h.tests["a"].duration_ema) == (1, 0, 0.0)
    history.apply_run(h, make_run("r2", T0, {"a": 2.0}), alpha=0.3)
    assert h.tests["a"].duration_ema == 2.0  # first *passed* run initializes
    history.apply_run(h, make_run("r3", T0, {"a": 9.0}, outcome=outcome), alpha=0.3)
    entry = h.tests["a"]
    assert entry.duration_ema == 2.0
    assert (entry.runs, entry.timed_runs, entry.last_outcome) == (3, 1, outcome)
    assert entry.fail_count == (2 if outcome in ("failed", "error") else 0)


def test_profile_runs_update_counters_only() -> None:
    h = History()
    history.apply_run(h, make_run("r1", T0, {"a": 1.0}), alpha=0.3)
    profile_run = make_run("r2", T0, {"a": 50.0, "b": 9.0}, startup=7.0)
    profile_run.mode = "profile"
    history.apply_run(h, profile_run, alpha=0.3)
    assert (h.tests["a"].duration_ema, h.tests["a"].timed_runs, h.tests["a"].runs) == (1.0, 1, 2)
    assert (h.tests["b"].duration_ema, h.tests["b"].timed_runs, h.tests["b"].runs) == (0.0, 0, 1)
    assert h.overhead.samples_s == {}


# --- merge --------------------------------------------------------------------------


def test_merge_sorts_by_start_time_not_name(tmp_path: Path) -> None:
    # Names sort opposite to start times; EMA is order-sensitive, so the
    # result reveals which order was used.
    history.write_run(tmp_path, make_run("run_a", T0 + timedelta(seconds=10), {"t": 4.0}))
    history.write_run(tmp_path, make_run("run_b", T0, {"t": 1.0}))
    result = history.merge(tmp_path, alpha=0.5)
    assert result.merged == ["run_b", "run_a"]
    assert history.load_history(tmp_path).tests["t"].duration_ema == pytest.approx(2.5)


def test_merge_moves_files_and_is_idempotent(tmp_path: Path) -> None:
    history.write_run(tmp_path, make_run("run_1", T0, {"t": 1.0}))
    assert history.merge(tmp_path).merged == ["run_1"]
    assert not list(history.iter_run_files(tmp_path))
    assert (tmp_path / "runs" / "merged" / "run_1.json").exists()
    before = history.load_history(tmp_path)
    second = history.merge(tmp_path)
    assert second.merged == [] and second.already_merged == []
    assert history.load_history(tmp_path) == before


def _crash_when(monkeypatch: pytest.MonkeyPatch, predicate: Callable[[str], bool]) -> None:
    """Simulate the process being killed at the first os.replace matching predicate."""
    real_replace = os.replace

    def replace(src: object, dst: object) -> None:
        if predicate(str(dst)):
            raise KeyboardInterrupt
        real_replace(src, dst)

    monkeypatch.setattr(history.os, "replace", replace)


def test_merge_leaves_last_merged_runs_empty(tmp_path: Path) -> None:
    history.write_run(tmp_path, make_run("run_1", T0, {"t": 1.0}))
    history.merge(tmp_path)
    assert history.load_history(tmp_path).last_merged_runs == []


def test_crash_between_history_write_and_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history.write_run(tmp_path, make_run("run_1", T0, {"t": 1.0}))
    _crash_when(monkeypatch, lambda dst: "merged" in dst)
    with pytest.raises(KeyboardInterrupt):
        history.merge(tmp_path)
    monkeypatch.undo()

    crashed = history.load_history(tmp_path)
    assert crashed.tests["t"].runs == 1
    assert crashed.last_merged_runs == ["run_1"]
    assert [p.name for p in history.iter_run_files(tmp_path)] == ["run_1.json"]

    history.write_run(tmp_path, make_run("run_2", T0 + timedelta(seconds=1), {"t": 3.0}))
    result = history.merge(tmp_path, alpha=0.5)
    assert result.already_merged == ["run_1"]
    assert result.merged == ["run_2"]
    recovered = history.load_history(tmp_path)
    assert recovered.tests["t"].runs == 2  # run_1 not counted twice
    assert recovered.tests["t"].duration_ema == pytest.approx(2.0)
    assert recovered.last_merged_runs == []
    assert not list(history.iter_run_files(tmp_path))


def test_crash_after_move_before_list_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history.write_run(tmp_path, make_run("run_1", T0, {"t": 1.0}))
    calls = {"history": 0}

    def second_history_write(dst: str) -> bool:
        if dst.endswith("history.json"):
            calls["history"] += 1
            return calls["history"] == 2
        return False

    _crash_when(monkeypatch, second_history_write)
    with pytest.raises(KeyboardInterrupt):
        history.merge(tmp_path)
    monkeypatch.undo()

    assert history.load_history(tmp_path).last_merged_runs == ["run_1"]  # stale id
    assert not list(history.iter_run_files(tmp_path))
    result = history.merge(tmp_path)
    assert result.merged == [] and result.already_merged == []
    healed = history.load_history(tmp_path)
    assert healed.last_merged_runs == []
    assert healed.tests["t"].runs == 1


def test_late_run_with_earlier_start_is_merged(tmp_path: Path) -> None:
    # A run that started earlier but whose file arrives after a merge must
    # still be folded in (no time watermark). With incremental merges its
    # effect on the EMA depends on when it is merged - documented limitation.
    alpha = 0.3  # not 0.5: with two runs the EMA would be order-symmetric
    history.write_run(tmp_path, make_run("run_later", T0 + timedelta(seconds=10), {"t": 4.0}))
    history.merge(tmp_path, alpha=alpha)
    history.write_run(tmp_path, make_run("run_early", T0, {"t": 1.0}))
    result = history.merge(tmp_path, alpha=alpha)
    assert result.merged == ["run_early"]
    incremental = history.load_history(tmp_path).tests["t"]
    assert incremental.runs == 2
    assert incremental.duration_ema == pytest.approx(0.3 * 1.0 + 0.7 * 4.0)  # applied last

    # The same two runs in one merge are applied in start-time order.
    one_shot = tmp_path / "one_shot"
    history.write_run(one_shot, make_run("run_later", T0 + timedelta(seconds=10), {"t": 4.0}))
    history.write_run(one_shot, make_run("run_early", T0, {"t": 1.0}))
    history.merge(one_shot, alpha=alpha)
    single = history.load_history(one_shot).tests["t"]
    assert single.duration_ema == pytest.approx(0.3 * 4.0 + 0.7 * 1.0)
    assert single.duration_ema != pytest.approx(incremental.duration_ema)


def test_merge_skips_invalid_run_file(tmp_path: Path) -> None:
    history.write_run(tmp_path, make_run("run_ok", T0, {"t": 1.0}))
    bad = tmp_path / "runs" / "run_bad.json"
    bad.write_text('{"schema_version": 1}')
    result = history.merge(tmp_path)
    assert result.merged == ["run_ok"]
    assert result.invalid == [bad]
    assert bad.exists()


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_merge_rejects_bad_alpha(tmp_path: Path, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        history.merge(tmp_path, alpha=alpha)


def test_cli_merge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    history.write_run(tmp_path, make_run("run_1", T0, {"t": 1.0}))
    assert cli.main(["--dir", str(tmp_path), "merge", "--alpha", "0.5"]) == 0
    assert "merged 1 run(s)" in capsys.readouterr().out
    (tmp_path / "runs" / "run_bad.json").write_text("garbage")
    assert cli.main(["--dir", str(tmp_path), "merge"]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--dir", str(tmp_path), "merge", "--alpha", "2"])


# --- concurrency --------------------------------------------------------------------

N_PROCS = 4
RUNS_PER_PROC = 25
TESTS = [f"tests/test_m.py::test_{k}" for k in range(5)]


def _duration(proc: int, j: int, k: int) -> float:
    return round(0.01 * (1 + proc) + 0.001 * j + 0.1 * k, 6)


def _started(proc: int, j: int) -> datetime:
    # Interleaved, unique start times: the merge order is fully determined.
    return T0 + timedelta(seconds=j * N_PROCS + proc)


def _write_runs(directory: str, proc: int, barrier: multiprocessing.synchronize.Barrier) -> None:
    barrier.wait()
    for j in range(RUNS_PER_PROC):
        durations = {name: _duration(proc, j, k) for k, name in enumerate(TESTS)}
        history.write_run(
            Path(directory), make_run(history.new_run_id(), _started(proc, j), durations)
        )


def _expected_history(alpha: float) -> History:
    expected = History()
    order = sorted(
        ((proc, j) for proc in range(N_PROCS) for j in range(RUNS_PER_PROC)),
        key=lambda pj: _started(*pj),
    )
    for proc, j in order:
        durations = {name: _duration(proc, j, k) for k, name in enumerate(TESTS)}
        history.apply_run(expected, make_run("x", _started(proc, j), durations), alpha)
    return expected


def _concurrent_write_then_merge(directory: Path) -> History:
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(N_PROCS)
    procs = [
        ctx.Process(target=_write_runs, args=(str(directory), p, barrier)) for p in range(N_PROCS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert len(list(history.iter_run_files(directory))) == N_PROCS * RUNS_PER_PROC
    assert no_temp_files(directory)
    result = history.merge(directory, alpha=0.3)
    assert len(result.merged) == N_PROCS * RUNS_PER_PROC
    assert not result.invalid
    return history.load_history(directory)


def test_concurrent_run_files_then_deterministic_merge(tmp_path: Path) -> None:
    first = _concurrent_write_then_merge(tmp_path / "a")
    second = _concurrent_write_then_merge(tmp_path / "b")
    expected = _expected_history(alpha=0.3)
    assert first.tests == expected.tests
    assert second.tests == expected.tests
    assert first.tests[TESTS[0]].runs == N_PROCS * RUNS_PER_PROC


def _hammer(path: str, proc: int, barrier: multiprocessing.synchronize.Barrier) -> None:
    barrier.wait()
    for i in range(200):
        history.atomic_write_json(Path(path), {"proc": proc, "i": i, "pad": "x" * 50_000})


def test_concurrent_atomic_writes_never_expose_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "history.json"
    history.atomic_write_json(target, {"proc": -1, "i": -1, "pad": ""})
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(N_PROCS + 1)
    procs = [ctx.Process(target=_hammer, args=(str(target), p, barrier)) for p in range(N_PROCS)]
    for p in procs:
        p.start()
    barrier.wait()
    reads = 0
    while any(p.is_alive() for p in procs):
        data = json.loads(target.read_text())  # must always be complete JSON
        assert set(data) == {"proc", "i", "pad"}
        reads += 1
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert reads > 0
    assert json.loads(target.read_text())["i"] == 199
    assert no_temp_files(tmp_path)
