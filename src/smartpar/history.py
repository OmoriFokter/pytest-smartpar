"""Method step 5 (recording): run history without locks.

Readers-writers model:

* a normal run only *reads* history.json (a snapshot at start) and *writes* a
  separate ``runs/run_<utc>_<pid>.json``, so concurrent runs never contend for
  a shared file;
* ``smartpar merge`` folds the run files into history.json with an
  exponential moving average ``duration_ema = alpha*new + (1-alpha)*old``,
  processing runs sorted by start time, then moves them to ``runs/merged/``;
* every file is written atomically: a temporary file in the same directory,
  fsync, then ``os.replace`` (or ``os.link`` where the name must be new), so a
  reader sees either the old or the new complete file, never a partial one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartpar.models import (
    FAILED_OUTCOMES,
    UNTIMED_MODES,
    Flaky,
    History,
    RunRecord,
    SchemaError,
    TestHistory,
)

try:  # POSIX only; on Windows concurrent merges are not guarded.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger("smartpar.history")

HISTORY_FILE = "history.json"
FLAKY_FILE = "flaky.json"
RUNS_DIR = "runs"
MERGED_DIR = "merged"
MERGE_LOCK = ".merge.lock"
DEFAULT_ALPHA = 0.3


def _dump(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _write_temp(directory: Path, name: str, payload: bytes) -> Path:
    """Write payload to a fsync-ed temporary file next to the target."""
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return Path(tmp)


def atomic_write_json(path: Path, data: Any) -> None:
    """Replace ``path`` with ``data`` atomically."""
    # Serialize first: a TypeError must not leave a temp file.
    atomic_write_bytes(path, _dump(data))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _write_temp(path.parent, path.name, payload)
    try:
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_create_json(path: Path, data: Any) -> None:
    """Create ``path`` atomically; raise FileExistsError if it already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _write_temp(path.parent, path.name, _dump(data))
    try:
        os.link(tmp, path)  # fails instead of overwriting
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def load_history(smartpar_dir: Path) -> History:
    """Read history.json; an absent file is an empty history."""
    path = Path(smartpar_dir) / HISTORY_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return History()
    try:
        return History.from_dict(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: invalid JSON: {exc}") from exc


def load_flaky(smartpar_dir: Path) -> Flaky | None:
    """Read flaky.json; None if the detector has not run."""
    path = Path(smartpar_dir) / FLAKY_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return Flaky.from_dict(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: invalid JSON: {exc}") from exc


def save_history(smartpar_dir: Path, history: History) -> None:
    atomic_write_json(Path(smartpar_dir) / HISTORY_FILE, history.to_dict())


def new_run_id(pid: int | None = None, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"run_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{pid if pid is not None else os.getpid()}"


def write_run(smartpar_dir: Path, record: RunRecord) -> Path:
    """Write a run file under runs/ without touching history.json.

    The name comes from ``record.run_id``; if that name is taken (same pid,
    same microsecond), a numeric suffix is appended and ``run_id`` updated.
    """
    runs = Path(smartpar_dir) / RUNS_DIR
    base = record.run_id
    for attempt in range(1000):
        if attempt:
            record.run_id = f"{base}_{attempt}"
        path = runs / f"{record.run_id}.json"
        try:
            atomic_create_json(path, record.to_dict())
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"cannot find a free run file name for {base}")


def iter_run_files(smartpar_dir: Path) -> Iterator[Path]:
    yield from sorted((Path(smartpar_dir) / RUNS_DIR).glob("run_*.json"))


def apply_run(history: History, run: RunRecord, alpha: float) -> None:
    """Fold one run into the history (EMA of durations, counters).

    Only passed runs update ``duration_ema`` and ``timed_runs``: a skipped test
    did not do its work, and a failed one may have stopped early or hung, so
    neither duration estimates the cost of a normal run. The first passed run
    initializes the EMA. Runs in an instrumented mode (``UNTIMED_MODES``, e.g.
    profiling) never update durations. ``runs``, ``last_outcome`` and
    ``fail_count`` count every run.
    """
    timed = run.mode not in UNTIMED_MODES
    for rec in run.tests:
        entry = history.tests.setdefault(
            rec.nodeid,
            TestHistory(duration_ema=0.0, runs=0, last_outcome="", fail_count=0, timed_runs=0),
        )
        if timed and rec.outcome == "passed":
            if entry.timed_runs == 0:
                entry.duration_ema = rec.duration
            else:
                entry.duration_ema = alpha * rec.duration + (1 - alpha) * entry.duration_ema
            entry.timed_runs += 1
        entry.runs += 1
        entry.last_outcome = rec.outcome
        entry.fail_count += int(rec.outcome in FAILED_OUTCOMES)
    if timed and run.overhead_s is not None:
        samples = history.overhead.samples_s
        prev = samples.get(run.workers)
        samples[run.workers] = (
            run.overhead_s if prev is None else alpha * run.overhead_s + (1 - alpha) * prev
        )
        history.overhead.refit()


@dataclass
class MergeResult:
    merged: list[str] = field(default_factory=list)
    already_merged: list[str] = field(default_factory=list)
    invalid: list[Path] = field(default_factory=list)


@contextlib.contextmanager
def _merge_lock(smartpar_dir: Path) -> Iterator[None]:
    """Serialize concurrent ``smartpar merge`` calls (normal runs never take it)."""
    if fcntl is None:  # pragma: no cover
        yield
        return
    Path(smartpar_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(smartpar_dir) / MERGE_LOCK, "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _run_sort_key(run: RunRecord) -> tuple[datetime, str]:
    started = datetime.fromisoformat(run.started_at)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started, run.run_id


def merge(smartpar_dir: Path, alpha: float = DEFAULT_ALPHA) -> MergeResult:
    """Fold all pending run files into history.json and move them to runs/merged/.

    Crash safety (two-phase):

    1. history.json is written with the folded runs *and* their ids in
       ``last_merged_runs``;
    2. the run files are moved to runs/merged/, then history.json is written
       again with those ids removed, so the list stays empty between merges.

    A crash between 1 and 2 leaves run files whose ids are listed: the next
    merge only moves them. A crash after the move but before the second write
    leaves ids without files: the next merge drops them. No time watermark is
    used, so a late run with an earlier ``started_at`` is never lost.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    smartpar_dir = Path(smartpar_dir)
    result = MergeResult()
    with _merge_lock(smartpar_dir):
        history = load_history(smartpar_dir)
        previously_merged = set(history.last_merged_runs)
        pending: list[tuple[RunRecord, Path]] = []
        leftovers: list[Path] = []
        for path in iter_run_files(smartpar_dir):
            try:
                run = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, SchemaError, UnicodeDecodeError) as exc:
                log.warning("smartpar: skipping invalid run file %s: %s", path, exc)
                result.invalid.append(path)
                continue
            if run.run_id in previously_merged:
                leftovers.append(path)
                result.already_merged.append(run.run_id)
            else:
                pending.append((run, path))

        pending.sort(key=lambda item: _run_sort_key(item[0]))
        for run, _path in pending:
            apply_run(history, run, alpha)
            result.merged.append(run.run_id)

        # Phase 1: ids of every run folded in but not yet moved.
        in_flight = sorted(result.already_merged + result.merged)
        if pending:
            history.last_merged_runs = in_flight
            save_history(smartpar_dir, history)

        merged_dir = smartpar_dir / RUNS_DIR / MERGED_DIR
        merged_dir.mkdir(parents=True, exist_ok=True)
        for path in [p for _r, p in pending] + leftovers:
            os.replace(path, merged_dir / path.name)

        # Phase 2: all those files are moved; ids listed without a file
        # (crash after a previous move) are dropped as well.
        if history.last_merged_runs:
            history.last_merged_runs = []
            save_history(smartpar_dir, history)
    return result
