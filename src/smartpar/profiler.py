"""Method step 1 (profiling): per-test file access and in-process state changes.

* File read/write sets R and W of every test are recorded by an audit hook
  (``sys.addaudithook``). Accesses during setup, call and teardown, including
  fixtures, are attributed to the running test.
* Snapshots of process state before and after every test (globals of project
  modules, ``os.environ``, cwd, ``sys.path``, root logger) reveal tests that
  change state and do not restore it: polluter candidates.

Limitations (documented in README):

* only Python-level operations are visible; C extensions that call libc
  directly bypass audit hooks;
* ``open`` events carry no ``dir_fd``, so ``os.open(rel, dir_fd=fd)`` is
  resolved against the cwd; other events with a ``dir_fd`` and a relative path
  are counted as unresolved and skipped;
* symlinks are not resolved;
* session-scoped fixtures are attributed to the test during which they are set
  up or torn down;
* a value that can be neither pickled nor meaningfully repr-ed is compared by
  identity, so in-place changes of such objects are not seen.
"""

from __future__ import annotations

import contextlib
import fnmatch
import functools
import inspect
import json
import logging
import os
import pickle
import sys
import time
import uuid
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from smartpar.graph import build as build_graph
from smartpar.history import atomic_write_json, load_flaky, load_history
from smartpar.models import (
    ACCESS_KINDS,
    SOURCE_DETECT,
    SOURCE_PROFILE,
    Dependencies,
    FileAccess,
    Flaky,
    History,
    SchemaError,
    check_version,
    envelope,
)
from smartpar.scheduler import estimate_durations

log = logging.getLogger("smartpar.profiler")

PROJECT_PLACEHOLDER = "<project>"
DEPENDENCIES_FILE = "dependencies.json"
PARTIAL_PREFIX = "profile_"
WORKERINPUT_TOKEN = "smartpar_profile_token"

# A module whose snapshot takes longer than this switches to cheap mode.
SNAPSHOT_BUDGET_S = 0.05
REPR_LIMIT = 4096
# Default of the ini option ``smartpar_ignore_state`` (fnmatch patterns over
# state keys): variables maintained by pytest, xdist and pytest-cov themselves.
DEFAULT_IGNORE_STATE = (
    "env:PYTEST_CURRENT_TEST",
    "env:PYTEST_XDIST_WORKER",
    "env:PYTEST_XDIST_WORKER_COUNT",
    "env:COV_CORE_*",
)
# Category prefix for a grown functools cache (filled, not reconfigured).
CACHE_GROWTH = "cache_growth:"

# --- file access via audit hook --------------------------------------------------

# Event names verified against the CPython audit events table and by calling
# the functions on 3.12/3.14: os.unlink raises "os.remove", os.replace raises
# "os.rename". "os.unlink"/"os.replace" are kept in case a version adds them.
#   open             (path | fd, mode | None, flags)
#   os.remove        (path, dir_fd)
#   os.rename        (src, dst, src_dir_fd, dst_dir_fd)
#   os.mkdir         (path, mode, dir_fd)
#   os.rmdir         (path, dir_fd)
#   shutil.rmtree    (path, dir_fd)
#   shutil.copyfile  (src, dst)
#   shutil.move      (src, dst)
# os.* events pass dir_fd=-1 when no dir_fd was given (shutil.rmtree: None).
_EVENTS = frozenset(
    {
        "open",
        "os.remove",
        "os.unlink",
        "os.rename",
        "os.replace",
        "os.mkdir",
        "os.rmdir",
        "shutil.rmtree",
        "shutil.copyfile",
        "shutil.move",
    }
)
_NO_DIR_FD = (None, -1)
# Indexes of the recorder's sets, in the order of models.ACCESS_KINDS.
READ, WRITE, WRITE_TREE, CREATE_DIR = 0, 1, 2, 3
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_WRITE_MODE_CHARS = frozenset("wax+")


def is_write_open(mode: str | None, flags: int) -> bool:
    if mode is not None:
        return not _WRITE_MODE_CHARS.isdisjoint(mode)
    return bool(flags & _WRITE_FLAGS)


@dataclass
class AuditStats:
    errors: int = 0  # exceptions swallowed inside the hook
    fd_events: int = 0  # operations on an already open descriptor
    unresolved_dir_fd: int = 0  # relative path + dir_fd, cannot be resolved
    ignored: int = 0  # filtered out (stdlib, site-packages, tmp, /dev ...)
    existing_dir_mkdir: int = 0  # mkdir of a directory that already exists (still a creation)
    directory_opens: int = 0  # `open` of an existing directory: names no file


class FileAccessRecorder:
    """Collects R/W sets of the test in ``current``. Must never raise."""

    def __init__(self, ignore_dirs: Iterable[str | Path] = ()) -> None:
        self.current: str | None = None
        # nodeid -> (reads, writes, subtree writes, created directories)
        self.access: dict[str, tuple[set[str], set[str], set[str], set[str]]] = {}
        self.stats = AuditStats()
        self.set_ignore_dirs(ignore_dirs)

    def set_ignore_dirs(self, dirs: Iterable[str | Path]) -> None:
        normalized = {os.path.normpath(os.fspath(d)) for d in dirs if d}
        self._ignore_exact = frozenset(normalized)
        self._ignore_prefixes = tuple(d.rstrip(os.sep) + os.sep for d in normalized)

    def is_ignored(self, path: str) -> bool:
        sep = os.sep
        return (
            path.startswith(self._ignore_prefixes)
            or path in self._ignore_exact
            or path.endswith((".py", ".pyc"))
            or f"{sep}__pycache__" in path
            or f"{sep}site-packages{sep}" in path
            or path.endswith(f"{sep}site-packages")
        )

    def handle(self, event: str, args: tuple[Any, ...]) -> None:
        """Audit events arrive *before* the operation, so ``isdir`` sees the old state."""
        if event == "open":
            path = self._resolve(args[0], None)
            if path is not None and os.path.isdir(path):
                # A directory names no file that a test reads or writes: an
                # O_TMPFILE open creates an anonymous file, and since Python
                # 3.12 NamedTemporaryFile passes its directory to `open` with
                # an opener that creates a uniquely named file (the event shows
                # e.g. open('/tmp', 'w+')). Recording it as a write would make
                # every two tests with temporary files conflict.
                self.stats.directory_opens += 1
            else:
                self._add(path, WRITE if is_write_open(args[1], args[2]) else READ)
        elif event in ("os.remove", "os.unlink"):
            self._add(self._resolve(args[0], args[1]), WRITE)
        elif event in ("os.rmdir", "shutil.rmtree"):
            self._add(self._resolve(args[0], args[1]), WRITE_TREE)
        elif event in ("os.rename", "os.replace", "shutil.move"):
            src_fd, dst_fd = (args[2], args[3]) if len(args) > 2 else (None, None)
            src, dst = self._resolve(args[0], src_fd), self._resolve(args[1], dst_fd)
            # Moving a directory rewrites both subtrees.
            kind = WRITE_TREE if src is not None and os.path.isdir(src) else WRITE
            self._add(src, kind)
            self._add(dst, kind)
        elif event == "os.mkdir":
            path = self._resolve(args[0], args[2])
            if path is not None and os.path.isdir(path):
                self.stats.existing_dir_mkdir += 1  # makedirs(exist_ok=True) etc.
            # A creation whether or not the directory already exists, so the
            # result does not depend on files left over by earlier runs.
            self._add(path, CREATE_DIR)
        elif event == "shutil.copyfile":
            self._add(self._resolve(args[0], None), READ)
            self._add(self._resolve(args[1], None), WRITE)

    def _resolve(self, raw: Any, dir_fd: Any) -> str | None:
        """Absolute normalized path, or None if it cannot or need not be recorded."""
        if isinstance(raw, int):
            self.stats.fd_events += 1
            return None
        path = os.fsdecode(raw)
        if not os.path.isabs(path):
            if dir_fd not in _NO_DIR_FD:
                self.stats.unresolved_dir_fd += 1
                return None
            path = os.path.join(os.getcwd(), path)
        path = os.path.normpath(path)
        if self.is_ignored(path):
            self.stats.ignored += 1
            return None
        return path

    def _add(self, path: str | None, kind: int) -> None:
        if path is None:
            return
        assert self.current is not None
        sets = self.access.setdefault(self.current, (set(), set(), set(), set()))
        sets[kind].add(path)


# The audit hook cannot be removed, so it is installed at most once per
# process and dispatches to the currently active recorder (if any).
_active: FileAccessRecorder | None = None
_hook_installed = False


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    recorder = _active
    if recorder is None or recorder.current is None or event not in _EVENTS:
        return
    try:
        recorder.handle(event, args)
    except Exception:
        recorder.stats.errors += 1


def activate(recorder: FileAccessRecorder | None) -> None:
    global _active, _hook_installed
    if recorder is not None and not _hook_installed:
        sys.addaudithook(_audit_hook)
        _hook_installed = True
    _active = recorder


# --- state snapshots --------------------------------------------------------------


def _cheap_fingerprint(value: Any) -> tuple[Any, ...]:
    try:
        size = len(value)
    except Exception:
        size = None
    return ("c", type(value).__qualname__, id(value), size)


def _leaf_fingerprint(value: Any) -> tuple[Any, ...]:
    try:
        return ("p", hash(pickle.dumps(value, protocol=5)))
    except Exception:
        pass
    try:
        return ("r", repr(value)[:REPR_LIMIT])
    except Exception:
        return _cheap_fingerprint(value)


def fingerprint(value: Any, cheap: bool = False) -> tuple[Any, ...]:
    """pickle hash; for unpicklable objects with ``__dict__`` a shallow
    fingerprint of ``vars(obj)`` (a default repr would hide changes of a
    singleton instance's contents); otherwise repr, otherwise id+type."""
    if cheap:
        return _cheap_fingerprint(value)
    try:
        return ("p", hash(pickle.dumps(value, protocol=5)))
    except Exception:
        pass
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and not isinstance(value, (type, ModuleType)):
        try:
            items = sorted(attrs.items(), key=lambda kv: str(kv[0]))
            return (
                "d",
                type(value).__qualname__,
                tuple((str(k), _leaf_fingerprint(v)) for k, v in items),
            )
        except Exception:
            pass
    try:
        return ("r", repr(value)[:REPR_LIMIT])
    except Exception:
        return _cheap_fingerprint(value)


def _is_routine(value: Any) -> bool:
    return inspect.isroutine(value) or isinstance(
        value, (staticmethod, classmethod, property, functools.partial)
    )


def module_fingerprints(module: ModuleType, cheap: bool) -> dict[str, tuple[Any, ...]]:
    """Fingerprints of a module's data attributes (and its own classes' data attributes)."""
    out: dict[str, tuple[Any, ...]] = {}
    prefix = module.__name__
    for name, value in list(vars(module).items()):
        if name.startswith("__") and name.endswith("__"):
            continue
        if isinstance(value, ModuleType):
            continue
        cache_info = getattr(value, "cache_info", None)
        if callable(value) and callable(cache_info):
            with contextlib.suppress(Exception):
                out[f"{prefix}.{name}"] = ("cache", cache_info().currsize)
            continue
        if isinstance(value, type):
            # Class attributes of classes defined here, e.g. a singleton's
            # ``_instance``; methods and descriptors are code, not state.
            if value.__module__ != prefix:
                continue
            for attr, attr_value in list(vars(value).items()):
                if attr.startswith("__") and attr.endswith("__"):
                    continue
                if isinstance(attr_value, type) or _is_routine(attr_value):
                    continue
                out[f"{prefix}.{name}.{attr}"] = fingerprint(attr_value, cheap)
            continue
        if _is_routine(value):
            continue
        out[f"{prefix}.{name}"] = fingerprint(value, cheap)
    return out


@dataclass
class Snapshot:
    modules: dict[str, dict[str, tuple[Any, ...]]]
    process: dict[str, Any]
    slow_modules: dict[str, float] = field(default_factory=dict)


def _changed_key(key: str, old: Any, new: Any) -> str:
    """A grown functools cache gets its own category; a cleared one is a real change."""
    if (
        isinstance(old, tuple)
        and isinstance(new, tuple)
        and old[:1] == new[:1] == ("cache",)
        and new[1] > old[1]
    ):
        return CACHE_GROWTH + key
    return key


def diff(before: Snapshot, after: Snapshot, ignore: Iterable[str] = ()) -> list[str]:
    """Keys whose value differs, minus keys matching an ``ignore`` pattern.

    Modules imported during the test are not a change themselves (they become
    the baseline for later tests); side effects of their import on modules
    already loaded or on the environment are.
    """
    patterns = tuple(ignore)
    changed: set[str] = set()
    pairs = [
        (before.modules[n], after.modules[n]) for n in before.modules.keys() & after.modules.keys()
    ]
    pairs.append((before.process, after.process))
    for a, b in pairs:
        for key in a.keys() | b.keys():
            old, new = a.get(key), b.get(key)
            if old != new and not any(fnmatch.fnmatchcase(key, p) for p in patterns):
                changed.add(_changed_key(key, old, new))
    return sorted(changed)


class StateSnapshotter:
    def __init__(self, root: Path, ignore_dirs: Iterable[Path] = ()) -> None:
        self.root = Path(os.path.normpath(root))
        self.ignore_dirs = [Path(os.path.normpath(d)) for d in ignore_dirs]
        self.test_dirs: list[Path] = []
        self.cheap: set[str] = set()
        self._decision: dict[tuple[str, str], bool] = {}

    def set_test_dirs(self, dirs: Iterable[Path]) -> None:
        self.test_dirs = sorted({Path(os.path.normpath(d)) for d in dirs})
        self._decision.clear()

    def _is_project_module(self, name: str, file: str) -> bool:
        key = (name, file)
        decision = self._decision.get(key)
        if decision is None:
            path = Path(os.path.normpath(os.path.abspath(file)))
            decision = (
                path.is_relative_to(self.root)
                and "site-packages" not in path.parts
                and not any(path.is_relative_to(d) for d in self.ignore_dirs)
                and not any(path.is_relative_to(d) for d in self.test_dirs)
            )
            self._decision[key] = decision
        return decision

    def project_modules(self) -> list[ModuleType]:
        modules = []
        for name, module in list(sys.modules.items()):
            file = getattr(module, "__file__", None)
            if (
                isinstance(module, ModuleType)
                and isinstance(file, str)
                and self._is_project_module(name, file)
            ):
                modules.append(module)
        return modules

    def take(self) -> Snapshot:
        modules: dict[str, dict[str, tuple[Any, ...]]] = {}
        slow: dict[str, float] = {}
        for module in self.project_modules():
            t0 = time.perf_counter()
            modules[module.__name__] = module_fingerprints(module, module.__name__ in self.cheap)
            elapsed = time.perf_counter() - t0
            if elapsed > SNAPSHOT_BUDGET_S and module.__name__ not in self.cheap:
                slow[module.__name__] = elapsed
        process: dict[str, Any] = {f"env:{k}": v for k, v in os.environ.items()}
        process["cwd"] = os.getcwd()
        process["sys.path"] = tuple(sys.path)
        process["logging.root.level"] = logging.root.level
        process["logging.root.handlers"] = len(logging.root.handlers)
        return Snapshot(modules, process, slow)

    def promote_slow(self, *snapshots: Snapshot) -> None:
        """Switch slow modules to cheap mode (between tests, never within a pair)."""
        for snap in snapshots:
            for name, elapsed in snap.slow_modules.items():
                if name not in self.cheap:
                    log.warning(
                        "smartpar: snapshot of module %s took %.0f ms (> %.0f ms); "
                        "switching it to cheap mode (id+type+len)",
                        name,
                        elapsed * 1000,
                        SNAPSHOT_BUDGET_S * 1000,
                    )
                    self.cheap.add(name)


# --- results -----------------------------------------------------------------------


def to_resource(path: str, root: Path) -> str:
    """``<project>/rel/path`` inside the project root, the absolute path otherwise."""
    root_str = os.path.normpath(root)
    if path == root_str:
        return PROJECT_PLACEHOLDER
    if path.startswith(root_str.rstrip(os.sep) + os.sep):
        rel = os.path.relpath(path, root_str)
        return f"{PROJECT_PLACEHOLDER}/{Path(rel).as_posix()}"
    return path


@dataclass
class ProfileData:
    """Profile of one process (absolute paths); merged across xdist workers."""

    file_access: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    state_changes: dict[str, list[str]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    cheap_modules: list[str] = field(default_factory=list)
    collection_index: dict[str, int] = field(default_factory=dict)

    def to_dict(self, worker: str) -> dict[str, Any]:
        return {
            **envelope(),
            "worker": worker,
            "file_access": self.file_access,
            "state_changes": self.state_changes,
            "stats": self.stats,
            "cheap_modules": sorted(self.cheap_modules),
            "collection_index": self.collection_index,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProfileData:
        data = check_version(data, "partial profile")
        return cls(
            file_access=data["file_access"],
            state_changes=data["state_changes"],
            stats=data["stats"],
            cheap_modules=data["cheap_modules"],
            collection_index=data["collection_index"],
        )

    def update(self, other: ProfileData) -> None:
        self.file_access.update(other.file_access)
        self.state_changes.update(other.state_changes)
        for key, value in other.stats.items():
            self.stats[key] = self.stats.get(key, 0) + value
        self.cheap_modules = sorted(set(self.cheap_modules) | set(other.cheap_modules))
        # Every xdist worker collects the same items in the same order.
        self.collection_index.update(other.collection_index)


def access_from_raw(
    raw: dict[str, dict[str, list[str]]], root: Path, source: str
) -> dict[str, FileAccess]:
    """{nodeid: {"R"|"W"|"W_tree"|"C": [absolute paths]}} -> FileAccess with resources."""
    out: dict[str, FileAccess] = {}
    for nodeid, kinds in raw.items():
        access = out.setdefault(nodeid, FileAccess())
        for kind in ACCESS_KINDS:
            for path in kinds.get(kind, []):
                access.add(kind, to_resource(path, root), source)
    return out


def merge_access(target: dict[str, FileAccess], extra: dict[str, FileAccess]) -> None:
    """Add the accesses of ``extra`` (with their sources) to ``target``."""
    for nodeid, access in extra.items():
        dest = target.setdefault(nodeid, FileAccess())
        for kind in ACCESS_KINDS:
            for resource in getattr(access, kind):
                dest.add(kind, resource, access.source(kind, resource))


def build_dependencies(
    profile: ProfileData,
    root: Path,
    flaky: Flaky | None,
    history: History,
    extra_stats: dict[str, Any],
    previous: Dependencies | None = None,
) -> Dependencies:
    """Fresh file access and state changes plus the derived graph.

    Accesses seen only in the detector's runs (source "detect") are kept from
    ``previous``: one profiling run in one order does not see conditional
    accesses that other orders trigger. Order edges mirror the confirmed pairs
    of flaky.json (none without it). Groups here are informational; the plan
    recomputes them for the collected tests.
    """
    file_access = access_from_raw(profile.file_access, root, SOURCE_PROFILE)
    if previous is not None:
        merge_access(
            file_access,
            {n: a.only({SOURCE_DETECT}) for n, a in previous.file_access.items()},
        )
        file_access = {
            n: a for n, a in file_access.items() if any(getattr(a, k) for k in ACCESS_KINDS)
        }
    tests = set(profile.file_access) | set(profile.state_changes)
    durations = estimate_durations(tests, history).durations
    edges = flaky.order_edges() if flaky else []
    graph = build_graph(
        tests,
        file_access,
        edges,
        durations,
        isolate=flaky.nod_tests() if flaky else (),
        collection_index=profile.collection_index,
    )
    return Dependencies(
        file_access=file_access,
        state_changes={k: sorted(v) for k, v in profile.state_changes.items() if v},
        conflicts=graph.conflicts,
        order_edges=graph.order_edges,
        groups=graph.groups,
        collection_index=profile.collection_index,
        profile_stats={
            **extra_stats,
            **profile.stats,
            "cheap_modules": sorted(profile.cheap_modules),
        },
    )


def load_dependencies(smartpar_dir: Path) -> Dependencies | None:
    path = smartpar_dir / DEPENDENCIES_FILE
    try:
        return Dependencies.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, SchemaError) as exc:
        log.warning("smartpar: ignoring unreadable %s: %s", path, exc)
        return None


# --- pytest integration -------------------------------------------------------------


def venv_dirs() -> list[Path]:
    return [Path(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix} if p]


def ignore_dirs(config: pytest.Config) -> list[str]:
    """Directories whose files are not recorded (interpreter, tmp_path, /dev ...)."""
    dirs = [str(p) for p in venv_dirs()]
    dirs += [sys.base_exec_prefix, "/dev", "/proc", "/sys"]
    factory = getattr(config, "_tmp_path_factory", None)
    if factory is not None:
        with contextlib.suppress(Exception):
            dirs.append(str(factory.getbasetemp()))
    return dirs


class AuditOnlyPlugin:
    """File access recording without state snapshots (detector child runs)."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.recorder = FileAccessRecorder()

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.recorder.set_ignore_dirs(ignore_dirs(self.config))
        activate(self.recorder)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(
        self, item: pytest.Item, nextitem: pytest.Item | None
    ) -> Generator[None, object, object]:
        self.recorder.current = item.nodeid
        try:
            return (yield)
        finally:
            self.recorder.current = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        activate(None)

    def resources(self) -> dict[str, dict[str, list[str]]]:
        """Accesses as resources of this session's root (``<project>/...``), so a
        run in a project copy reports the same names as the project itself."""
        root = Path(self.config.rootpath)
        return {
            nodeid: {
                kind: sorted({to_resource(p, root) for p in paths})
                for kind, paths in zip(ACCESS_KINDS, sets, strict=True)
            }
            for nodeid, sets in self.recorder.access.items()
        }


class ProfilerPlugin:
    """Active in ``--smartpar-profile`` mode in every pytest process.

    The process that runs tests (the only one in a serial session, each worker
    under xdist) records file access and state changes. Workers write
    ``profile_<token>_<worker>.json``; the controller merges them.
    """

    def __init__(self, config: pytest.Config, smartpar_dir: Path, xdist_controller: bool) -> None:
        self.config = config
        self.smartpar_dir = smartpar_dir
        self.root = Path(config.rootpath)
        self.is_worker = hasattr(config, "workerinput")
        self.is_xdist_controller = xdist_controller
        self.runs_tests = not self.is_xdist_controller
        self.token = (
            config.workerinput[WORKERINPUT_TOKEN] if self.is_worker else uuid.uuid4().hex[:12]
        )
        self.recorder = FileAccessRecorder()
        self.snapshotter = StateSnapshotter(self.root, ignore_dirs=self._venv_dirs())
        self.state_changes: dict[str, list[str]] = {}
        self.collection_index: dict[str, int] = {}
        self.ignore_state: list[str] = list(config.getini("smartpar_ignore_state"))

    @staticmethod
    def _venv_dirs() -> list[Path]:
        return venv_dirs()

    def _ignore_dirs(self) -> list[str]:
        return ignore_dirs(self.config)

    # Called from pytest_configure_node, see plugin._XdistProfileHooks.
    def configure_node(self, node: Any) -> None:
        node.workerinput[WORKERINPUT_TOKEN] = self.token

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        if self.runs_tests:
            self.recorder.set_ignore_dirs(self._ignore_dirs())
            activate(self.recorder)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collection_index = {item.nodeid: i for i, item in enumerate(session.items)}
        self.snapshotter.set_test_dirs(item.path.parent for item in session.items)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(
        self, item: pytest.Item, nextitem: pytest.Item | None
    ) -> Generator[None, object, object]:
        before = self.snapshotter.take()
        self.recorder.current = item.nodeid
        try:
            return (yield)
        finally:
            self.recorder.current = None
            after = self.snapshotter.take()
            self.state_changes[item.nodeid] = diff(before, after, self.ignore_state)
            self.snapshotter.promote_slow(before, after)

    def _own_profile(self) -> ProfileData:
        return ProfileData(
            file_access={
                nodeid: {
                    kind: sorted(paths) for kind, paths in zip(ACCESS_KINDS, sets, strict=True)
                }
                for nodeid, sets in self.recorder.access.items()
            },
            state_changes=self.state_changes,
            stats=vars(self.recorder.stats).copy(),
            cheap_modules=sorted(self.snapshotter.cheap),
            collection_index=self.collection_index,
        )

    def _partial_path(self, worker: str) -> Path:
        return self.smartpar_dir / f"{PARTIAL_PREFIX}{self.token}_{worker}.json"

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.runs_tests:
            activate(None)
        if self.is_worker:
            worker = self.config.workerinput["workerid"]
            atomic_write_json(self._partial_path(worker), self._own_profile().to_dict(worker))
            return
        if self.is_xdist_controller:
            profile = ProfileData()
            for path in sorted(self.smartpar_dir.glob(f"{PARTIAL_PREFIX}{self.token}_*.json")):
                profile.update(ProfileData.from_dict(json.loads(path.read_text("utf-8"))))
                path.unlink()
            mode = "xdist"
        else:
            profile = self._own_profile()
            mode = "serial"
        if not profile.file_access and not profile.state_changes:
            return  # nothing ran
        try:
            flaky = load_flaky(self.smartpar_dir)
            hist = load_history(self.smartpar_dir)
        except SchemaError as exc:
            log.warning("smartpar: ignoring unreadable data file: %s", exc)
            flaky, hist = None, History()
        deps = build_dependencies(
            profile, self.root, flaky, hist, {"mode": mode}, load_dependencies(self.smartpar_dir)
        )
        atomic_write_json(self.smartpar_dir / DEPENDENCIES_FILE, deps.to_dict())
