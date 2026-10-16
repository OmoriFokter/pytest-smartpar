"""Phase 3: audit-hook file access, state snapshots and the --smartpar-profile mode."""

from __future__ import annotations

import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest

from smartpar import profiler
from smartpar.profiler import FileAccessRecorder, StateSnapshotter

# --- write detection ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "flags", "write"),
    [
        ("r", os.O_RDONLY, False),
        ("rb", os.O_RDONLY, False),
        ("w", 0, True),
        ("a", 0, True),
        ("x", 0, True),
        ("r+", 0, True),
        (None, os.O_RDONLY, False),
        (None, os.O_WRONLY, True),
        (None, os.O_RDWR, True),
        (None, os.O_RDONLY | os.O_APPEND, True),
        (None, os.O_RDONLY | os.O_CREAT, True),
        (None, os.O_RDONLY | os.O_TRUNC, True),
    ],
)
def test_is_write_open(mode: str | None, flags: int, write: bool) -> None:
    assert profiler.is_write_open(mode, flags) is write


# --- recorder ----------------------------------------------------------------------


@pytest.fixture
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FileAccessRecorder:
    monkeypatch.chdir(tmp_path)
    rec = FileAccessRecorder(ignore_dirs=[tmp_path / "ignored"])
    rec.current = "t::x"
    return rec


NONE: tuple[set[str], set[str], set[str], set[str]] = (set(), set(), set(), set())


def rw(rec: FileAccessRecorder) -> tuple[set[str], set[str], set[str], set[str]]:
    """(reads, writes, subtree writes, created directories) of the current test."""
    return rec.access.get("t::x", NONE)


def test_recorder_events_with_real_signatures(recorder: FileAccessRecorder, tmp_path: Path) -> None:
    p = str(tmp_path)
    recorder.handle("open", (f"{p}/r.txt", "r", os.O_RDONLY))
    recorder.handle("open", (f"{p}/w.txt", None, os.O_WRONLY | os.O_CREAT))
    recorder.handle("os.remove", (f"{p}/rm.txt", -1))
    recorder.handle("os.rename", (f"{p}/src", f"{p}/dst", -1, -1))  # src is not a dir
    recorder.handle("os.mkdir", (f"{p}/newdir", 0o777, -1))  # does not exist yet
    recorder.handle("os.rmdir", (f"{p}/rmdir", -1))
    recorder.handle("shutil.rmtree", (f"{p}/tree", None))
    recorder.handle("shutil.copyfile", (f"{p}/cp_src", f"{p}/cp_dst"))
    recorder.handle("shutil.move", (f"{p}/mv_src", f"{p}/mv_dst"))
    reads, writes, trees, created = rw(recorder)
    assert reads == {f"{p}/r.txt", f"{p}/cp_src"}
    assert writes == {
        f"{p}/{name}" for name in ("w.txt", "rm.txt", "src", "dst", "cp_dst", "mv_src", "mv_dst")
    }
    assert trees == {f"{p}/rmdir", f"{p}/tree"}
    assert created == {f"{p}/newdir"}


def test_mkdir_is_a_creation_whether_or_not_the_directory_exists(
    recorder: FileAccessRecorder, tmp_path: Path
) -> None:
    # So the result does not depend on directories left over by earlier runs.
    (tmp_path / "exists").mkdir()
    recorder.handle("os.mkdir", (str(tmp_path / "exists"), 0o777, -1))
    recorder.handle("os.mkdir", (str(tmp_path / "new"), 0o777, -1))
    assert rw(recorder) == (set(), set(), set(), {str(tmp_path / "exists"), str(tmp_path / "new")})
    assert recorder.stats.existing_dir_mkdir == 1


@pytest.mark.parametrize(
    ("mode", "flags"),
    [("w+", os.O_RDWR | os.O_CREAT), (None, os.O_RDWR), (None, os.O_RDONLY)],
)
def test_open_of_a_directory_is_not_an_access(
    recorder: FileAccessRecorder, tmp_path: Path, mode: str | None, flags: int
) -> None:
    recorder.handle("open", (str(tmp_path), mode, flags))
    assert rw(recorder) == NONE
    assert recorder.stats.directory_opens == 1


@pytest.mark.parametrize("event", ["os.rename", "os.replace", "shutil.move"])
def test_moving_a_directory_writes_both_subtrees(
    recorder: FileAccessRecorder, tmp_path: Path, event: str
) -> None:
    (tmp_path / "srcdir").mkdir()
    args = (str(tmp_path / "srcdir"), str(tmp_path / "dstdir"))
    recorder.handle(event, (*args, -1, -1) if event != "shutil.move" else args)
    moved = {str(tmp_path / "srcdir"), str(tmp_path / "dstdir")}
    assert rw(recorder) == (set(), set(), moved, set())


def test_recorder_relative_bytes_and_descriptors(
    recorder: FileAccessRecorder, tmp_path: Path
) -> None:
    recorder.handle("open", ("rel/a.txt", "w", 0))  # resolved against cwd
    recorder.handle("os.remove", (b"b.txt", -1))  # bytes path, dir_fd=-1 means "none"
    recorder.handle("os.remove", ("c.txt", 7))  # relative to an unknown dir_fd
    recorder.handle("open", (5, "r", 0))  # already open descriptor
    _reads, writes, _trees, _created = rw(recorder)
    assert writes == {str(tmp_path / "rel" / "a.txt"), str(tmp_path / "b.txt")}
    assert recorder.stats.unresolved_dir_fd == 1
    assert recorder.stats.fd_events == 1


@pytest.mark.parametrize(
    "path",
    [
        "{ignored}/x.txt",
        "{ignored}",
        "/dev/null",
        "/proc/self/status",
        "/somewhere/site-packages/pkg/data.json",
        "/somewhere/mod.py",
        "/somewhere/__pycache__/mod.cpython-312.pyc",
    ],
)
def test_recorder_ignores(recorder: FileAccessRecorder, tmp_path: Path, path: str) -> None:
    recorder.set_ignore_dirs([tmp_path / "ignored", "/dev", "/proc"])
    recorder.handle("open", (path.format(ignored=tmp_path / "ignored"), "w", 0))
    assert rw(recorder) == NONE
    assert recorder.stats.ignored == 1


def test_ignore_prefix_respects_directory_boundary(recorder: FileAccessRecorder) -> None:
    recorder.set_ignore_dirs(["/usr"])
    recorder.handle("open", ("/usrdata/file", "r", 0))
    assert rw(recorder)[0] == {"/usrdata/file"}


def test_audit_hook_never_raises_and_counts(recorder: FileAccessRecorder) -> None:
    profiler.activate(recorder)
    try:
        profiler._audit_hook("open", ())  # malformed args
        profiler._audit_hook("os.remove", (object(), -1))
    finally:
        profiler.activate(None)
    assert recorder.stats.errors == 2


def test_audit_hook_records_real_open(recorder: FileAccessRecorder, tmp_path: Path) -> None:
    profiler.activate(recorder)
    try:
        (tmp_path / "real.txt").write_text("x")
        recorder.current = None
        (tmp_path / "outside_test.txt").write_text("x")  # no current test: ignored
    finally:
        profiler.activate(None)
    recorder.current = "t::x"
    assert rw(recorder)[1] == {str(tmp_path / "real.txt")}


# --- fingerprints and snapshots ----------------------------------------------------


class Unpicklable:
    def __init__(self, value: int) -> None:
        self.value = value
        self.lock = __import__("threading").Lock()

    def __repr__(self) -> str:
        return f"Unpicklable({self.value})"


def test_fingerprint_fallbacks() -> None:
    assert profiler.fingerprint({"a": 1}) == profiler.fingerprint({"a": 1})
    assert profiler.fingerprint({"a": 1}) != profiler.fingerprint({"a": 2})
    assert profiler.fingerprint(Unpicklable(1))[0] == "d"  # shallow vars(obj)
    assert profiler.fingerprint(Unpicklable(1)) != profiler.fingerprint(Unpicklable(2))
    same = Unpicklable(1)
    assert profiler.fingerprint(same) == profiler.fingerprint(same)
    assert profiler.fingerprint(__import__("threading").Lock())[0] == "r"  # no __dict__
    assert profiler.fingerprint("x" * 10_000)[0] == "p"
    cheap = profiler.fingerprint([1, 2, 3], cheap=True)
    assert cheap[0] == "c" and cheap[1] == "list" and cheap[3] == 3


def make_module(name: str, file: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(file)
    exec(
        """
import functools, os
from pathlib import Path          # class from another module: skipped
SETTINGS = {"currency": "UAH"}
COUNTER = 0
def helper(): pass
@functools.lru_cache
def rate(c): return len(c)
class Registry:
    _instance = None
    def method(self): pass
    @property
    def prop(self): return 1
class LockedRegistry:
    def __init__(self):
        import threading
        self.lock = threading.Lock()  # makes the instance unpicklable
        self.items = []
        self.currency = "UAH"
registry = LockedRegistry()
""",
        module.__dict__,
    )
    return module


def test_module_fingerprints_selects_data(tmp_path: Path) -> None:
    module = make_module("fakepkg.state", tmp_path / "state.py")
    keys = set(profiler.module_fingerprints(module, cheap=False))
    assert keys == {
        "fakepkg.state.SETTINGS",
        "fakepkg.state.COUNTER",
        "fakepkg.state.rate",
        "fakepkg.state.Registry._instance",
        "fakepkg.state.registry",
    }


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, types.ModuleType]:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    module = make_module("fakepkg.state", tmp_path / "pkg" / "state.py")
    monkeypatch.setitem(sys.modules, "fakepkg.state", module)
    monkeypatch.setitem(
        sys.modules, "fake_test_mod", make_module("fake_test_mod", tmp_path / "tests" / "t.py")
    )
    return tmp_path, module


def test_snapshot_diff_detects_unrestored_changes(
    project: tuple[Path, types.ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, module = project
    snap = StateSnapshotter(root)
    snap.set_test_dirs([root / "tests"])
    before = snap.take()
    assert "fakepkg.state" in before.modules
    assert "fake_test_mod" not in before.modules  # test directories are excluded
    module.SETTINGS["currency"] = "USD"
    module.rate("abc")
    module.Registry._instance = object()
    monkeypatch.setenv("SMARTPAR_TEST_VAR", "1")
    monkeypatch.chdir(root)
    after = snap.take()
    assert profiler.diff(before, after) == [
        "cache_growth:fakepkg.state.rate",
        "cwd",
        "env:SMARTPAR_TEST_VAR",
        "fakepkg.state.Registry._instance",
        "fakepkg.state.SETTINGS",
    ]


def test_unpicklable_singleton_instance_change_is_detected(
    project: tuple[Path, types.ModuleType],
) -> None:
    root, module = project
    snap = StateSnapshotter(root)
    before = snap.take()
    module.registry.currency = "USD"  # registry = LockedRegistry(): has a Lock
    assert profiler.diff(before, snap.take()) == ["fakepkg.state.registry"]
    before = snap.take()
    module.registry.items.append("x")  # shallow vars(): a picklable attribute's content
    assert profiler.diff(before, snap.take()) == ["fakepkg.state.registry"]


def test_cache_clear_is_a_plain_state_change(project: tuple[Path, types.ModuleType]) -> None:
    root, module = project
    module.rate("abc")
    snap = StateSnapshotter(root)
    before = snap.take()
    module.rate.cache_clear()
    assert profiler.diff(before, snap.take()) == ["fakepkg.state.rate"]


def test_diff_ignore_patterns(
    project: tuple[Path, types.ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, module = project
    snap = StateSnapshotter(root)
    before = snap.take()
    monkeypatch.setenv("COV_CORE_SOURCE", "x")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw9")
    monkeypatch.setenv("APP_KEEP", "1")
    module.COUNTER = 5
    after = snap.take()
    assert profiler.diff(before, after, profiler.DEFAULT_IGNORE_STATE) == [
        "env:APP_KEEP",
        "fakepkg.state.COUNTER",
    ]
    assert profiler.diff(before, after, ["env:*", "fakepkg.*"]) == []


def test_new_module_becomes_baseline_and_import_side_effects_are_recorded(
    project: tuple[Path, types.ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, module = project
    snap = StateSnapshotter(root)
    monkeypatch.delenv("SMARTPAR_IMPORT_SIDE_EFFECT", raising=False)

    # Test A imports a new project module whose import mutates an already
    # loaded module and the environment.
    before_a = snap.take()
    late = types.ModuleType("fakepkg.late")
    late.__file__ = str(root / "pkg" / "late.py")
    late.FLAG = "initial"
    module.SETTINGS["currency"] = "EUR"  # side effect of importing fakepkg.late
    os.environ["SMARTPAR_IMPORT_SIDE_EFFECT"] = "1"
    monkeypatch.setitem(sys.modules, "fakepkg.late", late)
    after_a = snap.take()
    assert profiler.diff(before_a, after_a) == [
        "env:SMARTPAR_IMPORT_SIDE_EFFECT",
        "fakepkg.state.SETTINGS",
    ]  # the new module's own attributes are not a change

    # Test B: the new module is now part of the baseline.
    before_b = snap.take()
    assert "fakepkg.late" in before_b.modules
    late.FLAG = "changed"
    assert profiler.diff(before_b, snap.take()) == ["fakepkg.late.FLAG"]


def test_snapshot_restored_change_and_new_module_are_not_changes(
    project: tuple[Path, types.ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, module = project
    snap = StateSnapshotter(root)
    before = snap.take()
    module.SETTINGS["currency"] = "USD"
    module.SETTINGS["currency"] = "UAH"
    monkeypatch.setitem(sys.modules, "fakepkg.late", make_module("fakepkg.late", root / "late.py"))
    assert profiler.diff(before, snap.take()) == []


def test_snapshot_excludes_venv_dirs(project: tuple[Path, types.ModuleType]) -> None:
    root, _module = project
    snap = StateSnapshotter(root, ignore_dirs=[root / "pkg"])
    assert "fakepkg.state" not in snap.take().modules


def test_slow_module_switches_to_cheap_mode(
    project: tuple[Path, types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, module = project
    monkeypatch.setattr(profiler, "SNAPSHOT_BUDGET_S", -1.0)
    snap = StateSnapshotter(root)
    first = snap.take()
    assert first.modules["fakepkg.state"]["fakepkg.state.SETTINGS"][0] == "p"
    with caplog.at_level(logging.WARNING, logger="smartpar.profiler"):
        snap.promote_slow(first)
    assert "fakepkg.state" in snap.cheap
    assert "cheap mode" in caplog.text
    second = snap.take()
    assert second.modules["fakepkg.state"]["fakepkg.state.SETTINGS"][0] == "c"
    module.SETTINGS = {"replaced": True}  # rebinding is still visible in cheap mode
    assert profiler.diff(second, snap.take()) == ["fakepkg.state.SETTINGS"]


def test_to_resource(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    assert profiler.to_resource(str(root / "build" / "cfg.ini"), root) == "<project>/build/cfg.ini"
    assert profiler.to_resource(str(root), root) == "<project>"
    assert profiler.to_resource(str(tmp_path / "proj2" / "x"), root) == str(
        tmp_path / "proj2" / "x"
    )


def test_partial_profiles_merge() -> None:
    a = profiler.ProfileData({"t1": {"R": ["/a"], "W": []}}, {"t1": ["x"]}, {"errors": 1}, ["m"])
    b = profiler.ProfileData({"t2": {"R": [], "W": ["/b"]}}, {"t2": []}, {"errors": 2}, ["n"])
    merged = profiler.ProfileData()
    merged.update(profiler.ProfileData.from_dict(json.loads(json.dumps(a.to_dict("gw0")))))
    merged.update(b)
    assert set(merged.file_access) == {"t1", "t2"}
    assert merged.stats == {"errors": 3}
    assert merged.cheap_modules == ["m", "n"]


# --- --smartpar-profile end to end (subprocess: fresh audit hook, real env) ---------

APP_STATE = """
import functools
SETTINGS = {"currency": "UAH"}
class Registry:
    _instance = None
@functools.lru_cache
def rate(c):
    return len(c)
"""

TESTS = """
import os, shutil, sys, time
import pytest
from app import state

def test_write():
    os.makedirs("build", exist_ok=True)
    with open("build/cfg.ini", "w") as fh:
        fh.write("x")

def test_read():
    if os.path.exists("build/cfg.ini"):
        open("build/cfg.ini").read()
    else:
        open("README.txt").read()

def test_copy_and_remove():
    shutil.copyfile("README.txt", "build_copy.txt")
    os.remove("build_copy.txt")

def test_tmp_path_ignored(tmp_path):
    (tmp_path / "f.txt").write_text("x")

def test_pollute_dict():
    state.SETTINGS["currency"] = "USD"

def test_singleton():
    state.Registry._instance = object()

def test_cache():
    state.rate("abc")

def test_env():
    os.environ["APP_CURRENCY"] = "USD"

def test_chdir():
    os.chdir("build") if os.path.isdir("build") else os.chdir("/")

def test_sys_path():
    sys.path.append("/nonexistent")

def test_monkeypatch_restores(monkeypatch):
    monkeypatch.setitem(state.SETTINGS, "currency", "EUR")
    monkeypatch.setenv("APP_CURRENCY", "EUR")
    monkeypatch.chdir("/")
    monkeypatch.setattr(state.Registry, "_instance", 1)

def test_hook_installed_only_in_profile_mode():
    from smartpar import profiler
    assert profiler._hook_installed is (os.environ["EXPECT_HOOK"] == "1")
"""


@pytest.fixture
def profiled_project(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.mkpydir("app")
    (pytester.path / "app" / "state.py").write_text(APP_STATE)
    (pytester.path / "README.txt").write_text("readme")
    tests = pytester.mkdir("tests")
    (tests / "test_app.py").write_text(TESTS)
    return pytester


def run_profile(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *args: str
) -> pytest.RunResult:
    monkeypatch.setenv("EXPECT_HOOK", "1")
    return pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider", *args)


def rw_only(access: dict) -> dict:
    """R/W/W_tree of a file_access entry, without its source map."""
    return {k: v for k, v in access.items() if k != "source"}


def load(pytester: pytest.Pytester, name: str) -> dict:
    return json.loads((pytester.path / ".smartpar" / name).read_text())


NODE = "tests/test_app.py::"


def test_profile_serial(profiled_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_profile(profiled_project, monkeypatch, "-p", "no:randomly")
    result.assert_outcomes(passed=12)

    deps = load(profiled_project, "dependencies.json")
    access = deps["file_access"]
    assert access[NODE + "test_write"]["W"] == ["<project>/build/cfg.ini"]
    assert access[NODE + "test_write"]["C"] == ["<project>/build"]
    assert rw_only(access[NODE + "test_read"]) == {
        "R": ["<project>/build/cfg.ini"],
        "W": [],
        "W_tree": [],
        "C": [],
    }
    assert rw_only(access[NODE + "test_copy_and_remove"]) == {
        "R": ["<project>/README.txt"],
        "W": ["<project>/build_copy.txt"],
        "W_tree": [],
        "C": [],
    }
    assert NODE + "test_tmp_path_ignored" not in access

    changes = deps["state_changes"]
    assert changes == {
        NODE + "test_pollute_dict": ["app.state.SETTINGS"],
        NODE + "test_singleton": ["app.state.Registry._instance"],
        NODE + "test_cache": ["cache_growth:app.state.rate"],
        NODE + "test_env": ["env:APP_CURRENCY"],
        NODE + "test_chdir": ["cwd"],
        NODE + "test_sys_path": ["sys.path"],
    }  # test_monkeypatch_restores is absent: restored state is not a change
    assert deps["profile_stats"]["errors"] == 0
    assert deps["profile_stats"]["mode"] == "serial"
    # Only file conflicts create edges; state changes alone do not.
    assert deps["conflicts"] == [
        {
            "a": NODE + "test_read",
            "b": NODE + "test_write",
            "kind": "R-W",
            "resource": "<project>/build/cfg.ini",
        }
    ]
    assert [g["tests"] for g in deps["groups"]] == [[NODE + "test_read", NODE + "test_write"]]

    hist = load(profiled_project, "history.json")
    assert len(hist["tests"]) == 12
    assert all(t["runs"] == 1 and t["last_outcome"] == "passed" for t in hist["tests"].values())
    merged = list((profiled_project.path / ".smartpar" / "runs" / "merged").glob("run_*.json"))
    assert len(merged) == 1
    run = json.loads(merged[0].read_text())
    assert run["mode"] == "profile" and run["workers"] == 1
    assert run["randomly_reorder_disabled"] is False  # randomly was disabled with -p


def test_profile_parallel_matches_serial(
    profiled_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make test_read independent of test_write's order for a fair comparison.
    (profiled_project.path / "build").mkdir()
    (profiled_project.path / "build" / "cfg.ini").write_text("x")
    run_profile(profiled_project, monkeypatch, "-p", "no:randomly").assert_outcomes(passed=12)
    serial = load(profiled_project, "dependencies.json")

    result = run_profile(profiled_project, monkeypatch, "-n", "2")
    result.assert_outcomes(passed=12)
    parallel = load(profiled_project, "dependencies.json")
    for key in ("file_access", "state_changes"):
        assert parallel[key] == serial[key]
    assert parallel["profile_stats"]["mode"] == "xdist"
    assert not list((profiled_project.path / ".smartpar").glob("profile_*.json"))
    runs = sorted((profiled_project.path / ".smartpar" / "runs" / "merged").glob("run_*.json"))
    run = json.loads(runs[-1].read_text())
    assert run["workers"] == 2
    assert {t["worker"] for t in run["tests"]} == {"gw0", "gw1"}
    assert run["randomly_reorder_disabled"] is True


def test_profile_mirrors_flaky_edges_and_drops_stale_derived_data(
    profiled_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    smartpar_dir = profiled_project.path / ".smartpar"
    smartpar_dir.mkdir()
    (smartpar_dir / "dependencies.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "file_access": {"old::test": {"R": ["x"], "W": []}},
                "state_changes": {},
                "conflicts": [{"a": "old", "b": "old2", "resource": "x", "kind": "W-R"}],
                "order_edges": [{"before": "old", "after": "old2", "reason": "stale"}],
                "groups": [],
            }
        )
    )
    victim, polluter = NODE + "test_read", NODE + "test_pollute_dict"
    (smartpar_dir / "flaky.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": 20,
                "seeds": [1],
                "od_victims": [{"victim": victim, "polluters": [polluter]}],
                "od_brittle": [],
                "nod": [],
                "unresolved": [],
            }
        )
    )
    run_profile(profiled_project, monkeypatch).assert_outcomes(passed=12)
    deps = load(profiled_project, "dependencies.json")
    assert deps["order_edges"] == [
        {"before": victim, "after": polluter, "reason": "victim-before-polluter"}
    ]
    assert "old::test" not in deps["file_access"]
    assert all(c["a"] != "old" for c in deps["conflicts"])
    (group,) = deps["groups"]
    assert group["reasons"] == ["file", "od"]
    assert group["order"].index(victim) < group["order"].index(polluter)


def test_no_profiling_without_flag(
    profiled_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXPECT_HOOK", "0")
    # Without a smartpar mode pytest-randomly would shuffle these order-sensitive tests.
    result = profiled_project.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:randomly")
    result.assert_outcomes(passed=12)
    assert not (profiled_project.path / ".smartpar").exists()


OUTCOME_TESTS = """
import time, pytest

@pytest.fixture
def slow():
    time.sleep(0.05)
    yield
    time.sleep(0.05)

@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("teardown")

@pytest.fixture
def broken_setup():
    raise RuntimeError("setup")

def test_passed(slow):
    time.sleep(0.05)

def test_failed():
    assert False

def test_skipped():
    pytest.skip("no")

@pytest.mark.xfail
def test_xfailed():
    assert False

@pytest.mark.xfail
def test_xpassed():
    pass

def test_setup_error(broken_setup):
    pass

def test_teardown_error(broken_teardown):
    pass
"""


def test_run_record_outcomes_and_durations(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_outcomes=OUTCOME_TESTS)
    pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider")
    merged = list((pytester.path / ".smartpar" / "runs" / "merged").glob("run_*.json"))
    tests = {t["nodeid"].split("::")[1]: t for t in json.loads(merged[0].read_text())["tests"]}
    assert {name: t["outcome"] for name, t in tests.items()} == {
        "test_passed": "passed",
        "test_failed": "failed",
        "test_skipped": "skipped",
        "test_xfailed": "xfailed",
        "test_xpassed": "xpassed",
        "test_setup_error": "error",
        "test_teardown_error": "error",
    }
    passed = tests["test_passed"]
    assert passed["duration"] >= 0.15  # setup + call + teardown
    assert passed["end"] - passed["start"] >= 0.15
    hist = json.loads((pytester.path / ".smartpar" / "history.json").read_text())["tests"]
    # Profiling runs are instrumented: counters only, no durations.
    assert json.loads(merged[0].read_text())["mode"] == "profile"
    assert hist["test_outcomes.py::test_passed"]["runs"] == 1
    assert hist["test_outcomes.py::test_passed"]["timed_runs"] == 0
    assert hist["test_outcomes.py::test_passed"]["duration_ema"] == 0.0
    assert hist["test_outcomes.py::test_teardown_error"]["fail_count"] == 1


def test_audit_hook_does_not_break_file_heavy_tests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_many_files(tmp_path):
            for i in range(500):
                (tmp_path / f"f{i}").write_text("x")
                (tmp_path / f"f{i}").read_text()
        """
    )
    result = pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


def test_ignore_state_ini_option(pytester: pytest.Pytester) -> None:
    pytester.makeini(
        "[pytest]\npythonpath = .\nsmartpar_ignore_state =\n"
        "    env:PYTEST_CURRENT_TEST\n    env:APP_*\n    cwd\n"
    )
    pytester.makepyfile(
        """
        import os
        def test_env():
            os.environ["APP_CURRENCY"] = "USD"
            os.environ["COV_CORE_SOURCE"] = "x"  # default pattern replaced by the ini list
            os.chdir("/")
        """
    )
    pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider")
    deps = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    assert deps["state_changes"] == {
        "test_ignore_state_ini_option.py::test_env": ["env:COV_CORE_SOURCE"]
    }


def test_default_ignore_state(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import os
        def test_env():
            os.environ["COV_CORE_DATAFILE"] = "x"
            os.environ["PYTEST_XDIST_WORKER_COUNT"] = "3"
        """
    )
    pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider")
    deps = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    assert deps["state_changes"] == {}


def test_named_temporary_files_do_not_conflict(pytester: pytest.Pytester) -> None:
    # Since Python 3.12 NamedTemporaryFile emits open('<tmpdir>', 'w+') and
    # then creates a uniquely named file: two such tests share nothing.
    pytester.makepyfile(
        test_tmp="""
        import tempfile

        def test_one():
            with tempfile.NamedTemporaryFile("wb+") as fh:
                fh.write(b"1")

        def test_two():
            with tempfile.NamedTemporaryFile("wb+") as fh:
                fh.write(b"2")
        """
    )
    result = pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=2)
    deps = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    assert deps["conflicts"] == []
    assert deps["groups"] == []


DIRECTORY_TESTS = """
import os, shutil

def test_a_create_new_dir():
    os.makedirs("out/new", exist_ok=True)  # creates out and out/new

def test_b_makedirs_existing():
    os.makedirs("out/new", exist_ok=True)  # out/new exists: still a creation of it

def test_c_write_inside():
    with open("out/new/data.txt", "w") as fh:
        fh.write("x")

def test_d_read_inside():
    open("out/new/data.txt").read()

def test_e_rmtree():
    shutil.rmtree("out")

def test_f_rename_dir():
    os.makedirs("moved_src/inner")
    os.rename("moved_src", "moved_dst")
"""


# rmtree of out/ conflicts with everything at or below it; two creations of
# out/new do not conflict, nor does creating out/new with reading or writing a
# file inside it.
DIRECTORY_PAIRS = {
    ("test_a_create_new_dir", "test_e_rmtree"),
    ("test_b_makedirs_existing", "test_e_rmtree"),
    ("test_c_write_inside", "test_e_rmtree"),
    ("test_d_read_inside", "test_e_rmtree"),
    ("test_c_write_inside", "test_d_read_inside"),
}


def directory_pairs(deps: dict) -> set[tuple[str, str]]:
    return {(c["a"].split("::")[1], c["b"].split("::")[1]) for c in deps["conflicts"]}


def test_profile_directory_operations(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_dirs=DIRECTORY_TESTS)
    result = pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=6)
    deps = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    access = {k.split("::")[1]: rw_only(v) for k, v in deps["file_access"].items()}
    assert access["test_a_create_new_dir"] == {
        "R": [],
        "W": [],
        "W_tree": [],
        "C": ["<project>/out", "<project>/out/new"],
    }
    assert access["test_b_makedirs_existing"]["C"] == ["<project>/out/new"]
    assert access["test_e_rmtree"]["W_tree"] == ["<project>/out"]
    assert access["test_f_rename_dir"]["W_tree"] == ["<project>/moved_dst", "<project>/moved_src"]
    assert deps["profile_stats"]["existing_dir_mkdir"] == 1  # one mkdir of the existing leaf

    assert directory_pairs(deps) == DIRECTORY_PAIRS
    kinds = {c["kind"] for c in deps["conflicts"] if c["resource"] == "<project>/out/new"}
    assert kinds == {"C-W"}  # creations of out/new vs rmtree of out
    # Collection order is saved and drives the order inside the group.
    (group,) = deps["groups"]
    assert [t.split("::")[1] for t in group["order"]] == [
        "test_a_create_new_dir",
        "test_b_makedirs_existing",
        "test_c_write_inside",
        "test_d_read_inside",
        "test_e_rmtree",
    ]
    assert deps["collection_index"]["test_dirs.py::test_f_rename_dir"] == 5


def test_directory_conflicts_do_not_depend_on_leftovers(pytester: pytest.Pytester) -> None:
    # out/new is left over from an earlier run: the conflicts stay the same.
    (pytester.path / "out" / "new").mkdir(parents=True)
    pytester.makepyfile(test_dirs=DIRECTORY_TESTS)
    result = pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=6)
    deps = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    assert directory_pairs(deps) == DIRECTORY_PAIRS
