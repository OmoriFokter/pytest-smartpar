"""Phase 6: ``pytest --smartpar-detect`` end to end (real child processes)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TIMEOUT = 300

OD_TESTS = """
import os
from app import state

def test_polluter():
    state.SETTINGS["cur"] = "USD"

def test_victim():
    assert state.SETTINGS["cur"] == "UAH"

def test_setter():
    state.READY["x"] = 1

def test_brittle():
    assert state.READY.get("x") == 1

def test_nod():
    # Deterministic stand-in for a random failure: every other invocation
    # fails (counter file shared by all child processes, outside the project).
    with open(os.environ["NOD_COUNTER"], "ab") as fh:
        fh.write(b"x")
        size = fh.tell()
    assert size % 2 == 1
"""


@pytest.fixture
def od_project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("NOD_COUNTER", str(tmp_path / "nod_counter"))
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.mkpydir("app")
    (pytester.path / "app" / "state.py").write_text('SETTINGS = {"cur": "UAH"}\nREADY = {}\n')
    tests = pytester.mkdir("tests")
    (tests / "test_od.py").write_text(OD_TESTS)
    for i in range(6):
        (tests / f"test_f{i}.py").write_text(f"def test_f{i}(): pass\n")
    return pytester


def load(pytester: pytest.Pytester, name: str) -> dict:
    return json.loads((pytester.path / ".smartpar" / name).read_text())


N = "tests/test_od.py::"


def test_detect_finds_planted_dependencies(od_project: pytest.Pytester) -> None:
    od_project.runpytest_subprocess("--smartpar-profile", "-p", "no:randomly", timeout=TIMEOUT)
    result = od_project.runpytest_subprocess(
        "--smartpar-detect", "--smartpar-seed", "100", "--smartpar-runs", "20", timeout=TIMEOUT
    )
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*- smartpar detect -*",
            "*20 random runs (seeds 100..119)*a fresh project copy per run*",
            "*victims: 1, brittle: 1, NOD: 1, unresolved: 0*",
        ]
    )
    flaky = load(od_project, "flaky.json")
    assert flaky["seeds"] == list(range(100, 120))
    assert flaky["detector"]["base_seed"] == 100
    (victim,) = flaky["od_victims"]
    assert (victim["victim"], victim["polluters"]) == (N + "test_victim", [N + "test_polluter"])
    assert victim["state_hint"] == ["app.state.SETTINGS"]  # from profiling
    (brittle,) = flaky["od_brittle"]
    assert (brittle["test"], brittle["state_setters"]) == (N + "test_brittle", [N + "test_setter"])
    assert [n["test"] for n in flaky["nod"]] == [N + "test_nod"]
    deps = load(od_project, "dependencies.json")
    assert {(e["before"], e["after"]) for e in deps["order_edges"]} == {
        (N + "test_victim", N + "test_polluter"),
        (N + "test_setter", N + "test_brittle"),
    }
    # Groups now include the confirmed order pairs.
    assert any(set(g["tests"]) == {N + "test_victim", N + "test_polluter"} for g in deps["groups"])


RACE_TESTS = """
import time

def _roundtrip(value):
    with open("shared.txt", "w") as fh:
        fh.write(value)
    time.sleep(0.3)
    with open("shared.txt") as fh:
        assert fh.read() == value

def test_a():
    _roundtrip("a")

def test_b():
    _roundtrip("b")
"""


def test_parallel_runs_do_not_share_files(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_race=RACE_TESTS)
    # Control: two concurrent sessions in the same directory do interfere.
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "-p", "no:cacheprovider"],
            cwd=pytester.path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:randomly",
                "-p",
                "no:cacheprovider",
                "test_race.py::test_b",
                "test_race.py::test_a",
            ],
            cwd=pytester.path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
    ]
    assert any(p.wait(timeout=TIMEOUT) != 0 for p in procs)
    (pytester.path / "shared.txt").unlink()

    # The detector runs many sessions concurrently, each in its own copy.
    result = pytester.runpytest_subprocess(
        "--smartpar-detect",
        "--smartpar-runs",
        "8",
        "--smartpar-detect-workers",
        "8",
        timeout=TIMEOUT,
    )
    assert result.ret == 0
    flaky = json.loads((pytester.path / ".smartpar" / "flaky.json").read_text())
    assert (flaky["od_victims"], flaky["od_brittle"], flaky["nod"], flaky["unresolved"]) == (
        [],
        [],
        [],
        [],
    )
    assert flaky["detector"]["parallel_runs"] == 8
    assert not (pytester.path / "shared.txt").exists()  # nothing ran in the project itself


def test_budget_writes_partial_result(od_project: pytest.Pytester) -> None:
    result = od_project.runpytest_subprocess(
        "--smartpar-detect", "--smartpar-detect-budget", "0.01", timeout=TIMEOUT
    )
    assert result.ret == 0
    flaky = load(od_project, "flaky.json")
    assert flaky["unresolved"]
    assert all(u["reason"] == "budget" for u in flaky["unresolved"])
    assert flaky["detector"]["budget_s"] == 0.01


def test_detect_rejects_xdist(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest_subprocess("--smartpar-detect", "-n", "2")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--smartpar-detect runs its own child processes*"])


RANDOM_NOD = """
import random

def test_coin():
    # Plain random: pytest-randomly re-seeds it before every test. Isolated
    # runs must use different seeds, otherwise all of them would be identical.
    assert random.random() < 0.5
"""


def test_nod_on_plain_random_is_detected_with_distinct_isolation_seeds(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_coin=RANDOM_NOD)
    pytester.makeini("[pytest]\nsmartpar_max_pair_checks = 4\nsmartpar_max_orders_per_victim = 2\n")
    result = pytester.runpytest_subprocess(
        "--smartpar-detect", "--smartpar-seed", "500", "--smartpar-runs", "10", timeout=TIMEOUT
    )
    assert result.ret == 0
    flaky = json.loads((pytester.path / ".smartpar" / "flaky.json").read_text())
    assert [n["test"] for n in flaky["nod"]] == ["test_coin.py::test_coin"]
    meta = flaky["detector"]
    assert meta["limits"] == {"max_pair_checks": 4, "max_orders_per_victim": 2}
    assert meta["truncated_victims"] == 0
    assert meta["ordered_seed_start"] == 510  # right after the random-run seeds
    assert meta["ordered_runs"] >= 3


CONDITIONAL = """
import os

def test_a_read_if_exists():
    if os.path.exists("data.txt"):  # conditional access
        open("data.txt").read()

def test_b_write():
    with open("data.txt", "w") as fh:
        fh.write("x")
"""


def test_detector_runs_add_file_accesses_missed_by_profiling(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_cond=CONDITIONAL)
    pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:randomly", timeout=TIMEOUT)
    after_profile = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    assert after_profile["conflicts"] == []  # the reader ran first: nothing was opened
    (pytester.path / "data.txt").unlink()  # start the detector from the same clean state

    result = pytester.runpytest_subprocess(
        "--smartpar-detect", "--smartpar-seed", "7", "--smartpar-runs", "8", timeout=TIMEOUT
    )
    assert result.ret == 0
    deps = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    reader = deps["file_access"]["test_cond.py::test_a_read_if_exists"]
    assert reader["R"] == ["<project>/data.txt"]
    assert reader["source"]["R"] == {"<project>/data.txt": "detect"}
    writer = deps["file_access"]["test_cond.py::test_b_write"]
    assert writer["source"]["W"] == {"<project>/data.txt": "profile"}
    assert [(c["a"], c["b"], c["kind"]) for c in deps["conflicts"]] == [
        ("test_cond.py::test_a_read_if_exists", "test_cond.py::test_b_write", "R-W")
    ]

    # Profiling again keeps what only the detector saw.
    pytester.runpytest_subprocess("--smartpar-profile", "-p", "no:randomly", timeout=TIMEOUT)
    again = json.loads((pytester.path / ".smartpar" / "dependencies.json").read_text())
    assert len(again["conflicts"]) == 1
