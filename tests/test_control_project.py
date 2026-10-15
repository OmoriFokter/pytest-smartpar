"""Phase 7: the whole method on examples/control_project against ground_truth.json.

Slow (profile + 20 random runs + 11 parallel runs): marked ``control`` and
deselected by default; run with ``pytest -m control``. All detector runs use
fixed seeds, so the order-dependent part is reproducible; only the two NOD
tests are random by design (random.SystemRandom).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.control

EXAMPLE = Path(__file__).parents[1] / "examples" / "control_project"
GROUND_TRUTH = json.loads((EXAMPLE / "ground_truth.json").read_text())
TIMEOUT = 1800
DETECT_SEED = 2026
REPEATS = 10


def precision_recall(found: set, expected: set) -> tuple[float, float]:
    tp = len(found & expected)
    precision = tp / len(found) if found else 1.0
    recall = tp / len(expected) if expected else 1.0
    return precision, recall


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("control") / "control_project"
    shutil.copytree(
        EXAMPLE,
        path,
        ignore=shutil.ignore_patterns(".smartpar", "output", "__pycache__", ".pytest_cache"),
    )
    return path


def run(project: Path, *args: str) -> int:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=project,
        timeout=TIMEOUT,
        check=False,
        capture_output=True,
        text=True,
    )
    print(proc.stdout[-3000:])
    return proc.returncode


def data(project: Path, name: str) -> dict:
    return json.loads((project / ".smartpar" / name).read_text())


NOD = {n["test"] for n in GROUND_TRUTH["nod"]}


@pytest.fixture(scope="module")
def profiled(project: Path) -> Path:
    run(project, "--smartpar-profile")
    return project


@pytest.fixture(scope="module")
def detected(profiled: Path) -> Path:
    assert run(profiled, "--smartpar-detect", "--smartpar-seed", str(DETECT_SEED)) == 0
    return profiled


def test_file_conflicts_match_ground_truth(profiled: Path) -> None:
    deps = data(profiled, "dependencies.json")
    found = {(c["a"], c["b"], c["resource"], c["kind"]) for c in deps["conflicts"]}
    expected = {(c["a"], c["b"], c["resource"], c["kind"]) for c in GROUND_TRUTH["file_conflicts"]}
    assert precision_recall(found, expected) == (1.0, 1.0), found ^ expected


def test_restored_state_is_not_reported(profiled: Path) -> None:
    changes = data(profiled, "dependencies.json")["state_changes"]
    assert not set(GROUND_TRUTH["restored_state"]) & set(changes)
    assert len(data(profiled, "history.json")["tests"]) == GROUND_TRUTH["test_count"]


def test_od_pairs_and_brittle_match_ground_truth(detected: Path) -> None:
    flaky = data(detected, "flaky.json")
    assert flaky["seeds"] == list(range(DETECT_SEED, DETECT_SEED + 20))
    found_od = {(p, v["victim"]) for v in flaky["od_victims"] for p in v["polluters"]}
    expected_od = {(p["polluter"], p["victim"]) for p in GROUND_TRUTH["od_pairs"]}
    assert precision_recall(found_od, expected_od) == (1.0, 1.0), found_od ^ expected_od
    found_b = {(s, b["test"]) for b in flaky["od_brittle"] for s in b["state_setters"]}
    expected_b = {(b["setter"], b["test"]) for b in GROUND_TRUTH["brittle"]}
    assert precision_recall(found_b, expected_b) == (1.0, 1.0), found_b ^ expected_b


def test_nod_tests_are_never_order_dependent(detected: Path) -> None:
    flaky = data(detected, "flaky.json")
    od_related = {v["victim"] for v in flaky["od_victims"]}
    od_related |= {p for v in flaky["od_victims"] for p in v["polluters"]}
    od_related |= {b["test"] for b in flaky["od_brittle"]}
    od_related |= {s for b in flaky["od_brittle"] for s in b["state_setters"]}
    assert not NOD & od_related
    reported_nod = {n["test"] for n in flaky["nod"]}
    unresolved = {u["test"] for u in flaky["unresolved"]}
    assert reported_nod <= NOD  # nothing else is called NOD
    assert unresolved <= NOD  # every planted OD test was resolved


def test_parallel_runs_have_no_false_failures(detected: Path) -> None:
    run(detected, "-n", "4", "--smartpar")  # first run: collects durations
    from smartpar import history

    history.merge(detected / ".smartpar")
    failures = []
    for _ in range(REPEATS):
        run(detected, "-n", "4", "--smartpar")
        runs = sorted((detected / ".smartpar" / "runs").glob("run_*.json"))
        record = json.loads(runs[-1].read_text())
        assert record["mode"] == "smartpar"
        assert len(record["tests"]) == GROUND_TRUTH["test_count"]
        failures += [
            t["nodeid"]
            for t in record["tests"]
            if t["outcome"] in ("failed", "error") and t["nodeid"] not in NOD
        ]
    assert failures == []
