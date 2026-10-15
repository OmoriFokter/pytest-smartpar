"""Phase 8: experimental stand (bench/): statistics, run order, clean state, timing plugin."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("matplotlib")

from bench import analyze, run_benchmarks, setup_projects  # noqa: E402
from bench.setup_projects import Project  # noqa: E402

# --- statistics ------------------------------------------------------------------------


def test_summary_and_confidence_interval() -> None:
    s = analyze.summarize([10.0, 12.0, 11.0, 13.0, 9.0])
    assert s["mean"] == 11.0
    assert s["std"] == pytest.approx(math.sqrt(2.5))
    assert s["ci95"] == pytest.approx(2.776 * math.sqrt(2.5) / math.sqrt(5))
    assert math.isnan(analyze.summarize([3.0])["ci95"])
    assert analyze.t95(100) == 1.96


@pytest.mark.parametrize(
    ("df", "t"),
    # Two-sided 95 % critical values of Student's t (standard tables).
    [(1, 12.706), (2, 4.303), (4, 2.776), (9, 2.262), (19, 2.093), (29, 2.045), (30, 2.042)],
)
def test_student_t_critical_values(df: int, t: float) -> None:
    assert analyze.t95(df) == t
    assert len(analyze.T95) == 30
    assert all(a > b for a, b in zip(analyze.T95, analyze.T95[1:], strict=False))


def test_ci_for_five_repeats_uses_t_not_normal() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = analyze.summarize(values)
    assert s["ci95"] == pytest.approx(2.776 * s["std"] / math.sqrt(5))
    assert s["ci95"] > 1.96 * s["std"] / math.sqrt(5)


def test_karp_flatt_and_amdahl_are_consistent() -> None:
    for e in (0.0, 0.05, 0.3):
        for m in (2, 4, 8):
            assert analyze.karp_flatt(analyze.amdahl(e, m), m) == pytest.approx(e)
    assert analyze.karp_flatt(1.5, 1) is None


def test_break_even() -> None:
    assert analyze.break_even_runs(44.0, 5.7, 2.6) == 15  # ceil(44 / 3.1)
    assert analyze.break_even_runs(10.0, 5.0, 5.0) is None  # no gain: never
    assert analyze.break_even_runs(0.5, 5.0, 1.0) == 1


def rec(config: str, m: int, wall: float, before: list[str], after: list[str] | None = None):
    return {
        "project": "p",
        "config": config,
        "workers": m,
        "wall_s": wall,
        "failed_before_rerun": before,
        "failed_after_rerun": before if after is None else after,
        "predicted_wall_s": None,
    }


def test_failure_table_counts_false_failures_before_and_after() -> None:
    records = [
        rec("serial", 1, 8.0, ["t_nod"]),
        rec("serial", 1, 8.0, []),
        rec("xdist-load", 4, 3.0, ["t_od", "t_nod"]),
        rec("smartpar", 4, 2.5, ["t_new", "t_nod2"], after=["t_nod2"]),
    ]
    rows = {r["config"]: r for r in analyze.failure_table(records, {"p": {"t_nod2"}})}
    assert rows["xdist-load"]["false_failures_before_rerun"] == 1  # t_nod failed serially
    assert rows["smartpar"]["false_failures_before_rerun"] == 2
    assert rows["smartpar"]["false_failures_after_rerun"] == 1  # t_new passed when re-run
    assert rows["smartpar"]["false_failures_after_rerun_excl_nod"] == 0
    assert rows["serial"]["false_failures_after_rerun"] == ""


def test_time_table_speedup() -> None:
    records = [
        rec("serial", 1, 8.0, []),
        rec("serial", 1, 8.0, []),
        rec("smartpar", 4, 2.0, []),
        rec("smartpar", 4, 2.0, []),
    ]
    rows = {(r["config"], r["workers"]): r for r in analyze.time_table(records)}
    assert rows[("smartpar", 4)]["speedup"] == 4.0
    assert rows[("smartpar", 4)]["efficiency"] == 1.0
    assert rows[("smartpar", 4)]["karp_flatt"] == pytest.approx(0.0)


# --- run order -----------------------------------------------------------------------------


def test_jobs_and_shuffled_order() -> None:
    jobs = run_benchmarks.jobs_for(list(run_benchmarks.CONFIGS), [1, 2, 4])
    assert jobs.count(("serial", 1)) == 1
    assert len(jobs) == 1 + 3 * 3
    first = run_benchmarks.shuffled(jobs, 7, 0)
    assert first == run_benchmarks.shuffled(jobs, 7, 0)  # reproducible
    assert sorted(first) == sorted(jobs)
    orders = {tuple(run_benchmarks.shuffled(jobs, 7, r)) for r in range(5)}
    assert len(orders) > 1  # a new order in every repeat


def test_config_args(tmp_path: Path) -> None:
    assert run_benchmarks.config_args("serial", 4, tmp_path) == ["-p", "no:xdist"]
    assert run_benchmarks.config_args("xdist-loadfile", 2, tmp_path)[-2:] == ["--dist", "loadfile"]
    assert "--smartpar" in run_benchmarks.config_args("smartpar", 2, tmp_path)


# --- projects and clean state ---------------------------------------------------------------


def test_projects_yaml_pins_every_git_project() -> None:
    projects = setup_projects.load_projects()
    assert {p.name for p in projects} >= {"control_project", "rich", "attrs", "pydantic"}
    for p in projects:
        if p.local is None:
            assert len(p.sha) == 40 and all(c in "0123456789abcdef" for c in p.sha)


def test_projects_share_one_python_version_unless_overridden(tmp_path: Path) -> None:
    projects = setup_projects.load_projects()
    assert {p.python for p in projects} == {"3.12"}
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "python": "3.11",
                "projects": [
                    {
                        "name": "a",
                        "local": "x",
                        "tests": "t",
                        "install": [],
                        "pytest_args": [],
                        "deselect": [],
                    },
                    {
                        "name": "b",
                        "local": "x",
                        "tests": "t",
                        "install": [],
                        "pytest_args": [],
                        "deselect": [],
                        "python": "3.13",
                    },
                ],
            }
        )
    )
    assert [p.python for p in setup_projects.load_projects(path)] == ["3.11", "3.13"]


def test_load_projects_rejects_missing_sha(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "projects": [
                    {
                        "name": "x",
                        "repo": "https://example.invalid/x.git",
                        "tests": "t",
                        "install": [],
                        "pytest_args": [],
                        "deselect": [],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="40-char sha"):
        setup_projects.load_projects(path)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_clean_state_restores_the_pinned_commit(tmp_path: Path) -> None:
    project = Project(
        name="demo",
        tests="tests",
        install=[],
        pytest_args=[],
        deselect=[],
        repo="local",
        sha="0" * 40,
    )
    src, venv = project.paths(tmp_path)
    src.mkdir(parents=True)
    git(src, "init", "-q")
    (src / ".gitignore").write_text("output/\n")
    (src / "a.txt").write_text("original")
    git(src, "add", "-A")
    git(src, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    project.sha = git(src, "rev-parse", "HEAD")
    venv.mkdir()
    (venv / "keep").write_text("venv outside src")

    (src / "a.txt").write_text("modified")
    (src / "untracked.txt").write_text("x")
    (src / "output").mkdir()
    (src / "output" / "left.txt").write_text("leftover of an earlier run")
    (src / "__pycache__").mkdir()
    (src / "venv").mkdir()  # a venv inside src is kept as well
    (src / "venv" / "marker").write_text("")

    setup_projects.clean_state(project, tmp_path)
    assert (src / "a.txt").read_text() == "original"
    assert not (src / "untracked.txt").exists()
    assert not (src / "output").exists()
    assert not (src / "__pycache__").exists()
    assert (src / "venv" / "marker").exists()
    assert (venv / "keep").exists()


def test_physical_cores_positive() -> None:
    assert 1 <= setup_projects.physical_cores() <= (__import__("os").cpu_count() or 1)


# --- timing plugin -------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["-n", "2"]])
def test_timing_plugin(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    out = pytester.path / "timing.jsonl"
    monkeypatch.setenv("SMARTPAR_BENCH_TIMING", str(out))
    monkeypatch.setenv("PYTHONPATH", str(setup_projects.REPO_ROOT))
    pytester.makepyfile(
        "import pytest\n"
        "def test_ok(): pass\n"
        "def test_fail(): assert False\n"
        "def test_skip(): pytest.skip('x')\n"
    )
    pytester.runpytest_subprocess("-p", "bench.timing_plugin", "-p", "no:randomly", *extra)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    outcomes = {x["nodeid"].split("::")[1]: x["outcome"] for x in lines}
    assert outcomes == {"test_ok": "passed", "test_fail": "failed", "test_skip": "skipped"}
    workers = {x["worker"] for x in lines}
    assert workers == ({"master"} if not extra else workers - {"master"})
    assert all(x["end"] >= x["start"] for x in lines)


# --- analysis end to end on a synthetic series ----------------------------------------------


def test_analyze_synthetic_series(tmp_path: Path) -> None:
    series = tmp_path / "series"
    raw = series / "raw" / "p"
    raw.mkdir(parents=True)
    (series / "env.json").write_text(json.dumps({"projects": {"p": {"ground_truth": None}}}))
    (raw / "prep.json").write_text(
        json.dumps({"profile_s": 2.0, "detect_s": 8.0, "overhead_s": 10.0, "warmup_s": 3.0})
    )
    walls = {
        ("serial", 1): 8.0,
        ("xdist-load", 2): 5.0,
        ("xdist-loadfile", 2): 6.0,
        ("smartpar", 2): 4.0,
        ("xdist-load", 4): 3.0,
        ("xdist-loadfile", 4): 4.0,
        ("smartpar", 4): 2.0,
    }
    for (config, m), wall in walls.items():
        for r in range(3):
            record = rec(config, m, wall + 0.01 * r, [])
            if config == "smartpar":
                record["predicted_wall_s"] = wall * 0.9
            (raw / f"{config}_m{m}_r{r}.json").write_text(json.dumps(record))
            timing = [
                {
                    "nodeid": f"t{i}",
                    "worker": f"gw{i % m}",
                    "start": i,
                    "end": i + 0.5,
                    "duration": 0.5,
                    "outcome": "passed",
                }
                for i in range(6)
            ]
            (raw / f"{config}_m{m}_r{r}.timing.jsonl").write_text(
                "\n".join(json.dumps(t) for t in timing)
            )
    out = analyze.analyze(series)
    overhead = (out / "overhead.csv").read_text().splitlines()
    assert overhead[0].startswith("project,workers,profile_s")
    assert overhead[-1].endswith(",10")  # m=4: 10 s / (3.01 - 2.01) s per run
    for name in ("fig1_times", "fig2_gantt_p", "fig3_speedup_p", "fig4_false_failures"):
        assert (out / f"{name}.png").exists() and (out / f"{name}.svg").exists()
    assert "Час виконання" in (out / "times.md").read_text()
    rows = (out / "prediction.csv").read_text().splitlines()[1:]
    by_m = {int(r.split(",")[1]): float(r.split(",")[3]) for r in rows}
    # wall = w + 0.01 r, predicted = 0.9 w: mean error (w + 0.01 - 0.9 w) / (0.9 w)
    assert by_m[2] == pytest.approx((0.1 * 4.0 + 0.01) / 3.6 * 100)
    assert by_m[4] == pytest.approx((0.1 * 2.0 + 0.01) / 1.8 * 100)


def test_env_marks_uncommitted_tool_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_projects, "smartpar_dirty", lambda: [" M src/smartpar/plugin.py"])
    info = setup_projects.machine_info()
    assert info["smartpar_dirty"] is True
    assert info["smartpar_dirty_files"] == [" M src/smartpar/plugin.py"]
    assert len(info["smartpar_sha"]) == 40


def test_run_with_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    import os
    import sys
    import time

    pidfile = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    out = run_benchmarks.run_with_timeout(
        [sys.executable, "-c", script], tmp_path, dict(os.environ), timeout=2
    )
    assert out.timed_out and out.exit_code is None
    assert 2 <= out.wall_s < 30
    grandchild = int(pidfile.read_text())
    for _ in range(50):  # the kill is asynchronous; the zombie is reaped shortly after
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild (an xdist worker, say) survived the timeout")


def test_run_with_timeout_normal_exit(tmp_path: Path) -> None:
    import os
    import sys

    out = run_benchmarks.run_with_timeout(
        [sys.executable, "-c", "print('hi'); raise SystemExit(3)"], tmp_path, dict(os.environ), 30
    )
    assert (out.exit_code, out.timed_out) == (3, False)
    assert "hi" in out.output_tail


def test_timed_out_runs_are_not_measurements() -> None:
    records = [
        rec("serial", 1, 8.0, []),
        rec("smartpar", 4, 2.0, []),
        {**rec("smartpar", 4, 600.0, []), "timed_out": True},
    ]
    times = {(r["config"], r["workers"]): r for r in analyze.time_table(records)}
    assert times[("smartpar", 4)]["mean_s"] == 2.0 and times[("smartpar", 4)]["runs"] == 1
    failures = {r["config"]: r for r in analyze.failure_table(records)}
    assert failures["smartpar"]["timeouts"] == 1
