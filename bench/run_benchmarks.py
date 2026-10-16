"""Benchmark series: every configuration x number of workers x repeat.

    python bench/run_benchmarks.py --only control_project --repeats 5 --workers 1 2 4 8

Projects must be set up first (``setup_projects.py``). Per project:

1. **Preparation** (timed and reported separately as the method's overhead):
   clean state -> ``--smartpar-profile`` -> ``--smartpar-detect`` (same
   working tree, as in CI), then ``--warmup`` smartpar runs per m to measure
   durations and calibrate the overhead model (also timed separately). The
   smartpar data directory lives outside the project, so cleaning the project
   never deletes it.
2. **Measurement**: configurations ``serial`` (-p no:xdist), ``xdist-load``,
   ``xdist-loadfile`` and ``smartpar``, each with m in ``--workers`` (serial
   once), ``--repeats`` times. The order of the runs is shuffled anew in every
   repeat (seeded, recorded) to spread warm-up/cache effects. Every run starts
   from a clean state of the project (``git clean -xdf`` at the pinned SHA)
   and smartpar runs from a fresh copy of the prepared data, so runs do not
   influence each other.

All configurations load ``bench.timing_plugin`` and run with ``-p
no:randomly`` (the projects' normal order; the detector uses pytest-randomly on
its own). Raw data: ``<results>/<date>_<host>/raw/<project>/``; ``env.json``
describes the machine and versions of the series.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.setup_projects import (
    DEFAULT_PROJECTS,
    DEFAULT_WORK,
    REPO_ROOT,
    Project,
    clean_state,
    load_projects,
    machine_info,
    physical_cores,
    pinned_sha,
    select,
    venv_python,
    versions,
)

CONFIGS = ("serial", "xdist-load", "xdist-loadfile", "smartpar")
PROFILE_TIMEOUT_FACTOR = 3


class BenchError(RuntimeError):
    """A project cannot be benchmarked (e.g. profiling failed or timed out)."""


REQUIRED_WORKERS = (2, 4)  # standard GitHub Actions runners
FAILED = ("failed", "error")


@dataclass
class Outcome:
    wall_s: float
    exit_code: int | None  # None: killed after the timeout
    output_tail: str

    @property
    def timed_out(self) -> bool:
        return self.exit_code is None


def run_with_timeout(
    cmd: list[str], cwd: Path, env: dict[str, str], timeout: float | None
) -> Outcome:
    """Run ``cmd``; after ``timeout`` seconds kill its whole process group.

    pytest-xdist workers and smartpar's child runs are grandchildren: killing
    only the direct child would leave them running and skew later runs.
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return Outcome(time.perf_counter() - t0, proc.returncode, out[-4000:])
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return Outcome(time.perf_counter() - t0, None, (out or "")[-4000:])


def config_args(config: str, workers: int, data_dir: Path) -> list[str]:
    if config == "serial":
        return ["-p", "no:xdist"]
    if config == "xdist-load":
        return ["-n", str(workers), "--dist", "load"]
    if config == "xdist-loadfile":
        return ["-n", str(workers), "--dist", "loadfile"]
    if config == "smartpar":
        return ["-n", str(workers), "--smartpar", data_dir_arg(data_dir)]
    raise ValueError(config)


def data_dir_arg(data_dir: Path) -> str:
    """``--smartpar-dir`` as one ``--opt=value`` token.

    pytest looks for its configuration file before plugin options are known,
    so the value of a separate ``--smartpar-dir PATH`` would count as a test
    path: the rootdir would become the common ancestor of the project and the
    data directory, and the project's own pytest configuration would be lost.
    """
    return f"--smartpar-dir={data_dir}"


def jobs_for(configs: list[str], workers: list[int]) -> list[tuple[str, int]]:
    """(config, m) pairs; serial runs once (m = 1)."""
    jobs: list[tuple[str, int]] = []
    for config in configs:
        jobs += [("serial", 1)] if config == "serial" else [(config, m) for m in workers]
    return jobs


def shuffled(jobs: list[tuple[str, int]], seed: int, repeat: int) -> list[tuple[str, int]]:
    """A different, reproducible order in every repeat."""
    rng = random.Random(f"{seed}:{repeat}")
    return rng.sample(jobs, len(jobs))


class Bench:
    def __init__(
        self,
        project: Project,
        work: Path,
        raw: Path,
        run_timeout: float | None = None,
        detect_budget: float | None = None,
    ) -> None:
        self.project = project
        self.run_timeout = run_timeout
        self.detect_budget = detect_budget
        self.work = work
        self.src, venv = project.paths(work)
        self.python = venv_python(venv)
        self.raw = raw / project.name
        self.raw.mkdir(parents=True, exist_ok=True)
        self.master = self.raw / "smartpar_master"  # prepared data, never mutated later

    def pytest(
        self, args: list[str], timing: Path | None = None, timeout: float | None = None
    ) -> Outcome:
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        env.pop("SMARTPAR_BENCH_TIMING", None)
        if timing is not None:
            env["SMARTPAR_BENCH_TIMING"] = str(timing)
        cmd = [
            str(self.python),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "bench.timing_plugin",
            self.project.tests,
            *self.project.pytest_args,
            *[f"--deselect={d}" for d in self.project.deselect],
            *args,
        ]
        return run_with_timeout(cmd, self.src, env, timeout)

    # --- preparation ---------------------------------------------------------------

    def prepare(
        self, workers: list[int], detect_runs: int, detect_seed: int, warmup: int
    ) -> dict[str, Any]:
        if self.master.exists():
            shutil.rmtree(self.master)
        data = [data_dir_arg(self.master)]
        clean_state(self.project, self.work)
        # Profiling is instrumented and serial: allow several plain runs.
        profile_timeout = PROFILE_TIMEOUT_FACTOR * self.run_timeout if self.run_timeout else None
        profile = self.pytest(["--smartpar-profile", *data], timeout=profile_timeout)
        if profile.timed_out or not (self.master / "dependencies.json").exists():
            raise BenchError(
                f"{self.project.name}: profiling "
                + ("timed out" if profile.timed_out else f"failed (exit {profile.exit_code})")
                + f"\n{profile.output_tail}"
            )
        shutil.copy(self.master / "dependencies.json", self.raw / "deps_after_profile.json")
        # The detector's runs re-enable pytest-randomly themselves.
        # Plugin options as single tokens, like --smartpar-dir (see data_dir_arg).
        detect_args = [
            "--smartpar-detect",
            "-p",
            "randomly",
            f"--smartpar-runs={detect_runs}",
            f"--smartpar-seed={detect_seed}",
            *data,
        ]
        detect_timeout = None
        if self.detect_budget is not None:
            detect_args.append(f"--smartpar-detect-budget={self.detect_budget}")
            # The budget stops new child runs; one running child may still finish.
            detect_timeout = self.detect_budget + (self.run_timeout or 0) + 60
        detect = self.pytest(detect_args, timeout=detect_timeout)
        warmup_s = 0.0
        warmup_timeouts = 0
        for m in workers:
            for _ in range(warmup):
                clean_state(self.project, self.work)
                run = self.pytest(["-n", str(m), "--smartpar", *data], timeout=self.run_timeout)
                warmup_s += run.wall_s
                warmup_timeouts += run.timed_out
                self._merge(self.master)
        clean_state(self.project, self.work)
        flaky = self.master / "flaky.json"
        unresolved = json.loads(flaky.read_text())["unresolved"] if flaky.exists() else None
        prep = {
            "profile_s": profile.wall_s,
            "profile_exit_code": profile.exit_code,
            "detect_s": detect.wall_s,
            "detect_exit_code": detect.exit_code,
            "detect_timed_out": detect.timed_out,
            "detect_runs": detect_runs,
            "detect_seed": detect_seed,
            "detect_budget_s": self.detect_budget,
            "detect_budget_hits": None
            if unresolved is None
            else sum(1 for u in unresolved if u.get("reason") == "budget"),
            "warmup_runs_per_m": warmup,
            "warmup_s": warmup_s,
            "warmup_timeouts": warmup_timeouts,
            "run_timeout_s": self.run_timeout,
            "overhead_s": profile.wall_s + detect.wall_s,  # one-time cost of the method
        }
        (self.raw / "prep.json").write_text(json.dumps(prep, indent=2))
        return prep

    def _merge(self, data_dir: Path) -> None:
        subprocess.run(
            [str(self.python), "-m", "smartpar.cli", "--dir", str(data_dir), "merge"],
            check=True,
            capture_output=True,
        )

    # --- measurement -----------------------------------------------------------------

    def measure(self, config: str, m: int, repeat: int, position: int) -> dict[str, Any]:
        clean_state(self.project, self.work)
        tag = f"{config}_m{m}_r{repeat}"
        timing = self.raw / f"{tag}.timing.jsonl"
        data_dir = self.raw / f"{tag}.smartpar"
        if config == "smartpar":
            shutil.copytree(self.master, data_dir)
        run = self.pytest(config_args(config, m, data_dir), timing=timing, timeout=self.run_timeout)
        wall, rc, tail = run.wall_s, run.exit_code, run.output_tail
        tests = [json.loads(x) for x in timing.read_text().splitlines()] if timing.exists() else []
        failed_before = sorted(t["nodeid"] for t in tests if t["outcome"] in FAILED)
        record: dict[str, Any] = {
            "project": self.project.name,
            "config": config,
            "workers": m,
            "repeat": repeat,
            "position": position,
            "wall_s": wall,
            "exit_code": rc,
            # Killed after --run-timeout: wall_s is the timeout, not a measurement.
            "timed_out": run.timed_out,
            "tests": len(tests),
            "failed_before_rerun": failed_before,
            # Plain xdist does not re-run anything: after == before.
            "failed_after_rerun": failed_before,
            "timing_file": timing.name,
            "output_tail": tail,
        }
        if config == "smartpar":
            runs = sorted((data_dir / "runs").glob("run_*.json"))
            run = json.loads(runs[-1].read_text()) if runs else {}
            resolved = {r["nodeid"] for r in run.get("reruns", []) if r["suspected_dependency"]}
            tail_tests = [t for t in run.get("tests", []) if t["worker"].startswith("post:")]
            all_failed = {t["nodeid"] for t in run.get("tests", []) if t["outcome"] in FAILED}
            record.update(
                failed_before_rerun=sorted(all_failed),
                failed_after_rerun=sorted(all_failed - resolved),
                tests=len(run.get("tests", tests)),
                smartpar_mode=run.get("mode"),
                predicted_wall_s=run.get("predicted_wall_s"),
                smartpar_wall_s=run.get("wall_time_s"),
                reruns=run.get("reruns", []),
                tail_mode=run.get("tail_mode"),
                separate_pass_tests=len(tail_tests),
            )
            if runs:
                shutil.copy(runs[-1], self.raw / f"{tag}.run.json")
            shutil.rmtree(data_dir, ignore_errors=True)
        (self.raw / f"{tag}.json").write_text(json.dumps(record, indent=2))
        return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "bench" / "results")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--workers", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=20261015)
    parser.add_argument("--detect-runs", type=int, default=20)
    parser.add_argument("--detect-seed", type=int, default=2026)
    parser.add_argument(
        "--warmup", type=int, default=3, help="smartpar runs per m before measuring"
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        help="seconds per pytest run; a run that exceeds it is killed (with its workers) "
        f"and recorded as timed out (profiling gets {PROFILE_TIMEOUT_FACTOR}x)",
    )
    parser.add_argument(
        "--detect-budget",
        type=float,
        default=None,
        help="seconds for --smartpar-detect (passed as --smartpar-detect-budget)",
    )
    args = parser.parse_args(argv)

    cores = physical_cores()
    workers = sorted({m for m in args.workers if m <= cores})
    if dropped := sorted(set(args.workers) - set(workers)):
        print(f"skipping m={dropped}: only {cores} physical cores")
    missing = [m for m in REQUIRED_WORKERS if m not in workers]
    if missing:
        print(f"WARNING: m={missing} (GitHub Actions runner sizes) not measured")

    machine = machine_info()
    if machine["smartpar_dirty"]:
        bar = "!" * 78
        print(
            f"{bar}\nWARNING: uncommitted changes in the tool's code - this series cannot be "
            f"reproduced from smartpar_sha {machine['smartpar_sha'][:12]}:"
        )
        for line in machine["smartpar_dirty_files"]:
            print(f"    {line}")
        print(f"Commit them before a series for the article.\n{bar}")
    series = args.results / f"{datetime.now():%Y%m%d_%H%M%S}_{machine['host']}"
    raw = series / "raw"
    raw.mkdir(parents=True)
    projects = select(load_projects(args.projects), args.only)
    env = {
        **machine,
        "workers": workers,
        "configs": args.configs,
        "repeats": args.repeats,
        "order_seed": args.order_seed,
        "run_timeout_s": args.run_timeout,
        "detect_budget_s": args.detect_budget,
        "projects": {},
    }
    for project in projects:
        _src, venv = project.paths(args.work)
        sha = pinned_sha(project, args.work)
        env["projects"][project.name] = {
            "project_sha": sha,
            "ref": project.ref,
            "local": project.local,
            "status": project.status,
            "versions": versions(venv_python(venv)),
            "pytest_args": project.pytest_args,
            "deselect": project.deselect,
            "ground_truth": project.ground_truth,
        }
        if project.local is not None:
            # A project inside this repository: its source is the repository at
            # smartpar_sha; project_sha is only the workspace snapshot commit.
            env["projects"][project.name]["control_source"] = {
                "path": project.local,
                "smartpar_sha": machine["smartpar_sha"],
                "smartpar_dirty": machine["smartpar_dirty"],
            }
    (series / "env.json").write_text(json.dumps(env, indent=2))

    jobs = jobs_for(args.configs, workers)
    failed: list[str] = []
    for project in projects:
        bench = Bench(project, args.work, raw, args.run_timeout, args.detect_budget)
        print(f"== {project.name}: preparing (profile, detect, warm-up)")
        try:
            prep = bench.prepare(workers, args.detect_runs, args.detect_seed, args.warmup)
        except (BenchError, subprocess.CalledProcessError, OSError) as exc:
            # One broken project must not stop the series for the others.
            failed.append(project.name)
            (bench.raw / "error.txt").write_text(str(exc))
            print(f"   FAILED, skipping the project: {exc}")
            continue
        if prep["detect_budget_hits"] or prep["detect_timed_out"]:
            print(
                f"   WARNING: the detection budget ran out ({prep['detect_budget_hits']} tests "
                "unresolved for lack of budget): undetected order dependencies can make "
                "smartpar runs fail - raise --detect-budget for the article series"
            )
        print(
            f"   profile {prep['profile_s']:.1f}s, detect {prep['detect_s']:.1f}s"
            + (" (timed out)" if prep["detect_timed_out"] else "")
            + f", warm-up {prep['warmup_s']:.1f}s"
        )
        for repeat in range(args.repeats):
            order = shuffled(jobs, args.order_seed, repeat)
            (bench.raw / f"order_r{repeat}.json").write_text(json.dumps(order))
            for position, (config, m) in enumerate(order):
                rec = bench.measure(config, m, repeat, position)
                status = (
                    "TIMEOUT"
                    if rec["timed_out"]
                    else (
                        f"failed {len(rec['failed_before_rerun'])}/{len(rec['failed_after_rerun'])}"
                    )
                )
                print(f"   r{repeat} {config:15s} m={m}: {rec['wall_s']:6.2f}s {status}")
    print(f"series written to {series}")
    if failed:
        print(f"projects skipped after errors: {', '.join(failed)} (see raw/<project>/error.txt)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
