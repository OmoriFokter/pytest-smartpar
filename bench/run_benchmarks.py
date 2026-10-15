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
import json
import os
import random
import shutil
import subprocess
import sys
import time
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
REQUIRED_WORKERS = (2, 4)  # standard GitHub Actions runners
FAILED = ("failed", "error")


def config_args(config: str, workers: int, data_dir: Path) -> list[str]:
    if config == "serial":
        return ["-p", "no:xdist"]
    if config == "xdist-load":
        return ["-n", str(workers), "--dist", "load"]
    if config == "xdist-loadfile":
        return ["-n", str(workers), "--dist", "loadfile"]
    if config == "smartpar":
        return ["-n", str(workers), "--smartpar", "--smartpar-dir", str(data_dir)]
    raise ValueError(config)


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
    def __init__(self, project: Project, work: Path, raw: Path) -> None:
        self.project = project
        self.work = work
        self.src, venv = project.paths(work)
        self.python = venv_python(venv)
        self.raw = raw / project.name
        self.raw.mkdir(parents=True, exist_ok=True)
        self.master = self.raw / "smartpar_master"  # prepared data, never mutated later

    def pytest(
        self, args: list[str], timing: Path | None = None, timeout: float | None = None
    ) -> tuple[float, int, str]:
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
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=self.src,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return time.perf_counter() - t0, proc.returncode, proc.stdout[-4000:]

    # --- preparation ---------------------------------------------------------------

    def prepare(
        self, workers: list[int], detect_runs: int, detect_seed: int, warmup: int
    ) -> dict[str, Any]:
        if self.master.exists():
            shutil.rmtree(self.master)
        data = ["--smartpar-dir", str(self.master)]
        clean_state(self.project, self.work)
        # The detector's runs re-enable pytest-randomly themselves.
        profile_s, profile_rc, _ = self.pytest(["--smartpar-profile", *data])
        shutil.copy(self.master / "dependencies.json", self.raw / "deps_after_profile.json")
        detect_s, detect_rc, _ = self.pytest(
            [
                "--smartpar-detect",
                "-p",
                "randomly",
                "--smartpar-runs",
                str(detect_runs),
                "--smartpar-seed",
                str(detect_seed),
                *data,
            ]
        )
        warmup_s = 0.0
        for m in workers:
            for _ in range(warmup):
                clean_state(self.project, self.work)
                seconds, _rc, _ = self.pytest(["-n", str(m), "--smartpar", *data])
                warmup_s += seconds
                self._merge(self.master)
        clean_state(self.project, self.work)
        prep = {
            "profile_s": profile_s,
            "profile_exit_code": profile_rc,
            "detect_s": detect_s,
            "detect_exit_code": detect_rc,
            "detect_runs": detect_runs,
            "detect_seed": detect_seed,
            "warmup_runs_per_m": warmup,
            "warmup_s": warmup_s,
            "overhead_s": profile_s + detect_s,  # one-time cost of the method
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
        wall, rc, tail = self.pytest(config_args(config, m, data_dir), timing=timing)
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
    args = parser.parse_args(argv)

    cores = physical_cores()
    workers = sorted({m for m in args.workers if m <= cores})
    if dropped := sorted(set(args.workers) - set(workers)):
        print(f"skipping m={dropped}: only {cores} physical cores")
    missing = [m for m in REQUIRED_WORKERS if m not in workers]
    if missing:
        print(f"WARNING: m={missing} (GitHub Actions runner sizes) not measured")

    machine = machine_info()
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
        "projects": {},
    }
    for project in projects:
        _src, venv = project.paths(args.work)
        env["projects"][project.name] = {
            "sha": pinned_sha(project, args.work),
            "ref": project.ref,
            "local": project.local,
            "status": project.status,
            "versions": versions(venv_python(venv)),
            "pytest_args": project.pytest_args,
            "deselect": project.deselect,
            "ground_truth": project.ground_truth,
        }
    (series / "env.json").write_text(json.dumps(env, indent=2))

    jobs = jobs_for(args.configs, workers)
    for project in projects:
        bench = Bench(project, args.work, raw)
        print(f"== {project.name}: preparing (profile, detect, warm-up)")
        prep = bench.prepare(workers, args.detect_runs, args.detect_seed, args.warmup)
        print(
            f"   profile {prep['profile_s']:.1f}s, detect {prep['detect_s']:.1f}s, "
            f"warm-up {prep['warmup_s']:.1f}s"
        )
        for repeat in range(args.repeats):
            order = shuffled(jobs, args.order_seed, repeat)
            (bench.raw / f"order_r{repeat}.json").write_text(json.dumps(order))
            for position, (config, m) in enumerate(order):
                rec = bench.measure(config, m, repeat, position)
                print(
                    f"   r{repeat} {config:15s} m={m}: {rec['wall_s']:6.2f}s "
                    f"failed {len(rec['failed_before_rerun'])}/{len(rec['failed_after_rerun'])}"
                )
    print(f"series written to {series}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
