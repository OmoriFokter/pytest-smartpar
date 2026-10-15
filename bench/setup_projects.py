"""Experimental stand: clone the projects at pinned SHAs and install them in venvs.

    python bench/setup_projects.py [--only NAME ...]          # set up
    python bench/setup_projects.py --survey [--only NAME ...] # + one serial run each

Layout (``bench/work`` by default, git-ignored)::

    work/<name>/src    the project at its SHA (a clone, or a committed snapshot
                       of a project inside this repository)
    work/<name>/venv   its virtualenv: outside src, so ``git clean -xdf`` in src
                       never touches it

``clean_state`` restores src exactly to the pinned commit
(``git checkout --force`` + ``git clean -xdf``): leftover files of earlier runs
(e.g. outputs written by tests) change results, so every benchmark step
starts from it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTS = REPO_ROOT / "bench" / "projects.yaml"
DEFAULT_WORK = REPO_ROOT / "bench" / "work"
SNAPSHOT_IGNORE = shutil.ignore_patterns(
    ".smartpar", "output", "__pycache__", ".pytest_cache", ".venv", "venv"
)
# Excluded from `git clean` even if someone puts a venv inside src.
CLEAN_KEEP = (".venv", "venv")


@dataclass
class Project:
    name: str
    tests: str
    install: list[str]
    pytest_args: list[str]
    deselect: list[str]
    repo: str | None = None
    sha: str | None = None
    ref: str | None = None
    local: str | None = None
    ground_truth: str | None = None
    status: str = "selected"
    extra: dict[str, Any] = field(default_factory=dict)

    def paths(self, work: Path) -> tuple[Path, Path]:
        base = work / self.name
        return base / "src", base / "venv"


def load_projects(path: Path = DEFAULT_PROJECTS) -> list[Project]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    projects = []
    for entry in data["projects"]:
        entry = dict(entry)
        known = {k: entry.pop(k) for k in list(entry) if k in Project.__dataclass_fields__}
        project = Project(**known, extra=entry)
        if project.local is None and (
            not project.repo or not project.sha or len(project.sha) != 40
        ):
            raise ValueError(f"{project.name}: a git project needs repo and a 40-char sha")
        projects.append(project)
    return projects


def select(projects: list[Project], only: list[str] | None) -> list[Project]:
    if not only:
        return projects
    unknown = set(only) - {p.name for p in projects}
    if unknown:
        raise SystemExit(f"unknown project(s): {', '.join(sorted(unknown))}")
    return [p for p in projects if p.name in only]


def run(cmd: list[str], cwd: Path | None = None, **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, **kwargs)


def git(src: Path, *args: str) -> str:
    return run(["git", "-C", str(src), *args], capture_output=True).stdout.strip()


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# --- workspace ---------------------------------------------------------------------


def prepare_source(project: Project, work: Path) -> str:
    """src at the pinned commit; returns its SHA."""
    src, _venv = project.paths(work)
    if project.local is not None:
        if src.exists():
            shutil.rmtree(src)
        shutil.copytree(REPO_ROOT / project.local, src, ignore=SNAPSHOT_IGNORE)
        run(["git", "init", "-q"], cwd=src)
        run(["git", "add", "-A"], cwd=src)
        run(
            [
                "git",
                "-c",
                "user.name=bench",
                "-c",
                "user.email=bench@localhost",
                "commit",
                "-q",
                "-m",
                "snapshot",
            ],
            cwd=src,
        )
    else:
        if not (src / ".git").exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "-q", "--no-checkout", project.repo, str(src)])
        try:
            git(src, "cat-file", "-e", f"{project.sha}^{{commit}}")
        except subprocess.CalledProcessError:
            git(src, "fetch", "-q", "origin", project.sha)
        git(src, "checkout", "-q", "--force", "--detach", project.sha)
    clean_state(project, work)
    return git(src, "rev-parse", "HEAD")


def pinned_sha(project: Project, work: Path) -> str:
    src, _venv = project.paths(work)
    return project.sha or git(src, "rev-parse", "HEAD")


def clean_state(project: Project, work: Path) -> None:
    """src exactly at the pinned commit: no untracked, ignored or modified files."""
    src, _venv = project.paths(work)
    sha = pinned_sha(project, work)
    git(src, "checkout", "-q", "--force", "--detach", sha)
    excludes = [arg for keep in CLEAN_KEEP for arg in ("-e", keep)]
    git(src, "clean", "-xdfq", *excludes)
    head = git(src, "rev-parse", "HEAD")
    if head != sha:
        raise RuntimeError(f"{project.name}: HEAD {head} != pinned {sha}")


def installer() -> list[str] | None:
    """uv if available (much faster), else pip inside the venv."""
    uv = shutil.which("uv")
    return [uv, "pip", "install", "--python"] if uv else None


def setup_venv(project: Project, work: Path) -> Path:
    src, venv = project.paths(work)
    python = venv_python(venv)
    if not python.exists():
        run([sys.executable, "-m", "venv", str(venv)])
    for command in project.install:
        cmd = command.format(python=python).split()
        uv = installer()
        if uv and cmd[1:4] == ["-m", "pip", "install"]:
            cmd = [*uv, str(python), *cmd[4:]]
        run(cmd, cwd=src)
    # smartpar itself (with pytest, pytest-xdist, pytest-randomly in its ranges)
    uv = installer()
    base = [*uv, str(python)] if uv else [str(python), "-m", "pip", "install"]
    run([*base, "-e", str(REPO_ROOT)])
    return python


def versions(python: Path) -> dict[str, str]:
    code = (
        "import json, sys, importlib.metadata as m\n"
        "out = {'python': sys.version.split()[0]}\n"
        "for d in ('pytest', 'pytest-xdist', 'pytest-randomly', 'pytest-smartpar'):\n"
        "    try: out[d] = m.version(d)\n"
        "    except m.PackageNotFoundError: out[d] = None\n"
        "print(json.dumps(out))"
    )
    return json.loads(run([str(python), "-c", code], capture_output=True).stdout)


# --- machine description -------------------------------------------------------------


def physical_cores() -> int:
    """Distinct (physical id, core id) pairs from /proc/cpuinfo; logical count otherwise."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return os.cpu_count() or 1
    pairs, phys = set(), None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            phys = value.strip()
        elif key == "core id":
            pairs.add((phys, value.strip()))
    return len(pairs) or os.cpu_count() or 1


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def machine_info() -> dict[str, Any]:
    smartpar_sha = git(REPO_ROOT, "rev-parse", "HEAD")
    dirty = bool(git(REPO_ROOT, "status", "--porcelain", "--untracked-files=no"))
    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "cpu_model": cpu_model(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "ram_bytes": ram_bytes(),
        "os": platform.platform(),
        "runner_python": sys.version.split()[0],
        "smartpar_sha": smartpar_sha,
        "smartpar_dirty": dirty,
        "argv": sys.argv,
    }


# --- survey ------------------------------------------------------------------------------


def survey(project: Project, work: Path) -> dict[str, Any]:
    """One serial run: number of tests, wall time, failures out of the box."""
    src, venv = project.paths(work)
    clean_state(project, work)
    timing = work / project.name / "survey_timing.jsonl"
    env = {**os.environ, "SMARTPAR_BENCH_TIMING": str(timing), "PYTHONPATH": str(REPO_ROOT)}
    cmd = [
        str(venv_python(venv)),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:xdist",
        "-p",
        "no:randomly",
        "-p",
        "bench.timing_plugin",
        project.tests,
        *project.pytest_args,
        *[f"--deselect={d}" for d in project.deselect],
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=src, env=env, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t0
    tests = (
        [json.loads(line) for line in timing.read_text().splitlines()] if timing.exists() else []
    )
    failed = [t["nodeid"] for t in tests if t["outcome"] in ("failed", "error")]
    clean_state(project, work)
    return {
        "project": project.name,
        "tests": len(tests),
        "wall_s": round(wall, 2),
        "failures": len(failed),
        "failed": failed,
        "exit_code": proc.returncode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--survey", action="store_true", help="one serial run per project")
    args = parser.parse_args(argv)
    projects = select(load_projects(args.projects), args.only)
    results = []
    for project in projects:
        print(f"== {project.name}")
        sha = prepare_source(project, args.work)
        python = setup_venv(project, args.work)
        print(f"   at {sha}; {versions(python)}")
        if args.survey:
            results.append(survey(project, args.work))
            r = results[-1]
            print(f"   {r['tests']} tests, {r['wall_s']} s, {r['failures']} failures")
    if args.survey:
        out = REPO_ROOT / "bench" / "results" / f"survey_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps({"machine": machine_info(), "projects": results}, indent=2))
        print(f"survey written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
