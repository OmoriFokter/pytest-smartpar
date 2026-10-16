"""Experimental stand: clone the projects at pinned SHAs and install them in venvs.

    python bench/setup_projects.py [--only NAME ...]          # set up
    python bench/setup_projects.py --survey [--only NAME ...] # + one serial run each

Layout (``../smartpar-work`` next to the repository by default; never inside
it, see ``DEFAULT_WORK``)::

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
# Outside the repository: pytest looks for its configuration upwards from the
# tests, so a project without its own (e.g. networkx) would pick up this
# repository's pyproject.toml - rootdir, addopts - and the detector would copy
# the whole repository for every child run.
DEFAULT_WORK = REPO_ROOT.parent / f"{REPO_ROOT.name}-work"
SNAPSHOT_IGNORE = shutil.ignore_patterns(
    ".smartpar", "output", "__pycache__", ".pytest_cache", ".venv", "venv"
)
# Excluded from `git clean` even if someone puts a venv inside src.
CLEAN_KEEP = (".venv", "venv")
# Every xdist worker must collect the tests in the same order, otherwise xdist
# refuses to run ("Different tests were collected"). Parametrization over a
# set breaks this in two ways: the order of a set depends on the hash seed,
# random in every process by default, and a set constant compiled into a .pyc
# by one worker and loaded by another may iterate in a different order even
# with the same seed. Hence a fixed seed and no .pyc writes for every
# benchmarked run; the same environment for every configuration keeps the
# comparison fair (``clean_state`` removes __pycache__ before each run anyway).
BENCH_HASH_SEED = "0"


def bench_env(**extra: str) -> dict[str, str]:
    """Environment of a benchmarked pytest run (child processes inherit it)."""
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONHASHSEED": BENCH_HASH_SEED,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("SMARTPAR_BENCH_TIMING", None)
    return {**env, **extra}


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
    # Interpreter version of the project's venv, e.g. "3.12" (from the
    # top-level `python` of projects.yaml unless overridden per project).
    python: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def paths(self, work: Path) -> tuple[Path, Path]:
        base = work / self.name
        return base / "src", base / "venv"


def load_projects(path: Path = DEFAULT_PROJECTS) -> list[Project]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    default_python = data.get("python")
    projects = []
    for entry in data["projects"]:
        entry = {"python": default_python, **entry}
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


def python_version(python: Path) -> str:
    """'3.12' for the given interpreter."""
    code = "import sys; print('%d.%d' % sys.version_info[:2])"
    return run([str(python), "-c", code], capture_output=True).stdout.strip()


def create_venv(venv: Path, version: str | None) -> None:
    """A venv with the requested Python (uv can download it); else this interpreter."""
    uv = shutil.which("uv")
    if version is None:
        run([sys.executable, "-m", "venv", str(venv)])
    elif uv:
        run([uv, "venv", "-q", "--python", version, str(venv)])
    elif interpreter := shutil.which(f"python{version}"):
        run([interpreter, "-m", "venv", str(venv)])
    else:
        raise RuntimeError(f"Python {version} not found: install it or uv (it can fetch it)")


def setup_venv(project: Project, work: Path) -> Path:
    src, venv = project.paths(work)
    python = venv_python(venv)
    if python.exists() and project.python and python_version(python) != project.python:
        print(f"   venv has Python {python_version(python)}, need {project.python}: recreating")
        shutil.rmtree(venv)
    if not python.exists():
        create_venv(venv, project.python)
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


# Paths whose uncommitted changes make a series irreproducible from smartpar_sha.
TOOL_PATHS = ("src", "bench", "examples", "pyproject.toml")


def smartpar_dirty() -> list[str]:
    """Uncommitted (incl. untracked) changes in the tool's code, as `git status` lines."""
    out = git(REPO_ROOT, "status", "--porcelain", "--untracked-files=all", "--", *TOOL_PATHS)
    return [line for line in out.splitlines() if line.strip()]


def machine_info() -> dict[str, Any]:
    smartpar_sha = git(REPO_ROOT, "rev-parse", "HEAD")
    dirty = smartpar_dirty()
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
        "smartpar_dirty": bool(dirty),
        "smartpar_dirty_files": dirty,
        "argv": sys.argv,
    }


# --- survey ------------------------------------------------------------------------------


def survey(project: Project, work: Path) -> dict[str, Any]:
    """One serial run: number of tests, wall time, failures out of the box."""
    src, venv = project.paths(work)
    clean_state(project, work)
    timing = work / project.name / "survey_timing.jsonl"
    env = bench_env(SMARTPAR_BENCH_TIMING=str(timing))
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


def check_work_dir(work: Path) -> Path:
    """The workspace, absolute; refuse one inside this repository (see DEFAULT_WORK)."""
    work = work.resolve()
    if work == REPO_ROOT or REPO_ROOT in work.parents:
        raise SystemExit(
            f"--work {work} is inside the repository {REPO_ROOT}: projects without their "
            "own pytest configuration would use the repository's one. Use a directory "
            f"outside it (default: {DEFAULT_WORK})."
        )
    return work


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--survey", action="store_true", help="one serial run per project")
    args = parser.parse_args(argv)
    args.work = check_work_dir(args.work)
    projects = select(load_projects(args.projects), args.only)
    results: list[dict[str, Any]] = []
    out = REPO_ROOT / "bench" / "results" / f"survey_{datetime.now():%Y%m%d_%H%M%S}.json"
    failed = []
    for project in projects:
        print(f"== {project.name}")
        try:
            sha = prepare_source(project, args.work)
            python = setup_venv(project, args.work)
            print(f"   at {sha}; {versions(python)}")
            if args.survey:
                r = survey(project, args.work)
                print(f"   {r['tests']} tests, {r['wall_s']} s, {r['failures']} failures")
        except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
            # One broken project must not lose the others' results.
            failed.append(project.name)
            r = {"project": project.name, "error": str(exc)}
            print(f"   FAILED: {exc}")
        if args.survey:
            results.append(r)
            # Written after every project, so an interruption keeps what is done.
            out.write_text(json.dumps({"machine": machine_info(), "projects": results}, indent=2))
    if args.survey:
        print(f"survey written to {out}")
    if failed:
        print(f"set-up failed for: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
