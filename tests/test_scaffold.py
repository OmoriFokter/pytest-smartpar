"""Phase 1: the package is installed, the plugin is registered and every module is documented."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MODULES = [
    "smartpar",
    "smartpar.plugin",
    "smartpar.profiler",
    "smartpar.detector",
    "smartpar.graph",
    "smartpar.scheduler",
    "smartpar.xdist_scheduling",
    "smartpar.history",
    "smartpar.report",
    "smartpar.models",
    "smartpar.cli",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_has_docstring(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__ and module.__doc__.strip()


def test_plugin_registered_via_entry_point(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(["*smartpar*", "*--smartpar-profile*", "*--smartpar-dir*"])
    assert result.ret == 0


def test_plugin_is_inactive_without_flags(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize("flag", ["--smartpar", "--smartpar-profile", "--smartpar-detect"])
def test_unimplemented_mode_fails_loudly(pytester: pytest.Pytester, flag: str) -> None:
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest(flag)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*not implemented yet*"])


def test_console_script_installed() -> None:
    exe = shutil.which("smartpar", path=str(Path(sys.executable).parent))
    assert exe is not None
    proc = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
    assert proc.stdout.startswith("smartpar ")
