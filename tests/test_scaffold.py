"""Phase 1: the package is installed, the plugin is registered and every module is documented."""

from __future__ import annotations

import importlib
import re
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


def test_console_script_installed() -> None:
    exe = shutil.which("smartpar", path=str(Path(sys.executable).parent))
    assert exe is not None
    proc = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
    assert proc.stdout.startswith("smartpar ")


def test_supported_xdist_range_matches_pyproject() -> None:
    from smartpar.plugin import SUPPORTED_XDIST

    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    low, high = re.search(r'"pytest-xdist>=(\d+\.\d+),<(\d+\.\d+)"', text).groups()
    as_tuple = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
    assert (as_tuple(low), as_tuple(high)) == SUPPORTED_XDIST


def test_unsupported_xdist_selects_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import xdist

    from smartpar import plugin

    assert plugin.custom_scheduler_problem() is None  # the installed version is supported
    monkeypatch.setattr(xdist, "__version__", "3.9.0")
    assert "outside the tested range" in plugin.custom_scheduler_problem()
    monkeypatch.setattr(xdist, "__version__", "3.2.1")
    assert "outside the tested range" in plugin.custom_scheduler_problem()


def test_readme_documents_every_public_option(pytester: pytest.Pytester) -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    config = pytester.parseconfig()
    group = next(g for g in config._parser._groups if g.name == "smartpar")
    internal = {"--smartpar-child-report", "--smartpar-child-audit"}
    flags = {name for opt in group.options for name in opt.names() if name.startswith("--")}
    assert flags - internal, "no options found"
    for flag in sorted(flags - internal):
        assert f"`{flag}" in readme, f"{flag} is not documented in README.md"
    for ini in (
        "smartpar_ignore_state",
        "smartpar_max_pair_checks",
        "smartpar_max_orders_per_victim",
    ):
        assert f"`{ini}`" in readme
    for command in ("smartpar merge", "smartpar show", "smartpar report"):
        assert command in readme
