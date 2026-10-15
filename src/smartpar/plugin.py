"""Integration of the method into pytest: command-line options and mode dispatch.

Registered as the ``smartpar`` plugin through the ``pytest11`` entry point.
Modes: ``--smartpar-profile`` (step 1, profiling), ``--smartpar-detect``
(step 1, order-dependency detection) and ``--smartpar`` (steps 2-5: grouping,
scheduling, execution and recording).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smartpar.profiler import DEFAULT_IGNORE_STATE, ProfilerPlugin
from smartpar.report import RunCollector

# True when smartpar switched off pytest-randomly's reordering for this session.
# Written to plan.json / report.csv as ``randomly_reorder_disabled``.
RANDOMLY_REORDER_DISABLED = pytest.StashKey[bool]()

RANDOMLY_REORDER_WARNING = (
    "smartpar: pytest-randomly test reordering disabled (--randomly-dont-reorganize) "
    "to keep the planned/profiled order; random seeding stays active"
)

# Mode flag -> implementation phase from SPEC.md. Used to fail loudly instead of
# silently ignoring a mode that does not work yet.
_UNIMPLEMENTED_MODES = {
    "smartpar_detect": ("--smartpar-detect", 6),
    "smartpar": ("--smartpar", 5),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "smartpar_ignore_state",
        type="linelist",
        default=list(DEFAULT_IGNORE_STATE),
        help="fnmatch patterns of state keys ignored by --smartpar-profile "
        "(replaces the default list: " + ", ".join(DEFAULT_IGNORE_STATE) + ")",
    )
    group = parser.getgroup("smartpar", "dependency-aware parallel test execution")
    group.addoption(
        "--smartpar",
        action="store_true",
        default=False,
        help="Run tests according to the smartpar plan (use with -n).",
    )
    group.addoption(
        "--smartpar-profile",
        action="store_true",
        default=False,
        help="Profile tests: durations, file R/W sets and state changes.",
    )
    group.addoption(
        "--smartpar-detect",
        action="store_true",
        default=False,
        help="Detect order-dependent tests with randomized runs in child processes.",
    )
    group.addoption(
        "--smartpar-runs",
        type=int,
        default=20,
        help="Number of random-order runs for --smartpar-detect (default: 20).",
    )
    group.addoption(
        "--smartpar-seed",
        type=int,
        default=None,
        help="Base seed for --smartpar-detect (default: chosen and recorded).",
    )
    group.addoption(
        "--smartpar-detect-budget",
        type=float,
        default=None,
        help="Time budget in seconds for --smartpar-detect.",
    )
    group.addoption(
        "--smartpar-dir",
        default=".smartpar",
        help="Directory for smartpar data files (default: .smartpar).",
    )
    group.addoption(
        "--smartpar-alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor for test durations (default: 0.3).",
    )


def _randomly_active(config: pytest.Config) -> bool:
    return config.pluginmanager.hasplugin("randomly")


def apply_randomly_policy(config: pytest.Config) -> None:
    """Reconcile pytest-randomly with the selected smartpar mode.

    --smartpar-detect needs random orders, so pytest-randomly is mandatory.
    --smartpar and --smartpar-profile need a controlled order, so only the
    reordering of pytest-randomly is switched off (seeding of ``random`` etc.
    is left as configured by the user).
    """
    if config.getoption("smartpar_detect") and not _randomly_active(config):
        raise pytest.UsageError(
            "smartpar: --smartpar-detect requires pytest-randomly to shuffle test order, "
            "but it is not installed or disabled (-p no:randomly). "
            "Install it with: pip install pytest-randomly"
        )
    disabled = False
    if (
        (config.getoption("smartpar") or config.getoption("smartpar_profile"))
        and _randomly_active(config)
        and config.getoption("randomly_reorganize")
    ):
        config.option.randomly_reorganize = False
        disabled = True
    config.stash[RANDOMLY_REORDER_DISABLED] = disabled


# trylast: the terminal reporter is registered by pytest's own pytest_configure.
@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    apply_randomly_policy(config)
    is_controller = not hasattr(config, "workerinput")
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if config.stash[RANDOMLY_REORDER_DISABLED] and is_controller and terminal is not None:
        terminal.write_line(RANDOMLY_REORDER_WARNING, yellow=True)

    for dest, (flag, phase) in _UNIMPLEMENTED_MODES.items():
        if config.getoption(dest):
            raise pytest.UsageError(
                f"smartpar: {flag} is not implemented yet (planned in phase {phase})."
            )

    if config.getoption("smartpar_profile"):
        _start_profiling(config)


class _XdistProfileHooks:
    """xdist-only hooks; registered only when xdist is active (unknown hooks fail validation)."""

    def __init__(self, profiler: ProfilerPlugin) -> None:
        self.profiler = profiler

    def pytest_configure_node(self, node: object) -> None:
        self.profiler.configure_node(node)


def _smartpar_dir(config: pytest.Config) -> Path:
    path = Path(config.getoption("smartpar_dir"))
    return path if path.is_absolute() else Path(config.invocation_params.dir) / path


def is_xdist_controller(config: pytest.Config) -> bool:
    """Same predicate xdist uses to create its DSession (registration order of
    the two trylast pytest_configure hooks is not guaranteed)."""
    if hasattr(config, "workerinput") or config.getoption("collectonly"):
        return False
    return config.getoption("dist", default="no") != "no" and bool(
        config.getoption("tx", default=None)
    )


def _xdist_workers(config: pytest.Config) -> int:
    if not is_xdist_controller(config):
        return 1
    try:
        return int(config.getoption("numprocesses") or 1)
    except (TypeError, ValueError):
        return 1


def _start_profiling(config: pytest.Config) -> None:
    smartpar_dir = _smartpar_dir(config)
    profiler = ProfilerPlugin(config, smartpar_dir, is_xdist_controller(config))
    config.pluginmanager.register(profiler, "smartpar-profiler")
    if hasattr(config, "workerinput"):
        return
    if profiler.is_xdist_controller:
        config.pluginmanager.register(_XdistProfileHooks(profiler), "smartpar-profiler-xdist")
    collector = RunCollector(
        mode="profile",
        workers=_xdist_workers(config),
        randomly_reorder_disabled=config.stash[RANDOMLY_REORDER_DISABLED],
        smartpar_dir=smartpar_dir,
        merge_alpha=config.getoption("smartpar_alpha"),
    )
    config.pluginmanager.register(collector, "smartpar-run-collector")
