"""Integration of the method into pytest: command-line options and mode dispatch.

Registered as the ``smartpar`` plugin through the ``pytest11`` entry point.
Modes: ``--smartpar-profile`` (step 1, profiling), ``--smartpar-detect``
(step 1, order-dependency detection) and ``--smartpar`` (steps 2-5: grouping,
scheduling, execution and recording).
"""

from __future__ import annotations

import pytest

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
    "smartpar_profile": ("--smartpar-profile", 3),
    "smartpar_detect": ("--smartpar-detect", 6),
    "smartpar": ("--smartpar", 5),
}


def pytest_addoption(parser: pytest.Parser) -> None:
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
