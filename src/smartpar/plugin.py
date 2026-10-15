"""Integration of the method into pytest: command-line options and mode dispatch.

Registered as the ``smartpar`` plugin through the ``pytest11`` entry point.
Modes: ``--smartpar-profile`` (step 1, profiling), ``--smartpar-detect``
(step 1, order-dependency detection) and ``--smartpar`` (steps 2-5: grouping,
scheduling, execution and recording).
"""

from __future__ import annotations

import pytest

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


def pytest_configure(config: pytest.Config) -> None:
    for dest, (flag, phase) in _UNIMPLEMENTED_MODES.items():
        if config.getoption(dest):
            raise pytest.UsageError(
                f"smartpar: {flag} is not implemented yet (planned in phase {phase})."
            )
