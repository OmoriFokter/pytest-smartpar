"""pytest-smartpar: dependency-aware parallel execution of unit tests in CI.

Implements the method from the article "A Method for Parallel Execution of Unit
Tests in Continuous Integration Systems": profiling (file R/W sets, in-process
state changes), detection of order-dependent tests, grouping of dependent tests,
LPT scheduling of groups onto m workers, execution and history recording.
"""

__version__ = "0.1.0.dev0"
