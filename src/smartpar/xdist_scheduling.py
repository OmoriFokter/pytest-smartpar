"""Method step 4 (execution): pytest-xdist scheduler that executes plan.json.

Provides a custom scheduler via ``pytest_xdist_make_scheduler`` that sends each
worker its planned list of tests in the planned order, with a fallback to
``--dist loadgroup`` when the custom scheduler cannot be used.

Not implemented yet (phase 5).
"""
