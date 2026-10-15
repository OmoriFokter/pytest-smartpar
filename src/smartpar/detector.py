"""Method step 1 (profiling, continued): detection of order-dependent tests.

Runs the suite in child pytest processes in random orders, classifies failing
tests as NOD / order-dependent victim / brittle, locates polluters and state
setters by delta debugging and confirms every found pair by a re-run.

Not implemented yet (phase 6).
"""
