"""Method step 1 (profiling): per-test file access and in-process state changes.

Records the read set R and write set W of files for every test via
``sys.addaudithook`` and takes snapshots of process state (project module
globals, ``os.environ``, cwd, ``sys.path``, logging) before and after each test
to find tests that change state without restoring it (polluter candidates).

Not implemented yet (phase 3).
"""
