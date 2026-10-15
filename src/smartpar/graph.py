"""Method step 2 (grouping): dependency graph and test groups.

Builds "must not run concurrently" edges from Bernstein's conditions on files
(W(A)∩R(B)=∅, R(A)∩W(B)=∅, W(A)∩W(B)=∅) and order edges from detected
order dependencies, merges connected tests with union-find and orders tests
inside a group topologically.

Not implemented yet (phase 4).
"""
