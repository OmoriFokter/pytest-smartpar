"""Method step 2 (grouping): dependency graph and test groups.

Edges come from two sources only:

* file conflicts, i.e. violations of Bernstein's conditions (Bernstein, 1966)
  ``W(A)∩R(B)=∅``, ``R(A)∩W(B)=∅``, ``W(A)∩W(B)=∅`` - "not concurrently";
  a subtree write of a directory intersects every path below it;
* confirmed order dependencies from flaky.json - "``before`` runs before
  ``after``" (victim before polluter, state setter before brittle test).

State changes found by profiling are candidates for the detector and create
no edges by themselves.

Connected tests form a group (union-find) that runs on one worker; inside a
group tests are sorted topologically by the order edges, ties broken by the
collection order (durations inside one worker do not change the makespan).
Tests on a cycle of order edges, tests to isolate (NOD) and everything that
must run after them go to the isolated tail, executed last and sequentially.
"""

from __future__ import annotations

import bisect
import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from smartpar.models import Conflict, FileAccess, Group, OrderEdge

REASON_FILE = "file"
REASON_OD = "od"


# --- conflicts ---------------------------------------------------------------------


def find_conflicts(file_access: Mapping[str, FileAccess], tests: Iterable[str]) -> list[Conflict]:
    """Pairs of tests violating Bernstein's conditions, one entry per (a, b, resource).

    Uses an inverted index resource -> readers/writers, so only tests sharing a
    resource are compared (not all n^2 pairs). A subtree write (``W_tree``) of
    directory D conflicts with every access to D or a path below it, found by
    a range search over the sorted resources. Tests outside ``tests`` are
    ignored.
    """
    universe = set(tests)
    readers: dict[str, set[str]] = defaultdict(set)
    writers: dict[str, set[str]] = defaultdict(set)  # plain and subtree writes
    tree_writers: dict[str, set[str]] = defaultdict(set)
    for nodeid, access in file_access.items():
        if nodeid not in universe:
            continue
        for resource in access.R:
            readers[resource].add(nodeid)
        for resource in access.W:
            writers[resource].add(nodeid)
        for resource in access.W_tree:
            writers[resource].add(nodeid)
            tree_writers[resource].add(nodeid)

    conflicts: set[Conflict] = set()

    def add(writer: str, other: str, resource: str, other_writes: bool) -> None:
        if writer == other:
            return
        a, b = sorted((writer, other))
        a_writes = a == writer or other_writes
        b_writes = b == writer or other_writes
        conflicts.add(Conflict(a, b, resource, _kind(a_writes, b_writes)))

    for resource, ws in writers.items():
        for w in ws:
            for other in ws | readers.get(resource, set()):
                add(w, other, resource, other in ws)

    paths = sorted(readers.keys() | writers.keys())
    for directory, tws in tree_writers.items():
        # Every path below ``directory`` starts with "directory/" and sorts
        # before "directory0" ('0' follows '/').
        lo = bisect.bisect_left(paths, directory + "/")
        hi = bisect.bisect_left(paths, directory + "0")
        for path in paths[lo:hi]:
            ws = writers.get(path, set())
            for w in tws:
                for other in ws | readers.get(path, set()):
                    add(w, other, path, other in ws)
    return sorted(conflicts)


def _kind(a_writes: bool, b_writes: bool) -> str:
    if a_writes and b_writes:
        return "W-W"
    return "W-R" if a_writes else "R-W"


# --- union-find --------------------------------------------------------------------


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic representative: the smaller nodeid.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo

    def components(self) -> list[list[str]]:
        comps: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            comps[self.find(x)].append(x)
        return sorted(sorted(c) for c in comps.values())


# --- order --------------------------------------------------------------------------


def _cyclic_nodes(nodes: Iterable[str], succ: Mapping[str, set[str]]) -> set[str]:
    """Nodes on a directed cycle (Tarjan's SCC, iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    cyclic: set[str] = set()
    counter = 0
    for start in sorted(nodes):
        if start in index:
            continue
        work = [(start, iter(sorted(succ.get(start, ()))))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(succ.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    cyclic.update(component)
    return cyclic


def _descendants(seeds: Iterable[str], succ: Mapping[str, set[str]]) -> set[str]:
    seen = set(seeds)
    todo = list(seen)
    while todo:
        for child in succ.get(todo.pop(), ()):
            if child not in seen:
                seen.add(child)
                todo.append(child)
    return seen


def topo_order(
    nodes: Iterable[str], edges: Iterable[OrderEdge], rank: Mapping[str, int]
) -> list[str]:
    """Kahn's algorithm; ready tests in collection order (``rank``), then nodeid.

    Durations do not matter here: inside one worker they do not change the
    makespan. If a cycle remains, the smallest remaining nodeid is taken to
    keep going (only happens in the isolated tail; groups have no cycles).
    """
    members = set(nodes)
    succ: dict[str, set[str]] = defaultdict(set)
    indeg = dict.fromkeys(members, 0)
    for e in edges:
        if e.before in members and e.after in members and e.after not in succ[e.before]:
            succ[e.before].add(e.after)
            indeg[e.after] += 1

    unranked = len(rank)

    def key(n: str) -> tuple[int, str]:
        return (rank.get(n, unranked), n)

    ready = [key(n) for n in members if indeg[n] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    done: set[str] = set()
    while len(order) < len(members):
        if not ready:  # cycle: break it deterministically
            n = min(m for m in members if m not in done)
            indeg[n] = 0
            ready.append(key(n))
        _rank, n = heapq.heappop(ready)
        if n in done:
            continue
        done.add(n)
        order.append(n)
        for child in sorted(succ[n]):
            if child in done:
                continue
            indeg[child] -= 1
            if indeg[child] == 0:
                heapq.heappush(ready, key(child))
    return order


# --- graph --------------------------------------------------------------------------


@dataclass
class GraphResult:
    conflicts: list[Conflict]
    order_edges: list[OrderEdge]
    groups: list[Group]  # two or more tests
    singles: list[str]  # independent tests
    tail: list[str]  # isolated tail, in execution order
    cyclic: list[str] = field(default_factory=list)  # unresolved: on an order cycle


def build(
    tests: Iterable[str],
    file_access: Mapping[str, FileAccess],
    order_edges: Iterable[OrderEdge],
    durations: Mapping[str, float],
    isolate: Iterable[str] = (),
    collection_index: Mapping[str, int] | None = None,
) -> GraphResult:
    """Group ``tests``; edges referring to tests outside ``tests`` are dropped.

    ``collection_index`` (pytest's collection order) breaks ties of the order
    inside groups and the tail; tests without an index come last, by nodeid.
    """
    rank = collection_index or {}
    universe = sorted(set(tests))
    members = set(universe)
    conflicts = find_conflicts(file_access, universe)
    edges = sorted(
        {
            e
            for e in order_edges
            if e.before in members and e.after in members and e.before != e.after
        }
    )
    succ: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        succ[e.before].add(e.after)

    cyclic = _cyclic_nodes(members, succ)
    # Whatever must run after a tail test goes to the tail as well, otherwise
    # it would run before it.
    tail_set = _descendants(cyclic | (set(isolate) & members), succ)
    tail = topo_order(tail_set, edges, rank)

    uf = UnionFind(t for t in universe if t not in tail_set)
    for c in conflicts:
        if c.a not in tail_set and c.b not in tail_set:
            uf.union(c.a, c.b)
    for e in edges:
        if e.before not in tail_set and e.after not in tail_set:
            uf.union(e.before, e.after)

    groups: list[Group] = []
    singles: list[str] = []
    for component in uf.components():
        if len(component) == 1:
            singles.append(component[0])
            continue
        comp = set(component)
        reasons = []
        if any(c.a in comp and c.b in comp for c in conflicts):
            reasons.append(REASON_FILE)
        if any(e.before in comp and e.after in comp for e in edges):
            reasons.append(REASON_OD)
        groups.append(
            Group(
                id=f"g{len(groups)}",
                tests=component,
                order=topo_order(component, edges, rank),
                est_duration=sum(durations.get(t, 0.0) for t in component),
                reasons=reasons,
            )
        )
    return GraphResult(conflicts, edges, groups, singles, tail, sorted(cyclic))
