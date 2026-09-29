"""Is the middle case of the ledger trichotomy empty?

prop:splicetree attains 6m+1-2s when the merges can be routed through a forest
F = union_q T_q of max degree <= 3, where T_q is a spanning tree on the blocks
containing qubit q.  Since |F| = sum_q (d_q - 1) = s, a forest needs s <= m-1.

rem:ledgerfails shows s >= m fails.  That leaves the middle case:

    s <= m-1, but EVERY choice of trees forces a node of degree >= 4
    (or forces a cycle).

This script asks whether that case is ever non-empty.  If it is always empty, the
trichotomy collapses to a dichotomy -- the ledger is attained exactly when
s <= m-1, i.e. n >= 2m+1 -- and open problem (i) closes.

Pure combinatorics: no decoding, so it runs well past the n <= 6 decoding limit.
"""

import itertools
import random


def is_linear(T):
    return all(len(set(a) & set(b)) <= 1 for a, b in itertools.combinations(T, 2))


def spanning_trees(nodes):
    """All spanning trees on a small node set, as edge tuples (brute force)."""
    nodes = list(nodes)
    k = len(nodes)
    if k <= 1:
        return [()]
    all_edges = list(itertools.combinations(nodes, 2))
    out = []
    for E in itertools.combinations(all_edges, k - 1):
        parent = {v: v for v in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ok = True
        for a, b in E:
            ra, rb = find(a), find(b)
            if ra == rb:
                ok = False
                break
            parent[ra] = rb
        if ok:
            out.append(E)
    return out


def acyclic(edges, m):
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
    return True


def routable(T, n, maxdeg=3):
    """Can the merges be routed through a forest of max degree <= maxdeg?"""
    m = len(T)
    per_qubit = []
    for q in range(n):
        Bq = [i for i, S in enumerate(T) if q in S]
        if len(Bq) >= 2:
            per_qubit.append(spanning_trees(Bq))
    if not per_qubit:
        return True, 0
    for choice in itertools.product(*per_qubit):
        edges = [e for tree in choice for e in tree]
        if not acyclic(edges, m):
            continue
        deg = [0] * m
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        if max(deg) <= maxdeg:
            return True, max(deg)
    return False, None


def components(T):
    """Connected components of the block-overlap graph, as index lists."""
    m = len(T)
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in itertools.combinations(range(m), 2):
        if set(T[i]) & set(T[j]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    comp = {}
    for i in range(m):
        comp.setdefault(find(i), []).append(i)
    return list(comp.values())


def forest_possible_componentwise(T):
    """A forest decomposes over components, so s_C <= m_C - 1 must hold for EACH.

    s_C = sum over qubits of that component of (degree - 1) = 3*m_C - n_C.
    """
    for C in components(T):
        m_C = len(C)
        n_C = len({v for i in C for v in T[i]})
        s_C = 3 * m_C - n_C
        if s_C > m_C - 1:
            return False
    return True


def sweep(n, m, cap=None, seed=0):
    """All linear covering hypergraphs (m blocks, n qubits), sampled if huge."""
    triples = list(itertools.combinations(range(n), 3))
    cands = itertools.combinations(triples, m)
    if cap:
        pool = []
        rng = random.Random(seed)
        for i, T in enumerate(cands):          # reservoir sample, no full materialisation
            if len(pool) < cap:
                pool.append(T)
            else:
                j = rng.randrange(i + 1)
                if j < cap:
                    pool[j] = T
        cands = pool
    tested = mid = skipped = 0
    witnesses = []
    for T in cands:
        if len({v for S in T for v in S}) != n or not is_linear(T):
            continue
        if not forest_possible_componentwise(T):
            skipped += 1        # some component already has s_C >= m_C: rem:ledgerfails
            continue
        tested += 1
        ok, _ = routable(T, n)
        if not ok:
            mid += 1
            if len(witnesses) < 3:
                witnesses.append(T)
    return tested, mid, witnesses, skipped


if __name__ == "__main__":
    print("Searching for the middle case: s <= m-1 but no degree-<=3 forest routing.\n")
    print(f"  {'n':>2} {'m':>2} {'s':>2} {'componentwise-ok':>17} {'skipped':>8} {'middle-case':>12}")
    total_mid = 0
    for n in range(5, 12):
        for m in range(2, 8):
            s = 3 * m - n
            if s < 0 or s > m - 1:
                continue
            cap = 40000 if n >= 9 else None
            tested, mid, wit, skipped = sweep(n, m, cap=cap)
            if tested == 0 and skipped == 0:
                continue
            total_mid += mid
            print(f"  {n:>2} {m:>2} {s:>2} {tested:>17} {skipped:>8} {mid:>12}"
                  + ("   <-- NON-EMPTY" if mid else ""))
            for w in wit:
                print(f"        witness: {w}")
    print()
    if total_mid == 0:
        print("MIDDLE CASE EMPTY over the range searched:")
        print("  whenever EVERY COMPONENT has s_C <= m_C - 1, a degree-<=3 forest")
        print("  routing always exists.  The 'skipped' column counts patterns with")
        print("  some component at s_C >= m_C -- those are rem:ledgerfails, not this case.")
        print("  => conjectured: ledger attained <=> s_C <= m_C - 1 for every component")
        print("  NOTE the condition is componentwise; the global s <= m-1 does NOT suffice")
        print("  (a triangle beside a disjoint block satisfies it and still fails).")
    else:
        print(f"MIDDLE CASE NON-EMPTY: {total_mid} instances; the dichotomy would fail.")
