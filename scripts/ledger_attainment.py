"""Is the linear-overlap ledger 6m+1-2s ATTAINED?  Exhaustive test at n <= 6.

cor:ledger proves the FLOOR delta >= 6m+1-2s for any layer of m distinct CCZ blocks
with linear overlap (pairwise sharing at most one qubit) on n = 3m-s qubits.  Whether
it is attained is the question tested here (conj:ledger).

prop:splicetree attains it whenever the merge structure F = union_q T_q can be chosen
as a FOREST of max degree <= 3.  Since |F| = sum_q (d_q - 1) = s and a forest on m
blocks has at most m-1 edges, that engine requires

    s <= m-1     i.e.    n >= 2m+1.

This script asks what happens at s >= m (n <= 2m), where no forest choice exists.

Method: enumerate every linear 3-uniform hypergraph on n qubits up to relabelling,
compute the radical floor, the ledger value, and the TRUE delta by exact Reed-Muller
coset decoding (validate_amdist.min_coset_weight over RM(n-4,n)*).
"""

import itertools
from validate_amdist import ccz_on, rm_generators, residue_vec, min_coset_weight
from compare_bounds import cubic_radical_dim


def is_linear(T):
    return all(len(set(a) & set(b)) <= 1 for a, b in itertools.combinations(T, 2))


def canon(T, n):
    """Canonical form under qubit relabelling (brute force over S_n; n <= 6)."""
    best = None
    for p in itertools.permutations(range(n)):
        img = tuple(sorted(tuple(sorted(p[v] for v in S)) for S in T))
        if best is None or img < best:
            best = img
    return best


def delta_exact(T, n, G):
    c = {}
    for t in T:
        ccz_on(c, tuple(sorted(t)))
    return int(min_coset_weight(residue_vec(c, n), G))


def sweep(n, m):
    G = rm_generators(n)
    seen, rows = set(), []
    for T in itertools.combinations(itertools.combinations(range(n), 3), m):
        if len({v for S in T for v in S}) != n:      # must cover all n qubits
            continue
        if not is_linear(T):
            continue
        cf = canon(T, n)
        if cf in seen:
            continue
        seen.add(cf)
        s = 3 * m - n
        d = cubic_radical_dim(list(T), n)
        floor = 2 * (n - d) + (1 if d == 0 else 0)
        ledger = 6 * m + 1 - 2 * s
        dl = delta_exact(list(T), n, G)
        rows.append((cf, s, d, floor, ledger, dl))
    return rows


if __name__ == "__main__":
    for n, m in ((5, 2), (6, 2), (6, 3), (6, 4)):
        s = 3 * m - n
        forest_ok = s <= m - 1
        print(f"\n=== n={n}, m={m}, s={s}  "
              f"({'forest possible: splice engine applies' if forest_ok else 'S >= M: NO forest exists'}) ===")
        rows = sweep(n, m)
        if not rows:
            print("  (no linear covering hypergraphs)")
            continue
        att = sum(1 for r in rows if r[5] == r[4])
        print(f"  {'blocks':<34} {'d':>2} {'floor':>5} {'ledger':>6} {'delta':>5}  verdict")
        for cf, s_, d, fl, led, dl in sorted(rows, key=lambda r: (r[5] - r[4], r[0])):
            v = "ATTAINED" if dl == led else f"ABOVE by {dl-led}"
            print(f"  {str(cf):<34} {d:>2} {fl:>5} {led:>6} {dl:>5}  {v}")
        print(f"  -> {att}/{len(rows)} attain the ledger")
