"""Sparsity turns the two-sided bracket into a certified approximation factor.

Verifies three claims of the cubic-moment-form paper:

  prop:termcount      For a 3-local Ising cost layer with hypergraph H, couplings J
                      and fields h, the canonical residue ceiling of prop:ceiling is
                      EXACTLY the term count:  W(s) = |H| + |J| + |h|.
                      The degree/codegree mass carried by the moments cancels against
                      the downward solve, leaving z supported on the terms themselves.

  cor:sparsebracket   For any diagonal level-three gate with cubic support F and
                      quadratic support Q (both read off the moments),
                          W(s) <= 4|F| + |Q| + n,
                      hence delta/floor <= (4|F|+|Q|+n)/floor, a certified ratio.

  bounded degree      If every qubit meets at most D terms the ratio is O(D),
                      independent of n; tight at D=1 (disjoint CCZ layer, 7/6).

The floor is 2(n-d*) - r, computed here on the pure-cubic locus as 2(n-d) [+1 at d=0]
reusing gf2m_floor.radical_dim.  bracket_table.py is deliberately NOT imported: it
fails at import on a pre-existing ortools/protobuf version conflict.
"""

import random
from itertools import combinations
from collections import Counter

import gf2m_floor as GF


# ---------------------------------------------------------------- prop:ceiling

def canonical_residue(s, n):
    """The downward solve of prop:ceiling.  s maps sorted tuples (1<=|A|<=3) to F2."""
    z = {}
    for A in combinations(range(n), 3):
        z[A] = s.get(A, 0)
    for A in combinations(range(n), 2):
        acc = s.get(A, 0)
        for c in range(n):
            if c not in A:
                acc ^= z[tuple(sorted(A + (c,)))]
        z[A] = acc
    for a in range(n):
        acc = s.get((a,), 0)
        for b in range(n):
            if b != a:
                acc ^= z[tuple(sorted((a, b)))]
        for b, c in combinations([x for x in range(n) if x != a], 2):
            acc ^= z[tuple(sorted((a, b, c)))]
        z[(a,)] = acc
    return z


def ceiling(s, n):
    return sum(1 for v in canonical_residue(s, n).values() if v)


def ising_moments(H, J, h, n):
    """s_A for a 3-local Ising cost layer (Appendix D)."""
    s = {}
    for A in combinations(range(n), 3):
        s[A] = 1 if A in H else 0
    for A in combinations(range(n), 2):
        codeg = sum(1 for c in range(n)
                    if c not in A and tuple(sorted(A + (c,))) in H)
        s[A] = (1 if A in J else 0) ^ (codeg & 1)
    for a in range(n):
        degJ = sum(1 for b in range(n) if b != a and tuple(sorted((a, b))) in J)
        degH = sum(1 for B in combinations([x for x in range(n) if x != a], 2)
                   if tuple(sorted((a,) + B)) in H)
        s[(a,)] = (1 if (a,) in h else 0) ^ (degJ & 1) ^ (degH & 1)
    return s


def w_of_F(F):
    """Pure-cubic ceiling w(F) = #{A : 1<=|A|<=3, odd number of monomials contain A}."""
    c = Counter()
    for S in F:
        e = sorted(S)
        for r in (1, 2, 3):
            for A in combinations(e, r):
                c[A] += 1
    return sum(1 for v in c.values() if v & 1)


# ------------------------------------------------------------------ the checks

def check_termcount(trials=12, seed=3):
    print("== prop:termcount:  W(s) == |H| + |J| + |h|  for Ising cost layers ==")
    rng = random.Random(seed)
    ok = True
    print(f"  {'n':>2} {'|H|':>5} {'|J|':>5} {'|h|':>4} {'W(s)':>6} {'terms':>6}  match")
    for _ in range(trials):
        n = rng.choice([6, 7, 8, 9])
        H = {A for A in combinations(range(n), 3) if rng.random() < 0.3}
        J = {A for A in combinations(range(n), 2) if rng.random() < 0.3}
        h = {(a,) for a in range(n) if rng.random() < 0.4}
        W = ceiling(ising_moments(H, J, h, n), n)
        tot = len(H) + len(J) + len(h)
        ok &= (W == tot)
        print(f"  {n:>2} {len(H):>5} {len(J):>5} {len(h):>4} {W:>6} {tot:>6}  "
              f"{'YES' if W == tot else 'NO'}")
    print(f"  -> {'all match' if ok else 'MISMATCH'}\n")
    return ok


def check_sparsebound(trials=12, seed=17):
    print("== cor:sparsebracket:  W(s) <= 4|F| + |Q| + n  (general level-three) ==")
    rng = random.Random(seed)
    ok = True
    print(f"  {'n':>2} {'|F|':>5} {'|Q|':>5} {'W(s)':>6} {'bound':>6}  holds")
    for _ in range(trials):
        n = rng.choice([6, 7, 8])
        H = {A for A in combinations(range(n), 3) if rng.random() < 0.35}
        J = {A for A in combinations(range(n), 2) if rng.random() < 0.35}
        h = {(a,) for a in range(n) if rng.random() < 0.5}
        s = ising_moments(H, J, h, n)
        F = [A for A in combinations(range(n), 3) if s[A]]
        Q = [A for A in combinations(range(n), 2) if s[A]]
        W = ceiling(s, n)
        bound = 4 * len(F) + len(Q) + n
        ok &= (W <= bound)
        print(f"  {n:>2} {len(F):>5} {len(Q):>5} {W:>6} {bound:>6}  "
              f"{'OK' if W <= bound else 'VIOLATED'}")
    print(f"  -> {'bound holds throughout' if ok else 'VIOLATION'}\n")
    return ok


def check_degree_ratio(seed=5):
    print("== bounded degree: worst ceiling/floor over random max-degree-D hypergraphs ==")
    print("   (pure cubic; floor = 2(n-d) [+1 at d=0]; predicted bound 2D/3 + 1/2)")
    rng = random.Random(seed)
    ok = True
    print(f"  {'D':>2} {'n':>4} {'|F|':>5} {'d':>3} {'floor':>6} {'ceil':>6} "
          f"{'ratio':>7} {'bound':>7}")
    for D in (1, 2, 3, 4, 6):
        for n in (12, 18, 24):
            best, rec = 0.0, None
            for _ in range(40):
                deg = {v: 0 for v in range(n)}
                F = set()
                for _ in range(400):
                    S = tuple(sorted(rng.sample(range(n), 3)))
                    if S in F or any(deg[v] + 1 > D for v in S):
                        continue
                    F.add(S)
                    for v in S:
                        deg[v] += 1
                if not F:
                    continue
                Ff = [frozenset(S) for S in F]
                d = GF.radical_dim(Ff, n)
                floor = 2 * (n - d) + (1 if d == 0 else 0)
                if floor <= 0:
                    continue
                ceil = w_of_F(Ff)
                if ceil / floor > best:
                    best, rec = ceil / floor, (len(F), d, floor, ceil)
            if rec:
                bound = 2 * D / 3 + 0.5
                ok &= (best <= bound + 1e-9)
                print(f"  {D:>2} {n:>4} {rec[0]:>5} {rec[1]:>3} {rec[2]:>6} {rec[3]:>6} "
                      f"{best:>7.3f} {bound:>7.3f}  "
                      f"{'OK' if best <= bound + 1e-9 else 'VIOLATED'}")
    print(f"  -> {'bound holds; ratio flat in n for fixed D' if ok else 'VIOLATION'}\n")
    return ok


if __name__ == "__main__":
    a = check_termcount()
    b = check_sparsebound()
    c = check_degree_ratio()
    print("ALL CHECKS PASS" if (a and b and c) else "*** SOME CHECK FAILED ***")
