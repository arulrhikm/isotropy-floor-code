"""
r_min: the GL-orbit minimum of the quadratic rank -- a valid strengthening of thm:genLB, and a
direct answer to what rem:grading says is wrong with r.

Argument.  rem:grading says: "the alternating rank r is a complete
invariant of a quadratic form but not of a quadratic GATE, and no bound depending on r alone can be
tight on the quadratic side."  The mechanism: over Z_8,
2u + 2v = 2(u xor v) + 4uv, so a change of variables MOVES MASS BETWEEN DEGREES: relabelling by an
F_2 change of basis converts quadratic mass into cubic mass.  So r is basis-dependent.

But delta is NOT.  CNOT circuits realise GL(n,F_2), and delta is Clifford bi-invariant, so
delta(phi o L) = delta(phi) for every L in GL(n,F_2).  And d* is a GL-orbit invariant (the paper
relies on this: "the floor is a GL-orbit invariant", rem:ceilinvariance).  Since thm:genLB holds for
EVERY representative of the orbit,

    delta(U)  >=  max_L [ 2(n - d*) - r(phi o L) ]  =  2(n - d*) - r_min,
    r_min := min over L in GL(n,F_2) of r(phi o L).

r_min is by construction an invariant of the GATE, not of the form -- exactly what rem:grading
demands.  The bound is sound for any SUBSET of L sampled, since each L gives a valid bound
independently; sampling only makes r_min an over-estimate, hence the floor an under-estimate.

How L acts.  phi = sum_y c_y (y . x), so phi(Lx) = sum_y c_y ((L^T y) . x): the coefficient dict is
re-indexed by y -> L^T y.  A permutation of parities, nothing more.

COST.  |GL(4,2)| = 20160 (exhaustible); |GL(5,2)| ~ 10^7 (sampled).  So r_min is NOT O(n^3) as
stated -- it is a strengthening at a cost, and whether it has an efficient proxy is the open part.

Run:  python orbit_rank.py [gates_per_n]
"""
import itertools
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validate_amdist as AM
from experiment_mixed_nullity import build_phi, true_moments, dprime
from experiment_nu_eq_n_minus_d import f2_rank


def apply_L(coeff, Lt, n):
    """phi o L, where Lt[y] is L^T applied to the parity mask y.  Re-indexes the coefficients."""
    out = {}
    for y, c in coeff.items():
        yy = Lt[y]
        out[yy] = (out.get(yy, 0) + c) % 8
    return {y: c for y, c in out.items() if c % 8}


def transpose_action(cols, n):
    """cols: list of n column masks of an invertible matrix L.  Return the table y -> L^T y."""
    # (L^T y)_i = sum_j L_ji y_j = <col_i, y>.  So bit i of the image is parity(col_i & y).
    tab = [0] * (1 << n)
    for y in range(1 << n):
        v = 0
        for i in range(n):
            if bin(cols[i] & y).count("1") & 1:
                v |= (1 << i)
        tab[y] = v
    return tab


def gl_elements(n, limit=None, rng=None):
    """Invertible n x n matrices over F_2, as lists of column masks.  Exhaustive when small."""
    total_cap = 60000
    if limit is None and n <= 4:
        out = []
        for cols in itertools.product(range(1, 1 << n), repeat=n):
            # independence check
            piv, ok = [], True
            for c in cols:
                v = c
                for p in piv:
                    v = min(v, v ^ p)
                if not v:
                    ok = False
                    break
                piv.append(v)
                piv.sort(reverse=True)
            if ok:
                out.append(cols)
                if len(out) >= total_cap:
                    break
        return out
    rng = rng or random.Random(0)
    out, seen = [], set()
    tries = 0
    while len(out) < (limit or 2000) and tries < 200000:
        tries += 1
        cols = tuple(rng.randrange(1, 1 << n) for _ in range(n))
        piv, ok = [], True
        for c in cols:
            v = c
            for p in piv:
                v = min(v, v ^ p)
            if not v:
                ok = False
                break
            piv.append(v)
            piv.sort(reverse=True)
        if ok and cols not in seen:
            seen.add(cols)
            out.append(cols)
    return out


def rank_of(coeff, n):
    _, Qrows = true_moments(coeff, n)
    return f2_rank(Qrows, n)


def orbit_scan(coeff, n, Ls, tabs):
    """Return (r_here, r_min, dstar_stable)."""
    r0 = rank_of(coeff, n)
    d0 = dprime(coeff, n)
    rmin = r0
    dstable = True
    for tab in tabs:
        c2 = apply_L(coeff, tab, n)
        if not c2:
            continue
        r2 = rank_of(c2, n)
        if r2 < rmin:
            rmin = r2
            if dprime(c2, n) != d0:
                dstable = False
    return r0, rmin, d0, dstable


def run(gates=60, ns=(4, 5), seed=20260903, n_L=1500):
    rng = random.Random(seed)
    print("Building GL samples...")
    tabs_by_n = {}
    for n in ns:
        Ls = gl_elements(n, limit=(None if n <= 4 else n_L), rng=rng)
        if n <= 4:
            Ls = Ls[:n_L] if len(Ls) > n_L else Ls
        tabs_by_n[n] = [transpose_action(cols, n) for cols in Ls]
        print("  n=%d: %d basis changes (%s)"
              % (n, len(tabs_by_n[n]), "exhaustive subset" if n <= 4 else "random sample"))
    print()

    for n in ns:
        tabs = tabs_by_n[n]
        improved = 0
        tight_new = 0
        tight_old = 0
        tot = 0
        dfail = 0
        gain_sum = 0
        rows = []
        for _ in range(gates):
            p3 = rng.choice((0.15, 0.3, 0.5))
            p2 = rng.choice((0.15, 0.3, 0.5, 0.8))
            p1 = rng.choice((0.0, 0.25))
            triples = [t for t in itertools.combinations(range(n), 3) if rng.random() < p3]
            pairs = [p for p in itertools.combinations(range(n), 2) if rng.random() < p2]
            lin = [i for i in range(n) if rng.random() < p1]
            coeff, _ = build_phi(triples, pairs, lin, n)
            if not coeff:
                continue
            delta = AM.Tcount(coeff, n)
            r0, rmin, d0, dstable = orbit_scan(coeff, n, None, tabs)
            if not dstable:
                dfail += 1
            old = 2 * (n - d0) - r0 + (1 if r0 == 0 else 0)
            new = 2 * (n - d0) - rmin + (1 if rmin == 0 else 0)
            assert old <= delta, ("old bound violates delta", old, delta)
            assert new <= delta, ("ORBIT BOUND VIOLATES DELTA", new, delta, triples, pairs, lin, n)
            tot += 1
            if rmin < r0:
                improved += 1
                gain_sum += (new - old)
            if old == delta:
                tight_old += 1
            if new == delta:
                tight_new += 1
            rows.append((delta, old, new, r0, rmin))
        print("n=%d  gates=%d" % (n, tot))
        print("  d* stable across the orbit (must be all)   : %d / %d violations" % (dfail, tot))
        print("  r_min < r  (basis change lowers the rank)  : %d  (%.1f%%)"
              % (improved, 100.0 * improved / max(tot, 1)))
        print("  mean floor gain when it improves           : %.2f"
              % (gain_sum / max(improved, 1)))
        print("  floor exactly tight, ORIGINAL basis        : %d  (%.1f%%)"
              % (tight_old, 100.0 * tight_old / max(tot, 1)))
        print("  floor exactly tight, ORBIT-MINIMISED       : %d  (%.1f%%)"
              % (tight_new, 100.0 * tight_new / max(tot, 1)))
        print()


if __name__ == "__main__":
    # sanity: delta is GL-invariant, which is what makes the whole argument work
    n = 4
    c, _ = build_phi([(0, 1, 2)], [(2, 3)], [], n)
    d0 = AM.Tcount(c, n)
    rng = random.Random(1)
    tabs = [transpose_action(cols, n) for cols in gl_elements(n)[:200]]
    bad = 0
    for tab in tabs:
        c2 = apply_L(c, tab, n)
        if c2 and AM.Tcount(c2, n) != d0:
            bad += 1
    print("SELF-TEST: delta GL-invariant on 200 basis changes: %d mismatches (must be 0)\n" % bad)
    assert bad == 0
    run(gates=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
