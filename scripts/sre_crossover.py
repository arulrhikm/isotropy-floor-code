"""Does the SRE floor ever OVERTAKE the isotropy floor 2(n-d*)+1?  Yes, from n=9.

CLOSED FORM (verified against the 4^n Pauli sum in sre_vs_nullity).  For a pure-cubic
phi = 4f, the shifted function g_a = (-1)^{D_a f} has a QUADRATIC exponent, whose Walsh
spectrum is flat at 2^{-r_a/2} on 2^{r_a} points (r_a = symplectic rank of D_a f).  So

        M_2  =  n  -  log2 sum_{a} 2^{-r_a},       r_0 = 0.

The SRE is thus a SOFT count of derivative ranks; the isotropy floor 2(n-d*) is a HARD count
of the rank of the moment column span.  That is why they can cross.

Caps.  r_a <= n forces sum_a 2^{-r_a} >= 1 + (2^n-1)2^{-n} -> 2, so M_2 <= n-1 (attained
iff every nonzero derivative is bent), i.e.

        SRE floor  <=  ceil( (n-1)/(2-log2 3) )  ~  2.4094 (n-1),
        isotropy floor                           <=  2n+1.

2.4094(n-1) > 2n+1 exactly when n > 8.33.  So from n = 9 there is room, and this hill-climbs
on the derivative-rank profile to find a gate that uses it.
"""
import argparse, math, random, sys
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sre_vs_nullity import M2_T
from cubic_orbit_atlas import radical_dim

def bilinear_parts(n):
    """B[i] = list of n row-bitmasks: the bilinear form of the derivative of x_p x_q x_r in
    direction e_i, accumulated per monomial index."""
    import itertools
    mons = list(itertools.combinations(range(n), 3))
    per_mon = []
    for (p, q, r) in mons:
        rows = [[0]*n for _ in range(n)]          # rows[i] = bitmask over j
        for (i, j, k) in ((p,q,r),(q,r,p),(r,p,q)):
            rows[i][j] |= 1 << k
            rows[i][k] |= 1 << j
        per_mon.append(rows)
    return mons, per_mon

def rank_f2(rows, n):
    rank = 0
    rows = list(rows)
    for c in range(n):
        piv = next((i for i in range(rank, len(rows)) if (rows[i] >> c) & 1), None)
        if piv is None: continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] >> c) & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank

def m2(sel, per_mon, n):
    """sel = list of monomial indices.  Rows_i = XOR of per_mon[m][i] over chosen m."""
    Ri = [[0]*n for _ in range(n)]
    for m in sel:
        rm = per_mon[m]
        for i in range(n):
            for j in range(n):
                Ri[i][j] ^= rm[i][j]
    tot = 0.0
    for a in range(1 << n):
        rows = [0]*n
        for i in range(n):
            if (a >> i) & 1:
                for j in range(n):
                    rows[j] ^= Ri[i][j]
        tot += 2.0 ** -rank_f2(rows, n)
    return n - math.log2(tot)

def climb(n, iters, seed):
    rng = random.Random(seed)
    mons, per_mon = bilinear_parts(n)
    cur = set(rng.sample(range(len(mons)), len(mons)//2))
    best = m2(sorted(cur), per_mon, n)
    for it in range(iters):
        m = rng.randrange(len(mons))
        trial = set(cur); trial.symmetric_difference_update({m})
        if not trial: continue
        v = m2(sorted(trial), per_mon, n)
        if v > best:
            best, cur = v, trial
    tri = [mons[i] for i in sorted(cur)]
    nu = n - radical_dim(tri, n)
    return best, math.ceil(best / M2_T - 1e-9), 2 * nu + 1, tri

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[8, 9, 10])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(__doc__.strip())
    print(f"\n  {'n':>2s} {'max M_2':>8s} {'M_2/n':>6s} {'cap n-1':>8s} {'SRE':>4s} "
          f"{'2nu+1':>6s} {'SRE-(2nu+1)':>12s}")
    for n in a.n:
        v, s, fl, tri = climb(n, a.iters, a.seed)
        flag = "  <<< SRE WINS" if s > fl else ""
        print(f"  {n:2d} {v:8.4f} {v/n:6.3f} {n-1:8d} {s:4d} {fl:6d} {s-fl:12d}{flag}")
