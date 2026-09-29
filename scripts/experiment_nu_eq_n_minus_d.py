"""
Experiment: is the stabilizer-nullity / radical conjecture  nu = n - d  true for every
pure-cubic gate, and is the underlying set identity  A = R  exact?

Context. thm:cubicLB gives delta >= 2(n-d) for pure-cubic gates (d = radical dimension of the
cubic moment tensor). The identity nu = n - d (prop:nullity) upgrades the bound to the clean
monotone  delta >= 2 nu  (twice the unitary stabilizer nullity). This is exactly the quantity
that learning-of-bounded-magic algorithms (Bell difference sampling) control, so proving it
ties the T-count floor to a learnability parameter.

Claim under test (and the mechanism of a proof):
  For phi = 4 f with f cubic, the nullity set
      A = { a : Delta_a phi(y) = c + 4 ell(y), ell F_2-affine }
  equals the radical
      R = { w : sum_a w_a T_{a q r} = 0  for all q, r },   T = cubic moment tensor.
  Hence |A| = 2^d and nu = n - log2|A| = n - d.

Reason: phi=4f, Delta_a phi = 4 (Delta_a f) with Delta_a f Boolean of degree <= 2; it is affine
iff its quadratic part vanishes, and the quadratic part of Delta_a f is exactly the bilinear
form q,r -> T(a,q,r). So "Delta_a f affine" <=> T(a,.,.) = 0 <=> a in R.

We verify A = R as SETS (not just sizes) over many random cubics, then check delta and the gap.
"""
import itertools, random, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from verify_nullity import phi_cubic, in_A          # reuse the audited nullity machinery
import validate_amdist as AM                          # the audited exact decoder


# ---- F_2 linear algebra -------------------------------------------------------
def f2_rank(rows, ncols):
    """rank over F_2 of a list of integer bitmask rows (each < 2^ncols)."""
    rows = [r for r in rows if r]
    r = 0
    for col in range(ncols):
        piv = next((i for i in range(r, len(rows)) if (rows[i] >> col) & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i] >> col) & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def radical_set(triples, n):
    """R = {w in F_2^n : sum_a w_a T_{a q r} = 0 for all q<r}, returned as a set of int masks."""
    # constraint rows: one per ordered structural pair (q,r), entry a = T_{a,q,r}
    T = set()
    for (a, b, c) in triples:
        for p in itertools.permutations((a, b, c)):
            T.add(p)
    rows = []
    for q in range(n):
        for r in range(n):
            if q == r:
                continue
            row = 0
            for a in range(n):
                if (a, q, r) in T:        # T symmetric; entry 1 iff {a,q,r} a present distinct triple
                    row |= (1 << a)
            if row:
                rows.append(row)
    # R = nullspace of the matrix whose rows are 'rows' (constraints on w)
    # Build basis of solution space by Gaussian elimination on the constraint matrix.
    # Reduce rows to row-echelon, track pivot columns.
    mat = [r for r in rows if r]
    pivots = []
    rr = 0
    mat = mat[:]
    for col in range(n):
        piv = next((i for i in range(rr, len(mat)) if (mat[i] >> col) & 1), None)
        if piv is None:
            continue
        mat[rr], mat[piv] = mat[piv], mat[rr]
        for i in range(len(mat)):
            if i != rr and (mat[i] >> col) & 1:
                mat[i] ^= mat[rr]
        pivots.append(col); rr += 1
    free = [c for c in range(n) if c not in pivots]
    # express each pivot var in terms of free vars
    basis = []
    for fcol in free:
        w = 1 << fcol
        for i, pcol in enumerate(pivots):
            # row i has leading 1 at pcol; w_pcol = sum over free cols in that row
            if (mat[i] >> fcol) & 1:
                w |= (1 << pcol)
        basis.append(w)
    # enumerate the span
    R = {0}
    for b in basis:
        R |= {x ^ b for x in R}
    return R, len(free)


def A_set(triples, n):
    phi = phi_cubic(triples, n)
    return set(a for a in range(1 << n) if in_A(phi, a, n))


def delta_of(triples, n):
    coeff = {}
    for t in triples:
        AM.ccz_on(coeff, t)
    return AM.Tcount(coeff, n)


# ---- experiment ---------------------------------------------------------------
def run_identity_scan(ns, samples, seed=0):
    """Verify A == R (set identity) and nu == n-d over random cubics. No decoding; fast."""
    rng = random.Random(seed)
    bad = 0; total = 0; dist = {}
    for n in ns:
        triples_all = list(itertools.combinations(range(n), 3))
        for _ in range(samples):
            k = rng.randint(1, len(triples_all))
            triples = rng.sample(triples_all, k)
            R, d = radical_set(triples, n)
            A = A_set(triples, n)
            total += 1
            ok = (A == R)
            if not ok:
                bad += 1
                print(f"  COUNTEREXAMPLE n={n} triples={triples}  |A|={len(A)} |R|={len(R)} A==R:{A==R}")
            dist[(n, d)] = dist.get((n, d), 0) + 1
    print(f"[identity scan] A==R on {total-bad}/{total} random cubics over n={ns}", end="")
    print("   ALL PASS" if bad == 0 else f"   {bad} FAIL")
    return bad == 0


def run_delta_gap(cases):
    """For small n: compute (nu, d, 2(n-d), delta) and the gap delta - 2(n-d)."""
    print("\n[delta gap]   gate                         n  d  nu  2(n-d)  delta  gap  (nu=n-d?)")
    for name, triples, n in cases:
        R, d = radical_set(triples, n)
        A = A_set(triples, n)
        nu = n - (len(A).bit_length() - 1)         # log2|A|
        lb = 2 * (n - d)
        dlt = delta_of(triples, n)
        gap = dlt - lb
        flag = "yes" if (1 << d) == len(A) else "NO!"
        print(f"  {name:28s} {n:2d} {d:2d} {nu:3d}  {lb:5d}  {dlt:5d}  {gap:3d}   {flag}")


def run_gap_scan(ns, samples, seed=7):
    """Distribution of the gap delta - 2(n-d) over random cubics. Tests delta >= 2(n-d)+1."""
    rng = random.Random(seed)
    print("\n[gap scan]  testing delta >= 2(n-d)+1  (thm:cubicLB proves +1 only for d=0)")
    for n in ns:
        triples_all = list(itertools.combinations(range(n), 3))
        gapdist = {}; mingap_by_d = {}; viol = 0; tot = 0
        for _ in range(samples):
            k = rng.randint(1, len(triples_all))
            triples = rng.sample(triples_all, k)
            R, d = radical_set(triples, n)
            dlt = delta_of(triples, n)
            gap = dlt - 2 * (n - d)
            tot += 1
            gapdist[gap] = gapdist.get(gap, 0) + 1
            mingap_by_d[d] = min(mingap_by_d.get(d, 99), gap)
            if gap < 1:
                viol += 1
                if viol <= 5:
                    print(f"    gap<1: n={n} d={d} delta={dlt} 2(n-d)={2*(n-d)} triples={triples}")
        print(f"  n={n}: {tot} cubics | gap distribution {dict(sorted(gapdist.items()))}"
              f" | min gap per d {dict(sorted(mingap_by_d.items()))}"
              f" | gap<1: {viol}")


if __name__ == "__main__":
    print("self-test: CCZ has d=0, |A|=1, nu=3 =>")
    R, d = radical_set([(0, 1, 2)], 3); A = A_set([(0, 1, 2)], 3)
    print(f"  d={d}, |R|={len(R)}, |A|={len(A)}, A==R:{A==R}  {'OK' if d==0 and A=={0} else 'FAIL'}\n")

    run_identity_scan(ns=[4, 5, 6, 7], samples=400, seed=1)

    line = lambda n: [(i, i+1, i+2) for i in range(n-2)]
    par  = lambda m: [(3*i, 3*i+1, 3*i+2) for i in range(m)]
    comp = lambda n: list(itertools.combinations(range(n), 3))
    cyc6 = [(0,1,2),(2,3,4),(4,5,0)]
    run_delta_gap([
        ("CCZ", par(1), 3),
        ("2 parallel Toffolis", par(2), 6),
        ("3-local line f_5", line(5), 5),
        ("3-local line f_6", line(6), 6),
        ("elem symmetric e_3 (n=5)", comp(5), 5),
        ("elem symmetric e_3 (n=6) [deg]", comp(6), 6),
        ("cyclic cubic (n=6)", cyc6, 6),
    ])

    run_gap_scan(ns=[4, 5], samples=600, seed=7)
