"""
COUNTEREXAMPLE to  2(n - d_k) >= nu  at level k=4.

The inequality is false in general (rem:levelkdom). It has content only when the radical is
large: nu <= n always, so 2(n-d_k) >= n >= nu whenever d_k <= n/2. Random homogeneous forms
almost never have d_k > n/2, so random sweeps do not test it.

This script searches the large-radical regime deliberately (pullback construction: f is the
degree-k homogeneous part of g(pi x) for a surjection pi onto F_2^D, forcing d_k >= n-D) and
verifies any violation by FIVE independent checks before reporting it.

Run:  python levelk_counterexample.py
"""
import random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from levelk_nullity_study import (anf, degree, deriv, nullity, eval_homog_from_coeffs,
                                  top_radical_dim, radical_dim_and_filtration)
from levelk_dominance import apply_pi, homog_part, subspace_dim


def verify(coeffs, n, k, verbose=True):
    """Five independent checks on a claimed counterexample. Returns dict of measurements."""
    tt = eval_homog_from_coeffs(coeffs, n)

    # (1) f really homogeneous of degree exactly k
    a = anf(tt, n)
    degs = {bin(S).count("1") for S in range(1 << n) if a[S]}
    assert degs == {k}, "f is not homogeneous of degree %d: degrees present %s" % (k, sorted(degs))

    # (2) R = {a : deg Delta_a f <= k-2} computed directly, and checked to be a subspace
    R = [w for w in range(1 << n) if degree(deriv(tt, w, n), n) <= k - 2]
    Rset = set(R)
    for x in R[:64]:
        for y in R[:64]:
            assert (x ^ y) in Rset, "R is not closed under XOR"
    dR_log = len(R).bit_length() - 1
    assert (1 << dR_log) == len(R), "|R| = %d is not a power of two" % len(R)
    dR_basis = subspace_dim(R)
    assert dR_basis == dR_log, "two dim(R) computations disagree: %d vs %d" % (dR_basis, dR_log)

    # (3) d_k independently from the monomial support (poly-time, different code path)
    dk_poly = top_radical_dim(coeffs, n, k)
    assert dk_poly == dR_log, "top_radical_dim=%d disagrees with dim R=%d" % (dk_poly, dR_log)

    # (4) A = {a : deg Delta_a f <= 1}, brute force; subspace; nu
    A = [w for w in range(1 << n) if degree(deriv(tt, w, n), n) <= 1]
    Aset = set(A)
    for x in A:
        for y in A:
            assert (x ^ y) in Aset, "A is not closed under XOR"
    dA = len(A).bit_length() - 1
    assert (1 << dA) == len(A), "|A| = %d is not a power of two" % len(A)
    nu = n - dA
    nu2, A2 = nullity(tt, n)
    assert nu == nu2 and set(A2) == Aset, "nullity() disagrees with the direct computation"

    # (5) A <= R (prop:nullityk), and the filtration agrees
    assert Aset <= Rset, "A is not contained in R -- prop:nullityk would be violated"
    filt = radical_dim_and_filtration(tt, n, k)
    assert set(filt[1]) == Rset and set(filt[k - 2]) == Aset, "filtration endpoints disagree"

    D = n - dR_log
    out = dict(n=n, k=k, dk=dR_log, D=D, nu=nu, dimA=dA, gap=dR_log - dA,
               floor=2 * D, margin=nu - 2 * D, big_radical=(2 * dR_log > n),
               coeffs=sorted(coeffs))
    if verbose:
        print("  n=%d k=%d | d_k=%d (2d_k>n: %s) | D=n-d_k=%d | dim A=%d | nu=%d | "
              "floor 2(n-d_k)=%d | nu-floor=%+d"
              % (n, k, out["dk"], out["big_radical"], D, dA, nu, out["floor"], out["margin"]))
    return out


def search(n, D, k, trials, seed):
    rng = random.Random(seed)
    best = None
    for _ in range(trials):
        pi = [rng.randrange(1, 1 << n) for _ in range(D)]
        if subspace_dim(pi) < D:
            continue
        gc = [m for m in range(1 << D) if bin(m).count("1") == k and rng.random() < 0.6]
        if not gc:
            continue
        g_tt = eval_homog_from_coeffs(gc, D)
        comp = [g_tt[apply_pi(x, pi)] for x in range(1 << n)]
        coeffs = homog_part(comp, n, k)
        if not coeffs:
            continue
        try:
            out = verify(coeffs, n, k, verbose=False)
        except AssertionError as e:
            print("  VERIFY FAILED (script bug, not a counterexample): %s" % e)
            continue
        if best is None or out["margin"] > best["margin"]:
            best = out
    return best


if __name__ == "__main__":
    print("Searching the large-radical regime for  nu > 2(n-d_k)  at k=4 ...")
    hits = []
    for n, D in ((9, 4), (10, 4), (9, 5), (10, 5)):
        if D >= n:
            continue
        b = search(n, D, 4, trials=60, seed=4000 + 10 * n + D)
        if b is None:
            continue
        tag = "VIOLATION" if b["margin"] > 0 else "ok"
        print(" n=%2d D=%d -> best margin %+d  (%s)" % (n, D, b["margin"], tag))
        if b["margin"] > 0:
            hits.append(b)
    print()
    if not hits:
        print("no violation found in this pass")
    else:
        h = max(hits, key=lambda z: z["margin"])
        print("=" * 72)
        print("COUNTEREXAMPLE (all five verification checks passed):")
        verify(h["coeffs"], h["n"], h["k"])
        print()
        print("  f = sum of these degree-4 monomials (bitmask over %d variables):" % h["n"])
        print("   ", h["coeffs"])
        print()
        print("  Consequence: nu = %d exceeds the level-k isotropy floor 2(n-d_k) = %d,"
              % (h["nu"], h["floor"]))
        print("  so 2(n-d_k) does NOT dominate stabilizer nullity at level k.")
        print("  The level-k floor is max(nu, 2(n-d_k)), as prop:nullityk")
        print("  states it (rem:levelkdom).")
