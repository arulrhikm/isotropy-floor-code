"""
Audit of the two things `tab:atlas` silently assumes.

(C3) EVERY paper result flows through `moments_from_cubic` / `moments_from_quadratic`, but
     `sat_tcount.validate()` only ever exercises `moments_from_coeff` (the residue path that was
     checked against the meet-in-the-middle decoder).  Here we check the shortcut constructors
     against that validated path.

(C1) `cubic_orbit_atlas.atlas` evaluates delta at ONE BFS seed per orbit and then multiplies by the
     orbit size to get the coverage figure.  Nothing anywhere checks that delta is actually constant
     along an orbit -- so a wrong drop-rule in `transvection_map` would merge or split orbits and
     the census would be silently wrong.  Here we apply random L in GL(n,F_2) to each orbit
     representative and demand that delta, the radical dimension and the derivative rank spectrum
     are unchanged.

This is the test whose absence was the single largest risk in the census.  It is deliberately built
on DIFFERENT code from the atlas: the group action here is `cubic_part_after_L`, which expands the
substitution monomial-by-monomial, whereas the atlas walks transvection generators via
`transvection_map`.  A bug in either one alone will show up as a mismatch.
"""
import argparse, itertools, random, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_amdist as AM
from cubic_orbit_atlas import orbits, form_triples, radical_dim, monomials
from cubic_orbit_n7 import derivative_spectrum
from quadratic_locus import random_invertible
from sat_tcount import (moments_from_cubic, moments_from_quadratic, moments_from_coeff,
                        weight_of_moment_free_rep, tcount)


# ---------------------------------------------------------------- the group action

def cubic_part_after_L(triples, L, n):
    """Degree-3 homogeneous part of f o L, where (Lx)_a = <L[a], x>.

    Expands (L_a.x)(L_b.x)(L_c.x) over F_2 and keeps only the terms with three DISTINCT
    variables; a repeated variable reduces the degree (x_i^2 = x_i) and so falls into RM(2,n),
    which is exactly what the cubic moments quotient out.  Validated against a direct ANF
    extraction of the composed truth table (see `check_action`)."""
    out = set()
    for (a, b, c) in triples:
        A = [i for i in range(n) if (L[a] >> i) & 1]
        B = [i for i in range(n) if (L[b] >> i) & 1]
        C = [i for i in range(n) if (L[c] >> i) & 1]
        for i in A:
            for j in B:
                for k in C:
                    if i != j and j != k and i != k:
                        out ^= {tuple(sorted((i, j, k)))}
    return sorted(out)


def _anf(tt, n):
    a = list(tt)
    for i in range(n):
        s = 1 << i
        for x in range(1 << n):
            if x & s:
                a[x] ^= a[x ^ s]
    return a


def _compose_truthtable(triples, L, n):
    tt = [0] * (1 << n)
    for x in range(1 << n):
        y = 0
        for a in range(n):
            if bin(L[a] & x).count("1") & 1:
                y |= 1 << a
        v = 0
        for (p, q, r) in triples:
            if (y >> p) & 1 and (y >> q) & 1 and (y >> r) & 1:
                v ^= 1
        tt[x] = v
    return tt


def check_action(ns=(4, 5, 6), trials=30, seed=5):
    """The group action itself, against a direct ANF expansion of the composed truth table."""
    rng = random.Random(seed)
    bad = tot = 0
    for n in ns:
        T3 = list(itertools.combinations(range(n), 3))
        for _ in range(trials):
            tri = rng.sample(T3, rng.randint(1, min(5, len(T3))))
            L = random_invertible(n, rng)
            A = _anf(_compose_truthtable(tri, L, n), n)
            ref = sorted(tuple(i for i in range(n) if (m >> i) & 1)
                         for m in range(1 << n) if A[m] and bin(m).count("1") == 3)
            tot += 1
            bad += (ref != cubic_part_after_L(tri, L, n))
    print(f"   group action vs direct ANF: {tot-bad}/{tot} agree"
          f"   {'ok' if bad == 0 else '*** ACTION IS WRONG ***'}")
    return bad == 0


# ---------------------------------------------------------------- C3

def check_constructors(ns=(4, 5), trials=60, seed=11):
    """moments_from_cubic / moments_from_quadratic vs the validated moments_from_coeff path."""
    rng = random.Random(seed)
    badc = badq = tot = 0
    for n in ns:
        T3 = list(itertools.combinations(range(n), 3))
        T2 = list(itertools.combinations(range(n), 2))
        for _ in range(trials):
            tri = rng.sample(T3, rng.randint(1, len(T3)))
            c = {}
            for t in tri:
                AM.ccz_on(c, t)
            badc += (moments_from_cubic(tri, n) != moments_from_coeff(c, n))

            prs = rng.sample(T2, rng.randint(1, len(T2)))
            c = {}
            for p in prs:
                AM.cs_on(c, p)
            badq += (moments_from_quadratic(prs, n) != moments_from_coeff(c, n))
            tot += 1
    print(f"   moments_from_cubic     vs moments_from_coeff: {tot-badc}/{tot}"
          f"   {'ok' if badc == 0 else '*** MISMATCH ***'}")
    print(f"   moments_from_quadratic vs moments_from_coeff: {tot-badq}/{tot}"
          f"   {'ok' if badq == 0 else '*** MISMATCH ***'}")
    return badc == 0 and badq == 0


# ---------------------------------------------------------------- C1

def check_orbit_invariance(ns=(4, 5, 6), samples=6, seed=17, timeout=None):
    """For every orbit, transport the representative by random L and demand delta, d and the
    rank spectrum are unchanged.  This is what licenses weighting a single-point delta by the
    orbit size."""
    rng = random.Random(seed)
    ok = True
    for n in ns:
        reps, mons = orbits(n, verbose=False)
        t0 = time.time()
        checked = bad_d = bad_s = bad_delta = 0
        for f, size in reps:
            tri = form_triples(f, mons)
            if not tri:
                continue
            d0 = radical_dim(tri, n)
            s0 = derivative_spectrum(tri, n)
            delta0, st, _ = tcount(moments_from_cubic(tri, n), n, timeout=timeout)
            assert st == "OPTIMAL", f"delta not proved optimal for {tri}"
            for _ in range(samples):
                L = random_invertible(n, rng)
                tri2 = cubic_part_after_L(tri, L, n)
                if not tri2:                      # the zero form is its own orbit
                    bad_delta += 1
                    continue
                checked += 1
                if radical_dim(tri2, n) != d0:
                    bad_d += 1
                if derivative_spectrum(tri2, n) != s0:
                    bad_s += 1
                d2, st2, _ = tcount(moments_from_cubic(tri2, n), n, timeout=timeout)
                if st2 != "OPTIMAL" or d2 != delta0:
                    bad_delta += 1
        good = (bad_d == bad_s == bad_delta == 0)
        ok &= good
        print(f"   n={n}: {checked} transported forms | radical mismatches {bad_d}"
              f" | spectrum mismatches {bad_s} | DELTA mismatches {bad_delta}"
              f"   {'ok' if good else '*** ORBIT INVARIANCE FAILS ***'}   ({time.time()-t0:.0f}s)")
    return ok


# ---------------------------------------------------------------- A1 (ceiling, generalised)

def check_generalised_ceiling(ns=(4, 5, 6), trials=250, seed=23, timeout=None):
    """W(s), the weight of the top-down canonical residue, is an upper bound on delta for ANY
    diagonal level-three gate -- not just pure cubics -- and specialises to w(F) on pure cubics
    and to e+odd on pure quadratics."""
    from experiment_mixed_nullity import build_phi
    from second_order_floor import canonical_residue_weight
    from quadratic_locus import canonical_quadratic_weight
    rng = random.Random(seed)
    ok = True
    for n in ns:
        T3 = list(itertools.combinations(range(n), 3))
        T2 = list(itertools.combinations(range(n), 2))
        viol = tot = tight = 0
        spec_cubic = spec_quad = 0
        widths = {}
        for _ in range(trials):
            triples = rng.sample(T3, rng.randint(0, len(T3)))
            pairs = rng.sample(T2, rng.randint(0, len(T2)))
            lin = [i for i in range(n) if rng.random() < 0.5]
            if not (triples or pairs or lin):
                continue
            coeff, _ = build_phi(triples, pairs, lin, n)
            mm = moments_from_coeff(coeff, n)
            W, _ = weight_of_moment_free_rep(mm, n)
            delta, st, _ = tcount(mm, n, timeout=timeout)
            assert st == "OPTIMAL"
            tot += 1
            if W < delta:
                viol += 1
            if W == delta:
                tight += 1
            widths[W - delta] = widths.get(W - delta, 0) + 1
        # specialisations
        for _ in range(60):
            tri = rng.sample(T3, rng.randint(1, len(T3)))
            W, _ = weight_of_moment_free_rep(moments_from_cubic(tri, n), n)
            spec_cubic += (W == canonical_residue_weight(tri, n))
            prs = rng.sample(T2, rng.randint(1, len(T2)))
            W, _ = weight_of_moment_free_rep(moments_from_quadratic(prs, n), n)
            spec_quad += (W == canonical_quadratic_weight(prs, n))
        good = viol == 0 and spec_cubic == 60 and spec_quad == 60
        ok &= good
        print(f"   n={n}: {tot} mixed gates | W(s) < delta VIOLATIONS {viol}"
              f" | W(s)==delta on {tight}/{tot}"
              f" | specialises to w(F) {spec_cubic}/60, to e+odd {spec_quad}/60"
              f"   {'ok' if good else '*** FAILS ***'}")
        print(f"        bracket-top slack W-delta: {dict(sorted(widths.items()))}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ns = (4, 5) if args.quick else (4, 5, 6)
    print(__doc__.strip())
    print("=" * 78)
    r = {}
    print("\n[action] the GL(n,2) action used by this audit")
    r["action"] = check_action(ns)
    print("\n[C3] moment constructors vs the validated residue path")
    r["C3"] = check_constructors(ns if len(ns) < 3 else (4, 5))
    print("\n[A1] generalised ceiling W(s) on mixed-degree gates")
    r["A1"] = check_generalised_ceiling(ns, trials=120 if args.quick else 250)
    print("\n[C1] orbit-invariance of delta, d and the rank spectrum")
    r["C1"] = check_orbit_invariance(ns, samples=4 if args.quick else 6)
    print("\n" + "=" * 78)
    for k, v in r.items():
        print(f"   {k}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(r.values()) else 1)
