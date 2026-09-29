"""
The quadratic corner: does the layer of m disjoint CS gates (rank r = 2m) cost exactly 3m?

The upper bound 3m comes from paying 3 T-gates per CS (lem:quadcost).  Equality is known for
m <= 3; for larger m it is open (rem:quadopen).  For a general pure quadratic of rank r the
analogous bound delta <= 3r/2 is false (quadratic_locus.py).

The best PROVEN floor is r:
  - stabilizer nullity gives nu = n = r on a nondegenerate quadratic;
  - the isotropy floor 2(n - d*) - r gives 2r - r = r  (thm:genLB, sharp_general_bound.py).
So the whole question is the factor 3/2 against nullity's 1.

This script tests it computationally with the XOR-SAT oracle (sat_tcount.py), which reaches
n = 8, 10 where the meet-in-the-middle decoder cannot.

It also probes the two ways the 3/2 law could fail:
  (a) NON-SYMPLECTIC representatives of the same rank -- delta is a GL(n,2) invariant, so every
      rank-r alternating form is equivalent to the normal form and must give the same answer;
      we test random ones as a consistency check on the whole pipeline.
  (b) CARRY compression -- 2u + 2v = 2(u^v) + 4uv over Z_8, so quadratic mass can generate cubic
      mass.  The Fano configuration (7 CS gates on 3 qubits = CCZ, weight 7 not 21) is the
      extreme case, but that gate is NOT pure quadratic.  We check whether any pure-quadratic
      gate beats 3r/2 -- i.e. whether carries can compress while leaving the cubic moments zero.
"""
import argparse, itertools, random, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sat_tcount import (moments_from_quadratic, moment_keys, tcount, tcount_at_most)


def symplectic_pairs(m):
    """The normal form: m disjoint CS gates on n = 2m qubits, alternating rank r = 2m."""
    return [(2 * i, 2 * i + 1) for i in range(m)]


def rank_f2(pairs, n):
    rows = [0] * n
    for (a, b) in pairs:
        rows[a] |= 1 << b
        rows[b] |= 1 << a
    r = 0
    for col in range(n):
        piv = next((i for i in range(r, n) if (rows[i] >> col) & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(n):
            if i != r and (rows[i] >> col) & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def normal_form_ladder(max_m=5, timeout=None):
    """delta of the m-fold disjoint CS layer, m = 1..max_m, against the 3m prediction."""
    print("== symplectic normal form: m disjoint CS on n=2m, predicted delta = 3m = 3r/2 ==")
    print(f"   {'m':>2s} {'n':>2s} {'r':>3s} {'nu':>3s} {'floor':>12s} {'3r/2':>5s}"
          f" {'delta':>6s} {'status':>10s} {'secs':>7s}")
    results = []
    for m in range(1, max_m + 1):
        n = 2 * m
        pairs = symplectic_pairs(m)
        r = rank_f2(pairs, n)
        mm = moments_from_quadratic(pairs, n)
        t0 = time.time()
        d, st, wit = tcount(mm, n, ub=3 * m, timeout=timeout)
        el = time.time() - t0
        # nu = n for a nondegenerate quadratic; isotropy floor 2(n - 0) - r = r
        print(f"   {m:2d} {n:2d} {r:3d} {n:3d} {r:12d} {3*m:5d} {str(d):>6s} {st:>10s} {el:6.1f}s")
        results.append((m, n, r, d, st))
    return results


def random_rank_r(n, r, rng, tries=500):
    """A random alternating form on F_2^n of exact rank r, as a pair list."""
    T2 = list(itertools.combinations(range(n), 2))
    for _ in range(tries):
        k = rng.randint(r // 2, len(T2))
        pairs = rng.sample(T2, k)
        if rank_f2(pairs, n) == r:
            return pairs
    return None


def invariance_check(n=6, r=6, samples=12, timeout=None):
    """delta is a GL(n,2) invariant, so EVERY rank-r alternating form must cost the same as the
    normal form.  A disagreement would mean the pipeline (or the claim) is wrong."""
    print(f"\n== GL(n,2)-invariance: random rank-{r} alternating forms on n={n} ==")
    rng = random.Random(29)
    ref = 3 * (r // 2)
    bad = 0
    seen = {}
    for i in range(samples):
        pairs = random_rank_r(n, r, rng)
        if pairs is None:
            continue
        mm = moments_from_quadratic(pairs, n)
        d, st, _ = tcount(mm, n, timeout=timeout)
        seen[d] = seen.get(d, 0) + 1
        flag = "" if d == ref else "   <-- DIFFERS from normal form"
        if d != ref:
            bad += 1
            print(f"   form {i}: |pairs|={len(pairs):2d} delta={d} [{st}]{flag}  pairs={pairs}")
    print(f"   {samples} random rank-{r} forms: delta distribution {dict(sorted(seen.items()))}"
          f"   (normal form = {ref})")
    return bad == 0


def sweep_all_pure_quadratics(n, timeout=None):
    """EXHAUSTIVE over every pure-quadratic gate on n qubits (2^C(n,2) alternating forms q,
    each realised as the product of CS gates over the pairs of q -- which reproduces the
    moments (s_1, s_2, s_3) = (0, q, 0) exactly, since a qubit lying in several CS gates has
    its singleton parity cancel), binned by F_2-rank.

    NOTE the subtlety this measures.  delta is invariant under CNOT relabelling x -> Lx, but
    the moment vector is NOT graded-equivariant: over Z_8, 2u + 2v = 2(u^v) + 4uv, so a change
    of variables can turn quadratic mass into CUBIC mass.  Hence the pure-quadratic locus
    (s_1 = s_3 = 0) is not a union of GL(n,2)-orbits, and delta need not depend on r alone.
    The symplectic-normal-form argument leaves the locus, so 3r/2 is an upper bound for the
    NORMAL FORM, not for every pure quadratic of that rank.  We
    therefore impose NO upper bound and report the full distribution."""
    T2 = list(itertools.combinations(range(n), 2))
    N = 1 << len(T2)
    print(f"\n== exhaustive: all {N} pure-quadratic gates on n={n} ==")
    by_r = {}
    worst = []
    t0 = time.time()
    for mask in range(N):
        pairs = [T2[i] for i in range(len(T2)) if (mask >> i) & 1]
        r = rank_f2(pairs, n)
        if r == 0:
            continue
        mm = moments_from_quadratic(pairs, n)
        d, st, _ = tcount(mm, n, timeout=timeout)       # no ub: measure the truth
        assert st == "OPTIMAL", f"not proved optimal: {st}"
        b = by_r.setdefault(r, {})
        b[d] = b.get(d, 0) + 1
        if d > 3 * (r // 2):
            worst.append((r, d, pairs))
    print(f"   {'r':>3s} {'3r/2':>5s} {'max':>4s}  delta distribution")
    ok = True
    for r in sorted(by_r):
        pred = 3 * (r // 2)
        dist = dict(sorted(by_r[r].items()))
        agree = max(dist) <= pred
        ok &= agree
        print(f"   {r:3d} {pred:5d} {max(dist):4d}  {dist}"
              f"   {'<= 3r/2' if agree else '*** EXCEEDS 3r/2 ***'}")
    if worst:
        print(f"   {len(worst)} forms exceed 3r/2; smallest witnesses:")
        for r, d, pairs in sorted(worst, key=lambda t: (len(t[2]), t[1]))[:5]:
            print(f"      r={r} delta={d} > {3*(r//2)}   q = {pairs}")
    print(f"   ({time.time()-t0:.1f}s)")
    return ok, by_r, worst


def beat_3r2_search(m, timeout=600):
    """The main test: can ANY representation of the m-fold CS layer use fewer than 3m
    T-gates?  Asks the solver to prove UNSAT at weight 3m-1."""
    n = 2 * m
    pairs = symplectic_pairs(m)
    mm = moments_from_quadratic(pairs, n)
    print(f"\n== can the m={m} CS layer (n={n}, r={n}) be done in {3*m-1} T-gates? ==")
    t0 = time.time()
    ans, st = tcount_at_most(mm, n, 3 * m - 1, timeout=timeout)
    el = time.time() - t0
    verdict = {True: f"YES -- 3r/2 is REFUTED at m={m}",
               False: f"NO (proved UNSAT) -- delta >= {3*m} confirmed",
               None: "undecided (timeout)"}[ans]
    print(f"   {verdict}   [{st}, {el:.1f}s]")
    return ans


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-m", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--exhaustive-n", type=int, nargs="*", default=[4, 5])
    ap.add_argument("--skip-exhaustive", action="store_true")
    args = ap.parse_args()

    print(__doc__.strip())
    print("=" * 78)
    normal_form_ladder(args.max_m, args.timeout)
    invariance_check(n=6, r=6, timeout=args.timeout)
    if not args.skip_exhaustive:
        for n in args.exhaustive_n:
            sweep_all_pure_quadratics(n, args.timeout)
    for m in range(2, args.max_m + 1):
        beat_3r2_search(m, timeout=args.timeout or 900)
