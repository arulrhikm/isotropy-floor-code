"""
Hunting an invariant that sees the gap the radical cannot.

The n<=6 census (cubic_orbit_atlas.py) exposes the weakness: at n=6 the three NON-degenerate
orbits all have d=0, so the floor 2(n-d)+1 reads 13 on all three, yet their exact costs are
13, 15 and 17.  Whatever explains the gap is invisible to the radical.  This script measures a
battery of candidate invariants on every orbit at n<=6 and asks two separate questions:

    (Q1) does the invariant SEPARATE the orbits?   (necessary to explain the gap)
    (Q2) does some simple formula in it stay <= delta on every orbit, i.e. is it a candidate
         FLOOR rather than merely a label?

One tempting route is ruled out analytically before we start.  Over Z,
    Delta_a phi(x) = const - 2 * sum_{y : a.y = 1} c_y chi_y(x),
an exact identity, so EVERY coefficient of the derivative phase is even: the derivative gate is
realised by S gates alone and has T-count 0.  Differencing destroys precisely the quantity being
measured, so the derivative spectrum can be an orbit invariant but cannot itself carry a cost
bound.  That is why the candidates below are built from the residue and the form, not from
derivatives of the phase.

Invariants measured
  d          radical dimension                     (the incumbent; provably blind here)
  |f|min     monomials in the minimal representative
  w_can      weight of the canonical residue: res[y] = #{S in f : y subset S} mod 2, |y| <= 3
             (this is an UPPER bound on delta, being one coset element)
  bias       |sum_x (-1)^f| , the Walsh bias of f at 0
  rkspec     multiset of ranks of the quadratic forms Delta_a f  (a != 0)
  nlow       #{a != 0 : rank(Delta_a f) <= 2}, the count of "degenerate directions"
"""
import argparse, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubic_orbit_atlas import monomials, orbits, form_triples, radical_dim
from cubic_orbit_n7 import eval_form_packed, derivative_rank_arf_packed


def canonical_residue_weight(triples, n):
    """res[y] = #{S in triples : y subset S} mod 2 for 1 <= |y| <= 3 (0 for |y| >= 4).
    Its weight is an upper bound on delta."""
    T = [frozenset(t) for t in triples]
    w = 0
    for k in (1, 2, 3):
        for A in itertools.combinations(range(n), k):
            sA = frozenset(A)
            if sum(1 for S in T if sA <= S) % 2:
                w += 1
    return w


def walsh_bias(triples, n):
    tt = eval_form_packed(triples, n)
    ones = bin(tt).count("1")
    return abs((1 << n) - 2 * ones)


def rank_spectrum(triples, n):
    tt = eval_form_packed(triples, n)
    N = 1 << n
    spec = {}
    for a in range(1, N):
        r, _ = derivative_rank_arf_packed(tt, a, n, N)
        spec[r] = spec.get(r, 0) + 1
    return spec


def survey(ns=(4, 5, 6), timeout=None):
    from sat_tcount import moments_from_cubic, tcount
    rows = []
    for n in ns:
        reps, mons = orbits(n, verbose=False)
        for f, size in reps:
            tri = form_triples(f, mons)
            if not tri:
                continue
            d = radical_dim(tri, n)
            nu = n - d
            floor = 2 * nu + 1
            delta, st, _ = tcount(moments_from_cubic(tri, n), n, timeout=timeout)
            assert st == "OPTIMAL"
            spec = rank_spectrum(tri, n)
            rows.append(dict(n=n, size=size, tri=tri, k=len(tri), d=d, nu=nu,
                             floor=floor, delta=delta, gap=delta - floor,
                             w_can=canonical_residue_weight(tri, n),
                             bias=walsh_bias(tri, n),
                             spec=spec,
                             nlow=sum(c for r, c in spec.items() if r <= 2)))
    return rows


def report(rows):
    print(f"{'n':>2s} {'|f|':>3s} {'d':>2s} {'floor':>5s} {'delta':>5s} {'gap':>3s}"
          f" {'w_can':>5s} {'bias':>5s} {'nlow':>4s}  rank spectrum")
    for r in sorted(rows, key=lambda r: (r["n"], r["delta"])):
        spec = ",".join(f"{k}:{v}" for k, v in sorted(r["spec"].items()))
        print(f"{r['n']:2d} {r['k']:3d} {r['d']:2d} {r['floor']:5d} {r['delta']:5d}"
              f" {r['gap']:3d} {r['w_can']:5d} {r['bias']:5d} {r['nlow']:4d}  {spec}")

    print("\n--- Q1: does any invariant separate the n=6 non-degenerate orbits? ---")
    nd = [r for r in rows if r["n"] == 6 and r["d"] == 0]
    for name in ("k", "w_can", "bias", "nlow"):
        vals = [r[name] for r in nd]
        deltas = [r["delta"] for r in nd]
        sep = len(set(vals)) == len(vals)
        mono = sorted(zip(vals, deltas)) == sorted(zip(vals, deltas), key=lambda t: t[1])
        print(f"   {name:6s} values {vals} vs delta {deltas}"
              f"   separates: {sep}   order-consistent: {mono}")
    specs = [tuple(sorted(r["spec"].items())) for r in nd]
    print(f"   rkspec separates: {len(set(specs)) == len(specs)}")

    print("\n--- Q2: is w_can a valid FLOOR anywhere, or only a ceiling? ---")
    bad = [r for r in rows if r["w_can"] < r["delta"]]
    print(f"   w_can < delta on {len(bad)}/{len(rows)} orbits"
          f"  (w_can is one coset element, so w_can >= delta must hold ALWAYS)")
    eq = [r for r in rows if r["w_can"] == r["delta"]]
    print(f"   w_can == delta on {len(eq)}/{len(rows)} orbits"
          f"   -> the canonical residue is already a coset leader there")
    print("\n--- candidate floors, checked for validity (must be <= delta on EVERY orbit) ---")
    cands = {
        "2(n-d)+1            (incumbent)": lambda r: 2 * (r["n"] - r["d"]) + 1,
        "w_can - 2(n-d)              ": lambda r: r["w_can"] - 2 * (r["n"] - r["d"]),
        "(w_can + 2(n-d)+1)//2       ": lambda r: (r["w_can"] + 2 * (r["n"] - r["d"]) + 1) // 2,
        "2(n-d)+1 + 2*max(0,|f|-2)   ": lambda r: 2 * (r["n"] - r["d"]) + 1 + 2 * max(0, r["k"] - 2),
    }
    for name, fn in cands.items():
        vals = [(fn(r), r["delta"]) for r in rows]
        viol = [(v, d) for v, d in vals if v > d]
        tight = sum(1 for v, d in vals if v == d)
        print(f"   {name}: violations {len(viol)}/{len(vals)}, tight {tight}"
              + ("" if not viol else f"   INVALID (e.g. {viol[0]})"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[4, 5, 6])
    args = ap.parse_args()
    print(__doc__.strip())
    print("=" * 78)
    report(survey(args.n))
