"""
A Delsarte LP that reads the slice-rank spectrum (fact F1), tested against exact delta.

For a t-gate {CNOT,T} representation Y of a pure-cubic gate, C = colspace(Y) is an even
self-orthogonal code of length t and dimension n - dim K (= n when d = 0), and
      T(w) = Y^T diag(c(w)) Y,   so   wt(c(w)) >= rank T(w)   for every w in F_2^n.   (F1)
So the number of w with wt(c(w)) <= r is at most the number of w with rank T(w) <= r: the
slice-rank spectrum caps the low end of the weight distribution of C.  With the MacWilliams
identities, C subseteq C^perp (A^perp_r >= A_r), 1 in C^perp and (when d = 0) 1 not in C,
this is a linear program in the weight distribution A_r.  t_LP = least t for which it is
feasible.  A valid lower bound on delta; the question is whether it ever beats 2(n-d)+1.

Run:  python rank_spectrum_lp.py            (census n = 4,5,6 with exact delta, then U_2..U_4)
      python rank_spectrum_lp.py --no-census
"""
import argparse, itertools, os, sys, time
from math import comb
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_amdist import ccz_on, residue_vec, rm_generators, min_coset_weight
from field_oracle import mult_tensor_blocks
from cubic_orbit_atlas import orbits, form_triples, radical_dim


def f2rank(rows):
    rows = [int(r) for r in rows if r]
    rank = 0
    while rows:
        piv = min(rows)
        rows.remove(piv)
        rank += 1
        low = piv & -piv
        rows = [r ^ piv if r & low else r for r in rows]
        rows = [r for r in rows if r]
    return rank


def slice_rank(T, w, n):
    M = [0] * n
    for (a, b, c) in T:
        for (p, q, r) in ((a, b, c), (b, a, c), (c, a, b)):
            if (w >> p) & 1:
                M[q] ^= 1 << r
                M[r] ^= 1 << q
    return f2rank(M)


def rank_spectrum(T, n):
    """counts[r] = #{w != 0 : rank T(w) = r}."""
    counts = {}
    for w in range(1, 1 << n):
        r = slice_rank(T, w, n)
        counts[r] = counts.get(r, 0) + 1
    return counts


SLACK = 2   # F1' ; set to 0 for the weaker F1 (wt >= rank)
EXTEND = False   # --extend: add the constraints of the extended code C + <1>


def griesmer(dim, dmin):
    return sum(-(-dmin // (1 << i)) for i in range(dim))


def krawtchouk(t):
    """K[r][s] = Krawtchouk polynomial K_r(s) of length t."""
    K = np.zeros((t + 1, t + 1))
    for r in range(t + 1):
        for s in range(t + 1):
            K[r, s] = sum((-1) ** j * comb(s, j) * comb(t - s, r - j) for j in range(r + 1))
    return K


def lp_feasible(t, n, d, spectrum):
    """Is there a weight distribution of a code C of length t consistent with everything we know?
    Variables B_r = #{w in F_2^n : wt c(w) = r}, r = 0..t (B_0 = 2^{dim K}, dim K in [0, d]).
    For d = 0: B = A, the weight distribution of C, |C| = 2^n."""
    if d != 0:
        return None            # this experiment is about the d = 0 locus
    size = 2 ** n
    K = krawtchouk(t)
    nv = t + 1
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    # total
    A_eq.append([1.0] * nv); b_eq.append(size)
    # A_0 = 1
    row = [0.0] * nv; row[0] = 1.0; A_eq.append(row); b_eq.append(1.0)
    # odd weights and weight t vanish (even code; 1 not in C)
    for r in range(1, t + 1):
        if r % 2 == 1 or r == t:
            row = [0.0] * nv; row[r] = 1.0; A_eq.append(row); b_eq.append(0.0)
    # spectrum caps (F1'): wt c(w) >= rank T(w) + 2 for c(w) != 0, since the rows in the support
    # of c(w) sum to zero (isotropy) and both numbers are even.  So
    #   sum_{r' <= r} A_{r'} <= 1 + #{w != 0 : rank T(w) <= r - 2}.
    for r in range(0, t + 1):
        cum = sum(v for rk, v in spectrum.items() if rk <= r - SLACK)
        row = [1.0 if r2 <= r else 0.0 for r2 in range(nv)]
        A_ub.append(row); b_ub.append(1 + cum)
    # MacWilliams: A^perp_r = (1/|C|) sum_s A_s K_r(s) >= max(A_r, 0), and A^perp_t >= 1
    for r in range(t + 1):
        coefs = [K[r, s] / size for s in range(nv)]
        # -A^perp_r <= 0
        A_ub.append([-c for c in coefs]); b_ub.append(0.0)
        # A_r - A^perp_r <= 0
        row = [-c for c in coefs]; row[r] += 1.0
        A_ub.append(row); b_ub.append(0.0)
    coefs = [K[t, s] / size for s in range(nv)]
    A_ub.append([-c for c in coefs]); b_ub.append(-1.0)
    if EXTEND:
        # C' = C + <1> is a linear code of size 2|C| with distribution B_r = A_r + A_{t-r}
        # (1 is not in C).  Its MacWilliams transform is nonnegative; when t is even, 1 is
        # isotropic and C' is self-orthogonal, so also B^perp_r >= B_r.
        size2 = 2 * size
        for r in range(t + 1):
            coefs = [(K[r, s] + K[r, t - s]) / size2 for s in range(nv)]
            A_ub.append([-c for c in coefs]); b_ub.append(0.0)
            if t % 2 == 0:
                row = [-c for c in coefs]
                row[r] += 1.0
                row[t - r] += 1.0
                A_ub.append(row); b_ub.append(0.0)
    res = linprog(c=[0.0] * nv, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(0, None)] * nv,
                  method="highs")
    return res.status == 0


def t_lp(T, n, d, spectrum, tmax=None):
    floor = 2 * (n - d) + 1
    if d != 0:
        return None
    t = floor
    tmax = tmax or (7 * len(T) + 1)
    while t <= tmax:
        if lp_feasible(t, n, d, spectrum):
            return t
        t += 1
    return None


def exact_delta(T, n):
    coeff = {}
    for B in T:
        ccz_on(coeff, B)
    return min_coset_weight(residue_vec(coeff, n), rm_generators(n))


def census(ns):
    for n in ns:
        reps, mons = orbits(n, verbose=True)
        rows = []
        for f, size in reps:
            tri = form_triples(f, mons)
            if not tri:
                continue
            T = set(tuple(sorted(x)) for x in tri)
            d = radical_dim(tri, n)
            if d != 0:
                continue
            spec = rank_spectrum(T, n)
            floor = 2 * n + 1
            t0 = time.time()
            delta = exact_delta(T, n)
            tl = t_lp(T, n, d, spec)
            gr = griesmer(n, min(spec) + SLACK)
            rows.append((size, len(tri), spec, floor, gr, tl, delta, tri))
            assert tl is not None and tl <= delta, f"LP BOUND VIOLATED on {tri}: {tl} > {delta}"
            assert gr <= delta, f"GRIESMER VIOLATED on {tri}: {gr} > {delta}"
        rows.sort(key=lambda r: -r[0])
        print(f"\n== n={n}: {len(rows)} non-degenerate orbits ==")
        print(f"   {'orbit size':>11s} {'|f|':>4s}  {'floor':>5s} {'Gries':>5s} {'t_LP':>5s} {'delta':>5s}  rank spectrum")
        for size, k, spec, floor, gr, tl, delta, tri in rows:
            flag = "  <-- beats floor" if max(tl, gr) > floor else ""
            print(f"   {size:11d} {k:4d}  {floor:5d} {gr:5d} {tl:5d} {delta:5d}  {dict(sorted(spec.items()))}{flag}")


def field_oracles(ks, analytic=()):
    print("\n== GF(2^k) multiplication oracles ==")
    for k in ks:
        n = 3 * k
        if k in analytic:
            # every nonzero slice of the multiplication tensor has rank exactly 2k (kernel
            # {(u_x v/u_z, u_y v/u_z, v)}, dimension k); verified by enumeration for k <= 4
            d, spec, T = 0, {2 * k: (1 << n) - 1}, set()
        else:
            blocks = mult_tensor_blocks(k)
            T = set(tuple(sorted(b)) for b in blocks)
            d = radical_dim(blocks, n)
            spec = rank_spectrum(T, n)
        floor = 2 * (n - d) + 1
        gr = griesmer(n, min(spec) + SLACK)
        tl = t_lp(T, n, d, spec, tmax=12 * k + 20)
        print(f"   k={k} n={n} d={d}  floor {floor}  Griesmer[{n},{min(spec)+SLACK}] {gr}  t_LP {tl}"
              f"  rank spectrum {dict(sorted(spec.items()))}"
              + ("   (delta = 17)" if k == 2 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--n", type=int, nargs="*", default=[4, 5, 6])
    ap.add_argument("--k", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--analytic", type=int, nargs="*", default=[],
                    help="k values whose rank spectrum is taken from the closed form (all 2k)")
    ap.add_argument("--extend", action="store_true",
                    help="also constrain the extended code C + <1>")
    args = ap.parse_args()
    EXTEND = args.extend
    print(__doc__.strip()); print("=" * 78)
    if not args.no_census:
        census(args.n)
    field_oracles(args.k, analytic=set(args.analytic))
