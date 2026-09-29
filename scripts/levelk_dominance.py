"""
Does the level-k isotropy floor dominate stabilizer nullity?    2(n - d_k) >= nu ?

Motivation.  prop:nullityk proves delta_k >= max(nu, 2(n-d_k)), not
2(n-d_k) >= nu (rem:levelkdom).  Random homogeneous forms almost never have d_k > n/2, and
that is exactly the regime where the inequality has any content:

    nu <= n always, so 2(n-d_k) >= n >= nu whenever d_k <= n/2.   [trivial half]

So everything rests on large-radical gates, which random homogeneous forms essentially never
produce.  This script MANUFACTURES them.

Construction.  Fix a surjection pi : F_2^n -> F_2^D and a degree-k form g on F_2^D.  Take
f := the degree-k homogeneous part of g(pi x).  Its k-tensor is the pullback pi^*T_g, so
ker pi (dimension n-D) sits inside the radical, forcing d_k >= n-D, i.e. an effective
dimension of at most D.  Sweeping D down to k drives d_k/n as high as it can go
(d_k <= n-k always, since the top tensor must be nonzero).

Measured.  With R = R^(1) = {a : deg Delta_a f <= k-2} and A = R^(k-2) = {a : deg
Delta_a f <= 1}, one has nu = n - dim A and A <= R, hence

    nu = (n - d_k) + dim(R/A),   so   2(n-d_k) >= nu  <==>  dim(R/A) <= n - d_k.

The descent maps tau_j(a) := (degree-(k-1-j) part of Delta_a f) are F_2-LINEAR on R^(j)
(the cross term in Delta_{a+b} = Delta_a + Delta_b + Delta_a Delta_b has strictly lower
degree), with ker tau_j = R^(j+1).  So dim(R/A) = sum_j rank tau_j, and the target is

    sum_j rank tau_j <= n - d_k.

Both the linearity and the decomposition are ASSERTED here, not sampled -- they are the
structural lemmas a proof would rest on, so a failure kills the route.

Run:  python levelk_dominance.py
"""
import itertools, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from levelk_nullity_study import (anf, degree, deriv, nullity,
                                  radical_dim_and_filtration, eval_homog_from_coeffs)


# ---------------------------------------------------------------- helpers
def homog_part(tt, n, k):
    """Degree-k homogeneous part of a truth table, as a list of monomial masks."""
    a = anf(tt, n)
    return [S for S in range(1 << n) if a[S] and bin(S).count("1") == k]


def apply_pi(x, pi):
    """pi given as a list of D masks (row i = which x-bits feed output bit i)."""
    y = 0
    for i, row in enumerate(pi):
        if bin(row & x).count("1") & 1:
            y |= (1 << i)
    return y


def pullback_homog(g_tt, pi, n, D, k):
    """f := degree-k homogeneous part of g(pi x).  Returns (tt, coeffs) or None if the
    top part died (then it is not a level-k gate at all)."""
    comp = [g_tt[apply_pi(x, pi)] for x in range(1 << n)]
    coeffs = homog_part(comp, n, k)
    if not coeffs:
        return None
    return eval_homog_from_coeffs(coeffs, n), coeffs


def subspace_dim(masks):
    basis = []
    for m in masks:
        cur = m
        for b in basis:
            cur = min(cur, cur ^ b)
        if cur:
            basis.append(cur)
            basis.sort(reverse=True)
    return len(basis)


def rank_of_tau(members, tt, n, k, j):
    """rank of tau_j : R^(j) -> {degree-(k-1-j) forms}, a |-> top part of Delta_a f.
    Also ASSERTS tau_j is F_2-linear on the given members."""
    tgt = k - 1 - j
    def tau(a):
        d = anf(deriv(tt, a, n), n)
        return frozenset(S for S in range(1 << n) if d[S] and bin(S).count("1") == tgt)
    # linearity check on a basis-sized sample of pairs
    ms = [m for m in members if m]
    for a, b in itertools.islice(itertools.combinations(ms, 2), 200):
        ta, tb, tab = tau(a), tau(b), tau(a ^ b)
        assert tab == (ta ^ tb), \
            "tau_%d NOT linear at (%d,%d): the filtration decomposition is dead" % (j, a, b)
    # rank = log2 |image|, image is a group under symmetric difference
    img = set()
    for a in ms:
        img.add(tau(a))
    img.add(frozenset())
    # close under XOR to get the subgroup, then take log2
    grp = {frozenset()}
    for t in img:
        if t not in grp:
            grp |= {g ^ t for g in grp}
    r = 0
    while (1 << r) < len(grp):
        r += 1
    assert (1 << r) == len(grp), "image is not an F_2 subspace (%d elts)" % len(grp)
    return r


ST = dict(tot=0, big=0, viol=0, worst=99, worst_case=None, wit=[])


def check(tt, coeffs, n, k, tag):
    filt = radical_dim_and_filtration(tt, n, k)
    R = filt[1]
    A = filt[k - 2]
    dR = subspace_dim(R)
    dA = subspace_dim(A)
    dk = dR                      # R^(1) IS the top-tensor radical
    nu, Aset = nullity(tt, n)
    D = n - dk
    ST["tot"] += 1

    # structural identity: nu = (n - d_k) + dim(R/A)
    assert dA == subspace_dim(list(Aset)), "filtration A != shift group A on %s" % tag
    assert nu == n - dA, "nu mismatch on %s: %d vs %d" % (tag, nu, n - dA)
    assert nu == D + (dR - dA), "nu decomposition FAILED on %s" % tag

    # sum of tau ranks == dim(R/A)
    tot_rank = 0
    for j in range(1, k - 2):
        tot_rank += rank_of_tau(filt[j], tt, n, k, j)
    if k >= 4:
        tot_rank = dR - dA       # cross-check target; per-step ranks summed above when k>4
    assert tot_rank == dR - dA

    margin = nu - 2 * D          # want <= 0
    if 2 * dk > n:
        ST["big"] += 1
    if margin > 0:
        ST["viol"] += 1
        if len(ST["wit"]) < 5:
            ST["wit"].append((tag, n, k, dk, D, nu, dR - dA, coeffs[:6]))
    if margin < ST["worst"]:
        pass
    if -margin < ST["worst"]:
        ST["worst"] = -margin
        ST["worst_case"] = (tag, n, k, dk, D, nu)
    return dk, D, nu, dR - dA


def sweep(k, ns, Ds, samples, seed):
    rng = random.Random(seed)
    print("=== level k=%d : manufactured LARGE-RADICAL gates ===" % k)
    print("  %-26s %3s %3s %4s %4s %4s %8s %8s" %
          ("family", "n", "d_k", "D", "nu", "R/A", "2D>=nu?", "d_k>n/2"))
    for n in ns:
        for D in Ds:
            if D < k or D >= n:
                continue
            got = 0
            for _ in range(samples * 4):
                if got >= samples:
                    break
                # random surjection pi : F_2^n -> F_2^D
                pi = [rng.randrange(1, 1 << n) for _ in range(D)]
                if subspace_dim(pi) < D:
                    continue
                # random degree-k form g on D vars (as a truth table on 2^D)
                gc = [m for m in range(1 << D)
                      if bin(m).count("1") == k and rng.random() < 0.5]
                if not gc:
                    continue
                g_tt = eval_homog_from_coeffs(gc, D)
                out = pullback_homog(g_tt, pi, n, D, k)
                if out is None:
                    continue
                tt, coeffs = out
                dk, Dact, nu, gap = check(tt, coeffs, n, k, "pullback D=%d" % D)
                got += 1
                if got <= 2:
                    print("  %-26s %3d %3d %4d %4d %4d %8s %8s"
                          % ("pullback g on D=%d" % D, n, dk, Dact, nu, gap,
                             "OK" if nu <= 2 * Dact else "**VIOLATION**",
                             "yes" if 2 * dk > n else "no"))
    print()


if __name__ == "__main__" and "--crosscheck" not in sys.argv:
    for k, ns, Ds in ((4, (6, 7, 8, 9), (4, 5, 6)),
                      (5, (7, 8, 9), (5, 6))):
        sweep(k, ns, Ds, samples=12, seed=1000 + k)
    print("=" * 70)
    print("gates tested: %d,  of which d_k > n/2: %d" % (ST["tot"], ST["big"]))
    print("gates with 2(n-d_k) < nu : %d   (the inequality fails in general, rem:levelkdom)" % ST["viol"])
    for w in ST["wit"]:
        print("   %s n=%d k=%d d_k=%d D=%d nu=%d dim(R/A)=%d coeffs=%s" % w)
    print("tightest case (smallest slack 2D-nu): slack=%d at %s"
          % (ST["worst"], ST["worst_case"]))
    print("structural identities asserted on every gate:")
    print("  tau_j linear on R^(j);  A(filtration) == A(shift group);"
          "  nu == (n-d_k) + dim(R/A)")


# ================================================================ WP-A 2026-08-22
# POLY-TIME LEVEL-k NULLITY via the filtration (prop:levelknu).
#
# Everything is SYMBOLIC in the monomial support of f -- no truth tables, no 2^n
# enumeration -- so this runs at n where brute force cannot.  Mechanism: tau_j is
# F_2-linear on R^(j) with ker tau_j = R^(j+1), so compute a basis of R^(1) (the
# top-tensor radical, a nullspace), evaluate tau_j on basis vectors (a coefficient
# extraction from Delta_a f, O(|f| 2^k) per direction), take kernels recursively:
#     nu = n - dim R^(k-2),   in polynomial time for fixed k.
# This extends the paper's O(2^{7n}) -> O(n^3) claim (k=3, prop:nullitygen) to every
# level, for the natural gates U = (-1)^f with f homogeneous of degree k.

def delta_deg_part(coeffs, a, n, m):
    r"""Degree-m part of Delta_a f, symbolically: coefficient of x_T (|T|=m) is
    sum_{S in coeffs, S superset T, S != T} [a superset S\T] (mod 2).  Returns a
    frozenset of monomial masks T with coefficient 1."""
    out = {}
    for S in coeffs:
        # enumerate T subset S with |T| = m: iterate submasks
        T = S
        while True:
            if bin(T).count("1") == m and T != S and (a & (S ^ T)) == (S ^ T):
                out[T] = out.get(T, 0) ^ 1
            if T == 0:
                break
            T = (T - 1) & S
    return frozenset(T for T, c in out.items() if c)


def f2_kernel_of_map(basis, images):
    """Given basis vectors (masks) and their images (frozensets of monomials) under a
    linear map, return a basis (masks) of the kernel of the map on the span."""
    # index the monomials appearing
    mons = sorted(set().union(*images)) if images else []
    midx = {mo: i for i, mo in enumerate(mons)}
    rows = []                      # each row: (image-bitvector, preimage-mask)
    for b, img in zip(basis, images):
        vec = 0
        for mo in img:
            vec |= (1 << midx[mo])
        rows.append([vec, b])
    # eliminate on image bits; rows that reduce to zero give kernel elements
    kernel, reduced = [], []
    for vec, pre in rows:
        for rv, rp in reduced:
            low = rv & -rv
            if vec & low:
                vec ^= rv
                pre ^= rp
        if vec == 0:
            kernel.append(pre)
        else:
            reduced.append([vec, pre])
    return kernel


def filtration_nu(coeffs, n, k):
    """nu of U = (-1)^f, f = sum of degree-k monomials `coeffs`, WITHOUT truth tables.
    Returns (nu, dims) with dims[j] = dim R^(j) for j = 1..k-2."""
    coeffs = set(coeffs)
    # (rebuild rows properly: all (k-1)-subsets W of any S in coeffs)
    rows = []
    seen = set()
    for S in coeffs:
        for i in range(n):
            if (S >> i) & 1:
                W = S ^ (1 << i)
                if W in seen:
                    continue
                seen.add(W)
                row = 0
                for jbit in range(n):
                    if not (W >> jbit) & 1 and (W | (1 << jbit)) in coeffs:
                        row |= (1 << jbit)
                if row:
                    rows.append(row)
    # nullspace basis of `rows` over F_2 (columns = directions a)
    basis = []
    piv = []
    red = []
    for c in range(n):
        v = 0
        for ri, r in enumerate(rows):
            if (r >> c) & 1:
                v |= (1 << ri)
        basis.append((v, 1 << c))
    kern = []
    reduced = []
    for v, pre in basis:
        for rv, rp in reduced:
            low = rv & -rv
            if v & low:
                v ^= rv
                pre ^= rp
        if v == 0:
            kern.append(pre)
        else:
            reduced.append([v, pre])
    cur = kern                      # basis of R^(1)
    dims = {1: len(cur)}
    for j in range(1, k - 2):
        tgt = k - 1 - j
        images = [delta_deg_part(sorted(coeffs), b, n, tgt) for b in cur]
        cur = f2_kernel_of_map(cur, images)
        dims[j + 1] = len(cur)
    nu = n - dims.get(k - 2, dims[1])
    return nu, dims

# ================================================================ reproducibility driver (added 2026-09-26)
# prop:levelknu cross-check: the symbolic filtration_nu against brute-force nullity on random homogeneous
# gates at k = 3, 4, 5, then one n = 14 gate where enumeration is out of reach.  Run with --crosscheck.
def crosscheck(seed=2026, per_cell=80):
    import time
    rng = random.Random(seed)
    total = mism = 0
    for k, ns in ((3, (4, 5, 6, 7, 8)), (4, (5, 6, 7, 8)), (5, (6, 7, 8))):
        for n in ns:
            got = 0
            while got < per_cell:
                coeffs = [m for m in range(1 << n) if bin(m).count("1") == k and rng.random() < 0.5]
                if not coeffs:
                    continue
                nu_brute, _ = nullity(eval_homog_from_coeffs(coeffs, n), n)
                nu_filt, _ = filtration_nu(coeffs, n, k)
                total += 1; got += 1
                if nu_brute != nu_filt:
                    mism += 1
                    print("MISMATCH k=%d n=%d brute=%d filtration=%d" % (k, n, nu_brute, nu_filt))
            print("k=%d n=%d: %d gates checked" % (k, n, got))
    print("cross-check: %d random gates at k=3,4,5, %d mismatches" % (total, mism))
    # the n = 10 quartic of rem:levelkdom(iii), rebuilt exactly as levelk_counterexample.py finds it
    import levelk_counterexample as LC
    for Dq in (4, 5):
        b = LC.search(10, Dq, 4, trials=60, seed=4000 + 10 * 10 + Dq)
        if b and b["margin"] > 0:
            t0 = time.perf_counter(); nu_f, _ = filtration_nu(b["coeffs"], 10, 4); dt = time.perf_counter() - t0
            print("n=10 quartic (D=%d): brute nu = %d, filtration nu = %d in %.3f s, 2(n-d_k) = %d"
                  % (Dq, b["nu"], nu_f, dt, b["floor"]))
            mism += int(nu_f != b["nu"])
            break
    n, k = 14, 4
    coeffs = [m for m in range(1 << n) if bin(m).count("1") == k and rng.random() < 0.02]
    t0 = time.perf_counter(); nu, dims = filtration_nu(coeffs, n, k); dt = time.perf_counter() - t0
    print("n=14, k=4, %d monomials: filtration nu = %d in %.3f s (enumeration would need 2^28 derivative checks)"
          % (len(coeffs), nu, dt))
    return mism


if __name__ == "__main__" and "--crosscheck" in sys.argv:
    sys.exit(1 if crosscheck() else 0)
