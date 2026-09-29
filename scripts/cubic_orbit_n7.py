"""
The atlas at n=7, where the 2^35 cubic forms cannot be enumerated.

Strategy.  Every GL(7,2)-orbit contains a representative of minimal monomial support, and the
n<=6 census shows those minima are small (|f| <= 5 covers every orbit at n=6).  So we
enumerate all forms with |f| <= K, separate them into orbits with a cheap complete-looking
invariant, and check that the number of classes matches the count in the literature
(11 nonzero orbits of cubic forms in 7 variables, Hou; see also Brier-Langevin ITW 2003 for
n=9 and Khoruzhii-Gelss-Pokutta arXiv:2606.28473 for n=10).

The invariant used is the DERIVATIVE SPECTRUM: for every direction a != 0, the derivative
Delta_a f is a quadratic form (this is exactly the cubic-corner tameness of rem:cubicspecial),
so it is classified over F_2 by its rank and Arf invariant.  The multiset

    { (rank(Delta_a f), Arf(Delta_a f)) : a in F_2^n \\ {0} }

is a GL(n,2)-invariant, and it is computed in O(2^n n^2).  We verify on n<=6, where the true
orbits are known exactly from cubic_orbit_atlas.py, that this invariant is COMPLETE (separates
exactly the true orbits and no more) before trusting it at n=7.
"""
import argparse, itertools, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubic_orbit_atlas import monomials, orbits, form_triples, radical_dim


"""Truth tables are bit-packed into a single Python int of 2^n bits: bit y of `tt` is f(y).
Shifting by a in the ARGUMENT (y -> y ^ a) is a bit-permutation built from block swaps, so a
whole derivative is a handful of big-int operations rather than a 2^n-long loop.  That is what
makes the |f| <= 5 sweep at n=7 (324k forms x 127 directions) finish in minutes."""

_MASKS = {}


def _xor_shift(tt, a, n):
    """The packed truth table of y |-> f(y ^ a), by swapping blocks of size 2^k for each bit
    k of a."""
    key = (n,)
    if key not in _MASKS:
        # mask[k] selects the positions y with bit k clear
        ms = []
        for k in range(n):
            blk = (1 << (1 << k)) - 1            # 2^k ones
            period = 1 << (k + 1)
            unit = blk                            # low half of each period-block set
            m = 0
            rep = (1 << n) // period
            for _ in range(rep):
                m = (m << period) | unit
            ms.append(m)
        _MASKS[key] = ms
    ms = _MASKS[key]
    for k in range(n):
        if (a >> k) & 1:
            sh = 1 << k
            lo = tt & ms[k]
            hi = (tt >> sh) & ms[k]
            tt = (lo << sh) | hi
    return tt


def eval_form_packed(triples, n):
    """Packed truth table of f over F_2^n."""
    tt = 0
    for x in range(1 << n):
        v = 0
        for (a, b, c) in triples:
            if (x >> a) & 1 and (x >> b) & 1 and (x >> c) & 1:
                v ^= 1
        if v:
            tt |= 1 << x
    return tt


def derivative_rank_arf_packed(tt, a, n, N):
    """Delta_a f is quadratic; return (rank, arf-type) of it.  rank is that of the alternating
    form B(u,v) = D(0) ^ D(u) ^ D(v) ^ D(u^v); the type is read off the bias of D, which for a
    rank-R quadratic is 2^{n-1} -/+ 2^{n-1-R/2}."""
    D = tt ^ _xor_shift(tt, a, n)
    def Dv(y):
        return (D >> y) & 1
    d0 = Dv(0)
    # The polarization B(u,v) = D(0)+D(u)+D(v)+D(u+v) is ALTERNATING, so B(e_i,e_i)=0 must be
    # set explicitly: at i=j the last term degenerates to D(e_i) and the formula would return
    # D(0)+D(e_i) instead of 0.  (Symptom of getting this wrong: odd ranks appear, which is
    # impossible for an alternating form.)
    rows = []
    for i in range(n):
        row = 0
        for j in range(n):
            if i == j:
                continue
            b = d0 ^ Dv(1 << i) ^ Dv(1 << j) ^ Dv((1 << i) | (1 << j))
            row |= b << j
        rows.append(row)
    rr = list(rows)
    rank = 0
    for col in range(n):
        piv = next((k for k in range(rank, n) if (rr[k] >> col) & 1), None)
        if piv is None:
            continue
        rr[rank], rr[piv] = rr[piv], rr[rank]
        for k in range(n):
            if k != rank and (rr[k] >> col) & 1:
                rr[k] ^= rr[rank]
        rank += 1
    ones = bin(D).count("1")
    half = N >> 1
    if d0:
        ones = N - ones
    typ = 0 if ones < half else (1 if ones > half else 2)
    return rank, typ


def derivative_spectrum(triples, n):
    """The multiset { rank(Delta_a f) : a != 0 }, a GL(n,2) invariant.

    RANK ONLY.  The bias/Arf "type" returned alongside the rank is NOT invariant here and must
    not be included: we work with the cubic form modulo lower degree (the degree-<=2 debris of
    f o L enters the phase at coefficient 4, i.e. as Clifford CZ/Z, and changes neither delta nor
    the moments), and discarding it shifts the derivative's AFFINE part.  The polarization
    B(u,v)=D(0)+D(u)+D(v)+D(u+v) is blind to affine terms, so the rank survives; the bias is not,
    so the type does not.  Measured: rank-only invariant 90/90 under random L, (rank,type) only
    15/90.  Rank alone is still complete on n=4,5,6 against the exact orbits."""
    tt = eval_form_packed(triples, n)
    N = 1 << n
    spec = {}
    for a in range(1, N):
        r, _ = derivative_rank_arf_packed(tt, a, n, N)
        spec[r] = spec.get(r, 0) + 1
    return tuple(sorted(spec.items()))


def validate_invariant(ns=(4, 5, 6)):
    """Check on n<=6 that the derivative spectrum separates exactly the true GL-orbits."""
    print("== is the derivative spectrum a COMPLETE orbit invariant? (checked on true orbits) ==")
    ok = True
    for n in ns:
        reps, mons = orbits(n, verbose=False)
        sigs = {}
        for f, size in reps:
            tri = form_triples(f, mons)
            if not tri:
                continue
            s = derivative_spectrum(tri, n)
            sigs.setdefault(s, []).append(f)
        n_orbits = sum(1 for f, _ in reps if f)
        good = len(sigs) == n_orbits
        ok &= good
        print(f"   n={n}: {n_orbits} true orbits -> {len(sigs)} distinct spectra"
              f"   {'COMPLETE' if good else '*** COLLIDES: not complete ***'}")
    return ok


def enumerate_small_support(n, K, timeout=None, expected=None):
    from sat_tcount import moments_from_cubic, tcount
    mons = monomials(n)
    M = len(mons)
    print(f"\n== n={n}: forms with |f| <= {K} out of C({M},k) monomials ==")
    classes = {}
    t0 = time.time()
    total = 0
    for k in range(1, K + 1):
        for combo in itertools.combinations(range(M), k):
            tri = [mons[i] for i in combo]
            total += 1
            s = derivative_spectrum(tri, n)
            if s not in classes:
                classes[s] = tri
    print(f"   scanned {total} forms, found {len(classes)} distinct derivative spectra"
          f"   ({time.time()-t0:.1f}s)")
    if expected is not None:
        print(f"   literature count for n={n}: {expected} nonzero orbits"
              f"   {'-- ALL FOUND' if len(classes) == expected else '-- MISMATCH, raise K'}")
    rows = []
    for s, tri in classes.items():
        d = radical_dim(tri, n)
        nu = n - d
        floor = 2 * nu + 1
        mm = moments_from_cubic(tri, n)
        delta, st, _ = tcount(mm, n, timeout=timeout)
        assert st == "OPTIMAL", f"not optimal for {tri}: {st}"
        assert floor <= delta, f"FLOOR VIOLATED {tri}: {floor} > {delta}"
        rows.append((len(tri), d, nu, floor, delta, tri))
    rows.sort()
    print(f"   {'|f|':>4s} {'d':>2s} {'nu':>3s} {'floor':>6s} {'delta':>6s} {'gap':>4s}  representative")
    for k, d, nu, floor, delta, tri in rows:
        rep = "+".join("".join(str(i) for i in t) for t in tri)
        print(f"   {k:4d} {d:2d} {nu:3d} {floor:6d} {delta:6d} {delta-floor:4d}  {rep[:48]}")
    tight = sum(1 for r in rows if r[3] == r[4])
    print(f"   FLOOR IS EXACT on {tight}/{len(rows)} orbits")
    gaps = {}
    for r in rows:
        gaps[r[4] - r[3]] = gaps.get(r[4] - r[3], 0) + 1
    print(f"   gap distribution: {dict(sorted(gaps.items()))}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--expected", type=int, default=11)
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()
    print(__doc__.strip())
    print("=" * 78)
    if not validate_invariant():
        print("\nWARNING: the invariant is not complete on n<=6; n=7 classes may merge orbits.")
    enumerate_small_support(args.n, args.K, args.timeout, args.expected)
