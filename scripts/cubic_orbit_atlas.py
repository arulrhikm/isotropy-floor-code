"""
The cubic orbit atlas: exact delta on EVERY GL(n,2)-orbit of cubic forms.

The paper's exactness claims are stated on named families (parallel Toffoli layers, fan-out
layers, GF(2^k) oracles).  This replaces "tight on named families" with a census: for every
orbit of cubic forms, compare the floor 2(n-d)+1 against the exact T-count.

Why orbits are the right unit.  delta is invariant under CNOT relabelling x -> Lx, and the
PURE-CUBIC LOCUS IS CLOSED under that action (quadratic_locus.py): cubic mass sits at
coefficient 4, and 4z mod 8 depends only on z mod 2, so cubic mass is blind to the Z_8 carries
that a change of variables produces.  The induced action on the form is the standard one on
RM(3,n)/RM(2,n) -- f |-> the degree-3 part of f o L -- because the lower-degree debris of the
substitution enters the phase with coefficient 4, i.e. as CZ and Z gates, which are Clifford
and contribute no moments.  So each GL(n,2)-orbit has a single well-defined delta.

Orbits are computed here from scratch by BFS under transvection generators x_i <- x_i + x_j
(these generate GL(n,2)), so n <= 6 needs no external data.  Exact delta comes from the
XOR-SAT oracle (sat_tcount.py).
"""
import argparse, itertools, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def monomials(n):
    return list(itertools.combinations(range(n), 3))


def transvection_map(n, i, j, mons, index):
    """The permutation-with-cancellation induced on cubic monomials by x_i <- x_i + x_j.
    Returns a list `img` of bitmasks: monomial k maps to the XOR of the monomials in img[k].
    A term whose variable multiset repeats drops to degree <= 2, i.e. to zero mod RM(2,n)."""
    img = []
    for m in mons:
        out = 1 << index[m]                      # the monomial itself survives
        if i in m:
            rest = tuple(sorted(set(m) - {i}))
            if j not in rest:                    # else x_j^2 = x_j lowers the degree: drop
                t = tuple(sorted(rest + (j,)))
                out ^= 1 << index[t]
        img.append(out)
    return img


def apply_map(f, img):
    out = 0
    k = 0
    ff = f
    while ff:
        if ff & 1:
            out ^= img[k]
        ff >>= 1
        k += 1
    return out


def orbits(n, verbose=True):
    """All GL(n,2)-orbits on cubic forms in n variables, as (representative, size)."""
    mons = monomials(n)
    index = {m: k for k, m in enumerate(mons)}
    N = 1 << len(mons)
    gens = [transvection_map(n, i, j, mons, index)
            for i in range(n) for j in range(n) if i != j]
    seen = bytearray(N)
    reps = []
    t0 = time.time()
    for start in range(N):
        if seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for g in gens:
                nxt = apply_map(cur, g)
                if not seen[nxt]:
                    seen[nxt] = 1
                    stack.append(nxt)
        reps.append((start, size))
    if verbose:
        print(f"   n={n}: {len(reps)} orbits over {N} cubic forms  ({time.time()-t0:.1f}s)"
              f"   [sizes sum to {sum(s for _, s in reps)}]")
    assert sum(s for _, s in reps) == N
    return reps, mons


def form_triples(f, mons):
    return [mons[k] for k in range(len(mons)) if (f >> k) & 1]


def radical_dim(triples, n):
    """d = dim {w : sum_a w_a T_{aqr} = 0 for all q,r}, the cubic-tensor radical."""
    rows = []
    for q in range(n):
        for r in range(n):
            if q == r:
                continue
            row = 0
            for a in range(n):
                if a == q or a == r:
                    continue
                t = tuple(sorted((a, q, r)))
                if t in set(map(tuple, map(sorted, triples))):
                    row |= 1 << a
            if row:
                rows.append(row)
    # nullity of the row space
    rank = 0
    rr = list(rows)
    for col in range(n):
        piv = next((i for i in range(rank, len(rr)) if (rr[i] >> col) & 1), None)
        if piv is None:
            continue
        rr[rank], rr[piv] = rr[piv], rr[rank]
        for i in range(len(rr)):
            if i != rank and (rr[i] >> col) & 1:
                rr[i] ^= rr[rank]
        rank += 1
    return n - rank


def atlas(n, timeout=None, show=40):
    from sat_tcount import moments_from_cubic, tcount
    reps, mons = orbits(n)
    rows = []
    for f, size in reps:
        tri = form_triples(f, mons)
        if not tri:
            continue
        d = radical_dim(tri, n)
        nu = n - d
        floor = 2 * nu + 1                      # prop:nullity, valid for every pure cubic
        mm = moments_from_cubic(tri, n)
        t0 = time.time()
        delta, st, _ = tcount(mm, n, timeout=timeout)
        assert st == "OPTIMAL", f"not proved optimal for {tri}: {st}"
        assert floor <= delta, f"FLOOR VIOLATED: {tri} floor={floor} delta={delta}"
        rows.append((size, len(tri), d, nu, floor, delta, tri))
    rows.sort(key=lambda r: (-r[0],))
    tight = sum(1 for r in rows if r[4] == r[5])
    print(f"\n== n={n}: {len(rows)} nonzero orbits ==")
    print(f"   {'orbit size':>11s} {'|f|':>4s} {'d':>2s} {'nu':>3s} {'floor 2nu+1':>11s}"
          f" {'delta':>6s} {'gap':>4s}  representative")
    for size, k, d, nu, floor, delta, tri in rows[:show]:
        rep = "+".join("".join(str(i) for i in t) for t in tri)
        print(f"   {size:11d} {k:4d} {d:2d} {nu:3d} {floor:11d} {delta:6d} {delta-floor:4d}"
              f"  {rep[:44]}")
    if len(rows) > show:
        print(f"   ... {len(rows)-show} more orbits")
    covered = sum(r[0] for r in rows if r[4] == r[5])
    total = sum(r[0] for r in rows)
    print(f"   FLOOR IS EXACT on {tight}/{len(rows)} orbits"
          f"  =  {covered}/{total} forms ({100*covered/total:.1f}% by measure)")
    gaps = {}
    for r in rows:
        gaps[r[5] - r[4]] = gaps.get(r[5] - r[4], 0) + 1
    print(f"   gap distribution (delta - floor), by orbit: {dict(sorted(gaps.items()))}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[4, 5, 6])
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()
    print(__doc__.strip())
    print("=" * 78)
    for n in args.n:
        atlas(n, timeout=args.timeout)
