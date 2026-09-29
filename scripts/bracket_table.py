"""
Two-sided O(n^3) brackets on the application families of the paper.

The canonical-residue ceiling (prop:ceiling, which applies to every diagonal level-three gate)
supplies an upper bound at the same cost as the floor, so each application family gets a bracket

        2(n - d*) - r   <=   delta   <=   W(s),

both evaluated in O(n^3) from the phase polynomial, with no decoding.  Where exact decoding is
feasible we print delta too, so the bracket can be scored rather than merely quoted.

Cross-check built in: the GF(2^k) row must reproduce W_k = (7, 17, 36, 58, 90, 123) as computed
independently by `gf2m_floor.py`, which derives the same numbers from a bespoke "merged canonical
residue" for that one family.  Agreement is evidence that the general ceiling
specialises to that construction.
"""
import argparse, itertools, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_amdist as AM
from sat_tcount import moments_from_coeff, weight_of_moment_free_rep, tcount
from sharp_general_bound import genlb_bound
from experiment_mixed_nullity import build_phi
import gf2m_floor as GF


def bracket(coeff, n, want_delta=True, timeout=None):
    """(floor, ceiling, delta_or_None, pure) from the phase-polynomial coefficient dict.

    The floor is thm:genLB's 2(n - d*) - r, PLUS the +1 that
    prop:nullity supplies on the pure-cubic locus (delta >= 2nu+1 there, for every d).  Omitting
    that +1 understates every pure-cubic row by one and is the difference between the floor being
    exact and being off-by-one on the line, star and e_3 families."""
    mm = moments_from_coeff(coeff, n)
    ceiling, _ = weight_of_moment_free_rep(mm, n)
    floor, _old, dp, r, eps = genlb_bound(coeff, n)
    pure = (r == 0 and eps == 0)          # order-<=2 moments all vanish
    if pure:
        floor += 1                         # prop:nullity: delta >= 2nu+1
    delta = None
    if want_delta:
        d, st, _ = tcount(mm, n, timeout=timeout)
        delta = d if st == "OPTIMAL" else None
    return floor, ceiling, delta, pure


def row(name, coeff, n, known=None, want_delta=True, timeout=None):
    fl, ce, dl, _pure = bracket(coeff, n, want_delta, timeout)
    ds = str(dl) if dl is not None else "-"
    ks = "" if known is None else f"  [known {known}]"
    flag = ""
    if dl is not None and not (fl <= dl <= ce):
        flag = "   *** BRACKET VIOLATED ***"
    if known is not None and dl is not None and dl != known:
        flag += "   *** DISAGREES WITH PAPER ***"
    print(f"   {name:38s} n={n:2d}  [{fl:3d}, {ce:3d}]  width {ce-fl:3d}   delta={ds:>3s}{ks}{flag}")
    return fl, ce, dl, flag == ""


# ---------------------------------------------------------------- families

def toffoli_layers(ms=(1, 2, 3, 4, 5), timeout=None):
    print("\n== parallel Toffoli layer: m disjoint CCZ on n=3m (thm:parccz, delta = 6m+1) ==")
    ok = True
    for m in ms:
        n = 3 * m
        c = {}
        for i in range(m):
            AM.ccz_on(c, (3 * i, 3 * i + 1, 3 * i + 2))
        known = 6 * m + 1
        _, _, _, good = row(f"m={m}", c, n, known=known,
                            want_delta=(n <= 6), timeout=timeout)
        ok &= good
    print("   reading: the floor is exact (6m+1); the ceiling 7m is loose by m-1, since the")
    print("            canonical residue pays 7 per block and misses the shared-parity saving.")
    return ok


def fanout_layers(ms=(1, 2, 3, 4), timeout=None):
    print("\n== fan-out layer: m CCZ through one shared control, n=2m+1 (thm:fanout, 4m+3) ==")
    ok = True
    for m in ms:
        n = 2 * m + 1
        c = {}
        for i in range(m):
            AM.ccz_on(c, (0, 2 * i + 1, 2 * i + 2))
        _, _, _, good = row(f"m={m}", c, n, known=4 * m + 3,
                            want_delta=(n <= 6), timeout=timeout)
        ok &= good
    return ok


def field_oracles(ks=(1, 2, 3, 4, 5, 6), timeout=None):
    """The cross-check that matters: reproduce gf2m_floor.py's W_k from the general ceiling."""
    print("\n== GF(2^k) multiplication oracle (prop:fieldoracle, rem:gf2mbracket) ==")
    print("   cross-check: ceiling must equal gf2m_floor.py's bespoke W_k")
    ok = True
    for k in ks:
        n = 3 * k
        monos = GF.build_monomials(k)
        # the oracle's cubic form, as a set of triples (XOR of repeats)
        acc = set()
        for m in monos:
            acc ^= {tuple(sorted(m))}
        tri = sorted(acc)
        c = {}
        for t in tri:
            AM.ccz_on(c, t)
        fl, ce, dl, _p = bracket(c, n, want_delta=(n <= 6), timeout=timeout)
        Wk = GF.residue_weight(monos)
        agree = "" if Wk is None else ("  == gf2m W_k" if Wk == ce else f"  *** != gf2m W_k={Wk} ***")
        if Wk is not None and Wk != ce:
            ok = False
        ds = str(dl) if dl is not None else "-"
        print(f"   k={k}  n={n:2d}  floor={fl:3d} (6k+1={6*k+1:3d})  ceiling={ce:3d}"
              f"  width {ce-fl:3d}  delta={ds:>3s}{agree}")
    return ok


def qaoa_ising(cases=None, timeout=None):
    """MIXED-DEGREE cost layers -- the row that needs the general ceiling, because a
    3-local Ising layer carries quadratic mass and the pure-cubic w(F) does not apply."""
    print("\n== QAOA / Ising cost layers, mixed degree (prop:univfloor) ==")
    print("   these carry quadratic mass, so only the GENERALISED ceiling W(s) applies")
    ok = True
    if cases is None:
        cases = []
        # 3-local line with a sparse quadratic penalty, n = 5,6
        for n in (5, 6):
            tri = [(i, i + 1, i + 2) for i in range(n - 2)]
            for npen in (0, 1, 2):
                pairs = [(2 * j, 2 * j + 1) for j in range(npen)]
                cases.append((f"3-local line n={n}, {npen} penalty edge(s)", tri, pairs, [], n))
        # inhomogeneous: cubic + quadratic + local fields
        cases.append(("line n=6 + 2 edges + 3 fields",
                      [(i, i + 1, i + 2) for i in range(4)],
                      [(0, 1), (2, 3)], [0, 2, 4], 6))
    for name, tri, pairs, lin, n in cases:
        c, _ = build_phi(tri, pairs, lin, n)
        _, _, _, good = row(name, c, n, want_delta=(n <= 6), timeout=timeout)
        ok &= good
    return ok


def structured_families(timeout=None):
    print("\n== structured cubic families (prop:families) ==")
    ok = True
    for n in (5, 6):
        fam = {
            "line   x_i x_{i+1} x_{i+2}": [(i, i + 1, i + 2) for i in range(n - 2)],
            "cycle  (wrap-around)": [((i) % n, (i + 1) % n, (i + 2) % n) for i in range(n)],
            "star   x_0 x_i x_{i+1}": [(0, i, i + 1) for i in range(1, n - 1)],
            "e_3    complete cubic": list(itertools.combinations(range(n), 3)),
        }
        for name, tri in fam.items():
            c = {}
            for t in tri:
                AM.ccz_on(c, t)
            _, _, _, good = row(f"{name}  (n={n})", c, n, want_delta=(n <= 6), timeout=timeout)
            ok &= good
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    print(__doc__.strip())
    print("=" * 78)
    res = {}
    res["field"] = field_oracles(timeout=args.timeout)
    res["toffoli"] = toffoli_layers(timeout=args.timeout)
    res["fanout"] = fanout_layers(timeout=args.timeout)
    res["qaoa"] = qaoa_ising(timeout=args.timeout)
    res["families"] = structured_families(timeout=args.timeout)
    print("\n" + "=" * 78)
    for k, v in res.items():
        print(f"   {k}: {'ok' if v else 'PROBLEM'}")
    sys.exit(0 if all(res.values()) else 1)
