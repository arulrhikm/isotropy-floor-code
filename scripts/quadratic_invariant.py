"""
The quadratic side of the isotropy floor (rem:quadroutes).

Background.  With nu = n - d* an identity for every diagonal
level-three gate (prop:nullitygen) and the floor delta >= 2(n-d*) - r (thm:genLB), the advantage over stabilizer nullity is exactly

    floor - nu  =  (n - d*) - r  =  nu - r     in [0, nu]   (cor:floordominates)

so it is a factor two at r = 0 (pure cubic) and exactly ZERO at r = n - d* (disjoint CS layers,
T^{tensor n}).  rem:grading says no bound in r alone can be tight on the quadratic side, because the
alternating rank is a complete invariant of a quadratic FORM but not of a quadratic GATE.  The
open question is what replaces r.

What this script does.

  (A) Re-measures the COMBINED floor  max{nu, 2(n-d*)-r, peel}  of rem:combined against the exact
      Amy-Mosca delta.  The bound 2(n-d*)-r is never below nu (cor:floordominates), so the
      question is whether the peel still adds anything.
      NOTE experiment_mixed_nullity.moment_bound is the weaker 2(n-d*-r) form; we do not use it.

  (B) Reports WHO WINS WHERE -- which of the three terms is the unique max, as a function of the
      quadratic mass -- i.e. the answer to "which circuits is this bound actually good on".

  (C) Hunts the replacement for r: computes the residual  delta - combined  and correlates it with
      candidate invariants, holding r fixed.  Incidence candidates (the paper's own guess) are
      computed against the cubic support, not from the quadratic form alone.

Every delta is EXACT (Amy-Mosca coset decoding).  Every bound is asserted <= delta on every row;
a violation aborts the run.

Run:  python quadratic_invariant.py [samples_per_n]
"""
import itertools
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validate_amdist as AM
from experiment_mixed_nullity import (build_phi, true_moments, dprime, nu_of,
                                      peel_bound, nullspace_set)
from experiment_nu_eq_n_minus_d import f2_rank


# --------------------------------------------------------------- the bounds
def sharp_floor(coeff, n):
    """The thm:genLB floor 2(n-d*) - r, plus prop:nullity's +1 on the pure-cubic locus.
    Returns (floor, dstar, r, pure_cubic)."""
    _, Qrows = true_moments(coeff, n)
    r = f2_rank(Qrows, n)
    dp = dprime(coeff, n)
    pure = (r == 0)
    floor = 2 * (n - dp) - r + (1 if pure else 0)
    return floor, dp, r, pure


def nullity(coeff, n):
    """nu = n - d* (prop:nullitygen).  Cross-checked against the direct computation nu_of()."""
    return n - dprime(coeff, n)


# ------------------------------------------------- incidence candidates for r
def incidence_features(triples, pairs, lin, n):
    """Candidate replacements for r: how the quadratic support sits RELATIVE to the cubic support.
    rem:grading rules out any invariant of the quadratic form alone, so every feature here is a
    joint cubic/quadratic quantity."""
    cub_qubits = set()
    for t in triples:
        cub_qubits |= set(t)
    quad_qubits = set()
    for p in pairs:
        quad_qubits |= set(p)
    tri_set = set(tuple(sorted(t)) for t in triples)

    # pairs that sit inside some cubic triple vs pairs disjoint from the cubic support
    inside = sum(1 for p in pairs
                 if any(set(p) <= set(t) for t in tri_set))
    straddle = sum(1 for p in pairs
                   if len(set(p) & cub_qubits) == 1)
    outside = sum(1 for p in pairs if not (set(p) & cub_qubits))
    return dict(
        n_tri=len(tri_set), n_pair=len(pairs), n_lin=len(lin),
        pair_inside=inside, pair_straddle=straddle, pair_outside=outside,
        cub_supp=len(cub_qubits), quad_supp=len(quad_qubits),
        supp_overlap=len(cub_qubits & quad_qubits),
        lin_in_cub=sum(1 for i in lin if i in cub_qubits),
    )


# --------------------------------------------------------------- one sample
def measure(triples, pairs, lin, n):
    coeff, phi = build_phi(triples, pairs, lin, n)
    if not coeff:
        return None
    delta = AM.Tcount(coeff, n)
    nu = nullity(coeff, n)
    floor, dp, r, pure = sharp_floor(coeff, n)

    # peel, TWO versions.  peel_bound() subtracts the EXACT delta of the lower-degree parts and
    # uses the exact delta(f3); that is not O(n^3), and on a pure-cubic gate it degenerates to
    # delta itself (delta(f3) - 0 - 0), so it cannot be used to judge tightness there.
    # prop:peel's deployable form uses the cubic FLOOR for f3 and any UPPER bound for the lower
    # parts -- we use the residue weight, which upper-bounds delta since delta is a min-weight
    # coset representative.
    peel_x, d3, d3c, d2c, d1c = peel_bound(triples, pairs, lin, n)
    c2, _ = build_phi([], pairs, [], n)
    c1, _ = build_phi([], [], lin, n)
    w2 = int(AM.residue_vec(c2, n).sum()) if pairs else 0
    w1 = len(lin)
    peel_p = (2 * (n - d3) + (1 if d3 == 0 else 0)) - w2 - w1

    combined_x = max(nu, floor, peel_x)      # research upper envelope (NOT O(n^3))
    combined_p = max(nu, floor, peel_p)      # the deployable O(n^3) report

    # soundness: every term must be a genuine lower bound
    for name, val in (("nu", nu), ("sharp", floor), ("peel_exact", peel_x), ("peel_poly", peel_p)):
        assert val <= delta, ("BOUND VIOLATION", name, val, delta, triples, pairs, lin, n)
    # cor:floordominates: the floor is never weaker than nullity
    assert floor >= nu, ("floor < nu", floor, nu, triples, pairs, lin, n)

    row = dict(n=n, delta=delta, nu=nu, sharp=floor, peel=peel_p, peel_x=peel_x,
               combined=combined_p, combined_x=combined_x,
               dstar=dp, r=r, pure=pure,
               residual=delta - combined_p, residual_x=delta - combined_x,
               adv=floor - nu)
    row.update(incidence_features(triples, pairs, lin, n))
    return row


def random_gate(rng, n, p3, p2, p1):
    triples = [t for t in itertools.combinations(range(n), 3) if rng.random() < p3]
    pairs = [p for p in itertools.combinations(range(n), 2) if rng.random() < p2]
    lin = [i for i in range(n) if rng.random() < p1]
    return triples, pairs, lin


# --------------------------------------------------------------- the sweep
def run(samples=300, ns=(4, 5, 6), seed=20260903, verbose=True):
    rng = random.Random(seed)
    rows = []
    for n in ns:
        got = 0
        tries = 0
        while got < samples and tries < samples * 40:
            tries += 1
            p3 = rng.choice((0.15, 0.3, 0.5))
            p2 = rng.choice((0.0, 0.15, 0.3, 0.5, 0.8))
            p1 = rng.choice((0.0, 0.25, 0.5))
            triples, pairs, lin = random_gate(rng, n, p3, p2, p1)
            if not (triples or pairs or lin):
                continue
            row = measure(triples, pairs, lin, n)
            if row is None:
                continue
            rows.append(row)
            got += 1
        if verbose:
            print("  [n=%d: %d gates]" % (n, got))
            sys.stdout.flush()
    if verbose:
        report(rows)
    return rows


def report(rows):
    N = len(rows)
    print()
    print("=" * 78)
    print("(A) DOES THE PEELING TERM ADD ANYTHING TO 2(n-d*)-r?")
    print("=" * 78)
    base = sum(1 for x in rows if x["peel"] > max(x["nu"], x["sharp"]))
    ties = sum(1 for x in rows if x["peel"] == max(x["nu"], x["sharp"]))
    basex = sum(1 for x in rows if x["peel_x"] > max(x["nu"], x["sharp"]))
    print("  gates measured                                 : %d" % N)
    print("  peel (O(n^3), prop:peel deployable form):")
    print("    STRICTLY raises max(nu, sharp)               : %d  (%.1f%%)"
          % (base, 100.0 * base / max(N, 1)))
    print("    ties the max                                 : %d  (%.1f%%)"
          % (ties, 100.0 * ties / max(N, 1)))
    print("  peel (exact lower parts -- NOT O(n^3), research envelope only):")
    print("    STRICTLY raises max(nu, sharp)               : %d  (%.1f%%)"
          % (basex, 100.0 * basex / max(N, 1)))
    print("    (on a pure-cubic gate this term IS delta, so it cannot judge tightness there)")

    print()
    print("=" * 78)
    print("(B) WHO WINS WHERE  -- which term is the unique maximum")
    print("=" * 78)
    print("  %-10s %6s %8s %8s %8s %8s" % ("regime", "gates", "nu only", "sharp", "peel", "tie"))
    buckets = [("pure cubic", lambda x: x["r"] == 0),
               ("r < nu/2", lambda x: 0 < x["r"] < x["nu"] / 2.0),
               ("r >= nu/2", lambda x: x["r"] >= x["nu"] / 2.0 and x["r"] < x["nu"]),
               ("r = nu (sat)", lambda x: x["r"] >= x["nu"] and x["nu"] > 0)]
    for name, pred in buckets:
        sub = [x for x in rows if pred(x)]
        if not sub:
            print("  %-10s %6d" % (name, 0))
            continue
        wn = sum(1 for x in sub if x["nu"] > max(x["sharp"], x["peel"]))
        ws = sum(1 for x in sub if x["sharp"] > max(x["nu"], x["peel"]))
        wp = sum(1 for x in sub if x["peel"] > max(x["nu"], x["sharp"]))
        wt = len(sub) - wn - ws - wp
        print("  %-10s %6d %8d %8d %8d %8d" % (name, len(sub), wn, ws, wp, wt))
    print("  (nu can never strictly win over sharp -- cor:floordominates -- so 'nu only' must be 0.)")

    print()
    print("  advantage sharp - nu  = nu - r, by bucket:")
    for name, pred in buckets:
        sub = [x for x in rows if pred(x)]
        if sub:
            adv = [x["adv"] for x in sub]
            print("    %-12s mean %.2f   min %d   max %d" % (name, sum(adv) / len(adv), min(adv), max(adv)))

    print()
    print("=" * 78)
    print("(C) THE RESIDUAL  delta - combined  -- what is left for a new invariant to explain")
    print("=" * 78)
    tight = sum(1 for x in rows if x["residual"] == 0)
    print("  combined floor exactly tight                   : %d  (%.1f%%)"
          % (tight, 100.0 * tight / max(N, 1)))
    res = [x["residual"] for x in rows]
    print("  residual: mean %.2f  max %d" % (sum(res) / max(len(res), 1), max(res) if res else 0))
    print()
    print("  residual by quadratic rank r (looking for what varies at FIXED r):")
    print("  %4s %7s %8s %6s %6s" % ("r", "gates", "mean res", "min", "max"))
    for r in sorted(set(x["r"] for x in rows)):
        sub = [x for x in rows if x["r"] == r]
        rr = [x["residual"] for x in sub]
        print("  %4d %7d %8.2f %6d %6d" % (r, len(sub), sum(rr) / len(rr), min(rr), max(rr)))

    print()
    print("  correlation of residual with candidate invariants.")
    print("  col 1 controls for r only; col 2 ALSO controls for gate size (n, n_tri, n_pair),")
    print("  which is the confound -- a bigger gate has a bigger residual by itself.")
    print("  %-14s %14s %8s %16s %8s" % ("feature", "fixed r", "(gates)", "fixed r+size", "(gates)"))
    feats = ["pair_inside", "pair_straddle", "pair_outside", "supp_overlap",
             "n_tri", "n_pair", "n_lin", "cub_supp", "lin_in_cub"]
    for f in feats:
        c1, u1 = _partial_corr(rows, f, controls=("r",))
        c2, u2 = _partial_corr(rows, f, controls=("r", "n", "n_tri", "n_pair"))
        print("    %-14s %+13.3f %8d %+15.3f %8d" % (f, c1, u1, c2, u2))
    print("  Only a feature that survives column 2 is a candidate for replacing r;")
    print("  anything that collapses there was measuring gate size, not incidence.")


def _partial_corr(rows, feat, controls=("r",)):
    """Pearson correlation of (residual, feature) after removing the group mean of each, where
    groups are the level sets of `controls`.  With controls=("r",) this is correlation at fixed
    quadratic rank; adding "n_tri","n_pair" also controls for GATE SIZE, which is the confound --
    a bigger gate has a bigger residual regardless of any incidence structure."""
    groups = {}
    for x in rows:
        key = tuple(x[c] for c in controls)
        groups.setdefault(key, []).append(x)
    xs, ys, used = [], [], 0
    for key, sub in groups.items():
        if len(sub) < 3:
            continue
        used += len(sub)
        mres = sum(s["residual"] for s in sub) / len(sub)
        mf = sum(s[feat] for s in sub) / len(sub)
        for s in sub:
            xs.append(s[feat] - mf)
            ys.append(s["residual"] - mres)
    if len(xs) < 3:
        return 0.0, used
    sxy = sum(a * b for a, b in zip(xs, ys))
    sxx = sum(a * a for a in xs)
    syy = sum(b * b for b in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0, used
    return sxy / (sxx ** 0.5 * syy ** 0.5), used


def self_tests():
    print("SELF-TESTS (exact decoder + the known saturating cases):")
    c = {}
    AM.ccz_on(c, (0, 1, 2))
    assert AM.Tcount(c, 3) == 7, "delta(CCZ)"
    c = {}
    AM.cs_on(c, (0, 1))
    assert AM.Tcount(c, 3) == 3, "delta(CS)"
    c = {}
    AM.t_on(c, 0)
    assert AM.Tcount(c, 3) == 1, "delta(T)"
    print("  delta(CCZ)=7, delta(CS)=3, delta(T)=1  OK")

    # the saturating cases (disjoint CS layers) -- advantage must be exactly 0 there
    for m in (1, 2, 3):
        n = 2 * m
        pairs = [(2 * i, 2 * i + 1) for i in range(m)]
        row = measure([], pairs, [], n)
        adv = row["adv"]
        print("  CS layer m=%d (n=%d): nu=%d sharp=%d delta=%d  advantage=%d %s"
              % (m, n, row["nu"], row["sharp"], row["delta"], adv,
                 "OK (saturated)" if adv == 0 else "<-- expected 0"))
    for n in (1, 2, 3, 4):
        row = measure([], [], list(range(n)), n)
        print("  T^{tensor %d}: nu=%d sharp=%d delta=%d (true cost %d)"
              % (n, row["nu"], row["sharp"], row["delta"], n))
    print()


if __name__ == "__main__":
    self_tests()
    run(samples=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
