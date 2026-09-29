"""
Is there an a-priori bound  delta(kappa) <= f(d)  ?   (Target 1)

Motivation.  Every rank-based closure relocates a circuit into an
already-closed regime (cor:maxrank, cor:ancilla), and no rank argument bounds delta(kappa)
itself (rem:nokappabound).
The exception is prop:epsH, which is unconditional in n, h, a, K AND rho_G:

        t  >=  delta(U) - delta(kappa).

No bound delta(kappa) <= f(h,d,w) is known -- only the empirical delta(kappa) <= 3 on
width-one families (rem:w1status).  A proved f would give a hypothesis-free all-n floor
t >= delta(U) - f(d).  prop:epsH already carries a COMPUTABLE bound U(kappa) (the residue weight),
so "computable" is not the gap: U(kappa) needs kappa first (2^d terms per x, over 2^n inputs).
An f(d) bound would be A PRIORI -- readable off h and the output-rank, so a circuit FAMILY can be
priced at every n without evaluating a Gauss sum.

Design.  A bound that holds only on REALIZABLE circuits inherits the difficulty of conj:w1 and
buys nothing.  It is worth having only if it follows from the Gauss-sum algebra alone -- in which
case it must ALSO hold on the abstract families, including the acid test
kappa = 2(x0x1 ^ x0x2), delta = 3, N_ker = 0.  So every candidate shape is tested on BOTH
populations, separately, and a shape that survives only on realizable data is rejected.

MEASURED FIRST (w1_search.verify_one_violation): the acid test sits at n=3, d=3, delta(kappa)=3.
So delta(kappa) = d exactly there -- tight, NOT a refutation.  The question is whether anything
abstract beats d.

Populations:
  (A1) abstract width-one chains        -- w1_search.build_kappa_family, swept over d
  (A2) abstract width-two (d=2) forms   -- validate_w2_abstract.stress (causally constrained:
                                           constant cross-coupling, i.e. strict causality)
  (R)  realizable circuits              -- mined from validate_w2.run()'s dforms, which is already
                                           keyed by (n,h,d,w) with values (kappa table, N_ker)

Run:  python kappa_bound.py [abstract_trials] [realizable_trials_per_cfg]
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validate_amdist as AM
from w1_search import build_kappa_family, delta_of_table


# ----------------------------------------------------------- (A1) abstract width-one
def abstract_w1(trials=200000, seed=11, dmax=6, verbose=True):
    """Sweep the abstract bandwidth-1 chain family, recording max delta(kappa) per d.
    Mirrors w1_search.verify_one_violation's generator but sweeps d and keeps EVERY valid
    family, not just the N_ker=0 violations."""
    rng = random.Random(seed)
    best = {}          # d -> (max delta, n, Nker)
    counts = {}
    for _ in range(trials):
        n = rng.randint(2, 3)
        d = rng.randint(2, dmax)
        afun = []
        for _ in range(d):
            terms = []
            for _ in range(rng.randint(0, 2)):
                terms.append((rng.choice([1, 3, 5, 7]), rng.randrange(1 << n), rng.randint(0, 1)))
            for _ in range(rng.randint(0, 1)):
                terms.append((rng.choice([2, 4, 6]), rng.randrange(1 << n), rng.randint(0, 1)))
            afun.append(terms)
        gfun = [(rng.randrange(1, 4), rng.randrange(1 << n), rng.randint(0, 1)) for _ in range(d - 1)]
        ok, kappa, Nker = build_kappa_family(n, d, afun, gfun, couple4_only=False)
        if not ok:
            continue
        dk = delta_of_table(kappa, n)
        if dk is None:
            continue
        counts[d] = counts.get(d, 0) + 1
        cur = best.get(d)
        if cur is None or dk > cur[0]:
            best[d] = (dk, n, Nker)
    if verbose:
        print("(A1) ABSTRACT width-one chains (bandwidth 1, constant-modulus diagonal)")
        print("  %3s %9s %12s %6s %8s   %s" % ("d", "valid", "max delta", "n", "N_ker", "verdict vs d"))
        for d in sorted(best):
            dk, n, nk = best[d]
            print("  %3d %9d %12d %6d %8d   %s"
                  % (d, counts[d], dk, n, nk, "OK (<= d)" if dk <= d else "**EXCEEDS d**"))
    return best, counts


# ----------------------------------------------------------- (A2) abstract width-two, d=2
def abstract_w2(trials=200000, seed=20260706, verbose=True):
    """The faithful d=2 family of validate_w2_abstract (causally constrained: the cross-coupling
    coefficient is a CONSTANT, i.e. strict causality).  Returns max delta(kappa) at d=2."""
    import validate_w2_abstract as W2A
    rng = random.Random(seed)
    out = {}
    for n in (2, 3, 4):
        valid, viol, maxd, worst = W2A.stress(n, trials, rng)
        out[n] = (valid, viol, maxd)
    if verbose:
        print()
        print("(A2) ABSTRACT width-two, d=2 (validate_w2_abstract.stress, strict causality)")
        print("  %3s %9s %8s %12s   %s" % ("n", "valid", "viol", "max delta", "verdict vs d=2"))
        for n in sorted(out):
            v, x, md = out[n]
            print("  %3d %9d %8d %12d   %s"
                  % (n, v, x, md, "OK (<= 2)" if md <= 2 else "**EXCEEDS d=2**"))
    return out


# ----------------------------------------------------------- (R) realizable
def realizable(trials_per_cfg=40000, verbose=True):
    """Mine validate_w2.run()'s dforms: keyed (n,h,d,w) -> set of (kappa tuple, N_ker)."""
    import validate_w2 as W2
    res = W2.run(trials_per_cfg=trials_per_cfg, want_steps=False)
    dforms = res[4]
    best = {}
    counts = {}
    for (n, h, d, w), forms in dforms.items():
        for (kap, nk) in forms:
            dk = delta_of_table(list(kap), n)
            if dk is None:
                continue
            counts[d] = counts.get(d, 0) + 1
            cur = best.get(d)
            if cur is None or dk > cur[0]:
                best[d] = (dk, n, h, w, nk)
    if verbose:
        print()
        print("(R) REALIZABLE circuits (validate_w2 sampler, distinct kappa per stratum)")
        print("  %3s %9s %12s %6s %5s %5s %8s   %s"
              % ("d", "forms", "max delta", "n", "h", "w", "N_ker", "verdict vs d"))
        for d in sorted(best):
            dk, n, h, w, nk = best[d]
            print("  %3d %9d %12d %6d %5d %5d %8d   %s"
                  % (d, counts[d], dk, n, h, w, nk, "OK (<= d)" if dk <= d else "**EXCEEDS d**"))
    return best, counts


def verdict(a1, a2, rl):
    print()
    print("=" * 76)
    print("CANDIDATE SHAPES  --  a shape must survive on BOTH populations to be pursued")
    print("=" * 76)
    shapes = [("delta <= d", lambda dk, d: dk <= d),
              ("delta <= d+1", lambda dk, d: dk <= d + 1),
              ("delta <= 2d", lambda dk, d: dk <= 2 * d),
              ("delta <= 2d-1", lambda dk, d: dk <= 2 * d - 1)]
    pops = [("abstract w1", [(v[0], d) for d, v in a1.items()]),
            ("abstract w2", [(v[2], 2) for v in a2.values()]),
            ("realizable", [(v[0], d) for d, v in rl.items()])]
    print("  %-16s %s" % ("shape", "  ".join("%-14s" % p[0] for p in pops)))
    for name, f in shapes:
        cells = []
        for _, data in pops:
            bad = [(dk, d) for (dk, d) in data if not f(dk, d)]
            cells.append("FAILS d=%d" % bad[0][1] if bad else "holds")
        print("  %-16s %s" % (name, "  ".join("%-14s" % c for c in cells)))
    print()
    print("  A shape holding on REALIZABLE but failing on ABSTRACT data is conj:w1 in disguise")
    print("  and is NOT pursued -- it would need realizability, which is the open problem.")


if __name__ == "__main__":
    at = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    rt = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    print("SELF-TESTS:")
    c = {}; AM.ccz_on(c, (0, 1, 2)); assert AM.Tcount(c, 3) == 7
    c = {}; AM.cs_on(c, (0, 1)); assert AM.Tcount(c, 3) == 3
    acid = [0, 0, 0, 2, 0, 2, 0, 0]
    assert delta_of_table(acid, 3) == 3, "acid-test decode"
    print("  delta(CCZ)=7, delta(CS)=3, delta(acid test)=3  OK")
    print("  acid test measured at n=3, d=3 (w1_search.verify_one_violation): delta = d, TIGHT\n")
    a1, _ = abstract_w1(trials=at)
    a2 = abstract_w2(trials=at)
    rl, _ = realizable(trials_per_cfg=rt)
    verdict(a1, a2, rl)
