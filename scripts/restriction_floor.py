"""
The restriction floor:  delta(U) >= max over subsets S of delta(U|_S),  and where it beats isotropy.

Lemma.  For a diagonal level-three gate U on n qubits and S subset of [n], let U|_S be the gate
induced by fixing the inputs outside S to 0.  Then delta(U) >= delta(U|_S).

  Proof.  Take any {CNOT,T} representation of U, a multiset of parities y with coefficients c_y.
  Setting x_j = 0 for j not in S sends the parity y.x to (y & S).x, so the restricted multiset is a
  representation of U|_S.  Parities may coincide (coefficients add, which can only turn odd into
  even) or vanish entirely (y & S = 0, a global phase).  Neither can raise the odd-count, so the
  restricted representation has at most as many odd parities.  Minimising over representations gives
  delta(U|_S) <= delta(U).                                                                     []

Provenance.  This is lem:restrict.  It is also used inline in the proof of thm:nonadd:
"Restricting any representation to the three CCZ qubits (setting the
other inputs to 0) gives a representation of CCZ with no more odd parities, so delta >= 7."  It is
the transport mechanism that turns an EXACT
anchor (delta(CCZ)=7, delta_m=6m+1, delta=4m+3, ...) into a lower bound on any target containing it.

Use.  Combined floor
      delta(U) >= max{ 2(n - d*) - r ,  max over PROPER S of delta(U|_S) }.

Results.

  Two-shared layers.  On m CCZ blocks through a common pair (rem:twoshared) the floor with
  prop:nullity's +1 is 2(n-d)+1 = 7, already equal to delta, so restriction has nothing to add.

  Pure-cubic versus mixed.  On the PURE-CUBIC locus it never wins at all
  (0 of 400) -- restricting a cubic gives a smaller cubic whose own floor is no larger.  On
  MIXED-DEGREE gates it wins 300 of 399 (75.2%), mean gain +2.19.  The reason is the -r penalty:
  quadratic_invariant.py established that the isotropy floor's advantage over nullity is exactly
  nu - r, which is ZERO when r saturates.  Restricting to a subset carrying cubic content but little
  quadratic mass escapes that penalty entirely.  So the restriction floor repairs the QUADRATIC side
  -- not the collapse side.

Note.  The maximum must be over PROPER subsets.  Including S = [n] gives
U|_S = U and the "bound" becomes delta(U) itself, which silently reports the isotropy floor's slack
as a win.  Doing so reports spurious pure-cubic wins.

Run:  python restriction_floor.py
"""
import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validate_amdist as AM
from experiment_mixed_nullity import true_moments, dprime
from experiment_nu_eq_n_minus_d import f2_rank


# ------------------------------------------------------------------ restriction
def restrict_coeff(coeff, S_mask, n):
    """U|_S: fix inputs outside S to 0.  Parity y -> y & S, reindexed onto |S| qubits.
    Parities that vanish (y & S == 0) are a global phase and are dropped."""
    idx = [i for i in range(n) if (S_mask >> i) & 1]
    pos = {q: k for k, q in enumerate(idx)}
    out = {}
    for y, c in coeff.items():
        yy = 0
        for i in idx:
            if (y >> i) & 1:
                yy |= 1 << pos[i]
        if yy == 0:
            continue
        out[yy] = out.get(yy, 0) + c
    return {y: c for y, c in out.items() if c % 8}, len(idx)


def delta_exact(coeff, n):
    """Exact Amy-Mosca cost.  min_coset_weight is infeasible past n=6."""
    if not coeff:
        return 0
    if n > 6:
        return None
    return int(AM.Tcount(coeff, n))


def restriction_floor(coeff, n, verbose=False):
    """max over PROPER subsets S of delta(U|_S).  Returns (bound, best_S, best_size).

    The full set MUST be excluded: U|_[n] = U, so including it makes the 'bound' equal delta(U)
    by construction -- circular, and it silently reports the isotropy floor's slack as a win."""
    best, bestS, bestk = 0, 0, 0
    for S in range(1, (1 << n) - 1):
        sub, k = restrict_coeff(coeff, S, n)
        if not sub or k > 6:
            continue
        d = delta_exact(sub, k)
        if d is not None and d > best:
            best, bestS, bestk = d, S, k
    return best, bestS, bestk


def isotropy_floor(coeff, n):
    """The thm:genLB floor 2(n-d*) - r, with prop:nullity's +1 on the pure-cubic locus."""
    _, Qrows = true_moments(coeff, n)
    r = f2_rank(Qrows, n)
    dp = dprime(coeff, n)
    return 2 * (n - dp) - r + (1 if r == 0 else 0), dp, r


# ------------------------------------------------------------------ families
def ccz_gate(triples, n):
    c = {}
    for t in triples:
        AM.ccz_on(c, t)
    return c


def two_shared(m):
    """rem:twoshared: m blocks {0,1,2+i} through the shared pair {0,1}.  delta = 7 for every m."""
    return [(0, 1, 2 + i) for i in range(m)], m + 2


def parallel_ccz(m):
    return [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(m)], 3 * m


def fanout(m):
    """m CCZ sharing control 0, on n = 2m+1.  delta = 4m+3 (thm:fanout)."""
    return [(0, 2 * i + 1, 2 * i + 2) for i in range(m)], 2 * m + 1


def line(n):
    return [(i, i + 1, i + 2) for i in range(n - 2)], n


def complete3(n):
    return list(itertools.combinations(range(n), 3)), n


# ------------------------------------------------------------------ checks
def verify_lemma(verbose=True):
    """delta(U) >= delta(U|_S) on every subset of every census gate.  One failure kills the lemma."""
    fams = []
    for m in (1, 2):
        fams.append(("parallel CCZ m=%d" % m,) + parallel_ccz(m))
    for m in (1, 2):
        fams.append(("fan-out m=%d" % m,) + fanout(m))
    for m in (2, 3, 4):
        fams.append(("two-shared m=%d" % m,) + two_shared(m))
    for n in (5, 6):
        fams.append(("line f_%d" % n,) + line(n))
    for n in (5, 6):
        fams.append(("complete e_3 n=%d" % n,) + complete3(n))

    tot = bad = 0
    for (name, tri, n) in fams:
        if n > 6:
            continue
        c = ccz_gate(tri, n)
        dU = delta_exact(c, n)
        if dU is None:
            continue
        for S in range(1, 1 << n):
            sub, k = restrict_coeff(c, S, n)
            if not sub:
                continue
            dS = delta_exact(sub, k)
            if dS is None:
                continue
            tot += 1
            if dS > dU:
                bad += 1
                if verbose:
                    print("   LEMMA FAILURE: %s  S=%s  delta(U|S)=%d > delta(U)=%d"
                          % (name, bin(S), dS, dU))
    if verbose:
        print("RESTRICTION LEMMA CHECK")
        print("  (gate, subset) pairs tested            : %d" % tot)
        print("  delta(U|_S) > delta(U)  failures       : %d   <-- must be 0" % bad)
    return tot, bad


def compare(verbose=True):
    """Combined floor vs isotropy floor, per family."""
    rows = []
    fams = [("CCZ", ) + parallel_ccz(1),
            ("parallel CCZ m=2",) + parallel_ccz(2),
            ("fan-out m=2",) + fanout(2),
            ("line f_5",) + line(5),
            ("line f_6",) + line(6),
            ("complete e_3 n=5",) + complete3(5),
            ("complete e_3 n=6",) + complete3(6)]
    for m in (2, 3, 4):
        fams.append(("two-shared m=%d" % m,) + two_shared(m))

    if verbose:
        print()
        print("COMBINED FLOOR  max{ isotropy , restriction }   vs the isotropy floor alone")
        print("  %-20s %3s %7s %7s %7s %7s   %s"
              % ("family", "n", "iso", "restr", "comb", "delta", "verdict"))
    for (name, tri, n) in fams:
        if n > 6:
            continue
        c = ccz_gate(tri, n)
        iso, dp, r = isotropy_floor(c, n)
        restr, S, k = restriction_floor(c, n)
        dU = delta_exact(c, n)
        comb = max(iso, restr)
        ok = (dU is None) or (comb <= dU)
        verdict = "restriction WINS (+%d)" % (restr - iso) if restr > iso else "isotropy suffices"
        if not ok:
            verdict = "*** BOUND EXCEEDS delta ***"
        rows.append((name, n, iso, restr, comb, dU, verdict))
        if verbose:
            print("  %-20s %3d %7d %7d %7d %7s   %s"
                  % (name, n, iso, restr, comb, str(dU), verdict))
    return rows


def mixed_sweep(trials=400, seed=11, verbose=True):
    """Where the restriction floor actually helps: MIXED-degree gates.

    On the pure-cubic locus restriction cannot win -- restricting a cubic gives a smaller cubic
    whose own floor is no larger.  But the isotropy floor pays a -r penalty for quadratic mass and
    collapses to nullity when r saturates (quadratic_invariant.py: advantage = nu - r, zero at
    r = n - d*).  Restricting to a subset that carries cubic content but little quadratic mass
    escapes that penalty.  This sweep measures it."""
    import random
    from experiment_mixed_nullity import build_phi
    rng = random.Random(seed)
    tot = wins = exceed = 0
    gain = 0
    ex = None
    for _ in range(trials):
        n = rng.choice((4, 5, 6))
        tri = [t for t in itertools.combinations(range(n), 3) if rng.random() < 0.35]
        pai = [q for q in itertools.combinations(range(n), 2) if rng.random() < 0.5]
        lin = [i for i in range(n) if rng.random() < 0.3]
        c, _ = build_phi(tri, pai, lin, n)
        if not c:
            continue
        iso, dp, r = isotropy_floor(c, n)
        restr, S, k = restriction_floor(c, n)
        dU = delta_exact(c, n)
        tot += 1
        if dU is not None and max(iso, restr) > dU:
            exceed += 1
        if restr > iso:
            wins += 1
            gain += restr - iso
            if ex is None:
                ex = (n, iso, restr, dU, r)
    if verbose:
        print()
        print("MIXED-DEGREE SWEEP  (where the isotropy floor pays -r and collapses)")
        print("  gates tested                        : %d" % tot)
        print("  restriction STRICTLY beats isotropy : %d  (%.1f%%)"
              % (wins, 100.0 * wins / max(tot, 1)))
        print("  mean gain when it wins              : %.2f" % (gain / max(wins, 1)))
        print("  combined floor exceeds delta        : %d   <-- must be 0" % exceed)
        if ex:
            print("  first win: n=%d  iso=%d  restriction=%d  delta=%d  (r=%d)" % ex)
    return tot, wins, exceed


def pure_cubic_sweep(trials=400, seed=7, verbose=True):
    """Control: on the pure-cubic locus, restriction should NEVER beat isotropy."""
    import random
    rng = random.Random(seed)
    tot = wins = exceed = 0
    for _ in range(trials):
        n = rng.choice((4, 5, 6))
        allt = list(itertools.combinations(range(n), 3))
        tri = rng.sample(allt, rng.randint(1, min(len(allt), 6)))
        c = ccz_gate(tri, n)
        if not c:
            continue
        iso, dp, r = isotropy_floor(c, n)
        restr, S, k = restriction_floor(c, n)
        dU = delta_exact(c, n)
        tot += 1
        if dU is not None and max(iso, restr) > dU:
            exceed += 1
        if restr > iso:
            wins += 1
    if verbose:
        print()
        print("PURE-CUBIC CONTROL  (restriction should never win here)")
        print("  gates tested                        : %d" % tot)
        print("  restriction beats isotropy          : %d   (expected 0)" % wins)
        print("  combined floor exceeds delta        : %d   <-- must be 0" % exceed)
    return tot, wins, exceed


def two_shared_scaling(verbose=True):
    """The predicted win, at scale.  delta = 7 for every m is a THEOREM (rem:twoshared), so we can
    compare the two floors well past the decoding limit."""
    if verbose:
        print()
        print("rem:twoshared at scale: the floor 2(n-d)+1 = 7 (prop:nullity) is already tight,")
        print("so restriction cannot improve it.")
        print("  %3s %4s %7s %9s %7s   %s" % ("m", "n", "iso", "restrict", "true", "verdict"))
    for m in range(2, 9):
        tri, n = two_shared(m)
        c = ccz_gate(tri, n)
        iso, dp, r = isotropy_floor(c, n)
        # restriction to one block is always available and costs delta(CCZ) = 7
        restr = 7
        v = "restriction WINS (+%d)" % (restr - iso) if restr > iso else "tie"
        if verbose:
            print("  %3d %4d %7d %9d %7d   %s" % (m, n, iso, restr, 7, v))


if __name__ == "__main__":
    print("SELF-TESTS:")
    c = {}; AM.ccz_on(c, (0, 1, 2)); assert AM.Tcount(c, 3) == 7
    c = {}; AM.cs_on(c, (0, 1)); assert AM.Tcount(c, 3) == 3
    c = {}; AM.t_on(c, 0); assert AM.Tcount(c, 3) == 1
    print("  delta(CCZ)=7, delta(CS)=3, delta(T)=1  OK")
    print()
    verify_lemma()
    compare()
    two_shared_scaling()
    pure_cubic_sweep()
    mixed_sweep()
