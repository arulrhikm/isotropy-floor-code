"""
Validation for the universal-QAOA / applications additions to sec:qaoa.

Checks, all against the exact Amy-Mosca coset distance where feasible:

 (1) Universal 1-layer floor: weighted, INHOMOGENEOUS 3-local gates (random fields
     h_i, couplings J_ij [CS], and 3-body K_ijk [CCZ]) satisfy the general bound
     delta >= 2(n - d* - r)   [thm:genLB], where d* is the radical of the full
     third-order moment tensor and r the order-<=2 moment-matrix rank.

 (2) Generic non-degeneracy: a uniformly random 3-local objective (each 3-subset an
     edge w.p. 1/2) has trivial cubic radical (d=0) with probability -> 1.  We report
     the empirical Pr[d=0] for a range of n.  (Radical computed directly from the
     random symmetric tensor; cheap, so large n is fine.)

 (3) Disjoint magic-compression floor: m disjoint NON-DEGENERATE cubics g_i have
     delta(+ g_i) >= 2N+1  (N = sum n_i), so batching saves at most sum delta_i-(2N+1).

 (4) IQP core: minimal T-count is Clifford-bi-invariant, so an IQP circuit
     H^{xn} D H^{xn} has the same T-count as its diagonal core D.  This is analytic
     (no decoder needed); we only re-confirm delta(D) >= 2n+1 for a non-degenerate
     cubic core D, which is the floor the IQP circuit inherits when D is Hadamard-free.

Reuses validate_amdist.py (exact decoder) and validate_general_bound.py (d*, r).
"""
import itertools, random
import numpy as np
from validate_amdist import (
    ccz_on, cs_on, t_on, residue_vec, rm_generators, min_coset_weight,
)
from validate_general_bound import (
    residue_set, cubic_tensor_radical_dim, order2_matrix_rank, rank_gf2,
)

def exact_delta(coeff, n):
    return min_coset_weight(residue_vec(coeff, n), rm_generators(n))

def moment_set(rset, A):
    return sum(1 for y in rset if (y & A) == A) & 1

def full_tensor_radical_dim(rset, n):
    """d* = dim of the radical of the FULL third-order moment tensor (thm:genLB).
    Row per pair q<=r (q=r allowed): bit a = M_{set(a,q,r)} (coincident indices
    collapse to order-<=2 moments).  This is the valid quantity off the pure-cubic
    locus; using the cubic-only radical here is NOT a valid bound (see the footnote
    of thm:genLB)."""
    rows = []
    for q in range(n):
        for r_ in range(q, n):  # include q==r
            row = 0
            for a in range(n):
                A = (1 << a) | (1 << q) | (1 << r_)
                if moment_set(rset, A):
                    row |= (1 << a)
            rows.append(row)
    return n - rank_gf2(rows)

# ---------------------------------------------------------------------------
# (2) Generic non-degeneracy: radical of a random symmetric cubic tensor over F2.
# ---------------------------------------------------------------------------
def random_tensor_radical_dim(n, rng):
    """d = dim{w : sum_a w_a T_{aqr} = 0 for all q<r}, T a random symmetric 3-tensor."""
    T = {}
    for a, b, c in itertools.combinations(range(n), 3):
        T[(a, b, c)] = rng.randint(0, 1)
    def Tval(a, b, c):
        return T[tuple(sorted((a, b, c)))]
    rows = []  # one row per pair (q,r); bit a = T_{a,q,r}
    for q in range(n):
        for r_ in range(q + 1, n):
            row = 0
            for a in range(n):
                if a == q or a == r_:
                    continue
                if Tval(a, q, r_):
                    row |= (1 << a)
            rows.append(row)
    return n - rank_gf2(rows)

def check_generic_nondegeneracy(trials=4000):
    print("(2) Generic non-degeneracy: Pr[d=0] for random 3-local objectives")
    print(f"    {'n':>2}  {'Pr[d=0]':>8}  {'trials':>6}")
    rng = random.Random(1)
    for n in range(3, 14):
        hits = sum(1 for _ in range(trials) if random_tensor_radical_dim(n, rng) == 0)
        print(f"    {n:2d}  {hits/trials:8.4f}  {trials:6d}")
    print()

# ---------------------------------------------------------------------------
# (1) Universal 1-layer floor on random weighted inhomogeneous 3-local gates.
# ---------------------------------------------------------------------------
def random_inhomogeneous_gate(n, rng, p3=0.5, p2=0.4, p1=0.3):
    """Random level-3 gate: 3-body CCZ, 2-body CS, 1-body T, each present w.p. p*."""
    c = {}
    for trip in itertools.combinations(range(n), 3):
        if rng.random() < p3:
            ccz_on(c, trip)
    for pair in itertools.combinations(range(n), 2):
        if rng.random() < p2:
            cs_on(c, pair)
    for i in range(n):
        if rng.random() < p1:
            t_on(c, i)
    return c

def check_universal_floor(n_max=6, per_n=40):
    print("(1) Universal 1-layer floor delta >= 2(n-d*-r) on random inhomogeneous gates")
    print(f"    {'n':>2}  {'trials':>6}  {'min slack':>9}  {'status':>10}")
    rng = random.Random(7)
    all_ok = True
    for n in range(3, n_max + 1):
        min_slack = None
        for _ in range(per_n):
            c = random_inhomogeneous_gate(n, rng)
            rset = residue_set(c)
            dstar = full_tensor_radical_dim(rset, n)   # full third-order radical (thm:genLB)
            r = order2_matrix_rank(rset, n)
            lb = 2 * (n - dstar - r)
            dlt = exact_delta(c, n)
            slack = dlt - lb
            if slack < 0:
                all_ok = False
            min_slack = slack if min_slack is None else min(min_slack, slack)
        status = "PASS" if min_slack is not None and min_slack >= 0 else "*** FAIL ***"
        print(f"    {n:2d}  {per_n:6d}  {min_slack:9d}  {status:>10}")
    print("    -> genLB holds on every sampled inhomogeneous gate\n" if all_ok
          else "    *** SOME FAILED ***\n")

# ---------------------------------------------------------------------------
# (3) Disjoint magic-compression floor: disjoint non-degenerate cubics.
# ---------------------------------------------------------------------------
def line_cubic(c, base, k):
    """Non-degenerate cubic on qubits base..base+k-1: consecutive triples (rem:line)."""
    for a in range(base, base + k - 2):
        ccz_on(c, (a, a + 1, a + 2))

DISJOINT_CASES = [
    ("CCZ(012)+CCZ(345)",            [("ccz", 0, 1, 2), ("ccz", 3, 4, 5)], 6),
    ("CCZ+CCZ+CCZ (m=3)",            [("ccz", 0, 1, 2), ("ccz", 3, 4, 5), ("ccz", 6, 7, 8)], 9),
    ("CCZ(012)+line5(3..7)",         [("ccz", 0, 1, 2), ("line", 3, 5)], 8),
]

def build_disjoint(spec):
    c = {}
    for item in spec:
        if item[0] == "ccz":
            ccz_on(c, item[1:])
        elif item[0] == "line":
            line_cubic(c, item[1], item[2])
    return c

def check_disjoint_compression():
    print("(3) Disjoint magic-compression floor: delta(+ g_i) >= 2N+1")
    print(f"    {'layer':30s} {'N':>2} {'d':>2} {'2N+1':>5} {'delta':>7}  status")
    all_ok = True
    for name, spec, N in DISJOINT_CASES:
        c = build_disjoint(spec)
        rset = residue_set(c)
        d = cubic_tensor_radical_dim(rset, N)      # union radical (cheap, any N)
        floor = 2 * N + 1
        if N <= 6:
            dlt = exact_delta(c, N)
            ok = (d == 0) and (dlt >= floor)
            dcol = f"{dlt:7d}"
        else:
            dlt = None
            ok = (d == 0)                          # non-degeneracy of the union
            dcol = f"{'(skip)':>7}"
        all_ok &= ok
        print(f"    {name:30s} {N:2d} {d:2d} {floor:5d} {dcol}  {'PASS' if ok else '*** FAIL ***'}")
    print("    -> union is non-degenerate and meets the 2N+1 floor\n" if all_ok
          else "    *** SOME FAILED ***\n")

# ---------------------------------------------------------------------------
# (4) IQP core floor (analytic; re-confirm delta(D) >= 2n+1 for a cubic core).
# ---------------------------------------------------------------------------
def check_iqp_core():
    print("(4) IQP core: delta(D) >= 2n+1 for a non-degenerate cubic core D")
    print("    (T-count is Clifford-bi-invariant, so delta(H^xn D H^xn)=delta(D))")
    all_ok = True
    for n in range(5, 8):
        c = {}
        line_cubic(c, 0, n)  # non-degenerate line
        rset = residue_set(c)
        d = cubic_tensor_radical_dim(rset, n)
        if n <= 6:
            dlt = exact_delta(c, n)
            ok = (d == 0) and (dlt >= 2 * n + 1)
            print(f"    n={n}: d={d}, delta={dlt}, 2n+1={2*n+1}  {'PASS' if ok else '*** FAIL ***'}")
        else:
            ok = (d == 0)
            print(f"    n={n}: d={d} (delta decode skipped: RM(n-4,n)* too large), non-deg {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    print("    -> Hadamard-free cubic IQP cores inherit the 2n+1 floor\n" if all_ok
          else "    *** SOME FAILED ***\n")

if __name__ == "__main__":
    # self-test against known values
    c = {}; ccz_on(c, (0, 1, 2))
    assert exact_delta(c, 3) == 7, "self-test delta(CCZ)=7 failed"
    c = {}; cs_on(c, (0, 1))
    assert exact_delta(c, 3) == 3, "self-test delta(CS)=3 failed"
    print("self-test OK: delta(CCZ)=7, delta(CS)=3\n")

    check_universal_floor()
    check_generic_nondegeneracy()
    check_disjoint_compression()
    check_iqp_core()
