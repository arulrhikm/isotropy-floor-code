"""
Superseded by sharp_general_bound.py.  Kept because it checks thm:cubicLB on the named cases.

Two limits of what this script tests:

 (1) It uses the CUBIC radical d (cubic_tensor_radical_dim skips coincident indices at :57-58),
     not the full third-power radical d*.  With the cubic d the inequality is NOT VALID off the
     pure-cubic locus -- see the footnote to thm:genLB.  This script
     passes only because its 17 hand-picked CASES avoid the failure region.  The corrected d* is
     computed by experiment_mixed_nullity.dprime.
 (2) Even with d*, the -2r penalty is not the right price.  thm:genLB reads
     delta >= 2(n - d*) - r: the penalty is paid per symplectic PAIR, because over F_2
     the diagonal v.v is linear, so the Gram form is alternating on a hyperplane and leaves a
     Lagrangian as well as a radical.  See sharp_general_bound.py, which is the live test.

Validate the GENERAL isotropic lower bound (drops the pure-cubic hypothesis):

    delta(U)  >=  2 ( n - d - r ),        [superseded; see above]

where
    d = dim of the radical of the cubic moment tensor T_{abc} = M_{{a,b,c}},
    r = F_2-rank of the order-<=2 moment matrix G  (G_{ab}=M_{{a,b}}, G_{aa}=M_{{a}}).

Specialization: pure-cubic gates have G=0 (r=0), recovering delta >= 2(n-d)
[and the +1 when d=0], i.e. thm:cubicLB.

Ground truth delta is the exact Amy-Mosca coset distance to RM(n-4,n)*, reusing
validate_amdist.py.  We build each gate from CCZ/CS/T monomial phases, take its
residue parity set, compute (d, r) from the order-<=3 moments, and check the bound
against the exact delta.
"""
import itertools
import numpy as np
from validate_amdist import (
    residue_vec, rm_generators, min_coset_weight,
    ccz_on, cs_on, t_on, complete_cubic,
)

# ---------- residue parity set and moments ----------
def residue_set(coeff):
    """Set of parity bitmasks with odd coefficient."""
    return {y for y, c in coeff.items() if c % 2}

def moment(rset, A):
    """M_A = #{y in rset : A subseteq y} mod 2, A a bitmask."""
    return sum(1 for y in rset if (y & A) == A) & 1

# ---------- GF(2) rank ----------
def rank_gf2(int_rows):
    basis = []
    for r in int_rows:
        cur = int(r)
        for b in basis:
            if (cur ^ b) < cur:
                cur ^= b
        if cur:
            basis.append(cur)
            basis.sort(reverse=True)
    return len(basis)

# ---------- structure: cubic radical dim d, order-<=2 rank r ----------
def cubic_tensor_radical_dim(rset, n):
    # T_{abc} = M_{abc}; radical: w with sum_a w_a T_{a q r}=0 for all pairs q<r.
    # Row for pair (q,r): integer over a in [n], bit a = T_{a,q,r}.
    rows = []
    for q in range(n):
        for r_ in range(q + 1, n):
            row = 0
            for a in range(n):
                if a == q or a == r_:
                    continue
                A = (1 << a) | (1 << q) | (1 << r_)
                if moment(rset, A):
                    row |= (1 << a)
            rows.append(row)
    rank = rank_gf2(rows)
    return n - rank  # nullity = dim radical

def order2_matrix_rank(rset, n):
    # G_{ab}=M_{ab} (a!=b), G_{aa}=M_{a}; F_2 rank, rows as ints.
    rows = []
    for a in range(n):
        row = 0
        for b in range(n):
            if a == b:
                A = (1 << a)
            else:
                A = (1 << a) | (1 << b)
            if moment(rset, A):
                row |= (1 << b)
        rows.append(row)
    return rank_gf2(rows)

def exact_delta(coeff, n):
    r = residue_vec(coeff, n)
    G = rm_generators(n)
    return min_coset_weight(r, G)

# ---------- battery ----------
def gate(builders, n):
    c = {}
    for b in builders:
        b(c)
    return c

def CCZ(*t): return lambda c: ccz_on(c, t)
def CS(*p):  return lambda c: cs_on(c, p)
def T(i):    return lambda c: t_on(c, i)
def E3(n):   return lambda c: complete_cubic(c, n)

CASES = [
    ("CCZ(012)",                 [CCZ(0,1,2)],                         3),
    ("CCZ(012) [n4]",            [CCZ(0,1,2)],                         4),
    ("CCZ(012) [n5]",            [CCZ(0,1,2)],                         5),
    ("CCZ(012) [n6]",            [CCZ(0,1,2)],                         6),
    ("CCZ(012)+CS(01)",          [CCZ(0,1,2), CS(0,1)],                3),
    ("CCZ(012)+CS(01) [n4]",     [CCZ(0,1,2), CS(0,1)],                4),
    ("CCZ(012)+CS(01)+CS(12)",   [CCZ(0,1,2), CS(0,1), CS(1,2)],       3),
    ("CCZ(012)+T(0)",            [CCZ(0,1,2), T(0)],                   4),
    ("CCZ(012)+T(3) [n4]",       [CCZ(0,1,2), T(3)],                   4),
    ("CCZ(012)+CS(34) [n5]",     [CCZ(0,1,2), CS(3,4)],                5),
    ("CCZ(012)+CCZ(345) [n6]",   [CCZ(0,1,2), CCZ(3,4,5)],             6),
    ("CCZ(012)+CCZ(345)+CS(03)", [CCZ(0,1,2), CCZ(3,4,5), CS(0,3)],    6),
    ("CS(01)",                   [CS(0,1)],                            3),
    ("CS(01)+CS(12)",            [CS(0,1), CS(1,2)],                   3),
    ("E3 complete cubic [n5]",   [E3(5)],                              5),
    ("E3 complete cubic [n6]",   [E3(6)],                              6),
    ("CCZ(012)+CCZ(234) overlap",[CCZ(0,1,2), CCZ(2,3,4)],             5),
]

def main():
    print(f"{'gate':30s} {'n':>2} {'d':>2} {'r':>2} {'2(n-d-r)':>9} {'delta':>5}  status")
    print("-" * 70)
    all_ok = True
    for name, builders, n in CASES:
        c = gate(builders, n)
        rset = residue_set(c)
        d = cubic_tensor_radical_dim(rset, n)
        r = order2_matrix_rank(rset, n)
        lb = 2 * (n - d - r)
        dlt = exact_delta(c, n)
        ok = dlt >= lb
        # pure-cubic refinement check
        pure = (r == 0)
        ref = ""
        if pure and d == 0:
            ref = f"  [pure-cubic d=0: 2n+1={2*n+1}, delta>={2*n+1}? {dlt >= 2*n+1}]"
        elif pure:
            ref = f"  [pure-cubic: 2(n-d)={2*(n-d)}]"
        status = "PASS" if ok else "*** FAIL ***"
        all_ok &= ok
        print(f"{name:30s} {n:2d} {d:2d} {r:2d} {lb:9d} {dlt:5d}  {status}{ref}")
    print("-" * 70)
    print("ALL PASS" if all_ok else "SOME FAILED")

if __name__ == "__main__":
    main()
