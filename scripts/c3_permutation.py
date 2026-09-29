"""
C3 permutation gates and the cubic moment floor (Appendix E, sec:c3perm).

He-Robitaille-Tan (arXiv:2510.04993) prove every C3 permutation gate is, up to
Clifford, a *staircase* product of Toffolis (Thm 3.2), in bijection with
"descending multiplications" (Thm 4.2): commutative associative products on F2^n
with e_i e_i = 0 and e_i e_j in span{e_k : k>j}.  TOF_{i,j,k} is in the staircase
exactly when the structure constant c_{ij}^k = 1; c itself is symmetric in its
first two slots only, but its SUPPORT, read as unordered triples, is exactly the
block set, and the cubic moment tensor is that set's indicator.  See
check_structure_constants().

This script validates the bridge:
 (D1) mismatch-free staircase (= pairwise-commuting Toffolis = semi-Clifford,
      HRT Lem 5.12) is Clifford-conjugate to the diagonal CCZ layer on the same
      triples, so delta_T(pi) = delta(D_pi); we DECODE small mismatch-free cases
      and confirm equality with the overlap floor.
 (D2) every C3 permutation's diagonal companion D_pi has floor 2(n-d); we compute
      it for the U_k family and the Gottesman-Mochon gate U_3 (n=7: floor + naive
      upper bound only, exact decode infeasible at code dim 64).
 (mismatch) a staircase permutation is semi-Clifford iff no target qubit is later
      used as a control; we detect this directly on the Toffoli list.

Reuses overlap_layers.radical_dim and validate_amdist.{ccz_on,residue_vec,
rm_generators,min_coset_weight}.
"""
import itertools, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_layers import radical_dim
from validate_amdist import ccz_on, residue_vec, rm_generators, min_coset_weight


# ---------- staircase Toffoli lists ----------
# A Toffoli is a triple (i, j, k) with i<j controls, k target (0-indexed here).

def triples_of(toffolis):
    """The set of index-triples (unordered) carried by a Toffoli list."""
    return [tuple(sorted(t)) for t in toffolis]

def is_mismatch_free(toffolis):
    """HRT Lem 5.12 / Prop 5.13: a staircase Toffoli product is semi-Clifford iff
    it is mismatch-free -- no qubit used as a TARGET is later used as a CONTROL
    (in the applied order).  Toffolis given (i,j,k) = controls i,j, target k, in
    application order."""
    targeted = set()
    for (i, j, k) in toffolis:
        if i in targeted or j in targeted:
            return False  # a control was previously a target -> mismatch
        targeted.add(k)
    return True

def diag_companion_floor(toffolis, n):
    """Radical d and isotropy floor 2(n-d) (+1 if d=0) of the diagonal CCZ layer
    on the triples of the Toffoli list."""
    tris = triples_of(toffolis)
    d = radical_dim(tris, n)
    floor = 2 * (n - d) + (1 if d == 0 else 0)
    return d, floor

def diag_companion_decode(toffolis, n):
    """Exact delta of the diagonal CCZ companion (feasible only for small n)."""
    c = {}
    for t in triples_of(toffolis):
        ccz_on(c, t)
    r = residue_vec(c, n)
    naive = int(r.sum())
    if n > 6:
        return None, naive  # meet-in-the-middle decoder infeasible past n=6
    return min_coset_weight(r, rm_generators(n)), naive

def nullity(toffolis, n):
    """Stabilizer nullity of the diagonal CCZ companion = n - dim A; for a
    pure-cubic gate A = radical, so nu = n - d (prop:nullity)."""
    d = radical_dim(triples_of(toffolis), n)
    return n - d


# ---------- the U_k family (HRT Def 5.1 / Prop 5.2) ----------
# Qubits <-> nonempty subsets S of [k]; n = 2^k - 1.  For each pair of disjoint
# nonempty S, T with S<T apply TOF_{S,T,S|T}.  Index a subset by its bitmask.

def uk_toffolis(k):
    n = (1 << k) - 1
    idx = {}  # bitmask (1..2^k-1) -> qubit index 0..n-1, ordered by mask value
    for q, mask in enumerate(range(1, 1 << k)):
        idx[mask] = q
    toffolis = []
    masks = list(range(1, 1 << k))
    for S in masks:
        for T in masks:
            if S < T and (S & T) == 0:  # disjoint, S<T
                U = S | T
                toffolis.append((idx[S], idx[T], idx[U]))
    # staircase order: nondecreasing target
    toffolis.sort(key=lambda t: t[2])
    return toffolis, n

def uk_max_degree(k):
    """Max monomial degree in the polynomial rep of U_k is k (HRT Lem 5.7:
    pi_{[k]} contains a_1 a_2 ... a_k)."""
    return k


# ---------- Gottesman-Mochon U_3 (HRT sec 5.3) ----------
# U_3 = TOF_{1,6,7}TOF_{2,5,7}TOF_{3,4,7}TOF_{2,4,6}TOF_{1,4,5}TOF_{1,2,3} (1-indexed).

U3_TOFFOLIS_1IDX = [(1, 6, 7), (2, 5, 7), (3, 4, 7), (2, 4, 6), (1, 4, 5), (1, 2, 3)]

def u3_toffolis():
    # 0-indexed, in application (right-to-left of the product = staircase) order
    seq_1idx = list(reversed(U3_TOFFOLIS_1IDX))  # apply TOF_{1,2,3} first
    return [tuple(x - 1 for x in t) for t in seq_1idx], 7


# ---------- Cuccaro ripple-carry adder MAJ/carry Toffoli network ----------
# The MAJ block of a ripple-carry adder computes carries c_{i+1} = MAJ(a_i,b_i,c_i)
# whose Toffoli content is TOF(a_i, b_i, c_i-relative) chained through the carry.
# We model the carry-sharing Toffoli chain on 2m+1 wires: block i shares the
# running carry wire with block i+1 (an overlap / fan-through pattern).

def adder_carry_toffolis(m):
    """m carry Toffolis chained through a shared carry register (schematic):
    TOF(a_i, c_i, c_{i+1}) with c_i the running carry -> consecutive blocks share
    the carry wire (a linear-overlap chain)."""
    # wires: a_0..a_{m-1} (0..m-1), carries c_0..c_m (m..2m)
    tof = []
    for i in range(m):
        a_i = i
        c_i = m + i
        c_next = m + i + 1
        tof.append((a_i, c_i, c_next))
    n = 2 * m + 1
    return tof, n


# ---------- checks ----------

def check_selftests():
    print("SELF-TESTS")
    c = {}
    ccz_on(c, (0, 1, 2))
    assert min_coset_weight(residue_vec(c, 3), rm_generators(3)) == 7
    print("  delta(CCZ) = 7  OK")
    # single Toffoli is mismatch-free
    assert is_mismatch_free([(0, 1, 2)])
    print("  single Toffoli mismatch-free  OK")

def check_u3():
    print("\n(D2) Gottesman-Mochon U_3 diagonal companion")
    tof, n = u3_toffolis()
    d, floor = diag_companion_floor(tof, n)
    nu = nullity(tof, n)
    dec, naive = diag_companion_decode(tof, n)
    mf = is_mismatch_free(tof)
    print(f"  triples: {triples_of(tof)}")
    print(f"  n={n}  radical d={d}  floor={floor}  nullity nu={nu}  (floor/nu = {floor/nu:.2f}x)")
    print(f"  mismatch-free (semi-Clifford)? {mf}  [expect False: U_3 is non-semi-Clifford]")
    print(f"  exact decode: {dec}  naive residue weight (upper bound): {naive}")
    assert d == 0 and floor == 15 and nu == 7 and mf is False

def check_uk(kmax=5):
    print("\n(D2) U_k family: joint (width, T-floor) region")
    print("  k   n=2^k-1  #Toffolis  degree  HRT width floor  d  T-floor  nu")
    for k in range(3, kmax + 1):
        tof, n = uk_toffolis(k)
        d, floor = diag_companion_floor(tof, n)
        nu = nullity(tof, n)
        deg = uk_max_degree(k)
        width_floor = (1 << k) - 1  # HRT Thm 5.9
        print(f"  {k}   {n:6d}   {len(tof):8d}   {deg:5d}   {width_floor:14d}   {d}   {floor:13d}   {nu}")
        assert n == width_floor

def check_uk_general(kmax_steps=9, kmax_radical=6):
    """prop:jointregion -- the U_k shadow is non-degenerate for EVERY k.

    This used to be a spot-check for k<=5.  The proof is two steps, and both are
    checked here directly (which is what makes the general claim safe):

      (1) Q,R disjoint  =>  the ONLY P completing {P,Q,R} to a block is Q|B,
          so the radical relation reads w_{Q|R}=0 and every qubit of size >=2 dies;
      (2) Q={i}, R={i,j}, j!=i  =>  the ONLY completion is P={j}, so every
          singleton dies.

    Steps (1)+(2) touch every qubit, hence R={0}, d=0, floor 2n+1 = 2^(k+1)-1.
    """
    print("\n(D2') U_k non-degeneracy, general k -- verifying the two proof steps")
    print("  k     n  blocks  step1 unique  step2 unique  covers all   d  floor  2^(k+1)-1")
    for k in range(2, kmax_steps + 1):
        tof, n = uk_toffolis(k)
        tris = {frozenset(t) for t in triples_of(tof)}
        masks = list(range(1, 1 << k))
        idx = {m: q for q, m in enumerate(masks)}

        def completions(Q, R):
            return [P for P in range(n)
                    if P != Q and P != R and frozenset((P, Q, R)) in tris]

        step1 = all(completions(idx[A], idx[B]) == [idx[A | B]]
                    for A in masks for B in masks if A < B and not (A & B))
        step2 = all(completions(idx[1 << i], idx[(1 << i) | (1 << j)]) == [idx[1 << j]]
                    for i in range(k) for j in range(k) if i != j)
        covered = {idx[A | B] for A in masks for B in masks if A < B and not (A & B)}
        covered |= {idx[1 << j] for j in range(k)}
        full = (len(covered) == n)

        if k <= kmax_radical:
            d, floor = diag_companion_floor(tof, n)
            dtxt, ftxt = str(d), "%5d" % floor
            assert d == 0 and floor == 2 * n + 1 == (1 << (k + 1)) - 1
        else:
            d, dtxt, ftxt = None, "-", "    -"

        print(f"  {k}  {n:4d}  {len(tof):6d}     {str(step1):5s}         {str(step2):5s}"
              f"       {str(full):5s}      {dtxt}  {ftxt}  {(1 << (k + 1)) - 1:9d}")
        assert step1 and step2 and full
        assert n == (1 << k) - 1

    # the k=3 block set must be exactly the six triples quoted in prop:gm
    tof3, n3 = uk_toffolis(3)
    assert n3 == 7 and len({frozenset(t) for t in triples_of(tof3)}) == 6, \
        "prop:gm quotes six triples on seven qubits"
    print("  k=3 gives 6 triples on 7 qubits, matching prop:gm")


def check_structure_constants():
    """The bridge slide's claim, stated correctly.

    A descending multiplication is fixed by c_{ij}^k via  e_i e_j = sum_k c_{ij}^k e_k.
    It is symmetric in its FIRST TWO slots only (commutativity) and NOT in all three:
    for the subset-union product, c_{A,B}^{A|B} = 1 while c_{A,A|B}^{B} = 0.

    What carries over to the moment tensor is the SUPPORT: the set of unordered triples {i,j,k}
    with c_{ij}^k = 1 is exactly the block set, and the cubic moment tensor is that
    set's indicator (symmetric because a block is a set, not because c is).
    """
    print("\n(bridge) structure constants vs the cubic moment tensor")
    k = 3
    masks = list(range(1, 1 << k))

    def c(A, B, C):                      # e_A e_B = e_{A|B} if disjoint, else 0
        return 1 if (A & B) == 0 and C == (A | B) else 0

    # (a) symmetric in the first two slots
    assert all(c(A, B, C) == c(B, A, C) for A in masks for B in masks for C in masks)
    # (b) NOT symmetric in all three -- the exact counterexample the slide quotes
    A, B = 0b001, 0b010                  # A={1}, B={2}, A|B={1,2}
    assert c(A, B, A | B) == 1 and c(A, A | B, B) == 0
    print("   c_{A,B}^{A|B} = %d  but  c_{A,A|B}^{B} = %d   (A={1}, B={2})"
          % (c(A, B, A | B), c(A, A | B, B)))
    print("   so c is symmetric in its first two slots only, NOT a symmetric tensor")

    # (c) the support, as unordered triples, IS the block set
    idx = {m: q for q, m in enumerate(masks)}
    support = {frozenset((idx[A], idx[B], idx[C]))
               for A in masks for B in masks for C in masks
               if c(A, B, C) and len({A, B, C}) == 3}
    tof, n = uk_toffolis(k)
    blocks = {frozenset(t) for t in triples_of(tof)}
    assert support == blocks, "support of c must equal the block set"
    print("   support of c (as unordered triples) == block set: %s  (%d triples on %d qubits)"
          % (support == blocks, len(blocks), n))


def check_padding_necessity():
    """rem:padding -- the non-degeneracy hypothesis of prop:jointregion cannot be
    dropped.  Pad U_k with one IDLE qubit: it lies in no block, so its coordinate
    is a radical direction and d >= 1.  The padded gate is still a C_3 permutation
    still carrying a degree-k monomial, so "delta >= 2n+1 for every degree-k
    carrier" is FALSE.  What survives is 2(n-d)+1 = 2*nu+1, because padding raises
    n and d together -- i.e. nu, not n, is the padding-invariant quantity.
    """
    print("\n(D2'') rem:padding -- why prop:jointregion needs its hypothesis")
    print("   NB: diag_companion_floor() returns thm:cubicLB's form (the +1 only at")
    print("       d=0); the padding-invariant statement is prop:nullity's 2*nu+1,")
    print("       which carries the +1 for every d.  We compare against 2*nu+1.")
    print("    k |  n  d  2n+1  2nu+1 | padded: n'  d'  2n'+1  2nu'+1 | 2n'+1 valid?")
    for k in (3, 4):
        tof, n = uk_toffolis(k)
        d, _ = diag_companion_floor(tof, n)
        nu = nullity(tof, n)
        np_ = n + 1                      # same gate, one qubit no Toffoli touches
        dp, _ = diag_companion_floor(tof, np_)
        nup = nullity(tof, np_)
        naive_ok = (2 * np_ + 1) <= (2 * nup + 1)   # would 2n'+1 still be a bound?
        print(f"    {k} | {n:2d}  {d}  {2*n+1:4d}  {2*nu+1:5d} |"
              f"         {np_:2d}   {dp}   {2*np_+1:4d}   {2*nup+1:5d} | {naive_ok}")
        # unpadded: non-degenerate, and there 2n+1 and 2nu+1 agree
        assert d == 0 and 2 * n + 1 == 2 * nu + 1 == (1 << (k + 1)) - 1
        # padded: the idle qubit IS a radical direction
        assert dp == 1, "the idle qubit must contribute a radical direction"
        # nu, hence 2nu+1, is untouched by padding -- n is not
        assert nup == nu == (1 << k) - 1, "nu must be padding-invariant"
        assert 2 * nup + 1 == 2 * nu + 1 == (1 << (k + 1)) - 1
        # and 2n'+1 would OVERSTATE the floor, i.e. it is not a valid bound
        assert not naive_ok, "2n+1 must fail for the padded gate"
    print("   => 2n+1 is not available once d>0; 2*nu+1 is unchanged. nu is the")
    print("      invariant, which is exactly what rem:padding says.")


def check_mismatch_classifier():
    print("\n(mismatch) semi-Clifford classifier vs decode")
    # mismatch-free 2-block fan-out (shares control 0): TOF(0,1,2), TOF(0,3,4)
    fo = [(0, 1, 2), (0, 3, 4)]
    print(f"  fan-out TOF(0,1,2)TOF(0,3,4): mismatch-free={is_mismatch_free(fo)} (expect True)")
    d, floor = diag_companion_floor(fo, 5)
    dec, naive = diag_companion_decode(fo, 5)
    print(f"    diagonal companion: floor={floor}, exact delta={dec}  (fan-out 4m+3=11)")
    assert is_mismatch_free(fo) and floor == 11 and dec == 11
    # mismatched: TOF(0,1,2) then TOF(2,3,4) -- qubit 2 target then control
    mm = [(0, 1, 2), (2, 3, 4)]
    print(f"  chain TOF(0,1,2)TOF(2,3,4): mismatch-free={is_mismatch_free(mm)} (expect False)")
    assert not is_mismatch_free(mm)
    # its diagonal companion still gets a floor (D2), decoded:
    d, floor = diag_companion_floor(mm, 5)
    dec, naive = diag_companion_decode(mm, 5)
    print(f"    diagonal companion: floor={floor}, exact delta={dec}  (linear overlap 6m+1-2s, m=2,s=1 -> 11)")
    assert floor == 11 and dec == 11

def check_adder():
    print("\n(D4) ripple-carry adder carry chain (shared-carry Toffolis)")
    print("  m   n   d  floor  nu  exact delta")
    for m in range(1, 4):
        tof, n = adder_carry_toffolis(m)
        d, floor = diag_companion_floor(tof, n)
        nu = nullity(tof, n)
        dec, naive = diag_companion_decode(tof, n)
        mf = is_mismatch_free(tof)
        print(f"  {m}   {n}   {d}   {floor:12d}   {nu}   {dec}   (mismatch-free={mf})")

if __name__ == "__main__":
    check_selftests()
    check_u3()
    check_uk()
    check_uk_general()
    check_padding_necessity()
    check_structure_constants()
    check_mismatch_classifier()
    check_adder()
    print("\nALL PASS")
