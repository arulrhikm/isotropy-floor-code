"""
Overlapping CCZ layers: floors and constructions (sec:overlap).

Claims validated here:
 (1) LINEAR-OVERLAP LEMMA: if m CCZ blocks pairwise share at most one qubit
     (their triples form a linear 3-uniform hypergraph) and cover n qubits,
     the union's cubic tensor is non-degenerate (d=0), so delta >= 2n+1.
     Checked on the fan-out star, chains, and random linear hypergraphs.
 (2) FAN-OUT CONSTRUCTION: m CCZ blocks {0,a_i,b_i} sharing one control admit a
     weight-(4m+3) representation (each block lifted by the earlier ports,
     one cancelling pair per block), matching the floor 2n+1 = 4m+3.  Moment-verified for m<=15;
     cross-checked against exact decode at m<=2.
 (3) TWO-SHARED COLLAPSE: m CCZ blocks sharing a fixed PAIR have radical
     dimension d = n-3 (floor 2(n-d)=6), and delta = 7 for every m
     (Clifford-equivalent to one CCZ; cor:cparity).
 (4) Self-tests: delta(CCZ)=7; two disjoint CCZ decode to 13;
     fan-out decode m=2 gives 11.
"""
import itertools, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_amdist import ccz_on, residue_vec, rm_generators, min_coset_weight


# ---------- moment / radical utilities ----------

def cubic_tensor_from_blocks(blocks, n):
    """T[a][q][r] over F2 as dict on sorted triples."""
    T = {}
    for B in blocks:
        t = tuple(sorted(B))
        T[t] = T.get(t, 0) ^ 1
    return {t: v for t, v in T.items() if v}

def radical_dim(blocks, n):
    """dim of {w : sum_a w_a T_{a,q,r} = 0 for all q<r}."""
    T = cubic_tensor_from_blocks(blocks, n)
    rows = []
    for q, r in itertools.combinations(range(n), 2):
        row = [0] * n
        for a in range(n):
            if a in (q, r):
                continue
            if tuple(sorted((a, q, r))) in T:
                row[a] = 1
        rows.append(row)
    M = np.array(rows, dtype=np.uint8) % 2
    # rank over F2
    M = M.copy()
    rank = 0
    rows_, cols_ = M.shape
    pr = 0
    for c in range(cols_):
        piv = None
        for r_ in range(pr, rows_):
            if M[r_, c]:
                piv = r_
                break
        if piv is None:
            continue
        M[[pr, piv]] = M[[piv, pr]]
        for r_ in range(rows_):
            if r_ != pr and M[r_, c]:
                M[r_] ^= M[pr]
        pr += 1
        rank += 1
    return n - rank

def moments_le3(masks, n):
    M = {}
    for r in (1, 2, 3):
        for A in itertools.combinations(range(n), r):
            mA = 0
            for i in A:
                mA |= 1 << i
            M[A] = sum(1 for y in masks if (y & mA) == mA) & 1
    return M

def target_moments(blocks, n):
    tset = {tuple(sorted(B)) for B in blocks}
    # mod-2 multiplicity
    tmul = {}
    for B in blocks:
        t = tuple(sorted(B))
        tmul[t] = tmul.get(t, 0) ^ 1
    M = {}
    for r in (1, 2, 3):
        for A in itertools.combinations(range(n), r):
            M[A] = tmul.get(A, 0) if r == 3 else 0
    return M


# ---------- constructions ----------

def ccz_subsets(B):
    """The 7 nonempty subsets of a block, as bitmasks."""
    B = list(B)
    out = []
    for r in range(1, 4):
        for S in itertools.combinations(B, r):
            m = 0
            for i in S:
                m ^= 1 << i
            out.append(m)
    return out

def fanout_blocks(m):
    """Blocks {0, 2i-1, 2i}, i=1..m, on n = 2m+1 qubits."""
    return [(0, 2 * i - 1, 2 * i) for i in range(1, m + 1)], 2 * m + 1

def fanout_construction(m):
    """Fan-out construction of weight 4m+3.
    Block i emits CCZ({0,a_i,b_i}) shifted by E_i = {a_1..a_{i-1}},
    omitting the port singleton {a_i} for i<m; multiset cancellation mod 2."""
    blocks, n = fanout_blocks(m)
    parity_mul = {}
    E = 0
    for i, (h, a, b) in enumerate(blocks):
        for s in ccz_subsets((h, a, b)):
            if i < m - 1 and s == (1 << a):  # omit port singleton (a_i hands to child)
                continue
            y = s ^ E
            parity_mul[y] = parity_mul.get(y, 0) ^ 1
        E ^= 1 << a
    return sorted(y for y, v in parity_mul.items() if v), blocks, n


# ---------- checks ----------

def check_selftests():
    print("SELF-TESTS")
    c = {}
    ccz_on(c, (0, 1, 2))
    r = residue_vec(c, 3)
    d = min_coset_weight(r, rm_generators(3))
    print(f"  delta(CCZ) = {d}  (expect 7)")
    assert d == 7
    # two disjoint CCZ decode to 13 (thm:parccz at m=2)
    c = {}
    ccz_on(c, (0, 1, 2)); ccz_on(c, (3, 4, 5))
    d2 = min_coset_weight(residue_vec(c, 6), rm_generators(6))
    print(f"  delta(CCZ+CCZ disjoint) = {d2}  (expect 13)")
    assert d2 == 13

def check_linear_overlap():
    print("\n(1) LINEAR-OVERLAP LEMMA: pairwise-share-<=1 blocks => d=0")
    import random
    rnd = random.Random(7)
    ok = True
    # fan-out stars and chains
    for m in range(1, 13):
        blocks, n = fanout_blocks(m)
        d = radical_dim(blocks, n)
        ok &= (d == 0)
        chain = [(2 * i, 2 * i + 1, 2 * i + 2) for i in range(m)]  # path overlap
        nc = 2 * m + 1
        dc = radical_dim(chain, nc)
        ok &= (dc == 0)
    print(f"  fan-out m=1..12 and chains m=1..12: all d=0: {ok}")
    # random linear hypergraphs
    bad = 0
    for trial in range(300):
        n = rnd.randint(5, 12)
        blocks = []
        pairs_used = set()
        for _ in range(rnd.randint(2, n)):
            B = tuple(sorted(rnd.sample(range(n), 3)))
            prs = list(itertools.combinations(B, 2))
            if any(p in pairs_used for p in prs):
                continue
            if B in [tuple(sorted(x)) for x in blocks]:
                continue
            blocks.append(B)
            pairs_used.update(prs)
        if not blocks:
            continue
        support = sorted(set().union(*[set(B) for B in blocks]))
        remap = {q: i for i, q in enumerate(support)}
        blocks2 = [tuple(remap[q] for q in B) for B in blocks]
        n2 = len(support)
        if radical_dim(blocks2, n2) != 0:
            bad += 1
    print(f"  300 random linear 3-hypergraphs (restricted to support): degenerate = {bad} (expect 0)")
    assert ok and bad == 0

def check_fanout_construction(mmax=15):
    print("\n(2) FAN-OUT CONSTRUCTION: weight 4m+3, moments match")
    print("  m   n  weight  4m+3  moments-ok")
    allok = True
    for m in range(1, mmax + 1):
        rep, blocks, n = fanout_construction(m)
        tgt = target_moments(blocks, n)
        got = moments_le3(rep, n)
        ok = (got == tgt)
        w = len(rep)
        allok &= ok and (w == 4 * m + 3)
        print(f"  {m:2d}  {n:2d}  {w:5d}  {4*m+3:4d}   {ok}")
    assert allok
    # cross-check against exact decode at m=2
    c = {}
    ccz_on(c, (0, 1, 2)); ccz_on(c, (0, 3, 4))
    d = min_coset_weight(residue_vec(c, 5), rm_generators(5))
    print(f"  exact decode m=2 (n=5): delta = {d}  (expect 11 = 4*2+3)")
    assert d == 11
    # chain pattern decode at n=5
    c = {}
    ccz_on(c, (0, 1, 2)); ccz_on(c, (2, 3, 4))
    d = min_coset_weight(residue_vec(c, 5), rm_generators(5))
    print(f"  exact decode chain CCZ(012)+CCZ(234): delta = {d}  (expect 11)")
    assert d == 11

def check_two_shared():
    print("\n(3) TWO-SHARED COLLAPSE: blocks {a,b,c_i} share the pair {a,b}")
    print("  m   n  d  n-3  floor 2(n-d)  delta (decode)")
    for m in range(1, 4):
        n = 2 + m
        blocks = [(0, 1, 2 + i) for i in range(m)]
        d = radical_dim(blocks, n)
        c = {}
        for B in blocks:
            ccz_on(c, B)
        dec = min_coset_weight(residue_vec(c, n), rm_generators(n)) if n <= 6 else None
        print(f"  {m:2d}  {n:2d}  {d}  {n-3}       {2*(n-d)}        {dec}")
        assert d == n - 3
        if dec is not None:
            assert dec == 7

if __name__ == "__main__":
    check_selftests()
    check_linear_overlap()
    check_fanout_construction()
    check_two_shared()
    print("\nALL PASS")
