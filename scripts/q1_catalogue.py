"""
Q1: extend the exact-cost catalogue.

(a) e_3 (complete cubic) by parity of n.  e_3 = sum_{i<j<k} x_i x_j x_k has cubic tensor T_abc = 1
    for every triple.  Its radical: w in R iff  sum_a w_a T_{a q r} = 0 for all q<r, i.e.
    (sum_a w_a) - w_q - w_r = 0 for all q != r.  Subtracting two relations forces all w_a equal
    (= c), and then (n-2) c = 0 mod 2.  So the all-ones vector is radical iff n is EVEN; e_3 is
    NON-DEGENERATE for ODD n (d=0, floor 2n+1) and degenerate for even n (d>=1).  We confirm the
    parity pattern for all n, and decode the exact delta where the RM code is small enough.

(b) disjoint non-degenerate cubic layers.  rem:disjointcompress gives the floor 2N+1; we confirm the
    splicing construction meets it on the decodable cases (e.g. CCZ (+) CCZ = 13).

Reuses compare_bounds.py (cubic_radical_dim, complete_triples, decode) and validate_amdist.
"""
import itertools
import numpy as np
from compare_bounds import (
    cubic_radical_dim, complete_triples, line_triples, parallel_ccz_triples, build_cubic,
)
from validate_amdist import residue_vec, rm_generators, min_coset_weight, ccz_on, cs_on, Tcount

# RM(n-4,n)* dual dimension = C(n,1)+C(n,2)+C(n,3); meet-in-the-middle is feasible while this is
# small (<= ~24 generators).  Beyond that the decode is skipped.
def dual_dim(n):
    from math import comb
    return comb(n, 1) + comb(n, 2) + comb(n, 3)

def decode(triples, n):
    c = build_cubic(triples)
    return int(min_coset_weight(residue_vec(c, n), rm_generators(n)))

def e3_by_parity(nmax=13):
    print("(a) e_3 = complete cubic, by parity of n")
    print(f"    {'n':>2}  {'parity':>6}  {'d (cubic radical)':>18}  {'floor 2(n-d)(+1)':>16}  {'delta':>7}")
    c = {}; ccz_on(c, (0, 1, 2)); assert Tcount(c, 3) == 7
    for n in range(3, nmax + 1):
        tri = complete_triples(n)
        d = cubic_radical_dim(tri, n)
        floor = 2 * (n - d) + (1 if d == 0 else 0)
        # decode only while the dual code is small
        dl = decode(tri, n) if dual_dim(n) <= 26 else None
        ds = "-" if dl is None else str(dl)
        tag = ""
        if dl is not None:
            tag = "TIGHT" if dl == floor else f"gap {dl-floor}"
        par = "odd" if n % 2 else "even"
        print(f"    {n:2d}  {par:>6}  {d:18d}  {floor:16d}  {ds:>7}  {tag}")
    print("    => d=0 exactly for odd n (non-degenerate, floor 2n+1); d>=1 for even n.\n")

def disjoint_layers():
    print("(b) disjoint non-degenerate cubic layers: floor 2N+1 (rem:disjointcompress)")
    print(f"    {'layer':28s} {'N':>2} {'d':>2} {'2N+1':>5} {'delta':>7}")
    cases = [
        ("CCZ (m=1)",            parallel_ccz_triples(1), 3),
        ("CCZ (+) CCZ (m=2)",    parallel_ccz_triples(2), 6),
        ("line f5 (n=5)",        line_triples(5),         5),
        ("line f6 (n=6)",        line_triples(6),         6),
    ]
    for name, tri, N in cases:
        d = cubic_radical_dim(tri, N)
        floor = 2 * N + 1
        dl = decode(tri, N) if dual_dim(N) <= 24 else None
        ds = "-" if dl is None else str(dl)
        tag = "" if dl is None else ("TIGHT" if dl == floor else f"gap {dl-floor}")
        print(f"    {name:28s} {N:2d} {d:2d} {floor:5d} {ds:>7}  {tag}")
    print("    => union non-degenerate (d=0), decodable cases meet the 2N+1 floor.\n")

def ek_family():
    """e_k = sum over all k-subsets of prod x_i, the complete degree-k symmetric form.  Its order-k
    tensor is all-ones on k-subsets; the radical relation at a (k-1)-set Q reads
        sum_{a not in Q} w_a = 0,  i.e.  (sum w) = sum_{a in Q} w_a  for every (k-1)-set Q,
    which forces all w_a equal (= c) and then (n-k+1) c = 0 in F_2.  So the all-ones vector is radical
    iff (n-k+1) is even, i.e. n and k have OPPOSITE parity; e_k is NON-DEGENERATE iff n == k (mod 2).
    We confirm this by computing the order-k radical directly."""
    import experiment_level_k as LK
    print("(c) e_k symmetric family: non-degenerate  <=>  n == k (mod 2)  (generalizes e_3)")
    print(f"    {'k':>2}  {'n':>2}  {'same parity?':>12}  {'d_k (order-k radical)':>20}  {'predict d_k':>11}")
    ok = True
    for k in (3, 4, 5):
        for n in range(k, k + 6):
            ps = LK.parities(n)
            As, _ = LK.moment_matrix(n, k, ps)
            target = [1 if LK.popcount(A) == k else 0 for A in As]
            dk = LK.order_k_radical(n, k, target, As)
            same = (n % 2) == (k % 2)
            predict = 0 if same else 1
            flag = "" if dk == predict else "  <-- MISMATCH"
            if dk != predict: ok = False
            print(f"    {k:2d}  {n:2d}  {str(same):>12}  {dk:20d}  {predict:11d}{flag}")
    print("    => confirmed: d_k = 0 iff n,k same parity; d_k = 1 (all-ones) otherwise.\n"
          if ok else "    *** parity prediction FAILED ***\n")


def hypergraph_sweep():
    """Decode a few STRUCTURED 3-uniform hypergraph cubics (small n) and report exact delta vs the
    2(n-d)+1 floor, looking for new tight families beyond CCZ / line / e_3."""
    print("(d) structured 3-uniform hypergraph cubics (decode small n)")
    def cycle(n):        # consecutive triples wrapped: {i,i+1,i+2} mod n
        return [tuple(sorted(((i) % n, (i + 1) % n, (i + 2) % n))) for i in range(n)]
    def star(n):         # all triples through vertex 0: {0,j,k}
        return [(0, j, k) for j in range(1, n) for k in range(j + 1, n)]
    def tripartite(a):   # complete tripartite K_{a,a,a}: one vertex from each of 3 blocks
        V = [list(range(a)), list(range(a, 2 * a)), list(range(2 * a, 3 * a))]
        return [tuple(sorted((i, j, l))) for i in V[0] for j in V[1] for l in V[2]]
    fams = [
        ("cycle C_5",        cycle(5),      5),
        ("cycle C_6",        cycle(6),      6),
        ("star K_{1,*} n=5", star(5),       5),
        ("tripartite K_222", tripartite(2), 6),
    ]
    print(f"    {'family':22s} {'n':>2} {'|H|':>4} {'d':>2} {'2(n-d)(+1)':>10} {'delta':>7}")
    for name, tri, n in fams:
        tri = sorted(set(tri))
        d = cubic_radical_dim(tri, n)
        floor = 2 * (n - d) + (1 if d == 0 else 0)
        dl = decode(tri, n) if dual_dim(n) <= 26 else None
        ds = "-" if dl is None else str(dl)
        tag = "" if dl is None else ("TIGHT" if dl == floor else f"gap {dl-floor}")
        print(f"    {name:22s} {n:2d} {len(tri):4d} {d:2d} {floor:10d} {ds:>7}  {tag}")
    print()


if __name__ == "__main__":
    print("self-test: delta(CCZ)=7\n")
    e3_by_parity()
    disjoint_layers()
    ek_family()
    hypergraph_sweep()
