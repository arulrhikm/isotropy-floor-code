"""Isotropy floor vs residue weight for the GF(2^k) multiplication phase oracle.

U_k |x,y,z> = (-1)^{z.(x*y)} |x,y,z> on n = 3k qubits (prop:fieldoracle).

For each k this script:
  1. builds the cubic tensor T_{q,r,l} = [bit l of e_q * e_r]  (cross-register only),
  2. verifies the radical is trivial (d = 0, so floor = 2n+1 = 6k+1),
  3. computes the weight of the merged canonical residue: for each nonempty
     parity-support T (|T| <= 3 unrestricted subsets of the n variables), the
     residue coefficient is #{cubic monomials S : T subset of S} mod 2, so for a
     homogeneous cubic the odd parities are the T contained in an odd number of
     monomial supports.  This weight is an upper bound on delta(U_k) and equals
     17 at k=2 where exact decoding proved it optimal (rem:gf4).

Output: table  k, n, d, floor 6k+1, residue weight W_k  for k = 1..6.
"""

from itertools import combinations, product

# Irreducible polynomials over GF(2), poly bits: bit i = coefficient of X^i.
#
# W_k is basis-dependent: different irreducibles give different merged-residue
# weights, every one of them a valid upper bound on delta(U_k).  (delta itself
# is NOT basis-dependent -- two representations of GF(2^k) differ by a field
# isomorphism, which is F2-linear on each register, hence a CNOT circuit, hence
# Clifford, hence T-count preserving.)  So the sharpest ceiling this method can
# give is the minimum of W over all irreducibles of degree k, and the entries
# below realise that minimum.  Exhaustive sweep (sweep_gf.py):
#
#   k        1    2    3    4    5     6     7
#   #irred   1    1    2    3    6     9    18
#   min W    7   17   36   58   90   123   166
#   max W    7   17   37   68  109   168   237
#
IRRED = {
    1: 0b11,        # X + 1 (GF(2): multiplication mod X+1 is trivial field; use X)
    2: 0b111,       # X^2 + X + 1
    3: 0b1011,      # X^3 + X + 1
    4: 0b10011,     # X^4 + X + 1
    5: 0b100101,    # X^5 + X^2 + 1
    6: 0b1000011,   # X^6 + X + 1  -> W = 123.  The pentanomial X^6+X^4+X^3+X+1
                    # used previously is a valid but suboptimal basis (W = 161).
}


def gf_mult(a, b, k):
    """Carry-less multiply mod the degree-k irreducible polynomial."""
    if k == 1:
        return a & b
    poly = IRRED[k]
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a <<= 1
        if a >> k:
            a ^= poly
    return res


def build_monomials(k):
    """Supports {x_q, y_r, z_l} with T_{qrl} = 1, as frozensets of 0..3k-1.

    Variable indexing: x_q -> q, y_r -> k+r, z_l -> 2k+l.
    """
    monos = []
    for q in range(k):
        for r in range(k):
            prod_bits = gf_mult(1 << q, 1 << r, k)
            for l in range(k):
                if (prod_bits >> l) & 1:
                    monos.append(frozenset({q, k + r, 2 * k + l}))
    return monos


def radical_dim(monos, n):
    """dim of {w : sum_a w_a T_{a,p,q} = 0 for all pairs p<q} over F2."""
    # rows indexed by pairs (p,q); columns by a; entry T_{a,p,q}
    rows = {}
    for S in monos:
        a_, b_, c_ = sorted(S)
        for (p, q, a) in ((a_, b_, c_), (a_, c_, b_), (b_, c_, a_)):
            rows.setdefault((p, q), 0)
            rows[(p, q)] ^= 1 << a
    mat = [bits for bits in rows.values() if bits]
    # Gaussian elimination over F2
    rank = 0
    for col in range(n):
        piv = next((i for i in range(rank, len(mat)) if (mat[i] >> col) & 1), None)
        if piv is None:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        for i in range(len(mat)):
            if i != rank and (mat[i] >> col) & 1:
                mat[i] ^= mat[rank]
        rank += 1
    return n - rank


def residue_weight(monos):
    """#{nonempty T : T lies in an odd number of monomial supports}."""
    from collections import Counter
    cnt = Counter()
    for S in monos:
        elems = sorted(S)
        for size in (1, 2, 3):
            for T in combinations(elems, size):
                cnt[T] += 1
    return sum(1 for v in cnt.values() if v & 1)


def main():
    print(f"{'k':>2} {'n':>3} {'#monos':>7} {'d':>3} {'floor 6k+1':>11} {'W_k':>6}")
    for k in range(1, 7):
        n = 3 * k
        monos = build_monomials(k)
        d = radical_dim(monos, n)
        w = residue_weight(monos)
        floor = 2 * (n - d) + (1 if d == 0 else 0)
        print(f"{k:>2} {n:>3} {len(monos):>7} {d:>3} {floor:>11} {w:>6}")
        assert d == 0, f"radical not trivial at k={k}!"
    print("\nfloor = 2n+1 = 6k+1 (non-degenerate, prop:fieldoracle); "
          "W_k = merged-residue upper bound on delta(U_k); "
          "delta(U_1)=7 exact, delta(U_2)=17 exact by decoding (rem:gf4).")


if __name__ == "__main__":
    main()
