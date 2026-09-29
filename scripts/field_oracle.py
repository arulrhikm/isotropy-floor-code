"""
Field-multiplication phase oracle floor (prop:fieldoracle).

The oracle U|x,y,z> = (-1)^{z . (x*y)} |x,y,z>, with x,y,z in GF(2^k) and *
the field product, is a pure-cubic gate on n = 3k qubits whose tensor is the
multiplication tensor of GF(2^k).  Claims:
  (a) the tensor is non-degenerate (d = 0) for every k, so delta >= 6k+1;
  (b) k=1 is CCZ (delta = 7 = 6*1+1, tight);
  (c) k=2 (GF(4), n=6): exact decode of delta -- is the floor 13 attained?
Also reports the schoolbook upper bound (one CCZ per monomial after merging).
"""
import itertools, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_amdist import ccz_on, residue_vec, rm_generators, min_coset_weight
from overlap_layers import radical_dim

# GF(2^k) arithmetic with a given irreducible polynomial (bitmask incl. top bit)
IRRED = {1: 0b11, 2: 0b111, 3: 0b1011, 4: 0b10011}  # x, x^2+x+1, x^3+x+1, x^4+x+1

def gf_mul(a, b, k):
    p = IRRED[k]
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> k:
            a ^= p
    return r

def mult_tensor_blocks(k):
    """Triples (x_i, y_j, z_l) with coefficient 1 in z.(x*y):
    (x*y)_l = sum_{i,j} x_i y_j [e_i * e_j has bit l]."""
    blocks = []
    for i in range(k):
        for j in range(k):
            prod = gf_mul(1 << i, 1 << j, k)
            for l in range(k):
                if (prod >> l) & 1:
                    blocks.append((i, k + j, 2 * k + l))
    return blocks

def run():
    print("(a) non-degeneracy of the GF(2^k) multiplication tensor")
    for k in (1, 2, 3, 4):
        blocks = mult_tensor_blocks(k)
        n = 3 * k
        d = radical_dim(blocks, n)
        print(f"  k={k}  n={n}  #monomials={len(blocks)}  d={d}  floor 2n+1={2*n+1}"
              f"  schoolbook<=7*{len(blocks)}={7*len(blocks)}")
        assert d == 0
    print("  -> d=0 for k<=4; floor delta >= 6k+1")

    print("\n(b) k=1: the oracle is CCZ")
    c = {}
    for B in mult_tensor_blocks(1):
        ccz_on(c, B)
    dec = min_coset_weight(residue_vec(c, 3), rm_generators(3))
    print(f"  delta = {dec}  (expect 7 = 6*1+1, tight)")
    assert dec == 7

    print("\n(c) k=2: GF(4) multiplication oracle on n=6, exact decode")
    c = {}
    for B in mult_tensor_blocks(2):
        ccz_on(c, B)
    dec = min_coset_weight(residue_vec(c, 6), rm_generators(6))
    print(f"  delta = {dec}   floor 6k+1 = 13   naive residue weight = {int(residue_vec(c,6).sum())}")
    print(f"  floor attained: {dec == 13}")

if __name__ == "__main__":
    run()
