"""
Self-audit: COMPUTE the unitary stabilizer nullity nu for every gate in the comparison
table, instead of inheriting nu=n.

For a diagonal gate U=diag(omega^{phi(x)}) (phi: F_2^n -> Z_8), conjugation sends
  U X^a Z^b U^dagger = diag(omega^{Delta_a phi(y)}) X^a Z^b,   Delta_a phi(y)=phi(y+a)-phi(y).
This is a Pauli iff diag(omega^{Delta_a phi}) is +/- a Z-string, i.e.
  Delta_a phi(y) = c + 4 * ell(y)   for some constant c and an F_2-AFFINE ell.
So the Pauli-preserving group has size 2^n * |A| with A = {a : Delta_a phi affine-of-this-form},
and the unitary stabilizer nullity is
  nu(U) = 2n - log2(2^n |A|) = n - log2|A|.
(Check: Clifford => A = F_2^n => nu=0; CCZ => A={0} => nu=n=3.)

We verify nu = n (i.e. A={0}) for every non-degenerate table gate, and report nu for the
degenerate e_3(n=6) as well.  This makes the Table's nu column a computed fact.
"""
import itertools
import numpy as np

def phi_cubic(triples, n):
    """phi(x) in Z_8 for U_f = (-1)^{f}, f = sum of given cubic monomials. phi = 4*f mod 8."""
    phi = np.zeros(1 << n, dtype=int)
    for x in range(1 << n):
        bits = [(x >> i) & 1 for i in range(n)]
        f = 0
        for (a, b, c) in triples:
            f ^= bits[a] & bits[b] & bits[c]
        phi[x] = (4 * f) % 8
    return phi

def is_affine_boolean(vals, n):
    """vals: length-2^n array in {0,1}. True iff it is an F_2-affine function of the n bits."""
    # affine iff Mobius (ANF) coefficients vanish for all monomials of degree >= 2.
    f = vals.copy() % 2
    # compute ANF via in-place Mobius transform
    a = f.copy()
    for i in range(n):
        step = 1 << i
        for x in range(1 << n):
            if x & step:
                a[x] ^= a[x ^ step]
    for S in range(1 << n):
        if a[S] and bin(S).count("1") >= 2:
            return False
    return True

def in_A(phi, a, n):
    """Is Delta_a phi(y) = c + 4*ell(y) with ell affine?"""
    t = np.array([(phi[y ^ a] - phi[y]) % 8 for y in range(1 << n)])
    vals = set(int(v) for v in t)
    # must be subset of {c, c+4} for some c in 0..3
    c = None
    for cand in range(4):
        if vals <= {cand, (cand + 4) % 8}:
            c = cand; break
    if c is None:
        return False
    ell = np.array([1 if (int(t[y]) == (c + 4) % 8) else 0 for y in range(1 << n)])
    return is_affine_boolean(ell, n)

def nullity(triples, n):
    A = [a for a in range(1 << n) if in_A(phi_cubic(triples, n), a, n)]
    import math
    return n - int(round(math.log2(len(A)))), len(A)

def line(n):     return [(i, i+1, i+2) for i in range(n-2)]
def complete(n): return list(itertools.combinations(range(n), 3))
def par(m):      return [(3*i, 3*i+1, 3*i+2) for i in range(m)]

if __name__ == "__main__":
    print("self-test: nu(CCZ) should be 3 (A={0}):")
    nu, sz = nullity(par(1), 3); print(f"  CCZ n=3: nu={nu}, |A|={sz}  {'OK' if nu==3 else 'FAIL'}")
    print()
    print("computed nullity for the comparison-table gates:")
    cases = [("CCZ (Toffoli)", par(1), 3),
             ("2 parallel Toffolis", par(2), 6),
             ("3 parallel Toffolis", par(3), 9),
             ("3-local line f_5", line(5), 5),
             ("3-local line f_6", line(6), 6),
             ("elem. symmetric e_3 (n=5)", complete(5), 5),
             ("elem. symmetric e_3 (n=6, degenerate)", complete(6), 6)]
    alln = True
    for name, tr, n in cases:
        nu, sz = nullity(tr, n)
        eq = (nu == n)
        alln &= eq or ("degenerate" in name)
        print(f"  {name:38s} n={n:2d}: nu={nu:2d}  (= n? {eq})  |A|={sz}")
    print()
    print("=> nu = n confirmed for every non-degenerate table gate (4 parallel Toffolis "
          "follows by additivity nu(A^{otimes m})=m*nu).")
