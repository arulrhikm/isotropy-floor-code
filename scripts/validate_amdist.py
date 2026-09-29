r"""
Exact ancilla-free {CNOT,T} T-count via Amy-Mosca:
T-count(U_a) = min Hamming weight in the coset Res(a) + RM(n-4,n)*,
the code indexed by parities y in F_2^n \ {0} (length 2^n-1),
RM(n-4,n)* = punctured evaluations of monomials prod_{i in A} y_i, |A| <= n-4.

We compute the residue of a level-3 phase a(x) in Z_8 by least-squares-free
exact construction: find integer parity-coefficients m_y with
  sum_y m_y (y.x) == a(x) (mod 8) for all x,
then residue = m mod 2.  We obtain ONE rep by an explicit construction for the
families we use (sums of CCZ/CS/T monomials, complete-cubic), each expanded into
parities via the exact identities. Then brute/meet-in-middle min coset weight.
"""
import numpy as np, itertools, math

def parities(n):  # list of y in 1..2^n-1
    return list(range(1, 1<<n))

# ---- exact parity expansion of monomial phases (over Z, valid as Z_8) ----
# 4 * x_{i1}..x_{i3}  (CCZ on a triple)  -> sum over nonempty S subseteq triple of
#   (-1)^{|S|-1} chi_{xor of those vars};  coeff = (-1)^{|S|-1} (odd everywhere)
# 2 * x_i x_j (CS)    -> chi_i + chi_j - chi_{ij}   (coeffs 1,1,-1)
# 1 * x_i (T)         -> chi_i
# We accumulate integer coeff dict keyed by parity-bitmask y.

def add_term(coeff, ymask, c):
    coeff[ymask] = coeff.get(ymask,0) + c

def ccz_on(coeff, triple):  # 4*prod -> sum_{nonempty S} (-1)^{|S|-1} chi_{XOR}
    for r in range(1, len(triple)+1):
        for S in itertools.combinations(triple, r):
            y = 0
            for i in S: y ^= (1<<i)
            add_term(coeff, y, (-1)**(r-1))

def cs_on(coeff, pair):  # 2*x_i x_j
    i,j = pair
    add_term(coeff, 1<<i, 1); add_term(coeff, 1<<j, 1); add_term(coeff, (1<<i)|(1<<j), -1)

def t_on(coeff, i):  # 1*x_i
    add_term(coeff, 1<<i, 1)

def complete_cubic(coeff, n):  # 4 * sum_{i<j<k} x_i x_j x_k
    for trip in itertools.combinations(range(n),3):
        ccz_on(coeff, trip)

def phase_from_coeff(coeff, n):  # evaluate a(x) mod 8 for all x, to verify
    a = np.zeros(1<<n, dtype=int)
    xs = np.arange(1<<n)
    for y,c in coeff.items():
        # chi_y(x) = parity of (x & y)
        py = np.array([bin(x & y).count('1') & 1 for x in xs])
        a = (a + c*py) % 8
    return a

def residue_vec(coeff, n):
    L = (1<<n)-1
    r = np.zeros(L, dtype=np.uint8)
    for y,c in coeff.items():
        if c % 2: r[y-1] ^= 1
    return r

# ---- RM(n-1-k,n)* generator rows (monomials prod y_i, deg<=n-1-k), eval at y=1..2^n-1 ----
# The level-k free code is RM(n-1-k,n)*: residues whose order-<=k moments all vanish.  Level 3
# (T/CCZ) is the k=3 case, RM(n-4,n)*.  min-weight in Res + this code = delta_k (exact at k=3;
# conjectural but empirically verified via moment_delta at k>=4, see experiment_level_k.py).
def rm_generators_k(n, k):
    r = n - 1 - k
    L = (1<<n)-1
    ys = np.arange(1, 1<<n)
    rows=[]
    if r < 0: return np.zeros((0,L),dtype=np.uint8)
    for d in range(0, r+1):
        for A in itertools.combinations(range(n), d):
            maskA = 0
            for i in A: maskA |= (1<<i)
            # value at y = AND of bits of y at positions A == 1  -> (y & maskA)==maskA
            row = ((ys & maskA) == maskA).astype(np.uint8)
            rows.append(row)
    return np.array(rows, dtype=np.uint8)

def rm_generators(n):
    """Level-3 free code RM(n-4,n)* (the k=3 case of rm_generators_k)."""
    return rm_generators_k(n, 3)

def min_coset_weight(r, G, return_arg=False):
    # min over codewords c in rowspan(G) of popcount(r xor c)
    # return_arg=True also returns the argmin coset representative (r xor c) as a length-L uint8 vector.
    k = G.shape[0]
    if k == 0:
        return (int(r.sum()), np.array(r, dtype=np.uint8)) if return_arg else int(r.sum())
    L = len(r)
    # pack into uint64 words
    nb = (L + 63)//64
    def pack(v):
        out = np.zeros(nb, dtype=np.uint64)
        for i in range(L):
            if v[i]: out[i>>6] |= np.uint64(1) << np.uint64(i & 63)
        return out
    rp = pack(r)
    Gp = np.array([pack(G[i]) for i in range(k)], dtype=np.uint64)
    # meet in the middle: split generators
    k1 = k//2; k2 = k - k1
    def all_combos(gp, kk):
        cur = np.zeros((1,nb), dtype=np.uint64)
        for i in range(kk):
            cur = np.concatenate([cur, cur ^ gp[i]], axis=0)
        return cur
    A = all_combos(Gp[:k1], k1)          # 2^k1 x nb
    B = all_combos(Gp[k1:], k2)          # 2^k2 x nb
    # we want min popcount( rp xor a xor b ). Fix each a, xor with rp, then min over B.
    def popcount64(x):
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
        return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)) & np.uint64(0xff)
    best = L+1
    best_word = None
    # iterate over A (smaller), vectorize over B
    rpA = A ^ rp[None,:]    # 2^k1 x nb
    for a in range(rpA.shape[0]):
        xb = B ^ rpA[a][None,:]       # 2^k2 x nb = (r xor codeword) packed
        w = popcount64(xb).sum(axis=1)
        bmin = int(w.argmin()); m = int(w[bmin])
        if m < best:
            best = m
            if return_arg: best_word = xb[bmin].copy()
    if not return_arg:
        return best
    # unpack best_word (packed uint64 words) to a length-L 0/1 vector
    rep = np.zeros(L, dtype=np.uint8)
    for i in range(L):
        if (best_word[i >> 6] >> np.uint64(i & 63)) & np.uint64(1):
            rep[i] = 1
    return best, rep

def Tcount(coeff, n, verify_phase=None):
    if verify_phase is not None:
        a = phase_from_coeff(coeff, n)
        assert np.array_equal(a, verify_phase), "phase mismatch"
    r = residue_vec(coeff, n)
    G = rm_generators(n)
    return min_coset_weight(r, G)

if __name__=="__main__":
    # ---- VALIDATION ----
    print("VALIDATION (known exact values):")
    # CCZ n=3
    c={}; ccz_on(c,(0,1,2)); print("  CCZ (n=3):", Tcount(c,3), "expect 7")
    # CS n=3
    c={}; cs_on(c,(0,1)); print("  CS  (n=3):", Tcount(c,3), "expect 3")
    # T n=3
    c={}; t_on(c,0); print("  T   (n=3):", Tcount(c,3), "expect 1")
    # CCZ tensor I at n=4 (nontrivial code RM(0,4)*)
    c={}; ccz_on(c,(0,1,2)); print("  CCZ x I (n=4):", Tcount(c,4), "expect 7")
    # CS tensor I^2 at n=4
    c={}; cs_on(c,(0,1)); print("  CS x I^2 (n=4):", Tcount(c,4), "expect 3")
    # CCZ at n=5 (two spectators)
    c={}; ccz_on(c,(0,1,2)); print("  CCZ x I^2 (n=5):", Tcount(c,5), "expect 7")
    print()
    print("HYPOTHESIS GENERATION:")
    # additivity: CCZ + T (n=5, disjoint)
    c={}; ccz_on(c,(0,1,2)); t_on(c,3); print("  CCZ(123) + T(4)   n=5:", Tcount(c,5), "expect 8 if additive")
    # CCZ + CS (n=5, disjoint)
    c={}; ccz_on(c,(0,1,2)); cs_on(c,(3,4)); print("  CCZ(123)+ CS(45)  n=5:", Tcount(c,5), "expect 10 if additive")
    # CS + CS (n=4 disjoint)
    c={}; cs_on(c,(0,1)); cs_on(c,(2,3)); print("  CS(12) + CS(34)   n=4:", Tcount(c,4), "expect 6 if additive")
