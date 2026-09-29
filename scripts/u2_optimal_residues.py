"""
Every optimal representation of the GF(4) multiplication oracle U_2, and what forces 17.

U_2 on n = 6 qubits has floor 2n+1 = 13 and delta = 17.  The coset Res(U_2) + RM(2,6)* has
2^22 members; this script lists every member of weight 17 and, for each, the structure the
isotropy argument sees:

  * Y (17 x 6 over F_2), its column code C = colspace(Y) (self-orthogonal, dim 6),
  * the weight distribution of C and the check  wt(c(w)) >= rank T(w)  for every w   (fact F1:
    T(w) = Y^T diag(c(w)) Y, so the slice is the Gram matrix of Y on the support of c(w)),
  * g = dim C^perp / C  (= t - 2 dim C = 5), and which of the natural vectors span it:
    the all-ones vector, the same-register products c_a AND c_b, all products c_a AND c_b,
  * the register profile: how many rows touch 1, 2 or 3 of the registers x, y, z.

Run:  python u2_optimal_residues.py
"""
import itertools, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_amdist import ccz_on, residue_vec, rm_generators
from field_oracle import mult_tensor_blocks

K = 2
N = 3 * K
REG = [set(range(K)), set(range(K, 2 * K)), set(range(2 * K, 3 * K))]


def f2rank(rows):
    rows = [int(r) for r in rows if r]
    rank = 0
    while rows:
        piv = min(rows)
        rows.remove(piv)
        rank += 1
        low = piv & -piv
        rows = [r ^ piv if r & low else r for r in rows]
        rows = [r for r in rows if r]
    return rank


def span(rows):
    """All F_2 combinations of the given bitmask rows (list of ints)."""
    out = {0}
    for r in rows:
        out |= {v ^ r for v in out}
    return out


def tensor_of(blocks):
    T = set()
    for b in blocks:
        T.add(tuple(sorted(b)))
    return T


def slice_rank(T, w):
    """rank over F_2 of T(w)_{bc} = sum_a w_a T_{abc}."""
    M = [0] * N
    for (a, b, c) in T:
        for (p, q, r) in ((a, b, c), (b, a, c), (c, a, b)):
            if (w >> p) & 1:
                M[q] ^= 1 << r
                M[r] ^= 1 << q
    return f2rank(M)


def main():
    coeff = {}
    blocks = mult_tensor_blocks(K)
    for B in blocks:
        ccz_on(coeff, B)
    T = tensor_of(blocks)
    r = residue_vec(coeff, N)                     # length 63, index y-1
    G = rm_generators(N)                          # RM(2,6)*, 22 rows
    L = len(r)
    assert L <= 64 and G.shape[0] == 22

    def pack(v):
        return sum(int(v[i]) << i for i in range(L))

    rp = pack(r)
    gp = np.array([pack(G[i]) for i in range(G.shape[0])], dtype=np.uint64)
    t0 = time.time()
    cur = np.zeros(1, dtype=np.uint64)
    for g in gp:
        cur = np.concatenate([cur, cur ^ g])
    words = cur ^ np.uint64(rp)
    # popcount of 64-bit words
    x = words.copy()
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    w = ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)) & np.uint64(0xff)
    dmin = int(w.min())
    opt = words[w == dmin]
    print(f"coset size 2^22 scanned in {time.time()-t0:.1f}s: delta(U_2) = {dmin}, "
          f"{len(opt)} residues of that weight  (floor 2n+1 = {2*N+1})")
    hist = {}
    for wt in range(dmin, dmin + 6):
        hist[wt] = int((w == wt).sum())
    print(f"coset weight counts near the minimum: {hist}")

    profiles = {}
    for idx, word in enumerate(opt):
        word = int(word)
        rows = [y for y in range(1, 1 << N) if (word >> (y - 1)) & 1]
        t = len(rows)
        assert t == dmin
        # columns c_a as t-bit ints, bit j = row j contains qubit a
        cols = [sum(((y >> a) & 1) << j for j, y in enumerate(rows)) for a in range(N)]
        # moment check
        for A in range(1, N):
            pass
        for a in range(N):
            assert bin(cols[a]).count("1") % 2 == 0
            for b in range(a + 1, N):
                assert bin(cols[a] & cols[b]).count("1") % 2 == 0
                for c in range(b + 1, N):
                    par = bin(cols[a] & cols[b] & cols[c]).count("1") % 2
                    assert par == ((a, b, c) in T), (a, b, c)
        C = span(cols)
        dimC = len(C).bit_length() - 1
        assert dimC == N
        # weight distribution of C, indexed by w, and the F1 check
        wdist = {}
        f1_ok = True
        f1_tight = 0
        for wv in range(1, 1 << N):
            cw = 0
            for a in range(N):
                if (wv >> a) & 1:
                    cw ^= cols[a]
            wt = bin(cw).count("1")
            wdist[wt] = wdist.get(wt, 0) + 1
            rk = slice_rank(T, wv)
            if wt < rk:
                f1_ok = False
            if wt == rk:
                f1_tight += 1
        # C^perp
        full = 1 << t
        Cperp = [v for v in range(full)
                 if all(bin(v & c).count("1") % 2 == 0 for c in cols)]
        Cperp_set = set(Cperp)
        g = (len(Cperp).bit_length() - 1) - dimC
        ones = full - 1
        one_in_C = ones in C
        same = []
        allp = []
        for a in range(N):
            for b in range(a + 1, N):
                p = cols[a] & cols[b]
                allp.append(p)
                if any(a in R and b in R for R in REG):
                    same.append(p)
        assert all(p in Cperp_set for p in same)
        dim_C_same = (len(span(cols + same)).bit_length() - 1) - dimC
        dim_C_same_1 = (len(span(cols + same + [ones])).bit_length() - 1) - dimC
        allp_perp = [p for p in allp if p in Cperp_set]
        dim_C_allperp_1 = (len(span(cols + allp_perp + [ones])).bit_length() - 1) - dimC
        # register profile
        prof = [0, 0, 0]
        for y in rows:
            k = sum(1 for R in REG if any((y >> a) & 1 for a in R))
            prof[k - 1] += 1
        key = (tuple(sorted(wdist.items())), g, one_in_C, dim_C_same, dim_C_same_1,
               dim_C_allperp_1, tuple(prof), f1_ok, f1_tight)
        profiles.setdefault(key, []).append(rows)

    print(f"\n{len(profiles)} distinct structural profiles among the {len(opt)} optimal residues:")
    for key, members in sorted(profiles.items(), key=lambda kv: -len(kv[1])):
        wdist, g, one_in_C, ds, ds1, dap1, prof, f1_ok, f1_tight = key
        print(f"\n  x{len(members)}  weights of C {dict(wdist)}")
        print(f"      g = dim C^perp/C = {g}   1 in C: {one_in_C}")
        print(f"      dim (C + same-register products)/C = {ds};  with the all-ones vector: {ds1};"
              f"  with every product lying in C^perp, and 1: {dap1}")
        print(f"      rows touching 1/2/3 registers: {prof}")
        print(f"      F1 (wt c(w) >= rank T(w)) holds: {f1_ok};  equality on {f1_tight} of 63 w")
        rows = members[0]
        print("      example rows (x|y|z bits):",
              " ".join(f"{(y&3):02b}|{(y>>2)&3:02b}|{(y>>4)&3:02b}" for y in rows))
    # rank spectrum of U_2 itself
    spec = {}
    for wv in range(1, 1 << N):
        rk = slice_rank(T, wv)
        spec[rk] = spec.get(rk, 0) + 1
    print(f"\nslice-rank spectrum of U_2 over the 63 nonzero w: {spec}   (predicted: all 2k = 4)")


if __name__ == "__main__":
    main()
