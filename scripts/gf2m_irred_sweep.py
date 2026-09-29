"""Sweep W_k over ALL irreducible polynomials of degree k.

delta(U_k) is basis-independent: any two representations of GF(2^k) are related
by a field isomorphism, i.e. an invertible F2-linear change of variables on each
register, i.e. a CNOT circuit (Clifford), which preserves T-count.  The merged
canonical residue weight W is NOT basis-independent, so min over irreducibles is
a strictly better upper bound on delta(U_k) than any single choice.
"""
from itertools import combinations
from collections import Counter

def is_irred(poly, k):
    # trial-divide by all polys of degree 1..k//2
    for d in range(1, k // 2 + 1):
        for g in range(1 << d, 1 << (d + 1)):
            # polynomial remainder of poly by g
            r, gd = poly, g.bit_length() - 1
            while r.bit_length() - 1 >= gd and r:
                r ^= g << (r.bit_length() - 1 - gd)
            if r == 0:
                return False
    return True

def gf_mult(a, b, k, poly):
    if k == 1:
        return a & b
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a <<= 1
        if a >> k:
            a ^= poly
    return res

def build_monomials(k, poly):
    monos = []
    for q in range(k):
        for r in range(k):
            pb = gf_mult(1 << q, 1 << r, k, poly)
            for l in range(k):
                if (pb >> l) & 1:
                    monos.append(frozenset({q, k + r, 2 * k + l}))
    return monos

def residue_weight(monos):
    cnt = Counter()
    for S in monos:
        e = sorted(S)
        for size in (1, 2, 3):
            for T in combinations(e, size):
                cnt[T] += 1
    return sum(1 for v in cnt.values() if v & 1)

def polystr(p, k):
    terms = [f"X^{i}" if i > 1 else ("X" if i == 1 else "1")
             for i in range(k, -1, -1) if (p >> i) & 1]
    return "+".join(terms)

print(f"{'k':>2} {'#irred':>6} {'floor':>6} {'min W':>6} {'max W':>6}  {'all distinct W':<28} best polynomial")
for k in range(1, 8):
    if k == 1:
        print(f"{1:>2} {1:>6} {7:>6} {7:>6} {7:>6}  {'{7}':<28} X+1")
        continue
    results = []
    for p in range(1 << k, 1 << (k + 1)):
        if not (p & 1):        # X | p  => reducible
            continue
        if not is_irred(p, k):
            continue
        results.append((residue_weight(build_monomials(k, p)), p))
    results.sort()
    ws = sorted({w for w, _ in results})
    bw, bp = results[0]
    print(f"{k:>2} {len(results):>6} {6*k+1:>6} {bw:>6} {results[-1][0]:>6}  "
          f"{str(ws):<28} {polystr(bp,k)}")
