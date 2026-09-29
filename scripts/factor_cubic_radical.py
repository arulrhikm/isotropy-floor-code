"""Radical of the cubic moment form for f = u_0 * qbar, and for e_3.

Radical R = {w : sum_a w_a T_{a,p,q} = 0 for all pairs p<q}.  floor = 2(n-d)+1
when d = 0 (pure cubic, non-degenerate).  We need d = 0 for the construction to
MEET the floor rather than merely bound it.
"""
from itertools import combinations
import random

def radical_dim(blocks, n):
    rows = {}
    for S in blocks:
        a_, b_, c_ = sorted(S)
        for (p, q, a) in ((a_, b_, c_), (a_, c_, b_), (b_, c_, a_)):
            rows.setdefault((p, q), 0)
            rows[(p, q)] ^= 1 << a
    mat = [b for b in rows.values() if b]
    rk = 0
    for col in range(n):
        piv = next((i for i in range(rk, len(mat)) if (mat[i] >> col) & 1), None)
        if piv is None:
            continue
        mat[rk], mat[piv] = mat[piv], mat[rk]
        for i in range(len(mat)):
            if i != rk and (mat[i] >> col) & 1:
                mat[i] ^= mat[rk]
        rk += 1
    return n - rk

def rank2(M, N):
    M = [r[:] for r in M]; rk = 0
    for c in range(N):
        piv = next((i for i in range(rk, N) if M[i][c]), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]
        for i in range(N):
            if i != rk and M[i][c]:
                M[i] = [a ^ b for a, b in zip(M[i], M[rk])]
        rk += 1
    return rk

print("=== f = u_0 * qbar, Adj = K_N (the STAR S_n) ===")
for n in range(3, 16, 2):
    N = n - 1
    blocks = [frozenset({0, j, k}) for j in range(1, n) for k in range(j + 1, n)]
    d = radical_dim(blocks, n)
    print(f"n={n:3d}  d={d}  floor=2(n-d)+1={2*(n-d)+1 if d==0 else 2*(n-d):3d}  2n+1={2*n+1:3d}")

print()
print("=== f = u_0 * qbar, random INVERTIBLE Adj ===")
rng = random.Random(11)
for n in (5, 7, 9, 11, 13):
    shown = 0
    for _ in range(400):
        N = n - 1
        Adj = [[0]*N for _ in range(N)]
        for j in range(N):
            for k in range(j+1, N):
                b = rng.randint(0,1); Adj[j][k] = Adj[k][j] = b
        if rank2(Adj, N) != N:
            continue
        blocks = [frozenset({0, j+1, k+1}) for j in range(N) for k in range(j+1, N) if Adj[j][k]]
        d = radical_dim(blocks, n)
        print(f"n={n:3d}  |blocks|={len(blocks):3d}  rank(Adj)={N}  d={d}  "
              f"floor={2*(n-d)+1 if d==0 else 2*(n-d):3d}  meets_2n+1={d==0}")
        shown += 1
        if shown == 2: break

print()
print("=== f = u_0 * qbar, random SINGULAR Adj (control: floor should drop) ===")
for n in (7, 9):
    shown = 0
    for _ in range(400):
        N = n - 1
        Adj = [[0]*N for _ in range(N)]
        for j in range(N):
            for k in range(j+1, N):
                b = rng.randint(0,1); Adj[j][k] = Adj[k][j] = b
        r = rank2(Adj, N)
        if r == N: continue
        blocks = [frozenset({0, j+1, k+1}) for j in range(N) for k in range(j+1, N) if Adj[j][k]]
        if not blocks: continue
        d = radical_dim(blocks, n)
        print(f"n={n:3d}  rank(Adj)={r}/{N}  d={d}  floor={2*(n-d)+1 if d==0 else 2*(n-d):3d}")
        shown += 1
        if shown == 2: break

print()
print("=== complete cubic e_3 ===")
print("  odd n : cor:starexact construction, weight 2n+1, meets the floor")
print("  even n: rem:e3even construction,    weight 2n-1, one ABOVE the floor 2(n-1)")
for n in range(3, 16):
    blocks = [frozenset(c) for c in combinations(range(n), 3)]
    d = radical_dim(blocks, n)
    if n % 2:                      # odd n: the factorizable-cubic representation
        par = [frozenset({i}) for i in range(n)] + \
              [frozenset(set(range(n)) - {i}) for i in range(n)] + [frozenset(range(n))]
    else:                          # even n: one column folded (rem:e3even)
        last = n - 1
        par = [frozenset({i, last}) for i in range(last)] + \
              [frozenset(set(range(n)) - {i, last}) for i in range(last)] + \
              [frozenset(range(n))]
    ok = True
    for size in (1, 2, 3):
        for A in combinations(range(n), size):
            m = sum(1 for y in par if set(A) <= y) % 2
            if m != (1 if size == 3 else 0):
                ok = False
    fl = 2 * (n - d) + 1 if d == 0 else 2 * (n - d)
    gap = len(par) - fl
    print(f"n={n:3d}  d={d}  floor={fl:3d}  construction weight={len(par):3d}  "
          f"moments_ok={ok}  above floor by {gap}")
