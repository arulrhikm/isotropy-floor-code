"""Verify the factorizable-cubic construction: f = u_0 * qbar(u_1..u_N).

Blocks are {0,j,k} for each edge {j,k} of qbar.  Claim: if the adjacency Adj of
qbar is invertible over F2 (forcing N even, n = N+1 odd), then delta(f) = 2n+1,
realised by the parity set

    r' = {{0}}  u  {R_r u {0} : r}  u  {R_r : r}

where R_r is the r-th ROW of any (N+1) x N matrix V over F2 whose columns c_j
have even weight and satisfy Gram(c_j, c_k) = Adj[j][k] for j != k.
"""
from itertools import combinations
import random

def moments(parities, n):
    """M_A for all 1 <= |A| <= 3, as a dict."""
    M = {}
    for size in (1, 2, 3):
        for A in combinations(range(n), size):
            M[A] = sum(1 for y in parities if set(A) <= y) % 2
    return M

def target(N, Adj):
    """Syndrome of f = u_0 * qbar: M_{0,j,k} = Adj[j][k], all else 0."""
    n = N + 1
    T = {}
    for size in (1, 2, 3):
        for A in combinations(range(n), size):
            T[A] = 0
    for j in range(1, n):
        for k in range(j + 1, n):
            if Adj[j - 1][k - 1]:
                T[(0, j, k)] = 1
    return T

def build(V, N):
    """r' from the row sets of V ((N+1) x N over F2), columns indexed 1..N."""
    rows = []
    for r in range(N + 1):
        R = frozenset(j + 1 for j in range(N) if V[r][j])
        rows.append(R)
    if any(len(R) == 0 for R in rows):
        return None, "empty row"
    if len(set(rows)) != len(rows):
        return None, "duplicate rows"
    par = [frozenset({0})]
    par += [R | {0} for R in rows]
    par += [set(R) for R in rows]
    par = [frozenset(p) for p in par]
    if len(set(par)) != len(par):
        return None, "duplicate parities"
    return par, None

def gram(V, N):
    """Gram matrix of the columns of V over F2."""
    G = [[0] * N for _ in range(N)]
    for j in range(N):
        for k in range(N):
            G[j][k] = sum(V[r][j] * V[r][k] for r in range(N + 1)) % 2
    return G

def rank2(M, rows, cols):
    M = [row[:] for row in M]
    rk = 0
    for c in range(cols):
        piv = next((i for i in range(rk, rows) if M[i][c]), None)
        if piv is None:
            continue
        M[rk], M[piv] = M[piv], M[rk]
        for i in range(rows):
            if i != rk and M[i][c]:
                M[i] = [a ^ b for a, b in zip(M[i], M[rk])]
        rk += 1
    return rk

# ---- the canonical V for the STAR (Adj = complete graph K_N) ----
def star_V(N):
    V = [[0] * N for _ in range(N + 1)]
    for j in range(N):
        V[j][j] = 1
        V[N][j] = 1
    return V

# ---- search a V realising a given Adj, by randomised solve over even-weight cols ----
def find_V(N, Adj, tries=200000):
    rng = random.Random(0xC0FFEE)
    evens = [v for v in range(1 << (N + 1)) if bin(v).count("1") % 2 == 0 and v != 0]
    for _ in range(tries):
        cols = []
        ok = True
        for j in range(N):
            cand = [c for c in evens
                    if all((bin(c & cols[k]).count("1") % 2) == Adj[j][k]
                           for k in range(j))]
            if not cand:
                ok = False
                break
            cols.append(rng.choice(cand))
        if not ok:
            continue
        V = [[(cols[j] >> r) & 1 for j in range(N)] for r in range(N + 1)]
        return V
    return None

print("=== STAR: Adj = K_N (complete graph), n = N+1 odd ===")
for n in range(3, 20, 2):
    N = n - 1
    Adj = [[1 if j != k else 0 for k in range(N)] for j in range(N)]
    V = star_V(N)
    G = gram(V, N)
    offdiag_ok = all(G[j][k] == Adj[j][k] for j in range(N) for k in range(N) if j != k)
    par, err = build(V, N)
    if par is None:
        print(f"n={n:3d}  BUILD FAILED: {err}")
        continue
    M, T = moments(par, n), target(N, Adj)
    print(f"n={n:3d}  N={N:3d}  rank(Adj)={rank2(Adj,N,N):3d}/{N}  gram_ok={offdiag_ok}  "
          f"weight={len(par):3d}  floor=2n+1={2*n+1:3d}  moments_ok={M==T}")

print()
print("=== GENERAL: random invertible Adj (alternating, zero diagonal) ===")
rng = random.Random(7)
for n in (5, 7, 9, 11, 13):
    N = n - 1
    found = 0
    for attempt in range(60):
        Adj = [[0] * N for _ in range(N)]
        for j in range(N):
            for k in range(j + 1, N):
                b = rng.randint(0, 1)
                Adj[j][k] = Adj[k][j] = b
        if rank2(Adj, N, N) != N:
            continue
        V = find_V(N, Adj)
        if V is None:
            print(f"n={n}  no V found for an invertible Adj  <-- would refute")
            continue
        G = gram(V, N)
        offdiag_ok = all(G[j][k] == Adj[j][k] for j in range(N) for k in range(N) if j != k)
        par, err = build(V, N)
        if par is None:
            print(f"n={n}  build failed: {err}")
            continue
        M, T = moments(par, n), target(N, Adj)
        print(f"n={n:3d}  invertible Adj  gram_ok={offdiag_ok}  weight={len(par):3d}  "
              f"floor={2*n+1:3d}  meets={len(par)==2*n+1}  moments_ok={M==T}")
        found += 1
        if found == 2:
            break
