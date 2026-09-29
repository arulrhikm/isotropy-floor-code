r"""Verify the splice lemma and the carry-chain / fan-out constructions.

PPC set: |S| odd, order-1 and order-2 moments vanish, order-3 moments = block set.
CCZ(B) = all 7 nonempty subsets of B is PPC.

Splice: P *_y Q  =  (P \ {y})  sym-diff  (Q + y),   where (Q + y) = {s u y : s in Q}.
  disjoint case (X n Y = {}):            size |P| + |Q| - 1
  shared case  (X n Y = {c}, c not in y,
                {c} in Q, y u {c} in P):  size |P| + |Q| - 3  (one coincidence cancels)
"""
from itertools import combinations


def ccz(B):
    B = sorted(B)
    return {frozenset(c) for r in (1, 2, 3) for c in combinations(B, r)}


def lift(Q, y):
    return {frozenset(set(s) | set(y)) for s in Q}


def splice(P, Q, y):
    """Symmetric difference handles both the disjoint and the shared case."""
    return (set(P) - {frozenset(y)}) ^ lift(Q, y)


def moments(par, n):
    M = {}
    for size in (1, 2, 3):
        for A in combinations(range(n), size):
            M[A] = sum(1 for p in par if set(A) <= p) % 2
    return M


def target(blocks, n):
    T = {}
    for size in (1, 2, 3):
        for A in combinations(range(n), size):
            T[A] = 1 if (size == 3 and frozenset(A) in blocks) else 0
    return T


def check(par, blocks, n, label, floor):
    ok = moments(par, n) == target(set(blocks), n)
    print(f"{label:<34} n={n:3d}  weight={len(par):3d}  floor={floor:3d}  "
          f"meets={len(par)==floor}  moments_ok={ok}")
    return ok and len(par) == floor


print("=== ripple-carry CHAIN: B_i = {c_{i-1}, u_i, c_i}, blocks share one qubit in a path ===")
# layout: c_0..c_m at even indices, u_1..u_m at odd
allok = True
for m in range(1, 10):
    n = 2 * m + 1
    c = [2 * i for i in range(m + 1)]          # c_0 .. c_m
    u = [2 * i + 1 for i in range(m)]          # u_1 .. u_m
    blocks = [frozenset({c[i], u[i], c[i + 1]}) for i in range(m)]
    par = ccz(blocks[0])
    for i in range(1, m):
        y = frozenset(u[:i])                   # {u_1..u_i}
        assert c[i] not in y
        assert frozenset(set(y) | {c[i]}) in par, f"port pair missing at i={i}"
        assert frozenset({c[i]}) in ccz(blocks[i])
        par = splice(par, ccz(blocks[i]), y)
    allok &= check(par, blocks, n, f"carry chain m={m}", 4 * m + 3)

print()
print("=== FAN-OUT: m Toffolis through one shared control (thm:fanout, 4m+3) ===")
for m in range(1, 9):
    n = 2 * m + 1
    ctrl = 0
    a = [2 * i + 1 for i in range(m)]
    b = [2 * i + 2 for i in range(m)]
    blocks = [frozenset({ctrl, a[i], b[i]}) for i in range(m)]
    par = ccz(blocks[0])
    for i in range(1, m):
        y = frozenset(a[:i])
        assert ctrl not in y
        assert frozenset(set(y) | {ctrl}) in par, f"port pair missing at i={i}"
        par = splice(par, ccz(blocks[i]), y)
    allok &= check(par, blocks, n, f"fan-out m={m}", 4 * m + 3)

print()
print("=== DISJOINT layer (Campbell-Howard count 6m+1) ===")
for m in range(1, 8):
    n = 3 * m
    blocks = [frozenset({3 * i, 3 * i + 1, 3 * i + 2}) for i in range(m)]
    par = ccz(blocks[0])
    for i in range(1, m):
        # depth-i node on a path: accumulate EVERY ancestor port, not just the
        # immediate one (needed below depth one)
        y = frozenset(3 * j for j in range(i))
        par = splice(par, ccz(blocks[i]), y)
    allok &= check(par, blocks, n, f"disjoint chain m={m}", 6 * m + 1)

print()
print("ALL CHECKS PASS" if allok else "*** SOME CHECK FAILED ***")
