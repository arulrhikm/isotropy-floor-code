"""CS-floor sanity checks (sec:csfloor).
(1) The CS carry identity: 2(u.x)(v.x) = (u.x)+(v.x)-((u^v).x) as Z8 phases.
(2) The seven Fano lines {u,v,u^v} of PG(2,2) XOR to the full 7-point CCZ residue.
(3) delta_CS(CCZ) >= ceil(7/3) = 3, consistent with the known optimal 3.
"""
import itertools
import numpy as np

# (1) carry identity over all x in F2^2-like parity values
ok = True
for n in (2, 3, 4):
    for u in range(1, 1 << n):
        for v in range(1, 1 << n):
            if u == v:
                continue
            for x in range(1 << n):
                px = bin(u & x).count("1") & 1
                qx = bin(v & x).count("1") & 1
                rx = bin((u ^ v) & x).count("1") & 1
                ok &= (2 * px * qx - (px + qx - rx)) % 8 == 0
print("(1) carry identity 2(u.x)(v.x) = u.x + v.x - (u^v).x mod 8:", ok)
assert ok

# (2) Fano lines XOR
pts = list(range(1, 8))  # nonzero vectors of F2^3
lines = set()
for u, v in itertools.combinations(pts, 2):
    lines.add(frozenset({u, v, u ^ v}))
print("(2) #lines of PG(2,2):", len(lines), "(expect 7)")
acc = np.zeros(8, dtype=int)
for L in lines:
    for p in L:
        acc[p] ^= 1
full = all(acc[p] == 1 for p in pts) and acc[0] == 0
print("    XOR of all 7 lines = full 7-point set:", full)
assert len(lines) == 7 and full

# (3) floor arithmetic
import math
print("(3) delta_CS(CCZ) >= ceil(7/3) =", math.ceil(7 / 3), "(known optimal: 3)")
print("    parallel-layer bracket: [2m+1, 3m] for m disjoint CCZ")
print("ALL PASS")
