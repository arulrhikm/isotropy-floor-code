"""Single-qubit exact T-count versus unitary stabilizer nullity: the depth cap (LAYERED.md, Prop. D).

Matsumoto--Amano (arXiv:0806.3834): every single-qubit Clifford+T operator has a UNIQUE normal form
        (T | eps) (H T | S H T)^*  C,      C in the 24-element Clifford group (mod phase),
and its T-count is the number of T's in the normal form (Giles--Selinger arXiv:1312.6584 Lemma 4.10;
Gosset--Kliuchnikov--Mosca--Russell arXiv:1308.4134 Thm 1: T-count = sde of the Bloch representation).

Hence U_L = T (H T)^L is literally in normal form with L+1 T's:  t(U_L) = L+1.  Meanwhile the unitary
stabilizer nullity of ANY single-qubit unitary is at most 2, so the ratio t/nu is >= (L+1)/2 -- unbounded
in the depth L.  This script certifies both facts numerically without appealing to uniqueness:

  (1) BFS over all Clifford+T single-qubit unitaries with T-count <= K (as canonicalised float
      matrices mod global phase) gives the exact minimum T-count of U_L for L+1 <= K, and it is L+1.
  (2) nu(U_L) computed over the four Paulis is <= 2 (and equals 2 for L >= 1).

Run:  python ma_tcount_n1.py [--K 9]
"""
import argparse, itertools
import numpy as np

w = np.exp(1j * np.pi / 4)
H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)
T = np.diag([1, w]).astype(complex)
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
I = np.eye(2, dtype=complex)


def canon(U):
    """Key for U modulo global phase: divide by the phase of the entry of largest modulus (ties: first)."""
    flat = U.reshape(-1)
    k = int(np.argmax(np.round(np.abs(flat), 6)))
    V = U / (flat[k] / abs(flat[k]))
    return tuple(np.round(V.reshape(-1), 6).tolist())


def clifford_group():
    """All 24 single-qubit Cliffords mod phase, by closure of {H, S}."""
    seen = {canon(I): I}
    frontier = [I]
    while frontier:
        nxt = []
        for U in frontier:
            for G in (H, S):
                V = G @ U
                k = canon(V)
                if k not in seen:
                    seen[k] = V; nxt.append(V)
        frontier = nxt
    assert len(seen) == 24, len(seen)
    return list(seen.values())


def tcount_table(K):
    """dict canon(U) -> minimal T-count, for all Clifford+T single-qubit U with T-count <= K."""
    C24 = clifford_group()
    table = {}
    layer = {}
    for C in C24:
        table.setdefault(canon(C), 0); layer[canon(C)] = C
    for k in range(1, K + 1):
        new = {}
        for key, U in layer.items():
            for C in C24:
                V = T @ C @ U          # append a Clifford then a T; any word = product of such syllables
                kv = canon(V)
                if kv not in table and kv not in new:
                    new[kv] = V
        # close under left Cliffords (T-count unchanged)
        closed = {}
        for key, U in new.items():
            for C in C24:
                V = C @ U; kv = canon(V)
                if kv not in table and kv not in closed:
                    closed[kv] = V
        for kv, V in closed.items():
            table[kv] = k
        layer = closed
        print(f"  T-count {k}: {len(closed)} new unitaries (cumulative {len(table)})")
    return table


def nullity1(U):
    Ud = U.conj().T
    cnt = 0
    for P in (I, X, X @ Z, Z):
        M = U @ P @ Ud
        for Q in (I, X, X @ Z, Z):
            if abs(abs(np.trace(Q.conj().T @ M)) - 2) < 1e-8:
                cnt += 1; break
    return 2 - int(np.log2(cnt))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=9)
    args = ap.parse_args()
    print("building the T-count table by BFS")
    table = tcount_table(args.K)
    ok = True
    print(" L  t(T (HT)^L)  expected  nu")
    for L in range(0, args.K):
        U = T.copy()
        for _ in range(L):
            U = T @ H @ U          # U_L = T (H T)^L ... built as (T H)^L T
        t = table.get(canon(U))
        nu = nullity1(U)
        flag = "" if t == L + 1 else "  <-- MISMATCH"
        ok &= (t == L + 1) and nu <= 2
        print(f"{L:2d}  {t!s:>11}  {L+1:8d}  {nu}{flag}")
    print("RESULT:", "t(T(HT)^L) = L+1 and nu <= 2 for all L" if ok else "FAIL")


if __name__ == "__main__":
    main()
