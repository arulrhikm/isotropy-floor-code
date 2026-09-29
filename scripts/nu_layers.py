"""Unitary stabilizer nullity across a Clifford layer:  nu(D_2 C D_1)  for diagonal level-3 D_i.

CLAIM A (LAYERED.md, Theorem A).  Let C be a Clifford with symplectic matrix
    Sigma_C = [[A, B], [C', D]]        (columns: image of X_j -> (A e_j; C' e_j), Z_j -> (B e_j; D e_j))
in the (x-part; z-part) convention.  If B is invertible -- H^{(x)n}, or the QAOA mixer sqrt(X)^{(x)n}
at beta = pi/4, and up to Clifford invariance nothing else -- then

        nu(D_2 C D_1)  =  nu(D_1) + nu(D_2).

Mechanism.  D_1 X^a Z^b D_1^dag = X^a Z^b Delta_a with Delta_a a Z-diagonal Clifford, Pauli iff
a in A_1 = R*_1 (prop:nullitygen).  For a not in A_1 the symplectic image of C Delta_a C^dag has
top-right block B M_a B^T != 0, so it is not Pauli x Z-diagonal, so D_2 cannot conjugate it back to
a Pauli.  For a in A_1 the transported Pauli has x-part a' = Aa + Bb + u_a and D_2 fixes it iff
a' in A_2; b -> a' is a bijection when B is invertible.  Hence |Stab| = |A_1||A_2|.

General C.  Stab = {(a,b) : a in L,  M^{(2)}(Aa + Bb + u_a) = N_a},  L = {a : B M_a B^T = 0 = B M_a D^T},
where N_a is the lower block of C Delta_a C^dag (= D M_a D^T) and u_a its Pauli x-part.  a -> M_a is
linear (it is prop:nullitygen's contraction), so nu(D_2 C D_1) is an F_2-linear-algebra quantity.

This script brute-forces nu over all 4^n Paulis (n <= 4) and checks:
  A1  additivity for C = H^{(x)n}, C = CZ(N).CNOT(L).H^{(x)n}.CZ(N'), C = sqrt(X)^{(x)n};
      failure for C = H (x) I^{(x)(n-1)} and C = I.
  A2  for arbitrary random Cliffords: every a with a nonempty b-fibre lies in L, each fibre is a
      coset of B^{-1}(A_2), and the structural count reproduces the brute-force nu.
  L3  three and four layers saturate the cap nu <= 2n (Proposition D of LAYERED.md).
Self-tests: nu(CCZ) = 3, nu(CCZ H CCZ) = 6, nu(CS H CS) = 4.

Run:  python nu_layers.py [--trials 40] [--nmax 4]
"""
import argparse, itertools, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_amdist as AM
from experiment_mixed_nullity import build_phi, nu_of

OMEGA = np.exp(1j * np.pi / 4)
I2 = np.eye(2, dtype=complex)
X1 = np.array([[0, 1], [1, 0]], dtype=complex)
Z1 = np.diag([1, -1]).astype(complex)
H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S1 = np.diag([1, 1j]).astype(complex)
SX1 = H1 @ S1 @ H1                       # sqrt(X) up to phase: the beta = pi/4 QAOA mixer


# ---------------------------------------------------------------- matrices, qubit i = bit i

def op_on(single, i, n):
    M = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):
        M = np.kron(M, single if q == i else I2)
    return M


def pauli(a, b, n):
    """X^a Z^b with a, b bitmasks (qubit i = bit i)."""
    M = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):
        f = I2
        if (a >> q) & 1: f = f @ X1
        if (b >> q) & 1: f = f @ Z1
        M = np.kron(M, f)
    return M


def cnot(c, t, n):
    N = 1 << n
    M = np.zeros((N, N), dtype=complex)
    for x in range(N):
        y = x ^ (1 << t) if (x >> c) & 1 else x
        M[y, x] = 1
    return M


def cz(i, j, n):
    N = 1 << n
    d = np.array([-1 if ((x >> i) & 1) and ((x >> j) & 1) else 1 for x in range(N)], dtype=complex)
    return np.diag(d)


def diag_gate(phi):
    return np.diag(OMEGA ** np.asarray(phi))


def h_layer(n, wires=None):
    M = np.eye(1 << n, dtype=complex)
    for i in (range(n) if wires is None else wires):
        M = op_on(H1, i, n) @ M
    return M


def parity_vec(b, n):
    return np.array([bin(b & x).count("1") & 1 for x in range(1 << n)], dtype=bool)


# ---------------------------------------------------------------- Pauli decoding

def pauli_decode(M, n, tol=1e-7):
    """(a, b) if M is proportional to X^a Z^b, else None."""
    N = 1 << n
    idx = np.flatnonzero(np.abs(M[:, 0]) > tol)
    if len(idx) != 1:
        return None
    a = int(idx[0])
    E = M[np.arange(N) ^ a, np.arange(N)]
    if np.any(np.abs(np.abs(E) - 1) > tol):
        return None
    if abs(np.sum(np.abs(M) ** 2) - N) > tol * N:
        return None
    s = E / E[0]
    if np.any(np.abs(s.imag) > tol):
        return None
    sgn = s.real < 0
    b = sum(1 << i for i in range(n) if sgn[1 << i])
    if np.any(parity_vec(b, n) != sgn):
        return None
    return a, b


def brute_nullity(U, n, return_stab=False):
    Ud = U.conj().T
    stab = []
    for a in range(1 << n):
        for b in range(1 << n):
            if pauli_decode(U @ pauli(a, b, n) @ Ud, n) is not None:
                stab.append((a, b))
    nu = 2 * n - (len(stab).bit_length() - 1)
    assert len(stab) & (len(stab) - 1) == 0, "stabilizer count is not a power of two"
    return (nu, stab) if return_stab else nu


# ---------------------------------------------------------------- symplectic data

def symplectic(C, n):
    """Blocks (A, B, Cp, D) as n x n uint8 arrays: column j of (A;Cp) is the image of X_j, of (B;D) of Z_j."""
    A = np.zeros((n, n), np.uint8); B = A.copy(); Cp = A.copy(); D = A.copy()
    Cd = C.conj().T
    for j in range(n):
        a, b = pauli_decode(C @ pauli(1 << j, 0, n) @ Cd, n)
        for i in range(n):
            A[i, j] = (a >> i) & 1; Cp[i, j] = (b >> i) & 1
        a, b = pauli_decode(C @ pauli(0, 1 << j, n) @ Cd, n)
        for i in range(n):
            B[i, j] = (a >> i) & 1; D[i, j] = (b >> i) & 1
    return A, B, Cp, D


def f2_rank(M):
    M = (np.array(M, dtype=np.uint8) % 2).copy()
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def vec(mask, n):
    return np.array([(mask >> i) & 1 for i in range(n)], np.uint8)


def mask(v):
    return sum(int(x) << i for i, x in enumerate(v))


def zdiag_symplectic(K, n):
    """Lower block M of a Z-diagonal Clifford K: K X_j K^dag ~ X_j Z^{M e_j}.  Returns None if K is not one."""
    Kd = K.conj().T
    M = np.zeros((n, n), np.uint8)
    for j in range(n):
        dec = pauli_decode(K @ pauli(1 << j, 0, n) @ Kd, n)
        if dec is None or dec[0] != (1 << j):
            return None
        for i in range(n):
            M[i, j] = (dec[1] >> i) & 1
    return M


def delta_matrix(Dg, a, n):
    """Delta_a with D X^a D^dag = X^a Delta_a, i.e. Delta_a = X^a D X^a D^dag (diagonal)."""
    Xa = pauli(a, 0, n)
    return Xa @ Dg @ Xa @ Dg.conj().T


def structural_count(D1, D2, C, n):
    """Count |Stab(D_2 C D_1)| through the mechanism of Claim A (general C), and return diagnostics.
    Returns (count, checks) where checks records the structural assertions of check A2."""
    A, B, Cp, D = symplectic(C, n)
    Cd = C.conj().T
    M1 = {a: zdiag_symplectic(delta_matrix(D1, a, n), n) for a in range(1 << n)}
    M2 = {a: zdiag_symplectic(delta_matrix(D2, a, n), n) for a in range(1 << n)}
    assert all(m is not None for m in M1.values()) and all(m is not None for m in M2.values())
    A1 = {a for a, m in M1.items() if not m.any()}
    A2 = {a for a, m in M2.items() if not m.any()}
    L = {a for a, m in M1.items()
         if not ((B @ m @ B.T) % 2).any() and not ((B @ m @ D.T) % 2).any()}
    count = 0
    fibres = {}
    for a in range(1 << n):
        if a not in L:
            continue
        Y = C @ delta_matrix(D1, a, n) @ Cd
        idx = np.flatnonzero(np.abs(Y[:, 0]) > 1e-7)
        assert len(idx) == 1
        u = int(idx[0])
        K = pauli(u, 0, n) @ Y                      # should be Z-diagonal Clifford
        N = zdiag_symplectic(K, n)
        assert N is not None, "C Delta_a C^dag is not Pauli x Z-diagonal although a in L"
        assert np.array_equal(N, (D @ M1[a] @ D.T) % 2), "N_a != D M_a D^T"
        good_b = []
        for b in range(1 << n):
            ap = mask((A @ vec(a, n) + B @ vec(b, n) + vec(u, n)) % 2)
            if np.array_equal(M2[ap], N):
                good_b.append(b)
        if good_b:
            fibres[a] = good_b
            count += len(good_b)
    # structural checks
    Binv_A2 = {b for b in range(1 << n) if mask((B @ vec(b, n)) % 2) in A2}
    coset_ok = all(all(((b ^ bs[0]) in Binv_A2) for b in bs) and len(bs) == len(Binv_A2)
                   for bs in fibres.values())
    dim_ok = (count.bit_length() - 1) == (len(fibres).bit_length() - 1) + (len(Binv_A2).bit_length() - 1)
    B_invertible = f2_rank(B) == n
    checks = dict(A1_in_L=A1 <= L, fibres_cosets=coset_ok, dims_add=dim_ok,
                  B_invertible=B_invertible, L_equals_A1=(L == A1))
    return count, checks


# ---------------------------------------------------------------- random objects

def random_gate(n, rng):
    T3 = list(itertools.combinations(range(n), 3)); T2 = list(itertools.combinations(range(n), 2))
    while True:
        triples = rng.sample(T3, rng.randint(0, len(T3))) if T3 else []
        pairs = rng.sample(T2, rng.randint(0, len(T2))) if T2 else []
        lin = [i for i in range(n) if rng.random() < 0.5]
        if triples or pairs or lin:
            coeff, phi = build_phi(triples, pairs, lin, n)
            return coeff, phi, diag_gate(phi)


def random_clifford(n, rng, length=None):
    length = length or 4 * n * n
    M = np.eye(1 << n, dtype=complex)
    for _ in range(length):
        k = rng.randrange(3 if n > 1 else 2)
        if k == 0: M = op_on(H1, rng.randrange(n), n) @ M
        elif k == 1: M = op_on(S1, rng.randrange(n), n) @ M
        else:
            c, t = rng.sample(range(n), 2); M = cnot(c, t, n) @ M
    return M


def random_zdiag_clifford(n, rng):
    M = np.eye(1 << n, dtype=complex)
    for i in range(n):
        if rng.random() < 0.5: M = op_on(S1, i, n) @ M
    for i, j in itertools.combinations(range(n), 2):
        if rng.random() < 0.5: M = cz(i, j, n) @ M
    return M


def random_cnot_circuit(n, rng):
    M = np.eye(1 << n, dtype=complex)
    for _ in range(2 * n):
        if n > 1:
            c, t = rng.sample(range(n), 2); M = cnot(c, t, n) @ M
    return M


# ---------------------------------------------------------------- checks

def self_tests():
    n = 3
    ccz, _ = build_phi([(0, 1, 2)], [], [], n)
    Dccz = diag_gate(AM.phase_from_coeff(ccz, n))
    cs, _ = build_phi([], [(0, 1)], [], 2)
    Dcs = diag_gate(AM.phase_from_coeff(cs, 2))
    r = dict(nu_CCZ=brute_nullity(Dccz, 3), nu_CCZ_H_CCZ=brute_nullity(Dccz @ h_layer(3) @ Dccz, 3),
             nu_CS=brute_nullity(Dcs, 2), nu_CS_H_CS=brute_nullity(Dcs @ h_layer(2) @ Dcs, 2),
             nu_CS_CS=brute_nullity(Dcs @ Dcs, 2))
    ok = r == dict(nu_CCZ=3, nu_CCZ_H_CCZ=6, nu_CS=2, nu_CS_H_CS=4, nu_CS_CS=0)
    print("self-tests:", r, "OK" if ok else "FAIL")
    return ok


def check_A1(ns, trials, rng):
    """Additivity for B-invertible C; failure for partial H and identity."""
    bad = 0
    for n in ns:
        stats = {}
        for _ in range(trials):
            c1, p1, D1 = random_gate(n, rng); c2, p2, D2 = random_gate(n, rng)
            n1, _ = nu_of(p1, n); n2, _ = nu_of(p2, n)
            cases = {
                "H^n": h_layer(n),
                "CZ.CNOT.H^n.CZ": random_zdiag_clifford(n, rng) @ random_cnot_circuit(n, rng) @ h_layer(n)
                                  @ random_zdiag_clifford(n, rng),
                "sqrtX^n": np.linalg.multi_dot([op_on(SX1, i, n) for i in range(n)]) if n > 1 else SX1,
                "H(x)I": h_layer(n, wires=[0]),
                "I": np.eye(1 << n, dtype=complex),
            }
            for name, C in cases.items():
                nu = brute_nullity(D2 @ C @ D1, n)
                add = (nu == n1 + n2)
                s = stats.setdefault(name, [0, 0]); s[0] += add; s[1] += 1
                if name in ("H^n", "CZ.CNOT.H^n.CZ", "sqrtX^n") and not add:
                    bad += 1
                    print(f"  VIOLATION n={n} C={name}: nu={nu} nu1={n1} nu2={n2}")
        print(f"A1 n={n}: additive fraction per layer type: " +
              ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in stats.items()))
    return bad


def check_A2(ns, trials, rng):
    """General C: structural count == brute force, and the fibre / dimension structure."""
    bad = 0
    for n in ns:
        agree = 0; inv = 0; addv = 0
        for _ in range(trials):
            c1, p1, D1 = random_gate(n, rng); c2, p2, D2 = random_gate(n, rng)
            C = random_clifford(n, rng)
            nu_b, stab = brute_nullity(D2 @ C @ D1, n, return_stab=True)
            cnt, chk = structural_count(D1, D2, C, n)
            ok = (cnt == len(stab)) and all(chk[k] for k in ("A1_in_L", "fibres_cosets", "dims_add"))
            agree += ok
            if not ok:
                bad += 1
                print(f"  A2 MISMATCH n={n}: brute |Stab|={len(stab)} structural={cnt} checks={chk}")
            n1, _ = nu_of(p1, n); n2, _ = nu_of(p2, n)
            if chk["B_invertible"]:
                inv += 1
                if nu_b == n1 + n2: addv += 1
                else:
                    bad += 1; print(f"  A2 VIOLATION: B invertible but nu={nu_b} != {n1}+{n2}")
        print(f"A2 n={n}: structural count agrees {agree}/{trials}; B invertible in {inv} cases, additive in {addv}")
    return bad


def check_layers(ns, trials, rng):
    for n in ns:
        rows = []
        for _ in range(trials):
            gates = [random_gate(n, rng) for _ in range(4)]
            nus = [nu_of(g[1], n)[0] for g in gates]
            U = gates[0][2]
            out = []
            for L in range(1, 4):
                U = gates[L][2] @ h_layer(n) @ U
                out.append(brute_nullity(U, n))
            rows.append((nus, out))
        cap_hits = sum(1 for nus, out in rows if out[1] == 2 * n)
        two_ok = sum(1 for nus, out in rows if out[0] == nus[0] + nus[1])
        assert all(o <= 2 * n for _, out in rows for o in out)
        print(f"L3 n={n}: two-layer additive {two_ok}/{trials}; three layers hit the cap 2n={2*n} in "
              f"{cap_hits}/{trials}; sample rows (nu_i | nu after 2,3,4 layers): "
              + "; ".join(f"{nus}|{out}" for nus, out in rows[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ok = self_tests()
    ns = list(range(2, args.nmax + 1))
    bad = check_A1(ns, args.trials, rng)
    bad += check_A2(ns, max(6, args.trials // 3), rng)
    check_layers(ns, max(6, args.trials // 3), rng)
    print("RESULT:", "all checks passed" if (ok and bad == 0) else f"{bad} violations")


if __name__ == "__main__":
    main()
