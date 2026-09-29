"""Path-sum bookkeeping for FLAT targets (LAYERED.md, Theorem B1) and the causal realizability of the
tensor-product optimum (B-d).

A unitary on n qubits is *flat* if every entry has modulus 2^{-n/2} and is an eighth root of unity
times 2^{-n/2}:  <z|U|x> = 2^{-n/2} omega^{P'(x,z)}.  Claim B1: any circuit over {H, S, CNOT, T}
(clean ancillas allowed; none used here) for a flat U has h >= n Hadamards, and with h = n exactly,

        t  >=  delta_{2n}(P')          (P' read off the matrix, level 3 on the 2n variables (x,z)).

Check B-c.  Sample random {H,S,CNOT,T} circuits with h Hadamards; keep the flat ones; read P'(x,z);
verify P' is level 3 (Z_8 ANF: coefficient of a degree-k monomial divisible by 2^{k-1}, zero for k >= 4);
rebuild a parity representation; delta_{2n}(P') by CP-SAT; check t_raw >= delta.  Also confirm that no
circuit with h < n is flat (dimension count: a column has at most 2^h nonzero entries).

Check B-d.  For U = CCZ H^{(x)3} CCZ the coset floor is delta(CCZ (x) CCZ) = 13 (thm:nonadd), attained by a
representation with parities mixing x_0 with z-parities.  In an ancilla-free h = n circuit the wire
vector spans at most n forms at any time, so a parity supported on x_S u z_T needs |S| + |T| <= n
... more precisely it must lie in the span at some step.  We minimise over the 'prefix' realizable
sets: for an H order pi and step k the span is <x_{pi(k+1..n)}, z_{pi(1..k)}> (the case where the
consumed form at step j is x_{pi(j)} and the final alignment is the identity).  A minimum of 14 over all
orders is evidence (not proof) that 13 is not attainable ancilla-free at h = n.

Run:  python flat_pathsum.py [--samples 300] [--n 3]
"""
import argparse, itertools, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_amdist as AM
import sat_tcount as ST
import nu_layers as NL

T1 = np.diag([1, NL.OMEGA]).astype(complex)


def _have_cpsat():
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        return True
    except Exception:
        return False


HAVE_CPSAT = _have_cpsat()


# ---------------------------------------------------------------- circuits

def random_circuit(n, h, rng, ngates=None):
    """Gate list with exactly h Hadamards; other gates S, T, CNOT at random."""
    ngates = ngates or (4 * n + 2 * h)
    gates = []
    for _ in range(ngates):
        k = rng.randrange(3 if n > 1 else 2)
        if k == 0: gates.append(("T", rng.randrange(n)))
        elif k == 1: gates.append(("S", rng.randrange(n)))
        else: gates.append(("CX",) + tuple(rng.sample(range(n), 2)))
    for _ in range(h):
        gates.insert(rng.randrange(len(gates) + 1), ("H", rng.randrange(n)))
    return gates


def unitary(gates, n):
    U = np.eye(1 << n, dtype=complex)
    for g in gates:
        if g[0] == "H": U = NL.op_on(NL.H1, g[1], n) @ U
        elif g[0] == "S": U = NL.op_on(NL.S1, g[1], n) @ U
        elif g[0] == "T": U = NL.op_on(T1, g[1], n) @ U
        else: U = NL.cnot(g[1], g[2], n) @ U
    return U


def is_flat(U, n, tol=1e-9):
    return bool(np.all(np.abs(np.abs(U) - 2 ** (-n / 2)) < tol))


# ---------------------------------------------------------------- P' -> parity representation

def anf_z8(P, m):
    """Integer Moebius transform of P: {0,1}^m -> Z (values in 0..7), reduced mod 8."""
    a = np.array(P, dtype=np.int64)
    for i in range(m):
        step = 1 << i
        for x in range(1 << m):
            if x & step:
                a[x] = a[x] - a[x ^ step]
    return a % 8


def coeff_from_phase(P, m):
    """Parity coefficient dict for a level-3 Z_8 function on m variables, or None if not level 3."""
    a = anf_z8(P, m)
    coeff = {}
    for S in range(1, 1 << m):
        k = bin(S).count("1")
        v = int(a[S])
        if k >= 4:
            if v: return None
            continue
        if v % (1 << (k - 1)):
            return None
        idx = [i for i in range(m) if (S >> i) & 1]
        if k == 3 and v == 4: AM.ccz_on(coeff, tuple(idx))
        elif k == 2:
            for _ in range(v // 2): AM.cs_on(coeff, tuple(idx))
        elif k == 1:
            for _ in range(v): AM.t_on(coeff, idx[0])
    if not np.array_equal(AM.phase_from_coeff(coeff, m), (np.array(P) - P[0]) % 8):
        return None
    return coeff


def phase_of_flat(U, n):
    N = 1 << n
    P = np.zeros(1 << (2 * n), dtype=int)
    for x in range(N):
        for z in range(N):
            e = U[z, x] * np.sqrt(N)
            k = int(round(np.angle(e) / (np.pi / 4))) % 8
            assert abs(e - NL.OMEGA ** k) < 1e-8, "entry is not an eighth root of unity"
            P[x | (z << n)] = k
    return P


# ---------------------------------------------------------------- B-c

def check_flat_circuits(n, samples, rng):
    flat = 0; viol = 0; gaps = []; nonlevel3 = 0
    for h in range(0, n):
        for _ in range(max(20, samples // 5)):
            assert not is_flat(unitary(random_circuit(n, h, rng), n), n), f"flat circuit with h={h}<n"
    for _ in range(samples):
        gates = random_circuit(n, n, rng)
        U = unitary(gates, n)
        if not is_flat(U, n):
            continue
        flat += 1
        P = phase_of_flat(U, n)
        coeff = coeff_from_phase(P, 2 * n)
        if coeff is None:
            nonlevel3 += 1
            continue
        t_raw = sum(1 for g in gates if g[0] == "T")
        if HAVE_CPSAT:
            delta, status, _ = ST.tcount(ST.moments_from_coeff(coeff, 2 * n), 2 * n)
            assert status == "OPTIMAL"
        else:
            delta = AM.Tcount(coeff, 2 * n)          # meet-in-the-middle; fine for 2n <= 6
        gaps.append(t_raw - delta)
        if t_raw < delta:
            viol += 1
            print(f"  VIOLATION: t_raw={t_raw} < delta_2n={delta} for {gates}")
    print(f"B-c n={n}: {samples} sampled h=n circuits, {flat} flat, {nonlevel3} with non-level-3 P' (expect 0), "
          f"{viol} violations of t >= delta_2n(P'); gap t_raw - delta: min={min(gaps) if gaps else None}, "
          f"mean={np.mean(gaps) if gaps else None:.2f}, zero-gap on {sum(1 for g in gaps if g == 0)}")
    return viol + nonlevel3


# ---------------------------------------------------------------- B-d

def restricted_tcount(moments, m, allowed):
    """Min weight in the RM(m-4,m)* coset with z_y = 0 outside `allowed` (set of parity masks)."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    z = {y: model.NewBoolVar(f"z{y}") for y in range(1, 1 << m)}
    for y in z:
        if y not in allowed:
            model.Add(z[y] == 0)
    for A, mval in moments.items():
        lits = [z[y] for y in z if (y & A) == A]
        if mval:
            model.AddBoolXOr(lits)
        else:
            model.AddBoolXOr([lits[0].Not()] + lits[1:])   # XOR(!l0, l1, ...) = 1  <=>  XOR(l0, ...) = 0
    model.Minimize(sum(z.values()))
    solver = cp_model.CpSolver(); solver.parameters.num_workers = 8
    st = solver.Solve(model)
    if st == cp_model.INFEASIBLE:
        return None
    assert st == cp_model.OPTIMAL
    return int(solver.ObjectiveValue())


def prefix_allowed(order, n):
    """Union over k of parities supported on x_{order[k:]} u z_{order[:k]}."""
    allowed = set()
    for k in range(n + 1):
        vars_ = [i for i in order[k:]] + [n + i for i in order[:k]]
        for r in range(1, len(vars_) + 1):
            for S in itertools.combinations(vars_, r):
                allowed.add(sum(1 << i for i in S))
    return allowed


def check_ccz_realizability():
    n = 3
    c = {}; AM.ccz_on(c, (0, 1, 2)); AM.ccz_on(c, (3, 4, 5))
    mom = ST.moments_from_coeff(c, 6)
    free, _, _ = ST.tcount(mom, 6)
    best = None
    for order in itertools.permutations(range(n)):
        val = restricted_tcount(mom, 6, prefix_allowed(order, n))
        best = val if best is None else min(best, val)
    # what if one x-variable could be kept on a spare wire for the whole circuit (a dirty ancilla)?
    allowed_anc = set()
    for order in itertools.permutations(range(n)):
        for keep in range(n):
            base = prefix_allowed(order, n)
            allowed_anc |= {y | (1 << keep) for y in base} | base
    anc = restricted_tcount(mom, 6, allowed_anc)
    print(f"B-d CCZ H^3 CCZ: unrestricted delta(CCZ x CCZ) = {free}; min over prefix-realizable "
          f"ancilla-free sets = {best}; with one x-variable held on a spare wire throughout = {anc}")
    return free, best, anc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bad = 0
    print(f"exact delta backend: {'CP-SAT' if HAVE_CPSAT else 'meet-in-the-middle (2n <= 6 only)'}")
    for n in range(2, min(args.n, 3 if not HAVE_CPSAT else args.n) + 1):
        bad += check_flat_circuits(n, args.samples, rng)
    if HAVE_CPSAT:
        check_ccz_realizability()
    else:
        print("B-d skipped: needs CP-SAT (run under the scratch venv)")
    print("RESULT:", "all checks passed" if bad == 0 else f"{bad} problems")


if __name__ == "__main__":
    main()
