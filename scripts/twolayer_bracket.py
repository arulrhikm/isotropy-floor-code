"""Two-layer floor and bracket for the flat unitary U = D_2 H^{(x)n} D_1  (LAYERED.md, Theorem B / Cor. C).

<z|U|x> = 2^{-n/2} omega^{P(x,z)},   P(x,z) = phi_1(x) + phi_2(z) + 4 x.z  (mod 8),
a level-3 phase polynomial on 2n variables (x = bits 0..n-1, z = bits n..2n-1).  The Hadamard term
4 x_i z_i = 2 chi_i + 2 chi_{n+i} - 2 chi_{i,n+i} is even, so Res P = Res phi_1 (+) Res phi_2 and the
residue coset of P is that of the 2n-qubit gate D_1 (x) D_2:

        delta_{2n}(P) = delta(D_1 (x) D_2).

Checks (all exact, n <= 4 so 2n <= 8 for sat_tcount):
  B-a  the phase read off the matrix U equals P built from the coefficient dict (sign conventions);
       floor_2n <= delta_{2n}(P) <= delta_1 + delta_2 <= W_1 + W_2, where floor_2n is thm:genLB on 2n
       variables and equals floor_1 + floor_2 (+1 once when both are pure cubic and non-degenerate);
       strictness delta_{2n}(P) < delta_1 + delta_2 on CCZ|CCZ (13 < 14, thm:nonadd).
  B-b  nu(U) (brute force, nu_layers) = nu_1 + nu_2 and floor_2n >= 2 nu(U) on pure-cubic pairs.
  D    D-local layers: certified ratio (W_1 + W_2)/floor_2n = O(D), 7/6 at D = 1, for n up to 15.

Run:  python twolayer_bracket.py [--samples 8] [--nmax 4]
"""
import argparse, itertools, os, random, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_amdist as AM
import sat_tcount as ST
import gf2m_floor as GF
from sharp_general_bound import genlb_bound
from experiment_mixed_nullity import build_phi, nu_of
from sparse_bracket_ratio import w_of_F
import nu_layers as NL


# ---------------------------------------------------------------- the 2n-variable polynomial

def twolayer_coeff(c1, c2, n):
    """Coefficient dict of P(x,z) on 2n variables from the layer dicts (parity bitmask -> Z coefficient)."""
    c = {}
    for y, v in c1.items():
        AM.add_term(c, y, v)
    for y, v in c2.items():
        AM.add_term(c, y << n, v)
    for i in range(n):                      # 4 x_i z_i
        AM.add_term(c, 1 << i, 2); AM.add_term(c, 1 << (n + i), 2); AM.add_term(c, (1 << i) | (1 << (n + i)), -2)
    return c


def phase_from_matrix(U, n):
    """P(x,z) mod 8 read off U's entries, indexed v = x | (z << n).  Asserts flatness."""
    N = 1 << n
    P = np.zeros(1 << (2 * n), dtype=int)
    for x in range(N):
        for z in range(N):
            e = U[z, x] * np.sqrt(N)
            assert abs(abs(e) - 1) < 1e-9, "not flat"
            k = int(round(np.angle(e) / (np.pi / 4))) % 8
            assert abs(e - NL.OMEGA ** k) < 1e-9
            P[x | (z << n)] = k
    return P


def floor_of(coeff, n):
    """thm:genLB floor 2(n - d*) - r, plus (d*, r)."""
    _, _, dp, r, _ = genlb_bound(coeff, n)
    return 2 * (n - dp) - r, dp, r


def _have_cpsat():
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        return True
    except Exception:
        return False


HAVE_CPSAT = _have_cpsat()


def delta_exact(coeff, n):
    """Exact delta: CP-SAT when a compatible ortools is importable, else the meet-in-the-middle
    coset decoder (fast to n = 6, i.e. layers of n <= 3 qubits).  Run under the scratch venv for n = 4."""
    if HAVE_CPSAT:
        mom = ST.moments_from_coeff(coeff, n)
        val, status, _ = ST.tcount(mom, n)
        assert status == "OPTIMAL", status
        return val
    assert n <= 6, "2n > 6 needs the CP-SAT backend (ortools/protobuf conflict in this interpreter)"
    return AM.Tcount(coeff, n)


def ceiling_of(coeff, n):
    return ST.weight_of_moment_free_rep(ST.moments_from_coeff(coeff, n), n)[0]


def is_pure_cubic(coeff, n):
    mom = ST.moments_from_coeff(coeff, n)
    return all(v == 0 for A, v in mom.items() if bin(A).count("1") <= 2)


# ---------------------------------------------------------------- B-a / B-b

def analyse_pair(name, c1, c2, n, brute_nu=True):
    p1 = AM.phase_from_coeff(c1, n); p2 = AM.phase_from_coeff(c2, n)
    D1 = NL.diag_gate(p1); D2 = NL.diag_gate(p2)
    U = D2 @ NL.h_layer(n) @ D1
    c = twolayer_coeff(c1, c2, n)
    Pmat = phase_from_matrix(U, n)
    Pdict = AM.phase_from_coeff(c, 2 * n)
    assert np.array_equal(Pmat, Pdict), f"{name}: P(x,z) formula mismatch"
    f1, d1, r1 = floor_of(c1, n); f2, d2, r2 = floor_of(c2, n); f12, d12, r12 = floor_of(c, 2 * n)
    pc = is_pure_cubic(c1, n) and is_pure_cubic(c2, n)
    plus = 1 if (pc and d1 == 0 and d2 == 0) else 0
    dl1 = delta_exact(c1, n); dl2 = delta_exact(c2, n); dl12 = delta_exact(c, 2 * n)
    W1 = ceiling_of(c1, n); W2 = ceiling_of(c2, n); W12 = ceiling_of(c, 2 * n)
    nu1, _ = nu_of(p1, n); nu2, _ = nu_of(p2, n)
    nuU = NL.brute_nullity(U, n) if brute_nu else None
    row = dict(name=name, n=n, floor1=f1, floor2=f2, floor2n=f12, plus=plus, delta1=dl1, delta2=dl2,
               delta2n=dl12, W1=W1, W2=W2, W2n=W12, nu1=nu1, nu2=nu2, nuU=nuU, pure_cubic=pc,
               d12=d12, r12=r12)
    ok = True
    ok &= (f12 == f1 + f2)                       # block-diagonal moments: floors add (without the +1)
    ok &= (f12 + plus <= dl12 <= dl1 + dl2 <= W1 + W2)
    ok &= (W12 <= W1 + W2)
    ok &= (nuU is None or nuU == nu1 + nu2)
    if pc:
        ok &= (f12 + plus >= 2 * (nuU if nuU is not None else nu1 + nu2) + plus)
    row["ok"] = ok
    return row


def fmt(row):
    return (f"{row['name']:<18} n={row['n']}  floor {row['floor1']}+{row['floor2']}={row['floor2n']}"
            f"(+{row['plus']})  delta {row['delta1']}+{row['delta2']}={row['delta1']+row['delta2']} vs "
            f"delta_2n={row['delta2n']}  W {row['W1']}+{row['W2']} (W_2n={row['W2n']})  "
            f"nu {row['nu1']}+{row['nu2']} vs nu(U)={row['nuU']}  {'ok' if row['ok'] else 'FAIL'}")


def named_cases():
    out = []
    c, _ = build_phi([], [], [0], 1); out.append(("T|T", c, dict(c), 1))
    c, _ = build_phi([], [(0, 1)], [], 2); out.append(("CS|CS", c, dict(c), 2))
    c, _ = build_phi([], [], [0, 1], 2); out.append(("TT|TT", c, dict(c), 2))
    c, _ = build_phi([(0, 1, 2)], [], [], 3); out.append(("CCZ|CCZ", c, dict(c), 3))
    c2, _ = build_phi([], [(0, 1)], [2], 3); out.append(("CCZ|CS.T", c, c2, 3))
    c, _ = build_phi([(0, 1, 2), (1, 2, 3)], [], [], 4); out.append(("line4|line4", c, dict(c), 4))
    c, _ = build_phi([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], [], [], 4); out.append(("e3(4)|e3(4)", c, dict(c), 4))
    # a 3-local Ising cost layer with fields and couplings, twice with different terms
    c, _ = build_phi([(0, 1, 2)], [(2, 3)], [0, 3], 4); c2, _ = build_phi([(1, 2, 3)], [(0, 1)], [2], 4)
    out.append(("ising4|ising4'", c, c2, 4))
    return out


def random_pairs(ns, samples, rng):
    for n in ns:
        for k in range(samples):
            c1, _, _ = NL.random_gate(n, rng); c2, _, _ = NL.random_gate(n, rng)
            yield (f"rand{n}#{k}", c1, c2, n)


# ---------------------------------------------------------------- D: the D-local ratio

def random_dlocal_hypergraph(n, D, rng, tries=200):
    """Random set of triples with every vertex in at most D triples, greedy."""
    deg = Counter(); F = set()
    trip = list(itertools.combinations(range(n), 3)); rng.shuffle(trip)
    for S in trip:
        if all(deg[v] < D for v in S) and rng.random() < 0.8:
            F.add(S)
            for v in S: deg[v] += 1
    return F


def dlocal_sweep(ns, Ds, per_cell, rng):
    print("D-local pure-cubic layers: max certified ratio (W_1+W_2)/(floor_1+floor_2+plus) over samples")
    for D in Ds:
        line = []
        for n in ns:
            worst = 0.0; nondeg = 0; tot = 0
            for _ in range(per_cell):
                F1 = random_dlocal_hypergraph(n, D, rng); F2 = random_dlocal_hypergraph(n, D, rng)
                if not F1 or not F2: continue
                d1 = GF.radical_dim(F1, n); d2 = GF.radical_dim(F2, n)
                plus = 1 if d1 == 0 and d2 == 0 else 0
                fl = 2 * (n - d1) + 2 * (n - d2) + plus
                if fl == 0: continue
                ce = w_of_F(F1) + w_of_F(F2)
                worst = max(worst, ce / fl); tot += 1; nondeg += plus
            line.append(f"n={n}: {worst:.3f} ({nondeg}/{tot} nondeg)")
        print(f"  D={D}: " + " | ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bad = 0
    nmax = args.nmax if HAVE_CPSAT else min(args.nmax, 3)
    print(f"exact delta backend: {'CP-SAT' if HAVE_CPSAT else 'meet-in-the-middle (n <= 3 only)'}; nmax={nmax}")
    print("named cases")
    rows = []
    for name, c1, c2, n in named_cases():
        if n > nmax:
            continue
        row = analyse_pair(name, c1, c2, n); rows.append(row); print(" ", fmt(row)); bad += not row["ok"]
    ccz = next(r for r in rows if r["name"] == "CCZ|CCZ")
    print(f"  CCZ|CCZ strictness: delta_2n={ccz['delta2n']} < delta_1+delta_2={ccz['delta1']+ccz['delta2']}:",
          ccz["delta2n"] < ccz["delta1"] + ccz["delta2"])
    print("random mixed-degree pairs")
    strict = 0; tot = 0
    for name, c1, c2, n in random_pairs(range(1, nmax + 1), args.samples, rng):
        row = analyse_pair(name, c1, c2, n); print(" ", fmt(row)); bad += not row["ok"]
        tot += 1; strict += row["delta2n"] < row["delta1"] + row["delta2"]
    print(f"  delta_2n < delta_1 + delta_2 on {strict}/{tot} random pairs")
    dlocal_sweep(ns=(9, 12, 15), Ds=(1, 2, 3, 4, 6), per_cell=25, rng=rng)
    print("RESULT:", "all checks passed" if bad == 0 else f"{bad} failures")


if __name__ == "__main__":
    main()
