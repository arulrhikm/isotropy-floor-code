"""
Does the amortized stabilizer Renyi entropy beat the moment-radical nullity ON OUR CLASS?

The amortized SRE of Zhu-Chen-Shen-Liu-Yu-Wang (arXiv:2409.06659) is sometimes sharper than
nullity, for instance on the three-qubit quantum Fourier transform (QFT3 gets >= 6 against
nullity's 4).
But QFT3 is NOT a diagonal level-three gate, so that datum says nothing about the class the
cubic moment form is about.  This script settles the on-class question by census.

Comparison bound.  Zhu et al., Corollary 4:

        t(U)  >=  M_2^A(U) / (2 - log2 3)  >=  M_2(|Phi_U>) / (2 - log2 3),

with |Phi_U> the Choi state and 2 - log2 3 = M_2^A(T) = 0.4150375.  The right-hand
(Choi-state) form is the one that is classically evaluable from a description of the gate,
and the one their own numerics use, so it is what we compare against.  M_2 is the stabilizer
2-Renyi entropy of Leone-Oliviero-Hamma:

        M_2(psi)  =  log2 d  -  log2 sum_P <psi|P|psi>^4,     d = dim,  P over all Paulis.

Reduction.  For a diagonal gate U = diag(omega^phi(x)) the Choi state
is 2^{-n/2} sum_x omega^phi(x) |x>|x>, and a CNOT ladder (Clifford, so M_2-invariant) maps
|x>|x> -> |x>|0>.  Since M_2 is additive and vanishes on stabilizer states,

        M_2(|Phi_U>)  =  M_2(|psi_phi>),      |psi_phi> = 2^{-n/2} sum_x omega^phi(x)|x>,

an n-qubit phase state, not a 2n-qubit one.  Its Pauli spectrum is a Walsh-Hadamard transform
per shift: with g_a(x) = omega^{-(phi(x+a)-phi(x))},

        sum_P <P>^4  =  sum_{a,b} |ghat_a(b)|^4,   ghat_a(b) = 2^{-n} sum_x g_a(x)(-1)^{b.x},

so the whole 4^{2n} Pauli sum costs 2^n transforms of length 2^n.  n=6 is instant.
`--check` verifies the reduction against the direct 2n-qubit sum and reproduces the
published QFT3 value.

Compared.  On every GL(n,2)-orbit of cubic forms (the census of cubic_orbit_atlas):
    nu    = n - d          the moment-radical nullity, the incumbent adaptive floor
    floor = 2nu + 1        this paper's unitary floor (prop:nullity)
    sre   = ceil(M_2/(2-log2 3))    the incumbent SRE floor
    delta = exact T-count  (Amy-Mosca coset decoding; n <= 6)
"""
import argparse, itertools, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

M2_T = 2.0 - math.log2(3.0)          # = M_2^A(T) = 0.4150374992788438
OMEGA = np.exp(1j * np.pi / 4)


# ---------------------------------------------------------------- Pauli spectrum

def _fwht(A):
    """Unnormalised Walsh-Hadamard transform along the last axis of a 2-D complex array.

    Fully vectorised (no Python loop over butterfly blocks): at n=11 the naive version
    costs minutes per state, this costs milliseconds.
    """
    A = np.array(A, dtype=complex, copy=True)
    M, N = A.shape
    h = 1
    while h < N:
        A = A.reshape(M, N // (2 * h), 2, h)
        x = A[:, :, 0, :].copy()
        y = A[:, :, 1, :].copy()
        A[:, :, 0, :] = x + y
        A[:, :, 1, :] = x - y
        A = A.reshape(M, N)
        h <<= 1
    return A


def pauli_fourth_moment(psi, chunk=512):
    """sum_P <psi|P|psi>^4 over all 4^m Hermitian Paulis, for a normalised m-qubit state.

    <psi|X^a Z^b|psi> = sum_x conj(psi[x^a]) psi[x] (-1)^{b.x} = WHT_b of h_a(x).
    The Hermitian representative is i^{a.b} X^a Z^b and (i^{a.b})^4 = 1, so its expectation
    (which is real) has fourth power |WHT_b(h_a)|^4.  Shifts are batched in chunks so the
    working set stays at chunk*2^m complex numbers rather than 4^m.
    """
    N = psi.shape[0]
    idx = np.arange(N)
    total = 0.0
    for lo in range(0, N, chunk):
        shifts = np.arange(lo, min(lo + chunk, N))
        H = np.conj(psi[idx[None, :] ^ shifts[:, None]]) * psi[None, :]
        total += float(np.sum(np.abs(_fwht(H)) ** 4))
    return total


def sre2(psi, m):
    """Stabilizer 2-Renyi entropy M_2 of a pure state on m qubits."""
    return m - math.log2(pauli_fourth_moment(psi))


def sre2_phase_state(phase, n):
    """M_2 of 2^{-n/2} sum_x omega^{phase[x]} |x>, phase an int array mod 8."""
    psi = OMEGA ** (np.asarray(phase) % 8) / math.sqrt(1 << n)
    return sre2(psi, n)


def sre_floor(m2):
    """The Zhu et al. Corollary 4 T-count floor from a Choi-state M_2."""
    return math.ceil(m2 / M2_T - 1e-9)


# ---------------------------------------------------------------- gate phases

def phase_from_triples(triples, n):
    """phi(x) = 4 * sum over triples of x_a x_b x_c, mod 8, as an array of length 2^n."""
    N = 1 << n
    x = np.arange(N)
    bits = [(x >> i) & 1 for i in range(n)]
    phi = np.zeros(N, dtype=np.int64)
    for (a, b, c) in triples:
        phi += 4 * bits[a] * bits[b] * bits[c]
    return phi % 8


def choi_state_diagonal(phase, n):
    """The 2n-qubit Choi state 2^{-n/2} sum_x omega^phi(x)|x>|x>, for --check."""
    N = 1 << n
    psi = np.zeros(N * N, dtype=complex)
    for x in range(N):
        psi[(x << n) | x] = OMEGA ** (int(phase[x]) % 8)
    return psi / math.sqrt(N)


def choi_state_unitary(U, n):
    """|Phi_U> = (U tensor I) 2^{-n/2} sum_x |x>|x>, flattened, for arbitrary U."""
    N = 1 << n
    psi = np.zeros(N * N, dtype=complex)
    for x in range(N):
        for y in range(N):
            psi[(y << n) | x] = U[y, x]
    return psi / math.sqrt(N)


def qft(n):
    N = 1 << n
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.exp(2j * np.pi * j * k / N) / math.sqrt(N)


# ---------------------------------------------------------------- self-tests

def check():
    print("=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    ok = True

    # 1. M_2(|T>) = log2(4/3), the normalising constant itself.
    m = sre2_phase_state(np.array([0, 1]), 1)
    exp = math.log2(4 / 3)
    print(f"  M_2(|T>)           = {m:.10f}   expected log2(4/3) = {exp:.10f}")
    ok &= abs(m - exp) < 1e-9
    print(f"  2 - log2 3         = {M2_T:.10f}   (= M_2^A(T), Zhu et al. Thm 2)")

    # 1b. The WHT route against an explicit Pauli-matrix loop, on random states.
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]).astype(complex)
    rng = np.random.default_rng(0)
    worst = 0.0
    for m in (1, 2, 3):
        psi = rng.normal(size=1 << m) + 1j * rng.normal(size=1 << m)
        psi /= np.linalg.norm(psi)
        brute = 0.0
        for combo in itertools.product((I2, X, Y, Z), repeat=m):
            P = combo[0]
            for c in combo[1:]:
                P = np.kron(P, c)
            brute += float(np.real(np.conj(psi) @ P @ psi)) ** 4
        worst = max(worst, abs(brute - pauli_fourth_moment(psi)))
    print(f"  Pauli spectrum     : max |WHT route - explicit 4^m Pauli loop| = {worst:.2e} "
          f"on random states, m<=3")
    ok &= worst < 1e-9

    # 2. The Clifford reduction: M_2 of the 2n-qubit Choi state = M_2 of the n-qubit
    #    phase state.  Checked on every cubic form at n <= 3.
    worst = 0.0
    for n in (2, 3):
        mons = list(itertools.combinations(range(n), 3))
        for f in range(1 << len(mons)):
            tri = [t for k, t in enumerate(mons) if (f >> k) & 1]
            ph = phase_from_triples(tri, n)
            worst = max(worst, abs(sre2_phase_state(ph, n)
                                   - sre2(choi_state_diagonal(ph, n), 2 * n)))
    print(f"  Choi reduction     : max |M_2(Phi_U) - M_2(psi_phi)| = {worst:.2e} "
          f"over all cubics at n<=3")
    ok &= worst < 1e-9

    # 3. Reproduce the published QFT3 number.
    for n in (2, 3):
        m = sre2(choi_state_unitary(qft(n), n), 2 * n)
        tag = "   [Zhu et al. report >= 6]" if n == 3 else ""
        print(f"  QFT{n}: M_2(Phi) = {m:.6f}  ->  t >= {sre_floor(m)}{tag}")
        if n == 3:
            ok &= sre_floor(m) == 6

    # 4. Known small diagonal gates.  delta is exact by thm:rigidradius.
    cs = (2 * (np.arange(4) & 1) * ((np.arange(4) >> 1) & 1)) % 8
    cases = [("T", 1, np.array([0, 1]), 1, 1),
             ("CS", 2, cs, 2, 3),
             ("CCZ", 3, phase_from_triples([(0, 1, 2)], 3), 3, 7),
             ("CCZ+CCZ", 6, phase_from_triples([(0, 1, 2), (3, 4, 5)], 6), 6, 13)]
    print(f"  {'gate':<10s} {'M_2(Phi)':>10s} {'SRE':>5s} {'nu':>4s} {'2nu+1':>6s} {'delta':>6s}")
    for name, n, ph, nu, delta in cases:
        m = sre2_phase_state(ph, n)
        cubic = name.startswith("CCZ")
        note = "" if cubic else "   [not pure-cubic: 2nu+1 does not apply, use 2(n-d*)-r]"
        print(f"  {name:<10s} {m:10.6f} {sre_floor(m):5d} {nu:4d} {2*nu+1:6d} {delta:6d}{note}")
        ok &= sre_floor(m) <= delta
        ok &= (not cubic) or (2 * nu + 1 <= delta)

    print(f"\n  SELF-TESTS {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------- the census

def census(n, exact=True, show=1000):
    from cubic_orbit_atlas import orbits, form_triples, radical_dim
    reps, mons = orbits(n, verbose=True)

    dec = None
    if exact:
        try:
            from validate_amdist import residue_vec, rm_generators, min_coset_weight, ccz_on
            dec = (residue_vec, rm_generators(n), min_coset_weight, ccz_on)
        except Exception as e:
            print(f"   [exact delta unavailable: {e}]")

    rows = []
    for f, size in reps:
        tri = form_triples(f, mons)
        if not tri:
            continue
        d = radical_dim(tri, n)
        nu = n - d
        m2 = sre2_phase_state(phase_from_triples(tri, n), n)
        delta = None
        if dec is not None:
            residue_vec, G, min_coset_weight, ccz_on = dec
            coeff = {}
            for t in tri:
                ccz_on(coeff, t)
            delta = int(min_coset_weight(residue_vec(coeff, n), G))
        rows.append(dict(size=size, tri=tri, d=d, nu=nu, floor=2 * nu + 1,
                         m2=m2, sre=sre_floor(m2), delta=delta))

    rows.sort(key=lambda r: -r["size"])
    print(f"\n== n={n}: {len(rows)} nonzero cubic orbits ==")
    print(f"   {'orbit size':>10s} {'|f|':>4s} {'d':>2s} {'nu':>3s} {'M_2(Phi)':>9s} "
          f"{'SRE':>4s} {'2nu+1':>6s} {'delta':>6s}   representative")
    for r in rows[:show]:
        rep = "+".join("".join(str(i) for i in t) for t in r["tri"])
        dl = f"{r['delta']:6d}" if r["delta"] is not None else "     ?"
        print(f"   {r['size']:10d} {len(r['tri']):4d} {r['d']:2d} {r['nu']:3d} "
              f"{r['m2']:9.4f} {r['sre']:4d} {r['floor']:6d} {dl}   {rep[:34]}")

    tot = sum(r["size"] for r in rows)
    for key, lab in (("nu", "nu"), ("floor", "2nu+1")):
        hit = [r for r in rows if r["sre"] > r[key]]
        print(f"\n   SRE > {lab:<5s} on {len(hit):4d}/{len(rows)} orbits "
              f"= {sum(r['size'] for r in hit)}/{tot} forms "
              f"({100 * sum(r['size'] for r in hit) / tot:.1f}%)")
    if rows and rows[0]["delta"] is not None:
        viol = [r for r in rows if r["sre"] > r["delta"]]
        print(f"   soundness: SRE > delta on {len(viol)} orbits (must be 0)")
        assert not viol, viol[:3]
        print(f"   {'bound':<7s} {'exact on orbits':>16s} {'% of forms':>11s} {'mean slack':>11s}")
        for key, lab in (("sre", "SRE"), ("nu", "nu"), ("floor", "2nu+1")):
            ex = [r for r in rows if r[key] == r["delta"]]
            slack = sum(r["size"] * (r["delta"] - r[key]) for r in rows) / tot
            print(f"   {lab:<7s} {len(ex):>7d}/{len(rows):<8d} "
                  f"{100 * sum(r['size'] for r in ex) / tot:>10.1f}% {slack:>11.3f}")
    return rows


# ---------------------------------------------------------------- named families

def families(mmax=7):
    """The families where this paper's floor is a THEOREM, so delta is known for all m.

    The census only reaches n=6; the asymptotic ratio is what a resource estimator
    actually cares about, and it lives here.
        parallel Toffoli layer   m disjoint CCZs,          n=3m,   delta = 6m+1 (thm:parccz)
        fan-out layer            m CCZs on a shared CONTROL, n=2m+1, delta = 4m+3 (thm:fanout)
        ripple-carry chain       B_i = {c_{i-1}, u_i, c_i},  n=2m+1, delta = 4m+3 (prop:splicetree)
    (m CCZs sharing a whole PAIR is not a family: it collapses to CCZ, delta = 7.)
    M_2 is additive over tensor products, so the parallel family has M_2 = m*M_2(CCZ)
    exactly and the SRE floor is ceil(m * 1.540568 / 0.415037) = ceil(3.7118 m).
    """
    from cubic_orbit_atlas import radical_dim
    print("\n" + "=" * 78)
    print("NAMED FAMILIES: how the three bounds scale")
    print("=" * 78)
    NMAX = 11                       # direct M_2 costs O(4^n); beyond this use additivity

    def m2_parallel(m):
        """Exact by additivity of M_2 over tensor factors; verified directly for 3m <= NMAX."""
        v = m * sre2_phase_state(phase_from_triples([(0, 1, 2)], 3), 3)
        if 3 * m <= NMAX:
            tri = [(3 * j, 3 * j + 1, 3 * j + 2) for j in range(m)]
            direct = sre2_phase_state(phase_from_triples(tri, 3 * m), 3 * m)
            assert abs(direct - v) < 1e-9, (m, direct, v)
        return v

    specs = [("parallel Toffoli", lambda m: (3 * m, [(3 * j, 3 * j + 1, 3 * j + 2)
                                                     for j in range(m)]),
              lambda m: 6 * m + 1, m2_parallel),
             ("fan-out (shared control)",
              lambda m: (2 * m + 1, [(0, 2 * j + 1, 2 * j + 2) for j in range(m)]),
              lambda m: 4 * m + 3, None),
             ("ripple-carry chain",
              lambda m: (2 * m + 1, [(2 * j, 2 * j + 1, 2 * j + 2) for j in range(m)]),
              lambda m: 4 * m + 3, None)]
    for name, build, exact, m2fn in specs:
        print(f"\n  {name}:")
        print(f"   {'m':>2s} {'n':>3s} {'M_2(Phi)':>9s} {'SRE':>4s} {'nu':>4s} "
              f"{'2nu+1':>6s} {'delta':>6s} {'delta/SRE':>10s} {'delta/nu':>9s}")
        for m in range(1, mmax + 1):
            n, tri = build(m)
            if m2fn is None and n > NMAX:
                break
            nu = n - radical_dim(tri, n)
            m2 = m2fn(m) if m2fn else sre2_phase_state(phase_from_triples(tri, n), n)
            s, dl = sre_floor(m2), exact(m)
            print(f"   {m:2d} {n:3d} {m2:9.4f} {s:4d} {nu:4d} {2*nu+1:6d} {dl:6d} "
                  f"{dl/s:10.3f} {dl/nu:9.3f}")
            assert s <= dl, f"SRE floor {s} exceeds exact delta {dl} at m={m}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[3, 4, 5])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--families", action="store_true")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--show", type=int, default=1000)
    args = ap.parse_args()
    print(__doc__.strip())
    if args.check:
        sys.exit(0 if check() else 1)
    if args.families:
        families()
        sys.exit(0)
    for n in args.n:
        census(n, exact=not args.no_exact, show=args.show)
