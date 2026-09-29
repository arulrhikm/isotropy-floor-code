"""
Width-2 (and width-3) CIRCUIT-LEVEL stress test of the kernel inequality
    N_ker >= delta(kappa)
at width two and three (the open case beyond conj:w1), extending validate_w1.py.

Unlike validate_w1.py (which samples the DERIVED width-1 chain family), this script
samples measurement-free {H,CNOT,T(,S,Z,X)} CIRCUITS gate by gate:
  - wire values are affine forms over (x; y_1..y_h; 1),
  - CNOT = row op, phase gate T^c on a wire adds c*(wire value) to P,
  - H_j consumes the live value f_j into the coupling 4*f_j*y_j and emits y_j
    (strict causality holds by construction),
  - a terminal CNOT layer is solved for so that G(x,y)=(x,0^a) is satisfiable for
    every x (clean diagonal return); circuits where no such layer exists are discarded.
Then, exactly:
  - V = ker M_G, d = dim V, affine section z_0(x); fiber Gauss sum checked for
    diagonality (|Z| = 2^{h/2} for all x, phase an 8th root)  ->  kappa(x);
  - N_ker = number of merged odd parities whose y-part meets V;
  - delta(kappa) by the exact Amy-Mosca coset decoder (shared with validate_w1);
  - frontier width w = max_t dim(restriction of live y-span to V)  [live window];
  - removal filtration: dead subspaces D_1 c D_2 c ... c D_d = V read off the
    future-window spans; adapted basis; stepwise ledger
        Phi_k = N_ker^{(k)} - delta(floor kappa^{(k)}),
    with S_k(x) = sum over the dead subspace, kappa^{(k)} its phase (pi/8 units),
    N_ker^{(k)} = odd kernel parities no longer expressible in any future window.

Every sampled object is circuit-realizable BY CONSTRUCTION (the only permissive
relaxations: free X gates / affine constants, and free S/Z phases -- both within
the affine-coupling model).

Regression: w=1 samples must show 0 violations (cf. validate_w1.py: 2153 chains, 0 viol).
Acid test: delta(2(x0x1^x0x2)) = 3 via the decoder; the abstract width-1 family that
realizes it (w1_search.verify_one_violation) violates the inequality, and the circuit
sampler must NEVER produce it (or any violation).
"""
import itertools, cmath, math, random, sys, time
from collections import defaultdict
from validate_amdist import (residue_vec, rm_generators, min_coset_weight, add_term,
                             ccz_on, cs_on, phase_from_coeff)
from w1_search import delta_of_table

W8 = cmath.exp(1j * math.pi / 4)
TOL = 1e-9

# --------------------------- F2 linear algebra (bitmask rows) ---------------------------
class Span:
    """F2 span of bitmask vectors, reduced echelon by leading bit."""
    def __init__(self, vecs=()):
        self.b = {}                        # leadbit -> vector
        for v in vecs:
            self.add(v)
    def reduce(self, v):
        for l in sorted(self.b, reverse=True):
            if (v >> l) & 1:
                v ^= self.b[l]
        return v
    def add(self, v):
        v = self.reduce(v)
        if v:
            self.b[v.bit_length() - 1] = v
        return v
    def contains(self, v):
        return self.reduce(v) == 0
    def dim(self):
        return len(self.b)
    def elements(self):
        vs = list(self.b.values())
        out = [0]
        for v in vs:
            out += [o ^ v for o in out]
        return out

def rref(rows, rhs):
    """RREF of [rows | rhs] over F2. rows: list of ints; rhs: list of 0/1 ints.
    Returns (pivots dict col->(row,b), consistent)."""
    piv = {}
    for r, b in zip(rows, rhs):
        for c in sorted(piv, reverse=True):
            if (r >> c) & 1:
                rr, bb = piv[c]
                r ^= rr; b ^= bb
        if r == 0:
            if b:
                return None, False
            continue
        piv[r.bit_length() - 1] = (r, b)
    # full reduction
    for c in sorted(piv):
        r, b = piv[c]
        for c2 in sorted(piv, reverse=True):
            if c2 != c and (r >> c2) & 1:
                rr, bb = piv[c2]
                r ^= rr; b ^= bb
        piv[c] = (r, b)
    return piv, True

def solve_affine(rows, rhs, ncols):
    """One solution z (int) of rows.z = rhs over F2, plus nullspace basis. None if inconsistent."""
    piv, ok = rref(rows, rhs)
    if not ok:
        return None, None
    sol = 0
    for c, (r, b) in piv.items():
        if b:
            sol |= (1 << c)
    free = [c for c in range(ncols) if c not in piv]
    null = []
    for f in free:
        v = 1 << f
        for c, (r, b) in piv.items():
            if (r >> f) & 1:
                v |= (1 << c)
        null.append(v)
    return sol, null

def invert_matrix(cols, m):
    """cols: list of m ints (column bitmasks over m bits) forming invertible matrix M.
    Return rows of M^{-1} as list of m ints (row bitmasks)."""
    # Gaussian elimination on [M | I] acting on rows. Build rows of M from cols.
    rows = []
    for i in range(m):
        r = 0
        for j in range(m):
            if (cols[j] >> i) & 1:
                r |= (1 << j)
        rows.append(r)
    aug = [(rows[i], 1 << i) for i in range(m)]
    piv = {}
    for r, e in aug:
        for c in sorted(piv, reverse=True):
            if (r >> c) & 1:
                rr, ee = piv[c]
                r ^= rr; e ^= ee
        if r == 0:
            return None
        piv[r.bit_length() - 1] = (r, e)
    for c in sorted(piv):
        r, e = piv[c]
        for c2 in sorted(piv, reverse=True):
            if c2 != c and (r >> c2) & 1:
                rr, ee = piv[c2]
                r ^= rr; e ^= ee
        piv[c] = (r, e)
    inv_rows = [0] * m
    for c in range(m):
        inv_rows[c] = piv[c][1]
    return inv_rows

def popparity(x):
    return bin(x).count("1") & 1

# --------------------------- circuit sampling ---------------------------
def sample_circuit(rng, n, m, h, colive_bias=True):
    """Generate one random {H,CNOT,T^c,X} circuit. Returns
    (phase_terms, couplings, seg_wires) where
      phase_terms: list of (c, form, seg); couplings: list of (f_form, j);
      seg_wires: wire tuple at each segment 0..h (segment s = after H_s)."""
    CONST = 1 << (n + h)
    wires = [(1 << i) if i < n else 0 for i in range(m)]
    phase_terms = []
    couplings = []
    seg_wires = [tuple(wires)]
    def rowops(k):
        for _ in range(k):
            a = rng.randrange(m); b = rng.randrange(m)
            if a != b:
                wires[a] ^= wires[b]
        if rng.random() < 0.25:
            wires[rng.randrange(m)] ^= CONST
    def add_phase(seg):
        for _ in range(rng.randrange(3)):
            i = rng.randrange(m)
            v = wires[i]
            if v == 0 or v == CONST:
                continue
            c = rng.choice((1, 1, 1, 3, 5, 7, 2, 2, 6, 4))
            phase_terms.append((c, v, seg))
    for j in range(h):
        rowops(rng.randrange(4)); add_phase(j); rowops(rng.randrange(3))
        # choose consumed wire; with colive_bias prefer NOT consuming the freshest y
        cand = list(range(m))
        if colive_bias and j >= 1 and rng.random() < 0.5:
            fresh = 1 << (n + j - 1)
            nc = [i for i in cand if not (wires[i] >> n) & ((1 << h) - 1) & (fresh >> n)]
            if nc:
                cand = nc
        i = rng.choice(cand)
        couplings.append((wires[i], j))
        wires[i] = 1 << (n + j)
        add_phase(j)          # phases straight after birth (same segment j... see below)
        seg_wires.append(tuple(wires))
    rowops(rng.randrange(4)); add_phase(h)
    seg_wires[-1] = tuple(wires)   # not used for windows; windows fixed by H events
    return phase_terms, couplings, seg_wires, list(wires)

def final_align(wires, n, m, h):
    """Append a terminal CNOT layer R so that R.W(x,y) = (x,0^a) is solvable in y for
    every x. Returns G rows (list of m form ints) or None."""
    ymask = ((1 << h) - 1) << n
    # columns over F2^m
    def col(bit):
        c = 0
        for i in range(m):
            if (wires[i] >> bit) & 1:
                c |= (1 << i)
        return c
    ycols = [col(n + j) for j in range(h)]
    L = Span(ycols)
    ccol = col(n + h)
    if not L.contains(ccol):
        return None
    xcols = [col(i) for i in range(n)]
    # find l_i in L with {xcols[i] ^ l_i} independent (brute force, n<=3, |L|<=2^m)
    elems = L.elements()
    choice = None
    for combo in itertools.product(elems, repeat=n):
        us = [xcols[i] ^ combo[i] for i in range(n)]
        sp = Span(us)
        if sp.dim() == n:
            choice = us
            break
    if choice is None:
        return None
    # complete to invertible m x m (columns)
    cols = list(choice)
    sp = Span(cols)
    for k in range(m):
        if sp.add(1 << k):
            cols.append(1 << k)
        if len(cols) == m:
            break
    if len(cols) < m:
        return None
    Rrows = invert_matrix(cols, m)     # rows of R = Minv^{-1}
    if Rrows is None:
        return None
    G = []
    for i in range(m):
        g = 0
        for j in range(m):
            if (Rrows[i] >> j) & 1:
                g ^= wires[j]
        G.append(g)
    return G

# --------------------------- analysis ---------------------------
def eval_form(form, n, h, x, y):
    """Evaluate affine form (int bitmask) at x (n bits), y (h bits)."""
    v = popparity(form & x & ((1 << n) - 1))
    v ^= popparity(((form >> n) & ((1 << h) - 1)) & y)
    v ^= (form >> (n + h)) & 1
    return v

def analyze(n, m, h, sample, want_steps=True):
    phase_terms, couplings, seg_wires, wires = sample
    G = final_align(wires, n, m, h)
    if G is None:
        return ("noalign",)
    hmask = (1 << h) - 1
    MG = [(g >> n) & hmask for g in G]           # m rows over h y-cols
    # solve MG z = t(x) with t_i(x) = x_i (i<n) ^ (Ax)_i ^ c_i
    def target_bits(x):
        t = []
        for i in range(m):
            b = ((x >> i) & 1) if i < n else 0
            b ^= popparity(G[i] & x & ((1 << n) - 1))
            b ^= (G[i] >> (n + h)) & 1
            t.append(b)
        return t
    zc, null = solve_affine(MG, target_bits(0), h)
    if zc is None:
        return ("nosolve",)
    Zcols = []
    for i in range(n):
        zi, _ = solve_affine(MG, target_bits(1 << i), h)
        if zi is None:
            return ("nosolve",)
        Zcols.append(zi ^ zc)
    Vb = null                                     # basis of V (h-bit ints)
    d = len(Vb)
    if d == 0:
        return ("d0",)
    def z0(x):
        z = zc
        for i in range(n):
            if (x >> i) & 1:
                z ^= Zcols[i]
        return z
    def restrict(u):                              # functional u (h bits) -> d-bit vec on V
        r = 0
        for i, b in enumerate(Vb):
            if popparity(u & b):
                r |= (1 << i)
        return r
    # windows per segment s = 1..h (span of restrictions of live y-parts)
    windows = []
    for s in range(1, h + 1):
        vecs = [restrict((wv >> n) & hmask) for wv in seg_wires[s]]
        windows.append(Span([v for v in vecs if v]))
    w = max([sp.dim() for sp in windows] or [0])
    # future spans F_s = span(windows s..h), s = 1..h+1
    F = [None] * (h + 2)
    F[h + 1] = Span()
    for s in range(h, 0, -1):
        sp = Span(list(F[s + 1].b.values()))
        for v in windows[s - 1].b.values():
            sp.add(v)
        F[s] = sp
    # dead dimension after segment s = d - dim F[s+1]; removal steps where it jumps
    # adapted basis b_1..b_d of F2^d: dead flag D(s) = nullspace of F[s+1] rows
    steps = []            # (segment s, list of new dead basis vecs)
    dead = Span()
    basis = []
    for s in range(1, h + 1):
        rows = list(F[s + 1].b.values())
        _, nulls = solve_affine(rows, [0] * len(rows), d)
        newvecs = []
        for v in nulls:
            if dead.add(v):
                basis.append(v)
                newvecs.append(v)
        if newvecs:
            steps.append((s, len(newvecs)))
    if len(basis) != d:
        return ("filtration",)                    # should not happen
    # merged parities
    merged = defaultdict(int)
    for (c, form, seg) in phase_terms:
        if (form >> (n + h)) & 1:
            merged[form & ~(1 << (n + h))] = (merged[form & ~(1 << (n + h))] - c) % 8
        else:
            merged[form] = (merged[form] + c) % 8
    odd_kernel = []                               # (form, mres)
    for form, c in merged.items():
        if c % 2 == 0:
            continue
        mres = restrict((form >> n) & hmask)
        if mres:
            odd_kernel.append((form, mres))
    Nker = len(odd_kernel)
    # psi(x): kernel-free content at the section
    def psi(x):
        z = z0(x)
        p = 0
        for (c, form, seg) in phase_terms:
            if restrict((form >> n) & hmask) == 0:
                p += c * eval_form(form, n, h, x, z)
        for (f, j) in couplings:
            if restrict((f >> n) & hmask) == 0 and restrict(1 << j) == 0:
                p += 4 * eval_form(f, n, h, x, z) * ((z >> j) & 1)
        return p % 8
    def P(x, y):
        p = 0
        for (c, form, seg) in phase_terms:
            p += c * eval_form(form, n, h, x, y)
        for (f, j) in couplings:
            p += 4 * eval_form(f, n, h, x, y) * ((y >> j) & 1)
        return p % 8
    # coordinate -> y vector
    def vy(coord):
        y = 0
        for i in range(d):
            if (coord >> i) & 1:
                y ^= Vb[i]
        return y
    NX = 1 << n
    # endpoint Gauss sum
    kap = [0] * NX
    target_mod = 2 ** (h / 2)
    for x in range(NX):
        zx = z0(x); px = psi(x)
        Z = 0j
        for coord in range(1 << d):
            Z += W8 ** ((P(x, zx ^ vy(coord)) - px) % 8)
        if abs(abs(Z) - target_mod) > 1e-6:
            return ("notdiag",)
        ang = cmath.phase(Z) / (cmath.pi / 4)
        k = round(ang)
        if abs(ang - k) > 1e-6:
            return ("notdiag",)
        kap[x] = k % 8
    dk = delta_of_table(kap, n)
    if dk is None:
        return ("notlevel3",)
    # stepwise ledger
    step_info = []
    if want_steps and d >= 1:
        # cumulative dead-basis prefixes per removal step
        pref = 0
        seg_of_step = []
        for (s, cnt) in steps:
            for _ in range(cnt):
                pref += 1
                seg_of_step.append((s, pref))
        for (s, k) in seg_of_step:
            # N^{(k)}: odd kernel parities not expressible in future windows F[s+1]
            Fk = F[s + 1]
            Nk = sum(1 for (_, mres) in odd_kernel if not Fk.contains(mres))
            # partial sum over dead prefix span
            kap16 = [0] * NX
            coherent = True
            for x in range(NX):
                zx = z0(x); px = psi(x)
                S = 0j
                for bits in range(1 << k):
                    coord = 0
                    for i in range(k):
                        if (bits >> i) & 1:
                            coord ^= basis[i]
                    S += W8 ** ((P(x, zx ^ vy(coord)) - px) % 8)
                if abs(S) < TOL:
                    coherent = False
                    break
                ang = cmath.phase(S) / (cmath.pi / 8)
                a16 = round(ang)
                if abs(ang - a16) > 1e-6:
                    coherent = False
                    break
                kap16[x] = a16 % 16
            if not coherent:
                step_info.append((k, Nk, None, "incoherent"))
                continue
            k8 = [v // 2 for v in kap16]
            dks = delta_of_table(k8, n)
            if dks is None:
                step_info.append((k, Nk, None, "notlevel3"))
                continue
            step_info.append((k, Nk, dks, "ok"))
    return ("ok", d, w, Nker, dk, tuple(kap), step_info, steps,
            phase_terms, couplings, Vb, merged)

# --------------------------- acid test / decoder self-test ---------------------------
def self_tests():
    print("DECODER SELF-TESTS:")
    c = {}; ccz_on(c, (0, 1, 2))
    r = residue_vec(c, 3); print("  delta(CCZ)      =", min_coset_weight(r, rm_generators(3)), "(expect 7)")
    c = {}; cs_on(c, (0, 1))
    r = residue_vec(c, 3); print("  delta(CS)       =", min_coset_weight(r, rm_generators(3)), "(expect 3)")
    c = {}; add_term(c, 1, 1)
    r = residue_vec(c, 3); print("  delta(T)        =", min_coset_weight(r, rm_generators(3)), "(expect 1)")
    # acid identity: kappa(x) = 2(x0x1 ^ x0x2) = CS01 CS02 CCZ012, delta = 3
    n = 3
    tab = [(2 * (((x & 1) & ((x >> 1) & 1)) ^ ((x & 1) & ((x >> 2) & 1)))) % 8 for x in range(8)]
    dk = delta_of_table(tab, n)
    print("  delta(2(x0x1^x0x2)) =", dk, "(expect 3)  [the acid-test form: Nker=0 abstractly]")
    # cross-check it equals CS01*CS02*CCZ012
    c = {}; cs_on(c, (0, 1)); cs_on(c, (0, 2)); ccz_on(c, (0, 1, 2))
    ph = phase_from_coeff(c, 3)
    print("  table == CS01 CS02 CCZ012 ?", list(ph) == tab)
    assert dk == 3
    print()

# --------------------------- main sweep ---------------------------
def run(trials_per_cfg=120000, seed=20260705, want_steps=True):
    rng = random.Random(seed)
    cfgs = [(2, 3, 3), (2, 3, 4), (2, 4, 4), (2, 4, 5),
            (3, 4, 3), (3, 4, 4), (3, 4, 5), (3, 5, 4), (3, 5, 5)]
    stats = defaultdict(lambda: [0, 0, None, 0])   # (n,h,d,w) -> [valid, viol, minslack, tight]
    dforms = defaultdict(set)
    reasons = defaultdict(int)
    step_viol = 0
    step_checked = 0
    step_undef = 0
    viol_dumps = []
    tight_examples = {}
    t0 = time.time()
    for (n, m, h) in cfgs:
        for _ in range(trials_per_cfg):
            s = sample_circuit(rng, n, m, h)
            res = analyze(n, m, h, s, want_steps=want_steps)
            if res[0] != "ok":
                reasons[res[0]] += 1
                continue
            _, d, w, Nker, dk, kap, step_info, steps, pt, cp, Vb, merged = res
            key = (n, h, d, w)
            st = stats[key]
            st[0] += 1
            slack = Nker - dk
            if st[2] is None or slack < st[2]:
                st[2] = slack
            if slack == 0 and dk > 0:
                st[3] += 1
                if key not in tight_examples:
                    tight_examples[key] = (Nker, dk, kap)
            dforms[key].add((tuple(kap), Nker))
            if dk > Nker:
                st[1] += 1
                viol_dumps.append((n, m, h, d, w, Nker, dk, kap, pt, cp, Vb))
            for (k, Nk, dks, status) in step_info:
                if status != "ok":
                    step_undef += 1
                    continue
                step_checked += 1
                if dks > Nk:
                    step_viol += 1
                    if len(viol_dumps) < 50:
                        viol_dumps.append(("STEP", n, m, h, d, w, k, Nk, dks, kap, pt, cp, Vb))
        print(f"  [config n={n} m={m} h={h} done, {time.time()-t0:.0f}s]")
    print()
    print("REALIZABLE CIRCUIT SWEEP (random {H,CNOT,T} circuits, diagonal, level-3):")
    print(f"{'n':>2} {'h':>2} {'d':>2} {'w':>2} {'valid':>7} {'distinct':>8} {'viol':>5} "
          f"{'minslack':>8} {'tight':>6}")
    tot_valid = tot_viol = 0
    for key in sorted(stats):
        n, h, d, w = key
        v, x, ms, tg = stats[key]
        tot_valid += v; tot_viol += x
        print(f"{n:>2} {h:>2} {d:>2} {w:>2} {v:>7} {len(dforms[key]):>8} {x:>5} {ms:>8} {tg:>6}")
    print(f"\nTOTAL: {tot_valid} valid realizable configs, {tot_viol} endpoint violations.")
    print(f"STEPWISE LEDGER: {step_checked} well-defined steps checked, "
          f"{step_viol} violations, {step_undef} undefined (incoherent/not-level-3).")
    print(f"discard reasons: {dict(reasons)}")
    for vd in viol_dumps[:10]:
        print("VIOLATION:", vd)
    return tot_valid, tot_viol, step_viol, stats, dforms, tight_examples

if __name__ == "__main__":
    self_tests()
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
    run(trials_per_cfg=trials)
