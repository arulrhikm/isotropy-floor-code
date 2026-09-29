r"""
AND-compiled Clifford+T circuits and the isotropy floor (cor:andcircuits).

CLASS.  Measurement-free circuits on n data wires plus clean ancillas.  Every internal Hadamard
sits in an AND block on an ancilla t: a compute WINDOW  H_t . (phases on parities y+lambda, routed
onto t by CNOTs from the controls and back) . H_t [. S^p_t],  and later the mirror uncompute.  Each
block is a phase-free AND |a,b,0> -> |a,b,ab>.  Between compute and uncompute, the MIDDLE is any
{CNOT,T} work on data plus diagonal Cliffords touching each ancilla at most linearly (CZ(t,data),
S, Z on t): every parity involving an ancilla's middle value has an even coefficient.  Controls
are data wires (data-linear after a CNOT frame).

CLAIM (cor:andcircuits).  t >= delta(U) + A, A = number of AND blocks.
  lem:window: a window computing a non-affine AND carries >= 4 odd parities containing its variable,
  and window parities are distinct from every other parity, so t >= 8A + odd(M_D);
  after the delta collapse phi = M_D + (even Cliffords) + sum_i g_i (2 alpha_i + 4 L_i), each term on
  3 forms, so delta(U) <= odd(M_D) + 7A (lem:subadd).

PART 1 (must pass): templates, random in-class circuits, named gadget rows.
PART 2 (exploratory, report only): ladders (controls on earlier ancillas) and ODD phases on AND
values, filtered to level-3 diagonal targets.

Run:  python -u and_gadget_probe.py [--samples 400] [--seed 1]
"""
import argparse, itertools, os, random, sys, cmath
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reach_named_circuits import Builder, ccz_via_ancilla, ccz_shared_ancilla
from validate_w2 import analyze
from validate_w1 import delta_of_kappa

W8 = cmath.exp(1j * cmath.pi / 4)


# ------------------------------------------------------------------ circuit ops
# ('h',w) ('ph',w,c)=T^c ('cx',c,t) ('ccz',a,b,c) ('cz',a,b)

def simulate(ops, n, nw):
    """Return phi (Z_8 table on data) if the circuit is diagonal on data with clean ancillas, else None."""
    dim = 1 << nw
    phi = []
    for x in range(1 << n):
        v = np.zeros(dim, dtype=complex)
        v[x] = 1.0                                   # ancillas (bits n..nw-1) start at 0
        idx = np.arange(dim)
        for op in ops:
            k = op[0]
            if k == 'h':
                w = op[1]; b = 1 << w
                lo = v[(idx & b) == 0].copy(); hi = v[(idx & b) != 0].copy()
                nv = np.empty_like(v)
                nv[(idx & b) == 0] = (lo + hi) / np.sqrt(2)
                nv[(idx & b) != 0] = (lo - hi) / np.sqrt(2)
                v = nv
            elif k == 'ph':
                w, c = op[1], op[2] % 8
                v = v * np.where((idx >> w) & 1, W8 ** c, 1.0)
            elif k == 'cx':
                c, t = op[1], op[2]
                perm = np.where((idx >> c) & 1, idx ^ (1 << t), idx)
                nv = np.empty_like(v); nv[perm] = v; v = nv
            elif k == 'ccz':
                a, b, c = op[1:]
                m = ((idx >> a) & (idx >> b) & (idx >> c) & 1).astype(bool)
                v = np.where(m, -v, v)
            elif k == 'cz':
                a, b = op[1:]
                m = ((idx >> a) & (idx >> b) & 1).astype(bool)
                v = np.where(m, -v, v)
        amp = v[x]
        if abs(abs(amp) - 1) > 1e-9:
            return None
        ang = cmath.phase(amp) / (cmath.pi / 4)
        c = int(round(ang)) % 8
        if abs(ang - round(ang)) > 1e-7:
            return None
        phi.append(c)
    return phi


def to_builder(ops, n, nw):
    h = sum(1 for o in ops if o[0] == 'h')
    b = Builder(n, nw, h)
    for o in ops:
        k = o[0]
        if k == 'h':
            b.hadamard(o[1])
        elif k == 'ph':
            b.phase(o[1], o[2])
        elif k == 'cx':
            b.cnot(o[1], o[2])
        elif k == 'ccz':
            b.ccz(*o[1:])
        elif k == 'cz':
            b.cz(*o[1:])
    return b


def merged_odd(b):
    m = defaultdict(int)
    for (c, form, _) in b.phase_terms:
        m[form] = (m[form] + c) % 8
    return {f for f, c in m.items() if c % 2}


# ------------------------------------------------------------------ AND templates

def window_ops(a, b, t, coeffs):
    """Phases c_lambda on parities y + lambda, lambda in (0, a, b, a+b), routed onto t."""
    ops = []
    for lam, c in zip(((), (a,), (b,), (a, b)), coeffs):
        for w in lam:
            ops.append(('cx', w, t))
        if c % 8:
            ops.append(('ph', t, c))
        for w in reversed(lam):
            ops.append(('cx', w, t))
    return ops


def toffoli_compute(a, b, t):
    return [('h', t), ('ccz', a, b, t), ('h', t)]


def find_temp_and_templates():
    """All (coeffs in {1,7}^4, S-power p) making H.window.H.S^p a phase-free AND on (a,b,t)."""
    found = []
    for coeffs in itertools.product((1, 7), repeat=4):
        for p in (0, 2, 4, 6):
            ops = [('h', 2)] + window_ops(0, 1, 2, coeffs) + [('h', 2)] + ([('ph', 2, p)] if p else [])
            ok = True
            for a in (0, 1):
                for bb in (0, 1):
                    v = np.zeros(8, dtype=complex); v[a | (bb << 1)] = 1
                    idx = np.arange(8)
                    for op in ops:
                        if op[0] == 'h':
                            w = 1 << op[1]
                            lo = v[(idx & w) == 0].copy(); hi = v[(idx & w) != 0].copy()
                            nv = np.empty_like(v)
                            nv[(idx & w) == 0] = (lo + hi) / np.sqrt(2)
                            nv[(idx & w) != 0] = (lo - hi) / np.sqrt(2); v = nv
                        elif op[0] == 'ph':
                            v = v * np.where((idx >> op[1]) & 1, W8 ** op[2], 1.0)
                        elif op[0] == 'cx':
                            perm = np.where((idx >> op[1]) & 1, idx ^ (1 << op[2]), idx)
                            nv = np.empty_like(v); nv[perm] = v; v = nv
                    out = a | (bb << 1) | ((a & bb) << 2)
                    if abs(v[out] - 1) > 1e-9:
                        ok = False
            if ok:
                found.append((coeffs, p))
    return found


def temp_and_compute(a, b, t, tpl):
    coeffs, p = tpl
    return [('h', t)] + window_ops(a, b, t, coeffs) + [('h', t)] + ([('ph', t, p)] if p else [])


def inverse(ops):
    inv = []
    for o in reversed(ops):
        if o[0] == 'ph':
            inv.append(('ph', o[1], (-o[2]) % 8))
        else:
            inv.append(o)
    return inv


# ------------------------------------------------------------------ random in-class circuits

def random_circuit(rng, n, n_anc, n_and, tpls, ladder=False, odd_mid=False):
    """Build a circuit; return (ops, nw, meta).  meta records the per-AND data for the phi check."""
    nw = n + n_anc
    ops = []
    # data-linear frame: random CNOT network on data, undone at the end
    frame = []
    for _ in range(rng.randint(0, 2 * n)):
        c, t = rng.sample(range(n), 2)
        frame.append(('cx', c, t))
    ops += frame
    ands = []
    live = {}                                    # ancilla -> control pair while computed
    plan = []
    for i in range(n_and):
        anc = n + (i % n_anc)
        plan.append(anc)
    # schedule: process ANDs in groups of at most n_anc live at once
    i = 0
    while i < n_and:
        group = plan[i:i + n_anc]
        computed = []
        group_recs = []
        for anc in group:
            pool = list(range(n)) + ([w for w in computed] if ladder else [])
            a, b = rng.sample(pool, 2)
            kind = rng.choice(['tof'] + (['tmp'] if tpls else []))
            tpl = rng.choice(tpls) if kind == 'tmp' else None
            comp = toffoli_compute(a, b, anc) if kind == 'tof' else temp_and_compute(a, b, anc, tpl)
            ops += comp
            computed.append(anc)
            rec = dict(anc=anc, ctrl=(a, b), comp=comp)
            ands.append(rec); group_recs.append(rec)
        # middle
        for _ in range(rng.randint(1, 3 * n)):
            r = rng.random()
            if r < 0.35:
                ops.append(('ph', rng.randrange(n), rng.randrange(1, 8)))
            elif r < 0.55:
                c, t = rng.sample(range(n), 2)
                ops += [('cx', c, t), ('ph', t, rng.randrange(1, 8)), ('cx', c, t)]
            elif r < 0.80:
                ops.append(('cz', rng.choice(computed), rng.randrange(n)))
            else:
                anc = rng.choice(computed)
                c = rng.randrange(1, 8) if odd_mid else rng.choice((2, 4, 6))
                ops.append(('ph', anc, c))
        for rec in reversed(group_recs):
            ops += inverse(rec['comp'])
        i += len(group)
    ops += inverse(frame)
    return ops, nw, ands


def window_counts(b, n, ops):
    """Odd merged parities containing each window variable (first y of each compute/uncompute H pair)."""
    odd = merged_odd(b)
    # window variables: y_j born at an H on an ancilla whose previous value was 0 or which is the
    # first H of a pair; with our templates every ancilla's H events come in pairs (open, close),
    # and the window variable is the one born at the OPEN H.
    counts = []
    j = 0
    open_ = {}
    for o in ops:
        if o[0] == 'h':
            w = o[1]
            if w not in open_:
                open_[w] = j
            else:
                yj = open_.pop(w)
                bit = 1 << (n + yj)
                counts.append(sum(1 for f in odd if f & bit))
            j += 1
    return counts


def part1(samples, seed):
    rng = random.Random(seed)
    tpls = find_temp_and_templates()
    print(f"temporary-AND templates found: {len(tpls)}  e.g. {tpls[:2]}")
    assert tpls, "no phase-free 4-T temporary AND found"
    # the Toffoli template is phase-free by construction; verify once
    worst = None
    checked = 0
    for s in range(samples):
        n = rng.randint(3, 6)
        n_anc = rng.randint(1, 2)
        n_and = rng.randint(1, 3)
        ops, nw, ands = random_circuit(rng, n, n_anc, n_and, tpls)
        if nw > 9:
            continue
        phi = simulate(ops, n, nw)
        assert phi is not None, f"sample {s}: not diagonal/clean"
        dU, ok = delta_of_kappa(phi, n)
        assert ok, f"sample {s}: target not level-3"
        b = to_builder(ops, n, nw)
        t = len(merged_odd(b))
        A = len(ands)
        wc = window_counts(b, n, ops)
        assert all(c >= 4 for c in wc), f"sample {s}: window with < 4 odd parities: {wc}"
        assert t >= dU + A, f"sample {s}: VIOLATION t={t} < delta(U)+A={dU + A}  ops={ops}"
        slack = t - dU - A
        worst = slack if worst is None else min(worst, slack)
        checked += 1
    print(f"PART 1 random in-class: {checked} circuits, all t >= delta(U)+A; min slack {worst}")
    for fn in (ccz_via_ancilla, ccz_shared_ancilla):
        for m in (1, 2):
            bb, lab = fn(m)
            t = len(merged_odd(bb))
            dU = 6 * m + 1
            assert t >= dU + m, f"{lab}: t={t} < {dU + m}"
            print(f"  named {lab:<24s} t={t:>3d}  delta(U)={dU:>3d}  A={m}  t-(delta+A)={t - dU - m}")


def part2(samples, seed):
    rng = random.Random(seed + 1)
    tpls = find_temp_and_templates()
    stats = defaultdict(list)
    viol = []
    for mode in ('ladder', 'oddmid', 'both'):
        kept = 0
        for s in range(samples):
            n = rng.randint(3, 5)
            ops, nw, ands = random_circuit(rng, n, rng.randint(1, 2), rng.randint(1, 3), tpls,
                                           ladder=mode in ('ladder', 'both'),
                                           odd_mid=mode in ('oddmid', 'both'))
            if nw > 8:
                continue
            phi = simulate(ops, n, nw)
            if phi is None:
                continue
            dU, ok = delta_of_kappa(phi, n)
            if not ok:
                continue                                   # target not level-3: out of scope
            b = to_builder(ops, n, nw)
            t = len(merged_odd(b))
            kept += 1
            stats[mode].append(t - dU)
            if t < dU:
                viol.append((mode, n, t, dU, ops))
        arr = stats[mode]
        if arr:
            print(f"PART 2 {mode:<6s}: {kept} level-3 circuits, t - delta(U): min {min(arr)}, "
                  f"mean {sum(arr) / len(arr):.2f}; violations of t >= delta(U): "
                  f"{sum(1 for v in viol if v[0] == mode)}")
        else:
            print(f"PART 2 {mode:<6s}: no level-3 circuits kept")
    if viol:
        m, n, t, dU, ops = viol[0]
        print(f"  FIRST VIOLATION ({m}): n={n} t={t} delta={dU}\n  ops={ops}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    part1(a.samples, a.seed)
    part2(a.samples, a.seed)


if __name__ == "__main__":
    main()
