r"""
The GF(2^k) multiplier benchmark family (Feynman suite, gf2^k_mult.qc, k = 4..256), priced whole.

Each circuit is built from: H on output wires c, 3-argument 'Z'/'Zd' (= CCZ, since Z^dagger = Z),
and 2-argument 'tof' (= CNOT) among the c wires for the modular reduction.  Conjugating the whole
circuit by H on every c wire gives V = H_c U H_c.  In V's frame a CCZ acts on the current wire forms
and a CNOT c_i -> c_j met while the c wires are in the computational frame becomes the reversed CNOT
c_j -> c_i.  We track V mechanically and check that
  (i)   every CCZ has exactly one leg on a c wire, and that leg is in the H frame (else not mismatch-free);
  (ii)  V's linear part is the identity, so V is DIAGONAL: V = (-1)^{f(a,b,z)};
  (iii) f is cubic of the form sum_l z_l B_l(a,b), bilinear in (a,b);
  (iv)  for k <= 10, f(a,b,z) = z . (a * b) in GF(2^k) for the irreducible polynomial the circuit uses
        (found by search), checked on random inputs.
Then U = H_c V H_c is Clifford-equivalent to the phase oracle, so by thm:transfer its T-count in the
Hadamard-external model equals delta((-1)^f): floor 6k+1 when the radical d = 0 (prop:fieldoracle).

Also computes the canonical ceiling W(s) sparsely (top-down moment solve on the monomial support) and
checks it against the known values W = 7, 17, 36, 58, 90, 161 at k = 1..6 (polynomial basis).

Run:  python -u gf2k_family.py [--dmax 64]
"""
import argparse, glob, itertools, os, random, re, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BENCH = os.environ.get("ISOFLOOR_BENCH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks",
                     "qc")


def parse(path):
    names, gates = None, []
    for ln in open(path, encoding="utf-8"):
        t = ln.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == ".v":
            names = t[1:]
        elif t[0] in ("BEGIN", "END") or t[0].startswith("."):
            continue
        else:
            gates.append((t[0], t[1:]))
    return names, gates


def to_V(names, gates):
    """Track V = H_c U H_c.  Returns (monomials Counter over (i,j,l) index triples as frozensets of wire
    names, ok flag, message).  Wire forms are sets of original c-variables (XOR)."""
    cw = [w for w in names if w.startswith("c")]
    frame = {w: 1 for w in cw}           # 1 = H frame (V's computational frame after conjugation)
    form = {w: frozenset([w]) for w in cw}
    mons = Counter()
    for g, args in gates:
        if g == "H":
            (w,) = args
            if w not in frame:
                return None, False, f"H on non-output wire {w}"
            frame[w] ^= 1
        elif g in ("Z", "Zd") and len(args) == 3:
            cs = [w for w in args if w in frame]
            if len(cs) != 1:
                return None, False, f"CCZ {args} does not have exactly one output leg"
            c = cs[0]
            if frame[c] != 0:
                # frame[c]==0 after an odd number of H since the start of U means: in V's frame (which
                # prepends H) the wire is computational.  We started frame=1 for V's prepended H.
                return None, False, f"CCZ {args} acts on output wire {c} outside the H frame"
            a, b = [w for w in args if w not in frame]
            for v in form[c]:
                mons[frozenset((a, b, v))] ^= 1
        elif g == "tof" and len(args) == 2:
            s, t = args
            if s not in frame or t not in frame:
                return None, False, f"CNOT {args} touches a data wire"
            if frame[s] == 0 and frame[t] == 0:
                # in V's frame both wires are computational: an ordinary CNOT s -> t on the forms
                form[t] = form[t] ^ form[s]
            elif frame[s] == 1 and frame[t] == 1:
                # in V's frame both wires are in the H basis: H.CNOT(s->t).H = CNOT(t->s)
                form[s] = form[s] ^ form[t]
            else:
                return None, False, f"CNOT {args} across mixed frames"
        else:
            return None, False, f"unexpected gate {g} {args}"
    # V's final H (appending H_c) needs every wire back in frame 1 relative to the start
    if any(f != 1 for f in frame.values()):
        return None, False, "output wires do not end in the H frame"
    # linear part of V must be the identity on the forms that are in the computational frame of V ...
    # forms were updated only by CNOTs; V diagonal requires every c wire's form to return to itself
    # V = diag((-1)^f) . L' with L' the CNOT network on c: U = H_c V H_c is the phase oracle times a
    # CNOT network on c (Clifford), so delta_T(U) = delta((-1)^f) by lem:biinv.
    tail = sum(1 for w in cw if form[w] != frozenset([w]))
    mons = {m for m, v in mons.items() if v}
    return mons, True, f"ok (CNOT tail on {tail} output wires)"


def gf_mul(a, b, k, poly):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> k:
            a ^= poly
    return r


def irreducibles(k):
    out = []
    for p in range(1 << k, 1 << (k + 1)):
        if not p & 1:
            continue
        ok = True
        for d in range(1, k // 2 + 1):
            for q in range(1 << d, 1 << (d + 1)):
                # polynomial division p mod q
                r = p
                while r.bit_length() - 1 >= d:
                    r ^= q << (r.bit_length() - 1 - d)
                if r == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(p)
    return out


def f2_solve_consistent(G, B):
    """Is there M with B = G M over F2?  G, B: lists of int bitmask rows (width k)."""
    rows = [(g, b) for g, b in zip(G, B)]
    piv = {}
    for g, b in rows:
        while g:
            h = g.bit_length() - 1
            if h in piv:
                pg, pb = piv[h]
                g ^= pg; b ^= pb
            else:
                piv[h] = (g, b)
                break
        if g == 0 and b != 0:
            return False
    return True


def check_field(mons, k, trials=200):
    """Find an irreducible p and an invertible-or-not M with  f(a,b,z) = z . M(a*b mod p)."""
    idx = lambda s: int(s[1:])
    trip = []
    for m in mons:
        a = [w for w in m if w.startswith("a")][0]
        b = [w for w in m if w.startswith("b")][0]
        c = [w for w in m if w.startswith("c")][0]
        trip.append((idx(a), idx(b), idx(c)))
    rng = random.Random(1)
    samples = [(rng.randrange(1 << k), rng.randrange(1 << k), rng.randrange(1 << k)) for _ in range(trials)]
    def f(x, y, z):
        return sum(((x >> i) & (y >> j) & (z >> l) & 1) for i, j, l in trip) & 1
    def Bvec(x, y):
        return sum(f(x, y, 1 << l) << l for l in range(k))
    pairs = [(x, y) for x, y, _ in samples]
    Bs = [Bvec(x, y) for x, y in pairs]
    for p in irreducibles(k):
        G = [gf_mul(x, y, k, p) for x, y in pairs]
        if f2_solve_consistent(G, Bs):
            return p
    return None


def radical_dim_monos(mons, names):
    """d = dim of radical of the cubic tensor, by F2 elimination on the n slice matrices."""
    wires = sorted({w for m in mons for w in m})
    pos = {w: i for i, w in enumerate(wires)}
    n = len(wires)
    # slice(w_a) has entry (q,r) = 1 iff {a,q,r} in mons; represent slice as bitmask over pairs
    pair_id = {}
    vecs = [0] * n
    for m in mons:
        a, b, c = sorted(pos[w] for w in m)
        for (x, y, z) in ((a, b, c), (b, a, c), (c, a, b)):
            key = (min(y, z), max(y, z))
            if key not in pair_id:
                pair_id[key] = len(pair_id)
            vecs[x] ^= 1 << pair_id[key]
    piv = {}
    rank = 0
    for v in vecs:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                rank += 1
                break
    return n - rank, n


def canonical_ceiling(mons):
    """W(s) for a pure-cubic (-1)^f with monomial support mons: triples, then pairs, then singles."""
    pairs = Counter()
    for m in mons:
        for p in itertools.combinations(sorted(m), 2):
            pairs[p] ^= 1
    zp = {p for p, v in pairs.items() if v}
    single = Counter()
    for m in mons:
        for a in m:
            single[a] ^= 1
    for p in zp:
        for a in p:
            single[a] ^= 1
    zs = {a for a, v in single.items() if v}
    return len(mons) + len(zp) + len(zs)


def selftest_W():
    from field_oracle import mult_tensor_blocks
    want = {1: 7, 2: 17, 3: 36, 4: 58, 5: 90, 6: 161}
    ok = True
    for k, w in want.items():
        IRR = {1: 0b11, 2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011}
        mons = set()
        for i in range(k):
            for j in range(k):
                pr = gf_mul(1 << i, 1 << j, k, IRR[k])
                for l in range(k):
                    if (pr >> l) & 1:
                        mons ^= {frozenset((f"a{i}", f"b{j}", f"c{l}"))}
        got = canonical_ceiling(mons)
        print(f"  W check k={k}: {got} (expected {w})")
        ok &= (got == w)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmax", type=int, default=64)
    a = ap.parse_args()
    print("selftest W(s):")
    selftest_W()
    paths = sorted(glob.glob(os.path.join(BENCH, "gf2^*_mult.qc")),
                   key=lambda p: int(re.search(r"gf2\^(\d+)_", p).group(1)))
    print(f"{'k':>4s} {'n':>4s} {'CCZ':>6s} {'T_in=7CCZ':>9s} {'struct':>6s} {'field':>10s} {'d':>3s} "
          f"{'floor':>6s} {'W(s)':>7s} {'time':>6s}")
    for p in paths:
        k = int(re.search(r"gf2\^(\d+)_", p).group(1))
        t0 = time.time()
        names, gates = parse(p)
        nccz = sum(1 for g, x in gates if g in ("Z", "Zd") and len(x) == 3)
        mons, ok, msg = to_V(names, gates)
        if not ok:
            print(f"{k:>4d}  STRUCTURE FAILED: {msg}")
            continue
        field = check_field(mons, k) if k <= 10 else None
        fstr = (bin(field) if field else "none!") if k <= 10 else "-"
        if k <= a.dmax:
            d, n = radical_dim_monos(mons, names)
        else:
            d, n = None, 3 * k
        floor = 2 * n + 1 if d == 0 else (2 * (n - d) + 1 if d is not None else None)
        W = canonical_ceiling(mons)
        print(f"{k:>4d} {n:>4d} {nccz:>6d} {7 * nccz:>9d} {'ok':>6s} {fstr:>10s} "
              f"{'-' if d is None else d:>3} {str(floor):>6s} {W:>7d} {time.time() - t0:>5.1f}s", flush=True)


if __name__ == "__main__":
    main()
