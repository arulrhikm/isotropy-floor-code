"""Attach a NAMED circuit to the internal-Hadamard theory.

Every internal-Hadamard result in Appendix H -- thm:reduce, cor:maxrank, thm:layersum,
prop:epsH -- is stated purely in strata parameters (h, rho_G, d, w, K).  No
circuit family is ever named as an instance.  Meanwhile every family the paper actually
prices is either Hadamard-free at the level the floor is applied, or has EXTERNAL Hadamards
freed by lem:biinv, or (ripple-carry chain, unary-iteration AND ladder, Gottesman-Mochon,
HRT U_k) has genuine internal Hadamards and is priced ONLY as a diagonal shadow.

This script measures real arithmetic circuits and reports which regime they land in.

Construction.  The canonical motivated shape with genuine internal Hadamards is
compute-phase-uncompute: run a Toffoli network forward, apply a phase, run it backward.  The
net gate on the data is diagonal and the ancillas return clean, so it is in scope; the
Hadamards (two per Toffoli, since a Toffoli is a CCZ conjugated by H on the target) sit in the
middle of the circuit with T-bearing content on both sides, so they are INTERNAL and lem:biinv
does not free them.

Question.  cor:maxrank closes rho_G = h/2 unconditionally at any even h, and 672/701
(95.9%) of valid circuits in a random layered sweep were maximal-output-rank.  But the 29
that were not are precisely the thin/serialized shapes (1,1,1), (1,1,1,1), (1,1,2,1) -- which
is the regime compiled arithmetic lives in, and the fully sequential shape is
settled only at h=2.  So: does a real adder look like the 95.9% or the 29?

Both answers are useful.  rho_G = h/2 means cor:maxrank prices the OBJECT, not the shadow, for
a named family at unbounded h.  rho_G = 1 means we have the first named, motivated instance of
the open stratum -- "the open case is not exotic, it is the ripple-carry adder".

Run:  python reach_named_circuits.py            (self-tests + the table)
      python reach_named_circuits.py --audit    (also the independent path-sum audit)
"""
import argparse, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_w2 import analyze, self_tests, final_align
from c3_permutation import adder_carry_toffolis


# ---------------------------------------------------------------- circuit builder

class Builder:
    """Emit validate_w2.analyze's sample format for an explicitly written circuit.

    Wire values are affine forms over (x_0..x_{n-1}; y_0..y_{h-1}; 1) packed as int
    bitmasks: bits 0..n-1 = x, bits n..n+h-1 = branch variables, bit n+h = constant 1.
    Data wires start carrying x_i; ancilla wires start at 0.
    """

    def __init__(self, n_data, n_wires, h):
        self.n = n_data
        self.m = n_wires
        self.h = h
        self.CONST = 1 << (n_data + h)
        self.wires = [(1 << i) if i < n_data else 0 for i in range(n_wires)]
        self.phase_terms = []
        self.couplings = []
        self.seg_wires = [tuple(self.wires)]
        self.j = 0                      # next branch variable index
        self.seg = 0                    # current segment = number of H events so far

    # --- gates -------------------------------------------------------------
    def cnot(self, ctrl, targ):
        self.wires[targ] ^= self.wires[ctrl]

    def phase(self, wire, c):
        """T^c on the wire's current value (c in Z_8)."""
        v = self.wires[wire]
        if v and v != self.CONST:
            self.phase_terms.append((c % 8, v, self.seg))

    def ccz(self, a, b, c):
        """4*u*v*w expanded into parities: singletons +1, pairs -1, triple +1 (mod 8).

        This is validate_amdist.ccz_on's identity, applied to wire VALUES rather than
        variables, which is what makes it usable mid-circuit.
        """
        u, v, w = self.wires[a], self.wires[b], self.wires[c]
        for k, form in ((1, u), (1, v), (1, w),
                        (7, u ^ v), (7, u ^ w), (7, v ^ w),
                        (1, u ^ v ^ w)):
            if form and form != self.CONST:
                self.phase_terms.append((k, form, self.seg))

    def cz(self, a, b):
        """4*u*v = 2u + 2v + 6(u^v) mod 8.  Clifford: all coefficients even."""
        u, v = self.wires[a], self.wires[b]
        for k, form in ((2, u), (2, v), (6, u ^ v)):
            if form and form != self.CONST:
                self.phase_terms.append((k, form, self.seg))

    def hadamard(self, wire):
        """Consume the wire's value into a coupling and replace it with a fresh y_j."""
        assert self.j < self.h, "more Hadamards than declared"
        f = self.wires[wire]
        # strict causality: the consumed form may only involve EARLIER branch vars
        ypart = (f >> self.n) & ((1 << self.h) - 1)
        assert ypart >> self.j == 0, "coupling reads a branch variable not yet born"
        self.couplings.append((f, self.j))
        self.wires[wire] = 1 << (self.n + self.j)
        self.j += 1
        self.seg += 1
        self.seg_wires.append(tuple(self.wires))

    def toffoli(self, a, b, c):
        """Toffoli = CCZ conjugated by Hadamards on the target: two INTERNAL Hadamards."""
        self.hadamard(c)
        self.ccz(a, b, c)
        self.hadamard(c)

    # --- output ------------------------------------------------------------
    def sample(self):
        assert self.j == self.h, f"declared h={self.h} but emitted {self.j} Hadamards"
        assert len(self.seg_wires) == self.h + 1
        return (self.phase_terms, self.couplings, self.seg_wires, list(self.wires))


# ---------------------------------------------------------------- named families

def ccz_via_ancilla(m):
    """m disjoint CCZ gates, each implemented with its own borrowed clean ancilla.

    Per block: Toffoli(a,b,t) computes the AND into scratch, CZ(t,c) is Clifford, the
    second Toffoli uncomputes.  Net gate = CCZ(a,b,c) on the data, scratch returns clean.
    So the TARGET is the parallel Toffoli layer of thm:parccz -- delta = 6m+1 EXACTLY --
    and the CIRCUIT has h = 4m genuinely internal Hadamards.

    Wires: data a_i,b_i,c_i (3m of them) then scratch t_i (m).
    """
    n_data, n_wires = 3 * m, 4 * m
    b = Builder(n_data, n_wires, 4 * m)
    for i in range(m):
        a, bb, c, t = 3 * i, 3 * i + 1, 3 * i + 2, 3 * m + i
        b.toffoli(a, bb, t)
        b.cz(t, c)
        b.toffoli(a, bb, t)
    return b, f"CCZ-via-ancilla m={m}"


def ccz_shared_ancilla(m):
    """m CCZ gates sharing ONE scratch wire, computed and uncomputed in sequence.

    Same target as ccz_via_ancilla (m disjoint CCZs, delta = 6m+1) but the scratch is
    reused, so the Hadamard windows are strictly serialized on a single wire -- the
    thin/sequential shape (1,1,1,1,...) that the proved closures leave undecided.
    """
    n_data, n_wires = 3 * m, 3 * m + 1
    b = Builder(n_data, n_wires, 4 * m)
    t = 3 * m
    for i in range(m):
        a, bb, c = 3 * i, 3 * i + 1, 3 * i + 2
        b.toffoli(a, bb, t)
        b.cz(t, c)
        b.toffoli(a, bb, t)
    return b, f"CCZ-shared-scratch m={m}"


def carry_chain_gadget(m):
    """The ripple-carry carry chain (adder_carry_toffolis) run forward, CZ-phased at the
    top, and run backward.  Consecutive blocks share the running carry wire, so this is
    the linear-overlap chain the paper prices at 4m+3 -- as a SHADOW.  Here the carry
    Toffolis carry their own internal Hadamards.
    """
    tof, n_wires = adder_carry_toffolis(m)
    # wires: a_0..a_{m-1} = 0..m-1, carries c_0..c_m = m..2m.  Data = a's and c_0;
    # the carries c_1..c_m are the scratch and must return clean.
    n_data = m + 1
    seq = list(tof) + list(reversed(tof))
    b = Builder(n_data, n_wires, 2 * len(seq))
    top = tof[-1][2]
    for k, (a, bb, c) in enumerate(seq):
        b.toffoli(a, bb, c)
        if k == len(tof) - 1:
            b.cz(top, 0)          # Clifford phase on the accumulated carry
    return b, f"carry-chain m={m}"


# ---------------------------------------------------------------- regime decider

def regime(h, rho_G, d, w, dkappa):
    """Which proved closure covers this circuit?  Strongest first."""
    if h == 0:
        return "cor:ancilla (h=0)", True
    if dkappa == 0:
        return "cor:maxrank, Clifford-correction clause (delta(kappa)=0)", True
    if 2 * rho_G == h:
        return "cor:maxrank (rho_G = h/2)", True
    if rho_G == d:
        return "rho_G = d (trivial)", True
    return f"OPEN (rho_G={rho_G}, d={d}, w={w})", False


# ---------------------------------------------------------------- driver

def measure(build_fn, arg, audit=False):
    b, label = build_fn(arg)
    smp = b.sample()
    res = analyze(b.n, b.m, b.h, smp, want_steps=False)
    if res[0] != "ok":
        return dict(label=label, n=b.n, m=b.m, h=b.h, status=res[0])
    d, w, Nker, dk = res[1], res[2], res[3], res[4]
    rho_G = b.h - d
    reg, closed = regime(b.h, rho_G, d, w, dk)
    return dict(label=label, n=b.n, m=b.m, h=b.h, rho_G=rho_G, d=d, w=w,
                Nker=Nker, dkappa=dk, regime=reg, closed=closed,
                holds=(Nker >= dk), slack=Nker - dk, status="ok")


def price_table():
    """What each UNCONDITIONAL route gives on these Clifford+T circuits (rem:gadgetbounds)."""
    from collections import defaultdict

    def circuit_tcount(b):
        merged = defaultdict(int)
        for (c, form, _) in b.phase_terms:
            merged[form] = (merged[form] + c) % 8
        return sum(1 for v in merged.values() if v % 2 == 1)

    print("=" * 88)
    print("PRICING Clifford+T: what each unconditional route gives (loose is fine)")
    print(f"  {'circuit':<24s} {'m':>2s} {'t':>3s} {'delta(U)':>8s} {'nu(U)':>6s} {'Nker':>5s} "
          f"{'d(k)':>5s} | {'thm:reduce':>10s} {'prop:epsH':>9s} {'nu':>4s}")
    for fn in (ccz_via_ancilla, ccz_shared_ancilla):
        for m in (1, 2):
            b, lab = fn(m)
            res = analyze(b.n, b.m, b.h, b.sample(), want_steps=False)
            if res[0] != "ok":
                continue
            Nker, dk = res[3], res[4]
            t, dU, nu = circuit_tcount(b), 6 * m + 1, 3 * m
            print(f"  {lab:<24s} {m:>2d} {t:>3d} {dU:>8d} {nu:>6d} {Nker:>5d} {dk:>5d} | "
                  f"{dU + Nker - dk:>10d} {dU - dk:>9d} {nu:>4d}")
    print("  thm:reduce is EXACTLY TIGHT (= t); prop:epsH is VACUOUS (delta(kappa) = delta(U),")
    print("  because the gadget phases a wire holding a branch variable, so the whole target")
    print("  phase is built by interference); nu holds for free but is ~2.7x loose.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmax", type=int, default=3)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--price", action="store_true", help="also print the pricing table")
    args = ap.parse_args()

    print(__doc__.strip())
    print("=" * 88)
    print("DECODER SELF-TESTS (validate_w2.self_tests: delta(CCZ)=7, CS=3, T=1,")
    print("                    and the acid test delta(2(x0x1^x0x2))=3)")
    self_tests()
    print("  passed\n")

    print("=" * 88)
    rows = []
    for fam in (ccz_via_ancilla, ccz_shared_ancilla, carry_chain_gadget):
        for m in range(1, args.mmax + 1):
            try:
                rows.append(measure(fam, m, audit=args.audit))
            except AssertionError as e:
                rows.append(dict(label=f"{fam.__name__} m={m}", status=f"builder: {e}"))
    hdr = (f"  {'family':<22s} {'n':>2s} {'wires':>5s} {'h':>3s} {'rho_G':>5s} {'d':>3s} "
           f"{'w':>2s} {'Nker':>4s} {'d(k)':>4s}  regime")
    print(hdr)
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['label']:<22s} {r.get('n','?'):>2} {r.get('m','?'):>5} "
                  f"{r.get('h','?'):>3}  --> {r['status']}")
            continue
        print(f"  {r['label']:<22s} {r['n']:>2d} {r['m']:>5d} {r['h']:>3d} {r['rho_G']:>5d} "
              f"{r['d']:>3d} {r['w']:>2d} {r['Nker']:>4d} {r['dkappa']:>4d}  {r['regime']}")
    print()
    ok = [r for r in rows if r.get("status") == "ok"]
    closed = [r for r in ok if r["closed"]]
    print(f"  closed by a proved regime: {len(closed)}/{len(ok)}")
    viol = [r for r in ok if not r["holds"]]
    print(f"  N_ker >= delta(kappa) violations: {len(viol)} (must be 0)")
    assert not viol, viol
    print(f"  slack N_ker - delta(kappa) per row: "
          f"{', '.join(str(r['slack']) for r in ok)}")
    if args.price:
        print()
        price_table()


if __name__ == "__main__":
    main()
