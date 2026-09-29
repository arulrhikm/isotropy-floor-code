r"""
Certify T-count optimizers on the standard reversible benchmark suite with the isotropy floor.

PyZX's phase_block_optimize cuts a Clifford+T circuit into phase-polynomial blocks ({CNOT,T,CZ}
regions between Hadamards, found by pushing gates past Hadamards greedily) and runs TODD on each.
We intercept every block and compute, on EXACTLY the block TODD sees:

    floor   = isotropy floor 2(n - d*) - r        (thm:genLB; +1 when pure cubic and d* = 0)
    nu      = n - d*                              (prop:nullitygen)
    input   = odd parities of the block as given
    todd    = T-count TODD returns for the block
    delta   = exact Amy-Mosca optimum             (only when the block has <= 6 active qubits)

A block is CERTIFIED OPTIMAL when floor == min(input, todd): no {CNOT,T} resynthesis of that block
can use fewer T gates.  Summed over blocks, sum(floor) lower-bounds every optimizer that keeps this
Hadamard layering; optimizers that move Hadamards (gadgetization) are NOT bounded by it
(thm:layersum: that is the Hadamard hypothesis).

Everything is computed from the block's odd-parity matrix Y (t x n) in polynomial time:
    Q      = Y^T Y mod 2                 (order-1 moments on the diagonal, order-2 off it); r = rank Q
    T'(w)  = Y^T diag(Y w) Y mod 2       (full third-power moment tensor, coincident indices folded)
    d*     = dim { w : T'(w) = 0 }.

Benchmarks: shared/benchmarks/feynman/benchmarks/qc  (github.com/meamy/feynman @ d2c382a2).

Run:  python -u benchmark_certify.py                      (whole suite, default size cap)
      python -u benchmark_certify.py tof_5 gf2^4_mult     (named circuits)
      python -u benchmark_certify.py --selftest
"""
import argparse, glob, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BENCH = os.environ.get("ISOFLOOR_BENCH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks",
                     "qc")


# ------------------------------------------------------------------ F2 linear algebra (bit-packed)

def f2_rank(rows):
    """Rank over F2 of a list of Python-int bitmask rows."""
    piv = {}
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                break
    return len(piv)


def floor_from_parities(terms, n):
    """terms: dict parity-bitmask -> coeff mod 8.  Returns dict with floor, nu, dstar, r, t_odd, active."""
    odd = [p for p, c in terms.items() if c % 2 and p]
    active = 0
    for p, c in terms.items():
        if c % 8:
            active |= p
    act = [i for i in range(n) if (active >> i) & 1]
    m = len(act)
    idx = {q: k for k, q in enumerate(act)}
    t = len(odd)
    if t == 0:
        return dict(floor=0, nu=0, dstar=m, r=0, t_odd=0, active=m, pure=True)
    Y = np.zeros((t, m), dtype=np.uint8)
    for i, p in enumerate(odd):
        for q in act:
            if (p >> q) & 1:
                Y[i, idx[q]] = 1
    Yi = Y.astype(np.int64)
    Q = (Yi.T @ Yi) & 1
    r = f2_rank([int("".join(map(str, row[::-1])), 2) if row.any() else 0 for row in Q])
    # T'(e_a) = Y^T diag(Y[:, a]) Y ; radical = kernel of a -> vec(T'(e_a))
    vecs = []
    for a in range(m):
        sel = Yi[Y[:, a] == 1]
        Ma = (sel.T @ sel) & 1
        flat = Ma[np.triu_indices(m)]
        vecs.append(int("".join(map(str, flat[::-1].tolist())) or "0", 2))
    rank_T = f2_rank(vecs)
    dstar = m - rank_T
    # pure cubic: all order-1 and order-2 moments vanish (Q == 0)
    pure = not Q.any()
    fl = 2 * (m - dstar) - r
    if pure and dstar == 0 and m > 0:
        fl = 2 * m + 1
    elif pure and m - dstar > 0:
        fl = 2 * (m - dstar) + 1          # 2nu+1 on the pure-cubic locus (radical removed by CNOTs)
    return dict(floor=max(fl, 0), nu=m - dstar, dstar=dstar, r=r, t_odd=t, active=m, pure=pure)


def exact_delta(terms, n):
    """Exact Amy-Mosca optimum for blocks with <= 6 active qubits, else None."""
    import validate_amdist as AM
    active = 0
    for p, c in terms.items():
        if c % 8:
            active |= p
    act = [i for i in range(n) if (active >> i) & 1]
    m = len(act)
    if m == 0:
        return 0
    if m > 6:
        return None
    pos = {q: k for k, q in enumerate(act)}
    coeff = {}
    for p, c in terms.items():
        if c % 8 == 0:
            continue
        y = 0
        for q in act:
            if (p >> q) & 1:
                y |= 1 << pos[q]
        coeff[y] = (coeff.get(y, 0) + c) % 8
    r = AM.residue_vec(coeff, m)
    return int(AM.min_coset_weight(r, AM.rm_generators(m)))


# ------------------------------------------------------------------ PyZX hook

RECORDS = []


def install_hook(compute_exact=True):
    import pyzx.optimize as opt
    from pyzx import todd
    orig = todd.todd_simp

    def hooked(gates, qubits, quiet=True):
        poly, _ = todd.phase_gates_to_poly(gates, qubits)
        terms = {}
        for par, val in poly.terms.items():
            mask = 0
            for q in par:
                mask |= 1 << q
            terms[mask] = (terms.get(mask, 0) + val) % 8
        t0 = time.time()
        rec = floor_from_parities(terms, qubits)
        rec["t_floor"] = time.time() - t0
        rec["delta"] = exact_delta(terms, qubits) if compute_exact else None
        newgates, perm = orig(gates, qubits, quiet=quiet)
        from pyzx.circuit.gates import T as TGate
        rec["todd"] = sum(1 for g in newgates if g.name in ("T", "T*") or
                          (hasattr(g, "phase") and getattr(g.phase, "denominator", 1) == 4))
        RECORDS.append(rec)
        return newgates, perm

    opt.todd_simp = hooked


def load_qc(path):
    """Load a Feynman .qc file.  Controlled 'Zd' (Z-dagger) equals controlled 'Z' since Z^dagger = Z;
    PyZX's parser rejects the controlled form, so rewrite it before loading."""
    import pyzx as zx, tempfile
    lines = []
    for ln in open(path, encoding="utf-8").read().splitlines():
        tok = ln.split()
        if tok and tok[0] == "Zd" and len(tok) > 2:
            ln = "Z " + " ".join(tok[1:])
        lines.append(ln)
    with tempfile.NamedTemporaryFile("w", suffix=".qc", delete=False, encoding="utf-8") as f:
        f.write(chr(10).join(lines) + chr(10))
        tmp = f.name
    try:
        return zx.Circuit.load(tmp)
    finally:
        os.remove(tmp)


def expand_swaps(circ):
    """phase_block_optimize rejects SWAP; a SWAP is three CNOTs."""
    from pyzx.circuit.gates import CNOT
    gs = []
    for g in circ.gates:
        if g.name == 'SWAP':
            a, b = g.control, g.target
            gs += [CNOT(a, b), CNOT(b, a), CNOT(a, b)]
        else:
            gs.append(g)
    circ.gates = gs
    return circ


def tcount(circ):
    return sum(1 for g in circ.gates if hasattr(g, "phase") and getattr(g.phase, "denominator", 1) == 4)


def partition(circ):
    """Cut a HAD/CNOT/CZ/ZPhase circuit into its phase-polynomial blocks B1 H1 B2 H2 ... with PyZX's
    own greedy_consume_gates (gates are pushed past Hadamards where they commute).  Returns the
    list of blocks, each a dict parity-bitmask -> coeff mod 8."""
    from pyzx.optimize import greedy_consume_gates
    from pyzx import todd
    q = circ.qubits
    gates = {i: [] for i in range(q)}
    for i, g in enumerate(circ.gates):
        g = g.copy(); g.index = i
        if g.name in ('CNOT', 'CZ'):
            gates[g.control].append(g); gates[g.target].append(g)
        else:
            gates[g.target].append(g)
    blocks = []
    guard = 0
    while any(gates.values()):
        block, hads = greedy_consume_gates(gates, q)
        guard += 1
        if not block and not hads:
            raise RuntimeError("partition stalled")
        if block:
            poly, _ = todd.phase_gates_to_poly(block, q)
            terms = {}
            for par, val in poly.terms.items():
                mask = 0
                for b in par:
                    mask |= 1 << b
                terms[mask] = (terms.get(mask, 0) + val) % 8
            blocks.append(terms)
    return blocks


def core(circ):
    """Drop the trailing NOT/SWAP corrections phase_block_optimize appends (Clifford permutations,
    no T content).  Does NOT re-optimize: the circuit certified is exactly PyZX's output."""
    gs = list(circ.gates)
    while gs and gs[-1].name in ("NOT", "SWAP"):
        gs.pop()
    assert not any(g.name in ("NOT", "SWAP") for g in gs), "NOT/SWAP inside the optimized body"
    c2 = circ.copy()
    c2.gates = gs
    return c2


def price_blocks(blocks, compute_exact):
    out = []
    for terms in blocks:
        r = floor_from_parities(terms, 64 if not terms else max(terms).bit_length())
        if r["t_odd"] == 0:
            continue
        r["delta"] = exact_delta(terms, max(terms).bit_length()) if compute_exact else None
        r["given"] = r["t_odd"]
        if r["delta"] is not None:
            assert r["floor"] <= r["delta"] <= r["given"], f"bracket violated: {r}"
        r["cert"] = (r["floor"] == r["given"])
        # stabilizer nullity certifies a block only when nu equals its T count; since the floor is
        # never below nu (cor:floordominates), nullity-certified implies floor-certified
        r["cert_nu"] = (r["nu"] == r["given"])
        assert r["floor"] >= r["nu"], f"floor below nullity: {r}"
        assert (not r["cert_nu"]) or r["cert"], f"nullity certifies a block the floor does not: {r}"
        out.append(r)
    return out


def teleport_optimize(c):
    """Phase teleportation (Kissinger-van de Wetering): full_reduce run so that only the location and
    value of phases change, never the graph structure, then read back as a circuit.  The output keeps
    the input's Hadamard layering.  Clifford corrections are separated exactly as in floor-only mode."""
    import pyzx as zx
    from pyzx.optimize import Optimizer
    g = zx.simplify.teleport_reduce(c.to_graph())     # returns the reduced graph; it does not act in place
    c2 = zx.Circuit.from_graph(g).to_basic_gates()
    c2 = expand_swaps(c2)
    body, _ = Optimizer(c2).parse_circuit(separate_correction=True, quiet=True)
    return c2, body


def run_circuit(path, compute_exact=True, optimize=True, seed=0, optimizer="todd"):
    import pyzx as zx
    from pyzx.optimize import Optimizer
    # PyZX's TODD shuffles matrix columns with the stdlib `random` (pyzx/todd.py), and its routing uses
    # numpy's generator; seeding both per circuit makes every row of the log reproducible and
    # independent of the order circuits are run in.
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    c = expand_swaps(load_qc(path).to_basic_gates())
    t_in = tcount(c)
    t0 = time.time()
    if optimize and optimizer == "teleport":
        out, body = teleport_optimize(c)
    elif optimize:
        out = zx.optimize.phase_block_optimize(c, pre_optimize=True, quiet=True)
        body = core(out)
    else:
        # floor-only mode: price the INPUT's own layering (after PyZX's basic gate cancellation, which
        # pushes NOT/SWAP corrections to the end); no T-count optimizer is run.
        body, _ = Optimizer(c).parse_circuit(separate_correction=True, quiet=True)
        out = body
    dt = time.time() - t0
    t1 = time.time()
    blocks = price_blocks(partition(body), compute_exact)
    tf = time.time() - t1
    return dict(name=os.path.basename(path)[:-3], qubits=c.qubits, t_in=t_in, t_out=tcount(out),
                blocks=len(blocks), certified=sum(r["cert"] for r in blocks),
                sum_floor=sum(r["floor"] for r in blocks), sum_best=sum(r["given"] for r in blocks),
                sum_nu=sum(r["nu"] for r in blocks),
                cert_nu=sum(r["cert_nu"] for r in blocks),
                t_cert=sum(r["given"] for r in blocks if r["cert"]),
                block_records=[{k: r[k] for k in ("active", "floor", "nu", "dstar", "r", "given", "delta",
                                                  "cert", "cert_nu", "pure")} for r in blocks],
                max_gap=max([r["given"] - r["floor"] for r in blocks] or [0]),
                exact_checked=sum(1 for r in blocks if r["delta"] is not None),
                exact_is_floor=sum(1 for r in blocks if r["delta"] is not None and r["delta"] == r["floor"]),
                exact_is_best=sum(1 for r in blocks if r["delta"] is not None and r["delta"] == r["given"]),
                t_floor=tf, t_pyzx=dt)


def selftest():
    # CCZ: pure cubic, floor 7 = delta
    terms = {}
    import validate_amdist as AM
    coeff = {}
    AM.ccz_on(coeff, (0, 1, 2))
    for y, c in coeff.items():
        terms[y] = c % 8
    r = floor_from_parities(terms, 3)
    assert r["floor"] == 7 and r["nu"] == 3, r
    assert exact_delta(terms, 3) == 7
    # two disjoint CCZ: floor 13 = delta
    coeff = {}
    AM.ccz_on(coeff, (0, 1, 2)); AM.ccz_on(coeff, (3, 4, 5))
    terms = {y: c % 8 for y, c in coeff.items()}
    r = floor_from_parities(terms, 6)
    assert r["floor"] == 13 and exact_delta(terms, 6) == 13, r
    # T on one qubit: floor 1
    r = floor_from_parities({1: 1}, 1)
    assert r["floor"] == 1, r
    # CS: floor 2*1-... check floor <= delta
    coeff = {}; AM.cs_on(coeff, (0, 1))
    terms = {y: c % 8 for y, c in coeff.items()}
    r = floor_from_parities(terms, 2)
    assert r["floor"] <= exact_delta(terms, 2), r
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max-qubits", type=int, default=40)
    ap.add_argument("--max-t", type=int, default=1500)
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--no-optimize", action="store_true", help="price the input layering; skip TODD")
    ap.add_argument("--blocks-out", default=None, help="append per-block records (JSON lines) here")
    ap.add_argument("--seed", type=int, default=0, help="seed for PyZX's TODD column shuffle (per circuit)")
    ap.add_argument("--optimizer", choices=["todd", "teleport"], default="todd",
                    help="todd: phase_block_optimize; teleport: teleport_reduce (layering-preserving)")
    ap.add_argument("--check-teleport", action="store_true",
                    help="tensor-compare input and teleport output on three small circuits, then exit")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    if a.check_teleport:
        import pyzx as zx
        for n in ("tof_3", "mod5_4", "barenco_tof_3"):
            c = expand_swaps(load_qc(os.path.join(BENCH, n + ".qc")).to_basic_gates())
            out, body = teleport_optimize(c)
            assert zx.compare_tensors(c, out, preserve_scalar=False), f"teleport output differs on {n}"
            print(f"teleport tensor check ok: {n}  T {tcount(c)} -> {tcount(out)}")
        return
    import pyzx as zx
    paths = [os.path.join(BENCH, n + ".qc") for n in a.names] if a.names else sorted(glob.glob(os.path.join(BENCH, "*.qc")))
    print(f"{'circuit':<18s} {'n':>4s} {'T_in':>6s} {'T_pyzx':>6s} {'blocks':>6s} {'cert':>9s} "
          f"{'sumFloor':>8s} {'sumBest':>8s} {'sumNu':>6s} {'maxGap':>6s} {'exact(=floor/=best)':>20s} "
          f"{'floor_s':>8s} {'pyzx_s':>7s} {'certNu':>6s} {'Tcert':>6s}", flush=True)
    for p in paths:
        try:
            c = load_qc(p).to_basic_gates()
        except Exception as e:
            print(f"{os.path.basename(p):<18s} load failed: {e}", flush=True)
            continue
        if c.qubits > a.max_qubits or tcount(c) > a.max_t:
            print(f"{os.path.basename(p)[:-3]:<18s} skipped (n={c.qubits}, T={tcount(c)})", flush=True)
            continue
        try:
            r = run_circuit(p, compute_exact=not a.no_exact, optimize=not a.no_optimize, seed=a.seed,
                            optimizer=a.optimizer)
        except Exception as e:
            print(f"{os.path.basename(p)[:-3]:<18s} failed: {type(e).__name__}: {e}", flush=True)
            continue
        ex = f"{r['exact_checked']}({r['exact_is_floor']}/{r['exact_is_best']})"
        print(f"{r['name']:<18s} {r['qubits']:>4d} {r['t_in']:>6d} {r['t_out']:>6d} {r['blocks']:>6d} "
              f"{r['certified']:>4d}/{r['blocks']:<4d} {r['sum_floor']:>8d} {r['sum_best']:>8d} "
              f"{r['sum_nu']:>6d} {r['max_gap']:>6d} {ex:>20s} {r['t_floor']:>8.2f} {r['t_pyzx']:>7.1f} "
              f"{r['cert_nu']:>6d} {r['t_cert']:>6d}",
              flush=True)
        if a.blocks_out:
            import json
            with open(a.blocks_out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"name": r["name"], "t_out": r["t_out"], "blocks": r["block_records"]}) + "\n")


if __name__ == "__main__":
    main()
