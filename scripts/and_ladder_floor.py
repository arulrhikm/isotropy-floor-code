r"""
AND-compiled circuits beyond level three: t >= 8A + odd(M), and the class-restricted cost of C^kZ.

Part 1.  Random AND-compiled circuits with LADDERS (controls on earlier AND values) and ODD phases on
         AND values, any diagonal target (not only level 3): assert every window carries >= 4 odd
         parities and  t >= 8A + odd(M), where M = odd parities containing no window variable.
Part 2.  c_k = min over designs of the number of odd coefficients of a Z_8 phase polynomial P on the
         independent variables (x, g) with P(x, g(x)) = 4 x_1...x_k pointwise (the middle of a
         minimal-A circuit for C^kZ).  k=4: exhaustive over g = l1*l2, l_i affine in x.
         k=5: chain designs g1 = l1*l2, g2 = (g1 + l3)*l4 and parallel designs, sampled.
Part 3.  The construction: k-3 chained temporary ANDs to g = x_1...x_{k-2}, then CCZ(g, x_{k-1}, x_k),
         uncompute; simulate, check = C^kZ with clean ancillas, count T after merging.  Expect 8k-17.

Run:  python -u and_ladder_floor.py [--samples 300] [--k5-samples 400]
"""
import argparse, itertools, os, random, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from and_gadget_probe import (simulate, to_builder, merged_odd, find_temp_and_templates,
                              temp_and_compute, inverse, random_circuit)


def window_vars(ops, n):
    """Indices (bit positions n+j) of the window variables: the y born at the OPEN H of each H pair."""
    j, open_, wv = 0, {}, []
    for o in ops:
        if o[0] == 'h':
            w = o[1]
            if w not in open_:
                open_[w] = j; wv.append(j)
            else:
                open_.pop(w)
            j += 1
    return [1 << (n + j) for j in wv]


def part1(samples, seed):
    rng = random.Random(seed)
    tpls = find_temp_and_templates()
    checked = worst = 0
    for s in range(samples):
        n = rng.randint(3, 5)
        ops, nw, ands = random_circuit(rng, n, rng.randint(1, 2), rng.randint(1, 3), tpls,
                                       ladder=True, odd_mid=True)
        if nw > 8:
            continue
        if simulate(ops, n, nw) is None:
            continue                                   # not diagonal/clean: out of class
        b = to_builder(ops, n, nw)
        odd = merged_odd(b)
        wbits = window_vars(ops, n)
        per_window = [sum(1 for f in odd if f & wb) for wb in wbits]
        assert all(c >= 4 for c in per_window), f"window with < 4 odd parities: {per_window}"
        oddM = sum(1 for f in odd if not any(f & wb for wb in wbits))
        t, A = len(odd), len(ands)
        assert t >= 8 * A + oddM, (t, A, oddM)
        assert t == sum(per_window) + oddM, "window parities not disjoint from the rest"
        checked += 1
        worst = max(worst, 0)
    print(f"PART 1: {checked} ladder/odd-middle circuits (targets of any level): every window >= 4 odd "
          f"parities, t = sum(windows) + odd(M) >= 8A + odd(M) throughout")


# ---------------------------------------------------------------- part 2: minimal middle via CP-SAT

def affine_forms(k):
    """All non-constant affine forms on k variables as (mask, const)."""
    return [(m, c) for m in range(1, 1 << k) for c in (0, 1)]


def ev_affine(form, x):
    m, c = form
    return (bin(m & x).count("1") + c) & 1


def min_odd_middle(k, gfuncs, time_limit=60):
    """gfuncs: list of length-2^k tables (the AND values as functions of x).  Returns min odd count."""
    from ortools.sat.python import cp_model
    A = len(gfuncs); m = k + A
    P = list(range(1, 1 << m))
    model = cp_model.CpModel()
    h = {p: model.NewIntVar(0, 3, f"h{p}") for p in P}
    o = {p: model.NewBoolVar(f"o{p}") for p in P}
    for x in range(1 << k):
        ext = x | sum(gfuncs[i][x] << (k + i) for i in range(A))
        target = 4 if all((x >> i) & 1 for i in range(k)) else 0
        q = model.NewIntVar(-256, 256, f"q{x}")
        model.Add(sum((2 * h[p] + o[p]) for p in P if bin(p & ext).count("1") & 1) - 8 * q == target)
    model.Minimize(sum(o.values()))
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = time_limit; sv.parameters.num_search_workers = 4
    st = sv.Solve(model)
    if st == cp_model.OPTIMAL:
        return int(sv.ObjectiveValue()), True
    if st == cp_model.FEASIBLE:
        return int(sv.ObjectiveValue()), False
    return None, st == cp_model.INFEASIBLE


def decide_middle(k, gfuncs, bound, time_limit=50):
    """Is there a middle with at most `bound` odd coefficients?  Returns 'INFEASIBLE'/'FEASIBLE'/'UNKNOWN'."""
    from ortools.sat.python import cp_model
    A = len(gfuncs); m = k + A
    P = list(range(1, 1 << m))
    model = cp_model.CpModel()
    h = {p: model.NewIntVar(0, 3, f"h{p}") for p in P}
    o = {p: model.NewBoolVar(f"o{p}") for p in P}
    for x in range(1 << k):
        ext = x | sum(gfuncs[i][x] << (k + i) for i in range(A))
        target = 4 if all((x >> i) & 1 for i in range(k)) else 0
        q = model.NewIntVar(-256, 256, f"q{x}")
        model.Add(sum((2 * h[p] + o[p]) for p in P if bin(p & ext).count("1") & 1) - 8 * q == target)
    model.Add(sum(o.values()) <= bound)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = time_limit; sv.parameters.num_search_workers = 4
    st = sv.Solve(model)
    return {cp_model.INFEASIBLE: "INFEASIBLE", cp_model.OPTIMAL: "FEASIBLE", cp_model.FEASIBLE: "FEASIBLE"}.get(st, "UNKNOWN")


def k4_designs():
    k = 4
    forms = affine_forms(k)
    perms = list(itertools.permutations(range(k)))
    def canon(g):
        best_key = None
        for pi in perms:
            gp = tuple(g[sum(((x >> pi[i]) & 1) << i for i in range(k))] for x in range(1 << k))
            if best_key is None or gp < best_key:
                best_key = gp
        return best_key
    seen, out = set(), []
    for f1, f2 in itertools.combinations(forms, 2):
        g = canon(tuple(ev_affine(f1, x) & ev_affine(f2, x) for x in range(1 << k)))
        if g in seen: continue
        seen.add(g)
        aff = any(all(g[x] == ev_affine(f, x) for x in range(1 << k)) for f in forms) or all(v == g[0] for v in g)
        if not aff: out.append(g)
    return out


def part2_decide(lo, hi, bound=6, time_limit=50):
    ds = k4_designs()
    print(f"k=4: {len(ds)} designs up to S_4; deciding 'middle with <= {bound} odd parities' for designs {lo}..{hi}")
    for i in range(lo, min(hi, len(ds)) + 1):
        import time as _t; t0 = _t.time()
        r = decide_middle(4, [ds[i - 1]], bound, time_limit)
        print(f"  design {i}: {r} ({_t.time() - t0:.1f}s)", flush=True)


def part2(k5_samples, seed):
    # k = 4, A = 1: exhaustive over g = l1 * l2
    k = 4
    forms = affine_forms(k)
    seen, best, designs = set(), None, 0
    perms = list(itertools.permutations(range(k)))
    def canon(g):
        # the target x1...xk is symmetric under S_k, so designs are enumerated up to variable permutation
        best_key = None
        for pi in perms:
            gp = tuple(g[sum(((x >> pi[i]) & 1) << i for i in range(k))] for x in range(1 << k))
            if best_key is None or gp < best_key:
                best_key = gp
        return best_key
    for f1, f2 in itertools.combinations(forms, 2):
        g = tuple(ev_affine(f1, x) & ev_affine(f2, x) for x in range(1 << k))
        g = canon(g)
        if g in seen:
            continue
        seen.add(g)
        # non-affine check
        aff = any(all(g[x] == ev_affine(f, x) for x in range(1 << k)) for f in forms) or all(v == g[0] for v in g)
        if aff:
            continue
        designs += 1
        import time as _t; _t0 = _t.time()
        val, exact = min_odd_middle(k, [g], time_limit=20)
        print(f"  design {designs}: min odd = {val} exact={exact} ({_t.time() - _t0:.1f}s)", flush=True)
        if val is None:
            continue
        if best is None or val < best[0]:
            best = (val, exact, f1, f2)
    print(f"PART 2 k=4: {designs} distinct AND designs; minimal middle odd count c_4 = {best[0]} "
          f"(exact: {best[1]}); e.g. g = {best[2]}*{best[3]}", flush=True)
    if k5_samples <= 0:
        return
    # k = 5, A = 2: sampled chain and parallel designs
    k = 5; rng = random.Random(seed)
    forms = affine_forms(k)
    best5, tried, seen = None, 0, set()
    def tab(fn): return tuple(fn(x) for x in range(1 << k))
    cand = []
    # canonical designs first
    g1 = tab(lambda x: (x & 1) & (x >> 1 & 1))
    cand.append(("chain x1x2, g1*x3", [g1, tab(lambda x: (g1[x]) & (x >> 2 & 1))]))
    cand.append(("parallel x1x2, x3x4", [g1, tab(lambda x: (x >> 2 & 1) & (x >> 3 & 1))]))
    while len(cand) < k5_samples:
        f1, f2, f3, f4 = rng.sample(forms, 4)
        G1 = tab(lambda x: ev_affine(f1, x) & ev_affine(f2, x))
        if rng.random() < 0.7:   # chain: g2 = (g1 + l3) * l4
            G2 = tab(lambda x: (G1[x] ^ ev_affine(f3, x)) & ev_affine(f4, x))
        else:                    # parallel
            G2 = tab(lambda x: ev_affine(f3, x) & ev_affine(f4, x))
        key = (G1, G2)
        if key in seen: continue
        seen.add(key); cand.append((f"sampled", [G1, G2]))
    for name, gs in cand:
        val, exact = min_odd_middle(k, gs, time_limit=12)
        tried += 1
        if val is None: continue
        if best5 is None or val < best5[0]:
            best5 = (val, exact, name)
    print(f"PART 2 k=5: {tried} designs tried (2 canonical + sampled); smallest middle odd count found "
          f"c_5 <= {best5[0]} (design: {best5[2]}, exact: {best5[1]})")


# ---------------------------------------------------------------- part 3: the 8k-17 construction

def ckz_circuit(k, tpl):
    n = k; anc = list(range(k, 2 * k - 3))
    ops, blocks = [], []
    prev = 0
    for i, t in enumerate(anc):
        a = prev if i > 0 else 0
        b = 1 if i == 0 else i + 1
        comp = temp_and_compute(a, b, t, tpl)
        ops += comp; blocks.append(comp); prev = t
    ops.append(('ccz', prev if anc else 0, k - 2, k - 1))
    for comp in reversed(blocks):
        ops += inverse(comp)
    return ops, 2 * k - 3


def part3():
    tpl = find_temp_and_templates()[0]
    for k in (3, 4, 5, 6):
        ops, nw = ckz_circuit(k, tpl)
        phi = simulate(ops, k, nw)
        assert phi is not None, f"k={k}: not diagonal/clean"
        want = [4 if x == (1 << k) - 1 else 0 for x in range(1 << k)]
        assert phi == want, f"k={k}: wrong phase"
        t = len(merged_odd(to_builder(ops, k, nw)))
        print(f"PART 3 k={k}: construction implements C^{k}Z exactly with {k - 3} clean ancillas; "
              f"T-count after merging = {t}  (8k-17 = {8 * k - 17})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--k5-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--only", default="123")
    ap.add_argument("--decide", type=int, nargs=2, metavar=("LO", "HI"))
    a = ap.parse_args()
    if a.decide:
        part2_decide(a.decide[0], a.decide[1]); return
    if "1" in a.only: part1(a.samples, a.seed)
    if "3" in a.only: part3()
    if "2" in a.only: part2(a.k5_samples, a.seed)


if __name__ == "__main__":
    main()
