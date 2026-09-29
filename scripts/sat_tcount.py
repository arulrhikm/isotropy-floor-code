"""
Exact T-count of a diagonal level-three gate as a small XOR-constrained min-weight problem.

By fact:moment, two residues lie in the same RM(n-4,n)* coset iff all their moments of order
<= 3 agree.  The counting is exact, not merely sound:

    dim RM(n-4,n)* = sum_{d<=n-4} C(n,d) = 2^n - 1 - ( n + C(n,2) + C(n,3) ),

so the order-<=3 moment map F_2^{2^n-1} -> F_2^{n + C(n,2) + C(n,3)} is surjective with kernel
EXACTLY the free code.  Hence

    delta(U)  =  min { wt(z) : z in F_2^{2^n-1},  XOR_{y >= A} z_y = m_A  for all 1<=|A|<=3 }.

That is 2^n - 1 boolean variables under only n + C(n,2) + C(n,3) XOR constraints -- 92 at n=8,
129 at n=9, 175 at n=10 -- which a CDCL/CP-SAT solver eats, whereas the meet-in-the-middle coset
search in validate_amdist.py enumerates 2^k codewords and dies at n=6 (k=22; k=64 at n=7).

Backend: OR-Tools CP-SAT (native AddBoolXOr + cardinality).  Falls back to python-sat.

Because this needs a solver, it runs under its own interpreter; see SATENV below or pass
--python.  Validated against validate_amdist.Tcount on every self-test and on random gates
at n<=6 (run `python sat_tcount.py --validate`).
"""
import argparse, itertools, sys, os, time

# ---------------------------------------------------------------- moment bookkeeping

def moment_keys(n):
    """All A subseteq [n] with 1 <= |A| <= 3, as bitmasks, in a fixed order."""
    keys = []
    for k in (1, 2, 3):
        for A in itertools.combinations(range(n), k):
            m = 0
            for i in A:
                m |= 1 << i
            keys.append(m)
    return keys


def moments_from_residue(res, n):
    """m_A = XOR over parities y >= A of res[y-1].  res is a 0/1 sequence of length 2^n-1."""
    out = {}
    for A in moment_keys(n):
        acc = 0
        for y in range(1, 1 << n):
            if (y & A) == A and res[y - 1]:
                acc ^= 1
        out[A] = acc
    return out


def moments_from_cubic(triples, n):
    """A pure cubic form f = sum_{abc in triples} x_a x_b x_c, gate (-1)^f = omega^{4f}.
    Its order-3 moments are the coefficients of f; orders 1 and 2 vanish (pure-cubic)."""
    out = {A: 0 for A in moment_keys(n)}
    for t in triples:
        m = 0
        for i in t:
            m |= 1 << i
        out[m] ^= 1          # XOR, not assign: a repeated monomial must cancel
    return out


def moments_from_quadratic(pairs, n):
    """A pure quadratic (CS-level) gate omega^{2 q(x)}, q = sum_{ab in pairs} x_a x_b.
    Order-2 moments are the coefficients of q; orders 1 and 3 vanish."""
    out = {A: 0 for A in moment_keys(n)}
    for p in pairs:
        m = (1 << p[0]) | (1 << p[1])
        out[m] ^= 1          # XOR, not assign: a repeated edge must cancel
    return out


def moments_from_coeff(coeff, n):
    """Moments of a gate built from validate_amdist's ccz_on/cs_on/t_on coefficient dict."""
    res = [0] * ((1 << n) - 1)
    for y, c in coeff.items():
        if c % 2:
            res[y - 1] ^= 1
    return moments_from_residue(res, n)


def weight_of_moment_free_rep(moments, n):
    """A trivial feasible point: the 'canonical' residue that sets z_A = m_A on the |A|<=3
    parities and 0 elsewhere is NOT generally feasible (m_A also counts supersets), so we
    solve the triangular system top-down to get a valid starting upper bound."""
    z = {}
    for k in (3, 2, 1):
        for A in itertools.combinations(range(n), k):
            m = 0
            for i in A:
                m |= 1 << i
            acc = moments[m]
            for y, v in z.items():
                if v and (y & m) == m and y != m:
                    acc ^= 1
            z[m] = acc
    return sum(z.values()), [y for y, v in z.items() if v]


# ---------------------------------------------------------------- CP-SAT backend

def _solve_cpsat(moments, n, ub=None, lb=None, timeout=None, workers=8, log=False,
                 decision_weight=None):
    from ortools.sat.python import cp_model
    L = (1 << n) - 1
    model = cp_model.CpModel()
    z = [model.NewBoolVar(f"z{y}") for y in range(1, 1 << n)]

    supersets = {A: [] for A in moment_keys(n)}
    for y in range(1, 1 << n):
        for A in supersets:
            if (y & A) == A:
                supersets[A].append(z[y - 1])
    for A, lits in supersets.items():
        if moments[A]:
            model.AddBoolXOr(lits)
        else:
            model.AddBoolXOr([lits[0].Not()] + lits[1:])   # XOR(!l0,l1..)=1 <=> XOR(l0,..)=0

    total = sum(z)
    if decision_weight is not None:
        model.Add(total <= decision_weight)
    else:
        if ub is not None:
            model.Add(total <= ub)
        if lb is not None:
            model.Add(total >= lb)
        model.Minimize(total)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    if timeout:
        solver.parameters.max_time_in_seconds = float(timeout)
    solver.parameters.log_search_progress = bool(log)
    status = solver.Solve(model)
    name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        witness = [y for y in range(1, 1 << n) if solver.Value(z[y - 1])]
        val = len(witness)
        # for the optimisation form, a FEASIBLE (timed-out) answer is only an upper bound
        return val, name, witness, int(solver.BestObjectiveBound()) if decision_weight is None else None
    return None, name, None, None


def tcount(moments, n, ub=None, timeout=None, workers=8, log=False):
    """Exact delta.  Returns (delta, status, witness_parities).  status=='OPTIMAL' means proved."""
    if ub is None:
        ub, _ = weight_of_moment_free_rep(moments, n)
    val, name, witness, bound = _solve_cpsat(moments, n, ub=ub, timeout=timeout,
                                             workers=workers, log=log)
    return val, name, witness


def tcount_at_most(moments, n, w, timeout=None, workers=8):
    """Decision form: is there a representation with <= w T-gates?  Returns (bool_or_None, status).
    None means the solver timed out without deciding.  Often far faster than optimising."""
    val, name, witness, _ = _solve_cpsat(moments, n, timeout=timeout, workers=workers,
                                         decision_weight=w)
    if name in ("OPTIMAL", "FEASIBLE"):
        return True, name
    if name == "INFEASIBLE":
        return False, name
    return None, name


# ---------------------------------------------------------------- validation

def validate(max_n=6, samples=150, seed=5):
    """Agree with the meet-in-the-middle decoder on every self-test and on random gates."""
    import random
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import validate_amdist as AM

    print("== named self-tests (must match validate_amdist) ==")
    named = []
    c = {}; AM.ccz_on(c, (0, 1, 2)); named.append(("CCZ n=3", c, 3, 7))
    c = {}; AM.cs_on(c, (0, 1)); named.append(("CS n=3", c, 3, 3))
    c = {}; AM.t_on(c, 0); named.append(("T n=3", c, 3, 1))
    c = {}; AM.ccz_on(c, (0, 1, 2)); named.append(("CCZ x I n=4", c, 4, 7))
    c = {}; AM.ccz_on(c, (0, 1, 2)); named.append(("CCZ x I^2 n=5", c, 5, 7))
    # non-additivity (thm:nonadd): delta(CCZ (+) T) = 7 = delta(CCZ), NOT 8
    c = {}; AM.ccz_on(c, (0, 1, 2)); AM.t_on(c, 3); named.append(("CCZ+T n=5 (thm:nonadd)", c, 5, 7))
    c = {}; AM.ccz_on(c, (0, 1, 2)); AM.ccz_on(c, (3, 4, 5))
    named.append(("CCZ+CCZ n=6 (thm:nonadd)", c, 6, 13))
    c = {}; AM.cs_on(c, (0, 1)); AM.cs_on(c, (2, 3)); named.append(("CS+CS n=4", c, 4, 6))
    c = {}
    for t in ((0, 1, 2), (2, 3, 4), (4, 5, 0)): AM.ccz_on(c, t)
    named.append(("cyclic cubic n=6", c, 6, 15))
    bad = 0
    for name, c, n, expect in named:
        mm = moments_from_coeff(c, n)
        d, st, _ = tcount(mm, n)
        ref = AM.Tcount(c, n)
        ok = (d == ref == expect) and st == "OPTIMAL"
        bad += not ok
        print(f"   {name:34s} sat={d:3d} [{st}]  mitm={ref:3d}  expect={expect:3d}"
              f"   {'ok' if ok else '*** MISMATCH ***'}")

    print("\n== random gates ==")
    rng = random.Random(seed)
    for n in range(3, max_n + 1):
        T3 = list(itertools.combinations(range(n), 3))
        T2 = list(itertools.combinations(range(n), 2))
        mism = 0
        t0 = time.time()
        for _ in range(samples):
            triples = rng.sample(T3, rng.randint(0, len(T3)))
            pairs = rng.sample(T2, rng.randint(0, len(T2)))
            lin = [i for i in range(n) if rng.random() < 0.5]
            if not (triples or pairs or lin):
                continue
            c = {}
            for t in triples: AM.ccz_on(c, t)
            for p in pairs: AM.cs_on(c, p)
            for i in lin: AM.t_on(c, i)
            mm = moments_from_coeff(c, n)
            d, st, _ = tcount(mm, n)
            ref = AM.Tcount(c, n)
            if d != ref or st != "OPTIMAL":
                mism += 1
                print(f"   *** n={n} sat={d}[{st}] mitm={ref}  triples={triples} pairs={pairs} lin={lin}")
        bad += mism
        print(f"   n={n}: {samples} gates, {mism} mismatches   ({time.time()-t0:.1f}s)")
    print("\nVALIDATION", "PASSED" if bad == 0 else f"FAILED ({bad} problems)")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--samples", type=int, default=150)
    args = ap.parse_args()
    if args.validate:
        sys.exit(0 if validate(args.max_n, args.samples) else 1)
    print(__doc__)
