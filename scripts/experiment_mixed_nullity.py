"""
Experiment: does the nullity identity extend to MIXED-degree level-three gates?

For a pure-cubic gate nu = n-d and delta >= 2nu+1 (prop:nullity, thm:cubicLB). This experiment
asks what happens for a general level-three gate with cubic radical dimension d and
quadratic-matrix rank r.
Question: for mixed gates (cubic + CS quadratic + T linear), what is nu, and does delta >= 2nu hold?

Analysis (to be checked): phi = 4*cubic + 2*quad + lin. A shift a is in the nullity group A iff
Delta_a phi(y) is constant mod 4 (kills the 2*quad derivative -> a in ker Q) AND the remaining
4*(cubic derivative) is 4*affine (-> a in cubic radical R). So the prediction is
    A = R  intersect  ker Q,   nu = n - dim(R cap ker Q).
This is generally NOT n-d-r. We test nu against both candidates, and test delta >= 2 nu.
"""
import itertools, random, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from verify_nullity import in_A
import validate_amdist as AM
from experiment_nu_eq_n_minus_d import radical_set, f2_rank


def nullspace_set(rows, n):
    """{w in F_2^n : row . w = 0 for all rows}, as a set of int masks (rows are bitmasks)."""
    mat = [r for r in rows if r]
    pivots, rr = [], 0
    for col in range(n):
        piv = next((i for i in range(rr, len(mat)) if (mat[i] >> col) & 1), None)
        if piv is None:
            continue
        mat[rr], mat[piv] = mat[piv], mat[rr]
        for i in range(len(mat)):
            if i != rr and (mat[i] >> col) & 1:
                mat[i] ^= mat[rr]
        pivots.append(col); rr += 1
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for fcol in free:
        w = 1 << fcol
        for i, pcol in enumerate(pivots):
            if (mat[i] >> fcol) & 1:
                w |= (1 << pcol)
        basis.append(w)
    S = {0}
    for b in basis:
        S |= {x ^ b for x in S}
    return S


def qmatrix_rows(pairs, lin, n):
    """rows of the symmetric quadratic moment matrix Q: Q_ab=[{a,b} in pairs], Q_aa=[a in lin]."""
    Q = [[0]*n for _ in range(n)]
    for (a, b) in pairs:
        Q[a][b] = Q[b][a] = 1
    for a in lin:
        Q[a][a] = 1
    rows = [sum((Q[i][j] & 1) << j for j in range(n)) for i in range(n)]
    return rows


def true_moments(coeff, n):
    """Compute the actual gate syndromes s_A (|A|<=3) from the residue, per Amy--Mosca.
    Returns (s3_triples set, Qrows) where s_ab = sum over parities y>={a,b} of residue[y]."""
    res = AM.residue_vec(coeff, n)
    ys = AM.parities(n)
    def M(A):
        m = 0
        for idx, y in enumerate(ys):
            if (y & A) == A and res[idx]:
                m ^= 1
        return m
    s3 = [tuple(sorted(t)) for t in itertools.combinations(range(n), 3)
          if M((1 << t[0]) | (1 << t[1]) | (1 << t[2]))]
    Q = [[0]*n for _ in range(n)]
    for a in range(n):
        Q[a][a] = M(1 << a)                       # order-1 syndrome s_a
        for b in range(a+1, n):
            Q[a][b] = Q[b][a] = M((1 << a) | (1 << b))   # order-2 syndrome s_ab
    Qrows = [sum((Q[i][j] & 1) << j for j in range(n)) for i in range(n)]
    return s3, Qrows


def build_phi(triples, pairs, lin, n):
    coeff = {}
    for t in triples: AM.ccz_on(coeff, t)
    for p in pairs:   AM.cs_on(coeff, p)
    for i in lin:     AM.t_on(coeff, i)
    return coeff, AM.phase_from_coeff(coeff, n)


def nu_of(phi, n):
    A = [a for a in range(1 << n) if in_A(phi, a, n)]
    return n - (len(A).bit_length() - 1), set(A)


def dprime(coeff, n):
    """Radical dimension of the FULL third-power moment tensor T'_{qra}=s_{{q,r,a}}, where
    coincident indices collapse to order-<=2 moments. This is the radical the thm:genLB proof
    actually controls (the step sum_a w_a c_q c_r c_a folds c_q^2=c_q onto the diagonal). For a
    pure-cubic gate the order-<=2 moments vanish, so T'=T and d'=d."""
    res = AM.residue_vec(coeff, n); ys = AM.parities(n)
    def M(A):
        m = 0
        for idx, y in enumerate(ys):
            if (y & A) == A and res[idx]: m ^= 1
        return m
    rows = []
    for q in range(n):
        for r in range(n):
            row = 0
            for a in range(n):
                if M((1 << q) | (1 << r) | (1 << a)): row |= (1 << a)
            if row: rows.append(row)
    return len(nullspace_set(rows, n)).bit_length() - 1


def run(ns, samples, do_delta_upto=5, seed=3):
    rng = random.Random(seed)
    for n in ns:
        T3 = list(itertools.combinations(range(n), 3))
        T2 = list(itertools.combinations(range(n), 2))
        eqA = eqNDR = dbl = genlb = tot = 0
        for _ in range(samples):
            triples = rng.sample(T3, rng.randint(0, len(T3)))
            pairs   = rng.sample(T2, rng.randint(0, len(T2)))
            lin     = [i for i in range(n) if rng.random() < 0.5]
            if not (triples or pairs or lin):
                continue
            coeff, phi = build_phi(triples, pairs, lin, n)
            s3, Qrows = true_moments(coeff, n)     # d, r from the ACTUAL gate syndromes
            R, d = radical_set(s3, n)
            r = f2_rank(Qrows, n)
            kerQ = nullspace_set(Qrows, n)
            RcapK = R & kerQ
            nu, A = nu_of(phi, n)
            tot += 1
            if A == RcapK: eqA += 1
            if nu == n - d - r: eqNDR += 1
            genlbp = 0
            if n <= do_delta_upto:
                dlt = AM.Tcount(coeff, n)
                dp = dprime(coeff, n)
                if dlt >= 2 * nu: dbl += 1
                if dlt >= 2 * (n - d - r): genlb += 1
                if dlt >= 2 * (n - dp - r): genlbp += 1
                run.gp = run.__dict__.get('gp', 0) + genlbp
        line = f"n={n}: {tot} gates | doubling delta>=2nu: {dbl}/{tot}"
        if n <= do_delta_upto:
            line += (f" | genLB as stated delta>=2(n-d-r): {genlb}/{tot}"
                     f" | genLB corrected delta>=2(n-d'-r): {run.__dict__.get('gp',0)}/{tot}")
        print(line)
        run.gp = 0


def moment_bound(coeff, n):
    """the general (mixed-degree) isotropy bound 2(n - d' - r), thm:genLB (corrected d')."""
    _, Qrows = true_moments(coeff, n)
    r = f2_rank(Qrows, n)
    dp = dprime(coeff, n)
    return 2 * (n - dp - r), dp, r


def peel_bound(triples, pairs, lin, n):
    """The 'peeling' lower bound: split the phase 4 f3 + 2 f2 + f1 into its degree parts and use
    subadditivity of delta (lem:subadd).  Since  f3-gate = U . (f2-gate)^-1 . (f1-gate)^-1,
        delta(f3-gate) <= delta(U) + delta(f2-gate) + delta(f1-gate),
    hence   delta(U) >= delta(f3) - delta(f2) - delta(f1).
    The cubic part carries the strong floor 2(n-d3)+1; the lower parts are cheap when sparse.  We
    subtract the EXACT delta of each lower part (an over-estimate would also be valid), so the bound
    is sound.  Returns (peel, d3, delta3, delta2, delta1)."""
    c3, _ = build_phi(triples, [], [], n)     # cubic gate  omega^{4 f3}
    c2, _ = build_phi([], pairs, [], n)       # quad gate   omega^{2 f2}
    c1, _ = build_phi([], [], lin, n)         # linear gate omega^{f1}
    delta3 = AM.Tcount(c3, n) if triples else 0
    delta2 = AM.Tcount(c2, n) if pairs else 0
    delta1 = len(lin)                         # distinct singletons: delta(f1) = #odd linear terms
    # cubic radical d3 (of the cubic-only tensor) for reporting
    s3, _ = true_moments(c3, n)
    _, d3 = radical_set(s3, n)
    peel = delta3 - delta2 - delta1
    return peel, d3, delta3, delta2, delta1


def cs_layer_demo(ms=(1, 2, 3)):
    """The clean 'quadratic mass -> nullity wins' example: m disjoint CS gates on n=2m qubits.
    Pure quadratic (embedded at level 3: CS = 2xy mod 8), so the -2r penalty zeroes the moment
    bound while stabilizer nullity nu = n is a valid floor.  Shows why the combined estimator
    max(nu, moment-bound) is what is reported."""
    print("CS-layer (m disjoint CS on n=2m): moment bound collapses, nullity wins")
    for m in ms:
        n = 2 * m
        pairs = [(2 * i, 2 * i + 1) for i in range(m)]
        coeff, phi = build_phi([], pairs, [], n)
        nu, _ = nu_of(phi, n)
        mb, dp, r = moment_bound(coeff, n)
        dlt = AM.Tcount(coeff, n) if n <= 6 else None
        combined = max(nu, mb)
        ds = "n/a" if dlt is None else str(dlt)
        print(f"  m={m} n={n}: nu={nu:2d}  moment-bound 2(n-d'-r)={mb:2d} (r={r},d'={dp})"
              f"  max(nu,mb)={combined:2d}  delta={ds:>3s}  <- nullity beats the moment bound")


def crossover_sweep(ns=(4, 5), samples=400, seed=7):
    """Map the crossover: as quadratic mass (rank r) grows, the moment bound 2(n-d'-r) falls
    below stabilizer nullity nu.  Bin outcomes by r; also check the combined estimator
    max(nu, 2(n-d'-r)) is always a valid floor (<= delta) and how often it strictly beats nu."""
    rng = random.Random(seed)
    for n in ns:
        T3 = list(itertools.combinations(range(n), 3))
        T2 = list(itertools.combinations(range(n), 2))
        by_r = {}                                   # r -> [mb<nu, mb==nu, mb>nu]
        comb_valid = comb_beats_nu = tot = 0
        for _ in range(samples):
            triples = rng.sample(T3, rng.randint(0, len(T3)))
            pairs   = rng.sample(T2, rng.randint(0, len(T2)))
            lin     = [i for i in range(n) if rng.random() < 0.5]
            if not (triples or pairs or lin):
                continue
            coeff, phi = build_phi(triples, pairs, lin, n)
            nu, _ = nu_of(phi, n)
            mb, dp, r = moment_bound(coeff, n)
            dlt = AM.Tcount(coeff, n)
            tot += 1
            b = by_r.setdefault(r, [0, 0, 0])
            b[0 if mb < nu else (1 if mb == nu else 2)] += 1
            combined = max(nu, mb)
            if combined <= dlt: comb_valid += 1
            if combined > nu:   comb_beats_nu += 1
        print(f"n={n}: {tot} mixed gates | combined max(nu,mb) valid (<=delta): {comb_valid}/{tot}"
              f" | combined strictly beats nu: {comb_beats_nu}/{tot}")
        print(f"       crossover by quadratic rank r  (mb=2(n-d'-r)):")
        for r in sorted(by_r):
            lo, eq, hi = by_r[r]
            print(f"         r={r}: mb<nu {lo:3d} | mb=nu {eq:3d} | mb>nu {hi:3d}"
                  f"   (of {lo+eq+hi})")


def peel_interval_sweep(ns=(4, 5), samples=500, seed=13):
    """The peel is TWO-SIDED: subadditivity gives both
        delta(U) >= delta(f3) - delta(f2) - delta(f1)     (lower)
        delta(U) <= delta(f3) + delta(f2) + delta(f1)     (upper),
    so  delta(U) is pinned to  delta(f3) +/- eps,  eps = delta(f2)+delta(f1).  When the lower-order
    part is sparse, eps is small and delta(U) is an eps-APPROXIMATION of the cubic cost delta(f3).
    We measure eps and the ACTUAL gap g = delta(U) - delta(f3): the two-sided bound guarantees
    |g| <= eps, and we report how much smaller |g| usually is (i.e. how close delta(U) is to the
    cubic value in practice)."""
    rng = random.Random(seed)
    for n in ns:
        T3 = list(itertools.combinations(range(n), 3))
        T2 = list(itertools.combinations(range(n), 2))
        tot = interval_ok = tight_pin = 0
        sum_eps = sum_absg = 0
        exactly_cubic = 0            # delta(U) == delta(f3)
        for _ in range(samples):
            triples = rng.sample(T3, rng.randint(1, len(T3)))
            pairs = rng.sample(T2, rng.randint(0, max(1, len(T2) // 3)))
            lin = [i for i in range(n) if rng.random() < 0.3]
            coeff, phi = build_phi(triples, pairs, lin, n)
            dU = AM.Tcount(coeff, n)
            c3, _ = build_phi(triples, [], [], n); d3 = AM.Tcount(c3, n)
            c2, _ = build_phi([], pairs, [], n); d2 = AM.Tcount(c2, n) if pairs else 0
            d1 = len(lin)
            eps = d2 + d1
            g = dU - d3
            tot += 1
            if -eps <= g <= eps: interval_ok += 1
            if eps <= 2: tight_pin += 1        # eps<=2: delta(U) within 2 of the cubic value
            if g == 0: exactly_cubic += 1
            sum_eps += eps; sum_absg += abs(g)
        print(f"n={n}: {tot} gates | two-sided interval holds (|g|<=eps): {interval_ok}/{tot}")
        print(f"   mean eps (bound width) = {sum_eps/tot:.2f} | mean |delta(U)-delta(f3)| = "
              f"{sum_absg/tot:.2f}   (actual gap is {sum_absg/max(1,sum_eps)*100:.0f}% of the bound)")
        print(f"   delta(U) == delta(f3) exactly: {exactly_cubic}/{tot}"
              f" | pinned within 2 of cubic (eps<=2): {tight_pin}/{tot}")


def peel_levelk_sweep(ks=(4,), n_extra=1, samples=300, seed=17):
    """Level-k peel: for a level-k gate with phase 2^{k-1} f_k + (lower), subadditivity gives
        delta_k(U) >= delta_k(top) - delta_k(lower),
    the top-degree floor minus the lower-order cost.  We test it on random pure-top-plus-sparse-lower
    level-k gates (k=4) where the exact level-k decoder is feasible (small n)."""
    import experiment_level_k as LK
    rng = random.Random(seed)
    for k in ks:
        n = k + n_extra
        if n > 6:
            continue
        ps = LK.parities(n)
        As, _ = LK.moment_matrix(n, k, ps)
        mod = 1 << k
        ok = valid = tot = 0
        for _ in range(samples):
            # random residue: some order-k parities + a few lower-order ones
            topmask = 0
            for j, A in enumerate(As):
                if LK.popcount(A) == k and rng.random() < 0.4:
                    topmask |= (1 << j)
            lowmask = 0
            for j, A in enumerate(As):
                if 1 <= LK.popcount(A) < k and rng.random() < 0.15:
                    lowmask |= (1 << j)
            if topmask == 0:
                continue
            full = topmask | lowmask
            dfull, _, _ = LK.moment_delta(n, k, [(full >> j) & 1 for j in range(len(As))])
            dtop, _, _ = LK.moment_delta(n, k, [(topmask >> j) & 1 for j in range(len(As))])
            dlow, _, _ = LK.moment_delta(n, k, [(lowmask >> j) & 1 for j in range(len(As))]) if lowmask else (0, 0, 0)
            tot += 1
            if dfull >= dtop - dlow: valid += 1
            if dfull >= dtop: ok += 1     # does the top floor alone already lower-bound? (usually not)
        print(f"level k={k}, n={n}: {tot} gates | peel delta_k>=delta(top)-delta(low) valid: {valid}/{tot}"
              f" | top alone would suffice: {ok}/{tot}")


def peel_sweep(ns=(4, 5), samples=500, seed=11):
    """Test the peeling bound.  For each mixed gate compute the three estimators
        nu,  moment = 2(n-d'-r),  peel = delta(f3) - delta(f2) - delta(f1),
    the current combined  max(nu, moment)  and the new combined  max(nu, moment, peel).
    Check (a) every estimator is a VALID floor (<= delta), and (b) how often peel lifts the combined
    floor strictly above max(nu, moment).  Bin the wins by quadratic rank r (peel should win in the
    high-r / sparse-coupling regime where the moment bound collapses below nu)."""
    rng = random.Random(seed)
    for n in ns:
        T3 = list(itertools.combinations(range(n), 3))
        T2 = list(itertools.combinations(range(n), 2))
        tot = peel_valid = old_valid_c = new_valid_c = peel_lifts = 0
        wins_by_r = {}
        peel_best = 0
        for _ in range(samples):
            triples = rng.sample(T3, rng.randint(0, len(T3)))
            # bias toward SPARSE-but-nonzero quadratic (few couplings), where peel should shine
            pairs = rng.sample(T2, rng.randint(0, max(1, len(T2) // 3)))
            lin = [i for i in range(n) if rng.random() < 0.3]
            if not triples:
                continue
            coeff, phi = build_phi(triples, pairs, lin, n)
            dlt = AM.Tcount(coeff, n)
            nu, _ = nu_of(phi, n)
            mb, dp, r = moment_bound(coeff, n)
            peel, d3, d3g, d2g, d1g = peel_bound(triples, pairs, lin, n)
            tot += 1
            if peel <= dlt: peel_valid += 1
            old_c = max(nu, mb)
            new_c = max(nu, mb, peel)
            if old_c <= dlt: old_valid_c += 1
            if new_c <= dlt: new_valid_c += 1
            if new_c > old_c:
                peel_lifts += 1
                wins_by_r[r] = wins_by_r.get(r, 0) + 1
            if peel == new_c and peel > max(nu, mb):
                peel_best += 1
        print(f"n={n}: {tot} sparse-quadratic mixed gates")
        print(f"   peel valid (<= delta):            {peel_valid}/{tot}")
        print(f"   max(nu,moment) valid:             {old_valid_c}/{tot}")
        print(f"   max(nu,moment,PEEL) valid:        {new_valid_c}/{tot}")
        print(f"   PEEL strictly lifts the floor:    {peel_lifts}/{tot}"
              f"   (peel is the unique best in {peel_best})")
        if wins_by_r:
            print(f"   lifts binned by quadratic rank r: "
                  + ", ".join(f"r={r}:{c}" for r, c in sorted(wins_by_r.items())))


if __name__ == "__main__":
    print("mixed-degree nullity experiment (cubic + CS quadratic + T linear)")
    print("findings: (1) the cubic doubling delta>=2nu FAILS for mixed gates;")
    print("          (2) genLB as stated with d=radical(cubic tensor) can fail;")
    print("          (3) with d'=radical(full third-power tensor s_{{q,r,a}}) it holds.\n")
    run(ns=[4, 5], samples=500)

    print("\n--- LOW END: when quadratic mass makes NULLITY the better floor ---")
    cs_layer_demo()
    print()
    crossover_sweep()
    print("\nReading: the moment bound 2(n-d'-r) pays -2r for quadratic mass and gets nothing back,")
    print("so it drops below nu as r grows (crossover); nu never doubles here. Report the combined")
    print("max(nu, 2(n-d'-r), rigidity) -- always valid, and it beats plain nu on cubic-dominated gates.")

    print("\n--- PEELING BOUND: delta(U) >= delta(f3) - delta(f2) - delta(f1) (Q2) ---")
    print("Complements the moment bound: peel pays the quadratic/linear T-COST, not the rank, so it")
    print("wins in the sparse-but-high-rank regime where 2(n-d'-r) falls below nullity.")
    peel_sweep()

    print("\n--- PEEL IS TWO-SIDED: delta(U) in [delta(f3)-eps, delta(f3)+eps], eps=delta(f2)+delta(f1) ---")
    print("So delta(U) is an eps-approximation of the cubic cost; sparse lower orders => tight pin.")
    peel_interval_sweep()

    print("\n--- LEVEL-k PEEL: delta_k(U) >= delta(top-degree part) - delta(lower parts) ---")
    peel_levelk_sweep()
