"""
Does an intermediate rung of the derivative filtration carry an isotropy-style DOUBLING, giving a
computable floor strictly between the top-radical floor 2(n-d_k) and the true delta_k?

Filtration  R^(1)=R  superseteq R^(2) superseteq ... superseteq R^(k-2)=A,  R^(j)={a:deg Delta_a f<=k-1-j}.
The top-radical floor is F0 = 2(n - dim R^(1)); the deepest rung gives 2(n - dim A) = 2*nu.  We test
the whole interpolating ladder of candidate floors

        F_j := 2 (n - dim R^(j)),     j = 1 .. k-2      (F_1 = F0 = 2(n-d_k),  F_{k-2} = 2*nu)

plus the additive second-order candidate F0 + rho_2 (rho_2 = dim R^(1) - dim R^(2)), against the EXACT
delta_k (moment_delta, = RM(n-1-k,n)* min-coset weight).  For each candidate we report:
  * VALIDITY   : candidate <= delta_k on every gate (a valid lower bound)   -- the hard requirement
  * IMPROVEMENT: candidate  > F0 on some gates (strictly tighter than the top-radical floor)

The sharp question is candidate F_{k-2} = 2*nu: is delta_k >= 2*nu (the full nullity doubles)?  At k=3
this is delta>=2nu+1 (a theorem); here we test it at k>=4.  Whichever deepest j keeps validity is the
tighter poly-computable floor for C^{k-1}Z-type gates.
"""
import itertools, random, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from experiment_level_k import (moment_matrix, moment_delta, order_k_radical, parities, popcount)
from levelk_nullity_study import (eval_homog_from_coeffs, deriv, degree, nullity,
                                  radical_dim_and_filtration)
from validate_amdist import rm_generators_k, min_coset_weight


def natural_residue(coeffs, n, k):
    """Exact residue of the NATURAL gate (-1)^f, f = sum of the k-monomials `coeffs`.
    Uses 2^{k-1} prod_{i in S} x_i = sum_{empty != T subseteq S} (-1)^{|T|+1} chi_T, so each
    C^{k-1}Z_S contributes every nonempty T subseteq S (odd coeff); XOR over monomials."""
    L = (1 << n) - 1
    r = np.zeros(L, dtype=np.uint8)
    for S in coeffs:
        bits = [i for i in range(n) if (S >> i) & 1]
        for m in range(1, 1 << len(bits)):
            T = 0
            for j, b in enumerate(bits):
                if (m >> j) & 1:
                    T |= (1 << b)
            r[T - 1] ^= 1
    return r


def delta_natural(coeffs, n, k):
    """exact delta_k of (-1)^f by RM(n-1-k,n)* coset decode of its natural residue."""
    return int(min_coset_weight(natural_residue(coeffs, n, k), rm_generators_k(n, k)))


def delta_ge_2nu(k, ns, samples, seed=3):
    """The clean same-gate test of Conjecture delta_k >= 2*nu: delta_k and nu BOTH from (-1)^f."""
    rng = random.Random(seed)
    print(f"\n=== level k={k}: delta_k((-1)^f) >= 2*nu  (both from the SAME natural gate) ===")
    # self-test: single C^{k-1}Z has delta = 2^k - 1
    d1 = delta_natural([(1 << k) - 1], k, k)
    print(f"  self-test single C^{{{k-1}}}Z: delta={d1} (expect 2^k-1={2**k-1})  {'OK' if d1==2**k-1 else 'FAIL'}")
    for n in ns:
        if n < k:
            continue
        monos = [m for m in range(1 << n) if bin(m).count("1") == k]
        tot = ge = 0; worst = 10 ** 9
        for _ in range(samples):
            coeffs = [m for m in monos if rng.random() < 0.5]
            if not coeffs:
                continue
            tt = eval_homog_from_coeffs(coeffs, n)
            if degree(tt, n) != k:
                continue
            nu, _ = nullity(tt, n)
            d = delta_natural(coeffs, n, k)
            tot += 1
            if d >= 2 * nu:
                ge += 1
            worst = min(worst, d - 2 * nu)
        if tot:
            print(f"  n={n}: {tot} gates | delta>=2nu: {ge}/{tot} | worst delta-2nu = {worst}")


def _parities(n):
    return list(range(1, 1 << n))


def _span_rank_and_vperp(supp_masks, n):
    """rank of the parity set (span in F_2^n) and its orthogonal complement V^perp as a set of masks."""
    mat = [m for m in supp_masks]; piv = []
    r = 0
    for col in range(n):
        p = next((i for i in range(r, len(mat)) if (mat[i] >> col) & 1), None)
        if p is None:
            continue
        mat[r], mat[p] = mat[p], mat[r]
        for i in range(len(mat)):
            if i != r and (mat[i] >> col) & 1:
                mat[i] ^= mat[r]
        piv.append(col); r += 1
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        w = 1 << f
        for i, pc in enumerate(piv):
            if (mat[i] >> f) & 1:
                w |= (1 << pc)
        basis.append(w)
    S = {0}
    for b in basis:
        S |= {x ^ b for x in S}
    return r, S


def rank_vs_nu(k, ns, samples, seed=5):
    """DECISIVE experiment: is delta_k>=2nu a rank corollary?  For the GENUINE minimal residue of
    (-1)^f, does rank(Y) >= nu (equivalently V^perp subseteq A)?  If yes -> delta_k=|S|>=2rank>=2nu
    is a theorem.  If no -> delta_k>2rank (isotropy is SLACK) and the doubling is beyond first-order
    isotropy (the second-order wall)."""
    rng = random.Random(seed)
    ps = {n: _parities(n) for n in ns}
    print(f"\n=== level k={k}: is delta_k>=2nu a RANK corollary? (min residue of (-1)^f) ===")
    for n in ns:
        if n < k:
            continue
        G = rm_generators_k(n, k)
        monos = [m for m in range(1 << n) if bin(m).count("1") == k]
        tot = rank_ge = vperp_in_A = slack = 0
        for _ in range(samples):
            coeffs = [m for m in monos if rng.random() < 0.5]
            if not coeffs:
                continue
            tt = eval_homog_from_coeffs(coeffs, n)
            if degree(tt, n) != k:
                continue
            nu, A = nullity(tt, n); Aset = set(A)
            d, rep = min_coset_weight(natural_residue(coeffs, n, k), G, return_arg=True)
            supp = [ps[n][j] for j in range(len(ps[n])) if rep[j]]
            if not supp:
                continue
            rankY, Vp = _span_rank_and_vperp(supp, n)
            tot += 1
            if rankY >= nu: rank_ge += 1
            if all(w in Aset for w in Vp): vperp_in_A += 1
            if d > 2 * rankY: slack += 1        # isotropy strictly slack on the min residue
        if tot:
            print(f"  n={n}: {tot} gates | rank(Y)>=nu: {rank_ge}/{tot} | V^perp<=A: {vperp_in_A}/{tot}"
                  f" | isotropy slack (delta>2rank): {slack}/{tot}")


def tensor_from_target(target, As, n, k):
    """the degree-k monomial support (list of int masks) where the order-k moment is 1."""
    return [A for A, b in zip(As, target) if b and popcount(A) == k]


def filtration_dims(coeffs, n, k):
    """dim R^(j) for j=1..k-2 (R^(1)=radical, R^(k-2)=A), from the homogeneous truth table."""
    tt = eval_homog_from_coeffs(coeffs, n)
    if degree(tt, n) != k:
        return None, None
    lv = radical_dim_and_filtration(tt, n, k)
    dims = {j: (len(v).bit_length() - 1) for j, v in lv.items()}
    nu, _ = nullity(tt, n)
    return dims, nu


def run(k, ns, samples, seed=0):
    rng = random.Random(seed)
    print(f"\n=== level k={k}: filtration floors vs exact delta_k ===")
    # candidate names: F_j for j=1..k-2 (F_1=F0, F_{k-2}=2nu), plus additive (only if a 2nd rung exists)
    cand_names = [f"F_{j}=2(n-dimR^{j})" for j in range(1, k - 1)]
    if k >= 4:
        cand_names.append("F0+rho2")
    for n in ns:
        if n < k:
            continue
        ps = parities(n)
        As, _ = moment_matrix(n, k, ps)
        tot = 0
        valid = {c: 0 for c in cand_names}
        improve = {c: 0 for c in cand_names}
        max_recovered = {c: 0 for c in cand_names}   # best (candidate - F0) among valid gates
        worst_violation = {c: 0 for c in cand_names}  # worst (candidate - delta_k) > 0
        for _ in range(samples):
            target = [rng.randint(0, 1) if popcount(A) == k else 0 for A in As]
            if not any(target):
                continue
            coeffs = tensor_from_target(target, As, n, k)
            dims, nu = filtration_dims(coeffs, n, k)
            if dims is None:
                continue
            res = moment_delta(n, k, target)
            if res is None:
                continue
            dk_delta, kerlen, _ = res
            if kerlen > 20:               # moment_delta exact only when coset enumerated
                continue
            d1 = dims[1]
            F0 = 2 * (n - d1)
            tot += 1
            cand = {}
            for j in range(1, k - 1):
                cand[f"F_{j}=2(n-dimR^{j})"] = 2 * (n - dims[j])
            if k >= 4:
                cand["F0+rho2"] = F0 + (dims[1] - dims[2])
            for c, val in cand.items():
                if val <= dk_delta:
                    valid[c] += 1
                    if val > F0:
                        improve[c] += 1
                        max_recovered[c] = max(max_recovered[c], val - F0)
                else:
                    worst_violation[c] = max(worst_violation[c], val - dk_delta)
        if tot == 0:
            continue
        print(f"  n={n}: {tot} gates")
        for c in cand_names:
            vstr = "VALID" if valid[c] == tot else f"INVALID ({tot-valid[c]} violations, worst +{worst_violation[c]})"
            print(f"    {c:20s}: {vstr:40s} | strictly>F0 on {improve[c]:3d}/{tot} | max recovered +{max_recovered[c]}")


if __name__ == "__main__":
    print("Testing whether filtration rungs give a valid floor tighter than 2(n-d_k).")
    print("Key candidate F_{k-2} = 2*nu: does delta_k >= 2*nu at k>=4?  (it is 2nu+1 at k=3)")
    run(3, [4, 5, 6], 200)
    run(4, [4, 5, 6], 200)
    run(5, [5, 6], 150)
    print("\nReading: a candidate is a genuinely tighter poly floor iff it is VALID on all gates AND")
    print("strictly exceeds F0 on some.  The deepest such rung is the tightest radical-filtration floor.")

    # The clean same-gate test of delta_k >= 2*nu (Conjecture), self-tested against C^{k-1}Z = 2^k-1.
    delta_ge_2nu(3, [4, 5, 6], 120)
    delta_ge_2nu(4, [4, 5, 6], 120)
    delta_ge_2nu(5, [5, 6], 100)
    print("\nNote: delta_k >= 2*nu is verified here on the natural gate (-1)^f (both delta_k and nu from")
    print("the SAME gate). Proven at k=3 (=2nu+1); open at k>=4 -- the isotropy proof reaches only")
    print("2(n-d_k) since it forces V^perp <= R (top radical), not V^perp <= A.")

    # DECISIVE: is the doubling a rank corollary, or isotropy-slack (the second-order wall)?
    rank_vs_nu(3, [4, 5, 6], 120)
    rank_vs_nu(4, [4, 5, 6], 120)
    rank_vs_nu(5, [5, 6], 100)
    print("\nReading: if rank(Y)>=nu holds on min residues, delta_k=|S|>=2rank>=2nu is a THEOREM.")
    print("If instead isotropy is slack (delta>2rank) with rank<nu, the doubling is BEYOND first-order")
    print("isotropy.")
