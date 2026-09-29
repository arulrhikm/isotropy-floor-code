"""
Level-k nullity vs. moment radical, for the NATURAL gate U = (-1)^{f(x)} with f homogeneous
of degree k (the C^{k-1}Z-type finest gate: k=3 is CCZ, k=4 is C^3Z, ...).

This corrects experiment_level_k.py, whose nullity was computed on a min-weight-residue lift
(`canonical_phi`) rather than on the natural gate (-1)^f.  The two are different unitaries with
the same order-<=k moments but different stabilizer nullity, so the earlier "nu = n-d_k, 100%"
was a lift artifact.

What is actually true (established here):
  * A subseteq R  ALWAYS  ==>  nu >= n - d_k   (nullity is at least the moment-radical codim).
  * R subseteq A  (hence nu = n-d_k)  is TRUE at k=3, FALSE for k>=4.
    Clean counterexample: f = e_4 (all degree-4 monomials) on n=5.  a = all-ones lies in R
    (kills the top slice) but Delta_a f = 1 + e_2(y) has degree 2, so a not in A; nu = 5 > n-d_4 = 4.
  * So at k>=4 the moment radical and nullity DIVERGE.  Neither dominates a priori:
    the isotropy floor is 2(n-d_k); nullity is a separate floor nu >= n-d_k that can exceed it.
    Report max(nu, 2(n-d_k)).

A radical filtration R^{(1)} superseteq ... superseteq A explains the gap: R^{(j)} = directions a
with deg(Delta_a f) <= k-1-j.  R^{(1)}=R (top slice killed), and A = R^{(k-2)} (killed down to
affine).  nu = n - dim A.  The isotropy floor only uses R^{(1)}; nullity uses the deepest level.
"""
import itertools, random, sys, os
from math import comb

sys.path.insert(0, os.path.dirname(__file__))


# ---- Boolean f as truth table; ANF via Mobius; homogeneous degree-k generator ----
def anf(tt, n):
    a = tt[:]
    for i in range(n):
        st = 1 << i
        for x in range(1 << n):
            if x & st:
                a[x] ^= a[x ^ st]
    return a


def degree(tt, n):
    a = anf(tt, n)
    return max((bin(S).count("1") for S in range(1 << n) if a[S]), default=-1)


def eval_homog_from_coeffs(coeffs, n):
    """truth table of f = sum_{S in coeffs} prod_{i in S} x_i."""
    tt = [0] * (1 << n)
    for x in range(1 << n):
        v = 0
        for S in coeffs:
            if (x & S) == S:
                v ^= 1
        tt[x] = v
    return tt


def random_homog(n, k, rng):
    monos = [m for m in range(1 << n) if bin(m).count("1") == k]
    chosen = [m for m in monos if rng.random() < 0.5]
    return eval_homog_from_coeffs(chosen, n), chosen


def elem_symm(n, k):
    monos = [m for m in range(1 << n) if bin(m).count("1") == k]
    return eval_homog_from_coeffs(monos, n), monos


def deriv(tt, a, n):
    return [tt[y ^ a] ^ tt[y] for y in range(1 << n)]


def radical_dim_and_filtration(tt, n, k):
    """R^{(j)} = {a : deg(Delta_a f) <= k-1-j}, j=1..k-2 (so R^{(1)}=R, R^{(k-2)}=A).
    Each is a subspace; return dims via counting (small n)."""
    # membership sets
    levels = {}
    for j in range(1, k - 1):
        thr = k - 1 - j
        levels[j] = [a for a in range(1 << n) if degree(deriv(tt, a, n), n) <= thr]
    return levels


def nullity(tt, n):
    A = [a for a in range(1 << n) if degree(deriv(tt, a, n), n) <= 1]
    return n - (len(A).bit_length() - 1), A


# ---- cheap top-tensor radical directly from the degree-k monomial support ----
def _f2_nullspace_dim(rows, ncols):
    """dim of right nullspace of a 0/1 matrix given as integer-bitmask rows, over F2."""
    rows = [r for r in rows if r]
    piv = 0
    for col in range(ncols):
        p = next((i for i in range(piv, len(rows)) if (rows[i] >> col) & 1), None)
        if p is None:
            continue
        rows[piv], rows[p] = rows[p], rows[piv]
        for i in range(len(rows)):
            if i != piv and (rows[i] >> col) & 1:
                rows[i] ^= rows[piv]
        piv += 1
    return ncols - piv


def top_radical_dim(coeffs, n, k):
    """dim of the radical R = {a : T(a,.)=0} of the order-k tensor T (support = coeffs),
    computed directly (poly time): rows indexed by (k-1)-subsets W, column i set iff W|{i} in coeffs.
    R is the right nullspace; d_k = n - rank."""
    cset = set(coeffs)
    rows = []
    for W in itertools.combinations(range(n), k - 1):
        Wm = 0
        for i in W:
            Wm |= (1 << i)
        row = 0
        for i in range(n):
            if not ((Wm >> i) & 1) and ((Wm | (1 << i)) in cset):
                row |= (1 << i)
        if row:
            rows.append(row)
    return _f2_nullspace_dim(rows, n)


# ---- adversarial families designed to maximize the radical (the only regime that can flip the sign) ----
def block_esymm(block_sizes, k, n):
    """f = sum over disjoint blocks B of e_k(x_B).  Odd-size blocks contribute an all-ones-of-block
    radical direction with NON-affine derivative (the e_4 mechanism), stacking dim(R\\A)."""
    coeffs = []
    base = 0
    for b in block_sizes:
        idx = list(range(base, base + b))
        for c in itertools.combinations(idx, k):
            m = 0
            for i in c:
                m |= (1 << i)
            coeffs.append(m)
        base += b
    return eval_homog_from_coeffs(coeffs, n), coeffs


def product_plus_esymm(n, k, extra_block, rng):
    """single product x_0..x_{k-1} (huge radical, but invariant) PLUS an e_k on a disjoint block,
    to try to break invariance of some radical directions while keeping the radical large."""
    coeffs = [sum(1 << i for i in range(k))]
    idx = list(range(k, k + extra_block))
    for c in itertools.combinations(idx, k):
        coeffs.append(sum(1 << i for i in c))
    return eval_homog_from_coeffs(coeffs, n), coeffs


def sign_search(rng, k=4):
    """Hunt for a gate with nu > 2(n-d_k) (would flip the sign).  Pre-filter on the cheap radical
    d_k (only d_k>n/2 can flip); pay the O(n*4^n) nullity only for large-radical candidates."""
    print(f"\n=== SIGN SEARCH k={k}: can nu exceed 2(n-d_k)?  (worst margin nu-2(n-d_k)) ===")
    worst = -10 ** 9
    worst_desc = None
    worst_dhalf = -10 ** 9    # worst margin restricted to the d_k>n/2 regime
    tested = big = 0

    def consider(coeffs, n, desc):
        nonlocal worst, worst_desc, worst_dhalf, tested, big
        if not coeffs:
            return
        dk = top_radical_dim(coeffs, n, k)
        # cheap gate object for nullity
        tt = eval_homog_from_coeffs(coeffs, n)
        if degree(tt, n) != k:
            return
        tested += 1
        # only pay nullity when it could matter (small n always; large n only if d_k>n/2)
        if n > 9 and 2 * dk <= n:
            return
        nu, _ = nullity(tt, n)
        margin = nu - 2 * (n - dk)
        if margin > worst:
            worst, worst_desc = margin, f"{desc}: n={n} d_k={dk} nu={nu} 2(n-d_k)={2*(n-dk)}"
        if 2 * dk > n:
            big += 1
            if margin > worst_dhalf:
                worst_dhalf = margin
            if margin > 0:
                print(f"  !!! VIOLATION {desc}: n={n} d_k={dk} nu={nu} > 2(n-d_k)={2*(n-dk)}  coeffs={coeffs}")

    # structured big-radical families
    for n in range(k, 10):
        consider(elem_symm(n, k)[1], n, "e_k")
    for blocks in ([5, 5], [5, 5, 5], [5, 7], [7, 7], [5, 5, 7], [9], [5, 9], [7, 9]):
        n = sum(blocks)
        if n <= 16:
            consider(block_esymm(blocks, k, n)[1], n, f"block_e_k{blocks}")
    for extra in range(k, 7):
        n = k + extra
        consider(product_plus_esymm(n, k, extra, rng)[1], n, f"product+e_k(+{extra})")
    # DANGEROUS-REGIME HUNT: reach d_k>n/2 (needs LOW tensor rank = near a product), then try to
    # break A=R with a small parity perturbation.  A pure single monomial has d_k=n-k>n/2 (safe, A=R).
    for n in range(k + 1, 11):
        mono = sum(1 << i for i in range(k))
        consider([mono], n, "single-monomial")                      # d_k=n-k>n/2, expect A=R (safe)
        # product + a few extra k-monomials sharing structure, to perturb while keeping radical big
        for extra_sets in ([sum(1 << i for i in list(range(1, k)) + [k])],   # share k-1 vars
                            [sum(1 << i for i in list(range(2, k)) + [k, k + 1])] if n >= k + 2 else []):
            if extra_sets:
                consider([mono] + extra_sets, n, "product+few")
    # low-rank sums of products of linear forms (big radical, generic), lightly perturbed
    for n in range(k + 2, 11):
        for _ in range(60):
            # 2-3 random rank-1 degree-k terms (each a product of k independent coordinate-ish forms)
            terms = []
            for _t in range(rng.randint(2, 3)):
                sup = rng.sample(range(n), k)
                terms.append(sum(1 << i for i in sup))
            consider(list(set(terms)), n, "low-rank-sum")
    # random, radical pre-filter; n<=8 to keep the O(n*4^n) nullity affordable
    for n in range(k, 9):
        for _ in range(300):
            coeffs = [m for m in range(1 << n) if bin(m).count("1") == k and rng.random() < 0.5]
            consider(coeffs, n, "random")

    print(f"  tested {tested} gates ({big} with d_k>n/2).")
    print(f"  worst margin nu-2(n-d_k) overall: {worst}   [{worst_desc}]")
    print(f"  worst margin in the d_k>n/2 regime: "
          f"{worst_dhalf if big else 'no d_k>n/2 gate found'}")
    print(f"  => 2(n-d_k) >= nu holds in all tested cases."
          if worst <= 0 else "  => VIOLATION FOUND (see above).")


def in_radical(tt, a, n, k):
    return degree(deriv(tt, a, n), n) <= k - 2


def study(k, ns, samples, rng):
    print(f"\n=== level k={k}: natural gate (-1)^f, f homogeneous deg {k} ===")
    for n in ns:
        if n < k:
            continue
        tot = A_sub_R = nu_ge = nu_eq = mb_beats = 0
        nu_gt_examples = 0
        for _ in range(samples):
            tt, _ = random_homog(n, k, rng)
            if degree(tt, n) != k:      # want genuinely degree-k gates
                continue
            R = [a for a in range(1 << n) if in_radical(tt, a, n, k)]
            dk = (len(R).bit_length() - 1)
            nu, A = nullity(tt, n)
            tot += 1
            if all(a in R for a in A): A_sub_R += 1
            if nu >= n - dk: nu_ge += 1
            if nu == n - dk: nu_eq += 1
            else: nu_gt_examples += 1   # nu > n-dk (strict): radical misses structure
            mb = 2 * (n - dk)
            if mb >= nu: mb_beats += 1
        if tot == 0:
            continue
        print(f"  n={n}: {tot:3d} gates | A<=R:{A_sub_R}/{tot} | nu>=n-d_k:{nu_ge}/{tot}"
              f" | nu==n-d_k:{nu_eq}/{tot} (strict nu>:{nu_gt_examples})"
              f" | moment-floor 2(n-d_k)>=nu:{mb_beats}/{tot}")


if __name__ == "__main__":
    rng = random.Random(0)

    print("SELF-TEST / counterexample: f = e_4 on n=5  (U = prod_{|S|=4} C^3Z_S)")
    tt, _ = elem_symm(5, 4)
    R = [a for a in range(1 << 5) if in_radical(tt, a, 5, 4)]
    dk = len(R).bit_length() - 1
    nu, A = nullity(tt, 5)
    allones = (1 << 5) - 1
    print(f"  d_4 = {dk}  (all-ones in R: {allones in R})")
    print(f"  Delta_(all-ones) f has degree {degree(deriv(tt, allones, 5), 5)}  "
          f"(all-ones in A: {allones in A})")
    print(f"  nu = {nu},  n - d_4 = {5 - dk}   -> identity nu=n-d_k is {'TRUE' if nu==5-dk else 'FALSE'}")
    print(f"  moment floor 2(n-d_4) = {2*(5-dk)},  nullity floor = {nu}  "
          f"-> moment still {'beats' if 2*(5-dk)>=nu else 'LOSES to'} nullity")
    fl = radical_dim_and_filtration(tt, 5, 4)
    dims = {j: (len(v).bit_length() - 1) for j, v in fl.items()}
    print(f"  radical filtration dims (R^(1)=R ... R^(k-2)=A): {dims}")

    print("\nk=3 (CCZ-type): identity nu=n-d SHOULD hold (A=R)")
    study(3, [3, 4, 5, 6], 200, rng)
    print("\nk=4 (C^3Z-type): identity SHOULD break (A subsetneq R possible)")
    study(4, [4, 5, 6], 200, rng)
    print("\nk=5:")
    study(5, [5, 6, 7], 120, rng)

    print("\nReading: A<=R and nu>=n-d_k are UNCONDITIONAL. nu==n-d_k is a k=3 phenomenon; at k>=4")
    print("nullity can strictly exceed n-d_k (radical misses higher-derivative structure). The")
    print("isotropy floor 2(n-d_k) still typically beats nu, but the floor is max(nu, 2(n-d_k)).")

    # SIGN QUESTION: is 2(n-d_k) >= nu ALWAYS?  hunt the only regime that could flip it (d_k>n/2).
    sign_search(rng, k=4)
    sign_search(rng, k=5)
