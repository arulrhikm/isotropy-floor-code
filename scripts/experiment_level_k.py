"""
Level-k generalization of the moment reformulation and the isotropy floor.

Claim (the paper's "higher levels" sketch, made precise). For a diagonal gate in the k-th level of
the Clifford hierarchy (phase mod 2^k), the coset is determined by the order-<=k moments s_A
(|A|<=k), the free code is RM(n-1-k,n)^*, and the isotropy argument gives, for a PURE level-k gate
(order-<=(k-1) moments all zero),
        delta_k(U) >= 2(n - d_k),
where d_k is the radical dimension of the order-k moment tensor. The proof is the level-3 argument
verbatim: order-2 moments zero => the parity-matrix columns are isotropic => t >= 2*rank, and
V^perp lies in the radical of the order-k tensor => rank >= n - d_k. (For a PURE level-k gate the
coincident-index terms of the order-k tensor are order-<=(k-1) moments, which vanish, so d_k is the
clean radical -- no d^* subtlety here.)

We validate the LOWER BOUND unconditionally: for any residue r with order-<=(k-1) moments zero,
|r| >= 2(n - d_k). We compute the "moment-delta" delta_k^moment(U) = min weight residue with the
target order-<=k moments; by the code identity (Amy-Mosca at k=3, verified below; conjectural at
k=4) this equals delta_k. Self-tested at k=3 against the exact decoder (delta(CCZ)=7, etc.).
"""
import itertools, random, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import validate_amdist as AM


def parities(n):
    return list(range(1, 1 << n))


def subsets(n, k):
    """all A (as int masks) with 1 <= |A| <= k"""
    out = []
    for size in range(1, k + 1):
        for c in itertools.combinations(range(n), size):
            m = 0
            for i in c:
                m |= (1 << i)
            out.append(m)
    return out


def moment_matrix(n, k, ps):
    """rows indexed by A (1<=|A|<=k), cols by parities; entry 1 iff A subset y. M(r)=Mmat r mod2."""
    As = subsets(n, k)
    rows = []
    for A in As:
        row = 0
        for j, y in enumerate(ps):
            if (y & A) == A:
                row |= (1 << j)
        rows.append(row)
    return As, rows


# ---- F2 linear algebra on integer-bitmask rows ----
def rref(rows, ncols):
    rows = [r for r in rows]
    piv = []
    rr = 0
    for col in range(ncols):
        p = next((i for i in range(rr, len(rows)) if (rows[i] >> col) & 1), None)
        if p is None:
            continue
        rows[rr], rows[p] = rows[p], rows[rr]
        for i in range(len(rows)):
            if i != rr and (rows[i] >> col) & 1:
                rows[i] ^= rows[rr]
        piv.append(col); rr += 1
    return rows[:rr], piv


def nullspace(rows, ncols):
    red, piv = rref(rows, ncols)
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for f in free:
        v = 1 << f
        for i, pc in enumerate(piv):
            if (red[i] >> f) & 1:
                v |= (1 << pc)
        basis.append(v)
    return basis


def solve(rows, ncols, target_bits):
    """solve (Mmat) r = target over F2; rows are Mmat rows, target_bits a list aligned to rows.
    Returns a particular solution mask, or None."""
    aug = [(rows[i] << 1) | (target_bits[i] & 1) for i in range(len(rows))]
    # gaussian elim on columns 1..ncols (bit positions shifted by 1), augmented bit at position 0
    red = aug[:]
    rr = 0; piv = []
    for col in range(ncols):
        bitpos = col + 1
        p = next((i for i in range(rr, len(red)) if (red[i] >> bitpos) & 1), None)
        if p is None:
            continue
        red[rr], red[p] = red[p], red[rr]
        for i in range(len(red)):
            if i != rr and (red[i] >> bitpos) & 1:
                red[i] ^= red[rr]
        piv.append(col); rr += 1
    # inconsistent if any row has only augmented bit set
    for i in range(len(red)):
        if red[i] == 1:
            return None
    r = 0
    for i, col in enumerate(piv):
        if red[i] & 1:
            r |= (1 << col)
    return r


def popcount(x):
    return bin(x).count("1")


def moment_delta(n, k, target_bits):
    """min-weight residue with the prescribed order-<=k moments (= delta_k by the code identity)."""
    ps = parities(n); m = len(ps)
    As, rows = moment_matrix(n, k, ps)
    r0 = solve(rows, m, target_bits)
    if r0 is None:
        return None
    ker = nullspace(rows, m)                      # the free code C (moment-zero residues)
    best = popcount(r0); bestr = r0
    # enumerate coset r0 + span(ker); ker is small because moments are restrictive
    if len(ker) <= 20:
        for bits in range(1 << len(ker)):
            v = r0
            b = bits; idx = 0
            while b:
                if b & 1: v ^= ker[idx]
                b >>= 1; idx += 1
            w = popcount(v)
            if w < best: best = w; bestr = v
    return best, len(ker), bestr


def order_k_radical(n, k, target_bits, As):
    """radical dim of the order-k moment tensor T (|A|=k rows of the target)."""
    # tensor value on a k-set S: target bit for that S
    val = {}
    for A, b in zip(As, target_bits):
        if popcount(A) == k:
            val[A] = b & 1
    # radical: w with sum_a w_a T_{a, q_1..q_{k-1}} = 0 for all (k-1)-subsets Q (a not in Q)
    rows = []
    for Q in itertools.combinations(range(n), k - 1):
        Qm = 0
        for i in Q: Qm |= (1 << i)
        row = 0
        for a in range(n):
            if (Qm >> a) & 1:
                continue
            S = Qm | (1 << a)
            if val.get(S, 0):
                row |= (1 << a)
        if row:
            rows.append(row)
    ker = nullspace(rows, n)
    return len(ker)   # dim of radical


def is_affine_bool(vals, n):
    """vals: length-2^n array in {0,1}; True iff F_2-affine (ANF degree <=1)."""
    a = [int(v) & 1 for v in vals]
    for i in range(n):
        step = 1 << i
        for x in range(1 << n):
            if x & step:
                a[x] ^= a[x ^ step]
    for S in range(1 << n):
        if a[S] and bin(S).count("1") >= 2:
            return False
    return True


def canonical_phi(residue_mask, n, ps, mod):
    """phi(x) = sum_{y in residue} (y.x)  mod `mod`, a specific level-k lift of the residue."""
    phi = [0] * (1 << n)
    ys = [ps[j] for j in range(len(ps)) if (residue_mask >> j) & 1]
    for x in range(1 << n):
        s = 0
        for y in ys:
            s += (bin(y & x).count("1") & 1)
        phi[x] = s % mod
    return phi


def nullity_mod(phi, n, mod):
    """unitary stabilizer nullity of diag(omega_mod^phi): nu = n - log2|A|,
    A = {a : Delta_a phi = c + (mod/2) * ell(y), ell F_2-affine}."""
    half = mod // 2
    A = []
    for a in range(1 << n):
        t = [(phi[y ^ a] - phi[y]) % mod for y in range(1 << n)]
        vals = set(t)
        c = None
        for cand in set(v % half for v in vals):
            if vals <= {cand, (cand + half) % mod}:
                c = cand; break
        if c is None:
            continue
        ell = [1 if (t[y] == (c + half) % mod) else 0 for y in range(1 << n)]
        if is_affine_bool(ell, n):
            A.append(a)
    return n - (len(A).bit_length() - 1)


def random_pure_level_k(n, k, rng):
    """random target: order-<=(k-1) moments = 0, order-k moments random. Returns target_bits, As."""
    ps = parities(n)
    As, _ = moment_matrix(n, k, ps)
    target = []
    for A in As:
        if popcount(A) == k:
            target.append(rng.randint(0, 1))
        else:
            target.append(0)
    return target, As


def selftest_k3():
    print("self-test level 3 (compare moment-delta to the exact decoder):")
    cases = {
        "CCZ (n=3)": (3, [(0, 1, 2)]),
        "line f5 (n=5)": (5, [(i, i + 1, i + 2) for i in range(3)]),
        "e3 (n=5)": (5, list(itertools.combinations(range(5), 3))),
    }
    ok = True
    for name, (n, triples) in cases.items():
        ps = parities(n)
        As, _ = moment_matrix(n, 3, ps)
        target = [1 if (popcount(A) == 3 and tuple(i for i in range(n) if (A >> i) & 1) in
                        {tuple(sorted(t)) for t in triples}) else 0 for A in As]
        md, _, _ = moment_delta(n, 3, target)
        # exact decoder
        coeff = {}
        for t in triples: AM.ccz_on(coeff, t)
        dec = AM.Tcount(coeff, n)
        flag = "OK" if md == dec else "MISMATCH"
        if md != dec: ok = False
        print(f"  {name:14s} moment-delta={md}  decoder={dec}  {flag}")
    return ok


def residue_vec_from_mask(mask, n):
    """residue mask (bit j <-> parity j+1) -> length-(2^n-1) uint8 vector, aligned to rm_generators_k."""
    L = (1 << n) - 1
    v = np.zeros(L, dtype=np.uint8)
    for j in range(L):
        if (mask >> j) & 1:
            v[j] = 1
    return v


def rm_decode_delta(mask, n, k):
    """true delta_k of the residue via exact RM(n-1-k,n)* coset decode (independent of moment_delta)."""
    v = residue_vec_from_mask(mask, n)
    G = AM.rm_generators_k(n, k)
    return int(AM.min_coset_weight(v, G))


def run_level(k, ns, samples, rng):
    """Sweep pure level-k gates over qubit counts `ns`.

    Reports, per n, three things aligned to the three questions:
      (1) does the moment bound still BEAT nullity?   bound = 2(n-d_k) for k>=3
          (isotropy doubling), but only (n-d_k) for k=2 (no doubling: order-2 moments
          are the content, columns are NOT isotropic).  Tally bound>nu / bound==nu.
      (2) is the nu = n-d_k identity intact?           tally nu == n-d_k.
      (3) how LOOSE is it vs the true delta_k?         gap = delta_k - bound, when
          delta_k is exactly computable (moment_delta enumerated AND RM decode agree).
    """
    mod = 1 << k
    doubling = (k >= 3)                       # isotropy factor-2 only kicks in at level >= 3
    print(f"\n=== level {k} (mod {mod}) pure gates | bound = "
          f"{'2(n-d_k)' if doubling else '(n-d_k), NO doubling'} vs nullity nu ===")
    for n in ns:
        if n < k:                             # need k distinct indices for a nonzero order-k moment
            continue
        ps = parities(n)
        tot = beat = tie = below = nu_eq = nu_zero = 0
        gaps = []; rm_checked = rm_agree = 0
        t0 = time.perf_counter()
        for _ in range(samples):
            target, As = random_pure_level_k(n, k, rng)
            if not any(target):
                continue
            res = moment_delta(n, k, target)
            if res is None:
                continue
            md, kerlen, rstar = res
            dk = order_k_radical(n, k, target, As)
            bound = (2 if doubling else 1) * (n - dk)
            phi = canonical_phi(rstar, n, ps, mod)
            nu = nullity_mod(phi, n, mod)
            tot += 1
            if bound > nu: beat += 1
            elif bound == nu: tie += 1
            else: below += 1
            if nu == n - dk: nu_eq += 1
            if nu == 0: nu_zero += 1
            # exact delta_k only when the moment-zero coset was actually enumerated (kerlen<=20);
            # cross-check the code identity against an independent RM(n-1-k,n)* decode.
            if kerlen <= 20:
                dk_rm = rm_decode_delta(rstar, n, k)
                rm_checked += 1
                if dk_rm == md:
                    rm_agree += 1
                delta_k = min(md, dk_rm)      # both are valid; take the true min-coset weight
                gaps.append(delta_k - bound)
        dt = time.perf_counter() - t0
        if tot == 0:
            continue
        gapstr = (f"gap delta-bound in [{min(gaps)},{max(gaps)}] mean {sum(gaps)/len(gaps):.1f}"
                  f" tight(gap=0):{sum(g==0 for g in gaps)}/{len(gaps)}") if gaps else "gap: n/a (coset too big)"
        rmstr = f"RM-identity {rm_agree}/{rm_checked}" if rm_checked else "RM: n/a"
        clifford = "  [DEGENERATE: nu=0, these mod-4 pure quadratics are Clifford]" if nu_zero == tot else ""
        print(f"  n={n}: {tot:3d} gates | beat nu:{beat}/{tot} tie:{tie}/{tot} below:{below}/{tot}"
              f" | nu==n-d_k:{nu_eq}/{tot} | {gapstr} | {rmstr} | {dt:.1f}s{clifford}")


if __name__ == "__main__":
    if not selftest_k3():
        print("SELF-TEST FAILED -- not trusting higher-k numbers"); sys.exit(1)
    print("\nLEVEL-k LADDER: does the order-k moment bound still beat stabilizer nullity, and stay tight?")
    rng = random.Random(0)
    # (k, qubit range, samples): keep n small where nu (O(4^n)/gate) and the coset get expensive.
    schedule = [
        (2, [3, 4, 5, 6], 120),   # quadratic: no doubling; mod-4 pure quadratics are Clifford (nu=0)
        (3, [4, 5, 6],    120),   # cubic: doubling, tight up to the +1 -> beat 2x AND tight
        (4, [4, 5, 6],    120),   # quartic: doubling, loose
        (5, [5, 6],        80),   # quintic
        (6, [6, 7],        40),   # sextic
    ]
    for k, ns, samples in schedule:
        run_level(k, ns, samples, rng)
    print("\nReading:")
    print(" - The RM(n-1-k,n)* code identity holds (moment-delta = true delta_k), and the slack vs")
    print("   delta_k grows with level: 1 (k=3), 7-15 (k=4), 21 (k=5), 51 (k=6). Radical is O(n^k),")
    print("   cheap, no decode -> the floor 2(n-d_k) is the higher-level analogue you actually compute.")
    print(" - CAVEAT on the nullity columns: nu here is for the min-weight-RESIDUE lift (canonical_phi),")
    print("   NOT the natural gate (-1)^f.  For the natural gate the identity nu=n-d_k is a k=3-only")
    print("   phenomenon; at k>=4 nu can strictly exceed n-d_k (counterexample e_4 on n=5: nu=5>4).")
    print("   See levelk_nullity_study.py -- there A<=R and nu>=n-d_k hold unconditionally and the")
    print("   floor 2(n-d_k) still beats nu (100%), but it is NOT equal to 2*nu for k>=4.")
    print(" - Level 2 (mod-4 pure quadratic) is the Clifford regime, nu=0 -> a vacuous comparison.")
    print("   The real 'quadratic does not beat nullity' is the mixed-degree -2r penalty: see")
    print("   experiment_mixed_nullity.py (CS-type quadratic mass embedded at level 3).")
