"""
Stress-test the width-1 kernel inequality   N_ker >= delta(kappa)   (conj:w1).

Width-1 (sole-carrier) realizable kernel data over V = F_2^d:
  - ODD content: each odd parity meets a SINGLE kernel direction i:  c*(v_i XOR l(x)),
    c in {1,3,5,7}, l affine in x.                       (these are the N_ker parities)
  - FORCED EVEN chain couplings (one per swap):  6*(v_i XOR v_{i+1} XOR b(x)).
The correction kappa(x) is read off the fiber Gauss sum  sum_{v in F_2^d} w^{R_x(v)} = M * w^{kappa(x)}.
We require constant modulus (diagonality) and an 8th-root quotient (valid Z_8 phase), then
compute delta(kappa) by the exact Amy-Mosca decoder and check N_ker >= delta(kappa).

NEGATIVE CONTROL: an h=3 four-T CCZ candidate (odd content on one direction, NO forced
coupling) -- it should VIOLATE the inequality (N_ker=4 < delta=7), which is exactly why
realizability (the forced coupling) is needed.
"""
import itertools, cmath, random
import numpy as np
from validate_amdist import residue_vec, rm_generators, min_coset_weight, add_term

W = cmath.exp(1j * cmath.pi / 4)   # omega = e^{i pi/4}

# ---------- kappa truth table -> Z_8 monomial ANF -> Amy-Mosca delta ----------
def mobius_anf(f, n):
    r"""Z_8 monomial ANF: a_S = sum_{T subseteq S} (-1)^{|S\T|} f(1_T) mod 8."""
    a = {}
    for S in range(1 << n):
        bitsS = [i for i in range(n) if (S >> i) & 1]
        tot = 0
        sub = S
        while True:
            t = bin(S ^ sub).count("1")     # |S \ T|
            tot += ((-1) ** t) * f[sub]
            if sub == 0:
                break
            sub = (sub - 1) & S
        a[S] = tot % 8
    return a

def delta_of_kappa(kappa, n):
    """delta(kappa) via Amy-Mosca; kappa is a length-2^n Z_8 table. Returns (delta, ok)."""
    a = mobius_anf(kappa, n)
    coeff = {}
    for S, aS in a.items():
        if aS == 0:
            continue
        w = bin(S).count("1")
        if w == 0:
            continue                      # global phase
        if w == 1:
            add_term(coeff, S, aS)        # aS * x_i  -> parity {i}
        elif w == 2:
            if aS % 2:
                return None, False        # odd coeff on quadratic -> not level-3
            m = (aS // 2) % 4
            i, j = [k for k in range(n) if (S >> k) & 1]
            for _ in range(m):
                add_term(coeff, 1 << i, 1); add_term(coeff, 1 << j, 1)
                add_term(coeff, (1 << i) | (1 << j), -1)
        elif w == 3:
            if aS % 4:
                return None, False        # not a multiple of 4 -> not level-3
            m = (aS // 4) % 2
            trip = [k for k in range(n) if (S >> k) & 1]
            for _ in range(m):
                for r in range(1, 4):
                    for T in itertools.combinations(trip, r):
                        y = 0
                        for i in T: y ^= (1 << i)
                        add_term(coeff, y, (-1) ** (r - 1))
        else:
            return None, False            # degree >= 4 -> not level-3
    r = residue_vec(coeff, n)
    G = rm_generators(n)
    return int(min_coset_weight(r, G)), True

# ---------- evaluate a width-1 config: build R_x, Gauss-sum, extract kappa ----------
def parity(mask, x):
    return bin(mask & x).count("1") & 1

def eval_config(n, d, odd_terms, couplings):
    """
    odd_terms: list of (i, c, amask, e)  -> term c*(v_i XOR (amask.x XOR e))
    couplings: list of (i, bmask, e)     -> term 6*(v_i XOR v_{i+1} XOR (bmask.x XOR e))
    Returns (valid, N_ker, kappa_table) ; kappa_table length 2^n (Z_8) if valid else None.
    """
    NX, NV = 1 << n, 1 << d
    kappa = [0] * NX
    mod0 = None
    for x in range(NX):
        G = 0j
        for v in range(NV):
            R = 0
            for (i, c, amask, e) in odd_terms:
                vi = (v >> i) & 1
                R += c * (vi ^ parity(amask, x) ^ e)
            for (i, bmask, e) in couplings:
                vi = (v >> i) & 1; vj = (v >> (i + 1)) & 1
                R += 6 * (vi ^ vj ^ parity(bmask, x) ^ e)
            G += W ** (R % 8)
        m = abs(G)
        if m < 1e-9:
            return False, 0, None                      # zero amplitude -> not diagonal
        if mod0 is None:
            mod0 = m
            # realizability: |G| = 2^{h/2}, h = d + rho_G, 1 <= rho_G <= d.
            # so m^2 must be a power of two with exponent in [d+1, 2d].
            ex = round(2 * np.log2(m))
            if abs(2 * np.log2(m) - ex) > 1e-6 or not (d + 1 <= ex <= 2 * d):
                return False, 0, None                  # modulus not a valid 2^{h/2}
        elif abs(m - mod0) > 1e-6:
            return False, 0, None                      # non-constant modulus -> not diagonal
        ang = cmath.phase(G) / (cmath.pi / 4)
        k = round(ang) % 8
        if abs(ang - round(ang)) > 1e-6:
            return False, 0, None                      # not an 8th root -> not a Z_8 phase
        kappa[x] = k
    N_ker = len(odd_terms)
    return True, N_ker, kappa

# ---------- an h=3 candidate as a negative control ----------
def candidate_control():
    # 4 odd parities on ONE direction (i=0), sections l1,l2,l3 independent and l4=l1^l2^l3.
    # n=3, l_p = x_p ; l4 = x0^x1^x2.  NO forced coupling.
    n = 3
    odd = [(0, 1, 1 << 0, 0), (0, 1, 1 << 1, 0), (0, 1, 1 << 2, 0),
           (0, 1, (1 << 0) | (1 << 1) | (1 << 2), 0)]
    valid, Nk, kappa = eval_config(n, 1, odd, [])
    if not valid:
        return "candidate: Gauss sum not a valid diagonal phase (d=1 single direction)"
    dk, ok = delta_of_kappa(kappa, n)
    return (f"candidate (no forced coupling): N_ker={Nk}, delta(kappa)={dk}, "
            f"inequality holds? {Nk >= dk}  (expect VIOLATION 4 < 7)")

# ---------- randomized search over realizable width-1 chains ----------
def random_config(n, d, max_odd_per_dir, rng):
    odd = []
    for i in range(d):
        for _ in range(rng.randint(0, max_odd_per_dir)):
            c = rng.choice([1, 3, 5, 7])
            amask = rng.randrange(1 << n)
            e = rng.randint(0, 1)
            odd.append((i, c, amask, e))
    couplings = []
    for i in range(d - 1):
        bmask = rng.randrange(1 << n)
        e = rng.randint(0, 1)
        couplings.append((i, bmask, e))     # forced even coupling at every swap
    return odd, couplings

def search(n, d, trials, rng):
    valid_ct = viol_ct = 0
    worst = None
    max_delta = 0
    dumped = False
    for _ in range(trials):
        odd, coup = random_config(n, d, 3, rng)
        if not odd:
            continue
        valid, Nk, kappa = eval_config(n, d, odd, coup)
        if not valid:
            continue
        dk, ok = delta_of_kappa(kappa, n)
        if not ok:
            continue
        valid_ct += 1
        max_delta = max(max_delta, dk)
        slack = Nk - dk
        if worst is None or slack < worst[0]:
            worst = (slack, Nk, dk)
        if Nk < dk:
            viol_ct += 1
            if not dumped:
                dumped = True
                print(f"   *** VIOLATION n={n} d={d}: N_ker={Nk} < delta(kappa)={dk}")
                print(f"       odd={odd}\n       coup={coup}")
    return valid_ct, viol_ct, max_delta, worst

if __name__ == "__main__":
    print("=" * 70)
    print("NEGATIVE CONTROL (h=3 candidate, no forced coupling):")
    print("  " + candidate_control())
    print()
    print("REALIZABLE width-1 chains (forced even coupling at every swap):")
    rng = random.Random(20260626)
    total_valid = total_viol = 0
    for n in (3, 4):
        for d in (2, 3, 4):
            vc, xc, maxd, worst = search(n, d, 200000, rng)
            total_valid += vc; total_viol += xc
            ws = (f"min slack={worst[0]} at (N_ker={worst[1]},delta={worst[2]})"
                  if worst else "no valid configs")
            print(f"  n={n} d={d}: {vc:5d} valid, {xc} viol, max delta(kappa)={maxd};  {ws}")
    print("=" * 70)
    print(f"TOTAL: {total_valid} valid realizable width-1 configs, {total_viol} violations.")
    print("PASS: N_ker >= delta(kappa) on every realizable width-1 config." if total_viol == 0
          else "FAIL: a realizable width-1 config violated the inequality.")
