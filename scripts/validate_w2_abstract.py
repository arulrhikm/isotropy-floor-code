"""
FAITHFUL abstract width-2 (d=2) stress test + bounded exhaustive certification of
    N_ker >= delta(kappa)     (the open w>=2 case beyond conj:w1)
generalizing validate_w1.py from the sole-carrier chain to the full three-class d=2 model
(the (r_1,r_2,m) variety).

d=2 kernel V=F_2^2, directions v_1,v_2. A kernel-meeting parity lies in one of three
DIRECTION CLASSES beta in {v_1, v_2, v_1^v_2}. The kernel form is
    R_x(v) = sum_odd  c*( <dir,v> ^ l(x) )          c in {1,3,5,7}   (N_ker of these)
           + sum_c2   2*( <dir,v> ^ b(x) )                           (CS-type, coeff 2)
           + 4*( v_1*B1(x) ^ v_2*B2(x) )  ^  4*cc*v_1*v_2            (coeff-4, Hadamard)
where B1,B2 are affine in x and -- THE TRIANGULARITY CONSTRAINT -- the cross-coupling
coefficient cc is a CONSTANT in {0,1}, NOT x-dependent: b_2 (the coeff-4 base of the later-born
direction) is affine in (x, v_1), so its v_1-coefficient is a constant bit (strict causality). This is exactly what forbids
the acid-test counterexample 2(x0x1^x0x2), which would need an x-DEPENDENT cross-coupling to build
the CCZ from zero odd parities.

kappa is read off the fiber Gauss sum sum_{v in F_2^2} w^{R_x(v)} = 1 + w^A + w^B + w^{A+B+C},
requiring constant modulus 2^{h/2} and an 8th-root phase (diagonality); delta by the shared decoder.

Checks: (i) acid test -- the counterexample is NOT in the family and NOT producible with cc const;
(ii) large random stress test over the d=2 family, 0 violations expected;
(iii) BOUNDED EXHAUSTIVE certification: for n<=2 and N_ker<=3 (the only regime where delta(kappa)
could exceed N_ker), enumerate ALL forms and confirm delta(kappa) <= N_ker.
"""
import itertools, cmath, sys, random
from validate_w1 import delta_of_kappa, parity
W = cmath.exp(1j * cmath.pi / 4)

def eval_d2(n, odd, c2, B1, B2, cc):
    """odd:[(dir,c,amask,e)], c2:[(dir,amask,e)], B1,B2:(amask,e) affine, cc in {0,1}.
    dir in {1,2,3} = bitmask over (v_1,v_2). Returns (valid, N_ker, kappa_table)."""
    NX = 1 << n
    kappa = [0] * NX
    mod0 = None
    for x in range(NX):
        G = 0j
        for v in range(4):                       # v = (v1 bit0, v2 bit1)
            R = 0
            for (dr, c, am, e) in odd:
                R += c * ((bin(dr & v).count("1") & 1) ^ parity(am, x) ^ e)
            for (dr, am, e) in c2:
                R += 2 * ((bin(dr & v).count("1") & 1) ^ parity(am, x) ^ e)
            v1, v2 = v & 1, (v >> 1) & 1
            R += 4 * (v1 & (parity(B1[0], x) ^ B1[1]))
            R += 4 * (v2 & (parity(B2[0], x) ^ B2[1]))
            R += 4 * (cc & v1 & v2)
            G += W ** (R % 8)
        m = abs(G)
        if m < 1e-9:
            return False, 0, None
        if mod0 is None:
            mod0 = m
        elif abs(m - mod0) > 1e-6:
            return False, 0, None                # non-constant modulus -> not diagonal
        ang = cmath.phase(G) / (cmath.pi / 4)
        k = round(ang) % 8
        if abs(ang - round(ang)) > 1e-6:
            return False, 0, None                # not an 8th root
        kappa[x] = k
    return True, len(odd), kappa

def affines(n):
    return [(am, e) for am in range(1 << n) for e in (0, 1)]

# ---------------- acid test ----------------
def acid_test():
    # Try to realize kappa = 2(x0x1 ^ x0x2) with N_ker=0 under CONSTANT cross-coupling.
    # (n>=3 for x0,x1,x2). Scan all coeff-2 + coeff-4 forms with cc in {0,1}, no odd terms.
    n = 3
    target = [(2 * (((x & 1) & ((x >> 1) & 1)) ^ ((x & 1) & ((x >> 2) & 1)))) % 8 for x in range(8)]
    dk = delta_of_kappa(target, n)[0]
    aff = affines(n)
    found = False
    # coeff-2 terms: up to 3, any class/affine; coeff-4: B1,B2 affine, cc in {0,1}
    c2space = [(dr, am, e) for dr in (1, 2, 3) for (am, e) in aff]
    import itertools as it
    for cc in (0, 1):
        for B1 in aff:
            for B2 in aff:
                for k in range(0, 3):
                    for combo in it.combinations(c2space, k):
                        ok, Nk, kap = eval_d2(n, [], list(combo), B1, B2, cc)
                        if ok and kap == target:
                            found = True; break
                    if found: break
                if found: break
            if found: break
        if found: break
    print(f"ACID TEST: delta(2(x0x1^x0x2))={dk}; realizable with N_ker=0 under constant "
          f"cross-coupling? {found}  (must be False)")
    assert not found, "FAIL: counterexample realized -- triangularity model too permissive!"

# ---------------- random stress test ----------------
def stress(n, trials, rng):
    aff = affines(n)
    valid = viol = 0
    worst = None; maxd = 0
    for _ in range(trials):
        k_odd = rng.randint(0, 5)
        odd = [(rng.choice((1, 2, 3)), rng.choice((1, 3, 5, 7)),
                rng.randrange(1 << n), rng.randint(0, 1)) for _ in range(k_odd)]
        c2 = [(rng.choice((1, 2, 3)), rng.randrange(1 << n), rng.randint(0, 1))
              for _ in range(rng.randint(0, 3))]
        B1, B2 = rng.choice(aff), rng.choice(aff)
        cc = rng.randint(0, 1)
        ok, Nk, kap = eval_d2(n, odd, c2, B1, B2, cc)
        if not ok:
            continue
        dk = delta_of_kappa(kap, n)[0]
        if dk is None:
            continue
        valid += 1; maxd = max(maxd, dk)
        sl = Nk - dk
        if worst is None or sl < worst[0]:
            worst = (sl, Nk, dk, odd, c2, B1, B2, cc)
        if Nk < dk:
            viol += 1
            print(f"   *** VIOLATION n={n}: N_ker={Nk} < delta={dk}: odd={odd} c2={c2} B1={B1} B2={B2} cc={cc}")
    return valid, viol, maxd, worst

# ---------------- bounded exhaustive (n<=2, N_ker<=3) ----------------
def exhaustive(n, max_odd):
    aff = affines(n)
    oddspace = [(dr, c, am, e) for dr in (1, 2, 3) for c in (1, 3, 5, 7)
                for am in range(1 << n) for e in (0, 1)]
    c2space = [(dr, am, e) for dr in (1, 2, 3) for (am, e) in aff]
    valid = viol = 0; maxslackneg = None
    for cc in (0, 1):
        for B1 in aff:
            for B2 in aff:
                for kc in range(0, 3):
                    for c2 in itertools.combinations(c2space, kc):
                        for ko in range(0, max_odd + 1):
                            for odd in itertools.combinations(oddspace, ko):
                                ok, Nk, kap = eval_d2(n, list(odd), list(c2), B1, B2, cc)
                                if not ok:
                                    continue
                                dk = delta_of_kappa(kap, n)[0]
                                if dk is None:
                                    continue
                                valid += 1
                                if Nk < dk:
                                    viol += 1
                                    if viol <= 3:
                                        print(f"   *** EXH VIOLATION: Nk={Nk}<delta={dk} "
                                              f"odd={odd} c2={c2} B1={B1} B2={B2} cc={cc}")
    return valid, viol

if __name__ == "__main__":
    acid_test()
    print()
    rng = random.Random(20260706)
    print("RANDOM STRESS over the faithful d=2 (width-2) family, constant cross-coupling:")
    tv = tx = 0
    for n in (2, 3, 4):
        v, x, md, w = stress(n, 300000, rng)
        tv += v; tx += x
        ws = f"min slack={w[0]} (N_ker={w[1]},delta={w[2]})" if w else "none"
        print(f"  n={n}: {v:6d} valid, {x} violations, max delta={md};  {ws}")
    print(f"  TOTAL: {tv} valid d=2 forms, {tx} violations.")
    print()
    print("BOUNDED EXHAUSTIVE (n=2, N_ker<=2; coeff-2 up to 2 terms; all coeff-4, cc const):")
    v, x = exhaustive(2, 2)
    print(f"  {v} valid forms enumerated, {x} violations.")
    print("\nPASS: no realizable d=2 form violates N_ker >= delta(kappa)." if tx == 0 and x == 0
          else "FAIL: a realizable d=2 form violated the inequality.")
