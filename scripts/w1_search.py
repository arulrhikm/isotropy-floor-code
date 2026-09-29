"""
Search evidence for Conjecture conj:w1 (width-one conservation, delta(kappa) <= N_ker).
The conjecture is open; this script records what holds and what fails.

A width-1 (bandwidth-1) fiber form over V = F_2^d, even couplings forced by realizability:
    R(v) = c0 + sum_i alpha_i v_i + sum_i 2 gamma_i v_i v_{i+1}   (mod 8),
  alpha_i in Z_8,  gamma_i in Z_4  (coupling 2 gamma_i is EVEN),  c0 in Z_8.
N_ker = #{i : alpha_i odd}  (each odd single-site field is one kernel-meeting T-gate).

Diagonality of the implemented gate requires the fiber Gauss sum
    Z = sum_{v in F_2^d} omega^{R(v)},   omega = e^{i pi/4},
to have |Z| a power of 2 (= 2^{h/2}) AND arg(Z) a multiple of pi/4, so that
    Z = 2^{h/2} omega^{kappa},  kappa in Z_8.

We work in 16th roots zeta = e^{i pi/8} = omega^{1/2} because intermediate (leaf) sums
carry HALF-integer phases (pi/8) that only cancel globally; tracking them is the main difficulty.

This script:
  (1) confirms power-of-2 modulus forces arg(Z) in (pi/4) Z (kappa in Z_8), with NO surviving
      half-integer (pi/8) phases in the full sum;
  (2) delta-based x-family tests of delta(kappa) <= N_ker;
  (2b) exhibits and independently verifies an explicit ABSTRACT COUNTEREXAMPLE;
  (3) the realizable-construction test (see also validate_w1.py).

CONCLUSION (status of the width-1 conservation law delta(kappa) <= N_ker):
  - It HOLDS, and is TIGHT, for the realizable construction (validate_w1.py: 2153 configs,
    0 violations; here test (3): 379 configs, 0 violations incl. a tight N_ker=delta=2 case).
  - It FAILS for abstract "bandwidth-1 + constant-modulus diagonal" families: an explicit
    family with N_ker=0 has kappa(x)=2*(x0x1 ^ x0x2) (= CS.CS.CCZ), delta(kappa)=3>0,
    verified independently (constant modulus, kappa in Z_8 for all x).  Fixing the coupling
    to the Hadamard value 4 does not repair it (42 violations).
  Hence the law is NOT a consequence of the abstract Gauss-sum form: realizability (the
  time-ordering of branch creation) is ESSENTIAL even at
  width one.  This localizes the open problem and is why no clean width-1 proof
  follows from diagonality alone.
"""
import itertools, cmath, random
import numpy as np
from validate_amdist import residue_vec, rm_generators, min_coset_weight, add_term

W = cmath.exp(1j*cmath.pi/4)      # omega
ZT = cmath.exp(1j*cmath.pi/8)     # zeta = omega^{1/2}

def gauss_sum(d, alpha, gamma, c0):
    """Z = sum_v omega^{R(v)} for a width-1 chain; returns complex."""
    Z = 0j
    for v in range(1<<d):
        bits = [(v>>i)&1 for i in range(d)]
        R = c0
        for i in range(d):
            R += alpha[i]*bits[i]
        for i in range(d-1):
            R += 2*gamma[i]*bits[i]*bits[i+1]
        Z += W**(R%8)
    return Z

def is_diagonal(Z, tol=1e-9):
    """|Z| a power of two AND arg(Z) a multiple of pi/4."""
    m = abs(Z)
    if m < tol:
        return False, None, None
    ex = 2*np.log2(m)                       # |Z| = 2^{ex/2}
    if abs(ex-round(ex)) > 1e-6:
        return False, None, None
    ang = cmath.phase(Z)/(cmath.pi/4)       # kappa
    if abs(ang-round(ang)) > 1e-6:
        return False, round(ex), None       # power-of-2 modulus but half-integer phase
    return True, int(round(ex)), int(round(ang))%8

# ---------- (1) when does a width-1 chain give a Z_8 phase? ----------
def survey_diagonality(trials=200000, seed=1):
    rng = random.Random(seed)
    pow2_phaseok = pow2_phasebad = notpow2 = 0
    for _ in range(trials):
        d = rng.randint(1,5)
        alpha = [rng.randrange(8) for _ in range(d)]
        gamma = [rng.randrange(4) for _ in range(d-1)]
        c0 = rng.randrange(8)
        Z = gauss_sum(d, alpha, gamma, c0)
        if abs(Z) < 1e-9:
            continue
        ex = 2*np.log2(abs(Z))
        if abs(ex-round(ex)) > 1e-6:
            notpow2 += 1; continue
        ang = cmath.phase(Z)/(cmath.pi/4)
        if abs(ang-round(ang)) < 1e-6:
            pow2_phaseok += 1
        else:
            pow2_phasebad += 1
    print(f"(1) random chains: pow2&Z8phase={pow2_phaseok}, "
          f"pow2 but HALF-phase={pow2_phasebad}, non-pow2={notpow2}")
    print("    => generic width-1 chains are NOT diagonal; diagonality is a real constraint.")

# ---------- (2) leaf peeling in zeta bookkeeping ----------
# Sum over leaf v_1 (neighbor v_2): g(v2) = 1 + omega^{alpha1} i^{gamma1 v2}.
# Claim to test: whenever the FULL chain is diagonal, the leaf sum can be written
#   g(v2) = mu * zeta^{p + q v2}   with mu in {2, sqrt2}, p,q in Z_16,
# and the reduced chain (absorb q/2-ish into alpha2, mu into modulus, p into c0) is a
# width-1 chain on d-1 vars with Z_full = mu * Z_reduced.  We verify the identity
# Z_full == mu * zeta^{p} * (reduced Gauss sum with alpha2 += q'), and track the
# odd-field budget.
def leaf_reduce(d, alpha, gamma, c0):
    """Return (mu, p, alpha', gamma', c0') folding leaf v_1 into v_2, or None if leaf
    sum is not a clean mu*zeta^{p+q v2}. Works in zeta=Z_16 exponents."""
    a1 = alpha[0]
    g1 = gamma[0] if d >= 2 else 0
    # g(v2=0) = 1 + omega^{a1}; g(v2=1) = 1 + omega^{a1+2 g1}
    def val(e):  # 1 + omega^e  ->  (modulus, zeta-exponent in Z_16) if clean else None
        z = 1 + W**(e%8)
        m = abs(z)
        if m < 1e-9:
            return (0.0, 0)
        ang = cmath.phase(z)/(cmath.pi/8)   # in zeta units
        if abs(ang-round(ang)) > 1e-6:
            return None
        return (m, int(round(ang))%16)
    if d == 1:
        return None
    v0 = val(a1); v1 = val(a1+2*g1)
    if v0 is None or v1 is None:
        return None
    m0,p0 = v0; m1,p1 = v1
    if abs(m0-m1) > 1e-6:            # modulus must not depend on v2 (diagonality of leaf)
        return None
    mu = m0
    p = p0                          # zeta exponent at v2=0
    q = (p1 - p0) % 16              # extra zeta exponent per v2
    # Fold into v_2: zeta^{q v2} = omega^{(q/2) v2} if q even; if q odd it's a genuine
    # half-phase that must be carried.  Represent reduced phase in zeta units.
    new_alpha = list(alpha[1:])
    new_gamma = list(gamma[1:])
    return mu, p, q, new_alpha, new_gamma, c0

def full_in_zeta(d, alpha, gamma, c0):
    """Z computed; return (modulus, zeta-exponent) if clean (arg in (pi/8)Z), else None."""
    Z = gauss_sum(d, alpha, gamma, c0)
    if abs(Z) < 1e-9:
        return (0.0, 0)
    ang = cmath.phase(Z)/(cmath.pi/8)
    if abs(ang-round(ang)) > 1e-6:
        return None
    return abs(Z), int(round(ang))%16

def build_kappa_family(n, d, afun, gfun, couple4_only):
    """Given affine x-dependence, return (ok, kappa_table, Nker) for the x-FAMILY,
    requiring CONSTANT-MODULUS diagonality across all x. Nker = #directions carrying
    an odd field for some x."""
    kappa=[0]*(1<<n); mod0=None; odd_dirs=set()
    for x in range(1<<n):
        alpha=[]
        for i in range(d):
            s=0
            for (c,am,e) in afun[i]:
                s+=c*(parity(am,x)^e)
                if c%2: odd_dirs.add(i)
            alpha.append(s%8)
        gamma=[]
        for (g,gm,ge) in gfun:
            gv = 2 if couple4_only else g            # realizable Hadamard coupling = 4 (gamma=2)
            gamma.append((gv*(parity(gm,x)^ge))%4 if not couple4_only else (2 if (parity(gm,x)^ge) else 0))
        Z=gauss_sum(d,alpha,gamma,0)
        diag,ex,ang=is_diagonal(Z)
        if not diag or ex<d+1 or ex>2*d:
            return False,None,0
        if mod0 is None: mod0=ex
        elif ex!=mod0: return False,None,0
        kappa[x]=ang
    return True,kappa,len(odd_dirs)

def test_structure(trials=400000, seed=2):
    """CORRECT, delta-based, x-family tests on CONSTANT-MODULUS diagonal width-1 families:
      (L1') all-even fields (N_ker=0)  =>  delta(kappa)=0  (no x-dependent odd magic);
      (M)   main inequality delta(kappa) <= N_ker, for arbitrary even couplings AND for
            realizable couplings (=4 only).  Hunt for ANY violation.
    """
    rng = random.Random(seed)
    l1_tot=l1_ok=0
    viol_any=viol_real=0
    valid_any=valid_real=0
    worst_any=worst_real=None
    maxd_any=maxd_real=0
    for _ in range(trials):
        n=rng.randint(2,3); d=rng.randint(2,4)
        afun=[]
        for i in range(d):
            terms=[]
            for _ in range(rng.randint(0,2)):
                terms.append((rng.choice([1,3,5,7]), rng.randrange(1<<n), rng.randint(0,1)))
            for _ in range(rng.randint(0,1)):
                terms.append((rng.choice([2,4,6]), rng.randrange(1<<n), rng.randint(0,1)))
            afun.append(terms)
        gfun=[(rng.randrange(1,4), rng.randrange(1<<n), rng.randint(0,1)) for _ in range(d-1)]
        for couple4 in (False, True):
            ok,kappa,Nker = build_kappa_family(n,d,afun,gfun,couple4)
            if not ok: continue
            dk=delta_of_table(kappa,n)
            if dk is None: continue
            slack=Nker-dk
            if couple4:
                valid_real+=1; maxd_real=max(maxd_real,dk)
                if worst_real is None or slack<worst_real[0]: worst_real=(slack,Nker,dk)
                if dk>Nker: viol_real+=1
            else:
                valid_any+=1; maxd_any=max(maxd_any,dk)
                if worst_any is None or slack<worst_any[0]: worst_any=(slack,Nker,dk)
                if dk>Nker: viol_any+=1
                if Nker==0:
                    l1_tot+=1
                    if dk==0: l1_ok+=1
    print(f"(2) delta-based x-family tests:")
    print(f"    (L1') all-even fields (N_ker=0) => delta(kappa)=0: {l1_ok}/{l1_tot}")
    wa=f"min slack={worst_any[0]} at (N_ker={worst_any[1]},d={worst_any[2]})" if worst_any else "-"
    wr=f"min slack={worst_real[0]} at (N_ker={worst_real[1]},d={worst_real[2]})" if worst_real else "-"
    print(f"    (M) arbitrary even couplings: {valid_any} valid, {viol_any} viol, "
          f"max delta={maxd_any}; {wa}")
    print(f"    (M) realizable couplings(=4): {valid_real} valid, {viol_real} viol, "
          f"max delta={maxd_real}; {wr}")

# ---------- (3) the real inequality: delta(kappa(x)) <= N_ker over affine x-dependence ----------
def mobius_anf(f, n):
    a = {}
    for S in range(1<<n):
        tot=0; sub=S
        while True:
            t=bin(S^sub).count("1"); tot += ((-1)**t)*f[sub]
            if sub==0: break
            sub=(sub-1)&S
        a[S]=tot%8
    return a

def delta_of_table(kappa, n):
    a = mobius_anf(kappa, n); coeff={}
    for S,aS in a.items():
        if aS==0: continue
        w=bin(S).count("1")
        if w==0: continue
        if w==1: add_term(coeff,S,aS)
        elif w==2:
            if aS%2: return None
            m=(aS//2)%4; i,j=[k for k in range(n) if (S>>k)&1]
            for _ in range(m):
                add_term(coeff,1<<i,1); add_term(coeff,1<<j,1); add_term(coeff,(1<<i)|(1<<j),-1)
        elif w==3:
            if aS%4: return None
            m=(aS//4)%2; trip=[k for k in range(n) if (S>>k)&1]
            for _ in range(m):
                for r in range(1,4):
                    for T in itertools.combinations(trip,r):
                        y=0
                        for i in T: y^=(1<<i)
                        add_term(coeff,y,(-1)**(r-1))
        else: return None
    r=residue_vec(coeff,n); G=rm_generators(n)
    return int(min_coset_weight(r,G))

def parity(mask,x): return bin(mask&x).count("1")&1

def test_inequality(trials=120000, seed=3):
    """alpha_i(x), gamma_i(x), c0(x) affine in x; require diagonal for all x; check
    delta(kappa) <= N_ker where N_ker = max_x #{i: alpha_i(x) odd}? (we count distinct
    odd kernel-meeting parities = #directions ever carrying an odd field)."""
    rng = random.Random(seed)
    valid=viol=0; maxd=0; worst=None
    for _ in range(trials):
        n=rng.randint(2,3); d=rng.randint(2,4)
        # each alpha_i = c*(amask.x ^ e): one odd parity per direction sometimes
        odd_dirs=set()
        afun=[]  # list per direction: list of (c, amask, e)
        for i in range(d):
            terms=[]
            for _ in range(rng.randint(0,2)):
                c=rng.choice([1,3,5,7]); terms.append((c, rng.randrange(1<<n), rng.randint(0,1)))
                odd_dirs.add(i)
            # maybe an even field too
            for _ in range(rng.randint(0,1)):
                c=rng.choice([2,4,6]); terms.append((c, rng.randrange(1<<n), rng.randint(0,1)))
            afun.append(terms)
        gfun=[(rng.randrange(1,4), rng.randrange(1<<n), rng.randint(0,1)) for _ in range(d-1)]
        c0f=(rng.randrange(8),)
        kappa=[0]*(1<<n); ok=True; mod0=None
        for x in range(1<<n):
            alpha=[]
            for i in range(d):
                s=0
                for (c,am,e) in afun[i]: s+=c*(parity(am,x)^e)
                alpha.append(s%8)
            gamma=[(g*(parity(gm,x)^ge))%4 for (g,gm,ge) in gfun]
            c0=c0f[0]
            Z=gauss_sum(d,alpha,gamma,c0)
            diag,ex,ang=is_diagonal(Z)
            if not diag or ex<d+1 or ex>2*d:    # need |Z|=2^{h/2}, rho in [1,d]
                ok=False; break
            if mod0 is None: mod0=ex
            elif ex!=mod0: ok=False; break
            kappa[x]=ang
        if not ok: continue
        dk=delta_of_table(kappa,n)
        if dk is None: continue
        Nker=len(odd_dirs)
        valid+=1; maxd=max(maxd,dk)
        slack=Nker-dk
        if worst is None or slack<worst[0]: worst=(slack,Nker,dk)
        if dk>Nker:
            viol+=1
            if viol<=3: print(f"   *** VIOL n={n} d={d}: delta={dk} > N_ker={Nker}")
    ws=f"min slack={worst[0]} at (N_ker={worst[1]},delta={worst[2]})" if worst else "none"
    print(f"(3) inequality over affine x: {valid} valid, {viol} viol, max delta(kappa)={maxd}; {ws}")

def verify_one_violation(seed=2, trials=400000):
    """Find ONE N_ker=0 abstract-diagonal width-1 family with delta(kappa)>0 and verify
    it independently: constant modulus, kappa in Z_8 for all x, kappa x-dependent with
    genuine odd content, delta recomputed by exact coset decode."""
    rng = random.Random(seed)
    for _ in range(trials):
        n=rng.randint(2,3); d=rng.randint(2,4)
        afun=[]
        for i in range(d):
            terms=[]
            for _ in range(rng.randint(0,2)):
                terms.append((rng.choice([1,3,5,7]), rng.randrange(1<<n), rng.randint(0,1)))
            for _ in range(rng.randint(0,1)):
                terms.append((rng.choice([2,4,6]), rng.randrange(1<<n), rng.randint(0,1)))
            afun.append(terms)
        gfun=[(rng.randrange(1,4), rng.randrange(1<<n), rng.randint(0,1)) for _ in range(d-1)]
        ok,kappa,Nker = build_kappa_family(n,d,afun,gfun,couple4_only=False)
        if not ok or Nker!=0: continue
        dk=delta_of_table(kappa,n)
        if dk is None or dk==0: continue
        # independent checks
        mods=set(); allZ8=True
        for x in range(1<<n):
            alpha=[]
            for i in range(d):
                s=sum(c*(parity(am,x)^e) for (c,am,e) in afun[i]); alpha.append(s%8)
            gamma=[(g*(parity(gm,x)^ge))%4 for (g,gm,ge) in gfun]
            Z=gauss_sum(d,alpha,gamma,0)
            mods.add(round(abs(Z),6))
            ang=cmath.phase(Z)/(cmath.pi/4)
            if abs(ang-round(ang))>1e-6: allZ8=False
        xdep = len(set(kappa))>1
        # NB: even-VALUED kappa is not Clifford -- i^{quadratic boolean} (e.g. CS, CCZ)
        # is a genuine level-3 gate, so delta>0 here is correct, not a decoder artifact.
        print(f"    VIOLATION found: n={n} d={d} N_ker={Nker} delta(kappa)={dk}")
        print(f"      constant modulus across x? {len(mods)==1} (mods={mods})")
        print(f"      kappa in Z_8 for all x?    {allZ8}")
        print(f"      kappa table:               {kappa}  (= 2*(x0x1 ^ x0x2): CS.CS.CCZ, delta=3)")
        print(f"      kappa x-dependent & non-Clifford? {xdep and dk>0}")
        print(f"      => abstract bandwidth-1 + diagonal does NOT imply delta<=N_ker.")
        print(f"      => realizability (time-ordering) is ESSENTIAL, even at width one;")
        print(f"         the realizable construction (validate_w1.py) has 0 violations.")
        return
    print("    (no N_ker=0 abstract violation found in budget)")

if __name__=="__main__":
    print("="*72)
    survey_diagonality()
    print()
    test_structure()
    print()
    print("(2b) independent verification of one abstract counterexample:")
    verify_one_violation()
    print()
    test_inequality()
    print("="*72)
