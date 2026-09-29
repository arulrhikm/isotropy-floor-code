"""
Why the cubic corner is tame and the quadratic corner is not: a GRADING asymmetry (rem:grading),
and why the quadratic cost is not a function of the rank r alone (rem:quadopen).

A level-three phase is  phi = 4 f_3 + 2 f_2 + f_1  (mod 8).  Under a CNOT relabelling x -> Lx
(Clifford, so delta is unchanged) the variables become XORs of the old ones, and over Z the
XOR differs from the sum by carries:  u + v = (u ^ v) + 2uv.  The two degrees respond
differently to that:

  * CUBIC mass sits at coefficient 4, and  4z mod 8  depends only on  z mod 2.  So a cubic
    term is BLIND to carries: 4 y_a y_b y_c is the F_2 product.  The pure-cubic locus
    (order-<=2 moments vanish) is therefore CLOSED under GL(n,2).
  * QUADRATIC mass sits at coefficient 2, and  2z mod 8  depends on  z mod 4.  So
    2u + 2v = 2(u ^ v) + 4uv:  a change of variables CONVERTS quadratic mass into cubic mass.
    The pure-quadratic locus is NOT closed under GL(n,2).

Consequence: the argument "put the alternating form in symplectic normal form by a CNOT
relabelling, leaving r/2 disjoint CS gates, so delta(U_2) <= 3r/2" LEAVES the locus it is
arguing about -- the normal form generally acquires a CCZ.  The claim delta(U_2) <= 3r/2 is
false (rem:quadopen).

  minimal counterexample:  q = x0 x1 + x0 x2  on n=4.
  Alternating rank r = 2, so the claim gives delta <= 3.  The true cost is 4.
  Confirmed three ways: XOR-SAT (sat_tcount), meet-in-the-middle (validate_amdist), and
  thm:rigidradius (the residue has weight 4 <= 7, hence delta = 4 with no search at all).

What survives is a bound in the right invariant.  The canonical residue of a pure quadratic
with edge set E(q) is  { {a,b} : q_ab = 1 }  together with  { {a} : deg_q(a) odd }, so

        delta(U_2)  <=  e(q) + odd(q),          e = |E(q)|,  odd = #{a : deg_q(a) odd},

computable in O(n^2), and EXACT whenever e + odd <= 7 by the rigidity radius.  Rank is simply
not the controlling parameter: a star K_{1,k} has rank 2 but costs about 2k.
"""
import itertools, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_amdist as AM
from experiment_mixed_nullity import true_moments
from experiment_nu_eq_n_minus_d import f2_rank


# ---------------------------------------------------------------- phase <-> moments under GL

def transpose_mask(L, y):
    """L^T y = XOR of the rows L[a] over a in y.  Because chi_y(Lx) = chi_{L^T y}(x), a CNOT
    relabelling acts on the phase-polynomial coefficient dict by permuting parity masks
    through L^T -- exact, with no Z_8 inversion needed."""
    z = 0
    a = 0
    while y:
        if y & 1:
            z ^= L[a]
        y >>= 1
        a += 1
    return z


def relabel(coeff, L, n):
    """The gate conjugated by the CNOT circuit x -> Lx (Clifford, so delta is unchanged)."""
    out = {}
    for y, c in coeff.items():
        z = transpose_mask(L, y)
        out[z] = out.get(z, 0) + c
    return out


def random_invertible(n, rng):
    while True:
        L = [rng.randrange(1 << n) for _ in range(n)]
        if f2_rank(L, n) == n:
            return L


def order_le2_mass(coeff, n):
    _, Q = true_moments(coeff, n)
    return any(Q)


def cubic_mass(coeff, n):
    s3, _ = true_moments(coeff, n)
    return bool(s3)


def gl_closure_test(n=5, trials=300, seed=11):
    """Apply random CNOT relabellings to (a) pure cubic and (b) pure quadratic gates, and count
    how often the gate LEAVES its locus.  Prediction: cubic never does (4z mod 8 sees only
    z mod 2, so cubic mass is carry-blind), quadratic often does (2z mod 8 sees z mod 4)."""
    rng = random.Random(seed)
    T3 = list(itertools.combinations(range(n), 3))
    T2 = list(itertools.combinations(range(n), 2))

    cub_left = quad_left = cub_tot = quad_tot = 0
    quad_delta_changed = 0
    for _ in range(trials):
        # (a) pure cubic
        c = {}
        for t in rng.sample(T3, rng.randint(1, len(T3))):
            AM.ccz_on(c, t)
        assert not order_le2_mass(c, n)
        L = random_invertible(n, rng)
        cub_tot += 1
        if order_le2_mass(relabel(c, L, n), n):
            cub_left += 1

        # (b) pure quadratic
        c = {}
        for p in rng.sample(T2, rng.randint(1, len(T2))):
            AM.cs_on(c, p)
        assert not cubic_mass(c, n)
        L = random_invertible(n, rng)
        c2 = relabel(c, L, n)
        quad_tot += 1
        if cubic_mass(c2, n):
            quad_left += 1
        if AM.Tcount(c, n) != AM.Tcount(c2, n):     # sanity: delta MUST be invariant
            quad_delta_changed += 1

    print(f"== GL({n},2)-closure of the two loci, {trials} random relabellings each ==")
    print(f"   pure CUBIC gates that acquired order-<=2 mass : {cub_left}/{cub_tot}"
          f"   {'(locus IS closed, as predicted)' if cub_left == 0 else '*** NOT CLOSED ***'}")
    print(f"   pure QUADRATIC gates that acquired cubic mass : {quad_left}/{quad_tot}"
          f"   {'<- locus is NOT closed, as predicted' if quad_left else '(closed?!)'}")
    print(f"   sanity: delta changed under relabelling in    : {quad_delta_changed}/{quad_tot}"
          f"   (must be 0 -- delta is Clifford-invariant)")
    return cub_left == 0 and quad_left > 0 and quad_delta_changed == 0


# ---------------------------------------------------------------- the corrected upper bound

def canonical_quadratic_weight(pairs, n):
    """e(q) + odd(q): the weight of the canonical residue of the pure quadratic with edge set
    `pairs`.  An O(n^2) upper bound on delta (lem:quadcost)."""
    deg = [0] * n
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    return len(pairs) + sum(1 for a in range(n) if deg[a] % 2)


def three_halves_refutation(n=4):
    """The counterexample, and a sweep showing how often 3r/2 fails vs how e+odd behaves."""
    print(f"\n== the claim delta <= 3r/2 on ALL pure quadratics, n={n} ==")
    T2 = list(itertools.combinations(range(n), 2))
    worst = None
    viol = tot = 0
    ub_ok = 0
    rows = []
    for mask in range(1, 1 << len(T2)):
        pairs = [T2[i] for i in range(len(T2)) if (mask >> i) & 1]
        rowsQ = [0] * n
        for a, b in pairs:
            rowsQ[a] |= 1 << b
            rowsQ[b] |= 1 << a
        r = f2_rank(rowsQ, n)
        c = {}
        for p in pairs:
            AM.cs_on(c, p)
        s3, _ = true_moments(c, n)
        assert not s3, "not pure quadratic"
        d = AM.Tcount(c, n)
        cw = canonical_quadratic_weight(pairs, n)
        tot += 1
        if d > 3 * r // 2:
            viol += 1
            if worst is None or len(pairs) < len(worst[0]):
                worst = (pairs, r, d)
        if d <= cw:
            ub_ok += 1
        rows.append((tuple(pairs), r, d, cw))
    print(f"   {tot} pure-quadratic gates")
    print(f"   delta > 3r/2  (3r/2 claim fails)      : {viol}/{tot}")
    print(f"   delta <= e+odd (lem:quadcost)         : {ub_ok}/{tot}")
    if worst:
        pairs, r, d = worst
        print(f"   minimal counterexample: q = {pairs}   r={r}  3r/2={3*r//2}  delta={d}")
    print(f"   {'q (edges)':28s} {'r':>2s} {'3r/2':>5s} {'e+odd':>6s} {'delta':>6s}")
    for pairs, r, d, cw in sorted(rows, key=lambda t: (len(t[0]), t[2]))[:12]:
        mark = "  <- 3r/2 fails" if d > 3 * r // 2 else ""
        print(f"   {str(list(pairs)):28s} {r:2d} {3*r//2:5d} {cw:6d} {d:6d}{mark}")
    return viol, tot


if __name__ == "__main__":
    print(__doc__.strip())
    print("=" * 78)
    gl_closure_test(n=5)
    three_halves_refutation(4)
    three_halves_refutation(5)
