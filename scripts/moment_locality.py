"""
How does inserting a T on a parity y move the residue and the moments?
And can that be turned into a GREEDY reconstruction that beats the canonical ceiling?

PART 1 -- LOCALITY (closed forms, asserted exactly, not sampled).
Inserting a T on parity y is add_term(coeff, y, 1): it flips exactly ONE residue
coordinate (bit y-1), and flips moment m_A for exactly the A subset y with
1 <= |A| <= 3.  So the moment matrix M[A][y] = [A subset y] has

    column weight (moments moved per inserted T) = |y| + C(|y|,2) + C(|y|,3)
    row weight    (parities that move a given m_A) = 2^{n-|A|}

i.e. insertion is GRADED and SURGICAL forward (only moments BELOW y in the subset
order move; a T on a single qubit moves exactly one moment) and MASSIVELY DEGENERATE
backward (each moment is served by exponentially many parities).  That asymmetry is
the whole reason a floor is hard and a ceiling is easy.

PART 2 -- GREEDY RECONSTRUCTION (matching pursuit over M).
prop:ceiling's canonical solve W(s) is the triangular downward solve RESTRICTED to
|y| <= 3.  Greedy matching pursuit is the same problem unrestricted over all 2^n - 1
parities.  Question: does it beat W(s)?  Scored against exact delta (Amy-Mosca
meet-in-the-middle, n <= 6) on the rows where the bracket is known to be loose:

    family (n=6)          W(s)   delta
    complete cubic e_3     20      11    <-- largest gap among the families
    wrapped cycle C_6      18      17
    Toffoli layer m=2      14      13
    fan-out m=2 / star     12      11

If greedy wins, it is a strictly better O(poly) ceiling: every bracket in the paper
tightens and some open rows close.  If it never wins, that is also a result -- the
canonical downward solve is greedy-optimal, worth a remark.

Every greedy output is re-verified to carry the TARGET moments (round-trip through
moments_from_residue) and compared against exact delta; a weight below delta is a BUG,
not a discovery, and is asserted against.

Run:  python moment_locality.py
"""
import itertools, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import validate_amdist as AM
from sat_tcount import moment_keys, moments_from_residue, weight_of_moment_free_rep


# ---------------------------------------------------------------- moment matrix
def moment_columns(n):
    """cols[y] = bitmask over moment_keys(n) of the moments that parity y flips."""
    keys = moment_keys(n)
    idx = {A: i for i, A in enumerate(keys)}
    cols = {}
    for y in AM.parities(n):
        m = 0
        for A, i in idx.items():
            if (y & A) == A:
                m |= (1 << i)
        cols[y] = m
    return keys, cols


def target_mask(coeff, n, keys):
    res = AM.residue_vec(coeff, n)
    mom = moments_from_residue(res, n)
    m = 0
    for i, A in enumerate(keys):
        if mom[A]:
            m |= (1 << i)
    return m


def part1_locality(nmax=6):
    print("=" * 74)
    print("PART 1 -- locality of a T insertion (closed forms asserted exactly)")
    print("=" * 74)
    for n in range(3, nmax + 1):
        keys, cols = moment_columns(n)
        # column weights: |y| + C(|y|,2) + C(|y|,3)
        by_wt = {}
        for y in AM.parities(n):
            k = bin(y).count("1")
            pred = k + (k * (k - 1)) // 2 + (k * (k - 1) * (k - 2)) // 6
            got = bin(cols[y]).count("1")
            assert got == pred, "column weight n=%d y=%d: %d != %d" % (n, y, got, pred)
            by_wt.setdefault(k, []).append(got)
        # row weights: 2^{n-|A|}
        for A in keys:
            pred = 1 << (n - bin(A).count("1"))
            got = sum(1 for y in AM.parities(n) if (y & A) == A)
            assert got == pred, "row weight n=%d A=%d: %d != %d" % (n, A, got, pred)
        # a single-qubit T flips exactly one moment
        for a in range(n):
            assert bin(cols[1 << a]).count("1") == 1
        summ = ", ".join("|y|=%d->%d" % (k, by_wt[k][0]) for k in sorted(by_wt))
        print("  n=%d  moments moved per T:  %s   |  row weights 2^(n-|A|) = %s"
              % (n, summ,
                 ",".join(str(1 << (n - s)) for s in (1, 2, 3))))
    print("  VERDICT: forward map is graded/triangular (only A subset y move, Theta(|y|^3)");
    print("           of them, exactly 1 for a single-qubit T); backward map is")
    print("           2^(n-|A|)-degenerate.  Surgical forward, hopeless backward.")
    print()


# ---------------------------------------------------------------- graded greedy
# WHY NOT NAIVE MATCHING PURSUIT: a lone level-3 discrepancy m_A cannot be reduced by
# any single column (every y containing A also flips all the lower moments inside y),
# so pursuit stalls at step 0.  The real freedom is that moments only exist up to
# size 3, so a level-3 moment can be fixed by ANY y superset A -- 2^{n-3} choices --
# and the canonical solve W(s) always takes y = A.  Using ONE heavy parity to cover
# many level-3 moments at once is what can beat it; the price is level-2/1 cleanup,
# which is bounded by C(n,2) + n no matter how bad it gets.

def graded_greedy(tgt_mom, n, rng, restarts=60):
    """Stage 3: greedy set-cover of the level-3 discrepancies by parities |y| >= 3.
    Stage 2/1: forced cleanup with weight-2 / weight-1 parities (any heavier parity
    would re-disturb level 3).  Returns the best support found."""
    lvl3 = [A for A in moment_keys(n) if bin(A).count("1") == 3]
    lvl2 = [A for A in moment_keys(n) if bin(A).count("1") == 2]
    lvl1 = [A for A in moment_keys(n) if bin(A).count("1") == 1]
    cands = [y for y in AM.parities(n) if bin(y).count("1") >= 3]
    cover = {y: [A for A in lvl3 if (y & A) == A] for y in cands}
    best = None
    for _ in range(restarts):
        need = {A for A in lvl3 if tgt_mom[A]}
        chosen = set()
        while need:
            scored = []
            for y in cands:
                c = cover[y]
                gain = sum(1 for A in c if A in need) - sum(1 for A in c if A not in need)
                scored.append((gain, y))
            mg = max(s0 for s0, _ in scored)
            if mg <= 0:
                break
            pick = rng.choice([y for g, y in scored if g == mg])
            chosen ^= {pick}
            for A in cover[pick]:
                need ^= {A}
        if need:
            continue
        # stage 2 then stage 1: forced, so just accumulate the residual moments
        vec = {}
        for y in chosen:
            vec[y] = vec.get(y, 0) ^ 1
        supp = {y for y, b in vec.items() if b}
        cur = _moments_of(supp, n)
        for A in lvl2:
            if cur[A] != tgt_mom[A]:
                supp ^= {A}
                cur = _moments_of(supp, n)
        for A in lvl1:
            if cur[A] != tgt_mom[A]:
                supp ^= {A}
                cur = _moments_of(supp, n)
        if all(cur[A] == tgt_mom[A] for A in moment_keys(n)):
            if best is None or len(supp) < len(best):
                best = set(supp)
    return best


def _moments_of(supp, n):
    vec = np.zeros(len(AM.parities(n)), dtype=np.uint8)
    for y in supp:
        vec[y - 1] ^= 1
    return moments_from_residue(vec, n)


def free_code_descent(support, n, rng, iters=6000):
    """Descend inside the coset by XORing random RM(n-4,n)* codewords (moments are
    invariant, so every weight reached is a legal ceiling)."""
    G = AM.rm_generators(n)
    if G is None or len(G) == 0:
        return set(support)
    k = len(G)
    vec = np.zeros(len(AM.parities(n)), dtype=np.uint8)
    for y in support:
        vec[y - 1] ^= 1
    best, bw = vec.copy(), int(vec.sum())
    cur, cw = vec.copy(), bw
    for _ in range(iters):
        word = np.zeros_like(cur)
        for _j in range(rng.randint(1, min(3, k))):
            word ^= G[rng.randrange(k)]
        cand = cur ^ word
        w = int(cand.sum())
        if w <= cw:
            cur, cw = cand, w
            if w < bw:
                best, bw = cand.copy(), w
    return {i + 1 for i in range(len(best)) if best[i]}


# ---------------------------------------------------------------- families
def fam_line(n):      return [(i, i + 1, i + 2) for i in range(n - 2)]
def fam_cycle(n):     return [(i, (i + 1) % n, (i + 2) % n) for i in range(n)]
def fam_complete(n):  return list(itertools.combinations(range(n), 3))
def fam_star(n):      return [(0, j, k) for j in range(1, n) for k in range(j + 1, n)]
def fam_layer(m):     return [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(m)]
def fam_fanout(m):    return [(0, 2 * i + 1, 2 * i + 2) for i in range(m)]


def build(triples, n):
    coeff = {}
    for t in triples:
        AM.ccz_on(coeff, t)
    return coeff


def part2_greedy(seed=4242):
    rng = random.Random(seed)
    print("=" * 74)
    print("PART 2 -- graded greedy vs the canonical ceiling W(s)")
    print("=" * 74)
    rows = [("CCZ",                 fam_layer(1),    3),
            ("3-local line f_5",    fam_line(5),     5),
            ("star S_5",            fam_star(5),     5),
            ("fan-out m=2",         fam_fanout(2),   5),
            ("3-local line f_6",    fam_line(6),     6),
            ("star S_6",            fam_star(6),     6),
            ("Toffoli layer m=2",   fam_layer(2),    6),
            ("wrapped cycle C_6",   fam_cycle(6),    6),
            ("complete cubic e_3",  fam_complete(6), 6)]
    print("  %-20s %3s %6s %7s %8s %7s  %s"
          % ("family", "n", "W(s)", "greedy", "gr+desc", "delta", "verdict"))
    wins = 0
    for name, triples, n in rows:
        coeff = build(triples, n)
        keys = moment_keys(n)
        res = AM.residue_vec(coeff, n)
        tgt_mom = moments_from_residue(res, n)
        Ws, _ = weight_of_moment_free_rep(tgt_mom, n)
        g = graded_greedy(tgt_mom, n, rng)
        gw = len(g) if g is not None else None
        gd = free_code_descent(g, n, rng) if g is not None else None
        gdw = len(gd) if gd is not None else None
        delta = AM.Tcount(coeff, n) if n <= 6 else None
        for supp, tag in ((g, "greedy"), (gd, "gr+desc")):
            if supp is None:
                continue
            cur = _moments_of(supp, n)
            assert all(cur[A] == tgt_mom[A] for A in keys),                 "%s moments do not round-trip on %s" % (tag, name)
            if delta is not None:
                assert len(supp) >= delta,                     "%s weight %d < exact delta %d on %s -- BUG" % (tag, len(supp), delta, name)
        pool = [x for x in (gw, gdw) if x is not None]
        if not pool:
            print("  %-20s %3d %6d %7s %8s %7s  greedy FAILED to cover"
                  % (name, n, Ws, gw, gdw, delta))
            continue
        bestg = min(pool)
        if bestg < Ws:
            verdict = "BEATS W by %d" % (Ws - bestg)
            wins += 1
        elif delta is not None and bestg == delta:
            verdict = "=delta"
        else:
            verdict = "ties W" if bestg == Ws else "worse"
        if delta is not None and bestg == delta:
            verdict += " (=delta, EXACT)"
        print("  %-20s %3d %6d %7s %8s %7s  %s"
              % (name, n, Ws, gw, gdw, delta, verdict))
    print()
    print("  graded greedy (or +descent) strictly beat the canonical ceiling on "
          "%d/%d families" % (wins, len(rows)))
    print()


if __name__ == "__main__" and "--incremental" not in sys.argv:
    part1_locality(6)
    part2_greedy()


# ================================================================ WP-C' 2026-08-22
# INCREMENTAL RECOMPILATION.  The canonical downward solve is TRIANGULAR: z_A depends
# only on m_A and the already-set z_y with y a strict superset of A (|y| <= 3).  A spec
# edit that adds/removes a gate with monomial set B flips m_A exactly on the A subset B
# (1 <= |A| <= 3).  Because the propagation only ever moves DOWN the subset order and
# starts inside the cone {A subset B}, the canonical residue changes on EXACTLY the
# subsets of B: at most 2^{|B|} - 1 <= 7 coordinates per edited gate, independent of n.
# So the canonical ceiling supports O(1)-per-edit recompilation -- re-cover the cone,
# leave everything else untouched.

def incremental_canonical(z_old, mom_new, changed_B, n):
    """Update the canonical residue after a spec edit confined to monomial set B.
    Recomputes z only on subsets of B; asserts every other coordinate is untouched
    (verified against a from-scratch solve by the caller)."""
    z = dict(z_old)
    cone = []
    T = changed_B
    while True:
        if T != 0 and bin(T).count("1") <= 3:
            cone.append(T)
        if T == 0:
            break
        T = (T - 1) & changed_B
    # downward solve restricted to the cone, sizes 3 -> 2 -> 1
    for size in (3, 2, 1):
        for A in cone:
            if bin(A).count("1") != size:
                continue
            acc = mom_new[A]
            for y in z:
                if y != A and (y & A) == A:          # y strict superset of A, weight<=3
                    acc ^= z[y]
            z[A] = acc
    return z, cone


def demo_incremental(seed=99):
    rng = random.Random(seed)
    print("=" * 74)
    print("INCREMENTAL RECOMPILATION -- one-gate edits touch <= 7 residue coordinates")
    print("=" * 74)
    for n in (5, 6):
        base = fam_line(n)
        coeff = build(base, n)
        mom = _moments_of({A for A in ()}, n)                 # placeholder init
        mom = moments_from_residue(AM.residue_vec(coeff, n), n)
        Wz, zsupp = weight_of_moment_free_rep(mom, n)
        z_old = {}
        for y in zsupp:
            z_old[y] = z_old.get(y, 0) ^ 1
        z_old = {y: b for y, b in z_old.items() if b}
        # pad the dict to all weight<=3 parities for the update rule
        for y in AM.parities(n):
            if bin(y).count("1") <= 3:
                z_old.setdefault(y, 0)
        for trial in range(4):
            tri = tuple(sorted(rng.sample(range(n), 3)))
            B = sum(1 << i for i in tri)
            coeff2 = build(base, n)
            AM.ccz_on(coeff2, tri)                            # the edit
            mom2 = moments_from_residue(AM.residue_vec(coeff2, n), n)
            z_new, cone = incremental_canonical(z_old, mom2, B, n)
            # ground truth from scratch
            W2, zsupp2 = weight_of_moment_free_rep(mom2, n)
            z_truth = {}
            for y in zsupp2:
                z_truth[y] = z_truth.get(y, 0) ^ 1
            z_truth = {y: b for y, b in z_truth.items() if b}
            for y in AM.parities(n):
                if bin(y).count("1") <= 3:
                    z_truth.setdefault(y, 0)
            assert z_new == z_truth, "incremental != from-scratch at n=%d edit=%s" % (n, tri)
            untouched = [y for y in z_old if (y & B) != y]
            assert all(z_new[y] == z_old[y] for y in untouched), "edit leaked outside the cone"
            print("  n=%d edit CCZ%s: recomputed %d coords (cone of B), %d untouched, "
                  "W %d -> %d  [matches from-scratch]"
                  % (n, tri, len(cone), len(untouched), sum(z_old.values()), sum(z_new.values())))
    print()

# ================================================================ reproducibility driver (added 2026-09-26)
# rem:incremental: demo_incremental above was never called by the original main block.  Run with --incremental
# (the default run is unchanged).
if __name__ == "__main__" and "--incremental" in sys.argv:
    demo_incremental()
