"""
lem:paircharge, checked over EVERY representation of many pure-quadratic gates.

For a pure quadratic gate, any {CNOT,T} representation, and any edge {a,b} of the order-2 moment
graph, split the parity support S into

    A = {v in S : a in v, b not in v},  B = {v : b in v, a not in v},  C = {v : a,b in v}.

Purity gives M_{a} = M_{b} = 0 and M_{ab} = 1, so over F_2
    |A| + |C| = 0,   |B| + |C| = 0,   |C| = 1,
hence |A|, |B| and |C| are ALL ODD.  In particular at least three parities meet every edge and at
least one contains both endpoints, so delta >= 3 whenever the graph is nonempty -- which pins
delta(CS) = 3 with no decoding.

This does not check a single representative: it enumerates the WHOLE coset (every element of
Res + RM(n-4,n)^*) so the parity conditions are tested against every legal representation, not
just the canonical one.
"""
import argparse, itertools, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_amdist as AM
from experiment_mixed_nullity import true_moments


def all_coset_elements(res, G, cap=200000):
    k = G.shape[0]
    out = []
    for mask in range(1 << k):
        if len(out) >= cap:
            break
        c = res.copy()
        for i in range(k):
            if (mask >> i) & 1:
                c = c ^ G[i]
        out.append(c)
    return out


def run(ns=(4, 5), trials=60, seed=4):
    rng = random.Random(seed)
    bad = tot = 0
    min_meet = None
    for n in ns:
        T2 = list(itertools.combinations(range(n), 2))
        G = AM.rm_generators(n)
        for _ in range(trials):
            pairs = rng.sample(T2, rng.randint(1, len(T2)))
            c = {}
            for p in pairs:
                AM.cs_on(c, p)
            s3, Q = true_moments(c, n)
            assert not s3, "constructed gate is not pure quadratic"
            E = [(a, b) for a in range(n) for b in range(a + 1, n) if (Q[a] >> b) & 1]
            if not E:
                continue
            for z in all_coset_elements(AM.residue_vec(c, n), G):
                S = [y for y in range(1, 1 << n) if z[y - 1]]
                for (a, b) in E:
                    A = sum(1 for v in S if (v >> a) & 1 and not (v >> b) & 1)
                    B = sum(1 for v in S if (v >> b) & 1 and not (v >> a) & 1)
                    C = sum(1 for v in S if (v >> a) & 1 and (v >> b) & 1)
                    tot += 1
                    if not (A % 2 and B % 2 and C % 2):
                        bad += 1
                    m = A + B + C
                    min_meet = m if min_meet is None else min(min_meet, m)
    print(f"   {tot} (representation, edge) pairs checked over full cosets, n in {list(ns)}")
    print(f"   violations of 'all three counts odd': {bad}"
          f"   {'ok' if bad == 0 else '*** LEMMA REFUTED ***'}")
    print(f"   minimum parities meeting an edge: {min_meet}   (lemma predicts >= 3)")
    return bad == 0 and (min_meet is None or min_meet >= 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()
    print(__doc__.strip())
    print("=" * 78)
    sys.exit(0 if run(trials=args.trials) else 1)
