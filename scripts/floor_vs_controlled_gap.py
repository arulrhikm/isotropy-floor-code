"""Gap analysis for floor_vs_controlled.py: is floor == 2*mu exactly on controlled unitaries?"""
import itertools, random, collections
from floor_vs_controlled import controlled_gate, odd_parities, floor_of, target_moments, lempel

rng = random.Random(7)
for n in (3, 4, 5, 6, 7, 8, 9, 10, 12):
    m = n - 1
    pairs = list(itertools.combinations(range(m), 2))
    tally = collections.Counter()
    for _ in range(4000):
        alpha = [rng.randrange(4) for _ in range(m)]
        beta = frozenset(p for p in pairs if rng.random() < rng.choice([0.1, 0.3, 0.6, 0.9]))
        fl = floor_of(odd_parities(controlled_gate(n, alpha, beta), n), n)
        mu = lempel(target_moments(n, alpha, beta))
        ch = 2 * mu + (mu & 1)
        tally[('floor==2mu', fl == 2 * mu, 'mu_odd', mu & 1, 'gap', ch - fl)] += 1
    print(f'n={n}:', dict(tally), flush=True)
