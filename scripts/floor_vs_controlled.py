"""Does the isotropy floor equal Campbell-Howard's exact T-count on controlled unitaries?

Campbell & Howard (PRA 95, 022316; arXiv:1606.01904), Thm 5: for U = |0><0| (x) I + |1><1| (x) U_t^2
with U_t in D_3, tau[U] = 2 mu[U_t] (+1 if mu is odd), exact, where mu[U_t] is the T-count of U_t
modulo CCZ gates, i.e. Lempel's minimal factorisation of the order-1/2 moment matrix of U_t.

U_t^2 is a diagonal Clifford, so the class is exactly the gates with phase
    phi(x) = x_h * (2 L(x') + 4 Q(x')),   L = sum alpha_i x_i (alpha in Z_4),  Q = sum beta_ij x_i x_j,
i.e. CS gates and CCZ gates all through one hub qubit h.

For random gates in this class we compare
    floor   : isotropy floor (pure-cubic: 2(n-d)+1 if d<n, else 2(n-d*)-r),
    tau_CH  : 2 mu + (mu mod 2), mu = Lempel(Q_t),
    delta   : exact T-count by Reed-Muller coset decoding (n <= 6).
"""
import itertools, random, sys
import numpy as np
from validate_amdist import ccz_on, cs_on, t_on, residue_vec, Tcount


def rank_f2(M):
    M = [list(r) for r in M]
    rows, cols = len(M), len(M[0]) if M else 0
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(rows):
            if i != r and M[i][c]:
                M[i] = [a ^ b for a, b in zip(M[i], M[r])]
        r += 1
    return r


def odd_parities(coeff, n):
    r = residue_vec(coeff, n)
    return [y + 1 for y in range(len(r)) if r[y]]


def moment(P, A):  # A: bitmask; parity of #{y in P : A subset y}
    return sum(1 for y in P if (y & A) == A) & 1


def floor_of(P, n):
    bit = [1 << i for i in range(n)]
    s1 = [moment(P, bit[a]) for a in range(n)]
    s2 = [[moment(P, bit[a] | bit[b]) for b in range(n)] for a in range(n)]  # a==b gives s1
    pure = not any(s1) and not any(s2[a][b] for a in range(n) for b in range(n) if a != b)
    Q = [[s2[a][b] for b in range(n)] for a in range(n)]
    r = rank_f2(Q)

    def s3(a, p, q):  # moment of the SET {a,p,q} (coincident indices collapse)
        return moment(P, bit[a] | bit[p] | bit[q])

    # full-moment radical R*: w with sum_a w_a s3(a,p,q) = 0 for all p,q
    rows_full = [[s3(a, p, q) for a in range(n)] for p in range(n) for q in range(p, n)]
    d_star = n - rank_f2(rows_full)
    if pure:
        rows_cub = [[(s3(a, p, q) if len({a, p, q}) == 3 else 0) for a in range(n)]
                    for p in range(n) for q in range(p + 1, n)]
        d = n - rank_f2(rows_cub) if rows_cub else n
        return (2 * (n - d) + 1) if d < n else 0
    return 2 * (n - d_star) - r


def lempel(Q):
    if not any(any(r) for r in Q):
        return 0
    rk = rank_f2(Q)
    return rk if any(Q[i][i] for i in range(len(Q))) else rk + 1


def controlled_gate(n, alpha, beta):
    """hub = n-1; alpha[i] in Z4 (i<n-1); beta = set of pairs (i,j), i<j<n-1."""
    h = n - 1
    c = {}
    for i, a in enumerate(alpha):
        for _ in range(a):
            cs_on(c, (i, h))
    for (i, j) in beta:
        ccz_on(c, (i, j, h))
    return c


def target_moments(n, alpha, beta):
    """moment matrix Q_t of U_t = omega^{sum alpha_i x_i + 2 sum beta x_i x_j} on n-1 qubits."""
    m = n - 1
    c = {}
    for i, a in enumerate(alpha):
        for _ in range(a):
            t_on(c, i)
    for (i, j) in beta:
        cs_on(c, (i, j))
    P = odd_parities(c, m)
    bit = [1 << i for i in range(m)]
    return [[moment(P, bit[a] | bit[b]) for b in range(m)] for a in range(m)]


def run(n, trials, decode, rng):
    m = n - 1
    pairs = list(itertools.combinations(range(m), 2))
    stats = dict(total=0, floor_eq_ch=0, floor_lt_ch=0, floor_gt_ch=0, ch_eq_delta=0, decoded=0)
    examples = []
    seen = set()
    for _ in range(trials):
        alpha = [rng.randrange(4) for _ in range(m)]
        beta = frozenset(p for p in pairs if rng.random() < rng.choice([0.2, 0.5, 0.8]))
        key = (tuple(alpha), beta)
        if key in seen:
            continue
        seen.add(key)
        c = controlled_gate(n, alpha, beta)
        P = odd_parities(c, n)
        fl = floor_of(P, n)
        mu = lempel(target_moments(n, alpha, beta))
        ch = 2 * mu + (mu & 1)
        stats['total'] += 1
        if fl == ch:
            stats['floor_eq_ch'] += 1
        elif fl < ch:
            stats['floor_lt_ch'] += 1
            if len(examples) < 5:
                examples.append((alpha, sorted(beta), fl, ch))
        else:
            stats['floor_gt_ch'] += 1
            examples.append(('FLOOR>CH', alpha, sorted(beta), fl, ch))
        if decode:
            dl = Tcount(c, n)
            stats['decoded'] += 1
            stats['ch_eq_delta'] += int(dl == ch)
            assert fl <= dl, ('floor above exact', alpha, beta, fl, dl)
    return stats, examples


if __name__ == '__main__':
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    for n, trials, decode in [(3, 400, True), (4, 1500, True), (5, 1500, True), (6, 300, True),
                              (7, 3000, False), (8, 3000, False), (10, 3000, False)]:
        st, ex = run(n, trials, decode, rng)
        print(f'n={n}: {st}', flush=True)
        for e in ex:
            print('   ', e, flush=True)
