"""
Numeric sanity checks for the epsilon-robust floor (thm:epsrigid).

Exact arithmetic in Z[omega]/sqrt2^k, omega = exp(i pi/4): entries are
(a + b w + c w^2 + d w^3)/sqrt2^k with integer a,b,c,d and w^4 = -1.

Checks:
 (1) sde(V) <= 2 n (t+1) for random Clifford+T circuits (n=1,2; t<=6).
     [The tex uses the Aaronson-Gottesman-based bound sde(Clifford) <= 2n.]
 (2) RIGIDITY: for every circuit V in the sample and every entry v of V,
     either v equals an eighth root of unity (or 0) exactly, or its distance
     from every eighth root (and 0) is >= 2^-(sde(V)+1).
     This is the Liouville gap that powers the dichotomy.
 (3) DICHOTOMY EXAMPLE (n=1): the closest t=0 circuit (Clifford) to the T
     gate is at operator distance >= 2^-4 (the theorem's radius at n=1,t=0);
     the actual distance is far larger.
"""
import itertools, random, math, cmath

# ---- ring Z[w]/sqrt2^k: value = (vec, k), vec = (a,b,c,d) ----
W = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))

def cval(vec, k):
    a, b, c, d = vec
    return (a + b * W + c * W**2 + d * W**3) / (2 ** (k / 2))

def mul_vec(u, v):
    # (sum u_i w^i)(sum v_j w^j), w^4 = -1
    out = [0, 0, 0, 0]
    for i, ui in enumerate(u):
        if ui == 0:
            continue
        for j, vj in enumerate(v):
            if vj == 0:
                continue
            p = i + j
            s = ui * vj
            if p >= 4:
                p -= 4
                s = -s
            out[p] += s
    return tuple(out)

def add_vec(u, v):
    return tuple(x + y for x, y in zip(u, v))

SQRT2 = (0, 1, 0, -1)  # w - w^3 = sqrt2

def div_sqrt2(vec):
    """Return u with u*sqrt2 = vec, or None. vec*sqrt2/2 = u."""
    m = mul_vec(vec, SQRT2)
    if any(x % 2 for x in m):
        return None
    return tuple(x // 2 for x in m)

class Mat:
    """Matrix over the ring with common denominator sqrt2^k."""
    def __init__(self, entries, k):
        self.e = entries  # list of rows of vec4
        self.k = k

    def __matmul__(self, other):
        n = len(self.e)
        out = [[(0, 0, 0, 0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                acc = (0, 0, 0, 0)
                for l in range(n):
                    acc = add_vec(acc, mul_vec(self.e[i][l], other.e[l][j]))
                out[i][j] = acc
        return Mat(out, self.k + other.k).reduce()

    def reduce(self):
        while self.k > 0:
            divided = []
            ok = True
            for row in self.e:
                r2 = []
                for v in row:
                    d = div_sqrt2(v)
                    if d is None:
                        ok = False
                        break
                    r2.append(d)
                if not ok:
                    break
                divided.append(r2)
            if not ok:
                break
            self.e = divided
            self.k -= 1
        return self

    def sde(self):
        return self.k

def eye(n):
    dim = 1 << n
    return Mat([[(1, 0, 0, 0) if i == j else (0, 0, 0, 0) for j in range(dim)]
                for i in range(dim)], 0)

def gate_1q(n, q, g):
    """Embed 1-qubit gate g (2x2 rows of (vec,k)) at qubit q of n."""
    dim = 1 << n
    (g00, g01), (g10, g11), gk = g
    rows = [[(0, 0, 0, 0)] * dim for _ in range(dim)]
    for x in range(dim):
        bit = (x >> q) & 1
        x0 = x & ~(1 << q)
        x1 = x | (1 << q)
        if bit == 0:
            rows[x][x0] = g00
            rows[x][x1] = g01
        else:
            rows[x][x0] = g10
            rows[x][x1] = g11
    return Mat(rows, gk).reduce()

def gate_cnot(n, c, t):
    dim = 1 << n
    rows = [[(0, 0, 0, 0)] * dim for _ in range(dim)]
    for x in range(dim):
        y = x ^ (1 << t) if (x >> c) & 1 else x
        rows[x][y] = (1, 0, 0, 0)
    return Mat(rows, 0)

H1 = (((1, 0, 0, 0), (1, 0, 0, 0)), ((1, 0, 0, 0), (-1, 0, 0, 0)), 1)
S1 = (((1, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 1, 0)), 0)
T1 = (((1, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 1, 0, 0)), 0)

EIGHTH = [cval((1, 0, 0, 0), 0) * W**j for j in range(8)] + [0j]

def random_circuit(n, t, n_clifford, rnd):
    V = eye(n)
    tcount = 0
    gates = []
    for _ in range(n_clifford + t):
        gates.append("T" if tcount < t and rnd.random() < 0.4 else
                     rnd.choice(["H", "S", "CX"] if n > 1 else ["H", "S"]))
        if gates[-1] == "T":
            tcount += 1
    rnd.shuffle(gates)
    tused = 0
    for g in gates:
        if g == "T":
            V = V @ gate_1q(n, rnd.randrange(n), T1)
            tused += 1
        elif g == "H":
            V = V @ gate_1q(n, rnd.randrange(n), H1)
        elif g == "S":
            V = V @ gate_1q(n, rnd.randrange(n), S1)
        else:
            c, t_ = rnd.sample(range(n), 2)
            V = V @ gate_cnot(n, c, t_)
    return V, tused

def check_sde_bound_and_rigidity():
    rnd = random.Random(11)
    print("(1)+(2) sde bound and entry rigidity on random circuits")
    worst_ratio = None
    viol_sde = viol_rig = 0
    total = 0
    for n in (1, 2):
        for trial in range(120 if n == 1 else 60):
            t = rnd.randrange(0, 7)
            V, tused = random_circuit(n, t, rnd.randrange(2, 12), rnd)
            total += 1
            K = V.sde()
            bound = 2 * n * (tused + 1)
            if K > bound:
                viol_sde += 1
            gap = 2.0 ** (-(K + 1))
            for row in V.e:
                for vec in row:
                    z = cval(vec, V.k)
                    dists = [abs(z - r) for r in EIGHTH]
                    dmin = min(dists)
                    if dmin > 1e-12:  # not exactly a root/0
                        if dmin < gap - 1e-12:
                            viol_rig += 1
                        r = dmin / gap
                        if worst_ratio is None or r < worst_ratio:
                            worst_ratio = r
    print(f"  circuits tested: {total}")
    print(f"  sde(V) <= 2n(t+1) violations: {viol_sde}  (expect 0)")
    print(f"  rigidity |v - w^j| >= 2^-(sde+1) violations: {viol_rig}  (expect 0)")
    print(f"  worst distance/gap ratio: {worst_ratio:.3f}  (>= 1 required)")
    assert viol_sde == 0 and viol_rig == 0

def check_dichotomy_1q():
    print("\n(3) n=1 dichotomy: distance from T to the t=0 set (Cliffords)")
    # enumerate the 1-qubit Clifford group from H,S (192 elements incl. phases)
    seen = {}
    frontier = [eye(1)]
    key = lambda M: (tuple(tuple(v for v in row) for row in M.e), M.k)
    seen[key(frontier[0])] = frontier[0]
    while frontier:
        nxt = []
        for M in frontier:
            for g in (H1, S1):
                M2 = M @ gate_1q(1, 0, g)
                k2 = key(M2)
                if k2 not in seen:
                    seen[k2] = M2
                    nxt.append(M2)
        frontier = nxt
    print(f"  1q Clifford group elements generated (with phases): {len(seen)}")
    Tm = gate_1q(1, 0, T1)
    Tc = [[cval(v, Tm.k) for v in row] for row in Tm.e]
    best = None
    for M in seen.values():
        Mc = [[cval(v, M.k) for v in row] for row in M.e]
        # operator distance via singular values of difference (2x2): use Frobenius/sqrt2 lower bound
        diff = [[Tc[i][j] - Mc[i][j] for j in range(2)] for i in range(2)]
        fro = math.sqrt(sum(abs(diff[i][j]) ** 2 for i in range(2) for j in range(2)))
        opl = fro / math.sqrt(2)  # ||A||_op >= ||A||_F / sqrt(rank)
        best = opl if best is None else min(best, opl)
    thr = 2.0 ** (-4)
    print(f"  min over Cliffords of ||T - C||_op lower bound: {best:.4f}")
    print(f"  theorem radius 2^-(2n(t+1)+2) at n=1,t=0: {thr:.4f}")
    assert best > thr

if __name__ == "__main__":
    check_sde_bound_and_rigidity()
    check_dichotomy_1q()
    print("\nALL PASS")
