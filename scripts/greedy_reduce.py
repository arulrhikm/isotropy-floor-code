"""
Constructive UPPER bound for delta(QAOA line) at n where exact decoding is infeasible.
Start from the naive residue (symmetric difference of the n-2 CCZ residues) and greedily
XOR low-weight RM(n-4,n)* codewords (punctured 4-flat indicators) to descend in Hamming
weight.  Any weight reached is a valid construction => an upper bound on delta.  If it hits
the proven lower bound 2n+1, that pins delta=2n+1 (Conjecture C) for that n.

Words are Python ints over coordinates y=1..2^n-1 (bit (y-1)); coordinate 0 is punctured.
"""
import itertools, random, importlib.util, os, sys
spec = importlib.util.spec_from_file_location("A", os.path.join(os.path.dirname(__file__), "validate_amdist.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

def popcount(x): return bin(x).count("1")

def naive_residue_int(n):
    c = {}
    for i in range(n-2):
        A.ccz_on(c, (i,i+1,i+2))
    w = 0
    for y,coef in c.items():
        if coef % 2:
            w ^= (1 << (y-1))   # y in 1..2^n-1 -> bit y-1
    return w

def _flat_from(basis, off, n):
    dim = len(basis); word = 0
    for bits in range(1<<dim):
        p = off
        for j in range(dim):
            if (bits>>j)&1: p ^= basis[j]
        if p != 0:                      # coord 0 punctured (not in code support)
            word ^= (1 << (p-1))
    return word

def rand_flat_indicator(n, dim=4, support=None):
    """Indicator int of a dim-flat. If `support` (list of y-masks) given, anchor the
    offset and basis to actual support parities so the flat overlaps the residue."""
    if support and len(support) >= dim+1:
        pts = random.sample(support, dim+1)
        off = pts[0]
        basis = []
        for q in pts[1:]:
            v = q ^ off
            t = v
            for b in basis: t = min(t, t ^ b)
            if t != 0: basis.append(v)
        # top up with random independent vectors if degenerate
        while len(basis) < dim:
            v = random.randrange(1, 1<<n); t=v
            for b in basis: t=min(t,t^b)
            if t!=0: basis.append(v)
        return _flat_from(basis, off, n)
    basis = []
    while len(basis) < dim:
        v = random.randrange(1, 1<<n); t = v
        for b in basis: t = min(t, t ^ b)
        if t != 0: basis.append(v)
    return _flat_from(basis, random.randrange(0, 1<<n), n)

def cur_support(w):
    return [ i+1 for i in range((w.bit_length())) if (w>>i)&1 ]

def greedy(n, iters=40000, seed=0):
    random.seed(seed)
    w = naive_residue_int(n)
    best = popcount(w)
    target = 2*n+1
    stall = 0
    while stall < iters and best > target:
        # half the moves support-guided (anchored on current ones), half random
        sup = cur_support(w) if (stall % 2 == 0) else None
        g = rand_flat_indicator(n, support=sup)
        nw = w ^ g
        pc = popcount(nw)
        if pc <= best:                  # allow equal moves (plateau walk) sometimes
            if pc < best:
                w = nw; best = pc; stall = 0
            elif random.random() < 0.15:
                w = nw; stall += 1
            else:
                stall += 1
        else:
            stall += 1
    return best, w

def verify_moments(word, n):
    masks = [ (i+1) for i in range((1<<n)-1) if (word>>i)&1 ]
    # target line moments
    import itertools as it
    def M(A):
        return sum(1 for y in masks if all((y>>a)&1 for a in A)) & 1
    ok = True
    for a in range(n):
        if M((a,))!=0: ok=False
    for a,b in it.combinations(range(n),2):
        if M((a,b))!=0: ok=False
    for a,b,c in it.combinations(range(n),3):
        tgt = 1 if (b==a+1 and c==a+2) else 0
        if M((a,b,c))!=tgt: ok=False
    return ok, len(masks)

def best_over_seeds(n, seeds, iters):
    bb = 10**9; bw=None
    for s in seeds:
        b,w = greedy(n, iters=iters, seed=s)
        if b < bb: bb=b; bw=w
        if bb == 2*n+1: break
    return bb, bw

if __name__=="__main__":
    lo = int(sys.argv[1]) if len(sys.argv)>1 else 5
    hi = int(sys.argv[2]) if len(sys.argv)>2 else 12
    print("n  naive  best  2n+1  hit?  moments_ok")
    for n in range(lo, hi+1):
        naive = popcount(naive_residue_int(n))
        best, w = best_over_seeds(n, range(1, 31), iters=8000)
        ok, wt = verify_moments(w, n)
        hit = "YES" if best==2*n+1 else ("(<)" if best<2*n+1 else "no")
        print(f"{n:2d}  {naive:5d}  {best:4d}  {2*n+1:4d}   {hit:3s}   {ok}", flush=True)
