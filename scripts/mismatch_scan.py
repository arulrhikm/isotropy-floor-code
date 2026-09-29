r"""
Which circuits of the Feynman suite are mismatch-free AS A WHOLE, so that thm:transfer prices the
entire circuit (not just its phase blocks)?

A circuit qualifies when it can be written H_O . D . H_O . L with O an "output" set of wires, D a
diagonal phase polynomial and L a CNOT network on O (Clifford), which is what gf2k_family.py verified
for the multipliers.  Concretely, tracking V = H_O U H_O gate by gate with a frame bit per output wire:
  * O := every wire that ever carries an H, or is the target of a 3-arg 'tof' (Toffoli);
  * every H must sit on a wire of O;
  * every CCZ ('Z'/'Zd' with 3 args) must have exactly one leg in O and that leg must be in V's
    computational frame; every Toffoli must have its target in O in V's H frame and controls outside O
    (Toffoli = H CCZ H on the target, so it reads as a CCZ with the output leg in the other frame);
  * a 2-arg 'tof' (CNOT) may act on two O wires in the same frame, on two non-O wires (linear), or
    from a data wire into an output wire (a CZ in the H frame, a linear substitution otherwise), but
    never with an output wire as the control;
  * single-qubit X / Z / S / T on non-O wires are diagonal-or-linear and allowed (T adds a linear
    phase term: level 3 still); Z/S/T on O wires only inside V's computational frame; X anywhere.
Circuits that fail are reported with the first offending gate.  Circuits that pass get the cubic
form f of D, its radical d and the floor 2(n-d)+1 (pure cubic) or the general floor when D has
quadratic/linear content.

Run:  python -u mismatch_scan.py
"""
import glob, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gf2k_family import parse, radical_dim_monos, canonical_ceiling, BENCH


def scan(names, gates):
    O = set()
    for g, args in gates:
        if g == "H":
            O.add(args[0])
        elif g == "tof" and len(args) == 3:
            O.add(args[2])
    frame = {w: 1 for w in O}             # 1 = H frame in V (V prepends H_O)
    form = {w: frozenset([w]) for w in names}
    mons = Counter()
    lin = Counter()                        # linear/quadratic terms from T/S on data wires (not needed for d)
    for g, args in gates:
        if g == "H":
            frame[args[0]] ^= 1
        elif g in ("Z", "Zd") and len(args) == 3:
            legs = [w for w in args if w in O]
            if len(legs) != 1:
                return None, f"CCZ {args}: {len(legs)} output legs"
            c = legs[0]
            if frame[c] != 0:
                return None, f"CCZ {args}: output leg {c} in H frame"
            a, b = [w for w in args if w not in O]
            for u in form[a]:
                for v in form[b]:
                    for x in form[c]:
                        mons[frozenset((u, v, x))] ^= 1
        elif g == "tof" and len(args) == 3:
            a, b, c = args
            if c not in O or a in O or b in O:
                return None, f"Toffoli {args}: target/controls not output/data"
            if frame[c] != 1:
                return None, f"Toffoli {args}: target {c} not in H frame"
            for u in form[a]:
                for v in form[b]:
                    for x in form[c]:
                        mons[frozenset((u, v, x))] ^= 1
        elif g == "tof" and len(args) == 2:
            s, t = args
            if s not in O and t in O:
                # data -> output.  With the output in U's computational frame (V frame 1), H_t CNOT H_t = CZ:
                # a diagonal Clifford (quadratic phase), no effect on the cubic form.  In V's frame 0 the
                # output is being read linearly into nothing -- it would XOR data into an output value
                # that later CCZs read, i.e. a linear substitution on that output form.
                if frame[t] == 1:
                    continue
                form[t] = form[t] ^ form[s]
                continue
            if s in O and t not in O:
                return None, f"CNOT {args}: output wire used as a control"
            if s in O:
                if frame[s] != frame[t]:
                    return None, f"CNOT {args} across frames"
                if frame[s] == 0:
                    form[t] = form[t] ^ form[s]
                else:
                    form[s] = form[s] ^ form[t]
            else:
                form[t] = form[t] ^ form[s]
        elif g in ("X", "Z", "S", "Sd", "T", "Td", "Y") and len(args) == 1:
            w = args[0]
            if w in O and frame[w] != 0 and g != "X":
                return None, f"{g} on output wire {w} in H frame"
            # X on an output wire is H X H = Z (diagonal) in one frame and an affine flip in the other;
            # either way it leaves the cubic form's top part unchanged, so it is always allowed.
            lin[(g, w)] += 1
        else:
            return None, f"unhandled gate {g} {args}"
    if any(f != 1 for f in frame.values()):
        return None, "an output wire ends in the wrong frame"
    if any(len(m) != 3 for m in mons):
        return None, "degenerate monomial (repeated variable): not level 3"
    return {m for m, v in mons.items() if v}, "ok"


def main():
    paths = sorted(glob.glob(os.path.join(BENCH, "*.qc")))
    print(f"{'circuit':<20s} {'result':<44s} {'n':>4s} {'CCZ':>6s} {'d':>4s} {'floor':>6s}")
    for p in paths:
        name = os.path.basename(p)[:-3]
        try:
            names, gates = parse(p)
            mons, msg = scan(names, gates)
        except Exception as e:
            print(f"{name:<20s} error: {e}")
            continue
        if mons is None:
            print(f"{name:<20s} {msg[:44]:<44s}")
            continue
        if not mons:
            print(f"{name:<20s} {'ok but no cubic content':<44s}")
            continue
        d, n = radical_dim_monos(mons, names)
        print(f"{name:<20s} {msg:<44s} {n:>4d} {len(mons):>6d} {d:>4d} {2 * (n - d) + 1:>6d}", flush=True)


if __name__ == "__main__":
    main()
