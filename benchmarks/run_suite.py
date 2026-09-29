"""Certify an optimizer's output on the benchmark suite, one circuit per process under a time cutoff.

    python benchmarks/run_suite.py --seed 4                    # TODD (PyZX phase_block_optimize), the paper's seed
    python benchmarks/run_suite.py --optimizer teleport         # phase teleportation (deterministic)
    python benchmarks/run_suite.py --seed 4 --only tof_10 adder_8 csum_mux_9

Each circuit runs `scripts/benchmark_certify.py --max-t 400 --seed S --optimizer O NAME` in its own
process with a 240-second cutoff, as in the paper; a circuit that exceeds it is logged as
`timeout/killed`.  Output goes to out/benchmark/ (never to results/, which holds the paper's record):
    out/benchmark/<optimizer>_seed<S>.log          one row per circuit, columns as in the script header
    out/benchmark/<optimizer>_seed<S>_blocks.jsonl per-block records
The PyZX TODD step shuffles at random; benchmark_certify.py seeds it per circuit, so a row depends
only on (circuit, seed).  The floor_s and pyzx_s columns are wall-clock times and will differ.
"""
import argparse, glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QC = os.environ.get("ISOFLOOR_BENCH") or os.path.join(HERE, "qc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--optimizer", choices=["todd", "teleport"], default="todd")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--max-t", type=int, default=400)
    ap.add_argument("--only", nargs="*", help="circuit names (default: every .qc file)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "benchmark"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.optimizer}_seed{a.seed}"
    log, blocks = os.path.join(a.out, tag + ".log"), os.path.join(a.out, tag + "_blocks.jsonl")
    open(blocks, "w").close()
    names = a.only or [os.path.basename(p)[:-3] for p in sorted(glob.glob(os.path.join(QC, "*.qc")))]
    with open(log, "w") as fh:
        fh.write(f"# benchmarks/run_suite.py --optimizer {a.optimizer} --seed {a.seed} --max-t {a.max_t}, "
                 f"{a.timeout} s cutoff per circuit\n")
        for n in names:
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "benchmark_certify.py"), "--max-t", str(a.max_t),
                   "--seed", str(a.seed), "--optimizer", a.optimizer, "--blocks-out", blocks, n]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, env={**os.environ, "ISOFLOOR_BENCH": QC})
                lines = [l for l in p.stdout.splitlines() if l.strip()]
                row = lines[-1] if lines and not lines[-1].startswith("circuit") else f"{n} failed (exit {p.returncode})"
            except subprocess.TimeoutExpired:
                row = f"{n:<18s} timeout/killed"
            fh.write(row + "\n"); fh.flush()
            print(row, flush=True)
    print(f"wrote {log}")


if __name__ == "__main__":
    main()
