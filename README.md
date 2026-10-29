# isotropy-floor-code

Code and data for the paper "Exact T-counts of CCZ layers from an isotropy bound"
(Arul Rhik Mazumder, 2026, [arXiv:2610.01024](https://arxiv.org/abs/2610.01024)). Each computational claim in the paper and its
supplementary notes (listed in [CLAIMS.md](CLAIMS.md)) can be
rerun here with one command.

The paper's lower bounds are proved analytically. The scripts here check those statements on
examples, compute the exact T-counts quoted for specific gates, and certify the output of two
T-count optimizers on the Feynman benchmark suite. No proof in the paper depends on a search.

## Setup

Python 3.11 or later.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```
python reproduce.py --tier quick       # 23 claims, about 3 minutes
python reproduce.py --tier standard    # adds 31 claims; about 1.5 hours in all
python reproduce.py --tier full        # adds the benchmark suites and four long searches (hours)
python reproduce.py --list             # every claim, the paper labels it backs, and its tier
python reproduce.py atlas controlled   # named claims only
```

Each claim runs in its own process and writes its output to `out/<id>.txt`. That output is compared
with `expected/<id>.txt` after wall-clock times are removed, and the run ends with
`N/N claims reproduced`. The six full-tier claims have no file in `expected/`. The two benchmark
runs the paper quotes are in `results/benchmark/` instead.

`CLAIMS.md` lists every claim with its paper labels, appendix, command and tier.

## Layout

| Path | Contents |
|---|---|
| `reproduce.py`, `claims.py` | the runner and the list of claims |
| `scripts/` | the experiment scripts. They import one another by module name, so they share one directory |
| `expected/` | reference output for every quick and standard claim |
| `benchmarks/qc/` | the 45 circuits of the Feynman benchmark suite (github.com/meamy/feynman, commit `d2c382a2`) |
| `benchmarks/run_suite.py` | certifies an optimizer's output on the suite, one circuit per process with a 240-second cutoff |
| `benchmarks/benchmark_seeds.sh`, `benchmark_teleport.sh` | the two suite runs of Appendix D (benchmark certificates) |
| `results/benchmark/` | the recorded benchmark runs the paper quotes (see `results/README.md`) |
| `tools/claims_md.py` | regenerates `CLAIMS.md` from `claims.py` |

## Notes on reproducibility

Every sampled experiment fixes its random seed in the code, and `reproduce.py` sets
`PYTHONHASHSEED=0`.

PyZX's TODD optimizer shuffles matrix columns at random. `benchmark_certify.py` seeds that shuffle for
each circuit, so a benchmark row depends only on the circuit and the seed. The paper reports seed 4,
and `results/benchmark/seeds/` holds seeds 0 to 4. The optimizer's output depends on the PyZX version,
so the benchmark numbers hold for PyZX 0.10.3 only.

The recorded outputs were produced with Python 3.13, numpy 2.4, scipy 1.16, OR-Tools 9.15 and
PyZX 0.10.3. The CP-SAT searches use up to 8 threads, and no claim needs more than a few GB of memory.

## Licence

The code is under the MIT licence (`LICENSE`). The circuits in `benchmarks/qc/` are redistributed
under the BSD licence of the Feynman project (`benchmarks/qc/LICENSE-feynman.md`).

## Citing

See `CITATION.cff`. The arXiv identifier will be added when the paper is posted.
