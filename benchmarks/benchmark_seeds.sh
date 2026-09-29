#!/bin/sh
# TODD over the whole suite at seeds 0-4 (Appendix F.2 quotes seed 4 and the range across seeds).
# Output: out/benchmark/todd_seed<S>.log and _blocks.jsonl.  Several hours in all.
set -e
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4; do
    python benchmarks/run_suite.py --optimizer todd --seed "$s"
done
