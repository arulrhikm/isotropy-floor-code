#!/bin/sh
# Phase teleportation over the whole suite (deterministic, so one run).
# Output: out/benchmark/teleport_seed4.log and _blocks.jsonl.
set -e
cd "$(dirname "$0")/.."
python benchmarks/run_suite.py --optimizer teleport
