# The benchmark runs the paper quotes

These are the outputs behind Appendix F.2 of the paper, kept here because regenerating them takes
hours. `python reproduce.py benchmark_todd benchmark_teleport` regenerates them into `out/benchmark/`, and
`benchmarks/benchmark_seeds.sh` reruns all five seeds.

| File | What it is |
|---|---|
| `benchmark/todd_seed4_suite.log` | PyZX's TODD pipeline at seed 4, the run the paper reports (24 circuits, 193 of 311 blocks certified) |
| `benchmark/seeds/todd_seed{0..4}.log`, `..._blocks.jsonl` | the five seeds, with per-block records (active qubits, floor, nu, d*, r, exact delta where decoded); the paper quotes the range across them |
| `benchmark/teleport_suite.log`, `teleport_blocks.jsonl` | phase teleportation over the suite (27 circuits, 252 of 337 blocks) |
| `benchmark/floor_only_input_layering.log` | the floor priced on the input's own layering, for the circuits too large for the optimizer |

Log columns: circuit, n, T in, T out, blocks, certified/blocks, sum of floors, block total, sum of nu,
largest per-block gap, exact decodes as `a(b/c)` (a decoded, b at the floor, c at the optimizer's count),
floor time, optimizer time, blocks certified by nullity alone, T gates in certified blocks. The two time
columns are wall-clock and differ between machines.
