# Claims index

Every computational result in the paper, the command that reproduces it, and its tier. Run one with
`python reproduce.py <id>`, or a whole tier with `python reproduce.py --tier quick|standard|full`.
This file is generated from `claims.py` by `tools/claims_md.py`.

| id | paper labels | App. | what it checks | command | tier | reference |
|---|---|---|---|---|---|---|
| `ground_truth` | lem:dmin, thm:rigidradius | A-B | Exact coset decoding (ground truth used by every other script): delta(CCZ)=7, delta(CS)=3, non-additivity 13 < 14 | `python scripts/validate_amdist.py` | quick | reference recorded |
| `compare_bounds` | supplementary (not in the paper) | B | Nullity versus the floor 2(n-d)+1 versus exact delta on the structured cubics | `python scripts/compare_bounds.py` | quick | reference recorded |
| `nullity_table` | supplementary (not in the paper) | B | Stabilizer nullity of the comparison-table gates by Pauli enumeration | `python scripts/verify_nullity.py` | quick | reference recorded |
| `nullity_identity` | prop:nullity, prop:nullitygen | B | nu = n - d and A = R as sets on random pure cubics, n = 4..7 | `python scripts/experiment_nu_eq_n_minus_d.py` | standard | reference recorded |
| `general_bound_cases` | thm:cubicLB | B | The floor against exact delta on the named cases | `python scripts/validate_general_bound.py` | quick | reference recorded |
| `sharp_floor` | thm:genLB, cor:floordominates | B | The floor 2(n-d*)-r against exact delta on 9,000 random mixed gates: no violations, never below nu | `python scripts/sharp_general_bound.py` | standard | reference recorded |
| `atlas` | tab:atlas | B | Every GL(n,2) orbit of cubic forms for n <= 6: floor, exact delta, the 39.4% exact share | `python scripts/cubic_orbit_atlas.py --n 4 5 6` | standard | reference recorded |
| `atlas_audit` | prop:ceiling, tab:atlas | B | Audit of the census: constructors, orbit invariance of delta, d and the rank spectrum; the ceiling on mixed gates | `python scripts/audit_atlas.py` | standard | reference recorded |
| `graded_ceiling` | supplementary (not in the paper) | B | T-insertion locality and the graded set-cover ceiling (beats W(s) on 4 of 9 families) | `python scripts/moment_locality.py` | quick | reference recorded |
| `incremental` | supplementary (not in the paper) | B | One-gate edits recompute at most 7 residue coordinates and match synthesis from scratch | `python scripts/moment_locality.py --incremental` | quick | reference recorded |
| `paircharge` | supplementary (not in the paper) | B | |A|, |B|, |C| odd on every (representation, edge) pair | `python scripts/paircharge_check.py` | standard | reference recorded |
| `quadratic_locus` | rem:grading | B | Grading asymmetry; delta <= 3r/2 fails on pure quadratics; delta <= e+odd holds | `python scripts/quadratic_locus.py` | standard | reference recorded |
| `peel` | supplementary (not in the paper) | B | The peel against 2(n-d*)-r (6.5% / 12.6%), nullity never the strict maximum | `python scripts/quadratic_invariant.py 900` | standard | reference recorded |
| `peel_two_sided` | supplementary (not in the paper) | B | Two-sided peel and the low-end crossover on mixed gates | `python scripts/experiment_mixed_nullity.py` | standard | reference recorded |
| `restriction` | lem:restrict | A-B | The restriction floor: 0/400 wins on pure cubics, 300/399 on mixed gates | `python scripts/restriction_floor.py` | standard | reference recorded |
| `orbit_rank` | supplementary (not in the paper) | K | r = rank Q is a GL-orbit invariant (24,200 basis changes) | `python scripts/orbit_rank.py` | standard | reference recorded |
| `overlap` | lem:linearoverlap, thm:fanout, rem:twoshared | C | Linear-overlap radical, a weight-(4m+3) fan-out construction, the shared-pair collapse | `python scripts/overlap_layers.py` | quick | reference recorded |
| `splice` | lem:splice, prop:splicetree, thm:parccz, thm:fanout | C | Splice constructions: disjoint 6m+1, fan-out and ripple-carry 4m+3, moments matched | `python scripts/splice_engine.py` | quick | reference recorded |
| `ledger` | rem:ledgerfails | C | Ledger attainment on every linear covering hypergraph n <= 6 (the triangle costs 15 against 13) | `python scripts/ledger_attainment.py` | quick | reference recorded |
| `merge_forest` | conj:ledger | C | The componentwise count always admits a degree-3 forest routing (194,670 patterns) | `python scripts/merge_forest_exists.py` | standard | reference recorded |
| `controlled` | prop:controlled | C | On controlled unitaries the floor is 2mu (2mu+1 if pure-cubic); Campbell-Howard's count equals exact delta for n <= 6 | `python scripts/floor_vs_controlled.py 1` | standard | reference recorded |
| `controlled_gap` | prop:controlled | C | floor = 2mu or 2mu+1 and the gap to Campbell-Howard's count on 36,000 random controlled unitaries, n <= 12 | `python scripts/floor_vs_controlled_gap.py` | standard | reference recorded |
| `qaoa_apps` | prop:univfloor, rem:iqp | D, F | Universal cost-layer floor, generic non-degeneracy, disjoint compression, IQP cores | `python scripts/validate_qaoa_apps.py` | standard | reference recorded |
| `factor_cubic` | prop:factorcubic, cor:starexact | D | Factorizable cubics: weight-(2n+1) construction with matching moments | `python scripts/factor_cubic.py` | quick | reference recorded |
| `factor_radical` | rem:e3even | D | Radical along the factorizable family; e_3 at even n | `python scripts/factor_cubic_radical.py` | quick | reference recorded |
| `field_oracle` | prop:fieldoracle, rem:gf4 | D | GF(2^k) oracle is non-degenerate; delta(GF(4)) = 17 by exact decoding | `python scripts/field_oracle.py` | quick | reference recorded |
| `slice_rank` | lem:slicegram, prop:fieldrank, rem:gf4 | D | Every weight-17 representation of the GF(4) oracle, the slice-rank check wt(c(w)) >= rank T(w) | `python scripts/u2_optimal_residues.py` | standard | reference recorded |
| `slice_rank_lp` | cor:slicelp | D | Delsarte LP on the slice-rank spectrum: 13, 14, 14 on the three n = 6 classes with d = 0 | `python scripts/rank_spectrum_lp.py` | standard | reference recorded |
| `gf_brackets` | rem:gf2mbracket | F | Field-oracle brackets [6k+1, W_k], W_k = (7,17,36,58,90,123) | `python scripts/gf2m_floor.py` | quick | reference recorded |
| `gf_irreducibles` | rem:gf2mbracket | F | W_k over every irreducible polynomial of degree k <= 7 | `python scripts/gf2m_irred_sweep.py` | quick | reference recorded |
| `term_count` | prop:termcount, cor:sparsebracket | D | Cost-layer ceiling equals the term count; the O(D) sparse bracket, 7/6 at D = 1 | `python scripts/sparse_bracket_ratio.py` | quick | reference recorded |
| `brackets` | tab:brackets | D | Two-sided brackets on the application families | `python scripts/bracket_table.py` | standard | reference recorded |
| `c3_permutations` | thm:transfer, prop:gm | E | Mismatch classifier, semi-Clifford decode = floor, Gottesman-Mochon [15,28] | `python scripts/c3_permutation.py` | quick | reference recorded |
| `families` | prop:families | F | The standard cubic families: e_k, cycle, star, K_{2,2,2} | `python scripts/q1_catalogue.py` | quick | reference recorded |
| `line_family` | prop:families | F | Weight-(2n+1) witnesses for the line family, 7 <= n <= 12 | `python scripts/greedy_reduce.py 7 12` | standard | reference recorded |
| `gf2k_family` | sec:benchintro | F | The twelve GF(2^k) multiplier benchmarks priced as whole circuits | `python scripts/gf2k_family.py` | standard | reference recorded |
| `mismatch_scan` | sec:benchintro, app:bench | F | Whole-circuit mismatch scan of the suite: only the multipliers and mod5_4 pass | `python scripts/mismatch_scan.py` | quick | reference recorded |
| `benchmark_selftest` | app:bench | F | The certification pipeline's self-test and the teleportation tensor check | `python scripts/benchmark_certify.py --check-teleport` | quick | reference recorded |
| `benchmark_todd` | app:bench, tab:benchfull, tab:benchtight | F | The TODD suite at the paper's seed (results/benchmark/todd_seed4_suite.log) | `python benchmarks/run_suite.py --seed 4` | full | paper's run in `results/benchmark/todd_seed4_suite.log` |
| `benchmark_teleport` | tab:benchteleport, tab:benchtight | F | Phase teleportation over the suite (results/benchmark/teleport_suite.log) | `python benchmarks/run_suite.py --optimizer teleport` | full | paper's run in `results/benchmark/teleport_suite.log` |
| `levelk_ladder` | cor:mctoffoli | G | Level-k slack ladder 1, 7-15, 21, 51 and the C^{n-1}Z currency | `python scripts/experiment_level_k.py` | standard | reference recorded |
| `levelk_nullity` | prop:nullityk | G | Natural-gate nullity against the moment radical at level k | `python scripts/levelk_nullity_study.py` | standard | reference recorded |
| `levelk_dominance` | prop:nullityk | G | The decomposition nu = (n - d_k) + sum rank tau_j on large-radical gates | `python scripts/levelk_dominance.py` | standard | reference recorded |
| `levelk_counterexample` | prop:nullityk | G | The n = 10 quartic with nu = 9 > 8 = 2(n - d_k) | `python scripts/levelk_counterexample.py` | standard | reference recorded |
| `levelk_crosscheck` | supplementary (not in the paper) | G | Symbolic filtration nu against brute force on 960 gates, the n = 10 quartic, an n = 14 gate | `python scripts/levelk_dominance.py --crosscheck` | standard | reference recorded |
| `nu_doubling` | supplementary (not in the paper) | G | Filtration floors, delta_k >= 2 nu at k = 4, 5, rank Y < nu on 49 of 115 | `python scripts/filtration_floor.py` | standard | reference recorded |
| `cs_fano` | prop:csfloor, cor:csbracket | G | CS carry identity, Fano-line XOR, the ceiling of 7/3 | `python scripts/cs_fano_check.py` | quick | reference recorded |
| `and_gadgets` | cor:andcircuits, lem:window | H | Random AND-compiled circuits: t >= delta(U) + A, windows carry >= 4 odd parities | `python scripts/and_gadget_probe.py --seed 1 --samples 500` | standard | reference recorded |
| `and_levels` | thm:andlevel, cor:ckzclass | H | t >= 8A + odd(M) at every level; the 8k-17 construction; c_4 = 7 over 19 designs | `python scripts/and_ladder_floor.py` | full | none (full tier) |
| `width_one` | conj:w1 | H | N_ker >= delta(kappa) on realizable width-one chains | `python scripts/validate_w1.py` | standard | reference recorded |
| `kappa_bound` | supplementary (not in the paper) | H | No bound delta(kappa) <= f(d): 3 at n = 2, 7 at n = 3, 4 | `python scripts/kappa_bound.py` | full | none (full tier) |
| `eps_rigidity` | lem:sde, lem:liouville, thm:epsrigid | I | sde <= 2N(t+1), the entry gap, the t = 0 dichotomy in exact Z[omega] arithmetic | `python scripts/eps_robust_probe.py` | quick | reference recorded |
| `purification` | lem:krausprop, prop:paulifeedforward | I | Kraus proportionality and the purification T-ledger | `python scripts/purification_ledger.py` | quick | reference recorded |
| `sre` | sec:disc, tab:rate | 2, K | Amortized SRE against nullity and the floor on every orbit n <= 6 and the named families | `python scripts/sre_vs_nullity.py --check --families` | standard | reference recorded |
| `sre_crossover` | op:entropy | 4, K | Hill-climb for an SRE crossing: M2/n reaches 0.760 at n = 11 against 0.830 needed | `python scripts/sre_crossover.py` | standard | reference recorded |
| `layered_nullity` | rem:layerednullity, rem:layeredcap | K | nu adds across a Hadamard layer when B is invertible; structural stabilizer count | `python scripts/nu_layers.py` | standard | reference recorded |
| `layered_floor` | rem:layeredfloor | K | Flat circuits at h = n obey t >= delta_2n(P') | `python scripts/flat_pathsum.py` | standard | reference recorded |
| `layered_bracket` | rem:layeredfloor | K | delta_2n(P) = delta(D1 x D2), the two-layer bracket and the O(D) ratio | `python scripts/twolayer_bracket.py` | full | none (full tier) |
| `single_qubit_layers` | rem:layeredcap | K | T(HT)^L has T-count L+1 and nu <= 2 (36,816 classes) | `python scripts/ma_tcount_n1.py` | quick | reference recorded |
| `quadratic_corner` | supplementary (not in the paper) | B | Pure quadratics: the disjoint CS layer at m <= 3; the search at m = 4 | `python scripts/quadratic_corner.py` | full | none (full tier) |

60 claims: 23 quick, 31 standard, 6 full; 54 have a recorded reference output.
