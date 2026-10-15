# EXP-5: Uncertainty-aware adaptive torsional diffusion (matched-compute sample allocation and selection)

Author: R-SAMPLE, round 5. Status: P1 spec complete 2026-10-15 (not yet verified). Evidence IDs refer to
`round5/ledger/evidence_SAMPLE.md`; analysis scripts and outputs in `round5/ledger/sample_data/` (README there).
Shared evaluation rules are in `round5/specs/EVAL.md` and are not repeated here.

## 0. Verdict in one paragraph

On QM9, an adaptive per-molecule sample allocator has little to win over a good *static* rule. With the released TD
model and pooled 3 x 2K samples per molecule, an ORACLE allocator that knows each molecule's true recall curve reaches
AMR-R 0.1400 Å at TD's total budget (Σ 2K_i). That is −0.038 Å against TD's own 2K rule (0.1784), but only −0.0042 Å
against a cross-fitted static rotatable-bond rule (0.1442) [E-SAMPLE-022, E-SAMPLE-023]. The gap is −0.0080 Å at half
the budget and −0.019 Å at a quarter. The GT-free online signals we could compute offline (Good–Turing missing mass,
distinct rotamers, MMFF energy spread) recover none of that gap; they correlate with the oracle gain no better than
n_rot does (Spearman 0.44–0.45 against 0.47) [E-SAMPLE-024]. Most of the "adaptive" headroom is an artefact of TD's
2K rule, which uses the test-set count K and gives single-conformer molecules only 2 samples [E-SAMPLE-003,
E-SAMPLE-022]. Extra compute pays more through **selection** at a fixed 2K output: dropping the 25 % highest-MMFF
proposals and then taking torsion farthest points improves every metric at 2x proposals (AMR-R 0.1778 → 0.1699,
COV-R@0.1 39.9 → 42.6 %, AMR-P 0.2200 → 0.2119) [E-SAMPLE-025]. **Novelty: INCREMENTAL at best.** The allocation
algorithm (water-filling or greedy over concave gains) is GAINS [E-SAMPLE-010/011]. Uncertainty-driven adaptive compute
is AdaDiff and Seij et al. [E-SAMPLE-012, -015]. Ensemble-growth stopping is CREST [E-SAMPLE-042], rotor-based caps are
OMEGA [E-SAMPLE-040], and Good–Turing convergence is Koukos and Glykos [E-SAMPLE-043]. Oversample-and-select is Zhou
et al., Ma et al. and Seij et al. [E-SAMPLE-082, -013, -015]. No paper we found allocates conformer samples per molecule
under a matched budget, so a claim scoped exactly to that (set-level, GT-free, matched NFE, beating static rules and
PG) would be new **only if it wins**, and our data says it probably will not on QM9.
**Recommendation: run Exp 5 as a cheap, pre-registered test with a hard kill criterion (§11).** Its likely durable
outputs are (a) the matched-compute evaluation protocol and the finding about the K-proportional budget (EVAL.md), and
(b) an energy-filtered diversity selector as a baseline/add-on for Exp 1. Do not use Exp 5 as the headline
contribution.

## 1. Question and hypothesis

- **Q.** At a matched total compute budget, does allocating TD sampling compute per molecule from GT-free signals,
  and selecting the reported conformers from the proposals, beat the best fixed-budget rule?
- **H1 (allocation).** A controller using online signals beats the strongest static rule (rotor-strata, fitted on
  validation) on AMR-R and COV-R@0.5 at equal NFE. Prior on QM9 from §3: an effect of at most ≈ 0.004 Å (oracle
  bound), and probably ≈ 0.
- **H2 (selection).** At a fixed 2K output and c = 2 proposals per output slot, a GT-free selector beats random
  selection (= TD) on recall *without losing precision*, and beats PG at matched NFE. Prior from §3: about −0.008 Å
  AMR-R vs random. Untested against PG.
- **H0 honesty rule.** The user's warning holds: "a different stopping rule is not novel unless it beats fixed budgets
  at matched compute". Every comparison in §5 is at matched NFE, and the fixed-budget arms include the strongest
  static rule, not only TD's 2K rule.

## 2. Prior art and novelty check (2020–2026)

| Work | What it does | Relation to Exp 5 | Evidence |
|---|---|---|---|
| TD steps ablation (Table 9) | 3/5/10/20/50 steps on DRUGS: COV-R 42.9/58.9/70.6/72.7/73.1 | Step count is a weak lever above 10 steps. An adaptive-steps arm is low value. | E-SAMPLE-001, -002 |
| TD 2K rule | 2K samples for K reference conformers | The benchmark's budget uses test-set information (K) | E-SAMPLE-003 |
| Particle Guidance (ICLR 2024) | Joint repulsive sampling within a molecule's 2K set; TD + PG 77.0 / 0.543 on DRUGS | Closest competitor at ~1x NFE. Diversity *within* a set; never changes per-molecule counts. Mandatory baseline. | E-SAMPLE-030..033 |
| GAINS (arXiv 2026-09) | Best-of-K search budget over denoising steps; water-filling optimum; offline anchor + online slack | Same maths as our allocator; the algorithm is prior art | E-SAMPLE-010, -011 |
| AdaDiff (ECCV 2024) | Uncertainty-based early exit per step | Uncertainty-aware adaptive compute is prior art | E-SAMPLE-012 |
| Ma et al. 2025 | Inference-time search over noises with verifiers | Best-of-N selection is prior art | E-SAMPLE-013 |
| Gotta Go Fast 2021 | Adaptive step-size SDE solver | Adaptive steps are prior art | E-SAMPLE-014 |
| Seij et al. 2026 | Laplace uncertainty for molecular diffusion; oversample + filter; no gain on GEOM-Drugs | Uncertainty filtering is prior art and failed on DRUGS | E-SAMPLE-015 |
| OMEGA | Rotor-based max-conformer ranges, energy window, lowest-energy output | Static rotor rule = our strongest fixed baseline | E-SAMPLE-040, -041 |
| CREST iMTD-GC / v4 | Restart when a lower conformer appears (max 5 cycles); stop when < 5 % new conformers and < 1 % entropy gain | The ensemble-growth stopping rule is established practice | E-SAMPLE-042 |
| Koukos & Glykos 2013 | Good–Turing missing mass over RMSD clusters for MD convergence | Good–Turing signal is prior art | E-SAMPLE-043 |
| Zhou et al. 2023 / Zhang reply | RDKit oversampling + K-means beats DL on recall; demand for equal budgets | Oversample-and-cluster is prior art; budgets must match | E-SAMPLE-082, -083 |

Search negatives are in the ledger, section Z. No ML conformer paper allocates samples per molecule.
**Verdict: INCREMENTAL.** A defensible claim, if the experiment wins, would be: "the first matched-NFE study of
per-molecule sample allocation and GT-free selection for torsional diffusion; it shows the K-proportional benchmark
budget is suboptimal, and an energy-filtered diversity selector improves recall and precision at 2x NFE". Even then,
reviewers will call the selector a variant of Zhou et al. clustering. The novelty red-teamer must rule on this before
any paper text is written.

## 3. What our data says the controller can exploit (oracle-allocator headroom)

Data: released TD QM9 model, sampling seeds 0/1/2 pooled (N_i = 6K_i), 927 molecules. Exact expected recall for a
random n-subset was computed with hypergeometric formulas (`curves.py`). Realised prefixes over 5 random pool orders
gave the same numbers (`alloc2.py`). These are values on the success set, not the TD convention. ORACLE = test-set
GT, analysis only.

1. **More samples keep helping.** AMR-R 0.1783 (1x), 0.1535 (1.5x), 0.1400 (2x), 0.1255 (3x); COV-R@0.1 39.5 → 55.4 %
   [E-SAMPLE-020].
2. **The gain is everywhere, larger for flexible molecules.** 1x → 3x: −0.031 Å at n_rot = 0 and −0.085 Å at
   n_rot ≥ 4. Rigid molecules gain because every sample re-draws the ETKDG seed, a "local-structure lottery".
   `generate_confs.py:153` skips the model when n_rot = 0, so those samples are pure RDKit seeds. The top 20 % of
   molecules hold 48 % of the gain [E-SAMPLE-021].
3. **Matched-budget allocators** (AMR-R Å; COV-R@0.5 %) [E-SAMPLE-022, -023]:

| Budget (× Σ2K) | TD 2K rule (uses K) | uniform | static n_rot (cross-fit) | two-stage GT-free signal | Good–Turing sequential | ORACLE (AMR obj.) | Oracle − static |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.3122 / 77.3 | 0.2401 / 84.4 | 0.2368 / 85.1 | 0.2365 / 85.3 | 0.2448 / 84.0 | 0.2178 / 88.9 | −0.019 Å / +3.7 pt |
| 0.5 | 0.2414 / 87.6 | 0.1804 / 92.7 | 0.1767 / 93.9 | 0.1772 / 93.8 | 0.2005 / 91.6 | 0.1687 / 95.3 | −0.008 Å / +1.4 pt |
| 1.0 | 0.1784 / 95.1 | 0.1450 / 97.0 | 0.1442 / 97.7 | 0.1443 / 97.7 | 0.1694 / 95.6 | 0.1400 / 98.1 | −0.004 Å / +0.4 pt |

   **Headline number (the bound for Exp 5):** at TD's budget the oracle allocator gains 0.0042 Å AMR-R (2.9 %) and
   0.44 COV-R@0.5 points over the best static rule. The bound is 0.008 Å at half budget. Caveat: all rows are capped at
   the pool size 6K_i, and the cap binds for 24–71 % of molecules under uniform allocation. The cap leaks K into every
   row and makes these lower bounds. Step S0 (§5) removes the cap.
4. **Precision is unaffected by random reallocation in expectation.** The mean of a per-sample quantity over a
   random subset does not depend on n. Allocation therefore moves recall only, while selection moves both
   [E-SAMPLE-022].
5. **Selection at a fixed 2K output** (AMR-R / COV-R@0.1 / AMR-P / COV-P@0.5) [E-SAMPLE-025]: random
   0.1778 / 39.9 / 0.2200 / 91.2; torsion-FPS 2x 0.1656 / 39.0 / 0.2365 / 89.7; MMFF-filter 25 % + FPS 2x
   0.1699 / 42.6 / 0.2119 / 91.9; 3x 0.1653 / 41.8 / 0.2204 / 91.0; ORACLE 2x 0.1398, 3x 0.1255.
6. **Signals.** Spearman with the oracle 1x → 3x gain: Good–Turing (first 6 samples) 0.451, distinct rotamers 0.446,
   MMFF E SD 0.441, n_rot 0.474, K 0.273 [E-SAMPLE-024].
7. **Transfer to DRUGS (INFERENCE).** DRUGS molecules average 6.5 rotatable bonds and 44 atoms
   [E-SAMPLE-085], with wider K, so per-molecule curve dispersion, and with it oracle headroom (GAINS' dispersion
   argument, E-SAMPLE-010), should be larger. This is untested. Measure it in S0-DRUGS before claiming anything.

## 4. Method

### 4.1 Compute unit
One score-model evaluation of one conformer at one σ is 1 NFE. Sample i costs `steps_i` NFE. The ensemble-disagreement
probes cost `E × probes` NFE per probed conformer. MMFF/xTB signals are counted in CPU core-seconds and reported next
to NFE; they are never hidden. RDKit seeding is reported as CPU time (EVAL §8). Matched compute means equal total NFE
± 1 %, and equal CPU seconds ± 10 % where CPU signals are used.

### 4.2 Signals (all GT-free; computed per molecule after each batch)
- **S0 static prior**: n_rot (heavy, TD's `edge_mask.sum()` at `generate_confs.py:138`), n_heavy, ring count, and
  fused/bridged flag. Always present; this is the static baseline.
- **S1 ensemble disagreement**: at probe noise levels σ ∈ {σ_max·0.5, σ_mid, σ_min·4}, the per-torsion score of
  E = 3 independently trained models (CTRL_rematch seeds s0–s2, same architecture) on the same noisy state.
  `u = mean_torsions Var_e[score_e] / score_norm(σ)`, averaged over the batch. The hook is the model call at
  `diffusion/sampling.py:186-187`. Cost: 2 extra NFE per probe and conformer, charged to the budget. Prior art for
  uncertainty signals: E-SAMPLE-012, -015; Seij et al. found no gain on DRUGS, so treat S1 as the riskiest signal.
- **S2 torsion-space novelty**: (a) Good–Turing missing mass f1/n over rotamer strings (each rotatable heavy torsion
  binned in 3 × 120°) [E-SAMPLE-043]; (b) the new-mode rate of the last batch, i.e. the share of its rotamer strings
  unseen before. This is CREST's ensemble-growth criterion [E-SAMPLE-042]. (c) The mean PG kernel similarity
  `exp(−‖Δτ‖²/h)` with PG's torsion wrapping (`sampling.py:217-221`) [E-SAMPLE-031].
- **S3 energy**: MMFF94 single-point energy of each final conformer (RDKit; xtb 6.7.1 GFN2 single point on gnode118
  as the alternative; 0.135 s median per QM9 optimisation, single points cheaper [E-SAMPLE-091]). Signals: the drop in
  best energy over the last batch, and the share of samples inside a 10 kcal/mol window (OMEGA's default window
  [E-SAMPLE-040]).
- **S4 coverage-estimate stability (self-coverage)**: with δ_s = 0.5 Å (QM9) / 0.75 Å (DRUGS), the share of the last
  batch whose heavy-atom RMSD (`GetBestRMS`, the evaluator's metric) to some earlier sample is below δ_s; 1 − this is
  the GT-free discovery rate. This is the RMSD form of S2, as in Koukos & Glykos. Cap the cost at 32 references per
  new sample.

### 4.3 Gain model (fitted on validation only)
On a 1000-molecule QM9 *validation* subset (TD split index 1; GT conformers from the standardized pickles, never test),
generate pools of 10 × 2K. For every molecule and batch boundary n, record the signals and the realised marginal gain
in the objective. The objective is molecule-averaged AMR-R; COV-R@0.5 is a secondary objective with its own fit. Fit
`ĝ(n | S0..S4) ≥ 0`, monotone non-increasing in n, with a 2-parameter saturating form per molecule,
`f(n) = a·(1 − (1 − p)^n)` (PG Fig. 5 form [E-SAMPLE-032]). Here a and p come from a small gradient-boosted regressor
on the signals. A static-only fit (S0) gives baseline B3. The test set is used once, for the final run.

### 4.4 Controller (Track D: allocation; Track P adds selection)
```
input: molecules M, total NFE budget B, batch size b (QM9 4, DRUGS 16), stage-1 size n1 (QM9 6, DRUGS 16)
for i in M: draw n1 samples (seeded ETKDG + TD reverse SDE, 20 steps); compute S0..S4
spent = Σ n1·steps (+ probe NFE)
heap = max-heap on ĝ_i(n_i) / (b·steps)            # marginal gain per NFE (water-filling, GAINS [E-SAMPLE-010])
while spent + b·steps ≤ B:
    i = pop(heap); draw b more samples for i; update S2..S4 (and S1 if probes are on); spent += cost
    if n_i < cap_i: push(i, ĝ_i(n_i))               # cap_i = 20·n1 safety cap (QM9), reported
Track D output: all n_i samples of molecule i
Track P output: select exactly 2K_i of the n_i proposals: drop the q = 25 % highest-MMFF-energy proposals (keep
                ≥ 2K_i), then torsion farthest-point sampling (circular distance, start from the lowest-energy one)
```
Greedy on concave gains is optimal for separable allocations (E-SAMPLE-010). The offline-anchor-plus-online-slack
design follows GAINS' regret result [E-SAMPLE-011]. Track P needs n_i ≥ 2K_i, so Track P allocates only the extra
(c − 1)·Σ2K_i proposals.

### 4.5 What is and is not oracle
K_i is used by Track P only to fix the output size, as the benchmark requires. The controller and gain model never
see K_i or GT. Track D never uses K_i. All GT-based curves in §3 are ORACLE analysis.

## 5. Matched-compute protocol

- **S0 (pools, GPU, first):** with the released QM9 model, generate 10 × 2K per molecule for test and for the
  validation subset, sampling seeds 0–9 (existing `--seed`, `generate_confs.py:48, :94-95`). Recompute §3 without the
  cap, giving the uncapped oracle headroom. **Gate:** if the uncapped oracle − static gap at 1x is < 0.003 Å AMR-R
  and < 0.5 COV-R@0.5 points, stop Track D (§11).
- **Track D arms** (budget B ∈ {0.5, 1, 2} × Σ2K NFE-equivalent, 20 steps): D0 TD 2K rule (reference; uses K);
  D1 uniform n; D2 static rotor rule (OMEGA-style ranges, fitted on validation); D3 static S0 gain model; D4 the full
  controller (S0–S4); D5 = D4 with random batch order (control for adaptivity); ORACLE allocator (analysis row).
- **Track P arms** (output exactly 2K_i; proposals c ∈ {1, 2} × Σ2K; NFE matched): P0 TD (c = 1); P1 TD c = 2 +
  random selection (equals P0 in expectation; NFE control); P2 TD + PG at c = 1, with the tuned README
  hyper-parameters (`generate_confs.py:38-46`), and PG's NFE overhead (kernel cost) reported; P3 uniform c = 2 +
  selector; P4 static allocation + selector; P5 controller + selector; P6 PG + selector at c = 2 (composition); ORACLE
  selection (analysis row).
- **Matching rule:** arms in one comparison differ by ≤ 1 % in total NFE. Probe NFE and CPU signal seconds are added
  to the arm that uses them. A 10-steps arm (half NFE per sample, twice the samples) is one optional extra row, only if
  S0 shows the steps curve is flat (TD Table 9, E-SAMPLE-001).
- **Seeds:** 3 sampling seeds per arm (the model is fixed; training seeds do not apply to the released checkpoint).
  Report the seed SD next to every paired CI (EVAL §7).

## 6. Ablations (each removes one thing from D4/P5 at matched NFE)

| Ablation | Removes | Answers |
|---|---|---|
| −S1 | ensemble disagreement (and its probe NFE, returned to the budget) | Is model uncertainty worth its NFE? |
| −S2 | torsion novelty / Good–Turing | Does mode discovery add to n_rot? |
| −S3 | energy signals (and the energy filter in the selector) | Value of physics |
| −S4 | self-coverage | Value of RMSD-based convergence |
| S0 only | all online signals (= D3/P4) | Is anything adaptive needed? |
| random order | adaptivity, same totals (= D5) | Controls for the total-count effect |
| selector: random / FPS only / energy only / filter + FPS | selection components | Where Track P's gain comes from |
| PG on/off | within-set diversity | Additivity with PG |

Pre-declared primary contrasts (Holm family): D4 − D2 and D4 − D3 at B = 1x; P5 − P3, P5 − P2 and P3 − P0 at c = 2.
Each is tested on AMR-R and COV-R@0.5 (TD convention), so 10 tests. Everything else is secondary or descriptive.

## 7. Evaluation

Follow `round5/specs/EVAL.md`: TD-P QM9 then DRUGS; both failure conventions; COV-R@0.1 on the success intersection
as key secondary; precision always beside recall (Track D changes per-molecule counts, Track P changes selection);
strata by n_rot, K, rings and scaffold novelty; OOD as in EVAL §5. Two Exp-5-specific rules:
- **Track D numbers are not comparable with published TD-P tables** (the per-molecule counts differ). Print them in
  their own table with the D0 reference. Only Track P rows go into the TD-P comparison table.
- Report recall-vs-NFE curves (B = 0.25–2x) for D1, D2, D4 and ORACLE. The headline is the area between the D4 and D2
  curves, with a paired bootstrap CI.

**OOD test of the controller:** fit the gain model on QM9 validation; apply it unchanged to (a) the QM9 scaffold-novel
stratum (87 molecules [E-SAMPLE-052]), (b) GEOM-DRUGS with the released DRUGS checkpoint (refit only S0's rotor
ranges on DRUGS validation, as a second arm), and (c) GEOM-XL if DRUGS runs. The controller passes OOD if D4 − D2
keeps its sign and stays within its CI on (b).

## 8. Coding instructions (`flexitors/`; TD stays frozen)

| New file | Contents | Touchpoints / sources |
|---|---|---|
| `flexitors/sampling/td_backend.py` | `TDSampler(model_dir, ckpt, seed)` with `draw(smi, n, rng) -> list[Mol]`: wraps `get_seed`, `embed_seeds`, `perturb_seeds`, `sample`, `pyg_to_mol` for incremental batches. Per-batch ETKDG seed = hash(global seed, molecule index, batch index) so batches are reproducible and disjoint. | `torsional-diffusion/generate_confs.py:64-69` (`embed_func`, seeded), `:125-180` (`sample_confs`), `:138` (n_rot), `:150-163`; `diffusion/sampling.py:40-60` (`get_seed`), `:63-102` (`embed_seeds`), `:105-113`, `:116-246` (`sample`) |
| `flexitors/sampling/budget.py` | `NFECounter` (per molecule: samples × steps + probe NFE), `CPUTimer`; `assert_matched(arms, tol=0.01)` | — |
| `flexitors/sampling/signals.py` | `rotamer_strings`, `good_turing_missing_mass`, `new_mode_rate`, `pg_kernel_similarity` (wrapping as `sampling.py:217-221`), `self_coverage(new, old, delta, max_ref=32)` (GetBestRMS on heavy atoms as `evaluate_confs.py:111-115`), `mmff_single_point`, `xtb_single_point` (calls `flexitors/eval/energy.py` from EXP-1), `ensemble_disagreement(models, data, sigmas)` | `round5/ledger/sample_data/signals.py` (prototype); `diffusion/sampling.py:184-188` (σ loop, model call), `:217-221` |
| `flexitors/sampling/gain_model.py` | saturating fit `a(1−(1−p)^n)`; regressor from signals to (a, p); `fit(val_records)`, `predict(signals, n)`; monotonicity test | PG Fig. 5 form [E-SAMPLE-032] |
| `flexitors/sampling/controller.py` | §4.4 loop; `allocate(molecules, budget, gain_model, batch, n1, cap)`; static variants D1–D3; logs every decision (molecule, n, signals, ĝ) to JSONL | GAINS water-filling [E-SAMPLE-010] (cite in a comment with PDF p. 1) |
| `flexitors/sampling/select.py` | `select_2k(mols, k2, energy, q=0.25, method='filter_fps')`; `fps_torsion`; `random`; `lowest_energy` | `round5/ledger/sample_data/select.py` (prototype) |
| `flexitors/eval/allocation_curves.py` | exact hypergeometric COV/AMR curves for a pool; oracle greedy allocator; uncapped/capped reporting | `round5/ledger/sample_data/curves.py`, `alloc.py`, `alloc2.py` (port and test) |
| `experiments/exp5_adaptive/` | `config.yaml` (arms, budgets, seeds), `run_pools.sbatch` (S0), `run_arms.sbatch`, `README.md` | `slurm/` patterns of round 2/4 (packed inference) |

The output pickle format equals `generate_confs.py:208-213` (`{smi: [Mol]}`), so `evaluate_confs.py --out_results`
scores every arm unchanged. Every arm also writes `budget.json` (NFE, CPU s, wall s per molecule).

## 9. Tests (`flexitors/tests/`)
- `test_td_backend.py`: with a fixed seed, `draw(smi, n)` in one call equals the concatenation of two calls of n/2
  (same seeds), and matches `generate_confs.py` output for `--confs_per_mol n --seed s` on 3 molecules (RMSD < 1e-4 Å).
- `test_budget.py`: NFE equals Σ samples × steps (+ probes); `assert_matched` rejects a 2 % mismatch.
- `test_curves.py`: hypergeometric expectation equals the Monte-Carlo mean over 20k random subsets (± 3 SE) on 5
  molecules; at n = N it equals the full-pool value.
- `test_controller.py`: on synthetic concave curves the greedy result equals brute-force optimum (small M); the budget
  is never exceeded; the controller never reads K or GT (static check that it is not passed).
- `test_select.py`: output size is exactly 2K; the selected set is a subset of the proposals; `q = 0` with FPS equals
  pure FPS; deterministic given rng.
- `test_signals.py`: Good–Turing equals f1/n on a hand-made list; self-coverage of a set with itself is 1.

## 10. Compute (GPU-h on RTX 3090; measured anchor: one 1x QM9 test pass = 14 min wall [E-SAMPLE-053])

| Step | GPU-h | CPU |
|---|---|---|
| S0 QM9 pools: test 10 × 2K + validation 10 × 2K | ≈ 2.4 + 2.4 | evaluation of 10x pools, ≈ 10 × current eval cost (packed; hours on 6 CPUs) |
| S1 probes (3 models × 3 σ per conformer, on the validation pools) | ≈ 0.5 | — |
| Track D + P arms on QM9 test (≈ 14 arms × 3 seeds, mostly ≤ 2x) | ≈ 15 | MMFF single points: seconds; xtb optional ≈ 0.1 s each |
| DRUGS S0 (released DRUGS model, 3 × 2K pool on test + 300 validation molecules) | ≈ 10–20 (UNVERIFIED; PG reports > 200 samples per DRUGS molecule on average [E-SAMPLE-034]) | DRUGS evaluation is CPU-heavy (GetBestRMS on ~25 heavy atoms; measure on 20 molecules first) |
| DRUGS arms (only if QM9 passes §11) | ≈ 30–60 (UNVERIFIED) | as above |

QM9 total ≈ 20 GPU-h (about one node-day on a single GPU); DRUGS ≈ 40–80 GPU-h, gated. No training is needed.

## 11. Risks and kill criteria

- **K1 (after S0, QM9):** uncapped oracle − static gap at 1x < 0.003 Å AMR-R → drop Track D and report the
  negative result (it is still a useful protocol finding). Our capped estimate is 0.0042 Å [E-SAMPLE-023].
- **K2 (validation):** if D4 does not beat D3 on validation by ≥ 0.002 Å with a paired CI excluding 0, do not run
  D4 on test. Run only the static and selection arms.
- **K3 (Track P):** if P3/P5 do not beat PG (P2) at matched NFE on recall *and* precision, the selector is reported
  as a baseline only.
- **Risk: metric gaming.** Track D shifts samples to molecules where per-molecule recall is cheap. This is exactly
  the "recall rewards oversampling" critique [E-SAMPLE-082]. Mitigation: Track D gets its own table, precision is
  always reported, and Track P keeps the benchmark's 2K output.
- **Risk: stochastic failures.** Failures vary by seed (65–69 [E-SAMPLE-051]). A controller that re-draws seeds for
  failed molecules changes the universe. Fix it: a molecule that fails stage 1 is a failure in every arm. No retries
  beyond TD's own.
- **Risk: DRUGS transfer.** Seij et al.'s uncertainty filter worked on QM9 and not on DRUGS [E-SAMPLE-015]; expect
  the same for S1.
- **Risk: novelty.** Every component has prior art (§2). The contribution is at most an empirical, matched-compute
  study. The novelty red-teamer must rule before any claim.
