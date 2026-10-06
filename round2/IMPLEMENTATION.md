# Round 2 implementation report (coding agent 2)

Written 2026-10-06, 22:45 IST. Spec: `code_plan_2.md` as amended by `DECISION.md` (votes 1 and 3 read).

**Status at handback**
- Code, tests, data builders, the D4/D5 measurements and the round-1 regression tests are done.
- Three items were still running:
  - the GPU smoke rerun (job 10180);
  - the production data builder (job 10122: paired cache + S1 seed pickles);
  - the S5 MMFF standardization (job 10105 → featurize job 10106).
- **No training or inference arrays were submitted.**

**Commits.** All on local `main`, not pushed:
- 14241b7 TD flags
- d503658 tools geometry/builders
- 0b514d6 S0/gate/tests
- dd9e70c slurm
- d4d2ff1 automorphism + pair_ok handling
- two test/smoke fixes
- the last commit: trimmed matrix, PACK=1, CPUs

The cluster checkout is synced up to d4d2ff1 + smoke fix. **The last commit (eval-set rename / trimmed matrix) is NOT yet synced.** It must not be synced while smoke 10180 runs, because the job reads the old set names. Sync afterwards with `bash round2/sync_cluster.sh <last synced local commit>`.

## 1. What changed (file:line, repo state after the commits)

### TD

All new behaviour is behind default-off flags.

- **`utils/parsing.py:30-37,44`**
  - `--l_jitter`, `--l_mix_p_gt`, `--l_interp`, `--log_timing`, `--fail_on_nan`, `--limit_train_iters`, `--lambda_embed_dim`.
- **`utils/dataset.py:18-36`**
  - `interp_x_torch`: linear interpolation with terminal bond lengths restored.
- **`utils/dataset.py:38-100`** (`TorsionNoiseTransform`)
  - The default branch runs the original lines unchanged.
  - The jitter is drawn fresh at every access.
  - `_select_paired_L` keeps the RDKit and GT positions paired by conformer index.
  - S3/B1cap draw a Bernoulli choice; S4 draws λ ~ U[0,1] and sets `node_lambda`.
- **`utils/dataset.py:~139-165`**
  - Stale-cache guard.
  - S4 load-time `pair_ok` filter (DECISION D1: never interpolate an unsafe pair).
- **`utils/dataset.py` `featurize_mol`**
  - Carries `gt_pos` and `pair_ok` through the same loop iteration as `pos`.
- **`utils/dataset.py` `construct_loader`**
  - Passes the new flags via `getattr`.
- **`diffusion/score_model.py:51,65-67,70,76,194-199`**
  - λ embedding concatenated where σ enters. With `lambda_embed_dim=0` the layer shapes are identical.
- **`utils/utils.py:18-20`**
  - Passes `lambda_embed_dim`.
- **`utils/training.py`**
  - Data-wait timer, NaN fail-fast and an iteration limit, all opt-in.
- **`train.py:27-36`**
  - Prints the TIMING line when `--log_timing` is set.
- **`generate_confs.py:52,92,106-112,163`**
  - `--l_level`. Guard: it is required for, and only for, λ-models.
- **`diffusion/sampling.py:109,169-172`**
  - Constant `node_lambda` per batch, with an assertion that it is constant.

### Tools

- **`tools/lgeom.py`**
  - Heavy-atom Kabsch.
  - Relabelling by heavy-atom graph automorphism, then terminal-atom relabelling.
  - `interp_x`.
  - `pair_check` (CIP stereo, sp3 inversion, bond deviation ≤ 0.05 Å, non-bonded distance ≥ 0.9 Å, all at λ = 0.5).
  - `l_subset_errors`: F5 errors on ring/acyclic × all/heavy subsets.
  - `set_internal_subset`: the all-atom acyclic set (D2).
- **`tools/build_paired_pickles.py`**
  - Pairs by `geom_id`; `geom_id` is present in the QM9 raw and std pickles (verified on the cluster).
  - Duplicates are disambiguated by recomputing `conf['rmsd']` (defect 1).
  - Unsafe pairs are kept and flagged.
- **`tools/make_l_seed_pickles.py`**
  - **etkdg source:** writes an ETKDG twin and an MMFF version of the same embeddings. MMFF runs in try/except, and a molecule that fails is dropped from both pickles (defect 3).
  - **noise source:** 2 draws per GT conformer, with common random numbers across σ (vote_3 P2-4).
  - **interp source:**
    - follows the training recipe of `standardize_confs.py:71-119` exactly, including discarding the DE return value, as `:106` does (defect 4);
    - writes λ ∈ {0, .25, .5, .75, 1} and the A5-ring/A5-acyc oracles from the same matched pair;
    - writes the F5 table `l_error.csv` and per-pickle population reports.
- **`tools/paired_compare.py`**
  - `--universe intersection` and `--report_failures`.
- **`tools/local_structure_analysis.py`**
  - `--de_restarts`, `--de_seed` and `--mode leak`. Defaults reproduce round 1.
- **`tools/s4_gate.py`** (new).
- **Tests:** `tools/tests/{test_lgeom,test_seed_builder,test_round2_model,golden_dump}.py`.

### Slurm

- **`common.sh`**
  - `WORK`/`RES` can be overridden.
  - New paths `QM9_SEEDS2`, `QM9_PAIRED`.
  - Opt-in provenance guard (`R2_PROVENANCE=1`), so round-1 calls are unchanged.
  - `run_evalset` packer.
- **`ablation_train_array.sbatch`**
  - `paired` variant.
  - Optional 7th field `GT_GEN_ARGS`.
  - `GT_EVAL=2` (gtLcycle only; D8 trim).
- **`ablation_inference_array.sbatch:45`**
  - MODEL is `eval`-expanded.
- **`featurize_qm9.sbatch`**
  - `paired` variant.
- **New files:**
  - `r2_evalsets.tsv`, `r2_eval_models_{head,post}.tsv`, `r2_eval_array.sbatch`
  - `build_round2_data.sbatch`, `smoke_round2.sbatch`, `ablations_smoke_round2.tsv`
  - `ablations_train_round2{,_noS4}.tsv`, `submit_round2.sh`
  - `round2/sync_cluster.sh`

## 2. DECISION items and defects

| Item | How it was handled |
|---|---|
| D1 | Heavy-atom fit applied to all atoms, plus terminal relabelling and `pair_ok`. **Extensions, raised here:** (a) relabelling by heavy-atom graph automorphism, best heavy RMSD; (b) `interp_x` restores terminal bond lengths. Reason for (b): single-H rotors (O–H, N–H) are not torsions in TD, so DE never matches them, and plain linear interpolation shrinks their bonds by 0.1–0.15 Å. Unsafe pairs are dropped only from S4, at load time. S3/B1cap keep every verified pair, so their RDKit half equals CTRL_rematch's (F4). |
| D2 | `set_internal_subset` uses the all-atom acyclic set. Test T8: the kept ring part is exact to 1e-6. |
| D3/D8 | Trimmed matrix (§6). CR is never cut. CR gtLcycle ×3 is added. |
| D4 | Packing **not enabled** (`PACK=1`); see §5. |
| D5 | 4 CPUs / 3 loader workers; see §5. |
| D6 | B1cap = `paired --l_mix_p_gt 1.0`, 1 seed, wave 3. |
| D7 | Gate implemented (`s4_gate.py`; `submit_round2.sh gate`); status in §4. |
| D9 | Golden tests; exact `mmff_error` count (`grep -hx mmff_error .../logs/worker_*.log | wc -l`, as plan 1 §7); data-wait timer; NaN-only fail-fast; no resume implemented, so nothing is forced; S5 standardization on 4 CPUs. |
| Defect 1 | Ties are disambiguated by recomputing the rmsd; tested in `test_paired_builder_on_fake_std_pickles`. |
| Defect 2 | Exact `mmff_error` count by the grep above. |
| Defect 3 | MMFF in try/except; a failing molecule is dropped from both twin pickles. |
| Defect 4 | Test-time interp seeds discard the DE optimum, exactly as `standardize_confs.py:106` does. |
| Defect 5 | Schedule recomputed with `PACK=1`; see §6. |
| Defect 6 | Default paths verified by the golden tests (§3). |

## 3. Test results

**Local** (rdkit 2026.03.6, no PyG): `test_lgeom` + `test_seed_builder`: **13 passed**.

**Cluster CPU** (torch 1.13.1, PyG 2.0.4, e3nn 0.5.1, rdkit 2022.9.5): **27 passed** in total.

```
PASSED test_T1 ... test_T9 (9), test_seed_builder (4)
PASSED test_C1_lambda_dim0_is_unchanged_architecture, test_C1b_released_checkpoint_loads_strictly,
       test_C2_C3_lambda_plumbing_symmetry, test_C4_generate_guards, test_C5_jitter_fresh_and_dataset_untouched,
       test_C6_worker_seeding_and_reproducibility, test_C7_paired_selection_formulas, test_C7b_torch_interp_equals_lgeom,
       test_C8_featurize_keeps_pairing_through_reacted_filter, test_C10_stale_cache_guard,
       test_C11_fail_on_nan_and_limit_iters, test_C12_s4_pair_ok_filter
golden: 2 passed (test_golden_model_transform_featurize_identical, test_golden_generate_identical)
```

**Golden finding.** The round-1 code itself is not bit-reproducible across processes on CPU.
- Two runs of the snapshot differ by 2–8e-6 in `edge_pred`, even single-threaded with `PYTHONHASHSEED=0`.
- `radius_graph` and `scatter` were checked and are deterministic.
- So these are compared **exactly**: state_dict, transform outputs, post-call RNG states, featurized std and raw datapoints.
- The forward output and generated coordinates are compared against the snapshot-vs-snapshot noise floor. Both passed.

## 4. Pairing, smoke, S4 gate

**Pairing probe** (rematch `000.pickle`, 918 molecules, 6231 conformers):
- `PAIRING_CLEAN 6231/6231 = 100.000%`: every pair verified by `geom_id` and the rmsd recomputation, with 0 graph mismatches.
- `PAIR_OK 5740/6231 = 92.12%`.
- Unsafe pairs:

  | reason | count |
  |---|---|
  | bond_dev | 323 |
  | inversion | 151 |
  | stereo | 15 |
  | clash | 2 |

- These are mostly poorly DE-matched pairs (heavy RMSD 0.4–1.0 Å). Automorphism relabelling did not change the rate.
- On files 000+001: 9921/10818 = 91.71% `pair_ok`.
- The full-data number comes from job 10122 (`$QM9_PAIRED/SUMMARY.txt`; the job also prints CACHE lines comparing paired with rematch).

**Smoke.**
- The first run (10123) failed in featurization: the raw smoke cache was built without `--std_pickles ""`. Fixed.
- The rerun (10180) was still queued at handback.
- Its log, `$LOGS/tordiff_r2_smoke_10180.log`, will contain TIMING / SUMMARY / `SMOKE_*_FAILED` lines.

**S4 gate status: OPEN.**

| Gate item | Status |
|---|---|
| (1) pairing ≥ 98% clean | **PASS** on the probe (100%) |
| (2) `pair_ok` | reported, not thresholded: 92% |
| (3) V18 | needs the head-phase runs (CR/B1 `S1_lam1.00` vs gtLcycle) |
| (4) smoke | needs 10180 |

- **For coordinator decision:** S4 trains on about 92% of conformers and about 96% of molecules. On 000.pickle, 35 of 918 molecules had no safe pair. That is a population difference from CTRL_rematch, biased toward flexible molecules where matching is poor. S3 and B1cap are not affected.

## 5. D4 and D5 measurements

**D4** (job 10124, released model; results in `results_r2test/`). The packed rerun was **not identical** to round 1, and neither was an **unpacked** rerun:

| run | COV-R_mean | MAT-R_mean |
|---|---|---|
| R0 round 1 | 89.0293 | 0.1752 |
| R0 unpacked rerun | 89.0286 | 0.1752 |
| R0 packed rerun | 89.0400 | 0.1752 |
| A1c round 1 | 96.1923 | 0.0762 |
| A1c packed | 96.1958 | 0.0761 |
| A2 round 1 | 88.9339 | 0.1507 |
| A2 packed | 88.9407 | 0.1507 |

- The pipeline is not bit-reproducible: forward-pass nondeterminism (§3). So the D4 criterion "exactly" cannot be met even without packing.
- The packed deviations are the same size as the unpacked rerun's.
- Timing: one unpacked run 877 s; three packed runs 1792 s. That is **1.47× throughput, not 3×**.
- Per the literal DECISION, packing is off (`PACK=1`).
- **For coordinator decision:** enabling `PACK=3` is supported by this evidence.

**D5** (job 10125, CTRL recipe on `cache_rematch`, 400 iterations, node fully loaded):

| configuration | data_wait | throughput |
|---|---|---|
| 3 workers pinned to 4 cores | 2.9 s of 62.2 s = **4.59%** | 6.43 it/s |
| 8 workers on 10 cores | 3.0 s of 63.0 s = 4.79% | 6.35 it/s |

- 4.59% is below 5%, so the choice is **4 CPUs / 3 workers** (`TRAIN_CPUS=4`, `LOADER_WORKERS=3`).
- Caveat: 6.4 it/s implies about 14–16 h per 100 epochs under this load, against the 10.7 h of round 1. Throughput is GPU- or launch-bound, not loader-bound.

## 6. Run matrix and GPU-h (PACK=1, 0.25 GPU-h per inference run, 10.7 h per training run)

**Training: 13 runs, 139.1 GPU-h.**

| wave | runs |
|---|---|
| 1 | B6_jit0.04pa ×3, S3 s0 |
| 2 | S4 ×3 + S3 s1 (gate pass) **or** B3 ×3 + S3 s1 (gate fail) |
| 3 | B3 ×3 + B1cap (or B1cap + B6_0.02) |
| 4 | B6_jit0.02pa |

In-job evaluations: 23 runs, 5.75 GPU-h.

**Head phase: 71 runs.**
- B1 ×3: S1core (8) + A5 (2).
- B1 s0: V18.
- CR ×3: S1core (8) + gtLcycle + λ=1.00 + pre_mmff.
- CB s0: S1panel (6).
- S0 null run.

**Post-training phase: 51 runs.**
- B6 ×3 and S3 ×2: S1panel (6).
- B1cap: S1panel4 (4).
- S4 ×3: S4core (4).
- B3 ×3: S5.

**Total about 175.4 GPU-h**, against the D8 cap of ≤ 175. Trimmed so far: S6, noise 0.01, the S4 descriptive runs, CB s1/s2, and the B6_0.02 panel. These are kept as sets S1x / S4x / S6 if budget frees.

**Wall time:** about 4.5 h for the head phase plus 4 training waves of about 11.5 h, so about 50 h. That becomes about 60 h if training really runs at 6.4 it/s.

## 7. Exact submit commands (after smoke 10180 and data 10122 finish; from a shell on gnode118)

```
ssh ada; srun -n 1 -c 1 --mem=4G -t 00:30:00 -A plafnet2 -p plafnet2 -w gnode118 --pty bash
cd /scratch/nishanth.r/tordiff/slurm
bash submit_round2.sh head              # 8 tasks, %4
bash submit_round2.sh gate              # prints S4_GATE PASS|FAIL
bash submit_round2.sh train pass        # or: train fail   (13 or 10 training runs %4, then the post panels %3)
```

Already running: `prep` was done by hand (std mmff 10105 → featurize mmff 10106).

## Fixes (FIXES.md X1–X13, applied 2026-10-07)

| Fix | Commit | Where | What |
|---|---|---|---|
| X1 | 9da0295 | `tools/make_l_seed_pickles.py` `build_interp`; `tools/lgeom.py` `pair_check_grid`, `seed_sources` | Molecules are no longer dropped for one unsafe pair. λ0/λ1 and A5 use every matched pair. λ ∈ {0.25, 0.5, 0.75} replace unsafe pairs by the molecule's safe pairs (pair_ok checked on that λ grid); a molecule is dropped only if it has no safe pair. ETKDG retries with random coordinates. Writes per-set kept/replaced/dropped counts and `S4_test_subset.txt`. |
| X2 | 045dc3a, 981556f | `slurm/submit_round2.sh` (6 CPUs / 30000M train; 8 / 40000M eval); `r2_eval_array`, `build_round2_data`, `smoke_round2`, `featurize_qm9` headers | Every request is ≤ 5000 MB per CPU. Checked by the dry run: all sbatch headers and every generated sbatch command. |
| X3 | 045dc3a | `slurm/common.sh` `run_evalset`; `ablation_train_array.sbatch` (`R2_STRICT=1`) | Every tag must have eval.pkl + SUMMARY, otherwise it is logged as `EVAL_FAILED` and the task exits non-zero. |
| X4 | 045dc3a | `common.sh` (default PACK=3; `code=`/`pack=` lines in provenance.txt), `r2_eval_array.sbatch`, `submit_round2.sh` | Packing on by default; provenance records the code commit and PACK. |
| X5 | 981556f | `slurm/r2_eval_models_head.tsv` | CR ×3 on S1A5. |
| X6 | 045dc3a | `common.sh:17-20` | WORK/RES come from `$PROJECT` only; the smoke job uses `R2_WORK`/`R2_RES`. On ada, `$WORK` is empty. |
| X7 | 981556f | `submit_round2.sh` train | B3 runs in its own array with `afterok:$MMFF_JOB` (10106), unless cache_mmff already exists. |
| X8 | 981556f | `submit_round2.sh` train | One panel job per model, `afterok` on that model's own training task. |
| X9 | 45cdb96 | `tools/s4_gate.py` | V18 passes if \|diff\| ≤ max(3 × seed SD, 0.003 Å) or the paired bootstrap CI contains 0. |
| X10 | ad7acda | `tools/analyze_round2.py` (+ `tools/tests/test_analyze_round2.py`) | Pre-declared contrasts and Holm families; secondary on the intersection; COV-R@0.05 exploratory; failure counts; seed SD on every row; hierarchical S2 non-inferiority; S4 subset re-scoring with the limitation note; dose-response against measured L error; MMFF_PRE counts. |
| X11 | 9da0295, 88eb0a1 | seed builder MMFF codes; `diffusion/sampling.py` `try_mmff` counters + `MMFF_PRE` line in `generate_confs.py` | Per-conformer MMFF outcomes are counted; the old sum let a -1 cancel a +1. try_mmff behaves as before. |
| X12 | 13e1de6 | `tools/local_structure_analysis.py` `leak_job` | Asserts that each generated conformer keeps its seed's bond lengths (seed order). |
| X13 | 45cdb96 | `tools/s4_gate.py` docstring | Updated for the pair_ok ruling. |

**Tests.**
- Local: 16 passed (lgeom 11, seed builder 4, analysis driver 1).
- Cluster: **30 passed**.
- Golden noise floors: model_out snapshot-vs-snapshot 7.0e-6 against patched 5.5e-7; generate coordinates 0.87 Å against 0.69 Å. The generate test is weak, because SDE trajectories diverge.
- Smoke 10180 passed (round-1 path, all 4 training flags, packed S1 sets).
- Full paired build (10122): PAIRING_CLEAN 99.999%; PAIR_OK 87.7%; cache_paired has 100% of rematch molecules.

**X1 population counts: pending.** Job 10375 (`slurm/x1_rebuild_seeds.sbatch`) was still running at handback. Its log `logs/tordiff_r2_x1seeds_10375.log` prints per-set `n_molecules … (% kept)` and the S4 subset size. The X11 MMFF counts and the determinism check of the etkdg pickles are in the same log. The old build kept 639/997 for every λ/A5 set.

**Dry run** (`DRY_RUN=1 MMFF_JOB=10106 bash submit_round2.sh head | train pass | train fail`):

| phase | jobs |
|---|---|
| head | 1 array: 8 tasks %4, 8 CPU / 40000M, PACK=3 |
| train pass | main array of 10 runs (%4, 6 CPU / 30000M, 3 workers); B3 array of 3 runs (`afterok:10106`); 12 panel jobs, each `afterok:<own training task>` |
| train fail | 7 + 3 runs; 9 panel jobs; S4 panels skipped |

Every request is ≤ 5000 MB per CPU. The fake-job-id counter in dry-run mode was fixed after this run.

**Estimate at the measured 6.4 it/s.**
- One training run: 82,895 training molecules / 32 = 2,591 iterations/epoch ≈ 405 s, plus validation ≈ 425 s/epoch, so **≈ 11.8 h per 100 epochs**. Round 1 took 10.6–11.9 h with 8 workers.
- In-job evaluations: 23 runs × 0.24 h.
- Head phase: 77 runs; post phase: 49 runs. Packed at 1.47× throughput, that is 0.163 GPU-h per run.

| | gate pass | gate fail |
|---|---|---|
| training | 13 × 11.8 = 153.4 GPU-h | 10 × 11.8 = 118 GPU-h |
| in-job evaluations | 5.5 GPU-h | 4.1 GPU-h |
| head | 12.6 GPU-h | 12.6 GPU-h |
| post | 8.0 GPU-h | 6.0 GPU-h |
| **total** | **≈ 180 GPU-h** | **≈ 141 GPU-h** |

The pass case is above the 175 cap; the overrun is reported rather than trimmed (FIXES).

**Wall clock (gate pass):** about 4 h for the head phase, then 13 training runs on 4 GPUs (4 waves × ~12.3 h); panels fill the last wave. That gives **≈ 50–55 h**. B3 cannot start before featurize 10106, at about 15:00 on 7 Oct.

**Not yet synced to the cluster:** the dry-run id fix (last commit).
