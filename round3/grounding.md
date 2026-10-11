# Round 3 P3: code grounding (grounder)

Written 2026-10-11 by the code grounder. Inputs: the 17 cards in `round3/cards/`, the final "Round 2 verdicts" sections of
`round3/ledger/verify_V1.md`, `verify_V2.md`, `verify_V3.md`, and `round3/ledger/disputes.md` (D-210 OPEN). Code read at
the repo state of 2026-10-11 (`torsional-diffusion/`, `tools/`, `slurm/`, `round2/IMPLEMENTATION.md`). Cluster checked
read-only through `srun -n 1 -c 1 --mem=2G -t 00:10:00 -A plafnet2 -p plafnet2 -w gnode118` (jobs 13056, 13057, 13059,
16419, 16424). Nothing was installed, submitted, edited or committed on the cluster. The only computation run there was
20 GFN2-xTB optimisations (job 13057), in a throwaway `/tmp` directory that was deleted afterwards.

Paths are relative to `C:\Users\HP\Desktop\tor_diff`. `TD` = `torsional-diffusion/`. "packed run" = one condition
(generate + evaluate) inside a `PACK=3` evaluation job.

## 1. Summary table

| Card | Scout call | Grounder verdict | New / changed code | GPU-h | CPU-h | Main condition or change |
|---|---|---|---|---|---|---|
| C-EVAL-01 | YES | IMPLEMENTABLE WITH CHANGES | ~200 lines (2 tools), 1 test | 0 (+0.5 optional) | 1–2 | Fit L error within molecules on the 906-molecule λ population, minus the 258 molecules with reused λ seeds; make L error the same statistic in every condition; fix R2-1 wording |
| C-EVAL-02 | YES | IMPLEMENTABLE WITH CHANGES | ~150 lines, 2 tests | ~0.9 (card: 0) | 2–4 (unmeasured) | Item (d) needs new random-GT runs for S3/CR; (a) is not exactly constant for same-L arms; pairing check via seed bond lengths |
| C-EVAL-03 | YES | IMPLEMENTABLE WITH CHANGES | ~250 lines (xtb tool + val-set builder), 3 tests | ~0.5 (card: 0) | ~7 (card: 22–67) | xtb 6.7.1 binary exists on gnode118 (no install); fix `utils/xtb.py`; add a graph/stereo filter (the evaluator has none for generated conformers); a validation subset needs a new val-set path plus small inference runs |
| C-EVAL-04 | YES / YES / MAYBE | B-mmff AS SPECIFIED; B-xtb and B-clust WITH CHANGES | 0 / shared with C-GEOM-01 / ~150 lines + 3-line flag | ~2.6 (TD+cluster arm only) | ~10–13 | B-mmff = `--no_model --no_random --pre_mmff`; B-clust K-means feature space unspecified; fix R2-2 (25 %, not 30 %) |
| C-EVAL-05 | YES | IMPLEMENTABLE WITH CHANGES | ~60 lines (fix TD's `optimize_confs.py`, dedup count) + shared filter | 0 | ~2 per arm (card: 8–25) | `optimize_confs.py` hard-codes the DRUGS csv, truncates at 32 conformers and has a flag bug |
| C-EVAL-06 | NO / not R3 | NOT FEASIBLE THIS ROUND (agrees with NO) | — | — | — | (b) hooks exist (`--xtb`); ORCA 6.1.1 module listed on ada (untested) if the panel overrules |
| C-GEOM-01 | YES | IMPLEMENTABLE WITH CHANGES | ~120 lines builder + shared xtb tool + gate tool (~150) | 1.0–1.9 | ~5 seeds + ~2 gate | Write to a new seed dir (round-2 provenance hashes); S4 λ "chosen on validation" has no code path, so pre-declare λ |
| C-GEOM-02 | YES | IMPLEMENTABLE WITH CHANGES | ~450–550 lines, 3 tests | 1.0–1.9 | 3–9 (library unmeasured) | Build the transplant as the A5ring recipe with a template ring; mirror/stereo handling; per-key cap; leakage assert |
| C-GEOM-03 | YES (conditional) | CONDITIONAL: WITH CHANGES only if D-210 step 0(a) is conclusive; else NOT FEASIBLE THIS ROUND | ~200 lines, separate env | ≤ 1 (ET-Flow, unmeasured) + 1.0–1.9 | ~1 | Our cluster data cannot map `QM9.zip` indices to SMILES; downloading QM9.zip needs user permission; env risk lower than stated (driver 580, internet OK) |
| C-GEOM-04 | NO (R3) | NOT FEASIBLE THIS ROUND (agrees) | — (two-stage reopen needs 0 TD lines) | — | — | Reopen criteria are computable with the C-GEOM-01 gate tool |
| C-GEOM-05 | NO | NOT FEASIBLE THIS ROUND (agrees) | — | — | — | Install argument weaker than stated (driver supports CUDA 12, internet OK); level-of-theory argument stands; xTB is 0.14 s/opt |
| C-GEOM-06 | MAYBE | IMPLEMENTABLE WITH CHANGES | ~30 lines | ≤ 1 (gated) | ~1–2 | In RDKit 2022.9.5 `srETKDGv3()` also turns macrocycle torsions OFF; single-factor needs `ETKDGv3()` + `useSmallRingTorsions=True`; same `randomSeed` ≠ common random numbers |
| C-ROBUST-01 | YES (gated) | IMPLEMENTABLE WITH CHANGES | Arm 0: config only; Arm 1: ~200 lines incl. tests | 2.6 (Arm 0) + 26.8 (Arm 1) | ~2 | Add hybrids in a post-pass over the existing paired pickles; per-hybrid residual and clash filters; new cache variant |
| C-ROBUST-02 | MAYBE | IMPLEMENTABLE WITH CHANGES (config only) | ~10 TSV/driver lines | ~26.5 (+2 for S4 controls at λ 0.25/0.75) | 0 | Drop `--l_level`; evaluate on S1 sets; λ "picked on validation" has no code path; gate data now exist |
| C-ROBUST-03 | YES (gated) | IMPLEMENTABLE AS SPECIFIED (config level) | ~10 slurm lines | ~26 | 2–5 | New paired-MMFF std dir, cache and VARIANT; gate data (S5 ×3) now exist |
| C-ROBUST-04 | NO | Guided variant NOT FEASIBLE THIS ROUND; score interpolation WITH CHANGES (~40 lines) | — | (3–5 if overruled) | — | Agrees with NO |
| C-ROBUST-05 | NO | Ratio sweep AS SPECIFIED (0 lines); curriculum WITH CHANGES | — | (~73 if overruled) | — | Agrees with NO |

Cost basis (measured, section 2.2): one 100-epoch training ≈ 11.7 GPU-h; one packed condition ≈ 0.17 GPU-h; one unpacked
condition ≈ 0.26 GPU-h; one GFN2-xTB optimisation of a QM9 seed ≈ 0.14 s median, ≤ 0.4 s, single core. CPU-h for xTB
work adds ~0.1 s per conformer of Python/IO overhead (0.3 s per conformer used throughout).

## 2. Shared findings

### 2.1 Cluster facts (read-only checks, 2026-10-08 and 2026-10-11)
- **gnode118 runs Ubuntu 22.04.5** (the login node ada is Rocky 9.8). The ada module tree has `u22/xtb/6.7.1`,
  `u22/crest/gnu-12` and `u22/orca/6.1.1`. `module` is not initialised in a non-login `srun bash` (the `module load`
  failed silently), but the binary `/data3/u22/xtb-6.7.1/bin/xtb` is mounted and runs on gnode118. So **no xtb
  installation is needed**: call it by absolute path, with `LD_LIBRARY_PATH=/data3/u22/xtb-6.7.1/lib`. Version 6.7.1
  (edcfbbe), not GEOM's 6.2.3 [E-EVAL-056]. CREST and ORCA were only seen in the module list; they were not run.
- **The TD venv on gnode118 is RDKit 2022.09.5, numpy 1.23.5, torch 1.13.1+cu117** (printed by `$PROJECT/venv/bin/python`).
  `rdkit.Chem.rdDetermineBonds` imports, so 3D connectivity re-perception is available for the graph checks below.
- **NVIDIA driver 580.178.04** (`/proc/driver/nvidia/version`). It runs CUDA 11.8 and 12.x builds. **gnode118 has
  outbound internet**: `https://pypi.org/simple/etflow/` and `https://zenodo.org/api/records/14226681` both return 200.
  `$PROJECT/micromamba/bin` exists. Separate envs (ET-Flow, ML potentials) are therefore an install-permission question,
  not a technical blocker.
- **Data:** `data/QM9/qm9/` holds 133,232 raw pickles. `data/QM9` holds TD's `split.npy`, `test_smiles.csv`,
  `test_mols.pkl`, all round-2 caches and std dirs, and `round2_seeds/` (all S1 pickles + `l_error.csv`). There is
  **no GEOM `summary_qm9.json` and no ET-Flow file** (relevant to D-210).
- **Round 2 is finished.** Every round-2 training run has `.train_done`: S3 s0/s1, S4 s0–s2, B3 s0–s2, B1cap s0, B6_0.02
  s0. Their panels exist: S3 S1panel ×2, S4 S4core ×3, B3 `S5_nommff` ×3, B1cap S1panel4. This grounding does not touch
  those jobs or dirs.
- **Cache sizes:** `cache_paired.train` is 1.32 GB, `cache_rematch.train` 0.94 GB, `cache_raw.train` 1.31 GB.

### 2.2 Measured timings
- **Training:** a round-2 training log has `Epoch 99: TIMING epoch_time 377.7s ... iters 2591 (6.86 it/s)`; with
  validation this gives the brief's ≈ 11.7 h per 100 epochs. Used: **11.7 GPU-h per training run**.
- **Packed evaluation** (`logs/tordiff_r2_eval_*.log`, `done: ... wall`): 6 conditions in 3,489–3,584 s and 4 conditions
  in 2,532–2,571 s give **≈ 0.17 GPU-h per condition**. A single unpacked condition takes 940–946 s (**0.26 GPU-h**).
- **GFN2-xTB** (job 13057): 10 random test molecules, single thread (`OMP_NUM_THREADS=1`), `xtb in.xyz --opt normal
  --norestart`, wall time per call including process start-up:
  - All 20 calls returned rc 0 and wrote `.xtboptok`.
  - GT conformer 0 of each molecule converged after 1 iteration with 0.000 Å heavy-atom displacement. On this probe,
    the GEOM-QM9 references are already GFN2-xTB 6.7.1 minima.
  - ETKDG seeds needed 17–63 iterations and moved by 0.09–0.32 Å heavy-atom RMSD.
  - Median 0.135 s, mean 0.157 s, max 0.383 s per optimisation.
  - The cards assumed 0.5–3 s; measured cost is 5–15× lower.
  - Caveat: n = 20 on 10 molecules. The 60 strained cages have no ETKDG seed, so they are not in any xTB-on-seed workload.

### 2.3 Shared infrastructure several cards need (build once)

**S-1. xTB runner.** Used by C-GEOM-01, C-EVAL-03, -04 (B-xtb), -05, and C-GEOM-02 (optional `_xtb` variant). TD has
`utils/xtb.py:xtb_optimize` [E-GEOM-010, E-EVAL-034] and `optimize_confs.py`. Neither is safe to use as is:
1. `utils/xtb.py:5-7` sets the work dir `my_dir = /tmp/<pid>` **at import**. Pool workers forked after the import share
   the parent's pid dir. They then write and read the same `xtb.xyz` / `xtbopt.xyz` (`:44-45`). Another molecule's
   optimised coordinates can be read back silently whenever two molecules have the same atom count, which is common in
   QM9. This affects `make_l_seed_pickles.py`, which uses a `Pool` (`:309-310`). **Fix:** use one
   `tempfile.mkdtemp()` per call.
2. There is no `--norestart` (`:49`), so a stale `xtbrestart` from the previous molecule sits in the reused cwd. Pass
   `--norestart`, as the probe did.
3. Convergence is never checked (`:47-64`). Use `.xtboptok` (present on all 20 probe runs) and the return code. Store
   `xtb_converged` per conformer.
4. The runtime parse `out.split(b'\n')[-12]` (`:51-52`) happens to hit the total cpu-time line in 6.7.1 (probe output,
   line −12). It is version-fragile, though. If it throws, the function returns `None` **before** writing coordinates,
   so a caller that ignores `None` silently keeps the unrelaxed seed. `optimize_confs.py:41-42` drops the conformer
   instead. **Fix:** drop the parse and count every fallback.
5. There is no `--chrg`. Pass the RDKit net formal charge, plus `--uhf 0`. Do not add `--alpb` (gas phase, as GEOM
   [E-EVAL-055]).
6. `xtb_optimize` writes into `mol.GetConformer()` (`:59`), the default conformer. Pass single-conformer mols
   (`make_l_seed_pickles.split_confs`, `:73-74`).
7. Set `OMP_NUM_THREADS=1` for each xtb process when running many in parallel.

Estimated size: ~150 lines (`tools/xtb_relax.py`, importable from both venv tools). Tests:
- GT pass-through: displacement < 1e-3 Å and E_relax < 0.05 kcal/mol;
- two-worker Pool collision test;
- charged-molecule test;
- non-convergence fallback.

**S-2. Graph/stereo filter for relaxed or transplanted structures.** The cards C-EVAL-03 §4, C-EVAL-04 §4 and C-EVAL-05
§4 say "the evaluator's SMILES filter must run after relaxation". **Our evaluator filters only GT conformers**:
`clean_confs` is applied to `true_mols` at `evaluate_confs.py:86`, and model conformers enter `worker_fn` unfiltered
(`:106-115`). `GetBestRMS` uses the RDKit graph carried by the mol object, not the coordinates, so a bond broken by xTB
shows up only as a large RMSD, never as a failure. Needed: re-perceive connectivity with
`rdDetermineBonds.DetermineConnectivity` (available on the cluster RDKit), compare the heavy-atom graph, and compare
`lgeom.stereo_labels` (`tools/lgeom.py:188`) before vs after. A failing conformer falls back (seeds) or is dropped and
counted (post-relaxation). ~60 lines and 1 test (a deliberately stretched bond).

**S-3. Seed-gate tool.** Used by C-GEOM-01/02/03/06, the reopen rule of C-GEOM-04, and B-xtb. Everything the D-205
gate needs exists, except two items: the common rigid set and pucker coverage.
- (i) Ring/acyclic bond and angle RMSD are already in `l_error.csv` via `lgeom.l_subset_errors` (`lgeom.py:317-338`),
  for any new tag added to the F5 loop (`make_l_seed_pickles.py:107-114`).
- (ii) The > 10° share, on both bases, is computed from the same rows.
- (iii) **Pucker coverage** (min over seeds per GT conformer of ring-dihedral RMSD) is new: ~25 lines using
  `lgeom.internal_sets` / `_dih_deg` (`:272-314`). Note that the existing ETKDG/MMFF rows use a Hungarian assignment
  on heavy-angle RMSD (`make_l_seed_pickles.py:106-114`, E-GEOM-009): only L of the 2L seeds get a row.
- (iv) **Seed AMR-R** needs no new scoring code. Run
  `generate_confs.py --seed_confs <pickle> --seed_confs_cycle --no_model --no_random` (CPU; `generate_confs.py:20-21`,
  `:100-105`, `:150-153`), then `evaluate_confs.py --out_results`. Both are the same code path as every arm (round-1
  A1c sanity rows did exactly this).
- **Common rigid set:** TD's own criterion is the empty `edge_mask` from `get_transformation_mask`
  (`TD/utils/torsion.py:9-39`, applied in `diffusion/sampling.py:58-59`, skipped at `generate_confs.py:150-153`).
  Compute it from the seed pickle's GT-graph mol, **not** from `tools/breakdown.py`, which bins by corrected SMILES
  (`breakdown.py:33-50`); that difference is D-205's ≤ 0.006 Å model leak. V3's advisory applies: a heavy-atom
  criterion would be "every TD-rotatable bond has exactly one heavy atom on its smaller side". It keeps ≈ 30 % of the
  set, because H-rotor rotation never changes heavy-atom RMSD. Report n under both criteria.
- Size ~150 lines; CPU ≈ 0.25 h per scored seed set.

**S-4. Round-3 eval sets and seed paths.** `r2_eval_array.sbatch:23` lets `R2_EVALSETS` be overridden, so add
`slurm/r3_evalsets.tsv` and a `QM9_SEEDS3` variable in `common.sh` (the missing-pickle check at
`r2_eval_array.sbatch:40-43` matches any `$QM9_[A-Z0-9_]*/` path). **Never write into `round2_seeds/`.** The provenance
guard hashes every seed-pickle argument (`common.sh:111-114`), so overwriting a round-2 pickle makes every later
re-evaluation of a round-2 dir fail as "produced from other inputs", and it breaks the paired-by-pickle analysis.

**S-5. Validation-set path (does not exist).** Several cards make a rule of the form "chosen on validation, never on
test":
- C-GEOM-01 step 3 (S4 λ);
- C-ROBUST-02 (S4 λ for MMFF);
- C-GEOM-02 (template key level);
- C-EVAL-03 (200 validation molecules).

Every eval set in `slurm/r2_evalsets.tsv` runs on `test_smiles.csv` (`common.sh:39-40`, `:126-133`), and no tool writes
a validation csv or GT pickle (searched `tools/`, `slurm/`). Building one is ~70 lines:
1. reproduce TD's split exactly as `TD/utils/dataset.py:167-179` does: `sorted(split.npy[1])` indexing
   `sorted(glob(raw/*.pickle))`;
2. write `val_smiles.csv` (same 3 columns as the test csv) and `val_mols.pkl`;
3. build val seed pickles with `make_l_seed_pickles.py --test_csv/--true_mols` pointed at them.

Until this exists, any "chosen on validation" rule must be replaced by a pre-declared value (e.g. S4 λ = 0 on every
non-oracle L, as round 2 did).

**S-6. Round-3 analysis driver.** `tools/analyze_round2.py` hard-codes run names, seeds and contrasts (`:43-89`).
Round-3 contrasts need a copy with pre-declared contrasts, committed before results exist (same pattern).

## 3. Per-card grounding

### C-EVAL-01: literature reporting block and "how good must L be"
**Touchpoints**
- `tools/paired_compare.py` already gives δ ∈ {0.05, 0.1, 0.25, 0.5} (`:32`), an intersection universe (`:41`,
  `:105-106`) and failure counts (`:42`, `:107-113`) [E-EVAL-037]. Its per-molecule table (`:46-60`) has AMR means
  only. **Add** `tools/report_block.py` (~80 lines). It reads `eval.pkl` (written at `evaluate_confs.py:188-191`) and
  prints, per arm:
  - AMR-R/P mean and median, on the union and on the intersection;
  - COV-R/P at the four thresholds;
  - `n_evaluated` / `n_model_failures` / `n_additional_failures`, as in `evaluate_confs.py:174-175`;
  - the literature rows as constants tagged with their E-IDs (E-EVAL-001…008).
- **Add** `tools/l_error_regression.py` (~120 lines). It joins per-molecule AMR-R (same reduction as
  `analyze_round2.py:95-107`) with per-molecule L error from `round2_seeds/l_error.csv`. The keys differ:
  `l_error.csv` uses the csv column-0 `smiles` (`make_l_seed_pickles.py:114, 231`), while `eval.pkl` is keyed by
  `(smi, corrected)`. Map them as `analyze_round2.py:166-167` does, and assert ≥ 99 % join coverage.

**Transfer check**
- Literature rows are δ = 0.5 Å TD-protocol AMR-R means over successes [E-EVAL-001…005]. Our AMR is a `nanmean` over
  successes and our COV counts failures as 0 (`evaluate_confs.py:166-171`, [E-EVAL-033]). Compatible, with the
  population caveat the card already states.
- **The regression x-axis is not one statistic across conditions.** For λ / A5 / noise sets every seed has an L-error
  row (`make_l_seed_pickles.py:226-231`). For ETKDG/MMFF only the L Hungarian-assigned seeds of the 2L do
  (`:106-114`, E-GEOM-009), and these are the best-angle-matching seeds, biased low. Either recompute ETKDG/MMFF L error
  over all 2L seeds (each against its best GT) or plot them as labelled reference points only.
- **Cross-molecule regression confounds L error with molecule difficulty**: ring-rich molecules have both higher
  ring-dihedral error and higher AMR-R. The λ series gives each molecule several L levels, so fit within molecules
  (per-molecule curves, or a mixed model with molecule intercepts; bootstrap over molecules).
- **Coverage of the λ series:**
  - CR ×3 and B1 ×3 have λ 0 / 0.25 / 0.5 / 0.75 (`r2_eval_models_head.tsv`, S1core);
  - λ = 1.00 exists for CR ×3 (S1CR) and B1 s0 only (V18);
  - S3 has λ 0 / 0.5 only (S1panel; checked on gnode118).

  The λ 0.25–0.75 sets have 906 molecules, λ 0 / 1 have 955 (IMPLEMENTATION X1). Restrict the fit to the 906-molecule
  intersection; this removes the population mixing V1 bounded (0.851–0.863).
- **Seed reuse in the interpolating λ sets** (found by the impact analyst, confirmed in code). In λ 0.25 / 0.5 / 0.75
  an unsafe pair's slot i is filled with the blend of another, safe pair (`make_l_seed_pickles.py:218`, `:224-226`),
  and its `l_error.csv` row is measured against that source pair's GT (`:231`, `err_row(..., al['Y_al'], ...,
  src_gt_idx=src)`). This affects 1,268 slots in 258 molecules (IMPLEMENTATION X1). Under `--seed_confs_cycle` those
  slots also lose their own-conformer link. For a clean dose axis, fit on the 648 molecules with no replaced slot, or
  add a sensitivity row with them excluded; `meta['replaced']` (`:228`) identifies them.
- The random-GT-conformer ceiling exists for B1 (round-1 GT_EVAL = 1, `ablations_train.tsv`) but not for CR (GT_EVAL 0
  in round 1; gtLcycle only in round 2, `ablation_train_array.sbatch:127-137`). Optional: CR ×3 random-GT runs,
  3 × 0.17 = 0.5 GPU-h.

**Slip to fix (V1 R2-1):** §3/§5 "CTRL ... never reaches the ET-Flow range (0.0803 Å at λ = 1)" is wrong: 0.0803 lies
inside 0.073–0.083. Correct reading (V1 `v1_lambda.py`): CTRL touches the 0.083 Å re-run level only at λ ≈ 0.98 and never
reaches the published 0.073 Å.

**Reuse:** `paired_compare.py`, `analyze_round2.py` helpers, every existing `eval.pkl`, `l_error.csv`. No new runs.
**Size / risk:** ~200 lines plus 1 synthetic test (a known slope must be recovered). Silent-failure modes:
- a raw-vs-corrected key mis-join that drops molecules unevenly;
- reading the cycled oracle as reachable;
- extrapolating the B1 curve past its single λ = 1 seed.

**Verdict: IMPLEMENTABLE WITH CHANGES.** The changes are the within-molecule fit on the 906-molecule population, a
homogeneous L-error statistic, and the R2-1 wording. **Cost:** 0 GPU-h (0.5 if CR random-GT runs are added);
1–2 CPU-h.

### C-EVAL-02: local-geometry, stratified and conformer-level diagnostics
**Touchpoints**
- (a) `tools/geometry_metrics.py` already pairs every GT conformer with its RMSD-matched generated conformer and back
  (`:124-137`), using the lowest-RMSD heavy-atom map (`:103-122`, [E-EVAL-052]). It reports heavy bond/angle/torsion
  **MAE**. **Add** ring/acyclic split and endocyclic-dihedral **RMSD** by calling
  `lgeom.l_subset_errors(gen_h[0], gen_xyz[k], gt_xyz[l][list(amap)])` inside `pair()`. That puts the numbers on the
  same scale as `l_error.csv` (~25 lines; keep the MAE columns).
- (b) Micro AMR = mean over all GT rows of `min_k RMSD[l, k]` from the `eval.pkl` matrices (~15 lines next to
  `paired_compare.per_mol`, `:46-60`). Strata: use the TD `edge_mask` rigid flag from S-3 and ring/acyclic from the GT
  mol's `RingInfo`. Do not use `breakdown.py`'s corrected-SMILES bins (D-205).
- (c) Improvement/downgrade per generated conformer: `d_k = min_l RMSD[l, k]` per arm; rates at 0.02–0.20 Å
  (~40 lines).
- (d) Random-GT runs, see below.

**Transfer check**
- "Generated bond lengths/angles are exactly the seed's": true in TD. Torsion updates are rigid rotations about
  bridges (`utils/torsion.py:9-39`), and round 2 asserted it (`local_structure_analysis.py` leak_job, FIXES X12)
  [E-EVAL-036]. **But (a) is not exactly constant for arms that share L** (card §3, S3 vs CTRL). The matched GT
  conformer is chosen by RMSD, which depends on the model's torsions. Expect small, non-zero differences.
- (c) pairing by index (FM-refiner pairs a sample with its own refined version [E-EVAL-009]). In our code:
  - `eval.pkl` column k is generated conformer k in generation order (`generate_confs.py:170`,
    `evaluate_confs.py:93, 106-115`);
  - with `--seed_confs --seed_confs_cycle`, conformer k always gets seed `k mod len` (`diffusion/sampling.py:86`);
  - the MMFF twin pickle's seed k is the relaxed seed k of the same embedding (`make_l_seed_pickles.py:93-103`; key
    sets identical, `:331-333`).

  Pairing is therefore valid for S3 vs CTRL on one pickle and for CTRL-MMFF vs CTRL-ETKDG. Initial torsions are common
  random numbers only while both runs consume `np.random` / `torch` identically (`perturb_seeds` `sampling.py:105-110`;
  SDE noise `:191`), and a molecule skipped in one arm shifts the stream from then on. Validity of the pairing does not
  need that, only the same seed per index. **Check:** bond lengths of conformer k must agree between arms to 1e-5 Å.
- (d) Round-2 in-job evaluations ran the cycled GT oracle only (`GT_EVAL=2`, `ablation_train_array.sbatch:127-137`),
  so S3, S4 and CR have **no random-GT run**. Add `--seed_confs $QM9_GT_SEED_CONFS` without `--seed_confs_cycle`:
  S3 ×2 + CR ×3 = 5 × 0.17 ≈ **0.9 GPU-h** (the card says 0).
- Zero check: round-1 `A1c_gtL_cycle_sanity` (`--no_model --no_random`, `ablations_inference.tsv:23`) must give
  all-zero local errors.

**Size / risk:** ~150 lines plus 2 tests (GT zero check; pairing bond-length check). `geometry_metrics.py` has never
produced a synced output [E-EVAL-036], and its runtime is unmeasured; it does up to 200 substructure maps × matched
pairs per molecule (`:39`, `:93-94`). Time one arm before scheduling 12. **Verdict: IMPLEMENTABLE WITH CHANGES**
((d) needs runs; the (a)-constancy caveat; the (c) pairing check). **Cost:** ~0.9 GPU-h; 2–4 CPU-h (UNVERIFIED).

### C-EVAL-03: GFN2-xTB relaxation metrics (E_relax, displacement to own minimum)
**Touchpoints:** S-1 runner (`tools/xtb_relax.py`). It runs a single point plus `--opt` per conformer of a `confs.pkl`
or seed pickle and stores:
- E(input), E(opt), `xtb_converged`, the relaxed coordinates, `graph_ok` / `stereo_ok` (S-2);
- displacement bond / angle / ring-dihedral from `lgeom.l_subset_errors(mol, X_gen, X_relaxed)` (same atom order, no
  matching needed);
- heavy torsion displacement from `geometry_metrics.topology` (`:48-65`);
- heavy RMSD to its own minimum (`lgeom.rmsd_on` after `kabsch_fit`, `lgeom.py:65-78`).

Energies: Hartree → kcal/mol × 627.509, as `utils/xtb.py:36`. The "within 6 kcal/mol of the lowest GT" metric needs the
GT arm's energies, which the same tool produces. The validation subset needs S-5 plus inference runs on 200 validation
molecules for every learned arm: ~8 arms × ~0.2 of a run ≈ **0.3–0.5 GPU-h**. The card says 0 GPU-h, but there are no
generated validation conformers today.

**Transfer check**
- References are gas-phase GFN2-xTB minima [E-EVAL-012/013/055]. The runner uses gas phase (no `--alpb`, as
  `xtb_optimize` `utils/xtb.py:43-50`) and GFN2, xtb's default.
- Version 6.7.1, not 6.2.3 [E-EVAL-056]. The probe's GT pass-through (10/10 molecules, 1 cycle, 0.000 Å) says the
  zero check will hold; the full GT arm confirms it.
- E_relax in our evaluation needs no test GT, so it is non-oracle and usable for selection, but only on validation
  molecules.

**Cost:** 200 test molecules × ~25.4 conformers (25,312 ETKDG seeds over the test set, IMPLEMENTATION X11) ≈ 5.1k
conformers per arm. The 8 arms plus GT come to ≈ 43k calls × (0.3 s opt + ~0.05 s single point) ≈ **4.2 CPU-h**;
doubling for the validation subset gives **≈ 8 CPU-h**. The card estimated 22–67 CPU-h, so this is ~5× less, and under
1 h wall on 16 cores. **Size / risk:** S-1 + S-2 + S-5 ≈ 280 lines, 4 tests. Silent failures:
- the tmp-dir collision (S-1.1), which would make energies look plausible but belong to another molecule;
- unconverged optimisations counted as minima;
- graph changes scored as conformers;
- mean E_relax dominated by a few cages (report the median, as the card says).

**Verdict: IMPLEMENTABLE WITH CHANGES** (no install needed; wrapper fixes; graph filter; val-set path and small GPU for
the validation half).

### C-EVAL-04: non-learned baselines
- **B-mmff: AS SPECIFIED, zero code.** Run `generate_confs.py --no_model --no_random --pre_mmff --seed {0,1,2}`, the
  round-1 A0 line plus `--pre_mmff` (`ablations_inference.tsv:13`; `generate_confs.py:18, 20-21, 75`). For pairing with
  CTRL on the S1 set, the seed-pickle variant is
  `--seed_confs $QM9_SEEDS2/L_etkdg2L_mmff.pkl --seed_confs_cycle --no_model --no_random`.
  - `--model_dir PRE` is still required for the yaml merge (`generate_confs.py:85-88`); no model is loaded (`:100`).
  - With `CUDA_VISIBLE_DEVICES=""` it is a CPU job (`:73`).
  - Cost ≈ 0.3 CPU-h per run (evaluation-dominated).
- **B-xtb: WITH CHANGES.** It is the `L_etkdg2L_mmff_xtb` pickle of C-GEOM-01 run with `--no_model --no_random`. For 3
  RDKit seeds, run the builder with `--seed 1/2/3` (`make_l_seed_pickles.py:59`): ≈ 3 × 25.3k calls × 0.3 s ≈
  **6.3 CPU-h** (card: 25–70). Needs S-1 and S-2.
- **B-clust: WITH CHANGES.** New `tools/cluster_baseline.py` (~150 lines):
  - samplers 1:1:4 = ETKDG L + random torsions (the `--no_model` path, round-1 `A0_rdkitL_random_tors`), ETKDG, and
    ETKDG+MMFF;
  - N_e = min(20 N_ref, 2000), K-means to 2 N_ref [E-EVAL-025], with N_ref from csv column 1, as TD's 2K
    (`generate_confs.py:187`).

  **Our evidence does not fix the K-means feature space** (E-EVAL-025 quotes only "K-means with 2 N_ref clusters").
  Pre-register one, e.g. heavy-atom coordinates after `AlignMol` to the first sample, and output the member nearest
  each centroid (a centroid is not a valid conformer). Results depend on this choice. ≈ 2–5 CPU-h.
- **TD + clustering: WITH CHANGES.** `generate_confs.py:183-187` supports only `2 × n_confs` or a fixed
  `--confs_per_mol`, so add `--confs_factor` (default 2, ~3 lines). Generate 20K, cluster to 2K with the same tool, and
  evaluate. GPU ≤ 10 × 0.26 ≈ **2.6 GPU-h** per sampling seed (card: 3–4).
- **Transfer:** Zhou's numbers are GeoDiff-protocol [E-EVAL-026]; only the recipe transfers (card §4). The ETKDG
  version is v1 [E-GEOM-052], already recorded.
- **Slip to fix (V1 R2-2):** §3 "(30 % of all molecules) are rigid" should read "(25 % of all molecules; 30 % of
  molecules are rigid overall)".

**Verdict:** B-mmff AS SPECIFIED; B-xtb and B-clust/TD+cluster IMPLEMENTABLE WITH CHANGES. **Cost:** ≈ 2.6 GPU-h;
≈ 10–13 CPU-h in total.

### C-EVAL-05: xTB-after yardstick
**Touchpoints:** TD already ships `optimize_confs.py`, which runs `xtb_optimize` plus `xtb_energy` per conformer of a
`confs.pkl`. Fix four defects before using it on QM9:
1. `:24` hard-codes `data/DRUGS/test_smiles_corrected.csv`; add `--test_csv`.
2. `:16, :35` `--limit 32` truncates conformers per molecule. A QM9 2K above 32 would be cut silently and recall would
   change. Set the limit off.
3. `:40` reads `if xtb_energy:`, which tests the imported *function* (always true), not `args.xtb_energy`. The flag is
   ignored (harmless here, but wrong).
4. `:41-42` and `:44` drop a conformer on any failure without counting it.

Alternatively, reuse the S-1 runner on `confs.pkl`. Add the S-2 graph filter and the distinct-conformer count (pairwise
heavy `GetBestRMS` > 0.05 Å within a molecule, ~30 lines). The output `confs_xtb.pkl` goes to the unchanged
`evaluate_confs.py`: atom order and the mol graph are preserved, so `GetBestRMS` works as is.

**Transfer check**
- "TD's RMSD tables are unrelaxed": in our runs `--post_mmff` defaults to off (`generate_confs.py:19`) and `gen_eval`
  never passes it (`common.sh:126-128`) [E-EVAL-035]. Confirmed for our numbers.
- A3's MMFF-after hook sits at `diffusion/sampling.py:258-262`. Running xTB-after offline from `confs.pkl` is
  equivalent, except that `mol.rmsd` / energy fields are not recomputed (they are unused with `--no_energy`).
- Pre-register how dropped conformers count: dropped from both recall and precision matrices, `n_model` reported per
  molecule.

**Reuse:** every arm's `confs.pkl` exists:
- R0 s0 and A2 from round 1;
- CTRL s0 / S3 s0 / B1 s0 `S1_etkdg2L` from round 2 (checked: S3 tags present on gnode118);
- C-EVAL-03's 200-molecule relaxed set.

**Cost:** full set ≈ 25.3k calls × 0.3 s ≈ **2.1 CPU-h per arm** (card: 8–25); 5 arms ≈ 11 CPU-h; 0 GPU-h.
**Verdict: IMPLEMENTABLE WITH CHANGES** (fix `optimize_confs.py` or use S-1; S-2 filter; dedup metric).

### C-EVAL-06: DFT re-referencing (NO) and Boltzmann ensemble properties (not R3)
- (b) hooks exist: `generate_confs.py --xtb` (`:35`, `:176-179`) calls `populate_likelihood` → `xtb_energy` (dipole,
  gap; `diffusion/likelihood.py:114-130`), and `utils/boltzmann.py` has the resampler. Our repo has **no**
  ensemble-property aggregation script. Requirements for running (b):
  - that script (~100 lines);
  - `NO_ENERGY=0`, because `gen_eval` defaults to `--no_energy` (`common.sh:123`);
  - `--likelihood` with `--ode` (`generate_confs.py:60-61`);
  - QM9 molecules with 0 rotatable bonds are then dropped (`generate_confs.py:171-174`). This changes the population;
    it is the round-1 bug noted at `:89-91`.
- (a) ORCA 6.1.1 appears in the ada module list (`u22/orca/6.1.1`). It was not run on gnode118; r2SCAN-3c timing is
  unmeasured.

**Verdict: NOT FEASIBLE THIS ROUND** as a round-3 endpoint (agrees with the card's NO / not-R3). If the panel
overrules, (a)'s 50-molecule probe needs an ORCA test plus ~tens of CPU-h (UNVERIFIED), and (b) needs the aggregation
script and the population caveat above.

### C-GEOM-01: GFN2-xTB-relaxed ETKDG seeds as test-time L
**Touchpoints**
- `tools/make_l_seed_pickles.py` `build_etkdg` (`:83-125`). Add `--xtb_bin` (default None, which keeps the current
  behaviour). After `seeds` (`:93`) and `seeds_mm` (`:103`), relax each single-conformer seed with S-1, giving
  `L_etkdg2L_xtb` and `L_etkdg2L_mmff_xtb`. On non-convergence, graph change or stereo flip (S-2), fall back per seed to
  the unrelaxed parent and record meta fields `xtb_converged`, `xtb_fallback`, `xtb_s`, `E_xtb`. Fallback keeps the key
  sets identical, so the twin assert (`:331-333`) can be extended to the xtb tags.
- Add both tags to the F5 loop tuple (`:107`). Their `l_error.csv` rows then come for free, using the same Hungarian
  rule as ETKDG/MMFF.
- Run with `--sources etkdg --seed 1` into a **new** `--out_dir` (`data/QM9/round3_seeds`, S-4). The etkdg pickles
  reproduce bit-identically (IMPLEMENTATION X1 log), so assert that the new `L_etkdg2L.pkl` equals the round-2 file;
  this makes the xTB seeds common-random-number twins of the round-2 ETKDG/MMFF seeds.
- Gate: S-3. GPU: `slurm/r3_evalsets.tsv` lines `S1_etkdg2L_xtb | 20 | 0 | --seed_confs $QM9_SEEDS3/L_etkdg2L_xtb.pkl
  --seed_confs_cycle` (and `_mmff_xtb`), plus `r3_eval_models.tsv` (CR ×3, B1 ×3, S3 ×2, S4 ×3).
- **S4 needs `--l_level`** (`generate_confs.py:106-110`), and the card's "λ chosen on validation" has no code path
  (S-5). Pre-declare λ = 0, the round-2 rule for every non-oracle L, or build S-5 first.

**Transfer check (source assumption → our code)**
- GEOM references are gas-phase GFN2-xTB minima [E-GEOM-015/016, E-EVAL-055]. The runner uses gas phase (no `--alpb`;
  `utils/xtb.py:43-50`, [E-GEOM-010]). xtb 6.7.1 vs CREST's 6.2.3. Probe: 10/10 GT conformers stay put (0.000 Å,
  1 cycle), so a seed that reaches the reference basin should land on reference L, as card §2 infers from
  [E-GEOM-018/019]. CREST's internal optimisation thresholds remain unknown (card §4). `--opt normal` vs `tight` is a
  pre-declared choice; tight costs little at 0.14 s/opt.
- L|τ coupling (card §4): xTB relaxes at the seed's ETKDG torsions, and TD then re-randomises the torsions
  (`generate_confs.py:150-151`, `perturb_seeds`). This is the same situation as the existing MMFF arm, so the comparison
  is like-for-like.
- Train/test shift for CTRL: CTRL was trained on unrelaxed ETKDG-v1 matched L (`standardize_confs.py:79`). xTB L is a
  shift, as MMFF L was.
- Rigid molecules: TD skips model and perturbation when `edge_mask` is empty (`generate_confs.py:150-153`), so seed =
  output and the S-3 seed AMR-R on the common rigid set is exactly what TD would return (D-205).
- Populations: the 60 ETKDG failures have no seeds in any etkdg-family pickle (`build_etkdg` returns None, `:90-92`),
  so ETKDG, MMFF and xTB share one population.

**Optional arms**
- B (pucker-diverse): `--seeds_per_gt 8` embeddings (a ~5-line change to `K = 2*n_csv` at `:88`), ≈ 100k calls ×
  0.3 s ≈ 8 CPU-h. Add a ring-dihedral fingerprint dedup and the 6 kcal/mol window from the xTB energies (~80 lines),
  then sample 2L.
- C (CREST): the `u22/crest/gnu-12` module is listed on ada and untested on gnode118. At 0.5 core-h per molecule
  [E-GEOM-016] that is ≈ 500 CPU-h, with a validity flag (it is the GT generator).

**Size / risk:** builder ~120 lines + S-1 (~150) + S-2 (~60) + S-3 (~150), 5 tests. Silent failures:
- the S-1.1 Pool collision (the builder uses a `Pool`, `:309-310`);
- fallback seeds not counted, so "xTB L" is partly ETKDG L (report the fallback share, and gate only if ≥ 95 % of seeds
  relaxed);
- overwriting `round2_seeds` and breaking provenance (S-4).

**Verdict: IMPLEMENTABLE WITH CHANGES** (xtb via the existing module path, no install; S-1/S-2 fixes; new seed dir;
pre-declared S4 λ or S-5). **Cost:**
- CPU: 2 × 25.3k calls × 0.3 s ≈ **4.2 CPU-h** plus builder overhead, so ≈ 5 CPU-h (< 30 min wall on 16 cores); gate
  ≈ 5 seed sets × 0.25 ≈ 1.3 CPU-h.
- GPU: 6 packed runs ≈ **1.0 GPU-h**; 11 runs with S3/S4 ≈ **1.9 GPU-h** (card: 1.5–3).

### C-GEOM-02: training-set ring templates
**Touchpoints**
- New `tools/ring_templates.py`, library step (~250 lines):
  - Training split exactly as TD reads it: `sorted(np.load(split.npy)[0])` indexing `sorted(glob(qm9/*.pickle))`
    (`TD/utils/dataset.py:167-179`), over 133,232 raw files on gnode118.
  - GT conformers per molecule from the raw pickles, cleaned as `clean_confs`.
  - Key = ring-system component (fused/spiro/bridged rings merged via shared atoms): `MolFragmentToSmiles` over ring
    atoms + ring bonds, isomeric and stereo-free.
  - Values = ring-atom coordinates clustered per key at < 0.1 Å after symmetry-aware Kabsch (substructure matches of the
    key), with counts.
  - Cap the instances per key (e.g. 2,000 random); common 3-/5-rings would otherwise dominate CPU.
  - **Leakage assert:** no non-isomeric canonical SMILES of a library source equals any test (`test_smiles.csv` col 2)
    or validation molecule.
- Transplant step: **reuse the A5ring construction instead of writing a new substituent-reattachment routine.**
  1. Embed the target molecule with `coordMap` pinning its ring atoms to a sampled template (RDKit, [E-GEOM-039]); this
     gives Y with the template ring.
  2. Copy the ETKDG seed's acyclic bonds/angles onto Y with `lgeom.set_internal_subset(base, Y, X_seed)`.

  That is the exact recipe of `L_A5ring_cyc_ORACLE` (`make_l_seed_pickles.py:236`; `lgeom.py:341-369`, which preserves
  ring bonds, angles and endocyclic dihedrals of the base exactly). The template arm is then directly comparable to its
  ORACLE upper bound A5ring (CTRL 0.119), with the same known 0.58° acyclic-angle residual [E-ROBUST-019].
- Output `L_tmpl` (+ `L_tmpl_xtb` via S-1) into `round3_seeds`; F5 rows via `err_row`; fallback = the unmodified seed
  (counted).

**Transfer check**
- OMEGA's assumption that ring conformations transfer between molecules sharing a ring system [E-GEOM-038/044] maps
  cleanly onto TD in one respect: **TD never changes ring geometry**. Ring bonds are never "rotatable" (removing a ring
  bond leaves the graph connected; `utils/torsion.py:16-18`), so an imposed pucker reaches the output unchanged
  (card §3's premise holds in our code).
- **Handedness:** a template pucker taken from another molecule can be the mirror image relative to the target's
  stereocentres. Check `lgeom.stereo_labels` (`:188-195`) after the transplant, retry with the reflected template, and
  fall back if both flip. Not in the card.
- `coordMap` embedding can fail for strained cages and fused 3/4-rings (card §4). Count failures by ring class.
- `set_internal_subset` swallows failures (`lgeom.py:358-366`, `except Exception: pass`). Store the F5 residual per seed
  and treat a large residual (pre-declare a threshold) as a fallback.
- Key level (ring-only vs + first shell) "chosen on validation" needs S-5, or choose it by training-split
  leave-one-out coverage (no test data).

**Size / risk:** ~450–550 lines + 3 tests:
- leakage assert;
- self-template round-trip: imposing a molecule's own GT ring must reproduce A5ring's L error to 1e-3 (ORACLE test
  only);
- mirror/stereo test.

Silent failures: train/val/test leakage through key-equivalent molecules (only identical molecules are excluded; that
is the intended non-oracle use), the stereo flips above, and uncounted fallbacks. Coding ≈ 2–3 days (card: 1–2).
**Verdict: IMPLEMENTABLE WITH CHANGES** (A5-style construction; mirror handling; per-key cap; leakage assert; S-5 or
leave-one-out for the key level). **Cost:** library 2–6 CPU-h (UNVERIFIED; conformer count not measured), transplant
≈ 1 CPU-h, `_xtb` ≈ 2 CPU-h; GPU 1.0–1.9 GPU-h (6–11 packed runs).

### C-GEOM-03: ET-Flow `qm9-o3` L as a diagnostic, with D-210 as a hard precondition
**D-210 grounding (step 0(a)).** TD's test set is `test_smiles.csv`. TD's split is `split.npy` indices into the
**sorted** raw file list (`TD/utils/dataset.py:168, 171, 179`). Per V3, `qm9-o3` trained from `QM9/train_indices.npy`
in ET-Flow's own processed order, which the old ET-Flow preprocessing derived from GEOM's `summary_qm9.json`. Our
cluster has **neither `summary_qm9.json` nor any ET-Flow file** (checked `data/QM9/`, `downloads/`). So:
1. Step 0(a) needs `QM9.zip` (2.9 MB, Zenodo 14226681). It is a download, so it needs the user's explicit permission;
   gnode118 can reach Zenodo.
2. If `QM9.zip` contains SMILES (or a SMILES-ordered processed list), intersect its train+val SMILES with our test
   column 2 (non-isomeric canonical) and with column 0. Zero overlap means PASS.
3. If it contains only index arrays, mapping them needs GEOM's `rdkit_folder` summary, a large download not on the
   cluster, plus a re-run of ET-Flow's old preprocessing. Treat that as **inconclusive = STOP** (orchestrator rule).
   Equal split *sizes* prove nothing.
4. ORACLE relabelling is not a substitute for the gate here. Memorised L would pass the λ = 0.75 spec, and that is
   exactly the card's question.

**Touchpoints if step 0 passes**
- Separate env: `$PROJECT/micromamba` exists. Python 3.10/3.11, numpy 1.26.4, torch 2.x with cu121 or cu118 (driver
  580.178.04 supports both), `pip install etflow` (internet OK). The install needs user approval in P5/P6.
- `tools/etflow_seeds.py` in that env (~80 lines): `predict()` on csv column 0 SMILES, 2L per molecule; write plain
  arrays (coordinates + ET-Flow atom order + SMILES) to `.npz`. Do not pickle RDKit mols across RDKit versions.
- TD-venv side (~120 lines, reusing `make_l_seed_pickles` helpers): map ET-Flow atoms onto the GT-graph mol (the same
  graph the round-2 seeds use, `make_l_seed_pickles.py:84-85`) by heavy-atom substructure match. Hydrogens follow their
  parent; equivalent H permutations are irrelevant to TD. Run `stereo_ok` vs GT (`:104`, `:116`) and drop flipped
  seeds. With `--seed_confs_cycle` fewer seeds still work (`sampling.py:86`); a molecule with 0 valid seeds becomes a
  failure, counted. Write `L_etflow.pkl` + F5 rows. Write the raw ET-Flow conformers as a `confs.pkl` for the unchanged
  evaluator (reference row).

**Transfer check**
- Chirality: ET-Flow's post hoc whole-molecule flip from RDKit tags [E-GEOM-053/054] can leave partial inversions, so
  `stereo_ok` is needed (card §4 ✓).
- ET-Flow predicts H positions; TD's L includes H. Fine.
- Our seeds use the GT graph's atom order, so the mapping is the only new risk. Add a test: map an ET-Flow-format
  output of a GT conformer back and require 0 RMSD.

**Verdict: CONDITIONAL.**
- If step 0(a) conclusively shows no TD test molecule in `qm9-o3`'s train/val: IMPLEMENTABLE WITH CHANGES (separate
  env; npz hand-off; stereo filter).
- If inconclusive or not permitted to download: **NOT FEASIBLE THIS ROUND**.

The environment risk is lower than the card's own §4 suggests (driver, internet, micromamba checked). **Cost:**
- ET-Flow sampling ≤ 1 GPU-h (unmeasured) + 1.0–1.9 GPU-h inference;
- ~1 CPU-h;
- env set-up ≈ half a day.

### C-GEOM-04: learned ring-pucker component (not this round)
A joint Cremer–Pople factor touches every TD layer:
- the score head returns per-edge torsion scores only (`edge_pred`, used at `diffusion/sampling.py:188-192`);
- the noise transform perturbs torsions only (`utils/dataset.py:71-76`);
- the sampler updates only rotatable bonds (`sampling.py:290-298` → `perturb_batch`), and `modify_conformer` rotates
  only bridges (`utils/torsion.py:9-39`).

So it means a new manifold, head, transform and sampler update, at or above the round-2 300-line estimate
[E-GEOM-049]. The two-stage reopening route (PuckerFlow rings as fixed substructures [E-GEOM-021]) needs **zero TD
changes**: it is one more seed pickle through `--seed_confs` (`generate_confs.py:22`, `sampling.py:84-91`), built with
the C-GEOM-02 transplant code. Reopen criteria 1 and 3 are computable with S-3 (> 10° share by ring class, common-rigid
seed AMR-R). **Verdict: NOT FEASIBLE THIS ROUND** (agrees with the card's NO). No cost this round.

### C-GEOM-05: ML-potential relaxation (not for QM9)
- Environment facts checked on gnode118: TD venv python 3.9 / torch 1.13.1+cu117 / numpy 1.23.5 [E-GEOM-014].
  Separate envs are possible: micromamba, internet, and driver 580 for CUDA-12 torch ≥ 2.8 as AIMNet2 needs
  [E-GEOM-036]. The install argument in card §4 is therefore weaker than written. The level-of-theory argument (GFN2
  references [E-GEOM-015/016] vs DFT-trained potentials [E-GEOM-027]) is unaffected by code.
- The cost argument is now measured: GFN2-xTB costs 0.14 s per QM9 seed (section 2.2), so an ML potential has no speed
  case at QM9 size.
- If overruled, the §5 CPU probe would reuse the S-1/S-3 pipeline with the potential in place of xtb: a separate venv
  plus ~60 lines, ≈ half a day.

**Verdict: NOT FEASIBLE THIS ROUND** (agrees with NO).

### C-GEOM-06: srETKDGv3 seeds
**Touchpoints:** `make_l_seed_pickles.py:89` calls `AllChem.EmbedMultipleConfs(emb, numConfs=K, randomSeed=args.seed)`
with no parameter object, which is ETKDG v1 on 2022.9.5 [E-GEOM-052]. Add `--etkdg_params {v1,v3,v3sr}` (default v1
keeps the call unchanged): build a params object, set `ps.randomSeed = args.seed`, call `EmbedMultipleConfs(emb, K, ps)`,
and name the tags `L_etkdgv3_2L` / `L_sretkdg2L` (+ `_mmff`). ~30 lines; output to `round3_seeds`.

**Transfer check (measured on the cluster venv, RDKit 2022.09.5):**

| params object | ETversion | small-ring | macrocycle |
|---|---|---|---|
| `ETKDG()` | 1 | False | False |
| `ETKDGv3()` | 2 | False | True |
| `srETKDGv3()` | 2 | True | **False** |

So the card's "v3 → srv3 (small-ring terms only)" is **not single-factor** in our RDKit: `srETKDGv3()` also switches
macrocycle torsions off. For a single factor, use `ps = ETKDGv3(); ps.useSmallRingTorsions = True`. QM9 rings are
small, so the macrocycle switch is probably inert, but the code should not rely on that.
- "Same random seed" does not give common random numbers: different parameter sets consume the RNG differently, so the
  variants are different embeddings. Pairing is at molecule level only. Embedding failures can differ (`:90-92`), so
  compare on the common set (S-3).
- The CSD-derived small-ring preferences [E-GEOM-050] vs the GFN2 reference stay a modelling caveat, not a code issue.

**Verdict: IMPLEMENTABLE WITH CHANGES** (the explicit small-ring flag; molecule-level pairing). **Cost:** CPU ≈ 1–2 h
(embedding + MMFF for 2 variants, as the round-2 builder); gate ≈ 1 CPU-h; GPU ≤ 1.0 GPU-h, only if the gate passes.

### C-ROBUST-01: error-matched, factorised L augmentation
**Arm 0 (gate), config only.** The seed pickles and eval-set lines exist: S1A5 (`r2_evalsets.tsv`, A5 lines) and
`S1_lam0.75_cyc_ORACLE`, which is only under the S1core label. Add a round-3 set label (S-4), e.g. `R3A0` with those
three lines, and a models TSV for S3 s0/s1 and B6_jit0.04pa s0–s2. These tags do not yet exist for these models: their
S1panel excluded A5 and λ 0.75 (checked on gnode118), so the provenance guard (`common.sh:107-122`) creates them
cleanly. 15 packed runs ≈ **2.6 GPU-h** (card ≈ 3). The ORACLE-gated experiment selection stays a validity-judge item.

**Arm 1 touchpoints**
- **Hybrids:** a post-pass `tools/add_hybrids.py` (~80 lines) over the existing `standardized_pickles_paired/NNN.pickle`.
  This is better than editing `build_paired_pickles.py`: re-pairing is avoided, and the RDKit and GT halves stay
  byte-identical to S3's. Per conformer:
  - `hyb_gtring_pos = lgeom.set_internal_subset(mol, Y_al, X)` (GT rings + RDKit acyclic; the A5ring recipe,
    `make_l_seed_pickles.py:236`);
  - `hyb_rdring_pos = lgeom.set_internal_subset(mol, X, Y_al)` (A5acyc recipe, `:237`), with `X = rd_mol` positions
    and `Y_al = conf['gt_pos_aligned']` (`build_paired_pickles.py:146`);
  - the F5 residual vs the source (`lgeom.l_subset_errors`) and a min non-bonded distance check (as `lgeom.pair_check`).

  Output: `standardized_pickles_paired_hyb/`; never overwrite `standardized_pickles_paired` (`cache_paired` is built
  from it).
- **Dataset:** `TD/utils/dataset.py` `featurize_mol`. Append the two hybrid lists in the **same loop iteration** as
  `pos` / `gt_pos` (`:306-311`), so the reacted-conformer filter (`:297-304`) keeps all four aligned. Extend the
  stale-cache guard (`:141-144`) to the hybrids when the new flag is on.
- **Transform:** `TorsionNoiseTransform.__init__` (`:38-47`) gets a new default-off `l_mix4` (four probabilities;
  `needs_gt=True`; mutually exclusive with `l_interp`, as `:44`). `_select_paired_L` (`:79-102`) picks one of the four
  positions with a single `np.random` draw. That keeps S3's RNG pattern, so `--resume` determinism carries over
  (train.py `load_resume_state` restores the RNGs, `:37-63`). Delete the hybrid lists from `data` after selection, as
  `:100-102` does for `gt_pos`, otherwise collation of the lists breaks.
- **Flags and slurm:** `utils/parsing.py` gets `--l_mix4` (`:30-37` block; the name must not collide with
  `generate_confs.py` args, per the comment at `:30-31`). `construct_loader` (`:347-352`) passes it via `getattr`.
  `slurm/featurize_qm9.sbatch:29-39` and `ablation_train_array.sbatch:53-63` get a `paired_hyb` case →
  `cache_paired_hyb`.
- **Tests:** extend `tools/tests/test_round2_model.py` C7 (paired selection formulas): the four endpoints are exact,
  the empirical probabilities are within tolerance, the default path is unchanged, and the stale-cache guard fires.

**Transfer check**
- The TD target is relative to the selected conformer (`dataset.py:74-76`, [E-ROBUST-016]), so every hybrid has a
  self-consistent score target. Card §4 asked the grounder to check torsion drift:
  - `rdMolTransforms.SetAngleDeg` rotates the moved fragment about the normal of the angle plane. A dihedral defined
    through that angle's own atoms is preserved; dihedrals about the same bond defined through other substituents shift
    by up to the angle change (a few degrees; RDKit acyclic angle error ≈ 2.8–4°, [E-ROBUST-006]).
  - So `hyb_gtring` carries GT torsions to within a few degrees, not exactly. This is consistent for training because
    the target is relative, but log the per-hybrid torsion difference to its base on TD's rotatable bonds (~15 lines).
- `set_internal_subset` silently skips failed settings (`lgeom.py:358-366`). Its known residual is 0.58° acyclic angle
  RMSD [E-ROBUST-019]. Pre-declare a residual threshold above which a hybrid falls back to its base (count it).
- Unsafe pairs (12.3 % of verified pairs fail `pair_ok`, IMPLEMENTATION Fixes) are kept by S3. Hybrids do not
  interpolate, so keep them too, behind the clash check.
- Card §4 (transplant reproduces ETKDG error only): code cannot change that. MMFF L stays the held-out test (S1 MMFF set
  exists).

**Memory:** `cache_paired.train` is 1.32 GB vs 0.94 GB for rematch, so each extra per-conformer position list costs
≈ 0.37 GB, and `cache_paired_hyb` ≈ 2.1 GB. Round-2 paired runs ran in 30,000 MB / 6 CPUs with 3 loader workers;
check MaxRSS in a smoke run.

**Cost**
- Hybrids: ≈ 82,895 training molecules (IMPLEMENTATION, featurized count) × ~6.8 conformers (000.pickle ratio) × 2 ×
  2.2 ms [E-ROBUST-015] ≈ 0.7 CPU-h, plus IO ≈ 1 CPU-h.
- Featurize: one serial CPU job (`featurize_qm9.sbatch:43-45`), ~1 h expected, 6 h limit.
- GPU: Arm 0 2.6 + training 2 × 11.7 = 23.4 + evaluation 10 conditions × 2 seeds × 0.17 = 3.4, so **≈ 29.4 GPU-h**
  (card: 33–40).

**Verdict: IMPLEMENTABLE WITH CHANGES** (post-pass hybrid builder; residual/clash/torsion-drift logging; new variant and
cache; ~200 lines including tests).

### C-ROBUST-02: blind λ interpolation (S4 without λ input)
**Touchpoints:** none in TD. The training line is
`S4blind_interp | paired | --l_interp --log_timing --fail_on_nan | | 2 | <seed> |`.
- Nothing couples `--l_interp` to `--lambda_embed_dim`: `dataset.py:44` only forbids mix+interp, and
  `score_model.py:194` skips λ when `lambda_embed_dim = 0`. The transform still writes `data.node_lambda`
  (`dataset.py:95`), which is collated and ignored. This is harmless, but add a one-line test: one blind training step
  runs.
- In-job generate fields must be empty: `--l_level` makes `generate_confs.py:107-110` exit for a non-λ model (V2 note).
- The pair_ok filter (`dataset.py:145-162`) applies identically to S4's ✓ [E-ROBUST-058], so blind vs S4 isolates the
  λ input.
- The resume check (`ablation_train_array.sbatch:73-84`) works for a new NAME.

**Eval:** the blind model cannot use the `S4*` sets (they carry `--l_level`). Use S1core/S1panel + S1A5 instead. The
seed pickles are shared, so blind on `S1_lam0.50` is paired with S4 on `S4_lam0.50_l0.50`, and likewise for
etkdg/mmff/λ0 and gtLcycle (S4 used `--l_level 1`).
- S4 controls at λ 0.25 / 0.75 were trimmed (S4x lines exist in `r2_evalsets.tsv` but never ran): +6 packed runs
  ≈ 1.0 GPU-h.
- S4 has **no defined λ for A5ring/A5acyc**; compare blind with S3 there.
- "S4 λ picked on validation for MMFF" has no code path (S-5). Use round 2's λ = 0 on MMFF (it exists).

**Gate data exist now:** S4 ×3 S4core and S3 ×2 S1panel are complete on gnode118, so the card's gate (S4 vs S3 at MMFF
and λ 0.5, on S4's subset via `S4_test_subset.txt`, `analyze_round2.py:168-169`) is a CPU computation today.

**Verdict: IMPLEMENTABLE WITH CHANGES** (config only; no `--l_level`; S1 sets; pre-declared λ; optional S4x controls).
**Cost:** 2 × 11.7 + 2 × 9 × 0.17 ≈ **26.5 GPU-h** (+1.0 for S4x controls); 0 CPU-h beyond analysis (reuses
`cache_paired`).

### C-ROBUST-03: sampler-matched mixture (S3-MMFF)
**Touchpoints:** zero TD code.
- `tools/build_paired_pickles.py --std_dir data/QM9/standardized_pickles_mmff --raw_dir data/QM9/qm9/ --out_dir
  data/QM9/standardized_pickles_paired_mmff --n_workers 16` (args `:50-56`). Its pairing check recomputes
  `AlignMol(RemoveHs(matched), RemoveHs(gt))` against the stored `conf['rmsd']` (`:88-94`). The MMFF variant stored
  exactly that value: `standardize_confs.py:108`, after `mmff_func` at `:83-87` and matching at `:89-107`. `geom_id`
  comes from the raw conformer dicts, so the pairing logic transfers unchanged [E-ROBUST-057].
- Add `QM9_PAIRED_MMFF` next to `common.sh:46`, and a `paired_mmff` case in `featurize_qm9.sbatch:29-39` and
  `ablation_train_array.sbatch:53-63` (+2 lines each) → `cache_paired_mmff`.
- Training line: `S3mmff_mixL_p0.5 | paired_mmff | --l_mix_p_gt 0.5 --log_timing --fail_on_nan | --pre_mmff | 2 | s |`.
  The in-job default eval is then MMFF L on the SMILES path, as B3's (`ablations_train_round2.tsv` B3 lines).

**Transfer check**
- TD's matching as domain adaptation [E-ROBUST-021/022]: our MMFF std pickles are matched after relaxation
  (`standardize_confs.py:83-107`), so the "expert relabelling" step exists in code ✓.
- The GT half is `gt_pos_aligned` Kabsch-aligned onto the MMFF conformer (`build_paired_pickles.py:136-146`) ✓.
- The pairing / pair_ok rates for MMFF pairs are UNVERIFIED (card §4). The tool prints `PAIRING_CLEAN` / `PAIR_OK`
  (`:174-186`). Require ≥ 98 % clean, as the round-2 S4 gate item (1) did.
- Silent-failure mode: `ConformerDataset` reuses **any** existing cache file at the given path (`dataset.py:124-129`;
  `featurize_qm9.sbatch:12-13`), and the stale-cache guard only checks `gt_pos` presence (`:141-144`). A wrong
  `--cache` would silently train S3 on RDKit pairs. Use a new cache name and assert, at load, that the mean RDKit-half
  bond error matches the MMFF level.

**Gate data exist now:** B3 s0–s2 in-job `--pre_mmff` (`steps20_seed0`) and CR ×3 `S5ctrl_pre_mmff` are complete, so
the gate is computable on CPU today.

**Verdict: IMPLEMENTABLE AS SPECIFIED** (config level; the path variable, cache name and pairing-rate check are routine).
**Cost:** CPU pairing 1–4 h (card; the round-2 analogue was not timed here) + featurize ~1 h. GPU 2 × 11.7 + ~8
conditions × 2 × 0.17 ≈ **26.1 GPU-h**.

### C-ROBUST-04: CFG-style L guidance / CTRL–B1 score interpolation
- **Guided single-model variant: NOT FEASIBLE THIS ROUND.** A "weak-L" score at the current torsion state would need
  a second conformer carrying weak L *and* the current torsions at every step. TD tracks torsion state only as updates
  relative to its own seed (`total_perturb`, `sampling.py:290-298`), and two different L seeds have different
  reference torsions. There is no code path for transferring torsions between them (card §4 inference confirmed).
- **Score interpolation diagnostic: IMPLEMENTABLE WITH CHANGES** (~40 lines). Add `--model_dir2`, `--score_w` to
  `generate_confs.py`, load the second model as at `:100-105`, and form `edge_pred = w·s1 + (1−w)·s2` at
  `sampling.py:186-192`. CR and B1 share the architecture and featurisation (`QM9_MODEL_ARGS`, `common.sh:48`) ✓; S4
  would need a λ. 2× forward per step ≈ 2 × 0.26 GPU-h per run, so 9 runs ≈ 3–5 GPU-h.

Agrees with the card's NO: it cannot sample the mixture [E-ROBUST-049].

### C-ROBUST-05: S3 ratio sweep / curriculum
- **Ratio sweep: AS SPECIFIED, zero code.** `--l_mix_p_gt` takes any value (`dataset.py:45`, `:98`). Each value needs
  its own NAME, so the resume check (`ablation_train_array.sbatch:73-84`) works. 6 trainings × 11.7 + 6 × 3 × 0.17
  ≈ 73 GPU-h.
- **Curriculum: WITH CHANGES** (~20 lines). The transform has no epoch access. `train.py:73-76` would set
  `train_loader.dataset.transform.l_mix_p_gt = p(epoch)` before each epoch. Loader workers are re-forked each epoch
  (`persistent_workers` is not set, `dataset.py:363-366`), so the change propagates. Make p a pure function of the
  epoch, so `--resume` needs no extra state.
- S3 now has 2 seeds (both complete), so S3's own seed spread, which card §3 calls unknown, is computable from
  existing files.

Agrees with the card's NO (headroom argument; not a code question).

## 4. Budget roll-up (cards the scouts called YES / MAYBE)

| Card | GPU-h (grounded) | GPU-h (card) | CPU-h (grounded) | CPU-h (card) |
|---|---|---|---|---|
| C-EVAL-01 | 0 (0.5 opt.) | 0 | 1–2 | ~2 |
| C-EVAL-02 | ~0.9 | 0 | 2–4 (unmeasured) | 2–4 |
| C-EVAL-03 | ~0.3–0.5 | 0 | ~8 | 22–67 |
| C-EVAL-04 | ~2.6 | 3–4 | ~10–13 | ~31–80 |
| C-EVAL-05 | 0 | 0 | ~2 per arm (~11 for 5) | 8–25 per arm |
| C-GEOM-01 | 1.0–1.9 | 1.5–3 | ~6 | ≤ 1 day |
| C-GEOM-02 | 1.0–1.9 | 1.5–3 | 3–9 | unmeasured |
| C-GEOM-03 | ≤ 1 + 1.0–1.9 (only if D-210 passes) | ≤ 2.5 | ~1 | — |
| C-GEOM-06 | ≤ 1 (gated) | ≤ 1.5 | 2–3 | — |
| C-ROBUST-01 | ~29.4 | 33–40 | ~2 | < 1 |
| C-ROBUST-02 | ~26.5 (+1.0) | 29–37 | 0 | 0 |
| C-ROBUST-03 | ~26.1 | 28–36 | 2–5 | 1–4 |
| **All YES/MAYBE together** | **≈ 88–96** (≈ 82 of it is the three ROBUST trainings) | | **≈ 48–64** | |

Shared infrastructure (S-1 … S-6) is ≈ 600–700 lines before any card-specific code: xtb runner 150, graph filter 60,
gate tool 150, val set 70, eval-set TSVs and paths 20, analysis driver 150+. Most of it is CPU tooling that several
cards share; build it once.

## 5. Cross-read with impact analyst
`round3/impact.md` was read on 2026-10-11 in its full version (summary table, §1–§5). Below: answers to the five
assumptions the analyst asked me to check (impact §5 (a)–(e)), then the disagreements.

### 5.1 Analyst's assumptions checked against code
- **(a) xTB seeds through the S1 seed path with `l_error.csv` rows: HOLDS.** Add the tags to the F5 loop
  (`make_l_seed_pickles.py:107`). The rows then use the same Hungarian heavy-angle assignment as ETKDG/MMFF
  (`:106-114`), so the tail metric is like-for-like across ETKDG / MMFF / xTB. All three share the limitation that only
  L of the 2L seeds get rows.
- **(b) Generated bond lengths and angles equal the seed's exactly: HOLDS.** Torsion updates rotate the bridge-side
  fragment about the bond axis (`utils/torsion.py:9-39`; ring bonds are never rotatable). This preserves every bond
  length and every bond angle; round 2 asserted it on real runs (FIXES X12).
- **(c) S3/S4 on new seed pickles: HOLDS for S3; PARTLY for S4.** S4 requires `--l_level` (`generate_confs.py:106-110`).
  "λ picked on validation" has **no code path**: there is no validation csv / GT pickle / seed pickle (S-5). Until S-5
  is built, S4 must use a pre-declared λ (round 2's λ = 0 for every non-oracle L).
- **(d) Round-1 `gtL` used a random GT conformer: HOLDS, with a caveat that changes the analyst's anchor G (5.2.1).**
  `steps20_seed0_gtL` passes `--seed_confs` without `--seed_confs_cycle` (`ablation_train_array.sbatch:133`), so each
  generated conformer draws `random.choice` over the molecule's own GT conformers (`diffusion/sampling.py:86`), with
  replacement.
- **(e) λ-family seed-reuse defect: HOLDS.** The locator is `make_l_seed_pickles.py:216-231`, not `:156-186` (the
  docstring is `:153-163`). Replaced slots take `pairs[src]` (`:225`), and their F5 row is measured against the
  source's `Y_al` (`:231`). I adopted this in C-EVAL-01's grounding (fit on the 648 molecules with no replaced slot, or
  as a sensitivity row).

### 5.2 Disagreements
1. **Anchor G is not leak-free.** Impact §3c calls the random-GT-conformer L "no conformer-identity leak, unlike
   `gtLcycle`". In code it samples **with replacement from the molecule's own GT conformers** (5.1 (d)). On the 955
   λ-population molecules (from `l_error.csv`, λ 1.00 rows):
   - 23.9 % have a single GT conformer, so for them `gtL` ≡ `gtLcycle`;
   - on average 92 % of a molecule's GT conformers (87.5 % conformer-weighted) have their own L among the 2L random
     draws.

   So anchor G removes only part of the identity leak. B1's 0.037 (random) vs 0.021 (cycled) is that partial removal,
   not its absence. The floors CTRL ≈ 0.086 and B1 ≈ 0.038, and the low-X end of the §3d table, are therefore
   **optimistic ORACLE floors**. That matters most for the rigid and few-conformer molecules, where L = 1 is common. A
   leak-free anchor needs L from a *different* GT conformer. That exists only as a floor estimate
   (`local_structure_analysis.py --mode test`, `floor_gtother`) and is impossible for L = 1 molecules. I suggest the
   analyst state this and widen the low-tail predictions upward. This also affects C-EVAL-01's "B1 reaches 0.073 at
   X ≤ 14 %".
2. **xtb needs no installation.** Impact §0 (C-EVAL-03: "YES if xtb installs"; "shares the C-GEOM-01 install") and
   the priority note ("once xtb is installed") assume an install step. The `u22/xtb/6.7.1` module binary runs on
   gnode118 as is (section 2.1). The CPU cost is ~5–15× below the cards' assumptions (0.14 s per optimisation). The
   C-EVAL-03/04/05 CPU work can start as soon as S-1/S-2 are written. One caveat for the analyst's "reference-like
   acyclic L" question: the probe checked GT conformers (they stay put), not whether ETKDG seeds reach the reference
   basin. That is exactly what C-GEOM-01's gate measures.
3. **Gate design (impact §4.3): agreement, with a code note.** The analyst is right that the D-205 rigid-set seed AMR-R
   rule can pass on bond/angle gains alone. S-3 computes the tail share and pucker coverage directly, so a tail-only or
   coverage-only go rule costs no extra code. I agree that at ≤ 2 GPU-h the GPU step of C-GEOM-01 is barely worth
   gating. B1 and S3 can be evaluated on xTB seeds with no change; S4 only with a pre-declared λ (5.1 (c)).
4. **C-EVAL-02 GPU:** impact §0 says 0 GPU-h. Item (d), the random-GT oracle "next to the cycled one", has no existing
   run for S3, S4 or CR (round-2 in-job evaluations are `GT_EVAL=2`, cycled only). It needs ≈ 0.9 GPU-h. Note also
   point 1: random-GT is itself not a clean ceiling.
5. **C-EVAL-01 feasibility:** impact §1 says the per-molecule regression is "not possible locally". It is possible on
   the cluster at CPU cost. Every run's `eval.pkl` is written by `gen_eval` (`common.sh:130-134`), and `l_error.csv`
   is in `round2_seeds/`. Fit within molecules (my C-EVAL-01 change), not across molecules, because L error and
   molecule difficulty are confounded.
6. **Round-2 data now exist on gnode118** (checked 2026-10-11):
   - S3 s1 and S4 s0–s2 panels (`S1panel`, `S4core`) and B3 s0–s2 in-job `--pre_mmff` evaluations are complete;
   - the C-ROBUST-02 gate (S4 vs S3 at MMFF / λ 0.5) and the C-ROBUST-03 gate (S5 ≤ 0.150) can be computed now on CPU,
     paired, from the `eval.pkl` files.

   The analyst's "likely fails" calls for both can be replaced by measured gates before the panel.
7. **C-GEOM-06 single-factor design:** the analyst predicts "−0.005 to −0.010 vs v1 (worst +0.005: v1→v3 shift)". In
   our RDKit, `srETKDGv3()` differs from `ETKDGv3()` in two switches: small-ring on **and** macrocycle off. The
   prediction should be attached to the corrected contrast `ETKDGv3()` + `useSmallRingTorsions=True` (my C-GEOM-06
   change).
8. **Minor cost differences** (no change of call): C-GEOM-02 coding 2–3 days (impact 1–2); C-ROBUST-01 ≈ 29 GPU-h
   (impact 30–37 + 3); C-ROBUST-02 ≈ 26.5 GPU-h (impact 29–37); C-EVAL-04's TD+cluster arm ≈ 2.6 GPU-h (impact 3–4).

No disagreement on C-GEOM-04, C-GEOM-05, C-EVAL-06, C-ROBUST-04 or C-ROBUST-05: both of us agree with the scouts' NO.
For C-GEOM-03 both of us make it conditional on D-210. I add that our cluster holds no file that can map `QM9.zip`'s
indices to SMILES, so the check is likely inconclusive unless `QM9.zip` itself carries SMILES. Downloading it needs
user permission.

