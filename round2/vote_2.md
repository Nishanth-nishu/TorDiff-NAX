# Round 2 vote, coding agent 2

Written 2026-10-06. I read code_plan_1.md and code_plan_3.md in full. My own plan is code_plan_2.md. Paths are relative to
`C:\Users\HP\Desktop\tor_diff`, and `TD/` = `torsional-diffusion/` @ 08b5987.

## 1. Bugs and validity problems found

### Plan 1
1. **Methyl and other terminal groups collapse under interpolation; this applies to plan 3 too.**
   - The plan: all-atom Kabsch (`code_plan_1.md` §1 `kabsch_align`), then `interp` with the identity atom map. There is
     no relabelling of graph-equivalent terminal atoms.
   - Why it matters: DE matching uses popsize 15, maxiter 15 and an H-inclusive objective (`TD/standardize_confs.py:15-16,106-107`).
     That can leave terminal rotors about 120° off. `notes/code_concerns.md` A.7 reports H-RMSD of 1.03 Å on 2 of 5 seeds.
   - Local check (scratchpad, RDKit 2026.03.6): a methyl rotated by 120° and interpolated at λ = 0.5 gives C–H lengths of
     0.62–0.67 Å. With Hungarian relabelling per parent atom they stay at 1.11 Å.
   - Plan 1's claim "the difference is small" (§2.2) contradicts that note. F5 reporting would only *detect* the problem
     after the S4 training data was already corrupted.
2. **No stereo or clash guard on interpolated pairs.**
   - An enantiomeric or diastereomeric pair interpolates through a planar centre. Nothing in `pair_std_with_raw` or
     `_select_paired_L` excludes such pairs.
   - The rate is small: GT-graph ETKDG matched GT stereo in 99.93% of cases (`cluster_sync/results/analysis/local_structure_test.log`).
   - It is still non-zero and unmonitored.
3. **The 72-run S1 grid breaks F4 for S3/S4.**
   - CTRL_rematch is run only on the interp series (§3.2). Yet the S3/S4 panels use etkdgG, mmffG and noise (§5, §6.4),
     and §3.4 compares "arms = B1, B6, S3, S4" against "CTRL (base or rematch)".
   - On the non-interp conditions, S3/S4 would therefore be compared with CTRL_base. That is not their single-factor
     control (SHORTLIST F4).
4. **The interp series built from rematch pickles is not on the gtLcycle population** (§3.1, "Interp population").
   - It is capped at 30 conformers (`TD/standardize_confs.py:59-67`), drops DE-failed conformers (`:111-114`), and uses
     raw-pickle order.
   - So for those molecules: λ = 1 ≠ gtLcycle; some GT conformers have no own-L seed; and the `i % len` source mapping no
     longer indexes `clean_confs` order.
   - This weakens F3 anchoring and breaks any per-source F5 or leak analysis on these pickles. Plan 1 acknowledges the
     coverage difference but not the source-index mismatch.
   - Building the test-time matched seed directly from the test GT list, as plan 3 §S1 and plan 2 §3.1 do, avoids it.
5. **No guard against reusing stale results.** `gen_eval` skips whenever `confs.pkl` exists (`slurm/common.sh:102`), and
   plan 1 relies on unique names only. A re-built pickle under the same name, or a retrained model, would silently reuse
   old outputs.
6. **The noise draws are independent across σ** (§3.1 table). Common random numbers would make the dose-response
   smoother at no cost. This is minor.

### Plan 3
1. **The same terminal-collapse problem** (`interpolate_L`, §2.1: all-atom Kabsch, identity map, no relabelling). It
   also has no stereo or clash guard.
2. **The test-time matched seed is optimised differently from the training data.**
   - §S1 builder: "KEEP the optimum returned by optimize_rotatable_bonds". The training pickles keep the last DE
     evaluation instead (`TD/standardize_confs.py:106` discards the return value; plan 3 §2.1 found this itself).
   - So the test-time λ = 0 and λ_in = λ_build conditions for S4 come from a slightly different L/τ distribution than
     S4 and CTRL_rematch saw in training. For consistency, use the training recipe exactly.
3. **G-learn may kill valid runs.** "train_loss ≥ base_train_loss after epoch 0 → exit 4" (§2.4) is an unverified
   heuristic, and the plan admits this. Jittered or mixed-L arms have a different base-loss scale.
   - `--min_its 6` is a placeholder. Under the 4-CPU or co-location regime it could abort healthy runs.
   - Both must be calibrated on M0 logs, or demoted to warnings.
4. **The resume logic changes the meaning of completion** (`slurm/ablation_train_array.sbatch:61-65` replaced).
   - Resuming after a code change could mix code versions within one run. Its compatibility check compares only selected
     yaml keys.
   - Add a sha of the patch or commit to the yaml, and refuse to resume on a mismatch.
5. **The training regime changes in the middle of the comparison.**
   - Fewer loader workers (4 CPUs, 3 workers) changes the per-worker RNG streams relative to the round-1 controls.
   - This is scientifically harmless, because training seeds are independent replicates anyway. Still, the "bit-identical
     CTRL/B1" claim (T2) holds only for the transform, not for whole runs. Say so.
6. **Packing (3 `gen_eval` processes per GPU) is sound in principle**, because each process is seeded independently
   (`TD/generate_confs.py:93-94`). It is unvalidated, though: one packed re-run of an existing round-1 tag must reproduce
   its SUMMARY. Packing also needs the stale-output guard (plan 2 §5.5), because concurrent and re-submitted
   `gen_eval` calls share the skip-if-exists logic.
7. **Noise uses an independent RNG per σ** (`default_rng([seed, mol_idx, int(s*1e4)])`), so there are no common random
   numbers. This is minor.

**Credit to plan 3.** Its §7.1 finding is real: CTRL_rematch has no GT-L runs (`slurm/ablations_train.tsv:17-19`,
GT_EVAL 0). My plan covered this only for s0 (λ = 1.00). The `S1CR r2_gtLc_ORACLE` lines should be adopted.

## 2. Votes on contested decisions

| # | decision | vote | reason |
|---|---|---|---|
| D1 | F1 alignment | **Heavy-atom Kabsch fit applied to all atoms, plus relabelling of terminal equivalent atoms (2 passes), plus a `pair_ok` guard (stereo, max bond deviation ≤ 0.05 Å, non-bonded distance ≥ 0.9 Å) with endpoint snapping in S4** | The heavy fit reproduces the stored `conf['rmsd']` exactly, so it doubles as the pairing check. Relabelling removes a measured 0.6 Å C–H collapse that all-atom Kabsch cannot fix. |
| D2 | S1.5 acyclic set | **Includes H (all-atom)** | All three plans converge on this. Without H, A5-ring carries GT H geometry and A5-acyc carries RDKit H geometry, so the arms are not complements. |
| D3 | S1 model count | **About 90 runs:** B1 ×3 and CTRL_base ×3 on all 11 conditions, plus CTRL_rematch ×3 on every condition where S3/S4/S5 are evaluated (etkdg, mmff, noise, λ series), plus CTRL_rematch gtLcycle (plan 3 S1CR). A5 gets no CTRL_rematch. | F4 requires CTRL_rematch wherever S3/S4 are read. 72 runs breaks this, and 108 wastes runs on A5. |
| D4 | Inference packing | **Yes: pack 3 per GPU, grouped per model** | It does not change any per-run RNG path. Adopt it only with (a) one packed reproduction of an existing round-1 run and (b) the provenance guard against stale outputs. |
| D5 | Training CPUs / workers, 2 per GPU | **4 CPUs and 3 workers only after G-loader shows data_wait < 5%. Co-location off by default**, enabled only if G-coloc passes *and* the budget needs it | Throughput changes do not affect validity, but untested co-location risks OOM and slowdowns. Keep the round-1 worker count if G-loader is ambiguous. |
| D6 | B1cap control | **Yes, 1 seed** (`--mix_gt_p 1.0` on the paired cache) | It is the only clean single-factor reading of the λ = 1 end of S3/S4. It covers both the 30-conformer cap and the `rdkit_no_embed` drop, which plan 3's cap-fraction check does not. |
| D7 | S4 this round | **Yes, gated. The gate:** (1) CPU tests pass, including the strict-load regression and the λ guards; (2) pairing ≥ 99%, with \|Kabsch heavy RMSD − `conf['rmsd']`\| < 1e-3; (3) `pair_ok` ≥ 98% with relabelling in place; (4) the λ = 1.00 pickle reproduces gtLcycle within 0.003 Å on CTRL_rematch s0; (5) the S4 smoke trains and generates. If the gate passes by about T+12 h, wave 2 is fine; otherwise use the last wave or round 3. | The incremental code is small and the risk sits in the shared paired data, which this gate validates. |
| D8 | Budget and trims | **Target about 165 GPU-h**, packed inference included (≈ 13 training runs × 11 h + about 25 GPU-slot-h inference). Trim in this order: S4 mmff λ 0.25/0.5 descriptive runs; the S6 B1 lines; B6 σ 0.02 panel; CTRL_base on the λ 0.25/0.75 conditions; noise 0.01 | Keeps every pre-registered contrast and the F4/F5 controls, and drops only descriptive cells. |
| D9 | Base plan | **Plan 1 as the base.** It has the cleanest diff: default paths verbatim, golden tests M1/M2/M8/M9 and C5–C7, featurize-time pairing that reuses the exact rematch L, and the 7th TSV field. | The grafts are listed below. |

### D9 grafts
- **From plan 2:**
  - `lgeom.align_pair` (heavy fit + `permute_terminal` + `pair_ok`, stereo via `tools/local_structure_analysis.py:217-222`) and the S4 endpoint snap;
  - the `gen_eval` provenance guard (args + sha256 of checkpoint and seed pickles);
  - the V18 λ = 1.00 pipeline reproduction check;
  - common-random-number noise;
  - the S0 data-validity scan: GEOM conformer graph identity across conformers, which B1 assumes (`TD/utils/dataset.py:224,234,239`), and train/test SMILES overlap;
  - building test-time interp/A5 seeds from the test GT list in `clean_confs` order with the *training* matching recipe, not from rematch pickles;
  - the smoke cache-path guard.
- **From plan 3:**
  - packed eval sets (`run_evalset`);
  - the S1CR CTRL_rematch gtLcycle and pre_mmff lines;
  - the per-epoch `data_wait` timer;
  - NaN-loss fail-fast;
  - atomic `last_model.pt` writes and resume, with a commit-sha check;
  - `tools/lsa_lib.py`, which lets new tools import the floor helpers without argv parsing;
  - `--mode seeds` model-free floors;
  - G-loader.
  - Keep G-learn and `--min_its` as warnings until calibrated.

## 3. Ranked preference
1. **Plan 1**, as the base with the grafts above. It is the safest diff and has the best regression tests.
2. **Plan 2 (mine).** It has the most complete validity layer: relabelling, `pair_ok`, provenance, the pipeline check.
   Its engineering is less minimal (a separate paired-pickle builder), and it missed CTRL_rematch GT-L for s1/s2.
3. **Plan 3.** It has the best throughput engineering. It carries the same alignment defect, unvalidated fail-fast
   heuristics, and more invasive training-loop changes (resume, worker regime) for the base.
