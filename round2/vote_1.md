# Vote 1 (coding agent 1): review of code plans 2 and 3, and votes on D1-D9

Written 2026-10-06 after reading `code_plan_2.md` and `code_plan_3.md` in full and re-reading my own `code_plan_1.md`.
Line numbers refer to commit 08b5987. No code was changed.

## 1. Bugs and validity problems found

### 1.1 Plan 2

- **(P2-a) The V18 pipeline check is impossible for CTRL_rematch.**
  - Plan 2 §3.3 says `S1_lam1.00` on CTRL_rematch s0 "must reproduce the existing `steps20_seed0_gtLcycle` of the same model".
  - CTRL_rematch was trained with `GT_EVAL=0` (`slurm/ablations_train.tsv:17-19`), so no gtLcycle run exists for it.
  - Fix: add `gtLcycle` runs for CTRL_rematch s0-2 (plan 3 §7 item 1 found this). They are also the F4 GT-end control
    for S3 and S4.
- **(P2-b) The pairing fallback aborts on degenerate keys.**
  - §4 step 1 falls back to the key (`boltzmannweight`, `totalenergy`), which "must be unique … otherwise abort".
  - Near-duplicate GEOM conformers can share energies, so one duplicate would abort the whole build.
  - Fix: tie-break among the candidates with the `conf['rmsd']` recomputation from step 3, which is already the
    decisive check. Abort only if the tie-break still leaves more than one candidate with *different* coordinates.
- **(P2-c) The `mmff_error` count misses the tail of each worker.**
  - §6 S5 reads "the last cumulative `long_term_log` print". That dict is printed only every 20 molecules
    (`torsional-diffusion/standardize_confs.py:156-162`), so up to 19 molecules per worker are missed.
  - `log_error` prints the bare key at every failure (`:47-50`, `:87`), so `grep -hx mmff_error` is exact.
  - Plan 3 §S5 has the same problem.
- **(P2-d) The provenance guard changes a default path.**
  - It is added to `gen_eval` (`slurm/common.sh:96-117`).
  - On a round-1 directory that is touched again, the hash is written from the *current* inputs, so it certifies
    nothing about the original run.
  - Acceptable as a forward-looking guard, but it should say "provenance from round 2 onward". Round-1 tables must not
    claim provenance.
- **(P2-e) The S4 endpoint snap makes the realised λ distribution depend on the molecule.** Unsafe pairs snap to
  λ ∈ {0,1}, which correlates λ with molecule type. Fine if `pair_ok` is ≥ 98% (its own gate). Report the snap fraction
  per ring-size stratum.
- **(P2-f) Footprint.** 9 new tool scripts plus 2 new sbatch files. Several can be flags on existing tools:
  - `floor_summary` and `leak_test` → modes of `local_structure_analysis.py`;
  - `l_error_report` → part of the seed builder.

  This is not a correctness bug.

### 1.2 Plan 3

- **(P3-a) The builder can crash on MMFF.**
  - The §S1 `job()` calls `AllChem.MMFFOptimizeMoleculeConfs` with no `try`.
  - `try_mmff` (`torsional-diffusion/diffusion/sampling.py:21-26`) exists precisely because this raises for molecules
    without MMFF parameters. Inside a `Pool`, one such molecule kills the builder.
  - Fix: wrap it, and drop the molecule from **both** etkdg and mmff pickles (plan 2's twin rule).
- **(P3-b) The interpolation uses all-atom Kabsch and no terminal relabelling** (`interpolate_L`, §2.1).
  - Plan 2's local check found C–H shrinking to about 0.62 Å at λ = 0.5 for a methyl rotated by 120°.
  - All-atom fitting also lets the hydrogens pull the rotation.
  - Both S1.4 and S4 inherit this. My plan 1 has the same defect (§1.3).
- **(P3-c) The S1 interp seeds keep the DE optimum, not the stored conformer.**
  - The S1 interp seeds use `m = optimize_rotatable_bonds(...)` "KEEP the optimum".
  - The training data (rematch, used by S4) stores the *last DE evaluation* (`torsional-diffusion/standardize_confs.py:106`
    discards the return value; plan 3 §2.1).
  - So the test-time λ = 0 L does not come from the training-time λ = 0 distribution. The bias is small, but it is
    exactly the train/test L shift the arm is about.
  - Use the same recipe as standardization: discard the return value, as `:106` does.
- **(P3-d) Default-path changes in training.**
  - **Resume.** `--resume` and the replacement of the move-aside logic at `slurm/ablation_train_array.sbatch:61-65`
    change what happens to every existing TSV line. A resumed run is not reproducible, because the DataLoader worker
    RNG is not restored.
  - **G-learn.** The exit-4 rule "train loss ≥ base loss after epoch 0" is an UNMEASURED heuristic that can kill a
    valid run. B6, with jitter, is the likely victim.
  - **`--min_its`.** It is a placeholder value baked into production TSV lines.
  - Keep `data_wait` and the non-finite-loss check; both are passive. Make resume opt-in with the old default kept.
    Drop G-learn and `--min_its` unless M0 calibrates them.
- **(P3-e) Changing loader workers from 8 to 3 changes the random streams relative to the round-1 controls.**
  - It is not a bias, but it is a recipe difference from CTRL and B1. It is acceptable only if stated.
- **(P3-f) The budget rests on unmeasured packing.**
  - "≈146 slot-h" assumes 3 packed evaluations per GPU at about 0.33 h each, which is UNMEASURED.
  - If packing gives less, plan 3 is about 175 GPU-h, the same as plans 1 and 2.

### 1.3 My own plan 1 (stated for fairness)

- **(P1-a) CTRL_rematch coverage is missing.**
  - It is used only on the interp series, yet the S3/S4 panels (etkdgG, mmffG, noise) are compared against it (F4).
  - It also has no gtLcycle runs (`slurm/ablations_train.tsv:17-19`).
- **(P1-b) All-atom Kabsch with no terminal relabelling** (same as P3-b).

---

## 2. Votes on contested decisions

| # | decision | vote | one-line reason |
|---|---|---|---|
| D1 | F1 alignment | **Heavy-atom Kabsch fit applied to all atoms, + plan 2's terminal relabelling (`permute_terminal`), + `interp_safety`/`pair_ok`** | The heavy fit reproduces `conf['rmsd']` (`standardize_confs.py:108`), which is the pairing check. Relabelling removes a measured C–H collapse at no cost to model invariance. I withdraw my all-atom choice |
| D2 | S1.5 acyclic set includes H | **Yes (all-atom)** | All 3 plans agree; only then are A5-ring and A5-acyc exact complements |
| D3 | S1 control-model count | **Full 9 models × all conditions (~105) if packing passes; otherwise plan 2's 95** | F4 needs CTRL_rematch on every condition that S3/S4 are read on. My 72-run design lacked it (P1-a). Add CTRL_rematch gtLcycle ×3 in either case |
| D4 | inference packing | **Yes, gated**: measure 1 vs 3 per GPU on one model's eval set (smoke/M0); keep 1 output directory per condition | Inference is CPU-bound (RDKit + `modify_conformer`). Packing does not change any per-run RNG path; only the wall time is at stake |
| D5 | training CPUs/workers and 2-per-GPU sharing | **4-6 CPUs / 3-5 workers only after G-loader (`data_wait` < 5%); no co-location by default** (only if G-coloc shows ≥ 1.6× aggregate) | Round 1 lost about 10 GPU-h to CPU contention (`RESULTS_QM9.md:95`), so cutting requests is right. Co-location is unmeasured and adds OOM/slowdown risk to 10-h runs |
| D6 | B1cap control | **Yes, 1 seed**, unless plan 3's CPU check shows the 30-conformer cap removes < 2% of conformers *and* the `rdkit_no_embed` molecule-set difference is < 1% | It is the only clean read of the S3/S4 λ = 1 end against B1 (verify_A R2A-2(c)), and it costs zero code |
| D7 | S4 this round | **Yes, wave 2 with S5 as the automatic fallback (plan 3).** Gate at about T+12 h: pairing ≥ 99%, `pair_ok` ≥ 98%, golden strict-load test of old checkpoints, λ parity/guard tests, achieved L error monotone in λ with `worst_bond_dev` ≤ 0.05 Å, and a 1-epoch smoke | The paired data and alignment are needed by S1.4 and S3 anyway. The S4-only code is about 15-25 lines behind `lambda_embed_dim=0` |
| D8 | budget target and trims | **Target ≤ 175 GPU-h.** Training plus in-job evaluations ≈ 137 (+11.5 for B1cap); inference ≤ 25 slot-h with packing | Trim order if packing fails: (1) the S4 descriptive MMFF λ ∈ {0.25, 0.5}; (2) CTRL_base on interp 0.25/0.75; (3) the S2/S3 panels down to {etkdg, mmff, noise0.04, interp0.5, gtLc}; (4) the S6 B1 lines; (5) the B6 σ0.02 panel down to 2 conditions. Never trim CTRL_rematch on conditions used by S3/S4 |
| D9 | base plan and grafts | **Base: plan 2.** Graft from plan 1: (i) golden byte-identity tests against a HEAD snapshot (`state_dict`, transform outputs **and post-call RNG state**, featurized datapoints, `generate_confs` coordinates, `make_seed_pickles`/`paired_compare`/`local_structure_analysis` outputs); (ii) a smoke test that runs through the production `ablation_train_array.sbatch` path (its missing-cache check at `:57` prevents writing a truncated cache), plus plan 2's `/smoke/` cache guard; (iii) prefer flags on existing tools over new scripts where plan 2 adds one-off tools; (iv) the `MODEL` eval expansion and the optional 7th TSV field. Graft from plan 3: (i) eval-set packing and M0 measurements; (ii) the `data_wait` timer and non-finite-loss check; (iii) CTRL_rematch gtLcycle runs; (iv) the CPU core budget; (v) S5-in-wave-2 fallback; (vi) the cap-loss quantification. **Reject** plan 3's G-learn exit, mandatory resume and `--min_its` placeholders | Plan 2 is the only plan whose geometry layer (heavy fit, relabelling, stereo/clash checks, `pair_ok`) and validity list (V1-V21) address the failure modes of interpolated L. Plans 1 and 3 contribute the regression discipline and the throughput |

---

## 3. Ranked preference

1. **Plan 2.** It has the most complete validity design. It found the terminal-H collapse, which plans 1 and 3 both
   miss. Its defects (P2-a to P2-f) are small and local.
2. **Plan 3.** It has the best throughput model, and it found the missing CTRL_rematch GT-L runs. Its issues are a
   builder crash (P3-a), the interpolation flaw (P3-b) and optional default-path changes (P3-d), all fixable by
   rejecting or patching specific items.
3. **Plan 1 (mine).** It is the leanest diff and has the strongest default-identity tests. But it has an experimental
   control gap (CTRL_rematch on the S3/S4 conditions, and no CTRL_rematch gtLcycle) and the same interpolation flaw.
   I rank it last because the control gap is a design error rather than an optional feature.
