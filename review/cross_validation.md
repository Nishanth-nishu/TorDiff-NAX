# Cross-validation of the QM9 TD ablation study (experimental design and math)

Scope: experimental design, statistics, the floor mathematics, the evidence chain for the FlexiTors gaps (as corrected in `papers/gap_synthesis.md` §0), and SLURM correctness. Metric-code correctness is covered separately in `review/metric_verification.md`; this review uses its fixes (symmetry-aware `heavy_automorphisms` in `tools/local_structure_analysis.py`) and does not repeat them.

## How the votes were taken

For every contested point, three reviewers argued independently from the sources. The majority was adopted.
- **R1, methodologist:** statistics, bounds, confounds.
- **R2, TD literalist:** argues only from the TD paper and the upstream code.
- **R3, FlexiTors advocate:** argues for the reading most favourable to the FlexiTors motivation.

Votes are written as `R1/R2/R3 → outcome`. Uncontested findings are marked 3-0.

**Severity:**
- **blocker:** invalidates a headline claim as the plan stood.
- **major:** biases or confounds an arm, or can silently corrupt results.
- **minor:** hygiene, or clarity of interpretation.

---

## Summary of the top issues

| ID | Severity | Issue | Status |
|---|---|---|---|
| M1 | **blocker** (for the G1-on-QM9 claim) | The "TD sits on the RDKit floor (0.178 vs 0.17 Å)" comparison is invalid. The 0.17 Å figure and our `floor_rmsd` are one-to-one conformer-matching numbers, biased *upwards* against AMR-R. | Fixed: new estimators `floor_best_sym` and `floor_run` (run-matched); plan reworded |
| C4 | major | A1 (`random.choice` of GT L) is an *independent-L* oracle, not an oracle for L matched to τ. The plan had no arm that isolates τ↔L coupling (G3). | Fixed: `--seed_confs_cycle` (commit c8d2dc4) and arms A1c / A1c-sanity |
| E1 | major | No arm tested the *specific* FlexiTors claim that low-dimensional **acyclic-angle** relaxation removes the floor, as opposed to stereo, rings or bond lengths. gap_synthesis §5 row A1(b) was never implemented. | Fixed: J-decomp columns (`angle_oracle_rmsd`, `local_oracle_rmsd`, `floor_gtother`, ring/stereo strata) |
| C3 | major | B2/B3/B4 use locally regenerated pickles but are compared with CTRL trained on the *shipped* pickles (unknown RDKit version, unseeded). The variant is confounded with RDKit version and matching noise. | Fixed: `VARIANT=rematch` plus `CTRL_rematch_100ep` |
| X1 | major | Training scripts skip training when `best_model.pt` exists. That file is written at every val improvement, so a time-limit kill leaves a half-trained model that later jobs silently evaluate. | Fixed: `train_complete` (`.train_done` or last epoch) in all training and inference scripts |
| S1 | major | The plan claimed 3 sampling seeds for every inference arm, but the TSV had them only for R0-R2. There was no bootstrap tool at all. | Fixed: seed-1/2 lines for the ★★★ arms, `tools/paired_compare.py`, pre-registered endpoints with Holm correction |
| C2 | major | G1 (`--seed_mols`) embeds from the GEOM mol. If that mol carries no stereo tags, ETKDG picks stereo at random, so G1 would not isolate stereo error. | Fixed: `make_seed_pickles.py` assigns stereo from GT conformer 0's 3D |
| M2 | major (interpretive) | H-inclusive DE in conformer matching biases the stored `conf['rmsd']` floor **upwards**, and it gives noisy heavy-torsion training targets. | Fixed: `--heavy_objective` (commit 49a4811) and arm B5 |

---

## 1. Does each arm isolate one variable?

| ID | Sev. | Issue | Evidence | Vote | Fix |
|---|---|---|---|---|---|
| C1 | minor | A1/A1c (GT-L seeds) never fail to embed, while R0 fails on the molecules ETKDG cannot embed. Those count as COV = 0 and are excluded from AMR, so the arms are compared on different populations. | `diffusion/sampling.py:embed_seeds` returns `[]` on an embedding failure; `evaluate_confs.py` counts failures as COV 0 and drops them from AMR (`nanmean`) | 3-0 | `paired_compare.py` reports COV over the union of molecules (failure = 0) and AMR over molecules finite in *both* arms, each with its own n |
| C2 | major | G1 stereo source (above). Also, the floor tool embeds from the GT graph while TD at test time embeds from the SMILES. A wrong-stereo TD seed is therefore part of TD's error but not of the `--mode test` floor. | `make_seed_pickles.py` (old line 66, `copy.deepcopy(confs[0])`); `sampling.py:get_seed` (SMILES); `local_structure_analysis.py job` (GT graph) | R1 fix / R2 fix / R3 fix → 3-0 | Stereo is now assigned from 3D (verified locally: 5/5 re-embeddings reproduce `C/C=C/[C@H](C)O`). The run-matched floor (`--mode run`) uses the run's own seeds and stereo, so it has no mismatch. **Rebuild `data/QM9/test_gt_seed_mols.pkl` if it already exists on the cluster.** |
| C3 | major | The B-arm control is confounded (above). | `standardize_confs.py:78` unseeded `EmbedMultipleConfs`; the shipped pickles' RDKit version is unknown (code_concerns §C, last bullet) | R1 yes / R2 yes (TD's pickles came from a different code state, code_concerns #25) / R3 no (10 GPU-h for a small nuisance) → 2-1 **add control** | `standardize_qm9.sbatch VARIANT=rematch`, TRN `CTRL_rematch_100ep` (line 1). The plan now says B2/B3/B5 are compared with CTRL_rematch. CTRL_rematch − CTRL_base measures the nuisance itself |
| C4 | major | The A1 oracle type (above). | `sampling.py:66` `random.choice(seed_confs[smi])`. A GT conformer whose own L is never drawn (probability (1−1/L)^{2L} ≈ 13%) must be matched with another conformer's L | R1 keep both / R2 keep random (upstream behaviour) / R3 replace with cycle → 2-1 **keep A1, add A1c** | `--seed_confs_cycle` gives conformer i the L of GT conformer i mod L. INF adds `A1c_gtL_cycle_model` (seeds 0-2) and `A1c_gtL_cycle_sanity`; TRN GT_EVAL also runs `steps20_seed0_gtLcycle`. **A1c − A1 = the model-based value of coupling** (compared automatically in `paired_A1c_vs_A1_*.md`). Caveat: the model was trained on RDKit L, so GT L is off-distribution. B1 evaluated on GT L is the in-distribution counterpart |
| C5 | minor | E-inf changes σ_min at a fixed 20 steps, which also changes the log step size. So the arm confounds endpoint with step size. | `sampling.py:105` log-linear schedule over [σ_max, σ_min] | 3-0 | Not changed. Read E-inf together with C_steps50; if the effect is ambiguous, add an E-inf line at 40 steps |
| C6 | minor | B3 changes train L and test L together (MMFF). A2 gives the "test-only" cell; the "train-only" cell (B3 without `--pre_mmff`) is missing. | `ablations_train.tsv` B3 line | 3-0 | Optional: add a TRN line with an empty GEN_ARGS for B3 |
| C7 | pass | Yaml-overrides-CLI trap. The only keys shared by the train yaml and the generate CLI are `likelihood`, `seed` and `batch_size`, and all three are restored. There is no `confs_per_mol` collision (it is not a train argument), so K = 2L holds. | `utils/parsing.py` (train args) vs `generate_confs.py:14-50`; restore loop at `generate_confs.py:90-91` | 3-0 | — |
| C8 | pass | RDKit seeding: `randomSeed = seed + 1` avoids the degenerate seed 0. The same ETKDG seed per molecule across arms pairs the arms on RDKit L. | `generate_confs.py:62-66` | 3-0 | The plan now states the pairing explicitly |
| C9 | minor | E/Z: non-ring double bonds are rotatable in TD (footnote 6, p. 4; F.3). The floor tool includes them in its torsion set, so it "repairs" E/Z the same way TD can. | `local_structure_analysis.py relevant_torsions` has no bond-order check, consistent with `utils/torsion.py` | R1 include / R2 include (TD definition) / R3 exclude, so as not to credit TD with E/Z repair → 2-1 **include** | Report the floor split by `stereo_match_best` and by `n_stereo_double` (`stereo_check.py`). **FlexiTors design note:** a stereo-locked torsion factor for double bonds is a free improvement independent of angles, and must not be counted as angle-factor evidence |
| C10 | minor | J's Spearman ρ(floor, AMR) is confounded by molecule size, flexibility and n_true, since both quantities grow with them. | `breakdown.py` J section | 3-0 | Comment added in `breakdown.py`. The causal readout is now J-run, not ρ |

## 2. Statistical validity

| ID | Sev. | Issue | Evidence | Vote | Fix |
|---|---|---|---|---|---|
| S1 | major | There were no replicates for the ★★★ inference arms and no CI tool. | Old `ablations_inference.tsv` had seeds only for R0-R2; `grep bootstrap tools/` found nothing | 3-0 | 10 lines added (A1, A1c, A2, A3, G1 at seeds 1 and 2); array is now `0-41`. New `tools/paired_compare.py`: replicates are averaged per molecule, then a paired bootstrap over molecules gives the CI, p, Holm-adjusted p and the reference replicate SD (tested on synthetic data). `analysis_qm9.sbatch` runs it for all arms vs R0 |
| S2 | major | Training arms have one seed, and the 100-epoch control has one seed, so the training-seed variance is unknown. | `ablation_train_array.sbatch` uses a global `SEED` | R1 ≥3 CTRL seeds / R2 TD reports single runs (Table 8) / R3 ≥3 for small effects → 2-1 **3 CTRL seeds** | Budgeted plan (user decision, 4 days × 4 GPUs): 3 training seeds each for CTRL_base, CTRL_rematch, B1, B2 and B5. The seed is now a TSV field. The other training arms are deferred |
| S3 | major | Multiple comparisons: about 40 arms × 8 metrics × 7 thresholds. | Plan §1 | 3-0 | Pre-registered primary endpoints: **AMR-R mean and COV-R@0.5 Å** (user decision), Holm-corrected across arms. Everything else is exploratory |
| S4 | contested | Headroom and choice of δ. At 0.5 Å QM9 is saturated (S23D, `related/2025_gurev_s23d.md` l. 26; FM-refiner Table 2). At 0.05 Å, most arms may sit near 0 (the GT-GT local variability alone is about 1° of angle RMS in the synthetic test). | TD Table 7 medians are all 100 | R1 0.25 / R2 0.5 (paper protocol) / R3 0.1 → no majority; the median (0.25) was proposed. **Overridden by user decision: 0.5 Å** | Primary endpoints are AMR-R mean and COV-R@0.5 (`paired_compare.py` and `analysis_qm9.sbatch` defaults). The 0.05-0.25 Å sweep is secondary / exploratory. Residual risk: COV-R@0.5 has little power on saturated QM9, so AMR-R carries most of the primary evidence |
| S5 | minor | The reproduction target "within ±0.005 Å of 0.178" (gap_synthesis §5 A0) is unrealistic without a noise model. | Nothing yet measures seed noise | 3-0 | Judge reproduction against the R0/R1/R2 spread (`ref_replicate_sd` column) |

## 3. Mathematics of the "RDKit local-structure floor"

Notation:
- **L̂_k** is the RDKit local structure of seed or generated conformer k.
- **F(k, l) = min_τ RMSD((L̂_k, τ), C*_l)** is the best RMSD to GT conformer l over torsions τ.
- **AMR-R contribution of GT conformer l** in a run with conformers C_1..C_K is **a_l = min_k RMSD(C_k, C*_l)**.

A torsion update never changes L, so every C_k equals (L̂_k, τ_k), and therefore **a_l ≥ min_k F(k, l)**. This is the only rigorous lower bound, and it holds only (i) on the **same** L̂_k as the run, (ii) with the **min over all K = 2L** conformers, (iii) at the **global** optimum over τ, and (iv) with the **same symmetry-aware RMSD** as the evaluator.

| ID | Sev. | Finding | Vote |
|---|---|---|---|
| M1 | **blocker** | TD's 0.17 Å (App. H: "approximately calculated by conformer matching", fulltext l. 1602-1603), TD's 0.324 Å (F.1: "expected RMSDmin of an optimal *assignment*", l. 1285), the stored `conf['rmsd']` and the old `floor_rmsd` all violate (ii). Each GT conformer gets **one** seed out of L under a one-to-one Hungarian assignment, instead of the best of 2L with no constraint. They violate (i) too: different ETKDG draws from the run. Every violation raises the number, so these figures are **not** lower bounds on AMR-R, and "0.178 ≈ 0.17" does not show that TD is floor-limited. Synthetic check (4 molecules, `scratchpad/t`): `floor_rmsd` 0.58 vs `floor_best` 0.09 Å for the same GT conformer. | R1 invalid / R2 invalid: TD itself says "approximate" / R3 valid as a rough indicator → 2-1 **invalid as evidence**. Severity: R1 blocker / R2 blocker / R3 major → blocker |
| M1-fix | — | `--mode test` now draws K = 2L seeds. `floor_rmsd` is kept as the training analogue. **`floor_best_sym`** is the min over the top-3 of the 2L seeds by transplant cost, with a symmetry-aware re-score; this is the distributional estimate. New **`--mode run`** gives `floor_run(l) = min(a_l, min over the top-3 k of DE(k, l))` on the run's own conformers, satisfying (i), (ii) up to top-m and (iv); (iii) holds up to DE optimality. `floor_run ≤ a_l` holds by construction. `breakdown.py` now uses `floor_best_sym` and labels the ceiling ESTIMATED | 3-0 |
| M1-dir | — | Residual bias of `floor_run`: DE is heuristic, and only the top-m candidates are optimised, so `floor_run ≥` the true floor. The measured `torsion_headroom = a_l − floor_run` is therefore a **lower bound** on the true torsional headroom. The bias favours "TD is floor-limited", which is the FlexiTors-friendly direction. Mitigations: maxiter 50 with polish, and the run's own torsions and GT torsions as extra candidates. With ≤ 4-5 heavy torsions in QM9, DE is near-global. **Do not read a small headroom as proof without the DE-convergence spot check:** rerun 50 molecules with `--maxiter 200 --floor_topm 10` | 3-0 |
| M2 | major | H-inclusive DE (`utils/standardization.py:57-60`, `AlignMol` with all atoms) optimises a different objective from the evaluator's heavy-atom metric. The resulting heavy RMSD is therefore ≥ the heavy-optimal value, so `conf['rmsd']` is biased **upwards** as a floor (it overstates the floor). The same noise enters TD's heavy-torsion *training targets*, which could inflate TD's own error above its floor, so this is a TD-fixable error, not an L error. Local toy (`CCOC(=O)CCN`, 15 iterations): H objective 0.138 Å vs heavy objective 0.100 Å heavy RMSD | 3-0 |
| M2-fix | — | `standardize_confs.py --heavy_objective` (commit 49a4811), `VARIANT=heavy`, TRN `B5_match_heavy`. The floor tool already uses a heavy (automorphism-aware) objective | 3-0 |
| M3 | — | Upper bounds. `transplant_rmsd_*` (GT torsions copied, no optimisation) is achievable, so it is an upper bound on F(k, l); `floor ≤ transplant` holds by construction (GT torsions are a DE candidate). `angle_oracle_rmsd` / `local_oracle_rmsd` set acyclic angles (and bonds) exactly to GT. That is **more** freedom than FlexiTors' k modes within ±Δ, so they are an *optimistic* (lower) estimate of what the B^k factor can reach: if even they leave most of the floor, B^k is unmotivated. Sequential `SetAngle` at branching centres cannot meet all 2n−3 constraints, which pushes back up slightly | 3-0 |
| M4 | — | Is `floor_run` a valid lower bound on TD error? Yes, for the *torsion-only error given that run's L*, modulo M1-dir. It says nothing about a model that changes L, which is exactly the point | 3-0 |

## 4. Evidence chain for the gap claims (as corrected in gap_synthesis §0)

| Claim | What supports it now | What would falsify it | Arm that can show it |
|---|---|---|---|
| **G1: RDKit L imposes a floor that matters on QM9** | TD App. H (fulltext l. 1598-1605); OMEGA median AMR-R 0.126 vs TD 0.147 (TD Table 7). **Weakened by M1:** the "0.178 ≈ 0.17" comparison is not valid | `torsion_headroom` is a large share of AMR-R; A1 − R0 is within the R0-R2 spread; C/D/F move AMR as much as A1 does | **J-run** (new), A1, C/D/F |
| **G1-specific: the floor is due to bond angles** (the premise of B^k) | TD F.1: angle RMSE 4.1° (analysis l. 149). TD itself argues angles are stiff | `floor_best − angle_oracle_rmsd` is a small share; the floor sits in stereo-mismatched or ring pairs. Synthetic example: a cyclopropane with random stereo keeps a 0.5 Å floor that the angle oracle does not remove | **J-decomp** (new) |
| **G2: conformer matching costs accuracy** (non-physical targets, floor) | TD Table 8 (train on GT L collapses; that is B1) | B5/B4 close most of the gap, making it a cheap matching fix rather than a new model; B1 on GT L is no better than CTRL on RDKit L | B1 (2×2 with A1/A1c), B4, B5 vs CTRL_rematch |
| **G3: one-directional coupling** | TD F.1: GT-other-L floor is 0.284 vs 0.324 Å on DRUGS (l. 1289). Most of the DRUGS floor survives a *perfect independent* L sampler, so only L\|τ coupling can remove it. This is actually the strongest in-paper argument for G3, and it is **not** in gap_synthesis | A1c ≈ A1; `floor_gtother` ≈ 0 on QM9 (then a better independent sampler suffices); B2 ≈ CTRL_rematch at small δ | **A1c vs A1** (new), `floor_gtother` (new), B2 |

Contested: is B2 (random L̂-C pairing) a G3 test? R1 no: it still DE-optimises each pair and only tests the assignment. R2 no: TD frames it as a matching ablation (Table 8). R3 yes. → 2-1: **B2 is not a coupling test**; A1c − A1 is the G3 arm.

Contested: is QM9 the right place to motivate FlexiTors? R1: QM9 has few heavy rotors, so the torsion headroom is inherently small. R2: TD App. H names QM9 as where L matters most. R3: yes. → 3-0 keep QM9 for the floor analyses, but claims about drug-like molecules need DRUGS (plan "Follow-ups").

**Recommendation for gap_synthesis.md (not edited here; it is the research agent's file):** change "TD's QM9 AMR-R of 0.178 already sits on it" (§0, G1 row) to "is close to TD's conformer-matching estimate (an upper-biased floor; see review M1)". Add TD F.1's 0.284 Å GT-other-L result as the G3 argument.

## 5. SLURM correctness

| ID | Sev. | Issue | Evidence | Fix |
|---|---|---|---|---|
| X1 | major | Silent evaluation of half-trained models (above). The resubmit path and `FORCE_TRAIN` also reused stale `confs.pkl` via gen_eval's skip-if-exists | `train.py:36-46` saves best at every improvement; old sbatch tested `best_model.pt` | `common.sh: train_complete` (`.train_done`, or `last_model.pt` epoch ≥ N−1, which also back-fills runs that finished before the marker existed). Incomplete dirs are moved aside, never deleted. The inference array refuses our own checkpoints that are not complete. `FORCE_TRAIN` moves old results aside |
| X2 | pass | Array indexing: inference 42 non-comment lines ↔ `--array=0-41`; train 16 ↔ `0-15`. `grep -vE '^\s*(#\|$)'` skips comments and blank lines | counted after the edits | Header comments updated ("keep in sync") |
| X3 | minor | Seed pickles were built inside array tasks without a lock, so concurrent A1/G1 tasks race | `ablation_inference_array.sbatch` | `flock` around `make_seed_pickles.py` |
| X4 | minor | The analysis runtime grew (about 9 DE runs per GT conformer, plus automorphism maps) | 1.2 s per GT conformer measured locally for `--mode test`, 0.3 s for `--mode run` | `analysis_qm9.sbatch --time` raised from 12 h to 24 h; steps skip existing outputs, so a resubmit resumes per step |
| X5 | pass / human | Times: baseline 3 d for an estimated 12-25 h; TRN 2 d for an estimated 5-15 h; no resume, so the limits must cover full runs. They do, **if** the plafnet2 partition MaxTime is ≥ 3 d, which is not verified | `train_qm9_baseline.sbatch:9` | **Human:** check `scontrol show partition plafnet2`. Also check the GPU count on gnode118 against `%4` + `%2` + the baseline |
| X6 | minor | `/scratch/nishanth.r` is local to gnode118. `sbatch /scratch/...` from a login node cannot read the script, and `--output` needs `logs/` to exist before submission | `common.sh:4-11`, `setup_env.sh:15` | Submit from gnode118 (or from a `$HOME` copy of `slurm/`). Create `logs/` first (setup_env's srun line) |
| X7 | minor | The cluster needs the updated patch. `setup_env.sh` re-checks out upstream and re-runs `git am`, so re-running its repo step picks up all 6 commits | `setup_env.sh:118-124` | Plan step 1 updated |

## 6. Changes made

**Code** (branch `flexitors-ablation-hooks`; `patches/0001-ablation-hooks.patch` regenerated with 6 commits; `git am` onto upstream 5f713b42 verified):
- `c8d2dc4` Generate: `--seed_confs_cycle` (`generate_confs.py`, `diffusion/sampling.py`).
- `49a4811` Standardize: `--heavy_objective` (`standardize_confs.py`, `utils/standardization.py`; smoke-tested).
- (`6218977` is the metric verifier's commit on top.)

**Tools:**
- `local_structure_analysis.py`: K = 2L seeds, `floor_best(_sym)`, `floor_gtother`, angle and local oracles, ring and stereo columns, `--mode run`. The metric verifier then made its objectives automorphism-aware.
- `breakdown.py`: uses `floor_best_sym`; the ceiling is labelled estimated; confounding note added.
- `make_seed_pickles.py`: stereo from 3D.
- New: `paired_compare.py`.
- `review/tests/test_metrics.py`: 0 failed after all changes.

**SLURM:**
- `common.sh`: `train_complete`.
- Training scripts: completion markers.
- Inference array: completion check and `flock`; array range 0-41.
- `ablations_inference.tsv`: +12 lines.
- `ablations_train.tsv`: +CTRL_rematch and +B5 (array 0-15).
- `standardize_qm9` / `featurize_qm9` / train array: `rematch` and `heavy` variants.
- `analysis_qm9.sbatch`: J-run, paired comparisons, 24 h.

**Plan** (`notes/ablation_plan_qm9.md`):
- §0.3 caveat (M1).
- §1 uncertainty and pre-registered endpoints.
- New rows: A0-floor wording, A1c, CTRL_rematch, B5, J-run, J-decomp.
- G1 stereo note.
- Falsifier table in §5.
- Submission steps and budget.

## 6b. Budget revision (user decisions, applied after the first report)

- **Primary δ = 0.5 Å.** Primary endpoints are AMR-R mean and COV-R@0.5, with the small-δ sweep secondary. Applied to `paired_compare.py` (defaults), `breakdown.py` (help text and sweep label), `analysis_qm9.sbatch` and the plan.
- **Budget: 4 days × 4 GPUs (384 GPU-h), target ≤ 330.**
  - Inference: 23 lines on the released checkpoint, `--array=0-22%4`, `--time` 6 h.
  - Training: 15 runs (CTRL_base, B1, CTRL_rematch, B2, B5 × 3 seeds), `--array=0-14%4`, `--time` 36 h, started `afterany` the inference array so at most 4 GPUs are in use.
  - Total: about 160 GPU-h nominal, about 243 GPU-h upper.
  - Deferred lines moved to `slurm/ablations_{inference,train}_deferred.tsv`.
  - Schedule table in plan §3.3.
- CPU `--time` tightened: featurize 6 h, standardize 12 h (resumable). All jobs are ≤ 4-00:00:00. Each analysis submission waits for the previous one, so they do not clobber each other's outputs.

## 7. Needs a human decision

1. ~~Primary δ~~: resolved, 0.5 Å.
2. ~~GPU budget~~: resolved. The budgeted plan is about 243 GPU-h upper. Epoch time is unmeasured; re-plan if CTRL_base s0 runs slower than 8 min/epoch (plan §3.1).
3. ~~Partition MaxTime / GPU count~~: resolved (4 days, 4 GPUs). Still unverified: the CPU core count of gnode118 (plan §3.2).
4. Whether to edit `papers/gap_synthesis.md` as recommended in §4.
5. Whether to add the optional B3-train-only line (C6) and an E-inf 40-step line (C5).
6. The working copy of `torsional-diffusion/` was left on `master` (its state at the start). The branch holds all commits.
