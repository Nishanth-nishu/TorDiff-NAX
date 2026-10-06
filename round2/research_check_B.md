# Research check B: the implemented round-2 logic against the literature

Written 2026-10-07 by research agent B (critique side). I did not read research agent A's check.

**Scope.** Code at HEAD 47ceacf; the diff is `503191f..HEAD`. I read:
- `tools/lgeom.py`, `tools/make_l_seed_pickles.py`, `tools/build_paired_pickles.py` (header only), `tools/s4_gate.py`
- `tools/paired_compare.py`, `tools/local_structure_analysis.py`
- the TD diffs (`utils/dataset.py`, `diffusion/score_model.py`, `diffusion/sampling.py`, `generate_confs.py`)
- `slurm/r2_evalsets.tsv`, `slurm/ablations_train_round2.tsv`

**Conventions**
- Code citations are `path:line` at HEAD.
- Paper quotes are copied from `papers/**/fulltext`. Pages are PDF pages (form-feed count). The Å glyph is written as "Å". Text broken by two-column extraction is quoted only where it is contiguous.
- [B-calc] marks numbers I computed locally; anything I could not verify is marked UNVERIFIED.
- I did not edit any code.

## Summary verdict table

| Item | Verdict | One-line reason | Fix |
|---|---|---|---|
| I1 Cartesian λ-interpolation (heavy Kabsch, automorphism relabel, terminal bonds restored) | **CORRECT WITH CAVEAT** | The two endpoints are torsion-matched, so the difference is small and mostly L. Then linear Cartesian interpolation after Kabsch is the Euclidean geodesic used by flow-matching conformer models (ET-Flow), and it keeps rings closed. λ is not a calibrated error scale and midpoints are not physical states. The test-time λ family drops whole molecules. | Plot against the measured F5 error (`l_error.csv`), not λ. Report the population dropped by `make_l_seed_pickles.py:182-184`. An internal-coordinate path is only needed if a round-3 model diffuses internals. |
| I2 S2 jitter σ = 0.04 Å per axis | **CORRECT WITH CAVEAT** | At σ = 0.04 the heavy-angle RMSD is 3.3° (RDKit 3.9°) and the heavy-bond RMSD 0.056 Å (RDKit 0.036 Å) [B-calc], so the magnitude brackets RDKit. But it is iid, zero-mean and fixed-level, whereas RDKit error is systematic and ring-dominated. Cascaded diffusion recommends Gaussian augmentation, ideally amortised over the level. | Keep it. Interpret the results through S1's measured-error axis. If B6 helps only partly, the next step is level-conditioned jitter (Ho §3.2) or RDKit-structured noise (S3/S4). Note that the val loss is jittered too. |
| I3 S4 λ embedding concatenated with σ; λ = 0 on RDKit L, λ = 1 on GT L | **CORRECT WITH CAVEAT** | This is exactly Ho et al.'s "extra time embedding input for s", entering where σ enters (`score_model.py:192-203`). At the endpoints λ means "what kind of L"; for a future generated L, λ must be estimated. | Keep it. Report that S4 at λ = 0 vs CTRL_rematch is the fair comparison. Add a calibration curve, using the `S4x` λ grid on MMFF L, before claiming any intermediate-λ use. |
| I4 S3 Bernoulli 0.5 RDKit/GT, equal caps | **CORRECT** (minor caveat) | Same-conformer pairing with equal ≤ 30 caps (`dataset.py:79-98`) removes the B1 data confound. Mixing 50 % of examples has a direct precedent (Ho et al.). Caveat: there is no origin flag, so the model can infer the L source from L itself. | None required. Interpret S3 together with B1cap (p = 1) and CTRL_rematch (p = 0). |
| I5 S5 MMFF before conformer matching | **CORRECT WITH CAVEAT** | Upstream TD code option (`standardize_confs.py:18, 83-87`). MMFF94s is used at both train and test (`standardization.py:191-193`, `sampling.py:21-26`). The TD paper gives no guidance. MMFF minima are not GEOM's GFN2-xTB minima (Nikitin et al.). Test-time MMFF failures are silent. | Count test-time `try_mmff` failures (it returns False and leaves L unrelaxed). Report `mmff_error` training drops, already planned. |
| I6 S4 trains only on pair_ok pairs; controls re-scored on a subset | **CORRECT WITH CAVEAT** (selection bias not removed) | Re-scoring controls on a *test* subset does not neutralise a *training*-population difference. The dropped pairs are poorly matched, flexible molecules. Separately, the λ seed family drops whole test molecules with any unsafe pair. | Cheapest: keep unsafe pairs but draw λ ∈ {0, 1} only for them (the endpoints are always valid). Otherwise train one CTRL_rematch seed on the same filtered set. Report what the dropped molecules are (n torsions, rings). |
| I7 Evaluation (intersection, failures, Holm, leak test, DE restarts) | **CORRECT WITH CAVEAT** | Matches the TD/GeoMol code convention, plus a pre-declared intersection secondary that reflects the known RDKit-failure issue (MCF). The leak test has correct permutation and Boltzmann nulls. Gaps: Holm families are not scripted yet; the bootstrap ignores training-seed variance; the leak test silently assumes generated-conformer order. | Pre-script the `paired_compare` calls (fixed arm lists; ORACLE arms excluded from confirmatory families). Add a seed-level (hierarchical) bootstrap or require effect > 2 × seed SD. Assert `len(gens) == K` and the order in the leak test. Add COV@0.05 for FM-refiner comparability. |
| I8 Consistency with TD's design | **CORRECT** | All L changes happen before the torus perturbation, so the score target `edge_rotate` and the wrapped-normal DSM are unchanged (`dataset.py:59-76`). σ schedule and defaults are unchanged. S4 at λ = 0 is TD's conformer-matched distribution. DE quirk reproduced (defect 4). | None. Keep the golden tests. |

---

## I1. Measuring L quality by Cartesian interpolation

**What the code does**
- **Test-time seeds** (`tools/make_l_seed_pickles.py:146-197`):
  - Follows the training recipe: ETKDG on the GT graph, von-Mises cost, Hungarian matching, DE with the return value discarded as `standardize_confs.py:106` does (`:162-175`).
  - Then `lgeom.align_pair` (`tools/lgeom.py:133-155`): picks the heavy-atom automorphism with the lowest heavy Kabsch RMSD (the GetBestRMS correspondence), applies a heavy-atom Kabsch fit to all atoms, and relabels terminal atoms twice.
  - Then `pair_check` at λ = 0.5 (`lgeom.py:212-256`): CIP labels, sp3 signed-volume flips, bond deviation ≤ 0.05 Å, non-bonded distance ≥ 0.9 Å.
  - Then `interp_x` (`lgeom.py:169-185`): linear interpolation, with every terminal atom put back on the linearly interpolated bond length.
  - **Population:** a single failing pair drops the whole molecule from every λ/A5 pickle (`make_l_seed_pickles.py:182-184`, with a shared-population assert at `:278-280`).
- **Training (S4):** the identical formula in torch (`torsional-diffusion/utils/dataset.py:18-34`; tested equal).

**What the literature implies**
- ET-Flow builds its probability paths as linear Cartesian interpolants after Kabsch alignment. It does so precisely to shorten paths and preserve structure: "rotationally aligning them using the Kabsch algorithm" (ET-Flow p. 4) and "It(x0, x1) = tx1 + tx0" (p. 4).
- Riemannian flow matching notes that on Euclidean space the geodesic is the straight line (RFM p. 2). So Cartesian linear interpolation is the canonical path when the object lives in R^3n.
- Internal-coordinate approaches (GO-Flow, FoldingDiff) argue for internal manifolds *for generation*. They criticise "applying Gaussian noise uniformly to all atomic coordinates" (GO-Flow p. 2). They say nothing about interpolation used as an error probe.
- Rings are the obstacle for internal-coordinate interpolation. TD itself puts ring torsions inside L: "torsion angles in cycles (or rings), which cannot be rotated independently, are considered part of the local structure L" (TD p. 4). Interpolating ring internals independently breaks ring closure, while a Cartesian blend of two closed rings stays closed to first order.

**Assessment**
- The matched RDKit conformer carries DE-fitted torsions close to GT; the training residual has a 0.109 Å median (`local_structure_std.log`). So X − Y is a small displacement dominated by L, and linear Cartesian interpolation agrees with internal-coordinate interpolation to first order. Where it does not (large-RMSD pairs), `pair_check` catches it: the 323 bond_dev / 151 inversion failures in `IMPLEMENTATION.md` §4 are exactly the poorly matched 0.4–1.0 Å pairs.
- **Automorphism relabelling** before blending is required. Without it, a methyl or isopropyl is interpolated through a planar centre. This matches the GetBestRMS correspondence used by the evaluator.
- **Restoring terminal bond lengths** is a justified correction. O–H/N–H rotors are not TD torsions (≥ 2 atoms per side; `local_structure_analysis.py:108-133` uses the same rule), so their H is not torsion-matched, and chord shortening would be an artefact.
- **Caveats**
  - (a) λ is a path coordinate, not an L-quality measure. Error along the path is roughly linear in λ for small displacements but differs per molecule. The x-axis of the dose-response must be the measured F5 error (`lgeom.l_subset_errors`, `l_error.csv`), as SHORTLIST F5 requires.
  - (b) Midpoints are not physical: they are neither on a PES path nor minima. They are valid as probes, not as "plausible molecules".
  - (c) The error direction is RDKit's *systematic* error, which makes it a more realistic perturbation than isotropic noise. That is a strength.
  - (d) Population bias: dropping molecules with any unsafe pair biases the λ/A5 family toward rigid, well-matched molecules. Comparisons are valid only within that family (the gate already does this, `s4_gate.py:52-56`). Report the dropped count and the dropped molecules' torsion/ring counts.

**Verdict.** CORRECT WITH CAVEAT. Internal-coordinate interpolation is not needed for round 2.

---

## I2. S2 Gaussian jitter of GT L, σ = 0.04 Å per axis

**What the code does.** In `TorsionNoiseTransform.__call__`, after the conformer is chosen, it applies `data.pos + l_jitter * randn_like(pos)` to **all atoms**, freshly at each access (`utils/dataset.py:59-63`). This happens before the torsion perturbation (`:74-76`), so `edge_rotate` is unchanged. The same transform also serves the validation loader (`:347-355`).

**Is σ sensible?** Simulation [B-calc] (5 MMFF-optimised QM9-like molecules, 200 draws; `scratchpad/jit.py`):

| | heavy-bond RMSD | X–H bond RMSD | heavy-angle RMSD |
|---|---|---|---|
| σ = 0.04 | 0.056 Å | 0.056 Å | 3.30° |
| σ = 0.02 | 0.028 Å | — | 1.64° |
| RDKit vs GT (`local_structure_test.log`) | 0.036 Å | — | 3.88° |
| GT vs GT | 0.004 Å | — | 1.68° |

- σ = 0.04 matches RDKit's angle error and overshoots its bond error by ~1.5×. σ = 0.02 is ~GT-GT for angles. **So the pair {0.02, 0.04} brackets the relevant range.**
- **Mismatch in structure:**
  - RDKit error is systematic: biased ETKDG bond-length templates, ring geometry. Rings carry ≥ 76 % of the L-attributable floor (research_B §2).
  - Jitter is iid and zero-mean, so the model can learn to "average it out" in a way that does not transfer to a biased shift.
  - A side effect: jitter also moves heavy torsions by ~2–3°. That is comparable to σ_min = 0.01π = 1.8° (`utils/parsing.py:25`) and slightly blurs the training target.

**Literature**
- Cascaded diffusion: conditioning augmentation is effective because it "alleviates compounding error in cascading pipelines due to train-test mismatch" (Ho p. 3). Gaussian noise is the most effective at low resolution (Ho p. 6).
- Ho et al. also find it practical to train "amortized over the strength of conditioning augmentation and pick the best strength in a post-training hyperparameter search" (Ho p. 6).
- Ning et al.: "explicitly modelling the prediction error during training" (p. 2).
- S2 uses one fixed level with no level input. That is a valid simplest instance, but at test time on clean GT L the model sees an L that is *cleaner* than anything it trained on.

**Verdict.** CORRECT WITH CAVEAT.
- Keep it as implemented: it is the pre-registered single-factor test against B1.
- If B6 is ambiguous, next try σ ~ U[0, σ_max] with σ as input (that is S4-like), or structured noise.
- The jittered validation loss means best-epoch selection optimises for jittered L. That is consistent with the training objective, but state it.

---

## I3. S4 λ-conditioning

**What the code does**
- λ ~ U[0, 1] per training sample (`dataset.py:91-95`).
- λ is embedded with the same sinusoidal `get_timestep_embedding`, scaled to [0, 1e4] like σ, and concatenated to the σ embedding (`diffusion/score_model.py:192-203`). It therefore enters the node and edge input embeddings exactly where σ does (`:70, :76`).
- At inference a constant CLI λ is given (`sampling.py:169-172`); the guard requires it for, and only for, λ-models (`generate_confs.py:106-112`).
- Evaluation: λ = 0 on RDKit/MMFF seeds and λ = 1 on GT-cycle (`slurm/r2_evalsets.tsv:41-48`; `ablations_train_round2.tsv:14-16`).

**Literature**
- Ho et al.'s truncated/non-truncated augmentation trains a single network that takes the corrupted input "along with s, and this can be accomplished using a single network with an extra time embedding input for s" (Ho p. 8). S4 is a direct transfer: the λ embedding plays the role of s.

**Caveats**
- (a) At λ = 0 the training data are exactly the matched RDKit conformers. So S4@λ0 vs CTRL_rematch is the like-for-like test, and the right primary comparison.
- (b) λ is not an observable property of a new L. For a future FlexiTors L generator the level must be estimated or tuned, as in Ho's post-hoc search. The `S4x` grid on MMFF L (`r2_evalsets.tsv:43-44`) is the right first calibration and should not be trimmed if budget allows.
- (c) λ = 1 at test uses `--l_level 1` on gtLcycle. The V18 check (`s4_gate.py:47-60`) correctly verifies that the pipeline, not the model, reproduces GT at λ = 1.

**Verdict.** CORRECT WITH CAVEAT.

---

## I4. S3 Bernoulli mixing

**What the code does**
- `_select_paired_L` (`dataset.py:79-104`) picks one conformer index k (uniform or Boltzmann). It then uses the matched RDKit L `pos[k]` or its own aligned GT L `gt_pos[k]` with probability p = 0.5.
- Both come from the same ≤ 30 matched conformers of the rematch pickles, so the caps are equal (B1cap is the same with p = 1).
- The pairing is preserved through the reacted-conformer filter (`dataset.py:305-311`).

**Literature**
- Ho et al. apply augmentation to half the data: "During training, we apply this blurring augmentation to 50% of the examples. During inference, no augmentation is applied" (Ho p. 7).
- TD's rationale for matching (TD p. 7) is respected for the RDKit half.

**Caveat.** With no origin indicator, the model can infer the source from L itself: RDKit angles and bonds have a distinct signature. So S3 may learn two regimes instead of a smooth robust map. This is a question to answer, not a defect: B1cap (p = 1) and CTRL_rematch (p = 0) bracket it.

**Verdict.** CORRECT. The equal-cap design fixes the B1 data confound noted in round 1 (`standardize_confs.py:58-67, 74-82` vs raw).

---

## I5. S5 MMFF before conformer matching

**What the code does**
- `standardize_confs.py --mmff` (upstream option, `:18`) relaxes all ETKDG seeds with MMFF94s before matching (`:83-87`; `utils/standardization.py:189-197`). Molecules with an MMFF exception are dropped as `mmff_error`.
- At test time `--pre_mmff` uses the same MMFF94s on the SMILES path (`diffusion/sampling.py:21-26, 64`). Seed pickles get MMFF at build time (SHORTLIST F2; `make_l_seed_pickles.py` etkdg source).

**Literature**
- The TD paper never reports this variant; it mentions MMFF only for the Boltzmann generator (TD p. 10). **NO LITERATURE GUIDANCE from TD.**
- Zhou et al. obtain strong QM9 baselines with RDKit + MMFF: "utilizing the RDKit and MMFF force field optimization methods" (Zhou p. 3).
- Nikitin et al. warn about "reliance on force fields in- consistent with the reference data" (p. 1). GEOM conformers are GFN2-xTB/CREST, so MMFF minima are a biased target.
- Round-1 evidence nevertheless shows MMFF L is closer to GT: angle RMSD 2.44° vs 3.89° (`local_structure_test_mmff.log`). The floor drops 0.0993 → 0.0832 (`A2_pre_mmff/floor_run.log`).

**Caveats**
- (a) Train/test asymmetry on failure. Training drops `mmff_error` molecules, while test-time `try_mmff` silently returns False and leaves unrelaxed L. Count those molecules.
- (b) `MMFFOptimizeMoleculeConfs` non-convergence (default 200 iterations) is ignored in both paths. That is consistent, so it is acceptable.
- (c) MMFF also moves torsions before matching. That is harmless, because DE re-matches them.

**Verdict.** CORRECT WITH CAVEAT. It is a legitimate "best fixed-L TD" baseline, not a physics-correct L.

---

## I6. S4 on pair_ok-safe pairs only

**What the code does**
- At load, conformers with `pair_ok = False` are removed, and molecules left with none are dropped (`dataset.py:144-162`). Measured: ~8 % of conformers and ~4 % of molecules (`IMPLEMENTATION.md` §4; 35/918 molecules on `000.pickle`).
- The user ruling (`DECISION.md`) re-scores CTRL_rematch and B1 "on S4's own molecule set".

**Problem**
- The filtered pairs are the poorly DE-matched ones (heavy RMSD 0.4–1.0 Å, `IMPLEMENTATION.md` §4), i.e. flexible molecules.
- S4 thus sees fewer flexible training examples than CTRL_rematch. That is a **training-distribution** difference, and re-scoring on a **test** subset cannot remove it: the test molecules are different molecules. It only equalises the evaluation population.
- If "S4's own molecule set" means the λ-family test population (`make_l_seed_pickles.py:182-184`), that is itself a rigid-biased subset.
- Both biases favour S4 on flexible test molecules being *worse* and on rigid ones being *comparable*. So the direction is conservative for S4 overall, but it confounds the interpretation.

**Literature.** Filtering training data is standard: TD drops molecules that cannot be matched (`standardize_confs.py:60, 76, 82`), and Nikitin et al. remove fragmented GEOM molecules (p. 8). The requirement is that compared arms share the filter. No paper addresses this design.

**Fix options**
- Preferred, no extra GPU: keep unsafe pairs in S4 but sample λ ∈ {0, 1} only for them. The endpoints are the valid matched-RDKit and GT conformers, so the training population equals CTRL_rematch's. This needs about 5 lines in `_select_paired_L` and the filter at `dataset.py:144-162`.
- Alternative: 1 seed of CTRL_rematch trained with the same pair_ok filter.
- In either case, tabulate the dropped molecules by torsion count and ring class.

**Verdict.** CORRECT WITH CAVEAT. The selection bias is real, but bounded at ~4 % of molecules; fix it as above.

---

## I7. Evaluation logic

**What the code does**
- **Primary endpoints:** union, failures = COV 0 (`paired_compare.py:69-77`; same as `evaluate_confs.py:152`).
- **Intersection universe:** `--universe intersection` = molecules present in every file of the reference and every arm (`paired_compare.py:105-106`). `--report_failures` prints failure counts.
- **Holm:** over the `--primary` rows of one call (`:90-98, :145`). A family is therefore defined by the call.
- **Leak test** (`local_structure_analysis.py:446-531`):
  - Builds a torsion-space RMS distance with circular wrapping and min over heavy automorphisms (`:471`).
  - Computes the hit rate of the nearest GT == source (i mod L) (`:472-474`).
  - Two nulls: a 200-permutation null of source labels (controls for the nearest-index distribution) and a Boltzmann null (mean weight of the source; the hit rate of a model that ignores L and samples GT conformers by Boltzmann weight).
  - Positive and negative controls: `A1c_gtL_cycle_sanity` (no model) and its random-torsion version.
- **DE restarts:** seeds `base + 7919·r`, min kept. The default reproduces round 1 (`:187-195`).

**Literature comparison**
- TD and GeoMol report mean and median COV/AMR with failures handled by the shared evaluator code. TD never reports failure counts.
- MCF documents that TD silently fails on RDKit embedding failures: "we found that 25 molecules failed to be generated" (MCF p. 7). Reporting failures plus an intersection endpoint is therefore an improvement over the literature, not a deviation.
- S23D (p. 16) and FM-refiner (p. 7) say δ = 0.5 is saturated. FM-refiner uses δ = 0.05, so COV@0.1 is a reasonable middle ground. **Add COV-R@0.05 as a reported row**, already available in the SWEEP lines, for literature comparability.
- Nikitin et al.'s pitfalls concern valency/stability metrics and force-field inconsistency (p. 1), not RMSD coverage. The round-2 code uses no stability metric, so those pitfalls are avoided. Their recommended Δ bond/angle/torsion metrics (p. 11) are partly covered by F5 (`lgeom.l_subset_errors`) for seeds; extend it to generated conformers via `tools/geometry_metrics.py`.

**Caveats / fixes**
- (a) Holm families are only defined by which arms are passed. No round-2 analysis script exists in the diff yet. Pre-script the calls (fixed arm lists per family) before the results arrive, and keep ORACLE arms out of confirmatory families (SHORTLIST says oracles are analysis only).
- (b) `combine` averages the replicate files per molecule (`paired_compare.py:69-77`), so the bootstrap CI reflects molecule sampling only, not training-seed variance (round-1 CTRL training-seed SD 0.0013 Å). For training arms, add a two-level bootstrap (seeds, then molecules), or require |effect| > 2 × `ref_replicate_sd`. This matters for the S2 non-inferiority margin of 0.004 Å.
- (c) The leak test assumes generated conformer i in `confs.pkl` came from seed i. It guards `n_seed == L` (`:457`) but not `len(gens) == K` or the order. Assert both. **UNVERIFIED** whether `generate_confs.py` can drop single conformers.
- (d) The λ seed family drops molecules. Under the union primary they count as failures (COV 0). Evaluate ORACLE λ runs on the intersection only, as `s4_gate.py:52-56` already does.
- (e) The DE-restart check is correct. Report the max per-molecule change, not only the mean.

**Verdict.** CORRECT WITH CAVEAT.

---

## I8. Consistency with TD's design

- **Conformer matching rationale.** TD: training on GT L causes "a distributional shift at test time" (TD p. 7).
  - S2, S3, B1cap and S4 intentionally train on GT L as tested remedies, each with a control (B1 / CTRL_rematch).
  - S4's λ = 0 data are exactly the matched distribution TD prescribes. Nothing silently changes the default path; the golden tests confirm defaults are unchanged (`IMPLEMENTATION.md` §3).
- **Torus score.** All L modifications (jitter, mix, interpolation) happen before `modify_conformer` with the wrapped-normal torsion updates. So the DSM target `edge_rotate` (`dataset.py:74-76`) is still the exact torus perturbation of the conditioning conformer, as TD's derivation requires.
- **σ schedule.** Unchanged: `parsing.py:25-26` defaults, and the same log-uniform σ sampling. The λ embedding uses the same embedding function and scale as σ, so no new scale is introduced.
- **DE quirk.** The stored matched conformer is the last DE/polish evaluation, because `optimize_rotatable_bonds`'s return value is discarded at `standardize_confs.py:106`. The test-time λ seeds replicate this (`make_l_seed_pickles.py:166-170`), so λ = 0 test L comes from the training λ = 0 distribution. That is correct and faithful.

**Verdict.** CORRECT. No contradiction with TD's design.

---

## Citation table

| Claim | PDF path | Page | Exact quote |
|---|---|---|---|
| TD's reason for conformer matching | papers/core/2022_jing_torsional_diffusion.pdf | 7 | "there will be a distributional shift at test time, where only approximate local structures from p^G(L) are available" |
| Ring torsions belong to L in TD | papers/core/2022_jing_torsional_diffusion.pdf | 4 | "torsion angles in cycles (or rings), which cannot be rotated independently, are considered part of the local structure L" |
| TD uses MMFF only for the Boltzmann generator | papers/core/2022_jing_torsional_diffusion.pdf | 10 | "generator trained with MMFF [Halgren, 1996] ener- gies can sample the corresponding Boltzmann density" |
| Kabsch alignment before linear Cartesian paths | papers/related/2024_hassan_etflow.pdf | 4 | "by rotationally aligning them using the Kabsch algorithm (Kabsch, 1976)" |
| Linear interpolant in Cartesian flow matching | papers/related/2024_hassan_etflow.pdf | 4 | "The general interpolation between samples x0 0 and x1 1 can be defined as: It(x0, x1) = tx1 + tx0." |
| Closed-form geodesics on Euclidean / torus spaces | papers/related/2023_chen_riemannian_flow_matching.pdf | 2 | "On simple geometries, where geodesics are known in closed form (e.g., Euclidean space, hyper- sphere, hyperbolic space, torus, or any of their product spaces)" |
| Critique of isotropic Cartesian noise | papers/related/2026_liu_goflow.pdf | 2 | "applying Gaussian noise uniformly to all atomic coordinates." |
| Conditioning augmentation fixes train-test mismatch | papers/related/2021_ho_cascaded_diffusion.pdf | 3 | "conditioning augmentation is effective because it alleviates compounding error in cascading pipelines due to train-test mismatch" |
| Gaussian noise is the effective augmentation | papers/related/2021_ho_cascaded_diffusion.pdf | 6 | "what we found most effective at low resolutions is adding Gaussian noise (forward process noise)" |
| Amortised augmentation level | papers/related/2021_ho_cascaded_diffusion.pdf | 6 | "train super-resolution models amortized over the strength of conditioning augmentation and pick the best strength in a post-training hyperparameter search" |
| Level as an extra embedding input (S4 template) | papers/related/2021_ho_cascaded_diffusion.pdf | 8 | "this can be accomplished using a single network with an extra time embedding input for s." |
| Augmentation on 50 % of examples (S3 template) | papers/related/2021_ho_cascaded_diffusion.pdf | 7 | "During training, we apply this blurring augmentation to 50% of the examples. During inference, no augmentation is applied" |
| Input perturbation for exposure bias | papers/related/2023_ning_input_perturbation.pdf | 2 | "explicitly modelling the prediction error during training" |
| RDKit + MMFF baselines | papers/related/2023_zhou_dl_conformation_critique.pdf | 3 | "utilizing the RDKit and MMFF force field optimization methods" |
| Force-field inconsistency pitfall | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 1 | "reliance on force fields in- consistent with the reference data" |
| Nikitin's bugs are valency/stability bugs | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 1 | "including incorrect valency definitions, bugs in bond order calculations" |
| Fragmented GEOM molecules removed | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 8 | "These failures produced fragmented molecules and unstable valencies" |
| Recommended Δ local-geometry metrics | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 11 | "we suggest to assess differences in bond lengths, bond angles, and torsion angles of generated and optimized counterparts." |
| TD failures documented by MCF | papers/related/2023_wang_mcf.pdf | 7 | "we found that 25 molecules failed to be generated" |
| QM9 at 0.5 Å saturated | papers/related/2025_gurev_s23d.pdf | 16 | "indicating that benchmark results on the QM9 metric are likely to be saturated." |
| FM-refiner uses δ = 0.05 Å | papers/related/2025_xu_fm_refiner.pdf | 7 | "we adopt the more challenging COV threshold of [δ] = 0.05 Å" ([δ]: glyph missing in the extracted text) |

No new papers were needed. Every source above is already in `papers/INDEX.md`; the Ho and Ning rows were added by research agent A.
