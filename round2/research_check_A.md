# Round 2, research agent A: literature cross-check of the implemented code (I1–I7)

Written 2026-10-07 by research agent A (design side). I did not read research agent B's output.

Code state: `git -C C:\Users\HP\Desktop\tor_diff diff 503191f..HEAD` (HEAD = 47ceacf).

Scope: I read `SHORTLIST.md`, `DECISION.md` (including the user rulings), `IMPLEMENTATION.md`, the TD diff, and
`tools/{lgeom,build_paired_pickles,make_l_seed_pickles,s4_gate}.py`. The S4 eval matrix is in
`slurm/r2_evalsets.tsv` and `slurm/ablations_train_round2.tsv`.

Conventions:
- Page numbers are PDF pages; for every PDF cited, the printed page number is the same.
- Paths are relative to `C:\Users\HP\Desktop\tor_diff`; TD paths are under `torsional-diffusion/`.
- The new numbers below (jitter → bond/angle error) come from a 7-molecule simulation I ran with `tools/lgeom.py`
  `l_subset_errors`, 20 draws per σ, on MMFF-optimised ETKDG geometries (§I2). They are indicative, not a QM9-wide
  measurement.

## Summary verdicts

| Item | Verdict | One-line reason | Fix |
|---|---|---|---|
| I1 Cartesian RDKit↔GT interpolation (heavy Kabsch, automorphism + terminal relabelling, terminal bond lengths restored) as the L-quality knob | **CORRECT WITH CAVEAT** | Aligned Cartesian interpolation is the standard path in Cartesian flow matching (ET-Flow) and keeps rings closed by construction, which internal-coordinate interpolation does not (RINGER, PuckerFlow). But λ is not an "L-only" knob: pairs with residual torsion mismatch also interpolate torsions, and puckers pass through flattened rings. | Use the measured F5 error (`l_error.csv`), not λ, as the x-axis of every dose-response. Check chirality and bonds on a λ grid (0.1 steps), not only at λ = 0.5. Tighten or at least report the 0.9 Å clash threshold. No switch to internal coordinates needed. |
| I2 S2 isotropic jitter σ = 0.04 Å/axis, fresh per sample | **CORRECT WITH CAVEAT** | Same mechanism as CDM's Gaussian conditioning augmentation: noise on the conditioning input only, target unchanged. σ = 0.04 matches RDKit's angle error (~3.9° vs 3.88°) but overshoots its bond error (~0.055 vs 0.036 Å). Every sample is jittered (no clean examples), and the noise is isotropic, not RDKit-shaped. | Keep it as pre-registered. Interpret it as robustness to *random* L error. Optional round-3 variant: sample σ per example from [0, σ_max] with a clean fraction (CDM's range + 50% rule). Report the F5 errors of jittered seeds next to RDKit's. |
| I3 S4 λ concatenated to the σ embedding; test λ = 0 on RDKit, λ = 1 on GT | **CORRECT WITH CAVEAT** | This is CDM's amortised conditioning ("extra time embedding"), and λ is a parity-even scalar, so equivariance is kept. Endpoint protocol is right. For non-endpoint L (MMFF, a learned L), λ is a hyperparameter that CDM picks by post-hoc search; here it must be picked on validation, never on test. | Pick λ for MMFF / learned L on validation molecules. Label the test-set λ grid (S4x) descriptive. Optional: point masses at λ ∈ {0, 1} in training. |
| I4 S3 Bernoulli 0.5 mix, same conformer index, equal caps | **CORRECT** | Matches CDM's "augment 50% of examples" pattern, and pairing and caps make the two halves comparable. The RDKit half is identical to CTRL_rematch. The B1cap control isolates the cap. | None (the unconditioned mixture is the intended contrast to S4). |
| I5 S5 MMFF before conformer matching | **CORRECT WITH CAVEAT** | Train/test consistent: MMFF94s in both `standardize_confs.py --mmff` and `--pre_mmff`. But "strongest fixed-L baseline" is overstated: GEOM geometries are GFN2-xTB optima and MMFF disagrees with GFN2-xTB. Our own floors drop only 0.0986 → 0.0934 Å (mean). Test-time MMFF failures are silently skipped while training drops the molecule. | Call it the "MMFF fixed-L baseline". Report `mmff_error` and test-time MMFF failures. An xTB-relaxed variant is the stronger round-3 baseline. |
| I6 S4 drops pair_ok failures (~8% conformers, ~4% molecules) | **CORRECT WITH CAVEAT (confound, cheap fix)** | TD's conformer matching relies on a *complete* assignment so that training L has the RDKit marginal. Dropping poorly matched pairs breaks that, and it is a **training**-population change. Re-scoring controls on a test subset does not remove it. | Keep the unsafe pairs in S4 but train them only at the endpoints λ ∈ {0, 1} (no midpoints). The population is then identical to CTRL_rematch / S3. Alternatively add a control trained on the same filtered set. |
| I7 Consistency with TD's design (torus score, σ schedule, parity, conformer matching) | **CORRECT** | The torus score and loss, the σ sampling and the inference schedule are untouched. λ enters as a scalar where σ does. Jitter is parity-symmetric. S2/S3/S4 depart from conformer matching on purpose, and that departure is the experiment. Train and test DE handling are identical (defect 4). | Stale docstring: `tools/s4_gate.py:4` says pair_ok failures are "dropped and counted" in the pairing number; since d4d2ff1 they are kept and flagged. Fix the comment. |

---

## I1. Cartesian interpolation as the L-quality knob

### What the code does

**Alignment** (`tools/lgeom.py:133-154` `align_pair`):
- Y (GT) is relabelled by the heavy-atom graph automorphism with the lowest heavy-atom Kabsch RMSD (`:103-130`, `:142-146`).
- It is then Kabsch-fitted on heavy atoms and the fit is applied to all atoms; proper rotation only (`:65-73`).
- Terminal atoms that are graph-equivalent are relabelled by Hungarian matching, twice (`:80-100`, `:149-152`).

**Interpolation** (`tools/lgeom.py:169-184` `interp_x`; torch twin `torsional-diffusion/utils/dataset.py:18-34`):
- x_λ = (1-λ)X + λY.
- Every degree-1 atom is then placed back at the linearly interpolated bond length along the interpolated direction.

**Guard** (`tools/lgeom.py:212-254` `pair_check`), evaluated only at λ = 0.5:
- CIP labels equal at both endpoints.
- No sign flip at sp3 centres between X, x_0.5 and Y.
- Bond deviation from linear ≤ 0.05 Å (`:26`).
- Min distance between atoms ≥ 3 bonds apart ≥ 0.9 Å (`:27`).

**Where pairs come from:**
- Training pairs are the CTRL_rematch matched conformers, re-attached to their own GT by `geom_id` and verified by recomputing `conf['rmsd']` (`tools/build_paired_pickles.py:64-137`).
- Test seeds rebuild the training recipe exactly (`tools/make_l_seed_pickles.py:146-192`, DE return value discarded as at `standardize_confs.py:106`).
- A test molecule with any failing pair is dropped from the whole λ/A5 family (`make_l_seed_pickles.py:22`).
- F5 achieved errors are written per seed (`tools/lgeom.py:303-323`).

### What the literature does or implies

**Internal coordinates.** GO-Flow, FoldingDiff, RINGER and PuckerFlow put noise or paths on internal coordinates (C1, C2, C3, C4). They pay for it:
- RINGER needs a constrained-optimisation post-processing step to close rings (C3).
- GO-Flow needs internal-coordinate Jacobians (C1).
- PuckerFlow's selling point is that internal coordinates can incorporate "chemical constraints ... such as ring closure", which Cartesian models do not enforce (C4).

**Cartesian.** Cartesian flow-matching models (ET-Flow) use linear Cartesian paths between Kabsch-aligned endpoints as their standard construction (C5).

For an L-quality *diagnostic* and an *input-augmentation* path, Cartesian interpolation after heavy-atom alignment is therefore a recognised construction:
- Rings stay closed in the sense that ring bonds remain bonds, with lengths checked by `pair_check`.
- Chirality is preserved when the endpoints agree, and that is checked.

Internal-coordinate interpolation would bring ring-closure failures into exactly the stratum (rings) where D1 of `research_A.md` showed the QM9 floor lives.

### Where it falls short

1. **λ is not an "L-only" axis.**
   - Matched pairs still differ in torsions: the DE is H-inclusive with maxiter 15, and unsafe pairs have heavy RMSD 0.4–1.0 Å (`IMPLEMENTATION.md` §4).
   - x_λ therefore also interpolates torsion differences and non-matched rotors.
   - This is harmless at test time, because `perturb_seeds` re-randomises torsions (`torsional-diffusion/diffusion/sampling.py:94-102`).
   - It does mean λ = 0.5 is not "half the L error". The F5 table (`l_error.csv`) is the right x-axis; λ is only an index.
2. **Puckers.** Two different puckers interpolate through a flattened, strained ring, not along a physical pucker path (cf. GO-Flow's critique of uniform Cartesian treatment, C6). This shows up as `ring_dihedral_rmsd` in F5, and should be reported by ring stratum.
3. **The guard is a single-point check.**
   - The signed volume along a linear path is a cubic in λ, so a flip and flip-back between grid points is possible, though rare.
   - The clash threshold of 0.9 Å is far below normal non-bonded contact distances. I found no literature value in our corpus (NO LITERATURE GUIDANCE).
   - Fix: evaluate `pair_check` on λ ∈ {0.1, …, 0.9} (cheap), and report the `min_nonbonded` quantiles already stored in `stats/*.json` (`build_paired_pickles.py:163-166`).

**Verdict: CORRECT WITH CAVEAT.**

## I2. S2 isotropic Gaussian jitter

### What the code does

`torsional-diffusion/utils/dataset.py:59-63`:
- Adds `l_jitter * randn` to all atoms of the selected GT conformer.
- The noise is fresh per access, because `get()` deep-copies.
- It is applied **before** the torsion noise (`:71-76`), so the target `edge_rotate` is unchanged.
- Applied to every sample, at a fixed σ = 0.04 Å per axis (`slurm/ablations_train_round2.tsv` B6 lines; 0.02 exploratory).
- No σ input to the model.

### What the literature does

**CDM (C7, C8, C9):**
- Applies Gaussian noise or blur to the **conditioning input** of the second-stage model, not to its target.
- Draws the strength from a range ("randomly sample from a fixed range during training").
- Augments 50% of examples (blur), and amortises over the level with an embedding.
- The mechanism matches S2: L is TD's conditioning input, and the torsion target is unchanged.

**Ning et al. (C10, C11):**
- Perturb the *network input* with Gaussian noise to simulate inference-time error, again leaving the target alone.
- They use one constant strength, chosen from the measured error range.

So both variants exist in the literature: a fixed σ (Ning) and a randomised σ with clean examples (CDM).

### Is σ = 0.04 Å/axis sensible?

RDKit's actual L error on the test set (`cluster_sync/results/analysis/local_structure_test.csv`, means over 7554 GT conformers):

| quantity | heavy atoms | all atoms |
|---|---|---|
| bond RMSE | 0.0360 Å | 0.0316 Å |
| angle RMSE | 3.88° | 3.69° |
| GT-vs-GT bond variability | 0.0040 Å | |
| GT-vs-GT angle variability | 1.67° | |

My jitter simulation (7 QM9-like molecules × 20 draws, `lgeom.l_subset_errors`):

| σ per axis | bond RMSE heavy / all (Å) | angle RMSE heavy / all (°) | ring angle (°) | endocyclic dihedral (°) |
|---|---|---|---|---|
| 0.02 | 0.027 / 0.028 | 1.61 / 1.89 | 1.44 | 1.17 |
| 0.04 | 0.054 / 0.055 | 3.33 / 3.88 | 3.07 | 2.60 |

- σ = 0.04 reproduces RDKit's **angle** error magnitude.
- It overshoots the **bond** error by about 1.5×, and is about 14× the natural GT bond variability.
- σ = 0.02 matches neither well.

The bigger mismatch is shape:
- Jitter is independent per atom (high-frequency).
- RDKit's error is systematic: ring geometry carries most of the floor, per D1 in `research_A.md`.
- CDM found that *structured* augmentation (blur) works better than Gaussian noise when the error is not white (C8).

**Verdict: CORRECT WITH CAVEAT.** The implementation is a faithful conditioning augmentation, and σ = 0.04 is a defensible "angle-matched" level. Two consequences:
- S2 tests robustness to *random* L error (what a learned L-denoiser leaves behind), not to RDKit's systematic error. S3/S4 cover the latter.
- Because no sample is clean, the model never sees exact GT L. The GT-L evaluation (an ORACLE condition) is therefore slightly off-distribution for S2.

**Fix (optional, round 3):** σ ~ U[0, σ_max] with ~50% clean samples, or σ given as an input (CDM amortisation). Keep the pre-registered arm unchanged.

## I3. S4 λ conditioning and test protocol

### What the code does

`torsional-diffusion/diffusion/score_model.py:194-199`:
- Computes `get_timestep_embedding(λ·1e4, lambda_embed_dim=32)` and concatenates it to the σ embedding.
- The result enters node and edge features exactly where σ does; the layers are widened at `:70` and `:76`.

Training and inference:
- **Training:** λ ~ U[0,1] per sample (`utils/dataset.py:90-95`).
- **Inference:** a constant CLI `--l_level` (`generate_confs.py`, guard on λ-models; `diffusion/sampling.py:169-172`).

Evals:
- In-job: λ = 0 on SMILES/RDKit L and λ = 1 on gtLcycle (`slurm/ablations_train_round2.tsv` S4 lines).
- S4core: λ = 0 on etkdg / MMFF seeds, plus λ_in = λ_build at 0 and 0.5 (ORACLE).
- S4x: an MMFF λ grid on test (`slurm/r2_evalsets.tsv` S4 block).

### Literature

- **CDM** amortises a second-stage model over the augmentation level "by providing s as an extra time embedding input to the network". It picks s **post hoc** by a search (C12, C9). S4 follows this design.
- **DiffDock** product-space diffusion treats each factor's noise independently (C13); λ is the analogue of a second time variable.
- **Symmetry:** λ is a scalar (0e) feature, so the pseudoscalar output requirement of TD (C14) is untouched.

### Assessment

- λ = 0 on RDKit L and λ = 1 on GT L are the right endpoint tests. TD's complete-assignment argument (C15) makes conformer-matched RDKit L distributed like test-time ETKDG L, so λ = 0 on unmatched ETKDG seeds is in-distribution. The I6 caveat below qualifies this.
- For L that is not on the RDKit→GT segment (MMFF now, a learned L later), the "true" λ is undefined. CDM's answer is a validation search.
- The S4x MMFF grid is run **on the test set**, so it must be labelled descriptive and must not be used to pick λ.
- U[0,1] puts no probability mass exactly at the endpoints. This is harmless with a smooth embedding, but point masses (e.g. 25% each at 0 and 1, as R2A-2 proposed) would match CDM's clean-example practice. Optional.

**Verdict: CORRECT WITH CAVEAT.** Fix: choose λ for MMFF / learned L on validation molecules only.

## I4. S3 Bernoulli mix

### What the code does

`torsional-diffusion/utils/dataset.py:79-102`:
- Picks one conformer index k, then uses GT L `gt_pos[k]` with probability 0.5, otherwise the matched RDKit `pos[k]`.
- Both come from the same capped paired pickle (≤ 30 conformers; `build_paired_pickles.py` adds GT to the rematch pickles without re-embedding: docstring `:6-7`).
- No λ input.
- Unsafe pairs are kept: no midpoints are used, so the RDKit half equals CTRL_rematch's data (F4).
- B1cap (p = 1.0, 1 seed) controls for the cap versus uncapped raw B1.

### Literature

- CDM applies augmentation to 50% of examples, keeping the rest clean (C9). A 50/50 mixture without a level input is the non-amortised variant.
- TD (C16) shows that training on GT L alone hurts on RDKit L. The mixture tests whether half the data at the RDKit end prevents that without a label.
- Torsional-GFN shows that a torsion model conditioned on L can follow L shifts (C17). This is the reason to expect an unlabelled mixture to still work if L itself is informative.

**Verdict: CORRECT.** The GT half is the automorphism-relabelled GT. That is a graph automorphism of the same labelled graph, so it does not change what the model sees (`lgeom.py:136-139`).

## I5. S5 MMFF before matching

### What the code does

**Training side:**
- `standardize_confs.py:83-87` calls `utils/standardization.py:191-198` `mmff_func` (MMFF94s, default iterations) on all ETKDG seeds before Hungarian matching and DE.
- An exception drops the molecule (`mmff_error`).

**Test side:**
- `--pre_mmff` relaxes the ETKDG seeds with `try_mmff` (MMFF94s, `diffusion/sampling.py:21-26`, `:64`) on the SMILES path (`generate_confs.py:145`).
- A failure is silently ignored, and the seed stays unrelaxed.

**Evaluation:**
- B3 ×3, with the in-job eval `--pre_mmff` and a no-MMFF cell.
- Controls: CTRL_rematch with and without `--pre_mmff`.

### Literature

- GEOM conformers are GFN2-xTB geometries.
- MMFF systematically disagrees with GFN2-xTB: about 16 kcal/mol mean relaxation energy on GFN2-xTB structures (C18, C19).
- Modern generators already beat MMFF "in terms of alignment with GFN2-xTB" (C19).
- TD notes that OMEGA's better local structure gives better QM9 numbers (C20). A better fixed L sampler should help; MMFF is one cheap choice, not the strongest.
- Our own data agree. MMFF lowers the RDKit-L floor only from 0.0986 to 0.0934 Å (mean; median 0.0606 → 0.0462), and angle RMSE from 3.88° to 2.44° (`local_structure_test_mmff.log`).

**Verdict: CORRECT WITH CAVEAT.** It is a consistent, fair, cheap fixed-L baseline, but not the "strongest". Fixes:
- (a) rename it;
- (b) count test-time `try_mmff` failures, or make them consistent with training (drop the molecule, or log it);
- (c) a GFN2-xTB-relaxed variant is the principled stronger baseline (round 3).

## I6. S4 pair_ok selection

### What the code does

`torsional-diffusion/utils/dataset.py:145-162`:
- For `--l_interp` (S4 only), drops every conformer with `pair_ok = False`, and molecules left with none, in both the train and val loaders.
- Re-normalises the weights.

Measured: 92.1% of conformers kept on the probe; on 000.pickle, 35 of 918 molecules lost.

The failures are mostly poorly matched pairs (`IMPLEMENTATION.md` §4: bond_dev 323, inversion 151, stereo 15, clash 2). The user ruled to run on the safe pairs and re-score controls on S4's molecule subset.

### Literature

TD's conformer matching is justified by the **completeness** of the assignment (C15). Every GT conformer is paired with an RDKit seed, so the L seen in training has the RDKit marginal, and "there is no distributional shift in the local structures seen during training and inference".

Removing the worst-matched pairs breaks this for S4's λ ≈ 0 regime:
- The training L marginal is no longer the RDKit marginal. It loses exactly the seeds that RDKit gets most wrong, plus the flexible or stereo-tricky molecules.
- At test, S4 still faces all RDKit seeds.

This is a **training-population** difference. Re-scoring CTRL_rematch/B1 on a subset of *test* molecules does not remove it:
- the S4 model was trained on a different, easier training set;
- the test molecules most affected are not identified by the training filter.

There is a second, smaller selection on the test side:
- The λ/A5 seed families drop any test molecule with a failing pair (`make_l_seed_pickles.py:22`).
- Within-family B1−CTRL comparisons are fair, but they generalise only to the matched-well subpopulation. Report the dropped count beside every S1 λ/A5 result; `population.txt` already exists.

**Verdict: CORRECT WITH CAVEAT (a fixable confound).**

**Fix (cheap, keeps the user's ruling intact):** do not drop unsafe pairs in S4. Train them only at the endpoints, λ ~ Bernoulli({0, 1}), with no midpoint. Endpoints are exact, so no unsafe midpoint is ever built (DECISION D1 is respected).
- Code: in `utils/dataset.py:145-162` keep the conformers but keep their `pair_ok` flags.
- In `_select_paired_L` (`:90-95`), use `lam = float(np.random.uniform() < 0.5)` when `pair_ok[k]` is False.
- The S4 population is then identical to CTRL_rematch and S3, and the subset re-scoring becomes unnecessary.
- If the code is frozen, add a 1-seed control "CTRL_rematch on the S4-filtered set" (paired cache, `--l_mix_p_gt 0` with the same filter).

## I7. Consistency with TD's own design

**Torus score and loss: unchanged.** The only diff in `utils/training.py` adds timing, NaN fail-fast and the iteration limit; with defaults it is numerically the original loop.

**σ sampling: unchanged.**
- log-uniform σ in training (`utils/dataset.py:71`).
- Geometric inference schedule (`diffusion/sampling.py:114`).
- λ enters through the same sinusoidal embedding with the same [0, 1e4] scaling as σ (`score_model.py:194-199`).

**Parity.** TD requires an SE(3)-invariant, parity-equivariant (pseudoscalar) output (C14).
- λ is a parity-even scalar input, so it cannot break this.
- Isotropic jitter is inversion-symmetric.
- The torsion target remains the relative rotation `edge_rotate`, applied after the L change. This is consistent with TD's relative-update parameterisation.

**Conformer matching.** S2/S3/S4/B1cap deliberately train on GT or partly GT L, which TD shows hurts when tested on RDKit L (C16). That is the experimental question, not a defect. S3 and S4 keep TD's matched RDKit L as one endpoint of the training distribution (subject to I6).

**DE return value.** `standardize_confs.py:106` discards the DE result and keeps the conformer state the optimiser left in place. The test-time interp seeds reproduce this exactly (`make_l_seed_pickles.py:171`; defect 4), so the train and test λ = 0 distributions agree. This mirrors TD's code, not its text; no literature issue.

**Minor.** `tools/s4_gate.py:4` docstring is stale: pair_ok failures are no longer part of the "pairing clean" number. Fix the comment, so the gate's meaning is not misread. The user has already ruled on the threshold.

**Verdict: CORRECT.**

---

## Citation table

| # | Claim | PDF path | Page | Exact quote |
|---|---|---|---|---|
| C1 | GO-Flow: internal coordinates are the key component | papers/related/2026_liu_goflow.pdf | 7 | "that modeling internal coordinates (bond lengths, angles, and tor- sions) via entropic optimal transport is the most critical factor for generating high-fidelity molecular structures." |
| C2 | FoldingDiff noises angles with a wrapped normal | papers/related/2022_wu_foldingdiff.pdf | 5 | "to sample from a wrapped normal instead of a standard normal (Jing et al., 2022)" |
| C3 | RINGER: internal-coordinate reconstruction needs ring-closure fixing | papers/related/2023_grambow_ringer.pdf | 5 | "Adopting a sequential reconstruction method such as NeRF accumulates small errors that result in inadequate ring closure for macrocycles." |
| C4 | PuckerFlow: Cartesian models cannot enforce ring closure | papers/related/2026_schaufelberger_puckerflow.pdf | 2 | "cannot incorporate chemical constraints [27] such as ring closure (see Fig. 1b)." |
| C5 | ET-Flow: Kabsch-aligned Cartesian interpolation paths | papers/related/2024_hassan_etflow.pdf | 4 | "we reduce the transport costs between samples from the harmonic prior 0 and samples from the data distribution 1 by rotationally aligning them using the Kabsch algorithm" |
| C6 | GO-Flow: critique of uniform Cartesian treatment | papers/related/2026_liu_goflow.pdf | 2 | "applying noise uniformly to all Cartesian coordinates. This assumption is physically flawed" |
| C7 | CDM: conditioning augmentation fixes train-test mismatch | papers/related/2021_ho_cascaded_diffusion.pdf | 3 | "We empirically find that conditioning augmentation is effective because it alleviates compounding error in cascading pipelines due to train-test mismatch" |
| C8 | CDM: Gaussian noise at low resolution, blur at high resolution | papers/related/2021_ho_cascaded_diffusion.pdf | 6 | "what we found most effective at low resolutions is adding Gaussian noise (forward process noise), and for high resolutions, randomly applying Gaussian blur to z." |
| C9 | CDM: strength sampled from a range; 50% of examples augmented | papers/related/2021_ho_cascaded_diffusion.pdf | 7 | "randomly sample from a fixed range during training. We perform hyper-parameter search to find the range for . During training, we apply this blurring augmentation to 50% of the examples." (σ glyph missing in the fulltext) |
| C10 | DDPM-IP: perturb GT inputs to simulate inference errors | papers/related/2023_ning_input_perturbation.pdf | 1 | "consisting in perturbing the ground truth samples to simulate the inference time pre- diction errors." |
| C11 | DDPM-IP: Gaussian input perturbation at training time | papers/related/2023_ning_input_perturbation.pdf | 4 | "we explicitly model the prediction er- ror using a Gaussian input perturbation at training time." |
| C12 | CDM: augmentation level given as an extra time embedding, searched post hoc | papers/related/2021_ho_cascaded_diffusion.pdf | 18 | "amortized over the truncation time s by providing s as an extra time embedding input to the network (Section 2), allowing us to perform a more fine grained search over s without retraining the model." |
| C13 | DiffDock: independent per-factor diffusion in a product space | papers/related/2022_corso_diffdock.pdf | 6 | "it suffices to sample from the diffusion kernel and regress against its score in each group independently." |
| C14 | TD: score must be SE(3)-invariant, parity-equivariant (pseudoscalar) | papers/core/2022_jing_torsional_diffusion.pdf | 6 | "the score model must be invariant under SE(3) but equivari- ant (change sign) under parity inversion of the input point cloud-- i.e. it must output a set of pseudoscalars" |
| C15 | TD: the complete assignment guarantees no L shift | papers/core/2022_jing_torsional_diffusion.pdf | 19 | "The complete assignment resulting from the linear sum solution guarantees that there is no distributional shift in the local structures seen during training and inference." |
| C16 | TD: training on GT L hurts at test | papers/core/2022_jing_torsional_diffusion.pdf | 26 | "although the training and validation score matching loss of this model is significantly lower, its inference performance reflects the detrimental effect of the local structure distributional shift." |
| C17 | Torsional-GFN: an L-conditioned torsion model follows L shifts | papers/related/2025_ezzine_torsional_gfn.pdf | 4 | "Torsional-GFN is able to follow the shift of the modes in the energy landscape when sampling torsion angles for unseen local structures." |
| C18 | GEOM geometries reflect GFN2-xTB (shown for GEOM-Drugs; for QM9 UNVERIFIED) | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 10 | "Thus, the observed bond lengths reflect the energy landscape of GFN2-xTB" |
| C19 | MMFF disagrees with GFN2-xTB; ML models beat MMFF on alignment | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 10 | "current state-of-the-art generative models can now outperform MMFF precision on GEOM-Drugs in terms of alignment with GFN2-xTB." |
| C20 | TD: OMEGA's better L gives better QM9 results | papers/core/2022_jing_torsional_diffusion.pdf | 26 | "is only on par with or slightly worse than OMEGA, which, evidently, has a better local structures for these small molecules." |
