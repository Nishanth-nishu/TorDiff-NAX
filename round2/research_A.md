# Round 2, research agent A: design-side ablations for learning L (QM9)

Written 2026-10-06 by research agent A. Scope: how prior work learns local structure L (bond lengths, angles, ring
puckers) together with torsions, and which minimal QM9 experiments test whether a learned or denoised L closes the gap
between R0 (AMR-R 0.176 A) and the GT-L oracle of the GT-trained model (B1 + GT L, AMR-R 0.0369 A).
Conventions follow `round2/BRIEF.md`. Paper pages are PDF pages; for every cited PDF the printed page number is the same
as the PDF page (checked from the page footers in the fulltext files; GO-Flow p. 7 has no printed number). Code
citations are `path:line`, relative to `C:\Users\HP\Desktop\tor_diff`.

New papers added this round (PDF, full text with form-feed page breaks, and a row in `papers/INDEX.md`):
- `papers/related/2021_ho_cascaded_diffusion.pdf` (Ho et al., Cascaded Diffusion Models, arXiv 2106.15282)
- `papers/related/2023_ning_input_perturbation.pdf` (Ning et al., Input Perturbation Reduces Exposure Bias, arXiv 2301.11706)

---

## 0. What the round-1 data already say about design (read first)

These numbers come from files that already exist. Nothing here is new compute.

**D1. Ring geometry dominates the QM9 RDKit-L floor; acyclic angles explain little.** I recomputed per-stratum means
from `cluster_sync/results/analysis/local_structure_test.csv` (7554 GT conformers without errors). Molecules are
grouped by smallest ring size (0 = acyclic).

| stratum (n GT confs) | floor_best_sym | + GT acyclic angles (angle_oracle) | + GT acyclic angles and bonds (local_oracle) | floor_gtother (perfect independent L sampler) | angle RMSE vs assigned seed (deg) |
|---|---|---|---|---|---|
| all (7554) | 0.0986 | 0.0885 | 0.0735 | 0.0269 | 3.88 |
| acyclic (2110) | 0.0510 | 0.0400 | **0.0081** | 0.0137 | 4.19 |
| any ring (5444) | 0.1171 | 0.1072 | 0.0989 | 0.0323 | 3.76 |
| smallest ring 3 (2819) | 0.0771 | 0.0649 | 0.0541 | 0.0221 | 3.76 |
| smallest ring 4 (1050) | 0.1054 | 0.0960 | 0.0879 | 0.0237 | 3.67 |
| smallest ring 5 (1115) | **0.2117** | 0.2044 | 0.1992 | 0.0388 | 3.60 |
| smallest ring 6 (328) | 0.1370 | 0.1317 | 0.1298 | 0.0699 | 3.92 |
| smallest ring >= 7 (132) | 0.2180 | 0.2190 | 0.2183 | 0.1760 | 5.49 |

Reading:
- In acyclic molecules, acyclic angles alone remove 0.011 A of floor. Angles plus bond lengths remove almost all of it
  (0.051 to 0.008 A).
- In ring molecules (72% of GT conformers), GT acyclic angles and bonds remove only 0.018 A of 0.117 A. GT ring
  geometry from another GT conformer (`floor_gtother`) removes most of it.
- The 5-membered-ring stratum is the worst (0.21 A floor vs 0.039 A with GT L).
- Caveat: the oracles are applied to the single lowest-transplant-cost seed (`tools/local_structure_analysis.py:285`,
  `:295-298`), while `floor_best_sym` is a minimum over the top-3 seeds (`:280`). The share removed by the oracles is
  therefore a conservative (low) estimate.
- Caveat: `set_acyclic_local` sets angles one at a time and cannot satisfy all 2n-3 constraints at a branching centre
  (`tools/local_structure_analysis.py:196-215`).

**Design consequence.** Candidate (c), a joint angle+torsion factor restricted to non-ring angles, can reach at most
about 0.01 A of the floor on QM9, and only about 0.025 A with acyclic bond lengths added. On QM9 the learned L must
include ring geometry, and bond lengths matter for acyclic molecules. That points to a Cartesian or full-L refiner, or
a ring factor, rather than an acyclic angle factor (see R2A-6, which is deferred).

**D2. MMFF relaxation barely moves the mean floor.** `local_structure_test_mmff.log`: `floor_best_sym` mean 0.0986 ->
0.0934, median 0.0606 -> 0.0462; angle RMSE 3.88 -> 2.44 deg; bond RMSE 0.036 -> 0.020 A. MMFF improves typical
molecules but not the ring-dominated tail.

**D3. Coupling (G3) is larger for the GT-trained model.** Two separate paired tables give:
- `paired_B1_vs_CTRLbase_gtL.md`: B1 + random GT L: AMR-R 0.0369.
- `paired_B1_vs_CTRLbase_gtLcycle.md`: B1 + own GT L (A1c-style): AMR-R 0.0212.

The difference is about -0.016 A. For the released model the same difference was -0.0037 A (BRIEF, A1c). This is
**not a paired test** (the two values come from different tables), so treat it as indicative. If it holds, coupling L to
τ (L|τ) becomes worth about 4x more once the torsion model is trained on accurate L. A two-stage pipeline with an
independent L sampler would leave this ~0.016 A on the table, and joint or iterative L|τ would recover it.

---

## 1. Proposed ablations

All arms use QM9 (TD split), 1000 test molecules, K = 2L, generation seed 0, heavy-atom symmetry-aware RMSD, and
`tools/paired_compare.py` (paired bootstrap, Holm on the primaries). **Primary endpoints:** AMR-R mean and COV-R@0.5.
Secondary: COV-R@0.05/0.1, AMR-P, geometry metrics (`tools/geometry_metrics.py`), and strata by ring size
(`tools/breakdown.py`).

Existing controls (no new training):
- CTRL_base (std pickles, 3 seeds, 100 epochs).
- B1 (raw GT pickles, 3 seeds, 100 epochs).

Cost basis: 10.7 GPU-h per 100-epoch training run and 0.25 GPU-h per inference+eval run (BRIEF).

### R2A-1: L-quality dose-response and partial-oracle gate (inference only) — priority 1

- **Hypothesis.** B1 beats CTRL once the test-time L error is below some threshold. The test-time L at which the curves
  cross (the break-even L quality) is the accuracy target for any learned or denoised L. The partial oracles show
  *which* L component (acyclic angles/bonds vs rings) unlocks B1's advantage, and so decide between an acyclic angle
  factor (c) and a ring-aware or Cartesian L model (b).
- **Gaps.** G2 (primary), G1.
- **Models.** CTRL_base s0-s2 and B1 s0-s2 (existing checkpoints).
- **Test-time L sources.** All are built as `--seed_confs` pickles keyed by raw SMILES, with 2L seeds or one seed per
  GT conformer; torsions are re-randomised by `perturb_seeds`.
  1. **RDKit-from-GT-graph control.** ETKDG seeds embedded from the GT graph, unmodified. This is the control for 4-5,
     which must use the same atom order. Round-1 G1 showed GT-graph seeding has no effect on the released model.
  2. **MMFF** (`--pre_mmff`, `generate_confs.py:18`, `:137`). A2 exists only for the released checkpoint, so it is
     needed for CTRL and B1.
  3. **GFN2-xTB-relaxed RDKit L** (via `torsional-diffusion/utils/xtb.py:44` `xtb_optimize`; needs an `xtb` binary on
     gnode118, conda-forge). GEOM geometries are GFN2-xTB optima (evidence E14 is for GEOM-Drugs; for GEOM-QM9 this is
     UNVERIFIED in our corpus). xTB relaxation is therefore the cheapest **non-oracle, non-learned** approximation of
     "GT-quality bond lengths and angles inside RDKit's basin".
  4. **Acyclic-GT partial oracle.** GT-graph ETKDG seed with acyclic heavy-atom angles and bonds set to those of a
     random GT conformer (reuse `tools/local_structure_analysis.py:196-215` `set_acyclic_local`, with_bonds=True); rings
     stay RDKit.
  5. **Ring-GT partial oracle.** A GT conformer whose acyclic angles and bonds are overwritten with an ETKDG seed's
     values (the same function with the roles swapped), so rings are GT and everything else is RDKit.
  6. **Interpolation series** λ ∈ {0, 0.5, 0.75}. For each GT conformer, take its DE-matched RDKit seed (the Hungarian +
     DE `floor_rmsd` path of `local_structure_analysis.py --mode test`; add a dump of the matched conformer, which is
     heavy-atom aligned to GT as in `standardize_confs.py:108`) and set x_λ = (1-λ)·x_matched + λ·x_GT. λ = 1 is the
     existing A1 (GT L) run; λ = 0 is the one-seed-per-GT-conformer RDKit control for this series.
- **Runs.** 8 new sources (1, 2, 3, 4, 5, λ = 0, 0.5, 0.75) × 2 models × 3 training seeds = 48 inference runs.
- **Control arm.** The same model on the RDKit-from-GT-graph control (source 1) for sources 2-5; λ = 0 for the series.
  The B1 vs CTRL contrast is computed within each source.
- **Primary metric and expected direction.** ΔAMR-R(B1 - CTRL) per source: positive (B1 worse) at RDKit, negative at
  GT L (round 1: +0.054 and -0.047). Expected crossing between λ = 0.5 and 0.75. Expected: xTB L negative or near zero,
  MMFF still positive (D2), ring-GT oracle closer to the GT-L value than the acyclic-GT oracle (D1).
- **Decision rules.**
  - If acyclic-GT recovers less than a third of the GT-L gain, drop (c) for QM9.
  - If xTB-relaxed L already gives B1 < CTRL, "physics-relaxed L + GT-trained TD" becomes the round-3 baseline to beat,
    and R2A-5's refiner must beat xTB.
- **Evidence.** E1, E2, E3, E13, E14, E19, E22, E28 (table in §3).
- **Code touchpoints.**
  - `torsional-diffusion/generate_confs.py:22`, `:75-82`, `:119-134` (`--seed_confs` path).
  - `torsional-diffusion/diffusion/sampling.py:75` (random.choice over seeds), `:94-102` (torsion randomisation).
  - `tools/make_seed_pickles.py:53-77` (seed-pickle writer to extend).
  - `tools/local_structure_analysis.py:163`, `:196-215`, `:285-298`.
  - `torsional-diffusion/utils/xtb.py:44`.
  - No change to TD itself.
- **Cost.** 48 × 0.25 = **12 GPU-h**. CPU: DE matching about 3-10 h on 32 cores (round-1 estimate for `--mode test`).
  xTB: about 15k optimisations of molecules with 9 or fewer heavy atoms; runtime not measured, but expected well under 1 h
  on 32 cores. Could be cut to 1 training seed per model for the λ series (−12 runs).

### R2A-2: λ-conditioned L-interpolation training (conditioning augmentation, CDM-style) — priority 2

- **Hypothesis.** One torsion model trained on a continuum of local structures between the conformer-matched RDKit L
  and the GT L, and told the L quality λ as an extra embedding, matches CTRL on RDKit L (λ = 0), matches B1 on GT L
  (λ = 1), and beats both on intermediate L such as xTB, MMFF or a learned L. This is the cheapest precursor of joint
  diffusion: L gets its own "time" λ, as in a DiffDock-style product space, but is not yet denoised.
- **Gaps.** G2, G1.
- **Setup.**
  - Re-run conformer matching (VARIANT=rematch) with one change: keep the GT positions next to the matched RDKit
    conformer. Store `conf['gt_pos']` before `conf['rd_mol']` is overwritten (`torsional-diffusion/standardize_confs.py:116`).
    Atom order is shared because `mol_rdkit` is copied from GT conformer 0 (`:73`), and the matched conformer is already
    heavy-atom aligned to GT by `AllChem.AlignMol` (`:108`).
  - In `TorsionNoiseTransform` (`torsional-diffusion/utils/dataset.py:24-43`):
    1. pick a conformer index (`:29`);
    2. draw λ from the mixture 25% λ = 0, 25% λ = 1, 50% U(0,1);
    3. set pos = (1-λ)·x_rdkit + λ·x_GT;
    4. then apply the torsion noise exactly as now (`:37-42`).
  - Put `data.node_lambda` next to `node_sigma` (`:38`). In `torsional-diffusion/diffusion/score_model.py:188-193`,
    concatenate a second sinusoidal embedding of λ to node_attr/edge_attr. Widen the input layers by sigma_embed_dim
    (`:65-75`).
  - At inference, add a `--l_level` CLI argument (generate_confs) and set `node_lambda` with `node_sigma` in
    `torsional-diffusion/diffusion/sampling.py:171`.
  - 100 epochs, **3 training seeds**.
- **Test-time L.** RDKit (λ = 0), GT (λ = 1), plus the xTB and MMFF L of R2A-1 with λ chosen on 200 **validation**
  molecules (CDM's post-hoc search, E6). Pre-register λ ∈ {0, 0.25, 0.5}. Do not tune λ on the test set.
- **Control arms.** CTRL_base on RDKit L (non-inferiority, margin +0.005 A). B1 on GT L (non-inferiority, margin
  +0.005 A). CTRL_base and B1 on xTB L (superiority).
- **Primary metric and expected direction.** AMR-R. Expected: ≤ CTRL + 0.005 on RDKit L; ≤ B1 + 0.005 on GT L; lower
  than both on xTB L.
- **Risk.** λ = 0 targets are conformer-matched (non-physical torsion targets). Mixing them with GT targets may blur the
  score. R2A-3 isolates this.
- **Evidence.** E4-E9, E15, E16, E24.
- **Code touchpoints.** As above. Also `torsional-diffusion/utils/parsing.py:25-29` (new args) and
  `torsional-diffusion/utils/dataset.py:200-241` (`featurize_mol` must carry `gt_pos`, paired by index with `pos`).
  Estimated about 60 lines changed. The new score-model input dimension means checkpoints are not compatible with
  CTRL/B1 (`strict=True` load, `generate_confs.py:102`).
- **Cost.** Standardisation CPU about 1-4 h on 32 cores (round-1 estimate). Training 3 × 10.7 = 32 GPU-h. Evaluation
  (4 L sources × 3 seeds, plus about 6 validation λ-search runs) about 4.5 GPU-h. **About 37 GPU-h.**

### R2A-3: Discrete mixture of RDKit-L and GT-L training data (candidate d) — priority 3

- **Hypothesis.** Without any conditioning, a 50/50 mixture of matched-RDKit and GT conformers gives a model that is
  no worse than CTRL on RDKit L and much better than CTRL on GT or near-GT L. If it costs much on either end, the λ
  input of R2A-2 is necessary (the CDM "amortised" result, E6).
- **Gaps.** G2.
- **Setup.** Same paired pickles as R2A-2. λ ~ Bernoulli(0.5) in the transform. No λ embedding, so the architecture is
  unchanged and inference is the standard `generate_confs.py`. 100 epochs, **2 seeds**.
  - Fallback if the paired standardisation slips: load the std pickle and the raw pickle for the same SMILES inside
    `filter_smiles` (`torsional-diffusion/utils/dataset.py:126-184`; raw-path logic `:143-148`) and concatenate both
    conformer lists.
  - Note: B1 uses all raw GEOM conformers (`featurize_mol` loops over every conformer, `:209`), while std pickles are
    capped by `confs_per_mol` in standardisation (`standardize_confs.py:59-67`). Use the same cap for both halves.
- **Test-time L.** RDKit, GT, xTB, MMFF.
- **Control arms.** CTRL_base (RDKit L), B1 (GT L), and R2A-2 (conditioning on vs off).
- **Primary metric and expected direction.** AMR-R. Expected: between CTRL and B1 on both ends (a small loss on each),
  and better than both on xTB L.
- **Evidence.** E5-E7, E16.
- **Code touchpoints.** `torsional-diffusion/utils/dataset.py:24-43`, `:126-184`, `:200-241`.
- **Cost.** 2 × 10.7 + 4 × 2 × 0.25 = **about 23 GPU-h**.

### R2A-4: L-noise augmentation of the GT-trained torsion model (candidate a) — priority 4

- **Hypothesis.** Training on GT L with isotropic Cartesian jitter (σ_L ~ logU[0.005, 0.05] A on all atoms, applied
  before the torsion noise; the score target `edge_rotate` is unchanged) makes the torsion score robust to L error. This
  is input perturbation / Gaussian conditioning augmentation (E5, E8-E10).
  - Gaussian jitter is not RDKit's systematic error: σ = 0.025 A gives bond errors of about 0.035 A (close to RDKit's
    0.036 A) but angle errors of only about 1-2 deg (my estimate, not measured). So this arm mainly tests robustness to
    the **residual error of a learned L denoiser** (which looks Gaussian-like), not to raw RDKit L.
  - It is the cheapest arm to implement and needs no pairing.
- **Gaps.** G2.
- **Setup.** Raw GT pickles (as B1). In `torsional-diffusion/utils/dataset.py:41`, add
  `data.pos = data.pos + σ_L·randn` before `modify_conformer`. Add `--l_jitter_max` in
  `torsional-diffusion/utils/parsing.py:25-29`. 100 epochs, **2 seeds**. Optional variant: also feed σ_L as an
  embedding, as in R2A-2.
- **Test-time L.** RDKit, GT, xTB, MMFF, and GT + Gaussian jitter at σ ∈ {0.02, 0.04} A (seed pickles built from GT
  conformers, CPU).
- **Control arm.** B1 (same data, no jitter).
- **Primary metric and expected direction.** AMR-R. Expected: worse than B1 on clean GT L by at most 0.01; better than B1
  on jittered GT and xTB L; probably still worse than CTRL on raw RDKit L, because the RDKit error is systematic, not
  isotropic.
- **Evidence.** E5, E8-E10, E25.
- **Code touchpoints.** `torsional-diffusion/utils/dataset.py:37-42`, `torsional-diffusion/utils/parsing.py:25-29`.
  About 5 lines.
- **Cost.** 2 × 10.7 + 6 × 2 × 0.25 = **about 24 GPU-h**.

### R2A-5: Two-stage pipeline with a learned Cartesian L-refiner (candidate b; pilot) — priority 5

- **Hypothesis.** A small equivariant denoiser trained on GT conformers at **low Cartesian noise**, started from the
  ETKDG seed mid-trajectory (FM-refiner/EBD style), produces L whose quality lies past the R2A-1 break-even point.
  Feeding that L to B1 / R2A-2 / R2A-3 then beats R0 and CTRL.
  - This is the direct test of "a learned L closes the gap".
  - It handles rings and bond lengths, which D1 shows are where the QM9 floor is.
  - It also gives a learned L|τ refinement *after* torsion sampling (a learned A3, which targets the D3 coupling term).
- **Gaps.** G1, G2, G3 (post-hoc variant).
- **Setup.**
  - Reuse the TD e3nn trunk (`torsional-diffusion/diffusion/score_model.py:79-106`; the irreps already carry 1o
    features).
  - Replace the bond head (`:113-128`, `:141-152`) with a per-atom TensorProductConvLayer to `1x1o` (a vector score
    per atom).
  - Training: GT conformers (raw pickles) with Gaussian VE noise σ ~ logU[0.005, 0.3] A on all atoms, plus a uniform
    torsion perturbation so the refiner sees non-GT torsions. Loss: denoising score matching, replacing
    `torus.score`/`score_norm` in `torsional-diffusion/utils/training.py:19-25` with the Gaussian score -ε/σ.
  - Inference: ETKDG seed → 20 reverse-SDE steps from σ_start ∈ {0.05, 0.1, 0.2} A (chosen on validation) to σ = 0.005
    → keep L, re-randomise torsions (`torsional-diffusion/diffusion/sampling.py:94-102`) → TD model.
  - Post-hoc variant: TD sample → refiner from σ_start = 0.05 A.
  - **1 seed**, 100 epochs (a pilot).
- **Evaluation.**
  1. L error of refined seeds vs GT (bond/angle RMSE) and their floor (`local_structure_analysis.py --mode test` on the
     refined seeds, CPU). This is the gate, compared against the RDKit and xTB floors of R2A-1.
  2. AMR-R of {CTRL, B1, R2A-2 best, R2A-3} on refined L.
- **Control arms.** The same TD models on RDKit L and on xTB L (R2A-1).
- **Primary metric and expected direction.** AMR-R of B1 (or R2A-2) on refined L: lower than CTRL on RDKit L (0.18).
  Stretch: below CTRL on GT L (0.084).
- **Evidence.** E11, E13, E17, E18, E20, E21, E27.
- **Code touchpoints.** New head in `torsional-diffusion/diffusion/score_model.py` (about 40 lines); new Cartesian noise
  transform modelled on `torsional-diffusion/utils/dataset.py:24-43`; loss in `torsional-diffusion/utils/training.py:7-34`;
  new refine function in `torsional-diffusion/diffusion/sampling.py` (modelled on `:105-189`); seeds handed over via the
  `--seed_confs` pickle path (`torsional-diffusion/generate_confs.py:75-82`), so TD inference is unchanged. About
  150-250 lines. This is the only arm with real development risk (about 1-2 days of coding).
- **Cost.** 1 × 10.7 (training; the denoiser is the same size as TD; not measured) + about 12 inference runs × 0.25 = **about
  14 GPU-h**. A second seed (+10.7) only if the gate passes.

### R2A-6: Joint wrapped-normal angle+torsion diffusion on non-ring angles (candidate c) — DEFERRED, gated

- **Hypothesis.** A FoldingDiff/RINGER-style joint factor over acyclic bond angles (B^k in `papers/gap_synthesis.md`
  §4) closes part of the L gap.
- **Why deferred.**
  - D1 caps its reachable effect on the QM9 floor at about 0.01 A (about 0.025 A with acyclic bond lengths).
  - It needs a parity-invariant angle head, a Jacobian update for angles, and joint training: an estimated 300+ lines
    adapting `torsional-diffusion/utils/torsion.py:57-75` (angle analogue of `modify_conformer`: rotate the branch about
    the normal of the angle plane) and `torsional-diffusion/diffusion/score_model.py:113-161`.
  - RINGER shows that ring angles in internal coordinates need a ring-closure post-processing step (E12), so extending
    it to rings is not cheap.
- **Gate to run in round 3.** R2A-1 source 4 (acyclic-GT oracle) recovers at least a third of B1's GT-L gain. If instead
  the ring-GT oracle carries the gain, the round-3 factor should be a ring factor (PuckerFlow Cremer-Pople, E13/E23) or
  the R2A-5 refiner.
- **Gaps.** G1, G3. **Cost if run.** 3 × 10.7 + dev = about 33 GPU-h.

### R2A-7 (optional, if budget remains): consistent better fixed L sampler (B3-MMFF, existing deferred line) — priority 6

- **Hypothesis.** Conformer matching on MMFF-relaxed seeds plus test `--pre_mmff` beats CTRL by more than A2 (-0.026 on
  the released model), because train and test L are then consistent.
- **Role.** The non-learned baseline that learned-L arms must beat. If R2A-1 shows xTB L is much better than MMFF L, an
  xTB version is the stronger baseline, but xTB-relaxing about 1M training conformers is CPU-heavy (not estimated
  reliably), so run MMFF first.
- **Setup.** `slurm/ablations_train_deferred.tsv` B3 line (VARIANT=mmff, `standardize_confs.py:83-87`). **2 seeds.**
- **Control.** CTRL_rematch with `--pre_mmff`. **Metric.** AMR-R, expected lower.
- **Evidence.** E2, E3. **Cost.** 2 × 10.7 + 4 × 0.25 = about 22 GPU-h.

---

## 2. Ranked shortlist (at most 6) and budget

| rank | ID | type | training runs | GPU-h | answers |
|---|---|---|---|---|---|
| 1 | R2A-1 dose-response + partial oracles + xTB/MMFF L, on existing CTRL and B1 | inference only | 0 | 12 | How good must L be? Which L component matters (acyclic vs ring)? Gates (b) vs (c) |
| 2 | R2A-2 λ-conditioned RDKit↔GT interpolation training | train | 3 | 37 | Can one torsion model work across L quality (precursor of joint diffusion)? |
| 3 | R2A-3 discrete RDKit/GT mixture (d) | train | 2 | 23 | Is conditioning on L quality needed? |
| 4 | R2A-4 Gaussian L-jitter augmentation (a) | train | 2 | 24 | Robustness to residual (denoiser-like) L error |
| 5 | R2A-5 Cartesian L-refiner pilot → two-stage (b) | dev + train | 1 | 14 | Does a learned L reach the break-even quality? |
| 6 | R2A-7 B3-MMFF consistent fixed-L baseline | train | 2 | 22 | Non-learned baseline for learned L |

- **Totals.** 10 training runs, about 132 GPU-h, which is about 33 h wall time on 4 GPUs. That leaves about 6-8 training
  slots of the ~16-18-run round for other agents' arms.
- **Ordering.** Launch R2A-2/R2A-3/R2A-4 training first: they have no dependency on R2A-1. Run R2A-1 inference on GPU
  gaps while they train. Start R2A-5 coding immediately; train it on day 2.
- **Deferred.** R2A-6 (c), until the R2A-1 gate.

---

## 3. Evidence and citation table

Exact quotes are copied from the fulltext files. Ellipses are not used inside quotes; line-break hyphens are kept as in
the text.

| # | Claim | PDF path | Page | Exact quote |
|---|---|---|---|---|
| E1 | TD: training on GT L causes a test-time shift that hurts | papers/core/2022_jing_torsional_diffusion.pdf | 7 | "there will be a distributional shift at test time, where only approximate local structures from p^G(L) are available. We found that this shift significantly hurts performance." |
| E2 | TD: the GT-L-trained model has lower loss but worse inference (our B1) | papers/core/2022_jing_torsional_diffusion.pdf | 26 | "although the training and validation score matching loss of this model is significantly lower, its inference performance reflects the detrimental effect of the local structure distributional shift." |
| E3 | TD: on QM9, a better L sampler (OMEGA) matters | papers/core/2022_jing_torsional_diffusion.pdf | 26 | "is only on par with or slightly worse than OMEGA, which, evidently, has a better local structures for these small molecules." |
| E4 | TD names relaxing rigid L as future work | papers/core/2022_jing_torsional_diffusion.pdf | 22 | "We leave to future work the exploration of relaxations of the rigid local structures assumption" |
| E5 | CDM: a cascade fails from train-test mismatch, fixed by conditioning augmentation | papers/related/2021_ho_cascaded_diffusion.pdf | 3 | "We empirically find that conditioning augmentation is effective because it alleviates compounding error in cascading pipelines due to train-test mismatch" |
| E6 | CDM: amortise over the augmentation level and choose it after training | papers/related/2021_ho_cascaded_diffusion.pdf | 18 | "amortized over the truncation time s by providing s as an extra time embedding input to the network (Section 2), allowing us to perform a more fine grained search over s without retraining the model." |
| E7 | CDM: the mismatch arises when stage-1 samples are out of distribution for a stage-2 model trained on GT (the exact B1 pattern) | papers/related/2021_ho_cascaded_diffusion.pdf | 18 | "This occurs when low- resolution model samples are out of distribution compared to the ground truth data on which the super-resolution model is trained." |
| E8 | CDM: Gaussian noise is the effective augmentation | papers/related/2021_ho_cascaded_diffusion.pdf | 6 | "what we found most effective at low resolutions is adding Gaussian noise (forward process noise)" |
| E9 | DDPM-IP: perturb GT inputs during training to simulate inference errors | papers/related/2023_ning_input_perturbation.pdf | 1 | "consisting in perturbing the ground truth samples to simulate the inference time pre- diction errors." |
| E10 | DDPM-IP: model the prediction error with Gaussian input perturbation | papers/related/2023_ning_input_perturbation.pdf | 4 | "we explicitly model the prediction er- ror using a Gaussian input perturbation at training time." |
| E11 | EBD: a two-stage model that corrects the cheminformatics prior instead of freezing it | papers/related/2024_park_equivariant_blurring_diffusion.pdf | 1 | "generation of fine atomic details from the coarse-grained approximated structure while allowing the latter to be adjusted simultaneously." |
| E12 | RINGER: internal-coordinate reconstruction of rings needs a ring-closure step | papers/related/2023_grambow_ringer.pdf | 5 | "Adopting a sequential reconstruction method such as NeRF accumulates small errors that result in inadequate ring closure for macrocycles." |
| E13 | PuckerFlow already feeds learned ring L into a pretrained TD (two-stage), with no retraining on that L | papers/related/2026_schaufelberger_puckerflow.pdf | 21 | "we sample the ring structures using PuckerFlow, and use these as local substructures for a pretrained model of torsional diffusion using publicly available model weights" |
| E14 | GEOM geometries reflect GFN2-xTB (shown for GEOM-Drugs; for QM9 UNVERIFIED) | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 10 | "Thus, the observed bond lengths reflect the energy landscape of GFN2-xTB" |
| E15 | DiffDock: product-space diffusion proceeds independently per factor (an L-level is a second time variable) | papers/related/2022_corso_diffdock.pdf | 6 | "it suffices to sample from the diffusion kernel and regress against its score in each group independently." |
| E16 | Torsional-GFN: a torsion model conditioned on L generalises to unseen L | papers/related/2025_ezzine_torsional_gfn.pdf | 4 | "Torsional-GFN is able to follow the shift of the modes in the energy landscape when sampling torsion angles for unseen local structures." |
| E17 | FM-refiner: start from upstream samples instead of noise | papers/related/2025_xu_fm_refiner.pdf | 3 | "instead of starting from pure noise, we initialize sampling from upstream-generated conformers x^1, thereby skipping the inherently hard-to-learn high-noise phase." |
| E18 | ET-Flow: Cartesian models learn QM9 L well (AMR-R 0.073) | papers/related/2024_hassan_etflow.pdf | 19 | "ET-Flow (QM9 RS) 96.47 100.00 0.073 0.047 94.05 100.00 0.098 0.039" |
| E19 | Corso thesis App. C (same text as TD App. F.1): RDKit angles and lengths look accurate marginally (4.1 deg, 0.03 A), yet the floor exists | papers/related/2023_corso_intrinsic_diffusion.pdf | 90 | "with a RMSE of 0.03 Å for bond lengths and 4.1° for bond angles on GEOM-DRUGS." (the Å and ° glyphs are garbled in the fulltext file) |
| E20 | GO-Flow: modelling internal coordinates is the most important component | papers/related/2026_liu_goflow.pdf | 7 | "that modeling internal coordinates (bond lengths, angles, and tor- sions) via entropic optimal transport is the most critical factor for generating high-fidelity molecular structures." |
| E21 | GO-Flow ablation size (DRUGS, GD-P) | papers/related/2026_liu_goflow.pdf | 7 | "the w/o C variant achieves a COV-R of only 90.26% and a MAT-R of 0.8512 Å, which is sub- stantially inferior to the full GO-Flow model (94.82% and 0.7971 Å" (Å garbled in the fulltext) |
| E22 | TD: rings are delegated to the L sampler | papers/core/2022_jing_torsional_diffusion.pdf | 23 | "Therefore, torsional diffusion relies on the local structure sampler pG(L) to accurately model cycle conformations." |
| E23 | Corso thesis: a ring-pucker factor on hyperspheres | papers/related/2023_corso_intrinsic_diffusion.pdf | 63 | "employing the ring puckering coordinates [18] to model the flexibility of ring conformations as points on hyperspheres." |
| E24 | Corso thesis: the manifold as a soft constraint (the direction of joint L) | papers/related/2023_corso_intrinsic_diffusion.pdf | 64 | "using such manifold as a soft constraint or inductive bias" |
| E25 | Corso thesis: a model trained on GT structures can tolerate inaccurate input structures (DiffDock) | papers/related/2023_corso_intrinsic_diffusion.pdf | 63 | "results have shown that DiffDock is robust to inaccuracies in the structure" |
| E26 | FoldingDiff: wrapped-normal noise on angles and dihedrals | papers/related/2022_wu_foldingdiff.pdf | 5 | "to sample from a wrapped normal instead of a standard normal (Jing et al., 2022)" |
| E27 | GeoMol: predicts local structure from the graph first | papers/related/2021_ganea_geomol.pdf | 4 | "First, we predict the local 3D structure of each non-terminal atom, which we deem local structure (LS)" |
| E28 | TD: GT-other-L floor vs RDKit L (DRUGS) | papers/core/2022_jing_torsional_diffusion.pdf | 21 | "it is only slightly larger than the average RMSDmin of 0.284 Å resulting from matching a ground truth conformer to the local structure of another randomly chosen ground truth conformer" (Å garbled in the fulltext) |

E-numbers used per arm:
- R2A-1: E1, E2, E3, E13, E14, E19, E22, E28.
- R2A-2: E4-E9, E15, E16, E24.
- R2A-3: E5-E7, E16.
- R2A-4: E5, E8-E10, E25.
- R2A-5: E11, E13, E17, E18, E20, E21, E27.
- R2A-6: E12, E20, E23, E26.
- R2A-7: E2, E3.

## 4. Data and code citations (round-1 files)

| Claim | Source |
|---|---|
| Stratified floors (D1) | `cluster_sync/results/analysis/local_structure_test.csv` (my groupby over `min_ring_size`); overall means in `local_structure_test.log` |
| MMFF floors and angle/bond RMSE (D2) | `cluster_sync/results/analysis/local_structure_test_mmff.log` |
| B1 + GT L 0.0369, CTRL + GT L 0.0842 | `cluster_sync/results/analysis/paired_B1_vs_CTRLbase_gtL.md` |
| B1 + own GT L 0.0212, CTRL 0.0820 (D3) | `cluster_sync/results/analysis/paired_B1_vs_CTRLbase_gtLcycle.md` |
| B1 on RDKit L 0.2342 vs CTRL 0.1806 | `cluster_sync/results/analysis/paired_B1_vs_CTRLbase.md` |
| Oracles use the single best seed; floor_best uses top-m | `tools/local_structure_analysis.py:280`, `:285`, `:295-298` |
| Matched RDKit conformer is aligned to GT; GT atom order shared | `torsional-diffusion/standardize_confs.py:73`, `:108`, `:116` |
| Torsion noise and conformer choice in training | `torsional-diffusion/utils/dataset.py:29`, `:37-42` |
| σ embedding (where λ would go) | `torsional-diffusion/diffusion/score_model.py:188-193` |
