# C-ROBUST-03: Sampler-matched mixture: train on the L you will deploy, with conformer-matched targets, mixed with true L

## 1. Idea (scout)
Generalise S3 from "RDKit-matched + true L" to "deployed-sampler-matched + true L". The first instance is S3-MMFF: a
50/50 per-sample mix of MMFF-relaxed, conformer-matched L and the paired true L, tested with MMFF L at inference. It
sets a design rule for any Q1 L source (xTB, learned refiner). Training L must come from that source, and the torsion
targets must be re-matched to it (conformer matching). The model must never be trained on generated L with
un-matched true torsions.

## 2. Where it comes from (scout)
- TD itself: training on true L causes a test-time shift that significantly hurts [E-ROBUST-020]. Conformer matching
  solves it as domain adaptation [E-ROBUST-022] and guarantees the same L distribution in training and inference
  [E-ROBUST-021].
- Cascades: a stage trained on ground truth fails on out-of-distribution upstream samples [E-ROBUST-024], and
  conditioning augmentation helps because it removes that train-test mismatch [E-ROBUST-023].
- Imitation learning: when the learner's own outputs set its inputs, the i.i.d. assumption fails [E-ROBUST-044]. DAgger
  fixes this by training on the states the learner induces, labelled by the expert [E-ROBUST-045].
- The naive alternative is scheduled sampling, which feeds generated inputs but keeps the original targets. Its
  objective is improper and gives an inconsistent learner [E-ROBUST-042].
- Domain randomisation: training on several sources makes the target look like one more variation [E-ROBUST-036].

## 3. Why it could matter here (scout)
- MMFF L is the best non-oracle L we have. Against raw ETKDG it cuts heavy-bond error by about 40% and heavy-angle error
  by about 30% (0.018 vs 0.031 Å; 1.97 vs 2.80°). It helps CTRL (0.1525 vs 0.1779), but B1 cannot use it (0.2323)
  [E-ROBUST-009]. Hypothesis (INFERENCE): a model trained on MMFF L could (revised after V2 card note; formerly
  "halves").
- B1 learns its targets on true L. CTRL learns on RDKit L with matched targets [E-ROBUST-016]. S3 shows that mixing the
  two keeps most of both (0.182 RDKit, 0.033 true L, 1 seed) [E-ROBUST-004, E-ROBUST-001]. S3's gain on RDKit L is in
  flexible molecules, where B1 fails [E-ROBUST-011].
- Conformer matching is the "expert relabelling" step: it moves the targets to the torsions that best fit the GT
  conformer given the sampler's L [E-ROBUST-022, E-ROBUST-016]. With it, training on generated L is DAgger-like
  [E-ROBUST-045], not scheduled-sampling-like [E-ROBUST-042]. INFERENCE.
- It is mostly plumbing. The S3 switch exists [E-ROBUST-014]. The paired builder takes any standardized-pickle
  directory, and MMFF-matched pickles already exist for S5/B3 [E-ROBUST-057].

## 4. Assumptions that may not transfer (scout)
- DAgger's guarantee concerns sequential decisions whose inputs depend on the learner. Here the L sampler does not
  depend on the torsion model (no feedback loop), so only the "train on the deployed input distribution" half applies
  [E-ROBUST-044].
- MMFF-relaxed matched conformers may pair with GT less cleanly than ETKDG ones. The pairing and pair_ok rates are
  UNVERIFIED [E-ROBUST-057].
- MMFF still leaves most of RDKit's ring tail: 20% of ring seeds > 10° [E-ROBUST-009]. S3-MMFF cannot fix ring error.
  It only lets the model use MMFF's better bonds and angles.
- Value depends on S5 (MMFF-matched training alone, round 2, pending). If S5 does not beat CTRL + MMFF L (0.152-0.154)
  [E-ROBUST-009, E-ROBUST-059], matched training on MMFF adds little, and the GT half only re-tests S3.
- The rule is general, but each new source needs a full re-standardisation of the training set. That is CPU-heavy for
  xTB (not estimated) and needs a trained refiner for learned L.

## 5. Minimal experiment (scout)
- **Gate (pre-declared).** Run if round-2 S5 (B3_match_mmff, 3 seeds, in-job `--pre_mmff` evaluation) is ≤ 0.150 Å
  AMR-R, i.e. at least 0.004 below CTRL_rematch + `--pre_mmff` (0.154) [E-ROBUST-059].
- **Build.** `tools/build_paired_pickles.py --std_dir standardized_pickles_mmff --out_dir standardized_pickles_paired_mmff`
  (CPU), then featurize a `cache_paired_mmff` and add a `paired_mmff` VARIANT to `slurm/ablation_train_array.sbatch`.
- **Arm.** S3-MMFF: `--l_mix_p_gt 0.5` on the MMFF pairs, 100 epochs, 2 seeds.
- **Controls.** S5 (MMFF-matched only), S3 (RDKit-matched/GT), CTRL_rematch with MMFF L, B1.
- **Test conditions.** Primary (non-oracle): MMFF L (`S1_etkdg2L_mmff` and the `--pre_mmff` path), RDKit L. Secondary
  (ORACLE): GT cycled, λ 0.5, A5ring.
- **Primary metric and direction.** AMR-R on MMFF L ≤ min(S5, CTRL + MMFF) (superiority vs CTRL + MMFF). AMR-R on GT L
  within 0.005 Å of S3. RDKit L: report only (the model was not trained for it).
- **Cost.** Pairing and featurisation on CPU (about 1-4 h on 32 cores, by analogy with round-2 pairing). Training 2 ×
  12-16 GPU-h; evaluation about 2 × 8 × 0.25 GPU-h. About 28-36 GPU-h.
- **Extension (not in the minimal run).** Three-source randomisation {GT, RDKit-matched, MMFF-matched}. It needs the two
  paired caches merged per conformer.

## 6. Scout's own call (scout)
Worth trying: YES (gated on S5). The principle is the best-supported one in this lens: TD's own fix, the cascade
literature and DAgger agree. It is the recipe every Q1 L source will need. MMFF is the cheapest real test of it. If the
S5 gate fails, downgrade to MAYBE and wait for a better L source from Q1.

## 7. Code grounding (grounder)
- **Touchpoints:** zero TD code.
  - `tools/build_paired_pickles.py --std_dir data/QM9/standardized_pickles_mmff --out_dir
    data/QM9/standardized_pickles_paired_mmff` (`:50-56`). Its pairing check recomputes the stored
    `AlignMol(RemoveHs(matched), RemoveHs(gt))` (`:88-94`). The MMFF variant stored exactly that, after `mmff_func`
    (`standardize_confs.py:83-87, :108`), and `geom_id` comes from the raw dicts, so pairing transfers unchanged
    [E-ROBUST-057].
  - Add `QM9_PAIRED_MMFF` (`common.sh:46`) and a `paired_mmff` case in `featurize_qm9.sbatch:29-39` /
    `ablation_train_array.sbatch:53-63`.
  - Training line `S3mmff_mixL_p0.5 | paired_mmff | --l_mix_p_gt 0.5 ... | --pre_mmff | 2 | s |`.
- **Transfer check:**
  - Matching as domain adaptation [E-ROBUST-021/022] exists in code: MMFF std pickles are matched after relaxation
    (`standardize_confs.py:83-107`).
  - The GT half is aligned onto the MMFF conformer (`build_paired_pickles.py:136-146`).
  - The pairing rate is UNVERIFIED (§4). The tool prints `PAIRING_CLEAN`/`PAIR_OK` (`:174-186`); require ≥ 98 %.
  - **Silent failure:** `ConformerDataset` reuses any cache at the given path (`dataset.py:124-129`), and the stale
    guard only checks `gt_pos` (`:141-144`). Use a new cache name and assert MMFF-level bond error in the RDKit half.
- **Gate:** its data now exist (B3 s0–s2 in-job `--pre_mmff`; CR ×3 `S5ctrl_pre_mmff`).
- **Verdict: IMPLEMENTABLE AS SPECIFIED** (config level).
- **Cost:** CPU pairing 1–4 h + featurize ~1 h; GPU ≈ 26.1.
Detail, line numbers and cluster checks: `round3/grounding.md` §3 (shared pieces S-1…S-6 in §2.3).
## 8. Predicted effect on our project (analyst)
Data: `round3/impact.md` §2–3; `cluster_sync/results/qm9_default/A2_pre_mmff/floor_run.log`. PRELIMINARY round-2
numbers.
- **Ceiling.**
  - On MMFF L the rigid stratum is fixed by L (0.128, model-free), so only flexible bins can move (CTRL on MMFF:
    0.132 / 0.172 / 0.190 / 0.224).
  - The torsion-oracle floor with MMFF L (A2) is a per-molecule mean of 0.097, against 0.143 observed.
  - Training-scheme effects measured so far: B2/B5 +0.002/+0.006 (worse); S3 vs CTRL on RDKit L +0.003.
- **Expected effect.** S3-MMFF vs CTRL + MMFF: 0 to −0.006.
  - Matched targets remove CTRL's train/test L shift, but CTRL already uses MMFF L about as well as the matched-MMFF
    control (0.1525 vs S5ctrl `--pre_mmff` 0.154).
  - The true-L half pays off only on reference-like L. B1 cannot exploit MMFF L (0.232), so no B1-type gain is
    expected here.
  - MMFF's residual is the pucker tail (impact.md §3c), which this card cannot fix (card §4).
  - P(S5 gate ≤ 0.150) ≈ 30 %.
- **Worst case.** +0.003 (S3-like cost) for 28–36 GPU-h.
- **Detectability.** −0.005 needs 3 seeds (CTRL seed SD 0.001; paired model contrast ±0.002).
- **Decision value.** The principle (train on the deployed sampler with matched targets) is the right follow-up for
  xTB L if C-GEOM-01 lands in branch (B), where B1 fails on xTB L. MMFF is the wrong test bed.
- **Worth trying: MAYBE → NO for now** (≤ 0.2 mÅ per GPU-h). Redirect to an xTB-matched S3 after C-GEOM-01.

**Revised after cross-read (2026-10-11).**
- The gate data exist (B3 s0–s2 in-job `--pre_mmff`, CR ×3 `S5ctrl`). Compute S5 ≤ 0.150 on CPU before the panel; my
  "P ≈ 30 %" is a prior.
- Cost ≈ 26.1 GPU-h.
- Leak note (impact.md §3g): the true-L half teaches the model to use own-conformer L, which MMFF L never provides.
  This supports the expected gain of ≤ 0.006.
- Call unchanged: **MAYBE → NO** unless the measured gate passes.

## 9. Panel (judges)
