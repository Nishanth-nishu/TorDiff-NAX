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
- MMFF L is the best non-oracle L we have. It halves the bond/angle error (bonds 0.018 Å, angles 1.97°) and helps CTRL
  (0.1525 vs 0.1779), but B1 cannot use it (0.2323) [E-ROBUST-009]. A model trained on MMFF L could.
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
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
