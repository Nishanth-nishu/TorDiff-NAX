# C-ROBUST-01: Error-matched, factorised L augmentation (RDKit error transplanted ring-wise and acyclic-wise)

## 1. Idea (scout)
Replace isotropic jitter (S2) with training L that carries RDKit's *measured* error, one component at a time. For each
training sample, draw L from four paired versions of the same conformer: true L; RDKit-matched L (CTRL data); true
rings + RDKit-matched acyclic geometry; RDKit-matched rings + true acyclic geometry. The two hybrids are built offline
from the existing paired cache with the A5 hybrid builder. No conditioning input. The test is whether one model keeps
S3's two endpoints *and* exploits partly correct L (good rings only, or good bonds/angles only), which is what any
round-3 L source (Q1 refiner, Q3 ring component, MMFF/xTB relaxation) will deliver.

## 2. Where it comes from (scout)
- Cascaded diffusion: conditioning augmentation works because it removes the train-test mismatch between a stage
  trained on ground truth and the samples it receives [E-ROBUST-023]. The augmentation had to match the upstream error:
  Gaussian noise helped at low resolution, but the same augmentation gave negative results at 128×128 and 256×256,
  where blur was used instead [E-ROBUST-025, E-ROBUST-056]. CDM mixed clean and augmented inputs 50/50 in training
  [E-ROBUST-026].
- Input perturbation (DDPM-IP) simulates inference-time errors during training [E-ROBUST-029] and acts as a
  regulariser that smooths the prediction function [E-ROBUST-030].
- For molecules, SliDe argues that isotropic coordinate noise is an inappropriate assumption [E-ROBUST-034] and
  perturbs bond lengths, angles and torsions separately with physically set variances [E-ROBUST-035].
- Domain randomisation: with enough variability in training, the target looks like one more variation
  [E-ROBUST-036]. Practitioners fit the randomisation to the real data [E-ROBUST-053] and avoid overly wide
  distributions that hinder learning [E-ROBUST-052].
- Score models are inaccurate where training data are sparse [E-ROBUST-050]. Larger noise covers more of that region
  but corrupts the data more [E-ROBUST-051].
- ProteinMPNN: 0.02 Å backbone noise improved recovery on AlphaFold models but lowered it on exact PDB structures
  [E-ROBUST-031], because exact coordinates carry information about the target that predicted structures lack
  [E-ROBUST-032].

## 3. Why it could matter here (scout)
- Isotropic noise half-fixes B1. B1 scores 0.236 on RDKit L and 0.020 on true L; S2 (σ = 0.04 Å/axis) scores 0.206 and
  0.054 [E-ROBUST-001, E-ROBUST-002, E-ROBUST-003]. Noise buys some robustness and costs most of the true-L gain.
- Isotropic noise does not look like RDKit's error. At 0.04 Å/axis it gives 1.75× RDKit's bond error and 1.18× its
  angle error, but almost none of RDKit's wrong ring conformations: 0.34% of ring seeds have endocyclic-dihedral error
  > 10° under noise, against 28.6% for RDKit [E-ROBUST-006]. For RDKit, that share is 48-92% for smallest rings of 4-7
  atoms [E-ROBUST-007].
- RDKit-derived error hurts B1 more than larger random error. GT rings with RDKit-matched acyclic geometry (bonds 0.027 Å,
  angles 3.5°, no ring error) give 0.188. GT + 0.04 Å noise (bonds 0.055 Å, angles 3.3°, some ring error) gives 0.154
  (unpaired, different molecule sets) [E-ROBUST-008].
- The two trained models use different parts of L. CTRL gains from true rings (A5ring 0.119), not from true acyclic
  geometry (0.182); B1 gains more from true acyclic geometry (0.158) than from true rings (0.188) [E-ROBUST-010]. A model
  that has seen each component wrong on its own is the natural way to get both gains. This matters most for a ring
  component (Q3): with CTRL, true rings alone take AMR-R from 0.178 on RDKit L to 0.119 (unpaired, 936 vs 955
  molecules) [E-ROBUST-001, E-ROBUST-010].
- Mixing in real RDKit error already works. B1's loss on RDKit L is in flexible molecules (≥ 4 rotatable bonds: B1
  0.394, CTRL 0.239), and S3's 50/50 mix brings it back to 0.253 [E-ROBUST-011]. The open question is partial L, and the
  round-2 panel never evaluates S2 or S3 on the A5 hybrids [E-ROBUST-013].
- It reuses existing parts. The per-sample switch and the paired cache exist [E-ROBUST-014]. The hybrid builder exists
  and runs at about 2 ms per call in a local timing (two hybrids per training conformer) [E-ROBUST-015].

## 4. Assumptions that may not transfer (scout)
- Every source is an analogy (images, pre-training, proteins). None trains a conditional model on L with a systematic,
  ring-heavy error [E-ROBUST-025, E-ROBUST-034, E-ROBUST-031].
- The transplant reproduces ETKDG's error only. A learned L source (Q1) will have a different error (smaller bias,
  different ring failures). Mitigation: MMFF L serves as a held-out "other sampler" test, and an optional small
  isotropic jitter can be added on top. Robustness to an unseen sampler is not guaranteed.
- The hybrid builder is approximate: A5acyc seeds keep a 0.58° residual in acyclic angles [E-ROBUST-019]. Training
  hybrids would inherit that.
- Hybrids keep the torsions of their base geometry (GT torsions for the GT-ring hybrid, matched torsions for the
  RDKit-ring hybrid). The TD target is relative to the selected conformer [E-ROBUST-016], so this is consistent, but
  angle setting moves fragments and may shift torsions slightly. The grounder should check this.
- Four-way mixing gives true L only 25% of samples (S3: 50%). That may cost true-L accuracy (S3 0.033 vs B1 0.020)
  [E-ROBUST-004].
- Part of S2's residual gap may be a train/test asymmetry rather than the wrong error type: S2 only ever saw jittered L,
  and was tested on un-jittered L. CDM's non-truncated variant applies the same corruption at sampling time
  [E-ROBUST-028]. An optional Arm 0b (S2 on RDKit L + 0.04 Å/axis jitter) separates the two, but it needs a new seed
  pickle.
- Leak caveat: all `_cyc_ORACLE` test conditions use each molecule's own GT conformers, so any information the true
  acyclic geometry carries about its own torsions (the ProteinMPNN-type "memory", [E-ROBUST-032]) inflates A5acyc and λ
  results. RDKit and MMFF L (non-oracle) are therefore the primary test conditions.

## 5. Minimal experiment (scout)
- **Arm 0 (gate, inference only).** S3 s0/s1 and S2 s0-s2 on `S1_A5ring_cyc_ORACLE`, `S1_A5acyc_cyc_ORACLE` and
  `S1_lam0.75_cyc_ORACLE` (existing seed pickles, existing sets in `slurm/r2_evalsets.tsv`): 15 runs, about 2-4 GPU-h.
  Gate rule (pre-declared): if S3 scores ≤ 0.125 on A5ring (CTRL 0.119) **and** ≤ 0.165 on A5acyc (B1 0.158), S3
  already handles partial L; stop here and report. Otherwise run Arm 1.
- **Arm 1 (train): S3-factorised.** Per sample, uniform over {GT, RDKit-matched, GT ring + RDKit acyclic, RDKit ring +
  GT acyclic}. Hybrids are precomputed offline per paired conformer (CPU, estimated < 1 h on 32 cores at about 2 ms per
  call; conformer count not checked), stored next to `gt_pos_aligned`, and selected in `_select_paired_L`. 100 epochs,
  2 seeds (3 if budget allows).
- **Controls.** S3 (2 seeds, round 2), CTRL_rematch (3), B1 (3), S2 (3); all on the same panel.
- **Test conditions.** Primary (non-oracle): RDKit ETKDG (`S1_etkdg2L`), MMFF (`S1_etkdg2L_mmff`). Secondary (ORACLE):
  GT cycled, A5ring, A5acyc, λ 0.25/0.5/0.75, noise 0.02.
- **Primary metrics and direction.** AMR-R on RDKit L: non-inferior to CTRL_rematch (margin +0.005 Å). AMR-R on A5ring:
  lower than S3 (expected ≤ 0.12). Secondary: A5acyc ≤ 0.16; GT ≤ 0.04; COV-R@0.1. Paired bootstrap on common molecules
  (`tools/paired_compare.py`).
- **Cost.** Arm 0: about 3 GPU-h. Arm 1: 2 × 12-16 GPU-h training plus 2 × 10 conditions × 0.25 GPU-h, so about 30-37
  GPU-h. Total about 33-40 GPU-h.

## 6. Scout's own call (scout)
Worth trying: YES (gated by Arm 0). This is the only candidate whose augmentation matches RDKit's measured error
structure. It reuses existing code, and it directly tests whether the torsion model can use the partial L improvements
that Q1/Q3 sources will produce. Arm 0 costs about 3 GPU-h and can cancel it.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
