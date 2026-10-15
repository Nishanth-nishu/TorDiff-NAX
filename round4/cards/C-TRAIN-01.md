# C-TRAIN-01: Leak-free reference-L training: sibling-conformer L with conformer-matched torsions

## 1. Idea (scout)
Keep S3's 50/50 recipe but replace its "own GT L" half with **the GT L of a different conformer j of the same
molecule, whose rotatable torsions are fitted (TD conformer matching) to the target conformer i**. The training input
then has reference-quality bonds, angles and ring puckers but carries the imprint of conformer j's torsions, not the
target's, exactly like a deployed xTB / template / learned seed (which carries the imprint of its seed's random
torsions). Single-conformer molecules contribute only RDKit-matched L. The test is whether a torsion model trained this
way GAINS from improved-but-imperfect L (MMFF, xTB, leak-free true L) instead of losing, as S3 and B1 do.

## 2. Where it comes from (scout)
- TD's own fix for the train/test L shift is conformer matching: replace each training conformer's L by sampler L and
  fit only the torsions to the GT conformer [E-TRAIN-022, E-TRAIN-023]. TD also computed, but never trained on, the
  "GT L of another randomly chosen GT conformer" matching floor: 0.284 Å on DRUGS vs 0.324 Å for RDKit L
  [E-TRAIN-020]. This card uses that sibling L as a training source.
- TD's model trained directly on GT L had a *significantly lower* training and validation loss yet did worse at test
  [E-TRAIN-021]. INFERENCE: a much easier loss is what one expects if own-conformer L lets the model read the target's
  torsions off L.
- Identity leaks through the conditioning input are a named failure mode elsewhere: AlphaFold2 backbones carry
  sub-Å artefacts that "may give away" amino-acid identity, so ESM-IF1 adds 0.1 Å noise to stop the model learning
  generator-specific rules [E-TRAIN-030, E-TRAIN-031]; ProteinMPNN attributes its exact-backbone advantage to "memory"
  of the target in refined coordinates [E-TRAIN-033]; HarmonicFlow/FlowSite noise their pocket construction so the
  model cannot infer the ligand position from how the pocket was built [E-TRAIN-034].
- Torsional-GFN trains p(τ | G, L) with L drawn from MD, independent of any target torsions, and generalises
  zero-shot to unseen L [E-TRAIN-046] (small workshop study).

## 3. Why it could matter here (scout)
- **The measured failure.** S3 is significantly worse than CTRL on RDKit L (+0.0032 Å, Holm p = 0.0096)
  [E-TRAIN-001] and much worse on MMFF L (+0.039); B1 +0.080; S2 +0.022 [E-TRAIN-002]. From ETKDG to MMFF L, CTRL
  improves by 0.025 Å and S2 by 0.033, B1 does not move and S3 gets *worse* by 0.011 [E-TRAIN-003]. S3 beats CTRL
  only on exact own-conformer true L (−0.049, ORACLE) and loses at λ = 0.5 and on noisy true L [E-TRAIN-005].
- **Why (INFERENCE, the card's hypothesis).** CTRL's training L is a fresh ETKDG embedding, generated independently of
  the target and only torsion-fitted to it [E-TRAIN-007], so L never tells CTRL where the target's torsions are; it
  learns to use L as local geometry, and gains from MMFF. S3's GT half is the conformer's own L with its own torsion
  target [E-TRAIN-008]: a relaxed L encodes the torsions at which it was relaxed, so the model can read them off. At
  inference every L carries only the seed's random torsions [E-TRAIN-009]; the cleaner and more GT-like the L looks
  (MMFF), the more S3 trusts a false imprint. The round-3 leak estimate agrees: on multi-conformer molecules B1 goes
  from 0.027 Å with own L to an implied 0.249 with other conformers' L (≈ its RDKit 0.256), CTRL 0.102 → 0.138
  [E-TRAIN-006] (an estimate; the direct split-half test exists now [E-TRAIN-018]).
- **Why sibling L should work.** It is reference-level L (what xTB/templates aim to deliver, E-TRAIN-020) with a
  non-target imprint (what every deployed L has, E-TRAIN-009), combined with matched targets, which is the only recipe
  that has improved real use so far (S5: −0.0041 Å on MMFF L) [E-TRAIN-004].
- **Predicted effect on our numbers.**
  - RDKit L: ≈ CTRL (0.178–0.181): the RDKit half is CTRL's data and the sibling half cannot leak.
  - MMFF L: 0.148–0.155 vs CTRL 0.1525, i.e. it should *keep* CTRL's MMFF gain (S3 lost 0.039 there). This is the
    decisive, non-oracle readout.
  - Split-half (leak-free) true L, ORACLE: better than CTRL by 0 to −0.03 Å (CTRL's implied other-L value 0.138 on the
    multi-conformer subset), with a small own − split gap (B1/S3 should show a large one).
  - xTB L (C-GEOM-01): ≤ CTRL by 0 to −0.01. The upside is bounded: TD's sibling-L matching floor is only 12 % below
    RDKit's on DRUGS [E-TRAIN-020].
- **Paper value.** If it works, "identity leakage in L-conditioned torsion training and its removal by sibling-L
  matching" is a clean, falsifiable methods finding that explains B1/S3/S4 and every true-L "ceiling" number we have.
  Novelty must still pass the gate (TD F.1 computed the floor, not the training recipe).

## 4. Assumptions that may not transfer (scout)
- About 25 % of QM9 test molecules have one GT conformer (round3/impact.md §3g); in training those get no sibling and
  fall back to RDKit-matched L. Fewer reference-L samples than S3.
- Siblings of a molecule often share puckers; then sibling L ≈ own L except for the torsion imprint (≈ 1–3° angle
  relaxations). That is precisely the signal to remove, but if the imprint is weak the card degenerates to S3 (and
  would fail the same way). Conversely, siblings with a different pucker give large matching residuals and noisier
  targets (TD F.4: matched conformers are not minima; harder learning).
- All literature support is analogy (proteins, docking) except TD's own matching and sibling floor; nobody has trained a
  torsion model on sibling L.
- Requires the sibling GT conformers to share atom order with the target (same raw GEOM record; build_paired_pickles.py
  checks same_graph at :108) [E-TRAIN-014]. Grounder to confirm.
- The imprint hypothesis itself is INFERENCE; the split-half run (leak = own − split for B1/S3/CTRL) tests it first.

## 5. Minimal experiment (scout)
- **Gate (CPU + the running split-half test):** proceed if the split-half analysis shows a leak (own − split) for
  S3/B1 at least 3× CTRL's. If there is no leak, the hypothesis is wrong and the card is dropped.
- **Build (CPU, no TD model code):** a `sibling` standardisation variant. In `standardize_confs.py` the L pool is
  `EmbedMultipleConfs` (:78-79); replace it by the molecule's other GT conformers (j ≠ i), build the von-Mises cost
  (:89-93) with the diagonal forbidden, assign, and DE-fit torsions with the existing `optimize_rotatable_bonds`
  (standardization.py:29; call at standardize_confs.py:104-107). Store the result as the "GT half" of a paired cache
  (`tools/build_paired_pickles.py`, `gt_pos_aligned` at :124) so that `--l_mix_p_gt 0.5` (dataset.py:98) trains on
  {RDKit-matched, sibling-matched}. CPU ≈ one rematch run (1–4 h on 32 cores) + pairing (≈ 4 h on 14 CPUs, round 2)
  + featurize ≈ 1 h.
- **Arm:** S3-sib, `--l_mix_p_gt 0.5` on the sibling cache, 100 epochs, 3 seeds.
- **Optional arm (if the gate shows siblings rarely differ):** de-imprinted own L: randomise the target's torsions,
  relax L (MMFF or xTB, heavy-atom torsions free), then DE-fit torsions back to the target.
- **Controls:** CTRL_rematch (3), S3 (2), S5 (3), B1 (3): all existing checkpoints.
- **Test conditions:** primary non-oracle: RDKit L, MMFF L (`S1_etkdg2L`, `S1_etkdg2L_mmff`); xTB seeds when
  C-GEOM-01 exists. Secondary ORACLE: split-half split / own (tools/make_split_half.py) [E-TRAIN-018].
- **Primary contrast and direction:** S3-sib vs CTRL_rematch on MMFF L, non-inferiority margin +0.004 Å, and S3-sib
  vs S3 on MMFF L, expected ≈ −0.03 Å. Key secondary: leak (own − split) of S3-sib ≈ CTRL's. Paired, Holm, two-level
  bootstrap over seeds and molecules (round-3 validity V-4).
- **Cost:** 3 × 11.7 ≈ 35 GPU-h training + ≈ 3 GPU-h evaluation (6 conditions × 3 seeds × 0.17); ≈ 10 CPU-h.

## 6. Scout's own call (scout)
Worth trying: YES — it targets the one failure every true-L recipe shares (they lose on better-but-imperfect L), costs
no model code, and either fixes it or falsifies the leak explanation; gate it on the split-half result.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
