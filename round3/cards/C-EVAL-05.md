# C-EVAL-05: GFN2-xTB post-relaxation of generated conformers ("xTB-after") as a yardstick for learned L

## 1. Idea (scout)
Relax the final conformers of TD-family arms with GFN2-xTB and re-score them with the unchanged evaluator (AMR/COV at
0.05–0.5 Å, recall and precision). This is the xTB analogue of our MMFF-after arm A3. It answers one question the panel
will be asked about any FlexiTors result: does a learned L beat simply relaxing TD's output at the reference level of
theory, and are the two complementary (FlexiTors + xTB-after)? The relaxed structures come for free from C-EVAL-03.

## 2. Where it comes from (scout)
- Reference conformers are GFN2-xTB minima [E-EVAL-012, E-EVAL-013], and xTB re-optimisation leaves such structures
  essentially unchanged (E_relax ≈ 0) [E-EVAL-018]. A relaxed TD conformer therefore lands on the same potential-energy
  surface as the references, unlike an MMFF-relaxed one, whose minima differ from xTB's by 1.22° (angles) and 4.89°
  (torsions) on drug-like molecules [E-EVAL-020].
- TD's own property test applies GFN2-xTB relaxation and shows it removes most of the error caused by RDKit local
  structure (E_min error 36.94 → 0.13 kcal/mol, DRUGS) [E-EVAL-022]; TD calls relaxation "necessary for any method"
  [E-EVAL-023]. TD relaxes only in its ensemble-property test, not for its RMSD tables [E-EVAL-024]; none of the QM9
  tables cited in C-EVAL-01 describes a relaxation before RMSD (not checked paper by paper: UNVERIFIED).
- Energy-based post-processing trades recall for precision: EnFlow's energy selection lowers COV-R 96.26 → 90.91 % and
  raises COV-P 95.48 → 96.15 % [E-EVAL-011].
- Hook: TD already has a post-generation MMFF switch and an xTB optimiser [E-EVAL-035, E-EVAL-034].

## 3. Why it could matter here (scout)
- Our MMFF-after arm A3 made recall worse (AMR-R 0.1863 vs 0.1752 Å) and precision better (AMR-P 0.1641 vs 0.2191 Å)
  [E-EVAL-041, E-EVAL-038]. Round 1 read this as "local structure has to be right during generation" [E-EVAL-053]. INFERENCE: part of A3's recall loss is that MMFF minima are not xTB minima [E-EVAL-020, E-EVAL-021]; xTB-after
  separates "relaxation is bad for recall" from "MMFF is the wrong target".
- The planned round-3 levers (better L at test time, models that use good L) aim to move TD from ~0.18 Å toward the
  oracle region (0.08 Å for the standard model with true L [E-EVAL-042]). If xTB-after alone recovers a large part of
  that, a learned L must show a gain *on top of* it or a cost advantage; if xTB-after hurts recall like A3, the
  "relax-after" route is closed and learned L is the only way (strengthening the FlexiTors case).
- Rigid ring molecules dominate the macro average [E-EVAL-048]; for them xTB-after acts only on L (no torsions to
  move), so it is a direct non-learned competitor of any ring-L component.

## 4. Assumptions that may not transfer (scout)
- Post-relaxed RMSD numbers are not comparable to literature RMSD rows, which (as far as checked, [E-EVAL-024]) are
  computed on unrelaxed samples; report them as a yardstick row, never as "our method".
- Relaxation can merge distinct samples into the same minimum (fewer distinct conformers per molecule) and can flip
  ring puckers or move torsions to the nearest xTB minimum, which may be a minimum missing from the CREST set; the
  recall direction is genuinely uncertain (A3 and EnFlow both lost recall [E-EVAL-041, E-EVAL-011]).
- xTB may break or change strained molecules [E-EVAL-017]; the evaluator's SMILES filter must run after relaxation and
  dropped conformers counted.
- The DRUGS property result [E-EVAL-022] is about Boltzmann-weighted properties, not RMSD; it shows that relaxation
  fixes local structure, not that it improves coverage.
- xtb install/runtime on gnode118 UNVERIFIED [E-EVAL-032].

## 5. Minimal experiment (scout)
- Arms (CPU only, reuse C-EVAL-03's relaxed structures on the same 200-molecule subset, then the full test set if
  informative): R0 seed 0 + xTB-after; A2 + xTB-after; CTRL s0 ETKDG L + xTB-after; S3 s0 ETKDG L + xTB-after;
  B1 s0 ETKDG L + xTB-after.
- Controls: the same arms unrelaxed; A3 (MMFF-after); A1 (ORACLE) as the reference for "true L".
- Metrics: AMR-R/AMR-P mean and median; COV-R/COV-P at 0.05/0.1/0.5 Å; number of distinct relaxed conformers per
  molecule (heavy-atom RMSD > 0.05 Å); paired bootstrap on the intersection.
- Expected (uncertain, two-sided): AMR-P clearly better than unrelaxed (as A3); COV-R@0.05 much higher than 3.9 %
  (bonds/angles become xTB-exact); AMR-R anywhere from slightly worse than R0 (A3-like collapse) to well below A2.
  Pre-register: xTB-after counts as "strong" if AMR-R ≤ 0.150 Å (A2 level) on the intersection.
- Cost: 0 GPU-h; ≈ 5k xTB optimisations per arm on the subset (shared with C-EVAL-03), ≈ 27k per arm on the full set
  (≈ 8–25 CPU-h per arm at an assumed 1–3 s per optimisation, UNVERIFIED).

## 6. Scout's own call (scout)
Worth trying: YES as a yardstick (nearly free once C-EVAL-03 runs, and the panel cannot judge a learned L without it);
not to be reported as a method or compared with literature RMSD rows.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
