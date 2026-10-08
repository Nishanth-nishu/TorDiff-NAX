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
- TD's own property test applies GFN2-xTB relaxation and shows it removes most of the unrelaxed energy error
  (E_min error 36.94 → 0.13 kcal/mol, DRUGS) [E-EVAL-022]; TD calls relaxation of local structures "necessary for any
  method" [E-EVAL-023]. TD relaxes in its ensemble-property test [E-EVAL-024]; in the TD code the RMSD evaluation runs
  on unrelaxed samples unless `--post_mmff` is set [E-EVAL-035]. Whether each QM9 paper cited in C-EVAL-01 relaxes
  before RMSD was not checked paper by paper (UNVERIFIED). (Revised after D-007: E-EVAL-024 alone does not show the
  RMSD tables are unrelaxed; the code path does, for TD.)
- A different post-hoc mechanism, EnFlow's *selection* by a learned energy (3K generated, the 2K lowest kept; no
  relaxation), also trades recall for precision: COV-R 96.26 → 90.91 %, COV-P 95.48 → 96.15 % [E-EVAL-011]. It is cited
  only as an example of post-hoc filtering costing recall, not as evidence about relaxation. (Revised after D-007: the
  original called it "energy-based post-processing" and used it as relaxation evidence.)
- Hook: TD already has a post-generation MMFF switch and an xTB optimiser [E-EVAL-035, E-EVAL-034].

## 3. Why it could matter here (scout)
- Our MMFF-after arm A3 made recall worse (AMR-R 0.1863 vs 0.1752 Å) and precision better (AMR-P 0.1641 vs 0.2191 Å)
  [E-EVAL-041, E-EVAL-038]. Round 1 read this as "local structure has to be right during generation" [E-EVAL-053]. INFERENCE: part of A3's recall loss is that MMFF minima are not xTB minima [E-EVAL-020, E-EVAL-021]; xTB-after
  separates "relaxation is bad for recall" from "MMFF is the wrong target".
- The planned round-3 levers (better L at test time, models that use good L) aim to move TD from ~0.18 Å toward the
  oracle region (0.08 Å for the standard model with true L [E-EVAL-042]). If xTB-after alone recovers a large part of
  that, a learned L must show a gain *on top of* it or a cost advantage; if xTB-after hurts recall like A3, the
  "relax-after" route is closed and learned L is the only way (strengthening the FlexiTors case).
- Rigid molecules (0 heavy torsions) are 30.4 % of molecules and 26.0 % of the macro AMR-R sum, and they carry
  54.9 % of the gain from true ring geometry (ORACLE) [E-EVAL-057]; for them xTB-after acts only on L (no rotatable
  torsions to move), so on that stratum it is a direct non-learned competitor of any ring-L component. (Revised after
  D-005: the original said "rigid ring molecules dominate the macro average", which conflated few-conformer with rigid
  molecules.)

## 4. Assumptions that may not transfer (scout)
- Post-relaxed RMSD numbers are not comparable to literature RMSD rows (TD's are unrelaxed per its code path
  [E-EVAL-035]; others not checked); report them as a yardstick row, never as "our method".
- Relaxation can merge distinct samples into the same minimum (fewer distinct conformers per molecule) and can flip
  ring puckers or move torsions to the nearest xTB minimum, which may be a minimum missing from the CREST set; the
  recall direction is genuinely uncertain (MMFF relaxation A3 lost recall [E-EVAL-041]; EnFlow's selection, a different
  mechanism, also lost recall [E-EVAL-011]).
- xTB may break or change strained molecules [E-EVAL-017]; the evaluator's SMILES filter must run after relaxation and
  dropped conformers counted.
- The DRUGS property result [E-EVAL-022] is about Boltzmann-weighted properties, not RMSD; it shows that relaxation
  fixes local structure, not that it improves coverage.
- xtb install/runtime on gnode118 UNVERIFIED [E-EVAL-032]; use GEOM's settings (gas phase, xtb 6.2.3 if available)
  [E-EVAL-055, E-EVAL-056].

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

Revision history (scout, P2): Revised after D-005 and D-007 (rigid share instead of "rigid ring molecules dominate";
EnFlow relabelled as learned-energy selection; TD-unrelaxed claim now cites the code path; reference settings added).
The call is unchanged: the core premise (xTB-after lands on the reference PES) was VERIFIED.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
