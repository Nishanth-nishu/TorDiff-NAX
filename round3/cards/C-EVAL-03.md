# C-EVAL-03: GFN2-xTB relaxation metrics for generated conformers (E_relax, displacement to own minimum)

## 1. Idea (scout)
On a fixed subset of test molecules (and a matching validation subset), relax every generated conformer with GFN2-xTB
and report, per arm: median and mean relaxation energy E_relax = E(generated) − E(relaxed); the bond, angle and torsion
displacement between each conformer and its own relaxed structure (Nikitin et al.'s metric); heavy-atom RMSD to the
own minimum; and the fraction of relaxed conformers whose energy lies within 6 kcal/mol of the lowest GT conformer.
GT conformers run through the same pipeline are the zero check. The relaxed structures are kept, because C-EVAL-05
re-scores them at no extra cost.

## 2. Where it comes from (scout)
- GEOM conformers are CREST structures optimised with GFN2-xTB [E-EVAL-012], and for GEOM-QM9 the original DFT
  geometries were re-optimised with xTB first [E-EVAL-013]; CREST ran with default settings except the charge, i.e.
  gas phase [E-EVAL-055], with xTB 6.2.3 [E-EVAL-056]. So the reference conformers are gas-phase GFN2-xTB minima.
- Re-optimising GFN2-xTB-optimised GEOM structures with GFN2-xTB gives E_relax ≈ 0, whereas MMFF evaluation of the same
  structures gives ~16 kcal/mol [E-EVAL-018]; MMFF94 energy is called unsuitable for models trained on GFN2-xTB data
  [E-EVAL-021]. Nikitin et al. propose bond/angle/torsion differences to the own xTB-optimised counterpart
  [E-EVAL-019] and find that diffusion models already beat MMFF on bonds, angles and E_relax (MMFF→xTB: 1.12 × 10⁻² Å,
  1.22°, mean E_relax 11.4 kcal/mol on GEOM-Drugs), but not on torsions, where MMFF's 4.89° is smaller than the
  diffusion models' 5.58–8.58° [E-EVAL-020]. (Revised after D-007: the original said they beat MMFF "on these",
  including torsions.)
- TD's own ensemble-property test shows how strongly energies react to rigid RDKit local structure: unrelaxed TD
  conformers have a median E_min error of 36.94 kcal/mol on DRUGS, 0.13 kcal/mol after GFN2-xTB relaxation
  [E-EVAL-022], and TD states unrelaxed errors are "far too large" to be useful [E-EVAL-023].
- CREST keeps conformers within 6.0 kcal/mol [E-EVAL-014], which gives a natural energy-window check.
- TD already has an `xtb_optimize` wrapper [E-EVAL-034]; conda-forge ships linux-64 xtb builds, including the
  reference version 6.2.3 [E-EVAL-032, E-EVAL-056] (Revised after D-003).

## 3. Why it could matter here (scout)
- **Reference-free.** Every L-quality number we have (λ, `l_error.csv`, local errors vs GT) needs test-set GT and is
  ORACLE for model selection [E-EVAL-049]. E_relax needs only the generated structure, so it can select among L
  sources or checkpoints on validation molecules. For molecules where ETKDG fails it is available only for arms whose L
  does not come from ETKDG (TD-family arms produce no structure there). (Revised after V1 note, P2.)
- **More sensitive to L than RMSD.** RDKit L is 0.036 Å / 3.88° from GT vs 0.004 Å / 1.67° between GT conformers
  (round-1 seed set) [E-EVAL-047]; bond-length errors of a few hundredths of an Å cost little RMSD but a lot of energy
  (INFERENCE from E-EVAL-022 on DRUGS, where every method, including GeoMol with its own local structure, has large
  unrelaxed energy errors). Arms that share ETKDG L and differ only in the torsion model (S3 0.1816 vs CTRL 0.1774 Å
  [E-EVAL-043, E-EVAL-050]) should have near-identical bond/angle terms; any E_relax difference between them isolates
  torsions placed in strained regions.
- **Tests the MMFF reading.** MMFF L helps AMR-R (0.1752 → 0.1507 Å [E-EVAL-038, E-EVAL-040]), but MMFF minima are not
  xTB minima [E-EVAL-018, E-EVAL-020]. E_relax under xTB says whether MMFF L is genuinely closer to the reference PES,
  which matters for choosing MMFF vs xTB vs learned L as the round-3 L source.

## 4. Assumptions that may not transfer (scout)
- All published E_relax numbers are for GEOM-Drugs and de novo generators [E-EVAL-018, E-EVAL-020]; for QM9
  conformers (≤ 9 heavy atoms) the magnitudes will be smaller and the arm differences are unknown.
- E_relax rewards agreement with GFN2-xTB, not with higher-level truth: on BACE, DFT re-optimisation moves CREST
  geometries by 0.36 Å mean RMSD [E-EVAL-015]. That is acceptable here because the RMSD references are xTB too.
- xTB optimisation can change the molecule in strained cases (GEOM's own QM9 runs changed the graph for 11.6 % of
  molecules [E-EVAL-017]); relaxed structures must be re-checked with the evaluator's SMILES filter.
- Install and runtime on gnode118 are UNVERIFIED (round 2 skipped xTB, round2/SHORTLIST.md:49) [E-EVAL-032]. Settings
  to match the reference: GFN2-xTB, gas phase, neutral unless charged, xtb 6.2.3 if it installs [E-EVAL-055,
  E-EVAL-056] (Revised after D-003; the "gas phase assumed" item is now settled).
- Mean E_relax is outlier-dominated (Nikitin report median and mean); a handful of strained cages could dominate.

## 5. Minimal experiment (scout)
- Subset: 200 random test molecules (fixed list, ETKDG-embeddable) + 200 validation molecules; all 2K conformers of
  each arm (≈ 5k conformers per arm at K ≈ 13.7).
- Arms: GT conformers (zero check), A0 ETKDG only, R0, A2 (MMFF-before), CTRL s0 on ETKDG and MMFF L, S3 s0 and B1 s0 on
  ETKDG L, A1 (ORACLE, sanity), plus any xTB-L arm from SCOUT GEOM's cards.
- Metrics: median/mean E_relax (kcal/mol), mean |Δbond| (Å), |Δangle| (°), |Δtorsion| (°), RMSD to own minimum (Å),
  fraction within 6 kcal/mol of the lowest GT energy; bootstrap CI over molecules.
- Expected: GT ≈ 0; ETKDG-L arms (R0, CTRL, S3, B1) high and nearly identical (same L); MMFF-L arms lower in bonds but
  not to zero; ORACLE-L arms near zero in bonds/angles, non-zero in torsions.
- Cost: 0 GPU-h. ≈ 40k GFN2-xTB optimisations for the 200 test molecules and ≈ 80k with the 200 validation molecules;
  at an assumed 1–3 s each (UNVERIFIED; time 100 first) 22–67 CPU-h, about 1–2 h wall on 32–48 cores; plus ≤ 30 min to
  install xtb from conda-forge. (Revised after V1 note: the earlier 40k omitted the validation subset.)

## 6. Scout's own call (scout)
Worth trying: YES — cheap CPU, reference-free, and the only metric in our plan that does not need test-set GT;
downgrade to MAYBE if xtb cannot be installed on gnode118 within ~1 h.

Revision history (scout, P2): Revised after D-003 and D-007 (torsion qualification, xtb 6.2.3 / gas phase settled,
ETKDG-failure scope, cost with validation subset). The call is unchanged.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
