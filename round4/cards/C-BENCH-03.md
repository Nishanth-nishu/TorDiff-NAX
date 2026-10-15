# C-BENCH-03: Required extra metrics: GFN2-xTB E_relax, local-geometry and ring errors, runtime; ensemble properties on DRUGS only

## 1. Idea (scout)
Add four reference-free or cost metrics to every main-table row: (a) GFN2-xTB relaxation energy E_relax of each
generated conformer (`xtb --opt`, gas phase, the level of the GEOM references), mean and median per molecule;
(b) bond-length, bond-angle and torsion change between each generated conformer and its own xTB-optimised
counterpart, plus a ring-specific line (endocyclic-dihedral change; share of 4+-atom rings twisted > 10°);
(c) stereo-correctness rate (CIP labels of generated vs input); (d) runtime in CPU core-seconds per conformer (TD's
protocol) and GPU throughput, including seed preparation (MMFF/xTB/template). Boltzmann-weighted ensemble properties
(TD/ET-Flow protocol) only on a published 100-molecule DRUGS subset, if DRUGS is run.

## 2. Where it comes from (scout)
- E_relax with GFN2-xTB is the level-consistent energy metric: GEOM conformers re-optimised with GFN2-xTB relax by
  ≈ 0, MMFF evaluation gives ≈ 16 kcal/mol [E-BENCH-037]; MMFF energies are "not suitable" for GFN2-xTB-trained models
  [E-BENCH-040].
- Geometry changes vs own xTB-optimised counterpart are the recommended local metrics [E-BENCH-038]; diffusion models
  beat MMFF on bonds/angles/E_relax but not on torsions [E-BENCH-039].
- A third-party implementation exists (xtb --opt, "total energy gain", pair geometry, mean ± std over 5 subsets)
  [E-BENCH-030]; GEOM-Drugs Revisited's "corrected evaluation" otherwise concerns valency/stability, which does not
  apply to conformer generation [E-BENCH-029].
- Ensemble properties: TD/ET-Flow protocol (100 DRUGS molecules, min(2K, 32), xTB relaxation) [E-BENCH-041]; the
  subset is not released, so MCF used 3 random subsets with mean ± std [E-BENCH-042].
- Runtime: TD reports CPU core-seconds per conformer on 10 molecules × 8 conformers (TD 20 steps 4.90, RDKit 0.10)
  [E-BENCH-043].
- Chirality: ET-Flow's post-hoc correction only fixes globally inverted molecules [E-BENCH-027].
- TD's own analysis: rigid molecules' error is set by the local structure given [E-BENCH-045]; reviewers aware of
  Zhou et al. will read gains toward xTB minima as agreement with the reference level [E-BENCH-051].

## 3. Why it could matter here (scout)
- The project's thesis is that ring/local geometry is the floor (true rings 0.178 → 0.119 Å, ORACLE; brief) and that
  MMFF geometry gives the only real-use gain so far (−0.029 Å) [E-BENCH-068, E-BENCH-069]. RMSD-to-reference cannot show
  *why*; E_relax and ring-dihedral errors measure local geometry directly, without any oracle.
- TD freezes RDKit L, whose seeds move 0.09–0.32 Å under xTB [E-BENCH-072]; INFERENCE: TD's E_relax will be
  large and an L-correcting method's small, a difference that does not depend on the conformer-identity leak.
- Seed preparation cost (xTB ≈ 0.14 s per QM9 seed, ≈ 1–2 s per DRUGS seed estimated [E-BENCH-060]) is of the same
  order as TD's per-conformer cost [E-BENCH-043]; reviewers will ask for the cost line.

## 4. Assumptions that may not transfer (scout)
- If our method uses xTB-relaxed seeds, its E_relax is partly low by construction (only torsions move afterwards); the
  TD + xTB-after yardstick (round-3 C-EVAL-05) and a torsion-only E_relax decomposition are needed to avoid a circular
  claim.
- xtb version: GEOM used 6.2.3, gnode118 has 6.7.1 [E-BENCH-071]; QM9 GT conformers do not move under 6.7.1 (10/10) [E-BENCH-072],
  so the version difference is negligible at the geometry level, but say so.
- Ensemble properties on QM9 would rely on CREST weights, which GEOM calls inaccurate (C-BENCH-07); keep them to DRUGS
  as in TD/ET-Flow.
- E_relax distributions have heavy tails (a few failed optimisations); report medians and failure counts.

## 5. Minimal experiment (scout)
- QM9: for each main-table method, xTB-optimise all generated conformers (≈ 15k per method: 7,554 GT conformers × 2 [E-BENCH-073];
  × 0.157 s mean ≈ 40 CPU-min per method [E-BENCH-072]); compute (a)–(c) with the
  GEOM-revisited scripts or our `tools/geometry_metrics.py` variant pointed at the xTB-optimised partner.
- Runtime: TD protocol (10 random test molecules × 8 conformers, 8 threads) for TD, our method with each seed source,
  ET-Flow; plus wall-clock of the full test set on one 3090.
- DRUGS (main venue only): same metrics on the 1000 test molecules (≈ 1–2 s per xTB call, CPU budget to be measured);
  ensemble properties on a fixed, published 100-molecule subset, 3 subsets if affordable.
- GPU: 0 (runtime benchmark aside). CPU: < 10 CPU-h for QM9.

## 6. Scout's own call (scout)
Worth trying: YES — oracle-free, cheap on QM9, directly measures the quantity the method claims to improve, and is the
evaluation Nikitin et al. ask of GEOM-trained models.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
