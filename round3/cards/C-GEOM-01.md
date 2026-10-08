# C-GEOM-01: GFN2-xTB-relaxed ETKDG seeds as test-time L (the reference level of theory, CPU-gated)

## 1. Idea (scout)
Relax the same ETKDG embeddings we already use (and their MMFF versions) with GFN2-xTB, the level at which every GEOM
conformer was optimised, and use them as test-time L for the existing CTRL_rematch and B1 checkpoints (no training).
A CPU-only gate (L error, pucker-tail share, rigid-molecule AMR-R) decides whether any GPU time is spent. Optional arm:
oversample ETKDG, xTB-relax, and keep distinct ring puckers inside a 6 kcal/mol window.

## 2. Where it comes from (scout)
- GEOM conformers are CREST ensembles whose geometries are optimised with GFN2-xTB [E-GEOM-015]; for QM9 the inputs
  were re-optimised with xTB and CREST ran with default (gas-phase, inferred) settings, at 0.5 core-h per molecule
  [E-GEOM-016].
- Re-optimising GEOM references with GFN2-xTB changes them by ≈ 0 (relaxation energy ≈ 0; 0.00 × 10⁻² Å bonds, 0.001°
  angles, 0.01° torsions), while MMFF and GFN2-xTB minima differ by 1.12 × 10⁻² Å, 1.22° and 4.89° (GEOM-Drugs)
  [E-GEOM-018, E-GEOM-019]. So an xTB optimisation that starts in the reference basin should land on the reference L.
- TD already contains an `xtb_optimize` wrapper (gas phase) that needs only an `xtb` binary [E-GEOM-010]; xtb is a
  standalone binary on conda-forge (linux-64, 6.7.1) or a release tarball, independent of our torch/python stack
  [E-GEOM-034, E-GEOM-035].
- Practice: force-field relaxation only partly repairs ETKDG ring puckers (piperazine: many twisted rings after MMFF)
  [E-GEOM-045]; GFN2-xTB occasionally fragments molecules [E-GEOM-046], and strained QM9 molecules reacted under
  CREST/GFN2 [E-GEOM-047].
- CREST's 6.0 kcal/mol window is the GEOM pipeline's notion of "accessible" conformers (optional arm B) [E-GEOM-017].

## 3. Why it could matter here (scout)
Target numbers from `l_error.csv` (heavy atoms, mean per seed; AMR-R from BRIEF, unpaired, preliminary)
[E-GEOM-001, E-GEOM-002, E-GEOM-005, E-GEOM-007, E-GEOM-013]:

| test-time L | ring bond Å | ring angle ° | ring dihedral ° (share > 10°) | acyclic bond Å | acyclic angle ° | CTRL | B1 |
|---|---|---|---|---|---|---|---|
| ETKDG | 0.032 | 2.38 | 10.7 (28.6%) | 0.029 | 2.84 | 0.178 | 0.236 |
| MMFF-relaxed ETKDG | 0.020 | 1.42 | 8.1 (20.2%) | 0.017 | 2.03 | 0.152 | 0.232 |
| λ = 0.50 (ORACLE) | 0.019 | 1.34 | 3.2 (11.2%) | 0.017 | 2.01 | 0.136 | 0.181 |
| **λ = 0.75 (ORACLE, B1 = CTRL)** | **0.010** | **0.67** | **1.6 (0.01%)** | **0.009** | **1.00** | 0.116 | 0.115 |
| true L + noise 0.02 Å/axis (ORACLE) | 0.027 | 1.38 | 1.0 (0%) | 0.027 | 1.67 | 0.098 | 0.117 |

- The λ = 0.75 row is the accuracy spec: ring bonds ≤ ~0.01 Å, ring angles ≤ ~0.7°, acyclic angles ≤ ~1°, and
  essentially **no ring seed with ring-dihedral RMSD > 10°**. MMFF already matches λ = 0.5 on bonds/angles; what it
  misses is the pucker tail (20% vs 11% vs 0%) [E-GEOM-002, E-GEOM-007].
- The noise row shows CTRL tolerates 0.027 Å / 1.4° random bond/angle error when puckers are right (0.098), but B1
  does not (0.117) [E-GEOM-005]. B1 needs both right puckers and near-reference bonds/angles; xTB is the only cheap
  source that targets the reference bonds/angles exactly (INFERENCE from E-GEOM-018/019).
- Rigid molecules (≈ 30% of the test set) give a model-free gate: their AMR-R is the L quality itself (ETKDG 0.152,
  MMFF 0.128, λ = 0.5 0.107, λ = 0.75 0.077, true rings 0.043) [E-GEOM-004, E-GEOM-006].
- Ring-only truth helps CTRL (0.178 → 0.119) but not B1 (0.188) [E-GEOM-013]: xTB relaxes rings and acyclic L
  together, so it is the arm that can also move B1.

## 4. Assumptions that may not transfer (scout)
- **Basin retention.** Local optimisation keeps the ETKDG pucker basin wherever a barrier separates puckers (6-ring
  chair/boat/twist, 4-ring pucker sign). MMFF did not remove the tail (20% > 10°) [E-GEOM-002]; xTB may not either.
  Which part of MMFF's tail is level-of-theory offset (fixable) vs wrong basin (not fixable) is unknown: this is
  exactly what the CPU gate measures.
- **Reference-level match is inferred for QM9.** The "≈ 0 relaxation" result is for GEOM-Drugs [E-GEOM-018]; QM9
  being gas-phase GFN2 is inferred from "default CREST arguments" [E-GEOM-016]. CREST's internal optimisation
  thresholds may differ from `xtb --opt` defaults (small residual, UNVERIFIED).
- **L|τ coupling.** Whole-molecule optimisation also relaxes torsions; TD then re-randomises torsions, so acyclic angles
  sit at the seed's torsional state, not the sampled one. Ring L is unaffected (ring bonds are not rotatable).
- **Strained QM9 chemistry.** 11.6% of QM9 graphs changed under CREST [E-GEOM-047]; plain optimisation is milder, but
  seeds must be rejected (fallback to the ETKDG seed) if connectivity or stereo changes [E-GEOM-046].
- **Assignment caveat.** ETKDG/MMFF ring-dihedral errors are measured against angle-assigned GT conformers
  [E-GEOM-009]; the gate must add a pucker-coverage metric (min over seeds per GT conformer).
- CTRL was trained on conformer-matched RDKit L, so xTB L is a train/test shift for CTRL (MMFF was too, and helped).

## 5. Minimal experiment (scout)
- **Step 0, install (CPU, time-box 30 min):** `micromamba create -p $PROJECT/xtbenv -c conda-forge xtb` (separate env;
  the TD venv is untouched) or the 6.7.1 release tarball. Not to be done by this scout.
- **Step 1, seeds (CPU):** extend the S1 seed builder so that the existing `L_etkdg2L` embeddings (same random seed,
  common random numbers) are also written as `L_etkdg2L_xtb` (GFN2-xTB from ETKDG) and `L_etkdg2L_mmff_xtb` (from the
  MMFF seeds), with per-seed graph/stereo checks, convergence flags, wall time, and `l_error.csv` rows. ~25k
  optimisations of ≤ 29-atom molecules; runtime UNVERIFIED (guess 0.5–2 s each ⇒ ≤ 1 h wall on 16 cores; time 200
  first).
- **Step 2, CPU gate (pre-declared):** for each L source report ring/acyclic bond and angle RMSD, share of ring seeds
  with ring-dihedral RMSD above 10°, pucker coverage, and the rigid-subset AMR-R of the seeds themselves. Proceed to GPU only if the xTB source
  beats MMFF on the > 10° share (≤ 15%) **or** on rigid AMR-R (≤ 0.110). Report it against the λ table above.
- **Step 3, GPU (inference only):** CTRL_rematch s0–2 and B1 s0–2 on `L_etkdg2L_xtb` (or the `_mmff_xtb` variant if
  it wins the gate) = 6 runs; add S3 (both seeds) and S4 (λ chosen on validation, never on test) if their checkpoints
  are final = +3–5 runs. Controls: the same checkpoints on `L_etkdg2L` and `L_etkdg2L_mmff` (already run).
  Primary: paired AMR-R (δ = 0.5 Å) CTRL-on-xTB vs CTRL-on-MMFF (expected lower; scout range 0.125–0.150), and
  B1-on-xTB vs CTRL-on-xTB (expected B1 ≤ CTRL only if the gate reaches the λ = 0.75 row; otherwise B1 stays worse).
  Secondary: COV-R@0.1 on the success intersection, rigid subset, ring-size strata. 1 generation seed.
- **Optional arm B (pucker-diverse):** 4 × 2L ETKDG embeddings → xTB → deduplicate by ring-dihedral fingerprint →
  drop seeds > 6 kcal/mol above the molecule's lowest xTB energy [E-GEOM-017] → sample 2L. +2 GPU runs.
- **Optional arm C (ceiling, not a method; validity flag):** CREST on the test SMILES (≈ 500 core-h,
  [E-GEOM-016]). Non-oracle (SMILES only) but it is the generator that produced GT, so report it only as a ceiling.
- **Cost:** 1.5–3 GPU-h (6–12 × 0.25) + ≤ 1 day CPU. No training. Follow-up only if it wins: xTB-matched training
  (S5 analogue, ~1M training conformers, CPU cost not estimated).

## 6. Scout's own call (scout)
Worth trying: YES — cheapest non-oracle source that targets the GT's own level of theory, no training, and the CPU gate
answers "is the pucker tail a force-field offset or a wrong basin?" before any GPU time.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
