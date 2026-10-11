# C-EVAL-04: Non-learned baselines that any FlexiTors result must beat (ETKDG+MMFF, ETKDG+xTB, matched-budget clustering)

## 1. Idea (scout)
Run a small panel of non-learned baselines on the TD split, at exactly 2K conformers per molecule, scored with the same
evaluator and on the same success intersection as the learned arms:
- **B-mmff:** ETKDG + MMFF94s optimisation, no torsion model (missing today; we only have it *inside* TD as A2).
- **B-xtb:** ETKDG (+ MMFF pre-clean) + GFN2-xTB optimisation, no torsion model (missing; the reference level of theory).
- **B-clust:** Zhou et al.'s RDKit + Clustering (oversample ETKDG / random-dihedral / ETKDG+MMFF, K-means to 2K), and,
  for fairness, TD with the same oversample-then-cluster budget.
- **Protocol knob:** record the ETKDG version used for seeds (ours is v1); an ETKDGv3 seed variant is left to SCOUT
  GEOM's L-source cards and is not run here.

## 2. Where it comes from (scout)
- Zhou et al. show that a parameter-free RDKit + Clustering recipe (samplers 1:1:4, N_e = min(20 N_ref, 2000), K-means
  to 2 N_ref) [E-EVAL-025] beats most deep models on recall on the GeoDiff QM9 protocol (COV 97.65 %, MAT 0.1902 Å vs
  RDKit 83.26 % / 0.3447 Å) [E-EVAL-026]. Their ablation shows the gain comes from oversampling (2 N_ref budget: MAT
  0.2223 Å) and from the MMFF sampler (without it: 0.2511 Å) [E-EVAL-027].
- Zhang et al. answer that the comparison is unfair unless deep models get the same ~2000-sample-and-cluster budget
  [E-EVAL-028].
- Literature QM9 rows (TD protocol) for non-learned tools: RDKit AMR-R 0.235 Å, OMEGA 0.177 Å [E-EVAL-003, E-EVAL-004].
  No published GEOM-QM9 (TD split) number exists for ETKDG + GFN2-xTB (search negative, ledger §G).
- The references are GFN2-xTB minima [E-EVAL-012, E-EVAL-013]; MMFF is not at that level of theory [E-EVAL-021] and
  its minima differ from xTB's by 1.22° in angles and 4.89° in torsions on drug-like molecules [E-EVAL-020].
- A practitioner blog reports missed and high-energy ETKDG conformers for large systems (not QM9-relevant) and,
  anecdotally, a preference for twist boats over chairs; it says such problems "can be ameliorated" by generating
  thousands of conformers and deduplicating, at a cost in speed and redundancy [E-EVAL-029]. (Revised after D-002: the
  earlier "ring pathologies that oversampling only partly fixes" overstated the source.) RDKit's default seed generator changed from ETKDGv1 to ETKDGv3 in 2024.03 [E-EVAL-030]; our TD seeds
  come from a call without parameters on RDKit 2022.9.5, i.e. ETKDGv1 [E-EVAL-031].
- The xTB wrapper already exists in TD [E-EVAL-034]; xtb is on conda-forge, including GEOM's version 6.2.3
  [E-EVAL-032, E-EVAL-056] (Revised after D-003).

## 3. Why it could matter here (scout)
- **We do not know how much of A2's gain is MMFF alone.** ETKDG alone 0.2298 Å [E-EVAL-039]; TD on ETKDG 0.1752 Å
  [E-EVAL-038]; TD on MMFF-relaxed ETKDG 0.1507 Å [E-EVAL-040], reproduced by our CTRL (0.1518 Å) [E-EVAL-050]. Without
  B-mmff the "model contribution" on MMFF L is unmeasured.
- **B-xtb could be a strong QM9 baseline.** INFERENCE: reference conformers are xTB minima [E-EVAL-013], about half of
  the test molecules have ≤ 3 GT conformers and are ring-containing [E-EVAL-048], and about half of those (30 % of all
  molecules) are rigid [E-EVAL-057]; for the rigid ones an xTB-optimised ETKDG conformer in the right ring pucker is
  close to *the* reference. (Revised after D-005: few-conformer is not the same as rigid.) If B-xtb approaches TD's 0.15–0.18 Å
  without any learning, a learned-L FlexiTors arm has to be judged against it, not against raw ETKDG.
- **Recall can be bought.** Zhou's numbers (GD protocol, not comparable to ours [E-EVAL-026]) show that the sampling
  budget moves recall (COV 91.23 % at a 2 N_ref budget vs 97.65 % with oversampling), while the MMFF sampler improves
  MAT, not COV (removing it raises COV to 98.01 % but worsens MAT to 0.2511 Å) [E-EVAL-027]. (Revised after V1 note,
  P2.) every recall gain we claim should be checked against precision (AMR-P, COV-P)
  and against B-clust at the same budget [E-EVAL-028].
- Measured L error shows MMFF reaches λ≈0.5-level bonds/angles but keeps ETKDG-level ring dihedrals (8.07° vs 10.65°)
  [E-EVAL-049]; B-xtb tells whether a physics optimiser at the right level of theory also fixes ring dihedrals.

## 4. Assumptions that may not transfer (scout)
- Zhou's and Zhang's results use the GeoDiff/ConfGF split, 200 test molecules and conformer caps, not TD's split
  [E-EVAL-026, E-EVAL-028]; only the recipe transfers, not the numbers.
- All RDKit-based baselines inherit the ~65 ETKDG failures [E-EVAL-046]; compare on the intersection and report
  failures separately.
- B-clust's scores depend on its sampling budget [E-EVAL-027] (Zhang et al. call the comparison unfair without equal
  budgets [E-EVAL-028]); it is a check on recall claims, not a target. (Revised after V1 note: "recall-gaming by
  design" was the scout's characterisation, not shown by E-EVAL-027.) With N_e = min(20 K, 2000)
  plus the two quarter-size samplers [E-EVAL-025] it draws ≈ 1.5 × 20 K ≈ 15× more structures than 2K (K ≈ 13.7).
- xTB optimisation from distorted ETKDG geometries may change topology or stereo in strained molecules
  [E-EVAL-017]; apply the evaluator's SMILES filter after relaxation. xtb install/runtime on gnode118 UNVERIFIED
  [E-EVAL-032].
- Changing the seed generator (ETKDGv1 → v3) at test time only would create a train/test L mismatch for models trained
  on v1-matched conformers; that is why the v3 variant is not proposed here.

## 5. Minimal experiment (scout)
- Arms (all at 2K per molecule, full test set, 3 RDKit seeds): B-mmff, B-xtb (200-molecule pilot first, full set if the
  pilot extrapolates to < 1 day wall on 48 cores), B-clust (RDKit side, N_e = min(20 K, 2000)); TD R0 with 20K samples per molecule K-means-clustered to 2K (one
  sampling seed).
- Controls: A0 (ETKDG only, 0.2298 Å), R0 (0.1752 Å), A2 (0.1507 Å), OMEGA 0.177 Å as a literature row.
- Primary metric: AMR-R mean on the intersection; secondary AMR-P, COV-R/P at 0.05/0.1/0.5 Å.
- Expected (scout's guesses, no source): B-mmff between A0 and A2 (≈ 0.19–0.21 Å, i.e. the model adds ~0.04–0.06 Å
  on MMFF L); B-xtb ≤ B-mmff,
  possibly below 0.15 Å on rigid molecules (uncertain, no published number); B-clust better recall than A0 but worse
  precision than R0; TD + clustering better recall than R0.
- Cost: CPU ≈ 1–3 h (B-mmff), ≈ 25–70 CPU-h (B-xtb, full set × 3 seeds ≈ 82k optimisations at an assumed 1–3 s each, UNVERIFIED), ≈ 5 CPU-h (B-clust); GPU ≈ 3–4 GPU-h for the
  TD-oversampled arm (10× the samples of one ~20-min inference run).

## 6. Scout's own call (scout)
Worth trying: YES for B-mmff and B-xtb (cheap, and they define what "better L" must beat); MAYBE for B-clust (only as a
matched-budget sanity check of recall claims, not as a target).

Revision history (scout, P2): Revised after D-002, D-003, D-005 and V1 notes (blog scope, xtb 6.2.3, rigid vs
few-conformer, what Zhou's ablation shows about MMFF, B-clust wording). The calls are unchanged.

## 7. Code grounding (grounder)
- **B-mmff: AS SPECIFIED, zero code.**
  - `generate_confs.py --no_model --no_random --pre_mmff --seed {0,1,2}` (round-1 A0 line + `--pre_mmff`,
    `ablations_inference.tsv:13`).
  - Or, paired with CTRL: `--seed_confs L_etkdg2L_mmff.pkl --seed_confs_cycle --no_model --no_random`.
  - CPU job; `--model_dir` is still needed for the yaml (`generate_confs.py:85-88`).
- **B-xtb: WITH CHANGES.** It is C-GEOM-01's `L_etkdg2L_mmff_xtb` with `--no_model --no_random`, built with builder
  seeds 1–3 (`make_l_seed_pickles.py:59`). xtb 6.7.1 is already on gnode118 (no install). Measured 0.14 s/opt gives
  ≈ 6 CPU-h for 3 seeds (the card said 25–70). Needs the fixed xTB runner and the 3D graph/stereo filter: the
  evaluator does not filter generated conformers (`evaluate_confs.py:86`).
- **B-clust: WITH CHANGES.** New `tools/cluster_baseline.py` (~150 lines). The random-dihedral sampler is the
  `--no_model` path. **The K-means feature space is not fixed by our evidence** (E-EVAL-025); pre-register it and
  output the members nearest each centroid.
- **TD + clustering** needs `--confs_factor` in `generate_confs.py:183-187` (~3 lines); ≈ 2.6 GPU-h.
- **Slip (V1 R2-2):** "(30 % of all molecules) are rigid" should read "25 % of all molecules (few-conformer ∩ rigid);
  30 % of molecules are rigid overall".
- **Verdict:** B-mmff AS SPECIFIED; B-xtb, B-clust and TD+cluster IMPLEMENTABLE WITH CHANGES.
- **Cost:** ≈ 2.6 GPU-h; ≈ 10–13 CPU-h.
Detail, line numbers and cluster checks: `round3/grounding.md` §3 (shared pieces S-1…S-6 in §2.3).
## 8. Predicted effect on our project (analyst)
Data and model: `round3/impact.md` §3 (`impact_data/cond_means.txt`).
- **Ceiling.** n/a (baselines).
- **Expected effect.**
  - **B-mmff.** The rigid stratum equals the MMFF seeds, 0.128 (model-free, S1). Flexible bins start from A0 (ETKDG
    only: 0.197 / 0.284 / 0.328 / 0.391), relaxed toward MMFF minima; A3 (TD then MMFF-after: 0.17 / 0.23 / 0.25 /
    0.29) is the nearest analogue. Expected B-mmff ≈ 0.19–0.21, against A0 0.230 and TD + MMFF-before 0.151.
  - **B-xtb.** Rigid stratum ≈ xTB seeds, 0.10–0.13 (impact.md §3d, X = 30–35 %). Flexible molecules relax their
    torsions into xTB minima, which are GT basins when the seed is close. Overall 0.16–0.20; P(≤ 0.1525, i.e. it
    becomes the bar) ≈ 20 %.
  - **B-clust / TD + clustering.** Buys recall. Zhou's oversampling gain was ≈ 0.03 Å MAT on the GD protocol; for TD
    I expect −0.005 to −0.02 AMR-R with worse precision.
- **Detectability.** Differences vs TD (≥ 0.02) are clear.
- **Decision value.** High. Without B-mmff and B-xtb, the torsion model's contribution on good L (and so any FlexiTors
  L gain) cannot be attributed.
- **Worth trying: YES** for B-mmff and B-xtb (CPU; B-xtb shares C-GEOM-01's install). **MAYBE** for B-clust
  (3–4 GPU-h, a recall sanity check only).

## 9. Panel (judges)
