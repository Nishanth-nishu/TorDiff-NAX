# C-BENCH-01: QM9 reporting standard: one evaluator, both failure conventions, fine thresholds, two-level CIs

## 1. Idea (scout)
Fix, before any new result exists, how every QM9 number in the paper is produced and printed: TD protocol split and
2K sampling; every method's generated conformers scored by our (TD's) evaluator; δ = 0.5 Å AMR-R / COV-R as primary
(user ruling) with COV-R@0.1 on the success intersection as key secondary, plus a coverage-vs-threshold curve
(0.05–1.0 Å) and a δ = 0.05 Å column; each comparison table printed twice, once in the TD convention (failure = 0 %
coverage, AMR over successes) and once on the molecule intersection shared by all rows; 3 training seeds per learned
arm with the two-level (seeds, then molecules) bootstrap we already run; per-row columns for training conformers per
molecule (KG), chirality handling, parameter count, training/test L source and sample budget; and stratified tables
(rotatable bonds, ring content, number of GT conformers).

## 2. Where it comes from (scout)
- The TD protocol: GeoMol's random 106586/13323/1000 QM9 split with SMILES/RDKit filtering [E-BENCH-001], 2K
  conformers per molecule with K GT conformers [E-BENCH-002], δ = 0.5 Å on QM9 in both conventions [E-BENCH-004],
  KG ≤ 30 in training but all CREST conformers at test [E-BENCH-003].
- Published tables silently differ in comparability factors: KG 20 vs 30 and chirality handling [E-BENCH-005], top-30 by
  Boltzmann weight in ET-Flow's current preprocessing [E-BENCH-006], 995 instead of 1000 test molecules for MCF/S23D
  [E-BENCH-010].
- Failure handling differs between evaluators: TD's counts failures as 0 % coverage and drops reacted GT conformers
  [E-BENCH-007]; ET-Flow's skips failed or multi-fragment molecules entirely [E-BENCH-008]; RMSD definition is the same
  (heavy-atom GetBestRMS) [E-BENCH-009].
- Fine thresholds: QM9 at 0.5 Å is called saturated, and the FM-refiner paper reports δ = 0.05 Å [E-BENCH-047]
  with values for ET-Flow (75.72 %), DMT (72.90 %), MCF-B (66.82 %) [E-BENCH-019]; a paired per-conformer
  improvement/downgrade rate is a published, cheap addition [E-BENCH-046].
- Statistics: no TD-P RMSD table reports training-seed variation (section Z); GeoMol reports 3 sampling runs
  (±0.0012 Å AMR-R on QM9) [E-BENCH-023], EnFlow 10 runs [E-BENCH-044]. Our round-2 driver already does paired
  bootstrap, Holm, intersection secondary, seed SDs and hierarchical bootstrap [E-BENCH-066, E-BENCH-070].
- TD itself stratifies AMR by rotatable bonds and notes that rigid molecules' error is set by L [E-BENCH-045].
- The fairness question about TD's training L source was raised publicly [E-BENCH-036].

## 3. Why it could matter here (scout)
- Our effects are small: S3 − CTRL on RDKit L is +0.0032 Å [+0.0013, +0.0052] [E-BENCH-067]; S5 − CTRL with MMFF at
  test is −0.0041 Å [E-BENCH-068]. They are detectable only because the analysis is paired, pre-declared and
  seed-aware; the paper must keep that machinery.
- Failure accounting alone moves TD's QM9 COV-R from the published 92.8 % to 89.0 % on its own released samples
  (65 failures) [E-BENCH-011, E-BENCH-069], larger than any recipe effect we have. A table mixing published and
  re-scored numbers without saying so would be wrong by more than our claimed gains.
- Our real-use gain is from geometry (MMFF seeds 0.1507 vs TD 0.1752 Å AMR-R, COV-R@0.05 17.9 vs 3.9 %)
  [E-BENCH-069]; that gain is most visible at fine thresholds, so the threshold curve and δ ≤ 0.1 columns carry the
  scientific point while δ = 0.5 stays primary.
- INFERENCE: the TD-convention table penalises TD-family methods for the 60 RDKit cages; the intersection table
  shows method quality where all methods run. Printing both pre-empts both reviewer objections.

## 4. Assumptions that may not transfer (scout)
- Cartesian baselines (ET-Flow, MCF) have no RDKit failures on QM9 in their own pipelines; on our evaluator they may
  still lose molecules (multi-fragment SMILES, stereo parsing). The intersection must be computed after re-scoring,
  not assumed.
- The intersection excludes strained cages, which are ring-heavy; a ring-geometry method might look better or worse
  there. Stratifying by ring content shows this explicitly.
- Two-level bootstrap with 2–3 seeds gives wide seed components; with S3's 2 seeds the hierarchical interval is
  coarse. Three seeds is the floor for any arm in the main table.
- Reporting many thresholds invites multiple-testing complaints: keep the pre-declared primary/secondary split and
  label the curve descriptive.

## 5. Minimal experiment (scout)
- No GPU. Code only: extend the round-2/round-4 analysis driver (`tools/analyze_round2.py` pattern,
  `tools/paired_compare.py`) with (a) a "published comparison" block that prints our re-scored rows beside the
  literature rows with protocol columns (KG, chirality, params, n molecules, failure convention, source table/page);
  (b) coverage curves 0.05–1.0 Å in 0.05 steps from the eval pickles; (c) strata by n_rot_heavy, ring content and
  n_true (bins already exist in `breakdown.log`); (d) the IR/DR paired rate vs TD.
- Inputs: existing eval pickles; re-scored baselines from C-BENCH-02.
- Cost: ≈ 1 day of code + < 2 CPU-h.
- Primary metric and direction: unchanged (AMR-R mean, COV-R@0.5; lower/higher is better). This card changes
  reporting, not results.

## 6. Scout's own call (scout)
Worth trying: YES — cheap, removes the largest known source of non-comparability (failure convention, 3.8 COV
points), and most of the machinery already exists.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
