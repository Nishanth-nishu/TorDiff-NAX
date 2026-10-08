# C-EVAL-01: Literature-comparable reporting block and "how good must L be to reach SOTA" analysis

## 1. Idea (scout)
Attach one fixed reporting block to every round-3 result so that our numbers can be placed against current QM9
papers without hidden protocol differences: AMR-R/AMR-P (mean, median) and COV-R/COV-P at δ ∈ {0.05, 0.1, 0.25, 0.5} Å,
failure count, and AMR on the success intersection of all compared arms. Add one CPU analysis on the existing round-2
λ-sweep: how much L improvement (in measured L error, not λ) a model needs before its AMR-R reaches the published
Cartesian models (ET-Flow 0.073 Å, DMT-B 0.090 Å, MCF-B 0.103 Å), using the random-GT-conformer oracle as the realistic
ceiling and the cycled oracle only as an upper bound.

## 2. Where it comes from (scout)
- Published GEOM-QM9 numbers at δ = 0.5 Å (TD protocol): ET-Flow AMR-R 0.073 Å / COV-R 96.47 % [E-EVAL-001];
  MCF-B 0.103 Å / 95.0 % [E-EVAL-002]; DMT-B 0.090 Å / 95.2 % and OMEGA 0.177 Å [E-EVAL-003]; AvgFlowDiT 0.082 Å, RDKit
  0.235 Å [E-EVAL-004]; S23D-B 0.090 Å, evaluated on 995 molecules [E-EVAL-005]. The TD row everyone copies is 0.178 Å /
  92.80 % [E-EVAL-001].
- QM9 at δ = 0.5 Å is argued to be saturated (mean AMR < 0.1 Å) [E-EVAL-006]; the FM-refiner paper therefore reports
  COV at δ = 0.05 Å [E-EVAL-007], where MCF-B reaches 66.82 %, ET-Flow 75.72 % and the best pipeline 79.50 %
  [E-EVAL-008]. ET-Flow's own threshold plots show its advantage over TD is largest at low thresholds (DRUGS)
  [E-EVAL-010].
- Accounting differences: our evaluator scores a failed molecule as 0 % coverage but drops it from AMR (nanmean)
  [E-EVAL-033]; TD fails on 65 molecules (60 strained cages) and the published 92.8 % COV-R is not reproducible from the
  paper's own released samples (88.8 %) [E-EVAL-046]. INFERENCE: GEOM attributes its QM9 graph changes during CREST to
  strained molecules [E-EVAL-017], the same class ETKDG cannot embed, so reference quality and seed availability are
  both weakest on the subset TD drops from AMR.
- The infrastructure exists: `paired_compare.py` already takes δ ∈ {0.05, 0.1, 0.25, 0.5} and an intersection universe
  [E-EVAL-037].

## 3. Why it could matter here (scout)
- **Where we sit (δ = 0.5 Å, AMR-R mean).** TD baseline 0.1752 Å on 935 molecules with 65 failures [E-EVAL-038];
  MMFF-before 0.1507 Å [E-EVAL-040] (0.1518 Å with our retrained CTRL [E-EVAL-050]); S3 on RDKit L 0.1816 Å
  [E-EVAL-043]. So every non-oracle arm sits at OMEGA/TD level (0.15–0.18 Å vs OMEGA 0.177 Å [E-EVAL-003]), well behind MCF-B
  (0.103 Å) and ET-Flow (0.073 Å). Only ORACLE arms reach or beat the literature: true L + released model 0.0801 Å [E-EVAL-042], S3 on cycled
  true L 0.0331 Å [E-EVAL-043], B1 on cycled true L 0.0199 Å [E-EVAL-044].
- **At δ = 0.05 Å the gap is an order of magnitude.** TD 3.91 % [E-EVAL-038], MMFF-before 17.89 % [E-EVAL-040], S3 on
  RDKit L 3.56 % [E-EVAL-043] vs 66.8–79.5 % in the literature [E-EVAL-008]; the true-L oracle with the released model
  gives 66.72 % [E-EVAL-042], i.e. about MCF-B level.
- **How good must L be?** With L 75 % of the way to cycled true L (ORACLE), CTRL gives 0.1164 Å and B1 0.1158 Å; B1
  reaches 0.0199 Å at λ = 1 [E-EVAL-044]. INFERENCE (linear interpolation of B1 between λ = 0.75 and 1.0, seed 0):
  B1 crosses ET-Flow's 0.073 Å only at λ ≈ 0.86, and the λ = 0.75 L already has bond / angle / ring-dihedral error of
  0.0097 Å / 0.97° / 1.56° [E-EVAL-049]. MMFF L (0.0182 Å / 1.97° / 8.07°) is far from that [E-EVAL-049].
- **Oracle leak.** B1 on cycled true L gives 0.0215 Å but on true L from a random GT conformer 0.0374 Å [E-EVAL-045];
  the cycled number partly measures conformer identity carried by L, so it is not a reachable ceiling.
- INFERENCE for the panel: on QM9, a FlexiTors L sampler is only "competitive with the state of the art" if it gets
  most of the way to the oracle; a gain from 0.176 to, say, 0.13 Å is real progress for TD-family methods but still
  leaves the method behind every Cartesian model in E-EVAL-001…005. The report should say this plainly.

## 4. Assumptions that may not transfer (scout)
- Literature rows use different populations (1000 vs 995 molecules [E-EVAL-005]; ours 935 successes [E-EVAL-038]),
  different training-conformer caps and chirality handling [E-EVAL-005], and most papers copy baseline rows instead of
  re-running them [E-EVAL-004]. Only differences larger than these protocol effects should be read.
- TD drops the 65 molecules ETKDG cannot embed from AMR [E-EVAL-033, E-EVAL-046]. Cartesian generators do not start
  from ETKDG seeds, so (INFERENCE, not checked per paper) their AMR includes these hard molecules; our AMR-R is
  therefore flattered relative to ET-Flow's, not penalised.
- λ is a Cartesian blend toward the cycled GT conformer, not a natural L-error axis; the interpolation is seed 0 and
  linear, and the λ = 0.75 runs are scored on 906 molecules while the λ = 1.00 runs are scored on 955 [E-EVAL-044], so
  the "λ ≈ 0.86" figure mixes populations. The proposed per-molecule regression on measured L error fixes this.
- At δ = 0.05 Å every RDKit-L arm sits at 3–4 % [E-EVAL-038, E-EVAL-043], so 0.05 Å only separates learned-L or
  oracle arms; and because L differs between GT conformers of one molecule (0.004 Å bonds, 1.67° angles
  [E-EVAL-047]), even a perfect conformer-independent L sampler would not reach 100 % there.

## 5. Minimal experiment (scout)
- Arms: existing eval pickles of R0 (3 seeds), A2, A1 (ORACLE), CTRL_rematch ×3 on ETKDG / MMFF / λ-sweep / gtL /
  gtLcycle, B1 ×3 on the same, S3 s0/s1.
- Analysis (CPU only): `paired_compare.py --universe intersection --thresholds 0.05 0.1 0.25 0.5` for each family;
  a literature table with our rows and the published rows of E-EVAL-001…008, failure counts shown; per-molecule AMR-R
  regressed on per-molecule measured L error from `l_error.csv` to read off the L error at which each model reaches
  0.073 / 0.090 / 0.103 Å (bootstrap CI over molecules).
- Primary readout: L error (bond Å, angle °, ring dihedral °) needed to reach each literature level, per model
  (CTRL, B1, S3). Expected: B1/S3 need roughly λ ≥ 0.85-level L; CTRL never reaches 0.073 Å even at λ = 1 (0.0803 Å).
- Cost: 0 GPU-h; ~2 CPU-h.

## 6. Scout's own call (scout)
Worth trying: YES — free, uses existing pickles, and it sets the bar every other round-3 card must clear to be "worth
trying relative to the state of the art".

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
