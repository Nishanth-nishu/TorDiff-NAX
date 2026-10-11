# C-EVAL-01: Literature-comparable reporting block and "how good must L be to reach SOTA" analysis

## 1. Idea (scout)
Attach one fixed reporting block to every round-3 result so that our numbers can be placed against current QM9
papers without hidden protocol differences: AMR-R/AMR-P (mean, median) and COV-R/COV-P at δ ∈ {0.05, 0.1, 0.25, 0.5} Å,
failure count, and AMR on the success intersection of all compared arms. Add one CPU analysis on the existing round-2
λ-sweep: how much L improvement (in measured L error, not λ) a model needs before its AMR-R reaches the published
Cartesian models (ET-Flow 0.073–0.083 Å depending on the run, DMT-B 0.090 Å, MCF-B 0.103 Å), using the random-GT-conformer oracle as the realistic
ceiling and the cycled oracle only as an upper bound.

## 2. Where it comes from (scout)
- Published GEOM-QM9 numbers at δ = 0.5 Å (TD protocol): ET-Flow AMR-R 0.073 Å / COV-R 96.47 % [E-EVAL-001];
  MCF-B 0.103 Å / 95.0 % [E-EVAL-002]; DMT-B 0.090 Å / 95.2 % and OMEGA 0.177 Å [E-EVAL-003]; AvgFlowDiT 0.082 Å, RDKit
  0.235 Å [E-EVAL-004]; S23D-B 0.090 Å, evaluated on 995 molecules [E-EVAL-005]. The TD row everyone copies is 0.178 Å /
  92.80 % [E-EVAL-001].
- QM9 at δ = 0.5 Å is argued to be saturated (mean AMR < 0.1 Å) [E-EVAL-006]; the FM-refiner paper therefore reports
  COV at δ = 0.05 Å [E-EVAL-007], where its own re-runs give MCF-B 66.82 % (AMR-R 0.101 Å), ET-Flow 75.72 % (AMR-R
  0.083 Å) and the best pipeline 79.50 % (AMR-R 0.070 Å) [E-EVAL-008]. (Revised after D-006: these δ = 0.05 Å numbers
  belong to FM-refiner's re-runs, not to the published ET-Flow 0.073 Å run; the "ET-Flow" target is therefore a range,
  0.073 Å published [E-EVAL-001] to 0.083 Å re-run [E-EVAL-008].) ET-Flow's own threshold plots show its advantage over TD is largest at low thresholds (DRUGS)
  [E-EVAL-010].
- Accounting differences: our evaluator scores a failed molecule as 0 % coverage but drops it from AMR (nanmean)
  [E-EVAL-033]; TD fails on 65 molecules (60 strained cages) and the published 92.8 % COV-R is not reproducible from the
  paper's own released samples (88.8 %) [E-EVAL-046]. (Revised after D-006: an inference linking GEOM's reacted QM9
  graphs to our cage failures was withdrawn; nothing connects the two.)
- The infrastructure exists: `paired_compare.py` already takes δ ∈ {0.05, 0.1, 0.25, 0.5} and an intersection universe
  [E-EVAL-037].

## 3. Why it could matter here (scout)
- **Where we sit (δ = 0.5 Å, AMR-R mean).** TD baseline 0.1752 Å on 935 molecules with 65 failures [E-EVAL-038];
  MMFF-before 0.1507 Å [E-EVAL-040] (0.1518 Å with our retrained CTRL [E-EVAL-050]); S3 on RDKit L 0.1816 Å
  [E-EVAL-043]. So every non-oracle arm sits at OMEGA/TD level (0.15–0.18 Å vs OMEGA 0.177 Å [E-EVAL-003]), well behind MCF-B
  (0.103 Å) and ET-Flow (0.073–0.083 Å). Only ORACLE arms reach or beat the literature: true L + released model 0.0801 Å [E-EVAL-042], S3 on cycled
  true L 0.0331 Å [E-EVAL-043], B1 on cycled true L 0.0199 Å [E-EVAL-044].
- **At δ = 0.05 Å the gap is an order of magnitude.** TD 3.91 % [E-EVAL-038], MMFF-before 17.89 % [E-EVAL-040], S3 on
  RDKit L 3.56 % [E-EVAL-043] vs 66.8–79.5 % for FM-refiner's runs (AMR-R 0.070–0.101 Å in the same runs) [E-EVAL-008];
  the true-L oracle with the released model gives 66.72 % [E-EVAL-042], about the level of FM-refiner's MCF-B run.
  Different evaluators: compare orders of magnitude only.
- **How good must L be?** With L 75 % of the way to cycled true L (ORACLE), CTRL gives 0.1164 Å and B1 0.1158 Å;
  at λ = 1 CTRL gives 0.0803 Å and B1 0.0199 Å [E-EVAL-044]. INFERENCE (Revised after D-006; linear interpolation of
  B1 seed 0 between λ = 0.75 and 1.00, recomputed by the scout and by V1): B1 reaches the ET-Flow range 0.083–0.073 Å
  at λ ≈ 0.84–0.86, DMT-B's 0.090 Å at λ ≈ 0.82 and MCF-B's 0.103 Å at λ ≈ 0.78; CTRL reaches 0.103 Å only near
  λ ≈ 0.84 and never reaches the ET-Flow range (0.0803 Å at λ = 1). The λ = 0.75 L already has bond / angle /
  ring-dihedral error of 0.0097 Å / 0.97° / 1.56° [E-EVAL-049]. MMFF L (0.0182 Å / 1.97° / 8.07°) is far from that [E-EVAL-049].
- **Oracle leak.** B1 on cycled true L gives 0.0215 Å but on true L from a random GT conformer 0.0374 Å [E-EVAL-045];
  the cycled number partly measures conformer identity carried by L, so it is not a reachable ceiling.
- INFERENCE for the panel: on QM9, a FlexiTors L sampler is only "competitive with the state of the art" if it gets
  most of the way to the oracle; a gain from 0.176 to, say, 0.13 Å is real progress for TD-family methods but still
  leaves the method behind every Cartesian model in E-EVAL-001…005. The report should say this plainly.

## 4. Assumptions that may not transfer (scout)
- Literature rows use different populations (1000 vs 995 molecules [E-EVAL-005]; ours 935 successes [E-EVAL-038]),
  different training-conformer caps and chirality handling [E-EVAL-005], and most papers copy baseline rows instead of
  re-running them [E-EVAL-004]. Only differences larger than these protocol effects should be read.
- TD drops the 65 molecules ETKDG cannot embed from AMR [E-EVAL-033, E-EVAL-046]. The direction of the resulting bias
  against Cartesian models is unknown: they too evaluate reduced sets (995 molecules for MCF/S23D [E-EVAL-005]) and
  ET-Flow reports RDKit-related generation failures on GEOM-XL [E-EVAL-058]. (Revised after D-006: the earlier claim
  that our AMR-R is "flattered" was withdrawn.)
- λ is a Cartesian blend toward the cycled GT conformer, not a natural L-error axis; the interpolation is seed 0 and
  linear, and the λ = 0.75 runs are scored on 906 molecules while the λ = 1.00 runs are scored on 955 [E-EVAL-044], so
  the crossing mixes populations (V1's bounds for this: 0.851–0.863; quadratic fit 0.872); λ = 1.00 exists only for
  B1 seed 0. The proposed per-molecule regression on measured L error fixes this.
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
  (CTRL, B1; S3 once it has a λ sweep). Expected: B1 needs roughly λ ≥ 0.84-level L for the ET-Flow range; CTRL never
  reaches it even at λ = 1 (0.0803 Å). (Revised after D-006: S3 removed from this expectation, it has no λ sweep.)
- Cost: 0 GPU-h; ~2 CPU-h.

## 6. Scout's own call (scout)
Worth trying: YES — free, uses existing pickles, and it sets the bar every other round-3 card must clear to be "worth
trying relative to the state of the art".

Revision history (scout, P2): Revised after D-006. Original §1/§3 used ET-Flow 0.073 Å as the single target while
quoting the δ = 0.05 Å coverage of a 0.083-Å run; original §2 inference on reacted graphs vs cages and §4 "flattered"
claim withdrawn; original §5 included S3 in the λ expectation. The call is unchanged.

## 7. Code grounding (grounder)
- **Touchpoints:**
  - new `tools/report_block.py` (~80 lines) on top of `paired_compare.py` (thresholds `:32`, intersection
    `:41`/`:105-106`, failures `:107-113`);
  - new `tools/l_error_regression.py` (~120 lines), which joins `eval.pkl` per-molecule AMR (keyed by corrected
    SMILES) to `l_error.csv` (keyed by csv col 0) via `analyze_round2.py:166-167`, with an assert of ≥ 99 % join
    coverage.
- **Transfer check:**
  - Literature δ = 0.5 Å AMR means over successes match our `nanmean`/failure convention (`evaluate_confs.py:166-171`,
    [E-EVAL-033]).
  - Three changes are needed for the L-error analysis:
    1. fit **within molecules** across the λ series; a cross-molecule regression confounds L error with molecule
       difficulty;
    2. use one L-error statistic: ETKDG/MMFF rows cover only the L Hungarian-assigned seeds
       (`make_l_seed_pickles.py:106-114`), while λ rows cover every seed;
    3. restrict to the 906-molecule λ population and exclude or flag the 258 molecules whose λ 0.25–0.75 slots reuse
       another pair's seed. Their error is measured against the source GT (`make_l_seed_pickles.py:218-231`; found by
       the impact analyst).
  - λ = 1 exists for CR ×3 and B1 s0 only.
  - CR has no random-GT run (optional, 0.5 GPU-h).
- **Slip (V1 R2-1):** "CTRL never reaches the ET-Flow range" is wrong as written, since 0.0803 Å is inside
  0.073–0.083. Correct: it reaches the 0.083 Å re-run level only at λ ≈ 0.98 and never reaches the published 0.073 Å.
- **Reuse:** all existing `eval.pkl` files, `l_error.csv`, `analyze_round2.py` helpers.
- **Size / risk:** ~200 lines + 1 synthetic test. Risks: key mis-join; reading the cycled oracle as reachable.
- **Verdict: IMPLEMENTABLE WITH CHANGES.**
- **Cost:** 0 GPU-h (+0.5 optional); 1–2 CPU-h.
Detail, line numbers and cluster checks: `round3/grounding.md` §3 (shared pieces S-1…S-6 in §2.3).
## 8. Predicted effect on our project (analyst)
Data and model: `round3/impact.md` §1, §3f (`impact_data/mixture_model.out`). Round-2 numbers are PRELIMINARY.
- **Ceiling.** n/a (analysis, 0 GPU-h).
- **Already computable from synced data** (reference-like acyclic L, ETKDG population):
  - CTRL's floor is 0.086 (random-GT-conformer L). It reaches neither the published ET-Flow 0.073 nor the 0.083
    re-run level. The cycled λ = 1 value (0.080) includes conformer identity.
  - B1 reaches 0.073 at a ring tail ≤ ≈ 14 %, 0.083 at ≤ 18 %, DMT-B 0.090 at ≤ 21 %, and MCF-B 0.103 at ≤ 26 %
    (> 10° share, 49.6 % basis).
  - These replace the λ interpolation (λ ≈ 0.84–0.86). In λ ∈ (0,1), 10 % of slots (258 molecules) reuse another GT
    conformer's seed, and their l_error rows are measured against the source GT, so λ is not a clean L axis
    (impact.md §1).
- **Expected effect.** The per-molecule regression proposed in §5 needs the cluster's eval pickles or
  `breakdown.csv` (not synced). I expect it to confirm that no CTRL arm reaches the published state of the art on any
  L, and that B1-type models need a tail ≲ 15 % plus reference-like acyclic L. Worst case: nothing beyond this, at no
  GPU cost.
- **Detectability.** n/a.
- **Decision value.** High: it sets the bar. Any CTRL-based FlexiTors arm is capped at ≈ 0.086 on QM9, so a
  state-of-the-art claim needs a B1/S3-type model with near-reference L.
  - Suggested change: use the random-GT-conformer oracle (`gtL`) as the reachable ceiling, and either drop the λ
    interpolation or restrict it to non-replaced slots.
- **Worth trying: YES.**

**Revised after cross-read (2026-10-11).**
1. **Leak.** The thresholds above (B1 reaches 0.073 at a tail ≤ 14 %; CTRL floor 0.086) assume every target gets its
   own conformer's L, and the random-GT anchor is mostly leaky (grounder 5.2.1). Leak-corrected (impact.md §3g): with
   other-conformer L only, the CTRL floor is 0.11–0.145 and B1's 0.19–0.27, so no literature level is reachable.
   Reaching 0.073–0.103 needs an L source that puts seeds into the targets' own basins.
2. **Fit population.** I agree with the grounder: a within-molecule fit, with the 648 molecules without replaced λ
   slots as primary and the 906 as sensitivity. The λ axis is cycled (own-conformer identity), so add the leak
   inversion.
3. **Feasibility.** Possible on the cluster at CPU cost.

Call unchanged: **YES**. Its value goes up: it now also states that the state of the art needs own-basin L.

## 9. Panel (judges)
