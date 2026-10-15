# C-BENCH-07: NO: GEOM-Drugs-Revisited stability metrics, CREST-weighted Boltzmann properties on QM9, or GD-protocol numbers as headline evaluation

## 1. Idea (scout)
Three popular-looking evaluation additions that we should *not* adopt as headline metrics: (a) the "corrected"
molecule-stability / valency evaluation of GEOM-Drugs Revisited; (b) Boltzmann-weighted ensemble properties on
GEOM-QM9 using CREST weights; (c) GD-protocol (GeoDiff/ConfGF, 200 test molecules, δ = 1.25 Å on DRUGS) or
RDKit + Clustering-style oversampled numbers mixed into our TD-P tables. Also not: adopting ET-Flow's evaluator as our
evaluator (it drops failed molecules).

## 2. Where it comes from (scout)
- (a) GEOM-Drugs Revisited's evaluation bug is in the valency/stability metric for de novo 3D generation
  [E-BENCH-029]; its energy benchmark is the transferable part [E-BENCH-030, E-BENCH-037].
- (b) GEOM's CREST weights are "rather inaccurate" and DFT weights exist only for 1,511 BACE species [E-BENCH-048];
  TD and ET-Flow run properties only on a DRUGS subset after xTB relaxation [E-BENCH-041]; even then the subset is not
  reproducible across papers [E-BENCH-042].
- (c) TD-P vs GD-P differ in split, test size and DRUGS threshold [E-BENCH-002, E-BENCH-004]; Zhou's cheap baseline
  numbers are on GD-P with ≈ 2000 samples [E-BENCH-049] and their fairness depends on equal sample budgets
  [E-BENCH-050].
- ET-Flow's evaluator skips failed and multi-fragment molecules [E-BENCH-008].

## 3. Why it could matter here (scout)
- Conformer generation receives the molecular graph, so valency/stability is fixed by the input; the metric cannot
  distinguish TD, our method or ET-Flow (INFERENCE from E-BENCH-029).
- QM9 property errors would be weighted by inaccurate CREST weights [E-BENCH-048] and would mostly measure relaxation,
  not generation quality (TD's E_min error is 36.94 kcal/mol unrelaxed vs 0.13 after xTB relaxation on DRUGS [E-BENCH-041]).
- Mixing GD-P or 2000-sample numbers into TD-P tables, or adopting a failure-dropping evaluator, would flatter TD-family
  methods by ≈ 4 COV-R points on QM9 (our 65 failures; E-BENCH-011, E-BENCH-069) and invite exactly the comparability
  objections in E-BENCH-049/050.

## 4. Assumptions that may not transfer (scout)
- If a later method generates the molecular graph or changes bonding (it should not), stability would become relevant.
- If GEOM later releases DFT weights for QM9, (b) could be revisited; none exist now [E-BENCH-048].
- ET-Flow's convention is still useful as a *secondary* view (the success-intersection table in C-BENCH-01 serves that
  purpose without changing the primary).

## 5. Minimal experiment (scout)
- None. Keep: GFN2-xTB E_relax and geometry deviations (C-BENCH-03), DRUGS-only ensemble properties (C-BENCH-05), the
  TD failure convention as primary with an intersection table as secondary (C-BENCH-01).
- Cost saved: the valency tables and property pipeline for QM9 (≈ 1–2 days of code, ≈ 10 CPU-h of xTB singlepoints).

## 6. Scout's own call (scout)
Worth trying: NO — none of the three measures what our method changes, (b) rests on weights the dataset authors call
inaccurate, and (c) breaks comparability with the TD-P tables we must compete in.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
