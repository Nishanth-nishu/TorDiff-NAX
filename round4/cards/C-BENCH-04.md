# C-BENCH-04: Leak-free oracle protocol (split-half other-conformer L) as the only reported true-geometry ceiling

## 1. Idea (scout)
In the paper, report a true-geometry ceiling only from the split-half design already built in round 4: GT conformers
of each multi-conformer molecule are split into halves A/B, seeds (L) come from one half and recall is scored on the
other half, both directions averaged; the same run with seeds from the scored half gives the leaky "own" value, so
leak = own − split is measured per molecule. Add three reporting rules: (i) state the universe (n_clean ≥ 2, ≈ 76 % of
molecules) and print the excluded single-conformer molecules separately; (ii) print every model beside the
model-free COPY baseline (other half's GT conformers used as predictions) and stratify by the minimum L-distance
between halves to show residual leakage through near-duplicate conformers; (iii) never use these numbers for model
selection, and drop own-conformer (`gtL`, `gtLcycle`) values from the paper except as the "leaky" comparison column.

## 2. Where it comes from (scout)
- Precedent: TD App. F.1 measured matching a GT conformer with another randomly chosen GT conformer's local structure
  (0.284 Å on DRUGS vs 0.324 Å for RDKit L), i.e. TD's authors already treat "another conformer's L" as the honest
  reference-level L [E-BENCH-061]. The only generative analogue we found is Ezzine et al.'s test on unseen local
  structures (MD, Boltzmann target) [E-BENCH-065]. No paper read defines a split-half generation oracle (section Z).
- Our leak: `gtL` draws seeds with replacement from the molecule's own GT conformers, ≈ 92 % of targets see their own
  L, and B1's true-L advantage is almost entirely conformer identity (inferred, not measured) [E-BENCH-062].
- The implementation exists (alternating split over evaluator-cleaned conformers, n_clean = 1 excluded and counted,
  COPY baseline, pre-declared contrasts) [E-BENCH-063].
- Universe: 225 of 940 molecules (24 %) have one GT conformer; they are 98 % ring-containing and have the highest
  RDKit-L floor [E-BENCH-064].
- GT conformers of one molecule share bonds/angles almost exactly (0.004 Å, 1.67°) [E-BENCH-073], so the other half's
  L differs from own L mainly in ring puckers and coupled angles.

## 3. Why it could matter here (scout)
- The "true geometry" numbers that motivated rounds 2–4 (B1 0.037 Å, S3 0.033 Å with own-conformer L; round4/BRIEF.md) are
  suspected leakage [E-BENCH-062]; round-2's final analysis shows S3 is better than CTRL only on exact own-conformer
  true L (round4/BRIEF.md, Round-2 FINAL) and worse on RDKit L by +0.0032 Å [E-BENCH-067]. A reviewer who spots an
  own-conformer oracle will discount every "ceiling" claim; the split-half table is the defensible version.
- It answers the brief's open question (how much real headroom better L gives) on ≈ 715 molecules without inverting
  P0 (the round-3 estimate amplified noise 8–14×; round3/impact.md §3g).

## 4. Assumptions that may not transfer (scout)
- Residual leak: CREST ensembles contain near-duplicate conformers (rotamers of terminal groups, symmetric
  equivalents), so the other half can contain a conformer whose L is nearly the target's; COPY and the L-distance
  stratification bound this, they do not remove it.
- Half-ensembles: each run generates 2|B| conformers against |B| targets, so absolute split-half AMR-R is not
  comparable to full-ensemble AMR-R or to published numbers; only paired contrasts (own vs split, model vs model,
  model vs COPY) are meaningful.
- The excluded single-conformer 24 % are exactly the rigid, ring-rich molecules where ring geometry matters most; the
  ring claim on them must come from a non-oracle L source (xTB/template seeds) instead.
- Alternating split by stored order is deterministic but not random; if GEOM stores conformers by energy, halves are
  energy-interleaved, which is fine but should be stated.

## 5. Minimal experiment (scout)
- Already scheduled by the coding agents (`slurm/r4_leakfree_array.sbatch`, `tools/analyze_leakfree.py`): 4 conditions
  × 11 model runs (CR 3, B1 3, S3 2, S4 3 seeds) = 44 packed inference conditions × ≈ 0.17 GPU-h ≈ 7.5 GPU-h
  [E-BENCH-058], plus CPU for COPY.
- Additions proposed here (CPU only, ≈ 0.5 day code): per-molecule minimum heavy-atom L-RMSD between halves (bonds,
  angles, ring dihedrals), and a split table stratified by its tertiles; a single-conformer table on non-oracle L
  sources.
- Primary metric: AMR-R (and COV-R@0.5) of split-half, paired; expected direction: B1 and S3 lose most of their
  own-L advantage (round-3 inversion predicts B1 0.16–0.35 Å vs 0.02–0.05 own).

## 6. Scout's own call (scout)
Worth trying: YES — it is the only oracle a reviewer will accept, it is already implemented, and the literature offers
no ready-made alternative (TD F.1 is the precedent to cite).

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
