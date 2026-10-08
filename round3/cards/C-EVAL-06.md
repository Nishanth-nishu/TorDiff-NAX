# C-EVAL-06: Re-reference QM9 at DFT level (NO) and add Boltzmann ensemble-property errors (not for round 3)

## 1. Idea (scout)
Two popular "make the benchmark more chemical" moves, assessed together because both are proposed as answers to
"GEOM is only xTB": (a) re-optimise the GEOM-QM9 test references at a DFT level (or use QM9's original DFT
geometries) and re-score our arms against them; (b) add TD/ET-Flow-style Boltzmann-weighted ensemble-property errors
(E, dipole, HOMO–LUMO gap, E_min after GFN2-xTB relaxation) as a round-3 endpoint.

## 2. Where it comes from (scout)
- The GEOM-QM9 references are GFN2-xTB minima, re-optimised from QM9's DFT geometries [E-EVAL-013, E-EVAL-012]. On
  BACE, DFT re-optimisation moves CREST geometries by 0.36 Å mean RMSD (0.25 Å median heavy-atom) [E-EVAL-015]. Zhou et
  al. argue GEOM's semi-empirical accuracy is insufficient for applications needing DFT-quality conformers
  [E-EVAL-051].
- TD (100 DRUGS molecules, min(2K, 32) conformers, GFN2-xTB relaxation) [E-EVAL-024] and ET-Flow (same design)
  [E-EVAL-054] use Boltzmann-weighted ensemble properties as a chemical check alongside RMSD.
- GEOM's authors present GEOM as a recall/diversity benchmark, warn that CREST conformer weights are inaccurate, and
  say probability benchmarks should use GEOM's DFT weights [E-EVAL-016], which exist only for the 1,511 BACE species
  [E-EVAL-059]. (Revised after D-001: the recommendation was omitted before.)

## 3. Why it could matter here (scout)
- Many of our AMR-R differences are a few thousandths to hundredths of an Å (e.g. S3 vs CTRL on ETKDG L: 0.1816 vs
  0.1774 Å [E-EVAL-043, E-EVAL-050]); they are far below the xTB-vs-DFT geometry gap reported for drug-like molecules [E-EVAL-015]; one could argue that learning
  xTB's L more exactly is chasing the reference method's own error.
- However, for what round 3 must decide (which L source and training scheme to build on, and how we compare with the
  QM9 literature), neither (a) nor (b) is informative:
  - (a) breaks comparability with every published number [E-EVAL-001…008], all computed against the same xTB
    references, and our oracle analyses (true L = xTB L) would no longer be oracles. QM9's DFT geometry is one structure
    per molecule, so ensemble recall cannot be scored against it (INFERENCE from E-EVAL-013: GEOM-QM9's other
    conformers exist only at xTB level).
  - (b) Without relaxation the property errors are "far too large" to be meaningful [E-EVAL-023]; with GFN2-xTB
    relaxation they fall sharply (TD E_min 36.94 → 0.13 kcal/mol on DRUGS [E-EVAL-022]) and errors from global
    flexibility "become important" [E-EVAL-023]. Relaxation is local, so the ring pucker and torsion basin an L choice
    produces do survive it; the metric is therefore NOT insensitive to what FlexiTors changes (Revised after D-001:
    the earlier "by construction insensitive" was WITHDRAWN). The remaining reasons not to add it in round 3: for QM9
    the Boltzmann weights would come from CREST, which GEOM calls inaccurate, and GEOM's DFT weights exist only for BACE
    [E-EVAL-016, E-EVAL-059]; and whether relaxed conformers land in the right basins is measured more directly, without
    any weights, by re-scoring the relaxed structures against the references (C-EVAL-05) and by E_relax (C-EVAL-03).
- The useful part of the concern, "is the generated L physically right without looking at GT?", is captured more
  cheaply and more sharply by E_relax and displacement-to-own-minimum (C-EVAL-03), which keeps the xTB level the
  benchmark was built on.

## 4. Assumptions that may not transfer (scout)
- The 0.36 Å xTB→DFT shift was measured on drug-like BACE molecules in implicit water [E-EVAL-015]; for gas-phase QM9
  molecules the shift is probably much smaller, which weakens the motivation for (a) further.
- Ensemble properties might matter more on GEOM-DRUGS (where TD reports them [E-EVAL-024]); this NO is for the
  QM9-only phase.
- If a downstream application (e.g. QM property prediction from generated conformers) becomes a project goal, DFT
  references become relevant again (the Zhou et al. argument [E-EVAL-051]).

## 5. Minimal experiment (scout)
Not proposed. If the panel disagrees, the cheapest probe is: r2SCAN-3c (or B3LYP-D3/def2-SVP) re-optimisation of the
reference conformers of 50 test molecules, then the RMSD between xTB and DFT references per conformer; if the median is
below the AMR-R differences we care about (≤ 0.01 Å), (a) is moot. CPU cost not estimated (needs a DFT code on the
cluster; availability not checked).

## 6. Scout's own call (scout)
Worth trying: NO for (a) — re-referencing breaks comparability with every published number and the oracle analyses.
(b) not for round 3 (MAYBE for the DRUGS phase) — it is sensitive to basin choices, but on QM9 it rests on CREST
weights GEOM calls inaccurate, and C-EVAL-03/05 measure the same thing more directly.

Revision history (scout, P2): Revised after D-001. Original §3(b) and §6 argued that post-relaxation ensemble
properties are "by construction insensitive to local-structure changes", based on the overstated E-EVAL-023 reading
("dominate"); WITHDRAWN. The call for (b) changed from NO to "not for round 3 / MAYBE for DRUGS"; (a) stays NO.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
