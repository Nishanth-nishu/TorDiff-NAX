# C-EVAL-02: Local-geometry, stratified and conformer-level diagnostics for every round-3 arm

## 1. Idea (scout)
Report, next to AMR/COV, four cheap diagnostics computed from the existing eval and conformer pickles:
(a) bond-length, bond-angle and endocyclic ring-dihedral error of the *generated* conformers against their RMSD-matched
GT conformer (`tools/geometry_metrics.py`, extended with the ring-dihedral column of `l_error.csv`);
(b) AMR-R both per molecule (macro, current) and per GT conformer (micro), and by stratum (rigid / ring-containing /
flexible acyclic);
(c) a paired, conformer-level improvement / downgrade rate at tolerances 0.02–0.20 Å between arms that share the same
ETKDG seed conformers;
(d) the random-GT-conformer oracle reported next to the cycled one, so ceilings are not over-stated.

## 2. Where it comes from (scout)
- Nikitin et al. recommend bond, angle and torsion differences as interpretable local-geometry metrics
  [E-EVAL-019]. Their version compares each structure with its own GFN2-xTB minimum (that variant is C-EVAL-03); the
  vs-matched-GT version here is the oracle-side complement.
- The FM-refiner paper reports per-conformer improvement and downgrade rates on GEOM-QM9 at tolerances 0.02–0.20 Å
  [E-EVAL-009]. The reason given here (ensemble means hide whether a change helps most conformers a little or a few a
  lot) is the scout's INFERENCE, not FM-refiner's stated motivation (Revised after V1 note, P2).
- QM9 at δ = 0.5 Å is saturated, so finer views are needed to see local-structure effects [E-EVAL-006, E-EVAL-007].
- Our tooling already exists: `geometry_metrics.py` (vs matched GT) and `paired_compare.py` (intersection universe,
  fine thresholds) [E-EVAL-036, E-EVAL-037]; the round-1 analysis job calls `geometry_metrics.py`, but no output is
  in the synced results or any report [E-EVAL-036].

## 3. Why it could matter here (scout)
- **Averaging and strata change what the headline number measures** (Revised after D-005). Molecules with 1–3 GT
  conformers are 48 % of the per-molecule average but 10 % of conformers, and 97–98 % of them contain a ring
  [E-EVAL-048]; per-molecule vs per-conformer floor means are 0.1145 vs 0.0986 Å [E-EVAL-047]. Rigid molecules (0 heavy
  torsions) are a smaller group: 30.4 % of molecules and 26.0 % of the macro AMR-R sum on ETKDG L, but with true ring
  geometry (ORACLE) they drop 0.185 → 0.043 Å and carry 54.9 % of the whole ring-oracle gain [E-EVAL-057]. INFERENCE:
  a ring-L improvement would show up disproportionately in the rigid stratum, a torsion-model change in the flexible
  strata; reporting both averages and the strata makes visible which part of L (or of the model) an arm fixed.
- **Separating "better L" from "better use of L".** In TD the generated bond lengths and angles are exactly the
  seed's [E-EVAL-036], so (a) directly measures the L that reached the output. On the same file and basis, ETKDG L is 0.0314 Å /
  2.80° / 10.65° (bond / angle / ring dihedral) and MMFF L 0.0182 Å / 1.97° / 8.07° [E-EVAL-049], i.e. MMFF cuts bond
  error 1.7× and angle error 1.4× but ring dihedrals only 1.3×. For context only (different seed set and file): GT
  conformers of one molecule differ by 0.004 Å / 1.67° [E-EVAL-047]. (Revised after D-007: the earlier text set the
  round-1 ETKDG values 0.036 Å / 3.88° next to the round-2 MMFF values.) For S3/S4/B1-type arms (same L, different model) (a) is constant and (c) shows whether the model
  changed many conformers slightly or few strongly.
- **Ceiling honesty.** B1 on cycled true L 0.0215 Å vs random-GT-conformer true L 0.0374 Å [E-EVAL-045]: the cycled
  oracle includes conformer identity. Learned-L arms should be judged against the random-conformer ceiling.
- At δ = 0.05 Å RDKit-L arms are at 3–4 % COV-R [E-EVAL-038], so fine-threshold coverage alone cannot rank them;
  (a)–(c) can.

## 4. Assumptions that may not transfer (scout)
- Matched-GT local errors inherit the RMSD matching (a conformer matched to the "wrong" GT conformer gets a larger L
  error); report recall-side and precision-side matchings as the tool already does [E-EVAL-036].
- FM-refiner's improvement/downgrade rate pairs a sample with its own refined version [E-EVAL-009]; between our arms
  pairing is only valid when the arms start from the same seeded ETKDG conformers and the same sampling seed (expected
  for the S1 seed-pickle arms, to be confirmed by the grounder; not for arms with different seed generators).
- Strata boundaries (rigid = 0 rotatable heavy bonds, ring vs acyclic) are our choice, not a literature standard; fix
  them before looking at round-3 results.
- `geometry_metrics.py` was fixed for sp-centre torsions and atom mapping in round-1 review [E-EVAL-052] but has no
  recorded production run [E-EVAL-036]; the first run needs a sanity check (GT pass-through must give 0).

## 5. Minimal experiment (scout)
- Arms: R0 seed 0, A2, A1 (ORACLE), CTRL_rematch s0 and B1 s0 on ETKDG / MMFF / gtL / gtLcycle, S3 s0 on ETKDG /
  gtLcycle; GT pass-through as a zero check.
- Outputs per arm: local errors (bond Å, angle °, ring dihedral °, heavy torsion °), macro and micro AMR-R/AMR-P,
  stratified AMR-R (rigid / ring / acyclic), and improvement/downgrade rates at 0.02 / 0.05 / 0.10 / 0.20 Å for the
  paired comparisons S3 vs CTRL and CTRL-MMFF vs CTRL-ETKDG (same seeds).
- Expected direction: MMFF arms lower bond/angle error; their AMR-R gain is about the same in the rigid stratum as
  overall (0.152 → 0.127 Å vs 0.177 → 0.152 Å, CTRL s0 [E-EVAL-057]), unlike true ring geometry, whose gain is
  concentrated there [E-EVAL-057]; S3 vs CTRL on
  ETKDG L ≈ symmetric improvement/downgrade (no net change, 0.182 vs 0.178 Å); B1/S3 on random-GT L between CTRL and
  the cycled value.
- Cost: 0 GPU-h; ~2–4 CPU-h on 16 cores (geometry matching is the slow part).

## 6. Scout's own call (scout)
Worth trying: YES — no GPU, tools exist, and it tells the panel *which* part of L each arm fixed, which AMR-R alone
cannot.

Revision history (scout, P2): Revised after D-005 and D-007. Original §3 lead bullet "Our headline metric is mostly a
ring-L metric ... 48 % of the macro average" (few-conformer conflated with rigid) withdrawn and replaced with the rigid
share and the ring-oracle gain (E-EVAL-057); original ETKDG-vs-MMFF comparison mixed files; original §5 expected the
largest MMFF gain in the rigid stratum, which the breakdown does not show. The call is unchanged: the diagnostics do
not depend on the withdrawn inference.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
