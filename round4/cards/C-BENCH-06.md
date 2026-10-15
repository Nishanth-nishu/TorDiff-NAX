# C-BENCH-06: Minimal experiment sets: workshop paper vs main-venue paper

## 1. Idea (scout)
Fix now which experiments each paper type needs, so that GPU requests at the decision gate map onto a paper.

**Workshop paper (QM9 only, ≈ 4–8 pages):**
1. Main table (C-BENCH-01 format, both failure conventions): RDKit ETKDG, ETKDG+MMFF, ETKDG+xTB; TD released and
   CTRL_rematch; TD/CTRL on MMFF and xTB seeds; the best method arm(s), 3 training seeds; ET-Flow re-scored under our
   evaluator (C-BENCH-02). δ = 0.5 Å primary, COV-R@0.1 (intersection) secondary, threshold curve, δ = 0.05 column.
2. Leak-free oracle table (C-BENCH-04): split-half for CTRL, B1, S3, S4 and the method, with COPY and the leaky "own"
   column, on the n_clean ≥ 2 universe.
3. Local-geometry/energy table (C-BENCH-03): E_relax, bond/angle/torsion/ring-dihedral deviation, stereo rate, runtime.
4. Stratified analysis (rigid vs flexible, ring content, n_true).

**Main-venue paper (adds):**
5. GEOM-DRUGS: D0 baselines (TD released samples, ET-Flow, RDKit/MMFF), D1 L-source variants on TD's released model,
   D2 trained method + CTRL (C-BENCH-05); GEOM-XL with the DRUGS models.
6. Ablations of each method component on QM9 (3 seeds) and the key one on DRUGS (1 seed).
7. Ensemble properties on a published 100-molecule DRUGS subset; runtime on DRUGS.
8. Optionally MCF/DMT re-scored, and the ET-Flow-style scaffold split on QM9.

## 2. Where it comes from (scout)
- What every TD-P paper reports: QM9 and DRUGS tables at δ = 0.5 / 0.75 Å with 2K samples [E-BENCH-002, E-BENCH-012,
  E-BENCH-013]; GEOM-XL and ensemble properties in TD, ET-Flow, MCF [E-BENCH-041, E-BENCH-042]; runtime [E-BENCH-043];
  threshold curves and fine thresholds [E-BENCH-019, E-BENCH-047]; OOD splits in ET-Flow [E-BENCH-022].
- Published numbers to beat: QM9 ET-Flow 0.073 Å [E-BENCH-015], DMT-B 0.090 Å [E-BENCH-017], MCF 0.103 Å
  [E-BENCH-016]; DRUGS best 0.349 Å (DMT-L + Refiner) [E-BENCH-018].
- Baseline re-runs move by 3–5 points [E-BENCH-017, E-BENCH-020, E-BENCH-025]; reviewers' known objections: recall
  gameable by oversampling and cheap baselines [E-BENCH-049, E-BENCH-050], GEOM reference level [E-BENCH-051].
- Costs: QM9 11.7 GPU-h per 100-epoch run, 0.17 GPU-h per packed evaluation [E-BENCH-058]; DRUGS 1.6–4.4 GPU-days per
  100-epoch run (estimate) [E-BENCH-060]; xTB 0.14 s per QM9 seed [E-BENCH-072].

## 3. Why it could matter here (scout)
- Our position (round4/BRIEF.md, Round-2 FINAL): no learned training recipe has improved real use; the only real-use
  gains are geometry (MMFF −0.029 Å; S5 −0.004 Å beyond MMFF) [E-BENCH-068]; best real-use AMR-R 0.150 Å vs ET-Flow's
  published 0.073 Å [E-BENCH-015].
- INFERENCE: a main-venue paper cannot claim state of the art on QM9 or DRUGS RMSD metrics from where we are; it needs
  either a method that closes most of the 2× gap to Cartesian models in real use, or a contribution whose value does
  not depend on that gap (e.g. a geometry module that improves every TD-family model on RMSD *and* E_relax at stated
  cost, with leak-free evidence). A workshop paper is feasible now as a careful analysis: where TD's error comes from
  (ring geometry), why own-conformer oracles mislead (split-half), and how much cheap geometry (MMFF/xTB) buys.
- Every number in the workshop set except the method arm and ET-Flow re-scoring already exists or is scheduled.

## 4. Assumptions that may not transfer (scout)
- Venue expectations are inferred from the papers' contents, not from reviews (OpenReview could not be read, section
  Z); a human should skim the ET-Flow and NExT-Mol review threads (forum IDs in section Z) before submission.
- The workshop narrative relies on the split-half results (C-BENCH-04), still pending; if split-half shows no headroom
  from better L, the method story weakens and the analysis story strengthens.
- DRUGS cost is an estimate until the smoke test (C-BENCH-05).

## 5. Minimal experiment (scout)
- Workshop, new GPU: method arm 3 seeds × 11.7 GPU-h ≈ 35 GPU-h; split-half ≈ 7.5 GPU-h (scheduled); ET-Flow QM9
  re-scoring ≈ 2–4 GPU-h; evaluation panels for new seed sources ≈ 5–10 GPU-h. Total ≈ 50–60 GPU-h ≈ 15 h on 4 GPUs.
  CPU: xTB seeds and E_relax ≈ 10–20 CPU-h.
- Main venue, additional: DRUGS D0–D1 ≈ 20–40 GPU-h; D2 ≈ 3–9 GPU-days (100 epochs, 2 runs) or 8–22 GPU-days
  (250 epochs); QM9 ablations ≈ 35 GPU-h per component (3 seeds); DRUGS ablation ≈ 1.6–4.4 GPU-days per component.
  Order: workshop set → DRUGS D0/D1 → decide D2.
- Primary endpoints unchanged (AMR-R mean, COV-R@0.5; DRUGS at 0.75 Å).

## 6. Scout's own call (scout)
Worth trying: YES — the workshop set is affordable within one 4-day job window and mostly built; the main-venue set is
feasible in compute only with the staged DRUGS plan, and is scientifically justified only if the QM9 method beats
TD + MMFF/xTB seeds in real use on the leak-free-validated design.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
