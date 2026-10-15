# C-BENCH-02: Baselines re-run from released checkpoints under our evaluator (TD, ET-Flow, MCF, DMT, RDKit/MMFF/xTB)

## 1. Idea (scout)
Do not quote baseline numbers only from papers. Generate (or take the authors' released) conformers for the 1000 TD-P
test molecules from every baseline that has a public checkpoint, and score all of them with the same evaluator,
molecule list and sample budget (2K) as our arms: TD (released `qm9_default`, `drugs_default` and its released
`drugs_steps20.pkl`), ET-Flow (`qm9-o3`, `drugs-o3`), MCF (QM9; DRUGS-B), DMT-B/L if the OSF checkpoints match the TD-P
split, and the non-learned baselines RDKit ETKDG, ETKDG + MMFF, ETKDG + GFN2-xTB, TD on MMFF/xTB seeds. Print the
published value next to the re-scored value for each row.

## 2. Where it comes from (scout)
- Re-runs differ from published numbers: TD + particle guidance 77.0 → 73.8 % COV-R on re-run [E-BENCH-017]; ET-Flow
  precision 74.38 → 69.79 % COV-P when re-run by EnFlow [E-BENCH-020]; FM-refiner's ET-Flow / MCF-L / DMT-L DRUGS
  values differ from the originals [E-BENCH-018]; AvgFlow copies baseline values from papers [E-BENCH-021].
- ET-Flow's code changed after the paper: its README warns configs may not reproduce [E-BENCH-024]; a user could not
  reproduce the QM9 numbers with the released checkpoint (recall 91 %, precision 94.5 %) while DRUGS reproduced
  [E-BENCH-025]; the training alignment had a reflection bug fixed after publication [E-BENCH-026]; its current
  preprocessing indexes TD's split into an unsorted file list (possible train/test overlap if retrained; UNVERIFIED)
  [E-BENCH-028].
- Evaluators differ in failure handling and reference filtering [E-BENCH-007, E-BENCH-008, E-BENCH-009].
- Released artefacts exist: TD DRUGS model + released samples + 25.3 GB data [E-BENCH-031, E-BENCH-032]; ET-Flow
  QM9 and DRUGS checkpoints [E-BENCH-033]; MCF QM9 and DRUGS S/B/L checkpoints [E-BENCH-034]; DMT-B/L checkpoints
  (dataset per checkpoint UNVERIFIED) [E-BENCH-035].
- Cheap baselines must get the same sample budget [E-BENCH-049, E-BENCH-050].
- Retraining ET-Flow on DRUGS is out of reach (8 A100s × 250 epochs) [E-BENCH-054].

## 3. Why it could matter here (scout)
- Our claim is "beats TD in real use" and our honest gap to the state of the art is large: best real-use AMR-R
  0.150 Å (S5 with MMFF) [E-BENCH-068] vs published ET-Flow 0.073 Å [E-BENCH-015]. If ET-Flow's QM9 figure does not
  reproduce under our evaluator (E-BENCH-025 suggests recall 91 % rather than 96.5 %), the size of that gap, and what
  reviewers will compare us against, changes.
- Our own reproduction of TD already differs from the paper by 3.8 COV-R points [E-BENCH-069]; the same will be true
  for every baseline unless all are scored identically.
- RDKit + MMFF (0.1507 Å with TD; ETKDG alone 0.2298 Å) [E-BENCH-069] and our S5 gain [E-BENCH-068] show geometry
  alone moves AMR-R by ≈ 0.03 Å; a reviewer will ask for the matching non-learned baseline (ETKDG + xTB, TD + xTB
  seeds) before crediting a learned geometry model.

## 4. Assumptions that may not transfer (scout)
- ET-Flow, MCF and DMT need their own environments (PyTorch Lightning / different torch); round 3 estimated ≈ 0.5 day
  of environment work for ET-Flow (round3/impact.md, C-GEOM-03 row). MCF/DMT environments are unmeasured.
- Atom order and hydrogens: ET-Flow builds test molecules from TD's `test_mols.pkl` [E-BENCH-006], so its outputs map
  onto our evaluator; MCF uses its own processed 995-molecule set [E-BENCH-010] and needs a SMILES-keyed mapping, with
  the unmatched molecules reported.
- Older ET-Flow commit vs current code: use the released checkpoint with the commit the author named (E-BENCH-025)
  for inference; do not retrain ET-Flow.
- Sampling seeds: published baselines report one run; we should run ≥ 2 sampling seeds for TD and ET-Flow on QM9 to
  show run-to-run spread (GeoMol: ±0.0012 Å, E-BENCH-023).

## 5. Minimal experiment (scout)
- QM9 (workshop and main): TD released (done: R0), ET-Flow `qm9-o3` (50 steps, 2 sampling seeds), RDKit ETKDG / +MMFF /
  +xTB (CPU; round-3 C-EVAL-04 design), TD on MMFF and xTB seeds (done / C-GEOM-01). Optional: MCF-QM9, DMT-B QM9.
- DRUGS (main): score TD's released `drugs_steps20.pkl` (CPU only, reproduces 72.7 / 0.582 or not); ET-Flow
  `drugs-o3` inference; MCF-B inference if the environment works; RDKit/MMFF baselines (CPU).
- Outputs: one CSV of per-molecule RMSD matrices per method (our `eval.pkl` format) → C-BENCH-01 tables.
- Cost (estimates, measure on 20 molecules first): ET-Flow QM9 ≈ 1–2 GPU-h per sampling seed; MCF QM9 (1000 steps)
  several GPU-h; DRUGS inference for ET-Flow/MCF ≈ 5–20 GPU-h each (UNVERIFIED; DRUGS has many more conformers per
  molecule); TD DRUGS released samples 0 GPU-h. Environment work 1–2 days. Downloads: ET-Flow checkpoints ≈ 0.1 GB each,
  TD DRUGS archive 25.3 GB (user approval needed for downloads on the cluster).
- Pre-declared contrast for the paper: method − TD and method − ET-Flow on the intersection universe, AMR-R and
  COV-R@0.5 (primary), COV-R@0.1 (secondary).

## 6. Scout's own call (scout)
Worth trying: YES for TD, ET-Flow (QM9 + DRUGS) and the RDKit/MMFF/xTB baselines; MAYBE for MCF and DMT (include if
their environments install in ≤ 1 day). Reviewers will not accept copied baseline numbers when re-runs are known to
move by 3–5 points.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
