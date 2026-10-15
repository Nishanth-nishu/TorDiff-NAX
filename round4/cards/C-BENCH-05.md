# C-BENCH-05: GEOM-DRUGS plan: staged, inference-first with released TD checkpoint, then one trained arm

## 1. Idea (scout)
Run GEOM-DRUGS in three gated stages so that GPU-days are spent only after cheap evidence:
- **D0 (no training, mostly CPU):** fetch TD's DRUGS archive; score TD's *released* DRUGS samples and the RDKit /
  RDKit+MMFF baselines with our evaluator (does 72.7 % / 0.582 Å reproduce under our failure convention?); run
  ET-Flow `drugs-o3` inference (C-BENCH-02).
- **D1 (inference only):** TD's released `drugs_default` model on alternative L sources (MMFF, xTB, ring templates if
  C-GEOM-02 exists): the DRUGS analogue of our QM9 A2/C-GEOM-01 result, which needs no DRUGS training.
- **D2 (training, user approval):** our method arm and a CTRL retrained in the same pipeline, TD-P split, 1 seed each
  (literature norm), 100 or 250 epochs depending on the measured smoke-test cost.
Before D2, measure: a 1-epoch smoke run (`--limit_train_iters`), featurisation time and RAM, xTB time on 20 DRUGS
seeds, and conformer-matching time on one 1000-molecule shard if the method changes the training L.

## 2. Where it comes from (scout)
- Protocol: TD-P DRUGS split 243473/30433/1000, δ = 0.75 Å, 2K samples [E-BENCH-001, E-BENCH-002]; results to compare
  against: TD 72.7 / 0.582 [E-BENCH-013], ET-Flow 79.53 / 0.452 [E-BENCH-014], MCF-L 84.7 / 0.390 [E-BENCH-016], DMT-L
  85.8 / 0.375 [E-BENCH-017], DMT-L + Refiner 87.47 / 0.349 [E-BENCH-018], S23D-B 87.0 / 0.380 [E-BENCH-005],
  AvgFlowDiT-L 82.0 / 0.409 [E-BENCH-021] (COV-R % / AMR-R Å).
- Released artefacts: TD DRUGS model (6.4 MB), its generated test conformers (457 MB) and data archive (25.3 GB)
  [E-BENCH-031]; its config equals our architecture with 74 node features [E-BENCH-032]; ET-Flow DRUGS checkpoints
  [E-BENCH-033]; MCF DRUGS S/B/L [E-BENCH-034]. The raw GEOM archive is 50.1 GB and needed only for other groups'
  preprocessing [E-BENCH-056].
- Cost anchors: TD DRUGS 250 epochs = 4–11 days on one A6000 [E-BENCH-052]; GeoMol DRUGS/QM9 time ratio 3.4
  [E-BENCH-053]; ET-Flow used 8 A100s [E-BENCH-054]; DRUGS molecules average 44.4 atoms and 6.5 rotatable bonds
  [E-BENCH-055]; TD featurisation ≈ 2 h on one core and DRUGS matching as 300 parallel 1000-molecule workers
  [E-BENCH-057]; our QM9 epoch 377.7 s, 11.7 GPU-h per 100 epochs, eval 0.17 GPU-h per packed condition [E-BENCH-058];
  QM9 featurise ≈ 8 min, caches ≈ 1 GB [E-BENCH-059]; xTB 0.14 s per QM9 seed [E-BENCH-072].
- [BENCH-calc] One DRUGS run on a 3090 ≈ 23–63 min per epoch → 1.6–4.4 GPU-days (100 epochs) or 4–11 GPU-days
  (250 epochs); 2 arms × 3 seeds × 250 epochs = 24–66 GPU-days (6–17 days of the node); caches ≈ 7–10 GB; xTB ≈ 1–2 s
  per DRUGS seed [E-BENCH-060].
- Ensemble properties, if run: TD/ET-Flow 100-molecule protocol, publish the subset [E-BENCH-041, E-BENCH-042].

## 3. Why it could matter here (scout)
- The brief's goal is QM9 *then* GEOM-DRUGS; every main-venue conformer paper since TD reports DRUGS
  [E-BENCH-013, E-BENCH-014, E-BENCH-016, E-BENCH-017, E-BENCH-018]. QM9 at 0.5 Å is considered saturated [E-BENCH-047].
- On QM9 the only real-use gain is geometry (MMFF −0.029 Å) [E-BENCH-068]; TD's App. F.1 shows the local-structure
  floor on DRUGS is large (0.324 Å matched RDKit L vs 0.284 Å other-GT L) [E-BENCH-061], against TD's 0.582 Å AMR-R,
  so DRUGS error is dominated by torsions, not L. INFERENCE: an L-only method may gain relatively less on DRUGS than
  on QM9; D1 measures this for free before any training.
- D0 also tells us whether our evaluator reproduces TD's DRUGS numbers; on QM9 it did not (88.8 vs 92.8 % COV-R)
  [E-BENCH-011].

## 4. Assumptions that may not transfer (scout)
- RTX 3090 ≈ RTX A6000 for this fp32 workload and data loading keeps up on 6 CPUs (unmeasured); the 4–11 day range
  is TD's own spread and may hide hyper-parameter-search runs.
- TD's released DRUGS model was trained in another code state and RDKit version (as with QM9 CTRL_base vs
  CTRL_rematch, review/cross_validation.md C3); a trained method arm should be compared with a CTRL retrained in our
  pipeline, which doubles D2's cost, or the paper must state the nuisance.
- DRUGS has stereo-rich, larger molecules: ETKDG failure counts, MMFF failures and xTB failures differ from QM9 and
  must be counted per method.
- /scratch capacity on gnode118 for ≈ 25 GB compressed + extracted data + ≈ 10 GB caches is unverified; memory cap
  5000 MB per CPU (≈ 30 GB at 6 CPUs) should hold a ≈ 10 GB cache.
- GEOM-XL (102 molecules > 100 atoms) is free once DRUGS works (TD's released model; TD and MCF report it) but RMSDs
  are computed without symmetry there.

## 5. Minimal experiment (scout)
- D0: download `drugs.tar.gz` (25.3 GB, TD Drive; needs the user's approval) + released `drugs_default`; evaluate
  `drugs_steps20.pkl` (CPU, ≈ hours); RDKit, RDKit+MMFF baselines (CPU); ET-Flow `drugs-o3` (≈ 5–20 GPU-h, measure on
  20 molecules first). Cost ≈ 0.5–1 day wall, ≤ 20 GPU-h.
- D1: TD `drugs_default` × {RDKit L, MMFF L, xTB L} × 1 sampling seed (xTB seeds CPU ≈ 1–2 s each; count seeds after
  download); GPU inference ≈ 2–5 GPU-h per condition (UNVERIFIED; QM9 is 0.17–0.26 GPU-h). Primary: AMR-R, COV-R at
  δ = 0.75 Å; secondary COV-R@0.5 / 0.25 and E_relax (C-BENCH-03).
- D2 (gated on D1 showing a gain and on the QM9 paper result): method + CTRL, 1 seed, 100 epochs ≈ 3–9 GPU-days total,
  or 250 epochs ≈ 8–22 GPU-days total, run in parallel on 2 GPUs with checkpoint resume.

## 6. Scout's own call (scout)
Worth trying: YES for D0 and D1 (cheap, decisive about whether geometry matters on DRUGS); MAYBE for D2 (only if D1
shows a real-use gain and the smoke test confirms ≤ 4 GPU-days per 100 epochs).

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
