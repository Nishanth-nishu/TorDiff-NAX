# Round 4 (paper track) brief

Written 2026-10-11. Every round-4 agent reads `orchestration/README.md` (protocol v2), its role file in
`orchestration/roles/`, and this brief first. Paths are relative to `C:\Users\HP\Desktop\tor_diff`.

## Goal
Find, verify, implement and test the changes most likely to make this project **publishable**: a novel method
(e.g. FlexiTors: learned local structure L, especially ring geometry, + torsional diffusion) or another contribution
that beats Torsional Diffusion (TD) **in real use, without oracle information**, on QM9 and then GEOM-DRUGS. Every
decision must rest on verified sources (papers and technical blogs, with PDF page / section and exact quote in the
ledgers) and on our own measured numbers.

## Honest starting position (read `EXPERIMENTS.md` and the doc for details)
- Real use (no oracle): TD 0.176 Å AMR-R; our best learned recipe S3 (50/50 RDKit/true-L training) 0.182 (ties TD);
  best overall MMFF-relaxed seeds 0.150 (A2 released model; S5 trained+tested on MMFF, 3 seeds, 0.150). Published
  learned models on the same TD-protocol QM9 split: ET-Flow 0.073–0.083, DMT 0.090, MCF 0.103 (round-3
  `round3/ledger/evidence_EVAL.md`, verified by V1).
- Where the error is (verified): ring geometry. True rings alone: CTRL 0.178 → 0.119 Å; true acyclic geometry alone:
  0.182 (no gain). RDKit has 49.6 % of 4+-atom rings twisted > 10°, MMFF 35.1 %. Up to ~72 % of TD's per-molecule
  error is a geometry floor (upper bound).
- Oracle caveat (UNVERIFIED, must be settled first): own-conformer true-geometry tests leak conformer identity
  (`round3/impact.md` §3g). B1's 0.037 and S3's 0.033 "true geometry" numbers may be mostly leakage.
- Round-3 candidate cards (17, verified ledgers, judged once) are inputs, not decisions: `round3/cards/`,
  `round3/grounding.md`, `round3/impact.md`, `round3/panel/score_*_r1.md`. Top consensus: xTB-relaxed seeds
  (C-GEOM-01) and ring templates (C-GEOM-02). xtb 6.7.1 runs on gnode118 via environment module, ~0.14 s per QM9 seed.
- Closest prior art to FlexiTors (round-1 index): GO-Flow (flow matching on internal coordinates), PuckerFlow (ring
  pucker flow matching), Ringer, Corso's intrinsic-diffusion thesis, ET-Flow, Equivariant Blurring Diffusion.

## Round-2 FINAL paired analysis (2026-10-12; `cluster_sync/round2/results/analysis_r2/contrasts.csv`)
Pre-declared contrasts, paired molecule bootstrap, Holm over the primary family. AMR-R differences in Å (arm − ref).
- **S3 (50/50 mix) on RDKit L vs CTRL_rematch: +0.0032 [+0.0013, +0.0052], Holm p = 0.0096 — significantly WORSE**,
  not a tie. S3 on MMFF L: +0.039 worse (S3, like B1, cannot use MMFF geometry). S3 at blend λ=0.5: +0.0037 worse;
  on noisy true L (0.04 Å): +0.007 worse. S3 is better than CTRL **only** on exact own-conformer true L (−0.049),
  the condition suspected of conformer-identity leakage. The S3 "robust and better" claim is therefore NOT supported;
  the leak-free test decides what remains.
- S2 (noise σ 0.04): +0.025 worse than CTRL_base on RDKit L; non-inferiority (margin 0.004) fails.
- S4 (λ-conditioned): +0.0033 [−0.0009, +0.0078] on RDKit L, not significant; −0.039 on own-conformer true L.
- S5 (MMFF-matched training, tested with MMFF): −0.0041 [−0.0064, −0.0019] vs CTRL_rematch with MMFF at test —
  a small real gain; −0.029 vs CTRL_rematch on RDKit L. Without MMFF at test: +0.003 worse.
- B1 vs CTRL on RDKit L: +0.058; on MMFF L: +0.080; at λ=0.75: −0.001 (tie); on true acyclic-only L: −0.024.
**Bottom line: no learned training recipe has improved real use. The only real-use gains are from geometry
(MMFF, ~−0.026 to −0.029 Å). A publishable method must produce better geometry, and its torsion model must be able to
use it — every model trained on true or mixed L so far gets WORSE on improved-but-imperfect geometry (MMFF).**

## Questions round 4 must answer
- Q1 Novelty: what contribution is genuinely new relative to GO-Flow, PuckerFlow, ET-Flow, Ringer, TD and its
  follow-ups (search 2024–2026 thoroughly), and which claims would reviewers reject?
- Q2 Method: which architecture or method change gives the largest real-use gain per effort, and how does it map
  onto TD's code (product manifold T^m × ring-pucker space; internal-coordinate refiner; xTB/learned seeds + robust
  torsion model; joint training; ...)?
- Q3 Evaluation: what evaluations and baselines will reviewers require (leak-free oracle tests, GEOM-DRUGS, xTB
  energies, fine thresholds, failure handling, seeds/CIs, runtime), and which public evaluation code should we adopt?
- Q4 Plan: what is the minimal experiment set for a workshop paper, and what for a main-venue paper?

## Constraints
- Compute: gnode118, 4× RTX 3090, ~11.7 h per 100-epoch QM9 training on 6 CPUs; packed inference 15–25 min. GEOM-DRUGS
  training is ~10× larger (estimate; measure). Jobs may run 4 days and resume from checkpoints. GPU-heavy runs need the
  user's approval at the decision gate; code work does not.
- Primary endpoint stays AMR-R mean and COV-R at δ = 0.5 Å (user pre-registered); COV-R@0.1 on the success
  intersection is the key secondary.
- Repo: the user is the sole author. Commit as the configured repo author; never add co-author lines.
