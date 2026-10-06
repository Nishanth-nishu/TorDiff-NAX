# Round 2 brief: shared context for all agents

Written 2026-10-06. Every agent in round 2 reads this first.

## Goal
FlexiTors-Diffusion: extend Torsional Diffusion (TD; Jing et al. 2022) from torsions only to torsions + local
structure (bond angles, maybe lengths, ring puckers). Phase 1 = QM9 ablations only (GEOM-QM9, TD split, 1000 test
molecules, 2K samples, heavy-atom RMSD, primary threshold delta = 0.5 A; primary endpoints AMR-R mean and COV-R).

## Where things are (all paths relative to C:\Users\HP\Desktop\tor_diff)
- `torsional-diffusion/` : TD code, branch `flexitors-ablation-hooks` (our patches on top of upstream 5f713b4).
- `papers/` : paper store. `papers/INDEX.md` is the index; PDFs in `papers/core`, `papers/related`; layout-preserved
  full text in `papers/related/fulltext/*.txt` and `papers/core/*.fulltext.txt` (pages separated by form feed \f, so
  page N = text after the (N-1)th \f). `papers/gap_synthesis.md` = gaps G1 (rigid local structure), G2 (train/test
  local-structure shift), G3 (torsion/angle coupling).
- `notes/ablation_plan_qm9.md` : round-1 plan. `notes/code_walkthrough.md`, `notes/code_concerns.md`.
- `review/cross_validation.md`, `review/metric_verification.md` : round-1 reviews.
- `RESULTS_QM9.md` : round-1 interim results (2 Oct). Superseded by the numbers below for the training arms.
- `cluster_sync/results/analysis/*.md` : paired bootstrap tables (common molecules, Holm-corrected) from jobs
  1089/1090, finished 3 Oct. `cluster_sync/results/*/*/summary.txt` : per-run SUMMARY lines.
- `slurm/` : sbatch scripts + `common.sh` (paths on cluster: /scratch/nishanth.r/tordiff, node gnode118 only).
- `tools/` : analysis tools (paired_compare.py, local_structure_analysis.py, breakdown.py, ...).

## Round-1 results that are now final (paired, common molecules, 3 seeds, Holm)
Inference on released checkpoint, reference R0 (AMR-R 0.176):
- A1 true GT local structure + model: AMR-R -0.093 [-0.099, -0.086]; COV-R@0.5 +7.5.
- A1c (own GT conformer's L): further -0.0037 [-0.0063, -0.0013] vs A1 (G3 coupling, small on QM9).
- A2 MMFF before diffusion: -0.026 [-0.031, -0.021]; COV-R@0.5 unchanged (+0.05, n.s.).
- A3 MMFF after: AMR-R +0.010 worse, AMR-P -0.057 better.
- G1 GT stereo: no effect.
Training arms (100 epochs, 3 training seeds each):
- CTRL_base (std pickles) AMR-R 0.1806. CTRL_rematch (our regenerated pickles) 0.1786 (-0.002).
- B1 = trained on GT local structures ("raw", no conformer matching) and tested on RDKit L: AMR-R 0.234 (+0.054 WORSE),
  COV-R -6.1. Same B1 model tested with GT L: AMR-R 0.0369 vs CTRL_base with GT L 0.0842 (-0.047), COV-R@0.05 +23.5.
  => The torsion model is tuned to the local-structure distribution it saw in training (G2 is real and large).
  A model trained on true L is far better IF it gets true L at test time. This is the core case for learning L.
- B2 (DE-only, random pairing): +0.002 worse. B5 (match on heavy-atom RMSD): +0.006 worse.
Remaining queued/deferred ideas: `slurm/ablations_*_deferred.tsv` (steps, ODE, sigma, width/depth, parity, B3 MMFF
matching, B4).

## Compute
gnode118, 4x RTX 3090 (verify), plafnet2 account/partition, max job 4 days. One QM9 training run = ~10.7 h on 1 GPU
(100 epochs). One inference+eval run = ~15 min. Login: `ssh ada` (key auth, gateway ada-gw1). /scratch only from
gnode118 via srun/sbatch with `-w gnode118`. Never use a password.

## Rules for every agent
1. Every claim about a paper cites: local PDF path, page number (PDF page, and printed page if different), and the
   EXACT quoted text (copy from the fulltext file; <= 40 words per quote). If a needed paper is missing, download the
   arXiv/OpenReview PDF into `papers/related/`, extract text with page breaks into `papers/related/fulltext/`, and add
   a row to `papers/INDEX.md`.
2. Every claim about code cites `path:line` in `torsional-diffusion/` or `tools/`.
3. Never invent a quote, number or page. If you cannot verify, write UNVERIFIED.
4. Write your output to the file you are told to; do not edit other agents' files.
