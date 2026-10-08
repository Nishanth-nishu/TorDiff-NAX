# Round 3 brief (P0)

Written 2026-10-08 by the orchestrator. Every round-3 agent reads `orchestration/README.md`, its role file in
`orchestration/roles/`, and this brief first. Paths are relative to `C:\Users\HP\Desktop\tor_diff`.

## Goal of round 3
Choose the next QM9 ablations for FlexiTors-Diffusion (torsional diffusion + learned local structure L), each one
grounded in verified literature (papers and technical blogs), mapped onto our code, and predicted against our own
numbers before it is run. Output of P1–P4: a ranked list of cards with a "worth trying" verdict for the user.

## What we know (verified, round 1; `RESULTS_QM9.md`, `cluster_sync/results/analysis/*.md`)
AMR-R mean (Å, lower better), δ = 0.5 Å primary, paired bootstrap on common molecules.
- Baseline TD (released model, RDKit L): 0.176. Paper 0.178. Our retrained CTRL_base 0.181, CTRL_rematch 0.179.
- True L + model (A1, ORACLE): 0.081. Own-conformer L (A1c, ORACLE): 0.077. True L + random torsions: 0.157.
- MMFF before diffusion (A2): 0.150. MMFF after (A3): 0.186 recall, better precision. GT stereo (G1): no effect.
- B1 = trained on true L: 0.234 on RDKit L (collapse), 0.037 on true L (0.021 own-conformer cycle), ORACLE.
- B2 / B5 (matching variants): +0.002 / +0.006 worse. Conformer matching is not the lever.
- Torsion-oracle floor by ring size (`round2/research_A.md` D1, reproduced in `round2/verify_A.md`): acyclic
  0.051 → 0.008 with true acyclic angles + bonds; ring molecules (72% of conformers) 0.117 → 0.099 only.
- At δ = 0.5 Å coverage saturates (~89% for every arm); differences show in AMR-R and COV-R@0.05–0.25.

## Round-2 results so far (PRELIMINARY: unpaired means over each run's molecules, not the pre-declared analysis)
Source: `cluster_sync/round2/results/<run>/<tag>/summary.txt` (synced 2026-10-08 10:15 IST). Means over 3 training
seeds unless noted. All `*_ORACLE` conditions use test-set true geometry.

| Test-time L | CTRL_rematch | B1 (trained on true L) | n mols |
|---|---|---|---|
| RDKit ETKDG (`S1_etkdg2L`) | 0.178 | 0.236 | 936 |
| MMFF-relaxed ETKDG (`S1_etkdg2L_mmff`) | 0.152 | 0.232 | 936 |
| Matched RDKit seed, λ = 0 (ORACLE-selected) | 0.197 | 0.261 | 955 |
| Aligned blend λ = 0.25 (ORACLE) | 0.160 | 0.222 | 906 |
| λ = 0.50 (ORACLE) | 0.136 | 0.181 | 906 |
| λ = 0.75 (ORACLE) | 0.116 | 0.115 | 906 |
| λ = 1.00, true L cycled (ORACLE) | 0.080 | 0.020 (seed 0) | 955 |
| True L + noise 0.02 Å/axis (ORACLE) | 0.098 | 0.117 (seeds vary 0.092–0.134) | 996 |
| True L + noise 0.04 Å/axis (ORACLE) | 0.119 | 0.154 | 996 |
| True ring geometry only (A5ring, ORACLE) | **0.119** | 0.188 | 955 |
| True acyclic geometry only (A5acyc, ORACLE) | 0.182 | 0.158 | 955 |

Round-2 training arms finished so far (in-job evaluation, `steps20_seed0` = RDKit L, `_gtLcycle` = true L ORACLE):

| Arm | RDKit L | True L cycled (ORACLE) |
|---|---|---|
| S2: B1 + Gaussian jitter 0.04 Å/axis (3 seeds) | 0.206 | 0.054 |
| S3: per-sample 50/50 RDKit/true L mix (seed 0 only) | **0.182** | **0.033** |

Readings to test, not conclusions:
- With the model in the loop, true **ring** geometry recovers most of the gain (0.178 → 0.119); true acyclic
  geometry alone does nothing for the standard model (0.182).
- B1 beats CTRL only once L is ≥ ~75% of the way to the truth (λ ≈ 0.75). Measured per-seed L error for each
  condition is in `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv` (use it, not λ).
- Random-noise augmentation (S2) only partly fixes B1's brittleness; the 50/50 mix (S3) may give both robustness
  and the true-L gain, pending its second seed and paired analysis.
- MMFF L helps the standard model (−0.026) but not B1.

## Still running (results by ~10 Oct)
S4 λ-conditioned training ×3, S3 seed 1, B1cap, S2 σ = 0.02, S5 (MMFF-matched training) ×3, and all post-training
evaluation panels. The round-3 panel (P4) will be re-run on final round-2 numbers if they change a conclusion.

## Open questions for round 3
- Q1: Which **source of local structure** can get close enough to true L (λ ≈ 0.75 or better), especially for
  rings, at test time without oracle information? (learned refiners, ring-pucker models, ML potentials, xTB, ...)
- Q2: How should the torsion model be trained so it **uses good L and tolerates imperfect L** (mixtures, error-
  matched augmentation, quality conditioning, training on generated L, ...)?
- Q3: Is a dedicated **ring-geometry component** worth building, and in what coordinates?
- Q4: Are we measuring the right thing (xTB/DFT references, fine thresholds, energies), and how do our numbers
  compare with current QM9 results (e.g. ET-Flow AMR-R 0.073 per `papers/INDEX.md`)?

## Constraints
- QM9 only this phase. GEOM-DRUGS later.
- Compute: gnode118, 4× RTX 3090. One 100-epoch training ≈ 11.7 h; one inference + eval ≈ 15–25 min (packed 3/GPU).
  Planning budget for round 3: up to ~300 GPU-h (user confirms in P5). Jobs may run 4 days and resume from checkpoints.
- Primary endpoint stays AMR-R mean and COV-R at δ = 0.5 Å; COV-R@0.1 on the success intersection is the key secondary.
- Code: TD at `torsional-diffusion/` with round-2 flags (`--l_jitter`, `--l_mix_p_gt`, λ conditioning, `--resume`),
  tools in `tools/` (`lgeom.py`, `make_l_seed_pickles.py`, `analyze_round2.py`), SLURM in `slurm/`.
- Existing literature store: `papers/INDEX.md`, `papers/gap_synthesis.md`, round-2 verified evidence in
  `round2/research_A.md`, `research_B.md`, `verify_*.md`, `research_check_*.md`. Reuse their verified E-IDs by
  copying the entry into your own ledger with `Reused from: <file> <ID>` and re-checking the quote.
