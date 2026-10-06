# Round 2: merged, verified ablation shortlist (QM9)

Merged 2026-10-06 from `research_A.md` (R2A-*) and `research_B.md` (R2-*), with every required change taken from
`verify_A.md` and `verify_B.md`. Coding agents implement exactly this list. If a change looks wrong, raise it in your
plan; do not silently deviate.

## Endpoints (pre-declared before any round-2 run)
- **Primary (unchanged, user pre-registered):** AMR-R mean and COV-R at δ = 0.5 Å, TD protocol (union, failures = 0
  coverage), paired bootstrap on common molecules, Holm correction (`tools/paired_compare.py`).
- **Key secondary, its own Holm family:** COV-R@0.1 on the **success intersection** (molecules every compared arm
  generated for).
- **Also reported:** failure count per arm; COV-R@0.5 on the intersection (sensitivity analysis); AMR-P / COV-P.
- **Oracle labelling:** every condition that uses test-set GT local structure (GT L, own-L/cycle, noisy GT, λ > 0
  interpolation, GT ring-only, GT acyclic-only, the Hungarian-selected λ = 0 seed) is labelled ORACLE in tables and
  file names (`*_ORACLE`). Oracle results are analysis, never model selection.

## Common fixes the verifiers found (apply everywhere)
- F1. `standardize_confs.py:108` aligns a hydrogen-stripped copy, so matched RDKit conformers are NOT aligned to GT.
  Any interpolation between RDKit L and GT L needs an explicit (Kabsch, all-atom, same atom order) alignment step.
- F2. `--pre_mmff` only acts on the SMILES path (`generate_confs.py:136-137`), not on `--seed_confs`. MMFF variants
  of seed pickles must be relaxed when the seed pickle is built.
- F3. Non-cycle GT-L runs draw with replacement (about 13.5% of GT L never drawn). Use `--seed_confs_cycle`
  consistently for all GT-derived test conditions, and use the existing gtLcycle / A1c runs as their anchors.
- F4. The RDKit-end control for anything built from regenerated pickles is **CTRL_rematch**, not CTRL_base
  (`slurm/standardize_qm9.sbatch:17-19`).
- F5. For every perturbed test-time L, report the achieved bond-length / angle / ring-subset RMSD to GT, and the
  CPU torsion-oracle floor (`tools/local_structure_analysis.py --mode run`). Compare arms by B1 − CTRL per condition,
  not by absolute curves (noise raises the floor for every model).
- F6. Noise σ is per axis (Cartesian), stated in file names; σ = 0.04 per axis ≈ 0.057 Å per bond endpoint pair.

## Arms

### S0. CPU hygiene and diagnostics (no GPU), from R2-0 with fixes
- Re-run all paired tables (round 1 + round 2) with the intersection secondary endpoint and failure counts.
- Clean floor numbers: exclude multi-fragment molecules; compare on matched molecule sets; check optimiser
  convergence (report the floor change with 2× restarts on a 100-molecule subset).
- Leak test (does B1 read torsions from L?), redesigned per verify_B §4 R2-0(e): in **torsion space**, is the
  generated τ nearest to its source GT conformer's τ vs other GT τ (Boltzmann-weighted)? Nulls: the no-model
  `A1c_gtL_cycle_sanity` (+ random torsions) and CTRL on gtLcycle.

### S1. L-quality dose-response (inference only, ~15 GPU-h), from R2A-1 + R2-1 + R2-5
Models: CTRL_base ×3, CTRL_rematch ×3, B1 ×3 training seeds. Test-time L sources:
1. RDKit (existing runs).
2. MMFF-relaxed seeds built inside the seed pickle (F2). Not oracle.
3. GT + isotropic noise, σ ∈ {0.01, 0.02, 0.04} Å per axis, cycle (ORACLE).
4. Aligned RDKit↔GT interpolation, λ ∈ {0.25, 0.5, 0.75} (F1), cycle (ORACLE); λ = 1 is the existing gtLcycle.
5. GT ring geometry only (A5-ring) and GT acyclic geometry only (A5-acyc), both built on every seed with the same
   mechanism where possible; report the achieved subset RMSDs (F5) and stratify by torsion count and ring size (ORACLE).
6. Optional: GFN2-xTB-relaxed seeds, only if xtb installs in under 30 min on gnode118; otherwise skip and note it.
Output: B1 − CTRL per condition, and the L error ε* at which B1 stops beating CTRL.

### S2. B6: noise-augmented GT-L training (~43 GPU-h), from R2A-4 + R2-2
B1 recipe (raw GT L), plus Gaussian jitter of all atom positions at each training sample, σ = 0.04 Å per axis
(3 seeds), and σ = 0.02 (1 seed, exploratory). Control: B1 (single factor). Target: CTRL_base on RDKit L.
Pre-registered success: (a) non-inferior to CTRL_base on RDKit L, margin 0.004 Å AMR-R; (b) better than CTRL on GT L
(ORACLE). Evaluate also on S1 conditions 2–4 (+6 B1 control runs on jittered GT L).

### S3. Mixed-L training (~23 GPU-h), from R2A-3
Each training sample uses either its conformer-matched RDKit L or its GT L (Bernoulli 0.5), with both halves capped
at the same confs_per_mol. 2 seeds. Controls: CTRL_rematch (RDKit end) and B1 (GT end).

### S4. λ-conditioned interpolation training (~37 GPU-h), from R2A-2
Training L = aligned interpolation (F1) between matched RDKit L and GT L, λ ~ U[0, 1], with λ given to the score model
as an extra scalar input (cascaded-diffusion noise conditioning). At test time, λ = 0 on RDKit L and λ = 1 on GT L
(ORACLE). 3 seeds. Control: CTRL_rematch. Note the confound: B1 uses uncapped raw conformers (`dataset.py:209`) while
std/rematch pickles are capped (`standardize_confs.py:59-67`); add a capped-B1 note or control.
If S4 is too large to implement and test in time, it moves to round 3. The coding agents decide this by vote.

### S5. B3: MMFF-matched TD training (~35 GPU-h), from R2-4 + R2A-7
Re-standardize with MMFF-relaxed RDKit conformers before matching (`mmff` variant, `slurm/ablations_train_deferred.tsv`
line B3_match_mmff). 3 seeds. Controls: CTRL_rematch with and without `--pre_mmff` (+3 inference runs). Report
molecules lost to MMFF errors.

### S6. Sampler checks (~3 GPU-h, low priority), from R2-6
Steps {10, 50}, σ_min_inf {0.005π}, ODE, on the released model (3 sampling seeds) and B1_gtL (descriptive). Lines
already exist in `slurm/ablations_inference_deferred.tsv`.

### Deferred to round 3
R2A-5 (Cartesian L-refiner pilot; GO-Flow evidence argues against a pure-Cartesian refiner), R2A-6 (joint non-ring
angle diffusion; gated on the S1 A5-acyc result).

## Budget
Training runs: S2 4 + S3 2 + S4 3 + S5 3 = 12 × ~10.7 h ≈ 128 GPU-h. Inference ≈ 25 GPU-h. Total ≈ 153 GPU-h, about
38 h on 4 GPUs. Order: S5 standardization (CPU) and S0 first; then waves of 4 training runs; inference fills gaps.
