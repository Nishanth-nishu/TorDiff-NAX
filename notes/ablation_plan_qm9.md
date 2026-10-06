# QM9 ablation plan: torsional diffusion baseline → motivating FlexiTors-Diffusion

**Scope.** GEOM-QM9 only, using the TD protocol:
- GeoMol split 106,586 / 13,323 / 1,000 test molecules.
- K = 2L generated conformers per molecule.
- Heavy-atom symmetry-aware RMSD (`GetBestRMS`), strict `<`.

Everything here can be launched with the scripts in `slurm/`. The code hooks are in `patches/0001-ablation-hooks.patch` and the analyses in `tools/`. File and line references are in `notes/code_walkthrough.md`; code issues are in `notes/code_concerns.md`. The literature context is in `papers/gap_synthesis.md` and `papers/core/tordiff_analysis.md`, both from the research agent.

---

## 0. Framing (read first; updated with the literature review)

1. **The train/test local-structure shift (G2) is not a new gap.** TD identifies it and removes it with conformer matching (TD §4.1, App. E).
   - Training on GT local structures is one of *their* ablations. On DRUGS it collapses COV-R from 72.7 to 34.8 (TD Table 8).
   - So our arm **B1 is a reproduction of Table 8 on QM9**, not a discovery.
   - The actual gap is what the workaround costs: (i) an **accuracy floor** set by RDKit local structure (about 0.17 Å on QM9, TD App. H; about 0.324 Å on DRUGS, TD Table 5); (ii) non-physical, non-minimum training targets; (iii) one-directional coupling. Torsions adapt to L; L never adapts to torsions.
2. **At δ = 0.5 Å, QM9 is saturated.** The median COV is 100 for every modern method, and TD's own QM9 table has median COV-R and COV-P of 100 for all baselines.
   - The headline numbers are therefore **AMR (MAT) mean and median** plus a **COV threshold sweep down to 0.05 Å** (0.05, 0.1, 0.25, 0.5).
   - The patched `evaluate_confs.py` prints `SWEEP` lines, and `tools/breakdown.py` recomputes any threshold from the saved matrices.
   - Add **local-geometry metrics**: bond-length MAE, bond-angle MAE and torsion MAE of each generated conformer against its matched GT conformer (`tools/geometry_metrics.py`). Optionally add GFN2-xTB relaxation energy (§4).
3. **QM9 is where the L floor bites — hypothesis, not yet a fact.** TD's QM9 AMR-R (0.178 mean) is close to the ≈0.17 Å figure of TD App. H. OMEGA, which has better local structure, gets a better AMR-R median (0.126 vs 0.147), according to TD App. H.
   - **Caveat (review/cross_validation.md M1).** The ≈0.17 Å figure is a *conformer-matching* number: one RDKit seed per GT conformer (Hungarian, L seeds), H-inclusive DE objective, 15 DE iterations. Each of these biases it **upwards** relative to the quantity AMR-R has to be compared with: the best of K = 2L seeds with heavy-atom, symmetry-aware RMSD. So "0.178 ≈ 0.17" does **not** show that TD is at its floor. The true test-time floor may be well below 0.17 Å, in which case TD still has torsional headroom on QM9.
   - The key FlexiTors evidence is to **measure the floor correctly and per molecule**. Use `floor_best_sym` (K = 2L independent seeds) and, best, the **run-matched floor** `floor_run` (`local_structure_analysis.py --mode run`, on the run's own conformers). Then show that the remaining error tracks it.

**Reference numbers.** TD Table 7, QM9, δ = 0.5 Å, extracted by the research agent. Use them for the reproduction check (P0/R0).

| Method | COV-R mean | COV-R median | AMR-R mean | AMR-R median | COV-P mean | COV-P median | AMR-P mean | AMR-P median |
|---|---|---|---|---|---|---|---|---|
| RDKit | 85.1 | 100.0 | 0.235 | 0.199 | 86.8 | 100.0 | 0.232 | 0.205 |
| OMEGA | 85.5 | 100.0 | 0.177 | 0.126 | 82.9 | 100.0 | 0.224 | 0.186 |
| TD | 92.8 | 100.0 | 0.178 | 0.147 | 92.7 | 100.0 | 0.221 | 0.195 |

---

## 1. Metrics recorded for every arm

- COV-R, AMR-R, COV-P and AMR-P, as mean and median, at δ = 0.5 Å (the TD QM9 protocol, **primary**). Plus the COV sweep at δ ∈ {0.05, 0.1, 0.25, 0.75} Å (**secondary / exploratory**), the number of model failures, and the number of "additional failures".
- Local geometry of generated conformers vs matched GT (`geometry_metrics.py`): bond MAE (Å), angle MAE and max error (°), and heavy-torsion MAE (°). Both recall-side and precision-side matching are reported.
- Per-molecule breakdown (`breakdown.py`) by heavy-atom count, number of heavy-atom rotatable bonds, number of rotatable bonds as the repo defines them (including H rotors), and number of GT conformers.
- Uncertainty (review S1-S3):
  - 3 sampling seeds (R0-R2) for every model-based inference arm in the budget (A1, A1c, A2, A3, G1; `*_seed1/2` lines), all on the released `qm9_default` checkpoint. The no-model arms (A0, A1-rand, sanity checks) have one seed.
  - All seed-s arms share ETKDG seed s+1 with R<s>, so they are paired per molecule on the same RDKit L.
  - Training seeds: 3 each for CTRL_base, CTRL_rematch, B1, B2 and B5 (100 epochs). Every trained model is evaluated with generation seed 0, so all training arms are also paired on the same RDKit L. The 250-epoch baseline retrain is deferred; the released checkpoint is the baseline.
  - `tools/paired_compare.py` does the stats: replicates are averaged per molecule, then a paired bootstrap over molecules. It reports the CI, the bootstrap p, the Holm-adjusted p and the reference replicate SD. The COV population (failures = 0) and the AMR population (finite in both arms) are reported with separate n.
  - **Pre-registered primary endpoints (user decision): AMR-R mean and COV-R mean at δ = 0.5 Å** (the TD paper's QM9 threshold), Holm-corrected across arms. These are the `paired_compare.py` defaults (`--primary MAT-R COV-R@0.5`). Everything else is secondary / exploratory and is not multiplicity-corrected: the 0.05-0.25 Å sweep, medians, precision metrics and strata. Known limitation: QM9 COV at 0.5 Å is near saturation (medians are 100 in TD Table 7), so COV-R@0.5 has little power. AMR-R mean carries most of the primary evidence, and the small-δ sweep may show effects that the primary endpoints cannot confirm.

---

## 2. Ablation matrix

Priority: ★★★ = directly motivates FlexiTors (run first); ★★ = protocol and reproduction; ★ = nice to have.

"Where" gives the array line: **INF** = `slurm/ablation_inference_array.sbatch` + `ablations_inference.tsv`; **TRN** = `slurm/ablation_train_array.sbatch` + `ablations_train.tsv`; **ANA** = `slurm/analysis_qm9.sbatch`.

Runtime estimates are for one GPU plus 12-16 CPU cores. They are **not measured**; they come from code inspection and the paper's reported costs. Recalibrate after R0.

**Budget status.** The arms that fit the 4-day / 4-GPU budget are listed in §3. Everything else in this matrix is **DEFERRED** (§3.4): P1, C (steps), D (ODE), E-inf, E-train, F, G2 (no parity), H (resampling pools), B3, B4, the 250-epoch baseline retrain, and xTB (§4). Their TSV lines live in `slurm/ablations_*_deferred.tsv`.

### (a) Local structure at test time: GT L vs RDKit L, and the floor ★★★

| ID | Arm | Change / flag | Where | GPU | Est. runtime | Hypothesis tested / expected outcome |
|---|---|---|---|---|---|---|
| **A0-floor** | **Direct RDKit-L floor on the test set.** Put the GT torsions on the RDKit L ("transplant"), then also optimise the torsions by DE ("floor"). For each GT conformer, the heavy-atom RMSD to GT | `tools/local_structure_analysis.py --mode test` (plus `--mmff_seeds`) | ANA | no | ~3-10 h on 32 cores (about 9 DE runs per GT conformer plus automorphism maps; not measured, and `analysis_qm9.sbatch` now allows 24 h) | `floor_rmsd` (one-to-one, the conformer-matching analogue) should come out near 0.17 Å (TD App. H). `floor_best_sym` (best of K = 2L seeds, symmetry-aware) is the number to compare with AMR-R, and is expected lower. The fraction of GT conformers with `floor_best_sym` > δ is an **estimated** COV-R ceiling for torsion-only models on RDKit L. It is not a hard bound: DE is heuristic and the ETKDG draws are independent of the run (`breakdown.py` prints it as `ESTIMATED COV-R ceiling`). The hard, run-paired version is **J-run** below. Also produced: `floor_gtother` (other GT conformers' L as seeds) and `angle_oracle_rmsd` / `local_oracle_rmsd` (acyclic angles, or angles + bond lengths, set to GT). See **J-decomp** |
| A0-train | Training-set matching residual `conf['rmsd']` from the shipped standardized pickles | `local_structure_analysis.py --mode std` | ANA | no | minutes | Reproduces the TD conformer-matching RMSD for QM9. Its distribution and per-molecule tail are the "projection error" FlexiTors would shrink |
| A0-rdkit | Pure ETKDG baseline (RDKit row of Table 7) | `--no_model --no_random` | INF `A0_rdkit_etkdg_only` | no* | ~0.5 h | Should reproduce 85.1 / 0.235 |
| A0-rand | RDKit L + uniform random torsions (Table 8 "random torsions") | `--no_model` | INF `A0_rdkitL_random_tors` | no* | ~0.5 h | Lower bound for the torsion model |
| **A1** | **Oracle L:** GT local structures (each seed is a random GT conformer of the molecule), torsions randomised, then the trained model runs | `--seed_confs data/QM9/test_gt_seed_confs.pkl` (from `tools/make_seed_pickles.py`) | INF `A1_gtL_model` | yes | ~1 h | The gap to R0 in AMR and in sweep COV at 0.05-0.25 Å measures how much of TD's QM9 error is due to L. Caveat: the model was trained on RDKit L, so GT L is off-distribution for the score net. Read it together with B1 as the 2×2 {train L} × {test L} matrix |
| A1-sanity | GT L, GT torsions, no model | `--seed_confs ... --no_model --no_random` | INF | no* | ~0.5 h | **Pipeline check.** Each generated conformer is a GT conformer drawn at random (`random.choice`, `sampling.py:66`). Expect **COV-P = 100% and AMR-P = 0 at every δ**. COV-R should be about 1 − (1 − 1/L)^{2L} (≈ 87% for large L, 100% for L = 1), and a GT conformer that is never drawn has a nonzero min-RMSD. If precision is not perfect, the CSV key convention or `clean_confs` is off |
| A1-rand | GT L + random torsions | `--seed_confs ... --no_model` | INF | no* | ~0.5 h | Torsion-only difficulty with perfect L |
| **A1c** | **Coupled-L oracle:** as A1, but each generated conformer i gets the L of GT conformer i mod L (`--seed_confs_cycle`, patch). A1 (`random.choice`) is a *perfect but τ-independent* L sampler; A1c gives every GT conformer its *own* L | INF `A1c_gtL_cycle_model` (+ seeds 1, 2); `A1c_gtL_cycle_sanity` must give COV-P = 100 and AMR-P = 0 exactly, and COV-R = 100 with AMR-R = 0 whenever K ≥ L (K = 2 × CSV count, L = count after `clean_confs`) | yes | ~1 h | **A1c − A1 is the model-based value of τ↔L coupling (G3)**, and A1 − R0 the value of a better *independent* L sampler (G1). If A1c ≈ A1, coupling is worthless on QM9 and a better L sampler (OMEGA/LoQI-style) would do; that falsifies the joint-diffusion part of FlexiTors. Tested in `paired_A1c_vs_A1_*.md` |
| A2 | MMFF-relaxed RDKit L before sampling | `--pre_mmff` | INF | yes | ~1.5 h | A cheap L improvement without retraining; tests L-sensitivity of a model trained on unrelaxed L. Pair with B3 |
| A3 | MMFF after sampling | `--post_mmff` | INF | yes | ~1.5 h | Post-hoc L relaxation. If AMR at small δ improves, L error is the limit and a learned L factor is justified |

\* CPU-only work, but these rows are kept in the GPU array for simplicity.

### (b) Conformer matching on/off: reproduce TD Table 8 on QM9 ★★★ (B1, B2) / ★★ (B3, B4)

| ID | Arm | Change / flag | Where | GPU | Est. runtime | Hypothesis / expected outcome |
|---|---|---|---|---|---|---|
| **B1** | **Train on GT L** (no conformer matching; raw GEOM conformers) and test on RDKit L **and** on GT L | `--std_pickles ""` (the `pickle_dir=None` path in `dataset.py`). Needs `featurize_qm9.sbatch VARIANT=raw` | TRN `B1_train_gtL` (evaluates both) | yes | featurize ~1 h CPU + ~5-10 h train (100 epochs) | **Reproduction of Table 8** "train on GT L". Expect a large COV drop on RDKit L; on GT L it should be the best of the 2×2. Together with A1 this gives the full matrix (train RDKit-matched / GT) × (test RDKit / GT) |
| **B2** (vs **CTRL_rematch**) | **Only D.E. matching, random L̂-C pairing** (Table 8 row) | `standardize_confs.py --no_match` (skips the Hungarian assignment; still DE-optimises torsions). Run `standardize_qm9.sbatch VARIANT=nomatch`, then featurize, then train | TRN `B2_DEonly_randpair` | yes (train) | standardize ~1-4 h on 32 cores; train ~5-10 h | Table 8 shows almost no loss on DRUGS (72.5 vs 72.7), which is TD's own evidence that the pairing between L and torsions barely matters. On QM9, measure it at small δ. If it matters there, that is evidence for coupling (G3) |
| B3 (vs **CTRL_rematch**) | Matching with MMFF-relaxed seeds, tested with `--pre_mmff` | `VARIANT=mmff` | TRN `B3_match_mmff` | yes | as B2 | Better L at train and test, still frozen. Shows how far a better *fixed* L-sampler moves the floor, compared with a learned L factor |
| **CTRL_rematch** | The default matching recipe **re-run locally** (`standardize_qm9.sbatch VARIANT=rematch`) | — | TRN `CTRL_rematch_100ep` | yes | as B2 | **Matched control for B2/B3/B4/B5.** The shipped `standardized_pickles` came from an unknown RDKit version with unseeded ETKDG. Comparing locally regenerated variants with CTRL_base would confound the variant with RDKit version and matching noise (review C3). CTRL_rematch − CTRL_base measures that nuisance directly |
| B5 (vs CTRL_rematch) | Conformer matching with a **heavy-atom** DE objective instead of the H-inclusive one | `VARIANT=heavy` (`standardize_confs.py --heavy_objective`, patch) | TRN `B5_match_heavy` | yes | as B2 | Tests code_concerns #7: H-driven matching gives noisy heavy-torsion targets. A gain here is a cheap TD fix, **not** FlexiTors evidence, and it must be subtracted before attributing residual error to L |
| B4 | Stronger torsion matching (DE maxiter 50 instead of 15) | `VARIANT=de50`; add a TRN line if wanted | — | yes | as B2 | Checks that the floor is due to L, not an under-converged DE (see concerns #7) |
| CTRL | Baseline recipe at the same 100-epoch budget | — | TRN `CTRL_base_100ep` | yes | ~5-10 h | Matched control for every TRN arm |

### (c) Number of diffusion steps ★★

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| C | `--inference_steps` ∈ {5, 10, 20 (R0), 50, 100} | INF `C_steps*` | yes | 0.3-4 h each (roughly linear in steps; RDKit embedding is a fixed cost) | TD Table 9 on DRUGS saturates at about 20 steps. On QM9, check whether AMR at 0.05-0.1 Å still improves with more steps. If it plateaus, the error is from L, not from sampling |

### (d) ODE vs SDE ★★

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| D | `--ode` at 20 and 50 steps | INF `D_ode_*` | yes | as C | Table 8 on DRUGS: ODE ≈ SDE. The patch stops the training yaml's `likelihood: full` from silently turning on the expensive divergence (concerns #1) |

### (e) σ schedule ★

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| E-inf | Inference-only endpoints: `--sigma_min_inf` 0.005π / 0.05π, `--sigma_max_inf` 0.5π. The model's σ embedding keeps its training range | INF `E_inf_*` | yes | ~1 h each | A smaller final σ gives finer torsion resolution. If AMR does not move, the residual error is not torsional |
| E-train | Training with `--sigma_min 0.0157` and with `--sigma_max 1.57` | TRN `E_train_*` | yes | ~5-10 h each | A σ_max below π breaks the uniform-prior assumption; expect it to hurt recall |

### (f) Model size and number of conv layers ★

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| F-irreps | 1st vs 2nd order irreps, using the **released** `qm9_1order` and `qm9_default` checkpoints | INF `P0`, `P1` | yes | ~1 h each | Free reproduction of the Table 8 "first-order irreps" row on QM9 |
| F-depth | `--num_conv_layers` ∈ {2, 3, 4 (CTRL), 6} | TRN `F_layers*` | yes | 4-14 h | Capacity vs floor. If gains saturate while AMR stays at ~0.17 Å, capacity is not the bottleneck; L is |
| F-width | (ns, nv) ∈ {(16,4), (32,8), (64,16)} | TRN `F_width_*` | yes | 4-15 h | Same as F-depth |
| F-radius | `--max_radius` ∈ {3, 5, 7} Å | TRN `F_radius*` | yes | 5-12 h | How much the model needs non-local context (sterics). Relevant to how much the head relies on L |

### (g) Chirality and parity handling ★★ (G1, G-stereo) / ★ (G2)

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| G-stereo | Fraction of generated conformers whose 3D-perceived stereo (CIP centres + E/Z) matches some GT conformer, split into molecules with chiral centres and molecules with stereo double bonds | `tools/stereo_check.py` on R0, A1, A2 and G1 | ANA | no | minutes | Chirality is fixed by the ETKDG seed (never a feature; torsion updates are rigid). Double bonds are "rotatable", so the model can flip E/Z. A wrong stereoisomer can never be matched by `GetBestRMS`. Quantifies how much residual error is a stereo mismatch rather than a geometry error |
| G1 | Seeds embedded from the **GT molecular graph** (atom order; stereo tags now perceived from GT conformer 0's 3D by `make_seed_pickles.py`, review C2 — rebuild `test_gt_seed_mols.pkl` if it predates this) instead of the CSV SMILES | `--seed_mols data/QM9/test_gt_seed_mols.pkl` | INF `G1_gt_graph_seed_mols` | yes | ~1 h | Separates stereo-assignment error from L error. Training seeds are embedded from the GT graph (`standardize_confs.py:72`); test seeds come from SMILES. Any gain here is a train/test asymmetry in stereo, not in L |
| G2 | **Parity-invariant head** (0e instead of 0o) | `--no_parity` (patch) | TRN `G_no_parity` | yes | ~5-10 h | Reproduces the Table 8 "no parity equivariance" collapse. `tools/smoke_test.py` also checks s(−x) = −s(x) for the normal model. **For FlexiTors:** the angle head must be **parity-invariant (0e)** while the torsion head stays 0o. This arm is the negative control for using the wrong symmetry |

### (h) Low-temperature / likelihood-based resampling ★

Upstream has **no** low-temperature or likelihood-based resampling for conformer generation. The likelihood code is only used by the Boltzmann generator. We add it as a post-processing step.

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| H-pools | 4 SDE pools (seeds 11-14) and 4 ODE+exact-likelihood pools (seeds 21-24), each 2K conformers, generated **without** `--no_energy` so that MMFF energies and `euclidean_dlogp` are stored | INF `H_pool_*` | yes | SDE ~1.5 h each; ODE+`--likelihood full` ~5-10× slower (one extra forward pass per torsion per step) | — |
| H0-H3 | From the 4× pool, keep 2K conformers: uniform (control), MMFF Boltzmann weights at 300/1000 K, lowest-MMFF top-k, and full importance weights exp(−E/kT − euclidean_dlogp) | `tools/resample_confs.py` | ANA | no | minutes + evaluation | Precision (COV-P, AMR-P) should rise at some cost to recall. MMFF on frozen RDKit L is a strained energy, so weights may be dominated by L strain. That motivates relaxing L (FlexiTors) before energy-based reweighting. Molecules with 0 rotatable bonds are missing from `--likelihood` runs; they are filled from R0 |

### (i) Per molecule size and per number of rotatable bonds ★★★ (free)

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| I | Stratify COV/AMR (and the sweep) by heavy atoms (≤5 … 9), heavy-atom rotatable bonds (0 … 4+), repo-definition rotatable bonds, and number of GT conformers | `tools/breakdown.py` (run automatically by `gen_eval` and ANA) | no | no | seconds | If AMR is flat or grows **with molecule size at a fixed number of torsions**, the error is L-driven (more angles, more ring atoms), not torsion-driven. Split also by ring content: 3-, 4- and 5-membered rings, which are strained and common in QM9 |

### (j) Local-structure error vs final conformer RMSD ★★★ (free once A0 exists)

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| **J-run** | **Run-matched floor.** For each GT conformer l, the torsions of the top-3 generated conformers of the run are re-optimised against l, keeping their exact local structures (DE with the run's torsions as a candidate, symmetry-aware). `floor_run ≤ observed min RMSD` holds by construction. Seeds, stereo and K are the run's own | `local_structure_analysis.py --mode run` on R0, A0_rdkit, A2, A3, G1 (ANA) | no | ~0.5-2 h per run on 32 cores | Decomposes AMR-R into `floor_run` (removable only by changing L) and `torsion_headroom` = obs − floor_run (removable by a better torsion model or sampler). **FlexiTors is motivated on QM9 only if floor_run is most of AMR-R** and headroom is small. A large headroom falsifies "TD is floor-limited on QM9" |
| **J-decomp** | Split `floor_best_sym` into stereo (`stereo_match_best` false), ring (`min_ring_size` > 0, `n_ring_atoms`), acyclic angles (`floor_best` − `angle_oracle_rmsd`) and acyclic bond lengths (`angle_oracle` − `local_oracle`). Add `floor_gtother` = the floor with a perfect but τ-independent L sampler | columns of `--mode test` (ANA) | no | included in A0-floor | FlexiTors' B^k factor can only remove the **acyclic-angle** share, and C^r the ring share. Stereo errors are out of reach for both, since ±10-15° angle modes cannot invert a centre. If the angle share is small, the B^k design is unmotivated on QM9. If `floor_gtother ≈ floor_best_sym`, the RDKit marginal is fine and only coupling can help. If `floor_gtother ≈ 0`, a better independent L sampler suffices |

Older J (correlational):

| ID | Change | Where | GPU | Est. | Hypothesis |
|---|---|---|---|---|---|
| J | Per molecule: RDKit-vs-GT bond RMSD, angle RMSD, transplant RMSD and DE floor (from A0). Join them with each run's per-molecule AMR/COV; report Spearman ρ and tertile tables. Also report the angle MAE of generated conformers vs matched GT (`geometry_metrics.py`) | `breakdown.py --local_csv` + `geometry_metrics.py` | ANA | no | ~1 h | **Core FlexiTors evidence:** a strong positive ρ between floor and AMR-R, most of TD's AMR explained by the floor, and the largest gap to oracle L (A1) in the worst-floor tertile. Also compare angle MAE for TD (identical to the seed) vs GT-GT variability. FlexiTors should target the difference |

---

## 3. Budgeted plan: 4 days on gnode118, 4 GPUs (user decision)

Hard limits: partition maximum 4-00:00:00 per job; gnode118 has 4 GPUs, so at most 4 × 96 = 384 GPU-h. Target: **≤ 330 GPU-h**, keeping the rest as a failure margin. Training cannot resume, so every training job must finish inside its own `--time`.

### 3.1 What is kept, in priority order

| # | Block | Jobs | GPU-h (nominal / upper) |
|---|---|---|---|
| 1 | Floor and oracle analyses: A0-floor, J-decomp, A0-train (CPU) | `analysis_qm9.sbatch` submission 1 | 0 |
| 2 | Inference on the **released** `qm9_default` (`ablations_inference.tsv`, 23 lines): R0-R2; A0-rdkit, A0-rand; A1 × 3 seeds; A1-sanity, A1-rand; A1c × 3 seeds; A1c-sanity; A2 × 3; A3 × 3; G1 × 3. Then J-run, G-stereo, I and paired comparisons (CPU) | `ablation_inference_array.sbatch` (`--array=0-22%4`) + analysis submission 2 | 18 model lines × 1-2 h + 5 no-model lines × 0.5 h = **20.5 / 38.5** |
| 3 | CTRL_base_100ep × 3 training seeds (+ GT-L evaluation, random and cycle) | TRN lines 0-2 | 3 × 10.5 / 16 = **31.5 / 48** |
| 4 | B1 (train on GT L) × 3 seeds, a Table 8 reproduction (+ GT-L evaluation) | TRN lines 3, 7, 11 | 3 × 10.5 / 16 = **31.5 / 48** |
| 4 | B2 (DE only, random pairing) × 3 seeds, a Table 8 reproduction | TRN lines 8-10 | 3 × 8.5 / 12 = **25.5 / 36** |
| 5 | CTRL_rematch_100ep × 3 seeds (the control for B2 and B5) | TRN lines 4-6 | 3 × 8.5 / 12 = **25.5 / 36** |
| 5 | B5 (heavy-atom matching objective) × 3 seeds | TRN lines 12-14 | 3 × 8.5 / 12 = **25.5 / 36** |
| — | `setup_env.sh` smoke test | — | 0.5 |
| | **Total** | | **≈ 160 nominal / ≈ 243 upper** (≤ 330 target, ≤ 384 hard cap) |

Where the per-run training numbers come from:
- Nominal: 100 epochs × 4.5 min/epoch = 7.5 h, plus about 1 h per `gen_eval`. Runs with GT_EVAL = 1 do three evaluations (RDKit L, GT L, GT L cycle), the others one.
- Upper: 6 min/epoch, plus 2 h per evaluation.
- The epoch time is **not measured**. Check the first epochs of CTRL_base s0 (T + about 16 h). At > 8 min/epoch the upper total passes about 300 GPU-h; drop B5 to 2 seeds first, then CTRL_rematch/B2 to 2. At > 15 min/epoch, 100 epochs no longer fit the 36 h `--time`: stop and re-plan.

**Seed trade-off.**
- 3 training seeds per training arm (15 runs) buy a paired CI that includes training-seed variance for every Table 8 reproduction and for B5. With 1 seed, a small B2 or B5 effect (TD's DRUGS B2 gap is 0.2 COV-R points) cannot be told apart from seed noise.
- The alternative was 1 seed for B1, whose effect TD reports as a collapse (Table 8). That would free about 32 GPU-h (upper), enough for 2 deferred arms (G2 and B3). We kept 3 B1 seeds because the size of the QM9 effect is unknown and B1 also carries the 2 × 2 {train L} × {test L} GT-L evaluations.
- Sampling seeds: 3 for every model-based inference arm (cheap, about 1-2 GPU-h each). Deferred inference arms would get 1 seed.
- Margin: about 87 GPU-h at the upper estimates (about 170 nominal). Spend it only on failed or killed runs first, then on deferred arms in the order of §3.4.

### 3.2 Submission sequence (all from gnode118; `/scratch` is node-local)

```
J0=$(sbatch --parsable setup_env.sh)                                   # GPU 0.5 h
J1=$(sbatch --parsable --dependency=afterok:$J0 download_data.sh)      # CPU; builds seed pickles
J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,VARIANT=std featurize_qm9.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,VARIANT=raw featurize_qm9.sbatch)
AN1=$(sbatch --parsable --dependency=afterok:$J1 analysis_qm9.sbatch)  # submission 1: floor / J-decomp (CPU)
S1=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,VARIANT=rematch standardize_qm9.sbatch)
F1=$(sbatch --parsable --dependency=afterok:$S1 --export=ALL,VARIANT=rematch featurize_qm9.sbatch)
S2=$(sbatch --parsable --dependency=afterok:$S1 --export=ALL,VARIANT=nomatch standardize_qm9.sbatch)
F2=$(sbatch --parsable --dependency=afterok:$S2 --export=ALL,VARIANT=nomatch featurize_qm9.sbatch)
S3=$(sbatch --parsable --dependency=afterok:$S2 --export=ALL,VARIANT=heavy standardize_qm9.sbatch)
F3=$(sbatch --parsable --dependency=afterok:$S3 --export=ALL,VARIANT=heavy featurize_qm9.sbatch)
I=$(sbatch --parsable --dependency=afterok:$J0:$J1 ablation_inference_array.sbatch)   # --array=0-22%4
AN2=$(sbatch --parsable --dependency=afterany:$I:$AN1 analysis_qm9.sbatch)  # submission 2 (after 1: no clobbering)
T=$(sbatch --parsable --dependency=afterany:$I,afterok:$J2:$J3 ablation_train_array.sbatch)  # --array=0-14%4
sbatch --dependency=afterany:$T:$AN2 analysis_qm9.sbatch               # submission 3: training-arm comparisons
```

- **GPU throttle:** the inference array (%4) and the training array (%4) never overlap (`afterany`), so at most 4 GPUs are in use. The analysis, standardize and featurize jobs are CPU-only.
- The `rematch`, `nomatch` and `heavy` caches are not a hard dependency of the training array. Their lines start in waves 2-4, about 16-48 h after the array starts, and the CPU chain (3 × (1-4 h standardize + 1 h featurize)) ends by about T + 18 h. A line that starts before its cache exists fails fast; resubmit it with `--array=<i>`.
- **CPU contention (unverified):** 4 inference tasks × 12 cores + standardize (32) + analysis (32) need about 112 cores. If gnode118 has fewer, the CPU jobs queue behind each other; the schedule still fits because they run in the GPU-bound phase.

### 3.3 Schedule (upper estimates; T = submission of `setup_env.sh`)

| Wall clock | GPU 0 | GPU 1 | GPU 2 | GPU 3 | CPU side |
|---|---|---|---|---|---|
| T+0 – 1 h | setup_env smoke test | — | — | — | — |
| T+1 – 3 h | — | — | — | — | download; featurize std and raw; analysis 1 (floor, runs about 3-8 h) |
| T+3 – 15 h | inference lines 0,4,8,… | lines 1,5,9,… | lines 2,6,10,… | lines 3,7,11,… (23 lines, ≤ 2 h each, about 6 rounds) | standardize and featurize rematch → nomatch → heavy (done by about T+18 h) |
| T+15 – 31 h | CTRL_base s0 | CTRL_base s1 | CTRL_base s2 | B1 s0 | analysis 2 (J-run, stereo, paired; about 4-8 h) |
| T+31 – 47 h | CTRL_rematch s0 | CTRL_rematch s1 | CTRL_rematch s2 | B1 s1 | — |
| T+47 – 63 h | B2 s0 | B2 s1 | B2 s2 | B1 s2 | — |
| T+63 – 75 h | B5 s0 | B5 s1 | B5 s2 | **free: re-runs / margin** | — |
| T+75 – about 85 h | — | — | — | — | analysis 3 (training-arm paired comparisons, breakdowns) |

At the nominal estimates everything ends at about T+55 h. At the upper estimates it ends at about T+85 h, 11 h inside the 96 h window. Array tasks start in index order as slots free up, so in practice the waves overlap rather than run in lock-step.

`--time` per job (all ≤ 4-00:00:00):

| Job | `--time` | Why |
|---|---|---|
| inference task | 6 h | 3 × the 2 h upper estimate |
| training task | 36 h | 16 h upper estimate, with room for 2 × slower epochs |
| `train_qm9_baseline` (deferred) | 3 d | 250 epochs at the upper estimate is about 25 h |
| analysis | 24 h | — |
| standardize | 12 h | resumable |
| featurize | 6 h | — |
| download | 4 h | — |
| setup | 4 h | — |

### 3.4 Deferred (not in the budget; run only from leftover margin, in this order)

| Order | Arm | Why deferred | How to run |
|---|---|---|---|
| 1 | G2 no-parity (TD Table 8 negative control) | a symmetry sanity check, not gap evidence | `ablations_train_deferred.tsv` |
| 2 | B3 (MMFF matching), B4 (DE maxiter 50) | secondary matching variants; B5 covers the matching-noise question | same (also needs `standardize VARIANT=mmff/de50`) |
| 3 | C (steps), D (ODE), E-inf (σ) | sampler checks; about 16 GPU-h at 1 seed | `ablations_inference_deferred.tsv` |
| 4 | P1 (1st-order irreps) | reproduction of a Table 8 row | same |
| 5 | H pools + H0-H3 resampling | ODE+likelihood pools are 5-10 × slower (about 40+ GPU-h) | same, then `analysis_qm9.sbatch` |
| 6 | E-train, F-depth/width/radius | capacity checks, 8 runs × about 12 h | `ablations_train_deferred.tsv` |
| 7 | 250-epoch baseline retrain × 2 | the released checkpoint is used instead (about 50 GPU-h) | `train_qm9_baseline.sbatch` |
| 8 | xTB relaxation (§4) | CPU, but needs an xtb install and a script change | §4 |

Consequence for §5: the falsifier "C/D/E/F move AMR as much as A1 does" is not tested inside the budget. The G1 falsification relies on J-run (`torsion_headroom`) and on A1 − R0 against the R0-R2 spread.

---

## 4. Optional: xTB relaxation metrics

These follow the GEOM-revisited recommendation: GFN2-xTB E_relax and Δ-angle/Δ-length after relaxation.

- Needs an `xtb` binary on gnode118 (conda-forge `xtb`, or the GitHub release tarball unpacked into `/scratch`).
- Upstream `optimize_confs.py` can do the relaxation, but it **hard-codes `data/DRUGS/test_smiles_corrected.csv`** (line 24). Make that path an argument first.
- Plan: a 100-molecule QM9 subset, min(2K, 32) conformers each (the TD Table 3 protocol). Report the median E_relax (energy drop on relaxation) and RMSD(before, after) for TD, A2 (pre-MMFF) and A1 (GT L). This quantifies the strain frozen into RDKit L.
- CPU-only; about 1-3 s per QM9 conformer.

---

## 5. What would count as strong motivation for FlexiTors

**Falsifiers (review C1-C5).** Any one of these undercuts the named claim. Each is testable with the arms above:

| Claim | Falsified if | Arm |
|---|---|---|
| G1: TD on QM9 is limited by RDKit L | `torsion_headroom` (obs − floor_run) is a large fraction of AMR-R; or A1 − R0 is within the R0/R1/R2 spread; or C/D/E/F move AMR as much as A1 does (C-F are deferred, §3.4) | J-run, A1 (C-F deferred) |
| The limit is *angles* (B^k), not stereo or rings | `floor_best − angle_oracle_rmsd` is a small share of `floor_best_sym`; or the floor is concentrated in stereo-mismatched or ring pairs | J-decomp |
| G3: τ↔L coupling matters | A1c ≈ A1 (paired CI includes 0); `floor_gtother` ≈ 0; B2 ≈ CTRL_rematch at small δ | A1c, J-decomp, B2 |
| G2: conformer matching costs accuracy | B5 (heavy objective) or B4 (de50) closes most of the gap. That makes it a cheap TD fix, not a FlexiTors argument | B4, B5 |

Positive evidence would look like this:

- In J-run, `floor_run` is most of R0's AMR-R and the `torsion_headroom` is small. In A0-floor/J-decomp, a large fraction of GT conformers has `floor_best_sym` above small-δ thresholds (0.1-0.25 Å), and a sizeable share of it goes away when acyclic angles are set to GT (`angle_oracle_rmsd`). Comparing `floor_rmsd` (the one-to-one conformer-matching number) with AMR-R is **not** valid evidence (M1).
- A1 (oracle L) improves AMR and small-δ COV well beyond R0, while C/D/E/F barely move it: sampler and capacity are not the bottleneck.
- B1 reproduces Table 8's collapse and B2 shows random pairing ≈ matched. The workaround works but caps accuracy, which argues for learning p(L | τ) instead of projecting.
- In I, error grows with ring content and heavy-atom count at a fixed number of torsions.
- Generated angle MAE is roughly equal to the ETKDG seed error of about 4°, well above GT-GT variability.
- A2/A3 (MMFF) give only partial gains, while H shows energy reweighting on frozen L is dominated by strain.
