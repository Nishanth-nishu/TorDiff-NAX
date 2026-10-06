# QM9 Ablation Results: Torsional Diffusion (interim)

**Updated:** 2 October 2026, 10:15 IST
**Status:** all 23 inference ablations on the released `qm9_default` checkpoint are done. Training ablations are running: CTRL_base seed 0 is done, 3 more are running, 11 are queued. Paired bootstrap statistics are still to come (analysis job 1089).

**Setup:**
- **Data:** GEOM-QM9 test set (1,000 molecules, 13,731 ground-truth conformers). 2K conformers are generated per molecule (K = number of GT conformers).
- **Metrics:** heavy-atom, symmetry-aware RMSD. The **primary threshold is δ = 0.5 Å**, fixed before any results.
- **Software:** RDKit 2022.9.5, PyTorch 1.13.1, RTX 3090 GPUs on gnode118.
- **Seeds:** numbers marked "3 seeds" are means over sampling seeds 0, 1 and 2. The seed-to-seed spread is about ±0.002 Å in AMR-R and ±0.4 points in COV-R.

> **How to read the metrics.** AMR-R (average minimum RMSD, recall) is the mean over GT conformers of the distance to the closest generated conformer. Lower is better. COV-R is the % of GT conformers that have a generated conformer within 0.5 Å. Higher is better. "-P" (precision) versions swap the roles of generated and GT conformers. A *failure* is a molecule for which nothing could be generated. It counts as 0% coverage and is left out of AMR.

---

## 1. Main table (δ = 0.5 Å)

| ID | What the model gets | AMR-R mean | AMR-R median | COV-R | AMR-P mean | COV-P | Failures |
|---|---|---|---|---|---|---|---|
| **Paper** | TorDiff, Table 7 (p. 26) | 0.178 | 0.147 | 92.8 | 0.221 | 92.7 | not reported |
| **Released** | **The paper's own released conformers (`qm9_steps20.pkl`), scored with the repo evaluator** | 0.177 | 0.146 | 88.8 | 0.219 | 84.8 | 69 |
| Released-1o | Same, for the paper's first-order-irreps model | 0.183 | 0.150 | 88.2 | 0.226 | 84.5 | 69 |
| A0 | RDKit conformers only, no model | 0.230 | 0.195 | 83.9 | 0.235 | 85.0 | 65 |
| A0-rand | RDKit geometry, random torsions, no model | 0.242 | 0.211 | 84.9 | 0.318 | 75.2 | 65 |
| **R0–R2** | **TorDiff baseline: RDKit geometry + model (3 seeds)** | **0.176** | **0.143** | **88.6** | **0.220** | **85.0** | 65–69 |
| CTRL s0 | Same, using **our own retrained model** (100 epochs) | 0.179 | 0.147 | 89.0 | 0.225 | 84.9 | 65 |
| G1 | Baseline, with stereo assigned from the GT 3D structure (3 seeds) | 0.175 | 0.145 | 89.0 | 0.219 | 85.4 | 64–67 |
| A2 | RDKit geometry **relaxed with MMFF before** diffusion (3 seeds) | **0.150** | **0.120** | 88.7 | 0.168 | 86.6 | 65–69 |
| A3 | MMFF relaxation **after** diffusion (3 seeds) | 0.186 | 0.149 | 85.3 | 0.164 | 87.2 | 65–69 |
| A1-rand | **True** GT bond lengths and angles, random torsions, no model | 0.157 | 0.100 | 92.8 | 0.219 | 85.3 | 4 |
| **A1** | **True GT bond lengths and angles + model** (random GT conformer per sample; 3 seeds) | **0.081** | **0.038** | **96.1** | **0.102** | **94.1** | 4 |
| A1c | True GT geometry, each sample paired with its own GT conformer (3 seeds) | 0.077 | 0.032 | 96.1 | 0.102 | 94.1 | 4 |
| sanity | GT conformers passed straight through, cycled | 0.000 | 0.000 | 99.7 | 0.000 | 99.7 | 3 |

The released pickles hold only 931 molecules (25,318 conformers). They were read with a custom unpickler, because the Python attributes the original run attached to each RDKit `Mol` cannot be unpickled by any installed RDKit version (`slurm/eval_released.sbatch`).

---

## 2. What we found, and why it matters

### 2.1 The pipeline reproduces the paper
- **Average error matches:** our baseline gets AMR-R 0.176 Å (median 0.143) against the paper's 0.178 (0.147), and AMR-P 0.220 against 0.221.
- **Training reproduces too:** our own retrained model (CTRL s0) gets 0.179 Å, so both inference and training match the paper.
- **The evaluator is correct:** passing GT conformers straight through gives exactly 0.000 Å, which confirms that the evaluator and the GT-seed files are right.

### 2.2 Coverage is lower than the paper because of RDKit failures
- **What happens:** 65 of the 1,000 test molecules produce no conformers at all and count as 0% coverage. Our COV-R is therefore 88.6% against the paper's 92.8%. Leaving the failures out, COV-R would be about 95%.
- **Why they fail:**
  - **60 molecules:** RDKit's ETKDG cannot embed strained polycyclic cages, for example `C1C[C@]23C[C@H](C2)[C@@H]2[C@H]1[C@@H]23`. TorDiff starts from an RDKit structure, so it fails on these too.
  - **5 molecules:** rejected inputs such as the multi-fragment `C=C1C(=O)C=NN1C.N`.
- **Why it matters:** this is direct evidence for gap 1. RDKit's local geometry is not just imprecise; on strained systems it does not exist, and TorDiff cannot run at all. A1, which uses true geometry, fails on only 4 molecules.
- **The paper's run had the same failures.** Its released conformers cover only 931 of the 1,000 molecules (69 failures). Scored with the repo's own evaluator, they give COV-R **88.8%** and AMR-R 0.177 Å, almost identical to our runs (88.6%, 0.176 Å).
- **So the paper's 92.8% cannot be reproduced from its own released samples.** Most likely the published number left failures out of the coverage average, or came from a different evaluator version. The paper never reports failure counts.
- **Consequence:** our pipeline is a faithful reproduction. Comparisons in this study use our own numbers throughout, never the paper's 92.8.

### 2.3 RDKit's local geometry accounts for about half of TorDiff's error on QM9 (A1 vs R0)
- **Result:** with true bond lengths and angles, the same pretrained model's AMR-R falls from **0.176 to 0.081 Å (−54%)**, and the median from 0.143 to 0.038 Å (about 3.7× lower). COV-P rises from 85 to 94%.
- **The torsions aren't the problem:** with true geometry, *random* torsions and no model at all (A1-rand, 0.157 Å) already beat TorDiff on RDKit geometry (0.176 Å). On QM9, the frozen local structure costs more than the torsion model fixes.
- **Why it matters:** this is the core FlexiTors motivation. The remaining error in TorDiff is dominated by the bond geometry it cannot change.
- **Caveat:** A1 is evaluated on 996 molecules and R0 on 935, because the 61 cage molecules RDKit can't build are in A1 only. Those are among the hardest molecules, so this works *against* A1. The paired comparison on common molecules (job 1089) will give the exact effect with confidence intervals.

### 2.4 Cleaning up RDKit's geometry recovers part of that error (A2)
- **Result:** relaxing the RDKit structure with the MMFF force field before diffusion lowers AMR-R from 0.176 to **0.150 Å (−15%)**, consistently across 3 seeds. That is better than OMEGA's 0.177 in the paper, with no retraining.
- **Why it matters:** this is a cheap, physics-based correction of local structure, and it already closes about a quarter of the gap to true geometry (0.176 → 0.150, against the 0.081 oracle). A learned local-structure correction (FlexiTors) aims at the rest.
- **Caveat:** MMFF moves torsions slightly as well as bond geometry. The J-decomp analysis (job 1089) separates the two.

### 2.5 Relaxing *after* diffusion does not help recall (A3)
- **Result:** MMFF after sampling makes AMR-R worse (0.186) but AMR-P better (0.164). Relaxation pulls conformers into nearby energy minima, so they become more precise but less diverse.
- **Why it matters:** local structure has to be right *during* generation, not patched afterwards. That argues for diffusing it jointly with torsions, as FlexiTors proposes.

### 2.6 Coupling between torsions and bond geometry has a small but consistent effect on QM9 (A1c vs A1)
- **Result:** giving each sample the bond geometry from its *own* GT conformer, rather than a random one, lowers AMR-R from 0.081 to 0.077 Å (−5%), and the median from 0.038 to 0.032 Å. All 3 seeds agree.
- **Why it matters:** this is evidence for gap 3. On small QM9 molecules the effect is modest. Most of the gain comes from having *any* correct geometry (−54%) rather than geometry matched to the torsions (a further −5%).
- **Prediction:** the coupling effect should be larger on GEOM-DRUGS, where the paper's App. F.1 points that way (0.324 → 0.284 Å).

### 2.7 Stereo mismatch is not a factor (G1 ≈ R0)
- **Result:** assigning stereochemistry from the GT 3D structure changes nothing (0.175 vs 0.176 Å).
- **Why it matters:** the A1 gain is not a stereo artefact.

---

## 3. Bugs found and fixed along the way

All fixes are on branch `flexitors-ablation-hooks` (`patches/0001-ablation-hooks.patch`).

| Problem | Symptom | Fix |
|---|---|---|
| `log_det_jac` on molecules with 0 rotatable bonds | SVD crash on an empty Jacobian; every inference run died | Return log det = 0 (commit 392e6f4) |
| Multi-fragment GT seed molecules | `mask_rotate` assertion in the true-geometry runs | Reject them, as the SMILES path already does (04a799e) |
| `RemoveAllConformers` called on a `None` seed | G1 crash | Guard added (e809b45) |
| Featurization through `multiprocessing.Pool` | Shared-memory mmap limit hit at 106k molecules; job hung for 6 h | Featurize in a single process (about 8 min) |
| Job requests vs node limits (5 GB/core cap, 48 cores) | GPU jobs blocked by CPU jobs | Smaller requests; analysis jobs on 8 cores |
| `srun` without `-n 1` | Commands ran twice (git race, doubled logs) | Always `-n 1` |

**Cost:** about 8 h of wall-clock time lost to the hung featurize job, plus about 10 h with one GPU idle (CPU contention during standardization). The schedule still fits the 4-day window.

---

## 4. Still running / next

- **Training arms (job 1036):**
  - CTRL_base s1/s2 and B1 s0 are near the end; about 10.7 h per run (100 epochs plus evaluation).
  - Still to run: CTRL_rematch ×3, B1 s1/s2, B2 ×3, B5 ×3.
  - Expected finish: about 3 October, evening.
- **Analysis 2 (job 1089):** paired bootstrap CIs on common molecules (A1, A1c, A2 and G1 vs R0), the RDKit floor run on these conformers (J-run), and the decomposition of the floor by angle, length, ring and stereo (J-decomp).
- **Analysis 3 (job 1090):** the training-arm comparisons, B1 and B2 against the matched baselines.
- **Reference (job 2599):** the paper's released conformers scored with our evaluator.
