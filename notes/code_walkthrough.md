# Torsional Diffusion: code walkthrough (focus on QM9 and the frozen local-structure assumption)

Repo: https://github.com/gcorso/torsional-diffusion, cloned to `C:\Users\HP\Desktop\tor_diff\torsional-diffusion`.
Upstream HEAD is `5f713b42d7000307655f272471014c6127ea59be` (2023-10-10). The clone is left on `master`, untouched.
Our changes are on local branch `flexitors-ablation-hooks` and exported as `patches/0001-ablation-hooks.patch`.
**All `file:line` references below are to upstream `master`.**

The code is small: 2,642 lines of Python in 20 files. Everything runs from the repo root with relative paths such as `data/...` and `.p.npy`.

---

## 0. TL;DR

| Stage | Entry point | Key functions |
|---|---|---|
| Conformer matching (offline, CPU) | `standardize_confs.py` | `conformer_match` (52-121), `utils/standardization.py` (`optimize_rotatable_bonds` 29-43, `get_von_mises_rms` 175-182) |
| Featurisation + cache | `train.py` → `utils/dataset.py:construct_loader` (240-265) | `ConformerDataset.preprocess_datapoints` (80-114), `filter_smiles` (116-174), `utils/featurization.py:featurize_mol` (40-101) |
| Forward noising | `utils/dataset.py:TorsionNoiseTransform` (18-47) | `utils/torsion.py:modify_conformer` (57-75) |
| Score model | `diffusion/score_model.py:TensorProductScoreModel` (48-203) | e3nn `FullyConnectedTensorProduct` conv layers + bond-centred pseudoscalar head |
| Loss | `utils/training.py:train_epoch` (7-31) | `diffusion/torus.py:score` (37-46), `score_norm` (73-77) |
| Sampling | `generate_confs.py` → `diffusion/sampling.py:sample` (96-222) | `embed_seeds` (46-82), `perturb_seeds` (85-93) |
| Likelihood | `diffusion/likelihood.py` | `divergence_full` (24-39), `divergence_hutch` (42-56), `log_det_jac` (77-96) |
| Evaluation | `evaluate_confs.py` | `calc_performance_stats` (35-41), `GetBestRMS` (112) |

---

## 1. File-by-file map

### Top-level scripts
| File | Lines | Role |
|---|---|---|
| `train.py` | 116 | Builds the model, or restores it from `--restart_dir` (85-91; loads `best_model.pt` weights only, with no optimizer or epoch). Calls `construct_loader` (103) and `train()` (20-48). Each epoch runs train then val. `ReduceLROnPlateau.step(val_loss)` (33-34). Saves `best_model.pt` when val loss improves (36-39) and `last_model.pt` every epoch (41-46). Writes all args to `model_parameters.yml` (109-110). `boltzmann_train` (51-77) is only for the Boltzmann generator. |
| `generate_confs.py` | 185 | Inference. It loads `model_parameters.yml` **into `args`, overwriting CLI args with the same name** (76-78). It reads the test CSV positionally as `(raw_smi, n_confs, smi)` (154). By default it generates `2*n_confs` conformers (158). For each molecule: `get_seed` → `embed_seeds` (RDKit ETKDG, `embed_func` 58-60, `numThreads=5`) → `perturb_seeds` (uniform torsions) → `sample` → `pyg_to_mol` → `populate_likelihood` (146-147; MMFF energy is always computed). It pickles `{smi: [RDKit mols]}` (182-184). Flags: `--ode`, `--likelihood full/hutch`, `--pre_mmff`, `--post_mmff`, `--no_random`, `--no_model`, `--seed_confs`, `--seed_mols`, `--single_conf`, `--confs_per_mol`, `--inference_steps`, and the particle-guidance `--pg_*` flags. |
| `evaluate_confs.py` | 157 | Computes COV/MAT recall and precision against `test_mols.pkl`, covered in §6. |
| `standardize_confs.py` | 166 | Conformer matching, covered in §3. It writes one pickle per worker: `out_dir/{worker_id:03d}.pickle`, a dict `{name: mol_dic}` where `rd_mol` is replaced by the matched RDKit conformer. |
| `optimize_confs.py` | 50 | MMFF or xTB relaxation of generated conformers, plus properties. DRUGS paths are hard-coded (`data/DRUGS/test_smiles_corrected.csv`, line 24). Not needed for QM9. |
| `test_boltzmann.py` | 82 | ESS evaluation of the torsional Boltzmann generator (DRUGS). Not used here. |

### `diffusion/`
| File | Role |
|---|---|
| `torus.py` | Wrapped-normal machinery on the torus. `p` and `grad` (6-17) sum over ±N periodic images. Tables use a log-spaced grid: x in [1e-5π, π] with 5001 points, σ in [3e-3π, 2π] with 5001 points (20-24). They are cached to **`.p.npy` and `.score.npy` in the CWD** (26-34), each 5001×5001 float64, about 200 MB. `score(x, σ)` (37-46) is a table lookup with nearest-index rounding. `score_norm_` (66-70) is a Monte-Carlo estimate of E[s²] from 10,000 samples per σ, recomputed on every import. `score_norm(σ)` (73-77). |
| `score_model.py` | The e3nn model, covered in §4. |
| `sampling.py` | `get_seed` (29-43), `embed_seeds` (46-82), `perturb_seeds` (85-93), `sample` (96-222: reverse SDE/ODE plus particle guidance), `pyg_to_mol` (225-250), `InferenceDataset.apply_torsion_and_update_pos` (266-274). |
| `likelihood.py` | ODE divergence: `divergence_full` does n_torsions extra forward passes with finite differences (eps=0.01). `divergence_hutch` is a single Rademacher probe. `log_det_jac` (77-96) is the torsion→Cartesian Jacobian after removing rigid rotation. `populate_likelihood` (112-128) computes `euclidean_dlogp = dlogp − ½·log|det I| − log_det_jac` (119) and MMFF94s energy. `free_energy` (100-109) uses `kT=0.592` (99). |

### `utils/`
| File | Role |
|---|---|
| `parsing.py` | Training arguments (8-52). The defaults are for **DRUGS**: `--data_dir data/DRUGS/drugs/`, `--std_pickles data/DRUGS/standardized_pickles`, `--split_path data/DRUGS/split.npy`, `--dataset drugs`, `--in_node_features 74`, `--use_second_order_repr False`. QM9 needs `--dataset qm9 --in_node_features 44`. To match the released QM9 model you also need `--use_second_order_repr`. |
| `dataset.py` | `TorsionNoiseTransform` (18-47), `ConformerDataset` (50-237), `construct_loader` (240-265). Covered in §2. |
| `featurization.py` | `featurize_mol` (40-101) builds node features: one-hot type, then atomic number, aromatic, degree, hybridisation, implicit valence, formal charge, ring sizes 3-8, and number of rings. That is 5+39 = **44 for QM9** (`qm9_types` H,C,N,O,F, line 20) and 35+39 = 74 for DRUGS. Edge features are a 4-way bond-type one-hot. `chiral_tag` is computed (59, 63) but **never used**. `featurize_mol_from_smiles` (104-131) runs `MolFromSmiles` then `AddHs`, and filters fragments (`.`), molecules with no 4-atom path, and N<4. |
| `torsion.py` | `get_transformation_mask` (9-39) defines the degrees of freedom (the rotatable bonds). `modify_conformer` (57-75) is a rigid rotation of one side of each bond. `perturb_batch` (78-103). `get_torsion_angles` (110-124) is used only by particle guidance. |
| `standardization.py` | Conformer-matching primitives (EquiBind/GeoMol code): `SetDihedral` (19-20), `optimize_rotatable_bonds` (29-43; differential evolution), `OptimizeConformer.score_conformation` (57-60; the objective is `AlignMol` **with H**), `get_torsion_angles` (63-83), von Mises torsion averaging (149-182), `mmff_func` (185-192), `clean_confs` (195-209). |
| `training.py` | `train_epoch` and `test_epoch` (7-62), covered in §5. |
| `utils.py` | `get_model` (9-17), Adam plus `ReduceLROnPlateau(factor=0.7, patience=20, min_lr=lr/100)` (20-33), `time_limit` (64-73; SIGALRM, so Unix only). |
| `boltzmann.py` | Boltzmann-generator resamplers (not used for QM9 conformer generation). |
| `xtb.py` | xTB wrappers. **Creates `/tmp/<pid>` at import time** (lines 5-7). It is imported transitively by `generate_confs.py` through `likelihood.py`. |
| `visualise.py` | PDB trajectory dumper (`--dump_pymol`). |

---

## 2. Data pipeline (GEOM-QM9)

**Download.** The README only gives the Drive folder (`https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7`) and states no sizes. I measured them on 2026-09-29 from the Drive headers:

| File | Size |
|---|---|
| `qm9.tar.gz` (id `1c75BAevYvoShxyxZZw0o4-zIQlyj_-H9`) | **807,551,358 B (0.81 GB)**. Top-level dir `QM9/`; the first member is `QM9/test_mols.pkl` (7.75 MB) |
| `drugs.tar.gz` | 25,308,038,362 B (25.3 GB). Not needed |
| `xl.tar.gz` | Not needed |
| `workdir/qm9_default/` (2nd-order irreps) and `workdir/qm9_1order/` | `best_model.pt` (6.4 MB), `model_parameters.yml`, `qm9_steps20.pkl` |

- The expected layout comes from the README and GitHub issues #12 and #20; I could not list the whole archive. It is: `data/QM9/qm9/*.pickle` (raw GEOM `rdkit_folder` pickles, one per molecule, holding `{'smiles', 'conformers': [{'rd_mol', 'boltzmannweight', ...}]}`), `data/QM9/standardized_pickles/NNN.pickle`, `data/QM9/split.npy`, `data/QM9/test_smiles.csv`, and `data/QM9/test_mols.pkl`. `slurm/download_data.sh` prints the real layout.
- **Splits.** These are GeoMol's. `split.npy` is `[train_idx, val_idx, test_idx]`, and the indices point into **`sorted(glob(data_dir/*.pickle))`** (`dataset.py:82-87`). So the split depends on exactly which raw files are present. Issue #20 reports `IndexError: index 133112 ... size 133111` with the QM9 files from Drive; the patch drops out-of-range indices.
- **Standardized pickles** are located by position: `pickle_id = split_index // 1000` (`dataset.py:81, 91`). This is hard-coded and matches `standardize_confs.py --jobs_per_worker 1000`. Inside a pickle the key is `path[len(root):-7]` (`dataset.py:91`, `standardize_confs.py:153`). **`--data_dir` must therefore end in `/`** (issue #12: without it you get `num_samples=0`; the patch normalises it).
- **Filtering** (`filter_smiles`, 116-174) removes: missing std pickle or molecule, `.` in the SMILES, RDKit parse failure, no 4-atom path, N<4, featurisation failure (no conformer whose non-isomeric canonical SMILES matches), and **no rotatable bonds**. Counts are printed as `self.failures`.
- **Per molecule** (`featurize_mol`, 190-231): it keeps every conformer (up to 30 from matching) whose non-isomeric SMILES matches. `data.pos` is a *list* of coordinate tensors (212), and `data.weights` are normalised Boltzmann weights. `correct_mol`, the last kept conformer, is featurised.
- **Cache.** The entire list of `Data` objects is pickled to `<--cache>.train` and `<--cache>.val` (62-74). **Any existing file at that path is reused without checking the arguments.** The README says the first build takes "about 2h on single core CPU"; `--num_workers` uses a `multiprocessing.Pool`.
- **Loader.** A PyG `DataLoader` with `batch_size=32` and `shuffle=True` (257-259). **It has no `num_workers`**, so the per-sample numpy work in `TorsionNoiseTransform` (deepcopy, `modify_conformer`) runs in the main process. The patch adds `--loader_workers`.
- **Where the RDKit local structures come from.** In training, they are the conformer-matched pickles, i.e. RDKit seeds (§3). At test time they come from fresh ETKDG embeddings of the CSV SMILES (`generate_confs.py:58-60`, `sampling.py:46-82`).

---

## 3. Conformer matching (`standardize_confs.py` + `utils/standardization.py`)

For each raw GEOM molecule, `conformer_match(name, confs)` (52-121) does the following:
1. `clean_confs` keeps at most `--confs_per_mol` (default 30) GT conformers whose non-isomeric SMILES matches (59-66). Optionally it sorts or resamples them by Boltzmann weight (`--boltzmann top|resample`).
2. `rotable_bonds = get_torsion_angles(mol)` (73). This covers every bond whose removal disconnects the graph with ≥2 atoms on the smaller side, so it **includes CH3, OH and NH2 rotors and non-ring double/triple bonds**. The dihedral atoms are `(n0[0], u, v, n1[0])`.
3. **Local structures are created here.** A deep copy of the GT mol (same atom order and graph) has its conformers removed, and then `AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n_confs)` runs (72-78, with no random seed). `--mmff` MMFF-relaxes these seeds (82-86).
4. **Assignment.** `cost[i,j]` is the RMSD (with H) after setting seed j's torsions to von-Mises-averaged torsions of GT i (`get_von_mises_rms`, `standardization.py:175-182`). The Hungarian algorithm solves the assignment (88-92). **`--no_match` skips only this assignment step** (93-94); the torsions are still optimised.
5. **Torsion optimisation.** For each (GT i, seed j) pair, `optimize_rotatable_bonds` runs differential evolution (popsize 15, maxiter **15**) over *all* torsions. The objective is `AlignMol(seed, GT)` **including hydrogens** (`standardization.py:57-60`), and only `SetDihedral` is applied (105-106). The heavy-atom RMSD left over is stored as `conf['rmsd']` (107, 116). **`conf['rd_mol']` is replaced by the RDKit seed with matched torsions** (115).
6. Output: `{name: mol_dic}` per worker, 1000 molecules per worker (163-166).

The result is that the training conformers have **RDKit bond lengths, bond angles and ring puckers, with GT-like torsions**. Their heavy-atom RMSD to GT is `conf['rmsd']`. This is the "local-structure error" that torsion diffusion can never remove.

---

## 4. Score model (`diffusion/score_model.py`)

- **Inputs.**
  - `build_conv_graph` (179-203): the edges are the chemical bonds plus a radius graph (`radius_graph(pos, 5 Å)`, 181). Bond edges carry the 4-dim bond one-hot; radius edges get zeros (183-186).
  - σ is embedded as `log(σ/σ_min)/log(σ_max/σ_min)·10000` followed by a 32-dim sinusoidal embedding (188-189). It is concatenated to the node features and to the edge features.
  - Distances go through a 50-Gaussian smearing on [0, 5] Å (196-197). Edge spherical harmonics use l≤2 (201).
- **Convolutions.** `num_conv_layers` (default 4) `TensorProductConvLayer`s (14-45). Each is an e3nn `FullyConnectedTensorProduct(node_attr[dst] ⊗ SH(edge))` whose weights come from a 2-layer MLP on `[edge_attr, src scalars, dst scalars]` (22-28, 138). Aggregation is `scatter(..., reduce='mean')` (37), followed by residual zero-padding (38-40) and e3nn `BatchNorm` (42-43).
- **Irreps sequence** (79-92):
  - 1st order: `ns×0e` → `+nv×1o` → `+nv×1e` → `+ns×0o`.
  - 2nd order (`--use_second_order_repr`, used by the released `qm9_default`): adds `2e/2o`.
  - Defaults are ns=32, nv=8.
- **Torsion head** ("extrinsic-to-intrinsic", 141-155):
  - For each rotatable bond (`edge_index[:, edge_mask]`), it takes the bond centre and every atom within `max_radius` (`radius`, 163-177).
  - Edge SH are tensored with the l=2 SH of the bond axis (`final_tp = FullTensorProduct(SH, "2e")`, 113, 143-147).
  - A final TP conv (`bond_conv`) outputs **`ns×0o` pseudoscalars** (115-122). This means the torsion score flips sign under reflection, as a torsion should.
  - Then comes a bias-free `Linear-Tanh-Linear` MLP (124-128, which is odd, so parity is preserved).
- **Output.** `data.edge_pred` has one scalar per rotatable bond (154). `data.edge_sigma = node_sigma[src][edge_mask]` (155). With `scale_by_sigma`, which is always True (the flag is `store_true` with default True, `parsing.py:39`), the output is multiplied by `sqrt(E[s²](σ))` (157-160).

---

## 5. Diffusion on the hypertorus + training loop

- **Forward process** (`TorsionNoiseTransform.__call__`, `dataset.py:24-43`):
  - Pick one conformer uniformly (29), or Boltzmann-weighted with `--boltzmann_weight`.
  - Draw one σ per molecule, log-uniform in [σ_min, σ_max] (37). Defaults are `0.01·3.14` and `3.14`, which are the released QM9 values (0.0314 and 3.14).
  - Draw Δτ ~ N(0, σ²) independently per rotatable bond (40), apply it with `modify_conformer` (41), and store it in `data.edge_rotate` (42).
- **Target and loss** (`training.py:19-25`):
  - s = ∇log p_wrapped(Δτ; σ), read from the precomputed table (`torus.score`, which wraps Δτ into [-π, π)).
  - loss = mean over all torsions in the batch of (s − ŝ)² / E[s²](σ). Molecules with more torsions therefore weigh more.
  - The "base loss" is the same expression with ŝ = 0.
- **Optimisation.** Adam with lr 1e-3, batch 32, 250 epochs, plateau scheduler on the val loss. Model selection uses the best val loss (`train.py:36-39`).
- **Sampling** (`sampling.py:96-222`):
  - The σ schedule is `10**linspace(log10 σ_max, log10 σ_min, steps+1)[:-1]` (105). It is geometric and the last step is **above** σ_min; there is no final denoising step.
  - `dt = 1/steps` (106) and `g = σ·sqrt(2·ln(σ_max/σ_min))` (166).
  - SDE: Δτ = g²·dt·ŝ + g·sqrt(dt)·z (180). ODE: Δτ = ½·g²·dt·ŝ (175).
  - With `--likelihood` and `--ode`, dlogp += −½·g²·dt·div (176-178).
  - Particle guidance, when the `--pg_*` flags are set, is at 182-208.
  - Updates are applied with `apply_torsion_and_update_pos` (210, 266-274), and the batch is re-sent to the GPU every step.
  - The prior is uniform torsions from `perturb_seeds` (`sampling.py:85-93`) applied to ETKDG seeds.
- **Likelihood.** `euclidean_dlogp` (`likelihood.py:119`) converts the torus density to a density over Cartesian coordinates modulo rigid motion, using the inertia tensor and the torsion Jacobian. That Jacobian is computed at a **fixed local structure**. It is used for Boltzmann weights `exp(−E/kT − euclidean_dlogp)` in `free_energy` and `boltzmann.py:46`. **Conformer generation has no low-temperature or likelihood-based resampling.**

---

## 6. Evaluation and metrics (`evaluate_confs.py`)

- **Inputs.**
  - `--confs`: the generated pickle, keyed by the **corrected SMILES** (CSV column 3).
  - `--test_csv`: needs the columns `smiles` and `corrected_smiles` (54-55).
  - `--true_mols`: keyed by the **`smiles`** column (83).
- The GT conformers are filtered with `clean_confs`, which uses non-isomeric canonical SMILES (44-51, 83).
- **RMSD.** `AllChem.GetBestRMS(RemoveHs(true), RemoveHs(gen))` (112). This is a heavy-atom, symmetry-aware RMSD that allows **proper rotations only**, so an enantiomer is not matched. `--only_alignmol` uses `AlignMol` instead, which is not symmetry-aware.
- **Metrics** (`calc_performance_stats`, 35-41). For each molecule, M is the (n_true × n_gen) RMSD matrix:
  - COV-R(δ) = mean over GT of [min over gen < δ]
  - MAT-R = mean over GT of (min over gen)
  - COV-P(δ) = mean over gen of [min over GT < δ]
  - MAT-P = mean over gen of (min over GT)
  - The script prints the mean and median over molecules.
- **Thresholds.** `threshold = np.arange(0, 2.5, 0.125)` (32). The script prints **every** threshold from 0 to 2.375 Å. **No per-dataset threshold is hard-coded.**
  - The `--dataset` flag only matters for `xl` (75-76).
  - **QM9 uses δ = 0.5 Å** (the paper's convention; DRUGS uses 0.75 Å). You have to read the `threshold 0.5` block of the output. It is exactly representable, since 0.5 = 4×0.125, and the comparison is strict (`<`).
  - The patch adds a one-line `SUMMARY` at 0.5 for `--dataset qm9` and `--out_results` to dump the per-molecule matrices.
- **Failures.** A molecule missing from `--confs` is a "model failure". It counts as 0 coverage in COV (`+ [0]*num_failures`, 149-150; failure detected at 78-81) but is **excluded from MAT** (`nanmean`). Molecules whose GT set becomes empty after `clean_confs` are silently skipped (85-87).
- **K.** `generate_confs.py` makes 2×n_true conformers by default (158), i.e. K = 2L as in the paper.

---

## 7. Exact QM9 commands

The README is written for DRUGS. These are the QM9 equivalents; the flags come from `parsing.py`, the released `qm9_default/model_parameters.yml` (see §9), and issues #12 and #20.

```bash
# environment (README): conda env create -f environment.yml; conda activate torsional_diffusion; pip install e3nn
#   (environment.yml only pins python=3.9, rdkit, pyaml, matplotlib -> see §8 and slurm/setup_env.sh)

# data: download qm9.tar.gz (0.81 GB) from the Drive folder and extract into data/  -> data/QM9/...

# (optional) redo conformer matching; QM9 has ~133k raw pickles -> 134 workers of 1000
for i in $(seq 0 133); do
  python standardize_confs.py --out_dir data/QM9/standardized_pickles --root data/QM9/qm9/ \
         --confs_per_mol 30 --worker_id $i --jobs_per_worker 1000 &
done

# train (paper QM9 model = 2nd-order irreps; trailing slash on data_dir is required)
python train.py --log_dir workdir/qm9_default --dataset qm9 --in_node_features 44 --use_second_order_repr \
    --data_dir data/QM9/qm9/ --std_pickles data/QM9/standardized_pickles --split_path data/QM9/split.npy \
    --cache data/QM9/cache --num_workers 16

# sample (2K conformers per test molecule, 20 steps)
python generate_confs.py --test_csv data/QM9/test_smiles.csv --inference_steps 20 \
    --model_dir workdir/qm9_default --out workdir/qm9_default/qm9_steps20.pkl --tqdm --batch_size 128 --no_energy

# evaluate (read the 'threshold 0.5' block)
python evaluate_confs.py --confs workdir/qm9_default/qm9_steps20.pkl --test_csv data/QM9/test_smiles.csv \
    --true_mols data/QM9/test_mols.pkl --n_workers 10 --dataset qm9
```

Notes:
- The README's DRUGS evaluate command uses `--confs .../drugs_steps20.pkl` while the generate command writes `drugs_20steps.pkl`, so the names don't match.
- A custom `smiles.csv` **must have a header row**, because `pd.read_csv` eats the first line (issue #8).

---

## 8. Dependency versions

The repo pins almost nothing:
- `environment.yml` lists `python=3.9, rdkit, pyaml, matplotlib`. The pytorch/pyg channels are listed, but no packages from them.
- The README adds `pip install e3nn` and "install pyg matching your torch".
- Issue #8 reports a working install with torch 1.13 + cu117 + current PyG wheels.

Imports that no file declares: `torch`, `torch_geometric`, `torch_scatter`, `torch_cluster` (and `torch_sparse` for PyG < 2.3), `e3nn`, `spyrmsd` (imported at module level in `sampling.py`, so it is required even without particle guidance), `rmsd` (`standardization.py`), `networkx`, `pandas`, `scipy` (`bootstrap` needs ≥1.7), `tqdm`, and `pyyaml`. xTB is optional.

**Pinned set used in `slurm/setup_env.sh`** (wheel availability checked against download.pytorch.org, data.pyg.org and PyPI):

| Package | Profile `cu117` (default) | Profile `cu118` (Ada/Hopper GPUs) |
|---|---|---|
| python | 3.9 (≥3.9 required: dict `|` in `standardize_confs.py:160`) | 3.9 |
| torch | 1.13.1+cu117 | 2.0.1+cu118 |
| torch_geometric | 2.0.4 | 2.3.1 |
| torch_scatter / torch_sparse / torch_cluster | 2.1.1 / 0.6.17 / 1.6.1 (+pt113cu117) | 2.1.2 / 0.6.18 / 1.6.3 (+pt20cu118) |
| e3nn | **0.5.1** (≤0.5.0 crashes in `BatchNorm` on the 0o-only `bond_conv`, issue #3) | 0.5.1 |
| rdkit | 2022.9.5 | 2022.9.5 |
| numpy / scipy / pandas / networkx | 1.23.5 / 1.10.1 / 1.5.3 / 2.8.8 | same |
| spyrmsd / rmsd | 0.5.2 / 1.5.1 | same |

`torch_geometric.data.DataLoader` (imported in `dataset.py:10`) is still re-exported in PyG 2.0.4 through 2.5, which I checked in the source.

---

## 9. Released QM9 checkpoints (`workdir/qm9_default/model_parameters.yml`, read from Drive)

The released `qm9_default` config:
- `dataset: qm9`, `in_node_features: 44`, `num_conv_layers: 4`, `ns: 32`, `nv: 8`, `use_second_order_repr: true`
- `sigma_min: 0.0314`, `sigma_max: 3.14`, `lr: 0.001`, `batch_size: 32`, `n_epochs: 250`, `scheduler: plateau` (patience 20), `max_radius: 5.0`, `seed: 0`
- `std_pickles: data/QM9/standardized_multi_qm9/`, `split_path: data/QM9/splits/split0.npy`. These are internal paths that are not in the public layout.

It also has keys the public code doesn't know about (`aggregator`, `tp_aggregator`, `num_intermediate_conv_layers`, `pos_enc_dim`, `no_hydrogen`, `cache_path`), so **it was trained with an earlier internal version of the code**. `qm9_1order` is the same with `use_second_order_repr: false`. Neither file has a `likelihood` key.

---

## 10. Where the "frozen RDKit local structure" assumption lives

This is what FlexiTors has to change. Local structure (L) means bond lengths, bond angles, ring conformations and stereo. In this code, L is decided once by RDKit and then only rigid torsion rotations are ever applied.

| # | Location | What it freezes | What FlexiTors would change |
|---|---|---|---|
| 1 | `standardize_confs.py:72-78` (`EmbedMultipleConfs`), `82-86` (`--mmff`) | **Training L** is the ETKDG embedding of the GT graph | Keep GT L, or match angles as well (§3) |
| 2 | `standardize_confs.py:88-118` + `utils/standardization.py:19-26, 29-43, 57-60` | Matching optimises **only dihedrals** (`SetDihedral`). Residual `conf['rmsd']` = error from L | Optimise torsions plus bond angles, or skip matching and use GT directly |
| 3 | `utils/dataset.py:212` (`pos` from matched mols) and `utils/dataset.py:37-42` | The noise is applied **only to torsions**. `edge_rotate` is the only target | Add angle noise (e.g. a Gaussian on θ−θ₀, or a wrapped/reflected distribution on (0, π)) and an `angle_update` target |
| 4 | `utils/torsion.py:9-39` `get_transformation_mask` | The degrees of freedom are exactly the rotatable bonds (`edge_mask`, `mask_rotate`) | Add angle DOFs: triplets (i, j, k) plus a mask of the atoms that move when θ_ijk bends |
| 5 | `utils/torsion.py:57-75` `modify_conformer` (and `perturb_batch` 78-103) | A rigid rotation about the bond axis **preserves every bond length and angle** | An angle-bending operator: rotate the k-side about the axis n = (r_ji × r_jk) at atom j. Care is needed with rings (not tree-like) and the order of operations |
| 6 | `diffusion/score_model.py:141-160` | The head outputs one **0o pseudoscalar per rotatable bond**. `out_nodes = edge_mask.sum()` | Add an angle head. The angle score is **parity-even (0e)**, e.g. a centre at atom j with SH of both bond vectors |
| 7 | `utils/training.py:19-25` + `diffusion/torus.py` | The loss is only over torsion scores. `torus` is the only manifold | Joint loss: torus score + angle score (separate σ schedule; angles need a much smaller σ_max, a few degrees) |
| 8 | `generate_confs.py:58-60` (`embed_func`) and `diffusion/sampling.py:46-82` (`embed_seeds`), `get_seed` 29-43 | **Test-time L** is ETKDG from the SMILES string (not from the GT graph, so stereo may differ from GT). `--pre_mmff` is the only L refinement | The sampler must start L from ETKDG + noise and denoise it jointly |
| 9 | `diffusion/sampling.py:85-93` (`perturb_seeds`), `160-210` (reverse loop), `266-274` (`apply_torsion_and_update_pos`) | The prior and every reverse step touch only torsions | Joint reverse SDE on (τ, θ) |
| 10 | `diffusion/likelihood.py:24-56, 68-96, 119` | Divergence is only over torsions. `log_det_jac` is computed at fixed L | Extend the Jacobian and divergence to the angle coordinates |
| 11 | `utils/featurization.py:59, 63` | Chirality is never a feature; it is fixed by the seed's embedding | Needed once angles move, because inverting a centre becomes possible |
| 12 | `evaluate_confs.py:112` | Heavy-atom `GetBestRMS` is the only metric. Nothing measures L error | Add bond, angle and ring-pucker error metrics (`tools/local_structure_analysis.py`) |

The model also **conditions on L**: the radius graph, distances and bond vectors are computed from the current `pos` (`score_model.py:143, 166-170, 181, 196`). So any L mismatch between training (ETKDG from the GT graph) and test (ETKDG from SMILES, or GT L) is a covariate shift for the score network. Ablations A1, B1 and G1 in the plan measure this.

---

## 11. Local additions (not upstream)

- **`patches/0001-ablation-hooks.patch`** (branch `flexitors-ablation-hooks`, 3 commits). It adds:
  - `train.py`: seeding, and a `--scheduler none` crash fix.
  - `dataset.py`: `data_dir` trailing slash, the split index guard, and `--loader_workers`.
  - `generate_confs.py`: `--seed` (seeded ETKDG with `randomSeed = seed+1`, because RDKit seed 0 makes every conformer identical); CLI `likelihood` and `seed` are no longer overwritten by the yaml; `--sigma_min_inf/--sigma_max_inf` for the inference-only schedule.
  - `evaluate_confs.py`: `--out_results` and `--report_threshold`, a `SUMMARY` line, and `SWEEP` lines at 0.05/0.1/0.25/0.5/0.75/1.0/1.25 Å. 0.05 and 0.1 are not on the upstream 0.125 grid.
  - `score_model.py`, `utils.py`, `parsing.py`: a `--no_parity` flag that makes the torsion head output 0e instead of 0o. This reproduces the TD Table 8 "no parity equivariance" row.
- **`tools/`**:
  - `make_seed_pickles.py`: GT-L seeds for `--seed_confs`, GT-graph seeds for `--seed_mols`.
  - `local_structure_analysis.py`: ETKDG vs GT bond/angle RMSD; "transplant" RMSD (GT torsions copied onto the RDKit L); DE-optimised torsion-oracle floor; stereo match; training matching residuals `conf['rmsd']`.
  - `breakdown.py`: COV/MAT by heavy-atom count, #torsions and #GT conformers, plus Spearman correlation with L error.
  - `geometry_metrics.py`: bond-length, bond-angle and torsion MAE of generated conformers against their RMSD-matched GT conformer, with a symmetry-aware atom map.
  - `stereo_check.py`.
  - `resample_confs.py`: energy/likelihood importance resampling.
  - `smoke_test.py`: environment check, including a parity test.
  - All tools except `smoke_test.py` were smoke-tested locally on synthetic molecules (RDKit 2026.03). `smoke_test.py` needs e3nn and torch_cluster, which are not installed locally.
