# Code concerns: bugs, version traps, suspicious metric code

Line numbers refer to upstream `master` @ `5f713b42`. "Patched" means fixed or hooked in `patches/0001-ablation-hooks.patch`. Severity is how much it can affect QM9 results or runs: **H** = high, **M** = medium, **L** = low.

## A. Affects results or evaluation

1. **H. The training yaml silently overrides the CLI at inference** (`generate_confs.py:76-78`, `args.__dict__.update(yaml)`).
   - `parse_train_args` has `--likelihood` defaulting to `'full'` (`parsing.py:50`), and it is saved in `model_parameters.yml`. So for **any model trained with this code**, `args.likelihood='full'` at generation time, whatever the CLI says. The `assert args.ode or args.no_model` (54-55) ran *before* the override and never sees it.
   - Consequence (i): molecules with **0 rotatable bonds return `None`** (142-145). `evaluate_confs.py` then counts them as **model failures with COV = 0**, which deflates COV.
   - Consequence (ii): every `--ode` run computes the **full divergence**, one extra forward pass per torsion per step, which is roughly 5-10× slower.
   - The released QM9 yamls have no `likelihood` key (older internal code), so this only bites **our retrained models**. `seed` is also overridden (though unused).
   - *Patched:* the CLI `likelihood` and `seed` are restored after the yaml load.
2. **H. The QM9 split can index past the raw pickle list** (`dataset.py:82-87`, GitHub issue #20: `index 133112 ... size 133111`).
   - Split indices refer to `sorted(glob(raw/*.pickle))`. If even one raw file is missing, training crashes.
   - It also silently **shifts the molecule → split and std-pickle-id mapping** (`i // 1000`, line 91). A handful of molecules near pickle boundaries are then dropped as `smile_not_in_std_pickle`, and train/val membership can shift by one position.
   - *Patched:* out-of-range indices are dropped with a warning. `download_data.sh` prints the raw count vs the max index.
   - Remaining risk: if files are missing, our split is *not identical* to the paper's. The failure counts printed by `ConformerDataset` must be checked.
3. **M. `--data_dir` must end with `/`** (`dataset.py:91, 97`; `standardize_confs.py:125, 153`). Names are derived as `path[len(root):-7]`. Without the slash every lookup fails, giving `num_samples=0` (issue #12). *Patched* in `dataset.py`. `standardize_confs.py` still needs the slash; the SLURM scripts pass it.
4. **M. The featurisation cache is keyed only by path** (`dataset.py:62-67`). An existing `<cache>.train` is reused even if `--std_pickles`, `--data_dir`, `--split_path` or `--dataset` changed. Every data variant needs its own `--cache`; the SLURM scripts use `cache_{std,raw,nomatch,mmff,de50}`. Two jobs building the same cache at the same time will corrupt it, so `featurize_qm9.sbatch` builds it first.
5. **M. Evaluation failure accounting** (`evaluate_confs.py`). It matches the metric reference in `papers/gap_synthesis.md` §6: heavy atoms (`RemoveHs`), symmetry-aware `GetBestRMS`, strict `<`, failures = COV 0 and excluded from AMR. Subtleties:
   - **MAT/AMR excludes failed molecules** (`nanmean` / `nanmedian`, 152-155), while COV counts them as 0 (149-150). AMR looks better when the model fails more. Always report `n_model_failures` (the patched `SUMMARY` line does).
   - If `GetBestRMS` throws for *one* GT conformer, the whole GT row becomes NaN (113-116, "Additional failure"). The consequences:
     - That GT conformer counts as uncovered in COV-R.
     - **Every column's `min(axis=0)` becomes NaN**, so COV-P for that molecule is 0.
     - The molecule's AMR-R and AMR-P become NaN, so it is dropped from AMR.
     - So one bad conformer zeroes the molecule's precision coverage.
   - Molecules whose GT set is empty after `clean_confs` are **silently skipped** (85-87). They appear in neither COV nor failures.
   - **K ≠ 2L in general.** K = 2 × `n_conformers` from the CSV (`generate_confs.py:158`), but L is counted *after* `clean_confs` (83, 89). So K/L can deviate from 2.
   - AMR is printed unchanged in every threshold block (it is threshold-independent). There is **no per-dataset threshold** in code: QM9's 0.5 Å has to be read from the sweep (the `--dataset` flag only matters for `xl`, lines 75-76). The sweep grid `np.arange(0, 2.5, .125)` does **not contain 0.05 or 0.1 Å**. *Patched:* `SUMMARY` at δ = 0.5 for QM9, plus `SWEEP` lines at 0.05-1.25 Å.
   - The default `--test_csv` is the DRUGS `test_smiles_corrected.csv` (12). Always pass it explicitly.
   - `GetBestRMS` allows proper rotations only. A generated **enantiomer or E/Z isomer can never match** (see #9).
6. **M. Nothing is seeded.**
   - `train.py` parses `--seed` but never uses it.
   - Generation never seeds numpy/torch/`random`, and ETKDG uses `randomSeed=-1` (`generate_confs.py:59`, `standardize_confs.py:78`).
   - Results are therefore not reproducible run-to-run; plan for ≥3 sampling seeds.
   - *Patched* (train and generate). **Trap found while testing:** RDKit `EmbedMultipleConfs(..., randomSeed=0)` returns **identical conformers** (all torsions and angles equal; checked with RDKit 2026.03). The patch uses `seed + 1`, and the tools default to seed 1.
7. **M. Conformer matching optimises an H-inclusive RMSD with a small DE budget** (`standardization.py:57-60`, `standardize_confs.py:105-106`, `popsize=15, maxiter=15`).
   - Terminal CH3/OH/NH2 rotors are part of the torsion set (`get_torsion_angles` counts any bond with ≥2 atoms per side). They are optimised along with the heavy-atom torsions, and hydrogens drive the objective.
   - In a local test on an 8-heavy-atom molecule with 6 torsions, DE gave H-RMSD 1.03 Å on 2/5 seeds, while the heavy-atom RMSD was 0.07-0.10 Å.
   - So the stored heavy-atom `rmsd` (the QM9 "floor") is an **upper bound** on the true torsion-only floor. `tools/local_structure_analysis.py` recomputes it with a heavy-atom objective, heavy-relevant torsions only, more DE iterations, and a GT-torsion transplant start. B4 (`VARIANT=de50`) tests how sensitive training is to this.
8. **M. The return value of `optimize_rotatable_bonds` is discarded** (`standardize_confs.py:105`). The code relies on `score_conformation` having mutated `mol_rdkit_single` in place. `apply_changes` uses `copy.copy(mol)` (`standardization.py:24`), which for an RDKit Mol is a **full copy** (verified). So the stored conformer holds whatever torsions were *last evaluated*, not necessarily `result.x`. With SciPy 1.17 these coincided in my test, because DE evaluates the final best point last. That is SciPy-version dependent; the stored `rmsd` is at least consistent with the stored geometry.
9. **M. Stereo and train/test asymmetry.**
   - Chiral tags are computed but **not used as features** (`featurization.py:59, 63`), and torsion updates cannot change chirality.
   - Training seeds are embedded from the **GT mol** (`standardize_confs.py:72-78`, which inherits its stereo tags). Test seeds are embedded from the **CSV SMILES** (`sampling.py:29-43`). If the SMILES lacks stereo, ETKDG picks a stereoisomer at random, and `GetBestRMS` can never recover that.
   - Non-ring **double bonds are "rotatable"** (`torsion.py:9-39` has no bond-order check; TD App. F.3), so the model can flip E/Z.
   - Measured by `tools/stereo_check.py` and arm G1 (`--seed_mols`).
10. **L. The loss is averaged over all torsions in a batch** (`training.py:25`), not per molecule, so molecules with more torsions dominate. H-rotor torsions, which are irrelevant to heavy-atom RMSD, take up capacity and loss weight.
11. **L. Sampling details** (`sampling.py:105-106`). The schedule stops one step above σ_min (`[:-1]`) with no final noise-free denoising step. The Gaussian noise `z` is drawn on CPU with shape `edge_pred.shape`. These are faithful to the paper's Alg. 3, but worth knowing when comparing step counts.
12. **L. `kT` inconsistencies.**
    - `boltzmann.py:24` divides by **4148** instead of 4184 J/cal, a 0.9% error in kT.
    - `likelihood.py:99` hard-codes `kT = 0.592` kcal/mol (298 K) in `free_energy`.
    - `tools/resample_confs.py` uses 4184.
13. **L. `torus.py` table artefacts.**
    - For small σ with |x| near π, `p_` underflows to 0, so `score_ = 0/0 = NaN` (the "invalid value encountered in divide" warning in issue #8). Training never samples there, so the effect is benign.
    - `score_norm_` is a fresh Monte-Carlo estimate on every import (unseeded, 10k samples per σ), so the loss weighting changes slightly between runs.
    - The lookup rounds to the nearest grid point on a log grid.

## B. Crashes and robustness

14. **H (for the environment). e3nn ≤ 0.5.0 crashes in `BatchNorm`** for the `bond_conv` layer, whose output is `ns×0o` with no `0e` (`score_model.py:115-122`). `torch.cat([], out=running_mean)` fails (issue #3; DiffDock #14). **e3nn 0.5.1 added the `len(new_means) > 0` guard** (verified in the source), so the pin is 0.5.1.
15. **M. `train.py:45` saves `scheduler.state_dict()` unconditionally**, so `--scheduler none` crashes at the end of epoch 0. *Patched.*
16. **M. No real resume.** `--restart_dir` loads `best_model.pt` weights only (`train.py:85-91`). The optimizer state, LR schedule and epoch counter are lost, and `last_model.pt` (which does store them) is never read. SLURM time limits must cover a full run.
17. **M. The loader has no worker processes** (`dataset.py:257-259`). All per-sample numpy work (deepcopy, `modify_conformer` with scipy `Rotation`) runs in the main process, so the GPU is likely starved. *Patched:* `--loader_workers` (default 0 keeps the upstream behaviour; the SLURM scripts use 8).
18. **L. `diffusion/torus.py` writes `.p.npy` and `.score.npy` (≈ 200 MB each) into the CWD** on first import (26-34). Concurrent first imports race on the write. `setup_env.sh` precomputes them once, and all jobs `cd` into the repo.
19. **L. `utils/xtb.py` creates `/tmp/<pid>` at import** and writes `/tmp/<pid>.xyz`. `utils.time_limit` uses `SIGALRM`. Both are **Unix-only**, so the code cannot run natively on Windows; the particle-guidance path, for one, needs SIGALRM. This is fine on the cluster.
20. **L. `boltzmann_train`** (DRUGS Boltzmann generator only):
    - `best_val > val_ess` compares Python lists of per-molecule ESS lexicographically.
    - It saves the "best" model when ESS *decreases*, but higher ESS is better.
    - It uses the global `scheduler`.
    - Not used for QM9.
21. **L. `--pg_invariant type=bool`**: any non-empty string, including `"False"`, becomes True.
22. **L. `data.edge_pred = out.squeeze()`** (`score_model.py:154`) becomes 0-d when a batch has exactly one rotatable bond. It broadcasts correctly in the loss, but it is fragile.
23. **L. The `--scale_by_sigma` flag cannot be switched off** (`store_true` with default True, `parsing.py:39`). The parser default for `--use_second_order_repr` is False, while the class default and the released QM9 model use True. You must pass `--use_second_order_repr` to reproduce `qm9_default`.
24. **L. README and repo metadata.**
    - `environment.yml` lists no torch/PyG/e3nn/spyrmsd/rmsd/networkx/pandas.
    - The README's DRUGS evaluate command reads `drugs_steps20.pkl` but generate writes `drugs_20steps.pkl`.
    - A custom `smiles.csv` needs a header row (issue #8).
    - `optimize_confs.py:24` hard-codes the DRUGS CSV.
    - `spyrmsd` is imported at module level in `sampling.py`, so it is required even without particle guidance.
25. **L. The released checkpoints come from an older internal code version.** Their yaml has keys the public parser lacks (`aggregator`, `tp_aggregator`, `num_intermediate_conv_layers`, `pos_enc_dim`, `no_hydrogen`) and internal data paths (`standardized_multi_qm9`, `splits/split0.npy`). Loading works because `get_model` only reads the known keys, but it is **not guaranteed that the public training code reproduces them exactly**. The P0-vs-R0 comparison will tell.

## C. Version compatibility summary

- **Python ≥ 3.9** is required: dict `|` in `standardize_confs.py:160`.
- **torch_geometric**: `torch_geometric.data.DataLoader` is still re-exported in 2.0.4 through 2.5 (checked). PyG < 2.3 also needs `torch_sparse` at import. The pins are 2.0.4 (torch 1.13) or 2.3.1 (torch 2.0).
- **torch ≥ 2.6** changes the `torch.load` default to `weights_only=True`. That is fine for `best_model.pt` (a state_dict), but it is one reason to stay on 1.13 or 2.0.
- **numpy 2.x** is incompatible with torch 1.13 wheels. Pin 1.23.5.
- **GPU architecture**: torch 1.13.1+cu117 wheels target up to sm_86 (Ampere). For Ada (sm_89) or Hopper (sm_90) on gnode118, use `TORCH_PROFILE=cu118`. `cuda_check` prints the device capability and the arch list.
- **RDKit version** changes ETKDG defaults and output, and so changes the local structures, i.e. the floor itself. Pin one version (2022.9.5) for **both** standardisation and test-time embedding. The shipped standardized pickles were produced with an unknown RDKit version. This is a (small) train/test L mismatch that B-arms regenerated locally avoid.
