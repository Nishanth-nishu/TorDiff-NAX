# Plan P1 (round 5, Stage A): smallest clean `flexitors/` design

Planner: P1 (coding planner 1 of 3), 2026-10-16. Emphasis: smallest clean design — one gateway to the frozen
baseline, no copied TD code except where a TD script cannot be imported, no module that Stage A does not run.
Scope: `round5/DECISION.md` (Stage A). Sources: `round5/specs/EXP-1.md`, `EXP-3.md`, `EXP-5.md`, `EVAL.md` (revised
blocks are current), `round5/ledger/verify_V5.md`, `round5/ledger/disputes.md` (D-101, D-102 open).
Consult questions: `round5/consult/Q-P1-001.md` … `Q-P1-011.md`. Every default that depends on one is marked
**PENDING Q-P1-NNN**. Not read: `plan_P2.md`, `plan_P3.md`.

## 0. Summary

- **One gateway, no forks.** `flexitors/td.py` is the only file that touches `torsional-diffusion/` (and `tools/`).
  It puts both on `sys.path`, imports the baseline symbols that Stage A reuses, and re-exports them. A SHA-pin test
  (`test_baseline_frozen.py`) fails if any baseline file changes. TD code is copied only where the TD source is a
  top-level script that cannot be imported (`generate_confs.py` per-molecule driver, `training.py:25-31` loss
  expression, `utils/utils.py:9-20` model kwargs), and every such port has a golden test against the original.
- **One reverse loop.** `flexitors/sampling/reverse.py` is TD's SDE loop (`sampling.py:176-246`, SDE branch only)
  with one hook object. With no hook it is bit-identical to `diffusion.sampling.sample` (golden test). Exp 3
  (fold / guard) and Exp 1 (geometry step) are hooks on that loop; Particle Guidance and plain TD call the
  baseline `sample()` directly.
- **One generator CLI** (`flexitors.sampling.generate`) covers every Stage A GPU arm, writes the `{smi: [Mol]}`
  pickle of `generate_confs.py:210-213` plus `budget.json` (EVAL §8), so the unchanged `evaluate_confs.py` scores
  every arm (EVAL §0 rule 1). One CPU post-processing CLI (`flexitors.sampling.posthoc`) covers F-flip, Exp 3
  Step 0 snap and Exp 5 selection.
- **Stage A only.** Modules for V3, CP rings, reflected-kernel training (K2-train), LoQI edges (K3), xTB energy,
  validity, DAgger re-matching and the Exp 5 allocation controller are *not* created now; they are specified where
  needed and created when their stage is approved. Nothing is written that Stage A does not execute.
- **D-101 resolved** by a pool whose first N0 = 128 conformers are a K-free block identical in size for every
  molecule, topped up only where an arm's demand c·2K exceeds it; no arm reads the pool size and a test asserts no
  arm is ever capped (§2.4). **D-102 resolved** by an NFE-matched PG arm P2b (PG on 4K proposals, random 2K kept)
  and the revised primary family {P3 − P1, P3 − P2b} (§2.4). Both PENDING Q-P1-008 / Q-P1-009.
- **Budget:** core ≈ 19.5 GPU-h (smokes 0.45; Exp 1 M1 + Gate-0 eval + A6 preview ≈ 11.35; Exp 5 ≈ 6.8; Exp 3
  QM9 ≈ 0.9) plus Exp 3 GEOM-DRUGS ≈ 5 GPU-h after the D0 download is approved: ≈ 24.5 GPU-h, inside the 15–25
  window at central estimates; §5.3 lists the pre-declared cuts if a smoke measurement says otherwise.

## 1. Package layout and the baseline gateway

### 1.1 Tree (Stage A; every file listed is executed by a Stage A job or test)

```
flexitors/
  __init__.py                 docstring only; no imports (importing flexitors must not import torch)
  td.py                       the gateway to torsional-diffusion/ and tools/ (§1.2)
  geometry/
    dihedrals.py              TD rotatable quadruples, analytic dihedral gradients (W rows)
    projector.py              P_W = I − Cᵀ(CCᵀ)⁺C, C = [W; J_rig], batched, never forms a 3n×3n matrix
    constraints.py            EXP-1 §2.6 bond window + clash test, §2.8 chirality guard, backtracking step
    torsion_types.py          EXP-3 §3.1 TD-rotatable bond classes, E/Z targets φ_b
  models/
    flexitors_score.py        TensorProductScoreModel subclass: forward hook + per-atom 1o velocity head
  sampling/
    reverse.py                TD reverse SDE loop with one hook object (golden = sampling.sample)
    geometry_flow.py          Exp 1 hook: projected Euler geometry step + constraint layer (M1, A6/A6b sampler)
    stereo.py                 Exp 3: constrained-bond set, K1 freeze, K2-inf fold hook, per-step guard, F-flip, snap
    select.py                 Exp 5: torsion FPS, MMFF94 filter, random; select_2k
    budget.py                 NFE / CPU / wall accounting, budget.json, assert_matched
    generate.py               CLI: every Stage A generation arm (TD, PG, pool, K1*, K2-inf, F-filter, geom, A6)
    posthoc.py                CLI: flip (F-flip), snap (Exp 3 Step 0), select (Exp 5 Track P arms)
  training/
    transforms.py             PairedGeometryTransform (EXP-1 §2.2 training steps 1–5)
    losses.py                 td_score_loss (port of training.py:25-31), projected_l_loss, joint_loss
    train.py                  CLI: thin epoch loop, TD checkpoint/resume functions imported
  eval/
    paired.py                 per-molecule AMR/COV (R and P) from eval.pkl, TD-convention universe, contrasts
    stereo.py                 DB-only / CH-only labels (port of chem_data/ez_flip_analysis.py), E/Z fidelity, planarity
    rings.py                  ring-dihedral error of geometry-flowed L_etkdg2L seeds vs their paired GT (Gate 0)
  data/
    sets.py                   CLI: rigid stratum, QM9 validation subset, stereo subsets, E/Z-enriched DRUGS set, splice
  tests/                      §4 (unit, golden, smoke); conftest marks `cluster` tests that need torch/PyG/data
experiments/
  r5_cpu/        config.yaml  run.sbatch  README.md                (floor fix re-run, budget-matched re-score)
  exp1_m1/       config.yaml  run.sbatch  README.md  analyze.py    (M0, smoke, M1 = A11, Gate 0, A6 preview)
  exp3_stereo/   config.yaml  run.sbatch  README.md  analyze.py    (Step 0, QM9 + DRUGS inference arms)
  exp5_select/   config.yaml  run.sbatch  README.md  analyze.py    (pools, PG arms, selection, ORACLE headroom)
results/<exp>/   outputs on gnode118 (/scratch/nishanth.r/tordiff/results/r5/<exp>/), never committed
```

`analyze.py` is the fourth file per experiment because EVAL §0 rule 2 requires contrasts, families and decision rules
to be in code before any result exists (the `tools/analyze_round2.py:1-23` pattern). `run.sbatch` reads one row of
`config.yaml` per array index (rows = arm × seed × set), so there is one config and one sbatch per experiment.

Changes outside `flexitors/` and `experiments/` (both are `tools/`, not the frozen baseline):
`tools/local_structure_analysis.py` (floor fix, §2.1) and `tools/analyze_leakfree.py` (`--ref_subsample`, §2.1),
each with a regression test in `tools/tests/`.

### 1.2 The gateway `flexitors/td.py` (how the baseline is imported without modification)

```python
TD_ROOT: Path     # <repo>/torsional-diffusion   (env FLEXITORS_TD_ROOT overrides, for /scratch on gnode118)
TOOLS_ROOT: Path  # <repo>/tools
```
1. Inserts `TD_ROOT` and `TOOLS_ROOT` at the front of `sys.path` once. TD's top-level package names are `diffusion`
   and `utils`, so **no flexitors module or subpackage may be named `diffusion`, `utils` or `train`** (style rule
   S6, §4.4).
2. `diffusion/torus.py:26-34` loads `.p.npy`/`.score.npy` from the *current directory* and otherwise recomputes a
   5001 × 5001 table and writes it there; `:66-70` also draws from the global numpy RNG at import. `td.py` therefore
   (a) asserts `TD_ROOT/.p.npy` and `.score.npy` exist, (b) imports `diffusion.torus` inside
   `os.chdir(TD_ROOT)` / `finally: os.chdir(old)` (Python 3.9: no `contextlib.chdir`), and (c) is imported before
   any seeding. Every entry point imports `flexitors.td` first and seeds afterwards, exactly as
   `generate_confs.py:94-95` seeds after its imports.
3. Re-exports (the complete list of baseline symbols Stage A uses):

| Symbol | Baseline location |
|---|---|
| `get_seed`, `embed_seeds`, `perturb_seeds`, `pyg_to_mol`, `InferenceDataset`, `sample as td_sample` | `diffusion/sampling.py:40-60, 63-102, 105-113, 249-274, 277-298, 116-246` |
| `torus` (`score`, `score_norm`) | `diffusion/torus.py:37-46, 73-77` |
| `TensorProductScoreModel`, `get_timestep_embedding` | `diffusion/score_model.py:48-213, 230-240` |
| `modify_conformer`, `perturb_batch`, `get_torsion_angles`, `get_transformation_mask` | `utils/torsion.py:57-75, 78-103, 110-124, 9-39` |
| `ConformerDataset`, `interp_x_torch` | `utils/dataset.py:109-339, 18-34` |
| `featurize_mol_from_smiles`, `qm9_types`, `drugs_types` | `utils/featurization.py:104-131, 20-24` |
| `get_optimizer_and_scheduler`, `save_yaml_file` | `utils/utils.py:23-36, 39-45` |
| `SetDihedral`, `get_torsion_angles as rdkit_rotatable_quads` | `utils/standardization.py:19-20, 69-89` |
| `save_checkpoint`, `load_resume_state` (module `train`, guarded by `__main__` at `:144`) | `train.py:30-34, 37-62` |
| `boot`, `holm`, `hier_ci` (module `analyze_leakfree`, importable: used by `tools/tests/test_leakfree.py`) | `tools/analyze_leakfree.py:130, 153, 139` |
| `l_subset_errors`, `align_pair` (module `lgeom`) | `tools/lgeom.py:317-338, 133` |
| `rot_bond_counts` (module `breakdown`) | `tools/breakdown.py:32` |

`generate_confs.py`, `evaluate_confs.py` and `standardize_confs.py` parse `argv` at import, so they are never imported:
`evaluate_confs.py` is run unchanged as a subprocess by every sbatch (the `gen_eval` pattern of
`slurm/common.sh:102`), and `generate_confs.py` is the golden reference for `flexitors.sampling.generate`.

### 1.3 Public functions (signatures and one-line docstrings)

Types: `T` = `torch.Tensor`; B = conformers in a batch, n = atoms, m = rotatable bonds, c = guarded centres.

**geometry/dihedrals.py**
- `rotatable_quads(edge_index: T, edge_mask: T) -> T[m,4]` — one dihedral (c, a, b, d) per TD-rotatable edge, c/d the
  first neighbour of a/b in bond order; this is both TD rules (`sampling.py:129-142` and `standardization.py:69-89`,
  which coincide; test).
- `dihedral_gradients(pos: T[B,n,3], quads: T[m,4]) -> T[B,m,n,3]` — analytic ∇ₓτ_i (Wilson-B rows) for the sign
  convention of `torsion.py:110-124` (checked by finite differences).

**geometry/projector.py**
- `constraint_rows(pos: T[B,n,3], quads: T[B,q,4], quad_mask: T[B,q], atom_mask: T[B,n]) -> T[B,q+6,3n]` —
  C = [W; J_rig], J_rig = 3 translations + 3 rotations (x_k − x̄) × ê (TD Eq. 10, E-JOINT-044).
- `project(v: T[B,n,3], pos, quads, quad_mask, atom_mask) -> T[B,n,3]` — P_W v = v − Cᵀ(CCᵀ)⁺Cv; padded
  molecules (training) and same-molecule batches (sampling) through one code path.
- `constraint_rank(pos, quads, quad_mask, atom_mask) -> T[B]` — rank of C; the sampler counts rank < q+6 (diagnostic,
  EXP-1 §3 row 9).

**geometry/constraints.py**
- `guard_centres(mol: Chem.Mol) -> T[c,4]` — (centre, a, b, d) for every RDKit potential tetrahedral stereocentre and
  every ring-fusion atom with ≥ 3 heavy neighbours; a<b<d = lowest three canonical ranks (EXP-1 §2.8).
- `signed_volumes(pos: T[B,n,3], centres: T[c,4]) -> T[B,c]` — χ_c = (x_a − x_c)·[(x_b − x_c) × (x_d − x_c)].
- `step_limits(mol, seed_pos: T[n,3]) -> StepLimits` — bonded pairs with [0.85, 1.15] × seed length; non-bonded heavy
  pairs (not 1-2, not 1-3) with min distance 0.75 × (r_cov,i + r_cov,j) (EXP-1 §2.6 step 1).
- `guarded_step(pos: T[B,n,3], dx: T[B,n,3], limits, centres, ref_sign: T[B,c], max_halvings: int = 3)
  -> tuple[T, GuardCounts]` — per conformer: accept dx, else halve up to 3 times, else skip; counts bond/clash fails,
  chirality flips caught, backtracks, skips.

**geometry/torsion_types.py**
- `td_rotatable_bonds(mol) -> list[int]` — bridge bonds with smaller side > 1 atom (port of
  `chem_data/stereo_census.py:26-48`; equals `get_transformation_mask` edges // 2, test).
- `classify_bond(mol, bond_idx: int, potential_db: set[int]) -> str` — one of the 7 EXP-3 §3.1 classes.
- `is_structural_stereo_db(bond) -> bool` — order 2, acyclic, both atoms ≥ 2 heavy neighbours (EXP-3 §3.1).
- `double_bond_targets(mol, quads: dict[int, tuple]) -> dict[int, tuple[float | None, str]]` — φ_b ∈ {0, π} and source
  ∈ {smiles, seed, nonstereo}; nonstereo/unspecified bonds get `None` + planarity-only (target set {0, π}).

**models/flexitors_score.py**
- `class FlexiTorsScoreModel(TensorProductScoreModel)`: `__init__(**td_kwargs)` requires `lambda_embed_dim > 0` (the
  geometry clock t enters through TD's λ path, `score_model.py:194-199`; EXP-1 §5.1); registers a forward hook on
  `self.conv_layers[-1]` that stores the last atom features; `self.l_head = o3.Linear(last_irreps, '1x1o')`.
  `forward(data)` = `super().forward(data)` then `data.l_pred = self.l_head(atom_feats)` (n×3). No TD line copied.
- `build_model(args: Namespace) -> FlexiTorsScoreModel` — the kwargs mapping of `utils/utils.py:9-20` (ported; test
  that its TD keys equal `get_model(args).state_dict()` keys).

**sampling/reverse.py**
- `class StepHook(Protocol)`: `torsion(data, perturb: np.ndarray, k: int) -> np.ndarray`;
  `position(data, data_gpu, k: int) -> None`; `clock(k: int) -> float | None`.
- `reverse_diffusion(conformers, model, *, steps=20, sigma_max=π, sigma_min=0.01π, batch_size=32,
  torsion_step=True, sigma_input=None, hook=None) -> list[Data]` — `sampling.py:122-126, 176-246` (SDE branch);
  `sigma_input` fixes the σ fed to the network (τ-frozen mode).

**sampling/geometry_flow.py**
- `class GeometryFlowHook(StepHook)`: `__init__(mol, steps, guard=True)`; `clock(k) = k/steps`; `position()` does
  x ← guarded_step(x, (1/steps)·P_W(x) l_pred) and writes the conformer positions back; keeps `GuardCounts` and
  per-step timers (projector, constraints).

**sampling/stereo.py**
- `constrained_bonds(mol, data, arm: str) -> ConstrainedSet` — masked-edge indices, quads, φ_b, sources (QM9/DRUGS).
- `freeze(data, cset, wrong: bool = False) -> None` — K1 / K1-wrong / K1-canonical: seed check (|δ_b| > π/2 → rotate
  by π; `wrong` rotates to the opposite isomer), then drop the bonds from `edge_mask` and `mask_rotate`.
- `fold_half_circle(delta: np.ndarray) -> np.ndarray` — wrap to (−π, π], then x > π/2 ↦ π − x, x < −π/2 ↦ −π − x
  (EXP-3 §3.2 K2).
- `class StereoHook(StepHook)`: `__init__(cset, fold: bool)`; `torsion()` folds constrained δ_b (K2-inf) or passes;
  `position()` counts E/Z violations (|δ_b| > π/2) and planarity violations (|δ_b| > 20°, last 3 steps).
- `flip_to_target(mol, cset, only_offending=True) -> Chem.Mol` — F-flip / Step 0 snap via `SetDihedral`.

**sampling/select.py**
- `heavy_torsions(mol) -> list[tuple]` (port of `sample_data/signals.py:14-28`), `mmff94_energy(mol) -> float`,
  `fps(X: np.ndarray, k: int, start: int) -> np.ndarray` (port of `sample_data/select.py:11-18`),
  `select_2k(mols, k2: int, method: str, q: float, rng) -> np.ndarray`.

**sampling/budget.py**
- `class Budget`: `add(smi, *, n_out, n_prop, steps, n_rot, wall_s, cpu_s, extra_nfe=0, mmff_cpu_s=0, diag=None)`;
  `write(path)`; `nfe(smi) = n_prop·steps·[n_rot > 0] + extra_nfe` (`generate_confs.py:153` skips the model at
  n_rot = 0); `assert_matched(budgets: dict[str, Budget], tol=0.01) -> None` (EXP-5 §4.1).

**training/transforms.py**
- `class PairedGeometryTransform(BaseTransform)`: `__init__(sigma_min, sigma_max, sigma_b, t_eps)`; attribute
  `needs_gt = True` (activates the stale-cache guard `dataset.py:141-144`); `__call__(data)` → data with `pos = x_t`,
  `node_sigma`, `edge_rotate`, `node_lambda = t`, `l_target = u`, `quad_index` (4×m, PyG auto-increments `*_index`),
  `l_dof = 3n − 6 − m`.

**training/losses.py**
- `td_score_loss(data) -> T` (port of `training.py:25-31`), `projected_l_loss(data) -> T`,
  `joint_loss(data, w_tau: float, w_geo: float) -> tuple[T, dict]`.

**eval/paired.py**
- `per_molecule(eval_pkl: str, thrs=(0.5, 0.1)) -> pd.DataFrame` — AMR-R/P, COV-R/P per molecule (evaluate_confs.py
  `:38-44` definitions).
- `universe(frames, csv, convention: str) -> pd.Index` — TD convention (failure → COV 0, AMR excluded,
  `evaluate_confs.py:81-84, 152, 166-167`) or success intersection (EVAL §3).
- `contrast(arm: list[pd.DataFrame], ref: list[pd.DataFrame], metric, univ, rng, n_boot=5000) -> dict` — seed-mean
  per molecule, imported `boot` and `hier_ci`; `holm_family(rows) -> rows` with imported `holm`.

**eval/stereo.py** — `labels(mol) -> tuple[str | None, str | None]` (port of `ez_flip_analysis.py:37-52`),
`ez_fidelity(confs, targets) -> pd.DataFrame`, `planarity(confs, quads) -> pd.DataFrame`.

**eval/rings.py** — `ring_errors(flowed: dict, l_error_csv: str, true_mols: dict) -> pd.DataFrame` — per seed
`lgeom.l_subset_errors` against the seed's paired GT conformer (PENDING Q-P1-003).

**data/sets.py** (CLI subcommands) — `rigid`, `qm9_val`, `stereo_subset`, `ez_enriched`, `splice` (§2).

## 2. Stage A implementation per experiment

Order of work: §2.1 (CPU, no GPU) and all unit/golden tests first; then the 20-molecule smoke of every arm; then GPU
runs in the order of §5.

### 2.1 CPU tasks (first)

**C1. Floor bug, `tools/local_structure_analysis.py:302`** (EXP-1 §9 item 1, E-JOINT-031, D-003).
- Bug: `for jj in np.argsort(cost_gg[i])[:args.floor_topm]` (`--floor_topm` default 3, `:74`) keeps the target i,
  whose self cost is `np.inf` (`:281-282`), whenever n_GT ≤ 3, so the "other GT conformer" floor is ≈ 1e-7 Å on the
  544 rows with 2 ≤ n_GT ≤ 3 (V5 closure note on D-003).
- Fix (≈ 6 lines, no behaviour change for n_GT > 3): a module-level helper next to `rms` (`:112`)
  ```python
  def other_topm(costs: np.ndarray, i: int, m: int, other_half: bool = False) -> list[int]:
      """The m cheapest candidates j != i; other_half keeps j % 2 != i % 2 (alternating split, make_split_half.py:10-14)."""
  ```
  `:302` becomes `for jj in other_topm(cost_gg[i], i, args.floor_topm):`. The same loop also fills a new column
  `floor_gtother_half` with `other_half=True`: the "other-half-only variant matching the split-half design" that
  EXP-1 §9 asks for. Parity matches `make_split_half.py` because both index the GT conformers in `clean_confs` order
  (`local_structure_analysis.py:238`, `make_split_half.py` `clean_confs`) and `--max_gt_confs` (default 30) keeps a
  prefix, which preserves parity (asserted in the test).
- Regression test `tools/tests/test_lsa_floor.py`: (1) for n = 2…6 with `costs[i] = inf`, `other_topm` never returns
  i and returns min(m, n − 1) indices; the old expression does return i for n ≤ 3 (the bug, asserted so it cannot
  come back); (2) `other_half` returns only opposite-parity indices; (3) end-to-end `--mode test` on a 3-molecule
  fixture (n_GT = 2, 3, 5; `--popsize 2 --maxiter 2`, reusing the fixture builder of `tools/tests/test_leakfree.py`)
  gives `floor_gtother` > 1e-3 Å on the n_GT = 2 and 3 rows.
- Re-run on gnode118 (CPU only; `experiments/r5_cpu/run.sbatch` row `lsa_test`; ≈ 1.2 s per GT conformer per core,
  `slurm/analysis_qm9.sbatch` header, so ≈ 1–2 h on 16 CPUs with the extra other-half DE runs). Acceptance: on every
  n_GT > 3 row all old columns equal `cluster_sync/results/analysis/local_structure_test.csv` (tolerance 1e-9); only
  the n_GT ≤ 3 `floor_gtother` values change. The `--mmff_seeds` file is re-run only if R-JOINT needs it
  (PENDING Q-P1-011). The recomputed E-JOINT-010 table goes to R-JOINT (they own the ledger).

**C2. Budget-matched leak-free re-score** (EXP-1 §3.6, D-001; "to be run before any Exp-1 GPU time").
- `tools/analyze_leakfree.py` gains `--ref_subsample N` (default 0: every output byte-identical to today). For each
  run of CR (s0–2), B1 (s0–2), S3 (s0–1), S4 (s0–2) it reads `steps20_seed0/eval.pkl` (`REF_TAGS['ref_rdkitL']`,
  `:74`). Per molecule of the leak-free universe and draw r = 0…N−1 (rng = `default_rng([seed, r, mol_idx])`):
  columns S_B ⊂ all generated conformers, |S_B| = 2|B|, without replacement; rmin over S_B on the rows `idx_B` of
  `split.tsv`; the same for A with 2|A|; direction mean as `view()` (`:116-128`); mean over draws → `ref_half`.
  No `confs.pkl` is needed: `eval.pkl` already holds the RMSD of every generated conformer to every cleaned target.
- New pre-declared contrasts, written into the file header before the run: `BM.<m>.split-ref_half` for m ∈ {CR, B1,
  S3, S4} on AMR-R and COV-R@0.5 (COV-R@0.1 key secondary), molecule bootstrap of seed-mean differences (`boot`,
  `:130`) plus the hierarchical interval (`hier_ci`, `:139`); the distinct-target view `split_d − ref_half_d`
  (`nn_masks`, `:169`) as secondary. Family: one Holm family of 4 × 2 tests, labelled `budget`
  (**PENDING Q-P1-011**). N = 10 draws as specified.
- Tests (extend `tools/tests/test_leakfree.py`): (1) without the flag, `contrasts.csv` and `per_model.csv` of the
  existing fixture are byte-identical to the pre-change output; (2) with |S| = n_model, `ref_half` equals the full
  reference restricted to the target-half rows (deterministic); (3) over 2000 draws the mean matches the exact
  hypergeometric expectation (the `sample_data/curves.py` formula, test-only) within 3 SE; (4) `n_true == n_clean`
  for every molecule (row order = cleaned order).
- Run: gnode118 CPU, minutes. Output `$RES/analysis_r4_leakfree_bm/`. Owner of the reading: R-JOINT / orchestrator.

**C3. Data builders** (`flexitors/data/sets.py`, CPU): `rigid` (n_rot_heavy = 0 by `tools/breakdown.py:32`; 289 test
molecules, E-JOINT-007 — asserted), `qm9_val` (1000-molecule validation subset, EVAL §1; **PENDING Q-P1-010**),
`stereo_subset` (molecules with ≥ 1 constrained bond, §2.3), `ez_enriched` (after D0, §2.3). Each writes a
`--filter_mols` pickle and, where needed, a csv + `true_mols` pickle in the formats `evaluate_confs.py:32-34` reads.

**C4. Exp 3 Step 0 and Exp 1 M0** (CPU, ORACLE) — §2.3 and §2.2.

### 2.2 Exp 1: M0, smoke, M1 (Gate 0), A6/A6b sampler

Stage A runs M0 → 20-molecule smoke → 1-epoch smoke → (re-stated ask) → M1 = arm A11 → Gate 0 → A6 preview if GO.
A6b is specified and its sampler is built (it is the M1 evaluation sampler); its re-match + retrain belongs to M2.

**Projector (revised, D-004).** `geometry/projector.py` implements P_W = I − Cᵀ(CCᵀ)⁺C with C = [W; J_rig]
(EXP-1 §2.2 "Revised version"; E-JOINT-032):
- W: one row per TD-rotatable bond, the analytic gradient of the dihedral (c, a, b, d) from `rotatable_quads`
  (TD's rule, `standardization.py:69-89`, which coincides with `sampling.py:129-142`). Analytic form: standard
  Wilson-B dihedral derivative, written for the sign convention of `torsion.py:110-124` and **verified by central
  finite differences in `test_dihedrals.py`** (no external formula source is cited; the spec specifies the object,
  not the formula).
- J_rig: 3 unit translations and 3 rotations (x_k − x̄) × ê over the real atoms (TD Eq. 10, PDF p. 7, E-JOINT-044).
- `project` never forms the 3n×3n matrix: y = (CCᵀ)⁺(Cv), P_W v = v − Cᵀy, with `torch.linalg.pinv(CCᵀ,
  hermitian=True)` so rank-deficient cases (linear fragments, m = 0, padding rows) need no branch. Batched over
  conformers (sampling: same molecule, quads broadcast) and over padded molecules (training: `to_dense_batch` of
  PyG). No per-molecule Python loop (D-006). The spec's "dense for n ≤ 64, else lstsq" becomes one implicit path,
  which is smaller and exact for every n.
- Rank: `constraint_rank` is logged per molecule; rank < m + 6 is counted (EXP-1 §3 row 9).

**Chirality guard (EXP-1 §2.8, mandatory after D-004).** `constraints.guard_centres(mol)`: every RDKit potential
tetrahedral stereocentre (`Chem.FindPotentialStereo`, version pinned to the cluster's 2022.09.5) plus every
ring-fusion atom with ≥ 3 heavy neighbours (the spec's bridgehead rule), with neighbours a<b<d = the three lowest
`CanonicalRankAtoms`. The reference sign is the seed's χ_c. `guarded_step` rejects any step that flips a sign,
breaks the ±15 % bond window or the 0.75 × Σr_cov clash rule (EXP-1 §2.6 step 1), halves it up to 3 times, then
skips it; flips caught, backtracks and skips are counted per conformer and reported per arm (EXP-1 §3 row 9). The
ET-Flow post hoc flip (E-JOINT-132) is not built (it is a separate arm, not Stage A).

**Model (EXP-1 §2.4, §5.1).** `FlexiTorsScoreModel` = TD's network unchanged plus:
- the geometry clock t through TD's λ input (`score_model.py:194-199`, the round-2 S4 hook; `lambda_embed_dim = 32`
  as S4, `slurm/ablations_train_round2.tsv:15`);
- `l_head = o3.Linear(conv_layers[-1].out_irreps, '1x1o')`, which reads only the `nv × 1o` channels of the last atom
  layer (`score_model.py:83-89, 141-143`; E-JOINT-021) — a parity-odd vector, as TD Prop. 2 requires for a Cartesian
  velocity (E-JOINT-042; `test_parity.py`);
- the last atom features are captured by a forward hook on `conv_layers[-1]`, so `forward` is TD's, not a copy.
- Known risk: TD's `forward` with `edge_mask.sum() == 0` (true rigid molecules, never seen in TD training because
  `dataset.py:259-262` and `standardize_confs.py:74` drop them) is untested; `test_model.py` runs it first. If
  `torch_cluster.radius` fails on an empty bond set, the subclass overrides `build_bond_conv_graph` only for that
  case (documented deviation, golden test unaffected).

**Training (EXP-1 §2.2 "Training", steps 1–6).** `PairedGeometryTransform.__call__`:
1. k as `dataset.py:86-88` (uniform, or Boltzmann weights if `boltzmann_weight`); x_R = `data.pos[k]`,
   x_G = `data.gt_pos[k]` (paired cache; E-JOINT-023; `del data.gt_pos` as `dataset.py:100`).
2. σ ~ logU[0.01π, π] (`dataset.py:71`); t ~ U[t_ε, 1 − t_ε] independently (E-JOINT-072); **t_ε = 1e-3, PENDING
   Q-P1-001** (γ′(t) = σ_b(1 − 2t)/(2√(t(1 − t))) diverges at the endpoints, and the spec does not say how ET-Flow
   handles it).
3. Δτ ~ N(0, σ²)^m (`dataset.py:74`); the same Δτ on both endpoints with `modify_conformer` (`torsion.py:57-75`).
4. z ~ N(0, I); x_t = `interp_x_torch`(x_R′, x_G′, t) + γ(t)·P_W(x_R′) z, γ = σ_b√(t(1 − t)) (ET-Flow Eq. 2–5,
   PDF p. 4, E-JOINT-076; `dataset.py:18-34`).
5. u = P_W(x_t)[x_G′ − x_R′ + γ′(t) P_W(x_R′) z]; `edge_rotate = Δτ` (TD target unchanged).
6. `joint_loss = w_τ·td_score_loss + w_geo·Σ_mol ‖P_W(x_t) l_pred − u‖² / (3n − 6 − m)`, mean over molecules.
   `td_score_loss` is the expression of `training.py:25-31` (golden test against `train_epoch`).
`needs_gt = True` activates TD's stale-cache guard (`dataset.py:141-144`). Whether the S4 `pair_ok` filter
(`dataset.py:145-162`) applies, whether the round-2 paired cache must be rebuilt (EXP-1 §7 risk 2), and whether
true rigid (m = 0) training molecules must be added: **PENDING Q-P1-002** (default: reuse the cache, apply the
`pair_ok` filter by setting the attribute the guard reads, keep TD's m ≥ 1 training set and report the m = 0 test
sub-stratum separately).

`training/train.py` is a ≈ 40-line loop: TD's `ConformerDataset` (`dataset.py:109`) with our transform (TD's
`construct_loader`, `:342-372`, hard-codes its own transform, so loaders are built directly), TD's
`get_optimizer_and_scheduler` (`utils.py:23-36`), TD's `save_checkpoint` / `load_resume_state` (`train.py:30-62`;
SIGUSR1 requeue as `train.py:146-149`), best checkpoint by validation loss (EXP-1 §4 control 8),
`save_yaml_file` for `model_parameters.yml`. Hyper-parameters are not retyped: `experiments/exp1_m1/config.yaml`
names the CTRL_rematch `model_parameters.yml` as the base and overrides only `n_epochs: 30`, `lambda_embed_dim: 32`,
`w_tau: 0`, `w_geo: 1`, `sigma_b`, `t_eps`, data paths and seed. "Train only the V2 geometry head" is read as
"train the whole network on the geometry loss only, torsion noise still applied" (that is what the
0.3 × 11.7 GPU-h cost assumes): **PENDING Q-P1-001** (alternative: frozen pretrained trunk, head only).

**Geometry-flow sampler (M1 evaluation = the A6/A6b independent L sampler).** `generate --arm geom`:
`get_seed` → `embed_seeds` (seeded exactly as `generate_confs.py:64-69`) → **no** `perturb_seeds` (τ frozen at the
ETKDG seed) → `reverse_diffusion(torsion_step=False, sigma_input=σ_min, hook=GeometryFlowHook)` → `pyg_to_mol`.
Per step k = 0…N−1 (N = 20, the TD step count, so NFE equals TD's): one network call, x ← guarded_step(x,
(1/N)·P_W(x)·l_pred) with clock t_k = k/N (EXP-1 §2.2 "Sampling"; §2.5 row "L, V2 bridge": uniform Δt = 1/N).
**σ fed to the network in τ-frozen mode: σ_min, PENDING Q-P1-001.** Unlike TD (`generate_confs.py:153`), the model
also runs for m = 0 molecules (they are most of what Gate 0 measures). Diagnostics per conformer in `budget.json`:
NFE, projector and constraint time share, backtracks, skips, flips caught, rank deficits, and the induced drift of
every TD quadruple per step (EXP-1 §3 row 9).

**M0 (CPU, ORACLE, ≈ 1 CPU-h)** in `experiments/exp1_m1/analyze.py m0`, on the round-2 test seeds
`round2_seeds/L_etkdg2L.pkl` with their Hungarian pairs (k, src_gt_idx) from `l_error.csv`
(`make_l_seed_pickles.py:108-114`): Y_al = `lgeom.align_pair`(X, Y); oracle step X′ = X + P_W(X)(Y_al − X).
(a) per-TD-quadruple dihedral change for steps scaled to 0.01 / 0.02 / 0.05 Å RMS and the full step, beside the old
P_L = I − Bᵀ(BBᵀ)⁺B, B = [J_τ; J_rig], built in `analyze.py` only from TD's `dx_dtau` (`likelihood.py:68-74`) as the
negative control (V5: 0.19–0.25 rad/Å); (b) fraction of ‖Y_al − X‖² inside range P_W; (c) chirality flips (with and
without the guard); kill rule: the oracle step recovers < 50 % of the rigid-stratum AMR-R gap (ETKDG 0.152 → 0).
"Recovery" = (0.152 − AMR-R of the seed set with every paired seed replaced by its stepped version) / 0.152, scored
by `evaluate_confs.py --filter_mols rigid.pkl` (**PENDING Q-P1-003**). On m = 0 molecules P_W removes only rigid
motion, so recovery there is ≈ 100 % by construction; the informative M0 numbers are (a), (c) and (b) on the ≥ 4-ring
seeds. M0 passes before any GPU job.

**Smoke runs (before the Gate-0 ask, D-006).**
- `exp1_m1` row `smoke20`: `--limit_train_mols 20`, 2 epochs, σ_b = 0.15, then `generate --arm geom` on 20 rigid test
  molecules and `evaluate_confs.py`. Checks: finite decreasing loss, `l_pred` shape n×3, guard counters present,
  `budget.json` written. ≈ 0.05 GPU-h.
- `exp1_m1` row `smoke1ep`: 1 epoch on the full QM9 training set, σ_b = 0.15, `--log_timing`-style epoch timer, data
  wait, projector share of step time, peak GPU memory. Compared with TD's 377.7 s/epoch (E-JOINT-008). ≈ 0.12–0.18
  GPU-h. **The M1 ask is restated from this number: M1 = 3 × 30 × (measured epoch + validation). If it exceeds the
  16 GPU-h ceiling (EXP-1 §5.2), M1 does not start and the orchestrator asks the user.**

**M1 = A11 (EXP-1 §5.2, §4 row A11).** 3 runs (σ_b ∈ {0.05, 0.15, 0.3} Å), 30 epochs, seed 0, full QM9 training
set, w_τ = 0, run in parallel on 3 GPUs with `--resume` + requeue. 10.5 GPU-h at TD speed, ceiling 16.

**Gate 0 (`analyze.py gate0`, rules written before the runs).**
- Sets: QM9 validation rigid stratum (selection only, EXP-1 §4 control 8); QM9 test rigid stratum (289 molecules,
  E-JOINT-007); L_etkdg2L seeds of molecules whose largest ring has ≥ 4 atoms (4,181 seeds, E-JOINT-004 rev.; passed
  with `--seed_confs L_etkdg2L.pkl --seed_confs_cycle` and n = 2·n_csv so seed i is used exactly once,
  `sampling.py:83-86`).
- Metrics: rigid AMR-R and COV-R@0.5 (TD convention, `evaluate_confs.py --filter_mols`); share of ≥ 4-ring seeds with
  ring-dihedral RMSD > 10° (`eval/rings.py`: `lgeom.l_subset_errors` against the seed's paired GT conformer from
  `l_error.csv`, **PENDING Q-P1-003** on keeping that pairing); EXP-1 §3 row 9 costs; stereo preservation
  (`tools/stereo_check.py`, unchanged).
- Rule: σ_b* = argmin validation rigid AMR-R. **GO iff test rigid AMR-R(σ_b*) ≤ 0.10 Å and ≥ 4-ring share > 10°
  (σ_b*) ≤ 20 %** (EXP-1 §5.2). All three σ_b reported. Pipeline check: `rings.py` on the *unflowed* seeds must
  reproduce 49.6 % (`test_rings.py`).

**A6 / A6b.**
- *A6 preview (Stage A, only if GO; ≈ 0.35 GPU-h)*: `generate --arm a6` = geometry flow with the σ_b* checkpoint
  (τ frozen), then TD's own `perturb_seeds` + `td_sample` with the frozen A0 torsion model (CTRL_rematch s0) on that
  L, full QM9 test, evaluated as usual. Labelled "A6-preview (M1 geometry head, w_τ = 0), secondary". It is the same
  code path M2's A6 uses; it gives an early read of the S3/B1 brittleness regime (E-JOINT-005).
- *A6b (M2, not run in Stage A)*: the same τ-frozen sampler run on the **training** molecules' RDKit seeds
  (`data.pos[k]` of the paired cache), GT torsions re-matched to that L, and a torsion head trained on it (EXP-1 §4
  row A6b). TD's `standardize_confs.py` embeds its own ETKDG conformers (`:77-78`) and parses `argv` at import, so
  M2 adds `flexitors/training/rematch.py`, a port of `standardize_confs.py` `conformer_match` that takes the seed L
  as input, with a golden test (identity rollout ⇒ identical output on 5 molecules). Re-train (S5 recipe) vs DAgger
  fine-tune from A0, and the matching CPU cost: **PENDING Q-P1-004** (EXP-1 §5.4 Q-*-003).
- *E/Z in Exp 1*: in M1 τ is frozen and every TD-rotatable double bond is a row of W, so its dihedral is fixed to
  first order; Stage B adopts Exp 3's decision (**PENDING Q-P1-005**, the EXP-1 §5.4 Q-*-002 consult).

### 2.3 Exp 3: stereo-constrained sampler (QM9, GEOM-DRUGS)

Stage A = Step 0 + inference-only arms with the released checkpoints (`qm9_default`, `drugs_default`). K2-train, K3
and K3 + K2 need retraining and are not built (no reflected-kernel score, no `score_reflected`, no stereo edges, no
`flexitors/diffusion/` package — that name would also shadow TD's `diffusion`, style rule S6).

**Torsion classification (EXP-3 §3.1).** `torsion_types.td_rotatable_bonds` ports `chem_data/stereo_census.py:26-48`
(validated 0 / 996 mismatches, E-CHEM-060) and is tested against TD's own `get_transformation_mask`
(`torsion.py:9-39`): rotatable bond index = masked edge index // 2, because `featurize_mol` appends edges 2j, 2j+1 per
bond j (`featurization.py:86-95`). `classify_bond` gives the 7 classes (port of `stereo_census.py:51-73`, with
`double_stereo` split into specified/unspecified and `single_other` named `single_sp3`). Stereo flags store both
RDKit `FindPotentialStereo` (E-CHEM-070; cluster RDKit 2022.09.5 is the runtime version) and the structural test;
**constraints are chosen by the structural test** so results do not depend on the RDKit version (EXP-3 §3.1,
E-CHEM-060). Both versions' subsets are reported (the 2026.03.6 flags come from `chem_data/census_qm9_local.csv`).

**Targets φ_b (EXP-3 §3.1).** Reference atoms (c, a, b, d) = `rotatable_quads` (the TD rule of `sampling.py:135-142`).
Specified bonds: SMILES stereo atoms (s1, s2) + STEREOE/Z/CIS/TRANS; each of c ≠ s1, d ≠ s2 toggles cis ↔ trans;
φ_b = 0 (cis) or π (trans). Unspecified and `double_nonstereo` bonds: planarity only, target set {0, π}, nearest
value to the seed's own 3D state, flagged and reported separately. Triple bonds: only in K1-canonical.

**One subtle point the spec leaves implicit.** TD's update Δ is a rotation about pos[u] − pos[v] of the
`mask_rotate` side (`torsion.py:68-72`, "DIFFERENT FROM THE PAPER"), so a torsion update changes the measured
dihedral of (c, a, b, d) by +Δ or −Δ depending on edge orientation. `stereo.constrained_bonds` measures this sign
s_b once per bond (apply Δ = 1e-3, measure) and every δ-space operation (seed check, fold, flip) uses it. Tested.

**Arms (all with `StereoHook` for per-step logging, EXP-3 §3.3; 20 steps; 2K; seeds 0–2):**

| Arm | Mechanism | Touchpoint |
|---|---|---|
| O | TD unchanged; hook logs only (golden = TD) | `reverse.py` ≡ `sampling.py:176-246` |
| O + F-flip | CPU post hoc: offending bonds (|δ_b| > π/2) set to φ_b with `SetDihedral` | `standardization.py:19-20`; E-CHEM-046 |
| O + F-filter | drop conformers whose DB-only label ≠ target, regenerate the deficit up to 3 extra rounds, extra NFE counted (EXP-3 §3.4; without replacement recall drops, E-CHEM-064) | `generate.py` loop around one molecule |
| K1 | seed check (|δ_b| > π/2 → rotate by π), then drop constrained bonds from `edge_mask` / `mask_rotate` before `perturb_seeds` (VonMisesNet rule, E-CHEM-043) | `sampling.py:58-59, 105-108`; `generate_confs.py:150-153` |
| K1-wrong | as K1, frozen at the opposite isomer (specified bonds only) — negative control | same |
| K1-canonical | K1 on all double + triple TD-rotatable bonds | same |
| K2-inf | after `perturb_seeds`, constrained δ_b folded into I = [−π/2, π/2] (uniform on I, the K2 prior); each step perturb_b ← s_b·(fold(δ_b + s_b·perturb_b) − δ_b); TD's score unchanged ("thresholding" view, E-CHEM-048; TD σ(t), E-CHEM-013) | `sampling.py:204, 234` |

The per-step guard computes δ_b for all constrained bonds in batch with `get_torsion_angles` (`torsion.py:110-124`)
and counts E/Z violations (|δ_b| > π/2) and planarity violations (|δ_b| > 20° in the last 3 steps) per step and
molecule (EXP-3 §3.3); counts go to `budget.json` → `diag`. Final labels: `eval/stereo.labels` (port of
`ez_flip_analysis.py:37-52`; test reproduces `ezflip_qm9.out` R0 65/526, E-CHEM-061) against the SMILES target and
against the GT set; chirality via signed volumes for completeness.

**Step 0 (CPU, ORACLE for unspecified bonds; EXP-3 §5).** `posthoc snap` on R0 `confs.pkl`: every TD-rotatable double
bond set to its target (specified: SMILES; unspecified: the state of the nearest GT conformer by the `eval.pkl`
row minimum — **PENDING Q-P1-006**), then `evaluate_confs.py`. The ceiling decides only whether retrained arms ever
run on QM9 (not in Stage A); it is reported either way.

**QM9 cost trick (PENDING Q-P1-006).** Every Stage A Exp 3 operator is the identity on a molecule with no
constrained bond (K1 removes nothing, K2 folds nothing). So arms are generated only on the stereo subset
(102 molecules with a TD-rotatable double bond, E-CHEM-050/-060, plus triple-bond molecules for K1-canonical), and
the all-molecule non-inferiority universe is built by `sets.py splice`: outside the subset every arm *and* O carry
the released R_s conformers of the same seed, so the paired difference there is exactly 0, which is the true effect.
A test asserts that on a molecule without constrained bonds each arm reproduces TD bit-for-bit for the same RNG
state. This cuts QM9 Exp 3 from ≈ 10 GPU-h (EXP-3 §7) to ≈ 1.

**GEOM-DRUGS (after D0).** Precondition: the user's explicit approval of the EVAL §9 D0 download (`drugs.tar.gz`
25.3 GB + released `drugs_default` 6.4 MB, E-SAMPLE-077) and a /scratch space check; DECISION.md approves the
experiment, the protocol still requires per-download permission (orchestration README rule on downloads). Then:
20-molecule smoke (measures DRUGS per-conformer cost before any full run, EVAL §9); arms O, K1, K1-wrong, K2-inf
(+ F-flip on CPU, F-filter top-ups) × 3 seeds on the 141 DRUGS-test molecules with a TD-rotatable potential stereo
double bond (E-CHEM-051), 2K, δ = 0.75 Å, spliced as above; the E/Z-enriched DRUGS-validation set (≈ 500 molecules
with ≥ 1 specified acyclic stereo C=C/C=N, fixed seed, EXP-3 §4 (iii)) for the fidelity endpoint only.
**Defaults to fit the budget, PENDING Q-P1-007:** enriched set with 32 conformers per molecule (K-free; fidelity is a
per-conformer fraction) and 1 sampling seed; K1-canonical not run on DRUGS in Stage A.

**Analysis (`experiments/exp3_stereo/analyze.py`, pre-declared; PENDING Q-P1-006 for the family).** Primary
endpoint E/Z fidelity on molecules with specified stereo, and planarity (share of constrained bonds within 20°)
(EXP-3 §4). Default primary family: {K1 − O, K2-inf − O, F-flip − O, F-filter − O} × {fidelity, planarity}, paired
molecule bootstrap, Holm. Co-primary non-inferiority on all molecules: AMR-R margin +0.004 Å, COV-R@0.5 margin
−0.5 points, hierarchical (seed, molecule) bootstrap (EVAL §7). Secondary: K1-wrong − K1 (negative control),
K1-canonical − K1, stereo-subset AMR-R/P and COV-R@0.5/0.1, violations per step, NFE. QM9 reports both RDKit
versions' subsets and the 3 molecules whose GT ensembles mix E/Z (E-CHEM-061).

### 2.4 Exp 5: selection at matched compute (D-101, D-102)

Stage A = Track P only (DECISION.md: allocation dropped). No controller, gain model, signals or `td_backend`
incremental sampler is built. Released QM9 model, 20 steps, sampling seeds 0–2.

**D-101: a pool that leaks no K.** V5's finding is that a cap N_i = 6K_i *binds* and thereby tells every allocator
which molecules saturate. The fix is that no cap may ever bind and no arm may read the pool size:
- `generate --arm pool` draws, per molecule, n_i = max(N0, c_max·2K_i) conformers in one ETKDG + TD call
  (N0 = 128, c_max = 2). The **first N0 conformers are the K-free block**: the same size for every molecule, and
  distributed exactly as an N0-only run, whatever n_i is.
- Track P arms read the prefix [0, c·2K_i) (K is used only to fix the benchmark's output size 2K, EXP-5 §4.5).
  The ORACLE headroom recompute reads only [0, N0). `test_pool.py` asserts that every arm's demand ≤ n_i ("the cap
  never binds") and that all K-free blocks have size N0.
- N0 = 128 is D-101's suggestion; at 2× budget the uniform rule needs ≈ 54 per molecule (27.3 at 1×,
  `sample_data/alloc.out`), so no K-free rule up to 2× is capped; only the ORACLE allocator can reach N0, and its
  share at N0 is printed.
- To save GPU, only seed 0 carries the N0 block; seeds 1–2 draw n_i = 4K_i (Track P needs nothing more).
  **PENDING Q-P1-008** (N0; one seed vs three; whether the uncapped headroom recompute is still wanted).

**D-102: PG at matched NFE.** PG adds no network call; its cost is the kernel (wall-clock). Arms (all output exactly
2K_i; NFE per molecule from `budget.json`, matched within 1 %, EXP-5 §5):

| Arm | Proposals (NFE) | Output | Role |
|---|---|---|---|
| P0 | TD, pool [0, 2K) (2K·20) | all | baseline |
| P1 | TD, pool [0, 4K) (4K·20) | random 2K | NFE control for P3 (= TD in expectation) |
| P2 | PG on 2K (2K·20) | all | PG at 1× (vs P0) |
| **P2b** | **PG on 4K jointly (4K·20)** | **random 2K** | **NFE-matched PG comparator (D-102)** |
| P3 | TD, pool [0, 4K) (4K·20) + MMFF94 CPU | selector | main selection arm |
| P6 | PG on 4K (P2b's proposals) + MMFF94 CPU | selector | composition |
| ORACLE | pool [0, 4K) | greedy min-AMR | analysis row only |

Primary Holm family (revised for Stage A, **PENDING Q-P1-009**): {P3 − P1, P3 − P2b} × {AMR-R, COV-R@0.5}, TD
convention, 3 sampling seeds, molecule bootstrap of seed means; precision printed beside every row (EVAL §2). K3 kill
rule (EXP-5 §11) applied to P3 vs P2b on recall *and* precision. Secondary: P6 − P2b, P2 − P0, P2b − P1, P1 − P0
(sanity ≈ 0). P4/P5 need the dropped allocator and are not run. MMFF CPU core-seconds are reported beside NFE; PG
arms report kernel wall-clock (EXP-5 §4.1: CPU signals are never hidden).

**PG settings (PENDING Q-P1-009).** TD's README gives only DRUGS-tuned PG hyper-parameters
(`torsional-diffusion/README.md:71-90`; CLI `generate_confs.py:38-46`). Default: the "minimizing recall error"
permutation-invariant set, `--batch_size 128` as in the README. TD's PG repulsion acts within a loader batch
(`sampling.py:207-229`, n = `data.num_graphs`), so for 4K > 128 the coupling is per batch of 128, as in the published
runs. `generate --arm pg` simply calls `td_sample(..., pg_*)`.

**Selector (D-102 (a)).** Re-chosen on the QM9 validation pool (seed 0, n_i = 4K_i; set from **Q-P1-010**) before
the test pools are generated, then frozen in `config.yaml` (committed). Candidates: {FPS, lowest MMFF94,
MMFF-filter q ∈ {0.25, 0.5} + FPS} × FPS start {lowest energy (EXP-5 §4.4), random (prototype
`sample_data/select.py:13-18`)}; criterion: lowest validation AMR-R among candidates whose AMR-P is not worse than
random's (**PENDING Q-P1-009**). Torsions = heavy non-ring, non-triple, non-sp bonds (port of
`sample_data/signals.py:14-28`); energy = MMFF94 single point (EXP-5 §4.2 S3 says MMFF94; TD's `mmff_energy` is
MMFF94s, `likelihood.py:19-21`, so it is not reused).

**Scoring.** Each pool and each PG generation is scored once by the unchanged `evaluate_confs.py --out_results`
(full RMSD matrix). `posthoc select` writes the selected column indices (and a `confs.pkl` copy for archiving);
`eval/paired.per_molecule(eval_pkl, columns=...)` takes column subsets of that matrix. This gives exactly the
evaluator's numbers (each RMSD entry is independent of the other columns) at a fraction of the CPU; `test_pool.py`
checks equality against a separate `evaluate_confs.py` run on 20 molecules (EVAL §0 rule 1).

**ORACLE headroom (analysis; `analyze.py headroom`).** On the seed-0 K-free block: exact expected recall of a random
n-subset (hypergeometric, port of `sample_data/curves.py`), uniform, scaled-2K and cross-fitted static n_rot rules
and the greedy ORACLE allocator at 0.5 / 1 / 2 × Σ2K, with "share at N0" printed. This re-states the 0.0042 Å figure
as the uncapped number V5 asked for (D-101), labelled ORACLE; it reopens nothing unless the user asks.

## 3. Mechanism → source → touchpoint table

Every row's citation is copied from the spec section named; where the spec gives no external source the row says so
(the code comment then cites the spec section only). Baseline line numbers were read on 2026-10-16 at HEAD 9faa09f.

| # | Mechanism | Spec section; evidence | flexitors location | Baseline touchpoint (path:line) |
|---|---|---|---|---|
| 1 | One dihedral per TD-rotatable bond | EXP-1 §2.2 rev.; E-JOINT-032 | `geometry/dihedrals.py` | `utils/standardization.py:69-89`; `diffusion/sampling.py:129-142` |
| 2 | Analytic dihedral gradients W | EXP-1 §2.2 rev.; E-JOINT-032 (no formula source; FD test) | `geometry/dihedrals.py` | sign of `utils/torsion.py:110-124` |
| 3 | Rigid rows J_rig | EXP-1 §2.2; TD Eq. 10 PDF p. 7, E-JOINT-044 | `geometry/projector.py` | `diffusion/likelihood.py:59-65, 85-93` |
| 4 | P_W = I − Cᵀ(CCᵀ)⁺C | EXP-1 §2.2 rev. (D-004); E-JOINT-032 | `geometry/projector.py` | — (new) |
| 5 | Old P_L (M0 negative control only) | EXP-1 §5.2 M0 (a); E-JOINT-032 | `experiments/exp1_m1/analyze.py` | `diffusion/likelihood.py:68-74` (`dx_dtau`) |
| 6 | Bond window, clash test, halving backtrack | EXP-1 §2.6 step 1 (design; no external source) | `geometry/constraints.py` | — |
| 7 | Chirality guard (signed volumes, bridgeheads) | EXP-1 §2.8 rev.; E-JOINT-032, E-JOINT-026 | `geometry/constraints.py` | — |
| 8 | Per-atom `1o` velocity head | EXP-1 §2.4 table; E-JOINT-021, E-JOINT-042 (TD Prop. 2, PDF p. 6) | `models/flexitors_score.py` | `diffusion/score_model.py:83-89, 141-143` |
| 9 | Geometry clock through λ embedding | EXP-1 §5.1 (`flexitors_score.py` row) | `models/flexitors_score.py` | `diffusion/score_model.py:192-199` |
| 10 | Paired endpoints, same Δτ on both | EXP-1 §2.2 steps 1–3; E-JOINT-023 | `training/transforms.py` | `utils/dataset.py:71, 74-76, 79-102`; `utils/torsion.py:57-75` |
| 11 | Independent clocks σ, t | EXP-1 §2.2 step 2; E-JOINT-072 | `training/transforms.py` | `utils/dataset.py:71` |
| 12 | Interpolant x_t and velocity target u | EXP-1 §2.2 steps 4–5; ET-Flow Eq. 2–5 PDF p. 4, E-JOINT-076 | `training/transforms.py` | `utils/dataset.py:18-34` |
| 13 | Loss: TD DSM + projected L loss / (3n−6−m) | EXP-1 §2.2 step 6; E-JOINT-022 | `training/losses.py` | `utils/training.py:25-31` |
| 14 | Training loop, checkpoint, resume | EXP-1 §5.1 (`train.py` row) | `training/train.py` | `train.py:30-62, 146-149`; `utils/utils.py:23-45` |
| 15 | TD reverse SDE step | EXP-1 §2.1; TD Alg. 3 PDF p. 17, E-JOINT-024 | `sampling/reverse.py` | `diffusion/sampling.py:122-126, 176-246` (step `:204`) |
| 16 | Projected Euler geometry step, Δt = 1/N | EXP-1 §2.2 "Sampling", §2.5; E-JOINT-076 | `sampling/geometry_flow.py` | — |
| 17 | τ-frozen independent L sampler (M1, A6, A6b) | EXP-1 §4 rows A6/A6b (D-005), §5.2 | `sampling/generate.py --arm geom/a6` | `diffusion/sampling.py:63-102`; `generate_confs.py:64-69, 150-153` |
| 18 | TD rotatable rule, bond classes | EXP-3 §3.1; E-CHEM-020, E-CHEM-060, E-CHEM-070 | `geometry/torsion_types.py` | `utils/torsion.py:9-39`; `round5/ledger/chem_data/stereo_census.py:26-73` |
| 19 | E/Z targets φ_b | EXP-3 §3.1 (no external source) | `geometry/torsion_types.py` | `diffusion/sampling.py:135-142` |
| 20 | K1 freeze (+ wrong, canonical) | EXP-3 §3.2 K1; E-CHEM-043 | `sampling/stereo.py` | `diffusion/sampling.py:58-59, 105-108`; `generate_confs.py:150-153` |
| 21 | K2-inf half-circle fold | EXP-3 §3.2 K2; E-CHEM-013, E-CHEM-048 | `sampling/stereo.py` | `diffusion/sampling.py:204, 234`; `utils/torsion.py:68-72` (update sign) |
| 22 | Per-step stereo guard | EXP-3 §3.3 | `sampling/stereo.py` | `utils/torsion.py:110-124`; `diffusion/sampling.py:184-234` |
| 23 | F-flip | EXP-3 §3.4; E-CHEM-046 | `sampling/stereo.py` | `utils/standardization.py:19-20` |
| 24 | F-filter with replacement | EXP-3 §3.4; E-CHEM-064 | `sampling/generate.py` | `evaluate_confs.py:47-54` (non-isomeric filter, E-CHEM-028) |
| 25 | DB-only / CH-only labels, fidelity, planarity | EXP-3 §3.3, §4; E-CHEM-061, E-CHEM-063 | `eval/stereo.py` | `round5/ledger/chem_data/ez_flip_analysis.py:37-79` |
| 26 | Step 0 snap ceiling (ORACLE for unspecified) | EXP-3 §5 Step 0 | `sampling/stereo.py`, `posthoc snap` | `utils/standardization.py:19-20` |
| 27 | K-free pool block, cap never binds | EXP-5 §5 S0; D-101 (V5) | `sampling/generate.py --arm pool` | `generate_confs.py:183-187` (2K rule) |
| 28 | MMFF94 filter + torsion FPS selector | EXP-5 §4.4 Track P, §4.2 S3; E-SAMPLE-025 | `sampling/select.py` | `round5/ledger/sample_data/select.py:11-18, 44-47`; `signals.py:14-28` |
| 29 | Particle Guidance arms (P2, P2b, P6) | EXP-5 §5 P2/P6; E-SAMPLE-030…033; D-102 | `sampling/generate.py --arm pg` | `diffusion/sampling.py:128-232`; `generate_confs.py:38-46` |
| 30 | NFE / CPU accounting, matching ±1 % | EXP-5 §4.1; EVAL §8 | `sampling/budget.py` | `generate_confs.py:153` |
| 31 | Hypergeometric expected recall (ORACLE headroom) | EXP-5 §3; E-SAMPLE-020, E-SAMPLE-022 | `experiments/exp5_select/analyze.py` | `round5/ledger/sample_data/curves.py` |
| 32 | Paired bootstrap, Holm, hierarchical CI | EVAL §7; E-SAMPLE-088, E-SAMPLE-089 | `eval/paired.py` (imports) | `tools/analyze_leakfree.py:130-160` |
| 33 | TD-convention universe | EVAL §3; E-SAMPLE-065 | `eval/paired.py` | `evaluate_confs.py:81-84, 152, 166-167` |
| 34 | Budget-matched ref_half | EXP-1 §3.6 (D-001) | `tools/analyze_leakfree.py` | `tools/analyze_leakfree.py:74, 116-128` |
| 35 | Floor fix + other-half floor | EXP-1 §9 item 1; E-JOINT-031 | `tools/local_structure_analysis.py` | `tools/local_structure_analysis.py:74, 281-282, 302` |
| 36 | Rigid stratum | EXP-1 §3 row 10; E-JOINT-007 | `data/sets.py rigid` | `tools/breakdown.py:32` |
| 37 | Ring-dihedral error vs paired GT | EXP-1 §5.2 Gate 0; E-JOINT-004 rev. | `eval/rings.py` | `tools/lgeom.py:317-338`; `tools/make_l_seed_pickles.py:108-114` |

## 4. Tests, dead-code audit, code style

### 4.1 Unit and golden tests (`flexitors/tests/`; `@pytest.mark.cluster` = needs torch/PyG/data, run on gnode118
with `srun -w gnode118 -p plafnet2 -A plafnet2 --cpus-per-task 4 --mem 16G pytest -m cluster`; the rest also run
locally, but stereo tests are pinned to the cluster's RDKit 2022.09.5)

| File | What it checks |
|---|---|
| `test_baseline_frozen.py` | LF-normalised SHA-256 of every TD file `td.py` imports, plus `generate_confs.py`, `evaluate_confs.py`, `standardize_confs.py`, equal to pins recorded at Stage A start (the three pins of `tools/tests/test_leakfree.py` reused) |
| `test_td_gateway.py` | importing `flexitors.td` leaves cwd unchanged, finds the torus tables, does not recompute them; no flexitors module is named `diffusion`, `utils`, `train` |
| `test_dihedrals.py` | `rotatable_quads` = `standardization.get_torsion_angles` bonds/end atoms on 50 QM9 (+ 50 DRUGS after D0) molecules; W vs central FD, float64, rel. err < 1e-5 |
| `test_projector.py` | EXP-1 §5.3: idempotent, symmetric; ‖W·P_W‖, ‖J_rig·P_W‖ < 1e-6; rank 3n−6−m; m = 0; padded batch = per-molecule; old P_L passes "P_L·J_τᵀ ≈ 0" yet fails W·P_L ≈ 0 (documented negative example) |
| `test_invariance.py` | 0.05 Å RMS projected step moves each TD quadruple < 1e-3 rad and drift ∝ step² (0.01/0.02/0.05 Å); guard catches a deliberate 0.3 Å flipping step; TD torsion step leaves bonds/angles unchanged (> 1e-5 fails) |
| `test_constraints.py` | enantiomer (−x) flagged; bond stretch backtracked; GT conformers pass; counters add up |
| `test_parity.py` | on x and −x: `edge_pred` negates (0o), `l_pred` negates (1o) |
| `test_model.py` | `FlexiTorsScoreModel.edge_pred` bit-identical to `TensorProductScoreModel(lambda_embed_dim=32)` with copied weights; `build_model` TD keys = `get_model` keys; m = 0 forward runs |
| `test_transform.py` | same Δτ on both endpoints (TD-quadruple differences unchanged mod 2π); P_W(x_t) u = u; t ∈ [t_ε, 1−t_ε]; σ_b = 0 gives x_t = interp |
| `test_losses.py` | `td_score_loss` = TD `train_epoch` loss on a fixed batch (`limit_iters=1`, lr 0); projected loss 0 when `l_pred = u` |
| `test_reverse.py` (golden) | `reverse_diffusion(hook=None)` and with a no-op hook bit-identical to `td_sample` (3 molecules × 8 conformers, same RNG) |
| `test_generate.py` (golden) | `generate --arm td --seed 3` = `generate_confs.py --seed 3 --no_energy` on a 5-molecule csv (positions ≤ 1e-6 Å, same keys and counts); NFE = n·steps·[m > 0] |
| `test_stereo.py` | EXP-3 §6 tests 1–7: TD rule replication; φ_b on (E)/(Z)-but-2-ene and oximes with reference-atom swaps; fold keeps δ in I and is an involution at ±π/2; update sign s_b; K1 frozen torsions bit-identical through sampling; zero violations for K1/K2-inf on 20 molecules, non-zero for O on the 38; `eval/stereo` reproduces 65/526 (R0); arms = TD on molecules with no constrained bond |
| `test_select.py` | EXP-5 §9: exactly 2K; subset of proposals; q = 0 + FPS = pure FPS; deterministic; FPS port = prototype on a fixture |
| `test_budget.py` | NFE accounting; `assert_matched` rejects a 2 % mismatch |
| `test_pool.py` | cap never binds; K-free block size N0 for all molecules; column-subset metrics = separate `evaluate_confs.py` run (20 molecules) |
| `test_paired.py` | EVAL §10: reproduces `R0_base_seed0/summary.txt` (AMR-R 0.1752, COV-R 89.0293, 65 failures); reproduces `contrasts.csv:50` (S3 − CR +0.00323 [+0.00127, +0.00523]) with the same seed |
| `test_rings.py` | `rings.ring_errors` on the unflowed L_etkdg2L seeds reproduces `l_error.csv` (4,181 ≥ 4-ring seeds, 49.6 % > 10°) |
| `tools/tests/test_lsa_floor.py`, `tools/tests/test_leakfree.py` (extended) | §2.1 C1, C2 |

### 4.2 Smoke on gnode118 (20 molecules per flag; one sbatch row each; nothing larger starts before all pass)
Exp 1 `smoke20` and `smoke1ep` (§2.2), `geom`, `a6`; Exp 3 every arm on 20 stereo molecules (QM9) and, after D0, on
20 DRUGS molecules; Exp 5 `pool` (with N0), `pg` at c = 1 and c = 2, `posthoc select`. Each checks exit 0, expected
conformer counts, `budget.json` present and matched, `evaluate_confs.py` runs, guard counters plausible. Every
`run.sbatch` also gets `sbatch --test-only`. Resource rule: ≤ 5000 MB per CPU (plafnet2), `--nodelist=gnode118`,
skip-if-done and `--resume` + requeue as `slurm/ablation_train_array.sbatch`.

### 4.3 Dead-code audit (before every push; run by the implementer, checked by both reviewers)
1. `flexitors/tests/test_no_dead_code.py` (AST, no new dependency): every top-level function, class and method in
   `flexitors/` is referenced by name from non-test code in `flexitors/` or `experiments/`; every imported name is
   used; no module is unreachable from an entry point (`generate`, `posthoc`, `train`, `sets`, `analyze.py`).
2. Every `config.yaml` key is read by code, and every CLI flag appears in some `config.yaml` or `run.sbatch`
   (an unused flag is dead surface).
3. `ruff --select F401,F811,F841` if it is already in the venv; it is not downloaded for this.
4. `grep` for `TODO`, `XXX`, commented-out code, `print(` in library modules: must be empty.

### 4.4 Code-style rules
- S1 Python 3.9 (cluster env), `from __future__ import annotations`, type hints and a docstring (one line + args) on
  every public function.
- S2 Above each mechanism: `# EXP-n §x; E-ID (paper, PDF p. N, eq. M)` exactly as in §3; ports start with
  `# port of <path:lines>` and have a golden test.
- S3 Defaults reproduce TD bit-for-bit; new behaviour only through an explicit `--arm` or config key.
- S4 Functions ≤ ≈ 50 lines; classes only for stateful objects (hooks, transform, model, budget); no inheritance
  beyond the TD model and PyG transform.
- S5 Seeds explicit and set after `import flexitors.td`; no hidden global state besides the RNGs.
- S6 Absolute imports (`from flexitors.geometry import projector`); no module or package named `diffusion`, `utils`,
  `train`.
- S7 Library code logs, never prints; CLIs print one `SUMMARY` line (the `evaluate_confs.py:177` pattern).
- S8 ORACLE code paths and outputs carry `oracle` in their names and an `ORACLE` tag (README rule 5).
- S9 No `except:` that hides errors except where TD already tolerates them (embedding failures), each counted.
- S10 LF endings; no notebooks; commits by the repo author only, no co-author lines.

## 5. Compute and Stage A schedule

### 5.1 Basis (measured anchors only)
- Training: 11.7 GPU-h per 100-epoch QM9 run on one RTX 3090 (377.7 s/epoch + validation; E-JOINT-008, V5 verified).
- Inference: one packed full QM9 test pass at 1× (Σ2K ≈ 25.3 k conformers on the 927-molecule success set,
  `sample_data/alloc.out`) = 15–25 min of a GPU slot (task brief; 14 min 18 s wall unpacked, E-SAMPLE-053). Used as
  0.33 GPU-h per 1×-equivalent (range 0.25–0.42), linear in conformers. ETKDG/CPU work dominates these runs.
- GEOM-DRUGS per-conformer cost is **unmeasured**; its rows are estimates until the 20-molecule smoke.
- Pool size factor: Σ max(128, 4K_i) / Σ2K_i ≈ 5.4 (≈ 857 molecules with K ≤ 32 at 128, the K > 30 tail at 4K;
  `sample_data/strata.out` K bins). Estimate, printed exactly by `sets.py` before submission.

### 5.2 GPU-h per task

| Task | GPU-h (central; range) | Basis |
|---|---|---|
| CPU tasks C1–C4, M0, Exp 3 Step 0, all unit/golden tests | 0 | CPU on gnode118 (≈ 3–5 CPU-h) |
| Smoke rows (all experiments, 20 molecules each) | 0.3 (0.2–0.4) | |
| Exp 1 `smoke1ep` | 0.15 (0.12–0.18) | 1 epoch at TD speed + unmeasured overhead (EXP-1 §6) |
| Exp 1 M1 = A11, 3 σ_b × 30 epochs | 10.5 (ceiling 16) | 0.3 × 11.7 × 3 |
| Exp 1 Gate-0 evaluation (val + test rigid stratum, ≥ 4-ring seeds, 3 σ_b) | 0.5 (0.3–0.7) | rigid molecules ≈ 25 % of Σ2K; seeds 4,181 × 20 steps |
| Exp 1 A6 preview (only if GO) | 0.35 (0.25–0.42) | one 1× pass + geometry flow |
| Exp 3 QM9 arms on the stereo subset (O, K1, K1-wrong, K1-canonical, K2-inf × 3 seeds; F-filter top-ups) | 0.9 (0.6–1.3) | subset ≈ 15 % of Σ2K per arm-seed |
| Exp 5 validation pool (4K, seed 0) | 0.66 (0.5–0.85) | 2× |
| Exp 5 test pools (seed 0 max(128, 4K); seeds 1–2 4K) | 3.1 (2.4–3.9) | 5.4× + 2 × 2× |
| Exp 5 PG arms (P2 1× and P2b 2×, × 3 seeds) | 3.0 (2.3–4.5) | 9× plus invariant-kernel overhead ≤ 1.5× |
| **Core subtotal** | **≈ 19.5 (ceiling ≈ 26)** | |
| Exp 3 DRUGS (smoke; O, K1, K1-wrong, K2-inf × 3 seeds on 141 test molecules; enriched set 4 arms × 1 seed × 32 confs) | ≈ 5 (3.7–7.1) | estimate; restated after the DRUGS smoke |
| **Stage A total** | **≈ 24.5** | inside 15–25 only at the central estimates |

### 5.3 Cut rules (pre-declared, applied in this order if a smoke measurement pushes the total above 25 GPU-h)
1. `smoke1ep` projects M1 > 16 GPU-h → M1 does not start; orchestrator asks the user (EXP-1 §5.2).
2. PG invariant-kernel overhead > 1.2× in smoke → use the README's non-invariant recall set (saves ≈ 0.5–1.5).
3. Drop the A6 preview (0.35).
4. DRUGS enriched set → 16 conformers per molecule (saves ≈ 0.4); then DRUGS K1-wrong on 1 seed (≈ 1).
5. Anything still over budget is deferred to Stage B and reported, never silently dropped.

### 5.4 Schedule (gnode118, 4 GPUs shared; GPU numbers are slots)

| Step | When | GPUs | GPU-h | Gate to next step |
|---|---|---|---|---|
| S0 | Day 0 | none | 0 | C1 regression passes; C2 run; M0 passes its kill rule; all unit + golden tests green; Step 0 table written |
| S1 | Day 1 a.m. | 1 | 0.45 | every 20-molecule smoke passes; `smoke1ep` gives the M1 ask (restated to the orchestrator) |
| S2 | Day 1–2 | 0–2 | 10.5 | M1 runs (≈ 3.5–5 h wall each, resume on) |
| S3 | Day 1–2 | 3 (PACK = 3) | ≈ 7.6 | Exp 5 validation pool → selector frozen in `config.yaml` (commit) → test pools → PG arms; Exp 3 QM9 arms |
| S4 | Day 2 | 0 | 0.85 | Gate-0 evaluation → GO/NO-GO report; A6 preview if GO; CPU: pool evaluation and Exp 3/5 analyses |
| S5 | after user approval of D0 | 0–3 | ≈ 5 | D0 download + released-sample re-score (CPU), DRUGS smoke (restate cost), DRUGS Exp 3 arms |
| | | | **≈ 24.4** | |

Exp 2 and Exp 4 are not touched (DECISION.md). M2 (A4, A6, A6b) needs Gate 0 = GO and a new user approval.

## 6. Consult questions filed (`round5/consult/`)

| ID | To | Spec section | Decision that depends on it | Default used in this plan |
|---|---|---|---|---|
| Q-P1-001 | R-JOINT | EXP-1 §2.2, §5.2, §5.4 (Q-*-001) | t endpoint handling (γ′ singular), σ fed in τ-frozen sampling, whole-network vs head-only M1 training, V2 interpolant chirality | t_ε = 1e-3; σ_min; whole network, w_τ = 0; no rejection in V2 training (flips counted) |
| Q-P1-002 | R-JOINT | EXP-1 §2.2 step 1, §7 risk 2 | paired cache reuse vs rebuild; `pair_ok` filter; adding m = 0 training molecules | reuse; apply `pair_ok`; TD training set (m ≥ 1) |
| Q-P1-003 | R-JOINT | EXP-1 §5.2 M0/M1 | M0 "recovery" definition; ring metric pairing; σ_b selection on validation | AMR-R-based recovery; keep `l_error.csv` pairs; argmin val rigid AMR-R |
| Q-P1-004 | R-JOINT (+ orchestrator for timing) | EXP-1 §4 A6b, §5.4 (Q-*-003) | A6b recipe (retrain vs DAgger fine-tune) and re-matching CPU cost, to cost M2 | retrain with S5 recipe; cost UNVERIFIED |
| Q-P1-005 | R-CHEM (cc R-JOINT) | EXP-1 §2.8, §5.4 (Q-*-002); EXP-3 | which E/Z operator FlexiTors inherits in Stage B | M1 τ frozen (no exposure); adopt Exp 3's winning inference arm |
| Q-P1-006 | R-CHEM | EXP-3 §3.1–3.4, §4, §5 | subset generation + splicing; primary family; Step 0 target for unspecified bonds; F-filter output when < 2K valid | as §2.3 |
| Q-P1-007 | R-CHEM | EXP-3 §4 (ii)–(iii), §7 | DRUGS arm set, seeds and sample count on the enriched set under the budget | O, K1, K1-wrong, K2-inf; enriched 32 confs × 1 seed |
| Q-P1-008 | R-SAMPLE | EXP-5 §5 S0, §11 K1; D-101 | N0, N0 block on one seed vs three, whether the uncapped headroom recompute is wanted | N0 = 128, seed 0 only, yes |
| Q-P1-009 | R-SAMPLE | EXP-5 §5 Track P, §6, §11 K3; D-102 | revised Track-P arm list and primary family; PG hyper-parameters on QM9; selector candidates and validation rule | as §2.4 |
| Q-P1-010 | R-SAMPLE (EVAL owner) | EVAL §1 (validation subset), §0 rule 3 | how the QM9 validation subset and its GT pickle/csv are built (used by Exp 1 and Exp 5) | see file |
| Q-P1-011 | R-JOINT | EXP-1 §3.6, §9 item 1 | Holm family of the budget-matched re-score; draws; whether the `--mmff_seeds` floor file is re-run | one family of 8; N = 10; test file only |

## 7. Risks and what this plan deliberately leaves out
- **m = 0 forward pass** may fail in TD's bond-conv graph (never exercised by TD training); first test, contained
  override if needed (§2.2).
- **Gate 0 trains on m ≥ 1 molecules but is read mostly on rigid ones** (Q-P1-002). If R-JOINT wants m = 0 training
  molecules, a paired cache for them (no torsion matching needed) is a CPU build of ≈ hours, before M1.
- **Golden equality may be bitwise only on CPU**; GPU e3nn/scatter is non-deterministic (round-2 IMPLEMENTATION.md
  §3 note quoted in `train.py:41-42`). Golden tests run on CPU; GPU smoke compares statistics, not bits.
- **Pool evaluation CPU cost** grows as Σ K·n_i (large-K molecules dominate); measured on 20 molecules first, run
  with the packed eval array pattern (`slurm/r2_eval_array.sbatch`).
- **DRUGS numbers are unmeasured**; the D0 download needs explicit user approval (25.3 GB).
- Left out on purpose (built only when their stage is approved): V2 full / DAgger / `rematch.py` (M2), V3a/V3b,
  `rings.py` CP code, `reconstruct.py`, scalar/pucker heads, validity / energy / timing modules, reflected-kernel
  training (K2-train), stereo edges (K3), Exp 5 signals / gain model / controller / adaptive allocation, Exp 2, Exp 4.
