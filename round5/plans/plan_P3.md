# Plan P3 (round 5, Stage A): throughput and reliability on gnode118

Coding planner P3 of 3, written 2026-10-16. Status: COMPLETE (all sections filled).
Scope = `round5/DECISION.md` Stage A: `flexitors/` package; Exp 1 up to gate M1 (smoke first); Exp 3 stereo-constrained
sampler, inference on QM9 and GEOM-DRUGS with released checkpoints; Exp 5 selection at matched compute (allocation
dropped); CPU items EXP-1 §3.6 (budget-matched leak-free re-score) and the floor bug at
`tools/local_structure_analysis.py:302`. Budget 15–25 GPU-h. Disputes D-101 and D-102 resolved in §6.

Conventions: `path:line` locators were read in this session from the local checkout (TD at `torsional-diffusion/`,
equal to the cluster checkout plus the same patches). Spec references are `EXP-n §x` / `EVAL §x`; evidence IDs are
cited only where a spec cites them for the same statement. INFERENCE = my engineering judgement. PENDING Q-P3-NNN =
default used until the consult is answered (`round5/consult/`).

## 0. Summary

**Design stance.** The throughput lever is not faster kernels. It is **not generating what can be reused or
spliced**, and **never re-running a job because a failure was found late**. Five choices follow from that:
1. **Per-molecule seeding** (§3.2). Each molecule's RNG is seeded from (global seed, SMILES, batch index), so:
   - an Exp 3 constraint arm is bit-identical to its O run on every molecule without a constrained bond, and only the
     ≈ 10 % (QM9) / 14 % (DRUGS) affected molecules are generated per arm, then spliced (≈ 10× less GPU);
   - Exp 5's 2× pool is O's own 2K plus one extra 2K batch (common random numbers);
   - a killed or OOM-crashed generation resumes exactly at the molecule where it stopped.
2. **One reverse loop for every inference arm** (`flexitors/sampling/constrained_sampler.py`), golden-tested
   bit-identical to TD's `sample()` (`torsional-diffusion/diffusion/sampling.py:116-246`) when no hook is active.
3. **Round-2 reliability machinery reused unchanged in spirit:** provenance guard (`slurm/common.sh:107-122`), PACK=3
   packing (`:147-172`), R2_STRICT output checks (`slurm/ablation_train_array.sbatch:139-146`), resume + SIGUSR1
   requeue (`torsional-diffusion/train.py:30-62,107-110`; `ablation_train_array.sbatch:68-122`), atomic
   checkpoints, ≤ 5000 MB per CPU (`slurm/submit_round2.sh:9-10`). Added: in-job **OOM retry with backoff**
   (the round-2 OOM failures 10860/5 and 11176 cost two manual reruns, `round2/RUNLOG.md`), partial-output resume
   for generation, a **GPU-h ledger** that refuses submissions beyond the Stage A cap, and one `STATUS.md` so the
   agents read one file instead of dozens of logs.
4. **Every gate that the specs make mechanical is enforced by a script** (smoke PASS → GPU wave; smoke-measured M1
   projection > 16 GPU-h → `FX_BUDGET_STOP`; DRUGS 20-molecule timing → arm list under the 4.5 GPU-h cap). The agents
   are needed only for decisions the specs leave to people.
5. **Fail fast before a GPU is touched:** every job checks inputs, hashes, library versions, the TD torus tables in
   the working directory, xtb version, disk, and free GPU memory first.

**Stage A cost (§12):** ≈ 18.6 GPU-h expected, ≈ 27 GPU-h worst case (M1 at its 16 GPU-h ceiling), auto-trimmed to
≤ 25 by a pre-declared cut order; ≈ 75–220 CPU-h, most of it the DRUGS RMSD evaluation (unmeasured, measured first on 20
molecules). Wall clock ≈ 14–16 h on gnode118 if the queue is free, ≈ 2 days with typical contention.

**Downloads (one approval, §8):** TD's `drugs.tar.gz` (25,308,038,362 B), `drugs_default/best_model.pt`
(6,392,121 B), `drugs_default/model_parameters.yml` (size not recorded; text, ≈ 1 KB), `drugs_default/drugs_steps20.pkl`
(457,404,318 B); total ≈ 25.77 GB, from the TD authors' shared Drive (file IDs in §8).

**Consults filed:** Q-P3-001 … Q-P3-009 (§13).

## 1. Cluster facts this plan is built on

| Fact | Source |
|---|---|
| gnode118: 4 × RTX 3090 (24 GB), 48 CPUs shared, Ubuntu 22.04.5; login node ada is Rocky 9.8 | task brief; `round3/grounding.md:42` |
| plafnet2: MaxMemPerCPU 5000 MB, otherwise SLURM silently raises the CPU count | `slurm/submit_round2.sh:9-10`; round-2 FIXES X2 |
| 4-day time limit; requeue only on request (JobRequeue = 0) | `slurm/ablation_train_array.sbatch:9-24` |
| `/scratch/nishanth.r/tordiff` is local to gnode118; `$HOME` has a 30 GB quota | `slurm/common.sh:4-11` |
| Venv: Python 3.9, torch 1.13.1+cu117, PyG 2.0.4, e3nn 0.5.1, RDKit 2022.09.5, numpy 1.23.5 | `slurm/setup_env.sh` (cu117 profile); `round3/grounding.md:47` |
| `torsional-diffusion/diffusion/torus.py:26-34` writes `.p.npy`/`.score.npy` into the **CWD** on first import (race) | `notes/code_concerns.md:64`; `setup_env.sh` step 4 precomputes them in `$REPO` |
| xtb 6.7.1: `module` is not initialised in a non-login `srun bash`; binary `/data3/u22/xtb-6.7.1/bin/xtb` runs with `LD_LIBRARY_PATH=/data3/u22/xtb-6.7.1/lib`; 0.135 s median per QM9 `--opt normal` | `round3/grounding.md:42-46,65-72` |
| gnode118 has outbound internet (PyPI and Zenodo returned 200) | `round3/grounding.md:48-50` |
| Measured QM9 test generation, released model, unpacked: 25,330 conformers in 14:18 wall, 159 % CPU, max RSS 3.75 GB | `round5/ledger/sample_data/R0_base_seed0_generate.log:2063-2075` |
| Packed evaluation ≈ 0.17 GPU-h per QM9 condition (unpacked 0.26); 100-epoch training ≈ 11.7 GPU-h (377.7 s/epoch) | `round3/grounding.md:56-63` |
| GPU OOM caused by other jobs on the same GPU, twice (ours used 2 GB of 24) | `round2/RUNLOG.md` (10860 task 5; 11176) |
| DRUGS test: 1000 molecules, 106,778 GT conformers (2K run = 213,556 samples, 8.4 × QM9); the 141 stereo-DB molecules have 10,104 GT conformers (2K = 20,208), 25.5 heavy atoms on average | computed here from `round5/ledger/chem_data/drugs_test_smiles_corrected_geomol.csv` and `census_drugs.csv` |

INFERENCE from the 159 % CPU line: QM9 generation is partly CPU-bound (the per-torsion numpy rotation in
`torsional-diffusion/utils/torsion.py:57-75` runs per conformer per step). That is why packing 3 processes per GPU
helped, and DRUGS (≈ 4× more rotatable bonds) will be even more CPU-bound, so DRUGS inference jobs get 8 CPUs for 3
packed processes, as in round 2.

## 2. `flexitors/` package layout and module APIs

Layout follows `round5/PLAN.md` "Code structure" (`geometry/`, `models/`, `sampling/`, `training/`, `eval/`, `data/`)
and the module tables of EXP-1 §5.1, EXP-3 §6, EXP-5 §8 and EVAL §10. **Only modules that a Stage A job calls are
written** (dead-code rule, §11); Stage B modules are listed at the end and not created. Three reconciliations with the
spec tables (organisation only, no mechanism change):
- EXP-3 §6 `flexitors/diffusion/reflected_torus.py` → `flexitors/sampling/reflected_torus.py`, and
  `flexitors/postprocess/stereo_fix.py` → `flexitors/sampling/stereo_fix.py` (PLAN's package list has no
  `diffusion/` or `postprocess/`).
- EXP-5 §8 `flexitors/sampling/budget.py` and EVAL §10 `flexitors/eval/budget.py` and EXP-1 §5.1
  `flexitors/eval/timing.py` are **one** module, `flexitors/eval/budget.py` (EVAL §10 says "shared with EXP-5").
- EXP-1 §5.1 `flexitors/sampling/schedules.py` is folded into the two samplers (TD's schedule is one line,
  `sampling.py:125-126`; the bridge schedule is `linspace`). EXP-1 §5.1 `flexitors/eval/leakfree.py` ("wrappers only,
  with no new analysis code") is not created: the only Stage A change is `--ref_subsample` in
  `tools/analyze_leakfree.py`, which EXP-1 §3.6 assigns to that tool.

All code is Python 3.9-compatible (`from __future__ import annotations` for `X | Y` hints), typed, docstrings, and
every mechanism carries its source comment (`# EXP-3 §3.2 K2 / E-CHEM-048`, `# TD Alg. 3, PDF p. 17`, …).

```
flexitors/
  __init__.py              version; require_td_runtime(): REPO on sys.path, CWD has .p.npy/.score.npy (else exit 2)
  _td.py                   the ONLY place that imports TD modules (get_model, featurize_*, get_seed, embed_seeds,
                           perturb_seeds, modify_conformer, get_torsion_angles, torus, pyg_to_mol, InferenceDataset)
  data/
    provenance.py          stamp / check_or_raise (JSON), sha256_file, code_version()
    valset.py              build_qm9_valset(...)                                    EVAL §1      (PENDING Q-P3-007)
    subsets.py             write_subset_csv, rigid_stratum, ring4_population, affected_molecules
    splice.py              splice_confs(base, arm, keys) + verify_unaffected_identical(...)
  geometry/
    torsion_types.py       td_quadruples, classify_torsions, double_bond_targets          EXP-3 §3.1
    jacobian.py            dihedral_grad_rows, rigid_rows (batched torch)                  EXP-1 §2.2, §5.1
    projector.py           BatchedProjector (P_W = I − Cᵀ(CCᵀ)⁺C, never dense 3n×3n)         EXP-1 §2.2 (rev. D-004)
    internal.py            bond lengths, angles, dihedrals, signed volumes, stereocentre neighbour order
    constraints.py         ConstraintLayer (bond window, clash, chirality backtrack) + counters EXP-1 §2.6, §2.8
  models/
    heads.py               VectorL1oHead                                                    EXP-1 §2.4
    flexitors_score.py     FlexiTorsScoreModel(TensorProductScoreModel)                      EXP-1 §5.1
  training/
    transforms.py          PairedBridgeTransform (V2 bridge, M1)                             EXP-1 §2.2 steps 1–5
    losses.py              td_loss (verbatim), projected_l_loss                              EXP-1 §2.2 step 6
    train.py               CLI loop; TD resume/requeue helpers reused
  sampling/
    td_backend.py          TDSampler (seeding, embedding, partial resume, budget.json)       EXP-5 §8
    constrained_sampler.py reverse_sde(...) = sampling.py:176-246 + hooks                    EXP-3 §6, EXP-5 §8
    stereo_guard.py        StereoGuard hook (log / freeze / fold)                            EXP-3 §3.3
    reflected_torus.py     fold_half_circle (Stage A: K2-inf only)                           EXP-3 §3.2
    stereo_fix.py          f_flip, f_filter (+ Step-0 snap)                                   EXP-3 §3.4, §5
    select.py              select_2k, fps_torsion, mmff_single_points                         EXP-5 §4.4, §8
    joint_sampler.py       GeometrySampler (M1 read-out: L steps, τ frozen)                   EXP-1 §2.2 "Sampling"
  eval/
    metrics.py  stats.py  strata.py  report.py  budget.py                                      EVAL §10
    stereo_eval.py  rings_eval.py  geometry_eval.py  validity.py  energy.py                   EXP-1 §3, §5.1; EXP-3 §6
  tests/                   one test file per module (§10)
experiments/
  common/  fx_common.sh  submit_stageA.sh  status.py  smoke_stageA.sbatch  cpu_stageA.sbatch  infer_array.sbatch
  exp1/    config_m1.yml   run.sbatch (MODE=smoke|train|eval)   README.md
  exp3/    config.yml      arms.tsv   README.md   analysis.py (pre-declared)
  exp5/    config.yml      arms.tsv   README.md   analysis.py (pre-declared)
results/ on gnode118: $PROJECT/results/r5/<exp>/<model>/<tag>/   (synced to cluster_sync/round5/)
```

### 2.1 Module APIs (Stage A)

**`flexitors/_td.py`.** `import_td() -> types.SimpleNamespace`. Inserts `$REPO` on `sys.path`, asserts
`os.path.exists('.p.npy')` in the CWD before `import diffusion.torus` (else `SystemExit(2, "run from $REPO")`), and
returns the TD symbols. Rationale: `torus.py:26-34` would otherwise silently recompute 2 × 200 MB tables per process in
a new CWD, and concurrent first imports race (`notes/code_concerns.md:64`).

**`flexitors/data/provenance.py`.**
- `code_version() -> dict`: flexitors commit (from `flexitors/.commit`, written by the sync script), TD checkout
  `git rev-parse HEAD`, sha256 of `flexitors/**/*.py` (manifest).
- `stamp(out_dir: Path, *, inputs: dict[str, Path], params: dict) -> dict`: writes `provenance.json` (inputs with
  sha256, params, code version, PACK, rng_mode, host, SLURM ids).
- `check_or_raise(out_dir, expected: dict) -> None`: same rules as `slurm/common.sh:107-122` — existing provenance
  must equal expected (except host/SLURM ids); outputs without provenance → `StaleOutputError`.

**`flexitors/data/valset.py`** (EVAL §1; PENDING Q-P3-007). `build_qm9_valset(raw_dir: Path, split_path: Path,
n: int = 1000, seed: int = 0, out_dir: Path) -> dict[str, Path]` → `val_smiles.csv` (`smiles, n_conformers,
corrected_smiles`), `val_mols.pkl` (`{smiles: [Mol]}` cleaned by `evaluate_confs.py:47-54`'s rule), `val_list.txt`,
`population.txt`. Uses TD's sorted-raw-pickle split indexing (`torsional-diffusion/utils/dataset.py:165-180`).

**`flexitors/data/subsets.py`.** `write_subset_csv(csv: Path, keys: Sequence[str], out: Path)`;
`rigid_stratum(csv) -> list[str]` (TD `edge_mask.sum() == 0`, EXP-1 §3 row 10); `ring4_population(csv) -> list[str]`
(≥ 1 ring of ≥ 4 atoms, EXP-1 §2.2 σ_b paragraph, D-002); `affected_molecules(csv, classes: set[TorsionClass]) ->
list[str]`.

**`flexitors/data/splice.py`.** `splice_confs(base: Path, arm: Path, keys: Sequence[str], out: Path) -> None` (arm
values replace base values for `keys`; asserts `set(arm) ⊆ keys`, failures in `keys` are carried over as failures);
`verify_unaffected_identical(base: Path, rerun: Path, keys: Sequence[str], atol: float) -> None` (the smoke check
that an arm run on unaffected molecules reproduces O).

**`flexitors/geometry/torsion_types.py`** (EXP-3 §3.1).
- `td_quadruples(mol: Chem.Mol, data) -> torch.LongTensor[m, 4]`: one (c, a, b, d) per TD-rotatable directed edge in
  `data.edge_mask` order, with TD's neighbour rule of `torsional-diffusion/utils/standardization.py:69-90`
  (`n0[0], e0, e1, n1[0]`), mapped onto the PyG edge direction. Used by both Exp 1's W rows (EXP-1 §2.2 "TD's quadruple
  rule") and Exp 3's guard, so the dihedral that is preserved and the dihedral that is constrained are the same
  definition. A test compares it with the PG rule of `sampling.py:135-141` and records the mismatch rate.
- `classify_torsions(mol, data) -> list[str]` over `{single_sp3, single_conjugated, single_amide_like,
  double_stereo_specified, double_stereo_unspecified, double_nonstereo, triple}`; stores both the RDKit
  `FindPotentialStereo` flag and the structural test; constrains by the structural test (EXP-3 §3.1). Ported (not
  copied) from `round5/ledger/chem_data/stereo_census.py`.
- `double_bond_targets(mol, quads, classes) -> tuple[np.ndarray, list[str]]`: φ_b ∈ {0, π} and its source
  (`smiles` / `seed3d` / `none`), per EXP-3 §3.1 (reference-atom swaps toggle cis↔trans).

**`flexitors/geometry/jacobian.py`** (EXP-1 §2.2 rev. D-004, §5.1).
- `dihedral_grad_rows(pos: Tensor[N,3], quads: LongTensor[M,4]) -> Tensor[M,4,3]`: analytic Wilson-B rows (each row
  touches 4 atoms; stored compactly, never as M × 3N).
- `rigid_rows(pos: Tensor[N,3], batch: LongTensor[N]) -> Tensor[N,6,3]`: 3 translations + 3 rotations about each
  graph's centroid (TD Eq. 10, PDF p. 7 [E-JOINT-044]).

**`flexitors/geometry/projector.py`.** `BatchedProjector(pos, batch, quads, quad_batch, rcond=1e-8)`:
builds per-graph Gram blocks G_g = C_g C_gᵀ of size (m_g + 6), pads to max(m_g) + 6 (≤ 16 on QM9), solves batched with
`torch.linalg.eigh` + rank threshold (pseudo-inverse); `apply(v: Tensor[N,3]) -> Tensor[N,3]` computes
v − Cᵀ G⁺ C v by scatter; `rank() -> LongTensor[B]`; `rank_deficient() -> BoolTensor[B]` (counted, never silent). No
per-molecule Python loop (D-006).

**`flexitors/geometry/constraints.py`** (EXP-1 §2.6, §2.8). `ConstraintLayer(bond_tol=0.15, clash_frac=0.75,
max_halvings=3, chirality=True)`; `step(x, dx, ctx) -> tuple[Tensor, StepCounters]` with counters `backtracks`,
`skips`, `chirality_flips_pre_guard`, `bond_violations`, `clashes` per molecule (all reported, EXP-1 §3 row 9).

**`flexitors/models/heads.py`, `flexitors_score.py`** (EXP-1 §2.4, §5.1).
- `VectorL1oHead(irreps_in: o3.Irreps)`: `o3.Linear(<the nv×1o channels of irreps_in>, '1x1o')` per atom → Tensor[N,3]
  (`1o`, odd under x → −x; EXP-1 §2.4 table, [E-JOINT-021]).
- `FlexiTorsScoreModel(TensorProductScoreModel)`: `__init__(*, geometry_head: bool, **td_kwargs)`; the bridge clock t
  enters through the existing `lambda_embed_dim` slot (`score_model.py:65-79,194-199`), so `data.node_lambda = t`.
  The 1o node features are captured with a forward hook on `self.conv_layers[-1]` — TD's `forward`
  (`score_model.py:134-165`) is **not** copied, so with `geometry_head=False` the model is TD's class unchanged
  (golden test G3). Writes `data.l_pred`.

**`flexitors/training/transforms.py`** (EXP-1 §2.2 steps 1–5). `PairedBridgeTransform(sigma_b: float,
eps_t: float = 1e-3 (PENDING Q-P3-002), sigma_min, sigma_max)`; `__call__(data) -> data` with `pos = x_t`,
`edge_rotate` (TD target), `node_sigma`, `node_lambda = t`, `l_target = u`, `l_mask` (0 for `pair_ok = False`, PENDING
Q-P3-001). Same Δτ applied to both endpoints with TD's `modify_conformer` (`utils/torsion.py:57-75`); the RNG draw order
for σ and Δτ is TD's (`utils/dataset.py:71-76`), the conformer choice is `_select_paired_L`'s (`:79-99`), and
`interp_x_torch` (`:18-34`) provides `interp`.

**`flexitors/training/losses.py`.** `td_loss(edge_pred, edge_rotate, edge_sigma) -> Tensor` (verbatim
`utils/training.py:25-31`); `projected_l_loss(l_pred, l_target, proj: BatchedProjector, l_mask, dof) -> Tensor` (EXP-1
§2.2 step 6, normalised per molecule by 3n − 6 − m, PENDING Q-P3-002).

**`flexitors/training/train.py`.** CLI mirroring `torsional-diffusion/train.py` flags plus `--tau_weight`,
`--l_weight`, `--sigma_b`, `--eps_t`, `--geometry_head`, `--log_timing`, `--fail_on_nan`, `--limit_train_iters`,
`--resume`. Imports `save_checkpoint` / `load_resume_state` from TD's `train.py:30-62` (atomic `last_model.pt`, RNG
states) and installs the same SIGUSR1 → finish epoch → exit 99 handler (`train.py:107-110,146-149`). Logs per epoch:
`TIMING` (as `train.py:80-84`), both losses, loss by t-decile, masked share, projector rank deficiencies.

**`flexitors/sampling/td_backend.py`** (EXP-5 §8).
`TDSampler(model_dir: Path, ckpt: str = 'best_model.pt', steps: int = 20, batch_size: int = 128,
rng_mode: Literal['td_stream', 'per_molecule'] = 'per_molecule', global_seed: int = 0)`. Loads the yml exactly as
`generate_confs.py:85-105` (including the ablation-hooks override of `likelihood`/`seed`/`l_level`).
- `draw(row: TestRow, n: int, batch_index: int = 0, hooks: Hooks | None = None, pg: PGParams | None = None) ->
  list[Chem.Mol] | None` (None = failure, as `sample_confs` returns None, `generate_confs.py:134-148`);
- `run_csv(csv: Path, out: Path, *, n_rule: Literal['2K', 'cK'], c: int = 1, keys: Sequence[str] | None = None,
  hooks_factory=None, pg=None, checkpoint_every: int = 50) -> None`: writes `confs.pkl` in `generate_confs.py:208-213`
  format (`{corrected_smiles: [Mol]}`), `budget.json` (NFE, CPU s, GPU wall s per molecule; EVAL §8), and
  `confs.partial.pkl` + `done.txt` every 50 molecules (atomic `os.replace`), from which a restarted task resumes.
- `rng_mode='td_stream'` seeds once per run as `generate_confs.py:94-95` and exists only for golden test G1;
  `per_molecule` is the production mode (§3.2).

**`flexitors/sampling/constrained_sampler.py`.** `reverse_sde(conformers, model, *, sigma_max, sigma_min, steps,
batch_size, hooks: Hooks | None, pg: PGParams | None) -> list[Data]`: a line-for-line port of
`sampling.py:122-246` (SDE branch, PG branch; ODE/likelihood dropped — no Stage A arm uses them) with two hook points:
`hooks.before_loop(data)` (after `perturb_seeds`, e.g. K1 seed check, K2 fold into I) and
`hooks.after_step(sigma_idx, data)` (after `apply_torsion_and_update_pos`, `sampling.py:234`: guard, fold).
`Hooks.edge_mask_edit(data)` implements K1/K1-canonical/K1-wrong freezing (remove constrained bonds from `edge_mask`
and the matching `mask_rotate` rows before `perturb_seeds`, `sampling.py:105-108`).

**`flexitors/sampling/stereo_guard.py`** (EXP-3 §3.3). `StereoGuard(quads_c, phi, mode: Literal['log', 'freeze',
'fold'])`; `after_step(step, data) -> None` computes δ_b = wrap(τ_b − φ_b) for all constrained bonds of the batch with
`get_torsion_angles` (`utils/torsion.py:110-124`), counts |δ_b| > π/2 (E/Z) and |δ_b| > 20° in the last 3 steps
(planarity), per step and molecule; `fold` mode applies `fold_half_circle` and rotates the moving side by the change
(`modify_conformer`).

**`flexitors/sampling/reflected_torus.py`.** `fold_half_circle(delta: np.ndarray) -> np.ndarray` (wrap to (−π, π],
then x > π/2 ↦ π − x, x < −π/2 ↦ −π − x; EXP-3 §3.2, `# Lou & Ermon 2023, PDF p. 1` as the spec asks). Stage B adds
`p_reflected`, `score_reflected`, `score_norm_reflected` for K2-train.

**`flexitors/sampling/stereo_fix.py`** (EXP-3 §3.4, §5 Step 0). `f_flip(mol, quads, phi, *, snap: bool = True) ->
Chem.Mol`; `snap_all(mol, quads, phi) -> Chem.Mol` (Step 0); `f_filter(sampler, row, mols, target, max_rounds=3) ->
tuple[list[Mol], int]` (returns the extra NFE; replacement batches use `batch_index = 1, 2, 3`).

**`flexitors/sampling/select.py`** (EXP-5 §4.4, §8). `select_2k(pool: list[Mol], k2: int, method: str, *, q: float,
energies: np.ndarray | None, rng: np.random.Generator) -> np.ndarray` (indices; |out| = k2 exactly, subset of pool);
`fps_torsion(tors: np.ndarray, k, start)`; `mmff_single_points(mols) -> tuple[np.ndarray, float]` (energies, CPU s).
Ported from `round5/ledger/sample_data/select.py` (prototype) with the test of EXP-5 §9.

**`flexitors/sampling/joint_sampler.py`** (EXP-1 §2.2 "Sampling"). `GeometrySampler(model, steps=20,
constraints: ConstraintLayer, tau_mode='frozen')` (PENDING Q-P3-003 item 2); `run(conformers) -> (list[Data],
counters)`: x ← x + Δt · P_W(x) v_θ(x, t) for t = 0 … 1 − Δt, then `constraints.step`; per-step timers for P_W and the
constraint layer (EXP-1 §3 row 9). Runs for m = 0 molecules too (PENDING Q-P3-003 item 3).

**`flexitors/eval/*`** (EVAL §10; EXP-1 §5.1).
- `metrics.per_molecule(eval_pkl, thresholds) -> pd.DataFrame` (TD convention and intersection aggregates;
  `evaluate_confs.py:38-44,150-187`).
- `stats.paired_bootstrap`, `holm`, `hierarchical` (numerics ported from `tools/analyze_round2.py:123-159`).
- `strata.bins(csv)` (EVAL §4; `tools/breakdown.py` bins).
- `budget.NFECounter`, `budget.assert_matched(arms, tol=0.01)`, `budget.gpu_hours_ledger(sacct_rows)`.
- `report.table(...)` (universe, failures per seed, both conventions, CIs, seed SD, NFE columns; EVAL §0 rule 4).
- `stereo_eval.labels(mol) -> (db, ch)` (ported from `round5/ledger/chem_data/ez_flip_analysis.py`), `ez_fidelity`,
  `planarity`.
- `rings_eval.ring_dihedral_error(gen, ref)` and share > 10° (EXP-1 §3 row 4, via `tools/lgeom.py`).
- `geometry_eval` (wraps `tools/geometry_metrics.py`), `validity.check(mol, ref_graph)` (sanitise; `rdDetermineBonds`
  connectivity equals the input graph; clashes; EXP-1 §3 row 5).
- `energy.XTBRunner` (§7.3).

**Stage B modules — not created in Stage A:** `geometry/rings.py`, `geometry/reconstruct.py` (V3b),
`models/stereo_edges.py` (K3), `training/dagger.py` (A5/A6b), `sampling/gain_model.py`, `sampling/controller.py`,
`sampling/signals.py` S1 ensemble disagreement, `eval/allocation_curves.py` (Track D dropped by DECISION),
`reflected_torus.score_reflected` (K2-train).

## 3. Shared runtime: provenance, seeding, failure detection, OOM retry, resume, budget ledger

### 3.1 Provenance (reused, extended)
Every output directory gets `provenance.json` (§2.1) before any work; the shell side keeps `R2_PROVENANCE=1` semantics
(`slurm/common.sh:107-122`). Extensions over round 2: flexitors code manifest hash, `rng_mode`, `global_seed`, the arm
config hash, the subset key list hash (for spliced arms), and the base O run's provenance hash (a spliced arm is valid
only against the O run it was spliced into).

### 3.2 Per-molecule seeding (new; the main throughput lever)
- Seed for molecule `smi`, batch `b`: `h = int(sha256(f"{global_seed}|{smi}|{b}").hexdigest()[:15], 16)`;
  `random.seed(h)`, `np.random.seed(h % 2**32)`, `torch.manual_seed(h)`; ETKDG `randomSeed = 1 + h % (2**31 − 2)`
  (never 0: RDKit seed 0 makes every conformer identical, `generate_confs.py:66`).
- Keyed by **SMILES, not row index**, so a subset csv (affected molecules, 20-molecule smoke, DRUGS 141) reproduces
  the full run molecule for molecule. (EXP-5 §8 says "hash(global seed, molecule index, batch index)"; the SMILES key is
  the same idea made subset-invariant.)
- Consequences used in §5–6: (i) arms differ from O only on molecules where a hook acts; (ii) Exp 5's extra batch
  `b = 1` is independent of O's `b = 0`; (iii) resume after a crash is exact.
- Cost: O must be regenerated with flexitors (the round-1 R0–R2 used one stream per run). This is ≈ 0.5 GPU-h for 3
  seeds and buys the splicing savings (≈ 3 GPU-h on QM9, far more on DRUGS).
- Caveat (INFERENCE): GPU scatter/e3nn reductions are not bitwise deterministic (`tools/tests/golden_dump.py:21-23`
  measured ~2e-6 on CPU threads). Spliced arms therefore **copy** O's conformers for unaffected molecules instead of
  regenerating them, and the "identical on unaffected molecules" claim is tested bitwise on CPU (1 thread) and within
  1e-3 Å on GPU (test S-E3c, §10).

### 3.3 Failure detection (R2_STRICT, extended)
Each array task ends with `fx_strict_check <out_dir>`: `confs.pkl`, `eval.pkl`, a `SUMMARY` line, `budget.json`,
`provenance.json`, and (generation) `done.txt` covering every csv row; the number of model failures must equal O's for
spliced arms (the same molecules fail embedding, because the ETKDG seed is per molecule). Any miss → `FX_FAIL <tag>
<reason>` and exit 3 (same contract as `ablation_train_array.sbatch:139-146`). Silent fallbacks are counted and printed:
guard backtracks/skips, projector rank deficiencies, xtb non-convergence, MMFF setup failures (as round-2 X11,
`sampling.py:21-37`).

### 3.4 OOM retry (new)
`with_oom_retry <log> <max=3> -- cmd ...` in `experiments/common/fx_common.sh`:
- run; on non-zero exit, grep the log for `CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory`;
- OOM → print `FX_OOM_RETRY n` with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` (who holds the GPU),
  sleep 300 · 2^(n−1) s, rerun the **same** command (same seeds, same batch size: changing the batch size would change
  the RNG-to-conformer mapping); generation resumes from `confs.partial.pkl`, training from `last_model.pt`
  (`--resume`);
- any other error → fail fast (no retry), exit code preserved.
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` (supported by torch 1.13) to limit fragmentation (INFERENCE: helps
  when a co-located job fragments the pool; harmless otherwise).
- GPU pre-flight: log `nvidia-smi` free memory of the assigned device; if < 6 GB free, wait up to 30 min (poll every
  60 s) before starting, then proceed and rely on the retry.
Expected effect: the two round-2 OOM incidents would each have cost one retry inside the job instead of a manual
resubmission and an agent round trip.

### 3.5 Resume and requeue
- Training: `experiments/exp1/run.sbatch MODE=train` copies the round-2 logic of `ablation_train_array.sbatch:68-122`
  (resume only if the saved yml matches the arm; `--signal=B:USR1@900`, rc 99 → `scontrol requeue`; `.train_done`
  marker; epoch check before evaluation). M1 runs are ≈ 4–6 h, far below the 4-day limit, so requeue is a safety net.
- Generation: `done.txt` + `confs.partial.pkl` every 50 molecules; a rerun with matching provenance skips done molecules.
  QM9 runs are short (≈ 15 min) but DRUGS subsets are ≈ 0.5 h and the released-sample scoring is many CPU hours.
- Evaluation: `evaluate_confs.py` is unchanged and not resumable; DRUGS evaluation is therefore sharded by molecule
  (one `--filter_mols`-style subset csv per shard, `evaluate_confs.py:17,69-76`), and shards are merged by
  `flexitors/eval/metrics.merge_shards` (pure concatenation of per-molecule matrices; test: merged = unsharded on 20
  molecules).

### 3.6 GPU-h ledger and status (new; protects the budget and the agents' rate limit)
- `experiments/common/status.py` reads `sacct -X -S <stageA_start> --name=fx_% --format=JobID,JobName,State,Elapsed,
  AllocTRES,ExitCode` and every `provenance.json`/`budget.json`, and writes one `STATUS.md`: job states, GPU-h used per
  experiment against the cap, each tag's status, `FX_FAIL`/`FX_OOM_RETRY` lines with log paths, pending gates.
- `submit_stageA.sh` refuses a submission whose estimate would push the ledger past 25 GPU-h (`FX_BUDGET_STOP`); the
  estimate per job comes from `experiments/*/config*.yml` and, after the smoke, from measured rates.
- The orchestrator syncs one file per check (`STATUS.md`, a few KB) instead of reading logs: this is the main saving of
  agent calls during the run.

### 3.7 Fail-fast pre-flight (every job, before any GPU work)
`fx_preflight` checks, in order, and exits 2 with `FX_PREFLIGHT <reason>`: venv versions (torch 1.13.1, RDKit
2022.09.5, e3nn 0.5.1; EXP-3 §3.1 pins RDKit); `$REPO/.p.npy` and CWD = `$REPO`; inputs exist and match the hashes in
the arm config (released `qm9_default/best_model.pt` 6,388,281 B, `slurm/download_data.sh` line for it; DRUGS
checkpoint 6,392,121 B, §8); free disk on `/scratch` ≥ 2 × expected output; `cuda_check` (`slurm/common.sh:75-85`);
xtb `--version` = 6.7.1 when the job uses xtb; the output dir's provenance; and `nlines` of every TSV equals the array
size (the round-1 "keep in sync" comment, `ablation_train_array.sbatch:15-16`, made a check).

## 4. Exp 1 Stage A (M0, smoke, M1 = Gate 0)

Stage A for Exp 1 is EXP-1 §5.2 up to and including M1 (A11), nothing of M2. Spec mechanisms and where they land:

| Mechanism (spec, evidence) | Code | TD touchpoints reused (read this session) |
|---|---|---|
| W = dihedral-gradient rows, TD quadruple per rotatable bond (EXP-1 §2.2 rev. D-004, [E-JOINT-032]) | `geometry/jacobian.py::dihedral_grad_rows`, `geometry/torsion_types.py::td_quadruples` | `utils/standardization.py:69-90` (quadruple rule), `utils/torsion.py:9-39` (edge_mask / mask_rotate) |
| J_rig rows (EXP-1 §2.2; TD Eq. 10, PDF p. 7 [E-JOINT-044]) | `geometry/jacobian.py::rigid_rows` | `diffusion/likelihood.py:59-65` (inertia), `:84-93` (rigid removal pattern) |
| P_W = I − Cᵀ(CCᵀ)⁺C, batched, rank check (EXP-1 §2.2, §5.1; D-006 "no per-molecule loop") | `geometry/projector.py::BatchedProjector` | — (numpy `likelihood.py:77-98` loops per molecule; not reused) |
| Paired endpoints, same Δτ on both (EXP-1 §2.2 steps 1–3) | `training/transforms.py::PairedBridgeTransform` | `utils/dataset.py:79-99` (`_select_paired_L`), `:71-76` (σ, Δτ draw), `utils/torsion.py:57-75` (`modify_conformer`) |
| Bridge x_t = interp + γ(t) P_L(x_R') z, γ = σ_b√(t(1−t)) (EXP-1 §2.2 step 4, ET-Flow Eq. 2–5 [E-JOINT-076]) | same | `utils/dataset.py:18-34` (`interp_x_torch`) |
| Target u and projected loss / (3n − 6 − m) (EXP-1 §2.2 steps 5–6) | `training/losses.py::projected_l_loss` | `utils/training.py:25-31` (TD loss kept verbatim) |
| `1o` per-atom L head (EXP-1 §2.4, [E-JOINT-021]) | `models/heads.py::VectorL1oHead` | `diffusion/score_model.py:83-89` (irreps), `:141-143` (last conv layer → hook) |
| Second clock through the λ slot (EXP-1 §5.1 `flexitors_score.py` row) | `models/flexitors_score.py` | `score_model.py:65-79,192-199` (round-2 S4 λ embedding) |
| Constraint layer: bond window ±15 %, clash 0.75 × Σr_cov, halve ≤ 3× then skip; chirality backtrack (EXP-1 §2.6, §2.8, [E-JOINT-032]) | `geometry/constraints.py` | — |
| Geometry Euler sampling, P_W(x) v (EXP-1 §2.2 "Sampling") | `sampling/joint_sampler.py` | `diffusion/sampling.py:40-102` (seed, embed) via `TDSampler` |

### 4.1 M0 (CPU, EXP-1 §5.2 M0; ORACLE, analysis only)
- Job `cpu_stageA.sbatch TASK=m0` (8 CPUs, 40000M, ≤ 2 h). Script `experiments/exp1/m0_oracle_step.py` (an analysis
  script, not a package module).
- Inputs: test-set matched pairs from `$QM9_SEEDS2/L_lam0.00_cyc_ORACLE.pkl` (matched RDKit seed) and
  `L_lam1.00_cyc_ORACLE.pkl` (own GT), built in the same loop of `tools/make_l_seed_pickles.py:152-182` (Hungarian
  matching of seeds to GT); pairing is re-verified on load (same labelled graph; `AlignMol` equals the stored
  `l_error.csv` row). If the check fails for a molecule, it is dropped and counted.
- Per pair: (a) dihedral drift of TD's quadruple per rotatable bond after x_R + s·P(x_G − x_R) for P ∈ {P_W, P_L (the
  superseded projector, kept only here as the negative control V5 measured at 0.19–0.25 rad/Å)} and s ∈ {0.01, 0.02,
  0.05 Å RMS steps, full step}; (b) captured fraction ‖P_W(x_G − x_R)‖²/‖x_G − x_R‖² and heavy-atom RMSD to GT after
  the oracle step; (c) chirality flips (signed volumes, §2.8 definition).
- Kill criterion (EXP-1 §5.2): < 50 % recovery of the RDKit → own-GT RMSD gap. PENDING Q-P3-003 item 5: applied to the
  m > 0 ≥ 4-ring population, because on the rigid stratum P_W only removes rigid motion and recovery is ≈ 100 % by
  construction (INFERENCE); rigid rows are reported as the sanity check. **M0 does not gate the smoke run** (they run
  in parallel; the smoke run is cheap); a FAIL blocks M1 submission (`FX_GATE_M0_FAIL`).
- Output `results/r5/exp1/m0/{m0_rows.csv, m0_summary.txt}`.

### 4.2 Data for M1
- Training data: `cache_paired.{train,val}` (round-2 job 10122; `slurm/common.sh:45-46`), PENDING Q-P3-001 (reuse; geometry
  loss masked for `pair_ok = False`). The stale-cache guard of `utils/dataset.py:141-145` (gt_pos in every
  datapoint) is reused by setting `needs_gt = True` on the new transform.
- Validation molecules for σ_b selection: the QM9 validation subset (§7.4, PENDING Q-P3-007), rigid stratum and
  ≥ 4-ring population of it; test rigid stratum (289 molecules, `breakdown.log` bin 0, EXP-1 §3 row 10 [E-JOINT-007])
  and test ≥ 4-ring population for the single GO read-out.

### 4.3 Smoke run (EXP-1 §5.2: "Before the Gate-0 ask, a 1-epoch smoke run (≈ 0.12–0.18 GPU-h) measures the real
overhead")
Inside `smoke_stageA.sbatch` (§9), after the 20-molecule checks:
1. **1 epoch, full training set**, σ_b = 0.15 Å, seed 0, `--log_timing --fail_on_nan`. Records epoch time T₁ against
   TD's 377.7 s/epoch (`round3/grounding.md:56-58`): overhead f = T₁ / 377.7. Also records loss per t-decile (finite,
   bounded; PENDING Q-P3-002), the masked share (Q-P3-001), projector rank deficiencies (must be 0 on valid molecules),
   and peak RSS / GPU memory.
2. **Packing test, 300 iterations** (`--limit_train_iters 300`, the round-2 D5 hook, `utils/parsing.py:37`): 1 process
   vs 3 concurrent processes on the same GPU; throughput ratio s = 3·it/s(packed) / it/s(single).
3. **Projection** written to `smoke/m1_projection.json`: M1 GPU-h = 3 × 30 × T₁(1 + val share) / 3600 if unpacked;
   / s if packed. Rule (mechanical): pack the three σ_b runs on one GPU iff s ≥ 1.8 **and** packed peak GPU memory
   ≤ 12 GB (headroom for co-located jobs); otherwise one GPU each. If projected M1 > 16 GPU-h (EXP-1 §5.2 ceiling),
   `run.sbatch MODE=train` exits `FX_BUDGET_STOP` before training and the ask is restated to the user from the measured
   number (as the spec requires).
INFERENCE on packing: TD training runs 6.86 it/s with batch 32 (`round3/grounding.md:56-57`), small kernels, a
CPU-side `torus.score` call per iteration (`utils/training.py:26-31`), so GPU utilisation is probably well below 100 %
and packing may give s ≈ 2–2.5. Unmeasured; the rule above decides.

### 4.4 M1 = Gate 0 (A11): 3 σ_b × 30 epochs × 1 seed (EXP-1 §5.2, §4 A11)
- Config `experiments/exp1/config_m1.yml`: σ_b ∈ {0.05, 0.15, 0.3} Å (EXP-1 §2.2), `--tau_weight 0 --l_weight 1`, τ
  noise applied (EXP-1 §5.2), 30 epochs, seed 0, QM9 recipe of `ablation_train_array.sbatch:103-107` (4 layers, ns 32,
  nv 8, σ ∈ [0.0314, 3.14], lr 1e-3, batch 32), `--lambda_embed_dim 32` for t (PENDING Q-P3-009: whole network from
  scratch), loader workers 3, 6 CPUs / 30000M per run (round-2 D5 / FIXES X2).
- Checkpoint: `best_model.pt` by validation geometry loss (EXP-1 §4 control 8: no test selection).
- Read-out (`run.sbatch MODE=eval`, 1 GPU, PACK=3, 8 CPUs):
  1. **Validation** (choose σ_b): for each σ_b, `GeometrySampler` (N = 20, τ frozen, PENDING Q-P3-002/003) on the
     validation rigid stratum and validation ≥ 4-ring population; 2K samples per molecule; the same per-molecule
     ETKDG seeds for all three σ_b and for the ETKDG/MMFF reference rows (§3.2). Metrics: rigid AMR-R
     (`evaluate_confs.py` unchanged), ring-dihedral share > 10° (`rings_eval`), drift per step on TD's quadruple,
     chirality flips before guard and backtracks (EXP-1 §3 rows 6, 9), validity. Pre-declared choice: lowest
     validation rigid AMR-R; tie (|Δ| < 0.002 Å) → lower ring share.
  2. **Test, once, chosen σ_b only**: test rigid stratum (289) and ≥ 4-ring population; GO iff rigid AMR-R ≤ 0.10 Å
     and ring share > 10° ≤ 20 % (EXP-1 §5.2). `experiments/exp1/gate0.py` prints `FX_GATE0 GO|NO-GO` with both numbers
     and the ETKDG (0.152) / MMFF (0.128) references from the same seeds.
  3. Energy (EXP-1 §3 row 7) on the chosen σ_b's test rigid outputs: xtb single point + `--opt normal` per conformer
     (§7.3), E_relax and RMSD moved; CPU only, ≈ 289 × ~8 × 0.3 s ≈ 0.2 CPU-h (`round3/grounding.md:36-37` cost basis).
- Expected GPU: 10.5 GPU-h at TD speed, ≤ 16 (ceiling), ≈ 5–7 if packing passes (INFERENCE); read-out ≈ 0.4 GPU-h.

### 4.5 What Exp 1 Stage A deliberately does not build
A4/A5/A6/A6b/A7/A8 (M2), DAgger, V3a/V3b, CP rings, NeRF/SLSQP — Stage B, after Gate 0. No code for them is written
now (§11).

## 5. Exp 3 Stage A (Step 0, QM9 and DRUGS inference arms)

Mechanisms (EXP-3 §3) and code:

| Mechanism (spec, evidence) | Code | TD touchpoints |
|---|---|---|
| TD rotatable rule replicated; torsion classes; structural stereo test (EXP-3 §3.1; E-CHEM-020, -070) | `geometry/torsion_types.py` | `utils/torsion.py:9-39`; `utils/standardization.py:69-90` |
| φ_b target from SMILES stereo, else seed 3D (EXP-3 §3.1) | `torsion_types.double_bond_targets` | `diffusion/sampling.py:135-141` (reference-atom pattern), `utils/torsion.py:110-124` |
| K1 freeze (+ seed check: rotate by π if |δ_b| > π/2) (EXP-3 §3.2; VonMisesNet rule [E-CHEM-043]) | `constrained_sampler.Hooks.edge_mask_edit`, `before_loop` | `sampling.py:105-108` (perturb only masked edges), `generate_confs.py:150-151` |
| K1-wrong (opposite isomer), K1-canonical (all double + triple) (EXP-3 §5) | same hook, other target | same |
| K2-inf: TD score + fold into I = [−π/2, π/2] after each step (EXP-3 §3.2; Lou & Ermon [E-CHEM-048]) | `stereo_guard` mode `fold`, `reflected_torus.fold_half_circle` | `sampling.py:204` (SDE step), `:234` (apply torsion) |
| Per-step verification and counters (EXP-3 §3.3) | `sampling/stereo_guard.py` | `utils/torsion.py:110-124` |
| F-flip, F-filter with replacement (EXP-3 §3.4; [E-CHEM-046], [E-CHEM-064]) | `sampling/stereo_fix.py` | `evaluate_confs.py:47-54` (filter is non-isomeric, so E/Z errors are invisible to it [E-CHEM-028]) |
| E/Z fidelity, planarity, DB-only / CH-only labels (EXP-3 §4) | `eval/stereo_eval.py` | port of `round5/ledger/chem_data/ez_flip_analysis.py` |

Note on K1 (from reading the model): freezing removes the bond from `edge_mask`, so `build_bond_conv_graph`
(`score_model.py:167-181`) simply emits no score for it and `bond_conv` gets `out_nodes = edge_mask.sum()`
(`:154`); the atom graph (`build_conv_graph`, `:183-213`) is unchanged. `mask_rotate` rows are indexed by masked edges
(`utils/torsion.py:31-37`), so the matching row must be dropped together with the edge (unit test).

### 5.1 Step 0 (CPU, EXP-3 §5; ORACLE for unspecified targets)
`cpu_stageA.sbatch TASK=exp3_step0` (8 CPUs, ≤ 2 h): on the existing R0 outputs
(`$RES/qm9_default/R0_base_seed0/confs.pkl`), `stereo_fix.snap_all` sets every TD-rotatable double bond to its target
(SMILES target if specified; GT-derived target otherwise, labelled ORACLE), writes `confs_snapped.pkl`, and re-scores
with the unchanged evaluator. Rule (EXP-3 §5): if the all-molecule AMR-R ceiling is < 0.002 Å, QM9 retrained arms move
to DRUGS (Stage B anyway); QM9 inference arms run regardless (they are cheap with splicing).

### 5.2 QM9 inference arms (released `qm9_default`, 3 sampling seeds)
- **O** (TD, per-molecule seeding), seeds 0, 1, 2, full 1000 molecules (shared with Exp 5 P0).
- **F-flip** (CPU, on O), **F-filter** (on O; replacement draws only for molecules with wrong conformers; extra NFE in
  `budget.json`).
- **K1, K1-wrong, K2-inf** on the 102 molecules with a TD-rotatable double bond; **K1-canonical** on those plus the
  molecules with a TD-rotatable triple bond (`affected_molecules`, EXP-3 §2 table: 108 double, 123 triple bonds). Each
  run generates only affected molecules, then `splice_confs` into O of the same seed. Unaffected molecules are not
  regenerated.
- Evaluation: unchanged `evaluate_confs.py` on every spliced `confs.pkl` (CPU, in-job, PACK=3); `stereo_eval` on the
  stereo subsets of both RDKit versions (cluster 2022.09.5 perceived: 38 molecules; the 2026.03.6 list from
  `round5/ledger/chem_data/census_qm9_local.csv`, used as a key list only) (EXP-3 §4).
- Pre-declared analysis `experiments/exp3/analysis.py`, written before the runs: primary E/Z fidelity and planarity
  (K arms − O, F arms − O) on specified-stereo molecules; co-primary non-inferiority AMR-R margin +0.004 Å and COV-R
  −0.5 points (EXP-3 §4), Holm over the primary family, paired molecule bootstrap 5000, 3 seeds (EVAL §7); K1-wrong
  is the negative control (EXP-3 §5).

### 5.3 GEOM-DRUGS inference arms (released `drugs_default`; PENDING Q-P3-004)
- After D0 staging (§8): `infer_array.sbatch` 20-molecule DRUGS timing task (generation s/conformer, evaluation
  s/RMSD call, peak RSS of the evaluator). It writes `drugs_rates.json`; the arm list is then cut to stay under 4.5
  GPU-h (order of cuts: K1-canonical, then K1-wrong).
- Arms: O, K1, K1-wrong, K1-canonical, K2-inf on the 141 stereo-DB molecules (+19 non-stereo double / triple for
  K1-canonical), 1 sampling seed; F-flip / F-filter on O. Estimated 20,208 samples per arm (§1), ≈ 0.5–0.7 GPU-h per
  arm packed (INFERENCE: 0.8 × QM9's sample count × ≈ 3–4 × per-sample cost for 25.5 vs ≈ 9 heavy atoms and ≈ 4× the
  rotatable bonds).
- All-molecule non-inferiority by splicing argument (PENDING Q-P3-004 item 3); absolute level from scoring the
  released `drugs_steps20.pkl` (EVAL §9 D0, CPU, §8).
- DRUGS RMSD evaluation sharded by molecule (§3.5), 16 CPUs × 5000 MB, workers = 15; E/Z and planarity metrics first
  (minutes), RMSD later.

## 6. Exp 5 Stage A (Track P selection at matched NFE; D-101, D-102)

DECISION drops allocation (Track D, headroom −0.0042 Å per V5), so Stage A is Track P only: output exactly 2K_i per
molecule, chosen from c × 2K_i proposals (EXP-5 §4.4 Track P, §4.5), QM9 only.

| Arm | Proposals (NFE) | Output | Status |
|---|---|---|---|
| P0 TD | 2K_i (1×) | all | = Exp 3 O, shared (no extra GPU) |
| P1 TD + random | 4K_i (2×) = O's batch 0 + batch 1 | random 2K_i (rng seeded) | NFE control for P3 (D-102) |
| P2 TD + PG | 2K_i (1×) | all | PG at 1×, compared with P0 only (D-102 b) |
| P3 TD + selector | 4K_i (2×), same pool as P1 | selector 2K_i | main arm |
| P6r PG + random | 4K_i PG pool (2×) | random 2K_i | **NFE-matched PG control** (D-102 b), PENDING Q-P3-006 |
| P6 PG + selector | same pool as P6r | selector 2K_i | composition (EXP-5 §5 P6) |
| ORACLE selection | pool of P3 | GT-greedy | analysis row only (EVAL §0 rule 3) |

- **D-101 resolution (PENDING Q-P3-005 item 1).** With no allocation, the pool size is c × (the benchmark's output
  size) per molecule; K_i is used only for the output size, as EXP-5 §4.5 permits. The relabelling of 0.1450 /
  0.0042 Å asked by D-101 is a spec-text change (R-SAMPLE's); no Stage A code consumes those numbers. A K-free fixed pool
  max(c·2K_i, 128) is ready as one extra config line (≈ +0.4 GPU-h) if R-SAMPLE wants it.
- **D-102 resolution.** (a) The selector is re-chosen on the QM9 validation subset (EVAL §0 rule 3): pools of 4K_i on
  the 1000 validation molecules, 1 seed; candidates {random, torsion-FPS, lowest-MMFF, MMFF-filter q ∈ {0.25, 0.5} +
  FPS}; pre-declared criterion (PENDING Q-P3-005 item 2) applied by `experiments/exp5/choose_selector.py`, which writes
  `selector.json`; the test arms read only that file. (b) PG at 2× NFE is P6r / P6, with one batch holding all 4K_i
  conformers of a molecule because PG repels only within a loader batch (`sampling.py:207,229`; PENDING Q-P3-006 item 2),
  non-invariant README recall setting (`torsional-diffusion/README.md:86`; PENDING Q-P3-006 item 1).
- **Matching (EXP-5 §4.1, §5).** `budget.assert_matched` checks total NFE within 1 % for P1 vs P3 vs P6r vs P6
  (identical by construction: 20 steps × 4K_i); MMFF single-point CPU seconds and PG kernel time are reported as their
  own columns (EXP-5 §4.1), not converted into NFE.
- **Pre-declared analysis** (`experiments/exp5/analysis.py`, before any test run; PENDING Q-P3-005/006): primary family
  {P3 − P1, P3 − P6r} × {AMR-R, COV-R@0.5}, Holm; secondary P3 − P0 (labelled 2× vs 1× NFE), P2 − P0, P6 − P6r, AMR-P /
  COV-P beside every recall number (EVAL §2), COV-R@0.1 on the success intersection; 3 sampling seeds, paired molecule
  bootstrap, seed SD printed (EVAL §7). K3 (EXP-5 §11): if P3 does not beat P6r on recall **and** precision, the selector
  is reported as a baseline only.
- **GPU:** batch 1 for seeds 0–2 (3 runs), PG 2K (3 runs), PG 4K (3 × 2 run-equivalents), validation 4K (2) ≈ 14 QM9
  run-equivalents × 0.17 GPU-h ≈ 2.4 GPU-h (+ PG overhead, measured). Selection itself is CPU (MMFF single points on
  ≈ 100k conformers at a few ms each, INFERENCE).

## 7. CPU items (leak-free ref_half, floor-bug fix, xtb runner, validation set)

### 7.1 Budget-matched leak-free re-score (EXP-1 §3.6, D-001; ORACLE, analysis only)
- Change: `tools/analyze_leakfree.py` gains `--ref_subsample N` (default 0 = off; output byte-identical to now).
  With N > 0, for each family in {CR, B1, S3, S4} and seed, read the existing `steps20_seed0/eval.pkl` (the
  `ref_rdkitL` tag, `tools/analyze_leakfree.py:78`), and per molecule of the 744-molecule universe build `ref_half`:
  rows = target-half GT rows (alternating cleaned index, `tools/make_split_half.py` docstring), columns = a random
  2·|half| subset of the generated conformers, averaged over N = 10 draws (`default_rng(0)`). Report split − ref_half
  with the existing `boot` (`:130-137`) in a new `reference_half` family (descriptive; it is not in the primary Holm
  family), beside the existing `R.*` rows (`:318-347`).
- Only `eval.pkl` is needed (the RMSD matrices); `confs.pkl` is not read. Runtime: minutes on 4 CPUs.
- Tests: with the subsample equal to all columns, ref_half equals ref exactly (G10, §10); a hand-made 3 × 6 matrix gives
  the hypergeometric expectation within MC error.

### 7.2 Floor-bug fix (EXP-1 §9 item 1, [E-JOINT-031])
- Fix at `tools/local_structure_analysis.py:302`: candidates `[j for j in np.argsort(cost_gg[i]) if j != i]
  [:args.floor_topm]` (the self cost is `np.inf`, `:281`, so it sorts last and was included whenever n_GT ≤ 3).
- New option `--floor_pool {all_other, other_half}` with `--split_tsv data/QM9/round4_splithalf/split.tsv`
  (PENDING Q-P3-003 item 4): `other_half` restricts candidates to the other half's cleaned GT indices; also writes
  `floor_gtother_halfmin` (minimum over all other-half candidates). Default `all_other` = the fixed old behaviour.
- Re-run `--mode test` in full on gnode118 (`cpu_stageA.sbatch TASK=floor`, 16 CPUs × 4000 MB as
  `slurm/analysis_qm9.sbatch:6-7`): 7,554 GT-conformer rows (the current `local_structure_test.csv`) at ≈ 1.2 s per row
  per core (`slurm/analysis_qm9.sbatch:19`) ≈ 2.5 CPU-h of DE plus embedding; ≈ 0.5–1 h wall. A full re-run (rather than
  only the 452 molecules with n_GT ≤ 3) doubles as the regression test: all rows with n_GT > 3 must be byte-identical to
  the old CSV (G11), because the seeds are fixed (`--seed 1`, `:77`).
- Then re-emit the corrected floor table (replacing `round2/research_A.md:24-34`'s 0.0323 / 0.0137 / 0.0269, EXP-1 §9)
  into `results/r5/cpu/floor_fixed/`.

### 7.3 xtb 6.7.1 runner and the TD temp-dir race
- **Race (verified in code):** `torsional-diffusion/utils/xtb.py:5-7` fixes `my_dir = /tmp/<pid>` at import; forked
  `Pool` workers share it and read/write the same `xtb.xyz` / `xtbopt.xyz` (`:44-45`), so another molecule's optimised
  coordinates can be read back when atom counts match (`round3/grounding.md:80-84`, S-1.1). `xtb_energy` writes
  `/tmp/<pid>.xyz` (`:11`) but runs in the shared `my_dir` (`:22`); there is no `--norestart`, so a stale `xtbrestart`
  is reused (`:49`, S-1.2); convergence is never checked (S-1.3); the runtime parse is version-fragile (`:32-33`,
  `:51-52`, S-1.4); no `--chrg` (S-1.5).
- **Fix in TD (bug fix, allowed by PLAN "frozen except bug fixes"):** one commit to `utils/xtb.py`: drop the import-time
  directory; each call uses `tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))`, passes `--norestart`, and
  checks `.xtboptok`; output parsing otherwise unchanged (minimal diff). Shipped to the cluster as
  `patches/0002-xtb-tempdir.patch` via `round2/sync_cluster.sh`'s format-patch path. No current TD call path in our runs
  uses xtb (`--xtb` / `optimize_confs.py`), so no existing result changes.
- **flexitors runner** `flexitors/eval/energy.py::XTBRunner(binary, level='normal', threads=1)` implements all seven
  S-1 points of `round3/grounding.md:78-104`: per-call `mkdtemp` under `$TMPDIR`, `--norestart`, `.xtboptok` + return
  code → `xtb_converged`, no runtime parsing, `--chrg <RDKit formal charge> --uhf 0`, no `--alpb` (gas phase),
  single-conformer mols only, `OMP_NUM_THREADS=1`; `single_point(mol) -> dict`, `optimize(mol) -> (Mol, dict)`,
  `relax_report(mols) -> DataFrame` (E_relax, RMSD moved, post-relaxation connectivity via `validity`).
- **Environment** (`fx_common.sh::xtb_env`): try `source /etc/profile.d/modules.sh; module load u22/xtb/6.7.1`; if that
  fails (non-login shells, `round3/grounding.md:42-44`), use `/data3/u22/xtb-6.7.1/bin/xtb` with
  `LD_LIBRARY_PATH=/data3/u22/xtb-6.7.1/lib` (the round-3 probe configuration); then require
  `xtb --version | grep -q 6.7.1` or exit `FX_PREFLIGHT xtb`. `ulimit -s unlimited` before DRUGS-size molecules
  (INFERENCE from xtb's usual advice; harmless).
- Tests (S-1 list, `round3/grounding.md:101-104`): GT pass-through (< 1e-3 Å, E_relax < 0.05 kcal/mol), **two-worker
  Pool collision test** (two different molecules with the same atom count optimised concurrently 100 times: every
  output keeps its own atom identities and energies), charged molecule, forced non-convergence fallback counted.

### 7.4 QM9 validation subset (EVAL §1; PENDING Q-P3-007)
`cpu_stageA.sbatch TASK=valset` (8 CPUs, ≤ 1 h): `flexitors/data/valset.py` → `data/QM9/round5_val/` (csv, mols,
list, population). Needed by Exp 1 (σ_b) and Exp 5 (selector) before their test runs. The molecule list is committed
(EVAL §1 "list committed").

## 8. Data staging and the single download approval

### 8.1 Already on gnode118 (checked by `fx_preflight`; nothing to download)
| Item | Path under `$PROJECT` | Used by |
|---|---|---|
| QM9 raw pickles, split, test csv/mols | `data/QM9/{qm9/,split.npy,test_smiles.csv,test_mols.pkl}` | all |
| Released QM9 model | `workdir/pretrained/qm9_default/` (best_model.pt 6,388,281 B) | Exp 3, Exp 5 |
| Paired training cache | `data/QM9/cache_paired.{train,val}` (1.32 GB, `round3/grounding.md:56`) | M1 (PENDING Q-P3-001) |
| Round-2 seed pickles + `l_error.csv` | `data/QM9/round2_seeds/` | M0 |
| Split-half inputs | `data/QM9/round4_splithalf/` | §7.1, §7.2 |
| R0 released-model outputs | `results/qm9_default/R0_base_seed0/` | Exp 3 Step 0 |
| CR/B1/S3/S4 `steps20_seed0/eval.pkl` | `results/qm9_*_e100_s*/steps20_seed0/` | §7.1 |
| `local_structure_test.csv` (old) | `results/analysis/` | §7.2 regression check |

### 8.2 To download — one approval for all four files (TD authors' shared Drive, linked from
`torsional-diffusion/README.md:26,34`; IDs and byte sizes from `papers/blogs/td_gdrive_listing.txt`, E-BENCH-031,
E-BENCH-032)

| # | File | Drive file ID | Size (bytes) | Destination on gnode118 |
|---|---|---|---|---|
| 1 | `drugs.tar.gz` (data, splits, standardized pickles) | `1j_Pj6XZjJLf1tVQzjGWo2EFct2Nj_BvR` | 25,308,038,362 | `$PROJECT/downloads/drugs.tar.gz` |
| 2 | `workdir/drugs_default/best_model.pt` | `10tu38EDGe55M2v-0HtkMRYiPCcge4M5u` | 6,392,121 | `$PROJECT/workdir/pretrained/drugs_default/` |
| 3 | `workdir/drugs_default/model_parameters.yml` | `15uNAG2Le6lNY-VFq6TvXvncOfvr3N-De` | not recorded (text; the verbatim yml in the snapshot is ≈ 1 KB) | same |
| 4 | `workdir/drugs_default/drugs_steps20.pkl` (TD's own DRUGS test conformers) | `1HXY4FWt1CkQ28zPlnFzc6ZM_ZEnMRSoF` | 457,404,318 | same |

- Download URL pattern (the one `slurm/download_data.sh` already uses):
  `https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t` (first `gdown --continue`, then
  `curl -C -` resume; the HTML/quota-page guard and the expected-size check of `download_data.sh::gdrive_get`).
  Folder: `https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7`.
- **Total ≈ 25.77 GB (24.0 GiB).** Run as a CPU job on gnode118 (`slurm/download_drugs.sbatch`, 2 CPUs × 4000 MB,
  ≤ 12 h), which writes `data/DRUGS/DOWNLOAD_MANIFEST.txt` (size + sha256 of each file).
- **Not requested now (Stage B / Phase 5):** `xl.tar.gz` (`1hHBhw8WE1lCMc3sfDcxEP-v80tR01rju`, size not recorded) and
  `drugs_default/xl_steps20.pkl` (`1GMQ2xkq9Tolk44uV_rkfNX6EKnz-kLW6`, size not recorded) for GEOM-XL (EVAL §5 tier 2);
  ET-Flow `drugs-o3.ckpt` (100,676,519 B, Zenodo record 14226681, E-BENCH-033), which needs a separate environment.

### 8.3 Extraction (selective, to save disk and time)
- Pre-flight: `df -h /scratch` must show ≥ 60 GB free (archive 25.3 GB + extracted subset + caches; the full extracted
  size is unmeasured, EVAL §9 says "/scratch space check").
- `tar -tzf drugs.tar.gz > data/DRUGS/archive_listing.txt` first (one decompression pass, ≈ 10–20 min, INFERENCE), then
  extract only what Stage A reads: `DRUGS/test_smiles*.csv`, `DRUGS/test_mols.pkl`, `DRUGS/split*.npy` /
  `DRUGS/splits/`. The raw `DRUGS/drugs/*.pickle` and standardized pickles stay in the archive until Stage B needs them
  (E/Z-enriched validation set, training). Member names are taken from the listing, not assumed.
- The archive is kept (not deleted) so Stage B can extract more without a second download.
- Checks after extraction: `test_mols.pkl` has 1000 keys; csv rows 1000; Σ n_conformers = 106,778 (matches the
  GeoMol csv in `round5/ledger/chem_data/`); the released model loads strictly with `--in_node_features 74`
  (`model_parameters.yml`, E-BENCH-032).

## 9. Job design (sbatch, arrays, dependencies, resources)

All jobs: `--partition=plafnet2 --account=plafnet2 --nodelist=gnode118`, names `fx_*` (for the ledger), logs in
`$PROJECT/logs/%x_%A_%a.log`, memory ≤ 5000 MB per CPU, `source experiments/common/fx_common.sh` (which sources
`slurm/common.sh`), `diagnostics; fx_preflight; set -euo pipefail` as the round-2 scripts do.

| Job | Type | Resources | Wall cap | Depends on | Content |
|---|---|---|---|---|---|
| `fx_dl` `slurm/download_drugs.sbatch` | CPU | 2 CPU, 8000M | 12 h | user approval | §8.2–8.3 |
| `fx_cpu` `cpu_stageA.sbatch` array | CPU | per task: tests 4/16000M; ref_half 4/16000M; floor 16/64000M; valset 8/40000M; exp3_step0 8/40000M; m0 8/40000M; xtb_probe 4/16000M | 2 h each | — (ref_half, floor, valset, m0, step0 independent) | §4.1, §5.1, §7 |
| `fx_smoke` `smoke_stageA.sbatch` | GPU | 1 GPU, 8 CPU, 40000M | 3 h | `afterok:fx_cpu[tests]` | §10.3 |
| `fx_m1` `exp1/run.sbatch MODE=train` array 0-2 (or 1 packed task) | GPU | 1 GPU, 6 CPU, 30000M per run (18 CPU / 90000M packed) | 12 h, `--requeue --signal=B:USR1@900` | `afterok:fx_smoke`, `fx_cpu[m0]` PASS | §4.4 |
| `fx_m1eval` `exp1/run.sbatch MODE=eval` | GPU | 1 GPU, 8 CPU, 40000M | 3 h | `afterok:fx_m1`, `fx_cpu[valset]` | §4.4 read-out |
| `fx_inf` `infer_array.sbatch` (QM9, TSV-driven, PACK=3) | GPU | 1 GPU, 8 CPU, 40000M per task | 6 h | `afterok:fx_smoke`, `fx_cpu[valset]` | §5.2, §6 |
| `fx_drugs` `infer_array.sbatch` (DRUGS) | GPU | 1 GPU, 8 CPU, 40000M | 8 h | `afterok:fx_dl,fx_smoke` | §5.3 (timing task first, then arms) |
| `fx_dreval` evaluation shards | CPU | 16 CPU, 80000M per shard | 8 h | `afterok:fx_drugs` / `fx_dl` (D0) | §5.3, D0 scoring |
| `fx_ana` analysis drivers | CPU | 4 CPU, 16000M | 1 h | `afterany` of the producers | `exp*/analysis.py`, `status.py` |

- **TSV-driven inference** (round-2 `r2_evalsets.tsv` pattern): `experiments/exp3/arms.tsv` and
  `experiments/exp5/arms.tsv`, fields `SET | TAG | MODEL | SEED | BATCH_INDEX | KEYS | ARM_ARGS`. One array task = one
  model × set; inside it, `run_evalset`-style packing with PACK=3 (`slurm/common.sh:147-172`), each tag through
  `fx_gen_eval` = `TDSampler.run_csv` → splice (if KEYS) → `evaluate_confs.py` unchanged → `stereo_eval` → strict check.
  Task order puts the full O runs first, because every spliced arm needs its O.
- **Concurrency:** at most 3 GPU tasks of ours at once by default (`GPU_MAX=3`, `--array=…%3`), leaving one GPU for
  co-located users (INFERENCE: fewer collisions; the measured OOMs came from sharing a GPU, `round2/RUNLOG.md`). Raised to
  4 if `squeue -w gnode118` shows the node otherwise idle at submission. CPU total stays ≤ 44 of 48 at peak (3 GPU jobs
  × 8 + one 16-CPU CPU job + 4).
- **Submission** `experiments/common/submit_stageA.sh plan|cpu|smoke|gpu|drugs` with `DRY_RUN=1` (prints every
  `sbatch` with resources and dependencies, fake ids, as `slurm/submit_round2.sh:212-221`). `gpu` submits M1, the
  M1 read-out and the QM9 inference wave in one go with `afterok` chains, after checking `SMOKE_PASS` and the ledger.
  `drugs` waits for `DOWNLOAD_MANIFEST.txt`.
- **Code to the cluster:** `round2/sync_cluster.sh`, extended to `git archive … tools slurm flexitors experiments`
  and to write `flexitors/.commit` (the provenance code version). TD bug fixes go as format-patches, as in round 2.

## 10. Tests (unit, golden baseline equality, 20-molecule smoke on gnode118)

### 10.1 Unit tests (`flexitors/tests/`, run locally on CPU where the stack allows, and on gnode118 in `fx_cpu[tests]`)
From the specs, unchanged: EXP-1 §5.3 (`test_projector.py`, `test_invariance.py`, `test_parity.py`,
`test_constraints.py`, `test_golden_td.py`, `test_transform.py`; `test_rings.py` is Stage B), EXP-3 §6 tests (1)–(7),
EXP-5 §9 (`test_td_backend.py`, `test_budget.py`, `test_select.py`, `test_signals.py` limited to the Stage A signals —
MMFF energies; `test_curves.py`/`test_controller.py` are Track D, not built), EVAL §10 tests. Added by this plan:
- `test_seeding.py`: per-molecule seeds are subset-invariant (molecule X in a 3-row csv equals X in the 20-row csv,
  CPU, 1 thread, bitwise); batch 1 ≠ batch 0; ETKDG seed never 0.
- `test_resume.py`: kill `run_csv` after molecule 30 (simulated), restart, output equals an uninterrupted run (bitwise,
  CPU).
- `test_splice.py`: spliced pickle has O's objects for unaffected keys; a key outside the list raises.
- `test_k1_mask.py`: removing a bond from `edge_mask` also drops its `mask_rotate` row; frozen torsions bit-identical
  through `reverse_sde` (EXP-3 test (5)).
- `test_quadruples.py`: `td_quadruples` vs the PG rule of `sampling.py:135-141` on 50 QM9 + 50 DRUGS molecules
  (mismatch rate recorded; the TD-rule replication vs `get_transformation_mask`, EXP-3 test (1)).
- `test_xtb.py`: the four S-1 tests of §7.3 (cluster only; skipped without xtb).
- `test_oom_retry.sh` (bash): a fake command that fails once with "CUDA out of memory" is retried once; a fake
  `ValueError` is not retried.
- `test_provenance.py`: changed input hash → `StaleOutputError`; outputs without provenance → error.
- `tools/tests/test_leakfree.py` +: `--ref_subsample` with the full column set reproduces the reference (G10).
- `tools/tests/test_floor.py` (new): n_GT = 2 and 3 toy cases exclude the target itself; `other_half` restricts to
  the other half.

### 10.2 Golden baseline equality (TD frozen; these block the GPU wave)
| ID | Claim | How |
|---|---|---|
| G1 | `TDSampler(rng_mode='td_stream')` + `reverse_sde` with no hooks reproduces `generate_confs.py --seed 0` | 20 QM9 test molecules, CPU 1 thread: positions bitwise; GPU: ≤ 1e-4 Å (EXP-5 §9 asks < 1e-4 Å on 3 molecules) |
| G1-PG | same with the README non-invariant PG flags | 5 molecules, CPU bitwise |
| G2 | `FlexiTorsScoreModel(geometry_head=False)` = `TensorProductScoreModel` on a fixed batch; with the head on, `edge_pred` unchanged | CPU, `tools/tests/golden_dump.py` pattern (EXP-1 §5.3 `test_golden_td.py`, A3 control) |
| G3 | `td_loss` = TD `train_epoch` loss on one fixed batch | CPU |
| G4 | `eval/metrics` reproduces `R0_base_seed0/summary.txt` (AMR-R 0.1752, COV-R 89.0293, 65 failures) | EVAL §10 |
| G5 | `eval/stats` reproduces `contrasts.csv:50` (+0.00323 [+0.00127, +0.00523]) | EVAL §10 |
| G6 | `stereo_eval` reproduces `ezflip_qm9.out`'s R0 line (65/526) | EXP-3 test (7) |
| G7 | after the xtb patch, `utils/xtb.py` single-process outputs equal the round-3 probe (GT 0.000 Å) | `test_xtb.py` |
| G10 | ref_half with all columns = ref_rdkitL | §7.1 |
| G11 | floor re-run: every n_GT > 3 row byte-identical to the old CSV | §7.2 |

### 10.3 20-molecule smoke on gnode118 (`smoke_stageA.sbatch`, 1 GPU, ≤ 3 h; writes `SMOKE_PASS` only if all pass)
1. `pytest -q flexitors/tests tools/tests` on the cluster stack (RDKit 2022.09.5).
2. G1 on GPU for 20 QM9 test molecules.
3. Exp 3 on 20 stereo-DB QM9 molecules (from the 38): O, K1, K1-wrong, K1-canonical, K2-inf, F-flip, F-filter; guard
   counts zero E/Z violations for K1/K2 and non-zero for O (EXP-3 test (6)); splice + unchanged evaluator → SUMMARY;
   `verify_unaffected_identical` on 5 unaffected molecules re-run through an arm (≤ 1e-3 Å on GPU; S-E3c).
4. Exp 5 on 20 QM9 molecules: 4K pool from batches 0+1, selector, P6r with one batch per molecule; `assert_matched`
   passes; `budget.json` present.
5. Exp 1: 20-molecule `GeometrySampler` with an untrained head (counters, drift ≤ 1e-3 rad per 0.05 Å step,
   chirality guard active), then the **1-epoch smoke** and the **packing test** of §4.3 → `m1_projection.json`.
6. OOM drill: the retry wrapper is exercised once on a fake OOM (no GPU needed).
7. xtb: 20 molecules single point + opt through `XTBRunner` in a 2-worker Pool.
Expected GPU ≈ 0.5 GPU-h (the 1-epoch run dominates: 0.12–0.18 GPU-h by EXP-1 §6, × f).

## 11. Dead-code audit

Run before each push (PLAN.md rule), by `experiments/common/deadcode.sh`:
1. `vulture flexitors experiments --min-confidence 80` (installed in the cluster venv or run locally) plus a grep audit:
   every public function in `flexitors/` is referenced by a test **and** by a Stage A job or another module; every CLI
   flag is read; every config key is consumed (`yaml` keys vs `argparse` dests).
2. Only Stage A modules exist (§2.1 list); the Stage B list is enforced by a check that those paths do not exist yet.
3. No copied TD code: `flexitors/_td.py` is the only TD importer; a grep for long line-identical blocks against
   `torsional-diffusion/` (the one allowed port is `reverse_sde`, covered by G1).
4. Known dead or misleading TD code found while planning (record in `round5/IMPLEMENTATION.md`; fix only what a Stage A
   job uses):
   - `torsional-diffusion/optimize_confs.py:40` `if xtb_energy:` tests the imported **function** (always true), so
     `--xtb_energy` (`:14`) is never read; `:24` hard-codes the DRUGS csv. Not used in Stage A; noted for EXP-4.
   - `utils/xtb.py:5-7` import-time side effect (fixed, §7.3).
   - `diffusion/torus.py:6-17` defines `p`/`grad` used only to build the tables, then `p` is redefined at `:49`
     (shadowing, not dead; noted).
   - `diffusion/sampling.py:198-202` ODE / likelihood branch: not ported into `reverse_sde` (no Stage A arm uses it).
5. The ORACLE-only scripts (M0, Step 0, ref_half, floor) live in `experiments/` or `tools/`, never in the package
   import path of a sampler.

## 12. Schedule (hour by hour), GPU-h and CPU-h

Coding and review (≈ 3–4 days of agent work: package + tests ≈ 2,000–2,500 lines, INFERENCE) precede hour 0. Hour 0
= the user approves the download and the Stage A submission. Times assume a free queue; contention shifts but does
not reorder.

| Hour | GPU 1 | GPU 2 | GPU 3 | GPU 4 / CPU |
|---|---|---|---|---|
| 0–0.5 | — | — | — | sync; `fx_cpu`: tests (0.5 h), ref_half, valset, xtb probe; `fx_dl` starts (network-bound) |
| 0.5–2 | `fx_smoke` (20-mol suites, 1-epoch M1, packing test) | — | — | `fx_cpu`: floor re-run (16 CPU), M0, Step 0 |
| 2 | **auto-gate**: `SMOKE_PASS`, M0 PASS, `m1_projection.json` ≤ 16 GPU-h → `submit_stageA.sh gpu` | | | |
| 2–7.5 | M1 σ_b 0.05 (or all 3 packed) | M1 σ_b 0.15 | M1 σ_b 0.3 | — |
| 2–5.5 | — | — | — | `fx_inf` QM9 wave on GPU 4: O × 3 seeds first, then Exp 5 batch-1, PG, validation pools, Exp 3 spliced arms |
| ≈ 4–6 | — | — | — | `fx_dl` done (assumed; 25.3 GB at ≥ 20 MB/s ≈ 21 min, Drive throttling can make it hours) → selective extract → D0 scoring of `drugs_steps20.pkl` (CPU shards) |
| 5.5–6 | — | — | — | DRUGS 20-mol timing → `drugs_rates.json` → arm list |
| 6–9.5 | — | — | — | `fx_drugs` arms on GPU 4 (≤ 4.5 GPU-h) |
| 5.5–8 | — | — | — | CPU: QM9 Exp 3 / Exp 5 analyses; `choose_selector.py` on validation (must finish before the Exp 5 test selection, which is offline CPU) |
| 7.5–8.5 | `fx_m1eval` (validation σ_b choice → test once → `FX_GATE0`) | — | — | xtb energy on the M1 test outputs (CPU) |
| 9.5–16 | — | — | — | DRUGS RMSD evaluation shards (CPU-heavy), D0 completes; `status.py` final `STATUS.md` |

If packing passes (§4.3), GPUs 2–3 are free from hour 2 and the DRUGS arms move onto them as soon as the data is
extracted, cutting the wall clock by ≈ 3 h.

**GPU-h (Stage A)**
| Item | Expected | Worst case | Basis |
|---|---|---|---|
| Smoke (incl. 1-epoch run, packing test) | 0.5 | 0.7 | EXP-1 §6 smoke row; 20-mol suites ≈ 0.2 |
| M1 (A11) | 10.5 (≈ 5–7 if packed) | 16.0 | EXP-1 §5.2 / §6 |
| M1 read-out (validation × 3 σ_b + test × 1) | 0.4 | 0.6 | ≈ 600 molecules × 2K, small; INFERENCE |
| QM9 O × 3 seeds | 0.5 | 0.8 | 3 × 0.17 (packed, `round3/grounding.md:59-61`) |
| Exp 3 QM9 spliced arms (5 arms × 3 seeds × ≈ 10–20 % of molecules) | 0.3 | 0.5 | |
| Exp 5 (batch 1 × 3, PG 2K × 3, PG 4K × 3, validation 4K) | 2.4 | 3.0 | §6 |
| DRUGS timing + arms | 3.5 | 4.5 (hard cap) | §5.3 |
| Retry reserve (OOM, one rerun) | 0.5 | 1.0 | |
| **Total** | **≈ 18.6** | **≈ 27.1 → trimmed to ≤ 25** | |

Pre-declared trim order if the ledger projects > 25 GPU-h (applied by `submit_stageA.sh`): (1) DRUGS K1-canonical,
(2) DRUGS K1-wrong, (3) Exp 5 P6 (keep P6r, the NFE control), (4) DRUGS to O + K1 + K2-inf only. M1, Exp 3 QM9 and
Exp 5 P0–P3/P6r are never trimmed (they are what DECISION approved first).

**CPU-h (Stage A)**
| Item | CPU-h | Basis |
|---|---|---|
| Tests (cluster) | 2 | |
| ref_half re-score | < 0.5 | matrices only |
| Floor re-run | ≈ 10 (16 CPU × ≤ 1 h incl. embedding) | `slurm/analysis_qm9.sbatch:19` |
| M0, Step 0, validation set | ≈ 5 | |
| In-job QM9 evaluations (counted inside GPU jobs' 8 CPUs) | (≈ 30) | 0.17 GPU-h packed per condition |
| MMFF single points, selection | ≈ 2 | INFERENCE |
| xtb (M1 rigid test outputs + tests) | ≈ 1 | 0.3 s/conformer (`round3/grounding.md:36-37`) |
| DRUGS: D0 scoring of 1000 molecules + arms (141 molecules × ≤ 5 arms) | **≈ 50–200, unmeasured** | ≈ 2 × 10⁷ `GetBestRMS` calls for D0 (106,778 GT × 2K) at an unknown 5–30 ms each; measured on 20 molecules first, and the D0 full scoring may be deferred if it exceeds 100 CPU-h |
| **Total** | **≈ 75–220** (plus ≈ 30 inside GPU jobs) | |

## 13. Consult questions filed (`round5/consult/`)

| ID | To | Decision | Default (PENDING) |
|---|---|---|---|
| Q-P3-001 | R-JOINT | reuse round-2 `cache_paired` for M1 or rebuild; geometry loss on `pair_ok = False` | reuse; mask |
| Q-P3-002 | R-JOINT | bridge γ'(t) singularity (t clip ε); ODE sampling, N = 20; loss normalisation | ε = 1e-3; ODE N = 20; per molecule |
| Q-P3-003 | R-JOINT | Gate-0 read-out: σ_b on validation, test once; τ frozen; m = 0 geometry steps; other-half floor variant; M0 kill criterion on m > 0 (rigid case is automatic) | as listed |
| Q-P3-004 | R-CHEM | Stage A DRUGS scope: 141 stereo molecules, 1 seed, splicing for all-molecule NI, RDKit version | as listed, 4.5 GPU-h cap |
| Q-P3-005 | R-SAMPLE | D-101 pool for Track P; D-102 (a) validation selection criterion; c = 2 only | c·2K_i; AMR-R s.t. AMR-P ≤ +0.002 |
| Q-P3-006 | R-SAMPLE | D-102 (b) PG settings on QM9, one batch = whole set, P6r as the NFE-matched control | README non-invariant recall, untuned |
| Q-P3-007 | R-SAMPLE | QM9 validation-set construction (GT source, filtering, columns, sampling) | raw-pickle GT, cleaned, default_rng(0) |
| Q-P3-008 | R-CHEM (+JOINT) | E/Z handling in Exp 1 (the consult EXP-1 §2.8 asks for); one torsion classifier | measure only in M1; one classifier |
| Q-P3-009 | R-JOINT | M1 "geometry head only": train trunk from scratch, or freeze / fine-tune a TD trunk | whole network from scratch |

Not filed: EXP-1 §5.4 Q-*-003 (DAgger re-matching cost) — Stage B only.

## 14. Risks specific to running this on gnode118

1. **Co-located GPU memory (happened twice).** Mitigated by the in-job retry (§3.4), resumable generation and
   training, PACK=3 processes of ≈ 2–4 GB each, ≤ 3 of our GPU jobs at once, and the free-memory pre-flight.
2. **Drive download throttling / quota pages** for a 25.3 GB file: `gdrive_get`'s HTML and size checks plus `curl -C -`
   resume; the DRUGS jobs depend on the manifest, so nothing GPU waits idle — QM9 work runs meanwhile.
3. **Unmeasured costs** (projector overhead, packing gain, DRUGS per-sample cost, DRUGS `GetBestRMS` cost): each is
   measured on the smallest possible job before the expensive one, and each has a mechanical rule (§4.3, §5.3, §12).
4. **Non-bitwise GPU reductions** could make "unaffected molecules identical" false at 1e-6 Å: splicing copies O's
   conformers instead of regenerating, so the claim is only needed for the smoke check, at a tolerance.
5. **RDKit version drift** (C=N stereo, E-CHEM-060): every stereo label is computed on the cluster stack; the local
   2026.03.6 list is used only as a key list.
6. **Shared CPUs** (48): the CPU-heavy DRUGS evaluation is sharded and capped at 16 CPUs per shard, one shard at a time
   while GPU jobs run.
7. **Agent rate limits:** one `STATUS.md`, mechanical gates, DRY_RUN for every submission, and failures that announce
   themselves (`FX_FAIL`, `FX_OOM_RETRY`, `FX_BUDGET_STOP`, `FX_PREFLIGHT`) — so a stalled run costs one status read,
   not a log-reading session.

---
Status: COMPLETE (2026-10-16). Defaults marked PENDING change only if the consult answers differ.
