# Round 5 coding plan P2 (Stage A): emphasis on scientific validity

Author: coding planner P2, 2026-10-16. Independent plan (P1 and P3 not read). Inputs: `orchestration/README.md`
(protocol v2), `orchestration/roles/implementer_reviewer.md`, `round5/PLAN.md`, `round5/DECISION.md`,
`round5/specs/EXP-1.md` (rev. 1), `EXP-3.md`, `EXP-5.md`, `EVAL.md`, `round5/ledger/verify_V5.md`,
`round5/ledger/disputes.md` (D-101, D-102 open), `round2/FIXES.md`, `EXPERIMENTS.md` (S1, S3, Part 2b), and the code in
`torsional-diffusion/` and `tools/`. No code was changed and nothing was committed.

Status: complete (all sections written). Consult questions Q-P2-001 … Q-P2-011 are filed in `round5/consult/`.

Conventions. `path:line` refers to the files as they are on `main` at 9faa09f. "§" refers to the spec named in the
row. ORACLE = uses test-set ground truth; analysis only (README rule 5). INFERENCE = my reasoning, not a source.
Decisions a source leaves open are filed in `round5/consult/Q-P2-NNN.md` and the default used here is marked
**PENDING Q-P2-NNN**.

## 0. Scope of Stage A (from DECISION.md)

| Item | Spec | What Stage A delivers | What it does NOT build (no dead code, §6) |
|---|---|---|---|
| CPU-1 budget-matched leak-free re-score | EXP-1 §3.6, §9 item 2; D-001 | `tools/analyze_leakfree.py --ref_subsample` (default off) + run on gnode118 files | — |
| CPU-2 floor-bug fix | EXP-1 §9 item 1; D-003; E-JOINT-031 | fix at `tools/local_structure_analysis.py:302`, `--floor_other_half` variant, re-run of n_GT ≤ 3 rows | — |
| Exp 1 M0 (CPU) | EXP-1 §5.2 M0 | geometry core (W, P_W, constraints, chirality guard) + offline ORACLE projection analysis | V3a/V3b, CP rings, NeRF, SLSQP |
| Exp 1 smoke + M1 = Gate 0 (A11) | EXP-1 §5.2 M1, §2.2, §6 | V2 geometry head, paired transform, projected loss, joint sampler (τ-frozen mode), Gate-0 evaluation | DAgger, A6/A6b training, V3 |
| Exp 3 QM9 (+ DRUGS if budget and download approval) | EXP-3 §3.1–3.4, §5 Step 0, inference arms | torsion classifier, stereo guard, K1 / K1-wrong / K1-canonical / K2-inf, F-filter, F-flip, stereo_eval | K2-train, K3 (retrained arms) |
| Exp 5 selection at matched compute | EXP-5 §4.4 Track P, §5, §6 selector rows; D-101, D-102 | K-free test pool, validation pool, selector chosen on validation, PG NFE-matched arms | Track D controller, gain model, S1 ensemble probes |

Order of work (each step gates the next only where stated): CPU-1, CPU-2 and Exp 3 Step 0 first (no GPU); then
the 20-molecule smokes; then M0 → Exp 1 smoke → M1; Exp 3 and Exp 5 inference run on the fourth GPU while M1 trains.
EXP-1 §3.6 says the D-001 re-score must run "before any Exp-1 GPU time"; the schedule (§7) respects that.

## 1. Package layout

Consistent with `round5/PLAN.md` "Code structure": `torsional-diffusion/` stays frozen; `flexitors/` has exactly the
six subpackages PLAN.md names (`geometry/`, `models/`, `sampling/`, `training/`, `eval/`, `data/`) plus `tests/`.
EXP-3 §6 names two extra directories (`flexitors/diffusion/`, `flexitors/postprocess/`); I map them into the PLAN.md
subpackages (`sampling/reflected.py`, `sampling/stereo_fix.py`) so the layout stays as approved. Only Stage A modules
are created; every module below is used by a Stage A run or analysis.

```
flexitors/
  __init__.py
  td_compat.py              # the ONLY place that imports TD; path + cwd handling, RNG streams, provenance
  geometry/
    internal.py             # atan2 dihedrals, bonds, angles, signed volumes, TD quadruples
    jacobian.py             # analytic dihedral gradients W, rigid basis J_rig (batched torch, float64)
    projector.py            # P_W = I - C^T (C C^T)^+ C applied implicitly; rank / conditioning diagnostics
    constraints.py          # EXP-1 §2.6 bond window, clash test, chirality guard, step halving, counters
    torsion_types.py        # EXP-3 §3.1 TD-rotatable classifier, structural stereo test, targets phi_b
  models/
    heads.py                # VectorL1oHead (1o readout)
    flexitors_score.py      # FlexiTorsScoreModel(TensorProductScoreModel): + geometry head + geometry clock
  training/
    transforms.py           # PairedGeometryTransform (EXP-1 §2.2 steps 1-5)
    losses.py               # td_loss (verbatim training.py:25-31), projected_l_loss
    train.py                # CLI training loop (resume, fail-on-NaN, limit_iters), logs both losses
  sampling/
    reverse.py              # ONE reverse loop = sampling.py:184-235 with hooks (geometry step, per-edge update hooks)
    schedules.py            # TD sigma schedule (sampling.py:125-126), geometry times
    td_backend.py           # seeded, per-molecule-stream TD sampling (pools, O-paired); PG arms call TD sample()
    stereo_guard.py         # EXP-3 §3.3 per-step delta_b, violation counters, K1 freeze mask, K2 fold hook
    reflected.py            # fold_half_circle only (K2-inf); the reflected score is Stage B (K2-train)
    stereo_fix.py           # EXP-3 §3.4 F-filter (with resampling and NFE count), F-flip
    select.py               # EXP-5 §4.4 Track P selectors (random, FPS, lowest-E, filter+FPS)
    budget.py               # NFE / CPU-s accounting, assert_matched, budget.json
  eval/
    metrics.py  stats.py  strata.py  report.py           # EVAL §10
    validity.py             # EXP-1 §3 row 5: sanitise, connectivity from 3D, clashes
    stereo_eval.py          # EXP-3 §3.3/§4 labels (geometric + RDKit), E/Z fidelity, planarity
    rings_eval.py           # ring-dihedral error exactly as l_error.csv (E-JOINT-004 rev.)
    geometry_eval.py        # wrapper of tools/geometry_metrics.py
    energy.py               # xtb 6.7.1 single point / opt (used by the Gate-0 level-matched reference, §3.2)
    hypergeom.py            # exact expected min/coverage of a random column subset (ref_half check, D-101 analysis)
    selection_guard.py      # refuses test/ORACLE inputs to any selection or gate decision
    timing.py               # per-step timers, core-seconds per conformer
  data/
    val_subset.py           # EVAL §1 QM9 validation subset (1000 molecules, fixed seed) in test-file format
    ez_targets.py           # committed E/Z target table (one pinned RDKit), PENDING Q-P2-007
  tests/                    # §5
experiments/exp1/ exp3/ exp5/   # config_*.yml, sbatch_*.sh, README.md each; results/exp{1,3,5}/
```

Tool changes outside `flexitors/` (both default-off, golden-tested): `tools/analyze_leakfree.py` (`--ref_subsample`),
`tools/local_structure_analysis.py` (bug fix + `--floor_other_half`). Nothing in `torsional-diffusion/` changes.

## 2. Module APIs

All functions are typed, have docstrings, and carry the source comment required by PLAN.md
(`# <spec §> / <E-ID>`); positions are `torch.float64` inside `geometry/` (float32 only at the model boundary).

### 2.1 `td_compat.py`
- `import_td() -> None`: inserts `torsional-diffusion/` into `sys.path` and **changes cwd to the TD repo root for the
  import of `diffusion.torus`**. Reason: `torus.py:26-34` loads `.p.npy`/`.score.npy` from the cwd and otherwise
  recomputes and writes them there, and `torus.py:66-70` draws `np.random` at import time (unseeded). Asserts the
  SHA-256 of both tables equals the copies in the TD repo.
- `MolRng(global_seed: int, mol_key: str, stream: str)`: per-molecule RNG via `np.random.SeedSequence` → seeds
  `random`, `numpy`, `torch` (CPU) at the start of each molecule, and returns an independent `torch.Generator` for
  geometry noise (`stream='geom'`) so geometry draws never shift the TD torsion stream.
- `legacy_seed(seed)`: reproduces `generate_confs.py:94-95` exactly (one global seed), used only by golden tests.
- `provenance() -> dict`: git commit, dirty flag, RDKit / torch / e3nn versions, PACK, hostname; written next to every
  output (round2/FIXES.md X4).

### 2.2 `geometry/`
- `internal.dihedral(pos[B,n,3], quads[m,4]) -> [B,m]`: atan2 form in float64. **Not** TD's
  `get_torsion_angles` (`torsion.py:110-124`), which clamps cos at 1 − 1e-5 (`:120`) and so cannot resolve angles
  below ≈ 4.5e-3 rad or near π — too coarse for the 1e-3 rad drift test (EXP-1 §5.3) and for planarity.
- `internal.td_quadruples(mol, rule) -> LongTensor[m,4]`: `rule='td'` reproduces `standardization.py:69-90`
  (`n0[0], e0, e1, n1[0]`), keyed to TD's `edge_mask` order with the bond direction checked; `rule='heavy_first'`
  prefers heavy neighbours. Default `td` (EXP-1 §2.2), **PENDING Q-P2-001**.
- `internal.tetra_centres(mol) -> (centre[k], nbr[k,3])`: every atom of degree 4 (the structural set of
  `tools/lgeom.py:232`), neighbours ordered by `CanonicalRankAtoms`; `internal.signed_volume(pos, centres)`.
- `jacobian.dihedral_gradients(pos, quads) -> W[B,m,n,3]`: analytic Wilson-B rows (EXP-1 §2.2; E-JOINT-032).
  Returns `sin_min[B,m]` = min(sin θ_abc, sin θ_bcd) for the degeneracy test.
- `jacobian.rigid_basis(pos) -> [B,6,n,3]`: 3 translations + 3 rotations x_k × ê (TD Eq. 10, EXP-1 §2.2,
  E-JOINT-044).
- `projector.WProjector(pos, quads, batch, sin_floor)`: builds C = [W; J_rig] per molecule, normalises each row,
  drops rows with `sin_min < sin_floor` (counted; **PENDING Q-P2-001**), forms the (m'+6)×(m'+6) Gram matrix in
  float64, and solves with an eigen-threshold pseudo-inverse (relative 1e-10). Never forms the 3n×3n matrix, so cost
  is O(B·(m+6)²·n) and batched over molecules (D-006). Methods: `apply(v) -> v − Cᵀ G⁺ C v`; attributes `rank`
  (asserted = m'+6, or m'+5 for a linear molecule, logged), `cond`, `n_dropped_rows`.
- `constraints.ConstraintLayer(bond_tol=0.15, clash_frac=0.75, max_halvings=3, guard_chirality=True,
  chi_min=0.05)`: `step(x, dx, ref) -> (x_new, StepStats)` implements EXP-1 §2.6 steps 1–2 exactly (±15 % of seed
  length; 0.75 × sum of covalent radii; halve up to 3 times, then skip). `chi_min` (Å³) excludes near-planar
  degree-4 atoms from the sign test so noise cannot trigger it (INFERENCE; reported). `StepStats` counts halvings,
  skips, chirality flips **before** the guard, and guard back-tracks (EXP-1 §2.8; §3 row 9).
- `torsion_types.td_rotatable_bonds(mol)`, `classify_torsion(bond) -> TorsionClass` (the 7 classes of EXP-3 §3.1),
  `structural_stereo(bond) -> bool` (order 2, acyclic, each end has ≥ 1 heavy substituent besides the partner),
  `double_bond_targets(mol, quads, table) -> (phi[m], source[m])` with `source ∈ {smiles, seed, oracle_gt}`.

### 2.3 `models/`
- `heads.VectorL1oHead(ns, nv)`: e3nn `o3.Linear` from the `nv×1o` channels of the last atom layer
  (`score_model.py:83-89`, `:141-143`) to one `1o` vector per atom (EXP-1 §2.4, E-JOINT-021).
- `flexitors_score.FlexiTorsScoreModel(**td_kwargs, geometry_head: bool, geom_clock_dim: int)`: **calls the TD
  constructor first, then creates the head**, so `torch.manual_seed(s)` gives TD-identical base weights
  (golden G3). The geometry clock t enters through the existing extra-embedding slot (`lambda_embed_dim`,
  `score_model.py:67-79`, `:194-199`) with `data.node_lambda = t` — the same plumbing round 2 tested; with
  `geometry_head=False, geom_clock_dim=0` the class is TD's model. Forward returns `data.edge_pred` (unchanged) and
  `data.l_pred [N,3]`; the released checkpoint loads into the base with `strict=True` for the TD part.

### 2.4 `training/`
- `transforms.PairedGeometryTransform(sigma_min, sigma_max, sigma_b, geom_rng)`: EXP-1 §2.2 steps 1–5: index k as
  `dataset.py:79-99`, σ and Δτ drawn **with the TD RNG calls in TD order** (`dataset.py:71-76`), the same Δτ applied
  to x_R and x_G via `modify_conformer`, t ~ U[0,1] and z from `geom_rng` only, `x_t = interp_x_torch(x_R', x_G', t)
  + γ(t)·P_W(x_R') z`, target `u = P_W(x_t)[x_G' − x_R' + γ'(t) P_W(x_R') z]`. Stores per-sample diagnostics:
  residual rotatable-torsion mismatch |τ(x_G) − τ(x_R)| on TD quadruples, fraction of ‖x_G − x_R‖² removed by P_W,
  chirality sign changes of x_t vs x_R' (counted, not rejected, **PENDING Q-P2-003**).
- `losses.td_loss(edge_pred, edge_rotate, edge_sigma)`: verbatim `training.py:25-31`.
  `losses.projected_l_loss(v_pred, u, proj, dof)`: ‖P_W v_θ − u‖² / (3n − 6 − m') per molecule, mean over batch
  (EXP-1 §2.2 step 6). Total = `w_tors · td_loss + w_geom · projected_l_loss`; if `w_geom == 0` the geometry term is
  **not computed** (avoids `0 · NaN = NaN` reaching the trunk; golden G3b).
- `train.py`: `--w_tors --w_geom --sigma_b --limit_iters --resume --fail_on_nan`; loop modelled on
  `training.py:8-49`; logs both losses and the diagnostics above; checkpoint selection by **validation** loss only.

### 2.5 `sampling/`
- `reverse.reverse_loop(confs, model, sigma_schedule, steps, *, torsion_on=True, geometry=None, edge_hooks=(),
  counters)`: reproduces `sampling.py:184-235` statement for statement when `geometry is None` and `edge_hooks == ()`
  (golden G1). Noise `z` is drawn on the **full TD edge set** (`sampling.py:191`) before any hook, and hooks only edit
  the per-edge `perturb` vector, so arms that freeze or fold some bonds see the same random numbers on every other
  bond (common random numbers). `geometry` = `GeometryStep(proj, constraints, dt_schedule)` applies
  `x ← x + Δt·P_W(x) v` then the constraint layer, joint / L-first / L-last (EXP-1 §2.2 sampling step 3; Stage A uses
  joint only). When `geometry` is set, molecules with `edge_mask.sum() == 0` are **not skipped** (TD skips them,
  `generate_confs.py:153`), because the rigid stratum is the Gate-0 read-out.
- `td_backend.TDSampler(model_dir, ckpt, global_seed, mode='paired'|'legacy')`: `draw(smi, n, batch_idx)`; ETKDG
  `randomSeed = hash(global_seed, mol, batch) + 1` (never 0, `generate_confs.py:66`); torsion RNG from `MolRng`.
  PG arms call TD's own `sample()` (`sampling.py:116-246`) unchanged.
- `stereo_guard.StereoGuard(quads, targets, sign, mode)`: `delta(pos)`; `freeze_hook` (multiplies the constrained
  edges' update by 0, K1); `fold_hook` (K2-inf); per-step violation counters (|δ_b| > π/2; |δ_b| > 20° in the last 3
  steps). `sign[m]` is the measured relation between a `modify_conformer` update and the change of the quadruple's
  dihedral (see V-E3-2 in §4).
- `reflected.fold_half_circle(delta)`: wrap to (−π, π], then x > π/2 ↦ π − x, x < −π/2 ↦ −π − x (EXP-3 §3.2 K2).
- `stereo_fix.f_filter(mols, target, resample_fn, max_rounds=3) -> (mols, extra_nfe)`; `f_flip(mol, bonds)`.
- `select.select_2k(proposals: list[Mol], n_out: int, energy: ndarray, tors: ndarray, method, q, rng) -> idx`.
  Signature has **no RMSD, GT or K argument other than n_out** (static test). `energy` NaN (MMFF failure) handling is
  explicit and counted.
- `budget.NFECounter`, `budget.assert_matched(arms, tol=0.01)`; every generation writes `budget.json` (EVAL §8).

### 2.6 `eval/` and `data/`
- `metrics.per_molecule(eval_pkl, thrs)`, `aggregate(df, convention='td'|'intersection')` (EVAL §2–3).
- `stats.paired_bootstrap`, `holm`, `hierarchical` — ports of `tools/analyze_round2.py:123-159`, numerics unchanged.
- `hypergeom.expected_min(rmsd_row, s)`, `expected_cov(rmsd_row, s, t)`: exact expectation over a uniformly random
  s-subset of columns (sorted values; P(min = r_(j)) = C(N−j, s−1)/C(N, s)), the formulas of
  `round5/ledger/sample_data/curves.py`.
- `selection_guard.assert_selectable(path_or_frame)`: raises if the input is tagged `split=test` or `ORACLE`. Every
  gate and every hyper-parameter choice (σ_b, w, checkpoint, selector, PG settings) goes through it.
- `stereo_eval.labels(mol, bonds) -> {geometric, rdkit_db, rdkit_ch}`; `ez_fidelity`, `planarity`, `ez_absent_from_gt`.
- `rings_eval.ring_dihedral_error(mol, X, Y)`: the `l_error` definition (E-JOINT-004); population filter "largest
  ring ≥ 4" as in D-002.
- `data.val_subset.build(split_path, raw_dir, n=1000, seed)`: writes `val_smiles.csv` (`smiles, n_conformers,
  corrected_smiles`) and `val_mols.pkl` in the exact format of the test files (EVAL §1), **PENDING Q-P2-010**.

## 3. Stage A implementation per experiment

Each row: mechanism → spec section / E-ID it implements → touchpoints → the validity checks of §4 that guard it.

### 3.0 CPU items (before any Exp-1 GPU time)

**CPU-1, budget-matched leak-free reference** (EXP-1 §3.6; D-001; E-JOINT-001 rev.).
- `tools/analyze_leakfree.py`: new `--ref_subsample N_DRAWS` (default 0 = current behaviour, byte-identical output,
  G8). In the reference block (`:318-347`, `REF_TAGS` at `:78`), for each molecule and target half H ∈ {A, B}: take
  the rows of half H from the full-set `steps20_seed0` `eval.pkl` (row order = cleaned GT order; the split rule is
  `tools/make_split_half.py:19-22`), draw 2·|H| columns without replacement from the 2·n_csv generated columns, take
  row minima, average over N_DRAWS = 10 draws (the spec's procedure), average the two directions as `view()` does
  (`:116-128`). Output `ref_half` and contrasts `split − ref_half` per model (CR, B1, S3, S4) with the existing
  paired bootstrap (`:130`).
- Exactness check (INFERENCE, added): the same quantity is computed in closed form by `eval/hypergeom.py`; the 10-draw
  mean must lie within 3 Monte-Carlo SE of it for ≥ 99 % of molecules (V-L3). The closed form is reported beside the
  spec's 10-draw value; the spec's value is the one quoted.
- Assertions: `n_rows == n_clean` per molecule (otherwise halves are misaligned); `2·|H| ≤ 2·n_csv` (else that
  molecule is counted and excluded, in all views); the universe equals the split-half universe (743 molecules,
  E-JOINT-001) and is printed.

**CPU-2, floor bug** (EXP-1 §9 item 1; D-003; E-JOINT-031).
- `tools/local_structure_analysis.py:302`: replace `np.argsort(cost_gg[i])[:args.floor_topm]` by the first
  `floor_topm` indices of `np.argsort(cost_gg[i])` **with i removed** (the self cost is `np.inf`, `:281`, which only
  sorts it last). New flag `--floor_other_half` restricts candidates to the other half (parity of the cleaned index,
  as `make_split_half.py`), giving the split-half analogue V5 asked for.
- Golden: rows with n_GT > 3 must be unchanged (the bug cannot touch them), G9. Re-run the n_GT ≤ 3 molecules on
  gnode118 (CPU); write `floor_gtother_fixed` and `floor_other_half` as new columns, never overwriting the old CSV.

**CPU-3, Exp 3 Step 0 ceiling** (EXP-3 §5 Step 0). On R0 `confs.pkl`/`eval.pkl`: snap every TD-rotatable double bond
of each generated conformer to its planar target with a rigid side rotation (`modify_conformer`, `torsion.py:57-75`),
re-score with the unmodified evaluator. Specified bonds: target from the committed table (§3.3); unspecified bonds:
target from GT, labelled ORACLE (needs a GT→generated atom map, V-E3-5). Decision rule exactly as EXP-3 §5: ceiling
< 0.002 Å all-molecule AMR-R → QM9 inference arms only.

### 3.1 Exp 1 M0: geometry core and the offline projection analysis (CPU)

| Mechanism | Spec / E-ID | Touchpoints (reuse) |
|---|---|---|
| W: one dihedral gradient per rotatable bond on TD's quadruple | EXP-1 §2.2 "Revised version"; E-JOINT-032; D-004 | quadruple rule `standardization.py:69-90` (`n0[0]` at `:84-87`); edge order `torsion.py:9-39` |
| J_rig: 3 translations + 3 rotations | EXP-1 §2.2; TD Eq. 10 (E-JOINT-044) | rigid removal pattern `likelihood.py:84-93` |
| P_W = I − Cᵀ(CCᵀ)⁺C, batched | EXP-1 §2.2, §5.1 `projector.py`; D-006 (batched) | replaces the per-molecule numpy loop of `likelihood.py:77-98` |
| Constraint layer, chirality guard | EXP-1 §2.6 steps 1–2, §2.8 | degree-4 set as `tools/lgeom.py:232-241` |
| Negative control P_L (tangent projector) | EXP-1 §5.2 M0 (a), §5.3 | `dx_dtau` `likelihood.py:68-74` — used **only inside tests/M0 script**, not in the package |

M0 analysis script `experiments/exp1/m0_projection.py` (ORACLE). For each matched pair (x_R, x_G) — x_R the
torsion-matched RDKit conformer, x_G its own GT conformer Kabsch-aligned and terminal-relabelled
(`tools/lgeom.py:133-155`; `tools/build_paired_pickles.py` for train/val, `round2_seeds` for test):
- (a) dihedral drift of each TD quadruple after x_R + P_W(x_R)(x_G − x_R), in rad, with the P_L step beside it
  (expected 0.19–0.25 rad/Å first-order drift for P_L, E-JOINT-032);
- (b) fraction of ‖x_G − x_R‖² inside range P_W, and heavy-atom RMSD recovery 1 − RMSD(after)/RMSD(before);
- (c) chirality flips, bond-length distortion (max |Δr| per bond class, terminal X–H separately — the methyl lesson of
  `EXPERIMENTS.md` S1), and connectivity from 3D.
- **Population and decision set (deviation flagged).** The spec states the kill criterion on the rigid stratum. For a
  molecule with no TD-rotatable bond, W is empty and P_W only removes rigid motion, so after Kabsch alignment the
  projected step reproduces x_G almost exactly: the test is vacuous there (INFERENCE from the definition). The
  rigid stratum is defined by heavy rotatable bonds (`n_rot_heavy = 0`, E-JOINT-007), so many rigid molecules do have
  methyl/OH rotors in W, but those quadruples are H-terminated. Default: report M0 on the rigid stratum (sanity,
  expected ≈ 100 %) **and** on molecules with ≥ 1 heavy rotatable bond and a ring of ≥ 4 atoms, and apply the 50 %
  kill criterion to the latter; compute the decision on **validation** pairs (test pairs reported, ORACLE)
  — **PENDING Q-P2-004**.

### 3.2 Exp 1 smoke and M1 (Gate 0, arm A11)

| Mechanism | Spec / E-ID | Touchpoints |
|---|---|---|
| 1o vector head on the last atom layer | EXP-1 §2.4 (parity table), §5.1 `heads.py`; E-JOINT-021, -042 | `score_model.py:83-89` (irreps), `:141-143` (layer loop) |
| Geometry clock via extra embedding | EXP-1 §5.1 `flexitors_score.py` | `score_model.py:67-79`, `:192-199` (λ slot) |
| Paired transform, same Δτ on both endpoints | EXP-1 §2.2 steps 1–4; E-JOINT-023 | `dataset.py:79-99`, `:71-76`, `:18-34`; `torsion.py:57-75` |
| Interpolant anchored at RDKit, γ(t) = σ_b√(t(1−t)) | EXP-1 §2.2 step 4, §2.5; ET-Flow Eq. 2–5 (E-JOINT-076) | `interp_x_torch` `dataset.py:18-34` |
| Projected loss / (3n − 6 − m), w | EXP-1 §2.2 steps 5–6 | `training.py:25-31` (torsion term verbatim) |
| M1 recipe: torsion loss weight 0, τ noise applied, 30 epochs × σ_b ∈ {0.05, 0.15, 0.3} Å, full QM9 train | EXP-1 §5.2 M1, §2.2 "Choice of σ_b"; D-006 | `torsional-diffusion/train.py` options, QM9 args `slurm/common.sh` `QM9_MODEL_ARGS` |
| Geometry Euler step + constraint layer, τ frozen at seed | EXP-1 §2.2 Sampling, §2.6 | `sampling.py:184-235` → `reverse.py` |
| Rigid-stratum and ≥ 4-ring read-out | EXP-1 §5.2 M1 GO rule; §3 rows 4, 10; E-JOINT-004 rev., -007 | `tools/breakdown.py:32-80` (n_rot_heavy bins), `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv` |

Steps:
1. **Smoke (1 epoch, ≈ 0.12–0.18 GPU-h)**, EXP-1 §5.2: measures s/epoch with the head and projector against the
   377.7 s/epoch TD reference (E-JOINT-008), projector share of step time, rank failures, dropped W rows, NaN count.
   The M1 ask is restated from this number before M1 is submitted (D-006).
2. **M1**: 3 runs in parallel on 3 GPUs, `w_tors = 0`, `w_geom = 1`; checkpoint by validation geometry loss.
3. **Gate-0 evaluation.** Sampler mode "τ frozen": no `perturb_seeds` (`sampling.py:105-108`), torsion updates 0,
   geometry steps on for N = 20 steps, molecules with m = 0 not skipped. For the rigid stratum, heavy-atom RMSD does
   not depend on τ (only H rotors are rotatable there) — asserted, V-E1-12. Metrics: rigid AMR-R (evaluator
   unchanged, 2K per molecule); share of ≥ 4-ring seeds with ring-dihedral RMSD > 10° on the paired seeds of
   `l_error.csv` (same population and code path as the 49.6 % baseline, golden G7), plus the best-over-GT variant
   (**PENDING Q-P2-004**); drift, chirality flips, backtracks, skips, bond-length MAE.
4. **Choice of σ_b and the GO decision on validation** (EXP-1 §4 control 8; EVAL §0 rule 3): validation rigid
   stratum and validation ≥ 4-ring seeds; the test numbers are produced once, for the chosen σ_b, after the choice is
   written to `results/exp1/gate0_decision.json` (hash recorded). The spec's §5.2 text names the 289 test molecules,
   so the population used for the decision is **PENDING Q-P2-004**.
5. **Level-matched reference at Gate 0 (CPU, added from EXP-1 §4 control 2 / arm A2):** on the rigid stratum the
   torsion model plays no role, so A2 reduces to "ETKDG seeds relaxed with GFN2-xTB" scored directly (xtb 6.7.1,
   ≈ 0.14 s per QM9 optimisation, E-JOINT-008). Also ETKDG + MMFF (0.128 on test, E-JOINT-007). Reported beside the
   Gate-0 number so that "the head beats free optimisation" is visible at the first gate, not only in Stage B.
6. **A3 golden** (EXP-1 §4 control 4, §5.3 `test_golden_td.py`) passes before M1 is submitted.

E/Z in Exp 1 (EXP-1 §2.8 "E/Z", §5.4 Q-*-002): Stage A only measures it on Gate-0 outputs with `eval/stereo_eval.py`
(shared with Exp 3); how V2 inherits the Exp 3 constraint in Stage B is **PENDING Q-P2-006**.

What Stage A does for Stage B controls without building them: the τ-frozen sampler mode is exactly A6/A6b's
independent L sampler; geometry-head checkpoints are hashed so A5 and A6b provably share one head (V-E1-14).

### 3.3 Exp 3: stereo-constrained sampler, inference arms

| Mechanism | Spec / E-ID | Touchpoints |
|---|---|---|
| TD-rotatable classifier, 7 classes; structural stereo test drives constraints | EXP-3 §3.1; E-CHEM-020, -050, -060, -070 | `torsion.py:9-39`; port of `round5/ledger/chem_data/stereo_census.py:26-75` |
| Target φ_b ∈ {0, π} on TD reference atoms | EXP-3 §3.1 | reference atoms `sampling.py:135-142`; `get_torsion_angles` `torsion.py:110-124` (replaced by atan2) |
| K1 freeze (seed check rotates by π if |δ_b| > π/2) | EXP-3 §3.2 K1; E-CHEM-043 | `sampling.py:105-108` (seed randomisation), `:204`, `:234` |
| K1-wrong, K1-canonical | EXP-3 §5 table | same |
| K2-inf: TD score, fold to [−π/2, π/2] | EXP-3 §3.2 K2-inf; E-CHEM-048 | `sampling.py:204` (perturb), `torsion.py:57-75` |
| Per-step guard and counters | EXP-3 §3.3 | `sampling.py:184-234` |
| F-filter (replacement mandatory, ≤ 3 extra rounds, NFE counted), F-flip | EXP-3 §3.4; E-CHEM-064, -046 | `evaluate_confs.py:47-54` (non-isomeric filter means E/Z-wrong conformers are scored, E-CHEM-028) |
| Labels and endpoints | EXP-3 §3.3, §4; E-CHEM-061..064 | `round5/ledger/chem_data/ez_flip_analysis.py`, `tools/stereo_check.py:30-45` |

Implementation choices that keep the arms paired (INFERENCE, each checked by a test):
- **K1 as an update mask, not an `edge_mask` edit.** EXP-3 §3.2 removes the bond from `edge_mask`. In TD the conv graph
  (`score_model.py:183-213`) does not depend on `edge_mask`, and `bond_conv` scores each rotatable bond from its own
  neighbourhood (`:145-154`), so the other bonds' scores are the same either way. Multiplying the constrained edges'
  update by 0 is therefore equivalent for the model, keeps `mask_rotate` untouched, and keeps the noise vector the
  same size as in O (common random numbers). Equivalence is asserted on a fixture (V-E3-3). `modify_conformer` skips
  zero updates (`torsion.py:60`), so frozen torsions stay bit-identical (EXP-3 §6 test 5).
- **Per-molecule RNG streams** (`MolRng`): with TD's single global seed (`generate_confs.py:94-95`), any arm that draws a
  different number of random values (K1 draws fewer in `perturb_seeds`) desynchronises every later molecule. With
  per-molecule streams and noise drawn on the full edge set, molecules without a constrained bond produce the same
  trajectory in every arm. The O arm is therefore **re-run in paired mode** ("O-paired", 3 seeds); the released R0–R2
  outputs are kept for the reproduction of E-CHEM-061 only.
- **Affected-molecule runs.** K arms are generated only for molecules with ≥ 1 constrained bond (≈ 102 QM9 molecules
  with a TD-rotatable double bond, plus triple-bond molecules for K1-canonical), and spliced with O-paired outputs for
  the all-molecule non-inferiority endpoint. Splicing is valid only if unaffected molecules are reproduced; checked on
  20 unaffected molecules per arm (V-E3-4).
- **Version-independent E/Z.** Constraints use the structural test (EXP-3 §3.1). Targets for "specified" bonds come
  from a table computed once with a pinned RDKit and committed (`data/ez_targets.py`), keyed by SMILES and TD atom
  indices; the primary label is geometric (sign of cos φ on the TD quadruple), with RDKit `AssignStereochemistryFrom3D`
  labels reported under both versions — **PENDING Q-P2-007**.
- **DRUGS** (EXP-3 §4 (ii), EVAL §9 D0): needs the user's explicit approval of the 25.3 GB download (protocol) and is
  scheduled last within the budget (§7); a 20-molecule timing run precedes any DRUGS arm (EXP-3 §7 "measure one run
  first").

### 3.4 Exp 5: selection at matched compute (Track P only)

| Mechanism | Spec / E-ID | Touchpoints |
|---|---|---|
| Incremental, reproducible TD sampling | EXP-5 §8 `td_backend.py` | `generate_confs.py:64-69`, `:125-180`; `sampling.py:40-60`, `:63-102`, `:105-113`, `:116-246` |
| K-free fixed-size test pool | D-101 (requested action 1) | — |
| Selectors: random, torsion FPS, lowest MMFF E, MMFF-filter q + FPS | EXP-5 §4.4 Track P, §6 selector row; E-SAMPLE-025 | prototype `round5/ledger/sample_data/select.py:13-41` (port, with fixes below) |
| Selector chosen on validation; test used once | D-102 (a); EVAL §0 rule 3; EXP-5 §4.3 | — |
| NFE accounting, ±1 % matching | EXP-5 §4.1, §5 "Matching rule" | `sampling.py:184-188` (one NFE per conformer per step) |
| PG arms at matched NFE | D-102 (b); EXP-5 §5 Track P; E-SAMPLE-030..034 | `sampling.py:128-175`, `:206-232`; `generate_confs.py:38-46` |

Design:
- **Test pool (D-101).** Per molecule, N_i = max(128, 12 K_i) conformers, 20 steps, generated in batches with hashed
  per-batch seeds so that every column block is an i.i.d. sample. The **first 128 columns are the K-free pool** V5 asked
  for: on the QM9 test csv the largest K is 30 (computed from
  `round5/ledger/chem_data/qm9_test_smiles_from_local_structure_test.csv`: Σ2K = 15,108 over the 940 rows with K;
  2K ≤ 60), so 128 ≥ 4K for every molecule. Any allocation or headroom re-analysis (ORACLE, CPU) may read only
  columns < 128 (asserted). The columns beyond 128 exist only so that the three Track-P replicates use **disjoint**
  blocks [4K·r, 4K·(r+1)), r = 0, 1, 2 (EXP-5 §5 "3 sampling seeds"); 23 % of molecules need them. Total
  ≈ 10.1 × Σ2K. V5's max(10 × 2K, 128) is still K-dependent above K = 6; a fixed 128 is K-free and cheaper — **PENDING
  Q-P2-008**. The cap-binding share of the uniform and oracle allocators on the 128 columns is reported.
- **Validation pool.** 4K per molecule (c = 2 exactly) on the EVAL §1 validation subset: the selector choice needs
  nothing more, and Track D (which would need a K-free validation pool) is dropped in Stage A.
- **Selector choice on validation (D-102 a).** Candidate set = the prototype's set {random, FPS, lowest-E,
  filter 25 % + FPS, filter 50 % + FPS} at c = 2; the selection rule is written into the analysis script before the
  validation pool exists: minimise AMR-R subject to AMR-P ≤ AMR-P(random) + 0.004 Å and COV-P@0.5 ≥ COV-P(random)
  − 0.5 pt (the round-2 non-inferiority margins, EXP-3 §4) — **PENDING Q-P2-009**. Then test once.
- **Two prototype degeneracies fixed and tested.** (i) At c = 2, `filter 50 %` keeps `max(n, ceil(0.5 · 2n)) = n`
  conformers (`select.py:40`), so FPS returns all of them and the arm equals "lowest MMFF E" exactly (the two rows of
  `select.out` are identical). The port asserts `|keep| > n_out` or labels the arm degenerate. (ii) The spec starts
  FPS at the lowest-energy proposal (EXP-5 §4.4); the prototype starts at a random one (`select.py:16`). Default: the
  spec — **PENDING Q-P2-009**.
- **PG at matched NFE (D-102 b).** Rows: P0 TD c = 1; P1 TD c = 2 + random (NFE control); P2 PG c = 1 (NFE-matched
  to P0 only); P3 TD c = 2 + selector; **P6r PG with 2 × 2K particles + random 2K** and **P6 PG 2 × 2K + selector**
  (both NFE-matched to P3). K3 (EXP-5 §11) is judged on P3 vs P6r. PG hyper-parameters for QM9 are tuned on
  validation with a tuning budget equal to the selector's (5 configurations), **PENDING Q-P2-009**.
- **PG batch rule.** TD's PG kernel works within one loader batch (`n = data.num_graphs`, `sampling.py:207-229`), so
  with the default `batch_size = 32` (`generate_confs.py:34`) a molecule with 2K > 32 gets PG only inside chunks. PG
  arms set `batch_size ≥ max proposals per molecule` (≤ 120) and assert one batch per molecule (V-E5-6).
- **Energy**: MMFF94 single points of generated conformers on CPU (EXP-5 §4.2 S3); core-seconds reported next to NFE
  (EXP-5 §4.1). NaN energies are counted and treated as +∞ for filtering (as the prototype, `select.py:36-40`), and the
  count is reported because it is a hidden filter.

## 4. Validity checks (risk → assertion / test / analysis)

"Hard" = the run aborts or the result is not tabled. "Report" = printed in every table of that arm.

### 4.1 Cross-cutting (lessons of rounds 1–4)
| ID | Risk (lesson) | Assertion / test / analysis | Kind |
|---|---|---|---|
| V-X1 | Oracle leakage into decisions (round-4 leak-free test; README rule 5) | `selection_guard.assert_selectable` on every gate, σ_b / w / checkpoint / selector / PG choice; ORACLE outputs only under `results/<exp>/oracle/` with `ORACLE` in the file name; unit test feeds a test-tagged frame and expects a raise | Hard |
| V-X2 | Silent population loss (358 of 997 dropped, `round2/FIXES.md` X1) | every builder and every arm writes `population.txt` (eligible / kept / replaced / dropped, with reasons); the analysis aborts if an arm's universe differs from the declared one; TD-convention universe = all 1000 rows, intersection reported beside it (EVAL §3) | Hard |
| V-X3 | Failures handled one way only | both conventions per arm and seed (EVAL §3); molecule failures from `generate_confs`-style embedding are failures in every arm (EXP-5 §11 risk "stochastic failures"); invalid conformers (validity.py) kept in the primary (evaluator unchanged), and a secondary view with invalid conformers' columns set to +∞ (sample count unchanged) — **PENDING Q-P2-011** | Hard + Report |
| V-X4 | Comparisons chosen after results (`tools/analyze_round2.py:1-23` pattern) | each experiment's analysis script with contrasts, Holm families and decision rules committed **before** the first run; a test checks the family list against the spec tables (EXP-1 §4, EXP-3 §4, EXP-5 §6 as amended by D-102) | Hard |
| V-X5 | Code changes the baseline silently (golden tests round 2) | G1–G10 (§5.2) run on CPU before any submission | Hard |
| V-X6 | Non-reproducible runs (round-1: COV-R 89.0286 vs 89.0293 across processes) | provenance (commit, dirty flag, RDKit/torch versions, PACK) next to every output; GPU nondeterminism documented; paired designs rely on per-molecule RNG streams, not on bit equality | Report |
| V-X7 | Resource requests (plafnet2 5000 MB/CPU, FIXES X2; packed failures hidden, X3) | sbatch lint test: mem/CPU ≤ 5000 MB; packed sub-jobs exit non-zero on any failure | Hard |
| V-X8 | Precision hidden behind recall (EVAL §2) | `report.py` refuses a table without AMR-P / COV-P beside AMR-R / COV-R | Hard |

### 4.2 Exp 1
| ID | Risk | Assertion / test / analysis | Kind |
|---|---|---|---|
| V-E1-1 | P_W does not annihilate the dihedral gradients (D-004) | `‖W P_W v‖ / ‖W‖‖v‖ < 1e-10` (float64) and `‖J_rig P_W v‖` likewise, on 50 QM9 molecules × 10 random v; idempotent and symmetric to 1e-12 | Hard (test) |
| V-E1-2 | Analytic W wrong | central finite differences of `internal.dihedral` (h = 1e-5 Å) match W to rel. 1e-6; sign convention checked on a hand-built butane | Hard (test) |
| V-E1-3 | "First order only" not true in practice | drift on TD quadruples after one projected step at RMS 0.005 / 0.01 / 0.02 / 0.05 Å: log-log slope in [1.8, 2.2], and < 1e-3 rad at 0.05 Å (EXP-1 §5.3); P_L negative control shows slope ≈ 1 and 0.19–0.25 rad/Å (E-JOINT-032) | Hard (test) + Report |
| V-E1-4 | Drift accumulates over 20 steps | per-trajectory cumulative drift of every TD quadruple (atan2) reported per arm (EXP-1 §3 row 9); alarm if median > 1°. Option `restore_torsions` (re-impose the m quadruple dihedrals by rigid side rotations after each geometry step, which changes no bond length or angle) exists only as an ablation; default off as the spec says — **PENDING Q-P2-002** | Report |
| V-E1-5 | "Torsion moved through L" (INFERENCE): when the TD quadruple of a bond uses an H (`standardization.py:84-87` takes `n0[0]`, which can be H), P_W fixes that H-dihedral while the heavy-atom dihedral about the same bond can still move; the geometry head could then do torsion work and be credited as "L" | diagnostic: change of **all** heavy-atom dihedrals about TD-rotatable bonds per geometry step and per trajectory; share of TD quadruples that contain an H; ablation `rule='heavy_first'` — **PENDING Q-P2-001** | Report |
| V-E1-6 | Ill-conditioned C (linear segments: TD treats triple bonds as rotatable, 123 in QM9 test, E-CHEM-050/060; angles near 180° make dihedral gradients blow up as 1/sin θ) | row normalisation; float64 eigen-solve with relative threshold; rows with min(sin θ) < sin 10° dropped and counted; `rank == m' + 6` (or m' + 5 if linear) asserted per molecule per step; `cond(G)` logged; test on HC≡C–CH2–OH and on a molecule with an allene | Hard + Report |
| V-E1-7 | Ring opening / bond breaking | connectivity from 3D (`rdDetermineBonds`) equals the input graph for every output conformer; ±15 % bond window (EXP-1 §2.6); a tighter diagnostic (share of bonds off the seed by > 0.05 Å) because 15 % of a C–H is 0.16 Å | Hard (count) + Report |
| V-E1-8 | Methyl/X–H collapse under linear steps (`EXPERIMENTS.md` S1: C–H 1.09 → 0.62 Å) | interpolant uses `interp_x_torch` (terminal bonds kept, existing test C7b); M0 reports terminal X–H length after the projected oracle step; unit test with a methyl rotated by 120° | Hard (test) + Report |
| V-E1-9 | Chirality flips (V5: 1 in 20 random 0.3 Å steps) | guard after every geometry step on all degree-4 atoms with |χ_seed| ≥ chi_min (version-independent, as `lgeom.py:232`), plus RDKit-perceived centres as a reported subset; flips before the guard and back-tracks reported (EXP-1 §2.8); tests: enantiomer (−x) flagged, deliberate 0.3 Å flipping step caught, GT conformers vs their own seed pass | Hard + Report |
| V-E1-10 | Transform: different Δτ on the two endpoints, or the geometry draw shifts TD's noise stream | test: τ(x_R') − τ(x_G') = τ(x_R) − τ(x_G) on TD quadruples (EXP-1 §5.3 `test_transform.py`); with `w_geom = 0` the TD RNG state after N samples equals TD's transform (G3b) | Hard (test) |
| V-E1-11 | Residual torsion mismatch in the matched pair puts torsion signal into the L target (INFERENCE: x_R torsions are DE-matched, not exact) | per-sample diagnostics: |Δτ| distribution on TD quadruples, fraction of ‖x_G − x_R‖² removed by P_W; report per epoch — target definition **PENDING Q-P2-003** | Report |
| V-E1-12 | Rigid stratum read-out polluted by torsions; TD skips m = 0 (`generate_confs.py:153`) | assert: on n_rot_heavy = 0 molecules random TD torsion updates change Kabsch-aligned heavy-atom positions by < 1e-6 Å; the joint sampler runs the geometry head on m = 0 molecules (count asserted = stratum size) | Hard (test) |
| V-E1-13 | Parity / equivariance wrong silently (EXP-1 §2.4) | x → −x: torsion head negates, 1o head negates; random rotation R: 1o output rotates by R (< 1e-5); translation invariance; checked on the released checkpoint + random head | Hard (test) |
| V-E1-14 | "Win by more optimisation" | (a) NFE per conformer = TD's (one forward pass, two heads), counted; (b) no force field in any primary arm, the constraint layer is pure geometry and its back-tracks/skips are reported; (c) Gate-0 level-matched references ETKDG + MMFF and ETKDG + xTB-opt on the same seeds (§3.2 step 5); (d) projector wall-clock share reported so A12 (equal wall-clock V1) can be sized; (e) A3 golden (equal code path); (f) training GPU-h per arm recorded so DAgger's extra epochs can be matched in Stage B — **PENDING Q-P2-005** | Report (+ Hard for e) |
| V-E1-15 | Gate decided on test | σ_b and GO decided on validation strata (V-X1); test numbers produced after the decision file is hashed — **PENDING Q-P2-004** | Hard |
| V-E1-16 | Gate metric not comparable with its baseline | ring-dihedral metric computed by `rings_eval` must reproduce ETKDG 49.6 % / MMFF 35.1 % on `l_error.csv` ≥ 4-ring seeds (D-002; G7) before it is applied to model output | Hard (golden) |
| V-E1-17 | A6b and A5 later use different geometry heads | checkpoint SHA-256 written into every run's provenance; the Stage B analysis asserts equality | Hard (Stage B) |
| V-E1-18 | Budget-matched reference wrong (D-001) | 10-draw MC within 3 SE of the closed form; universe = split-half universe; `n_rows == n_clean`; with `s = n_cols` the re-score reproduces `ref_rdkitL` exactly | Hard (test) |
| V-E1-19 | Floor fix wrong (D-003) | unit test with a mocked cost matrix: self index never among the candidates; `--floor_other_half` returns only other-parity indices; rows with n_GT > 3 byte-identical to the existing CSV (G9) | Hard (test) |

### 4.3 Exp 3
| ID | Risk | Assertion / test / analysis | Kind |
|---|---|---|---|
| V-E3-1 | TD rule replicated wrongly | `td_rotatable_bonds` equals `get_transformation_mask` (`torsion.py:9-39`) on 50 QM9 + 50 DRUGS molecules (EXP-3 §6 test 1); census reproduces 2,880 / 108 / 102 (V5 recompute; G10) | Hard (test) |
| V-E3-2 | Sign convention: `modify_conformer` rotates by `pos[u] − pos[v]` ("DIFFERENT FROM THE PAPER", `torsion.py:68`), so a +Δ update may change the measured dihedral by +Δ or −Δ depending on quadruple orientation; a wrong sign folds the wrong way and puts the conformer on the wrong side | per bond, `sign_b` is measured once (apply +1e-3 rad, read atan2 change) and stored; test on 200 random bonds that Δφ = sign_b·Δ to 1e-6; the fold hook works in dihedral space and maps back with sign_b | Hard (test) |
| V-E3-3 | K1 as update mask ≠ spec's `edge_mask` removal | fixture: model scores of unconstrained bonds identical (< 1e-6) with and without the constrained bond in `edge_mask` | Hard (test) |
| V-E3-4 | Arms not paired; splicing invalid | per-molecule RNG streams; 20 unaffected molecules generated in each K arm equal O-paired (CPU: bit-identical; GPU: max deviation reported, alarm > 1e-3 Å); noise drawn on the full edge set | Hard (CPU test) + Report |
| V-E3-5 | Atom-index mismatch between GT and generated molecules (GEOM GT mols vs SMILES-built mols) when a GT reference is needed (Step 0 ORACLE target, "E/Z absent from GT") | heavy-atom substructure map GT → generated; the mapped quadruple must give the same structural classification; symmetric maps that swap the reference atoms are tested (a graph-equivalent swap means the bond is not stereogenic, so the label is unaffected) | Hard (test) |
| V-E3-6 | RDKit-version-dependent stereo perception (38 vs 103 molecules, E-CHEM-060) | constraints by the structural test; target table committed once (pinned RDKit, version recorded); primary label geometric; RDKit labels under both 2022.09.5 and 2026.03.6 reported; test: structural bond set identical under both versions on the QM9 test set — **PENDING Q-P2-007** | Hard + Report |
| V-E3-7 | TD's `get_torsion_angles` clamps cos (`torsion.py:120`): planarity near 0/π is mis-measured | all planarity and δ_b computed with atan2 (`internal.dihedral`); test against `rdMolTransforms.GetDihedralRad` to 1e-9 | Hard (test) |
| V-E3-8 | "Fewer degrees of freedom" mistaken for "correct stereo" | K1-wrong negative control (EXP-3 §5) always run with K1 | Hard (analysis) |
| V-E3-9 | Freezing a random isomer for unspecified bonds hurts recall | specified and unspecified bonds reported separately; GT isomer never used outside ORACLE (EXP-3 §8) | Report |
| V-E3-10 | GT ensembles with mixed E/Z (3 of 38) | listed separately in every Exp 3 table (EXP-3 §4) | Report |
| V-E3-11 | F-filter gets extra compute | extra NFE counted; F-filter sits in its own table with an NFE column (EVAL §1 (ii)) | Report |
| V-E3-12 | Seed check (rotate by π) creates clashes | clash test after the rotation; count; the conformer is kept (TD convention) | Report |
| V-E3-13 | K2-inf fold leaves δ outside I | test: fold output in [−π/2, π/2] for 1e6 random inputs; involution at the boundary (EXP-3 §6 test 4) | Hard (test) |
| V-E3-14 | Guard counts are wrong | zero violations for K1/K2 on 20 molecules, non-zero for O on the 38 stereo molecules (EXP-3 §6 test 6); `stereo_eval` reproduces 65/526 for R0 (EXP-3 §6 test 7; G6) | Hard (test) |

### 4.4 Exp 5
| ID | Risk | Assertion / test / analysis | Kind |
|---|---|---|---|
| V-E5-1 | K leaks into "K-free" rows through the pool cap (D-101) | K-free pool = first 128 columns; any allocation/headroom code asserts it reads only those; cap-binding share reported | Hard |
| V-E5-2 | Selector chosen on the test set (D-102 a) | candidate set and selection rule in the committed script; choice on validation through `assert_selectable`; test evaluated once; every test number carries a paired CI and the hierarchical (pool replicate, molecule) CI | Hard |
| V-E5-3 | Pools are not i.i.d. (prefix ≠ fresh sample) | `draw(n)` equals the concatenation of two `draw(n/2)` calls (EXP-5 §9); ETKDG with fixed `randomSeed` gives identical conformers across two runs and across `numThreads` 1 vs 5; two disjoint replicate blocks have the same expected AMR-R within CI | Hard (test) |
| V-E5-4 | Compute not matched (D-102 b) | `budget.assert_matched` (±1 % NFE) for every pre-declared contrast of the Stage A family P3 − P1, P3 − P6r, P6 − P6r (EXP-5 §6's family
  minus the Track-D arms P4/P5, with P0 replaced by its NFE-matched twin P1 — **PENDING Q-P2-009**); MMFF core-seconds reported beside NFE (EXP-5 §4.1) | Hard |
| V-E5-5 | Selector degenerate (filter 50 % at c = 2 equals lowest-E, `select.py:40`) | assert `|keep| > n_out` or label the arm degenerate and drop it from the candidate set | Hard |
| V-E5-6 | PG works only inside loader batches (`sampling.py:207-229`) | PG arms assert one batch per molecule (`batch_size ≥ proposals`) | Hard |
| V-E5-7 | Selector reads GT or K | static test: `select_2k` signature and module imports contain no RMSD / GT / evaluator; K enters only as `n_out = 2K` (EXP-5 §4.5) | Hard (test) |
| V-E5-8 | MMFF failures act as a hidden filter | NaN energies counted per arm; sensitivity row with NaN → median energy | Report |
| V-E5-9 | Failure sets differ between arms | all Track-P arms are subsets of one pool, so the universe is identical by construction (asserted); a molecule whose pool has fewer than 12K valid conformers is counted, and arms use the blocks that exist | Hard |
| V-E5-10 | Validation set built differently from the test set | `val_subset.build` applied to the **test** split must reproduce `test_smiles.csv` / `test_mols.pkl` (row count, n_conformers, corrected_smiles) for ≥ 99 % of rows — **PENDING Q-P2-010** | Hard (golden) |

## 5. Tests

All in `flexitors/tests/` (pytest), CPU, `torch.set_num_threads(1)` as `tools/tests/golden_dump.py` does (multi-threaded
scatter is non-deterministic at ~2e-6). Local where the data allow; the rest as a CPU sbatch on gnode118.

### 5.1 Unit tests (one file per module; the IDs of §4 they enforce)
| File | Checks |
|---|---|
| `test_internal.py` | atan2 dihedral vs `GetDihedralRad` (1e-9); `td_quadruples('td')` equals `standardization.get_torsion_angles` on 100 QM9 molecules and is aligned with `edge_mask` order; degree-4 centre set equals `lgeom.py:232` (V-E3-7) |
| `test_jacobian.py` | W vs finite differences; rigid basis spans the 6 rigid motions (V-E1-2) |
| `test_projector.py` | EXP-1 §5.3 list (idempotent, symmetric, W·P_W ≈ 0, J_rig·P_W ≈ 0, rank 3n−6−m, m = 0 works); dropped rows on near-linear quadruples; linear molecule rank m'+5; the P_L test "P_L·J_τᵀ ≈ 0" kept as the documented negative example (V-E1-1, -6) |
| `test_invariance.py` | drift < 1e-3 rad at 0.05 Å and step² scaling; TD torsion step changes no bond length/angle (> 1e-5 fails; E-JOINT-026 regression) (V-E1-3) |
| `test_constraints.py` | enantiomer flagged; bond stretch back-tracked then skipped after 3 halvings; GT conformers pass; chi_min excludes planar centres; counters add up (V-E1-7, -9) |
| `test_parity.py` | §4 V-E1-13 |
| `test_transform.py` | same Δτ on both endpoints; terminal bonds kept; geometry RNG separate (V-E1-10, -8) |
| `test_losses.py` | `td_loss` equals `training.py:25-31` on a batch; `w_geom = 0` skips the geometry term; NaN guard |
| `test_reverse.py` | hooks edit only the per-edge update; `z` drawn on the full edge set; m = 0 molecules not skipped with geometry on (V-E1-12) |
| `test_torsion_types.py` | EXP-3 §6 tests 1–2 (TD rule; φ_b on (E)/(Z)-but-2-ene and oximes with reference-atom swaps); structural set identical under both RDKit versions (V-E3-1, -6) |
| `test_stereo_guard.py` | sign_b on 200 bonds; K1 frozen torsions bit-identical; fold range and involution; violation counts (V-E3-2, -13, -14) |
| `test_k1_equivalence.py` | update-mask K1 equals `edge_mask` removal for the model scores of the other bonds (V-E3-3) |
| `test_stereo_eval.py` | geometric label equals RDKit's on molecules where RDKit perceives the bond; GT→generated atom map (V-E3-5) |
| `test_select.py` | EXP-5 §9 (exact output size, subset, q = 0 FPS = pure FPS, deterministic); degeneracy flag; static no-GT test (V-E5-5, -7) |
| `test_budget.py` | NFE = Σ samples × steps; `assert_matched` rejects 2 % (EXP-5 §9) |
| `test_td_backend.py` | `draw(n)` = two `draw(n/2)`; ETKDG determinism across runs and thread counts (V-E5-3) |
| `test_hypergeom.py` | closed form equals the MC mean over 20k subsets (± 3 SE) on 5 molecules; at s = N equals the full value (EXP-5 §9 `test_curves.py`) |
| `test_selection_guard.py` | test- or ORACLE-tagged inputs raise (V-X1) |
| `test_leakfree_ref_half.py` | V-E1-18 on a synthetic eval.pkl with known halves (pattern `tools/tests/test_leakfree.py`) |
| `test_floor_fix.py` | V-E1-19 on a mocked cost matrix |
| `test_sbatch_lint.py` | mem/CPU ≤ 5000 MB; `set -euo pipefail`; packed failures propagate (V-X7) |

### 5.2 Golden baseline equality (must pass before any submission)
| ID | Equality | Reference |
|---|---|---|
| G1 | `reverse_loop` without hooks, legacy RNG = TD `sample()` on 5 molecules: positions identical (max abs diff 0, CPU) | `sampling.py:116-246` |
| G2 | `TDSampler(mode='legacy')` = `generate_confs.py --seed s --confs_per_mol n` on 3 molecules (RMSD < 1e-4 Å, EXP-5 §9) | `generate_confs.py` |
| G3a | `FlexiTorsScoreModel(geometry_head=True)` under `manual_seed(0)`: base weights identical to `get_model`; `edge_pred` identical on a fixture; released checkpoint loads strictly into the base | `score_model.py`, `tools/tests/golden_dump.py` M1 |
| G3b | A3: 20 training iterations with `w_geom = 0`: base parameters and torsion losses identical to TD's `train_epoch` with the same seeds (CPU) | EXP-1 §4 control 4, §5.3 `test_golden_td.py` |
| G4 | `metrics` reproduces `R0_base_seed0/summary.txt` (AMR-R 0.1752, COV-R 89.0293, 65 failures) | EVAL §10 |
| G5 | `stats` reproduces `contrasts.csv:50` (S3 vs CR +0.00323 [+0.00127, +0.00523]) | EVAL §10 |
| G6 | `stereo_eval` reproduces R0 65/526 | EXP-3 §6 test 7 |
| G7 | `rings_eval` reproduces ETKDG 49.6 % and MMFF 35.1 % on the ≥ 4-ring seeds of `l_error.csv` | D-002, E-JOINT-004 rev. |
| G8 | `analyze_leakfree.py` without `--ref_subsample` byte-identical outputs; with s = full reproduces `ref_rdkitL` | D-001 |
| G9 | fixed `local_structure_analysis.py`: n_GT > 3 rows identical to `local_structure_test.csv` on a 30-molecule re-run | D-003 |
| G10 | census: 2,880 TD-rotatable bonds, 108 double in 102 molecules | V5 Part 1 summary |
| G11 | `strata` reproduces the "by n_rot_heavy" bins of `R0_base_seed0/breakdown.log`; scaffold-novel count 87 | EVAL §10 |

### 5.3 20-molecule smoke on gnode118 (one sbatch, `experiments/smoke_stageA.sbatch`, ≈ 0.3 GPU-h)
Fixed list of 20 QM9 test molecules chosen to cover: rigid (n_rot_heavy = 0, with and without methyl), ≥ 4-ring with
rotatable bonds, fused/bridged, a triple bond, an allene or linear fragment, ≥ 2 stereocentres, a specified C=N, a C=C
stereo bond, an unspecified stereo DB, K = 1 and K = 30. For each new flag: geometry training 20 iterations (loss
finite, both diagnostics logged, s/iter recorded); joint sampler τ-frozen and joint (counters, drift, validity);
Gate-0 evaluation path end to end on the 20 molecules; Exp 3 arms O-paired, K1, K1-wrong, K1-canonical, K2-inf,
F-filter, F-flip (guard counts, splice check V-E3-4); Exp 5 pool (N_i columns, budget.json), selection, PG c = 2 with
one batch per molecule; every analysis script on the smoke outputs (dry run; asserts the declared families). The
smoke also checks provenance files and `population.txt` exist for every output.

## 6. Dead-code audit

Rule (PLAN.md): no dead code, audit before each push. Procedure for the implementer and both reviewers:
1. `vulture flexitors/ --min-confidence 60` and `coverage run -m pytest flexitors/tests` + the smoke: every public
   function is covered by a test **or** called by a Stage A CLI; uncovered public code is removed or justified in
   `round5/IMPLEMENTATION.md`.
2. Every CLI flag is read somewhere and exercised by the smoke; a flag with no consumer is removed.
3. **Not created in Stage A** (would be dead code now): K2-train reflected score and its σ table, `ConstrainedNoiseTransform`,
   `stereo_edges.py` (K3), DAgger, V3a/V3b, `rings.py` CP/NeRF/SLSQP, Track-D controller, gain model, signals S1/S2/S4,
   `torsion_tangents` in the package (the P_L negative control lives in `tests/helpers.py`).
4. Copy audit: no TD function is copied into `flexitors/` except `td_loss` (verbatim, required by EXP-1 §5.1) and the
   reverse loop (needed for hooks; G1 proves equality). `grep` for duplicated bodies of `modify_conformer`,
   `perturb_batch`, `get_torsion_angles`, `clean_confs`.
5. Existing dead or misleading code noticed in the baseline (reported, not changed — TD is frozen):
   `torus.py` defines `p` twice (`:6-10` table builder, `:49-57` lookup; the second shadows the first after the table
   is built), and draws unseeded `np.random` at import (`:66-70`); `generate_confs.py:168` uses the global `smi_idx`
   inside `sample_confs`. None of these change Stage A numerics once `td_compat` imports TD before seeding; G1/G2
   guard it.
6. Prototype ports (`round5/ledger/*/select.py`, `curves.py`, `stereo_census.py`, `ez_flip_analysis.py`) are ported
   with tests, and the prototypes stay as the research record (not imported).

## 7. Stage A schedule and GPU budget

Basis: 11.7 GPU-h per 100-epoch QM9 training (E-JOINT-008; 377.7 s/epoch logged); one 1× QM9 test pass = 14 min wall
in one job (E-SAMPLE-053), i.e. ≈ 0.24 GPU-h per Σ2K conformers; packed inference 15–25 min per run. Inference
estimates below are **unpacked** (conservative); n_rot = 0 molecules use no model, so real GPU use is lower.

| # | Item | GPU-h (at TD speed) | GPU-h (ceiling) | Gate before it |
|---|---|---|---|---|
| 0 | CPU-1 ref_half, CPU-2 floor fix, CPU-3 Exp 3 Step 0, M0, val subset, E/Z target table, all goldens | 0 | 0 | — |
| 1 | 20-molecule Stage A smoke (§5.3) | 0.3 | 0.3 | goldens G1–G11 green |
| 2 | Exp 1 one-epoch smoke (EXP-1 §5.2) | 0.12 | 0.18 | CPU-1 done (EXP-1 §3.6); M0 not killed |
| 3 | Exp 1 M1: 3 σ_b × 30 epochs, full QM9 train | 10.5 | 16.0 | item 2 measured; ask restated to the orchestrator |
| 4 | Gate-0 evaluation: validation strata × 3 σ_b, then test once | 0.5 | 0.6 | item 3 |
| 5 | Exp 3 QM9: O-paired 3 seeds (0.75) + K1, K1-wrong, K1-canonical, K2-inf, F-filter on affected molecules × 3 seeds (≈ 0.9) | 1.7 | 2.0 | CPU-3 |
| 6 | Exp 5: validation pool 4K (0.5); test pool max(128, 12K) ≈ 10.1 × Σ2K (2.4); PG tuning 5 configs at c = 2 on 500 validation molecules (1.2); PG test c = 1 and c = 2 × 3 replicates (2.2) | 6.3 | 6.8 | validation selection hashed before the test pool is scored |
| | **Core Stage A total** | **19.4** | **25.9** | |
| 7 | Exp 3 DRUGS (released checkpoint): 20-molecule timing, then O + K1 + K1-wrong + K2-inf on the 141 stereo test molecules | ≤ remainder of 25 | — | user approval of the 25.3 GB download (EVAL §9 D0); QM9 Exp 3 done |

If item 2 shows the ceiling is needed, the core exceeds 25 GPU-h by ≈ 1. Cut order (pre-declared): (1) DRUGS moves to
a follow-up ask; (2) PG tuning to 3 configurations (−0.5); (3) K1-canonical to 1 seed (−0.15). Never cut: goldens,
smokes, the three σ_b runs, O-paired, K1-wrong (negative control), P1 and P6r (NFE controls).

Wall-clock (4 shared GPUs on gnode118):
- **Days 1–3 (CPU only).** Package core + tests; G1–G11; CPU-1/CPU-2/CPU-3 on gnode118 CPUs; M0; validation subset and
  E/Z table. Consult answers are folded in as they arrive; PENDING defaults are used otherwise and flagged in
  `IMPLEMENTATION.md`.
- **Day 4.** Smoke sbatch; Exp 1 one-epoch smoke; restated M1 ask; Exp 5 validation pool on GPU 3.
- **Days 4–5.** M1 on GPUs 0–2 (≈ 3.5 h each at TD speed, ≤ 5.3 h at the ceiling). GPU 3: Exp 3 QM9 arms, then the Exp 5
  test pool (generated blind: not scored until the validation decision file is hashed), then PG.
- **Day 6.** Gate-0 decision on validation → test once; Exp 5 selector decision on validation → test once; Exp 3
  analysis; report with both failure conventions, CIs, seed SDs, NFE and CPU-s columns. DRUGS only if approved and
  budget remains.

Hand-off artefacts per item: `results/<exp>/<run>/{confs.pkl, eval.pkl, budget.json, provenance.txt,
population.txt}`, decision files (`gate0_decision.json`, `exp5_selector_decision.json`) with SHA-256, and the
pre-declared analysis outputs.

## 8. Consult questions filed

| ID | To | Spec section | Decision at stake | Default used here |
|---|---|---|---|---|
| Q-P2-001 | JOINT | EXP-1 §2.2 (W rows), §5.1 `jacobian.py` | which quadruple per rotatable bond (TD `n0[0]` may be H) and what to do with near-linear quadruples | TD rule; drop rows with min sin θ < sin 10°, count; heavy-atom drift diagnostic |
| Q-P2-002 | JOINT | EXP-1 §2.2 (first order only), §5.3 | measure second-order drift only, or restore the m dihedrals exactly after each geometry step | measure only; restoration as an ablation flag |
| Q-P2-003 | JOINT | EXP-1 §2.2 training steps 4–5, §2.8 | L target when the matched pair has residual torsion mismatch; reject or keep x_t samples that flip chirality in V2 | spec target + diagnostics; count, do not reject |
| Q-P2-004 | JOINT | EXP-1 §5.2 M0 kill criterion, M1 GO rule, §4 control 8 | M0 population (vacuous on the rigid stratum); validation vs test for σ_b and GO; ring-dihedral metric pairing | m ≥ 1 ring molecules for the kill test; validation decides; paired metric primary, best-over-GT reported |
| Q-P2-005 | JOINT | EXP-1 §4 controls 3–4, §5.2 M2, §6 | whether DAgger's extra training epochs need an epochs-matched control arm (Stage B) | record training GPU-h per arm now; no extra arm |
| Q-P2-006 | CHEM (cc JOINT) | EXP-1 §2.8 "E/Z", §5.4 Q-*-002; EXP-3 §3.2 | how V2 inherits the Exp 3 E/Z constraint | Stage A measures only |
| Q-P2-007 | CHEM | EXP-3 §3.1 targets, §4 primary endpoint, §8 | version-independent "specified" set and target source; primary E/Z label (geometric vs RDKit 3D) | committed target table (pinned RDKit); geometric label primary, RDKit labels in both versions reported |
| Q-P2-008 | SAMPLE | D-101; EXP-5 §5 S0, §11 K1 | pool size (fixed 128 vs max(10 × 2K, 128)); whether the uncapped headroom re-analysis (ORACLE, CPU) runs in Stage A | max(128, 12K) with the first 128 columns K-free; re-analysis runs on CPU, labelled ORACLE |
| Q-P2-009 | SAMPLE | D-102; EXP-5 §4.4, §5 Track P, §6, §11 K3 | selector candidate set and pre-declared validation rule; FPS start; Stage A Holm family without Track D; NFE-matched PG arms and PG tuning budget | as §3.4 |
| Q-P2-010 | SAMPLE | EVAL §1 (validation subset) | how `corrected_smiles`, `n_conformers` and the GT cleaning are built for validation molecules | builder validated by reproducing the test files |
| Q-P2-011 | SAMPLE (EVAL owner) | EVAL §3; EXP-1 §3 row 5 | how geometrically invalid conformers (broken connectivity) enter AMR/COV | keep in primary; +∞ columns as secondary; counts reported |

EXP-1 §5.4 also asks for Q-*-001 (σ_b, w, clock): I use the spec defaults (σ_b ∈ {0.05, 0.15, 0.3} Å, w ∈ {0.1, 1},
shared vs independent clock as specified) and propose no change, so no source is needed; and Q-*-003 (DAgger
re-matching cost), which needs round-1 logs that only the orchestrator has — not filed by me.

## 9. Summary of what makes this plan hard to fool
- Every gate and every choice reads validation data through one guard; ORACLE data are path-segregated (V-X1).
- Arms are paired by construction (per-molecule RNG streams, noise on the full edge set, hooks that edit updates
  only), so differences are not RNG artefacts (V-E3-4).
- Each new mechanism has a golden or negative control: G1–G11, P_L as the negative control of P_W, K1-wrong for K1,
  P1/P6r for selection, ETKDG + MMFF / + xTB beside Gate 0.
- The two ways the geometry step could quietly do torsion work (H-terminated quadruples, accumulated second-order
  drift) are measured on heavy-atom dihedrals, not only on the quadruple the projector fixes (V-E1-4, V-E1-5).
- Version-dependent chemistry perception never defines a constraint or a primary endpoint (V-E3-6).
- Budget statements are K-free where the dispute requires it, NFE-matched where the dispute requires it, and the
  remaining K use (output size 2K) is the benchmark's own (D-101, D-102).
