# Round 2, coding agent 3: implementation plan for S0-S6 (focus on throughput)

Written 2026-10-06. I worked alone and did not read `code_plan_1.md` or `code_plan_2.md`. The spec is
`round2/SHORTLIST.md` (arms S0-S6, fixes F1-F6). I used `verify_A.md` and `verify_B.md` for corrections, and
`research_A.md` (evidence E*) and `research_B.md` (evidence `B#n`, numbered as in verify_B §1) for evidence.

**Conventions**
- Code paths are relative to `torsional-diffusion/` unless they start with `tools/` or `slurm/`.
- Line numbers refer to the current `main` (08b5987).
- **UNMEASURED** marks a number I could not ground in a local log. Section 1.1 gives the cluster commands that
  ground it in 10 minutes.
- No code was changed. This document is the plan only.

---

## 0. Summary

1. **Inference is CPU-bound, not GPU-bound.** Pack 3 generate+evaluate processes on each GPU, and group every test
   condition for a model into one job ("eval sets").
   - Cost: about 218 inference runs in about 24 GPU-slot-h, against about 55 GPU-h unpacked.
   - No sampler change is needed, so the round-1 per-run RNG paths stay valid.
2. **Training is GPU or launch-bound once the loader has 3 workers.** The round-1 jobs asked for 10 CPUs with 8 loader
   workers (`slurm/ablation_train_array.sbatch:7,70`), and that is what starved the node.
   - Plan: 4 CPUs per training job with `--loader_workers 3`. A per-epoch `data_wait` timer proves it is enough
     (gate G-loader).
   - Two trainings per GPU is a measured gate (G-coloc), not the default.
3. **Data caches are shared.**
   - `cache_raw` serves B1 and S2 (it already exists).
   - One new `cache_rematch_paired` serves both S3 and S4. It holds the CTRL_rematch matched L plus the
     Kabsch-aligned GT L of the same conformer (F1).
   - `cache_mmff` serves S5.
   - One CPU builder writes every S1 seed pickle and the F5 "achieved-L" table in one pass.
   - On-the-fly jitter, mixing and interpolation each add one tensor op per sample. Precomputing would only cut
     augmentation diversity, so it is not worth doing (§1.3).
4. **CPU ordering.** The S5 MMFF standardization is not on the critical path: S5 trains in wave 3, at T+26 h.
   - It runs on only 4 cores from T0.
   - The S1 seed builder and the paired featurization are on the critical path, so they get cores first.
5. **Fail fast.**
   - Before any GPU is used: local unit tests (local Python has `rdkit 2026.03.6` and `torch 2.14.0+cpu`, but no
     PyG or e3nn), then a cluster CPU smoke test (20 molecules, 1 epoch, every arm).
   - Inside each job: abort after 200 iterations if the it/s is below a threshold, and on any non-finite loss.
   - Training is made resumable (true resume from `last_model.pt`).
6. **Timeline.** About T+37 h to the last training eval and T+39 h with the final CPU analysis. That is about
   146 GPU-slot-h and about 650 allocated CPU core-h (≈ 18 cores on average).
7. **S4 fits this round, gated.** S3 already needs the paired cache and the alignment, so S4 adds only about 40 lines:
   the λ embedding, 3 flags and the sampler hook.
   - It trains in wave 2 (T+15 h). If its smoke gate fails by then, S5 takes wave 2 and S4 moves to round 3.
   - No GPU idles either way.

---

## 1. Throughput model (what limits what)

### 1.1 What the local files say, and what to measure first (M0, 10 min, before T0)

| quantity | value | source |
|---|---|---|
| training run, 100 epochs incl. 1-3 evals | ~10.7 h | `RESULTS_QM9.md:102`, BRIEF |
| inference + eval run | ~15 min | BRIEF |
| train molecules | ~106k | `RESULTS_QM9.md:91` (featurize "106k molecules") |
| serial featurization | ~8 min | `RESULTS_QM9.md:91` |
| GPU lost to CPU contention in round 1 | ~10 h | `RESULTS_QM9.md:95` |
| node limits | 48 cores, 5 GB/core memory cap | `RESULTS_QM9.md:92` |
| `--mode run` floor | 0.3 s per GT conformer per core | `review/cross_validation.md:102` |
| `--mode test` floor | 1.2 s per GT conformer per core | `review/cross_validation.md:102` |
| analysis job 1089 (5 run-floors + breakdowns + geometry + stereo + paired tables) | 38 min on 16 cores | `cluster_sync/logs/tordiff_analysis_1089.log:2,399` |

Derived:
- 106k / 32 ≈ 3.3k iterations per epoch.
- At about 9.9 h per 100 epochs that is about 6 min per epoch, or about 9-10 it/s. **UNMEASURED directly.**
- GPU memory per training is **UNMEASURED**. No nvidia-smi output was synced.

**M0 commands** (run on gnode118 with `srun -n 1 -c 1 -w gnode118 ...`; read-only):
```bash
sacct -j 1036 --format=JobID%22,Elapsed,TotalCPU,MaxRSS,AllocCPUS,State     # training array: CPU use = TotalCPU/Elapsed
grep -ho '[0-9.]*it/s' $LOGS/tordiff_qm9_trainabl_1036_0.log | tail -5      # train it/s (tqdm)
grep -h 'Elapsed (wall' $RES/qm9_default/*/generate.log | head               # /usr/bin/time -v (common.sh:44,104)
for d in $RES/qm9_default/R0_base_seed0; do stat -c '%y %n' $d/generate.log $d/confs.pkl $d/eval.pkl; done  # eval duration
sacct --name=tordiff_std --format=JobID,Elapsed,TotalCPU,AllocCPUS           # rematch standardization cost -> S5 estimate
ls -la $QM9_DIR/cache_* ; sinfo -N -n gnode118 -o '%C %e'                    # cache sizes; free cores (A/I/O/T) and memory
```
- The numbers below use the BRIEF figures. M0 can only move the hour boundaries; the wave structure stays.

### 1.2 Training: where the time goes

Per sample (`utils/dataset.py:189-193`, `:24-43`) the CPU work is:
- `copy.deepcopy(data)`, which also copies the RDKit mol and up to 30 position tensors;
- one `random.choice`;
- `modify_conformer`: one scipy rotation per rotatable bond (`utils/torsion.py:57-75`).

That is roughly 1 ms per sample, so 3 loader workers deliver about 3k samples/s, or about 90 batches/s. The GPU
step runs at about 10 it/s. So **3 workers are about 10x more than enough**, and the round-1 `--loader_workers 8`
(`slurm/ablation_train_array.sbatch:70`) only competed with the single-threaded main process. That main process
launches every kernel, and `.cpu()` syncs it each step (`utils/training.py:20-24`).

The 1 ms figure is UNMEASURED. Gate G-loader measures it directly with the new `data_wait` timer.

### 1.3 Do the new augmentations add CPU?

| arm | per-sample extra work | cost relative to deepcopy + rotations | precompute? |
|---|---|---|---|
| S2 jitter | one `torch.randn_like(pos)` on ~18×3 values | < 2 % | no: a fresh draw each epoch is the point of the augmentation (E8, E9, E10; B#24, B#26) |
| S3 Bernoulli | one `random.random()` and a tensor pick | ~0 | no |
| S4 interpolation | one fused `(1-λ)a + λb`. The Kabsch alignment (the only expensive part, an SVD) is done **once at featurization** | ~0 | alignment yes (in the cache), λ no |

The extra cost of the paired cache is memory, not CPU.
- `pos_gt` doubles the position lists, and `get()` deep-copies them (`utils/dataset.py:193`).
- The transform deletes `pos_gt` before collation, so batches do not grow.
- Check the RAM with M0 (`MaxRSS` of round-1 jobs, and `ls -la cache_rematch*`). At 4 CPUs the 5 GB/core cap
  allows 20 GB. If MaxRSS × 1.5 > 20 GB, use 5 CPUs.

### 1.4 Two trainings on one 24 GB GPU?

- **Memory.** The model is small (ns 32, nv 8, 4 layers; `slurm/common.sh:42`, `slurm/ablation_train_array.sbatch:71`).
  QM9 batches of 32 molecules are about 600 atoms. Memory should be ≪ 12 GB, but it is UNMEASURED.
- **Throughput.** Without MPS, two processes time-slice the GPU. That gives a gain only if each process alone leaves
  idle gaps (launch-bound e3nn). Unknown.
- **Gate G-coloc** (20 min, head phase, GPU1):
  - Run the CTRL recipe on `cache_rematch` for 300 iterations alone, then two copies at once, with an `nvidia-smi`
    sidecar logging every 10 s.
  - **Decision:** co-locate only if (a) the aggregate it/s is ≥ 1.6× solo, (b) peak memory is ≤ 10 GB per process,
    and (c) `sinfo` shows ≥ 8 idle cores beyond the 4-per-run budget.
  - If yes, Plan B (§5.3) moves 8 runs into wave 1 and cuts the makespan to about 28 h. Otherwise use Plan A.

### 1.5 Inference: pack it

- Each run is about 15 min (BRIEF). The work is RDKit ETKDG embedding (`generate_confs.py:63-68`), CPU torsion
  updates for every conformer at every step (`diffusion/sampling.py:219`, `utils/torsion.py:78-103`), and the
  `evaluate_confs.py` RMSD pool (`:130-141`).
- The GPU sees one molecule at a time, with K = 2L ≈ 15 conformers per batch. It is idle most of the time.
- **Packing:** 3 concurrent `gen_eval` per GPU job, `OMP_NUM_THREADS=1`, `EVAL_WORKERS=2`. While one process
  evaluates on CPU, the others use the GPU.
- **Grouping:** one job per model runs all of that model's conditions ("eval set"). Model loading, env activation and
  `cuda_check` then happen once per model, not once per condition.
- A multi-condition sampler (several conditions in one batch) would be faster still. I reject it for round 2: it
  changes the RNG consumption per run and needs sampler edits for a gain the packing already captures.

---

## 2. Shared infrastructure (used by several arms)

### 2.1 F1: Kabsch alignment and interpolation. New file `utils/lgeom.py` (repo side; no argparse, no PyG import)

```python
import numpy as np

def kabsch_align(P, Q):
    """Return P (n,3) rigidly moved onto Q (n,3); rows = same atoms in the same order. Proper rotation (det=+1)."""
    p0, q0 = P.mean(0), Q.mean(0)
    H = (P - p0).T @ (Q - q0)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (P - p0) @ R.T + q0

def kabsch_rmsd(P, Q):
    return float(np.sqrt(((kabsch_align(P, Q) - Q) ** 2).sum(1).mean()))

def interpolate_L(X_rd, X_gt, lam):
    """F1: all-atom Kabsch of GT onto the matched RDKit conformer, then linear interpolation (x_lambda)."""
    return (1.0 - lam) * X_rd + lam * kabsch_align(X_gt, X_rd)

def ring_acyclic_sets(mol):
    """bond / angle index lists split into ring vs acyclic, with the definition of
    tools/local_structure_analysis.py:200-209 (an angle is a ring angle iff BOTH of its bonds are ring bonds)."""
    ...
def l_metrics(mol, X, X_ref):
    """F5: bond-length RMSD (A) and angle RMSD (deg) of X vs X_ref, for {all, heavy} x {all, ring, acyclic}."""
    ...
```

- **Why this is needed.** `standardize_confs.py:108` aligns `REMOVE_HS` copies (`:33`), so the stored conformer
  (`:116`) is not aligned by that line (F1).
- **Additional finding (verify on the cluster's RDKit).**
  - `optimize_rotatable_bonds` returns its optimum (`utils/standardization.py:42-44`), but `standardize_confs.py:106`
    discards the return value.
  - The stored conformer therefore holds the dihedrals of the **last** DE objective evaluation. The objective also
    moves the probe with `AlignMol` (`utils/standardization.py:61-66`, `RMSD = AllChem.AlignMol` at `:12`).
  - I checked locally (RDKit 2026.03.6) that `copy.copy(mol)` in `apply_changes` (`:23-26`) gives an
    **independent** conformer, so the optimum is really lost.
  - Every matched pickle (round 1 and round 2) shares this, so it is not a confound between arms. It does mean the
    stored conformer is approximately all-atom aligned (from the last evaluation), not unaligned. The explicit Kabsch
    makes this moot.
  - Recommendation: report it, and do **not** fix it this round. Fixing it would break the CTRL_rematch pairing.

### 2.2 Paired cache for S3/S4: `cache_rematch_paired`

The data source is the existing `standardized_pickles_rematch` (the CTRL_rematch data, F4) plus the raw GEOM pickle of
the same molecule.

**`utils/dataset.py`**
- `ConformerDataset.__init__` (`:51-52`): add `pair_gt=False` and store it.
- `construct_loader` (`:260-266`): pass `pair_gt=getattr(args, 'pair_gt', False)`.
- `filter_smiles` std branch (`:128-141`): it knows the file stem `smile`. Change `:172` to
  `self.featurize_mol(mol_dic, gt_name=smile if self.pair_gt else None)`.
- `featurize_mol` (`:200-241`), inside the conformer loop after the canonical check (`:219-220`):
```python
from utils.lgeom import kabsch_align, kabsch_rmsd
def _conf_key(c):  # GEOM conformer identity; geom_id is UNVERIFIED for QM9 -> energy tuple fallback
    return c.get('geom_id', None) or (round(c.get('totalenergy', 0.0), 6), round(c.get('relativeenergy', 0.0), 6))

# before the loop
if gt_name is not None:
    raw = self.open_pickle(osp.join(self.root, gt_name + '.pickle'))
    gt_by_key = defaultdict(list)
    for c in raw['conformers']:
        gt_by_key[_conf_key(c)].append(c['rd_mol'])
    pos_gt = []
# in the loop, after pos/weights would be appended:
if gt_name is not None:
    X_rd = mol.GetConformer().GetPositions()
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    best = None
    for g in gt_by_key.get(_conf_key(conf), []):
        if g.GetNumAtoms() != mol.GetNumAtoms():
            continue
        X_gt = g.GetConformer().GetPositions()
        r = kabsch_rmsd(X_rd[heavy], X_gt[heavy])
        # pairing check: must reproduce standardize_confs.py:108 (heavy AlignMol, identity map) stored at :117
        if 'rmsd' not in conf or abs(r - conf['rmsd']) < 1e-3:
            best = X_gt; break
    if best is None:
        self.failures['gt_pair_missing_or_mismatch'] += 1
        pos.pop(); weights.pop(); continue
    pos_gt.append(torch.tensor(kabsch_align(best, X_rd), dtype=torch.float))   # F1, stored in the RDKit frame
# after the loop
if gt_name is not None:
    data.pos_gt = pos_gt
```
- **Fail fast.** In `preprocess_datapoints` (`:114-122`), after 1000 molecules, abort if
  `failures['gt_pair_missing_or_mismatch'] > 0.01 × conformers seen`. A wrong pairing key then fails in about
  1 minute, not at the end of a 30-minute featurization.
- **Capping.** The GT half is automatically the *same capped* conformer set as the RDKit half
  (`standardize_confs.py:59-67`). That is the "equal caps" requirement of S3, by construction.
- **Slurm.**
  - `slurm/featurize_qm9.sbatch:29-38`: add `rematch_paired) STD="$QM9_DIR/standardized_pickles_rematch"; EXTRA=--pair_gt ;;`
    and pass `$EXTRA` at `:42-44`.
  - `slurm/ablation_train_array.sbatch:46-55`: add the same case. The cache path is then `cache_rematch_paired`,
    because caches are keyed by path only (`notes/code_concerns.md:19`).
- **Cost.** About 20-30 min on 1 core. This is the std featurization (~8 min) plus raw-pickle reads and one SVD per
  conformer. Expected value; UNMEASURED.

### 2.3 Transform: new flags in `TorsionNoiseTransform` (`utils/dataset.py:18-43`)

```python
def __init__(self, sigma_min=0.01*np.pi, sigma_max=np.pi, boltzmann_weight=False,
             l_noise=0.0, mix_gt_p=0.0, lambda_mode='none'):
    ...; self.l_noise, self.mix_gt_p, self.lambda_mode = l_noise, mix_gt_p, lambda_mode

def __call__(self, data):
    n = len(data.pos)
    # index draw replaces :26-29; random.randrange(n) consumes the RNG exactly like random.choice(seq)
    # (both call _randbelow(n)), so CTRL/B1 stay bit-identical (unit test T2)
    idx = random.choices(range(n), data.weights, k=1)[0] if self.boltzmann_weight else random.randrange(n)
    pos = data.pos[idx]
    if self.lambda_mode == 'uniform':                    # S4: lambda ~ U[0,1], given to the model
        lam = random.random()
        pos = (1.0 - lam) * pos + lam * data.pos_gt[idx]  # pos_gt is pre-aligned (F1, section 2.2)
        data.node_lambda = torch.full((data.num_nodes,), lam, dtype=torch.float)
    elif self.mix_gt_p > 0 and random.random() < self.mix_gt_p:   # S3: Bernoulli(p) GT vs matched RDKit
        pos = data.pos_gt[idx]
    if 'pos_gt' in data:
        del data['pos_gt']                                # keep batches small; data is a deepcopy (:193)
    if self.l_noise > 0:                                  # S2: per-axis Cartesian jitter (F6)
        pos = pos + self.l_noise * torch.randn_like(pos)
    data.pos = pos
    ... # :31-43 unchanged (edge_mask, sigma, torsion update)
```
- **RNG in loader workers.** `torch.randn_like` uses the torch RNG, which PyTorch reseeds per worker. That is
  sufficient on the cluster stack (torch 1.13.1, `slurm/setup_env.sh:76`).
- **`construct_loader` (`:255-256`).** Pass `l_noise=getattr(args,'l_noise',0.)`,
  `mix_gt_p=getattr(args,'mix_gt_p',0.)` and `lambda_mode=getattr(args,'lambda_mode','none')`.
- **Fail-fast assert (`:271`).** If `mix_gt_p > 0` or `lambda_mode != 'none'`, assert
  `hasattr(dataset.datapoints[0], 'pos_gt')`, with the message "featurize VARIANT=rematch_paired".
- **Note.** The same transform is used for val (`:255`), so the val loss of S2/S3/S4 is jittered or mixed too. Best
  checkpoint selection stays within-arm, which is acceptable. It is stated in the results.

### 2.4 `train.py` / `utils/training.py`: resume, timing, fail-fast

**`utils/parsing.py` (after `:29`)**
```python
parser.add_argument('--resume', action='store_true', default=False, help='[r2] continue from log_dir/last_model.pt')
parser.add_argument('--min_its', type=float, default=0.0, help='[r2] abort if < this many it/s after --min_its_after iters')
parser.add_argument('--min_its_after', type=int, default=200)
parser.add_argument('--l_noise', type=float, default=0.0, help='[r2 S2] per-axis Cartesian jitter (A) on the training L')
parser.add_argument('--mix_gt_p', type=float, default=0.0, help='[r2 S3] P(use paired GT L) per sample')
parser.add_argument('--lambda_mode', choices=['none', 'uniform'], default='none', help='[r2 S4]')
parser.add_argument('--pair_gt', action='store_true', default=False, help='[r2] featurize: attach Kabsch-aligned GT L')
# feature block (:32-35)
parser.add_argument('--lambda_embed_dim', type=int, default=0, help='[r2 S4] sinusoidal embedding of lambda (0 = off)')
```

**`utils/training.py:7-34`, `train_epoch`**
- Add a `data_wait` timer: time between the end of one step and the arrival of the next batch.
- Add `if not math.isfinite(loss.item()): raise FloatingPointError`. `loss.item()` is already computed at `:29`, so
  the check is free.
- Add the rate gate:
  ```python
  if min_its and it == min_its_after:
      rate = it / (time.time() - t0)
      if rate < min_its: raise RuntimeError(f'FAIL-FAST {rate:.2f} it/s')
  ```
- Return `loss_avg, base_avg, {'wait': t_wait, 'total': t_total}`.

**`train.py`**
- `:27`: unpack the timings. Print
  `Epoch {e} time {total:.0f}s data_wait {wait:.0f}s ({%})`. This line answers "GPU-bound or loader-bound" in every
  log from now on.
- `:62` (boltzmann path): unpack too.
- After epoch 0: if `train_loss >= base_train_loss`, print `FAIL-FAST: not learning` and `sys.exit(4)`. On round-1
  logs the trained loss is well below the base loss after epoch 0 (UNMEASURED; check M0 logs first, and drop the
  gate if CTRL itself violates it).
- **Resume** (replaces the "move aside" logic):
```python
start_epoch, best_val_loss, best_epoch = 0, math.inf, 0
ck = os.path.join(args.log_dir, 'last_model.pt')
if args.resume and os.path.exists(ck):
    s = torch.load(ck, map_location='cpu')
    model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
    if scheduler and s.get('scheduler'): scheduler.load_state_dict(s['scheduler'])
    start_epoch = s['epoch'] + 1
    best_val_loss, best_epoch = s.get('best_val_loss', math.inf), s.get('best_epoch', 0)
    random.setstate(s['py_rng']); np.random.set_state(s['np_rng']); torch.set_rng_state(s['torch_rng'])
```
- `train()` (`:20-46`): loop `range(start_epoch, args.n_epochs)`. Add `best_val_loss`, `best_epoch` and the 3 RNG
  states to the dict at `:41-46`. Write the checkpoint atomically (`torch.save(..., tmp); os.replace(tmp, ck)`), so a
  kill during a save never corrupts the checkpoint.
- **`slurm/ablation_train_array.sbatch:61-65`.** Replace "move aside and retrain" with: if `last_model.pt` exists, no
  `.train_done` exists, and `model_parameters.yml` matches the current args (compare `l_noise`, `mix_gt_p`,
  `lambda_mode`, `lambda_embed_dim`, `seed`, `std_pickles`), then add `--resume`. Otherwise move aside, as now.

### 2.5 Packed evaluation: eval sets

**New `slurm/r2_evalsets.tsv`.** Fields: `SET | TAG | STEPS | SEED | generate args`. `$VARS` are expanded. Tags
follow the oracle rule: any condition using test GT L ends in `_ORACLE`.
```
# S1 conditions (all GT-derived ones use --seed_confs_cycle, F3)
S1 | r2_etkdg_gtgraph            | 20 | 0 | --seed_confs $R2S/etkdg_gtgraph.pkl --seed_confs_cycle
S1 | r2_mmff_gtgraph             | 20 | 0 | --seed_confs $R2S/mmff_gtgraph.pkl --seed_confs_cycle
S1 | r2_noise0.01ax_ORACLE       | 20 | 0 | --seed_confs $R2S/noise0.01ax.pkl --seed_confs_cycle
S1 | r2_noise0.02ax_ORACLE       | 20 | 0 | --seed_confs $R2S/noise0.02ax.pkl --seed_confs_cycle
S1 | r2_noise0.04ax_ORACLE       | 20 | 0 | --seed_confs $R2S/noise0.04ax.pkl --seed_confs_cycle
S1 | r2_interp0.00_ORACLE        | 20 | 0 | --seed_confs $R2S/interp0.00.pkl --seed_confs_cycle
S1 | r2_interp0.25_ORACLE        | 20 | 0 | --seed_confs $R2S/interp0.25.pkl --seed_confs_cycle
S1 | r2_interp0.50_ORACLE        | 20 | 0 | --seed_confs $R2S/interp0.50.pkl --seed_confs_cycle
S1 | r2_interp0.75_ORACLE        | 20 | 0 | --seed_confs $R2S/interp0.75.pkl --seed_confs_cycle
S1 | r2_ringGT_ORACLE            | 20 | 0 | --seed_confs $R2S/ringGT.pkl --seed_confs_cycle
S1 | r2_acycGT_ORACLE            | 20 | 0 | --seed_confs $R2S/acycGT.pkl --seed_confs_cycle
# extra lines for CTRL_rematch (no gtLcycle in round 1: ablations_train.tsv:17-19 has GT_EVAL=0)
S1CR | r2_gtLc_ORACLE            | 20 | 0 | --seed_confs $QM9_GT_SEED_CONFS --seed_confs_cycle
S1CR | r2_rdkit_premmff          | 20 | 0 | --pre_mmff
# S2 = steps20_seed0 + r2_gtLc_ORACLE + the 11 S1 lines minus ringGT/acycGT  (12 runs)
# S3 = steps20_seed0 + r2_gtLc_ORACLE + etkdg/mmff + interp x4            (8 runs)
S4 | r2_lam0_rdkit               | 20 | 0 | --lambda_inf 0
S4 | r2_lam1_gtLc_ORACLE         | 20 | 0 | --lambda_inf 1 --seed_confs $QM9_GT_SEED_CONFS --seed_confs_cycle
S4 | r2_lam0_etkdg_gtgraph       | 20 | 0 | --lambda_inf 0 --seed_confs $R2S/etkdg_gtgraph.pkl --seed_confs_cycle
S4 | r2_lam0_mmff_gtgraph        | 20 | 0 | --lambda_inf 0 --seed_confs $R2S/mmff_gtgraph.pkl --seed_confs_cycle
S4 | r2_lamX_interpX_ORACLE x4   | 20 | 0 | --lambda_inf X --seed_confs $R2S/interpX.pkl --seed_confs_cycle   (X = 0, .25, .5, .75)
S5 | steps20_seed0_premmff       | 20 | 0 | --pre_mmff
S5 | steps20_seed0               | 20 | 0 |
S5 | r2_mmff_gtgraph             | 20 | 0 | --seed_confs $R2S/mmff_gtgraph.pkl --seed_confs_cycle
S6 | C_steps10_seed{0,1,2}       | 10 | 0/1/2 |
S6 | C_steps50_seed{0,1,2}       | 50 | 0/1/2 |
S6 | E_inf_sigmin_0.005pi_seed{0,1,2} | 20 | 0/1/2 | --sigma_min_inf 0.0157
S6 | D_ode_steps20_seed{0,1,2}   | 20 | 0/1/2 | --ode
S6B1 | <same 4 settings>_gtLc_ORACLE | .. | 0 | --seed_confs $QM9_GT_SEED_CONFS --seed_confs_cycle
```
- The seed-0 S6 lines reuse the existing deferred names (`slurm/ablations_inference_deferred.tsv:7,8,10,12`).

**`slurm/common.sh`**
- Move `trim` here; it is now duplicated in `slurm/ablation_train_array.sbatch:40` and
  `slurm/ablation_inference_array.sbatch:38`.
- Add `export R2S=$QM9_DIR/r2_seeds` and `export R2_EVALSETS=$PROJECT/slurm/r2_evalsets.tsv`.
- Add:
```bash
run_evalset() {   # run_evalset MODEL_DIR SET : every line of SET, at most $PACK in flight on this job's GPU
    local model_dir=$1 set=$2 fail=$TMPDIR/evalset_fail_$$; : > "$fail"
    while IFS='|' read -r S TAG ST SD ARGS; do
        [[ "$(trim "$S")" == "$set" ]] || continue
        while (( $(jobs -rp | wc -l) >= ${PACK:-3} )); do wait -n || true; done
        ( OMP_NUM_THREADS=1 EVAL_WORKERS=${EVAL_WORKERS:-2} STEPS=$(trim "$ST") SEED=$(trim "$SD") \
          gen_eval "$model_dir" "$(trim "$TAG")" $(eval echo "$(trim "${ARGS:-}")") || echo "$TAG" >> "$fail" ) &
    done < <(grep -vE '^\s*(#|$)' "$R2_EVALSETS")
    wait; [[ ! -s $fail ]] || { echo "FAILED tags:"; cat "$fail"; return 1; }
}
```
- In `gen_eval` (`:96-117`), wrap `evaluate_confs.py` with `$TIMECMD` too (`:110`), so the next round has measured
  evaluation times.

**New `slurm/r2_eval_array.sbatch`**
- Settings: `--gres=gpu:1 --cpus-per-task=4 --mem=20G --time=08:00:00 --array=0-N%4`.
- Each line of `slurm/r2_eval_models.tsv` (`MODEL_DIR | SET[,SET]`) becomes one task.
- Each task checks `train_complete` (`slurm/common.sh:122-130`), as `slurm/ablation_inference_array.sbatch:50-53`
  does, then runs `run_evalset`.
- It refuses to start if any `$R2S/*.pkl` it needs is missing. There is no lazy building, unlike
  `slurm/ablation_inference_array.sbatch:54-59`.

**Training lines.** In `slurm/ablation_train_array.sbatch:77-86`, a GEN field starting with `@` means "run this eval
set in-job after training, with `PACK=3`". Old lines behave as before.

### 2.6 S0 and F5 analysis tools (CPU)

- **`tools/paired_compare.py`**
  - Add `--intersection` after `:99`:
    `universe = sorted(set.intersection(*sets))`, where `sets` are the per-file success sets of the ref and of every
    arm.
  - Print the per-file failure count: `len(union) - len(d.index)`.
  - Holm families: run one invocation per family.
    - Primary: union, `--primary MAT-R COV-R@0.5`.
    - Key secondary: `--intersection --primary COV-R@0.1`.
    - Sensitivity: `--intersection --primary COV-R@0.5` (no decisions taken on it).
- **`tools/local_structure_analysis.py`**
  - Move the helpers (`:83-215`, plus `heavy_automorphisms`) into `tools/lsa_lib.py`. The module currently parses
    argv at import (`:62-79`), so the new tools cannot import it.
  - Add `--exclude_multifrag` (skip if `'.'` is in the corrected SMILES or `GetMolFrags > 1`).
  - Add `--de_restarts R`: run `torsion_floor` with seeds `args.seed + r` and keep the minimum.
  - Add `--mode seeds --seed_confs X.pkl`. This is `job()` (`:231-324`) with the ETKDG block (`:246-252`) replaced by
    the condition's seed list. It writes `floor_seed_sym` per GT conformer, the model-free F5 floor of that condition.
- **New `tools/leak_test.py`** (S0 (e), verify_B §4 R2-0(e)).
  - Inputs: a `--seed_confs_cycle` run's `confs.pkl`, the matching seed pickle, `true_mols` and the test CSV.
  - For generated conformer k of a molecule, the source GT is `k mod len(seed_confs[raw_smi])`
    (`diffusion/sampling.py:75`). `confs.pkl` keeps conformer order (`generate_confs.py:162`).
  - Compute heavy-rotor torsions with `relevant_torsions` (`tools/local_structure_analysis.py:108-133`). The
    circular distance to each GT conformer's τ is a minimum over heavy automorphisms (`:140-152`).
  - Statistic: hit = nearest GT is the source. Weights are Boltzmann weights from the raw pickle if
    `$QM9_RAW/<smiles>.pickle` exists (UNVERIFIED mapping), else uniform.
  - Output: per-molecule hit rate minus Boltzmann chance. A paired bootstrap over molecules compares
    B1_gtLc vs CTRL_base_gtLc, with nulls:
    - `A1c_gtL_cycle_sanity` (identity; it must give hit = 1);
    - a new no-model run `A1c_gtL_cycle_random_tors` (`--seed_confs_cycle --no_model`, CPU only).

---

## 3. Per-arm plans

### S0. CPU hygiene and diagnostics (no GPU)

| item | code | data / cache | cost |
|---|---|---|---|
| (a) all paired tables with intersection + failures | `tools/paired_compare.py --intersection` (§2.6) | existing `eval.pkl` on the cluster | ~1 CPU-h |
| (b) clean floors | `--exclude_multifrag`; matched sets: a post-hoc script `tools/floor_summary.py` restricts `floor_gtother` comparisons to `n_gt > 1` (verify_B §3(b)) | existing `local_structure_test*.csv` | minutes |
| (c) optimiser convergence | `local_structure_analysis.py --mode test --limit_mols 100 --de_restarts 2` | — | ~0.5 CPU-h (100 mol × ~7.6 GT × 1.2 s × 2) |
| (d) leak test | `tools/leak_test.py` (§2.6) | `confs.pkl` of CTRL_base/B1 `steps20_seed0_gtLcycle`, A1c sanity; new no-model run | ~1 CPU-h |

- **New script `slurm/analysis_r2.sbatch`.** CPU, 4 cores, resumable per step like `slurm/analysis_qm9.sbatch`. It
  runs (a)-(d), then (after each wave) the per-condition `tcompare` calls (`slurm/analysis_qm9.sbatch:137-157`
  pattern) in union and intersection mode.
- **Tests.** T1 unit test of `--intersection` on two synthetic `eval.pkl` dicts (local; pandas only). The leak test is
  run on the A1c sanity run first: it must give hit = 1.000.

### S1. L-quality dose-response (inference only)

**New `tools/make_r2_seed_pickles.py`** (CPU, Pool). One pass writes all conditions and the F5 table.
- Molecules come from the same CSV/`true_mols` loop as `tools/make_seed_pickles.py:45-65`.
- The GT list is `clean_confs` (`:38-42`), in the **same order** as `test_gt_seed_confs.pkl`. The cycle source
  mapping and `evaluate_confs.py` therefore agree.
```python
def job(item):
    raw_smi, mol_idx, gts = item                        # gts: clean GT conformers, L = len(gts)
    L, base = len(gts), Chem.Mol(gts[0])
    if len(Chem.GetMolFrags(base)) > 1: return raw_smi, {}, [dict(smiles=raw_smi, error='multifrag')]
    assert all(same_atom_order(base, g) for g in gts)   # elements + bond list; else reorder via SubstructMatch
    out, rows = defaultdict(list), []
    # (2) GT-graph ETKDG seeds K = 2L, and the same embeds MMFF-relaxed (F2; == sampling.try_mmff, :21-26)
    emb = Chem.Mol(base); emb.RemoveAllConformers()
    if len(AllChem.EmbedMultipleConfs(emb, numConfs=2*L, randomSeed=args.seed)) == 2*L:
        out['etkdg_gtgraph'] = split_confs(emb)
        mm = Chem.Mol(emb); AllChem.MMFFOptimizeMoleculeConfs(mm, mmffVariant='MMFF94s')
        out['mmff_gtgraph'] = split_confs(mm)
    # (4,5) conformer-matched RDKit seed per GT conformer: the standardize_confs.py:73-106 recipe, but KEEP the
    # optimum returned by optimize_rotatable_bonds (section 2.1 finding) -> less interpolation artefact
    rd = Chem.Mol(base); rd.RemoveAllConformers()
    ok = len(AllChem.EmbedMultipleConfs(rd, numConfs=L, randomSeed=args.seed + 7)) == L
    if ok:
        tors = get_torsion_angles(rd)                   # utils/standardization.py:69
        cost = np.array([[get_von_mises_rms(gts[i], rd, tors, j) if tors else heavy_kabsch(rd, j, gts[i])
                          for j in range(L)] for i in range(L)])
        _, col = linear_sum_assignment(cost)
        for i in range(L):
            m = single_conf(rd, int(col[i]))
            if tors: m = optimize_rotatable_bonds(m, gts[i], tors, popsize=15, maxiter=15)
            X_rd, X_gt = pos(m), pos(gts[i])
            for lam in (0.0, 0.25, 0.5, 0.75):
                out[f'interp{lam:.2f}'].append(with_pos(base, interpolate_L(X_rd, X_gt, lam)))   # F1
            # same mechanism for both subset arms (verify_B R2-5): set_acyclic_local copies ACYCLIC bonds+angles
            # (all atoms incl. H) from src onto base; ring geometry stays from base
            out['acycGT'].append(transfer_acyclic(base=with_pos(base, X_rd), src=gts[i]))   # GT acyclic, RDKit rings
            out['ringGT'].append(transfer_acyclic(base=with_pos(base, X_gt), src=m))        # GT rings, RDKit acyclic
    # (3) GT + isotropic per-axis noise (F6), 2 independent draws per GT conf -> 2L seeds; cycle source = k mod L
    for s in (0.01, 0.02, 0.04):
        rng = np.random.default_rng([args.seed, mol_idx, int(round(s * 1e4))])
        out[f'noise{s:.2f}ax'] = [with_pos(base, pos(g) + rng.normal(0.0, s, pos(g).shape)) for _ in (0, 1) for g in gts]
    # F5: achieved L error of every seed vs ITS source GT conformer (bond/angle, ring/acyclic, heavy/all)
    for cond, seeds in out.items():
        for k, sm in enumerate(seeds):
            src = gts[k % L] if cond not in ('etkdg_gtgraph', 'mmff_gtgraph') else None
            rows.append(dict(smiles=raw_smi, cond=cond, k=k, **(l_metrics(sm, pos(sm), pos(src)) if src else {})))
    return raw_smi, out, rows
```
- `transfer_acyclic` wraps `set_acyclic_local` (`tools/local_structure_analysis.py:196-215`) with
  `bonds_angles(..., heavy_only=False)` (`:83-95`). Treating H as acyclic makes the two subset arms exact complements:
  - ring bonds and ring angles always come from `base`;
  - everything else comes from `src`.
- **Multi-conformer source.** For `etkdg/mmff_gtgraph`, F5 is reported against the nearest GT conformer.
- **Outputs.** `$QM9_DIR/r2_seeds/<cond>.pkl` (`{raw_smi: [single-conformer mols]}`, the format of
  `diffusion/sampling.py:75-80`), `achieved_L.csv`, and `build.log` with embed-failure counts per condition.
- **Populations.**
  - Noise conditions: 996 molecules (the GT-L population).
  - Matched conditions: fewer, because ETKDG fails on cages (`RESULTS_QM9.md` §2.2).
  - Curves are therefore read on the intersection, and failures are reported (S0).
- **Oracle labels.** As in the SHORTLIST endpoints: every `interp*` (including the Hungarian-selected `interp0.00`),
  `noise*`, `ringGT` and `acycGT` is `_ORACLE`. `etkdg_gtgraph` and `mmff_gtgraph` are not: they use the GT graph and
  stereo only, which G1 showed has no effect.
- **Cost.** About 1000 molecules × (3L embeds + 2L MMFF + L² von-Mises costs + L DE at popsize 15 × 15 iterations).
  At the round-1 standardization rate of 0.3-2 s per molecule (`slurm/standardize_qm9.sbatch:23`), that is about
  0.1-0.6 CPU-h, plus F5 metrics. With 4 cores, under 30 min.
- **Optional S1.6 (xTB).**
  - A 30-minute CPU gate job runs `timeout 1800 micromamba create -p $PROJECT/xtbenv -c conda-forge xtb`.
  - If it succeeds, add `--xtb_bin` to the builder. It relaxes the `etkdg_gtgraph` seeds with `utils/xtb.py:44`
    (`xtb_optimize`, per-PID work dir `:5`, so the Pool is safe) into `xtb_gtgraph.pkl`.
  - Cost: about 15k optimizations × 1-3 s ≈ 4-12 CPU-h, run on 4 cores during wave 1, plus 9 inference runs at the
    end.
  - If the gate fails, skip S1.6 and note it.

**F5 floors per condition.**
- `local_structure_analysis.py --mode seeds --seed_confs $R2S/<cond>.pkl` for each of the 11 conditions. About 7.6k
  GT conformers × 3 DE runs × ~0.13 s ≈ 1 CPU-h each, so about 11 CPU-h.
- `--mode run` on CTRL_rematch s0 for each condition: 0.6 CPU-h each.

**Models and runs.** `slurm/r2_eval_models.tsv`:
- 9 lines for CTRL_base s0-2, CTRL_rematch s0-2 and B1 s0-2 with set `S1`.
- CTRL_rematch lines also get `S1CR`.
- 1 line for `$WORK/pretrained/qm9_default` with set `S6`, and 1 line for B1 s0 with set `S6B1`.

Count: 9 × 11 + 3 × 2 = **105 runs**. The full grid is larger than the SHORTLIST's ~15 GPU-h (≈ 60 runs). Packing
brings it to about 11 GPU-slot-h. If CPU is short, drop `interp0.25`/`interp0.75` for CTRL_base first: CTRL_rematch is
the matched control (F4).

**Analysis.**
- Per condition: `tcompare` with ref = CTRL (CB for B1; CR for the S3/S4/S5 arms) and arm = B1, union and
  intersection.
- ε* is read where B1 − CTRL crosses 0 against the achieved bond/angle RMSD from `achieved_L.csv` (F5), not against
  nominal λ or σ.

**Tests.** Local T4-T6 (§4.1). Cluster: the builder on `--limit_mols 20`, then `run_evalset` on 2 conditions × 5
molecules inside the CPU smoke (§4.2).

### S2. B6: noise-augmented GT-L training

- **Code.** `--l_noise` (§2.3). No new cache: it uses `cache_raw`, shared with B1.
- **New lines in `slurm/r2_train.tsv`** (format of `slurm/ablations_train.tsv:2-5`; GEN field `@S2`, GT_EVAL 0):
```
B6_jit0.04ax   | raw | --l_noise 0.04 --min_its 6 | @S2 | 0 | 0
B6_jit0.04ax   | raw | --l_noise 0.04 --min_its 6 | @S2 | 0 | 1
B6_jit0.04ax   | raw | --l_noise 0.04 --min_its 6 | @S2 | 0 | 2
B6_jit0.02ax   | raw | --l_noise 0.02 --min_its 6 | @S2 | 0 | 0      # exploratory, 1 seed
```
  - `--min_its 6` is a placeholder of about 60 % of CTRL's rate. Set it from M0.
- The `ax` in the names states per-axis σ (F6). σ = 0.04 per axis is about 0.057 Å per bond (verify_A §3 jitter
  table).
- **Evidence.** Conditioning augmentation against train-test mismatch: E5, E8, B#23, B#24. Input perturbation is an
  analogy (verify_A E9 note): E9, E10, B#26.
- **Analysis.** Pre-registered:
  - (a) non-inferior to CTRL_base on RDKit L: the upper 95 % CI of AMR-R(B6) − AMR-R(CB) < 0.004 Å;
  - (b) better than CB on `r2_gtLc_ORACLE`.
  - Single-factor control: B1. The B1 jittered-GT controls are the S1 `noise*` runs on B1 s0-2 (verify_A R2A-4(a)).
- **Runs.** 4 trainings plus 4 × 12 = 48 inference runs.
- **Tests.**
  - T2 (bit-identity of the transform when `l_noise = 0`).
  - T3 (with σ_torsion → 0, the per-axis SD of pos_out − pos_in is 0.040 ± 0.003, and the bond RMSD is about 0.057).
  - 20-molecule smoke.

### S3. Mixed-L training (Bernoulli 0.5)

- **Code.** `--mix_gt_p 0.5` and the paired cache `cache_rematch_paired` (§2.2), shared with S4. Equal caps hold by
  construction (§2.2).
- **TSV:**
```
S3_mixL_p0.5   | rematch_paired | --mix_gt_p 0.5 --min_its 6 | @S3 | 0 | 0
S3_mixL_p0.5   | rematch_paired | --mix_gt_p 0.5 --min_its 6 | @S3 | 0 | 1
```
- **Controls.** CTRL_rematch (RDKit end, F4) with its new `r2_gtLc_ORACLE` runs (S1CR), and B1 (GT end).
- **B1 cap confound.** A capped-B1 control is free in code: `rematch_paired | --mix_gt_p 1.0`. It is scheduled only
  if a slot frees (§5.4).
  - First quantify the confound on CPU: the fraction of train conformers lost by the 30-conformer cap is
    `sum(max(0, n_raw - 30)) / sum(n_raw)` over the raw pickles. This is a featurization by-product: log it in the
    paired featurize.
  - If it is below 2 %, a note is enough.
- **Evidence.** E5, E6, E7 (CDM train-test mismatch). E16 is weak (verify_A).
- **Runs.** 2 trainings plus 2 × 8 = 16 inference runs.
- **Tests.**
  - T7: the GT fraction over 2000 draws is 0.50 ± 0.03.
  - `pos_gt` is absent after the transform, and PyG collates a batch of 8.
  - Pairing failures are 0 on 20 molecules.

### S4. λ-conditioned interpolation training

**Model.** The σ embedding enters at `diffusion/score_model.py:188-193`:
- `:188` maps σ to [0, 10000];
- `:189` makes the sinusoidal embedding (`get_timestep_embedding`, `:220-230`);
- `:191-192` concatenate it to the edge attributes;
- `:193` concatenates it to the node attributes.

The input widths are set at `:66` (node) and `:72` (edge). λ goes through the same path:
```python
# __init__ signature (:49-52): add lambda_embed_dim=0 ; store self.lambda_embed_dim
nn.Linear(in_node_features + sigma_embed_dim + lambda_embed_dim, ns)                       # :66
nn.Linear(in_edge_features + sigma_embed_dim + lambda_embed_dim + radius_embed_dim, ns)    # :72
# build_conv_graph, after :189
if self.lambda_embed_dim:
    lam_emb = get_timestep_embedding(data.node_lambda * 10000, self.lambda_embed_dim)      # same [0,1e4] scale as :188
    node_sigma_emb = torch.cat([node_sigma_emb, lam_emb], 1)                                 # :191-193 then carry it
```
- `utils/utils.py:10-18`: pass `lambda_embed_dim=getattr(args, 'lambda_embed_dim', 0)`.
- With 0, the parameter shapes are unchanged, so every existing checkpoint still loads with `strict=True`
  (`generate_confs.py:102`, `train.py:93`).

**Sampling**
- `generate_confs.py`: add `--lambda_inf` (float, default None) after `:51`.
- `:91-92`: add `'lambda_inf'` to the list of CLI keys protected from the training-yaml overwrite. The training yaml
  never contains it, but this guards against future name clashes.
- If `getattr(args, 'lambda_embed_dim', 0) > 0` and `lambda_inf is None`, exit with "λ-model needs --lambda_inf".
- `diffusion/sampling.py:105-109`: add `lambda_inf=None` to the `sample()` signature. After `:168`:
  ```python
  if lambda_inf is not None:
      data_gpu.node_lambda = torch.full((data.num_nodes,), float(lambda_inf), device=device)
  ```
  `model()` returns the same Data object (`:173`, `score_model.py:154-161`), so it persists over the steps.
- `generate_confs.py:146-155`: pass `lambda_inf=args.lambda_inf`.

**Training**
- `--pair_gt` is already in the cache; the run adds `--lambda_mode uniform --lambda_embed_dim 32`. λ ~ U[0,1] per
  sample, per the spec.
- Raised, not applied: the endpoints λ = 0/1 have zero probability mass under U[0,1]. A 20/60/20 point-mass mixture
  would train the test-time endpoints directly. It is a spec change, so it is left to the vote.

**TSV:**
```
S4_lamcond     | rematch_paired | --lambda_mode uniform --lambda_embed_dim 32 --min_its 6 | @S4 | 0 | 0..2  (3 lines)
```
- **Control and analysis.** CTRL_rematch (F4) on RDKit (`steps20_seed0` vs `r2_lam0_rdkit`), and B1 / CR on GT L (all
  ORACLE). The B1 cap confound is noted (§S3).
- The `r2_lamX_interpX_ORACLE` runs give the "matched λ" curve against S1's `interp*` runs on CR and B1.
- **Evidence.** E5 and E6 (amortised conditioning level). E15, E16 and E24 are analogy or weak only (verify_A §1).
  The arm changes data and architecture together; S3 isolates the conditioning (verify_A R2A-2(e)).
- **Runs.** 3 trainings plus 3 × 8 = 24 inference runs.
- **Tests (T8).**
  - Forward and backward pass with `lambda_embed_dim=32`.
  - The output changes when λ goes 0 → 1 on the same input (the embedding is live).
  - The released `qm9_default` and one round-1 CTRL checkpoint load strictly with the new code.
  - `generate_confs.py --lambda_inf 0 --limit_mols 5` runs end to end.
  - A λ-model run without `--lambda_inf` exits non-zero.

### S5. B3: MMFF-matched TD training

- **Code.** None in Python: `standardize_confs.py --mmff` (`:18`, `:83-87`) and `VARIANT=mmff` already exist
  (`slurm/standardize_qm9.sbatch:36`, `slurm/featurize_qm9.sbatch:33`, `slurm/ablation_train_array.sbatch:50`).
- **Slurm change.** Allow `--cpus-per-task` and the xargs `-P` to be set from the environment
  (`slurm/standardize_qm9.sbatch:6,56`), and submit it with 4 cores at T0. It is resumable per worker file (`:51`), so
  a later resubmit with more cores continues.
- **TSV.** Replaces the deferred line `slurm/ablations_train_deferred.tsv:4`; GEN `@S5` holds both ± `--pre_mmff`:
```
B3_match_mmff  | mmff | --min_its 6 | @S5 | 0 | 0..2   (3 lines)
```
- **Controls.** CTRL_rematch with and without `--pre_mmff`. "With" is the new S1CR line `r2_rdkit_premmff` (3 runs).
  "Without" is the existing `steps20_seed0`.
- **MMFF losses.** Sum the last `long_term_log` dict per worker log (`standardize_confs.py:156-162`, key `mmff_error`
  set at `:87`) with `tools/std_log_summary.py`. Compare `Fetched N mols` of `cache_mmff` against `cache_rematch`
  (`utils/dataset.py:121`).
- **Fail fast.** Featurize mmff aborts if it fetches fewer than 95 % of the rematch molecule count.
- **Evidence.** E2, E3; TD p. 26 OMEGA (B#7).
- **Runs.** 3 trainings plus 3 × 3 + 3 = 12 inference runs.

### S6. Sampler checks (low priority)

- **Code.** None. The flags exist: `--inference_steps`, `--sigma_min_inf` (`generate_confs.py:49`) and `--ode`.
  `likelihood` is protected (`:91-92`), so `--ode` alone does not compute divergences.
- **Runs.** 12 on the released model (eval set `S6`) plus 4 on B1 s0 with `_gtLc_ORACLE` (eval set `S6B1`). B1 is
  descriptive, single training seed (verify_B R2-6). Total 16; `C_steps50` costs about 2.5× a 20-step run.
- **Analysis.** `paired_compare` with ref R0 (3 sampling seeds), as in `slurm/analysis_qm9.sbatch:111-126`.

---

## 4. Tests

### 4.1 Local (Windows; `python -c "import rdkit, torch"` → rdkit 2026.03.6, torch 2.14.0+cpu)

- Not available locally: `torch_geometric`, `e3nn`, `torch_scatter`, `torch_cluster`, `spyrmsd`, `rmsd`.
- `utils/standardization.py` imports `rmsd` (`:3`), so T5 needs `pip install rmsd` (pure Python) or a stub.
- Files: `tools/tests/test_r2_local.py`, plain asserts, about 2 min.

| id | test |
|---|---|
| T1 | `paired_compare --intersection` on synthetic eval dicts: the universe is the intersection; failure counts are right; COV@0.1 is computed only on the intersection |
| T4 | Kabsch: a random rotation + translation is recovered (RMSD < 1e-6). A mirrored input gives det(R) = +1 and RMSD > 0. `interpolate_L` at λ = 0 returns X_rd exactly. At λ = 1 its RMSD to GT after alignment is < 1e-6. RMSD to GT decreases with λ. Report the bond shrinkage of x_0.5 on `CCOC(=O)CCN` (two ETKDG embeds, one MMFF-relaxed as fake GT) |
| T5 | Builder on a fake test set: 5 SMILES, "GT" = 3 MMFF-relaxed ETKDG conformers each, with fake `test_smiles.csv` and `test_mols.pkl`. Checks: list lengths (L for interp/ring/acyc; 2L for noise/etkdg/mmff); single conformer per mol; same atom order; per-axis noise SD within 10 % of σ; `ringGT` ring-subset bond/angle RMSD to GT = 0 (exact, base geometry); `acycGT` ring-subset RMSD to GT = that of the RDKit seed; the converse acyclic subsets are reported; `interp0.00` L error equals the matched seed's; multi-fragment molecules are skipped |
| T6 | Cycle bookkeeping: for `noise*` the source of conformer k is `k mod L`, for 2L generated conformers (reimplement `diffusion/sampling.py:75` indexing in the test) |

### 4.2 Cluster CPU smoke: `tools/smoke_test_r2.py`

- Run it with `srun -n 1 -c 4 -w gnode118` and **no GPU**, after the existing `tools/smoke_test.py`. About 15 min.
- It runs from the repo root on the real data.

**Checks**

| id | test |
|---|---|
| T2 | Bit-identity: the old transform (copied into the test) and the new one at default flags give identical `pos`, `edge_rotate` and `node_sigma` for 50 samples under the same seeds. CTRL/B1 semantics are unchanged |
| T3 | S2 jitter statistics (§S2) |
| T7 | S3 mixing (§S3) |
| T8 | S4 model (§S4) |
| T9 | Pairing on 20 molecules: `gt_pair_missing_or_mismatch == 0`; ‖heavy Kabsch RMSD − `conf['rmsd']`‖ < 1e-3. It also prints the raw conformer keys, which answers UNVERIFIED `geom_id` |
| T10 | Resume: `--n_epochs 1`, then `--resume --n_epochs 2`. The epoch counter continues, the Adam `state[...]['step']` continues, and `best_val_loss` is carried over |
| T11 | Fail-fast: `--min_its 1e9 --min_its_after 1` exits non-zero. A planted NaN (monkeypatched loss) raises `FloatingPointError` |

**20-molecule / 1-epoch smoke, every arm**
- Arms: CTRL recipe on rematch, S2, S3, S4. Each uses its own cache path, `$TMPDIR/smoke_<arm>`. A shared path would
  poison the real caches, because caches are keyed by path only.
- For each arm, `train.py --limit_train_mols 20 --n_epochs 1 --batch_size 8 --cache $TMPDIR/smoke_<arm> ...`, then
  `run_evalset` restricted to 2 tags with `--limit_mols 5`, then `evaluate_confs.py --limit_mols 5`.
- Assert that a SUMMARY line exists.
- S5 needs no smoke beyond its featurize count check: its Python path is the CTRL path.

### 4.3 In-production gates

| gate | when | rule | action on failure |
|---|---|---|---|
| G-smoke | T−0.5 h (CPU) | §4.2 all pass | arms that fail are held; the others proceed |
| G-pair | first ~1 min of the paired featurize | pairing failures < 1 % | abort; fix the key |
| G-loader | head, GPU1, 10 min | `data_wait` < 5 % of epoch time with 3 workers | raise to 5 workers / 6 CPUs |
| G-coloc | head, GPU1, 20 min | §1.4 | Plan B (§5.3) |
| G-rate | every training, iteration 200 (~20-40 s) | it/s ≥ `--min_its` | exit 3; the array task shows FAILED in minutes |
| G-learn | every training, end of epoch 0 (~6 min) | train loss < base loss; loss finite | exit 4 |
| G-S4 | T+14 h | T8 passed and the S4 smoke trained | S4 in wave 2; else S5 in wave 2 (§5.4) |
| G-xtb | head (CPU) | installs in < 30 min | otherwise skip S1.6 |

---

## 5. Schedule

### 5.1 Cores (assume 24 free of 48: "24 were in use"; check with `sinfo -o %C` at T0)

| phase | GPU jobs | CPU jobs | cores |
|---|---|---|---|
| head (T+0 to 3.5) | 4 eval jobs × 4 cores (PACK 3) | std-mmff 4; builder 4 (≤ 0.5 h), then F5 floors / S0 4 | 24 |
| waves | 4 trainings × 4 (1 main + 3 loader workers; in-job evals PACK 3 at the end) | std-mmff 4 (until done); analysis 4 | 24 |

### 5.2 Plan A (one training per GPU), hour by hour

Before T0:
- T−6 to T−1 h: write code and run the local tests (§4.1).
- T−1 h: sync and run M0 (§1.1).
- T−0.5 h: CPU smoke (§4.2).
- Then submit the CPU jobs (std-mmff, builder, paired featurize, xtb gate).

| wall clock | GPU0 | GPU1 | GPU2 | GPU3 | CPU side |
|---|---|---|---|---|---|
| T+0 – 0.5 | eval CB s0 (S1) | G-loader + G-coloc (CTRL recipe, 300 it each) | eval released (S6, 12 runs) | eval CB s1 (S1) | std-mmff (4c, ~8-17 h); builder (4c, ≤ 30 min); featurize rematch_paired (1c, ~30 min); xtb gate |
| T+0.5 – 3.5 | eval CB s2, CR s0 | eval CR s1, B1 s0 | eval CR s2, B1 s1 | eval B1 s2, B1 s0 S6B1 | F5 `--mode seeds` ×11 (4c, ~3 h); S0 (a)-(d) |
| T+3.5 – 13.5 | **W1** train S2 jit0.04 s0 | S2 jit0.04 s1 | S2 jit0.04 s2 | S3 s0 | std-mmff → featurize mmff (≈ T+18 at the latest); F5 `--mode run`; xTB seeds (optional) |
| T+13.5 – 14.8 | in-job eval @S2 (12 runs) | @S2 | @S2 | @S3 (8 runs) | analysis_r2 for wave-1 tables |
| T+14.8 – 24.8 | **W2** S4 s0 | S4 s1 | S4 s2 | S3 s1 | — |
| T+24.8 – 25.8 | @S4 (8) | @S4 | @S4 | @S3 | analysis_r2 |
| T+25.8 – 35.8 | **W3** S5 s0 | S5 s1 | S5 s2 | S2 jit0.02 s0 | — |
| T+35.8 – 37.1 | @S5 (3) | @S5 | @S5 | @S2 (12) | — |
| T+37 – 39 | (optional: xTB S1.6 runs, 9 × packed, ~1 h) | free | free | free | final analysis_r2 (union + intersection; F5; ε*) |

- The head runs about 121 inference runs (S1 105 + S6 16). That is 10 eval jobs of 10-14 runs each, packed 3 per
  GPU, at about 1.3 h per job. Three rounds on 4 GPUs take about 3.3 h.
- The SLURM arrays are `r2_eval_array` (`%4`) for the head. Then 3 training sbatch arrays (`r2_train.tsv` lines in
  wave order, `%4`), each submitted with `--dependency=afterany:<previous>`, so that no more than 4 GPUs are in use
  (the round-1 rule, `slurm/ablation_train_array.sbatch:15-16`).

### 5.3 Plan B (G-coloc passes: 2 trainings per GPU)

- W1' (T+3.5): 8 runs: S2 ×4, S3 ×2, plus 2 more.
  - The 2 extra slots go to S4 s0/s1 if G-S4 has already passed.
  - S5 cannot fill them: `cache_mmff` is not ready yet. Otherwise use them for capped-B1 and a spare.
- W2' (≈ T+16): S4 ×3, S5 ×3.
- Each run is about 1.25× slower, but 2 run per GPU.
- Makespan ≈ 3.5 + 2 × 12.5 + evals ≈ 30 h. CPU: 8 × 4 = 32 cores > 24, so Plan B needs `LOADER_WORKERS=2`, 3 CPUs
  per run, and `sinfo` confirming free cores.

### 5.4 Fallbacks (no GPU idles)

- **G-S4 fails at T+14.**
  - W2 = S5 ×3 + S3 s1. `cache_mmff` is needed by T+14.8. std-mmff at 4 cores from T0 finishes by about T+8-17
    (UNMEASURED). If it is late, swap W2 GPU0-2 to S2 jit0.02 + capped-B1 + 1 spare, and put S5 in W3.
  - W3 = S2 jit0.02 + capped-B1 (`rematch_paired --mix_gt_p 1.0`) + 2 free GPUs. S4 then moves to round 3.
- **A training fails G-rate / G-learn.** The array task exits within minutes. Fix it and resubmit that index; it
  resumes (§2.4).
- **Node or time-limit kill.** Resume from `last_model.pt`. At most one epoch (~6 min) is lost.

### 5.5 Budget

| item | runs | GPU-slot-h | CPU core-h (allocated) |
|---|---|---|---|
| training S2 4, S3 2, S4 3, S5 3 | 12 | ~120 (12 × 10 h, BRIEF minus evals) | 480 (12 × 10 h × 4) |
| inference, packed 3/GPU, ~0.33 h each | 105 S1 + 16 S6 + 48 S2 + 16 S3 + 24 S4 + 12 S5 ≈ 221 | ~24 (unpacked: ~55 GPU-h at 0.25 h) | ~90 (UNMEASURED per run; M0) |
| smoke + gates | — | ~1.5 | ~2 |
| S5 standardize mmff | — | 0 | 30-70 (UNMEASURED; scale from the rematch `sacct`) |
| featurize rematch_paired + mmff | — | 0 | ~1 |
| S1 builder (+ optional xTB) | — | 0 | ~1-2 (+4-12) |
| F5 floors (11 seeds-mode + 11 run-mode) | — | 0 | ~18 |
| S0 | — | 0 | ~5 |
| **total** | | **≈ 146** (36.5 h on 4 GPUs; Plan A ends at T+37) | **≈ 630-690** (≈ 18 cores on average) |

The SHORTLIST estimate of 153 GPU-h (with about 25 for inference) is in the same range. Packing pays for the larger
S1 grid.

---

## 6. Recommendation on S4

**Vote: run S4 this round, in wave 2, behind gate G-S4. S5 is the automatic fallback.**
- **Marginal code is small.** The paired cache (§2.2), the alignment (§2.1) and the transform hook (§2.3) are needed
  by S1 and S3 anyway. S4 adds:
  - the λ branch in the transform (5 lines);
  - the model embedding (`diffusion/score_model.py:66,72,189`; about 8 lines);
  - `get_model` (1 line);
  - 2 parsing flags;
  - `--lambda_inf` in generate and sample (about 8 lines).
  It does not change the strict-load path for any existing checkpoint (embedding dim 0).
- **Time is enough.** S4 is not needed until T+14.8. Its smoke test (T8) needs about 10 minutes of CPU.
- **Risk is contained.** A bad S4 costs 30 GPU-h in the worst case (3 × 10 h). G-rate and G-learn turn a broken
  model into a failure within minutes, not after 10 h.
- **Interpretation.** S4 is read together with S3, which isolates conditioning. Both are pre-registered, so running
  them in parallel adds no forking-paths problem.

What would make me vote "round 3": if T9 (pairing) cannot be made to pass by T+3 (the GEOM `geom_id` key is
UNVERIFIED). S3 then also slips, and the arm should not be rushed.

## 7. Points raised against the spec (not silently changed)

1. **CTRL_rematch has no GT-L runs.** `slurm/ablations_train.tsv:17-19` has GT_EVAL 0, but S3/S4 need it at the GT
   end. Added: `S1CR r2_gtLc_ORACLE` (3 runs).
2. **The S1 full grid is about 105 runs, not about 60.** It is affordable only because of packing. A drop order is
   given (§S1).
3. **λ = 0 (Hungarian-matched RDKit seed, ORACLE) is added to the interpolation series.** It is the same-population
   anchor of the curve.
4. **MMFF seed pickle (F2) gets a paired non-MMFF twin (`etkdg_gtgraph`) from the same embeds.** It is the only
   exactly single-factor MMFF contrast on the GT-graph population.
5. **The noise conditions use 2L seeds** (two independent draws per GT conformer). The cycle then never reuses the
   same noisy L twice, and the source stays `k mod L`.
6. **`standardize_confs.py:106` discards the DE optimum** (§2.1). This is a finding, not a fix. It does not confound
   round-2 arms.
7. **S4 λ ~ U[0,1] gives no mass at the test-time endpoints** (§S4). Point-mass mixing is left to the vote.
8. **The 0.004 Å non-inferiority margin for S2 is taken from the SHORTLIST.** With 3 training seeds (SD about
   0.0013; `research_B` §1.1) it is about 3 SD, which is adequate.
