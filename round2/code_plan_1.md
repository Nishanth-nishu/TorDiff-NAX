# Round 2 code plan 1 (coding agent 1): minimal-diff implementation of S0-S6

Written 2026-10-06. Spec: `round2/SHORTLIST.md` (arms S0-S6, fixes F1-F6). Rationale for fixes: `verify_A.md`, `verify_B.md`.
Code line numbers refer to the repo state at commit `08b5987` (`torsional-diffusion/` = upstream 5f713b4 + ablation hooks,
identical in content to the cluster `$REPO` = upstream + `patches/0001-ablation-hooks.patch`). No code was changed while
writing this plan.

Design rule used throughout: **every new behaviour sits behind a new flag whose default is "off", and the "off" path
runs the old lines unchanged** (same statements and the same RNG calls in the same order). New data lives under new cache
or pickle paths only. All runs go through the existing TSV + array + `gen_eval` path (`slurm/common.sh:96-117`). The
only new files are:
- one pure numpy/RDKit helper module, `torsional-diffusion/utils/lgeom.py`;
- three TSVs;
- two test files.

---

## 0. Summary of the diff

| file | change | default path |
|---|---|---|
| `utils/lgeom.py` (NEW, numpy+RDKit only, no PyG) | `kabsch_align`, `interp`, `pair_std_with_raw`, `local_partition`, `copy_acyclic_local`, `l_errors` | not imported unless a flag is on |
| `utils/parsing.py:25-29`, `:34` | `--pair_gt`, `--l_source`, `--l_mix_p`, `--l_lambda_end_mass`, `--l_jitter`, `--lambda_embed_dim` | all off / 0 |
| `utils/dataset.py:18-43` (transform) | L selection by `l_source`, jitter, `node_lambda` | `l_source='default'`, `l_jitter=0` run lines 26-29 verbatim |
| `utils/dataset.py:51-63,128-141,200-241,255-266` | `pair_gt`: attach GT conformer to each matched conformer, Kabsch-align, store `data.pos_gt` | `pair_gt=False`: no raw-pickle read, no `pos_gt` |
| `diffusion/score_model.py:49-75,188-193` | `lambda_embed_dim`: λ embedding concatenated to the sigma embedding | `lambda_embed_dim=0`: identical layers and parameter order |
| `utils/utils.py:10-18` | pass `lambda_embed_dim=getattr(args,'lambda_embed_dim',0)` | released/round-1 yamls lack the key, so 0 |
| `diffusion/sampling.py:105-109,171` | `l_level` kwarg sets `data_gpu.node_lambda` | `None`: no attribute set |
| `generate_confs.py:47-51,99-104,146-155` | `--l_level` + guard | `None` |
| `tools/make_seed_pickles.py` | `--variants`, `--out_dir_variants`, `--noise_sigmas`, `--lambdas`, `--std_pickles`, `--raw_dir`, `--seed` | old two outputs byte-identical when `--variants` absent |
| `tools/paired_compare.py` | `--universe {union,intersection}`, `--report_failures`, `--csv` | `union`, off, off |
| `tools/local_structure_analysis.py` | `--restarts N`, `--mode summarize` (multi-fragment exclusion, n_gt>1 subset), `--mode leak` | `restarts=1` uses `seed=args.seed` exactly as now |
| `tools/smoke_test.py` | `--lambda` flag exercises the λ model | off |
| `slurm/ablation_train_array.sbatch` | variant `rematchgt`; 7th TSV field `GT_GEN_ARGS` | 6-field lines parse the same, and empty vars expand to nothing |
| `slurm/featurize_qm9.sbatch` | variant `rematchgt` (`--pair_gt`) | unchanged cases |
| `slurm/ablation_inference_array.sbatch:39` | `MODEL` is `eval`-expanded (like ARGS at `:40`), so `$WORK/<run>` works | PRE/PRE1/BASE contain no `$` |
| `slurm/common.sh:39-40` | `QM9_SEEDS2=$QM9_DIR/seeds_r2` | additive |
| `slurm/analysis_qm9.sbatch` | round-2 block (existence-guarded, like every block there) | round-1 blocks untouched |
| NEW TSVs | `ablations_train_round2.tsv`, `ablations_inference_round2.tsv`, `round2_smoke.tsv` | |
| NEW tests | `tools/tests/test_round2_cpu.py` (local), `tools/tests/test_round2_model.py` (cluster CPU) | |

`standardize_confs.py` is **not** modified (see §2.1, the key design decision). Deployment: commit locally, then
`git diff 08b5987 -- torsional-diffusion > patches/0002-round2-hooks.patch`. On the cluster:
- snapshot `$REPO` to `$PROJECT/repo_r1_snapshot`, used by the golden tests;
- `git apply` the patch in `$REPO`;
- copy `tools/` and `slurm/` as `setup_env.sh:30-31` does.

Running jobs are unaffected: Python has already imported its modules, and DataLoader workers are forked.

---

## 1. Shared infrastructure: `utils/lgeom.py` (used by S1, S3, S4, F1, F5)

A pure numpy+RDKit module, so it is importable without torch_geometric and unit-testable on the local machine (§9).

```python
# utils/lgeom.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
REMOVE_HS = lambda m: Chem.RemoveHs(m, sanitize=False)   # same as standardize_confs.py:33

def kabsch_align(P, Q):
    """F1. Rigidly move P (n,3) onto Q (n,3); ALL atoms, same atom order (SHORTLIST F1). Proper rotation only."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (P - pc) @ R.T + qc

def interp(P_aligned, Q, lam):
    return (1.0 - lam) * P_aligned + lam * Q          # x_lambda; lam=0 -> matched RDKit, lam=1 -> GT

def pair_std_with_raw(std_confs, raw_confs, tol=1e-5):
    """Re-attach the GT conformer to each conformer-matched conformer of a std pickle (standardize_confs.py overwrites
    conf['rd_mol'] at :116 and keeps the rest of the GEOM dict). Candidate = same 'geom_id' if the key exists
    (UNVERIFIED for QM9 pickles; checked by test M6), else all raw conformers. Accept a candidate only if
    AlignMol(RemoveHs(matched), RemoveHs(gt)) reproduces the stored conf['rmsd'] (computed by exactly this call at
    standardize_confs.py:108) to < tol. Returns list of (std_conf, gt_mol or None, status)."""
    raw_h = [REMOVE_HS(c['rd_mol']) for c in raw_confs]
    by_id = {c['geom_id']: k for k, c in enumerate(raw_confs) if 'geom_id' in c}
    out = []
    for c in std_confs:
        m_h = REMOVE_HS(c['rd_mol'])
        cand = [by_id[c['geom_id']]] if c.get('geom_id') in by_id else range(len(raw_confs))
        hits = [k for k in cand if abs(AllChem.AlignMol(Chem.Mol(m_h), raw_h[k]) - c['rmsd']) < tol]
        hits = _dedup_identical_coords(hits, raw_confs)           # duplicates with identical coordinates are equivalent
        if len(hits) == 1 and _same_elements(c['rd_mol'], raw_confs[hits[0]]['rd_mol']):
            out.append((c, raw_confs[hits[0]]['rd_mol'], 'ok'))
        else:
            out.append((c, None, 'pair_not_found' if not hits else 'pair_ambiguous'))
    return out

def local_partition(mol):
    """ALL-atom bonds/angles split into ring (both bonds of the angle in a ring) and acyclic (everything else,
    incl. exocyclic angles and every X-H)."""

def copy_acyclic_local(mol, cid, src_conf):
    """All-atom generalisation of tools/local_structure_analysis.py:196-215 (set_acyclic_local, heavy_only=True there):
    copy src acyclic bond lengths then acyclic angles onto conformer cid (sequential SetBondLength/SetAngleDeg; the
    moved side is always across an acyclic bond). Branching centres cannot satisfy all constraints, so the achieved
    error is measured by l_errors (F5)."""

def l_errors(mol, cid, gt_mol, gt_cid=-1):
    """F5: bond-length RMSD (A) and angle RMSD (deg) to GT for subsets {heavy, all-atom, ring, acyclic}."""
```

`local_structure_analysis.py` keeps its own heavy-atom `set_acyclic_local`. I copy rather than refactor, so the validated
round-1 tool stays byte-identical.

---

## 2. S3/S4 data: the `rematchgt` variant (F1, F4, equal caps)

### 2.1 Design decision: pair post hoc, do not re-standardize

`standardize_confs.py:79` embeds ETKDG **unseeded**. Re-running matching with a "keep GT" flag would draw new RDKit L,
which makes the S3/S4 RDKit half differ from CTRL_rematch's data. Instead I keep the **exact** `standardized_pickles_rematch`
conformers and re-attach each one's GT partner from the raw pickle at featurization time. The std key is the raw file stem
(`standardize_confs.py:154`, `f[len(root):-7]`), so the raw path is `osp.join(self.root, smile + '.pickle')`, the same
expression as the raw branch at `dataset.py:144-147`.

Consequences:
- **F4 holds exactly.** The S3/S4 RDKit end is CTRL_rematch's training data, conformer for conformer.
- **Equal caps (S3).** Both halves are the same ≤30 conformers (`standardize_confs.py:59-67`) of the same molecules, so
  the GT:RDKit ratio is 1:1 per molecule. This fixes verify_A §4 R2A-3(c). It is also why the GT half is *not* B1's data
  (B1 is uncapped, `dataset.py:209`, and also keeps `rdkit_no_embed` molecules). See the B1cap control in §5.
- **No CPU standardization run** for S3/S4; only a featurize run (~1-2 h CPU).

If test M6 (§9) shows the pairing failing on more than 1% of conformers, the fallback is a 2-line `--keep_gt` flag in
`standardize_confs.py` (store `conf['gt_mol'] = mol` before `:116`) plus a new standardization `VARIANT=rematchgt`. This
loses the exact identity with CTRL_rematch, which must then be stated as a recipe-level control.

### 2.2 Code

`utils/parsing.py`, after `:29`:

```python
parser.add_argument('--pair_gt', action='store_true', default=False, help='[round2] std-pickle mode: attach the GT conformer (raw pickle) to every matched conformer, Kabsch-aligned (F1); stored as data.pos_gt')
parser.add_argument('--l_source', choices=['default', 'gt', 'mix', 'interp'], default='default', help='[round2] which L to train on (needs --pair_gt cache unless default)')
parser.add_argument('--l_mix_p', type=float, default=0.5, help='[round2] P(GT L) per sample for --l_source mix (S3)')
parser.add_argument('--l_lambda_end_mass', type=float, default=0.0, help='[round2] interp: prob. of drawing lambda from {0,1} instead of U[0,1] (S4 spec = 0)')
parser.add_argument('--l_jitter', type=float, default=0.0, help='[round2] per-axis Gaussian sigma (A) added to ALL atoms per sample (S2, F6)')
```

And after `:34`:

```python
parser.add_argument('--lambda_embed_dim', type=int, default=0, help='[round2] sinusoidal embedding dim of the L level lambda (S4); 0 = off')
```

`utils/dataset.py`:
- **ctor** (`:51-52`): add `pair_gt=False`, and store `self.pair_gt = pair_gt` next to `:63`.
- **`construct_loader`** (`:260-266`): pass `pair_gt=getattr(args, 'pair_gt', False)`.
- **`filter_smiles`, std branch**, after `:141`:

```python
            if self.pair_gt:
                raw_path = osp.join(self.root, smile + '.pickle')
                if not osp.exists(raw_path):
                    self.failures['pair_raw_not_found'] += 1; return False
                pairs = pair_std_with_raw(mol_dic['conformers'], self.open_pickle(raw_path)['conformers'])
                for _, _, st in pairs:
                    if st != 'ok': self.failures[st] += 1
                mol_dic = dict(mol_dic, conformers=[dict(c, gt_mol=g) for c, g, st in pairs if st == 'ok'])
```

- **`featurize_mol`**: replace the append at `:222` only when `self.pair_gt` is set. The else-branch is the old line.

```python
            if self.pair_gt:
                Q = conf['gt_mol'].GetConformer().GetPositions()
                P = kabsch_align(mol.GetConformer().GetPositions(), Q)          # F1: matched RDKit moved onto GT
                pos.append(torch.tensor(P, dtype=torch.float)); pos_gt.append(torch.tensor(Q, dtype=torch.float))
            else:
                pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
```

  At `:239` add `if self.pair_gt: data.pos_gt = pos_gt`. Lists are index-paired with `data.pos` and `data.weights`.

Note on F1. SHORTLIST F1 says all-atom Kabsch; verify_A §2 suggests heavy-atom Kabsch applied to all atoms. I follow the
spec (all-atom). Under DE matching the H rotors are matched too (H-inclusive objective by default), so the difference is
small. The actually achieved bond and angle error of x_λ is measured either way (§3.4, F5), which catches the linear-blend
bond shrinkage that verify_A §4 R2A-1(b) warns about.

### 2.3 Cache build

In `slurm/featurize_qm9.sbatch:29-38`, add:

```bash
  rematchgt) STD="$QM9_DIR/standardized_pickles_rematch"; EXTRA="--pair_gt" ;;
```

`EXTRA=""` is initialised before the case. `$EXTRA` is appended to the `train.py --n_epochs 0` call at `:42-44`, and
`CACHE=cache_rematchgt` follows from `:39`. The same case line goes into `slurm/ablation_train_array.sbatch:46-55`, with
`$VEXTRA` appended at `:72`. The cache is reused regardless of args (`dataset.py:67`), so passing the flag at train time
is only for the yaml record. The failure counters (`pair_not_found`, `pair_ambiguous`) are printed by `dataset.py:122`
into the featurize log and are reported.

---

## 3. S1: L-quality dose-response (inference only), with F1, F2, F3, F5, F6

### 3.1 One builder, one pass: `tools/make_seed_pickles.py --variants ...`

New optional args go after `:30-35`. When `--variants` is absent, `--out_gt_confs` and `--out_gt_mols` are processed exactly
as now (`:53-77`). Golden test C5 checks that the old two outputs are byte-identical.

```text
--variants etkdgG,etkdgG_mmff,noise,interp,ringGT,acycGT[,etkdgG_xtb]
--out_dir_variants $QM9_SEEDS2      --seed 1 (never 0, cf. generate_confs.py:65)   --n_workers 16
--noise_sigmas 0.01 0.02 0.04       --lambdas 0 0.25 0.5 0.75 1
--std_pickles $QM9_DIR/standardized_pickles_rematch --raw_dir $QM9_RAW   (interp only)   --xtb_path (optional)
```

All ETKDG-derived variants are built from the **same** embedding in a single pass, so they differ only in the factor
under test:
- The embedding is of `m0` (the GT-graph mol with stereo from 3D, as `:69-71`), `RemoveAllConformers`, with
  `EmbedMultipleConfs(numConfs=2L, randomSeed=seed, numThreads=1)`.
- L = number of clean GT conformers. Seed list order is `e[j + L*r]` for GT conformer j and replicate r ∈ {0,1}.
- If fewer than 2L conformers embed, the molecule is dropped from **all** ETKDG variants and logged.

| variant (file in `$QM9_SEEDS2`) | seed j+L·r | oracle? | answers |
|---|---|---|---|
| `etkdgG.pkl` | e[j+L·r] unmodified | no (GT graph/stereo, cf. G1) | control for MMFF/ring/acyc, same population and seeds |
| `etkdgG_mmff.pkl` | e[j+L·r] after `MMFFOptimizeMoleculeConfs(mmffVariant='MMFF94s')` (same call as `sampling.py:21-26`); on failure the conformer is kept unrelaxed and counted (= `try_mmff` semantics) | no | **F2**: MMFF relaxation inside the seed pickle |
| `gt_noise{σ}ax.pkl` | GT_j + N(0, σ²) per axis on all atoms, independent draw per r | ORACLE | S1.3 (**F6**: per-axis σ in the name) |
| `interp_lam{λ}.pkl` | x_λ = (1-λ)·Kabsch(matched_j → GT_j) + λ·GT_j, from rematch pickles via `pair_std_with_raw` | ORACLE (λ = 0 too: Hungarian-selected seed) | S1.4 (**F1**) |
| `ringGT_acycE.pkl` | copy(GT_j), then `copy_acyclic_local(·, src=e[j+L·r])` → rings exactly GT, everything else ETKDG | ORACLE | S1.5 A5-ring |
| `acycGT_ringE.pkl` | copy(e[j+L·r]), then `copy_acyclic_local(·, src=GT_j)` → rings ETKDG, acyclic ≈ GT | ORACLE | S1.5 A5-acyc |
| `etkdgG_xtb.pkl` (optional) | `utils/xtb.py:43-67 xtb_optimize` on each seed | no | S1.6, only if `xtb` installs in under 30 min; otherwise skipped and noted |

Each variant also writes `<variant>.csv` with one row per seed, from `lgeom.l_errors` against GT_j:
- bond RMSD and angle RMSD on the heavy, all-atom, ring and acyclic subsets;
- `n_torsions_heavy` and the smallest ring size, for the S1.5 strata.

This is **F5 part 1** (the achieved L error, reported per condition as per-molecule means).

Design points:
- **A5-ring and A5-acyc use the same mechanism on every seed** (verify_B §4 R2-5). They form the off-diagonal of a 2×2
  {ring L: ETKDG/GT} × {acyclic L: ETKDG/GT} whose diagonal is `etkdgG` (E,E) and gtLcycle (G,G). The acyc arm's residual
  error at branching centres is the asymmetry, and the CSV measures it.
- **Interp population.** The rematch pickles cover every raw molecule, including the test set, because standardization
  runs over all raw pickles (`standardize_qm9.sbatch:44-56`). They are capped at 30 and miss `no_rotable_bonds` and
  `rdkit_no_embed` molecules (`standardize_confs.py:76,82`). So λ ∈ {0,…,1} is built **including the λ = 1 endpoint on
  the same population**: λ = 1 from the builder is "gtLcycle restricted to the matched subset". The existing gtLcycle run
  (F3 anchor) is still compared, and the coverage difference is reported. Test-molecule lookup goes through
  `mol_dic['smiles']` == the csv `smiles` column (the key `make_seed_pickles.py:57` uses for `true_mols`).
- **Noise with cycle.** The list holds 2L entries, `[noisy(GT_j, draw r)]`, so with `--seed_confs_cycle` and 2L
  samples every GT conformer gets two independent noise draws (`sampling.py:75`: `i % len`).
- Every S1 run uses `--seed_confs_cycle` (**F3**).

### 3.2 Models per condition (deviation, raised)

SHORTLIST S1 lists CTRL_base, CTRL_rematch and B1 (3 seeds each) for every condition, which is 9 × 12 = 108 runs. I propose
using the control that F4 asks for:
- **CTRL_rematch** for the interp series (built from rematch pickles);
- **CTRL_base** for the others (the round-1 B1 contrast);
- B1 always.

That is 72 runs. Adding CTRL_rematch on the other conditions is 18 more runs (4.5 GPU-h) if the vote wants it.

### 3.3 TSV lines (`slurm/ablations_inference_round2.tsv`; format of `ablation_inference_array.sbatch:37`)

One-line `MODEL` change in `ablation_inference_array.sbatch:39`: `MODEL=$(eval echo "$(trim "$MODEL")")`. The result
directory is `$RES/<run>/<NAME>`, so `tspec` in `analysis_qm9.sbatch:139-143` averages training seeds by tag unchanged.
Generator (the TSV is committed as its output):

```bash
S2=\$QM9_DIR/seeds_r2
for s in 0 1 2; do for m in CTRL_base_100ep B1_train_gtL; do M="\$WORK/qm9_${m}_e100_s$s"
  echo "S1_etkdgG_cyc            | $M | 20 | 0 | 1 | --seed_confs $S2/etkdgG.pkl --seed_confs_cycle"
  echo "S1_mmffG_cyc             | $M | 20 | 0 | 1 | --seed_confs $S2/etkdgG_mmff.pkl --seed_confs_cycle"
  for n in 0.01 0.02 0.04; do echo "S1_noise${n}ax_cyc_ORACLE | $M | 20 | 0 | 1 | --seed_confs $S2/gt_noise${n}ax.pkl --seed_confs_cycle"; done
  echo "S1_ringGT_cyc_ORACLE     | $M | 20 | 0 | 1 | --seed_confs $S2/ringGT_acycE.pkl --seed_confs_cycle"
  echo "S1_acycGT_cyc_ORACLE     | $M | 20 | 0 | 1 | --seed_confs $S2/acycGT_ringE.pkl --seed_confs_cycle"
done
for m in CTRL_rematch_100ep B1_train_gtL; do M="\$WORK/qm9_${m}_e100_s$s"
  for l in 0.00 0.25 0.50 0.75 1.00; do echo "S1_interp${l}_cyc_ORACLE | $M | 20 | 0 | 1 | --seed_confs $S2/interp_lam${l}.pkl --seed_confs_cycle"; done
done; done
```

That gives 3 × (2 × 7 + 2 × 5) = **72 lines**. Example literal line:

`S1_noise0.04ax_cyc_ORACLE | $WORK/qm9_B1_train_gtL_e100_s0 | 20 | 0 | 1 | --seed_confs $QM9_DIR/seeds_r2/gt_noise0.04ax.pkl --seed_confs_cycle`

### 3.4 F5 part 2: floors, and the B1 − CTRL contrast

Round-2 block appended to `slurm/analysis_qm9.sbatch`, existence-guarded like `:46-53`:
- `local_structure_analysis.py --mode run` (unchanged tool, `:351-404`) on the B1 s0 run of every S1 condition. Under
  cycle the L of generated conformer i is seed i for every model, so this floor is effectively per condition.
- `tcompare` (`:144-149`) per condition: ref = CTRL (base or rematch), arms = B1, B6, S3, S4, so every comparison is
  arm − CTRL **within** a condition (F5), never absolute curves.
- ε* is taken from `paired_compare.py --csv` rows: B1 − CTRL ΔAMR-R per condition is joined with the per-molecule mean
  achieved angle and bond RMSD from the builder CSVs, and linearly interpolated at the zero crossing. It is reported per
  error type, as verify_B §4 R2-1 asks: bond, angle and ring subset separately.
- S1.5 strata: `breakdown.py` already joins per-molecule results. Add the "fraction of gap closed",
  (etkdgG − arm)/(etkdgG − interp1.00 or gtLcycle), on the intersection, stratified by `n_torsions_heavy` (0 vs > 0)
  and smallest ring size (3/4/5/6/≥7), as verify_B §4 R2-5 asks.

Evidence: E1, E2, E22, E28 (research_A §3); research_B §7 rows "The shift hurts" (TD p.7) and "TD weak on puckered and
fused rings" (TD p.23). The ring-vs-acyclic hypothesis comes from D1 in research_A §0, as corrected by verify_A §3.

---

## 4. S2: B6 noise-augmented GT-L training

Code (`utils/dataset.py`, transform). `TorsionNoiseTransform.__init__` (`:19-22`) gains
`l_source='default', l_mix_p=0.5, l_lambda_end_mass=0.0, l_jitter=0.0` (stored). `construct_loader:255-256` passes the
four from `args` via `getattr(args, name, default)`, so pickled or old args still work. `__call__`:

```python
    def __call__(self, data):
        if self.l_source == 'default':
            # select conformer  (lines 26-29, VERBATIM: same RNG calls as upstream)
            if self.boltzmann_weight:
                data.pos = random.choices(data.pos, data.weights, k=1)[0]
            else:
                data.pos = random.choice(data.pos)
        else:
            self._select_paired_L(data)                                  # S3/S4/B1cap, section 5/6
        if self.l_jitter > 0:                                            # S2: per-sample, per-axis, ALL atoms (F6)
            data.pos = data.pos + self.l_jitter * torch.randn_like(data.pos)
        ...                                                              # lines 31-43 unchanged (torsion noise)
```

- The jitter is drawn fresh at every `get` (the PyG `Dataset.__getitem__` applies the transform after
  `dataset.py:193`'s deepcopy). It is applied before the torsion update (`:40-41`), so the target `edge_rotate` is
  unchanged.
- It uses the torch RNG, which `DataLoader` seeds per worker. Python `random` and numpy are seeded per worker too, in
  torch ≥ 1.9; the cluster torch version is checked by test M2.
- Validation uses the same transform object (`construct_loader:255`), so the val loss and the `best_model.pt` choice
  see the same jitter.

Run config:
- Data: VARIANT `raw` (B1 recipe, cache_raw exists).
- σ = 0.04 Å per axis, 3 seeds (≈ 0.057 Å per bond, F6, verify_A §3 jitter table); σ = 0.02, 1 seed, exploratory.
- Control: B1 (single factor). Target: CTRL_base on RDKit L.
- Pre-registered (SHORTLIST): (a) non-inferior to CTRL_base on RDKit L with margin 0.004 Å AMR-R; (b) better than
  CTRL on GT L (ORACLE).
- Evidence: E5, E8, E9, E10 (research_A; verify_A notes that E9/E10 are an analogy); research_B §7 "Gaussian noise is
  the effective augmentation" (Ho p.6, verify_B #24).

Train TSV lines (`slurm/ablations_train_round2.tsv`, fields as `ablation_train_array.sbatch:39` + new 7th field):

```text
B6_jitter0.04ax   | raw | --l_jitter 0.04 |  | 1 | 0 |
B6_jitter0.04ax   | raw | --l_jitter 0.04 |  | 1 | 1 |
B6_jitter0.04ax   | raw | --l_jitter 0.04 |  | 1 | 2 |
B6_jitter0.02ax   | raw | --l_jitter 0.02 |  | 1 | 0 |
```

`GT_EVAL=1` already gives RDKit, gtL and gtLcycle (`ablation_train_array.sbatch:77-86`).

Panel inference (in `ablations_inference_round2.tsv`, same generator with `M=$WORK/qm9_B6_jitter0.04ax_e100_s$s`):
- B6 σ = 0.04: `S1_etkdgG_cyc`, `S1_mmffG_cyc`, `S1_noise{0.02,0.04}ax_cyc_ORACLE`, `S1_interp0.50_cyc_ORACLE`, × 3
  seeds = 15 runs.
- B6 σ = 0.02: etkdgG, mmffG, noise 0.02/0.04 = 4 runs.

That is 19 runs. SHORTLIST's "+6 B1 control runs on jittered GT L" are already among the S1 B1 lines (noise 0.02/0.04 ×
3 seeds).

---

## 5. S3: mixed-L training (Bernoulli, equal caps)

`_select_paired_L` (shared with S4 and B1cap):

```python
    def _select_paired_L(self, data):
        n = len(data.pos)
        i = random.choices(range(n), data.weights, k=1)[0] if self.boltzmann_weight else random.randrange(n)
        P, Q = data.pos[i], data.pos_gt[i]                     # P: matched RDKit Kabsch-aligned onto Q (F1)
        if self.l_source == 'gt':                              # B1cap: GT L on the capped rematch population
            data.pos = Q
        elif self.l_source == 'mix':                           # S3: Bernoulli(l_mix_p), same index -> equal caps
            data.pos = Q if random.random() < self.l_mix_p else P
        elif self.l_source == 'interp':                        # S4
            if self.l_lambda_end_mass > 0 and random.random() < self.l_lambda_end_mass:
                lam = float(random.random() < 0.5)
            else:
                lam = random.random()                          # spec: U[0,1]
            data.pos = (1.0 - lam) * P + lam * Q
            data.node_lambda = torch.full((data.num_nodes,), lam, dtype=torch.float)
        del data.pos_gt                                         # keep collation identical to the default batch
```

TSV:

```text
S3_mix_p0.5   | rematchgt | --l_source mix --l_mix_p 0.5 |  | 1 | 0 |
S3_mix_p0.5   | rematchgt | --l_source mix --l_mix_p 0.5 |  | 1 | 1 |
B1cap_gtL     | rematchgt | --l_source gt                |  | 1 | 0 |      # OPTIONAL, see below
```

- Controls: CTRL_rematch (RDKit end, F4) and B1 (GT end).
- Panel: etkdgG, mmffG, noise 0.02/0.04, interp 0.50 × 2 seeds = 10 runs.
- Evidence: E5-E7 (research_A; verify_A says E7's "exact B1 pattern" is an analogy). E16 is weak support only
  (verify_A E16).

**B1cap (optional, 1 seed, ~11.5 GPU-h).** This is the capped-B1 control that SHORTLIST S4 asks for: GT L restricted to
exactly the conformers and molecules S3/S4 see. It needs no code beyond `--l_source gt`. It separates "λ-conditioning /
mixing" from "the 30-conformer cap and the rematch molecule set" (verify_A §4 R2A-2(c); verify_B §2 T7). If the budget
forbids it, the confound is stated as a note.

---

## 6. S4: λ-conditioned interpolation training

### 6.1 Where λ enters the model

`diffusion/score_model.py`:
- **Ctor** (`:49-52`): add `lambda_embed_dim=0`; store it next to `:56`.
- **`:66`**: `nn.Linear(in_node_features + sigma_embed_dim + lambda_embed_dim, ns)`.
- **`:72`**: `nn.Linear(in_edge_features + sigma_embed_dim + lambda_embed_dim + radius_embed_dim, ns)`.
- **`build_conv_graph`**, after `:189`:

```python
        if self.lambda_embed_dim > 0:   # [round2] S4: L-level conditioning, same sinusoidal scheme and 0..10000 range as sigma (:188)
            lam_emb = get_timestep_embedding(data.node_lambda * 10000, self.lambda_embed_dim)
            node_sigma_emb = torch.cat([node_sigma_emb, lam_emb], 1)
```

`node_sigma_emb` feeds both the edge features (`:191-192`, via `edge_index[0]`) and the node features (`:193`), so λ
reaches both, exactly like σ. It then propagates through the conv trunk (`:137-139`) into the bond head (`:141-152`).
With `lambda_embed_dim=0` no layer shape changes and no parameter is added, so `get_model` under the same
`torch.manual_seed` gives an identical `state_dict` (golden test M1). A missing `node_lambda` on a λ-model raises
`AttributeError`, which is the intended loud failure.

### 6.2 Plumbing

- `utils/utils.py:10-18`: `lambda_embed_dim=getattr(args, 'lambda_embed_dim', 0)`.
- `diffusion/sampling.py:105-109`: new kwarg `l_level=None`. After `:171`:
  `if l_level is not None: data_gpu.node_lambda = l_level * torch.ones(data.num_nodes, device=device)`.
- `generate_confs.py`, after `:51`: `--l_level` (float, default None). Guard after `:104`:

  ```python
  lam_model = getattr(args, 'lambda_embed_dim', 0) > 0
  if lam_model != (args.l_level is not None): raise SystemExit('--l_level is required for, and only for, lambda-conditioned models')
  ```

  Pass `l_level=args.l_level` into the `sample(...)` call at `:146-155`.
- The training yaml overrides CLI keys of the same name (`generate_confs.py:85-86`). None of the new train flags
  (`pair_gt`, `l_*`, `lambda_embed_dim`) shares a name with a generate flag. `l_level` is generate-only, and the test
  suite asserts this.
- Checkpoints of λ models cannot load as non-λ models and vice versa (`strict=True`, `generate_confs.py:102`). That is
  intended and is carried by the yaml.

### 6.3 Train-array change for GT eval with λ

In `slurm/ablation_train_array.sbatch:39`:

```bash
IFS='|' read -r NAME VARIANT TRAIN_ARGS GEN_ARGS GT_EVAL TSEED GT_GEN_ARGS <<< "${LINES[$IDX]}"
```

- `GT_GEN_ARGS` is trimmed, and `$GT_GEN_ARGS` is appended to the two GT `gen_eval` calls at `:82` and `:85`.
- 6-field lines leave it empty, so the commands are unchanged. `TSEED` used to take "the rest of the line"; for a
  6-field line that is just its own field.

```text
S4_lambda_interp | rematchgt | --l_source interp --lambda_embed_dim 32 | --l_level 0 | 1 | 0 | --l_level 1
S4_lambda_interp | rematchgt | --l_source interp --lambda_embed_dim 32 | --l_level 0 | 1 | 1 | --l_level 1
S4_lambda_interp | rematchgt | --l_source interp --lambda_embed_dim 32 | --l_level 0 | 1 | 2 | --l_level 1
```

### 6.4 Evaluation

- Required (SHORTLIST): λ = 0 on RDKit L (the in-job default eval) and λ = 1 on GT L (gtL/gtLcycle, ORACLE).
- Added (descriptive, ORACLE): the interp series with matched λ (`S4_interp{λ}_cyc_lam{λ}_ORACLE`, λ ∈ {0, 0.25, 0.5,
  0.75, 1}, × 3 = 15) and `S1_mmffG_cyc` at λ = 0 (3). That is 18 runs.
- No λ search on the test set. A validation λ search (CDM's post-hoc search, E6) needs validation seed pickles. That is
  deferred.
- Control: CTRL_rematch (F4); B1 and B1cap for the λ = 1 end.
- Evidence: E5, E6 (research_A; verify_A notes that E6's link to an L-quality λ is an analogy); research_B §7
  "Amortised augmentation strength" (Ho p.6, verify_B #25). E15, E16 and E24 are weak per verify_A and are not relied on.

Deviation, raised: research_A proposed λ drawn as 25% λ = 0, 25% λ = 1, 50% U(0,1). SHORTLIST says U[0,1]. I implement
U[0,1], and the mixture is reachable with `--l_lambda_end_mass 0.5` without code change.

---

## 7. S5: B3 MMFF-matched TD

No new code. The existing deferred line `B3_match_mmff | mmff | | --pre_mmff | 0 | 0`
(`slurm/ablations_train_deferred.tsv:4`) is copied for seeds 0-2 into the round-2 train TSV.

- Prerequisites (CPU, at t = 0): `VARIANT=mmff sbatch standardize_qm9.sbatch` (case `:36`), then
  `VARIANT=mmff sbatch featurize_qm9.sbatch` (case `:33`).
- Inference lines:
  - `S5_pre_mmff | $WORK/qm9_CTRL_rematch_100ep_e100_s{0,1,2} | 20 | 0 | 1 | --pre_mmff` (3 runs). This is the
    matched control; the no-MMFF CTRL_rematch runs already exist.
  - `S5_no_pre_mmff | $WORK/qm9_B3_match_mmff_e100_s{0,1,2} | 20 | 0 | 1 |` (3 runs, the train-only cell, research_B R2-4).
- **Molecules lost to MMFF errors.** Count `grep -hx mmff_error $QM9_DIR/standardized_pickles_mmff/logs/worker_*.log | wc -l`.
  `log_error` prints the bare key (`standardize_confs.py:47-50,87`); the 20-molecule progress dicts would miss the tail.
  Also count the key-set difference against `standardized_pickles_rematch`.
- Evidence: E2, E3 (research_A); research_B §7 "OMEGA's better L on QM9" (TD p.26).

---

## 8. S0 and S6

### 8.1 S0 (CPU)

- `tools/paired_compare.py`:
  - `--universe intersection`: at `:99`, intersect the molecule sets over every file of ref and arms instead of taking
    the union.
  - `--report_failures`: adds `n_fail_ref` and `n_fail_arm` columns (mean over replicate files of universe minus present).
  - `--csv PATH`: machine-readable rows.
  - Defaults reproduce today's tables (golden test C6).
  - The key secondary is run as a separate call:
    `--universe intersection --thresholds 0.1 0.5 --primary COV-R@0.1 --report_failures`. Holm at `:128-131` then
    covers only the COV-R@0.1 family. COV-R@0.5 in the same table is the sensitivity analysis.
  - Re-run for all round-1 `tcompare` / `paired_vs_R0` calls and all round-2 ones (analysis block).
- `tools/local_structure_analysis.py`:
  - `--restarts N` (default 1): in `torsion_floor` (`:179-181`), run DE with `seed=args.seed + r` for r < N and keep
    the minimum. With N = 1 the call is identical. Report the floor change for `--restarts 2 --limit_mols 100` on R0 and A2.
  - `--mode summarize --in_csv X [--exclude_multifrag] [--min_n_gt 2]`: per-molecule means on matched molecule sets
    (verify_B §3(b)), without re-running DE.
  - `--mode leak` (verify_B §4 R2-0(e), redesigned in torsion space):
    - Inputs: `--confs` (a gtLcycle `confs.pkl`), `--seeds $QM9_GT_SEED_CONFS`, `--raw_dir` (Boltzmann weights).
    - Source of generated conformer i is `i % L_seed` (`sampling.py:75`).
    - Distance d(k,l) = circular RMS over `relevant_torsions` heavy torsions, minimised over `heavy_automorphisms`
      (both exist, `:108-152`).
    - Output per molecule (L ≥ 2): source-hit rate, chance 1/L, and Boltzmann chance = mean_k w_src(k). Weights come
      from raw pickles matched by coordinates; if that fails, weights are uniform and counted.
    - Nulls: `A1c_gtL_cycle_sanity` (positive control, should be ≈ 100%), a new no-model random-torsion cycle run
      `A1c_gtL_cycle_random_tors | PRE | 20 | 0 | 1 | --seed_confs $QM9_GT_SEED_CONFS --seed_confs_cycle --no_model`
      (1 inference line), and CTRL_base gtLcycle (existing).

### 8.2 S6 (low priority)

The seed-0 lines exist in `ablations_inference_deferred.tsv:7,8,10,12` (`C_steps10`, `C_steps50`, `D_ode_steps20`,
`E_inf_sigmin_0.005pi`). Add `_seed1` and `_seed2` copies (8 lines, PRE) and 4 B1 lines:
`S6_B1_<setting>_gtLc_ORACLE | $WORK/qm9_B1_train_gtL_e100_s0 | <steps> | 0 | 1 | --seed_confs $QM9_GT_SEED_CONFS --seed_confs_cycle [--ode|--sigma_min_inf 0.0157]`.
That is 16 runs. B1 is descriptive (verify_B §4 R2-6). Evidence: research_B §7 "More steps help at 0.05 on QM9"
(FM-refiner p.14, verify_B #19, model-dependent).

---

## 9. Tests

The local machine has Python 3.11, rdkit 2026.03.6, torch 2.14 (CPU), numpy/scipy/pandas/networkx/yaml, but **no
torch_geometric/e3nn/torch_scatter** (checked: `import torch_geometric` fails). So the tests are split in two.

### 9.1 `tools/tests/test_round2_cpu.py` (runs locally and on the cluster; plain asserts, no pytest needed)

- **C1 Kabsch.**
  - A random rotation + translation of a random cloud is recovered (RMSD < 1e-8).
  - det(R) = +1 for a mirrored input (no reflection).
  - `interp(.,.,0) == P_aligned` and `interp(.,.,1) == Q` exactly.
  - Kabsch of P onto itself is the identity.
- **C2 Pairing.**
  - Build GT = 6 ETKDG+MMFF conformers of `CCOC(=O)CCN`. Build "matched" = fresh ETKDG conformers with torsions set by
    `optimize_rotatable_bonds` (`utils/standardization.py:29-44`) against a permuted GT, and store `rmsd` as
    `standardize_confs.py:108` does.
  - `pair_std_with_raw` recovers the permutation. A duplicated GT conformer is resolved by `_dedup_identical_coords`.
    A corrupted `rmsd` gives `pair_not_found`.
- **C3 Jitter statistics (F6).** σ = 0.04 per axis gives per-axis SD 0.04 ± 2% and C-C bond RMSE ≈ 0.057 Å; σ = 0.02
  gives ≈ 0.028 Å (matches verify_A §3).
- **C4 Builder variants** on 3 molecules, with a synthetic `test_csv` and `true_mols` (ETKDG+MMFF stand-ins for GT):
  - etkdgG is identical across two invocations (determinism, `numThreads=1`).
  - mmffG differs from etkdgG only after MMFF.
  - ringGT: ring bond and angle RMSD to GT == 0.
  - acycGT: acyclic bond RMSD < 1e-3 Å; ring error equals etkdgG's.
  - noise: per-axis SD ≈ σ, and the two draws per GT conformer differ.
  - interp λ = 1 == GT coordinates.
  - every list has length 2L (L for interp), and every seed mol has the GT atom order (element sequence equal).
- **C5 Golden: make_seed_pickles.** Without `--variants`, the outputs of HEAD (`git show 08b5987:tools/make_seed_pickles.py`)
  and the new script on the synthetic inputs are byte-identical (sha256).
- **C6 Golden: paired_compare.** Synthetic eval.pkl files (3 arms, some molecules missing): the default-flag stdout and
  `--out` md are identical to HEAD. Intersection mode drops exactly the missing molecules. The Holm family equals
  `--primary`.
- **C7 Golden: local_structure_analysis.** `--mode test --limit_mols 3 --restarts 1` gives a CSV identical to HEAD.
  `--restarts 2` gives floors ≤ the `--restarts 1` values row-wise.

### 9.2 `tools/tests/test_round2_model.py` (cluster, CPU)

Run with `srun -p plafnet2 -A plafnet2 -w gnode118 -c 8 --mem 32G -t 00:45:00`, inside the venv. It imports
`$PROJECT/repo_r1_snapshot` as HEAD for the golden comparisons.

- **M1** `get_model` (QM9 args, `lambda_embed_dim` absent or 0) under `torch.manual_seed(0)` gives a `state_dict` equal
  to HEAD's tensor-for-tensor. The forward output on a fixed batch is equal.
- **M2** Transform defaults: on deep copies of 50 cache_std datapoints with all RNGs seeded identically, the new
  `TorsionNoiseTransform()` gives `pos`, `edge_rotate` and `node_sigma` equal to HEAD's. The `random`, `np.random` and
  `torch` RNG states after the call are equal (no extra RNG draw). A DataLoader with `num_workers=2` gives different
  sigmas across workers (per-worker seeding check).
- **M3** λ model:
  - `node_lambda` = 0 vs 1 gives different `edge_pred`;
  - a missing `node_lambda` raises;
  - reflection parity still holds (`smoke_test.py:59-71` logic, error < 1e-3);
  - `sample(..., l_level=0.3)` runs 2 steps.
- **M4** mix: over 10k draws the GT fraction is 0.5 ± 0.015. The chosen pos equals `pos[i]` or `pos_gt[i]`.
  `pos_gt` is gone after the transform. Batch collation succeeds.
- **M5** interp: with `sigma_min = sigma_max = 1e-9` (no torsion move), `pos == (1-λ)P + λQ` and `node_lambda ∈ [0,1]`.
- **M6** Real-data pairing on 300 molecules of `standardized_pickles_rematch`:
  - pairing success ≥ 99% (also reports whether `geom_id` exists);
  - element sequences match;
  - for 100 test molecules, the raw-pickle GT conformers equal the `test_mols.pkl` conformers (coordinates < 1e-4 Å).
    If not, interp λ = 1 is not gtLcycle and is reported as such.
  - **Gate for S3/S4 (wave 2).**
- **M7** generate guard: a λ-model dir without `--l_level`, and a non-λ dir with `--l_level`, both exit non-zero.
  The new train flag names ∩ generate flag names = ∅.
- **M8** Golden featurization: `ConformerDataset(..., limit_molecules=50, cache=None)` on std and raw gives datapoints
  equal to HEAD's (all tensors and lists).
- **M9** Golden generation: `generate_confs.py` (released model, `--limit_mols 5 --seed 0 --no_energy`, CPU) gives a
  `confs.pkl` whose coordinates are identical to HEAD's.

### 9.3 Smoke test on gnode118 (GPU, 20 molecules / 1 epoch), through the production path

`slurm/round2_smoke.tsv` contains one line per new config:

```text
SMK_default  | std       | --limit_train_mols 20                                                  | --limit_mols 20 | 0 | 0 |
SMK_B6       | raw       | --limit_train_mols 20 --l_jitter 0.04                                  | --limit_mols 20 | 0 | 0 |
SMK_S3       | rematchgt | --limit_train_mols 20 --l_source mix                                   | --limit_mols 20 | 0 | 0 |
SMK_S4       | rematchgt | --limit_train_mols 20 --l_source interp --lambda_embed_dim 32          | --limit_mols 20 --l_level 0 | 0 | 0 |
```

Run as `TABLE=$PROJECT/slurm/round2_smoke.tsv N_EPOCHS=1 sbatch --array=0-3%4 --time=02:00:00 ablation_train_array.sbatch`.
- `limit_train_mols` truncates **after** loading an existing cache (`dataset.py:67-80`), so the real caches are never
  rewritten. All 4 caches must exist first.
- The `_e1` run names keep these runs apart from production ones.
- Then a 6-line inference smoke: `TABLE=.../round2_smoke_inf.tsv`, with the S4 model at `--l_level 0.5` and B6 on 5 seed
  variants, all with `--limit_mols 20`.
- `tools/smoke_test.py --lambda` (new flag) runs the parity and sampling check on a `lambda_embed_dim=32` model.
- Pass = every job reaches `done:`, has a SUMMARY line and no NaN loss. Afterwards remove `$WORK/qm9_SMK_*_e1_*` and
  `$RES/qm9_SMK_*`.

---

## 10. Risk to existing behaviour, and how defaults stay unchanged

| change | risk | guarantee |
|---|---|---|
| transform | RNG call order changes, so round-1 training is not reproducible | default branch is lines 26-29 verbatim; jitter block skipped at 0; **M2** asserts equal outputs and equal post-call RNG state |
| dataset `pair_gt` | default path reads raw pickles; cache contamination | gated by flag; new cache path `cache_rematchgt`; **M8** golden datapoints |
| score model | parameter order or init changes for old configs; old checkpoints fail `strict=True` | no new module when dim = 0; **M1** state_dict and forward equality; **M9** released-model generation equality |
| `get_model` / yaml | old yamls lack the key | `getattr(..., 0)` |
| sampling / generate | `node_lambda` leaks into the non-λ path; yaml clobbers a CLI flag (`:85-86`) | set only if `l_level` is not None; guard; **M7** name-collision check |
| make_seed_pickles | changed outputs for round-1 jobs (`ablation_*_array.sbatch` call it under flock) | variants only with `--variants`; **C5** sha256 golden |
| paired_compare / local_structure_analysis | round-1 tables change | defaults unchanged; **C6/C7** golden |
| sbatch edits | broken parsing of existing TSVs | 7th field optional; `MODEL` eval only expands `$`; dry-run check: `IDX=k bash -n` and an `echo` of the parsed fields for every line of all 6 TSVs, then a diff of the printed train/gen commands before and after (a 15-line harness run on the login node with `python` stubbed) |
| scientific: interp seeds from capped rematch | λ series not comparable to gtLcycle | builder λ = 1 endpoint on the same population; report coverage |
| scientific: all-atom vs heavy Kabsch (F1) | H misalignment inflates x_λ error | F5 CSV reports the achieved error per λ; switchable in one line if the vote prefers heavy |
| CPU contention on gnode118 | standardize (16 cpus) + featurize + builder slow 3 training jobs (10 cpus each) | check `nproc` first; run the builder and featurize before wave 2; standardize mmff with `--cpus-per-task` sized to the free cores |

---

## 11. Schedule (4 GPUs, gnode118)

CPU at t = 0 (no GPU):
- deploy the patch;
- C-tests;
- M-tests (45 min);
- `make_seed_pickles --variants` (~1 h, 16 cores);
- `VARIANT=mmff` standardize (est. 1-4 h at 32 procs per `standardize_qm9.sbatch:23`, UNVERIFIED; longer at 16) →
  featurize mmff (~1 h);
- `VARIANT=rematchgt` featurize (~1-2 h);
- S0 analyses;
- smoke (GPU, about 1 h on 4 GPUs at t ≈ 2-3 h; it delays wave 1 by about 1 h, or runs on GPU3 while wave 1 starts on GPUs 0-2).

| t (h) | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| 0-11.5 | B6 σ.04 s0 | B6 σ.04 s1 | B6 σ.04 s2 | smoke, then S1 inference (46 runs) |
| 11.5-23 | S4 s0 (gate M6 + smoke) | S4 s1 | S4 s2 | rest of S1 (26) + S6 (16) + leak null (1) |
| 23-34.5 | B3 s0 | B3 s1 | B3 s2 | S3 s0 |
| 34.5-46 | S3 s1 | B6 σ.02 s0 | B1cap s0 (optional) or panel inference | S2 panel (15) + S4 panel (18) |
| 46-49 | S3 panel (10), B6 σ.02 panel (4), S5 (6), spread over 4 GPUs | | | |

Mechanics:
- The train array (`ablations_train_round2.tsv`, ordered B6 ×3, S4 ×3, B3 ×3, S3 s0, S3 s1, B6σ.02, B1cap) is submitted
  with `%3`.
- S1 runs as a separate inference array with `%1`. When it drains (t ≈ 22 h), `scontrol update JobId=<train> ArrayTaskThrottle=4`.
- Panel inference is a second inference array submitted with `--dependency=afterok:<train array>`. Its seed pickles and
  models exist by then.
- If S4's gate fails at t ≈ 11 h: move B3 to wave 2 and S3 s0/s1 to wave 3, and drop S4 (round 3).

**GPU-hours** (10.7 h per training run, 0.25 h per inference run, from BRIEF):

| item | runs | GPU-h |
|---|---|---|
| S2 training (4) + in-job evals (3 each) | 4 + 12 | 42.8 + 3.0 |
| S3 training (2) + evals | 2 + 6 | 21.4 + 1.5 |
| S4 training (3) + evals | 3 + 9 | 32.1 + 2.25 |
| S5 training (3) + evals | 3 + 3 | 32.1 + 0.75 |
| S1 inference | 72 | 18.0 |
| S2 / S3 / S4 panels | 19 / 10 / 18 | 11.75 |
| S5 extra inference | 6 | 1.5 |
| S6 + leak null | 17 | 4.25 |
| smoke | ~10 | ~1.5 |
| **core total** | | **≈ 173 GPU-h (≈ 43-49 h wall on 4 GPUs)** |
| optional B1cap | 1 + 3 | +11.45 |

This is about 20 GPU-h more than SHORTLIST's 153. The extra comes from:
- the same-seed etkdgG control for F2/S1.5;
- the λ = 0 and λ = 1 builder endpoints;
- the trained-arm panels, which SHORTLIST's 25 GPU-h inference line did not cost.

Trims if needed (in order):
- S2/S3 panels to {etkdgG, mmffG, interp0.50}: −4 GPU-h;
- S1 noise 0.01: −1.5;
- S6 B1 lines: −1;
- S4 matched-λ series to {0.5}: −3.

Without S4 the total is ≈ 135 GPU-h.

---

## 12. Recommendation on S4

**S4 fits in round 2, gated.** Reasons:
1. **Marginal code is small.** About 15 lines beyond what S1.4 and S3 need anyway: 4 in `score_model.py`, 1 in
   `utils.py`, 1 parsing flag, 3 in `sampling.py`, 5 in `generate_confs.py`, and the `interp` branch shared with S3.
   The expensive part, paired Kabsch-aligned RDKit/GT data (F1), is required by S1.4 and S3 regardless.
2. **The default path is provably untouched** (M1/M9 golden tests), so S4 adds no risk to other arms.
3. **It fits the wave plan** in the slot after S2 without delaying S5.

Gate before wave 2 (t ≈ 11 h): M6 pairing ≥ 99% and S4 smoke passes (finite loss, `--l_level` 0 and 1 generate, parity
holds). If the gate fails, S4 moves to round 3 and the plan above runs without it (≈ 135 GPU-h).

Scientific caveat to carry into the vote: S4 changes data (continuum of L) and architecture (λ input) together. S3 isolates
the conditioning only approximately, because S3 has Bernoulli endpoints without λ and no λ input (verify_A §4 R2A-2(e)).
B1cap is the cleanest way to read S4's λ = 1 end.
