# Round 2 code plan, coding agent 2: correctness and validity first

Written 2026-10-06 by coding agent 2 of 3. I read BRIEF.md, SHORTLIST.md, verify_A.md, verify_B.md and research_A/B.md.
I did not read code_plan_1.md or code_plan_3.md, and I changed no code.

**Citation conventions**
- Paths are relative to `C:\Users\HP\Desktop\tor_diff`.
- `TD/` = `torsional-diffusion/` on local `main` @ 08b5987, which contains the ablation hooks.
- Code claims cite `path:line`.
- Literature evidence uses research_A IDs (E1–E28), all VERIFIED in verify_A §1, and research_B rows (B#n), numbered as in verify_B §1.
- "Local check" means a script I ran myself in the scratchpad (RDKit 2026.03.6, numpy). The cluster has RDKit 2022.9.5 (`slurm/setup_env.sh:100`), so cluster behaviour is UNVERIFIED until the cluster CPU tests (§9) run.

---

## 0. Summary of design decisions (and the places where I deviate or raise a point)

1. **One shared, offline-validated geometry layer** (`tools/lgeom.py`) for all L manipulation:
   - Kabsch fit;
   - terminal-atom relabelling;
   - stereo check;
   - interpolation safety;
   - subset L errors.

   Training code only reads precomputed, aligned arrays, so the GPU code changes stay small:
   - about 40 lines in `TD/utils/dataset.py`;
   - about 25 lines in `TD/diffusion/score_model.py`;
   - about 15 lines in `TD/generate_confs.py` and `TD/diffusion/sampling.py`.
2. **S3/S4 training data = the existing `standardized_pickles_rematch`, with GT positions paired back in.**
   - We do not re-run conformer matching (`tools/build_paired_pickles.py`).
   - The RDKit half of S3/S4 is then *bit-identical* to CTRL_rematch's training data. F4 makes CTRL_rematch the control, so S3/S4 against CTRL_rematch is a cleaner single-factor contrast than a re-standardization would give.
   - The pairing is verified by recomputing the stored `conf['rmsd']` (`TD/standardize_confs.py:108,117`).
3. **F1 alignment.**
   - The fit uses heavy atoms (proper rotation), and the transform is applied to all atoms. This follows verify_A's wording ("heavy-atom Kabsch, with the transform applied to all atoms", verify_A.md:64). SHORTLIST F1 says "all-atom".
     - **Raised:** I read F1 as "transform all atoms". A fit on H atoms would be dominated by H rotors, which DE matched with an H-inclusive objective. That objective can leave H RMSD near 1 Å (`notes/code_concerns.md` A.7).
     - The heavy fit reproduces `conf['rmsd']` exactly, and that equality is the pairing check.
   - **New validity issue found:** linear interpolation with the identity atom map collapses rotated terminal groups.
     - Local check: rotate a methyl's H atoms by 120° and interpolate. At λ = 0.5, C–H shrinks from 1.09 Å to about 0.62 Å.
     - After relabelling graph-equivalent terminal atoms (Hungarian per parent atom), C–H stays at about 1.11 Å.
     - So F1 needs a third step: terminal-atom relabelling (§1.3).
4. **F2 (MMFF).**
   - MMFF seeds are built in a pickle together with their *unrelaxed twin*: the same ETKDG embedding before relaxation, from the GT graph.
   - Both run through `--seed_confs --seed_confs_cycle` with exactly K = 2·n_csv seeds per molecule, so every seed is used exactly once.
   - Twin and MMFF therefore differ only by MMFF: same graph, same population, same pre-relaxation coordinates.
5. **A5 partition (S1.5).**
   - Both arms come from the *same matched (RDKit, GT) pair* with the *same* function, with the roles swapped.
   - Unlike `set_acyclic_local`'s heavy-only use (`tools/local_structure_analysis.py:242,297`), the acyclic set **includes H** (all-atom).
   - Otherwise A5-ring would carry GT H geometry and A5-acyc RDKit H geometry, and the two arms would not be complementary. **Raised as a deliberate deviation.**
6. **S4 fits this round, but as the last training wave behind a go/no-go gate (§8).**
   - Almost all of its risk (pairing, alignment, interpolation sanity) is shared with S3 and the S1 λ series, which we must validate anyway.
   - The only S4-specific code is the λ embedding. It is protected by a strict-load regression test.
7. **Extra single-factor control recommended:** `B1cap` (paired pickles, GT L always: `--l_mix_p_gt 1.0`, 1 seed).
   - It removes the cap/population confound between B1 (all raw conformers, `TD/utils/dataset.py:209`) and S3/S4 (≤30 conformers, `TD/standardize_confs.py:59-67`).
   - If S4 is deferred, B1cap takes one of its slots.
8. **Budget is above the shortlist.** It is about 187 GPU-h against 153, because F3/F4/F5 add inference: cycle-family conditions and CTRL_rematch on S1. Training and inference lanes balance at about 47–50 h wall on 4 GPUs. Trim options are in §10.

---

## 1. Shared infrastructure (used by S1–S4)

### 1.1 `tools/lgeom.py` (new, CPU, rdkit + numpy + scipy only, so unit-testable locally)

```python
def kabsch_fit(P, Q, idx):
    """Rigidly move Q onto P, fitting on atom indices idx (proper rotation, det=+1). Returns moved Q (all atoms)."""
    Pc, Qc = P[idx].mean(0), Q[idx].mean(0)
    H = (Q[idx] - Qc).T @ (P[idx] - Pc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1., 1., d]) @ U.T
    return (Q - Qc) @ R.T + Pc
```
- **Local check.** On an ETKDG pair, the heavy-atom RMSD after `kabsch_fit` equals `AllChem.AlignMol(RemoveHs(a), RemoveHs(b))` to 5e-16 Å.
- **Local check.** For a mirror image, the RMSD stays large (1.26 Å), so no reflection is allowed. This matches `standardize_confs.py:108`.

```python
def permute_terminal(mol, X, Y):
    """Relabel graph-equivalent terminal atoms of Y (degree 1, same parent, element, bond order, charge) so that
    sum |X - Y| is minimal (Hungarian per group). This is a graph automorphism of the shared molecular graph, so
    the model sees the same labelled graph either way. Returns Y[perm], perm."""
    perm = np.arange(len(Y))
    for p in mol.GetAtoms():
        groups = defaultdict(list)
        for b in p.GetBonds():
            a = b.GetOtherAtom(p)
            if a.GetDegree() == 1:
                groups[(a.GetAtomicNum(), b.GetBondTypeAsDouble(), a.GetFormalCharge())].append(a.GetIdx())
        for idx in map(np.array, groups.values()):
            if len(idx) < 2: continue
            C = np.linalg.norm(X[idx][:, None] - Y[idx][None], axis=-1)
            r, c = linear_sum_assignment(C)
            perm[idx[r]] = idx[c]
    return Y[perm], perm
```

`align_pair(mol, X_rdkit, Y_gt)` is the canonical F1 procedure:
1. Kabsch on heavy atoms (fit Y onto X).
2. `permute_terminal`.
3. Kabsch again.
4. `permute_terminal` again; this second pass is idempotent in tests.

It returns `Y_al`, `perm`, `heavy_rmsd` and `n_swaps`.

`interp_safety(mol, X, Y_al, lams=(0.25, 0.5, 0.75))` returns:
- `worst_bond_dev`, the maximum over bonds and λ of |d(x_λ) − ((1−λ)d_X + λd_Y)|;
- `min_nonbonded`, the minimum distance between atoms with topological distance ≥ 3 (`Chem.GetDistanceMatrix`) over the λ grid.

`pair_ok(...)` (thresholds pre-declared now, before any data are seen) is true only if all three hold:
- `stereo_ok` (`stereo_labels` from `tools/local_structure_analysis.py:217-222`, imported, equal for X and Y_al);
- `worst_bond_dev ≤ 0.05 Å`;
- `min_nonbonded ≥ 0.9 Å`.

`l_subset_errors(mol, X, Y_ref)` returns bond-length RMSD (Å) and angle RMSD (°):
- for all atoms and for heavy atoms only;
- split into a **ring** subset (bond in a ring; angle whose two bonds are both in rings) and an **acyclic** subset (the rest);
- plus the endocyclic dihedral RMSD (°, circular) as the pucker error.

Bonds and angles are enumerated as in `tools/local_structure_analysis.py:83-101`, with `heavy_only=False` for the all-atom version.

`set_internal_subset(mol, conf, ref_conf, n_passes=3)` generalises `set_acyclic_local` (`tools/local_structure_analysis.py:196-214`):
- It copies **all-atom** acyclic bond lengths and acyclic/exocyclic angles from `ref_conf`.
- It repeats 3 passes, because sequential `SetAngle` at branching centres disturbs angles set earlier (docstring `:197-198`).
- Ring atoms never move. `SetAngleDeg`/`SetBondLength` move only the side of the acyclic bond (`:210`). So ring bonds, ring angles and endocyclic dihedrals of the base are preserved exactly. This is asserted (§9 T8).

`assert_same_graph(mols)`: identical atomic-number sequence, and identical bond set {(i, j, bond order)}, for all mols of one molecule.

### 1.2 Why match before interpolating (validity rationale)

Interpolating a GT conformer with an *unmatched* RDKit conformer would interpolate torsions as well as L. The result would be geometrically meaningless, with bonds shrinking across torsion differences (verify_A §4 R2A-1(b)).

The DE-matched RDKit conformer has torsions close to GT (`TD/standardize_confs.py:106-107`). x_λ therefore varies mainly in L. Its torsions are re-randomised at test time anyway (`TD/diffusion/sampling.py:94-99`) and noised in training (`TD/utils/dataset.py:40-41`).

### 1.3 Terminal-atom relabelling: the extra F1 step

DE matching uses an H-inclusive objective with popsize 15 and maxiter 15 (`TD/standardize_confs.py:15-16,106-107`). It can leave methyl or NH2 groups rotated by about 120° (`notes/code_concerns.md` A.7). With the identity map, the midpoint geometry then puts H atoms about 0.6 Å from the carbon (local check above).

`permute_terminal` fixes this exactly for degree-1 atoms. Non-terminal symmetric groups (for example swapped isopropyl methyls) are *not* relabelled. They show up as a large `heavy_rmsd` and fail `pair_ok` through `worst_bond_dev` or `min_nonbonded`. Those pairs are not interpolated (§4.3).

---

## 2. S0: CPU hygiene and diagnostics (no new GPU training)

### 2.1 Code
1. **`tools/paired_compare.py`.**
   - Add `--universe {union,intersection}` at `:99`. Intersection = the molecules present in **every file** of ref and arms.
   - Add per-file failure counts: `per_mol` (`:41-55`) also returns `R.get('num_failures')`, which `TD/evaluate_confs.py:190` stores in the pickle.
   - Holm families need no new logic. Holm runs only over `--primary` (`:128-131`).
     - The **primary** family is one call: `--universe union --primary MAT-R COV-R@0.5`.
     - The **key secondary** family is a second call: `--universe intersection --primary COV-R@0.1`.
     - COV-R@0.5 is reported in the intersection call as a non-primary sensitivity row.
2. **`tools/local_structure_analysis.py`.**
   - Add `--de_restarts R` (default 1): `torsion_floor` (`:163-184`) runs DE with seeds `args.seed + 7919*r` and keeps the minimum.
   - Add `--de_seed`, decoupled from the ETKDG seed. Today `args.seed` drives both `EmbedMultipleConfs` (`:248`) and DE (`:180`), so a "restart" with a different seed would also change the seeds. That would be a confound.
3. **`tools/floor_summary.py`** (new). Re-summarises `local_structure_test.csv` as follows:
   - excludes SMILES containing `.`;
   - computes per-molecule means;
   - restricts to the matched population (`n_gt > 1`) whenever `floor_gtother` is involved (verify_A §3 item 4; verify_B §3(b)).
4. **`tools/leak_test.py`** (new): the torsion-space leak test (verify_B §4 R2-0(e)).
   - **Source index.** Generated conformer i of a cycle run is seeded from GT conformer `i mod L` (`TD/diffusion/sampling.py:75`). Generation order is kept into `confs.pkl` (`TD/generate_confs.py:162,200`).
   - **Assertion.** `len(seed_confs[raw_smi]) == len(clean_confs(corr, true_mols[key]))`. Both lists come from the same `clean_confs` order (`tools/make_seed_pickles.py:38-42,61`; `TD/evaluate_confs.py:47,86`).
   - **Distance.** Heavy relevant torsions (`relevant_torsions`, `tools/local_structure_analysis.py:108-133`). Distance = RMS of wrapped differences, minimised over heavy automorphisms (`heavy_automorphisms`, `:140-152`) mapped onto the torsion index tuples.
   - **Statistic.** hit = (nearest GT in τ == source). Excess = hit rate − permutation null (sources shuffled within the molecule, 1000 permutations). A Boltzmann null, Σ_j w_j·P(nearest = j), is also reported.
   - **Inputs.** B1 ×3 and CTRL_base ×3 `steps20_seed0_gtLcycle`, plus two calibration runs:
     - `A1c_gtL_cycle_sanity`: identity, the ceiling; it already exists (`slurm/ablations_inference.tsv`, line `A1c_gtL_cycle_sanity`).
     - **new** `A1c_gtL_cycle_random_tors`: `--seed_confs $QM9_GT_SEED_CONFS --seed_confs_cycle --no_model`, the floor null (≈0.25 GPU-h).
   - **Output.** Paired bootstrap over molecules of excess(B1) − excess(CTRL).
5. **Data-validity scan, `tools/data_validity_scan.py`** (new, CPU). It applies to all round-1 and round-2 training arms:
   - (a) `assert_same_graph` over **all GEOM conformers** of every train/val/test molecule.
     - B1 silently assumes this: it uses the last conformer's graph for all positions (`TD/utils/dataset.py:224,234,239`).
     - So do the GT seed pickles: graph from conformer 0, positions from conformer i (`TD/diffusion/sampling.py:34,75,80`).
     - Report the count. If > 0, list those molecules and mark them in every B1/GT-L table.
   - (b) Train/val/test overlap of non-isomeric canonical SMILES (split from `QM9_SPLIT`, test from `test_smiles.csv`). Expect 0. A non-zero count is reported as possible leakage.
   - (c) Molecules with `.` and multi-fragment graphs, listed (they are rejected at `TD/diffusion/sampling.py:38-40` and `TD/utils/dataset.py:152-154`).
   - (d) The DE result discarded at `TD/standardize_confs.py:106`.
     - `optimize_rotatable_bonds` returns `apply_changes(...)`, which works on `copy.copy(mol)` (`TD/utils/standardization.py:24,42`). The return value is not assigned.
     - Local check: `copy.copy` of an RDKit Mol is an independent copy. The *stored* matched conformer therefore carries the torsions of the **last DE/polish evaluation**, not `result.x`.
     - Expected to be negligible: the polish step ends near the optimum.
     - Measure it on 100 molecules (stored-conformer heavy RMSD vs re-applied `result.x`). Report it; this is upstream behaviour shared by all conformer-matching arms.

### 2.2 S0 runs
- Re-run every round-1 paired table twice (union/primary, then intersection/secondary).
- Floor summaries.
- 100-molecule `--de_restarts 1` vs `2`, at a fixed `--seed 1` and with the DE seed varied.
- 50-molecule `--maxiter 200 --floor_topm 10` spot check (verify_B R2-0(c)).
- Leak test.
- Validity scan.

Cost: about 10–14 CPU-h on 16 cores, and 0.25 GPU-h.

---

## 3. S1: L-quality dose-response (inference only)

### 3.1 New builder: `tools/make_l_seed_pickles.py` (CPU)

| `--source` | built per molecule | pickle keys / order | family | label |
|---|---|---|---|---|
| `etkdg` | K = 2·n_csv ETKDG seeds from the GT graph, **plus** an MMFF94s-relaxed copy of the *same* embeddings (written as two pickles in one pass) | raw_smi → list of K one-conformer mols | non-oracle | `S1_etkdg2L`, `S1_etkdg2L_mmff` |
| `noise` | GT conformer i + σ·z_i, with z_i ~ N(0, I) per axis on all atoms, fixed per (mol, i) and **shared across σ** (common random numbers) | raw_smi → L mols in `clean_confs` order | cycle | `S1_noise{0.01,0.02,0.04}pa_cyc_ORACLE` |
| `interp` | for each GT conformer i: its DE-matched ETKDG seed (rematch recipe), `align_pair`, x_λ = (1−λ)X + λY_al | same | cycle | `S1_lam{0.00,0.25,0.50,0.75}_cyc_ORACLE` (+ `lam1.00` pipeline check) |
| `a5` | the same matched pair: **A5-acyc** = X with all-atom acyclic L set from Y_al; **A5-ring** = Y_al with all-atom acyclic L set from X (`set_internal_subset`) | same | cycle | `S1_A5ring_cyc_ORACLE`, `S1_A5acyc_cyc_ORACLE` |
| `xtb` (optional, S1.6) | as `etkdg`, then `xtb_optimize` (`TD/utils/xtb.py:44`) | as `etkdg` | non-oracle | `S1_etkdg2L_xtb` |

**Details that matter for validity**
- **Graph and stereo origin.** `base = deepcopy(clean GT confs[0])`, then `AssignStereochemistryFrom3D`, exactly as `tools/make_seed_pickles.py:69-70`. Then `RemoveAllConformers`, then `EmbedMultipleConfs(randomSeed=1)`.
  - `randomSeed` is **never 0**: RDKit seed 0 gives identical conformers (`TD/generate_confs.py:64-67`; `notes/code_concerns.md` A.6). The builder asserts `seed >= 1`.
  - Round-1 seeds embedded from the GT graph already matched the GT stereo in 99.93% of cases (`cluster_sync/results/analysis/local_structure_test.log`). The builder re-measures this and stores the rate.
- **MMFF.**
  - Uses `AllChem.MMFFOptimizeMoleculeConfs(mmffVariant='MMFF94s')`, the same call as `try_mmff` (`TD/diffusion/sampling.py:21-26`). A2 used that path.
  - The (not_converged, energy) flags are stored in `.meta.pkl`.
  - If MMFF throws for a molecule, the molecule is dropped from **both** pickles. This guarantees identical key sets (asserted), and the count is reported.
- **Interpolation recipe = the CTRL_rematch / S4 training recipe.**
  - Von Mises Hungarian (`TD/standardize_confs.py:90-93`, `TD/utils/standardization.py:181-188`), then DE with popsize 15, maxiter 15 and the H-inclusive objective (`TD/standardize_confs.py:104-107`), using the same `optimize_rotatable_bonds`.
  - This makes the test-time λ = 0 L-distribution match CTRL_rematch's and S4's training distribution.
- **Per-seed metadata** (`<pickle>.meta.pkl`): `{raw_smi: [{src_gt_idx, kind, lam, sigma, seed, stereo_ok, pair_ok, heavy_rmsd, n_swaps, worst_bond_dev, min_nonbonded, mmff_not_converged}]}`. Plus `<pickle>.population.txt` with the key count and the list of molecules that were dropped, with reasons.
- **Pickle assertions before writing**, per molecule:
  - every mol has exactly 1 conformer, because the positions are read from `GetConformers()[0]` (`TD/diffusion/sampling.py:80`);
  - `assert_same_graph(seeds + [GT conf0])`;
  - all positions are finite;
  - cycle family: `len(list) == L(clean_confs)`; `etkdg`: `len == 2·n_csv`, as `TD/generate_confs.py:179`.
- **Pairing is fixed.** `--seed_confs_cycle` everywhere (F3). For `etkdg`, K = 2·n_csv seeds with cycle means each seed is used exactly once, which is equivalent to the SMILES path except for graph origin.

### 3.2 F5: achieved L error and floors
- `tools/l_error_report.py` (new) writes `l_error_<pickle>.csv` with one row per seed:
  - **Cycle family:** `l_subset_errors(seed, Y_ref)` against its source GT conformer. The reference is the relabelled GT (stored in meta), so that H angles compare like-for-like.
  - **`etkdg`/`mmff`/`xtb`:** against the Hungarian-assigned GT conformer on an L-error cost (heavy atoms, plus all atoms after `permute_terminal`).
  - Columns: bond/angle RMSD × {all, heavy} × {ring, acyclic}, plus the endocyclic dihedral RMSD.
- **CPU torsion-oracle floor.** `tools/local_structure_analysis.py --mode run` (`:351-404`) on the CTRL_rematch s0 run and the B1 s0 run of every condition: 11 × 2 jobs.
  - This is the floor for *exactly* those seeds and K.
  - Compare arms by B1 − CTRL per condition, not by absolute curves (F5).
- `tools/dose_response.py` (new):
  - B1 − CTRL per condition from `paired_compare` outputs, plotted against the achieved angle-RMSD and bond-RMSD columns;
  - ε* = linear interpolation between adjacent conditions where the sign changes;
  - CI by joint molecule bootstrap;
  - the noise series and the λ series reported **separately**, because white noise is not RDKit's structured error (verify_B §4 R2-1).

### 3.3 Runs (new TSV, generated)
- `slurm/make_round2_inference_tsv.py` writes `slurm/ablations_inference_round2.tsv`, so there are no hand-typed lines.
- Line format (`slurm/ablation_inference_array.sbatch:37`): `NAME | MODEL | 20 | 0 | 1 | --seed_confs $QM9_DIR/round2_seeds/<pickle> --seed_confs_cycle [--l_level x]`.
- Example lines:
```
S1_noise0.02pa_cyc_ORACLE | $WORK/qm9_B1_train_gtL_e100_s0       | 20 | 0 | 1 | --seed_confs $QM9_DIR/round2_seeds/L_noise0.02pa_rs1_cyc.pkl --seed_confs_cycle
S1_etkdg2L_mmff           | $WORK/qm9_CTRL_rematch_100ep_e100_s1 | 20 | 0 | 1 | --seed_confs $QM9_DIR/round2_seeds/L_etkdg2L_mmff_rs1.pkl --seed_confs_cycle
```
- **Required patch.** `slurm/ablation_inference_array.sbatch:45` assigns `MODEL_DIR=$MODEL` without expansion. Only ARGS is `eval`-expanded (`:40`). Change it to `*) MODEL_DIR=$(eval echo "$MODEL") ;;`.
- **Model × condition matrix.**
  - CTRL_base ×3 and B1 ×3 on all 11 conditions: 66 runs.
  - CTRL_rematch ×3 on the 9 conditions that S2–S4 reuse (etkdg2L, mmff, noise ×3, λ ×4): 27 runs. These are the F4 controls for S3/S4.
  - **Pipeline check:** `S1_lam1.00_cyc_ORACLE` on CTRL_rematch s0 and B1 s0 (2 runs). It must reproduce the existing `steps20_seed0_gtLcycle` of the same model within 0.003 Å MAT-R. The only differences are a rigid motion and a terminal-atom automorphism, and the model is E(3)-invariant. This is the end-to-end proof that alignment, relabelling and seed-pickle plumbing are correct.
  - Total: 95 runs, about 24 GPU-h at 0.25 h per run.
- **Evidence.** E1, E2, E3, E19, E22, E28 (research_A R2A-1); B#1–3, B#6–7. S1.5: E22, E23; B#8, B#9.

---

## 4. Paired training pickles for S3/S4: `tools/build_paired_pickles.py` (CPU)

Inputs:
- `$QM9_DIR/standardized_pickles_rematch/NNN.pickle`, the CTRL_rematch data (`slurm/standardize_qm9.sbatch:17-19`);
- the raw GEOM pickles `$QM9_RAW/<name>.pickle`.

Output: `$QM9_DIR/standardized_pickles_paired/NNN.pickle`. It keeps the same file layout and keys, because `TD/utils/dataset.py:101,132` locate a molecule by `i // 1000`.

For each std molecule `name` and each std conformer `c`:
1. **Find its raw GT conformer `r`.**
   - Primary key: `c['geom_id']`. The std conf dict *is* the raw conf dict, mutated in place at `TD/standardize_confs.py:116-119`, so every raw key survives.
   - That GEOM QM9 conf dicts carry `geom_id` is UNVERIFIED. Check it first with a one-liner on the cluster.
   - Fallback: the tuple (`boltzmannweight`, `totalenergy`), which must be unique.
   - The match must be exactly one conformer, otherwise abort.
2. **`assert_same_graph([c['rd_mol'], r['rd_mol']])`.**
   - `c['rd_mol']` has the graph of raw conformer 0 (`TD/standardize_confs.py:73`).
   - This checks the GEOM same-atom-order assumption per pair.
3. **Align.** X = RDKit positions, Y = GT positions, then `align_pair` → Y_al.
   - **Pairing check (decisive):** |heavy_rmsd − c['rmsd']| < 1e-3 Å, because `c['rmsd']` is exactly the heavy, identity-map, proper-rotation alignment RMSD of these two coordinate sets (`TD/standardize_confs.py:108,117`).
   - A wrong pairing or a wrong atom order would almost surely fail it.
   - If more than 0.1% of conformers fail, abort the build. Otherwise drop the failing conformers and list them.
4. **Store.**
   - `c['gt_pos_aligned'] = Y_al.astype(float32)`.
   - `c['pair_ok']`, `c['stereo_ok']`, `c['pair_heavy_rmsd']`, `c['n_swaps']`.
   - `rd_mol` is left untouched.
5. **Population report.**
   - Molecules and conformers per file, which must equal the rematch counts minus listed drops.
   - The `pair_ok` fraction. **Gate:** ≥ 98% of pairs `pair_ok`, otherwise stop and review before S4.
   - Distributions of `worst_bond_dev`, `min_nonbonded`, `n_swaps` and `stereo_ok`.

Then `featurize_qm9.sbatch VARIANT=paired` writes `cache_paired`.
- Add `paired) STD="$QM9_DIR/standardized_pickles_paired" ;;` to the case blocks at `slurm/featurize_qm9.sbatch:29-38` and `slurm/ablation_train_array.sbatch:46-55`.
- After the build, assert that `cache_paired.train` has the same `canonical_smi` list as `cache_rematch.train`.

The S3 GT half uses `gt_pos_aligned` too.
- This is equivalent to raw GT up to a rigid motion and a graph automorphism of terminal atoms.
- The model is E(3)-invariant (torsion scores are 0o/0e outputs, `TD/diffusion/score_model.py:118`) and permutation-equivariant over atoms with identical features.
- So the training signal is identical, and only one extra array is stored. Tested in §9 C9.

---

## 5. Training-side code changes (`TD/`)

### 5.1 `TD/utils/parsing.py` (after `:29`)
```python
parser.add_argument('--l_jitter', type=float, default=0.0, help='[round2 S2] per-axis Cartesian sigma (A) added to ALL atoms of the selected conformer, fresh per sample')
parser.add_argument('--l_mix_p_gt', type=float, default=0.0, help='[round2 S3] P(use paired GT L) per sample; needs paired pickles')
parser.add_argument('--l_interp', action='store_true', default=False, help='[round2 S4] x = (1-lam) x_rdkit + lam x_gt_aligned, lam~U[0,1]; needs paired pickles')
parser.add_argument('--lambda_embed_dim', type=int, default=0, help='[round2 S4] sinusoidal lambda embedding size (0 = architecture unchanged)')
```
Names must not collide with `generate_confs.py` CLI arguments. The training yaml is merged into the generate args (`TD/generate_confs.py:84-92`), so the inference flag is a different name: `--l_level`.

### 5.2 `TD/utils/dataset.py`
**Transform** (`:18-43`). The default path is kept *byte-identical*, so all round-1 arms reproduce exactly.
```python
class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi, boltzmann_weight=False,
                 l_jitter=0.0, l_mix_p_gt=0.0, l_interp=False):
        ...
        assert not (l_mix_p_gt > 0 and l_interp), 'S3 and S4 are separate arms'
        self.l_jitter, self.l_mix_p_gt, self.l_interp = l_jitter, l_mix_p_gt, l_interp
        self.needs_gt = l_mix_p_gt > 0 or l_interp

    def __call__(self, data):
        if not self.needs_gt:                                   # original lines :26-29, unchanged
            if self.boltzmann_weight: data.pos = random.choices(data.pos, data.weights, k=1)[0]
            else: data.pos = random.choice(data.pos)
        else:
            assert hasattr(data, 'gt_pos') and len(data.gt_pos) == len(data.pos), 'stale cache: no paired gt_pos'
            k = random.choices(range(len(data.pos)), data.weights, k=1)[0] if self.boltzmann_weight \
                else random.randrange(len(data.pos))           # an INDEX, so RDKit and GT stay paired
            x, y = data.pos[k], data.gt_pos[k]
            if self.l_interp:
                lam = float(np.random.uniform())
                if not data.pair_ok[k]:
                    lam = float(lam >= 0.5)                     # unsafe pair: snap to an endpoint, label stays truthful
                data.pos = (1 - lam) * x + lam * y
                data.node_lambda = lam * torch.ones(data.num_nodes)
            else:
                data.pos = y if np.random.uniform() < self.l_mix_p_gt else x
            data.l_conf_idx = k                                 # for tests only (collates to a tensor)
            del data.gt_pos, data.pair_ok                       # data is a deepcopy (get(), :193); never collate the lists
        if self.l_jitter > 0:
            data.pos = data.pos + self.l_jitter * torch.randn_like(data.pos)   # fresh per access; torch RNG
        ... # :31-43 unchanged (torsion noise applied to the (jittered / interpolated) conformer)
```
- **Why this is correct for the score target.** The torsion perturbation is applied to whatever `data.pos` holds (`:41`). Its increments are the target (`:42`; loss at `TD/utils/training.py:19-25`). The forward kernel is defined relative to the jittered or interpolated conformer, so the target needs no change.

**Featurization** (`:207-239`): carry the paired arrays *inside the same loop iteration* that appends `pos`. Conformers that "reacted" are skipped at `:219-220` before the append, so the pairing survives that filter.
```python
pos, weights, gt_pos, pair_ok = [], [], [], []
for conf in confs:
    ...                                                   # :210-221 unchanged
    pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
    weights.append(conf['boltzmannweight'])
    if 'gt_pos_aligned' in conf:
        gt_pos.append(torch.tensor(conf['gt_pos_aligned'], dtype=torch.float))
        pair_ok.append(bool(conf['pair_ok']))
...
if gt_pos:
    assert len(gt_pos) == len(pos) and all(g.shape == p.shape for g, p in zip(gt_pos, pos))
    data.gt_pos, data.pair_ok = gt_pos, pair_ok
```

**Stale-cache guard.** After loading the cache (`:67-70`), when `transform.needs_gt` holds, assert that every datapoint has `gt_pos`. This needs the transform to be passed in before the check; `construct_loader` already builds it first (`:255-256`).

**`construct_loader`** (`:255-256`): pass the three new args. The validation loader uses the **same** transform (`:259-266`). So val loss, and with it `best_model.pt` selection (`TD/train.py:30-39`), is measured on the arm's own training distribution. This is as in round 1, where B1 validated on GT L and CTRL on matched L. It is stated in the results.

### 5.3 `TD/diffusion/score_model.py` (S4 only; λ enters where σ enters)
- `__init__` (`:49-52`): add `lambda_embed_dim=0`, store it as `self.lambda_embed_dim`.
- `node_embedding` (`:65-69`): input `in_node_features + sigma_embed_dim + lambda_embed_dim`.
- `edge_embedding` (`:71-75`): input `in_edge_features + sigma_embed_dim + lambda_embed_dim + radius_embed_dim`.
- `build_conv_graph` (`:188-193`, where the σ embedding is built and concatenated):
```python
node_sigma_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim)          # :189
if self.lambda_embed_dim > 0:
    assert hasattr(data, 'node_lambda'), 'lambda-conditioned model needs data.node_lambda'
    node_lam_emb = get_timestep_embedding(data.node_lambda * 10000, self.lambda_embed_dim)  # same scale as :188
    node_sigma_emb = torch.cat([node_sigma_emb, node_lam_emb], 1)                  # rides along both concats below
edge_sigma_emb = node_sigma_emb[edge_index[0].long()]                              # :191
edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)                              # :192
node_attr = torch.cat([data.x, node_sigma_emb], 1)                                 # :193
```
- λ is a scalar (0e) input, like σ, so equivariance and parity are untouched. The parity test is repeated with λ (§9 C3).
- `TD/utils/utils.py:9-18` `get_model`: pass `lambda_embed_dim=getattr(args, 'lambda_embed_dim', 0)`. Old yamls have no such key, so they get 0, the architecture is unchanged, and strict loading (`TD/generate_confs.py:102`) of the released, CTRL and B1 checkpoints keeps working.

### 5.4 Inference: `TD/generate_confs.py`, `TD/diffusion/sampling.py`
- **`TD/generate_confs.py`.**
  - Add `--l_level` (float, default None) next to `:51`.
  - Add `'l_level'` to the CLI-restore loop at `:91-92`.
  - After the model build (`:99-104`):
    - if `getattr(args, 'lambda_embed_dim', 0) > 0` and `args.l_level is None`, call `sys.exit('lambda-conditioned model: pass --l_level')`;
    - in the opposite case, also `sys.exit`;
    - require 0 ≤ l_level ≤ 1.
  - Pass `l_level=args.l_level` into `sample(...)` (`:146-155`).
- **`TD/diffusion/sampling.py:105-115`** `sample(..., l_level=None)`. After `data_gpu = copy.deepcopy(data).to(device)` (`:168`):
```python
if l_level is not None:
    data_gpu.node_lambda = float(l_level) * torch.ones(data.num_nodes, device=device)
```
  - This is set once per batch. `model(data_gpu)` returns the same object, so the value persists through the σ loop (`:169-173`). `divergence` reuses `data_gpu` (`TD/diffusion/likelihood.py:12-16,35-37`).
  - **λ never comes from GT at test time.** It is a single CLI scalar for all molecules. Assert `torch.unique(data_gpu.node_lambda).numel() == 1` once per batch.
  - Any λ-series pickle evaluated with λ_in = λ_build is ORACLE by construction of the *L*, and is named so.

### 5.5 Slurm (`slurm/`)
- **`ablation_train_array.sbatch`.**
  - Add `paired` to the case block (`:46-55`).
  - Add a 7th TSV field `GT_GEN_ARGS`: read it at `:39`, and append it to the two GT runs at `:82,85`. The S4 GT runs need `--l_level 1`.
- **`common.sh` `gen_eval` (`:96-117`): provenance guard against stale outputs.** Today it skips generation if `confs.pkl` exists (`:102`), whatever the inputs were.
```bash
local new; new=$( { printf 'model=%s\nargs=%s\n' "$model_dir" "$*"; sha256sum "$model_dir/best_model.pt";
                    for a in "$@"; do [[ -f $a ]] && sha256sum "$a"; done; } )
if [[ -s $out/provenance.txt ]]; then
    [[ "$(cat "$out/provenance.txt")" == "$new" ]] || { echo "ERROR: $out was produced from other inputs"; return 1; }
else echo "$new" > "$out/provenance.txt"; fi
```
  - Round-1 directories get a provenance file on their next touch. No round-1 directory is re-generated.
- **`ablation_inference_array.sbatch:45`**: the `eval` expansion of MODEL (§3.3).
- New: `slurm/build_round2_data.sbatch`, CPU, 32 cores, `VARIANT`-like `STEP` switch: `paired | seeds_etkdg | seeds_noise | seeds_interp_a5 | l_error | floors | s0`.
- New: `slurm/smoke_round2.sbatch` (§9.3).

---

## 6. Per-arm specifications

### S1: see §3. Test-time L sources, ORACLE labels, F2/F3/F5 built in.
- **xTB (S1.6).** Attempt `conda install -c conda-forge xtb` (or a static binary) on gnode118 with a 30-minute time box.
- If it works:
  - `--source etkdg --xtb`, using `TD/utils/xtb.py:44` (per-PID `/tmp` work dir, multiprocessing-safe per verify_A §2);
  - 9 extra runs (CTRL_base, B1, CTRL_rematch ×3), 2.25 GPU-h.
- Otherwise record "skipped: xtb install > 30 min".

### S2: B6 noise-augmented GT-L training
- **Code:** §5.1 and §5.2 (`--l_jitter`).
- **Data:** existing `cache_raw` (`slurm/featurize_qm9.sbatch:39`). Jitter is applied in the transform, never cached, so no new cache.
- **TSV** (`slurm/ablations_train_round2.tsv`, 7 fields):
```
B6_jit0.04pa   | raw | --l_jitter 0.04 |  | 1 | 0 |
B6_jit0.04pa   | raw | --l_jitter 0.04 |  | 1 | 1 |
B6_jit0.04pa   | raw | --l_jitter 0.04 |  | 1 | 2 |
B6_jit0.02pa   | raw | --l_jitter 0.02 |  | 1 | 0 |
```
- σ is per axis and stated in the names (F6). σ = 0.04 per axis gives about 0.057 Å per bond endpoint pair and about 3.3° per angle (verify_A §3 simulation).
- **Fresh vs fixed.**
  - `get()` returns a deepcopy (`TD/utils/dataset.py:193`), and the transform runs at every access, so the jitter is fresh per sample and per epoch.
  - DataLoader workers (`--loader_workers 8`, `slurm/ablation_train_array.sbatch:70`): torch reseeds python/torch/numpy per worker from a base seed drawn from the main RNG each epoch (torch ≥ 1.9 behaviour; the cluster has torch 1.13.1, `slurm/setup_env.sh:76`).
  - Both are tested (§9 C5, C6).
- **Evaluation.**
  - Training job: RDKit, gtL, gtLcycle.
  - Plus S1 conditions etkdg2L, mmff, noise ×3 and λ ×4, for 4 models: 36 runs.
- **Pre-registered success:**
  - (a) non-inferior to CTRL_base on RDKit L, margin +0.004 Å AMR-R;
  - (b) better than CTRL_base on gtLcycle (ORACLE).
- **Single-factor control:** B1. B6_jit0.02pa is exploratory (1 seed).
- **Evidence:** E5, E8, E9, E10 (research_A R2A-4); B#23, B#24, B#26. E9/E10 are an analogy (verify_A §1).

### S3: mixed-L training
- **Code:** §5.2 (`--l_mix_p_gt 0.5`).
- **Data:** `standardized_pickles_paired` and `cache_paired` (§4).
- **Equal caps by construction.** Every datapoint has the same ≤30 conformers in both halves (`len(gt_pos) == len(pos)` is asserted), so the per-molecule GT:RDKit ratio is 1:1 in expectation. This fixes verify_A R2A-3(c).
- **TSV:**
```
S3_mixL_p0.5   | paired | --l_mix_p_gt 0.5 |  | 1 | 0 |
S3_mixL_p0.5   | paired | --l_mix_p_gt 0.5 |  | 1 | 1 |
B1cap_gtL      | paired | --l_mix_p_gt 1.0 |  | 1 | 0 |      # recommended extra control (see §0.7)
```
- **Controls:** CTRL_rematch (RDKit end; identical RDKit data) and B1 (GT end), plus B1cap for the cap-matched GT end.
- **Evaluation:** 2 models × 9 S1 conditions = 18 runs.
- **Evidence:** E5, E6, E7 (research_A R2A-3).

### S4: λ-conditioned interpolation training
- **Code:** §5.2 (`--l_interp`), §5.3, §5.4.
- **Data:** `cache_paired`.
- **λ distribution.** λ ~ U[0,1] as the shortlist specifies (research_A had proposed 25% λ = 0, 25% λ = 1, 50% U).
  - **Raised:** with U[0,1], the λ = 0/1 test points lie at the edge of the training support. I keep the shortlist's choice and pre-declare it.
- **Unsafe pairs.** For pairs that fail `pair_ok` (stereo mismatch, terminal mis-assignment, clash), λ is snapped to an endpoint. Their L is never a chemically impossible midpoint, and the λ label stays truthful.
- **TSV:**
```
S4_lamcond     | paired | --l_interp --lambda_embed_dim 32 | --l_level 0 | 1 | 0 | --l_level 1
S4_lamcond     | paired | --l_interp --lambda_embed_dim 32 | --l_level 0 | 1 | 1 | --l_level 1
S4_lamcond     | paired | --l_interp --lambda_embed_dim 32 | --l_level 0 | 1 | 2 | --l_level 1
```
- **Test conditions.**
  - Training job: RDKit at λ = 0; gtL and gtLcycle at λ = 1 (ORACLE).
  - Inference TSV:
    - etkdg2L at λ = 0;
    - mmff at λ ∈ {0, 0.25, 0.5}, pre-registered and descriptive only, with no selection on test;
    - λ-series pickles with λ_in = λ_build (ORACLE).
  - 8 runs per model, 24 runs in total.
- **Controls:** CTRL_rematch (λ = 0 end), and B1cap / B1 (λ = 1 end). The cap confound is handled by B1cap (verify_A R2A-2(c)).
- **Evidence:** E5, E6. E15, E16 and E24 are weak or analogy only (verify_A §1).

### S5: B3, MMFF-matched TD
- **Code:** none.
- **Data:**
  - `standardize_qm9.sbatch VARIANT=mmff` (`--mmff`, `TD/standardize_confs.py:18,83-87`), then `featurize_qm9.sbatch VARIANT=mmff`.
  - Start it at t0. The 12 CPU-h estimate is UNVERIFIED (verify_B R2-4).
- **TSV:** the existing deferred line (`slurm/ablations_train_deferred.tsv:4`), copied into the round-2 table for seeds 0–2: `B3_match_mmff | mmff | | --pre_mmff | 0 | s |`.
- **Inference lines:**
  - `S5ctrl_pre_mmff` on CTRL_rematch s0–2 (`--pre_mmff`, SMILES path, where MMFF does act: `TD/generate_confs.py:136-137`);
  - `S5_nommff` on B3 s0–2 (the train-only cell).
- **Report:**
  - `mmff_error`: the last cumulative `long_term_log` print in `standardize_qm9` worker logs (`TD/standardize_confs.py:156-162`);
  - the molecule-count difference `cache_mmff` vs `cache_rematch` (composition change).
  - `mmff_func` ignores convergence flags (`TD/utils/standardization.py:191-198`); note it as a limitation.
- **Evidence:** E2, E3; B#7.

### S6: sampler checks
- **TSV:** reuse the deferred lines (`slurm/ablations_inference_deferred.tsv:7,8,10,12`), adding `_seed1` and `_seed2` copies on PRE.
- Add 4 lines on `$WORK/qm9_B1_train_gtL_e100_s0` with `--seed_confs $QM9_GT_SEED_CONFS --seed_confs_cycle`. That is the gtLcycle anchor (F3), descriptive (verify_B R2-6).
- 16 runs.
- **Evidence:** B#19 (FM-refiner Table 6; model-dependent per verify_B).

---

## 7. Validity-check list (risk → concrete check → where → action)

| # | risk | check (assertion / test) | where | on failure |
|---|---|---|---|---|
| V1 | Atom order of the GT mol ≠ the matched RDKit mol | `assert_same_graph` per pair, and heavy Kabsch RMSD == stored `conf['rmsd']` (±1e-3) | `build_paired_pickles.py` | drop the pair, list it; abort if > 0.1% |
| V2 | GEOM conformers of one molecule differ in atom order (B1, gt seeds, S2) | `assert_same_graph` over all conformers of every molecule | `data_validity_scan.py` (S0) | flag molecules in every GT-L table |
| V3 | H vs heavy: H rotors dominate the fit; identity-map H interpolation collapses C–H | heavy-only fit; `permute_terminal`; `worst_bond_dev ≤ 0.05 Å` | `lgeom.align_pair`, `interp_safety` | `pair_ok = False` → endpoint snap (S4), flagged (S1) |
| V4 | Chirality flip from interpolation (enantiomeric pair passes through a planar centre) | `stereo_labels` equal at both endpoints (`tools/local_structure_analysis.py:217-222`) | paired + seed builders | `pair_ok = False` |
| V5 | Chirality flip from noise / A5 SetAngle | stereo labels of every built seed == source GT; count flips (expected 0 at σ ≤ 0.04) | seed builder | drop the seed's molecule from that pickle, report it |
| V6 | Rings "breaking" or clashing under noise / interpolation | ring-bond max deviation and `min_nonbonded ≥ 0.9 Å` reported per pickle; graph topology is fixed by featurization (`TD/utils/featurization.py:86-95`), so only geometry can degrade | `l_error_report.py`, `interp_safety` | report; exclude-sensitivity analysis |
| V7 | A5 arms built by different mechanisms | the same `set_internal_subset` on the same matched pair; assert the A5-ring ring subset error == 0 and the A5-acyc ring subset == RDKit; report the achieved acyclic residual | builder + T8 | report; stratify |
| V8 | Stale featurization cache (`TD/utils/dataset.py:67-70` reuses any file) | per-variant cache name `cache_paired`; `needs_gt` ⇒ every datapoint has `gt_pos`; `cache_paired` SMILES == `cache_rematch` | dataset + featurize | fail fast |
| V9 | Smoke run poisons a production cache: `--limit_train_mols` is applied *before* caching (`:87-88`) but *after* loading (`:79-80`) | smoke uses `$QM9_DIR/smoke/cache_*` only; `smoke_round2.sbatch` refuses a cache path without `/smoke/` | smoke script | abort |
| V10 | Stale generation outputs (`common.sh:102` skips if `confs.pkl` exists) | `provenance.txt` (args + sha256 of checkpoint and seed pickles) | `gen_eval` | abort that run |
| V11 | Seeds | RDKit `randomSeed ≥ 1` asserted; training seeds in TSV field 6; generation seed 0 everywhere (paired across models); builder seeds in file names (`_rs1`); noise uses common random numbers via `default_rng([seed, mol_idx, i])` | builders, TSV | n/a |
| V12 | Jitter fixed once instead of fresh / duplicated across workers | C5, C6 (§9) | cluster CPU test | fix before T1 |
| V13 | Test molecules leaking into training | split overlap scan (S0 5b); paired builder adds only GT of molecules already in the std pickles, and the split filter stays in `preprocess_datapoints` (`:85-97`); all test-GT-derived L are labelled ORACLE | S0 | report |
| V14 | λ leaking GT information at test time | λ is a single CLI scalar; per-batch assertion that it is constant; no per-molecule λ; validation-only selection (none this round) | `sampling.py` | abort |
| V15 | Pickles silently dropping molecules | `population.txt` per pickle; paired = rematch counts; eval `n_evaluated` + `num_failures` per file; intersection analysis; B1 − CTRL contrasts use the *same* pickle, so populations cancel | all builders, `paired_compare` | report |
| V16 | Bernoulli ratio / cap wrong | `tools/train_input_stats.py`: 20k transform draws → GT fraction 0.50 ± 0.01; λ histogram; achieved L error of training inputs per λ bin and per jitter σ (should give ≈0.057 Å / 3.3° at σ = 0.04) | CPU on cluster | fix before T3 |
| V17 | Checkpoint incompatibility / architecture regression | strict-load regression test with outputs identical to the pre-patch code (C1) | cluster CPU test | fix |
| V18 | gtLcycle anchor vs λ = 1 plumbing | `lam1.00` pickle reproduces gtLcycle MAT-R within 0.003 Å | inference | stop the S1 λ analysis |
| V19 | MMFF composition / convergence | identical key sets for the twin pickles; not-converged fraction; `mmff_error` count for B3 | builder, S5 | report |
| V20 | Multi-fragment molecules | excluded consistently (`TD/diffusion/sampling.py:38-40`); listed in S0 | S0 | n/a |
| V21 | Val selection on a noisy val set | val uses the same transform; best epoch logged; report last-epoch vs best-epoch AMR on 1 seed of B6 as a sensitivity check | analysis | report |

---

## 8. S4 recommendation: yes, last wave, gated

- **Why it fits.** S4-specific code is small (§5.3, §5.4, the λ branch in §5.2). It is fully covered by C1–C4 (strict-load regression, λ assertion, parity with λ, λ sensitivity). The heavy machinery (paired pickles, alignment, interpolation safety) is needed for S3 and the S1 λ series anyway.
- **Go/no-go gate** at about t = 30 h, before wave T4. All of these must hold:
  1. all tests in §9 pass on the cluster;
  2. the paired build passes V1 (≤ 0.1% pairing failures) and has ≥ 98% `pair_ok`;
  3. the S1 λ-series pickles show monotone achieved L error in λ, and V18 holds;
  4. the S4 smoke run completes and `generate_confs` exits non-zero without `--l_level`.
- **If any check fails,** S4 moves to round 3. T4 then runs B1cap s1/s2 (or more inference), with no idle GPUs.
- **My vote: include S4 (conditional).**

---

## 9. Tests

### 9.1 Local unit tests: `tools/tests/test_lgeom.py`
- Local Python has rdkit 2026.03.6, torch 2.14 CPU, numpy, scipy and pytest 9.1. It has no torch_geometric, e3nn or spyrmsd, so only geometry tests run here.

| test | assertion |
|---|---|
| T1 Kabsch | random rotation + translation → RMSD < 1e-6; equals `AlignMol(RemoveHs, RemoveHs)` on an ETKDG pair to 1e-6; mirror input stays high (proper rotations only) |
| T2 pairing check | for a DE-matched pair built with `optimize_rotatable_bonds`, `align_pair(...).heavy_rmsd == AlignMol` value |
| T3 terminal relabelling | a methyl rotated 120°: identity midpoint C–H < 0.7 Å, after `permute_terminal` within 0.03 Å of the endpoint mean (local check: 0.62 → 1.11 Å); idempotent; CF3 F's relabelled; nitro O's (different bond order or charge) never swapped |
| T4 interpolation | λ = 0/1 reproduce X/Y_al exactly; achieved bond/angle RMSD to GT decreases monotonically over λ on 20 QM9-like SMILES |
| T5 stereo | enantiomer pair → `stereo_ok = False`, `pair_ok = False`; diastereomer likewise |
| T6 noise | empirical per-axis SD ≈ σ (n = 1e5, ±2%); C–C bond RMSE ≈ √2σ (0.057 Å at 0.04, cf. verify_A table); common random numbers: noise(0.04) / noise(0.02) == 2 exactly; 0 stereo flips at σ = 0.04 on the 20 SMILES |
| T7 RDKit seed guard | builder raises for `randomSeed=0` |
| T8 A5 partition | A5-ring: ring bonds, ring angles and endocyclic dihedrals equal GT to 1e-6; A5-acyc: ring subset equals RDKit to 1e-6; acyclic residual after 3 passes reported (and below 1 pass) |
| T9 builder assertions | one conformer per seed mol; `assert_same_graph` catches a permuted-atom-order copy (Chem.RenumberAtoms) |

### 9.2 Cluster CPU tests: `tools/tests/test_round2_model.py`
Run with `srun -w gnode118 -p plafnet2 -A plafnet2 -c 8 --gres=gpu:0 -t 1:00:00` in `$VENV` (torch 1.13.1, PyG 2.0.4, e3nn 0.5.1, rdkit 2022.9.5).
- Also re-run `test_lgeom.py` there, to cover RDKit-version differences.

| test | assertion |
|---|---|
| C1 regression | before patching, save `edge_pred` of the released `qm9_default` on a fixed batch (seeded); after the patch, `lambda_embed_dim=0` loads strictly and outputs are bitwise equal |
| C2 λ plumbing | λ-model without `node_lambda` → AssertionError; output differs between λ = 0 and λ = 1 |
| C3 symmetry | λ-model: rotation invariance (|Δ| < 1e-4) and the parity flip (as `tools/smoke_test.py:59-71`) |
| C4 generate guards | λ-model without `--l_level` → non-zero exit; non-λ model with `--l_level` → non-zero exit |
| C5 fresh jitter | `dataset[0]` twice → different pos; `datapoints[0].pos[0]` unchanged after 100 `get`s |
| C6 worker seeding | DataLoader with 2 workers over 2 epochs: no two jitter vectors identical across workers or epochs; same `--seed` → identical first batch (reproducible) |
| C7 pairing in transform | synthetic datapoint with `gt_pos[k] = pos[k] + k`: the result equals the formula for the drawn `l_conf_idx` (mix and interp) |
| C8 featurize pairing | a synthetic mol_dic with one "reacted" conformer: `gt_pos` stays aligned with `pos` |
| C9 GT relabel invariance | model output on raw GT vs `gt_pos_aligned` (rigid motion + terminal relabelling): the multiset of `edge_pred` per bond is equal (< 1e-4) |
| C10 stale cache | `--l_mix_p_gt 0.5` on `cache_rematch` (no `gt_pos`) → assertion fires |

### 9.3 20-molecule / 1-epoch smoke test: `slurm/smoke_round2.sbatch`
1 GPU, about 45 minutes, on gnode118:
1. Run the §9.1 and §9.2 tests.
2. Build a mini paired pickle from `standardized_pickles_rematch/000.pickle` into `$QM9_DIR/smoke/std_paired`.
3. Featurize with `--limit_train_mols 20 --n_epochs 0 --cache $QM9_DIR/smoke/cache_paired` (V9).
4. Train for 1 epoch, 20 molecules, with separate smoke caches and log dirs, for these configs:
   - `--l_jitter 0.04` on raw;
   - `--l_mix_p_gt 0.5`, `--l_mix_p_gt 1.0`;
   - `--l_interp --lambda_embed_dim 32` on paired.

   Assert:
   - the loss is finite;
   - the yaml contains the new keys;
   - `last_model.pt` reloads strictly.
5. Run `make_l_seed_pickles.py --limit_mols 20` for every source, then `l_error_report.py`, then the population assertions.
6. Run `generate_confs --limit_mols 20` with each model on etkdg2L, noise0.04 and lam0.50, plus the S4 model with and without `--l_level`. Then `evaluate_confs`, then `paired_compare --universe intersection`.
7. V18 in miniature: the released model on `lam1.00` vs the GT seed pickle with cycle, both on CPU for determinism. Per-molecule MAT-R |Δ| median < 0.005 Å.
8. Timing: generation for 20 molecules on GPU vs CPU-only. If CPU is less than 3× slower, the S1/S2 inference can run as CPU jobs in parallel with training (frees GPU-h; §10).

---

## 10. Four-GPU schedule and GPU-h

**Cost basis.**
- 10.7 GPU-h per 100-epoch training run (BRIEF).
- 0.25 GPU-h per inference + eval run (BRIEF).
- A training job also runs 1 or 3 default evaluations (`slurm/ablation_train_array.sbatch:77-86`).

**Lanes.**
- Training array `--array=...%3`.
- Inference array `%1`.
- Together they use at most 4 GPUs at any time; each array's throttle works as in round 1 (`slurm/ablation_train_array.sbatch:15-16`).

| time (h) | lanes 0–2 (training, %3) | lane 3 (inference, %1) | CPU (gnode118) |
|---|---|---|---|
| 0–1 | smoke_round2 (1 GPU) | — | cluster tests; start the mmff standardization; paired build → featurize paired; seed builders; S0 |
| 1–12.5 | **T1:** B6_jit0.04pa s0, s1, s2 | S6 (16 runs), S5ctrl_pre_mmff (3), A1c random-tors null (1) | seed pickles finish (DE matching about 3–10 h, research_A R2A-1); l_error; S0 |
| 12.5–24 | **T2:** B3_match_mmff s0, s1, s2 | S1 on CTRL_base / B1 / CTRL_rematch (95 runs, about 24 GPU-h, continues) | `--mode run` floors |
| 24–35.5 | **T3:** S3_mixL s0, s1; B6_jit0.02pa s0 | S1 continues; S2 extras (36 runs) | gate check at about 30 h |
| 35.5–47 | **T4:** S4_lamcond s0, s1, s2 (gated) — fallback: B1cap s0 + inference | S5_nommff (3), S3 extras (18), S4 extras (24, once T4 ends) | analysis |

- B1cap s0: if S4 runs, B1cap needs a 13th slot. Recommended in T3 instead of B6_jit0.02pa if the user prefers controls over exploration.

| arm | training GPU-h | inference GPU-h | total |
|---|---|---|---|
| S0 | 0 | 0.25 | 0.25 |
| S1 | 0 | 95 × 0.25 = 23.75 (+2.25 if xTB) | 23.75 |
| S2 | 4 × (10.7 + 0.75) = 45.8 | 36 × 0.25 = 9 | 54.8 |
| S3 | 2 × 11.45 = 22.9 | 18 × 0.25 = 4.5 | 27.4 |
| S4 | 3 × 11.45 = 34.4 | 24 × 0.25 = 6 | 40.4 |
| S5 | 3 × 10.95 = 32.9 | 6 × 0.25 = 1.5 | 34.4 |
| S6 | 0 | 16 × ~0.3 = 5 | 5 |
| smoke | 0.75 | — | 0.75 |
| **total** | **136.7** | **50** | **≈ 187 GPU-h (about 47–50 h wall on 4 GPUs)** |

(+ B1cap 11.45 + 2.25 if it is run as a 13th training run.)

**Trim options**, in order:
1. Move eval to CPU jobs if smoke step 8 allows. This removes most of the 50 inference GPU-h from the GPU lanes.
2. CTRL_base only s0 on the λ and A5 conditions: −12 runs, −3 GPU-h.
3. Drop the S4 mmff λ ∈ {0.25, 0.5} descriptive runs: −6 runs.
4. Defer S4 entirely: −40 GPU-h.

The cost of S5 standardization (about 12 CPU-h, unverified) and the seed builders (3–10 CPU-h) falls on CPU, not GPU.

---

## 11. File list (new or changed)

**Changed**
- `TD/utils/parsing.py`, `TD/utils/dataset.py`, `TD/diffusion/score_model.py`, `TD/utils/utils.py`, `TD/generate_confs.py`, `TD/diffusion/sampling.py`.
- `tools/paired_compare.py`, `tools/local_structure_analysis.py`.
- `slurm/common.sh`, `slurm/ablation_train_array.sbatch`, `slurm/ablation_inference_array.sbatch`, `slurm/featurize_qm9.sbatch`.

**New**
- `tools/lgeom.py`, `tools/build_paired_pickles.py`, `tools/make_l_seed_pickles.py`, `tools/l_error_report.py`, `tools/dose_response.py`, `tools/leak_test.py`, `tools/floor_summary.py`, `tools/data_validity_scan.py`, `tools/train_input_stats.py`.
- `tools/tests/test_lgeom.py`, `tools/tests/test_round2_model.py`.
- `slurm/ablations_train_round2.tsv`, `slurm/make_round2_inference_tsv.py` (writes `slurm/ablations_inference_round2.tsv`), `slurm/build_round2_data.sbatch`, `slurm/smoke_round2.sbatch`.

**Data on `/scratch`**
- `$QM9_DIR/standardized_pickles_paired/`, `$QM9_DIR/cache_paired.{train,val}`.
- `$QM9_DIR/standardized_pickles_mmff/`, `$QM9_DIR/cache_mmff.{train,val}`.
- `$QM9_DIR/round2_seeds/*.pkl` with `.meta.pkl`, `.population.txt` and `l_error_*.csv`.
- `$QM9_DIR/smoke/`.
