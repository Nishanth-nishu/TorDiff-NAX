# Metric verification (METRIC-VERIFIER, 2026-09-29)

Scope: every metric the QM9 ablations will report. That covers `evaluate_confs.py` on branch `flexitors-ablation-hooks` (upstream code plus the patch's SUMMARY and SWEEP additions) and the tools `tools/geometry_metrics.py`, `tools/local_structure_analysis.py` and `tools/breakdown.py`.

Sources are cited as *file, section/page*:
- **TD** = `papers/core/2022_jing_torsional_diffusion.pdf` (text in `.fulltext.txt`).
- **GS** = `papers/gap_synthesis.md`.
- **TA** = `papers/core/tordiff_analysis.md`.
- **CC** = `notes/code_concerns.md`.
- Related papers are cited from `papers/related/fulltext/*.txt`.

Tests: `review/tests/test_metrics.py`. It needs only RDKit, with no data and no GPU. Run it with `python review/tests/test_metrics.py`; it takes about 1 min. **All 35 checks pass** after the fixes below, on RDKit 2026.03.6, numpy 2.4.6 and scipy 1.17.1. The test exports `evaluate_confs.py` from the branch with `git show` and runs it end-to-end on synthetic pickles.

---

## 1. Verdict table: `evaluate_confs.py` (branch HEAD `6218977`)

`M` is the per-molecule matrix `rmsd[n_true=L, n_model=K]`.

| # | Item | Code (branch) | Literature definition | Verdict |
|---|---|---|---|---|
| 1 | **COV-R(δ)** | `coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0)` | COV-R := (1/L)·\|{l : ∃k, RMSD(C_k, C*_l) < δ}\|. TD Eq. 35, App. G.3, p. 24. GeoMol Eq. 5 (`2021_ganea_geomol.txt` l. 563-575) | **MATCHES**. Checked against a literal Eq. 35 implementation: `calc_performance_stats` on a hand matrix, and end-to-end |
| 2 | **AMR-R / MAT-R** | `amr_recall = rmsd_array.min(axis=1).mean()` | AMR-R := (1/L)·Σ_l min_k RMSD (TD Eq. 35, p. 24). MAT-R is the same quantity (GeoDiff Eq. 11, `2022_xu_geodiff.txt` l. 541) | **MATCHES** |
| 3 | **COV-P(δ)** | `np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1)` | "precision metrics are obtained by swapping ground truth and generated conformers" (TD G.3, p. 24; GeoMol l. 575) | **MATCHES** |
| 4 | **AMR-P / MAT-P** | `amr_precision = rmsd_array.min(axis=0).mean()` | same swap | **MATCHES** |
| 5 | Heavy atoms | `AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))` | Heavy-atom RMSD is TD's reference-code convention (TA table, p. 24; GS §6). Zhou et al. define RMSD over heavy atoms (`2023_zhou_dl_conformation_critique.txt` l. 479). The TD paper text itself does not say "heavy" | **MATCHES** the code that produced TD's tables. The paper wording is AMBIGUOUS (poll P2) |
| 6 | Symmetry-aware RMSD | `GetBestRMS` (not `AlignMol`, unless `--only_alignmol`) | G.3 p. 24: XL RMSDs are computed "without testing all possible symmetries … therefore … an upper bound". This implies DRUGS and QM9 RMSDs are symmetry-aware | **MATCHES**. Verified: a renumbered copy gives 0; swapping symmetric methyls gives AlignMol 0.954 Å and GetBestRMS 0; the matrix equals an independent Kabsch + automorphism RMSD to 3e-8 Å. **Caveat:** proper rotations only, so a mirror image gives 0.423 Å (reflection is not allowed, which is chemically correct). The RDKit default `symmetrizeConjugatedTerminalGroups=True` makes the result **RDKit-version dependent**: a COOH O/O swap gives 0.000 Å by default and 0.090 Å with the flag off (poll P3) |
| 7 | Strict `<` | `< threshold` | TD Eq. 35 and GeoMol Eq. 5 use `< δ`. GeoDiff Eq. 10 (l. 536-538) and EBD use `≤ δ` (GS §6) | **MATCHES** TD. Tested: an RMSD of exactly 0.5 is *not* covered at δ = 0.5. Numerically irrelevant for continuous RMSDs (poll P4) |
| 8 | δ = 0.5 Å for QM9 | grid `np.arange(0, 2.5, .125)[4] == 0.5` exactly. SUMMARY defaults to 0.5 when `--dataset qm9` | TD Table 7 caption, p. 26. GeoMol l. 580-581 ("0.5 Å for GEOM-QM9") | **MATCHES** |
| 9 | K = 2L | `sample_confs(raw_smi, 2 * n_confs, smi)` (`generate_confs.py`). L = `len(clean_confs(...))` in the evaluator | "we generate 2K conformers for a molecule with K ground truth conformers" (TD §4.2 p. 8; G.3 p. 24; GeoMol l. 586-587; GeoDiff l. 546-547) | **DEVIATES (minor, inherited from TD's own code).** K is 2 × the CSV count, counted *before* `clean_confs`, so K/L ≥ 2 whenever GT conformers are dropped. This raises COV-R slightly. It is identical to how TD's numbers were produced, so it does not affect comparability. `n_true` and `n_model` are saved in `eval.pkl`; report `mean(n_model != 2*n_true)` (poll P5) |
| 10 | Model failures | `+ [0] * num_failures` in COV; AMR uses `np.nanmean`/`np.nanmedian` over evaluated molecules only | Paper silent. GS §6 and CC A.5 document the code convention | **MATCHES** TD code. Semantically AMBIGUOUS: AMR improves when hard molecules fail (poll P1). The SUMMARY reports `n_model_failures` |
| 11 | "Additional failures" (GetBestRMS raises) | the whole GT row becomes NaN. `min(axis=0)` then NaN-poisons every column, so the molecule's COV-P is 0, COV-R counts that row as uncovered, and AMR-R/AMR-P are NaN (excluded) | not in any paper | **MATCHES** upstream. Tested with a molecule whose generated set has a different graph. One bad GT conformer zeroes the molecule's precision (poll P8). `n_additional_failures` is reported |
| 12 | GT filtering | `clean_confs`: non-isomeric canonical SMILES match | TD G.1 p. 23: molecules whose CREST conformers all fail the SMILES check are filtered | **MATCHES**. Tested: a wrong-SMILES GT conformer is removed (L = 1). A molecule with an empty GT set is skipped silently: it is not counted as a failure and is not in the denominator |
| 13 | Mean and median | means and medians over molecules of per-molecule values | TD Table 7 reports Mean and Med for each metric (p. 26). Macro-average (GS §6) | **MATCHES** |
| 14 | SUMMARY line (patch) | before the fix, took the nearest grid value to `--report_threshold`. Now computed at exactly that value from `results[..]['rmsd']` | n/a | **Bug fixed (B1, low severity)**. At 0.5 and 0.75 the numbers are unchanged, verified end-to-end |
| 15 | SWEEP lines (patch) | `np.min(r['rmsd'], axis=1) < thr`, failures count as 0, mean and median | same as rows 1 and 3 | **MATCHES**. Checked against an independent computation at 0.05, 0.1, 0.25 and 0.5 |
| 16 | AMR printed in every threshold block | threshold-independent by definition | Eq. 35 | **MATCHES** (it is cosmetic) |

## 2. New tools

| Tool / quantity | Check | Verdict |
|---|---|---|
| `geometry_metrics.py`: atom map gen→GT | `gt_h[0].GetSubstructMatches(gen_h[0], uniquify=False)` gives `amap[i]` = the GT index of gen atom `i`. Direction is correct: on a renumbered copy with identical geometry, bond MAE is 1e-16 Å and angle MAE 6e-15° | **MATCHES** |
| same: map *selection* | Old code chose the automorphism with the lowest **angle MAE**. That biases the angle MAE low (it is a min over maps) and can give the torsion error of a symmetry-equivalent permutation, unlike the GetBestRMS correspondence that paired (l, k) | **Changed** to the automorphism with the lowest aligned heavy-atom RMSD, i.e. GetBestRMS's own criterion (fix B3; poll P7) |
| same: torsions, periodicity | `abs((s - g + 180) % 360 - 180)` lies in [0, 180]. Tested: +179° vs −179° gives 2°. RDKit `GetDihedralDeg` is signed in (−180, 180] (tested) | **MATCHES** |
| same: torsion set | Before the fix: all acyclic heavy bonds with heavy neighbours on both sides, **including bonds at sp centres** (C≡C, C≡N and the single bonds next to them). The dihedral a-u-v-d is ill-conditioned there, because an angle is about 179°. **A 0.02 Å wiggle of a nitrile N gave a 28.8° torsion MAE** (RMSD 0.010 Å) | **BUG (B3), fixed**: bonds with an sp-hybridised endpoint are skipped. After the fix the same test gives 0.0°. Known limit: the relative rotation across a linear chain (R-C≡C-R') is not measured |
| same: units | bond MAE in Å, angle and torsion MAE in degrees, RMSD in Å | OK |
| same: GT atom order | one map (computed on `gt_h[0]`) is applied to every GT conformer. This assumes all GEOM conformers of a molecule share an atom order, the same assumption upstream `standardize_confs.py` makes with its identity-map `AlignMol` | AMBIGUOUS, low risk (open question Q3) |
| `local_structure_analysis.py`: RMSD in transplant and floor | Before the fix: `AlignMol` with the **identity** heavy map, so it was not symmetry-aware. Consider an isopropyl or gem-dimethyl centre whose two equivalent groups are embedded in swapped pro-R/pro-S positions. No torsion change can fix that under the identity map, but GetBestRMS (in the evaluator) matches it. **Test: CC(C)CC#N floor 0.459 Å → 0.073 Å after the fix.** This inflated `floor_rmsd` and deflated the "COV-R ceiling" in `breakdown.py`, which is the core FlexiTors evidence (plan J/A0) | **BUG (B2), fixed**: `heavy_automorphisms()` gives all heavy-atom automorphism maps, and the DE objective, the transplant cost and the run-matched floor use the min over maps. Tested equal to `GetBestRMS(RemoveHs)` to 9e-11 Å on 4 molecules. The residual difference is RDKit's conjugated-terminal symmetrisation (e.g. COOH) |
| same: `relevant_torsions` | Same bridge rule as upstream `get_torsion_angles` (`utils/standardization.py:63-83`), restricted to bonds with ≥ 2 heavy atoms on both sides. Tested on CCCCO: repo count 4, heavy-relevant 2 | **MATCHES** |
| same: torsion floor | With identical L and scrambled torsions: 0.866 Å before optimisation, 3e-7 Å after. floor ≤ transplant always holds (tested on 3 molecules). Floors are finite for nitrile and alkyne molecules | OK. The floor is an upper bound because DE is heuristic |
| same: sp-centre torsions in `rel_t` | Setting an ill-conditioned dihedral (such as C-C≡C-C) to a GT value does not produce NaN (tested). It **randomises that DOF in `transplant_rmsd_*`** when seed and GT L differ. DE floors are unaffected, because DE explores the full rotation | minor, documented (not fixed) |
| same: bond and angle RMSD | same indexing (seeds are embedded from the GT mol). Units are Å and degrees | OK |
| `breakdown.py` | Its sweep reproduces the evaluator SWEEP exactly (tested at 0.5). The per-molecule `cov_*`/`mat_*` use `np.nanmin`/`np.nanmean`, so only for molecules with "additional failures" the per-stratum tables differ from the evaluator (they ignore the NaN row instead of zeroing COV-P). The ceiling uses `floor > δ`, while COV uses `< δ`; that is equivalent up to measure zero | **MATCHES**. The nanmin difference is negligible and documented |

## 3. Can our QM9 numbers be compared with TD Table 7 (p. 26)?

**Yes, as far as the metric code is concerned.** The formulas, threshold, heavy-atom symmetric RMSD, K rule, failure handling and mean/median aggregation are identical to TD's released evaluator, which inherits GeoMol's. The following can still make the numbers **not comparable**:

1. **RDKit version.**
   - (a) ETKDG changed between versions, and that changes L. It is the dominant QM9 error term: TD p. 26 puts it at ≈ 0.17 Å.
   - (b) `GetBestRMS(symmetrizeConjugatedTerminalGroups=True)` is the default in recent RDKit and changes RMSDs for COOH, carboxylates, nitro groups and similar (0 vs 0.090 Å in our test).
   - TD's `environment.yml` does not pin RDKit, so TD's version is unknown. The cluster pins 2022.9.5 (`slurm/setup_env.sh:100`).
   - Record `rdkit.__version__` in every `eval.pkl`, and check whether the flag exists there: `python -c "from rdkit.Chem import rdMolAlign; print(rdMolAlign.GetBestRMS.__doc__)"`.
2. **Test set and CSV.** Use TD's `data/QM9/test_smiles.csv` and `test_mols.pkl` (1000 molecules, GeoMol split; TD G.1 p. 23). Always pass `--test_csv` explicitly, because the default is DRUGS (CC A.5). MCF's processed set has 995 molecules (GS §6).
3. **Model.** A retrained model is not the released checkpoint. The released QM9 yaml comes from older internal code (CC #25), so P0 (released) vs R0 (retrained) must be reported separately.
4. **The `likelihood` yaml override (CC #1).** Without the patch, retrained models drop molecules with 0 rotatable bonds, which counts them as failures with COV 0. The patch fixes this. Any unpatched run is not comparable.
5. **Sampling stochasticity.** TD reports a single run. Report the mean ± sd over ≥ 3 `--seed`s; the difference in the third decimal of AMR is noise.
6. **Inference settings.** 20 steps (TD Table 9, p. 27; TA l. 29), no `--pre_mmff`/`--post_mmff`, and the default σ schedule.
7. **Failure counts are not reported by TD.** Our `n_model_failures` and `n_additional_failures` must be ~0 for a like-for-like comparison. AMR is computed over successful molecules only.
8. **Floors are not TD's floor.** TD's 0.17 Å QM9 lower bound (p. 26) and 0.324 Å DRUGS bound (App. F.1, Table 5, p. 19-20) come from conformer matching: an H-inclusive objective, DE with maxiter 15, one-to-one with L seeds. Our `floor_*` columns use a heavy-atom, symmetry-aware objective with maxiter 50, and `floor_best_sym` uses K = 2L seeds. These should be *lower*, so do not quote them as "TD's floor".
9. **Stereo.** Test seeds come from the CSV SMILES. Stereo lost there cannot be recovered, because GetBestRMS allows proper rotations only. This is the same in TD, but it matters if `--seed_mols` or GT-L arms are compared with TD rows.

## 4. Genuinely ambiguous decisions: 3-reviewer polls

Each poll has three reviewers argue independently from the sources.
- **R1 is the "reproducibility" reviewer**: stay identical to TD's code.
- **R2 is the "literature definition" reviewer**: follow the paper text.
- **R3 is the "benchmark critic"**: follows Zhou et al. 2023 and GEOM-revisited.

**P1. Model failures: COV = 0 but excluded from AMR.**
- R1: this is exactly TD's evaluator (GS §6; CC A.5), and changing it breaks comparison with Table 7.
- R2: Eq. 35 (TD p. 24) is per molecule and undefined without conformers. Counting failures as COV 0 is the natural extension, and AMR simply does not exist for them. Keep it, but report the count.
- R3: AMR then rewards failing on hard molecules (Zhou 2023 warns that the metrics are gameable, GS §6 "Known pitfalls"). Add a secondary AMR on the intersection of molecules that succeed in every arm.
- **Vote: 3-0 keep the TD convention for headline numbers. 2-1 (R2 and R3) also report intersection-AMR whenever any arm has failures > 0.**

**P2. Heavy-atom vs all-atom RMSD.**
- R1: heavy atoms (TD and GeoMol code).
- R2: GeoDiff Eq. 10 just says "RMSD … after alignment by Kabsch". Zhou l. 479 is explicitly heavy-atom. Hydrogen positions come from ETKDG or xTB and would dominate the noise.
- R3: macrocycle benchmarks (RINGER/CREMP, GS §6 table) use all-atom RMSD, but that is not the QM9 protocol.
- **Vote: 3-0 heavy atoms.**

**P3. `symmetrizeConjugatedTerminalGroups` (RDKit ≥ ~2022.09 default True).**
- R1: do not touch the evaluator. Pin RDKit and record the version.
- R2: the papers say "symmetry-aware" (TD G.3) without defining equivalence classes. Treating C(=O)O as symmetric is chemically defensible for carboxylates but not for neutral COOH.
- R3: pass the flag explicitly so results do not change with RDKit upgrades.
- **Vote: 2-1 keep the code and pin plus record the RDKit version (R1, R2). R3 dissents.** Open question Q1: verify the flag's default on 2022.9.5 on the cluster.

**P4. Strict `<` vs `≤`.**
- R1 and R2: TD Eq. 35 and GeoMol Eq. 5 both use `<`.
- R3: GeoDiff and EBD use `≤`, and the difference is measure-zero for float RMSDs.
- **Vote: 3-0 keep `<`.**

**P5. K = 2 × CSV count vs 2 × L after `clean_confs`.**
- R1: keep; this is how TD's numbers were produced.
- R2: the paper says K = 2L, so K/L > 2 inflates COV-R. Report the fraction of molecules with `n_model != 2*n_true`.
- R3: agrees with R2, and would rerun the affected molecules with exactly 2L as a sensitivity check if the fraction is > 1%.
- **Vote: 3-0 keep for headline numbers. 2-1 add the diagnostic (R2, R3).**

**P6. Headline threshold on QM9.**
- R1: δ = 0.5 (Table 7).
- R2: same, but at 0.5 the median is saturated at 100 for every method (Table 7; FM-refiner Table 2 uses 0.05 Å, GS §6).
- R3: headline AMR plus a small-δ sweep.
- **Vote: 3-0 report δ = 0.5 for comparability, plus the SWEEP at 0.05, 0.1 and 0.25 and AMR as the discriminating metrics** (as in `notes/ablation_plan_qm9.md` §1).

**P7. Atom map for the local-geometry errors (`geometry_metrics.py`).**
- R1: use the correspondence GetBestRMS used, because the pair was chosen by that RMSD.
- R2: bond and angle are internal coordinates, so a map minimising angle error isolates L error.
- R3: min-over-maps of the reported statistic is selection-biased low, and torsion errors then come from a different permutation than the RMSD.
- **Vote: 2-1 RMSD-best map (R1, R3). Implemented.**

**P8. One "additional failure" zeroes the molecule's COV-P.**
- R1: keep; upstream behaviour.
- R2: the definition would drop only the bad GT conformer from the min over GT.
- R3: keep, but make it visible.
- **Vote: 2-1 keep upstream behaviour.** `n_additional_failures` is already printed. If it is > 0 in any arm, inspect those molecules before comparing.

**P9. Which floor defines the "COV-R ceiling".** This was also raised by the parallel cross-validation work in `breakdown.py`.
- R1: `floor_rmsd` is the one-to-one analogue of TD's conformer-matching bound.
- R2: AMR-R is a min over K = 2L generated conformers, so the comparable quantity is `floor_best_sym`.
- R3: neither is a hard bound, because seeds are independent draws and DE is heuristic. Label it "estimated".
- **Vote: 2-1 use `floor_best_sym` and call it an estimate (R2, R3).** `breakdown.py` already does this.

## 5. Bugs found and fixed

| ID | Severity | Where | Fix | Evidence |
|---|---|---|---|---|
| B1 | low | `evaluate_confs.py` SUMMARY (patch commit `d0279bb`) snapped `--report_threshold` to the nearest point of the 0.125 grid, so 0.1 was reported as 0.125 | Branch commit **`6218977`** "Evaluate: SUMMARY at the exact --report_threshold". `patches/0001-ablation-hooks.patch` was regenerated with `git format-patch master..flexitors-ablation-hooks` (now 6 patches) and verified to apply cleanly to `master` with a tree identical to branch HEAD | test "--report_threshold 0.1 honoured exactly". SUMMARY at 0.5 is unchanged |
| B2 | **high for the FlexiTors argument** | `tools/local_structure_analysis.py`: the transplant, floor and run-floor objectives used the identity atom map, so they were not symmetry-aware | `heavy_automorphisms()`, and `heavy_rmsd` accepts a list of maps and takes the min. `job()` uses all maps; `run_job()` optimises over all maps instead of only the best-at-start one | CC(C)CC#N floor 0.459 → 0.073 Å. Equal to GetBestRMS to 9e-11 Å |
| B3 | medium | `tools/geometry_metrics.py`: torsion MAE included ill-conditioned sp-centre dihedrals, and the atom map was chosen by angle MAE | skip bonds with an sp endpoint; choose the map by aligned heavy-atom RMSD | 0.02 Å N wiggle: torsion MAE 28.8° → 0.0° |

The tools live outside the git repo, so B2 and B3 are file edits marked `[metric-verifier fix]`. Pre-fix copies are in the session scratchpad only.

**Concurrency note:** another agent rewrote `local_structure_analysis.py` (adding `--mode run`, `floor_best_sym` and oracles) while this review was running. My first B2 edit was overwritten, so the fix was re-applied surgically on top of their version. Only `heavy_rmsd`, the new `heavy_automorphisms`, `amap = ...` in `job()` and the map loop in `run_job()` changed. If that file is rewritten again, re-check that `heavy_automorphisms` is still used; `test_metrics.py` will fail otherwise.

**Cost note (B2):** the DE objective now costs `#automorphisms` × AlignMol per evaluation, and `--mode test` runs about 9 DE optimisations per GT conformer. QM9 molecules have few automorphisms (≤ ~12 typical), but budget roughly 2-5× CPU time for A0.

## 6. Open questions

- **Q1.** RDKit 2022.9.5 on the cluster: does `GetBestRMS` have `symmetrizeConjugatedTerminalGroups`, and what is its default? Which RDKit produced TD Table 7? Record `rdkit.__version__` in `eval.pkl`, which the patch does not do yet.
- **Q2.** How often does K ≠ 2L on the QM9 test set? Compute `np.mean([r['n_model'] != 2*r['n_true'] for r in R['results'].values()])` on the first R0 `eval.pkl`.
- **Q3.** Do all GEOM conformers of a molecule share the atom order of `gt[0]`? `geometry_metrics.py` and the whole upstream standardisation assume it. A cheap check on `test_mols.pkl`: `all(Chem.MolToSmiles(c, canonical=False) == Chem.MolToSmiles(gt[0], canonical=False))` per molecule.
- **Q4.** Rotation across linear chains (R-C≡C-R') is neither in the torsion MAE nor well defined in `transplant_rmsd_*`. Measuring it needs reference atoms beyond the sp chain. The prevalence of such chains in the QM9 test set was not measured, since no data is available locally.
- **Q5.** Intersection-AMR (P1) and the K ≠ 2L diagnostic (P5) are not implemented in the evaluator. They would be a small `breakdown.py` addition once several arms exist.
