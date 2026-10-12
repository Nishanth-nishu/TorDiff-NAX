# Leak-free true-geometry oracle test (split-half), round 4

Status: RUNNING (method and counts written 2026-10-12 before any model result; results filled in after job 17518 and
the analysis). ORACLE: every number here uses test-set true geometry. Analysis only, never model selection (README
rule 5).

## 1. Question
Round 2 reported that S3 keeps most of the true-geometry gain: on `steps20_seed0_gtLcycle` S3 scores 0.0334 Å AMR-R
against 0.0822 for CTRL_rematch (−0.049 Å, `cluster_sync/round2/results/analysis_r2/contrasts.csv`,
`S3.gtLc_vs_CR_ORACLE`). In that condition every target conformer's own local structure (L) is among the seeds, so a
model can rebuild the target from its own L (conformer-identity leak; `round3/impact.md` §3g, validity judge V-2/V-3).
On every non-oracle condition S3 is not better than CTRL (RDKit L: +0.003 Å). This test removes the leak and measures it.

## 2. Method
- **Molecules.** QM9 test set (`test_smiles.csv`, 1000 rows). GT conformers are filtered exactly as
  `evaluate_confs.py` `clean_confs` does. Molecules with ≥ 2 clean GT conformers are used; single-conformer molecules
  have no "other" conformer and are excluded and counted.
- **Split** (`tools/make_split_half.py`, deterministic): alternating by cleaned GT index, half A = indices 0, 2, 4, …,
  half B = 1, 3, 5, … (|A| = ⌈n/2⌉, |B| = ⌊n/2⌋). No RNG; balances the halves over GEOM's stored order.
- **Four generations per checkpoint** (seed 0, 20 steps, `--seed_confs_cycle`, batch 128, PACK=3 on one RTX 3090):

  | condition | seeds (true L of) | generated | recall against | own L among seeds? |
  |---|---|---|---|---|
  | split_AtoB | half A | 2·\|B\| | half B | no |
  | split_BtoA | half B | 2·\|A\| | half A | no |
  | own_BtoB | half B | 2·\|B\| | half B | yes |
  | own_AtoA | half A | 2·\|A\| | half A | yes |

  **split-half** = mean of the two split directions; **own** = mean of the two own directions; **leak** = own − split
  (per molecule, then averaged). For each target half, split and own runs have the same targets and the same number of
  generated conformers; only the seed half differs, so the leak is a paired, within-molecule measurement (no division
  by P0, unlike the §3g estimate).
- **No evaluator change.** `evaluate_confs.py` runs unchanged; the GT subset is passed as a subset `true_mols` pickle
  (`true_A.pkl` / `true_B.pkl`) and a restricted csv. Golden tests: the evaluator, generator and sampler are
  byte-identical to round 3 (LF sha256 pins), and on a fixture the subset evaluation equals the corresponding rows of
  the full-GT RMSD matrix exactly (`tools/tests/test_leakfree.py`).
- **Seed provenance.** `tools/check_split_seeds.py` asserts that generated conformer i keeps the bond lengths of seed
  i mod |half| of the intended half (max deviation ≤ 1e-5 Å in the smoke runs), i.e. the target half's own L never
  enters a split run (`tools/tests/test_leakfree_cluster.py`, 6 + 6 passed on CTRL_rematch s0 and S4 s0, 20 molecules).
- **Models** (`slurm/r4_leakfree_models.tsv`): CTRL_rematch s0–s2 (CR), B1 s0–s2, S3 s0–s1, S4 s0–s2 (`--l_level 1`).
- **Analysis** (`tools/analyze_leakfree.py`, committed before any result): AMR-R and COV-R@0.5 (primary), COV-R@0.1
  (key secondary). Universe = molecules with a finite AMR-R in all 44 files. Model value = per-molecule mean over
  training seeds, then mean over molecules. Pre-declared primary contrasts S3 − CR and B1 − CR on split-half (one Holm
  family, 2 contrasts × 2 metrics); CIs from a paired molecule bootstrap (5000) and a hierarchical bootstrap (training
  seeds, then molecules). Secondary: S4 − CR, own-L contrasts, within-model leak, leak differences, and a
  distinct-target sensitivity (below). Reference: the round-2 full-set RDKit-L and gtLcycle runs of the same checkpoints
  re-scored on the same molecules.

## 3. Molecule counts (`$QM9_DIR/round4_splithalf/population.txt`)
| | molecules | GT conformers |
|---|---|---|
| test csv rows | 1000 | |
| no clean GT conformer | 3 | |
| single clean conformer (excluded) | 253 | |
| **eligible (n_clean ≥ 2)** | **744** | 12,978 (A 6,651 / B 6,327) |
| by n_clean: 2–3 / 4–10 / 11–30 / > 30 | 243 / 281 / 135 / 85 | |

**Near-duplicates across halves (found by the builder, before any model result).** For every eligible GT conformer,
the heavy-atom RMSD (GetBestRMS, as the evaluator) to the nearest GT conformer of the other half: median 0.178 Å;
23.7 % < 0.05 Å, 39.5 % < 0.1 Å, 55.3 % < 0.25 Å, 79.0 % < 0.5 Å. So for about 40 % of targets the other half holds a
conformer that is practically the same heavy-atom structure (INFERENCE, not checked: GEOM QM9 keeps conformers that
differ mainly in H positions, e.g. O–H / N–H rotamers). For those targets the split-half seeds still carry an own-like L and torsions. Hence two additions,
declared before any result:
- **COPY baseline** (model-free): the other half's GT conformers used directly as predictions. Its AMR-R is the mean
  nearest-other-half RMSD (0.334 Å on all eligible molecules). It is the seed contribution a model must beat
  (validity judge V-8).
- **Distinct-target view** (`split_d` / `own_d` / `leak_d`): the same metrics on targets whose nearest other-half
  conformer is ≥ 0.1 Å away only.

## 4. Results
(to be filled)

## 5. Interpretation
(to be filled)

## 6. Caveats
(to be filled)
