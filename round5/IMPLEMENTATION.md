# Round 5 Stage A: implementation log

Implementer, 2026-10-17. Base plan `round5/plans/plan_P2.md` with the grafts and fixes of `round5/DECISION.md`;
binding answers: EXP-1 §10, EXP-3 §9, EXP-5 §12, EVAL §1.1 / §3.1 and every `## Answer` in `round5/consult/`.
Commits are local (not pushed); the repo author is the only author.

## Milestone 1 (CPU only, no GPU)

### 1. Commits

| Commit | Content |
|---|---|
| `c594757` | `flexitors/` skeleton, `td.py` gateway with SHA pins, exact dihedrals, TD torsion / stereo-bond rules, validity layer |
| `dfd4f0e` | Floor-bug fix at `tools/local_structure_analysis.py:302` and the split.tsv other-half floor, with regression test |
| `300ace2` | `tools/analyze_leakfree.py --ref_subsample`: budget-matched ref_half, BUDGET Holm family |
| `5baf2a8`, `f59d4b1`, `9f4c695`, `948e5ed` | `round5/sync_cluster.sh`, `experiments/r5_cpu/` (one sbatch, six tasks) |
| `7277926` | QM9 validation-set builder, provenance records, n_rot_heavy rule |
| `38e0640` | Gate-0 validation seed sets and baselines |
| `d00100a` | E/Z target-table builder |
| `240ab9c` | Results: validation set files, E/Z target table, re-score summaries |

### 2. Files and functions, with the spec section or E-ID each implements

**Package layout (DECISION graft 1; round5/PLAN.md "Code structure").** `flexitors/{geometry,models,sampling,training,eval,data}/`
plus `tests/`. `models/`, `sampling/` and `training/` hold only a docstring in milestone 1: no module is created before
a Stage A job runs it (no dead code).

| File:line | Function | Implements |
|---|---|---|
| `flexitors/td.py:30` | `BASELINE_PINS` | DECISION graft 1 (plan_P1 §1.2): LF-normalised SHA-256 of all 20 tracked TD python files; evaluator / generator / sampler pins equal the round-3 pins of `tools/tests/test_leakfree.py` |
| `flexitors/td.py:59` | `changed_baseline_files` | baseline integrity, written into every provenance record |
| `flexitors/td.py:69` | `import_baseline` | the only `sys.path` / cwd manipulation; `diffusion.*` imported with cwd = TD root so `torus.py:26-34` reads its tables instead of recomputing them in the caller's cwd |
| `flexitors/provenance.py:25` | `provenance`, `write_provenance` | plan_P2 V-X6 (commit, versions, host, argv, baseline integrity) |
| `flexitors/geometry/internal.py:12` | `dihedral` | DECISION fix 7, EXP-1 §10.1 item 2: exact atan2 (TD's `utils/torsion.py:120` clamps cos to ±(1 − 1e-5)) |
| `flexitors/geometry/internal.py:36` | `signed_volume` | EXP-1 §2.8 χ_c |
| `flexitors/geometry/torsion_types.py:44` | `td_rotated_side`, `td_rotatable_bonds` | port of `utils/torsion.py:9-39` (E-CHEM-020), incl. the tie rule of `sorted(..., key=len)[0]` |
| `flexitors/geometry/torsion_types.py:67` | `heavy_rotatable_bonds` | n_rot_heavy of `tools/breakdown.py:32-50`; rigid stratum = n_rot_heavy = 0 (DECISION fix 4, E-JOINT-007) |
| `flexitors/geometry/torsion_types.py:86` | `td_reference_quadruple` | `utils/torsion.py:19-24` (masked edge orientation) + `diffusion/sampling.py:135-142` (first neighbours) |
| `flexitors/geometry/torsion_types.py:103` | `is_structural_stereo` | EXP-3 §3.1 structural test; imine N–H = non-stereo (Q-P2-007 answer) |
| `flexitors/geometry/torsion_types.py:111` | `smiles_phi_target` | EXP-3 §3.1 target rule (stereo atoms, toggles for c ≠ s1, d ≠ s2) |
| `flexitors/eval/validity.py:77` | `perceived_bonds` | EXP-1 §3 row 5: connectivity from 3D (`rdDetermineBonds`) |
| `flexitors/eval/validity.py:92` | `clash_count` | EXP-1 §2.6 step 1: 0.75 × sum of covalent radii, non-bonded heavy pairs |
| `flexitors/eval/validity.py:111` | `bond_window_violations` | EXP-1 §2.6 step 1: ±15 % of the reference length; ring bonds counted separately (ring closure, EXP-1 §2.7) |
| `flexitors/eval/validity.py:122` | `stereocentres`, `chirality_flips` | EXP-1 §2.8: RDKit-perceived centres, neighbours in canonical rank order, signed-volume sign vs reference |
| `flexitors/eval/validity.py:146` | `ez_state` | EXP-3 §9.3 primary label (match iff cos(τ_b − φ_b) > 0 on TD's quadruple), planarity |δ_b| ≤ 20° (EXP-3 §3.3) |
| `flexitors/eval/validity.py:159` | `validate_conformer`, `ValidityReport.valid` | EVAL §3.1 invalidity definition (counted, never dropped) |
| `flexitors/data/qm9_val.py:75` | `self_check` | EVAL §1.1 item 4 acceptance test (Q-P2-010): candidate rules scored on the 1000 test rows |
| `flexitors/data/qm9_val.py:145` | `build` | EVAL §1.1 items 1-5: split index 1 over sorted raw pickles (`utils/dataset.py:165-179`), `default_rng(0).permutation`, first 1000 passing, raw uncapped GT |
| `flexitors/data/qm9_val.py:127` | `population_lines` | EVAL §1.1 item 5 (K, n_rot, n_heavy, ring classes, scaffold-novel share vs test) |
| `flexitors/data/gate0_baselines.py:52` | `ring_seeds` | E-JOINT-004 rev.: paired seeds of largest-ring ≥ 4 molecules, share > 10° |
| `flexitors/data/gate0_baselines.py:59` | `rigid_amr` | E-JOINT-007 rigid stratum read-out |
| `flexitors/data/gate0_baselines.py:66` | `thresholds` | EXP-1 §10.3 thresholds and the > 10 % relative scaling rule |
| `flexitors/data/ez_targets.py:41` | `table_rows`, `embed_check` | EXP-3 §9.3 / Q-P2-007: committed target table (RDKit 2026.03.6), each SMILES target cross-checked by ETKDG |
| `tools/local_structure_analysis.py:243` | `floor_candidates` | EXP-1 §9 item 1 / E-JOINT-031 / D-003 fix: the target conformer is removed before slicing (`:316`) |
| `tools/local_structure_analysis.py:89`, `:318-331` | `--floor_other_half` | EXP-1 §10.5 / Q-P1-011: top-3 among the other half of `split.tsv`, cap after splitting, all-other-half minimum for halves ≤ 10 |
| `tools/analyze_leakfree.py:435`, `:447` | `subset_min_expectation`, `subset_cover_prob` | EXP-1 §10.5 exact expectations E[min] = Σ r₍ᵢ₎ C(N−i, s−1)/C(N, s), P(cov) = 1 − C(N−c, s)/C(N, s) |
| `tools/analyze_leakfree.py:523` | `budget_matched` | EXP-1 §3.6 / §10.5: ref_half, 10-draw MC check, BUDGET Holm family of 8 (CR, B1, S3, S4 × AMR-R, COV-R@0.5), hierarchical intervals, COV-R@0.1 and split_d − ref_half_d secondary |
| `round5/sync_cluster.sh`, `experiments/r5_cpu/run.sbatch` | | round2/sync_cluster.sh pattern; tasks tests / rescore / floor / floor_mmff / valset / gate0, ≤ 5000 MB per CPU |

### 3. Tests

Local (Windows, Python 3.11, RDKit 2026.03.6, no torch_geometric):
```
python -m pytest -q flexitors/tests          -> 56 passed, 2 skipped in 17.15s   (skipped: the 2 cluster-only tests)
python -m pytest -q tools/tests/test_lsa_floor.py tools/tests/test_leakfree.py tools/tests/test_leakfree_budget.py
                                             -> 14 passed in 11.37s
```
gnode118 (TD venv, Python 3.9.23, RDKit 2022.09.5; `cluster_sync/round5/tests/`):
```
flexitors/tests                              -> 55 passed, 3 skipped in 10.18s
  -m cluster: test_td_gateway.py .  test_torsion_types.py .   -> 2 passed
  (torus import keeps the cwd and does not recompute the tables; td_rotatable_bonds + td_reference_quadruple equal
   TD's get_transformation_mask and the sampling.py:129-142 construction on 100 QM9 molecules)
  skipped: E-CHEM-066 counts and table reproducibility (pinned to RDKit 2026.03.6), l_error.csv golden (file not on the cluster copy)
tools tests (floor, leak-free, budget)       -> 14 passed in 5.57s
```
What the tests pin: baseline SHA pins (local and on the cluster copy: `baseline_changed_files: []` in every
provenance record); dead-code audit (AST: every definition referenced, imports used, every CLI flag read, no TODO;
`validate_conformer` is the one test-only API, declared in `TEST_ONLY_API` with its milestone-2 consumer); atan2 vs
RDKit to 1e-9 and resolution at 1e-6 rad; E-CHEM-066 counts (QM9 37 / 3, DRUGS 140 / 126); φ targets against ETKDG
embeddings incl. a reference-atom swap; validity on relaxed conformers, enantiomer, bond stretch, ring opening, clash,
E→Z flip; the floor regression (old code gives 0.0 Å on a two-conformer rigid bicycle, fixed code 0.0873 Å, and
n_gt > 3 rows are unchanged by construction); subset expectations equal full enumeration; ring baseline on the test
`l_error.csv` reproduces 4,181 seeds, 49.6 % and 35.1 %.

### 4. Validation-set self-check (EVAL §1.1; `round5/data/qm9_val/`)

| Candidate `corrected_smiles` rule | Rows reproduced (smiles, n_conformers, test_mols count, corrected_smiles) |
|---|---|
| GEOM `mol_dic['smiles']` | 90.2 % |
| canonical isomeric SMILES of conformer 0 | 80.2 % |
| GEOM smiles if conformer 0 has its graph, else conformer 0 | 99.0 % |
| **GEOM smiles if any conformer has its graph, else conformer 0 (chosen)** | **99.1 % → PASS** |

`n_conformers` = stored conformer count (1000/1000; the clean count matches 99.6 %); `test_mols.pkl` = all raw
conformers, positions identical for 1000/1000. The 9 residual mismatches (radical GEOM keys whose corrected SMILES
differs from both candidates only in a stereo label or kept the radical form) are listed in `self_check.txt`.
The validation subset: 1000 molecules, 1000 walked positions, 0 excluded; `val_mols.pkl` sha256
`e96c6274…f5ef` (pickle on /scratch, not committed). Population vs test: K mean 13.90 vs 13.76 (max 738 vs 727),
rigid (n_rot_heavy = 0) 328 vs 318, ring ≥ 4 polycyclic 487 vs 485, scaffold-novel 8.9 % vs 8.5 %.

### 5. Budget-matched leak-free re-score (EXP-1 §3.6 / §10.5; ORACLE, analysis only)

`cluster_sync/round5/rescore/`. Universe: 743 split-half molecules; 32 are absent from some full-set RDKit-L run
(ETKDG embedding failures; ref_half undefined) and are excluded from every view, leaving 711 (606 distinct-target).
The round-4 part of the run reproduces `analysis_r4_leakfree/leakfree_summary.txt` byte for byte.

| Model | split AMR-R | ref_half AMR-R | split − ref_half AMR-R [95 % CI] | Holm p | COV-R@0.5 diff (pt) [95 % CI] | Holm p | old gap split − ref_rdkitL |
|---|---|---|---|---|---|---|---|
| CR | 0.2214 | 0.2598 | **−0.0384** [−0.0482, −0.0289] | < 0.001 | +0.07 [−1.31, +1.48] | 0.91 | +0.034 |
| B1 | 0.3059 | 0.3228 | −0.0170 [−0.0310, −0.0030] | 0.060 | −4.41 [−6.67, −2.12] | < 0.001 | +0.049 |
| S3 | 0.2803 | 0.2632 | **+0.0171** [+0.0048, +0.0296] | 0.021 | −8.70 [−10.62, −6.86] | < 0.001 | +0.089 |
| S4 | 0.2548 | 0.2617 | −0.0069 [−0.0178, +0.0041] | 0.47 | −5.34 [−6.98, −3.74] | < 0.001 | +0.066 |

Hierarchical intervals agree (e.g. CR −0.0482 to −0.0284). Secondary, distinct targets only (n = 606):
split_d − ref_half_d AMR-R CR −0.0147 [−0.0269, −0.0026], B1 +0.0666, S3 +0.0851, S4 +0.0469 (all CIs exclude 0).
COV-R@0.1 split − ref_half is +19.7 to +25.2 pt for every model (near-duplicate targets in the other half,
E-JOINT-002). MC check (10 draws per half): dataset means within 0.0005 Å of the exact expectation (SD over draws
0.0009–0.0027 Å); 96.8 % of per-molecule values within 3 MC SE (10-draw SE estimates).

### 6. Contradictions and open points (raised, not silently resolved)

1. **The re-score changes the Phase-2 conclusion in `round5/PLAN.md`.** "With another conformer's true geometry every
   model is worse than with RDKit geometry" was a budget artefact: the full-budget reference used 2·n_csv conformers
   against 2|H| in the split runs. At matched budget, other-conformer GT geometry is *better* than RDKit geometry on
   AMR-R for CR (−0.038 Å, Holm p < 0.001) and B1 (−0.017, Holm p = 0.06), equal for S4, and worse only for S3; on
   COV-R@0.5 it is equal for CR and worse for B1, S3, S4. On distinct targets only CR still gains (−0.015 Å). EXP-1
   §3.6's pre-declared predictions and the "geometry helps only when coupled" argument should be re-read by R-JOINT
   with these numbers; per EXP-1 §10.5 the result is descriptive and selects nothing.
2. **EVAL §1.1 filters vs the test file.** Under the cluster RDKit the test csv itself contains 2 rows whose
   `corrected_smiles` do not parse, 3 rows without a clean GT conformer, and a multi-fragment row (`N#N.[NH]…`). I
   applied the EVAL filters as written (they excluded 0 validation molecules, so the set is unaffected) and did not
   add a fragment filter.
3. **`corrected_smiles` rule.** Neither single candidate named in EVAL §1.1 reaches 99 %; the chosen rule combines the
   two (GEOM smiles unless no conformer has its graph). Owner R-SAMPLE to confirm.
4. **ref_half universe.** 32 molecules without an RDKit-L full-set result are excluded (listed in
   `budget_excluded.csv`); the round-4 `ref_rdkitL` rows had averaged over whatever was present.
5. **Scaffold-novel count.** 85 of 998 parseable test rows vs 87 of 1000 in `sample_data/ood.out` (different RDKit
   and row filter); G11 should pin one of them.
6. The structural test (EXP-3 §3.1) counts double bonds with identical substituents on one end (e.g. `CC=C(C)C`) as
   structural; that is the spec's definition and affects planarity bookkeeping only (such bonds carry no SMILES target).

### 7. Not yet run (queued on gnode118, CPU)

`gate0` (validation ETKDG / MMFF seeds, rigid AMR-R and ≥ 4-ring share, thresholds; jobs `r5_gate0`) and the floor
re-run (`r5_floor`, `r5_floor_mmff`, new files `local_structure_test_fixed.csv`; the round-1 CSVs stay untouched). The
plafnet2 backfill window is shorter than an hour at the moment, so these 4-CPU jobs wait for free cores (expected start
≈ 12:00 on 2026-10-17). Their results will be added here before the Gate-0 ask.


### Conformance (JOINT): budget-matched leak-free re-score (milestone 1 §5)
Checked 2026-10-17 against EXP-1 §3.6 and §10.5 (as pre-declared on 2026-10-16), by reading
`tools/analyze_leakfree.py:426-505` and `:523` and the outputs in `cluster_sync/round5/rescore/`.

| Pre-declared mechanism (EXP-1) | Code / output | Ruling |
|---|---|---|
| ref_half = 2·|H|-subset of the full RDKit-L run scored against half H, direction-averaged (§3.6, D-001) | `ref_half_molecule` :456-476, s = 2·len(idx[h]); averaged over A and B as `view()` | MATCHES |
| Exact expectation as primary: E[min] = Σ r₍ᵢ₎ C(N−i, s−1)/C(N, s); P(cov) = 1 − C(N−c, s)/C(N, s) (§10.5) | `subset_min_expectation` :435-444, `subset_cover_prob` :447-453 (strict <, as the evaluator) | MATCHES |
| 10 random draws as a consistency check (§10.5) | MC rows; dataset means within ≤ 0.0005 Å of exact | MATCHES |
| One Holm family of 8 (CR, B1, S3, S4 × AMR-R, COV-R@0.5); paired molecule bootstrap; hierarchical interval beside (§10.5) | `budget_matched` :523; `budget_contrasts.csv` family = budget, p_holm, hier_lo/hi | MATCHES |
| Secondary: COV-R@0.1 and split_d − ref_half_d (§10.5) | rows tagged secondary | MATCHES |
| Same universe for every view | 32 molecules excluded from every view and model because ref_half is undefined (missing full-set RDKit-L result, or 2|H| larger than the generated count); 711 analysed | JUSTIFIED DEVIATION (743 → 711, listed in `budget_excluded.csv`; the exclusion is paired, so contrasts are unbiased with respect to it) |
| Run before any Exp-1 GPU time (§3.6) | run in milestone 1, CPU only | MATCHES |

No MISMATCH. The interpretation changed after the result and is recorded in EXP-1 §11 / Revision 3:
- the distinct-target view is promoted to primary *for the coupling reading only*;
- the BUDGET family itself is unchanged.

That promotion is a post-hoc reading choice and is labelled as such. — R-JOINT

## Milestone 1 fixes (after review_P1.md and review_P2.md, 2026-10-17)

Every BLOCKER and SHOULD-FIX is addressed; NITs as listed. Commits `a5b5d32..db53b1b` (repo author only, not pushed).

| Finding | Fix | Commit | Test / evidence |
|---|---|---|---|
| P1 B1 / P2 S1: gateway chdir only for `diffusion*`; `utils.utils`, `utils.training`, `train` import torus indirectly | every baseline import (TD or tools) runs with cwd = TD root; TD imports first assert the tables exist; torus table SHA-256 pinned (`TORUS_TABLE_PINS`) | `a5b5d32` | `test_td_gateway.py`: local probe module records cwd = TD root; cluster: `diffusion.torus`, `utils.training`, `utils.utils`, `diffusion.sampling` from a temp cwd write no `.npy` there and leave the TD tables' mtimes unchanged; table SHA pins — all pass on gnode118 |
| P2 B1: bicyclo[1.1.1]pentanes get an extra perceived bond | connectivity = no graph bond missing, and no extra perceived bond absent from the **reference's own** perception; GT calibration CLI | `76785ae`, `7b82ae5` | bicyclopentane tests; **GT calibration** (`cluster_sync/round5/validity_cal/gt_calibration.txt`, RDKit 2022.09.5): every cleaned GT conformer vs the molecule's first GT conformer — QM9 test 13,231 conformers / 997 molecules: 12 invalid (0.09 %, 9 molecules); QM9 val 13,892 / 1000: 12 invalid (0.09 %, 9 molecules). Missing bonds, unexplained extra bonds, clashes, sanitisation: 0 everywhere. All 24 residuals are signed-volume flips, almost all at ring N (degree-3 bridgehead or fused-ring amine N), i.e. N invertomers inside the GEOM ensembles (plus one GT ensemble with a reacted conformer); reported, not hidden |
| P2 B2: Gate-0 scaling divided by a published value from another recipe | thresholds scaled by validation ÷ **same-recipe** test value; the published 0.152 / 0.128 Å, 49.6 / 35.1 % only enter a `recipe_check` (flagged if > 10 % apart) | `54e3173` | `test_gate0_baselines.py` (ratio exactly 1 for identical recipes; flag behaviour) |
| P1 S1 / P2 S3: bond window inside `valid` | bond window (EXP-1 §2.6) reported as `n_bond_window_all` / `_ring`, not part of EVAL §3.1 `valid`; ring closure diagnostic only | `76785ae` | 16 % stretch still perceived → `valid` True with window flagged |
| P2 S2: `valid` passed without a reference | `valid` is None (undetermined) without a reference | `76785ae` | `test_no_reference_is_undetermined` |
| P2 S4 / P1 S6: RDKit-version-dependent chirality, no bridgeheads | structural centres (every degree-4 atom + every degree-3 atom in ≥ 2 rings) with a |χ_ref| ≥ 0.05 Å³ floor; for an independent reference only centres with pairwise topologically distinct neighbours (`ref_is_seed=False`) | `76785ae` | enantiomer, quinuclidine-type bridgehead inversion, seed-vs-relaxed and cross-conformer tests |
| P1 S6: checks not reusable by the M2 constraint layer | array-level batched primitives `clash_index`/`clash_mask`, `bond_window_mask`, `chirality_centres`/`chirality_flip_mask` + thin Mol wrappers | `76785ae` | `test_clash_detected_and_primitives_are_batched` |
| P1 S2 / P2 S5: E/Z table indices unchecked under RDKit 2022.09.5 | cluster test: for every table row, same structural bond and TD quadruple under 2022.09.5 + TD featurisation; ETKDG under 2022.09.5 honours every specified target | `53290e9` | passes on gnode118 (in the 67 passed) |
| P1 S3: two n_rot_heavy implementations | `flexitors/data/strata.py` (m, n_rot_heavy) from the single `torsion_types` implementation | `72980b4` | cluster test: equals `n_rot_heavy` and `n_rot_repo` of `tools/breakdown.py`'s output for CTRL_rematch s0 — passes |
| P2 S6: split.tsv alignment not asserted in the floor re-run | per molecule, pre-cap clean count must equal `split.tsv` n_clean, else a `split_misaligned` error row; cap-after-splitting reading written in the help text | `b6e4aa8` | `test_split_misalignment_is_reported` |
| P1 S4 / P2 S7: `cmp` could not fail the job | `cmp … || exit 3`, result written to `cmp.txt` | `02e39f6` | rerun: "round-4 summary reproduced byte for byte" (`cluster_sync/round5/rescore/cmp.txt`) |
| P1 S5: sync ran `rm -rf $P/round5` | only `round5/ledger/chem_data` and `round5/data` are replaced | `02e39f6` | — |
| P1 S7 / P2 N1: dead-code audit counted docstring mentions | AST references only (Name, imported name, `x.name`), recursion excluded, scope (no methods) stated; `TEST_ONLY_API` now empty (`validate_conformer` has a production caller, the calibration CLI) | `03a5d31` | `test_comment_mentions_do_not_count` |
| P2 S8a: m = 0 molecules carry no model | post-hoc m = 0 / m > 0 split (`--strata`, family "post hoc") | `d68ba5d`, `db53b1b` | see table below |
| P2 S8b: distinct-target view promoted after the result | `_d` rows carry the note "pre-declared secondary; as primary for coupling (EXP-1 §11): post hoc"; summary header states it | `d68ba5d` | `test_leakfree_budget.py` |
| P2 S8c / N8: informative exclusion, descriptive MC check | both stated in the summary header | `d68ba5d` | — |
| NITs | P1 N2 (escaped tab), N3/P2 N4 (cap reading), N4 (toggled φ asserted), N5 (double bond selected explicitly), N6 (seed vs relaxed reference), N7 (`n_bond_window_all`), N9 (provenance dirty dirs), N11 (`_graph`); P2 N2 (sanitisation docstring), N3 (old slice contains the target) | `b6e4aa8`, `76785ae`, `02e39f6`, `b2e9ed4` | — |

Not changed: P1 N8 (the `.txt` summary already lists `embed_failed` among specified rows; there are none), N10 (the RDKit
version is in every provenance record and the calibration header), P2 N5 (the floor re-run keeps split-half molecules
only; G9 is to be checked on the intersection with the old CSV once it runs), N6 (the heavy-first W quadruple is a
milestone-2 function), N7 (the `corrected_smiles` rule still needs R-SAMPLE's sign-off; the acceptance rows were also
used to choose among the four candidates).

**Re-score, post-hoc m split** (`budget_contrasts.csv`, family "post hoc"; BUDGET rows unchanged by the rerun):

| Model | m = 0 (n = 14) AMR-R diff [95 % CI] | m > 0 (n = 697) AMR-R diff [95 % CI] | m > 0 COV-R@0.5 (pt) |
|---|---|---|---|
| CR | +0.107 [+0.016, +0.195] | **−0.041** [−0.051, −0.032] | +0.5 [−0.9, +1.8] |
| B1 | +0.107 (identical: no model runs) | −0.019 [−0.034, −0.005] | −4.1 |
| S3 | +0.107 | +0.015 [+0.003, +0.027] | −8.5 |
| S4 | +0.107 | −0.009 [−0.020, +0.002] | −5.1 |

The m = 0 rows are identical across models, as they must be (the split output there is the other half's GT
conformers). They go the other way and are only 14 molecules, so CR's gain comes entirely from molecules where the
torsion model runs.

**Tests after the fixes.** Local: `flexitors/tests` 62 passed, 8 skipped (cluster-only); tools 15 passed.
gnode118 (`cluster_sync/round5/tests/`): `flexitors/tests` 67 passed, 3 skipped (2026-pinned E-CHEM-066 counts and table
reproducibility, local-only `l_error.csv`); tools 15 passed.

Still queued (unchanged): `gate0` (now with the like-for-like scaling) and the floor re-run (with the alignment
assert).
