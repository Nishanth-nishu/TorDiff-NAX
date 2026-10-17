# Code review P2 (validity focus): Stage A milestone 1, commits c594757..ff58972

Reviewer: planner P2, 2026-10-17. Checked against `round5/DECISION.md` (10 grafts/fixes), the validity table of
`round5/plans/plan_P2.md` §4, and the binding answers (EXP-1 §10, EXP-3 §9, EXP-5 §12, EVAL §1.1/§3.1, every
`## Answer` in `round5/consult/`). No code was edited and nothing was committed.

**Local re-run** (Windows, Python 3.11, RDKit 2026.03.6):
- `python -m pytest -q flexitors/tests` → 56 passed, 2 skipped.
- `python -m pytest -q tools/tests/test_lsa_floor.py tools/tests/test_leakfree.py tools/tests/test_leakfree_budget.py`
  → 14 passed.

Both match `round5/IMPLEMENTATION.md` §3.

**What is right.**
- Baseline: the SHA pins cover every tracked TD `.py` file and agree with the round-3 pins.
- Dihedrals: the atan2 implementation matches RDKit to 1e-9.
- Rigid stratum = n_rot_heavy = 0, a faithful port of `tools/breakdown.py:32-50` (DECISION fix 4).
- TD rotatable rule and reference quadruple: checked against TD's own code on the cluster.
- E/Z targets: every specified target agrees with an ETKDG embedding. I also checked that RDKit 2026's
  `GetStereoAtoms()` for STEREOE/Z are the CIP-ranked atoms on hand cases, so the Z/E→cis/trans reading at
  `torsion_types.py:131-134` holds.
- Floor fix: the bug is fixed exactly at `:302`, and rows with n_gt > 3 are provably unchanged.
- ref_half:
  - the exact subset expectation is correct (verified by enumeration with ties);
  - the BUDGET Holm family has exactly 8 tests;
  - the hierarchical interval pairs split and ref_half within the training seed;
  - the analysed universe is identical across models and views.

## BLOCKER

**B1. The connectivity check flags correct strained bicycles as invalid** (`flexitors/eval/validity.py:163-175`,
`:257-265`).
- Cause: `connectivity_ok = perceived_bonds(mol) == graph`, using `rdDetermineBonds.DetermineConnectivity`'s default
  covalent factor.
- Reproduction: MMFF-relaxed, chemically correct conformers of bicyclo[1.1.1]pentanes — `C1C2CC1C2`,
  `OC12CC(C1)C2`, `N#CC12CC(C1)C2` — each get **one extra perceived bond** (the 1,3-bridgehead pair, ≈ 1.85 Å) and
  `validate_conformer(...).valid == False`. Norbornane, cubane, bicyclobutane and spiro rings pass.
- Scale: a SMARTS for 1,3-bridged four-membered rings matches 40 of the 1000 validation molecules (an upper bound
  for the motif).
- Failure scenario:
  - EVAL §3.1 invalid counts and the "validity-adjusted" secondary are inflated for exactly the cage scaffolds where
    Exp 1's geometry head acts most;
  - a V2 arm that slightly lengthens the bridgehead distance would be credited with "fewer invalid conformers" than
    TD;
  - GT conformers themselves would fail.
- `test_validity.py:32-36` includes norbornane but not a bicyclo[1.1.1] case.
- Fix:
  - define a connectivity failure as a graph bond **missing** from the perception (bond breaking), plus extra
    perceived bonds that are **not** present in the reference (seed/GT) perception;
  - before any milestone-2 use, run `validate_conformer` on every QM9 test and validation GT conformer and report the
    false-invalid rate (target ≈ 0), as a golden.

**B2. The Gate-0 threshold scaling compares two different recipes** (`flexitors/data/gate0_baselines.py:91`, with
`:33-34`, `:79-87`).
- What the code does: `thresholds(res['val'], PUBLISHED_TEST)` divides the validation rigid AMR-R by the published
  0.152 Å.
- The recipe mismatch:
  - the validation value comes from round-2-style `L_etkdg2L` seeds scored directly;
  - 0.152 Å comes from the CTRL_rematch `generate_confs.py` run (`breakdown.log` bin 0, unseeded ETKDG,
    E-JOINT-007).
- The same-recipe test value `res['test']` is computed at `:79-87` and then not used.
- Failure scenario: a systematic recipe offset of ≥ 10 % (for example from seed count, pairing, or embedding from
  the GT graph) triggers, or hides, the EXP-1 §10.3 scaling. The rigid-AMR-R GO threshold, and so Gate 0, then
  moves for a reason unrelated to validation vs test.
- Fix:
  - scale by `res['val'] / res['test']` (same recipe);
  - assert that the same-recipe test value reproduces 0.152 / 0.128 Å within a pre-declared tolerance before the
    Gate-0 ask;
  - if it does not, consult R-JOINT. EXP-1 §10.3's wording names 0.152, but its intent is like-for-like.
- The ring criterion is already like-for-like: 49.6 % comes from the same `l_error.csv` recipe.

## SHOULD-FIX

**S1. The gateway's chdir covers only `diffusion.*` imports** (`flexitors/td.py:79-89`).
- `utils/training.py:5` imports `diffusion.torus`, and `utils/utils.py` imports `diffusion.score_model`, which also
  imports torus. So `import_baseline('utils.training')` or `import_baseline('utils.utils')` (milestone 2: `td_loss`,
  `get_model`) imports torus in the **caller's** cwd.
- The table guard at `:81-83` is then bypassed: torus recomputes and writes 2 × 200 MB tables there.
- With parallel array tasks sharing a cwd, one process can load another's half-written `.npy`, i.e. corrupted score
  tables and wrong scores without an error.
- Fix: run every baseline import under `chdir(TD_ROOT)`, or import `diffusion.torus` under chdir on first call. Also
  pin the SHA-256 of `.p.npy` / `.score.npy` (plan_P2 §2.1).

**S2. `ValidityReport.valid` passes when the reference checks were never run** (`validity.py:137-141`).
- `not self.n_bond_violations` and `not self.n_chirality_flips` are True for None, so a conformer screened without a
  reference is "valid" with tetrahedral stereo unchecked. `test_validity.py:125-128` asserts exactly this.
- EVAL §3.1 counts "stereo ≠ input" as invalid.
- Fix: return None ("undetermined") or require `ref`.

**S3. `valid` includes a criterion EVAL §3.1 does not have** (`validity.py:140-141`).
- `valid` includes `n_bond_violations` (the ±15 % window). EVAL §3.1 defines invalid as: sanitisation failure,
  perception ≠ graph, a clash, or stereo ≠ input. The bond window is EXP-1 §2.6's constraint-layer rule.
- Failure: a conformer with a 16 % stretch but intact connectivity is invalid here and valid by the declared
  definition, so the validity-adjusted secondary would not be the pre-declared one.
- Fix: report the window separately, or have R-SAMPLE amend EVAL §3.1.

**S4. Chirality is version-dependent and noisy near planar centres** (`validity.py:208-229`).
- `stereocentres` uses `FindPotentialStereo`, which differs between the cluster's 2022.09.5 and 2026.03.6 (the
  E-CHEM-060 lesson). `np.sign` of a near-zero χ counts noise as a flip.
- Fix: add the structural degree-4 set (`tools/lgeom.py:232`) with a |χ_ref| floor as the version-independent count;
  record the RDKit version with every count.

**S5. The E/Z table is not validated under the generation RDKit** (`flexitors/tests/test_ez_targets.py:35-43`).
- The reproducibility test is skipped on the cluster. `test_torsion_types.py:116` checks the quadruple *function* on
  100 QM9 molecules under 2022.09.5, but not the committed **rows**.
- Add a cluster test (CPU, seconds) that:
  - for every row, `featurize_mol_from_smiles(corrected)` under 2022.09.5 has a double bond at `bond_idx` with
    `td_reference_quadruple == (c, a, b, d)`;
  - 2022.09.5's ETKDG honours all 138 specified DRUGS targets.
- If 2022 drops a C=N specification, O's seeds are random for that bond and the K1 seed check carries the whole
  effect. That must be known before the Exp 3 primary endpoint (DRUGS, 126 molecules) is read.

**S6. Other-half floor: the split indices are used without an alignment check** (`tools/local_structure_analysis.py:316-319`, `:371-374`).
- `split.tsv` indices are cleaned-GT indices from `make_split_half.py`. The script recomputes `clean_confs` itself
  and never asserts that its pre-cap count equals `sp.n_clean`.
- Any difference (filter or RDKit version) silently mis-assigns halves.
- Fix: add `assert len(clean_confs(corrected, confs)) == n_clean` per molecule (before the cap), and count failures.

**S7. The rescore task does not fail when the round-4 summary is not reproduced** (`experiments/r5_cpu/run.sbatch:47`).
- `cmp A B && echo "round-4 summary reproduced"` does not stop the job under `set -e`: a failing command inside an
  `&&` list is exempt. The job still prints `done rescore`.
- The "byte for byte" claim (IMPLEMENTATION §5) is therefore not enforced, which is round-2 FIXES X3 again.
- Fix: `cmp … || { echo FX_FAIL round4_summary; exit 3; }`, and sync `stdout.txt` and the cmp line to
  `cluster_sync/round5/rescore/`.

**S8. Reading of the re-score** (`cluster_sync/round5/rescore/budget_matched_summary.txt`; IMPLEMENTATION §6.1 and
the JOINT conformance note).
- (a) **Rigid molecules carry no model.** For molecules with `edge_mask.sum() == 0`, TD skips the model
  (`generate_confs.py:153`). Their split output is therefore the other half's GT conformers verbatim: the COPY
  baseline, for every model alike. Part, or all, of CR's −0.038 Å may come from molecules on which no torsion model
  ran.
  - Report split − ref_half stratified by m = 0 / m > 0 (CPU, existing files) before any "coupling" reading.
- (b) **The distinct-target view was promoted after the result.** R-JOINT promoted `split_d` to primary "for the
  coupling reading" after seeing the numbers; the sign flips for B1/S3/S4 in that view. That is post hoc.
  - Every downstream sentence must lead with the pre-declared BUDGET family (EXP-1 §10.5) and label the split_d
    reading exploratory.
- (c) **The 32 excluded molecules are not missing at random.** They are ETKDG-embedding failures. The exclusion is
  paired, but it is informative: those are exactly the cages where GT-L seeds succeed and RDKit-L cannot. State this
  beside every number.

## NIT

- **N1.** The dead-code audit counts docstring/comment mentions in a function's own file as references
  (`flexitors/tests/test_no_dead_code.py:57`, `len(re.findall(...)) > 1`), and any attribute with the same name
  anywhere (`:30-38`). A function named only in its own docstring passes. Count AST `Name`/`Attribute` uses,
  excluding the definition.
- **N2.** `is_sanitizable` (`validity.py:154-160`) is vacuous for generated conformers: `SanitizeMol` ignores
  coordinates, and the graph is TD's input graph. Say so in the docstring, or drop it from `valid`.
- **N3.** `tools/tests/test_lsa_floor.py:33-42` does not assert that the **old** slice includes `i` for n ≤ 3, which
  the docstring and IMPLEMENTATION ("old code gives 0.0 Å") claim. One extra assert keeps the bug from returning
  unnoticed.
- **N4.** For the 85 molecules with > 30 clean conformers, other-half candidates are cut to cleaned indices < 30
  (`local_structure_analysis.py:318`), i.e. ≈ 15 per half, and targets ≥ 30 are never scored. This is a reasonable
  reading of "cap after splitting", but write it down and count those molecules in the summary.
- **N5.** The `floor` task keeps only split-half molecules (`local_structure_analysis.py:386`), so G9 ("n_gt > 3 rows
  unchanged") must be checked on the intersection with the old CSV. State it.
- **N6.** `td_reference_quadruple` is TD's first-neighbour rule. That is correct for Exp 3, but EXP-1 §10.1.2
  requires heavy-first rows for W. Milestone 2 needs a separate heavy-first function, not a reuse of this one.
- **N7.** `qm9_val.py:99-101`: the `corrected_smiles` rule is the argmax over four candidates, two of them built after
  seeing the single-rule scores, on the same 1000 rows that form the acceptance test (99.0 vs 99.1 %). This is low
  stakes (identifier format, no geometry). It still needs R-SAMPLE's sign-off, as IMPLEMENTATION §6.3 already asks,
  and a sentence saying the acceptance test was also used for selection.
- **N8.** The MC check's "share within 3 SE = 0.968" uses a 10-draw SE estimate, so it is not a pass/fail test. Fine,
  since the exact value is primary; label it descriptive.

## Verdict
Milestone 1 is sound as an analysis record. The ref_half statistics, the floor fix and the validation set are
correct, with S6–S8 to tighten.
- **B1 must be fixed and calibrated** on GT conformers before `validity.py` feeds any milestone-2 evaluation.
- **B2 must be fixed** (or answered by R-JOINT) before the Gate-0 summary is used for the M1 ask.
- **S1** should land before any milestone-2 code imports `utils.training` or `utils.utils`.
