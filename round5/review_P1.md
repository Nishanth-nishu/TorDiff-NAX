# Code review P1: Stage A milestone 1 (clean code and structure)

Reviewer: coding planner P1, 2026-10-17. Scope: commits `c594757..ff58972`, read against `round5/DECISION.md`, the
binding spec sections and the design rules of `round5/plans/plan_P1.md`:
- one baseline gateway;
- no dead code;
- ports only where TD cannot be imported, each with a golden test;
- every mechanism cites a real spec § or E-ID.

No code was edited and nothing was committed.

**Re-run locally (Windows, Python 3.11, RDKit 2026.03.6, no torch_geometric):**
- `python -m pytest -q flexitors/tests` gives 56 passed, 2 skipped;
- `python -m pytest -q tools/tests/test_lsa_floor.py tools/tests/test_leakfree.py tools/tests/test_leakfree_budget.py`
  gives 14 passed.

Both match `round5/IMPLEMENTATION.md` §3. I did not re-run the cluster-only tests; they are recorded in
`cluster_sync/round5/tests/`.

**Citation spot-check.** Every E-ID cited in the new code exists in the ledgers and says what the comment claims:
- E-JOINT-004, -007 and -031 in `evidence_JOINT.md`;
- E-CHEM-020 and -066 in `evidence_CHEM.md`.

The spec sections cited also exist with matching content: EXP-1 §2.6/§2.7/§2.8/§3.6/§9/§10.1–10.5, EXP-3
§3.1/§3.3/§9.2–9.5, EVAL §1.1/§3.1, EXP-5 §12.4, and DECISION grafts/fixes 1, 4–7. No invented citation was found.

**Overall.** The structure follows the agreed design closely:
- one gateway;
- empty `models/` / `sampling/` / `training/` packages until a job needs them;
- small, typed, documented functions;
- ports marked as ports and golden-tested on gnode118;
- tests that check behaviour (hand-built geometry, enumeration oracles, reproductions of published counts), not just
  shapes.

One blocker concerns the gateway's main guarantee, which will break in milestone 2.

## BLOCKER

**B1. `import_baseline` changes the cwd only for names that start with `diffusion`, but TD modules import
`diffusion.torus` indirectly** (`flexitors/td.py:79-80`).
- The chain: `utils/utils.py:6` → `diffusion/score_model.py:11` → `diffusion.torus`. `utils/training.py:5` imports
  `diffusion.torus` directly. `train.py:5-9` imports both, and `utils/boltzmann.py:1` imports `diffusion.sampling`.
- Failure scenario: in milestone 2, `import_baseline('utils.utils')` (for `get_model`) or `import_baseline('train')`
  (for `save_checkpoint` / `load_resume_state`, plan_P1 §1.2) is called from `experiments/…` or `$RES`. `torus.py:26-34`
  then finds no `.p.npy` in the caller's cwd, recomputes the 5001 × 5001 tables (minutes per process, a race under
  PACK=3) and writes ≈ 400 MB into that directory. This is exactly what the gateway docstring (`td.py:8-11`) promises
  cannot happen.
- `test_torus_import_uses_td_tables_and_keeps_cwd` (`flexitors/tests/test_td_gateway.py:80-86`) imports
  `diffusion.torus` directly and cannot see the bug.
- Fix: apply the table check and `os.chdir(TD_ROOT)` to every import that resolves into `TD_ROOT`, not by name
  prefix. Simplest is to do it for every non-tools name, or whenever `name.split('.')[0]` is `diffusion`, `utils`,
  `train`, `generate_confs`, …. Extend the cluster test to `import_baseline('utils.training')` and
  `import_baseline('utils.utils')` from a temporary cwd.

## SHOULD-FIX

**S1. `ValidityReport.valid` adds a criterion that EVAL §3.1 does not have** (`flexitors/eval/validity.py:54-55`).
- `not self.n_bond_violations` makes the ±15 % bond window (an EXP-1 §2.6 *step limit* of the constraint layer) an
  invalidity criterion.
- EVAL §3.1 defines invalid as: sanitisation fails, 3D bond perception ≠ input graph, a clash, or stereo ≠ input.
- A 16 % stretched bond that RDKit still perceives would make a conformer "invalid" in the validity-adjusted secondary
  metric. That changes a pre-declared EVAL metric without a consult.
- Keep the counts in the report, drop them from `valid` (or file a consult with R-SAMPLE).
- `rings_closed` (`:54`, `:179`) is implied by `connectivity_ok`. Keep it as a diagnostic field, but it need not be in
  `valid`.

**S2. The E/Z target table's atom indices are never checked under the RDKit that generates conformers**
(`flexitors/tests/test_ez_targets.py:319-327`, skipped unless RDKit = 2026.03.6).
- The table is keyed by `bond_idx` and `(c, a, b, d)` of `AddHs(MolFromSmiles(...))` built under 2026.03.6
  (`flexitors/data/ez_targets.py:155`, `:168-180`). Generation runs under the cluster's 2022.09.5 (EXP-3 §9.4).
- Only "specified" and φ are meant to be version-pinned; the structural set and quadruples are version-independent
  by design. That independence is asserted nowhere on the cluster.
- Failure scenario: any atom- or bond-order difference in SMILES parsing between versions would silently attach
  targets to the wrong bond in milestone-2 K1/K2.
- Add an unpinned test, run on the cluster, that recomputes `structural_stereo_bonds` for every table row and
  compares `bond_idx` and the quadruple only.

**S3. Two implementations of n_rot_heavy, with no equivalence test.**
- `flexitors/geometry/torsion_types.py:130-140` (`heavy_rotatable_bonds`, used for population descriptors,
  `flexitors/data/qm9_val.py:124`).
- `tools/breakdown.py:32-50`, which defines the rigid stratum that Gate 0 actually reads (`gate0_baselines.py:59-63`
  reads `breakdown.csv`).
- The hand cases (`test_torsion_types.py:201-206`) do not show the two agree on real data. `population.txt` reports
  318 rigid test molecules, while E-JOINT-007 says 289 *evaluated*; the difference is plausibly the 65 failures, but
  it is unchecked.
- Add a cluster test comparing `heavy_rotatable_bonds` with the `n_rot_heavy` column of `R0_base_seed0/breakdown.csv`
  (`breakdown.py` parses argv at import, so compare against its output file rather than importing it).

**S4. The round-4 reproduction check in the rescore task cannot fail the job**
(`experiments/r5_cpu/run.sbatch:47`).
- `cmp … && echo "round-4 summary reproduced"`: under `set -e`, a failing command on the left of `&&` does not exit
  the script. A mismatch therefore only omits one echo line.
- IMPLEMENTATION §5 relies on "reproduces byte for byte". Make it `cmp … || { echo FAIL; exit 3; }`.

**S5. `round5/sync_cluster.sh:107` runs `rm -rf $P/round5` on every sync.**
- Today `$P/round5` holds only the copied fixtures. But milestone-2 agents extending the layout (e.g. a decision file
  or a cache next to `round5/data`) would lose it silently on the next sync.
- Replace only the two copied subtrees (`round5/ledger/chem_data`, `round5/data`).

**S6. Shape validity primitives for reuse by the milestone-2 constraint layer** (readability and extension).
- `clash_count`, `bond_window_violations` and `chirality_flips` (`validity.py:92-143`) take RDKit `Mol` objects one
  conformer at a time.
- The EXP-1 §2.6/§2.8 guarded step needs the same three tests on batched position arrays at every sampler step. As
  written, milestone 2 will re-implement them, which is the duplication the dead-code / no-copy rules aim to prevent.
- Split each check into an array-level primitive (positions + precomputed index arrays: bonded pairs, non-bonded heavy
  pairs with limits, centre rows) and a thin `Mol` wrapper. Then the guard and the evaluator provably use one
  definition.
- While doing this, add the ring-fusion bridgeheads that EXP-1 §2.8 also guards ("signed volume at each bridgehead").
  `stereocentres` (`:122-133`) covers only `FindPotentialStereo` tetrahedral centres.

**S7. The dead-code audit is weaker than its docstring states** (`flexitors/tests/test_no_dead_code.py`).
- `:149`: `len(re.findall(rf'\b{name}\b', own)) > 1` counts any mention of the name in its own module, including
  docstrings and comments. A function named in its own module's docstring therefore passes.
- `:133-135`: only top-level definitions are checked, not methods or properties (e.g. `ValidityReport.valid` is used
  only by tests today).
- `:120-130`: attribute names from all modules are pooled, so generic names (`build`, `positions`, `main`) are
  "referenced" by unrelated code.
- Use AST references only: strip docstrings by walking `ast.Name` / `ast.Attribute` and exclude the defining node.
  Restrict pooled attribute matches to `module.name` forms. Methods may stay out of scope, but say so in the
  docstring.

## NIT

- **N1.** `flexitors/td.py:76-78`. `TOOLS_ROOT` and `TD_ROOT` stay on `sys.path` for the rest of the process, after
  any first call. Document this in the docstring, because importing any flexitors data module makes bare
  `import utils` resolve to TD.
- **N2.** `tools/local_structure_analysis.py:371`, `sep='	'`. A literal tab character inside the quotes is invisible
  in review; write `'\t'`.
- **N3.** `tools/local_structure_analysis.py:319-320`. The other-half candidates are `j < n` after the global
  30-conformer cap, so each half is effectively capped at about 15 when n_clean > 30. JOINT's "cap applied after
  splitting" (EXP-1 §10.5) could also mean 30 per half. State the reading in the help text and in IMPLEMENTATION §2.
- **N4.** `flexitors/tests/test_torsion_types.py:131-136`. The "reference-atom swap" hand case asserts the quadruple
  and stereo atoms but not the toggled φ, which the comment says is the point. The parametrised embedding test
  (`:139-149`, `'C(C)(/F)=C/C'`) covers it; add `assert b.phi == <expected>` so the hand case tests what it claims.
- **N5.** `flexitors/tests/test_torsion_types.py:152-155`. The bond is picked by list position (`[1] if len > 1`), so
  the test passes vacuously if the bond order changes. Select the C=C bond explicitly (`GetBondType() == DOUBLE`).
- **N6.** `flexitors/tests/test_validity.py:32-36`. `ref=m` makes the bond-window and chirality checks trivially
  zero. Use a second embedding, or the unrelaxed seed, as `ref`.
- **N7.** `flexitors/eval/validity.py:111-119`. Ring bonds are counted inside "all" and again separately. The
  docstring says so, but the report field names (`n_bond_violations`, `n_ring_bond_violations`) do not; rename to
  `n_bond_violations_all`.
- **N8.** `flexitors/data/ez_targets.py:203-205`. `embed_failed` rows are accepted silently. Print their count in the
  `.txt` summary (it is in the per-dataset `value_counts` only for specified rows).
- **N9.** `flexitors/provenance.py:119-120`. `dirty` checks only `flexitors` and `tools`. Add `experiments` and
  `round5/data`, which also feed the jobs.
- **N10.** `flexitors/eval/validity.py:98`. `GetRcovalent` values differ slightly between RDKit versions, so the clash
  count is version-dependent at the margin. Record the RDKit version in reports (provenance already does), or pin the
  radii table.
- **N11.** `flexitors/data/qm9_val.py:45`. `_graph` has no docstring and accepts either a str or a Mol. Type it
  (`str | Chem.Mol`) or split it into two helpers.
- **N12.** `TEST_ONLY_API = {'validate_conformer'}` (`test_no_dead_code.py:111-113`) makes all of `eval/validity.py`
  test-only in milestone 1. This is documented and acceptable. Milestone 2 must give it a production caller or remove
  the entry; the test enforces that.

## What is good and should be kept as the pattern for milestones 2–3
- `flexitors/td.py`: one module, SHA pins of all 20 TD files with the round-3 pins cross-checked
  (`test_baseline_frozen.py:30-47`), and a test that no other module touches `sys.path` or the cwd
  (`test_td_gateway.py:65-70`).
- Ports declared as ports and golden-tested against TD's own code on the cluster: `td_rotated_side` and
  `td_reference_quadruple` vs `get_transformation_mask` and the `sampling.py:129-142` construction
  (`test_torsion_types.py:174-198`).
- Oracles that are not the implementation: exact subset expectations checked by full enumeration
  (`test_leakfree_budget.py:31-38`); φ targets checked by independent ETKDG embeddings; ring baselines reproducing
  4,181 / 49.6 % / 35.1 %; the floor bug reproduced on a hand-made two-conformer case (`test_lsa_floor.py:58-71`).
- Pre-declared analysis text in `tools/analyze_leakfree.py` (module doc, BUDGET family) before the outputs existed,
  and round-4 outputs kept byte-identical.
- IMPLEMENTATION §6 raises the contradictions (Phase-2 conclusion reversed at matched budget, `corrected_smiles` rule,
  scaffold count) instead of resolving them silently.
