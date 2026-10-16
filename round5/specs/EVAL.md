# EVAL: Shared evaluation protocol for all round-5 experiments and Phase 5 generalisation

Author: R-SAMPLE, round 5. Status: P1 spec 2026-10-15; revised 2026-10-16 after D-101 and consults Q-P1-010,
Q-P2-010, Q-P2-011, Q-P3-007 (§1, §1.1, §3.1). Evidence IDs refer to
`round5/ledger/evidence_SAMPLE.md`. Entries E-SAMPLE-061…091 are round-4 BENCH entries with re-checked quotes; no
round-4 verifier has finished them. Builds on `round4/cards/C-BENCH-01…07`. Experiment specs (EXP-1…5) may add
metrics but must not change anything below without a consult (`round5/consult/`) answered by R-SAMPLE.

## 0. Rules that apply to every experiment
1. One evaluator for every row: `torsional-diffusion/evaluate_confs.py` with `--out_results` (per-molecule RMSD
   matrices). Published baselines are quoted *and* re-scored where checkpoints or samples exist (C-BENCH-02).
2. Contrasts, families and decision rules are written into the analysis script **before** results exist (the
   round-2 pattern, `tools/analyze_round2.py:1-23`).
3. Hyper-parameters, thresholds and controllers are tuned on **validation** molecules (TD split index 1), never on
   the 1000 test molecules. ORACLE quantities (test-set GT geometry, the K-aware oracle allocator, leak-free halves)
   are analysis only and printed in separate, labelled tables.
4. Every table states the universe (n molecules), failure convention, sample budget per molecule, NFE, training-L
   source, KG, chirality handling and parameter count (C-BENCH-01; E-SAMPLE-064).

## 1. Datasets, splits, sample counts
| Set | Split | Threshold δ | Samples | Use |
|---|---|---|---|---|
| GEOM-QM9 | GeoMol/TD random split 106586 / 13323 / 1000 [E-SAMPLE-061] | 0.5 Å [E-SAMPLE-063] | 2K per molecule, K = CREST conformers [E-SAMPLE-062, -003] | development, all experiments |
| QM9 validation subset | 1000 molecules from split index 1, built as in §1.1 (Revised after Q-P1-010/Q-P2-010/Q-P3-007) | 0.5 Å | 2K | tuning only |
| GEOM-DRUGS | 243473 / 30433 / 1000 [E-SAMPLE-061] | 0.75 Å [E-SAMPLE-062] | 2K | Phase 5, staged (§9) |
| GEOM-XL | 102 molecules, TD/MCF protocol | AMR only, as TD Table 6 [E-SAMPLE-005] | 2K | size OOD (§5) |

- **Sample budget is part of the method.** TD's 2K rule uses the test-set count K [E-SAMPLE-003]. At matched total
  samples, a uniform allocation gives AMR-R 0.1450 against 0.1784 Å for the 2K rule on our 927-molecule set
  [E-SAMPLE-022]. *Revised after D-101:* that uniform figure redistributes budget from capped pools, so it is
  K-informed. A strictly K-free uniform rule gives 0.1617 Å using only 69 % of the budget [E-SAMPLE-026], so the
  direction holds. Recall keeps rising with samples (0.1783 → 0.1255 Å from 1x to 3x) [E-SAMPLE-020]. Therefore:
  (i) main-table rows use exactly 2K per molecule; (ii) any method that changes per-molecule counts, oversamples or
  selects reports NFE and sits in its own table with a matched-NFE TD row (Zhang et al. demand equal budgets
  [E-SAMPLE-083]; Zhou et al. show oversampling inflates recall [E-SAMPLE-082]).
- QM9 at δ = 0.5 Å is called saturated [E-SAMPLE-081], so finer thresholds are required secondaries (§2).

### 1.1 QM9 validation subset (Revised after Q-P1-010, Q-P2-010, Q-P3-007; 2026-10-16)
One shared validation set serves the whole programme: Exp 1 σ_b selection, the Exp 5 selector, PG settings and
anything else tuned. It must follow the test-set conventions. The provenance of TD's `test_mols.pkl` filtering is not
in our repository, so the builder is **validated on the test split before it is used on validation**.
1. **Universe.** Index 1 of `split.npy`, indexed into the sorted raw-pickle list exactly as
   `torsional-diffusion/utils/dataset.py:165-179` does (13,323 molecules).
2. **Order.** `order = np.random.default_rng(0).permutation(sorted(split[1]))`. Walk `order` and keep the first 1000
   molecules that pass the builder's filters (step 4). Exclusions are listed with reasons.
3. **Ground truth.** GT conformers are the raw GEOM `rd_mol` of every conformer of the molecule, uncapped. The test
   csv has K up to 727, and TD tests on all CREST conformers [E-SAMPLE-054, -003]. They are not the standardized
   pickles, which hold RDKit-matched conformers (the wording of EXP-5 §4.3 meant "the molecules of split 1"). There
   is no cap at 30. That cap is a training-only rule.
4. **Fields and filters.** `smiles` = the raw GEOM key (the pickle file stem; the test csv's `smiles` column has this
   form). `corrected_smiles` and `n_conformers` follow whichever candidate rule reproduces the **test** files. The
   candidates are: `corrected_smiles` = GEOM `mol_dic['smiles']`, or the canonical isomeric SMILES of the first
   conformer's `rd_mol`; and `n_conformers` = the stored conformer count, or the count after `clean_confs`.
   - **Acceptance test (Q-P2-010 default, adopted).** Applied to split index 2, the builder must reproduce
     `test_smiles.csv` (`smiles`, `n_conformers`, `corrected_smiles`) and the per-molecule conformer counts of
     `test_mols.pkl` for ≥ 99 % of the 1000 rows. Residual mismatches are listed in `population.txt`.
   - Filters are the TD/GeoMol ones [E-SAMPLE-061]: drop molecules whose conformers all differ in SMILES, and
     molecules RDKit cannot parse. Single-fragment rules are applied only if the test files show them.
   - **Do not** filter on ETKDG embeddability or on featurisation success at generation time. Those failures are part
     of the protocol (65 on test [E-SAMPLE-069]) and must also occur on validation.
   - `clean_confs` stays inside the evaluator (`evaluate_confs.py:47-54`), as for test.
5. **Commit.** `val_smiles.csv`, the selected indices, a `population.txt` comparing K, n_rot, n_heavy, ring classes
   and the scaffold-novel share with test, and the sha256 of `val_mols.pkl`. The pickle itself is not committed
   (size).
6. **Reuse.** Exp 1 and Exp 5 use the same 1000 molecules. If a stratum is too small (e.g. Exp 1's rigid stratum), an
   experiment extends it with the next molecules of the *same* `order`, so sets stay nested, and declares the
   extension before any result.

## 2. Metrics
| Tier | Metric | Definition / tool |
|---|---|---|
| Primary (pre-registered, `round4/BRIEF.md`) | AMR-R mean, COV-R@0.5 (QM9) / @0.75 (DRUGS) mean, **TD convention** | `evaluate_confs.py` summary (`:163-177`) |
| Key secondary | COV-R@0.1 (QM9) on the success intersection | round-2 driver definition, `tools/analyze_round2.py:10`; `tools/paired_compare.py:41` (`--universe intersection`) |
| Secondary | COV-R sweep 0.05–1.25 Å (curve), δ = 0.05 Å column (FM-refiner reports it [E-SAMPLE-074]); AMR-P, COV-P; medians | `evaluate_confs.py:178-187` |
| Physical (EXP-1 table rows 3–9, adopted here) | bond / angle / torsion MAE, ring-dihedral error, validity, stereo preservation, GFN2-xTB E_relax | `tools/geometry_metrics.py`, `tools/stereo_check.py`, `flexitors/eval/energy.py` (EXP-1 §5.1) |
| DRUGS only | Boltzmann ensemble properties on a published 100-molecule subset, min(2K, 32) conformers, xTB-relaxed [E-SAMPLE-079] | TD §4.4 protocol; subset IDs committed |
| Cost | NFE; GPU wall s per molecule and per conformer; CPU core-s per conformer (TD's unit [E-SAMPLE-080]) | `flexitors/eval/timing.py` (§8) |

Precision is printed beside recall in every table. A recall gain with a precision loss is reported as a trade-off, not
a win (EXP-5 shows that selection can move them in opposite directions [E-SAMPLE-025]).

## 3. Failure handling (report both conventions)
- **TD convention (primary).** The universe is all 1000 test rows. A molecule with no generated conformers counts as
  COV = 0 and is left out of AMR (`evaluate_confs.py:81-84`, `:152`, `:166-167`) [E-SAMPLE-065]. Ground-truth
  conformers whose SMILES changed are removed (`clean_confs`, `:47-54`), and molecules left with none are skipped in
  every arm alike.
- **ET-Flow convention (secondary).** ET-Flow's evaluator *skips* molecules with missing positions, several fragments,
  or fewer than 2K conformers [E-SAMPLE-066]; MCF/S23D used 995 molecules [E-SAMPLE-068]. We report the **success
  intersection** across all arms of a table (molecules every arm generated for) as the ET-Flow-like number. We do not
  run ET-Flow's code on our files.
- Our released-model reproduction: 65 failures (60 strained cages ETKDG cannot embed, 5 rejected inputs)
  [E-SAMPLE-069]. These explain why our COV-R (89.0 %) sits below the published 92.8 % [E-SAMPLE-090, -070].
- **Failures are partly stochastic** (65 / 69 / 69 over three sampling seeds of the same model [E-SAMPLE-051]). Every
  table prints failures per arm and seed. Paired tests use the fixed 1000-row universe (TD convention) and the
  intersection (secondary). A method that adds retries or a fallback embedder for failed molecules must report that
  as a separate change, with its own row.
- "Additional failures" (NaN RMSD rows, `evaluate_confs.py:117-120`) are counted and printed.

### 3.1 Conformer-level invalidity (Revised after Q-P2-011; 2026-10-16)
A generated conformer is invalid if any of these holds: it fails sanitisation; bond perception from 3D differs from
the input graph; it has a clash (EXP-1 §3 row 5); or its stereo differs from the input (EXP-3 / `tools/stereo_check.py`).
- **Primary (a):** the evaluator is unchanged. Invalid conformers are scored like any other conformer, as in TD and
  every published table. The number of invalid conformers per arm and seed is printed beside every primary row.
- **Secondary (b), "validity-adjusted":** each invalid conformer keeps its slot, so the output count stays 2K.
  - For recall, its RMSD column is set to +∞, so it can never cover a reference.
  - For precision, it counts as uncovered in COV-P. AMR-P is computed over valid conformers only, with the invalid
    count printed.
- **(c) Dropping invalid conformers is never allowed.** It changes the sample count (recall falls, precision rises).
  EXP-1 §3 row 5 already says "failures counted, never dropped".
- Any method that *filters* invalid conformers before output (e.g. a selector) must declare the filter as part of
  the method, keep the output at exactly 2K, and report it in its own row.

## 4. Strata (always printed for the primary metrics; Holm not applied, descriptive)
- **Flexibility:** heavy rotatable bonds 0 / 1 / 2 / 3 / 4+ (QM9); DRUGS 0–3 / 4–6 / 7–9 / 10–12 / 13+ (TD Fig. 8 range
  [E-SAMPLE-004]). TD's own rotatable count (`edge_mask.sum()`) is printed beside the heavy-atom count.
- **Size:** n_heavy (QM9 ≤ 9; DRUGS quintiles).
- **Reference ensemble size K:** 1 / 2–3 / 4–10 / 11–30 / > 30. Recall growth with samples differs by K and n_rot
  [E-SAMPLE-021].
- **Rings:** acyclic / one ring / fused, bridged or spiro / strained cage (the ETKDG failure class). This is the
  mechanism stratum for Exp 1.
- **Scaffold novelty** (§5 tier 1).
- Implementation: `tools/breakdown.py` (existing bins) extended in `flexitors/eval/strata.py`.

## 5. OOD set (Phase 5 generalisation)
| Tier | Set | Needs | Status |
|---|---|---|---|
| 1 (free) | **QM9 scaffold-novel stratum**: test molecules whose Bemis–Murcko scaffold (RDKit, no chirality) occurs in no training molecule. 87 / 1000 molecules (802 scaffold-seen, 111 acyclic); they fail more often (15 / 87) and are rigid (mean n_rot 0.26) [E-SAMPLE-052] | nothing; list in `round5/ledger/sample_data/ood_strata.csv` | descriptive only (n = 87, wide CI) |
| 2 | **GEOM-XL** (102 molecules, 136 atoms and 32 rotatable bonds on average) with the released DRUGS checkpoint [E-SAMPLE-005]; AMR-R/AMR-P, as TD | DRUGS stage D0 | after D0 |
| 3 | **Cross-dataset**: DRUGS-trained model on the QM9 test set (ET-Flow reports 86.68 % / 0.218 Å [E-SAMPLE-075]) | DRUGS checkpoint | after D0 |
| 4 | **Scaffold re-split with retraining** (ET-Flow's SS: QM9 95.00 / 0.083, DRUGS 76.06 / 0.644 [E-SAMPLE-075]) | a new split and training runs | only for the final method of Exp 1, user approval |

An OOD claim needs tier 2 or 3. Tier 1 is reported as a stratum and is never used alone.

## 6. Leak-free oracle (ORACLE, analysis only)
- Use the split-half design already built (`tools/make_split_half.py`, `tools/check_split_seeds.py`,
  `tools/analyze_leakfree.py`) [E-SAMPLE-087]. TD's App. F.1 "another conformer's L" (0.284 Å on DRUGS) is the
  published precedent [E-SAMPLE-086].
- Universe: 744 molecules with ≥ 2 clean GT conformers (253 single-conformer molecules excluded). The reference rows
  are CTRL_rematch split 0.2195 vs own 0.1603 vs RDKit L 0.1873 Å, and COPY 0.3336 Å [E-SAMPLE-050].
- Rules: print the split, own and leak (own − split) columns with the COPY baseline. Never compare half-ensemble
  numbers with full-ensemble or published values. Never select a model or hyper-parameter with them. Own-conformer
  (`gtL`, `gtLcycle`) values appear only as the "leaky" column.

## 7. Statistics
- **Paired molecule bootstrap** of the per-molecule difference (arm − ref), 5000 resamples, percentile 95 % CI and
  two-sided bootstrap p (`tools/analyze_round2.py:133-137`); **Holm** over the pre-declared primary family
  (`:154-159`). Each experiment's primary family is (its pre-declared contrasts) × (AMR-R, COV-R@0.5). The key
  secondary has its own Holm family. Universe handling follows `:123-130` (failure → COV 0).
- **Seeds.** Trained arms use 3 training seeds. Values are per-molecule seed means, and the seed SD is printed next to
  every CI [E-SAMPLE-088]. Inference-only arms (EXP-5, DRUGS D1) use 3 sampling seeds. GeoMol's 3-run spread on QM9 was
  ± 0.0012 Å AMR-R [E-SAMPLE-076]. Non-inferiority claims use the hierarchical (seeds, then molecules) bootstrap
  (`:140-151`) with a margin declared in advance.
- **Power reference:** round 2 resolved a +0.0032 Å AMR-R difference with CI [+0.0013, +0.0052], n = 934
  [E-SAMPLE-089]. Effects < 0.002 Å are below what QM9 can resolve with one sampling seed.
- Medians, strata and OOD tiers are descriptive (no Holm). Every p-value states its family.

## 8. Runtime and compute reporting
- Report per arm: total NFE; NFE per conformer; GPU wall-clock per molecule (packed, PACK = 3, as in round 2); CPU
  core-seconds per conformer for anything run on CPU (RDKit seeding, MMFF, xTB, selection); and the RDKit/ETKDG share.
- TD's published runtime unit is CPU core-seconds per conformer on 10 DRUGS molecules × 8 conformers (TD 20 steps
  4.90 s; RDKit 0.10 s) [E-SAMPLE-080]; reproduce that unit once per method on CPU.
- Anchor: the released QM9 model generated the full test set in 14 min 18 s wall in one job [E-SAMPLE-053]; xTB costs
  ≈ 0.14 s per QM9 optimisation [E-SAMPLE-091].
- Every generation script writes `budget.json` (NFE, CPU s, wall s per molecule). An arm without it is not tabled.

## 9. GEOM-DRUGS staging (reuses C-BENCH-05; each stage gated)
| Stage | What | Needs | Cost (estimate) |
|---|---|---|---|
| D0 | Download TD's DRUGS archive (25.3 GB), released `drugs_default` model (6.4 MB) and TD's released DRUGS samples `drugs_steps20.pkl` (457 MB) [E-SAMPLE-077]. Score the released samples with our evaluator (does 72.7 % / 0.582 Å reproduce under both failure conventions? [E-SAMPLE-071]). Run RDKit and RDKit + MMFF baselines | **user approval for the download** (protocol: downloads need explicit permission); /scratch space check | CPU hours; ≤ 20 GPU-h with ET-Flow `drugs-o3` re-scoring [E-SAMPLE-072] |
| D1 | Inference-only arms with the released DRUGS model (config equals our QM9 architecture with 74 node features [E-SAMPLE-078]): EXP-5 pools and arms, MMFF/xTB seeds, stereo checks for EXP-3 | D0 done | ≈ 10–60 GPU-h depending on arms (PG notes > 200 samples per DRUGS molecule on average [E-SAMPLE-034]) |
| D2 | Trained arms (EXP-1 method + a CTRL retrained in the same pipeline) | QM9 result and D1 gain; user approval | TD trained 250 epochs in 4–11 days on one A6000 [E-SAMPLE-084]; our estimate is 1.6–4.4 GPU-days per 100 epochs (E-BENCH-060, not re-checked) |
- Compare against TD 72.7 / 0.582 [E-SAMPLE-071], ET-Flow 79.53 / 0.452 [E-SAMPLE-072] and, quoted, MCF / DMT /
  S23D (round-4 BENCH entries). DRUGS molecules average 44.4 atoms and 6.5 rotatable bonds [E-SAMPLE-085], so measure
  xTB and evaluation cost on 20 molecules before any full run.
- Publish the DRUGS 100-molecule property subset IDs [E-SAMPLE-079].

## 10. Code (`flexitors/eval/`; coordinates with EXP-1 §5.1, which owns validity / geometry / stereo / energy / leakfree / timing)
| File | Contents | Touchpoints |
|---|---|---|
| `flexitors/eval/metrics.py` | load `eval.pkl`; per-molecule AMR-R/P and COV-R/P at any δ; TD-convention and intersection aggregates; sweep | `torsional-diffusion/evaluate_confs.py:38-44`, `:150-187`; `tools/analyze_round2.py:95-107` (`per_mol`) |
| `flexitors/eval/stats.py` | `paired_bootstrap`, `holm`, `hierarchical_upper`, family bookkeeping; port with unchanged numerics | `tools/analyze_round2.py:123-159` |
| `flexitors/eval/strata.py` | §4 bins; scaffold-novel flag from the training split | `tools/breakdown.py:32-80`; `round5/ledger/sample_data/ood.py`; `torsional-diffusion/utils/dataset.py:165-179` (split indexing) |
| `flexitors/eval/budget.py` (shared with EXP-5) | read `budget.json`; NFE-matching checks | EXP-5 §8 |
| `flexitors/eval/report.py` | one table writer: universe, failures per seed, both conventions, CIs, seed SDs, NFE columns | — |

Tests (`flexitors/tests/test_eval_*.py`):
- `metrics` reproduces `R0_base_seed0/summary.txt` exactly from its `eval.pkl` (AMR-R 0.1752, COV-R 89.0293, 65 failures).
- `stats` matches `contrasts.csv:50` (S3 vs CR, +0.00323 [+0.00127, +0.00523]) with the same seed.
- `strata` reproduces the "by n_rot_heavy" bins of `R0_base_seed0/breakdown.log`.
- The scaffold-novel count equals 87.
