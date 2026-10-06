# Verification of round2/research_A.md (citation and claim verifier)

Written 2026-10-06. Method: every quote in §3 was searched in the cited fulltext file (whitespace, hyphenation and
non-alphanumeric characters normalised), and the page was set by counting form feeds. The two new PDFs were also checked
directly with `pdftotext` (page 1 title and authors, and the page of each quote). Every `path:line` was opened. All
numbers were recomputed with my own scripts from `cluster_sync/results/`. research_A.md was not edited.

## 1. Paper citations (E1-E28)

All 28 PDFs and fulltext files exist. All 28 quotes are found verbatim on the stated PDF page. E9, E10, E16, E20 and E21
fail a naive substring search only because the source page has two columns; reading the columns in order gives the exact
quote. Printed page = PDF page holds wherever a footer exists. EBD p.1 (preprint) and GO-Flow p.7 have no printed number.

| # | claim (short) | location | verdict | note |
|---|---|---|---|---|
| E1 | GT-L training causes a test-time shift | TD p.7 | VERIFIED | §4.1 |
| E2 | GT-L model: lower loss, worse inference | TD p.26 | VERIFIED | Table 8 item 5; tested on RDKit L, which is our B1-on-RDKit setting |
| E3 | QM9: OMEGA has better L | TD p.26 | VERIFIED | QM9 paragraph |
| E4 | relaxing rigid L left to future work | TD p.22 | VERIFIED | |
| E5 | CDM conditioning augmentation vs train-test mismatch | CDM p.3 | VERIFIED | |
| E6 | amortise over the augmentation level | CDM p.18 | VERIFIED | s is the truncation time of the 32->64 model. The link to an L-quality λ is an analogy |
| E7 | mismatch = stage-1 samples out of distribution | CDM p.18 | VERIFIED | "exact B1 pattern" is the author's analogy |
| E8 | Gaussian noise is the effective augmentation | CDM p.6 | VERIFIED | true at low resolution only, as the quote says |
| E9 | DDPM-IP perturbs GT inputs | Ning p.1 | VERIFIED | Two-column abstract. DDPM-IP perturbs the denoiser's own noisy input x_t, not a conditioning variable, so this is an analogy |
| E10 | Gaussian input perturbation | Ning p.4 | VERIFIED | same note as E9 |
| E11 | EBD adjusts the coarse prior | EBD p.1 | VERIFIED | |
| E12 | RINGER: NeRF needs ring closure | RINGER p.5 (printed 5/28) | VERIFIED | The quote is about **macrocycles**. Its use in R2A-6 for small QM9 rings is an extrapolation |
| E13 | PuckerFlow rings + pretrained TD | PuckerFlow p.21 | VERIFIED | |
| E14 | GEOM bonds reflect GFN2-xTB | Nikitin p.10 | VERIFIED | GEOM-Drugs only. research_A correctly marks QM9 as UNVERIFIED |
| E15 | product-space diffusion is independent per group | DiffDock p.6 | VERIFIED (quote) / DOES NOT SUPPORT (gloss) | The quote is about independent forward kernels. The gloss "an L-level is a second time variable" is the author's own inference; R2A-2's λ is not diffused |
| E16 | Torsional-GFN follows unseen L | T-GFN p.4 | VERIFIED | Two-column. Torsional-GFN is trained from an energy, not from data. Weak support for a data-trained model |
| E17 | FM-refiner starts from upstream samples | FM-refiner p.3 | VERIFIED | |
| E18 | ET-Flow QM9 AMR-R 0.073 | ET-Flow p.19 | VERIFIED | Appendix Table 9 (OOD table). The same numbers appear in main-text Table 2 (QM9). Better to cite Table 2 |
| E19 | RDKit RMSE 0.03 Å / 4.1° | Corso thesis p.90 | VERIFIED | Same sentence is in TD App. F.1 (fulltext line 1271). These are DRUGS numbers |
| E20 | GO-Flow: internal coordinates most critical | GO-Flow p.7 | VERIFIED (quote) / DOES NOT SUPPORT for R2A-5 | Cited for the **Cartesian** refiner R2A-5. GO-Flow p.7 says the w/o-C variant "essentially degrades to a Cartesian-based" model and is worse, so it argues **against** a pure-Cartesian L model. Fine for R2A-6 |
| E21 | GO-Flow w/o C: COV-R 90.26 / MAT-R 0.8512 | GO-Flow p.7 | VERIFIED | DRUGS, GD-P (matches INDEX row) |
| E22 | rings delegated to the L sampler | TD p.23 | VERIFIED | |
| E23 | ring-pucker hypersphere factor | Corso thesis p.63 | VERIFIED | |
| E24 | manifold as a soft constraint | Corso thesis p.64 | VERIFIED | Generic future-work list. Weak support |
| E25 | DiffDock robust to inaccurate input structures | Corso thesis p.63 | VERIFIED | The source says "preliminary" and is about protein structures. Weak support |
| E26 | FoldingDiff wrapped normal | FoldingDiff p.5 | VERIFIED | |
| E27 | GeoMol predicts LS first | GeoMol p.4 | VERIFIED | |
| E28 | GT-other-L floor 0.284 Å (DRUGS) | TD p.21 | VERIFIED | |

**New papers.**
- `2021_ho_cascaded_diffusion.pdf`: page 1 is "Cascaded Diffusion Models for High Fidelity Image Generation" by Ho,
  Saharia, Chan, Fleet, Norouzi and Salimans (arXiv 2106.15282v3). Correct.
- `2023_ning_input_perturbation.pdf`: page 1 is "Input Perturbation Reduces Exposure Bias in Diffusion Models" by Ning,
  Sangineto, Porrello, Calderara and Cucchiara (arXiv 2301.11706v3; "Published as a conference paper at ICML 2023").
  Correct.
- Both fulltext files have form-feed pages that agree with pdftotext. INDEX.md rows 52-53 are accurate (venue, arXiv id,
  pages 3/6/18 and 1/4).

## 2. Code citations

| citation | claim | verdict | note |
|---|---|---|---|
| generate_confs.py:18, :22 | `--pre_mmff`, `--seed_confs` | VERIFIED | |
| generate_confs.py:75-82, :119-134, :137 | seed-pickle path; mmff passed to embed_seeds | VERIFIED | **But** `mmff=args.pre_mmff` is passed only on the SMILES path (`:136-137`). The `--seed_confs` branch (`:132-134`) ignores `--pre_mmff`. See R2A-1 below |
| generate_confs.py:102 | strict=True load | VERIFIED | |
| sampling.py:75, :94-102, :171, :105 | random.choice; perturb_seeds; node_sigma; sample() | VERIFIED | |
| utils/xtb.py:44 | xtb_optimize | VERIFIED | Uses a per-PID work dir `/tmp/{pid}` (`:5`), so multiprocessing is safe |
| standardize_confs.py:59-67, :73, :83-87, :116 | cap; mol_rdkit from GT conf 0; MMFF variant; rd_mol overwrite | VERIFIED | |
| **standardize_confs.py:108** | "matched conformer is already heavy-atom aligned to GT by AllChem.AlignMol" | **WRONG** | `:108` is `AllChem.AlignMol(REMOVE_HS(mol_rdkit_single), REMOVE_HS(mol))`, and `REMOVE_HS` (`:33`) is `Chem.RemoveHs(x, sanitize=False)`, which returns a **new** molecule. Only that temporary copy is moved. `mol_rdkit_single`, the object stored at `:116`, is **not aligned**. `optimize_rotatable_bonds` (`utils/standardization.py:29-44`) only sets torsions. R2A-1 source 6 and R2A-2/R2A-3 must align explicitly: heavy-atom Kabsch, with the transform applied to all atoms |
| utils/dataset.py:24-43, :29, :37-42, :38, :41 | transform; conformer pick; sigma; modify_conformer | VERIFIED | |
| utils/dataset.py:126-184, :143-148, :200-241, :209 | filter_smiles; raw path; featurize_mol loop | VERIFIED | |
| score_model.py:65-75, :79-106, :113-128, :141-152, :113-161, :188-193 | input embeddings; trunk with 1o; bond head; sigma embedding | VERIFIED | |
| training.py:7-34, :19-25 | train_epoch; torus score loss | VERIFIED | |
| parsing.py:25-29 | where new train args go | VERIFIED | sigma_min/max block |
| torsion.py:57-75 | modify_conformer | VERIFIED | |
| tools/make_seed_pickles.py:53-77 | seed-pickle writer | VERIFIED | |
| tools/local_structure_analysis.py:163, :196-215, :280, :285, :295-298 | torsion_floor; set_acyclic_local; top-m; single best seed jb; oracles | VERIFIED | Oracles copy the **target conformer's own** GT angles/bonds (`gt.GetConformer()`, `:297`), i.e. an own-L oracle |
| slurm B3 line, standardize VARIANT | B3_match_mmff exists | VERIFIED | `slurm/ablations_train_deferred.tsv:4` |
| CPU estimates (3-10 h DE; 1-4 h standardise) | round-1 estimates | VERIFIED | `notes/ablation_plan_qm9.md:64`, `:82`; `slurm/standardize_qm9.sbatch:23` |
| "~16-18-run round" | | UNVERIFIABLE | Not in BRIEF or notes |

## 3. Recomputed numbers (own scripts)

**D1 strata** (`local_structure_test.csv`; 7611 rows, of which 7554 have no error; 57 are embed_failed). Every
per-conformer mean in the D1 table matches to 4 dp: all 0.0986/0.0885/0.0735/0.0269/3.88; acyclic
0.0510/0.0400/0.0081/0.0137; any ring 0.1171/0.1072/0.0989/0.0323; r3 0.0771; r4 0.1054; r5 0.2117; r6 0.1370; r≥7
0.2180/5.49. n per stratum matches (r≥7 = 124 seven-rings + 8 nine-rings). Ring share 72.1% of conformers. **VERIFIED.**

Problems with how D1 is read:
1. **"The 5-membered-ring stratum is the worst": WRONG** even in the table itself. r≥7 is 0.2180 > r5 0.2117.
2. **Weighting.** D1 is a per-GT-conformer mean, but AMR-R is a mean over molecules. Per-molecule means:

   | stratum | n molecules | floor_best_sym | local_oracle |
   |---|---|---|---|
   | all | 940 | 0.1145 | 0.0978 |
   | acyclic | 107 | 0.0589 | 0.0097 |
   | any ring | 833 | 0.1217 | 0.1091 |
   | r3 | 383 | 0.1059 | |
   | r4 | 179 | 0.1362 | |
   | r5 | 177 | 0.1239 | |
   | r6 | 77 | 0.1136 | |
   | r≥7 | 17 | 0.3379 | |

   Molecule-weighted, r5 is not exceptional. Ring molecules are 89% of molecules, not 72%. The conclusion that rings
   dominate gets stronger.
3. **"(c) can reach at most ~0.01 Å (0.025 with bonds)" contradicts research_A's own caveat.** The oracle is applied to
   the single best seed. `floor_best_sym` is a minimum over the top 3 seeds. So the gain is under-estimated, and "at
   most" is the wrong direction. There is also an opposite bias: the oracle uses the conformer's **own** L, which is
   optimistic. Neither bias is quantified, so "≈0.01-0.025 Å, bias direction unknown" is the defensible statement.
4. `floor_gtother` is averaged over 7329 rows (molecules with ≥2 GT conformers), while the other columns use 7554. This
   is negligible except for r≥7: on that subset `floor_best_sym` = 0.2067, not 0.2180.
5. `set_acyclic_local` can make the floor **worse**: `local_oracle_rmsd` > `floor_rmsd` in 4.1% of rows (10.7% for r6).

**D2** (`local_structure_test_mmff.log`): 0.0986→0.0934, median 0.0606→0.0462, angle 3.88→2.44°, bond 0.036→0.020 Å.
**VERIFIED.**

**D3 / B1 own-L vs random-L.** Per-run `summary.txt` MAT-R mean (all n=996, 4 model failures, same generation seed 0):

| training seed | B1 gtL | B1 gtLcycle | diff | CTRL_base gtL | CTRL_base gtLcycle | diff |
|---|---|---|---|---|---|---|
| s0 | 0.0374 | 0.0215 | -0.0159 | 0.0823 | 0.0810 | -0.0013 |
| s1 | 0.0385 | 0.0215 | -0.0170 | 0.0842 | 0.0821 | -0.0021 |
| s2 | 0.0349 | 0.0206 | -0.0143 | 0.0863 | 0.0830 | -0.0033 |
| mean | 0.0369 | 0.0212 | **-0.0157** | 0.0843 | 0.0820 | -0.0022 |

- The means match research_A (0.0369 / 0.0212 / 0.0842 / 0.0820).
- A training-seed-paired comparison is possible. All 3 seeds agree in sign. The spread of the difference is about
  ±0.0014, which is small next to -0.016.
- B1's own-L gain is about 4x the released model's (-0.0037, A1c) and about 7x CTRL_base's (-0.0022).
- Molecule-level bootstrap is **not possible locally**: the eval.pkl files are not synced, and only breakdown.log and
  summary.txt are local. Run `tools/paired_compare.py --ref B1_gtL=... --arm B1_gtLc=...` on the cluster.
- **Interpretation caveat (DOES NOT SUPPORT the D3 conclusion as written).** `gtLcycle` gives every GT conformer its
  **own** exact L twice (`sampling.py:73-75`). With random.choice, a given GT L is missed with probability about e^-2 ≈
  13.5%. So the -0.016 mixes two things: (i) L-τ coupling, and (ii) the oracle advantage of having the target's exact
  L among the seeds, which is test-set leakage.
- A joint L|τ sampler cannot reproduce (ii). "Joint or iterative L|τ would recover it" should read "is an upper bound
  on what L|τ coupling could recover (own-L oracle)".

Other BRIEF numbers checked: B1 on RDKit 0.2342 vs 0.1806 (n=934); CTRL_rematch -0.002; B1 gtL vs CTRL gtL -0.0473.
**VERIFIED.**
- Note: the RDKit contrast (+0.054) is on 934 molecules and the GT-L contrast (-0.047) is on 996. The round-1
  dose-response endpoints use different populations.

Jitter check (R2A-4), by simulation of an sp3 angle with 1.5 Å bonds and isotropic per-coordinate σ:

| σ (Å) | bond RMSE (Å) | angle RMSE (°) |
|---|---|---|
| 0.02 | 0.028 | 1.65 |
| 0.025 | 0.035 | 2.06 |
| 0.04 | 0.057 | 3.3 |

- The claim "0.035 Å bonds" is VERIFIED.
- The claim "angles only about 1-2°" is slightly low. It is about 2.1° for C-C and higher for C-H. At the top of the
  training range (0.05 Å), angle error exceeds RDKit's 3.88°.

## 4. Per-ablation verdicts

**R2A-1: APPROVE WITH CHANGES.** Cost arithmetic is right: 48 × 0.25 = 12 GPU-h. Required changes:
- (a) **Source 2 (MMFF) is not built like its control.** `--pre_mmff` acts only on the SMILES path (`generate_confs.py:136-137`),
  not on `--seed_confs`. So source 2 differs from source 1 in graph origin and in population (935 vs 996 molecules).
  Either MMFF-relax the GT-graph ETKDG seeds inside the seed pickle, or use the existing SMILES-path run
  (`steps20_seed0`) as the MMFF control. The same applies to xTB: build it as a seed pickle.
- (b) **Source 6 needs explicit alignment.** The `:108` alignment does not exist. Without it, x_λ is meaningless.
  Linear Cartesian interpolation also shrinks bonds wherever matched and GT torsions differ (e.g. methyl H's). Report
  the bond/angle RMSE of x_λ, so that λ is mapped to measured L error rather than used as a nominal value.
- (c) **λ = 1 is not "the existing A1 run".** One seed per GT conformer, derived from that conformer, is the own-L
  (gtLcycle / A1c) condition. Use `--seed_confs_cycle` for the whole λ series and the existing gtLcycle runs as λ = 1.
- (d) **Label oracles.** Sources 4, 5 and every λ > 0 are TEST-SET ORACLES, because they use test GT L. λ = 0 is also
  oracle-selected: the RDKit seed is chosen by Hungarian assignment on transplant cost to the test GT. Decision rules
  that pick the design from these oracles are analysis only, not evaluation.
- The expected crossing at λ 0.5-0.75 has no supporting evidence. Treat it as a guess.

**R2A-2: APPROVE WITH CHANGES.** Cost is right: 32.1 + 4.5 ≈ 37 GPU-h. Required changes:
- (a) **Explicit alignment** (same bug as R2A-1 source 6).
- (b) **Control at the RDKit end must be CTRL_rematch, not CTRL_base.** The data come from regenerated rematch pickles,
  and `slurm/standardize_qm9.sbatch:17-19` itself requires this.
- (c) **The λ = 1 end is not B1's data.** std pickles are capped at confs_per_mol (`standardize_confs.py:59-67`), while
  B1 uses all raw conformers (`dataset.py:209`). Non-inferiority to B1 therefore confounds λ-conditioning with the
  conformer cap. Either add a B1-capped control or note the confound.
- (d) **GT-L test conditions are ORACLE** and must be labelled as such.
- (e) The arm changes data and architecture together. That is acceptable only because R2A-3 isolates the conditioning.
- Supporting citations E15, E16 and E24 are weak or analogy-only.

**R2A-3: APPROVE WITH CHANGES.** Cost is right: 21.4 + 2 ≈ 23 GPU-h. Required changes:
- (a) CTRL_rematch as the RDKit-end control.
- (b) The same alignment fix if the interpolation code is reused (with Bernoulli λ it is not needed).
- (c) The fallback path concatenates the std and raw lists, which changes the per-molecule GT:RDKit ratio. Cap both
  halves equally, as research_A says.
- (d) "Better than both on xTB L" has no supporting evidence.
- (e) Only 2 seeds against 3-seed controls. Acceptable.

**R2A-4: APPROVE WITH CHANGES.** This is a single-factor change against B1, so the control is correct.
- (a) The cost omits B1 control runs on jittered GT L (σ 0.02/0.04 × 3 seeds = 6 runs, +1.5 GPU-h).
- (b) Jittered-GT test seeds are ORACLE (built from test GT) and must be labelled.
- (c) Correct the angle-error estimate (§3).
- The expected direction is plausible.

**R2A-5: APPROVE WITH CHANGES (pilot).** Cost is consistent: 10.7 + 3 ≈ 14 GPU-h, and training time is an
unmeasured assumption, as research_A states.
- (a) Drop or reframe E20. GO-Flow's ablation shows a Cartesian variant is worse, which argues against a pure-Cartesian
  refiner. Keep E18 as the pro-Cartesian evidence.
- (b) State explicitly that the refiner is trained on the TRAIN split only, and that σ_start is chosen on validation
  (already stated).
- (c) The gate compares refined-seed floors with RDKit/xTB floors computed by `--mode test`, which uses test GT. That is
  fine for analysis, but the floor must not be used to tune σ_start.

**R2A-6 (deferred): APPROVE deferral.**
- The rationale rests on D1's "at most 0.01 Å", which is overstated (§3 item 3). The gate on the R2A-1 source-4 oracle
  is the right fix.
- E12 is about macrocycles, not small rings.

**R2A-7: APPROVE WITH CHANGES.** The control (CTRL_rematch + `--pre_mmff`) is correct and isolates the training change.
- That control run does not exist yet: it needs 3 extra inference runs (+0.75 GPU-h), which are not in the cost.
- Also report against CTRL_rematch without MMFF.

**Totals.**
- 10 training runs and 132 GPU-h: the arithmetic is right (12 + 37 + 23 + 24 + 14 + 22). With the omitted control
  runs it is about 135 GPU-h.
- "~16-18-run round": UNVERIFIABLE from BRIEF.

## 5. Summary

All 28 paper quotes are exact and on the stated pages. The two new PDFs are the claimed papers, and their INDEX rows are
accurate. Problems:
- E20 contradicts the Cartesian refiner it is cited for.
- E12, E15, E16, E24 and E25 are analogies or weak support.

One code claim is WRONG: `standardize_confs.py:108` aligns a RemoveHs copy, so matched conformers are not aligned to GT.
This breaks R2A-1 source 6 and R2A-2 as written.

`--pre_mmff` is ignored on the `--seed_confs` path, so the R2A-1 MMFF source does not share its control.

The D1 numbers reproduce exactly, but:
- "5-ring stratum worst" is wrong (r≥7 is higher).
- The means are conformer-weighted, not molecule-weighted.
- "at most 0.01 Å" contradicts research_A's own caveat.

The D3 numbers reproduce, and all 3 training seeds agree in sign (-0.0159/-0.0170/-0.0143). But the own-L condition is a
test-set oracle that also guarantees L coverage. So -0.016 is an upper bound on what coupling could recover, not an
amount a joint model would recover.

Several conditions need explicit oracle labels. R2A-2 and R2A-3 need CTRL_rematch as the control.
