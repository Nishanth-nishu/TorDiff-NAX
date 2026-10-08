# Verification: VERIFIER V1 on SCOUT EVAL ledger (round 3, P2)

Verifier: V1. Started 2026-10-08. Scope: `round3/ledger/evidence_EVAL.md` (E-EVAL-001 … 054) and cards
`round3/cards/C-EVAL-01 … 06`. Method per `orchestration/roles/verifier.md`: every paper quote checked against the PDF
text (`pdftotext -layout` and raw order, page by page), page 1 title/authors checked, surrounding paragraph read; our-data
values re-read from the files and derived numbers recomputed with my own scripts (scratchpad `v1_*.py`); blog quotes
checked against the live URL (WebFetch) and the local snapshot.

Verdicts: VERIFIED / WRONG LOCATOR / QUOTE MISMATCH / DOES NOT SUPPORT / UNVERIFIABLE. Non-VERIFIED entries have a
thread D-0NN in `ledger/disputes.md`.

## Verdict table
| Entry | Verdict | Dispute | Note |
|---|---|---|---|
| E-EVAL-001 | VERIFIED | — | PDF p. 7 Table 2; raw text order CGCF, GeoDiff, GeoMol, TD, MCF, ET-Flow, ET-Flow-SO(3) gives ET-Flow = 6th numeric row = 96.47/100.00/0.073/0.047/94.05/100.00/0.098/0.039; TD row 92.80/0.178. Same numbers as "ET-Flow (QM9 RS)" in Table 9 p. 19. Caveat for cards: Table 9 also lists ET-Flow (QM9 SS, scaffold split) at AMR-R 0.083, and AvgFlow (p. 7, "ET-Flow-SS (8.3M)" 0.083) and FM-refiner (p. 7, "ET-Flow (8.3M)" 0.083) cite/re-run ET-Flow at 0.083, EnFlow reproduces it at 0.076 (p. 5). 0.073 is the most favourable ET-Flow QM9 number in the literature. |
| E-EVAL-002 | VERIFIED | — | PDF p. 6 Table 1, raw "MCF\n95.0 100.0 0.103 0.044 93.7 100.0 0.119 0.055"; text: "We report our results with base size model (i.e. MCF-B) in Tab. 1"; split "as described in Ganea et al. (2021)" (same as TD). |
| E-EVAL-003 | VERIFIED | — | PDF p. 8 Table 4(b); raw order OMEGA, GeoMol, GeoDiff, TD, MCF-B, DMT-B with numbers in the same order; both quoted rows exact. Caption: baselines copied from Jing 2022 / Corso 2024 / Wang 2024 (OMEGA row is TD's). Protocol: App. D "Following (Wang et al., 2024; Jing et al., 2022), we use the dataset split of ... 106586/13323/1000 for GEOM-QM9, provided by (Ganea et al., 2021)" and 2K generation, so DMT-B is TD-P. |
| E-EVAL-004 | VERIFIED | — | PDF p. 7 Table 1; RDKit, AvgFlowDiT (52M) rows and "Baseline values are taken from the corresponding papers." exact (caption). |
| E-EVAL-005 | VERIFIED | — | PDF p. 16 App. B.2.1 + Table 5; all three quoted fragments exact. KG is defined in Table 1 caption (p. 6 raw) as "number of conformers for each molecule in test split used during training" (ambiguous wording, but it is the per-molecule conformer cap); MCF-B KG 10, TD/ET-Flow KG 30, chirality column as claimed. |
| E-EVAL-006 | VERIFIED | — | PDF p. 16 B.2.1, quote exact. "These models" = S23D-B, MCF-B, ET-Flow (TD's 0.178 is > 0.1). |
| E-EVAL-007 | VERIFIED | — | PDF p. 7 Table 2 caption, exact after δ/Å glyph normalisation. FM-refiner states the TD/GeoMol 1,000-molecule QM9 test set (p. 7 §5.1) — TD-P. |
| E-EVAL-008 | VERIFIED | — | PDF p. 7 Table 2; the three quoted rows exact; DMT (55M) 72.90 also in table; DMT-B + Refiner 79.50 is the highest COV-R mean. Caveat: these baseline AMR-R values differ from the originals (MCF-B 0.101 vs 0.103; ET-Flow 0.083 vs 0.073; DMT 0.087 vs 0.090), i.e. FM-refiner re-ran or used other variants; the δ = 0.05 Å coverages belong to these re-runs, not to the published 0.073-Å ET-Flow. |
| E-EVAL-009 | VERIFIED | — | PDF p. 14 Table 7 caption exact; App. A.4 on p. 15; §5.4 adds "Each conformer is paired only with its own refined counterpart (one-to-one)". |
| E-EVAL-010 | VERIFIED | — | PDF p. 7 Fig. 3 caption exact; GEOM-DRUGS. The entry's side-note "INFERENCE that the same holds on QM9 is supported by our own sweeps, E-EVAL-038/040" is not supported: E-EVAL-038/040 contain no ET-Flow run. The only QM9 low-threshold ET-Flow vs TD contrast is FM-refiner Table 2 vs our TD sweep (different evaluators). Claim itself is fine. |
| E-EVAL-011 | VERIFIED | — | PDF p. 5 Table 1; raw order gives 14 labels and 14 numeric rows (CGCF … EnFlow-SO(3) 5, EnFlow-SO(3) 50); 14th = 50 steps 96.26/0.076/95.48/0.083; EnergySel 50 row 90.91/0.119/96.15/0.069. Mapping confirmed. Caveat for cards: EnergySel draws 3K candidates and keeps the 2K with lowest *learned* energy (table notes) — a selection step, not a relaxation. |
| E-EVAL-012 | VERIFIED | — | PDF p. 3 (printed 3/18), CREST section, exact. Same page: "All geometries accumulated throughout the sampling process are optimized with a tight convergence threshold". |
| E-EVAL-013 | VERIFIED | — | PDF p. 5, quote exact; sub-heading is "Initial structure generation" (inside "Conformer generation"), not "CREST simulation". The quoted sentence alone shows only that the CREST *seed* was xTB-optimised; the conclusion "references are GFN2-xTB minima" holds with p. 3 (E-EVAL-012: MTD on the GFN2-xTB PES, all geometries GFN2-xTB-optimised) plus p. 5 "CREST simulation": "Default values were used for all CREST arguments, except for the charge of each geometry" (vacuum; implicit water only for BACE, pp. 2, 6; DFT re-optimisation (CENSO) only for 534 BACE species, p. 4). I found no DFT step for QM9 anywhere in the paper. GEOM-QM9 references = gas-phase GFN2-xTB minima: confirmed. This also settles C-EVAL-03's open item "gas phase assumed, to be checked in GEOM's Methods". |
| E-EVAL-014 | VERIFIED | — | PDF p. 3, exact; p. 4 repeats "the CREST safety window of 6.0 kcal/mol"; GEOM used CREST defaults (p. 5), so 6.0 applies to GEOM-QM9. |
| E-EVAL-015 | VERIFIED | — | PDF p. 8 Technical Validation, both fragments exact. Context: CENSO (r2scan-3c) on BACE in implicit water (pp. 4, 6); only conformers with ΔG ≤ 2.5 kcal/mol compared; against the most similar CREST geometry the RMSD is 0.33 ± 0.19. The QM9 transfer is correctly labelled INFERENCE. |
| E-EVAL-016 | VERIFIED | — | PDF p. 2, exact. The next sentence (context the cards omit): "Therefore, benchmarks that include conformer probabilities should use the DFT weights provided in GEOM" (DFT weights exist only for BACE). |
| E-EVAL-017 | VERIFIED | — | PDF p. 7 Technical Validation, both fragments exact. 100 − 88.4 = 11.6 % (used in C-EVAL-03) is consistent with "All of the failed QM9 graphs underwent some sort of reaction". The INFERENCE (failed-graph molecules overlap our 60 ETKDG cage failures) is not tested anywhere. Molecules whose graph changed are largely removed by TD's SMILES filter, while our 60 cages are in the test set with valid references. |
| E-EVAL-018 | VERIFIED | — | PDF p. 10, exact. Table 2 (p. 13) "Mean E MMFF relax" for GEOM-Drugs = 16.4 ± 0.2, consistent. |
| E-EVAL-019 | VERIFIED | — | PDF p. 11, exact. Context: "how close a generated structure is to the closest local minima of the given energy function" (own minimum, not GT). |
| E-EVAL-020 | VERIFIED | — | Quote PDF p. 12, exact. Table 2 p. 13: raw-text column order GEOM-Drugs, MMFF→GFN2-xTB, EQGAT-diff, JODO, Megalodon, SemlaFlow, FlowMol2, Megalodon-flow; MMFF→xTB = 1.12 (×10⁻² Å), 1.22°, 4.89°, median E_relax 9.84, mean 11.4 kcal/mol, confirmed. Caveat: the diffusion models' torsion deviations (EQGAT 8.58°, JODO 6.01°, Megalodon 5.58°) are *worse* than MMFF's 4.89°, so "surpass MMFF" holds for bonds, angles and E_relax only. C-EVAL-03 §2 "diffusion models already beat MMFF on these" overstates the torsion part. |
| E-EVAL-021 | VERIFIED | — | PDF p. 3 Introduction, exact. |
| E-EVAL-022 | VERIFIED | — | PDF p. 27 Table 10; raw order RDKit, OMEGA, GeoMol, GeoDiff, TD; TD 36.91/0.92/4.93/36.94 → 0.22/0.35/0.54/0.13; RDKit E_min 39.14, OMEGA 16.45. App. H (p. 28) says these are "median absolute errors ... of the generated vs CREST ensembles". Note on the INFERENCE: GeoMol, which predicts its own local structure, is worse (43.68) than RDKit, so "dominated by frozen RDKit local structure" is not uniquely supported. |
| E-EVAL-023 | DOES NOT SUPPORT (partial) | D-001 | Quote PDF p. 28, exact. The claim says that after relaxation "the errors from global flexibility *dominate*"; the source says they "become important". First half supported, second half overstated. |
| E-EVAL-024 | VERIFIED | — | PDF p. 9 §4.4, exact. Note: it does not by itself show that TD's RMSD tables are unrelaxed (C-EVAL-05 §2 uses it that way). That follows from the code (no relaxation unless `--post_mmff`), not from this quote. |
| E-EVAL-025 | VERIFIED | — | PDF p. 4 §3, both fragments exact. |
| E-EVAL-026 | VERIFIED | — | PDF p. 5 Table 1, both rows exact; p. 4: 200 test molecules, 50–500 conformers each, GD-P. Correctly labelled not comparable with TD-P. |
| E-EVAL-027 | VERIFIED | — | PDF p. 5 Table 2, exact. Note: removing the energy sampler *raises* COV (98.01 vs 97.65) while worsening MAT, so MMFF buys accuracy, not coverage. |
| E-EVAL-028 | VERIFIED | — | PDF p. 5, quote exact (title and authors match arXiv 2304.10494; the PDF carries no arXiv stamp). The side note is wrong in part: Zhang Table 2 (p. 9) does not simply reproduce Zhou's QM9 numbers. RDKit 81.82/0.3027 (Zhou 83.26/0.3447) and CGCF 83.48/0.2984 (Zhou 78.05/0.4219) differ; "Inf. Phy. Monkey (RDKit+Clustering)" 97.65/0.1902 matches. No card relies on the side note. |
| E-EVAL-029 | DOES NOT SUPPORT (partial) | D-002 | Live URL fetched 2026-10-08 (curl + WebFetch): quote exact in the live page and in the snapshot; heading and date (datePublished 2026-06-17) correct; meta author Corin Wagen (visible byline: Corin Wagen, Nick Casetti, Jonathon Vandezande, Eli Mann). But the claim drops the source's scope: missing and high-energy conformers are reported "for large systems (e.g. PROTACs)", not for ETKDG in general. Only the twist-boat remark is unscoped (anecdote, no numbers). For QM9 (≤ 9 heavy atoms) the scope matters. C-EVAL-04's "oversampling only partly fixes" is not in the source ("can be ameliorated ... but slow ... replicas converge on the same few conformers"). |
| E-EVAL-030 | VERIFIED | — | Live raw ReleaseNotes.md (curl, 479,359 bytes) line 2122 under "# Release_2024.03.1" / "## Backwards incompatible changes": identical to the snapshot and the quote, incl. "EmbedMultipleConfis" and "ETKDGV1". The same bullet also changes pruning RMSD to heavy atoms only. |
| E-EVAL-031 | VERIFIED | — | generate_confs.py:64-68 exact (call at :67-68); RESULTS_QM9.md:9 "RDKit 2022.9.5"; notes/code_walkthrough.md:219 also lists rdkit 2022.9.5. Every other seed path also embeds without a params object (tools/make_l_seed_pickles.py:89, :169, :173; standardize_confs.py:79; tools/local_structure_analysis.py:260), so all our L (test and training-matched) is default ETKDG = v1 on 2022.9.5 per E-EVAL-030. The INFERENCE label is appropriate. |
| E-EVAL-032 | QUOTE MISMATCH | D-003 | Live api.anaconda.org JSON (curl 2026-10-08): latest_version 6.7.1; linux-64 files (all label `main`) exist for 6.2.3, 6.3.0, 6.3.1, 6.3.2, 6.3.3, 6.4.0, 6.4.1, 6.5.0, 6.5.1, 6.6.0, 6.6.1, 6.7.1. The "quote" (a scout-derived list starting at 6.4.1) omits six versions. The claim (linux-64 builds, latest 6.7.1) holds. Useful extra: GEOM used xTB 6.2.3 with CREST 2.9 (GEOM p. 8, Code availability), and 6.2.3 is on conda-forge, so the reference xTB version can be matched exactly. |
| E-EVAL-033 | WRONG LOCATOR | D-004 | The quoted `_cr = [...] + [0] * num_failures` is torsional-diffusion/evaluate_confs.py:166 (not within 168-171); `'MAT-R_mean': float(np.nanmean(amr_recall))` is :171. Correct locator :166-171 (the printed block with the same convention is :152-155). Behaviour as claimed: failures add 0 to COV, AMR uses nanmean over evaluated molecules (NaN "additional failures" are dropped too). |
| E-EVAL-034 | VERIFIED | — | torsional-diffusion/utils/xtb.py:43-63 exact (def at :43, cmd at :49). Notes: no `--gfn`/charge/solvent flag (xtb default GFN2, gas phase, which matches GEOM-QM9); it updates only the default conformer; it uses a per-process /tmp dir. |
| E-EVAL-035 | VERIFIED | — | sampling.py:249-262 (`if mmff:` :258, MMFFOptimizeMoleculeConfs MMFF94s :260); generate_confs.py:19 (`--post_mmff`) and :170 (`pyg_to_mol(mol, conf, args.post_mmff, ...)`). |
| E-EVAL-036 | VERIFIED | — | tools/geometry_metrics.py:2 and :13 exact; slurm/analysis_qm9.sbatch:65 calls it; `find cluster_sync -iname "geometry*"` gives no hits (re-run 2026-10-08); no RESULTS or round-2 file reports its output. |
| E-EVAL-037 | VERIFIED | — | tools/paired_compare.py:32, :41, :105-106 exact. |
| E-EVAL-038 | VERIFIED | — | all_summaries.txt:192 (SUMMARY) and :189-190 (SWEEP) exact. |
| E-EVAL-039 | VERIFIED | — | all_summaries.txt:112 and :109 exact. |
| E-EVAL-040 | VERIFIED | — | all_summaries.txt:164, :153, :154 exact. |
| E-EVAL-041 | VERIFIED | — | all_summaries.txt:176 exact; R0 AMR-P 0.2191 at :192. |
| E-EVAL-042 | VERIFIED | — | all_summaries.txt:144, :133 exact. Note: A1 is on 996 molecules (4 failures) vs R0 on 935, so the populations differ (RESULTS_QM9.md §2.3 caveat). |
| E-EVAL-043 | VERIFIED | — | Both summary.txt files exact (S3 on RDKit L: 935 molecules; gtLcycle: 996). |
| E-EVAL-044 | VERIFIED | — | All four summary.txt files exact, incl. n_evaluated 906/94 and 955/45. The population files show λ = 0.75 keeps 906 (49 extra drops, "no safe pair") and λ = 1.00 keeps 955. λ = 1.00 exists only for B1 seed 0 in the sync. |
| E-EVAL-045 | VERIFIED | — | all_summaries.txt:7, :11, :67, :71 exact; all on 996 molecules. |
| E-EVAL-046 | VERIFIED | — | RESULTS_QM9.md:49 and :53 (§2.2) exact; :50 gives the 5 rejected inputs; :52 the 88.8 %. |
| E-EVAL-047 | VERIFIED | — | local_structure_test.log:17, :52-55 exact. Caveat for cards: these ETKDG errors (0.036 Å / 3.88°) come from the round-1 seed set and differ from the round-2 l_error.csv ETKDG values on the same heavy-atom basis (0.0314 Å / 2.80°). Do not juxtapose them with l_error.csv MMFF values. |
| E-EVAL-048 | DOES NOT SUPPORT (inference) | D-005 | The numbers reproduce exactly with my own script (`v1_macro.py`) once the 57 embed-failure rows are excluded (940 molecules): bins 225/227/…/213, n_gt ≤ 3 = 48.09 % of molecules and 10.18 % of GT conformers, ring 98.2 % / 97.4 %, floor 0.1364/0.1315 vs 0.0888. But the entry's INFERENCE ("about half determined by rigid, ring-containing molecules where the torsion model has little to do") and the cards' "rigid ring molecules … 48 %" are not supported. Only 52 % of the n_gt ≤ 3 molecules have 0 heavy rotatable torsions; rigid molecules (0 heavy torsions) are 30.4 % of molecules. In CTRL_rematch s0 on ETKDG L (breakdown.log, `v1_breakdown.py`), n_true ≤ 3 is 48.1 % of molecules and 46 % of the AMR-R sum, while rigid (n_rot_heavy = 0) is 30.4 % of molecules and 26 % of the AMR-R sum. n_true = 1 molecules have the *lowest* AMR-R (0.149 vs 0.181–0.190), so the high floor does not make them the largest contributors. |
| E-EVAL-049 | VERIFIED | — | Recomputed with `v1_lerr.py`: row means ETKDG 0.0314/2.7963/10.6522, MMFF 0.0182/1.9733/8.0655, λ0.75 0.0097/0.9668/1.5643, λ0.50 0.0186/1.9332/3.1561 (ring dihedral over the ~7.3k ring-containing rows). The same to ±0.0003 on the 888 molecules common to λ0.75/λ1.00/ETKDG/MMFF. Per-molecule means are higher (ETKDG 0.0360/3.42/11.77; λ0.75 0.0110/0.98/1.87). |
| E-EVAL-050 | VERIFIED | — | Both summary.txt files exact (936 molecules, 64 failures). |
| E-EVAL-051 | VERIFIED | — | PDF p. 6 Discussion, exact; the scope is "for quantum property prediction". |
| E-EVAL-052 | VERIFIED | — | review/metric_verification.md:140 B3 row exact (the row continues with the 28.8° → 0.0° test); `[metric-verifier fix]` comments at tools/geometry_metrics.py:57 and :104. |
| E-EVAL-053 | VERIFIED | — | RESULTS_QM9.md:68-69 exact. |
| E-EVAL-054 | VERIFIED | — | ET-Flow PDF p. 7 §4.4, exact ("a minimum of 2K and a maximum of 32 conformers per molecule"). |

**Counts:** VERIFIED 49; DOES NOT SUPPORT 3 (023, 029, 048; all partial: wording, scope, inference); QUOTE MISMATCH 1 (032);
WRONG LOCATOR 1 (033); UNVERIFIABLE 0.

## Special checks requested by the orchestrator
1. **GEOM-QM9 references = GFN2-xTB minima.** Confirmed from the GEOM PDF itself (arXiv 2006.05531v4). p. 3: MTD on the
   GFN2-xTB PES; "Geometries from the MTD runs are then optimized with GFN2-xTB"; all accumulated geometries are
   re-optimised with a tight threshold. p. 5: QM9 DFT geometries are re-optimised with xTB only to seed CREST; "Default
   values were used for all CREST arguments, except for the charge" (gas phase). DFT appears only for BACE (single
   points for 1,511 species, CENSO re-optimisation for 534; pp. 1, 4–6). xTB 6.2.3 + CREST 2.9 (p. 8). No QM9 DFT step
   anywhere. E-EVAL-012/013 VERIFIED.
2. **Literature AMR-R / COV@0.05 (ET-Flow, DMT, MCF).** All QM9 numbers in the ledger come from QM9 tables at δ = 0.5 Å
   under TD-P (Ganea split, 1000 test molecules, 2K): ET-Flow Table 2 (p. 7); MCF Table 1 (p. 6, "split as described in
   Ganea et al."); DMT NExT-Mol Table 4(b) (p. 8; App. D states the 106586/13323/1000 split); FM-refiner Table 2 (p. 7,
   same 1,000 molecules) for δ = 0.05 Å. GD-P numbers (Zhou, Zhang: 200 molecules) are labelled non-comparable in
   E-EVAL-026 and C-EVAL-04 §4. Caveat (D-006): ET-Flow's 0.073 is its best variant/split; three other papers use or
   re-run ET-Flow at 0.076–0.083, and the δ = 0.05 Å ET-Flow coverage (75.72 %) belongs to a run with AMR-R 0.083.
   FM-refiner's MCF-B (0.101) and DMT (0.087) are also re-runs, not the published 0.103/0.090.
3. **λ≈0.86.** Recomputed (`v1_lambda.py`, from the round-2 summary.txt files): 0.8616 (B1 s0, linear 0.75→1.00);
   0.8599 with the 3-seed λ 0.75 mean; 0.851–0.863 across population bounds (906 vs 955 molecules); 0.872 with a
   quadratic through λ 0.5/0.75/1.0. Correct and robust. The target choice moves it more (0.836 for 0.083 Å). The L error
   at λ 0.75 (0.0097 Å / 0.97° / 1.56°) reproduces from l_error.csv (`v1_lerr.py`). Per-molecule eval tables are not
   synced, so an exact paired intersection recompute was not possible locally; the bound above replaces it.
4. **"Rigid ring molecules are 48 % of the per-molecule average."** Not supported (D-005). 48.1 % are *few-conformer*
   (n_gt ≤ 3) molecules; 97–98 % of them contain a ring, but only 52 % of them are rigid. Rigid molecules are 30.4 % of
   molecules and carry 26 % of the macro AMR-R sum in CTRL_rematch s0 on ETKDG L. Few-conformer molecules carry 46 %.
5. **Blog snapshots vs live URLs.** Rowan: live = snapshot, quote exact, scope dropped in the claim (D-002). RDKit release
   notes: live = snapshot = quote. conda-forge xtb: snapshot list incomplete vs live JSON (D-003).

## Card rulings

### C-EVAL-01: SUPPORTED (corrections needed, D-006)
Every literature and our-data number used rests on VERIFIED entries (001–008, 033, 037, 038, 040, 042–047, 049, 050), and
the λ≈0.86 inference reproduces. Unsupported or mis-sourced sentences:
- §2 INFERENCE "reference quality and seed availability are both weakest on the subset TD drops from AMR": no evidence
  links GEOM's reacted QM9 graphs (E-EVAL-017) to our 60 cage failures.
- §3/§2 mixes ET-Flow 0.073 Å (E-EVAL-001) with the δ = 0.05 Å ET-Flow coverage from a run whose AMR-R is 0.083
  (E-EVAL-008). The "ET-Flow" target should be stated as a range 0.073–0.083.
- §4 "our AMR-R is therefore flattered relative to ET-Flow's": direction not established. MCF/S23D score 995 molecules,
  and ET-Flow reports RDKit-related failures on GEOM-XL (ET-Flow p. 19).
- §5 "B1/S3 need roughly λ ≥ 0.85-level L": true for B1 (0.86); S3 has no λ sweep, so S3 is unsupported.

### C-EVAL-02: WEAKENED (D-005, D-007)
The diagnostics themselves rest on VERIFIED entries (009, 019, 036, 037, 045, 047, 052). The card's lead motivation
does not:
- §3 "Our headline metric is mostly a ring-L metric" plus "48 % of the macro average" rests on the rejected E-EVAL-048
  inference. The few-conformer share is 48 %, but rigid molecules are 30 % of molecules and 26 % of the AMR-R sum. A
  ring-L argument can be rebuilt from A5ring (rigid molecules 0.152 → 0.043 Å with true ring geometry), not from 048.
- §3 compares ETKDG L from local_structure_test.log (0.036 Å / 3.88°) with MMFF L from l_error.csv (0.0182 Å / 1.97°).
  On the same file ETKDG is 0.0314 Å / 2.80° (D-007.1).
- §2 "because ensemble means hide whether a change helps most conformers a little or a few a lot [E-EVAL-009]" is the
  scout's rationale, not FM-refiner's (it says only that it tests "at a micro level").

### C-EVAL-03: SUPPORTED (minor fixes)
Core premise verified: GEOM-QM9 references are gas-phase GFN2-xTB minima (012, 013); xTB re-optimisation of xTB minima
gives E_relax ≈ 0 (018); own-minimum displacement metric (019); MMFF unsuitable (021); TD has an xTB wrapper (034); xtb
is on conda-forge (032 claim holds). The open item "gas phase assumed, to be checked in GEOM's Methods" is now settled:
CREST defaults except charge, i.e. gas phase (GEOM p. 5). Fixes:
- §2 "diffusion models already beat MMFF on these": not for torsions (Nikitin Table 2: 5.58–8.58° vs 4.89°) (D-007.2).
- §3 "can be computed for test molecules where ETKDG fails": for TD-family arms there is no generated structure when
  ETKDG fails. This holds only for non-ETKDG L sources.
- §5 cost: "≈ 40k optimisations" matches about 8 arms × 5k on the 200 test molecules only. Adding the 200 validation
  molecules roughly doubles it. The 1–3 s per optimisation is still UNVERIFIED (flagged by the scout).
- E-EVAL-023 overstatement (D-001) does not affect this card's wording ("far too large" is quoted correctly).

### C-EVAL-04: SUPPORTED (minor fixes, D-002)
The recipe and ablation (025–028), literature rows (003, 004), reference level (012, 013, 021), seed generator (030,
031), and our arms (038–040, 046, 049, 050) are VERIFIED. I checked that no ETKDG+MMFF no-model arm exists (only A0,
A0-rand, A2, A3), so the "B-mmff missing" premise holds. Fixes:
- §2 "ETKDG ring pathologies ... that oversampling only partly fixes [E-EVAL-029]": the source says "can be
  ameliorated"; the miss/high-energy remark is for large systems (D-002).
- §4 "B-clust is a recall-gaming baseline by design [E-EVAL-027]": a characterisation, not shown by 027. Removing MMFF
  *raises* COV in Zhou's ablation.
- §3 "about half of the test molecules have ≤ 3 GT conformers and are ring-containing" is numerically correct (48 %,
  97–98 % ring) but must not be read as "rigid" (D-005).
- Arithmetic checked: 30K draws = 15 × 2K; K = 13,731/1000 = 13.73 (RESULTS_QM9.md:7); 3 × 1000 × 27.46 ≈ 82k.

### C-EVAL-05: WEAKENED (D-005, D-007.3–4)
The core premise (xTB-after puts samples on the reference PES: 012, 013, 018; hook exists: 034, 035; A3 result: 041,
053) is VERIFIED. Three supporting sentences are not:
- §3 "Rigid ring molecules dominate the macro average [E-EVAL-048]; for them xTB-after acts only on L (no torsions to
  move)": wrong for about half of the few-conformer group. Rigid molecules are 30 % of molecules and 26 % of the AMR-R sum.
- §2 and §4 use EnFlow EnergySel (E-EVAL-011) as evidence that energy post-processing or relaxation loses recall. It is
  learned-energy *selection* (3K → 2K), not relaxation.
- §2 "TD relaxes only in its ensemble-property test, not for its RMSD tables [E-EVAL-024]": not shown by 024 (true per
  the code, E-EVAL-035).

### C-EVAL-06: WEAKENED (part (b) rationale; D-001)
Part (a), no DFT re-referencing, is SUPPORTED: references are xTB minima (012, 013); published numbers all use them
(001–008); the xTB→DFT shift is measured only for BACE in water (015); Zhou's DFT argument is an opinion (051). Part (b)'s
key sentence "the metric is by construction insensitive to the thing FlexiTors changes" rests on the overstated
E-EVAL-023 reading ("dominate"). Local xTB relaxation does not erase ring-pucker or torsion-basin choices, which are what a
FlexiTors L changes. Also omitted: GEOM's own recommendation, in the sentence after E-EVAL-016, to use its DFT weights for
probability benchmarks (available only for BACE). The NO call may still be right on cost and comparability grounds, but
the "by construction" argument should go.

## Summary
54 entries: 49 VERIFIED, 3 DOES NOT SUPPORT (partial), 1 QUOTE MISMATCH, 1 WRONG LOCATOR, 0 UNVERIFIABLE. Disputes
D-001 … D-005 (entries) and D-006, D-007 (card-level). Cards: C-EVAL-01 SUPPORTED, C-EVAL-02 WEAKENED, C-EVAL-03
SUPPORTED, C-EVAL-04 SUPPORTED, C-EVAL-05 WEAKENED, C-EVAL-06 WEAKENED. No card loses its core mechanism. Scripts:
scratchpad `v1/v1_lerr.py`, `v1/v1_lambda.py`, `v1/v1_macro.py`, `v1/v1_breakdown.py`; page extractions
`v1/*.layout.txt`, `v1/*.raw.txt`.
