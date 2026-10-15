# Evidence ledger: SCOUT BENCH (lens: what a publishable evaluation requires; brief Q3 and Q4)

Agent tag BENCH. Round 4, P1. Started 2026-10-11. Template: `orchestration/templates/evidence_entry.md`.

Conventions
- Paper quotes checked against `pdftotext -layout -f N -l N <pdf>` (PDF page N, 1-based); printed page given when it differs.
- `[δ]` stands for the Greek delta lost in extraction.
- Entries copied from round 3 carry "Reused from: round3/ledger/evidence_EVAL.md E-EVAL-NNN (VERIFIED by V1/V2)".
- Repository, README, data-listing and GitHub-issue snapshots are in `papers/blogs/` with URL and access date at the top
  (OpenReview could not be read; section Z). Multi-fragment quotes are joined by "…" and total ≤ 40 words.
- Status: P1 complete 2026-10-12; 73 entries (E-BENCH-001…073), cards C-BENCH-01…07.
- Computations are marked [BENCH-calc]; scripts in the session scratchpad
  (`C:\Users\HP\AppData\Local\Temp\claude\C--Users-HP-Desktop-tor-diff\ebb0c292-63dd-41c3-ab3b-16fa6f4d7fcc\scratchpad\bench\`).

## A. Protocols, splits, sample counts, thresholds (QM9 and DRUGS)
### E-BENCH-001
- Claim: The TD protocol (TD-P) uses GeoMol's random splits, 243473/30433/1000 molecules for GEOM-DRUGS and 106586/13323/1000 for GEOM-QM9, and filters out molecules whose CREST conformers all have a SMILES different from the molecule's or that RDKit cannot handle.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 23, App. G.1 "Dataset details"
- Exact quote:
  > "train/validation/test of 243473/30433/1000 for GEOM-DRUGS and 106586/13323/1000 for GEOM-QM9." … "molecules whose CREST conformers all have a canonical SMILES different from the SMILES of the molecule" … "or that cannot be handled by RDKit, are filtered out."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01, C-BENCH-05
- Confidence: high

### E-BENCH-002
- Claim: TD-P generates 2K conformers for a molecule with K ground-truth conformers and uses δ = 0.75 Å on GEOM-DRUGS (most earlier work, the GeoDiff/ConfGF "GD-P" protocol, used 1.25 Å).
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 8, Table 1 caption and Sec. 4.2 "Evaluation"
- Exact quote:
  > "We compute Coverage with a threshold of [δ] = 0.75 Å to better distinguish top methods." … "most prior works, which used [δ] = 1.25 Å." … "we generate 2K conformers for a molecule with K ground truth conformers."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01, C-BENCH-05, C-BENCH-07
- Confidence: high

### E-BENCH-003
- Claim: TD trains on at most the first 30 CREST conformers per molecule (K_G ≤ 30) but at test time generates twice the number of all CREST conformers.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 17, App. C "Training and inference procedures"
- Exact quote:
  > "during training, we limit KG to 30 i.e. we only consider the first 30 conformers found by CREST (typically those with the largest Boltzmann weight)." … "set K to be twice the number of conformers returned by CREST."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-004
- Claim: GeoMol (origin of the splits) used δ = 0.5 Å for GEOM-QM9 and δ = 1.25 Å for GEOM-DRUGS; TD changed only the DRUGS threshold, so QM9 at 0.5 Å is common to both conventions.
- Source type: paper
- Locator:
  - paper: papers/related/2021_ganea_geomol.pdf, PDF p. 9, Sec. 5 (evaluation setup)
- Exact quote:
  > "set [δ] = 0.5Å for GEOM-QM9 and [δ] = 1.25Å for GEOM-DRUGS."
- Shows vs infers: SHOWS. INFERENCE: GeoMol's own DRUGS numbers (δ = 1.25 Å) must not be mixed into TD-P DRUGS tables (δ = 0.75 Å); TD-P tables use TD's re-evaluation of GeoMol (E-BENCH-013).
- Supports card(s): C-BENCH-01, C-BENCH-05, C-BENCH-07
- Confidence: high

### E-BENCH-005
- Claim: S23D's DRUGS table records two comparability factors that differ between published models: training conformers per molecule (KG = 20 for MCF and S23D, 30 for TD and ET-Flow) and whether chirality is used or corrected; S23D-B with chirality reaches COV-R 87.0 % / AMR-R 0.380 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 8, Table 1 (columns SIZE, KG, CHIRAL.)
- Exact quote:
  > "`KG' is number of conformers for each molecule in test split used during training. `Chirality' indicates whether chirality information was used" … "TOR. DIFF. 30 YES 72.7 80.0 0.582 0.565" … "S23D-B-1/13 (M) 25 20 YES 87.0 94.0 0.380 0.356"
- Shows vs infers: SHOWS ("test split" is the source's wording; the column is the training cap). INFERENCE: a fair comparison table should state KG, chirality handling and model size per row.
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-006
- Claim: ET-Flow's current preprocessing keeps the top-30 training conformers by Boltzmann weight (not CREST order), takes test references from TD's `test_mols.pkl`, and drops multi-fragment molecules.
- Source type: code (third-party repo)
- Locator:
  - code: github.com/shenoynikhil/ETFlow `scripts/prepare_data.py`:101, :128-131, :153-157, :237-248 (snapshot papers/blogs/etflow_eval_code.txt, accessed 2026-10-11)
- Exact quote:
  > "keep_top_n: int = 30," … "confs = sorted(confs, key=lambda x: x["boltzmannweight"], reverse=True)" / "confs = confs[:keep_top_n]  # Keep top 30 conformers" … "data_dir / partition.upper() / "test_mols.pkl""
- Shows vs infers: SHOWS (current main branch; the README says preprocessing changed after the paper, E-BENCH-027).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-007
- Claim: Our (TD's) evaluator counts a molecule with no generated conformers as 0 % coverage but leaves it out of AMR; it also drops ground-truth conformers whose heavy-atom SMILES differs from the molecule's (`clean_confs`) and skips molecules left with no clean conformer.
- Source type: code
- Locator:
  - code: torsional-diffusion/evaluate_confs.py:47-54 (`clean_confs`), :81-90 (model failure; poor ground truth), :166-171 (COV with `[0] * num_failures`; `np.nanmean`)
- Exact quote:
  > "if conf_smi == smi:" / "good_ids.append(i)" … "print('model failure', corrected_smi)" / "num_failures += 1" … "print(f'poor ground truth conformers: {corrected_smi}')"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-033 (COV/AMR failure convention, VERIFIED by V1 after D-004); the `clean_confs` part is new.
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-008
- Claim: ET-Flow's evaluator skips a molecule entirely (no zero coverage) when generated or reference positions are missing, when its SMILES has several fragments, or when fewer than 2K conformers were generated; its COV and AMR are averaged over the molecules that remain.
- Source type: code (third-party repo)
- Locator:
  - code: github.com/shenoynikhil/ETFlow `etflow/commons/covmat.py`:135-140, :151-156, :196-200 (snapshot papers/blogs/etflow_eval_code.txt, accessed 2026-10-11)
- Exact quote:
  > "log.info("skipping due to missing pos_gen or pos_ref")" / "continue" … "log.info("skipping due to disconnected molecule")" … "log.info("skipping due to insufficient number of generated conformers")"
- Shows vs infers: SHOWS. INFERENCE: TD-evaluator and ET-Flow-evaluator coverage numbers differ in failure handling; a model that loses molecules to RDKit (TD and all our arms, 65/1000 on QM9, E-BENCH-011) is penalised in COV by the TD evaluator but would not be by ET-Flow's. Published tables mix the two conventions without saying so.
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-009
- Claim: ET-Flow's evaluator uses heavy-atom, symmetry-aware `GetBestRMS` by default (NaN on an RDKit exception), scores thresholds 0.05–3.0 Å in 0.05 steps, generates 2 × (number of stored references), and has no `clean_confs` filter on the references.
- Source type: code (third-party repo)
- Locator:
  - code: github.com/shenoynikhil/ETFlow `etflow/commons/covmat.py`:57-69, :113; `scripts/eval.py`:77-79 (snapshot papers/blogs/etflow_eval_code.txt)
- Exact quote:
  > "probe = RemoveHs(probe)" … "rmsd = MA.GetBestRMS(probe, ref)" / "except:  # noqa" / "rmsd = np.nan" … "thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05)," … "num_samples = 2 * count"
- Shows vs infers: SHOWS. INFERENCE: the RMSD definition matches TD's (heavy atoms, GetBestRMS, E-BENCH-007), but the reference set can differ where GEOM conformers reacted (TD's evaluator filters them). Scoring every method's generated files with one evaluator is the only safe comparison (C-BENCH-02).
- Supports card(s): C-BENCH-02
- Confidence: high (code); medium (size of the reference-set difference, not measured)

### E-BENCH-010
- Claim: MCF and S23D evaluated GEOM-QM9 on 995 test molecules (MCF's processed set), not 1000.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16, App. B.2.1
- Exact quote:
  > "The processed test dataset contained 995 molecules rather than the 1000 indexed in the test set"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-005 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-011
- Claim: In our TD-protocol QM9 runs, 65 of 1000 test molecules produce no TD conformers (60 strained cages ETKDG cannot embed, 5 rejected inputs), and the paper's own released QM9 samples scored with the repo evaluator give COV-R 88.8 % vs the published 92.8 %.
- Source type: our-data
- Locator:
  - our-data: RESULTS_QM9.md, §2.2 "Coverage is lower than the paper because of RDKit failures"
- Exact quote:
  > "**60 molecules:** RDKit's ETKDG cannot embed strained polycyclic cages" … "**So the paper's 92.8% cannot be reproduced from its own released samples.**"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-046 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

## B. Results tables we must compare against
### E-BENCH-012
- Claim: TD's GEOM-QM9 table (TD-P, δ = 0.5 Å) reports COV-R / AMR-R means: RDKit 85.1 % / 0.235 Å, OMEGA 85.5 / 0.177, GeoMol 91.5 / 0.225, GeoDiff 76.5 / 0.297, Torsional diffusion 92.8 / 0.178 (AMR-P 0.221).
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 26, Table 7
- Exact quote:
  > "Table 7: Performance of various methods on the GEOM-QM9 dataset test-set ([δ] = 0.5Å)." … "RDKit 85.1 100.0 0.235 0.199" … "OMEGA 85.5 100.0 0.177 0.126" … "Torsional diffusion 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01, C-BENCH-02, C-BENCH-06
- Confidence: high

### E-BENCH-013
- Claim: TD's GEOM-DRUGS table (TD-P, δ = 0.75 Å) reports COV-R / AMR-R / COV-P / AMR-P means: RDKit ETKDG 38.4 / 1.058 / 40.9 / 0.995, OMEGA 53.4 / 0.841 / 40.5 / 0.946, GeoMol 44.6 / 0.875 / 43.0 / 0.928, GeoDiff 42.1 / 0.835 / 24.9 / 1.136, Torsional Diffusion 72.7 / 0.582 / 55.2 / 0.778.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 8, Table 1
- Exact quote:
  > "RDKit ETKDG 38.4 28.6 1.058 1.002 40.9 30.8 0.995 0.895" … "OMEGA 53.4 54.6 0.841 0.762 40.5 33.3 0.946 0.854" … "Torsional Diffusion 72.7 80.0 0.582 0.565 55.2 56.9 0.778 0.729"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-02, C-BENCH-05, C-BENCH-06
- Confidence: high

### E-BENCH-014
- Claim: ET-Flow's GEOM-DRUGS table (δ = 0.75 Å, 50 steps) reports COV-R / AMR-R / COV-P / AMR-P means: ET-Flow 79.53 / 0.452 / 74.38 / 0.541, ET-Flow-SS 79.62 / 0.439 / 75.19 / 0.517, ET-Flow-SO(3) 78.18 / 0.480 / 67.27 / 0.637; it copies MCF-S 79.4 / 0.512, MCF-B 84.0 / 0.427, MCF-L 84.7 / 0.390.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 6, Table 1 (row labels and numbers are on separate lines in the extraction; label order GeoDiff, GeoMol, Torsional Diff., MCF-S, MCF-B, MCF-L, ET-Flow, ET-Flow-SS, ET-Flow-SO(3) matched to the numeric rows in order; the first three numeric rows equal TD Table 1, E-BENCH-013, which confirms the mapping)
- Exact quote:
  > "Table 1: Molecule conformer generation results on GEOM-DRUGS ( [δ] = 0.75Å)." … "79.53 84.57 0.452 0.419 74.38 81.04 0.541 0.470" … "79.62 84.63 0.439 0.406 75.19 81.66 0.517 0.442" … "84.7 92.2 0.390 0.247 66.8 71.3 0.618 0.530"
- Shows vs infers: SHOWS (row mapping medium-high confidence; MCF rows also verified against MCF Table 2, E-BENCH-016)
- Supports card(s): C-BENCH-02, C-BENCH-05, C-BENCH-06
- Confidence: high

### E-BENCH-015
- Claim: ET-Flow reports on GEOM-QM9 (δ = 0.5 Å) COV-R 96.47 %, AMR-R 0.073 Å, COV-P 94.05 %, AMR-P 0.098 Å; TD row 92.80 / 0.178.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 7, Table 2
- Exact quote:
  > "96.47 100.00 0.073 0.047 94.05 100.00 0.098 0.039"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-001 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02, C-BENCH-06
- Confidence: high

### E-BENCH-016
- Claim: MCF's own GEOM-DRUGS table gives COV-R / AMR-R / COV-P / AMR-P means MCF-S 79.4 / 0.512 / 57.4 / 0.761, MCF-B 84.0 / 0.427 / 64.0 / 0.667, MCF-L 84.7 / 0.390 / 66.8 / 0.618, and its QM9 table MCF 95.0 / 0.103 / 93.7 / 0.119.
- Source type: paper
- Locator:
  - paper: papers/related/2023_wang_mcf.pdf, PDF p. 6, Table 1 (QM9) and Table 2 (DRUGS) (side by side in the layout extraction)
- Exact quote:
  > "MCF-S 79.4 87.5 0.512 0.492 57.4 57.6 0.761 0.715" … "MCF 95.0 100.0 0.103 0.044 93.7 100.0 0.119 0.055 MCF-B 84.0 91.5 0.427 0.402 64.0 66.2 0.667 0.605" … "MCF-L 84.7 92.2 0.390 0.247 66.8 71.3 0.618 0.530"
- Shows vs infers: SHOWS
- Reused from (QM9 row): round3/ledger/evidence_EVAL.md E-EVAL-002 (VERIFIED by V1); DRUGS rows new.
- Supports card(s): C-BENCH-02, C-BENCH-05, C-BENCH-06
- Confidence: high

### E-BENCH-017
- Claim: NExT-Mol's GEOM-DRUGS table reports DMT-B 85.4 / 0.401 and DMT-L 85.8 / 0.375 (COV-R % / AMR-R Å), TD with particle guidance 77.0 / 0.543 as published but 73.8 / 0.566 when re-run with the authors' code; on GEOM-QM9 DMT-B 95.2 / 0.090.
- Source type: paper
- Locator:
  - paper: papers/related/2025_liu_nextmol.pdf, PDF p. 8, Table 4(a) DRUGS and 4(b) QM9 (in the extraction the method labels of 4(a) are shifted one line against the numbers; label order OMEGA, GeoMol, GeoDiff, Torsional Diffusion, TD w/ PG, TD w/ PG*, MCF-S, MCF-B, DMT-B, DMT-B PC samp., MCF-L, DMT-L matched to the numeric rows in order; OMEGA/GeoMol/GeoDiff/TD rows equal TD Table 1, which confirms the mapping)
- Exact quote:
  > "* denotes reproduction using their codes." … "77.0 82.6 0.543 0.520 68.9 78.1 0.656 0.594" … "73.8 79.3 0.566 0.539 65.2 70.8 0.680 0.615" … "85.8 92.3 0.375 0.346 67.9 72.5 0.598 0.527" … "DMT-B, ours 55M 95.2 100.0 0.090 0.036"
- Shows vs infers: SHOWS (row mapping medium-high confidence). INFERENCE: even an inference-only TD variant lost 3.2 COV-R points on re-run, so published numbers of baselines should be re-run where checkpoints exist.
- Reused from (QM9 DMT-B row): round3/ledger/evidence_EVAL.md E-EVAL-003 (VERIFIED by V1).
- Supports card(s): C-BENCH-02, C-BENCH-06
- Confidence: medium (row mapping)

### E-BENCH-018
- Claim: The FM-refiner paper re-runs released baselines on GEOM-DRUGS (δ = 0.75 Å) and gets values that differ from the original papers: ET-Flow 80.15 / 0.458 (COV-R / AMR-R; reported 79.53 / 0.452), MCF-L 85.10 / 0.390, DMT-L 85.95 / 0.378; its best row is DMT-L + Refiner 87.47 / 0.349, COV-P 75.91, AMR-P 0.497.
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, Table 1
- Exact quote:
  > "MCF-L (242M) 85.10 92.86 0.390 0.343 66.63 70.00 0.623 0.546" … "ET-Flow (8.3M) 80.15 85.71 0.458 0.429 73.89 80.56 0.556 0.494" … "DMT-L + Refiner 87.47 94.12 0.349 0.319 75.91 81.51 0.497 0.446"
- Shows vs infers: SHOWS. INFERENCE (that these are re-runs, not copies): the values differ from E-BENCH-014/016/017; the paper does not say in this caption how they were obtained.
- Supports card(s): C-BENCH-02, C-BENCH-05, C-BENCH-06
- Confidence: high (numbers); medium (re-run interpretation)

### E-BENCH-019
- Claim: On GEOM-QM9 the FM-refiner paper adds a δ = 0.05 Å coverage column: ET-Flow 75.72 % (AMR-R 0.083), DMT 72.90 % (0.087), MCF-B 66.82 % (0.101), best DMT-B + Refiner 79.50 % (0.070).
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, Table 2
- Exact quote:
  > "MCF-B (62M) 66.82 67.86 0.101 0.050" … "ET-Flow (8.3M) 75.72 87.23 0.083 0.031" … "DMT-B + Refiner 79.50 89.44 0.070 0.026 80.37 97.92 0.076 0.021"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-008 (VERIFIED by V1); DMT row added from the same table.
- Supports card(s): C-BENCH-01, C-BENCH-06
- Confidence: high

### E-BENCH-020
- Claim: EnFlow re-ran ET-Flow on GEOM-DRUGS and reports a reproduction gap: ET-Flow as published 79.53 / 0.452 / 74.38 / 0.541 vs reproduced 79.54 / 0.470 / 69.79 / 0.604 (COV-R / AMR-R / COV-P / AMR-P); EnFlow itself 78.8 / 0.475 / 70.7 / 0.590 at 100 steps.
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_enflow.pdf, PDF p. 6, Table 2
- Exact quote:
  > "ET-Flow (8.3M) 1000 79.53 84.57 0.452 0.419 74.38 81.04 0.541 0.470" … "ET-Flowreproduced (8.3M) 79.54 85.00 0.470 0.444 69.79 75.53 0.604 0.538"
- Shows vs infers: SHOWS. INFERENCE: precision of a released flow model moved by 4.6 COV-P points on re-run, larger than many claimed improvements; re-running baselines under one pipeline is necessary (C-BENCH-02).
- Supports card(s): C-BENCH-02, C-BENCH-06
- Confidence: high

### E-BENCH-021
- Claim: AvgFlow reports on GEOM-DRUGS AvgFlowDiT-L COV-R 82.0 % / AMR-R 0.409 Å (100 steps) and states that baseline values in its tables are taken from the original papers (not re-run).
- Source type: paper
- Locator:
  - paper: papers/related/2025_cao_avgflow.pdf, PDF p. 8, Table 2 (caption and row)
- Exact quote:
  > "Baseline values are taken from the corresponding papers." … "AvgFlowDiT-L (64M) 100 82.0 87.3 0.409 0.381 75.7"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-02, C-BENCH-06
- Confidence: high

### E-BENCH-022
- Claim: ET-Flow also reports a GEOM-QM9 scaffold split (COV-R 95.00 %, AMR-R 0.083 Å), a DRUGS scaffold split (76.06 / 0.644) and a DRUGS-trained model tested on QM9 (86.68 / 0.218), as out-of-distribution evaluations beside GEOM-XL.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 19, App. D, Table 9 (labels and numbers on separate lines; label order QM9 RS, QM9 SS, DRUGS→QM9, DRUGS RS, DRUGS SS matched to rows in order; the QM9 RS and DRUGS RS rows equal Tables 2 and 1, which confirms the mapping)
- Exact quote:
  > "Table 9: Additional OOD results. We use RS and SS to indicate Random Split and Scaffold Split respectively." … "95.00 100.00 0.083 0.029 90.25 100.00 0.124 0.053" … "86.68 100.00 0.218 0.160"
- Shows vs infers: SHOWS (row mapping medium-high confidence)
- Supports card(s): C-BENCH-06
- Confidence: medium

### E-BENCH-023
- Claim: GeoMol reports confidence intervals from 3 repeated test-time runs of the same trained model (2K conformers per run): QM9 AMR-R 0.2254 ± 0.0012 Å, COV-R 91.52 ± 0.36 %.
- Source type: paper
- Locator:
  - paper: papers/related/2021_ganea_geomol.pdf, PDF p. 18, App. I (method) and PDF p. 19, Table 5 (values)
- Exact quote:
  > "The same trained model is run for 3 times for a given test molecule with K ground truth conformers, where one run means generating 2K conformers." … "QM9 91.52 100.00 0.2254 0.1938" … "± 0.36 ± 0.00 ± 0.0012 ± 0.0024"
- Shows vs infers: SHOWS (± glyph lost in extraction). INFERENCE: sampling-seed noise on QM9 AMR-R (≈ 0.001 Å) is the same order as our training-seed SD (CTRL median 0.0009 Å, round3/impact.md §2), so sampling seeds alone under-state the uncertainty of a training recipe.
- Supports card(s): C-BENCH-01
- Confidence: high

## C. Evaluation code and known evaluator bugs
### E-BENCH-024
- Claim: The ET-Flow README warns that its data preprocessing was changed after the paper and that the shipped configs may no longer reproduce the published results.
- Source type: blog (repository README)
- Locator:
  - blog: https://github.com/shenoynikhil/ETFlow README.md, M. Hassan / N. Shenoy et al., section "Preprocessing Data", accessed 2026-10-08 (SCOUT GEOM) and re-read 2026-10-11, snapshot papers/blogs/etflow_repo_readme.txt (line 75)
- Exact quote:
  > "I have changed some parts of the data preprocessing scripts to make it more efficient. However, these changes might mean that the configs might not lead to the same results as the one reported in the paper."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-02
- Confidence: high

### E-BENCH-025
- Claim: A user reports that ET-Flow's released QM9 checkpoint with the current preprocessing does not reproduce the paper's QM9 numbers (precision 94.5 %, recall 91 %, vs published COV-P 94.05 / COV-R 96.47), while DRUGS did reproduce; the author points to an older commit and to an RDKit-version change that filters more molecules.
- Source type: blog (GitHub issue)
- Locator:
  - blog: https://github.com/shenoynikhil/ETFlow/issues/31, StefanosPert (report, 2025-11-02) and shenoynikhil (reply, 2025-11-02), accessed 2026-10-11, snapshot papers/blogs/etflow_github_issues.txt (lines 7-21)
- Exact quote:
  > "able to reproduce the results for GEOM_Drugs" … "couldn't achieve the precision and recall numbers reported in the paper for QM9" … "precision of 94.5% and recall of 91%" … "rdkit version changed resulting in more complexes getting filtered out"
- Shows vs infers: SHOWS (a user report; which COV threshold "precision/recall" refers to is not stated, presumably δ = 0.5 Å). INFERENCE: the 0.073 Å QM9 figure is not guaranteed reproducible from the public code; we must re-score ET-Flow's checkpoint ourselves before quoting a gap (C-BENCH-02).
- Supports card(s): C-BENCH-02
- Confidence: medium (single user report)

### E-BENCH-026
- Claim: ET-Flow's Kabsch alignment used in training lacked a reflection check (an improper rotation could be returned); it was fixed after a user report in November 2024, after the NeurIPS paper.
- Source type: blog (GitHub issue)
- Locator:
  - blog: https://github.com/shenoynikhil/ETFlow/issues/11, VincentBt, 2024-11-26, accessed 2026-10-11, snapshot papers/blogs/etflow_github_issues.txt (lines 85-104)
- Exact quote:
  > "I noticed a mistake in your rotation matrix computation (part of the Kabsch algorithm for alignment)" … "if torch.det(R) < 0:  # reflection"
- Shows vs infers: SHOWS. INFERENCE: published ET-Flow checkpoints predate the fix; their numbers are what they are, but a retrained ET-Flow from the current code is a different model.
- Supports card(s): C-BENCH-02
- Confidence: high

### E-BENCH-027
- Claim: ET-Flow's chirality handling is a post-hoc global flip, which corrects a conformer only when every chiral centre is inverted; a user found random chirality in molecules with several chiral centres and the author called it "a method issue".
- Source type: blog (GitHub issue)
- Locator:
  - blog: https://github.com/shenoynikhil/ETFlow/issues/24, wdzwjc (2025-04-24) and shenoynikhil (2025-04-24), accessed 2026-10-11, snapshot papers/blogs/etflow_github_issues.txt (lines 61-84)
- Exact quote:
  > "if all chiral centers have been assigned the opposite chirality, you can just flip the molecule around the z-axis (multiply by -1). This is what we propose in the paper as `post-hoc` correction."
- Shows vs infers: SHOWS. INFERENCE: a stereo-correctness rate (fraction of generated conformers whose CIP labels match the input) is a cheap metric that favours TD-style models (chirality fixed by the RDKit seed) and should be reported beside RMSD metrics.
- Supports card(s): C-BENCH-02, C-BENCH-03
- Confidence: high (quote); medium (inference)

### E-BENCH-028
- Claim: ET-Flow's current `prepare_data.py` indexes TD's `split.npy` train/val indices into an unsorted `Path.glob("*")` listing of the raw GEOM folder, whereas TD indexes the same file into a sorted `glob('*.pickle')` list; unless the filesystem returns files in sorted order, the processed train/val sets are not TD's split and could overlap the 1000 test molecules (which ET-Flow takes from `test_mols.pkl`).
- Source type: code (third-party repo and ours)
- Locator:
  - code: github.com/shenoynikhil/ETFlow `scripts/prepare_data.py`:199-206 (snapshot papers/blogs/etflow_eval_code.txt); torsional-diffusion/utils/dataset.py:168-179
- Exact quote:
  > (ET-Flow) "all_pkl_files = list((raw_path / partition).glob("*"))" … "train_pkl_paths = [all_pkl_files[i] for i in train_split_indices]" ; (TD) "smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))" … "smiles = smiles[split]"
- Shows vs infers: SHOWS the code difference. INFERENCE, UNVERIFIED (not executed): a train/test overlap in the current ET-Flow preprocessing. It does not affect the released checkpoints (trained before the April 2025 revamp, E-BENCH-024); it matters only if we retrain ET-Flow from the current repo. Must be checked (compare processed train SMILES with test SMILES) before any retraining.
- Supports card(s): C-BENCH-02
- Confidence: high (code difference); low (that an overlap actually occurs)

### E-BENCH-029
- Claim: The "corrected evaluation" of GEOM-Drugs Revisited fixes the molecule-stability (valency) metric used for de novo 3D molecule generation (a bug that counted implausible valencies as valid) and adds GFN2-xTB energy/geometry checks; it does not change conformer-ensemble COV/AMR evaluation.
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 3, Introduction
- Exact quote:
  > "three major problems with the commonly used "molecular stability" metric, which measures whether atoms have valid valencies. One of the original implementations contained a bug that causes chemically implausible valencies to be counted as valid, leading to inflated stability scores."
- Shows vs infers: SHOWS (bug in the stability metric). INFERENCE: for conformer generation from a given graph, valency/stability is fixed by the input graph, so only the paper's GFN2-xTB energy and geometry benchmark transfers (C-BENCH-03, C-BENCH-07).
- Supports card(s): C-BENCH-03, C-BENCH-07
- Confidence: high

### E-BENCH-030
- Claim: The GEOM-Drugs Revisited repository's energy benchmark runs `xtb <xyz> --opt --charge <q>` (GFN2-xTB default level, no solvent) on each generated structure, parses xtb's "total energy gain" in kcal/mol as E_relax, compares initial vs optimised bond lengths, angles and torsions, and reports mean ± std over `--n_subsets` (5 in the README) subsets.
- Source type: code (third-party repo)
- Locator:
  - code: github.com/isayevlab/geom-drugs-3dgen-evaluation `scripts/energy_benchmark/xtb_optimization.py`:62, :104-106; `rmsd_energy.py`:43-48, :86-90; README "compute_pair_geometry.py" (snapshot papers/blogs/geom_revisited_repo.txt, accessed 2026-10-11)
- Exact quote:
  > "command = f"xtb {xyz_filename} --opt --charge {charge} --namespace {output_prefix} > {output_filename}"" … "total_energy_gain = float(line.split()[6])  # in kcal/mol" … "Compare initial and optimized structures by computing changes in bond lengths, bond angles, and torsion angles."
- Shows vs infers: SHOWS. INFERENCE: these scripts take SDF input and can score our generated conformers unchanged (one SDF per method), giving reviewers a recognised, third-party metric implementation; xtb 6.7.1 is already on gnode118 (round3/grounding.md §2.2).
- Supports card(s): C-BENCH-03
- Confidence: high

### E-BENCH-031
- Claim: The TD authors' shared Drive provides a released GEOM-DRUGS model (`workdir/drugs_default/best_model.pt`, 6.4 MB) with its hyper-parameters and the authors' own generated DRUGS test conformers (`drugs_steps20.pkl`, 457 MB), plus `drugs.tar.gz` (25.3 GB: data, splits, standardized pickles).
- Source type: blog (data repository listing)
- Locator:
  - blog: https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7, TD authors (Jing, Corso et al.), folder listing and HTTP Content-Range sizes, accessed 2026-10-11, snapshot papers/blogs/td_gdrive_listing.txt
- Exact quote (listing lines):
  > "drugs.tar.gz  id 1j_Pj6XZjJLf1tVQzjGWo2EFct2Nj_BvR  Content-Range: bytes 0-0/25308038362" … "best_model.pt       id 10tu38EDGe55M2v-0HtkMRYiPCcge4M5u  Content-Range: bytes 0-0/6392121" … "drugs_steps20.pkl   id 1HXY4FWt1CkQ28zPlnFzc6ZM_ZEnMRSoF  Content-Range: bytes 0-0/457404318"
- Shows vs infers: SHOWS (sizes; slurm/download_data.sh:11 already said 25.3 GB, now confirmed). INFERENCE: a DRUGS TD baseline, and any inference-only L-source change on top of TD (MMFF/xTB/template seeds), needs no DRUGS training (C-BENCH-05).
- Supports card(s): C-BENCH-02, C-BENCH-05
- Confidence: high

### E-BENCH-032
- Claim: TD's released DRUGS model uses the same architecture settings as our QM9 runs (4 conv layers, ns 32, nv 8, second-order irreps, radius 5 Å, batch 32, lr 1e-3, plateau patience 20, 250 epochs) but 74 node features and conformer-matched pickles from `drugs_standardized_multi_assignment_alignHs`.
- Source type: blog (released config file)
- Locator:
  - blog: same Drive, `workdir/drugs_default/model_parameters.yml`, accessed 2026-10-11, snapshot papers/blogs/td_gdrive_listing.txt (verbatim yml)
- Exact quote:
  > "batch_size: 32" … "in_node_features: 74" … "n_epochs: 250" … "num_conv_layers: 4" … "std_pickles: data/DRUGS/drugs_standardized_multi_assignment_alignHs/" … "use_second_order_repr: true"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-05
- Confidence: high

### E-BENCH-033
- Claim: ET-Flow's released checkpoints on Zenodo cover QM9 (`qm9-o3.ckpt`, 100.7 MB) and DRUGS (`drugs-o3.ckpt` 100.7 MB, `drugs-so3.ckpt` 109.4 MB), with split files (QM9.zip, DRUGS.zip, XL.zip).
- Source type: blog (data repository metadata)
- Locator:
  - blog: https://zenodo.org/records/14226681 (API https://zenodo.org/api/records/14226681), ET-Flow authors, created 2024-11-26, accessed 2026-10-11, snapshot papers/blogs/zenodo_etflow_14226681.txt
- Exact quote:
  > "file: drugs-o3.ckpt | size: 100676519" … "file: qm9-o3.ckpt | size: 100681751" … "file: drugs-so3.ckpt | size: 109393319"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-02, C-BENCH-05
- Confidence: high

### E-BENCH-034
- Claim: The MCF repository releases processed GEOM-QM9/DRUGS/XL data and pretrained checkpoints (QM9 model and DRUGS S/B/L) with test configs to reproduce the paper's tables.
- Source type: blog (repository README)
- Locator:
  - blog: https://github.com/apple/ml-mcf README.md, Wang et al. (Apple), section "Checkpoint & Reproducibility", accessed 2026-10-11, snapshot papers/blogs/mcf_repo_readme.txt
- Exact quote:
  > "We provide MCF model trained on GEOM-QM9 (`mcf_qm9.ckpt`) as well as MCF models of different sizes trained on GEOM-DRUGS, including small (S) `mcf_drugs_s.ckpt`, base (B) `mcf_drugs_b.ckpt`, and large (L) `mcf_drugs_l.ckpt`."
- Shows vs infers: SHOWS. INFERENCE: MCF's evaluation set is its own processed set (995 QM9 molecules, E-BENCH-010); re-scoring its generated conformers with our evaluator on the shared molecules is the fair comparison.
- Supports card(s): C-BENCH-02
- Confidence: high

### E-BENCH-035
- Claim: The NExT-Mol repository releases pretrained DMT-B and DMT-L checkpoints (OSF) and preprocessed GEOM-QM9/GEOM-DRUGS data for conformer prediction; its documented training uses 8 A100 GPUs.
- Source type: blog (repository README)
- Locator:
  - blog: https://github.com/acharkq/NExT-Mol README.md, Liu et al., sections "You can find the pretrained..." and "3D Conformer prediction", accessed 2026-10-11, snapshot papers/blogs/nextmol_repo_readme.txt
- Exact quote:
  > "You can find the pretrained DMT-B and DMT-L checkpoints in the following link [OSF]" … "**GEOM-QM9 dataset (with 8 A100 GPUs):**"
- Shows vs infers: SHOWS. UNVERIFIED: which dataset each OSF checkpoint was trained on (the README does not say; the OSF folder was not opened).
- Supports card(s): C-BENCH-02
- Confidence: medium

### E-BENCH-036
- Claim: A public GitHub issue on the TD repository questioned the fairness of comparing TD (trained on conformer-matched RDKit-L conformers) with baselines trained on true conformers; the authors replied that GeoMol and GeoDiff were trained on the same data and that test evaluation always uses ground truth.
- Source type: blog (GitHub issue)
- Locator:
  - blog: https://github.com/gcorso/torsional-diffusion/issues/4, wang1215789 (2022-09-30) and gcorso (2022-09-30, 2022-11-16), accessed 2026-10-11, snapshot papers/blogs/td_github_issues.txt
- Exact quote:
  > "your model is train on a different train data, does it is fair to other models?" … "The other machine learning baselines (GeoMol and GeoDiff) are trained with the exact same data."
- Shows vs infers: SHOWS (practitioner question, not a peer review). INFERENCE: any change to the training L source (xTB/MMFF-matched training, as S5 did) invites the same question; reporting the training L source per row answers it.
- Supports card(s): C-BENCH-01
- Confidence: high (quote); low (as evidence of reviewer behaviour)

## D. Required extra metrics (energy, local geometry, ensemble, runtime) and statistics
### E-BENCH-037
- Claim: GFN2-xTB re-optimisation of GFN2-xTB-optimised GEOM conformers gives a mean relaxation energy near zero, whereas MMFF evaluation of the same structures gives about 16 kcal/mol (GEOM-Drugs); so E_relax(GFN2-xTB) is the level-consistent energy metric for GEOM-trained generators.
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 10
- Exact quote:
  > "For conformers optimized with GFN2-xTB (as in GEOM-Drugs), the mean relaxation energy difference Erelax when re-optimized with GFN2-xTB is close to zero, as expected. In contrast, the same structures evaluated with MMFF show a mean Erelax of around 16 kcal/mol"
- Shows vs infers: SHOWS (Drugs). INFERENCE (from E-EVAL-013): GEOM-QM9 references are also GFN2-xTB minima, and our grounder measured 0.000 Å movement of GT conformer 0 under xtb 6.7.1 on 10/10 molecules (round3/grounding.md §2.2).
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-018 (VERIFIED by V1).
- Supports card(s): C-BENCH-03
- Confidence: high

### E-BENCH-038
- Claim: Nikitin et al. recommend reporting differences in bond lengths, bond angles and torsions between each generated structure and its own GFN2-xTB-optimised counterpart (not vs a GT conformer).
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 11, "GFN2-xTB energy-based geometry benchmark"
- Exact quote:
  > "we suggest to assess differences in bond lengths, bond angles, and torsion angles of generated and optimized counterparts."
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-019 (VERIFIED, round2 verify_B #22).
- Supports card(s): C-BENCH-03
- Confidence: high

### E-BENCH-039
- Claim: In Nikitin et al.'s Table 2 (GEOM-Drugs), MMFF-optimised structures relaxed with GFN2-xTB move by 1.12 × 10⁻² Å (bonds), 1.22° (angles), 4.89° (torsions) and release a mean E_relax of 11.4 kcal/mol; diffusion models beat MMFF on bonds/angles/E_relax but not on torsions (EQGAT-diff 8.58°, JODO 6.01°, Megalodon 5.58°).
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 12 (text) and PDF p. 13, Table 2
- Exact quote:
  > "These results clearly demonstrate that diffusion-based models already surpass MMFF in structural precision."
- Shows vs infers: SHOWS (table values medium confidence: scrambled extraction; values confirmed by V1 in round 3).
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-020 (VERIFIED after D-007).
- Supports card(s): C-BENCH-03
- Confidence: high (quote); medium (table values)

### E-BENCH-040
- Claim: Nikitin et al. state that MMFF94 energy evaluation is not suitable for models trained on GFN2-xTB-optimised data.
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 3, Introduction
- Exact quote:
  > "the use of energy evaluations at inappropriate levels of theory, such as MMFF94, which is not suitable for assessing models trained on GFN2-xTB-optimized data."
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-021 (VERIFIED by V1).
- Supports card(s): C-BENCH-03, C-BENCH-07
- Confidence: high

### E-BENCH-041
- Claim: TD's ensemble-property protocol (followed by ET-Flow) uses a random 100-molecule GEOM-DRUGS subset, min(2K, 32) conformers per molecule, GFN2-xTB relaxation, then Boltzmann-weighted E, μ, Δε, E_min compared with the CREST ensemble; TD's unrelaxed E_min error is 36.94 kcal/mol and 0.13 after relaxation.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 9, Sec. 4.4; PDF p. 27, Table 10
- Exact quote:
  > "For a random 100-molecule subset of DRUGS, we generate min(2K, 32) conformers per molecule, relax the conformers with GFN2-xTB" … "Tor. Diff. 36.91 0.92 4.93 36.94 0.22 0.35 0.54 0.13"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-024 and E-EVAL-022 (VERIFIED by V1); ET-Flow uses the same protocol, E-EVAL-054 (VERIFIED).
- Supports card(s): C-BENCH-03, C-BENCH-05, C-BENCH-07
- Confidence: high

### E-BENCH-042
- Claim: MCF could not obtain TD's 100-molecule property subset, so it drew three random subsets and reported mean and standard deviation over three runs with different seeds; the ensemble-property benchmark is therefore not exactly reproducible across papers.
- Source type: paper
- Locator:
  - paper: papers/related/2023_wang_mcf.pdf, PDF p. 15, App. (ensemble properties, Tab. 7)
- Exact quote:
  > "have the access to the exact subset of DRUGS used in (Jing et al., 2022), we randomly pick three subsets and report the averaged and standard deviation over three individual runs with different random seeds."
- Shows vs infers: SHOWS (the clause begins "…do not" on the previous line). INFERENCE: if we run the property benchmark on DRUGS we should publish our subset IDs and report subset-to-subset spread.
- Supports card(s): C-BENCH-03, C-BENCH-05
- Confidence: high

### E-BENCH-043
- Claim: TD reports runtime as CPU core-seconds per conformer (Intel i9-9920X, 8 threads, 10 random DRUGS test molecules × 8 conformers, excluding model loading): RDKit 0.10, TD 20 steps 4.90, TD 5 steps 1.76, GeoDiff 305.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 25, App. G.3 "Runtime evaluation"; PDF p. 9, Table 2
- Exact quote:
  > "We benchmark the methods on CPU (Intel i9-9920X) to enable comparison with RDKit." … "We select 10 molecules at random from the GEOM-DRUGS test set and generate 8 conformers per molecule using each method." … "RDKit - 1.002 0.895 0.10"
- Shows vs infers: SHOWS. INFERENCE: any added step (xTB seed relaxation, ring sampler, refiner) must be costed in the same unit; xTB at ≈ 0.14 s per QM9 seed (round3/grounding.md §2.2) is comparable to TD's per-conformer cost, so it is not free on DRUGS, whose molecules average 44.4 atoms (E-BENCH-055) against at most 29 for QM9.
- Supports card(s): C-BENCH-03
- Confidence: high

### E-BENCH-044
- Claim: EnFlow reports its ground-state results as mean ± standard deviation over 10 independent generation runs.
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_enflow.pdf, PDF p. 7, Table 3 notes
- Exact quote:
  > "Results of ET-Flow and EnFlow (JustFM) are reported as mean ± standard deviation over 10 independent runs, each generating one conformation per molecule."
- Shows vs infers: SHOWS (± glyph lost in extraction). With E-BENCH-023 (GeoMol, 3 runs) and E-BENCH-042 (MCF, 3 seeds): the field reports run-to-run spread for some tables, but none of the TD-P RMSD tables above reports training-seed variation.
- Supports card(s): C-BENCH-01
- Confidence: high

### E-BENCH-045
- Claim: TD reports AMR as a function of the number of rotatable bonds (GEOM-DRUGS) and notes that for very few rotatable bonds TD's error is set by the local structures it is given, so it equals RDKit's.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 25, App. H "Performance vs size", Figure 8
- Exact quote:
  > "With very few rotatable bonds, the error of torsional diffusion depends mostly on the quality of the local structures it was given, and therefore it has a similar error as RDKit."
- Shows vs infers: SHOWS. INFERENCE: a stratified table (by rotatable bonds and by ring content) is the natural place to show a ring/local-geometry contribution; our round-3 breakdown already has these bins (E-EVAL-057).
- Supports card(s): C-BENCH-01, C-BENCH-03
- Confidence: high

### E-BENCH-046
- Claim: The FM-refiner paper adds a conformer-level paired metric: the share of conformers whose precision RMSD improves vs degrades by more than a tolerance after refinement (improvement / downgrade rate).
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 14, Table 7 caption
- Exact quote:
  > "Table 7: Improvement rate (IR) and Downgrade rate (DR) (%) computed by RMSD-precision with a relative tolerance [δ] (Å) on GEOM-QM9."
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-009 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-03
- Confidence: high

### E-BENCH-047
- Claim: Two published arguments make QM9 at δ = 0.5 Å a weak headline: S23D calls the benchmark "likely saturated" (mean AMR < 0.1 Å), and the FM-refiner paper switches to δ = 0.05 Å because median COV is already 100 % at 0.5 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16, App. B.2.1; papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, Table 2 caption
- Exact quote:
  > "benchmark results on the QM9 metric are likely to be saturated." … "recent work already achieves 100% median COV at the commonly used threshold [δ] = 0.5 Å" … "we adopt the more challenging COV threshold of [δ] = 0.05 Å."
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-006 and E-EVAL-007 (VERIFIED).
- Supports card(s): C-BENCH-01, C-BENCH-06
- Confidence: high

### E-BENCH-048
- Claim: GEOM's CREST conformer weights are, by the GEOM authors' own statement, inaccurate, and DFT-quality weights exist only for the 1,511 BACE species, not for QM9.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 2 (Background & Summary) and PDF p. 1 (Abstract)
- Exact quote:
  > "the CREST statistical weights for each conformer are rather inaccurate." … "benchmarks that include conformer probabilities should use the DFT weights provided in GEOM." … "Ensembles of 1,511 species with BACE-1 inhibition data are also labeled with high-quality DFT free energies"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-016 and E-EVAL-059 (VERIFIED after D-001).
- Supports card(s): C-BENCH-07
- Confidence: high

## E. What reviewers criticised (OpenReview)
Note: the OpenReview review threads could not be read in this round (bot-verification wall; see section Z). The entries
below are the published critiques of the benchmark and practitioner reports that a reviewer is likely to raise; they are
not reviews.

### E-BENCH-049
- Claim: Zhou et al.'s cheap "RDKit + Clustering" baseline (random-dihedral, ETKDG and ETKDG+MMFF samplers 1:1:4, N_e = min(20 N_ref, 2000) samples, K-means to 2 N_ref outputs) reaches COV 97.65 % / MAT 0.1902 Å on the GD-protocol QM9 set, against plain RDKit 83.26 % / 0.3447 Å; recall metrics reward oversampling.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhou_dl_conformation_critique.pdf, PDF p. 4, Sec. 3 (method) and PDF p. 5, Table 1
- Exact quote:
  > "Ne = min(20Nref , 2000)" … "we employ the K-means algorithm with 2Nref clusters" … "RDKit + Clustering 97.65 100.00 0.1902 0.1818"
- Shows vs infers: SHOWS (GD protocol, 200 molecules; not comparable to TD-P numbers).
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-025 and E-EVAL-026 (VERIFIED by V1).
- Supports card(s): C-BENCH-02, C-BENCH-06
- Confidence: high

### E-BENCH-050
- Claim: Zhang et al. reply that the RDKit + Clustering comparison is unfair unless deep models are given the same ~2000-sample-then-cluster budget.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhang_infinite_physical_monkey.pdf, PDF p. 5
- Exact quote:
  > "At the very least, they should also sample 2000 conformations for DL methods followed by clustering."
- Shows vs infers: SHOWS. INFERENCE: whichever cheap baseline we include, the sample budget must be equal (2K per molecule) or reported per row.
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-028 (VERIFIED by V1).
- Supports card(s): C-BENCH-02
- Confidence: high

### E-BENCH-051
- Claim: Zhou et al. argue that GEOM's semi-empirical reference accuracy is insufficient for applications needing DFT-quality conformers, i.e. agreement with GEOM is agreement with GFN2-xTB minima.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhou_dl_conformation_critique.pdf, PDF p. 6, Discussion
- Exact quote:
  > "for quantum property prediction, we require conformations that are close to the DFT-optimized conformations. The accuracy of GEOM's semi-empirical DFT is insufficient to meet these requirements."
- Shows vs infers: SHOWS (opinion, no measurement). INFERENCE: a method that gains by moving seeds toward GFN2-xTB minima (C-GEOM-01) will be read by reviewers as "fitting the reference level"; the paper should state the reference level explicitly and add the xTB energy metric (C-BENCH-03).
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-051 (VERIFIED by V1).
- Supports card(s): C-BENCH-03, C-BENCH-07
- Confidence: high (quote); low (effect size)

## F. GEOM-DRUGS feasibility (data, time, checkpoints)
### E-BENCH-052
- Claim: TD's GEOM-DRUGS models were trained for 250 epochs, taking 4 to 11 days on a single NVIDIA RTX A6000; the QM9 model used the same hyper-parameters.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 24, App. G.2 "Training and tuning details"
- Exact quote:
  > "the torsional diffusion models were trained on NVIDIA RTX A6000 GPUs for 250 epochs with the Adam optimizer (taking from 4 to 11 days on a single GPU)." … "for GEOM-QM9 a new model with the same hyperparameters was trained."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-05
- Confidence: high

### E-BENCH-053
- Claim: GeoMol's 250-epoch training took about 1.8 days on QM9 and 6.1 days on DRUGS on one V100 (ratio ≈ 3.4).
- Source type: paper
- Locator:
  - paper: papers/related/2021_ganea_geomol.pdf, PDF p. 19, App. K "Compute details"
- Exact quote:
  > "training GEOMOL for the full 250 epochs takes approximately 1.8 days for the QM9 dataset and 6.1 days for the DRUGS dataset."
- Shows vs infers: SHOWS (different model; used only as a second DRUGS/QM9 cost ratio).
- Supports card(s): C-BENCH-05
- Confidence: high

### E-BENCH-054
- Claim: ET-Flow's DRUGS training used 250 epochs × 5000 batches of 64 per GPU on 8 A100 GPUs (QM9: 200 epochs on 4 A100s), i.e. far more compute than our node; retraining ET-Flow on DRUGS is out of scope and its released checkpoint must be used.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 16, App. A.4
- Exact quote:
  > "For GEOM-DRUGS, we train ET-Flow for a fixed 250 epochs with a batch size of 64 and 5000 training batches per epoch per GPU on 8 A100 GPUs."
- Shows vs infers: SHOWS (first clause). INFERENCE: second clause (our node has 4 RTX 3090).
- Supports card(s): C-BENCH-02, C-BENCH-05
- Confidence: high

### E-BENCH-055
- Claim: GEOM's drug-like molecules average 44.4 atoms (24.9 heavy) and 6.5 rotatable bonds (max 181 atoms, 53 rotatable bonds); QM9 molecules have at most 9 heavy (29 total) atoms.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 2, "Background & Summary" (Table 1 discussion)
- Exact quote:
  > "containing an average of 44.4 atoms (24.9 heavy atoms), up to a maximum of 181 atoms (91 heavy atoms)." … "mean (6.5) and maximum (53) number of rotatable bonds." … "The QM9 dataset is limited to 9 heavy atoms (29 total atoms)"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-05
- Confidence: high

### E-BENCH-056
- Claim: GEOM's raw RDKit pickles (`rdkit_folder.tar.gz`, needed by ET-Flow's preprocessing) are 50.1 GB compressed on Harvard Dataverse; TD's own DRUGS archive is 25.3 GB (E-BENCH-031).
- Source type: blog (data repository metadata)
- Locator:
  - blog: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/JNGTDF, GEOM dataset (Axelrod & Gómez-Bombarelli), version 4.0 released 2022-02-11, accessed 2026-10-11, snapshot papers/blogs/geom_dataverse_files.txt
- Exact quote:
  > "id 4327252 | filename rdkit_folder.tar.gz | filesize 50137579520"
- Shows vs infers: SHOWS. INFERENCE: TD's 25.3 GB archive is enough for TD-architecture DRUGS work; the 50 GB GEOM archive is needed only to re-run another group's preprocessing. /scratch on gnode118 must hold ≈ 25 GB compressed + extracted data + caches (extracted size not measured).
- Supports card(s): C-BENCH-05
- Confidence: high

### E-BENCH-057
- Claim: The TD README says first-time featurisation takes about 2 h on a single CPU core (default dataset DRUGS) and runs conformer matching for DRUGS as 300 parallel workers of 1000 molecules each with 30 conformers per molecule.
- Source type: code (repo README)
- Locator:
  - code: torsional-diffusion/README.md:38, :54-58
- Exact quote:
  > "a featurisation procedure starts (about 2h on single core CPU, faster with more cores)" … "python standardize_confs.py --out_dir data/DRUGS/standardized_pickles --root data/DRUGS/drugs/ --confs_per_mol 30 --worker_id $i --jobs_per_worker 1000 &"
- Shows vs infers: SHOWS. UNVERIFIED: the per-molecule matching time on DRUGS (not stated anywhere we found).
- Supports card(s): C-BENCH-05
- Confidence: high (quote); low (cost of matching)

### E-BENCH-058
- Claim: Our measured QM9 training cost on gnode118 is 377.7 s per epoch for 2591 iterations (6.86 it/s, batch 32), ≈ 11.7 GPU-h per 100-epoch run including validation; packed evaluation costs ≈ 0.17 GPU-h per condition (0.26 unpacked).
- Source type: our-data
- Locator:
  - our-data: round3/grounding.md:62-66 (§2.2 "Measured timings", quoting a round-2 training log and `logs/tordiff_r2_eval_*.log`)
- Exact quote:
  > "a round-2 training log has `Epoch 99: TIMING epoch_time 377.7s ... iters 2591 (6.86 it/s)`; with validation this gives the brief's ≈ 11.7 h per 100 epochs." … "give **≈ 0.17 GPU-h per condition**."
- Shows vs infers: SHOWS (P3-verified grounding document)
- Supports card(s): C-BENCH-05, C-BENCH-06
- Confidence: high

### E-BENCH-059
- Claim: QM9 featurisation in a single process takes about 8 minutes on our cluster, and the QM9 training caches are 0.94–1.32 GB.
- Source type: our-data
- Locator:
  - our-data: EXPERIMENTS.md:286; round3/grounding.md:59
- Exact quote:
  > "| Featurise in a single process | A multiprocessing shared-memory limit hung the job for 6 h at 106k molecules | Lost ~8 h; single process takes ~8 min |" … "`cache_paired.train` is 1.32 GB, `cache_rematch.train` 0.94 GB, `cache_raw.train` 1.31 GB."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-05
- Confidence: high

### E-BENCH-060
- Claim: [BENCH-calc] Estimated cost of one TD-architecture GEOM-DRUGS training run on one RTX 3090: 23–63 min per epoch (3.7–10× our QM9 epoch), so 1.6–4.4 GPU-days for 100 epochs and 4–11 GPU-days for TD's 250 epochs; a 3-seed, 2-arm DRUGS study (6 runs × 250 epochs) needs 24–66 GPU-days, i.e. 6–17 days of the whole node. Featurisation ≈ 2 h CPU; caches ≈ 7–10 GB RAM; xTB on DRUGS test seeds ≈ 1–2 s each (vs 0.14 s on QM9).
- Source type: our-data (calculation)
- Locator:
  - our-data: computed from E-BENCH-052 (4–11 days / 250 epochs on an A6000), E-BENCH-058 (377.7 s per QM9 epoch), E-BENCH-053 (GeoMol ratio 3.4 as a cross-check: 250 QM9 epochs ≈ 29 GPU-h on our node × 3.4 ≈ 4.1 days, inside TD's range), E-BENCH-055 (atoms), E-BENCH-059 (cache sizes); scratchpad `bench/drugs_cost.txt`
- Exact quote (calculation):
  > "TD DRUGS epoch = (4..11 d x 86400 s) / 250 = 1382..3802 s = 23..63 min ; QM9 epoch 377.7 s -> ratio 3.7..10.1" … "100 ep = 1.6..4.4 d"
- Shows vs infers: INFERENCE. Assumes an RTX 3090 ≈ an RTX A6000 for this fp32 workload (similar fp32 throughput; not measured) and that our data loading keeps up (6 CPUs). Cache size scales QM9's 0.94–1.32 GB by ≈ 2.9× molecules × ≈ 2.5× atoms per conformer (both rough). xTB cost scales the measured 0.14 s by (44.4/18 atoms)^2–3 ≈ 6–15×, where 18 is an assumed QM9 mean atom count (UNVERIFIED; measure on 20 DRUGS seeds). Must be replaced by a measured 1-epoch smoke run (`--limit_train_iters`, parsing.py:37) before any GPU budget is approved.
- Supports card(s): C-BENCH-05, C-BENCH-06
- Confidence: medium (TD-derived range); low (cache and xTB scaling)

## G. Leak-free oracle protocols
### E-BENCH-061
- Claim: The only published "other-conformer local structure" test we found is TD App. F.1: matching a GT conformer with the local structure of another randomly chosen GT conformer of the same molecule gives a mean RMSD_min of 0.284 Å on GEOM-DRUGS, against 0.324 Å with matched RDKit local structures; it is a model-free matched-RMSD floor, not a generation test.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 21, App. F.1 "Matched conformer RMSD"
- Exact quote:
  > "it is only slightly larger than the average RMSDmin of 0.284 Å resulting from matching a ground truth conformer to the local structure of another randomly chosen ground truth conformer"
- Shows vs infers: SHOWS. INFERENCE: TD's authors already treated "another conformer's L" as the honest reference-level L; our split-half test is the generative version of the same idea and can cite this as precedent. No paper among those read (TD, ET-Flow, MCF, NExT-Mol/DMT, GeoMol, S23D, FM-refiner, AvgFlow, EnFlow, GEOM, GEOM-revisited, Zhou, Ezzine, GO-Flow, PuckerFlow) defines a split-half or leak-free true-geometry generation protocol (search record in section Z).
- Supports card(s): C-BENCH-04
- Confidence: high (quote); medium (negative search)

### E-BENCH-062
- Claim: Our round-1/2 "true L" conditions leak conformer identity: `gtL` draws seeds with replacement from the molecule's own GT conformers, so about 92 % of target conformers have their own L among the seeds, and B1's true-L advantage is almost entirely conformer identity (inferred other-conformer AMR-R 0.16–0.35 Å vs 0.016–0.046 own).
- Source type: our-data
- Locator:
  - our-data: round3/impact.md §3g (lines 139-170; `round3/impact_data/ntrue_leak.py`, `.out`)
- Exact quote:
  > "The grounder showed that round-1 `gtL` draws each seed with replacement from the molecule's *own* GT conformers (`diffusion/sampling.py:86`)." … "B1's true-L advantage is almost entirely conformer identity."
- Shows vs infers: SHOWS the analysis; the leak-corrected values are inverted estimates, not measurements (validity judge, round3/panel/score_VALIDITY_r1.md:76).
- Supports card(s): C-BENCH-04
- Confidence: high (leak mechanism); medium (inverted values)

### E-BENCH-063
- Claim: A split-half leak-free oracle is already implemented (uncommitted, round 4): GT conformers are cleaned exactly as the evaluator does, split alternately by cleaned index into halves A/B, seeds come from one half and recall is scored on the other; molecules with one clean conformer are excluded and counted; a model-free "copy the other half" baseline is included.
- Source type: code
- Locator:
  - code: tools/make_split_half.py:1-31 (docstring), :57; tools/analyze_leakfree.py:1-40 (pre-declared design); slurm/r4_leakfree_array.sbatch (untracked, checked 2026-10-12)
- Exact quote:
  > "split ALTERNATING BY CLEANED GT INDEX" … "Molecules with n_clean = 1 (no other conformer exists) and n_clean = 0 are excluded and counted." … "split_AtoB  seeds A, 2|B| confs, recall vs B"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-04
- Confidence: high

### E-BENCH-064
- Claim: [BENCH-calc] About 24 % of QM9 test molecules (225 of the 940 in the local-structure file) have a single GT conformer and cannot enter any leak-free other-conformer test, so the split-half universe is ≈ 715 molecules (76 %); these single-conformer molecules are ring-rich (98 % contain a ring) and have the highest RDKit-L floor (0.136 Å).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/local_structure_test.csv via round3 E-EVAL-048 (scratchpad `macro_micro.py` output "1 225 0.1364 0.9822 0.4267"); arithmetic in scratchpad `bench/drugs_cost.txt`
- Exact quote:
  > "1 225 0.1364 0.9822 0.4267" ; "split-half universe: 940 - 225 = 715 molecules = 76.1 %"
- Shows vs infers: SHOWS (counts; E-EVAL-048 VERIFIED by V1 and reproduced). INFERENCE: the leak-free ceiling describes the multi-conformer 76 %, where the ring-L problem is smaller; ring-geometry claims on single-conformer molecules need a non-oracle L source (xTB / template) instead.
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-048 (counts).
- Supports card(s): C-BENCH-04
- Confidence: high

### E-BENCH-065
- Claim: Ezzine et al. evaluate a torsion sampler conditioned on local structure on "unseen local structures" (held-out L of the same molecules, MD data), the closest analogue of a leak-free L test in the generative literature we found; it targets a Boltzmann density, not GEOM recall.
- Source type: paper
- Locator:
  - paper: papers/related/2025_ezzine_torsional_gfn.pdf, PDF p. 3, Sec. 3.2 "Results"
- Exact quote:
  > "whether the model can generalize to unseen local structures and unseen molecules."
- Shows vs infers: SHOWS (workshop paper, small MD set).
- Supports card(s): C-BENCH-04
- Confidence: medium

## H. Our data / our code
### E-BENCH-066
- Claim: Our round-2 analysis already implements the statistics reviewers ask for: pre-declared contrasts, TD-protocol union universe with model failure = COV 0, paired molecule bootstrap with Holm over the primary family, the success intersection for COV-R@0.1, training-seed SD on every row, and a hierarchical (seeds then molecules) bootstrap for non-inferiority.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/README.md (lines 6-14)
- Exact quote:
  > "TD protocol (union of molecules, model failure = COV 0), paired molecule bootstrap, Holm over the PRIMARY family" … "training-seed SD of the arm and ref aggregates next to the bootstrap CI"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01
- Confidence: high

### E-BENCH-067
- Claim: Final round-2 result: S3 on RDKit L is worse than CTRL_rematch by +0.0032 Å AMR-R [+0.0013, +0.0052], Holm p = 0.0096, n = 934 (training-seed SDs 0.0004 for S3 with 2 seeds, 0.0013 for CTRL with 3 seeds).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:50 (`S3.rdkit_vs_CR`, MAT-R, primary)
- Exact quote:
  > "S3.rdkit_vs_CR,primary,union,MAT-R,primary,934,S3,steps20_seed0,CR,steps20_seed0,,0.1818677432916211,0.1786359860697768,0.0032317572218443077,0.0012702605359765939,0.005228308316968585,0.0012,0.00038785291373951076,2,0.0013366058280782855,3,,ok,,,0.0096,"
- Shows vs infers: SHOWS. INFERENCE: differences of 0.003 Å are detectable with this design (CI half-width ≈ 0.002 Å), so the evaluation is powered for the effects we care about on QM9; the same design should be the paper's.
- Supports card(s): C-BENCH-01, C-BENCH-06
- Confidence: high

### E-BENCH-068
- Claim: The only real-use gains so far come from geometry: S5 (MMFF-matched training, MMFF at test) beats CTRL_rematch with MMFF at test by −0.0041 Å [−0.0064, −0.0019] and CTRL_rematch on RDKit L by −0.0287 Å [−0.0339, −0.0235] (AMR-R 0.1500 vs 0.1786).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:194 (`S5.premmff_vs_CRpremmff`) and :210 (`S5.premmff_vs_CR`)
- Exact quote:
  > "S5.premmff_vs_CRpremmff,primary,union,MAT-R,primary,934,B3,steps20_seed0,CR,S5ctrl_pre_mmff,,0.14997364091242826,0.15403840862692844,-0.004064767714500185,-0.006439843470125827,-0.0018997309350040797" … "S5.premmff_vs_CR,primary,union,MAT-R,primary,934,B3,steps20_seed0,CR,steps20_seed0,,0.14997364091242826,0.1786359860697768,-0.028662345157348532"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-02, C-BENCH-06
- Confidence: high

### E-BENCH-069
- Claim: Released TD (QM9) with RDKit L, sampling seed 0, scores AMR-R 0.1752 Å on 935 evaluated molecules (65 failures), COV-R@0.05 3.91 %; RDKit ETKDG alone 0.2298 Å; MMFF-relaxed seeds (A2) 0.1507 Å with COV-R@0.05 17.89 %.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, lines `qm9_default/R0_base_seed0`, `qm9_default/A0_rdkit_etkdg_only`, `qm9_default/A2_pre_mmff`
- Exact quote:
  > "qm9_default/R0_base_seed0 threshold=0.5000 COV-R_mean=89.0293 COV-R_median=100.0000 MAT-R_mean=0.1752" … "qm9_default/A0_rdkit_etkdg_only threshold=0.5000 COV-R_mean=83.9217 COV-R_median=100.0000 MAT-R_mean=0.2298" … "qm9_default/A2_pre_mmff threshold=0.5000 COV-R_mean=88.9339 COV-R_median=100.0000 MAT-R_mean=0.1507"
- Shows vs infers: SHOWS. INFERENCE: our re-scored TD COV-R (89.0 %) is 3.8 points below TD's published 92.8 % because of failure accounting (E-BENCH-011); any table that mixes our numbers with published ones must say so.
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-038, E-EVAL-039, E-EVAL-040 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-070
- Claim: `tools/paired_compare.py` supports thresholds 0.05/0.1/0.25/0.5 Å and a success-intersection universe for paired bootstrap comparisons.
- Source type: code
- Locator:
  - code: tools/paired_compare.py:32, :41
- Exact quote:
  > "parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.05, 0.1, 0.25]," … "parser.add_argument('--universe', choices=['union', 'intersection'], default='union')"
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-037 (VERIFIED).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high

### E-BENCH-071
- Claim: GEOM's conformer ensembles were generated with CREST 2.9 and xTB 6.2.3 (gnode118 has xtb 6.7.1).
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 8, "Code availability"
- Exact quote:
  > "CREST version 2.9 was used with xTB version 6.2.3 to generate the initial CREs."
- Shows vs infers: SHOWS
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-056 (VERIFIED, D-003).
- Supports card(s): C-BENCH-03
- Confidence: high

### E-BENCH-072
- Claim: Measured GFN2-xTB cost on gnode118 (xtb 6.7.1, single thread, `--opt normal`): median 0.135 s, mean 0.157 s per QM9 optimisation; GT conformer 0 of each of 10 test molecules converged in 1 iteration with 0.000 Å displacement, while ETKDG seeds needed 17–63 iterations and moved 0.09–0.32 Å.
- Source type: our-data
- Locator:
  - our-data: round3/grounding.md:66-73 (§2.2, job 13057)
- Exact quote:
  > "GT conformer 0 of each molecule converged after 1 iteration with 0.000 Å heavy-atom displacement." … "ETKDG seeds needed 17–63 iterations and moved by 0.09–0.32 Å heavy-atom RMSD." … "Median 0.135 s, mean 0.157 s, max 0.383 s per optimisation."
- Shows vs infers: SHOWS (n = 20 calls on 10 molecules)
- Supports card(s): C-BENCH-03, C-BENCH-05, C-BENCH-06
- Confidence: high (measurement); medium (small sample)

### E-BENCH-073
- Claim: The QM9 local-structure analysis covers 940 test molecules with 7,554 GT conformers; ETKDG seeds differ from their assigned GT conformer by bond RMSD 0.036 Å and angle RMSD 3.88°, against 0.004 Å / 1.67° between different GT conformers of the same molecule.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/local_structure_test.log, lines `floor_best_sym`, `bond_rmsd_assigned`, `angle_rmsd_assigned`, `bond_rmsd_gtgt`, `angle_rmsd_gtgt`
- Exact quote:
  > "floor_best_sym: n=7554 mean=0.0986 median=0.0606 per-molecule-mean=0.1145" ; "bond_rmsd_assigned: mean=0.0360" ; "angle_rmsd_assigned: mean=3.8821" ; "bond_rmsd_gtgt: mean=0.0040" ; "angle_rmsd_gtgt: mean=1.6735"
- Shows vs infers: SHOWS. INFERENCE (for C-BENCH-04): GT conformers of one molecule share bonds/angles almost exactly (0.004 Å, 1.7°), so in a split-half test the other half's L differs from the target's own L mainly in ring puckers and torsion-coupled angles; the residual leak through near-duplicate conformers should be measured per molecule (min L-RMSD between halves).
- Reused from: round3/ledger/evidence_EVAL.md E-EVAL-047 (VERIFIED by V1).
- Supports card(s): C-BENCH-03, C-BENCH-04
- Confidence: high

## Z. Search negatives
(Recorded so verifiers can re-run them; not evidence entries.)
- **OpenReview reviews could not be read.** Every OpenReview endpoint tried on 2026-10-11 (API v2 `api2.openreview.net/notes?forum=...`,
  API v1, the forum page via WebFetch, and the forum page in the desktop browser pane) returned a browser-verification
  challenge ("Verifying your browser ... Complete the check below to continue"). Completing bot checks is not allowed for
  agents, so no review text was read and no reviewer criticism is cited as such. The OpenReview *search* API worked and
  gave these forum IDs for a human to open (signing in skips the check): ET-Flow NeurIPS 2024 `avsZ9OlR60`; NExT-Mol
  ICLR 2025 `p66a00KLWN`; Particle Guidance ICLR 2024 `KqbCvIFBY7`; Equivariant Blurring Diffusion NeurIPS 2024
  `Aj0Zf28l6o`; MCF ICML 2024 `I44Em5D5xy` (ICML reviews are usually not public); "Enhancing Molecular Conformer Generation
  via Fragment-Augmented Diffusion Pretraining" (TMLR, accepted) `t5WzHOniAF` (a further TD-P competitor not yet in
  `papers/`, flagged for SCOUT PRIOR). The TD NeurIPS 2022 forum ID was not found (only the DBLP record `lLUdc5NlKh`).
  Substitutes used instead: published critiques (Zhou, Zhang, S23D, FM-refiner, GEOM-revisited; E-BENCH-047, -049..-051)
  and public GitHub issues (E-BENCH-025..-027, -036).
- No paper among those read defines a split-half or other-conformer *generation* oracle (searched the local fulltexts for
  "another conformer", "other conformer", "held-out conformer", "leak", "oracle", "true local structure" on 2026-10-11;
  hits only TD App. F.1 (E-BENCH-061), Ezzine (E-BENCH-065), and irrelevant uses).
- No TD-P paper reports training-seed variation for its RMSD tables (TD, ET-Flow, MCF, NExT-Mol, AvgFlow, S23D, FM-refiner,
  EnFlow main RMSD tables checked by grepping each PDF page for "seed", "standard deviation", "±"); run-to-run spread
  appears only for GeoMol (3 sampling runs), MCF ensemble properties (3 subsets/seeds) and EnFlow ground-state (10 runs).
- GitHub API rate limit (60/h unauthenticated) was hit while fetching TD issue #15 a second time; its text in
  `papers/blogs/td_github_issues.txt` was copied from the first, successful API response in the same session.
- Not opened: the NExT-Mol OSF folder (which dataset each DMT checkpoint was trained on), MCF's checkpoint archive size,
  ET-Flow's older commit `98bba05` preprocessing (whether it sorted the raw file list).
