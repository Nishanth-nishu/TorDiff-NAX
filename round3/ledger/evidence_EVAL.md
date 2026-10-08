# Evidence ledger: SCOUT EVAL (lens: evaluation, references, baselines; brief Q4)

Agent tag EVAL. Round 3, P1. Started 2026-10-08. Template: `orchestration/templates/evidence_entry.md`.

Conventions used in this ledger
- Paper quotes were checked against `pdftotext -layout -f N -l N <pdf>` (PDF page N, 1-based). For every paper used
  here the PDF page equals the printed page unless stated. Table rows are quoted as they appear in the layout text;
  where a row label sits on a different line from its numbers, the label-to-number mapping was read from the
  non-layout text order and is stated.
- The Greek delta glyph is lost in the extracted text of several PDFs; it is written `[δ]` inside quotes.
- Our-data entries quote the exact `SUMMARY` / `SWEEP` line (truncated with "..." only at the end). Computed values
  are marked [EVAL-calc] with the script path in the scratchpad
  (`C:\Users\HP\AppData\Local\Temp\claude\C--Users-HP-Desktop-tor-diff\ebb0c292-63dd-41c3-ab3b-16fa6f4d7fcc\scratchpad\`).
- New papers stored this round: `papers/related/2023_zhang_infinite_physical_monkey.pdf` (+ fulltext, INDEX row added).
  `papers/related/2022_axelrod_geom.pdf` (arXiv 2006.05531v4) was fetched by SCOUT EVAL and SCOUT GEOM at the same time
  (identical 1,637,253-byte file; SCOUT GEOM's copy and INDEX row are the ones on disk); page numbers here refer to that PDF.
  New snapshots: `papers/blogs/rowan_openconf_2026.txt`, `papers/blogs/rdkit_release_notes_2024_03.txt`,
  `papers/blogs/condaforge_xtb_metadata_eval.txt` (P2, D-003).
- P2 (2026-10-08): entries revised after V1's verdicts carry "Revised after D-0xx" with the original text kept as superseded;
  new entries E-EVAL-055…059 are in section H. Responses are in `round3/ledger/disputes.md` under D-001…D-007.

---------------------------------------------------------------------------------------------------------------------
## A. Where current QM9 methods sit (TD protocol, GEOM-QM9, 1000-molecule test set)

### E-EVAL-001
- Claim: ET-Flow reports on GEOM-QM9 at δ = 0.5 Å COV-R mean 96.47 %, AMR-R mean 0.073 Å (median 0.047), COV-P 94.05 %, AMR-P 0.098 Å; the TD row in the same table is 92.80 % / 0.178 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 7, Table 2 (ET-Flow row = 6th numeric row, mapping from the non-layout text order: CGCF, GeoDiff, GeoMol, Torsional Diff., MCF, ET-Flow, ET-Flow-SO(3)); same numbers labelled "ET-Flow (QM9 RS)" in Table 9, PDF p. 19
- Exact quote:
  > "Table 2: Molecule conformer generation results on GEOM-QM9 ( [δ] = 0.5Å)" … "96.47 100.00 0.073 0.047 94.05 100.00 0.098 0.039"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-06
- Confidence: high (re-checked; round 2 verify_B #18 already cites ET-Flow 0.073 from p. 19)

### E-EVAL-002
- Claim: MCF (base model) reports on GEOM-QM9 COV-R 95.0 %, AMR-R 0.103 Å, COV-P 93.7 %, AMR-P 0.119 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2023_wang_mcf.pdf, PDF p. 6, Table 1
- Exact quote:
  > "MCF 95.0 100.0 0.103 0.044 93.7 100.0 0.119 0.055"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-06
- Confidence: high

### E-EVAL-003
- Claim: NExT-Mol reports on GEOM-QM9 DMT-B COV-R 95.2 %, AMR-R 0.090 Å, and lists OMEGA at COV-R 85.5 %, AMR-R 0.177 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2025_liu_nextmol.pdf, PDF p. 8, Table 4(b) "Performances on the GEOM-QM9 dataset"
- Exact quote:
  > "OMEGA - 85.5 100.0 0.177 0.126 82.9 100.0 0.224 0.186" … "DMT-B, ours 55M 95.2 100.0 0.090 0.036 93.8 100.0 0.108 0.049"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-04, C-EVAL-06
- Confidence: high

### E-EVAL-004
- Claim: AvgFlow's GEOM-QM9 table lists RDKit at AMR-R 0.235 Å (COV-R 85.1 %), OMEGA 0.177 Å, and AvgFlowDiT (52M) at AMR-R 0.082 Å; baseline rows are copied from the original papers.
- Source type: paper
- Locator:
  - paper: papers/related/2025_cao_avgflow.pdf, PDF p. 7, Table 1
- Exact quote:
  > "RDKit 85.1 100 0.235 0.199 86.8 100 0.232 0.205" … "AvgFlowDiT (52M) 96.0 100 0.082 0.030 95.0 100 0.088 0.039" … "Baseline values are taken from the corresponding papers."
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-04, C-EVAL-06
- Confidence: high

### E-EVAL-005
- Claim: S23D-B (with chirality correction) reports GEOM-QM9 AMR-R 0.090 Å, COV-R 96.0 %, and evaluates on 995 test molecules (MCF's processed set), not 1000; its Table 5 also shows that compared QM9 models differ in training conformers per molecule (KG 10 for MCF-B, 30 for TD/ET-Flow) and in chirality correction.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16, App. B.2.1 and Table 5
- Exact quote:
  > "The processed test dataset contained 995 molecules rather than the 1000 indexed in the test set" … "MCF-B 64 10 NO 95.0 100.0 0.103 0.044" … "S23D-B-1/13 (S) 24.8 30 YES 96.0 100.0 0.090 0.047 93.8 100.0 0.111 0.059"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-06
- Confidence: high

### E-EVAL-006
- Claim: S23D argues that the QM9 benchmark is likely saturated because mean AMR of current models is < 0.1 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16, App. B.2.1
- Exact quote:
  > "Mean AMR values across these models are < 0.1, which is around 10% of the typical bond length between atoms, indicating that benchmark results on the QM9 metric are likely to be saturated."
- Shows vs infers: SHOWS (the argument is from AMR < 0.1 Å, not from δ = 0.5 Å coverage; see round 2 verify_B #16)
- Reused from: round2/research_B.md (§3 and table row "QM9 at 0.5 Å saturated"), verified in round2/verify_B.md #16 (VERIFIED quote / PARTIAL SUPPORT); quote re-checked here.
- Supports card(s): C-EVAL-01, C-EVAL-02, C-EVAL-06
- Confidence: high

### E-EVAL-007
- Claim: The FM-refiner paper evaluates GEOM-QM9 coverage at δ = 0.05 Å because median COV is already 100 % at δ = 0.5 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, Table 2 caption
- Exact quote:
  > "Since recent work already achieves 100% median COV at the commonly used threshold [δ] = 0.5 Å and a median AMR below 0.05 Å, we adopt the more challenging COV threshold of [δ] = 0.05 Å."
- Shows vs infers: SHOWS
- Reused from: round2/research_B.md table row "FM-refiner uses δ = 0.05 Å on QM9", verified in round2/verify_B.md #17 (VERIFIED); quote re-checked here.
- Supports card(s): C-EVAL-01, C-EVAL-02, C-EVAL-06
- Confidence: high

### E-EVAL-008
- Claim: At δ = 0.05 Å on GEOM-QM9 the FM-refiner paper reports COV-R mean 66.82 % (MCF-B), 75.72 % (ET-Flow, AMR-R 0.083), 72.90 % (DMT) and 79.50 % (DMT-B + Refiner, AMR-R 0.070, the best row).
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, Table 2
- Exact quote:
  > "MCF-B (62M) 66.82 67.86 0.101 0.050" … "ET-Flow (8.3M) 75.72 87.23 0.083 0.031" … "DMT-B + Refiner 79.50 89.44 0.070 0.026 80.37 97.92 0.076 0.021"
- Shows vs infers: SHOWS
- Reused from (ET-Flow row only): round2/research_B.md, verified in round2/verify_B.md #18 (VERIFIED); other rows new, re-checked here.
- Supports card(s): C-EVAL-01, C-EVAL-06
- Confidence: high

### E-EVAL-009
- Claim: The FM-refiner paper reports a conformer-level paired metric on GEOM-QM9: the percentage of conformers whose precision RMSD improves vs degrades by more than a tolerance (0.02–0.20 Å) after refinement.
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 14, Table 7 caption (method description in App. A.4, PDF p. 15)
- Exact quote:
  > "Table 7: Improvement rate (IR) and Downgrade rate (DR) (%) computed by RMSD-precision with a relative tolerance [δ] (Å) on GEOM-QM9."
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-02
- Confidence: high

### E-EVAL-010
- Claim: ET-Flow reports coverage as a function of threshold and states that its advantage over TD is largest at low thresholds (GEOM-DRUGS).
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 7, Figure 3 caption
- Exact quote:
  > "ET-Flow outperforms TorsionDiff by a large margin especially in a lower threshold region."
- Shows vs infers: SHOWS (DRUGS only).
- Revised after V1 note on E-EVAL-010 (P2): original side note "INFERENCE that the same holds on QM9 is supported by our own sweeps, E-EVAL-038/040" WITHDRAWN; our sweeps contain no ET-Flow run, so the QM9 low-threshold contrast rests only on FM-refiner's re-runs (E-EVAL-008) vs our TD sweep (different evaluators).
- Supports card(s): C-EVAL-01
- Confidence: high

### E-EVAL-011
- Claim (Revised after D-007): In EnFlow, *selection* by a learned energy (3K candidates generated, the 2K with the lowest learned energy kept; no relaxation) lowers recall and raises precision for EnFlow-SO(3) at 50 steps on GEOM-QM9: COV-R 96.26 → 90.91 %, AMR-R 0.076 → 0.119 Å, COV-P 95.48 → 96.15 %, AMR-P 0.083 → 0.069 Å.
- Original claim (superseded): "In EnFlow, energy-based selection of EnFlow-SO(3) samples (50 steps, GEOM-QM9) lowers recall and raises precision: ..." (mechanism not stated; C-EVAL-05 misused it as relaxation evidence).
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_enflow.pdf, PDF p. 5, Table 1 and its "Table notes" (unselected EnFlow-SO(3) 50-step row identified from the non-layout text order; the EnergySel row carries its own label)
- Exact quote:
  > "EnFlow-SO(3)-EnergySel 50 90.91 100.00 0.119 0.021 96.15 100.00 0.069 0.021" … "50 96.26 100.00 0.076 0.028 95.48 100.00 0.083 0.034" … "3K candidate conformations are generated and the 2K conformations with the lowest learned energy are retained"
- Shows vs infers: SHOWS (row mapping for the unselected row: medium confidence)
- Supports card(s): C-EVAL-05
- Confidence: medium

---------------------------------------------------------------------------------------------------------------------
## B. What the GEOM-QM9 reference actually is

### E-EVAL-012
- Claim: GEOM conformers are CREST metadynamics structures optimised with GFN2-xTB.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf (arXiv 2006.05531v4), PDF p. 3, section "CREST"
- Exact quote:
  > "Geometries from the MTD runs are then optimized with GFN2-xTB."
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05, C-EVAL-06
- Confidence: high

### E-EVAL-013
- Claim: For GEOM-QM9, the DFT geometries of QM9 were re-optimised with xTB before seeding CREST, i.e. the QM9 reference ensembles are at the GFN2-xTB level, not the QM9 DFT level.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 5, Methods "Conformer generation", sub-heading "Initial structure generation" (locator Revised after V1 verdict on E-EVAL-013; gas-phase CREST defaults: E-EVAL-055)
- Exact quote:
  > "since it is recommended to seed CREST with a structure optimized at the GFN2-xTB level of theory, we re-optimized each QM9 geometry with xTB before using it in CREST."
- Shows vs infers: SHOWS (closes the round-2 gap "for QM9 UNVERIFIED", round2/verify_A.md E14)
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05, C-EVAL-06
- Confidence: high

### E-EVAL-014
- Claim: CREST keeps conformers up to an energy window of 6.0 kcal/mol by default.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 3, section "CREST"
- Exact quote:
  > "The default Ewin = 6.0 kcal/mol provides a safety net around errors in the xTB energies"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-03
- Confidence: high

### E-EVAL-015
- Claim: Re-optimising CREST (xTB) geometries with DFT moves them by a mean RMSD of 0.36 Å (median heavy-atom RMSD 0.25 Å), measured on BACE drug-like molecules in implicit water.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 8, Technical Validation, Figure 5 discussion
- Exact quote:
  > "the geometries change very little during optimization, with a mean RMSD of only 0.36 Å." … "The median RMSD among heavy atoms is 0.25 Å"
- Shows vs infers: SHOWS for BACE/water. INFERENCE (not shown): the xTB-vs-DFT geometric gap for small gas-phase QM9 molecules is probably smaller, but it is not measured; AMR-R differences of a few hundredths of an Å therefore measure agreement with GFN2-xTB minima, not with a higher-level "true" geometry.
- Supports card(s): C-EVAL-03, C-EVAL-06
- Confidence: high (quote); low (QM9 transfer)

### E-EVAL-016
- Claim: The GEOM authors present GEOM as a recall/diversity benchmark but state that CREST conformer weights are inaccurate.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 2, Background & Summary
- Exact quote:
  > "Hence GEOM is an excellent benchmark for the recall and diversity of conformer generation methods. However, the CREST statistical weights for each conformer are rather inaccurate. Therefore, benchmarks that include conformer probabilities should use the DFT weights provided in GEOM."
- Shows vs infers: SHOWS. Revised after D-001 (P2): quote extended with GEOM's own recommendation; the DFT weights exist only for the BACE subset (E-EVAL-059).
- Supports card(s): C-EVAL-06
- Confidence: high

### E-EVAL-017
- Claim: Graph re-identification of CREST conformers succeeded for 88.4 % of QM9 molecules; the failed QM9 graphs had undergone a reaction during CREST.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 7, Technical Validation
- Exact quote:
  > "The graph re-attribution procedure succeeded for 88.4% of the QM9 molecules" … "All of the failed QM9 graphs underwent some sort of reaction, which can be explained by the presence of highly strained and unstable molecules."
- Shows vs infers: SHOWS.
- Revised after D-006: original INFERENCE "strained QM9 molecules are also where ETKDG fails (our 60 cage failures); reference quality and seed availability are both worst on the same strained subset" WITHDRAWN. Nothing links GEOM's reacted graphs to our 60 cages, and molecules with changed graphs are largely removed by TD's SMILES filter (V1).
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05
- Confidence: high (quote); medium (inference)

---------------------------------------------------------------------------------------------------------------------
## C. Energy and local-geometry metrics

### E-EVAL-018
- Claim: GFN2-xTB re-optimisation of GFN2-xTB-optimised GEOM conformers gives a mean relaxation energy close to zero, whereas MMFF evaluation of the same structures gives about 16 kcal/mol (GEOM-Drugs).
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 10
- Exact quote:
  > "For conformers optimized with GFN2-xTB (as in GEOM-Drugs), the mean relaxation energy difference Erelax when re-optimized with GFN2-xTB is close to zero, as expected. In contrast, the same structures evaluated with MMFF show a mean Erelax of around 16 kcal/mol"
- Shows vs infers: SHOWS (GEOM-Drugs). INFERENCE: the same holds for GEOM-QM9 because its references are also GFN2-xTB minima (E-EVAL-013), so E_relax(GFN2-xTB) of a generated conformer is a reference-free measure of how far it is from an xTB minimum.
- Supports card(s): C-EVAL-03, C-EVAL-05
- Confidence: high

### E-EVAL-019
- Claim: Nikitin et al. recommend measuring differences in bond lengths, bond angles and torsions between each generated structure and its own GFN2-xTB-optimised counterpart.
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 11, "GFN2-xTB energy-based geometry benchmark"
- Exact quote:
  > "we suggest to assess differences in bond lengths, bond angles, and torsion angles of generated and optimized counterparts."
- Shows vs infers: SHOWS. Note (round 2 verify_B #22): the comparison is to the generated structure's own xTB-optimised counterpart, NOT to a GT conformer; `tools/geometry_metrics.py` (vs matched GT) is a different metric.
- Reused from: round2/research_B.md table row "Recommended local-geometry metrics", verified in round2/verify_B.md #22 (VERIFIED quote / PARTIAL SUPPORT); quote re-checked here.
- Supports card(s): C-EVAL-02, C-EVAL-03
- Confidence: high

### E-EVAL-020
- Claim: Nikitin et al. find that diffusion-based generators already surpass MMFF in structural precision against GFN2-xTB; in their Table 2, MMFF-optimised structures relaxed with GFN2-xTB move by 1.12 × 10⁻² Å (bonds), 1.22° (angles), 4.89° (torsions) and release a mean E_relax of 11.4 kcal/mol (GEOM-Drugs).
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 12 (text) and PDF p. 13, Table 2 (numbers read column-wise from the non-layout text; row order GEOM-Drugs, MMFF→GFN2-xTB, EQGAT-diff, JODO, Megalodon, SemlaFlow, FlowMol2, Megalodon-flow)
- Exact quote:
  > "These results clearly demonstrate that diffusion-based models already surpass MMFF in structural precision."
- Shows vs infers: SHOWS (text). Table numbers: SHOWS, medium confidence because the table layout is scrambled in extraction.
- Revised after D-007: the same Table 2 shows the diffusion models' torsion deviations (EQGAT-diff 8.58°, JODO 6.01°, Megalodon 5.58°) are *larger* than MMFF→xTB's 4.89°, so "surpass MMFF" holds for bonds, angles and E_relax, not for torsions (column order as above; values confirmed by V1).
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05
- Confidence: high (quote); medium (table values)

### E-EVAL-021
- Claim: Nikitin et al. state that MMFF94 energy evaluation is not suitable for models trained on GFN2-xTB-optimised data.
- Source type: paper
- Locator:
  - paper: papers/related/2025_nikitin_geom_drugs_revisited.pdf, PDF p. 3, Introduction
- Exact quote:
  > "the use of energy evaluations at inappropriate levels of theory, such as MMFF94, which is not suitable for assessing models trained on GFN2-xTB-optimized data."
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05
- Confidence: high

### E-EVAL-022
- Claim: In TD's ensemble-property test on GEOM-DRUGS, TD's unrelaxed conformers have a median E_min error of 36.94 kcal/mol (RDKit 39.14, OMEGA 16.45), which drops to 0.13 kcal/mol after GFN2-xTB relaxation.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 27, Table 10 (columns: without relaxation E, μ, Δε, E_min; with relaxation E, μ, Δε, E_min)
- Exact quote:
  > "Tor. Diff. 36.91 0.92 4.93 36.94 0.22 0.35 0.54 0.13"
- Shows vs infers: SHOWS (DRUGS). INFERENCE (Revised after V1 note, P2): unrelaxed energy errors are huge for every method in the table, including GeoMol, which predicts its own local structure (E_min 43.68); so the table shows that energies are very sensitive to small local-structure errors, not that RDKit's frozen L is the unique cause. Original wording "dominated by frozen RDKit local structure" withdrawn.
- Supports card(s): C-EVAL-03, C-EVAL-05, C-EVAL-06
- Confidence: high

### E-EVAL-023
- Claim (Revised after D-001): TD states that without relaxation the property errors of all methods are too large to be chemically useful, and that after relaxation the errors from global flexibility "become important".
- Original claim (superseded): "... and that after relaxation the errors from global flexibility dominate." ("dominate" is not in the source.)
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 28, App. H "Ensemble properties"
- Exact quote:
  > "For all methods, the errors without relaxation are far too large for the computed properties to be chemically useful" … "relaxation of local structures is necessary for any method, after which errors from global flexibility become important."
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-03, C-EVAL-05, C-EVAL-06
- Confidence: high

### E-EVAL-024
- Claim: TD's ensemble-property protocol uses a random 100-molecule DRUGS subset, min(2K, 32) conformers per molecule, and GFN2-xTB relaxation before computing Boltzmann-weighted properties.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 9, Sec. 4.4
- Exact quote:
  > "For a random 100-molecule subset of DRUGS, we generate min(2K, 32) conformers per molecule, relax the conformers with GFN2-xTB"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-05, C-EVAL-06
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## D. Cheap baselines and their caveats

### E-EVAL-025
- Claim: Zhou et al.'s "RDKit + Clustering" baseline mixes random-dihedral, ETKDG and ETKDG+MMFF samplers 1:1:4, draws N_e = min(20 N_ref, 2000) energy samples, and K-means-clusters them into 2 N_ref outputs.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhou_dl_conformation_critique.pdf, PDF p. 4, Sec. 3
- Exact quote:
  > "used in a ratio of 1:1:4, respectively. The number of energy samples, denoted by Ne, is determined by the formula Ne = min(20Nref , 2000)" … "we employ the K-means algorithm with 2Nref clusters"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-04
- Confidence: high

### E-EVAL-026
- Claim: On the GeoDiff/ConfGF QM9 protocol (200 molecules, not TD's split), RDKit + Clustering reaches COV 97.65 % and MAT 0.1902 Å versus plain RDKit (ETKDG + MMFF) 83.26 % and 0.3447 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhou_dl_conformation_critique.pdf, PDF p. 5, Table 1
- Exact quote:
  > "RDKit 83.26 90.78 0.3447 0.2935" … "RDKit + Clustering 97.65 100.00 0.1902 0.1818"
- Shows vs infers: SHOWS. Not comparable to TD-protocol numbers (different split, 200 molecules, conformer caps); see papers/INDEX.md protocol codes.
- Supports card(s): C-EVAL-04
- Confidence: high

### E-EVAL-027
- Claim: In Zhou et al.'s ablation, shrinking the energy-sampler budget to 2 N_ref raises MAT to 0.2223 Å, and removing the MMFF energy sampler raises MAT to 0.2511 Å (QM9, GD protocol): the gain comes from oversampling and from MMFF.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhou_dl_conformation_critique.pdf, PDF p. 5, Table 2
- Exact quote:
  > "2Nref 91.23 96.87 0.2223 0.2171" … "w/o Energy sampler 98.01 100.00 0.2511 0.2434"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-04
- Confidence: high

### E-EVAL-028
- Claim: Zhang et al. ("Infinite Physical Monkey") argue the RDKit + Clustering comparison is unfair unless the deep models are also given ~2000 samples followed by clustering.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhang_infinite_physical_monkey.pdf (arXiv 2304.10494), PDF p. 5 (no printed page numbers)
- Exact quote:
  > "indicating that the comparison in the previous is unfair. At the very least, they should also sample 2000 conformations for DL methods followed by clustering."
- Shows vs infers: SHOWS (argument). Revised after V1 note (P2): original side note "their Table 2 on PDF p. 9 reproduces Zhou's QM9 numbers" corrected: only the RDKit + Clustering row (97.65 / 0.1902) matches; their RDKit and CGCF rows differ from Zhou's.
- Supports card(s): C-EVAL-04
- Confidence: high

### E-EVAL-029
- Claim (Revised after D-002): A practitioner blog (Rowan) reports that ETKDG often misses conformers or generates high-energy conformers *for large systems (e.g. PROTACs)* and, unscoped, mentions an anecdotal "curious preference for twist boats over chairs"; it says these problems "can be ameliorated" by generating thousands of conformers and deduplicating, at a cost in speed and redundancy. Only the twist-boat anecdote is relevant to QM9-size molecules.
- Original claim (superseded): "... reports that ETKDG misses conformers, produces high-energy conformers, and has 'random pathologies' such as preferring twist boats over chairs; oversampling plus deduplication helps but is slow and redundant." (dropped the large-system scope)
- Source type: blog
- Locator:
  - blog: https://rowansci.substack.com/p/openconf-and-other-open-source-projects, Rowan; visible byline Corin Wagen, Nick Casetti, Jonathon Vandezande, Eli Mann (meta author Corin Wagen; byline per V1, D-002), 2026-06-17, section "openconf: Rapid Monte Carlo–Based Conformer Generation", accessed 2026-10-08, snapshot papers/blogs/rowan_openconf_2026.txt
- Exact quote:
  > "ETKDG often misses conformers or generates high-energy conformers for large systems (e.g. PROTACs) and has additional random pathologies (e.g. a curious preference for twist boats over chairs)."
- Shows vs infers: SHOWS (practitioner report, no numbers; evidence for practice, not proof). INFERENCE (weak, anecdote only): ring-pucker errors in ETKDG seeds are a known practitioner complaint, consistent with our ring-heavy floor; the large-system statements do not transfer to QM9.
- Supports card(s): C-EVAL-04
- Confidence: medium

### E-EVAL-030
- Claim: RDKit changed the default of EmbedMolecule()/EmbedMultipleConfs() from ETKDGv1 to ETKDGv3 in release 2024.03.1.
- Source type: blog (release notes; technical source)
- Locator:
  - blog: https://raw.githubusercontent.com/rdkit/rdkit/master/ReleaseNotes.md, RDKit developers, Release_2024.03.1, section "Backwards incompatible changes", accessed 2026-10-08, snapshot papers/blogs/rdkit_release_notes_2024_03.txt
- Exact quote:
  > "the functions EmbedMolecule() and EmbedMultipleConfis() now use ETKDGv3 by default (previously they were using ETKDGV1)"
- Shows vs infers: SHOWS ("Confis" typo is in the source)
- Supports card(s): C-EVAL-04
- Confidence: high

### E-EVAL-031
- Claim: TD seeds are embedded with `EmbedMultipleConfs` without a parameter object; on our RDKit 2022.9.5 this is the pre-2024.03 default, i.e. ETKDGv1, not ETKDGv3.
- Source type: code
- Locator:
  - code: torsional-diffusion/generate_confs.py:64-68 (embed_func); RDKit version in RESULTS_QM9.md:9
- Exact quote:
  > "AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=5," / "randomSeed=args.seed + 1 if args.seed is not None else -1)" … RESULTS_QM9.md:9 "**Software:** RDKit 2022.9.5"
- Shows vs infers: INFERENCE (ETKDGv1 follows from E-EVAL-030 + the version line; the parameter values on the cluster build were not printed)
- Supports card(s): C-EVAL-04
- Confidence: medium

### E-EVAL-032
- Claim: conda-forge provides linux-64 builds of the xtb program (latest 6.7.1), so a GFN2-xTB binary can be installed without compiling.
- Source type: blog (package metadata snapshot)
- Locator:
  - blog: https://api.anaconda.org/package/conda-forge/xtb, anaconda.org (conda-forge), JSON fields `latest_version` and `files[].basename`, accessed 2026-10-08, snapshot papers/blogs/condaforge_xtb_metadata_eval.txt (Revised after D-003: new verbatim snapshot by SCOUT EVAL; the earlier snapshot papers/blogs/condaforge_xtb_metadata.txt by SCOUT GEOM lists only 6.4.1 onward and is no longer cited here)
- Exact quote (JSON values, verbatim):
  > "latest_version": "6.7.1" ; "linux-64/xtb-6.2.3-h323e27b_0.tar.bz2" … (28 linux-64 files; versions 6.2.3, 6.3.0–6.3.3, 6.4.0, 6.4.1, 6.5.0, 6.5.1, 6.6.0, 6.6.1, 6.7.1)
- Original quote (superseded, QUOTE MISMATCH per V1): "linux-64 versions: ['6.4.1', '6.5.0', '6.5.1', '6.6.0', '6.6.1', '6.7.1']" (a derived, incomplete list).
- Shows vs infers: SHOWS availability, including xTB 6.2.3, the version GEOM used (E-EVAL-056), so the reference level can be matched exactly. Not shown: that it installs/runs on gnode118 (round 2 never ran it, round2/SHORTLIST.md:49).
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05
- Confidence: high (availability); UNVERIFIED (cluster install)

---------------------------------------------------------------------------------------------------------------------
## E. Our code

### E-EVAL-033
- Claim: Our evaluator counts a molecule with no generated conformers as 0 % coverage but leaves it out of AMR (nanmean), so AMR-R is averaged over successful molecules only.
- Source type: code
- Locator:
  - code: torsional-diffusion/evaluate_confs.py:166-171 (COV with `[0] * num_failures` at :166; `np.nanmean(amr_recall)` at :171; the printed block uses the same convention at :152-155). Revised after D-004 (was :168-171).
- Exact quote:
  > "_cr = [float(np.mean(np.min(r['rmsd'], axis=1) < report_thr)) for r in results.values()] + [0] * num_failures" … "'MAT-R_mean': float(np.nanmean(amr_recall))"
- Shows vs infers: SHOWS. INFERENCE (Revised after D-006): TD's AMR-R excludes the ~65 molecules ETKDG cannot embed; whether this flatters TD relative to Cartesian models is NOT established, because those models also evaluate reduced sets (995 molecules, E-EVAL-005) and ET-Flow reports RDKit-related failures on GEOM-XL (E-EVAL-058). Original wording "that ML Cartesian generators must still score, which flatters TD" withdrawn.
- Supports card(s): C-EVAL-01
- Confidence: high

### E-EVAL-034
- Claim: The TD code base already contains a GFN2-xTB geometry-optimisation helper that calls the `xtb` binary with `--opt <level>` and writes the optimised coordinates back into the RDKit conformer.
- Source type: code
- Locator:
  - code: torsional-diffusion/utils/xtb.py:43-63
- Exact quote:
  > "def xtb_optimize(mol, level, path_xtb):" … "cmd = [path_xtb, in_path, \"--opt\", level]"
- Shows vs infers: SHOWS (round 2 verify_A.md also VERIFIED `utils/xtb.py:44`)
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05
- Confidence: high

### E-EVAL-035
- Claim: A post-generation MMFF hook exists (`--post_mmff` → `pyg_to_mol(..., mmff=True)`), which is the place an xTB post-relaxation would plug in.
- Source type: code
- Locator:
  - code: torsional-diffusion/diffusion/sampling.py:249-261 and torsional-diffusion/generate_confs.py:19, :170
- Exact quote:
  > "if mmff:" / "AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')"
- Shows vs infers: SHOWS (hook location); INFERENCE (xTB variant would go here or in a separate CPU post-processing script on confs.pkl)
- Supports card(s): C-EVAL-05
- Confidence: high

### E-EVAL-036
- Claim: `tools/geometry_metrics.py` computes bond-length, bond-angle and heavy-torsion MAE of each generated conformer against its RMSD-matched GT conformer, and round-1 `slurm/analysis_qm9.sbatch` calls it, but no `geometry.csv` output is present in the synced results.
- Source type: code
- Locator:
  - code: tools/geometry_metrics.py:1-14 (docstring); slurm/analysis_qm9.sbatch:65; absence checked with `find cluster_sync -name "geometry*"` (no hits) on 2026-10-08
- Exact quote:
  > "Local-geometry error of GENERATED conformers against their matched ground-truth conformers" … "With torsional diffusion the generated bond lengths/angles are EXACTLY those of the ETKDG seed" … "python \"$TOOLS/geometry_metrics.py\" --results"
- Shows vs infers: SHOWS (tool + call). UNVERIFIED whether the outputs exist on the cluster (not synced, not reported anywhere in RESULTS_QM9.md or round 2 files).
- Supports card(s): C-EVAL-02
- Confidence: high (tool); low (whether it ran)

### E-EVAL-037
- Claim: `tools/paired_compare.py` already supports thresholds 0.05/0.1/0.25/0.5 Å and a success-intersection molecule universe for paired bootstrap comparisons.
- Source type: code
- Locator:
  - code: tools/paired_compare.py:32, :41, :105-106
- Exact quote:
  > "parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.05, 0.1, 0.25]," … "parser.add_argument('--universe', choices=['union', 'intersection'], default='union')"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-02
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## F. Our data

### E-EVAL-038
- Claim: Baseline TD (released model, RDKit L, sampling seed 0) has AMR-R 0.1752 Å on 935 evaluated molecules with 65 failures, and COV-R@0.05 = 3.91 %, COV-R@0.1 = 37.66 %.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, lines `qm9_default/R0_base_seed0 ...` and `qm9_default/R0_base_seed0/evaluate.log:SWEEP thr=0.050 / 0.100`
- Exact quote:
  > "qm9_default/R0_base_seed0 threshold=0.5000 COV-R_mean=89.0293 COV-R_median=100.0000 MAT-R_mean=0.1752 ... n_evaluated=935 n_model_failures=65 n_additional_failures=1" ; "SWEEP thr=0.050 COV-R_mean=3.91 COV-R_median=0.00 COV-P_mean=2.62"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-02, C-EVAL-03, C-EVAL-04, C-EVAL-05
- Confidence: high

### E-EVAL-039
- Claim: RDKit ETKDG conformers alone (no model) give AMR-R 0.2298 Å and COV-R@0.05 = 3.05 %.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, lines `qm9_default/A0_rdkit_etkdg_only ...`
- Exact quote:
  > "qm9_default/A0_rdkit_etkdg_only threshold=0.5000 COV-R_mean=83.9217 COV-R_median=100.0000 MAT-R_mean=0.2298" ; "SWEEP thr=0.050 COV-R_mean=3.05 COV-R_median=0.00"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-04
- Confidence: high

### E-EVAL-040
- Claim: MMFF relaxation of the ETKDG seed before diffusion (A2, released model) gives AMR-R 0.1507 Å and raises COV-R@0.05 from 3.9 % to 17.89 % and COV-R@0.1 to 49.96 %.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, lines `qm9_default/A2_pre_mmff ...`
- Exact quote:
  > "qm9_default/A2_pre_mmff threshold=0.5000 COV-R_mean=88.9339 COV-R_median=100.0000 MAT-R_mean=0.1507" ; "SWEEP thr=0.050 COV-R_mean=17.89 COV-R_median=0.00" ; "SWEEP thr=0.100 COV-R_mean=49.96 COV-R_median=50.00"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-03, C-EVAL-04
- Confidence: high

### E-EVAL-041
- Claim: MMFF relaxation after diffusion (A3) worsens recall (AMR-R 0.1863 Å) but improves precision (AMR-P 0.1641 Å vs 0.2191 for R0).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, line `qm9_default/A3_post_mmff threshold=...`
- Exact quote:
  > "qm9_default/A3_post_mmff threshold=0.5000 COV-R_mean=85.7152 COV-R_median=100.0000 MAT-R_mean=0.1863 MAT-R_median=0.1490 COV-P_mean=87.3408 COV-P_median=100.0000 MAT-P_mean=0.1641"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-05
- Confidence: high

### E-EVAL-042
- Claim: With true L (random GT conformer, ORACLE), the released model reaches AMR-R 0.0801 Å and COV-R@0.05 = 66.72 %.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, lines `qm9_default/A1_gtL_model ...`
- Exact quote:
  > "qm9_default/A1_gtL_model threshold=0.5000 COV-R_mean=96.1876 COV-R_median=100.0000 MAT-R_mean=0.0801" ; "SWEEP thr=0.050 COV-R_mean=66.72 COV-R_median=77.78"
- Shows vs infers: SHOWS (ORACLE)
- Supports card(s): C-EVAL-01, C-EVAL-05
- Confidence: high

### E-EVAL-043
- Claim: S3 (50/50 RDKit/true-L training, seed 0) gives AMR-R 0.1816 Å and COV-R@0.05 = 3.56 % on RDKit L, and AMR-R 0.0331 Å, COV-R@0.05 = 85.27 % on true L cycled (ORACLE).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/qm9_S3_mixL_p0.5_e100_s0/steps20_seed0/summary.txt and .../steps20_seed0_gtLcycle/summary.txt
- Exact quote:
  > "SUMMARY threshold=0.5000 COV-R_mean=88.7259 COV-R_median=100.0000 MAT-R_mean=0.1816" ; "SWEEP thr=0.050 COV-R_mean=3.56" ; (gtLcycle) "SUMMARY threshold=0.5000 COV-R_mean=98.6050 COV-R_median=100.0000 MAT-R_mean=0.0331" ; "SWEEP thr=0.050 COV-R_mean=85.27"
- Shows vs infers: SHOWS (gtLcycle = ORACLE)
- Supports card(s): C-EVAL-01, C-EVAL-03, C-EVAL-06
- Confidence: high

### E-EVAL-044
- Claim: With L blended 75 % of the way to the cycled true L (ORACLE), CTRL_rematch s0 gives AMR-R 0.1164 Å and B1 s0 0.1158 Å (906 molecules evaluated, 94 failures); at λ = 1.00 CTRL_rematch s0 gives 0.0803 Å and B1 s0 0.0199 Å (955 molecules, 45 failures), so the two λ levels are scored on different molecule sets.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s0/S1_lam0.75_cyc_ORACLE/summary.txt; .../qm9_CTRL_rematch_100ep_e100_s0/S1_lam1.00_cyc_ORACLE/summary.txt; cluster_sync/round2/results/qm9_B1_train_gtL_e100_s0/S1_lam0.75_cyc_ORACLE/summary.txt; .../qm9_B1_train_gtL_e100_s0/S1_lam1.00_cyc_ORACLE/summary.txt
- Exact quote:
  > (CTRL λ0.75) "SUMMARY threshold=0.5000 COV-R_mean=86.2270 COV-R_median=100.0000 MAT-R_mean=0.1164" ; (CTRL λ1.00) "SUMMARY threshold=0.5000 COV-R_mean=92.2239 COV-R_median=100.0000 MAT-R_mean=0.0803" ; (B1 λ0.75) "SUMMARY threshold=0.5000 COV-R_mean=85.3794 COV-R_median=100.0000 MAT-R_mean=0.1158" ; (B1 λ1.00) "SUMMARY threshold=0.5000 COV-R_mean=95.1646 COV-R_median=100.0000 MAT-R_mean=0.0199"
- Molecule counts (same files): "n_evaluated=906 n_model_failures=94" (both λ0.75 files); "n_evaluated=955 n_model_failures=45" (both λ1.00 files).
- Shows vs infers: SHOWS (ORACLE). 3-seed means in round3/BRIEF.md: 0.116 / 0.115 / 0.020.
- Supports card(s): C-EVAL-01
- Confidence: high

### E-EVAL-045
- Claim: For B1 (seed 0), true L taken from the GT conformer the sample is cycled to gives AMR-R 0.0215 Å, but true L from a random GT conformer gives 0.0374 Å (both ORACLE); for CTRL_base s0 the two are 0.0810 vs 0.0823.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, lines `qm9_B1_train_gtL_e100_s0/steps20_seed0_gtLcycle`, `..._gtL`, `qm9_CTRL_base_100ep_e100_s0/steps20_seed0_gtLcycle`, `..._gtL`
- Exact quote:
  > "qm9_B1_train_gtL_e100_s0/steps20_seed0_gtLcycle threshold=0.5000 COV-R_mean=98.9002 COV-R_median=100.0000 MAT-R_mean=0.0215" ; "qm9_B1_train_gtL_e100_s0/steps20_seed0_gtL threshold=0.5000 COV-R_mean=97.9257 COV-R_median=100.0000 MAT-R_mean=0.0374"
- Shows vs infers: SHOWS. INFERENCE: own-conformer L carries conformer identity (e.g. ring pucker), so "cycled" oracles over-state what any independently sampled L can reach; the random-GT-conformer oracle is the fairer ceiling.
- Supports card(s): C-EVAL-01, C-EVAL-02
- Confidence: high (numbers); medium (inference)

### E-EVAL-046
- Claim: 65 of 1000 test molecules produce no TD conformers (60 strained cages ETKDG cannot embed, 5 rejected inputs); the paper's own released samples, scored with the repo evaluator, give COV-R 88.8 % vs the published 92.8 %.
- Source type: our-data
- Locator:
  - our-data: RESULTS_QM9.md, §2.2 "Coverage is lower than the paper because of RDKit failures"
- Exact quote:
  > "**60 molecules:** RDKit's ETKDG cannot embed strained polycyclic cages" … "**So the paper's 92.8% cannot be reproduced from its own released samples.**"
- Shows vs infers: SHOWS (round-1 verified result)
- Supports card(s): C-EVAL-01, C-EVAL-04
- Confidence: high

### E-EVAL-047
- Claim: ETKDG seeds differ from their assigned GT conformer by bond RMSD 0.036 Å and angle RMSD 3.88°, versus 0.004 Å and 1.67° between different GT conformers of the same molecule; the best-seed RDKit-L floor is 0.0986 Å averaged over conformers but 0.1145 Å averaged per molecule.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/local_structure_test.log, lines `floor_best_sym`, `bond_rmsd_assigned`, `angle_rmsd_assigned`, `bond_rmsd_gtgt`, `angle_rmsd_gtgt`
- Exact quote:
  > "floor_best_sym: n=7554 mean=0.0986 median=0.0606 per-molecule-mean=0.1145" ; "bond_rmsd_assigned: mean=0.0360" ; "angle_rmsd_assigned: mean=3.8821" ; "bond_rmsd_gtgt: mean=0.0040" ; "angle_rmsd_gtgt: mean=1.6735"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-02, C-EVAL-03
- Confidence: high

### E-EVAL-048
- Claim: [EVAL-calc] Molecules with 1–3 GT conformers are 48.1 % of the test molecules (macro-average weight) but only 10.2 % of GT conformers; 97–98 % of them contain a ring and their RDKit-L floor is the highest (0.136 / 0.132 Å vs 0.089 Å for molecules with > 10 conformers).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/local_structure_test.csv (columns `floor_best_sym`, `n_ring_atoms`, `n_torsions_heavy`), computed by scratchpad `macro_micro.py` (group by SMILES, bin by number of rows)
- Exact quote (script output):
  > "1 225 0.1364 0.9822 0.4267" ; "2-3 227 0.1315 0.9736 0.8987" ; ">10 213 0.0888 0.6573 2.7465" ; "share of molecules with n_gt<=3: 0.481 ; share of conformers from them: 0.102"
- Shows vs infers: SHOWS (on the 940 molecules / 7554 GT conformers in that file; V1 reproduced the numbers).
- Revised after D-005: original INFERENCE "the macro-averaged AMR-R we report is about half determined by rigid, ring-containing molecules where the torsion model has little to do, so it is mostly a ring-L metric" WITHDRAWN. Few-conformer (n_gt ≤ 3) does not mean rigid: only 52.2 % of them have 0 heavy torsions; rigid molecules are 30.4 % of molecules and 26.0 % of the macro AMR-R sum (E-EVAL-057). Replacement INFERENCE: the per-molecule average gives few-conformer, ring-containing molecules ~5× the weight they have per conformer (48 % vs 10 %), so per-molecule and per-conformer averages can rank arms differently; the ring-L argument rests on E-EVAL-057, not on this entry.
- Supports card(s): C-EVAL-02, C-EVAL-04
- Confidence: high (calc); medium (inference)

### E-EVAL-049
- Claim: [EVAL-calc] Mean per-seed heavy-atom L error (bond RMSD Å / angle RMSD ° / endocyclic ring-dihedral RMSD °): ETKDG 0.0314 / 2.80 / 10.65; MMFF-relaxed ETKDG 0.0182 / 1.97 / 8.07; λ = 0.75 blend 0.0097 / 0.97 / 1.56 (ORACLE); λ = 0.50 0.0186 / 1.93 / 3.16 (ORACLE).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/data/QM9/round2_seeds/l_error.csv, `groupby('cond')` mean of `bond_rmsd_heavy_any`, `angle_rmsd_heavy_any`, `ring_dihedral_rmsd` over all rows (scratchpad `lerr_eval.py`)
- Exact quote (computed output rows):
  > "L_etkdg2L 0.0314 2.7963 ... 10.6522" ; "L_etkdg2L_mmff 0.0182 1.9733 ... 8.0655" ; "L_lam0.75_cyc_ORACLE 0.0097 0.9668 ... 1.5643"
- Shows vs infers: SHOWS (row means, not per-molecule means). INFERENCE: MMFF fixes bonds/angles to λ≈0.5 level but leaves ring dihedrals near ETKDG level (independently also reported by SCOUT GEOM).
- Supports card(s): C-EVAL-01, C-EVAL-02, C-EVAL-03, C-EVAL-04
- Confidence: high

### E-EVAL-050
- Claim: The retrained CTRL_rematch s0 model on MMFF-relaxed ETKDG L reproduces the A2 gain: AMR-R 0.1518 Å vs 0.1774 Å on raw ETKDG L, COV-R@0.05 16.62 % vs 4.07 %.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s0/S1_etkdg2L_mmff/summary.txt and .../S1_etkdg2L/summary.txt
- Exact quote:
  > (mmff) "SUMMARY threshold=0.5000 COV-R_mean=89.2386 COV-R_median=100.0000 MAT-R_mean=0.1518" ; "SWEEP thr=0.050 COV-R_mean=16.62" ; (etkdg) "MAT-R_mean=0.1774" ; "SWEEP thr=0.050 COV-R_mean=4.07"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-01, C-EVAL-03, C-EVAL-04, C-EVAL-06
- Confidence: high

### E-EVAL-051
- Claim: Zhou et al. argue that GEOM's semi-empirical (GFN2-xTB) reference accuracy is insufficient for applications that need DFT-quality conformers, and that conformer generation should be benchmarked by downstream needs.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhou_dl_conformation_critique.pdf, PDF p. 6, Discussion
- Exact quote:
  > "for quantum property prediction, we require conformations that are close to the DFT-optimized conformations. The accuracy of GEOM's semi-empirical DFT is insufficient to meet these requirements."
- Shows vs infers: SHOWS (an opinion, no measurement)
- Supports card(s): C-EVAL-06
- Confidence: high (quote); low (as evidence of size of the effect)

### E-EVAL-052
- Claim: Round-1 metric verification fixed `tools/geometry_metrics.py` to skip ill-conditioned sp-centre torsions and to choose the gen→GT atom map by aligned heavy-atom RMSD.
- Source type: code
- Locator:
  - code: review/metric_verification.md:140 (row B3); the fix is marked `[metric-verifier fix]` in tools/geometry_metrics.py (e.g. the comment above the sp-centre skip in `topology()`)
- Exact quote:
  > "| B3 | medium | `tools/geometry_metrics.py`: torsion MAE included ill-conditioned sp-centre dihedrals, and the atom map was chosen by angle MAE | skip bonds with an sp endpoint; choose the map by aligned heavy-atom RMSD |"
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-02
- Confidence: high

### E-EVAL-053
- Claim: Round 1 interpreted A3 (MMFF after diffusion) as relaxation pulling conformers into nearby minima, making them more precise but less diverse, and concluded that local structure must be right during generation.
- Source type: our-data
- Locator:
  - our-data: RESULTS_QM9.md:68-69, §2.5 "Relaxing *after* diffusion does not help recall (A3)"
- Exact quote:
  > "Relaxation pulls conformers into nearby energy minima, so they become more precise but less diverse." … "local structure has to be right *during* generation, not patched afterwards."
- Shows vs infers: SHOWS the round-1 interpretation (itself an INFERENCE from E-EVAL-041; MMFF-vs-xTB was not separated)
- Supports card(s): C-EVAL-05
- Confidence: high (that it was said); medium (that it is right)

### E-EVAL-054
- Claim: ET-Flow evaluates ensemble properties like TD: generated conformers (≥ 2K, ≤ 32 per molecule, 100 GEOM-DRUGS molecules) are relaxed with GFN2-xTB and Boltzmann-weighted properties are compared with the ground-truth ensemble.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 7, Sec. 4.4 "Ensemble Properties"
- Exact quote:
  > "These conformers are then relaxed using GFN2-xTB (Bannwarth et al., 2019), and the Boltzmann-weighted properties of the generated and ground truth ensembles are compared."
- Shows vs infers: SHOWS
- Supports card(s): C-EVAL-06
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## H. Entries added in P2 (answers to D-001 ... D-007)

### E-EVAL-055
- Claim: GEOM ran CREST with default settings except the molecular charge, i.e. without implicit solvent for the QM9 set (gas-phase GFN2-xTB).
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 5, Methods "CREST simulation"
- Exact quote:
  > "A single xTB-optimized structure was used as input to the CREST simulation of each species. Default values were used for all CREST arguments, except for the charge of each geometry."
- Shows vs infers: SHOWS defaults; "gas phase" is the CREST/xTB default (no solvent flag), confirmed by V1 (implicit water only for BACE, GEOM pp. 2, 6).
- Added in P2 (D-003, V1 note on E-EVAL-013).
- Supports card(s): C-EVAL-03, C-EVAL-05
- Confidence: high

### E-EVAL-056
- Claim: GEOM's initial conformer ensembles were generated with CREST 2.9 and xTB 6.2.3.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 8, "Code availability"
- Exact quote:
  > "CREST version 2.9 was used with xTB version 6.2.3 to generate the initial CREs."
- Shows vs infers: SHOWS. With E-EVAL-032 (xtb 6.2.3 on conda-forge linux-64), the reference xTB version can be matched exactly.
- Added in P2 (D-003).
- Supports card(s): C-EVAL-03, C-EVAL-04, C-EVAL-05
- Confidence: high

### E-EVAL-057
- Claim: [EVAL-calc] Rigid molecules (0 heavy rotatable torsions) are 30.4 % of test molecules and carry 26.0 % of the macro AMR-R sum of CTRL_rematch s0 on ETKDG L (25.4 % on MMFF L; their mean AMR-R is 0.152 Å on ETKDG L and 0.127 Å on MMFF L, vs 0.177 / 0.152 Å over all molecules); few-conformer molecules (n_true ≤ 3) are 48.1 % of molecules and 46.1 % of the sum, and only 52.2 % of them are rigid. With true ring geometry (A5ring, ORACLE) rigid molecules fall from 0.185 Å (λ = 0 base, same 955 molecules) to 0.043 Å and carry 54.9 % of the total ring-oracle gain.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s0/{S1_etkdg2L, S1_etkdg2L_mmff, S1_lam0.00_cyc_ORACLE, S1_A5ring_cyc_ORACLE}/breakdown.log, tables "by n_rot_heavy" and "by n_true"; cluster_sync/results/analysis/local_structure_test.csv; computed by scratchpad `d005_recompute.py` (AMR-R sum = Σ n_bin × MAT_R_bin)
- Exact quote (breakdown.log rows with padding whitespace collapsed, then script output):
  > (S1_etkdg2L) "| 0 | 284 | 96.549 | 0.152 | 95.321 | 0.188 |" ; (S1_etkdg2L_mmff) "| 0 | 284 | 96.54 | 0.127 | 97.155 | 0.127 |" ; (A5ring) "| 0 | 289 | 100 | 0.043 | 99.913 | 0.053 |" ; (λ0.00) "| 0 | 289 | 94.871 | 0.185 | 95.199 | 0.193 |" ; "rigid among few 52.2%" ; "rigid share 54.9%"
- Shows vs infers: SHOWS (ORACLE for A5ring). Same shares as V1's independent scripts (D-005). INFERENCE: ring L matters most for the rigid 30 % of molecules, which carry about a quarter of the macro AMR-R but more than half of the gain from true ring geometry.
- Added in P2 (D-005).
- Supports card(s): C-EVAL-02, C-EVAL-04, C-EVAL-05
- Confidence: high

### E-EVAL-058
- Claim: ET-Flow reports generation failures attributed to RDKit on GEOM-XL (27 molecules), so Cartesian models can also lose molecules in evaluation.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 19, App. D (GEOM-XL results, text after Table 8)
- Exact quote:
  > "we encountered 27 failed cases for generation likely due to RDKit failures, similar to the observations in MCF albeit with slightly different exact numbers."
- Shows vs infers: SHOWS (GEOM-XL only; no failure count is reported for QM9). Used only to say the direction of the failure-accounting bias is unknown.
- Added in P2 (D-006, source suggested by V1).
- Supports card(s): C-EVAL-01
- Confidence: high

### E-EVAL-059
- Claim: GEOM's DFT-level free energies (needed for accurate conformer weights) exist only for the 1,511 BACE species, not for QM9.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 1, Abstract
- Exact quote:
  > "Ensembles of 1,511 species with BACE-1 inhibition data are also labeled with high-quality DFT free energies in an implicit water solvent"
- Shows vs infers: SHOWS (BACE subset). INFERENCE: for QM9 only the CREST (xTB) weights exist, which GEOM calls inaccurate (E-EVAL-016).
- Added in P2 (D-001).
- Supports card(s): C-EVAL-06
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## G. Search negatives (recorded so verifiers can re-run them; not evidence entries)
- No paper in `papers/` and none found by web search (2026-10-08, queries: "GEOM-QM9 conformer generation baseline RDKit ETKDG then GFN2-xTB optimization coverage AMR"; "molecular conformer generation benchmark GFN2-xTB relaxation energy bond angle error evaluation GEOM 2025") reports an ETKDG + GFN2-xTB-optimisation baseline, or GFN2-xTB E_relax of generated conformers, on the GEOM-QM9 TD split.
- The Practical Cheminformatics blog (P. Walters) search page for "conformer" (https://practicalcheminformatics.blogspot.com/search?q=conformer, accessed 2026-10-08) returned no post on conformer-generation benchmarks; no snapshot taken.
- RDKit blog post "Optimizing conformer generation parameters" (G. Landrum, 2022-09-29) concerns embedding speed vs crystal-conformer RMSD/TFD on the Platinum set; not used (outside QM9/GEOM evaluation).
- Note on quote checking: in `2025_xu_fm_refiner.pdf` p. 7 the Å glyph extracts as "A" + a combining ring, so E-EVAL-007's second clause matches only after glyph normalisation (round 2 verify_B #17 checked it in the PDF).
