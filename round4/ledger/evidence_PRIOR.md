# Evidence ledger: SCOUT PRIOR (round 4, paper track)

Lens: the competitive landscape 2023–2026 and where a paper can fit (feeds Q1 novelty and Q4 plan).
Template: `orchestration/templates/evidence_entry.md`. IDs E-PRIOR-NNN. Written incrementally (skeleton first).
Our-data entries cite the result files used in rounds 1–3 and the round-2 final paired analysis.
PDF pages are pdftotext page indices of the stored PDF (= PDF page); printed page given where it differs or was checked.
OpenReview-only papers: the PDFs sit behind an OpenReview browser challenge, which was not bypassed; their abstracts
were taken from the OpenReview API and stored verbatim in `papers/blogs/openreview_abstracts_r4.txt` (accessed 2026-10-11).

## A. Reproducibility and evaluation of TD on QM9

### E-PRIOR-001
- Claim: The TD paper's own released QM9 conformers (`qm9_steps20.pkl`), scored with the repo evaluator at δ = 0.5 Å, give COV-R 88.77 % and AMR-R 0.1765 Å, with 931 molecules evaluated and 69 model failures.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt:208, `released/qm9_default` line (also RESULTS_QM9.md:21, 52–53; round-3 E-EVAL-046, VERIFIED)
- Exact quote:
  > released/qm9_default threshold=0.5000 COV-R_mean=88.7734 COV-R_median=100.0000 MAT-R_mean=0.1765 MAT-R_median=0.1461 … n_evaluated=931 n_model_failures=69
- Shows vs infers: SHOWS. Note (RESULTS_QM9.md:47): leaving failures out gives about 95 %, so 92.8 % matches neither accounting; the cause of the paper's number is UNVERIFIED.
- Supports card(s): C-PRIOR-01, C-PRIOR-05
- Confidence: high

### E-PRIOR-002
- Claim: The TD paper reports GEOM-QM9 COV-R 92.8 % (mean), AMR-R 0.178 Å, COV-P 92.7 %, AMR-P 0.221 Å.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 26, Table 7
- Exact quote:
  > Torsional diffusion 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-01, C-PRIOR-05
- Confidence: high

### E-PRIOR-003
- Claim: MCF (ICML 2024) reprints TD's QM9 row (92.8 % COV-R, 0.178 Å AMR-R) unchanged as its TD baseline.
- Source type: paper
- Locator:
  - paper: papers/related/2023_wang_mcf.pdf, PDF p. 6, GEOM-QM9 table
- Exact quote:
  > Tor. Diff. 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195
- Shows vs infers: SHOWS; INFERENCE that MCF did not re-score TD on QM9 (row identical to TD Table 7 in every digit).
- Supports card(s): C-PRIOR-05
- Confidence: high

### E-PRIOR-004
- Claim: NExT-Mol (ICLR 2025) and S23D (2025) also reprint the identical TD QM9 row.
- Source type: paper
- Locator:
  - paper: papers/related/2025_liu_nextmol.pdf, PDF p. 8, GEOM-QM9 table; papers/related/2025_gurev_s23d.pdf, PDF p. 16, Table 5
- Exact quote (NExT-Mol; the typo is in the PDF):
  > Torsoinal Diffusion 1.6M 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195
- Shows vs infers: SHOWS (the S23D row reads "TOR. DIFF. 30 YES 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195"). AvgFlow (PDF p. 7), ET-Flow, DiTMC (PDF p. 6) and EnFlow (PDF p. 5) also list 92.8 / 0.178 for TD.
- Supports card(s): C-PRIOR-05
- Confidence: high

### E-PRIOR-005
- Claim: MCF reports that TD cannot generate conformers when RDKit fails to embed a molecule (shown on GEOM-XL, 25 failed molecules).
- Source type: paper
- Locator:
  - paper: papers/related/2023_wang_mcf.pdf, PDF pp. 7–8, GEOM-XL paragraph (left column)
- Exact quote:
  > And RDKit can fail to find local structures and Torsional Diff. cannot generate conformers in these cases.
- Shows vs infers: SHOWS (GEOM-XL, not QM9)
- Supports card(s): C-PRIOR-05, C-PRIOR-01
- Confidence: high

### E-PRIOR-006
- Claim: ET-Flow independently found 27 TD generation failures on GEOM-XL, attributed to RDKit.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 19, appendix on GEOM-XL
- Exact quote:
  > provided by Torsional Diffusion, we encountered 27 failed cases for generation likely due to RDKit failures,
- Shows vs infers: SHOWS (GEOM-XL only)
- Supports card(s): C-PRIOR-05
- Confidence: high

### E-PRIOR-007
- Claim: GeoMol already noted that ETKDG cannot embed some QM9 fused-ring molecules with specified chirality.
- Source type: paper
- Locator:
  - paper: papers/related/2021_ganea_geomol.pdf, PDF p. 23 (printed 23), App. N.3 "QM9 results"
- Exact quote:
  > ETKDG fails to embed a single conformer for this structure with repeated attempts.
- Shows vs infers: SHOWS. INFERENCE: the failure mode is known since 2021; its effect on TD's published QM9 coverage is not published.
- Supports card(s): C-PRIOR-05
- Confidence: high

### E-PRIOR-008
- Claim: S23D argues the GEOM-QM9 benchmark is saturated and may no longer be appropriate for testing SOTA conformer generators.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16, appendix on QM9 results
- Exact quote:
  > the QM9 benchmark may no longer be appropriate for testing SOTA MCG models.
- Shows vs infers: SHOWS (author opinion). INFERENCE: reviewers will discount a QM9-only method paper.
- Supports card(s): C-PRIOR-01, C-PRIOR-06
- Confidence: high

### E-PRIOR-009
- Claim: S23D evaluates GEOM-QM9 on 995 test molecules (MCF's processed set), not 1000, so published QM9 test sets differ between papers.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16
- Exact quote:
  > The processed test dataset contained 995 molecules rather than the 1000 indexed in the test set of [10],
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-05
- Confidence: high

## B. What TD and its authors already say about local structure

### E-PRIOR-010
- Claim: The TD paper states that training on ground-truth local structures and testing on RDKit local structures (a distribution shift) significantly hurts performance.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 7, Sec. 4.1 "Conformer matching"
- Exact quote:
  > We found that this shift significantly hurts performance.
- Shows vs infers: SHOWS (GEOM-DRUGS). INFERENCE: prior art for our B1 brittleness result on RDKit L; the MMFF-L and mixed-training results are not in TD.
- Supports card(s): C-PRIOR-01, C-PRIOR-03
- Confidence: high

### E-PRIOR-011
- Claim: TD's ablation "Train on ground truth L" (tested on RDKit L) drops GEOM-DRUGS COV-R from 72.7 % to 34.8 % and raises AMR-R from 0.582 to 0.920 Å.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 27, Table 8
- Exact quote:
  > Train on ground truth L 34.8 22.4 0.920 0.909 22.3 7.8 1.182 1.136
- Shows vs infers: SHOWS (baseline row on the same page: "Baseline 72.7 80.0 0.582 0.565 55.2 56.9 0.778 0.729")
- Supports card(s): C-PRIOR-01, C-PRIOR-03
- Confidence: high

### E-PRIOR-012
- Claim: The TD paper names the local-structure quality of the cheminformatics method as a lower bound on TD's error.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 22, appendix "Limitations"
- Exact quote:
  > the error that torsional diffusion can achieve is lower bounded by the quality of the local structure from the selected cheminformatics method.
- Shows vs infers: SHOWS (a floor with an RDKit-L number on DRUGS, 0.324 Å; no ring/acyclic split, no QM9 model-in-the-loop test)
- Supports card(s): C-PRIOR-01, C-PRIOR-02
- Confidence: high

### E-PRIOR-013
- Claim: TD's App. F.1 already uses another ground-truth conformer's local structure (0.284 Å on DRUGS) as a measure of ground-truth L variability, i.e. an "other-conformer L" reference exists in the original paper, but only as a matching floor, not as a model test.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 21, App. F.1
- Exact quote:
  > resulting from matching a ground truth conformer to the local structure of another randomly chosen ground truth conformer,
- Shows vs infers: SHOWS. INFERENCE: our "own-conformer L leaks identity" point is a model-in-the-loop extension of this idea, not a new concept.
- Supports card(s): C-PRIOR-01
- Confidence: high

### E-PRIOR-014
- Claim: PIDM (JCIM 2024) states that TD relies entirely on ETKDGv3 for bonded parameters.
- Source type: paper
- Locator:
  - paper: papers/related/2024_williams_pidm.pdf, PDF p. 5, related-work paragraph on torsional diffusion
- Exact quote (superscript reference "8" after "ETKDGv3" omitted; line-break hyphen joined):
  > It relies entirely on the ETKDGv3 algorithm for the difficult task of establishing the bonded parameters.
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-01
- Confidence: high

### E-PRIOR-015
- Claim: Corso's IDM thesis (a TD co-author) says RDKit ring conformations work fine for small drug-like rings and suffer for larger, flexible rings and macrocycles.
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_intrinsic_diffusion.pdf, PDF p. 63 (printed 63), future work "Ring flexibility"
- Exact quote:
  > While this works fine for small rings typically present in drug-like molecules, it suffers for larger and more flexible rings, especially for macrocycles.
- Shows vs infers: SHOWS (an author statement, no measurement). INFERENCE: our QM9 result that small rings carry most of the geometry floor contradicts the "works fine for small rings" premise; that contrast is the publishable part.
- Supports card(s): C-PRIOR-01, C-PRIOR-02
- Confidence: high

### E-PRIOR-016
- Claim: The same thesis proposes integrating ring-puckering coordinates (points on hyperspheres) with torsion angles in the intrinsic-diffusion framework.
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_intrinsic_diffusion.pdf, PDF p. 63 (printed 63)
- Exact quote:
  > employing the ring puckering coordinates [18] to model the flexibility of ring conformations as points on hyperspheres.
- Shows vs infers: SHOWS (a proposal, not implemented)
- Supports card(s): C-PRIOR-02
- Confidence: high

## C. Closest prior art for learned ring / internal-coordinate structure

### E-PRIOR-017
- Claim: PuckerFlow (2026) generates ring puckers in Cremer–Pople coordinates, targets monocyclic rings without substituents, and says it can be extended to substituents by integrating with ETKDG or torsional diffusion.
- Source type: paper
- Locator:
  - paper: papers/related/2026_schaufelberger_puckerflow.pdf, PDF p. 6, dataset paragraph
- Exact quote:
  > PuckerFlow can be extended to also generate exocyclic substituents by integrating it with RDKit's ETKDG or torsional diffusion. While in this work, we target monocyclic ring systems,
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-02
- Confidence: high

### E-PRIOR-018
- Claim: PuckerFlow already demonstrates (qualitatively, in a figure) a PuckerFlow + torsional diffusion pipeline for exocyclic substituents.
- Source type: paper
- Locator:
  - paper: papers/related/2026_schaufelberger_puckerflow.pdf, PDF p. 10, Fig. 6 caption (Sec. 2.6 on PDF p. 9)
- Exact quote:
  > (b) PuckerFlow can be integrated into RDKit's ETKDG or torsional diffusion to model exocyclic substituents.
- Shows vs infers: SHOWS (examples, no GEOM benchmark). INFERENCE: "TD × ring-pucker model" is not new as an idea; only a benchmarked, jointly trained, multi-ring system would be.
- Supports card(s): C-PRIOR-02
- Confidence: high

### E-PRIOR-019
- Claim: PuckerFlow motivates itself with TD's reliance on ETKDG for cyclic substructures.
- Source type: paper
- Locator:
  - paper: papers/related/2026_schaufelberger_puckerflow.pdf, PDF pp. 2–3, Introduction (sentence crosses the page break)
- Exact quote:
  > For example, torsional diffusion relies on RDKit's ETKDG algorithm to sample cyclic substructures [23],
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-02
- Confidence: high

### E-PRIOR-020
- Claim: GO-Flow's ablation shows that removing its internal-coordinate (conformation) manifold drops GEOM-DRUGS COV-R from 94.82 % to 90.26 % (GD-P, δ = 1.25 Å).
- Source type: paper
- Locator:
  - paper: papers/related/2026_liu_goflow.pdf, PDF p. 7, Sec. 4.4.1; threshold on PDF p. 14, App. D
- Exact quote:
  > the w/o C variant achieves a COV-R of only 90.26% and a MAT-R of 0.8512 Å, which is substantially inferior to the full GO-Flow model (94.82% and 0.7971 Å, respectively).
- Shows vs infers: SHOWS (GD-P: "threshold δ is set to 0.5Å for QM9 and 1.25Å for Drugs", PDF p. 14). No ring-specific analysis found in the text (searched "ring", "cycl", "closure").
- Supports card(s): C-PRIOR-02, C-PRIOR-06
- Confidence: high

### E-PRIOR-021
- Claim: CycloFlow (ICLR 2027 submission) does flow matching on a bond length, bond angle and torsion at every heavy atom and closes rings analytically, for cyclic peptides (CREMP).
- Source type: paper (OpenReview abstract only)
- Locator:
  - paper: https://openreview.net/forum?id=66DX7Vesl4, abstract; snapshot papers/blogs/openreview_abstracts_r4.txt
- Exact quote:
  > transports a bond length, a bond angle and a torsion angle at every heavy atom by flow matching, each from a prior chosen for that internal coordinate, and closes the ring by solving six of these internal coordinates.
- Shows vs infers: SHOWS (abstract; macrocyclic peptides, not GEOM small molecules)
- Supports card(s): C-PRIOR-02
- Confidence: medium (abstract only)

### E-PRIOR-022
- Claim: MSGEN (ICLR 2026) argues that the spatial arrangement of key substructures such as scaffolds anchors the conformer distribution, and generates coarse substructures first.
- Source type: paper (OpenReview abstract only)
- Locator:
  - paper: https://openreview.net/forum?id=uYlNjHC7ag, abstract; snapshot papers/blogs/openreview_abstracts_r4.txt
- Exact quote:
  > we demonstrate that capturing the spatial arrangement of key substructures, such as scaffolds, is essential, as they serve as anchors that define the overall molecular distribution.
- Shows vs infers: SHOWS (abstract). INFERENCE: the closest published "scaffold geometry matters" claim; made for Cartesian generators, not for TD's frozen-L floor, and not ring-pucker specific.
- Supports card(s): C-PRIOR-01, C-PRIOR-02
- Confidence: medium (abstract only)

### E-PRIOR-023
- Claim: Tora3D (J. Cheminform. 2023) argues the opposite of our finding: that fixing RDKit local structures avoids ring errors that geometry-predicting models make.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhang_tora3d.pdf, PDF p. 10 (printed "Page 10 of 14"), Discussion
- Exact quote:
  > Using fixed local structures can avoid the prediction error for symmetric graph nodes (i.e., nodes with the same topology in graph) and ring structures that seriously undermines the accuracy of many 3D prediction model.
- Shows vs infers: SHOWS (their point is validity, e.g. non-planar rings from GeoMol, not accuracy against CREST references)
- Supports card(s): C-PRIOR-01
- Confidence: high

### E-PRIOR-024
- Claim: ET-Flow presents its result as showing that internal-geometry calculations are unnecessary.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 1, Abstract
- Exact quote:
  > alleviates the need for complex internal geometry calculations and large architectures, contrary to the prevailing methods in the field.
- Shows vs infers: SHOWS. INFERENCE: a reviewer of an internal-coordinate/torsion method will cite this stance.
- Supports card(s): C-PRIOR-06, C-PRIOR-02
- Confidence: high

### E-PRIOR-025
- Claim: MCF frames its contribution as dropping explicit molecular structure such as torsion modelling.
- Source type: paper
- Locator:
  - paper: papers/related/2023_wang_mcf.pdf, PDF p. 1, Abstract
- Exact quote (line-break hyphen "struc-ture" joined):
  > without making assumptions about the explicit structure of molecules (e.g. modeling torsional angles) we are able to radically simplify structure learning,
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-06
- Confidence: high

### E-PRIOR-026
- Claim: Torsional-GFN conditions a torus sampler on the local structure and reports zero-shot generalisation to unseen bond lengths and angles from MD.
- Source type: paper
- Locator:
  - paper: papers/related/2025_ezzine_torsional_gfn.pdf, PDF p. 1, Abstract
- Exact quote:
  > allows for zero-shot generalization to unseen bond lengths and angles coming from the MD simulations for such molecules.
- Shows vs infers: SHOWS (small MD set, energy-trained). INFERENCE: the closest published "torsion model robust to L" claim; not on GEOM.
- Supports card(s): C-PRIOR-03
- Confidence: high

### E-PRIOR-027
- Claim: PIDM learns bonded parameters from GFN2-xTB-optimised data and reports bonded-parameter accuracy above knowledge-based methods.
- Source type: paper
- Locator:
  - paper: papers/related/2024_williams_pidm.pdf, PDF p. 1, Abstract
- Exact quote:
  > high accuracy is achieved for bonded parameters, exceeding that of conventional, knowledge-based methods.
- Shows vs infers: SHOWS (evaluated against PDB/CSD and bonded-parameter errors, not GEOM COV/AMR; its own conclusion, PDF p. 47, says accuracy is "comparable to conventional methods")
- Supports card(s): C-PRIOR-02
- Confidence: high

### E-PRIOR-028
- Claim: VonMisesNet (ICML 2023) samples only rotatable-bond torsions and predicts average bond lengths and angles.
- Source type: paper
- Locator:
  - paper: papers/related/2023_swanson_vonmises.pdf, PDF p. 5, Sec. 2.3
- Exact quote:
  > Therefore, VonMisesNet predicts averages for bond lengths and bond angles.
- Shows vs infers: SHOWS (own Boltzmann datasets NMRShiftDB/GDB-17, not GEOM)
- Supports card(s): C-PRIOR-02
- Confidence: high

## D. Two-stage / refinement prior art

### E-PRIOR-029
- Claim: CoarsenConf conditions on the RDKit conformer and learns the coordinate correction from RDKit to the ground truth.
- Source type: paper
- Locator:
  - paper: papers/related/2023_lee_coarsenconf.pdf, PDF p. 6
- Exact quote:
  > of learning raw atom coordinates, we learn to predict the difference between the RDKit reference and ground truth
- Shows vs infers: SHOWS (TD-P split; precision-only metrics with a reduced sample budget, PDF p. 7)
- Supports card(s): C-PRIOR-03
- Confidence: high

### E-PRIOR-030
- Claim: An ML-forcefield refinement paper (ICML 2024 AI4Science workshop) refines conformers from generators and from plain RDKit, reporting a 45 % distance reduction.
- Source type: paper (OpenReview abstract only)
- Locator:
  - paper: https://openreview.net/forum?id=f56YcaCbRq, abstract; snapshot papers/blogs/openreview_abstracts_r4.txt
- Exact quote:
  > it can deliver competitive performance even when beginning with a simple initialization of conformers by RDKit,
- Shows vs infers: SHOWS (abstract)
- Supports card(s): C-PRIOR-03
- Confidence: medium (abstract only)

### E-PRIOR-031
- Claim: An ICLR 2027 submission builds a post-generation refiner for residual bond-length/angle errors that transfers zero-shot across generators.
- Source type: paper (OpenReview abstract only)
- Locator:
  - paper: https://openreview.net/forum?id=nENwgDM3Ak, abstract; snapshot papers/blogs/openreview_abstracts_r4.txt
- Exact quote:
  > generated conformers can still exhibit residual local geometric deviations in bond lengths and bond angles.
- Shows vs infers: SHOWS (abstract)
- Supports card(s): C-PRIOR-03, C-PRIOR-04
- Confidence: medium (abstract only)

### E-PRIOR-032
- Claim: The flow-matching refiner (2025) reaches GEOM-DRUGS COV-R 87.47 %, AMR-R 0.349 Å, AMR-P 0.497 Å with DMT-L + Refiner (TD-P, δ = 0.75 Å).
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, GEOM-DRUGS table
- Exact quote:
  > DMT-L + Refiner 87.47 94.12 0.349 0.319 75.91 81.51 0.497 0.446
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-03, C-PRIOR-06
- Confidence: high

### E-PRIOR-033
- Claim: An input-perturbation (exposure-bias) retraining of TD reports SOTA on GEOM-Drugs and parity on GEOM-QM9 under the GeoDiff protocol (40k training molecules, 200 test molecules).
- Source type: paper
- Locator:
  - paper: papers/related/2024_exposure_bias_conformations.pdf, PDF p. 1, Abstract; protocol PDF p. 4
- Exact quote:
  > the IP-enhanced Torsional Diffusion model, we achieve new state-of-the-art performance on the GEOM-Drugs dataset and are on par on GEOM-QM9.
- Shows vs infers: SHOWS (GD-P; QM9 MAT 0.2379 → 0.1924 for retrained TD vs TD-IP, PDF p. 4 Table II). It perturbs the diffusion input (torsions), not the local structure.
- Supports card(s): C-PRIOR-03
- Confidence: high

## E. Current state of the art (TD-P) and evaluation trends

### E-PRIOR-034
- Claim: DiTMC (NeurIPS 2025) reports GEOM-QM9 AMR-R 0.068–0.073 Å (three variants, 3 seeds) on the GeoMol/TD splits, the lowest published QM9 AMR-R found in this search.
- Source type: paper
- Locator:
  - paper: papers/related/2025_frank_ditmc.pdf, PDF p. 6, Table 1 and Sec. 5. The text dump misaligns row labels; values read in label order: aPE-B 96.1/0.073, rPE-B 96.3/0.070, PE(3)-B 95.7/0.068 (COV-R % / AMR-R Å).
- Exact quote:
  > On GEOM-QM9, our models produce diverse, high quality samples, outperforming the current state-of-the-art across all AMR-R, AMR-P, and COV-P metrics,
- Shows vs infers: SHOWS (split: "The data splits are taken from Ref. [64]", same page; Ref. 64 is GeoMol)
- Supports card(s): C-PRIOR-06
- Confidence: high (verifier: read values from the PDF table)

### E-PRIOR-035
- Claim: DiTMC's GEOM-DRUGS AMR-R is 0.415–0.444 Å with COV-R 78.7–80.8 % across its six variants (TD-P, δ = 0.75 Å).
- Source type: paper
- Locator:
  - paper: papers/related/2025_frank_ditmc.pdf, PDF p. 8, Table 3
- Exact quote:
  > Table 3: Results on GEOM-DRUGS for different generative models (parameter counts in parentheses).
- Shows vs infers: SHOWS (values read from the table in label order; caption quoted because the extracted rows are misaligned)
- Supports card(s): C-PRIOR-06
- Confidence: medium (verifier should read the PDF table)

### E-PRIOR-036
- Claim: Lyrebird (ICLR 2026 GEM workshop) trains an ET-Flow model on Butina-split GEOM-Drugs, GEOM-QM9 and CREMP, and adds an energy-based benchmark of lowest-basin sampling and relaxation.
- Source type: paper (OpenReview abstract only)
- Locator:
  - paper: https://openreview.net/forum?id=1SBcUq9Lmr, abstract; snapshot papers/blogs/openreview_abstracts_r4.txt
- Exact quote:
  > evaluate generalization by training jointly on Butina-split datasets of drug-like molecules from GEOM-Drugs and GEOM-QM9 and macrocyclic peptides from CREMP.
- Shows vs infers: SHOWS (abstract). INFERENCE: robustness/generalisation evaluation of conformer generators is already a workshop topic.
- Supports card(s): C-PRIOR-03, C-PRIOR-04
- Confidence: medium (abstract only)

### E-PRIOR-037
- Claim: PuckerFlow evaluates rings with a puckering RMSD (displacements from the ring mean plane) as well as all-atom RMSD.
- Source type: paper
- Locator:
  - paper: papers/related/2026_schaufelberger_puckerflow.pdf, PDF p. 7, Sec. 2.4
- Exact quote:
  > puckering RMSD, which measures the ability of the methods to reproduce the characteristic ring puckering by quantifying the atomic displacements from the mean plane;
- Shows vs infers: SHOWS (on its own ring set, comparing ETKDG, MCF and Euclidean models; not applied to TD/ET-Flow/DiTMC outputs on GEOM)
- Supports card(s): C-PRIOR-04
- Confidence: high

### E-PRIOR-038
- Claim: CurvFM (2026 preprint) adds Hessian-curvature supervision to flow matching and reports a 63 % lower median maximum-force deviation on GEOM-QM9 at unchanged structural accuracy.
- Source type: paper
- Locator:
  - paper: papers/related/2026_curvfm.pdf, PDF p. 1, Abstract
- Exact quote:
  > CurvFM substantially improves physical fidelity at unchanged global structural accuracy, reducing the median deviation in maximum force by 63% on GEOM-QM9
- Shows vs infers: SHOWS. INFERENCE: physical-fidelity metrics beyond RMSD are becoming expected for QM9 papers.
- Supports card(s): C-PRIOR-04
- Confidence: high

### E-PRIOR-039
- Claim: Published learned Cartesian models on the TD-P GEOM-QM9 split report AMR-R between 0.068 and 0.103 Å (DiTMC 0.068–0.073, ET-Flow 0.073, EnFlow 0.076–0.078, AvgFlowDiT 0.082, DMT-B 0.090, MCF-B 0.103).
- Source type: paper
- Locator:
  - paper: papers/related/2025_cao_avgflow.pdf, PDF p. 7, QM9 table; 2025_xu_enflow.pdf, PDF p. 5, Table 1; ET-Flow, DMT-B, MCF-B as verified in round 3 (E-EVAL-001, -002, -003); DiTMC E-PRIOR-034
- Exact quote (AvgFlow):
  > AvgFlowDiT (52M) 96.0 100 0.082 0.030 95.0 100 0.088 0.039
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-06, C-PRIOR-01
- Confidence: high

## F. Our measured findings (what a paper would rest on)

### E-PRIOR-050
- Claim: With true ring geometry and RDKit acyclic geometry (ORACLE, own-conformer cycled), the standard model (CTRL_rematch) reaches AMR-R 0.1186 Å (955 molecules), against 0.1779 Å on ETKDG L (935 molecules).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:402 (S1.B1_vs_CR.S1_A5ring_cyc_ORACLE, ref_value) and :274 (S1.B1_vs_CR.S1_etkdg2L, ref_value)
- Exact quote:
  > S1.B1_vs_CR.S1_A5ring_cyc_ORACLE,…,955,B1,…,CR,S1_A5ring_cyc_ORACLE,,0.1876942638772854,0.11860243512789748,…
- Shows vs infers: SHOWS. Caveats: different molecule sets (955 vs 935); the ring geometry comes from each target's own conformer, so part of the gain may be conformer-identity leakage (E-PRIOR-056); a leak-free ring test (other-conformer or template rings) is needed before any paper claim.
- Supports card(s): C-PRIOR-01, C-PRIOR-02, C-PRIOR-04
- Confidence: high (number); medium (interpretation)

### E-PRIOR-051
- Claim: With true acyclic geometry and RDKit rings (ORACLE), the same model gives AMR-R 0.1824 Å, i.e. no gain over ETKDG L.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:418 (S1.B1_vs_CR.S1_A5acyc_cyc_ORACLE, ref_value)
- Exact quote:
  > S1.B1_vs_CR.S1_A5acyc_cyc_ORACLE,…,955,B1,…,CR,S1_A5acyc_cyc_ORACLE,,0.158329278855473,0.18241186694248412,…
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-01, C-PRIOR-02
- Confidence: high

### E-PRIOR-052
- Claim: Without a model, the TD-style floor for ring-containing molecules only drops from 0.117 to 0.099 Å with true acyclic angles and bonds, while acyclic molecules drop from 0.051 to 0.008 Å.
- Source type: our-data
- Locator:
  - our-data: EXPERIMENTS.md:193–194 (E12 floor analyses, by ring size; reproduced by a round-2 verifier)
- Exact quote:
  > acyclic molecules 0.051 → 0.008 with true acyclic angles and bonds; **ring molecules (72 % of conformers) 0.117 → 0.099 only.**
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-01
- Confidence: high

### E-PRIOR-053
- Claim: 49.6 % of ETKDG seeds of molecules with a ring of ≥ 4 atoms have an endocyclic-dihedral RMSD above 10° (MMFF 35.1 %).
- Source type: our-data
- Locator:
  - our-data: round3/ledger/evidence_GEOM.md:49 (E-GEOM-002, VERIFIED in round3/ledger/verify_V3.md:18)
- Exact quote:
  > share > 10° is ETKDG 49.6%, MMFF 35.1%,
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-01, C-PRIOR-04
- Confidence: high

### E-PRIOR-054
- Claim: Round-2 final paired analysis: S3 (50/50 RDKit/true-L training) is significantly worse than CTRL_rematch on RDKit L, +0.0032 Å AMR-R [+0.0013, +0.0052], Holm p = 0.0096.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:50 (S3.rdkit_vs_CR, MAT-R, primary); round4/BRIEF.md "Round-2 FINAL paired analysis"
- Exact quote:
  > S3.rdkit_vs_CR,primary,union,MAT-R,primary,934,…,0.1818677432916211,0.1786359860697768,0.0032317572218443077,0.0012702605359765939,0.005228308316968585,…,0.0096,
- Shows vs infers: SHOWS
- Supports card(s): C-PRIOR-03, C-PRIOR-06
- Confidence: high

### E-PRIOR-055
- Claim: Models trained on true or mixed L get worse on MMFF geometry than the standard model: B1 +0.0798 Å and S3 +0.0394 Å AMR-R vs CTRL_rematch on MMFF-relaxed ETKDG L (CTRL 0.1525).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:290 (S1.B1_vs_CR.S1_etkdg2L_mmff) and :482 (S1.S3_vs_CR.S1_etkdg2L_mmff)
- Exact quote:
  > S1.S3_vs_CR.S1_etkdg2L_mmff,…,935,S3,…,0.19187000740152807,0.15246312923791927,0.03940687816360878,…
- Shows vs infers: SHOWS (B1 line: "…0.23227061127349388,0.15246312923791927,0.0798074820355746…")
- Supports card(s): C-PRIOR-01, C-PRIOR-03
- Confidence: high

### E-PRIOR-056
- Claim: Own-conformer true-L oracle results leak conformer identity: inverting the random-vs-cycled difference, B1's AMR-R with only other conformers' true L is about 0.35 / 0.23 / 0.16 / 0.17 Å (by n_true bin), versus 0.016–0.046 Å with its own conformer's L and 0.23–0.29 Å on RDKit L; a model-free copier scores 0.000 on cycled seeds.
- Source type: our-data
- Locator:
  - our-data: round3/ledger/v2_data/v2_ntrue_check.out:32 (inversion) and :14 (copier); round3/impact.md §3g
- Exact quote:
  > B1         2-3: own 0.016 -> other 0.353 (RDKit 0.233) | 4-10: own 0.025 -> other 0.228 (RDKit 0.262) | 11-30: own 0.038 -> other 0.159 (RDKit 0.264) | >30: own 0.046 -> other 0.169 (RDKit 0.291)
- Shows vs infers: SHOWS (inversion amplifies seed noise 8–14×, ±0.03–0.04 for B1 per impact.md §3g); the BRIEF lists the leak as to be settled by a direct leak-free test.
- Supports card(s): C-PRIOR-01, C-PRIOR-05
- Confidence: medium (derived, not a direct test)

### E-PRIOR-057
- Claim: The only real-use gains so far come from geometry: MMFF-relaxed seeds with MMFF-matched training (S5) give AMR-R 0.1500 Å, −0.0287 Å vs CTRL_rematch on RDKit L, and −0.0041 Å [−0.0064, −0.0019] vs CTRL_rematch with MMFF seeds.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:210 (S5.premmff_vs_CR) and :194 (S5.premmff_vs_CRpremmff)
- Exact quote:
  > S5.premmff_vs_CR,primary,union,MAT-R,primary,934,B3,…,0.14997364091242826,0.1786359860697768,-0.028662345157348532,-0.03390312523981321,-0.023450328109105734,…
- Shows vs infers: SHOWS. INFERENCE: our best real-use number (0.150 Å) is about 2× the best published TD-P QM9 AMR-R (0.068–0.078, E-PRIOR-034, -039).
- Supports card(s): C-PRIOR-03, C-PRIOR-06
- Confidence: high
