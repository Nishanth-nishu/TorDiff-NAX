# Conformer-generation landscape 2023–2026 and where a paper can fit (SCOUT PRIOR, round 4)

Written 2026-10-11/12. Feeds Q1 (novelty) and Q4 (plan). Evidence IDs refer to `round4/ledger/evidence_PRIOR.md`.

## How the search was done
- **arXiv API** (export.arxiv.org), abstract phrase queries, 2023-01-01 on: "conformer generation", "conformation
  generation", "molecular conformer", "torsional diffusion", "conformer" + "flow matching", "GEOM-DRUGS", "GEOM-QM9",
  ring/pucker/internal-coordinate/macrocycle combinations (≈ 320 unique hits, filtered by hand).
- **Semantic Scholar citations** of Torsional Diffusion (arXiv 2206.01729; 413 citing papers) and GeoMol
  (2106.07802; 185), filtered by title and year ≥ 2023 (≈ 107 candidates).
- **OpenReview API** search (terms above), covering ICLR 2024–2027 submissions, NeurIPS/ICML 2024–2026 and workshops
  (GEM, GenBio, AI4Science, AI4Mat, ML4Molecules, SPIGM). OpenReview PDFs are behind a browser challenge that was not
  bypassed; for OpenReview-only papers only the API abstract is used (`papers/blogs/openreview_abstracts_r4.txt`).
- Web search for ring-specific and TD-extension work (no ring-aware TD paper found beyond PuckerFlow's demo).
- Rate limits stopped some arXiv title lookups; papers marked "abstract only" or "not read" were not checked further.

Protocols: **TD-P** = GeoMol/TD split (QM9 1000 test molecules, δ 0.5 Å; DRUGS δ 0.75 Å; 2K samples). **GD-P** =
GeoDiff/ConfGF split (200 test molecules; DRUGS δ 1.25 Å). Numbers across protocols are not comparable. Numbers are
mean COV-R (%) / AMR-R (Å) unless marked; P = precision. "n/r" = not reported; "–" = not checked.

## 1. Table

### 1a. Torsion-space or fixed-local-structure methods (TD's family)
| Method | Venue / year | Representation | Learned L? | QM9 | DRUGS | Code | Takeaway |
|---|---|---|---|---|---|---|---|
| Torsional Diffusion (reference) | NeurIPS 2022 | torsions on T^m, RDKit L | no (RDKit) | TD-P 92.8 / 0.178 (paper, E-PRIOR-002); **88.8 / 0.177 from its own released samples** (E-PRIOR-001) | TD-P 72.7 / 0.582 | yes | Baseline; QM9 coverage not reproducible from its own samples |
| Particle Guidance (Corso et al.) | ICLR 2024 | TD + joint repulsive sampling | no | n/r | TD-P COV-R 77.0, COV-P 68.9 | yes | Inference-only diversity gain on TD |
| TD-IP (exposure bias; Wang et al.) | SMC 2024 | TD retrained with input perturbation | no | GD-P 84.5 / MAT 0.192 (retrained TD 83.1 / 0.238) | GD-P 91.6 / MAT 0.686 | yes | Only published TD *training* fix we found; perturbs torsion inputs, not L (E-PRIOR-033) |
| Tora3D (Zhang et al.) | J. Cheminform. 2023 | autoregressive torsions, RDKit L | no | n/r | GD-P 80.4 / MAT 0.927 | yes | Argues fixed RDKit L *avoids* ring errors (E-PRIOR-023), opposite to our finding |
| VonMisesNet (Swanson et al.) | ICML 2023 | von Mises mixtures per torsion | mean bonds/angles only | own datasets | own datasets | yes | Torsions + average L; Boltzmann (MMFF PT-HMC) targets, not GEOM (E-PRIOR-028) |
| GFlowNet conformers (Volokhova et al.) | Digital Discovery 2024 | torsions, fixed L, energy reward | no | own | own | – | Energy-trained torsion sampler |
| Torsional-GFN (Ezzine, Volokhova et al.) | ICML'25 GenBio WS; ICLR 2026 sub. | GFlowNet on T^m conditioned on L | no (conditions on L) | small MD set | – | – | Zero-shot to unseen L from MD (E-PRIOR-026) |
| TorSeq | ICLR 2024 sub. (OpenReview) | sequential torsion factorisation | no | abstract only | abstract only | – | Torsion correlations; not read |
| CoarsenConf (Reidenbach & Krishnapriyan) | JCIM 2024 | torsion-based coarse graining + learned Cartesian correction of RDKit | yes (ΔX from RDKit) | TD-P split, precision with reduced budget: COV-P 80.2 / AMR-P 0.149 (TD 75.9 / 0.262 same budget) | precision only | yes | Learns RDKit → GT correction (E-PRIOR-029); first "fix RDKit L" generator on TD splits |

### 1b. Learned internal coordinates, rings, physics-informed L
| Method | Venue / year | Representation | Learned L? | QM9 | DRUGS | Code | Takeaway |
|---|---|---|---|---|---|---|---|
| IDM thesis (Corso) | MIT thesis / arXiv 2023 | blueprint: torsions + ring-pucker hyperspheres | proposal | – | – | – | Says RDKit rings "work fine for small rings" (E-PRIOR-015); proposes pucker coordinates (E-PRIOR-016) |
| RINGER (Grambow et al.) | arXiv 2023 (ICLR 2024 sub.) | bond angles + torsions, macrocycles | yes | CREMP | – | yes | Internal-coordinate diffusion for macrocycles; post-hoc ring closure |
| PIDM (Williams & Inala) | JCIM 2024 | force-field bonded terms, diffusion | yes (bonds, angles, rings) | – | – (PDB/CSD, bonded-param errors) | – | Learned bonded parameters beat knowledge-based (E-PRIOR-027); says TD "relies entirely on ETKDGv3" (E-PRIOR-014) |
| COSMIC | JCIM 2024 | internal coordinates, adversarial | yes | not read | not read | – | Not read (no open copy found in time) |
| GO-Flow (Liu et al.) | arXiv 2605.25577 (2026) | FM on bonds × angles × torsions × SO(3) × R³ | yes | GD-P MAT-R 0.209 (text) | GD-P 94.8 / 0.797 | – | Closest FlexiTors prior art; ablation: internal manifold +4.6 COV-R (E-PRIOR-020); no ring analysis |
| PuckerFlow (Schaufelberger et al.) | Digital Discovery 2026 | FM in Cremer–Pople coordinates | yes (rings) | own ring set | – | – | Monocyclic, unsubstituted; already demonstrates PuckerFlow + TD (E-PRIOR-017/018) |
| CycloFlow | ICLR 2027 sub. | FM on (bond, angle, torsion) per heavy atom + analytic ring closure | yes | CREMP | – | anon. | Internal-coordinate cyclic peptides (E-PRIOR-021) |
| CycPepFlow, MuCO/MuCO2 | ICLR 2027 sub. / ICML 2026 | cyclic peptides | yes | CREMP-type | – | – | Ring coupling in peptides; abstract only |
| Vilya-1 | arXiv 2607.09998 (2026) | all-atom macrocycle foundation model | implicit | – | – | – | Macrocycles; not read |
| Torsion-aware FM for MOFs | NeurIPS 2025 | torsions (other domain) | – | – | – | – | Torsion representation outside molecules |

### 1c. Cartesian generators (L learned implicitly) — the SOTA on TD-P
| Method | Venue / year | Representation | Learned L? | QM9 (TD-P) | DRUGS (TD-P) | Code | Takeaway |
|---|---|---|---|---|---|---|---|
| MCF (Wang et al.) | ICML 2024 | Cartesian fields, non-equivariant | implicit | MCF-B 95.0 / 0.103 | MCF-L 84.7 / 0.390 | yes | "No torsion assumptions" stance (E-PRIOR-025); TD fails on 25 XL molecules (E-PRIOR-005) |
| ET-Flow (Hassan et al.) | NeurIPS 2024 | equivariant FM, harmonic prior | implicit | 96.5 / 0.073 | ET-Flow-SS 79.6 / 0.439 (as listed in DiTMC) | yes | "No need for internal geometry" (E-PRIOR-024) |
| EBD (Park & Shen) | NeurIPS 2024 | RDKit fragment prior + blurring diffusion | yes (corrects fragments) | GD-P 89.4 / MAT 0.237 | GD-P 92.6 / MAT 0.822 | yes | Fragment geometry corrected, not frozen |
| NExT-Mol / DMT | ICLR 2025 | Cartesian DiT + LM | implicit | DMT-B 95.2 / 0.090 | DMT-L 85.8 / 0.375 | yes | TD+PG reproduces lower than reported |
| AvgFlow (Cao et al.) | ICML 2025 | SO(3)-averaged FM, reflow | implicit | DiT 96.0 / 0.082 | DiT-L COV-P 75.7 | yes | Few-step sampling |
| DiTMC (Frank et al.) | NeurIPS 2025 | DiT with graph conditioning | implicit | **95.7–96.3 / 0.068–0.073** (E-PRIOR-034) | 78.7–80.8 / 0.415–0.444 (E-PRIOR-035) | yes | Current QM9 AMR-R best found; physical-validity metrics |
| S23D (Gurev et al.) | arXiv 2025 (ICML 2025 sub.) | standard transformer + ALiBi | implicit | 0.090 (995 molecules) | S23D-B 86.5 / 0.391 | yes | "QM9 may no longer be appropriate" (E-PRIOR-008) |
| FM-Refiner (Xu et al.) | arXiv 2025; ICLR 2026/2027 sub. | FM refiner from mid-trajectory | implicit | DMT-B+Ref AMR-R 0.070 (reports δ 0.05) | DMT-L+Ref 87.5 / 0.349 (E-PRIOR-032) | – | Two-stage refinement on top of any generator |
| EnFlow (Xu et al.) | arXiv 2512.22597 | FM + learned energy guidance | implicit | 95.7–96.3 / 0.076–0.078 | – | – | Energy-based selection, few steps |
| Lyrebird | ICLR 2026 GEM WS | ET-Flow on GEOM + CREMP, Butina split | implicit | abstract only | abstract only | – | Robustness/generalisation + energy benchmark (E-PRIOR-036) |
| MSGEN (Hu et al.) | ICLR 2026 | coarse key substructures → fine | implicit (scaffold-first) | abstract only | abstract only | – | "Scaffolds are anchors" (E-PRIOR-022) |
| FragDiff | TMLR 2025 | fragment-augmented pretraining | implicit | abstract only | abstract only | yes | Fragments help data-scarce / large molecules |
| CoFM; SSD; EVA-Flow; ConfDrift; CoordToken; CoFlo; EAD; ConformerDiff-Adaptive | 2025–2027 workshops/subs. | Cartesian / latent variants | implicit | mostly abstract only | – | – | Efficiency, priors, federated, one-step; not read |
| GD-P Cartesian group: EC-Conf, SDDiff, SE(3)-invariant acceleration, EquiFlow, EquiBoost, Transformer-flow ConfFlow, GADIFF, AGDIFF, PDCG, TensorVAE, METRIZATION pretraining | 2023–2026 | Cartesian / distances | implicit | GD-P | GD-P | mixed | Not read; GD-P numbers not comparable with ours |

### 1d. Refinement, physics fidelity, foundation data
| Method | Venue / year | What | Takeaway |
|---|---|---|---|
| ETF forcefield refinement | ICML 2024 AI4Science WS | ML force field refines generator or RDKit conformers | 45 % closer to references; works from RDKit (E-PRIOR-030) |
| "Prediction Is Not Action" | ICLR 2027 sub. | local bond/angle refiner, zero-shot across generators | Residual local errors remain in SOTA outputs (E-PRIOR-031) |
| GeoOpt-Net | arXiv 2601.22723 (2026) | single-step DFT-level geometry refinement | Refinement towards DFT, not GEOM/xTB |
| CurvFM | arXiv 2609.34592 (ICLR 2027 sub.) | Hessian-curvature-supervised FM | Physical fidelity metrics on QM9 (E-PRIOR-038) |
| PCFM constraints | AI4Mat ICLR 2026 WS | constraint-enforcing sampling on ET-Flow | bond length, planarity, stereo constraints |
| LoQI / ChEMBL3D | AI4Mat NeurIPS 2025 WS; ChemRxiv | 250M AIMNet2 conformers + diffusion | Foundation-data direction |
| STOL; FlexiFlow | 2025; ICML 2026 | fragments for large molecules; joint molecule+ensemble FM | Adjacent |

### 1e. Evaluation / benchmark papers
Zhou et al. 2023 and "Infinite Physical Monkey" 2023 (RDKit + clustering, GD-P); GEOM-Drugs Revisited (Digital
Discovery 2025: evaluation bugs, xTB-relaxation and Δ bond/angle/torsion metrics); S23D (QM9 saturated, 995-molecule
test set, E-PRIOR-008/009); FM-Refiner (QM9 at δ 0.05 Å); Lyrebird (energy benchmark, Butina split); PuckerFlow
(puckering RMSD, E-PRIOR-037).

## 2. What the landscape says about our findings (gap analysis)

| Our finding (evidence) | Already published? | Closest prior art | Verdict |
|---|---|---|---|
| Ring geometry dominates TD's QM9 floor: true rings 0.178 → 0.119 Å, true acyclic L no gain (E-PRIOR-050/051/052, ORACLE) | **Not as a measurement.** TD gives an overall RDKit-L floor (E-PRIOR-012); Corso says RDKit rings are fine for small rings (E-PRIOR-015); PuckerFlow motivates with TD's ring reliance (E-PRIOR-019); MSGEN says scaffolds matter (abstract) | TD App. F/H; IDM thesis; PuckerFlow; MSGEN | NOVEL as a quantified, model-in-the-loop decomposition on QM9 — but must survive a leak-free ring test first |
| Torsion models are brittle to the geometry they were trained on (B1 on RDKit +0.058, on MMFF +0.080; S3 on MMFF +0.039; E-PRIOR-055) | **Partly.** TD Table 8 "train on GT L" collapse on DRUGS (E-PRIOR-010/011); Torsional-GFN claims zero-shot to unseen L (E-PRIOR-026) | TD; Torsional-GFN | INCREMENTAL: the new part is that *better* geometry (MMFF) hurts true/mixed-L models, and that mixing does not fix it (E-PRIOR-054) |
| TD's 92.8 % QM9 coverage cannot be reproduced from its released conformers (88.8 %, 69 failures; E-PRIOR-001) | **No.** Failures are reported only for GEOM-XL (E-PRIOR-005/006); GeoMol noted ETKDG QM9 embedding failures (E-PRIOR-007); ≥ 5 later papers reprint 92.8 (E-PRIOR-003/004) | MCF, ET-Flow XL notes | NOVEL but small: a reproducibility note, not a paper |
| Own-conformer true-L oracles leak conformer identity (E-PRIOR-056) | **Not for model tests.** TD F.1 uses other-conformer L for a matching floor (E-PRIOR-013) | TD F.1 | NOVEL as a methodological caution; small |
| Only geometry has improved real use (MMFF −0.029; S5 −0.004 vs MMFF control; E-PRIOR-057) | MMFF/xTB pre-relaxation is standard practice; CoarsenConf/ETF/refiners learn geometry fixes (E-PRIOR-029/030/031) | refiners | Not a contribution by itself |
| Our best real-use AMR-R 0.150 Å vs published 0.068–0.103 Å (E-PRIOR-039) | — | DiTMC, ET-Flow, EnFlow | A "beats SOTA" method paper on QM9 is out of reach now |

**Where a paper can fit.** (1) An analysis paper ("what limits torsional diffusion") built on the ring/acyclic split,
the brittleness results and the evaluation fixes, made leak-free, with a ring-accuracy audit of current SOTA generators —
workshop now, journal (Digital Discovery / JCIM / TMLR) with GEOM-DRUGS. (2) A ring-aware torsional model is only
INCREMENTAL over PuckerFlow + GO-Flow and needs a real-use win to be publishable. (3) "Robust two-stage" and
"FlexiTors beats SOTA" framings are not supported by our numbers. See cards C-PRIOR-01 … 06.

## 3. PDFs added this round (papers/related/, fulltext in papers/related/fulltext/, rows in papers/INDEX.md)
2025_frank_ditmc, 2023_lee_coarsenconf, 2024_williams_pidm, 2023_swanson_vonmises, 2023_zhang_tora3d,
2024_exposure_bias_conformations, 2026_single_step_geometry_refinement, 2026_curvfm (8 PDFs). MSGEN, Lyrebird, CycloFlow,
"Prediction Is Not Action", ETF and FragDiff could not be downloaded (OpenReview challenge, no arXiv copy found); their
abstracts are in `papers/blogs/openreview_abstracts_r4.txt`.
