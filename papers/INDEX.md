# Literature index: FlexiTors-Diffusion knowledge base

Built 2026-09-29. Root: `papers/`. Every entry has a verified URL that was opened, a local PDF (except LoQI, see its note), a per-paper note (`.md`), and an extracted layout-preserving full text in `related/fulltext/<name>.txt` for grep/RAG.

**Gap codes:**
- **G1:** rigid local structure (bond lengths, angles and rings frozen from RDKit)
- **G2:** local-structure distribution shift between training and test / conformer matching
- **G3:** torsion–bond-angle coupling (sterics)

**Protocol codes:**
- **TD-P:** GeoMol/TD split (DRUGS 243473/30433/1000, QM9 106586/13323/1000); δ = 0.75 Å DRUGS, 0.5 Å QM9; 2K generated
- **GD-P:** ConfGF/GeoDiff split (40k training molecules × 5 conformers, 200 test molecules); δ = 1.25 Å DRUGS, 0.5 Å QM9

**Numbers from the two protocols are not comparable.**

**Other companion files:** `core/tordiff_analysis.md` (deep read of TD) · `gap_synthesis.md` (per-gap synthesis, FlexiTors design, QM9 ablations, **Metric reference**) · the local TD code in `../torsional-diffusion/` (`evaluate_confs.py` is the reference evaluator).

## Core
| File | Year | Venue | Gaps | Takeaway |
|---|---|---|---|---|
| core/2022_jing_torsional_diffusion.pdf (+ `tordiff_analysis.md`, `.fulltext.txt`) | 2022 | NeurIPS 2022 | defines G1–G3 | Diffusion on T^m for rotatable torsions; L from RDKit; conformer matching; DRUGS COV-R 72.7 / AMR-R 0.582 (δ = 0.75); conformer-matching floor estimates 0.324 Å (DRUGS) and 0.17 Å (QM9), upper-biased and not lower bounds on AMR-R (review M1); GT-other-L 0.284 Å (F.1, p. 21) is in-paper evidence for G3. |

## Related (sorted by relevance to FlexiTors)
| File (related/) | Year | Venue | Gaps | Protocol | One-line takeaway |
|---|---|---|---|---|---|
| 2026_liu_goflow | 2026 | arXiv 2605.25577 | G1 G2 G3 (addresses) | GD-P | **Closest prior art to FlexiTors:** flow matching on internal coordinates (lengths, angles, torsions) × SO(3) × R^3 with Jacobian composition; the ablation shows the internal-coordinate manifold helps (DRUGS COV-R 94.8 vs. 90.3 without it). |
| 2026_schaufelberger_puckerflow | 2026 | arXiv 2601.12859 | G1 (rings) | own ring set | Flow matching in Cremer-Pople puckering coordinates; closed rings by construction; composable with TD for exocyclic torsions; a ready-made "ring factor". |
| 2023_corso_intrinsic_diffusion | 2023 | MIT thesis, arXiv 2302.12255 | G1 G3 (proposals) | TD-P | TD co-author's IDM blueprint; proposes ring-puckering hyperspheres and treating the manifold as a soft constraint — the FlexiTors design spec. |
| 2022_corso_diffdock | 2022 | ICLR 2023 | technique | — | Product-manifold diffusion T(3) × SO(3) × SO(2)^m with disentangled factors — the math template for T^m × (angle factor). |
| 2023_grambow_ringer | 2023 | arXiv 2305.19800 | G1 G2 G3 (macrocycles) | CREMP | Joint bond-angle + torsion diffusion for macrocycles; shows TD inherits RDKit's ring/backbone distribution and needs ETKDGv3 macrocycle seeding. |
| 2022_wu_foldingdiff | 2022 | arXiv 2209.15611 | G1 G3 (proteins) | — | Wrapped-normal diffusion over 3 bond angles + 3 dihedrals per residue; simplest joint angle–torsion template. |
| 2025_ezzine_torsional_gfn | 2025 | ICML'25 GenBio workshop, arXiv 2507.11759 | G2 G3 (partial) | small MD set | GFlowNet on the torus conditioned on L; torsion modes shift when L changes, which is direct evidence for G3; generalises to unseen L. |
| 2024_park_equivariant_blurring_diffusion | 2024 | NeurIPS 2024 | G1 (partial) G2 | GD-P | RDKit fragment prior that is *corrected* by blurring diffusion (not frozen); DRUGS COV-R 92.6 / MAT-R 0.82 at δ = 1.25. |
| 2024_hassan_etflow | 2024 | NeurIPS 2024 | G1 G2 (Cartesian) | TD-P | Harmonic prior + alignment + equivariant transformer flow matching; DRUGS COV-P 75.2 (SS); QM9 AMR-R 0.073, well below TD's RDKit-L floor. |
| 2023_wang_mcf | 2023 | ICML 2024 | G1 G2 (Cartesian) | TD-P | Scaled non-equivariant diffusion; DRUGS COV-R 84.7 (L); XL: TD fails on 25/102 molecules because of RDKit embedding failures. |
| 2025_nikitin_geom_drugs_revisited | 2025 | Digital Discovery 2025 | eval (G1 G3 metrics) | — | GEOM evaluation bugs; recommends GFN2-xTB E_relax and Δ bond-length/angle/torsion metrics, which are the right local-structure metrics for FlexiTors. |
| 2025_gurev_s23d | 2025 | arXiv 2506.19834 | G1 G2 (Cartesian); eval | TD-P | ALiBi transformer diffusion; DRUGS COV-R 87.0 / AMR-R 0.380; argues **QM9 at δ = 0.5 Å is saturated**. |
| 2025_xu_fm_refiner | 2025 | arXiv 2510.04878 | G1 (partial); eval | TD-P (+ QM9 at δ = 0.05) | Flow-matching refiner starting mid-trajectory from upstream samples; best DRUGS numbers (DMT-L + refiner COV-R 87.5, AMR-P 0.497); uses **δ = 0.05 Å on QM9**. |
| 2025_cao_avgflow | 2025 | ICML 2025 | G1 G2 (Cartesian) | TD-P | SO(3)-averaged flow matching + reflow/distillation for 1–2 step sampling; DRUGS AvgFlowDiT-L COV-P 75.7. |
| 2025_xu_enflow | 2025 | arXiv 2512.22597 | G1 G2; G3 (partial, energy) | TD-P | Flow matching + learned energy guidance; few-step sampling; ground-state ranking; reproduction of ET-Flow shows a reporting gap (COV-P 69.8 vs. 74.4). |
| 2023_corso_particle_guidance | 2023 | ICLR 2024 | none (inference) | TD-P | Joint repulsive sampling over torsion differences; TD + PG DRUGS COV-R 77.0 / COV-P 68.9 without retraining. |
| 2023_stark_harmonicflow | 2023 | ICML 2024 | G1 G2 (docking) | — | Harmonic (Laplacian) prior + self-conditioned flow matching; the TD group's own move away from rigid product spaces. |
| 2023_chen_riemannian_flow_matching | 2023 | ICLR 2024 | technique | — | Simulation-free flow matching on tori and product manifolds; the enabling tool for a flow version of FlexiTors. |
| 2025_liu_nextmol | 2025 | ICLR 2025 | G1 G2 (Cartesian) | TD-P | DMT transformer diffusion + LM features; DRUGS DMT-L COV-R 85.8 / AMR-R 0.375; TD + PG reproduces lower than reported. |
| 2021_ganea_geomol | 2021 | NeurIPS 2021 | G1 G3 (partial) | TD-P origin | Predicts local structures + torsions in one pass from the graph; origin of the splits and 2K protocol. |
| 2022_xu_geodiff | 2022 | ICLR 2022 | G1 G3 (implicit) | GD-P | Equivariant Cartesian DDPM, 5000 steps; origin of the δ = 1.25 Å / 200-molecule protocol. |
| 2023_zhou_dl_conformation_critique | 2023 | arXiv 2302.07061 | eval | GD-P | RDKit + oversampling + clustering beats most deep models on recall; recall metrics are gameable. |
| 2023_grambow_cremp | 2023 | Sci. Data 2024 | G1 test bed | — | 36k macrocyclic peptides with CREST ensembles; OOD test set for rings/macrocycles. |
| 2025_zhu_stol | 2025 | arXiv 2511.12182 | G1 (partial, large molecules) | DFT sets | Fragment diffusion + assembly for large molecules trained only on small ones; an alternative L-sampler for XL. |
| 2025_nikitin_loqi (md only; PDF blocked) | 2025/26 | ChemRxiv | G1 G2 (Cartesian) | ChEMBL3D | 250M AIMNet2 conformers + stereo-aware diffusion; handles macrocycles and E/Z; foundation-model direction. |
| 2025_tedoldi_flexiflow | 2025 | arXiv 2511.17249 | none directly | de novo | Joint molecule + multi-conformer flow matching; CREST-referenced coverage sweeps. |
| 2021_ho_cascaded_diffusion (PDF + fulltext only; added round 2 by research agent A) | 2021 | JMLR 2022, arXiv 2106.15282 | G2 (technique) | — | Cascaded diffusion: a 2nd-stage model trained on GT conditioning fails on 1st-stage samples (train-test mismatch); fixed by Gaussian "conditioning augmentation", optionally amortised over the augmentation level fed as an extra embedding (p. 3, 6, 18). Template for an L-noise-augmented / L-level-conditioned torsion model. |
| 2023_ning_input_perturbation (PDF + fulltext only; added round 2 by research agent A) | 2023 | ICML 2023, arXiv 2301.11706 | G2 (technique) | — | DDPM-IP: perturb ground-truth inputs during training to simulate inference-time prediction errors (exposure bias) (p. 1, 4). Second precedent for L-noise augmentation. |

| 2022_axelrod_geom (PDF + fulltext only; added round 3 by SCOUT GEOM) | 2022 | Sci. Data 2022, arXiv 2006.05531 | data provenance (G1 reference level) | — | The GEOM dataset paper: QM9 conformers are CREST ensembles with GFN2-xTB-optimised geometries (QM9 seeds re-optimised with xTB, default gas-phase CREST settings; 0.5 core-h per QM9 molecule) (p. 3, 5). So an L relaxed with GFN2-xTB targets the reference level of our GT. |
| 2023_kovacs_mace_off (PDF + fulltext only; added round 3 by SCOUT GEOM) | 2023/25 | arXiv 2312.15211 (v5) | technique (ML potential) | — | MACE-OFF transferable organic ML force fields trained on SPICE at ωB97M-D3(BJ)/def2-TZVPPD (p. 3): a DFT-level target, not the GFN2-xTB level of GEOM references. |

| 2021_chan_ring_puckering (PDF + fulltext only; added round 3 by SCOUT GEOM; open-access author copy from ora.ox.ac.uk) | 2021 | J. Chem. Inf. Model. 61, 743–755 | G1 (rings) | CREST/GFN2 ring set | Cremer-Pople analysis of 140k molecules: ring conformations fall into relatively few canonical clusters; a knowledge-based pucker sampler reproduces CREST/GFN2 lowest-energy ring conformers at 0.09 Å mean RMSD, the residual being bond lengths/angles (p. 1, 11). Supports ring-template / pucker-library L sources. |

| 2025_sun_noise_conditioning (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2025 | ICML 2025, arXiv 2502.13129 | G2 (technique) | — | Most denoising generative models degrade gracefully without noise-level conditioning, because the level is inferable from the input (dimension-dependent); a separately predicted level behaves like none (p. 1, 4, 8). Bears on whether a torsion model needs an explicit L-quality input (S4 λ). |
| 2022_ho_classifier_free_guidance (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2022 | NeurIPS 2021 workshop, arXiv 2207.12598 | technique | — | CFG: joint conditional/unconditional training with a null token; p_uncond 0.1-0.2 beats 0.5; guidance trades diversity for fidelity (p. 4, 8, 9). |
| 2023_du_reduce_reuse_recycle (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2023 | ICML 2023, arXiv 2302.11552 | technique | — | Composing diffusion scores and running the plain reverse process does not sample the composed distribution; MCMC correction needed (p. 5). |
| 2015_bengio_scheduled_sampling (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2015 | NeurIPS 2015, arXiv 1506.03099 | G2 (analogy) | — | Scheduled sampling: curriculum from true to model-generated inputs to fight train/inference discrepancy (p. 1, 2). |
| 2015_huszar_scheduled_sampling_critique (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2015 | arXiv 1511.05101 | G2 (analogy) | — | The scheduled-sampling objective is improper/inconsistent and pushes models to ignore the conditioning content (p. 1, 4). |
| 2011_ross_dagger (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2011 | AISTATS 2011, arXiv 1011.0686 | G2 (analogy) | — | DAgger: train on the input distribution the learner induces, labelled by the expert (p. 1, 3); the template for "train on the deployed L source with re-matched torsions". |
| 2017_tobin_domain_randomization (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2017 | IROS 2017, arXiv 1703.06907 | G2 (analogy) | — | Domain randomisation: with enough training variability the target looks like one more variation (p. 1). |
| 2024_ni_sliced_denoising (PDF + fulltext only; added round 3 by SCOUT ROBUST) | 2024 | ICLR 2024, arXiv 2311.02124 | technique (molecular noise) | — | SliDe: isotropic coordinate noise is an inappropriate assumption for molecules; "BAT" noise on bond lengths, angles, torsions with force-field variances (p. 2). |
| 2022_dauparas_proteinmpnn (PDF + fulltext only; added round 3 by SCOUT ROBUST; bioRxiv 2022.06.03.494563 v1) | 2022 | Science 2022 (preprint) | G2 (analogy, proteins) | — | 0.02 Å backbone-noise training: better recovery on AlphaFold models, worse on exact PDB structures; exact coordinates carry "memory" of the target (p. 3 Table 1, p. 7). |

Round-3 fulltexts added by SCOUT ROBUST are reading-order extractions (`pdftotext` without `-layout`). A reading-order copy
of the round-2 Ning et al. fulltext is at `related/fulltext/2023_ning_input_perturbation.readingorder.txt`. Blog snapshots
added by SCOUT ROBUST (in `blogs/`): `song_2021_score.txt`, `weng_2019_domain_randomization.txt`,
`dieleman_2022_guidance.txt`, `weng_2021_diffusion_models.txt`.

| 2023_zhang_infinite_physical_monkey (PDF + fulltext only; added round 3 by SCOUT EVAL) | 2023 | arXiv 2304.10494 | eval | GD-P | Reply to Zhou et al.: a still more random "Infinite Stochastic Monkey" sampler also scores high COV/MAT with ~2000 samples; argues RDKit + Clustering is only a fair baseline if deep models get the same sample-and-cluster budget (p. 5, 9). |

Blog/technical snapshots added by SCOUT EVAL (in `blogs/`): `rowan_openconf_2026.txt` (Rowan, ETKDG pathologies),
`rdkit_release_notes_2024_03.txt` (RDKit default changed from ETKDGv1 to ETKDGv3 in 2024.03.1),
`condaforge_xtb_metadata_eval.txt` (verbatim conda-forge xtb JSON fields; linux-64 6.2.3–6.7.1). SCOUT EVAL's fulltext
is a `pdftotext -layout` extraction.

**Counts:** 1 core paper + 39 related entries (38 PDFs stored; LoQI note only).

| 2026_schaufelberger_tmcgen (PDF + fulltext only; added round 4 by SCOUT ARCH) | 2026 | arXiv 2606.00666 | technique (G1 angles, product manifold) | own TMC set | TMCgen (PuckerFlow group): simulation-free diffusion on S^2 coordination angles combined with per-ligand rotation and torsion diffusion; states the n-sphere generalisation could serve ring-system conformer ensembles (p. 6). Closest product-manifold precedent for an angle/pucker factor next to torsions; novelty risk for FlexiTors-ring. |
| 2025_frank_ditmc (PDF + fulltext only; added round 4 by SCOUT ARCH) | 2025 | NeurIPS 2025, arXiv 2506.15378 | G1 G2 (Cartesian) | TD-P | DiTMC: diffusion-transformer flow matching with harmonic prior and graph conditioning; QM9 AMR-R 0.068–0.073 (TD row 0.178), DRUGS AMR-R 0.415–0.444 (p. 6, 8); QM9 250 epochs ≈ 2 H100-days. Current QM9 frontier together with ET-Flow. |
| 2024_wang_td_input_perturbation (PDF + fulltext only; added round 4 by SCOUT ARCH) | 2024 | SMC 2024, arXiv 2409.14014 | G2 (technique, TD follow-up) | GD-P | Input perturbation (DDPM-IP) applied to a TD retrained on the ConfGF split: QM9 MAT 0.2379 → 0.1924 (GD-P, 200 molecules). Prior art for exposure-bias fixes on TD. |
| 2026_darouich_curvfm (PDF + fulltext only; added round 4 by SCOUT ARCH) | 2026 | arXiv 2609.34592 | G1 (physical fidelity of L) | own split | CurvFM: flow matching with a frozen ML-Hessian residual loss at the generated endpoint; −63 % median max-force deviation on GEOM-QM9 at unchanged structural accuracy (p. 1). An auxiliary loss option for a learned L generator. |

Blog/code snapshot added by SCOUT ARCH (in `blogs/`): `puckerflow_repo_readme.txt` (PuckerFlow README: torch 2.1.0+cu118,
MIT licence, built on the TD codebase). SCOUT ARCH fulltexts are `pdftotext -layout` extractions with form feeds.

Repository / data snapshots added by SCOUT BENCH (round 4, in `blogs/`; no new PDFs): `td_gdrive_listing.txt` (TD shared
Drive listing, archive sizes, released `drugs_default` config), `td_github_issues.txt` (TD issues #4, #15),
`etflow_github_issues.txt` (ET-Flow issues #31 QM9 non-reproduction, #26, #24 chirality, #11 Kabsch, #13),
`etflow_eval_code.txt` (ET-Flow `covmat.py`, `eval.py`, `eval_cov_mat.py`, `prepare_data.py`, line-numbered),
`zenodo_etflow_14226681.txt` (ET-Flow checkpoint sizes), `geom_dataverse_files.txt` (GEOM Dataverse file sizes),
`geom_revisited_repo.txt` (GEOM-Drugs Revisited README + xTB energy-benchmark scripts), `mcf_repo_readme.txt`,
`nextmol_repo_readme.txt`. OpenReview review threads could not be fetched (browser-verification wall); forum IDs are in
`round4/ledger/evidence_BENCH.md` section Z.

**Round 4, added by SCOUT TRAIN (2026-10-12; PDF + reading-order fulltext only, `pdftotext` without `-layout`):**

| File (related/) | Year | Venue | Gaps | Protocol | One-line takeaway |
|---|---|---|---|---|---|
| 2022_hsu_esm_if1 | 2022 | ICML 2022 (bioRxiv 2022.04.10.487779 v2) | G2 (analogy, proteins) | — | ESM-IF1: inverse folding trained on 12M AlphaFold2-predicted structures; predicted backbones carry sub-Å artefacts that "may give away" the target, so 0.1 Å noise is added (p. 15); more predicted data helps (p. 6). Template for training on upstream-model outputs and for identity leaks. |
| 2024_lu_dynamicbind | 2024 | Nat. Commun. 15:1071 | G2 (analogy, docking) | — | DynamicBind: training decoys morph native → AlphaFold-predicted structure instead of Gaussian noise, which "only" teaches reverting to the pre-noise state (p. 2). |
| 2024_wang_exposure_bias_conformers | 2024 | arXiv 2409.14014 | technique (TD exposure bias) | GD-P | Measures exposure bias in TD/ConfGF and adds input perturbation (TD-IP); QM9 MAT 0.2379 → 0.1924 vs a weak retrained TD baseline (p. 4). |
| 2024_bridging_gap_molecule_diffusion | 2024/25 | ACML 2025, arXiv 2411.05472 | technique (exposure bias) | CrossDocked | DiffGap: aligns training with inference trajectories in 3D SBDD diffusion (p. 1). |
| 2022_chen_analog_bits_selfcond | 2022 | ICLR 2023, arXiv 2208.04202 | technique | — | Self-conditioning: condition on the model's own previous x0 estimate; trained with 50 % zeroed input (p. 3). |
| 2024_chen_diffusion_forcing | 2024 | NeurIPS 2024, arXiv 2407.01392 | technique (joint L+τ) | — | Independent per-token noise levels in one denoiser (p. 1); template for independent L / torsion noise levels. |
| 2023_bao_unidiffuser | 2023 | ICML 2023, arXiv 2303.06555 | technique (joint L+τ) | — | Independent timesteps per modality let one network fit marginal, conditional and joint distributions (p. 4). |
| 2024_campbell_multiflow | 2024 | ICML 2024, arXiv 2402.04997 | technique (joint L+τ) | — | Multiflow: independent structure/sequence times; conditional inpainting by fixing one modality (p. 6). |
| 2023_klein_equivariant_flow_matching | 2023 | NeurIPS 2023, arXiv 2306.15030 | technique (OT coupling) | — | Equivariant OT flow matching: shorter paths, better sampling efficiency (p. 1). |
| 2021_meng_sdedit | 2021 | ICLR 2022, arXiv 2108.01073 | technique | — | SDEdit: noise an imperfect input then denoise; more noise = more realism, less faithfulness (p. 2). |
| 2024_ning_exposure_bias_epsilon_scaling | 2023/24 | ICLR 2024, arXiv 2308.15321 | technique (exposure bias) | — | Epsilon Scaling: training-free exposure-bias fix by scaling down the network output (p. 1). |

Blog snapshot added by SCOUT TRAIN (in `blogs/`): `chen_2024_diffusion_forcing_page.txt` (Diffusion Forcing project
page, accessed 2026-10-12).

### Added by R-CHEM (round 5, 2026-10-15): Exp 2 (matching-free training) and Exp 3 (stereo)
| File | Year | Venue | Gaps | Protocol | Takeaway |
|---|---|---|---|---|---|
| 2025_havens_adjoint_sampling | 2025 | arXiv 2504.11713 (v3) | G2 (data-free) | SPICE; GEOM-DRUGS subset at δ = 1.25 Å | Adjoint Sampling: diffusion sampler trained from an energy model only; a torsional variant samples torsions given bond lengths/angles with no conformer data (p. 12). Prior art for matching-free, energy-trained torsion models. |
| 2025_debortoli_distributional_diffusion | 2025 | ICML 2025, arXiv 2502.02483 | technique (distribution loss) | — | Replaces the diffusion regression loss with energy-score / kernel scoring rules so the denoiser learns p(x0 \| xt) (p. 3). Loss family for Exp 2 variant C. |
| 2024_akhoundsadegh_idem | 2024 | ICML 2024, arXiv 2402.06121 | technique (Boltzmann sampling) | toy n-body | iDEM: diffusion sampler for unnormalised densities from a replay buffer and an MC energy-score estimate (p. 1); costly in energy calls (Adjoint Sampling p. 11). |
| 2023_lou_reflected_diffusion | 2023 | ICML 2023, arXiv 2304.04740 | technique (constrained kernel) | images | Reflected diffusion: reflected SDE + constrained DSM keep samples in a bounded domain (p. 1). Basis for an E/Z half-circle kernel (Exp 3 K2). |
| 2023_fishman_constrained_domains | 2023 | TMLR 2023, arXiv 2304.05364 | technique (constrained kernel) | — | Diffusion on constrained domains (log-barrier, reflected Brownian motion). |
| 2025_balcerak_energy_matching | 2025 | arXiv 2504.10612 | technique (EBM + flow) | images, proteins | Energy Matching: unifies flow matching and EBMs; the objective EnFlow uses for its energy model. |

Code/blog snapshot added by R-CHEM (in `blogs/`): `loqi_repo_stereo.txt` (isayevlab/LoQI @ 04f3caef: E/Z and
tetrahedral auxiliary stereo edges, stereo metric; accessed 2026-10-15). LoQI PDF still blocked (403).

**Round 5, added by R-JOINT (2026-10-15; PDF + `pdftotext -layout` fulltext with form feeds):**

| File (related/) | Year | Venue | Gaps | Protocol | One-line takeaway |
|---|---|---|---|---|---|
| 2019_noe_boltzmann_generators | 2019 | Science 2019, arXiv 1812.01729 | technique (BG) | — | Original Boltzmann generator; Cartesian training distorted bonds/angles, fixed by normalised internal coordinates (p. 14, 25). A BG over bonds/angles is not new. |
| 2024_klein_transferable_boltzmann | 2024 | NeurIPS 2024, arXiv 2406.14426 | technique (BG) | dipeptides | First transferable BG; says TD's BG is conditioned on one fixed local structure and cannot sample the full Euclidean Boltzmann distribution (p. 2). |
| 2025_boltznce | 2025 | NeurIPS 2025, arXiv 2507.00846 | technique (BG likelihood) | dipeptides | Exact-likelihood Jacobians are impractical for large systems; learned EBM likelihood ~100x faster (p. 1). |
| 2026_schopmans_cg_boltzmann | 2026 | arXiv 2602.10637 | technique (BG) | peptides | Overlap and ESS collapse as system size grows (p. 1, 5). |
| 2026_geom_drugs_missing_conformers | 2026 | arXiv 2610.12523 | eval | GEOM-DRUGS | 2.3-7.0 % of generated conformers removed for reflection / stereo / post-relaxation connectivity change; reference sets are incomplete (p. 1). |
| 2026_jin_octmole | 2026 | arXiv 2610.16586 | none (de novo, multiscale) | QM9/DRUG generation | Multiscale octree flow for de novo generation; no torsion or local-structure factor (p. 1-2). |
| 2023_qiang_hierdiff | 2023 | ICML 2023, arXiv 2305.13266 | none (de novo, coarse-to-fine) | — | Coarse-to-fine fragment diffusion for de novo 3D molecules; motivation: atom-level models get rings wrong (p. 1). |

R-JOINT search log and scripts: `round5/ledger/joint_data/` (arXiv API, 18 queries).

**Round 5, added by R-SAMPLE (2026-10-15; PDF + `pdftotext -layout` fulltext with form feeds):**

| File (related/) | Year | Venue | Gaps | Protocol | One-line takeaway |
|---|---|---|---|---|---|
| 2026_cao_gains_global_scheduling | 2026 | arXiv 2609.22867 | technique (inference compute allocation) | images | Best-of-K search budget across denoising steps = separable concave integer program, water-filling optimal; adaptive policies have linear worst-case regret, so anchor offline + adapt online (p. 1). |
| 2023_tang_adadiff | 2023/24 | ECCV 2024, arXiv 2309.17074 | technique (adaptive compute) | images | Uncertainty-estimation module decides layer-wise early exit per sampling step (p. 1). |
| 2025_ma_inference_time_scaling_diffusion | 2025 | arXiv 2501.09732 | technique (test-time search) | images | Inference-time scaling by searching over noises with verifiers; step-count gains flatten after a few dozen steps (p. 1). |
| 2021_jolicoeur_gotta_go_fast | 2021 | arXiv 2105.14080 | technique (adaptive step size) | images | Adaptive-step-size SDE solver for score-based models replaces fixed-step Euler–Maruyama (p. 1). |
| 2013_koukos_good_turing_md | 2013 | arXiv 1309.1108 | convergence estimator | MD | Good–Turing estimate of the probability that an unseen configuration (RMSD > threshold from all seen) exists; a published sampling-convergence signal (p. 2). |
| 2026_uncertainty_molecular_diffusion | 2026 | arXiv 2606.13451 | technique (uncertainty, test-time filtering) | QM9 / GEOM-Drugs de novo | Laplace-based per-sample uncertainty for molecular diffusion; oversample+filter helps on QM9, not on GEOM-Drugs vs random subsampling (p. 1, p. 4). |

Snapshots added by R-SAMPLE (in `blogs/`): `omega_opt_params_snapshot.txt` (+ `omega_raw.html`; OMEGA `-maxConfRange`
rotor-based conformer caps, `-ewindow`, `-maxconfs`), `crest_docs_keywords_snapshot.txt` (+ `crest_raw.html`; CREST
`--mrest` restart cycles, `--scthr`/`--ssthr` ensemble-growth convergence thresholds). The CREST PCCP 2020 PDF
(Pracht, Bohle, Grimme, doi 10.1039/C9CP06869D) could not be downloaded (publisher bot page). Analysis data:
`round5/ledger/sample_data/`.
