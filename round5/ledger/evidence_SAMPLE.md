# Evidence ledger: R-SAMPLE (Exp 5 adaptive sampling + shared evaluation protocol / Phase 5)

Agent tag SAMPLE. Round 5. Started 2026-10-15. Template: `orchestration/templates/evidence_entry.md`.

Conventions
- Paper quotes checked against `pdftotext -layout -f N -l N <pdf>` (PDF page N, 1-based); printed page given when it differs.
- `[δ]` stands for the Greek delta lost in extraction.
- Entries copied from round 4 carry "Reused from: round4/ledger/evidence_BENCH.md E-BENCH-NNN". **No verifier has finished
  the round-4 BENCH ledger** (round4/ledger/verify_V1.md covers other ledgers); quotes were RE-CHECKED by SAMPLE against
  the PDF text on 2026-10-15 where marked "re-checked".
- Computations are marked [SAMPLE-calc]; scripts in the session scratchpad
  (`C:\Users\HP\AppData\Local\Temp\claude\C--Users-HP-Desktop-tor-diff\ebb0c292-63dd-41c3-ab3b-16fa6f4d7fcc\scratchpad\sample\`).
- Status: P1 complete 2026-10-15; 68 entries (A 001–005, B 010–015, C 040–043, D 030–034, E 020–026, G 050–053,
  F 061–091 reused from BENCH; H 054–058 and D 035 added in the consult round 2026-10-16). Spec files: round5/specs/EXP-5.md, round5/specs/EVAL.md.

## A. Prior art: steps / budgets in torsional diffusion
### E-SAMPLE-001
- Claim: TD's steps ablation (GEOM-DRUGS, δ = 0.75 Å, TD-P) gives COV-R mean / AMR-R mean 42.9 / 0.820 (3 steps), 58.9 / 0.698 (5), 70.6 / 0.600 (10), 72.7 / 0.582 (20, default), 73.1 / 0.578 (50); COV-P 24.1, 35.8, 50.2, 55.2, 57.6.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 27 (printed 27), Table 9
- Exact quote:
  > "Table 9: Ensemble RMSD results on GEOM-DRUGS for varying number of diffusion steps. 20 steps were used for all results reported elsewhere." … "10 70.6 78.8 0.600 0.580 50.2 48.3 0.827 0.791" … "50 73.1 80.4 0.578 0.557 57.6 60.7 0.753 0.699"
- Shows vs infers: SHOWS. INFERENCE: going from 20 to 50 steps (2.5x compute) buys +0.4 COV-R and −0.004 Å AMR-R on DRUGS, while halving to 10 steps costs −2.1 COV-R / +0.018 Å. Step count is a weak lever above 10 steps; sample count is the lever to test (section E).
- Supports: EXP-5 §2, §5 (step-count arm)
- Confidence: high

### E-SAMPLE-002
- Claim: TD's authors state that most of the gain is reached with 10 steps and that 20 → 50 steps gives only minor gains.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 28, App. H "Reverse diffusion steps"
- Exact quote:
  > "the majority of gain in performance over prior diffusion-based methods is attained with only 10 steps. We confirm that increasing the number of steps from the default of 20 to 50 only results in minor performance gains."
- Shows vs infers: SHOWS
- Supports: EXP-5 §2
- Confidence: high

### E-SAMPLE-003
- Claim: TD fixes the number of generated conformers per molecule at 2K, where K is the number of ground-truth (CREST) conformers, i.e. the per-molecule sample budget of the benchmark is set from test-set reference information.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 8, Sec. 4.2 "Evaluation"; PDF p. 17, App. C
- Exact quote:
  > "Following the literature, we generate 2K conformers for a molecule with K ground truth conformers." … "set K to be twice the number of conformers returned by CREST."
- Shows vs infers: SHOWS (quotes re-checked 2026-10-15; same text as E-BENCH-002/003). INFERENCE: the "2K rule" is itself an allocation that uses oracle information (K); an adaptive allocator that does not know K must be compared with K-free fixed rules, not only with 2K (E-SAMPLE-020).
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-002, E-BENCH-003 (not yet verified by a round-4 verifier; re-checked by SAMPLE)
- Supports: EXP-5 §1, §5; EVAL §1
- Confidence: high

### E-SAMPLE-004
- Claim: TD reports AMR by number of rotatable bonds on GEOM-DRUGS (Figure 8) and states that error rises with flexibility, and that for very few rotatable bonds TD's error is set by the given local structures.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 25, App. H "Performance vs size", Figure 8
- Exact quote:
  > "With very few rotatable bonds, the error of torsional diffusion depends mostly on the quality of the local structures it was given, and therefore it has a similar error as RDKit."
- Shows vs infers: SHOWS (re-checked 2026-10-15)
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-045
- Supports: EVAL §4 (rotatable-bond strata)
- Confidence: high

### E-SAMPLE-005
- Claim: TD evaluates its DRUGS-trained model on GEOM-XL (average 136 atoms, 32 rotatable bonds), described as out of distribution for DRUGS-trained models, and reports AMR only (Table 6: AMR-R mean RDKit 2.92, GeoMol 2.47, TD 2.05 Å; label/number rows offset by one line in the extraction).
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 25, App. H "Performance vs size" (text); PDF p. 26, Table 6
- Exact quote:
  > "The trend continues with the very large molecules in GEOM-XL (average 136 atoms and 32 rotatable bonds). These not only are larger and more flexible, but--for machine learning models trained on GEOM-DRUGS--are also out of distribution."
- Shows vs infers: SHOWS (table row mapping medium confidence: three method labels, three numeric rows, first numeric row printed beside the GeoMol label). INFERENCE: GEOM-XL with TD's released DRUGS checkpoint is a free size-OOD set once DRUGS is staged (EVAL §5).
- Supports: EVAL §5
- Confidence: high (text); medium (table mapping)

## B. Prior art: adaptive / budget-aware sampling in diffusion models
### E-SAMPLE-010
- Claim: GAINS (Cao, Tang, Li, Zheng, arXiv Sept 2026) treats inference-time search in diffusion sampling (best-of-K candidates per denoising step, scored by a verifier) as a budget-allocation problem across the steps of a trajectory; for fixed sensitivities the optimum is a separable concave integer program with water-filling structure, and its advantage over uniform allocation grows with sensitivity dispersion.
- Source type: paper
- Locator:
  - paper: papers/related/2026_cao_gains_global_scheduling.pdf, PDF p. 1, Abstract
- Exact quote:
  > "given a fixed budget of function evaluations, how should search effort be distributed across the steps of the denoising trajectory?" … "the optimal allocation solves a separable concave integer program with water-filling structure."
- Shows vs infers: SHOWS. INFERENCE: our across-molecule allocation is the same mathematical object (separable concave gains per molecule, greedy/water-filling is optimal); the allocation algorithm itself is therefore prior art and cannot be our contribution. GAINS allocates across steps of one trajectory with a verifier; it does not allocate the number of samples across inputs and does not address set-level (coverage) objectives.
- Supports: EXP-5 §2 (novelty), §4 (greedy allocator)
- Confidence: high

### E-SAMPLE-011
- Claim: GAINS proves that when sensitivities vary across instances no adaptive policy avoids worst-case regret linear in trajectory length, and recommends anchoring the allocation offline and adapting online only for instance-specific slack; it matches uniform-allocation quality with 20–50 % fewer function evaluations.
- Source type: paper
- Locator:
  - paper: papers/related/2026_cao_gains_global_scheduling.pdf, PDF p. 1, Abstract
- Exact quote:
  > "no adaptive policy can avoid worst-case regret that grows linearly in the trajectory length, which motivates a design that anchors the allocation offline and adapts online only to recover instance-specific slack."
- Shows vs infers: SHOWS. INFERENCE: the same design ("static offline allocation + small online correction") is what our data favours (E-SAMPLE-021/022): a cross-fitted static rule captures most of the reachable gain, online GT-free signals add ≈ 0.
- Supports: EXP-5 §2, §4
- Confidence: high

### E-SAMPLE-012
- Claim: AdaDiff (ECCV 2024) allocates computation per sampling step by early-exiting network layers, using a timestep-aware uncertainty estimation module as the exit criterion (compute per step, not samples per input).
- Source type: paper
- Locator:
  - paper: papers/related/2023_tang_adadiff.pdf, PDF p. 1, Abstract (arXiv 2309.17074v3)
- Exact quote:
  > "we propose AdaDiff, an adaptive framework that dynamically allocates computation resources in each sampling step" … "This uncertainty measurement serves as an indicator for determining whether to terminate the inference process."
- Shows vs infers: SHOWS. INFERENCE: uncertainty-driven adaptive compute in diffusion is established prior art; "uncertainty-aware" alone is not a contribution.
- Supports: EXP-5 §2
- Confidence: high

### E-SAMPLE-013
- Claim: Ma et al. (2025) scale diffusion inference-time compute beyond denoising steps by searching over initial noises with verifiers and search algorithms, noting that gains from more denoising steps flatten after a few dozen steps.
- Source type: paper
- Locator:
  - paper: papers/related/2025_ma_inference_time_scaling_diffusion.pdf, PDF p. 1, Abstract (arXiv 2501.09732v1)
- Exact quote:
  > "the performance gains typically flatten after a few dozen." … "we consider a search problem aimed at identifying better noises for the diffusion sampling process. We structure the design space along two axes: the verifiers used to provide feedback"
- Shows vs infers: SHOWS (images). INFERENCE: best-of-N with a verifier is the per-sample analogue of our "oversample then select" arm (E-SAMPLE-025); for conformer *sets* the verifier must be set-level (coverage) and GT-free, e.g. energy or diversity.
- Supports: EXP-5 §2
- Confidence: high

### E-SAMPLE-014
- Claim: Jolicoeur-Martineau et al. (2021) replace the fixed-step Euler–Maruyama solver of score-based models with an SDE solver with adaptive step sizes (per-sample step control), i.e. adaptive step size for diffusion sampling is prior art.
- Source type: paper
- Locator:
  - paper: papers/related/2021_jolicoeur_gotta_go_fast.pdf, PDF p. 1, Abstract (arXiv 2105.14080v1)
- Exact quote:
  > "Existing approaches rely on the Euler-Maruyama (EM) solver, which uses a fixed step size." … "we carefully devise an SDE solver with adaptive step sizes"
- Shows vs infers: SHOWS. INFERENCE: an adaptive step-count arm in Exp 5 is not novel by itself; with TD's flat steps curve above 10 steps (E-SAMPLE-001) it is also low-value.
- Supports: EXP-5 §2, §6
- Confidence: high

### E-SAMPLE-015
- Claim: Seij, Naesseth, Mandt & Jazbec (arXiv June 2026) estimate per-sample uncertainty of pretrained molecular diffusion models (Laplace approximation of the denoiser, variability of noise predictions along the trajectory) and use it for test-time scaling by oversampling and filtering; it helps on QM9 de novo generation but not on GEOM-Drugs.
- Source type: paper
- Locator:
  - paper: papers/related/2026_uncertainty_molecular_diffusion.pdf, PDF p. 1 (Abstract) and PDF p. 4, Figure 2 caption (arXiv 2606.13451v1)
- Exact quote:
  > "We propose a post-hoc method for estimating per-sample uncertainty in pretrained molecular diffusion models." … "In contrast to QM9, neither uncertainty- nor NLL-based filtering meaningfully improves molecular stability, atom stability, or validity over random subsampling"
- Shows vs infers: SHOWS (de novo 3D molecule generation, EDM/GeoLDM; not conformer generation). INFERENCE: model-uncertainty signals for molecular diffusion exist (prior art for "uncertainty-aware" sampling) and already failed to beat random subsampling on the larger DRUGS set; Exp 5 must test its uncertainty signal against random and static baselines, not assume it helps.
- Supports: EXP-5 §2, §4 (signal S1), §11
- Confidence: high

## C. Prior art: when classical conformer searches stop (CREST, OMEGA, MD convergence)
### E-SAMPLE-040
- Claim: OpenEye OMEGA sets the maximum number of output conformers per molecule from the number of rotatable bonds (`-maxConfRange` with `-rangeIncrement`), keeps conformers within an energy window of the global minimum (`-ewindow`, default 10 kcal/mol classic), and caps output at `-maxconfs` (default 200), keeping the lowest-energy ones.
- Source type: blog (vendor documentation)
- Locator:
  - blog: https://docs.eyesopen.com/applications/omega/omega/omega_opt_params.html, OpenEye Scientific, OMEGA documentation "Optional Parameters", options `-maxConfRange`, `-ewindow`, `-maxconfs`; accessed 2026-10-15, snapshot papers/blogs/omega_opt_params_snapshot.txt (raw HTML papers/blogs/omega_raw.html)
- Exact quote:
  > "The -maxConfRange flag allows the user to specify the maximum number of conformers to be output for a structure based on the number of rotatable bonds in the structure."
- Shows vs infers: SHOWS. INFERENCE: a rotor-count allocation rule is standard practice in the commercial baseline TD compares against; our "static n_rot strata" allocator (E-SAMPLE-022/023) is this rule, so any Exp 5 controller must beat it, not only TD's 2K rule.
- Supports: EXP-5 §2, §5 (baseline B3)
- Confidence: high

### E-SAMPLE-041
- Claim: OMEGA's `-maxconfs` output is the lowest-energy conformers among those passing the energy-window and RMSD-duplicate filters; output can be smaller than `-maxconfs` for rigid molecules.
- Source type: blog (vendor documentation)
- Locator:
  - blog: same page, option `-maxconfs`, snapshot papers/blogs/omega_opt_params_snapshot.txt
- Exact quote:
  > "For molecules with few conformational degrees of freedom, the number of output conformers might be lower than the value set for this flag but will never exceed it."
- Shows vs infers: SHOWS. INFERENCE: classical tools stop by saturation of distinct (RMSD-deduplicated) low-energy conformers, not by a fixed multiple of an unknown reference count; an energy-ranked selection arm (E-SAMPLE-025) mirrors this.
- Supports: EXP-5 §2, §4
- Confidence: high

### E-SAMPLE-042
- Claim: CREST's iMTD-GC workflow restarts its metadynamics stage when sampling finds a new lower conformer, up to a maximum number of restart cycles (`--mrest`, default 5); its newer `--v4` / `--entropy` modes stop when ensemble growth (% new conformers, default 5 % for v4) and entropy gain (default 1 % for v4) fall below thresholds.
- Source type: blog (developer documentation)
- Locator:
  - blog: https://crest-lab.github.io/crest-docs/page/documentation/keywords.html, CREST developers, "Keywords" page, options `--mrest`, `--scthr`, `--ssthr`; accessed 2026-10-15, snapshot papers/blogs/crest_docs_keywords_snapshot.txt (lines 567-568, 649-660)
- Exact quote:
  > "Adjust the iterative behaviorof iMTD-GC by setting the maximum number of possible MTD restart cycles. The [default] is 5 cycles." … "Specify the ensemble growth threshold (% new conformers) for --entropy and --v4 convergence."
- Shows vs infers: SHOWS (the restart-on-new-lower-conformer rule is stated by the xtb documentation, https://xtb-docs.readthedocs.io/en/latest/crestqcg.html, seen in a search snippet only, not snapshotted: UNVERIFIED). The CREST PCCP 2020 paper (Pracht, Bohle, Grimme) could not be downloaded (publisher returned an HTML bot page); its algorithm text is not quoted here.
- INFERENCE: "stop when the fraction of new conformers per batch falls below a threshold" is the established stopping rule for the reference method that produced GEOM itself; an ensemble-growth stopping rule for TD is therefore prior art in spirit, and a novelty claim needs a different mechanism or a matched-compute win.
- Supports: EXP-5 §2, §4 (signal S3 "new-mode rate"), §11
- Confidence: high (quote); medium (restart rule)

### E-SAMPLE-043
- Claim: Koukos & Glykos (2013) apply Good–Turing estimation of unobserved species to MD trajectories, using the RMSD matrix, to estimate the probability that a configuration farther than a threshold from all observed ones has not yet been seen.
- Source type: paper
- Locator:
  - paper: papers/related/2013_koukos_good_turing_md.pdf, PDF p. 2, Abstract (arXiv 1309.1108)
- Exact quote:
  > "we use the classical RMSD matrix as a means to answering the following question : "What is the probability that a molecular configuration with an RMSD (from all other already observed configurations) higher than a given threshold has not actually been observed ?""
- Shows vs infers: SHOWS. INFERENCE: the Good–Turing "missing mass" signal we tested (E-SAMPLE-023) is a published convergence estimator for molecular sampling; using it as a stopping signal for TD is incremental at best, and on our QM9 data it did not beat a static rotor rule.
- Supports: EXP-5 §2, §4
- Confidence: high

## D. Prior art: Particle Guidance and diversity-aware joint sampling
### E-SAMPLE-030
- Claim: Particle guidance (Corso et al., ICLR 2024) replaces i.i.d. sampling of the 2K conformers with joint sampling under a time-evolving repulsive potential; TD + PG on GEOM-DRUGS reports COV-R 77.0 / AMR-R 0.543 / COV-P 68.9 / AMR-P 0.656 vs TD 72.7 / 0.582 / 55.2 / 0.778, without retraining.
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_particle_guidance.pdf, PDF p. 8, Table 1 and Sec. 6.2
- Exact quote:
  > "Torsional Diffusion 72.7 80.0 0.582 0.565 55.2 56.9 0.778 0.729" … "TD w/ particle guidance 77.0 82.6 0.543 0.520 68.9 78.1 0.656 0.594"
- Shows vs infers: SHOWS. NExT-Mol reproduces TD + PG lower (73.8 / 0.566; E-BENCH-017, not re-verified here).
- Supports: EXP-5 §2 (closest prior art; mandatory baseline), §6
- Confidence: high

### E-SAMPLE-031
- Claim: PG's guidance kernel for TD is an RBF over torsion-angle differences, made invariant to atom permutations by taking the minimum over perturbations.
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_particle_guidance.pdf, PDF p. 8, Sec. 6.2
- Exact quote:
  > "we also define the guidance kernel on this manifold as an RBF kernel over the dihedral angle differences." … "we employ the simple strategy to take the minimum value of the original kernel under the different perturbations"
- Shows vs infers: SHOWS. INFERENCE: PG acts within a molecule's fixed sample set (who to sample); Exp 5 acts across molecules (how many to sample). They compose; a torsion-diversity *signal* for allocation reuses PG's kernel.
- Supports: EXP-5 §2, §4
- Confidence: high

### E-SAMPLE-032
- Claim: PG motivates non-i.i.d. sampling with the expected number of modes captured by i.i.d. sampling of N equiprobable modes, N(1 − ((N−1)/N)^x), against the optimum min(x, N).
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_particle_guidance.pdf, PDF p. 21, App. B, Figure 5 caption
- Exact quote:
  > "Plot of the functions y = N (1 - ( N-1 N )x) and y = min(x, N ) for N = 1000 represent- ing, respectively, the expected number of modes captured by I.I.D. sampling"
- Shows vs infers: SHOWS (fraction rendering lost in extraction). INFERENCE: the same saturating curve is what a per-molecule allocator exploits; the concavity that makes greedy allocation optimal is this diminishing-returns shape (E-SAMPLE-010, our curves E-SAMPLE-020).
- Supports: EXP-5 §3, §4
- Confidence: high

### E-SAMPLE-033
- Claim: PG caps its joint set at 128 conformers on DRUGS and runs batches of 128 until the 2K total is reached, because kernel cost is quadratic in the set size and some molecules need thousands of conformers.
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_particle_guidance.pdf, PDF p. 21, App. B.2 "Runtime and memory overhead"
- Exact quote:
  > "different molecules can have very different numbers of conformers with some even having thousands of them. For efficiency, we therefore cap the size of n in particle guidance to 128 and perform batches of 128 samples"
- Shows vs infers: SHOWS. INFERENCE: PG keeps TD's per-molecule 2K budget; it never changes how many samples a molecule receives.
- Supports: EXP-5 §2, §8 (batching)
- Confidence: high

### E-SAMPLE-034
- Claim: PG states that conformer generators are run more than 200 times per molecule on average on GEOM-DRUGS to produce i.i.d. samples (the 2K budget), and argues this is suboptimal for covering the distribution.
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_particle_guidance.pdf, PDF p. 8, Sec. 6.2
- Exact quote:
  > "running this model several times (more than 200 on average in the standard GEOM-DRUGS dataset) to generate a large number of I.I.D. samples."
- Shows vs infers: SHOWS. INFERENCE: a DRUGS 1x pool is ≈ 200k conformers for 1000 test molecules; DRUGS sampling cost estimates in EXP-5 §10 scale from this (UNVERIFIED until measured).
- Supports: EXP-5 §10
- Confidence: high

## E. Our data: what a controller could exploit (QM9, released TD model)
All entries: released TD QM9 model (`qm9_default`, 20 steps), sampling seeds 0/1/2 pooled (pool N_i = 3 x 2K_i),
927 molecules evaluated in all three runs (R0 935, R1/R2 931 evaluated; 1 molecule with an additional failure
dropped). Values are on this success set (failures excluded), NOT the TD convention. Scripts and outputs:
`round5/ledger/sample_data/` (README there). ORACLE rows use test-set ground truth and are analysis only.

### E-SAMPLE-020
- Claim: [SAMPLE-calc] Recall keeps improving well beyond TD's 2K samples: expected AMR-R 0.1783 Å at 1 x 2K, 0.1535 at 1.5x, 0.1400 at 2x, 0.1255 at 3x; COV-R@0.5 94.93 → 97.04 → 97.96 → 98.78 %; COV-R@0.1 39.50 → 45.61 → 49.87 → 55.40 %; at 0.5x AMR-R is 0.2399.
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/alloc.out, block "TD rule scaled: n_i = round(s * n_model_i)"
- Exact quote:
  > "s=1.0: budget/B=1.000 COV-R@0.5=0.9493 COV-R@0.1=0.3950 AMR-R=0.1783" … "s=2.0: budget/B=2.000 COV-R@0.5=0.9796 COV-R@0.1=0.4987 AMR-R=0.1400" … "s=3.0: budget/B=3.000 COV-R@0.5=0.9878 COV-R@0.1=0.5540 AMR-R=0.1255"
- Shows vs infers: SHOWS (exact hypergeometric expectation over random n-subsets of the pooled samples, `curves.py`). Sanity: realised 1x value 0.1773–0.1784 (select.out, alloc2.out); run R0 alone gives 0.1752 on its own 935 molecules.
- Supports: EXP-5 §3; EVAL §2 (sample-count reporting)
- Confidence: high

### E-SAMPLE-021
- Claim: [SAMPLE-calc] Gains from more samples are larger for flexible molecules but present in every stratum, including rigid ones: AMR-R change 1x → 3x is −0.031 Å (0 rotatable heavy bonds, n = 278), −0.047 (1, 294), −0.071 (2, 173), −0.074 (3, 93), −0.085 (4+, 89); K = 1 molecules −0.037. Spearman of the per-molecule gain with n_rot is 0.47, with K 0.27; the top 20 % of molecules hold 48 % of the total gain.
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/strata.out ("### by rot_bin", "### by K_bin", last lines)
- Exact quote:
  > "0        278    0.1566    0.1258      -0.0308" … "4+        89    0.2346    0.1501      -0.0845" … "share of total dAMR(1x->3x) from top 20% molecules: 0.4795466172552433"
- Shows vs infers: SHOWS. INFERENCE: for rigid molecules extra samples help because each TD sample starts from a fresh RDKit seed, i.e. extra samples are extra draws of local structure L (a "local-structure lottery"); this links Exp 5 to the ring-geometry error found in rounds 1–4.
- Supports: EXP-5 §3, §4; EVAL §4
- Confidence: high

### E-SAMPLE-022
- Claim: [SAMPLE-calc] At matched total samples (Σ 2K_i), TD's K-proportional rule is a poor allocator for molecule-averaged recall: a uniform allocation that ignores K gives AMR-R 0.1450 vs 0.1784 and COV-R@0.5 96.95 vs 95.05 %; the ORACLE greedy allocator (AMR objective) gives 0.1400 / 98.09 %; a cross-fitted static n_rot-strata rule 0.1442 / 97.65 %. At 0.5x budget: TD rule 0.2414, uniform 0.1804, static 0.1767, ORACLE 0.1687; at 0.25x: 0.3122, 0.2401, 0.2368, 0.2178.
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/alloc2.out (realised prefixes, 5 random pool permutations)
- Exact quote:
  > "1.0x sum(2K)              TD 2K rule (uses K) 0.1784     0.9505" … "1.0x sum(2K)                          uniform 0.1450     0.9695" … "1.0x sum(2K)  static n_rot strata (cross-fit) 0.1442     0.9765" … "1.0x sum(2K)          ORACLE greedy (AMR obj) 0.1400     0.9809"
- Shows vs infers: SHOWS. Caveat: every allocator is capped at the pool size N_i = 6K_i (the cap binds for 24–71 % of molecules under uniform, column "share at pool cap"), so the cap leaks K into all rows and the oracle headroom is a lower bound; uncapped curves need a larger sample pool (EXP-5 §5 step S0). Expected precision is unchanged by random-subset reallocation (the expectation of a per-sample mean does not depend on n), so these allocations trade nothing on COV-P/AMR-P in expectation.
- Supports: EXP-5 §3 (headroom), §5 (baselines must include uniform and static rules); EVAL §1
- Confidence: high (calculation); medium (cap caveat)
- Revised after D-101 (V5, 2026-10-16): "uniform (no K)" is relabelled "uniform with cap redistribution (K-informed)": the cap N_i = 6K_i tells every allocator which molecules saturate. The strictly K-free figure is E-SAMPLE-026. The static and two-stage rules were cross-fitted on other-fold TEST curves, so 0.1442 / 0.1443 are analysis numbers, not deployable baselines.

### E-SAMPLE-023
- Claim: [SAMPLE-calc] ORACLE headroom over the best K-free static rule at matched compute is small: AMR-R −0.0042 Å (0.1442 → 0.1400) and COV-R@0.5 +0.44 points at 1x Σ2K; −0.0080 Å / +1.41 points at 0.5x; −0.0190 Å / +3.74 points at 0.25x. A two-stage GT-free signal controller (Good-Turing singleton fraction of rotamer strings on 6 stage-1 samples, cross-fitted) reaches 0.1443 / 0.1772 / 0.2365, i.e. it recovers none of the headroom at 1x and 0.5x; a purely sequential Good-Turing controller is worse than uniform (0.1694 at 1x).
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/alloc2.out
- Exact quote:
  > "0.5x sum(2K)  static n_rot strata (cross-fit) 0.1767" … "0.5x sum(2K)     two-stage signal (cross-fit) 0.1772" … "0.5x sum(2K)          ORACLE greedy (AMR obj) 0.1687" … "1.0x sum(2K) Good-Turing sequential (GT-free) 0.1694"
- Shows vs infers: SHOWS. INFERENCE: an adaptive controller has at most ≈ 0.004–0.008 Å AMR-R (≈ 3–5 %) to gain over a static rule at TD-like budgets on QM9, and simple GT-free signals do not find it; DRUGS (more flexible, wider K spread) may have more headroom and must be measured before any claim.
- Supports: EXP-5 §0, §3, §11 (kill criterion)
- Confidence: high (calculation); medium (transfer to DRUGS)
- Revised after D-101 (V5, 2026-10-16): the INFERENCE "at most ≈ 0.004–0.008 Å" is WITHDRAWN. Because the cap removes exactly the saturation information an adaptive controller must discover, the capped oracle − static gap (0.0042 / 0.0080 / 0.0190 Å) is a LOWER bound on the uncapped headroom, not an upper bound. The uncapped value is to be recomputed on the K-free 128-sample block (EXP-5 §5 S0, ORACLE, CPU).

### E-SAMPLE-026
- Claim: [SAMPLE-calc] (added after D-101) A strictly K-free uniform rule (n per molecule fixed, capped at the pool N_i = 6K_i with NO redistribution, i.e. samples beyond the pool assumed worthless) gives AMR-R 0.1617 Å / COV-R@0.5 94.90 % at n = 27 while using only 68.9 % of the Σ2K budget (54 % of molecules capped); at n = 55 (105 % of budget) 0.1432 / 97.12 %. TD's 2K rule at 100 % of the budget gives 0.1783 / 94.93 % (E-SAMPLE-020).
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/strictcap.out (script strictcap.py on curves.pkl); reproduces V5's `capcheck.py` figure quoted in round5/ledger/disputes.md D-101
- Exact quote:
  > "uniform n=27: budget used/B=0.689 capped share=0.540 AMR-R=0.1617 COV-R@0.5=0.9490"
- Shows vs infers: SHOWS. INFERENCE: the direction of E-SAMPLE-022 survives without any K leak: a K-free uniform rule beats TD's K-proportional rule on AMR-R with 31 % less compute (pessimistic for uniform, since capped molecules get no further samples). The 0.1450 figure of E-SAMPLE-022 is "uniform with cap redistribution (K-informed)".
- Supports: EXP-5 §0, §3; EVAL §1; D-101
- Confidence: high

### E-SAMPLE-024
- Claim: [SAMPLE-calc] GT-free per-molecule signals measured on the first batch correlate with the oracle gain (1x → 3x AMR-R) about as well as the static rotatable-bond count: Spearman 0.451 (Good-Turing singleton fraction of rotamer strings, 6 samples), 0.446 (distinct rotamers in 2K samples), 0.441 (MMFF94 single-point energy SD), 0.474 (n_rot), 0.273 (K, oracle).
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/alloc2.out (last five lines)
- Exact quote:
  > "spearman(signal=GT-missing(first 6), oracle AMR gain 1x->3x) = 0.451" … "spearman(signal=MMFF E sd in 1x, oracle AMR gain 1x->3x) = 0.441" … "spearman(signal=n_rot, oracle AMR gain 1x->3x) = 0.474"
- Shows vs infers: SHOWS. INFERENCE: the cheap online signals are mostly proxies for flexibility; any Exp 5 signal must show *incremental* predictive value over n_rot (partial correlation / ablation), otherwise the controller reduces to OMEGA's rotor-based rule (E-SAMPLE-040).
- Supports: EXP-5 §4, §6
- Confidence: high

### E-SAMPLE-025
- Claim: [SAMPLE-calc] In the protocol-preserving framing (output exactly 2K per molecule, chosen from c x 2K proposals), GT-free selection is where extra compute pays: torsion farthest-point selection gives AMR-R 0.1656 (2x) vs 0.1778 random but lowers COV-P@0.5 (91.20 → 89.68 %); lowest-MMFF-energy selection worsens AMR-R (0.1872) but raises COV-R@0.1 (39.87 → 44.08 %) and precision; dropping the 25 % highest-MMFF-energy proposals then torsion FPS improves every metric at 2x (AMR-R 0.1699, COV-R@0.5 95.46 %, COV-R@0.1 42.56 %, AMR-P 0.2119, COV-P@0.5 91.93 %); ORACLE selection bound 0.1398 (2x) / 0.1255 (3x).
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/select.out (3 random pool orders; columns AMR_R, COV_R05, COV_R01, AMR_P, COV_P05)
- Exact quote:
  > "1x      random                        0.1778   0.9490   0.3987 0.2200   0.9120" … "torsion FPS                   0.1656   0.9675   0.3901 0.2365   0.8968" … "MMFF-filter 25% + torsion FPS 0.1699   0.9546   0.4256 0.2119   0.9193"
- Shows vs infers: SHOWS (single 2x/3x pools from 3 sampling seeds; no CI yet). INFERENCE: oversample-then-select is Zhou et al.'s RDKit + clustering idea applied to TD (E-SAMPLE-082/083, prior art) and Ma et al.'s best-of-N with a verifier (E-SAMPLE-013); an energy-filter + diversity selector is a cheap, GT-free Pareto improvement over random at 2x proposals and must be compared at matched NFE with PG (E-SAMPLE-030), which needs no extra proposals.
- Supports: EXP-5 §3, §4 (selector), §6
- Confidence: medium (point estimates; paired CIs to be computed in the experiment)
- Revised after D-102 (V5, 2026-10-16): TEST-SET EXPLORATORY. The selector was picked on the test set from 4 candidates × c ∈ {2, 3} with 3 pool orders and no CI; at c = 3 it no longer improves every metric (COV-P@0.5 91.04 < 91.20). At c = 2, "MMFF-filter 50 % + FPS" keeps exactly 2K proposals and is identical to "lowest MMFF E" (degenerate row). The comparison is at matched OUTPUT, not matched compute vs P0 (2x sampling NFE + MMFF single points); the NFE control is random-from-2x (P1). The selector is re-chosen on validation (EXP-5 §4.4).

## G. Our data for the shared evaluation protocol
### E-SAMPLE-050
- Claim: The leak-free split-half oracle (round 4) has 744 eligible QM9 test molecules (n_clean ≥ 2; 253 single-conformer and 3 no-clean-GT molecules excluded); seed-averaged CTRL_rematch AMR-R is 0.2195 Å split (other-half L) vs 0.1603 own-half L vs 0.1873 RDKit L, and the model-free COPY baseline is 0.3336.
- Source type: our-data
- Locator:
  - our-data: round4/leakfree_data/splithalf_population.txt (`eligible`, `excluded_single_conformer`); round4/leakfree_data/leakfree_summary.txt, "## Per model (seed-averaged ...)" row CR and COPY
- Exact quote:
  > "eligible 744" … "excluded_single_conformer 253" … "CR      0.2195    0.8564    0.5019  0.1603    0.8891    0.6304" … "COPY    0.3336    0.6976    0.3650"
- Shows vs infers: SHOWS (ORACLE analysis). INFERENCE: leak-free values are on half-ensembles and a 744-molecule universe; they are not comparable with full-ensemble numbers and must be printed in their own table (EVAL §6).
- Supports: EVAL §6
- Confidence: high

### E-SAMPLE-051
- Claim: [SAMPLE-calc] Sampling-seed reruns of the released QM9 model change the number of failed molecules (65, 69, 69 for seeds 0, 1, 2) and AMR-R (0.1752, 0.1768, 0.1761 Å), so TD-convention COV-R moves with failures (89.03, 88.18, 88.66 %).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/qm9_default/R0_base_seed0/summary.txt, R1_base_seed1/summary.txt, R2_base_seed2/summary.txt (first line each)
- Exact quote:
  > "COV-R_mean=88.1758 COV-R_median=100.0000 MAT-R_mean=0.1768" … "n_evaluated=931 n_model_failures=69 n_additional_failures=1"
- Shows vs infers: SHOWS (quote from R1; R0/R2 lines analogous). INFERENCE: failures are partly stochastic (ETKDG embedding of seeds), so a paired comparison must fix the molecule universe (union with failures = 0, and intersection) and report failure counts per arm and seed; otherwise a 4-molecule failure difference moves COV-R by ≈ 0.4 points, more than many effects we test.
- Supports: EVAL §3, §7
- Confidence: high

### E-SAMPLE-052
- Claim: [SAMPLE-calc] A scaffold-novel OOD stratum exists inside the TD QM9 test set without retraining: of 1000 test molecules, 87 have a Bemis–Murcko scaffold absent from all 106,586 training molecules (13,583 training scaffolds), 802 are scaffold-seen, 111 acyclic; scaffold-novel molecules fail more often (15/87) and are more rigid (mean n_rot 0.26, K 2.9); released-model AMR-R 0.1690 (novel) vs 0.1682 (seen) on successes, TD-convention COV-R 79.8 vs 90.5 %.
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/ood.out (script ood.py; split.npy and the raw-pickle file list from gnode118 `/scratch/nishanth.r/tordiff/data/QM9/`, read-only)
- Exact quote:
  > "train molecules 106586 distinct train scaffolds 13583" … "scaffold-novel   87        15 0.1690         0.9637 0.2639  2.8611        0.7976"
- Shows vs infers: SHOWS. INFERENCE: n = 87 is small (CI half-width for AMR-R roughly ±0.01 Å); it is a descriptive OOD stratum, not a powered test. A powered OOD claim needs ET-Flow-style scaffold re-splitting with retraining (E-SAMPLE-075), GEOM-XL, or cross-dataset transfer.
- Supports: EVAL §5
- Confidence: high (counts); medium (scaffold definition choice)

### E-SAMPLE-053
- Claim: Generating all 1000 QM9 test molecules with the released TD model (20 steps, 2K samples, 935 molecules succeeding, 25,330 conformers) took 14 min 18 s wall clock on gnode118 in one job (1173 s user CPU).
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/R0_base_seed0_generate.log:2066, :2069 (copy of `/scratch/nishanth.r/tordiff/results/qm9_default/R0_base_seed0/generate.log`)
- Exact quote:
  > "User time (seconds): 1173.38" … "Elapsed (wall clock) time (h:mm:ss or m:ss): 14:18.22"
- Shows vs infers: SHOWS. INFERENCE: ≈ 0.034 s wall per conformer including RDKit seeding, so a 3x-sample QM9 pool costs ≈ 45 min of one GPU; Exp 5 is cheap on QM9. The 25,330 count is the sum of n_model over evaluated molecules in eval.pkl (E-SAMPLE-020 inputs).
- Supports: EXP-5 §10; EVAL §8
- Confidence: high

## H. Added in the consult round (2026-10-16)
### E-SAMPLE-054
- Claim: [SAMPLE-calc] The QM9 test csv used by every arm has no cap on K: n_conformers ranges 1–727 (median 4, mean 13.7), 86 molecules have K > 30 and 80 have 4K > 128; Σ2K = 27,462 over the 1000 rows. Pool totals: 4K → 54,924 conformers (2.0 x Σ2K); max(128, 4K) → 151,088 (5.5x); max(128, 12K) → 232,804 (8.5x); fixed 128 → 128,000 (4.7x).
- Source type: our-data
- Locator:
  - our-data: round5/ledger/sample_data/pool_sizes.out (from a read-only copy of gnode118 `/scratch/nishanth.r/tordiff/data/QM9/test_smiles.csv`)
- Exact quote:
  > "K>32: 80  4K>128: 80 max K 727" … "max(128,4K) 151088 5.5 x sum2K"
- Shows vs infers: SHOWS (last line of pool_sizes.out: "rows 1000 K>30: 86 median K 4.0 mean K 13.731 sum2K 27462"). INFERENCE: a fixed 128-sample pool does not cover c = 2 Track-P demand (4K) for 80 molecules; the "max K = 30" figure in Q-P2-008 comes from a capped local-structure csv, not from the test csv. GPU estimates use the 0.034 s/conformer anchor of E-SAMPLE-053.
- Supports: EXP-5 §5, §10; Q-P1-008, Q-P2-008, Q-P3-005
- Confidence: high

### E-SAMPLE-055
- Claim: TD computes its per-conformer `mmff_energy` and its pre/post-MMFF relaxations with the MMFF94s variant.
- Source type: code
- Locator:
  - code: torsional-diffusion/diffusion/likelihood.py:19-20; torsional-diffusion/diffusion/sampling.py:29
- Exact quote:
  > "energy = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')).CalcEnergy()" … "res = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')"
- Shows vs infers: SHOWS. INFERENCE: the selector should reuse TD's MMFF94s single point (one force field in the programme); the test-set prototype (`sample_data/signals.py`) used RDKit's default MMFF94, a difference to note in conformance.
- Supports: EXP-5 §4.2; Q-P1-009
- Confidence: high

### E-SAMPLE-056
- Claim: TD's PG implementation couples only the conformers of one DataLoader batch: the repulsion term is computed inside the per-batch loop with n = data.num_graphs.
- Source type: code
- Locator:
  - code: torsional-diffusion/diffusion/sampling.py:123 (DataLoader, batch_size), :176 (batch loop), :207 (`n = data.num_graphs`), :229 (repulsion / n)
- Exact quote:
  > "for batch_idx, data in enumerate(loader):" … "n = data.num_graphs"
- Shows vs infers: SHOWS. With E-SAMPLE-033 (PG caps its joint set at 128 and runs batches of 128 on DRUGS) and the README's `--batch_size 128` for every PG command (E-SAMPLE-058), per-128 coupling IS the published PG protocol.
- Supports: EXP-5 §5 (PG arms); Q-P1-009, Q-P2-009, Q-P3-006
- Confidence: high

### E-SAMPLE-057
- Claim: The round-2 analysis pre-declared a non-inferiority margin of 0.004 Å AMR-R, judged with a hierarchical bootstrap.
- Source type: code
- Locator:
  - code: tools/analyze_round2.py:14, :57-58
- Exact quote:
  > "S2 margin : non-inferiority of B6 to CTRL_base on RDKit L, margin 0.004 A AMR-R, judged with a HIERARCHICAL bootstrap"
- Shows vs infers: SHOWS. INFERENCE: the same 0.004 Å margin is reused for the precision side-condition of the selector choice (AMR-P), so no new margin is invented; the COV-P margin (0.5 pt) is a new, declared choice (UNSOURCED, stated as such).
- Supports: EXP-5 §4.4; Q-P1-009, Q-P2-009, Q-P3-005
- Confidence: high

### E-SAMPLE-058
- Claim: The TD README gives four PG settings (permutation-invariant or not × minimising recall or precision error), all for the DRUGS model `workdir/drugs_default` on the DRUGS test csv with `--batch_size 128`, 20 steps; none for QM9.
- Source type: code (repo README)
- Locator:
  - code: torsional-diffusion/README.md:73-88
- Exact quote:
  > "python generate_confs.py --tqdm --batch_size 128 --no_energy --inference_steps=20 --model_dir=workdir/drugs_default --test_csv=data/DRUGS/test_smiles.csv --pg_invariant=True"
- Shows vs infers: SHOWS (quote is the prefix of README line 78). INFERENCE: on QM9 PG has no published setting, so PG gets the same validation-selection budget as our selector (EXP-5 §5).
- Supports: EXP-5 §5; Q-P1-009, Q-P2-009, Q-P3-006
- Confidence: high

### E-SAMPLE-035
- Claim: PG's authors state that fixed-potential PG parameters can be fit cheaply at inference time and that a too-large guidance weight harms sample quality.
- Source type: paper
- Locator:
  - paper: papers/related/2023_corso_particle_guidance.pdf, PDF p. 21, App. B.2 "Other limitations"
- Exact quote:
  > "in fixed potential particle guidance, its parameters can be easily fit at inference time, therefore, it is typically relatively inexpensive to test the optimal value of the guidance"
- Shows vs infers: SHOWS. INFERENCE: tuning the four README settings on the QM9 validation subset is the authors' own recommended practice, and gives PG the same tuning budget as the selector.
- Supports: EXP-5 §5; Q-P1-009, Q-P2-009, Q-P3-006
- Confidence: high

## F. Evaluation protocol (reused from round-4 BENCH; quotes RE-CHECKED 2026-10-15)
Each entry below is copied from `round4/ledger/evidence_BENCH.md` with its original ID kept in the "Reused from" line.
No round-4 verifier has finished the BENCH ledger. R-SAMPLE re-checked every quote fragment against the cited PDF page
(`pdftotext -layout -f N -l N`) or snapshot/code/data file with `round5/ledger/sample_data/recheck.py`; all 31 matched
(`round5/ledger/sample_data/recheck_bench.out`). Card references (C-BENCH-*) are to `round4/cards/`.

### E-SAMPLE-061
- Claim: The TD protocol (TD-P) uses GeoMol's random splits, 243473/30433/1000 molecules for GEOM-DRUGS and 106586/13323/1000 for GEOM-QM9, and filters out molecules whose CREST conformers all have a SMILES different from the molecule's or that RDKit cannot handle.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 23, App. G.1 "Dataset details"
- Exact quote:
  > "train/validation/test of 243473/30433/1000 for GEOM-DRUGS and 106586/13323/1000 for GEOM-QM9." … "molecules whose CREST conformers all have a canonical SMILES different from the SMILES of the molecule" … "or that cannot be handled by RDKit, are filtered out."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01, C-BENCH-05
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-001 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-062
- Claim: TD-P generates 2K conformers for a molecule with K ground-truth conformers and uses δ = 0.75 Å on GEOM-DRUGS (most earlier work, the GeoDiff/ConfGF "GD-P" protocol, used 1.25 Å).
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 8, Table 1 caption and Sec. 4.2 "Evaluation"
- Exact quote:
  > "We compute Coverage with a threshold of [δ] = 0.75 Å to better distinguish top methods." … "most prior works, which used [δ] = 1.25 Å." … "we generate 2K conformers for a molecule with K ground truth conformers."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01, C-BENCH-05, C-BENCH-07
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-002 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-063
- Claim: GeoMol (origin of the splits) used δ = 0.5 Å for GEOM-QM9 and δ = 1.25 Å for GEOM-DRUGS; TD changed only the DRUGS threshold, so QM9 at 0.5 Å is common to both conventions.
- Source type: paper
- Locator:
  - paper: papers/related/2021_ganea_geomol.pdf, PDF p. 9, Sec. 5 (evaluation setup)
- Exact quote:
  > "set [δ] = 0.5Å for GEOM-QM9 and [δ] = 1.25Å for GEOM-DRUGS."
- Shows vs infers: SHOWS. INFERENCE: GeoMol's own DRUGS numbers (δ = 1.25 Å) must not be mixed into TD-P DRUGS tables (δ = 0.75 Å); TD-P tables use TD's re-evaluation of GeoMol (E-BENCH-013).
- Supports card(s): C-BENCH-01, C-BENCH-05, C-BENCH-07
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-004 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-064
- Claim: S23D's DRUGS table records two comparability factors that differ between published models: training conformers per molecule (KG = 20 for MCF and S23D, 30 for TD and ET-Flow) and whether chirality is used or corrected; S23D-B with chirality reaches COV-R 87.0 % / AMR-R 0.380 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 8, Table 1 (columns SIZE, KG, CHIRAL.)
- Exact quote:
  > "`KG' is number of conformers for each molecule in test split used during training. `Chirality' indicates whether chirality information was used" … "TOR. DIFF. 30 YES 72.7 80.0 0.582 0.565" … "S23D-B-1/13 (M) 25 20 YES 87.0 94.0 0.380 0.356"
- Shows vs infers: SHOWS ("test split" is the source's wording; the column is the training cap). INFERENCE: a fair comparison table should state KG, chirality handling and model size per row.
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-005 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-065
- Claim: Our (TD's) evaluator counts a molecule with no generated conformers as 0 % coverage but leaves it out of AMR; it also drops ground-truth conformers whose heavy-atom SMILES differs from the molecule's (`clean_confs`) and skips molecules left with no clean conformer.
- Source type: code
- Locator:
  - code: torsional-diffusion/evaluate_confs.py:47-54 (`clean_confs`), :81-90 (model failure; poor ground truth), :166-171 (COV with `[0] * num_failures`; `np.nanmean`)
- Exact quote:
  > "if conf_smi == smi:" / "good_ids.append(i)" … "print('model failure', corrected_smi)" / "num_failures += 1" … "print(f'poor ground truth conformers: {corrected_smi}')"
- Shows vs infers: SHOWS
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-033 (COV/AMR failure convention, VERIFIED by V1 after D-004); the `clean_confs` part is new.
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-007 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-066
- Claim: ET-Flow's evaluator skips a molecule entirely (no zero coverage) when generated or reference positions are missing, when its SMILES has several fragments, or when fewer than 2K conformers were generated; its COV and AMR are averaged over the molecules that remain.
- Source type: code (third-party repo)
- Locator:
  - code: github.com/shenoynikhil/ETFlow `etflow/commons/covmat.py`:135-140, :151-156, :196-200 (snapshot papers/blogs/etflow_eval_code.txt, accessed 2026-10-11)
- Exact quote:
  > "log.info("skipping due to missing pos_gen or pos_ref")" / "continue" … "log.info("skipping due to disconnected molecule")" … "log.info("skipping due to insufficient number of generated conformers")"
- Shows vs infers: SHOWS. INFERENCE: TD-evaluator and ET-Flow-evaluator coverage numbers differ in failure handling; a model that loses molecules to RDKit (TD and all our arms, 65/1000 on QM9, E-BENCH-011) is penalised in COV by the TD evaluator but would not be by ET-Flow's. Published tables mix the two conventions without saying so.
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-008 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-067
- Claim: ET-Flow's evaluator uses heavy-atom, symmetry-aware `GetBestRMS` by default (NaN on an RDKit exception), scores thresholds 0.05–3.0 Å in 0.05 steps, generates 2 × (number of stored references), and has no `clean_confs` filter on the references.
- Source type: code (third-party repo)
- Locator:
  - code: github.com/shenoynikhil/ETFlow `etflow/commons/covmat.py`:57-69, :113; `scripts/eval.py`:77-79 (snapshot papers/blogs/etflow_eval_code.txt)
- Exact quote:
  > "probe = RemoveHs(probe)" … "rmsd = MA.GetBestRMS(probe, ref)" / "except:  # noqa" / "rmsd = np.nan" … "thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05)," … "num_samples = 2 * count"
- Shows vs infers: SHOWS. INFERENCE: the RMSD definition matches TD's (heavy atoms, GetBestRMS, E-BENCH-007), but the reference set can differ where GEOM conformers reacted (TD's evaluator filters them). Scoring every method's generated files with one evaluator is the only safe comparison (C-BENCH-02).
- Supports card(s): C-BENCH-02
- Confidence: high (code); medium (size of the reference-set difference, not measured)
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-009 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-068
- Claim: MCF and S23D evaluated GEOM-QM9 on 995 test molecules (MCF's processed set), not 1000.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16, App. B.2.1
- Exact quote:
  > "The processed test dataset contained 995 molecules rather than the 1000 indexed in the test set"
- Shows vs infers: SHOWS
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-005 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-010 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-069
- Claim: In our TD-protocol QM9 runs, 65 of 1000 test molecules produce no TD conformers (60 strained cages ETKDG cannot embed, 5 rejected inputs), and the paper's own released QM9 samples scored with the repo evaluator give COV-R 88.8 % vs the published 92.8 %.
- Source type: our-data
- Locator:
  - our-data: RESULTS_QM9.md, §2.2 "Coverage is lower than the paper because of RDKit failures"
- Exact quote:
  > "**60 molecules:** RDKit's ETKDG cannot embed strained polycyclic cages" … "**So the paper's 92.8% cannot be reproduced from its own released samples.**"
- Shows vs infers: SHOWS
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-046 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-011 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-070
- Claim: TD's GEOM-QM9 table (TD-P, δ = 0.5 Å) reports COV-R / AMR-R means: RDKit 85.1 % / 0.235 Å, OMEGA 85.5 / 0.177, GeoMol 91.5 / 0.225, GeoDiff 76.5 / 0.297, Torsional diffusion 92.8 / 0.178 (AMR-P 0.221).
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 26, Table 7
- Exact quote:
  > "Table 7: Performance of various methods on the GEOM-QM9 dataset test-set ([δ] = 0.5Å)." … "RDKit 85.1 100.0 0.235 0.199" … "OMEGA 85.5 100.0 0.177 0.126" … "Torsional diffusion 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01, C-BENCH-02, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-012 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-071
- Claim: TD's GEOM-DRUGS table (TD-P, δ = 0.75 Å) reports COV-R / AMR-R / COV-P / AMR-P means: RDKit ETKDG 38.4 / 1.058 / 40.9 / 0.995, OMEGA 53.4 / 0.841 / 40.5 / 0.946, GeoMol 44.6 / 0.875 / 43.0 / 0.928, GeoDiff 42.1 / 0.835 / 24.9 / 1.136, Torsional Diffusion 72.7 / 0.582 / 55.2 / 0.778.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 8, Table 1
- Exact quote:
  > "RDKit ETKDG 38.4 28.6 1.058 1.002 40.9 30.8 0.995 0.895" … "OMEGA 53.4 54.6 0.841 0.762 40.5 33.3 0.946 0.854" … "Torsional Diffusion 72.7 80.0 0.582 0.565 55.2 56.9 0.778 0.729"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-02, C-BENCH-05, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-013 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-072
- Claim: ET-Flow's GEOM-DRUGS table (δ = 0.75 Å, 50 steps) reports COV-R / AMR-R / COV-P / AMR-P means: ET-Flow 79.53 / 0.452 / 74.38 / 0.541, ET-Flow-SS 79.62 / 0.439 / 75.19 / 0.517, ET-Flow-SO(3) 78.18 / 0.480 / 67.27 / 0.637; it copies MCF-S 79.4 / 0.512, MCF-B 84.0 / 0.427, MCF-L 84.7 / 0.390.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 6, Table 1 (row labels and numbers are on separate lines in the extraction; label order GeoDiff, GeoMol, Torsional Diff., MCF-S, MCF-B, MCF-L, ET-Flow, ET-Flow-SS, ET-Flow-SO(3) matched to the numeric rows in order; the first three numeric rows equal TD Table 1, E-BENCH-013, which confirms the mapping)
- Exact quote:
  > "Table 1: Molecule conformer generation results on GEOM-DRUGS ( [δ] = 0.75Å)." … "79.53 84.57 0.452 0.419 74.38 81.04 0.541 0.470" … "79.62 84.63 0.439 0.406 75.19 81.66 0.517 0.442" … "84.7 92.2 0.390 0.247 66.8 71.3 0.618 0.530"
- Shows vs infers: SHOWS (row mapping medium-high confidence; MCF rows also verified against MCF Table 2, E-BENCH-016)
- Supports card(s): C-BENCH-02, C-BENCH-05, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-014 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-073
- Claim: ET-Flow reports on GEOM-QM9 (δ = 0.5 Å) COV-R 96.47 %, AMR-R 0.073 Å, COV-P 94.05 %, AMR-P 0.098 Å; TD row 92.80 / 0.178.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 7, Table 2
- Exact quote:
  > "96.47 100.00 0.073 0.047 94.05 100.00 0.098 0.039"
- Shows vs infers: SHOWS
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-001 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-015 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-074
- Claim: On GEOM-QM9 the FM-refiner paper adds a δ = 0.05 Å coverage column: ET-Flow 75.72 % (AMR-R 0.083), DMT 72.90 % (0.087), MCF-B 66.82 % (0.101), best DMT-B + Refiner 79.50 % (0.070).
- Source type: paper
- Locator:
  - paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, Table 2
- Exact quote:
  > "MCF-B (62M) 66.82 67.86 0.101 0.050" … "ET-Flow (8.3M) 75.72 87.23 0.083 0.031" … "DMT-B + Refiner 79.50 89.44 0.070 0.026 80.37 97.92 0.076 0.021"
- Shows vs infers: SHOWS
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-008 (VERIFIED by V1); DMT row added from the same table.
- Supports card(s): C-BENCH-01, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-019 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-075
- Claim: ET-Flow also reports a GEOM-QM9 scaffold split (COV-R 95.00 %, AMR-R 0.083 Å), a DRUGS scaffold split (76.06 / 0.644) and a DRUGS-trained model tested on QM9 (86.68 / 0.218), as out-of-distribution evaluations beside GEOM-XL.
- Source type: paper
- Locator:
  - paper: papers/related/2024_hassan_etflow.pdf, PDF p. 19, App. D, Table 9 (labels and numbers on separate lines; label order QM9 RS, QM9 SS, DRUGS→QM9, DRUGS RS, DRUGS SS matched to rows in order; the QM9 RS and DRUGS RS rows equal Tables 2 and 1, which confirms the mapping)
- Exact quote:
  > "Table 9: Additional OOD results. We use RS and SS to indicate Random Split and Scaffold Split respectively." … "95.00 100.00 0.083 0.029 90.25 100.00 0.124 0.053" … "86.68 100.00 0.218 0.160"
- Shows vs infers: SHOWS (row mapping medium-high confidence)
- Supports card(s): C-BENCH-06
- Confidence: medium
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-022 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-076
- Claim: GeoMol reports confidence intervals from 3 repeated test-time runs of the same trained model (2K conformers per run): QM9 AMR-R 0.2254 ± 0.0012 Å, COV-R 91.52 ± 0.36 %.
- Source type: paper
- Locator:
  - paper: papers/related/2021_ganea_geomol.pdf, PDF p. 18, App. I (method) and PDF p. 19, Table 5 (values)
- Exact quote:
  > "The same trained model is run for 3 times for a given test molecule with K ground truth conformers, where one run means generating 2K conformers." … "QM9 91.52 100.00 0.2254 0.1938" … "± 0.36 ± 0.00 ± 0.0012 ± 0.0024"
- Shows vs infers: SHOWS (± glyph lost in extraction). INFERENCE: sampling-seed noise on QM9 AMR-R (≈ 0.001 Å) is the same order as our training-seed SD (CTRL median 0.0009 Å, round3/impact.md §2), so sampling seeds alone under-state the uncertainty of a training recipe.
- Supports card(s): C-BENCH-01
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-023 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-077
- Claim: The TD authors' shared Drive provides a released GEOM-DRUGS model (`workdir/drugs_default/best_model.pt`, 6.4 MB) with its hyper-parameters and the authors' own generated DRUGS test conformers (`drugs_steps20.pkl`, 457 MB), plus `drugs.tar.gz` (25.3 GB: data, splits, standardized pickles).
- Source type: blog (data repository listing)
- Locator:
  - blog: https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7, TD authors (Jing, Corso et al.), folder listing and HTTP Content-Range sizes, accessed 2026-10-11, snapshot papers/blogs/td_gdrive_listing.txt
- Exact quote (listing lines):
  > "drugs.tar.gz  id 1j_Pj6XZjJLf1tVQzjGWo2EFct2Nj_BvR  Content-Range: bytes 0-0/25308038362" … "best_model.pt       id 10tu38EDGe55M2v-0HtkMRYiPCcge4M5u  Content-Range: bytes 0-0/6392121" … "drugs_steps20.pkl   id 1HXY4FWt1CkQ28zPlnFzc6ZM_ZEnMRSoF  Content-Range: bytes 0-0/457404318"
- Shows vs infers: SHOWS (sizes; slurm/download_data.sh:11 already said 25.3 GB, now confirmed). INFERENCE: a DRUGS TD baseline, and any inference-only L-source change on top of TD (MMFF/xTB/template seeds), needs no DRUGS training (C-BENCH-05).
- Supports card(s): C-BENCH-02, C-BENCH-05
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-031 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-078
- Claim: TD's released DRUGS model uses the same architecture settings as our QM9 runs (4 conv layers, ns 32, nv 8, second-order irreps, radius 5 Å, batch 32, lr 1e-3, plateau patience 20, 250 epochs) but 74 node features and conformer-matched pickles from `drugs_standardized_multi_assignment_alignHs`.
- Source type: blog (released config file)
- Locator:
  - blog: same Drive, `workdir/drugs_default/model_parameters.yml`, accessed 2026-10-11, snapshot papers/blogs/td_gdrive_listing.txt (verbatim yml)
- Exact quote:
  > "batch_size: 32" … "in_node_features: 74" … "n_epochs: 250" … "num_conv_layers: 4" … "std_pickles: data/DRUGS/drugs_standardized_multi_assignment_alignHs/" … "use_second_order_repr: true"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-05
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-032 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-079
- Claim: TD's ensemble-property protocol (followed by ET-Flow) uses a random 100-molecule GEOM-DRUGS subset, min(2K, 32) conformers per molecule, GFN2-xTB relaxation, then Boltzmann-weighted E, μ, Δε, E_min compared with the CREST ensemble; TD's unrelaxed E_min error is 36.94 kcal/mol and 0.13 after relaxation.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 9, Sec. 4.4; PDF p. 27, Table 10
- Exact quote:
  > "For a random 100-molecule subset of DRUGS, we generate min(2K, 32) conformers per molecule, relax the conformers with GFN2-xTB" … "Tor. Diff. 36.91 0.92 4.93 36.94 0.22 0.35 0.54 0.13"
- Shows vs infers: SHOWS
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-024 and E-EVAL-022 (VERIFIED by V1); ET-Flow uses the same protocol, E-EVAL-054 (VERIFIED).
- Supports card(s): C-BENCH-03, C-BENCH-05, C-BENCH-07
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-041 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-080
- Claim: TD reports runtime as CPU core-seconds per conformer (Intel i9-9920X, 8 threads, 10 random DRUGS test molecules × 8 conformers, excluding model loading): RDKit 0.10, TD 20 steps 4.90, TD 5 steps 1.76, GeoDiff 305.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 25, App. G.3 "Runtime evaluation"; PDF p. 9, Table 2
- Exact quote:
  > "We benchmark the methods on CPU (Intel i9-9920X) to enable comparison with RDKit." … "We select 10 molecules at random from the GEOM-DRUGS test set and generate 8 conformers per molecule using each method." … "RDKit - 1.002 0.895 0.10"
- Shows vs infers: SHOWS. INFERENCE: any added step (xTB seed relaxation, ring sampler, refiner) must be costed in the same unit; xTB at ≈ 0.14 s per QM9 seed (round3/grounding.md §2.2) is comparable to TD's per-conformer cost, so it is not free on DRUGS, whose molecules average 44.4 atoms (E-BENCH-055) against at most 29 for QM9.
- Supports card(s): C-BENCH-03
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-043 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-081
- Claim: Two published arguments make QM9 at δ = 0.5 Å a weak headline: S23D calls the benchmark "likely saturated" (mean AMR < 0.1 Å), and the FM-refiner paper switches to δ = 0.05 Å because median COV is already 100 % at 0.5 Å.
- Source type: paper
- Locator:
  - paper: papers/related/2025_gurev_s23d.pdf, PDF p. 16, App. B.2.1; papers/related/2025_xu_fm_refiner.pdf, PDF p. 7, Table 2 caption
- Exact quote:
  > "benchmark results on the QM9 metric are likely to be saturated." … "recent work already achieves 100% median COV at the commonly used threshold [δ] = 0.5 Å" … "we adopt the more challenging COV threshold of [δ] = 0.05 Å."
- Shows vs infers: SHOWS
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-006 and E-EVAL-007 (VERIFIED).
- Supports card(s): C-BENCH-01, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-047 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-082
- Claim: Zhou et al.'s cheap "RDKit + Clustering" baseline (random-dihedral, ETKDG and ETKDG+MMFF samplers 1:1:4, N_e = min(20 N_ref, 2000) samples, K-means to 2 N_ref outputs) reaches COV 97.65 % / MAT 0.1902 Å on the GD-protocol QM9 set, against plain RDKit 83.26 % / 0.3447 Å; recall metrics reward oversampling.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhou_dl_conformation_critique.pdf, PDF p. 4, Sec. 3 (method) and PDF p. 5, Table 1
- Exact quote:
  > "Ne = min(20Nref , 2000)" … "we employ the K-means algorithm with 2Nref clusters" … "RDKit + Clustering 97.65 100.00 0.1902 0.1818"
- Shows vs infers: SHOWS (GD protocol, 200 molecules; not comparable to TD-P numbers).
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-025 and E-EVAL-026 (VERIFIED by V1).
- Supports card(s): C-BENCH-02, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-049 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-083
- Claim: Zhang et al. reply that the RDKit + Clustering comparison is unfair unless deep models are given the same ~2000-sample-then-cluster budget.
- Source type: paper
- Locator:
  - paper: papers/related/2023_zhang_infinite_physical_monkey.pdf, PDF p. 5
- Exact quote:
  > "At the very least, they should also sample 2000 conformations for DL methods followed by clustering."
- Shows vs infers: SHOWS. INFERENCE: whichever cheap baseline we include, the sample budget must be equal (2K per molecule) or reported per row.
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-028 (VERIFIED by V1).
- Supports card(s): C-BENCH-02
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-050 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-084
- Claim: TD's GEOM-DRUGS models were trained for 250 epochs, taking 4 to 11 days on a single NVIDIA RTX A6000; the QM9 model used the same hyper-parameters.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 24, App. G.2 "Training and tuning details"
- Exact quote:
  > "the torsional diffusion models were trained on NVIDIA RTX A6000 GPUs for 250 epochs with the Adam optimizer (taking from 4 to 11 days on a single GPU)." … "for GEOM-QM9 a new model with the same hyperparameters was trained."
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-05
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-052 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-085
- Claim: GEOM's drug-like molecules average 44.4 atoms (24.9 heavy) and 6.5 rotatable bonds (max 181 atoms, 53 rotatable bonds); QM9 molecules have at most 9 heavy (29 total) atoms.
- Source type: paper
- Locator:
  - paper: papers/related/2022_axelrod_geom.pdf, PDF p. 2, "Background & Summary" (Table 1 discussion)
- Exact quote:
  > "containing an average of 44.4 atoms (24.9 heavy atoms), up to a maximum of 181 atoms (91 heavy atoms)." … "mean (6.5) and maximum (53) number of rotatable bonds." … "The QM9 dataset is limited to 9 heavy atoms (29 total atoms)"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-05
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-055 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-086
- Claim: The only published "other-conformer local structure" test we found is TD App. F.1: matching a GT conformer with the local structure of another randomly chosen GT conformer of the same molecule gives a mean RMSD_min of 0.284 Å on GEOM-DRUGS, against 0.324 Å with matched RDKit local structures; it is a model-free matched-RMSD floor, not a generation test.
- Source type: paper
- Locator:
  - paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 21, App. F.1 "Matched conformer RMSD"
- Exact quote:
  > "it is only slightly larger than the average RMSDmin of 0.284 Å resulting from matching a ground truth conformer to the local structure of another randomly chosen ground truth conformer"
- Shows vs infers: SHOWS. INFERENCE: TD's authors already treated "another conformer's L" as the honest reference-level L; our split-half test is the generative version of the same idea and can cite this as precedent. No paper among those read (TD, ET-Flow, MCF, NExT-Mol/DMT, GeoMol, S23D, FM-refiner, AvgFlow, EnFlow, GEOM, GEOM-revisited, Zhou, Ezzine, GO-Flow, PuckerFlow) defines a split-half or leak-free true-geometry generation protocol (search record in section Z).
- Supports card(s): C-BENCH-04
- Confidence: high (quote); medium (negative search)
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-061 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-087
- Claim: A split-half leak-free oracle is already implemented (uncommitted, round 4): GT conformers are cleaned exactly as the evaluator does, split alternately by cleaned index into halves A/B, seeds come from one half and recall is scored on the other; molecules with one clean conformer are excluded and counted; a model-free "copy the other half" baseline is included.
- Source type: code
- Locator:
  - code: tools/make_split_half.py:1-31 (docstring), :57; tools/analyze_leakfree.py:1-40 (pre-declared design); slurm/r4_leakfree_array.sbatch (untracked, checked 2026-10-12)
- Exact quote:
  > "split ALTERNATING BY CLEANED GT INDEX" … "Molecules with n_clean = 1 (no other conformer exists) and n_clean = 0 are excluded and counted." … "split_AtoB  seeds A, 2|B| confs, recall vs B"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-04
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-063 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-088
- Claim: Our round-2 analysis already implements the statistics reviewers ask for: pre-declared contrasts, TD-protocol union universe with model failure = COV 0, paired molecule bootstrap with Holm over the primary family, the success intersection for COV-R@0.1, training-seed SD on every row, and a hierarchical (seeds then molecules) bootstrap for non-inferiority.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/README.md (lines 6-14)
- Exact quote:
  > "TD protocol (union of molecules, model failure = COV 0), paired molecule bootstrap, Holm over the PRIMARY family" … "training-seed SD of the arm and ref aggregates next to the bootstrap CI"
- Shows vs infers: SHOWS
- Supports card(s): C-BENCH-01
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-066 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-089
- Claim: Final round-2 result: S3 on RDKit L is worse than CTRL_rematch by +0.0032 Å AMR-R [+0.0013, +0.0052], Holm p = 0.0096, n = 934 (training-seed SDs 0.0004 for S3 with 2 seeds, 0.0013 for CTRL with 3 seeds).
- Source type: our-data
- Locator:
  - our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv:50 (`S3.rdkit_vs_CR`, MAT-R, primary)
- Exact quote:
  > "S3.rdkit_vs_CR,primary,union,MAT-R,primary,934,S3,steps20_seed0,CR,steps20_seed0,,0.1818677432916211,0.1786359860697768,0.0032317572218443077,0.0012702605359765939,0.005228308316968585,0.0012,0.00038785291373951076,2,0.0013366058280782855,3,,ok,,,0.0096,"
- Shows vs infers: SHOWS. INFERENCE: differences of 0.003 Å are detectable with this design (CI half-width ≈ 0.002 Å), so the evaluation is powered for the effects we care about on QM9; the same design should be the paper's.
- Supports card(s): C-BENCH-01, C-BENCH-06
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-067 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-090
- Claim: Released TD (QM9) with RDKit L, sampling seed 0, scores AMR-R 0.1752 Å on 935 evaluated molecules (65 failures), COV-R@0.05 3.91 %; RDKit ETKDG alone 0.2298 Å; MMFF-relaxed seeds (A2) 0.1507 Å with COV-R@0.05 17.89 %.
- Source type: our-data
- Locator:
  - our-data: cluster_sync/results/analysis/all_summaries.txt, lines `qm9_default/R0_base_seed0`, `qm9_default/A0_rdkit_etkdg_only`, `qm9_default/A2_pre_mmff`
- Exact quote:
  > "qm9_default/R0_base_seed0 threshold=0.5000 COV-R_mean=89.0293 COV-R_median=100.0000 MAT-R_mean=0.1752" … "qm9_default/A0_rdkit_etkdg_only threshold=0.5000 COV-R_mean=83.9217 COV-R_median=100.0000 MAT-R_mean=0.2298" … "qm9_default/A2_pre_mmff threshold=0.5000 COV-R_mean=88.9339 COV-R_median=100.0000 MAT-R_mean=0.1507"
- Shows vs infers: SHOWS. INFERENCE: our re-scored TD COV-R (89.0 %) is 3.8 points below TD's published 92.8 % because of failure accounting (E-BENCH-011); any table that mixes our numbers with published ones must say so.
- Reused from (earlier): round3/ledger/evidence_EVAL.md E-EVAL-038, E-EVAL-039, E-EVAL-040 (VERIFIED by V1).
- Supports card(s): C-BENCH-01, C-BENCH-02
- Confidence: high
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-069 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)


### E-SAMPLE-091
- Claim: Measured GFN2-xTB cost on gnode118 (xtb 6.7.1, single thread, `--opt normal`): median 0.135 s, mean 0.157 s per QM9 optimisation; GT conformer 0 of each of 10 test molecules converged in 1 iteration with 0.000 Å displacement, while ETKDG seeds needed 17–63 iterations and moved 0.09–0.32 Å.
- Source type: our-data
- Locator:
  - our-data: round3/grounding.md:66-73 (§2.2, job 13057)
- Exact quote:
  > "GT conformer 0 of each molecule converged after 1 iteration with 0.000 Å heavy-atom displacement." … "ETKDG seeds needed 17–63 iterations and moved by 0.09–0.32 Å heavy-atom RMSD." … "Median 0.135 s, mean 0.157 s, max 0.383 s per optimisation."
- Shows vs infers: SHOWS (n = 20 calls on 10 molecules)
- Supports card(s): C-BENCH-03, C-BENCH-05, C-BENCH-06
- Confidence: high (measurement); medium (small sample)
- Reused from: round4/ledger/evidence_BENCH.md E-BENCH-072 (not yet verified by a round-4 verifier; quote re-checked by R-SAMPLE 2026-10-15: OK)

## Z. Search negatives
(Recorded so verifiers can re-run them; not evidence entries. Web searches 2026-10-15.)
- "adaptive sampling budget allocation per input diffusion model inference compute uncertainty" → AdaDiff, GAINS
  (2609.22867), ABCD (2505.14036, adaptive inference-time search; snippet only, not downloaded). None allocates the
  number of samples across inputs for a set-level (coverage) objective.
- "molecular conformer generation adaptive number of conformers per molecule stopping criterion convergence diffusion"
  → no diffusion/ML conformer paper with per-molecule adaptive sample counts. Hits: TD (steps ablation), Adjoint
  Sampling (conformer count grows with rotatable bonds, `2025_havens_adjoint_sampling` already in the store by another
  agent; not used here), e3fp/RDKit prune-after-minimise wrappers, a USPTO patent with "no new elite conformation for
  N iterations" stopping (not downloaded; patent, not peer reviewed).
- "conformer Good-Turing / unseen species / capture-recapture" → Koukos & Glykos 2013 (MD; downloaded), Thomson
  problem (2015, not relevant), Seep et al. 2021 macrocycle ensemble completeness (PMC8320181; completeness by PCA maps,
  no estimator; not downloaded).
- "uncertainty-aware OR adaptive conformer generation diffusion number of samples per molecule 2025 2026" → only
  Seij et al. 2026 (de novo, downloaded), DeeDiff (early exit), EC-Conf (fewer steps). No conformer paper allocates
  samples per molecule.
- CREST PCCP 2020 (doi 10.1039/C9CP06869D): PDF download returned an HTML bot page; the CREST documentation page was
  used instead (E-SAMPLE-042). OMEGA paper (Hawkins et al. 2010, JCIM) is paywalled; the vendor documentation was used
  (E-SAMPLE-040/041). Ebejer et al. 2012 (rotatable-bond-based conformer counts for RDKit) not obtained (paywalled).
- TD has no "coverage vs number of samples" experiment (searched every TD PDF page for "number of samples", "2K",
  "samples per"); Table 9 varies steps only. PG Figure 5 is a toy mode-count curve, not a conformer measurement.
