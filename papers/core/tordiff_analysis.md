# Torsional Diffusion for Molecular Conformer Generation — In-depth analysis

- **Citation:** Bowen Jing*, Gabriele Corso*, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola. *Torsional Diffusion for Molecular Conformer Generation.* NeurIPS 2022. arXiv:2206.01729 (v2, 1 Mar 2023).
- **PDF (local):** `papers/core/2022_jing_torsional_diffusion.pdf` (arXiv v2; contains the full supplementary material as Appendices A–H, pp. 14–28 — this is the same appendix as the NeurIPS supplement).
- **Full text (local, for grep/RAG):** `papers/core/2022_jing_torsional_diffusion.fulltext.txt`
- **URL:** https://arxiv.org/abs/2206.01729 · code: https://github.com/gcorso/torsional-diffusion (local clone at `torsional-diffusion/`)
- Page numbers below are the printed page numbers of the arXiv v2 PDF.

---

## 1. Problem decomposition (Sec. 3, p. 4)

A conformer C is split into intrinsic coordinates **(L, τ)**:
- **L — local structure:** bond lengths, bond angles, **cycle (ring) conformations**, chirality (App. A, p. 14; F.3, p. 22).
- **τ — torsions:** dihedral angles of *freely rotatable* bonds. A bond is rotatable if cutting it yields two connected components each with ≥ 2 atoms (p. 4). **Ring torsions are part of L**, and **double bonds count as rotatable** (footnote 6, p. 4; F.3).
- Model factorisation: p_G(L, τ) ≈ p̂_G(L)·p_θ,G(τ | L), where p̂_G(L) is **RDKit ETKDG** (not learned), and only p(τ | L) is learned with diffusion (p. 4). Justification: "the set of possible stable local structures L ... is very constrained and can be accurately predicted by ... RDKit ETKDG (see Appendix F.1)" (p. 4).
- Dimensionality: DRUGS molecules average n = 44 atoms (132-D Cartesian) but m = 7.9 rotatable torsions (p. 1).

## 2. Math

### 2.1 Diffusion on the hypertorus T^m (Sec. 3.1, pp. 4–5)
- Framework: Riemannian score-based generative modelling (De Bortoli et al. 2022) on compact manifold T^m = R^m / 2πZ^m.
- Forward SDE: rescaled Brownian motion, f(x,t)=0, g(t)=sqrt(d σ²(t)/dt), with exponential (VE) schedule **σ(t) = σ_min^{1−t} σ_max^{t}, σ_min = 0.01, σ_max = π, t∈(0,1)** (p. 4).
- Prior p_T: **uniform on T^m** (compactness; not Gaussian) (p. 4).
- **Perturbation kernel = wrapped normal** (Eq. 3, p. 4):
  p_{t|0}(τ′ | τ) ∝ Σ_{d∈Z^m} exp( −‖τ − τ′ + 2πd‖² / (2σ²(t)) ).
  Sampling: draw unwrapped isotropic normal, take elementwise mod 2π. The kernel score ∇_τ′ log p_{t|0} is **precomputed numerically** (truncated sum over d) and tabulated.
- **DSM loss** (Eq. 4, p. 4): J_DSM(θ) = E_t[ λ(t) E_{τ0∼p0, τt∼p_{t|0}} ‖ s(τt, t) − ∇ log p_{t|0}(τt | τ0) ‖² ], with λ(t) = 1 / E‖∇ log p_{t|0}‖² (precomputed). Tangent space of T^m is R^m, so loss is standard.
- Reverse process: geodesic random walk; on the torus exp_τ(δ) = τ + δ mod 2π, so it is just a wrapped Euler–Maruyama step (p. 5). Algorithm 3 (p. 17): Δτ = (g²(t)/N)·s + g(t)·z, z ~ N(0, 1/N), g(t) = σ_min^{1−t}σ_max^{t} sqrt(2 ln(σ_max/σ_min)). **20 steps** by default (Table 9, p. 27).

### 2.2 Extrinsic-to-intrinsic score (Sec. 3.2, p. 5; App. B.1, F.2)
- The score model s_G(C, t): R^{3n} × [0,T] → R^m takes the **3D point cloud** (extrinsic) and outputs one scalar per rotatable bond (intrinsic). Avoids choosing reference neighbours for dihedral definitions.
- **Proposition 1** (p. 5, proof pp. 14–15): rotating the c-side substructure about the bond axis r̂_bc by Δτ changes *any* definition of τ_i by exactly Δτ and leaves all other τ_j unchanged. So torsion updates are applied directly as relative rotations; updates applied sequentially in any order. Implementation rotates the smaller side (F.2, p. 21).
- **Parity (Sec. 3.3, Prop. 2, p. 6):** since p(τ(C)|L(C)) = p(τ(−C)|L(−C)), the score must satisfy s(−C) = −s(C): SE(3)-invariant **pseudoscalars**. Ablation: parity-invariant model fails completely (≈ random baseline; Table 8, p. 27).
- **Architecture (Sec. 3.4; App. D, pp. 17–19):** e3nn tensor-field network; embedding (radius graph r_max = 5 Å + chemical bonds, 74-D atom features for DRUGS / 44-D for QM9, 4-D bond one-hot), 4 interaction layers (irreps up to l = 2), final **pseudotorque layer**: filter = tensor product of spherical harmonics of neighbour direction with l = 2 representation of the bond axis (sign-free), outputs pseudoscalars via odd (tanh, no bias) dense layers.

### 2.3 Likelihood (Sec. 3.5, Prop. 3, pp. 6–7; proof App. B.3, p. 16)
- Probability-flow ODE, Eq. 7: log p0(τ0) = log pT(τT) − ½ ∫_0^T g²(t) ∇_τ·s_G(τt, t) dt; divergence computed **exactly** (feasible because m is small), not by Hutchinson, to get an unbiased estimate of p (not just log p).
- Conversion torsional → Euclidean density (Eq. 8): p_G(x | L) = p_G(τ | L) / (8π² sqrt(det g)), g_αβ = Σ_k J_α^(k)·J_β^(k) over m torsion + 3 rotation coordinates; J^(k)_i = displacement of atom k per unit torsion (cross product with bond axis, zero on the fixed side, centred), 8π² = volume of SO(3).
- Used for **torsional Boltzmann generator** (Sec. 3.6, Alg. 1, p. 7): importance-reweighted DSM using MMFF energies; ESS results Table 4 (p. 10). Only conditional p(C | L) — full p(C) would need a local-structure model with exact likelihoods (F.4, p. 23).

### 2.4 Conformer matching (Sec. 4.1, p. 7; App. E, pp. 19–20)
- Motivation (verbatim sense, p. 7): if trained on GT conformers (i.e. conditioned on GT local structures) there is **distribution shift at test time where only RDKit local structures are available**; "this shift significantly hurts performance."
- Procedure (Alg. 4): for a molecule with K GT conformers, generate K RDKit local structures L̂; cost matrix K×K = RMSD after fast von Mises torsion matching (Stärk et al. 2022, EquiBind); **linear sum assignment**; then per pair **differential evolution** over torsions to get Ĉ_i = (L̂_j, τ̂) minimising RMSD(C_i, Ĉ_i). Only the **training split** is matched.
- Table 5 (p. 20, 300 DRUGS mols): mean RMSD(C, Ĉ): original RDKit 1.448 Å; von Mises 0.728; differential evolution (random pairing) 0.379; full conformer matching **0.324 Å**. → TD calls 0.324 Å "an approximate lower bound on the achievable AMR" for methods that keep RDKit local structures (p. 19; F.1; F.4). **Caveat (review/cross_validation.md M1):** it is a one-to-one matching number (each GT conformer gets one RDKit seed out of L), while AMR-R takes the best of 2L generated conformers with no one-to-one constraint. So it is upper-biased, not a true lower bound on AMR-R; use the run-matched `floor_run` / `floor_best_sym` (`tools/local_structure_analysis.py`) instead. Matching a GT conformer to *another GT conformer's* local structure gives **0.284 Å** (F.1, p. 21); TD reads this as the GT ensemble's local-structure variability. For its G3 reading, see §6.
- For QM9 the analogous conformer-matching estimate is ≈ **0.17 Å** (App. H, p. 26: "approximately calculated by conformer matching"). The same M1 caveat applies: it is not a lower bound on AMR-R.

## 3. Evaluation protocol (Sec. 4.2, p. 8; App. G, pp. 23–25; code `torsional-diffusion/evaluate_confs.py`)

| Item | Value (source) |
|---|---|
| Datasets | GEOM-DRUGS (304k mols, avg 44 atoms); GEOM-QM9 (avg 11 atoms); **GEOM-XL** = all MoleculeNet species in GEOM with > 100 atoms (G.1 says "at least 100"), **102 molecules**, avg 136 atoms, 32 rotatable bonds, **test-only, evaluated with the DRUGS-trained model** (p. 8; G.1 p. 23; H p. 25) |
| Splits | From GeoMol (Ganea et al. 2021), random: DRUGS **243473 / 30433 / 1000**; QM9 **106586 / 13323 / 1000** (G.1, p. 23). Molecules whose CREST conformers all have non-matching canonical SMILES (reacted) or RDKit failures are filtered. |
| Training conformers | at most first **30** CREST conformers per molecule (App. C, p. 17) |
| # generated | **2K** conformers for a molecule with K ground-truth (CREST) conformers (p. 8; App. C; G.3). Code: `sample_confs(raw_smi, 2 * n_confs, smi)`. |
| RMSD | heavy-atom RMSD (`Chem.RemoveHs`) with symmetry-aware `AllChem.GetBestRMS`; for **XL, `AlignMol` without symmetry permutations → upper bound** (G.3, p. 24; `--only_alignmol`) |
| Thresholds δ | **DRUGS δ = 0.75 Å** (explicitly "different from most prior works, which used δ = 1.25 Å", Table 1 caption, p. 8); **QM9 δ = 0.5 Å** (Table 7, p. 26); **XL: no coverage reported, only AMR** (Table 6). Code evaluates a sweep `np.arange(0, 2.5, .125)`. |
| Metrics | COV-R, AMR-R, COV-P, AMR-P, mean and median over test molecules (Eq. 35, p. 24) |
| Failures | code: molecules with no model output count as coverage 0; AMR averaged with `nanmean` (failures excluded) |
| Properties | 100-mol DRUGS subset, min(2K, 32) confs, GFN2-xTB relaxed, Boltzmann-weighted E, μ, Δε, E_min median abs error (Sec. 4.4, p. 9; Table 10, p. 27) |
| Runtime | CPU i9-9920X, 8 threads, 10 mols × 8 confs (G.3, p. 25) |

Metric definitions (Eq. 35, p. 24), with L ground-truth {C*_l} and K = 2L generated {C_k}:
- COV-R = (1/L) |{ l : ∃k, RMSD(C_k, C*_l) < δ }|
- AMR-R (= MAT-R) = (1/L) Σ_l min_k RMSD(C_k, C*_l)
- Precision versions: swap roles (COV-P = (1/K)|{k : ∃l, RMSD < δ}|, AMR-P = (1/K) Σ_k min_l RMSD).
- Code uses strict `<` δ.

## 4. Reported numbers

### GEOM-DRUGS (Table 1, p. 8; δ = 0.75 Å)
| Method | COV-R mean | COV-R med | AMR-R mean | AMR-R med | COV-P mean | COV-P med | AMR-P mean | AMR-P med |
|---|---|---|---|---|---|---|---|---|
| RDKit ETKDG | 38.4 | 28.6 | 1.058 | 1.002 | 40.9 | 30.8 | 0.995 | 0.895 |
| OMEGA | 53.4 | 54.6 | 0.841 | 0.762 | 40.5 | 33.3 | 0.946 | 0.854 |
| GeoMol | 44.6 | 41.4 | 0.875 | 0.834 | 43.0 | 36.4 | 0.928 | 0.841 |
| GeoDiff | 42.1 | 37.8 | 0.835 | 0.809 | 24.9 | 14.5 | 1.136 | 1.090 |
| **Torsional Diffusion** | **72.7** | **80.0** | **0.582** | **0.565** | **55.2** | **56.9** | **0.778** | **0.729** |

### GEOM-QM9 (Table 7, p. 26; δ = 0.5 Å)
| Method | COV-R mean | COV-R med | AMR-R mean | AMR-R med | COV-P mean | COV-P med | AMR-P mean | AMR-P med |
|---|---|---|---|---|---|---|---|---|
| RDKit | 85.1 | 100.0 | 0.235 | 0.199 | 86.8 | 100.0 | 0.232 | 0.205 |
| OMEGA | 85.5 | 100.0 | 0.177 | 0.126 | 82.9 | 100.0 | 0.224 | 0.186 |
| GeoMol | 91.5 | 100.0 | 0.225 | 0.193 | 86.7 | 100.0 | 0.270 | 0.241 |
| GeoDiff | 76.5 | 100.0 | 0.297 | 0.229 | 50.0 | 33.5 | 0.524 | 0.510 |
| **Torsional Diffusion** | **92.8** | 100.0 | 0.178 | 0.147 | **92.7** | 100.0 | 0.221 | 0.195 |

Note: TD (p. 26) says its QM9 AMR-R (0.178) is "already very close [to the] lower bound of 0.17 Å". Per review M1 that comparison is not valid: 0.17 Å is a one-to-one matching estimate, not a lower bound on best-of-2L AMR-R. **OMEGA has better AMR-R** (0.177 mean, 0.126 median) "which, evidently, has a better local structures for these small molecules" (p. 26).

### GEOM-XL (Table 6, p. 26; AMR only, DRUGS-trained model, RMSD without symmetry = upper bound)
(The text extraction misaligns the row labels; the values below are reconciled with the text "reduces RDKit AMR by 30% on recall and 12% on precision", p. 25.)
| Model | AMR-R mean | AMR-R med | AMR-P mean | AMR-P med |
|---|---|---|---|---|
| RDKit | 2.92 | 2.62 | 3.35 | 3.15 |
| GeoMol | 2.47 | 2.39 | 3.30 | 3.15 |
| Torsional Diffusion | 2.05 | 1.86 | 2.94 | 2.78 |

### Steps / runtime (Table 2 p. 9; Table 9 p. 27)
Median AMR-R/AMR-P/runtime(core-s per conformer): RDKit 1.002/0.895/0.10; GeoMol 0.834/0.841/0.18; GeoDiff (5000 steps) 0.809/1.090/305; TD 5 steps 0.685/0.963/1.76; 10 steps 0.580/0.791/2.82; 20 steps 0.565/0.729/4.90. 50 steps: COV-R 73.1, AMR-R 0.578, COV-P 57.6, AMR-P 0.753 (means).

### Ablations on DRUGS (Table 8, p. 27; means COV-R / AMR-R / COV-P / AMR-P)
- Baseline 72.7 / 0.582 / 55.2 / 0.778
- Probability-flow ODE 73.1 / 0.577 / 55.3 / 0.779
- **Only D.E. matching (random L̂–C pairing, no assignment)** 72.5 / 0.588 / 53.8 / 0.794
- First-order irreps 70.1 / 0.605 / 51.4 / 0.817
- **Train on ground-truth L** (test on RDKit L) 34.8 / 0.920 / 22.3 / 1.182
- No parity equivariance 30.5 / 0.928 / 17.9 / 1.234
- Random torsions on RDKit L 30.9 / 0.922 / 18.2 / 1.228

### Properties (Table 3 p. 9 / Table 10 p. 27, median abs error after xTB relaxation; E, Δε, E_min kcal/mol; μ debye)
TD: E 0.22, μ 0.35, Δε 0.54, E_min 0.13 (best). **Without relaxation all methods have ~16–43 kcal/mol energy errors** (TD 36.91) — "relaxation of local structures is necessary for any method" (p. 28).

### Boltzmann generator (Table 4, p. 10): ESS out of 32 at 1000/500/300 K — TorsionalBG 20 steps 11.42/6.42/4.68 vs AIS-100 6.72/3.12/2.06.

## 5. Every limitation stated or implied (with citations)

Stated:
1. **Error lower-bounded by local-structure quality**: ≈ 0.324 Å on DRUGS with RDKit L (F.4, p. 22; E, p. 19; F.1, p. 21); ≈ 0.17 Å on QM9 (H, p. 26). Both are the paper's one-to-one conformer-matching estimates and are upper-biased relative to AMR-R (review M1).
2. **Conformer-matched training targets are not energy minima** (neither of unconditional nor conditional PES) → less physically interpretable, "potentially more difficult" learning, visible in train/val losses (F.4, p. 22).
3. **Rigid local structure**; authors explicitly propose as future work "relax the rigid local structure assumption by developing an efficient diffusion-based model over the full space of intrinsic coordinates while still incorporating chemical constraints" (Conclusion, p. 10) and relaxations "allowing some flexibility in the independent components" (F.4, p. 22).
4. **Rings**: ring conformations are the largest source of flexibility not modelled; OK for small/aromatic rings but "less true for puckered rings, fused rings, and larger cycles"; "does not address the longstanding difficulty ... with macrocycles (≥12 atoms)" (F.4, p. 23).
5. **Boltzmann generator only conditional on L**; full p(C) needs a local-structure model with exact likelihood (F.4, p. 23).
6. **Proteins/macromolecules**: torsion changes cause large distant displacements; not promising directly (F.4, p. 23).
7. **E/Z isomerism not captured** (double bonds treated as freely rotatable) (F.3, p. 22).
8. **QM9**: little benefit over cheminformatics; on par/slightly worse than OMEGA due to local structure (H, p. 26).
9. **XL**: out-of-distribution (trained on DRUGS), "can very likely be improved by training and tuning ... on larger molecules" (H, p. 25); XL RMSDs are symmetry-unaware upper bounds (G.3, p. 24).
10. **Unrelaxed conformers are energetically poor** (Table 10, p. 27): 36.9 kcal/mol median energy error without relaxation.
11. Slower than single-shot GeoMol (Sec. 4.3, p. 9); ~4.9 core-s/conformer at 20 steps.
12. Changed the coverage threshold from 1.25 Å to 0.75 Å for DRUGS (Table 1, p. 8) — comparability caveat with older papers.

Implied (not stated by the authors):
- L is sampled from RDKit **independently of τ** (Alg. 3, p. 17: local structures first, then torsions randomised); the model learns p(τ | L) but there is **no p(L | τ)** — bond angles cannot relax in response to the chosen torsions (steric clashes, e.g. gauche/1,5 interactions).
- Sequential torsion rotations with fixed L can produce clashes that can only be avoided by the torsional score (no local "give").
- The "Only D.E. matching" ablation (random L̂–C pairing ≈ same accuracy) suggests the learned model extracts little signal from the L–τ dependence in DRUGS under the RDKit-L paradigm.
- Uniform torus prior and fixed rotatable-bond definition: non-rotatable/ring/amide coupling is purely in L.

## 6. Verification of the professor's three gaps

**Gap 1 — Rigid local structure; fails when RDKit local geometry is poor (macrocycles, strained systems, GEOM-XL).**
- **Supported**: the paper's conformer-matching floor estimates (0.324 Å DRUGS, ~0.17 Å QM9; upper-biased, not lower bounds on AMR-R, per review M1), explicit statement that macrocycles/puckered/fused rings are not addressed (F.4, pp. 22–23), and conclusion's future-work proposal to relax local structure (p. 10). The QM9 result (Table 7) is the *clearest in-paper evidence*: TD sits at the RDKit-L lower bound and OMEGA's better local structures give better AMR-R.
- **Not supported as stated for GEOM-XL**: the paper never attributes XL error to poor local geometry. It attributes XL difficulty to size/flexibility (32 rotatable bonds) and out-of-distribution training, and TD improves XL AMR-R by 30% over RDKit (Table 6, p. 25–26). XL RMSDs are symmetry-unaware upper bounds. Whether local-structure error is a dominant XL error term is an **open, testable hypothesis** (e.g., compute conformer-matching RMSD lower bound on XL). "Strained systems" are not discussed at all in the paper.

**Gap 2 — Distribution shift: trained with GT local structures but tested with RDKit ones.**
- **Inaccurate as phrased.** The paper identifies exactly this shift (Sec. 4.1, p. 7) and **does not train on GT local structures**: conformer matching replaces every training conformer with an RDKit-local-structure conformer (App. E), which "guarantees that there is no distributional shift in the local structures seen during training and inference" (p. 19). Training on GT L is only an ablation, and it collapses (COV-R 72.7 → 34.8, Table 8).
- **What remains true / the real residual gap**: (a) conformer matching trades the shift for **biased, non-physical training targets** (F.4, p. 22) and an irreducible RMSD floor; (b) RDKit L is sampled independently of the target torsions, so the "shift" is removed only in the marginal over L, not in the physical joint; (c) test-time L comes from ETKDG's own biases, so any improvement in L at test time (e.g., a learned L-model) would re-introduce a shift unless training also uses that L-model. Recommended re-framing: "the conformer-matching workaround caps accuracy and trains on non-minima; a model that learns (L, τ) jointly would remove the need for it."

**Gap 3 — Ignores torsion–bond-angle coupling (sterics).**
- **Partially supported / implied, not stated.** The score network sees the full 3D structure and therefore models p(τ | L) — torsions *are* conditioned on local structure. What is missing is the reverse coupling p(L | τ): L is drawn by RDKit before and independently of τ and never updated. The paper does not discuss sterically induced bond-angle relaxation; instead it argues bond angles have little flexibility (RMSE 4.1°, narrow unimodal error; F.1, Fig. 5, p. 20) and that GT local-structure variability is only 0.284 Å (p. 21). **Read the other way, F.1 (p. 21) is the strongest in-paper evidence for G3:** another GT conformer's local structure still leaves an average RMSD_min of 0.284 Å on DRUGS, against 0.324 Å with RDKit local structures. Even a perfect independent sampler of GT local structures removes only about 0.04 Å of the floor; the rest needs local structure matched to the torsions (L|τ coupling). Both numbers carry the one-to-one-matching caveat of review M1. The random-pairing ablation (Table 8) still DE-optimises each pair, so it tests the assignment, not coupling. The energy-error tables (unrelaxed E error 36.9 kcal/mol, Table 10) indirectly support that the frozen-L geometries are strained. Framing should be: "one-directional coupling only; local structure cannot respond to torsional sterics."
