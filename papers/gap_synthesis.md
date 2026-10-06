# Gap synthesis and FlexiTors-Diffusion design

Sources: `core/tordiff_analysis.md` plus the notes in `related/`. Page and section citations refer to those notes and PDFs. Protocol codes: **TD-P** is the GeoMol/TD split with δ = 0.75 Å (DRUGS) and 0.5 Å (QM9); **GD-P** is the ConfGF/GeoDiff split with δ = 1.25 Å (DRUGS) and 200 test molecules.

---

## 0. Corrections to the framing (read first)

| Professor's gap | What the TD paper actually says | Suggested re-framing |
|---|---|---|
| G1: rigid local structure fails for macrocycles, strained systems and GEOM-XL | **Supported** for rings and macrocycles (TD App. F.4, pp. 22–23) and by the explicit floor: the conformer-matching RMSD with RDKit L is 0.324 Å on DRUGS (Table 5, p. 20; F.1, p. 21) and about 0.17 Å on QM9 (App. H, p. 26). OMEGA's better local structure gives a lower median QM9 AMR-R (0.126 vs 0.147; Table 7, p. 26). **Caveat (review M1):** these are one-to-one matching numbers (each GT conformer gets one RDKit seed out of L), while AMR-R takes the best of 2L generated conformers with no one-to-one constraint. So they are upper-biased estimates, not lower bounds on AMR-R, and TD's QM9 AMR-R of 0.178 being close to 0.17 does **not** show that TD is floor-limited. The valid comparison is a floor measured on the run's own conformers (`floor_run`, or the distributional `floor_best_sym`, in `tools/local_structure_analysis.py`). **Not claimed for GEOM-XL:** TD improves XL AMR-R by 30% over RDKit and attributes XL difficulty to size and out-of-distribution training. XL RMSDs are symmetry-unaware upper bounds. "Strained systems" are not discussed. | "RDKit L imposes an accuracy floor (size to be measured with `floor_run`; TD's own matching estimates are 0.17 Å QM9 / 0.32 Å DRUGS, upper-biased) and cannot represent ring or macrocycle flexibility. On XL, TD also *fails outright* when RDKit cannot embed the molecule: 25/102 failures reported by MCF, 27/102 by ET-Flow." The XL local-structure contribution is a hypothesis to test by computing the conformer-matching floor on XL. |
| G2: trained with GT local structures, tested with RDKit ones | **Inaccurate.** TD identifies this exact shift and removes it with conformer matching: training conformers are replaced by RDKit-L conformers with RMSD-optimal torsions (Sec. 4.1, App. E). Training on GT L appears only as an ablation, where it collapses COV-R from 72.7 to 34.8 (Table 8). | "Conformer matching removes the shift but (i) trains on non-physical, non-minimum targets (App. F.4), (ii) builds in the RDKit floor, and (iii) ties the model to ETKDG's biases at test time." The real gap is *the workaround*, not the shift. |
| G3: ignores torsion–bond-angle coupling | **Partially right.** The extrinsic score model sees 3D coordinates, so it *does* learn p(τ \| L). What is missing is p(L \| τ): L is drawn from RDKit independently of τ and never updated (Alg. 3). The paper never discusses sterics-induced angle relaxation, and argues angles are stiff (RMSE 4.1°, App. F.1). Its ablation with random L–C pairing loses almost nothing (Table 8), but that ablation still DE-optimises each pair, so it tests the assignment, not coupling (review M1/§4). **Strongest in-paper evidence for G3 (App. F.1, p. 21):** matching a GT conformer to *another GT conformer's* local structure still leaves an average RMSD_min of **0.284 Å** on DRUGS, against **0.324 Å** with RDKit local structures. A perfect *independent* sampler of GT local structures would therefore remove only about 0.04 Å of the floor; most of it can only be removed by local structure matched to the torsions, i.e. L\|τ coupling. TD reads the same number the other way ("only slightly larger": RDKit L is nearly as good as GT L). The same M1 caveat applies: both are one-to-one matching numbers. | "One-directional coupling: torsions adapt to L, but L cannot adapt to torsions." External evidence: Torsional-GFN shows torsion modes shift with L; GO-Flow's ablation shows internal-coordinate flows help. |

---

## 1. Gap 1: rigid local structure

**Strongest approaches from the literature**
1. **Internal-coordinate flows over bond lengths, angles and torsions:** GO-Flow (2026; Jacobian composition into Cartesian space; removing the internal-coordinate manifold costs 4.6 COV-R points on DRUGS under GD-P). RINGER (joint angle + torsion diffusion for macrocycles). FoldingDiff (wrapped noise on angles and dihedrals).
2. **Ring factor:** PuckerFlow (Cremer-Pople flow matching, composable with TD), and Corso's thesis proposal of "ring puckering coordinates ... on hyperspheres" (Sec. 5.2).
3. **Prior that is corrected, not frozen:** EBD corrects an RDKit fragment prior, and more so when the prior is worse. The FM-refiner starts mid-trajectory from upstream samples.
4. **Cartesian models** (MCF, ET-Flow, AvgFlow, S23D, DMT, EnFlow) show that learning all local structure gives much lower error: QM9 AMR-R of 0.07–0.10 Å against TD's 0.178. This compares AMR-R with AMR-R; whether TD is floor-limited must be checked with `floor_run` (review M1). They lose TD's low dimensionality, likelihoods and parsimony.
5. **Better L-samplers:** LoQI/ChEMBL3D (AIMNet2-quality local structure), StoL (fragment assembly for large molecules). For macrocycles, ETKDGv3 macrocycle seeding (RINGER App.).

## 2. Gap 2: train/test local-structure shift and conformer matching

**Strongest approaches**
1. **Bridge instead of projection:** with a learnable L-factor, pair each GT conformer with an RDKit seed (keep TD's linear-sum assignment) and train the angle factor to transport L̂_RDKit to L_GT. The training targets are then the true GT conformers (no non-physical targets) and inference still starts from RDKit. Analogues: ET-Flow's prior-to-data alignment (removing it drops COV-P from 58.9 to 47.1 in reduced training, Table 7), EBD's correctable prior, the FM-refiner.
2. **Remove the external sampler entirely:** Cartesian models, GO-Flow, RINGER.
3. Corso's IDM (Sec. 2.4.1) frames matching as "project data onto the manifold". Enlarging the manifold, with the angle factor, shrinks the projection error, which is the principled fix.

## 3. Gap 3: torsion–bond-angle coupling

**Strongest approaches**
1. Joint noising and denoising of angles and torsions in one model: RINGER, FoldingDiff, GO-Flow (with a cross-space coupling stage).
2. **Evidence that coupling matters:** TD's own App. F.1 (p. 21): the GT-other-L floor is 0.284 Å vs 0.324 Å for RDKit L on DRUGS, so an independent L sampler, even a perfect one, leaves most of the floor (see §0). Torsional-GFN (torsion energy modes shift with L; the model tracks the shift for unseen MD L). GO-Flow's ablation.
3. **Energy or physics guidance** to penalise strained combinations: EnFlow's learned energy guidance; GFN2-xTB relaxation metrics (GEOM-revisited) to *measure* strain.

---

## 4. Recommended FlexiTors design

**State space:** the product manifold **M = T^m × B^k (× C^r)**:
- **T^m:** rotatable-bond torsions, exactly as in TD (wrapped normal, uniform prior, extrinsic pseudotorque head).
- **B^k:** a *low-dimensional bond-angle relaxation* space anchored at the RDKit seed L̂.
  - **Recommended basis:** the k softest angle-bending normal modes of the seed. Compute them from an MMFF or GFN-FF Hessian restricted to bond-angle internal coordinates, or use a fixed chemically defined subset — the angles at the ends of each rotatable bond, where sterics act (1-3 angles flanking a rotatable bond).
  - Using a mode basis sidesteps the non-independence of the angles around one centre: a centre with n neighbours has only 2n−3 independent angles.
  - Coordinates u ∈ R^k are kept within a box or ball of ±Δ (Δ ≈ 10–15°). Forward noise is Gaussian VE with **σ_max ≈ 5–8°**, the same order as RDKit's 4.1° angle RMSE (TD App. F.1). The prior is a narrow Gaussian centred on the seed, the informed-prior idea from ET-Flow/HarmonicFlow.
- **C^r (optional, stage 2):** Cremer-Pople phases (circles) and amplitudes for non-aromatic 5–8-membered rings, following PuckerFlow. This handles ring and strained-ring flexibility.
- Bond lengths stay frozen: RMSE 0.03 Å, negligible for RMSD.

**Score model:** keep TD's e3nn trunk. Add an **angle head that outputs SE(3)- *and parity-invariant* scalars** — bond angles are parity-invariant, unlike torsions, whose scores must be pseudoscalars (TD Sec. 3.3). The per-angle outputs are projected onto the k modes. Updates are applied in 3D through the internal-coordinate Jacobian: GO-Flow's bond-angle Jacobian (App. C), or the pseudo-inverse of the Wilson B-matrix restricted to the mode space. Torsion updates stay as TD's relative rotations (Prop. 1).

**Process:**
- Train with independent per-factor noise levels, and a shared time t with factor-specific σ(t) schedules (DiffDock product-space recipe; RFM if using flow matching).
- Optionally schedule the angle factor to resolve late — a subspace or blurring idea (TD Sec. 2 cites subspace diffusion; see EBD).

**Training data:** keep conformer matching's *assignment* step, but not its projection. The torsion target is the GT torsion under the paired seed, and the angle target is the GT angle deviation projected onto the seed's modes. The residual outside the mode space measures the new floor, which should be far below 0.324 Å.

**Likelihood:** Prop. 3 extends by appending the angle-mode Jacobian columns to the metric g. The divergence stays cheap since dim = m + k. This preserves TD's Boltzmann-generator capability, which Cartesian competitors lack.

**Inference extras:** particle guidance on the torsion factor; optional energy guidance on the angle factor (EnFlow); 1–2-step reflow later (AvgFlow).

**Novelty and positioning versus GO-Flow:** GO-Flow learns full internal coordinates from noise under GD-P. FlexiTors differs in four ways:
1. an RDKit-anchored **low-dimensional** relaxation subspace, keeping TD's parsimony and 20-step budget;
2. evaluation under TD-P;
3. exact likelihoods and a Boltzmann generator;
4. an explicit ring factor.

---

## 5. QM9 ablations that would provide evidence

QM9 notes:
- **Primary (pre-registered) threshold: δ = 0.5 Å**, together with AMR. Smaller thresholds are secondary/exploratory.
- At δ = 0.5 Å COV is close to **saturated**: median COV is 100 for all modern methods, and S23D (App. B.2.1) and the FM-refiner (Table 2) say so explicitly.
- As secondary/exploratory analyses, report a **threshold sweep including δ = 0.05, 0.1 and 0.25 Å** and local-geometry metrics.
- QM9 is a reasonable place to test G1: TD App. H (p. 26) names local structure as the limiting factor there, and QM9 is rich in small strained rings and cages. Whether TD is actually floor-limited on QM9 must be measured with `floor_run` / `floor_best_sym`; the "0.178 vs 0.17 Å" comparison is invalid (review M1).

| # | Arm | Question | Evidence for FlexiTors if... |
|---|---|---|---|
| A0 | TD retrained with our evaluator (seeds × 3) | Reproduce 92.8 / 0.178 / 92.7 / 0.221 | Baseline within ±0.005 Å |
| A1 | Floors: (a) conformer-matching RMSD_min with RDKit L on the QM9 test set; (b) the same with FlexiTors's k-mode space; (c) GT-L oracle (GT local structure, TD torsions) | How much AMR is due to L? | (b) ≪ (a); the gap between A0 and (c) is large |
| A2 | FlexiTors with k ∈ {0, 2, 4, 8, 16, all angles} | Is a *low-dimensional* relaxation enough? | AMR-R falls below the TD baseline's run-matched RDKit-L floor (`floor_run`) and saturates at small k |
| A3 | Joint vs. sequential (torsions then angle refiner conditioned on τ) vs. **independent** (angles denoised from the seed without seeing τ, or τ frozen) | G3: does coupling matter? | Joint ≈ sequential > independent; the gain concentrates on molecules with gauche/1,5-clash motifs |
| A4 | Angle training targets: GT bridge vs. conformer-matched (TD-style) vs. GT L without pairing | G2: does removing the projection help? | GT bridge > conformer-matched; GT without pairing is worst (replicates TD Table 8) |
| A5 | σ_max for angles ∈ {2°, 5°, 8°, 15°}; seed-centred vs. ideal-value prior | Prior and noise design (cf. ET-Flow's prior ablation) | A clear optimum near the RDKit error scale |
| A6 | Ring factor on/off (Cremer-Pople), results split by ring content (3/4-membered, fused, none) | G1 for strained rings | Gains concentrated in ring-containing and strained strata |
| A7 | Parity-invariant vs. pseudoscalar angle head | Symmetry sanity check | Invariant head works; the wrong-symmetry head degrades |
| A8 | Stratify every arm by the per-molecule RDKit-L floor (tertiles of A1a) | Does improvement track local-structure error? | Largest ΔAMR in the worst-floor tertile, near zero in the best |

**Metrics for every arm:**
- **Primary:** COV-R, AMR-R, COV-P, AMR-P (mean and median) at δ = 0.5 Å. **Secondary/exploratory:** a sweep from 0.05 to 0.5 Å.
- **Bond-angle MAE** against the matched GT conformer.
- **GFN2-xTB E_relax, and Δ-angle / Δ-length after xTB relaxation** (GEOM-revisited protocol).
- Median xTB property errors (TD Table 3 protocol, 100-molecule subset).
- Runtime per conformer.
- Paired bootstrap CIs over the 1000 test molecules across 3 seeds.
- External ceilings for context: ET-Flow 0.073, AvgFlowDiT 0.082, MCF-B 0.103 (QM9 AMR-R mean).

Follow-ups beyond QM9: DRUGS under TD-P; the GEOM-XL floor analysis (A1 on XL), including a failure count; CREMP macrocycles under RINGER's protocol.

---

## 6. Metric reference (for verifying evaluation code)

**Definitions.** Let L = number of ground-truth (reference) conformers {C*_l} of a molecule, and K = number of generated conformers {C_k}. Canonically **K = 2L**: TD Sec. 4.2 and App. C; GeoMol Sec. 5; GeoDiff Sec. 5.2; MCF Sec. 5.1. In the TD code, K = 2 × `n_conformers` from the test CSV (`generate_confs.py`), while L is counted *after* `clean_confs` filtering in `evaluate_confs.py`, so K can differ slightly from 2L.

- **COV-R(δ)** = (1/L) · |{ l ∈ [1..L] : ∃ k ∈ [1..K], RMSD(C_k, C*_l) < δ }|
- **AMR-R** (also called **MAT-R**) = (1/L) · Σ_l min_k RMSD(C_k, C*_l)
- **COV-P(δ)** = (1/K) · |{ k ∈ [1..K] : ∃ l ∈ [1..L], RMSD(C_k, C*_l) < δ }|
- **AMR-P** (also called **MAT-P**) = (1/K) · Σ_k min_l RMSD(C_k, C*_l)

Sources:
- TD Eq. 35, App. G.3, p. 24 ("precision metrics are obtained by swapping ground truth and generated conformers").
- GeoMol Eq. 5 (Sec. 5).
- GeoDiff Eq. 11 (Sec. 5.2), which uses the COV/MAT naming.
- MCF App. A.2.4, Eqs. 3–6.

Each metric is computed **per molecule**. Tables report the **mean and the median over test molecules**, a macro-average (TD Table 1).

**Implementation details in the reference code** (`torsional-diffusion/evaluate_confs.py`, derived from GeoMol):
- RMSD over **heavy atoms only** (`Chem.RemoveHs`), **symmetry-aware** with `AllChem.GetBestRMS`, which covers optimal alignment and atom permutations.
- For **GEOM-XL**, `--only_alignmol` uses `AllChem.AlignMol`, which is **not** symmetry-corrected, so XL numbers are upper bounds (TD App. G.3).
- The coverage comparison is **strict `<` δ**. EBD and some GD-P papers use ≤ δ.
- The code computes COV at all thresholds `np.arange(0, 2.5, 0.125)`; 0.5 and 0.75 are on the grid.
- The recall matrix is `rmsd[n_true, n_model]`. COV-R uses `min(axis=1) < δ`, averaged over true conformers; COV-P uses `min(axis=0)`, averaged over generated ones.
- **Failures:** a molecule with no model output is counted as coverage 0, but excluded from AMR (`np.nanmean`). Molecules whose GT conformers all fail the SMILES check are skipped.
- Ground-truth conformers are filtered by canonical SMILES match without stereo (`clean_confs`). For XL, the corrected SMILES is used.

**Thresholds and protocols**

| Setting | δ | Test set / split | Source |
|---|---|---|---|
| **TD-P, GEOM-DRUGS** | **0.75 Å** (older works used 1.25 Å) | 1000 test molecules; GeoMol random split 243473/30433/1000 | TD Table 1 caption (p. 8), App. G.1 |
| **TD-P, GEOM-QM9** | **0.5 Å** | 1000 test molecules; 106586/13323/1000 (the MCF processed set has 995) | TD Table 7; S23D App. B.2.1 |
| **GEOM-XL** | **no coverage reported; AMR-R and AMR-P only** | 102 MoleculeNet molecules with ≥100 atoms (main text says ">100"), model trained on DRUGS; TD's checkpoint fails on 25–27 molecules in other groups' runs, so some papers report a 75/77-molecule subset | TD App. G.1, Table 6; MCF Table 3; ET-Flow Table 8 |
| GD-P (GeoDiff / ConfGF / GeoMol-original), DRUGS | 1.25 Å | 200 test molecules (14,324 conformers), 40k training molecules × 5 conformers | GeoDiff Sec. 5.1–5.2; GeoMol Sec. 5; EBD; GO-Flow |
| GD-P, QM9 | 0.5 Å | 200 test molecules (22,408 conformers) | GeoDiff |
| Tight QM9 (**secondary/exploratory only**; the pre-registered primary QM9 threshold is 0.5 Å) | 0.05 Å | TD-P test set | FM-refiner Table 2 |
| Rings (PuckerFlow) | 0.1 Å | own ring set | PuckerFlow Sec. 2 benchmark setup, Tables 1–2 |
| Macrocycles (RINGER / CREMP) | 0.75 Å all-atom; 0.1 Å ring RMSD; 0.05 ring TFD | CREMP 1000 test molecules | RINGER Fig. 3 / Table S.7 |

**Training-side protocol knobs that change results** (report them):
- Maximum number of GT conformers used per training molecule: TD uses 30 (App. C); S23D's "KG" column shows 10, 20 or 30 across papers.
- Chirality handling: TD takes chirality from RDKit; ET-Flow and S23D use post-hoc correction.
- Number of sampling steps.
- Whether MMFF or xTB relaxation is applied before scoring. TD Table 3 uses GFN2-xTB relaxation for properties only, not for RMSD.

**Known pitfalls**
- Recall is gameable by oversampling and clustering (Zhou et al. 2023) — always report precision.
- QM9 at δ = 0.5 Å is near-saturated (S23D; FM-refiner), but it stays the **pre-registered primary threshold** for this project; tighter thresholds are secondary/exploratory.
- TD's conformer-matching floors (0.324 Å DRUGS, Table 5 p. 20 / F.1 p. 21; about 0.17 Å QM9, App. H p. 26; 0.284 Å GT-other-L, F.1 p. 21) are one-to-one matching estimates. They are **not lower bounds on AMR-R**, which takes the best of 2L with no one-to-one constraint (review M1). Compare AMR-R only against run-matched floors (`floor_run` / `floor_best_sym`, `tools/local_structure_analysis.py`).
- Some papers' XL tables mislabel baseline rows (MCF Table 3 and ET-Flow Table 8: the "GeoDiff" row holds TD's RDKit numbers, and the header order is AMR-P, AMR-R where TD uses AMR-R, AMR-P).
- Reproductions can fall short of reported numbers: TD+PG in NExT-Mol Table 4, ET-Flow in EnFlow Table 2.
- GEOM contains xTB-fractured molecules (GEOM-revisited).
