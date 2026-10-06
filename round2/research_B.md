# Round 2, research agent B: what round 1 shows, threats to validity, and round-2 QM9 ablations

Written 2026-10-06 by research agent B. I worked independently and did not read `round2/research_A.md`.

**Conventions**
- Paths are relative to `C:\Users\HP\Desktop\tor_diff`.
- Result numbers are copied from the files named next to them. Numbers I computed myself from the shipped CSVs are marked **[B-calc]**, with the script logic described, so they can be re-run.
- Paper quotes are copied from `papers/**/fulltext` (pages are counted by form feeds, i.e. PDF pages). For TD the PDF page equals the printed page. In the fulltext, the Å glyph comes out as a replacement character; I write it as "Å".
- Anything I could not verify is marked UNVERIFIED.

---

## 0. Headline (TL;DR)

1. **B1 is not a mild shift; it collapses to no-skill.** B1 on RDKit L scores AMR-R 0.2342 and COV-R 88.85 (`paired_B1_vs_CTRLbase.md`). That is the level of having no model at all: A0_rdkit gets AMR-R 0.2298 and A0_rand 0.2422 (`all_summaries.txt`). TD Table 8 shows the same collapse on DRUGS: train-on-GT-L gives 0.920 against the random baseline's 0.922.
   - The same B1 network is the best model in the study when it gets GT L: 0.0369, or 0.0212 when each sample gets its own conformer's L.
   - B1 is also far more sensitive to whose L it gets than the released model is. Own-L vs random GT L is worth −0.016 Å for B1 against −0.0037 Å for the released model.
   - So B1 reads τ-information out of L. That is an extreme, brittle form of G2/G3, not a small bias.
   - The literature's name for this is train–test mismatch of the conditioning input (cascaded diffusion), and the standard fix is noise augmentation of the conditioning input. Round 1 never tested it.
2. **Most of the L floor on QM9 sits in rings, not in acyclic angles [B-calc].**
   - The per-molecule RDKit-L floor (`floor_best_sym`) is 0.1095 Å. A perfect independent L sampler (`floor_gtother`) gets 0.0258 Å.
   - Setting all acyclic heavy-atom angles and acyclic bond lengths to GT removes at most 0.0198 Å of the 0.0837 Å L-attributable part, i.e. **≤ 24 %**.
   - 89 % of test molecules contain a ring, and they carry 94 % of the floor mass.
   - 30 % of molecules (284/939) have no heavy-atom torsion at all. For them the torsion model can do nothing: their floor is 0.148 Å and the acyclic oracle barely moves it.
   - So an acyclic-angle factor alone cannot be the FlexiTors story on QM9; ring geometry has to be generated.
3. **The run-matched floor says ~72 % of R0's per-molecule error is L-limited.**
   - `floor_run` per-molecule mean is 0.1195 Å against an observed 0.1667 Å (`qm9_default/R0_base_seed0/floor_run.log`).
   - The per-GT-conformer means are 0.0993 vs 0.1661 Å, i.e. 60 %.
   - The torsion headroom (0.047 Å per molecule) is a lower bound, because DE is a heuristic optimiser.
4. **The A1 COV-R@0.5 gain (+7.5) is mostly failure accounting.**
   - `tools/paired_compare.py:64-72` counts a failure as COV 0 over the union of molecules. R0 fails on 65–69 molecules (ETKDG cannot embed them); A1 fails on 4.
   - On common molecules the gain is +1.0 to +4.8 points (bound below, [B-calc]), most likely near the low end.
   - The AMR-R conclusions are not affected: they are computed on common molecules.
5. **δ = 0.5 Å has almost no power for exactly the effects FlexiTors targets.**
   - A2 lowers AMR-R by 15 % (−0.026 Å), yet COV-R@0.5 moves by +0.05 (n.s.).
   - At 0.25 Å the same arm moves +4.6 [3.3, 6.0], and at 0.1 Å +12.4.
   - Round 2 should make COV-R@0.1, on the success intersection, co-primary with AMR-R. COV@0.5 stays as the protocol number, and COV@0.05 is added for comparison with FM-refiner/ET-Flow.

---

## 1. Interpretation of each round-1 finding

### 1.1 Reproduction and baseline (R0, CTRL)

- **The released checkpoint reproduces.**
  - R0 seeds 0–2: AMR-R 0.1752 / 0.1768 / 0.1761. Released pickles: 0.1765 (`all_summaries.txt`).
  - Sampling-seed SD of AMR-R is 0.0008 (`paired_vs_R0_qm9_default.md`, column `ref_replicate_sd`).
- **100-epoch retrains.**
  - CTRL_base AMR-R is 0.1806, with a training-seed SD of 0.0013 (`paired_B1_vs_CTRLbase.md`).
  - Our regenerated pickles (CTRL_rematch) are 0.002 better (CI [−0.0033, −0.0007]). That is a small nuisance, and the matched control for B2/B5 was the right call.
  - The 0.005 gap between 100-epoch CTRL and the released model (trained longer; upstream recipe) is training budget. Round-2 training arms must therefore be compared against 100-epoch controls, never against R0.
- **Detection limits.** With 3 seeds the effects ≥ ~0.004 Å are resolvable for training arms; the B2 effect of 0.0021 had Holm p = 0.029.

### 1.2 Failures and the COV-R@0.5 discrepancy (92.8 vs 88.8)

- **What fails.** Every arm seeded from RDKit has 65–69 model failures. GT-L arms have 4 (`all_summaries.txt`, `n_model_failures`).
  - `RESULTS_QM9.md` §2.2: 60 of them are ETKDG embedding failures on strained cages, and 5 are rejected inputs.
  - The mechanism is in the code: `diffusion/sampling.py:54-63` returns `[]` when ETKDG yields fewer conformers than requested. `evaluate_confs.py:81-84` then counts a model failure, and `evaluate_confs.py:152` appends `[0] * num_failures` to COV.
- **Excluding failures** gives a COV-R@0.5 of ~95 for every RDKit-L model: CTRL_base is 94.91 on 935 molecules in `paired_B1_vs_CTRLbase.md`.
  - The paper's 92.8 sits between 88.8 (failures counted) and ~95 (failures excluded). It cannot be reproduced from the released pickles (931 molecules, 88.8; `RESULTS_QM9.md` §2.2).
  - My reading: the paper's run probably had fewer embedding failures, e.g. a different RDKit version. **UNVERIFIED.** No round-1 conclusion depends on this number.
- **Local check [B-calc]** (RDKit 2026.03.6, which is not the cluster's 2022.9.5; first 4 conformers only):
  - Of the 57 molecules that fail in `local_structure_test.csv` (`error == embed_failed`), only 10 embed with default ETKDG and 14 with `useRandomCoords=True`.
  - So the failures are mostly intrinsic to ETKDG on cages, not an easy flag fix.
  - **Design implication.** A FlexiTors that still depends on ETKDG seeds inherits a ~6.5-point COV tax at δ = 0.5. Cartesian methods with a learned or harmonic prior (ET-Flow) do not have it.
  - A learned L sampler that needs no ETKDG (or a GT-free fallback) is a second, independent argument for learning L.

### 1.3 A1 / A1c: GT local structure + released model

- **AMR-R.** A1 gives −0.0926 [−0.099, −0.086] vs R0 on 927 common molecules (`paired_vs_R0_qm9_default.md`). This is the robust headline: the released model, given true L, halves its error.
- **COV-R@0.5 +7.5 is inflated** (threat T1 below). The universe is 996 molecules. R0 has ~61 more failures than A1, each scored 0.
  - R0's COV on its own successes: 88.98 × 1000/935 ≈ 95.2.
  - A1 on the 935 common molecules: (96.48·996 − 61·c)/935, where c is A1's coverage on the cage molecules.
  - That gives 96.25 for c = 100 and 100 for c = 42.5, so the common-molecule gain is between **+1.0 and +4.8**. **[B-calc]**; the exact value needs the eval.pkl files (proposal R2-0).
- **Small thresholds are where the effect lives.** COV-R@0.1: +42.7; COV-R@0.05: +62.8.
- **A1c vs A1 (own L vs random GT L).** −0.0037 [−0.0063, −0.0013] (`paired_A1c_vs_A1_qm9_default.md`), i.e. small coupling for a model trained on τ-independent (conformer-matched) L.
  - The same contrast inside B1 is much larger: B1_gtLc 0.0212 vs B1_gtL 0.0369 (`paired_B1_vs_CTRLbase_gtL.md`, `paired_B1_vs_CTRLbase_gtLcycle.md`).
  - Whether the model can exploit τ↔L coupling depends on the training L distribution. A matched-L model (CTRL) cannot learn it, because conformer matching decorrelates L from τ: the L̂ come from RDKit, independent of the GT torsions.
  - **G3 on QM9 is therefore not "small"; it is invisible to a conformer-matched model.**

### 1.4 A2 / A3: MMFF before vs after

- **A2 (MMFF before diffusion).** AMR-R −0.026 [−0.031, −0.021]; COV-R@0.5 +0.05 (n.s.); COV-R@0.25 +4.6; COV-R@0.1 +12.4 (`paired_vs_R0_qm9_default.md`).
- **The run-matched floors explain it** (`qm9_default/*/floor_run.log`, per GT-conformer means):

  | Arm | obs | floor_run | torsion_headroom |
  |---|---|---|---|
  | R0 | 0.1661 | 0.0993 | 0.0668 |
  | A2 | 0.1490 | 0.0832 | 0.0658 |

  - The floor drops by 0.016 and the observed error by 0.017, while the headroom is unchanged.
  - So the RDKit-matched model transfers its full skill to MMFF L, which it never saw in training.
  - **Contrast with B1** (§1.6): the CM-trained model is robust to an L shift; the GT-trained model is not. The broad, noisy L distribution of conformer-matched training acts as conditioning augmentation, which motivates R2-2.
- **A3 (MMFF after).** Headroom grows to 0.1221 (floor 0.0810, obs 0.2031). MMFF after sampling moves conformers away from GT torsions: recall gets worse, precision better (AMR-P −0.057).
  - A post-hoc force-field step cannot stand in for generating L.

### 1.5 G1 (GT stereo)

- AMR-R +0.0001 [−0.0035, 0.0038]. The floor is identical (`G1_gt_graph_seed_mols/floor_run.log` floor_run 0.0985 vs R0 0.0993).
- Stereo is not a QM9 factor. The test-mode stereo match is 0.9993 (`local_structure_test.log`).

### 1.6 B1: trained on GT L ("raw", no conformer matching)

**The 2×2 matrix** (AMR-R mean, 3 training seeds, generation seed 0):

| train \ test | RDKit L | GT L (random) | GT L (own, cycle) |
|---|---|---|---|
| CM / RDKit L (CTRL_base) | 0.1806 | 0.0842 | 0.0820 |
| GT L (B1) | 0.2342 | 0.0369 | 0.0212 |

Sources: `paired_B1_vs_CTRLbase.md`, `paired_B1_vs_CTRLbase_gtL.md`, `paired_B1_vs_CTRLbase_gtLcycle.md`.

**The literature reading.**
- TD names exactly this train/test L shift as the reason for conformer matching (TD p. 7). In TD Table 8 (DRUGS, p. 27) train-on-GT-L reaches the random baseline (COV-R 34.8 vs 30.9; AMR-R 0.920 vs 0.922). TD notes that its score-matching loss is "significantly lower" (p. 26).
- On QM9 the pattern is the same: B1 on RDKit L (0.2342) is between A0_rdkit (0.2298) and A0_rand (0.2422). COV-R@0.5 per training seed is 82.8 / 82.0 / 84.4, against A0_rdkit 83.9 (`all_summaries.txt`).
- **The model contributes nothing on RDKit L.** That is a qualitative collapse, not a dose-proportional bias.
- EBD names the same shift for RDKit fragment priors and fixes it by making the forward process end at the RDKit prior (EBD p. 5). GeoMol avoids it by predicting LS itself (GeoMol p. 4). ET-Flow avoids it with a harmonic Cartesian prior (ET-Flow p. 1).
- Outside chemistry, the same failure is the cascaded-diffusion train–test mismatch. The fix there is Gaussian conditioning augmentation, which is called "crucial" (Ho et al. p. 3, 6). Input perturbation for exposure bias is the same idea (Ning et al. p. 2).

**Is it just distribution shift (G2)? Mostly yes, with a specific mechanism I propose and that is testable.**
- In GEOM, L and τ are coupled: angles relax in response to torsion-dependent sterics. A model trained on GT pairs can read τ cues from L.
- Evidence:
  - (i) Own-L vs random-GT-L is worth 0.016 Å for B1 against 0.0037 for CTRL.
  - (ii) The training loss collapse reported by TD.
- With RDKit L those cues are noise. The model then outputs confident but wrong torsions, which is worse than leaving the ETKDG torsions.
- This is a **hypothesis (UNVERIFIED)**. The cheap leak diagnostic in R2-0 tests it.

**What round 1 cannot tell.**
- (a) How accurate L must be before a GT-trained τ-model stops collapsing. That is the spec for FlexiTors' L generator → R2-1.
- (b) Whether augmentation gives both robustness and the GT-L gain → R2-2.

**Minor confound (T7).** B1's training set differs from CTRL's in composition, not only in L:
- `standardize_confs.py:58-67` caps conformers at 30.
- `:74-76` drops molecules with no rotatable bond ("no_rotable_bonds").
- `:79-82` drops molecules that ETKDG cannot embed ("rdkit_no_embed").

The raw path (`utils/dataset.py:200-239`) keeps them. This is unlikely to explain a 0.05 Å swing in either direction, but it should be listed.

### 1.7 B2 (DE-only random pairing) and B5 (heavy-atom matching objective)

- **B2:** +0.0021 [0.0006, 0.0037] vs CTRL_rematch (Holm p = 0.029). This matches TD Table 8 on DRUGS (0.582 → 0.588): the Hungarian assignment buys little, as TD found. It is not a coupling test (`review/cross_validation.md` §4 vote).
- **B5:** +0.0058 [0.0038, 0.0080] worse, and worse at every small threshold (COV-R@0.1 −2.6). That contradicts the code-concern hypothesis (M2) that heavy-atom matching gives cleaner heavy-torsion targets.
  - A possible reason: heavy-objective DE leaves H-rotor torsions unoptimised, so the H geometry of the training conformers becomes unphysical and the network reads it. **UNVERIFIED.**
- **Conclusion:** cheap matching-objective tweaks do not move QM9. This argues against spending round-2 GPU on B4 (DE maxiter 50).

### 1.8 Training-set matching residual (A0-train)

- `local_structure_std.log`: the stored `conf['rmsd']` has mean 0.1628 Å and median 0.1086 Å; 95.8 % of conformers are > 0.05 Å.
- The CM-trained model is therefore trained on targets that are themselves ~0.16 Å from GT. That alone explains why CTRL cannot reach the sub-0.05 Å regime even when given GT L (CTRL_gtL COV-R@0.05 64.2 vs B1_gtL 87.7).
- This is the "non-physical targets" cost of conformer matching (G2-ii), and it is quantitatively large on QM9.

---

## 2. Decomposition of R0's error: angles vs lengths vs rings (question 2)

**Sources.** `cluster_sync/results/analysis/local_structure_test.csv` (`--mode test`, K = 2L ETKDG seeds from the GT graph, DE heavy-atom symmetric floors; definitions in `tools/local_structure_analysis.py:1-37`, oracles at `:196-214` and `:293-298`).

**The shipped log is distorted by one molecule.** The `local_structure_test.log` numbers are per-GT-conformer means and include the one multi-fragment molecule `C=C1C(=O)C=NN1C.N`, which has 30 GT conformers with floors of ~5 Å. That molecule is a model failure in every generation run (`RESULTS_QM9.md` §2.2; commit 04a799e).
- Excluding it [B-calc], the per-row means become: `floor_best_sym` 0.0796 (log: 0.0986); angle oracle 0.0694 (log: 0.0885); local oracle 0.0544 (log: 0.0735); `floor_gtother` 0.0250 (log: 0.0269).
- Per-molecule means (the weighting AMR-R uses) are below. **Use these, not the log.**

**Per-molecule floors** (939 molecules, multi-fragment excluded) [B-calc]:

| class | n mol | floor_best_sym (RDKit L) | + acyclic angles = GT | + acyclic bonds = GT | floor_gtother (perfect indep. L) | share of floor mass |
|---|---|---|---|---|---|---|
| all | 939 | 0.1095 | 0.1008 | 0.0927 | 0.0258 | 100 % |
| acyclic | 107 | 0.0589 | 0.0424 | **0.0097** | 0.0154 | 6 % |
| rings + heavy torsions | 548 | 0.0995 | 0.0891 | 0.0821 | 0.0207 | 53 % |
| rings, 0 heavy torsions | 284 | 0.1478 | 0.1455 | 0.1443 | 0.0518 | 41 % |

**Reading**
- **Acyclic angles + acyclic bond lengths** remove 0.0168 Å as in the table. Conservatively, taking min(floor, oracle) per row because sequential `SetAngle` can make the oracle worse than the floor (it does so in 26 % of rows for the angle oracle and 13 % for the local oracle), they remove 0.0198 Å. Either way that is **≤ 24 %** of the L-attributable 0.0837 Å.
- **For acyclic molecules, bond lengths matter as much as angles.** The angle oracle goes 0.0589 → 0.0424; adding bond lengths takes it to 0.0097, below the GT-GT level.
  - RDKit heavy-atom bond RMSD vs GT is 0.036 Å, against 0.004 Å between GT conformers. The angle RMSD is 3.89° vs 1.68° (`local_structure_test.log`, `bond_rmsd_assigned` / `angle_rmsd_gtgt`).
  - Bond lengths are ~9× noisier than GEOM's own variability, but they are a near-deterministic function of the graph, so they can be *predicted* (GeoMol-style) rather than diffused.
- **Ring-containing molecules carry 94 % of the floor**, and the acyclic oracle removes only ~10–17 % of theirs.
  - 284 molecules (30 %) have no heavy-atom torsion. Their error is 100 % L. R0's own breakdown agrees: `R0_base_seed0/breakdown.log` gives MAT-R 0.149 for `n_rot_heavy = 0` (281 molecules).
  - Their `floor_gtother` is 0.052. The GT ensembles of rigid QM9 molecules contain genuinely different ring geometries (puckers, ring-strain variants), and a torsion model cannot reach them.
- **MMFF seeds** (`local_structure_test_mmff.csv`, [B-calc]) lower the per-molecule floor to 0.0915. Most of the gain is in rigid rings (0.148 → 0.120), so a force field already fixes part of the ring geometry.
- **Small rings dominate the test set.** 60 % of molecules contain a 3- or 4-membered ring. For a 3-ring the Cremer-Pople puckering space is empty, and for a 4-ring it is one-dimensional.
  - QM9 ring error is therefore largely ring bond lengths and angles under strain, not only puckering. A pure pucker factor (PuckerFlow-style) covers 5/6-rings. Small strained rings need ring internal angles and lengths, or a Cartesian/learned ring refiner.

**Implications for FlexiTors (which internal coordinates to generate)**
1. Ring geometry is mandatory on QM9. TD itself scopes rings into L (TD p. 4) and admits the weakness for "puckered rings, fused rings" (TD p. 23). The Corso thesis proposes puckering coordinates (p. 63).
2. Acyclic bond lengths should be predicted, not taken from RDKit; that is cheap.
3. Acyclic bond angles are worth ≤ ~10 % on QM9. They are probably more valuable on DRUGS (TD F.1 0.324 vs 0.284 Å, p. 21) but are not the QM9 lever.
4. **Caveat.** The oracles set GT values exactly, which is *more* than any generator achieves, so these are optimistic per-coordinate gains. A ring oracle was never computed. R2-5 adds it, both as a CPU floor and *with the model in the loop*.

---

## 3. Is QM9 at δ = 0.5 Å saturated? (question 3)

- **Yes.** Every arm has a COV-R@0.5 median of 100 (`all_summaries.txt`). A2's 15 % AMR-R gain does not register at 0.5 Å (+0.05, n.s.), but does at 0.25 (+4.6) and 0.1 (+12.4).
- **The literature agrees.**
  - S23D argues that QM9 is "likely to be saturated" (S23D p. 16).
  - FM-refiner moves to δ = 0.05 Å because median COV is already 100 % at 0.5 Å (FM-refiner p. 7). It reports ET-Flow at COV-R@0.05 75.72 and AMR-R 0.083 (FM-refiner p. 7). ET-Flow's own QM9 AMR-R is 0.073 (ET-Flow p. 19).
- **Dynamic range at the candidate thresholds** (COV-R; RDKit-L arms → GT-L arms):

  | δ | R0 | A1 | B1_gtL | B1_gtLc |
  |---|---|---|---|---|
  | 0.05 | 3.9 | 66.7 | 87.7 | 92.3 |
  | 0.1 | 37.5 | 80.3 | 92.9 | 96.8 |
  | 0.25 | 74.1 | 89.6 | 95.8 | 98.6 |

  - At 0.05 every RDKit-L arm sits on the floor (2.5–4 %), so 0.05 only separates learned-L arms. The GT-GT ceiling there is set by `floor_gtother`: 7.7 % of GT conformers are > 0.05 Å [B-calc], so even a perfect independent L sampler is capped near ~92 %.
  - **0.1 Å has range on both sides** and the smallest relative replicate SD (0.24–0.39 points).
- **Recommendation**
  - Pre-register for round 2: **co-primary AMR-R mean and COV-R@0.1, both on the intersection of molecules that succeed in all compared arms**, with failure rate as a separate endpoint.
  - Keep COV-R@0.5 (failures = 0) as the protocol-comparability row.
  - Report COV-R@0.05 as the literature-comparison row (FM-refiner, ET-Flow).
  - This needs a user decision, since round 1 fixed δ = 0.5 by user decision (`review/cross_validation.md` S4 / §6b).

---

## 4. Threats to validity (question 4)

| ID | Threat | Which round-1 conclusion | Severity | Evidence | Fix |
|---|---|---|---|---|---|
| T1 | Failures count as COV 0 over the union of molecules | A1/A1c COV-R@0.5 +7.5 vs R0 (inflated; true +1.0 to +4.8) | major for COV claims; none for AMR | `tools/paired_compare.py:12-15, 64-72`; `evaluate_confs.py:81-84, 152` | R2-0: intersection COV + separate failure endpoint |
| T2 | Multi-fragment molecule (30 conformers × ~5 Å) inside the test-mode floor | `local_structure_test.log` per-row means (0.0986 → 0.0796) | moderate (the numbers are being quoted) | §2 [B-calc]. Nikitin et al.: fragmented GEOM molecules come from failed xTB (p. 8) | Exclude `'.'` SMILES in `local_structure_analysis.py`; report per-molecule means |
| T3 | DE floors are upper bounds; the convergence spot check (review M1-dir) was never run | "≈72 % L-limited", torsion headroom 0.047 | moderate (biased toward the FlexiTors story) | `review/cross_validation.md` §3 M1-dir | R2-0: `--maxiter 200 --floor_topm 10` on 50 molecules |
| T4 | Oracles: sequential `SetAngle` can make things worse (26 % of rows) | "acyclic share ≤ 24 %" | low (I report the conservative min) | `tools/local_structure_analysis.py:196-214`; [B-calc] | R2-5 ring/acyclic oracle via constrained embedding |
| T5 | Paper COV 92.8 vs our 88.8 | none (never used) | low | `RESULTS_QM9.md` §2.2 | Never compare COV@0.5 with literature; compare AMR and intersection COV |
| T6 | Literature QM9 numbers come from methods without ETKDG failures (ET-Flow uses a harmonic prior) | future SOTA claims | moderate for the paper | ET-Flow p. 1 | Report FlexiTors both with failures = 0 and on the intersection |
| T7 | B1 training-set composition differs from CTRL (30-conformer cap, no-rotor and no-embed molecules dropped) | B1 vs CTRL magnitude | low | `standardize_confs.py:58-67, 74-76, 79-82`; `utils/dataset.py:200-239` | Optional B1' on the CTRL molecule set (not in the shortlist) |
| T8 | Generation seed 0 only for the training arms | all B arms | low (training-seed SD 0.0013 is the dominant noise) | `slurm/ablations_train.tsv` | Fine; keep the same ETKDG pairing |
| T9 | Nikitin et al. evaluation bugs | none directly | none | Their bugs concern valency/stability metrics in de novo generation (Nikitin p. 1), not RMSD coverage. Their Δ bond/angle/torsion metrics are recommended (p. 11) | Add `geometry_metrics.py` outputs as secondary endpoints for FlexiTors |
| T10 | RDKit-version dependence (ETKDG L, `symmetrizeConjugatedTerminalGroups`) | cross-study comparability | low inside the study | `review/metric_verification.md` §3 | Pin 2022.9.5; record the version |
| T11 | B1's GT-L "upper bound" may partly be L→τ decoding of the source conformer | how to read 0.037 / 0.021 | interpretive | §1.6 | R2-0 leak diagnostic |

**Robust conclusions** (survive all of the threats above):
- A1 AMR-R −0.093
- B1's collapse on RDKit L
- B1 ≫ CTRL on GT L
- A2 AMR-R −0.026
- B2 and B5 are tiny or negative
- G1 is null
- Rings dominate the L floor

**Fragile conclusions:**
- Any COV-R@0.5 difference between arms with different failure counts
- The exact torsion/L split until DE convergence is checked

---

## 5. Proposed round-2 ablations

**Budget.** 2 days × 4 RTX 3090 = 192 GPU-h. One training run takes ~10.7 h and one inference+eval ~0.25 h. All arms below use the TD split, K = 2L, heavy-atom symmetric RMSD, 3 seeds where marked, and `paired_compare.py` with the endpoints of §3.

### R2-0: Evaluation hygiene and diagnostics (CPU only) **[priority 1, prerequisite]**

- **Hypothesis.** Round-1 COV conclusions change on the success intersection. B1 decodes τ from L. The floor/headroom split survives a stronger DE.
- **Gap.** Validity (T1–T3, T11).
- **Setup.**
  - (a) Add a `--universe intersection` option to `paired_compare.py` (`:99`) and re-run all round-1 tables, also at thresholds 0.1 / 0.25 / 0.05.
  - (b) Rerun `local_structure_analysis.py --mode test` summaries without multi-fragment SMILES, reporting per-molecule means.
  - (c) DE spot check: 50 molecules with `--maxiter 200 --floor_topm 10` for R0 and A2.
  - (d) `--mode run` floors for B1 (RDKit L) and for B1_gtL / CTRL_gtL.
  - (e) **Leak diagnostic** on the existing `steps20_seed0_gtLcycle` `confs.pkl` of B1 and CTRL_base. Generated conformer i was seeded with the L of GT conformer i mod L (`diffusion/sampling.py:75`). Compute the fraction of generated conformers whose nearest GT conformer is their L-source, against the chance rate 1/L.
- **Control.** CTRL_base on the same inputs.
- **Metric and expected direction.**
  - (a) The A1 COV@0.5 gain shrinks to ≲ +2.
  - (c) The headroom grows by < 10 %.
  - (e) B1's source-match rate ≫ CTRL's (if the leak hypothesis is right).
- **Evidence.** TD p. 26 (lower loss for GT-L training); review M1-dir.
- **Cost.** 0 GPU-h; ~6–10 CPU-h on 32 cores.
- **Priority.** 1. Must finish before round-2 numbers are read.

### R2-1: L-quality dose-response for a GT-trained vs a CM-trained torsion model (inference only) **[priority 1]**

- **Hypothesis.** B1's AMR-R degrades monotonically with L error and crosses CTRL at some error ε*. ε* is the accuracy spec for FlexiTors' L generator. CTRL is nearly flat.
- **Gap.** G2: how far can learned L be from GT before a τ|L model trained on GT L is worse than TD?
- **Setup.** Test-time L for B1 (3 training seeds) and CTRL_base (3 training seeds):
  - (1) GT L + isotropic Cartesian noise σ ∈ {0.01, 0.02, 0.04} Å. This needs a new hook `--seed_noise` in `diffusion/sampling.py` after seed selection (`:75-78`), applied before `perturb_seeds`.
  - (2) GT L relaxed with MMFF: a seed-pickle variant of `tools/make_seed_pickles.py`. `--pre_mmff` does not act on `--seed_confs` (`diffusion/sampling.py:54-63`).
  - (3) RDKit + MMFF (`--pre_mmff`). This is the "B1 trained on GT L, tested on MMFF-relaxed L" cell the brief asks for.
  - (4) RDKit L and plain GT L are already run.
  - Measure the L error of each condition with `tools/geometry_metrics.py` (bond MAE, angle MAE).
- **Control.** CTRL_base under the same conditions. The A1 / A2 arms of the released model serve as external anchors.
- **Metric.** AMR-R (intersection) and COV-R@0.1 against angle RMSD. Expected direction: B1 rises steeply between angle RMSD ~1.7° (GT-GT) and 3.9° (RDKit), and is probably already ≥ CTRL near MMFF level (2.4°). CTRL degrades only mildly.
- **Evidence.**
  - TD p. 7: the shift "significantly hurts performance".
  - EBD p. 5: the shift "can potentially harm the performance".
  - A2 vs B1 asymmetry (§1.4).
- **Cost.** 2 models × 3 seeds × 5 new conditions × 0.25 h ≈ 7.5 GPU-h. Hook code ~30 lines.
- **Priority.** 1.

### R2-2: B6, a conditioning-augmented GT-L training (noise-augmented L) **[priority 1]**

- **Hypothesis.** Training on GT L with random L perturbations removes B1's collapse on RDKit L (AMR-R ≤ CTRL's 0.181) while keeping most of the GT-L advantage (AMR-R on GT L ≤ ~0.05). If it works, FlexiTors' τ|L component can be trained on GT data, and the cascade "generate L, then τ" is viable without conformer matching.
- **Gap.** G2, plus the exposure bias of the FlexiTors cascade (its τ-model will see *generated* L).
- **Setup.**
  - Data variant `raw`, as B1.
  - In `TorsionNoiseTransform.__call__` (`utils/dataset.py:24-43`), after selecting `data.pos` (`:29`) and before the torsion perturbation (`:40-41`), add Cartesian noise σ_L ~ U[0, 0.04] Å to all atoms. This brackets the RDKit L error: a bond-RMS error of ~σ√2 ≈ 0.04 Å at the top of the range.
  - B6a: 3 training seeds at this range. B6b: 1 seed with σ_L fed as a node feature (the "amortized" variant), so the strength can be picked at test time.
  - Evaluate on RDKit L, `--pre_mmff`, GT L and GT-cycle.
- **Control.** B1 (σ_L = 0) and CTRL_base, both existing; paired.
- **Metric.**
  - AMR-R / COV-R@0.1 on RDKit L: expected far better than B1 and ≈ or < CTRL.
  - On GT L: expected slightly worse than B1, much better than CTRL.
- **Evidence.**
  - Ho et al. p. 3: conditioning augmentation "alleviates compounding error in cascading pipelines due to train-test mismatch".
  - Ho et al. p. 6: Gaussian noise works best, and the augmentation level can be amortised.
  - Ning et al. p. 2: modelling the prediction error during training.
  - TD p. 7: the shift motivates matching.
- **Cost.** 4 runs × 10.7 h ≈ 43 GPU-h, plus 4 × 4 evaluations × 0.25 h = 4 GPU-h.
- **Priority.** 1.
- **Risk.** Cartesian noise also perturbs H positions and torsions slightly, so it is not a pure L perturbation. Accept it for round 2. An internal-coordinate perturbation (angles/lengths only) is a round-3 refinement.

### R2-5: Ring vs acyclic local-structure oracle, with the model in the loop **[priority 1]**

- **Hypothesis.** GT *ring* geometry with RDKit acyclic geometry recovers most of the A1 − R0 gap; GT acyclic geometry with RDKit rings recovers ≤ ~25 %. This decides whether FlexiTors needs a ring factor (and of which kind) before an acyclic-angle factor.
- **Gap.** G1, i.e. which coordinates of L to generate (§2).
- **Setup.** Two new seed pickles:
  - **A5-ring:** for each GT conformer, ETKDG embedding with `coordMap` fixing all ring-atom coordinates to GT (a variant of `tools/make_seed_pickles.py`). Record embedding failures, which are themselves informative for the cages.
  - **A5-acyc:** an ETKDG seed with acyclic heavy bonds and angles set to GT via `set_acyclic_local` (`tools/local_structure_analysis.py:196-214`).
  - Run the released model (3 sampling seeds) and B1 (3 training seeds) with `--seed_confs` on both.
  - Add CPU floors (`floor_best_sym` analogue) for both pickles, stratified by smallest ring size (3/4/5/6).
- **Control.** R0 / CTRL on RDKit L (floor) and A1 / B1_gtL (full GT L, ceiling).
- **Metric.** The fraction of the AMR-R gap closed, (R0 − arm)/(R0 − A1), on the intersection. Expected: A5-ring ≥ 0.6, A5-acyc ≤ 0.25.
- **Evidence.**
  - [B-calc] floor decomposition (§2).
  - TD p. 4: ring torsions are part of L.
  - TD p. 23: weakness on puckered and fused rings.
  - Corso thesis p. 63: puckering coordinates.
  - PuckerFlow p. 9: a ring generator as a TD front end.
- **Cost.** 12 runs × 0.25 h = 3 GPU-h; ~4 CPU-h for the floors.
- **Priority.** 1.

### R2-4: B3, MMFF-matched TD (the strongest fixed-L baseline; deferred line `B3_match_mmff`) **[priority 2]**

- **Hypothesis.** Training on MMFF-relaxed matched L and testing with `--pre_mmff` beats A2 (the released model + MMFF, 0.150 → expected ~0.14). This is the "best TD without learning L" that FlexiTors has to beat on QM9.
- **Gap.** G1 baseline strength; the review's C6 train-only cell.
- **Setup.**
  - `standardize_qm9.sbatch VARIANT=mmff` (`standardize_confs.py --mmff`, `:18`), then featurize, then train 3 seeds.
  - Evaluate with `--pre_mmff` (matched) and without it (train-only cell).
- **Control.** CTRL_rematch tested with and without `--pre_mmff` (3 inference runs, new).
- **Metric.** AMR-R, COV-R@0.1. Expected: B3 + pre_mmff < CTRL_rematch + pre_mmff < CTRL_rematch.
- **Evidence.**
  - A2 floor drop 0.0993 → 0.0832 (`floor_run.log`).
  - MMFF test-mode floor 0.0915 vs 0.1095 per molecule ([B-calc]).
  - TD p. 26: OMEGA's better local structures win on QM9 ("evidently, has a better local structures").
- **Cost.** 3 × 10.7 + ~3 GPU-h of evaluation ≈ 35 GPU-h, plus ~12 CPU-h of standardization. Start it first so the cache is ready.
- **Priority.** 2.

### R2-6: Torsion-headroom checks (inference only; deferred lines C/D/E-inf) **[priority 3]**

- **Hypothesis.** The 0.047 Å per-molecule torsion headroom of R0 is partly sampling resolution: σ_min and the number of steps. If AMR does not move, the headroom is model error, and FlexiTors' torsion part should keep TD's sampler.
- **Gap.** Separating the error of the τ-model itself from the L floor.
- **Setup.** `C_steps50`, `E_inf_sigmin_0.005pi` and `D_ode_steps20` on the released checkpoint (3 seeds each), plus the same three on B1 with GT L (seed 0), where the τ-model is the only remaining error.
- **Control.** R0 and B1_gtL.
- **Metric.** AMR-R, COV-R@0.05/0.1. Expected: a small gain (≤ 0.005 Å) on RDKit L; a larger one on GT L at 0.05.
- **Evidence.** TD Table 9: steps saturate at ~20 on DRUGS (p. 27). FM-refiner Table 6: more steps help ET-Flow and MCF on QM9 at δ = 0.05 (MCF-B 20 → 100 steps: COV 62.13 → 68.90; p. 14).
- **Cost.** 12 × 0.25 = 3 GPU-h.
- **Priority.** 3.

### Deferred ablations: still worth running? (question 5)

| Deferred line | Verdict | Reason |
|---|---|---|
| B3_match_mmff | **Run** (R2-4) | Strongest fixed-L baseline; A2 shows the transfer works |
| C_steps*, D_ode*, E_inf* | **Run 3 of them** (R2-6), cheap | They separate sampler from model; essentially free |
| B4_match_de50 | Skip | B5 shows that matching-objective tweaks do not help (+0.0058 worse); B2 shows assignment barely matters |
| E_train_sigmin/sigmax | Skip | σ_max < π breaks the uniform prior; σ_min is better probed at inference (E_inf) first |
| F_layers*/F_width*/F_radius* | Skip for now | FlexiTors changes the architecture anyway, and capacity cannot move an L floor (R0 is ~72 % L-limited) |
| G_no_parity | Skip | Known failure (TD Table 8: 30.5, ≈ random) |
| H_pool_* | Skip | Resampling pools address precision/diversity, not the L floor |
| P1_released_1order | Optional (0.25 GPU-h) | Already scored from the released pickles: AMR-R 0.1826 (`all_summaries.txt`, `released/qm9_1order`) |

### Schedule (fits ~2 days on 4 GPUs; ~100 GPU-h used, ~90 GPU-h margin)

- **t = 0:**
  - Start the `mmff` standardization on CPU, and the R2-0 CPU analyses.
  - GPUs 0–2: B6a seeds 0–2.
  - GPU 3: the inference queue (R2-1 7.5 h + R2-5 3 h).
- **t ≈ 11 h:**
  - GPUs 0–2: B3 seeds 0–2 (the cache should be ready).
  - GPU 3: B6b.
- **t ≈ 22 h:** evaluations of B6/B3 (~7 GPU-h over 4 GPUs), R2-6, and the CTRL_rematch ± pre_mmff runs.
- **t ≈ 25 h:** analysis. The remaining ~20 h are margin for reruns, or for 3 B6 seeds at a second σ_L range if B6a is ambiguous.

---

## 6. Ranked shortlist (at most 6)

| Rank | ID | One line | GPU-h | What it decides |
|---|---|---|---|---|
| 1 | R2-0 | Intersection re-analysis, clean floors, DE spot check, L→τ leak diagnostic | 0 | Which round-1 claims stand; how to read the B1 oracle |
| 2 | R2-5 | GT-ring vs GT-acyclic L, with the released model and B1 in the loop | 3 | Which internal coordinates FlexiTors must generate (rings first?) |
| 3 | R2-2 | B6: GT-L training with Gaussian L-noise conditioning augmentation | ~47 | Whether a τ\|L model can be trained on GT L and survive imperfect L, i.e. whether the cascade is viable |
| 4 | R2-1 | L-quality dose-response (B1 vs CTRL; noise, MMFF-GT, RDKit + MMFF) | 7.5 | The accuracy spec ε* for FlexiTors' L generator |
| 5 | R2-4 | B3: MMFF-matched TD (± pre_mmff) | ~35 | The fixed-L baseline FlexiTors must beat |
| 6 | R2-6 | Steps / σ_min / ODE on R0 and B1_gtL | 3 | Sampler vs model share of the torsion headroom |

---

## 7. Citation table

| Claim | PDF path | Page | Exact quote |
|---|---|---|---|
| TD trains on matched L because of the train/test L shift | papers/core/2022_jing_torsional_diffusion.pdf | 7 | "if we train on the denoising score-matching loss with ground truth conformers--i.e., conditioned on ground truth local structures--there will be a distributional shift at test time" |
| The shift hurts | papers/core/2022_jing_torsional_diffusion.pdf | 7 | "We found that this shift significantly hurts performance." |
| GT-L training: lower loss, bad inference | papers/core/2022_jing_torsional_diffusion.pdf | 26 | "although the training and validation score matching loss of this model is significantly lower, its inference performance reflects the detrimental effect of the local structure distributional shift." |
| GT-L training ≈ random baseline on DRUGS | papers/core/2022_jing_torsional_diffusion.pdf | 27 | "Train on ground truth L 34.8 22.4 0.920 0.909" and "Random 30.9 13.2 0.922 0.923" (Table 8 rows) |
| Assignment barely matters (B2 analogue) | papers/core/2022_jing_torsional_diffusion.pdf | 27 | "Only D.E. matching 72.5 81.1 0.588 0.569" (Table 8 row) |
| L matters most on QM9 | papers/core/2022_jing_torsional_diffusion.pdf | 26 | "the accuracy of local structure significantly impacts the performance of torsional diffusion" |
| OMEGA's better L on QM9 | papers/core/2022_jing_torsional_diffusion.pdf | 26 | "is only on par with or slightly worse than OMEGA, which, evidently, has a better local structures for these small molecules" |
| Ring torsions are part of L in TD | papers/core/2022_jing_torsional_diffusion.pdf | 4 | "torsion angles in cycles (or rings), which cannot be rotated independently, are considered part of the local structure L." |
| TD weak on puckered and fused rings | papers/core/2022_jing_torsional_diffusion.pdf | 23 | "it is less true for puckered rings, fused rings, and larger cycles." |
| GT-other-L floor on DRUGS (G3) | papers/core/2022_jing_torsional_diffusion.pdf | 21 | "only slightly larger than the average RMSDmin of 0.284 Å resulting from matching a ground truth conformer to the local structure of another randomly chosen ground truth conformer" |
| EBD: RDKit-prior shift | papers/related/2024_park_equivariant_blurring_diffusion.pdf | 5 | "This distributional shift of the fragment structure can potentially harm the performance during the inference." |
| GeoMol predicts LS itself | papers/related/2021_ganea_geomol.pdf | 4 | "First, we predict the local 3D structure of each non-terminal atom, which we deem local structure (LS)" |
| GeoMol builds rings jointly | papers/related/2021_ganea_geomol.pdf | 16 | "when we first encounter a node that is part of a cycle of nodes X1, X2, . . . , Xn, we will jointly compute all the 3D coordinates of this cycle" |
| ET-Flow avoids internal geometry via a harmonic prior | papers/related/2024_hassan_etflow.pdf | 1 | "a well-designed flow matching approach with equivariance and harmonic prior alleviates the need for complex internal geometry calculations" |
| ET-Flow QM9 numbers | papers/related/2024_hassan_etflow.pdf | 19 | "ET-Flow (QM9 RS) 96.47 100.00 0.073 0.047" |
| QM9 at 0.5 Å saturated | papers/related/2025_gurev_s23d.pdf | 16 | "indicating that benchmark results on the QM9 metric are likely to be saturated." |
| FM-refiner uses δ = 0.05 Å on QM9 | papers/related/2025_xu_fm_refiner.pdf | 7 | "Since recent work already achieves 100% median COV at the commonly used threshold [δ] = 0.5 Å and a median AMR below 0.05 Å, we adopt the more challenging COV threshold of [δ] = 0.05 Å." ([δ]: the glyph is missing from the extracted text) |
| ET-Flow at δ = 0.05 | papers/related/2025_xu_fm_refiner.pdf | 7 | "ET-Flow (8.3M) 75.72 87.23 0.083 0.031" |
| More steps help at 0.05 on QM9 | papers/related/2025_xu_fm_refiner.pdf | 14 | "MCF-B 20 62.13 62.50 0.108 0.056" and "MCF-B 100 68.90 75.00 0.099 0.047" |
| Nikitin's bugs concern stability metrics | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 1 | "current evaluation protocols suffer from critical flaws, including incorrect valency definitions, bugs in bond order calculations" |
| Fragmented GEOM molecules | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 8 | "These failures produced fragmented molecules and unstable valencies" |
| Recommended local-geometry metrics | papers/related/2025_nikitin_geom_drugs_revisited.pdf | 11 | "we suggest to assess differences in bond lengths, bond angles, and torsion angles of generated and optimized counterparts." |
| Conditioning augmentation fixes cascade train–test mismatch | papers/related/2021_ho_cascaded_diffusion.pdf | 3 | "conditioning augmentation is effective because it alleviates compounding error in cascading pipelines due to train-test mismatch" |
| Gaussian noise is the effective augmentation | papers/related/2021_ho_cascaded_diffusion.pdf | 6 | "what we found most effective at low resolutions is adding Gaussian noise (forward process noise)" |
| Amortised augmentation strength | papers/related/2021_ho_cascaded_diffusion.pdf | 6 | "train super-resolution models amortized over the strength of conditioning augmentation and pick the best strength in a post-training hyperparameter search" |
| Input perturbation for exposure bias | papers/related/2023_ning_input_perturbation.pdf | 2 | "explicitly modelling the prediction error during training" |
| Ring puckering coordinates as the ring factor | papers/related/2023_corso_intrinsic_diffusion.pdf | 63 (PDF page; printed page not verified) | "employing the ring puckering coordinates [18] to model the flexibility of ring conformations as points on hyperspheres." |
| A ring generator as a TD front end | papers/related/2026_schaufelberger_puckerflow.pdf | 9 | "PuckerFlow- generated ring conformers serve as initial local substructures for these approaches" |

**Code citations used:**
- `tools/paired_compare.py:12-15, 64-72, 99`
- `torsional-diffusion/evaluate_confs.py:81-84, 152`
- `torsional-diffusion/diffusion/sampling.py:54-63, 75`
- `torsional-diffusion/utils/dataset.py:24-43, 200-239`
- `torsional-diffusion/standardize_confs.py:18, 58-67, 74-76, 79-82`
- `tools/local_structure_analysis.py:1-37, 196-214, 293-298`

**Result files used:**
- `cluster_sync/results/analysis/paired_*.md`, `all_summaries.txt`, `local_structure_{test,test_mmff,std}.{log,csv}`
- `cluster_sync/results/qm9_default/*/floor_run.log`, `R0_base_seed0/breakdown.log`
- `RESULTS_QM9.md`, `review/*.md`, `notes/ablation_plan_qm9.md`, `slurm/ablations_*_deferred.tsv`
