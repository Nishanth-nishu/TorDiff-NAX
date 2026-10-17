# Evidence ledger: R-JOINT (round 5)

Lens: joint local-geometry + torsional diffusion (Exp 1) and a full-molecule Boltzmann generator p(L, tau) (Exp 4).
Agent tag JOINT. Template: `orchestration/templates/evidence_entry.md`.
PDF pages are pdftotext page indices of the stored PDF counted by form feeds (= PDF page); printed page given where
it differs. Entries marked "Reused from" were copied from a round-3/4 ledger and **every quote was re-checked against
the stored PDF or data file in round 5** (the re-check is noted in the entry).

Sections:
- A. Our data (the leak-free result and the round-2 contrasts the design rests on)
- B. Our code (TD touchpoints for Exp 1 and Exp 4)
- C. Torsional Diffusion: the equations Exp 1 and Exp 4 extend
- D. Prior art on joint internal-coordinate / angle+torsion generation
- E. Prior art on multi-scale / hierarchical / coarse-to-fine conformer generation (2023-2026)
- F. Boltzmann generators and likelihood in internal coordinates
- G. Reconstruction, ring closure, stereochemistry
- H. Evaluation protocol sources
- I. Search log and novelty verdicts

(skeleton written first; entries appended below)

---

## A. Our data

### E-JOINT-001
- Claim: In the leak-free split-half oracle test, **another conformer's true local structure makes every model worse
  than RDKit local structure**: seed-averaged AMR-R on the split-half view is CR 0.2195, B1 0.3046, S3 0.2797,
  S4 0.2528 Å, against the same checkpoints' full-set RDKit-L reference of CR 0.1873, B1 0.2565, S3 0.1913,
  S4 0.1890 Å; the model-free copy-the-other-half baseline is 0.3336 Å. With each target's *own* true L the same
  models score CR 0.1603, B1 0.0303, S3 0.0525, S4 0.0780 Å.
- Source type: our-data
- Locator: our-data: `cluster_sync/round4/analysis_r4_leakfree/leakfree_summary.txt`, section "## Per model
  (seed-averaged; AMR-R in A, COV-R as fraction)", views `split`, `own`, `ref_rdkitL`, column AMR-R.
- Exact quote (CR and B1 rows, truncated to the AMR-R columns of split / own / ... / ref_rdkitL / ref_gtLcycle_full):
  > CR      0.2195    0.8564    0.5019  0.1603 ... 0.1873    0.9447    0.3438            0.1026
  > B1      0.3046    0.7373    0.4002  0.0303 ... 0.2565    0.8670    0.2297            0.0269
- Shows vs infers: SHOWS. ORACLE (test-set true geometry): analysis only, never model selection
  (`orchestration/README.md` rule 5). INFERENCE, and the design premise of Exp 1: the own-conformer gains are
  conformer-identity leakage, and a *better but not conformer-specific* L is not enough — geometry only helps when it
  is generated jointly with (coupled to) the torsions of the conformer being produced. This is the argument against an
  independent geometry predictor p(L) feeding a frozen torsion model, and for V2/V3 of Exp 1.
- Supports card(s): EXP-1 (all variants), EXP-4
- Confidence: high (numbers); high (the "coupled" conclusion, because the split view is a paired within-molecule
  measurement over the same targets with only the seed half changed, `round4/leakfree.md` §2)
- **Revised after D-001 and D-003 (V5).** The numbers stand. Two parts of the inference above are withdrawn.
  - (a) "Worse than RDKit L" is **not budget-matched**. The split view generates 2·|half| conformers per molecule
    (`tools/make_split_half.py:19-22`), while `ref_rdkitL` generates 2·n (`tools/analyze_leakfree.py:318-347`).
    AMR-R is a minimum over samples, so halving K raises it by itself. The split − ref_rdkitL gaps (CR +0.032,
    B1 +0.048, S3 +0.088, S4 +0.064 Å) are therefore UNVERIFIED until the budget-matched CPU re-score requested in
    D-001 is run. Only the paired own-vs-split leak contrast is budget-matched, and it stands.
  - (b) "Geometry only helps when coupled" is **withdrawn as an inference from this test**. With E-JOINT-010
    corrected, the failure is fully explained by a train/test L-distribution mismatch in the torsion models. The test
    gives no evidence that coupling is *necessary*. Coupling is now a **hypothesis**, tested by arm A6b (EXP-1 §4).
  - Confidence of the coupling conclusion: high → **none (hypothesis)**.
- **Revision 2, after the budget-matched re-score (2026-10-17, E-JOINT-011).** The comparative claim "another
  conformer's true L makes every model worse than RDKit L" is **withdrawn: it was a budget artefact**. At matched
  budget, other-conformer GT L is *better* for CR (−0.038 Å overall, −0.015 Å on distinct targets). It is worse for
  B1, S3 and S4 only on distinct targets (+0.047 to +0.085 Å) and on COV-R@0.5. The VERIFIED numbers above stay as
  recorded; only their inference changes.

### E-JOINT-002
- Claim: The split-half test's population: 744 of 1000 QM9 test molecules have ≥ 2 clean GT conformers, 743 are in the
  success intersection over 11 runs × 4 conditions, and the distinct-target universe (≥ 1 target whose nearest
  other-half conformer is ≥ 0.1 Å away) is 626 molecules / 7516 of 12,628 targets. About 40 % of eligible GT
  conformers have an other-half conformer within 0.1 Å, so the split view is a *conservative* (leak-reduced, not
  leak-zero) measurement and the distinct-target view `split_d` is the strict one (CR 0.2670, B1 0.4166, S3 0.3733,
  S4 0.3321, COPY 0.4656 Å).
- Source type: our-data
- Locator: our-data: `cluster_sync/round4/analysis_r4_leakfree/leakfree_summary.txt` header lines and the `split_d`
  columns; method and counts in `round4/leakfree.md` §2–§3.
- Exact quote (summary header):
  > eligible molecules (split.tsv, n_clean >= 2): 744
  > distinct-target universe (>= 1 target with other-half nn_rmsd >= 0.1 A): 626
- Shows vs infers: SHOWS. The "conservative" reading is a pre-declared caveat in `round4/leakfree.md` §3.
- Supports card(s): EXP-1 (evaluation), EXP-4
- Confidence: high

### E-JOINT-003
- Claim: With the model in the loop, ring geometry is the lever and acyclic geometry is not: on the matched-RDKit base
  (λ = 0, 955 molecules) true ring geometry alone takes CTRL_rematch from 0.1966 to 0.1186 Å AMR-R, while true
  acyclic geometry alone gives 0.1824 Å (no gain). Both are own-conformer ORACLE and must now be read as upper bounds
  that include the leakage of E-JOINT-001.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows
  `S1.B1_vs_CR.S1_lam0.00_cyc_ORACLE`, `S1.B1_vs_CR.S1_A5ring_cyc_ORACLE`, `S1.B1_vs_CR.S1_A5acyc_cyc_ORACLE`
  (`ref_value` column = CTRL_rematch).
- Exact quote (A5ring row, truncated):
  > S1.B1_vs_CR.S1_A5ring_cyc_ORACLE,...,955,B1,S1_A5ring_cyc_ORACLE,CR,S1_A5ring_cyc_ORACLE,,0.1877...,0.1186...
- Shows vs infers: SHOWS. Reused from E-ARCH-001 / E-PRIOR-050 / -051 (VERIFIED in round 3 as E-GEOM-013);
  **re-checked in round 5 against the contrasts file.**
- Supports card(s): EXP-1 (why the correction target is rings, not bonds/angles in general)
- Confidence: high (numbers); medium (magnitude, because of leakage)

### E-JOINT-004
- Claim: Ring-pucker error is a tail, not a uniform offset: among seeds of molecules with a ring of ≥ 4 atoms the
  share with endocyclic ring-dihedral RMSD > 10° is ETKDG 49.6 %, MMFF 35.1 %, λ = 0.50 19.4 %, λ = 0.75 0.02 %;
  mean per-seed ETKDG ring errors are 0.0318 Å (bonds), 2.38° (angles) and 10.65° (ring dihedral, median 0.44°).
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, columns `bond_rmsd_heavy_ring`,
  `angle_rmsd_heavy_ring`, `ring_dihedral_rmsd`, grouped by `cond`.
- Exact quote (file header line):
  > cond,smiles,k,src_gt_idx,bond_rmsd_all_ring,angle_rmsd_all_ring,...,angle_rmsd_heavy_any,ring_dihedral_rmsd
- Shows vs infers: SHOWS. Reused from E-ARCH-002 (round-3 E-GEOM-001/-002, VERIFIED, V3); the header line was
  re-checked in round 5. INFERENCE: a correction module must be able to make *large, discrete* pucker moves on a
  minority of seeds, not small smooth shifts everywhere — which is why a unimodal regression head is the wrong
  parameterisation and a diffusion/flow head is the right one.
- Supports card(s): EXP-1
- Confidence: high
- **Revised after D-002 (V5).** The pooled mean of 10.65° and median of 0.44° include 3,078 seeds whose largest ring
  is 3-membered; those always score 0.0°. I recomputed on the stored file (`l_error.csv`, cond `L_etkdg2L`,
  largest ring from RDKit ring perception of the SMILES):

  | Population | Seeds | Mean | Median | > 10° |
  |---|---|---|---|---|
  | Largest ring ≥ 4 | 4,181 | 18.50° | 9.79° | 49.6 % |
  | Largest ring = 3 | 3,078 | 0.00° | — | 0 % |

  For rings of four or more atoms the error is therefore **the bulk, not a minority tail**: half the seeds exceed
  10°. The inference is restated: the geometry factor must make large pucker moves on about half of all ≥ 4-ring
  seeds. That still needs a multimodal (diffusion/flow) head, now for the bulk of those seeds. Every ring-tail
  threshold in EXP-1 is defined on the ≥ 4-ring population, which is the population of the 49.6 % baseline.

### E-JOINT-005
- Claim: Models trained on true or mixed local structure are *brittle to better geometry*: against CTRL_rematch on
  MMFF-relaxed ETKDG L (CTRL 0.1525 Å), B1 is +0.0798 Å [+0.0722, +0.0877] and S3 is +0.0394 Å [+0.0338, +0.0454]
  worse; S3 on plain RDKit L is +0.0032 Å [+0.0013, +0.0052] worse (Holm p = 0.0096).
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows `S1.B1_vs_CR.S1_etkdg2L_mmff`,
  `S1.S3_vs_CR.S1_etkdg2L_mmff`, `S3.rdkit_vs_CR`; summarised in `round4/BRIEF.md`.
- Exact quote (round4/BRIEF.md):
  > S3 (50/50 mix) on RDKit L vs CTRL_rematch: +0.0032 [+0.0013, +0.0052], Holm p = 0.0096 — significantly WORSE
- Shows vs infers: SHOWS. Reused from E-ARCH-005 / E-PRIOR-055; **re-checked in round 5 against `round4/BRIEF.md`.**
  INFERENCE: in a joint model the torsion factor must be trained on the geometry the geometry factor actually produces
  (the DAgger / sampler-matched rule), or it lands in this regime.
- Supports card(s): EXP-1 (training protocol, controls)
- Confidence: high

### E-JOINT-006
- Claim: The only learned recipe with a real-use gain so far is training on the deployed L source: S5 (trained on
  MMFF-relaxed matched seeds, tested on MMFF seeds) beats CTRL_rematch tested on MMFF seeds by
  −0.0041 Å [−0.0064, −0.0019] (0.1500 vs 0.1540); the same model without MMFF at test is +0.0031 worse.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows `S5.premmff_vs_CRpremmff`,
  `S5.premmff_vs_CR`, `S5.nommff_vs_CR`.
- Exact quote (round4/BRIEF.md):
  > S5 (MMFF-matched training, tested with MMFF): −0.0041 [−0.0064, −0.0019] vs CTRL_rematch with MMFF at test
- Shows vs infers: SHOWS. Reused from E-ARCH-006 / E-PRIOR-057; **re-checked in round 5 against `round4/BRIEF.md`.**
- Supports card(s): EXP-1
- Confidence: high

### E-JOINT-007
- Claim: Rigid molecules (n_rot_heavy = 0; 289 of the evaluated test molecules, ≈ 30 %) are a model-free read-out of
  L quality: CTRL_rematch's rigid-bin AMR-R is ETKDG 0.152, MMFF 0.128, λ = 0 0.184 and true-ring-geometry-only
  0.043 Å, and CTRL and B1 agree there within ≈ 0.006 Å.
- Source type: our-data
- Locator: our-data:
  `cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s0/S1_A5ring_cyc_ORACLE/breakdown.log`, table
  "by n_rot_heavy", bin 0, column `MAT_R` (and the `S1_etkdg2L` / `S1_lam0.00_cyc_ORACLE` files of the same run).
- Exact quote (CTRL s0, S1_A5ring_cyc_ORACLE/breakdown.log):
  > | 0     | 289 | 100     |   0.043 |  99.913 |   0.053 |
- Shows vs infers: SHOWS. Reused from E-ARCH-004 (round-3 E-GEOM-004, VERIFIED). INFERENCE: this stratum is the
  cheapest Gate-0 read-out for the Exp-1 geometry factor, because no torsion model is involved.
- Supports card(s): EXP-1 (Gate 0, milestone 1)
- Confidence: high

### E-JOINT-008
- Claim: Measured compute on gnode118: one 100-epoch QM9 TD training takes ≈ 11.7 GPU-h on one RTX 3090 (6 CPU
  workers); packed inference of the 1000-molecule QM9 test set takes 15–25 min; one GFN2-xTB optimisation of a QM9
  seed takes ≈ 0.14 s median (≤ 0.4 s) on one core with xtb 6.7.1.
- Source type: our-data
- Locator: our-data: `round3/grounding.md:35-37` ("Cost basis"); `round4/BRIEF.md` §Constraints.
- Exact quote (round4/BRIEF.md):
  > Compute: gnode118, 4× RTX 3090, ~11.7 h per 100-epoch QM9 training on 6 CPUs; packed inference 15–25 min.
- Shows vs infers: SHOWS (grounder measurements). Reused from E-ARCH-063; **re-checked against `round4/BRIEF.md`.**
- Supports card(s): EXP-1 (compute), EXP-4 (feasibility)
- Confidence: medium (not re-measured this round)

### E-JOINT-009
- Claim: Our best real-use QM9 AMR-R is 0.150 Å (S5 with MMFF seeds); TD's own is 0.176–0.178 Å; the published
  Cartesian frontier on the same TD-protocol split is 0.068–0.103 Å (DiTMC 0.068–0.073, ET-Flow 0.073, DMT-B 0.090,
  MCF-B 0.103).
- Source type: our-data + paper
- Locator: our-data: `RESULTS_QM9.md`; paper: `papers/related/2025_frank_ditmc.pdf`, PDF p. 6, Table 1.
- Exact quote (DiTMC Table 1, **row reassembled from the column-shifted extraction**; in the raw text these numbers
  sit next to the "DMT-B (55M)" label. Revised after D-008):
  > Tors. Diff. (1.6M) 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195
- Shows vs infers: SHOWS. Reused from E-ARCH-052 / E-PRIOR-034 / -039. INFERENCE: Exp 1 cannot be framed as "beats
  SOTA on QM9"; its claim must be a *mechanism* claim (joint L–τ) with an internal-geometry and energy evaluation,
  plus the leak-free analysis.
- Supports card(s): EXP-1 (framing), EXP-4
- Confidence: high

### E-JOINT-010
- Claim: Without a model, another GT conformer's local structure is geometrically almost as good as the target's own:
  the torsion-optimised (conformer-matching) floor for ring molecules is 0.1171 Å per conformer with RDKit L, 0.0989
  with GT acyclic angles and bonds, and **0.0323 Å with another GT conformer's full L**; for acyclic molecules 0.0510
  → 0.0081 / 0.0137.
- Source type: our-data
- Locator: our-data: `cluster_sync/results/analysis/local_structure_test.csv` (7554 GT conformers); strata table
  `round2/research_A.md:24-34`, recomputed and VERIFIED in `round2/verify_A.md:79-82`.
- Exact quote (round2/research_A.md:28):
  > | any ring (5444) | 0.1171 | 0.1072 | 0.0989 | 0.0323 | 3.76 |
- Shows vs infers: SHOWS (floors from torsion optimisation against the target, no generative model). Reused from
  E-ARCH-010. INFERENCE, and the **necessary qualification of E-JOINT-001**: other-conformer L leaves only a 0.032 Å
  geometric floor, yet the trained models score 0.22–0.30 Å with it. So the leak-free failure is a failure of the
  *torsion models* to use L they were not trained with, not a property of the geometry. An independent L sampler is
  therefore not ruled out *geometrically*; it is ruled out *for torsion models trained on a different L
  distribution*. Coupling (V2/V3) is one way to remove that mismatch; DAgger-style training on the sampler's own L is
  the other, and the decoupled-ablation control (EXP-1 §4, arm A6) is what separates them.
- Supports card(s): EXP-1 §0, §2, §4
- Confidence: high (floor values); the inference is the spec's central caveat
- **Revised after D-003 (V5); the history above is kept.**
  - **The 0.0323 / 0.0137 / 0.0269 Å values are wrong, biased low by a code bug** (E-JOINT-031). At
    `tools/local_structure_analysis.py:302`, `np.argsort(cost_gg[i])[:args.floor_topm]` with `floor_topm = 3` (:74)
    includes the target itself whenever n_GT ≤ 3. I checked this independently in round 5 on
    `cluster_sync/results/analysis/local_structure_test.csv`: all 769 rows with `n_gt ≤ 3` have
    floor_gtother ≤ 2.0e-5 Å (mean 8.4e-8).
  - **Corrected values**, on the rows the bug does not touch (`n_gt > 3`; my recomputation agrees with V5):

    | Stratum | Conformers (molecules) | floor_gtother mean | floor_gtother median | floor_best_sym (RDKit L), same rows |
    |---|---|---|---|---|
    | Ring | 4,696 (391) | **0.0359 Å** | 0.0165 Å | **0.1143 Å** |
    | Acyclic | 2,089 (97) | 0.0138 Å | — | 0.0507 Å |
    | All | 6,785 (488) | 0.0291 Å | — | 0.0947 Å |

    These rows are molecules with ≥ 4 conformers, a different population from the full set.
  - **Corrected definition.** floor_gtother is an **oracle best-of-3**: differential evolution over the 3 GT
    conformers *closest to the target by transplant cost*, chosen from all of its other GT conformers. It is not
    "another conformer's L" and not an analogue of the split-half test (other half only, and ≈ 40 % of targets have a
    near-duplicate, E-JOINT-002).
  - **Corrected inference.** The direction survives: L taken from the target's closest other GT conformers is
    geometrically much better than RDKit L (0.036 vs 0.114 Å on the same rows). It follows that the leak-free failure
    is a **train/test L-distribution mismatch in the torsion models**. That is the S3/B1 brittleness of E-JOINT-005,
    and it is *not* evidence that L must be coupled to τ. This supersedes the coupling inference in E-JOINT-001; see
    its revision note.
  - Requested by V5 and not yet run: a fixed-code re-run for n_GT ≤ 3, and an other-half-only variant matching the
    split-half design. These are listed in EXP-1 §9 "Known issues".

### E-JOINT-011
- Claim: **Budget-matched leak-free re-score** (ORACLE; 711 molecules, 606 distinct-target; ref_half = exact
  expectation over 2|H|-subsets of the full RDKit-L run; BUDGET Holm family of 8). split − ref_half AMR-R:
  - CR −0.0384 [−0.0482, −0.0289] (Holm p < 0.001);
  - B1 −0.0170 [−0.0310, −0.0030] (Holm p 0.060);
  - S3 +0.0171 [+0.0048, +0.0296] (Holm p 0.021);
  - S4 −0.0069 [−0.0178, +0.0041] (n.s.).

  COV-R@0.5: CR +0.07 pt (n.s.), B1 −4.41, S3 −8.70, S4 −5.34 pt. On distinct targets (split_d − ref_half_d AMR-R):
  CR −0.0147 [−0.0269, −0.0026], B1 +0.0666, S3 +0.0851, S4 +0.0469. COV-R@0.1 split − ref_half is +19.7 to +25.2 pt
  for every model.
- Source type: our-data
- Locator: our-data: `cluster_sync/round5/rescore/budget_matched_summary.txt` ("## Contrasts" table);
  `budget_contrasts.csv`; `round5/IMPLEMENTATION.md` milestone 1 §5. Code: `tools/analyze_leakfree.py:426-505`
  (exact expectation, cover probability, ref_half per molecule), `:523` (`budget_matched`).
- Exact quote (budget_matched_summary.txt):
  > B.CR.split-ref_half     AMR-R 711      budget     0.2214     0.2598  -0.0384  -0.0482  -0.0289  0.0000  -0.0482  -0.0284  0.0000
  > B.CR.split_d-ref_half_d     AMR-R 606   secondary     0.2666     0.2813  -0.0147  -0.0269  -0.0026  0.0184  -0.0270  -0.0024     NaN
- Shows vs infers: SHOWS. R-JOINT checked the formulas and the exclusions in code (EXP-1 §11). INFERENCE:
  - (a) better-but-not-own L helps the RDKit-trained TD model modestly, so a decoupled learned-L arm has a
    distinct-target oracle ceiling of ≈ −0.015 Å;
  - (b) GT- or blend-trained torsion models are identity-dependent (worse on distinct targets);
  - (c) nothing here tests coupling.
- Supports card(s): EXP-1 §0, §3.6, §4 (A6), §7, §11
- Confidence: high (numbers and code check); medium (distinct-target view, threshold 0.1 Å, same-ensemble targets)

---

## B. Our code (TD touchpoints)

### E-JOINT-020
- Claim: TD's score network ends in a bond-centred tensor-product layer whose output irreps are pseudoscalars (`0o`)
  when parity is on, reduced by a 2-layer MLP to one real number per rotatable bond; the only generated quantity is
  `data.edge_pred`, scaled by sqrt(score_norm(σ)).
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/score_model.py:119-132`, `:154-164`
- Exact quote:
  > out_irreps=f'{ns}x0o' if parity else f'{ns}x0e',  # [ablation-hooks] 0e = parity-invariant ablation
  > out = self.bond_conv(node_attr, edge_index, edge_attr, edge_sh, out_nodes=data.edge_mask.sum(), reduce='mean')
  > data.edge_pred = data.edge_pred * torch.sqrt(score_norm)
- Shows vs infers: SHOWS. Reused from E-ARCH-020 (VERIFIED, round 4); **re-read in round 5, line numbers confirmed.**
  INFERENCE: an angle/length head must be **parity-even (`0e`)** because bond lengths and bond angles are true scalars
  (invariant under inversion), unlike torsions and ring-pucker amplitudes/phases, which are pseudoscalars (`0o`).
- Supports card(s): EXP-1 §2.4
- Confidence: high

### E-JOINT-021
- Claim: With `use_second_order_repr=True` the atom convolution layers carry vector (`1o`) and higher irreps, so
  per-atom equivariant vector outputs can be read from the last atom layer without changing the trunk.
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/score_model.py:83-89`, `:141-143`
- Exact quote:
  > f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
- Shows vs infers: SHOWS. Reused from E-ARCH-021 (VERIFIED); **re-read in round 5.**
- Supports card(s): EXP-1 §2.2, §2.4
- Confidence: high

### E-JOINT-022
- Claim: TD's training transform perturbs torsions only: one σ per sample drawn log-uniformly in [σ_min, σ_max],
  an i.i.d. normal torsion update per rotatable bond applied by `modify_conformer`, the update stored as the
  regression target `edge_rotate`, and a DSM loss divided by the expected score norm.
- Source type: code
- Locator: code: `torsional-diffusion/utils/dataset.py:71-76`; loss `torsional-diffusion/utils/training.py:25-31`
- Exact quote:
  > sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
  > torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
  > loss = ((score - pred) ** 2 / score_norm).mean()
- Shows vs infers: SHOWS. Reused from E-ARCH-022 (VERIFIED); **re-read in round 5.**
- Supports card(s): EXP-1 §2.5, §5
- Confidence: high

### E-JOINT-023
- Claim: The round-2 code already provides the three pieces a coupled geometry factor needs: (i) a paired cache with,
  per training conformer k, the matched RDKit L in `data.pos[k]` and its own Kabsch-aligned, terminal-relabelled GT L
  in `data.gt_pos[k]`; (ii) an aligned interpolant `interp_x_torch(x, y, lam, edge_index)` that restores terminal
  (degree-1) atoms onto the interpolated bond length; (iii) a λ embedding concatenated to the σ embedding on both
  node and edge features.
- Source type: code
- Locator: code: `torsional-diffusion/utils/dataset.py:18-34`, `:79-99`;
  `torsional-diffusion/diffusion/score_model.py:65-79`, `:194-199`
- Exact quote:
  > x, y = data.pos[k], data.gt_pos[k]
  > data.pos = interp_x_torch(x, y, lam, data.edge_index)
  > node_lam_emb = get_timestep_embedding(data.node_lambda * 10000, self.lambda_embed_dim)
- Shows vs infers: SHOWS. Reused from E-ARCH-023 (VERIFIED); **re-read in round 5.**
- Supports card(s): EXP-1 §2.2, §5
- Confidence: high

### E-JOINT-024
- Claim: TD's sampler integrates only torsions: a geometric σ schedule of `steps+1` points, per step one model
  evaluation, g(σ) = σ·sqrt(2 ln(σ_max/σ_min)), the reverse-SDE increment `g²·eps·score + g·sqrt(eps)·z` (ODE variant
  `0.5·g²·eps·score`), applied by rotating one side of each rotatable bond; L never changes after seeding.
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/sampling.py:125`, `:184-204`, `:234`, `:290-298`
- Exact quote:
  > sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
  > perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z
  > conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
- Shows vs infers: SHOWS. Reused from E-ARCH-024 (VERIFIED); **re-read in round 5, line numbers confirmed.**
- Supports card(s): EXP-1 §2.1, §2.5, §5
- Confidence: high

### E-JOINT-025
- Claim: A bond counts as rotatable in TD only if removing it disconnects the molecular graph with ≥ 2 atoms on the
  smaller side, so **every ring bond is excluded from the torsion space** and ring geometry is fixed by the seed;
  there is no bond-order test, so non-ring double bonds *are* treated as rotatable.
- Source type: code
- Locator: code: `torsional-diffusion/utils/torsion.py:16-31`; the double-bond consequence is stated in
  `tools/stereo_check.py:1-11`.
- Exact quote:
  > G2.remove_edge(*edges[i])
  > if not nx.is_connected(G2):
  > non-ring double bonds count as "rotatable" (get_transformation_mask has no bond-order check), so the model CAN flip E/Z
- Shows vs infers: SHOWS. Reused from E-ARCH-025 (VERIFIED) and extended in round 5 with the E/Z consequence read
  from our own `tools/stereo_check.py` docstring.
- Supports card(s): EXP-1 §2.7, §2.8
- Confidence: high

### E-JOINT-026
- Claim: Torsion updates are applied as **rigid rotations of one side of the bond** about the bond axis
  (`R.from_rotvec`), so bond lengths and bond angles are mathematically invariant under TD sampling and tetrahedral
  chirality can never change; only dihedrals move.
- Source type: code
- Locator: code: `torsional-diffusion/utils/torsion.py:57-75`
- Exact quote:
  > rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
  > pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: this is exactly the invariance V2/V3 give up, so the stereo and
  bond/angle-validity checks that TD never needed become mandatory for Exp 1.
- Supports card(s): EXP-1 §2.6, §2.8, §3
- Confidence: high

### E-JOINT-027
- Claim: Any external L source plugs into TD without model changes through `--seed_confs` (a pickle of RDKit mols
  keyed by SMILES); with `seed_confs_cycle` conformer i takes the local structure of seed `i mod len(seeds[smi])`,
  otherwise a random seed is chosen.
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/sampling.py:84-91`; `torsional-diffusion/generate_confs.py` CLI.
- Exact quote:
  > seed_mol = seed_confs[smi][i % len(seed_confs[smi])] if seed_confs_cycle else random.choice(seed_confs[smi])
- Shows vs infers: SHOWS. Reused from E-ARCH-026 (VERIFIED); **re-read in round 5.** INFERENCE: this is the hook the
  V1 baseline and every "L from an external source" control uses, including the leak-free oracle.
- Supports card(s): EXP-1 §3, §4
- Confidence: high

### E-JOINT-028
- Claim: TD's likelihood code already computes, per sampled conformer, (i) the torsional `dlogp` accumulated along the
  probability-flow ODE, (ii) the exact divergence by m+1 extra network evaluations (`divergence_full`), (iii) the
  inertia tensor, (iv) `log_det_jac` as the sum of log singular values of the m × 3n Jacobian of the torsion
  coordinates projected to remove translation and rotation, and (v) the Euclidean log-density correction
  `euclidean_dlogp = dlogp − 0.5 log|det I| − log_det_jac`.
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/likelihood.py:24-39` (full divergence), `:59-98` (inertia tensor,
  dx/dτ, log_det_jac), `:114-122` (`populate_likelihood`); ODE accumulation
  `torsional-diffusion/diffusion/sampling.py:198-202`.
- Exact quote:
  > mol.euclidean_dlogp = mol.dlogp - 0.5 * np.log(np.abs(np.linalg.det(mol.inertia_tensor))) - mol.log_det_jac
  > div += (data_gpu.edge_pred[i::n_bonds].cpu().numpy() - score[i::n_bonds]) / eps
  > dlogp += -0.5 * g ** 2 * eps * div
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: this is the exact insertion point for Exp 4 — the term that
  must be added is the log-density of the L factor plus the Jacobian of the extra coordinates.
- Supports card(s): EXP-4 §3
- Confidence: high

### E-JOINT-029
- Claim: TD's Boltzmann machinery is complete and runnable: `BoltzmannResampler.resample` draws K samples with the ODE
  sampler and `likelihood`, computes MMFF94s energies, forms log-weights `−E/kT − euclidean_dlogp`, and reports
  `ess = 1 / Σ w²`; `BaselineResampler` is the annealed-importance-sampling baseline with Metropolis–Hastings torsion
  proposals; `test_boltzmann.py` restricts the test set to molecules with 3–7 rotatable bonds and uses the first
  conformer's local structure only.
- Source type: code
- Locator: code: `torsional-diffusion/utils/boltzmann.py:22-53` (resampler, ESS), `:56-125` (AIS baseline);
  `torsional-diffusion/test_boltzmann.py:30-37`; `torsional-diffusion/utils/dataset.py:314-316`.
- Exact quote:
  > logweights.append(-energy / kT - mol.euclidean_dlogp)
  > data.ess = 1 / np.sum(data.weights ** 2)
  > if data.mask_rotate.shape[0] < 3 or data.mask_rotate.shape[0] > 7: continue
  > # torsional Boltzmann generator uses only the local structure of the first conformer
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: the code comment at `dataset.py:315` is the exact statement of
  the limitation Exp 4 removes — the existing generator conditions on **one fixed L** and therefore samples
  p(τ | L₀), not p(x).
- Supports card(s): EXP-4 §0, §3, §4
- Confidence: high

### E-JOINT-030
- Claim: Our repo already has the two evaluation tools Exp 1 needs beyond COV/AMR: `tools/geometry_metrics.py`
  (per-matched-pair heavy-atom bond-length MAE in Å, bond-angle MAE in degrees and heavy rotatable-bond torsion MAE,
  using the same heavy-atom graph isomorphism GetBestRMS uses) and `tools/stereo_check.py` (chirality / E–Z
  consistency of generated conformers against ground truth, via `AssignStereochemistryFrom3D` and canonical isomeric
  SMILES).
- Source type: code
- Locator: code: `tools/geometry_metrics.py:1-21` (docstring and metric definition); `tools/stereo_check.py:1-14`,
  `:31-41`.
- Exact quote:
  > recall side   : each GT conformer l  <-> generated conformer k* = argmin_k RMSD[l, k]
  > With torsional diffusion the generated bond lengths/angles are EXACTLY those of the ETKDG seed (rigid torsion
  > updates), so the angle MAE here is the local-structure error that a FlexiTors angle factor must reduce.
- Shows vs infers: SHOWS. NEW in round 5.
- Supports card(s): EXP-1 §3
- Confidence: high

### E-JOINT-031
- Claim: **Known bug.** In `tools/local_structure_analysis.py`, the other-GT-conformer floor takes the 3 smallest
  transplant costs from `cost_gg[i]`, a row that contains the target i itself. The self cost is `np.inf`, which only
  sorts it last, so whenever n_GT ≤ 3 the target conformer is one of the "others" and the floor is ≈ 0.
- Source type: code (+ our-data effect)
- Locator: code: `tools/local_structure_analysis.py:302` (loop) and `:74` (`--floor_topm` default 3); effect in
  `cluster_sync/results/analysis/local_structure_test.csv` (769 rows with `n_gt ≤ 3`: max floor_gtother 1.97e-5 Å).
- Exact quote:
  > for jj in np.argsort(cost_gg[i])[:args.floor_topm]:
- Shows vs infers: SHOWS (code read and effect recomputed in round 5). Found by V5 (D-003). It affects
  `round2/research_A.md:24-34` and every entry that reused its `floor_gtother` column (E-ARCH-010, E-JOINT-010,
  E-CHEM-065). Fix: exclude i before slicing. **Not edited by R-JOINT** (research agents do not change code); it is
  listed in EXP-1 §9.
- Supports card(s): EXP-1 §9
- Confidence: high

### E-JOINT-032
- Claim: The L-subspace projector P_L = I − Bᵀ(BBᵀ)⁺B with B = [J_τ; J_rig] (EXP-1 §2.2, first version) removes the
  torsion **tangent** directions ∂x/∂τ_i. It does **not** annihilate the dihedral **gradients** ∇_x τ_i, which are the
  dual (Wilson-B) rows, so a projected step still changes the rotatable dihedrals at first order. V5 measured
  0.245 rad/Å (`CCC1CCCC1CO`) and 0.185 rad/Å (`CC(C)C1=CC=CC=C1CCO`), the same as an unprojected step, and 0.000 rad/Å
  for the orthogonal projector onto ker W, where W stacks one dihedral gradient per rotatable bond. V5 also found
  chirality flips in 1 of 20 random 0.3 Å-RMS projected steps.
- Source type: our-data (verifier recomputation; V5 scratch `projector_test.py`, quoted in `round5/ledger/disputes.md`
  D-004)
- Locator: our-data: `round5/ledger/disputes.md` D-004; `round5/ledger/verify_V5.md` Summary row "EXP-1 projector
  math".
- Exact quote (disputes.md D-004):
  > first-order drift is 0.245 rad/Å (`CCC1CCCC1CO`) and 0.185 rad/Å (`CC(C)C1=CC=CC=C1CCO`);
- Shows vs infers: SHOWS (4 molecules). It is also correct analytically: the change in τ_i is ∇τ_i·δx, and
  span{∂x/∂τ_j} ≠ span{∇τ_j} in general. **EXP-1 claim (1) "changes no torsion to first order" is withdrawn.** It is
  replaced by the ker-W projector P_W = I − Cᵀ(CCᵀ)⁺C with C = [W; J_rig] (EXP-1 §2.2, revised). One dihedral is
  fixed per rotatable bond: TD's own quadruple rule (n0[0], b, c, n1[0]) from
  `torsional-diffusion/utils/standardization.py:69-90` (`get_torsion_angles`). Other quadruples around the same bond
  differ from the fixed one by bond-angle or improper terms, which belong to L, so they may change.
- Supports card(s): EXP-1 §2.2, §2.7, §2.8, §5.3
- Confidence: high

### E-JOINT-033
- Claim: The round-2 "λ-blend defect" (an unsafe λ ∈ (0,1) pair replaced by another safe pair of the same molecule)
  lives in the **test-time λ seed pickles** (`tools/make_l_seed_pickles.py`), **not** in the training paired cache. The
  training cache (`tools/build_paired_pickles.py`) attaches each matched conformer's own GT conformer by `geom_id`,
  verifies it against the stored DE RMSD, and keeps-and-flags `pair_ok` failures rather than replacing them. TD's
  standardisation also excludes molecules with no rotatable bond from training.
- Source type: code
- Locator: code: `tools/make_l_seed_pickles.py:158`; `tools/build_paired_pickles.py:20`;
  `torsional-diffusion/standardize_confs.py:76`.
- Exact quote:
  > whose pair is unsafe gets the x_lambda of one of that molecule's SAFE pairs instead (cycling over them, so
  > conf['pair_reason']. rd_mol is untouched. Only verification failures (steps 1-3) are dropped; pair_ok failures are kept
  > if not rotable_bonds: return log_error("no_rotable_bonds")
- Shows vs infers: SHOWS. NEW (after Q-P1-002 / Q-P3-001). It corrects EXP-1 §7 risk 2: the training cache
  `cache_paired` can be reused.
- Supports card(s): EXP-1 §2.2, §7
- Confidence: high

---

## C. Torsional Diffusion: the equations Exp 1 and Exp 4 extend

### E-JOINT-040
- Claim: TD defines the local structure L to include ring torsions, takes approximate samples of p_G(L) from RDKit and
  learns only p_G(τ | L).
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 4 (printed 4), §3.
- Exact quote:
  > Thus, torsion angles in cycles (or rings), which cannot be rotated independently, are considered part of the local structure L.
- Shows vs infers: SHOWS. Reused from E-ARCH-030 (VERIFIED); **re-checked in round 5 against the stored fulltext.**
- Supports card(s): EXP-1 §1, §2.1; EXP-4 §2
- Confidence: high

### E-JOINT-041
- Claim: TD's forward kernel on T^m is the wrapped normal, sampled by drawing from the corresponding unwrapped
  isotropic normal and taking elementwise mod 2π; the score head produces one real-valued pseudoscalar per rotatable
  bond, extracted from the pseudotorque layer.
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 4 (Eq. 3–4) and PDF p. 6 (§3.4).
- Exact quote:
  > We thus sample from the perturbation kernel by sampling from the corresponding unwrapped isotropic normal and taking elementwise mod 2.
  > We extract the pseudoscalar outputs of this filter to produce a single real-valued pseudoscalar prediction i for each rotatable bond.
- Shows vs infers: SHOWS (π and δτ glyphs lost in extraction). Reused from E-ARCH-031 (VERIFIED); **both sentences
  re-checked in round 5 on PDF pages 4 and 6 of the stored fulltext.**
- Supports card(s): EXP-1 §2.4, §2.5
- Confidence: high

### E-JOINT-042
- Claim: TD requires the score model to be SE(3)-invariant but **parity-equivariant** (sign-changing) because the
  physical energy is parity-invariant: p(τ(C) | L(C)) = p(τ(−C) | L(−C)) implies ∇log p_t(τ(C)) = −∇log p_t(τ(−C)),
  so the outputs must be pseudoscalars in R^m (Prop. 2, Eq. 6).
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 6 (§3.3, Prop. 2, Eq. 6); proof PDF p. 15
  (App. B.2, Eq. 20–22).
- Exact quote:
  > the score model must be invariant under SE(3) but equivari- ant (change sign) under parity inversion of the input point cloud-- i.e. it must output a set of pseudoscalars in Rm.
- Shows vs infers: SHOWS. NEW in round 5 (the parity *argument*, not just the code fact of E-JOINT-020).
  INFERENCE: the same argument run for a bond length r or a bond angle θ gives r(−C) = r(C) and θ(−C) = θ(C), so
  their scores are parity-**invariant** and their head must be `0e`; a Cremer–Pople coordinate flips sign under
  inversion (E-JOINT-067) and so needs `0o`. This is the parity bookkeeping the spec must get right.
- Supports card(s): EXP-1 §2.4
- Confidence: high

### E-JOINT-043
- Claim: TD computes exact likelihoods with the probability-flow ODE,
  log p₀(τ₀) = log p_T(τ_T) − ∫₀^T ½ g²(t) ∇·s_G(τ_t, t) dt (Eq. 7), and computes the divergence **directly** rather
  than by Hutchinson's estimator, because Hutchinson gives an unbiased estimate of log p₀ but a biased estimate of
  p₀, which is unsuitable for importance weighting; the direct computation is feasible because the torsional space is
  low-dimensional.
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 6 (§3.5, Eq. 7).
- Exact quote:
  > However, this gives a biased estimate of p0( ), which is unsuitable for our applications. Thus, we compute the divergence term directly, which is feasible here (unlike in Euclidean diffusion) due to the reduced dimensionality of the torsional space.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE (central to Exp 4's feasibility): adding k geometry coordinates
  raises the per-step cost of the exact divergence from m+1 to m+k+1 network evaluations, so the cost of exact
  likelihood scales linearly in the number of extra coordinates — the single hardest constraint on Exp 4's design.
- Supports card(s): EXP-4 §2, §3, §5
- Confidence: high

### E-JOINT-044
- Claim: TD's Prop. 3 converts the torsional density to a Euclidean one:
  p_G(x | L) = p_G(τ | L) / (8π² sqrt(det g)) with g_αβ = Σ_k J_α^(k)·J_β^(k), where α, β run over the m torsions plus
  3 global rotations; J for a torsion is the centred cross product of the bond unit vector with the displacement from
  the rotation axis for atoms on the rotating side and 0 otherwise (Eq. 9), and J for the three rotations is x_k × ê
  (Eq. 10); 8π² is the volume of SO(3).
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 6 (§3.5, Prop. 3, Eq. 8) and PDF p. 7
  (Eq. 9–10); full derivation PDF p. 16 (App. B.3, Eq. 23–30).
- Exact quote:
  > Let M be (m + 3)-dimensional manifold embedded in 3n-dimensional Euclidean space formed by the set of all centered conformers with fixed local structures but arbitrary torsion angles and orientation.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: Exp 4's manifold is (m + k + 3)-dimensional with k geometry
  coordinates, so exactly the same construction applies with J extended by ∂x/∂(geometry coordinate); the
  "fixed local structures" clause is the assumption Exp 4 removes.
- Supports card(s): EXP-4 §2
- Confidence: high

### E-JOINT-045
- Claim: TD's energy-based training alternates resampling and score matching: samples are drawn from the model,
  reweighted by w̃ = p̃(τ | L)/q(τ | L) using Prop. 3, and the DSM loss is approximated under the reweighted
  distribution; the resampler may use **fewer steps** than the inference flow, an option the paper says is
  unavailable to existing Boltzmann generators.
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 7 (§3.6, Alg. 1).
- Exact quote:
  > we can use very few steps (a shallow flow) during resampling to accelerate training, and then increase the number of steps (a deeper flow) for better approximations during inference--an option unavailable to existing Boltzmann generators.
- Shows vs infers: SHOWS. NEW in round 5.
- Supports card(s): EXP-4 §3, §5
- Confidence: high

### E-JOINT-046
- Claim: TD's torsional Boltzmann generator is evaluated by effective sample size out of 32 samples on GEOM-DRUGS
  molecules with 3–7 rotatable bonds, using **the local structures of the first ground-truth conformers**, against
  annealed importance sampling: at 300 K, ESS is Uniform 1.02, AIS(100 steps) 2.06, Torsional BG(5) 3.04,
  Torsional BG(20) 4.68; at 1000 K 1.71 / 6.72 / 7.28 / 11.42.
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 10 (§4.5, Table 4).
- Exact quote:
  > We train and test on GEOM- DRUGS molecules with 3-7 rotatable bonds and use the local structures of the first ground-truth conform- ers.
  > Torsional 5 7.28 3.60 3.04
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: this is the only baseline Exp 4 can be compared against, and it
  conditions on a fixed ground-truth L — so Exp 4's p(L, τ) is **not** directly comparable to it on ESS unless the
  same L-conditioned restriction is also run.
- Supports card(s): EXP-4 §0, §1, §4
- Confidence: high

### E-JOINT-047
- Claim: TD's own conclusion names the extension Exp 1 attempts: "A natural extension is to relax the rigid local
  structure assumption by developing an efficient diffusion-based model over the full space of intrinsic coordinates
  while still incorporating chemical constraints."
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 10 (§5 Conclusion).
- Exact quote:
  > A natural extension is to relax the rigid local structure assumption by developing an efficient diffusion-based model over the full space of intrinsic coordinates while still incorporating chemical constraints.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE (novelty, both ways): the *idea* of Exp 1 is TD's own stated
  future work, so "we propose relaxing rigid L" is not a novel idea claim; what can be new is the **coupling**
  mechanism, the evidence that coupling is necessary (E-JOINT-001), and a working construction with ring closure and
  stereo preservation.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-048
- Claim: TD states the error achievable by torsional diffusion is lower-bounded by the quality of the local structure
  from the chosen cheminformatics method, and that training on ground-truth L and testing on RDKit L significantly
  hurts performance (DRUGS ablation: COV-R 72.7 → 34.8 %, AMR-R 0.582 → 0.920 Å).
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 22 (App., "Limitations"), PDF p. 7 (§4.1),
  PDF p. 27 (Table 8).
- Exact quote:
  > the error that torsional diffusion can achieve is lower bounded by the quality of the local structure from the selected cheminformatics method.
  > Train on ground truth L 34.8 22.4 0.920 0.909 22.3 7.8 1.182 1.136
- Shows vs infers: SHOWS. Reused from E-PRIOR-010 / -011 / -012 (VERIFIED, round 4); **re-checked in round 5.**
- Supports card(s): EXP-1 §1, §4
- Confidence: high

### E-JOINT-049
- Claim: TD App. F.1 already uses another randomly chosen ground-truth conformer's local structure as a measure of
  ground-truth L variability (0.284 Å on DRUGS), but only as a conformer-matching floor, never as a model test.
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 21 (App. F.1).
- Exact quote:
  > resulting from matching a ground truth conformer to the local structure of another randomly chosen ground truth conformer,
- Shows vs infers: SHOWS. Reused from E-PRIOR-013 (VERIFIED); **re-checked in round 5.** INFERENCE: our split-half
  test (E-JOINT-001) is the model-in-the-loop version of this quantity, which TD never ran.
- Supports card(s): EXP-1 §3, §8
- Confidence: high

---

## D. Prior art: joint internal-coordinate / angle + torsion generation

### E-JOINT-060
- Claim: RINGER represents macrocycles in **redundant** internal coordinates with bond lengths fixed, diffuses bond
  angles and torsions **jointly** with a wrapped normal, and must rebuild Cartesians by an SLSQP-constrained
  least-squares fit because a cycle with fixed bond distances contains three redundant torsions and two redundant
  bond angles.
- Source type: paper
- Locator: paper: `papers/related/2023_grambow_ringer.pdf`, PDF p. 5 (representation and Eq. 3).
- Exact quote:
  > Macrocycles with fixed bond distances contain three redundant torsional angles and two redundant bond angles.
  > we treat this as a constrained optimization problem using the Sequential Least Squares Quadratic Programming (SLSQP) algorithm66 to ensure valid Cartesian coordinates while satisfying distance constraints
- Shows vs infers: SHOWS (macrocycles, CREMP). Reused from E-ARCH-044 (VERIFIED, round 4); **both sentences
  re-checked in round 5 and both are on PDF p. 5 of the stored fulltext.** INFERENCE (novelty): "joint bond-angle +
  torsion diffusion" is **prior art** — RINGER did it in 2023. What RINGER does not do: drug-like polycyclic small
  molecules, GEOM/TD-protocol benchmarking, bond lengths, or a coupled-vs-decoupled test.
- Supports card(s): EXP-1 §1, §2.6, §2.7, §8
- Confidence: high

### E-JOINT-061
- Claim: RINGER reports that TD retrained on CREMP inherits RDKit's ring-backbone distribution exactly and that
  default RDKit seeding gives very poor performance; RINGER's authors therefore built TD's *training* local structures
  from ground-truth xTB-optimised conformers followed by MMFF94 optimisation.
- Source type: paper
- Locator: paper: `papers/related/2023_grambow_ringer.pdf`, PDF p. 6 (structural analysis) and PDF p. 19 (SI
  baselines, "Torsional Diffusion").
- Exact quote:
  > (note that TorDiff relies on RDKit for backbone sampling, and hence produces an identical plot)
  > we seeded the training local structures using the ground-truth xTB-optimized conformers with subsequent MMFF94 force-field optimization.
- Shows vs infers: SHOWS. Reused from E-ARCH-045 / E-ARCH-049 (VERIFIED); **first quote re-checked on PDF p. 6 in
  round 5.** INFERENCE: independent in-domain support for the S5 lesson (E-JOINT-006) — practitioners already match
  TD's training L to the deployed, physics-processed L source.
- Supports card(s): EXP-1 §4
- Confidence: high

### E-JOINT-062
- Claim: FoldingDiff diffuses six internal angles per residue (three bond angles, three dihedrals) with a wrapped
  normal, rebuilds Cartesians by NeRF with bond distances fixed to average lengths, and reports a "lever arm effect"
  whereby errors early in the chain significantly alter the whole structure.
- Source type: paper
- Locator: paper: `papers/related/2022_wu_foldingdiff.pdf`, PDF p. 4 (§3) and PDF p. 10 (Discussion).
- Exact quote:
  > converted back to 3D Cartesian coordinates by iteratively adding atoms to the protein backbone as described in Parsons et al. (2005), fixing bond distances to average lengths
  > errors early in the chain to significantly alter the overall generated structure - a sort of "lever arm effect."
- Shows vs infers: SHOWS (proteins, backbone only, no rings). Reused from E-ARCH-047 (VERIFIED); **both quotes
  re-checked in round 5 on PDF p. 4 and p. 10.** INFERENCE: the lever-arm effect is the strongest argument *against*
  a sequential-NeRF internal-coordinate parameterisation for V3 and *for* the Cartesian-subspace construction, in
  which an error in one coordinate stays local.
- Supports card(s): EXP-1 §2.3, §2.6
- Confidence: high

### E-JOINT-063
- Claim: GO-Flow decomposes a conformer into centre of mass, a unit-quaternion orientation and internal coordinates
  (bond lengths, angles, periodic torsions), learns one flow per manifold with entropic-OT couplings for the internal
  coordinates, and maps internal velocities to Cartesian velocities with an explicit Jacobian projection.
- Source type: paper
- Locator: paper: `papers/related/2026_liu_goflow.pdf`, PDF p. 3–4 (§3.1–3.4, Eq. 1–9, Alg. 1).
- Exact quote:
  > We therefore employ a mathematically rigorous Jacobian-based projection to map internal velocities back to Cartesian updates.
- Shows vs infers: SHOWS. Reused from E-ARCH-035 (VERIFIED); **re-checked on PDF p. 4 in round 5.** INFERENCE:
  GO-Flow is the closest full-internal-coordinate prior art and goes **internal → Cartesian**; our proposed
  construction goes the other way (Cartesian velocity **projected onto** the local-structure subspace), which is why
  ring closure is free for us and a problem for them (E-JOINT-064).
- Supports card(s): EXP-1 §1, §2.3, §8
- Confidence: high

### E-JOINT-064
- Claim: GO-Flow does **not** address ring closure: a full-text search of the paper for "ring", "cycl", "closure" and
  "NeRF" finds only an atom feature (`in_ring`) and one qualitative remark about baselines' distorted rings; its
  internal coordinates for ring bonds are therefore redundant and unconstrained.
- Source type: paper
- Locator: paper: `papers/related/2026_liu_goflow.pdf`, searched the whole stored fulltext; the only substantive hit
  is PDF p. 15 (App. E); the atom feature table is PDF p. 14.
- Exact quote (PDF p. 15; the extraction interleaves two columns, so the sentence reads broken in the text dump):
  > valid bond lengths and angles, devoid of the disconnected components or distorted rings often seen in baseline ... methods.
- Shows vs infers: SHOWS by absence (whole-text search; absence of evidence, not proof). Reused from E-ARCH-037;
  **re-run in round 5 on the stored fulltext and the hit confirmed on PDF p. 15.** INFERENCE: GO-Flow's claimed
  internal-coordinate advantage is undemonstrated for ring puckers — the part of L our data say matters
  (E-JOINT-003).
- Supports card(s): EXP-1 §1, §2.7, §8
- Confidence: medium (absence claim)

### E-JOINT-065
- Claim: GO-Flow is evaluated only on the ConfGF/GeoDiff protocol (40,000 training molecules × 5 conformers, 200 test
  molecules) with 700k iterations on eight A6000 GPUs; its own ablation without the internal-coordinate manifold has
  GEOM-DRUGS COV-R 90.26 % / MAT-R 0.8512 Å against the full model's 94.82 % / 0.7971 Å.
- Source type: paper
- Locator: paper: `papers/related/2026_liu_goflow.pdf`, PDF p. 7 (§4.4.1, Table 3) and PDF p. 14 (App. D.1/D.4).
- Exact quote:
  > the w/o C variant achieves a COV-R of only 90.26% and a MAT-R of 0.8512 Å, which is substantially inferior to the full GO-Flow model (94.82% and 0.7971 Å, respectively).
- Shows vs infers: SHOWS. Reused from E-PRIOR-020 / E-ARCH-036 (VERIFIED). **Not comparable with our TD-protocol
  numbers** (different split, 200 test molecules, δ 1.25 Å on DRUGS).
- Supports card(s): EXP-1 §1, §3
- Confidence: high

### E-JOINT-066
- Claim: PuckerFlow does flow matching in Cremer–Pople coordinates (an N-ring has N − 3 CP coordinates from a discrete
  Fourier transform of the out-of-plane displacements), rebuilds rings from CP coordinates using **precomputed
  dictionary bond lengths and angles** parameterised by bond type, element and ring size, and therefore produces
  closed rings by construction.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 3–5 (§2.1–2.2, Eq. 1–2) and PDF p. 16
  (SI A.1.2, Eq. 8–10).
- Exact quote:
  > It uses precomputed bond lengths and angles from the training data, parameterized by bond type, element, and ring size.
- Shows vs infers: SHOWS. Reused from E-ARCH-038 (VERIFIED); **re-checked in round 5 and the quote is on PDF p. 5.**
- Supports card(s): EXP-1 §2.3, §2.7
- Confidence: high

### E-JOINT-067
- Claim: Each Cremer–Pople coordinate is a **pseudoscalar** (rotation-invariant, sign-changing under parity), produced
  in PuckerFlow by equivariant filters on the ring mean plane followed by a scaled DFT, in an e3nn tensor-field
  network of the same class as TD's.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 6 (§2.2) and PDF p. 20–21 (SI A.5).
- Exact quote:
  > each Cremer-Pople coordinate being a pseudoscalar, i.e., invariant under rotation but changing sign under parity inversion.
- Shows vs infers: SHOWS. Reused from E-ARCH-040 (VERIFIED); **re-checked on PDF p. 6 in round 5.** INFERENCE: a CP
  head is the same symmetry class as TD's torsion head (`0o`), so both can sit in one network — and, by the same
  argument, bond lengths and bond angles must use a **parity-even (`0e`)** head (E-JOINT-042).
- Supports card(s): EXP-1 §2.4
- Confidence: high

### E-JOINT-068
- Claim: Reconstructing a ring from its *true* CP coordinates with PuckerFlow's dictionary bond lengths and angles
  leaves a median RMSD of 0.046 Å to the original structure.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 18 (SI A.3) and Fig. 9 on PDF p. 19.
- Exact quote:
  > The median RMSD of 0.046 Å confirms that the reconstructed geometries closely reproduce the reference structures
- Shows vs infers: SHOWS (substituent-free monocycles). Reused from E-ARCH-041 (VERIFIED); **re-checked on PDF p. 18
  in round 5.** INFERENCE, and a hard design constraint: a *perfect* CP factor with dictionary bonds/angles still
  carries a ≈ 0.05 Å ring residual — comparable to ET-Flow's entire QM9 median — so a CP variant must keep the seed's
  own bond lengths/angles rather than dictionary values, or it cannot reach below ≈ 0.05 Å on ring molecules.
- Supports card(s): EXP-1 §2.3, §2.7, §7
- Confidence: high

### E-JOINT-069
- Claim: PuckerFlow's authors state that the architectures of PuckerFlow and torsional diffusion are compatible and
  propose **joint flow matching over the cyclic and torsional degrees of freedom** as future work; their released
  code builds on the TD codebase.
- Source type: paper (+ code repository snapshot)
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 10 (§3) and PDF p. 21 (SI A.5.4);
  code: https://github.com/digital-chemistry-laboratory/puckerflow README, snapshot
  `papers/blogs/puckerflow_repo_readme.txt` (accessed 2026-10-12 by SCOUT ARCH).
- Exact quote:
  > making an extension feasible that performs joint flow matching over the different degrees of freedom to simultaneously sample the cyclic and torsional degrees of freedom.
- Shows vs infers: SHOWS. Reused from E-ARCH-042 (VERIFIED); **re-checked on PDF p. 10 in round 5.**
  **Revised after D-008: the source calls the joint flow a *feasible extension*. The word "announced" below is
  withdrawn; read "proposed as feasible".**
  INFERENCE (novelty risk, high): the *exact* V3b construction is announced future work of an active group that
  already has a working CP flow on the TD codebase. A FlexiTors paper competes with them on time, so the claim must
  rest on what they do not have: the coupling evidence (E-JOINT-001), polycyclic systems (E-JOINT-071) and the
  TD-protocol GEOM benchmark.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-070
- Claim: TMCgen (the PuckerFlow group, 2026) already combines an "angle-type" manifold factor with torsions in one
  diffusion — simulation-free diffusion on S² for metal–ligand coordination directions plus per-ligand rotation and
  torsion diffusion, with each factor updated separately — and the authors say the n-sphere generalisation could
  serve the conformational ensembles of molecular ring systems.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_tmcgen.pdf`, PDF p. 5 (§2.1) and PDF p. 6 (§2.2, Eq. 3–4).
- Exact quote:
  > we apply updates to the diffusion over the coordination environment, the rotations and the torsions for each ligand separately
  > such as for the conformational ensembles of molecular ring systems, we focus on the 2-sphere in the main text.
- Shows vs infers: SHOWS (metal complexes). Reused from E-ARCH-046 (VERIFIED).
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-071
- Claim: QM9's bad ring seeds are split across ring classes that a Cremer–Pople factor cannot represent equally: of
  885 test ring molecules, 420 have only isolated rings and **465 contain a fused, bridged or spiro system**; of the
  2074 ETKDG seeds with ring-dihedral RMSD > 10°, 449 (22 %) are in fused/bridged/spiro systems; 832 of 885 ring
  molecules carry an exocyclic heavy atom.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv` with `cond == L_etkdg2L`, joined to an
  RDKit ring-system analysis of the SMILES (round-3 scripts `rings.py`, `p2.py`, cited in E-GEOM-011).
- Exact quote (column used):
  > ring_dihedral_rmsd
- Shows vs infers: SHOWS (counts). Reused from E-ARCH-003 (round-3 E-GEOM-011 + D-209 addendum, VERIFIED).
  INFERENCE, decisive for the coordinate choice: more than half of QM9's ring molecules are polycyclic, where CP
  coordinates of the constituent rings are not independent and closure needs RINGER-style constrained optimisation
  (E-JOINT-060). This is why the primary variant is the Cartesian-subspace one and CP is a secondary ablation.
- Supports card(s): EXP-1 §2.3, §2.7
- Confidence: high

### E-JOINT-072
- Claim: On a product manifold the forward diffusion proceeds independently in each factor and the tangent space is a
  direct sum, so it suffices to sample each factor's diffusion kernel and regress each factor's score independently
  (DiffDock's T(3) × SO(3) × SO(2)^m recipe).
- Source type: paper
- Locator: paper: `papers/related/2022_corso_diffdock.pdf`, PDF p. 6, §4.3 "Diffusion on the product space".
- Exact quote (the extraction hyphenates "pro- ceeds" across a column break):
  > since P is a product manifold, the forward diffusion pro- ceeds independently in each manifold
  > Thus, it suffices to sample from the diffusion kernel and regress against its score in each group independently.
- Shows vs infers: SHOWS. Reused from E-ARCH-032 (VERIFIED); **re-checked in round 5 — both sentences are on PDF
  p. 6; the round-4 quote silently repaired the hyphenation, this entry records the raw text.**
- Supports card(s): EXP-1 §2.3, §2.5
- Confidence: high

### E-JOINT-073
- Claim: Riemannian flow matching is simulation-free on manifolds with closed-form geodesics, explicitly including the
  torus and any product of such spaces.
- Source type: paper
- Locator: paper: `papers/related/2023_chen_riemannian_flow_matching.pdf`, PDF p. 2 (§1) and PDF p. 5 (Eq. 15,
  Prop. 3.2).
- Exact quote:
  > On simple geometries, where geodesics are known in closed form (e.g., Euclidean space, hypersphere, hyperbolic space, torus, or any of their product spaces), Riemannian Flow Matching remains completely simulation-free.
- Shows vs infers: SHOWS. Reused from E-ARCH-061 (VERIFIED); **re-checked on PDF p. 2 in round 5.**
- Supports card(s): EXP-1 §2.3
- Confidence: high

### E-JOINT-074
- Claim: Torsional-GFN samples torsions conditioned on the molecular graph **and the local structure L**, and follows
  the shift of the torsional energy modes when L changes (unseen MD bond lengths and angles) — direct evidence that
  p(τ | L) genuinely depends on L.
- Source type: paper
- Locator: paper: `papers/related/2025_ezzine_torsional_gfn.pdf`, PDF p. 2 (Eq. 2) and PDF p. 4 (Results).
- Exact quote:
  > Torsional-GFN is able to follow the shift of the modes in the energy landscape when sampling torsion angles for unseen local structures.
- Shows vs infers: SHOWS (a handful of small molecules, energy-trained, not GEOM). Reused from E-ARCH-048 (VERIFIED);
  **re-checked on PDF p. 4 in round 5.** INFERENCE: the closest published claim of a torsion model *robust* to L; it
  is the counter-evidence to our brittleness result, and it is also the published precedent for "condition the
  torsion model on L", which V2/V3 do by construction.
- Supports card(s): EXP-1 §1, §4, §8
- Confidence: medium (small study)

### E-JOINT-075
- Claim: CoarsenConf conditions on the RDKit conformer and learns the **coordinate correction** from RDKit to the
  ground truth, on the TD-protocol split (precision metrics with a reduced sample budget).
- Source type: paper
- Locator: paper: `papers/related/2023_lee_coarsenconf.pdf`, PDF p. 6.
- Exact quote:
  > of learning raw atom coordinates, we learn to predict the difference between the RDKit reference and ground truth
- Shows vs infers: SHOWS. Reused from E-PRIOR-029 (VERIFIED). INFERENCE: "learn a Cartesian correction to the RDKit
  conformer" is prior art; what CoarsenConf does *not* do is restrict the correction to the local-structure subspace,
  keep a torsion-diffusion factor, or report recall on the full TD protocol.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-076
- Claim: ET-Flow's conditional flow matching uses the interpolant x_t = t x₁ + (1 − t) x₀ with noise σ√(t(1 − t)) z
  and target v = x₁ − x₀ + ((1 − 2t) / (2√(t(1 − t)))) σ z (Eq. 2–5), a harmonic prior Kabsch-aligned to each data
  sample, and a **post hoc chirality flip**; in its DRUGS ablation a Gaussian prior raises AMR-R from 0.557 to
  0.640 Å and removing alignment to 0.622 Å.
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 4 (§3.1–3.2, Eq. 1–5), PDF p. 5 (§3.3–3.5),
  PDF p. 19 (Table 7).
- Exact quote:
  > we reduce the transport costs between samples from the harmonic prior 0 and samples from the data distribution 1 by rotationally aligning them using the Kabsch algorithm
  > ET-Flow (Gaussian Prior) 66.53 73.01 0.640 0.625 44.41 40.88 0.903 0.864
- Shows vs infers: SHOWS (ρ glyphs lost). Reused from E-ARCH-050 / -051 (VERIFIED); **first quote re-checked on PDF
  p. 4 in round 5.** INFERENCE: the source distribution of a flow matters a great deal (0.083 Å of AMR-R here), which
  is the argument for anchoring the geometry factor at the RDKit L rather than at noise; and the post hoc chirality
  flip is published precedent for a stereo-repair step.
- Supports card(s): EXP-1 §2.2, §2.5, §2.8
- Confidence: high

### E-JOINT-080
- Claim: HarmonicFlow found it helpful to parameterise the flow network to predict the endpoint x₁ rather than the
  velocity (x₁ − x₀).
- Source type: paper
- Locator: paper: `papers/related/2023_stark_harmonicflow.pdf`, PDF p. 4 (§3; the extraction interleaves two columns).
- Exact quote:
  > Notably, we find it helpful to parameterize v to predict x1 instead of (x1 - x0).
- Shows vs infers: SHOWS (docking). Reused from E-ARCH-060 (VERIFIED, round 4); **re-checked on PDF p. 4 in round 5**
  (the sentence is split across the two-column interleave). INFERENCE: x₁-prediction removes the γ′(t) divergence of
  the stochastic-interpolant velocity target at t → 0, 1 (Q-P1-001, Q-P3-002).
- Supports card(s): EXP-1 §2.2
- Confidence: high

### E-JOINT-081
- Claim: PuckerFlow trains its CP flow with **x₁ prediction**, drawing t ~ U(0, 1) and regressing u_θ(x_t, t) onto x₁
  (SI Algorithm 1).
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 20 (SI Algorithm 1).
- Exact quote:
  > Algorithm 1: Flow Matching training with Cremer-Pople coordinate (x1) prediction
- Shows vs infers: SHOWS. NEW in round 5 (the round-4 E-ARCH-039 summarised it; the heading was re-extracted here).
  INFERENCE: the closest-prior-art geometry flow uses the same parameterisation EXP-1 V2 now adopts.
- Supports card(s): EXP-1 §2.2
- Confidence: high

### E-JOINT-082
- Claim: ET-Flow trains with t ~ U(0, 1) (no clipping stated in the main text), samples by Euler integration from
  t = 0 to 1, and applies stochastic sampling only in the final range t ∈ [0.8, 1.0].
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 4 (§3.1, Eq. 5) and PDF p. 5 (§3.3).
- Exact quote:
  > For training, we sample (i) x0 0, x1 1, and t U (0, 1), (ii) interpolate according to Equation 2, (iii) add noise from a standard Gaussian,
  > We apply stochastic sampling only during the final part of the inte- gration steps, specifically within the range t [0.8, 1.0].
- Shows vs infers: SHOWS (glyphs ∼, ρ, ∈ lost in extraction). NEW in round 5. INFERENCE: the source gives no endpoint
  clipping rule for the γ′ term. EXP-1 avoids the issue by x₁ prediction (E-JOINT-080/-081) rather than inventing an
  ε.
- Supports card(s): EXP-1 §2.2
- Confidence: high

### E-JOINT-077
- Claim: The flow-matching refiner departs from pure-noise initialisation and adopts a base distribution that retains
  the ground-truth signal, so that at test time the ODE can start from any upstream sample; it is implemented by
  fine-tuning ET-Flow's architecture and weights, and its QM9 table has a TD row (0.178 Å) but **no "TD + Refiner"
  row**.
- Source type: paper
- Locator: paper: `papers/related/2025_xu_fm_refiner.pdf`, PDF p. 4 (§4.1, Eq. 4–9), PDF p. 7 (§5.1 and Tables 1–2).
- Exact quote:
  > Since our goal is to refine upstream conformers rather than regenerate them, we depart from the usual pure-noise initialization and adopt a base distribution x0 that retains the ground-truth signal.
  > Torsional Diff. - - 0.178 0.147 - - 0.221 0.195
- Shows vs infers: SHOWS. Reused from E-ARCH-054 / -055 (VERIFIED). INFERENCE: a *post hoc* learned refiner on TD
  output is untested in the literature, so it is an available control arm rather than prior art; but it is also the
  single most likely reviewer question ("why not just refine TD's output?"), so Exp 1 must include that contrast.
- Supports card(s): EXP-1 §4, §8
- Confidence: high

### E-JOINT-078
- Claim: Corso's intrinsic-diffusion thesis (a TD co-author) names ring flexibility as the largest source of
  flexibility TD does not model, proposes ring-puckering coordinates on hyperspheres as an extra factor, and proposes
  using the manifold as a **soft constraint** on a full-dimensional diffusion; it also states that RDKit ring
  conformations work fine for small drug-like rings.
- Source type: paper
- Locator: paper: `papers/related/2023_corso_intrinsic_diffusion.pdf`, PDF p. 92 (App. C.1.2), PDF p. 63
  (future work "Ring flexibility"), PDF p. 64 (future work).
- Exact quote:
  > The largest source of flexibility in molecular conformations that is not directly accounted for by torsional diffusion is the variability in ring conformations.
  > using such manifold as a soft constraint or inductive bias to make the full dimensional diffusion more efficient
  > While this works fine for small rings typically present in drug-like molecules, it suffers for larger and more flexible rings, especially for macrocycles.
- Shows vs infers: SHOWS (proposals, no experiment). Reused from E-ARCH-033 / E-PRIOR-015 / -016 (VERIFIED);
  **all three quotes re-checked in round 5: PDF p. 92, p. 64 and p. 63 respectively.** INFERENCE: our QM9
  measurement (E-JOINT-003, E-JOINT-004) contradicts the "fine for small rings" premise, and that contrast is a
  genuine, citable contribution.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-079
- Claim: A targeted arXiv search over 2023-01 → 2026-10 (18 abstract queries; joint torsion/bond-angle, multi-scale
  and hierarchical conformer generation, puckering, product manifolds, ring closure, Boltzmann generators and ESS)
  found **no paper that generates ring puckers or bond angles jointly with TD-style torsions for drug-like organic
  small molecules**, and no Boltzmann generator over torsions *and* local structure. The query
  `abs:torsion AND abs:"bond angle" AND (abs:diffusion OR abs:"flow matching")` returns exactly one conformer paper
  in the window: GO-Flow (arXiv 2605.25577). The query `abs:"Boltzmann generator" AND abs:torsion` returns nothing in
  the window. Two web searches (2026-10-15) for a joint bond-angle/torsion/pucker diffusion model and for a
  full-conformer Boltzmann generator with bond lengths and angles returned no such paper either.
- Source type: our-data (search log)
- Locator: our-data: `round5/ledger/joint_data/search_log.txt` (18 queries, newest-first, rows ≥ 2023-01-01; scripts
  `round5/ledger/joint_data/arxq.py`, `arxabs.py`, `run_search.sh`); WebSearch runs of 2026-10-15.
- Exact quote (search_log.txt):
  > === QUERY: abs:torsion AND abs:"bond angle" AND (abs:diffusion OR abs:"flow matching")
  > 2026-05-25 2605.25577v1 | Geometric Flow Matching for Molecular Conformation Generation via Manifold Decomposition
- Shows vs infers: SHOWS what the searches returned. **Absence of evidence, not proof**: the search is arXiv-centric,
  arXiv abstract search is token-based (so "bond angles" plural does not match `"bond angle"` — RINGER is missed by
  that query and was found by hand), and OpenReview PDFs sit behind a browser challenge that was not bypassed. For
  the novelty red-teamer to re-run independently.
- Supports card(s): EXP-1 §8, EXP-4 §7
- Confidence: medium

---

## E. Prior art: multi-scale / hierarchical / coarse-to-fine molecular generation (2023–2026)

### E-JOINT-090
- Claim: Equivariant Blurring Diffusion (NeurIPS 2024) is an explicitly **hierarchical, multiscale conformer
  generator**: stage 1 produces a coarse-grained fragment-level 3D structure from the molecular graph, stage 2
  generates fine atomic detail from that coarse structure while allowing the coarse structure to be adjusted.
- Source type: paper
- Locator: paper: `papers/related/2024_park_equivariant_blurring_diffusion.pdf`, PDF p. 1 (title and Abstract).
- Exact quote:
  > How can diffusion models process 3D geometries in a coarse-to-fine manner, akin to our multiscale view of the world?
  > i) generation of coarse-grained fragment-level 3D structure from the molecular graph, and ii) generation of fine atomic details from the coarse-grained approximated structure while allowing the latter to be adjusted simultaneously.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE (novelty, important): **the "multi-scale conformer generation"
  framing is already taken.** A FlexiTors paper must not claim to be the first multi-scale conformer model.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-091
- Claim: EBD's coarse scale is **fragment centroids**, not local structure: the initial atom coordinates come from
  RDKit distance geometry and the fragment coordinates are the averages of their constituent atoms; there is no
  torsion factor anywhere in EBD, and the coarse variable it corrects is a set of fragment centres of mass.
- Source type: paper
- Locator: paper: `papers/related/2024_park_equivariant_blurring_diffusion.pdf`, PDF p. 3 (§3.2) and PDF p. 8 (Fig. 5
  discussion of the fragment correction).
- Exact quote:
  > To generate the initial coordinates of fragments, we utilize RDKit distance geometry [5, 31], an efficient cheminformatics tool, instead of training an additional deep generative model.
  > we define the initial fragment coordinates xf as averages of their constituent atom coordinates, Mx^a.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: EBD is **not** prior art for a learned local-structure factor:
  moving fragment centroids leaves the *intra*-fragment geometry (bond lengths, angles, ring puckers) at whatever
  RDKit produced, which is precisely the error our data isolate (E-JOINT-003/-004). This is the sharp distinction a
  reviewer will probe, and it is defensible.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-092
- Claim: EBD is evaluated on the GeoDiff/ConfGF protocol, and its own ablation includes a "fine-to-fine" variant
  whose prior is the RDKit *atom* coordinates rather than the fragment coarse structure; that fine-to-fine variant is
  more accurate than the coarse-to-fine one.
- Source type: paper
- Locator: paper: `papers/related/2024_park_equivariant_blurring_diffusion.pdf`, PDF p. 8 (§5.2, Table 2).
- Exact quote:
  > We measured the performance when the prior is x^a pRDKit(xa). Table 2 presents coarse-to-fine (C2F) result when |S| = 50 and fine-to-fine (F2F) result. As expected, F2F shows more accurate results compared to C2F as it includes more detail in the prior.
- Shows vs infers: SHOWS (GD-P; not comparable with our numbers). NEW in round 5. INFERENCE, and a direct argument
  for our design: EBD's own ablation says *more detail in the prior is better*. Anchoring the geometry factor at the
  full RDKit atom coordinates (not at a coarse surrogate and not at noise) is therefore the empirically supported
  choice.
- Supports card(s): EXP-1 §2.2, §2.5
- Confidence: high

### E-JOINT-093
- Claim: HierDiff (ICML 2023) is coarse-to-fine **de novo 3D molecule generation**, not conformer generation: it
  generates coarse-grained fragment-node geometries by equivariant diffusion, decodes them to fragments and assembles
  an atomic structure; its stated motivation is that atom-resolution models "ignore intrinsic local structures such
  as rings".
- Source type: paper
- Locator: paper: `papers/related/2023_qiang_hierdiff.pdf`, PDF p. 1 (Abstract).
- Exact quote:
  > existing methods usually generate molecules in atom resolution and ignore intrinsic local structures such as rings, which leads to poor quality in generated structures, especially when generating large molecules.
- Shows vs infers: SHOWS. NEW in round 5 (PDF + fulltext downloaded this round). INFERENCE: the "rings are the local
  structure that atom-level models get wrong" observation is published, but for de novo generation; it has never been
  measured *with a torsion model in the loop on a conformer benchmark*, which is E-JOINT-003.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-094
- Claim: OctMole (arXiv 2610.16586, October 2026) is a multiscale octree flow for **de novo** 3D molecular
  generation: a hierarchical occupancy flow resolves molecular support coarse-to-fine, then a joint atom flow assigns
  atom types and in-cell offsets; it reports QM9 and GEOM-DRUG results as a generation (not conformer-ensemble)
  benchmark.
- Source type: paper
- Locator: paper: `papers/related/2026_jin_octmole.pdf`, PDF p. 1 (Abstract) and PDF p. 2 (contributions).
- Exact quote:
  > Existing atom-based diffusion and flow models operate largely at a single scale, while dense spatial fields become costly at high resolution. We introduce OctMole, a multiscale octree flow framework for 3D molecular generation.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: the newest multi-scale molecular generator is still de novo
  generation on a spatial hierarchy, with no internal-coordinate or torsion factor; it does not overlap Exp 1's
  mechanism, but it does further occupy the "multiscale" vocabulary.
- Supports card(s): EXP-1 §1, §8
- Confidence: high

### E-JOINT-095
- Claim: MSGEN (ICLR 2026) argues that the spatial arrangement of key substructures such as scaffolds anchors the
  conformer distribution and generates coarse substructures first.
- Source type: paper (OpenReview abstract only)
- Locator: paper: https://openreview.net/forum?id=uYlNjHC7ag, abstract; snapshot
  `papers/blogs/openreview_abstracts_r4.txt` (accessed 2026-10-11 by SCOUT PRIOR).
- Exact quote:
  > we demonstrate that capturing the spatial arrangement of key substructures, such as scaffolds, is essential, as they serve as anchors that define the overall molecular distribution.
- Shows vs infers: SHOWS (abstract only; the PDF is behind an OpenReview browser challenge that was not bypassed).
  Reused from E-PRIOR-022. INFERENCE: the closest published "scaffold/substructure geometry matters" claim for
  conformer generation; it is a Cartesian coarse-to-fine claim, not a local-structure one, and not ring-pucker
  specific.
- Supports card(s): EXP-1 §1, §8
- Confidence: medium (abstract only)

### E-JOINT-096
- Claim: A dual-scale equivariant flow-matching method (arXiv 2410.07539, 2024) already separates training and
  inference into coarse-grained and all-atom stages **as a Boltzmann generator**, for Y6 molecular clusters from MD.
- Source type: paper (abstract, arXiv API)
- Locator: paper: https://arxiv.org/abs/2410.07539, abstract, retrieved 2026-10-15 with
  `round5/ledger/joint_data/arxabs.py`; row in `round5/ledger/joint_data/search_log.txt` under
  `abs:"Boltzmann generator" AND abs:conformer`.
- Exact quote:
  > we developed a dual-scale flow matching method that separates training and inference into coarse-grained and all-atom stages and enhances both the accuracy and efficiency of standard flow matching samplers.
- Shows vs infers: SHOWS (abstract; **no PDF stored — this entry is abstract-only and must not be cited for any
  mechanism detail**). INFERENCE: "multi-scale Boltzmann generator" exists for intermolecular packing of clusters,
  not for intramolecular L–τ of a single drug-like molecule.
- Supports card(s): EXP-4 §1, §7
- Confidence: low (abstract only, no PDF)

---

## F. Boltzmann generators and likelihood in internal coordinates

### E-JOINT-110
- Claim: The original Boltzmann generator (Noé et al. 2019) already models bond lengths and bond angles: training
  directly on Cartesian coordinates gave large energies and distorted bond lengths and angles, which was fixed by a
  coordinate transformation in which atoms are measured in internal coordinates (bond lengths, angles and torsions
  with respect to parent atoms) and then normalised; the paper notes that whitening and IC normalisation are
  essential because they put large whole-molecule fluctuations on the same scale as stiff coordinates such as bond
  lengths.
- Source type: paper
- Locator: paper: `papers/related/2019_noe_boltzmann_generators.pdf` (arXiv 1812.01729; Science 2019), PDF p. 14
  (BPTI section) and PDF p. 25 (Methods, "Mixed Coordinate transformation layer").
- Exact quote:
  > Training a Boltzmann Generator directly on the Cartesian coordinates resulted in large energies, and un- realistic structures with distorted bond lengths and angles.
  > whitening and IC normalization are essential for training Boltzmann Generators for complex molecules, as this sets large fluctuations of the whole molecule on the same scale as small vibrations of stiff coordinates such as bond lengths.
- Shows vs infers: SHOWS (a single protein / small systems, per-system flow). NEW in round 5. INFERENCE (novelty,
  negative): "a Boltzmann generator over bond lengths and angles" is **not novel** — it is the original formulation.
  Exp 4's novelty can only be *transferability across molecules* combined with the full L + τ space, and the
  per-coordinate scale normalisation is a solved engineering requirement we must copy, not invent.
- Supports card(s): EXP-4 §1, §2, §7
- Confidence: high

### E-JOINT-111
- Claim: Transferable Boltzmann Generators (Klein & Noé, NeurIPS 2024) state explicitly that torsional diffusion's
  Boltzmann generator is conditioned on a **single local structure for each molecule, namely fixed bonds and angles**,
  and is consequently unable to generate samples from the full Boltzmann distribution in Euclidean space; their own
  framework is the first transferable Boltzmann generator, evaluated on dipeptides.
- Source type: paper
- Locator: paper: `papers/related/2024_klein_transferable_boltzmann.pdf` (arXiv 2406.14426), PDF p. 2 (§1,
  related-work paragraph) and PDF p. 1 (Abstract).
- Exact quote:
  > However, the distribution is conditioned on a single local structure for each molecule, namely fixed bonds and angles. Consequently, in contrast to our work, they are unable to generate samples from the full Boltzmann distribution in Euclidean space.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: this is the published statement of exactly the gap Exp 4 would
  close, and it also means **a transferable Boltzmann generator over full Euclidean coordinates already exists** (TBG,
  for dipeptides, in Cartesian coordinates). The remaining gap is narrow and specific: a *transferable* Boltzmann
  generator over p(L, τ) for drug-like molecules with exact likelihoods on the GEOM/TD protocol.
- Supports card(s): EXP-4 §0, §1, §7
- Confidence: high

### E-JOINT-112
- Claim: Corso's thesis states that importance-sampling the full Boltzmann distribution would need a model over local
  structures that also provides exact likelihoods, which RDKit does not provide.
- Source type: paper
- Locator: paper: `papers/related/2023_corso_intrinsic_diffusion.pdf`, PDF p. 93, App. C.1.2 "Boltzmann generation".
- Exact quote:
  > one would need a model () over local structures that also provides exact likelihoods. This is not the case with RDKit
- Shows vs infers: SHOWS (p(L) glyphs lost in extraction). Reused from E-ARCH-034 (VERIFIED); **re-checked on PDF
  p. 93 in round 5.** INFERENCE: the TD authors themselves name the exact missing component Exp 4 must supply, which
  is good for motivation and bad for novelty-of-idea.
- Supports card(s): EXP-4 §0, §2, §7
- Confidence: high

### E-JOINT-113
- Claim: BoltzNCE (2025) states that obtaining exact likelihoods requires computing costly Jacobians during
  integration, which is impractical for large molecular systems, and reports a ≈ 100× speed-up by replacing them with
  a learned energy-based likelihood model on dipeptides.
- Source type: paper
- Locator: paper: `papers/related/2025_boltznce.pdf` (arXiv 2507.00846), PDF p. 1 (Abstract), PDF p. 4 (Eq. 6
  discussion), PDF p. 10 (conclusion).
- Exact quote:
  > obtaining these likelihoods requires computing costly Jacobians during integration, which is impractical for large molecular systems.
- Shows vs infers: SHOWS (dipeptides). NEW in round 5. INFERENCE: the exact-likelihood cost is the recognised
  bottleneck of this whole line, and TD's advantage was only that the torsional space is small (E-JOINT-043). Adding
  k ≈ 40–60 local-structure coordinates on QM9 removes that advantage, which is the central feasibility risk of
  Exp 4; a learned-likelihood fallback exists but breaks the "exact likelihood" claim.
- Supports card(s): EXP-4 §3, §5, §6
- Confidence: high

### E-JOINT-114
- Claim: Coarse-Grained Boltzmann Generators (2026) state that as system size increases even expressive generative
  models exhibit diminishing overlap with the target distribution, giving high-variance importance weights and
  ineffective reweighting; they quantify reliability with the normalised effective sample size.
- Source type: paper
- Locator: paper: `papers/related/2026_schopmans_cg_boltzmann.pdf` (arXiv 2602.10637), PDF p. 1 (§1) and PDF p. 5
  (§ metrics).
- Exact quote (the extraction interleaves two columns):
  > As system size increases, even expres- sive generative models exhibit diminishing overlap with the ... target distribution, resulting in high-variance importance
  > The reliability of importance reweighting is quantified by ... the normalized effective sample size (ESS) (Kish, 1965),
- Shows vs infers: SHOWS (peptides and solvent). NEW in round 5. INFERENCE, and the pre-declared expectation for
  Exp 4: moving from m ≈ 3 torsions to m + k ≈ 45 coordinates should *lower* ESS unless the extra factor is very
  tightly concentrated; so the honest Exp-4 hypothesis is "ESS will drop, and we will measure by how much", not
  "ESS will improve".
- Supports card(s): EXP-4 §4, §5, §6
- Confidence: high

---

## G. Reconstruction, ring closure, stereochemistry

### E-JOINT-130
- Claim: Cremer–Pople analysis of 140k molecules shows ring conformations fall into relatively few canonical
  clusters, and a knowledge-based pucker sampler reproduces CREST/GFN2 lowest-energy ring conformers at 0.09 Å mean
  RMSD, the residual being bond lengths and angles.
- Source type: paper
- Locator: paper: `papers/related/2021_chan_ring_puckering.pdf` (J. Chem. Inf. Model. 61, 743–755; open-access
  author copy from ora.ox.ac.uk), PDF p. 1 and PDF p. 11.
- Shows vs infers: SHOWS. Reused from round-3 SCOUT GEOM (`papers/INDEX.md` row). **Quote not re-extracted in round 5
  — recorded here as UNVERIFIED-IN-R5 and used only as background, not for any Exp-1 mechanism.**
- Supports card(s): EXP-1 §2.3 (background)
- Confidence: low (not re-checked this round)

### E-JOINT-131
- Claim: Our own tooling states the two stereochemical facts that define the Exp-1 risk: TD's torsion updates are
  rigid rotations so tetrahedral chirality is frozen at whatever the RDKit seed had, and non-ring double bonds count
  as rotatable because `get_transformation_mask` has no bond-order test, so TD *can* flip E/Z; evaluation uses
  `GetBestRMS` (proper rotations only), so a wrong enantiomer or diastereomer is never matched.
- Source type: code
- Locator: code: `tools/stereo_check.py:1-11` (module docstring, with its own `path:line` citations into
  `utils/torsion.py` and `utils/featurization.py`).
- Exact quote:
  > Conversely, non-ring double bonds count as "rotatable" (get_transformation_mask has no bond-order check), so the model CAN flip E/Z. Evaluation uses GetBestRMS (proper rotations only), so a wrong enantiomer/diastereomer is never matched.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: once the geometry factor moves atoms, chirality is no longer
  frozen, so a chirality constraint becomes mandatory in V2/V3 where TD needed none; the E/Z issue already exists in
  V1 and belongs to Exp 3 (R-CHEM), so Exp 1 should *measure* it and not re-solve it.
- Supports card(s): EXP-1 §2.8, §3
- Confidence: high

### E-JOINT-132
- Claim: ET-Flow applies a post hoc chirality correction to generated conformers.
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 5 (§3.5).
- Shows vs infers: SHOWS (as recorded in E-ARCH-050, VERIFIED round 4). **The exact sentence was not re-extracted in
  round 5; the mechanism is cited only as "published precedent exists for a post hoc stereo repair" and the spec
  marks it UNVERIFIED-IN-R5.**
- Supports card(s): EXP-1 §2.8
- Confidence: low (not re-checked this round)

### E-JOINT-133
- Claim: A 2026 audit of GEOM-DRUGS removed 2.3–7.0 % of submitted conformers from five public generators because of
  reflections, incorrect stereochemistry and post-relaxation connectivity changes, and found settled low-energy
  structures outside the 0.75 Å reference radius; augmenting the references with conformers from held-out
  architectures raised coverage-precision for every generator by 1.48–2.99 points.
- Source type: paper
- Locator: paper: `papers/related/2026_geom_drugs_missing_conformers.pdf` (arXiv 2610.12523), PDF p. 1 (Abstract).
- Exact quote:
  > settled, low-energy structures remain outside the 0.75 Å reference radius after removing reflections, incorrect stereochemistry, and post-relaxation connectivity changes (2.3-7.0% of submitted conformers, depending on architecture).
- Shows vs infers: SHOWS (GEOM-DRUGS, five generators). NEW in round 5. INFERENCE, directly usable: (i) reflection,
  stereochemistry and **post-relaxation connectivity change** are the three screens a 2026 reviewer expects before
  any RMSD or energy number is reported, so they belong in the Exp-1 protocol; (ii) precision metrics are limited by
  reference incompleteness, which is a further reason to lead with AMR-R/COV-R and internal-geometry metrics rather
  than COV-P.
- Supports card(s): EXP-1 §3, §4
- Confidence: high

### E-JOINT-134
- Claim: GEOM-QM9 reference conformers are GFN2-xTB-level structures (QM9 geometries were re-optimised with xTB
  before seeding CREST), so a local structure relaxed with GFN2-xTB targets the reference level of our ground truth.
- Source type: paper
- Locator: paper: `papers/related/2022_axelrod_geom.pdf`, PDF p. 5.
- Exact quote:
  > to seed CREST with a structure optimized at the GFN2-xTB level of theory, we re-optimized each QM9 geometry with xTB before using it in CREST.
- Shows vs infers: SHOWS. Reused from E-ARCH-014 / round-3 E-GEOM-015 (VERIFIED); **re-checked in round 5.**
  INFERENCE: the xTB-relaxed-seed arm is therefore not a hack but a *level-matched* baseline, and it is the control
  that Exp 1 must beat to claim anything (see the "winning by more optimisation" controls).
- Supports card(s): EXP-1 §3, §4
- Confidence: high

### E-JOINT-135
- Claim: Post-hoc physical relaxation after diffusion trades recall for precision in our own measurements: MMFF after
  TD (arm A3) gives AMR-R 0.1863 Å vs 0.1752 for TD and AMR-P 0.1641 vs 0.2191.
- Source type: our-data
- Locator: our-data: round-1 A3 run as cited in `round3/ledger/evidence_EVAL.md` (E-EVAL-041, VERIFIED by V1).
- Exact quote (E-EVAL-041 claim):
  > MMFF relaxation after diffusion (A3) worsens recall (AMR-R 0.1863 Å) but improves precision (AMR-P 0.1641 Å vs 0.2191 for R0).
- Shows vs infers: SHOWS. Reused from E-ARCH-013. INFERENCE: "just relax the output" is not a free win on the primary
  endpoint, which is why the relaxation arms must be reported separately and never folded into the primary arm.
- Supports card(s): EXP-1 §3, §4
- Confidence: high

### E-JOINT-136
- Claim: Cascaded diffusion models fail when a second stage trained on ground-truth conditioning is fed first-stage
  samples, and the published fix is Gaussian "conditioning augmentation", optionally amortised by feeding the
  augmentation level to the model as an extra embedding.
- Source type: paper
- Locator: paper: `papers/related/2021_ho_cascaded_diffusion.pdf` (arXiv 2106.15282), PDF p. 3, p. 6, p. 18.
- Shows vs infers: SHOWS (images). Reused from the round-2 index row and round-3 ROBUST ledger. **Quote not
  re-extracted in round 5; cited as background for why V2/V3 must not be trained as a naive cascade.** Our own
  attempt at the conditioning-augmentation idea (S2, σ 0.04 Å L-jitter) made things worse by +0.025 Å
  (`round4/BRIEF.md`), which is the in-domain counter-evidence.
- Supports card(s): EXP-1 §4, §7
- Confidence: low (not re-checked this round); the S2 counter-evidence is high confidence

### E-JOINT-137
- Claim: DAgger: when the learner's outputs determine its future inputs the i.i.d. assumption fails, and the fix is
  to train on the state distribution the learned policy itself induces, labelled by the expert.
- Source type: paper
- Locator: paper: `papers/related/2011_ross_dagger.pdf`, PDF p. 1 (§1) and PDF p. 3 (§3 "Dataset Aggregation").
- Exact quote:
  > DAGGER (Dataset Aggregation), an iterative algorithm that trains a deterministic policy that achieves good performance guarantees under its induced distribution of states.
- Shows vs infers: SHOWS. Reused from E-ARCH-062 / round-3 E-ROBUST-044 (VERIFIED, V2). INFERENCE: the formal name
  for the stage-2 procedure V2/V3 require, with S5 (E-JOINT-006) as our in-domain evidence that it is the only thing
  that has worked.
- Supports card(s): EXP-1 §2.2, §4, §5
- Confidence: high

---

## H. Evaluation protocol sources

### E-JOINT-150
- Claim: PuckerFlow evaluates rings with a **puckering RMSD** (atomic displacements from the ring mean plane) in
  addition to all-atom RMSD.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 7 (§2.4).
- Exact quote:
  > puckering RMSD, which measures the ability of the methods to reproduce the characteristic ring puckering by quantifying the atomic displacements from the mean plane;
- Shows vs infers: SHOWS (on their own ring set; never applied to TD/ET-Flow/DiTMC output on GEOM).
  Reused from E-PRIOR-037 (VERIFIED). INFERENCE: adopting this metric on GEOM is cheap and makes the ring claim
  directly comparable to the closest prior art.
- Supports card(s): EXP-1 §3
- Confidence: high

### E-JOINT-151
- Claim: GEOM-Drugs-Revisited (Digital Discovery 2025) documents evaluation bugs in the GEOM benchmark and recommends
  GFN2-xTB relaxation energies (E_relax) and Δ bond-length / bond-angle / torsion metrics.
- Source type: paper
- Locator: paper: `papers/related/2025_nikitin_geom_drugs_revisited.pdf`, PDF p. 10 (and the repo snapshot
  `papers/blogs/geom_revisited_repo.txt`).
- Shows vs infers: SHOWS (as recorded in `papers/INDEX.md` and round-3 SCOUT EVAL). **Quote not re-extracted in
  round 5; the metric list is used, the exact sentence is marked UNVERIFIED-IN-R5 and must be re-checked by the
  verifier before it appears in a paper.**
- Supports card(s): EXP-1 §3
- Confidence: low (not re-checked this round)

### E-JOINT-152
- Claim: CurvFM reports a 63 % reduction in the median deviation of the maximum force on GEOM-QM9 at unchanged global
  structural accuracy, i.e. physical-fidelity metrics can move a lot while RMSD metrics do not move at all.
- Source type: paper
- Locator: paper: `papers/related/2026_darouich_curvfm.pdf` (arXiv 2609.34592), PDF p. 1 (Abstract).
- Exact quote:
  > CurvFM substantially improves physical fidelity at unchanged global structural accuracy, reducing the median deviation in maximum force by 63% on GEOM-QM9
- Shows vs infers: SHOWS (own split, not TD-P). Reused from E-ARCH-071 / E-PRIOR-038 (VERIFIED). INFERENCE: the
  reverse is also possible — Exp 1 could improve bond/angle fidelity substantially and move AMR-R not at all. The
  protocol must therefore pre-declare both an RMSD endpoint and a physical-fidelity endpoint, and must not reinterpret
  one as the other after the fact.
- Supports card(s): EXP-1 §3, §4, §7
- Confidence: high

### E-JOINT-153
- Claim: TD reports median AMR and runtime in **core-seconds per conformer** on CPU (TD at 5/10/20 steps: 1.76 /
  2.82 / 4.90 core-s, against RDKit 0.10 and GeoDiff 305) and ensemble properties (E, μ, Δε, E_min) after GFN2-xTB
  relaxation of a random 100-molecule DRUGS subset with min(2K, 32) conformers each.
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 9 (§4.4, Tables 2 and 3).
- Exact quote:
  > Torsional 5 0.685 0.963 1.76
  > we generate min(2K, 32) conformers per molecule, relax the conformers with GFN2-xTB [Bannwarth et al., 2019],9 and compare the Boltzmann-weighted properties of the generated and ground truth ensembles.
- Shows vs infers: SHOWS. NEW in round 5. INFERENCE: these are the two comparability requirements for Exp 1's
  timing and energy sections — core-seconds per conformer (not GPU wall-clock) and xTB-relaxed Boltzmann-weighted
  properties.
- Supports card(s): EXP-1 §3
- Confidence: high

### E-JOINT-154
- Claim: Recall-style coverage metrics on these benchmarks are gameable by sample budget: RDKit plus oversampling and
  clustering beats most deep models on recall (Zhou et al. 2023), and an even more random "Infinite Stochastic
  Monkey" sampler also scores highly with ≈ 2000 samples, so a deep model is only fairly compared at an equal
  sample-and-cluster budget.
- Source type: paper
- Locator: paper: `papers/related/2023_zhou_dl_conformation_critique.pdf`; `papers/related/2023_zhang_infinite_physical_monkey.pdf`,
  PDF p. 5 and p. 9.
- Shows vs infers: SHOWS (GD-P protocol in both). Reused from the round-3 SCOUT EVAL index rows. **Quotes not
  re-extracted in round 5; the budget-matching requirement is adopted and the exact sentences are marked
  UNVERIFIED-IN-R5.**
- Supports card(s): EXP-1 §4 (controls)
- Confidence: medium (claim well established in our index; quote not re-checked)

### E-JOINT-155
- Claim: S23D argues the GEOM-QM9 benchmark at δ = 0.5 Å may no longer be appropriate for testing state-of-the-art
  conformer generators, and the flow-matching refiner reports QM9 coverage at δ = 0.05 Å instead.
- Source type: paper
- Locator: paper: `papers/related/2025_gurev_s23d.pdf`, PDF p. 16; `papers/related/2025_xu_fm_refiner.pdf`, PDF p. 7.
- Exact quote:
  > the QM9 benchmark may no longer be appropriate for testing SOTA MCG models.
- Shows vs infers: SHOWS (author opinion). Reused from E-PRIOR-008 (VERIFIED). INFERENCE: COV-R@0.1 on the success
  intersection (our pre-registered key secondary) and the internal-geometry metrics carry the weight of any QM9-only
  claim; GEOM-DRUGS is required for a main venue.
- Supports card(s): EXP-1 §3, §7
- Confidence: high

---

## I. Novelty verdicts (summary; the reasoning is in the specs)

| Claim | Verdict | Decisive prior art |
|---|---|---|
| "Relax TD's rigid local structure / model internal coordinates jointly with torsions" (as an idea) | **PRIOR ART** | TD's own conclusion [E-JOINT-047]; Corso's thesis [E-JOINT-078]; RINGER [E-JOINT-060]; FoldingDiff [E-JOINT-062]; GO-Flow [E-JOINT-063]; PuckerFlow's future work [E-JOINT-069]; TMCgen [E-JOINT-070] |
| "Joint bond-angle + torsion diffusion" | **PRIOR ART** | RINGER 2023, for macrocycles [E-JOINT-060]; FoldingDiff, for protein backbones [E-JOINT-062] |
| "Joint ring-pucker + torsion generation for drug-like small molecules, benchmarked on GEOM/TD protocol, including polycyclic systems" | **NOVEL but contested** | none found [E-JOINT-079]; announced as future work by PuckerFlow's group [E-JOINT-069]; INCREMENTAL over PuckerFlow + GO-Flow without a real-use win |
| The L-subspace projection construction (a Cartesian geometry factor made orthogonal to the torsion and rigid directions using TD's own likelihood Jacobian) | **NOVEL (method detail)** | GO-Flow projects the other way, internal → Cartesian [E-JOINT-063]; DiffDock's factors are disjoint by construction, not projected [E-JOINT-072]; no precedent found [E-JOINT-079] |
| Evidence that coupling L to τ is *necessary* (leak-free split-half test + the decoupled-ablation control) | ~~NOVEL~~ → **HYPOTHESIS, not a finding (revised after D-001/D-003/D-005)**: the leak-free result is explained by L-distribution mismatch [E-JOINT-010 rev.]; only arm A6b can test coupling | TD App. F.1 uses other-conformer L only as a matching floor, never as a model test [E-JOINT-049] |
| (revised after D-004) The L-subspace projector gives exact separation from torsions | **WITHDRAWN as stated**: P_L projects out the torsion tangents, not the dihedral gradients [E-JOINT-032]. Replaced by the ker-W projector; a "first-order dihedral-preserving geometry step" is a minor method detail | — |
| "First multi-scale / hierarchical conformer generator" | **PRIOR ART — do not claim** | EBD [E-JOINT-090]; HierDiff [E-JOINT-093]; MSGEN [E-JOINT-095]; OctMole [E-JOINT-094] |
| "A learned local-structure correction to RDKit conformers" | **PRIOR ART** | CoarsenConf [E-JOINT-075]; EBD (fragment centroids only) [E-JOINT-091]; FM-refiner [E-JOINT-077] |
| "A Boltzmann generator that models bond lengths and angles" | **PRIOR ART** | Noé et al. 2019 [E-JOINT-110]; Kim et al. ICLR 2024 (`papers/INDEX.md` has no PDF for this one — abstract only) |
| "A *transferable* Boltzmann generator over full Euclidean coordinates" | **PRIOR ART** | Transferable Boltzmann Generators, dipeptides [E-JOINT-111] |
| "A transferable Boltzmann generator over p(L, τ) with exact likelihoods for drug-like molecules on GEOM" | **NOVEL, feasibility unproven** | the gap is stated by TBG [E-JOINT-111] and by Corso's thesis [E-JOINT-112]; blocked by divergence cost [E-JOINT-043, E-JOINT-113] and ESS collapse with dimension [E-JOINT-114] |
