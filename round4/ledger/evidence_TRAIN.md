# Evidence ledger: SCOUT TRAIN (round 4, P1)

Lens: training objectives, data and robustness that let a torsion (or joint L + torsion) model exploit good local
structure L and survive imperfect L (round4/BRIEF.md Q2). Written 2026-10-11 by scout TRAIN. Paths are relative to
`C:\Users\HP\Desktop\tor_diff`.

Conventions
- Paper pages are PDF pages; printed page = PDF page unless stated. Quotes are matched after collapsing whitespace.
  Glyphs lost in extraction are noted under the quote. Fulltexts added this round are `pdftotext` reading-order
  extractions with form feeds between pages.
- Round-2 numbers are PRELIMINARY (unpaired means) unless a round-3 entry says otherwise. ORACLE marks conditions built
  from test-set true geometry; none is used for model selection.
- Entries reused from round 3 carry `Reused from: <ID> (VERIFIED in <file>)`; their quotes were re-checked against the
  local fulltext / result file in this round.
- Status: IN PROGRESS (skeleton written first, per README rule 6).
- Framing update (coordinator, 2026-10-12, round4/BRIEF.md "Round-2 FINAL paired analysis"): S3 is significantly
  worse than CTRL on RDKit L and much worse on MMFF L; it wins only on own-conformer true L (suspected leak). The lens
  question is therefore: how do we train a torsion model that GAINS from improved-but-imperfect L (MMFF, xTB, learned)?
- Round-2 FINAL numbers are paired (molecule bootstrap, Holm over the primary family) and come from
  `cluster_sync/round2/results/analysis_r2/contrasts.csv`. To make quotes checkable, the MAT-R / COV-R@0.1 union rows
  were extracted verbatim by `round4/ledger/train_data/extract_contrasts.py` (reads only that CSV; rerun from the repo
  root) into `round4/ledger/train_data/contrasts_extract.txt`; every extract line starts with its CSV line number.
  MAT-R in that file is AMR-R (mean). Arm/ref labels: CR = CTRL_rematch, CB = CTRL_base, B6 = S2 (σ 0.04 Å),
  B3 = S5 (MMFF-matched training).

---------------------------------------------------------------------------------------------------------------------
## A. Our data and code

### E-TRAIN-001
- Claim: In the final paired analysis, S3 (50/50 RDKit-matched / own-GT L) is significantly worse than CTRL_rematch on
  RDKit L: AMR-R 0.1819 vs 0.1786, +0.0032 Å [+0.0013, +0.0052], Holm p = 0.0096 (2 vs 3 seeds, 934 molecules).
- Source type: our-data
- Locator: our-data: cluster_sync/round2/results/analysis_r2/contrasts.csv line 50 (contrast S3.rdkit_vs_CR, MAT-R,
  union), extracted in round4/ledger/train_data/contrasts_extract.txt
- Exact quote (extract line):
  > line 50: S3.rdkit_vs_CR MAT-R n=934 S3:steps20_seed0 vs CR:steps20_seed0 arm=0.1819 ref=0.1786 diff=+0.0032 [+0.0013, +0.0052] seeds=2v3 p_holm=0.0096
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-01, C-TRAIN-02
- Confidence: high

### E-TRAIN-002
- Claim: On MMFF-relaxed ETKDG L (non-oracle), every model trained on true L is worse than CTRL_rematch, paired:
  B1 +0.0798 Å, S3 +0.0394 Å, S2 (noise σ 0.04) +0.0216 Å (CTRL_rematch 0.1525; 935 molecules).
- Source type: our-data
- Locator: our-data: contrasts.csv lines 290, 482, 466 (S1.*_vs_CR.S1_etkdg2L_mmff, MAT-R, union), extracted in
  round4/ledger/train_data/contrasts_extract.txt
- Exact quote (extract lines):
  > line 290: S1.B1_vs_CR.S1_etkdg2L_mmff MAT-R n=935 B1:S1_etkdg2L_mmff vs CR:S1_etkdg2L_mmff arm=0.2323 ref=0.1525 diff=+0.0798 [+0.0722, +0.0877] seeds=3v3 p_holm=NA
  > line 482: S1.S3_vs_CR.S1_etkdg2L_mmff MAT-R n=935 S3:S1_etkdg2L_mmff vs CR:S1_etkdg2L_mmff arm=0.1919 ref=0.1525 diff=+0.0394 [+0.0338, +0.0454] seeds=2v3 p_holm=NA
  > line 466: S1.B6_vs_CR.S1_etkdg2L_mmff MAT-R n=935 B6:S1_etkdg2L_mmff vs CR:S1_etkdg2L_mmff arm=0.1741 ref=0.1525 diff=+0.0216 [+0.0168, +0.0267] seeds=3v3 p_holm=NA
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-01, C-TRAIN-02
- Confidence: high

### E-TRAIN-003
- Claim: Going from raw ETKDG L to MMFF-relaxed ETKDG L (same seeds, same 935 molecules), CTRL_rematch improves by
  0.0254 Å (0.1779 → 0.1525) and S2 by 0.0325 Å (0.2066 → 0.1741), B1 barely moves (0.2356 → 0.2323), and S3 gets
  WORSE by 0.0105 Å (0.1814 → 0.1919). Only the models never trained on own-conformer GT L at full weight (CTRL; S2,
  whose GT L is always jittered) gain from the better geometry.
- Source type: our-data
- Locator: our-data: contrasts.csv lines 274/290 (B1 and CR values), 434/466 (B6), 450/482 (S3), arm_value and
  ref_value columns, extracted in round4/ledger/train_data/contrasts_extract.txt
- Exact quote (extract lines; the etkdg2L rows):
  > line 274: S1.B1_vs_CR.S1_etkdg2L MAT-R n=935 B1:S1_etkdg2L vs CR:S1_etkdg2L arm=0.2356 ref=0.1779 diff=+0.0578 [+0.0508, +0.0648] seeds=3v3 p_holm=NA
  > line 434: S1.B6_vs_CR.S1_etkdg2L MAT-R n=935 B6:S1_etkdg2L vs CR:S1_etkdg2L arm=0.2066 ref=0.1779 diff=+0.0287 [+0.0227, +0.0348] seeds=3v3 p_holm=NA
  > line 450: S1.S3_vs_CR.S1_etkdg2L MAT-R n=935 S3:S1_etkdg2L vs CR:S1_etkdg2L arm=0.1814 ref=0.1779 diff=+0.0035 [+0.0016, +0.0056] seeds=2v3 p_holm=NA
  (MMFF rows: E-TRAIN-002.)
- Shows vs infers: SHOWS the four deltas (unpaired across conditions, same molecules and seeds; no CI for the
  ETKDG → MMFF change itself). INFERENCE: whatever S3 learned from its own-GT half makes it worse on cleaner but
  non-own L; the leading hypothesis (card C-TRAIN-01) is that S3/B1 read the target's torsions from an L that was
  relaxed at those torsions, and MMFF L carries the imprint of the seed's random torsions instead.
- Supports card(s): C-TRAIN-01, C-TRAIN-02
- Confidence: high (numbers), medium (reading)

### E-TRAIN-004
- Claim: S5 (TD trained on MMFF-relaxed, conformer-matched L) tested with MMFF L beats CTRL_rematch tested with MMFF L
  by −0.0041 Å [−0.0064, −0.0019] (0.1500 vs 0.1540, 3 vs 3 seeds); tested on raw RDKit L it is +0.0031 Å worse than
  CTRL_rematch. This is the only training-recipe change in rounds 1–2 that improved real use.
- Source type: our-data
- Locator: our-data: contrasts.csv lines 194 and 226 (S5.premmff_vs_CRpremmff, S5.nommff_vs_CR; MAT-R, union),
  extracted in round4/ledger/train_data/contrasts_extract.txt
- Exact quote (extract lines):
  > line 194: S5.premmff_vs_CRpremmff MAT-R n=934 B3:steps20_seed0 vs CR:S5ctrl_pre_mmff arm=0.1500 ref=0.1540 diff=-0.0041 [-0.0064, -0.0019] seeds=3v3 p_holm=0.0
  > line 226: S5.nommff_vs_CR MAT-R n=934 B3:S5_nommff vs CR:steps20_seed0 arm=0.1817 ref=0.1786 diff=+0.0031 [+0.0013, +0.0049] seeds=3v3 p_holm=NA
- Shows vs infers: SHOWS ("only recipe that improved real use" is the BRIEF's bottom line, round4/BRIEF.md:41-43)
- Supports card(s): C-TRAIN-02, C-TRAIN-01
- Confidence: high

### E-TRAIN-005
- Claim: Against CTRL_rematch, S3 is better only on exact own-conformer true L (ORACLE, −0.0488 Å); it is worse at
  blend λ = 0.5 (+0.0037), on true L + 0.04 Å noise (+0.0074), and not significantly different on true L + 0.02 Å
  noise (−0.0030 [−0.0063, +0.0004]); all ORACLE cycled conditions.
- Source type: our-data
- Locator: our-data: contrasts.csv lines 66, 610, 546, 514 (MAT-R, union), extracted in contrasts_extract.txt
- Exact quote (extract lines):
  > line 66: S3.gtLc_vs_CR_ORACLE MAT-R n=996 S3:steps20_seed0_gtLcycle vs CR:steps20_seed0_gtLcycle arm=0.0334 ref=0.0822 diff=-0.0488 [-0.0541, -0.0438] seeds=2v3 p_holm=0.0
  > line 610: S1.S3_vs_CR.S1_lam0.50_cyc_ORACLE MAT-R n=906 S3:S1_lam0.50_cyc_ORACLE vs CR:S1_lam0.50_cyc_ORACLE arm=0.1401 ref=0.1364 diff=+0.0037 [+0.0012, +0.0061] seeds=2v3 p_holm=NA
  > line 546: S1.S3_vs_CR.S1_noise0.04pa_cyc_ORACLE MAT-R n=996 S3:S1_noise0.04pa_cyc_ORACLE vs CR:S1_noise0.04pa_cyc_ORACLE arm=0.1260 ref=0.1186 diff=+0.0074 [+0.0044, +0.0104] seeds=2v3 p_holm=NA
- Shows vs infers: SHOWS (ORACLE conditions; context only, never for model selection)
- Supports card(s): C-TRAIN-01
- Confidence: high

### E-TRAIN-006
- Claim: Estimated by inverting random-GT vs cycled runs (round-1 in-job), on multi-conformer molecules B1 scores
  0.027 Å with its own conformer's GT L but an implied 0.249 Å with only other conformers' GT L (≈ its 0.256 on
  RDKit L); CTRL_base scores 0.102 → implied 0.138 (0.189 on RDKit L). The true-L advantage of GT-L-trained models is
  mostly conformer identity.
- Source type: our-data
- Locator: our-data: round3/ledger/v2_data/v2_ntrue_check.out, last two lines ("multi-conformer weighted"); method in
  round3/impact.md §3g
- Exact quote:
  > B1: multi-conformer weighted: cycled 0.027  gtL 0.048  implied other-L 0.249  RDKit in-job 0.256
  > CTRL_base: multi-conformer weighted: cycled 0.102  gtL 0.105  implied other-L 0.138  RDKit in-job 0.189
- Shows vs infers: INFERENCE-heavy: the "implied other-L" values are an estimate that divides by P0 ≈ 0.07–0.13 and
  amplifies noise 8–14×; the validity judge ruled it "an estimate, not a measurement" (round3/panel/
  score_VALIDITY_r1.md V-3). The qualitative B1 reading was accepted there. A leak-free run is in progress (BRIEF).
- Supports card(s): C-TRAIN-01
- Confidence: medium

### E-TRAIN-007
- Claim: CTRL's training L is a fresh ETKDG embedding per molecule, assigned to GT conformers by a Hungarian match on
  a torsion cost, after which only its torsions are optimised (differential evolution) toward the GT conformer. Its
  bond lengths, angles and rings are therefore generated independently of the target conformer.
- Source type: code
- Locator: code: torsional-diffusion/standardize_confs.py:78-79 (embed), :89-93 (cost + assignment), :104-107 (DE on
  rotatable torsions of the assigned embedding)
- Exact quote (standardize_confs.py:79, 93, 106):
  > AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n_confs)
  > row_ind, col_ind = linear_sum_assignment(cost_matrix)
  > optimize_rotatable_bonds(mol_rdkit_single, mol, rotable_bonds,
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-01, C-TRAIN-02, C-TRAIN-06
- Confidence: high

### E-TRAIN-008
- Claim: In S3/B1cap's paired cache, the GT L of each training conformer is that same conformer's own GT geometry
  (looked up by geom_id / energy key and RMSD check), Kabsch-aligned onto the matched RDKit conformer; the transform
  then feeds it with that conformer's own torsion target.
- Source type: code
- Locator: code: tools/build_paired_pickles.py:112 and :124; torsional-diffusion/utils/dataset.py:88 and :98
- Exact quote (build_paired_pickles.py:124; dataset.py:88, 98):
  > c['gt_pos_aligned'] = al['Y_al'].astype(np.float32)
  > x, y = data.pos[k], data.gt_pos[k]
  > data.pos = y if np.random.uniform() < self.l_mix_p_gt else x
- Shows vs infers: SHOWS. Reused in part from E-ROBUST-014 / E-ROBUST-016 (VERIFIED in round3/ledger/verify_V2.md).
- Supports card(s): C-TRAIN-01
- Confidence: high

### E-TRAIN-009
- Claim: At inference, TD keeps each seed's L fixed (optionally MMFF-relaxed at the ETKDG seed's own torsions) and then
  draws every rotatable torsion uniformly, so a deployed L never carries an imprint of the target conformer's torsions,
  only of the seed's original, random ones.
- Source type: code
- Locator: code: torsional-diffusion/diffusion/sampling.py:75 (MMFF on the seeds) and :107 (uniform torsions)
- Exact quote (sampling.py:75, 107):
  > if mmff: try_mmff(mol)
  > torsion_updates = np.random.uniform(low=-np.pi,high=np.pi, size=data_conf.edge_mask.sum())
- Shows vs infers: SHOWS (code). "Imprint" (bond angles relaxed at given torsions encode them) is INFERENCE, consistent
  with TD's own G3 discussion and E-TRAIN-022/023.
- Supports card(s): C-TRAIN-01, C-TRAIN-02
- Confidence: high (code), medium (imprint reading)

### E-TRAIN-010
- Claim: On the released model with RDKit L (R0), the run-matched floor is 0.1195 Å per-molecule mean and the torsion
  headroom (what a perfect torsion model could still remove) only 0.0471 Å per molecule.
- Source type: our-data
- Locator: our-data: cluster_sync/results/qm9_default/R0_base_seed0/floor_run.log lines 10 and 17
- Exact quote:
  > floor_run: n=7190 mean=0.0993 median=0.0765 per-molecule-mean=0.1195
  > torsion_headroom: n=7190 mean=0.0668 median=0.0237 per-molecule-mean=0.0471
- Shows vs infers: SHOWS (upper bound on the floor share, heuristic optimiser; EXPERIMENTS.md E12)
- Supports card(s): C-TRAIN-04, C-TRAIN-05
- Confidence: high

### E-TRAIN-011
- Claim: Changing the conformer-matching recipe did not help on QM9: random L–conformer pairing (B2) +0.0021 Å
  [0.0006, 0.0037] and heavy-atom DE objective (B5) +0.0058 Å [0.0038, 0.0080] vs CTRL_rematch (3 v 3 seeds).
- Source type: our-data
- Locator: our-data: cluster_sync/results/analysis/paired_B_vs_CTRLrematch.md lines 5 and 15
- Exact quote:
  > | B2    | MAT-R      | 934 |  0.1786 |      0.1808 |  0.0021 |  0.0006 |  0.0037 |   0.0096 |             0.0013 | True      |             3 |             3 |   0.0288 |
  > | B5    | MAT-R      | 934 |  0.1786 |      0.1845 |  0.0058 |  0.0038 |  0.008  |   0      |             0.0013 | True      |             3 |             3 |   0      |
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-06
- Confidence: high

### E-TRAIN-012
- Claim: On rigid molecules (0 rotatable heavy-atom torsions, 284 molecules) the AMR-R is set by L, not by the torsion
  model: B1 and CTRL_base agree within 0.003 Å on ETKDG L (0.1553 vs 0.1530) and on MMFF L (0.1273 vs 0.1280).
- Source type: our-data
- Locator: our-data: round3/impact_data/cond_means.txt lines 107-108 and 118-119 (per-bin table, n_rot_heavy = 0)
- Exact quote:
  >    r2       qm9_B1_train_gtL                  S1_etkdg2L 284.0000 0.1553 0.0006
  >    r2       qm9_B1_train_gtL             S1_etkdg2L_mmff 284.0000 0.1273 0.0025
  >    r2    qm9_CTRL_base_100ep                  S1_etkdg2L 284.0000 0.1530    NaN
  >    r2    qm9_CTRL_base_100ep             S1_etkdg2L_mmff 284.0000 0.1280    NaN
- Shows vs infers: SHOWS (unpaired, PRELIMINARY round-2 per-bin means; round3/impact.md §3a)
- Supports card(s): C-TRAIN-05
- Confidence: high

### E-TRAIN-013
- Claim: ETKDG L has 49.6 % of ring-≥4 seeds with endocyclic-dihedral error > 10°; MMFF relaxation lowers this only to
  35.1 %, and on rigid molecules from 57.3 % to 34.9 %.
- Source type: our-data
- Locator: our-data: round3/impact_data/lerror_profile.out, rows L_etkdg2L / L_etkdg2L_mmff, column tail10_ge4
- Exact quote:
  >                L_etkdg2L     all 12655.0000 936.0000     0.0318      2.3788       10.6564           18.5015      0.2857      0.4961      0.3112     0.0292      2.8355    0.0314     2.7963
  >           L_etkdg2L_mmff     all 12655.0000 936.0000     0.0198      1.4161        8.0687           14.0087      0.2020      0.3506      0.2184     0.0170      2.0322    0.0182     1.9733
- Shows vs infers: SHOWS. Reused from E-ROBUST-006 (VERIFIED, verify_V2.md) for the ETKDG value; the MMFF row is the
  same file (the analyst's and V3's 35.06 %).
- Supports card(s): C-TRAIN-02, C-TRAIN-05
- Confidence: high

### E-TRAIN-014
- Claim: TD's training-side matcher takes any L-carrying molecule as its first argument and fits its rotatable
  torsions to a reference conformer, so a GT sibling conformer (or an xTB-relaxed seed) can replace the ETKDG embedding
  as the L source without new matching code.
- Source type: code
- Locator: code: torsional-diffusion/utils/standardization.py:29 (signature); standardize_confs.py:104-107 (call)
- Exact quote (standardization.py:29):
  > def optimize_rotatable_bonds(mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=0, popsize=15, maxiter=500,
- Shows vs infers: SHOWS (signature). That a sibling GT conformer is a valid `mol` argument (same atom order, same
  graph) is INFERENCE; GEOM conformers of one molecule share atom order in the raw pickles used by
  build_paired_pickles.py (its same_graph check, :108), which the grounder should confirm.
- Supports card(s): C-TRAIN-01, C-TRAIN-02
- Confidence: medium

### E-TRAIN-015
- Claim: TD already has a Boltzmann-weighted conformer sampler for training (`--boltzmann_weight`, default off): the
  training conformer is drawn with probability equal to its CREST Boltzmann weight instead of uniformly.
- Source type: code
- Locator: code: torsional-diffusion/utils/parsing.py:28; torsional-diffusion/utils/dataset.py:53-56 and :87
- Exact quote (parsing.py:28; dataset.py:54):
  > parser.add_argument('--boltzmann_weight', action='store_true', default=False, help='Whether to sample conformers based on B.w.')
  > data.pos = random.choices(data.pos, data.weights, k=1)[0]
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-06
- Confidence: high

### E-TRAIN-016
- Claim: The score model already has a hook for a second conditioning level next to σ: S4's λ is sinusoidally
  embedded and concatenated to the σ embedding of every node and edge (lambda_embed_dim > 0), so a separate L noise
  level t_L can enter the same way.
- Source type: code
- Locator: code: torsional-diffusion/diffusion/score_model.py:194-199 (and :70, :76 layer widths)
- Exact quote (score_model.py:198-199):
  > node_lam_emb = get_timestep_embedding(data.node_lambda * 10000, self.lambda_embed_dim)
  > node_sigma_emb = torch.cat([node_sigma_emb, node_lam_emb], 1)
- Shows vs infers: SHOWS. Reused from E-ROBUST-017 (VERIFIED, verify_V2.md); re-checked.
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-017
- Claim: S4 (blend-level λ as an extra input) is not significantly different from CTRL_rematch on RDKit L
  (+0.0033 Å [−0.0009, +0.0078], Holm p = 0.835); its only gains are on own-conformer cycled ORACLE blends
  (λ = 0.5: −0.0090 Å on the S4 subset).
- Source type: our-data
- Locator: our-data: contrasts.csv lines 98 and 642 (MAT-R, union), extracted in
  round4/ledger/train_data/contrasts_extract.txt
- Exact quote (extract lines):
  > line 98: S4.rdkit_l0_vs_CR MAT-R n=934 S4:steps20_seed0 vs CR:steps20_seed0 arm=0.1820 ref=0.1786 diff=+0.0033 [-0.0009, +0.0078] seeds=3v3 p_holm=0.8351999999999999
  > line 642: S4.lam0.50_vs_CR@S4subset MAT-R n=906 S4:S4_lam0.50_l0.50_cyc_ORACLE vs CR:S1_lam0.50_cyc_ORACLE arm=0.1274 ref=0.1364 diff=-0.0090 [-0.0134, -0.0047] seeds=3v3 p_holm=NA
- Shows vs infers: SHOWS (S4 trains on blends toward the conformer's OWN GT L, so its λ > 0 gains share the identity
  caveat of E-TRAIN-005/006: INFERENCE)
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-018
- Claim: A leak-free oracle test now exists in the repo: GT conformers are split into alternating halves A/B; seeds
  from one half generate conformers scored only against the other half ("split"), next to the same-half "own" runs, so
  leak = own − split is paired and needs no P0 inversion.
- Source type: code
- Locator: code: tools/make_split_half.py:1-30 (docstring); analysis driver tools/analyze_leakfree.py:1-15
- Exact quote (make_split_half.py docstring):
  > split_AtoB : seeds = A's GT conformers (cycled), 2|B| generated (csv_B), recall vs B only (true_B)   own L absent
- Shows vs infers: SHOWS (tool; results not yet available to this scout). It is the natural primary ORACLE test for
  "uses reference-level, non-own L".
- Supports card(s): C-TRAIN-01
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## B. Papers

### B1. Torsional Diffusion itself (core paper)

### E-TRAIN-020
- Claim: TD reports that matching a GT conformer to the local structure of ANOTHER randomly chosen GT conformer of the
  same molecule gives an average RMSDmin of 0.284 Å on GEOM-DRUGS, below the 0.324 Å obtained with RDKit L, i.e.
  reference-level L from a sibling conformer is a better (and leak-free) L source than RDKit.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 21, Appendix F.1 "RDKit local structures"
- Exact quote:
  > it is only slightly larger than the average RMSDmin of 0.284 Å resulting from matching a ground truth conformer to the local structure of another randomly chosen ground truth conformer
  (Å is extracted as U+FFFD; "RMSDmin" subscript flattened)
- Shows vs infers: SHOWS (a matching floor, no trained model; DRUGS). Using sibling L as TRAINING input is INFERENCE.
- Supports card(s): C-TRAIN-01
- Confidence: high

### E-TRAIN-021
- Claim: TD's model trained directly on GT L has a much LOWER training and validation score-matching loss than the
  conformer-matched model, yet performs worse at inference on RDKit L.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 26, Appendix H "Ablation experiments", item 5
- Exact quote:
  > although the training and validation score matching loss of this model is significantly lower, its inference performance reflects the detrimental effect of the local structure distributional shift.
- Shows vs infers: SHOWS. INFERENCE: a much lower loss is what one expects if own-conformer GT L lets the model read
  the target torsions off L (the identity shortcut of E-TRAIN-006); TD attributes it only to the L shift.
- Supports card(s): C-TRAIN-01
- Confidence: high (quote), medium (reading)

### E-TRAIN-022
- Claim: TD: training on true L creates a test-time shift that significantly hurts performance.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 7, Section 4.1 "Conformer matching"
- Exact quote:
  > there will be a distributional shift at test time, where only approximate local structures from p^G(L) are available. We found that this shift significantly hurts performance.
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-020 (VERIFIED in round3/ledger/verify_V2.md); quote re-checked in the core fulltext p. 7
- Supports card(s): C-TRAIN-01, C-TRAIN-02
- Confidence: high

### E-TRAIN-023
- Claim: TD's conformer matching assigns sampler L to GT conformers so that training and inference see the same L
  distribution.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 19, Appendix E
- Exact quote:
  > The complete assignment resulting from the linear sum solution guarantees that there is no distributional shift in the local structures seen during training and inference.
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-021 (VERIFIED in round3/ledger/verify_V2.md); re-checked p. 19
- Supports card(s): C-TRAIN-02
- Confidence: high

### E-TRAIN-024
- Claim: In TD's own ablation, random pairing of RDKit L with GT conformers (DE torsion fit only, no optimal
  assignment) is only marginally worse than full conformer matching.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 26, Appendix H "Ablation experiments", item 3
- Exact quote:
  > this performs only marginally worse than full conformer matching.
- Shows vs infers: SHOWS (DRUGS; Table 8 p. 27)
- Supports card(s): C-TRAIN-06
- Confidence: high

### E-TRAIN-025
- Claim: TD's matching study: the optimal assignment before differential evolution improves the average matched RMSD
  by 15 % over random pairing (DRUGS, 300 molecules).
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 20, Appendix E, text above Table 5
- Exact quote:
  > In "Conformer matching," the cost-minimizing assignment prior to differential evolution provides a 15% improvement in average RMSD.
- Shows vs infers: SHOWS (a matching-RMSD gain, not a model gain; E-TRAIN-024 shows the model gain is marginal)
- Supports card(s): C-TRAIN-06
- Confidence: high

### E-TRAIN-026
- Claim: TD trains on at most the first 30 CREST conformers per molecule, typically those with the largest Boltzmann
  weight.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 17, Appendix C
- Exact quote:
  > we limit KG to 30 i.e. we only consider the first 30 conformers found by CREST (typically those with the largest Boltzmann weight).
  (subscript G flattened)
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-06
- Confidence: high

### E-TRAIN-027
- Claim: AMR-R averages, with equal weight, over every GT conformer the minimum RMSD to any generated conformer; it does
  not weight GT conformers by Boltzmann weight.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 24, Appendix G, Eq. (35)
- Exact quote (math glyphs lost in extraction):
  > For K = 2L let {Cl}l[1,L] and {Ck}k[1,K] be respectively the sets of ground truth and generated conformers
  > AMR-R := 1 min RMSD(Ck, Cl) (35) L k[1..K ] l[1..L]
  (Eq. 35 reads AMR-R = (1/L) Σ_{l∈[1..L]} min_{k∈[1..K]} RMSD(C_k, C_l))
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-06
- Confidence: high

### B2. Training on the deployed upstream model's outputs, and identity leaks

### E-TRAIN-030
- Claim: ESM-IF1 (inverse folding): AlphaFold2-predicted backbones, even after Amber relaxation, carry sub-Ångström
  artefacts that can give away the target (amino-acid identity).
- Source type: paper
- Locator: paper: papers/related/2022_hsu_esm_if1.pdf (bioRxiv 2022.04.10.487779 v2; ICML 2022), PDF p. 15,
  Appendix A.1 "Noise on AlphaFold2-predicted backbone coordinates"
- Exact quote:
  > Even after Amber relaxation, the backbone coordinates predicted by AlphaFold2 contain artifacts in the sub-Angstrom scale that may give away amino acid identities.
- Shows vs infers: SHOWS. INFERENCE (analogy): an L relaxed at given torsions carries their imprint in the same way
  (E-TRAIN-009); own-conformer GT L is the extreme case.
- Supports card(s): C-TRAIN-01
- Confidence: high (quote), medium (analogy)

### E-TRAIN-031
- Claim: ESM-IF1 adds 0.1 Å Gaussian noise to predicted backbones specifically so that the model does not learn
  generator-specific rules, after seeing a gap between performance on predicted and on experimental structures.
- Source type: paper
- Locator: paper: papers/related/2022_hsu_esm_if1.pdf, PDF p. 15, Appendix A.1 (same paragraph)
- Exact quote:
  > To prevent the model from learning non-generalizable AlphaFold2-specific rules, we added Gaussian noise at the 0.1A scale on predicted backbone coordinates.
  (Å extracted as "A")
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-01, C-TRAIN-02
- Confidence: high

### E-TRAIN-032
- Claim: ESM-IF1 trains on millions of upstream-model (AlphaFold2) predictions mixed with experimental structures, and
  a higher share of predicted structures improves held-out performance.
- Source type: paper
- Locator: paper: papers/related/2022_hsu_esm_if1.pdf, PDF p. 6, Figure 6 caption (b)
- Exact quote:
  > Effect of increasing the mixing ratio during training between predicted and experimental structures. A higher ratio of predicted structures improves performance for both GVP-GNN-large and GVP-Transformer.
- Shows vs infers: SHOWS (protein inverse folding; the predicted structures are inputs, the native sequence the
  target). Transfer to "train the torsion model on deployed-sampler L" is INFERENCE.
- Supports card(s): C-TRAIN-02
- Confidence: high (quote), medium (transfer)

### E-TRAIN-033
- Claim: ProteinMPNN attributes the advantage of models trained on exact backbones to target information hidden in
  exact coordinates ("memory") that predicted structures lack.
- Source type: paper
- Locator: paper: papers/related/2022_dauparas_proteinmpnn.pdf, PDF p. 7
- Exact quote:
  > Crystallographic refinement may impart some memory of amino acid identity in the backbone coordinates which is recovered by the model trained on perfect backbones but not present in predicted structures
- Shows vs infers: SHOWS (hedged "may"). Analogy to own-conformer L is INFERENCE.
- Reused from: E-ROBUST-032 (VERIFIED in round3/ledger/verify_V2.md); re-checked p. 7
- Supports card(s): C-TRAIN-01
- Confidence: high (quote), medium (analogy)

### E-TRAIN-034
- Claim: HarmonicFlow/FlowSite add noise to the pocket cutoff distances and centre partly to stop the model from
  inferring the target (ligand positions) from how its conditioning input was constructed.
- Source type: paper
- Locator: paper: papers/related/2023_stark_harmonicflow.pdf (arXiv 2310.05764 v4), PDF p. 6, Section 4.1 "Datasets"
- Exact quote:
  > The motivation for the noisy distance cutoffs and pocket center is to alleviate distribution shifts during inference time and to prevent the models from inferring the ligand positions from the cutoff with which the pocket was constructed.
  (two-column page: contiguous in the right column of `pdftotext -layout -f 6 -l 6`, lines 22-26; the reading-order
  fulltext interleaves the left column's "3. Ablations and Flow Matching Investigations" list)
- Shows vs infers: SHOWS (a conditioning-construction leak named and blocked by the TD authors' group). Analogy to L
  constructed from the target conformer is INFERENCE.
- Supports card(s): C-TRAIN-01
- Confidence: high (quote), medium (analogy)

### E-TRAIN-035
- Claim: DynamicBind argues that decoys made by Gaussian noise teach a model only to revert to the pre-noise
  structure, which does not prepare it for real upstream (AlphaFold) inputs.
- Source type: paper
- Locator: paper: papers/related/2024_lu_dynamicbind.pdf (Nat. Commun. 15:1071, 2024), PDF p. 2, "DynamicBind
  architectures"
- Exact quote:
  > When decoys are generated using Gaussian noise, the model primarily learns only to revert to the most chemically stable conformation, often the conformation before the noise was added.
- Shows vs infers: SHOWS (an argument, protein–ligand docking). INFERENCE: consistent with our S2 (noise) result
  (E-TRAIN-002/003: S2 gains from MMFF but stays worse than CTRL).
- Supports card(s): C-TRAIN-02, C-TRAIN-01
- Confidence: medium (argument, no ablation quoted)

### E-TRAIN-036
- Claim: DynamicBind instead builds training decoys by morphing the native structure toward the upstream model's
  (AlphaFold) prediction for the same protein, i.e. its training inputs come from the deployed upstream source.
- Source type: paper
- Locator: paper: papers/related/2024_lu_dynamicbind.pdf, PDF p. 2, "DynamicBind architectures"
- Exact quote:
  > our method employs a morph-like transformation to produce protein decoys. In the process of doing this, the native conformation is gradually transitioned towards the AlphaFold-predicted conformation.
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-02
- Confidence: high

### E-TRAIN-037
- Claim: Equivariant Blurring Diffusion names the mismatch between GT fragment coordinates and RDKit-approximated ones as
  a distribution shift that can harm inference.
- Source type: paper
- Locator: paper: papers/related/2024_park_equivariant_blurring_diffusion.pdf, PDF p. 5, Section 3.3
- Exact quote:
  > However, there exists a mismatch between the ground truth coordinates xf and the approximation coordinates x^f from RDKit in the generative processes. This distributional shift of the fragment structure can potentially harm the performance during the inference.
  (hat and superscripts flattened in extraction)
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-02
- Confidence: high

### E-TRAIN-038
- Claim: EBD trains on real RDKit conformers as its coarse input: it optimally assigns RDKit conformers to GT
  conformers (linear sum assignment) and aligns each GT conformer to its assigned RDKit conformer.
- Source type: paper
- Locator: paper: papers/related/2024_park_equivariant_blurring_diffusion.pdf, PDF p. 17, Appendix C.1
- Exact quote:
  > we assigned optimal RDKit conformer to each ground truth conformer using linear sum assignment problem [6]. After finding the optimal matching, we aligned each ground truth conformer to its assigned RDKit conformer using the Kabsch algorithm
- Shows vs infers: SHOWS (GD-P protocol, 5 conformers per molecule)
- Supports card(s): C-TRAIN-02
- Confidence: high

### E-TRAIN-039
- Claim: EBD reports that, trained this way, the model corrects RDKit fragment coordinates more the worse the RDKit
  prior is.
- Source type: paper
- Locator: paper: papers/related/2024_park_equivariant_blurring_diffusion.pdf, PDF p. 8, "Correction on fragment
  coordinates" (Fig. 5 text)
- Exact quote:
  > In other words, the lower
  > the quality of the coarse-grained prior, the more ac- curately the model tend
  (the Fig. 5 caption sits between the two fragments in the extraction; "ac- curately" is a line-break hyphen; the
  sentence continues "...s to correct" after the figure)
- Shows vs infers: SHOWS (200 DRUGS molecules, fragment-centroid RMSD)
- Supports card(s): C-TRAIN-02
- Confidence: medium (sentence split by a figure in the extraction)

### E-TRAIN-040
- Claim: Ringer's retraining of TD on macrocycles used training L derived from the GT xTB conformers after MMFF94
  optimisation, and RDKit L at inference.
- Source type: paper
- Locator: paper: papers/related/2023_grambow_ringer.pdf, PDF p. 19, SI S.11 "Conformer Generation Baselines",
  paragraph "Torsional Diffusion"
- Exact quote:
  > we seeded the training local structures using the ground-truth xTB-optimized conformers with subsequent MMFF94 force-field optimization. At inference time, we seeded the local structures by generating RDKit conformers
- Shows vs infers: SHOWS (no ablation of this choice is reported). INFERENCE: relaxing the GT with the deployed force
  field moves training L toward the deployed L distribution but keeps the own-conformer imprint.
- Supports card(s): C-TRAIN-02, C-TRAIN-01
- Confidence: high (quote), low (effect)

### E-TRAIN-041
- Claim: DAgger trains on the input distribution the deployed policy induces, labelled by the expert.
- Source type: paper
- Locator: paper: papers/related/2011_ross_dagger.pdf, PDF p. 3, Section 3 "Dataset Aggregation"
- Exact quote:
  > DAGGER (Dataset Aggregation), an iterative algorithm that trains a deterministic policy that achieves good performance guarantees under its induced distribution of states.
- Shows vs infers: SHOWS (sequential decisions; here the L sampler does not depend on the torsion model, so only the
  "train on the deployed inputs, expert-labelled" half transfers; conformer matching is the expert relabelling: INFERENCE)
- Reused from: E-ROBUST-045 (VERIFIED in round3/ledger/verify_V2.md)
- Supports card(s): C-TRAIN-02
- Confidence: high (quote), medium (transfer)

### E-TRAIN-042
- Claim: Cascaded diffusion: conditioning augmentation works because it alleviates compounding error from train–test
  mismatch between stages.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 3, Section 1
- Exact quote:
  > We empirically find that conditioning augmentation is effective because it alleviates compounding error in cascading pipelines due to train-test mismatch
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-023 (VERIFIED in round3/ledger/verify_V2.md)
- Supports card(s): C-TRAIN-02, C-TRAIN-03
- Confidence: high

### E-TRAIN-043
- Claim: Cascaded diffusion: the right augmentation depends on the upstream error; Gaussian noise helped at low
  resolution but similar augmentation gave negative results at 128×128/256×256.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 18, Section 4.4
- Exact quote:
  > While we found Gaussian noise augmentation to be a key ingredient to boost the performance of our cascaded models at low resolutions, our initial experiments with similar augmentations for 128×128 and 256×256 upsampling yielded negative results.
  (× extracted as U+FFFD)
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-025 (VERIFIED in round3/ledger/verify_V2.md)
- Supports card(s): C-TRAIN-05
- Confidence: high

### E-TRAIN-044
- Claim: Flow-matching refiner (2025): trained on GT plus isotropic Gaussian noise (a data-centred base distribution),
  then applied to samples of other upstream conformer models.
- Source type: paper
- Locator: paper: papers/related/2025_xu_fm_refiner.pdf (arXiv 2510.04878), PDF p. 4, Section 4.1 "Refiner definition"
- Exact quote:
  > Since our goal is to refine upstream conformers rather than regenerate them, we depart from the usual pure-noise initialization and adopt a base distribution x0 that retains the ground-truth signal.
- Shows vs infers: SHOWS (Cartesian refiner; the next line defines x0 = x1 + σε)
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-045
- Claim: The flow-matching refiner is claimed to be plug-and-play across upstream conformer generators with no
  per-model tuning.
- Source type: paper
- Locator: paper: papers/related/2025_xu_fm_refiner.pdf, PDF p. 2, contributions list item 4
- Exact quote:
  > The proposed refiner is plug-and-play across the conformer generated by different upstream models and needs no per-model tuning.
- Shows vs infers: SHOWS (claim; upstream ET-Flow/DMT/MCF errors are small Cartesian errors, unlike RDKit's ring tail:
  INFERENCE)
- Supports card(s): C-TRAIN-03
- Confidence: medium

### E-TRAIN-046
- Claim: Torsional-GFN learns p(τ | G, L) from an energy reward with L drawn from MD simulations (not tied to a target
  conformer) and generalises zero-shot to unseen bond lengths and angles.
- Source type: paper
- Locator: paper: papers/related/2025_ezzine_torsional_gfn.pdf (ICML 2025 GenBio workshop), PDF p. 1, Abstract
- Exact quote:
  > allows for zero-shot generalization to unseen bond lengths and an- gles coming from the MD simulations for such molecules
  (two-column page: in the reading-order fulltext the abstract column is interleaved with the introduction; the quote
  is contiguous in the left column of `pdftotext -layout -f 1 -l 1`, lines 26-29; "an- gles" is a line-break hyphen)
- Shows vs infers: SHOWS (small MD set, energy-trained). INFERENCE: training on L that is independent of any target
  torsions is what makes L-generalisation possible.
- Supports card(s): C-TRAIN-01
- Confidence: medium (workshop paper, small set)

### B3. Joint / two-stage objectives (independent noise levels, truncation, SDEdit)

### E-TRAIN-050
- Claim: Cascaded diffusion's truncated conditioning augmentation stops the upstream reverse process at an
  intermediate step s > 0 and trains the downstream model on the correspondingly noised conditioning input.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 7, Section 3.2 "Truncated Conditioning
  Augmentation"
- Exact quote:
  > Truncated conditioning augmentation refers to truncating the low resolution reverse process to stop at timestep s > 0, instead of 0
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-051
- Claim: Non-truncated conditioning augmentation runs the upstream sampler to completion and then corrupts its sample
  with the same forward noise used in training.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 8, Section 3.3
- Exact quote:
  > in non-truncated conditioning augmentation we always sample z0 using the full, non-truncated low resolution reverse process; then we corrupt z0 using the forward process into zs
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-028 (VERIFIED in round3/ledger/verify_V2.md)
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-052
- Claim: Multiflow trains its joint structure–sequence flow with independent times for the two modalities, so one
  model learns all relative corruption levels.
- Source type: paper
- Locator: paper: papers/related/2024_campbell_multiflow.pdf (ICML 2024, arXiv 2402.04997 v2), PDF p. 6, Section 4
- Exact quote:
  > Our independent t, t~objective enables the model to learn over different relative levels of corruption between the sequence and structure.
  ("t~" is the extraction of t with a tilde; no space before "objective" in the extraction)
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-053
- Claim: Multiflow uses the decoupled times to do conditional generation with the same model, by holding one modality
  clean while the other is generated.
- Source type: paper
- Locator: paper: papers/related/2024_campbell_multiflow.pdf, PDF p. 6, Section 4
- Exact quote:
  > The advantage of training with decoupled time schedules is that we have freedom to arbitrarily sample with any combination of (t, t~). We use this to perform conditional inpainting where one of the modalities is fixed
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-054
- Claim: UniDiffuser samples the noise levels of two modalities independently in training, so one network fits
  marginal, conditional and joint distributions.
- Source type: paper
- Locator: paper: papers/related/2023_bao_unidiffuser.pdf (ICML 2023, arXiv 2303.06555 v2), PDF p. 4, Section 3.1
- Exact quote:
  > tx and ty are uniformly sampled from {1, 2, . . . , T } independently. We call our method UniDiffuser because it captures multiple distributions in a unified way.
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-055
- Claim: Diffusion Forcing trains a denoiser with independent per-token noise levels, so different parts of the
  input can sit at different noise levels at sampling time.
- Source type: paper
- Locator: paper: papers/related/2024_chen_diffusion_forcing.pdf (NeurIPS 2024, arXiv 2407.01392), PDF p. 1, Abstract
- Exact quote:
  > a new training paradigm where a diffusion model is trained to denoise a set of tokens with independent per-token noise levels.
- Shows vs infers: SHOWS (video / planning)
- Supports card(s): C-TRAIN-03
- Confidence: high

### E-TRAIN-056
- Claim: SDEdit: noising an imperfect input and denoising it with a generative SDE trades faithfulness to the input for
  realism; more noise gives more realism and less faithfulness.
- Source type: paper
- Locator: paper: papers/related/2021_meng_sdedit.pdf (ICLR 2022, arXiv 2108.01073), PDF p. 2, Section 1
- Exact quote:
  > when we add more Gaussian noise and run the SDE for longer, the synthesized images are more realistic but less faithful.
- Shows vs infers: SHOWS (images). INFERENCE: noising a deployed L to level t_L before joint denoising is the same
  trade-off (keep the seed's pucker vs move toward the learned L distribution).
- Supports card(s): C-TRAIN-03
- Confidence: high (quote), medium (transfer)

### E-TRAIN-057
- Claim: Removing noise-level conditioning mostly degrades denoisers gracefully, because the level is inferable from
  the input.
- Source type: paper
- Locator: paper: papers/related/2025_sun_noise_conditioning.pdf, PDF p. 1, Abstract
- Exact quote:
  > To our surprise, most models exhibit graceful degradation, and in some cases, they even perform better without noise conditioning.
- Shows vs infers: SHOWS (images). Bears on whether t_L must be given or can be inferred (and on S4's null λ result).
- Reused from: E-ROBUST-037 (VERIFIED in round3/ledger/verify_V2.md)
- Supports card(s): C-TRAIN-03
- Confidence: high (quote), medium (transfer)

### B4. Exposure bias inside the torsion trajectory

### E-TRAIN-060
- Claim: Wang et al. (2024) measure exposure bias in score-based conformer models including TD and adapt input
  perturbation (IP) to TD's training, claiming better accuracy and diversity and "on par" with the state of the art on
  GEOM-QM9.
- Source type: paper
- Locator: paper: papers/related/2024_wang_exposure_bias_conformers.pdf (arXiv 2409.14014 v1), PDF p. 1, Abstract
- Exact quote:
  > Especially by using the IP-enhanced Torsional Diffusion model, we achieve new state-of-the-art performance on the GEOM-Drugs dataset and are on par on GEOM-QM9.
- Shows vs infers: SHOWS (claim; preprint, no venue found)
- Supports card(s): C-TRAIN-04
- Confidence: medium

### E-TRAIN-061
- Claim: Their IP perturbs the training input with extra Gaussian noise beyond the target noise level, with the
  perturbation strength set by grid search over the last half of the sampling trajectory.
- Source type: paper
- Locator: paper: papers/related/2024_wang_exposure_bias_conformers.pdf, PDF p. 3 (Section IV.B "Input Perturbation")
  and p. 4 (Algorithm 3)
- Exact quote:
  > to mitigate the inconsistency between training and generation phases, we perturb the input with a Gaussian noise in the training of the score function to enhance its
  (p. 3; the sentence continues on p. 4, where Algorithm 3 "Training of SGMs with Input Perturbation" follows)
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-04
- Confidence: high

### E-TRAIN-062
- Claim: On GEOM-QM9 under the GeoDiff/ConfGF protocol (not TD's split), their retrained TD scores COV 83.07 % /
  MAT 0.2379 Å and TD-IP 84.49 % / 0.1924 Å (mean); the retrained TD baseline is far weaker than TD's own QM9 result.
- Source type: paper
- Locator: paper: papers/related/2024_wang_exposure_bias_conformers.pdf, PDF p. 4, Table II and its footnote
- Exact quote:
  > Tor. Diff.-IP 84.49 100.00 0.1924 0.1530
  > The rest are obtained from our own experiments and Tor. Diff. is retrained.
  (the Tor. Diff. row's values 83.07 / 0.2379 are in column order in the extraction: COV-mean list then MAT-mean list)
- Shows vs infers: SHOWS (different protocol, threshold 0.5 Å, weak retrained baseline)
- Supports card(s): C-TRAIN-04
- Confidence: medium (table extracted column-wise)

### E-TRAIN-063
- Claim: Epsilon Scaling is a training-free remedy for exposure bias that scales down the network output during
  sampling.
- Source type: paper
- Locator: paper: papers/related/2024_ning_exposure_bias_epsilon_scaling.pdf, PDF p. 1, Abstract (ICLR 2024,
  arXiv 2308.15321 v6)
- Exact quote:
  > we propose a simple, yet effective, training-free method called Epsilon Scaling to alleviate the exposure bias.
- Shows vs infers: SHOWS (images)
- Supports card(s): C-TRAIN-04
- Confidence: high

### E-TRAIN-064
- Claim: DiffGap (ACML 2025) adapts a 3D molecule diffusion model to its own inference-time input biases during
  training (exposure bias in structure-based generation).
- Source type: paper
- Locator: paper: papers/related/2024_bridging_gap_molecule_diffusion.pdf (arXiv 2411.05472 v2), PDF p. 1, Abstract
- Exact quote:
  > By dynamically aligning intermediate denoising steps with realistic generation trajectories, DiffGap enables the diffusion model to adapt to input biases in advance during the training phase.
- Shows vs infers: SHOWS (de novo SBDD, CrossDocked2020; not conformer generation)
- Supports card(s): C-TRAIN-04
- Confidence: high (quote), low (transfer)

### E-TRAIN-065
- Claim: DDPM-IP perturbs ground-truth inputs during training to simulate inference-time prediction errors.
- Source type: paper
- Locator: paper: papers/related/2023_ning_input_perturbation.pdf, PDF p. 1, Abstract (reading-order extraction
  papers/related/fulltext/2023_ning_input_perturbation.readingorder.txt)
- Exact quote:
  > consisting in perturbing the ground truth samples to simulate the inference time prediction errors.
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-029 (VERIFIED in round3/ledger/verify_V2.md)
- Supports card(s): C-TRAIN-04
- Confidence: high

### B5. Self-conditioning, alignment / OT coupling

### E-TRAIN-070
- Claim: Self-conditioning feeds the model its own previous clean-sample estimate as an extra input during sampling.
- Source type: paper
- Locator: paper: papers/related/2022_chen_analog_bits_selfcond.pdf (ICLR 2023, arXiv 2208.04202 v2), PDF p. 3,
  Section 3 "Self-Conditioning"
- Exact quote:
  > Here we propose a technique for the model to directly condition on previously generated samples of its own during the iterative sampling process, which can significantly improve the sample quality of diffusion models.
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-04
- Confidence: high

### E-TRAIN-071
- Claim: Self-conditioning is trained by zeroing the self-condition input half the time and otherwise using a detached
  first-pass estimate, at small extra training cost.
- Source type: paper
- Locator: paper: papers/related/2022_chen_analog_bits_selfcond.pdf, PDF p. 3, Section 3 "Self-Conditioning"
- Exact quote:
  > With some probability (e.g., 50%), we set x~0 = 0 which falls back to modeling without Self-Conditioning.
  (tilde flattened)
- Shows vs infers: SHOWS
- Supports card(s): C-TRAIN-04
- Confidence: high

### E-TRAIN-072
- Claim: In HarmonicFlow's ablation (docking, PDBBind), removing self-conditioning lowers the fraction of poses
  < 2 Å from 20.5 % to 14.3 %.
- Source type: paper
- Locator: paper: papers/related/2023_stark_harmonicflow.pdf, PDF p. 8, Table 4 "Flow matching investigation"
- Exact quote (table rows):
  > NO SELF-CONDITIONING 14.3 29.8 3.7
  > HARMONICFLOW = 0.5 20.5 34.5 3.4
  (σ lost in extraction: the second row is HARMONICFLOW σ = 0.5)
- Shows vs infers: SHOWS (Cartesian docking flow; not a torsion model)
- Supports card(s): C-TRAIN-04
- Confidence: high

### E-TRAIN-073
- Claim: ET-Flow Kabsch-aligns prior samples to data samples to shorten flow-matching paths.
- Source type: paper
- Locator: paper: papers/related/2024_hassan_etflow.pdf, PDF p. 4, Section 3.2 "Alignment"
- Exact quote:
  > we reduce the transport costs between samples from the harmonic prior 0 and samples from the data distribution 1 by rotationally aligning them using the Kabsch algorithm
  (ρ0, ρ1 extracted as "0", "1")
- Shows vs infers: SHOWS (Cartesian prior; TD's torsion prior at σ_max is near-uniform on the torus, so there is no
  prior–data coupling to align: INFERENCE)
- Supports card(s): C-TRAIN-06
- Confidence: high

### E-TRAIN-074
- Claim: Equivariant (OT) flow matching yields shorter integration paths and better sampling efficiency for
  Boltzmann-generator flows.
- Source type: paper
- Locator: paper: papers/related/2023_klein_equivariant_flow_matching.pdf, PDF p. 1, Abstract (NeurIPS 2023,
  arXiv 2306.15030 v2)
- Exact quote:
  > Our results show that the equivariant flow matching objective yields flows with shorter integration paths, improved sampling efficiency, and higher scalability compared to existing methods.
- Shows vs infers: SHOWS (efficiency, not accuracy, is the claimed gain)
- Supports card(s): C-TRAIN-06
- Confidence: high

### B6. Ring geometry sources (context for C-TRAIN-05 and C-TRAIN-02)

### E-TRAIN-080
- Claim: Ring conformations in ~140k CREST/GFN2 molecules fall into relatively few canonical clusters.
- Source type: paper
- Locator: paper: papers/related/2021_chan_ring_puckering.pdf, PDF p. 1 (printed 743), Abstract
- Exact quote:
  > we show that the ring conformations can be classified into relatively few conformational clusters, based on their canonical forms. The number of such canonical clusters increases slowly with ring size.
- Shows vs infers: SHOWS
- Reused from: E-GEOM-042 (VERIFIED in round3/ledger/verify_V3.md)
- Supports card(s): C-TRAIN-05
- Confidence: high

### E-TRAIN-081
- Claim: GEOM conformers (QM9 included) are CREST geometries optimised with GFN2-xTB, so xTB-relaxed L targets the
  reference level of our GT.
- Source type: paper
- Locator: paper: papers/related/2022_axelrod_geom.pdf, PDF p. 3, "Methods", subsection "CREST"
- Exact quote:
  > Geometries from the MTD runs are then optimized with GFN2-xTB.
- Shows vs infers: SHOWS
- Reused from: E-GEOM-015 (VERIFIED in round3/ledger/verify_V3.md)
- Supports card(s): C-TRAIN-02
- Confidence: high

### E-TRAIN-082
- Claim: AvgFlow's authors note that RDKit's ring-atom geometry can limit torsional diffusion's sample quality.
- Source type: paper
- Locator: paper: papers/related/2025_cao_avgflow.pdf (ICML 2025, arXiv 2507.09785), PDF p. 2, Section 1
- Exact quote:
  > The generation quality of RDKit, especially for atom coordinates in rings, can also impact the sample
  (the sentence ends "quality of torsional diffusion" on the next column/page in the extraction)
- Shows vs infers: SHOWS (an assertion, no measurement)
- Supports card(s): C-TRAIN-02, C-TRAIN-05
- Confidence: medium

### E-TRAIN-083
- Claim: PuckerFlow's authors composed sampled rings with a pretrained TD as fixed local substructures (two-stage, no
  TD retraining).
- Source type: paper
- Locator: paper: papers/related/2026_schaufelberger_puckerflow.pdf, PDF p. 21, SI A.5.4 "Generating Exocyclic
  Substituents"
- Exact quote:
  > we sample the ring structures using PuckerFlow, and use these as local substructures for a pretrained model of torsional diffusion using publicly available model weights
- Shows vs infers: SHOWS (qualitative demo, no AMR numbers)
- Reused from: E-GEOM-021 (VERIFIED in round3/ledger/verify_V3.md)
- Supports card(s): C-TRAIN-02, C-TRAIN-03
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## C. Blogs (intuition and practice, not proof)

### E-TRAIN-090
- Claim: Practitioner summary of cascaded diffusion: truncated conditioning augmentation stops the upstream diffusion
  early (t > 0) and hands the partially denoised input to the next stage.
- Source type: blog
- Locator: blog: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/, Lilian Weng, 2021-07-11 (updated
  later), section "Scale up Generation Resolution and Quality", accessed 2026-10-08 (snapshot by scout ROBUST),
  re-read 2026-10-12, snapshot papers/blogs/weng_2021_diffusion_models.txt line 729
- Exact quote:
  > Truncated conditioning augmentation stops the diffusion process early at step $t > 0$ for low resolution.
- Shows vs infers: SHOWS (secondary source for E-TRAIN-050)
- Supports card(s): C-TRAIN-03
- Confidence: medium

### E-TRAIN-091
- Claim: A practitioner (Rowan) reports, as an anecdote, that ETKDG has "a curious preference for twist boats over
  chairs", i.e. a structured ring-pucker error rather than random noise.
- Source type: blog
- Locator: blog: https://rowansci.substack.com/p/openconf-and-other-open-source-projects, Rowan (byline Corin Wagen,
  Nick Casetti, Jonathon Vandezande, Eli Mann), 2026-06-17, section "openconf: Rapid Monte Carlo–Based Conformer
  Generation", accessed 2026-10-08, snapshot papers/blogs/rowan_openconf_2026.txt line 16
- Exact quote:
  > ETKDG often misses conformers or generates high-energy conformers for large systems (e.g. PROTACs) and has additional random pathologies (e.g. a curious preference for twist boats over chairs).
- Shows vs infers: SHOWS (anecdote; only the twist-boat remark is relevant to QM9 size)
- Reused from: E-EVAL-029 (VERIFIED after D-002 in round3/ledger/verify_V1.md)
- Supports card(s): C-TRAIN-05
- Confidence: medium

### E-TRAIN-092
- Claim: Rowan's openconf perturbs ETKDG seeds with ring-specific Monte Carlo moves (ring flip, crankshaft) and
  re-optimises with MMFF94, i.e. structured pucker-space moves are standard practice for exploring ring conformations.
- Source type: blog
- Locator: blog: https://rowansci.substack.com/p/openconf-and-other-open-source-projects, Rowan, 2026-06-17, section
  "openconf: Rapid Monte Carlo–Based Conformer Generation", accessed 2026-10-08, re-read 2026-10-12, snapshot
  papers/blogs/rowan_openconf_2026.txt lines 28-30 and 36
- Exact quote:
  > Ring flip: reflect e.g. a cyclohexane ring from one chair conformation to its mirror image
  > Crankshaft: rotate an arc of ring atoms around the axis connecting two anchor atoms
- Shows vs infers: SHOWS (practice; no numbers for these moves)
- Supports card(s): C-TRAIN-05
- Confidence: medium

### E-TRAIN-093
- Claim: The Diffusion Forcing authors frame per-component noise levels as fractional masking, unifying "denoise
  everything together" and "condition on clean context".
- Source type: blog
- Locator: blog: https://boyuan.space/diffusion-forcing/, Boyuan Chen et al. (MIT), project page (paper 2024; page
  notes a 2025 update), section "Diffusion Forcing", accessed 2026-10-12, snapshot
  papers/blogs/chen_2024_diffusion_forcing_page.txt lines 74-79
- Exact quote:
  > One can view noises in diffusion as varying levels of masking and establish a unified view: full-sequence diffusion denoise all frames at once with the same noise level
- Shows vs infers: SHOWS (author's page; intuition). INFERENCE: for L and τ, "L at zero noise" is TD (L given) and
  "same noise level" is a joint model; independent levels interpolate between them.
- Supports card(s): C-TRAIN-03
- Confidence: medium
