# Evidence ledger: SCOUT ARCH (round 4, P1)

Lens: novel architectures / methods that learn local structure L (bond lengths, angles, ring puckers) jointly with or
before torsions (brief Q2, feeding Q1). Agent tag ARCH. Template: `orchestration/templates/evidence_entry.md`.
Entries are appended incrementally. Page numbers are PDF pages counted by form feeds in the `pdftotext` fulltext
(`papers/related/fulltext/<name>.txt`), which equal the PDF page index; printed page given where it differs.
Entries reused from round 3 carry "Reused from" with the round-3 ID and its verdict.

## Our data (reused, VERIFIED in round 3, plus round-2 FINAL paired contrasts)

### E-ARCH-001
- Claim: With the model in the loop, ring geometry is the lever: on the like-for-like matched-RDKit base (λ = 0, 955
  molecules), true ring geometry alone (A5ring, RDKit acyclic L) takes CTRL_rematch from 0.1966 to 0.1186 Å AMR-R
  and B1 from 0.2614 to 0.1877; true acyclic geometry alone (A5acyc, RDKit rings) gives CTRL 0.1824 and B1 0.1583.
  All are ORACLE (own-conformer, cycled L). Reused from E-GEOM-013 (revised after D-201; VERIFIED, V3) and the
  E-ROBUST addendum after D-110; values re-read from the round-2 final contrasts file.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows `S1.B1_vs_CR.S1_lam0.00_cyc_ORACLE`,
  `S1.B1_vs_CR.S1_A5ring_cyc_ORACLE`, `S1.B1_vs_CR.S1_A5acyc_cyc_ORACLE` (universe union, MAT-R; arm_value = B1,
  ref_value = CTRL_rematch).
- Exact quote (contrasts.csv, A5ring row, truncated):
  > S1.B1_vs_CR.S1_A5ring_cyc_ORACLE,...,MAT-R,primary,955,B1,S1_A5ring_cyc_ORACLE,CR,S1_A5ring_cyc_ORACLE,,0.1877...,0.1186...
- Shows vs infers: SHOWS (3-seed means, 955 molecules). ORACLE with own-conformer L (leak caveat E-ARCH-009).
- Supports card(s): C-ARCH-01, C-ARCH-02, C-ARCH-04
- Confidence: high

### E-ARCH-002
- Claim: Ring-pucker error is a tail: among seeds of molecules with a ring of ≥ 4 atoms, the share with endocyclic
  ring-dihedral RMSD > 10° is ETKDG 49.6 %, MMFF 35.1 %, λ = 0.50 19.4 %, λ = 0.75 0.02 %, true L + 0.02 Å noise 0 %;
  mean per-seed ring errors are ETKDG 0.0318 Å (bonds) / 2.38° (angles) / 10.65° (ring dihedral, median 0.44°) and
  MMFF 0.0199 Å / 1.42° / 8.07°. Reused from E-GEOM-001 and E-GEOM-002 (revision; both VERIFIED, V3).
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, columns `bond_rmsd_heavy_ring`,
  `angle_rmsd_heavy_ring`, `ring_dihedral_rmsd`, grouped by `cond` (all-3-ring molecules excluded for the shares).
- Exact quote (header line of the file):
  > cond,smiles,k,src_gt_idx,bond_rmsd_all_ring,angle_rmsd_all_ring,...,angle_rmsd_heavy_any,ring_dihedral_rmsd
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

### E-ARCH-003
- Claim: QM9's bad ring seeds are split across ring classes that differ in how a Cremer–Pople model can represent them:
  of 885 test ring molecules, 420 have only isolated rings and 465 contain a fused, bridged or spiro system; of the
  2074 ETKDG seeds with ring-dihedral RMSD > 10°, isolated 4-rings hold 519, isolated 5-, 6-, 7-rings 635, 222, 140,
  and fused/bridged/spiro systems 449 (22 %); molecules whose rings are all isolated 5–8-membered hold 48.1 %; 832 of
  885 ring molecules carry an exocyclic heavy atom. By smallest ring, the ETKDG > 10° share is 4-ring 60 %, 5-ring
  48 %, 6-ring 66 %, 7-ring 92 %. Reused from E-GEOM-011 (with D-209 addendum) and E-GEOM-003 (VERIFIED, V3).
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, `cond == L_etkdg2L`, joined to an RDKit
  ring-system analysis of the SMILES (round-3 scripts `rings.py`, `p2.py`, cited in E-GEOM-011).
- Exact quote (same header as E-ARCH-002):
  > ring_dihedral_rmsd
- Shows vs infers: SHOWS (counts).
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-004
- Claim: Rigid molecules (≈ 30 % of test molecules) are a model-free read-out of L quality: their AMR-R is set by the
  seed L, not the torsion model (CTRL vs B1 within ≈ 0.006 Å); CTRL rigid-bin AMR-R is ETKDG 0.152, MMFF 0.128, λ = 0
  0.184 and true ring geometry only (A5ring, ORACLE) 0.043. Rigid molecules carry 26.0 % of the macro AMR-R sum and
  54.9 % of the total ring-oracle gain. Reused from E-GEOM-004 (revised after D-205), E-GEOM-006 addendum and
  E-EVAL-057 (all VERIFIED).
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s{0,1,2}/{S1_etkdg2L,S1_A5ring_cyc_ORACLE,S1_lam0.00_cyc_ORACLE}/breakdown.log`,
  table "by n_rot_heavy", bin 0, `MAT_R`.
- Exact quote (CTRL s0, S1_A5ring_cyc_ORACLE/breakdown.log):
  > | 0     | 289 | 100     |   0.043 |  99.913 |   0.053 |
- Shows vs infers: SHOWS. INFERENCE (used in the cards): any learned L generator can be scored on this stratum, with
  no torsion model, before any torsion training is spent.
- Supports card(s): C-ARCH-01, C-ARCH-02, C-ARCH-03
- Confidence: high

### E-ARCH-005
- Claim: Round-2 FINAL paired analysis: S3 (50/50 RDKit/true-L training) is significantly worse than CTRL_rematch on
  RDKit L (+0.0032 Å [+0.0013, +0.0052], Holm p = 0.0096) and much worse on MMFF-relaxed L (+0.0394 [+0.0338,
  +0.0454]); B1 is worse on MMFF L by +0.0798 [+0.0722, +0.0877]; S3 at blend λ = 0.5 is +0.0037 worse. Models trained
  on true or mixed L lose on improved-but-imperfect L.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows `S3.rdkit_vs_CR` (primary),
  `S1.S3_vs_CR.S1_etkdg2L_mmff`, `S1.B1_vs_CR.S1_etkdg2L_mmff`, `S1.S3_vs_CR.S1_lam0.50_cyc_ORACLE` (union, MAT-R);
  summarised in `round4/BRIEF.md` §"Round-2 FINAL paired analysis".
- Exact quote (round4/BRIEF.md):
  > S3 (50/50 mix) on RDKit L vs CTRL_rematch: +0.0032 [+0.0013, +0.0052], Holm p = 0.0096 — significantly WORSE
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01, C-ARCH-02, C-ARCH-05
- Confidence: high

### E-ARCH-006
- Claim: The only learned recipe with a real-use gain is training on the deployed L source: S5 (TD trained on
  MMFF-relaxed matched seeds, tested on MMFF seeds) beats CTRL_rematch tested on MMFF seeds by −0.0041 Å
  [−0.0064, −0.0019] (0.1500 vs 0.1540) and CTRL on RDKit L by −0.0287; the same S5 model without MMFF at test is
  +0.0031 worse than CTRL.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows `S5.premmff_vs_CRpremmff`,
  `S5.premmff_vs_CR`, `S5.nommff_vs_CR` (union, MAT-R, n = 934).
- Exact quote (round4/BRIEF.md):
  > S5 (MMFF-matched training, tested with MMFF): −0.0041 [−0.0064, −0.0019] vs CTRL_rematch with MMFF at test
- Shows vs infers: SHOWS. INFERENCE: the torsion model must be trained on the L distribution it will see (a DAgger /
  sampler-matched rule), which every two-stage card below adopts.
- Supports card(s): C-ARCH-01, C-ARCH-02, C-ARCH-05
- Confidence: high

### E-ARCH-007
- Claim: A torsion model that is told the quality level of its L uses improved-but-imperfect L better than CTRL when
  the level is right: S4 (trained on aligned RDKit→true blends with λ as input) evaluated on λ = 0.50 blends with
  `--l_level 0.5` beats CTRL_rematch on the same blends by −0.0090 Å [−0.0134, −0.0047] (0.1274 vs 0.1364, 906
  molecules, ORACLE blends); at λ = 0 it is +0.0025 [−0.0021, +0.0073] (n.s.).
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows `S4.lam0.50_vs_CR@S4subset`,
  `S4.lam0.00_vs_CR@S4subset` (union, MAT-R).
- Exact quote (contrasts.csv, truncated):
  > S4.lam0.50_vs_CR@S4subset,...,MAT-R,...,906,S4,S4_lam0.50_l0.50_cyc_ORACLE,CR,S1_lam0.50_cyc_ORACLE,,0.1274...,0.1364...
- Shows vs infers: SHOWS. ORACLE blends (built from the true L; ≈ 10 % of λ slots reuse another conformer's blend,
  `round3/impact.md` §1 data defect). INFERENCE: a joint model whose L follows a known path (the bridge clock of
  C-ARCH-01) puts the torsion head in exactly this "level known" regime.
- Supports card(s): C-ARCH-01
- Confidence: medium (oracle blends; one comparison)

### E-ARCH-008
- Claim: The λ dose curve (aligned RDKit→true blends, own-conformer, ORACLE): CTRL_rematch AMR-R at λ 0 / 0.25 /
  0.50 / 0.75 = 0.1966 / 0.1596 / 0.1364 / 0.1156 and B1 0.2614 / 0.2219 / 0.1812 / 0.1147; B1 − CTRL at 0.75 is
  −0.0010 [−0.0070, +0.0046] (tie); true L cycled gives CTRL 0.0822, B1 0.0212.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/analysis_r2/contrasts.csv`, rows
  `S1.B1_vs_CR.S1_lam{0.00,0.25,0.50,0.75}_cyc_ORACLE` (union, MAT-R; n 955/906/906/906); true-L values from rows
  `S3.gtLc_vs_CR_ORACLE` (ref 0.0822) and `B1cap.gtLc_vs_B1_ORACLE` (ref 0.0212).
- Exact quote (contrasts.csv, λ 0.75 row, truncated):
  > S1.B1_vs_CR.S1_lam0.75_cyc_ORACLE,...,906,B1,S1_lam0.75_cyc_ORACLE,CR,S1_lam0.75_cyc_ORACLE,,0.1147...,0.1156...
- Shows vs infers: SHOWS. Caveat (`round3/impact.md` §1): 1268 of 12,707 λ ∈ (0,1) slots reuse another GT conformer's
  blend, so λ is not a clean dose axis.
- Supports card(s): C-ARCH-01
- Confidence: high (values); medium (as a dose axis)

### E-ARCH-009
- Claim: The true-L oracles leak conformer identity: inverting the round-1 random-GT-conformer runs per n_true bin, B1
  given another conformer's reference L would score ≈ 0.16–0.35 Å (vs 0.016–0.046 with its own L) and CTRL ≈
  0.10–0.17; leak-free floors at zero pucker tail are CTRL 0.111–0.145 and B1 0.19–0.27 (vs 0.086 / 0.038 leaky).
- Source type: our-data (round-3 impact analysis, cross-read with the code grounder; not checked by a P2 verifier)
- Locator: our-data: `round3/impact.md` §3g table and "Leak-corrected floors"; `round3/impact_data/ntrue_leak.py`,
  `ntrue_leak.out`.
- Exact quote (round3/impact.md §3g):
  > **Reading.** B1's true-L advantage is almost entirely conformer identity.
- Shows vs infers: INFERENCE (a model-based inversion; the source states noise is amplified 8–14× per bin).
- Supports card(s): C-ARCH-01, C-ARCH-02, C-ARCH-05
- Confidence: medium

### E-ARCH-010
- Claim: Without a model, L from another GT conformer of the same molecule is almost as good as own L: the round-1
  best-symmetry RDKit-L floor for ring molecules is 0.1171 Å per conformer, 0.0989 with GT acyclic angles and bonds,
  and 0.0323 with another GT conformer's full L ("perfect independent L sampler"); for acyclic molecules 0.0510 →
  0.0081 (GT acyclic angles + bonds) and 0.0137 (other GT conformer).
- Source type: our-data
- Locator: our-data: `cluster_sync/results/analysis/local_structure_test.csv` (7554 GT conformers); strata table in
  `round2/research_A.md:24-34`, recomputed and VERIFIED in `round2/verify_A.md:79-82`.
- Exact quote (round2/research_A.md:28):
  > | any ring (5444) | 0.1171 | 0.1072 | 0.0989 | 0.0323 | 3.76 |
- Shows vs infers: SHOWS (floors from torsion optimisation, no model). INFERENCE: an independent reference-level L
  sampler leaves a small floor; the large leak gap in E-ARCH-009 is a property of the trained torsion models, not of
  the geometry, so it should be fixable by training (C-ARCH-05).
- Supports card(s): C-ARCH-02, C-ARCH-04, C-ARCH-05
- Confidence: high

### E-ARCH-011
- Claim: The round-3 ring-tail mixture model predicts CTRL AMR-R on real samplers as ≈ 0.086 + 0.20·X (X = share of
  ≥ 4-ring seeds with ring-dihedral RMSD > 10°); it fits ETKDG (pred. 0.178, obs. 0.178) but over-predicts the MMFF
  gain (pred. 0.1446, obs. 0.1525; real samplers realise 57–76 % of the predicted gain).
- Source type: our-data (round-3 impact analysis)
- Locator: our-data: `round3/impact.md` §3c and §3g ("CTRL model check by n_true bins"); `round3/impact_data/mixture_model.out`.
- Exact quote (round3/impact.md §3c):
  > CTRL's AMR on real samplers is therefore almost entirely a function of the ring tail:
- Shows vs infers: INFERENCE (two-anchor interpolation; its anchor G is leaky, E-ARCH-009).
- Supports card(s): C-ARCH-02, C-ARCH-03
- Confidence: medium

### E-ARCH-012
- Claim: At the 0.01–0.03 Å / 1–1.5° level, pucker correctness matters more than bond/angle accuracy for CTRL: true L
  + 0.02 Å/axis noise (ring bond 0.027 Å, angle 1.38°, ring dihedral 0.96°) gives 0.098, better than the λ = 0.75 blend
  (0.010 Å, 0.67°, but ring dihedral 1.56°, biased toward RDKit) at 0.116. Reused from E-GEOM-005 (VERIFIED, V3).
- Source type: our-data
- Locator: our-data: `round3/BRIEF.md` table rows "True L + noise 0.02" and "λ = 0.75"; errors from E-ARCH-002.
- Exact quote (round3/BRIEF.md):
  > | True L + noise 0.02 Å/axis (ORACLE) | 0.098 | 0.117 (seeds vary 0.092–0.134) | 996 |
- Shows vs infers: SHOWS numbers; INFERENCE as stated (unpaired, different populations).
- Supports card(s): C-ARCH-02, C-ARCH-04
- Confidence: medium

### E-ARCH-013
- Claim: Post-hoc physical relaxation after diffusion trades recall for precision: MMFF after TD (A3) gives AMR-R
  0.1863 vs 0.1752 for TD and AMR-P 0.1641 vs 0.2191. Reused from E-EVAL-041 (VERIFIED, V1).
- Source type: our-data
- Locator: our-data: round-1 A3 run as cited in E-EVAL-041 (`round3/ledger/evidence_EVAL.md`).
- Exact quote (E-EVAL-041 claim):
  > MMFF relaxation after diffusion (A3) worsens recall (AMR-R 0.1863 Å) but improves precision (AMR-P 0.1641 Å vs 0.2191 for R0).
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-03
- Confidence: high

### E-ARCH-014
- Claim: GEOM-QM9 reference conformers are GFN2-xTB-level structures (QM9 seeds re-optimised with GFN2-xTB before
  CREST, whose metadynamics geometries are GFN2-xTB-optimised), and re-optimising GEOM references with GFN2-xTB barely
  moves them. Reused from E-GEOM-015, E-GEOM-016 and E-GEOM-018 (VERIFIED, V3).
- Source type: paper
- Locator: paper: `papers/related/2022_axelrod_geom.pdf`, PDF p. 3 and p. 5; `papers/related/2025_nikitin_geom_drugs_revisited.pdf`, PDF p. 10.
- Exact quote (Axelrod p. 5):
  > to seed CREST with a structure optimized at the GFN2-xTB level of theory, we re-optimized each QM9 geometry with xTB before using it in CREST.
- Shows vs infers: SHOWS (the "barely moves" part is shown for GEOM-Drugs; QM9 transfer is INFERENCE, as in E-GEOM-018).
- Supports card(s): C-ARCH-05
- Confidence: high

### E-ARCH-015
- Claim: The round-3 analyst predicted, for CTRL on GFN2-xTB-relaxed ETKDG seeds, AMR-R ≈ 0.146 (range 0.13–0.155,
  worst ≈ MMFF 0.153), and for models trained on true/mixed L a range 0.10–0.22 that depends on the share s of target
  conformers that get a seed in their own basin.
- Source type: our-data (round-3 impact analysis; prediction, not a measurement)
- Locator: our-data: `round3/impact.md` §0 (C-GEOM-01 row) and §3g.
- Exact quote (round3/impact.md §0, C-GEOM-01 row):
  > CTRL 0.146 (0.13–0.155; worst ≈ MMFF 0.153).
- Shows vs infers: INFERENCE (model prediction). Since then the round-2 final analysis showed S3 and B1 lose on MMFF L
  (E-ARCH-005), which moves the true/mixed-L arms toward the bad end of that range.
- Supports card(s): C-ARCH-05
- Confidence: medium

### E-ARCH-016
- Claim: The round-2 design estimate for adding even an acyclic joint bond-angle factor to TD was 300+ changed lines
  (parity-invariant angle head, Jacobian update, joint training). Reused from E-GEOM-049 (VERIFIED as an estimate, V3).
- Source type: our-data (round-2 planning estimate)
- Locator: our-data: `round2/research_A.md:257-261` (R2A-6 "Why deferred").
- Exact quote:
  > It needs a parity-invariant angle head, a Jacobian update for angles, and joint training: an estimated 300+ lines
- Shows vs infers: SHOWS the estimate; its accuracy is unverified.
- Supports card(s): C-ARCH-04
- Confidence: medium

## Our code (TD touchpoints)

### E-ARCH-020
- Claim: TD's score network ends in a bond-centred tensor-product layer whose output irreps are pseudoscalars (`0o`)
  when parity is on, reduced by an MLP to one real number per rotatable bond; the only generated quantity is the
  torsion score `edge_pred`.
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/score_model.py:119-132`, `:145-158`
- Exact quote:
  > out_irreps=f'{ns}x0o' if parity else f'{ns}x0e',  # [ablation-hooks] 0e = parity-invariant ablation
  > out = self.bond_conv(node_attr, edge_index, edge_attr, edge_sh, out_nodes=data.edge_mask.sum(), reduce='mean')
- Shows vs infers: SHOWS. INFERENCE: a ring-pucker head (Cremer–Pople coordinates are pseudoscalars, E-ARCH-040) can
  reuse the same `0o` construction centred on ring centroids instead of bonds; an L-velocity head needs `1o` node
  outputs, which the atom layers already carry (E-ARCH-021).
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

### E-ARCH-021
- Claim: With `use_second_order_repr` the atom convolution layers carry vector (`1o`) and higher features, so per-atom
  equivariant vector outputs can be read from the last atom layer without changing the trunk.
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/score_model.py:83-89`, `:141-143`
- Exact quote:
  > f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01
- Confidence: high

### E-ARCH-022
- Claim: TD's training transform perturbs torsions only: a single σ per sample, wrapped-normal torsion updates applied
  by `modify_conformer`, and the update vector is the regression target.
- Source type: code
- Locator: code: `torsional-diffusion/utils/dataset.py:71-76`; loss `torsional-diffusion/utils/training.py:25-31`
- Exact quote:
  > torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
  > loss = ((score - pred) ** 2 / score_norm).mean()
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

### E-ARCH-023
- Claim: The round-2 code already provides (i) a paired cache holding, per training conformer k, the matched RDKit L
  `data.pos[k]` and its own Kabsch-aligned, terminal-relabelled GT L `data.gt_pos[k]`, (ii) an aligned blend
  `interp_x_torch(x, y, λ)` that keeps terminal bond lengths, and (iii) a λ embedding concatenated to the σ embedding.
- Source type: code
- Locator: code: `torsional-diffusion/utils/dataset.py:18-34`, `:79-99`; `torsional-diffusion/diffusion/score_model.py:65-79`, `:194-199`
- Exact quote:
  > x, y = data.pos[k], data.gt_pos[k]
  > data.pos = interp_x_torch(x, y, lam, data.edge_index)
  > node_lam_emb = get_timestep_embedding(data.node_lambda * 10000, self.lambda_embed_dim)
- Shows vs infers: SHOWS. INFERENCE: these are exactly the pieces a learned L bridge (C-ARCH-01) needs: paired
  endpoints, a physically repaired interpolant, and a clock input.
- Supports card(s): C-ARCH-01
- Confidence: high

### E-ARCH-024
- Claim: The sampler integrates only torsions: each step evaluates the model, forms the reverse-SDE torsion increment
  and applies it by rotating bond sides; L never changes after seeding.
- Source type: code
- Locator: code: `torsional-diffusion/diffusion/sampling.py:184-204`, `:234`, `:290-298`
- Exact quote:
  > perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z
  > conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01, C-ARCH-02, C-ARCH-03
- Confidence: high

### E-ARCH-025
- Claim: A bond is rotatable in TD only if removing it disconnects the graph with ≥ 2 atoms on the smaller side, so
  every ring bond is excluded from the torsion space and ring geometry is fixed by the seed.
- Source type: code
- Locator: code: `torsional-diffusion/utils/torsion.py:16-31`
- Exact quote:
  > G2.remove_edge(*edges[i])
  > if not nx.is_connected(G2):
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-026
- Claim: Any external L source plugs into TD without model changes through `--seed_confs` (a pickle of RDKit mols
  keyed by SMILES); each sample takes one seed conformer's coordinates as its L.
- Source type: code
- Locator: code: `torsional-diffusion/generate_confs.py:22`, `:76-79`; `torsional-diffusion/diffusion/sampling.py:84-91`
- Exact quote:
  > seed_mol = seed_confs[smi][i % len(seed_confs[smi])] if seed_confs_cycle else random.choice(seed_confs[smi])
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-02, C-ARCH-03, C-ARCH-05
- Confidence: high

## Papers: internal-coordinate / ring generative models

### E-ARCH-030
- Claim: TD treats ring torsions as part of the local structure L and takes approximate samples of p(L) from RDKit,
  learning only p(τ | L).
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 4, §3.
- Exact quote:
  > Thus, torsion angles in cycles (or rings), which cannot be rotated independently, are considered part of the local structure L.
  > Thus, we use RDKit to provide approximate samples from pG(L), and develop a diffusion-based generative model to learn distributions pG( | L) over torsion angles
- Shows vs infers: SHOWS (τ glyph lost in extraction).
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

### E-ARCH-031
- Claim: TD's forward kernel on T^m is the wrapped normal, sampled as an unwrapped isotropic normal taken mod 2π, with
  precomputed scores and a denoising-score-matching loss weighted by the inverse expected score norm (Eq. 3–4); the
  score head outputs one SE(3)-invariant, parity-equivariant pseudoscalar per rotatable bond.
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 4 (Eq. 3–4) and p. 6 (§3.3–3.4).
- Exact quote:
  > We thus sample from the perturbation kernel by sampling from the corresponding unwrapped isotropic normal and taking elementwise mod 2.
  > We extract the pseudoscalar outputs of this filter to produce a single real-valued pseudoscalar prediction i for each rotatable bond.
- Shows vs infers: SHOWS (π and δτ glyphs lost).
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

### E-ARCH-032
- Claim: DiffDock's product-space recipe: on a product manifold the forward diffusion runs independently in each
  factor and the tangent space is a direct sum, so one samples each factor's kernel and regresses each factor's score
  independently.
- Source type: paper
- Locator: paper: `papers/related/2022_corso_diffdock.pdf`, PDF p. 6, §4.3 "Diffusion on the product space".
- Exact quote:
  > since P is a product manifold, the forward diffusion proceeds independently in each manifold
  > Thus, it suffices to sample from the diffusion kernel and regress against its score in each group independently.
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

### E-ARCH-033
- Claim: Corso's thesis names ring flexibility as the largest source of flexibility TD does not model, says TD's error
  is lower-bounded by the cheminformatics local structure, proposes ring-puckering coordinates on hyperspheres as an
  extra IDM factor, and proposes using the extrinsic manifold as a soft constraint on a full-dimensional diffusion.
  Partly reused from E-GEOM-032 (p. 63, VERIFIED).
- Source type: paper
- Locator: paper: `papers/related/2023_corso_intrinsic_diffusion.pdf`, PDF p. 92 (App. C.1.2), p. 63 and p. 64 (§ future work).
- Exact quote:
  > The largest source of flexibility in molecular conformations that is not directly accounted for by torsional diffusion is the variability in ring conformations.
  > using such manifold as a soft constraint or inductive bias to make the full dimensional diffusion more efficient
- Shows vs infers: SHOWS (proposals; no experiment).
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

### E-ARCH-034
- Claim: Corso's thesis notes that importance-sampling the full Boltzmann distribution would need a model of the local
  structure with exact likelihoods, which RDKit does not provide.
- Source type: paper
- Locator: paper: `papers/related/2023_corso_intrinsic_diffusion.pdf`, PDF p. 93, App. C.1.2 "Boltzmann generation".
- Exact quote:
  > one would need a model () over local structures that also provides exact likelihoods. This is not the case with RDKit
- Shows vs infers: SHOWS (p(L) glyphs lost). INFERENCE: a learned L factor with a likelihood (C-ARCH-02's CP flow)
  adds this capability; a Cartesian bridge (C-ARCH-01) does not by default.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-035
- Claim: GO-Flow decomposes a conformer into centre of mass, a unit-quaternion orientation and internal coordinates z
  (bond lengths, angles, periodic torsions), learns one flow per manifold (linear OT, SLERP geodesics, entropic-OT
  Sinkhorn couplings for z) and maps internal velocities to Cartesian velocities with explicit Jacobians (Eq. 9):
  v = v_trans + ω × (x − c) + J(z) v_conf; training is a three-stage curriculum (per-manifold, coupled with a
  Cartesian consistency loss, optional likelihood fine-tuning).
- Source type: paper
- Locator: paper: `papers/related/2026_liu_goflow.pdf`, PDF p. 3–4 (§3.1–3.4, Eq. 1–9, Alg. 1), p. 11–12 (App. A.3–A.4, B.5).
- Exact quote:
  > We require a transport plan that respects the distribution's geometry, necessitating entropic optimal transport rather than simple linear interpolation.
  > We therefore employ a mathematically rigorous Jacobian-based projection to map internal velocities back to Cartesian updates.
- Shows vs infers: SHOWS (symbols lost in extraction; equation content from Alg. 1 and App. A.4, Eq. 35–37 and 45).
- Supports card(s): C-ARCH-04, C-ARCH-06
- Confidence: high

### E-ARCH-036
- Claim: GO-Flow is evaluated only on the ConfGF/GeoDiff protocol (40,000 training molecules × 5 conformers, 200 test
  molecules, δ 0.5 Å QM9 / 1.25 Å DRUGS), trained for 700k iterations on eight A6000 GPUs; on QM9 its MAT-R is 0.2086
  vs 0.2130 for its ET-Flow re-run in the same table, and the variant without the internal-coordinate manifold ("w/o
  C") has MAT-R 0.2149 (COV-P 53.86 vs 69.42).
- Source type: paper
- Locator: paper: `papers/related/2026_liu_goflow.pdf`, PDF p. 14 (App. D.1, D.4, Table 5), p. 7 (Table 2, Table 3).
  Table rows read from `pdftotext -raw` (the layout fulltext scrambles the columns).
- Exact quote:
  > Each dataset comprises 40,000 molecules for the training set and 5,000 molecules for the validation set, with each molecule containing 5 conformers
  > w/o C 90.26 0.8512 62.47 1.1531 88.06 0.2149 53.86 0.4367
- Shows vs infers: SHOWS. Not comparable with TD-protocol numbers (different split, 200 molecules). INFERENCE: on QM9
  the whole internal-coordinate manifold is worth ≈ 0.006 Å MAT-R in GO-Flow's own ablation.
- Supports card(s): C-ARCH-04, C-ARCH-06
- Confidence: high

### E-ARCH-037
- Claim: GO-Flow does not address ring closure: its internal coordinates for rings are redundant, yet the paper's only
  ring statements are an atom feature ("in_ring") and a qualitative remark about baselines' distorted rings.
- Source type: paper
- Locator: paper: `papers/related/2026_liu_goflow.pdf`, full-text search for "ring|cycl|closure|NeRF" (PDF p. 14
  Table 4, p. 15 App. E are the only hits).
- Exact quote (p. 15):
  > devoid of the disconnected components or distorted rings often seen in baseline methods
- Shows vs infers: SHOWS by absence (search of the whole text). INFERENCE: GO-Flow's claimed internal-coordinate
  advantage is not demonstrated for ring puckers, the part of L our data say matters (E-ARCH-001).
- Supports card(s): C-ARCH-02, C-ARCH-04
- Confidence: medium (absence claim)

### E-ARCH-038
- Claim: PuckerFlow does flow matching on the Cremer–Pople (CP) manifold: an N-ring has N − 3 CP coordinates
  (amplitude/phase pairs q_m cos φ_m, q_m sin φ_m from a discrete Fourier transform of out-of-plane displacements z_j,
  Eq. 1–2), rings are rebuilt from CP coordinates with precomputed bond lengths and angles (Cremer's segment
  method, SI Eq. 8–10), and closed rings come out by construction.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 3–5 (§2.1–2.2, Eq. 1–2), p. 16 (SI A.1.2, Eq. 8–10).
- Exact quote:
  > the 3N Euclidean degrees of freedom in the Cartesian space are reduced to the most relevant N - 3 degrees of freedom corresponding to the Cremer-Pople coordinates.
  > It uses precomputed bond lengths and angles from the training data, parameterized by bond type, element, and ring size.
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-039
- Claim: PuckerFlow's training: a uniform prior in CP space with amplitudes bounded at 0.8, 0.56, 0.4 for m = 2, 3, 4,
  straight conditional paths x_t = t x1 + (1 − t) x0, an x1-predicting network (loss ‖u_θ(x_t, t) − x1‖²), Euler
  sampling x_t ← x_t + Δt (x̃1 − x_t)/(1 − t), 30 inference steps; the bounded prior plus the convexity of valid CP
  space keeps rings valid during training.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 19 (SI A.4.2), p. 20 (Alg. 1–2), p. 6 (§2.2), p. 8.
- Exact quote:
  > In practice, we sample uniformly in Cremer-Pople space, restricting the maximum amplitudes to 0.8, 0.56, and 0.4 for coordinates of order m = 2, 3, and 4, respectively
  > combined with flow matching's straight conditional training paths and the convexity of valid Cremer-Pople space, ensures valid rings throughout training.
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-040
- Claim: PuckerFlow's output is SE(3)-invariant and parity-odd: each CP coordinate is a pseudoscalar, produced by a
  "cyclic Fourier filter" (equivariant filters placed on the ring mean plane, then a scaled DFT) that adapts to ring
  size; the network is an e3nn tensor-field network like TD's.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 6 (§2.2), p. 20–21 (SI A.5, A.5.3).
- Exact quote:
  > each Cremer-Pople coordinate being a pseudoscalar, i.e., invariant under rotation but changing sign under parity inversion.
- Shows vs infers: SHOWS. INFERENCE: the same symmetry class as TD's torsion scores (E-ARCH-031), so a CP head can sit
  next to TD's pseudotorque head in one network.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-041
- Claim: Reconstructing a ring from its true CP coordinates with PuckerFlow's dictionary bond lengths/angles has a
  median RMSD of 0.046 Å to the original; dictionary values differ from true ones by median 0.85° (mean 1.96°) and
  0.008 Å (mean 0.015 Å).
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 18 (SI A.3), Fig. 9 on p. 19.
- Exact quote:
  > The median RMSD of 0.046 Å confirms that the reconstructed geometries closely reproduce the reference structures
- Shows vs infers: SHOWS (on substituent-free monocycles). INFERENCE: with dictionary bonds/angles a perfect CP
  factor still carries a ≈ 0.05 Å ring residual — about the size of ET-Flow's whole QM9 median (0.047); keeping the
  seed's own bonds/angles or relaxing afterwards is needed.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-042
- Claim: PuckerFlow's authors propose joint flow matching of cyclic and torsional degrees of freedom as future work and
  say the architectures of PuckerFlow and TD are compatible; their released code builds on the TD codebase.
- Source type: paper (+ code repository)
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 10 (§3) and p. 21 (SI A.5.4);
  blog/code: https://github.com/digital-chemistry-laboratory/puckerflow README, L. Schaufelberger, A. Hartgers,
  K. Jorner, accessed 2026-10-12, section "Acknowledgements", snapshot `papers/blogs/puckerflow_repo_readme.txt`.
- Exact quote:
  > making an extension feasible that performs joint flow matching over the different degrees of freedom to simultaneously sample the cyclic and torsional degrees of freedom.
  > The implementation builds on the codebase of [Torsional Diffusion](https://github.com/gcorso/torsional-diffusion)
- Shows vs infers: SHOWS. INFERENCE (Q1): a joint CP × T^m model is explicitly the next step of an active group; a
  FlexiTors-ring paper competes with them on time.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-043
- Claim: PuckerFlow's README pins torch 2.1.0+cu118 and unpinned e3nn / torch_geometric, i.e. a different stack from
  our TD venv (torch 1.13.1+cu117, e3nn 0.5.1, E-GEOM-014), and the code is MIT-licensed.
- Source type: blog (code repository README)
- Locator: blog: https://github.com/digital-chemistry-laboratory/puckerflow, L. Schaufelberger et al., README section
  "Setup", accessed 2026-10-12, snapshot `papers/blogs/puckerflow_repo_readme.txt`.
- Exact quote:
  > pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
- Shows vs infers: SHOWS. INFERENCE: a two-stage pipeline needs a separate environment (as ET-Flow did in round 3).
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-044
- Claim: RINGER represents macrocycles in redundant internal coordinates with fixed bond lengths, diffuses bond angles
  and torsions jointly with a wrapped normal, and rebuilds Cartesians by an SLSQP-constrained least-squares fit
  (Eq. 3) because a cycle with fixed bonds has three redundant torsions and two redundant angles. Partly reused from
  E-GEOM-033 (p. 5, VERIFIED).
- Source type: paper
- Locator: paper: `papers/related/2023_grambow_ringer.pdf`, PDF p. 4 (representation, wrapped-normal kernel) and p. 5 (Eq. 3).
- Exact quote:
  > Macrocycles with fixed bond distances contain three redundant torsional angles and two redundant bond angles.
  > we treat this as a constrained optimization problem using the Sequential Least Squares Quadratic Programming (SLSQP) algorithm66 to ensure valid Cartesian coordinates while satisfying distance constraints
- Shows vs infers: SHOWS (macrocycles, CREMP).
- Supports card(s): C-ARCH-02, C-ARCH-04
- Confidence: high

### E-ARCH-045
- Claim: In RINGER's benchmark, TD retrained on CREMP inherits RDKit's ring backbone distribution and only becomes
  competitive when seeded with macrocycle-tuned ETKDG; default seeding gives very poor performance.
- Source type: paper
- Locator: paper: `papers/related/2023_grambow_ringer.pdf`, PDF p. 6 (Structural analysis) and p. 19 (SI baselines).
- Exact quote:
  > (note that TorDiff relies on RDKit for backbone sampling, and hence produces an identical plot)
  > which generates low-quality conformers for macrocycles and leads to very poor performance.
- Shows vs infers: SHOWS (macrocycles). INFERENCE: the L source dominates TD outside drug-like rings too.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-046
- Claim: TMCgen (PuckerFlow's group, 2026) is a manifold diffusion that combines simulation-free diffusion on S²
  (metal–ligand coordination directions, closed-form heat kernel, scores projected on the tangent space) with ligand
  rotation and torsion diffusion, updating each factor separately; the authors state the n-sphere generalisation could
  serve the conformer ensembles of molecular ring systems.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_tmcgen.pdf`, PDF p. 5 (§2.1) and p. 6 (§2.2, Eq. 3–4).
- Exact quote:
  > we apply updates to the diffusion over the coordination environment, the rotations and the torsions for each ligand separately
  > such as for the conformational ensembles of molecular ring systems, we focus on the 2-sphere in the main text.
- Shows vs infers: SHOWS. INFERENCE (Q1): "angle-type manifold factor + torsions in one diffusion" already exists
  (for metal complexes); a FlexiTors claim must be the ring-pucker factor for organic molecules on GEOM, and the same
  group signals that direction.
- Supports card(s): C-ARCH-02
- Confidence: high

### E-ARCH-047
- Claim: FoldingDiff diffuses six internal angles per residue (three bond angles, three dihedrals) with a wrapped
  normal, rebuilds Cartesians by NeRF with bond lengths fixed to averages, and reports a lever-arm effect where early
  errors alter the whole structure.
- Source type: paper
- Locator: paper: `papers/related/2022_wu_foldingdiff.pdf`, PDF p. 4 (§3) and p. 10 (Discussion).
- Exact quote:
  > converted back to 3D Cartesian coordinates by iteratively adding atoms to the protein backbone as described in Parsons et al. (2005), fixing bond distances to average lengths
  > errors early in the chain to significantly alter the overall generated structure - a sort of "lever arm effect."
- Shows vs infers: SHOWS (proteins).
- Supports card(s): C-ARCH-04
- Confidence: high

### E-ARCH-049
- Claim: When RINGER's authors retrained TD on CREMP, they built its training local structures from the ground-truth
  xTB-optimised conformers followed by MMFF optimisation (to handle the RDKit → xTB distribution shift), and seeded test
  local structures with macrocycle-tuned RDKit + MMFF.
- Source type: paper
- Locator: paper: `papers/related/2023_grambow_ringer.pdf`, PDF p. 19 (SI baselines, "Torsional Diffusion") and p. 18.
- Exact quote:
  > we seeded the training local structures using the ground-truth xTB-optimized conformers with subsequent MMFF94 force-field optimization.
- Shows vs infers: SHOWS (macrocycles). INFERENCE: practitioners already match TD's training L to a physics-processed
  source; with S5 (E-ARCH-006) this is the empirical case for C-ARCH-05.
- Supports card(s): C-ARCH-05
- Confidence: high

### E-ARCH-048
- Claim: Torsional-GFN samples torsions conditioned on the graph and the local structure L and follows the shift of
  torsional energy modes when L changes (unseen MD bond lengths/angles), i.e. p(τ | L) depends on L.
- Source type: paper
- Locator: paper: `papers/related/2025_ezzine_torsional_gfn.pdf`, PDF p. 2 (Eq. 2) and p. 4 (Results).
- Exact quote:
  > Torsional-GFN is able to follow the shift of the modes in the energy landscape when sampling torsion angles for unseen local structures.
- Shows vs infers: SHOWS (a handful of small molecules, energy-trained).
- Supports card(s): C-ARCH-01, C-ARCH-05
- Confidence: medium (small study)

## Papers: Cartesian flow / diffusion conformer models (the competitors)

### E-ARCH-050
- Claim: ET-Flow is Cartesian conditional flow matching: linear interpolant x_t = t x1 + (1 − t) x0 with noise
  σ√(t(1 − t)) z, target v = x1 − x0 + (1 − 2t)/(2√(t(1 − t))) z (Eq. 2–5), a harmonic prior rotationally Kabsch-aligned
  to each data sample, an equivariant-transformer vector field, stochastic sampling only in t ∈ [0.8, 1], and a post
  hoc chirality flip.
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 4 (§3.1–3.2, Eq. 1–5), p. 5 (§3.3–3.5).
- Exact quote:
  > we reduce the transport costs between samples from the harmonic prior 0 and samples from the data distribution 1 by rotationally aligning them using the Kabsch algorithm
- Shows vs infers: SHOWS (ρ glyphs lost).
- Supports card(s): C-ARCH-01, C-ARCH-03, C-ARCH-06
- Confidence: high

### E-ARCH-051
- Claim: In ET-Flow's DRUGS ablation (reduced 50-epoch setting), replacing the harmonic prior by a Gaussian prior
  raises AMR-R from 0.557 to 0.640 Å and removing alignment to 0.622; ET-Flow on QM9 trains 200 epochs on 4 A100 GPUs.
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 19, Table 7 (rows read with `pdftotext -raw`);
  PDF p. 16, App. A (training details).
- Exact quote:
  > ET-Flow (Gaussian Prior) 66.53 73.01 0.640 0.625 44.41 40.88 0.903 0.864
  > For GEOM-QM9, we train ET-Flow for 200 epochs with a batch size of 128, and use all of the training dataset per epoch on 4 A100 GPUs.
- Shows vs infers: SHOWS. INFERENCE: the source distribution of a flow matters a lot; a prior that already carries
  chemistry (harmonic, or RDKit L in C-ARCH-01) is the cheap lever.
- Supports card(s): C-ARCH-01, C-ARCH-06
- Confidence: high

### E-ARCH-052
- Claim: Published TD-protocol QM9 numbers (δ = 0.5 Å, GeoMol split, 2K samples): ET-Flow AMR-R 0.073 (re-runs 0.076–
  0.083), MCF-B 0.103, DMT-B 0.090, AvgFlowDiT 0.082, DiTMC 0.068–0.073, TD 0.178. Reused from E-EVAL-001…005, 008
  (VERIFIED, V1) plus DiTMC (new).
- Source type: paper
- Locator: paper: `papers/related/2025_frank_ditmc.pdf`, PDF p. 6, Table 1 (rows read with `pdftotext -raw`); others as
  in E-EVAL-001 (ET-Flow p. 7), E-EVAL-002 (MCF p. 6), E-EVAL-003 (NExT-Mol p. 8), E-EVAL-004 (AvgFlow p. 7).
- Exact quote (DiTMC Table 1, raw):
  > DiTMC+PE(3)-B (8.6M) 95.7 100.0 0.068 0.021 93.4 100.0 0.089 0.032
  > Tors. Diff. (1.6M) 92.8 100.0 0.178 0.147 92.7 100.0 0.221 0.195
- Shows vs infers: SHOWS. DiTMC uses the GeoMol split and 2K samples (p. 6, §5) and averages three seeds.
- Supports card(s): C-ARCH-01, C-ARCH-03, C-ARCH-06
- Confidence: high

### E-ARCH-053
- Claim: On GEOM-DRUGS (TD protocol, δ = 0.75 Å), AMR-R is TD 0.582, ET-Flow-SS 0.439, DiTMC 0.415–0.444, MCF-L
  0.390, DMT-L 0.375; DiTMC trains QM9 for 250 epochs (≈ 2 H100-days) and DRUGS under a 9 H100-day budget.
- Source type: paper
- Locator: paper: `papers/related/2025_frank_ditmc.pdf`, PDF p. 8, Table 3 (raw extraction), PDF p. 25 ("Compute Budget
  and Training Times").
- Exact quote:
  > Tors. Diff. (1.6M) 72.7 80.0 0.582 0.565 55.2 56.9 0.778 0.729
  > All models on GEOM-QM9 are trained for 250 epochs, which requires 2 days of training on Nvidia H100 GPU for aPE and rPE models
- Shows vs infers: SHOWS. INFERENCE: on DRUGS the Cartesian frontier is ≈ 0.38–0.44 Å vs TD's 0.58, so a TD-based
  method must cut ≈ 25–35 % of TD's DRUGS error to be competitive.
- Supports card(s): C-ARCH-01, C-ARCH-06
- Confidence: high

### E-ARCH-054
- Claim: FM-refiner is upstream-agnostic: its training base is the data plus Gaussian noise (x0 = x1 + σε, σ = 1,
  Kabsch-aligned), linear interpolant with ET-Flow-style noise schedule (Eq. 4–9); at test it starts the ODE from any
  upstream sample; implemented by fine-tuning ET-Flow's architecture and weights.
- Source type: paper
- Locator: paper: `papers/related/2025_xu_fm_refiner.pdf`, PDF p. 4 (§4.1, Eq. 4–9), p. 5 (Alg. 1), p. 7 (§5.1).
- Exact quote:
  > Since our goal is to refine upstream conformers rather than regenerate them, we depart from the usual pure-noise initialization and adopt a base distribution x0 that retains the ground-truth signal.
  > We implement our refiner by fine-tuning the open-source ET-Flow (Hassan et al., 2024) architecture and weights.
- Shows vs infers: SHOWS. Mechanism of starting from upstream samples and the reachable-range limit are already in
  E-GEOM-029 / E-GEOM-030 (VERIFIED).
- Supports card(s): C-ARCH-01, C-ARCH-03
- Confidence: high

### E-ARCH-055
- Claim: FM-refiner's QM9 gains (TD protocol, δ = 0.05 Å COV) are ET-Flow 0.083 → 0.076, DMT 0.087 → 0.070, MCF-B
  0.101 → 0.100 AMR-R; the paper has a TD row (0.178) but no "TD + Refiner" row, and on DRUGS it refines MCF-L,
  ET-Flow and DMT-L but not TD.
- Source type: paper
- Locator: paper: `papers/related/2025_xu_fm_refiner.pdf`, PDF p. 7, Tables 1–2 (raw extraction).
- Exact quote:
  > ET-Flow + Refiner 78.40 88.89 0.076 0.028 77.36 89.94 0.103 0.031
  > Torsional Diff. - - 0.178 0.147 - - 0.221 0.195
- Shows vs infers: SHOWS. INFERENCE (Q1): "TD + learned Cartesian refiner" is untested in the literature, so C-ARCH-03
  is an application, not prior art; the paper states no code release (searched the full text for "github").
- Supports card(s): C-ARCH-03
- Confidence: high

### E-ARCH-058
- Claim: FM-refiner argues its updates are small for samples already near a target basin, so it avoids basin
  switching and preserves the upstream model's diversity.
- Source type: paper
- Locator: paper: `papers/related/2025_xu_fm_refiner.pdf`, PDF p. 6 (§4.2, "Representation mismatch implications").
- Exact quote:
  > If a generated conformer is already near a target basin, the predicted velocities are low-magnitude, which ensures a small update and avoids basin switching, therefore preserving diversity.
- Shows vs infers: SHOWS (an argument plus empirical fits in their §5.5). INFERENCE: the opposite of our A3 result
  for MMFF-after (E-ARCH-013) is claimed for a learned refiner; untested on TD samples.
- Supports card(s): C-ARCH-03
- Confidence: medium

### E-ARCH-056
- Claim: MCF argues scaling a non-equivariant Cartesian model beats domain-specific inductive biases; DMT denoises
  Gaussian perturbations of atom coordinates with relational attention; EBD starts its reverse process from an RDKit
  fragment-level coarse structure instead of noise. Partly reused from E-GEOM-051 (EBD, VERIFIED).
- Source type: paper
- Locator: paper: `papers/related/2023_wang_mcf.pdf` PDF p. 1 (Abstract); `papers/related/2025_liu_nextmol.pdf` PDF p. 4
  (Fig. 2 caption); `papers/related/2024_park_equivariant_blurring_diffusion.pdf` PDF p. 2 (§1).
- Exact quote:
  > scaling up the model capacity leads to large gains in generalization performance without enforcing inductive biases like rotational equivariance
  > we design EBD to generate 3D molecular conformers from coarse-grained fragment approximated structures as Fig. 1, rather than from random noise.
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01, C-ARCH-06
- Confidence: high

### E-ARCH-057
- Claim: AvgFlow (SO(3)-averaged flow matching + reflow/distillation) and EnFlow (flow matching + a learned energy
  guiding sampling) are both Cartesian-coordinate methods aimed at speed or energy calibration, not at a separate
  local-structure factor.
- Source type: paper
- Locator: paper: `papers/related/2025_cao_avgflow.pdf` PDF p. 1 (Abstract); `papers/related/2025_xu_enflow.pdf` PDF p. 1 (Abstract).
- Exact quote:
  > For fast training, we introduce the SO(3)-Averaged Flow training objective, which leads to faster convergence to better generation quality compared to conditional optimal transport flow or Kabsch-aligned flow.
  > By integrating generative dynamics with a learned energy model, EnFlow guides sampling toward low-energy regions of the conformational landscape
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-06
- Confidence: high

## Papers: manifold / prior techniques

### E-ARCH-060
- Claim: HarmonicFlow's harmonic prior p0(x0) ∝ exp(−½ x0ᵀ L x0) (L the bond-graph Laplacian) samples bonded atoms
  close together; with the conditional path N(t x1 + (1 − t) x0, σ²) and an x1-predicting network it outperforms an
  isotropic Gaussian start.
- Source type: paper
- Locator: paper: `papers/related/2023_stark_harmonicflow.pdf`, PDF p. 4 (§3.1 "Harmonic Prior").
- Exact quote:
  > We choose a harmonic prior as in EigenFold (Jing et al., 2023) that samples atoms to be close to each other if they are connected by a bond.
  > Notably, we find it helpful to parameterize v to predict x1 instead of (x1 - x0).
- Shows vs infers: SHOWS.
- Supports card(s): C-ARCH-01, C-ARCH-06
- Confidence: high

### E-ARCH-061
- Claim: Riemannian flow matching is simulation-free on manifolds with closed-form geodesics — explicitly including the
  torus and any product of such spaces — with conditional paths x_t = exp_{x1}(κ(t) log_{x1}(x0)), κ(t) = 1 − t
  (Eq. 15).
- Source type: paper
- Locator: paper: `papers/related/2023_chen_riemannian_flow_matching.pdf`, PDF p. 2 (§1) and p. 5 (Eq. 15, Prop. 3.2).
- Exact quote:
  > On simple geometries, where geodesics are known in closed form (e.g., Euclidean space, hypersphere, hyperbolic space, torus, or any of their product spaces), Riemannian Flow Matching remains completely simulation-free.
- Shows vs infers: SHOWS. INFERENCE: T^m (torsions) × R^{2k} (CP coordinates as Euclidean pairs, PuckerFlow-style) ×
  R^{3n} (an L bridge) is such a product, so a flow version of TD with extra factors needs no simulation in training.
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: high

## Papers found in the 2024–2026 search

### E-ARCH-062
- Claim: DAgger: when the learner's outputs determine its future inputs the i.i.d. assumption fails, and the fix is to
  train on the states the learned policy itself induces, labelled by the expert. Reused from E-ROBUST-044 / -045
  (VERIFIED, V2).
- Source type: paper
- Locator: paper: `papers/related/2011_ross_dagger.pdf`, PDF p. 1 (§1) and p. 3 (§3 "Dataset Aggregation").
- Exact quote:
  > DAGGER (Dataset Aggregation), an iterative algorithm that trains a deterministic policy that achieves good performance guarantees under its induced distribution of states.
- Shows vs infers: SHOWS. INFERENCE: in a two-stage or joint L + τ model, the torsion head must be trained on L
  produced by the L generator itself (E-ARCH-006 is our in-domain evidence).
- Supports card(s): C-ARCH-01, C-ARCH-02, C-ARCH-05
- Confidence: high

### E-ARCH-063
- Claim: Measured on gnode118, one GFN2-xTB optimisation of a QM9 seed takes ≈ 0.14 s median (≤ 0.4 s) on one core,
  and one 100-epoch QM9 TD training ≈ 11.7 GPU-h; GT conformers do not move under xtb 6.7.1 optimisation (10/10).
- Source type: our-data (round-3 code grounder measurements)
- Locator: our-data: `round3/grounding.md:35-37` ("Cost basis") and §2.2; GT stability in `round3/impact.md` §5 B.3.
- Exact quote (round3/grounding.md):
  > one GFN2-xTB optimisation of a QM9 seed ≈ 0.14 s median, ≤ 0.4 s, single core.
- Shows vs infers: SHOWS (grounder measurements; not re-checked by a verifier).
- Supports card(s): C-ARCH-02, C-ARCH-05
- Confidence: medium

### E-ARCH-070
- Claim: Input perturbation (DDPM-IP) has already been applied to TD: on the ConfGF/GeoDiff protocol (40k training
  molecules × 5 conformers, 200 test molecules, TD retrained) it lowers QM9 MAT mean from 0.2379 to 0.1924 Å.
- Source type: paper
- Locator: paper: `papers/related/2024_wang_td_input_perturbation.pdf` (arXiv 2409.14014, SMC 2024), PDF p. 4 (§IV.B,
  Eq. 4) and p. 5 (§V.A, Table II); text from `pdftotext -raw`.
- Exact quote:
  > Tor. Diff. 83.07 100.00 0.2379 0.2085
  > Tor. Diff.-IP 84.49 100.00 0.1924 0.1530
- Shows vs infers: SHOWS (GD-P protocol; not comparable with TD-P). INFERENCE (Q1): "noise/perturbation robustness for
  TD" is prior art; our S2 (σ 0.04 Å L-jitter) failed on RDKit L (+0.025, BRIEF), so a robustness claim must be
  L-specific, not generic input perturbation.
- Supports card(s): C-ARCH-05
- Confidence: high

### E-ARCH-071
- Claim: CurvFM adds a residual between a frozen ML Hessian predictor evaluated at the generated endpoint and at the
  reference to the flow-matching loss, and on GEOM-QM9 reduces the median deviation in maximum force by 63 % at
  unchanged global structural accuracy.
- Source type: paper
- Locator: paper: `papers/related/2026_darouich_curvfm.pdf` (arXiv 2609.34592), PDF p. 1 (Abstract), p. 2 (§1).
- Exact quote:
  > CurvFM substantially improves physical fidelity at unchanged global structural accuracy, reducing the median deviation in maximum force by 63% on GEOM-QM9
- Shows vs infers: SHOWS. INFERENCE: physics-informed losses on the generated endpoint improve local geometry without
  moving RMSD metrics; an optional regulariser for an L generator, not a route to lower AMR.
- Supports card(s): C-ARCH-01
- Confidence: medium (own split, not TD-P)

### E-ARCH-072
- Claim: A targeted arXiv search (2024-01 to 2026-10; queries on conformer + torsion + ring/bond angle, conformer
  generation + flow matching/diffusion + GEOM, "torsional diffusion", "internal coordinates", "Cremer-Pople", "local
  structure") found no paper that generates ring puckers (or bond angles) jointly with TD-style torsions for organic
  small molecules; the closest are PuckerFlow (rings only, two-stage demo with TD), TMCgen (angles + torsions for metal
  complexes), GO-Flow (internal coordinates without ring closure), FM-refiner (Cartesian refiner, no TD row).
- Source type: our-data (search log)
- Locator: our-data: `round4/ledger/arch_data/search_log.txt` (queries and hits; script `arxq.py`, `run_search.sh`);
  web searches of 2026-10-11/12 (WebSearch, extended) for "ring puckering torsional diffusion joint generation" on
  ChemRxiv/OpenReview returned no such paper.
- Exact quote (search_log.txt, Cremer-Pople query):
  > === QUERY: abs:"Cremer-Pople"
- Shows vs infers: SHOWS what the searches returned; absence of evidence, not proof (arXiv-centric; ChemRxiv and
  conference-only papers may be missed). For the novelty red-teamer to re-run.
- Supports card(s): C-ARCH-01, C-ARCH-02
- Confidence: medium

## Blogs / technical posts / talks
