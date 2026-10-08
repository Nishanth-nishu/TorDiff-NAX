# Evidence ledger: SCOUT ROBUST (round 3, P1)

Lens: how to train the torsion score model so it exploits good local structure L yet tolerates imperfect L (BRIEF Q2).
Written 2026-10-08 by scout ROBUST. Paths are relative to `C:\Users\HP\Desktop\tor_diff`.

Conventions
- Paper pages are PDF pages; printed page = PDF page unless stated. Fulltexts of the papers added in this round were
  extracted with `pdftotext` in reading order (no `-layout`), so two-column text is contiguous; form feeds separate pages.
  Quotes are matched after collapsing whitespace. Glyphs lost in extraction are noted under the quote.
- All "our-data" numbers are PRELIMINARY round-2 values (unpaired means over each run's own molecules, BRIEF).
  ORACLE marks conditions built from test-set true geometry; none is used for model selection.
- Derived tables T1-T3 come from `round3/ledger/robust_data/lerror_stats.py` (reads only existing files; rerun from
  the repo root); output saved as `round3/ledger/robust_data/lerror_stats.txt`. T3 values are means of the per-seed
  `summary.txt` lines (MAT-R_mean = AMR-R).
- Reused round-2 entries are re-checked against the fulltext in this round (whitespace-collapsed match).
- P2 (2026-10-08): after round3/ledger/verify_V2.md, entries E-006, E-007, E-008 were revised; E-010 got an addendum;
  locators were fixed in E-043, E-046, E-047, E-049, E-058; a note was added to E-018; E-060 and E-061 were added.
  Each change is marked `Revised after D-1xx` / `Added after D-1xx`, with the superseded wording under History. Tables
  T4a-c and T5 were appended to lerror_stats.py/.txt; T1-T3 are byte-identical to the version V2 checked.
  `robust_data/check_quotes.py` re-checks every quote (61/61 found).

---------------------------------------------------------------------------------------------------------------------
## A. Our data and code

### E-ROBUST-001
- Claim: A torsion model trained on true L (B1) is much worse than the RDKit-L-trained control on RDKit ETKDG L: AMR-R 0.2356 (3 training seeds, 0.2327-0.2381) vs CTRL_rematch 0.1779 (3 seeds), 936 molecules.
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 (from cluster_sync/round2/results/qm9_B1_train_gtL_e100_s{0,1,2}/S1_etkdg2L/summary.txt, MAT-R_mean = 0.2361 / 0.2381 / 0.2327, and qm9_CTRL_rematch_100ep_e100_s{0,1,2}/S1_etkdg2L/summary.txt, MAT-R_mean = 0.1774 / 0.1779 / 0.1783)
- Exact quote (T3 lines):
  > qm9_B1_train_gtL                S1_etkdg2L        3    0.2356   0.2327   0.2381     936
  > qm9_CTRL_rematch_100ep                S1_etkdg2L        3    0.1779   0.1774   0.1783     936
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-002
- Claim: On true L (λ = 1, each molecule's own GT conformers cycled; ORACLE) B1 reaches AMR-R 0.0199 (seed 0 only) vs CTRL_rematch 0.0805 (3 seeds), 955 molecules.
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 (qm9_B1_train_gtL_e100_s0/S1_lam1.00_cyc_ORACLE/summary.txt MAT-R_mean = 0.0199; qm9_CTRL_rematch_100ep_e100_s{0,1,2}/S1_lam1.00_cyc_ORACLE/summary.txt MAT-R_mean = 0.0803 / 0.0804 / 0.0808)
- Exact quote (T3 lines):
  > qm9_B1_train_gtL     S1_lam1.00_cyc_ORACLE        1    0.0199   0.0199   0.0199     955
  > qm9_CTRL_rematch_100ep     S1_lam1.00_cyc_ORACLE        3    0.0805   0.0803   0.0808     955
- Shows vs infers: SHOWS (ORACLE condition)
- Supports card(s): none (context only)
- Confidence: high (single B1 seed at λ = 1)

### E-ROBUST-003
- Claim: Isotropic Gaussian jitter of true L during training (S2, σ = 0.04 Å/axis, 3 seeds) only partly removes B1's brittleness: RDKit L 0.2060 (vs B1 0.2356, CTRL 0.1779) and true L cycled 0.0541 (vs B1 0.0199).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 (cluster_sync/round2/results/qm9_B6_jit0.04pa_e100_s{0,1,2}/steps20_seed0/summary.txt MAT-R_mean = 0.2078 / 0.2037 / 0.2064; .../steps20_seed0_gtLcycle/summary.txt MAT-R_mean = 0.0532 / 0.0533 / 0.0557)
- Exact quote (T3 lines):
  > qm9_B6_jit0.04pa             steps20_seed0        3    0.2060   0.2037   0.2078     935
  > qm9_B6_jit0.04pa    steps20_seed0_gtLcycle        3    0.0541   0.0532   0.0557     996
- Shows vs infers: SHOWS (the RDKit-L test set here is the in-job one, n = 935, not the S1 seed pickle, n = 936)
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-004
- Claim: A per-sample 50/50 mix of the paired RDKit-matched and true L (S3, seed 0 only) gives AMR-R 0.1816 on RDKit L and 0.0331 on true L cycled, against CTRL_rematch 0.0822 on true L cycled (3 seeds).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 (cluster_sync/round2/results/qm9_S3_mixL_p0.5_e100_s0/steps20_seed0/summary.txt MAT-R_mean = 0.1816; .../steps20_seed0_gtLcycle/summary.txt MAT-R_mean = 0.0331; qm9_CTRL_rematch_100ep_e100_s{0,1,2}/steps20_seed0_gtLcycle MAT-R_mean = 0.0816 / 0.0827 / 0.0824)
- Exact quote (T3 lines):
  > qm9_S3_mixL_p0.5             steps20_seed0        1    0.1816   0.1816   0.1816     935
  > qm9_S3_mixL_p0.5    steps20_seed0_gtLcycle        1    0.0331   0.0331   0.0331     996
  > qm9_CTRL_rematch_100ep    steps20_seed0_gtLcycle        3    0.0822   0.0816   0.0827     996
- Shows vs infers: SHOWS (one seed; S3's RDKit-L control in the same in-job setting is CTRL_rematch 0.179, BRIEF)
- Supports card(s): C-ROBUST-01, C-ROBUST-02, C-ROBUST-03, C-ROBUST-04, C-ROBUST-05
- Confidence: medium (1 seed)

### E-ROBUST-005
- Claim: On the λ-blend series (ORACLE) B1 overtakes CTRL_rematch only at λ = 0.75 (0.1146 vs 0.1156); at λ = 0.5 B1 is still 0.045 Å worse (0.1812 vs 0.1364).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 (S1_lam0.50_cyc_ORACLE and S1_lam0.75_cyc_ORACLE summaries, 3 seeds each)
- Exact quote (T3 lines):
  > qm9_B1_train_gtL     S1_lam0.50_cyc_ORACLE        3    0.1812   0.1800   0.1830     906
  > qm9_B1_train_gtL     S1_lam0.75_cyc_ORACLE        3    0.1146   0.1116   0.1165     906
  > qm9_CTRL_rematch_100ep     S1_lam0.50_cyc_ORACLE        3    0.1364   0.1354   0.1372     906
  > qm9_CTRL_rematch_100ep     S1_lam0.75_cyc_ORACLE        3    0.1156   0.1146   0.1164     906
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-02, C-ROBUST-04
- Confidence: high

### E-ROBUST-006
Revised after D-108 (2026-10-08). Original claim kept under History.
- Claim: Isotropic jitter does not reproduce RDKit's L error structure. Its decisive gap is the ring tail. On the 936 common molecules, endocyclic-dihedral RMSD > 10° occurs in 28.6% of ETKDG ring seeds and 30.8% of matched RDKit (λ = 0) seeds, against 0.35% under 0.04 Å/axis noise. Among seeds of molecules with a ring of ≥ 4 atoms it is 49.6% (ETKDG) vs 0.58% (noise). In bond and angle RMSD, noise 0.04 is larger than RDKit in bonds only. Against the matched RDKit L that CTRL/S3 train on (heavy bonds 0.0344 Å, heavy angles 3.94°), noise 0.04 (0.0549 Å, 3.31°) is 1.60× the bond error and 0.84× the angle error. Against the ETKDG seed rows (0.0314 Å, 2.80°) it is 1.75× / 1.18×, but those rows are biased low: per GT conformer, the ETKDG seed is chosen by a Hungarian assignment that minimises heavy-atom angle RMSD (tools/make_l_seed_pickles.py:106-114).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T4a (common 936 molecules) and T4b (ring ≥ 4 split); T1 for the all-molecule rows (from cluster_sync/round2/data/QM9/round2_seeds/l_error.csv; columns bond_rmsd_heavy_any, angle_rmsd_heavy_any, ring_dihedral_rmsd, computed by tools/lgeom.py:317-337)
- Exact quote (T4a and T4b lines):
  > L_etkdg2L                    bond_heavy=0.0314 angle_heavy=2.796 angle_heavy_acyc=2.836 ringdih_gt10=0.2856
  > L_lam0.00_cyc_ORACLE         bond_heavy=0.0344 angle_heavy=3.939 angle_heavy_acyc=4.067 ringdih_gt10=0.3076
  > L_noise0.04pa_cyc_ORACLE     bond_heavy=0.0549 angle_heavy=3.311 angle_heavy_acyc=3.371 ringdih_gt10=0.0035
  > L_etkdg2L                    only_3_rings: n=3078 max_err=0.000000 | ring_ge4: n=4181 frac_gt10=0.4961
  > L_noise0.04pa_cyc_ORACLE     only_3_rings: n=6178 max_err=0.000000 | ring_ge4: n=8736 frac_gt10=0.0058
- Shows vs infers: SHOWS (the numbers; the Hungarian selection is in tools/make_l_seed_pickles.py:106-114, comment "Hungarian assignment GT conformer j <- seed on heavy angle RMSD"). Reading the ring tail as wrong ring conformations is INFERENCE (median 0.44° vs mean 10.65°, T1)
- History: original claim (superseded): "Isotropic jitter does not reproduce RDKit's L error structure. ETKDG L vs GT: heavy bonds 0.031 Å, heavy angles 2.80°, endocyclic dihedrals mean 10.65° / median 0.44° with 28.6% of ring seeds > 10°. Noise 0.04 Å/axis: bonds 0.055 Å (1.75×), angles 3.31° (1.18×), endocyclic dihedrals mean 1.93° with 0.34% > 10°." The T1 lines it quoted stay valid: `L_etkdg2L 12655 0.0314 2.7963 ...` and `L_noise0.04pa_cyc_ORACLE 25762 0.0549 3.3077 ...`.
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-007
Revised after D-101 (2026-10-08). Original claim kept under History.
- Claim: ETKDG ring seeds exceed 10° endocyclic-dihedral RMSD in 48-92% of seeds whose smallest ring has 4 to ≥ 7 atoms (4: 60%, 5: 48%, 6: 67%, ≥ 7: 92%; not monotone in ring size). Three-membered rings carry no such error by construction: every endocyclic dihedral of a 3-ring is identically 0, and all 3078 seeds of 3-ring-only molecules have error 0. Under 0.04 Å/axis noise every bin is ≤ 3.3%.
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T2 (frac_gt10 by smallest SSSR ring for L_etkdg2L and L_noise0.04pa_cyc_ORACLE) and T4b (3-ring-only seeds)
- Exact quote (T2, L_etkdg2L rows for rings 4 and 6; T4b line):
  > 4         1079  15.208  13.674      0.604
  > 6          337  39.030  40.197      0.665
  > L_etkdg2L                    only_3_rings: n=3078 max_err=0.000000 | ring_ge4: n=4181 frac_gt10=0.4961
- Shows vs infers: SHOWS
- History: original claim (withdrawn wording): "RDKit's endocyclic-dihedral error grows with ring size: share of ETKDG ring seeds with error > 10° is 9% (smallest ring 3), 60% (4), 48% (5), 67% (6), 92% (≥ 7); the same shares under 0.04 Å/axis noise are 0.0-3.3%." V2 (D-101) is right on both points: "grows" is false between 4 and 5, and the 9% for ring 3 is a dilution artefact. Recomputed: 3078 of the 4312 ring-3 seeds are 3-ring-only with error 0 (T4b).
- Supports card(s): C-ROBUST-01, C-ROBUST-02
- Confidence: high (ring size is the smallest ring of the SMILES, capped at 7)

### E-ROBUST-008
Revised after D-102 (2026-10-08). Original claim withdrawn, kept under History.
- Claim: B1 is hurt more by A5ring L (GT rings + RDKit-matched acyclic geometry) than by GT + 0.04 Å/axis noise. Its AMR-R is 0.1877 vs 0.1536, and every A5ring seed (0.1813-0.1922) is worse than every noise seed (0.1376-0.1635). CTRL_rematch is hurt equally by both (0.1186 vs 0.1186). The two L conditions differ in error profile rather than size. On the 955 A5 molecules, noise 0.04 has 2.0× A5ring's heavy-bond error (0.0549 vs 0.0270 Å), but 0.94× its heavy-angle error (3.31 vs 3.51°) and 0.82× its heavy acyclic-angle error (3.37 vs 4.11°), plus some ring error (1.92° vs 0).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 rows qm9_B1_train_gtL / qm9_CTRL_rematch_100ep for S1_A5ring_cyc_ORACLE and S1_noise0.04pa_cyc_ORACLE; T4c (L error on the common 955 molecules)
- Exact quote (T3 and T4c lines):
  > qm9_B1_train_gtL      S1_A5ring_cyc_ORACLE        3    0.1877   0.1813   0.1922     955
  > qm9_B1_train_gtL S1_noise0.04pa_cyc_ORACLE        3    0.1536   0.1376   0.1635     996
  > qm9_CTRL_rematch_100ep      S1_A5ring_cyc_ORACLE        3    0.1186   0.1183   0.1189     955
  > qm9_CTRL_rematch_100ep S1_noise0.04pa_cyc_ORACLE        3    0.1186   0.1173   0.1196     996
  > L_A5ring_cyc_ORACLE          bond_heavy=0.0270 angle_heavy=3.51 angle_heavy_acyc=4.11 angle_all_acyc=3.60 ringdih_mean=0.00
  > L_noise0.04pa_cyc_ORACLE     bond_heavy=0.0549 angle_heavy=3.31 angle_heavy_acyc=3.37 angle_all_acyc=3.98 ringdih_mean=1.92
- Shows vs infers: SHOWS (unpaired: AMR-R on 955 vs 996 molecules, the 955 a subset; no per-molecule outputs synced). It does NOT show that RDKit-type error is worse "per unit" of error than random error. Attributing B1's extra loss to the structured (RDKit-matched) acyclic error is INFERENCE, consistent with the larger heavy acyclic-angle error of A5ring.
- Supports card(s): C-ROBUST-01
- Confidence: medium
- History: original claim (withdrawn after D-102): "Per unit of bond/angle error, RDKit-derived (systematic) acyclic error hurts B1 more than isotropic noise: B1 on GT rings + RDKit-matched acyclic geometry (A5ring: bonds 0.027 Å, angles 3.51°, ring dihedrals 0) scores 0.1877, while B1 on GT + 0.04 Å/axis noise (bonds 0.055 Å, angles 3.31°, ring dihedrals 1.93°) scores 0.1536." The "per unit" and "larger noise" readings fail because noise 0.04 has less heavy-atom angle error than A5ring (T4c).

### E-ROBUST-009
- Claim: MMFF-relaxed ETKDG L has lower bond/angle error than raw ETKDG (bonds 0.018 Å, angles 1.97°, ring dihedrals mean 8.07°, 20% > 10°) and helps CTRL (0.1525 vs 0.1779) but not B1 (0.2323 vs 0.2356).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T1 row L_etkdg2L_mmff; T3 rows S1_etkdg2L_mmff
- Exact quote (T1 and T3 lines):
  > L_etkdg2L_mmff            12655           0.0182            1.9733               1.9126              0.0112       7263        8.0655          0.4307      25.8895       0.2945        0.2018        0.1257
  > qm9_B1_train_gtL           S1_etkdg2L_mmff        3    0.2323   0.2293   0.2354     936
  > qm9_CTRL_rematch_100ep           S1_etkdg2L_mmff        3    0.1525   0.1518   0.1528     936
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-010
- Claim: The two models exploit different parts of L (ORACLE partial-L conditions): CTRL gains from true ring geometry (A5ring 0.1186) but not from true acyclic geometry (A5acyc 0.1824); B1 gains more from true acyclic geometry (A5acyc 0.1583) than from true rings (A5ring 0.1877).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 rows S1_A5ring_cyc_ORACLE and S1_A5acyc_cyc_ORACLE (3 seeds each, 955 molecules)
- Exact quote (T3 lines):
  > qm9_B1_train_gtL      S1_A5acyc_cyc_ORACLE        3    0.1583   0.1573   0.1595     955
  > qm9_B1_train_gtL      S1_A5ring_cyc_ORACLE        3    0.1877   0.1813   0.1922     955
  > qm9_CTRL_rematch_100ep      S1_A5acyc_cyc_ORACLE        3    0.1824   0.1816   0.1837     955
  > qm9_CTRL_rematch_100ep      S1_A5ring_cyc_ORACLE        3    0.1186   0.1183   0.1189     955
- Shows vs infers: SHOWS
- Addendum after D-110 (2026-10-08; claim unchanged, VERIFIED): against the like-for-like base λ = 0 (matched RDKit seed, same 955 molecules), both models gain about equally from true rings. CTRL goes 0.1966 → 0.1186 (−0.078) and B1 0.2614 → 0.1877 (−0.074). They differ in the acyclic gain: CTRL 0.1966 → 0.1824 (−0.014), B1 0.2614 → 0.1583 (−0.103). Base rows (T3):
  > qm9_B1_train_gtL     S1_lam0.00_cyc_ORACLE        3    0.2614   0.2556   0.2680     955
  > qm9_CTRL_rematch_100ep     S1_lam0.00_cyc_ORACLE        3    0.1966   0.1964   0.1968     955
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-011
- Claim: B1's loss on RDKit L sits in molecules with rotatable bonds, not in the L floor: AMR-R for molecules with 0 rotatable heavy-atom torsions is 0.155 (B1 s0), 0.152 (CTRL_rematch s0), 0.150 (S3 s0), 0.154 (S2 s0); with ≥ 4 it is 0.394, 0.239, 0.253, 0.321.
- Source type: our-data
- Locator: our-data: cluster_sync/round2/results/qm9_B1_train_gtL_e100_s0/S1_etkdg2L/breakdown.log, qm9_CTRL_rematch_100ep_e100_s0/S1_etkdg2L/breakdown.log, qm9_S3_mixL_p0.5_e100_s0/steps20_seed0/breakdown.log, qm9_B6_jit0.04pa_e100_s0/steps20_seed0/breakdown.log; table "by n_rot_heavy", column MAT_R
- Exact quote (B1 s0, rows 0 and 4+):
  > | 0     | 284 |  97.606 |   0.155 |  96.23  |   0.19  |
  > | 4+    |  88 |  69.697 |   0.394 |  57.345 |   0.476 |
- Shows vs infers: SHOWS (single seed per model; S3/S2 rows use the in-job set of 934 molecules, B1/CTRL the S1 set of 935)
- Supports card(s): C-ROBUST-01, C-ROBUST-03
- Confidence: high

### E-ROBUST-012
- Claim: B1 is unstable across training seeds under small isotropic L noise: on GT + 0.02 Å/axis noise its AMR-R ranges 0.0918-0.1341 over 3 seeds, while CTRL_rematch ranges 0.0973-0.0977.
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 rows S1_noise0.02pa_cyc_ORACLE
- Exact quote (T3 lines):
  > qm9_B1_train_gtL S1_noise0.02pa_cyc_ORACLE        3    0.1170   0.0918   0.1341     996
  > qm9_CTRL_rematch_100ep S1_noise0.02pa_cyc_ORACLE        3    0.0976   0.0973   0.0977     996
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-05
- Confidence: high

### E-ROBUST-013
- Claim: The round-2 post-training panel for S2 (B6) and S3 contains no partial-L (A5ring/A5acyc) and no λ = 0.25/0.75 conditions, so it cannot say whether these models exploit partially correct L.
- Source type: code
- Locator: code: slurm/r2_evalsets.tsv:5-9 (set definitions) and :30-31 (S1A5 lines); slurm/r2_eval_models_post.tsv (B6 and S3 rows use set S1panel)
- Exact quote (slurm/r2_evalsets.tsv:7):
  > #   S1panel = 6 conditions (no lam 0.25/0.75)            -> CB s0, B6 x3, S3 x2
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-014
- Claim: The round-2 training-L options are per-sample switches in the data transform: S2 adds fresh isotropic jitter to all atoms, S4 blends the paired RDKit-matched and GT positions with λ ~ U[0,1] and stores λ as a node feature, S3 picks one of the two paired L by a Bernoulli draw.
- Source type: code
- Locator: code: torsional-diffusion/utils/dataset.py:59-63 (S2), :90-95 (S4), :96-98 (S3)
- Exact quote (dataset.py:93-95, 98):
  > lam = float(np.random.uniform())
  > data.pos = interp_x_torch(x, y, lam, data.edge_index)
  > data.node_lambda = lam * torch.ones(data.num_nodes)
  > data.pos = y if np.random.uniform() < self.l_mix_p_gt else x
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01, C-ROBUST-02, C-ROBUST-03, C-ROBUST-04, C-ROBUST-05
- Confidence: high

### E-ROBUST-015
- Claim: The repo already has a function that builds ring/acyclic hybrid geometries (keep the rings of one geometry, copy acyclic bonds and non-ring angles from another), used for the A5 test seeds; it ran at about 2 ms per call in a local timing by the scout.
- Source type: code
- Locator: code: tools/make_l_seed_pickles.py:236-237 (A5ring / A5acyc use it); tools/lgeom.py:341-368 (set_internal_subset). Timing: scout's local run (100 calls on 5 QM9-sized molecules, 2.18 ms per call), not a stored result file: UNVERIFIED on the cluster.
- Exact quote (tools/make_l_seed_pickles.py:236-237):
  > for tag, fn in (('L_A5ring_cyc_ORACLE', lambda X, Y: lgeom.set_internal_subset(base, Y, X)),
  > ('L_A5acyc_cyc_ORACLE', lambda X, Y: lgeom.set_internal_subset(base, X, Y))):
- Shows vs infers: SHOWS (code); the timing is a scout measurement
- Supports card(s): C-ROBUST-01
- Confidence: high (code), medium (timing)

### E-ROBUST-016
- Claim: The TD training target is the torsion update applied to the selected conformer, so the "clean" torsions a model learns to return to are those of whatever L-carrying conformer is selected (matched torsions for RDKit-matched L, true torsions for GT L).
- Source type: code
- Locator: code: torsional-diffusion/utils/dataset.py:74-76
- Exact quote:
  > torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
  > data.pos = modify_conformer(data.pos, data.edge_index.T[edge_mask], mask_rotate, torsion_updates)
  > data.edge_rotate = torch.tensor(torsion_updates)
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01, C-ROBUST-03, C-ROBUST-04, C-ROBUST-05
- Confidence: high

### E-ROBUST-017
- Claim: S4's λ enters the score model exactly where σ enters, and at inference one CLI λ is applied to every node of every molecule, so a λ-conditioned model cannot adapt the level per molecule or per ring.
- Source type: code
- Locator: code: torsional-diffusion/diffusion/sampling.py:181-183; torsional-diffusion/diffusion/score_model.py:194-199; torsional-diffusion/generate_confs.py:52
- Exact quote (sampling.py:181-182):
  > # [round2 S4] constant CLI lambda for every node of the batch; never per-molecule (code_plan_2 V14)
  > data_gpu.node_lambda = float(l_level) * torch.ones(data.num_nodes, device=device)
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-02, C-ROBUST-04
- Confidence: high

### E-ROBUST-018
- Claim: S4's λ conditioning is switched on only by `--lambda_embed_dim 32` on top of `--l_interp`; with the default lambda_embed_dim = 0 the architecture is the unconditioned one, so an S4 run without the λ input needs no code change.
- Source type: code
- Locator: code: slurm/ablations_train_round2.tsv:15-17 (S4 lines); torsional-diffusion/diffusion/score_model.py:65-70
- Exact quote (slurm/ablations_train_round2.tsv:15):
  > S4_lamcond       | paired | --l_interp --lambda_embed_dim 32 --log_timing --fail_on_nan | --l_level 0    | 2 | 0 | --l_level 1
- Shows vs infers: SHOWS (score_model.py:65 comment: "0 = no extra input, identical layer shapes"); that `--l_interp` alone trains cleanly is INFERENCE (no assert couples the two flags in train.py or utils/dataset.py; not run). Note added after V2 review: a blind training line must also drop `--l_level 0` / `--l_level 1` from the in-job generate fields, because generate_confs.py:107-110 exits when `--l_level` is given to a model without λ conditioning (config, not code)
- Supports card(s): C-ROBUST-02
- Confidence: medium

### E-ROBUST-019
- Claim: The existing ring/acyclic hybrid builder leaves a residual error in the geometry it copies: in the A5acyc test seeds (true acyclic geometry requested) the all-atom acyclic angle RMSD to GT is 0.58°, not 0.
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T1 row L_A5acyc_cyc_ORACLE, column angle_all_acyc_mean = 0.5834
- Exact quote (T1 line):
  > L_A5acyc_cyc_ORACLE       12795           0.0140            1.2608               0.5834              0.0000       7375        9.2834          0.5854      29.2790       0.3738        0.3035        0.1784
- Shows vs infers: SHOWS (tools/lgeom.py:345-346 documents the cause: sequential setting at branching centres)
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-057
- Claim: The paired-pickle builder takes any standardized-pickle directory, and MMFF-matched standardized pickles already exist as a training variant (S5/B3), so an MMFF-matched/GT paired cache can be built with the existing tool.
- Source type: code
- Locator: code: tools/build_paired_pickles.py:50; slurm/ablation_train_array.sbatch:57
- Exact quote (tools/build_paired_pickles.py:50):
  > parser.add_argument('--std_dir', default='data/QM9/standardized_pickles_rematch')
- Shows vs infers: SHOWS (that the pairing checks pass at a similar rate for MMFF-relaxed matched conformers is UNVERIFIED)
- Supports card(s): C-ROBUST-03
- Confidence: medium

### E-ROBUST-059
- Claim: CTRL_rematch with MMFF relaxation of its own ETKDG seeds at inference (`--pre_mmff`, the S5 control) scores AMR-R 0.1540 (3 seeds, 935 molecules).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T3 (cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s{0,1,2}/S5ctrl_pre_mmff/summary.txt MAT-R_mean = 0.1536 / 0.1531 / 0.1554)
- Exact quote (T3 line):
  > qm9_CTRL_rematch_100ep           S5ctrl_pre_mmff        3    0.1540   0.1531   0.1554     935
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-058
- Claim: S4 and S3 do not train on the same conformers: the pair_ok filter (dropping about 8% of conformers) runs only when `--l_interp` is set, so S3 keeps the pairs that S4 drops; S3 vs S4 therefore differs in data as well as in the continuum and the λ input.
- Source type: code
Revised after D-107 (locator only).
- Locator: code: torsional-diffusion/utils/dataset.py:145-162; round2/DECISION.md:27 (user ruling, about 8% of conformers)
- Exact quote (dataset.py:145-147):
  > if getattr(transform, 'l_interp', False):
  > # [round2 S4] DECISION D1: pairs that fail the pair_ok guard (stereo / inversion / bond / clash at
  > # lambda = 0.5) are never interpolated: drop those conformers, and molecules left with none; counted.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-02
- Confidence: high

### E-ROBUST-060
Added after D-109 / V2 recompute 5 (2026-10-08).
- Claim: Like-for-like round-1 in-job values on the same in-job test sets as S2/S3 (RDKit L n = 935, gtLcycle n = 996, 3 training seeds each): CTRL_rematch on RDKit L 0.1786; B1 on RDKit L 0.2341; B1 on true L cycled 0.0212 (ORACLE). Against them, S3 (seed 0) is 0.003 Å behind CTRL on RDKit L (0.1816) and 0.012 Å behind B1 on true L (0.0331), and S2 is 0.028 Å better than B1 on RDKit L (0.2060).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T5 (from cluster_sync/results/qm9_{CTRL_rematch_100ep,B1_train_gtL}_e100_s{0,1,2}/steps20_seed0{,_gtLcycle}/summary.txt)
- Exact quote (T5 lines):
  > qm9_B1_train_gtL                 steps20_seed0            n_seeds=3 amr_mean=0.2341 amr_min=0.2299 amr_max=0.2364 n_mols=935
  > qm9_B1_train_gtL                 steps20_seed0_gtLcycle   n_seeds=3 amr_mean=0.0212 amr_min=0.0206 amr_max=0.0215 n_mols=996
  > qm9_CTRL_rematch_100ep           steps20_seed0            n_seeds=3 amr_mean=0.1786 amr_min=0.1773 amr_max=0.1800 n_mols=935
- Shows vs infers: SHOWS (S3 and S2 values from E-ROBUST-004 / E-ROBUST-003)
- Supports card(s): C-ROBUST-01, C-ROBUST-05
- Confidence: high (S3 itself 1 seed)

### E-ROBUST-061
Added after D-109 (2026-10-08).
- Claim: At S3's two endpoints, training-seed spread of the reference models is small (0.001-0.007 Å range over 3 seeds): CTRL_rematch on RDKit L 0.1773-0.1800 (in-job) / 0.1774-0.1783 (S1 pickle); B1 on RDKit L 0.2299-0.2364 (in-job); B1 on true L cycled 0.0206-0.0215; CTRL_rematch on true L cycled 0.0816-0.0827. S3's own seed spread is unknown (1 seed).
- Source type: our-data
- Locator: our-data: round3/ledger/robust_data/lerror_stats.txt, T5 rows (in-job) and T3 rows S1_etkdg2L / steps20_seed0_gtLcycle
- Exact quote (T5 and T3 lines):
  > qm9_CTRL_rematch_100ep           steps20_seed0            n_seeds=3 amr_mean=0.1786 amr_min=0.1773 amr_max=0.1800 n_mols=935
  > qm9_B1_train_gtL                 steps20_seed0_gtLcycle   n_seeds=3 amr_mean=0.0212 amr_min=0.0206 amr_max=0.0215 n_mols=996
  > qm9_CTRL_rematch_100ep    steps20_seed0_gtLcycle        3    0.0822   0.0816   0.0827     996
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-05
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## B. Papers

### E-ROBUST-020
- Claim: TD reports that training on ground-truth L creates a test-time distribution shift that significantly hurts performance.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 7, Section 4.1 "Conformer matching"
- Exact quote:
  > there will be a distributional shift at test time, where only approximate local structures from p^G(L) are available. We found that this shift significantly hurts performance.
- Shows vs infers: SHOWS
- Reused from: round2/research_A.md E1 (VERIFIED in round2/verify_A.md); quote re-checked in papers/core/2022_jing_torsional_diffusion.fulltext.txt p. 7
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-021
- Claim: TD's conformer matching assigns RDKit L to GT conformers so that training and inference see the same L distribution.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 19, Appendix E (conformer matching)
- Exact quote:
  > The complete assignment resulting from the linear sum solution guarantees that there is no distributional shift in the local structures seen during training and inference.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-022
- Claim: TD frames the L shift as domain adaptation, solved by aligning the true and RDKit L distributions (and matching torsions to minimise RMSD).
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 8, Section 4.1
- Exact quote:
  > Instead, we view the distributional shift as a domain adaptation problem that can be solved by optimally aligning pG(L) and p^G(L).
  (subscripts and hat as extracted in the fulltext)
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-023
- Claim: CDM: conditioning augmentation works because it alleviates compounding error from train-test mismatch in cascades.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 3, Section 1
- Exact quote:
  > We empirically find that conditioning augmentation is effective because it alleviates compounding error in cascading pipelines due to train-test mismatch
- Shows vs infers: SHOWS
- Reused from: round2/research_A.md E5 (VERIFIED); re-checked in papers/related/fulltext/2021_ho_cascaded_diffusion.txt p. 3
- Supports card(s): C-ROBUST-01, C-ROBUST-03
- Confidence: high

### E-ROBUST-024
- Claim: CDM: the mismatch arises when stage-1 samples are out of distribution for a stage-2 model trained on ground truth.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 18, Section 4.3
- Exact quote:
  > This occurs when low- resolution model samples are out of distribution compared to the ground truth data on which the super-resolution model is trained.
- Shows vs infers: SHOWS (the analogy to B1 is INFERENCE)
- Reused from: round2/research_A.md E7 (VERIFIED); re-checked p. 18
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-025
- Claim: CDM: the right augmentation type depends on the upstream error: Gaussian noise helped at low resolution but similar augmentation gave negative results at 128×128/256×256, where Gaussian blur was used instead.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 18, Section 4.4 "Experiments at 128×128 and 256×256"
- Exact quote:
  > While we found Gaussian noise augmentation to be a key ingredient to boost the performance of our cascaded models at low resolutions, our initial experiments with similar augmentations for 128×128 and 256×256 upsampling yielded negative results.
  (the × glyph is extracted as U+FFFD in the fulltext)
- Shows vs infers: SHOWS (that the augmentation must match OUR error structure is INFERENCE by analogy)
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-026
- Claim: CDM applied blurring augmentation to 50% of training examples and none at inference: a clean/corrupted mixture of the conditioning input.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 7, Section 3.1 "Blurring Augmentation"
- Exact quote:
  > During training, we apply this blurring augmentation to 50% of the examples. During inference, no augmentation is applied to low resolution inputs.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01, C-ROBUST-05
- Confidence: high

### E-ROBUST-027
- Claim: CDM: models can be amortised over the augmentation strength and the strength picked after training.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 6, Section 3
- Exact quote:
  > In some cases, we found it more practical to train super-resolution models amortized over the strength of conditioning augmentation and pick the best strength in a post-training hyperparameter search
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-02
- Confidence: high

### E-ROBUST-028
- Claim: CDM's non-truncated conditioning augmentation corrupts the stage-1 sample with the same forward noise at sampling time, so train and test inputs carry the same corruption.
- Source type: paper
- Locator: paper: papers/related/2021_ho_cascaded_diffusion.pdf, PDF p. 8, Section 3.3
- Exact quote:
  > in non-truncated conditioning augmentation we always sample z0 using the full, non-truncated low resolution reverse process; then we corrupt z0 using the forward process into zs
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-029
- Claim: DDPM-IP perturbs ground-truth inputs during training to simulate inference-time prediction errors.
- Source type: paper
- Locator: paper: papers/related/2023_ning_input_perturbation.pdf, PDF p. 1, Abstract (two-column; the round-2 layout fulltext interleaves the columns, so this round adds a reading-order extraction, papers/related/fulltext/2023_ning_input_perturbation.readingorder.txt, where the quote is contiguous)
- Exact quote:
  > consisting in perturbing the ground truth samples to simulate the inference time prediction errors.
  (line-break hyphen "pre- diction" joined; two-column page, read in column order)
- Shows vs infers: SHOWS (DDPM-IP perturbs the denoiser's own noisy input, not a conditioning variable: analogy only, as round2/verify_A.md notes)
- Reused from: round2/research_A.md E9 (VERIFIED); re-checked in a reading-order extraction of the PDF
- Supports card(s): C-ROBUST-01
- Confidence: medium (analogy)

### E-ROBUST-030
- Claim: DDPM-IP describes input perturbation as a regulariser that smooths the network's prediction function.
- Source type: paper
- Locator: paper: papers/related/2023_ning_input_perturbation.pdf, PDF p. 2, Section 1
- Exact quote:
  > which forces the network to smooth its prediction function
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-031
- Claim: ProteinMPNN: training with 0.02 Å Gaussian noise on backbone coordinates improved sequence recovery on AlphaFold models but decreased it on unperturbed PDB structures (robustness traded for accuracy on exact inputs).
- Source type: paper
- Locator: paper: papers/related/2022_dauparas_proteinmpnn.pdf (bioRxiv 2022.06.03.494563 v1), PDF p. 7
- Exact quote:
  > We found that training models on backbones to which Gaussian noise (std=0.02Å) had been added improved sequence recovery on confident protein structure models generated by AlphaFold (average pLDDT>80.0) from UniRef50
  (Å may be extracted as U+FFFD)
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01, C-ROBUST-05
- Confidence: high

### E-ROBUST-032
- Claim: ProteinMPNN attributes the exact-structure advantage to information about the target hidden in exact coordinates ("memory") that predicted structures lack, and prefers the robust model.
- Source type: paper
- Locator: paper: papers/related/2022_dauparas_proteinmpnn.pdf, PDF p. 7
- Exact quote:
  > Crystallographic refinement may impart some memory of amino acid identity in the backbone coordinates which is recovered by the model trained on perfect backbones but not present in predicted structures
- Shows vs infers: SHOWS (that B1's own-conformer advantage is the same kind of leak, L carrying a fingerprint of its conformer's torsions, is INFERENCE; round 1 measured B1 0.0369 with another GT conformer's L vs 0.0212 with its own, BRIEF / round2/research_A.md D3)
- Supports card(s): C-ROBUST-01
- Confidence: high (quote), medium (analogy)

### E-ROBUST-033
- Claim: ProteinMPNN Table 1 (experiment 4): noise-trained vs native-trained sequence recovery 47.9 vs 50.8 on PDB test structures and 48.5 vs 46.9 on AlphaFold models.
- Source type: paper
- Locator: paper: papers/related/2022_dauparas_proteinmpnn.pdf, PDF p. 3, Table 1 (left value = native coordinates, right = 0.02 Å noise)
- Exact quote:
  > Experiment 4 Experiment 3 with random instead of forward decoding 1.678 mln 50.8/47.9 4.74/5.25 46.9/48.5
  (table row in the reading-order extraction, whitespace collapsed; columns: parameters, PDB test accuracy, PDB perplexity, AlphaFold-model accuracy)
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-05
- Confidence: high

### E-ROBUST-034
- Claim: SliDe argues that assuming isotropic coordinate noise (coordinate denoising) is an inappropriate assumption for molecules that biases the learned force field.
- Source type: paper
- Locator: paper: papers/related/2024_ni_sliced_denoising.pdf (ICLR 2024, arXiv 2311.02124), PDF p. 2, Section 1
- Exact quote:
  > due to inappropriate assumptions such as assuming a molecular force field is isotropic in coordinate denoising (Zaidi et al., 2022)
- Shows vs infers: SHOWS (SliDe's task is pre-training, not robustness of a conditional model: analogy)
- Supports card(s): C-ROBUST-01
- Confidence: high (quote), medium (transfer)

### E-ROBUST-035
- Claim: SliDe's "BAT noise" perturbs bond lengths, angles and torsions separately, with variances set by force-field parameters, and approximates the true molecular distribution better than isotropic noise.
- Source type: paper
- Locator: paper: papers/related/2024_ni_sliced_denoising.pdf, PDF p. 2, Section 1
- Exact quote:
  > BAT noise introduces Gaussian noise to bond lengths, angles, and torsion angles, and their respective variances are predetermined by parameters within the energy function. This approach allows BAT noise to better approximate the true molecular distribution
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01
- Confidence: high

### E-ROBUST-036
- Claim: Domain randomisation: train on enough variability that the real target looks like one more variation.
- Source type: paper
- Locator: paper: papers/related/2017_tobin_domain_randomization.pdf (IROS 2017, arXiv 1703.06907), PDF p. 1, Abstract
- Exact quote:
  > With enough variability in the simulator, the real world may appear to the model as just another variation.
- Shows vs infers: SHOWS (images/robotics; transfer to L sources is INFERENCE)
- Supports card(s): C-ROBUST-01, C-ROBUST-03
- Confidence: high (quote), medium (transfer)

### E-ROBUST-037
- Claim: Removing noise-level conditioning from denoising generative models mostly causes graceful degradation, and sometimes improves results.
- Source type: paper
- Locator: paper: papers/related/2025_sun_noise_conditioning.pdf (ICML 2025, arXiv 2502.13129), PDF p. 1, Abstract
- Exact quote:
  > To our surprise, most models exhibit graceful degradation, and in some cases, they even perform better without noise conditioning.
- Shows vs infers: SHOWS (for the diffused variable's noise level; our λ is the quality of the L part of the same input: analogy)
- Supports card(s): C-ROBUST-02
- Confidence: high (quote), medium (transfer)

### E-ROBUST-038
- Claim: Sun et al.: the noise level is inferable from the input because p(t|z) is concentrated, and this concentration depends on data dimensionality.
- Source type: paper
- Locator: paper: papers/related/2025_sun_noise_conditioning.pdf, PDF p. 4, Section 4.2 "Concentration of Posterior p(t|z)"
- Exact quote:
  > We note that the concentration of ppt|zq depends on data dimensionality
  ("ppt|zq" is the extraction of p(t|z))
- Shows vs infers: SHOWS (that a QM9 molecule, about 18 atoms, is low-dimensional enough to blur p(λ|L) is INFERENCE)
- Supports card(s): C-ROBUST-02
- Confidence: high

### E-ROBUST-039
- Claim: Sun et al.: conditioning on a noise level predicted by a separate pre-trained network behaves like no conditioning at all.
- Source type: paper
- Locator: paper: papers/related/2025_sun_noise_conditioning.pdf, PDF p. 8, Section 6.2 (variants b-d, Fig. 7)
- Exact quote:
  > Notably, consistent behavior is observed for all models (iDDPM, EDM, and FM) studied here: the results of (b), (c), and (d) are similar.
- Shows vs infers: SHOWS ((b) = level predicted by a pre-trained network, (d) = no conditioning)
- Supports card(s): C-ROBUST-02
- Confidence: high

### E-ROBUST-040
- Claim: Scheduled sampling: the train/inference input discrepancy (true vs model-generated inputs) makes errors accumulate.
- Source type: paper
- Locator: paper: papers/related/2015_bengio_scheduled_sampling.pdf (NeurIPS 2015, arXiv 1506.03099), PDF p. 1, Abstract
- Exact quote:
  > This discrepancy between training and inference can yield errors that can accumulate quickly along the generated sequence.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-05
- Confidence: high

### E-ROBUST-041
- Claim: Scheduled sampling is a curriculum that gradually moves training from true inputs to the model's own outputs.
- Source type: paper
- Locator: paper: papers/related/2015_bengio_scheduled_sampling.pdf, PDF p. 2, Section 1
- Exact quote:
  > We propose to change the training process in order to gradually force the model to deal with its own mistakes, as it would have to during inference.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-05
- Confidence: high

### E-ROBUST-042
- Claim: Huszár shows that the scheduled-sampling objective is improper and gives an inconsistent learning algorithm.
- Source type: paper
- Locator: paper: papers/related/2015_huszar_scheduled_sampling_critique.pdf (arXiv 1511.05101), PDF p. 1, Abstract
- Exact quote:
  > the objective function underlying scheduled sampling is improper and leads to an inconsistent learning algorithm.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-03, C-ROBUST-05
- Confidence: high

### E-ROBUST-043
- Claim: Huszár: scheduled sampling pushes models toward a trivial solution that ignores the content of the conditioning prefix.
- Source type: paper
Revised after D-103 (locator only).
- Locator: paper: papers/related/2015_huszar_scheduled_sampling_critique.pdf, PDF p. 4, Section 4.1 "Scheduled sampling formulated as KL divergence minimisation" (section starts p. 3)
- Exact quote:
  > Based on this analysis we suggest that scheduled sampling works by pushling models towards a trivial solution of memorising distribution of symbols conditioned on their position in the sequence,
  ("pushling" sic)
- Shows vs infers: SHOWS (that heavy RDKit-L mixing could push the torsion model to ignore L detail is INFERENCE)
- Supports card(s): C-ROBUST-05
- Confidence: high (quote), medium (transfer)

### E-ROBUST-044
- Claim: DAgger: when the learner's own outputs determine its future inputs, the i.i.d. assumption fails and errors compound.
- Source type: paper
- Locator: paper: papers/related/2011_ross_dagger.pdf (AISTATS 2011, arXiv 1011.0686), PDF p. 1, Section 1
- Exact quote:
  > since the learner's prediction affects future input observations/states during execution of the learned policy, this violate the crucial i.i.d. assumption made by most statistical learning approaches.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-045
- Claim: DAgger trains on states the learned policy itself induces, labelled by the expert, and obtains guarantees under that induced distribution.
- Source type: paper
- Locator: paper: papers/related/2011_ross_dagger.pdf, PDF p. 3, Section 3 "Dataset Aggregation"
- Exact quote:
  > DAGGER (Dataset Aggregation), an iterative algorithm that trains a deterministic policy that achieves good performance guarantees under its induced distribution of states.
- Shows vs infers: SHOWS (mapping "expert relabelling" onto conformer matching of a new L source is INFERENCE)
- Supports card(s): C-ROBUST-03
- Confidence: high

### E-ROBUST-046
- Claim: Classifier-free guidance: an unconditional-training probability of 0.5 was consistently worse than 0.1 or 0.2, which performed about equally.
- Source type: paper
Revised after D-104 (locator only).
- Locator: paper: papers/related/2022_ho_classifier_free_guidance.pdf (arXiv 2207.12598), PDF p. 8, Section 4.2 "Varying the unconditional training probability"
- Exact quote:
  > We find puncond = 0.5 consistently performs worse than puncond {0.1, 0.2} across the entire IS/FID frontier; puncond {0.1, 0.2} perform about equally as well as each other.
  (p_uncond and the ∈ sign are lost in extraction)
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-04, C-ROBUST-05
- Confidence: high

### E-ROBUST-047
- Claim: Classifier-free guidance increases sample fidelity at the expense of diversity.
- Source type: paper
Revised after D-105 (locator only).
- Locator: paper: papers/related/2022_ho_classifier_free_guidance.pdf, PDF p. 9, Section 5 "Discussion" (last paragraph)
- Exact quote:
  > any guidance method that increases sample fidelity at the expense of diversity must face the question of whether decreased diversity is acceptable.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-04
- Confidence: high

### E-ROBUST-048
- Claim: Classifier-free guidance trains one network on both conditional and unconditional scores by replacing the condition with a null token.
- Source type: paper
- Locator: paper: papers/related/2022_ho_classifier_free_guidance.pdf, PDF p. 4, Section 3.2
- Exact quote:
  > for the unconditional model we can simply input a null token for the class identifier c when predicting the score
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-04
- Confidence: high

### E-ROBUST-049
- Claim: Summing or mixing diffusion scores and running the ordinary reverse process does not sample the composed distribution.
- Source type: paper
Revised after D-106 (locator only).
- Locator: paper: papers/related/2023_du_reduce_reuse_recycle.pdf (ICML 2023, arXiv 2302.11552 v6), PDF p. 5, Section 4 "Scaling Compositional Generation with Diffusion Models" (before 4.1)
- Exact quote:
  > does not correspond to sampling from the composed model, and thus reverse diffusion sampling will generate incorrect samples from composed distributions.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-04
- Confidence: high

---------------------------------------------------------------------------------------------------------------------
## C. Blogs (intuition and practice, not proof)

### E-ROBUST-050
- Claim: Score matching learns the score poorly where training data are sparse, because the loss is weighted by the data density.
- Source type: blog
- Locator: blog: https://yang-song.net/blog/2021/score/, Yang Song, 2021-05-05, section "Naive score-based generative modeling and its pitfalls", accessed 2026-10-08, snapshot papers/blogs/song_2021_score.txt (post source file; the live site refused connections on the access date)
- Exact quote:
  > The key challenge is the fact that the estimated score functions are inaccurate in low density regions, where few data points are available for computing the score matching objective.
- Shows vs infers: SHOWS (for the diffused variable; that a B1 trained only on GT L has no data, hence an unreliable score, at RDKit-like L is INFERENCE)
- Supports card(s): C-ROBUST-01
- Confidence: medium

### E-ROBUST-051
- Claim: Noise scale trade-off: larger perturbations cover more of the low-density region but corrupt the data more; smaller ones corrupt less but cover less.
- Source type: blog
- Locator: blog: https://yang-song.net/blog/2021/score/, Yang Song, 2021-05-05, section "Score-based generative modeling with multiple noise perturbations", accessed 2026-10-08, snapshot papers/blogs/song_2021_score.txt
- Exact quote:
  > Larger noise can obviously cover more low density regions for better score estimation, but it over-corrupts the data and alters it significantly from the original distribution.
- Shows vs infers: SHOWS (S2's σ = 0.04 trade-off, E-ROBUST-003, is the same pattern: INFERENCE)
- Supports card(s): C-ROBUST-01
- Confidence: medium

### E-ROBUST-052
- Claim: Practitioners guide domain randomisation because overly wide randomisation distributions can hinder learning.
- Source type: blog
- Locator: blog: https://lilianweng.github.io/posts/2019-05-05-domain-randomization/, Lilian Weng, 2019-05-05, section "Guided Domain Randomization", accessed 2026-10-08, snapshot papers/blogs/weng_2019_domain_randomization.txt
- Exact quote:
  > Another is to avoid infeasible solutions that might arise from overly wide randomization distributions and thus might hinder successful policy learning.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01
- Confidence: medium

### E-ROBUST-053
- Claim: Real-data-guided domain randomisation fits the randomisation parameters so that simulated data match the real distribution.
- Source type: blog
- Locator: blog: https://lilianweng.github.io/posts/2019-05-05-domain-randomization/, Lilian Weng, 2019-05-05, section "Match Real Data Distribution", accessed 2026-10-08, snapshot papers/blogs/weng_2019_domain_randomization.txt
- Exact quote:
  > In the case of real-data-guided DR, we would like to learn the randomization parameters $\xi$ that bring the state distribution in simulator close to the state distribution in the real world.
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-01
- Confidence: medium

### E-ROBUST-054
- Claim: Dieleman: guidance improves adherence to the condition and sample quality at a large cost in diversity.
- Source type: blog
- Locator: blog: https://sander.ai/2022/05/26/guidance.html, Sander Dieleman, 2022-05-26, section "Classifier-free guidance", accessed 2026-10-08, snapshot papers/blogs/dieleman_2022_guidance.txt
- Exact quote:
  > Clearly, guidance represents a trade-off: it dramatically improves adherence to the conditioning signal, as well as overall sample quality, but at great cost to diversity
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-04
- Confidence: medium

### E-ROBUST-055
- Claim: Dieleman: in practice, conditioning dropout removes the condition 10-20% of the time.
- Source type: blog
- Locator: blog: https://sander.ai/2022/05/26/guidance.html, Sander Dieleman, 2022-05-26, section "Classifier-free guidance", accessed 2026-10-08, snapshot papers/blogs/dieleman_2022_guidance.txt
- Exact quote:
  > some percentage of the time, the conditioning information \(y\) is removed (10-20% tends to work well).
- Shows vs infers: SHOWS
- Supports card(s): C-ROBUST-04, C-ROBUST-05
- Confidence: medium

### E-ROBUST-056
- Claim: Weng's summary of CDM: the most effective conditioning noise depends on the stage (Gaussian noise at low resolution, blur at high), and it is applied only in training.
- Source type: blog
- Locator: blog: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/, Lilian Weng, 2021-07-11, section "Scale up Generation Resolution and Quality", accessed 2026-10-08, snapshot papers/blogs/weng_2021_diffusion_models.txt
- Exact quote:
  > They found the most effective noise is to apply Gaussian noise at low resolution and Gaussian blur at high resolution.
- Shows vs infers: SHOWS (secondary source for E-ROBUST-025)
- Supports card(s): C-ROBUST-01
- Confidence: medium
