# Evidence ledger: SCOUT GEOM (round 3, P1)

Lens: sources of local structure L at test time, especially ring geometry (brief Q1, Q3).
Agent tag GEOM. Template: `orchestration/templates/evidence_entry.md`. Entries are appended incrementally.

## Our data

Reproduction for E-GEOM-001..004 (local, RDKit 2026.03.6, pandas): read
`cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, group by `cond`, average the named columns over seeds (rows);
"ring molecules" = SMILES with at least one RDKit ring (885 of 996 molecules); smallest ring size from
`Chem.MolFromSmiles(smiles).GetRingInfo().AtomRings()`. Scripts: scratchpad `lerr.py`, `lerr2.py` (contents are the
groupby described here; no other processing). Note the reference conformer for `L_etkdg2L` / `L_etkdg2L_mmff` rows is
the GT conformer Hungarian-assigned on heavy-atom angle RMSD (E-GEOM-009), so their ring-dihedral errors are per
assigned pair, not "best pucker over all seeds".

### E-GEOM-001
- Claim: Mean per-seed L error (heavy-atom ring bonds Å / ring angles ° / endocyclic ring dihedrals °, all seeds) is:
  ETKDG 0.0318 / 2.38 / 10.65; MMFF-relaxed ETKDG 0.0199 / 1.42 / 8.07; λ=0.50 0.0189 / 1.34 / 3.16; λ=0.75 0.0101 /
  0.67 / 1.56; true L + noise 0.02 Å/axis 0.0267 / 1.38 / 0.96; noise 0.04 Å/axis 0.0534 / 2.77 / 1.93. Medians of the
  ring dihedral are much smaller (ETKDG 0.44°, MMFF 0.43°, λ=0.75 0.12°), i.e. the ETKDG/MMFF mean is a tail.
  Heavy-atom acyclic bond Å / angle °: ETKDG 0.0292 / 2.84; MMFF 0.0170 / 2.03; λ=0 0.0322 / 4.12; λ=0.50 0.0172 /
  2.01; λ=0.75 0.0090 / 1.00; noise 0.02 0.0269 / 1.67; noise 0.04 0.0538 / 3.37; A5ring 0.0322 / 4.11; A5acyc 0 / 0.48.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, columns `bond_rmsd_heavy_ring`,
  `angle_rmsd_heavy_ring`, `ring_dihedral_rmsd`, `bond_rmsd_heavy_acyc`, `angle_rmsd_heavy_acyc`, groupby `cond`
  mean / median (row counts: etkdg 12655, mmff 12655, lam0.50 12707, lam0.75 12707, noise 25762 each).
- Exact quote (header line of the file):
  > cond,smiles,k,src_gt_idx,bond_rmsd_all_ring,angle_rmsd_all_ring,...,angle_rmsd_heavy_any,ring_dihedral_rmsd
- Shows vs infers: SHOWS (computed values).
- Supports card(s): C-GEOM-01, C-GEOM-03, C-GEOM-05
- Confidence: high

### E-GEOM-002
- Claim (original, superseded where the revision below differs): Among seeds of ring molecules, the fraction with endocyclic ring-dihedral RMSD > 10° is ETKDG 28.6%, MMFF
  20.2%, λ=0.25 22.3%, λ=0.50 11.2%, λ=0.75 0.01%, noise 0.02 Å 0.00%, noise 0.04 Å 0.34%; > 20°: ETKDG 17.9%, MMFF
  12.6%, λ=0.50 0.04%, λ=0.75 0%. Ring-pucker (dihedral) error, not bond/angle size, is what separates ETKDG/MMFF from
  λ≥0.5 quality.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, `(ring_dihedral_rmsd > 10).mean()` per
  `cond`, ring molecules only (n seeds: etkdg 7259, mmff 7259, lam0.75 7285, noise0.02 14914).
- Exact quote: (computed; same file and header as E-GEOM-001)
  > ring_dihedral_rmsd
- Shows vs infers: SHOWS (fractions). INFERENCE in the last sentence (the separating variable), based on bond/angle
  errors of MMFF ≈ λ=0.50 (E-GEOM-001) while dihedral tails differ.
- Supports card(s): C-GEOM-01, C-GEOM-02, C-GEOM-03, C-GEOM-04, C-GEOM-05, C-GEOM-06
- Confidence: high (numbers); medium (inference)
- Revised after D-201..D-209 thread (coordinator note on grouping; recomputed by GEOM, scratchpad `p2.py`): a
  3-membered ring's "dihedral" (a,b,c,a) is identically 0, so the 113 molecules whose rings are all 3-membered add
  only zeros. Excluding them (715 molecules, 4181 ETKDG rows): share > 10° is ETKDG 49.6%, MMFF 35.1%, λ=0 52.1%,
  λ=0.25 38.6%, λ=0.50 19.4%, λ=0.75 0.02%, noise 0.02 Å 0%, noise 0.04 Å 0.58%; > 20°: ETKDG 31.1%, MMFF 21.8%,
  λ=0.50 0.07%, λ=0.75 0%. The ordering ETKDG > MMFF > λ0.5 >> λ0.75 is unchanged; cards now quote both bases.

### E-GEOM-003
- Claim: By smallest ring size, ETKDG median ring-dihedral RMSD and fraction > 10° are: 3-ring 0.0° / 9%; 4-ring
  13.7° / 60%; 5-ring 9.3° / 48%; 6-ring 40.2° / 66%; 7-ring 82.3° / 92% (n molecules 381 / 176 / 175 / 77 / 16).
  MMFF: 4-ring 8.3° / 44%; 5-ring 4.7° / 37%; 6-ring 5.7° / 45%; 7-ring 55.1° / 58%. λ=0.75: 0.0 / 3.4 / 2.6 / 2.8 /
  4.5° and 0% > 10° in every size. Ring bond error is flat across sizes (ETKDG 0.030–0.038 Å).
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, ring molecules, groupby smallest ring
  size, columns `ring_dihedral_rmsd` (median, share > 10), `bond_rmsd_heavy_ring` (mean).
- Exact quote: (computed; same file and header as E-GEOM-001)
  > ring_dihedral_rmsd
- Shows vs infers: SHOWS. Caveat: 4-membered rings (176 molecules) are outside PuckerFlow's training domain
  (E-GEOM-020); 3-rings have no pucker.
- Supports card(s): C-GEOM-02, C-GEOM-04, C-GEOM-06
- Confidence: high

### E-GEOM-004
- Claim (original, superseded where the revision below differs): For rigid molecules (0 rotatable heavy-atom bonds; 284–318 of ~936–996 molecules, ≈30%), MAT-R (= AMR-R)
  depends on the L source and not on the torsion model: CTRL_rematch / B1 (3-seed means) = ETKDG 0.152 / 0.155; MMFF
  0.128 / 0.127; λ=0.50 0.107 / 0.109; λ=0.75 0.077 / 0.074; true ring geometry only (A5ring) 0.043 / 0.048; true L +
  noise 0.02 Å 0.032 / 0.029; true L 0.005 (CTRL) / 0.001 (B1 seed 0). So the rigid subset is a model-free, CPU-only
  test bed for any L source.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/qm9_{CTRL_rematch_100ep,B1_train_gtL}_e100_s{0,1,2}/<cond>/breakdown.log`,
  table "by n_rot_heavy", bin `0`, column `MAT_R`, mean over the 3 seed files (B1 λ=1.00: s0 only).
- Exact quote (CTRL s0, S1_etkdg2L/breakdown.log, by n_rot_heavy):
  > | 0     | 284 |  96.549 |   0.152 |  95.321 |   0.188 |
- Shows vs infers: SHOWS (values). INFERENCE: "model-free test bed": CTRL and B1 agree within 0.006 Å in every
  condition in this bin (largest gap λ=0: 0.184 vs 0.190), and a molecule without rotatable bonds has no torsion to sample, so generated conformers equal
  the L seeds up to alignment.
- Supports card(s): C-GEOM-01, C-GEOM-02, C-GEOM-03, C-GEOM-04, C-GEOM-06
- Confidence: high
- Revised after D-205: the logged bin 0 is not exactly model-free (λ=1.00, true L, gives CTRL 0.004 / 0.004 / 0.007
  vs B1 0.001; ETKDG CTRL 0.152 / 0.151 / 0.154 vs B1 0.155 / 0.156 / 0.155 per seed, V3 recomputation), because
  `tools/breakdown.py` bins by corrected SMILES and a few bin-0 molecules have a TD-rotatable bond; and the bin-0
  population differs by L source (ETKDG/MMFF 284, λ0.25–0.75 258, λ0/λ1/A5 289, noise 318). The values above are
  therefore indicative (model dependence ≤ 0.006 Å, different molecule sets). All CPU gates in C-GEOM-01/02/04/06 now
  score the seed sets directly (heavy-atom AMR-R of seeds vs GT, no model) on one common rigid set defined by TD's own
  empty `edge_mask`, and recompute the ETKDG / MMFF / λ references on that same set.

### E-GEOM-005
- Claim: With the standard model, random isotropic L error with large bond/angle error but small ring-dihedral error
  (noise 0.02 Å/axis: ring bond 0.027 Å, angle 1.38°, ring dihedral 0.96°) gives AMR-R 0.098, better than λ=0.75 (ring
  bond 0.010 Å, angle 0.67°, ring dihedral 1.56°, systematic toward RDKit) at 0.116; MMFF (ring bond 0.020 Å, angle
  1.42°, ring dihedral 8.07°) gives 0.152.
- Source type: our-data
- Locator: our-data: `round3/BRIEF.md` table rows "True L + noise 0.02", "λ = 0.75", "MMFF-relaxed ETKDG" (CTRL_rematch
  column); L errors from E-GEOM-001.
- Exact quote (BRIEF.md line 35):
  > | True L + noise 0.02 Å/axis (ORACLE) | 0.098 | 0.117 (seeds vary 0.092–0.134) | 996 |
- Shows vs infers: SHOWS the numbers. INFERENCE: for AMR-R, getting ring dihedrals (pucker) right matters more than
  bond-length/angle accuracy at the 0.01–0.03 Å / 1–1.5° level. Caveat: populations differ (996 vs 906 molecules), the
  λ series also carries acyclic error, and these are unpaired means.
- Supports card(s): C-GEOM-01
- Confidence: medium

### E-GEOM-006
- Claim: The ring gain from true ring geometry (A5ring) for CTRL is concentrated in molecules with few rotatable bonds:
  CTRL MAT-R by n_rot_heavy bin 0 / 1 / 2 / 3 / 4+ = ETKDG 0.152 / 0.161 / 0.195 / 0.213 / 0.243 vs A5ring 0.043 / 0.104
  / 0.162 / 0.193 / 0.249 (3-seed means). True acyclic geometry (A5acyc) gives 0.180 / 0.168 / 0.193 / 0.187 / 0.212.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s{0,1,2}/{S1_etkdg2L,S1_A5ring_cyc_ORACLE,S1_A5acyc_cyc_ORACLE}/breakdown.log`,
  "by n_rot_heavy", `MAT_R`.
- Exact quote (CTRL s0, S1_A5ring_cyc_ORACLE/breakdown.log):
  > | 0     | 289 | 100     |   0.043 |  99.913 |   0.053 |
- Shows vs infers: SHOWS. Caveat: A5ring/A5acyc use the λ=0 matched-seed population (955 molecules), ETKDG uses 936.
- Supports card(s): C-GEOM-01, C-GEOM-02
- Confidence: high
- Addendum after D-201/D-207 (like-for-like base; values from the same breakdown.log files, `S1_lam0.00_cyc_ORACLE`,
  CTRL 3-seed means, 289 rigid molecules): CTRL λ=0 by bin 0 / 1 / 2 / 3 / 4+ = 0.184 / 0.181 / 0.209 / 0.209 / 0.254 →
  A5ring 0.043 / 0.104 / 0.162 / 0.193 / 0.249. The gain is still concentrated in the 0–1-rotor bins; cards now quote the
  λ=0 base (rigid 0.184 → 0.043) rather than the ETKDG base (different population and seeds).

### E-GEOM-007
- Claim: MMFF-relaxed L reaches λ≈0.5-level ring bond/angle error (0.0199 Å / 1.42° vs λ=0.50 0.0189 Å / 1.34°) but
  keeps a pucker tail (20.2% of ring seeds > 10° vs 11.2%), and its AMR-R is worse than λ=0.50 for both models (CTRL
  0.152 vs 0.136; B1 0.232 vs 0.181). The B1/CTRL crossing needs λ≈0.75-level L (CTRL 0.116 vs B1 0.115).
- Source type: our-data
- Locator: our-data: `round3/BRIEF.md` table (rows MMFF-relaxed ETKDG, λ = 0.50, λ = 0.75); E-GEOM-001/002.
- Exact quote (BRIEF.md line 33):
  > | λ = 0.75 (ORACLE) | 0.116 | 0.115 | 906 |
- Shows vs infers: SHOWS. Unpaired means; populations 936 vs 906.
- Supports card(s): C-GEOM-01, C-GEOM-03, C-GEOM-04, C-GEOM-05
- Confidence: high

## Our code (and further our-data entries E-GEOM-011..013, 049)

### E-GEOM-008
- Claim: TD embeds test-time and training-matching seeds with `EmbedMultipleConfs` without a parameter object, so the
  ETKDG variant is RDKit's keyword default; small-ring torsion terms (`useSmallRingTorsions`) are not switched on. The
  cluster pins RDKit 2022.9.5.
- Source type: code
- Locator: code: `torsional-diffusion/generate_confs.py:67-68`; `torsional-diffusion/standardize_confs.py:79`;
  `slurm/setup_env.sh:100`; round-2 seed builder `tools/make_l_seed_pickles.py:89`.
- Exact quote:
  > AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=5,
  > randomSeed=args.seed + 1 if args.seed is not None else -1)
  > rdkit==2022.9.5 \
- Shows vs infers: SHOWS. Which ETKDG version the keyword default selects in 2022.9.5 is UNVERIFIED (E-GEOM-041).
- Supports card(s): C-GEOM-06
- Confidence: high

### E-GEOM-009
- Claim: For non-oracle seeds (ETKDG, MMFF), `l_error.csv` pairs each seed with the GT conformer chosen by a Hungarian
  assignment on heavy-atom angle RMSD; ring dihedrals are not part of the assignment cost.
- Source type: code
- Locator: code: `tools/make_l_seed_pickles.py:106-114`
- Exact quote:
  > C = np.array([[lgeom.l_subset_errors(base, p, g)['angle_rmsd_heavy_any'] for p in P] for g in G])
- Shows vs infers: SHOWS. INFERENCE: for molecules whose GT ensemble holds several puckers, some ETKDG/MMFF ring-dihedral
  error may be assignment mismatch rather than a missing pucker; a pucker-coverage metric (min over seeds) is needed in
  the CPU gate of every card.
- Supports card(s): C-GEOM-01
- Confidence: high

### E-GEOM-010
- Claim: TD already ships a GFN2/xTB geometry-optimisation wrapper that calls the `xtb` binary with `--opt <level>` in a
  per-PID work directory and writes the optimised coordinates back into the RDKit conformer; it needs only an `xtb`
  executable path.
- Source type: code
- Locator: code: `torsional-diffusion/utils/xtb.py:5-7`, `:43-64`
- Exact quote:
  > def xtb_optimize(mol, level, path_xtb):
  > cmd = [path_xtb, in_path, "--opt", level]
- Shows vs infers: SHOWS. Note `xtb_energy` (`:14-15`) has an `--alpb water` switch but `xtb_optimize` has none, so it
  optimises in the gas phase.
- Supports card(s): C-GEOM-01
- Confidence: high

### E-GEOM-011
- Claim (original, superseded where the revision below differs): Of 885 test ring molecules, 420 contain only isolated rings (no two rings share an atom) and 465 contain a
  fused, bridged or spiro system. Of the 2074 ETKDG seeds with ring-dihedral RMSD > 10°, 78% are in isolated-ring
  molecules and 22% in fused/bridged/spiro ones. By (category, smallest ring): isolated 4-rings 519 bad seeds (76% of
  their seeds), isolated 5-rings 635 (49%), 6-rings 222 (67%), 7-rings 140 (92%); fused systems 449 in total. Molecules
  whose rings are all isolated 5–8-membered rings (PuckerFlow's ring-size domain) number 223 and hold 48.1% of the
  bad seeds.
- Source type: our-data
- Locator: our-data: `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, `cond == L_etkdg2L`, joined to an RDKit
  ring analysis of the SMILES (ring systems = rings merged when they share ≥ 1 atom); scratchpad `rings.py`.
- Exact quote: (computed; same file and header as E-GEOM-001)
  > ring_dihedral_rmsd
- Shows vs infers: SHOWS (counts). Caveat E-GEOM-009 (assignment on angles, not puckers).
- Supports card(s): C-GEOM-02, C-GEOM-04
- Confidence: high
- Addendum after D-209 (recomputed, `p2.py`): 53 of the 885 test ring molecules have no exocyclic heavy atom; 832
  have at least one.

### E-GEOM-012
- Claim (original, superseded where the revision below differs): The 885 test ring molecules contain 534 distinct ring-system components (ring atoms and ring bonds only, with
  atom/bond types; substituents removed). Within the test set alone, 50.8% of ring molecules have every ring-system
  component also present in another test molecule (45.2% among the 345 molecules with any ETKDG seed > 10°).
- Source type: our-data
- Locator: our-data: SMILES column of `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`; RDKit
  `MolFragmentToSmiles` over each ring system; leave-one-out count (scratchpad `rings.py`).
- Exact quote: (computed)
  > cond,smiles,k,src_gt_idx,...
- Shows vs infers: SHOWS the within-test recurrence. INFERENCE: the training split is ~107× larger (106,586 vs 1000
  molecules, `papers/INDEX.md` TD-P), so the share of test ring systems with a training-set template should be much
  higher; UNVERIFIED until counted on the cluster against the training pickles.
- Supports card(s): C-GEOM-02
- Confidence: high (numbers); low (train-set inference)
- Revised after D-207(e) (recomputed, `p2.py`): the numbers above use isomeric fragment SMILES as the key (RDKit
  default, stereo tags on ring atoms). With a stereo-free key: 421 distinct components, leave-one-out coverage 66.3% of
  ring molecules, 58.6% among the 345 molecules with an ETKDG seed > 10°. C-GEOM-02 states both keys.

### E-GEOM-013
- Claim (original, superseded where the revision below differs): Improving ring geometry alone helps the standard model but not B1: with true ring geometry and RDKit acyclic
  geometry (A5ring) CTRL reaches 0.119 and B1 0.188; with true acyclic geometry and RDKit rings (A5acyc) CTRL 0.182 and
  B1 0.158.
- Source type: our-data
- Locator: our-data: `round3/BRIEF.md` lines 37–38 (from `cluster_sync/round2/results/<run>/S1_A5*/summary.txt`).
- Exact quote:
  > | True ring geometry only (A5ring, ORACLE) | **0.119** | 0.188 | 955 |
- Shows vs infers: SHOWS (unpaired 3-seed means, preliminary). INFERENCE: a ring-only L source is a CTRL-side lever;
  a B1-side gain needs acyclic bonds/angles at near-reference quality too.
- Supports card(s): C-GEOM-01, C-GEOM-02, C-GEOM-03, C-GEOM-04
- Confidence: high
- Revised after D-201 (the original inference "helps CTRL but not B1" is WITHDRAWN): A5ring and A5acyc are built from
  the matched λ=0 pairs (`tools/make_l_seed_pickles.py:236-246`), so the like-for-like base is λ=0 on the same 955
  molecules (BRIEF line 30: CTRL 0.197, B1 0.261). True rings alone: CTRL 0.197 → 0.119 (−0.078), B1 0.261 → 0.188
  (−0.074), i.e. both models gain about equally, but B1 stays above CTRL. True acyclic geometry alone: CTRL 0.197 →
  0.182 (−0.014), B1 0.261 → 0.158 (−0.103), and only this puts B1 below CTRL (0.158 < 0.182). Revised INFERENCE: ring
  accuracy helps both models; for B1 to overtake CTRL the acyclic bonds/angles must also be near-reference. Exact quote
  added: "| Matched RDKit seed, λ = 0 (ORACLE-selected) | 0.197 | 0.261 | 955 |" and "| True acyclic geometry only
  (A5acyc, ORACLE) | 0.182 | 0.158 | 955 |" (BRIEF lines 30, 38).

### E-GEOM-014
- Claim (original, superseded where the revision below differs): The cluster TD environment is python 3.9, torch 1.13.1+cu117, torch-geometric 2.0.4, e3nn 0.5.1, rdkit
  2022.9.5, built with micromamba from conda-forge.
- Source type: code
- Locator: code: `slurm/setup_env.sh:56`, `:76-79`, `:99-100`; `round2/IMPLEMENTATION.md:130`.
- Exact quote:
  > TORCH_SPEC="torch==1.13.1+cu117"; TORCH_INDEX="https://download.pytorch.org/whl/cu117"
  > e3nn==0.5.1 opt-einsum-fx==0.1.4 "sympy<1.13" \
- Shows vs infers: SHOWS.
- Supports card(s): C-GEOM-03, C-GEOM-05
- Confidence: high
- Revised after D-202: the environment is a pip venv on python 3.9; `slurm/setup_env.sh:43-58` uses a system/module
  python 3.9 and only falls back to a micromamba conda-forge python; packages are pip-installed (`:62-103`). Which branch
  ran on gnode118 is not recorded. Versions unchanged. Also relevant to C-GEOM-03 (D-206): the TD venv pins
  `numpy==1.23.5` (`:68`, `:89`), and the script has an alternative `cu118` profile with `torch==2.0.1+cu118` and
  torch-geometric 2.3.1 (`:80-84`), a CUDA-11.8 torch 2.x route already used by our tooling (not tested for ET-Flow).

### E-GEOM-049
- Claim: The round-2 design estimate for adding even a non-ring (acyclic) joint angle factor to TD was 300+ changed
  lines (parity-invariant angle head, Jacobian update, joint training), and RINGER-style ring angles would also need a
  ring-closure step.
- Source type: our-data (round-2 planning document; an estimate, not a measurement)
- Locator: our-data: `round2/research_A.md:257-261` (R2A-6 "Why deferred").
- Exact quote:
  > It needs a parity-invariant angle head, a Jacobian update for angles, and joint training: an estimated 300+ lines
- Shows vs infers: SHOWS the estimate (by research agent A, round 2); its accuracy is unverified.
- Supports card(s): C-GEOM-04
- Confidence: medium

## Papers

### E-GEOM-015
- Claim: GEOM conformers (QM9 included) come from CREST runs whose metadynamics geometries are optimised with GFN2-xTB.
- Source type: paper
- Locator: paper: `papers/related/2022_axelrod_geom.pdf`, PDF p. 3 (printed 3/18), section "Methods",
  subsection "CREST".
- Exact quote:
  > Geometries from the MTD runs are then optimized with GFN2-xTB.
- Shows vs infers: SHOWS.
- Supports card(s): C-GEOM-01, C-GEOM-05
- Confidence: high

### E-GEOM-016
- Claim: For QM9, each input geometry was re-optimised with GFN2-xTB before CREST, and CREST ran with default
  arguments except the charge; a QM9 CREST job cost 0.5 core-hours on average.
- Source type: paper
- Locator: paper: `papers/related/2022_axelrod_geom.pdf`, PDF p. 5 (printed 5/18), section "Conformer generation",
  subsections "Initial structure generation" and "CREST simulation".
- Exact quote:
  > to seed CREST with a structure optimized at the GFN2-xTB level of theory, we re-optimized each QM9 geometry with xTB before using it in CREST.
  > took an average of 0.04 wall hours on 13 cores (0.5 core hours)
- Shows vs infers: SHOWS. INFERENCE: "default arguments" (same page: "Default values were used for all CREST
  arguments, except for the charge") means no implicit-solvent flag, i.e. gas-phase GFN2-xTB minima; consistent with
  p. 2 ("vacuum conformer-rotamer ensembles") but that sentence is about the MoleculeNet sets, so the QM9 solvent
  setting is inferred, not quoted.
- Supports card(s): C-GEOM-01, C-GEOM-05
- Confidence: high (quote); medium (gas-phase inference)

### E-GEOM-017
- Claim: CREST keeps conformers within a 6.0 kcal/mol window, which the GEOM authors argue captures most accessible
  conformers.
- Source type: paper
- Locator: paper: `papers/related/2022_axelrod_geom.pdf`, PDF p. 4 (printed 4/18), section "Methods",
  subsection "DFT" (opening paragraph comparing CREST/xTB with DFT).
- Exact quote:
  > Further, the CREST safety window of 6.0 kcal/mol ensures that the vast majority of accessible conformers should be present in the CRE.
- Shows vs infers: SHOWS. Used only to set an energy window for the optional pucker-selection arm (INFERENCE).
- Supports card(s): C-GEOM-01
- Confidence: high

### E-GEOM-018
- Claim: Re-optimising GEOM reference conformers with GFN2-xTB changes them by almost nothing (relaxation energy ≈ 0).
  Reused from `round2/research_A.md` E14 (verified for GEOM-Drugs in `round2/verify_A.md`); quote re-checked.
- Source type: paper
- Locator: paper: `papers/related/2025_nikitin_geom_drugs_revisited.pdf`, PDF p. 10 (printed 10), section on energy
  metrics.
- Exact quote:
  > For conformers optimized with GFN2-xTB (as in GEOM-Drugs), the mean relaxation energy difference Erelax when re-optimized with GFN2-xTB is close to zero, as expected.
- Shows vs infers: SHOWS for GEOM-Drugs. INFERENCE for QM9: same CREST/GFN2-xTB pipeline (E-GEOM-015/016), so a
  GFN2-xTB optimisation started in the right basin should land on the reference geometry.
- Supports card(s): C-GEOM-01, C-GEOM-05
- Confidence: high (Drugs); medium (QM9 transfer)

### E-GEOM-019
- Claim: In Nikitin et al. Table 2 (GEOM-Drugs, 5 × 1000 molecules), GFN2-xTB re-optimisation of GEOM references moves
  bond lengths 0.00 × 10⁻² Å, angles 0.001°, torsions 0.01°, while MMFF-optimised vs GFN2-xTB-optimised structures
  differ by 1.12 × 10⁻² Å (bonds), 1.22° (angles) and 4.89° (torsions).
- Source type: paper
- Locator: paper: `papers/related/2025_nikitin_geom_drugs_revisited.pdf`, PDF p. 13 (printed 13), Table 2, rows
  "GEOM-Drugs" and "MMFF → GFN2-xTB" (row label arrow glyph lost in extraction). Text from `pdftotext -raw` (the
  layout-mode fulltext scrambles this table).
- Exact quote (raw extraction, ± glyphs garbled):
  > GEOM-Drugs 0.00±0.001 0.001±0.001 0.01±0.01 0.000±0.0001 0.001±0.001 16.4±0.2
  > MMFF GFN2-xTB 1.12±0.01 1.22±0.004 4.89±0.10 9.84±0.06 11.4±0.2 0.00±0.05
- Shows vs infers: SHOWS (DRUGS). INFERENCE: MMFF and GFN2-xTB minima differ at the level of our MMFF ring/angle L
  error (E-GEOM-001), so part of MMFF's residual L error is a level-of-theory offset that GFN2-xTB would remove.
- Supports card(s): C-GEOM-01, C-GEOM-05
- Confidence: high (numbers); medium (inference)

### E-GEOM-046
- Claim: GFN2-xTB geometry optimisation occasionally fails by fragmenting molecules (0.18% of GEOM-Drugs removed for
  this), so any xTB-relaxed L needs a connectivity/valence check.
- Source type: paper
- Locator: paper: `papers/related/2025_nikitin_geom_drugs_revisited.pdf`, PDF p. 8 (printed 8).
- Exact quote:
  > We removed molecules from GEOM-Drugs that were fragmented into multiple disconnected components due to failed GFN2-xTB geometry optimization. This led to the exclusion of 0.18% of the dataset
- Shows vs infers: SHOWS (Drugs). For QM9 see E-GEOM-047. Rate for a plain xTB optimisation (no metadynamics) of our
  ETKDG seeds is UNVERIFIED and must be counted in the CPU gate.
- Supports card(s): C-GEOM-01
- Confidence: high

### E-GEOM-047
- Claim: In GEOM, the original graph could be re-identified from CREST conformers for only 88.4% of QM9 molecules; all
  failed QM9 graphs had undergone some reaction, which the authors attribute to highly strained, unstable molecules.
- Source type: paper
- Locator: paper: `papers/related/2022_axelrod_geom.pdf`, PDF p. 7 (printed 7/18), "Technical Validation".
- Exact quote:
  > The graph re-attribution procedure succeeded for 88.4% of the QM9 molecules
  > All of the failed QM9 graphs underwent some sort of reaction, which can be explained by the presence of highly strained and unstable molecules.
- Shows vs infers: SHOWS (CREST, i.e. metadynamics + GFN2-xTB optimisation). INFERENCE: strained QM9 rings (3-/4-rings,
  cages) can rearrange under GFN2-xTB; an xTB-relaxed L source must reject seeds whose graph or stereo changes (the
  TD evaluation already filters GT conformers whose graph changed, `clean_confs`, `tools/make_l_seed_pickles.py:68-70`).
- Supports card(s): C-GEOM-01, C-GEOM-02
- Confidence: high (quote); medium (inference)

### E-GEOM-020
- Claim: PuckerFlow is trained and benchmarked only on 5–8-membered rings, extracted without substituents, and on
  monocyclic systems (non-aromatic, per its preprocessing).
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 6 (printed 6), Sec. 2.3 "Data and
  benchmarking methodology".
- Exact quote:
  > we focus on small- and medium-sized rings (five- to eight-membered rings)
  > we train all models only on the cyclic systems without substituents.
  > While in this work, we target monocyclic ring systems
  > We exclude rings containing radical electrons as well as aromatic rings
- Shows vs infers: SHOWS. Transfer to QM9: E-GEOM-011 (4-rings and fused systems are outside this domain).
- Supports card(s): C-GEOM-04
- Confidence: high

### E-GEOM-021
- Claim: PuckerFlow's authors fed PuckerFlow rings as fixed local substructures into the pretrained TD model (two-stage,
  no TD retraining); joint generation is future work. Reused from `round2/research_A.md` E13 (VERIFIED in
  `round2/verify_A.md`); quote re-checked.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 21 (printed 21), SI A.5.4 "Generating
  Exocyclic Substituents".
- Exact quote:
  > we sample the ring structures using PuckerFlow, and use these as local substructures for a pretrained model of torsional diffusion using publicly available model weights
- Shows vs infers: SHOWS (qualitative demo; the paper reports no TD AMR/COV numbers for this pipeline).
- Supports card(s): C-GEOM-04
- Confidence: high

### E-GEOM-022
- Claim: On PuckerFlow's ring benchmark (puckering-displacement RMSD, 0.1 Å threshold, unrelaxed), recall
  coverage / AMR are PuckerFlow 75.8% / 0.09 Å, ETKDG with small-ring torsions 60.1% / 0.13 Å, plain ETKDG 53.3% /
  0.14 Å, MCF 60.0% / 0.12 Å.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 23 (printed 23), SI Table 6 (with
  standard errors; main-text Table 1 on p. 7 has the same means but a scrambled layout in the fulltext).
- Exact quote (Table 6 rows, ± glyphs garbled):
  > RDKit ETKDG (Small Cycles)  0.17 ± 0.011  51.4 ± 2.0  0.13 ± 0.008  60.1 ± 1.8
  > RDKit ETKDG                 0.18 ± 0.011  43.6 ± 3.6  0.14 ± 0.010  53.3 ± 2.7
- Shows vs infers: SHOWS. Column order: precision AMR, precision coverage, recall AMR, recall coverage (then the same
  four after MMFF relaxation). PuckerFlow row (same table): 0.13, 67.5, 0.09, 75.8.
- Supports card(s): C-GEOM-04, C-GEOM-06
- Confidence: high

### E-GEOM-023
- Claim: On the same benchmark scored by all-atom RMSD, the gaps shrink: recall AMR unrelaxed PuckerFlow 0.15 Å,
  ETKDG (small cycles) 0.17 Å, ETKDG 0.17 Å; after MMFF relaxation 0.14 / 0.14 / 0.15 Å.
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 24 (printed 24), SI Table 7.
- Exact quote (Table 7 rows, ± garbled):
  > RDKit ETKDG (Small Cycles)  0.22 ± 0.011  37.7 ± 2.9  0.17 ± 0.009  44.5 ± 2.8  0.17 ± 0.009  55.8 ± 0.8  0.14 ± 0.009  56.3 ± 2.6
  > RDKit ETKDG                 0.22 ± 0.012  35.6 ± 3.5  0.17 ± 0.010  42.9 ± 3.3  0.18 ± 0.011  54.2 ± 1.8  0.15 ± 0.010  54.6 ± 3.0
- Shows vs infers: SHOWS. PuckerFlow row: 0.18, 47.4, 0.15, 51.0, 0.16, 61.5, 0.14, 61.0. INFERENCE: on an all-atom
  RMSD metric like ours, a learned pucker model buys ~0.02 Å over ETKDG on isolated substituent-free rings, and
  ≤ 0.01 Å once both are MMFF-relaxed.
- Supports card(s): C-GEOM-04, C-GEOM-06
- Confidence: high (numbers); medium (inference)

### E-GEOM-024
- Claim: TD states that ring conformations are delegated to the local-structure sampler and that this works less well
  for puckered and fused rings. Partly reused from `round2/research_A.md` E22 (VERIFIED).
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 23, App. F.4 "Rings".
- Exact quote:
  > Although this is true for a large number of relatively small rings (especially aromatic ones) present in many drug-like molecules, it is less true for puckered rings, fused rings, and larger cycles.
- Shows vs infers: SHOWS.
- Supports card(s): C-GEOM-02, C-GEOM-04
- Confidence: high

### E-GEOM-025
- Claim: On GEOM-QM9 (TD Table 7), OMEGA has recall AMR mean / median 0.177 / 0.126 Å vs TD 0.178 / 0.147 and RDKit
  0.235 / 0.199, and TD attributes OMEGA's edge to better local structures. Partly reused from `round2/research_A.md`
  E3 (VERIFIED).
- Source type: paper
- Locator: paper: `papers/core/2022_jing_torsional_diffusion.pdf`, PDF p. 26, App. H "Small molecules" and Table 7.
- Exact quote:
  > OMEGA, which, evidently, has a better local structures for these small molecules.
  > OMEGA                85.5 100.0 0.177 0.126     82.9 100.0 0.224 0.186
- Shows vs infers: SHOWS. Mechanism of OMEGA's L: E-GEOM-026.
- Supports card(s): C-GEOM-02
- Confidence: high

### E-GEOM-026
- Claim: The PuckerFlow benchmark's comparison of RDKit variants shows that the small-ring torsion terms (srETKDG,
  "Small Cycles") raise unrelaxed pucker recall coverage from 53.3% to 60.1% but leave all-atom recall AMR unchanged
  (0.17 → 0.17 Å). (Summary of E-GEOM-022/023 for the srETKDG card.)
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 23–24, SI Tables 6–7; the variant is
  defined on PDF p. 6 (Sec. 2.3).
- Exact quote (p. 6):
  > an extended ETKDG variant that introduces a torsional-angle potential for small rings (Small Cycles)
- Shows vs infers: SHOWS (on substituent-free 5–8 rings, not QM9).
- Supports card(s): C-GEOM-06
- Confidence: high

### E-GEOM-027
- Claim: MACE-OFF is trained to ωB97M-D3(BJ)/def2-TZVPPD DFT energies and forces (SPICE), not to GFN2-xTB.
- Source type: paper
- Locator: paper: `papers/related/2023_kovacs_mace_off.pdf`, PDF p. 3, Sec. "B. Training data".
- Exact quote (ω glyph lost in extraction):
  > The MACE-OFF models are trained to reproduce the energies and forces computed at the B97M-D3(BJ)/def2-TZVPPD level of quantum mechanics
- Shows vs infers: SHOWS. INFERENCE: relaxing with MACE-OFF targets DFT minima, a different level from the GFN2-xTB
  references of GEOM (E-GEOM-015/016); AIMNet2 likewise targets a DFT level (stated in its docs/paper, not checked
  here: UNVERIFIED).
- Supports card(s): C-GEOM-05
- Confidence: high (quote); medium (inference)

### E-GEOM-028
- Claim: On GEOM-QM9 (TD protocol, δ = 0.5 Å), ET-Flow reports recall AMR mean / median 0.073 / 0.047 Å and COV-R
  96.47%, vs MCF 0.103 and TD 0.178. Reused from `round2/research_A.md` E18 (VERIFIED; verifier advised citing main
  Table 2).
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 7 (printed 7), Table 2.
- Exact quote (Table 2 row for ET-Flow; method labels are separated from the numbers in the fulltext):
  > 96.47 100.00 0.073 0.047  94.05 100.00 0.098 0.039
- Shows vs infers: SHOWS. Row assignment: the seven label rows (CGCF, GeoDiff, GeoMol, Torsional Diff., MCF, ET-Flow,
  ET-Flow-SO(3)) map in order to the seven number rows; TD's row reads 0.178, matching TD's paper value.
- Supports card(s): C-GEOM-03
- Confidence: high

### E-GEOM-048
- Claim (original, superseded where the revision below differs): ET-Flow's chirality-corrected variant is the SO(3) architecture; the default O(3) model is not
  chirality-corrected (the released QM9 checkpoint is `qm9-o3`, E-GEOM-037).
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 7 (printed 7), Table 2 caption.
- Exact quote:
  > ET-Flow - SO(3) is ET-Flow using the SO(3) architecture for chirality correction.
- Shows vs infers: SHOWS. INFERENCE: `qm9-o3` samples need a stereo check (reflect or drop mirror images) before use
  as L seeds.
- Supports card(s): none after revision
- Confidence: high
- Revised after D-204 (original claim WITHDRAWN): ET-Flow's base O(3) model *is* chirality-corrected, post hoc (oriented
  volume vs RDKit tags, whole-conformer flip on mismatch), and that corrected model is the reported 0.073; SO(3) is an
  architectural alternative [E-GEOM-053]. The released `qm9-o3` config defaults to `parity_switch = "post_hoc"` and
  `sample()` applies it, so `predict()` returns corrected conformers [E-GEOM-054]. A stereo check on the seeds stays (a
  whole-molecule flip cannot fix a sample with only some stereocentres inverted).

### E-GEOM-029
- Claim: FM-refiner starts sampling from upstream conformers instead of noise. Reused from `round2/research_A.md` E17
  (VERIFIED); quote re-checked.
- Source type: paper
- Locator: paper: `papers/related/2025_xu_fm_refiner.pdf`, PDF p. 3 (printed 3).
- Exact quote:
  > instead of starting from pure noise, we initialize sampling from upstream-generated conformers x^1, thereby skipping the inherently hard-to-learn high-noise phase.
- Shows vs infers: SHOWS.
- Supports card(s): C-GEOM-03
- Confidence: high

### E-GEOM-030
- Claim: FM-refiner can only correct upstream conformers whose error is within the noise range covered by its base
  distribution ("reachable range").
- Source type: paper
- Locator: paper: `papers/related/2025_xu_fm_refiner.pdf`, PDF p. 5 (printed 5), "Design implications".
- Exact quote (the noise-bound symbol is lost in extraction, shown as a double space):
  > The conformers by the upstream model with error no larger than  lie within the refiner's reachable range
- Shows vs infers: SHOWS. INFERENCE: a low-noise Cartesian refiner started from ETKDG cannot flip a wrong pucker basin
  (6-ring chair/boat, 4-ring pucker sign) unless its noise range covers that move; this caps what an R2A-5-style
  refiner can fix in our pucker tail (E-GEOM-002/003).
- Supports card(s): C-GEOM-03
- Confidence: high (quote); medium (inference)

### E-GEOM-031
- Claim: GO-Flow's ablation finds modelling internal coordinates the most critical component (the variant without it
  degrades to a Cartesian model). Reused from `round2/research_A.md` E20 (VERIFIED quote; the verifier noted it argues
  against a pure-Cartesian L model).
- Source type: paper
- Locator: paper: `papers/related/2026_liu_goflow.pdf`, PDF p. 7, Sec. 4.4.1.
- Exact quote:
  > modeling internal coordinates (bond lengths, angles, and tor- sions) via entropic optimal transport is the most critical factor for generating high-fidelity molecular structures.
- Shows vs infers: SHOWS (DRUGS, GD-P protocol).
- Supports card(s): C-GEOM-03, C-GEOM-04
- Confidence: high

### E-GEOM-032
- Claim: Corso's thesis proposes modelling ring flexibility with ring-puckering coordinates on hyperspheres. Reused
  from `round2/research_A.md` E23 (VERIFIED).
- Source type: paper
- Locator: paper: `papers/related/2023_corso_intrinsic_diffusion.pdf`, PDF p. 63.
- Exact quote:
  > employing the ring puckering coordinates [18] to model the flexibility of ring conformations as points on hyperspheres.
- Shows vs infers: SHOWS (a proposal; no experiment).
- Supports card(s): C-GEOM-04
- Confidence: high

### E-GEOM-033
- Claim: RINGER (internal-coordinate diffusion for macrocycles) needs a ring-closure step because sequential NeRF
  reconstruction accumulates error. Reused from `round2/research_A.md` E12 (VERIFIED; about macrocycles).
- Source type: paper
- Locator: paper: `papers/related/2023_grambow_ringer.pdf`, PDF p. 5.
- Exact quote:
  > Adopting a sequential reconstruction method such as NeRF accumulates small errors that result in inadequate ring closure for macrocycles.
- Shows vs infers: SHOWS for macrocycles; extrapolation to 3–7-rings is INFERENCE.
- Supports card(s): C-GEOM-04
- Confidence: medium


### E-GEOM-042
- Claim: Across ~140k molecules (CREST/GFN2 conformers), ring conformations fall into relatively few canonical
  clusters, and the number of clusters grows slowly with ring size.
- Source type: paper
- Locator: paper: `papers/related/2021_chan_ring_puckering.pdf`, PDF p. 1 (printed 743), Abstract.
- Exact quote:
  > we show that the ring conformations can be classified into relatively few conformational clusters, based on their canonical forms. The number of such canonical clusters increases slowly with ring size.
- Shows vs infers: SHOWS. INFERENCE: a per-ring-system pucker library from the QM9 training set stays small and can
  be enumerated (C-GEOM-02).
- Supports card(s): C-GEOM-02
- Confidence: high

### E-GEOM-043
- Claim: A knowledge-based pucker sampler (no minimisation) reproduced CREST/GFN2 lowest-energy conformations of 20
  simple ring systems at 0.09 Å average RMSD, the residual coming from bond lengths and angles, which the authors say
  local geometry optimisation would fix.
- Source type: paper
- Locator: paper: `papers/related/2021_chan_ring_puckering.pdf`, PDF p. 11 (printed 753), section "Ring
  Reconstruction".
- Exact quote (Å glyph garbled in the fulltext):
  > our proposed method gives low average TFD values (0.05) and an average RMSD value of 0.09 Å on the selected cyclic molecules.
  > Note that the large RMSD values are ascribed to the deviation in bond lengths and bond angles.
- Shows vs infers: SHOWS (20 molecules, no acyclic rotors; small test). INFERENCE: template pucker + GFN2-xTB
  relaxation (C-GEOM-02 + C-GEOM-01) addresses both residuals.
- Supports card(s): C-GEOM-02
- Confidence: medium (small benchmark)

### E-GEOM-044
- Claim: Chan et al. describe OMEGA-style knowledge-based samplers as relying on discrete prespecified ring templates
  and heuristic rules.
- Source type: paper
- Locator: paper: `papers/related/2021_chan_ring_puckering.pdf`, PDF p. 2 (printed 744), Introduction.
- Exact quote:
  > Unlike knowledge-based sampling methods, e.g., OMEGA,29 which rely on a set of discrete prespecified ring templates and heuristic rules for sampling
- Shows vs infers: SHOWS (third-party description of OMEGA; complements E-GEOM-038).
- Supports card(s): C-GEOM-02
- Confidence: high

### E-GEOM-051
- Claim: Equivariant Blurring Diffusion generates atomic detail from a coarse fragment-level structure while letting
  that coarse structure be adjusted, i.e. it corrects a cheminformatics prior rather than freezing it. Reused from
  `round2/research_A.md` E11 (VERIFIED in `round2/verify_A.md`); quote re-checked.
- Source type: paper
- Locator: paper: `papers/related/2024_park_equivariant_blurring_diffusion.pdf`, PDF p. 1, Abstract.
- Exact quote:
  > generation of fine atomic details from the coarse-grained approximated structure while allowing the latter to be adjusted simultaneously.
- Shows vs infers: SHOWS (GD-P protocol, DRUGS/QM9 at its own thresholds).
- Supports card(s): C-GEOM-03
- Confidence: high

## Blogs / technical posts / software docs

### E-GEOM-034
- Claim: xtb is installable from conda-forge (`conda install xtb`) or as a precompiled tarball from the GitHub release
  page; it is a standalone binary (no Python/torch dependency).
- Source type: blog (software documentation)
- Locator: blog: https://xtb-docs.readthedocs.io/en/latest/setup.html, Grimme group (xtb documentation), undated live
  docs, section "Setup and Installation" → "Installing with Conda" / "Precompiled Binaries from GitHub", accessed
  2026-10-08, snapshot `papers/blogs/xtb_docs_setup.txt`.
- Exact quote:
  > Installing xtb from the conda-forge channel can be achieved by adding conda-forge to your channels with:
  > A precompiled version of the program can be obtained from the latest release page on GitHub.
- Shows vs infers: SHOWS. INFERENCE: independent of the cluster's torch 1.13.1 / python 3.9 stack (separate
  micromamba env or tarball; TD's wrapper only needs a binary path, E-GEOM-010).
- Supports card(s): C-GEOM-01, C-GEOM-05
- Confidence: high

### E-GEOM-035
- Claim: conda-forge ships linux-64 builds of xtb (latest 6.7.1) and of CREST (latest 3.0.2).
- Source type: blog (package index metadata)
- Locator: blog: https://api.anaconda.org/package/conda-forge/xtb and .../crest, anaconda.org API, accessed 2026-10-08,
  fields `latest_version` and linux-64 file versions, snapshot `papers/blogs/condaforge_xtb_metadata.txt`.
- Exact quote:
  > linux-64 versions: ['6.4.1', '6.5.0', '6.5.1', '6.6.0', '6.6.1', '6.7.1']
  > summary: Conformer-Rotamer Ensemble Sampling Tool based on the xtb Semiempirical Extended Tight-Binding Program Package
- Shows vs infers: SHOWS (package availability; not an install test on gnode118).
- Supports card(s): C-GEOM-01, C-GEOM-05
- Confidence: high

### E-GEOM-036
- Claim: Python packaging of ML potentials vs our stack (python 3.9, torch 1.13.1, e3nn 0.5.1): `aimnet` 0.2.0
  (AIMNet2) requires python ≥ 3.11 and torch ≥ 2.8; `torchani` 2.9.0 requires python ≥ 3.10 and torch 2.0–2.13 (the
  older 2.2.4, Nov 2023, has torch unpinned); `mace-torch` 0.3.16 requires python ≥ 3.9, torch ≥ 1.12 and pins
  `e3nn==0.4.4`; `tblite` 0.7.0 ships wheels for cp310+ only.
- Source type: blog (package index metadata)
- Locator: blog: https://pypi.org/pypi/<package>/json (PyPI JSON API) for aimnet, torchani, torchani/2.2.4, mace-torch,
  tblite, accessed 2026-10-08, fields `requires_python`, `requires_dist`, wheel file names, snapshot
  `papers/blogs/pypi_metadata_mlpot_xtb_etflow.txt`.
- Exact quote:
  > ### package: aimnet version: 0.2.0
  > requires_python: >=3.11
  > ... 'requests>=2.32.3', 'torch>=2.8' ...
  > requires_dist: ['torch>=1.12', 'e3nn==0.4.4', ...
- Shows vs infers: SHOWS (declared metadata). INFERENCE: AIMNet2 and current TorchANI cannot go into the TD env;
  MACE-OFF can only go into a separate venv (e3nn pin conflicts with TD's 0.5.1, E-GEOM-014); none was install-tested.
- Supports card(s): C-GEOM-03, C-GEOM-05
- Confidence: high (metadata); medium (install inference)

### E-GEOM-037
- Claim: ET-Flow is pip-installable (`etflow`), ships a pretrained `qm9-o3` checkpoint with automatic download, uses
  the torsional-diffusion splits, and its dev environment pins `pytorch-cuda==12.1` and `numpy==1.26.4`.
- Source type: blog (code repository README / packaging)
- Locator: blog: https://github.com/shenoynikhil/ETFlow (README.md, pyproject.toml, env.yml on branch main), M. Hassan,
  N. Shenoy et al., undated (repo HEAD), sections "Install ET-Flow", "Generating Conformations for Custom Smiles",
  "Preprocessing Data", accessed 2026-10-08, snapshot `papers/blogs/etflow_repo_readme.txt`.
- Exact quote:
  > We currently support the following configurations and checkpoint:
  > - `qm9-o3`
  > For the splits and test mols, download the files from the [torsional diffusion]
  > - pytorch::pytorch-cuda==12.1
- Shows vs infers: SHOWS. INFERENCE: ET-Flow needs its own conda env with torch ≥ 2.1 (pytorch-cuda 12.1 builds
  exist only for torch 2.x); on the RTX 3090 this needs an NVIDIA driver supporting CUDA 12.1 (gnode118 driver version
  UNVERIFIED). Same split as TD ⇒ no test leakage into its training set (to be confirmed by checking the
  `qm9-o3` training config). The README also warns that preprocessing changes may not reproduce paper numbers.
- Supports card(s): C-GEOM-03
- Confidence: high (repo facts); medium (env inference)

### E-GEOM-038
- Claim: OMEGA builds molecules from a fragment library and takes ring conformations from that library, detaching
  substituents and enumerating every combination of ring conformations (plus invertible nitrogen puckers).
- Source type: blog (software documentation)
- Locator: blog: https://docs.eyesopen.com/applications/omega/theory/omega_classic_theory.html, OpenEye Scientific
  (OMEGA docs), undated, section "OMEGA Theory" (model building), accessed 2026-10-08, snapshot
  `papers/blogs/openeye_omega_classic_theory.txt`.
- Exact quote:
  > Ring conformations are taken from the same fragment library used to build the initial model.
  > attempts to generate every possible combination of ring conformations possible for a given structure.
- Shows vs infers: SHOWS (OMEGA's method). INFERENCE: this template mechanism is the most plausible reason for TD's
  remark that OMEGA has better QM9 local structures (E-GEOM-025); the docs do not test that.
- Supports card(s): C-GEOM-02
- Confidence: high (method); medium (link to E-GEOM-025)

### E-GEOM-039
- Claim: RDKit's small-ring torsion terms are off by default (`useSmallRingTorsions` default False); preconfigured
  parameter objects ETKDG, ETKDGv2, ETKDGv3 and srETKDGv3 exist; `coordMap` constrains chosen atoms to given 3D
  coordinates during embedding.
- Source type: blog (software documentation)
- Locator: blog: https://www.rdkit.org/docs/RDKit_Book.html, RDKit contributors (G. Landrum et al.), live docs (current
  release, not 2022.09), section "Conformer Generation" → "Parameters Controlling Conformer Generation", accessed
  2026-10-08, snapshot `papers/blogs/rdkit_book.txt` (lines ~8164–8285).
- Exact quote:
  > useSmallRingTorsions : (default False) use the sr part of srETDKGv3
  > coordMap : (default empty) can be used to provide 3D coordinates which will be used to constrain the positions of some of the atoms in the molecule.
- Shows vs infers: SHOWS (current docs). The 2022.9.5 keyword defaults are UNVERIFIED (see E-GEOM-041).
- Supports card(s): C-GEOM-02, C-GEOM-06
- Confidence: high

### E-GEOM-050
- Claim: ETKDG's torsion preferences (which the small-ring terms extend) come from the Cambridge Structural Database
  (crystal structures), not from gas-phase quantum-chemical minima.
- Source type: blog (software documentation)
- Locator: blog: https://www.rdkit.org/docs/RDKit_Book.html, RDKit contributors, live docs, section "Conformer
  Generation", accessed 2026-10-08, snapshot `papers/blogs/rdkit_book.txt` (lines ~8164–8173).
- Exact quote:
  > which modifies step 5 above to also use torsion angle preferences from the Cambridge Structural Database (CSD) to correct the conformers after distance geometry has been used to generate them.
- Shows vs infers: SHOWS. INFERENCE: ring-torsion terms bias puckers toward crystal-phase statistics, a different
  target from GEOM's gas-phase GFN2-xTB minima.
- Supports card(s): C-GEOM-06
- Confidence: high

### E-GEOM-040
- Claim: Since RDKit 2018.09, the Python `EmbedMolecule()` / `EmbedMultipleConfs()` use ETKDG by default.
- Source type: blog (software documentation)
- Locator: blog: https://www.rdkit.org/docs/BackwardsIncompatibleChanges.html, RDKit contributors, live docs, section
  "Release 2018.09" → "The conformation generation code now uses ETKDG by default when called from Python", accessed
  2026-10-08, snapshot `papers/blogs/rdkit_backwards_incompatible_changes.txt`.
- Exact quote:
  > The Python functions EmbedMolecule() and EmbedMultipleConfs() now use the ETKDG algorithm by default instead of standard distance geometry.
- Shows vs infers: SHOWS. Does not say which ETKDG version later releases default to.
- Supports card(s): C-GEOM-06
- Confidence: high

### E-GEOM-045
- Claim (original, superseded where the revision below differs): A practitioner write-up on piperazine (RDKit 2020.03.2, 500 conformers) finds ETKDG (v1 parameters) gives
  many non-chair rings (naive |torsion| classifier: 254 chair / 228 twisted / 18 boat), MMFF optimisation leaves many
  twisted (366 / 131 / 3), and srETKDGv3 still gives 28% twisted rings once the torsion-sign pattern is checked
  (358 chair / 142 twisted).
- Source type: blog
- Locator: blog: https://github.com/sunhwan/blog/blob/master/posts/2021-02-24-RDKit-ETKDG-Piperazine/index.ipynb, GitHub
  user sunhwan, 2021-02-24, sections "Ring conformation using ETKDG v1", "Optimize Geometry using MMFF", "ETKDG
  version 3", accessed 2026-10-08, snapshot `papers/blogs/sunhwan_rdkit_etkdg_piperazine.txt` (lines 126, 187, 271, 274).
- Exact quote:
  > OK, so turns out, 358 (72%) of the conformers were actually in chair conformation and 142 (28%) of the conformers were in twisted conformers.
  > Counter({'chair': 366, 'twisted': 131, 'boat': 3})
- Shows vs infers: SHOWS (one molecule; a blog, so evidence for practice, not proof). Caveat: the post's prose
  ("Still about 50% ... twisted") does not match its own MMFF counter output; the counter lines are cited. INFERENCE:
  consistent with our 6-ring numbers (ETKDG 66% of seeds > 10°, MMFF 45%; E-GEOM-003): force-field relaxation and the
  small-ring torsion terms only partly remove wrong puckers.
- Supports card(s): C-GEOM-06
- Confidence: medium
- Revised after D-203: the post's first run ("ETKDG v1") passes a bare `EmbedParameters()` (cell 6), which has
  `useExpTorsionAnglePrefs=False` and `useBasicKnowledge=False` (local check, RDKit 2026.03.6: ET False, K False), i.e.
  plain distance geometry; the MMFF counts are MMFF on plain-DG output, scored with the naive classifier the author later
  shows over-counts chairs. **Only the srETKDGv3 result counts**: with the torsion-sign check, 358 chair / 142 twisted
  (28% twisted) for piperazine. The ETKDG/MMFF part of the claim is WITHDRAWN; C-GEOM-01 now cites E-GEOM-002 for "MMFF
  only partly repairs puckers". Supports only C-GEOM-06 after revision.

### E-GEOM-041
- Claim (original, superseded where the revision below differs): In RDKit 2026.03.6 (local), the keyword defaults of `EmbedMultipleConfs` are `useSmallRingTorsions=False`,
  `useMacrocycleTorsions=True`, `ETversion=2`, `useMacrocycle14config=True` (i.e. ETKDGv3 without the small-ring
  terms). The cluster's 2022.9.5 defaults were not checked (UNVERIFIED); the current docs also list
  `useSmallRingTorsions` as default False (E-GEOM-039).
- Source type: code (library docstring, local check)
- Locator: code: `python -c "from rdkit.Chem import rdDistGeom as d; print(d.EmbedMultipleConfs.__doc__)"`, RDKit
  2026.03.6, Windows, 2026-10-08.
- Exact quote:
  > (bool)useSmallRingTorsions=False [, (bool)useMacrocycleTorsions=True [, (int)ETversion=2 [, (bool)useMacrocycle14config=True
- Shows vs infers: SHOWS (local version only).
- Supports card(s): C-GEOM-06
- Confidence: high (local); low (cluster)
- Revised after D-208 (resolved with a new source, E-GEOM-052): in RDKit 2022.9.5 (cluster) the python keyword
  defaults are `useExpTorsionAnglePrefs=true`, `useBasicKnowledge=true`, `useSmallRingTorsions=false`,
  `useMacrocycleTorsions=false`, `ETversion=1`, i.e. **ETKDG v1**. So all cluster seeds (training-time matching and
  test-time) are ETKDG v1, not v3.

## Entries added in P2 (disputes D-201..D-209)

### E-GEOM-052
- Claim: In RDKit 2022.09.5 (the cluster's version) the python `EmbedMultipleConfs` keyword defaults are
  `useExpTorsionAnglePrefs=true`, `useBasicKnowledge=true`, `useSmallRingTorsions=false`, `useMacrocycleTorsions=false`,
  `ETversion=1`, i.e. ETKDG v1 without small-ring or macrocycle terms.
- Source type: code (library source at the release tag)
- Locator: code: RDKit `Release_2022_09_5`, `Code/GraphMol/DistGeomHelpers/Wrap/rdDistGeom.cpp:346-351`
  (https://github.com/rdkit/rdkit/blob/Release_2022_09_5/Code/GraphMol/DistGeomHelpers/Wrap/rdDistGeom.cpp, accessed
  2026-10-08), snapshot `papers/blogs/rdkit_2022_09_5_rdDistGeom_wrapper.txt`. Found first by V3 (D-208), re-fetched
  and checked by GEOM.
- Exact quote:
  > python::arg("useSmallRingTorsions") = false,
  > python::arg("useMacrocycleTorsions") = false,
  > python::arg("ETversion") = 1),
- Shows vs infers: SHOWS. With E-GEOM-008 (TD calls `EmbedMultipleConfs` without a parameter object): every cluster
  seed is ETKDG v1.
- Supports card(s): C-GEOM-01, C-GEOM-06
- Confidence: high

### E-GEOM-053
- Claim: ET-Flow's base O(3) model uses a post hoc chirality correction (compare the oriented volume with the RDKit
  chiral tags and flip the conformation on mismatch); the SO(3) architecture is the alternative, and the base method
  (reported 0.073 on QM9) is the post-hoc-corrected one.
- Source type: paper
- Locator: paper: `papers/related/2024_hassan_etflow.pdf`, PDF p. 5 (printed 5), §3.4 (chirality correction).
- Exact quote:
  > In the case of a mismatch, we simply flip the conformation against the z-axis.
  > Our base method (ET-Flow) corresponds to using the post hoc correction whereas the SO(3) variant is referred by ET-Flow-SO(3).
- Shows vs infers: SHOWS.
- Supports card(s): C-GEOM-03
- Confidence: high

### E-GEOM-054
- Claim: In the released `etflow` code, the `qm9-o3` configuration uses the default `ModelArgsSchema`, whose
  `parity_switch` defaults to `"post_hoc"`, and `BaseFlow.sample()` (called by `predict()`) applies the parity switch
  after integration; the `qm9-o3.ckpt` sits in Zenodo record 14226681 together with `QM9.zip` split files, while the
  scaffold splits are a separate record (16551316, `scaffold_data.tar.gz`).
- Source type: blog (code repository + data-record metadata)
- Locator: blog: https://github.com/shenoynikhil/ETFlow, `etflow/commons/configs.py:118`, `:189-196`;
  `etflow/models/model.py:459-462`, `:467`, `:525` (branch main); https://zenodo.org/api/records/14226681 and 16551316;
  M. Hassan, N. Shenoy et al.; accessed 2026-10-08; snapshot `papers/blogs/etflow_repo_code_chirality.txt`.
- Exact quote:
  > parity_switch: Literal["post_hoc"] = "post_hoc"
  > if self.parity_switch == "post_hoc":
  > files: ['drugs-o3.ckpt', 'qm9-o3.ckpt', 'DRUGS.zip', 'QM9.zip', 'drugs-so3.ckpt', 'XL.zip']
- Shows vs infers: SHOWS. INFERENCE: `qm9-o3` was most likely trained on the random (TD) split shipped next to it, not
  the scaffold split; still to be confirmed by comparing `QM9.zip`'s test SMILES with TD's `test_smiles.csv` (step 0 of
  C-GEOM-03).
- Supports card(s): C-GEOM-03
- Confidence: high (code facts); medium (split inference)

### E-GEOM-055
- Claim: PuckerFlow's authors state the approach extends to fused and spiro rings by computing Cremer–Pople coordinates
  for each component ring separately (not demonstrated in the paper).
- Source type: paper
- Locator: paper: `papers/related/2026_schaufelberger_puckerflow.pdf`, PDF p. 6 (printed 6), Sec. 2.3 (same paragraph as
  E-GEOM-020). Pointed out by V3 (D-209).
- Exact quote:
  > our approach can also be applied to fused and spiro rings, where the Cremer-Pople coordinates can be determined for each component ring separately
- Shows vs infers: SHOWS (a stated extension, not an experiment).
- Supports card(s): C-GEOM-04
- Confidence: high
