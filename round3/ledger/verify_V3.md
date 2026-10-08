# Verification V3: round-3 ledger evidence_GEOM.md (51 entries) and cards C-GEOM-01..06

Verifier: V3. Written 2026-10-08. Role: `orchestration/roles/verifier.md` (adversarial).
Status: P2 pass complete (all 51 entries and 6 cards checked). Disputes D-201..D-209 opened in `disputes.md`.

Method: paper quotes checked against my own `pdftotext -raw` / `-layout` extraction of each PDF (pages split on form
feed; page 1 title/authors checked; surrounding paragraph read). Code `path:line` opened. Our-data values re-read from
the result files and every derived number recomputed with my own scripts (scratchpad `v3_lerr.py`, `v3_sym.py`,
`v3_share.py`, `v3_rings.py`, `v3_rigid.py`). Software claims checked against the live package indexes (anaconda.org
and PyPI JSON APIs fetched 2026-10-08), the ET-Flow repo raw files (README.md, env.yml, pyproject.toml on `main`),
the RDKit `Release_2022_09_5` source, and WebFetch of the xtb, OpenEye and RDKit doc pages.

## Verdict table

| Entry | Type | Verdict | Dispute | One-line reason |
|---|---|---|---|---|
| E-GEOM-001 | our-data | VERIFIED | — | All 38 numbers and the row counts reproduce exactly (grouping below) |
| E-GEOM-002 | our-data | VERIFIED | — | All shares and n reproduce exactly. Caveat: 113 all-3-ring molecules add structural zeros (see R2) |
| E-GEOM-003 | our-data | VERIFIED | — | Reproduces; trivial: λ=0.75 5-rings have 1 seed > 10° (0.07%), 2 molecules with smallest ring 9 omitted |
| E-GEOM-004 | our-data | VERIFIED | D-205 (advisory) | Numbers and quote exact; code path supports "model-free" but bin 0 is not exactly model-free and populations differ (258–318) |
| E-GEOM-005 | our-data | VERIFIED | — | BRIEF line 35 exact; 3-seed means 0.0976 / 0.1156 / 0.1525; caveats stated by scout |
| E-GEOM-006 | our-data | VERIFIED | — | All 15 bin means reproduce; like-for-like base for A5ring is λ=0, conclusion unchanged |
| E-GEOM-007 | our-data | VERIFIED | — | Reproduces (B1 MMFF 0.2323, λ0.5 0.1812, λ0.75 0.1146) |
| E-GEOM-008 | code | VERIFIED | D-208 (advisory) | Lines exact. Resolved the open part: cluster RDKit 2022.9.5 keyword default = ETKDG**v1** |
| E-GEOM-009 | code | VERIFIED | — | `make_l_seed_pickles.py:110` exact; no symmetry relabelling either (checked, not a driver of the tail) |
| E-GEOM-010 | code | VERIFIED | — | `utils/xtb.py:5-7, 43-64` as claimed; caveat: brittle runtime parse can silently drop the result |
| E-GEOM-011 | our-data | VERIFIED | — | 420/465, 78/22%, 519/635/222/140, 449, 223 molecules, 48.1% all reproduce |
| E-GEOM-012 | our-data | VERIFIED | — | 534 / 50.8% / 45.2% / 345 reproduce with isomeric fragment SMILES (stereo-free key: 421 / 66.3% / 58.6%) |
| E-GEOM-013 | our-data | DOES NOT SUPPORT | D-201 | Numbers right; "helps CTRL but not B1" is wrong: B1 also improves (0.236→0.188; like-for-like λ=0 base 0.261→0.188) |
| E-GEOM-014 | code | DOES NOT SUPPORT (minor) | D-202 | Versions verified; "built with micromamba from conda-forge" is not what the script does (pip venv) |
| E-GEOM-015 | paper | VERIFIED | — | GEOM p. 3 exact; context: CREST, GFN2-xTB, tight final optimisation |
| E-GEOM-016 | paper | VERIFIED | — | GEOM p. 5 both quotes exact; gas-phase correctly labelled inference |
| E-GEOM-017 | paper | VERIFIED | — | GEOM p. 4 exact |
| E-GEOM-018 | paper | VERIFIED | — | Nikitin p. 10 exact (Δ glyph lost) |
| E-GEOM-019 | paper | VERIFIED | — | Nikitin p. 13 Table 2 rows exact (metrics are mean abs. deviations, all atoms) |
| E-GEOM-020 | paper | VERIFIED | — | PuckerFlow p. 6, four quotes exact |
| E-GEOM-021 | paper | VERIFIED | — | PuckerFlow p. 21 SI A.5.4 exact |
| E-GEOM-022 | paper | VERIFIED | — | p. 23 Table 6 rows exact; p. 7 Table 1 same means |
| E-GEOM-023 | paper | VERIFIED | — | p. 24 Table 7 rows exact |
| E-GEOM-024 | paper | VERIFIED | — | TD p. 23, App. F.4 "Limitations", paragraph "Rings", exact |
| E-GEOM-025 | paper | VERIFIED | — | TD p. 26 Table 7 row and sentence exact |
| E-GEOM-026 | paper | VERIFIED | — | p. 6 quote exact; numbers as E-022/023 |
| E-GEOM-027 | paper | VERIFIED | — | MACE-OFF (arXiv 2312.15211v5) p. 3 exact (ω lost) |
| E-GEOM-028 | paper | VERIFIED | — | ET-Flow p. 7 Table 2 row exact |
| E-GEOM-029 | paper | VERIFIED | — | FM-refiner p. 3 exact |
| E-GEOM-030 | paper | VERIFIED | — | FM-refiner p. 5 "Design implications" exact (σ lost) |
| E-GEOM-031 | paper | VERIFIED | — | GO-Flow p. 7 §4.4.1 exact |
| E-GEOM-032 | paper | VERIFIED | — | Corso thesis p. 63 exact |
| E-GEOM-033 | paper | VERIFIED | — | RINGER p. 5 exact |
| E-GEOM-034 | blog/docs | VERIFIED | — | Live page and snapshot match; conda-forge xtb 6.7.1 has no python dependency (checked build `depends`) |
| E-GEOM-035 | blog/API | VERIFIED | — | latest 6.7.1 / 3.0.2 confirmed; the snapshot's linux-64 xtb list is incomplete (API: 6.2.3–6.7.1) |
| E-GEOM-036 | blog/API | VERIFIED | — | All metadata exact on PyPI JSON today |
| E-GEOM-037 | blog/repo | VERIFIED | — | Quotes exact on `main`; but the CUDA-12 pin is dev env only (see D-206 for the card) |
| E-GEOM-038 | blog/docs | VERIFIED | — | Quotes exact; section heading is "Torsion Driving (TD) in OMEGA", not "OMEGA Theory" |
| E-GEOM-039 | blog/docs | VERIFIED | — | Live RDKit 2026.03.6 docs match |
| E-GEOM-040 | blog/docs | VERIFIED | — | Live page matches; no later default change listed |
| E-GEOM-041 | code (local) | VERIFIED | D-208 (advisory) | Local docstring exact; cluster 2022.9.5 = ETversion 1, macrocycle torsions off (RDKit source) |
| E-GEOM-042 | paper | VERIFIED | — | Chan 2021 p. 1 (743) exact; paper identity confirmed |
| E-GEOM-043 | paper | VERIFIED | — | Chan p. 11 (753) exact; 20 molecules, no acyclic rotors |
| E-GEOM-044 | paper | VERIFIED | — | Chan p. 2 (744) exact (hyphenation only) |
| E-GEOM-045 | blog | DOES NOT SUPPORT | D-203 | The post's "ETKDG v1" run uses a bare `EmbedParameters()` = plain DG (ET and K off); only the srETKDGv3 result is ETKDG |
| E-GEOM-046 | paper | VERIFIED | — | Nikitin p. 8 exact |
| E-GEOM-047 | paper | VERIFIED | — | GEOM p. 7 both quotes exact |
| E-GEOM-048 | paper | DOES NOT SUPPORT | D-204 | Same paper p. 5 §3.4: the base O(3) ET-Flow **is** chirality-corrected (post hoc); SO(3) is the alternative |
| E-GEOM-049 | our-data | VERIFIED | — | `round2/research_A.md:257` / `:260` exact |
| E-GEOM-050 | blog/docs | VERIFIED | — | Live page matches; PuckerFlow p. 6 says the same about ET/CSD |
| E-GEOM-051 | paper | VERIFIED | — | EBD p. 1 exact; p. 3–4, 8 confirm the coarse prior is RDKit DG and is corrected |

Counts: VERIFIED 47, DOES NOT SUPPORT 4 (E-013, E-014 minor, E-045, E-048), QUOTE MISMATCH 0, WRONG LOCATOR 0,
UNVERIFIABLE 0.

## Recomputation from l_error.csv (brief point 1)

File `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`, 191,897 rows, 12 conditions. My grouping, stated exactly:
- **Unit = one csv row** (one seed paired with one GT reference). Means / medians / shares are taken over rows within
  `cond` (NaN skipped). For `L_etkdg2L` / `_mmff` there is one row per GT conformer (only the Hungarian-assigned seed
  out of 2L, `make_l_seed_pickles.py:106-114`), so "per seed" means "per assigned seed": 12,655 rows, 936 molecules.
  λ and A5 rows: one per GT slot (12,707 rows / 906 mols for λ 0.25–0.75; 12,795 / 955 for λ 0, λ 1, A5). Noise: 2L
  rows per molecule (25,762 / 996).
- **Ring molecule** = RDKit SSSR of the csv SMILES has ≥ 1 ring (885 of 996). Two SMILES without a ring
  (`N=C(N)C(=O)C(=O)C=O`, `N=CNC(=N)C(=O)C=O`, 60 rows) do have a ring-dihedral value, i.e. their GT graph has a ring;
  they are excluded, as the scout did.
- **Smallest ring** = min length over `AtomRings()`.

R1. Means (rows): ETKDG ring bond 0.0318 Å / ring angle 2.379° / ring dihedral 10.652°; MMFF 0.0199 / 1.417 / 8.066;
λ0.50 0.0189 / 1.339 / 3.156; **λ0.75 0.0101 / 0.670 / 1.564**; noise 0.02 0.0267 / 1.385 / 0.963; noise 0.04
0.0534 / 2.768 / 1.929. Acyclic bond / angle: ETKDG 0.0292 / 2.836; MMFF 0.0170 / 2.032; λ0 0.0322 / 4.122; λ0.50
0.0172 / 2.011; **λ0.75 0.0090 / 1.004**; noise 0.02 0.0269 / 1.672; noise 0.04 0.0538 / 3.370; A5ring 0.0322 / 4.113;
A5acyc 0 / 0.479. Medians of ring dihedral: ETKDG 0.436, MMFF 0.431, λ0.75 0.122. So "λ = 0.75 ≈ ring bonds
0.010 Å, ring angles 0.67°, acyclic angles 1.0°" is exact under this grouping. Sensitivity: averaging per molecule
first gives λ0.75 0.0113 Å / 0.82° / acyclic 1.02°, ETKDG 0.0367 / 3.13°.

R2. Share of ring-molecule rows with ring-dihedral RMSD > 10°: ETKDG 28.57% (2074 / 7259), MMFF 20.20% (1466 / 7259),
λ0 30.36%, λ0.25 22.29%, λ0.50 11.24%, λ0.75 0.014% (1 / 7285), noise 0.02 0%, noise 0.04 0.34%. > 20°: ETKDG 17.92%,
MMFF 12.58%, λ0.50 0.04%, λ0.75 0. All as claimed. **Caveat on interpretation**: a 3-ring's "dihedral" (a,b,c,a) is
identically 0 (`lgeom._dih_deg` with d = a), so the 113 molecules whose rings are all 3-membered contribute only zeros.
Without them: ETKDG 49.6%, MMFF 35.1%, λ0.50 19.4%, λ0.75 0.02%. Per-molecule mean of the share: ETKDG 34.7%,
MMFF 18.5%; share of molecules with any bad seed: 41.7% / 27.6%. The ordering ETKDG > MMFF > λ0.5 >> λ0.75 holds
under every grouping.

R3. Is part of the ETKDG tail a labelling artefact? ETKDG/MMFF rows compare identity-labelled atoms (no automorphism
relabelling), while λ / A5 rows use `lgeom.align_pair`'s relabelled reference. Test: molecules with a ring-permuting
heavy-atom automorphism (180 of 885) have a *lower* ETKDG > 10° share (22.3%) than the rest (30.5%), and λ0 (relabelled
reference, same sampler) shows the same tail (30.4%). Paired over 827 molecules, ETKDG 0.348 vs λ0 0.374 (non-symmetric)
and 0.340 vs 0.375 (symmetric). So the tail is real, not a labelling artefact.

R4. Rigid bin (`breakdown.log`, "by n_rot_heavy", bin 0; parsed from all 84 run × condition logs): rigid share of
evaluated molecules 28.5–31.9% (258–318 molecules), so "≈ 30%" holds. TD code path for these molecules:
`generate_confs.py:150-153` skips perturbation and the model when TD's own `edge_mask` is empty; when only H rotors exist
(methyl, OH, =CH2) every rotated side's single heavy atom lies on the rotation axis (`utils/torsion.py:57-74`), so heavy
atoms do not move; evaluation is heavy-atom `GetBestRMS(RemoveHs(...))` (`evaluate_confs.py:115`). The reasoning is
right. **But** the logged bin-0 MAT-R still differs between checkpoints given identical seed pickles and identical n:
λ = 1.00 (true L, should be 0.000) gives CTRL 0.004 / 0.004 / 0.007 and B1 0.001; ETKDG gives CTRL 0.152 / 0.151 /
0.154 and B1 0.155 / 0.156 / 0.155. So `breakdown.py`'s bin 0 (computed from the corrected SMILES) holds a few
molecules TD does rotate; model dependence ≤ ~0.006 Å. A CPU gate that scores the seeds directly is exactly model-free;
the logged bin-0 values are not a perfect reference, and they sit on different populations per L source (D-205).

## Notes per entry (non-trivial ones)

- **E-GEOM-004.** Values: CTRL / B1 3-seed means ETKDG 0.152 / 0.155, MMFF 0.128 / 0.127, λ0.50 0.107 / 0.109, λ0.75
  0.077 / 0.074, A5ring 0.043 / 0.048, noise 0.02 0.032 / 0.029, true L 0.005 / 0.001 (B1 s0); largest gap λ0
  0.184 / 0.190. All exact. Populations: ETKDG/MMFF 284, λ0.25–0.75 258, λ0/λ1/A5 289, noise 318. See R4 and D-205.
- **E-GEOM-006.** Like-for-like base for A5ring/A5acyc is λ = 0 (same matched pairs, 955 molecules): CTRL λ0 by bin
  0.184 / 0.181 / 0.209 / 0.209 / 0.254 → A5ring 0.043 / 0.104 / 0.162 / 0.193 / 0.249. Gain still concentrated in
  0–1 rotor bins, so the claim stands.
- **E-GEOM-008 / 041.** RDKit `Release_2022_09_5` `Code/GraphMol/DistGeomHelpers/Wrap/rdDistGeom.cpp:337-351`
  (`EmbedMultipleConfs` python args): `useExpTorsionAnglePrefs = true`, `useBasicKnowledge = true`,
  `useSmallRingTorsions = false`, `useMacrocycleTorsions = false`, `ETversion = 1`. So the cluster's seeds (training
  matching and test time) are **ETKDG v1**, not v3. Consequence for C-GEOM-06 in D-208.
- **E-GEOM-010.** Wrapper as claimed (gas phase, `--opt <level>` where level is the convergence level, GFN2 by xtb
  default). For P3: `xtb_optimize` parses a runtime line at a fixed offset (`out.split(b'\n')[-12]`, `:51-52`) *before*
  reading `xtbopt.xyz`; if the 6.7.1 output trailer differs, the exception path returns None and the conformer is left
  unoptimised. All calls share one per-PID directory (xtb leaves `xtbrestart` etc. there). TD also ships
  `optimize_confs.py:39-43` that already drives `xtb_optimize` with `--xtb_path`/`--level`.
- **E-GEOM-012.** The 534 / 50.8% / 45.2% numbers reproduce only with `MolFragmentToSmiles(..., isomericSmiles=True)`
  (the default), i.e. the key carries ring-atom stereo tags. A stereo-free key gives 421 components, 66.3% LOO coverage
  overall and 58.6% among the 345 bad-seed molecules. Worth stating the key; it makes the card's coverage case stronger.
- **E-GEOM-013.** See D-201. Like-for-like (λ0 base, 955 molecules, 3-seed means): CTRL 0.1966 → A5ring 0.1186
  (−0.078), A5acyc 0.1824 (−0.014); B1 0.2614 → A5ring 0.1877 (−0.074), A5acyc 0.1583 (−0.103). Against the ETKDG
  base the scout used, B1 still improves 0.2356 → 0.1877. True rings help both models by about the same amount; what
  differs is that true acyclic geometry helps B1 much more, and that B1 starts worse.
- **E-GEOM-014.** `setup_env.sh:43-58` uses a system/module python3.9 and only falls back to a micromamba
  conda-forge python; all packages are pip-installed into a venv (`:62-103`). Versions confirmed by
  `round2/IMPLEMENTATION.md:130`.
- **E-GEOM-015/016/017/047 (brief point 2).** The GEOM PDF is the claimed paper (Axelrod & Gómez-Bombarelli, "GEOM:
  Energy-annotated molecular conformations...", 18 pp.). p. 3: CREST MTD geometries "are then optimized with
  GFN2-xTB", and all accumulated geometries get a final tight optimisation; p. 5: QM9 SMILES used as given, QM9 DFT
  geometries re-optimised with xTB as CREST seeds, CREST run with default arguments except the charge. DFT appears only
  as CENSO re-optimisation of 534 BACE species and single points on BACE (p. 4–6). So the GEOM-QM9 conformers are
  GFN2-xTB geometries. VERIFIED. Gas phase is a fair inference (default CREST has no implicit solvent; the water runs are
  explicitly BACE-only).
- **E-GEOM-018/019.** Paper identity: Nikitin, Dunn, Koes, Isayev, "GEOM-Drugs Revisited". Table 2 metrics are mean
  absolute differences over all bonds/angles/torsions (p. 12), not RMSDs; the inference compares them with our
  heavy-atom RMSDs. Fine as labelled INFERENCE.
- **E-GEOM-020.** Same paragraph (p. 6) also says the approach "can also be applied to fused and spiro rings, where the
  Cremer-Pople coordinates can be determined for each component ring separately". C-GEOM-04 should cite this when it
  argues fused systems are out of reach.
- **E-GEOM-034/035 (brief point 3, xtb).** anaconda.org API today: xtb latest 6.7.1, linux-64 builds 6.2.3–6.7.1 (all
  label `main`); 6.7.1 linux-64 `depends`: glibc, OpenMP, BLAS/LAPACK, gfortran runtime, dftd4, mctc-lib, multicharge,
  tblite (Fortran library), cpcm-x. No python, no torch. crest latest 3.0.2, linux-64 2.11.1 / 2.11.2 / 2.12 / 3.0.2.
  The xtb docs page (WebFetch) matches the snapshot and says the GitHub tarball binary is statically linked. Independence
  from the TD python/torch stack: VERIFIED.
- **E-GEOM-036 (brief point 3, ML potentials).** PyPI JSON today: `aimnet` 0.2.0 (Isayev lab, "AIMNet Machine Learned
  Interatomic Potential") requires_python ≥ 3.11, `torch>=2.8`; earlier releases 0.1.1 (≥ 3.11, torch ≥ 2.8), 0.1.0
  (≥ 3.11, torch ≥ 2.4), 0.0.1 (≥ 3.10, torch ≥ 2.5), so no `aimnet` release fits python 3.9 / torch 1.13. `torchani`
  2.9.0 ≥ 3.10, torch 2.0–2.13; 2.2.4 (2023-11-14) `torch` unpinned. `mace-torch` 0.3.16 ≥ 3.9, `torch>=1.12`,
  `e3nn==0.4.4` (0.3.6 pins the same). `tblite` 0.7.0 wheels cp310–cp313 (plus sdist). All VERIFIED.
- **E-GEOM-037 (brief point 3, ET-Flow).** README / env.yml / pyproject on `main` match the snapshot and quotes. But
  "ET-Flow needs CUDA 12" is not established: `pytorch-cuda==12.1` is only in the dev `env.yml`; the PyPI `etflow`
  0.1.2 pins `numpy==1.26.4` and leaves `torch` unpinned, and the README advises installing pytorch first. Latest
  `lightning` 2.6.6 needs torch ≥ 2.1 and python ≥ 3.10; `lightning` 2.2.0 accepted torch ≥ 1.13. A cu118 torch 2.x
  (the repo's own `TORCH_PROFILE=cu118`, torch 2.0.1) is a plausible route; untested. The numpy pin alone already rules
  out the TD venv (numpy 1.23.5). README also points to separate "Scaffold Splits and Checkpoints" (zenodo 16551316):
  which split `qm9-o3` was trained on must be checked (scout already flags it).
- **E-GEOM-045.** Notebook cell 6: `params = Chem.rdDistGeom.EmbedParameters()` then `EmbedMultipleConfs(..., params=params)`.
  A bare `EmbedParameters` has `useExpTorsionAnglePrefs = False`, `useBasicKnowledge = False` (RDKit Book parameter
  list; local check: ET False, K False). That run is plain distance geometry, not ETKDG v1, although the post calls it
  that. The MMFF counts (366 / 131 / 3) are MMFF on plain-DG output, scored with the naive |torsion| classifier the
  author later shows over-counts chairs. Only "srETKDGv3: 358 chair / 142 twisted (sign check)" is an ETKDG result.
- **E-GEOM-048.** ET-Flow p. 5 §3.4: "Our base method (ET-Flow) corresponds to using the post hoc correction whereas
  the SO(3) variant is referred by ET-Flow-SO(3)." The 0.073 in Table 2 is the O(3) model *with* post hoc chirality
  correction (oriented-volume check against RDKit tags, flip on mismatch). Whether `BaseFlow.predict` applies it must be
  checked in the package; a stereo check on the seeds is still sensible.
- **Brief point 4 (new papers).** `2022_axelrod_geom.pdf` = GEOM (Axelrod, Gómez-Bombarelli; 18 pp.);
  `2023_kovacs_mace_off.pdf` = "MACE-OFF: Short Range Transferable Machine Learning Force Fields for Organic Molecules"
  (Kovács, Moore, ..., Csányi), arXiv:2312.15211v5, 22 Aug 2025; `2021_chan_ring_puckering.pdf` = "Understanding Ring
  Puckering in Small Molecules and Cyclic Peptides" (Chan, Hutchison, Morris), JCIM 2021, 61, 743–755. All three are the
  claimed papers; INDEX rows 55, 56, 58 describe them correctly.

## Card rulings

Rule used: a card is SUPPORTED if every factual sentence rests on a VERIFIED or still-open entry and its core mechanism
has VERIFIED evidence; WEAKENED if the core holds but sentences that carry the card's case or its go/no-go reasoning
are unsupported; UNSUPPORTED if the core mechanism has no VERIFIED entry.

### C-GEOM-01 (GFN2-xTB-relaxed seeds): SUPPORTED, with three fixes
Core VERIFIED: GEOM-QM9 references are GFN2-xTB (E-015/016), GFN2 re-optimisation of references is ≈ null and
MMFF↔GFN2 offsets are of the size of our MMFF residual (E-018/019), TD has an xTB wrapper (E-010), xtb is a standalone
conda-forge binary (E-034/035), the λ-table spec (E-001/002/005/007). Fixes:
1. §3 "Ring-only truth helps CTRL (0.178 → 0.119) but not B1 (0.188) ... so it is the arm that can also move B1" rests
   on E-013 (D-201). B1 gains as much from true rings as CTRL; the right statement is that only acyclic truth lets B1
   overtake CTRL (A5acyc B1 0.158 < CTRL 0.182).
2. §2 "force-field relaxation only partly repairs ETKDG ring puckers (piperazine)" rests on E-045 (D-203); our own E-002
   (MMFF 20.2% vs ETKDG 28.6%) supports the same sentence.
3. §5 gate "rigid AMR-R ≤ 0.110" compares xTB seeds (ETKDG population, 284 rigid molecules) with λ0.5 (258 molecules)
   and uses logged bin-0 values that carry ≤ 0.006 Å model dependence (D-205). Compute all gate metrics on one common
   molecule set.
Unsourced but labelled: runtime guess, "~1M training conformers".

### C-GEOM-02 (training-set ring templates): SUPPORTED, with fixes
Core VERIFIED: OMEGA's template mechanism (E-038/044), OMEGA's QM9 edge (E-025, link labelled inference), clustered
puckers and a 0.09 Å knowledge-based sampler (E-042/043), coordMap (E-039), pucker tail and coverage proxies
(E-002/003/011/012). Fixes (D-207): "60–92% for 4- to 7-membered rings" is wrong for 5-rings (48%), range is 48–92%;
"True ring geometry alone takes CTRL from 0.178 to 0.119" mixes the ETKDG base (936) with the λ0-based oracle (955), the
like-for-like gain is 0.197 → 0.119; "~10⁶ training conformers" has no entry; the rigid-AMR gate threshold 0.107 is a
different population (D-205).

### C-GEOM-03 (ET-Flow L as a diagnostic): WEAKENED
The core (ET-Flow 0.073 on GEOM-QM9, installable, `qm9-o3` checkpoint, TD split for preprocessing) is VERIFIED
(E-028/037), as are the refiner citations (E-029/030/031/051). But the two reasons the scout gives for MAYBE are not
supported (D-204, D-206): (a) "the paper's chirality-corrected variant is the SO(3) one" is contradicted by ET-Flow
§3.4: the O(3) base model with post hoc correction is the reported 0.073; (b) "ET-Flow needs torch ≥ 2.1 / CUDA 12.1
... RTX 3090 needs a recent driver" is not established: only the dev env pins CUDA 12.1, the pip package does not pin
torch. A separate env is still needed (numpy 1.26.4 pin), so the conclusion "cannot share the TD venv" stands. Add: the
README's scaffold-split checkpoints mean the `qm9-o3` training split must be confirmed. Net: the card's risk and cost
are overstated; its evidence is otherwise sound.

### C-GEOM-04 (learned pucker component, not this round): SUPPORTED, with two corrections (D-209)
Core VERIFIED: PuckerFlow domain and numbers (E-020..023), two-stage TD demo (E-021), QM9 domain mismatch counts
(E-011, E-003), Corso/GO-Flow/RINGER (E-031..033), round-2 cost estimate (E-049). Corrections: "Substituents matter in
QM9 (every test molecule has them)": 53 of 885 test ring molecules have no exocyclic heavy atom; and PuckerFlow p. 6
itself says the method extends to fused/spiro rings per component ring, so "fused systems need ... closure
constraints" is INFERENCE (E-033 is about macrocycles). Also inherits the "0.178 → 0.119" base issue (D-201/D-207).

### C-GEOM-05 (ML-potential relaxation, not for QM9): SUPPORTED
Every factual sentence rests on VERIFIED entries (E-027, E-015/016, E-018/019, E-036, E-014 versions, E-001/002/007,
E-034/035). Checked beyond the entry: no `aimnet` release on PyPI fits python 3.9 / torch 1.13 (all need ≥ 3.10 and
torch ≥ 2.4). The DFT-vs-GFN2 target argument is correctly labelled INFERENCE / UNVERIFIED.

### C-GEOM-06 (srETKDGv3 seeds): SUPPORTED, with one design correction (D-208)
Core VERIFIED (E-039/040/041/008, E-022/023/026, E-050, E-002/003/004); the srETKDGv3 piperazine sentence uses the part of
E-045 that is an ETKDG result (28% twisted), so it stands. Design correction: the cluster default is ETKDG v1, so
`srETKDGv3` vs `L_etkdg2L` changes ETversion 1 → 2, macrocycle torsions and the 1-4 config **and** small-ring torsions at
once. Add a plain `ETKDGv3` arm (same random seed) to isolate the small-ring terms; "expected ≤ 0.005 Å" is then a
statement about two changes, not one.
