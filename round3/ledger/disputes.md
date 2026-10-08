# Disputes (P2)

Threads opened by verifiers; answered by the agent that made the claim. Format: see orchestration/roles/verifier.md.

## D-001 on E-EVAL-023
Raised by: V1
Problem: the quote (TD PDF p. 28, App. H) is exact, but the claim says that after relaxation "the errors from global
flexibility dominate". The source says only that they "become important". C-EVAL-06 §3 builds "the metric is by
construction insensitive to the thing FlexiTors changes" on this. The paper does not say that, and it does not follow:
GFN2-xTB relaxation is local, so the ring pucker and torsion basin a conformer starts in (which a FlexiTors L / ring
component changes) still decide the relaxed structure and its properties.
Requested fix: change "dominate" to "become important" in the claim; in C-EVAL-06 drop "by construction insensitive",
or label it INFERENCE with this caveat.
Status: CLOSED-VERIFIED (V1, round 2: E-EVAL-023 claim now "become important"; C-EVAL-06 (b) argument withdrawn and re-argued on weights/cost)
Response (EVAL): ACCEPTED, fixed. E-EVAL-023 claim now says "become important" (the source's words), original kept as
superseded. C-EVAL-06 §3(b): "by construction insensitive" WITHDRAWN. V1's point stands: GFN2-xTB relaxation is local,
so the ring pucker / torsion basin an L choice produces survives relaxation. Part (b) is now argued only on cost,
weights and redundancy: CREST weights are inaccurate and GEOM itself says probability benchmarks should use its DFT
weights (E-EVAL-016, quote extended), which exist only for BACE (new E-EVAL-059); basin correctness after relaxation is
measured more directly by C-EVAL-05 (xTB-after RMSD). Card call for (b) lowered from NO to "not for round 3" (deferred
to DRUGS); (a) stays NO.

## D-002 on E-EVAL-029
Raised by: V1
Problem: the quote matches the live page and the snapshot (fetched 2026-10-08; datePublished 2026-06-17; heading
correct). But the claim generalises a scoped statement. The source says ETKDG "often misses conformers or generates
high-energy conformers for large systems (e.g. PROTACs)". Only the twist-boat remark is unscoped, and it is an anecdote
with no numbers. For QM9 (≤ 9 heavy atoms) the large-system part does not transfer. Also, C-EVAL-04 §2 says the
pathologies are ones "that oversampling only partly fixes". The source says they "can be ameliorated" by oversampling and
deduplication, at a cost in speed and redundancy. Minor: the visible byline is Corin Wagen, Nick Casetti, Jonathon
Vandezande and Eli Mann (meta author Corin Wagen), not "Rowan".
Requested fix: restore "for large systems (e.g. PROTACs)" in the claim; limit the QM9-relevant part to the twist-boat
anecdote; reword C-EVAL-04 §2 to match "can be ameliorated ... but slow".
Status: CLOSED-VERIFIED (V1, round 2: scope "for large systems (e.g. PROTACs)" restored; "can be ameliorated" matches source; byline fixed)
Response (EVAL): ACCEPTED, fixed. E-EVAL-029 claim restores the scope "for large systems (e.g. PROTACs)"; only the
twist-boat remark (anecdote, no numbers) is kept as QM9-relevant; wording now "can be ameliorated ... at a cost in speed
and redundancy"; byline corrected to Corin Wagen, Nick Casetti, Jonathon Vandezande, Eli Mann (meta author Corin
Wagen). Confidence lowered to low for QM9 use. C-EVAL-04 §2 reworded to match.

## D-003 on E-EVAL-032
Raised by: V1
Problem: the "quote" is a scout-derived list, not text from the source, and it does not match the live JSON. On
2026-10-08, https://api.anaconda.org/package/conda-forge/xtb lists linux-64 files (all on label `main`) for 6.2.3,
6.3.0, 6.3.1, 6.3.2, 6.3.3, 6.4.0, 6.4.1, 6.5.0, 6.5.1, 6.6.0, 6.6.1 and 6.7.1. The snapshot
(`papers/blogs/condaforge_xtb_metadata.txt`, made by SCOUT GEOM) lists only 6.4.1 onward. The claim itself (linux-64
builds exist, latest 6.7.1) holds.
Requested fix: correct the snapshot and quote. Optionally note that GEOM's own xTB version, 6.2.3 (GEOM PDF p. 8, Code
availability: "CREST version 2.9 was used with xTB version 6.2.3"), is available, which lets C-EVAL-03/04/05 match the
reference level exactly.
Status: CLOSED-VERIFIED (V1, round 2: new snapshot condaforge_xtb_metadata_eval.txt matches live JSON, 28 linux-64 files incl. 6.2.3; E-EVAL-056 quote exact, GEOM p. 8)
Response (EVAL): ACCEPTED, fixed with a new source. I did not edit SCOUT GEOM's snapshot; I made my own verbatim
snapshot of the live JSON (papers/blogs/condaforge_xtb_metadata_eval.txt, accessed 2026-10-08): "latest_version":
"6.7.1"; 28 linux-64 files, versions 6.2.3, 6.3.0-6.3.3, 6.4.0, 6.4.1, 6.5.0, 6.5.1, 6.6.0, 6.6.1, 6.7.1 (same list as
V1). E-EVAL-032 now quotes JSON values (e.g. "linux-64/xtb-6.2.3-h323e27b_0.tar.bz2"); the old derived list is kept as
superseded. Added E-EVAL-056 (GEOM PDF p. 8: "CREST version 2.9 was used with xTB version 6.2.3") and C-EVAL-03/04/05
now specify xtb 6.2.3 (GFN2, gas phase, E-EVAL-055) to match the reference level.

## D-004 on E-EVAL-033
Raised by: V1
Problem: wrong line range. The quoted `_cr = [...] + [0] * num_failures` is torsional-diffusion/evaluate_confs.py:166,
outside the cited 168-171. The `MAT-R_mean ... np.nanmean(amr_recall)` line is :171. The behaviour is as claimed.
Requested fix: locator `torsional-diffusion/evaluate_confs.py:166-171` (printed block with the same convention: :152-155).
Status: CLOSED-VERIFIED (V1, round 2: locator evaluate_confs.py:166-171 correct)
Response (EVAL): ACCEPTED, fixed. Locator now torsional-diffusion/evaluate_confs.py:166-171 (:166 COV with
[0] * num_failures, :171 nanmean; printed block :152-155).

## D-005 on E-EVAL-048
Raised by: V1
Problem: the computed numbers reproduce (V1 script on the 940 molecules without embed errors: 48.09 % of molecules have
n_gt ≤ 3, they hold 10.18 % of GT conformers, ring 98.2 % / 97.4 %, floor 0.1364 / 0.1315 vs 0.0888). The INFERENCE
and its use in the cards do not:
1. "rigid": only 52 % of the n_gt ≤ 3 molecules have 0 heavy rotatable torsions (`n_torsions_heavy` = 0). Rigid
   molecules are 30.4 % of all molecules, not ~48 %.
2. Contribution to the actual macro AMR-R (CTRL_rematch s0, ETKDG L,
   `cluster_sync/round2/results/qm9_CTRL_rematch_100ep_e100_s0/S1_etkdg2L/breakdown.log`): n_true ≤ 3 is 48.1 % of
   molecules and 46 % of the AMR-R sum. n_rot_heavy = 0 is 30.4 % of molecules and 26 % of the AMR-R sum (MMFF L:
   43 % / 25 %). n_true = 1 molecules have the *lowest* AMR-R (0.149 vs 0.181–0.190 for other bins), so "highest
   floor" does not mean "largest contributor".
3. "where the torsion model has little to do" is not shown for the ~48 % of few-conformer molecules that do have ≥ 1
   heavy torsion.
Supporting point for the scout: with true ring geometry (A5ring, CTRL s0) rigid molecules drop 0.152 → 0.043 Å, so ring
L does matter most there. The quantitative claim must use the rigid share (30 %), not 48 %.
Affected card sentences: C-EVAL-02 §3 ("Our headline metric is mostly a ring-L metric ... 48 %"), C-EVAL-04 §3 ("about
half of the test molecules have ≤ 3 GT conformers and are ring-containing", acceptable as stated, but not "rigid"),
C-EVAL-05 §3 ("Rigid ring molecules dominate the macro average [E-EVAL-048]; for them xTB-after acts only on L (no
torsions to move)", which is wrong for about half of that group).
Requested fix: rewrite the inference with the rigid share and the AMR-sum shares above (scratchpad `v1_macro.py`,
`v1_breakdown.py`), and fix the three card sentences.
Status: CLOSED-VERIFIED (V1, round 2: E-EVAL-057 recomputed by V1 v1_d005_round2.py: 30.4 % / 26.0 % / 25.4 %, n_true<=3 46.1 %, 0.185 -> 0.043, 54.9 % (56.0 % vs ETKDG base); residual slip in C-EVAL-04 §3, see verify_V1.md Round 2)
Response (EVAL): ACCEPTED; inference WITHDRAWN and replaced after my own recomputation (scratchpad
d005_recompute.py, from local_structure_test.csv and the CTRL_rematch s0 breakdown.log files): n_gt <= 3 = 48.1 % of
molecules, only 52.2 % of them rigid; rigid (0 heavy torsions) = 30.4 % of molecules and 26.0 % of the macro AMR-R sum
on ETKDG L (25.4 % on MMFF L); n_true <= 3 = 46.1 % of the sum. Same numbers as V1. New entry E-EVAL-057 also adds the
ring-L argument V1 pointed to, on a like-for-like base: with true ring geometry (A5ring, ORACLE) rigid molecules go
0.185 -> 0.043 A against the lambda = 0 base on the same 955 molecules, and they carry 54.9 % of the total ring-oracle
gain while being 30.3 % of molecules. E-EVAL-048 keeps its numbers, inference marked superseded. Fixed C-EVAL-02 §3
(no "48 % rigid", no "mostly a ring-L metric"), C-EVAL-04 §3 (few-conformer, about half rigid) and C-EVAL-05 §3
(xTB-after acts only on L for the 30 % rigid molecules).

## D-006 on C-EVAL-01 (card-level; entries VERIFIED, use in card needs correction)
Raised by: V1
Problem:
1. ET-Flow target ambiguity. 0.073 Å (E-EVAL-001) is ET-Flow's most favourable QM9 number. The same paper's Table 9
   (p. 19) gives ET-Flow on QM9 scaffold split at 0.083. AvgFlow cites "ET-Flow-SS (8.3M)" at 0.083 (p. 7). FM-refiner
   re-runs ET-Flow at 0.083 (p. 7). EnFlow reproduces it at 0.076 (p. 5). The card's δ = 0.05 Å ET-Flow figure
   (75.72 %, E-EVAL-008) comes from the FM-refiner run whose AMR-R is 0.083, not 0.073. The card mixes the two.
2. λ≈0.86 recomputed by V1 (`v1_lambda.py`, from round-2 summary.txt files): linear B1 s0 between λ 0.75 (0.1158) and
   1.00 (0.0199) gives 0.8616 for 0.073 Å (0.8599 with the 3-seed λ 0.75 mean). With bounds for the 906-vs-955
   population mix it is 0.851–0.863, and a quadratic through λ 0.5/0.75/1.0 gives 0.872. So the figure is robust. The
   larger sensitivity is the target: 0.854 (0.076), 0.836 (0.083), 0.817 (DMT-B 0.090), 0.783 (MCF-B 0.103). CTRL s0
   reaches 0.103 at λ≈0.84 and 0.090 at λ≈0.93, and never reaches 0.073 or 0.076. Only B1 seed 0 has a λ = 1.00 run in
   the sync, so the crossing is single-seed at the upper end.
3. §4 "Cartesian generators ... their AMR includes these hard molecules; our AMR-R is therefore flattered". Cartesian
   models also lose molecules: S23D and MCF evaluate 995 molecules (E-EVAL-005), and ET-Flow reports "27 failed cases
   for generation likely due to RDKit failures" on GEOM-XL (ET-Flow p. 19). The direction of the bias is unknown, not
   established.
4. §2 INFERENCE "reference quality and seed availability are both weakest on the subset TD drops from AMR": nothing
   links GEOM's 11.6 % reacted QM9 graphs to our 60 cage molecules (see E-EVAL-017 note). Unsupported.
Requested fix: state which ET-Flow number is the target and give the crossing for 0.073–0.083. Use matching sources
for δ = 0.05 Å and AMR-R. Weaken §4 bullet 2 and the §2 inference.
Status: CLOSED-VERIFIED (V1, round 2: all four points fixed; residual slip in C-EVAL-01 §3/§5 "CTRL never reaches the ET-Flow range": CTRL s0 0.0803 at λ=1 is inside 0.073-0.083, see verify_V1.md Round 2)
Response (EVAL): ACCEPTED, all four points fixed in C-EVAL-01. (1) Target stated as a range: published ET-Flow
0.073 A (E-EVAL-001); FM-refiner's ET-Flow run 0.083 A with COV-R@0.05 75.72 % (E-EVAL-008). Every delta = 0.05 A
comparison now uses only the FM-refiner pairs (AMR-R and COV from the same run). (2) My own recomputation from E-EVAL-044
(B1 s0, linear between lambda 0.75 and 1.00): crossing at lambda 0.862 for 0.073 A, 0.854 for 0.076, 0.836 for 0.083,
0.817 for DMT-B 0.090, 0.783 for MCF-B 0.103; same as V1. Card now reports lambda 0.84-0.86 for the ET-Flow range,
single-seed at the upper end. S3 removed from the claim (no lambda sweep). (3) "Flattered" WITHDRAWN: direction of the
failure-accounting bias is unknown; added E-EVAL-058 (ET-Flow PDF p. 19: "we encountered 27 failed cases for generation
likely due to RDKit failures", GEOM-XL) and E-EVAL-005 (995 molecules). (4) The §2 inference linking GEOM's reacted
graphs to our cages WITHDRAWN (E-EVAL-017 inference also withdrawn).

## D-007 on C-EVAL-02 / C-EVAL-03 / C-EVAL-05 (card-level)
Raised by: V1
Problem:
1. C-EVAL-02 §3 places ETKDG L error from `local_structure_test.log` (0.036 Å / 3.88°, round-1 seeds, E-EVAL-047) next
   to MMFF L error from `l_error.csv` (0.0182 Å / 1.97°, E-EVAL-049). On the same file the ETKDG values are
   0.0314 Å / 2.80° / 10.65°. The juxtaposition overstates MMFF's bond/angle gain (it looks like 2×, it is 1.7× / 1.4×).
2. C-EVAL-03 §2 "find that diffusion models already beat MMFF on these" (bond/angle/torsion): in Nikitin Table 2 the
   diffusion models' torsion deviations (5.58–8.58°) are worse than MMFF→xTB's 4.89°.
3. C-EVAL-05 §2 cites EnFlow EnergySel (E-EVAL-011) as "energy-based post-processing trades recall for precision" and §4
   as evidence that relaxation loses recall ("A3 and EnFlow both lost recall"). EnergySel generates 3K and *discards* the
   1K with the highest learned energy (EnFlow p. 5 table notes). That is selection, not relaxation, and it is not
   evidence about xTB-after.
4. C-EVAL-05 §2 "TD relaxes only in its ensemble-property test, not for its RMSD tables [E-EVAL-024]": E-EVAL-024 does
   not show the second half. Cite the code path instead (no relaxation unless `--post_mmff`, E-EVAL-035).
Requested fix: use l_error.csv for both ETKDG and MMFF in C-EVAL-02. Qualify the torsion statement in C-EVAL-03.
Re-label the EnFlow analogy in C-EVAL-05 as selection.
Status: CLOSED-VERIFIED (V1, round 2: all four points fixed in C-EVAL-02/03/05; ratios 1.73x/1.42x/1.32x confirmed)
Response (EVAL): ACCEPTED, all four fixed. (1) C-EVAL-02 §3 now uses l_error.csv for both: ETKDG 0.0314 A / 2.80 deg /
10.65 deg vs MMFF 0.0182 / 1.97 / 8.07 (E-EVAL-049); the GT-GT spread (E-EVAL-047) is cited separately as context, not
as a like-for-like comparison. (2) C-EVAL-03 §2 qualified: diffusion models beat MMFF on bonds, angles and E_relax, not
on torsions (5.58-8.58 deg vs 4.89 deg); E-EVAL-020 revised. (3) C-EVAL-05: EnFlow EnergySel relabelled as learned-energy
*selection* (3K -> 2K, no relaxation; quote added to E-EVAL-011 from the Table notes); it is no longer cited as evidence
about relaxation, only as a separate example of a recall/precision trade-off from post-hoc filtering. (4) "TD does not
relax before RMSD" now cites the code path (no relaxation unless --post_mmff, E-EVAL-035) instead of E-EVAL-024.
Also fixed from the verdict table: C-EVAL-03 "computable where ETKDG fails" limited to non-ETKDG L sources; cost doubled
for the validation subset; C-EVAL-02 §2 rationale marked as the scout's INFERENCE; C-EVAL-04 §4 B-clust wording; side
notes of E-EVAL-010, 022, 028 corrected.

## D-101 on E-ROBUST-007
Raised by: V2
Problem: DOES NOT SUPPORT (wording). The shares reproduce exactly (V2 own script, `round3/ledger/v2_data/v2_lerror.out`),
but the claim "RDKit's endocyclic-dihedral error grows with ring size" is not monotone: smallest ring 4 = 60.4%,
5 = 48.1% (by largest ring: 45.7%, 43.1%). The "9% (smallest ring 3)" is a metric artefact: 3078 of the 4312 seeds
in that bin belong to molecules whose only rings are 3-membered, where every endocyclic dihedral is identically 0;
among the 1234 seeds that also carry a larger ring the share is 31.7%. Suggested claim: "ETKDG ring seeds exceed
10° endocyclic-dihedral RMSD in 48-92% of seeds whose smallest ring has 4 to ≥ 7 atoms (4: 60%, 5: 48%, 6: 67%,
≥ 7: 92%); 3-membered rings carry no such error by construction; under 0.04 Å/axis noise all bins are ≤ 3.3%."
C-ROBUST-02 §3 should drop or footnote the 9% figure.
Status: CLOSED-VERIFIED (V2 re-check): revised E-007 matches V2 recompute (60/48/67/92%, 3078 3-ring-only seeds all 0, noise <= 3.3%); C-ROBUST-02 dropped the 9%.
Response (ROBUST): ACCEPTED, entry revised. Recomputed independently (round3/ledger/robust_data/lerror_stats.py, new T4b): all 3078 seeds of 3-ring-only ETKDG molecules have endocyclic-dihedral error exactly 0 (max 0.000000), so the ring-3 bin is diluted by construction; 4 (60.4%) > 5 (48.1%), so 'grows with ring size' is false. E-ROBUST-007 now reads: 48-92% of seeds whose smallest ring has 4 to >= 7 atoms (4: 60%, 5: 48%, 6: 67%, >= 7: 92%; not monotone), 3-rings carry no such error by construction, noise <= 3.3% in every bin; marked 'Revised after D-101' with the original claim under History. C-ROBUST-02 §3 drops the 9% figure; C-ROBUST-01 §3 uses the 48-92% range with the 3-ring caveat.

## D-102 on E-ROBUST-008
Raised by: V2
Problem: DOES NOT SUPPORT. The comparison is not like-for-like in L-error magnitude or molecules.
(1) Magnitude (common 955 molecules, l_error.csv): A5ring heavy bonds 0.0270 Å, heavy angles 3.51°, heavy acyclic
angles 4.11°, all-atom acyclic angles 3.60°; noise 0.04: 0.0549 Å, 3.31°, 3.37°, 3.98°. Noise is 2× larger in
bonds but SMALLER in heavy-atom angles (0.94×) and heavy acyclic angles (0.82×). Noise 0.02 matches A5ring's bonds
(0.0275 Å) but has half its angle error (B1 0.117). So "per unit of bond/angle error" is not established, and the
card's "larger random error" is true for bonds only.
(2) Molecules: 0.1877 = B1 s0/s1/s2 S1_A5ring_cyc_ORACLE (0.1896/0.1922/0.1813, n 955); 0.1536 = S1_noise0.04pa_cyc_ORACLE
(0.1598/0.1376/0.1635, n 996). A5ring's 955 are a subset of the 996; no per-molecule outputs are synced, so no
paired check is possible.
What the data do support: B1 is hurt more by A5ring than by noise 0.04 (every seed), while CTRL is hurt equally
(0.1186 vs 0.1186). Please restate as that B1-specific, unpaired observation and drop "per unit" / "larger".
Status: CLOSED-VERIFIED (V2 re-check): per-unit claim withdrawn; revised E-008 (0.1877 vs 0.1536, CTRL 0.1186 both, profiles 2.0x/0.94x/0.82x, unpaired) matches V2 numbers; attribution labelled INFERENCE.
Response (ROBUST): ACCEPTED; original claim WITHDRAWN and replaced. Recomputed on the common 955 molecules (new T4c): A5ring heavy bonds 0.0270 A / heavy angles 3.51 deg / heavy acyclic angles 4.11 deg vs noise 0.04 0.0549 / 3.31 / 3.37 - your numbers reproduce, so 'per unit of bond/angle error' and 'larger random error' are not established. E-ROBUST-008 now states only the B1-specific, unpaired observation: B1 0.1877 (A5ring) vs 0.1536 (noise 0.04), every A5ring seed worse than every noise seed, CTRL equal (0.1186 vs 0.1186), with the error profiles stated (2.0x bonds, 0.94x heavy angles, 0.82x heavy acyclic angles) and the 955-vs-996 caveat; attribution to structured acyclic error is labelled INFERENCE. Marked 'Revised after D-102'; original wording kept under History. C-ROBUST-01 §3 rewritten to match.

## D-103 on E-ROBUST-043
Raised by: V2
Problem: WRONG LOCATOR. Quote exact on PDF p. 4, but it sits in §4.1 "Scheduled sampling formulated as KL divergence
minimisation" (starts p. 3), not §2. Content note for C-ROBUST-05: Huszár's result concerns feeding generated inputs
with the ORIGINAL targets (Eq. 4-6); C-ROBUST-05 §4 itself argues S3's targets are consistent, so citing E-043 as "a
real risk at high RDKit share" contradicts the card's own argument. Please fix the locator and revise the card use.
Status: CLOSED-VERIFIED (V2 re-check): locator now p. 4, Sec. 4.1; C-ROBUST-05 no longer cites Huszar as a risk for consistent targets.
Response (ROBUST): ACCEPTED. Locator fixed to PDF p. 4, Section 4.1 'Scheduled sampling formulated as KL divergence minimisation' (section starts p. 3; checked with pdftotext -layout). Content point also accepted: Huszár's failure needs generated inputs with the ORIGINAL targets, which S3 does not have (E-ROBUST-016). C-ROBUST-05 §4 no longer calls it 'a real risk'; it now says neither scheduled sampling's motivation nor Huszár's failure mode applies, and the drift-toward-CTRL argument is restated as a separate, labelled data-weighting INFERENCE. E-ROBUST-043 stays in §2 as context only.

## D-104 on E-ROBUST-046
Raised by: V2
Problem: WRONG LOCATOR. Quote exact on PDF p. 8, but in §4.2 "Varying the unconditional training probability", not
§4.1. Content verified.
Status: CLOSED-VERIFIED (V2 re-check): locator now p. 8, Sec. 4.2; quote unchanged.
Response (ROBUST): ACCEPTED. Locator fixed to PDF p. 8, Section 4.2 'Varying the unconditional training probability'. Quote and claim unchanged.

## D-105 on E-ROBUST-047
Raised by: V2
Problem: WRONG LOCATOR. Quote exact on PDF p. 9, but in §5 "Discussion" (last paragraph), not §6 "Conclusion".
Content verified (paragraph discusses CFG's disadvantages).
Status: CLOSED-VERIFIED (V2 re-check): locator now p. 9, Sec. 5 Discussion; quote unchanged.
Response (ROBUST): ACCEPTED. Locator fixed to PDF p. 9, Section 5 'Discussion' (last paragraph). Quote and claim unchanged.

## D-106 on E-ROBUST-049
Raised by: V2
Problem: WRONG LOCATOR. Quote exact on PDF p. 5 (arXiv v6), but in §4 "Scaling Compositional Generation with
Diffusion Models" (begins p. 4 right column; left column of p. 5 precedes §4.1), not §3. Content verified; Fig. 2 on
the same page covers both product and mixture.
Status: CLOSED-VERIFIED (V2 re-check): locator now p. 5 (v6), Sec. 4 before 4.1; quote unchanged.
Response (ROBUST): ACCEPTED. Locator fixed to PDF p. 5 (arXiv v6), Section 4 'Scaling Compositional Generation with Diffusion Models', before 4.1. Quote and claim unchanged.

## D-107 on E-ROBUST-058
Raised by: V2
Problem: WRONG LOCATOR (off by one). The quoted lines are torsional-diffusion/utils/dataset.py:145-147 and the
filter block is :145-162 (line 144 is the stale-cache assert). Substance verified.
Status: CLOSED-VERIFIED (V2 re-check): locator now dataset.py:145-162, quote :145-147, both checked.
Response (ROBUST): ACCEPTED. Locator fixed to torsional-diffusion/utils/dataset.py:145-162, quoted lines :145-147 (checked: line 144 is the stale-cache assert, the filter's print ends on :162).

## D-108 on E-ROBUST-006 (advisory; entry VERIFIED)
Raised by: V2
Problem: Numbers reproduce exactly, but the RDKit reference is selection-biased. tools/make_l_seed_pickles.py:105-114
keeps, per GT conformer, only the ETKDG seed picked by a Hungarian assignment that MINIMISES heavy-atom angle RMSD
over 2L seeds; noise rows use their own source GT. Against the matched RDKit L that CTRL/S3 train on (L_lam0.00, same
936 molecules: bonds 0.0344 Å, angles 3.94°), noise 0.04 is 1.60× the bond error and 0.84× the angle error, not
1.75× / 1.18×. Please state the reference in the claim and add the λ = 0 comparison; C-ROBUST-01 §3 should not say
noise exceeds RDKit's angle error. The ring-tail contrast is robust (28.6% / 30.8% vs 0.35%; 49.6% vs 0.58% among
seeds with a ring of ≥ 4 atoms).
Status: CLOSED-VERIFIED (V2 re-check): revised E-006 names the Hungarian-selected reference (make_l_seed_pickles.py:106-114) and gives 1.60x/0.84x vs matched lam=0; T4a/T4b equal V2 recompute.
Response (ROBUST): ACCEPTED. Confirmed tools/make_l_seed_pickles.py:106-114 assigns each GT conformer the ETKDG seed by a Hungarian assignment on heavy-atom angle RMSD, so the L_etkdg2L rows are biased low. Recomputed (new T4a, common 936 molecules): matched RDKit L (lambda = 0) 0.0344 A / 3.939 deg vs noise 0.04 0.0549 A / 3.311 deg = 1.60x bonds, 0.84x angles; ring tail > 10 deg 28.6% (ETKDG) / 30.8% (lambda 0) vs 0.35% (noise); among molecules with a ring of >= 4 atoms 49.6% vs 0.58% (T4b). E-ROBUST-006 revised to name the reference, give both comparisons, and say noise exceeds RDKit in bonds only; C-ROBUST-01 §3 no longer says noise exceeds RDKit's angle error. Marked 'Revised after D-108', original under History.

## D-109 on C-ROBUST-05 (card use of E-ROBUST-012)
Raised by: V2
Problem: §3 uses E-012 (B1's 0.092-0.134 spread under 0.02 Å noise) to argue that 0.004 / 0.013 Å effects "sit near
our noise". Seed ranges at the relevant endpoints are 0.001-0.005 Å (CTRL RDKit L 0.1774-0.1783; B1 RDKit L
0.2327-0.2381; CTRL gtLcycle 0.0816-0.0827; B1 gtLcycle round 1 0.0206-0.0215). The evidence does not support the
sentence; S3's own seed variance is unknown. Also the gaps should use like-for-like round-1 in-job values (CTRL 0.1786,
B1 gtLcycle 0.0212): 0.003 / 0.012 Å.
Status: CLOSED-VERIFIED (V2 re-check): E-012 use withdrawn; new E-060/E-061 (0.1786, 0.2341, 0.0212; ranges 0.001-0.007 A) equal V2 round-1 recompute; gaps 0.003/0.012 A.
Response (ROBUST): ACCEPTED. The E-ROBUST-012 sentence is withdrawn from C-ROBUST-05 (wrong condition). Recomputed like-for-like round-1 in-job values (new T5): CTRL_rematch RDKit L 0.1786 (0.1773-0.1800), B1 RDKit L 0.2341 (0.2299-0.2364), B1 gtLcycle 0.0212 (0.0206-0.0215), all 3 seeds; added as E-ROBUST-060 (values) and E-ROBUST-061 (endpoint seed ranges 0.001-0.007 A). C-ROBUST-05 now uses gaps 0.003 / 0.012 A and states the gaps are resolvable with 3 seeds; the NO rests on headroom (<= 0.003 A on the primary non-oracle endpoint, <= 0.012 A on an ORACLE one), the expected trade-off between endpoints (labelled INFERENCE) and cost. I still vote NO, on these corrected grounds.

## D-110 on C-ROBUST-01 (card sentences beyond E-008)
Raised by: V2
Problem: (a) §2 "The augmentation had to match the upstream error" is attributed to CDM but CDM states no matching
principle (E-025 reports Gaussian noise failed and blur worked at high resolution); label INFERENCE. (b) §2 ProteinMPNN
"because exact coordinates carry information" overstates a hedged source ("may impart", E-032). (c) §3 "The two
trained models use different parts of L": against the like-for-like λ = 0 base (same 955 molecules) B1 gains as much
from true rings (0.2614 → 0.1877) as CTRL (0.1966 → 0.1186); only the acyclic gain differs. (d) §3 "true rings alone
take AMR-R from 0.178 to 0.119" should use the λ = 0 base (0.197 → 0.119).
Status: CLOSED-VERIFIED (V2 re-check): (a) INFERENCE label added, (b) "may impart" hedge, (c) lam=0 base addendum to E-010 matches T3, (d) 0.197 -> 0.119 used.
Response (ROBUST): ACCEPTED, all four parts. (a) C-ROBUST-01 §2 now says CDM found the augmentation type mattered and labels 'should match the upstream error' as our INFERENCE. (b) ProteinMPNN is hedged ('may impart', stated by the authors as a hypothesis). (c) Using the lambda = 0 base (same 955 molecules, T3): CTRL 0.1966 -> 0.1186 with true rings, B1 0.2614 -> 0.1877, i.e. both gain about equally from rings; only B1 gains much from true acyclic geometry (-0.103 vs -0.014). Added as an addendum to E-ROBUST-010 and §3 rewritten; the factorised-training rationale is labelled INFERENCE. (d) Ring-component effect now quoted as 0.197 -> 0.119 (-0.078) on the like-for-like base. Also adopted: like-for-like B1 in-job values (E-ROBUST-060), and Arm 0's ORACLE gate stated as experiment selection only, flagged for the validity judge. Call stays YES (gated): the core mechanism rests on VERIFIED entries.


## D-201 on E-GEOM-013
Raised by: V3
Problem: the numbers are right (BRIEF lines 37–38; 3-seed means CTRL A5ring 0.1186, B1 0.1877, CTRL A5acyc 0.1824, B1
0.1583), but the claim "Improving ring geometry alone helps the standard model but not B1" does not follow. B1 improves
with true rings even against the ETKDG base used in the claim (0.2356 → 0.1877). A5ring/A5acyc are built from the matched
λ = 0 pairs (`tools/make_l_seed_pickles.py:236-246`), so the like-for-like base is λ = 0 (same 955 molecules): CTRL
0.1966 → A5ring 0.1186 (−0.078), B1 0.2614 → 0.1877 (−0.074). True rings help both models equally; true acyclic geometry
helps B1 much more (−0.103 vs −0.014). The inference "a ring-only L source is a CTRL-side lever" is not supported. Same
point as V2's D-110(c). Used in C-GEOM-01 §3 ("so it is the arm that can also move B1"), C-GEOM-02 §3 and C-GEOM-04 §3
("0.178 → 0.119").
Requested fix: restate as "true rings alone help both models by ≈ 0.075 Å (λ0 base) but leave B1 above CTRL; only true
acyclic geometry puts B1 below CTRL"; quote ring gains against the λ0 base (0.197 → 0.119), or state that the ETKDG base
is a different population (936 vs 955) and seed set.
Status: CLOSED-VERIFIED (V3 R2: E-013 revision and λ0-base numbers recomputed; cards 01/02/04 fixed)
Response (GEOM): ACCEPTED; inference WITHDRAWN. Recomputed on the like-for-like λ = 0 base (BRIEF lines 30, 37, 38; same 955 molecules): true rings alone CTRL 0.197 → 0.119 (−0.078), B1 0.261 → 0.188 (−0.074); true acyclic alone CTRL −0.014, B1 −0.103, and only A5acyc puts B1 below CTRL (0.158 < 0.182). E-GEOM-013 now carries a 'Revised after D-201' block (original kept, marked superseded); E-GEOM-006 has a λ = 0-base addendum (rigid 0.184 → 0.043, one rotor 0.181 → 0.104). Cards fixed: C-GEOM-01 §3 (rings help both; xTB is the B1 lever because it also relaxes acyclic L, labelled INFERENCE), C-GEOM-02 §3 and §6, C-GEOM-04 §3 now quote 0.197 → 0.119 / 0.261 → 0.188.

## D-202 on E-GEOM-014 (minor)
Raised by: V3
Problem: versions verified (`slurm/setup_env.sh:76-79`, `:99-100`; `round2/IMPLEMENTATION.md:130`). But "built with
micromamba from conda-forge" is not what the script does. `setup_env.sh:43-58` takes a system/module python3.9 and only
falls back to a micromamba conda-forge python if none is found; all packages are pip-installed into a venv
(`:62-103`). Nothing on disk shows which branch ran on gnode118.
Requested fix: "pip venv on python 3.9 (micromamba/conda-forge python only as a fallback, `setup_env.sh:43-58`)".
Status: CLOSED-VERIFIED (V3 R2; nit: the cu118 profile exists in setup_env.sh but default is cu117, "already used" is not shown)
Response (GEOM): ACCEPTED. E-GEOM-014 revised: pip venv on python 3.9; micromamba/conda-forge python only as the fallback (`setup_env.sh:43-58`), packages pip-installed (`:62-103`); branch used on gnode118 not recorded. Added for D-206: the venv pins `numpy==1.23.5` (`:68`, `:89`) and the script has a `cu118` profile (`torch==2.0.1+cu118`, PyG 2.3.1, `:80-84`).

## D-203 on E-GEOM-045
Raised by: V3
Problem: the quotes are exact (live notebook = snapshot). But the "ETKDG v1" run in the post (cell 6) is
`params = Chem.rdDistGeom.EmbedParameters()` passed to `EmbedMultipleConfs(..., params=params)`. A bare
`EmbedParameters` has `useExpTorsionAnglePrefs = False` and `useBasicKnowledge = False` (RDKit Book parameter list, also
in the scout's own `papers/blogs/rdkit_book.txt`; local check ET False, K False), so it is plain distance geometry, not
ETKDG v1. The post's label is wrong. The MMFF counts (366 / 131 / 3) are MMFF on plain-DG output, scored with the naive
|torsion| classifier the author later shows over-counts chairs. Only "srETKDGv3: 358 chair / 142 twisted (sign check)"
is an ETKDG result. C-GEOM-01 §2 uses the ETKDG/MMFF part ("force-field relaxation only partly repairs ETKDG ring
puckers (piperazine)").
Requested fix: limit the claim to the srETKDGv3 result; describe the first run as "default EmbedParameters (plain DG)".
In C-GEOM-01 §2 cite E-GEOM-002 (MMFF 20.2% vs ETKDG 28.6% > 10°) for "MMFF only partly repairs puckers" instead.
Status: CLOSED-VERIFIED (V3 R2: E-045 limited to srETKDGv3; C-GEOM-01 re-sourced to E-002)
Response (GEOM): ACCEPTED. Checked locally (RDKit 2026.03.6): bare `EmbedParameters()` has ET False, K False, so the post's first run is plain DG. E-GEOM-045 revised: only 'srETKDGv3: 358 chair / 142 twisted (28%)' is kept; the ETKDG/MMFF part is WITHDRAWN. C-GEOM-01 §2 now cites E-GEOM-002 (MMFF 20.2% vs ETKDG 28.6% > 10°; 35.1% vs 49.6% excl. all-3-ring molecules) and no longer cites E-045. E-045 now supports C-GEOM-06 only.

## D-204 on E-GEOM-048
Raised by: V3
Problem: the caption quote (p. 7) is exact, but the claim "the default O(3) model is not chirality-corrected" is
contradicted by the same paper, p. 5 §3.4: "Our base method (ET-Flow) corresponds to using the post hoc correction
whereas the SO(3) variant is referred by ET-Flow-SO(3)." The O(3) model applies an oriented-volume check against the
RDKit chiral tags and flips the conformer on mismatch. The 0.073 in Table 2 is that corrected O(3) model. C-GEOM-03 §4 and
§6 (the "O(3) chirality fix" as a reason for MAYBE) rest on this.
Requested fix: "ET-Flow (O(3)) uses a post hoc chirality correction (§3.4); SO(3) is an architectural alternative".
Check whether `etflow`'s `BaseFlow.predict` applies the correction; keep a stereo check on the seeds either way.
Status: CLOSED-VERIFIED (V3 R2: E-053 quotes exact p. 5; E-054 code facts checked on main, configs.py:118, :189, model.py:459-462, :525)
Response (GEOM): ACCEPTED; E-GEOM-048's claim WITHDRAWN (entry kept with a revision block; supports no card). New E-GEOM-053 (ET-Flow p. 5 §3.4: post hoc flip on oriented-volume mismatch; 'Our base method (ET-Flow) corresponds to using the post hoc correction'). New E-GEOM-054 checks the package as requested: `etflow/commons/configs.py:118` `parity_switch = "post_hoc"` is the `ModelArgsSchema` default used by `QM9_O3` (`:189-196`), and `model.py:459-462` applies it inside `sample()`, which `predict()` calls (`:525`); snapshot `papers/blogs/etflow_repo_code_chirality.txt`. C-GEOM-03 §2/§4 rewritten; a per-seed stereo check stays because the flip is whole-molecule.

## D-205 on E-GEOM-004 (advisory; entry VERIFIED) and the rigid-subset gates of C-GEOM-01/02/04/06
Raised by: V3
Problem: values and quote are exact, and TD's code path supports the reasoning (`generate_confs.py:150-153` skips
perturbation and model when `edge_mask` is empty; H-only rotors leave heavy atoms on the rotation axis,
`utils/torsion.py:57-74`; evaluation is heavy-atom `GetBestRMS`, `evaluate_confs.py:115`). But (a) the logged bin 0 is
not exactly model-free. With identical seed pickles and n, λ = 1.00 (true L, should be 0.000) gives CTRL
0.004 / 0.004 / 0.007 and B1 0.001, and ETKDG gives CTRL 0.152 / 0.151 / 0.154 vs B1 0.155 / 0.156 / 0.155. So
`tools/breakdown.py`'s bin 0 (from corrected SMILES) contains a few molecules TD does rotate. (b) The rigid sets differ
per L source: ETKDG/MMFF 284, λ0.25–0.75 258, λ0/λ1/A5 289, noise 318. The gate thresholds "rigid AMR-R ≤ 0.110"
(C-GEOM-01), "≤ 0.107 (λ = 0.5 level)" (C-GEOM-02) and "> 0.077" (C-GEOM-04) compare a new source on the ETKDG
population with λ values on 258 molecules.
Requested fix: in the CPU gates, score the seed sets directly (AMR-R of seeds vs GT, no model) on one common rigid set
defined by TD's own `edge_mask` heavy-atom criterion, and recompute the λ / MMFF / ETKDG reference values on that set.
Status: CLOSED-VERIFIED (V3 R2: gates redesigned; note "empty edge_mask" is stricter than the heavy-atom criterion and shrinks the set)
Response (GEOM): ACCEPTED. E-GEOM-004 revised: logged bin-0 values are indicative (≤ 0.006 Å model dependence; populations 258–318). Gate redesigned in C-GEOM-01 §5 Step 2 and adopted by C-GEOM-02/03/04/06: score the seed sets directly (heavy-atom symmetric AMR-R of seeds vs GT, no model) on one common rigid set = test molecules with an empty TD `edge_mask` present in every compared source; recompute ETKDG / MMFF / λ references on that set in the same run. Go rules are now non-oracle comparisons (beat MMFF or ETKDG by ≥ 0.010 Å rigid AMR-R or ≥ 5 points of > 10° share); λ references are interpretation yardsticks only (flagged as experiment selection for the validity judge). All absolute thresholds (0.110, 0.107, 0.077) removed.

## D-206 on C-GEOM-03 (card sentences beyond E-GEOM-037)
Raised by: V3
Problem: §4 "ET-Flow needs torch ≥ 2.1 / CUDA 12.1 ... RTX 3090 is supported by CUDA 12.1 only with a recent enough
driver" and §6 "it hinges on a separate CUDA-12 environment" are not supported. `pytorch-cuda==12.1` is only in the dev
`env.yml`. The PyPI `etflow` 0.1.2 pins `numpy==1.26.4` but leaves `torch` unpinned, and the README says to install
pytorch first. Latest `lightning` (2.6.6) needs torch ≥ 2.1 and python ≥ 3.10, but `lightning` 2.2.0 accepted
torch ≥ 1.13. A cu118 torch 2.x env (cf. `slurm/setup_env.sh` profile `cu118`) is a plausible untested route. The
separate-env conclusion stands because of the numpy pin. The README also lists separate "Scaffold Splits and
Checkpoints" (zenodo 16551316): which split `qm9-o3` was trained on is not established. Plus the chirality point
(D-204).
Requested fix: replace the CUDA-12 requirement with "own env (numpy 1.26.4, torch 2.x; CUDA version open, try cu118
first)"; add a step-0 check of the `qm9-o3` training split; revise §6 accordingly.
Status: CLOSED-VERIFIED (V3 R2: CUDA and chirality fixed; split provenance still open, see D-210)
Response (GEOM): ACCEPTED, and the call changes MAYBE → YES. Both MAYBE reasons are withdrawn: chirality is corrected in the released model (D-204, E-GEOM-053/054), and CUDA 12 is only the dev env.yml choice. C-GEOM-03 §4 now says: own env because of the numpy 1.26.4 pin (TD venv has 1.23.5, E-GEOM-014 revision); torch/CUDA open, try a cu118 torch 2.x first. Split: Zenodo record 14226681 holds `qm9-o3.ckpt` with `QM9.zip`, the scaffold splits are a separate record (16551316) (E-GEOM-054); that `qm9-o3` used TD's random split is labelled INFERENCE, and step 0 now compares `QM9.zip`'s split with our `test_smiles.csv` as a kill criterion. Card stays a diagnostic, ranked below C-GEOM-01/02. I did not cite the lightning version details (no entry of mine).

## D-207 on C-GEOM-02 (card sentences)
Raised by: V3
Problem: (a) §3 "60–92% for 4- to 7-membered rings [E-GEOM-003]". E-003 gives 5-rings 48% (my recomputation 48.1%), so
the range is 48–92%. (b) §3 "True ring geometry alone takes CTRL from 0.178 to 0.119" mixes the ETKDG base (936
molecules) with the λ0-based oracle (955); like-for-like 0.197 → 0.119 (D-201). (c) §5 "~10⁶ training conformers" has
no evidence entry. (d) §5 gate "rigid AMR-R ≤ 0.107" is on a different population (D-205). (e) Optional: E-012's
coverage numbers depend on the key including stereo tags (isomeric SMILES). A stereo-free key gives 421 components and
66.3% / 58.6% leave-one-out coverage, which helps the card. State the key.
Requested fix: correct (a) and (b), add a source or "UNVERIFIED" to (c), pair the gate as in D-205.
Status: CLOSED-VERIFIED (V3 R2: all five parts fixed; recomputed)
Response (GEOM): ACCEPTED. (a) 'molecules whose smallest ring has 4–7 atoms: 48–92%'. (b) like-for-like 0.197 → 0.119 (CTRL) and 0.261 → 0.188 (B1) with rigid 0.184 → 0.043 (E-GEOM-006 addendum). (c) '~10⁶ training conformers' replaced by 'count not established here, UNVERIFIED' (also in C-GEOM-01). (d) gate paired as in D-205. (e) Recomputed (scratchpad `p2.py`): isomeric key 534 / 50.8% / 45.2%; stereo-free key 421 / 66.3% / 58.6% (345 bad-seed molecules) — both now in E-GEOM-012 and C-GEOM-02 §3.

## D-208 on E-GEOM-041 / E-GEOM-008 (advisory; entries VERIFIED) and C-GEOM-06 design
Raised by: V3
Problem: the open part of E-041 is now resolved. RDKit `Release_2022_09_5`,
`Code/GraphMol/DistGeomHelpers/Wrap/rdDistGeom.cpp:337-351` (python `EmbedMultipleConfs` args):
`useExpTorsionAnglePrefs = true`, `useBasicKnowledge = true`, `useSmallRingTorsions = false`,
`useMacrocycleTorsions = false`, `ETversion = 1`. So the cluster's training-matching and test-time seeds are ETKDG
**v1**. C-GEOM-06's `srETKDGv3` arm against `L_etkdg2L` therefore changes ETversion 1 → 2, macrocycle torsions, 1-4
config **and** small-ring torsions together, and its "≤ 0.005 Å" expectation covers two changes, not one.
Requested fix: update E-041 / C-GEOM-06 §2 with the 2022.9.5 defaults; add an `L_etkdgv3_2L` arm (ETKDGv3 without small
rings, same random seed) so the small-ring terms are isolated. CPU only, no extra GPU unless it passes the gate.
Status: CLOSED-VERIFIED (V3 R2: E-052 = Release_2022_09_5 rdDistGeom.cpp:346-351, re-fetched; C-GEOM-06 single-factor)
Response (GEOM): ACCEPTED, with my own check of the source: new E-GEOM-052 (RDKit `Release_2022_09_5` `rdDistGeom.cpp:346-351`: ET/K true, small-ring false, macrocycle false, `ETversion = 1`; snapshot `papers/blogs/rdkit_2022_09_5_rdDistGeom_wrapper.txt`). E-GEOM-041 revised accordingly. C-GEOM-06 is now single-factor: `L_etkdg2L` (v1) vs `L_etkdgv3_2L` (ETKDGv3, small rings off) vs `L_sretkdg2L` (srETKDGv3), same random seed; the ≤ 0.005 Å expectation now applies only to the v3 → srv3 step, and no expectation is stated for v1 → v3 (no source). C-GEOM-01's table labels ETKDG as v1.

## D-209 on C-GEOM-04 (card sentences)
Raised by: V3
Problem: (a) §3 "Substituents matter in QM9 (every test molecule has them)": 53 of the 885 test ring molecules have no
exocyclic heavy atom (RDKit on the l_error.csv SMILES). (b) §4 "fused systems need per-ring Cremer–Pople coordinates
plus closure constraints [E-GEOM-020, E-GEOM-033]": PuckerFlow p. 6 (same paragraph as E-020) says the approach "can
also be applied to fused and spiro rings, where the Cremer-Pople coordinates can be determined for each component ring
separately". E-033 is about macrocycle NeRF reconstruction, so "closure constraints" for fused 5/6 systems is
INFERENCE. (c) Inherits the "0.178 → 0.119" base issue (D-201).
Requested fix: "almost every (832 of 885 ring molecules)"; label (b) INFERENCE and cite PuckerFlow's per-component-ring
statement. The card's NO call is unaffected.
Status: CLOSED-VERIFIED (V3 R2: 832/885 recomputed; E-055 quote exact p. 6)
Response (GEOM): ACCEPTED. (a) Recomputed: 53 of 885 ring molecules have no exocyclic heavy atom → 'almost every (832 of 885)' (E-GEOM-011 addendum; C-GEOM-04 §3). (b) New E-GEOM-055 (PuckerFlow p. 6, per-component-ring extension to fused/spiro); C-GEOM-04 §3/§4 cite it and label extra closure constraints for fused systems as INFERENCE (E-033 is macrocycle-only). (c) λ = 0 base used (D-201). NO call unchanged.

## GEOM note (coordinator item, no thread number): l_error grouping
Response (GEOM): ACCEPTED (V3 recomputation R2, reproduced with scratchpad `p2.py`). 3-ring "dihedrals" are identically 0, so the 113 all-3-ring molecules add zeros. Excluding them: share of ring seeds > 10° is ETKDG 49.6%, MMFF 35.1%, λ0 52.1%, λ0.25 38.6%, λ0.50 19.4%, λ0.75 0.02%, noise 0.04 0.58%. E-GEOM-002 has a revision block; C-GEOM-01 (table now shows both bases), -02, -04, -05, -06 quote both bases; the gate uses the excl.-all-3-ring basis.


## D-210 on C-GEOM-03 / E-GEOM-054 (split provenance of `qm9-o3`; validity)
Raised by: V3
Problem: that `qm9-o3` was trained on TD's (GeoMol) QM9 split is not established, and the sources give reasons to doubt
that step 0(a) can settle it in 1 h. (1) ET-Flow p. 6 states the GeoMol split (243473/30433/1000) for DRUGS only; for
QM9 it says only "we train and test model on GEOM-QM9". (2) At the release commit of the checkpoints (`4e1c5aa942`,
2024-12-12), `configs/qm9-o3.yaml:15-16` trains from `QM9/train_indices.npy` / `QM9/val_indices.npy`. These are
presumably the files in Zenodo `QM9.zip` (2.9 MB), and they index ET-Flow's own processed order (old
`scripts/prepare_data.py` iterates `summary_qm9.json`), not TD's file list. To map them to SMILES you need ET-Flow's old
processed `smiles.npz`, or a re-run of that preprocessing. (3) Current `main` (`scripts/prepare_data.py:199-206`)
applies TD's `split.npy` indices to an unsorted `list(glob("*"))`, while TD indexes a sorted list
(`torsional-diffusion/utils/dataset.py:171`). So a model retrained from `main` would not get TD's train set either. If
any TD test molecule is in `qm9-o3`'s training set, ET-Flow L for it is partly memorised GT: ORACLE-contaminated, and it
could falsely meet the λ = 0.75 spec.
Requested fix: in C-GEOM-03 step 0(a), state the check as "map `QM9.zip` train/val indices to SMILES (old ET-Flow
preprocessing) and intersect with TD `test_smiles.csv`". Add: "if the mapping cannot be done or is inconclusive, stop
(or label every ET-Flow-L result ORACLE)". Label E-054's split statement INFERENCE with these two counter-points. The
YES call should read "YES, conditional on a conclusive step 0(a)".
Status: OPEN
