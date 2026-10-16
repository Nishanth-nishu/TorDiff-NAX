# Verification V5 (round 5): EXP-1, EXP-4 (ledger JOINT) and EXP-2, EXP-3 (ledger CHEM)

Verifier: V5, 2026-10-15. Protocol v2 (`orchestration/README.md`), role `orchestration/roles/verifier.md`.
Scratch scripts (not committed, session scratchpad `v5/`): `qcheck.py` (re-extracts every cited PDF with `pdftotext`
in both layout and reading order, normalises, and checks every `>` quote against the claimed page), `floor010.py`
(E-JOINT-010), `projector_test.py` (EXP-1 §2.2), `census_v5.py` (EXP-3 census). The same checks can be re-run from the
code shown in the dispute threads.

## Summary

| Check | Result |
|---|---|
| Paper quotes, both ledgers (99 quote lines) | 92 match on the claimed page automatically. The other 7 were found by hand on the claimed page (the PDF text drops the Å glyph, splits table columns, or interleaves two columns). **No quote is invented and none is on the wrong page.** |
| Prior-art context (RINGER, GO-Flow, FoldingDiff, PuckerFlow, EBD, Noé 2019, TBG, BoltzNCE, VonMisesNet, GeoMol OT, Torsional-GFN, Adjoint Sampling, TD §3.6 / Table 8 / App. E / App. H) | Every NOVELTY / PRIOR ART verdict is backed by its source. Two absence claims hold on my own extraction: GO-Flow has no closure, NeRF or cycl* treatment, and EBD has no torsion factor. PuckerFlow's joint CP × torsion flow is described as a *feasible extension*, not an announced one (wording only). |
| LoQI | Code claims VERIFIED: I fetched `stereo.py` live at commit 04f3caef and lines 3–4 match the snapshot. Claims from the paper PDF (0.956 / 0.992 stereo accuracy; "TD produces incorrect stereoisomers") are **UNVERIFIABLE** because the PDF is blocked; the author had already labelled them UNVERIFIED. |
| **E-JOINT-010 (0.032 Å floor)** | **Value is wrong because of a code bug.** `tools/local_structure_analysis.py:302` takes `argsort(cost_gg[i])[:3]`, and that includes the target conformer itself whenever n_GT ≤ 3. All 544 such rows have a floor of about 1e-7. On the n_GT > 3 rows, which the bug does not touch, the ring floor is **0.0359 Å** (4,696 conformers). On those same rows the RDKit-L floor is 0.1143 Å. The floor is also an oracle best-of-3 over *all* other GT conformers, ranked by closeness to the target, so it is not the split-half analogue. The direction of the claim survives. The number and the phrase "another conformer's L" do not. → D-003 |
| E-JOINT-001 / E-CHEM-054 (leak-free) | All numbers VERIFIED. But "every model is worse with another conformer's L than with RDKit L" compares the split view (2·\|half\| samples per molecule) against `ref_rdkitL` (2·n samples). Halving the sample budget raises AMR-R by itself, so the CR +0.032 Å gap is not budget-matched. The paired own-vs-split leak contrast is unaffected. → D-001 |
| **EXP-1 projector math** | Well defined (rank B = m+6, dim range P_L = 3n−6−m, checked on 4 molecules). Rings stay closed trivially, because the atoms are Cartesian. Chirality **can** flip (1 of 20 random steps at 0.3 Å RMS), so the §2.8 backtrack is needed. **But claim (1), "changes no rotatable torsion to first order", is false for dihedral angles.** P_L removes the m torsion *tangent* directions ∂x/∂τ. It does not annihilate the dihedral *gradients* ∇τ. Measured first-order dihedral drift is 0.19–0.25 rad per Å of step, the same as an unprojected step. The orthogonal projector onto ker W (W = dihedral gradients) gives 0.000. → D-004 |
| Parity (TD §3.3, PDF p. 6) | VERIFIED. Torsion 0o, bond lengths and angles 0e, Cremer–Pople pairs 0o, Cartesian velocity 1o are all consistent with Prop. 2. |
| EXP-1 coupling claim | **Not supported as designed.** The A6 control runs the frozen A0 torsion model, which was trained on RDKit L. E-JOINT-010 itself names exactly that train/test L mismatch as the cause of the leak-free failure. So A5 − A6 cannot separate "coupling" from "torsion head trained on its own L". → D-005 |
| R-CHEM QM9 census | **All recomputed exactly**: 2,880 TD-rotatable bonds, 108 of them double; 65/526 = 12.4 % E/Z-absent-from-GT (R0); 16.6° mean double-bond twist (GT 1.8°); AMR-R 0.257 vs 0.166. Caveats: 12.4 % is the R0 seed only (R0–R2 give 12.4 / 10.3 / 10.5 %, mean 11.1 %), and 108/2,880 = 3.75 %. |
| Code locators | `featurization.py:19,59,63,101` VERIFIED: `chiral_tag` is built and never returned. `sampling.py:107` VERIFIED: uniform re-randomisation of every rotatable torsion, called at `generate_confs.py:150-151`. JOINT locators `likelihood.py:68-98`, `score_model.py:83-89,119-132,141-143`, `training.py:25-31`, `dataset.py:79-99`, `sampling.py:204`, `torsion.py:57` all VERIFIED. |
| E-JOINT-004 (ring "tail") | Numbers VERIFIED (49.6 / 35.1 / 19.4 / 0.02 %; mean 10.65°, median 0.44°). But the mean and median pool 3,078 three-membered-ring seeds, which always sit at 0°. For rings of 4 or more atoms the median is 9.8° and 49.6 % of seeds exceed 10°, so this is a bulk error, not "a minority of seeds". → D-002 |
| GPU-h vs measured | The 11.7 GPU-h basis matches `round3/grounding.md` §2.2 (377.7 s/epoch logged, plus validation). No raw training log is present locally. EXP-2 and EXP-3 arithmetic checks out. EXP-1 M1 is costed three ways (3 / 4–6 / 12 GPU-h) → D-006. EXP-4 variant D on 100 molecules computes to ≈ 5–14 GPU-h, not 3–6 → D-007. The "+15 % projector overhead" is unmeasured. |

**Per-spec rulings:**
- **EXP-1: WEAKENED.** The prior-art table is sound. Two of the three claimed contributions are not supported as written: the exact torsion separation (D-004) and coupling being necessary (D-001, D-003, D-005).
- **EXP-4: SUPPORTED**, with two corrections (D-007; and the τ in the chart is a chart coordinate, not a dihedral).
- **EXP-2: SUPPORTED.** The framing paragraph inherits D-001 and D-003.
- **EXP-3: SUPPORTED.** One headline-number correction (D-008).

---

## Per-entry verdicts (checked entries; everything else was sampled out and is listed at the end)

### Quote audit (automated, both ledgers)
`qcheck.py` re-extracted each cited PDF itself and did not use the scouts' fulltexts. 92 of 99 quote lines matched on
the claimed page. I found the remaining 7 by hand:

| Entry | Reason the automated match failed | Verdict |
|---|---|---|
| E-JOINT-009 | DiTMC Table 1 extracts with the method names shifted by one row: the "Tors. Diff." label sits beside GeoMol's numbers (91.5 / 0.225), and 92.8 / 0.178 sits beside "DMT-B". The ledger's "raw extraction" line is a *reassembly*. The values are correct when rows are read in order (TD 0.178, MCF-B 0.103, DMT-B 0.090, ET-Flow 0.073, DiTMC 0.073 / 0.070 / 0.068). | QUOTE MISMATCH (cosmetic; numbers VERIFIED) → D-008 |
| E-JOINT-046 | Text is "GEOMDRUGS" in the extraction; found on p. 10 | VERIFIED |
| E-JOINT-065 | Å glyph lost; found on p. 7 | VERIFIED |
| E-JOINT-068 | Å glyph lost; found on p. 18 | VERIFIED |
| E-JOINT-076 | Table 7 extracts the last two rows unlabelled. Row order gives w/o-Alignment 0.622 and Gaussian Prior 0.640, as claimed. Found on p. 19 | VERIFIED |
| E-JOINT-133 | Å glyph lost; found on p. 1 | VERIFIED |
| E-CHEM-046 | Two columns interleave ("post / hoc"); found on p. 5 | VERIFIED |

### Prior art that carries a NOVELTY verdict (context read, not only the quote)
- **E-JOINT-060 RINGER** (p. 5): VERIFIED. The source is about macrocycles, has redundant internal coordinates, and uses SLSQP. Ruling "joint angle + torsion diffusion = PRIOR ART" is correct.
- **E-JOINT-061 RINGER/TD** (p. 6, 19): VERIFIED.
- **E-JOINT-062 FoldingDiff** (p. 4, 10): VERIFIED, including the lever-arm sentence.
- **E-JOINT-063 GO-Flow** (p. 4): VERIFIED.
- **E-JOINT-064 GO-Flow absence**: VERIFIED on my own extraction. The only ring hits are "distorted rings … in baseline methods" and the atom feature; there is no "closure", "NeRF" or "cycl*". It remains an absence claim.
- **E-JOINT-065 GO-Flow**: VERIFIED.
- **E-JOINT-066, -067, -068 PuckerFlow**: VERIFIED.
- **E-JOINT-069 PuckerFlow extension** (p. 10): quote VERIFIED. The README snapshot does say "builds on the codebase of Torsional Diffusion" (snapshot line 71). The paper calls a joint flow "feasible"; it does not announce it. EXP-1 should write "proposed", not "announced" (D-008). The novelty risk the author flags is real either way.
- **E-JOINT-070 TMCgen**: quote VERIFIED.
- **E-JOINT-074 Torsional-GFN** (p. 4): VERIFIED.
- **E-JOINT-075 CoarsenConf**: VERIFIED.
- **E-JOINT-077 FM-refiner**: VERIFIED.
- **E-JOINT-078 Corso thesis**: VERIFIED.
- **E-JOINT-090 / -091 / -092 EBD**: VERIFIED. My extraction has "torsion" only in reference titles, which confirms "no torsion factor".
- **E-JOINT-093 HierDiff, -094 OctMole**: VERIFIED.
- **E-JOINT-095 MSGEN, -096 dual-scale BG**: abstract or snapshot only, not re-fetched. The author already restricts them to abstract-level use. UNVERIFIABLE beyond the abstract (no dispute; scope is stated).
- **E-JOINT-110 Noé 2019** (p. 14, 25): VERIFIED. Internal coordinates plus normalisation were essential, so "BG over bonds/angles = PRIOR ART" is correct.
- **E-JOINT-111 TBG** (p. 2): VERIFIED. The TBG paper states TD's fixed-L gap verbatim.
- **E-JOINT-112 Corso, -113 BoltzNCE, -114 CG-BG**: VERIFIED.
- **E-JOINT-079 search log**: VERIFIED as a record (18 queries in `joint_data/search_log.txt`). As the author says, it is absence evidence, not proof.
- **E-CHEM-009 TD App. E Alg. 4** (p. 19): VERIFIED.
- **E-CHEM-010 TD Table 8** (p. 27) **and App. H items 3 and 5** (p. 26): VERIFIED. Item 3 defines "Only D.E. matching" as random assignment; item 5 defines "Train on ground truth L".
- **E-CHEM-011 / -012 TD §3.6, Alg. 1** (p. 7): VERIFIED.
- **E-CHEM-014 TD App. G.2**: VERIFIED.
- **E-CHEM-030 GeoMol OT loss** (p. 7): VERIFIED.
- **E-CHEM-031**: VERIFIED.
- **E-CHEM-032 / -034 Adjoint Sampling** (p. 12, 11): VERIFIED. Torsions are conditioned on bonds and angles and trained data-free.
- **E-CHEM-036 Torsional-GFN**: VERIFIED.
- **E-CHEM-040 / -041 / -042 LoQI code**: VERIFIED. The snapshot matches, and the live fetch of `stereo.py` at 04f3caef reproduces lines 3–4 exactly. The LoQI *paper* claims are UNVERIFIABLE because the PDF is blocked; the author marked them UNVERIFIED, so no dispute.
- **E-CHEM-043 / -044 VonMisesNet** (p. 2, 16): VERIFIED.
- **E-CHEM-045 S23D, -046 ET-Flow, -048 Reflected diffusion**: VERIFIED.
- **E-CHEM-049**: title only, as the author states. Acceptable for a PRIOR ART pointer; not for mechanism detail.

### Our-data entries
- **E-JOINT-001**: values VERIFIED against `cluster_sync/round4/analysis_r4_leakfree/leakfree_summary.txt:15-19`. `ref_rdkitL` was re-scored on the same universe (`tools/analyze_leakfree.py:318-347`). The INFERENCE "geometry only helps when coupled" is **DOES NOT SUPPORT** (D-001). The split-vs-RDKit comparison is not budget-matched: the split view generates 2·|half| samples per molecule (`tools/make_split_half.py:19-22`), while the reference generates 2·n. The inference also conflicts with E-JOINT-010's own reading.
- **E-JOINT-002**: header counts VERIFIED (744 / 743 / 626; 12,628 / 7,516).
- **E-JOINT-004**: numbers VERIFIED by recompute (ETKDG 0.4961, MMFF 0.3506, λ0.50 0.1944, λ0.75 0.0002 on rings of 4+ atoms; mean 10.65°, median 0.44°, n = 7,263, 2,074 seeds > 10°). The "tail" inference **DOES NOT SUPPORT** (D-002).
- **E-JOINT-008**: VERIFIED against `round3/grounding.md:61-63` (377.7 s/epoch, 2,591 iters, 6.86 it/s). The 11.7 h includes validation and was not re-derivable locally.
- **E-JOINT-010**: **DOES NOT SUPPORT as stated** (D-003). My recompute of the published table reproduces 0.0323 / 0.1171 / 0.0137 exactly, so the transcription is right. The underlying column is biased by the self-inclusion bug.
- **E-JOINT-071**: not re-checked (round-3 VERIFIED).
- **E-CHEM-050**: VERIFIED. My independent replication of TD's rule gives 996 molecules, 2,880 rotatable bonds and 108 double bonds in 102 molecules (= 100 stereo + 2 non-stereo in the census).
- **E-CHEM-051** (DRUGS census): not re-run (sampled out).
- **E-CHEM-054**: values VERIFIED; the inference inherits D-001.
- **E-CHEM-060**: VERIFIED (`census_qm9_cluster.out`). The 108 count does not depend on the RDKit version.
- **E-CHEM-061**: VERIFIED. Recomputed from `ezflip_qm9.csv`: R0 65/526, R1 54/526, R2 55/526; any-wrong 0.421 / 0.500 / 0.500; A0 147/526.
- **E-CHEM-062**: VERIFIED from `ezflip_qm9.out` (CH-ok 1.000; A2 0.970).
- **E-CHEM-063**: VERIFIED. Recomputed mean twist 16.6° per molecule and 16.6° conformer-weighted (GT 1.8°); AMR-R 0.257 vs 0.166.
- **E-CHEM-064**: VERIFIED from `ezflip_qm9.out:10`.
- **E-CHEM-065**: inherits D-003.
- **E-CHEM-052, -053, -055, -056**: not re-checked (sampled out; they reuse round-4 files).

### Code entries
- **E-CHEM-022**: VERIFIED. `featurization.py:19` has bond-type one-hot only; `:63` builds `chiral_tag`, which is never used; `:101` returns `Data(x, edge_index, edge_attr, z)`. The atom features at `:65-82` (hybridisation, degree, rings) carry no chirality or E/Z.
- **E-CHEM-023**: VERIFIED (`sampling.py:105-108`; `generate_confs.py:150-151`).
- **E-JOINT-020, -021, -022, -023, -024, -026, -028**: VERIFIED. `likelihood.py` `dx_dtau` is at :68-74 and `log_det_jac` at :77-98; the J_τ rows already have translation and rotation removed by the inertia-tensor correction.

---

## Per-spec rulings

### EXP-1 — WEAKENED
Supported:
- The prior-art table. Every cited quote is real and on its page, and the PRIOR ART verdicts (RINGER, FoldingDiff, GO-Flow, PuckerFlow/TMCgen, CoarsenConf, EBD/HierDiff/OctMole/MSGEN, Torsional-GFN) are correct.
- The parity table (§2.4).
- Ring closure "by representation". This is true but trivial: Cartesian atoms cannot "open" a ring, and only the §2.6 bond check protects bond integrity.
- The need for the §2.8 chirality backtrack. P_L steps do flip stereocentres; I saw 1 flip in 20 random 0.3 Å RMS steps on `C[C@H](O)C1CCOC1`.

Not supported as written:
1. **§2.2 claim (1), §0 "the geometry factor cannot move torsions", and §1/§8 "exact separation from torsions"** (D-004). P_L = I − Bᵀ(BBᵀ)⁺B removes the torsion *tangent* directions. A dihedral changes at first order by ∇τ·δx, and ∇τ is not in the span of the tangents. Measured: P_L steps move the rotatable-bond dihedrals as much as unprojected steps do (ratio 1.00–1.08 at 1e-4, 0.05 and 0.3 Å RMS), and the first-order drift is 0.185–0.245 rad/Å. The orthogonal projector onto ker W (W = the m dihedral gradients, one quadruple per bond) has exactly zero first-order drift and the same dimension 3n−6−m.
   - In V3a's chart x = modify(x_R + d, τ), τ is a *chart* coordinate, so the construction is self-consistent there.
   - In V2 the torsion head reads real dihedrals, and every geometry step shifts them.
   - M0 measure (a) "torsion drift ≈ 0" will fail if it is measured as dihedral change.
2. **"Coupling is necessary" (novelty point 2)** (D-005, with D-001 and D-003). A6 feeds learned L into the frozen A0 torsion model, which was trained on RDKit L. By the spec's own E-JOINT-010 reasoning, A6 is expected to fail from L-distribution mismatch alone, whether or not coupling matters. A5 − A6 therefore cannot isolate coupling. A decoupled arm whose torsion head is DAgger-trained on the independent sampler's L is required: an "A6b", the S5 recipe applied to learned L.
3. **§2.2 σ_b rationale and the Gate-1 "ring tail"** (D-002). For QM9 molecules with a ring of 4+ atoms, ETKDG ring error is a bulk effect (median 9.8°), not a minority tail.
4. **§5.2 / §6 compute** (D-006). M1 is costed at about 1 GPU-h per A11 point (§2.2), 4–6 GPU-h (§5.2) and ≈ 12 GPU-h (§6). With 11.7 GPU-h per 100 epochs, 3 × 30 epochs is ≈ 10.5–12 GPU-h unless training is restricted to the rigid stratum, and the spec does not say how big that stratum is.

### EXP-4 — SUPPORTED (minor corrections)
- The chart x(τ, d, ω) = C(R(ω)·modify(x_A + B_A d, τ)) is well defined. Its dimension is m+k+3 = 3n−3, and at d = 0 it is locally bijective because B_A is orthogonal to the τ tangents and to the rigid directions.
- D-004 does not hurt here, because τ is the chart coordinate. The spec should still say so, so that nobody later reads τ as "the dihedral".
- The Jacobian, the q(x) formula and the k = 0 reduction to TD Prop. 3 are consistent with TD App. B.3 (quote VERIFIED).
- The novelty rows (Noé 2019, TBG, TD BG, BoltzNCE, CG-BG) are all VERIFIED. "NOVEL, narrowly" is fair.
- Correction (D-007): the variant-D cost is understated. 100 molecules × 32 samples × 3 T × 20 × (m+k+1) ≈ 0.9–1.6 × 10⁷ NFE. At the spec's own 2–3 × 10⁴ NFE/min that is ≈ 5–14 GPU-h, not 3–6.
- H2 depends on EXP-1 A5, so it inherits D-005's caveat. The H1-vs-H2 ESS contrast is a cleaner coupling test than A5 − A6 only if both share the same torsion factor, and the spec should state that they do.

### EXP-2 — SUPPORTED
- Novelty rows: B is PRIOR ART (TD Table 8, App. H item 3; B2). C is PRIOR ART for the loss (GeoMol OT, distributional diffusion). D is PRIOR ART for the mechanism (TD §3.6, Torsional-GFN, Adjoint Sampling, EnFlow). All are VERIFIED, and the INCREMENTAL rulings follow.
- The compute arithmetic is consistent (3 × 11.7 = 35; 20 epochs ≈ 2.3 GPU-h; total 280–415 GPU-h).
- The framing paragraph quotes the 0.032 Å floor (D-003: 0.036 on unbiased rows, with an oracle best-of-others selection) and the leak-free comparison (D-001). Neither changes the design. Both must be fixed in the text.

### EXP-3 — SUPPORTED
- Every QM9 census number recomputes exactly. The cited code lines do what is claimed.
- The novelty rows are VERIFIED: TD's special-edge suggestion, the LoQI code, VonMisesNet's canonical rotatable definition, the post hoc flips in GeoMol and ET-Flow, and reflected/constrained kernels.
- Corrections (D-008):
  - The §0 headline "12.4 %" is the R0 seed only; report "10–12 % (R0–R2; mean 11.1 %)", as the §2 table already does.
  - 108/2,880 = 3.75 %, so write "3.8 %" or "3.75 %".
  - The E/Z analysis covers the 38 molecules that RDKit 2022.09.5 perceives, out of 102 with a TD-rotatable double bond. State this coverage next to the headline.
- The DRUGS census was not re-run (sampled out).

---

# Part 2 (2026-10-16): EXP-5, EVAL.md, evidence_SAMPLE.md; closure of D-001 to D-008

## Summary (Part 2)

| Check | Result |
|---|---|
| Reproduction of the R-SAMPLE pipeline | I re-ran `curves.py`, `alloc.py`, `alloc2.py` (budgets 0.25, 0.5, 1.0) and `select.py` with `python -I` on the copied cluster pickles. `curves.pkl` is byte-identical to theirs, and `alloc.out`, `alloc2.out` and `select.out` are line-identical. Every quoted number reproduces: oracle 0.1400, TD 2K rule 0.1784, static 0.1442, uniform 0.1450, −0.0080 at 0.5x, −0.019 at 0.25x, Spearman 0.451 / 0.446 / 0.441 / 0.474 / 0.273. |
| Logic of the expected curves | Correct. They are hypergeometric expectations for a random n-subset of the 6K pool, and realised prefixes agree (0.1784 vs 0.1783). |
| "2K rule uses test-set K" | VERIFIED. `generate_confs.py:183-187` calls `sample_confs(raw_smi, 2 * n_confs, smi)` with `n_confs` = the `n_conformers` column of `test_smiles.csv` (GT count). TD PDF p. 8 confirms, and Zhou et al. use N_ref the same way (p. 4). |
| GT leakage into the "GT-free" signals | `signals.py` reads only generated conformers (`confs.pkl`) and MMFF94 single points, so the **signals are GT-free**. **But the K-dependent pool cap N_i = 6K_i leaks K into every allocator row**, including "uniform (no K)", static, two-stage and Good–Turing. All of them water-fill the budget that capped molecules cannot use into the uncapped ones. A strictly K-free uniform rule (cap kept, no redistribution, samples beyond the pool assumed worthless) gives **AMR-R 0.1617 using only 69 % of the budget**. 54 % of molecules are capped at n = 27, all with K ≤ 4. The direction (the 2K rule is a poor allocator) is robust; the size, and the "K-free 0.1450", are not. The static and two-stage rules are also cross-fitted on *test-set* GT curves of the other fold, so 0.1442 is an analysis number, not a deployable one. → D-101 |
| "Headroom ≤ 0.004 Å" | Inconsistent across documents. E-SAMPLE-022 correctly calls the capped oracle headroom a **lower bound**, but EXP-5 §0 and H1 ("at most ≈ 0.004 Å (oracle bound)") treat it as an upper bound. S0's planned fix (10 × 2K pools) is still K-dependent: K = 1 gives 20 samples, below the 27 per molecule of uniform at 1x. → D-101 |
| Selection result (2x, MMFF-25 % + torsion FPS) | Output count is **matched** (exactly 2K). Compute is **not matched**: it uses 2x sampling NFE plus MMFF single points. It is **GT-free**: MMFF energies and torsions come from generated conformers, and FPS uses an RNG start. "Random from 2x" equals TD and is the correct NFE control. The improvement on every metric at 2x is reproduced (0.1778 → 0.1699; COV-R@0.1 +2.7; AMR-P 0.2200 → 0.2119; COV-P 91.2 → 91.9). Caveats: (a) the variant (25 % vs 50 % filter, FPS vs energy) and c = 2 were picked by looking at the test set, with no CI and 3 pool orders; (b) EXP-5 §5 lists P2 (PG at c = 1) against P3 (c = 2) as "NFE matched", which it is not. → D-102 |
| Prior-art quotes (GAINS, AdaDiff, Seij et al., Ma et al., Gotta-Go-Fast, Koukos–Glykos, PG ×5, Zhou, TD, GeoMol, ET-Flow, S23D, GEOM, Zhang) | 37 quote lines: 26 matched automatically. The other 11 were found by hand on the claimed pages; they failed only because of `[δ]` substitutions, quotes spanning two pages, or Å glyphs. PDF identity (title and authors on page 1) was checked for the five new PDFs. **All VERIFIED.** OMEGA and CREST are documentation snapshots, not re-fetched; the CREST restart rule and the CREST paper are self-labelled UNVERIFIED and stay so. |
| EVAL.md failure handling | VERIFIED. TD: `evaluate_confs.py:81-84` handles model failures; `:152` and `:166-167` append `[0] * num_failures` to COV only, and AMR uses `nanmean`; `:47-54` is `clean_confs`; `:117-120` handles additional failures. ET-Flow: snapshot `covmat.py:136-137`, `:139-140`, `:153-155` all `continue`. A live fetch of `main` confirms that skipped molecules are excluded from the COV/MAT averages, not counted as 0. Timing anchor 14:18 wall (`R0_base_seed0_generate.log:2069`) VERIFIED. `generate_confs.py:153` (model skipped when n_rot = 0) VERIFIED. |

**Rulings:**
- **EXP-5: WEAKENED** (D-101, D-102). The negative/INCREMENTAL verdict is well sourced and probably right. But the kill-criterion number (0.0042 Å) is a cap-shaped lower bound presented as an upper bound, and the S0 design does not remove the K-dependence.
- **EVAL.md: SUPPORTED**, with one correction (D-101): §1 must not call the 0.1450 uniform allocation "K-free".

## Per-entry verdicts (SAMPLE, checked)
- **E-SAMPLE-003**: VERIFIED (TD p. 8, p. 17; code `generate_confs.py:187`).
- **E-SAMPLE-010 / -011 GAINS**: VERIFIED (p. 1; arXiv 2609.22867, Cao, Tang, Li, Zheng).
- **E-SAMPLE-012 AdaDiff**: VERIFIED (p. 1, arXiv 2309.17074v3).
- **E-SAMPLE-013 Ma et al.**: VERIFIED (p. 1).
- **E-SAMPLE-014 Gotta Go Fast**: VERIFIED.
- **E-SAMPLE-015 Seij et al.**: VERIFIED (p. 1 abstract; p. 4 Fig. 2 caption on GEOM-Drugs).
- **E-SAMPLE-030 to -034 Particle Guidance**: VERIFIED (p. 8 Table 1 and §6.2; p. 21 App. B, B.2).
- **E-SAMPLE-040 / -041 OMEGA, -042 CREST**: snapshot only, not re-fetched. The quotes are as the author states; partial items are self-labelled UNVERIFIED.
- **E-SAMPLE-043 Koukos & Glykos**: VERIFIED (p. 2).
- **E-SAMPLE-020, -021, -023, -024, -025**: VERIFIED (reproduced exactly). The inference in -023 ("at most ≈ 0.004–0.008 Å") **DOES NOT SUPPORT** as an upper bound (D-101).
- **E-SAMPLE-022**: VERIFIED, including its own cap caveat. Its "K-free" uniform label is wrong (D-101).
- **E-SAMPLE-053**: VERIFIED.
- **E-SAMPLE-062, -063, -072, -076, -079, -080, -081, -082, -086**: VERIFIED on the claimed pages (found by hand).
- **E-SAMPLE-065 / -066**: VERIFIED. ET-Flow confirmed live, not only from the snapshot.
- Others (-001, -002, -004, -005, -061, -064, -068, -070, -071, -073, -074, -075, -083, -084, -085): quotes matched automatically, VERIFIED. Entries -050 to -052, -069 and -087 to -091: not checked (sampled out).

## Closure of D-001 to D-008
I checked each response against the edited files:
- **D-001**: E-JOINT-001 and E-CHEM-054 are revised to descriptive. The ref_half re-score is in EXP-1 §3.6 and EXP-2 §4 and is assigned to the analysis driver. **CLOSED-VERIFIED** (the re-score itself is still to be run).
- **D-002**: the recompute matches mine exactly, and the σ_b rationale and gates are restated on the ≥ 4-ring population. **CLOSED-VERIFIED**.
- **D-003**: E-JOINT-010 / E-CHEM-065 are revised, EXP-1 §0 and EXP-2 §0 cite 0.036 vs 0.114 Å, and the bug is logged (EXP-1 §9, E-JOINT-031). **CLOSED-VERIFIED**. One nit in the JOINT response: "all 769 rows with n_GT ≤ 3 have floor_gtother ≤ 1.97e-5". Of those 769, 225 are n_GT = 1 rows with NaN; the biased rows number 544.
- **D-004**: EXP-1 §2.2 now uses P_W = I − Cᵀ(CCᵀ)⁺C with C = [W; J_rig]. This is correct: rigid motions lie in ker W, so rank C = m + 6 and the dimension is 3n−6−m. "Exact separation" is struck, and M0 (a) and the tests are re-declared. **CLOSED-VERIFIED**.
- **D-005**: A6b is in §4, the primary Holm family is A5/A8 − A6b, and coupling is ruled a HYPOTHESIS. **CLOSED-VERIFIED**.
- **D-006**: one figure: 10.5 GPU-h at TD speed, 16 GPU-h ceiling, smoke run first. **CLOSED-VERIFIED**.
- **D-007**: corrected to 4.8–13.6 / 48–136 GPU-h. The added shared-sample reweighting across temperatures is valid: q(x) has no T, so one sample set from q can be importance-weighted to each e^{−E/kT}, at 1.6–4.5 GPU-h. **CLOSED-VERIFIED**.
- **D-008**: items 1–4 are all edited (E-JOINT-009 label; "proposed"; EXP-3 §0 gives 12.4 / 10.3 / 10.5 %, mean 11.1 %, with 38/102 coverage; 3.75 %). **CLOSED-VERIFIED**.

**Updated Part-1 rulings after closure:**
- **EXP-1**: WEAKENED → **SUPPORTED as revised**. Coupling is now a pre-registered hypothesis with a valid control (A6b), and the projector is correct.
- **EXP-2, EXP-3, EXP-4**: SUPPORTED.
