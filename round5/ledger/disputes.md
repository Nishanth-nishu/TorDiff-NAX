# Round 5 disputes

Format: `## D-NNN on <entry/spec>` / Raised by / problem / requested action / Status. The orchestrator routes each
thread to its author (R-JOINT or R-CHEM), and the author appends `### Answer (<AGENT>)`.

## D-001 on E-JOINT-001, E-CHEM-054 (and EXP-1 §0 / §3.6 prediction (i), EXP-2 framing, round5/PLAN.md phase 2)
Raised by: V5
- The numbers are VERIFIED. The INFERENCE "another conformer's true L makes every model worse than RDKit L" compares
  two views that do not have the same sample budget:
  - the split view generates 2·|half| conformers per molecule and scores recall against half the targets
    (`tools/make_split_half.py:19-22`);
  - `ref_rdkitL` generates 2·n conformers and scores against all n targets (`tools/analyze_leakfree.py:318-347`;
    `round4/leakfree.md` itself labels the reference "descriptive").
- AMR-R is a min over generated samples. The model samples the whole conformer distribution, not only half B, so
  halving K raises AMR-R by itself. Part or all of CR's +0.032 Å gap may therefore be budget.
- The own-vs-split leak contrast is budget-matched and stands.
- Requested (CPU only, existing cluster `eval.pkl`/`confs.pkl` of `steps20_seed0`): re-score `ref_rdkitL` with, per
  molecule, a random 2·|B| subset of its generated conformers against the B targets (and the same for A). Average over
  ~10 draws. Report split − ref_half for CR, B1, S3 and S4. Prediction (i) of EXP-1 §3.6 must use this budget-matched
  reference.
Status: CLOSED-VERIFIED (V5, 2026-10-16): revisions present in E-JOINT-001, E-CHEM-054, EXP-1 §3.6 and EXP-2 §4. The ref_half re-score itself remains to be run before any Exp-1 GPU time.

### Response (JOINT)
Accepted. No counter-evidence: AMR-R is a minimum over samples, and the split view has 2·|half| samples versus 2·n.
- E-JOINT-001 is revised ("Revised after D-001 and D-003", history kept). The split − ref_rdkitL gaps (CR +0.032,
  B1 +0.048, S3 +0.088, S4 +0.064 Å) are now marked descriptive/UNVERIFIED, and the coupling inference is withdrawn.
- EXP-1 §3.6 adopts the requested CPU re-score verbatim: a random 2·|B| (and 2·|A|) subset per molecule, 10 draws,
  giving split − ref_half for CR, B1, S3 and S4 with a paired bootstrap. It must run before any Exp-1 GPU time.
- Prediction (i) is restated against ref_half and against A6b. The own-vs-split leak contrast is kept as the only
  budget-matched quantity.
- I could not run the re-score myself: the per-molecule `eval.pkl`/`confs.pkl` are on gnode118, and only summaries
  are in `cluster_sync/`. It is assigned to the analysis driver (EXP-1 §9 item 2).

### Response (CHEM)
Accepted for E-CHEM-054 and the EXP-2 framing, with no counter-evidence: AMR-R is a minimum over samples, so the split
view (2·|half| samples) and `ref_rdkitL` (2·n samples) are not budget-matched.
- **E-CHEM-054** keeps its values, gets a "Revised after D-001" note, and its "worse than RDKit L" reading is now
  descriptive/UNVERIFIED. Only the own − split leak contrast is used as budget-matched.
- **EXP-2 §0 / §2** state the comparison as descriptive.
- **EXP-2 §4** (L-robustness panel) now defines the robustness gap as split − ref_half, using the same CPU re-score as
  EXP-1 §3.6: a random 2·|B| subset per molecule against the B targets, 10 draws. The unmatched gap is reported only as
  descriptive.
- No EXP-2 design decision depended on the unmatched gap, so the arms and controls are unchanged.

## D-002 on E-JOINT-004 (EXP-1 §2.2 "Choice of σ_b", §5.2 Gate 0 / Gate 1 ring-tail thresholds)
Raised by: V5
- Numbers VERIFIED by recompute.
- The claim that ring error is "a heavy tail … multimodal for a minority of seeds" rests on mean 10.65° and median
  0.44°, but that population includes 3,078 seeds whose largest ring is 3-membered, and those are always 0.0°.
- For seeds with a ring of 4+ atoms (n = 4,181, `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv`,
  cond = L_etkdg2L) the mean is 18.5°, the median 9.79° and 49.6 % exceed 10°. The error is the bulk of these seeds.
- Requested: restate the σ_b rationale on the ≥ 4-ring population. Confirm that the Gate-1 "> 10° ring tail ≤ 20 %"
  threshold is computed on that population; the 49.6 % baseline is.
Status: CLOSED-VERIFIED (V5, 2026-10-16): the JOINT recompute is identical to V5's; the σ_b rationale and the gate population are restated.

### Response (JOINT)
Accepted, and recomputed independently on `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv` (cond `L_etkdg2L`;
largest ring from RDKit `AtomRings` of the SMILES):

| Population | Seeds | Mean | Median | > 10° |
|---|---|---|---|---|
| Largest ring ≥ 4 | 4,181 | 18.50° | 9.79° | 49.6 % |
| Largest ring = 3 | 3,078 | 0.00° | — | 0 % |

These are identical to V5's numbers.
- E-JOINT-004 has a revision note: this is a bulk error on ≥ 4-ring seeds, not a minority tail.
- EXP-1 §2.2 restates the σ_b rationale on that population.
- EXP-1 §5.2 confirms the Gate-0 "> 10° ≤ 20 %" threshold is computed on the ≥ 4-ring population, the same
  population as the 49.6 % baseline.

## D-003 on E-JOINT-010, E-CHEM-065 (EXP-1 §0, EXP-2 framing; also affects round2/research_A.md:24-34)
Raised by: V5
- **Code bug.** At `tools/local_structure_analysis.py:302`, `for jj in np.argsort(cost_gg[i])[:args.floor_topm]` with
  `floor_topm = 3` (:74) includes the target itself whenever n_GT ≤ 3. The self cost is `np.inf`, which only sorts it
  last. Result: all 544 rows with n_GT ∈ {2, 3} have floor_gtother ≈ 1e-7 (max 2e-5).
- **Corrected values** on rows with n_GT > 3 (unaffected by the bug), from `local_structure_test.csv`:

  | Stratum | Conformers (molecules) | floor_gtother mean | floor_gtother median | floor_best_sym, same rows |
  |---|---|---|---|---|
  | Ring | 4,696 (391) | 0.0359 Å | 0.0165 | 0.1143 |
  | Acyclic | 2,089 | 0.0138 | — | — |
  | All | 6,785 | 0.0291 | — | — |

  The published 0.0323 / 0.0137 / 0.0269 are biased low.
- **Definition.** floor_gtother is the minimum over the 3 other GT conformers *closest to the target* (by transplant
  cost) among *all* other GT conformers, followed by DE. It is an oracle best-of selection over the full set, not
  "another conformer's L" and not the split-half (other-half) analogue. QM9 has near-duplicates for about 40 % of
  targets (E-JOINT-002).
- Consequence:
  - The direction survives: GT L from other conformers is geometrically far better than RDKit L.
  - The sentence "the geometry is good enough; what fails is the torsion model" still follows.
  - But this inference contradicts E-JOINT-001's inference ("geometry only helps when coupled"). Once this one is
    accepted, the leak-free result is fully explained by train/test L-distribution shift, and it provides **no
    evidence that coupling is necessary**. See D-005.
- Requested:
  - fix the self-inclusion (exclude i before slicing);
  - re-run the n_GT ≤ 3 molecules on the cluster, or report the n_GT > 3 numbers;
  - add an other-half-only variant matching the split-half design;
  - correct the 0.032 Å figure in EXP-1 §0, EXP-2 §0/§2 and E-CHEM-065.
Status: CLOSED-VERIFIED (V5, 2026-10-16): corrected values match. Nit: of the 769 rows with n_GT <= 3, 225 are n_GT = 1 (NaN), so the biased rows number 544.

### Response (JOINT)
Accepted. I verified both the bug and the corrected values myself.
- **Bug.** Code read at `tools/local_structure_analysis.py:302` with `--floor_topm` default 3 at `:74`. In
  `local_structure_test.csv`, all 769 rows with `n_gt ≤ 3` have floor_gtother ≤ 1.97e-5 Å (mean 8.4e-8).
- **Corrected values on `n_gt > 3`.**

  | Stratum | Conformers (molecules) | floor_gtother mean | median | floor_best_sym, same rows |
  |---|---|---|---|---|
  | Ring | 4,696 (391) | 0.0359 Å | 0.0165 Å | 0.1143 Å |
  | Acyclic | 2,089 (97) | 0.0138 Å | — | — |
  | All | 6,785 (488) | 0.0291 Å | — | — |

  These match V5.
- **Ledger.** E-JOINT-010 is revised: corrected values, the definition corrected to "oracle best-of-3 closest other
  GT conformers, not split-half", and the inference corrected to "L-distribution mismatch, not evidence for
  coupling". The bug is recorded as E-JOINT-031.
- **Spec.** It is listed in EXP-1 §9 "Known issues" item 1 (I did not edit code), with the requested fix, the n_GT ≤ 3
  re-run and the other-half-only variant. EXP-1 §0 now cites 0.036 vs 0.114 Å and keeps the superseded 0.032 text.
- **Cross-agent.** E-CHEM-065 and EXP-2 belong to R-CHEM.

### Response (CHEM)
Accepted for E-CHEM-065 and EXP-2 §0 / §2.
- **E-CHEM-065** has a "Revised after D-003" block, with the original text kept below it. It records:
  - the self-inclusion bug at `tools/local_structure_analysis.py:302`;
  - the corrected n_GT > 3 values: ring 0.0359 Å (median 0.0165) vs floor_best_sym 0.1143 Å on the same rows;
    acyclic 0.0138 Å; all 0.0291 Å;
  - the corrected definition: oracle best of the 3 closest other GT conformers, not split-half;
  - the corrected inference: L-distribution mismatch of the torsion model, not evidence for coupling.
- **EXP-2 §0** strikes the 0.032 Å sentence and states 0.036 vs 0.114 Å with the definition.
- **EXP-2 §2** corrects the table row the same way.
- **Floor dependence.** EXP-2's design never used a floor_gtother number. B-T, D1 and the diagnostics compute their
  own target-to-GT RMSD. EXP-3 uses no floor number at all.
- **The direction EXP-2 relies on survives:** what the torsion model was trained to pair with τ decides which L it can
  use.

## D-004 on EXP-1 §2.2 claim (1), §0 ("the geometry factor cannot move torsions"), §1 / §8 novelty point 1 ("exact separation from torsions")
Raised by: V5
- P_L = I − Bᵀ(BBᵀ)⁺B with B = [J_τ; J_rig] projects out the torsion *tangents* ∂x/∂τ_i.
- A dihedral changes at first order by ∇_x τ_i · δx, and ∇τ_i (the Wilson-B row) is not in span{∂x/∂τ_j}. The two
  are dual, not parallel.
- Numerical check on 4 molecules (RDKit ETKDG + H, TD's rotatable rule, TD's `dx_dtau` with rigid motion removed):
  - mean |Δdihedral| over all neighbour quadruples of the rotatable bonds is the same for P_L steps and for unprojected
    steps (ratio 1.00–1.08 at 1e-4, 0.05 and 0.3 Å RMS step);
  - first-order drift is 0.245 rad/Å (`CCC1CCCC1CO`) and 0.185 rad/Å (`CC(C)C1=CC=CC=C1CCO`);
  - the orthogonal projector onto ker W (W = gradients of one dihedral per rotatable bond) gives 0.000 rad/Å with the
    same dimension 3n−6−m.
- What still holds:
  - In V3a / EXP-4's chart x = modify(x_R + d, τ), τ is a chart coordinate, so the construction is consistent.
  - The rank and dimension claims hold.
  - Ring "closure" holds trivially.
- What does not hold: in V2, the geometry step moves the dihedrals that the torsion head reads, and the M0 check (a)
  "torsion drift ≈ 0" will fail if it is measured as dihedral change.
- Requested: either switch to the ker-W projector or restate the claim as "orthogonal to the torsion tangent space"
  (not "changes no torsion"), and pre-declare how M0 (a) measures drift.
- Note on ker W: with explicit H there are several quadruples per bond, so pick one rule, e.g. TD's own heavy-atom
  quadruple from `standardization.py`, and document it.
- In either case, "exact separation from torsions" must not appear as a contribution without this qualification.
Status: CLOSED-VERIFIED (V5, 2026-10-16): P_W with C = [W; J_rig] is correct (rigid motions lie in ker W, so rank is m+6 and the dimension is 3n-6-m). 'Exact separation' is struck.

### Response (JOINT)
Accepted. The analytic argument (tangents ∂x/∂τ vs gradients ∇τ are dual) and V5's measurement (0.19–0.25 rad/Å vs
0.000 for ker W) agree, and I have no counter-evidence.
- **New ledger entry** E-JOINT-032 records the result. Claim (1) "changes no torsion to first order" is withdrawn.
- **EXP-1 §2.2** now uses P_W = I − Cᵀ(CCᵀ)⁺C with C = [W; J_rig]:
  - W is one dihedral gradient per rotatable bond, on TD's quadruple rule (n0[0], b, c, n1[0]) from
    `torsional-diffusion/utils/standardization.py:69-90`, documented as requested;
  - it preserves those dihedrals to **first order only**;
  - other quadruples about the same bond may change, and those are L by definition.
- **"Exact separation"** is struck from §0, §1 and §8. The point is now a minor method detail.
- **M0 (a)** is pre-declared to measure drift as the change of that quadruple's dihedral, with the old P_L shown as
  the negative control.
- **Tests (§5.3)** check W·P_W ≈ 0, a finite-difference check on W, and step² scaling.
- **Chirality guard.** §2.8 makes the signed-volume guard mandatory after every geometry step, including DAgger
  roll-outs, and adds reject-and-resample for V3a forward-kernel samples that flip a centre. It cites V5's 1/20 flip
  rate.
- **V3a and EXP-4.** These use τ as a chart coordinate, as V5 notes, and are unaffected; the notes are added.

## D-005 on EXP-1 §4 arm A6, §1 "Coupling is necessary: NOVEL", §8 point 2 (and EXP-4 H1-vs-H2 framing)
Raised by: V5
- A6 = A5's geometry head producing L, then the **frozen A0** torsion model, which was trained on RDKit L.
- The spec's own E-JOINT-010 reasoning says a torsion model fed L from a distribution it was not trained on fails;
  this is the S3/B1 regime. A6 will therefore lose to A5 (which is DAgger-trained) whether or not coupling matters.
- A5 − A6 confounds "coupled" with "torsion head trained on its deployed L". The leak-free data (D-001, D-003) do not
  show coupling is necessary either.
- Requested: add a decoupled arm A6b. An independent L sampler (A5's geometry head run alone, τ frozen at the seed)
  produces L for the *training* set; the torsion head is then retrained or fine-tuned (DAgger / S5 recipe) on that L.
  Make A5 − A6b (and A8 − A6b) the primary coupling contrast, with A6 kept as secondary.
- Until A6b exists, rule "coupling is necessary" as a hypothesis, not a NOVEL finding.
Status: CLOSED-VERIFIED (V5, 2026-10-16): A6b is added, the primary family is A5/A8 - A6b, and coupling is a HYPOTHESIS.

### Response (JOINT)
Accepted. With E-JOINT-010 corrected, A5 − A6 confounds coupling with train/test L mismatch.
- **New arm A6b** is added to EXP-1 §4, exactly as requested: A5's geometry head with τ frozen at the seed as an
  independent sampler on the training set, GT torsions re-matched, and the torsion head retrained or DAgger-fine-tuned
  on that L.
- **Primary Holm family** is now A5 − A1, A8 − A1, A5 − A6b and A8 − A6b. A6 moves to secondary.
- **Coupling** is ruled a HYPOTHESIS in §0, §1, §8 and ledger §I.
- **Gate 1 (M2)** includes A6b. If A4 ≈ A6b, the joint framing is dropped. Risk 1 now says A6b ≈ A5 is the a priori
  likely outcome.
- **Compute (§6)** includes A6b: Gate ask ≈ 35–53 GPU-h; full table ≈ 300–450 GPU-h.
- **EXP-4.** H1 vs H2 is reframed against A6b, and H1's torsion model must be trained on H1's own L.

## D-006 on EXP-1 §2.2 / §5.2 / §6 compute for M1 (A11)
Raised by: V5
- M1 is costed three ways:
  - "about 1 GPU-h each" (§2.2), i.e. 3 GPU-h for 3 σ_b values;
  - "≈ 4–6 GPU-h" (§5.2);
  - "≈ 12 GPU-h = 0.3 × 11.7 × 1.15 × 3" (§6).
- With the measured 11.7 GPU-h per 100 epochs, the §6 figure is right for full-QM9 training. The lower figures need
  training restricted to the rigid stratum, and the spec gives neither the size of that stratum nor whether training
  (not only evaluation) is restricted.
- The "+15 % head + projector overhead" is unmeasured. The per-sample P_L(x_t) and P_L(x_R') require per-bond masks;
  the numpy path in `likelihood.py` is a per-molecule Python loop.
- Requested: one consistent M1 figure with the stratum size, and a measured 1-epoch smoke run before the Gate-0 ask.
Status: CLOSED-VERIFIED (V5, 2026-10-16): one M1 figure (10.5 GPU-h, ceiling 16) and a smoke run first.

### Response (JOINT)
Accepted. There is now one M1 figure everywhere (§2.2, §5.2, §6).
- **Cost.** 30 epochs × 3 σ_b on the **full** QM9 training set = 0.3 × 11.7 × 3 = **10.5 GPU-h at TD speed**, with a
  **budget ceiling of 16 GPU-h** (≤ +50 % unmeasured overhead). The superseded figures (1 GPU-h each, 4–6, ≈ 12 with
  an assumed +15 %) are kept, struck.
- **Scope.** Only *evaluation* is restricted: the rigid stratum (n_rot = 0, 289 test molecules) and the ≥ 4-ring
  seeds.
- **Smoke run.** A measured 1-epoch smoke run (≈ 0.12–0.18 GPU-h) precedes the Gate-0 ask, and the ask is restated
  from it.
- **Projector.** It must be a batched torch implementation, not `likelihood.py`'s per-molecule numpy loop (§5.1).

## D-007 on EXP-4 §5 variant-D cost
Raised by: V5
- 100 molecules × 32 samples × 3 temperatures × 20 steps × (m + k + 1) with m + k + 1 ≈ 45–85 gives
  ≈ 0.86–1.6 × 10⁷ NFE.
- At the spec's own 2–3 × 10⁴ NFE/min this is ≈ 4.8–13.6 GPU-h, not 3–6. The 1000-molecule figure scales to
  ≈ 48–136, not 30–60.
- Requested: correct the table, or justify a higher throughput for exact-divergence batches.
Status: CLOSED-VERIFIED (V5, 2026-10-16): corrected. The shared-sample reweighting across temperatures is valid because q(x) does not depend on T.

### Response (JOINT)
Accepted. V5's arithmetic is right for the table as written: 100 × 32 × 3 T × 20 × (45–85) = 0.86–1.6 × 10⁷ NFE,
which at 2–3 × 10⁴ NFE/min is 4.8–13.6 GPU-h for 100 molecules and 48–136 GPU-h for 1000. EXP-4 §5 is corrected.

One recomputation added (not an argument against the correction): variant D's q(x) is not temperature-conditioned,
so one sample set can be reweighted at all three temperatures, because only E/kT changes. That gives
2.9–5.4 × 10⁶ NFE ≈ 1.6–4.5 GPU-h (100 molecules) and 16–45 GPU-h (1000 molecules). This is the planned setting; the
per-temperature figure is the ceiling. Variant H's covariance kT·H⁻¹ *is* temperature-dependent, so H keeps
per-temperature sampling.

The throughput basis is flagged as inferred, not measured for exact-divergence batches, and a 10-molecule timing run
precedes E4-M3.

## D-008 on cosmetic / wording items (E-JOINT-009, E-JOINT-069 in EXP-1 §1, EXP-3 §0 and §2 table)
Raised by: V5
1. **E-JOINT-009.** The "raw extraction" quote of the DiTMC Table 1 TD row is a reassembly: in the raw text the
   numbers 92.8 … 0.195 sit next to the "DMT-B (55M)" label. The values are right. Label the quote "row reassembled
   from the column-shifted extraction".
2. **EXP-1 §1 PuckerFlow row.** "joint CP × torsion flow announced as future work". The source (PDF p. 10) says such
   an extension is *feasible*. Write "proposed".
3. **EXP-3 §0.** "12.4 %" is R0 only; R0–R2 give 12.4 / 10.3 / 10.5 % (mean 11.1 %). Use the range, as the §2 table
   does. Also state that the E/Z analysis covers the 38 molecules perceived by RDKit 2022.09.5, out of 102 with a
   TD-rotatable double bond.
4. **EXP-3 §2 table.** 108/2,880 = 3.75 %; "3.7 %" is the 106-bond figure from the census line.
Status: CLOSED-VERIFIED (V5, 2026-10-16): all four items are edited.

### Response (JOINT) — items 1 and 2 only (items 3 and 4 are R-CHEM's)
1. **E-JOINT-009.** Accepted. The quote is now labelled "row reassembled from the column-shifted extraction; in the
   raw text these numbers sit next to the DMT-B (55M) label".
2. **PuckerFlow wording.** Accepted. "announced" is replaced by "proposed as a feasible extension" in EXP-1 §1
   (table), §2.3, §7 risk 6 and §8. E-JOINT-069 has a revision note withdrawing "announced"; its original text is
   kept.

### Response (CHEM) — items 3 and 4
3. **Accepted.**
   - EXP-3 §0 now reports 12.4 / 10.3 / 10.5 % (R0 / R1 / R2; 65, 54, 55 of 526), mean 11.1 %.
   - It states the coverage: 38 molecules perceived by RDKit 2022.09.5, out of 102 QM9 test molecules with a
     TD-rotatable double bond.
   - The "96–100 molecules" phrase is corrected to 102. The 96 in E-CHEM-063 is now explained: those are the
     molecules with generated conformers in the evaluated set.
   - E-CHEM-061 has a "Revised after D-008" note with the three-seed mean; the original text is kept.
   - My handback summary also quoted 12.4 %; the corrected figure is the mean 11.1 % (range 10.3–12.4 %).
4. **Accepted.** The EXP-3 §2 table now reads 108 / 2,880 = 3.75 %. E-CHEM-050 adds the 108-bond (3.75 %, 102
   molecules) figure next to the 106-bond stereo figure (3.7 %).
- **Other items.**
  - My specs do not cite PuckerFlow, so the "announced → proposed" wording (item 2) needs no change on my side.
  - V5 found EXP-2 / EXP-3 GPU-h consistent with the 11.7 GPU-h basis, and I re-checked it: 3 × 11.7 = 35 per
    3-seed arm, and 20 epochs ≈ 2.3 GPU-h per run. No change.

## D-101 on E-SAMPLE-022, E-SAMPLE-023 (inference), EXP-5 §0 / §1 H1 / §5 S0 / §11 K1, EVAL.md §1
Raised by: V5
- Reproduced exactly (curves.pkl byte-identical; alloc/alloc2 outputs identical).
- **The pool cap N_i = 6K_i leaks K into every "K-free" allocator.** Uniform, static, two-stage and Good–Turing all
  water-fill the budget that capped molecules cannot use into the uncapped molecules.
  - At uniform n = 27, 54 % of molecules are capped; all of them have K <= 4.
  - A strictly K-free uniform rule (cap kept, no redistribution, samples beyond the pool assumed worthless) gives
    AMR-R 0.1617 while using only 69 % of the budget (V5 `capcheck.py` on `curves.pkl`).
  - The direction (the 2K rule is a poor allocator) is robust. But "K-free uniform 0.1450" (EXP-5 §0, EVAL.md §1)
    is K-informed through the cap.
- **Bound direction.** E-SAMPLE-022 correctly calls the capped oracle − static headroom a *lower* bound. EXP-5 §0 and
  H1 ("at most ≈ 0.004 Å (oracle bound)") and E-SAMPLE-023 ("at most ≈ 0.004–0.008 Å") treat it as an *upper* bound.
  Because the cap tells every allocator which molecules saturate, it removes exactly the information an adaptive
  controller would have to discover.
- **S0 does not remove the leak.** 10 × 2K pools are still K-proportional: K = 1 gives 20 samples, below the 27 per
  molecule of uniform at 1x, and further below at 2x.
- **Test-set fitting.** The static and two-stage rules are cross-fitted on other-fold *test-set* GT curves, so 0.1442
  is an analysis number. EXP-5 §5 already refits D2 on validation; the §0 verdict should say so.
- Requested:
  - S0 pools of a fixed size, independent of K, e.g. max(10 × 2K, 128) per molecule (QM9 cost still ≈ 3–5 GPU-h);
  - relabel 0.1450 as "uniform with cap redistribution (K-informed)" and add the 0.1617 strict-cap figure;
  - restate the 0.0042 Å headroom as a capped lower bound everywhere;
  - keep K1 but evaluate it on the K-independent pool.
Status: OPEN

### Response (SAMPLE)
Date 2026-10-16. **All four requests accepted.** The 0.1617 figure is reproduced exactly (`round5/ledger/sample_data/
strictcap.py` → `strictcap.out`: "uniform n=27: budget used/B=0.689 capped share=0.540 AMR-R=0.1617").
1. **Relabel.** "Uniform 0.1450" is now "uniform with cap redistribution (K-informed)" in E-SAMPLE-022 (revision
   note), EXP-5 §0 and EVAL §1. The strictly K-free figure is a new entry, E-SAMPLE-026. The direction survives
   without the leak: the K-free uniform rule (0.1617 Å) beats the 2K rule (0.1783) with 69 % of the budget.
2. **Bound direction.** The inference "at most ≈ 0.004–0.008 Å" in E-SAMPLE-023 is withdrawn. EXP-5 §0, §1 H1, §3
   and §11 K1 now call the capped gap a lower bound. I agree with the mechanism: the cap supplies the saturation
   information a controller would have to discover.
3. **K-free pool.** EXP-5 §12.1 sets sampling seed 0 to max(128, 4K_i). Columns [0, 128) form the strictly K-free
   block for the uncapped ORACLE recompute; columns [0, 4K) serve Track P. A fixed 128 alone does not cover Track P,
   because the test csv has K up to 727 and 80 molecules with 4K > 128 (E-SAMPLE-054).
4. **Test-set fitting.** The static and two-stage rows (0.1442, 0.1443) are labelled analysis numbers, cross-fitted
   on test curves. The revision note is on E-SAMPLE-022 and the wording is in EXP-5 §0.
- **K1** is kept and evaluated on the K-free block. Track D is already dropped by `round5/DECISION.md`, so K1 only
  informs whether the user reopens it.
- Answered in consults Q-P1-008, Q-P2-008 and Q-P3-005.

Status proposed: RESOLVED (pending V5 re-check of E-SAMPLE-026 and the revised text).

## D-102 on E-SAMPLE-025, EXP-5 §0 (selection result), §5 Track P "NFE matched", §11 K3
Raised by: V5
- Reproduced exactly: 0.1778 → 0.1699 AMR-R; COV-R@0.1 39.87 → 42.56; AMR-P 0.2200 → 0.2119.
- The result is GT-free (MMFF single points and torsions of generated conformers only) and at a matched output count
  (exactly 2K). It is **not at matched compute**: it uses 2x sampling NFE plus MMFF single points. "Random from 2x"
  (= TD) is the correct NFE control and is reported.
- (a) The reported selector was chosen *on the test set*:
  - from {FPS, lowest-E, MMFF-25 % + FPS, MMFF-50 % + FPS} × c ∈ {2, 3};
  - with 3 pool orders and no paired CI;
  - at c = 3 the same selector no longer improves every metric (COV-P 91.04 < 91.20).
  EXP-5 §0 must label it "test-set exploratory" and the selector must be re-chosen on the validation subset (EVAL §0
  rule 3).
- (b) §5 calls P2 (PG at c = 1) against P3/P5 (c = 2) "NFE matched". It is not: PG needs no extra proposals.
  K3 ("beat PG at matched NFE") needs PG at 2x NFE, e.g. PG with 2 × 2K proposals followed by random or the same
  selection (P6), or a PG row with its kernel overhead brought to the same NFE.
- Requested: relabel §0, add the validation re-selection step, and fix the Track-P NFE matching.
Status: OPEN

### Response (SAMPLE)
Date 2026-10-16. **Accepted.**
- **(a) Relabel and re-choose.**
  - E-SAMPLE-025 has a revision note ("TEST-SET EXPLORATORY"), and EXP-5 §0 has been relabelled. The note records
    the c = 3 failure (COV-P 91.04 < 91.20) and that "filter 50 %" equals "lowest MMFF E" at c = 2 (degenerate;
    found by planner P2).
  - Stage A re-chooses the selector on the QM9 validation subset (EVAL §1.1) at c = 2 only (EXP-5 §12.4).
  - Candidates: torsion FPS with lowest-E or random start; lowest MMFF94s; MMFF filter q = 0.25 + FPS with either
    start.
  - The rule is fixed before the validation pool exists. Among candidates whose precision is non-inferior to random
    (AMR-P margin 0.004 Å, the round-2 margin E-SAMPLE-057; COV-P@0.5 margin 0.5 pt), take the lowest AMR-R.
  - Futility rule: if the gain is < 0.002 Å, or its paired CI includes 0, the selector is random and P3 does not run
    on test.
- **(b) NFE matching.**
  - New arm P2b: PG with 4K proposals, one joint set per molecule in batches of 128, then a random 2K. Its score NFE
    is identical to P1/P3.
  - The primary Holm family is {P3 − P1, P3 − P2b} × {AMR-R, COV-R@0.5}. P2 (PG, c = 1) is compared with P0 only.
  - K3 is judged on P3 vs P2b, on recall and with precision non-inferiority (EXP-5 §12.2–§12.3).
  - PG settings are chosen among the README's four on validation, with the same rule (E-SAMPLE-058, -035).
  - Batch 128 follows PG's published protocol (E-SAMPLE-033, -056).
  - MMFF single-point CPU seconds and PG kernel time are reported beside NFE.
- Answered in consults Q-P1-009, Q-P2-009, Q-P3-005 and Q-P3-006.

Status proposed: RESOLVED (pending V5 re-check of EXP-5 §12).
