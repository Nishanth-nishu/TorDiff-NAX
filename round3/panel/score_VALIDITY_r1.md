# Scorecard: JUDGE VALIDITY, round 1 (P4)

Written 2026-10-11. Lens: validity (confounds, controls, single-factor changes, oracle and test-set leakage, statistical
power, pre-declared endpoints, population/coverage, and whether a card's conclusion follows from its result either
way). Scored independently: I did not open the other judges' files.

Inputs: all 17 cards in `round3/cards/` (sections 1–8; the latest "Revised after cross-read" blocks are treated as
current), `round3/grounding.md`, `round3/impact.md` (+ `impact_data/ntrue_leak.out`, `noise_power.out`), and the final
"Round 2 verdicts" in `ledger/verify_V1.md`, `verify_V2.md`, `verify_V3.md`, plus `ledger/disputes.md`. Evidence status
used: every E-EVAL (59), E-ROBUST (61) and E-GEOM entry is VERIFIED after round 2. The one exception is the split
statement in E-GEOM-054, which is contested and still open (D-210).

## 1. Summary table

Scores 1–5: Ev = evidence, Rel = relevance, Gain = expected gain vs noise, Cost (5 = cheap), Risk (5 = low),
Val = FlexiTors value. For CPU analysis and diagnostic cards, "Gain" scores whether the measurement will resolve its
question above its own noise, because their effect on AMR-R is zero by construction.

| Card | Ev | Rel | Gain | Cost | Risk | Val | Total /30 | Worth trying | Validity blocker |
|---|---|---|---|---|---|---|---|---|---|
| C-GEOM-01 xTB seeds | 4 | 5 | 3 | 4 | 4 | 5 | **25** | **YES** (once the blocker text is fixed) | **BLOCKER (text-level):** §5 step 3 picks the reported xTB variant by a test-GT gate. Fix: run both variants or pre-declare one |
| C-EVAL-01 lit. block + "how good must L be" | 5 | 4 | 3 | 5 | 3 | 4 | **24** | **YES** | NONE (conditions in §3) |
| C-EVAL-04 non-learned baselines | 4 | 4 | 4 | 4 | 3 | 4 | **23** | **YES** (B-mmff, B-xtb); MAYBE (B-clust) | B-clust / TD+cluster only: K-means feature space and the matched sampling budget are not pre-registered |
| C-EVAL-05 xTB-after yardstick | 4 | 3 | 3 | 5 | 4 | 4 | **23** | **YES** (yardstick only) | NONE (conditions in §3) |
| C-EVAL-02 diagnostics | 4 | 4 | 3 | 4 | 3 | 4 | **22** | **YES** | NONE (strongly recommend the split-half other-conformer design in place of the P0 inversion) |
| C-EVAL-03 xTB E_relax | 4 | 3 | 3 | 4 | 4 | 4 | **22** | **YES** | NONE (S-1 / S-2 are correctness preconditions) |
| C-GEOM-02 ring templates | 3 | 5 | 3 | 3 | 3 | 4 | **21** | **YES** (after C-GEOM-01, once the blocker text is fixed) | **BLOCKER (text-level):** the "best template source" and the key level are chosen on test GT, because no validation path exists. Fix: pre-declare them (key by training-split leave-one-out) or run all, or build S-5 |
| C-GEOM-06 srETKDGv3 seeds | 3 | 4 | 2 | 5 | 3 | 3 | **20** | **MAYBE** (CPU part YES; GPU only by the pre-declared tail rule) | GPU step only: "CTRL on the winning variant" is selection on test GT. Fix: pre-declare the single-factor contrasts |
| C-GEOM-03 ET-Flow L | 4 | 4 | 3 | 3 | 1 | 3 | **18** | **MAYBE, blocked** (NO unless D-210 is resolved conclusively) | **BLOCKER:** D-210 open. Possible test-set leakage into `qm9-o3` would inflate exactly the quantity measured. Fix: a SMILES-level zero-overlap check of `QM9.zip` train+val vs `test_smiles.csv`; anything else means STOP |
| C-ROBUST-03 S3-MMFF | 4 | 3 | 2 | 2 | 3 | 3 | **17** | **NO** (MAYBE only if the measured S5 gate passes) | If run: 2 seeds cannot resolve the expected 0 to −0.006; the "GT L within 0.005 of S3" endpoint is own-conformer ORACLE. Fix: 3 seeds; non-oracle primary only |
| C-GEOM-04 learned pucker model | 3 | 4 | 1 | 1 | 2 | 3 | **14** | **NO** (this round) | None for this round. Reopen criterion 2 cannot be met while C-GEOM-03 is blocked; restate it |
| C-GEOM-05 ML-potential relaxation | 3 | 3 | 1 | 3 | 2 | 2 | **14** | **NO** | None (not proposed). If overruled: the 100 "validation" molecules need S-5 |
| C-ROBUST-01 factorised aug. | 2 | 3 | 1 | 2 | 2 | 3 | **13** | Arm 0 **MAYBE** (only with a leak-free partial-L condition); Arm 1 **NO** as written | **BLOCKER:** every partial-L condition (A5ring/A5acyc/λ0.75 `_cyc_ORACLE`) carries own-conformer identity, and the non-oracle primary (non-inferiority on RDKit/MMFF) cannot detect the hypothesised benefit, so no outcome supports the card's conclusion. Fix: non-own partial-L test conditions; Arm 1 ≥ 3 seeds |
| C-ROBUST-02 blind λ | 3 | 2 | 1 | 2 | 3 | 2 | **13** | **NO** | If run: equivalence (±0.005) with 2 seeds is underpowered (needs 3–6); S4's "λ picked on validation" has no code path; mid-λ conditions are leaky. Fix: ≥ 4 seeds, pre-declared λ, measured gate first |
| C-EVAL-06 DFT re-ref / ensemble props | 4 | 1 | 1 | 2 | 2 | 2 | **12** | **NO** | If run: (a) replaces the reference, which voids every oracle and literature comparison; (b) `--likelihood` drops 0-rotor molecules (population change) and rests on CREST weights |
| C-ROBUST-04 L guidance / score interp. | 3 | 2 | 1 | 3 | 2 | 1 | **12** | **NO** | If run: the per-source w has no validation path, so it would be tuned on test; the λ 0.5 condition is leaky ORACLE |
| C-ROBUST-05 p_gt sweep / curriculum | 3 | 2 | 1 | 1 | 3 | 2 | **12** | **NO** | If run: Δ ≤ 0.003 needs ≥ 4 seeds per arm; the true-L endpoint measures conformer identity, not L use |

Blockers that stop a card in this round: **C-GEOM-01** and **C-GEOM-02** (text-level; one sentence each fixes them),
**C-GEOM-03** (D-210), and **C-ROBUST-01** (design). C-GEOM-06's blocker affects only its optional GPU step. The "if run"
lines on NO cards record what would have to change if the user overrules.

## 2. Cross-cutting validity rulings (apply to every card)

**V-1. ORACLE-scored gates.** The CPU gates of C-GEOM-01/02/03/06 and the reopen rule of C-GEOM-04 score seeds against
test-set GT (L error, > 10° tail, pucker coverage, seed AMR-R). By README rule 5 that is ORACLE data, even when both
compared sources are non-oracle. My ruling on the D-205 / V2 / V3 referrals:
- A gate may decide **whether** a GPU step runs (experiment selection) only if (a) it is pre-declared, (b) its metrics
  and outcome are reported whatever they are, and (c) the GPU-stage contrasts are fixed before the gate is computed.
- A gate may **never** decide **which variant** becomes the reported method ("the `_mmff_xtb` variant if it wins",
  "the best template source", "the winning variant", "λ chosen on validation" when no validation set exists). That is
  model selection on test GT.
- The clean general fix is the validation-set path S-5 (~70 lines, grounding §2.3): build the seed sets for the
  `split.npy[1]` molecules and run every gate, variant choice and key-level choice there. Short of that, pre-declare
  the variant or run all variants and report all.

**V-2. Own-conformer leak (impact §3g; grounding 5.2.1).** Every `_cyc_ORACLE` condition (λ sweep, A5ring, A5acyc,
noise, gtLcycle), and round-1 `gtL` for about 92 % of targets, puts each target conformer's own L among the seeds.
- B1-type behaviour on these conditions measures how well a model uses conformer identity, not how well it uses
  reachable L.
- Any card endpoint on these conditions is therefore ORACLE, with no decision weight for "the model uses good L". This
  affects C-ROBUST-01 (A5 hybrids, GT ≤ 0.04), C-ROBUST-02 (mid-λ, true L), C-ROBUST-03 (GT within 0.005 of S3) and
  C-ROBUST-05 (true-L endpoint).
- The A5ring "bound" cited by C-GEOM-02 is not a bound for template L either: it is own-conformer rings on a
  different seed base and a different population (V3 trivial note).

**V-3. The P0 leak inversion is an estimate, not a measurement.** `ntrue_leak.out` divides gtL − gtLcycle by
P0 ≈ 0.07–0.13. Two consequences:
- Noise is amplified 8–14× (impact §3g). CTRL's per-bin differences (+0.0070, −0.0003, +0.0037, +0.0017) sit within
  noise.
- The identification assumption is unchecked. The inversion attributes the whole gap to "no own-L draw", but gtL also
  differs from gtLcycle in copy multiplicity (Binomial vs exactly 2 copies per conformer). A multiplicity effect of
  only 0.002 Å moves A_other by 0.015–0.03 Å, in an unknown direction.

The qualitative B1 reading (B1's true-L advantage is mostly conformer identity) is consistent across bins and I accept
it as a working hypothesis. The leak-free floors (CTRL 0.111–0.145, B1 0.19–0.27) are not measurements.

**Direct, leak-free design (recommended; ~50 lines; same GPU cost as the random-GT runs).** For molecules with
n_true ≥ 2:
1. Split the GT conformers at random into halves A and B.
2. Build the seed pickle from A's L only, cycled.
3. Score recall twice from one run's RMSD matrix: AMR-R over B's targets (own L absent: A_other) and over A's targets
   (own L present: A_own).

This gives a within-molecule, paired leak estimate with no division by P0. It does need a small evaluator change: a
target subset taken from the `eval.pkl` matrix rows. The existing `floor_gtother` in
`tools/local_structure_analysis.py:14` is the model-free precedent. I recommend it for C-EVAL-02 (d), C-EVAL-01 and
the C-ROBUST-01 fix.

**V-4. Seed variance belongs in every model contrast.** `noise_power.out` CIs resample molecules after averaging
replicates, which treats the checkpoints as fixed.
- B1 is unstable across seeds on off-distribution L: SD 0.014 on noise 0.04 and 0.022 on noise 0.02, against 0.003 on
  RDKit L.
- xTB or template L is off-distribution for B1. "B1 − CTRL CI ≈ 0.006" (C-GEOM-01 §8) may therefore be 2–4× too narrow
  on xTB L.
- The S-6 analysis driver must bootstrap over seeds and molecules (two-level) for model contrasts and report per-seed
  values. S3 has only 2 seeds.

**V-5. Pre-declaration.**
- Primary: AMR-R mean and COV-R at δ = 0.5 Å. Key secondary: COV-R@0.1 on the success intersection (user ruling).
- Fine-threshold rows (0.05 Å, micro AMR, strata) are exploratory and may not be promoted after results are seen.
- The S-6 driver, with a named primary contrast per card and a fixed order for the rest (Holm, or a hierarchy), must
  be committed before any round-3 result exists, as round 2 did.
- C-GEOM-01 alone has ≥ 8 candidate contrasts: 4 model families × 2 xTB variants, plus S4 at two λ.

**V-6. Populations.** Every contrast must be on the common success intersection of the arms compared:
- 936 molecules for the etkdg family, including xTB and template seeds (the fallback keeps key sets identical);
- 955 for λ0/λ1/A5; 906 for λ0.25–0.75 (648 without replaced slots); 996 for noise and round-1 gtL;
- S4's pair_ok subset (`S4_test_subset.txt`) for every S4 contrast.

ET-Flow (C-GEOM-03) can produce seeds for some of the 60 cages ETKDG cannot embed. Its comparisons with TD-family arms
must be restricted to the intersection, with its extra molecules reported separately.

**V-7. Correctness preconditions (not card blockers, because the cards adopt them in §7).** Before any xTB number
counts:
- S-1.1: TD's `utils/xtb.py` shares one tmp dir across Pool workers, which silently swaps coordinates between
  molecules with equal atom counts;
- `--norestart`, a convergence flag and a charge;
- the S-2 graph/stereo filter (the evaluator never filters generated conformers, `evaluate_confs.py:86`);
- a fallback share reported, with gating only if ≥ 95 % of seeds relaxed;
- the GT pass-through zero check.

**V-8. Model contribution vs seed contribution.** On reference-level L, a true-L-trained model can reconstruct the
seed's own conformer: TD re-randomises torsions, but L encodes the seed's torsional basin. Any B1/S3/S4 arm on xTB,
template or ET-Flow L must therefore be compared with the same seeds run with `--no_model --no_random` on the same
molecules: B-xtb (C-EVAL-04), the S-3 seed run, and raw ET-Flow. Otherwise a gain cannot be attributed to the torsion
model.

## 3. Per-card scorecards

### C-GEOM-01: GFN2-xTB-relaxed ETKDG seeds as test-time L
Scores: Evidence 4, Relevance 5, Expected gain 3, Cost 4, Risk 4, FlexiTors value 5. **Total 25.**
**Worth trying: YES**, once the blocker text below is fixed.

**Reason.**
- **Mechanism.** The mechanism rests on VERIFIED entries: GEOM-QM9 references are gas-phase GFN2-xTB minima
  [E-GEOM-015, E-GEOM-016]; GFN2 re-optimisation of references is ≈ 0, and MMFF↔GFN2 offsets have the size of our MMFF
  residual [E-GEOM-018, E-GEOM-019]. The QM9 transfer is an inference (Drugs data; §4). The grounder's probe partly
  closes it: 10/10 GT conformers stay put under xtb 6.7.1 (grounding §2.2).
- **Design: a clean single-factor contrast.**
  - `L_etkdg2L_xtb` and `L_etkdg2L_mmff_xtb` are common-random-number twins of the round-2 ETKDG/MMFF seeds (assert
    bit-identical `L_etkdg2L`, §7).
  - They share the 936-molecule population (the 60 ETKDG failures have no seeds in any etkdg-family pickle).
  - The L|τ coupling is the same as in the MMFF arm (§7 transfer check).
  - The checkpoints are unchanged, so CTRL-xTB vs CTRL-MMFF isolates the L source.
- **Power (impact §2, §8).**
  - The CTRL contrast is borderline: central −0.006 to −0.008 against an MDE of ≈ 0.007 for an L-source contrast.
  - The decision does not hinge on that contrast alone. The tail share, pucker coverage and seeds' COV-R (s) are
    measured directly on ≈ 25k seeds, so "force-field offset vs wrong basin" is answered either way. The conclusion
    follows from either outcome.
- **The leak finding raises the card's value (§8 revised).** B1/S3 on xTB L test whether a non-oracle source can supply
  own-basin L at all. That is legitimate non-oracle own L, and it is the decisive unknown for FlexiTors.
- **Weak points.**
  - Branch probabilities in §8 rest on the P0 inversion (V-3).
  - B1's seed variance on off-distribution L may make B1 − CTRL wider than the quoted ±0.006 (V-4).
  - S3 has 2 seeds.
  - Optional arm C (CREST on test SMILES) is the generator of the references. It must never appear as a method, only
    as a labelled reference-generator ceiling; it is not costed this round anyway.

**Validity blocker: YES (text-level, cheap).** §5 step 3 says to run CTRL/B1 "on `L_etkdg2L_xtb` (or the `_mmff_xtb`
variant if it wins the gate)". The gate is scored against test GT, so this selects the reported method on ORACLE data
(V-1, README rule 5). The bias is small, but it lands on a borderline primary contrast.

Fix, either:
- run both variants (+6 packed runs ≈ 1 GPU-h) and report both, with `_mmff_xtb` pre-declared as primary; or
- pre-declare one variant now.

**Conditions (pre-registration in S-6, not blockers).**
1. Primary contrast: P1 = CTRL-xTB − CTRL-MMFF (δ = 0.5 AMR-R). Then a fixed order: P2 = S3-xTB − CTRL-xTB, P3 =
   B1-xTB − CTRL-xTB. S4 at λ = 0 (round-2 rule) and λ = 1, reported separately and never "best of".
2. Model contribution: compare each arm with the no-model xTB seeds on the same molecules (V-8).
3. Two-level bootstrap and per-seed values (V-4).
4. Fallback share ≥ 95 % and the S-1/S-2 fixes (V-7).
5. Run the GPU step regardless of the gate, as impact §4.3 recommends. That removes the forking path entirely.

### C-GEOM-02: training-set ring templates (OMEGA-style pucker library)
Scores: Evidence 3, Relevance 5, Expected gain 3, Cost 3, Risk 3, FlexiTors value 4. **Total 21.**
**Worth trying: YES** (second, after C-GEOM-01), once the blocker text below is fixed.

**Reason.**
- **Evidence.** It is real but indirect:
  - OMEGA's template mechanism is VERIFIED [E-GEOM-038, E-GEOM-044].
  - Its QM9 edge (0.126 vs TD 0.147) is a whole-tool comparison [E-GEOM-025] (§4 says so).
  - The knowledge-based pucker sampler's 0.09 Å is on another set [E-GEOM-043].
  - Coverage is only a test-set leave-one-out proxy (45–59 %) [E-GEOM-012]; training coverage is UNVERIFIED (§3).
- **Leakage design is sound.**
  - The library comes from `split.npy[0]` only, with an assert that no non-isomeric canonical SMILES of a source equals
    a test or validation molecule (§7).
  - Sharing a ring system with a training molecule is legitimate learned knowledge, not leakage.
  - The self-template round-trip that must reproduce A5ring's L error is a proper fidelity control.
- **Single-factor contrasts.**
  - `L_tmpl` vs `L_etkdg2L` changes only ring geometry (ETKDG acyclic L is kept by `set_internal_subset`).
  - `L_tmpl_xtb` vs `L_etkdg2L_xtb` likewise.
  - The §5 primary (vs ETKDG) is the attributable contrast, but it is the easy one. The relevance contrast must be vs
    MMFF (best non-oracle), since central −0.012 is detectable (±0.005).
  - Pre-declare both, with those roles.
- **Covered-subset analysis.** It is valid as a secondary, because coverage is decided from SMILES only (non-oracle).
  Pre-declare it.
- **Template L is other-molecule L.** For B1/S3 every template ring is non-own (§8 revised), so the card is a CTRL card.
  That is honest in §8.
- **A5ring is a soft yardstick, not a lower bound (V-2).** It is own-conformer rings on a different seed base and a
  different population.
- **Risks.** Transplant failures on cages and fused 3/4-rings, mirror pucker vs stereocentres, and
  `set_internal_subset` silently swallowing failures (`lgeom.py:358-366`). The grounder's residual threshold and
  stereo retry address them.

**Validity blocker: YES (text-level).** §5 step 4 runs CTRL/B1 "on the best template source", and §4 picks the key
level "on validation molecules", but no validation path exists (S-5). As written, both choices fall to the test-GT gate
(V-1).

Fix, any of:
- pre-declare the key level by training-split leave-one-out coverage (grounder) and run `L_tmpl` and `L_tmpl_xtb`
  both;
- or pre-declare a single source;
- or build S-5 and choose on validation.

### C-GEOM-03: ET-Flow `qm9-o3` L as a diagnostic
Scores: Evidence 4, Relevance 4, Expected gain 3, Cost 3, Risk 1, FlexiTors value 3. **Total 18.**
**Worth trying: MAYBE, blocked.** It is NO this round unless D-210 closes conclusively.

**Reason.**
- **Evidence.** The mechanism evidence is VERIFIED: ET-Flow 0.073 [E-GEOM-028]; installable, with a separate env
  [E-GEOM-037, E-GEOM-036]; chirality correction in the released model [E-GEOM-053, E-GEOM-054].
- **The split is not established.** That `qm9-o3` trained on TD's split is an INFERENCE contested by V3 (D-210 OPEN):
  - ET-Flow states the GeoMol split for DRUGS only;
  - the checkpoint's config trains from its own `train_indices.npy` in ET-Flow's processed order;
  - current `main` maps TD's `split.npy` onto an unsorted file list.
- **Why this is the worst possible place for a test-set leak.** The card measures whether ET-Flow L reaches the
  λ = 0.75 spec and whether B1/S3 gain from it, i.e. the own-basin share (§8 revised). Memorised test conformers would
  raise exactly that share, so a contaminated result would be falsely positive. The card's conclusion would then not
  follow from its result.
- **The check is likely inconclusive.** The grounder found no `summary_qm9.json` or ET-Flow file on the cluster, so
  index arrays alone cannot be mapped. The download needs user permission.
- **Second confound, even with a clean split.** A positive B1-on-ET-Flow result may mean "TD + B1 re-samples ET-Flow's
  own conformers" (§8 revised). The raw ET-Flow row is the required control (V-8). Comparisons must be on the
  intersection, since ET-Flow may embed molecules ETKDG cannot (V-6).

**Validity blocker: YES (D-210).** Fix: obtain `QM9.zip` (user permission) and show, at SMILES level (test csv columns
0 and 2, non-isomeric canonical), zero overlap between `qm9-o3` train+val and `test_smiles.csv`. If only index arrays
exist, or the mapping is in any way inconclusive: STOP. Relabelling every number ORACLE is not an acceptable
substitute here (grounder), because the contaminated quantity is the one being tested.

### C-GEOM-04: learned ring-pucker component (not this round)
Scores: Evidence 3, Relevance 4, Expected gain 1, Cost 1, Risk 2, FlexiTors value 3. **Total 14.**
**Worth trying: NO** (this round).

**Reason.**
- Agree with the card and with every downstream role.
- PuckerFlow's evidence is VERIFIED but off-domain for about half of QM9's pucker errors:
  - trained on substituent-free monocyclic 5–8 rings [E-GEOM-020];
  - 832/885 QM9 ring molecules carry substituents, and isolated 4-rings and fused systems hold 47 % of the bad seeds
    [E-GEOM-011, E-GEOM-003];
  - its all-atom gain over ETKDG is 0.01–0.02 Å [E-GEOM-023].
- The pre-declared reopen criteria (§5) are good validity practice.
- Criterion 1's λ = 0.75 yardstick is own-conformer ORACLE. It is acceptable as experiment selection under V-1,
  reported either way.

**Validity blocker: NONE** (no experiment this round). Note for the reopen rule: criterion 2 ("C-GEOM-03 shows a learned
model gets the tail below 1 %") cannot be met while C-GEOM-03 is blocked by D-210. Restate it as "a learned L source
with a verified clean split (or one trained by us on TD's split) gets the tail below 1 %". Without that, the rule is
unsatisfiable or, worse, satisfiable through a leaky source.

### C-GEOM-05: ML-potential relaxation (not for QM9)
Scores: Evidence 3, Relevance 3, Expected gain 1, Cost 3, Risk 2, FlexiTors value 2. **Total 14.**
**Worth trying: NO.**

**Reason.**
- The card's central argument is itself a validity argument, and I accept it. The reference is GFN2-xTB
  [E-GEOM-015, E-GEOM-016], while MACE-OFF and similar potentials target DFT [E-GEOM-027]. A DFT-trained relaxation
  converges to systematically different minima: an inference by analogy with MMFF↔GFN2 [E-GEOM-019]; a DFT↔GFN2 QM9
  offset is UNVERIFIED.
- The B1 branch it could unlock needs reference-level (own-basin) L, which it cannot deliver better than xTB.
- The install argument is weaker than written (grounder: micromamba, internet, driver 580). The cost case is gone: xTB
  takes 0.14 s/opt.
- C-GEOM-01 answers everything this card could, at the reference level of theory.

**Validity blocker: NONE** (not proposed). If overruled: the §5 probe's "100 validation molecules" needs S-5 to exist;
on test molecules it would be ORACLE-scored.

### C-GEOM-06: srETKDGv3 seeds as a zero-cost ring-L control
Scores: Evidence 3, Relevance 4, Expected gain 2, Cost 5, Risk 3, FlexiTors value 3. **Total 20.**
**Worth trying: MAYBE.** The CPU part is YES. The GPU part runs only by the pre-declared tail rule, with the fix below.

**Reason.**
- **Single-factor design (D-208, grounding §3).** After D-208 the design is single-factor in intent. The grounder
  measured that `srETKDGv3()` in RDKit 2022.9.5 also switches macrocycle torsions off, so the true single factor is
  `ETKDGv3()` + `useSmallRingTorsions = True` (§7, §8 revised).
- **No common random numbers.** Different parameter sets consume the RNG differently, so pairing is molecule-level
  only, with MDE ≈ 0.008–0.010 Å (impact §8). The predicted CTRL effect (−0.005 to −0.011) is at or below the MDE, so
  a null GPU result would be uninterpretable.
- **The CPU half is well powered.** Tail shares by ring size over ≈ 25k seeds resolve a few-point change easily. That
  is the informative part, and it is the baseline every ring-L source must beat.
- **The evidence predicts a small effect.** Pucker coverage rises 53.3 → 60.1 % on PuckerFlow's set, with no all-atom
  AMR change [E-GEOM-022, E-GEOM-023]. Twisted 6-rings persist [E-GEOM-045]. The preferences are CSD-derived, not GFN2
  [E-GEOM-050].
- **Confound.** CTRL was matched to v1 L [E-GEOM-052], so any CTRL change is net of a train/test shift. That is
  acceptable, because it is the deployable quantity, but it must be stated.

**Validity blocker: GPU step only.** §5 "Primary: paired AMR-R of CTRL on the winning variant vs `L_etkdg2L`" selects
on test GT (V-1). Fix:
- pre-declare the two contrasts: v1 → ETKDGv3 (version step) and ETKDGv3 → ETKDGv3 + small-ring (the factor of
  interest);
- run GPU on the small-ring variant only if its tail (ring ≥ 4 basis) drops ≥ 5 points vs ETKDGv3;
- report the gate either way.
The CPU arm has no blocker.

### C-EVAL-01: literature reporting block and "how good must L be"
Scores: Evidence 5, Relevance 4, Expected gain 3, Cost 5, Risk 3, FlexiTors value 4. **Total 24.**
**Worth trying: YES.**

**Reason.**
- **Evidence.**
  - The literature rows are VERIFIED with populations and protocols stated [E-EVAL-001 … E-EVAL-005, E-EVAL-008].
  - The accounting differences are made explicit: nanmean over successes, failures as 0 % coverage [E-EVAL-033]; the
    65 TD failures and the non-reproducible 92.8 % [E-EVAL-046].
  - The ET-Flow target is now a range, 0.073 published vs 0.083 re-run (D-006 closed).
- **The reporting block is itself a validity tool.** Fixed thresholds, the success-intersection universe and failure
  counts for every arm stop post hoc threshold shopping (V-5).
- **The "how good must L be" analysis carried three confounds, all adopted as fixes in §7/§8 revised.**
  1. Cross-molecule regression confounds L error with molecule difficulty: fit within molecules.
  2. ETKDG/MMFF L-error rows are Hungarian-selected, biased low [E-EVAL-049, D-108]: use one homogeneous statistic.
  3. 1,268 λ slots in 258 molecules reuse another pair's seed and are measured against the source GT: 648 molecules
     primary, 906 as sensitivity.
- **What remains is the leak.**
  - The λ axis is cycled, so the regression measures distance to the target's own L (V-2).
  - The leak-corrected thresholds (§8 revised: no literature level is reachable without own-basin L) come from the P0
    inversion (V-3), which is an estimate with ±0.015–0.04 noise and an unchecked identification assumption.
- **Conclusion.** The card's conclusion is descriptive, so it follows whatever the numbers are, provided they carry
  the right labels.

**Validity blocker: NONE.** Conditions:
- Every threshold is labelled ORACLE, own-conformer or leak-corrected estimate, and is never used to select anything.
- Add the split-half other-conformer condition (V-3) as the measured leak-free ceiling if C-EVAL-02 (d) builds it.
- Fix the R2-1 wording (CTRL touches 0.083 only at λ ≈ 0.98 and never reaches 0.073).
- Make the grounder's verdict line say 648 (primary), not 906.

### C-EVAL-02: local-geometry, stratified and conformer-level diagnostics
Scores: Evidence 4, Relevance 4, Expected gain 3, Cost 4, Risk 3, FlexiTors value 4. **Total 22.**
**Worth trying: YES.**

**Reason.**
- **Evidence.** All VERIFIED and reproduced by V1:
  - the metrics are standard [E-EVAL-019, E-EVAL-009];
  - the tooling exists [E-EVAL-036, E-EVAL-037];
  - the weighting facts were recomputed (48.1 % of molecules vs 10.2 % of conformers; rigid stratum 30.4 % of
    molecules, 54.9 % of the ring-oracle gain) [E-EVAL-048, E-EVAL-057].
- **Items (a)–(c) separate L effects from model effects.** That is the core validity need of round 3.
  - The rigid stratum is a model-free L readout (impact §3a).
  - (c) pairs conformers by index, which the grounder showed is valid when both arms use the same seed per index.
- **Required checks (in §7).**
  - Bond lengths of conformer k agree between arms to 1e-5 Å.
  - The rigid flag comes from TD's `edge_mask`, not `breakdown.py` (D-205).
  - (a) is not exactly constant for same-L arms (the matched GT depends on torsions).
  - Strata are fixed before any round-3 result (§4).
- **Item (d) is now the decision-relevant one** (it decides how S3/B1 respond to non-own L, §8 revised). As specified,
  it is weak on validity:
  - random-GT runs are ≈ 92 % leaky;
  - the conclusion would come from the P0 inversion, with noise amplified 8–14×;
  - S3 has 2 seeds;
  - the identification assumption is unchecked (V-3).
  - A null "S3 shows no penalty" would therefore not be credible.

**Validity blocker: NONE** (diagnostic card). Strong recommendation: replace or supplement (d) with the split-half
other-conformer design (V-3). It costs the same packed runs (≈ 0.9 GPU-h for S3 ×2 + CR ×3; add B1 ×3 at
+0.5 GPU-h) plus ~50 lines. It gives a paired within-molecule own-vs-other contrast with no division by P0. Report
n_true ≥ 2 only.

### C-EVAL-03: GFN2-xTB relaxation metrics (E_relax, displacement to own minimum)
Scores: Evidence 4, Relevance 3, Expected gain 3, Cost 4, Risk 4, FlexiTors value 4. **Total 22.**
**Worth trying: YES.**

**Reason.**
- **Evidence.** The premise is VERIFIED:
  - the references are gas-phase GFN2-xTB minima, xtb 6.2.3 [E-EVAL-012, E-EVAL-013, E-EVAL-055, E-EVAL-056];
  - E_relax ≈ 0 for re-optimised references, and the Nikitin et al. metrics exist [E-EVAL-018, E-EVAL-019,
    E-EVAL-020].
  - The QM9 magnitudes are unknown (§4).
- **The only GT-free quality metric in the plan.** That makes it the one legitimate route to model and L-source
  selection under README rule 5, provided selection uses validation molecules: S-5 plus ~0.3–0.5 GPU-h (grounding).
  This is a validity asset for FlexiTors, so I score Value 4 although its AMR relevance is modest.
- **Known blind spot.** Relaxation is local, so E_relax largely misses the pucker tail (impact §8). Read a "no
  difference" as "bond/angle parity", not as "same L quality".
- **Confound.** TD keeps the seed's acyclic angles at re-randomised torsions (L|τ coupling), so E_relax mixes L
  strain with steric clashes from placed torsions. The card's same-L arm comparison (§3) isolates the torsion part;
  pre-declare it.
- **Version.** xtb 6.7.1 vs 6.2.3 is mitigated: GT conformers sit at 0.000 Å on the probe. The full GT pass-through is
  the zero check.

**Validity blocker: NONE.** Preconditions (V-7):
- the S-1 runner (tmp-dir race, `--norestart`, convergence, charge) and the S-2 graph/stereo filter;
- report the median as well as the mean (cage outliers);
- the fixed 200-molecule list is drawn before any arm is relaxed.

### C-EVAL-04: non-learned baselines (B-mmff, B-xtb, B-clust)
Scores: Evidence 4, Relevance 4, Expected gain 4, Cost 4, Risk 3, FlexiTors value 4. **Total 23.**
**Worth trying: YES** for B-mmff and B-xtb; **MAYBE** for B-clust / TD + clustering.

**Reason.**
- **These are the controls every L-source card needs (V-8).**
  - Without B-mmff, A2's gain cannot be split into "MMFF alone" vs "model on MMFF L" [E-EVAL-038, E-EVAL-039,
    E-EVAL-040, E-EVAL-050].
  - Without the no-model xTB seeds, a B1/S3 gain on xTB L cannot be attributed to the torsion model.
- **Design.**
  - B-mmff is zero code and pairs with CTRL through the seed-pickle variant (§7).
  - B-xtb is the C-GEOM-01 pickle with `--no_model --no_random`, so it shares population and seeds.
  - Differences vs TD (≥ 0.02 Å) are well above noise (impact §8).
- **Evidence.** Zhou's recipe and ablation are VERIFIED, but on the GeoDiff protocol [E-EVAL-025, E-EVAL-026,
  E-EVAL-027]. Only the recipe transfers (§4).
- **Validity problem: B-clust / TD + clustering.**
  - The K-means feature space is unspecified (grounding: "results depend on this choice"). That is a researcher degree
    of freedom on a recall-sensitive endpoint.
  - The budgets are not matched: B-clust draws ≈ 1.5 × 20K (§4), while TD + clustering generates 20K. Zhang et al.'s
    fairness point [E-EVAL-028] is exactly budget matching.
- **Slip.** R2-2 (25 % vs 30 %) needs fixing.

**Validity blocker: B-clust / TD + clustering only.** Fix: pre-register the feature space (e.g. heavy-atom coordinates
after `AlignMol` to the first sample, medoid output, fixed K-means seed) and an identical N_e for both clustering arms,
before any run. B-mmff and B-xtb: NONE (V-7 preconditions for B-xtb).

### C-EVAL-05: xTB post-relaxation ("xTB-after") as a yardstick
Scores: Evidence 4, Relevance 3, Expected gain 3, Cost 5, Risk 4, FlexiTors value 4. **Total 23.**
**Worth trying: YES** (yardstick only).

**Reason.**
- **Evidence.** The premise is VERIFIED [E-EVAL-012, E-EVAL-013, E-EVAL-018, E-EVAL-020]. That our RMSDs are unrelaxed
  is shown by the code path [E-EVAL-035] (D-007 fixed).
- **Pre-declared criterion.** The card pre-declares "strong if AMR-R ≤ 0.150 on the intersection" (§5). That is good
  practice, and the result is informative in both directions (§8: a strong result reframes FlexiTors' value; a weak
  one confirms L must be right during generation).
- **Validity caveats.**
  1. xTB-after moves torsions as well as L (to the nearest minimum), so it is not an L-only yardstick. It must not be
     read as "what better L alone would give".
  2. "Then the full test set if informative" (§5) is a data-dependent expansion. At ≈ 2 CPU-h per arm, pre-declare
     the full set from the start.
  3. Dropped or graph-changed conformers must be counted the same way in recall and precision, with `n_model`
     reported (grounder).
  4. Never place it in a literature table (§4 agrees).

**Validity blocker: NONE.** Conditions: items 2–3 above pre-registered; the V-7 preconditions; the `optimize_confs.py`
defects fixed (DRUGS csv, `--limit 32` truncation, the `xtb_energy` flag bug).

### C-EVAL-06: DFT re-referencing (NO) and Boltzmann ensemble properties (not round 3)
Scores: Evidence 4, Relevance 1, Expected gain 1, Cost 2, Risk 2, FlexiTors value 2. **Total 12.**
**Worth trying: NO.**

**Reason.** Agree with the card, the grounder and the analyst.
- (a) re-referencing replaces the reference against which every literature row [E-EVAL-001 … E-EVAL-008] and every
  oracle analysis is defined. Our "true L" would no longer be the truth, which is a validity break, not an
  improvement.
- QM9's DFT geometry is one structure per molecule, so ensemble recall cannot be scored [E-EVAL-013].
- (b) rests on CREST weights that GEOM itself calls inaccurate, with DFT weights only for BACE
  [E-EVAL-016, E-EVAL-059]. The `--likelihood` path drops 0-rotor molecules and so changes the QM9 population
  (grounding §3).

**Validity blocker if overruled:** (a) cannot coexist with our endpoints; report it only as a side table. (b) needs a
population-preserving likelihood path and must be labelled weight-dependent.

### C-ROBUST-01: error-matched, factorised L augmentation
Scores: Evidence 2, Relevance 3, Expected gain 1, Cost 2, Risk 2, FlexiTors value 3. **Total 13.**
**Worth trying:** Arm 0 **MAYBE**, only with a leak-free partial-L condition added. Arm 1 **NO** as written.

**Reason.**
- **Evidence.** It is analogy only, as card §4 says ("Every source is an analogy"): cascaded diffusion
  [E-ROBUST-023, E-ROBUST-025, E-ROBUST-026], SliDe [E-ROBUST-034, E-ROBUST-035], ProteinMPNN [E-ROBUST-031,
  E-ROBUST-032]. Evidence is therefore ≤ 2.
- **Our-data motivation.** It is VERIFIED but now reads differently. B1 0.188 on A5ring vs 0.154 on noise 0.04
  [E-ROBUST-008] and the λ0-base deltas [E-ROBUST-010] are all on `_cyc_ORACLE` conditions. Those put each target's
  own-conformer components in the seed (V-2), and impact §3g shows that B1's true-L advantage is mostly conformer
  identity.
- **Arm 0.**
  - S3 (trained half on own true L) on own-conformer A5 hybrids can score well by exploiting identity through the
    true component.
  - That would pass the gate (S3 ≤ 0.125 on A5ring and ≤ 0.165 on A5acyc) and stop the experiment with the false
    conclusion "S3 already handles partial L".
  - Using ORACLE conditions to select an experiment is acceptable in principle (V-1; V2's referral), but here the
    gate's reading is invalid, not just ORACLE.
- **Arm 1.**
  - The partial-L benefit is tested only on the same leaky ORACLE conditions (A5ring lower than S3; A5acyc ≤ 0.16;
    GT ≤ 0.04).
  - The non-oracle primary is non-inferiority on RDKit/MMFF L, which is the expected outcome under both hypotheses
    (impact §8: ≈ S3 ± 0.003).
  - So no possible result supports "the model exploits reachable partial L". A positive result can be identity; a
    null is uninformative.
- **Power and other caveats.**
  - Non-inferiority at +0.005 needs ≥ 3 seeds (impact §8); 2 are planned.
  - The 25 % true-L share and hybrid torsion drift (≈ 3–4°, grounding) are secondary.
  - The analyst's flagged fifth source ("GT L of conformer j with the torsion target of conformer i") is the
    mechanism that matches deployable L. It is not in this card.

**Validity blocker: YES.** Fix (both needed before Arm 1; the first is enough to unblock Arm 0):
1. Evaluate partial L on non-own sources. Either:
   - A5ring/A5acyc hybrids built from the other half of a split-half partition of the GT conformers (V-3; n_true ≥ 2);
     or
   - the non-oracle partial-L seeds from C-GEOM-02 (`L_tmpl`) and C-GEOM-01 (xTB) as primary partial-L test
     conditions.
   Recast the Arm 0 gate on those.
2. Arm 1 with ≥ 3 seeds, a pre-declared non-inferiority margin, and a partial-L superiority endpoint on a non-own
   condition.

### C-ROBUST-02: blind continuous L interpolation (S4 without the λ input)
Scores: Evidence 3, Relevance 2, Expected gain 1, Cost 2, Risk 3, FlexiTors value 2. **Total 13.**
**Worth trying: NO** (reopen only if the measured gate passes and the power fix is adopted).

**Reason.**
- **The design is the cleanest single-factor contrast in the ROBUST set.** Blind vs S4 shares the paired cache, the
  pair_ok filter [E-ROBUST-058] and the λ ~ U[0,1] blend, so it isolates the λ input. The card itself notes S3 vs S4 is
  confounded (§3).
- **Validity fails on three counts.**
  1. **Power.** The primary is equivalence within ±0.005 of S4 at RDKit and true L. With seed SD 0.002–0.003 that
     needs 3–6 seeds per arm (impact §8), and 2 are planned. A "no difference" result would be absence of evidence, so
     the card's deployment conclusion ("no quality estimate needed") would not follow.
  2. **Undefined comparator.** The comparator "S4 with λ picked on validation for MMFF" has no code path (S-5;
     grounding §3). Pre-declare λ = 0 (the round-2 rule).
  3. **Leaky conditions.** The mid-λ conditions, where the λ input is supposed to matter, are cycled ORACLE blends
     with 10 % replaced slots (impact §1). The true-L endpoint is own-conformer (V-2), so the λ question is tested on
     conditions that are not deployable.
- **Evidence.** Sun et al.'s result [E-ROBUST-037, E-ROBUST-038, E-ROBUST-039] is VERIFIED, but it is about the noise
  level of the generated variable, not the quality of a conditioning input (§4).
- **The gate's data exist and must be computed before the panel** (impact §8 revised). The 20 % pass probability is a
  prior.

**Validity blocker (if run):**
- ≥ 4 seeds per arm, or drop the equivalence claim and pre-declare a superiority test with a stated MDE;
- S4 λ pre-declared;
- non-oracle MMFF/xTB as the conditions where "blind vs λ" is decided;
- the measured gate reported first.

### C-ROBUST-03: sampler-matched mixture (S3-MMFF)
Scores: Evidence 4, Relevance 3, Expected gain 2, Cost 2, Risk 3, FlexiTors value 3. **Total 17.**
**Worth trying: NO for now.** MAYBE only if the measured S5 gate passes and 3 seeds are funded. The recipe should be
kept for xTB-matched training.

**Reason.**
- **Evidence.** It is direct and VERIFIED: TD's own conformer matching as domain adaptation [E-ROBUST-020,
  E-ROBUST-021, E-ROBUST-022]; the cascade mismatch [E-ROBUST-023, E-ROBUST-024]. DAgger is an analogy, and §4
  correctly limits it [E-ROBUST-044, E-ROBUST-045].
- **The factor structure is valid.**
  - S5 (MMFF-matched only) is the control for "matched training on MMFF L".
  - S3-MMFF vs S5 isolates the true-L half.
  - S5 vs CTRL + MMFF isolates matching.
  - The card lists all three.
- **Validity problems.**
  1. **Power.** The expected effect is 0 to −0.006 (impact §8). −0.005 needs 3 seeds, and 2 are planned. A null would
     not separate "matching on MMFF does not help" from "underpowered".
  2. **Leaky endpoint.** "GT L within 0.005 of S3" is an own-conformer ORACLE endpoint (V-2). The true-L half teaches
     identity use that MMFF L never provides (§8 revised), so it is not a valid success criterion for a deployable
     recipe.
  3. **Unverified pairing.** The MMFF pairing and pair_ok rates are UNVERIFIED (§4) and could change the training
     population relative to S3 (require ≥ 98 % clean, grounder).
  4. **Silent cache failure.** `ConformerDataset` reuses any existing cache path, so S3-MMFF could train on RDKit pairs
     unnoticed. A load-time assert on MMFF-level bond error is needed (grounding §3).
- **Gate.** The S5 data exist, so compute it before deciding (the 30 % pass probability is a prior).

**Validity blocker (if run):**
- 3 seeds;
- non-oracle MMFF L as the only primary, with the GT-L endpoint demoted to an ORACLE descriptive row;
- pairing rate ≥ 98 % reported;
- the cache-identity assert.

### C-ROBUST-04: CFG-style L guidance / CTRL–B1 score interpolation
Scores: Evidence 3, Relevance 2, Expected gain 1, Cost 3, Risk 2, FlexiTors value 1. **Total 12.**
**Worth trying: NO.**

**Reason.**
- Agree with all roles. The CFG evidence is VERIFIED [E-ROBUST-046, E-ROBUST-047, E-ROBUST-048], but the mechanism
  does not map: L is not a droppable token (§4), and the grounder confirmed there is no code path for a weak-L score
  at the current torsion state.
- Score summation does not sample the mixture [E-ROBUST-049].
- The ceiling on every non-oracle L is CTRL (impact §8).
- Validity problem: w is a per-source hyperparameter, like S4's λ [E-ROBUST-017], with no validation path (S-5). It
  would be tuned on test, and the only condition where B1 helps (λ ≥ 0.75, or λ 0.5 in §5) is cycled ORACLE.

**Validity blocker (if run):** w pre-declared or chosen on validation molecules; report RDKit/MMFF only as decision
conditions.

### C-ROBUST-05: S3 ratio sweep / curriculum
Scores: Evidence 3, Relevance 2, Expected gain 1, Cost 1, Risk 3, FlexiTors value 2. **Total 12.**
**Worth trying: NO.**

**Reason.**
- Agree with all roles.
- The headroom is 0.003 Å on the non-oracle primary and 0.012 Å on an ORACLE endpoint [E-ROBUST-060, E-ROBUST-061;
  D-109].
- After the leak finding (V-2), the true-L side of the trade-off measures conformer-identity use, not deployable L
  use, so "moving S3 toward B1 on true L" is not a validity-relevant gain.
- The literature motivations are correctly withdrawn or limited: Huszár's failure needs un-matched targets [E-ROBUST-042,
  E-ROBUST-043; D-103], and the CFG ratio result is about guidance branches [E-ROBUST-046].
- S3's seed spread is now computable from 2 seeds (0.182/0.182 on RDKit L; grounder).

**Validity blocker (if run):** a Δ of 0.003 needs ≥ 4 seeds per arm at SD 0.0015 (`noise_power.out`), and 3 are
planned. The true-L endpoint must be replaced by a non-own reference-level condition (split-half, V-3) or by xTB L.

## 4. What would change my calls
- **C-GEOM-01 and C-GEOM-02 become unconditional YES** once their §5 says "run both variants / pre-declared variant"
  (or S-5 exists). No re-scoring is needed.
- **C-GEOM-03 becomes YES** (Risk 1 → 4) only if `QM9.zip` shows zero SMILES overlap with `test_smiles.csv` in train
  and val.
- **C-ROBUST-01 Arm 0 becomes YES** if it adds the split-half (non-own) A5 conditions or runs on `L_tmpl`/xTB seeds.
  Arm 1 then becomes MAYBE, pending Arm 0 and 3 seeds.
- **C-ROBUST-02 / 03 move to MAYBE** only if the measured gates pass, and only with the seed counts above.
- **If the split-half measurement (V-3) shows CTRL also pays a large other-conformer penalty** (the bin 2–3 point
  estimate implies +0.10, which is within noise today), the CTRL predictions for C-GEOM-01/02 move up. Their calls stay
  YES, because both are informative either way.
