# Scorecard: JUDGE SCIENCE, round 1 (P4 panel)

Written 2026-10-11 by the science judge. Lens: **does the card's answer move FlexiTors-Diffusion (learned local
structure L + torsions) forward, whichever way it comes out?** Scored independently. I did not read the other judges'
files.

Inputs: all 17 cards (§1–8; the latest "Revised after cross-read" blocks are current), `round3/grounding.md`,
`round3/impact.md` (including the leak-corrected floors in §3g), the final "Round 2 verdicts" in
`ledger/verify_V1/V2/V3.md`, and `ledger/disputes.md`. Every E-ID cited here is VERIFIED as of the round-2 verdicts:
V1 59/59, V2 61/61, V3 all entries after the D-201…D-209 revisions. The split statement in E-GEOM-054 is contested
(D-210 OPEN).

## Summary table

| Card | Evid. | Relev. | Exp. gain | Cost | Risk | FlexiTors value | Total /30 | Worth trying | Validity blocker |
|---|---|---|---|---|---|---|---|---|---|
| C-GEOM-01 xTB-relaxed seeds | 4 | 5 | 3 | 4 | 4 | 5 | **25** | **YES** | NONE (conditional on S-1/S-2, pre-declared S4 λ) |
| C-EVAL-02 diagnostics + random-GT runs | 4 | 5 | 3† | 5 | 4 | 5 | **26** | **YES** | NONE |
| C-EVAL-05 xTB-after yardstick | 4 | 4 | 4† | 5 | 4 | 4 | **25** | **YES** | NONE |
| C-GEOM-02 ring templates | 4 | 5 | 4 | 3 | 3 | 5 | **24** | **YES** | NONE (conditional on leakage assert, stereo check) |
| C-EVAL-01 literature block + L spec | 5 | 4 | 3† | 5 | 4 | 4 | **25** | **YES** | NONE (648-molecule within-molecule fit) |
| C-EVAL-04 non-learned baselines | 3 | 4 | 4† | 4 | 4 | 4 | **23** | **YES** (B-mmff, B-xtb); MAYBE (B-clust) | NONE |
| C-EVAL-03 xTB E_relax metrics | 4 | 3 | 2† | 4 | 4 | 3 | **20** | MAYBE | NONE |
| C-GEOM-03 ET-Flow L diagnostic | 3 | 4 | 3 | 3 | 2 | 4 | **19** | MAYBE | **YES: D-210 OPEN** |
| C-GEOM-06 srETKDGv3 seeds | 3 | 4 | 2 | 5 | 3 | 2 | **19** | MAYBE (CPU part only) | NONE |
| C-ROBUST-03 sampler-matched mix (S3-MMFF) | 4 | 4 | 2 | 2 | 4 | 3 | **19** | MAYBE (measured S5 gate) | NONE |
| C-ROBUST-01 factorised error-matched aug. | 2 | 5 | 2 | 3 | 3 | 3 | **18** | MAYBE (Arm 0 only) | NONE (flag for validity judge) |
| C-GEOM-04 learned pucker component | 3 | 5 | 2 | 1 | 2 | 3 | **16** | NO (this round) | NONE |
| C-GEOM-05 ML-potential relaxation | 2 | 3 | 1 | 3 | 3 | 2 | **14** | NO | NONE |
| C-ROBUST-02 blind λ (S4 without input) | 2 | 3 | 1 | 2 | 3 | 2 | **13** | NO (unless measured gate passes) | NONE |
| C-ROBUST-04 L guidance / score interp. | 2 | 2 | 1 | 4 | 2 | 1 | **12** | NO | NONE |
| C-ROBUST-05 p_gt sweep / curriculum | 2 | 3 | 1 | 1 | 4 | 1 | **12** | NO | NONE |
| C-EVAL-06 DFT re-reference / ensemble props | 2 | 1 | 1 | 2 | 2 | 1 | **9** | NO | NONE |

My priority order among the YES cards (science lens; this is not the total order):
**C-GEOM-01** (with B1, S3, S4 and the s metric) → **C-EVAL-02 (d)** (own- vs other-conformer L sensitivity) →
**C-EVAL-05** (paired with C-GEOM-01 as a relax-before vs relax-after test) → **C-GEOM-02** → **C-EVAL-04**
(B-mmff, B-xtb) → **C-EVAL-01**.

## Conventions and cross-cutting points

1. **† Expected gain for measurement cards.** C-EVAL-01…05 propose no arm meant to lower AMR-R, and the analyst
   gives "n/a". For these cards I score the size and clarity of the signal the card is designed to measure, relative
   to its noise (impact.md §2): 5 = large and unambiguous, 1 = predictable or below noise. Other judges may want to
   re-normalise this column.
2. **The finding that reorders round 3 is the conformer-identity leak (impact.md §3g; grounding.md §5.2.1).** Round-1
   `gtL` draws with replacement from the molecule's own GT conformers (code-grounded: `diffusion/sampling.py:86`;
   92 % own-L share). Inverted per n_true bin, B1 with *other* conformers' exact GT L comes out at ≈ 0.16–0.35, i.e.
   RDKit level, and CTRL loses ≈ 0.03 (noisy). Status: this is a P3 analysis that both P3 agents agree on. The code
   facts are grounded; the inverted magnitudes are model-based, with noise amplified 8–14×. They are **not** a VERIFIED
   ledger entry. I therefore use them for expected-effect and value judgements, never as mechanism evidence. The
   literature has a verified analogue: ProteinMPNN's hypothesis that exact coordinates carry target "memory" that
   predicted structures lack (E-ROBUST-032).
3. **What this means for FlexiTors (INFERENCE).** If most of B1's true-L gain is conformer identity, then
   "local structure" in TD is partly a carrier of the conformer's torsional state, through the L|τ coupling of
   acyclic angles. An independent, conformer-agnostic L sampler cannot unlock true-L-trained torsion models. Either L
   has to be generated in the target's own basin, jointly with or conditioned on the torsions, or the torsion model has
   to be trained to use reference-level L from *other* conformers. The cards that measure this were scored up on
   FlexiTors value: C-GEOM-01's s metric, C-EVAL-02 (d), and the pairing of C-EVAL-05 with C-GEOM-01. Cards whose
   rationale rests on cycled own-conformer conditions were scored down: C-ROBUST-01 Arm 1, C-ROBUST-05.
4. **Panel note (not scored; INFERENCE).** Two cheap experiments, neither carded, would test point 3 more directly
   than anything on the list:
   - (i) An "other-conformer L" seed pickle: each target gets L only from a *different* GT conformer of the same
     molecule (n_true ≥ 2), run with CTRL, B1 and S3. That is about 0.5 GPU-h and has none of the inversion's noise
     amplification. It could replace or accompany C-EVAL-02 (d).
   - (ii) The analyst's "fifth source" training (impact.md §5 D): GT L of conformer j paired with conformer i's torsion
     target. It tests whether a torsion model can learn to use non-own reference L at all.
5. **Preliminary data.** S3, S4 and B3/S5 numbers are BRIEF in-job means (unpaired). Scores that lean on them could
   move once the final paired analysis lands: C-GEOM-01's S3 arm and C-ROBUST-02/03/05.

## Per-card scores

### C-GEOM-01: GFN2-xTB-relaxed ETKDG seeds as test-time L
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 4 | 5 | 3 | 4 | 4 | 5 | **25** |

**Worth trying: YES** (my first priority).

Reason. The core premise is VERIFIED:
- GEOM-QM9 references are gas-phase GFN2-xTB minima (E-GEOM-015/016).
- Re-optimising references at GFN2 moves them by ≈ 0, while MMFF minima are offset from them (E-GEOM-018/019).
- The grounder's probe reproduces "GT stays put" under xtb 6.7.1 (§7).

What no source shows is whether ETKDG seeds reach the reference basin. MMFF reached it only partly: the tail went from
49.6 % to 35.1 % (E-GEOM-002). That unknown is the scientific content of the card. After the cross-read it is also the
most decision-relevant unknown in round 3: s, the share of target conformers that get an own-basin xTB seed (§8
revision; impact §3g).

The outcome moves FlexiTors either way:
- **High s, with S3/B1 below CTRL:** non-learned, reference-level relaxation can feed true-L-trained torsion models.
  FlexiTors' learning effort then goes to the ring tail and to basin coverage.
- **Low s:** no local relaxation supplies usable L, and FlexiTors has to generate L jointly with the torsional basin
  (cross-cutting point 3).

Independently of s, the CTRL arm settles whether MMFF's residual pucker tail is a force-field offset (fixable) or a
wrong basin (not fixable) (§4, §8).

Expected gain is only moderate. CTRL is −0.008 vs MMFF, borderline against the paired ±0.005. The best arm, S3, is
predicted at 0.10–0.16 (central 0.13; PRELIMINARY inputs). The GPU cost is ≤ 2.4 GPU-h, and the CPU tooling (S-1/S-2/S-3)
is shared by five cards.

Science requirements for the run:
- Run B1 s0–2, S3 s0/s1, and S4 at pre-declared λ = 0 and λ = 1 alongside CTRL. Without B1/S3 the card cannot answer
  the own-basin question.
- Add s (the seeds' own COV-R@0.05–0.1 from the `--no_model --no_random` run) and pucker coverage to the gate.
- Interpret through the tail and s, not through rigid-set seed AMR-R. That rule can pass on bonds and angles alone (§8,
  impact §4.3).

**Validity blocker:** NONE, provided that:
- the S-1 runner fixes the shared tmp-dir race and checks convergence;
- S-2 is a graph/stereo filter with counted fallbacks, and the gate runs only if ≥ 95 % of seeds relaxed;
- outputs go to a new `round3_seeds` dir (provenance guard);
- the S4 λ values are pre-declared (no validation path, S-5);
- the ORACLE λ rows are used as yardsticks only.

### C-GEOM-02: training-set ring-template L
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 4 | 5 | 4 | 3 | 3 | 5 | **24** |

**Worth trying: YES** (after C-GEOM-01's CPU tooling exists).

Reason. The evidence for the mechanism is direct for the tool:
- OMEGA's ring-conformation library and substituent re-attachment (E-GEOM-038/044);
- ring conformations fall into few canonical clusters (E-GEOM-042);
- a knowledge-based pucker sampler reproduces CREST/GFN2 rings to 0.09 Å, with the residual in bonds and angles
  (E-GEOM-043).

Two pieces are weaker. OMEGA's QM9 advantage (E-GEOM-025) is a whole-tool comparison, and our coverage figure is a
test-set proxy (E-GEOM-012). The ORACLE ceiling is solid: true rings alone take CTRL from 0.197 to 0.119 on the λ0 base
(E-GEOM-013), and the gain is concentrated in low-rotor molecules (E-GEOM-006).

For FlexiTors the card answers brief Q3 in its most useful form: can the ring part of L be a *discrete choice from a
training-data library*, with learning reserved for torsions and for ring systems the library does not cover?
- **Positive:** a pucker-library stage becomes part of FlexiTors' L, and it fixes the scope of any learned ring
  component (C-GEOM-04).
- **Negative:** split by coverage failure vs transplant failure, it triggers C-GEOM-04's reopen criteria with exactly
  the data they need.

The expected CTRL result is detectable: 0.14 (0.11–0.16) against 0.178 on ETKDG and 0.1525 on MMFF, with a paired
L-source CI of ±0.005 (§8). Templates come from other molecules, so for B1/S3 the ring part is non-own L (§8
revision); the CTRL arm carries the decision. The cost is mostly 2–3 days of coding (§7). Building the transplant as
the A5ring recipe with a template ring keeps the arm directly comparable with its ORACLE bound (§7).

**Validity blocker:** NONE, provided that:
- the library is built from the training split only, with a leakage assert against test and validation SMILES;
- mirror/stereo templates are checked and retried;
- `set_internal_subset` failures are counted and fall back (§7);
- the key level is chosen by training-split leave-one-out, since no validation set exists.

### C-GEOM-03: pretrained ET-Flow (`qm9-o3`) L as a diagnostic
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 3 | 4 | 3 | 3 | 2 | 4 | **19** |

**Worth trying: MAYBE** (YES only after a conclusive, leak-free step 0(a)).

Reason. The card asks the most FlexiTors-shaped question in the GEOM lens: can a *learned* model produce L that meets
the λ = 0.75 spec? The evidence is indirect, though:
- ET-Flow's 0.073 Å (E-GEOM-028) is a whole-conformer number and does not separate L from torsions.
- The argument that a refiner cannot reach wrong puckers (E-GEOM-030) is INFERENCE.

After the cross-read, the B1/S3 arms would also be hard to read. ET-Flow samples are conformer-specific, so a B1 gain
would mostly mean that TD is resampling ET-Flow's own conformers (own-basin L). It would not show that learned L works
as an independent input (§8 revision). The CPU gate on the seeds (ring tail, bond/angle error) stays informative about
learned-L quality.

The dominant problem is validity:
- Whether `qm9-o3` saw TD test molecules is unsettled (D-210 OPEN; V3 Round 2; grounding §7).
- Memorised L would inflate exactly the own-basin share this experiment measures.
- Our cluster cannot map `QM9.zip` indices to SMILES.
- The download needs the user's permission.

C-GEOM-01 answers much of the same question at the reference level, without the leakage risk.

**Validity blocker:** **YES: D-210 OPEN.** The absence of TD test molecules in `qm9-o3`'s train/val set must be
proven by a SMILES-level intersection in step 0(a); downloading `QM9.zip` requires the user's permission. If the check
is inconclusive, every ET-Flow-L result is ORACLE-contaminated and the card cannot run this round.

### C-GEOM-04: dedicated learned ring-pucker component (not this round)
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 3 | 5 | 2 | 1 | 2 | 3 | **16** |

**Worth trying: NO** (this round; keep the reopen rule).

Reason. The target is right: rings carry the floor (E-GEOM-013, E-GEOM-002). A Cremer–Pople factor is the natural ring
extension of FlexiTors' torsion manifold (E-GEOM-032), so the answer would shape the design directly. The evidence that
a learned pucker model helps on *our* metric is weak:
- PuckerFlow's in-domain all-atom gain is 0.02 Å unrelaxed and 0.01 Å after MMFF (E-GEOM-023).
- It is trained on substituent-free monocycles (E-GEOM-020), while 832 of 885 QM9 ring molecules carry substituents.
- About half of the tail is in 4-rings and fused systems (E-GEOM-011, E-GEOM-003).
- A joint factor touches every TD layer (§7).

The pre-declared reopen rules (§5) are well chosen, and C-GEOM-01/02 produce the data they need (tail by ring class
after the non-learned sources).

Note for round 4: once C-GEOM-02's transplant code exists, a two-stage seed from a pretrained PuckerFlow on the
in-domain subset needs zero TD changes (§7). That would be the cheapest direct test of learned ring L.

**Validity blocker:** NONE (not run).

### C-GEOM-05: ML-potential relaxation of seeds
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 2 | 3 | 1 | 3 | 3 | 2 | **14** |

**Worth trying: NO.**

Reason. The case against the card is sound:
- The references are GFN2-xTB minima (E-GEOM-015/016), while MACE-OFF is trained to ωB97M-D3 DFT (E-GEOM-027). An ML
  potential therefore relaxes toward a different minimum. This is INFERENCE by analogy with the MMFF offsets
  (E-GEOM-019); no DFT↔GFN2 offset for QM9 is in our sources.
- A local optimiser cannot fix wrong basins better than xTB.
- xTB costs 0.14 s per seed (§7), which removes the speed argument.

Every outcome is dominated by C-GEOM-01. The card would matter only for DRUGS-scale molecules or DFT references,
neither of which is in this phase.

**Validity blocker:** NONE (not run).

### C-GEOM-06: srETKDGv3 seeds as a zero-cost ring-L control
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 3 | 4 | 2 | 5 | 3 | 2 | **19** |

**Worth trying: MAYBE** (the CPU part YES inside the C-GEOM-01 builder; GPU only through the tail gate).

Reason. The evidence predicts a small effect on our metric:
- On PuckerFlow's benchmark the small-ring terms raise pucker coverage from 53.3 % to 60.1 % but leave all-atom AMR
  unchanged (E-GEOM-022/023/026).
- Twisted rings persist in practice (E-GEOM-045).
- ETKDG's preferences are CSD-derived, not GFN2 (E-GEOM-050).

The analyst's −0.005 to −0.011 is below the MDE for different embeddings (≈ 0.008–0.010, §8).

Scientifically the card is a control: the free, knowledge-based baseline that every ring-L source must beat. The
v1 → v3 step also documents a protocol variable, since our seeds are ETKDG v1 (E-GEOM-052). Its seed-level metrics (tail
by ring size) are worth having from the C-GEOM-01 builder run at CPU cost. It will not change FlexiTors' design.

**Validity blocker:** NONE, provided the "small-ring only" contrast uses `ETKDGv3()` + `useSmallRingTorsions=True`
(`srETKDGv3()` also turns macrocycle terms off in RDKit 2022.9.5, §7), and pairing is done at molecule level, since
there are no common random numbers.

### C-EVAL-01: literature-comparable reporting block and "how good must L be"
| Evidence | Relevance | Expected gain † | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 5 | 4 | 3 | 5 | 4 | 4 | **25** |

**Worth trying: YES.**

Reason. The evidence is as direct as it gets for a reporting card. The literature rows are VERIFIED published numbers
(E-EVAL-001…005, -008), and every "where we sit" value is our own verified data (E-EVAL-038, -040, -042…045, -049).

The scientific payload is no longer the λ interpolation in §3. Two things undercut it:
- 258 molecules reuse another conformer's λ seed (impact §1).
- λ is cycled, so every λ point carries own-conformer identity (§8 revision 2).

The payload is the leak-corrected statement in §8 revision 1. With other-conformer L only, CTRL's floor is 0.11–0.145
and B1's is 0.19–0.27, so no literature level (0.073–0.103 Å) is reachable without own-basin L (impact §3g). That
changes what FlexiTors must deliver to claim state of the art on QM9: conformer-specific L, not a better molecule-level
L. The within-molecule fit on the 648 clean molecules (§7, §8 revision 2) turns this into a CI-backed accuracy spec for
jointly generated L, which is the quantity a FlexiTors L component would be trained against.

The † score is moderate because the headline answer is already computed. The card is free (0 GPU-h, 1–2 CPU-h).

**Validity blocker:** NONE, provided that:
- the fit is within-molecule on the 648-molecule population, with the 906 as sensitivity;
- cycled and λ = 1 values are labelled ORACLE upper bounds and carry the §3g inversion;
- the R2-1 wording is fixed: CTRL never reaches 0.073 and touches 0.083 only at λ ≈ 1.

### C-EVAL-02: local-geometry, stratified and conformer-level diagnostics (incl. random-GT runs)
| Evidence | Relevance | Expected gain † | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 4 | 5 | 3 | 5 | 4 | 5 | **26** |

**Worth trying: YES** (my second priority, driven by item (d)).

Reason. The mechanism evidence is our own verified data:
- TD keeps the seed's bonds and angles (E-EVAL-036).
- Rigid molecules carry 54.9 % of the ring-oracle gain (E-EVAL-057).
- B1 scores 0.0215 on cycled true L vs 0.0374 on random-GT true L (E-EVAL-045).

The literature (E-EVAL-019, E-EVAL-009) only motivates the form of the metrics.

From the science lens, item (d) is the most valuable thing in the card, more so after the cross-read than the scout
expected. Random-GT runs of S3 and CTRL_rematch, plus the per-n_true inversion (impact §3g; card §8 revision), measure
how much of each model's true-L gain is conformer identity:
- For B1 the inversion already says almost all of it.
- S3 is the open case. It decides whether S3-type training can gain from any non-own L source (C-GEOM-01/02).

Either answer changes FlexiTors:
- **S3 keeps a gain on non-own reference L:** an independent L sampler plus a robust torsion model is viable.
- **It does not:** L must be generated jointly with, or conditioned on, the torsional basin.

The rigid-stratum readout (§8: CTRL and B1 agree within 0.006 Å on every L) separates L effects from model effects for
every other card. Items (a)–(c) are routine, and the S3-vs-CTRL improvement/downgrade split is expected to be nearly
symmetric. That is why the † score is 3.

Judge recommendation (INFERENCE, not scored): the inversion amplifies noise 8–14×. Add the "other-conformer L" seed
pickle from cross-cutting point 4(i). It measures the same quantity directly for about the same GPU cost.

**Validity blocker:** NONE, provided that:
- random-GT is reported as ≈ 92 % own L, with the inversion, never as a ceiling;
- the strata (TD `edge_mask` rigid) are fixed before round-3 results exist;
- by-index pairing is verified through seed bond lengths (§7).

### C-EVAL-03: GFN2-xTB relaxation metrics (E_relax, displacement to own minimum)
| Evidence | Relevance | Expected gain † | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 4 | 3 | 2 | 4 | 4 | 3 | **20** |

**Worth trying: MAYBE** (run only as a by-product of the C-EVAL-05 / C-GEOM-01 relaxations).

Reason. The evidence for the metric itself is direct and VERIFIED:
- Nikitin's E_relax and displacement to the own minimum (E-EVAL-019/020);
- GEOM references are gas-phase GFN2 minima (E-EVAL-012/013/055);
- TD's relaxation result (E-EVAL-022/023);
- the grounder's probe shows GT conformers stay put under xtb 6.7.1 (§7), so the zero check works.

The scientific problem is what the answer can change:
- E_relax and displacement are local. A wrong pucker or a wrong torsion basin still sits near a minimum (§8). The
  cross-read made basin identity, not bond/angle accuracy, the decisive property of an L source (impact §3g).
- The outcomes for the planned arms are largely predictable: same-L arms come out identical, MMFF lower, and xTB-L
  seeds near zero by construction. Hence † = 2.
- The one FlexiTors-relevant use is a GT-free validation signal for a future learned L generator. That need weakens
  once a validation split exists (S-5).

If it runs, the "within 6 kcal/mol of the lowest GT conformer" fraction is the only part that can see high-energy
puckers, and it should be reported first.

**Validity blocker:** NONE, provided that the S-1 runner fixes (tmp-dir race, `--norestart`, convergence, charge) and
the S-2 graph/stereo filter are in place (§7).

### C-EVAL-04: non-learned baselines (B-mmff, B-xtb, B-clust)
| Evidence | Relevance | Expected gain † | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 3 | 4 | 4 | 4 | 4 | 4 | **23** |

**Worth trying: YES** for B-mmff and B-xtb; **MAYBE** for B-clust and TD + clustering (low science value).

Reason. B-mmff and B-xtb answer an attribution question that every FlexiTors L result inherits: how much of a gain is
the L alone, and how much is the torsion model on top?

Evidence:
- our verified A0/R0/A2 data (E-EVAL-038/039/040);
- the references are xTB minima (E-EVAL-012/013);
- Zhou and Zhang (E-EVAL-025…028) use another split and transfer only the recipe. Hence Evidence 3.

B-xtb is informative either way (§8):
- **It reaches ≤ 0.1525** (analyst P ≈ 20 %): a physics-only pipeline is the bar on QM9, and FlexiTors' torsion model
  has to justify itself on the flexible molecules.
- **It does not:** the model's contribution on reference-level L is measured.

Both arms are CPU-only and share C-GEOM-01's seeds (§7: B-xtb is `L_etkdg2L_mmff_xtb` run with `--no_model
--no_random`; B-mmff needs zero code). B-clust and TD + clustering (2.6 GPU-h; K-means feature space unspecified, §7)
address recall accounting, not FlexiTors' design.

**Validity blocker:** NONE, provided that:
- B-clust's feature space is pre-registered;
- every arm is scored on the success intersection, with failures reported;
- the R2-2 slip is fixed (25 % few-conformer ∩ rigid, 30 % rigid overall).

### C-EVAL-05: GFN2-xTB post-relaxation ("xTB-after") yardstick
| Evidence | Relevance | Expected gain † | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 4 | 4 | 4 | 5 | 4 | 4 | **25** |

**Worth trying: YES** (yardstick only, never reported as a method).

Reason. The premise that an xTB-relaxed sample lands on the reference PES is VERIFIED (E-EVAL-012/013/018/020). Our A3
arm (E-EVAL-041) is the MMFF analogue: precision up, recall down.

The answer is informative whichever way it falls (§8):
- **CTRL + xTB-after reaches the A2 level** (≤ 0.150; analyst P ≈ 35 %): a learned L must beat a cheap post-hoc
  relaxation, and FlexiTors' value shifts toward diversity and cost.
- **It repeats A3:** this confirms that L has to be right *during* generation, which is FlexiTors' founding premise.

It is also the cleanest probe of L|τ coupling on the list (INFERENCE). On the same ETKDG seeds:
- C-GEOM-01 relaxes *before* torsion sampling, so L is inconsistent with the sampled torsions.
- xTB-after relaxes *after* sampling, so L is consistent with them.

The pair therefore tests whether L must be generated together with the torsions, the central FlexiTors design choice.
Both cards share the S-1 runner. The cost is 0 GPU-h and ≈ 11 CPU-h (§7).

**Validity blocker:** NONE, provided that:
- the handling of dropped conformers (graph changes, non-convergence) and the distinct-conformer count are
  pre-registered;
- TD's `optimize_confs.py` defects are fixed: the DRUGS csv path, the 32-conformer limit and the flag bug (§7);
- results are never compared with literature RMSD rows (§4).

### C-EVAL-06: DFT re-referencing (NO) and Boltzmann ensemble properties (not round 3)
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 2 | 2 | 1 | **9** |

**Worth trying: NO** (agree with the scout; (b) is a reasonable metric for the DRUGS phase).

Reason.
- (a) rests on a geometry shift measured for BACE in implicit water (E-EVAL-015), with no gas-phase QM9 measurement,
  i.e. analogy only. It would break comparability with every published number (E-EVAL-001…008) and would turn our
  oracle analyses into non-oracles (§3).
- (b) is sensitive to basin choices (D-001 correction). On QM9, though, it relies on CREST weights that GEOM calls
  inaccurate (E-EVAL-016, -059), and C-EVAL-03/05 measure the basin question more directly.

Neither answer would change FlexiTors' QM9 design.

**Validity blocker:** NONE (not proposed).

### C-ROBUST-01: error-matched, factorised L augmentation
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 2 | 5 | 2 | 3 | 3 | 3 | **18** |

**Worth trying: MAYBE.** Arm 0 can run, trimmed to S3 on A5ring/A5acyc (≈ 1 GPU-h). Arm 1 as designed: NO.

Reason. The mechanism, augmentation that matches the upstream error, has only analogy support: CDM
(E-ROBUST-023/025/026), SliDe (E-ROBUST-034/035), ProteinMPNN (E-ROBUST-031/032) and domain randomisation
(E-ROBUST-036). So Evidence ≤ 2, even though the motivation from our own data is VERIFIED: isotropic noise has no ring
tail (E-ROBUST-006/007), and the models respond differently to ring and acyclic L (E-ROBUST-008/010).

The cross-read undercuts the rationale. "Only B1 gains much from true acyclic geometry (−0.103)" (E-ROBUST-010) is
measured on *cycled own-conformer* L, and impact §3g suggests that most of B1's true-L advantage is conformer identity.
That is ProteinMPNN's "memory" hypothesis (E-ROBUST-032) appearing in our setting. All four training sources pair L
with its own conformer's torsion target, so Arm 1 would teach the model to exploit partial own-conformer L, which no
non-oracle source provides (§8 revision 1). Expected primary gain is ≈ S3 ± 0.003 on RDKit/MMFF L, for ≈ 27 GPU-h.

Arm 0 still answers one useful, one-sided question cheaply: does S3 keep CTRL's ring gain on A5ring? A5ring is an
optimistic, own-conformer condition, so if S3 does not keep the gain there, it will not gain on template L. The
training experiment with more FlexiTors value is the analyst's "fifth source" (cross-cutting point 4(ii)). It tests
whether a torsion model can learn to use reference-level non-own L at all. Arm 1 should be redesigned around it before
it is reconsidered.

**Validity blocker:** NONE from the science lens. Flag for the validity judge: Arm 0's gate uses ORACLE cycled
conditions to select an experiment (README rule 5), and those conditions carry own-conformer identity (impact §3g), so
the gate is optimistic.

### C-ROBUST-02: blind continuous L interpolation (S4 without the λ input)
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 2 | 3 | 1 | 2 | 3 | 2 | **13** |

**Worth trying: NO** (reopen only if the measured gate passes).

Reason. The evidence is analogy only: Sun et al.'s result concerns the noise level of the generated variable in images,
and its concentration argument is dimension-dependent (E-ROBUST-037/038/039). Our data already answer most of the
question. S3 has no level input, matches S4 on RDKit L, and is better on true L (0.033–0.034 vs 0.041–0.044; BRIEF,
PRELIMINARY, and the two trained on different subsets).

The gate data exist on gnode118 (§7) and should be computed on CPU before any GPU is spent. A blind-vs-conditioned
answer would matter to FlexiTors only if it kept a separate L source of unknown quality. C-GEOM-01's S3 vs S4
(pre-declared λ) on xTB L gives that answer for ≈ 1 GPU-h instead of ≈ 26.5. Two seeds are also underpowered for
±0.005 equivalence (§8).

**Validity blocker:** NONE.

### C-ROBUST-03: sampler-matched mixture (S3-MMFF)
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 4 | 4 | 2 | 2 | 4 | 3 | **19** |

**Worth trying: MAYBE** (only if the measured S5 gate passes; better redirected to an xTB-matched S3 after C-GEOM-01).

Reason. The principle is the best-evidenced one in the ROBUST lens, and it comes from TD itself: training on true L
causes a damaging shift, which conformer matching to the deployed sampler fixes (E-ROBUST-020/021/022). The cascade
and DAgger sources are supporting analogies (E-ROBUST-023/024/044/045). It is also the recipe FlexiTors will need for
any generated L.

MMFF is the wrong test bed, though:
- CTRL already uses MMFF L about as well as the matched-MMFF control (0.1525 vs 0.154, §8).
- MMFF keeps 35 % of the pucker tail, which no training can fix (§4).
- The true-L half teaches the use of own-conformer L, which MMFF never supplies (§8 revision).

The expected effect is 0 to −0.006 for ≈ 26 GPU-h. The informative instance is an xTB-matched S3, if C-GEOM-01 shows
that CTRL or S3 gains on xTB L. The S5 gate can be computed on CPU now (§7).

**Validity blocker:** NONE, provided that:
- a new cache name and an assert on MMFF-level bond error in the RDKit half guard against the silent stale-cache reuse
  (§7);
- the pairing rate is ≥ 98 %.

### C-ROBUST-04: CFG-style L guidance / CTRL–B1 score interpolation
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 2 | 2 | 1 | 4 | 2 | 1 | **12** |

**Worth trying: NO** (agree).

Reason. Three problems, each sufficient:
- L is not a droppable token in TD. Code confirms there is no path that transfers a torsion state between L seeds (§4,
  §7).
- Guidance trades diversity for fidelity, against our recall endpoints (E-ROBUST-047/054).
- Summed scores do not sample the composed distribution (E-ROBUST-049).

The ceiling on every non-oracle L is CTRL (§8). Neither outcome would change FlexiTors.

**Validity blocker:** NONE.

### C-ROBUST-05: S3 ratio sweep / curriculum
| Evidence | Relevance | Expected gain | Cost | Risk | FlexiTors value | Total |
|---|---|---|---|---|---|---|
| 2 | 3 | 1 | 1 | 4 | 1 | **12** |

**Worth trying: NO** (agree).

Reason. The headroom is 0.003 Å on the primary endpoint and 0.012 Å on an ORACLE endpoint (E-ROBUST-004/060/061). The
cross-read makes the ORACLE endpoint less meaningful still: if B1's true-L edge is mostly conformer identity (impact
§3g), moving S3 toward B1 there buys little usable capability. Curriculum motivations do not apply, because nothing
compounds and S3's targets are matched (E-ROBUST-040/042/043; §4). The cost is 72–96 GPU-h for a predicted monotone
trade-off. A ratio question should first be asked on a deployable L source (S3 on xTB L, inside C-GEOM-01).

**Validity blocker:** NONE.
