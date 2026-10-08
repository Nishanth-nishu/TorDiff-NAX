# Verification V2: round-3 ledger evidence_ROBUST.md (59 entries) and cards C-ROBUST-01..05

Verifier: V2. Written 2026-10-08. Role: orchestration/roles/verifier.md (adversarial). Disputes: D-101..D-110 in
`round3/ledger/disputes.md`.

Method (all reproducible; scripts and outputs in `round3/ledger/v2_data/`, written independently of the scout's
`robust_data/lerror_stats.py`, run with `python -I` from the repo root):
- our-data: `v2_lerror.py` re-reads `cluster_sync/round2/data/QM9/round2_seeds/l_error.csv` with the csv module
  (output `v2_lerror.out`): per-condition means, ring-size splits by SSSR min and max ring size, and common-molecule
  comparisons. `v2_summaries.py` re-reads every `summary.txt` (round 2: `v2_summaries_round2.out`; round-1 in-job
  runs: `v2_summaries_round1.out`). Breakdown tables read directly from `breakdown.log`.
- papers: own `pdftotext` extraction of each cited PDF (not the scout's fulltexts), page split on form feed; quote
  matched whitespace-collapsed and, where glyphs are lost (×, Å, ∈), after dropping punctuation (`v2_quotes.py`);
  surrounding paragraph read for every quote; section headings checked with `pdftotext -layout`.
- blogs: snapshot quote and section checked (`v2_blogs.py`); live URL fetched with WebFetch on 2026-10-08.
- code: every `path:line` opened. Hybrid-builder timing re-measured (`v2_timing.py`).

## Verdict table

| Entry | Type | Verdict | Dispute | One-line reason |
|---|---|---|---|---|
| E-ROBUST-001 | our-data | VERIFIED | | 0.2361/0.2381/0.2327 and 0.1774/0.1779/0.1783 reproduced, n = 936 |
| E-ROBUST-002 | our-data | VERIFIED | | 0.0199 (s0) and 0.0803/0.0804/0.0808 reproduced, n = 955 |
| E-ROBUST-003 | our-data | VERIFIED | | 0.2078/0.2037/0.2064 and 0.0532/0.0533/0.0557 reproduced; mixed test sets noted by scout |
| E-ROBUST-004 | our-data | VERIFIED | | 0.1816 / 0.0331 / CTRL gtLcycle 0.0816-0.0827 reproduced; round-1 in-job CTRL_rematch 0.1786 confirmed |
| E-ROBUST-005 | our-data | VERIFIED | | numbers exact; caveat: at λ = 0.75 B1 vs CTRL (0.1146 vs 0.1156) is a tie within seed spread |
| E-ROBUST-006 | our-data | VERIFIED | D-108 (advisory) | T1 reproduced to 4 d.p.; caveat: RDKit reference is Hungarian-assigned on heavy angles (biased low); vs the matched RDKit L used in training, noise is 1.60× bonds but 0.84× angles |
| E-ROBUST-007 | our-data | DOES NOT SUPPORT | D-101 | shares reproduced, but "grows with ring size" is false between 4 (60%) and 5 (48%); the ring-3 9% is a metric artefact (3-rings have identically zero endocyclic dihedral error) |
| E-ROBUST-008 | our-data | DOES NOT SUPPORT | D-102 | not like-for-like: noise 0.04 has 2× A5ring's bond error but less heavy-atom angle error (3.31 vs 3.51°; acyclic heavy 3.37 vs 4.11°); "per unit of bond/angle error" not established; unpaired |
| E-ROBUST-009 | our-data | VERIFIED | | T1/T3 rows reproduced |
| E-ROBUST-010 | our-data | VERIFIED | | numbers exact; caveat: against the matched base λ = 0 (same 955 mols) CTRL does gain 0.014 Å from true acyclic geometry, and B1 gains as much from true rings (0.074) as CTRL (0.078) |
| E-ROBUST-011 | our-data | VERIFIED | | all eight breakdown values read from the four breakdown.log files |
| E-ROBUST-012 | our-data | VERIFIED | | 0.1251/0.0918/0.1341 vs 0.0977/0.0977/0.0973 reproduced |
| E-ROBUST-013 | code | VERIFIED | | r2_evalsets.tsv:5-9, 30-31 and r2_eval_models_post.tsv:4-8 as claimed |
| E-ROBUST-014 | code | VERIFIED | | dataset.py:59-63, 90-95, 96-98 match quote and claim |
| E-ROBUST-015 | code | VERIFIED | | make_l_seed_pickles.py:236-237 exact; set_internal_subset is lgeom.py:341-369 (return on 369); my timing 2.5 ms/call (18 QM9 mols) |
| E-ROBUST-016 | code | VERIFIED | | dataset.py:74-76 exact; target relative to the selected conformer |
| E-ROBUST-017 | code | VERIFIED | | sampling.py:181-183, score_model.py:194-199, generate_confs.py:52 as claimed |
| E-ROBUST-018 | code | VERIFIED | | tsv:15-17, score_model.py:65-70, parsing.py:45 default 0; no coupling assert; blind run must drop `--l_level` from the gen-args fields (generate_confs.py:107-110 exits otherwise) |
| E-ROBUST-019 | our-data | VERIFIED | | angle_rmsd_all_acyc mean 0.5834 reproduced; lgeom.py:345-346 documents it |
| E-ROBUST-020 | paper | VERIFIED | | TD p. 7 §4.1, exact; context supports |
| E-ROBUST-021 | paper | VERIFIED | | TD p. 19 App. E, exact |
| E-ROBUST-022 | paper | VERIFIED | | TD p. 8 §4.1, exact; "change torsion angles to minimize RMSD" is in the preceding sentence |
| E-ROBUST-023 | paper | VERIFIED | | CDM p. 3 §1, exact |
| E-ROBUST-024 | paper | VERIFIED | | CDM p. 18 §4.3, exact (hyphenation only) |
| E-ROBUST-025 | paper | VERIFIED | | CDM p. 18 §4.4, exact except × glyph |
| E-ROBUST-026 | paper | VERIFIED | | CDM p. 7 §3.1, exact |
| E-ROBUST-027 | paper | VERIFIED | | CDM p. 6 §3, exact |
| E-ROBUST-028 | paper | VERIFIED | | CDM p. 8 §3.3, exact; same model/training as truncated CA, corruption applied at sampling |
| E-ROBUST-029 | paper | VERIFIED | | DDPM-IP p. 1 Abstract, exact; analogy labelled |
| E-ROBUST-030 | paper | VERIFIED | | DDPM-IP p. 2 §1, exact; "training regularization" in same sentence |
| E-ROBUST-031 | paper | VERIFIED | | ProteinMPNN p. 7, exact except Å glyph; "PDB recovery decreased" is in the same sentence |
| E-ROBUST-032 | paper | VERIFIED | | p. 7 exact; note the source hedges ("may impart") |
| E-ROBUST-033 | paper | VERIFIED | | p. 3 Table 1 row and caption (left native, right 0.02 Å; last column AlphaFold models) |
| E-ROBUST-034 | paper | VERIFIED | | SliDe p. 2 §1, exact |
| E-ROBUST-035 | paper | VERIFIED | | SliDe p. 2 §1, exact |
| E-ROBUST-036 | paper | VERIFIED | | Tobin p. 1 Abstract, exact |
| E-ROBUST-037 | paper | VERIFIED | | Sun p. 1 Abstract, exact |
| E-ROBUST-038 | paper | VERIFIED | | Sun p. 4 §4.2, exact (glyphs as extracted) |
| E-ROBUST-039 | paper | VERIFIED | | Sun p. 8 §6.2 "Alternative Noise-conditioning Scenarios", exact; (b) = pre-trained predictor, (d) = unconditional |
| E-ROBUST-040 | paper | VERIFIED | | Bengio p. 1 Abstract, exact |
| E-ROBUST-041 | paper | VERIFIED | | Bengio p. 2 §1, exact |
| E-ROBUST-042 | paper | VERIFIED | | Huszár p. 1 Abstract, exact |
| E-ROBUST-043 | paper | WRONG LOCATOR | D-103 | quote exact on p. 4 but in §4.1 "Scheduled sampling formulated as KL divergence minimisation", not §2 |
| E-ROBUST-044 | paper | VERIFIED | | DAgger p. 1 §1, exact |
| E-ROBUST-045 | paper | VERIFIED | | DAgger p. 3 §3, exact; expert relabelling in the following sentences |
| E-ROBUST-046 | paper | WRONG LOCATOR | D-104 | quote exact on p. 8 but in §4.2 "Varying the unconditional training probability", not §4.1 |
| E-ROBUST-047 | paper | WRONG LOCATOR | D-105 | quote exact on p. 9 but in §5 "Discussion", not §6 "Conclusion"; supports claim in context |
| E-ROBUST-048 | paper | VERIFIED | | CFG p. 4 §3.2, exact |
| E-ROBUST-049 | paper | WRONG LOCATOR | D-106 | quote exact on p. 5 but in §4 "Scaling compositional generation with diffusion models" (before 4.1), not §3 |
| E-ROBUST-050 | blog | VERIFIED | | snapshot = post source; rendered site refused connection (ECONNREFUSED) for V2 too; raw GitHub source fetched live: quote and section confirmed |
| E-ROBUST-051 | blog | VERIFIED | | as E-050; section "Score-based generative modeling with multiple noise perturbations" |
| E-ROBUST-052 | blog | VERIFIED | | live page and snapshot agree; section "Guided Domain Randomization" |
| E-ROBUST-053 | blog | VERIFIED | | live page and snapshot agree; section "Match Real Data Distribution" |
| E-ROBUST-054 | blog | VERIFIED | | live page and snapshot agree; section "Classifier-free guidance" |
| E-ROBUST-055 | blog | VERIFIED | | live page and snapshot agree; same section |
| E-ROBUST-056 | blog | VERIFIED | | live page and snapshot agree; note Weng's "only applied to training" is not true of CDM's Gaussian CA (E-028) |
| E-ROBUST-057 | code | VERIFIED | | build_paired_pickles.py:50 and sbatch:57 as claimed; standardize_confs.py:83-117 relaxes before computing the stored rmsd, so the pairing check is consistent for MMFF pickles |
| E-ROBUST-058 | code | WRONG LOCATOR | D-107 | substance verified; quoted lines are dataset.py:145-147 (block 145-162), not 144-146 (144-161) |
| E-ROBUST-059 | our-data | VERIFIED | | 0.1536/0.1531/0.1554, n = 935 reproduced |

Counts: VERIFIED 52, WRONG LOCATOR 5 (E-043, E-046, E-047, E-049, E-058), DOES NOT SUPPORT 2 (E-007, E-008),
QUOTE MISMATCH 0, UNVERIFIABLE 0. Total 59.

## The 9 PDFs added this round (identity check, page 1 of my own extraction)
All nine are the claimed papers: `2011_ross_dagger` (Ross, Gordon, Bagnell, arXiv 1011.0686v3), `2015_bengio_scheduled_sampling`
(Bengio, Vinyals, Jaitly, Shazeer, arXiv 1506.03099v3), `2015_huszar_scheduled_sampling_critique` (Huszár, arXiv
1511.05101v1, "How (not) to train your generative model"), `2017_tobin_domain_randomization` (Tobin et al., arXiv
1703.06907v1), `2022_dauparas_proteinmpnn` (bioRxiv 10.1101/2022.06.03.494563, posted 4 June 2022),
`2022_ho_classifier_free_guidance` (Ho & Salimans, arXiv 2207.12598v1), `2023_du_reduce_reuse_recycle` (Du et al.,
arXiv 2302.11552v6, 14 Sep 2024), `2024_ni_sliced_denoising` (Ni et al., arXiv 2311.02124v1 "Preprint. Under review",
i.e. the pre-ICLR version; INDEX venue "ICLR 2024" refers to the later publication), `2025_sun_noise_conditioning`
(Sun, Jiang, Zhao, He, arXiv 2502.13129v2). Rows 60-68 of `papers/INDEX.md` describe them correctly. Page numbers
cited are for these exact versions (Du v6, Sun v2); other versions will paginate differently.

## Recomputed data claims (the ones the orchestrator flagged)

Every T1, T2 and T3 number in `robust_data/lerror_stats.txt` reproduces exactly with my own code. What changes is
the reading of some of them.

1. "Isotropic 0.04 Å/axis noise gives 1.75× RDKit's bond error and 1.18× its angle error" (E-006, C-ROBUST-01 §3).
   Arithmetic right (0.0549/0.0314 = 1.75; 3.3077/2.7963 = 1.18; same on the 936 common molecules). But the
   `L_etkdg2L` error rows are not "RDKit's error per seed": `tools/make_l_seed_pickles.py:105-114` keeps, for each GT
   conformer, only the ETKDG seed chosen by a Hungarian assignment that minimises heavy-atom angle RMSD over 2L
   seeds. The angle error (and the correlated bond error) is therefore a best-case value, while the noise rows are
   measured against their own source GT. The RDKit L that CTRL/S3 actually train on (matched seed = `L_lam0.00`,
   compared with its paired GT, same 936 molecules) has heavy bonds 0.0344 Å and heavy angles 3.94°: against it,
   noise 0.04 is 1.60× the bond error and **0.84×** the angle error. "Noise is larger than RDKit error" holds for
   bonds only. The ring-tail statement is robust: > 10° in 28.6% (ETKDG) / 30.8% (matched λ = 0) of ring seeds vs
   0.35% for noise on common molecules.
2. "0.34% of ring seeds have dihedral error > 10° vs 28.6% for RDKit" (E-006). Reproduced (0.0034 vs 0.2856; the
   metric is the per-seed RMSD over endocyclic dihedrals). Denominator note: 3078 of the 7263 ETKDG ring seeds (42%)
   belong to molecules whose only rings are 3-membered, where every endocyclic dihedral is identically 0 for any
   geometry. Among seeds with a ring of ≥ 4 atoms the shares are 49.6% (ETKDG) vs 0.58% (noise). Same conclusion,
   larger contrast.
3. "48-92% for 4-7-membered rings" (E-007). Reproduced by smallest SSSR ring: 60.4% (4), 48.1% (5), 66.5% (6),
   91.9% (≥ 7). By largest ring: 45.7%, 43.1%, 67.1%, 90.5%. The range is right; the entry's "grows with ring size"
   is not monotone (5 < 4 either way). The "9% (smallest ring 3)" is a dilution artefact: 3078 of those 4312 seeds
   have only 3-rings (error 0 by construction); among the 1234 that also have a larger ring the share is 31.7%.
4. "B1 hurt more by RDKit-type acyclic error (0.188) than by larger random noise (0.154)" (E-008, C-ROBUST-01 §3).
   Sources: 0.188 = mean of `cluster_sync/round2/results/qm9_B1_train_gtL_e100_s{0,1,2}/S1_A5ring_cyc_ORACLE/summary.txt`
   (0.1896 / 0.1922 / 0.1813; n_evaluated 955, 45 failures); 0.154 = mean of `.../S1_noise0.04pa_cyc_ORACLE/summary.txt`
   (0.1598 / 0.1376 / 0.1635; n 996, 4 failures). Same molecules? No: A5ring's 955 are a strict subset of noise's 996
   (41 molecules dropped from the λ/A5 family by a failing pair); no per-molecule outputs are synced, so no paired
   comparison is possible. Same L-error magnitude? No, and not uniformly "larger" for noise:

   | L condition (common 955 mols) | heavy bonds Å | heavy angles ° | heavy acyclic angles ° | all-atom acyclic angles ° | ring dih. mean ° | B1 AMR-R | CTRL AMR-R |
   |---|---|---|---|---|---|---|---|
   | A5ring (GT rings + matched-RDKit acyclic) | 0.0270 | 3.51 | 4.11 | 3.60 | 0 | 0.1877 | 0.1186 |
   | GT + noise 0.04 Å/axis | 0.0549 | 3.31 | 3.37 | 3.98 | 1.92 | 0.1536 | 0.1186 |
   | GT + noise 0.02 Å/axis | 0.0275 | 1.64 | 1.67 | 1.99 | 0.96 | 0.1170 | 0.0976 |

   Noise 0.04 is 2× larger in bonds but 0.94× (heavy angles) and 0.82× (heavy acyclic angles) of A5ring; noise 0.02
   matches A5ring's bonds but has half its angle error. Neither noise level matches A5ring on both, so "per unit of
   bond/angle error" and "larger random noise" are not established. What the data do show (direction robust: every
   B1 A5ring seed is worse than every B1 noise-0.04 seed): B1 is hurt more by A5ring than by noise 0.04 while CTRL is
   hurt equally by both (0.1186 vs 0.1186). That B1-specific sensitivity is a legitimate, unpaired observation; the
   "systematic vs random per unit error" reading is not.
5. Mixed test sets in the cards. S2/S3 numbers are in-job (RDKit L n = 935; gtLcycle n = 996); B1/CTRL numbers are S1
   seed pickles (n = 936; λ = 1 n = 955, B1 seed 0 only). Like-for-like round-1 in-job values exist
   (`cluster_sync/results/*/steps20_seed0*/summary.txt`): CTRL_rematch RDKit L 0.1786 (3 seeds, 935), B1 RDKit L 0.2341
   (3 seeds, 935), B1 gtLcycle 0.0212 (3 seeds, 996). With them, S3's gaps are 0.003 Å (not 0.004) behind CTRL and
   0.012 Å (not 0.013) behind B1. No conclusion changes; the cards should cite the like-for-like values.
6. Seed noise at the endpoints (for C-ROBUST-05): per-seed AMR-R ranges are CTRL RDKit L 0.1774-0.1783 (S1) /
   0.1773-0.1800 (in-job), B1 RDKit L 0.2327-0.2381, CTRL gtLcycle 0.0816-0.0827, B1 gtLcycle 0.0206-0.0215. The
   0.092-0.134 spread of E-012 is B1 under 0.02 Å noise, not the noise level at S3's endpoints.

## Notes per entry (non-trivial ones only)

- E-005: "overtakes only at λ = 0.75" is literally true on means (0.1146 < 0.1156) but seed ranges overlap
  (B1 0.1116-0.1165, CTRL 0.1146-0.1164); "draws level at λ ≈ 0.75" is the defensible wording. No dispute.
- E-006: see recompute 1-2. Advisory thread D-108 asks the scout to name the reference (Hungarian-assigned ETKDG)
  and add the matched-λ0 comparison; numbers stay VERIFIED.
- E-010: against the like-for-like base λ = 0 (same 955 molecules, same pairs): CTRL 0.1966 → A5ring 0.1186 (−0.078),
  → A5acyc 0.1824 (−0.014); B1 0.2614 → A5ring 0.1877 (−0.074), → A5acyc 0.1583 (−0.103). "CTRL does not gain from
  acyclic" is "gains little"; and B1 gains about as much from true rings as CTRL does. The claim's two comparisons are
  still correct as stated, hence VERIFIED.
- E-015: timing reproduced locally (2.49 ms mean, 2.53 median, 4.15 max per call, 18 QM9 molecules, two ETKDG
  embeddings each); cluster timing still unmeasured, as the scout says.
- E-018: VERIFIED. A blind S4 line must also drop `--l_level 0` / `--l_level 1` from the in-job generate fields,
  otherwise generate_confs.py:107-110 exits ("--l_level is required for, and only for, lambda-conditioned models").
  Config, not code.
- E-032: the source hedges ("may impart some memory"); quote and claim ("attributes") are fine. C-ROBUST-01 §2 states
  it as fact ("because exact coordinates carry information"), see card notes.
- E-043: besides the locator, note that Huszár's analysis is about feeding generated inputs with the ORIGINAL targets
  (Eq. 4-6, p. 4); the ε → 0 optimum is the factorised Q = P_x1 P_x2. This matters for how C-ROBUST-05 uses it.
- E-049: Fig. 2 on the same page shows reverse diffusion failing for both product and mixture composition, so
  "summing or mixing" is supported.
- E-056: Weng's following sentence ("conditioning noise is only applied to training but not at inference") is
  inaccurate for CDM's Gaussian conditioning augmentation, which is applied at sampling (E-028); only the blur
  augmentation is training-only (E-026). The entry only uses the first sentence, so VERIFIED; do not cite the second.
- E-057: VERIFIED. Additional check: `torsional-diffusion/standardize_confs.py:83-87` relaxes with MMFF before
  matching and `:108` computes the stored rmsd on the relaxed, matched conformer, so build_paired_pickles' step-2
  check (re-align and compare with conf['rmsd']) is consistent for the MMFF pickles. The pair_ok rate stays
  UNVERIFIED.

## Card rulings

### C-ROBUST-01: WEAKENED
Core mechanism (isotropic noise lacks RDKit's ring-error tail; S2/S3 never evaluated on partial L) rests on VERIFIED
entries (E-006 numbers, E-013, E-014, E-015, E-019) and verified analogies (E-023, E-025, E-026, E-034, E-035). The
following sentences are not supported as written:
1. §3 "RDKit-derived error hurts B1 more than larger random error ... [E-008]": E-008 DOES NOT SUPPORT; noise 0.04 is
   not larger in heavy-atom (acyclic) angles (recompute 4). Rewrite as the B1-specific, unpaired observation.
2. §3 "At 0.04 Å/axis it gives 1.75× RDKit's bond error and 1.18× its angle error": reference-dependent; against the
   matched RDKit L used in training it is 1.60× / 0.84× (recompute 1). Keep the ring-tail contrast, which is robust.
3. §2 "The augmentation had to match the upstream error": CDM reports that Gaussian noise failed and blur worked at
   high resolution (E-025); it does not state a matching principle. This is INFERENCE and must be labelled.
4. §2 "because exact coordinates carry information about the target": ProteinMPNN says "may impart" (E-032); hedge it.
5. §3 "The two trained models use different parts of L ... A model that has seen each component wrong on its own is
   the natural way to get both gains": against the matched λ = 0 base, B1 gains about as much from true rings as CTRL
   (−0.074 vs −0.078); only the acyclic gain differs (E-010 note). The second sentence is unlabelled INFERENCE.
6. §3 "true rings alone take AMR-R from 0.178 on RDKit L to 0.119": cross-condition; the like-for-like base is
   λ = 0 (0.197 → 0.119, same 955 molecules). Direction unchanged, effect larger.
7. §4 "S3 0.033 vs B1 0.020": mixed sets; like-for-like B1 gtLcycle is 0.0212 (recompute 5).
Validity note for judges (not a verifier ruling): the Arm-0 stop/go gate is decided on ORACLE conditions (A5ring,
A5acyc). It selects an experiment, not a model, but the README rule 5 should be confirmed by the validity judge.

### C-ROBUST-02: SUPPORTED
Every factual sentence rests on VERIFIED entries or an open locator-only dispute (E-058, substance verified). Minor:
§3 "Our blind discrete model already infers the regime from L" is INFERENCE (S3's two endpoint scores are consistent
with it but do not show it); label it. §3 "9% of seeds with smallest ring 3 ... against 48-92%": the 9% is a dilution
artefact (recompute 3); the unevenness claim stands without it. §5: the blind arm also needs the `--l_level` gen
fields removed (E-018 note).

### C-ROBUST-03: SUPPORTED
TD, CDM, DAgger and Huszár are quoted correctly in context; the DAgger/scheduled-sampling mapping is labelled
INFERENCE and §4 states the no-feedback-loop limit. One unsupported sentence: §3 "It halves the bond/angle error
(bonds 0.018 Å, angles 1.97°)": MMFF vs raw ETKDG is 0.58× bonds and 0.70× heavy angles (E-009 / T1), so "cuts bond
error by ~40% and angle error by ~30%". "A model trained on MMFF L could" is the card's hypothesis; label it.

### C-ROBUST-04: SUPPORTED
The card recommends NO, and its reasons are either verified (E-046, E-048, E-049, E-054, E-055, E-005, E-017) or
labelled INFERENCE. Pending: section locators of E-046, E-047, E-049 (D-104..D-106); none changes the content.

### C-ROBUST-05: WEAKENED
The NO verdict leans on two claims that the evidence does not support:
1. §3 "Effects that small sit near our noise. B1's AMR-R spans 0.092-0.134 across 3 training seeds under 0.02 Å noise
   [E-012]": E-012 is B1's instability under small L noise, not the seed noise at S3's endpoints. Measured seed ranges
   there are 0.001-0.005 Å (recompute 6), smaller than the 0.003/0.012 Å gaps. S3's own seed variance is unknown (1
   seed). The "≥ 3 seeds" advice may still be prudent, but not for this reason.
2. §4 "Huszár's 'ignore the conditioning' failure [E-043] is a real risk at high RDKit share": Huszár's failure comes
   from feeding generated inputs with the original targets (inconsistent objective). The card's own previous bullet
   argues S3's targets are consistent (each L carries its matched torsions, E-016), so Huszár's mechanism does not
   apply. A low-p_gt mixture drifting toward CTRL is a data-weighting effect, a different argument (INFERENCE).
Also: §3 gaps "0.004 / 0.013" use mixed test sets (like-for-like 0.003 / 0.012, recompute 5); §3 "No ratio can gain
more than that at the two endpoints" and §2 "Some trade-off is inherent" are unlabelled inferences (one protein
experiment cannot establish the latter for us).

## Round 2 verdicts (after the scout's responses to D-101..D-110; 2026-10-08)

Re-check method: every revised or new number compared with my own outputs (`v2_data/v2_lerror.out`,
`v2_summaries_round2.out`, `v2_summaries_round1.out`) and an extra csv pass for the ring ≥ 4 / 3-ring-only split.
`git diff` confirms T1-T3 in `robust_data/lerror_stats.txt` are unchanged and that no paper quote line changed in
`evidence_ROBUST.md`; only locators, our-data claims and new T4a-c/T5 lines changed.

| Entry | Round-1 verdict | Round-2 verdict | Thread | Re-check |
|---|---|---|---|---|
| E-ROBUST-006 | VERIFIED (advisory) | VERIFIED | D-108 CLOSED-VERIFIED | T4a = my common-936 numbers (λ0 0.0344 Å / 3.939°, noise 0.0549 / 3.311, gt10 0.2856 / 0.3076 / 0.0035); T4b 49.61% / 0.58%; Hungarian at make_l_seed_pickles.py:106-114 confirmed |
| E-ROBUST-007 | DOES NOT SUPPORT | VERIFIED | D-101 CLOSED-VERIFIED | 60.4 / 48.1 / 66.5 / 91.9%, "not monotone"; 3078 3-ring-only seeds all exactly 0 (my check: 0 of 3078 non-zero); noise ≤ 3.3% |
| E-ROBUST-008 | DOES NOT SUPPORT | VERIFIED | D-102 CLOSED-VERIFIED | per-unit claim withdrawn; 0.1877 vs 0.1536, seed ranges 0.1813-0.1922 vs 0.1376-0.1635, CTRL 0.1186 vs 0.1186, T4c profile = my table; attribution labelled INFERENCE |
| E-ROBUST-010 | VERIFIED | VERIFIED | (D-110c) | addendum λ0 base 0.2614 / 0.1966 matches my summaries; deltas −0.074 / −0.078 / −0.103 / −0.014 correct |
| E-ROBUST-018 | VERIFIED | VERIFIED | | `--l_level` note added, matches generate_confs.py:107-110 |
| E-ROBUST-043 | WRONG LOCATOR | VERIFIED | D-103 CLOSED-VERIFIED | p. 4, §4.1 (starts p. 3) |
| E-ROBUST-046 | WRONG LOCATOR | VERIFIED | D-104 CLOSED-VERIFIED | p. 8, §4.2 |
| E-ROBUST-047 | WRONG LOCATOR | VERIFIED | D-105 CLOSED-VERIFIED | p. 9, §5 Discussion |
| E-ROBUST-049 | WRONG LOCATOR | VERIFIED | D-106 CLOSED-VERIFIED | p. 5 (v6), §4 before 4.1 |
| E-ROBUST-058 | WRONG LOCATOR | VERIFIED | D-107 CLOSED-VERIFIED | dataset.py:145-162, quote :145-147 |
| E-ROBUST-060 (new) | - | VERIFIED | D-109 | T5 = my round-1 summaries: CTRL_rematch 0.1786 (0.1773-0.1800, n 935), B1 0.2341 (0.2299-0.2364, n 935), B1 gtLcycle 0.0212 (0.0206-0.0215, n 996); gaps 0.003 / 0.012 / 0.028 Å correct |
| E-ROBUST-061 (new) | - | VERIFIED | D-109 | ranges 0.0009 / 0.0027 / 0.0065 / 0.0009 / 0.0011 Å, i.e. 0.001-0.007 Å as stated |

Final counts over 61 entries: VERIFIED 61; WRONG LOCATOR, QUOTE MISMATCH, DOES NOT SUPPORT, UNVERIFIABLE 0.
Threads D-101..D-110: 10 CLOSED-VERIFIED, 0 CLOSED-REJECTED.

### Final card rulings
- **C-ROBUST-01: SUPPORTED** (was WEAKENED). All seven flagged sentences are fixed: the ratios are now 1.60× / 0.84×
  against matched RDKit L, the B1 observation is stated as unpaired and different-profile, the λ = 0 base is used
  (0.197 → 0.119), the like-for-like in-job B1 values are used (0.234 / 0.021), the CDM matching rule and the factorised
  rationale are labelled INFERENCE, and ProteinMPNN is hedged. The ORACLE Arm-0 gate is described as experiment
  selection and left to the validity judge. Residual nit, not blocking: §2 still says CDM augmentation "removes" the
  mismatch, where the source says "alleviates" (E-023).
- **C-ROBUST-02: SUPPORTED** (unchanged). The 9% figure is gone, "infers the regime" is labelled INFERENCE, and the
  `--l_level` caveat is in §5.
- **C-ROBUST-03: SUPPORTED** (unchanged). "Halves" is replaced by about 40% / 30%, which I checked (0.0182 / 0.0314 =
  0.58; 1.973 / 2.796 = 0.71), and the MMFF-trained hypothesis is labelled.
- **C-ROBUST-04: SUPPORTED** (unchanged; locators fixed).
- **C-ROBUST-05: SUPPORTED** (was WEAKENED). The E-012 noise argument and the Huszár "real risk" sentence are
  withdrawn. Gaps are now the like-for-like 0.003 / 0.012 Å, endpoint seed spread is cited from E-061, and the trade-off
  and drift readings are labelled INFERENCE. The NO now rests on headroom, an inferred trade-off and cost. That is a
  judgement for the panel, not a factual error.
