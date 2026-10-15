# Round 3 P4 panel, Delphi round 1: ENGINEERING judge

Written 2026-10-11 by the engineering judge. Round 1: scored independently; no other judge's file was read.
Lens: cost, feasibility, reuse of round-2 machinery, schedule on gnode118 (4× RTX 3090, shared node, 48 CPUs,
plafnet2 5000 MB/CPU cap), failure modes and implementation risk.

Inputs read: `orchestration/README.md`, `roles/judge.md`, `templates/scorecard.md`, `round3/BRIEF.md`, all 17 cards
(§1–§8, latest "Revised after cross-read" blocks taken as current), `round3/grounding.md` (full), `round3/impact.md`
(full), `ledger/verify_V1.md`, `verify_V2.md`, `verify_V3.md` (card rulings and round-2 verdicts), `ledger/disputes.md`
(D-205…D-210). Only VERIFIED entries are used: V1 59/59, V2 61/61, V3 all VERIFIED after round 2, except that the split
statement in E-GEOM-054 is contested (D-210 OPEN). Engineering spot-checks I made myself (read-only, local repo):
`torsional-diffusion/utils/xtb.py:5-7` sets `my_dir = /tmp/<pid>` at import, and `tools/make_l_seed_pickles.py:309`
builds a `Pool`, so the S-1 race the grounder describes is real; `slurm/r2_eval_array.sbatch:23` takes an overridable
`R2_EVALSETS`. Round-2 operational facts are from `round2/RUNLOG.md` and `round2/review_3.md`.

## 0. Scoring conventions (engineering judge)
- Cost basis (grounding §2.2, measured): training 11.7 GPU-h per 100-epoch run; packed condition 0.17 GPU-h; unpacked
  0.26 GPU-h; GFN2-xTB 0.14 s per QM9 seed (0.3 s incl. overhead).
- **Expected gain for analysis/metric cards (C-EVAL-01…05, C-ROBUST-01 Arm 0)** cannot move AMR-R by construction. I
  score them on the decision they unlock, capped at 3, so that they neither lose to nor beat method cards on this line.
- **Cost** 5 = ≤ 1 GPU-h and ≤ ~200 new lines, no change inside TD; 4 = ≤ 5 GPU-h and ≤ ~500 new lines, or a
  zero-code config run of ≤ 5 GPU-h; 3 = ≤ 5 GPU-h but ≥ 2 days of new code or a separate software environment;
  2 = 25–40 GPU-h; 1 = > 50 GPU-h or weeks of code. Shared tooling (grounding S-1…S-6, ≈ 600–700 lines) is charged
  once, to C-GEOM-01, the first card that needs it; the EVAL cards that reuse it are scored on their own increment.
- **Risk** (5 = low): chance of failing for reasons unrelated to the idea (bugs, silent failures, confounds, power).
- My **Worth trying** follows the panel rule applied to my own scores: YES needs total ≥ 21 (mean ≥ 3.5), no blocker,
  and feasibility this round; gated or conditional cards are MAYBE; NO otherwise. Two cards are split by arm because
  their arms differ by an order of magnitude in cost (C-EVAL-04, C-ROBUST-01).

## 1. Summary table

| Card | Evid. | Relev. | Gain | Cost | Risk | FT value | Total /30 | Worth trying | GPU-h (my budget) | CPU-h | Validity blocker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C-GEOM-01 xTB seeds | 5 | 5 | 3 | 4 | 4 | 5 | **26** | **YES** (do not gate the GPU step) | 3.7 | ~7.5 | NONE (S-1 fixes are a precondition; they are in §7) |
| C-GEOM-02 ring templates | 4 | 5 | 4 | 3 | 3 | 5 | **24** | **YES** (CPU-gated) | 3.2 | 5.5–9.5 | NONE (leakage assert + mirror check; in §7) |
| C-EVAL-04 B-mmff + B-xtb | 4 | 4 | 3 | 5 | 4 | 4 | **24** | **YES** | 0 | ~7 | NONE |
| C-EVAL-01 literature block / L-error fit | 4 | 3 | 3 | 5 | 4 | 4 | **23** | **YES** | 0 (shares C-EVAL-02 d) | 1–2 | NONE |
| C-EVAL-05 xTB-after yardstick | 4 | 4 | 3 | 5 | 3 | 4 | **23** | **YES** | 0 | ~11 | NONE (pre-register drop accounting) |
| C-EVAL-02 diagnostics + random-GT runs | 4 | 4 | 3 | 5 | 3 | 4 | **23** | **YES** | 0.85 | 2–4 (unmeasured) | NONE |
| C-ROBUST-01 Arm 0 (S3/S2 on A5, λ0.75) | 3 | 4 | 3 | 4 | 4 | 3 | **21** | **YES** | 2.55 | 0 | NONE (ORACLE gate = experiment selection; validity judge to confirm) |
| C-GEOM-06 srETKDG seeds | 3 | 4 | 2 | 5 | 4 | 3 | **21** | **YES** (CPU; GPU only via its gate) | ≤ 1.0 (gated) | ~3 | NONE (use the §7 single-factor flags) |
| C-EVAL-03 xTB E_relax metrics | 4 | 3 | 2 | 5 | 3 | 3 | 20 | MAYBE (test-side metrics come free from C-EVAL-05's run) | 0 (+0.5 val. half) | +1–2 (8 standalone) | NONE |
| C-GEOM-03 ET-Flow L | 3 | 4 | 3 | 3 | 2 | 3 | 18 | MAYBE, blocked | ≤ 1 + 1.0–1.9 | ~1 + env | **YES: D-210 OPEN** |
| C-ROBUST-03 S3-MMFF | 3 | 3 | 2 | 2 | 4 | 3 | 17 | MAYBE (measured S5 gate first) | 26.1 (2 seeds), ~39 (3) | ≤ 4 h wall at 14 CPUs | NONE |
| C-ROBUST-01 Arm 1 (4-way factorised training) | 2 | 4 | 2 | 2 | 3 | 3 | 16 | MAYBE (gated on Arm 0 and C-GEOM-02) | 26.8 (2 seeds), ~40 (3) | ~2 | NONE |
| C-EVAL-04 B-clust + TD+cluster | 3 | 2 | 2 | 4 | 2 | 2 | 15 | MAYBE (defer) | 2.6 | 2–5 | NONE (pre-register the K-means space) |
| C-GEOM-04 learned pucker component | 3 | 4 | 1 | 1 | 1 | 3 | 13 | NO (this round) | — | — | NONE |
| C-GEOM-05 ML-potential relaxation | 3 | 3 | 1 | 3 | 2 | 1 | 13 | NO | — | — | NONE |
| C-ROBUST-02 blind λ | 2 | 3 | 1 | 2 | 3 | 2 | 13 | NO (unless the measured gate passes; then ≥ 3 seeds) | 26.5 (+1.0) | ~0 | NONE |
| C-EVAL-06 DFT re-ref / ensemble props | 3 | 1 | 1 | 3 | 2 | 2 | 12 | NO | — | — | NONE |
| C-ROBUST-04 L guidance / score interp. | 2 | 2 | 1 | 4 | 2 | 1 | 12 | NO | — | — | NONE |
| C-ROBUST-05 p_gt sweep / curriculum | 2 | 2 | 1 | 1 | 4 | 2 | 12 | NO | — | — | NONE |

**YES set (engineering):** C-GEOM-01, C-GEOM-02, C-EVAL-01, C-EVAL-04 (B-mmff, B-xtb), C-EVAL-05, C-EVAL-02,
C-ROBUST-01 Arm 0, C-GEOM-06 (CPU part; GPU through its gate).
**Cost of the YES set (section 3):** ≈ 10.3–11.9 GPU-h (≈ 12.4–14.3 with a 20 % rerun reserve for co-located OOMs)
and ≈ 40–50 CPU-h. Code is ≈ 600–700 lines of shared tooling plus ≈ 900–1,000 lines of card code. That coding takes
≈ 5–6 working days and dominates wall time. None of it needs a new training run.
**Blocker:** C-GEOM-03 (D-210 OPEN). Gates to compute on CPU now, before P5: C-ROBUST-02 (S4 vs S3 at MMFF and λ 0.5)
and C-ROBUST-03 (S5 ≤ 0.150); both read existing `eval.pkl` files and cost minutes.

## 2. Per-card scores and reasons
Highest total first. "§n" = card section; "S-n" = shared piece in `grounding.md` §2.3.

### C-GEOM-01: GFN2-xTB-relaxed ETKDG seeds as test-time L
Evidence 5 · Relevance 5 · Expected gain 3 · Cost 4 · Risk 4 · FlexiTors value 5 · **Total 26** · **Worth trying: YES**

Reason: it has the best ratio of decision value to cost in the round. The core is VERIFIED: the references are
gas-phase GFN2-xTB minima [E-GEOM-015/016], and GFN2 re-optimisation of the references is ≈ null [E-GEOM-018/019]. The
grounder measured what used to be the main engineering unknowns (§7). The xtb 6.7.1 module binary runs on gnode118 with
no install. It takes a median 0.135 s per optimisation, 20 of 20 calls returned rc 0, and GT conformers stay at
0.000 Å. So the seeds cost ≈ 5 CPU-h (< 30 min wall on 16 cores) and the GPU step ≈ 3.7 GPU-h, even with both xTB
variants and S4 at two λ values (section 3). Reuse is high:
- the builder's F5 loop gives `l_error.csv` rows for free;
- `r2_eval_array.sbatch` takes an overridable `R2_EVALSETS`;
- the provenance guard, the CR/B1/S3/S4 checkpoints and every ETKDG/MMFF control already exist.

The new code is ~120 builder lines plus the shared S-1/S-2/S-3 (~360 lines, 5 tests), which C-EVAL-03/04/05 and
C-GEOM-02/06 then reuse. Risk is 4, not 5, because four silent-failure modes exist. Each has a concrete, testable
mitigation in §7:
1. **Temp-dir race.** `utils/xtb.py:5-7` fixes `/tmp/<pid>` at import, and the builder forks a `Pool` (`:309`), so
   molecules with the same atom count can swap coordinates. The two-worker collision test is mandatory.
2. **Hidden fallbacks.** Seeds that fall back unrelaxed would pass as xTB L. Report the fallback share and require
   ≥ 95 % relaxed.
3. **Provenance.** Writing into `round2_seeds` breaks the provenance hashes. Use the new `round3_seeds` dir, and assert
   that the regenerated `L_etkdg2L` is bit-identical to round 2's.
4. **S4 λ.** There is no λ-on-validation path (S-5). Pre-declare λ = 0 and λ = 1, as impact §5 C2 asks.

Engineering recommendations:
- Do not gate the GPU step; the gate costs more decision time than the GPU (impact §4.3). Use the gate metrics, plus
  the s = seed COV-R@0.05–0.1 metric (impact §3g), to interpret the result.
- Run **both** `L_etkdg2L_xtb` and `L_etkdg2L_mmff_xtb` (+1.4 GPU-h) rather than picking one by test-GT gate metrics.
  Pre-declare `_xtb` as primary.

Expected gain is 3: CTRL is −0.008 central and borderline against the ±0.005 paired half-width, and the best arm (S3)
is 0.10–0.16 (§8 revision). The value is 5, because either outcome decides whether non-learned L can feed
true-L-trained models.
**Validity blocker: NONE**, provided the S-1 fixes and tests listed in §7 are implemented. Built on TD's
`utils/xtb.py` unchanged, the results would be invalid.

### C-GEOM-02: training-set ring templates
Evidence 4 · Relevance 5 · Expected gain 4 · Cost 3 · Risk 3 · FlexiTors value 5 · **Total 24** · **Worth trying: YES**

Reason: it has the largest predicted non-oracle gain that can be detected with the existing checkpoints: CTRL ≈ 0.14,
i.e. −0.037 vs ETKDG and −0.012 vs MMFF (§8). It aims straight at the ring tail. Its mechanism is VERIFIED but partly
by analogy: OMEGA's templates [E-GEOM-038/044], few pucker clusters and a 0.09 Å knowledge-based sampler
[E-GEOM-042/043], and `coordMap` [E-GEOM-039]. OMEGA's QM9 edge [E-GEOM-025] is a labelled inference. It is the most
code in my YES set: ~450–550 lines + 3 tests, 2–3 days (§7), so Cost is 3. GPU is cheap: ≈ 3.2 GPU-h for `L_tmpl` and
`L_tmpl_xtb`.

Reuse lowers the risk. The transplant is the A5ring recipe (`lgeom.set_internal_subset`,
`make_l_seed_pickles.py:236`), and a self-template round-trip test must reproduce A5ring's L error. Risk is still 3:
- the library pass over 133,232 raw pickles has unmeasured CPU and IO (2–6 CPU-h), so a per-key cap is needed;
- mirror-image templates need a stereo check plus a reflected retry, which the card's §5 does not have;
- `set_internal_subset` swallows failures (`lgeom.py:358-366`), so a residual threshold is needed;
- `coordMap` can fail on cages and fused 3-/4-rings;
- the key level needs either S-5 or a training-split leave-one-out choice.

The CPU gate catches poor coverage or poor fidelity before any GPU time. The card sits on the critical path after S-1,
S-3 and S-4.
**Validity blocker: NONE**, provided the leakage assert (no library source equals a test or validation SMILES) and
the stereo check are in place (§7).

### C-EVAL-04 (B-mmff + B-xtb): non-learned baselines
Evidence 4 · Relevance 4 · Expected gain 3 (cap) · Cost 5 · Risk 4 · FlexiTors value 4 · **Total 24** · **Worth trying: YES**

Reason: almost free, and §8 rates the decision value high: without these baselines the torsion model's contribution
on good L cannot be attributed.
- **B-mmff** needs zero code (`--no_model --no_random --pre_mmff`, §7): ≈ 1 CPU-h for 3 seeds.
- **B-xtb** is C-GEOM-01's `L_etkdg2L_mmff_xtb` evaluated with `--no_model --no_random`. Seed 1 is the same run as the
  C-GEOM-01 gate's seed-only AMR-R over all molecules (the analyst's s metric), so it costs nothing extra. Seeds 2–3
  cost ≈ 4 CPU-h more.

Evidence: E-EVAL-012/013/020/021, plus our own A0/A2 rows E-EVAL-038–040, all VERIFIED. Risks: B-xtb inherits the S-1
and S-2 risks, and the 60 ETKDG failures mean the comparison must use the intersection. Fix V1's R2-2 slip (25 % vs
30 %) before reporting.
**Validity blocker: NONE.**

### C-EVAL-01: literature reporting block and "how good must L be"
Evidence 4 · Relevance 3 · Expected gain 3 (cap) · Cost 5 · Risk 4 · FlexiTors value 4 · **Total 23** · **Worth trying: YES**

Reason: pure CPU on existing `eval.pkl` files and `l_error.csv` (1–2 CPU-h). It adds ~200 lines on top of
`paired_compare.py` [E-EVAL-037], plus a synthetic known-slope test (§7). The literature rows E-EVAL-001…008 are
VERIFIED (V1 round 2). Each engineering risk is named and testable:
- a raw-vs-corrected key mis-join (assert ≥ 99 % join coverage);
- a non-homogeneous L-error statistic (ETKDG/MMFF rows are Hungarian-assigned; §7 change 2);
- λ seed reuse in 258 molecules (fit on the 648 clean molecules; §8 revision 2);
- the cycled, leaky λ axis (§8 revision 1).

Its optional CR random-GT runs are the same three runs as C-EVAL-02 (d), so budget them once. V1's R2-1 wording must
be fixed. The value is 4 because the card now states that the state-of-the-art band needs own-basin L.
**Validity blocker: NONE.** Label the λ-based fits ORACLE.

### C-EVAL-05: xTB-after yardstick
Evidence 4 · Relevance 4 · Expected gain 3 (cap) · Cost 5 · Risk 3 · FlexiTors value 4 · **Total 23** · **Worth trying: YES**

Reason: CPU only, about 2.1 CPU-h per arm on the full set and ≈ 11 CPU-h for 5 arms. Every input `confs.pkl` already
exists (§7), and the evaluator is unchanged because atom order is preserved. TD's `optimize_confs.py` has four defects
(hard-coded DRUGS csv at `:24`; `--limit 32` truncation; the `if xtb_energy:` flag bug; uncounted drops). I recommend
the S-1 runner rather than patching it, so that one xTB code path serves all cards. Configure S-1 to record E(input),
E(opt) and displacement per conformer: C-EVAL-03's test-side metrics then fall out of this run.

Risk is 3 for three reasons. Dropped-conformer accounting changes recall and must be pre-registered (§7). Relaxation
merges samples, so the dedup count is needed. Graph changes need S-2. The evidence is VERIFIED (E-EVAL-018/020/022/023/
035; A3 is our data, E-EVAL-041). The decision value is high either way (§8).
**Validity blocker: NONE**, once the drop rule is pre-registered.

### C-EVAL-02: local-geometry, stratified and conformer-level diagnostics
Evidence 4 · Relevance 4 · Expected gain 3 (cap) · Cost 5 · Risk 3 · FlexiTors value 4 · **Total 23** · **Worth trying: YES**

Reason: (a)–(c) are ~80 lines on `geometry_metrics.py` and the `eval.pkl` matrices. Item (d) is 5 packed runs,
≈ 0.85 GPU-h; the card said 0 (§7). It is config only: `--seed_confs $QM9_GT_SEED_CONFS` without `--seed_confs_cycle`,
for S3 s0/s1 and CR s0–s2. Per GPU-h it is the most informative run in the round: impact §3g / §8 says the S3/CR
penalty for non-own L decides how to read C-GEOM-01's B1 and S3 results. It can run today.

Risk is 3:
- `geometry_metrics.py` has never had a production run, and its runtime is unmeasured (up to 200 substructure maps
  per molecule). Time one arm before scheduling twelve.
- (a) is not exactly constant for arms that share L.
- The (c) pairing needs a check: conformer-k bond lengths must agree to 1e-5 Å.
- Strata must come from TD's `edge_mask`, not `breakdown.py` (D-205).

Evidence E-EVAL-009/019/036/047/049/057 is VERIFIED.
**Validity blocker: NONE.** Report random-GT with the §3g inversion, not as a ceiling.

### C-ROBUST-01 Arm 0: S3 and S2 on the A5 hybrids and λ 0.75 (inference only)
Evidence 3 · Relevance 4 · Expected gain 3 (cap) · Cost 4 · Risk 4 · FlexiTors value 3 · **Total 21** · **Worth trying: YES**

Reason: config only: a new set label and a models TSV for S3 s0/s1 and B6_jit0.04pa s0–s2. That is 15 packed runs,
≈ 2.55 GPU-h (§7). These tags do not exist yet for these models, so the provenance guard creates them cleanly. The run
measures S3's response to partial L, which no round-2 panel ran [E-ROBUST-013]. The analyst needs it to narrow the S3
bracket on template L (§8). It can run today, with no code risk.
**Validity blocker: NONE** from engineering. The validity judge should confirm that an ORACLE-based go/no-go for an
*experiment* is acceptable under README rule 5 (V2 note).

### C-GEOM-06: srETKDGv3 seeds as a ring-L control
Evidence 3 · Relevance 4 · Expected gain 2 · Cost 5 · Risk 4 · FlexiTors value 3 · **Total 21** · **Worth trying: YES** (CPU part; GPU only via its gate)

Reason: ~30 lines (`--etkdg_params` in the builder) and ≈ 3 CPU-h including the gate. It runs in the same builder job
as C-GEOM-01 and gives the free baseline every ring-L source must beat. The grounder measured an engineering
correction: in RDKit 2022.9.5, `srETKDGv3()` also switches macrocycle torsions off. A true single-factor v3 → v3sr
contrast therefore needs `ETKDGv3()` with `useSmallRingTorsions = True` (§7).

The expected effect is small [E-GEOM-022/023]. The variants are different embeddings (the same `randomSeed` does not
give common random numbers), so the MDE is ≈ 0.008–0.010 (§8). The gated GPU step (≤ 1 GPU-h) will therefore probably
be underpowered. This is why Gain is 2 and the GPU part stays behind the ≥ 5-point tail gate.
**Validity blocker: NONE**, if the §7 flags are used. §5 still names `srETKDGv3()` for the "small-ring only" step;
that wording should follow §7.

### C-EVAL-03: GFN2-xTB relaxation metrics (E_relax, displacement)
Evidence 4 · Relevance 3 · Expected gain 2 · Cost 5 · Risk 3 · FlexiTors value 3 · **Total 20** · **Worth trying: MAYBE**

Reason: cheap once S-1/S-2 exist: ≈ 8 CPU-h standalone, against the card's obsolete 22–67 CPU-h (§7). Its selling
point is GT-free selection on validation molecules (§3). That needs S-5, which does not exist (~70 lines), plus
≈ 0.3–0.5 GPU-h of new validation inference. The analyst also notes it is blind to the pucker tail that drives CTRL
(§8: medium value). The test-side metrics come almost free as a by-product of C-EVAL-05's relaxation run if S-1
records E(input), E(opt) and displacement: +1–2 CPU-h for single points and the GT zero-check arm. Run them that way.
Add the validation half only when S-5 is built for other reasons.

Risk is 3, from the same S-1 failure modes: energies of another molecule, unconverged structures counted as minima,
graph changes, and cage outliers (report medians). Evidence E-EVAL-012/013/018–023/055/056 is VERIFIED.
**Validity blocker: NONE.**

### C-GEOM-03: ET-Flow `qm9-o3` L as a diagnostic
Evidence 3 · Relevance 4 · Expected gain 3 · Cost 3 · Risk 2 · FlexiTors value 3 · **Total 18** · **Worth trying: MAYBE (blocked)**

Reason: D-210 is OPEN, and engineering adds that the check will probably be inconclusive. The cluster holds no
`summary_qm9.json` and no ET-Flow file that could map `QM9.zip`'s indices to SMILES (§7). Downloading `QM9.zip` also
needs the user's explicit permission. If the check clears, the build is feasible:
- a separate micromamba env (numpy 1.26.4 pin; torch 2.x on cu118 or cu121; driver 580 and internet both checked);
- ~200 lines, with an npz hand-off (no cross-version RDKit pickles) and an atom-mapping round-trip test;
- ≤ 1 GPU-h of ET-Flow sampling (unmeasured) + 1.0–1.9 GPU-h of inference, and about half a day of env work.

Risk is 2 (split provenance, env, atom mapping). C-GEOM-01 answers most of the same question at the reference level
without the leak risk (§8).
**Validity blocker: YES. D-210 OPEN** (`qm9-o3` train/val vs TD test overlap not shown). Stop unless step 0(a) shows
zero SMILES overlap.

### C-ROBUST-03: sampler-matched mixture (S3-MMFF)
Evidence 3 · Relevance 3 · Expected gain 2 · Cost 2 · Risk 4 · FlexiTors value 3 · **Total 17** · **Worth trying: MAYBE** (measured S5 gate first)

Reason: the lowest implementation risk of the three training cards. TD needs zero code changes:
`build_paired_pickles.py` reuses the pairing check (§7), and each slurm file gets +2 lines. The round-2 analogue (job
10122: pairing plus all seeds, `round2/review_3.md:84`) took 3 h 39 min on 14 CPUs with MaxRSS 11.2 GB, which bounds
the CPU step. There is one silent failure: `ConformerDataset` reuses any cache found at the path (`dataset.py:124-129`).
Use a new cache name and assert MMFF-level bond error in the RDKit half.

Against that, it costs ≈ 26 GPU-h for 2 seeds for an expected 0 to −0.006 Å (§8), and detecting −0.005 needs 3 seeds
(≈ 39 GPU-h). Mechanism evidence is TD's own conformer matching [E-ROBUST-020–022], with DAgger as a labelled analogy.
The S5 gate can be computed today on CPU from B3 s0–s2 and CR `S5ctrl`. If it fails, the recipe is better spent on
xTB-matched training after C-GEOM-01. INFERENCE: one xTB pass over the training conformers is about 82,895 molecules ×
~6.8 conformers × 0.3 s ≈ 47 CPU-h per relaxed copy (grounding counts). That is feasible on CPU; not measured.
**Validity blocker: NONE.**

### C-ROBUST-01 Arm 1: four-way factorised training
Evidence 2 · Relevance 4 · Expected gain 2 · Cost 2 · Risk 3 · FlexiTors value 3 · **Total 16** · **Worth trying: MAYBE** (gated)

Reason: the mechanism is analogy-only (CDM, SliDe, ProteinMPNN; E-ROBUST-023/025/034/031), so Evidence is ≤ 2.
Implementation is ~200 lines inside TD: the hybrid post-pass, an `l_mix4` transform branch, flags, slurm cases and C7
test extensions (§7). It also needs a new ≈ 2.1 GB cache. Round-1 MaxRSS was 15–24 GB with 8 workers, against a
30 GB limit at 6 CPUs × 5000 MB, so a smoke-run MaxRSS check is needed.

Training costs ≈ 27 GPU-h for 2 seeds. That is underpowered for the +0.005 non-inferiority margin (≥ 3 seeds,
≈ 40 GPU-h; §8), and the expected primary gain is ≈ 0. Silent-failure modes:
- `set_internal_subset` swallows failures;
- `SetAngleDeg` shifts torsions by ~3–4°;
- collation breaks if the hybrid lists are not deleted after selection.

The analyst also notes that the four-way mix never shows *other-conformer* GT L, which is what xTB and template
sources deliver (§8 revision 1). Run it only if Arm 0 fails its gate and C-GEOM-02 shows that template L is worth
deploying.
**Validity blocker: NONE.** Pre-declare the residual threshold.

### C-EVAL-04 (B-clust + TD+cluster)
Evidence 3 · Relevance 2 · Expected gain 2 · Cost 4 · Risk 2 · FlexiTors value 2 · **Total 15** · **Worth trying: MAYBE** (defer)

Reason: it needs ~150 lines (`cluster_baseline.py`), 3 lines for `--confs_factor` and ≈ 2.6 GPU-h; memory and time at
10× samples per molecule are unmeasured. The result depends on a K-means feature space that the evidence does not fix:
E-EVAL-025 quotes only "K-means with 2 N_ref clusters". Zhou's numbers are on the GeoDiff protocol [E-EVAL-026]. It
is a recall-budget sanity check, so run it only when a round-3 arm claims a recall gain.
**Validity blocker: NONE**, if the feature space is pre-registered.

### C-GEOM-04: learned ring-pucker component
Evidence 3 · Relevance 4 · Expected gain 1 · Cost 1 · Risk 1 · FlexiTors value 3 · **Total 13** · **Worth trying: NO** (this round)

Reason: a joint Cremer–Pople factor touches the score head, the transform and the sampler (§7). That is ≥ 300 lines
plus a training run per seed [E-GEOM-049], i.e. weeks of work. The in-domain all-atom gain is small [E-GEOM-023], and
half of QM9's tail is off-domain [E-GEOM-011]. The two-stage reopen route needs zero TD lines (one more seed pickle
built with C-GEOM-02's code), and its reopen criteria are computable with the S-3 tool, so keep that path for round 4.
**Validity blocker: NONE.**

### C-GEOM-05: ML-potential relaxation
Evidence 3 · Relevance 3 · Expected gain 1 · Cost 3 · Risk 2 · FlexiTors value 1 · **Total 13** · **Worth trying: NO**

Reason: C-GEOM-01 dominates it on every engineering axis:
- no speed case, since xTB takes 0.14 s per seed;
- a separate environment (AIMNet2 needs python ≥ 3.11 and torch ≥ 2.8; MACE pins e3nn 0.4.4 against TD's 0.5.1
  [E-GEOM-036]);
- it relaxes toward DFT minima rather than the GFN2 references (E-GEOM-015/027; inference).

If it is ever wanted, the S-1/S-3 pipeline can be reused with ~60 lines (§7).
**Validity blocker: NONE.**

### C-ROBUST-02: blind λ interpolation
Evidence 2 · Relevance 3 · Expected gain 1 · Cost 2 · Risk 3 · FlexiTors value 2 · **Total 13** · **Worth trying: NO** (unless the measured gate passes)

Reason: zero code (a config line), but ≈ 26.5 GPU-h (+1.0 for S4x controls) for an equivalence test that 2 seeds
cannot power; 3–6 are needed (§8). The mechanism is an image-domain analogy whose concentration result depends on
dimension [E-ROBUST-037/038], which argues against QM9. The λ = 1 branch of its own gate already fails (S4 0.043 vs S3
0.034, BRIEF). The rest of the gate is computable on CPU today from S4core/S1panel `eval.pkl`, so run it. The
deployment question is answered for ≈ 1 GPU-h inside C-GEOM-01 (S3 vs S4 on xTB L).
**Validity blocker: NONE.**

### C-EVAL-06: DFT re-referencing / ensemble properties
Evidence 3 · Relevance 1 · Expected gain 1 · Cost 3 · Risk 2 · FlexiTors value 2 · **Total 12** · **Worth trying: NO**

Reason:
- (a) needs DFT on the cluster. The ORCA 6.1.1 module is listed on ada but untested on gnode118, and the r2SCAN-3c
  cost is unmeasured (§7). It would also break comparability with E-EVAL-001…008.
- (b) needs a new aggregation script, `NO_ENERGY=0` and `--ode --likelihood`. That combination drops QM9 molecules
  with 0 rotatable bonds and changes the population (§7).

Neither informs a round-3 decision (§8).
**Validity blocker: NONE** (not run).

### C-ROBUST-04: L guidance / CTRL–B1 score interpolation
Evidence 2 · Relevance 2 · Expected gain 1 · Cost 4 · Risk 2 · FlexiTors value 1 · **Total 12** · **Worth trying: NO**

Reason: the guided variant has no code path. TD tracks torsions only relative to its own seed
(`sampling.py:290-298`), so no torsion state can transfer between L seeds (§7). The interpolation diagnostic is ~40
lines and 3–5 GPU-h at 2× forward cost, for a ceiling equal to CTRL on every non-oracle L (§8). It is also a
non-principled mixture [E-ROBUST-049].
**Validity blocker: NONE.**

### C-ROBUST-05: S3 ratio sweep / curriculum
Evidence 2 · Relevance 2 · Expected gain 1 · Cost 1 · Risk 4 · FlexiTors value 2 · **Total 12** · **Worth trying: NO**

Reason: zero code, but ≈ 73 GPU-h (6 trainings) for ≤ 0.003 Å of headroom on the primary endpoint
[E-ROBUST-004/060]. Detecting that needs ≥ 4 seeds per arm (§8). The useful by-product is free: S3's seed spread is
computable today from the two finished S3 seeds (§7 data note).
**Validity blocker: NONE.**

## 3. Proposed schedule for the YES set
Basis: packed condition 0.17 GPU-h (a 4-condition task ≈ 0.71 h wall, 2-condition ≈ 0.4 h); a single-condition task
runs unpacked at 0.26 GPU-h; xTB 0.3 s per conformer including overhead, `OMP_NUM_THREADS=1`, 16 CPUs per CPU job.
Wall times assume the 4 GPUs are free; on a shared node, queueing dominates. **No YES item needs a training run.**

### 3.1 Waves

| Wave | Earliest start | Depends on | Content (model tasks × conditions) | GPU-h | Wall (4 GPUs / 16 CPUs) |
|---|---|---|---|---|---|
| **G1** (GPU) | day 0, no new code | `r3` set label + models TSV only | C-ROBUST-01 Arm 0: S3 s0/s1, B6 s0–s2 × {A5ring, A5acyc, λ0.75} (15). C-EVAL-02 (d) = C-EVAL-01 optional: random-GT `--seed_confs $QM9_GT_SEED_CONFS` (no cycle) for S3 s0/s1, CR s0–s2 (5). 8 tasks, 20 conditions | 3.4–3.7 | ≈ 1.0 h |
| **C1** (CPU) | day 0 | existing `eval.pkl`, `l_error.csv` | C-EVAL-01 analysis; **C-ROBUST-02 and C-ROBUST-03 gates** (S4 vs S3 at MMFF / λ 0.5 on S4's subset; S5 ≤ 0.150); S3 seed spread; B-mmff ×3 (`--no_model --no_random --pre_mmff`, `CUDA_VISIBLE_DEVICES=""`); time `geometry_metrics.py` on one arm | 0 | ≈ 3–4.5 CPU-h, < 1 h |
| **B1** (build) | days 0–2 | none | S-1 xTB runner (4 tests: GT pass-through, two-worker collision, charged molecule, non-convergence fallback; records E(in), E(opt), displacement); S-2 graph/stereo filter (stretched-bond test); S-4 `round3_seeds`, `QM9_SEEDS3`, `r3_evalsets.tsv`; S-3 gate tool (pucker coverage, common rigid set from `edge_mask`, seed AMR-R, s metric); builder `--xtb_bin` and `--etkdg_params {v1,v3,v3sr}` (v3sr = `ETKDGv3()` + `useSmallRingTorsions`); S-6 round-3 analysis driver with pre-declared contrasts, **committed before any result** | 0 | ≈ 2 days coding |
| **C2** (CPU) | day ~2 | B1 | C-GEOM-01 seeds `--sources etkdg --seed 1 --xtb_bin` (bit-identity assert vs round-2 `L_etkdg2L`) ≈ 5; C-GEOM-06 variants + MMFF twins ≈ 2; B-xtb builder seeds 2–3 ≈ 4.2; S-3 gate on all seed sets incl. seed-only AMR-R over all molecules (= B-xtb seed 1, s metric) ≈ 2.3 | 0 | ≈ 13.5 CPU-h, ≈ 1 h |
| **G2** (GPU) | right after C2, **not gated** | C2 (fallback share ≥ 95 %) | C-GEOM-01: `L_etkdg2L_xtb` (primary) × {CR ×3, B1 ×3, S3 ×2, S4 ×3 λ = 0, S4 ×3 λ = 1} + `L_etkdg2L_mmff_xtb` × {CR ×3, B1 ×3, S3 ×2}. 11 tasks, 22 conditions. If C-GEOM-06's gate passes (tail −5 points), add CR ×3 on ≤ 2 variants (+≤ 6 conditions, ≤ 1.0 GPU-h) to the CR tasks | 3.7 (+≤ 1.0) | ≈ 1.2 h |
| **C3** (CPU) | day ~3 | G1, G2 outputs; S-1, S-2 | C-EVAL-05 xTB-after on R0 s0, A2, CR s0 / S3 s0 / B1 s0 on ETKDG L ≈ 11 (+1–2 for C-EVAL-03's test-side single points and GT zero-check arm, if the panel keeps it); C-EVAL-02 (a)–(c) on every arm ≈ 2–4 (unmeasured); B-xtb evaluation ×3 ≈ 1 | 0 | ≈ 14–18 CPU-h, ≈ 1–1.5 h |
| **B2** (build) | days 2–5, parallel with C3 | S-1, S-3, S-4 | C-GEOM-02 library + transplant + tests (leakage assert, self-template round trip vs A5ring, mirror/stereo); key level chosen by training-split leave-one-out coverage (no test data) | 0 | ≈ 2–3 days coding |
| **C4** (CPU) | day ~5 | B2 | Library (2–6, unmeasured; per-key cap), transplant ≈ 1, `L_tmpl_xtb` ≈ 2, gate ≈ 0.5 | 0 | ≈ 5.5–9.5 CPU-h, < 1 h |
| **G3** (GPU) | day ~5–6 | C4 gate (non-oracle go rule) | C-GEOM-02: {`L_tmpl`, `L_tmpl_xtb`} × {CR ×3, B1 ×3, S3 ×2} + S4 ×3 λ = 0 on `L_tmpl`. 11 tasks, 19 conditions | 3.2–3.5 | ≈ 1.2 h |
| Analysis | after each wave | S-6 | Paired bootstrap, pre-declared contrasts only | 0 | ≈ 3 CPU-h total |

### 3.2 Totals for the YES set
- **GPU: ≈ 10.3–11.9 GPU-h.** G1 3.4–3.7 + G2 3.7 + G3 3.2–3.5 + C-GEOM-06 0–1.0 (gated). With a 20 % reserve for
  OOM reruns: **≈ 12.4–14.3 GPU-h**. That is < 5 % of the ~300 GPU-h planning budget. GPU wall ≈ 3.5 h in total if
  the GPUs are free.
- **CPU: ≈ 40–50 CPU-h** (C1 3–4.5, C2 13.5, C3 14–18, C4 5.5–9.5, analysis 3). The two unmeasured terms are
  `geometry_metrics.py` and the template library. Each CPU wave finishes in ≤ 1.5 h wall at 16 CPUs.
- **Code:** ≈ 600–700 lines of shared tooling (S-1…S-4, S-6) + ≈ 120 (C-GEOM-01 builder) + ≈ 30 (C-GEOM-06) + ≈ 200
  (C-EVAL-01) + ≈ 80 (C-EVAL-02) + ≈ 30 (C-EVAL-05 dedup) + ≈ 450–550 (C-GEOM-02). The wall-clock critical path is
  coding: B1 (≈ 2 days) → C2/G2 → B2 (≈ 2–3 days) → C4/G3, so **≈ 5–6 working days**.
- The grounder's 88–96 GPU-h for all YES/MAYBE cards (grounding §4) is ≈ 82 GPU-h of ROBUST trainings, which I do not
  put in the YES set.

### 3.3 Contingent wave G4 (MAYBE cards; user decision in P5, only after their gates)

| Arm | Trigger (measured, not predicted) | Seeds | GPU-h | CPU | Notes |
|---|---|---|---|---|---|
| C-ROBUST-03 S3-MMFF | C1: S5 (B3 ×3, `--pre_mmff`) ≤ 0.150 | 3 (2 underpowered for −0.005) | ≈ 35.1 + 4.1 ≈ 39 | pairing ≤ 4 h wall at 14 CPUs + featurize ≈ 1 h | new cache name + load-time assert; `PAIRING_CLEAN` ≥ 98 % |
| C-ROBUST-01 Arm 1 | G1 Arm 0 fails its gate **and** G3 shows template L worth deploying | 3 | ≈ 35.1 + 5.1 ≈ 40 | hybrids ≈ 1 + featurize ≈ 1 | smoke-run MaxRSS ≤ 30 GB at 6 CPUs |
| C-ROBUST-02 blind λ | C1: S4 beats S3 by ≥ 0.005 at MMFF or λ 0.5 | 3 | ≈ 35.1 + 4.6 ≈ 40 | 0 | no `--l_level` in generate fields |

If every trigger fires: ≈ 120 GPU-h, 9 trainings ≈ 3 rounds × ≈ 12 h on 4 GPUs (≈ 36–40 h wall plus evaluation),
using the round-2 `--resume` + requeue. If none fires, the better use of training budget is an xTB-matched S3 (the
C-ROBUST-03 recipe on C-GEOM-01's L). That is not a card yet; flag it to the scouts for round 4.

### 3.4 Failure modes and mitigations (schedule level)

| Risk | Evidence it happens | Mitigation |
|---|---|---|
| CUDA OOM caused by a co-located job | Twice in round 2: `RUNLOG.md` 10860 task 5 and 11176 (S4 s0; ours used 2 GB of 24) | Evaluation tasks are ≤ 1.2 h and tag-idempotent under the provenance guard: resubmit failed tags automatically once; 20 % GPU reserve; trainings use `--resume` + requeue |
| CPU contention on the shared 48-CPU node | Round 1 lost ≈ 10 GPU-h to it (`round2/vote_1.md` D5) | xTB with `OMP_NUM_THREADS=1`, 16 CPUs and a small `--mem-per-cpu`; do not overlap CPU waves with G4 training loaders |
| Memory cap of 5000 MB per CPU | Round-1 training MaxRSS 15–24 GB; data builder 11.2 GB at 14 CPUs | Arm 1 cache ≈ 2.1 GB: smoke MaxRSS before the array; xTB jobs need little memory |
| Silent xTB errors (coordinate swap, unconverged structures, graph change) | `utils/xtb.py:5-7` + `Pool` (`make_l_seed_pickles.py:309`); `optimize_confs.py` drops uncounted | S-1 / S-2 with tests; store `xtb_converged` and fallback counts; gate GPU use on ≥ 95 % relaxed seeds |
| Provenance break | `common.sh:111-114` hashes seed pickles | Write only to `round3_seeds`; extend the twin assert; never touch `round2_seeds` |
| Unmeasured runtimes | `geometry_metrics.py`, template library, ET-Flow sampling | Time one unit first (C1 includes the `geometry_metrics.py` timing) |
| Researcher degrees of freedom | xTB variant, S4 λ, key level, drop rule | Pre-declare all of them in S-6 before G2 |

## 4. Cross-cutting engineering notes for round 2
1. **Deduplicate runs.** C-EVAL-01's optional CR random-GT runs are C-EVAL-02 (d). B-xtb seed 1 is C-GEOM-01's seed-only
   gate run. C-EVAL-03's test-side metrics are a by-product of C-EVAL-05's relaxation if S-1 logs energies and
   displacements. Budget each once.
2. **Gate the expensive steps, not the cheap ones.** C-GEOM-01/02 GPU steps cost ≤ 4 GPU-h each; their CPU gates are
   for interpretation (and, for C-GEOM-02, for transplant fidelity). The ~26–40 GPU-h training cards should not be
   funded on priors: their gates (C-ROBUST-02, -03) are computable on CPU today and should be measured before P5.
3. **Do not choose between xTB variants on test-GT metrics.** Running both `_xtb` and `_mmff_xtb` costs +1.4 GPU-h and
   removes an ORACLE-informed selection step; pre-declare `_xtb` as primary.
4. **S-5 (validation set, ~70 lines) is worth building, off the critical path.** It removes three "pre-declare
   instead" compromises: S4 λ, the template key level and the C-EVAL-03 validation half. No YES item should wait for
   it.
5. **Not scored (not a card):** the analyst's fifth training source, "GT L of conformer j with conformer i's torsion
   target" (impact §5 D). Engineering-wise it is a `_select_paired_L` change. Whether the paired cache holds sibling
   GT conformers per conformer is UNVERIFIED; it may need a cache rebuild. It is a scout item for round 4.
6. **Wording fixes** before reporting: V1 R2-1 (C-EVAL-01), R2-2 (C-EVAL-04), and the `srETKDGv3()` wording in
   C-GEOM-06 §5 (use the §7 flags). None changes a score.
