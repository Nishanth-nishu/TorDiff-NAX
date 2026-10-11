# C-GEOM-01: GFN2-xTB-relaxed ETKDG seeds as test-time L (the reference level of theory, CPU-gated)

## 1. Idea (scout)
Relax the same ETKDG embeddings we already use (and their MMFF versions) with GFN2-xTB, the level at which every GEOM
conformer was optimised, and use them as test-time L for the existing CTRL_rematch and B1 checkpoints (no training).
A CPU-only gate (L error, pucker-tail share, rigid-molecule AMR-R) decides whether any GPU time is spent. Optional arm:
oversample ETKDG, xTB-relax, and keep distinct ring puckers inside a 6 kcal/mol window.

## 2. Where it comes from (scout)
- GEOM conformers are CREST ensembles whose geometries are optimised with GFN2-xTB [E-GEOM-015]; for QM9 the inputs
  were re-optimised with xTB and CREST ran with default (gas-phase, inferred) settings, at 0.5 core-h per molecule
  [E-GEOM-016].
- Re-optimising GEOM references with GFN2-xTB changes them by ≈ 0 (relaxation energy ≈ 0; 0.00 × 10⁻² Å bonds, 0.001°
  angles, 0.01° torsions), while MMFF and GFN2-xTB minima differ by 1.12 × 10⁻² Å, 1.22° and 4.89° (GEOM-Drugs)
  [E-GEOM-018, E-GEOM-019]. So an xTB optimisation that starts in the reference basin should land on the reference L.
- TD already contains an `xtb_optimize` wrapper (gas phase) that needs only an `xtb` binary [E-GEOM-010]; xtb is a
  standalone binary on conda-forge (linux-64, 6.7.1) or a release tarball, independent of our torch/python stack
  [E-GEOM-034, E-GEOM-035].
- Our own data: MMFF relaxation only partly removes the ETKDG pucker tail (ring seeds > 10°: 20.2% vs 28.6%; 35.1% vs
  49.6% once the all-3-ring molecules, whose ring dihedral is identically 0, are excluded) [E-GEOM-002]. GFN2-xTB
  occasionally fragments molecules [E-GEOM-046], and strained QM9 molecules reacted under CREST/GFN2 [E-GEOM-047].
  (Revised after D-203: the piperazine blog is no longer cited here; its first run was plain DG.)
- CREST's 6.0 kcal/mol window is the GEOM pipeline's notion of "accessible" conformers (optional arm B) [E-GEOM-017].

## 3. Why it could matter here (scout)
Target numbers from `l_error.csv` (heavy atoms, mean per seed; AMR-R from BRIEF, unpaired, preliminary)
[E-GEOM-001, E-GEOM-002, E-GEOM-005, E-GEOM-007, E-GEOM-013]:

| test-time L | ring bond Å | ring angle ° | ring dihedral ° (share > 10°: all ring mols / excl. all-3-ring mols) | acyclic bond Å | acyclic angle ° | CTRL | B1 |
|---|---|---|---|---|---|---|---|
| ETKDG (v1, E-GEOM-052) | 0.032 | 2.38 | 10.7 (28.6% / 49.6%) | 0.029 | 2.84 | 0.178 | 0.236 |
| MMFF-relaxed ETKDG | 0.020 | 1.42 | 8.1 (20.2% / 35.1%) | 0.017 | 2.03 | 0.152 | 0.232 |
| λ = 0.50 (ORACLE) | 0.019 | 1.34 | 3.2 (11.2% / 19.4%) | 0.017 | 2.01 | 0.136 | 0.181 |
| **λ = 0.75 (ORACLE, B1 = CTRL)** | **0.010** | **0.67** | **1.6 (0.01% / 0.02%)** | **0.009** | **1.00** | 0.116 | 0.115 |
| true L + noise 0.02 Å/axis (ORACLE) | 0.027 | 1.38 | 1.0 (0% / 0%) | 0.027 | 1.67 | 0.098 | 0.117 |

- The λ = 0.75 row is the accuracy spec: ring bonds ≤ ~0.01 Å, ring angles ≤ ~0.7°, acyclic angles ≤ ~1°, and
  essentially **no ring seed with ring-dihedral RMSD > 10°**. MMFF already matches λ = 0.5 on bonds/angles; what it
  misses is the pucker tail (20% vs 11% vs 0%; 35% vs 19% vs 0% without the all-3-ring molecules) [E-GEOM-002,
  E-GEOM-007].
- The noise row shows CTRL tolerates 0.027 Å / 1.4° random bond/angle error when puckers are right (0.098), but B1
  does not (0.117) [E-GEOM-005]. B1 needs both right puckers and near-reference bonds/angles; xTB is the only cheap
  source that targets the reference bonds/angles exactly (INFERENCE from E-GEOM-018/019).
- Rigid molecules (≈ 30% of the test set) are nearly model-free: their logged AMR-R tracks L quality (ETKDG 0.152,
  MMFF 0.128, λ = 0.5 0.107, λ = 0.75 0.077, true rings 0.043), but these logged values carry ≤ 0.006 Å model
  dependence and sit on different molecule sets per source (258–318), so they are indicative only; the gate below
  scores seeds directly on one common set [E-GEOM-004, E-GEOM-006].
- Revised after D-201: on the like-for-like λ = 0 base (955 molecules), true rings alone help both models about equally
  (CTRL 0.197 → 0.119, B1 0.261 → 0.188) but leave B1 above CTRL; only true acyclic geometry puts B1 below CTRL
  (0.158 vs 0.182) [E-GEOM-013]. xTB relaxes ring and acyclic L together, which is why it is the arm that can also move
  B1 past CTRL (INFERENCE).

## 4. Assumptions that may not transfer (scout)
- **Basin retention.** Local optimisation keeps the ETKDG pucker basin wherever a barrier separates puckers (6-ring
  chair/boat/twist, 4-ring pucker sign). MMFF did not remove the tail (20% > 10°) [E-GEOM-002]; xTB may not either.
  Which part of MMFF's tail is level-of-theory offset (fixable) vs wrong basin (not fixable) is unknown: this is
  exactly what the CPU gate measures.
- **Reference-level match is inferred for QM9.** The "≈ 0 relaxation" result is for GEOM-Drugs [E-GEOM-018]; QM9
  being gas-phase GFN2 is inferred from "default CREST arguments" [E-GEOM-016]. CREST's internal optimisation
  thresholds may differ from `xtb --opt` defaults (small residual, UNVERIFIED).
- **L|τ coupling.** Whole-molecule optimisation also relaxes torsions; TD then re-randomises torsions, so acyclic angles
  sit at the seed's torsional state, not the sampled one. Ring L is unaffected (ring bonds are not rotatable).
- **Strained QM9 chemistry.** 11.6% of QM9 graphs changed under CREST [E-GEOM-047]; plain optimisation is milder, but
  seeds must be rejected (fallback to the ETKDG seed) if connectivity or stereo changes [E-GEOM-046].
- **Assignment caveat.** ETKDG/MMFF ring-dihedral errors are measured against angle-assigned GT conformers
  [E-GEOM-009]; the gate must add a pucker-coverage metric (min over seeds per GT conformer).
- CTRL was trained on conformer-matched RDKit L, so xTB L is a train/test shift for CTRL (MMFF was too, and helped).

## 5. Minimal experiment (scout)
- **Step 0, install (CPU, time-box 30 min):** `micromamba create -p $PROJECT/xtbenv -c conda-forge xtb` (separate env;
  the TD venv is untouched) or the 6.7.1 release tarball. Not to be done by this scout.
- **Step 1, seeds (CPU):** extend the S1 seed builder so that the existing `L_etkdg2L` embeddings (ETKDG v1 on the
  cluster, E-GEOM-052; same random seed,
  common random numbers) are also written as `L_etkdg2L_xtb` (GFN2-xTB from ETKDG) and `L_etkdg2L_mmff_xtb` (from the
  MMFF seeds), with per-seed graph/stereo checks, convergence flags, wall time, and `l_error.csv` rows. ~25k
  optimisations of ≤ 29-atom molecules; runtime UNVERIFIED (guess 0.5–2 s each ⇒ ≤ 1 h wall on 16 cores; time 200
  first).
- **Step 2, CPU gate (pre-declared; revised after D-205):** score every seed set directly, without a model, on **one
  common molecule set**: (i) ring/acyclic bond and angle RMSD; (ii) share of ring seeds with ring-dihedral RMSD above
  10°, reported on both bases (all ring molecules / excluding all-3-ring molecules); (iii) pucker coverage (min over
  seeds per GT conformer); (iv) heavy-atom symmetric AMR-R of the seeds vs GT on the common rigid set = test molecules
  whose TD `edge_mask` is empty and that are present in every compared source. ETKDG, MMFF, xTB and the λ = 0.5 / 0.75
  references are all recomputed on that same set in the same run. Go rule (non-oracle comparison only): proceed to GPU
  if the xTB source beats MMFF on the common rigid set by ≥ 0.010 Å AMR-R **or** lowers the > 10° share (excl.
  all-3-ring basis) by ≥ 5 points. The ORACLE λ references are yardsticks for interpretation (experiment selection,
  flagged for the validity judge), never thresholds for a model.
- **Step 3, GPU (inference only):** CTRL_rematch s0–2 and B1 s0–2 on `L_etkdg2L_xtb` (or the `_mmff_xtb` variant if
  it wins the gate) = 6 runs; add S3 (both seeds) and S4 (λ chosen on validation, never on test) if their checkpoints
  are final = +3–5 runs. Controls: the same checkpoints on `L_etkdg2L` and `L_etkdg2L_mmff` (already run).
  Primary: paired AMR-R (δ = 0.5 Å) CTRL-on-xTB vs CTRL-on-MMFF (expected lower; scout range 0.125–0.150), and
  B1-on-xTB vs CTRL-on-xTB (expected B1 ≤ CTRL only if the gate reaches the λ = 0.75 row; otherwise B1 stays worse).
  Secondary: COV-R@0.1 on the success intersection, rigid subset, ring-size strata. 1 generation seed.
- **Optional arm B (pucker-diverse):** 4 × 2L ETKDG embeddings → xTB → deduplicate by ring-dihedral fingerprint →
  drop seeds > 6 kcal/mol above the molecule's lowest xTB energy [E-GEOM-017] → sample 2L. +2 GPU runs.
- **Optional arm C (ceiling, not a method; validity flag):** CREST on the test SMILES (≈ 500 core-h,
  [E-GEOM-016]). Non-oracle (SMILES only) but it is the generator that produced GT, so report it only as a ceiling.
- **Cost:** 1.5–3 GPU-h (6–12 × 0.25) + ≤ 1 day CPU. No training. Follow-up only if it wins: xTB-matched training
  (S5 analogue; number of training conformers not counted here, UNVERIFIED; CPU cost not estimated).

## 6. Scout's own call (scout)
Worth trying: YES — cheapest non-oracle source that targets the GT's own level of theory, no training, and the CPU gate
answers "is the pucker tail a force-field offset or a wrong basin?" before any GPU time.

## 7. Code grounding (grounder)
- **Cluster check:** xtb 6.7.1 is on gnode118 as the module binary `/data3/u22/xtb-6.7.1/bin/xtb`. **No install is
  needed.**
  - Probe: median 0.135 s per optimisation.
  - ETKDG seeds move 0.09–0.32 Å.
  - GT conformers stay at 0.000 Å, so the reference basins are xTB-6.7.1 minima on the probe.
- **Touchpoints:**
  - `tools/make_l_seed_pickles.py` `build_etkdg` (`:83-125`): add `--xtb_bin` (default off), relax `seeds` (`:93`)
    and `seeds_mm` (`:103`) into `L_etkdg2L_xtb` / `L_etkdg2L_mmff_xtb`, with per-seed fallback (non-converged, graph
    or stereo change) and meta flags;
  - add both tags to the F5 loop (`:107`), so `l_error.csv` rows use the same Hungarian rule as ETKDG/MMFF;
  - extend the twin assert (`:331-333`);
  - write to a **new** `round3_seeds` dir: the round-2 provenance guard hashes seed pickles (`common.sh:111-114`);
  - assert the regenerated `L_etkdg2L` equals round 2's, so the xTB seeds are common-random-number twins;
  - eval lines go in a new `r3_evalsets.tsv` (`R2_EVALSETS` is overridable, `r2_eval_array.sbatch:23`).
  - Shared pieces: the fixed xTB runner (TD's `utils/xtb.py` shares its tmp dir across Pool workers, `:5-7`, which
    silently swaps coordinates); the 3D graph/stereo filter; the seed-gate tool (pucker coverage; common rigid set from
    TD's `edge_mask`; seed AMR-R via `--no_model --no_random`).
- **Transfer check:**
  - Gas-phase GFN2, as the references [E-GEOM-015/016].
  - The L|τ coupling is the same as the MMFF arm: TD re-randomises torsions (`generate_confs.py:150-151`).
  - Rigid molecules: seed = output (`:150-153`).
  - The 60 ETKDG failures have no seed in any etkdg-family pickle.
  - **S4 needs `--l_level`, and "λ chosen on validation" has no code path** (no validation set exists). Pre-declare
    λ = 0, or build the validation set first.
- **Size / risk:** ~120 builder lines + ~360 shared, 5 tests. Report the fallback share; gate only if ≥ 95 % of seeds
  relaxed.
- **Verdict: IMPLEMENTABLE WITH CHANGES.**
- **Cost:** ≈ 5 CPU-h of seeds (the card said ≤ 1 day) + ≈ 1.3 CPU-h gate; GPU 1.0 (6 runs) – 1.9 (11 runs with
  S3/S4). Optional arm B ≈ 8 CPU-h. Optional arm C (CREST module listed on ada, untested) ≈ 500 CPU-h, with a validity
  flag.
Detail, line numbers and cluster checks: `round3/grounding.md` §3 (shared pieces S-1…S-6 in §2.3).
## 8. Predicted effect on our project (analyst)
Scripts and outputs: `round3/impact_data/` (`mixture_model.out`, `lerror_profile.out`, `noise_power.out`); summary in
`round3/impact.md` §0 and §3. All round-2 numbers are PRELIMINARY (unpaired).
- **Ceiling.** The best physically consistent, reference-level L with correct puckers is true L of a random GT
  conformer (round-1 in-job `gtL`, ORACLE, 3 seeds): CTRL_base 0.0843, B1 0.0369 on 996 molecules, which reweight to
  0.086 / 0.038 on the ETKDG bins. Rigid stratum: 0.018 / 0.011 (`cond_means.txt`). The cycled values (λ = 1:
  0.080 / 0.020) carry conformer identity (B1 leak +0.016) and are not reachable.
- **Expected effect: reasoning chain.**
  1. For CTRL, AMR-R on real samplers depends on the ring tail alone. A model built only from two ORACLE anchors
     (random-GT L; A5acyc) predicts ETKDG 0.178 (observed 0.178) and MMFF 0.145 (observed 0.1525). This gives
     CTRL ≈ 0.086 + 0.20·X + 0.008 calibration, where X = > 10° share on the 49.6 % basis (impact.md §3c).
  2. xTB is a local optimiser on the same embeddings, like MMFF, which took the tail from 49.6 % to 35.1 %. How much
     of the remaining 35 % is a force-field offset (fixable) and how much is a wrong basin (not fixable) is unknown.
     I take X = 30 % as central, with a range of 20–38 %.
  3. CTRL on xTB L is therefore 0.144 central (0.120–0.155), i.e. −0.008 vs MMFF (range +0.003 to −0.033). Bonds and
     angles at the reference level add at most −0.008 (noise 0.02 vs λ1 costs CTRL 0.016).
  4. B1 is bimodal.
     - (A) xTB's acyclic L is reference-like. This is plausible: GEOM references are GFN2 minima, and xTB re-optimising
       them changes ≈ 0 (E-GEOM-018). TD keeps the seed's bonds and angles, so a right-basin xTB seed is the same kind
       of L as another GT conformer's, which B1 handles (0.037). Then B1 ≈ 0.038 + 0.25·X, i.e. **0.10–0.12** at
       X = 30–35 %, beating CTRL by ≈ 0.03 and today's best non-oracle (0.1525) by ≈ 0.04.
     - (B) xTB's acyclic L is not reference-like. B1 is then MMFF-like, **0.20–0.23**. B1 fails on MMFF L (0.232)
       despite errors smaller than noise 0.02 (0.117 with right puckers).
     I give (A) about 50–60 %. Right-basin residuals should be convergence-level, but xTB minima reached from ETKDG
     need not be CREST conformers, and B1's response to small systematic offsets is unmeasured.
  5. S3 (2 seeds) and S4 (3 seeds): bracket 0.12–0.156 (impact.md §3d). In (A), S3 should sit near
     B1 + 0.2·(CTRL − B1) ≈ 0.12. In (B), S3 ≈ CTRL + 0.003.
  6. Transfer. The ≈ 0 relaxation and the MMFF→GFN2 offsets are measured on GEOM-Drugs. QM9 is small and
     ring-heavy, so the ring tail dominates and relaxation is cheap. The CREST vs `xtb --opt` convergence difference
     is UNVERIFIED but small.
- **Worst case.** xTB fixes no more puckers than MMFF (X ≥ 35 %) and branch (B) holds: CTRL ≈ 0.153 (no gain over
  MMFF), B1 ≈ 0.23. The cost is 1.0–2.4 GPU-h (grounder measured 0.17 GPU-h per packed run) and ≈ 6 CPU-h (xTB 0.14 s per
  optimisation).
- **Detectability.**
  - CTRL-on-xTB vs CTRL-on-MMFF is a paired L-source contrast with the same 3 checkpoints: half-width ≈ 0.005 (A2 vs
    R0: 0.0049). The central −0.008 is borderline; −0.015 or more is clear.
  - B1 − CTRL on xTB L (≈ −0.03 in A, ≈ +0.06 in B): CI ≈ 0.006 (round-1 B1 vs CTRL), clear with the existing
    3 + 3 checkpoints. No new training is needed.
- **CPU gate.** The rigid-set rule "≥ 0.010 Å vs MMFF" can pass on bond/angle improvement alone, without fixing any
  pucker: the oracle slope is 0.016 Å per degree, and MMFF 2.74° → ≈ 1.2° gives ≈ −0.025 Å on rigid molecules
  (`dose_response.out`). For CTRL only the tail metric predicts the full-population gain, and B1's outcome does not
  depend on the tail. Since the GPU step is ≤ 3 GPU-h, I recommend running it whenever the xTB seeds are built and
  using the gate metrics to interpret the result, not to decide whether to run.
- **Decision value: high either way.**
  - (A): non-learned reference-level L already unlocks true-L-trained models. FlexiTors should pair xTB-type L with
    B1/S3-type training and spend learning effort only on the ring tail (C-GEOM-02, then C-GEOM-04).
  - (B): the model needs λ ≥ 0.75-level exactness that no local relaxation gives. Robust training (S3/S4,
    C-ROBUST-01, xTB-matched C-ROBUST-03) is then the route, and plain B1 is dead.
  - The CTRL tail result separates force-field offset from wrong basin.
- **Worth trying: YES.** Add B1 s0–2, S3 s0/s1 and S4 s0–2 to the CTRL runs. Run S4 at two pre-declared
  levels, λ = 0 and λ = 1: no validation path exists (grounding.md S-5), and λ = 0 alone would hide branch (A).
  ≈ 2.4 GPU-h in total. Probability-weighted best-arm gain vs 0.1525 is 0.55 × 0.033 + 0.45 × 0.008 ≈ 0.022 Å, i.e.
  ≈ 9 mÅ per GPU-h.

**Revised after cross-read (2026-10-11; earlier text kept as history).**
- Round 1 of the cross-read: GPU cost changed from ≈ 3 to 1.0–2.4 GPU-h, and S4 moved to two pre-declared λ values,
  following the grounder's measured costs and missing validation path.
- Round 2: the grounder showed that anchor G leaks conformer identity (grounding.md 5.2.1). Recomputed in
  `impact_data/ntrue_leak.out` (impact.md §3g): with only other conformers' reference L, B1 ≈ 0.16–0.35 per n_true
  bin, i.e. RDKit level; CTRL loses ≈ 0.03 (noisy). Consequences:
  - **Ceiling:** the leak-free floors are CTRL 0.111–0.145 and B1 0.19–0.27, against 0.086 / 0.038 above.
  - **B1:** branch (A) is superseded, because reference-like acyclic L is not enough. B1 gains only on targets where
    some xTB seed lands in the target's *own* basin (pucker and torsions). Such a seed reproduces the target's GT L
    legitimately: GT conformers stay put under xtb 6.7.1 (grounder probe). With s = that share, B1 ≈ 0.10 (s = 0.75),
    0.14 (s = 0.55) or 0.17 (s = 0.35). Revised range 0.10–0.22, central ≈ CTRL.
  - **S3:** CTRL-like on non-own L and near-B1 on own L, so 0.10–0.16, central ≈ 0.13. It is now the most promising
    arm. Its own-vs-other-L response is measured by the C-EVAL-02 (d) random-GT runs (0.9 GPU-h); run them.
  - **CTRL:** 0.146 (0.13–0.155), unchanged. Real samplers realise 57–76 % of the predicted tail gain.
  - **Gate:** add s = COV-R@0.05–0.1 of the xTB seeds themselves (grounder's S-3 `--no_model --no_random` run, all
    molecules).
  - **Worth trying: still YES.** The best-arm gain is ≈ 0.012–0.015 Å, i.e. ≈ 5 mÅ per GPU-h (was ≈ 9). The decision
    value goes up: with s, the result tells FlexiTors whether any non-learned L source can feed true-L-trained models
    at all.

## 9. Panel (judges)
