# Experiments so far: what, why, outcome, and what it means

**Project:** FlexiTors-Diffusion, i.e. extending Torsional Diffusion (TD; Jing, Corso et al., NeurIPS 2022) so that
it also learns the local structure L (bond lengths, bond angles, ring shapes) instead of copying it from RDKit.
**Phase:** QM9 only. **Last updated:** 8 October 2026, 22:30 IST.

This file covers every experiment run so far. For each one it gives: why we ran it, where the idea came from, what we
changed in the code, what came out, what the result says, what we did next, and whether it was worth doing. Code and
infrastructure changes are explained the same way: what changed, what caused the need, and why.

---

## How to read this file

- **AMR-R** (average minimum RMSD, recall): for each true (GEOM) conformer, the distance in Å to the closest generated
  conformer, averaged. **Lower is better.** It is our main number.
- **COV-R@δ**: % of true conformers that have a generated conformer within δ Å. **Higher is better.** δ = 0.5 Å is the
  official QM9 threshold (fixed by you before any results); 0.05–0.25 Å are finer, secondary thresholds.
- **ORACLE**: the experiment gives the model information it would never have in real use (the test molecule's true
  geometry). Oracle results tell us *what is possible*, never which model to pick.
- **Paired**: compared molecule by molecule on the same molecules, with bootstrap 95% confidence intervals and Holm
  correction for multiple comparisons. **Preliminary**: a plain average over each run's own molecules, not yet paired.
- Baseline to compare everything against: **TD = 0.176 Å** (our reproduction; paper 0.178).
- "Seeds": training seeds (independently trained models) or sampling seeds (independent generation runs). Seed-to-seed
  noise is about ±0.002 Å AMR-R, so differences smaller than ~0.004 Å need paired statistics to mean anything.

### Timeline

| Date | What happened |
|---|---|
| 29 Sep | Project set up; literature review (26 papers); round-1 plan; code walkthrough; cluster environment |
| 30 Sep – 3 Oct | Round 1 on gnode118: 23 inference experiments, 15 training runs, floor analyses, paired statistics |
| 6 Oct | Round 1 results synced; round-2 design by research agents, verifiers and 3 voting coding agents |
| 7 Oct | Round 2 implemented, reviewed, literature-checked, fixed; checkpoint resume added; round 2 submitted |
| 8 Oct | Round 2 waves 1–2 finished; round-3 orchestration built; 3 scouts + 3 verifiers working |
| ~10 Oct | Round 2 training and evaluations expected to finish; paired analysis |

---

# Part 1. Round 1: what limits Torsional Diffusion on QM9?

**The question behind the whole round.** TD only moves rotatable torsions. Every bond length, bond angle and ring
shape is copied from an RDKit (ETKDG) structure and never changed. FlexiTors assumes this frozen local structure is
TD's main source of error. Round 1 was designed to **test that assumption before building anything**, and to try to
falsify it (plan: `notes/ablation_plan_qm9.md` §5 lists the falsifiers).

**Where the round-1 ideas came from.** The TD paper itself (its Table 7 numbers, its Table 8 ablations, its App. H
estimate of a ~0.17 Å floor on QM9, its App. F.1 "GT-other-L" result), the literature review in `papers/` (gaps G1–G3
in `papers/gap_synthesis.md`), and a code walkthrough of the TD repository (`notes/code_walkthrough.md`,
`notes/code_concerns.md`). Every arm was pre-registered in the plan before any result existed.

**Compute used:** ~23 inference runs on the released checkpoint + 15 training runs (100 epochs, ~11 h each) on
4× RTX 3090, plus CPU analyses. All runs finished by 3 October.

---

## E1. R0–R2: reproduce TD's QM9 result

- **Why:** every other experiment is a difference from this baseline. If the baseline were off, every difference
  would be off with it. A reproduction also checks our environment, data split and evaluator.
- **Where from:** TD paper, Table 7 (QM9 row: COV-R 92.8 %, AMR-R 0.178 Å, AMR-P 0.221 Å).
- **What we ran:** the released `qm9_default` checkpoint, RDKit starting structures, 20 SDE steps, 3 sampling seeds.
- **Outcome:** AMR-R **0.176** (median 0.143) vs paper 0.178 (0.147); AMR-P 0.220 vs 0.221. COV-R **88.6 %** vs paper
  92.8 %.
- **What it says:** the error numbers reproduce. Coverage does not, which led to E2.
- **What we did next:** investigated the coverage gap (E2) and used R0 as the reference for all paired comparisons.
- **Worth it?** Yes, essential. Cheap (3 × ~15 min GPU) and it made every later number trustworthy.

## E2. "Released": score the paper's own released conformers with our evaluator

- **Why:** to find out whether the 4-point coverage gap was our mistake or the paper's.
- **Where from:** the authors released their generated conformers (`qm9_steps20.pkl`).
- **What we changed:** a custom unpickler (`slurm/eval_released.sbatch`), because the released file carries Python
  attributes no current RDKit can read.
- **Outcome:** the paper's own conformers score COV-R **88.8 %**, AMR-R 0.177. They cover only 931 of the 1,000
  molecules: **69 molecules fail** (RDKit cannot build strained cages such as
  `C1C[C@]23C[C@H](C2)[C@@H]2[C@H]1[C@@H]23`, plus 5 multi-fragment inputs) and count as 0 % coverage.
- **What it says:** the paper's 92.8 % cannot be reproduced even from its own samples. It most likely excluded
  failures from the average. Our pipeline is faithful. It also shows the failures are themselves evidence for the
  project: where RDKit has no geometry at all, TD cannot run.
- **What we did next:** we always compare against our own numbers, never the paper's 92.8, and since round 2 we report
  failure counts and coverage on the "success intersection" (molecules every arm handled).
- **Worth it?** Yes. It stopped us from chasing a coverage gap that was never real.

## E3. Sanity: pass the true conformers straight through the evaluator

- **Why:** a broken evaluator would silently corrupt everything.
- **Outcome:** AMR-P = **0.000** exactly; COV-R 99.7 % (a GT conformer that is never drawn in the cycle has non-zero
  distance, as expected).
- **What it says:** the evaluator and the GT seed files are correct.
- **Worth it?** Yes; nearly free.

## E4. A0: RDKit conformers alone, no model

- **Why:** to measure what the TD model adds over plain RDKit.
- **Where from:** TD Table 7 "RDKit" row (0.235 Å).
- **Outcome:** AMR-R **0.230** (paper 0.235). Paired vs R0: **+0.054 Å** worse [CI 0.046, 0.061].
- **What it says:** the torsion model is useful: it removes about a quarter of RDKit's error.

## E5. A0-rand: RDKit geometry with random torsions, no model

- **Why:** the "no knowledge at all" floor for torsions.
- **Where from:** TD Table 8 "random torsions" row.
- **Outcome:** AMR-R **0.242**, worse than RDKit alone. Precision collapses (COV-P 75 %).
- **What it says:** the lower bound. Everything should beat this.

## E6. A1: true bond geometry + TD model (ORACLE)

- **Why:** the core test of the FlexiTors premise. Give the *same* pretrained model the true lengths, angles and ring
  shapes (from a random true conformer of the molecule) and keep everything else equal. Whatever error disappears is
  error caused by RDKit's local structure.
- **Where from:** TD App. H (estimates the RDKit-L floor at ≈0.17 Å on QM9); gap G1 in `papers/gap_synthesis.md`.
- **What we changed:** `--seed_confs` option in `generate_confs.py` + `tools/make_seed_pickles.py` to build seeds from
  true conformers; a guard rejecting multi-fragment seeds (a bug it exposed, see the bug table).
- **Outcome:** AMR-R **0.081** (median 0.038). Paired vs R0: **−0.093 Å** [−0.099, −0.086], i.e. **−54 %**. Coverage
  at 0.05 Å jumps from 3.9 % to 66.7 %. Failures drop from ~65 to 4.
- **What it says:** about half of TD's error on QM9 is local structure it is not allowed to change. **This is the main
  justification for FlexiTors.** Caveat: the model was trained on RDKit geometry, so true geometry is off-distribution
  for it; B1 (E15) tests the other side.
- **What we did next:** decomposed the floor (E12) and trained on true geometry (E15).
- **Worth it?** Yes, the single most informative experiment of round 1, for ~3 × 20 GPU-minutes.

## E7. A1-rand: true geometry + random torsions, no model (ORACLE)

- **Why:** how hard are torsions alone when the geometry is perfect?
- **Outcome:** AMR-R **0.157**, already better than TD on RDKit geometry (0.176).
- **What it says:** with correct geometry, even random torsions beat TD. The torsion model is not the bottleneck; the
  scaffold it is given is.

## E8. A1c: each sample gets its own true conformer's geometry (ORACLE)

- **Why:** tests gap G3, the coupling between torsions and bond geometry. A1 gives each sample the geometry of a
  random true conformer; A1c pairs each sample with its own. If coupling matters, A1c beats A1.
- **Where from:** TD App. F.1, where "GT-other-L" vs own L on GEOM-DRUGS gave 0.324 → 0.284 Å.
- **What we changed:** `--seed_confs_cycle` (round-robin assignment) in `generate_confs.py`.
- **Outcome:** AMR-R **0.077** vs A1 0.081. Paired: **−0.0037 Å** [−0.0063, −0.0013]; small but consistent across
  3 seeds.
- **What it says:** coupling exists on QM9 but is small for the standard model. Most of the gain comes from having
  *any* correct geometry. Round-2 verifiers later showed part of this gap is a sampling artefact: random assignment
  draws with replacement and leaves ~13.5 % of true conformers without their own geometry. Since round 2 every
  true-geometry test uses cycled assignment.
- **Worth it?** Yes, cheap, and it told us joint torsion+angle coupling is a second-order effect on QM9.

## E9. A2: MMFF-relax the RDKit structure before diffusion

- **Why:** a cheap, physics-based way to improve local structure without any learning. If it helps, local structure
  quality matters and a learned version should help more.
- **Where from:** standard practice; RDKit's MMFF94 force field; plan arm A2.
- **What we changed:** `--pre_mmff` flag.
- **Outcome:** AMR-R **0.150**. Paired vs R0: **−0.026 Å** [−0.031, −0.021], −15 %, with no retraining. COV-R@0.5
  unchanged (+0.05, not significant); COV-R@0.1 +12 points.
- **What it says:** better geometry helps immediately. It is also a free baseline that beats OMEGA (0.177) and that
  any FlexiTors result must beat.
- **What we did next:** round-2 arm S5 trains TD on MMFF-relaxed structures.
- **Worth it?** Yes. Strong, cheap, and it set the bar for the project.

## E10. A3: MMFF-relax after diffusion

- **Why:** can geometry be fixed after the fact instead of during generation?
- **Outcome:** AMR-R **0.186**, worse: +0.010 Å [0.003, 0.016]. But AMR-P improves by −0.057 Å (more accurate
  individual conformers). Run-matched floor analysis shows relaxing afterwards shifts torsions (torsion headroom
  doubles, 0.047 → 0.086 Å per molecule).
- **What it says:** post-hoc relaxation pulls conformers into nearby minima: more precise, less diverse. Geometry has
  to be right *during* generation, which argues for learning it jointly.
- **Worth it?** Yes. A clean negative result that rules out the cheapest alternative to FlexiTors.

## E11. G1: stereochemistry assigned from the true structure

- **Why:** the training structures are built from the true molecular graph, but the test structures from SMILES.
  Wrong stereo (a flipped chiral centre or double bond) cannot be fixed by any geometry model, so we needed to rule it
  out as a hidden cause of error.
- **What we changed:** `--seed_mols` with stereo perceived from the true 3D structure (`tools/make_seed_pickles.py`);
  a guard for missing seeds (a crash it exposed).
- **Outcome:** AMR-R 0.175 vs 0.176: **+0.0001 Å** [−0.0035, 0.0038], no effect.
- **What it says:** stereo mismatch is not part of TD's error on QM9, so A1's gain is genuinely geometry.
- **Worth it?** Yes, as a confound check. A null result here was the useful outcome.

## E12. Floor analyses (CPU only): how much error could a perfect torsion model remove?

- **Why:** to split TD's error into "fixable by better torsions" and "fixable only by changing local structure". This
  was the plan's main falsification test: if most error were torsional, FlexiTors would be unmotivated.
- **Where from:** TD App. H (floor ≈ 0.17 Å); round-1 review `review/cross_validation.md` M1, which warned that TD's
  number is biased upwards and must be recomputed run by run.
- **What we built:** `tools/local_structure_analysis.py` (`--mode test` and `--mode run`).
- **Outcomes:**
  - **Run-matched floor (J-run), on R0's own conformers:** observed error 0.167 Å per molecule = floor 0.120 +
    torsion headroom 0.047. So **up to ~72 %** of TD's per-molecule error needs a local-structure change. (A
    verifier noted this is an upper bound, because the floor optimiser is heuristic.)
  - **MMFF (A2):** floor drops to 0.097, headroom stays 0.046, so MMFF helps by fixing geometry, not torsions.
  - **RDKit vs true geometry:** bond RMSD 0.036 Å (true vs true: 0.004); angle RMSD 3.9° (true vs true: 1.7°).
  - **Decomposition (J-decomp):** floor 0.099 Å per conformer; setting true acyclic angles → 0.089; plus true acyclic
    bond lengths → 0.074; a perfect independent geometry sampler → 0.027.
  - **By ring size** (computed in round 2, reproduced by a verifier): acyclic molecules 0.051 → 0.008 with true
    acyclic angles and bonds; **ring molecules (72 % of conformers) 0.117 → 0.099 only.** Rings of 7+ atoms are the
    worst (0.218), then 5-rings (0.212).
- **What it says:** TD is mostly limited by local structure, and the part that matters most is **ring geometry**.
  A FlexiTors that only diffuses acyclic bond angles would reach only a small part of the error.
- **What we did next:** round-2 arm S1 tests ring-only vs acyclic-only true geometry *with the model in the loop*;
  round 3 searches for ring-geometry sources.
- **Worth it?** Yes. It changed the design target of the project, with zero GPU.

## E13. CTRL_base: retrain TD ourselves (3 seeds, 100 epochs)

- **Why:** the control for every training experiment; also checks that our training reproduces the paper.
- **Outcome:** AMR-R **0.181** (seed 0: 0.179). True geometry (ORACLE): 0.084 (own-conformer cycle 0.082).
- **What it says:** training reproduces. 100 epochs is close enough to the released model to serve as a control.

## E14. CTRL_rematch: same recipe, training data regenerated by us

- **Why:** the released training pickles were made with an unknown RDKit version. Any experiment that regenerates
  training data (B2, B5, S3, S4, S5) must be compared with a control whose data we also regenerated; otherwise the
  RDKit version would be confounded with the experiment.
- **Where from:** round-1 review C3.
- **Outcome:** AMR-R 0.179: **−0.002 Å** [−0.003, −0.001] vs CTRL_base, coverage unchanged.
- **What it says:** regeneration barely matters; the control is valid.

## E15. B1: train on true geometry (3 seeds), test on RDKit and on true geometry

- **Why:** does a model trained on correct geometry do better? And how sensitive is the torsion model to the geometry
  it was trained on (gap G2, train/test shift)?
- **Where from:** TD Table 8 "train on GT local structures" (on GEOM-DRUGS, COV-R collapses 72.7 → 34.8). B1 is a
  reproduction of that row on QM9, extended with a test on true geometry, which TD did not report.
- **What we changed:** a `raw` data variant without conformer matching (`--std_pickles ""` path); a single-process
  featurisation (a multiprocessing memory limit hung the job for 6 h, see bugs).
- **Outcome:**
  - On RDKit geometry: AMR-R **0.234**, paired **+0.054 Å** [0.047, 0.060] vs CTRL_base: as bad as no model at all
    (0.230). COV-R −6.1 points.
  - On true geometry (ORACLE): AMR-R **0.037** vs CTRL_base 0.084: **−0.047 Å** [−0.053, −0.042]; COV-R@0.05 +23.5
    points. With its own conformer's geometry: **0.021**.
- **What it says:** the torsion model is tuned to the geometry distribution it was trained on. Trained on true
  geometry it is the best model we have *if* it gets true geometry, and useless on RDKit geometry. So a learned
  geometry generator is worth up to ~0.14 Å, **but only if the torsion model tolerates imperfect geometry.**
- **What we did next:** round 2 was designed almost entirely around this: S1 measures how good geometry must be for B1
  to win; S2, S3, S4 try to make the model robust.
- **Worth it?** Yes. Expensive (~33 GPU-h) and it was the most decision-relevant result of the round.

## E16. B2: conformer matching by torsion optimisation only, random pairing (3 seeds)

- **Why:** TD's conformer matching pairs each true conformer with an RDKit structure. B2 tests whether that pairing
  matters (gap G3 again, from the training side).
- **Where from:** TD Table 8 "only D.E. matching, random pairing" (DRUGS: 72.5 vs 72.7, nearly no loss).
- **What we changed:** `--no_match` in `standardize_confs.py`.
- **Outcome:** AMR-R 0.181 vs CTRL_rematch 0.179: **+0.002 Å** [0.001, 0.004]; coverage unchanged.
- **What it says:** same as TD's DRUGS result: pairing barely matters.
- **Worth it?** Partly. It confirmed a paper result on our dataset, but ~33 GPU-h for a near-null effect. In hindsight
  1 seed would have been enough to rule out a large effect.

## E17. B5: conformer matching on heavy atoms only (3 seeds)

- **Why:** the code review (`notes/code_concerns.md` #7) found that TD's matching objective includes hydrogens, which
  could make the torsion targets noisy. A gain would have been a cheap TD fix (not FlexiTors evidence) that we would
  need to subtract first.
- **What we changed:** `--heavy_objective` in `standardize_confs.py`.
- **Outcome:** AMR-R 0.185: **+0.006 Å** [0.004, 0.008], slightly worse.
- **What it says:** the hydrogen-inclusive objective is not a problem. Together with B2: **conformer matching is not
  the lever; stop tuning it.**
- **Worth it?** Partly. It closed a suspected bug, at ~33 GPU-h.

## E18. Threshold sweep and paired statistics

- **Why:** QM9 coverage at 0.5 Å is close to 100 % for every method (median 100 in TD's own table), so it may hide real
  differences. Paired statistics separate real effects from seed noise.
- **What we built:** threshold sweep in `evaluate_confs.py` (0.05–1.25 Å), `tools/paired_compare.py` (paired bootstrap,
  Holm), `tools/breakdown.py`.
- **Outcome:** at 0.5 Å every arm sits near 89 %; at 0.1 Å MMFF-before adds 12 points and true geometry 43. A2 cuts
  error 15 % but moves COV-R@0.5 by only +0.05.
- **What it says:** the official coverage number cannot see the improvements FlexiTors targets; AMR-R and fine
  thresholds can. δ = 0.5 Å stays the primary threshold, as you pre-registered; COV-R@0.1 on the success
  intersection was added as the key secondary in round 2.

## Round-1 code and infrastructure changes, and why each was needed

All on the code branch now in `torsional-diffusion/`. Every new option is off by default, so the original behaviour is
unchanged.

| Change | Cause (what went wrong or was missing) | Why it was needed |
|---|---|---|
| Seeding (`--seed`), avoid RDKit `randomSeed=0` | Upstream parsed a seed but never used it; seed 0 is degenerate in RDKit | Paired comparisons need identical RDKit structures across arms |
| `--seed_confs`, `--seed_confs_cycle`, `--seed_mols` | No way to start from true geometry or true stereo | Needed for oracle arms A1, A1c, G1 |
| `--no_match`, `--heavy_objective` in standardisation | No way to vary conformer matching | Needed for B2, B5 |
| Threshold sweep; SUMMARY at the exact threshold; per-molecule result dump | Only one threshold; summary snapped to a 0.125 grid | Fine-threshold analysis and paired statistics |
| `--no_parity` | — | Prepared for a deferred parity ablation |
| `log_det_jac` returns 0 for molecules with 0 rotatable bonds | SVD crashed on an empty Jacobian; every inference run died | Without it nothing ran |
| Reject multi-fragment true-geometry seeds | `mask_rotate` assertion on `C=C1C(=O)C=NN1C.N` | A1 crashed |
| Guard `RemoveAllConformers` on a missing seed | G1 crashed | — |
| Featurise in a single process | A multiprocessing shared-memory limit hung the job for 6 h at 106k molecules | Lost ~8 h; single process takes ~8 min |
| Always `srun -n 1`; smaller CPU requests | Commands ran twice; CPU jobs blocked GPU jobs | Lost ~10 h of one GPU before the fix |

---

# Part 2. Round 2: how should the model get better geometry, and survive bad geometry?

**The question behind the whole round.** B1 showed a model trained on true geometry is excellent on good geometry
and useless on RDKit geometry. Round 2 asks two things:
1. **How good must the geometry be** before the true-geometry model wins? That is the accuracy target any
   FlexiTors geometry component must hit.
2. **Can training make the model robust,** so it still works with imperfect geometry and gains from better geometry?

**How the experiments were chosen** (files in `round2/`):
1. Two research agents proposed experiments independently: `research_A.md` (how other papers learn local structure),
   `research_B.md` (weak spots in our round-1 conclusions). Every claim has a PDF path, page and exact quote.
2. Two verifier agents opened every source and recomputed every number (`verify_A.md`, `verify_B.md`). All paper
   quotes held; they corrected three overstated numbers and one wrong code claim.
3. The merged, verified list is `SHORTLIST.md`. Three coding agents wrote independent implementation plans
   (`code_plan_1/2/3.md`), reviewed each other's plans and voted (`vote_*.md`, `DECISION.md`; plan 2 won 2–1).
4. Two coding agents reviewed the code (`review_1.md`, `review_3.md`); both research agents checked the implemented
   code against the papers (`research_check_A/B.md`); 13 fixes were applied (`FIXES.md`); checkpoint resume was added.

**Your decisions this round:** δ = 0.5 Å stays the single primary threshold; S4 trains on "safe" pairs only, with
controls re-scored on S4's molecules (kept after the research agents showed it leaves a small training-set
difference); inference packed 3 jobs per GPU; jobs may use the 4-day limit and resume from checkpoints.

**Status:** all inference experiments (S1) are done. Training: S2, S3, S4 done; B1cap, S2-0.02 and S5 running;
evaluation panels queued. **All round-2 numbers below are preliminary** (unpaired means) until the pre-declared paired
analysis (`tools/analyze_round2.py`) runs on the finished set (~10 Oct).

## S1. Geometry-quality sweep (inference only, existing models)

- **Why:** to find how accurate the test-time geometry must be for B1 (trained on true geometry) to beat the standard
  model. That number becomes the accuracy specification for FlexiTors' geometry component. It also tests, with the
  model in the loop, whether ring or acyclic geometry matters more (E12 only showed it without a model).
- **Where from:** research_A R2A-1 and research_B R2-1/R2-5 (`round2/research_*.md`); the round-1 floor split (E12).
  The way of blending RDKit and true structures (aligned Cartesian interpolation) follows ET-Flow's construction, as
  both literature checks confirmed (`round2/research_check_*.md`, item I1).
- **What we changed:**
  - `tools/lgeom.py`: heavy-atom Kabsch alignment applied to all atoms; relabelling of symmetry-equivalent atoms
    before blending; terminal bond lengths kept fixed; a `pair_ok` safety check (bond lengths, clashes, chirality) at
    blend levels 0.25 / 0.5 / 0.75.
    - **Cause:** a coding agent measured that a naive blend collapses a rotated methyl C–H bond from 1.09 Å to
      ~0.62 Å at the midpoint, and O–H / N–H bonds shrink by 0.10–0.15 Å. Without these fixes the blended
      structures would be physically broken.
  - `tools/make_l_seed_pickles.py`: builds every test geometry set (blends, noise, ring-only, acyclic-only, MMFF),
    with population counts and a per-seed error table (`l_error.csv`).
    - **Cause of a fix (blocker found in code review):** the first version dropped a whole test molecule if any
      single pair failed the safety check, losing **358 of 997** test molecules, mostly flexible ones. Fixed by
      replacing unsafe pairs with the molecule's safe pairs: now 955 molecules for the endpoint and ring/acyclic sets,
      906 for the partial blends (49 molecules have no safe pair).
- **Outcome** (AMR-R, mean of 3 training seeds unless noted; ORACLE except the first two rows):

| Test-time geometry | Standard model (CTRL_rematch) | B1 (trained on true geometry) |
|---|---|---|
| RDKit ETKDG | 0.178 | 0.236 |
| MMFF-relaxed ETKDG | 0.152 | 0.232 |
| Matched RDKit (blend 0) | 0.197 | 0.261 |
| Blend 0.25 | 0.160 | 0.222 |
| Blend 0.50 | 0.136 | 0.181 |
| Blend 0.75 | 0.116 | **0.115** |
| True (blend 1, own conformer) | 0.080 | **0.020** |
| True + noise 0.02 Å/axis | 0.098 | 0.117 |
| True + noise 0.04 Å/axis | 0.119 | 0.154 |
| True ring geometry only, RDKit acyclic | **0.119** | 0.188 |
| True acyclic geometry only, RDKit rings | 0.182 | 0.158 |

- **What it says:**
  1. **B1 only beats the standard model once geometry is about 75 % of the way from RDKit to the truth.** In measured
     terms (round-3 scout GEOM, verified): ring bonds within ~0.010 Å, ring angles within ~0.7°, acyclic angles
     within ~1°, and almost no ring with a twist error above 10°. RDKit has 28.6 % of rings above 10° (49.6 % when
     3-membered rings, which cannot twist, are excluded), and MMFF still 20 % (35 %).
  2. **Ring geometry is the lever.** True rings alone take the standard model from 0.178 to 0.119; true acyclic
     geometry alone does nothing (0.182). Both models gain about the same from true rings when measured on the same
     molecules (−0.078 vs −0.074 Å; a verifier corrected an earlier claim that B1 gains nothing).
  3. **B1 can't use MMFF geometry** (0.232), while the standard model gains 0.026 Å from it.
- **What we did next:** round-3 scouts were pointed at ring-geometry sources (xTB relaxation, ring templates,
  ring-pucker models) and at error-shaped training.
- **Worth it?** Yes, very. ~15 GPU-h, and it produced a concrete accuracy target and redirected the design to rings.

## S2 (B6). Train on true geometry with random noise added (σ = 0.04 Å per axis, 3 seeds)

- **Why:** B1 collapses on RDKit geometry because it never saw imperfect geometry in training. Adding noise to the
  geometry during training should make it robust.
- **Where from:** cascaded diffusion's "conditioning augmentation" (Ho et al. 2022) and "input perturbation" for
  exposure bias (Ning et al. 2023), both added and verified in round 2 (`round2/research_A.md`, `research_check_*.md`).
  σ = 0.04 Å was chosen so the noise matches RDKit's overall angle error (3.3° vs 3.9°).
- **What we changed:** `--l_jitter` in `train.py` / the dataset: fresh Gaussian noise on all atoms for every training
  sample (default off).
- **Outcome (preliminary):** RDKit geometry **0.206** (seeds 0.204–0.208); true geometry **0.054**.
- **What it says:** noise half-fixes B1: better than B1 on RDKit geometry (0.236) but still ~0.025 Å worse than the
  standard model (0.181), far outside the pre-registered 0.004 Å margin, so it will very likely fail its success test.
  It also gives up much of B1's true-geometry advantage (0.020 → 0.054). Why: round-3 scout ROBUST (verified) showed
  random noise has 1.6× RDKit's bond error but almost none of its wrong ring shapes (0.58 % vs 49.6 % of rings with
  4+ atoms twisted more than 10°). **The noise is the wrong shape:** RDKit's error is systematic and ring-heavy, not
  random. Both literature checks had flagged this risk before the run.
- **What we did next:** round-3 card C-ROBUST-01 proposes error-shaped augmentation (train on true, RDKit, and
  ring/acyclic hybrid geometry).
- **Worth it?** Yes as a test; it cost ~35 GPU-h and its "failure" explains *why* and points to the right fix.

## S3. Train on a 50/50 mix of RDKit and true geometry (2 seeds)

- **Why:** instead of approximating RDKit's error with noise, show the model real RDKit geometry half the time and true
  geometry the other half. It should then work on both.
- **Where from:** research_A R2A-3 (mixture / domain-randomisation idea); verified as correct in both literature
  checks (item I4).
- **What we changed:** a "paired" training cache that stores each training conformer's matched RDKit geometry and its
  true geometry (`tools/build_paired_pickles.py`; pairing verified on 99.999 % of pairs by recomputing each stored
  matching RMSD); `--l_mix_p_gt` in training (per-sample coin flip, both halves capped equally).
- **Outcome (preliminary):** seed 0: RDKit **0.182**, true **0.033**; seed 1: **0.182** / **0.034**.
- **What it says:** one model now matches standard TD on RDKit geometry (0.182 vs 0.181) **and** keeps most of the
  true-geometry gain (0.033 vs 0.080 for the standard model). This is exactly the property FlexiTors' torsion part
  needs: robust today, and better whenever the geometry gets better. It replicated across both seeds.
- **What we do next:** confirm with the paired analysis; round 3 builds on it (S3 is the base recipe for the
  error-shaped and deployed-source training cards).
- **Worth it?** Yes. Currently the best result of round 2, ~23 GPU-h.

## S4. Tell the model how good its geometry is (λ-conditioned training, 3 seeds)

- **Why:** a single model trained on blends of RDKit and true geometry, with the blend level λ given as an input, could
  adapt to any geometry quality at test time.
- **Where from:** cascaded diffusion's noise-level conditioning (Ho et al. 2022), which both literature checks rated a
  direct match to the "extra time embedding" idea.
- **What we changed:** `--l_interp` (training on aligned blends, λ ~ U[0,1]) and `--lambda_embed_dim` (λ concatenated
  to the noise-level embedding of the score model); `--l_level` at test time (one value for all molecules, never taken
  from the true structure). A gate (`tools/s4_gate.py`) checked pairing (99.999 %), safety-check rate (87.7 %,
  informational), that blend 1 reproduces the true-geometry result (within 0.004/0.002 Å), and a smoke test. Passed.
- **Your ruling:** S4 trains only on pairs that pass the safety check (87.7 %); its controls are re-scored on S4's
  molecules. Both research agents showed this leaves a training-set difference; reported as a limitation.
- **Outcome (preliminary):** RDKit (λ = 0) **0.183 / 0.181 / 0.182**; true geometry (λ = 1) **0.044 / 0.044 / 0.041**.
- **What it says:** S4 matches S3 on RDKit geometry but is worse on true geometry (≈0.043 vs 0.034). The extra input
  bought nothing over the simpler mix. Not a clean comparison: S4 also trained on ~12 % fewer pairs.
- **Worth it?** Partly. It cost ~35 GPU-h; a negative-leaning result that keeps the simpler S3 as the recipe. Round-3
  card C-ROBUST-02 (S4 without the λ input) is rated MAYBE and only worth running if S4 ever beats S3.

## S5 (B3). Train TD on MMFF-relaxed RDKit structures (3 seeds) — running

- **Why:** the strongest physics-only baseline. If training and testing both use MMFF geometry, how far does a
  non-learned geometry improvement go? Any FlexiTors result must beat it.
- **Where from:** round-1 deferred arm B3; research_B R2-4.
- **What we changed:** MMFF variant of conformer matching (`standardize_qm9.sbatch VARIANT=mmff`), with MMFF failure
  counting (MMFF calls had no error handling before; one bad molecule could crash a build).
- **Controls:** CTRL_rematch with and without MMFF at test: **0.154** / 0.178 (done).
- **Outcome:** pending (2 of 3 seeds training).
- **Note from the literature check:** GEOM's reference structures are GFN2-xTB minima, not MMFF, so S5 is the
  strongest *MMFF* baseline, not the strongest possible fixed-geometry baseline.

## B1cap. B1 on the same capped set of conformers the other arms use (1 seed) — running

- **Why:** B1 trains on every true conformer, while the matched datasets keep at most ~30 per molecule. B1's advantage
  might partly come from seeing more data. B1cap removes that confound.
- **Where from:** verify_A (confound flagged), voted in by all three coding agents (D6).
- **Outcome:** pending.

## S2 at σ = 0.02 Å (1 seed) — running

- **Why:** exploratory second noise level for S2. Pending.

## S6. Sampler checks (steps, σ_min, ODE) — dropped

- **Why dropped:** to keep the budget near 175 GPU-h, the panel trimmed the descriptive runs first. Round-1 evidence
  (E12: torsion headroom ~0.047 Å) already suggests sampler settings are not the bottleneck.

## S0. CPU re-analysis — pending

- **Why:** re-run every paired table on the success intersection, clean the floor numbers, and test whether B1 reads
  torsion hints from the geometry (a "leak" test in torsion space, with a no-model null). Runs after the training set
  is complete.

## Round-2 code and infrastructure changes, and why each was needed

| Change | Cause | Why it was needed |
|---|---|---|
| Training flags `--l_jitter`, `--l_mix_p_gt`, `--l_interp`, λ conditioning; test flag `--l_level` (all default off) | New experiments S2–S4 | Default paths proven unchanged by golden tests against a round-1 snapshot |
| `tools/lgeom.py` alignment, relabelling, terminal-bond fix, `pair_ok` | Naive blending collapsed C–H to ~0.62 Å; O–H/N–H shrank 0.10–0.15 Å | Physically valid blends for S1 and S4 |
| Seed builder keeps molecules, replaces unsafe pairs | 358 of 997 test molecules were silently dropped | Representative test sets (955 / 906 molecules) |
| Memory requests ≤ 5000 MB per CPU (6 CPUs / 30 GB per training) | The partition cap silently turned 48 GB into 10 CPUs per training | Four trainings would have taken 40 of 48 shared CPUs |
| Packed evaluations fail loudly; provenance records code commit and pack level | Failed packed evaluations were not recorded | Missing results would only surface at analysis time |
| Inference packing (3 per GPU) | Throughput; packed results vary no more than unpacked reruns (round-1 code is not bit-reproducible across processes: COV-R 89.0286 vs 89.0293) | 1.47× faster evaluation (your decision) |
| Checkpoint resume (`--resume`, atomic `last_model.pt` with optimiser, scheduler, best value and RNG states), 4-day limit, automatic requeue | A crashed or time-limited job would restart from zero | Your request; tested on gnode118 (kill → resumed at epoch 1; time-limit signal → requeued and finished) |
| Pre-declared analysis driver `tools/analyze_round2.py` | Without it, comparisons could be chosen after seeing results | Every comparison and correction family fixed in code before any result |
| `WORK`/`RES` paths derived only from the project directory | An environment variable on the cluster could redirect outputs to the 30 GB home folder | Safety |
| One evaluation task rerun (job 11155) | A GPU out-of-memory error while another job briefly occupied that GPU (our process used 1.2 GB of 24) | Caught by the new failure check |

## Infrastructure and repository changes (this period)

| Change | Cause | Why |
|---|---|---|
| SSH alias `ada` → `ada-gw1.iiit.ac.in` | `ada.iiit.ac.in` stopped accepting SSH | Your existing key works on the new gateway; no password used |
| One git repo, pushed to `Nishanth-nishu/TorDiff-NAX` | You asked to push everything | Code history merged in; PDFs, pickles, checkpoints and large CSVs excluded |
| LF line endings enforced | Windows line endings break bash scripts on the cluster | — |
| History rewritten so you are the only contributor; commits linked to your account via your GitHub no-reply address | Your request | `torsional-diffusion/LICENSE` (MIT, Jing & Corso) kept, as the license requires; backups in `.nested_git_backup/` |

---

# Part 3. Round 3: designing the next experiments (in progress, nothing run yet)

**Why a new process.** Rounds 1–2 showed what goes wrong without it: quotes cited for claims they did not support,
numbers overstated 2–4×, a 36 % test-set loss, and S2's literature support (random noise augmentation) not matching
our actual problem (systematic ring error). The new protocol (`orchestration/README.md`) requires, for every
candidate experiment: verified evidence from papers *and* blogs, a mapping onto our code, a predicted effect on *our*
numbers before any GPU time, and a "worth trying" verdict from a judge panel.

**Status:** 3 scouts wrote 17 cards with 166 evidence entries (`round3/cards/`, `round3/ledger/`). 3 verifiers checked
every entry: ROBUST 61/61 verified after its disputes closed; EVAL 49/54 and GEOM 47/51 verified before their
disputes, which are being answered now. Next: code grounding, impact prediction, judge panel, then your approval.

**Candidate experiments so far (scouts' own calls, before the panel):**

| Card | Idea | Why it could matter here | Scout's call |
|---|---|---|---|
| C-GEOM-01 | Relax RDKit structures with GFN2-xTB before diffusion | GEOM's own references are GFN2-xTB minima; MMFF already gave −0.026 Å | YES (CPU gate first) |
| C-GEOM-02 | Ring geometry from templates built from training molecules | Rings carry the floor (S1: true rings 0.178 → 0.119) | YES |
| C-GEOM-03 | Pretrained ET-Flow (0.073 Å) as the geometry source | Strongest published local structure | MAYBE (separate environment; verifier corrected both stated reasons) |
| C-GEOM-04 | Learned ring-pucker component (PuckerFlow / Cremer–Pople) | Directly targets wrong ring shapes | NO this round (too large) |
| C-GEOM-05 | ML force fields (AIMNet2, MACE-OFF) | — | NO (trained to DFT, not xTB; incompatible software stack) |
| C-GEOM-06 | srETKDGv3 starting structures | Better RDKit ring handling | MAYBE (cluster RDKit defaults to ETKDG v1; not single-factor) |
| C-ROBUST-01 | Error-shaped augmentation: train on true, RDKit and ring/acyclic hybrid geometry | S2's noise was the wrong shape | YES, gated on a ~3 GPU-h check |
| C-ROBUST-02 | S4 without the λ input | Isolates the λ input | MAYBE |
| C-ROBUST-03 | Train on the geometry source used at test time (MMFF first) | B1 can't use MMFF geometry | YES, gated on S5 |
| C-ROBUST-04 | Classifier-free-guidance-style geometry guidance | — | NO (geometry can't be dropped like a label; costs recall) |
| C-ROBUST-05 | Mixture-ratio sweep / curriculum | — | NO (S3 is already 0.001 Å from CTRL on RDKit L) |
| C-EVAL-01 | One comparable reporting block (fine δ, failures, intersection) | Our non-oracle results (0.150–0.182) sit at OMEGA level; published learned models reach 0.073–0.103 | YES (CPU) |
| C-EVAL-02 | Local-geometry error metrics of generated conformers | Shows where improvements come from | YES (CPU) |
| C-EVAL-03 | GFN2-xTB relaxation-energy metrics | Reference-free check of geometry quality | YES, if xtb installs |
| C-EVAL-04 | Non-learned baselines (ETKDG+MMFF, ETKDG+xTB, RDKit + clustering) | Any FlexiTors result must beat them | YES / MAYBE |
| C-EVAL-05 | xTB relaxation after TD, as a yardstick | — | YES |
| C-EVAL-06 | DFT re-referencing, Boltzmann ensemble metrics | — | NO (cost, not our reference) |

---

# Part 4. What the experiments say so far

1. **TD on QM9 is limited by the geometry it cannot change, not by its torsion model.** True geometry halves the error
   (E6); with true geometry even random torsions beat TD (E7); up to ~72 % of per-molecule error needs a geometry
   change (E12).
2. **The geometry that matters is ring geometry.** Floor split (E12) and, with the model in the loop, S1: true rings
   0.178 → 0.119, true acyclic geometry no gain. A FlexiTors that only diffuses acyclic angles would miss most of it.
3. **Conformer matching, stereo and sampler settings are not the lever** (B2, B5, G1, E12 headroom).
4. **A torsion model trained on true geometry is brittle** (B1). The fix that works so far is to train on a mix of
   real RDKit and true geometry (S3: 0.182 on RDKit, 0.033 on true geometry, 2 seeds). Random noise (S2) and the
   λ input (S4) were weaker.
5. **The accuracy target for a geometry generator** is roughly "blend 0.75": rings almost never twisted more than 10°,
   ring angles within ~0.7°, ring bonds within ~0.01 Å. RDKit and MMFF are far from it on rings.
6. **Cheap physics already helps** (MMFF before diffusion: −15 %). GEOM's own level of theory is GFN2-xTB, so xTB
   relaxation is the next cheap thing to test (round 3).
7. **The official 0.5 Å coverage cannot see these effects**; AMR-R and fine thresholds can.
8. **Honest position against the literature:** our non-oracle results (0.150–0.182 Å) are at the level of OMEGA, a
   classical generator; current learned models report 0.073–0.103 Å on the same protocol. The oracle results (0.020–
   0.033 Å) show the ceiling is far below that, if the geometry can be produced.

**Experiments that were clearly worth it:** E1, E2, E6, E9, E12, E15, S1, S3. **Useful but expensive for what they
showed:** B2, B5 (null results at ~33 GPU-h each), S4. **Informative failure:** S2 (showed random noise is the wrong
augmentation, which led directly to the round-3 error-shaped idea).

**Will this produce a real result?** The oracle experiments say the potential gain is large (0.18 → 0.02–0.03 Å) and
real. Whether FlexiTors can capture it depends on producing ring geometry close to the "blend 0.75" target without
oracle information. Round 3 tests the cheapest routes to that (xTB relaxation, ring templates, error-shaped training)
before anything is built from scratch.

---

**Source files:** `RESULTS_QM9.md`; `cluster_sync/results/analysis/*.md` (round-1 paired tables);
`cluster_sync/results/qm9_default/*/floor_run.log`, `cluster_sync/results/analysis/local_structure_test.log`;
`cluster_sync/round2/results/*/*/summary.txt`; `notes/ablation_plan_qm9.md`; `round2/*.md`; `round3/*`;
`orchestration/`.
