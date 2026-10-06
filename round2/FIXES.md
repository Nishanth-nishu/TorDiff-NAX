# Round 2: consolidated fix list before submission

Merged 2026-10-07 from `review_1.md` (coding agent 1), `review_3.md` (coding agent 3), `research_check_A.md` and
`research_check_B.md` (literature check of the implemented code). Both research agents found nothing WRONG and nothing
that contradicts TD's design; the items below are the caveats, plus the reviewers' bugs.

## User ruling kept (2026-10-07)
S4 trains on pair_ok-safe pairs only, and controls are re-scored on S4's molecule subset. Both research agents showed
this removes the test-set difference but not the training-set one (S4 trains on ~4% fewer molecules than its
controls). The user chose to keep the ruling. **Report this as a stated limitation next to every S4 result.**

## Blockers
- **X1. S1 seed pickles lose 358 of 997 test molecules** (all λ and A5 sets: 639 molecules;
  `tools/make_l_seed_pickles.py:182-184` drops a molecule if any one pair fails pair_ok). Fix:
  - A5ring / A5acyc do not interpolate, so they must not be pair_ok-filtered at all.
  - λ sets: replace each unsafe seed by cycling over that molecule's safe pairs; drop a molecule only if it has no
    safe pair. Write per-set kept/replaced/dropped counts to population.txt. Target ≥ 95% of the 997 molecules.
  - Check pair_ok on the λ grid {0.25, 0.5, 0.75}, not only λ = 0.5 (research_check_A I1).
- **X2. Memory requests silently raise CPU counts.** plafnet2 caps memory at 5000 MB per CPU, so `--mem=48G` with
  `--cpus-per-task=4` becomes ~10 CPUs. Set training to `--mem=24G` (round-1 peak 15-24 GB) and check every round-2
  sbatch for mem/CPU > 5000 MB.

## Should-fix
- X3. Packed evaluations hide failures (`slurm/common.sh:150`): the array task must exit non-zero if any packed
  sub-evaluation fails, and the failure must be logged.
- X4. `PACK=3` as default (user ruling). Write PACK and the code commit hash into each run's `provenance.txt`.
- X5. A5 has no control: add CTRL_rematch ×3 on the A5 sets.
- X6. `WORK` / `RES` must not inherit from the environment (`$WORK` may be set on ada → outputs land in `$HOME`).
  Derive them only from `$PROJECT`.
- X7. S5 training lines need an `afterok` dependency on the MMFF featurize job (10106 or its successor).
- X8. Restructure post-training evaluation dependencies so GPUs are not idle (~30 GPU-h lost, review_3).
- X9. S4 gate λ = 1 check: compare to gtLcycle with a tolerance derived from the measured seed SD, not exact.
- X10. Round-2 analysis driver (new script, run after the arrays): every comparison and Holm family pre-declared in
  code before results exist; S4 subset re-scoring of CTRL_rematch and B1 (user ruling); success-intersection
  COV-R@0.1 secondary; failure counts; COV-R@0.05 as exploratory; S1/S4 plotted against the measured L error
  (`l_error.csv`), not against λ (both research checks); report training-seed SD next to every bootstrap CI, and
  judge the S2 0.004 Å margin with seed variance included (research_check_B I7).
- X11. Count test-time MMFF failures in the S1 MMFF seed set and the S5 `--pre_mmff` controls (both research checks, I5).
- X12. The leak test must assert that generated conformers keep their seed order (research_check_B I7).
- X13. Fix the stale docstring at `tools/s4_gate.py:4`.

## Notes for the write-up (no code change)
- S2 tests robustness to random L error only; RDKit's error is systematic and mostly in rings (both research checks).
- S5 is the strongest MMFF baseline, not the strongest fixed-L baseline: GEOM references are GFN2-xTB, and MMFF lowers
  our mean floor only 0.0986 → 0.0934 Å (research_check_A I5).
- The pair_ok clash threshold of 0.9 Å is loose (research_check_A). It is kept so S4's training population is not
  shrunk further under the user's ruling; its value is reported.
- Measured training speed (~6.4 it/s under load) puts round 2 above the 175 GPU-h cap. Report the new estimate to the
  user rather than trimming more arms (review_1).
