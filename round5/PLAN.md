# Round 5: paper programme (user's 6-phase plan, 5 experiments)

Written 2026-10-15. Every round-5 agent reads `orchestration/README.md` (protocol v2), its role file, `round4/BRIEF.md`
(including the round-2 final analysis) and this plan first.

## Where the 6 phases stand
| Phase | Status | Evidence |
|---|---|---|
| 1 Reproduce baseline | Done | R0 0.176 vs paper 0.178 Å; retrain 0.181; evaluator sanity 0.000; paper's 92.8 % COV-R not reproducible from its own released conformers (88.8 %) — `RESULTS_QM9.md` |
| 2 Establish error source | Done, with a correction | Floors: up to ~72 % of per-molecule error needs a geometry change. **Leak-free split-half test (2026-10-15, `cluster_sync/round4/analysis_r4_leakfree/`): with another conformer's true geometry every model is worse than with RDKit geometry** (CTRL 0.220 vs 0.187 Å; B1 0.305; S3 0.280; S4 0.253; copy-other-half 0.334). Own-conformer gains (B1 0.030, S3 0.053) are conformer-identity leakage. **Conclusion: geometry helps only when it is coupled to the conformer being generated** — the case for joint L–τ modelling (Exp 1), and against an independent geometry predictor. A1 (−54 %) and the ring-only result (0.119) used own-conformer geometry and must be re-read as upper bounds that include leakage. |
| 3 Smallest viable extension | Next | Exp 1 stage 1: constrained local-geometry correction module |
| 4 Controlled ablations | Next | Baseline / correction alone / torsion alone / joint; same splits, samples, budgets |
| 5 Generalisation | Later | Size and flexibility strata, OOD set, GEOM-DRUGS (TD released DRUGS checkpoint + conformers, `round4/cards/C-BENCH-*`) |
| 6 Evidence-based novelty | Throughout | Novelty red-team before claims; compare against RINGER, GO-Flow, PuckerFlow, multi-scale literature |

## Experiments and order
Order chosen for value per effort and to keep progress unblocked by usage limits:
1. **Exp 5 — uncertainty-aware adaptive sampling** (medium; pretrained model; inference only). Fast real results.
2. **Exp 1 — joint local-geometry + torsion diffusion** (high; the main contribution). Stage 1 = coupled correction
   module conditioned on τ; stage 2 = joint diffusion.
3. **Exp 3 — stereochemistry-constrained torsional diffusion** (medium; needs an E/Z-enriched test set, likely from
   GEOM-DRUGS since QM9 has few stereo double bonds — to be counted).
4. **Exp 2 — matching-free training** (medium–high; round-1 B1/B2 are partial variants A/B already run).
5. **Exp 4 — full Boltzmann generator** (very high; scoped to a feasibility study unless Exp 1 succeeds).

## Code structure (target)
A clean package for new work, the original TD code kept as the frozen baseline:
- `torsional-diffusion/` — baseline (upstream + round-1/2 hooks), frozen except bug fixes.
- `flexitors/` — new package: `geometry/` (internal coordinates, ring puckers, constraints), `models/`, `sampling/`
  (incl. adaptive controller), `training/`, `eval/` (metrics, leak-free oracle, stereo checks, energy), `data/`.
- `experiments/<exp>/` — one config + one sbatch + one README per experiment; results in `results/<exp>/`.
- Rules: no dead code, typed functions with docstrings, tests per module, every mechanism cites its source in a
  comment (`# <paper>, PDF p. N, eq. M`), and a dead-code audit before each push.

## Agents (round 5)
- Research: R-JOINT (Exp 1 + Exp 4), R-CHEM (Exp 2 + Exp 3), R-SAMPLE (Exp 5 + Phase 5 evaluation). Each writes
  `round5/specs/EXP-<n>.md`: prior-art check (2023–2026, incl. RINGER and multi-scale methods; download, cite PDF page
  and exact quote), method with equations and sources, evaluation protocol, ablation table, coding instructions with
  `path:line` touchpoints, compute, risks, and a novelty verdict. Evidence goes in `round5/ledger/evidence_<TAG>.md`.
- Verifier: checks every spec's citations and numbers; disputes go back to the author.
- Coding: three agents plan per experiment, consult the research agents through `round5/consult/`, cross-review and
  vote; one implements, two review; the research author checks conformance before the push.
