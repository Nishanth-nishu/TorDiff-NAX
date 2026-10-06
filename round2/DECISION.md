# Round 2: implementation decision (tally of vote_1/2/3.md)

Tallied 2026-10-06. Ranking votes: agent 1 → 2 > 3 > 1; agent 2 → 1 > 2 > 3; agent 3 → 2 > 1 > 3.
**Base plan: `code_plan_2.md`** (2 of 3 first-place votes). The other plans' parts are grafted in as listed below.

| # | Decision | Votes | Outcome |
|---|---|---|---|
| D1 | F1 alignment | 3/3 | Heavy-atom Kabsch fit, applied to all atoms. Before interpolating, relabel symmetry-equivalent terminal atoms (methyl/NH2/CF3 H etc.). Per-pair `pair_ok` guard: bond-length/clash/chirality check at λ = 0.5; failing pairs are dropped and counted. |
| D2 | S1.5 acyclic set | 3/3 | Includes H, so ring-only and acyclic-only are exact complements. |
| D3 | S1 run matrix | 3/3 on keeping CTRL_rematch | Plan 2's matrix. Never cut CTRL_rematch runs; add CTRL_rematch gtLcycle ×3 (needed as the λ = 1 anchor, agent 1's finding). Use the full ~105 grid only if packing (D4) passes its test. |
| D4 | Inference packing | 3/3, conditional | Plan 3's 3 jobs/GPU with conditions batched per model, enabled only after (a) a timing test and (b) a packed rerun reproduces a round-1 SUMMARY line exactly; with a stale-output guard. Otherwise 1 job/GPU. |
| D5 | Training CPUs | 3/3, conditional | Add plan 3's per-epoch data-wait timer. Drop to 4 CPUs / 3 workers only if data-wait is < 5% of epoch time in the smoke test. No 2-per-GPU sharing by default. |
| D6 | B1cap control | 3/3 | Yes, 1 seed. |
| D7 | S4 this round | 3/3 | Yes, gated: pairing ≥ 98% clean, `pair_ok` passes, λ = 1 reproduces gtLcycle within seed SD, smoke test passes. If it passes, S4 runs in wave 2; if not, S5 takes its slot and S4 moves to round 3. |
| D8 | Budget | 3/3 | ≤ 175 GPU-h. Trim order: descriptive runs (S6, B1_gtL sampler checks), then non-cycle gtL runs. |
| D9 | Grafts | — | From plan 1: golden/regression tests proving defaults are byte-identical; the exact `mmff_error` count. From plan 3: inference packing (D4), data-wait timer, fail-fast on NaN only (no "not learning" or speed-based kills), resume only when explicitly requested (no forced resume), the CPU-light S5 standardization schedule. |

## Defects that must be fixed during implementation (raised in votes)
1. The plan 2 pairing fallback aborts the whole build on duplicate energies; it must disambiguate instead (vote_1).
2. `mmff_error` is undercounted in plans 2 and 3; use plan 1's exact count (vote_1, vote_3).
3. Seed builders must wrap MMFF in try/except like `diffusion/sampling.py:21-26` (vote_1).
4. Interpolation seeds and training data must handle the DE optimum the same way (`standardize_confs.py:106`, vote_1 / plan 3).
5. Plan 2's single inference lane under-states wall-clock (~53 h); the packed schedule (D4) fixes it if it passes (vote_3).
6. No change to default training behaviour (vote_1 on plan 3).
