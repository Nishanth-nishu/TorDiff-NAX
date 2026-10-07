# Round 2: run log (gnode118)

| When (IST) | Job | What | Result |
|---|---|---|---|
| 2026-10-07 | 10860 | Head evaluation array (8 model tasks, PACK=3) | 7/8 completed; task 5 failed (CUDA OOM: GPU briefly occupied by another job) |
| 2026-10-07 | 11155 | Rerun of head task 5 (CTRL_rematch s2) | Completed: all 13 tags have eval.pkl + SUMMARY |
| 2026-10-07 | — | S4 gate (`submit_round2.sh gate`) | PASS: pairing 99.999%, pair_ok 87.7% (info), V18 CR -0.0039 / B1 -0.0021 within tol 0.003, smoke clean |
| 2026-10-07 | 11168 | Main training array, 10 runs (B6 x3 + B6 0.02, S3 x2, S4 x3, B1cap), 4 at a time | running |
| 2026-10-07 | 11169 | S5 / B3 MMFF-matched training, 3 runs | queued |
| 2026-10-07 | 11170-11180, 11185 | Post-training evaluation panels, one per model (afterok on its training task) | queued |
