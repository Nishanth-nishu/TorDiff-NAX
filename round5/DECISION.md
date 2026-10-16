# Round 5 decisions

- **2026-10-16, user:** Stage A approved — build the clean `flexitors/` package for Exp 1 (first gate M1, with the
  smoke run first), Exp 3 (stereo-constrained sampler, inference on QM9 and GEOM-DRUGS with released checkpoints) and
  the selection part of Exp 5 (at matched compute; adaptive allocation dropped: headroom −0.0042 Å, V5). CPU: the
  budget-matched leak-free re-score (EXP-1 §3.6) and the fix for the floor bug at `tools/local_structure_analysis.py:302`.
  Budget ≈ 15–25 GPU-h. Exp 2 and Exp 4 wait for the Exp 1 gate.
- Open disputes to resolve in the plans: D-101 (fixed-size proposal pool, K-free) and D-102 (Particle Guidance arm
  must be NFE-matched).
