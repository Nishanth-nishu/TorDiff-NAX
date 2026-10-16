# Round 5 decisions

- **2026-10-16, user:** Stage A approved — build the clean `flexitors/` package for Exp 1 (first gate M1, with the
  smoke run first), Exp 3 (stereo-constrained sampler, inference on QM9 and GEOM-DRUGS with released checkpoints) and
  the selection part of Exp 5 (at matched compute; adaptive allocation dropped: headroom −0.0042 Å, V5). CPU: the
  budget-matched leak-free re-score (EXP-1 §3.6) and the fix for the floor bug at `tools/local_structure_analysis.py:302`.
  Budget ≈ 15–25 GPU-h. Exp 2 and Exp 4 wait for the Exp 1 gate.
- Open disputes to resolve in the plans: D-101 (fixed-size proposal pool, K-free) and D-102 (Particle Guidance arm
  must be NFE-matched).

## Stage A implementation decision (vote tally, 2026-10-16)
First-place votes: P1 → P2, P2 → P2, P3 → P3. **Base plan: `round5/plans/plan_P2.md`.** Binding research answers
override any plan text: EXP-1 rev 2 §10, EXP-3 §9, EXP-5 §12, EVAL §1.1/§3.1, and every `## Answer` in `round5/consult/`.

Grafts and fixes (all three votes, or one vote uncontested by the others):
1. Package layout and sampler hooks: P2's (one TD reverse loop with hooks; per-molecule RNG so arms stay paired within
   each molecule). P1's single import gateway `flexitors/td.py` + SHA-pin test of baseline files + dead-code test.
2. Cluster/job machinery: P3's (packing, provenance, failure detection, OOM retry inside the job, resumable generation,
   GPU-h ledger with a 25 GPU-h stop, auto-gates, `STATUS.md`). Drop P3's patch to TD `utils/xtb.py` (Stage A does not
   call it); the flexitors xtb runner uses a per-process temp dir.
3. K1 (E/Z freeze): frozen bonds must be excluded from the initial torsion randomisation (`perturb_seeds`) AND masked
   from updates — masking updates alone is wrong (P1 vote).
4. Rigid stratum = molecules with n_rot_heavy = 0, not `edge_mask.sum() == 0` (P1, P2 votes).
5. Gate 0 and M0 are decided on the validation split; the test set is read once afterwards (EXP-1 §10).
6. Missing in all plans: a CPU builder for the validation seed sets and baselines Gate 0 needs (incl. ring seeds), and
   the QM9 validation set per EXP-5 §12 (1000 molecules, seed-0 permutation of TD val split; must reproduce test-file
   construction for ≥ 99 % of rows).
7. Exact atan2 dihedrals everywhere drift/planarity is measured (not TD's clamped acos).
8. `pair_ok` failures are masked out of the geometry loss only, never dropped as whole conformers (EXP-1 §10).
9. Exp 5 pool and arms per EXP-5 §12 (K up to 727; seed-0 max(128, 4K) with a 128 K-free block; P2b NFE-matched PG).
10. Budget ≈ 23.5 GPU-h expected. Cut order if over 25: pack M1 first if the smoke allows; then Exp 5 P6; then reduced
    PG tuning; keep DRUGS K1 arms (Exp 3's primary endpoint is on DRUGS, EXP-3 §9). If still over 25 GPU-h, stop and
    ask the user.
