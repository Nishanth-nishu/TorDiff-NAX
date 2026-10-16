# Vote P3 (round 5, Stage A): cross-review of plan_P1 and plan_P2, conflicts with the research answers, votes

Planner P3, 2026-10-16. Read: `round5/plans/plan_P1.md`, `plan_P2.md`, all 31 files in `round5/consult/` (answers are
binding), EXP-1 §10 (rev. 2), EXP-3 §9, EXP-5 §12, EVAL §1.1 / §3.1. Orchestrator data used: round-1 full QM9
re-matching 2.9–4.3 h on 16 CPUs; MMFF variant 11.0 h on 6 CPUs; the DRUGS download (4 files) is approved and queued
as job 24342. No code was changed and nothing was committed.

Severity: **BLOCKER** = a run gives an invalid or unusable result, or breaks a binding decision. **SHOULD-FIX** = it
costs GPU-h, reruns or agent round trips. **NIT** = minor.

## 1. Defects

### 1.1 plan_P1

| # | Sev. | Where | Defect and failure scenario |
|---|---|---|---|
| P1-1 | BLOCKER | `plan_P1.md:460-466`; golden `test_generate.py` `:613`; S5 `:646` | **The splicing design has no per-molecule seeding.** The generator is golden-equal to `generate_confs.py` (one global seed, `generate_confs.py:94-95`). In a subset run, K1 draws fewer uniforms in `perturb_seeds` (`sampling.py:107`, `size=edge_mask.sum()`) and a shorter `z` (`sampling.py:191`). So after the first affected molecule, O-subset and K1-subset use different random numbers on every later molecule. Effects: the "bit-identical on unaffected molecules" test (`:465`) fails, and the arms are not paired by common random numbers. This violates EXP-3 §9.1 condition 1 ("per-molecule seeding"). |
| P1-2 | BLOCKER | `:394-395` | GO is decided on **test** rigid AMR-R. EXP-1 §10.3 now says that validation decides both σ_b\* and GO, and test is read once afterwards. |
| P1-3 | BLOCKER | `:361-371` | M0 runs on test `L_etkdg2L` seeds with an AMR-R "recovery" for the rigid stratum. That criterion is vacuous, and P1 says so itself (`:369-370`). EXP-1 §10.3 replaces it with an oracle-velocity integration test on `cache_paired.val` pairs (≥ 1 heavy rotatable bond and a ≥ 4-atom ring), with kill rules on 50 % recovery and 5 % skipped or flipped pairs. |
| P1-4 | BLOCKER | `:326-333`, `:351-359` | Training uses a velocity target with t_ε = 1e-3, and the sampler has no torsion restoration. Both are superseded: EXP-1 §10.1.4 uses x₁ prediction with t ~ U[0,1], and §10.2 turns restoration ON. Also, the W rule is "first neighbour" (`:127-129`); §10.1.2 makes it heavy-first, drops near-linear rows and uses atan2. |
| P1-5 | SHOULD-FIX | `:472-476`, `:680` | DRUGS is planned with 3 seeds plus the enriched validation set (≈ 5 GPU-h). EXP-3 §9.4 says 1 seed first, a hard 4.5–5 GPU-h cap, the enriched set moved to Stage B, and 140 structural molecules (not 141). |
| P1-6 | SHOULD-FIX | `:524-527` | The PG default is the permutation-**invariant** README setting, untuned. §12.5 requires choosing among the 4 README settings on validation at c = 1. Failure mode of the invariant kernel: `sampling.py:145-174` enumerates graph isomorphisms under `time_limit(10)` (SIGALRM). On a timeout it silently falls back to the non-invariant kernel (`:172-174`), so one arm can mix two kernels without any count. If invariant is chosen, every run must count `Timeout generating …` lines. |
| P1-7 | SHOULD-FIX | `:622-627`, `§5.4` | Reliability covers training resume and requeue only. There is no in-job OOM retry, no resumable generation, no GPU-h ledger and no status digest. Round 2 lost two jobs to co-located GPU OOM (`round2/RUNLOG.md`, 10860/5 and 11176), and each cost a manual rerun plus agent turns. In Stage A the same failure in the 1.4 GPU-h seed-0 pool would cost the whole pool. |
| P1-8 | SHOULD-FIX | `:25`, `:731` | xTB is left out entirely. The task brief required the temp-dir race fix at `torsional-diffusion/utils/xtb.py:5-7` (shared `/tmp/<pid>` across forked workers, `:44-45`), and EXP-1 §3 row 7 has no runner. It is not needed for Gate 0, but the fix must land before any xTB use. |
| P1-9 | NIT | `:443` | F-filter does not say what happens when fewer than 2K valid conformers remain. EXP-3 §9.2 now requires filling with the least-violating conformers, counted. |

What P1 does better (adopt): the gateway rules (no `diffusion` / `utils` / `train` top-level names; the torus import under `chdir`; SHA pins of the frozen baseline, `:92-100`, `:602`); one evaluator run per pool, with arms scored as column subsets of the RMSD matrix (`:538-542`, a large CPU saving); the per-bond update sign s_b (`:432-435`).

### 1.2 plan_P2

| # | Sev. | Where | Defect and failure scenario |
|---|---|---|---|
| P2-1 | BLOCKER | `plan_P2.md:341-349`, `:535` | The test pool is max(128, 12K), justified by "the largest K is 30". That figure comes from the capped local-structure csv. On `test_smiles.csv`, K runs to 727, and 86 molecules have K > 30 (E-SAMPLE-054). For K = 727, 12K = 8,724 conformers for one molecule; the pool would be 232,804 conformers, not ≈ 10.1 × Σ2K. The GPU estimate in item 6 is therefore wrong, and the "128 ≥ 4K for every molecule" claim is false for 80 molecules. EXP-5 §12.1 rejects this design. |
| P2-2 | SHOULD-FIX | `:365-367`, V-E5-6 `:438` | PG with `batch_size ≥` all proposals is rejected by §12.5: the published protocol uses batches of 128. On the cluster it is also risky: a 2,908-conformer batch for K = 727 builds an n × n × D torsion-difference tensor (`sampling.py:219-223`) and deep-copies the whole batch to the GPU (`:179`). That is a co-located-OOM magnet on a shared 3090. |
| P2-3 | BLOCKER | `:539-541` | Cut step (1), "DRUGS moves to a follow-up ask", removes Exp 3's **primary** endpoint. EXP-3 §9.3 makes DRUGS fidelity on the 126 specified molecules primary and QM9 confirmatory, and §9.4 gives DRUGS an approved 4.5–5 GPU-h cap. Item 7 (`:537`, "≤ remainder of 25") has no fixed cap. |
| P2-4 | BLOCKER | `:394`, `:143-145` | `restore_torsions` is off by default, chirality-flipping x_t samples are counted rather than rejected, and the target is a velocity. All three are superseded by EXP-1 §10.2, §10.1.5 (resample ≤ 5 times, then mask) and §10.1.4 (x₁ prediction). |
| P2-5 | SHOULD-FIX | `:71`, `:280-283` | xTB runs at Gate 0 through `eval/energy.py`, but nothing addresses the TD race (`utils/xtb.py:5-7`, `:44-45`), `--norestart`, convergence checks or the module environment (the module is not initialised in non-login shells, `round3/grounding.md:42-44`). If the runner reuses TD's `xtb_optimize` inside a Pool, molecules with equal atom counts can silently swap coordinates (S-1.1). |
| P2-6 | SHOULD-FIX | `:521-552` | Reliability is the round-2 baseline only. There is no OOM retry, no resumable generation, no GPU-h ledger and no machine-readable status; the schedule is in days, with the smoke on day 4. Same failure scenario as P1-7. |
| P2-7 | NIT | `:216-217` | The floor re-run covers only the n_GT ≤ 3 molecules of the test file, and G9 checks 30 molecules. EXP-1 §10.5 asks for both `--mode test` and `--mode test --mmff_seeds`. A full re-run on 16 CPUs takes about 1 h and gives a full-file regression check. |
| P2-8 | NIT | `:535` | PG tuning uses 5 configurations at c = 2 on 500 molecules. §12.5 says the 4 README settings at c = 1 on the 1000-molecule validation subset. |

What P2 does better (adopt): per-molecule RNG streams **plus** noise drawn on the full TD edge set, with hooks that edit only the per-edge update. K1 as an update mask (`:305-311`) therefore keeps common random numbers **within** an affected molecule, which my `edge_mask` edit does not. Also adopt `selection_guard` (V-X1), atan2 dihedrals (now binding, §10.1.2), the measured sign_b (V-E3-2), the hypergeometric closed form (now binding for ref_half, §10.5), and generating the test pool blind before the validation decision is hashed.

### 1.3 Gap shared by all three plans (including mine)
- **No validation ring-seed population.** EXP-1 §10.3 now reads the ≥ 4-ring criterion on validation, with the `l_error.csv`-style (k, src_gt_idx) pairing as primary. `l_error.csv` exists only for test. Stage A must therefore run `tools/make_l_seed_pickles.py --sources etkdg` on the validation csv: CPU, the same Hungarian pairing as `:106-114`, a few CPU-h. It must also compute the validation ETKDG and MMFF baselines that the > 10 % scaling rule needs. This job must finish before the M1 read-out.
- **Prefix property not guaranteed.** Sharing Exp 3's O-subset with Exp 5's P0 prefix requires `draw(n)` to equal the concatenation of two `draw(n/2)` calls (EXP-5 §9; P2 V-E5-3). With per-batch seeds and a 128-conformer pool batch, the first 2K of a max(128, 4K) call is not, in general, a 2K call. If that test fails, Exp 3 O is generated separately on its subset (≈ 0.1 GPU-h). Never assume the two are equal.

## 2. Conflicts with the binding research answers

| Topic (binding source) | P1 | P2 | P3 (mine) |
|---|---|---|---|
| x₁ prediction, t ~ U[0,1] (EXP-1 §10.1.4) | velocity + t_ε | velocity | velocity + ε (PENDING Q-P3-002, answered against) |
| W rule heavy-first, near-linear drop, atan2 (§10.1.2) | first neighbour | TD rule default, heavy-first as ablation, atan2 ✓ | TD rule |
| x_G\* pure-L target (§10.1.3) | missing | missing (diagnostic only) | missing |
| Chirality resample ≤ 5 then mask (§10.1.5) | count only | count only | not specified |
| Restoration ON (§10.2) | missing | off by default | missing |
| τ-frozen read-out σ = σ_min (§10.2) | ✓ | not specified | not specified |
| M0 = oracle integration on `cache_paired.val` (§10.3) | test seeds | validation pairs ✓ (single step) | test λ pickles |
| Validation decides σ_b\* **and** GO (§10.3) | test decides GO | ✓ | test decides GO |
| Whole network from scratch, clock via λ (§10.3) | ✓ | ✓ | ✓ |
| ref_half exact expectation + 10 draws (§10.5) | MC only | closed form ✓ | MC only |
| Floor: test + mmff_seeds files (§10.5) | test only | n_GT ≤ 3 only | test only |
| Per-molecule seeding; O regenerated on subset (EXP-3 §9.1) | ✗ (P1-1) | ✓ | ✓ (I regenerated O on all 1000; subset + R_s filler is enough for QM9) |
| NI on the affected subset, not all molecules (§9.3) | all molecules | all molecules | all molecules |
| Primary endpoints per dataset (§9.3) | fidelity | fidelity | fidelity |
| DRUGS 140 (+160 K1-can.), 1 seed, cap 4.5–5 (§9.4) | 141, 3 seeds, enriched set | DRUGS cut first | 141, 1 seed, 4.5 cap ✓ |
| Unspecified bonds: late planarity fold, not the seed state (§9.2) | seed state | seed state | seed state |
| Pool: seed 0 max(128, 4K), seeds 1–2 4K (EXP-5 §12.1) | ✓ | 12K ✗ | 2K + 2K, no K-free block |
| PG: 4 README settings chosen on validation at c = 1, batch 128 (§12.5) | invariant, untuned | 5 configs, batch = set ✗ | non-invariant untuned, batch = set ✗ |
| Selector rule +0.004 / −0.5 pt, futility, 5 candidates (§12.4) | "not worse" | ✓ | +0.002 ✗ |
| Validation set: no featurizer filter; builder must reproduce test (EVAL §1.1) | pending | ✓ | featurizer filter ✗ |
| Invalid conformers kept; secondary +∞ view (EVAL §3.1) | — | ✓ | — |

## 3. Votes (one line each)

1. **Base plan: P3.** It is the only plan that already satisfies EXP-3 §9.1 condition 1 *and* carries the
   cluster-reliability layer (OOM retry, resumable generation, GPU-h ledger, `STATUS.md`, xtb fix). Graft onto it
   P2's validity layer (§3 below) and P1's gateway and column-subset scoring.
2. **Package layout: P2's six-subpackage tree.** It equals mine, apart from names. Add P1's rules: one TD gateway
   module, no top-level `diffusion` / `utils` / `train` names, baseline SHA pins. Use one `eval/budget.py`, not
   three timing/budget modules.
3. **Sampler hook design: P2's.** One reverse loop ported from `sampling.py:176-246` (golden G1). `z` is drawn on the
   full edge set and hooks edit only the per-edge update. K1 is an update mask; `perturb_seeds` draws, then the frozen
   entries are zeroed. Use the measured sign_b and atan2. PG arms call TD's `sample()` under the per-molecule seed
   wrapper; do not port the PG branch (less code).
4. **D-101 pool: EXP-5 §12.1 (= P1's design).** Seed 0 draws max(128, 4K_i) in one call; seeds 1–2 draw 4K_i. A test
   asserts that arms read only prefixes and never read N_i. Score each pool once and take arms as column subsets
   (P1). Generate the test pools blind (P2).
5. **D-102 arms: §12.2 / §12.4 / §12.5.** Arms P0, P1, P2, P2b (PG 4K in batches of 128 → random 2K), P3, P6. The PG
   setting is chosen on validation among the 4 README settings. The selector is chosen with the
   +0.004 Å / −0.5 pt / futility rule. If the invariant kernel is chosen, PG timeouts are counted (P1-6).
6. **Exp 1 M1: EXP-1 rev. 2 §10 verbatim.** That is: x₁ prediction, heavy-first W with atan2, the x_G\* target,
   resample-then-mask, restoration ON, σ_min τ-frozen read-out, whole network from scratch, the new M0 on validation,
   and validation deciding σ_b\* and GO. Also add the validation ring-seed job (§1.3). Keep my mechanical smoke: the
   1-epoch overhead plus the packing test. Packing is the cheapest legal saving (≈ 3–5 GPU-h if s ≥ 1.8), and an
   `FX_BUDGET_STOP` fires above 16 GPU-h.
7. **Exp 3 scope: EXP-3 §9.**
   - QM9: 102 / 211-molecule subsets, O regenerated on the subset, R_s as filler, non-inferiority on the subset.
   - DRUGS: 140 molecules (+ the rest of 160 for K1-canonical), 1 seed, cap 4.5–5 GPU-h set by the 20-molecule
     timing run (DRUGS data arrive with job 24342).
   - Enriched set: Stage B.
8. **Cluster and job machinery: P3 §3 + §9.** Per-molecule seeding, JSON provenance, R2_STRICT checks with failure
   parity, OOM retry with backoff, resumable generation, GPU pre-flight, GPU-h ledger + `STATUS.md`, the xtb
   module / absolute-path environment, and the TD `utils/xtb.py` patch. Add P2's sbatch lint test and P1's
   `sbatch --test-only`.
9. **Schedule and cut order.** Recount with the binding answers:

   | Item | GPU-h |
   |---|---|
   | M1 | 10.5 (ceiling 16; ≈ 6–7 if packed) |
   | Exp 5 | 6–7 (§12.6) |
   | DRUGS | ≤ 5 |
   | QM9 Exp 3 | ≈ 0.5 |
   | Smokes, read-out, reserve | ≈ 1.5 |
   | **Total** | **≈ 23.5 expected, ≈ 29 worst case** |

   Cut order:
   - (0) Pack M1 if the smoke allows (no scientific cost).
   - (1) DRUGS K1-canonical.
   - (2) Exp 5 P6 (descriptive).
   - (3) DRUGS K1-wrong (then no "about stereo" claim, §9.4).
   - (4) Reduce PG validation tuning to the two non-invariant settings.
   - (5) Otherwise `FX_BUDGET_STOP` and ask the user.

   Never cut: the three σ_b runs, O / K1 / K2-inf, P0 / P1 / P2b / P3, the validation selections.

   DRUGS staging now runs in parallel with code building, so the GPU wave can start DRUGS as soon as the smoke passes.

## 4. Ranking

1. **P3**: covers the approved splicing precondition and the cluster-reliability layer. Its rev-2 conflicts (§2) are
   shared with P1, apart from its PG and selector-margin items, and are fixed by adopting §10 / §12 verbatim.
2. **P2**: the strongest validity scaffolding and the best hook design. It ranks below P3 because of P2-1 (a
   false-premise pool), P2-3 (cutting DRUGS removes Exp 3's primary endpoint) and the missing reliability layer
   (P2-6).
3. **P1**: the cleanest gateway and the cheapest pool scoring. It ranks last because of P1-1 (the splice is unpaired
   without per-molecule seeding) and P1-2/P1-3 (gate and M0 on test).
