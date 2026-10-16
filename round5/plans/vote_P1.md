# Vote of coding planner P1 (round 5, Stage A)

Written 2026-10-16 after reading `plan_P2.md`, `plan_P3.md`, every `## Answer` in `round5/consult/` (Q-P1-001…011,
Q-P2-001…011, Q-P3-001…009), EXP-1 rev. 2 §10, EXP-3 §9, EXP-5 §12, EVAL §1.1 and §3.1, plus the orchestrator's
data:
- re-matching took 2.9–4.3 h on 16 CPUs (jobs 538/540/542); the MMFF variant took 11.0 h on 6 CPUs (job 10105);
- the DRUGS download is approved and queued (job 24342).

Line numbers refer to the plan files as they are now.

## 1. Defects (a failure would occur even under the old specs)

### plan_P2
| # | Where | Defect | Failure scenario |
|---|---|---|---|
| P2-a | `plan_P2.md:165-166`, `:306-311` | K1 is implemented as an **update mask** in the reverse loop, and the bond stays in `edge_mask`. But TD's `perturb_seeds` (`sampling.py:105-108`, uniform draw at `:107`) randomises every masked edge before the loop starts, and P2 hooks only the loop (`td_backend`, `:162-164`, does not change `perturb_seeds`). | A "frozen" double bond starts from a uniformly random dihedral and is then held there, so K1 freezes a random isomer: E/Z fidelity ≈ 50 % by construction. The spec's `edge_mask` edit (EXP-3 §3.2) avoids this. |
| P2-b | `:341-349` | The pool size max(128, 12K) rests on "largest K is 30", read from the capped local-structure csv. | The test csv has K up to 727 (E-SAMPLE-054), so 12K reaches 8,724 conformers per molecule. The 2.4 GPU-h estimate (`:535`, Σ2K = 15,108) is wrong by about 2×. "128 ≥ 4K for every molecule" is false for 80 molecules. |
| P2-c | `:365-367`, V-E5-6 `:438` | PG arms "set `batch_size ≥ max proposals` (≤ 120)" and assert one batch per molecule. | With K = 727 a molecule needs 2,908 conformers in one batch. The assertion fails, or the run OOMs, and the setup is not PG's published protocol (E-SAMPLE-033). |
| P2-d | `:140-143` | Training draws t ~ U[0, 1] and regresses the target γ′(t)·P z, which diverges as t → 0 or 1; the plan has no handling. | Rare t near an endpoint gives targets of order 10–100 Å. With `--fail_on_nan` a NaN aborts M1; otherwise the loss is dominated by a few samples. |
| P2-e | `:211-217` | The floor re-run covers only the n_GT ≤ 3 rows, and the other-half variant uses parity over the capped list. | The other-half column of molecules with n_GT > 3 is never computed, so the split-half analogue V5 asked for is incomplete. |
| P2-f | `:109-110`, V-E1-9 `:399` | The chirality guard covers every degree-4 atom, not the RDKit stereocentres plus bridgeheads that EXP-1 §2.8 names. | CH₂/CH₃ "inversions" (a relabelling of identical H) are backtracked. Gate 0 then counts backtracks and skips that are not stereo events, which confounds EXP-1 §3 row 9. |
| P2-g | `:66-74`, `:503` | Stage A creates `geometry_eval.py`, `energy.py` (xTB), `strata.py`, `report.py`, `timing.py`, `selection_guard.py` and `vulture`/`coverage` tooling. `energy.py` serves only an xTB Gate-0 reference row that no spec asks for (`:280-283`). | Larger review surface; vulture/coverage need a package install (a download the user must approve). |

### plan_P3
| # | Where | Defect | Failure scenario |
|---|---|---|---|
| P3-a | `plan_P3.md:153` vs `:395`, `:429` | `rigid_stratum(csv)` is defined as TD `edge_mask.sum() == 0`. The Gate-0 stratum is n_rot_heavy = 0 (289 molecules, E-JOINT-007), and those molecules mostly have TD-rotatable methyl/OH rotors. | Gate 0 is read on a much smaller set than the one the 0.152 Å ETKDG baseline and the ≤ 0.10 Å threshold were defined on, so GO/NO-GO is decided on the wrong population. |
| P3-b | `:568-572` | Patches the frozen TD `utils/xtb.py`, although the plan says no Stage A TD call path uses it (`:571-572`). It also builds a separate `XTBRunner` (`:573-577`). | A change to the frozen baseline with no consumer is dead code, and two xTB code paths exist. It also breaks the SHA pins of `tools/tests/test_leakfree.py` style frozen-baseline checks if those are extended to `utils/`. |
| P3-c | `:662-663` vs `:756-757` | `GPU_MAX=3` (`--array=…%3`), yet the schedule runs M1 on GPUs 1–3 and the QM9 inference wave on GPU 4 at the same time. | Either the inference wave waits about 5.5 h behind M1, which voids the 14–16 h wall-clock, or the cap is ignored. |
| P3-d | `:376-379` | M0 reads `L_lam0.00_cyc_ORACLE.pkl` / `L_lam1.00_cyc_ORACLE.pkl`, the test-time λ seed pickles. | These are the pickles with the round-2 λ-blend defect (`tools/make_l_seed_pickles.py:158`, ≈ 10 % of slots carry another conformer's blend). The M0 recovery would be computed on mispaired x_R/x_G. |
| P3-e | `:230-236` | `reverse_sde` is a line-for-line port that includes the PG branch (`sampling.py:122-246`). | ≈ 60 copied lines that duplicate TD's own `sample()`. PG arms can call TD directly (P1, P2), so the copy is avoidable surface (G1-PG exists only because of it). |
| P3-f | `:556-557` | Claims a byte-identical re-run "because the seeds are fixed". | `evaluate`-side `Pool` order and DE are deterministic, but the new column changes the CSV schema. The comparison must be column-wise; a whole-file byte comparison fails spuriously. Minor. |

### plan_P1 (self-review, for fairness)
| # | Where | Defect |
|---|---|---|
| P1-a | `plan_P1.md` §1.3 `stereo.py`, §2.3 guard | δ_b and drift are measured with TD's `get_torsion_angles`, which clamps cos at ±(1 − 1e-5) (`torsion.py:120`) and cannot resolve < ≈ 4.5e-3 rad. As a result `test_invariance.py`'s 1e-3 rad criterion cannot pass or fail meaningfully (P2 caught this). |
| P1-b | §2.2 "Training" | The `pair_ok` filter was reused via the attribute that `dataset.py:145-162` reads. That drops whole conformers (and their τ loss), so the τ training data differ from A0/A3. |
| P1-c | §2.3 | O and the arms use `generate_confs.py`'s single global RNG stream. A subset run therefore does not reproduce the same molecules' trajectories, and splicing against R_s is valid only in distribution, not per molecule. |

## 2. Where each plan now conflicts with the research answers

| Binding decision | P1 | P2 | P3 |
|---|---|---|---|
| x₁ prediction, t ~ U[0, 1], no ε (EXP-1 §10.1.4) | conflicts (t_ε = 1e-3) | conflicts (velocity target, P2-d) | conflicts (`eps_t` 1e-3, `:201`) |
| W rows heavy-first, drop near-linear rows (< sin 10°), atan2 (§10.1.2) | conflicts (TD rule, acos) | partly (TD default, drop rule and atan2 already in) | conflicts (TD rule, `:163-167`) |
| Pure-L target x_G\* (§10.1.3) | conflicts | conflicts (`:142-143`) | conflicts |
| Exact torsion restoration ON in every sampler (§10.2) | conflicts | conflicts (off, `:394`) | conflicts |
| x_t chirality: resample ≤ 5, then mask (§10.1.5) | conflicts (count) | conflicts (count) | not addressed |
| `pair_ok`: mask geometry loss, keep τ loss (§10.1.1) | conflicts (P1-b) | not addressed | matches |
| Loss / rank(P_W) | conflicts (3n − 6 − m) | conflicts | conflicts |
| M0 = oracle-velocity integration on `cache_paired.val` (§10.3) | conflicts (test seeds, linear step) | conflicts (linear oracle step, `:235-250`) | conflicts (test λ pickles, `:376-379`) |
| Validation decides σ_b\* **and** GO; test read once (§10.3) | conflicts (test decides GO) | matches (`:276-279`) | conflicts (`:429-431`) |
| Validation ETKDG/MMFF baselines and paired ≥ 4-ring validation seeds, built on CPU first (§10.3) | missing | missing (no builder for validation pairs) | missing |
| A6b from fresh ETKDG; E_D = 20; new A1+ (§10.4; M2 only) | conflicts (cache seeds) | n/a | n/a |
| ref_half = exact expectation primary, one Holm family of 8 (§10.5) | conflicts (10 draws primary) | partly (closed form only as a check) | conflicts (10 draws, descriptive, `:540-542`) |
| Floor: `split.tsv` halves, cap after splitting, both `--mode test` and `--mmff_seeds` (§10.5) | conflicts (parity, test only) | conflicts (P2-e) | partly (`split.tsv` yes, test only `:553`) |
| Per-molecule seeding; O regenerated on the subset with arm seeding (EXP-3 §9.1) | conflicts (P1-c) | matches (MolRng, O-paired) | matches (`:289-302`) |
| K1 on specified structural bonds only; late planarity fold for unspecified/nonstereo bonds (§9.2) | conflicts (freezes all constrained bonds at the seed) | conflicts (`source = seed`) | conflicts (`seed3d`, `:172-173`) |
| QM9 endpoint = GT consistency on 37; DRUGS = fidelity on 126 (§9.3) | conflicts | partly | conflicts (`:478-479`) |
| DRUGS: 140 structural molecules, 1 seed first, 4.5–5 GPU-h cap, enriched set → Stage B (§9.4) | conflicts (141, 3 seeds, enriched set in Stage A) | conflicts (DRUGS only from the remainder, first to cut, `:537-540`) | nearly matches (141, cap 4.5) |
| Pool max(128, 4K) on seed 0, 4K on seeds 1–2 (EXP-5 §12.1) | **matches (adopted design)** | conflicts (12K, P2-b) | conflicts (c·2K only, `:511-514`) |
| PG: choose among the 4 README settings on validation at c = 1, batch 128 (§12.5) | conflicts (one fixed setting) | conflicts (5 configs at c = 2, one batch, `:363-367`, `:535`) | conflicts (untuned non-invariant, one batch, `:519-520`) |
| Selector: 5 candidates (q = 0.5 dropped, both FPS starts), margins, futility rule (§12.4) | partly (kept q = 0.5) | partly (kept q = 0.5, spec start) | partly (kept q = 0.5) |
| MMFF94s via TD `mmff_energy` (§12.4) | conflicts (MMFF94) | conflicts (MMFF94) | not fixed |
| Validity-adjusted secondary; invalid counts beside every primary row (EVAL §3.1) | conflicts (no `validity.py`) | matches | matches |
| Validation subset by `permutation(sorted(split[1]))`, builder must reproduce the test files ≥ 99 % (EVAL §1.1) | partly | matches (its acceptance test was adopted) | partly |

Counted against binding decisions: P2 matches or partly matches the most. P3 matches the Exp 3 seeding and
runtime. P1 matches D-101 exactly but conflicts on most Exp 1 points.

## 3. Votes (one line each)

| Item | Vote | Reason |
|---|---|---|
| Base plan | **P2**, with P3's runtime machinery and P1's gateway/module minimalism grafted in | Most of the binding answers adopted P2's defaults (validation-decides, atan2, near-linear drop, validity secondary, A1+ epoch control, validation builder test); P2's defects (P2-a/b/c/d) are local and now overridden by the answers. |
| Package layout | **P1's layout** (one `td.py` gateway re-exporting TD and `tools/` symbols, one `generate` CLI, one `posthoc` CLI, no xTB/strata/report/timing/selection-guard modules in Stage A) **plus** `eval/validity.py` (EVAL §3.1) and a seeding helper | Smallest surface that every Stage A job executes. P3's `_td.py` + `data/provenance.py` are also fine; P3's TD `xtb.py` patch (P3-b) is rejected. |
| Sampler hook design | **One reverse loop** (golden = `sample()`); hooks edit the per-edge `perturb` vector and run a position step; z drawn on the full TD edge set (P2, common random numbers); K1 by `edge_mask`/`mask_rotate` edit **before** `perturb_seeds` (spec, P1/P3, avoids P2-a); PG through TD's own `sample()`, not a port (P1/P2, avoids P3-e) | Exact equality with TD, no copied PG branch, and K1 freezes the correct isomer. |
| D-101 pool | **max(128, 4K) on seed 0, 4K on seeds 1–2; arms read prefixes; "cap never binds" test** (EXP-5 §12.1 = P1's proposal) | Adopted by R-SAMPLE. P2's 12K is built on a false K bound; P3's c·2K leaves D-101 open. |
| D-102 arms | **P0, P1, P2, P2b (PG 4K → random 2K), P3, P6 + ORACLE**; primary {P3 − P1, P3 − P2b}; PG chosen among the 4 README settings on validation at c = 1, batch 128 (EXP-5 §12.2–12.5) | All three plans had the NFE-matched PG arm; batch 128 and validation tuning are binding, and one-batch-per-molecule is infeasible at K = 727. |
| Exp 1 M1 design | **EXP-1 rev. 2 §10 verbatim**: whole network from scratch, w_τ = 0, x₁ prediction, heavy-first W with near-linear drop, x_G\* target, `pair_ok` and chirality masks, rank normalisation, restoration on, σ_min τ-frozen read-out, N = 20 Euler steps, M0 = oracle-velocity test on `cache_paired.val`, validation decides σ_b\* and GO | None of the plans implements this as written. A **CPU builder for validation paired ≥ 4-ring seeds and ETKDG/MMFF validation baselines** (Hungarian on heavy-angle RMSD, as `make_l_seed_pickles.py:108-114`) must be added to whichever base is chosen, because all three omit it. |
| Exp 3 scope | **EXP-3 §9**: per-molecule seeding (P3 §3.2); O regenerated on the subset with arm seeding, R_s / the released DRUGS pickle as filler only; K1 on specified structural bonds only; late planarity fold; geometric label primary; atan2 δ_b with the measured update sign s_b (P1/P2); QM9 GT consistency on 37; DRUGS 140 molecules, 1 seed first, 4.5–5 GPU-h hard cap; enriched set → Stage B | Binding, and it avoids P2-a and P1-c. |
| Cluster / job machinery | **P3's** (provenance stamps, per-molecule resume, OOM retry, `STATUS.md`, GPU-h ledger with `FX_BUDGET_STOP`, pre-flight, DRY_RUN), with `GPU_MAX` made consistent with the schedule (P3-c) and no TD file patches | It is the only plan that makes the mechanical gates and the budget self-enforcing. P1/P2 rely on the round-2 sbatch patterns alone. |
| Schedule and cut order | **≈ 23–24 GPU-h**: smoke 0.5; M1 10.5 (ceiling 16, restated after the 1-epoch smoke; P3's packing test allowed); Gate-0 validation + test read-out ≈ 0.6; Exp 5 ≈ 6–7 (EXP-5 §12.6); Exp 3 QM9 O-regen + spliced arms ≈ 0.8; DRUGS ≤ 4.5–5 (download job 24342 already queued). Cut order: (1) DRUGS K1-canonical, (2) DRUGS K1-wrong (with the EXP-3 §9.4 "no stereo claim" caveat), (3) Exp 5 P6 (keep P2b), (4) DRUGS seeds 2–3. Never cut: the 3 σ_b runs, O, P1/P2b, QM9 K1-wrong, any golden test | Combines P3's mechanical ledger with the binding caps. **M2 costing note:** re-matching measured at 2.9–4.3 h × 16 CPUs (≈ 46–69 CPU-h) per QM9 training-set roll-out, and 11.0 h × 6 CPUs (≈ 66 CPU-h) for the MMFF variant. A5 DAgger and A6b each need one, so the M2 ask carries ≈ 100–140 CPU-h of matching. |

## 4. Final ranking
1. **P2.** It aligns best with the binding answers and has the most complete validity checks. Its concrete defects
   (P2-a K1 randomisation, P2-b/c K-bound, P2-d γ′ divergence) are each fixed by one line of the answers or of the
   spec.
2. **P3.** It has the best runtime and the per-molecule seeding that EXP-3 §9.1 now requires. Its concrete defects
   (P3-a wrong rigid stratum at the gate, P3-b TD patch with no consumer, P3-c GPU cap vs schedule, P3-d defective M0
   pickles) touch the Exp 1 gate and the frozen-baseline rule.
3. **P1** (mine). Its smallest layout and its D-101 design were adopted. But it conflicts with most of the Exp 1
   decisions and with the Exp 3 seeding, and it has its own defects P1-a…c. I do not rank it first, because the other
   two plans' defects are no worse than mine.
