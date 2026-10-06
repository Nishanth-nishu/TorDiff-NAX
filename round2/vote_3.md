# Vote of coding agent 3 (round 2 implementation)

Written 2026-10-06 after reading `code_plan_1.md`, `code_plan_2.md` and my own `code_plan_3.md`. Code paths are relative
to `torsional-diffusion/` unless they start with `tools/` or `slurm/`. No code was changed.

---

## 1. Bugs and validity problems found

### Plan 1

| # | problem | evidence | severity |
|---|---|---|---|
| P1-1 | **The S3/S4 panels lack their F4 control.** §3.2 runs CTRL_rematch only on the interp series. But the S3 panel (etkdgG, mmffG, noise 0.02/0.04, interp0.50; §5) and the S4 `S1_mmffG_cyc` at λ = 0 (§6.4) are evaluated on conditions where no CTRL_rematch run exists. Their "arm − CTRL within a condition" contrast (F5) would have to use CTRL_base, which F4 forbids for regenerated-pickle arms (`slurm/standardize_qm9.sbatch:17-19`). | plan 1 §3.2–3.3, §5, §6.4 | validity, fixable with about 18 runs |
| P1-2 | **x_λ has a terminal-atom collapse.** The fit is all-atom with the identity atom map (`lgeom.kabsch_align` / `interp`, plan 1 §1). There is no relabelling of equivalent terminal atoms. A methyl or NH2 that DE left in another rotamer basin (DE maxiter 15, `standardize_confs.py:16,106-107`) then gives C–H ≈ 0.6 Å at λ = 0.5 (plan 2 §1.3 local check). This contaminates S1.4 and S4 training. My plan has the same flaw. | plan 1 §1, §2.2 | validity (S1.4, S4) |
| P1-3 | **Fit and pairing check use different atom sets.** The pairing check is heavy-atom (`AlignMol(RemoveHs…)`, matching `standardize_confs.py:108`), but the alignment actually applied is all-atom. Harmless for pairing, but the stored geometry is not the frame that was validated. | plan 1 §1 `pair_std_with_raw` vs §2.2 | minor |
| P1-4 | **CPU contention is not addressed.** Training stays at `--cpus-per-task=10` with 8 loader workers (`slurm/ablation_train_array.sbatch:7,70`). Next to it run a 16-core builder and the mmff standardization (`slurm/standardize_qm9.sbatch:6`). With about 24 free cores this repeats the round-1 loss of about 10 GPU-hours (`RESULTS_QM9.md:92,95`). Plan 1 §10 only says "check `nproc`". | plan 1 §11 | throughput |
| P1-5 | **The panel inference array waits for the whole training array.** It uses `--dependency=afterok:<train array>`, which waits for **all** tasks, so B6 panels cannot start at t = 34.5 h as the §11 table shows. They start at about 46 h. Any single failed training task blocks every panel, because `afterok` requires all tasks to succeed. | plan 1 §11 "Mechanics" | schedule |
| P1-6 | **Redundant non-cycle runs.** `GT_EVAL=1` on the B6/S3/S4 lines also runs the non-cycle `gtL` (`slurm/ablation_train_array.sbatch:82`). F3 asks for cycle consistently. This is 9 redundant runs. | plan 1 §4–6 TSV | waste (about 2 GPU-h) |
| P1-7 | **The interp λ = 1 endpoint differs from gtLcycle.** It is built from capped rematch pickles (≤ 30 conformers, rotor-less molecules dropped; `standardize_confs.py:59-67,76`), so it lacks some GT conformers. Plan 1 acknowledges this. It is acceptable only if the existing gtLcycle anchor is also shown on the same intersection. | plan 1 §3.1 | noted |

### Plan 2

| # | problem | evidence | severity |
|---|---|---|---|
| P2-1 | **The inference lane is saturated, so the makespan is understated.** Lane 3 (`%1`) has 196 runs: S6 16 + S5ctrl 3 + null 1 + S1 95 + S2 36 + S5 3 + S3 18 + S4 24. At 0.25 h each that is about 49 h of one GPU. The S4 extras can only start after T4 ends at 47 h. The real end is about 53 h, not "47–50 h". | plan 2 §10 table | schedule |
| P2-2 | **CPU over-subscription.** `build_round2_data.sbatch` uses 32 cores, the mmff standardization runs alongside, and 3 trainings take 10 CPUs with 8 workers each (`slurm/ablation_train_array.sbatch:7,70`, cited as is in plan 2 §6 S2). That is far above the about 24 free cores. The same round-1 failure mode as P1-4. | plan 2 §5.5, §10 | throughput |
| P2-3 | **`mmff_error` undercount.** Taking it from "the last cumulative `long_term_log` print" misses up to 19 molecules per worker. That dict is printed only every 20 molecules (`standardize_confs.py:156-162`). The exact count is the bare key printed by `log_error` (`:47-50`, called at `:87`), as plan 1 §7 does. My plan has the same bug. | plan 2 §6 S5 | reporting |
| P2-4 | **Noise seeds are reused.** The noise pickles have L entries in `clean_confs` order. Under cycle with 2L samples, each noisy L is used twice with the same realisation. Plans 1 and 3 store 2L independent draws. Not wrong (it mirrors gtLcycle's reuse), but it gives fewer distinct L per condition. Keep the common-random-numbers idea, but draw 2 replicates. | plan 2 §3.1 table | minor |
| P2-5 | **The seed-builder CPU estimate is from the wrong source.** "DE matching about 3–10 h (research_A R2A-1)" is the `--mode test` estimate, which runs about 9 DE runs per GT conformer (`slurm/analysis_qm9.sbatch:19-20`). The builder runs 1 DE per GT conformer, so about 0.1–0.6 CPU-h at the standardize rate (`slurm/standardize_qm9.sbatch:23`). It only matters for scheduling. | plan 2 §10 | minor |
| P2-6 | **S4 is gated late.** The gate is at t ≈ 30 h and S4 is the last wave. That is safe, but S4 results arrive last (about 53 h) and the gate input V18 (λ = 1.00 vs gtLcycle) is ready much earlier if S1 inference runs first. | plan 2 §8, §10 | schedule |

Plan 2's checks are mostly correct and valuable:
- heavy fit == stored `rmsd` (`standardize_confs.py:108,117`);
- `assert_same_graph` (`utils/dataset.py:224,234,239` assume one graph);
- `--de_seed` decoupling (`tools/local_structure_analysis.py:180,248` share `args.seed`);
- `provenance.txt` against the `slurm/common.sh:102` skip-if-exists;
- V18 pipeline check;
- `pair_ok` / stereo guards.

### Plan 3 (mine, for the record)

| # | problem |
|---|---|
| P3-1 | Same as P1-2 and P1-3: all-atom fit, no terminal relabelling. |
| P3-2 | Same as P2-3: `mmff_error` taken from the last dict. |
| P3-3 | No stereo / clash (`pair_ok`) guard on interpolated pairs. |
| P3-4 | Featurize-time pairing (inside `utils/dataset.py`) is harder to inspect than plan 2's offline `standardized_pickles_paired` and its population report. |

---

## 2. Votes on contested decisions

| D | vote | one-line reason |
|---|---|---|
| D1 alignment | **Heavy-atom Kabsch fit applied to all atoms, plus plan 2's terminal-atom relabelling, plus plan 2's `pair_ok` guard** | The heavy fit is exactly the frame the stored `rmsd` validates (`standardize_confs.py:108`). Relabelling is a graph automorphism that removes the 0.62 Å C–H collapse at no validity cost. |
| D2 acyclic set includes H | **Yes (all-atom)** | All three plans agree. Only then are A5-ring and A5-acyc exact complements (ring geometry from one source, everything else from the other). |
| D3 S1 control-model count | **Plan 2's matrix (95, ≈ 108 spirit): CB + B1 on all 11 conditions; CR on the 9 conditions that S2–S4 reuse; + 2 V18 runs.** If trimming is needed, cut CB s1/s2 on interp/A5, never CR. | F4 requires CR wherever S3/S4/S5 are compared (P1-1). Plan 1's 72 leaves those panels without a control. |
| D4 inference packing | **Pack 3 per GPU, conditions batched per model (plan 3), plus plan 2's provenance guard and its smoke timing test (GPU vs CPU-only generation)** | Generation and evaluation are CPU-bound (one molecule of about 15 conformers per batch, CPU torsion updates `utils/torsion.py:78-103`). Unpacked inference makes a ~49 h serial lane (P2-1). |
| D5 CPUs and workers | **4 CPUs and 3 loader workers per training, confirmed by the `data_wait` timer (gate G-loader). No 2-per-GPU sharing unless G-coloc shows ≥ 1.6× aggregate and free cores.** | Round-1 lost about 10 GPU-h to CPU contention (`RESULTS_QM9.md:95`). 10 CPUs × 3–4 trainings + builders exceeds the about 24 free cores (P1-4, P2-2). |
| D6 B1cap control | **Yes, 1 seed** | Zero extra code (`mix_gt_p = 1.0` on the paired cache). It is the only clean read of S3/S4's GT end against the 30-conformer cap (`utils/dataset.py:209` vs `standardize_confs.py:59-67`). |
| D7 S4 this round | **Yes.** Gate: pairing ≥ 99.9 %, `pair_ok` ≥ 98 %, V18 (λ = 1.00 reproduces gtLcycle within 0.003 Å), S4 smoke passes, and the generate guard works. Place it in **wave 2** if these pass by then (V18 is available about 4 h in, if S1 inference runs first); otherwise wave 3, with S5 swapped forward. | The marginal code is small and fully tested (strict-load regression). The shared machinery is required by S1.4 and S3 anyway. |
| D8 budget | **Target about 160–170 GPU-h.** Trim in this order until under target: | Keeps every F4 control and every pre-registered arm. |

D8 trim order:
1. Drop the non-cycle in-job `gtL` runs (F3; P1-6).
2. Drop CB s1/s2 on interp/A5.
3. Drop S4 mmff λ ∈ {0.25, 0.5}.
4. Drop S6 B1 lines.
5. Drop S1 noise 0.01.

Budget composition: 13 trainings (12 plus B1cap) at about 10 GPU-h of training each ≈ 130, plus packed inference about 25–30, plus smoke 1.5.

### D9: base plan and grafts

**Base: plan 2** (validity layer: `lgeom` with heavy fit, terminal relabel, `pair_ok`, stereo; offline paired pickles with a population report; V1–V21; V18 pipeline check; provenance guard; data-validity scan; `--de_seed`).

Graft from **plan 1**:
- The golden and regression tests: M1, M2 (equal outputs **and** equal post-call RNG state), M8, M9, and C5–C7 byte-identity of `make_seed_pickles`, `paired_compare` and `local_structure_analysis` defaults.
- The exact `mmff_error` count via `grep -x` (`standardize_confs.py:47-50`).
- 2L independent noise draws (combined with plan 2's common random numbers across σ).
- The `--l_lambda_end_mass` flag, so the U[0,1]-vs-endpoints question can be changed without new code.

Graft from **plan 3**:
- The throughput layer: per-model packed eval sets (`run_evalset`, PACK = 3); 4 CPUs and 3 workers per training with the `data_wait` timer; the mmff standardization on few cores off the critical path.
- The fail-fast gates: `--min_its` after 200 iterations, NaN abort, and the epoch-0 "learning" check.
- True resume from `last_model.pt` with atomic save and RNG state.
- The M0 measurements (`sacct`, it/s, MaxRSS, cache sizes, free cores).
- The head-phase schedule: all inference on existing models (S1, S6, CR gtLcycle) runs in the first about 3.5 h on 4 GPUs, then 3–4 training waves of 4, with in-job packed evals. That brings the makespan from about 49–53 h to about 37–41 h.

---

## 3. Final ranking

1. **Plan 2.** It has the fewest validity defects. It found the terminal-atom collapse, the heavy-fit/`rmsd` identity and the V18 end-to-end check. Its defects (P2-1 to P2-3) are schedule and reporting problems, fixed by grafts.
2. **Plan 1.** It has the safest diff discipline (golden tests, byte-identical defaults). But it has a real control gap (P1-1), the interpolation flaw (P1-2) and unchanged CPU contention (P1-4).
3. **Plan 3 (mine).** It has the best throughput design. But it shares the interpolation flaw (P3-1, which hits two arms: S1.4 and S4 training), lacks `pair_ok`/stereo guards, and has the `mmff_error` undercount.
