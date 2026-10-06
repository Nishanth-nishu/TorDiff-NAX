# Review 3: cluster / SLURM side of the round-2 implementation

Reviewed 2026-10-07 by coding agent 3.

**Scope**
- Commits `503191f..47ceacf`, excluding `reports/`.
- Cluster state was read read-only through `ssh ada` (`squeue`, `sacct`, `scontrol`) and one short `srun` that only read
  files on `/scratch`.
- No code was edited. No job was submitted or cancelled.
- Line numbers refer to HEAD `47ceacf`.

---

## 1. Findings

### BLOCKER

**B1. The 4-CPU decision (D5) is silently overridden. Each training gets about 10 CPUs.**
- `slurm/submit_round2.sh:43` passes `--cpus-per-task=4`, but `slurm/ablation_train_array.sbatch:8` keeps
  `--mem=48G`.
- The partition enforces `MaxMemPerCPU=5000` (`scontrol show partition plafnet2`). SLURM therefore raises the CPU
  count until the memory fits: 48 GiB / 5000 MB ≈ 10 CPUs.
- This is already observed on our own jobs:

| job | requested | allocated |
|---|---|---|
| 10105 (std-mmff) | 4 CPUs, 24G | 6 CPUs |
| 10180 (smoke) | 12 CPUs, 64G | 14 CPUs |
| 10122 (data) | 12 CPUs, 60G | 14 CPUs |

- **Failure scenario.**
  - 4 trainings need about 40 CPUs, plus 9 for each `r2_eval_array` task (8 CPUs, 40G).
  - The node has 48 cores and is shared. At 22:00 it was at `CPUAlloc=48/48` with only 2 of 4 GPUs allocated, which is
    exactly the round-1 failure mode.
  - The 4th training then pends for `Resources`, and a GPU idles for a whole wave (~15 GPU-h at the measured
    6.4 it/s).
- **Fix.** Set `--mem` explicitly together with `TRAIN_CPUS`.
  - Round-1 MaxRSS was 15.0–24.4 GB with 8 workers (`sacct -j 1036`; the larger figure is the raw/B1 lines).
  - With 3 workers, `--mem=24G` is the smallest safe value I would try. It gives about 5–6 CPUs per job, as 10105
    shows.
  - Check MaxRSS on the first wave-1 task.
  - Apply the same check to `r2_eval_array.sbatch:7-8` (8 CPUs/40G → 9 CPUs).

### SHOULD-FIX

| # | where | problem | failure scenario | fix |
|---|---|---|---|---|
| S1 | `slurm/common.sh:150` (`run_evalset`) + `:119-134` (`gen_eval`) | `gen_eval` runs on the left of `\|\|`, which **disables `set -e`** inside it. Its last command (`breakdown … \|\| echo`, `:133-134`) always returns 0. The only failure `run_evalset` can record is the provenance `return 1`. | A generate or evaluate crash (bad `--l_level`, missing pickle, OOM, a truncated `confs.pkl`) leaves no `eval.pkl`. The array task still ends COMPLETED, and `SMOKE_EVAL_FAILED` (`smoke_round2.sbatch:119,121`) can never fire. So the S4 gate's smoke criterion (`tools/s4_gate.py:64-68`) only checks the in-job `steps20_seed0`. Failures surface days later in the analysis. | End `gen_eval` with `[[ -s $out/eval.pkl ]] \|\| { echo "gen_eval FAILED $out"; return 1; }`. |
| S2 | `slurm/submit_round2.sh:17`, `common.sh:146`, `r2_eval_array.sbatch:43` | The default is still `PACK=1`, although the user ruled `PACK=3` (DECISION "User rulings"). | Running `bash submit_round2.sh head` as documented (`:6`) runs unpacked. The head phase takes about 1.5× longer, and the S4 gate and training start later. | Set the default to `PACK=${PACK:-3}` in `submit_round2.sh`, and write `pack=` into `provenance.txt`. |
| S3 | `slurm/submit_round2.sh:52-54` | The panel array depends `afterany` on **all** training tasks 0..n−2. **noS4 table:** wave 3 is only B1cap + B6_jit0.02 (`ablations_train_round2_noS4.tsv:41-42`). The panels wait for B1cap (task 8), so 2 GPUs idle for that whole run. **pass table:** wave 4 is one 15 h run (B6_jit0.02); the panels (~49 runs ≈ 3 h packed) finish early, and then 3 GPUs idle for ~12 h. | ~30–35 GPU-h idle at the measured 14–16 h per run (IMPLEMENTATION §5). | Submit one panel job per model with `afterok:${jt}_i` (a loop of `sbatch --array=k`), or move B6_jit0.02 earlier and B1cap into wave 3 so that every wave has 4. |
| S4 | `submit_round2.sh:36-44` vs `featurize 10106` | No dependency ties the B3 lines to `cache_mmff`. In the **fail** branch, B3 is wave 2 (`noS4.tsv:36-38`). | std-mmff (10105) has done 38/134 files in 4 h 39 min, so it will finish at about 14:00 on 7 Oct, and 10106 (featurize) after that. If training starts now, wave 2 starts at about 17:00 on 7 Oct, which is tight. If the cache is late, `ablation_train_array.sbatch:58-59` exits immediately, the array slot moves on, and those B3 tasks must be resubmitted by hand. | Submit the B3 lines as a separate array with `--dependency=afterok:10106`, or keep B3 in wave 3 in both tables. |
| S5 | `tools/s4_gate.py:57-59` | The V18 tolerance is the **training-seed SD** of gtLcycle MAT-R (round-1 B1: 0.0215 / 0.0215 / 0.0206 → SD ≈ 0.0005). The quantity being tested is a single λ = 1.00 run against a single gtLcycle run. Their difference is dominated by run-to-run sampling noise, which round 1 shows is not bit-reproducible: an unpacked R0 rerun moved COV-R 89.0293 → 89.0286. The two runs also differ by a rigid motion plus terminal relabelling, so the chaotic SDE trajectories diverge. | A spurious FAIL is plausible for B1 (tol ≈ 5e-4), and it would push S4 to round 3 for the wrong reason. | Use a paired molecule bootstrap CI of the λ1 − gtLc difference (must contain 0, or be within ±0.003 Å as plan 2 §3.3 pre-declared), or `max(seed SD, 0.003)`. |
| S6 | `tools/make_l_seed_pickles.py:22,182-183`; cluster `round2_seeds/*.population.txt` | Any unsafe pair drops the **whole molecule from the whole family**. The λ0.00–λ1.00 and both A5 pickles hold **639 of 997** test molecules: 194 bond_dev, 104 inversion, 6 stereo, 53 embed failures. `etkdg2L` holds 936. | The S1 λ curve, the V18 gate and the S1.5 ring-vs-acyclic result rest on a 64 % subset biased toward rigid / simple molecules. λ = 0, λ = 1 and A5 do not interpolate at all, so the λ = 0.5 safety check does not apply to them. | Also build λ0/λ1/A5 on the full matched population (about 936) and report both. Keep the 639 set only for λ ∈ {0.25, 0.5, 0.75} comparisons. State the subset in every table. |
| S7 | missing analysis step | Ruling 1 requires every S4 comparison to be re-scored on S4's molecule subset. `tools/paired_compare.py` supports `--universe intersection` (commit 0b514d6), but no sbatch or analysis block calls it for round 2. `slurm/analysis_qm9.sbatch` is unchanged. | The ruling is not executable from the scripts. Done by hand, it is error-prone, and the 639-molecule test subset (S6) and the training-side pair_ok drop are easily confused. | Add an `analysis_r2.sbatch` with the union and intersection `tcompare` calls per condition. Define "S4's molecule set" explicitly (test molecules present in every compared `eval.pkl`). |
| S8 | `slurm/common.sh:109-110` | The provenance covers the model checkpoint and the argument files, but **not the code version or PACK**. | A sampler or evaluator fix later in round 2 would silently reuse outputs that the old code produced. | Append `git -C $REPO rev-parse HEAD` and `PACK=$PACK` to `new`. |

### NIT

| # | where | note |
|---|---|---|
| N1 | `common.sh:120` | `-s confs.pkl` trusts a file truncated by a kill during `pickle.dump` (`generate_confs.py:203-205`). With S1 fixed, this becomes a loud failure; it would be better to write to `confs.pkl.tmp` and `mv`. |
| N2 | `ablation_train_array.sbatch:61-65` | There is no resume: an incomplete run restarts from scratch. This is consistent with DECISION D9 ("resume only when requested"), but nothing implements the "requested" path. At 14–16 h per run, one node hiccup costs one run. The `--time 1-12:00:00` limit is still fine. |
| N3 | `submit_round2.sh:31` | `ls -t … \| xargs grep -l … \| head -1` under `pipefail` aborts the script with no message if no smoke log matches. It also picks a log that may still be being written. |
| N4 | DECISION "User rulings" vs cluster `standardized_pickles_paired/SUMMARY.txt` | The ruling says pair_ok is 92.1 % (~8 % of conformers dropped). The summary says `PAIR_OK 776141/884632 = 87.736 %`, i.e. 12.3 % dropped (bond_dev 66 580, inversion 40 784, stereo 1 069, clash 58). Reconcile the numbers before they go into the report. |
| N5 | `smoke_round2.sbatch` d5 | The smoke TIMING lines (data_wait 18–22 %) come from 1-iteration epochs and are meaningless. The D5 decision correctly uses the 400-iteration measurement: 4.59 % vs 4.79 % (IMPLEMENTATION §5). |
| N6 | LF line endings | All committed `slurm/*` files are LF (`.gitattributes` `eol=lf`), and the cluster gets files via `git archive HEAD` (`round2/sync_cluster.sh:11`), so line endings are safe. `git ls-files --eol` shows `w/crlf` for `submit_round2.sh` and `r2_evalsets.tsv`, but both contain 0 CR bytes, so this is a stale index stat. Never `scp` working-tree files directly. |
| N7 | `featurize_qm9.sbatch` | It holds 9 CPUs (40G) for a serial (`--num_workers 1`) job. `--mem=16G` would free about 5 cores during wave 1. |

**What looked correct**
- The 7th TSV field and `GT_GEN_ARGS`.
- GT_EVAL=2 (cycle only).
- The S4 `--l_level` guard: smoke confirmed that the `confs.pkl` md5 differs between λ = 0, 0.25 and 0.5, so λ is live. The identical smoke SUMMARY values are a coincidence of a 1-iteration model.
- The smoke isolation (`data/QM9_smoke`, separate WORK and RES, refusal check).
- The `paired` variant wiring.
- `cache_paired` was built (1.32 GB train, the same size as `cache_raw`).

---

## 2. Status of the cluster jobs (02:12 IST, 7 Oct)

| job | what | state | notes |
|---|---|---|---|
| 10180 | smoke (MODE=smoke) | **COMPLETED** 22:43–22:59 (15 min 47 s), exit 0, MaxRSS 8.1 GB | All 4 smoke trainings reached `done:`. Every in-job and packed eval wrote a SUMMARY. No `SMOKE_*_FAILED` lines, but see S1: they could not have fired. Packed (PACK=3) S1 sets on the released model took 224 s for 20 molecules. |
| 10122 | data builder (paired + seeds) | **COMPLETED** 21:49–01:28 (3 h 39 min, 14 CPUs), MaxRSS 11.2 GB | PAIRING_CLEAN 99.999 % (884 632 / 884 640); PAIR_OK 87.7 %. Seeds: noise 996/997, etkdg2L 936/997, λ and A5 families 639/997 (S6). `etkdg_mmff_not_converged` = 1223 conformers. |
| 10105 | std-mmff | **RUNNING** 4 h 39 min (6 CPUs allocated for 4 requested; limit 2-00:00:00) | 38/134 worker files done, about 8 files/h, so it should finish at about 14:00 IST on 7 Oct. `grep -x mmff_error` = 0 so far. |
| 10106 | featurize mmff | **PENDING** (Dependency `afterok:10105`) | Limit 6 h. If 10105 fails, 10106 is never released: check it. |

At 02:12 the node had 26 of 48 CPUs and 2 of 4 GPUs allocated. Our jobs held only std-mmff's 6 CPUs.

---

## 3. PACK=1 vs PACK=3

**My view: use PACK=3.**
- The D4 "exact SUMMARY" criterion could not be met by anything. An **unpacked** rerun of round-1 R0 already differs
  (COV-R 89.0293 → 89.0286; IMPLEMENTATION §5), because GPU scatter reductions are non-deterministic and the SDE
  amplifies the differences. Packed deviations are the same size.
- Packing cannot change any RNG stream: each process seeds itself (`generate_confs.py --seed`). It only changes
  scheduling.
- The noise floor is about 1e-3 COV-R points. That is far below every contrast the plan cares about:
  - the training-seed SD of AMR-R is about 0.0013 Å;
  - the R0 sampling-seed SD of COV-R@0.5 is about 0.43 points;
  - the effects of interest are 0.004–0.05 Å.
- All round-2 contrasts are molecule-paired bootstraps between runs made under the same regime.

**Conditions**
1. Record `PACK` and the code commit in `provenance.txt` (S8).
2. Do not mix a packed control with an unpacked arm inside one contrast where avoidable. It is harmless at this noise
   level, but it keeps the audit trail clean.
3. Make PACK=3 the script default (S2).

The measured gain (1.47×) and the smoke timing (224 s for the full packed S1 set on 20 molecules) both support it.
