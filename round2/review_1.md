# Review 1 (coding agent 1): adversarial review of the round-2 implementation

Reviewed 2026-10-07: `git log 503191f..HEAD` (14241b7 … 5a119ac; the reports/ commit is ignored). I checked it
against DECISION.md D1-D9, its 6 listed defects, the two user rulings at its end (S4 trained on safe pairs with
subset re-scoring; PACK=3), and my vote_1.md. No code was edited, nothing was pushed and no jobs were submitted.

Tests I ran myself:
- `python -m pytest -q -p no:cacheprovider tools/tests/test_lgeom.py tools/tests/test_seed_builder.py` → **13 passed**
  (local: rdkit 2026.03.6, torch 2.14 CPU).
- `tools/tests/test_round2_model.py` cannot be collected locally (`No module named 'utils'`; it needs PyG/e3nn and
  the cluster env). That is expected. I reviewed its contents instead and accept the reported 27-pass cluster run as
  stated, but I did not reproduce it.
- I ran two bash experiments with `set -e` semantics (scratchpad `sete.sh`, `sete2.sh`). Results are under F2 and
  "Retracted" below.

## Findings (ranked)

| # | Severity | Where | Finding | Concrete failure scenario |
|---|---|---|---|---|
| F1 | **BLOCKER** (until quantified) | `tools/make_l_seed_pickles.py:182-184` (and `:174`, `:179`) | One failing pair (`pair_check`, DE error, graph mismatch) drops the **whole molecule** from every λ pickle **and** both A5 pickles. On training data 7.9% of *conformers* fail `pair_ok` (IMPLEMENTATION §4). A molecule survives only if *all* its L pairs pass, so the surviving test population can be far smaller than "~4% of molecules", and it is biased. The failures are poorly matched pairs (heavy RMSD 0.4-1.0 Å), so the survivors skew towards rigid, few-conformer molecules. The A5 ring/acyclic oracles never interpolate, so they do not need `pair_ok` at all, yet they inherit the cut. | The S1 dose-response (ε*) and the S1.5 ring-vs-acyclic answer are read on, say, 60-70% of the test set with the flexible molecules removed. Those are exactly the molecules where L↔τ matters, so ε* is biased optimistic. The user ruling ("S4 on safe pairs, controls re-scored on the subset") covers S4 *training*. It does not cover this **test-set** S1 population, which affects every model in S1. **Before the head phase:** read `$QM9_SEEDS2/L_lam0.50_cyc_ORACLE.population.txt` (job 10122). If more than about 10% of molecules are dropped: (a) build A5 from every verified pair, ignoring `pair_ok`; (b) report the λ series on its own population next to the λ = 1 builder anchor (already the design); (c) state the population in every S1 table. |
| F2 | **SHOULD-FIX** | `slurm/common.sh:150` (`run_evalset`), and `:133` (the last command of `gen_eval` is `breakdown … \|\| echo`) | `gen_eval` runs inside `( … \|\| echo "$TAG" >> fail )`. bash disables `set -e` for a command on the left of `\|\|`, so a failing `generate_confs.py` or `evaluate_confs.py` does not stop `gen_eval`. `gen_eval` then returns 0, because its last command always succeeds. **Verified:** a mock with the same structure printed "generate failed but continued … run_evalset sees success", exit 0. | Every packed evaluation (head and post phases, PACK=3 per the ruling) reports success even when a condition crashes (OOM, missing pickle key, CUDA error). "FAILED tags" never fires. The gap is found only when analysis meets a missing `eval.pkl`, possibly a day later. Fix: end `gen_eval` with `[[ -s $out/eval.pkl ]] \|\| return 1`, or check after `wait` that every tag has `eval.pkl`. |
| F3 | **SHOULD-FIX** | `slurm/r2_eval_models_head.tsv:6,8,10` (S1A5 only on B1); `slurm/r2_evalsets.tsv` comment above `:30` ("CR only if packing passed") | The A5 ring/acyclic oracles run on B1 ×3 only. There is no CTRL (CR or CB) on A5, so the F5 rule "compare by B1 − CTRL per condition" cannot be applied to S1.5. The stated condition for adding CR (packing passes) is now met by the PACK=3 ruling. | S1.5 conclusions would have to rest on absolute B1 curves, which F5 forbids. Add `S1A5` to the three CR lines: 6 runs, about 1 GPU-h packed. |
| F4 | **SHOULD-FIX** | `slurm/submit_round2.sh:17` (`PACK=${PACK:-1}`, comment "packing NOT enabled"); `slurm/r2_eval_array.sbatch` header; `round2/IMPLEMENTATION.md` §5/§6 | The defaults and comments contradict the PACK=3 ruling. | An operator runs `bash submit_round2.sh head` as documented in IMPLEMENTATION §7. It runs unpacked: about 1.5× longer wall time, and the budget table is wrong. Change the default to 3 or document `PACK=3 bash submit_round2.sh …`. Also update the D4 row in IMPLEMENTATION. |
| F5 | **SHOULD-FIX** | `slurm/common.sh:17-18` (`export WORK=${WORK:-…}`, `RES=${RES:-…}`) | This is a default-path change. Many HPC sites define `$WORK` in the login environment, and `sbatch` exports the environment by default (`--export=ALL`, also used explicitly in `submit_round2.sh:27,43,54`). Whether ada defines `WORK` is UNVERIFIED. | If ada sets `WORK=/home/...`, every round-1 and round-2 job writes checkpoints to `$HOME`, which has a 30 GB quota (`common.sh:4`). `train_complete` then finds nothing and **retrains**, and the inference array cannot find round-1 models. Fix: use a dedicated override name (`R2_WORK`/`R2_RES`, used only by the smoke job), or check `echo $WORK` on ada before submitting. |
| F6 | **SHOULD-FIX** | `slurm/submit_round2.sh:43` (`--cpus-per-task=$TRAIN_CPUS`, 4) together with `slurm/ablation_train_array.sbatch:8` (`--mem=48G`) | RESULTS_QM9.md:92 records a 5 GB/core memory cap on gnode118. If that is `MaxMemPerCPU`, SLURM raises the CPU count to ceil(48/5) = 10 (or rejects the job), so the D5 reduction does nothing. D5 itself was measured with `taskset` inside one job (`smoke_round2.sbatch` d5 branch), so it never exercised the allocator. | The node still runs 4 × 10 training CPUs plus 8-CPU eval jobs plus 4-CPU standardization, and the round-1 CPU contention (about 10 GPU-h lost) repeats. Check `scontrol show config \| grep MaxMemPer` and the MaxRSS of round-1 jobs. Pass `--mem=20G` if MaxRSS allows; the paired cache roughly doubles positions, so measure it on S3. |
| F7 | **SHOULD-FIX** | no round-2 block in `slurm/analysis_qm9.sbatch`; no driver anywhere | Nothing wires the read-out the SHORTLIST pre-declares: per-condition B1 − CTRL `tcompare` (union primary plus the intersection COV-R@0.1 family), the S4-subset re-scoring the user ruling requires, F5 floors (`local_structure_analysis.py --mode run` per condition), ε* interpolation against `l_error.csv`, the leak test on the `A1c_gtL_cycle_random_tors` null, and the `mmff_error` grep. | Not blocking for submission, because GPU jobs do not depend on it. But the "controls re-scored on S4's own molecule set" ruling currently has no implementation. Write it before results arrive so that analysis choices stay pre-registered. |
| F8 | NIT | `tools/make_l_seed_pickles.py:225-238` | All sources run inside one `try`. An exception in `build_etkdg` (an unexpected RDKit error outside the MMFF guard) skips `noise` and `interp` for that molecule. | The populations of the families become coupled for a reason unrelated to the factor under test. It is counted only as `exception`. Use a `try` per source. |
| F9 | NIT | `tools/make_l_seed_pickles.py:264` | `Pool(...)` is never closed or joined. | Harmless in a batch job, but workers can linger if the main process raises. |
| F10 | NIT | `tools/local_structure_analysis.py` `run_leak` | Boltzmann weights are looked up at `raw_dir/<csv smiles>.pickle`. That the raw stem equals the csv `smiles` is UNVERIFIED. On a miss it falls back silently to `w=None`, which makes `boltz_null` NaN. | The leak test reports only the permutation null. Print the number of molecules that got weights. |
| F11 | NIT | `torsional-diffusion/utils/dataset.py:145-163` (S4 `pair_ok` filter) | The filter also applies to the **val** split. That is consistent, but S4's `best_model.pt` is then selected on a different val population from CTRL_rematch's. | State it in the S4 results next to the training-population note. |

### Retracted (checked, not a bug)

I first suspected that `[[ "$GT_EVAL" == "1" ]] && SEED=0 … gen_eval …` (`ablation_train_array.sbatch`, GT branch)
disables `set -e` for the round-1 `GT_EVAL=1` lines. **It does not.** `gen_eval` is the last command of the `&&`
list, so `set -e` still applies. My mock exited at the first failure, exit 1. The round-1 error semantics are
unchanged.

## Checked and correct (default path and spec)

- **Training transform default branch** (`utils/dataset.py` `__call__`): the original statements run unchanged.
  Jitter is skipped at 0. No extra RNG draw; golden M2 compares the post-call RNG states exactly.
- **Score model** with `lambda_embed_dim=0`: identical shapes (C1, golden M1). `get_model` uses `getattr`, so old
  yamls give 0, and the released checkpoint loads strictly (C1b).
- **`train_epoch` defaults:** same divisor (`len(loader)`), and the NaN check runs only with `--fail_on_nan`. The
  timing adds `time.time()` calls only.
- **`generate_confs`:** `l_level` is protected from the yaml overwrite. The guard is two-sided. `sampling.sample`
  sets λ once per batch and asserts that it is constant (no GT leak, V14).
- **`paired_compare.py`:** `--universe union` is the default and `--report_failures` is opt-in, so round-1 tables are
  reproduced.
- **`local_structure_analysis.py`:** `de_restarts=1, de_seed=None` gives exactly the round-1 DE call.
- **Train-array 7th field and `GT_EVAL=2`:** 6-field lines parse identically. `MODEL` `eval` expansion is a no-op
  for PRE/PRE1/BASE.
- **The golden finding is credible and handled well.** Round-1 forward passes are not bit-reproducible across
  processes. Everything upstream of the forward pass is compared exactly; the forward output and coordinates are
  compared against a snapshot-vs-snapshot noise floor. This satisfies defect 6 as far as is physically possible.
- **D1 (heavy fit + relabel), defects 1-4: all done.**
  - Defect 1: ties are disambiguated by the rmsd recomputation (`build_paired_pickles.py:86-106`). Pairing
    verification is 100% on the probe.
  - Defect 2: exact `grep -hx mmff_error`.
  - Defect 3: MMFF is wrapped, and a failing molecule leaves both twin pickles (`make_l_seed_pickles.py:95-102`,
    with the assertion at `:276-277`).
  - Defect 4: the DE return value is discarded as `standardize_confs.py:106` does (`make_l_seed_pickles.py:171`).
- **D2:** the all-atom acyclic set holds, and the kept ring part is exact (T8).
- **The vote_1 items are fixed:**
  - CR gtLcycle ×3 is added (S1CR);
  - a provenance file is written only from round 2 on;
  - the twin rule holds;
  - CR is never cut.

## The automorphism relabelling (agent-2 extension)

**Accepted.** The model sees the same graph. QM9 node features do not include the chirality tag: `chiral_tag` is
computed in `utils/featurization.py:59-63` but never concatenated into `x` (`:96-99`). Mapping H counts per parent is
checked (`lgeom.py:123-125`).

One caveat for S3/B1cap. Their GT half is `Y_al`, which is GT relabelled by the automorphism with the lowest heavy-atom
RMSD to the *matched RDKit* conformer. That is still a valid GT conformer of the same labelled graph. It is fine.

## Opinion on the reported deviations

1. **Terminal atoms keep their interpolated bond length (`interp_x`).** Agree. Single-H rotors are not TD torsions
   (`get_torsion_angles` needs ≥ 2 atoms on each side), so DE never matches them. Without this fix, alcohols and
   amines would fail `pair_ok` by chemistry rather than by match quality. The endpoints are exact, and the torch and
   numpy versions are tested equal (C7b).
2. **Symmetric (non-terminal) groups relabelled.** Agree (see above). It turns a prochiral swap into a valid pair
   instead of a spurious inversion.
3. **`pair_ok` 92.1%; S4 drops failures at load.** Accepted per the user ruling for S4 training.
   - The **test-set** analogue (F1) is not covered by the ruling and must be quantified first.
   - Also report which molecule strata S4 lost (ring size, number of torsions, n_conf), because they are biased
     towards poorly matched, flexible molecules.
4. **PACK=1.** Superseded by the user ruling. Update the defaults and docs (F4).

## Opinion on the trimmed matrix

**Acceptable, with one restoration and one optional one.**

Accept these cuts:
- **S6.** Low priority and descriptive (verify_B R2-6); D8 named it first to trim.
- **noise 0.01.** The 0.02/0.04 points plus the λ series span the range.
- **S4 descriptive levels and the B6 0.02 panel.** Exploratory.
- **CTRL_base s1/s2 on S1.** CR ×3 is the F4 control for the dose-response, and CB ×3 still exists on RDKit L and
  gtLcycle, which are S2's pre-registered endpoints.

Restore or add:
- **(F3) CR on A5.** Required; otherwise S1.5 has no control.
- **With PACK=3 (about 1.47× throughput)** the freed time comfortably covers F3. If budget remains, restore noise 0.01
  on CR and B1 (the `S1x` set): 6 runs, about 1 GPU-h packed.

**Budget note.** The measured 6.4 it/s under load implies about 14-16 h per training run, not 10.7 h. The real
training cost is then about 185-210 GPU-h, above the D8 175 cap. This comes from slower hardware throughput, not from
more runs. Flag it to the user rather than trimming science further. The 36 h array time limit
(`ablation_train_array.sbatch:9`) still covers it.

## Verdict

Default-path behaviour is unchanged as far as can be tested; the bit-level floor is documented. Before the head phase:
- **Resolve F1**: quantify the λ/A5 test population, and decouple A5 from `pair_ok`.
- **Fix F2** (silent eval failures) and **F3** (no CTRL on A5).
- **Check F5 and F6** on ada; each is a one-line check.

F4 is a doc/default fix. F7 is needed before results are read.
