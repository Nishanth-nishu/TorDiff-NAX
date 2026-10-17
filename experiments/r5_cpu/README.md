# r5_cpu: Stage A milestone 1 CPU jobs (gnode118, no GPU)

One sbatch, one task per submission (`run.sbatch` header lists them):

```
bash round5/sync_cluster.sh                     # copy committed flexitors/ tools/ experiments/ to /scratch
scp experiments/r5_cpu/run.sbatch ada:r5_cpu.sbatch   # /scratch is not visible from the login node
ssh ada 'sbatch --export=ALL,TASK=<task> --job-name=r5_<task> ~/r5_cpu.sbatch'
```

| Task | Spec | Output (`$RES/round5/<task>/`) |
|---|---|---|
| `tests` | DECISION graft 1 (SHA pins, dead code), cluster-only golden tests | `flexitors_tests.txt`, `tools_tests.txt` |
| `rescore` | EXP-1 §3.6 / §10.5 (D-001, Q-P1-011) | `m_strata.csv`, `cmp.txt`, `budget_matched_summary.txt`, `budget_contrasts.csv`, `budget_per_model.csv`, `budget_mc_check.csv` |
| `floor`, `floor_mmff` | EXP-1 §9 item 1, §10.5 (D-003) | `local_structure_test_fixed.csv` (new file; the round-1 CSVs are not overwritten) |
| `valset` | EVAL §1.1, EXP-5 §12 | `val_smiles.csv`, `val_indices.csv`, `val_mols.pkl` (not committed), `population.txt`, `self_check.txt` |
| `validity_cal` | EVAL §3.1, review P2 B1 | `gt_calibration.csv`, `gt_calibration.txt` (GT conformers screened against the first GT conformer) |
| `gate0` | EXP-1 §10.3 | ETKDG / MMFF seeds of the validation set, their `eval.pkl` and breakdowns, `gate0_baselines.json` |

Everything here is ORACLE-free except `rescore` and `floor*`, which read test-set ground truth and are analysis only
(README rule 5). `gate0` scores ETKDG and MMFF seeds on the validation set (decision data, EXP-1 §10.3) and on the test
set (golden check against the published 0.152 Å / 49.6 % baselines; reported, never used to decide).
Small summaries are copied to `cluster_sync/round5/`.
