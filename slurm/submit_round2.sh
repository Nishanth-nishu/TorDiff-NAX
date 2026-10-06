#!/bin/bash
# Round-2 submission (round2/IMPLEMENTATION.md §8). Run on the login node from $PROJECT/slurm:
#   bash submit_round2.sh prep          # CPU: S5 MMFF standardization -> featurize mmff (already started on 2026-10-06)
#   bash submit_round2.sh head          # GPU: packed evaluation of the existing checkpoints (S1, S1A5, S1CR, V18, S6, S0)
#   bash submit_round2.sh gate          # S4 gate (DECISION D7) from the head outputs + smoke log; prints PASS / FAIL
#   bash submit_round2.sh train [pass|fail]   # training waves (+ the packed post-training evaluation, afterok)
# GPU budget: every array is throttled so that at most 4 GPUs are in use (gnode118 has 4).
set -euo pipefail
PROJECT=/scratch/nishanth.r/tordiff
S=$PROJECT/slurm
LOGS=$PROJECT/logs
# DECISION D5 outcome (smoke/d5 measurement, IMPLEMENTATION.md §5): CPUs and loader workers per training job
TRAIN_CPUS=${TRAIN_CPUS:-4}
LOADER_WORKERS=${LOADER_WORKERS:-3}
PACK=${PACK:-3}            # DECISION D4 outcome: packed evaluation, 3 gen_eval processes per GPU
nlines() { grep -cvE '^\s*(#|$)' "$1"; }

case "${1:-}" in
  prep)
    j=$(VARIANT=mmff sbatch --parsable --cpus-per-task=4 --mem=24G --time=2-00:00:00 "$S/standardize_qm9.sbatch")
    VARIANT=mmff sbatch --dependency=afterok:$j "$S/featurize_qm9.sbatch"
    ;;
  head)
    n=$(nlines "$S/r2_eval_models_head.tsv")
    MODELS=$S/r2_eval_models_head.tsv PACK=$PACK sbatch --export=ALL --array=0-$((n - 1))%4 "$S/r2_eval_array.sbatch"
    ;;
  gate)
    W=$PROJECT/workdir; R=$PROJECT/results; T=steps20_seed0_gtLcycle
    SMOKE_LOG=${SMOKE_LOG:-$(ls -t $LOGS/tordiff_r2_smoke_*.log | xargs grep -l 'qm9_SMK_S4' | head -1)}
    srun -n 1 -c 1 --mem=8G -t 00:20:00 -A plafnet2 -p plafnet2 -w gnode118 bash -c "
      source $PROJECT/slurm/common.sh; activate_env
      python $PROJECT/tools/s4_gate.py --paired_summary \$QM9_PAIRED/SUMMARY.txt --smoke_log $SMOKE_LOG \
        --v18 CR=$R/qm9_CTRL_rematch_100ep_e100_s0/S1_lam1.00_cyc_ORACLE/eval.pkl,$R/qm9_CTRL_rematch_100ep_e100_s0/$T/eval.pkl \
        --sd_files CR=$R/qm9_CTRL_rematch_100ep_e100_s0/$T/eval.pkl,$R/qm9_CTRL_rematch_100ep_e100_s1/$T/eval.pkl,$R/qm9_CTRL_rematch_100ep_e100_s2/$T/eval.pkl \
        --v18 B1=$R/qm9_B1_train_gtL_e100_s0/S1_lam1.00_cyc_ORACLE/eval.pkl,$R/qm9_B1_train_gtL_e100_s0/$T/eval.pkl \
        --sd_files B1=$R/qm9_B1_train_gtL_e100_s0/$T/eval.pkl,$R/qm9_B1_train_gtL_e100_s1/$T/eval.pkl,$R/qm9_B1_train_gtL_e100_s2/$T/eval.pkl"
    ;;
  train)
    case "${2:-}" in
      pass) TABLE=$S/ablations_train_round2.tsv ;;
      fail) TABLE=$S/ablations_train_round2_noS4.tsv ;;
      *) echo "usage: $0 train pass|fail   (the S4 gate decides; DECISION D7)"; exit 1 ;;
    esac
    n=$(nlines "$TABLE")
    jt=$(TABLE=$TABLE LOADER_WORKERS=$LOADER_WORKERS sbatch --parsable --export=ALL --cpus-per-task=$TRAIN_CPUS \
         --array=0-$((n - 1))%4 "$S/ablation_train_array.sbatch")
    echo "training array $jt ($n runs, $TRAIN_CPUS CPUs / $LOADER_WORKERS loader workers each)"
    # post-training packed evaluation: models of waves 1-3 start when all of their training tasks have ended (they run
    # next to the single wave-4 training, so %3 keeps <= 4 GPUs busy); the wave-4 model (last line) gets its own task.
    POST=$S/r2_eval_models_post.tsv
    if [[ "$2" == "fail" ]]; then grep -v 'S4_lamcond' "$S/r2_eval_models_post.tsv" > "$S/r2_eval_models_post_noS4.tsv"; POST=$S/r2_eval_models_post_noS4.tsv; fi
    grep -vE '^\s*(#|$)' "$POST" | grep -v 'B6_jit0.02pa' > "$S/r2_eval_models_post_main.tsv"
    grep 'B6_jit0.02pa' "$POST" > "$S/r2_eval_models_post_last.tsv"
    dep="afterany"; for i in $(seq 0 $((n - 2))); do dep="$dep:${jt}_$i"; done
    m=$(nlines "$S/r2_eval_models_post_main.tsv")
    MODELS=$S/r2_eval_models_post_main.tsv PACK=$PACK sbatch --export=ALL --dependency=$dep --array=0-$((m - 1))%3 "$S/r2_eval_array.sbatch"
    MODELS=$S/r2_eval_models_post_last.tsv PACK=$PACK sbatch --export=ALL --dependency=afterany:${jt}_$((n - 1)) --array=0 "$S/r2_eval_array.sbatch"
    ;;
  *) echo "usage: $0 prep|head|gate|train pass|fail"; exit 1 ;;
esac
