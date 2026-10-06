#!/bin/bash
# Round-2 submission (round2/IMPLEMENTATION.md §7, §Fixes). /scratch is visible only on gnode118, so run every phase from
# a shell ON gnode118, e.g.  ssh ada; srun -n 1 -c 1 --mem=4G -t 00:30:00 -A plafnet2 -p plafnet2 -w gnode118 --pty bash
# then  cd /scratch/nishanth.r/tordiff/slurm  and:
#   bash submit_round2.sh head                 # GPU: evaluation of the existing checkpoints (S1, A5, CR extras, V18, S0)
#   bash submit_round2.sh gate                 # S4 gate (DECISION D7 + user ruling) -> prints S4_GATE PASS / FAIL
#   bash submit_round2.sh train pass|fail      # training arrays + one post-training panel job per model
# DRY_RUN=1 bash submit_round2.sh ...  prints every sbatch command (resources, dependencies, env) without submitting.
# Resources (FIXES X2): plafnet2 enforces MaxMemPerCPU=5000 MB; every request below stays <= 5000 MB per CPU, otherwise
# SLURM silently raises the CPU count (round-1 training: --mem=48G -> 10 CPUs).
set -euo pipefail
PROJECT=/scratch/nishanth.r/tordiff
S=$PROJECT/slurm
LOGS=$PROJECT/logs
QM9=$PROJECT/data/QM9
# DECISION D5 + FIXES X2: 3 loader workers; 6 CPUs x 5000 MB = 30000 MB (round-1 peak RSS 15-24 GB with 8 workers; the
# paired cache holds a second position set, so 24 GB was not taken as the ceiling)
TRAIN_CPUS=${TRAIN_CPUS:-6}
TRAIN_MEM=${TRAIN_MEM:-30000M}
LOADER_WORKERS=${LOADER_WORKERS:-3}
PACK=${PACK:-3}            # user ruling 2026-10-06 / FIXES X4: packed evaluation, 3 gen_eval processes per GPU
EVAL_CPUS=8; EVAL_MEM=40000M
export R2_STRICT=1         # FIXES X3: in-job evaluations must produce eval.pkl + SUMMARY, else the array task fails
DRY=${DRY_RUN:-0}
nlines() { grep -cvE '^\s*(#|$)' "$1"; }
NDRY=0
sb() {  # sb [VAR=val ...] -- sbatch args... ; prints the job id
    local envs=()
    while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done; shift
    if [[ "$DRY" == "1" ]]; then
        NDRY=$((NDRY + 1)); echo "DRY sbatch ${envs[*]} $*" >&2; echo "DRYJOB$NDRY"
    else
        env "${envs[@]}" sbatch --parsable "$@"
    fi
}

case "${1:-}" in
  head)
    n=$(nlines "$S/r2_eval_models_head.tsv")
    j=$(sb MODELS=$S/r2_eval_models_head.tsv PACK=$PACK -- --export=ALL --cpus-per-task=$EVAL_CPUS --mem=$EVAL_MEM \
        --array=0-$((n - 1))%4 "$S/r2_eval_array.sbatch")
    echo "head evaluation array $j ($n model tasks, PACK=$PACK)"
    ;;
  gate)
    W=$PROJECT/workdir; R=$PROJECT/results; T=steps20_seed0_gtLcycle
    SMOKE_LOG=${SMOKE_LOG:-$LOGS/tordiff_r2_smoke_10180.log}
    [[ -s "$SMOKE_LOG" ]] || { echo "set SMOKE_LOG=<finished smoke log>"; exit 1; }
    # shellcheck disable=SC1091
    source "$PROJECT/slurm/common.sh"; activate_env
    python "$PROJECT/tools/s4_gate.py" --paired_summary "$QM9_PAIRED/SUMMARY.txt" --smoke_log "$SMOKE_LOG" --v18 CR=$R/qm9_CTRL_rematch_100ep_e100_s0/S1_lam1.00_cyc_ORACLE/eval.pkl,$R/qm9_CTRL_rematch_100ep_e100_s0/$T/eval.pkl --sd_files CR=$R/qm9_CTRL_rematch_100ep_e100_s0/$T/eval.pkl,$R/qm9_CTRL_rematch_100ep_e100_s1/$T/eval.pkl,$R/qm9_CTRL_rematch_100ep_e100_s2/$T/eval.pkl --v18 B1=$R/qm9_B1_train_gtL_e100_s0/S1_lam1.00_cyc_ORACLE/eval.pkl,$R/qm9_B1_train_gtL_e100_s0/$T/eval.pkl --sd_files B1=$R/qm9_B1_train_gtL_e100_s0/$T/eval.pkl,$R/qm9_B1_train_gtL_e100_s1/$T/eval.pkl,$R/qm9_B1_train_gtL_e100_s2/$T/eval.pkl
    ;;
  train)
    case "${2:-}" in
      pass) TABLE=$S/ablations_train_round2.tsv ;;
      fail) TABLE=$S/ablations_train_round2_noS4.tsv ;;
      *) echo "usage: $0 train pass|fail   (the S4 gate decides; DECISION D7)"; exit 1 ;;
    esac
    G=$S/r2_generated; mkdir -p "$G"
    # FIXES X7: the B3 (S5) lines need cache_mmff, built by the MMFF featurize job (10106 or its successor): they go in
    # their own array with an afterok dependency, so a late cache can never make them fail fast and be skipped.
    grep -vE '^\s*(#|$)' "$TABLE" | grep -v '^B3_match_mmff' > "$G/train_main.tsv"
    grep '^B3_match_mmff' "$TABLE" > "$G/train_b3.tsv"
    n=$(nlines "$G/train_main.tsv"); nb=$(nlines "$G/train_b3.tsv")
    TRES=(--export=ALL --cpus-per-task=$TRAIN_CPUS --mem=$TRAIN_MEM)
    jt=$(sb TABLE=$G/train_main.tsv LOADER_WORKERS=$LOADER_WORKERS R2_STRICT=1 -- "${TRES[@]}" --array=0-$((n - 1))%4 \
         "$S/ablation_train_array.sbatch")
    echo "main training array $jt ($n runs, $TRAIN_CPUS CPUs / $TRAIN_MEM / $LOADER_WORKERS loader workers)"
    if [[ -s $QM9/cache_mmff.train && -s $QM9/cache_mmff.val ]]; then BDEP=(); else
        MMFF_JOB=${MMFF_JOB:?cache_mmff missing: set MMFF_JOB=<featurize mmff job id, e.g. 10106>}
        BDEP=(--dependency=afterok:$MMFF_JOB); fi
    jb=$(sb TABLE=$G/train_b3.tsv LOADER_WORKERS=$LOADER_WORKERS R2_STRICT=1 -- "${TRES[@]}" "${BDEP[@]}" \
         --array=0-$((nb - 1))%3 "$S/ablation_train_array.sbatch")
    echo "B3 (S5) training array $jb ($nb runs) ${BDEP[*]}"
    # FIXES X8: one panel job per model, released by that model's own training task (afterok), so no GPU waits for a
    # whole wave. The run name follows ablation_train_array.sbatch: qm9_<NAME>_e100_s<SEED>.
    POST=$S/r2_eval_models_post.tsv
    while IFS='|' read -r MODEL SETS; do
        MODEL=$(echo "$MODEL" | xargs); SETS=$(echo "$SETS" | xargs)
        run=$(basename "$MODEL"); dep=""
        for pair in "main:$G/train_main.tsv:$jt" "b3:$G/train_b3.tsv:$jb"; do
            IFS=':' read -r _ tab jid <<< "$pair"; i=0
            while IFS='|' read -r NAME _ _ _ _ SEED _; do
                NAME=$(echo "$NAME" | xargs); SEED=$(echo "$SEED" | xargs)
                [[ "qm9_${NAME}_e100_s${SEED}" == "$run" ]] && dep="afterok:${jid}_$i"
                i=$((i + 1))
            done < "$tab"
        done
        [[ -n "$dep" ]] || { echo "skip panel for $run (not in this training table)"; continue; }
        f=$G/post_$run.tsv; echo "$MODEL | $SETS" > "$f"
        jp=$(sb MODELS=$f PACK=$PACK -- --export=ALL --cpus-per-task=$EVAL_CPUS --mem=$EVAL_MEM --dependency=$dep \
             --array=0 "$S/r2_eval_array.sbatch")
        echo "panel $run [$SETS] job $jp ($dep)"
    done < <(grep -vE '^\s*(#|$)' "$POST")
    ;;
  *) echo "usage: [DRY_RUN=1] $0 head|gate|train pass|fail"; exit 1 ;;
esac
