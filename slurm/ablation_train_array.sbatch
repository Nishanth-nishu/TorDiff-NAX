#!/bin/bash
#SBATCH --job-name=tordiff_qm9_trainabl
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%A_%a.log
#
# CORE training ablations on QM9 (CTRL x3, B1 x3, CTRL_rematch x3, B2 x3, B5 x3), one line of ablations_train.tsv per
# array index (15 lines -> --array=0-14; keep in sync). The 6th TSV field is the training seed.
# GPU throttle: gnode118 has 4 GPUs. Submit this array only AFTER the inference array has finished
#   (--dependency=afterany:<inference array jobid>) so that %4 here is the only GPU consumer -> <= 4 GPUs in use.
# --time: 100 epochs at the estimated 3-6 min/epoch = 5-10 h, + 1-3 gen_eval runs (~1-2 h each) = 7-16 h.
#   36 h leaves room for 2x slower epochs; well under the 4-day partition limit. There is NO resume: if the first
#   CTRL epochs (see the log) take > 15 min, stop and re-plan -- 100 epochs would not fit.
# Prerequisites: featurize_qm9.sbatch for every data VARIANT used (std, raw; rematch/nomatch/heavy additionally need
#   standardize_qm9.sbatch first). Lines whose cache is missing fail fast instead of racing to build it.
# Deferred arms: TABLE=$PROJECT/slurm/ablations_train_deferred.tsv sbatch --array=0-11%4 ablation_train_array.sbatch
# Env: N_EPOCHS (100), SEED (default when the TSV seed field is empty), LOADER_WORKERS (8).
# After training: SDE 20-step evaluation (generation seed 0 for every run -> all arms paired on the same RDKit L)
#   on RDKit L, plus on GT L (random and cycle) when GT_EVAL is 1.

PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
cuda_check

TABLE=${TABLE:-$PROJECT/slurm/ablations_train.tsv}
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$TABLE")
IDX=${SLURM_ARRAY_TASK_ID:-${IDX:-0}}
(( IDX < ${#LINES[@]} )) || { echo "index $IDX >= ${#LINES[@]} lines"; exit 1; }
IFS='|' read -r NAME VARIANT TRAIN_ARGS GEN_ARGS GT_EVAL TSEED GT_GEN_ARGS <<< "${LINES[$IDX]}"
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }
NAME=$(trim "$NAME"); VARIANT=$(trim "$VARIANT"); TRAIN_ARGS=$(trim "${TRAIN_ARGS:-}")
GEN_ARGS=$(trim "${GEN_ARGS:-}"); GT_EVAL=$(trim "${GT_EVAL:-0}")
N_EPOCHS=${N_EPOCHS:-100}; TSEED=$(trim "${TSEED:-}"); SEED=${TSEED:-${SEED:-0}}
# [round2] optional 7th field: extra generate args for the GT-L evals (S4 needs --l_level 1); 6-field lines: empty
GT_GEN_ARGS=$(trim "${GT_GEN_ARGS:-}")
RUN=qm9_${NAME}_e${N_EPOCHS}_s${SEED}

case "$VARIANT" in
  std)     STD="$QM9_STD" ;;
  raw)     STD="" ;;
  nomatch) STD="$QM9_DIR/standardized_pickles_nomatch" ;;
  mmff)    STD="$QM9_DIR/standardized_pickles_mmff" ;;
  de50)    STD="$QM9_DIR/standardized_pickles_de50" ;;
  rematch) STD="$QM9_DIR/standardized_pickles_rematch" ;;
  heavy)   STD="$QM9_DIR/standardized_pickles_heavy" ;;
  paired)  STD="$QM9_PAIRED" ;;   # [round2 S3/S4/B1cap] rematch pickles + aligned GT L (tools/build_paired_pickles.py)
  *) echo "unknown VARIANT $VARIANT"; exit 1 ;;
esac
CACHE=$QM9_DIR/cache_$VARIANT
[[ -s "$CACHE.train" && -s "$CACHE.val" ]] || { echo "missing $CACHE.{train,val}: sbatch featurize_qm9.sbatch with VARIANT=$VARIANT"; exit 1; }
echo "[$IDX] $NAME variant=$VARIANT run=$RUN train_args=$TRAIN_ARGS gen_args=$GEN_ARGS gt_eval=$GT_EVAL"

# Completion marker (review/cross_validation.md X1): best_model.pt exists after a time-limit kill too.
if [[ -s "$WORK/$RUN/best_model.pt" ]] && ! train_complete "$WORK/$RUN" "$N_EPOCHS"; then
    echo "WARNING: $WORK/$RUN incomplete (no .train_done) -> moving aside, retraining from scratch"
    mv "$WORK/$RUN" "$WORK/${RUN}_incomplete_$(date +%s)"
    [[ -d "$RES/$RUN" ]] && mv "$RES/$RUN" "$RES/${RUN}_incomplete_$(date +%s)"
fi
if ! train_complete "$WORK/$RUN" "$N_EPOCHS"; then
    # shellcheck disable=SC2086
    python train.py --log_dir "$WORK/$RUN" $QM9_MODEL_ARGS \
        --data_dir "$QM9_RAW" --std_pickles "$STD" --split_path "$QM9_SPLIT" --cache "$CACHE" \
        --n_epochs "$N_EPOCHS" --seed "$SEED" --loader_workers "${LOADER_WORKERS:-8}" \
        --num_conv_layers 4 --ns 32 --nv 8 --sigma_min 0.0314 --sigma_max 3.14 --lr 1e-3 --batch_size 32 \
        $TRAIN_ARGS
    touch "$WORK/$RUN/.train_done"
fi

# shellcheck disable=SC2086
SEED=0 STEPS=20 gen_eval "$WORK/$RUN" steps20_seed0 $GEN_ARGS
# GT_EVAL: 0 none; 1 gtL (random choice) + gtLcycle; 2 gtLcycle only ([round2] D8 trim of non-cycle gtL runs, F3)
if [[ "$GT_EVAL" == "1" || "$GT_EVAL" == "2" ]]; then
    flock "$QM9_DIR/.seed_pickles.lock" bash -c '[[ -s "$0" && -s "$1" ]] || python "$2/make_seed_pickles.py" \
        --test_csv "$3" --true_mols "$4" --out_gt_confs "$0" --out_gt_mols "$1"' \
        "$QM9_GT_SEED_CONFS" "$QM9_GT_SEED_MOLS" "$TOOLS" "$QM9_TEST_CSV" "$QM9_TEST_MOLS"
    # shellcheck disable=SC2086
    [[ "$GT_EVAL" == "1" ]] && SEED=0 STEPS=20 gen_eval "$WORK/$RUN" steps20_seed0_gtL --seed_confs "$QM9_GT_SEED_CONFS" $GT_GEN_ARGS
    # coupled-L oracle (each GT conformer's own L present; review C4) -- the random-choice run above is the
    # independent-L oracle. Their difference is the model-based estimate of what tau<->L coupling is worth.
    # shellcheck disable=SC2086
    SEED=0 STEPS=20 gen_eval "$WORK/$RUN" steps20_seed0_gtLcycle --seed_confs "$QM9_GT_SEED_CONFS" --seed_confs_cycle $GT_GEN_ARGS
fi
if [[ "${R2_STRICT:-0}" == "1" ]]; then
    # [round2 FIXES X3] gen_eval swallows generate/evaluate failures: fail the array task if an in-job eval is missing
    want=(steps20_seed0); [[ "$GT_EVAL" == "1" ]] && want+=(steps20_seed0_gtL)
    [[ "$GT_EVAL" == "1" || "$GT_EVAL" == "2" ]] && want+=(steps20_seed0_gtLcycle)
    for t in "${want[@]}"; do
        [[ -s $RES/$RUN/$t/eval.pkl ]] && grep -q '^SUMMARY' "$RES/$RUN/$t/summary.txt" || { echo "EVAL_FAILED $RUN/$t"; exit 3; }
    done
fi
echo "done: $(date)"
