#!/bin/bash
#SBATCH --job-name=tordiff_qm9_inf
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-22%4
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%A_%a.log
#
# CORE inference-only QM9 ablations on the RELEASED qm9_default checkpoint (R0-R2, A0, A1, A1c, A2, A3, G1).
# Each array index = one non-comment line of ablations_inference.tsv (23 lines -> --array=0-22; keep in sync).
# GPU throttle: %4 = all 4 GPUs of gnode118. Submit the training array with --dependency=afterany:<this jobid>, so
#   the two arrays never overlap (<= 4 GPUs in use at any time).
# Outputs: $RES/<model run>/<NAME>/{confs.pkl,eval.pkl,summary.txt,breakdown.csv}
# --time: 20 SDE steps on 1000 test mols x 2K confs is estimated at 0.5-2 h (RDKit embedding + CPU torsion updates
#   dominate; --pre_mmff/--post_mmff add MMFF on CPU). 6 h = 3x the upper estimate; a killed task only loses itself
#   (gen_eval skips finished steps on resubmit).
# Deferred lines: TABLE=$PROJECT/slurm/ablations_inference_deferred.tsv sbatch --array=0-17%4 --time=12:00:00 ...
#   (steps100 ~3-5 h; `--ode --likelihood full` costs (1 + #torsions) forward passes per step, ~5-10x).
# `--no_model` lines are CPU-only work but kept in the array for simplicity (~0.5 h of a GPU slot each).

PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
cuda_check

TABLE=${TABLE:-$PROJECT/slurm/ablations_inference.tsv}
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$TABLE")
IDX=${SLURM_ARRAY_TASK_ID:-${IDX:-0}}
(( IDX < ${#LINES[@]} )) || { echo "index $IDX >= ${#LINES[@]} lines"; exit 1; }
IFS='|' read -r NAME MODEL STEPS_ SEED_ NOE_ ARGS <<< "${LINES[$IDX]}"
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }
NAME=$(trim "$NAME"); MODEL=$(trim "$MODEL"); STEPS_=$(trim "$STEPS_"); SEED_=$(trim "$SEED_"); NOE_=$(trim "$NOE_")
ARGS=$(eval echo "$(trim "${ARGS:-}")")   # expand $QM9_* variables
case "$MODEL" in
  BASE) MODEL_DIR=${MODEL_BASE:-$WORK/qm9_baseline_s0} ;;
  PRE)  MODEL_DIR=$WORK/pretrained/qm9_default ;;
  PRE1) MODEL_DIR=$WORK/pretrained/qm9_1order ;;
  *)    MODEL_DIR=$MODEL ;;
esac
echo "[$IDX] $NAME model=$MODEL_DIR steps=$STEPS_ seed=$SEED_ no_energy=$NOE_ args=$ARGS"
[[ -s "$MODEL_DIR/best_model.pt" ]] || { echo "missing $MODEL_DIR/best_model.pt"; exit 1; }
# our own checkpoints must come from a FINISHED training (best_model.pt also exists after a time-limit kill; review X1)
if [[ "$MODEL_DIR" != */pretrained/* ]]; then
    N_EP=$(python -c "import yaml; print(yaml.safe_load(open('$MODEL_DIR/model_parameters.yml'))['n_epochs'])")
    train_complete "$MODEL_DIR" "$N_EP" || { echo "ERROR: $MODEL_DIR training incomplete (no .train_done / last epoch < $N_EP)"; exit 1; }
fi
if [[ "$ARGS" == *seed_confs* || "$ARGS" == *seed_mols* ]]; then
    # flock: several array tasks (A1*, G1*) can start together and would race on writing the pickles
    flock "$QM9_DIR/.seed_pickles.lock" bash -c '[[ -s "$0" && -s "$1" ]] || python "$2/make_seed_pickles.py" \
        --test_csv "$3" --true_mols "$4" --out_gt_confs "$0" --out_gt_mols "$1"' \
        "$QM9_GT_SEED_CONFS" "$QM9_GT_SEED_MOLS" "$TOOLS" "$QM9_TEST_CSV" "$QM9_TEST_MOLS"
fi

# shellcheck disable=SC2086
STEPS=$STEPS_ SEED=$SEED_ NO_ENERGY=$NOE_ gen_eval "$MODEL_DIR" "$NAME" $ARGS
echo "done: $(date)"
