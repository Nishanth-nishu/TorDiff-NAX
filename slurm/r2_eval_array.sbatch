#!/bin/bash
#SBATCH --job-name=tordiff_r2_eval
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M   # FIXES X2: <= 5000 MB per CPU, so SLURM does not raise the CPU count
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%A_%a.log
#
# [round2 D4] Packed evaluation (plan 3 §2.5 graft): one array task per line of $MODELS (r2_eval_models_*.tsv); the
# task runs every condition of its eval sets (r2_evalsets.tsv) with $PACK concurrent gen_eval processes on
# its one GPU (default PACK=3: user ruling, FIXES X4). Provenance guard on (R2_PROVENANCE=1). Submit with --array=0-<n-1>%<k> (see submit_round2.sh).
PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
export PACK=${PACK:-3}
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
cuda_check
export R2_PROVENANCE=1 R2_EVALSETS=${R2_EVALSETS:-$PROJECT/slurm/r2_evalsets.tsv}
MODELS=${MODELS:?set MODELS=<r2_eval_models_*.tsv>}
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$MODELS")
IDX=${SLURM_ARRAY_TASK_ID:-${IDX:-0}}
(( IDX < ${#LINES[@]} )) || { echo "index $IDX >= ${#LINES[@]} lines"; exit 1; }
IFS='|' read -r MODEL SETS <<< "${LINES[$IDX]}"
MODEL=$(echo "$MODEL" | xargs); SETS=$(echo "$SETS" | xargs)
case "$MODEL" in
  PRE) MODEL_DIR=$WORK/pretrained/qm9_default ;;
  *)   MODEL_DIR=$(eval echo "$MODEL") ;;
esac
[[ -s "$MODEL_DIR/best_model.pt" ]] || { echo "missing $MODEL_DIR/best_model.pt"; exit 1; }
if [[ "$MODEL_DIR" != */pretrained/* ]]; then
    N_EP=$(python -c "import yaml; print(yaml.safe_load(open('$MODEL_DIR/model_parameters.yml'))['n_epochs'])")
    train_complete "$MODEL_DIR" "$N_EP" || { echo "ERROR: $MODEL_DIR training incomplete"; exit 1; }
fi
# fail fast if a needed seed pickle is missing (no lazy building, unlike ablation_inference_array.sbatch:54-59)
for p in $(grep -vE '^\s*(#|$)' "$R2_EVALSETS" | awk -F'|' -v s=",$SETS," '{g=$1; gsub(/ /,"",g); if (index(s, ","g",")) print $5}' \
           | grep -o '\$QM9_[A-Z0-9_]*/[^ ]*\|\$QM9_GT_SEED_CONFS' | sort -u); do
    f=$(eval echo "$p"); [[ -s "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done
echo "[$IDX] model=$MODEL_DIR sets=$SETS pack=${PACK:-3}"
T0=$(date +%s)
run_evalset "$MODEL_DIR" "$SETS"
echo "done: $(date)  wall $(( $(date +%s) - T0 )) s"
