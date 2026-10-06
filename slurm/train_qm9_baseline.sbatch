#!/bin/bash
#SBATCH --job-name=tordiff_qm9_base
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# DEFERRED in the 4-day budget (the released qm9_default checkpoint is the baseline; notes/ablation_plan_qm9.md section 3).
# Baseline torsional diffusion on GEOM-QM9, matching the released qm9_default hyper-parameters
# (4 conv layers, ns=32, nv=8, 2nd-order irreps, sigma in [0.0314, 3.14], lr 1e-3, bs 32, 250 epochs, plateau LR).
# Then generates 2K conformers per test molecule with 20 SDE steps and evaluates (COV/MAT at 0.5 A).
#
# Submit after featurize_qm9.sbatch (VARIANT=std) finished:
#   sbatch --dependency=afterok:<featurize_jobid> train_qm9_baseline.sbatch
#   SEED=1 sbatch train_qm9_baseline.sbatch          # second training seed
# Env: SEED (0), N_EPOCHS (250), RUN (qm9_baseline_s$SEED), LOADER_WORKERS (8)
# Expected: ~3-6 min/epoch on one modern GPU (data loading is CPU-bound numpy) => ~12-25 h for 250 epochs.
# NOTE: upstream has no true resume (restart_dir reloads best_model.pt weights only, no optimizer/epoch), so the
#       time limit must cover the whole run.

PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
cuda_check
env_report

SEED=${SEED:-0}
N_EPOCHS=${N_EPOCHS:-250}
RUN=${RUN:-qm9_baseline_s$SEED}
CACHE=$QM9_DIR/cache_std
[[ -s "$CACHE.train" && -s "$CACHE.val" ]] || { echo "missing $CACHE.{train,val}: run featurize_qm9.sbatch first"; exit 1; }

# Completion marker (review/cross_validation.md X1): best_model.pt is written whenever the val loss improves, so after a
# time-limit kill it EXISTS for a half-trained model. Only .train_done (written after train.py exits 0) means finished.
# A leftover incomplete run is moved aside (never deleted) and training restarts from scratch (no real resume).
if [[ -s "$WORK/$RUN/best_model.pt" ]] && ! train_complete "$WORK/$RUN" "$N_EPOCHS"; then
    echo "WARNING: $WORK/$RUN has best_model.pt but no .train_done (killed run?) -> moving to ${RUN}_incomplete_$(date +%s)"
    mv "$WORK/$RUN" "$WORK/${RUN}_incomplete_$(date +%s)"
    [[ -d "$RES/$RUN" ]] && mv "$RES/$RUN" "$RES/${RUN}_incomplete_$(date +%s)"
fi
if ! train_complete "$WORK/$RUN" "$N_EPOCHS" || [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
    [[ "${FORCE_TRAIN:-0}" == "1" && -d "$RES/$RUN" ]] && mv "$RES/$RUN" "$RES/${RUN}_stale_$(date +%s)"  # stale confs.pkl would be reused by gen_eval
    python train.py --log_dir "$WORK/$RUN" $QM9_MODEL_ARGS \
        --data_dir "$QM9_RAW" --std_pickles "$QM9_STD" --split_path "$QM9_SPLIT" --cache "$CACHE" \
        --n_epochs "$N_EPOCHS" --seed "$SEED" --loader_workers "${LOADER_WORKERS:-8}" \
        --num_conv_layers 4 --ns 32 --nv 8 --sigma_min 0.0314 --sigma_max 3.14 --lr 1e-3 --batch_size 32
    touch "$WORK/$RUN/.train_done"
fi

# evaluation of the trained model (and of the released checkpoint as a reproduction reference)
SEED=0 STEPS=20 gen_eval "$WORK/$RUN" steps20_seed0
if [[ -s "$WORK/pretrained/qm9_default/best_model.pt" && ! -s "$RES/qm9_default/steps20_seed0/eval.pkl" ]]; then
    SEED=0 STEPS=20 gen_eval "$WORK/pretrained/qm9_default" steps20_seed0
fi
echo "done: $(date)"
