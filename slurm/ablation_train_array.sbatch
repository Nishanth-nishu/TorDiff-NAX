#!/bin/bash
#SBATCH --job-name=tordiff_qm9_trainabl
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%A_%a.log
#
# CORE training ablations on QM9 (CTRL x3, B1 x3, CTRL_rematch x3, B2 x3, B5 x3), one line of ablations_train.tsv per
# array index (15 lines -> --array=0-14; keep in sync). The 6th TSV field is the training seed.
# GPU throttle: gnode118 has 4 GPUs. Submit this array only AFTER the inference array has finished
#   (--dependency=afterany:<inference array jobid>) so that %4 here is the only GPU consumer -> <= 4 GPUs in use.
# --time: [round2] the full 4-day limit. RESUME (round2/IMPLEMENTATION.md, "Resume"): a run that stopped early
#   (time limit, node failure, scancel) continues from <run>/last_model.pt with train.py --resume (model, optimizer,
#   scheduler, epoch, best val loss, RNG states). 15 min before the time limit SLURM sends USR1 to this script
#   (--signal=B:USR1@900); it is forwarded to train.py, which finishes the current epoch's checkpoint and exits 99, and
#   the task requeues itself (--requeue; JobRequeue=0 on this cluster, so requeue happens only because this job asks).
#   After scancel or a crash, resubmit the same array index with the same TABLE: it resumes, it does not restart.
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
# [round2 resume] An incomplete run with a last_model.pt RESUMES, if its saved arguments match this line; otherwise
# (no checkpoint yet, or different arguments) it is moved aside and retrained from scratch, as before.
RESUME_FLAG=""
if [[ -e "$WORK/$RUN" ]] && ! train_complete "$WORK/$RUN" "$N_EPOCHS"; then
    if [[ -s "$WORK/$RUN/last_model.pt" ]] && python - "$WORK/$RUN/model_parameters.yml" "$SEED" "$STD" "$N_EPOCHS" "$TRAIN_ARGS" <<'PY'
import shlex, sys, yaml
y = yaml.safe_load(open(sys.argv[1]))
seed, std, n_ep, extra = sys.argv[2], sys.argv[3], sys.argv[4], shlex.split(sys.argv[5])
ok = str(y.get('seed')) == seed and str(y.get('std_pickles') or '') == std and str(y.get('n_epochs')) == n_ep
for i, tok in enumerate(extra):          # every --flag [value] of this TSV line must match the saved arguments
    if tok.startswith('--'):
        key = tok[2:]
        val = extra[i + 1] if i + 1 < len(extra) and not extra[i + 1].startswith('--') else True
        ok &= str(y.get(key)) == str(val)
sys.exit(0 if ok else 1)
PY
    then
        RESUME_FLAG="--resume"
        echo "RESUME: $WORK/$RUN incomplete with matching arguments -> train.py --resume"
    else
        echo "WARNING: $WORK/$RUN incomplete without a usable checkpoint -> moving aside, retraining from scratch"
        mv "$WORK/$RUN" "$WORK/${RUN}_incomplete_$(date +%s)"
        [[ -d "$RES/$RUN" ]] && mv "$RES/$RUN" "$RES/${RUN}_incomplete_$(date +%s)"
    fi
fi
if ! train_complete "$WORK/$RUN" "$N_EPOCHS"; then
    # [round2 resume] train.py runs in the background so that the USR1 trap fires at once; it is forwarded to
    # train.py (only meaningful with --resume, which installs the handler), which exits 99 after the next checkpoint
    TRAIN_PID=""
    trap 'echo "USR1 (time limit in 15 min): forwarding to train.py"; [[ -n "$TRAIN_PID" ]] && kill -USR1 "$TRAIN_PID"' USR1
    # every round-2 run gets --resume from its first start, so the SIGUSR1 handler is always installed
    [[ -z "$RESUME_FLAG" ]] && RESUME_FLAG="--resume"
    set +e
    # shellcheck disable=SC2086
    python train.py --log_dir "$WORK/$RUN" $QM9_MODEL_ARGS \
        --data_dir "$QM9_RAW" --std_pickles "$STD" --split_path "$QM9_SPLIT" --cache "$CACHE" \
        --n_epochs "$N_EPOCHS" --seed "$SEED" --loader_workers "${LOADER_WORKERS:-8}" \
        --num_conv_layers 4 --ns 32 --nv 8 --sigma_min 0.0314 --sigma_max 3.14 --lr 1e-3 --batch_size 32 \
        $TRAIN_ARGS $RESUME_FLAG &
    TRAIN_PID=$!
    while true; do wait "$TRAIN_PID"; rc=$?; kill -0 "$TRAIN_PID" 2>/dev/null || break; done  # wait returns on traps
    set -e
    trap - USR1
    if [[ $rc -eq 99 ]]; then
        echo "train.py stopped after a checkpoint (rc 99): requeueing ${SLURM_JOB_ID:-?} to resume"
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
    [[ $rc -eq 0 ]] || { echo "train.py failed (rc $rc); resubmit this index to resume from last_model.pt"; exit $rc; }
    # post-training evaluation only after training is truly complete (epoch check, not just a clean exit)
    python -c "import sys, torch; sys.exit(0 if torch.load('$WORK/$RUN/last_model.pt', map_location='cpu')['epoch'] >= $N_EPOCHS - 1 else 1)" \
        || { echo "ERROR: train.py exited 0 but the last epoch is < $N_EPOCHS - 1"; exit 1; }
    touch "$WORK/$RUN/.train_done"
fi
train_complete "$WORK/$RUN" "$N_EPOCHS" || { echo "ERROR: $WORK/$RUN not complete; no evaluation"; exit 1; }

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
