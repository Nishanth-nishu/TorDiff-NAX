# Shared environment + helpers for all torsional-diffusion QM9 jobs on gnode118.
# Sourced by every *.sbatch / *.sh in this folder. Do NOT put `set -e` here (callers enable it after diagnostics).
#
# Layout on /scratch (local to gnode118 only; nothing large goes to $HOME, which has a 30 GB quota):
#   $PROJECT/torsional-diffusion   upstream repo @ pinned commit + patches/0001-ablation-hooks.patch
#   $PROJECT/venv                  python 3.9 venv (pinned deps, see setup_env.sh)
#   $PROJECT/data/QM9              extracted qm9.tar.gz  (repo/data -> $PROJECT/data symlink)
#   $PROJECT/workdir/<run>         model checkpoints (best_model.pt, last_model.pt, model_parameters.yml)
#   $PROJECT/results/<run>/<tag>   generated conformers, evaluation pickles, breakdown CSVs
#   $PROJECT/logs                  SLURM logs (%x_%j.log)
#   $PROJECT/{slurm,tools,patches} copies of C:\Users\HP\Desktop\tor_diff\{slurm,tools,patches}

export PROJECT=${PROJECT:-/scratch/nishanth.r/tordiff}
export REPO=$PROJECT/torsional-diffusion
export VENV=$PROJECT/venv
export DATA=$PROJECT/data
export WORK=$PROJECT/workdir
export RES=$PROJECT/results
export TOOLS=$PROJECT/tools
export LOGS=$PROJECT/logs
export UPSTREAM_COMMIT=5f713b42d7000307655f272471014c6127ea59be

# keep every cache off $HOME
export TMPDIR=$PROJECT/tmp
export XDG_CACHE_HOME=$PROJECT/.cache
export PIP_CACHE_DIR=$PROJECT/.cache/pip
export TORCH_HOME=$PROJECT/.cache/torch
export MPLCONFIGDIR=$PROJECT/.cache/matplotlib
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# ---- QM9 paths (relative to $REPO; verify against the extracted archive, see download_data.sh output) ----
export QM9_DIR=${QM9_DIR:-data/QM9}
export QM9_RAW=${QM9_RAW:-$QM9_DIR/qm9/}                       # trailing slash REQUIRED (dataset.py slices len(root))
export QM9_STD=${QM9_STD:-$QM9_DIR/standardized_pickles}
export QM9_SPLIT=${QM9_SPLIT:-$QM9_DIR/split.npy}
export QM9_TEST_CSV=${QM9_TEST_CSV:-$QM9_DIR/test_smiles.csv}
export QM9_TEST_MOLS=${QM9_TEST_MOLS:-$QM9_DIR/test_mols.pkl}
export QM9_GT_SEED_CONFS=$QM9_DIR/test_gt_seed_confs.pkl         # built by tools/make_seed_pickles.py
export QM9_GT_SEED_MOLS=$QM9_DIR/test_gt_seed_mols.pkl
# Architecture of the released qm9_default model: 44 node features, 2nd-order irreps (parser default is 1st order!)
export QM9_MODEL_ARGS="--dataset qm9 --in_node_features 44 --use_second_order_repr"

TIMECMD=""; [[ -x /usr/bin/time ]] && TIMECMD="/usr/bin/time -v"
export TIMECMD
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$LOGS" "$WORK" "$RES" 2>/dev/null || true

diagnostics() {
    echo "==================== diagnostics ===================="
    echo "date        : $(date)"
    echo "host        : $(hostname)"
    echo "job         : ${SLURM_JOB_NAME:-n/a} id=${SLURM_JOB_ID:-n/a} array=${SLURM_ARRAY_TASK_ID:-n/a}"
    echo "partition   : ${SLURM_JOB_PARTITION:-n/a}  cpus=${SLURM_CPUS_PER_TASK:-n/a}  gpus=${CUDA_VISIBLE_DEVICES:-none}"
    echo "PROJECT     : $PROJECT"
    if [[ "$(hostname -s)" != gnode118* ]]; then echo "WARNING: not on gnode118 -- /scratch/nishanth.r is local to gnode118"; fi
    df -h "$PROJECT" 2>/dev/null | tail -1
    df -h "$HOME" 2>/dev/null | tail -1
    command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not available"
    echo "====================================================="
}

activate_env() {
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    cd "$REPO"
    export PYTHONPATH=$REPO:${PYTHONPATH:-}
}

cuda_check() {
    python - <<'EOF'
import sys, torch
print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'available', torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit('ERROR: CUDA not available inside the job')
print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0),
      'arch list', torch.cuda.get_arch_list())
x = torch.randn(256, 256, device='cuda'); print('matmul ok', float((x @ x).sum()) == float((x @ x).sum()))
EOF
}

env_report() {
    python - <<'EOF'
import importlib
for m in ['torch', 'torch_geometric', 'torch_scatter', 'torch_cluster', 'e3nn', 'rdkit', 'numpy', 'scipy',
          'pandas', 'networkx', 'spyrmsd', 'yaml']:
    try:
        print(f'{m:16s}', getattr(importlib.import_module(m), '__version__', '?'))
    except Exception as e:
        print(f'{m:16s} IMPORT FAILED: {e}')
EOF
}

# gen_eval MODEL_DIR TAG [extra generate_confs.py args...]
#   generate -> evaluate (patched, dumps per-molecule matrices) -> breakdown. Skips steps whose output exists.
#   Env: STEPS (default 20), BATCH (128), SEED (0), NO_ENERGY (1), EVAL_WORKERS ($SLURM_CPUS_PER_TASK)
gen_eval() {
    local model_dir=$1 tag=$2; shift 2
    local run; run=$(basename "$model_dir")
    local out=$RES/$run/$tag
    mkdir -p "$out"
    local energy_flag="--no_energy"; [[ "${NO_ENERGY:-1}" == "0" ]] && energy_flag=""
    if [[ ! -s $out/confs.pkl ]]; then
        echo "[gen_eval] generate $run/$tag : $*"
        $TIMECMD python generate_confs.py --model_dir "$model_dir" --test_csv "$QM9_TEST_CSV" \
            --inference_steps "${STEPS:-20}" --batch_size "${BATCH:-128}" --seed "${SEED:-0}" $energy_flag \
            --out "$out/confs.pkl" "$@" > "$out/generate.log" 2>&1
    fi
    if [[ ! -s $out/eval.pkl ]]; then
        echo "[gen_eval] evaluate $run/$tag"
        python evaluate_confs.py --confs "$out/confs.pkl" --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" \
            --dataset qm9 --n_workers "${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" --out_results "$out/eval.pkl" \
            > "$out/evaluate.log" 2>&1
    fi
    grep -E '^(SUMMARY|SWEEP)' "$out/evaluate.log" | tee "$out/summary.txt" || echo "no SUMMARY line (see $out/evaluate.log)"
    python "$TOOLS/breakdown.py" --results "$out/eval.pkl" --threshold 0.5 --out "$out/breakdown.csv" \
        > "$out/breakdown.log" 2>&1 || echo "breakdown failed (see $out/breakdown.log)"
}

# train_complete RUN_DIR N_EPOCHS : 0 if the training in RUN_DIR finished. Accepts the .train_done marker, or (runs
# trained before the marker existed) last_model.pt with epoch == N_EPOCHS-1, in which case the marker is written.
# best_model.pt alone is NOT evidence of completion: it is saved at every val improvement (review X1).
train_complete() {
    local d=$1 n=$2
    [[ -e "$d/.train_done" ]] && return 0
    [[ -s "$d/last_model.pt" ]] || return 1
    if python -c "import sys, torch; sys.exit(0 if torch.load('$d/last_model.pt', map_location='cpu')['epoch'] >= $n - 1 else 1)" 2>/dev/null; then
        touch "$d/.train_done"; return 0
    fi
    return 1
}
