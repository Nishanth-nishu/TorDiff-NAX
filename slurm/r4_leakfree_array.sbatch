#!/bin/bash
#SBATCH --job-name=tordiff_r4_leakfree
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M   # <= 5000 MB per CPU (round2 FIXES X2)
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%A_%a.log
#
# [round4 leak-free oracle test] Split-half true-geometry test (tools/make_split_half.py, tools/analyze_leakfree.py).
# ORACLE (test-set GT L): analysis only, never model selection. One array task per line of $MODELS
# (r4_leakfree_models.tsv); the task runs the 4 split-half conditions of its model, $PACK concurrent processes on its
# one GPU (default PACK=3, as r2_eval_array.sbatch / FIXES X4). Seed 0, 20 steps, --seed_confs_cycle.
#   split_AtoB : seeds_A, csv_B (2|B| confs), true_B      own_BtoB : seeds_B, csv_B, true_B
#   split_BtoA : seeds_B, csv_A (2|A| confs), true_A      own_AtoA : seeds_A, csv_A, true_A
# evaluate_confs.py is run UNCHANGED; the GT subset is given as a subset true_mols pickle (true_A / true_B).
# Inputs: $SPLIT (default $QM9_DIR/round4_splithalf, relative to the TD repo), built by make_split_half.py.
# Submit (from $PROJECT/slurm):  MODELS=$PWD/r4_leakfree_models.tsv sbatch --array=0-10%4 r4_leakfree_array.sbatch
PROJECT=/scratch/nishanth.r/tordiff
export PACK=${PACK:-3}
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
[[ "${R4_CPU:-0}" == "1" ]] || cuda_check
SPLIT=${SPLIT:-$QM9_DIR/round4_splithalf}
MODELS=${MODELS:-$PROJECT/slurm/r4_leakfree_models.tsv}
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$MODELS")
IDX=${SLURM_ARRAY_TASK_ID:-${IDX:-0}}
(( IDX < ${#LINES[@]} )) || { echo "index $IDX >= ${#LINES[@]} lines"; exit 1; }
IFS='|' read -r MODEL EXTRA <<< "${LINES[$IDX]}"
MODEL_DIR=$(eval echo "$(echo "$MODEL" | xargs)"); EXTRA=$(echo "${EXTRA:-}" | xargs)
[[ -s "$MODEL_DIR/best_model.pt" ]] || { echo "missing $MODEL_DIR/best_model.pt"; exit 1; }
N_EP=$(python -c "import yaml; print(yaml.safe_load(open('$MODEL_DIR/model_parameters.yml'))['n_epochs'])")
train_complete "$MODEL_DIR" "$N_EP" || { echo "ERROR: $MODEL_DIR training incomplete"; exit 1; }
for f in seeds_A.pkl seeds_B.pkl true_A.pkl true_B.pkl csv_A.csv csv_B.csv split.tsv; do
    [[ -s "$SPLIT/$f" ]] || { echo "ERROR: missing $SPLIT/$f (run tools/make_split_half.py)"; exit 1; }
done
RUN=$(basename "$MODEL_DIR")
LIMIT=${LIMIT_MOLS:+--limit_mols $LIMIT_MOLS}   # smoke tests only

# r4_gen_eval TAG SEEDS CSV TRUE : generate (seeded, cycled) -> evaluate on the GT subset; provenance-guarded like
# common.sh gen_eval (R2_PROVENANCE), plus the hashes of the csv / true-subset inputs.
r4_gen_eval() {
    local tag=$1 seeds=$2 csv=$3 true=$4
    local out=$RES/$RUN/$tag prov new
    mkdir -p "$out"; prov=$out/provenance.txt
    new=$( { printf 'model=%s\nextra=%s\nlimit=%s\nsteps=20 seed=0 cycle=1\n' "$MODEL_DIR" "$EXTRA" "$LIMIT";
             sha256sum "$MODEL_DIR/best_model.pt" "$seeds" "$csv" "$true" | cut -d' ' -f1;
             printf 'code=%s\npack=%s\ntools=%s\n' "$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" "$PACK" \
                    "$(sha256sum "$TOOLS/make_split_half.py" | cut -c1-16)"; } )
    if [[ -s $prov ]]; then
        [[ "$(cat "$prov")" == "$new" ]] || { echo "ERROR: $out was produced from other inputs (see $prov)"; return 1; }
    elif [[ -s $out/confs.pkl ]]; then
        echo "ERROR: $out has outputs but no provenance: refusing to reuse it"; return 1
    else
        echo "$new" > "$prov"
    fi
    if [[ ! -s $out/confs.pkl ]]; then
        echo "[r4] generate $RUN/$tag"
        # shellcheck disable=SC2086
        $TIMECMD python generate_confs.py --model_dir "$MODEL_DIR" --test_csv "$csv" --inference_steps 20 \
            --batch_size "${BATCH:-128}" --seed 0 --no_energy --seed_confs "$seeds" --seed_confs_cycle $LIMIT $EXTRA \
            --out "$out/confs.pkl" > "$out/generate.log" 2>&1
    fi
    if [[ ! -s $out/eval.pkl ]]; then
        echo "[r4] evaluate $RUN/$tag"
        # shellcheck disable=SC2086
        python evaluate_confs.py --confs "$out/confs.pkl" --test_csv "$csv" --true_mols "$true" --dataset qm9 \
            --n_workers "${EVAL_WORKERS:-2}" $LIMIT --out_results "$out/eval.pkl" > "$out/evaluate.log" 2>&1
    fi
    grep -E '^(SUMMARY|SWEEP)' "$out/evaluate.log" | tee "$out/summary.txt"
}

echo "[$IDX] model=$MODEL_DIR extra='$EXTRA' split=$SPLIT pack=$PACK"
T0=$(date +%s)
FAIL=$(mktemp "$TMPDIR/r4lf_fail.XXXX")
declare -a TAGS=()
while read -r COND SEEDS CSV TRUE; do
    TAG=r4lf_${COND}_ORACLE; TAGS+=("$TAG")
    while (( $(jobs -rp | wc -l) >= PACK )); do wait -n || true; done
    ( OMP_NUM_THREADS=1 r4_gen_eval "$TAG" "$SPLIT/$SEEDS" "$SPLIT/$CSV" "$SPLIT/$TRUE" || echo "$TAG" >> "$FAIL" ) &
done <<'EOC'
split_AtoB seeds_A.pkl csv_B.csv true_B.pkl
split_BtoA seeds_B.pkl csv_A.csv true_A.pkl
own_BtoB seeds_B.pkl csv_B.csv true_B.pkl
own_AtoA seeds_A.pkl csv_A.csv true_A.pkl
EOC
wait
for t in "${TAGS[@]}"; do
    if [[ ! -s $RES/$RUN/$t/eval.pkl ]] || ! grep -q '^SUMMARY' "$RES/$RUN/$t/summary.txt" 2>/dev/null; then
        grep -qx "$t" "$FAIL" || echo "$t" >> "$FAIL"
    fi
done
if [[ -s $FAIL ]]; then
    echo "FAILED tags ($RUN):"; sed 's/^/  EVAL_FAILED /' "$FAIL"; rm -f "$FAIL"; exit 1
fi
rm -f "$FAIL"
echo "all ${#TAGS[@]} tags have eval.pkl + SUMMARY; done: $(date)  wall $(( $(date +%s) - T0 )) s"
