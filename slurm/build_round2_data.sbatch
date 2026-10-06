#!/bin/bash
#SBATCH --job-name=tordiff_r2_data
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=12
#SBATCH --mem=60000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# [round2] CPU data builders (code_plan_2 §3-4; DECISION D1-D3, defects 1, 3, 4). STEP selects the work:
#   paired   tools/build_paired_pickles.py  rematch pickles + aligned GT L -> $QM9_PAIRED, then featurize cache_paired
#   seeds    tools/make_l_seed_pickles.py   S1 seed pickles + l_error.csv -> $QM9_SEEDS2
#   all      paired then seeds (sequential; each step is skipped if its outputs exist)
# CPU budget (vote_3 P2-2; DECISION D9): 12 cores by default, so that 4 training jobs (4-10 cores each) + the S5 MMFF
# standardization (4 cores) fit next to it on the 48-core node.
PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
export CUDA_VISIBLE_DEVICES=""
STEP=${STEP:-all}
NW=${SLURM_CPUS_PER_TASK:-8}

if [[ "$STEP" == "paired" || "$STEP" == "all" ]]; then
    python "$TOOLS/build_paired_pickles.py" --std_dir "$QM9_DIR/standardized_pickles_rematch" --raw_dir "$QM9_RAW" \
        --out_dir "$QM9_PAIRED" --n_workers "$NW" | tail -40
    python "$TOOLS/build_paired_pickles.py" --out_dir "$QM9_PAIRED" --summarize | tee "$QM9_PAIRED/SUMMARY.txt"
    CACHE=$QM9_DIR/cache_paired
    if [[ ! ( -s "$CACHE.train" && -s "$CACHE.val" ) ]]; then
        # same call as featurize_qm9.sbatch (serial featurization; job 522 shm limit)
        python train.py --log_dir "$WORK/_featurize_paired" --n_epochs 0 $QM9_MODEL_ARGS \
            --data_dir "$QM9_RAW" --std_pickles "$QM9_PAIRED" --split_path "$QM9_SPLIT" --cache "$CACHE" --num_workers 1
    fi
    python - <<PY
import pickle
for mode in ('train', 'val'):
    a = pickle.load(open('$QM9_DIR/cache_rematch.' + mode, 'rb')); b = pickle.load(open('$QM9_DIR/cache_paired.' + mode, 'rb'))
    sa, sb = {d.canonical_smi for d in a}, {d.canonical_smi for d in b}
    na, nb = sum(len(d.pos) for d in a), sum(len(d.pos) for d in b)
    assert all(hasattr(d, 'gt_pos') and len(d.gt_pos) == len(d.pos) for d in b)
    assert sb <= sa, 'paired molecules must be a subset of rematch molecules'
    print(f'CACHE {mode}: rematch {len(a)} mols / {na} confs; paired {len(b)} mols / {nb} confs '
          f'({100*len(b)/len(a):.2f}% mols, {100*nb/na:.2f}% confs)')
PY
fi

if [[ "$STEP" == "seeds" || "$STEP" == "all" ]]; then
    if [[ ! -s "$QM9_SEEDS2/l_error.csv" ]]; then
        python "$TOOLS/make_l_seed_pickles.py" --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" \
            --out_dir "$QM9_SEEDS2" --sources etkdg,noise,interp --n_workers "$NW" --seed 1 | tail -60
    fi
    head -1 "$QM9_SEEDS2"/*.population.txt
fi
echo "done: $(date)"
