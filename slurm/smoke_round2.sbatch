#!/bin/bash
#SBATCH --job-name=tordiff_r2_smoke
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# [round2] MODE=smoke | d4 | d5   (round2/IMPLEMENTATION.md; DECISION D4, D5, D7, D9)
#   smoke  20-molecule / 1-epoch training of every new flag through ablation_train_array.sbatch (smoke TSV), then the
#          S1/S1A5/S1CR/V18 inference conditions on the released model and the S4 set on the smoke S4 model, all with
#          --limit_mols 20. Smoke data live in data/QM9_smoke (never the production caches: V9).
#   d4     packing test: the D4 eval set (R0_base_seed0, A1c_gtL_cycle_model, A2_pre_mmff on the released model) packed
#          3 per GPU into results_r2test/; SUMMARY lines must equal round 1 exactly. Timing: one unpacked R0 first.
#   d5     data-wait: CTRL recipe on cache_rematch, 300 iterations, 3 loader workers pinned to 4 cores vs 8 workers.
PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
cuda_check
MODE=${MODE:-smoke}
SMK=$PROJECT/data/QM9_smoke

if [[ "$MODE" == "smoke" ]]; then
    mkdir -p "$SMK"
    for f in qm9 split.npy test_smiles.csv test_mols.pkl test_gt_seed_confs.pkl test_gt_seed_mols.pkl; do
        [[ -e "$SMK/$f" ]] || ln -s "$PROJECT/data/QM9/$f" "$SMK/$f"
    done
    export QM9_DIR=data/QM9_smoke QM9_RAW=data/QM9_smoke/qm9/ QM9_SPLIT=data/QM9_smoke/split.npy \
           QM9_TEST_CSV=data/QM9_smoke/test_smiles.csv QM9_TEST_MOLS=data/QM9_smoke/test_mols.pkl \
           QM9_GT_SEED_CONFS=data/QM9_smoke/test_gt_seed_confs.pkl QM9_GT_SEED_MOLS=data/QM9_smoke/test_gt_seed_mols.pkl \
           QM9_PAIRED=data/QM9_smoke/standardized_pickles_paired QM9_SEEDS2=data/QM9_smoke/round2_seeds \
           WORK=$PROJECT/workdir_r2smoke RES=$PROJECT/results_r2smoke
    mkdir -p "$WORK" "$RES"
    case "$QM9_DIR" in *smoke*) ;; *) echo "refusing: smoke must not touch production caches (V9)"; exit 1 ;; esac
    # paired pickles from the first rematch file only (the first 20 split molecules live there)
    python "$TOOLS/build_paired_pickles.py" --std_dir data/QM9/standardized_pickles_rematch --raw_dir data/QM9/qm9/ \
        --out_dir "$QM9_PAIRED" --files 000.pickle 001.pickle --n_workers 2 | tail -15
    for v in raw paired; do
        STDARG=""; [[ $v == paired ]] && STDARG="--std_pickles $QM9_PAIRED"
        [[ -s $QM9_DIR/cache_$v.train ]] || python train.py --log_dir "$WORK/_featurize_$v" --n_epochs 0 $QM9_MODEL_ARGS \
            --data_dir "$QM9_RAW" $STDARG --split_path "$QM9_SPLIT" --cache "$QM9_DIR/cache_$v" --num_workers 1 \
            --limit_train_mols 20
    done
    [[ -s $QM9_SEEDS2/l_error.csv ]] || python "$TOOLS/make_l_seed_pickles.py" --test_csv "$QM9_TEST_CSV" \
        --true_mols "$QM9_TEST_MOLS" --out_dir "$QM9_SEEDS2" --sources etkdg,noise,interp --limit_mols 20 --n_workers 4 | tail -25
    export TABLE=$PROJECT/slurm/ablations_smoke_round2.tsv N_EPOCHS=1 LOADER_WORKERS=3
    for i in 0 1 2 3; do
        IDX=$i bash "$PROJECT/slurm/ablation_train_array.sbatch" > "$RES/smk_train_$i.log" 2>&1 || echo "SMOKE_TRAIN_FAILED idx=$i"
        grep -E 'TIMING|Training Loss|Validation Loss|Traceback|Error|done:' "$RES/smk_train_$i.log" | tail -12 || true
    done
    for d in "$RES"/qm9_SMK_*; do for t in "$d"/*/summary.txt; do echo "$t: $(head -1 "$t")"; done; done
    export R2_PROVENANCE=1 R2_EVALSETS=$PROJECT/slurm/r2_evalsets.tsv GEN_EXTRA="--limit_mols 20" PACK=3
    T0=$(date +%s)
    ln -sfn "$PROJECT/workdir/pretrained/qm9_default" "$WORK/pretrained_link"
    run_evalset "$WORK/pretrained_link" "S1,S1A5,S1CR" || echo "SMOKE_EVAL_FAILED released"
    echo "[smoke] packed S1 on released model: $(( $(date +%s) - T0 )) s"
    run_evalset "$WORK/qm9_SMK_S4_lamcond_e1_s0" "S4" || echo "SMOKE_EVAL_FAILED S4"
    for t in "$RES"/pretrained_link/*/summary.txt "$RES"/qm9_SMK_S4_lamcond_e1_s0/*/summary.txt; do
        echo "$t: $(head -1 "$t")"; done
fi

if [[ "$MODE" == "d4" ]]; then
    export RES=$PROJECT/results_r2test R2_PROVENANCE=1 R2_EVALSETS=$PROJECT/slurm/r2_evalsets.tsv
    MD=$WORK/pretrained/qm9_default
    T0=$(date +%s)
    RES=$PROJECT/results_r2test/unpacked R2_PROVENANCE=0 SEED=0 STEPS=20 gen_eval "$MD" R0_base_seed0
    T1=$(date +%s); echo "D4_TIMING unpacked_one_run_s $((T1 - T0))"
    PACK=3 run_evalset "$MD" D4
    T2=$(date +%s); echo "D4_TIMING packed_three_runs_s $((T2 - T1))"
    for t in R0_base_seed0 A1c_gtL_cycle_model A2_pre_mmff; do
        a=$(grep '^SUMMARY' "$PROJECT/results/qm9_default/$t/summary.txt"); b=$(grep '^SUMMARY' "$RES/qm9_default/$t/summary.txt")
        if [[ "$a" == "$b" ]]; then echo "D4_REPRO $t IDENTICAL"; else echo "D4_REPRO $t DIFFERENT"; echo " r1: $a"; echo " r2: $b"; fi
    done
    a=$(grep '^SUMMARY' "$PROJECT/results/qm9_default/R0_base_seed0/summary.txt")
    b=$(grep '^SUMMARY' "$PROJECT/results_r2test/unpacked/qm9_default/R0_base_seed0/summary.txt")
    [[ "$a" == "$b" ]] && echo "D4_REPRO unpacked_R0 IDENTICAL" || { echo "D4_REPRO unpacked_R0 DIFFERENT"; echo " $b"; }
fi

if [[ "$MODE" == "d5" ]]; then
    D5W=$PROJECT/workdir_r2test
    CPUS=$(python -c "import os; c=sorted(os.sched_getaffinity(0)); print(','.join(map(str, c[:4])), ','.join(map(str, c[:10])))")
    set -- $CPUS; C4=$1; C10=$2
    for cfg in "3 $C4" "8 $C10"; do
        set -- $cfg
        echo "D5 loader_workers=$1 cores=$2"
        taskset -c "$2" python train.py --log_dir "$D5W/d5_w$1" $QM9_MODEL_ARGS --data_dir "$QM9_RAW" \
            --std_pickles "$QM9_DIR/standardized_pickles_rematch" --split_path "$QM9_SPLIT" --cache "$QM9_DIR/cache_rematch" \
            --n_epochs 1 --seed 0 --loader_workers "$1" --num_conv_layers 4 --ns 32 --nv 8 --sigma_min 0.0314 \
            --sigma_max 3.14 --lr 1e-3 --batch_size 32 --limit_train_iters 400 --log_timing 2>&1 | grep -E 'TIMING|Training Loss' || true
    done
fi
echo "done: $(date)"
