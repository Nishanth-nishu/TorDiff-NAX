#!/bin/bash
#SBATCH --job-name=r5_cpu
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# Round 5 Stage A milestone 1, CPU only (no GPU). One task per submission: sbatch --export=ALL,TASK=<task> run.sbatch
#   tests      flexitors + tools test suites in the TD venv (cluster-only tests included)
#   rescore    budget-matched leak-free reference (EXP-1 §3.6 / §10.5): tools/analyze_leakfree.py --ref_subsample 10
#   floor      fixed floor + split.tsv other-half floor, --mode test (EXP-1 §9 item 1, §10.5)
#   floor_mmff the same with --mmff_seeds
#   valset     QM9 validation subset (EVAL §1.1) with the test-file self-check
#   gate0      ETKDG / MMFF validation seeds and Gate-0 baselines (EXP-1 §10.3), after valset
# Outputs: $RES/round5/<task>/ (never overwritten: a task whose output marker exists is skipped).
# Memory: <= 5000 MB per CPU on plafnet2 (round2/FIXES.md X2).
PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
export CUDA_VISIBLE_DEVICES=""
export FLEXITORS_COMMIT
FLEXITORS_COMMIT=$(cat "$PROJECT/FLEXITORS_COMMIT")
export PYTHONPATH=$PROJECT
NW=${SLURM_CPUS_PER_TASK:-8}
TASK=${TASK:?set TASK}
O=$RES/round5/$TASK
mkdir -p "$O"
cd "$REPO"
echo "TASK $TASK commit $FLEXITORS_COMMIT out $O"
case $TASK in
  tests)
    (cd "$PROJECT" && python -m pytest -q -p no:cacheprovider flexitors/tests 2>&1 | tee "$O/flexitors_tests.txt")
    (cd "$PROJECT" && python -m pytest -q -p no:cacheprovider tools/tests/test_lsa_floor.py \
       tools/tests/test_leakfree.py tools/tests/test_leakfree_budget.py 2>&1 | tee "$O/tools_tests.txt")
    ;;
  rescore)
    [[ -s $O/budget_matched_summary.txt ]] || python "$TOOLS/analyze_leakfree.py" --res "$RES" \
      --split_dir "$QM9_DIR/round4_splithalf" --s4_subset "$QM9_SEEDS2/S4_test_subset.txt" --out "$O" \
      --ref_subsample 10 > "$O/stdout.txt"
    # the round-4 part must reproduce the committed round-4 analysis exactly
    cmp "$O/leakfree_summary.txt" "$RES/analysis_r4_leakfree/leakfree_summary.txt" && echo "round-4 summary reproduced"
    ;;
  floor|floor_mmff)
    EXTRA=""; [[ $TASK == floor_mmff ]] && EXTRA="--mmff_seeds"
    [[ -s $O/local_structure_test_fixed.csv ]] || python "$TOOLS/local_structure_analysis.py" --mode test $EXTRA \
      --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" --floor_other_half "$QM9_DIR/round4_splithalf/split.tsv" \
      --out "$O/local_structure_test_fixed.csv" --n_workers "$NW" > "$O/stdout.txt"
    ;;
  valset)
    [[ -s $O/val_smiles.csv ]] || python -m flexitors.data.qm9_val --raw_dir "$QM9_RAW" --split "$QM9_SPLIT" \
      --test_csv "$QM9_TEST_CSV" --test_mols "$QM9_TEST_MOLS" --train_scaffolds --out_dir "$O" | tee "$O/stdout.txt"
    ;;
  gate0)
    V=$RES/round5/valset
    for SET in val test; do
      if [[ $SET == val ]]; then CSV=$V/val_smiles.csv; MOLS=$V/val_mols.pkl; SEEDS=$O/val_seeds
      else CSV=$QM9_TEST_CSV; MOLS=$QM9_TEST_MOLS; SEEDS=$QM9_SEEDS2; fi
      if [[ $SET == val && ! -s $SEEDS/l_error.csv ]]; then
        python "$TOOLS/make_l_seed_pickles.py" --test_csv "$CSV" --true_mols "$MOLS" --out_dir "$SEEDS" \
          --sources etkdg --n_workers "$NW" > "$O/${SET}_seeds.log"
      fi
      for L in L_etkdg2L L_etkdg2L_mmff; do
        E=$O/${SET}_$L
        mkdir -p "$E"
        [[ -s $E/confs.pkl ]] || python -m flexitors.data.gate0_baselines confs --seeds "$SEEDS/$L.pkl" \
          --csv "$CSV" --out "$E/confs.pkl"
        [[ -s $E/eval.pkl ]] || python evaluate_confs.py --confs "$E/confs.pkl" --test_csv "$CSV" \
          --true_mols "$MOLS" --dataset qm9 --n_workers "$NW" --out_results "$E/eval.pkl" > "$E/eval.log" 2>&1
        [[ -s $E/breakdown.csv ]] || python "$TOOLS/breakdown.py" --results "$E/eval.pkl" --out "$E/breakdown.csv" \
          > "$E/breakdown.log"
      done
    done
    python -m flexitors.data.gate0_baselines summary --dir "$O" --val_l_error "$O/val_seeds/l_error.csv" \
      --test_l_error "$QM9_SEEDS2/l_error.csv" --val_csv "$V/val_smiles.csv" --test_csv "$QM9_TEST_CSV" \
      | tee "$O/gate0_baselines.txt"
    ;;
  *) echo "unknown TASK $TASK"; exit 2 ;;
esac
echo "done $TASK"
