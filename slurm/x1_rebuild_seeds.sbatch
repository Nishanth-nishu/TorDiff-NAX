#!/bin/bash
#SBATCH --job-name=tordiff_r2_x1seeds
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=12
#SBATCH --mem=60000M
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
# FIXES X1: rebuild only the lambda / A5 seed pickles (+ X11 MMFF counts from an etkdg re-run into a temp dir)
PROJECT=/scratch/nishanth.r/tordiff
source "$PROJECT/slurm/common.sh"
set -euo pipefail
activate_env
export CUDA_VISIBLE_DEVICES=""
S=$QM9_DIR/round2_seeds; T=$QM9_DIR/round2_seeds_x1; E=$QM9_DIR/round2_seeds_x11_etkdg
rm -rf "$T" "$E"
python "$TOOLS/make_l_seed_pickles.py" --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" --out_dir "$T" \
    --sources interp --n_workers 12 --seed 1 | tail -30
mkdir -p "$S/old_x0"
mv "$S"/L_lam* "$S"/L_A5* "$S/old_x0/"
cp "$S/l_error.csv" "$S/old_x0/l_error.csv"
cp "$T"/L_lam* "$T"/L_A5* "$T/S4_test_subset.txt" "$S/"
python - <<PY
import pandas as pd
a = pd.read_csv('$S/old_x0/l_error.csv'); b = pd.read_csv('$T/l_error.csv')
keep = a[~(a.cond.str.startswith('L_lam') | a.cond.str.startswith('L_A5'))]
pd.concat([keep, b]).to_csv('$S/l_error.csv', index=False)
print('l_error rows', len(keep), '+', len(b))
PY
python "$TOOLS/make_l_seed_pickles.py" --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" --out_dir "$E" \
    --sources etkdg --n_workers 12 --seed 1 | grep -E "COUNTS|molecules"
python - <<PY
import pickle, numpy as np
for t in ('L_etkdg2L', 'L_etkdg2L_mmff'):
    a = pickle.load(open('$S/' + t + '.pkl', 'rb')); b = pickle.load(open('$E/' + t + '.pkl', 'rb'))
    same = a.keys() == b.keys() and all(np.allclose(x.GetConformer().GetPositions(), y.GetConformer().GetPositions())
                                        for k in a for x, y in zip(a[k], b[k]))
    print('X11 determinism', t, 'identical' if same else 'DIFFERENT')
PY
for f in "$S"/*.population.txt; do echo "$(basename $f): $(head -1 $f)"; grep set_counts "$f" || true; done
echo "S4 test subset: $(wc -l < $S/S4_test_subset.txt)"
echo "done: $(date)"
