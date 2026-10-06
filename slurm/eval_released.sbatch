#!/bin/bash
#SBATCH --job-name=tordiff_evalrel
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
# Evaluate the paper's released QM9 conformers (qm9_steps20.pkl) with our patched evaluator: reproduction reference.
# The released Mols carry a pickled Python attribute dict (mol.rmsd etc.) whose __setstate__ form no installed RDKit
# accepts. A custom Unpickler rebuilds each Mol from its RDKit binary (the reduce args) and drops the attribute dict.
source /scratch/nishanth.r/tordiff/slurm/common.sh
set -euo pipefail
activate_env
for m in qm9_default qm9_1order; do
    out=$RES/released/$m; mkdir -p "$out"
    python - "$WORK/pretrained/$m/qm9_steps20.pkl" "$out/confs.pkl" <<'PY'
import pickle, sys
from rdkit import Chem

class _StubMol:
    def __init__(self, *args): self.args = args
    def __setstate__(self, state): self.state = state

class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'rdkit.Chem.rdchem' and name == 'Mol':
            return _StubMol
        return super().find_class(module, name)

def rebuild(x):
    if isinstance(x, _StubMol):
        return Chem.Mol(x.args[0])
    if isinstance(x, dict):
        return {k: rebuild(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(rebuild(v) for v in x)
    return x

d = rebuild(_Unpickler(open(sys.argv[1], 'rb')).load())
n_conf = sum(len(v) for v in d.values())
print('released pkl:', len(d), 'molecules,', n_conf, 'conformers')
pickle.dump(d, open(sys.argv[2], 'wb'))
PY
    python evaluate_confs.py --confs "$out/confs.pkl" --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" \
        --dataset qm9 --n_workers 8 --out_results "$out/eval.pkl" > "$out/evaluate.log" 2>&1
    echo "== $m"; grep -E "^(SUMMARY|SWEEP)" "$out/evaluate.log" || tail -20 "$out/evaluate.log"
done
