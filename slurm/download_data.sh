#!/bin/bash
#SBATCH --job-name=tordiff_data
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# Downloads GEOM-QM9 as preprocessed by the torsional-diffusion authors (NOT DRUGS: drugs.tar.gz is 25.3 GB) plus the
# released QM9 checkpoints, into /scratch/nishanth.r/tordiff/data (must run ON gnode118). No GPU needed.
#
# Source: shared Drive folder from the upstream README
#   https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7
#   qm9.tar.gz          id 1c75BAevYvoShxyxZZw0o4-zIQlyj_-H9   807,551,358 bytes (checked 2026-09-29), top-level dir QM9/
#   workdir/qm9_default (2nd-order irreps, the paper's QM9 model) and workdir/qm9_1order (1st-order irreps):
#       best_model.pt (~6.4 MB), model_parameters.yml, qm9_steps20.pkl (their generated test conformers)
# Expected archive content (per README + GitHub issues #12/#20; VERIFY in the log below):
#   QM9/qm9/*.pickle (raw GEOM pickles, 1 per molecule), QM9/standardized_pickles/NNN.pickle (conformer-matched,
#   1000 molecules each), QM9/split.npy (train/val/test indices into sorted raw pickle list),
#   QM9/test_smiles.csv, QM9/test_mols.pkl
# Extracted size was not measured locally (archive not downloaded); budget ~5-10 GB.

PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail

DL=$PROJECT/downloads
mkdir -p "$DL" "$DATA" "$WORK/pretrained/qm9_default" "$WORK/pretrained/qm9_1order"
[[ -x "$VENV/bin/python" ]] && source "$VENV/bin/activate"

gdrive_get() {  # gdrive_get FILE_ID OUT [EXPECTED_BYTES]
    local id=$1 out=$2 expect=${3:-}
    if [[ -s "$out" && ( -z "$expect" || $(stat -c %s "$out") == "$expect" ) ]]; then echo "have $out"; return 0; fi
    if command -v gdown >/dev/null 2>&1; then
        gdown --continue "$id" -O "$out" || true
    fi
    if [[ ! -s "$out" || ( -n "$expect" && $(stat -c %s "$out") != "$expect" ) ]]; then
        echo "gdown failed/incomplete -> curl with confirm token"
        local uuid
        uuid=$(curl -sL "https://drive.usercontent.google.com/download?id=${id}&export=download" \
               | grep -oE 'name="uuid" value="[^"]*"' | cut -d'"' -f4 || true)
        curl -L --fail --retry 5 -C - -o "$out" \
             "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t&uuid=${uuid}"
    fi
    local got; got=$(stat -c %s "$out")
    echo "$out: $got bytes"
    if [[ -n "$expect" && "$got" != "$expect" ]]; then echo "SIZE MISMATCH for $out (expected $expect)"; return 1; fi
    if head -c 200 "$out" | grep -qi '<html'; then echo "$out is an HTML page (Drive quota/virus-scan page?)"; return 1; fi
}

# ---------------------------------------------------------------- dataset
gdrive_get 1c75BAevYvoShxyxZZw0o4-zIQlyj_-H9 "$DL/qm9.tar.gz" 807551358
if [[ ! -d "$DATA/QM9" ]]; then
    echo "extracting (listing top-level entries first)"
    (tar -tzf "$DL/qm9.tar.gz" | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -rn | head -20) || true
    tar -xzf "$DL/qm9.tar.gz" -C "$DATA"
fi
[[ -e "$REPO/data" ]] || ln -s "$DATA" "$REPO/data"
du -sh "$DATA/QM9"; ls -la "$DATA/QM9"

# ---------------------------------------------------------------- released QM9 checkpoints (small)
gdrive_get 1x07OMdS5bvxOaj0q5D0bxbCICDhtG1hn "$WORK/pretrained/qm9_default/best_model.pt" 6388281
gdrive_get 1zx3ORi2zB4k7PgCJ6i1y1tTrC-3iwO_a "$WORK/pretrained/qm9_default/model_parameters.yml"
gdrive_get 1a9G_fXcIFuGckhr-00-C_-xP_yF6fRVC "$WORK/pretrained/qm9_default/qm9_steps20.pkl" || echo "(optional file failed)"
gdrive_get 1y4wRRDSbu2G8shYKtvYWfERCHs833zRo "$WORK/pretrained/qm9_1order/best_model.pt"
gdrive_get 1rrDMW5wzC2eDKbsRyUAj7K91qlmfwUel "$WORK/pretrained/qm9_1order/model_parameters.yml"
gdrive_get 1f_YQRkoeU12X8n8AY0v8c2xKZciDTjrO "$WORK/pretrained/qm9_1order/qm9_steps20.pkl" || echo "(optional file failed)"

# ---------------------------------------------------------------- layout / consistency checks (needs venv)
if [[ -x "$VENV/bin/python" ]]; then
    activate_env
    python - <<'EOF'
import glob, os, pickle, numpy as np, pandas as pd
d = 'data/QM9'
print('entries:', sorted(os.listdir(d)))
raw = sorted(glob.glob(os.path.join(d, 'qm9', '*.pickle')))
std = sorted(glob.glob(os.path.join(d, 'standardized_pickles', '*.pickle')))
print('raw pickles:', len(raw), ' standardized pickles:', len(std))
for name in ['split.npy', 'split0.npy']:
    p = os.path.join(d, name)
    if os.path.exists(p):
        s = np.load(p, allow_pickle=True)
        print(name, 'sizes', [len(x) for x in s], 'max index', max(int(np.max(x)) for x in s),
              '-> OUT OF RANGE (issue #20; patched dataset.py drops them)' if max(int(np.max(x)) for x in s) >= len(raw) else 'ok')
csv = pd.read_csv(os.path.join(d, 'test_smiles.csv'))
print('test_smiles.csv columns:', list(csv.columns), 'rows:', len(csv)); print(csv.head(3).to_string())
tm = pickle.load(open(os.path.join(d, 'test_mols.pkl'), 'rb'))
print('test_mols.pkl molecules:', len(tm), ' GT conformers total:', sum(len(v) for v in tm.values()))
col0 = csv.columns[0]
print('csv col0 keys found in test_mols:', csv[col0].isin(tm.keys()).mean())
EOF
    python "$TOOLS/make_seed_pickles.py" --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" \
        --out_gt_confs "$QM9_GT_SEED_CONFS" --out_gt_mols "$QM9_GT_SEED_MOLS"
fi
echo "data done: $(date)"
