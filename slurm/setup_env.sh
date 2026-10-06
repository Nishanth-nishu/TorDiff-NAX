#!/bin/bash
#SBATCH --job-name=tordiff_setup
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# One-time environment setup on gnode118 (everything under /scratch/nishanth.r/tordiff).
#
# BEFORE submitting (the log directory must exist or SLURM silently drops the log, and /scratch is only on gnode118):
#   srun -p plafnet2 -A plafnet2 -w gnode118 --time=00:05:00 mkdir -p /scratch/nishanth.r/tordiff/logs
#   copy C:\Users\HP\Desktop\tor_diff\{slurm,tools,patches} to $HOME/tordiff_src/ on the cluster (a few 100 KB)
#   sbatch $HOME/tordiff_src/slurm/setup_env.sh          (or: srun ... --pty bash $HOME/tordiff_src/slurm/setup_env.sh)
#
# Options (env vars):
#   TORCH_PROFILE=cu117 (default; torch 1.13.1 + PyG 2.0.4, closest to the 2022 paper stack; GPUs up to sm_86)
#   TORCH_PROFILE=cu118 (torch 2.0.1 + PyG 2.3.1; use if gnode118 has Ada/Hopper GPUs, sm_89/sm_90, or driver issues)
#   SRC_DIR=$HOME/tordiff_src    where slurm/ tools/ patches/ were copied
#   WHEELHOUSE=/path             offline install: pip --no-index --find-links $WHEELHOUSE (if gnode118 has no internet)
#   REPO_TARBALL=/path.tar.gz    offline: tarball of the torsional-diffusion clone instead of `git clone`

PROJECT=/scratch/nishanth.r/tordiff
SRC_DIR=${SRC_DIR:-$HOME/tordiff_src}
mkdir -p "$PROJECT/logs"
if [[ -f "$SRC_DIR/slurm/common.sh" ]]; then
    mkdir -p "$PROJECT"/{slurm,tools,patches}
    cp -r "$SRC_DIR"/slurm/. "$PROJECT/slurm/"; cp -r "$SRC_DIR"/tools/. "$PROJECT/tools/"; cp -r "$SRC_DIR"/patches/. "$PROJECT/patches/"
fi
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail

TORCH_PROFILE=${TORCH_PROFILE:-cu117}
PIP_EXTRA=()
[[ -n "${WHEELHOUSE:-}" ]] && PIP_EXTRA=(--no-index --find-links "$WHEELHOUSE")

# ------------------------------------------------------------------ 1. python 3.9 (repo needs >=3.9 for dict `|`)
PY=""
for cand in python3.9 /usr/bin/python3.9; do command -v "$cand" >/dev/null 2>&1 && PY=$(command -v "$cand") && break; done
if [[ -z "$PY" ]] && command -v module >/dev/null 2>&1; then
    module load python/3.9 2>/dev/null || module load python3/3.9 2>/dev/null || true
    command -v python3.9 >/dev/null 2>&1 && PY=$(command -v python3.9)
fi
if [[ -z "$PY" ]]; then
    echo "python3.9 not found -> bootstrapping micromamba into $PROJECT/micromamba"
    mkdir -p "$PROJECT/micromamba"
    if [[ ! -x "$PROJECT/micromamba/bin/micromamba" ]]; then
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$PROJECT/micromamba" bin/micromamba
    fi
    export MAMBA_ROOT_PREFIX=$PROJECT/micromamba
    "$PROJECT/micromamba/bin/micromamba" create -y -p "$PROJECT/py39" -c conda-forge python=3.9 pip
    PY=$PROJECT/py39/bin/python3.9
fi
echo "base python: $PY ($($PY --version))"

# ------------------------------------------------------------------ 2. venv + pinned deps
if [[ ! -x "$VENV/bin/python" ]]; then "$PY" -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install ${PIP_EXTRA[@]+"${PIP_EXTRA[@]}"} --upgrade "pip<24.1" wheel "setuptools<70"

cat > "$PROJECT/constraints.txt" <<'EOF'
numpy==1.23.5
scipy==1.10.1
pandas==1.5.3
networkx==2.8.8
EOF

case "$TORCH_PROFILE" in
  cu117)
    TORCH_SPEC="torch==1.13.1+cu117"; TORCH_INDEX="https://download.pytorch.org/whl/cu117"
    PYG_URL="https://data.pyg.org/whl/torch-1.13.1+cu117.html"
    PYG_EXT="torch-scatter==2.1.1+pt113cu117 torch-sparse==0.6.17+pt113cu117 torch-cluster==1.6.1+pt113cu117"
    PYG_SPEC="torch-geometric==2.0.4" ;;
  cu118)
    TORCH_SPEC="torch==2.0.1+cu118"; TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    PYG_URL="https://data.pyg.org/whl/torch-2.0.1+cu118.html"
    PYG_EXT="torch-scatter==2.1.2+pt20cu118 torch-sparse==0.6.18+pt20cu118 torch-cluster==1.6.3+pt20cu118"
    PYG_SPEC="torch-geometric==2.3.1" ;;
  *) echo "unknown TORCH_PROFILE=$TORCH_PROFILE"; exit 1 ;;
esac

if [[ -n "${WHEELHOUSE:-}" ]]; then
    pip install ${PIP_EXTRA[@]+"${PIP_EXTRA[@]}"} -c "$PROJECT/constraints.txt" numpy==1.23.5 "$TORCH_SPEC"
    # shellcheck disable=SC2086
    pip install ${PIP_EXTRA[@]+"${PIP_EXTRA[@]}"} $PYG_EXT
else
    pip install -c "$PROJECT/constraints.txt" numpy==1.23.5 "$TORCH_SPEC" --extra-index-url "$TORCH_INDEX"
    # shellcheck disable=SC2086
    pip install $PYG_EXT -f "$PYG_URL"
fi
pip install ${PIP_EXTRA[@]+"${PIP_EXTRA[@]}"} -c "$PROJECT/constraints.txt" \
    "$PYG_SPEC" \
    e3nn==0.5.1 opt-einsum-fx==0.1.4 "sympy<1.13" \
    rdkit==2022.9.5 \
    numpy==1.23.5 scipy==1.10.1 pandas==1.5.3 networkx==2.8.8 \
    spyrmsd==0.5.2 rmsd==1.5.1 \
    pyyaml==6.0.1 tqdm matplotlib==3.7.3 tabulate gdown==4.7.3
# e3nn 0.5.1 is the first release that guards BatchNorm against an empty running_mean list (GitHub issue #3 /
# DiffDock #14: bond_conv outputs only 0o irreps -> torch.cat([], out=...) crash with e3nn<=0.5.0).
pip freeze > "$PROJECT/requirements.lock.txt"
env_report

# ------------------------------------------------------------------ 3. repo @ pinned commit + ablation hooks patch
if [[ ! -d "$REPO/.git" ]]; then
    if [[ -n "${REPO_TARBALL:-}" ]]; then
        mkdir -p "$REPO" && tar -xzf "$REPO_TARBALL" -C "$REPO" --strip-components=1
    else
        git clone https://github.com/gcorso/torsional-diffusion "$REPO"
    fi
fi
cd "$REPO"
if [[ -d .git ]]; then
    git checkout -q "$UPSTREAM_COMMIT" 2>/dev/null || git checkout -q master
    if ! git log --oneline -5 | grep -q "Ablation hooks"; then
        git checkout -q -B flexitors-ablation-hooks
        git -c user.name=tordiff -c user.email=tordiff@localhost am "$PROJECT/patches/0001-ablation-hooks.patch" \
            || { git am --abort || true; git apply "$PROJECT/patches/0001-ablation-hooks.patch"; }
    fi
    git log --oneline -3
fi
# data symlink: repo/data -> $PROJECT/data
mkdir -p "$DATA"
[[ -e "$REPO/data" ]] || ln -s "$DATA" "$REPO/data"

# ------------------------------------------------------------------ 4. precompute wrapped-normal tables
# diffusion/torus.py writes .p.npy/.score.npy (~200 MB each, 5001x5001 float64) into the CWD on first import.
# Do it once here so concurrent array jobs never race on the write.
export PYTHONPATH=$REPO
( cd "$REPO" && [[ -s .p.npy && -s .score.npy ]] || python -c "import diffusion.torus" )
ls -la "$REPO"/.p.npy "$REPO"/.score.npy

# ------------------------------------------------------------------ 5. smoke test (imports, e3nn BatchNorm, GPU fwd/bwd, sampling)
cuda_check
python "$TOOLS/smoke_test.py"
echo "setup done: $(date)"
