#!/bin/bash
#SBATCH --job-name=tordiff_featurize
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# Builds the featurisation cache (<cache>.train / <cache>.val pickles) once, CPU only, so that training jobs never
# race on writing it. ConformerDataset reuses ANY existing file at that path regardless of arguments (dataset.py:64),
# so every data variant gets its own cache path.
#   VARIANT=std (default)  -> conformer-matched RDKit local structures   (cache data/QM9/cache_std)
#   VARIANT=raw            -> raw GEOM conformers = ground-truth L, no matching (ablation B1; cache data/QM9/cache_raw)
#   VARIANT=nomatch|mmff|de50|rematch|heavy -> re-standardized pickles from standardize_qm9.sbatch (B2/B3/B4/CTRL-rematch/B5)
# --time 6 h (~1 h expected; CPU only, no GPU budget).
# Implementation: train.py with --n_epochs 0 (builds loaders, writes cache, trains nothing).

PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
export CUDA_VISIBLE_DEVICES=""   # CPU job

VARIANT=${VARIANT:-std}
case "$VARIANT" in
  std)     STD="$QM9_STD" ;;
  raw)     STD="" ;;
  nomatch) STD="$QM9_DIR/standardized_pickles_nomatch" ;;
  mmff)    STD="$QM9_DIR/standardized_pickles_mmff" ;;
  de50)    STD="$QM9_DIR/standardized_pickles_de50" ;;
  rematch) STD="$QM9_DIR/standardized_pickles_rematch" ;;
  heavy)   STD="$QM9_DIR/standardized_pickles_heavy" ;;
  *) echo "unknown VARIANT"; exit 1 ;;
esac
CACHE=$QM9_DIR/cache_$VARIANT

if [[ -s "$CACHE.train" && -s "$CACHE.val" ]]; then echo "cache exists: $CACHE.{train,val}"; exit 0; fi
python train.py --log_dir "$WORK/_featurize_$VARIANT" --n_epochs 0 $QM9_MODEL_ARGS \
    --data_dir "$QM9_RAW" --std_pickles "$STD" --split_path "$QM9_SPLIT" --cache "$CACHE" \
    --num_workers 1   # serial: a Pool returns every tensor as a shm file -> exceeds vm.max_map_count on 106k mols (job 522)
ls -la "$CACHE".*
