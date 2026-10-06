#!/bin/bash
#SBATCH --job-name=tordiff_std
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# Re-runs conformer matching (standardize_confs.py) over all raw QM9 pickles to build alternative training sets
# (ablations B2/B3/B4). CPU only. The shipped data/QM9/standardized_pickles is the default variant -- do not overwrite it.
#   VARIANT=nomatch -> --no_match : skips the Hungarian (von-Mises cost) assignment of RDKit seeds to GT conformers but
#                      STILL optimises torsions by differential evolution (upstream flag semantics!)
#   VARIANT=mmff    -> --mmff     : MMFF94s-relax the ETKDG seeds before matching (pair with generate --pre_mmff)
#   VARIANT=de50    -> --max_iter 50 : better torsion matching (default 15 DE iterations)
#   VARIANT=rematch -> no flags      : the default recipe re-run HERE (same RDKit build/version as the variants). The
#                     shipped standardized_pickles came from an unknown RDKit version and unseeded ETKDG, so B2/B3/B4/B5
#                     must be compared against CTRL_rematch_100ep, not CTRL_base_100ep (review/cross_validation.md C3).
#   VARIANT=heavy   -> --heavy_objective (patch): heavy-atom DE objective (B5; code_concerns #7)
# --jobs_per_worker MUST stay 1000: utils/dataset.py hard-codes mols_per_pickle=1000 to locate a molecule's pickle.
# --time 12 h: resumable (finished NNN.pickle files are skipped on resubmit).
# Cost estimate: ~0.3-2 s per molecule (<=30 confs, DE popsize 15 x 15 iters) -> 133k mols / 32 procs ~ 1-4 h.

PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
export CUDA_VISIBLE_DEVICES=""

VARIANT=${VARIANT:-nomatch}
case "$VARIANT" in
  nomatch) FLAGS="--no_match" ;;
  mmff)    FLAGS="--mmff" ;;
  de50)    FLAGS="--max_iter 50" ;;
  rematch) FLAGS="" ;;                  # upstream recipe regenerated locally: the matched control for B2/B3/B4/B5
  heavy)   FLAGS="--heavy_objective" ;;  # B5: DE on the heavy-atom RMSD instead of the H-inclusive one
  *) echo "unknown VARIANT"; exit 1 ;;
esac
OUT=$QM9_DIR/standardized_pickles_$VARIANT
mkdir -p "$OUT" "$OUT/logs"
N_RAW=$(find "$QM9_RAW" -maxdepth 1 -name '*.pickle' | wc -l)
N_WORKERS=$(( (N_RAW + 999) / 1000 ))
echo "raw pickles: $N_RAW -> $N_WORKERS standardisation workers, variant=$VARIANT flags=$FLAGS"

run_one() {
    local i=$1 f
    f=$OUT/$(printf '%03d' "$i").pickle
    [[ -s "$f" ]] && return 0
    python standardize_confs.py --out_dir "$OUT" --root "$QM9_RAW" --confs_per_mol 30 --worker_id "$i" \
        --jobs_per_worker 1000 $FLAGS > "$OUT/logs/worker_$i.log" 2>&1
}
export -f run_one; export OUT FLAGS
seq 0 $((N_WORKERS - 1)) | xargs -P "${SLURM_CPUS_PER_TASK:-8}" -I{} bash -c 'run_one {}'
ls "$OUT"/*.pickle | wc -l
grep -h "ALL RMSD" "$OUT"/logs/worker_*.log | awk '{s+=$3; n++} END {if (n) print "mean matching RMSD over workers:", s/n}' || true
