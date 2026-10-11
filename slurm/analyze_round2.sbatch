#!/bin/bash
#SBATCH --job-name=tordiff_r2_analysis
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
# Round-2 pre-declared analysis (tools/analyze_round2.py, FIXES X10). CPU only. 8 CPUs x 5000 MB (plafnet2 cap).
PROJECT=/scratch/nishanth.r/tordiff
source $PROJECT/slurm/common.sh
diagnostics
set -euo pipefail
activate_env
export CUDA_VISIBLE_DEVICES=
python $TOOLS/analyze_round2.py --res $RES --seeds $QM9_SEEDS2 --test_csv $QM9_TEST_CSV --out $RES/analysis_r2
ls -la $RES/analysis_r2
