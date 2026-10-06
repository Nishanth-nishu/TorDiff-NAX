#!/bin/bash
#SBATCH --job-name=tordiff_analysis
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/nishanth.r/tordiff/logs/%x_%j.log
#
# CPU-only analyses (plan items A0/J local-structure error + torsion-only floor, I breakdown + threshold sweep,
# local-geometry metrics of generated conformers, G stereo, H resampling).
# Each step is skipped if its output exists, so submit it TWICE (plan section 3 schedule):
#   (1) right after download_data.sh: the test-set floor / J-decomp (--mode test) and training-set residuals, which need
#       no GPU output; (2) after the inference and training arrays: J-run, breakdowns, stereo, paired comparisons.
#   MODEL_RUN : results sub-dir of the inference model (default qm9_default = the released checkpoint)
# Primary endpoints (pre-registered): AMR-R mean and COV-R at 0.5 A (TD protocol), Holm-corrected. The 0.05-0.25 A
# sweep is secondary / exploratory.
# --time: the --mode test floor is ~1.2 s per GT conformer per core measured locally (about 9 DE runs each), i.e. a few
#   hours on 32 cores for the QM9 test set (x2 with --mmff_seeds); 24 h is a generous cap for either submission.

PROJECT=/scratch/nishanth.r/tordiff
# shellcheck disable=SC1091
source "$PROJECT/slurm/common.sh"
diagnostics
set -euo pipefail
activate_env
export CUDA_VISIBLE_DEVICES=""
NW=${SLURM_CPUS_PER_TASK:-8}
MODEL_RUN=${MODEL_RUN:-qm9_default}
A=$RES/analysis; mkdir -p "$A"

# J/A0: local-structure error of ETKDG vs GT + torsion-oracle floor (test set), and training-set matching residuals
[[ -s $A/local_structure_test.csv ]] || python "$TOOLS/local_structure_analysis.py" --mode test \
    --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" --out "$A/local_structure_test.csv" --n_workers "$NW" \
    | tee "$A/local_structure_test.log"
[[ -s $A/local_structure_test_mmff.csv ]] || python "$TOOLS/local_structure_analysis.py" --mode test --mmff_seeds \
    --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" --out "$A/local_structure_test_mmff.csv" --n_workers "$NW" \
    | tee "$A/local_structure_test_mmff.log"
[[ -s $A/local_structure_std.csv ]] || python "$TOOLS/local_structure_analysis.py" --mode std \
    --std_pickles "$QM9_STD" --out "$A/local_structure_std.csv" | tee "$A/local_structure_std.log"

# J (run-matched floor, review M1): torsions of the actual generated conformers re-optimised against each GT conformer,
# keeping their exact RDKit local structures. floor_run <= observed min RMSD by construction; obs - floor_run is what
# a better torsion model could still gain, floor_run is what only a local-structure change can remove.
for tag in R0_base_seed0 A2_pre_mmff A3_post_mmff G1_gt_graph_seed_mols A0_rdkit_etkdg_only; do
    d=$RES/$MODEL_RUN/$tag
    [[ -s $d/eval.pkl && -s $d/confs.pkl && ! -s $d/floor_run.csv ]] || continue
    python "$TOOLS/local_structure_analysis.py" --mode run --confs "$d/confs.pkl" --results "$d/eval.pkl" \
        --true_mols "$QM9_TEST_MOLS" --out "$d/floor_run.csv" --n_workers "$NW" > "$d/floor_run.log" 2>&1 \
        || echo "floor_run failed for $d"
    grep -E '^(obs_min_rmsd|floor_run|torsion_headroom):' "$d/floor_run.log" | sed "s#^#$tag #" || true
done

# I + J: per-molecule breakdown of every evaluated configuration, joined with local-structure errors
for d in "$RES"/*/*/; do
    [[ -s "$d/eval.pkl" ]] || continue
    python "$TOOLS/breakdown.py" --results "$d/eval.pkl" --threshold 0.5 --local_csv "$A/local_structure_test.csv" \
        --out "$d/breakdown_local.csv" > "$d/breakdown_local.log" 2>&1 || echo "breakdown failed for $d"
done

# local-geometry error of generated conformers vs matched GT (bond MAE, angle MAE, torsion MAE)
for d in "$RES"/*/*/; do
    [[ -s "$d/eval.pkl" && -s "$d/confs.pkl" && ! -s "$d/geometry.csv" ]] || continue
    python "$TOOLS/geometry_metrics.py" --results "$d/eval.pkl" --confs "$d/confs.pkl" --test_csv "$QM9_TEST_CSV"         --true_mols "$QM9_TEST_MOLS" --out "$d/geometry.csv" --n_workers "$NW" > "$d/geometry.log" 2>&1         || echo "geometry failed for $d"
done

# G: stereo consistency of generated conformers
for tag in R0_base_seed0 A1_gtL_model G1_gt_graph_seed_mols A2_pre_mmff; do
    d=$RES/$MODEL_RUN/$tag
    [[ -s $d/confs.pkl && ! -s $d/stereo.csv ]] || continue
    python "$TOOLS/stereo_check.py" --confs "$d/confs.pkl" --test_csv "$QM9_TEST_CSV" --true_mols "$QM9_TEST_MOLS" \
        --out "$d/stereo.csv" | tee "$d/stereo.log"
done

# H: likelihood/energy importance resampling from 4x oversampled pools (pools come from the inference array)
H=$RES/$MODEL_RUN
SDE_POOLS=("$H"/H_pool_sde_seed1*/confs.pkl)
ODE_POOLS=("$H"/H_pool_odelik_seed2*/confs.pkl)
resample_eval() {  # tag mode extra... ; pools in POOLS
    local tag=$1 mode=$2; shift 2
    local out=$H/$tag; mkdir -p "$out"
    [[ -s $out/confs.pkl ]] || python "$TOOLS/resample_confs.py" --in_confs "${POOLS[@]}" --fill_from "$H/R0_base_seed0/confs.pkl" \
        --out_confs "$out/confs.pkl" --oversample 4 --mode "$mode" "$@"
    [[ -s $out/eval.pkl ]] || python evaluate_confs.py --confs "$out/confs.pkl" --test_csv "$QM9_TEST_CSV" \
        --true_mols "$QM9_TEST_MOLS" --dataset qm9 --n_workers "$NW" --out_results "$out/eval.pkl" > "$out/evaluate.log" 2>&1
    grep -E '^(SUMMARY|SWEEP)' "$out/evaluate.log" | sed "s/^/$tag /" || true
}
if [[ -s ${SDE_POOLS[0]:-/nonexistent} ]]; then
    POOLS=("${SDE_POOLS[@]}")
    resample_eval H0_sde_uniform uniform
    resample_eval H1_sde_mmff_T300 weights --no_dlogp --temp 300
    resample_eval H1_sde_mmff_T1000 weights --no_dlogp --temp 1000
    resample_eval H3_sde_topk_energy topk_energy
fi
if [[ -s ${ODE_POOLS[0]:-/nonexistent} ]]; then
    POOLS=("${ODE_POOLS[@]}")
    resample_eval H2_ode_uniform uniform
    resample_eval H2_ode_lik_T300 weights --temp 300
    resample_eval H2_ode_lik_T1000 weights --temp 1000
fi

# S1-S3: paired bootstrap over molecules vs. R0 (seed replicates averaged per molecule; Holm over primary endpoints)
M=$RES/$MODEL_RUN
spec() {  # spec NAME tag1 tag2 ... -> NAME=path1,path2 (existing eval.pkl only; empty if none)
    local name=$1; shift; local ps=()
    for t in "$@"; do [[ -s $M/$t/eval.pkl ]] && ps+=("$M/$t/eval.pkl"); done
    (( ${#ps[@]} )) && { local IFS=,; echo "$name=${ps[*]}"; }
}
PRIMARY=(--thresholds 0.5 0.05 0.1 0.25 --primary MAT-R COV-R@0.5)   # 0.05-0.25: exploratory
REF=$(spec R0 R0_base_seed0 R1_base_seed1 R2_base_seed2 || true)
ARMS=()
for a in "A0_rdkit A0_rdkit_etkdg_only" "A0_rand A0_rdkitL_random_tors" \
         "A1 A1_gtL_model A1_gtL_model_seed1 A1_gtL_model_seed2" \
         "A1c A1c_gtL_cycle_model A1c_gtL_cycle_model_seed1 A1c_gtL_cycle_model_seed2" \
         "A2 A2_pre_mmff A2_pre_mmff_seed1 A2_pre_mmff_seed2" "A3 A3_post_mmff A3_post_mmff_seed1 A3_post_mmff_seed2" \
         "G1 G1_gt_graph_seed_mols G1_gt_graph_seed_mols_seed1 G1_gt_graph_seed_mols_seed2" \
         "C5 C_steps5" "C10 C_steps10" "C50 C_steps50" "C100 C_steps100" "D20 D_ode_steps20" "D50 D_ode_steps50" \
         "Emin005 E_inf_sigmin_0.005pi" "Emin05 E_inf_sigmin_0.05pi" "Emax05 E_inf_sigmax_0.5pi"; do
    # shellcheck disable=SC2086
    s_=$(spec $a || true); [[ -n "$s_" ]] && ARMS+=(--arm "$s_")
done
if [[ -n "$REF" && ${#ARMS[@]} -gt 0 ]]; then
    python "$TOOLS/paired_compare.py" --ref "$REF" "${ARMS[@]}" "${PRIMARY[@]}" \
        --out "$A/paired_vs_R0_$MODEL_RUN.md" | tee "$A/paired_vs_R0_$MODEL_RUN.log" || echo "paired_compare failed"
fi
# A1 (independent GT L) vs A1c (coupled GT L): model-based value of tau<->L coupling (review C4)
REF1=$(spec A1 A1_gtL_model A1_gtL_model_seed1 A1_gtL_model_seed2 || true)
ARM1=$(spec A1c A1c_gtL_cycle_model A1c_gtL_cycle_model_seed1 A1c_gtL_cycle_model_seed2 || true)
if [[ -n "$REF1" && -n "$ARM1" ]]; then
    python "$TOOLS/paired_compare.py" --ref "$REF1" --arm "$ARM1" "${PRIMARY[@]}" \
        --out "$A/paired_A1c_vs_A1_$MODEL_RUN.md" | tee "$A/paired_A1c_vs_A1_$MODEL_RUN.log" || true
fi

# Training arms (ablations_train.tsv): training seeds averaged per molecule, then paired over molecules. All trained
# models are evaluated with generation seed 0, so the arms share the same RDKit local structures.
tspec() {  # tspec NAME RUNPREFIX TAG -> NAME=$RES/<prefix>_s0/<tag>/eval.pkl,... over training seeds 0-2
    local name=$1 pre=$2 tag=$3 ps=()
    for s in 0 1 2; do [[ -s $RES/${pre}_s$s/$tag/eval.pkl ]] && ps+=("$RES/${pre}_s$s/$tag/eval.pkl"); done
    (( ${#ps[@]} )) && { local IFS=,; echo "$name=${ps[*]}"; }
}
tcompare() {  # tcompare OUTNAME REFSPEC ARMSPEC...
    local out=$1 ref=$2; shift 2; local arms=()
    for x in "$@"; do [[ -n "$x" ]] && arms+=(--arm "$x"); done
    [[ -n "$ref" && ${#arms[@]} -gt 0 ]] || return 0
    python "$TOOLS/paired_compare.py" --ref "$ref" "${arms[@]}" "${PRIMARY[@]}" --out "$A/$out.md" | tee "$A/$out.log" || true
}
E=e100
tcompare paired_B1_vs_CTRLbase "$(tspec CTRL_base qm9_CTRL_base_100ep_$E steps20_seed0 || true)" \
    "$(tspec B1 qm9_B1_train_gtL_$E steps20_seed0 || true)" "$(tspec CTRL_rematch qm9_CTRL_rematch_100ep_$E steps20_seed0 || true)"
tcompare paired_B_vs_CTRLrematch "$(tspec CTRL_rematch qm9_CTRL_rematch_100ep_$E steps20_seed0 || true)" \
    "$(tspec B2 qm9_B2_DEonly_randpair_$E steps20_seed0 || true)" "$(tspec B5 qm9_B5_match_heavy_$E steps20_seed0 || true)"
# 2x2 {train L} x {test L}: on GT local structures, B1 (trained on GT L) vs CTRL (trained on RDKit-matched L)
tcompare paired_B1_vs_CTRLbase_gtL "$(tspec CTRL_base_gtL qm9_CTRL_base_100ep_$E steps20_seed0_gtL || true)" \
    "$(tspec B1_gtL qm9_B1_train_gtL_$E steps20_seed0_gtL || true)"
tcompare paired_B1_vs_CTRLbase_gtLcycle "$(tspec CTRL_base_gtLc qm9_CTRL_base_100ep_$E steps20_seed0_gtLcycle || true)" \
    "$(tspec B1_gtLc qm9_B1_train_gtL_$E steps20_seed0_gtLcycle || true)"

# one table with every summary line
grep -H -E '^(SUMMARY|SWEEP thr=0.(050|100|250))' "$RES"/*/*/evaluate.log | sed "s#$RES/##; s#/evaluate.log:SUMMARY##" | sort > "$A/all_summaries.txt" || true
cat "$A/all_summaries.txt"
echo "done: $(date)"
