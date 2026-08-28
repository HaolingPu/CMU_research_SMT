#!/usr/bin/env bash
set -e

# Submit topup generation + QE pipeline + merge for minp 1e-4 and 1e-5.
#
# Flow per minp value:
#   1. Topup generation (array job)
#   2. QE prepare on topup dir only
#   3. QE 8-GPU predict
#   4. QE finalize + filter
#   5. Merge: copy topup QE-passed JSONs into existing qe3 dir

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"
OUTPUT_BASE="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp"
QE_THRESHOLD=3.0

submit_topup_pipeline() {
  local min_p="$1"
  local label="$2"
  local topup_sbatch="$3"
  local topup_dir_name="$4"

  local output_root="${OUTPUT_BASE}/consensus_decoding_en_zh_minp_${min_p}"
  local topup_dir="${output_root}/${topup_dir_name}"
  local metricx_run_dir="${output_root}/${topup_dir_name}-metricx"
  local topup_qe3_dir="${output_root}/${topup_dir_name}-qe3"
  local existing_qe3_dir="${output_root}/minp_${label}_qe3"

  echo "================================================"
  echo "min-p=${min_p} (${label})"
  echo "  topup_dir          : ${topup_dir}"
  echo "  metricx_run_dir    : ${metricx_run_dir}"
  echo "  topup_qe3_dir      : ${topup_qe3_dir}"
  echo "  existing_qe3_dir   : ${existing_qe3_dir}"

  # Stage 1: Topup generation
  local gen_id
  gen_id=$(sbatch --parsable "${topup_sbatch}")
  echo "  [Stage 1] Generate : job ${gen_id}"

  # Stage 2: QE prepare (afterok:generation)
  local prep_id
  prep_id=$(sbatch --parsable \
    --dependency="afterok:${gen_id}" \
    --export="ALL,EXPERIMENT_DIR=${topup_dir},METRICX_RUN_DIR=${metricx_run_dir},NUM_SHARDS=8" \
    "${SCRIPT_DIR}/run_metricx_qe_prepare.sbatch")
  echo "  [Stage 2] Prepare  : job ${prep_id} (afterok:${gen_id})"

  # Stage 3: QE 8-GPU predict (afterok:prepare)
  local predict_id
  predict_id=$(sbatch --parsable \
    --dependency="afterok:${prep_id}" \
    --export="ALL,METRICX_RUN_DIR=${metricx_run_dir}" \
    "${SCRIPT_DIR}/run_metricx_qe_8gpu.sbatch")
  echo "  [Stage 3] Predict  : job ${predict_id} (afterok:${prep_id})"

  # Stage 4: QE finalize + filter (afterok:predict)
  local final_id
  final_id=$(sbatch --parsable \
    --dependency="afterok:${predict_id}" \
    --export="ALL,METRICX_RUN_DIR=${metricx_run_dir},EXPERIMENT_DIR=${topup_dir},FILTERED_OUTPUT_DIR=${topup_qe3_dir},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=8" \
    "${SCRIPT_DIR}/run_metricx_qe_finalize.sbatch")
  echo "  [Stage 4] Finalize : job ${final_id} (afterok:${predict_id})"

  # Stage 5: Merge topup QE-passed into existing qe3 dir (afterok:finalize)
  local merge_id
  merge_id=$(sbatch --parsable \
    --dependency="afterok:${final_id}" \
    --job-name="minp_merge_${label}" \
    --partition=general \
    --qos=normal \
    --gres=gpu:L40S:1 \
    --time=00:30:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --output="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/minp_${label}_merge_%j.out" \
    --error="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/minp_${label}_merge_%j.err" \
    --wrap="set -e
echo '===== MERGE topup QE into existing qe3 ====='
echo 'src: ${topup_qe3_dir}'
echo 'dst: ${existing_qe3_dir}'
BEFORE=\$(find '${existing_qe3_dir}' -maxdepth 1 -name '*.json' -type f | wc -l)
echo \"Before: \${BEFORE} files\"
find '${topup_qe3_dir}' -maxdepth 1 -name '*.json' -type f -print0 | xargs -0 cp -n -t '${existing_qe3_dir}/'
AFTER=\$(find '${existing_qe3_dir}' -maxdepth 1 -name '*.json' -type f | wc -l)
echo \"After:  \${AFTER} files (added \$(( AFTER - BEFORE )))\"
echo '===== MERGE DONE ====='")
  echo "  [Stage 5] Merge   : job ${merge_id} (afterok:${final_id})"
  echo "  Pipeline: ${gen_id} -> ${prep_id} -> ${predict_id} -> ${final_id} -> ${merge_id}"
}

# minp=1e-4: topup 1k rows, 2 tasks
submit_topup_pipeline 0.0001 1em4 \
  "${SCRIPT_DIR}/run_minp_1em4_topup_general.sbatch" \
  topup_1k

# minp=1e-5: topup 5k rows, 4 tasks
submit_topup_pipeline 0.00001 1em5 \
  "${SCRIPT_DIR}/run_minp_1em5_topup_general.sbatch" \
  topup_5k

echo ""
echo "Both topup pipelines submitted."
