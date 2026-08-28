#!/usr/bin/env bash
set -e

# Submit 5 wait-k generation jobs (k=3,6,9,12,15), each 50k rows on general.
# Chain a MetricX QE pipeline (prepare -> 8-GPU predict -> finalize) after each
# generation job via afterok dependencies.
#
# Each k's output layout:
#   {OUTPUT_ROOT}/waitk_k{K}_50k_general/job_{JOB_ID}/task_{0..7}/*.json
#   {OUTPUT_ROOT}/waitk_k{K}_50k_general/all_50k-metricx/
#   {OUTPUT_ROOT}/waitk_k{K}_50k_general/waitk_k{K}_qe3/

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/wait-k/script"
MINP_QE_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"
OUTPUT_BASE="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/wait-k"
QE_THRESHOLD=3.0

submit_pipeline() {
  local k="$1"
  local label="k${k}"
  local gen_sbatch="${SCRIPT_DIR}/run_waitk_${label}_50k_general.sbatch"
  local output_root="${OUTPUT_BASE}/waitk_${label}_50k_general"

  # Stage 1: submit generation. sbatch --parsable returns SLURM_ARRAY_JOB_ID
  # which matches the ${JOB_ID} used inside common.sh for RUN_DIR layout.
  local gen_id
  gen_id=$(sbatch --parsable "${gen_sbatch}")
  local experiment_dir="${output_root}/job_${gen_id}"
  local metricx_run_dir="${output_root}/all_50k-metricx"
  local filtered_output_dir="${output_root}/waitk_${label}_qe3"

  echo "================================================"
  echo "wait-k=${k} (${label})"
  echo "  gen_sbatch         : ${gen_sbatch}"
  echo "  experiment_dir     : ${experiment_dir}"
  echo "  metricx_run_dir    : ${metricx_run_dir}"
  echo "  filtered_output_dir: ${filtered_output_dir}"
  echo "  [Stage 1] Generate : job ${gen_id} (50k rows, array 0-7)"

  # Stage 2: QE prepare (afterok:all array tasks of generation)
  local prep_id
  prep_id=$(sbatch --parsable \
    --dependency="afterok:${gen_id}" \
    --export="ALL,EXPERIMENT_DIR=${experiment_dir},METRICX_RUN_DIR=${metricx_run_dir},NUM_SHARDS=8" \
    "${MINP_QE_DIR}/run_metricx_qe_prepare.sbatch")
  echo "  [Stage 2] Prepare  : job ${prep_id} (afterok:${gen_id})"

  # Stage 3: QE 8-GPU predict
  local predict_id
  predict_id=$(sbatch --parsable \
    --dependency="afterok:${prep_id}" \
    --export="ALL,METRICX_RUN_DIR=${metricx_run_dir}" \
    "${MINP_QE_DIR}/run_metricx_qe_8gpu.sbatch")
  echo "  [Stage 3] Predict  : job ${predict_id} (afterok:${prep_id})"

  # Stage 4: QE finalize + filter
  local final_id
  final_id=$(sbatch --parsable \
    --dependency="afterok:${predict_id}" \
    --export="ALL,METRICX_RUN_DIR=${metricx_run_dir},EXPERIMENT_DIR=${experiment_dir},FILTERED_OUTPUT_DIR=${filtered_output_dir},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=8" \
    "${MINP_QE_DIR}/run_metricx_qe_finalize.sbatch")
  echo "  [Stage 4] Finalize : job ${final_id} (afterok:${predict_id})"
  echo "  Pipeline: ${gen_id} -> ${prep_id} -> ${predict_id} -> ${final_id}"
}

for K in 3 6 9 12 15; do
  submit_pipeline "${K}"
done

echo ""
echo "All 5 wait-k pipelines submitted."
echo ""
echo "Monitor with:   squeue -u haolingp -o \"%.10i %.15j %.8T %.12r\""
echo "Summary after:  cat \${OUTPUT_BASE}/waitk_k*_50k_general/all_50k-metricx/summary.txt"
