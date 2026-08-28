#!/usr/bin/env bash
set -e

# Submit MetricX QE pipeline (prepare -> 8gpu predict -> finalize) for all 4 minp values.
# Each minp gets its own 3-stage chain with SLURM dependencies.

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"
OUTPUT_BASE="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp"
QE_THRESHOLD=3.0

submit_pipeline() {
  local min_p="$1"
  local label="$2"
  local experiment_dir="$3"

  local output_root="${OUTPUT_BASE}/consensus_decoding_en_zh_minp_${min_p}"
  local metricx_run_dir="${output_root}/all_40k-metricx"
  local filtered_output_dir="${output_root}/minp_${label}_qe3"

  echo "================================================"
  echo "min-p=${min_p} (${label})"
  echo "  experiment_dir    : ${experiment_dir}"
  echo "  metricx_run_dir   : ${metricx_run_dir}"
  echo "  filtered_output_dir: ${filtered_output_dir}"

  # Stage 1: Prepare
  local prep_id
  prep_id=$(sbatch --parsable \
    --export="ALL,EXPERIMENT_DIR=${experiment_dir},METRICX_RUN_DIR=${metricx_run_dir},NUM_SHARDS=8" \
    "${SCRIPT_DIR}/run_metricx_qe_prepare.sbatch")
  echo "  [Stage 1] Prepare  : job ${prep_id}"

  # Stage 2: 8-GPU predict
  local predict_id
  predict_id=$(sbatch --parsable \
    --dependency="afterok:${prep_id}" \
    --export="ALL,METRICX_RUN_DIR=${metricx_run_dir}" \
    "${SCRIPT_DIR}/run_metricx_qe_8gpu.sbatch")
  echo "  [Stage 2] Predict  : job ${predict_id} (afterok:${prep_id})"

  # Stage 3: Finalize + filter
  local final_id
  final_id=$(sbatch --parsable \
    --dependency="afterok:${predict_id}" \
    --export="ALL,METRICX_RUN_DIR=${metricx_run_dir},EXPERIMENT_DIR=${experiment_dir},FILTERED_OUTPUT_DIR=${filtered_output_dir},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=8" \
    "${SCRIPT_DIR}/run_metricx_qe_finalize.sbatch")
  echo "  [Stage 3] Finalize : job ${final_id} (afterok:${predict_id})"
  echo "  Pipeline: ${prep_id} -> ${predict_id} -> ${final_id}"
}

# minp=1e-2: resumed into merged/task_merged/
submit_pipeline 0.01 1em2 \
  "${OUTPUT_BASE}/consensus_decoding_en_zh_minp_0.01/merged/task_merged"

# minp=1e-3: original job structure
submit_pipeline 0.001 1em3 \
  "${OUTPUT_BASE}/consensus_decoding_en_zh_minp_0.001/job_7108774"

# minp=1e-4: resumed into merged/task_merged/
submit_pipeline 0.0001 1em4 \
  "${OUTPUT_BASE}/consensus_decoding_en_zh_minp_0.0001/merged/task_merged"

# minp=1e-5: resumed into merged/task_merged/
submit_pipeline 0.00001 1em5 \
  "${OUTPUT_BASE}/consensus_decoding_en_zh_minp_0.00001/merged/task_merged"

echo ""
echo "All 4 minp QE pipelines submitted."
