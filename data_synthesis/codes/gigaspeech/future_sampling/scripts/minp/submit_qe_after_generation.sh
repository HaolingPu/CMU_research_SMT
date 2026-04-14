#!/usr/bin/env bash
set -e

# Submit QE<=3 filtering after a min-p generation job finishes.
#
# Usage:
#   bash submit_qe_after_generation.sh <min_p> <generation_job_id>
#
# Example:
#   bash submit_qe_after_generation.sh 0.05 7040000

MIN_P_VALUE="${1:?Usage: bash submit_qe_after_generation.sh <min_p> <generation_job_id>}"
GEN_JOB_ID="${2:?Usage: bash submit_qe_after_generation.sh <min_p> <generation_job_id>}"

OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_${MIN_P_VALUE}"
EXPERIMENT_DIR="${OUTPUT_ROOT}/job_${GEN_JOB_ID}"
METRICX_RUN_DIR="${OUTPUT_ROOT}/job_${GEN_JOB_ID}-metricx"
FILTERED_OUTPUT_DIR="${OUTPUT_ROOT}/job_${GEN_JOB_ID}-qe3"
QE_THRESHOLD=3.0
NUM_SHARDS=8
SCRIPT_DIR="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"

echo "Submitting QE<=3 pipeline"
echo "min_p=${MIN_P_VALUE}"
echo "generation_job_id=${GEN_JOB_ID}"
echo "output_root=${OUTPUT_ROOT}"
echo "experiment_dir=${EXPERIMENT_DIR}"
echo "metricx_run_dir=${METRICX_RUN_DIR}"
echo "filtered_output_dir=${FILTERED_OUTPUT_DIR}"
echo "qe_gpus=8"

PREP_JOB_ID=$(sbatch --parsable \
  --export="ALL,EXPERIMENT_DIR=${EXPERIMENT_DIR},METRICX_RUN_DIR=${METRICX_RUN_DIR},NUM_SHARDS=${NUM_SHARDS}" \
  "${SCRIPT_DIR}/run_metricx_qe_prepare.sbatch")
echo "[Stage 1] Prepare  : job ${PREP_JOB_ID}"

PREDICT_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB_ID}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SCRIPT_DIR}/run_metricx_qe_8gpu.sbatch")
echo "[Stage 2] Predict  : job ${PREDICT_JOB_ID} (afterok:${PREP_JOB_ID})"

FINAL_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${PREDICT_JOB_ID}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},EXPERIMENT_DIR=${EXPERIMENT_DIR},FILTERED_OUTPUT_DIR=${FILTERED_OUTPUT_DIR},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=${NUM_SHARDS}" \
  "${SCRIPT_DIR}/run_metricx_qe_finalize.sbatch")
echo "[Stage 3] Finalize : job ${FINAL_JOB_ID} (afterok:${PREDICT_JOB_ID})"

echo "Pipeline submitted: ${GEN_JOB_ID} -> ${PREP_JOB_ID} -> ${PREDICT_JOB_ID} -> ${FINAL_JOB_ID}"
