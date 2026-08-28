#!/usr/bin/env bash
# Submit 3-stage MetricX QE pipeline (8 GPU parallel) after consensus decoding job.
#
# Stage 1: Prepare — convert consensus JSONs to MetricX input + split into 8 shards
# Stage 2: Predict — 8-GPU array, each shard runs MetricX QE
# Stage 3: Finalize — merge shards, summarize, filter QE <= threshold
#
# Usage:
#   bash submit_production_qe_filter.sh <consensus_job_id>
#   bash submit_production_qe_filter.sh 7033006

set -e

CONSENSUS_JOB_ID="${1:?Usage: bash submit_production_qe_filter.sh <consensus_job_id>}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/gemma4/production}"
EXPERIMENT_DIR="${OUTPUT_ROOT}/job_${CONSENSUS_JOB_ID}"
METRICX_RUN_DIR="${OUTPUT_ROOT}/job_${CONSENSUS_JOB_ID}-metricx"
FILTERED_OUTPUT_DIR="${OUTPUT_ROOT}/job_${CONSENSUS_JOB_ID}-qe3"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS=8

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"

echo "============================================"
echo "Production QE Filter Pipeline (8 GPU)"
echo "============================================"
echo "Consensus job         : ${CONSENSUS_JOB_ID}"
echo "Experiment dir        : ${EXPERIMENT_DIR}"
echo "MetricX run dir       : ${METRICX_RUN_DIR}"
echo "Filtered output dir   : ${FILTERED_OUTPUT_DIR}"
echo "QE threshold          : ${QE_THRESHOLD}"
echo "Num shards            : ${NUM_SHARDS}"
echo ""

# Stage 1: Prepare (depends on consensus job)
PREP_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${CONSENSUS_JOB_ID}" \
    --export="ALL,EXPERIMENT_DIR=${EXPERIMENT_DIR},METRICX_RUN_DIR=${METRICX_RUN_DIR},NUM_SHARDS=${NUM_SHARDS}" \
    "${SCRIPT_DIR}/run_metricx_qe_consensus_prepare_8shards.sbatch")
echo "[Stage 1] Prepare     : job ${PREP_JOB_ID} (afterok:${CONSENSUS_JOB_ID})"

# Stage 2: 8-GPU predict (depends on prepare)
PREDICT_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${PREP_JOB_ID}" \
    --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
    "${SCRIPT_DIR}/run_metricx_qe_consensus_8gpu.sbatch")
echo "[Stage 2] Predict     : job ${PREDICT_JOB_ID} (afterok:${PREP_JOB_ID})"

# Stage 3: Finalize + filter (depends on all predict shards)
FINAL_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${PREDICT_JOB_ID}" \
    --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},EXPERIMENT_DIR=${EXPERIMENT_DIR},FILTERED_OUTPUT_DIR=${FILTERED_OUTPUT_DIR},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=${NUM_SHARDS}" \
    "${SCRIPT_DIR}/run_metricx_qe_consensus_finalize.sbatch")
echo "[Stage 3] Finalize    : job ${FINAL_JOB_ID} (afterok:${PREDICT_JOB_ID})"

echo ""
echo "Pipeline submitted: ${CONSENSUS_JOB_ID} -> ${PREP_JOB_ID} -> ${PREDICT_JOB_ID} -> ${FINAL_JOB_ID}"
