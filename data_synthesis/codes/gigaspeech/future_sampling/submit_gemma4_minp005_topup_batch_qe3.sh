#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Top-up pipeline for Gemma4 dual-base batched consensus decoding with min-p 0.05.
#
# Default setup matches the current situation:
#   - 30k main run already finished
#   - another 4k is being run from row 30000
#   - this script continues from row 34000 with an extra 3k
#
# You can override the defaults at submit time, for example:
#   START_OFFSET=36000 TOTAL_ROWS=2000 bash submit_gemma4_minp005_topup_batch_qe3.sh
# =============================================================================

GEN_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_dualbase_vllm_1000_batch.sbatch"
PREP_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_prepare_8shards.sbatch"
QE_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_4gpu.sbatch"
FINALIZE_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_finalize.sbatch"
SUMMARY_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_metrics_summary.sbatch"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug}"

START_OFFSET="${START_OFFSET:-34000}"
TOTAL_ROWS="${TOTAL_ROWS:-3000}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-32}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-4}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"

RUN_NAME="${RUN_NAME:-minp005-topup-${TOTAL_ROWS}-from-${START_OFFSET}-batch}"
RUN_TAG="${RUN_TAG:-gemma4/${RUN_NAME}}"
CONS_OUTPUT="${OUTPUT_BASE}/${RUN_TAG}"
METRICX_RUN_DIR_BASE="${OUTPUT_BASE}/${RUN_TAG}-metricx"
FILTERED_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}-qe3"

echo "[plan] input_tsv=${INPUT_TSV}"
echo "[plan] start_offset=${START_OFFSET}"
echo "[plan] total_rows=${TOTAL_ROWS}"
echo "[plan] run_tag=${RUN_TAG}"
echo "[plan] qe_threshold=${QE_THRESHOLD}"

echo "[submit] Step 1: consensus decoding top-up (${TOTAL_ROWS} rows from offset ${START_OFFSET})"
GEN_SUBMIT=$(sbatch \
  --array=0-$(( NUM_TASKS - 1 )) \
  --time=2-00:00:00 \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",ROW_OFFSET="${START_OFFSET}",NUM_TASKS="${NUM_TASKS}",NUM_FUTURES="${NUM_FUTURES}",SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}",NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES}" \
  "${GEN_SCRIPT}")
echo "  ${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

GEN_RUN_DIR="${CONS_OUTPUT}/job_${GEN_JOB_ID}"
METRICX_RUN_DIR="${METRICX_RUN_DIR_BASE}/job_${GEN_JOB_ID}"

echo "[submit] Step 2: metricx prep (shards=${NUM_QE_SHARDS})"
PREP_SUBMIT=$(sbatch \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${PREP_SCRIPT}")
echo "  ${PREP_SUBMIT}"
PREP_JOB_ID=$(echo "${PREP_SUBMIT}" | awk '{print $4}')

echo "[submit] Step 3: metricx QE array"
QE_SUBMIT=$(sbatch \
  --dependency=afterok:"${PREP_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${QE_SCRIPT}")
echo "  ${QE_SUBMIT}"
QE_JOB_ID=$(echo "${QE_SUBMIT}" | awk '{print $4}')

echo "[submit] Step 4: metricx finalize + summarize + QE filter"
FINALIZE_SUBMIT=$(sbatch \
  --dependency=afterok:"${QE_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}",EXPERIMENT_DIR="${GEN_RUN_DIR}",FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR}",QE_THRESHOLD="${QE_THRESHOLD}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${FINALIZE_SCRIPT}")
echo "  ${FINALIZE_SUBMIT}"
FINALIZE_JOB_ID=$(echo "${FINALIZE_SUBMIT}" | awk '{print $4}')

SUMMARY_TXT="${METRICX_RUN_DIR}/overall_metrics.txt"
echo "[submit] Step 5: overall summary"
SUMMARY_SUBMIT=$(sbatch \
  --dependency=afterok:"${FINALIZE_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",SUMMARY_TXT="${SUMMARY_TXT}" \
  "${SUMMARY_SCRIPT}")
echo "  ${SUMMARY_SUBMIT}"
SUMMARY_JOB_ID=$(echo "${SUMMARY_SUBMIT}" | awk '{print $4}')

echo ""
echo "========================================="
echo "Top-up pipeline submitted:"
echo "  Step 1 (consensus)       : ${GEN_JOB_ID}"
echo "  Step 2 (metricx prep)    : ${PREP_JOB_ID}"
echo "  Step 3 (metricx QE)      : ${QE_JOB_ID}"
echo "  Step 4 (finalize+filter) : ${FINALIZE_JOB_ID}"
echo "  Step 5 (summary)         : ${SUMMARY_JOB_ID}"
echo ""
echo "Run tag         : ${RUN_TAG}"
echo "Start offset    : ${START_OFFSET}"
echo "Top-up rows     : ${TOTAL_ROWS}"
echo "Experiment dir  : ${GEN_RUN_DIR}"
echo "MetricX dir     : ${METRICX_RUN_DIR}"
echo "Filtered output : ${FILTERED_OUTPUT_DIR}"
echo "========================================="
