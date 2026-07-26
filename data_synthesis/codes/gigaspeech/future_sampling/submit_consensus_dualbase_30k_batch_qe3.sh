#!/usr/bin/env bash
set -e

# =============================================================================
# 30k consensus decoding (batched, concurrent=32) + MetricX QE + filter QE<=3
#
# Pipeline:
#   1. consensus decoding (4 array tasks × 2 GPU, concurrent=32)
#   2. metricx prep (convert jsons → shards)
#   3. metricx QE (4 GPU array)
#   4. metricx finalize (summarize QE scores)
#   5. filter QE <= 3 + overall summary
# =============================================================================

GEN_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_dualbase_vllm_1000_batch.sbatch"
PREP_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_prepare_8shards.sbatch"
QE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_4gpu.sbatch"
FINALIZE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_finalize.sbatch"
SUMMARY_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_metrics_summary.sbatch"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
RUN_TAG="gemma4/minp005-30k-batch"
OUTPUT_BASE="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug"
CONS_OUTPUT="${OUTPUT_BASE}/${RUN_TAG}"
METRICX_RUN_DIR_BASE="${OUTPUT_BASE}/${RUN_TAG}-metricx"
FILTERED_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}-qe3"

TOTAL_ROWS=30000
NUM_TASKS=4
NUM_FUTURES=20
SECONDARY_NUM_FUTURES=10
FUTURE_TOKENS=20
NUM_CONCURRENT_CASES=32
NUM_QE_SHARDS=4
QE_THRESHOLD=3.0

# --- Step 1: Consensus decoding ---
echo "[submit] Step 1: consensus decoding (${TOTAL_ROWS} rows, ${NUM_TASKS} tasks, concurrent=${NUM_CONCURRENT_CASES})"
GEN_SUBMIT=$(sbatch \
  --array=0-$(( NUM_TASKS - 1 )) \
  --time=2-00:00:00 \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",NUM_TASKS="${NUM_TASKS}",NUM_FUTURES="${NUM_FUTURES}",SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}",NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES}" \
  "${GEN_SCRIPT}")
echo "  ${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

GEN_RUN_DIR="${CONS_OUTPUT}/job_${GEN_JOB_ID}"
METRICX_RUN_DIR="${METRICX_RUN_DIR_BASE}/job_${GEN_JOB_ID}"

# --- Step 2: MetricX prep ---
echo "[submit] Step 2: metricx prep (shards=${NUM_QE_SHARDS})"
PREP_SUBMIT=$(sbatch \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${PREP_SCRIPT}")
echo "  ${PREP_SUBMIT}"
PREP_JOB_ID=$(echo "${PREP_SUBMIT}" | awk '{print $4}')

# --- Step 3: MetricX QE (4 GPU array) ---
echo "[submit] Step 3: metricx QE array"
QE_SUBMIT=$(sbatch \
  --dependency=afterok:"${PREP_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${QE_SCRIPT}")
echo "  ${QE_SUBMIT}"
QE_JOB_ID=$(echo "${QE_SUBMIT}" | awk '{print $4}')

# --- Step 4: MetricX finalize (merge shards + summarize + filter QE<=3) ---
echo "[submit] Step 4: metricx finalize + summarize + QE filter"
FINALIZE_SUBMIT=$(sbatch \
  --dependency=afterok:"${QE_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}",EXPERIMENT_DIR="${GEN_RUN_DIR}",FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR}",QE_THRESHOLD="${QE_THRESHOLD}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${FINALIZE_SCRIPT}")
echo "  ${FINALIZE_SUBMIT}"
FINALIZE_JOB_ID=$(echo "${FINALIZE_SUBMIT}" | awk '{print $4}')

# --- Step 5: Overall summary ---
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
echo "Pipeline submitted:"
echo "  Step 1 (consensus)       : ${GEN_JOB_ID}   (array 0-3, 2 GPU each, concurrent=32)"
echo "  Step 2 (metricx prep)    : ${PREP_JOB_ID}"
echo "  Step 3 (metricx QE)      : ${QE_JOB_ID}   (array 0-3, 1 GPU each)"
echo "  Step 4 (finalize+filter) : ${FINALIZE_JOB_ID}"
echo "  Step 5 (summary)         : ${SUMMARY_JOB_ID}"
echo ""
echo "Experiment dir  : ${GEN_RUN_DIR}"
echo "MetricX dir     : ${METRICX_RUN_DIR}"
echo "Filtered output : ${FILTERED_OUTPUT_DIR}"
echo "========================================="
