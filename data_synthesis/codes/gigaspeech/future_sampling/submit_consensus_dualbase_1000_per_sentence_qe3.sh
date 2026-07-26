#!/usr/bin/env bash
# ============================================================
# Submit the full 1000-case consensus-decoding + per-sentence MetricX QE
# pipeline.
#
#   generation  (dual-base vLLM, 8-GPU sbatch, TOTAL_ROWS=1000, NUM_TASKS=4)
#     --> per-sentence MetricX prepare (convert by src_text_full units)
#     --> MetricX QE 4-GPU array
#     --> per-sentence finalize: AND filter, keep iff every sentence QE <= 3
#
# Outputs:
#   ${CONS_OUTPUT}/job_<id>/              per-utt consensus JSONs
#   ${METRICX_RUN_DIR}/metricx_output.jsonl
#   ${METRICX_RUN_DIR}/filter_report_per_utt.jsonl
#   ${FILTERED_OUTPUT_DIR}/               kept JSONs (AND filter passed)
# ============================================================
set -e

GEN_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_dualbase_vllm_1000_batch.sbatch"
PREP_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_prepare_per_sentence.sbatch"
QE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_4gpu.sbatch"
FINALIZE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_finalize_per_sentence.sbatch"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
RUN_TAG="${RUN_TAG:-consensus-dualbase-1000-per-sentence-qe3}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug}"
CONS_OUTPUT="${CONS_OUTPUT:-${OUTPUT_BASE}/${RUN_TAG}}"
METRICX_RUN_DIR_BASE="${METRICX_RUN_DIR_BASE:-${OUTPUT_BASE}/${RUN_TAG}-metricx}"
FILTERED_OUTPUT_BASE="${FILTERED_OUTPUT_BASE:-${OUTPUT_BASE}/${RUN_TAG}-filtered}"

TOTAL_ROWS="${TOTAL_ROWS:-1000}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-32}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MIN_P="${MIN_P:-0.0}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-4}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"

PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL:-google/gemma-4-E2B}"
SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL:-0.45}"
SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL:-0.45}"
INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL:-0.90}"

GEN_ARRAY_MAX=$(( NUM_TASKS - 1 ))

echo "[submit] RUN_TAG=${RUN_TAG}"
echo "[submit] TOTAL_ROWS=${TOTAL_ROWS} NUM_TASKS=${NUM_TASKS} (--array=0-${GEN_ARRAY_MAX}) concurrent=${NUM_CONCURRENT_CASES}"
echo "[submit] MIN_P=${MIN_P} NUM_QE_SHARDS=${NUM_QE_SHARDS} QE_THRESHOLD=${QE_THRESHOLD}"
echo "[submit] generation   : ${GEN_SCRIPT}"
echo "[submit] prepare      : ${PREP_SCRIPT}"
echo "[submit] metricx QE   : ${QE_SCRIPT}"
echo "[submit] finalize     : ${FINALIZE_SCRIPT}"

GEN_SUBMIT=$(sbatch \
  --array=0-${GEN_ARRAY_MAX} \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",NUM_TASKS="${NUM_TASKS}",NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES}",NUM_FUTURES="${NUM_FUTURES}",SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}",MIN_P="${MIN_P}",PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL}",SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL}",INSTRUCT_MODEL="${INSTRUCT_MODEL}",PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL}",SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL}",INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL}" \
  "${GEN_SCRIPT}")
echo "${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

GEN_RUN_DIR="${CONS_OUTPUT}/job_${GEN_JOB_ID}"
METRICX_RUN_DIR="${METRICX_RUN_DIR_BASE}/job_${GEN_JOB_ID}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_BASE}/job_${GEN_JOB_ID}"

echo "[submit] EXPERIMENT_DIR      : ${GEN_RUN_DIR}"
echo "[submit] METRICX_RUN_DIR     : ${METRICX_RUN_DIR}"
echo "[submit] FILTERED_OUTPUT_DIR : ${FILTERED_OUTPUT_DIR}"

PREP_SUBMIT=$(sbatch \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${PREP_SCRIPT}")
echo "${PREP_SUBMIT}"
PREP_JOB_ID=$(echo "${PREP_SUBMIT}" | awk '{print $4}')

QE_SUBMIT=$(sbatch \
  --dependency=afterok:"${PREP_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${QE_SCRIPT}")
echo "${QE_SUBMIT}"
QE_JOB_ID=$(echo "${QE_SUBMIT}" | awk '{print $4}')

FINALIZE_SUBMIT=$(sbatch \
  --dependency=afterok:"${QE_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}",EXPERIMENT_DIR="${GEN_RUN_DIR}",FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR}",QE_THRESHOLD="${QE_THRESHOLD}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${FINALIZE_SCRIPT}")
echo "${FINALIZE_SUBMIT}"
FINALIZE_JOB_ID=$(echo "${FINALIZE_SUBMIT}" | awk '{print $4}')

echo "[submit] generation  job id : ${GEN_JOB_ID}"
echo "[submit] prepare     job id : ${PREP_JOB_ID}"
echo "[submit] metricx     job id : ${QE_JOB_ID}"
echo "[submit] finalize    job id : ${FINALIZE_JOB_ID}"
echo "[submit] experiment dir     : ${GEN_RUN_DIR}"
echo "[submit] metricx dir        : ${METRICX_RUN_DIR}"
echo "[submit] filtered dir       : ${FILTERED_OUTPUT_DIR}"
