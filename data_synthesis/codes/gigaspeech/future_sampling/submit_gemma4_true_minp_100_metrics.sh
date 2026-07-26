#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# One-click pipeline: 100-case dual-base consensus decoding with true min-p
# using the user's validated 3-serve vLLM setup.
# =============================================================================

GEN_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_dualbase_served_100.sbatch"
PREP_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_prepare_8shards.sbatch"
QE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_4gpu.sbatch"
FINALIZE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_finalize.sbatch"
SUMMARY_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_metrics_summary.sbatch"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/gemma4}"
MIN_P="${MIN_P:-0.05}"
MIN_P_TAG="$(printf '%s' "${MIN_P}" | tr '.' 'p')"
RUN_TAG="${RUN_TAG:-true-minp-${MIN_P_TAG}-100-metrics}"

CONS_OUTPUT="${CONS_OUTPUT:-${OUTPUT_BASE}/${RUN_TAG}}"
METRICX_RUN_DIR_BASE="${METRICX_RUN_DIR_BASE:-${OUTPUT_BASE}/${RUN_TAG}-metricx}"

TOTAL_ROWS="${TOTAL_ROWS:-100}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-4}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-8}"
GEN_DEPENDENCY="${GEN_DEPENDENCY:-}"

PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL:-google/gemma-4-E2B}"
SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL:-0.45}"
SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL:-0.45}"

mkdir -p "${OUTPUT_BASE}"

echo "[pipeline] min_p              : ${MIN_P}"
echo "[pipeline] run tag            : ${RUN_TAG}"
echo "[pipeline] output             : ${CONS_OUTPUT}"
echo "[pipeline] total rows         : ${TOTAL_ROWS}"
echo "[pipeline] generation tasks   : ${NUM_TASKS}"
echo "[pipeline] instruct backend   : user_serves"

GEN_SBATCH_ARGS=()
if [[ -n "${GEN_DEPENDENCY}" ]]; then
  GEN_SBATCH_ARGS+=(--dependency=afterok:"${GEN_DEPENDENCY}")
fi
GEN_SUBMIT=$(sbatch \
  "${GEN_SBATCH_ARGS[@]}" \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",NUM_TASKS="${NUM_TASKS}",NUM_FUTURES="${NUM_FUTURES}",SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}",NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES}",MIN_P="${MIN_P}" \
  "${GEN_SCRIPT}")
echo "${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

GEN_RUN_DIR="${CONS_OUTPUT}/job_${GEN_JOB_ID}"
METRICX_RUN_DIR="${METRICX_RUN_DIR_BASE}/job_${GEN_JOB_ID}"
SUMMARY_TXT="${METRICX_RUN_DIR}/overall_metrics.txt"

PREP_SUBMIT=$(sbatch \
  --mail-type=NONE \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${PREP_SCRIPT}")
echo "${PREP_SUBMIT}"
PREP_JOB_ID=$(echo "${PREP_SUBMIT}" | awk '{print $4}')

QE_SUBMIT=$(sbatch \
  --mail-type=NONE \
  --dependency=afterok:"${PREP_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${QE_SCRIPT}")
echo "${QE_SUBMIT}"
QE_JOB_ID=$(echo "${QE_SUBMIT}" | awk '{print $4}')

FINALIZE_SUBMIT=$(sbatch \
  --mail-type=NONE \
  --dependency=afterok:"${QE_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${FINALIZE_SCRIPT}")
echo "${FINALIZE_SUBMIT}"
FINALIZE_JOB_ID=$(echo "${FINALIZE_SUBMIT}" | awk '{print $4}')

SUMMARY_SUBMIT=$(sbatch \
  --mail-type=END,FAIL \
  --dependency=afterok:"${FINALIZE_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",SUMMARY_TXT="${SUMMARY_TXT}" \
  "${SUMMARY_SCRIPT}")
echo "${SUMMARY_SUBMIT}"
SUMMARY_JOB_ID=$(echo "${SUMMARY_SUBMIT}" | awk '{print $4}')

echo
echo "[done] generation job id : ${GEN_JOB_ID}"
echo "[done] prep job id       : ${PREP_JOB_ID}"
echo "[done] qe array job id   : ${QE_JOB_ID}"
echo "[done] finalize job id   : ${FINALIZE_JOB_ID}"
echo "[done] summary job id    : ${SUMMARY_JOB_ID}"
echo "[done] experiment dir    : ${GEN_RUN_DIR}"
echo "[done] metricx dir       : ${METRICX_RUN_DIR}"
echo "[done] final summary     : ${SUMMARY_TXT}"
