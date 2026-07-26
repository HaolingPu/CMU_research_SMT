#!/usr/bin/env bash
set -e

GEN_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_vllm_100_8gpu.sbatch"
PREP_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_prepare_8shards.sbatch"
QE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_8gpu.sbatch"
FINALIZE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_finalize.sbatch"
SUMMARY_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_metrics_summary.sbatch"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
RUN_TAG="${RUN_TAG:-consensus-100-qe}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug}"

CONS_OUTPUT="${CONS_OUTPUT:-${OUTPUT_BASE}/${RUN_TAG}}"
METRICX_RUN_DIR="${METRICX_RUN_DIR:-${OUTPUT_BASE}/${RUN_TAG}-metricx}"
SUMMARY_TXT="${SUMMARY_TXT:-${METRICX_RUN_DIR}/overall_metrics.txt}"

TOTAL_ROWS="${TOTAL_ROWS:-100}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
FUTURE_TOKENS="${FUTURE_TOKENS:-24}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-8}"

echo "[submit] generation script: ${GEN_SCRIPT}"
GEN_SUBMIT=$(sbatch \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",NUM_TASKS="${NUM_TASKS}",NUM_FUTURES="${NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}" \
  "${GEN_SCRIPT}")
echo "${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

echo "[submit] metricx prep script: ${PREP_SCRIPT}"
PREP_SUBMIT=$(sbatch \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${CONS_OUTPUT}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${PREP_SCRIPT}")
echo "${PREP_SUBMIT}"
PREP_JOB_ID=$(echo "${PREP_SUBMIT}" | awk '{print $4}')

echo "[submit] metricx array script: ${QE_SCRIPT}"
QE_SUBMIT=$(sbatch \
  --dependency=afterok:"${PREP_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${QE_SCRIPT}")
echo "${QE_SUBMIT}"
QE_JOB_ID=$(echo "${QE_SUBMIT}" | awk '{print $4}')

echo "[submit] metricx finalize script: ${FINALIZE_SCRIPT}"
FINALIZE_SUBMIT=$(sbatch \
  --dependency=afterok:"${QE_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${FINALIZE_SCRIPT}")
echo "${FINALIZE_SUBMIT}"
FINALIZE_JOB_ID=$(echo "${FINALIZE_SUBMIT}" | awk '{print $4}')

echo "[submit] summary script: ${SUMMARY_SCRIPT}"
SUMMARY_SUBMIT=$(sbatch \
  --dependency=afterok:"${FINALIZE_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${CONS_OUTPUT}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",SUMMARY_TXT="${SUMMARY_TXT}" \
  "${SUMMARY_SCRIPT}")
echo "${SUMMARY_SUBMIT}"
SUMMARY_JOB_ID=$(echo "${SUMMARY_SUBMIT}" | awk '{print $4}')

echo "[submit] generation job id : ${GEN_JOB_ID}"
echo "[submit] metricx prep job id: ${PREP_JOB_ID}"
echo "[submit] metricx array job id: ${QE_JOB_ID}"
echo "[submit] metricx final job id: ${FINALIZE_JOB_ID}"
echo "[submit] summary job id    : ${SUMMARY_JOB_ID}"
echo "[submit] experiment dir    : ${CONS_OUTPUT}"
echo "[submit] metricx dir       : ${METRICX_RUN_DIR}"
echo "[submit] summary txt       : ${SUMMARY_TXT}"
