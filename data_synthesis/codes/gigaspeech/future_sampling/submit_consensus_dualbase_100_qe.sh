#!/usr/bin/env bash
set -euo pipefail

GEN_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_dualbase_vllm_100_8gpu.sbatch"
PREP_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_prepare_8shards.sbatch"
QE_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_4gpu.sbatch"
FINALIZE_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_finalize.sbatch"
SUMMARY_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_metrics_summary.sbatch"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
RUN_TAG="${RUN_TAG:-consensus-dualbase-100-qe}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug}"
CONS_OUTPUT="${CONS_OUTPUT:-${OUTPUT_BASE}/${RUN_TAG}}"
METRICX_RUN_DIR_BASE="${METRICX_RUN_DIR_BASE:-${OUTPUT_BASE}/${RUN_TAG}-metricx}"

TOTAL_ROWS="${TOTAL_ROWS:-100}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-4}"

PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL:-google/gemma-4-E2B}"
SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL:-0.45}"
SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL:-0.45}"
INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL:-0.90}"

echo "[submit] generation script: ${GEN_SCRIPT}"
GEN_SUBMIT=$(sbatch \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",NUM_TASKS="${NUM_TASKS}",NUM_FUTURES="${NUM_FUTURES}",SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}",PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL}",SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL}",INSTRUCT_MODEL="${INSTRUCT_MODEL}",PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL}",SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL}",INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL}" \
  "${GEN_SCRIPT}")
echo "${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

GEN_RUN_DIR="${CONS_OUTPUT}/job_${GEN_JOB_ID}"
METRICX_RUN_DIR="${METRICX_RUN_DIR_BASE}/job_${GEN_JOB_ID}"
SUMMARY_TXT="${METRICX_RUN_DIR}/overall_metrics.txt"

echo "[submit] metricx prep script: ${PREP_SCRIPT}"
PREP_SUBMIT=$(sbatch \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",NUM_SHARDS="${NUM_QE_SHARDS}" \
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
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",SUMMARY_TXT="${SUMMARY_TXT}" \
  "${SUMMARY_SCRIPT}")
echo "${SUMMARY_SUBMIT}"
SUMMARY_JOB_ID=$(echo "${SUMMARY_SUBMIT}" | awk '{print $4}')

echo "[submit] generation job id : ${GEN_JOB_ID}"
echo "[submit] metricx prep job id: ${PREP_JOB_ID}"
echo "[submit] metricx array job id: ${QE_JOB_ID}"
echo "[submit] metricx final job id: ${FINALIZE_JOB_ID}"
echo "[submit] summary job id    : ${SUMMARY_JOB_ID}"
echo "[submit] experiment dir    : ${GEN_RUN_DIR}"
echo "[submit] metricx dir       : ${METRICX_RUN_DIR}"
echo "[submit] summary txt       : ${SUMMARY_TXT}"
