#!/usr/bin/env bash
set -e

GEN_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_vllm_2500_8gpu.sbatch"
POST_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_filter_consensus_qe3.sbatch"

INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered_zh.tsv}"
CONS_OUTPUT="${CONS_OUTPUT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-2k}"
METRICX_RUN_DIR="${METRICX_RUN_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-2k-metricx}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-2k-qe3}"

TOTAL_ROWS="${TOTAL_ROWS:-2500}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
FUTURE_TOKENS="${FUTURE_TOKENS:-24}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"

echo "[submit] generation script: ${GEN_SCRIPT}"
GEN_SUBMIT=$(sbatch \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",NUM_TASKS="${NUM_TASKS}",NUM_FUTURES="${NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}" \
  "${GEN_SCRIPT}")
echo "${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

echo "[submit] postprocess script: ${POST_SCRIPT}"
POST_SUBMIT=$(sbatch \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${CONS_OUTPUT}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR}",QE_THRESHOLD="${QE_THRESHOLD}" \
  "${POST_SCRIPT}")
echo "${POST_SUBMIT}"
POST_JOB_ID=$(echo "${POST_SUBMIT}" | awk '{print $4}')

echo "[submit] generation job id : ${GEN_JOB_ID}"
echo "[submit] postprocess job id: ${POST_JOB_ID}"
