#!/usr/bin/env bash
# ============================================================
# 60k consensus-decoding run, top_k=5, + per-sentence QE≤3 + length filter
# Pipeline:
#   [gen]      1000_batch.sbatch    (array=0-3, L40S x2 each, concurrent=32,
#                                     --time=2-00:00:00 override for 60k)
#    -> afterok
#   [prepare]  per-sentence MetricX input converter
#    -> afterok
#   [metricx]  MetricX QE 4-shard array
#    -> afterok
#   [finalize] per-sentence AND filter (keep iff every sentence QE <= 3.0)
#    -> afterok
#   [length]   length_ratio filter (drop if ratio_ref>1.5 or ratio_src>2.5)
# ============================================================
set -e

GEN_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_dualbase_vllm_1000_batch.sbatch"
PREP_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_prepare_per_sentence.sbatch"
QE_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_4gpu.sbatch"
FINALIZE_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_finalize_per_sentence.sbatch"
LEN_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_length_filter.sbatch"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"

# Everything goes under one experiment dir:
#   ${EXP_ROOT}/generation/job_<id>/task_<i>/        ← raw consensus outputs
#   ${EXP_ROOT}/metricx/job_<id>/                    ← MetricX input/output/report
#   ${EXP_ROOT}/filtered-qe3/job_<id>/               ← per-sentence QE<=3 survivors
#   ${EXP_ROOT}/filtered-qe3-len/job_<id>/           ← + length-ratio filter survivors
EXP_ROOT="${EXP_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topk/consensus_decoding_en_zh_top_5_v2}"
CONS_OUTPUT="${CONS_OUTPUT:-${EXP_ROOT}/generation}"
METRICX_RUN_DIR_BASE="${METRICX_RUN_DIR_BASE:-${EXP_ROOT}/metricx}"
FILTERED_OUTPUT_BASE="${FILTERED_OUTPUT_BASE:-${EXP_ROOT}/filtered-qe3}"
LEN_FILTERED_BASE="${LEN_FILTERED_BASE:-${EXP_ROOT}/filtered-qe3-len}"

TOTAL_ROWS="${TOTAL_ROWS:-60000}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-48}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MIN_P="${MIN_P:-0.0}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-5}"
TOP_P="${TOP_P:-0.0}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-12}"
FUTURE_SOURCE_WINDOW_CHUNKS="${FUTURE_SOURCE_WINDOW_CHUNKS:-3}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-8}"
ROW_OFFSET="${ROW_OFFSET:-0}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
MAX_RATIO_SRC="${MAX_RATIO_SRC:-2.5}"
GEN_TIME_LIMIT="${GEN_TIME_LIMIT:-2-00:00:00}"

PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL:-google/gemma-4-E2B}"
SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL:-0.45}"
SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL:-0.45}"
INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL:-0.90}"

GEN_ARRAY_MAX=$(( NUM_TASKS - 1 ))
QE_ARRAY_MAX=$(( NUM_QE_SHARDS - 1 ))

echo "[submit] EXP_ROOT             : ${EXP_ROOT}"
echo "[submit]   generation         : ${CONS_OUTPUT}"
echo "[submit]   metricx            : ${METRICX_RUN_DIR_BASE}"
echo "[submit]   filtered-qe3       : ${FILTERED_OUTPUT_BASE}"
echo "[submit]   filtered-qe3-len   : ${LEN_FILTERED_BASE}"
echo "[submit] TOTAL_ROWS=${TOTAL_ROWS} NUM_TASKS=${NUM_TASKS} (--array=0-${GEN_ARRAY_MAX}) concurrent=${NUM_CONCURRENT_CASES}"
echo "[submit] CANDIDATE_TOP_K=${CANDIDATE_TOP_K} TOP_P=${TOP_P} MIN_P=${MIN_P} QE_THRESHOLD=${QE_THRESHOLD}"
echo "[submit] MAX_CONSENSUS_STEPS=${MAX_CONSENSUS_STEPS} FUTURE_SOURCE_WINDOW_CHUNKS=${FUTURE_SOURCE_WINDOW_CHUNKS}"
echo "[submit] MAX_RATIO_REF=${MAX_RATIO_REF} MAX_RATIO_SRC=${MAX_RATIO_SRC}"
echo "[submit] GEN_TIME_LIMIT=${GEN_TIME_LIMIT}"

# ---- [1] generation ----
GEN_SUBMIT=$(sbatch \
  --array=0-${GEN_ARRAY_MAX} \
  --time="${GEN_TIME_LIMIT}" \
  --export=ALL,INPUT_TSV="${INPUT_TSV}",OUTPUT_ROOT="${CONS_OUTPUT}",TOTAL_ROWS="${TOTAL_ROWS}",ROW_OFFSET="${ROW_OFFSET}",NUM_TASKS="${NUM_TASKS}",NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES}",NUM_FUTURES="${NUM_FUTURES}",SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES}",FUTURE_TOKENS="${FUTURE_TOKENS}",MIN_P="${MIN_P}",CANDIDATE_TOP_K="${CANDIDATE_TOP_K}",TOP_P="${TOP_P}",MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS}",FUTURE_SOURCE_WINDOW_CHUNKS="${FUTURE_SOURCE_WINDOW_CHUNKS}",PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL}",SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL}",INSTRUCT_MODEL="${INSTRUCT_MODEL}",PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL}",SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL}",INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL}" \
  "${GEN_SCRIPT}")
echo "${GEN_SUBMIT}"
GEN_JOB_ID=$(echo "${GEN_SUBMIT}" | awk '{print $4}')

GEN_RUN_DIR="${CONS_OUTPUT}/job_${GEN_JOB_ID}"
METRICX_RUN_DIR="${METRICX_RUN_DIR_BASE}/job_${GEN_JOB_ID}"
QE_FILTERED_DIR="${FILTERED_OUTPUT_BASE}/job_${GEN_JOB_ID}"
LEN_FILTERED_DIR="${LEN_FILTERED_BASE}/job_${GEN_JOB_ID}"

echo "[submit] EXPERIMENT_DIR   : ${GEN_RUN_DIR}"
echo "[submit] METRICX_RUN_DIR  : ${METRICX_RUN_DIR}"
echo "[submit] QE_FILTERED_DIR  : ${QE_FILTERED_DIR}"
echo "[submit] LEN_FILTERED_DIR : ${LEN_FILTERED_DIR}"

# ---- [2] prepare ----
PREP_SUBMIT=$(sbatch \
  --dependency=afterok:"${GEN_JOB_ID}" \
  --export=ALL,EXPERIMENT_DIR="${GEN_RUN_DIR}",METRICX_RUN_DIR="${METRICX_RUN_DIR}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${PREP_SCRIPT}")
echo "${PREP_SUBMIT}"
PREP_JOB_ID=$(echo "${PREP_SUBMIT}" | awk '{print $4}')

# ---- [3] metricx QE ----
QE_SUBMIT=$(sbatch \
  --dependency=afterok:"${PREP_JOB_ID}" \
  --array=0-${QE_ARRAY_MAX}%${NUM_QE_SHARDS} \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}" \
  "${QE_SCRIPT}")
echo "${QE_SUBMIT}"
QE_JOB_ID=$(echo "${QE_SUBMIT}" | awk '{print $4}')

# ---- [4] per-sentence AND finalize ----
FINALIZE_SUBMIT=$(sbatch \
  --dependency=afterok:"${QE_JOB_ID}" \
  --export=ALL,METRICX_RUN_DIR="${METRICX_RUN_DIR}",EXPERIMENT_DIR="${GEN_RUN_DIR}",FILTERED_OUTPUT_DIR="${QE_FILTERED_DIR}",QE_THRESHOLD="${QE_THRESHOLD}",NUM_SHARDS="${NUM_QE_SHARDS}" \
  "${FINALIZE_SCRIPT}")
echo "${FINALIZE_SUBMIT}"
FINALIZE_JOB_ID=$(echo "${FINALIZE_SUBMIT}" | awk '{print $4}')

# ---- [5] length filter ----
LEN_SUBMIT=$(sbatch \
  --dependency=afterok:"${FINALIZE_JOB_ID}" \
  --export=ALL,QE_FILTERED_DIR="${QE_FILTERED_DIR}",LEN_FILTERED_DIR="${LEN_FILTERED_DIR}",MAX_RATIO_REF="${MAX_RATIO_REF}",MAX_RATIO_SRC="${MAX_RATIO_SRC}" \
  "${LEN_SCRIPT}")
echo "${LEN_SUBMIT}"
LEN_JOB_ID=$(echo "${LEN_SUBMIT}" | awk '{print $4}')

echo "[submit] generation  job id : ${GEN_JOB_ID}"
echo "[submit] prepare     job id : ${PREP_JOB_ID}"
echo "[submit] metricx     job id : ${QE_JOB_ID}"
echo "[submit] finalize    job id : ${FINALIZE_JOB_ID}"
echo "[submit] length      job id : ${LEN_JOB_ID}"
echo "[submit] experiment dir     : ${GEN_RUN_DIR}"
echo "[submit] metricx dir        : ${METRICX_RUN_DIR}"
echo "[submit] qe-filtered dir    : ${QE_FILTERED_DIR}"
echo "[submit] len-filtered dir   : ${LEN_FILTERED_DIR}"
