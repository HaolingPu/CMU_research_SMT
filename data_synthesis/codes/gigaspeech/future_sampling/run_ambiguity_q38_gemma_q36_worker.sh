#!/usr/bin/env bash
# One 2-GPU decode worker: Qwen3.8 + Gemma samplers on GPU 0, Qwen3.6 translator/probe on GPU 1.
# Task TASK_ID decodes its ceil(TOTAL_ROWS / NUM_TASKS) rows of INPUT_TSV into OUTPUT_ROOT/task_NN.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FS="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
PYTHON="/data/user_data/haolingp/conda_envs/gemma4/bin/python"
SAMPLER_SERVER="${FS}/scripts/qwen38/serve_qwen38_gemma_colocated.sh"
TRANSLATOR_SERVER="${FS}/scripts/qwen36/serve_qwen36_35b.sh"
DECODER="${FS}/consensus_decoding_token_id_level_instruct.py"

QWEN38_MODEL="${QWEN38_MODEL:-/data/user_data/haolingp/models/Qwen3.8-27B-FP8}"
GEMMA_MODEL="${GEMMA_MODEL:-/data/user_data/haolingp/models/gemma-4-E2B-it}"
QWEN36_MODEL="${QWEN36_MODEL:-/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8}"
INPUT_TSV="${INPUT_TSV:?INPUT_TSV required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
TOTAL_ROWS="${TOTAL_ROWS:?TOTAL_ROWS required}"
NUM_TASKS="${NUM_TASKS:?NUM_TASKS required}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-16}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
# Rows the decoder may skip (recorded in verbose/dropped_rows.jsonl) instead of failing the task.
MAX_DROPPED_ROWS="${MAX_DROPPED_ROWS:-0}"
# Per-future next-token probabilities are left out of the verbose logs unless COMPACT_VERBOSE=0.
if [[ "${COMPACT_VERBOSE:-1}" == "0" ]]; then
  COMPACT_VERBOSE_FLAG=""
else
  COMPACT_VERBOSE_FLAG="--compact-verbose"
fi

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
START_ROW=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))
REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
TASK_ROWS=$(( REMAINING < ROWS_PER_TASK ? REMAINING : ROWS_PER_TASK ))
TASK_DIR="${OUTPUT_ROOT}/task_$(printf '%02d' "${TASK_ID}")"
LOG_DIR="${OUTPUT_ROOT}/serve_logs/task_$(printf '%02d' "${TASK_ID}")"
if (( TASK_ROWS <= 0 )); then
  echo "[SKIP] task ${TASK_ID} starts beyond TOTAL_ROWS=${TOTAL_ROWS}"
  exit 0
elif [[ -f "${TASK_DIR}/DONE.txt" ]]; then
  echo "[SKIP] ${TASK_DIR}/DONE.txt exists"
  exit 0
fi

PORT_BASE=$(( 8600 + 4 * TASK_ID ))
QWEN38_PORT=${PORT_BASE}
GEMMA_PORT=$(( PORT_BASE + 1 ))
QWEN36_PORT=$(( PORT_BASE + 2 ))
QWEN38_PID_FILE="/tmp/amb_q38_${SLURM_JOB_ID}_${TASK_ID}.pid"
GEMMA_PID_FILE="/tmp/amb_gemma_${SLURM_JOB_ID}_${TASK_ID}.pid"
QWEN36_PID_FILE="/tmp/amb_q36_${SLURM_JOB_ID}_${TASK_ID}.pid"
export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TASK_DIR}/per_utt" "${TASK_DIR}/verbose" "${LOG_DIR}"

cleanup() {
  set +e
  GPU=0 QWEN_PORT="${QWEN38_PORT}" GEMMA_PORT="${GEMMA_PORT}" \
    QWEN_PID_FILE="${QWEN38_PID_FILE}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" "${SAMPLER_SERVER}" stop
  GPU=1 PORT="${QWEN36_PORT}" PID_FILE="${QWEN36_PID_FILE}" "${TRANSLATOR_SERVER}" stop
}
trap cleanup EXIT INT TERM

wait_health() {
  local name=$1 port=$2 launcher_pid=$3
  for i in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    elif ! kill -0 "${launcher_pid}" 2>/dev/null; then
      echo "[ERROR] ${name} launcher exited before health check passed" >&2
      return 1
    fi
    sleep 1
  done
  echo "[ERROR] ${name} timed out" >&2
  return 1
}

echo "===== consensus decode task ${TASK_ID}: job=${SLURM_JOB_ID} node=$(hostname) rows=${START_ROW}+${TASK_ROWS} cases=${NUM_CONCURRENT_CASES} ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
GPU=0 QWEN_PORT="${QWEN38_PORT}" GEMMA_PORT="${GEMMA_PORT}" QWEN_MODEL="${QWEN38_MODEL}" GEMMA_MODEL="${GEMMA_MODEL}" \
  QWEN_PID_FILE="${QWEN38_PID_FILE}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" LOG_DIR="${LOG_DIR}/samplers" \
  "${SAMPLER_SERVER}" >"${LOG_DIR}/samplers.out" 2>"${LOG_DIR}/samplers.err" &
sampler_launcher_pid=$!
GPU=1 PORT="${QWEN36_PORT}" MODEL="${QWEN36_MODEL}" SERVED_MODEL_NAME=qwen36-translator PID_FILE="${QWEN36_PID_FILE}" \
  MAX_LEN=4096 MAX_NUM_SEQS=64 GPU_MEM_UTIL=0.85 \
  "${TRANSLATOR_SERVER}" >"${LOG_DIR}/translator.out" 2>"${LOG_DIR}/translator.err" &
translator_launcher_pid=$!
wait_health qwen38 "${QWEN38_PORT}" "${sampler_launcher_pid}"
wait_health gemma "${GEMMA_PORT}" "${sampler_launcher_pid}"
wait_health qwen36 "${QWEN36_PORT}" "${translator_launcher_pid}"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

start_ts=$(date +%s)
# shellcheck disable=SC2086  # COMPACT_VERBOSE_FLAG and EXTRA_DECODER_ARGS are word lists
"${PYTHON}" "${DECODER}" \
  --input-tsv "${INPUT_TSV}" --row-idx "${START_ROW}" --max-rows "${TASK_ROWS}" \
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}" --skip-existing --max-dropped-rows "${MAX_DROPPED_ROWS}" \
  --output-dir "${TASK_DIR}/per_utt" --verbose-dir "${TASK_DIR}/verbose" ${COMPACT_VERBOSE_FLAG} \
  --instruct-tokenizer-path "${QWEN36_MODEL}" --instruct-api-base "http://127.0.0.1:${QWEN36_PORT}/v1" \
  --instruct-api-model qwen36-translator \
  --targeted-sampler-tokenizer-path "${GEMMA_MODEL}" --targeted-sampler-api-base "http://127.0.0.1:${GEMMA_PORT}/v1" \
  --targeted-sampler-api-model gemma4-sampler \
  --targeted-sampler2-tokenizer-path "${QWEN38_MODEL}" --targeted-sampler2-api-base "http://127.0.0.1:${QWEN38_PORT}/v1" \
  --targeted-sampler2-api-model qwen38-sampler \
  --targeted-num-futures "${TARGETED_NUM_FUTURES}" --min-voters-ratio "${MIN_VOTERS_RATIO}" \
  ${EXTRA_DECODER_ARGS:-} \
  2>&1 | tee -a "${TASK_DIR}/run.log"

# A requeued task retries rows dropped by an earlier attempt, so a drop only counts while its JSON is missing.
decoded=$(find "${TASK_DIR}/per_utt" -maxdepth 1 -name '*.json' -type f | wc -l)
dropped=0
if [[ -f "${TASK_DIR}/verbose/dropped_rows.jsonl" ]]; then
  dropped=$(grep -o '"utt_id": "[^"]*"' "${TASK_DIR}/verbose/dropped_rows.jsonl" | cut -d'"' -f4 | sort -u \
    | while read -r utt_id; do [[ -f "${TASK_DIR}/per_utt/${utt_id}.json" ]] || echo "${utt_id}"; done | wc -l)
fi
if (( decoded + dropped != TASK_ROWS )); then
  echo "[ERROR] wrote ${decoded}/${TASK_ROWS} outputs (${dropped} dropped)" >&2
  exit 1
elif (( dropped > 0 )); then
  echo "[WARN] ${dropped} row(s) dropped; see ${TASK_DIR}/verbose/dropped_rows.jsonl" >&2
fi
printf 'task=%s rows=%s elapsed_seconds=%s\n' "${TASK_ID}" "${decoded}" "$(( $(date +%s) - start_ts ))" >"${TASK_DIR}/DONE.txt"
echo "[DONE] task ${TASK_ID}: ${decoded} rows"
