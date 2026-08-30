#!/usr/bin/env bash
# One 2-GPU decode worker: Qwen3.8 + Gemma samplers on GPU 0, Qwen3.6 probe on GPU 1.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
REPO="/home/haolingp/CMU_research_SMT"
FS="${REPO}/data_synthesis/codes/gigaspeech/future_sampling"
ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${ENV}/bin/python"

SAMPLER_SERVER="${FS}/scripts/qwen38/serve_qwen38_gemma_colocated.sh"
TRANSLATOR_SERVER="${FS}/scripts/qwen36/serve_qwen36_35b.sh"
DECODER="${FS}/consensus_decoding_token_id_level_instruct.py"

QWEN38_MODEL="${QWEN38_MODEL:-/data/user_data/haolingp/models/Qwen3.8-27B-FP8}"
GEMMA_MODEL="${GEMMA_MODEL:-/data/user_data/haolingp/models/gemma-4-E2B-it}"
QWEN36_MODEL="${QWEN36_MODEL:-/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8}"

INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
TOTAL_ROWS="${TOTAL_ROWS:-40000}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_TASKS="${NUM_TASKS:-12}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-4}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
FUTURE_SRC_WINDOW="${FUTURE_SRC_WINDOW:-1}"

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
START_ROW=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))
REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
if (( REMAINING <= 0 )); then
  echo "[SKIP] task ${TASK_ID} starts beyond TOTAL_ROWS=${TOTAL_ROWS}"
  exit 0
fi
if (( ROWS_PER_TASK > REMAINING )); then
  ROWS_PER_TASK=${REMAINING}
fi

TASK_DIR="${OUTPUT_ROOT}/task_$(printf '%02d' "${TASK_ID}")"
LOG_DIR="${OUTPUT_ROOT}/serve_logs/task_$(printf '%02d' "${TASK_ID}")"
DONE_FILE="${TASK_DIR}/DONE.txt"
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

if [[ -f "${DONE_FILE}" ]]; then
  echo "[SKIP] ${DONE_FILE} exists"
  exit 0
fi

cleanup() {
  set +e
  GPU=0 QWEN_PORT="${QWEN38_PORT}" GEMMA_PORT="${GEMMA_PORT}" \
    QWEN_PID_FILE="${QWEN38_PID_FILE}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" \
    "${SAMPLER_SERVER}" stop
  GPU=1 PORT="${QWEN36_PORT}" PID_FILE="${QWEN36_PID_FILE}" \
    "${TRANSLATOR_SERVER}" stop
}
trap cleanup EXIT INT TERM

wait_health() {
  local name=$1 port=$2 launcher_pid=$3
  for i in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    fi
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
      echo "[ERROR] ${name} launcher exited before health check passed" >&2
      return 1
    fi
    sleep 1
  done
  echo "[ERROR] ${name} timed out" >&2
  return 1
}

echo "===== ambiguity decode task ${TASK_ID} ====="
echo "job=${SLURM_JOB_ID} node=$(hostname) rows=${START_ROW}+${ROWS_PER_TASK}"
echo "samplers: qwen38=${QWEN38_PORT} gemma=${GEMMA_PORT}; translator: qwen36=${QWEN36_PORT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

GPU=0 QWEN_PORT="${QWEN38_PORT}" GEMMA_PORT="${GEMMA_PORT}" \
QWEN_MODEL="${QWEN38_MODEL}" GEMMA_MODEL="${GEMMA_MODEL}" \
QWEN_PID_FILE="${QWEN38_PID_FILE}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" \
LOG_DIR="${LOG_DIR}/samplers" "${SAMPLER_SERVER}" >"${LOG_DIR}/samplers.out" 2>"${LOG_DIR}/samplers.err" &
sampler_launcher_pid=$!

GPU=1 PORT="${QWEN36_PORT}" MODEL="${QWEN36_MODEL}" \
SERVED_MODEL_NAME=qwen36-translator PID_FILE="${QWEN36_PID_FILE}" \
MAX_LEN=4096 MAX_NUM_SEQS=64 GPU_MEM_UTIL=0.85 \
"${TRANSLATOR_SERVER}" >"${LOG_DIR}/translator.out" 2>"${LOG_DIR}/translator.err" &
translator_launcher_pid=$!

wait_health qwen38 "${QWEN38_PORT}" "${sampler_launcher_pid}"
wait_health gemma "${GEMMA_PORT}" "${sampler_launcher_pid}"
wait_health qwen36 "${QWEN36_PORT}" "${translator_launcher_pid}"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

start_ts=$(date +%s)
"${PYTHON}" "${DECODER}" \
  --input-tsv "${INPUT_TSV}" \
  --instruct-tokenizer-path "${QWEN36_MODEL}" \
  --instruct-api-base "http://127.0.0.1:${QWEN36_PORT}/v1" \
  --instruct-api-model qwen36-translator \
  --use-targeted-instruct-sampling \
  --targeted-sampler-api-base "http://127.0.0.1:${GEMMA_PORT}/v1" \
  --targeted-sampler-api-model gemma4-sampler \
  --targeted-sampler-tokenizer-path "${GEMMA_MODEL}" \
  --targeted-sampler2-api-base "http://127.0.0.1:${QWEN38_PORT}/v1" \
  --targeted-sampler2-api-model qwen38-sampler \
  --targeted-sampler2-tokenizer-path "${QWEN38_MODEL}" \
  --targeted-num-futures "${TARGETED_NUM_FUTURES}" \
  --future-source-window-chunks "${FUTURE_SRC_WINDOW}" \
  --min-voters-ratio "${MIN_VOTERS_RATIO}" \
  --row-idx "${START_ROW}" \
  --max-rows "${ROWS_PER_TASK}" \
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}" \
  --skip-existing \
  --output-jsonl "${TASK_DIR}/per_utt/_.jsonl" \
  --verbose --verbose-dir "${TASK_DIR}/verbose" \
  2>&1 | tee -a "${TASK_DIR}/run.log"

actual=$(find "${TASK_DIR}/per_utt" -maxdepth 1 -name '*.json' -type f | wc -l)
if (( actual != ROWS_PER_TASK )); then
  echo "[ERROR] wrote ${actual}/${ROWS_PER_TASK} outputs" >&2
  exit 1
fi
printf 'task=%s rows=%s elapsed_seconds=%s prompt=ambiguity_icl_v1\n' \
  "${TASK_ID}" "${actual}" "$(( $(date +%s) - start_ts ))" >"${DONE_FILE}"
echo "[DONE] task ${TASK_ID}: ${actual} rows"
