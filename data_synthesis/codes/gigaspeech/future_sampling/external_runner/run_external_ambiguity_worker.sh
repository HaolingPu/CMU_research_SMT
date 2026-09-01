#!/usr/bin/env bash
# Portable 2-GPU worker for a disjoint slice of the ambiguity-consensus decode.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FS_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "${FS_DIR}/../../../.." && pwd)}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
JOB_ID="${SLURM_JOB_ID:-manual}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-$(dirname "${PYTHON_BIN}")/vllm}"
GEMMA4_ENV="${GEMMA4_ENV:-$(cd -- "$(dirname "${PYTHON_BIN}")/.." && pwd)}"
INPUT_TSV="${INPUT_TSV:?Set INPUT_TSV to the frozen 40K input TSV}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a shared output directory}"
QWEN38_MODEL="${QWEN38_MODEL:?Set QWEN38_MODEL to Qwen3.8-27B-FP8}"
GEMMA_MODEL="${GEMMA_MODEL:?Set GEMMA_MODEL to gemma-4-E2B-it}"
QWEN36_MODEL="${QWEN36_MODEL:?Set QWEN36_MODEL to Qwen3.6-35B-A3B-FP8}"

# 20,004 is the boundary between BABEL's original task 5 and task 6.
ROW_OFFSET="${ROW_OFFSET:-20004}"
SLICE_ROWS="${SLICE_ROWS:-19996}"
NUM_TASKS="${NUM_TASKS:-8}"
OUTPUT_TASK_OFFSET="${OUTPUT_TASK_OFFSET:-100}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-12}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
FUTURE_SRC_WINDOW="${FUTURE_SRC_WINDOW:-1}"
ID_COLUMN="${ID_COLUMN:-id}"
PORT_BASE="${PORT_BASE:-8600}"

SAMPLER_SERVER="${FS_DIR}/scripts/qwen38/serve_qwen38_gemma_colocated.sh"
TRANSLATOR_SERVER="${FS_DIR}/scripts/qwen36/serve_qwen36_35b.sh"
DECODER="${FS_DIR}/consensus_decoding_token_id_level_instruct.py"

for path in "${PYTHON_BIN}" "${VLLM_BIN}" "${INPUT_TSV}" \
  "${QWEN38_MODEL}/config.json" "${GEMMA_MODEL}/config.json" \
  "${QWEN36_MODEL}/config.json"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing ${path}" >&2; exit 2; }
done
if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
  echo "[ERROR] TASK_ID=${TASK_ID} outside [0, $((NUM_TASKS - 1))]" >&2
  exit 2
fi

ROWS_PER_TASK=$(( (SLICE_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
SLICE_LOCAL_START=$(( TASK_ID * ROWS_PER_TASK ))
START_ROW=$(( ROW_OFFSET + SLICE_LOCAL_START ))
REMAINING=$(( SLICE_ROWS - SLICE_LOCAL_START ))
if (( REMAINING <= 0 )); then
  echo "[SKIP] task ${TASK_ID} starts beyond SLICE_ROWS=${SLICE_ROWS}"
  exit 0
fi
if (( ROWS_PER_TASK > REMAINING )); then
  ROWS_PER_TASK=${REMAINING}
fi

OUTPUT_TASK_ID=$(( OUTPUT_TASK_OFFSET + TASK_ID ))
TASK_LABEL=$(printf '%03d' "${OUTPUT_TASK_ID}")
TASK_DIR="${OUTPUT_ROOT}/task_${TASK_LABEL}"
LOG_DIR="${OUTPUT_ROOT}/serve_logs/task_${TASK_LABEL}"
DONE_FILE="${TASK_DIR}/DONE.txt"
TASK_PORT_BASE=$(( PORT_BASE + 4 * TASK_ID ))
QWEN38_PORT=${TASK_PORT_BASE}
GEMMA_PORT=$(( TASK_PORT_BASE + 1 ))
QWEN36_PORT=$(( TASK_PORT_BASE + 2 ))
QWEN38_PID_FILE="/tmp/ext_q38_${JOB_ID}_${TASK_ID}.pid"
GEMMA_PID_FILE="/tmp/ext_gemma_${JOB_ID}_${TASK_ID}.pid"
QWEN36_PID_FILE="/tmp/ext_q36_${JOB_ID}_${TASK_ID}.pid"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
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
    GEMMA4_ENV="${GEMMA4_ENV}" VLLM_BIN="${VLLM_BIN}" \
    "${SAMPLER_SERVER}" stop
  GPU=1 PORT="${QWEN36_PORT}" PID_FILE="${QWEN36_PID_FILE}" \
    GEMMA4_ENV="${GEMMA4_ENV}" VLLM_BIN="${VLLM_BIN}" \
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

echo "===== external ambiguity decode task ${TASK_ID} ====="
echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "job=${JOB_ID} node=$(hostname) rows=${START_ROW}+${ROWS_PER_TASK}"
echo "output_task=task_${TASK_LABEL} concurrent_cases=${NUM_CONCURRENT_CASES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

GPU=0 QWEN_PORT="${QWEN38_PORT}" GEMMA_PORT="${GEMMA_PORT}" \
QWEN_MODEL="${QWEN38_MODEL}" GEMMA_MODEL="${GEMMA_MODEL}" \
QWEN_PID_FILE="${QWEN38_PID_FILE}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" \
LOG_DIR="${LOG_DIR}/samplers" GEMMA4_ENV="${GEMMA4_ENV}" VLLM_BIN="${VLLM_BIN}" \
HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
"${SAMPLER_SERVER}" >"${LOG_DIR}/samplers.out" 2>"${LOG_DIR}/samplers.err" &
sampler_launcher_pid=$!

GPU=1 PORT="${QWEN36_PORT}" MODEL="${QWEN36_MODEL}" \
SERVED_MODEL_NAME=qwen36-translator PID_FILE="${QWEN36_PID_FILE}" \
MAX_LEN=4096 MAX_NUM_SEQS=64 GPU_MEM_UTIL=0.85 \
GEMMA4_ENV="${GEMMA4_ENV}" VLLM_BIN="${VLLM_BIN}" \
HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
"${TRANSLATOR_SERVER}" >"${LOG_DIR}/translator.out" 2>"${LOG_DIR}/translator.err" &
translator_launcher_pid=$!

wait_health qwen38 "${QWEN38_PORT}" "${sampler_launcher_pid}"
wait_health gemma "${GEMMA_PORT}" "${sampler_launcher_pid}"
wait_health qwen36 "${QWEN36_PORT}" "${translator_launcher_pid}"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

start_ts=$(date +%s)
"${PYTHON_BIN}" "${DECODER}" \
  --input-tsv "${INPUT_TSV}" \
  --id-column "${ID_COLUMN}" \
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
  --skip-existing --skip-existing-root "${OUTPUT_ROOT}" \
  --output-jsonl "${TASK_DIR}/per_utt/_.jsonl" \
  --verbose --compact-verbose --verbose-dir "${TASK_DIR}/verbose" \
  2>&1 | tee -a "${TASK_DIR}/run.log"

actual=$(find "${TASK_DIR}/per_utt" -maxdepth 1 -name '*.json' -type f | wc -l | tr -d ' ')
if (( actual != ROWS_PER_TASK )); then
  echo "[ERROR] task contains ${actual}/${ROWS_PER_TASK} outputs" >&2
  exit 1
fi
printf 'task=%s output_task=%s start_row=%s rows=%s elapsed_seconds=%s prompt=future_set_v2_two_groups\n' \
  "${TASK_ID}" "${OUTPUT_TASK_ID}" "${START_ROW}" "${actual}" \
  "$(( $(date +%s) - start_ts ))" >"${DONE_FILE}"
echo "[DONE] task ${TASK_ID}: ${actual} rows"
