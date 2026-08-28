#!/usr/bin/env bash
set -e

# Shared runner for wait-k 50k experiments on general (array=0-7, 1 GPU each).
# Usage: bash run_waitk_common.sh <K> <LABEL>   e.g. run_waitk_common.sh 3 k3

K_VALUE="${1:?Usage: bash run_waitk_common.sh <K> <LABEL>}"
K_LABEL="${2:?Usage: bash run_waitk_common.sh <K> <LABEL>}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${OUTPUT_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"

WAITK_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/wait-k"
FS_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${FS_DIR}/serve_instruct_gpu0.sh"
DECODER="${WAITK_DIR}/wait_k.py"
TOKENIZER="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/wait-k/waitk_${K_LABEL}_50k_general"

TOTAL_ROWS=50000
ROW_OFFSET=0
NUM_TASKS=8
MAX_NEW_TOKENS=1024
NUM_CONCURRENT_CASES=68
TARGET_LANG=Chinese
SKIP_EXISTING=1

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
ROW_START=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))
ROW_COUNT="${ROWS_PER_TASK}"
REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
if (( REMAINING <= 0 )); then
  echo "[SKIP] task ${TASK_ID} has no assigned rows"
  exit 0
fi
if (( REMAINING < ROW_COUNT )); then
  ROW_COUNT="${REMAINING}"
fi

# Wait-k uses different port base than LA (8300) to avoid collisions.
PORT_BASE=$(( 8500 + TASK_ID * 10 ))
INSTRUCT_PORT="${PORT_BASE}"

RUN_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/task_${TASK_ID}"
LOG_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/serve_logs"
DONE_FILE="${RUN_DIR}/DONE.txt"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

INSTRUCT_PID="/tmp/waitk_${K_LABEL}_instruct_${JOB_ID}_${TASK_ID}.pid"

stop_servers() {
  set +e
  PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_instruct_${TASK_ID}.log" 2>&1
}
trap stop_servers EXIT

wait_health() {
  name="$1"
  url="$2"
  for i in $(seq 1 300); do
    if curl -s "${url}" > /dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] ${name} not ready: ${url}"
  return 1
}

if [[ -f "${DONE_FILE}" ]]; then
  echo "[SKIP] ${DONE_FILE} exists"
  exit 0
fi

echo "===== wait-k=${K_VALUE} task ${TASK_ID} ====="
echo "job=${JOB_ID} node=$(hostname)"
echo "rows: start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS}"
echo "output=${RUN_DIR}"

PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &

wait_health "instruct" "http://127.0.0.1:${INSTRUCT_PORT}/health"

CMD=(
  "${PYTHON}" "${DECODER}"
  --input-tsv "${INPUT_TSV}"
  --mt-api-base "http://127.0.0.1:${INSTRUCT_PORT}/v1"
  --mt-api-model "qwen3-instruct"
  --mt-tokenizer-path "${TOKENIZER}"
  --target-lang "${TARGET_LANG}"
  --row-idx "${ROW_START}"
  --max-rows "${ROW_COUNT}"
  --wait-k "${K_VALUE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}"
  --output-dir "${RUN_DIR}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip-existing)
fi

"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
CMD_STATUS="${PIPESTATUS[0]}"
if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] wait-k failed with exit code ${CMD_STATUS}"
  exit "${CMD_STATUS}"
fi

JSON_COUNT=$(find "${RUN_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
# Allow up to 1% of utterances to be written as error JSONs (context-length
# etc). Downstream QE skips error JSONs anyway.
MIN_ACCEPTABLE=$(( ROW_COUNT * 99 / 100 ))
if (( JSON_COUNT < MIN_ACCEPTABLE )); then
  echo "[ERROR] incomplete task output: json_count=${JSON_COUNT}, expected≈${ROW_COUNT}, min=${MIN_ACCEPTABLE}"
  exit 1
fi
echo "[OK] json_count=${JSON_COUNT} (expected=${ROW_COUNT}, min=${MIN_ACCEPTABLE})"

{
  echo "completed_at=$(date)"
  echo "rows_start=${ROW_START}"
  echo "rows_count=${ROW_COUNT}"
  echo "wait_k=${K_VALUE}"
} > "${DONE_FILE}"

echo "===== done task ${TASK_ID} ====="
