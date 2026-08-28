#!/usr/bin/env bash
set -e

LABEL="${1:?Usage: bash run_deepseek_common.sh <label>}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${OUTPUT_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"

CODE_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${CODE_DIR}/serve_instruct_gpu0.sh"
DECODER="${CODE_DIR}/consensus_decoding_token_id_level_gpt.py"
TOKENIZER="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/deepseek/${LABEL}}"

# Default to 10 rows for the first sanity / speed test. Override via TOTAL_ROWS=100 etc.
TOTAL_ROWS="${TOTAL_ROWS:-10}"
ROW_OFFSET="${ROW_OFFSET:-0}"
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  NUM_TASKS="${SLURM_ARRAY_TASK_COUNT}"
elif [[ -n "${NUM_TASKS:-}" ]]; then
  NUM_TASKS="${NUM_TASKS}"
else
  NUM_TASKS=8
fi

NUM_FUTURES="${NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-32}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-8}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-5}"
MIN_P="${MIN_P:-0}"
TOP_P="${TOP_P:-0}"
TARGET_LANG="${TARGET_LANG:-Chinese}"
# DeepSeek future sampler (OpenAI-compatible /chat/completions backend).
CHAT_SAMPLER_MODEL="${CHAT_SAMPLER_MODEL:-deepseek-v4-pro}"
# JSON merged into the request body. For a reasoning model, set the documented
# thinking toggle, e.g. CHAT_EXTRA_BODY='{"thinking": {"type": "enabled"}}'.
CHAT_EXTRA_BODY="${CHAT_EXTRA_BODY:-}"
# Future-sampler prompt format: json = rich (id/text/sense_or_direction/translation_effect/confidence).
SAMPLER_PROMPT_FORMAT="${SAMPLER_PROMPT_FORMAT:-json}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENABLE_VERBOSE="${ENABLE_VERBOSE:-0}"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

# Pull ONLY DEEPSEEK_API_KEY / DEEPSEEK_API_BASE from a .env if present.
# We deliberately do NOT source the .env wholesale, so that tuning knobs like
# CHAT_EXTRA_BODY keep coming from script defaults (or sbatch --export),
# instead of being silently overridden by a stale .env preset.
for env_file in "${PWD}/.env" "${CODE_DIR}/.env" "/data/user_data/haolingp/.env"; do
  if [[ -f "${env_file}" ]]; then
    while IFS='=' read -r _key _val; do
      _val="${_val%\"}"; _val="${_val#\"}"
      _val="${_val%\'}"; _val="${_val#\'}"
      if [[ -z "${!_key:-}" ]]; then
        export "${_key}=${_val}"
      fi
    done < <(grep -E '^(DEEPSEEK_API_KEY|DEEPSEEK_API_BASE)=' "${env_file}")
    break
  fi
done

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[ERROR] DEEPSEEK_API_KEY is not set; cannot run DeepSeek sampler."
  exit 1
fi

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
ROW_START=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))
ROW_COUNT="${ROWS_PER_TASK}"
REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
if (( REMAINING <= 0 )); then
  echo "[SKIP] task ${TASK_ID} has no assigned rows (total=${TOTAL_ROWS}, tasks=${NUM_TASKS})"
  exit 0
fi
if (( REMAINING < ROW_COUNT )); then
  ROW_COUNT="${REMAINING}"
fi

PORT_BASE="${PORT_BASE:-$(( 8200 + TASK_ID * 10 ))}"
INSTRUCT_PORT=$(( PORT_BASE + 0 ))

RUN_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/task_${TASK_ID}"
LOG_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/serve_logs"
DONE_FILE="${RUN_DIR}/DONE.txt"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

INSTRUCT_PID="/tmp/deepseek_${LABEL}_instruct_${JOB_ID}_${TASK_ID}.pid"

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

echo "===== DeepSeek consensus task ${TASK_ID} ====="
echo "job=${JOB_ID} node=$(hostname) label=${LABEL}"
echo "rows: start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS} tasks=${NUM_TASKS}"
echo "chat_model=${CHAT_SAMPLER_MODEL} extra_body=${CHAT_EXTRA_BODY}"
echo "num_futures=${NUM_FUTURES} future_tokens=${FUTURE_TOKENS} max_steps=${MAX_CONSENSUS_STEPS}"
echo "concurrency=${NUM_CONCURRENT_CASES} top_k=${CANDIDATE_TOP_K} min_p=${MIN_P} top_p=${TOP_P}"
echo "output=${RUN_DIR}"

PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &

wait_health "instruct" "http://127.0.0.1:${INSTRUCT_PORT}/health"

CMD=(
  "${PYTHON}" "${DECODER}"
  --input-tsv "${INPUT_TSV}"
  --sampler-backend chat
  --sampler-prompt-format "${SAMPLER_PROMPT_FORMAT}"
  --chat-sampler-model "${CHAT_SAMPLER_MODEL}"
  --instruct-api-base "http://127.0.0.1:${INSTRUCT_PORT}/v1"
  --instruct-api-model "qwen3-instruct"
  --instruct-tokenizer-path "${TOKENIZER}"
  --target-lang "${TARGET_LANG}"
  --row-idx "${ROW_START}"
  --max-rows "${ROW_COUNT}"
  --num-futures "${NUM_FUTURES}"
  --future-tokens "${FUTURE_TOKENS}"
  --max-consensus-steps "${MAX_CONSENSUS_STEPS}"
  --candidate-top-k "${CANDIDATE_TOP_K}"
  --min-p "${MIN_P}"
  --top-p "${TOP_P}"
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}"
  --output-jsonl "${RUN_DIR}/results.jsonl"
)

if [[ -n "${CHAT_EXTRA_BODY}" ]]; then
  CMD+=(--chat-extra-body "${CHAT_EXTRA_BODY}")
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip-existing)
fi

if [[ "${ENABLE_VERBOSE}" == "1" ]]; then
  mkdir -p "${RUN_DIR}/verbose"
  CMD+=(--verbose --verbose-dir "${RUN_DIR}/verbose")
fi

START_TS=$(date +%s)
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
CMD_STATUS="${PIPESTATUS[0]}"
END_TS=$(date +%s)
WALL=$(( END_TS - START_TS ))

if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] decoder failed with exit code ${CMD_STATUS} after ${WALL}s"
  exit "${CMD_STATUS}"
fi

JSON_COUNT=$(find "${RUN_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
if [[ "${JSON_COUNT}" != "${ROW_COUNT}" ]]; then
  echo "[ERROR] incomplete task output: json_count=${JSON_COUNT}, expected=${ROW_COUNT}"
  exit 1
fi

{
  echo "completed_at=$(date)"
  echo "rows_start=${ROW_START}"
  echo "rows_count=${ROW_COUNT}"
  echo "wall_seconds=${WALL}"
  echo "sec_per_row=$(awk "BEGIN{ printf \"%.2f\", ${WALL} / ${ROW_COUNT} }")"
  echo "chat_model=${CHAT_SAMPLER_MODEL}"
  echo "chat_extra_body=${CHAT_EXTRA_BODY}"
  echo "num_futures=${NUM_FUTURES}"
  echo "candidate_top_k=${CANDIDATE_TOP_K}"
  echo "min_p=${MIN_P}"
  echo "top_p=${TOP_P}"
} > "${DONE_FILE}"

echo "===== done task ${TASK_ID} in ${WALL}s ($(awk "BEGIN{ printf \"%.2f\", ${WALL}/${ROW_COUNT} }") s/row) ====="
