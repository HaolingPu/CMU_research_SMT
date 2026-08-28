#!/usr/bin/env bash
set -e

# Consensus decoding with the Qwen3.6-35B-A3B future sampler served by a HOSTED
# API (OpenRouter by default) instead of the local vLLM server. Same GPT
# ambiguity-futures prompt (consensus_decoding_token_id_level_gpt.py); only the
# sampler endpoint moves off-node. The frozen instruct TRANSLATOR
# (Qwen3-30B-A3B-Instruct) still needs token-level logprobs, so it stays local
# on INSTRUCT_GPU.
#
# REQUIRED: OPENROUTER_API_KEY in the environment or /data/user_data/haolingp/.env
#
# Thinking: OpenRouter normalizes reasoning control across providers via the
# "reasoning" request field, not Qwen's chat_template_kwargs. The decoder reads
# only message.content, so reasoning output never contaminates the futures list.
# Pricing (2026-07): qwen/qwen3.6-35b-a3b = $0.14/M input, $1.00/M output.

LABEL="${1:?Usage: bash run_qwen36_api_common.sh <label>}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${OUTPUT_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"

CODE_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM="${GEMMA4_ENV}/bin/vllm"

DECODER="${CODE_DIR}/consensus_decoding_token_id_level_gpt.py"
INSTRUCT_MODEL="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
TOKENIZER="${INSTRUCT_MODEL}"
INSTRUCT_GPU="${INSTRUCT_GPU:-1}"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/qwen36_api/${LABEL}}"

TOTAL_ROWS="${TOTAL_ROWS:-10}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_TASKS="${NUM_TASKS:-1}"

NUM_FUTURES="${NUM_FUTURES:-20}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-32}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-10}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-5}"
MIN_P="${MIN_P:-0}"
TOP_P="${TOP_P:-0}"
TARGET_LANG="${TARGET_LANG:-Chinese}"

# Hosted Qwen3.6 sampler over OpenAI-compatible /chat/completions.
SAMPLER_API_BASE="${SAMPLER_API_BASE:-https://openrouter.ai/api/v1}"
CHAT_SAMPLER_MODEL="${CHAT_SAMPLER_MODEL:-qwen/qwen3.6-35b-a3b}"
# Load .env for OPENROUTER_API_KEY if not already exported.
if [[ -z "${OPENROUTER_API_KEY:-}" && -f /data/user_data/haolingp/.env ]]; then
  OPENROUTER_API_KEY=$(grep -E "^OPENROUTER_API_KEY=" /data/user_data/haolingp/.env | cut -d= -f2- | tr -d '"' | tr -d "'")
fi
SAMPLER_API_KEY="${SAMPLER_API_KEY:-${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (env or /data/user_data/haolingp/.env)}}"
# ENABLE_THINKING=1 -> OpenRouter reasoning enabled; 0 -> disabled.
# THINKING_BUDGET=N (with ENABLE_THINKING=1) caps thinking tokens per call via
# OpenRouter reasoning.max_tokens (-> Qwen thinking_budget): when exhausted the
# server force-closes the think block, so a call can never self-loop past N.
ENABLE_THINKING="${ENABLE_THINKING:-1}"
THINKING_BUDGET="${THINKING_BUDGET:-}"
if [[ "${ENABLE_THINKING}" == "1" && -n "${THINKING_BUDGET}" ]]; then
  CHAT_EXTRA_BODY="${CHAT_EXTRA_BODY:-{\"reasoning\": {\"max_tokens\": ${THINKING_BUDGET}}}}"
elif [[ "${ENABLE_THINKING}" == "1" ]]; then
  CHAT_EXTRA_BODY="${CHAT_EXTRA_BODY:-{\"reasoning\": {\"enabled\": true}}}"
else
  CHAT_EXTRA_BODY="${CHAT_EXTRA_BODY:-{\"reasoning\": {\"enabled\": false}}}"
fi
SAMPLER_PROMPT_FORMAT="${SAMPLER_PROMPT_FORMAT:-numbered}"
GPT_API_TIMEOUT="${GPT_API_TIMEOUT:-600}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENABLE_VERBOSE="${ENABLE_VERBOSE:-1}"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

ROW_START="${ROW_OFFSET}"
ROW_COUNT="${TOTAL_ROWS}"

PORT_BASE="${PORT_BASE:-$(( 8200 + TASK_ID * 10 ))}"
INSTRUCT_PORT=$(( PORT_BASE + 0 ))

RUN_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/task_${TASK_ID}"
LOG_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/serve_logs"
DONE_FILE="${RUN_DIR}/DONE.txt"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

INSTRUCT_PID_FILE="/tmp/qwen36api_${LABEL}_instruct_${JOB_ID}_${TASK_ID}.pid"

stop_servers() {
  set +e
  if [[ -f "${INSTRUCT_PID_FILE}" ]]; then
    PID=$(cat "${INSTRUCT_PID_FILE}")
    kill "${PID}" 2>/dev/null
    sleep 3
    kill -9 "${PID}" 2>/dev/null
    rm -f "${INSTRUCT_PID_FILE}"
  fi
  PORT_PID=$(lsof -ti :"${INSTRUCT_PORT}" 2>/dev/null)
  if [[ -n "${PORT_PID}" ]]; then
    kill ${PORT_PID} 2>/dev/null
  fi
}
trap stop_servers EXIT

wait_health() {
  name="$1"; url="$2"
  for i in $(seq 1 900); do
    if curl -s "${url}" > /dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"; return 0
    fi
    sleep 1
  done
  echo "[ERROR] ${name} not ready: ${url}"; return 1
}

if [[ -f "${DONE_FILE}" ]]; then
  echo "[SKIP] ${DONE_FILE} exists"; exit 0
fi

echo "===== Qwen3.6-35B API-sampler consensus task ${TASK_ID} ====="
echo "job=${JOB_ID} node=$(hostname) label=${LABEL}"
echo "rows: start=${ROW_START} count=${ROW_COUNT}"
echo "sampler=${CHAT_SAMPLER_MODEL} @ ${SAMPLER_API_BASE}"
echo "extra_body=${CHAT_EXTRA_BODY}"
echo "num_futures=${NUM_FUTURES} future_tokens=${FUTURE_TOKENS} max_steps=${MAX_CONSENSUS_STEPS}"
echo "concurrency=${NUM_CONCURRENT_CASES} top_k=${CANDIDATE_TOP_K} min_p=${MIN_P} top_p=${TOP_P}"
echo "output=${RUN_DIR}"

# Preflight a REAL request: verifies key, model id, and reasoning field shape.
PREFLIGHT_MAX_TOKENS=$(( 6000 + FUTURE_TOKENS * NUM_FUTURES * 6 ))
PREFLIGHT_OUT="/tmp/qwen36api_preflight_${JOB_ID}_${TASK_ID}.json"
PREFLIGHT_HTTP=$(curl -s -o "${PREFLIGHT_OUT}" -w "%{http_code}" \
  --max-time 120 "${SAMPLER_API_BASE}/chat/completions" \
  -H "Content-Type: application/json" -H "Authorization: Bearer ${SAMPLER_API_KEY}" \
  -d "{\"model\": \"${CHAT_SAMPLER_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: ok\"}], \"max_tokens\": ${PREFLIGHT_MAX_TOKENS}, \"reasoning\": {\"enabled\": false}}" || true)
if [[ "${PREFLIGHT_HTTP}" != "200" ]]; then
  echo "[ERROR] sampler preflight failed (HTTP ${PREFLIGHT_HTTP}):"
  head -c 500 "${PREFLIGHT_OUT}"; echo
  exit 1
fi
echo "[READY] hosted sampler preflight OK (max_tokens=${PREFLIGHT_MAX_TOKENS})"

echo "Starting instruct translator on GPU ${INSTRUCT_GPU}, port ${INSTRUCT_PORT} ..."
CUDA_VISIBLE_DEVICES="${INSTRUCT_GPU}" "${VLLM}" serve "${INSTRUCT_MODEL}" \
  --served-model-name qwen3-instruct \
  --dtype auto \
  --port "${INSTRUCT_PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-logprobs 100 > "${LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &
echo $! > "${INSTRUCT_PID_FILE}"

wait_health "instruct" "http://127.0.0.1:${INSTRUCT_PORT}/health"

CMD=(
  "${PYTHON}" "${DECODER}"
  --input-tsv "${INPUT_TSV}"
  --sampler-backend chat
  --sampler-prompt-format "${SAMPLER_PROMPT_FORMAT}"
  --chat-sampler-model "${CHAT_SAMPLER_MODEL}"
  --chat-api-base "${SAMPLER_API_BASE}"
  --chat-api-key "${SAMPLER_API_KEY}"
  --chat-extra-body "${CHAT_EXTRA_BODY}"
  --gpt-api-timeout "${GPT_API_TIMEOUT}"
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
  echo "sampler_model=${CHAT_SAMPLER_MODEL}"
  echo "sampler_api_base=${SAMPLER_API_BASE}"
  echo "chat_extra_body=${CHAT_EXTRA_BODY}"
  echo "num_futures=${NUM_FUTURES}"
  echo "candidate_top_k=${CANDIDATE_TOP_K}"
} > "${DONE_FILE}"

echo "===== done task ${TASK_ID} in ${WALL}s ($(awk "BEGIN{ printf \"%.2f\", ${WALL}/${ROW_COUNT} }") s/row) ====="
