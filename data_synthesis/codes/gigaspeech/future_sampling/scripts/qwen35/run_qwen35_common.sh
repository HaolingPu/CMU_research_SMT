#!/usr/bin/env bash
set -e

# Consensus decoding with the LOCAL Qwen3.5-122B-A10B-FP8 future sampler.
# Replaces the paid GPT-5.4-mini / DeepSeek API samplers: instead of an external
# API, we point the OpenAI-compatible /chat/completions backend at a locally
# served vLLM endpoint (see serve_qwen35_122b.sbatch, 4x L40S, port 8300).
#
# The frozen instruct TRANSLATOR (Qwen3-30B-A3B-Instruct) is still served
# locally on this job's single GPU; only the SAMPLER moved local.
#
# REQUIRED: SAMPLER_API_BASE must point at the running Qwen3.5 server, e.g.
#   SAMPLER_API_BASE=http://babel-s5-32:8300/v1

LABEL="${1:?Usage: bash run_qwen35_common.sh <label>}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/qwen35/${LABEL}}"

TOTAL_ROWS="${TOTAL_ROWS:-10}"
ROW_OFFSET="${ROW_OFFSET:-0}"
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  NUM_TASKS="${SLURM_ARRAY_TASK_COUNT}"
elif [[ -n "${NUM_TASKS:-}" ]]; then
  NUM_TASKS="${NUM_TASKS}"
else
  NUM_TASKS=1
fi

NUM_FUTURES="${NUM_FUTURES:-20}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-32}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-8}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-5}"
MIN_P="${MIN_P:-0}"
TOP_P="${TOP_P:-0}"
TARGET_LANG="${TARGET_LANG:-Chinese}"

# LOCAL Qwen3.5 future sampler (OpenAI-compatible /chat/completions backend).
CHAT_SAMPLER_MODEL="${CHAT_SAMPLER_MODEL:-qwen35-sampler}"
SAMPLER_API_BASE="${SAMPLER_API_BASE:?Set SAMPLER_API_BASE to the Qwen3.5 server, e.g. http://babel-s5-32:8300/v1}"
# vLLM accepts any bearer token; decoder requires a non-empty key.
SAMPLER_API_KEY="${SAMPLER_API_KEY:-EMPTY}"
# Qwen3.5 chain-of-thought. With ENABLE_THINKING=1 the server MUST be started with
# --reasoning-parser qwen3 (see serve_qwen35_122b.sbatch) so the thinking lands in
# message.reasoning_content and message.content stays clean JSON. ENABLE_THINKING=0
# disables thinking (no <think> block; works with or without the reasoning parser).
ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  CHAT_EXTRA_BODY="${CHAT_EXTRA_BODY:-{\"chat_template_kwargs\": {\"enable_thinking\": true}}}"
else
  CHAT_EXTRA_BODY="${CHAT_EXTRA_BODY:-{\"chat_template_kwargs\": {\"enable_thinking\": false}}}"
fi
# 'json' asks for 4 fields/continuation (text + sense_or_direction + translation_effect
# + confidence) + a note paragraph, but the decoder parser uses ONLY `text` -> ~75% of
# generated tokens are discarded (~1858 tok/call vs ~450 needed). 'numbered' emits a
# plain "1. ... 2. ..." list (parsed by parse_method_a_output) -> ~4x fewer tokens,
# ~4x faster sampling, identical decoding inputs. Default to the lean format.
SAMPLER_PROMPT_FORMAT="${SAMPLER_PROMPT_FORMAT:-numbered}"
# The chat sampler timeout is taken from --gpt-api-timeout in the decoder.
# Local Qwen3.5 in eager mode is ~13 tok/s, so a ~500-1500 tok futures call can
# take 40-120s+; raise well above the 120s default to avoid spurious timeouts.
GPT_API_TIMEOUT="${GPT_API_TIMEOUT:-600}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENABLE_VERBOSE="${ENABLE_VERBOSE:-1}"

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

INSTRUCT_PID="/tmp/qwen35dec_${LABEL}_instruct_${JOB_ID}_${TASK_ID}.pid"

stop_servers() {
  set +e
  PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_instruct_${TASK_ID}.log" 2>&1
}
trap stop_servers EXIT

wait_health() {
  name="$1"; url="$2"
  for i in $(seq 1 300); do
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

echo "===== Qwen3.5-122B consensus task ${TASK_ID} ====="
echo "job=${JOB_ID} node=$(hostname) label=${LABEL}"
echo "rows: start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS} tasks=${NUM_TASKS}"
echo "sampler=${CHAT_SAMPLER_MODEL} @ ${SAMPLER_API_BASE}"
echo "extra_body=${CHAT_EXTRA_BODY}"
echo "num_futures=${NUM_FUTURES} future_tokens=${FUTURE_TOKENS} max_steps=${MAX_CONSENSUS_STEPS}"
echo "concurrency=${NUM_CONCURRENT_CASES} top_k=${CANDIDATE_TOP_K} min_p=${MIN_P} top_p=${TOP_P}"
echo "output=${RUN_DIR}"

# Sanity: confirm the remote Qwen sampler is reachable before booting instruct.
if ! curl -s "${SAMPLER_API_BASE%/v1}/health" > /dev/null 2>&1 && ! curl -s "${SAMPLER_API_BASE}/models" > /dev/null 2>&1; then
  echo "[ERROR] Qwen3.5 sampler not reachable at ${SAMPLER_API_BASE} — is serve_qwen35_122b.sbatch running?"
  exit 1
fi
echo "[READY] remote Qwen3.5 sampler reachable"

# Preflight a REAL request at the decoder's max_tokens. /health alone is not
# enough: if the server's --max-model-len is below the decoder's budget
# (6000 reasoning + visible, ~10.4k for nf=20), every sampler call 400s and the
# decoder silently degrades to futureless decoding (calls=0 in gpt_sampler_usage)
# while still writing plausible-looking outputs. Fail loudly here instead.
VISIBLE_BUDGET=$(( FUTURE_TOKENS * NUM_FUTURES * 6 ))
ALT_BUDGET=$(( 200 * NUM_FUTURES + 400 ))
if (( ALT_BUDGET > VISIBLE_BUDGET )); then VISIBLE_BUDGET="${ALT_BUDGET}"; fi
PREFLIGHT_MAX_TOKENS=$(( 6000 + VISIBLE_BUDGET ))
PREFLIGHT_HTTP=$(curl -s -o /tmp/qwen35_preflight_${JOB_ID}_${TASK_ID}.json -w "%{http_code}" \
  --max-time 120 "${SAMPLER_API_BASE}/chat/completions" \
  -H "Content-Type: application/json" -H "Authorization: Bearer ${SAMPLER_API_KEY}" \
  -d "{\"model\": \"${CHAT_SAMPLER_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: ok\"}], \"max_tokens\": ${PREFLIGHT_MAX_TOKENS}, \"chat_template_kwargs\": {\"enable_thinking\": false}}" || true)
if [[ "${PREFLIGHT_HTTP}" != "200" ]]; then
  echo "[ERROR] sampler preflight failed (HTTP ${PREFLIGHT_HTTP}) at max_tokens=${PREFLIGHT_MAX_TOKENS}:"
  head -c 500 /tmp/qwen35_preflight_${JOB_ID}_${TASK_ID}.json; echo
  echo "        Server --max-model-len is likely too small — serve with MAX_LEN >= $(( PREFLIGHT_MAX_TOKENS + 2048 ))."
  exit 1
fi
echo "[READY] sampler preflight OK (max_tokens=${PREFLIGHT_MAX_TOKENS})"

PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &

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
  echo "min_p=${MIN_P}"
  echo "top_p=${TOP_P}"
} > "${DONE_FILE}"

echo "===== done task ${TASK_ID} in ${WALL}s ($(awk "BEGIN{ printf \"%.2f\", ${WALL}/${ROW_COUNT} }") s/row) ====="
