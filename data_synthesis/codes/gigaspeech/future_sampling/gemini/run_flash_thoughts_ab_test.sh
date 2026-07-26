#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/llm_future_sampling_thinking_policy_gemini_future_distribution.py"

MANIFEST="${1:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
UTT_ID="${2:-AUD0000000003_1}"
OUTPUT_ROOT="${3:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/flash_thoughts_ab_${UTT_ID}}"
GATE_API_BASE="${GATE_API_BASE:-http://127.0.0.1:8100/v1}"
GATE_API_MODEL_NAME="${GATE_API_MODEL_NAME:-qwen3-instruct}"
GEMINI_API_BASE="${GEMINI_API_BASE:-https://generativelanguage.googleapis.com/v1beta/openai/}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}"
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: python script not found: ${PY_SCRIPT}"
  exit 1
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

run_case() {
  local mode="$1"
  local thoughts_flag="$2"
  local out_dir="${OUTPUT_ROOT}/${mode}"

  mkdir -p "${out_dir}"

  echo ""
  echo "============================================================"
  echo "Running mode=${mode}"
  echo "  utt_id=${UTT_ID}"
  echo "  thoughts_flag=${thoughts_flag}"
  echo "  output_dir=${out_dir}"
  echo "============================================================"

  python "${PY_SCRIPT}" \
    --input-tsv "${MANIFEST}" \
    --output-root "${out_dir}" \
    --test-one \
    --utt-id "${UTT_ID}" \
    --verbose \
    --overwrite \
    --base-model-path "/data/user_data/haolingp/models/Qwen3-4B-Base" \
    --gate-api-base "${GATE_API_BASE}" \
    --gate-api-model-name "${GATE_API_MODEL_NAME}" \
    --thinking-api-base "${GEMINI_API_BASE}" \
    --thinking-model-name "gemini-3-flash-preview" \
    --fallback-model-name "gemini-3-flash-preview" \
    --final-completion-model-name "gemini-3-flash-preview" \
    ${thoughts_flag} \
    --thinking-reasoning-effort "high" \
    --fallback-reasoning-effort "high" \
    --thinking-temperature 0.1 \
    --fallback-temperature 0.1 \
    --thinking-max-tokens 4096 \
    --fallback-max-tokens 4096 \
    --final-completion-max-tokens 4096 \
    --num-futures 10 \
    --future-tokens 12 \
    --sample-temperature 1.0 \
    --probe-max-futures 2 \
    --probe-top-k-logprobs 10 \
    --probe-rollout-tokens 3 \
    --probe-rollout-max-chars 4 \
    --probe-distribution-chars 2 \
    --probe-avg-entropy-threshold 0.75 \
    --probe-js-threshold 0.20 \
    --probe-min-semantic-mass 0.10
}

run_case "with_thoughts" "--gemini-include-thoughts"
run_case "without_thoughts" "--no-gemini-include-thoughts"

echo ""
echo "Done."
echo "JSON outputs:"
echo "  ${OUTPUT_ROOT}/with_thoughts/${UTT_ID}.json"
echo "  ${OUTPUT_ROOT}/without_thoughts/${UTT_ID}.json"
echo "Usage summaries:"
echo "  ${OUTPUT_ROOT}/with_thoughts/gemini_usage_summary.json"
echo "  ${OUTPUT_ROOT}/without_thoughts/gemini_usage_summary.json"
echo "Verbose logs:"
echo "  ${OUTPUT_ROOT}/with_thoughts/verbose_${UTT_ID}.log"
echo "  ${OUTPUT_ROOT}/without_thoughts/verbose_${UTT_ID}.log"
