#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/llm_future_sampling_thinking_policy_gemini_future_distribution.py"

MANIFEST="${1:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
UTT_ID="${2:-AUD0000000003_1}"
OUTPUT_ROOT="${3:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/abc_test_${UTT_ID}}"
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

run_mode() {
  local mode="$1"
  local primary_model="$2"
  local fallback_model="$3"
  local final_model="$4"
  local out_dir="${OUTPUT_ROOT}/${mode}"

  mkdir -p "${out_dir}"

  echo ""
  echo "============================================================"
  echo "Running mode=${mode}"
  echo "  utt_id=${UTT_ID}"
  echo "  primary_model=${primary_model}"
  echo "  fallback_model=${fallback_model}"
  echo "  final_model=${final_model}"
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
    --thinking-model-name "${primary_model}" \
    --fallback-model-name "${fallback_model}" \
    --final-completion-model-name "${final_model}" \
    --gemini-include-thoughts \
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

run_mode "flash_only" "gemini-3-flash-preview" "gemini-3-flash-preview" "gemini-3-flash-preview"
run_mode "pro_only" "gemini-3.1-pro-preview" "gemini-3.1-pro-preview" "gemini-3.1-pro-preview"
run_mode "gated_flash_to_pro" "gemini-3-flash-preview" "gemini-3.1-pro-preview" "gemini-3.1-pro-preview"

echo ""
echo "Done."
echo "JSON outputs:"
echo "  ${OUTPUT_ROOT}/flash_only/${UTT_ID}.json"
echo "  ${OUTPUT_ROOT}/pro_only/${UTT_ID}.json"
echo "  ${OUTPUT_ROOT}/gated_flash_to_pro/${UTT_ID}.json"
echo "Verbose logs:"
echo "  ${OUTPUT_ROOT}/flash_only/verbose_${UTT_ID}.log"
echo "  ${OUTPUT_ROOT}/pro_only/verbose_${UTT_ID}.log"
echo "  ${OUTPUT_ROOT}/gated_flash_to_pro/verbose_${UTT_ID}.log"
