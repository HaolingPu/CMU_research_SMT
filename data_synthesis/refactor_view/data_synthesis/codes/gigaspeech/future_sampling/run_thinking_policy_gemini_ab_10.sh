#!/usr/bin/env bash
# Run a Gemini thinking-policy A/B test on the first 10 GigaSpeech utterances:
#   condition A: reasoning_effort=low
#   condition B: reasoning_effort=high
#
# Usage:
#   bash run_thinking_policy_gemini_ab_10.sh
# Optional overrides:
#   THINKING_MODEL_NAME=gemini-2.5-pro MAX_ROWS=10 PARALLEL_UTTERANCES=5 bash run_thinking_policy_gemini_ab_10.sh

set -euo pipefail

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at $HOME/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

conda activate vllm

MANIFEST_PATH="${MANIFEST_PATH:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy_gemini.py}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_gemini_flash_ab_10utt}"
THINKING_MODEL_NAME="${THINKING_MODEL_NAME:-gemini-2.5-flash}"
THINKING_API_BASE="${THINKING_API_BASE:-https://generativelanguage.googleapis.com/v1beta/openai/}"
MAX_ROWS="${MAX_ROWS:-10}"
PARALLEL_UTTERANCES="${PARALLEL_UTTERANCES:-10}"
NUM_FUTURES="${NUM_FUTURES:-5}"
FUTURE_TOKENS="${FUTURE_TOKENS:-10}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"
THINKING_TEMPERATURE="${THINKING_TEMPERATURE:-0.1}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-4096}"
FUTURE_SAMPLING_BATCH_SIZE="${FUTURE_SAMPLING_BATCH_SIZE:-4}"
FUTURE_SAMPLING_BATCH_WAIT="${FUTURE_SAMPLING_BATCH_WAIT:-0.05}"
ALIGN_DEVICE="${ALIGN_DEVICE:-cuda:0}"

export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  exit 1
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST_PATH}"
  exit 1
fi
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: script not found: ${SCRIPT_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT_BASE}"

run_one() {
  local effort="$1"
  local out_dir="${OUTPUT_ROOT_BASE}/reasoning_${effort}"
  mkdir -p "${out_dir}"

  echo "===== GEMINI THINKING POLICY: reasoning=${effort} ====="
  echo "time=$(date) node=$(hostname)"
  echo "output_root=${out_dir}"

  SIMALIGN_MODEL="/data/user_data/haolingp/models/LaBSE" CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_PATH}" \
    --input-tsv "${MANIFEST_PATH}" \
    --output-root "${out_dir}" \
    --task-id 0 \
    --num-tasks 1 \
    --max-rows "${MAX_ROWS}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --thinking-api-base "${THINKING_API_BASE}" \
    --thinking-model-name "${THINKING_MODEL_NAME}" \
    --thinking-reasoning-effort "${effort}" \
    --parallel-utterances "${PARALLEL_UTTERANCES}" \
    --future-sampling-batch-size "${FUTURE_SAMPLING_BATCH_SIZE}" \
    --future-sampling-batch-wait "${FUTURE_SAMPLING_BATCH_WAIT}" \
    --num-futures "${NUM_FUTURES}" \
    --future-tokens "${FUTURE_TOKENS}" \
    --sample-temperature "${SAMPLE_TEMPERATURE}" \
    --thinking-temperature "${THINKING_TEMPERATURE}" \
    --thinking-max-tokens "${THINKING_MAX_TOKENS}" \
    --align-device "${ALIGN_DEVICE}" \
    --verbose \
    --overwrite
}

run_one low
run_one high

echo "===== GEMINI THINKING POLICY A/B DONE ====="
echo "Results under ${OUTPUT_ROOT_BASE}"
