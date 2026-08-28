#!/usr/bin/env bash
set -e

# Smoke run: Prefix Alignment on first 5 cases, char-level chrF scorer, debug on.

ROOT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/prefix_alignment"
PYTHON="${PYTHON:-/data/user_data/haolingp/conda_envs/gemma4/bin/python}"
MT_API_BASE="${MT_API_BASE:-http://localhost:8100}"
MT_API_MODEL="${MT_API_MODEL:-qwen3-instruct}"
TOKENIZER="${TOKENIZER:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv}"
ROW_IDX="${ROW_IDX:-0}"
MAX_ROWS="${MAX_ROWS:-5}"
TARGET_LANG="${TARGET_LANG:-Chinese}"
TARGET_UNIT="${TARGET_UNIT:-char}"
LCP_MODE="${LCP_MODE:-char}"
SCORER="${SCORER:-chrf}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-5}"
OUT_DIR="${OUT_DIR:-${ROOT}/output/pa_5cases}"
OUT_JSONL="${OUT_JSONL:-${OUT_DIR}/pa_5cases.jsonl}"

mkdir -p "${OUT_DIR}"

"${PYTHON}" "${ROOT}/prefix_alignment.py" \
  --input-tsv "${INPUT_TSV}" \
  --id-key id \
  --src-trajectory-key src_trajectory \
  --target-key llm_reference_text \
  --mt-api-base "${MT_API_BASE}" \
  --mt-api-model "${MT_API_MODEL}" \
  --mt-tokenizer-path "${TOKENIZER}" \
  --target-lang "${TARGET_LANG}" \
  --target-unit "${TARGET_UNIT}" \
  --lcp-mode "${LCP_MODE}" \
  --scorer "${SCORER}" \
  --row-idx "${ROW_IDX}" \
  --max-rows "${MAX_ROWS}" \
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}" \
  --debug \
  --overwrite \
  --output-jsonl "${OUT_JSONL}" \
  --output-dir "${OUT_DIR}" \
  "$@"
