#!/usr/bin/env bash
set -e

ROOT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement"
PYTHON="${PYTHON:-/data/user_data/haolingp/conda_envs/gemma4/bin/python}"
MT_API_BASE="${MT_API_BASE:-http://localhost:8100}"
MT_API_MODEL="${MT_API_MODEL:-qwen3-instruct}"
TOKENIZER="${TOKENIZER:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
ROW_IDX="${ROW_IDX:-0}"
MAX_ROWS="${MAX_ROWS:-100}"
OUT_DIR="${OUT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_n2_seg4_rows${ROW_IDX}_${MAX_ROWS}}"

mkdir -p "${OUT_DIR}"

"${PYTHON}" "${ROOT}/local_agreement.py" \
  --mt-api-base "${MT_API_BASE}" \
  --mt-api-model "${MT_API_MODEL}" \
  --mt-tokenizer-path "${TOKENIZER}" \
  --la-n 2 \
  --segment-size 4 \
  --lcp-mode char \
  --target-lang Chinese \
  --row-idx "${ROW_IDX}" \
  --max-rows "${MAX_ROWS}" \
  --overwrite \
  --output-dir "${OUT_DIR}" \
  "$@"
