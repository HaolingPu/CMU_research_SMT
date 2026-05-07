#!/usr/bin/env bash
# Local-debug equivalent of the "Consensus Decoding (refactored)" entry in
# .vscode/launch.json. Runs the refactored consensus decoder against the
# already-running localhost vLLM servers (instruct on 8100, base on 8101/8102).
#
# Override knobs by exporting before running, e.g.:
# MAX_ROWS=10 MAX_CONSENSUS_STEPS=10 NUM_FUTURES=200 SECONDARY_NUM_FUTURES=100 FUTURE_TOKENS=10 NUM_CONCURRENT_CASES=1 bash scripts/synth/debug_consensus.sh
# INSTRUCT_API_MODEL=gemma4-e4b-it INSTRUCT_TOKENIZER_PATH=/data/user_data/siqiouya/ckpts/pretrained/llm/gemma-4-E4B-it MAX_ROWS=10 MAX_CONSENSUS_STEPS=10 NUM_FUTURES=200 SECONDARY_NUM_FUTURES=100 FUTURE_TOKENS=10 NUM_CONCURRENT_CASES=10 bash scripts/synth/debug_consensus.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/siqiouya/miniconda3/envs/consensus/bin/python}"
DECODER="${REPO_ROOT}/data_synthesis/codes-refactored/consensus_decoding_token_id_level.py"

# === Decoding knobs ===
ROW_IDX="${ROW_IDX:-0}"
MAX_ROWS="${MAX_ROWS:-1}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-5}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-1}"
TARGET_LANG="${TARGET_LANG:-Chinese}"

# === API endpoints (must already be serving) ===
BASE_API_BASE="${BASE_API_BASE:-http://0.0.0.0:8101/v1}"
BASE_API_MODEL="${BASE_API_MODEL:-gemma4-e2b}"
SECONDARY_BASE_API_BASE="${SECONDARY_BASE_API_BASE:-http://0.0.0.0:8102/v1}"
SECONDARY_BASE_API_MODEL="${SECONDARY_BASE_API_MODEL:-qwen3-4b-base}"
INSTRUCT_API_BASE="${INSTRUCT_API_BASE:-http://0.0.0.0:8100/v1}"
INSTRUCT_API_MODEL="${INSTRUCT_API_MODEL:-qwen3-instruct}"
INSTRUCT_TOKENIZER_PATH="${INSTRUCT_TOKENIZER_PATH:-/data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-30B-A3B-Instruct-2507-FP8}"

# === Output ===
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/debug_output/consensus}"
mkdir -p "${OUTPUT_DIR}/log"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"

exec "${PYTHON}" "${DECODER}" \
  --row-idx "${ROW_IDX}" \
  --max-rows "${MAX_ROWS}" \
  --max-consensus-steps "${MAX_CONSENSUS_STEPS}" \
  --num-futures "${NUM_FUTURES}" \
  --secondary-num-futures "${SECONDARY_NUM_FUTURES}" \
  --future-tokens "${FUTURE_TOKENS}" \
  --candidate-top-k "${CANDIDATE_TOP_K}" \
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}" \
  --base-api-base "${BASE_API_BASE}" \
  --base-api-model "${BASE_API_MODEL}" \
  --secondary-base-api-base "${SECONDARY_BASE_API_BASE}" \
  --secondary-base-api-model "${SECONDARY_BASE_API_MODEL}" \
  --instruct-api-base "${INSTRUCT_API_BASE}" \
  --instruct-api-model "${INSTRUCT_API_MODEL}" \
  --instruct-tokenizer-path "${INSTRUCT_TOKENIZER_PATH}" \
  --target-lang "${TARGET_LANG}" \
  --output-jsonl "${OUTPUT_DIR}/out.jsonl" \
  --verbose \
  --verbose-dir "${OUTPUT_DIR}/log"
