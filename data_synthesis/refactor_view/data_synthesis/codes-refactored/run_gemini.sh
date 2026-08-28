#!/usr/bin/env bash
# ============================================================
# Gemini future-sampling simultaneous translation on GigaSpeech.
# 8 GPUs in parallel via SLURM array.
#
# Override any path via environment variables before sbatch:
#   MANIFEST_PATH=/path/to/tsv OUTPUT_ROOT=/path/to/out sbatch run_gemini.sh
# ============================================================
#SBATCH --job-name=gem_future
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7%8
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL

set -e

source ~/.bashrc
conda activate vllm

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS="${NUM_TASKS:-8}"

# --- Configurable paths (override via env) ---
MANIFEST_PATH="${MANIFEST_PATH:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/refactored}"
BASE_MODEL_URL="${BASE_MODEL_URL:-http://localhost:8000/v1}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-4B-Base}"
THINKING_MODEL="${THINKING_MODEL:-gemini-3.1-pro-preview}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
MAX_ROWS="${MAX_ROWS:-}"
NUM_FUTURES="${NUM_FUTURES:-5}"
FUTURE_TOKENS="${FUTURE_TOKENS:-10}"

# --- Validate ---
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "===== START TASK ${TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID:-N/A} node=$(hostname) time=$(date)"
echo "model=${THINKING_MODEL} reasoning=${REASONING_EFFORT} futures=${NUM_FUTURES}"
echo "base_model_url=${BASE_MODEL_URL} base_model_name=${BASE_MODEL_NAME}"

MAX_ROWS_FLAG=""
if [[ -n "${MAX_ROWS}" ]]; then
  MAX_ROWS_FLAG="--max-rows ${MAX_ROWS}"
fi

python "${CODE_DIR}/main.py" \
  --input-tsv "${MANIFEST_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --base-model-url "${BASE_MODEL_URL}" \
  --base-model-name "${BASE_MODEL_NAME}" \
  --thinking-model-name "${THINKING_MODEL}" \
  --thinking-reasoning-effort "${REASONING_EFFORT}" \
  --num-futures "${NUM_FUTURES}" \
  --future-tokens "${FUTURE_TOKENS}" \
  --task-id "${TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  ${MAX_ROWS_FLAG} \
  --overwrite

echo "===== DONE TASK ${TASK_ID} ====="
