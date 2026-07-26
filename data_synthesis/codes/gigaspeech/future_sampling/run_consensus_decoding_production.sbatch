#!/usr/bin/env bash
# =============================================================================
# Production consensus decoding — 4 array tasks × 2 GPU each (8 GPU total).
# Each task serves its own 3 models on isolated ports and processes a chunk
# of rows.  No verbose logging — outputs only.
#
# Usage:
#   sbatch run_consensus_decoding_production.sbatch
#   TOTAL_ROWS=50000 MIN_P=0.05 sbatch run_consensus_decoding_production.sbatch
# =============================================================================
#SBATCH --job-name=cons_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=140G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --time=1-10:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_prod_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_prod_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
echo "===== START CONSENSUS DECODING TASK ${TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID:-NA} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"

GEMMA4_ENV="${GEMMA4_ENV:-/data/user_data/haolingp/conda_envs/gemma4}"
RUN_PYTHON="${RUN_PYTHON:-${GEMMA4_ENV}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${GEMMA4_ENV}/bin/vllm}"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

SCRIPT_DIR="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling"
SERVE_SCRIPT_INSTRUCT="${SCRIPT_DIR}/serve_instruct_gpu0.sh"
SERVE_SCRIPT_DUALBASE="${SCRIPT_DIR}/serve_dual_base_gpu1.sh"
SCRIPT_PATH="${SCRIPT_DIR}/consensus_decoding_token_id_level.py"
INSTRUCT_TOKENIZER_PATH="${INSTRUCT_TOKENIZER_PATH:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/gemma4/production/consensus_decoding_en_zh_minp0.05}"

TOTAL_ROWS="${TOTAL_ROWS:-40000}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-32}"
MIN_P="${MIN_P:-0.05}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-6}"
TARGET_LANG="${TARGET_LANG:-Chinese}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
ENABLE_VERBOSE="${ENABLE_VERBOSE:-1}"

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
ROW_START=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))
REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
if (( REMAINING <= 0 )); then
    echo "[SKIP] task ${TASK_ID} has no assigned rows."
    exit 0
fi
ROW_COUNT="${ROWS_PER_TASK}"
if (( REMAINING < ROW_COUNT )); then
    ROW_COUNT="${REMAINING}"
fi

PORT_BASE=$(( 8100 + TASK_ID * 10 ))
INSTRUCT_PORT=$(( PORT_BASE + 0 ))
PRIMARY_BASE_PORT=$(( PORT_BASE + 1 ))
SECONDARY_BASE_PORT=$(( PORT_BASE + 2 ))

RUN_ROOT="${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID:-manual}"
TASK_OUT_DIR="${RUN_ROOT}/task_${TASK_ID}"
RESULT_SENTINEL="${TASK_OUT_DIR}/DONE.txt"
RUN_LOG="${TASK_OUT_DIR}/run.log"
SERVE_LOG_DIR="${RUN_ROOT}/serve_logs"
INSTRUCT_PID_FILE="/tmp/vllm_serve_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.pid"
GEMMA_PID_FILE="/tmp/vllm_gemma4_e2b_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.pid"
QWEN_PID_FILE="/tmp/vllm_qwen3_4b_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.pid"

VERBOSE_DIR="${TASK_OUT_DIR}/verbose"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs
mkdir -p "${TASK_OUT_DIR}" "${VERBOSE_DIR}" "${SERVE_LOG_DIR}"

echo "Input TSV              : ${INPUT_TSV}"
echo "Output dir             : ${TASK_OUT_DIR}"
echo "Rows                   : offset=${ROW_OFFSET} start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS}"
echo "Num tasks              : ${NUM_TASKS}"
echo "Futures                : total=${NUM_FUTURES} secondary=${SECONDARY_NUM_FUTURES}"
echo "Future tokens          : ${FUTURE_TOKENS}"
echo "Min-p                  : ${MIN_P}"
echo "Candidate top-k        : ${CANDIDATE_TOP_K}"
echo "Concurrent cases       : ${NUM_CONCURRENT_CASES}"
echo "Skip existing          : ${SKIP_EXISTING}"
echo "Enable verbose         : ${ENABLE_VERBOSE}"
echo "Ports                  : instruct=${INSTRUCT_PORT} primary=${PRIMARY_BASE_PORT} secondary=${SECONDARY_BASE_PORT}"

wait_for_health() {
    local name="$1"
    local url="$2"
    for i in $(seq 1 300); do
        if curl -s "${url}" > /dev/null 2>&1; then
            echo "[READY] ${name} after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[ERROR] ${name} failed health check: ${url}"
    return 1
}

cleanup() {
    set +e
    PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID_FILE}" \
        bash "${SERVE_SCRIPT_INSTRUCT}" stop > "${SERVE_LOG_DIR}/instruct_stop_${TASK_ID}.log" 2>&1 || true
    GEMMA_PORT="${PRIMARY_BASE_PORT}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" \
    QWEN_PORT="${SECONDARY_BASE_PORT}" QWEN_PID_FILE="${QWEN_PID_FILE}" \
        bash "${SERVE_SCRIPT_DUALBASE}" stop > "${SERVE_LOG_DIR}/dualbase_stop_${TASK_ID}.log" 2>&1 || true
    if [[ -n "${SERVE_INSTRUCT_WRAPPER_PID:-}" ]]; then
        kill "${SERVE_INSTRUCT_WRAPPER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${SERVE_DUALBASE_WRAPPER_PID:-}" ]]; then
        kill "${SERVE_DUALBASE_WRAPPER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -f "${RESULT_SENTINEL}" ]]; then
    echo "[SKIP] ${RESULT_SENTINEL} exists."
    exit 0
fi

echo "[SERVE] Starting instruct serve script ..."
PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID_FILE}" VLLM_BIN="${VLLM_BIN}" \
    bash "${SERVE_SCRIPT_INSTRUCT}" > "${SERVE_LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &
SERVE_INSTRUCT_WRAPPER_PID=$!

echo "[SERVE] Starting dual-base serve script ..."
GEMMA_PORT="${PRIMARY_BASE_PORT}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" \
QWEN_PORT="${SECONDARY_BASE_PORT}" QWEN_PID_FILE="${QWEN_PID_FILE}" VLLM_BIN="${VLLM_BIN}" \
    bash "${SERVE_SCRIPT_DUALBASE}" > "${SERVE_LOG_DIR}/serve_dualbase_${TASK_ID}.log" 2>&1 &
SERVE_DUALBASE_WRAPPER_PID=$!

wait_for_health "instruct" "http://127.0.0.1:${INSTRUCT_PORT}/health"
wait_for_health "gemma4-e2b" "http://127.0.0.1:${PRIMARY_BASE_PORT}/health"
wait_for_health "qwen3-4b-base" "http://127.0.0.1:${SECONDARY_BASE_PORT}/health"

echo "[RUN] Launching consensus decoding ..."
RUN_CMD=(
"${RUN_PYTHON}" "${SCRIPT_PATH}" \
    --input-tsv "${INPUT_TSV}" \
    --base-api-base "http://127.0.0.1:${PRIMARY_BASE_PORT}/v1" \
    --base-api-model "gemma4-e2b" \
    --secondary-base-api-base "http://127.0.0.1:${SECONDARY_BASE_PORT}/v1" \
    --secondary-base-api-model "qwen3-4b-base" \
    --num-futures "${NUM_FUTURES}" \
    --secondary-num-futures "${SECONDARY_NUM_FUTURES}" \
    --future-tokens "${FUTURE_TOKENS}" \
    --max-consensus-steps "${MAX_CONSENSUS_STEPS}" \
    --min-p "${MIN_P}" \
    --candidate-top-k "${CANDIDATE_TOP_K}" \
    --instruct-api-base "http://127.0.0.1:${INSTRUCT_PORT}/v1" \
    --instruct-api-model "qwen3-instruct" \
    --instruct-tokenizer-path "${INSTRUCT_TOKENIZER_PATH}" \
    --target-lang "${TARGET_LANG}" \
    --row-idx "${ROW_START}" \
    --max-rows "${ROW_COUNT}" \
    --output-jsonl "${TASK_OUT_DIR}/results.jsonl" \
    --num-concurrent-cases "${NUM_CONCURRENT_CASES}"
)
if [[ "${SKIP_EXISTING}" == "1" || "${SKIP_EXISTING}" == "true" || "${SKIP_EXISTING}" == "TRUE" ]]; then
    RUN_CMD+=(--skip-existing)
fi
if [[ "${ENABLE_VERBOSE}" == "1" || "${ENABLE_VERBOSE}" == "true" || "${ENABLE_VERBOSE}" == "TRUE" ]]; then
    RUN_CMD+=(--verbose --verbose-dir "${VERBOSE_DIR}")
fi
"${RUN_CMD[@]}" 2>&1 | tee "${RUN_LOG}"

{
    echo "completed_at=$(date)"
    echo "rows_start=${ROW_START}"
    echo "rows_count=${ROW_COUNT}"
    echo "num_futures=${NUM_FUTURES}"
    echo "secondary_num_futures=${SECONDARY_NUM_FUTURES}"
    echo "min_p=${MIN_P}"
} > "${RESULT_SENTINEL}"

echo "===== DONE CONSENSUS DECODING TASK ${TASK_ID} ====="
