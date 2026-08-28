#!/usr/bin/env bash
# =============================================================================
# 100-row Chinese consensus decoding — 4 array tasks × 2 GPU (8 GPU total).
# Quick quality check with the new token-id continuation code.
# =============================================================================
#SBATCH --job-name=cons_100zh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=140G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --time=03:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_100zh_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_100zh_%A_%a.err

set -e
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
echo "===== START 100-zh quality check — task ${TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID:-NA} array_job=${SLURM_ARRAY_JOB_ID:-NA} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"

GEMMA4_ENV="${GEMMA4_ENV:-/data/user_data/haolingp/conda_envs/gemma4}"
RUN_PYTHON="${GEMMA4_ENV}/bin/python"
VLLM_BIN="${VLLM_BIN:-${GEMMA4_ENV}/bin/vllm}"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
SERVE_SCRIPT_INSTRUCT="${SCRIPT_DIR}/serve_instruct_gpu0.sh"
SERVE_SCRIPT_DUALBASE="${SCRIPT_DIR}/serve_dual_base_gpu1.sh"
SCRIPT_PATH="${SCRIPT_DIR}/consensus_decoding_token_id_level.py"
INSTRUCT_TOKENIZER_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"

TOTAL_ROWS=100
NUM_TASKS=4
ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
ROW_START=$(( TASK_ID * ROWS_PER_TASK ))
REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
if (( REMAINING <= 0 )); then
    echo "[SKIP] task ${TASK_ID} has no assigned rows."
    exit 0
fi
ROW_COUNT="${ROWS_PER_TASK}"
if (( REMAINING < ROW_COUNT )); then
    ROW_COUNT="${REMAINING}"
fi

RUN_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/gemma4/quality_check_100zh_${SLURM_ARRAY_JOB_ID:-manual}"
TASK_OUT_DIR="${RUN_ROOT}/task_${TASK_ID}"
VERBOSE_DIR="${TASK_OUT_DIR}/verbose"
SERVE_LOG_DIR="${RUN_ROOT}/serve_logs"
RESULT_SENTINEL="${TASK_OUT_DIR}/DONE.txt"

PORT_BASE=$(( 8100 + TASK_ID * 10 ))
INSTRUCT_PORT=$(( PORT_BASE + 0 ))
PRIMARY_BASE_PORT=$(( PORT_BASE + 1 ))
SECONDARY_BASE_PORT=$(( PORT_BASE + 2 ))
INSTRUCT_PID_FILE="/tmp/vllm_100zh_instruct_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"
GEMMA_PID_FILE="/tmp/vllm_100zh_gemma_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"
QWEN_PID_FILE="/tmp/vllm_100zh_qwen_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs
mkdir -p "${TASK_OUT_DIR}" "${VERBOSE_DIR}" "${SERVE_LOG_DIR}"

echo "Output dir : ${TASK_OUT_DIR}"
echo "Rows       : start=${ROW_START} count=${ROW_COUNT}"
echo "Ports      : instruct=${INSTRUCT_PORT} primary=${PRIMARY_BASE_PORT} secondary=${SECONDARY_BASE_PORT}"

wait_for_health() {
    local name="$1" url="$2"
    for i in $(seq 1 300); do
        if curl -s "${url}" > /dev/null 2>&1; then
            echo "[READY] ${name} after ${i}s"; return 0
        fi
        sleep 1
    done
    echo "[ERROR] ${name} failed health check: ${url}"; return 1
}

cleanup() {
    set +e
    PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID_FILE}" \
        bash "${SERVE_SCRIPT_INSTRUCT}" stop > "${SERVE_LOG_DIR}/instruct_stop_${TASK_ID}.log" 2>&1 || true
    GEMMA_PORT="${PRIMARY_BASE_PORT}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" \
    QWEN_PORT="${SECONDARY_BASE_PORT}" QWEN_PID_FILE="${QWEN_PID_FILE}" \
        bash "${SERVE_SCRIPT_DUALBASE}" stop > "${SERVE_LOG_DIR}/dualbase_stop_${TASK_ID}.log" 2>&1 || true
    kill "${SERVE_INSTRUCT_PID:-}" 2>/dev/null || true
    kill "${SERVE_DUALBASE_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -f "${RESULT_SENTINEL}" ]]; then
    echo "[SKIP] ${RESULT_SENTINEL} exists."
    exit 0
fi

echo "[SERVE] Starting instruct ..."
PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID_FILE}" VLLM_BIN="${VLLM_BIN}" \
    bash "${SERVE_SCRIPT_INSTRUCT}" > "${SERVE_LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &
SERVE_INSTRUCT_PID=$!

echo "[SERVE] Starting dual-base ..."
GEMMA_PORT="${PRIMARY_BASE_PORT}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" \
QWEN_PORT="${SECONDARY_BASE_PORT}" QWEN_PID_FILE="${QWEN_PID_FILE}" VLLM_BIN="${VLLM_BIN}" \
    bash "${SERVE_SCRIPT_DUALBASE}" > "${SERVE_LOG_DIR}/serve_dualbase_${TASK_ID}.log" 2>&1 &
SERVE_DUALBASE_PID=$!

wait_for_health "instruct" "http://127.0.0.1:${INSTRUCT_PORT}/health"
wait_for_health "gemma4-e2b" "http://127.0.0.1:${PRIMARY_BASE_PORT}/health"
wait_for_health "qwen3-4b-base" "http://127.0.0.1:${SECONDARY_BASE_PORT}/health"

echo "[RUN] Launching consensus decoding — task ${TASK_ID}, ${ROW_COUNT} rows, Chinese ..."
"${RUN_PYTHON}" "${SCRIPT_PATH}" \
    --input-tsv "${INPUT_TSV}" \
    --base-api-base "http://127.0.0.1:${PRIMARY_BASE_PORT}/v1" \
    --base-api-model "gemma4-e2b" \
    --secondary-base-api-base "http://127.0.0.1:${SECONDARY_BASE_PORT}/v1" \
    --secondary-base-api-model "qwen3-4b-base" \
    --num-futures 20 \
    --secondary-num-futures 10 \
    --future-tokens 20 \
    --max-consensus-steps 32 \
    --min-p 0.05 \
    --instruct-api-base "http://127.0.0.1:${INSTRUCT_PORT}/v1" \
    --instruct-api-model "qwen3-instruct" \
    --instruct-tokenizer-path "${INSTRUCT_TOKENIZER_PATH}" \
    --target-lang Chinese \
    --row-idx "${ROW_START}" \
    --max-rows "${ROW_COUNT}" \
    --verbose \
    --verbose-dir "${VERBOSE_DIR}" \
    --output-jsonl "${TASK_OUT_DIR}/results.jsonl" \
    --num-concurrent-cases 8 \
    2>&1 | tee "${TASK_OUT_DIR}/run.log"

{
    echo "completed_at=$(date)"
    echo "rows_start=${ROW_START}"
    echo "rows_count=${ROW_COUNT}"
} > "${RESULT_SENTINEL}"

echo "===== DONE task ${TASK_ID} ====="
