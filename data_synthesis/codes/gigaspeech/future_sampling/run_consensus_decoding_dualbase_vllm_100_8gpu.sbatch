#!/usr/bin/env bash
# =============================================================================
# 100-case consensus decoding with 3 vLLM serves and 4 array tasks.
#
# Per array task:
#   GPU 0 -> Qwen3-30B-A3B-Instruct-2507-FP8 (instruct API)
#   GPU 1 -> Gemma 4 E2B base (primary future sampler)
#   GPU 1 -> Qwen3-4B-Base (secondary future sampler)
#
# Defaults:
#   - 100 rows total
#   - 4 array tasks
#   - 10 futures from Gemma + 10 futures from Qwen base
#   - outputs under ${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID}/task_${TASK_ID}
# =============================================================================
#SBATCH --job-name=cons_dbase_100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=140G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --time=0-12:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_dbase_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_dbase_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START DUAL-BASE CONSENSUS TASK ${SLURM_ARRAY_TASK_ID:-0} ====="
echo "job_id=${SLURM_JOB_ID:-NA} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"

GEMMA4_ENV="${GEMMA4_ENV:-/data/user_data/haolingp/conda_envs/gemma4}"
RUN_PYTHON="${GEMMA4_ENV}/bin/python"

SCRIPT_PATH="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/consensus_decoding_token_id_level.py"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL:-google/gemma-4-E2B}"
SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
INSTRUCT_TOKENIZER_PATH="${INSTRUCT_TOKENIZER_PATH:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-dualbase-100}"

TOTAL_ROWS="${TOTAL_ROWS:-100}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-32}"
MIN_P="${MIN_P:-0.0}"
INSTRUCT_BACKEND="${INSTRUCT_BACKEND:-api}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL:-0.45}"
SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL:-0.45}"
INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL:-0.90}"

PRIMARY_BASE_MAX_MODEL_LEN="${PRIMARY_BASE_MAX_MODEL_LEN:-4096}"
SECONDARY_BASE_MAX_MODEL_LEN="${SECONDARY_BASE_MAX_MODEL_LEN:-4096}"
INSTRUCT_MAX_MODEL_LEN="${INSTRUCT_MAX_MODEL_LEN:-8192}"

INSTRUCT_SERVED_MODEL="${INSTRUCT_SERVED_MODEL:-qwen3-instruct}"
PRIMARY_BASE_SERVED_MODEL="${PRIMARY_BASE_SERVED_MODEL:-gemma4-e2b-base}"
SECONDARY_BASE_SERVED_MODEL="${SECONDARY_BASE_SERVED_MODEL:-qwen3-4b-base}"

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
ROW_START=$(( TASK_ID * ROWS_PER_TASK ))
REMAINING=$(( TOTAL_ROWS - ROW_START ))
if (( REMAINING <= 0 )); then
    echo "[SKIP] task ${TASK_ID} has no assigned rows."
    exit 0
fi
ROW_COUNT="${ROWS_PER_TASK}"
if (( REMAINING < ROW_COUNT )); then
    ROW_COUNT="${REMAINING}"
fi

PORT_BASE=$(( 8200 + TASK_ID * 10 ))
INSTRUCT_PORT=$(( PORT_BASE + 0 ))
PRIMARY_BASE_PORT=$(( PORT_BASE + 1 ))
SECONDARY_BASE_PORT=$(( PORT_BASE + 2 ))

INSTRUCT_API="http://127.0.0.1:${INSTRUCT_PORT}/v1"
PRIMARY_BASE_API="http://127.0.0.1:${PRIMARY_BASE_PORT}/v1"
SECONDARY_BASE_API="http://127.0.0.1:${SECONDARY_BASE_PORT}/v1"

RUN_ROOT="${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID:-manual}"
TASK_OUT_DIR="${RUN_ROOT}/task_${TASK_ID}"
VERBOSE_DIR="${TASK_OUT_DIR}/verbose"
RESULT_SENTINEL="${TASK_OUT_DIR}/DONE.txt"
INSTRUCT_LOG="${TASK_OUT_DIR}/instruct_server.log"
PRIMARY_BASE_LOG="${TASK_OUT_DIR}/primary_base_server.log"
SECONDARY_BASE_LOG="${TASK_OUT_DIR}/secondary_base_server.log"
RUN_LOG="${TASK_OUT_DIR}/run.log"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs
mkdir -p "${TASK_OUT_DIR}" "${VERBOSE_DIR}"

echo "Input TSV              : ${INPUT_TSV}"
echo "Output dir             : ${TASK_OUT_DIR}"
echo "Rows                   : start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS}"
echo "Futures                : total=${NUM_FUTURES} secondary=${SECONDARY_NUM_FUTURES}"
echo "Future tokens          : ${FUTURE_TOKENS}"
echo "Min-p                  : ${MIN_P}"
echo "Instruct backend       : ${INSTRUCT_BACKEND}"
echo "Ports                  : instruct=${INSTRUCT_PORT} primary=${PRIMARY_BASE_PORT} secondary=${SECONDARY_BASE_PORT}"
echo "Instruct model         : ${INSTRUCT_MODEL}"
echo "Primary base model     : ${PRIMARY_BASE_MODEL}"
echo "Secondary base model   : ${SECONDARY_BASE_MODEL}"

cleanup() {
    for pid_var in SECONDARY_BASE_PID PRIMARY_BASE_PID INSTRUCT_PID; do
        pid="${!pid_var:-}"
        if [[ -n "${pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

wait_for_api() {
    local name="$1"
    local url="$2"
    local pid="$3"
    local log_path="$4"
    local waited=0
    local max_wait=600
    until curl -sS "${url}/models" > /dev/null 2>&1; do
        sleep 5
        waited=$(( waited + 5 ))
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[ERROR] ${name} server exited before readiness."
            tail -n 80 "${log_path}" || true
            exit 1
        fi
        if (( waited >= max_wait )); then
            echo "[ERROR] ${name} server did not become ready within ${max_wait}s."
            tail -n 80 "${log_path}" || true
            exit 1
        fi
    done
    echo "[READY] ${name} after ${waited}s"
}

if [[ -f "${RESULT_SENTINEL}" ]]; then
    echo "[SKIP] ${RESULT_SENTINEL} exists."
    exit 0
fi

INSTRUCT_ARGS=()
if [[ "${INSTRUCT_BACKEND}" == "api" ]]; then
    echo "[vLLM] Starting instruct server on allocation-local GPU 0 ..."
    CUDA_VISIBLE_DEVICES=0 "${RUN_PYTHON}" -m vllm.entrypoints.openai.api_server \
        --model "${INSTRUCT_MODEL}" \
        --served-model-name "${INSTRUCT_SERVED_MODEL}" \
        --port "${INSTRUCT_PORT}" \
        --tensor-parallel-size 1 \
        --dtype auto \
        --gpu-memory-utilization "${INSTRUCT_GPU_UTIL}" \
        --max-model-len "${INSTRUCT_MAX_MODEL_LEN}" \
        --generation-config vllm \
        > "${INSTRUCT_LOG}" 2>&1 &
    INSTRUCT_PID=$!
    echo "[vLLM] Instruct PID=${INSTRUCT_PID}"
    wait_for_api "instruct" "${INSTRUCT_API}" "${INSTRUCT_PID}" "${INSTRUCT_LOG}"
    INSTRUCT_ARGS=(
        --instruct-api-base "${INSTRUCT_API}"
        --instruct-api-model "${INSTRUCT_SERVED_MODEL}"
        --instruct-tokenizer-path "${INSTRUCT_TOKENIZER_PATH}"
    )
elif [[ "${INSTRUCT_BACKEND}" == "vllm_engine" ]]; then
    echo "[Local] Using local vLLM engine instruct model on cuda:0"
    INSTRUCT_ARGS=(
        --instruct-use-vllm-engine
        --instruct-model-path "${INSTRUCT_MODEL}"
        --instruct-tokenizer-path "${INSTRUCT_TOKENIZER_PATH}"
        --gpu-memory-utilization "${INSTRUCT_GPU_UTIL}"
        --instruct-vllm-max-model-len "${INSTRUCT_MAX_MODEL_LEN}"
    )
else
    echo "[ERROR] Unsupported INSTRUCT_BACKEND=${INSTRUCT_BACKEND}"
    exit 1
fi

PY_VISIBLE_DEVICES=""
if [[ "${INSTRUCT_BACKEND}" == "vllm_engine" ]]; then
    PY_VISIBLE_DEVICES="0"
fi

echo "[vLLM] Starting primary base server on allocation-local GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 "${RUN_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${PRIMARY_BASE_MODEL}" \
    --served-model-name "${PRIMARY_BASE_SERVED_MODEL}" \
    --port "${PRIMARY_BASE_PORT}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --gpu-memory-utilization "${PRIMARY_BASE_GPU_UTIL}" \
    --max-model-len "${PRIMARY_BASE_MAX_MODEL_LEN}" \
    --language-model-only \
    --generation-config vllm \
    > "${PRIMARY_BASE_LOG}" 2>&1 &
PRIMARY_BASE_PID=$!
echo "[vLLM] Primary base PID=${PRIMARY_BASE_PID}"
wait_for_api "primary_base" "${PRIMARY_BASE_API}" "${PRIMARY_BASE_PID}" "${PRIMARY_BASE_LOG}"

echo "[vLLM] Starting secondary base server on allocation-local GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 "${RUN_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${SECONDARY_BASE_MODEL}" \
    --served-model-name "${SECONDARY_BASE_SERVED_MODEL}" \
    --port "${SECONDARY_BASE_PORT}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --gpu-memory-utilization "${SECONDARY_BASE_GPU_UTIL}" \
    --max-model-len "${SECONDARY_BASE_MAX_MODEL_LEN}" \
    --generation-config vllm \
    > "${SECONDARY_BASE_LOG}" 2>&1 &
SECONDARY_BASE_PID=$!
echo "[vLLM] Secondary base PID=${SECONDARY_BASE_PID}"
wait_for_api "secondary_base" "${SECONDARY_BASE_API}" "${SECONDARY_BASE_PID}" "${SECONDARY_BASE_LOG}"

echo "[RUN] Launching consensus decoding ..."
if [[ -n "${PY_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="${PY_VISIBLE_DEVICES}"
fi
"${RUN_PYTHON}" "${SCRIPT_PATH}" \
    --input-tsv "${INPUT_TSV}" \
    --base-model-path "${PRIMARY_BASE_MODEL}" \
    --base-api-base "${PRIMARY_BASE_API}" \
    --base-api-model "${PRIMARY_BASE_SERVED_MODEL}" \
    --secondary-base-model-path "${SECONDARY_BASE_MODEL}" \
    --secondary-base-api-base "${SECONDARY_BASE_API}" \
    --secondary-base-api-model "${SECONDARY_BASE_SERVED_MODEL}" \
    --num-futures "${NUM_FUTURES}" \
    --secondary-num-futures "${SECONDARY_NUM_FUTURES}" \
    --future-tokens "${FUTURE_TOKENS}" \
    --max-consensus-steps "${MAX_CONSENSUS_STEPS}" \
    --min-p "${MIN_P}" \
    "${INSTRUCT_ARGS[@]}" \
    --row-idx "${ROW_START}" \
    --max-rows "${ROW_COUNT}" \
    --verbose \
    --verbose-dir "${VERBOSE_DIR}" \
    --output-jsonl "${TASK_OUT_DIR}/results.jsonl" \
    2>&1 | tee "${RUN_LOG}"

{
    echo "completed_at=$(date)"
    echo "rows_start=${ROW_START}"
    echo "rows_count=${ROW_COUNT}"
    echo "num_futures=${NUM_FUTURES}"
    echo "secondary_num_futures=${SECONDARY_NUM_FUTURES}"
} > "${RESULT_SENTINEL}"

echo "===== DONE DUAL-BASE CONSENSUS TASK ${TASK_ID} ====="
