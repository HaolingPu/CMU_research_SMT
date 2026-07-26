#!/usr/bin/env bash
# =============================================================================
# 1000-case consensus decoding: 2 shared base + 6 instruct on a single 8-GPU node.
#
# GPU layout:
#   GPU 0   -> Gemma 4 E2B base  (shared primary future sampler)
#   GPU 1   -> Qwen3-4B-Base     (shared secondary future sampler)
#   GPU 2-7 -> 6× Qwen3-30B-A3B-Instruct-FP8 (one per worker)
#
# 6 Python workers each get their own instruct server, all share the 2 base servers.
# =============================================================================
#SBATCH --job-name=cons_6inst_1k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_6inst_1k_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_6inst_1k_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START 6-INSTRUCT CONSENSUS (1000 rows) ====="
echo "job_id=${SLURM_JOB_ID:-NA} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"

GEMMA4_ENV="${GEMMA4_ENV:-/data/user_data/haolingp/conda_envs/gemma4}"
RUN_PYTHON="${GEMMA4_ENV}/bin/python"

SCRIPT_PATH="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/consensus_decoding_token_id_level.py"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
PRIMARY_BASE_MODEL="${PRIMARY_BASE_MODEL:-google/gemma-4-E2B}"
SECONDARY_BASE_MODEL="${SECONDARY_BASE_MODEL:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
INSTRUCT_TOKENIZER_PATH="${INSTRUCT_TOKENIZER_PATH:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/gemma4/minp005-1k-6inst}"

TOTAL_ROWS="${TOTAL_ROWS:-1000}"
NUM_WORKERS=6
NUM_FUTURES="${NUM_FUTURES:-20}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-32}"
MIN_P="${MIN_P:-0.05}"

PRIMARY_BASE_GPU_UTIL="${PRIMARY_BASE_GPU_UTIL:-0.45}"
SECONDARY_BASE_GPU_UTIL="${SECONDARY_BASE_GPU_UTIL:-0.45}"
INSTRUCT_GPU_UTIL="${INSTRUCT_GPU_UTIL:-0.90}"

PRIMARY_BASE_MAX_MODEL_LEN="${PRIMARY_BASE_MAX_MODEL_LEN:-4096}"
SECONDARY_BASE_MAX_MODEL_LEN="${SECONDARY_BASE_MAX_MODEL_LEN:-4096}"
INSTRUCT_MAX_MODEL_LEN="${INSTRUCT_MAX_MODEL_LEN:-8192}"

INSTRUCT_SERVED_MODEL="${INSTRUCT_SERVED_MODEL:-qwen3-instruct}"
PRIMARY_BASE_SERVED_MODEL="${PRIMARY_BASE_SERVED_MODEL:-gemma4-e2b-base}"
SECONDARY_BASE_SERVED_MODEL="${SECONDARY_BASE_SERVED_MODEL:-qwen3-4b-base}"

# --- Ports ---
PRIMARY_BASE_PORT=8300
SECONDARY_BASE_PORT=8301
# Instruct servers on ports 8310-8315 (one per worker)

PRIMARY_BASE_API="http://127.0.0.1:${PRIMARY_BASE_PORT}/v1"
SECONDARY_BASE_API="http://127.0.0.1:${SECONDARY_BASE_PORT}/v1"

RUN_ROOT="${OUTPUT_ROOT}/job_${SLURM_JOB_ID:-manual}"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs
mkdir -p "${RUN_ROOT}"

echo "Input TSV              : ${INPUT_TSV}"
echo "Output root            : ${RUN_ROOT}"
echo "Total rows             : ${TOTAL_ROWS}"
echo "Workers                : ${NUM_WORKERS}"
echo "Futures                : total=${NUM_FUTURES} secondary=${SECONDARY_NUM_FUTURES}"
echo "Future tokens          : ${FUTURE_TOKENS}"
echo "Min-p                  : ${MIN_P}"
echo "Instruct model         : ${INSTRUCT_MODEL}"
echo "Primary base model     : ${PRIMARY_BASE_MODEL}"
echo "Secondary base model   : ${SECONDARY_BASE_MODEL}"

# --- Cleanup ---
ALL_SERVER_PIDS=()
ALL_WORKER_PIDS=()

cleanup() {
    echo "[CLEANUP] Stopping all servers and workers ..."
    for pid in "${ALL_WORKER_PIDS[@]}" "${ALL_SERVER_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${ALL_WORKER_PIDS[@]}" "${ALL_SERVER_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
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

# =============================================================================
# 1. Start shared base servers (GPU 0, GPU 1)
# =============================================================================
echo "[vLLM] Starting primary base server on GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 "${RUN_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${PRIMARY_BASE_MODEL}" \
    --served-model-name "${PRIMARY_BASE_SERVED_MODEL}" \
    --port "${PRIMARY_BASE_PORT}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --gpu-memory-utilization "${PRIMARY_BASE_GPU_UTIL}" \
    --max-model-len "${PRIMARY_BASE_MAX_MODEL_LEN}" \
    --language-model-only \
    --generation-config vllm \
    > "${RUN_ROOT}/primary_base_server.log" 2>&1 &
PRIMARY_BASE_PID=$!
ALL_SERVER_PIDS+=("${PRIMARY_BASE_PID}")
echo "[vLLM] Primary base PID=${PRIMARY_BASE_PID}"

echo "[vLLM] Starting secondary base server on GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 "${RUN_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${SECONDARY_BASE_MODEL}" \
    --served-model-name "${SECONDARY_BASE_SERVED_MODEL}" \
    --port "${SECONDARY_BASE_PORT}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --gpu-memory-utilization "${SECONDARY_BASE_GPU_UTIL}" \
    --max-model-len "${SECONDARY_BASE_MAX_MODEL_LEN}" \
    --generation-config vllm \
    > "${RUN_ROOT}/secondary_base_server.log" 2>&1 &
SECONDARY_BASE_PID=$!
ALL_SERVER_PIDS+=("${SECONDARY_BASE_PID}")
echo "[vLLM] Secondary base PID=${SECONDARY_BASE_PID}"

wait_for_api "primary_base" "${PRIMARY_BASE_API}" "${PRIMARY_BASE_PID}" "${RUN_ROOT}/primary_base_server.log"
wait_for_api "secondary_base" "${SECONDARY_BASE_API}" "${SECONDARY_BASE_PID}" "${RUN_ROOT}/secondary_base_server.log"

# =============================================================================
# 2. Start 6 instruct servers (GPU 2-7)
# =============================================================================
for w in $(seq 0 $(( NUM_WORKERS - 1 ))); do
    GPU_ID=$(( w + 2 ))
    INST_PORT=$(( 8310 + w ))
    INST_LOG="${RUN_ROOT}/instruct_server_${w}.log"

    echo "[vLLM] Starting instruct server ${w} on GPU ${GPU_ID}, port ${INST_PORT} ..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${RUN_PYTHON}" -m vllm.entrypoints.openai.api_server \
        --model "${INSTRUCT_MODEL}" \
        --served-model-name "${INSTRUCT_SERVED_MODEL}" \
        --port "${INST_PORT}" \
        --tensor-parallel-size 1 \
        --dtype auto \
        --gpu-memory-utilization "${INSTRUCT_GPU_UTIL}" \
        --max-model-len "${INSTRUCT_MAX_MODEL_LEN}" \
        --generation-config vllm \
        > "${INST_LOG}" 2>&1 &
    INST_PID=$!
    ALL_SERVER_PIDS+=("${INST_PID}")
    echo "[vLLM] Instruct ${w} PID=${INST_PID}"
done

# Wait for all instruct servers to be ready
for w in $(seq 0 $(( NUM_WORKERS - 1 ))); do
    INST_PORT=$(( 8310 + w ))
    INST_API="http://127.0.0.1:${INST_PORT}/v1"
    INST_LOG="${RUN_ROOT}/instruct_server_${w}.log"
    # Find the PID (offset: 2 base + w instruct)
    INST_PID="${ALL_SERVER_PIDS[$(( 2 + w ))]}"
    wait_for_api "instruct_${w}" "${INST_API}" "${INST_PID}" "${INST_LOG}"
done

# =============================================================================
# 3. Launch 6 Python workers in parallel
# =============================================================================
ROWS_PER_WORKER=$(( (TOTAL_ROWS + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "[RUN] Launching ${NUM_WORKERS} consensus workers ..."
for w in $(seq 0 $(( NUM_WORKERS - 1 ))); do
    ROW_START=$(( w * ROWS_PER_WORKER ))
    REMAINING=$(( TOTAL_ROWS - ROW_START ))
    if (( REMAINING <= 0 )); then
        echo "[SKIP] worker ${w} has no assigned rows."
        continue
    fi
    ROW_COUNT="${ROWS_PER_WORKER}"
    if (( REMAINING < ROW_COUNT )); then
        ROW_COUNT="${REMAINING}"
    fi

    INST_PORT=$(( 8310 + w ))
    INST_API="http://127.0.0.1:${INST_PORT}/v1"
    TASK_OUT_DIR="${RUN_ROOT}/task_${w}"
    VERBOSE_DIR="${TASK_OUT_DIR}/verbose"
    mkdir -p "${TASK_OUT_DIR}" "${VERBOSE_DIR}"

    echo "[RUN] Worker ${w}: rows ${ROW_START}..$(( ROW_START + ROW_COUNT - 1 )), instruct port ${INST_PORT}"

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
        --instruct-api-base "${INST_API}" \
        --instruct-api-model "${INSTRUCT_SERVED_MODEL}" \
        --instruct-tokenizer-path "${INSTRUCT_TOKENIZER_PATH}" \
        --row-idx "${ROW_START}" \
        --max-rows "${ROW_COUNT}" \
        --verbose \
        --verbose-dir "${VERBOSE_DIR}" \
        --output-jsonl "${TASK_OUT_DIR}/results.jsonl" \
        > "${TASK_OUT_DIR}/run.log" 2>&1 &
    WORKER_PID=$!
    ALL_WORKER_PIDS+=("${WORKER_PID}")
    echo "[RUN] Worker ${w} PID=${WORKER_PID}"
done

# =============================================================================
# 4. Wait for all workers to finish
# =============================================================================
echo "[WAIT] Waiting for ${#ALL_WORKER_PIDS[@]} workers to finish ..."
FAIL=0
for i in "${!ALL_WORKER_PIDS[@]}"; do
    pid="${ALL_WORKER_PIDS[$i]}"
    if ! wait "${pid}"; then
        echo "[FAIL] Worker ${i} (PID=${pid}) exited with error."
        FAIL=$(( FAIL + 1 ))
    else
        echo "[DONE] Worker ${i} (PID=${pid}) finished successfully."
    fi
done

# Write sentinel
{
    echo "completed_at=$(date)"
    echo "total_rows=${TOTAL_ROWS}"
    echo "num_workers=${NUM_WORKERS}"
    echo "failed_workers=${FAIL}"
} > "${RUN_ROOT}/DONE.txt"

if (( FAIL > 0 )); then
    echo "===== FINISHED WITH ${FAIL} FAILED WORKER(S) ====="
    exit 1
fi

echo "===== DONE 6-INSTRUCT CONSENSUS (1000 rows) ====="
