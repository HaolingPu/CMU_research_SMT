#!/usr/bin/env bash
# =============================================================================
# run_consensus_decoding_vllm_100_8gpu.sbatch
#
# 100-case consensus decoding sweep with 4 array tasks.
# Each array task requests 1 node with 2x L40S GPUs:
#   GPU 0 -> vLLM serve Qwen3-30B-A3B-Instruct-2507-FP8
#   GPU 1 -> local Qwen3-4B-Base future sampling
#
# Default layout:
#   task 0 -> rows   0-24
#   task 1 -> rows  25-49
#   task 2 -> rows  50-74
#   task 3 -> rows  75-99
#
# Submit:
#   sbatch /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_vllm_100_8gpu.sbatch
#
# Optional overrides at submit time:
#   sbatch --export=ALL,TOTAL_ROWS=100,NUM_TASKS=4,INPUT_TSV=/path/to/file.tsv \
#     /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_consensus_decoding_vllm_100_8gpu.sbatch
# =============================================================================
#SBATCH --job-name=cons_dec_100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --time=0-10:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_dec_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/cons_dec_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== START CONSENSUS DECODING TASK ${SLURM_ARRAY_TASK_ID:-0} ====="
echo "job_id=${SLURM_JOB_ID:-NA}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"

CONDA_BASE="/home/haolingp/miniconda3"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"
BASE_PYTHON="${CONDA_BASE}/bin/python"

SCRIPT_PATH="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/consensus_decoding_token_id_level.py"
MODEL_30B="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/user_data/haolingp/models/Qwen3-4B-Base}"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug}"

TOTAL_ROWS="${TOTAL_ROWS:-100}"
NUM_TASKS="${NUM_TASKS:-4}"
NUM_FUTURES="${NUM_FUTURES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-24}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
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

VLLM_PORT=$(( 8100 + TASK_ID ))
VLLM_API="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen3-instruct"

RUN_ROOT="${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID:-manual}"
TASK_OUT_DIR="${RUN_ROOT}/task_${TASK_ID}"
VERBOSE_DIR="${TASK_OUT_DIR}/verbose"
RESULT_SENTINEL="${TASK_OUT_DIR}/DONE.txt"
VLLM_LOG="${TASK_OUT_DIR}/vllm_server.log"
RUN_LOG="${TASK_OUT_DIR}/run.log"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs
mkdir -p "${TASK_OUT_DIR}" "${VERBOSE_DIR}"

echo "Input TSV   : ${INPUT_TSV}"
echo "Output dir  : ${TASK_OUT_DIR}"
echo "Rows        : start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS}"
echo "Num futures : ${NUM_FUTURES}"
echo "Future toks : ${FUTURE_TOKENS}"
echo "Serve port  : ${VLLM_PORT}"
echo "Base model  : ${BASE_MODEL_PATH}"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -f "${RESULT_SENTINEL}" ]]; then
    echo "[SKIP] ${RESULT_SENTINEL} exists."
    exit 0
fi

echo "[vLLM] Starting 30B server on allocation-local GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_30B}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --generation-config vllm \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[vLLM] PID=${VLLM_PID}"

MAX_WAIT=360
WAITED=0
until curl -sS "${VLLM_API}/models" > /dev/null 2>&1; do
    sleep 5
    WAITED=$(( WAITED + 5 ))
    if (( WAITED >= MAX_WAIT )); then
        echo "[ERROR] vLLM server did not become ready within ${MAX_WAIT}s."
        tail -n 60 "${VLLM_LOG}" || true
        exit 1
    fi
done

echo "[vLLM] Ready after ${WAITED}s"

echo "[RUN] Launching consensus decoding ..."
"${BASE_PYTHON}" "${SCRIPT_PATH}" \
    --input-tsv "${INPUT_TSV}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --base-device cuda:1 \
    --instruct-api-base "${VLLM_API}" \
    --instruct-api-model "${VLLM_MODEL_NAME}" \
    --instruct-tokenizer-path "${MODEL_30B}" \
    --row-idx "${ROW_START}" \
    --max-rows "${ROW_COUNT}" \
    --num-futures "${NUM_FUTURES}" \
    --future-tokens "${FUTURE_TOKENS}" \
    --max-consensus-steps 32 \
    --verbose \
    --verbose-dir "${VERBOSE_DIR}" \
    --output-jsonl "${TASK_OUT_DIR}/results.jsonl" \
    2>&1 | tee "${RUN_LOG}"

echo "completed_at=$(date)" > "${RESULT_SENTINEL}"
echo "rows_start=${ROW_START}" >> "${RESULT_SENTINEL}"
echo "rows_count=${ROW_COUNT}" >> "${RESULT_SENTINEL}"
echo "===== DONE CONSENSUS DECODING TASK ${TASK_ID} ====="
