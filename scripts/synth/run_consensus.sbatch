#!/usr/bin/env bash
#SBATCH --job-name=consensus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-t9-16,babel-o5-16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -o slurm_logs/consensus_%A_%a.out
#SBATCH -e slurm_logs/consensus_%A_%a.err

# Run token-ID-level consensus decoding via the refactored package
# (data_synthesis/codes-refactored/consensus_decoding/) on a slice of rows
# from the GigaSpeech case-robust manifest. The array index shards rows
# across tasks. Each task spins up its own instruct + dual-base vLLM
# servers on per-task ports, runs the decoder, then tears them down.
#
# Override knobs by exporting env vars before sbatch, e.g.:
#   TOTAL_ROWS=10000 NUM_TASKS=4 sbatch scripts/synth/run_consensus.sbatch
#   POLICY=minp MIN_P=0.05 sbatch scripts/synth/run_consensus.sbatch
#   POLICY=topp TOP_P=0.9  sbatch scripts/synth/run_consensus.sbatch

set -e

source /home/siqiouya/miniconda3/bin/activate consensus

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${OUTPUT_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"

# === Paths ===
REPO_ROOT="/home/siqiouya/code/CMU_research_SMT"
SERVE_DIR="${REPO_ROOT}/data_synthesis/codes/gigaspeech/future_sampling"
CONDA_ENV="/home/siqiouya/miniconda3/envs/consensus"
PYTHON="${CONDA_ENV}/bin/python"
VLLM="${CONDA_ENV}/bin/vllm"

SERVE_INSTRUCT="${SERVE_DIR}/serve_instruct_gpu0.sh"
SERVE_DUALBASE="${SERVE_DIR}/serve_dual_base_gpu1.sh"
DECODER="${REPO_ROOT}/data_synthesis/codes-refactored/consensus_decoding_token_id_level.py"
TOKENIZER="/data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/haolingp/manifest/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv}"

# === Sharding ===
TOTAL_ROWS="${TOTAL_ROWS:-25000}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_TASKS="${NUM_TASKS:-4}"

# === Decoding policy ===
POLICY="${POLICY:-topk}"           # topk | minp | topp
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-5}"
MIN_P="${MIN_P:-0.0}"
TOP_P="${TOP_P:-0.0}"

# === Sampling / consensus ===
NUM_FUTURES="${NUM_FUTURES:-200}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-100}"
FUTURE_TOKENS="${FUTURE_TOKENS:-20}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-20}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-32}"
TARGET_LANG="${TARGET_LANG:-Chinese}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENABLE_VERBOSE="${ENABLE_VERBOSE:-0}"

# === Output ===
RUN_NAME="${RUN_NAME:-consensus_${POLICY}_topk${CANDIDATE_TOP_K}_minp${MIN_P}_topp${TOP_P}_f${NUM_FUTURES}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/consensus_decoding/${RUN_NAME}}"

# === HF / tokenizer cache ===
export HF_HOME="${HF_HOME:-/data/user_data/siqiouya/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="false"
export PATH="${CONDA_ENV}/bin:${PATH}"

# === Per-task row range ===
ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
ROW_START=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))
ROW_COUNT="${ROWS_PER_TASK}"
LOCAL_REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
if (( LOCAL_REMAINING <= 0 )); then
  echo "[SKIP] task ${TASK_ID} has no assigned rows"
  exit 0
fi
if (( LOCAL_REMAINING < ROW_COUNT )); then
  ROW_COUNT="${LOCAL_REMAINING}"
fi

# === Per-task vLLM ports (avoid collisions across array tasks) ===
PORT_BASE=$(( 8400 + TASK_ID * 10 ))
INSTRUCT_PORT=$(( PORT_BASE + 0 ))
GEMMA_PORT=$(( PORT_BASE + 1 ))
QWEN_PORT=$(( PORT_BASE + 2 ))

RUN_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/task_${TASK_ID}"
LOG_DIR="${OUTPUT_ROOT}/job_${JOB_ID}/serve_logs"
DONE_FILE="${RUN_DIR}/DONE.txt"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

INSTRUCT_PID="/tmp/consensus_${POLICY}_instruct_${JOB_ID}_${TASK_ID}.pid"
GEMMA_PID="/tmp/consensus_${POLICY}_gemma_${JOB_ID}_${TASK_ID}.pid"
QWEN_PID="/tmp/consensus_${POLICY}_qwen_${JOB_ID}_${TASK_ID}.pid"

stop_servers() {
  set +e
  PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_instruct_${TASK_ID}.log" 2>&1
  GEMMA_PORT="${GEMMA_PORT}" GEMMA_PID_FILE="${GEMMA_PID}" \
  QWEN_PORT="${QWEN_PORT}" QWEN_PID_FILE="${QWEN_PID}" \
    bash "${SERVE_DUALBASE}" stop > "${LOG_DIR}/stop_dualbase_${TASK_ID}.log" 2>&1
}
trap stop_servers EXIT

wait_health() {
  name="$1"; url="$2"
  for i in $(seq 1 300); do
    if curl -s "${url}" > /dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] ${name} not ready: ${url}"
  return 1
}

if [[ -f "${DONE_FILE}" ]]; then
  echo "[SKIP] ${DONE_FILE} exists"
  exit 0
fi

echo "===== consensus task ${TASK_ID} START $(date) ====="
echo "job=${JOB_ID} node=$(hostname)"
echo "policy=${POLICY}  top_k=${CANDIDATE_TOP_K}  min_p=${MIN_P}  top_p=${TOP_P}"
echo "rows: start=${ROW_START} count=${ROW_COUNT} (total=${TOTAL_ROWS} offset=${ROW_OFFSET})"
echo "futures: total=${NUM_FUTURES}  secondary=${SECONDARY_NUM_FUTURES} -> primary=$((NUM_FUTURES - SECONDARY_NUM_FUTURES))"
echo "max_consensus_steps=${MAX_CONSENSUS_STEPS}  concurrent_cases=${NUM_CONCURRENT_CASES}"
echo "decoder=${DECODER}"
echo "output=${RUN_DIR}"

# Start the three vLLM servers in the background.
PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &

GEMMA_PORT="${GEMMA_PORT}" GEMMA_PID_FILE="${GEMMA_PID}" \
QWEN_PORT="${QWEN_PORT}" QWEN_PID_FILE="${QWEN_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_DUALBASE}" > "${LOG_DIR}/serve_dualbase_${TASK_ID}.log" 2>&1 &

wait_health "instruct"      "http://127.0.0.1:${INSTRUCT_PORT}/health"
wait_health "gemma4-e2b"    "http://127.0.0.1:${GEMMA_PORT}/health"
wait_health "qwen3-4b-base" "http://127.0.0.1:${QWEN_PORT}/health"

CMD=(
  "${PYTHON}" "${DECODER}"
  --input-tsv "${INPUT_TSV}"
  --base-api-base "http://127.0.0.1:${GEMMA_PORT}/v1"
  --base-api-model "gemma4-e2b"
  --secondary-base-api-base "http://127.0.0.1:${QWEN_PORT}/v1"
  --secondary-base-api-model "qwen3-4b-base"
  --instruct-api-base "http://127.0.0.1:${INSTRUCT_PORT}/v1"
  --instruct-api-model "qwen3-instruct"
  --instruct-tokenizer-path "${TOKENIZER}"
  --target-lang "${TARGET_LANG}"
  --row-idx "${ROW_START}"
  --max-rows "${ROW_COUNT}"
  --num-futures "${NUM_FUTURES}"
  --secondary-num-futures "${SECONDARY_NUM_FUTURES}"
  --future-tokens "${FUTURE_TOKENS}"
  --max-consensus-steps "${MAX_CONSENSUS_STEPS}"
  --candidate-top-k "${CANDIDATE_TOP_K}"
  --min-p "${MIN_P}"
  --top-p "${TOP_P}"
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}"
  --output-jsonl "${RUN_DIR}/results.jsonl"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip-existing)
fi
if [[ "${ENABLE_VERBOSE}" == "1" ]]; then
  mkdir -p "${RUN_DIR}/verbose"
  CMD+=(--verbose --verbose-dir "${RUN_DIR}/verbose")
fi

"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
CMD_STATUS="${PIPESTATUS[0]}"
if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] decoder failed with exit code ${CMD_STATUS}"
  exit "${CMD_STATUS}"
fi

JSON_COUNT=$(find "${RUN_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
echo "[INFO] task ${TASK_ID} json_count=${JSON_COUNT} expected=${ROW_COUNT}"
if [[ "${JSON_COUNT}" -lt "${ROW_COUNT}" ]]; then
  echo "[WARN] partial completion. Resubmit the SAME sbatch (with the same ROW_OFFSET);"
  echo "       --skip-existing will pick up where this left off."
  exit 0
fi

{
  echo "completed_at=$(date)"
  echo "policy=${POLICY}"
  echo "row_offset=${ROW_OFFSET}"
  echo "rows_start=${ROW_START}"
  echo "rows_count=${ROW_COUNT}"
  echo "candidate_top_k=${CANDIDATE_TOP_K}"
  echo "min_p=${MIN_P}"
  echo "top_p=${TOP_P}"
  echo "num_futures=${NUM_FUTURES}"
  echo "secondary_num_futures=${SECONDARY_NUM_FUTURES}"
} > "${DONE_FILE}"

echo "===== consensus task ${TASK_ID} DONE $(date) ====="
