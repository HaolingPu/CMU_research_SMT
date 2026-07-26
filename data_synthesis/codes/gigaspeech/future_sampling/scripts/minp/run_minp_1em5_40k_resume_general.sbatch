#!/usr/bin/env bash
#SBATCH --job-name=minp_1em5_res
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --array=0-3
#SBATCH --time=1-10:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/minp_1em5_resume_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/minp_1em5_resume_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

# Resume min-p=1e-4 on general: write into merged/ so --skip-existing
# sees the 33,594 already-completed utterances.

MIN_P_VALUE="0.00001"
MIN_P_LABEL="1em5"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

CODE_DIR="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${CODE_DIR}/serve_instruct_gpu0.sh"
SERVE_DUALBASE="${CODE_DIR}/serve_dual_base_gpu1.sh"
DECODER="${CODE_DIR}/consensus_decoding_token_id_level.py"
TOKENIZER="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_${MIN_P_VALUE}"

# Write directly into the merged directory so skip-existing works
RUN_DIR="${OUTPUT_ROOT}/merged/task_merged"

TOTAL_ROWS=40000
ROW_OFFSET=0
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  NUM_TASKS="${SLURM_ARRAY_TASK_COUNT}"
else
  NUM_TASKS=4
fi
NUM_FUTURES=20
SECONDARY_NUM_FUTURES=10
FUTURE_TOKENS=20
MAX_CONSENSUS_STEPS=32
NUM_CONCURRENT_CASES=32
CANDIDATE_TOP_K=20
TARGET_LANG=Chinese

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
ROW_START=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))
ROW_COUNT="${ROWS_PER_TASK}"
REMAINING=$(( TOTAL_ROWS - TASK_ID * ROWS_PER_TASK ))
if (( REMAINING <= 0 )); then
  echo "[SKIP] task ${TASK_ID} has no assigned rows"
  exit 0
fi
if (( REMAINING < ROW_COUNT )); then
  ROW_COUNT="${REMAINING}"
fi

PORT_BASE=$(( 8100 + TASK_ID * 10 ))
INSTRUCT_PORT=$(( PORT_BASE + 0 ))
GEMMA_PORT=$(( PORT_BASE + 1 ))
QWEN_PORT=$(( PORT_BASE + 2 ))

LOG_DIR="${OUTPUT_ROOT}/resume_serve_logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

INSTRUCT_PID="/tmp/minp_${MIN_P_LABEL}_res_instruct_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"
GEMMA_PID="/tmp/minp_${MIN_P_LABEL}_res_gemma_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"
QWEN_PID="/tmp/minp_${MIN_P_LABEL}_res_qwen_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"

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
  name="$1"
  url="$2"
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

echo "===== RESUME min-p=${MIN_P_VALUE} task ${TASK_ID} ====="
echo "node=$(hostname)"
echo "rows: start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS}"
echo "output=${RUN_DIR} (shared merged dir, skip-existing enabled)"

PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &

GEMMA_PORT="${GEMMA_PORT}" GEMMA_PID_FILE="${GEMMA_PID}" \
QWEN_PORT="${QWEN_PORT}" QWEN_PID_FILE="${QWEN_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_DUALBASE}" > "${LOG_DIR}/serve_dualbase_${TASK_ID}.log" 2>&1 &

wait_health "instruct" "http://127.0.0.1:${INSTRUCT_PORT}/health"
wait_health "gemma4-e2b" "http://127.0.0.1:${GEMMA_PORT}/health"
wait_health "qwen3-4b-base" "http://127.0.0.1:${QWEN_PORT}/health"

"${PYTHON}" "${DECODER}" \
  --input-tsv "${INPUT_TSV}" \
  --base-api-base "http://127.0.0.1:${GEMMA_PORT}/v1" \
  --base-api-model "gemma4-e2b" \
  --secondary-base-api-base "http://127.0.0.1:${QWEN_PORT}/v1" \
  --secondary-base-api-model "qwen3-4b-base" \
  --instruct-api-base "http://127.0.0.1:${INSTRUCT_PORT}/v1" \
  --instruct-api-model "qwen3-instruct" \
  --instruct-tokenizer-path "${TOKENIZER}" \
  --target-lang "${TARGET_LANG}" \
  --row-idx "${ROW_START}" \
  --max-rows "${ROW_COUNT}" \
  --num-futures "${NUM_FUTURES}" \
  --secondary-num-futures "${SECONDARY_NUM_FUTURES}" \
  --future-tokens "${FUTURE_TOKENS}" \
  --max-consensus-steps "${MAX_CONSENSUS_STEPS}" \
  --candidate-top-k "${CANDIDATE_TOP_K}" \
  --min-p "${MIN_P_VALUE}" \
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}" \
  --output-jsonl "${RUN_DIR}/results.jsonl" \
  --skip-existing \
  2>&1 | tee "${LOG_DIR}/run_task_${TASK_ID}.log"

CMD_STATUS="${PIPESTATUS[0]}"
if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] decoder failed with exit code ${CMD_STATUS}"
  exit "${CMD_STATUS}"
fi

echo "===== done resume task ${TASK_ID} ====="
