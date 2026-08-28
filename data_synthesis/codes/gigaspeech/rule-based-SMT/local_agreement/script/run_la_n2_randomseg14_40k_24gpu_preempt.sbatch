#!/usr/bin/env bash
#SBATCH --job-name=la2_rand40k_seg14
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-23
#SBATCH --time=1-10:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-p9-28,babel-s5-32,babel-m5-32,babel-n9-32
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_n2_randomseg14_40k_preempt_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_n2_randomseg14_40k_preempt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

# Local Agreement LA-N=2 with per-case random segment size sampled uniformly
# from 1..4 inside local_agreement.py (v3: widened from 1..3 to 1..4 + fixed
# the early-commit bug where window of 1 hyp was committed without agreement).
#
# 40k examples split across 24 array tasks (preempt partition):
#   task 0:  rows 0..1666
#   task 1:  rows 1667..3333
#   ...
#   task 23: rows ~38334..39999
#
# Re-run safety:
# - Each task writes one JSON per utterance under RUN_DIR.
# - No --overwrite is passed, so an interrupted/preempted task that
#   resubmits skips JSON files that already exist and continues missing cases.
# - DONE.txt is written only after the task has all expected JSON files.

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${OUTPUT_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"

LA_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement"
FS_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${FS_DIR}/serve_instruct_gpu0.sh"
DECODER="${LA_DIR}/local_agreement.py"
TOKENIZER="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k_seg14"

TOTAL_ROWS=40000
ROW_OFFSET=0
NUM_TASKS=24
LA_N=2
SEGMENT_SIZE_FALLBACK=1
LCP_MODE=char
MAX_NEW_TOKENS=1024
NUM_CONCURRENT_CASES=32
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

PORT_BASE=$(( 8400 + TASK_ID * 10 ))
INSTRUCT_PORT="${PORT_BASE}"

RUN_DIR="${OUTPUT_ROOT}/task_${TASK_ID}"
LOG_DIR="${OUTPUT_ROOT}/serve_logs"
DONE_FILE="${RUN_DIR}/DONE.txt"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

INSTRUCT_PID="/tmp/la_n2_randomseg14_40k_instruct_${JOB_ID}_${TASK_ID}.pid"

stop_servers() {
  set +e
  PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_instruct_${TASK_ID}.log" 2>&1
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

if [[ -f "${DONE_FILE}" ]]; then
  echo "[SKIP] ${DONE_FILE} exists"
  exit 0
fi

echo "===== LA-N=2 random-segment[1..4] 40k preempt task ${TASK_ID} ====="
echo "job=${JOB_ID} node=$(hostname)"
echo "partition=preempt qos=preempt_qos"
echo "rows: start=${ROW_START} count=${ROW_COUNT} total=${TOTAL_ROWS}"
echo "segment_size=random_uniform_1_4_per_case  (LA-bug fixed: window<la_n -> no commit)"
echo "output=${RUN_DIR}"

PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct_${TASK_ID}.log" 2>&1 &

wait_health "instruct" "http://127.0.0.1:${INSTRUCT_PORT}/health"

CMD=(
  "${PYTHON}" "${DECODER}"
  --input-tsv "${INPUT_TSV}"
  --mt-api-base "http://127.0.0.1:${INSTRUCT_PORT}/v1"
  --mt-api-model "qwen3-instruct"
  --mt-tokenizer-path "${TOKENIZER}"
  --target-lang "${TARGET_LANG}"
  --row-idx "${ROW_START}"
  --max-rows "${ROW_COUNT}"
  --la-n "${LA_N}"
  --segment-size "${SEGMENT_SIZE_FALLBACK}"
  --lcp-mode "${LCP_MODE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}"
  --output-dir "${RUN_DIR}"
)

"${CMD[@]}" 2>&1 | tee -a "${RUN_DIR}/run.log"
CMD_STATUS="${PIPESTATUS[0]}"
if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] local agreement failed with exit code ${CMD_STATUS}"
  exit "${CMD_STATUS}"
fi

EXPECTED_FILES="${ROW_COUNT}"
ACTUAL_FILES=$(ls -1 "${RUN_DIR}"/*.json 2>/dev/null | wc -l)
echo "EXPECTED ${EXPECTED_FILES} jsons, found ${ACTUAL_FILES}"
if (( ACTUAL_FILES < EXPECTED_FILES )); then
  echo "[WARN] missing $((EXPECTED_FILES - ACTUAL_FILES)) cases; not writing DONE.txt"
  exit 0
fi
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${DONE_FILE}"
echo "[DONE] task ${TASK_ID}"
