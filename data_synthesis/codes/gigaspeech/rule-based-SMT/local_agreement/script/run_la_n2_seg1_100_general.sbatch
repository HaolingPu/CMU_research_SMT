#!/usr/bin/env bash
#SBATCH --job-name=la2_seg1_100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_n2_seg1_100_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_n2_seg1_100_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

# Smoke run: Local Agreement LA-N=2, source trajectory segment size=1, first 100 cases.

JOB_ID="${SLURM_JOB_ID:-manual}"

LA_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement"
FS_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${FS_DIR}/serve_instruct_gpu0.sh"
DECODER="${LA_DIR}/local_agreement.py"
TOKENIZER="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
OUTPUT_ROOT="${LA_DIR}/output/la_n2_seg1_100_general"

ROW_START=0
ROW_COUNT=100
LA_N=2
SEGMENT_SIZE=1
LCP_MODE=char
MAX_NEW_TOKENS=1024
NUM_CONCURRENT_CASES=16
TARGET_LANG=Chinese
INSTRUCT_PORT=8800

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

RUN_DIR="${OUTPUT_ROOT}/job_${JOB_ID}"
LOG_DIR="${OUTPUT_ROOT}/serve_logs/job_${JOB_ID}"
DONE_FILE="${RUN_DIR}/DONE.txt"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

INSTRUCT_PID="/tmp/la_n2_seg1_100_instruct_${JOB_ID}.pid"

stop_servers() {
  set +e
  PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_instruct.log" 2>&1
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

echo "===== LA-N=2 seg=1 first 100 cases ====="
echo "job=${JOB_ID} node=$(hostname)"
echo "rows: start=${ROW_START} count=${ROW_COUNT}"
echo "output=${RUN_DIR}"

PORT="${INSTRUCT_PORT}" PID_FILE="${INSTRUCT_PID}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct.log" 2>&1 &

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
  --segment-size "${SEGMENT_SIZE}"
  --lcp-mode "${LCP_MODE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --num-concurrent-cases "${NUM_CONCURRENT_CASES}"
  --overwrite
  --output-dir "${RUN_DIR}"
)

"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
CMD_STATUS="${PIPESTATUS[0]}"
if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] local agreement failed with exit code ${CMD_STATUS}"
  exit "${CMD_STATUS}"
fi

JSON_COUNT=$(find "${RUN_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
if [[ "${JSON_COUNT}" != "${ROW_COUNT}" ]]; then
  echo "[ERROR] incomplete output: json_count=${JSON_COUNT}, expected=${ROW_COUNT}"
  exit 1
fi

{
  echo "completed_at=$(date)"
  echo "rows_start=${ROW_START}"
  echo "rows_count=${ROW_COUNT}"
  echo "la_n=${LA_N}"
  echo "segment_size=${SEGMENT_SIZE}"
} > "${DONE_FILE}"

echo "===== done ====="
