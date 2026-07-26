#!/usr/bin/env bash
# Array-job version of thinking_policy throughput trial:
#   - array task 0: controller + shared base model + simalign
#   - array tasks 1-7: one thinking-model vLLM server per task
#
# Submit:
#   sbatch run_thinking_policy_pool_10_array.sh

#SBATCH --job-name=think_pool_10_arr
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/think_pool_arr_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/think_pool_arr_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at $HOME/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi
conda activate vllm

export HF_HOME="/data/user_data/haolingp/hf_cache"

OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_pool_7srv_10utt_array"
STATE_DIR="${OUTPUT_ROOT}/array_job_${SLURM_ARRAY_JOB_ID}"
INTERNAL_LOG_DIR="${OUTPUT_ROOT}/task_logs"
TASK_LOG="${INTERNAL_LOG_DIR}/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
SCRIPT_PATH="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy.py"
MANIFEST_PATH="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
BASE_MODEL_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"
THINKING_MODEL_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Thinking-2507-FP8"
THINKING_MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${STATE_DIR}"
mkdir -p "${INTERNAL_LOG_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs

exec > >(tee -a "${TASK_LOG}") 2>&1

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "ERROR: MANIFEST not found: ${MANIFEST_PATH}"
  exit 1
fi
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: Script not found: ${SCRIPT_PATH}"
  exit 1
fi

echo "===== THINKING-POOL ARRAY JOB ====="
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-N/A} task_id=${SLURM_ARRAY_TASK_ID:-N/A} node=$(hostname) time=$(date)"
echo "task_log=${TASK_LOG}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  CONTROLLER_SUCCESS_FILE="${STATE_DIR}/controller.success"
  CONTROLLER_FAILED_FILE="${STATE_DIR}/controller.failed"
  CONTROLLER_DONE_FILE="${STATE_DIR}/controller.done"
  rm -f "${CONTROLLER_SUCCESS_FILE}" "${CONTROLLER_FAILED_FILE}" "${CONTROLLER_DONE_FILE}"

  controller_exit() {
    local status=$?
    trap - EXIT
    if [[ "${status}" == "0" ]]; then
      touch "${CONTROLLER_SUCCESS_FILE}"
      echo "[Controller] success"
    else
      touch "${CONTROLLER_FAILED_FILE}"
      echo "[Controller] failed with exit code ${status}"
    fi
    touch "${CONTROLLER_DONE_FILE}"
    exit "${status}"
  }
  trap controller_exit EXIT

  echo "[Controller] waiting for 7 thinking servers ..."
  ready_count=0
  for _ in $(seq 1 1800); do
    ready_count=$(find "${STATE_DIR}" -maxdepth 1 -type f -name 'server_*.ready' | wc -l | tr -d ' ')
    failed_count=$(find "${STATE_DIR}" -maxdepth 1 -type f -name 'server_*.failed' | wc -l | tr -d ' ')
    if [[ "${failed_count}" != "0" ]]; then
      echo "ERROR: at least one thinking server failed before readiness"
      find "${STATE_DIR}" -maxdepth 1 -type f -name 'server_*.failed' | sort
      exit 1
    fi
    if [[ "${ready_count}" == "7" ]]; then
      break
    fi
    sleep 2
  done

  if [[ "${ready_count}" != "7" ]]; then
    echo "ERROR: timed out waiting for all 7 thinking servers"
    find "${STATE_DIR}" -maxdepth 1 -type f | sort
    exit 1
  fi

  declare -a THINKING_BASES=()
  for task_id in 1 2 3 4 5 6 7; do
    endpoint_file="${STATE_DIR}/server_${task_id}.endpoint"
    if [[ ! -f "${endpoint_file}" ]]; then
      echo "ERROR: missing endpoint file ${endpoint_file}"
      exit 1
    fi
    endpoint="$(cat "${endpoint_file}")"
    THINKING_BASES+=("${endpoint}")
  done

  THINKING_API_BASES_CSV="$(IFS=,; echo "${THINKING_BASES[*]}")"
  echo "[Controller] thinking_api_bases=${THINKING_API_BASES_CSV}"

  echo "[Controller] checking remote health endpoints ..."
  for endpoint in "${THINKING_BASES[@]}"; do
    health_url="${endpoint%/v1}/health"
    ok=0
    for _ in $(seq 1 60); do
      if curl -s "${health_url}" > /dev/null 2>&1; then
        ok=1
        break
      fi
      sleep 2
    done
    if [[ "${ok}" != "1" ]]; then
      echo "ERROR: remote health check failed for ${endpoint}"
      exit 1
    fi
  done

  echo "[Controller] running thinking_policy ..."
  SIMALIGN_MODEL="/data/user_data/haolingp/models/LaBSE" CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_PATH}" \
    --input-tsv "${MANIFEST_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --task-id 0 \
    --num-tasks 1 \
    --max-rows 10 \
    --base-model-path "${BASE_MODEL_PATH}" \
    --thinking-api-bases "${THINKING_API_BASES_CSV}" \
    --thinking-model-name "${THINKING_MODEL_NAME}" \
    --parallel-utterances 10 \
    --future-sampling-batch-size 4 \
    --future-sampling-batch-wait 0.05 \
    --num-futures 5 \
    --future-tokens 10 \
    --sample-temperature 1.0 \
    --thinking-temperature 0.1 \
    --thinking-max-tokens 16384 \
    --overwrite

  echo "===== CONTROLLER DONE ====="
  exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
PORT=$((8100 + TASK_ID))
LOG_FILE="${OUTPUT_ROOT}/thinking_server_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"
READY_FILE="${STATE_DIR}/server_${TASK_ID}.ready"
FAIL_FILE="${STATE_DIR}/server_${TASK_ID}.failed"
ENDPOINT_FILE="${STATE_DIR}/server_${TASK_ID}.endpoint"
HOSTNAME_FULL="$(hostname -f 2>/dev/null || hostname)"

rm -f "${READY_FILE}" "${FAIL_FILE}" "${ENDPOINT_FILE}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[Server ${TASK_ID}] starting on ${HOSTNAME_FULL}:${PORT}"

CUDA_VISIBLE_DEVICES=0 vllm serve "${THINKING_MODEL_PATH}" \
  --served-model-name "${THINKING_MODEL_NAME}" \
  --reasoning-parser qwen3 \
  --dtype auto \
  --port "${PORT}" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!

ready=0
for i in $(seq 1 300); do
  if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    printf 'http://%s:%s/v1\n' "${HOSTNAME_FULL}" "${PORT}" > "${ENDPOINT_FILE}"
    touch "${READY_FILE}"
    echo "[Server ${TASK_ID}] ready after ~${i}s -> $(cat "${ENDPOINT_FILE}")"
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    touch "${FAIL_FILE}"
    echo "ERROR: server ${TASK_ID} died before readiness; check ${LOG_FILE}"
    exit 1
  fi
  sleep 1
done

if [[ "${ready}" != "1" ]]; then
  touch "${FAIL_FILE}"
  echo "ERROR: server ${TASK_ID} not ready after timeout; check ${LOG_FILE}"
  exit 1
fi

echo "[Server ${TASK_ID}] waiting for controller.done ..."
while true; do
  if [[ -f "${STATE_DIR}/controller.done" ]]; then
    if [[ -f "${STATE_DIR}/controller.failed" ]]; then
      echo "[Server ${TASK_ID}] controller.failed detected; stopping after controller error"
    else
      echo "[Server ${TASK_ID}] controller.done detected; stopping"
    fi
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    touch "${FAIL_FILE}"
    echo "ERROR: server ${TASK_ID} exited unexpectedly; check ${LOG_FILE}"
    exit 1
  fi
  sleep 5
done

exit 0
