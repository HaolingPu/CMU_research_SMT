#!/usr/bin/env bash
# Serve Qwen3.8-27B-FP8 and Gemma-4-E2B-it on one L40S for future sampling.
# The defaults target BABEL's 46 GiB usable L40S and intentionally keep short
# contexts and small sequence pools. They are smoke-test values, not throughput
# settings; tune only after both models are proven stable together.

set -euo pipefail

GEMMA4_ENV="${GEMMA4_ENV:-/data/user_data/haolingp/conda_envs/gemma4}"
VLLM_BIN="${VLLM_BIN:-${GEMMA4_ENV}/bin/vllm}"
QWEN_MODEL="${QWEN_MODEL:-/data/user_data/haolingp/models/Qwen3.8-27B-FP8}"
GEMMA_MODEL="${GEMMA_MODEL:-/data/user_data/haolingp/models/gemma-4-E2B-it}"
GPU="${GPU:-0}"
QWEN_PORT="${QWEN_PORT:-8310}"
GEMMA_PORT="${GEMMA_PORT:-8311}"
QWEN_GPU_MEM_UTIL="${QWEN_GPU_MEM_UTIL:-0.70}"
GEMMA_GPU_MEM_UTIL="${GEMMA_GPU_MEM_UTIL:-0.23}"
QWEN_MAX_LEN="${QWEN_MAX_LEN:-4096}"
GEMMA_MAX_LEN="${GEMMA_MAX_LEN:-4096}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-16}"
GEMMA_MAX_NUM_SEQS="${GEMMA_MAX_NUM_SEQS:-16}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
LOG_DIR="${LOG_DIR:-/tmp/qwen38_gemma_${SLURM_JOB_ID:-manual}}"
QWEN_PID_FILE="${QWEN_PID_FILE:-/tmp/qwen38_${SLURM_JOB_ID:-manual}.pid}"
GEMMA_PID_FILE="${GEMMA_PID_FILE:-/tmp/gemma4_${SLURM_JOB_ID:-manual}.pid}"

export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

stop_servers() {
  set +e
  for pid_file in "${QWEN_PID_FILE}" "${GEMMA_PID_FILE}"; do
    if [[ -f "${pid_file}" ]]; then
      pid=$(cat "${pid_file}")
      kill "${pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${pid}" 2>/dev/null || true
      rm -f "${pid_file}"
    fi
  done
  for port in "${QWEN_PORT}" "${GEMMA_PORT}"; do
    port_pid=$(lsof -ti :"${port}" 2>/dev/null || true)
    [[ -z "${port_pid}" ]] || kill ${port_pid} 2>/dev/null || true
  done
}

if [[ "${1:-}" == "stop" ]]; then
  stop_servers
  exit 0
fi

wait_health() {
  local name=$1
  local port=$2
  local pid_file=$3
  for ((i = 1; i <= READY_TIMEOUT; i++)); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    fi
    if [[ -f "${pid_file}" ]] && ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      echo "[ERROR] ${name} exited before becoming healthy" >&2
      return 1
    fi
    sleep 1
  done
  echo "[ERROR] ${name} was not healthy after ${READY_TIMEOUT}s" >&2
  return 1
}

[[ -f "${QWEN_MODEL}/config.json" ]] || { echo "Missing ${QWEN_MODEL}" >&2; exit 1; }
[[ -f "${GEMMA_MODEL}/config.json" ]] || { echo "Missing ${GEMMA_MODEL}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
trap stop_servers EXIT INT TERM

echo "Qwen3.8 + Gemma co-location test on GPU ${GPU}"
echo "Qwen: model=${QWEN_MODEL} util=${QWEN_GPU_MEM_UTIL} len=${QWEN_MAX_LEN} seqs=${QWEN_MAX_NUM_SEQS}"
echo "Gemma: model=${GEMMA_MODEL} util=${GEMMA_GPU_MEM_UTIL} len=${GEMMA_MAX_LEN} seqs=${GEMMA_MAX_NUM_SEQS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

"${VLLM_BIN}" serve "${QWEN_MODEL}" \
  --served-model-name qwen38-sampler \
  --dtype auto \
  --host 127.0.0.1 \
  --port "${QWEN_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${QWEN_MAX_LEN}" \
  --max-num-seqs "${QWEN_MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${QWEN_GPU_MEM_UTIL}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --trust-remote-code \
  >"${LOG_DIR}/qwen38.log" 2>&1 &
echo $! >"${QWEN_PID_FILE}"
wait_health qwen38 "${QWEN_PORT}" "${QWEN_PID_FILE}"

"${VLLM_BIN}" serve "${GEMMA_MODEL}" \
  --served-model-name gemma4-sampler \
  --dtype auto \
  --host 127.0.0.1 \
  --port "${GEMMA_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${GEMMA_MAX_LEN}" \
  --max-num-seqs "${GEMMA_MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GEMMA_GPU_MEM_UTIL}" \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --trust-remote-code \
  --enforce-eager \
  >"${LOG_DIR}/gemma4.log" 2>&1 &
echo $! >"${GEMMA_PID_FILE}"
wait_health gemma4 "${GEMMA_PORT}" "${GEMMA_PID_FILE}"

nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "Qwen endpoint: http://127.0.0.1:${QWEN_PORT}/v1"
echo "Gemma endpoint: http://127.0.0.1:${GEMMA_PORT}/v1"
wait
