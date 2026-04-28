#!/usr/bin/env bash
# Start two vLLM servers on GPU 1:
#   - google/gemma-4-E2B   on port 8101
#   - Qwen3-4B-Base        on port 8102
#
# Usage:
#   conda activate vllm
#   bash serve_gpu1.sh
#
# Stop all: bash serve_gpu1.sh stop
# Or:       kill $(cat /tmp/vllm_gemma4_e2b.pid) $(cat /tmp/vllm_qwen3_4b.pid)

set -e

export HF_HOME="/data/user_data/haolingp/hf_cache"
export CUDA_VISIBLE_DEVICES=1
VLLM_BIN="${VLLM_BIN:-vllm}"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-E2B}"
GEMMA_PORT="${GEMMA_PORT:-8101}"
GEMMA_PID_FILE="${GEMMA_PID_FILE:-/tmp/vllm_gemma4_e2b.pid}"
GEMMA_SERVED_NAME="${GEMMA_SERVED_NAME:-gemma4-e2b}"

QWEN_MODEL="${QWEN_MODEL:-/data/user_data/haolingp/models/Qwen3-4B-Base}"
QWEN_PORT="${QWEN_PORT:-8102}"
QWEN_PID_FILE="${QWEN_PID_FILE:-/tmp/vllm_qwen3_4b.pid}"

# ── stop mode ────────────────────────────────────────────────────────────────
if [[ "${1}" == "stop" ]]; then
  for pidfile in "${GEMMA_PID_FILE}" "${QWEN_PID_FILE}"; do
    if [[ -f "${pidfile}" ]]; then
      PID=$(cat "${pidfile}")
      if kill -0 "${PID}" 2>/dev/null; then
        echo "Killing $(basename ${pidfile} .pid) (pid ${PID})..."
        kill "${PID}" 2>/dev/null || true
        sleep 2
        kill -9 "${PID}" 2>/dev/null || true
      fi
      rm -f "${pidfile}"
    fi
  done
  exit 0
fi

# ── helper: kill stale process on a port ─────────────────────────────────────
kill_port() {
  local port=$1
  local pid_file=$2
  if [[ -f "${pid_file}" ]]; then
    OLD_PID=$(cat "${pid_file}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
      echo "Killing old server (pid ${OLD_PID})..."
      kill "${OLD_PID}" 2>/dev/null || true; sleep 3
      kill -9 "${OLD_PID}" 2>/dev/null || true; sleep 2
    fi
  fi
  PORT_PID=$(lsof -ti :"${port}" 2>/dev/null || true)
  if [[ -n "${PORT_PID}" ]]; then
    echo "Killing process on port ${port} (pid ${PORT_PID})..."
    kill ${PORT_PID} 2>/dev/null || true; sleep 2
  fi
}

kill_port "${GEMMA_PORT}" "${GEMMA_PID_FILE}"
kill_port "${QWEN_PORT}"  "${QWEN_PID_FILE}"

# ── launch gemma-4-E2B ────────────────────────────────────────────────────────
echo "Starting ${GEMMA_SERVED_NAME} on GPU 1, port ${GEMMA_PORT} ..."
"${VLLM_BIN}" serve "${GEMMA_MODEL}" \
  --served-model-name "${GEMMA_SERVED_NAME}" \
  --dtype auto \
  --port "${GEMMA_PORT}" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.45 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --enforce-eager &
echo $! > "${GEMMA_PID_FILE}"
echo "  PID: $(cat ${GEMMA_PID_FILE})"

# ── launch Qwen3-4B-Base ──────────────────────────────────────────────────────
echo "Starting Qwen3-4B-Base on GPU 1, port ${QWEN_PORT} ..."
"${VLLM_BIN}" serve "${QWEN_MODEL}" \
  --served-model-name qwen3-4b-base \
  --dtype auto \
  --port "${QWEN_PORT}" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.45 \
  --tensor-parallel-size 1 &
echo $! > "${QWEN_PID_FILE}"
echo "  PID: $(cat ${QWEN_PID_FILE})"

echo ""
echo "Waiting for both servers to be ready..."

# ── poll health endpoints ─────────────────────────────────────────────────────
gemma_ready=0
qwen_ready=0
for i in $(seq 1 300); do
  if [[ ${gemma_ready} -eq 0 ]] && curl -s http://localhost:${GEMMA_PORT}/health > /dev/null 2>&1; then
    echo "[${i}s] ${GEMMA_SERVED_NAME} is ready on port ${GEMMA_PORT}"
    gemma_ready=1
  fi
  if [[ ${qwen_ready} -eq 0 ]] && curl -s http://localhost:${QWEN_PORT}/health > /dev/null 2>&1; then
    echo "[${i}s] Qwen3-4B-Base is ready on port ${QWEN_PORT}"
    qwen_ready=1
  fi
  if [[ ${gemma_ready} -eq 1 && ${qwen_ready} -eq 1 ]]; then
    echo ""
    echo "Both servers are ready!"
    echo "  ${GEMMA_SERVED_NAME}   -> http://localhost:${GEMMA_PORT}"
    echo "  qwen3-4b-base          -> http://localhost:${QWEN_PORT}"
    wait
    exit 0
  fi
  sleep 1
done

echo "ERROR: one or both servers failed to start within 300s"
bash "$0" stop
exit 1
