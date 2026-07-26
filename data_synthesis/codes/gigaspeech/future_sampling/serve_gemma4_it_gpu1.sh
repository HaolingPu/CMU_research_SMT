#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for gemma-4-E2B-it (instruct).
# Used as the cross-family sampler in the targeted-instruct future-sampling
# pipeline, paired with Qwen3-30B-Instruct on serve_instruct_gpu0.sh.
#
# Usage (on a GPU node):
#   conda activate vllm
#   bash serve_gemma4_it_gpu1.sh           # start
#   bash serve_gemma4_it_gpu1.sh stop      # stop
#
# Defaults: GPU 1, port 8103, served-model-name gemma4-it.
# Override via env vars: GPU, PORT, MODEL, PID_FILE.

set -e

export HF_HOME="/data/user_data/haolingp/hf_cache"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
VLLM_BIN="${VLLM_BIN:-vllm}"

MODEL="${MODEL:-/data/user_data/haolingp/models/gemma-4-E2B-it}"
PORT="${PORT:-8103}"
PID_FILE="${PID_FILE:-/tmp/vllm_serve_gemma4_it.pid}"
SERVED_NAME="${SERVED_NAME:-gemma4-it}"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
      echo "Killing vllm serve (pid ${OLD_PID})..."
      kill "${OLD_PID}" 2>/dev/null || true
      sleep 3
      kill -9 "${OLD_PID}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  PORT_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
  if [[ -n "${PORT_PID}" ]]; then
    echo "Killing process on port ${PORT} (pid ${PORT_PID})..."
    kill ${PORT_PID} 2>/dev/null || true
    sleep 2
  fi
  exit 0
fi

# Kill any previous vllm serve on this port
if [[ -f "${PID_FILE}" ]]; then
  OLD_PID=$(cat "${PID_FILE}")
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Killing old vllm serve (pid ${OLD_PID})..."
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 3
    kill -9 "${OLD_PID}" 2>/dev/null || true
    sleep 2
  fi
fi
PORT_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
if [[ -n "${PORT_PID}" ]]; then
  echo "Killing process on port ${PORT} (pid ${PORT_PID})..."
  kill ${PORT_PID} 2>/dev/null
  sleep 2
fi

echo "Starting vLLM server on GPU ${CUDA_VISIBLE_DEVICES}, port ${PORT} ..."
echo "Model: ${MODEL}"
echo "Served name: ${SERVED_NAME}"
echo "To stop: bash $0 stop"

"${VLLM_BIN}" serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --dtype auto \
  --port "${PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-logprobs 100 &

echo $! > "${PID_FILE}"
echo "PID: $(cat "${PID_FILE}")"
echo "Waiting for server to be ready..."

READY_TIMEOUT="${READY_TIMEOUT:-900}"
for i in $(seq 1 "${READY_TIMEOUT}"); do
  if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "Server is ready! (took ~${i}s)"
    echo "Test with: curl http://localhost:${PORT}/v1/models"
    wait
    exit 0
  fi
  sleep 1
done

echo "ERROR: Server failed to start within ${READY_TIMEOUT}s"
kill "$(cat "${PID_FILE}")" 2>/dev/null
exit 1
