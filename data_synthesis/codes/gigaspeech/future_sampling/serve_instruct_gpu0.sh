#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for Qwen3-30B-A3B-Instruct.
#
# Usage (on a GPU node with 2 GPUs):
#   conda activate vllm
#   bash test_instruct_serve.sh
#
# This uses GPU 0.
# Server listens on port 8100. Kill with Ctrl+C or `kill $(cat /tmp/vllm_serve.pid)`.

set -e

export HF_HOME="/data/user_data/haolingp/hf_cache"
export CUDA_VISIBLE_DEVICES=0
VLLM_BIN="${VLLM_BIN:-vllm}"

MODEL="${MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
PORT="${PORT:-8100}"
PID_FILE="${PID_FILE:-/tmp/vllm_serve.pid}"

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
# Also kill anything else listening on the port
PORT_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
if [[ -n "${PORT_PID}" ]]; then
  echo "Killing process on port ${PORT} (pid ${PORT_PID})..."
  kill ${PORT_PID} 2>/dev/null
  sleep 2
fi

echo "Starting vLLM server on GPU 0, port ${PORT} ..."
echo "Model: ${MODEL}"
echo "To stop: bash $0 stop"

"${VLLM_BIN}" serve "${MODEL}" \
  --served-model-name qwen3-instruct \
  --dtype auto \
  --port "${PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-logprobs 100 &

echo $! > "${PID_FILE}"
echo "PID: $(cat "${PID_FILE}")"
echo "Waiting for server to be ready..."

# Poll until the health endpoint responds
for i in $(seq 1 300); do
  if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "Server is ready! (took ~${i}s)"
    echo "Run: python test_instruct_client.py"
    wait
    exit 0
  fi
  sleep 1
done

echo "ERROR: Server failed to start within 300s"
kill "$(cat "${PID_FILE}")" 2>/dev/null
exit 1
