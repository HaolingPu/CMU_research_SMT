#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for Qwen3-30B-A3B-Instruct.
#
# Usage (on a GPU node with 2 GPUs):
#   conda activate vllm
#   bash test_instruct_serve.sh
#
# This uses GPU 1 (so GPU 0 can run the base model simultaneously).
# Server listens on port 8100. Kill with Ctrl+C or `kill $(cat /tmp/vllm_serve.pid)`.

set -e

export HF_HOME="/data/user_data/haolingp/hf_cache"
export CUDA_VISIBLE_DEVICES=1

MODEL="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
PORT=8100

# Kill any previous vllm serve on this port
if [[ -f /tmp/vllm_serve.pid ]]; then
  OLD_PID=$(cat /tmp/vllm_serve.pid)
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Killing old vllm serve (pid ${OLD_PID})..."
    kill "${OLD_PID}" 2>/dev/null
    sleep 3
    kill -9 "${OLD_PID}" 2>/dev/null
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

echo "Starting vLLM server on GPU 1, port ${PORT} ..."
echo "Model: ${MODEL}"
echo "To stop: kill \$(cat /tmp/vllm_serve.pid)"

vllm serve "${MODEL}" \
  --served-model-name qwen3-instruct \
  --dtype auto \
  --port "${PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 &

echo $! > /tmp/vllm_serve.pid
echo "PID: $(cat /tmp/vllm_serve.pid)"
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
kill "$(cat /tmp/vllm_serve.pid)" 2>/dev/null
exit 1
