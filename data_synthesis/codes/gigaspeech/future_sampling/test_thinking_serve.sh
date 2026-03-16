#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for Qwen3-30B-A3B-Thinking.
#
# Usage (on a GPU node with 2 GPUs):
#   conda activate vllm
#   bash test_thinking_serve.sh
#
# This uses GPU 1 (so GPU 0 can run the base model simultaneously).
# Server listens on port 8001. Kill with Ctrl+C or `kill $(cat /tmp/vllm_thinking_serve.pid)`.
#
# Notes for vLLM 0.11.x:
# - Do not add a CLI flag like `--enable-thinking`; that is not a supported
#   `vllm serve` argument in this environment.
# - Keep reasoning enabled on the server with the Qwen3 parser.
# - If a specific request needs plain answer-only output, do it at request
#   time with `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`.

set -e

export HF_HOME="/data/user_data/haolingp/hf_cache"
export CUDA_VISIBLE_DEVICES=1

MODEL="/data/user_data/haolingp/models/Qwen3-30B-A3B-Thinking-2507-FP8"
SERVED_MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
PORT="${PORT:-8001}"
PID_FILE="/tmp/vllm_thinking_serve.pid"

# Kill any previous vllm thinking serve on this port
if [[ -f "${PID_FILE}" ]]; then
  OLD_PID=$(cat "${PID_FILE}")
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Killing old thinking vllm serve (pid ${OLD_PID})..."
    kill "${OLD_PID}" 2>/dev/null
    sleep 3
    kill -9 "${OLD_PID}" 2>/dev/null || true
    sleep 2
  else
    rm -f "${PID_FILE}"
  fi
fi

# Also kill anything else listening on the port
PORT_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
if [[ -n "${PORT_PID}" ]]; then
  echo "Killing process on port ${PORT} (pid ${PORT_PID})..."
  kill ${PORT_PID} 2>/dev/null || true
  sleep 2
fi

# Refuse to start if the port is still occupied after cleanup.
if timeout 1 bash -lc "cat < /dev/null > /dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "ERROR: port ${PORT} is still in use after cleanup; refusing to start a new thinking server"
  exit 1
fi

echo "Starting thinking vLLM server on GPU 1, port ${PORT} ..."
echo "Model: ${MODEL}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Reasoning parser: qwen3"
echo "Thinking mode: server default (per-request override supported via chat_template_kwargs)"
echo "To stop: kill \$(cat ${PID_FILE})"

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --reasoning-parser qwen3 \
  --dtype auto \
  --port "${PORT}" \
  --max-model-len 102928 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 &

# model-card examples may show different flags/parser combos on other versions.
# In this local vLLM 0.11.x environment we keep the supported qwen3 parser here
# and use request-level `enable_thinking=False` when we want answer-only output.

NEW_PID=$!
echo "${NEW_PID}" > "${PID_FILE}"
echo "PID: $(cat "${PID_FILE}")"
echo "Waiting for server to be ready..."

for i in $(seq 1 300); do
  if ! kill -0 "${NEW_PID}" 2>/dev/null; then
    echo "ERROR: thinking server process ${NEW_PID} exited before becoming ready"
    rm -f "${PID_FILE}"
    exit 1
  fi
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "Thinking server is ready! (took ~${i}s)"
    echo "API base: http://localhost:${PORT}/v1"
    wait
    exit 0
  fi
  sleep 1
done

echo "ERROR: Thinking server failed to start within 300s"
kill "$(cat "${PID_FILE}")" 2>/dev/null || true
rm -f "${PID_FILE}"
exit 1
