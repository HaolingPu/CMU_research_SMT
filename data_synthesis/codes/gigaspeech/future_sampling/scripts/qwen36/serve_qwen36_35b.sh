#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B-FP8 as the translation/probe model.
#
# Single L40S (FP8 weights 37.5GB): no TP, so none of the 122B NCCL hang
# mitigations are needed. Same qwen3_5_moe arch family as Qwen3.5-122B, so the
# gemma4 vLLM env serves it unchanged.
#
# Usage (debug node, GPU 0):
#   bash serve_qwen36_35b.sh          # foreground
#   bash serve_qwen36_35b.sh stop
set -e

GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
VLLM="${VLLM_BIN:-${GEMMA4_ENV}/bin/vllm}"
MODEL="${MODEL:-/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-8300}"
GPU="${GPU:-0}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-translator}"
# Gated-DeltaNet recurrent state is allocated PER SEQUENCE SLOT (~1.03GiB/layer
# at the vLLM default max_num_seqs=1024 -> ~32GiB for 30 linear-attn layers,
# which OOMs a 46GB L40S regardless of --gpu-memory-utilization). The decoder
# runs <=8 concurrent cases, so 64 slots is generous and costs ~2GiB.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PID_FILE="${PID_FILE:-/tmp/vllm_qwen36_sampler.pid}"
# Thinking lands in message.reasoning_content, message.content stays the clean
# numbered list the decoder parses (same contract as the Qwen3.5-122B sampler).
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# ENFORCE_EAGER=1 skips torch.compile (fallback if compile crashes; slower/token).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export PATH="${GEMMA4_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 3
    kill -9 "${OLD_PID}" 2>/dev/null || true
    rm -f "${PID_FILE}"
  fi
  PORT_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
  if [[ -n "${PORT_PID}" ]]; then
    kill ${PORT_PID} 2>/dev/null || true
    sleep 2
  fi
  exit 0
fi

echo "===== Qwen3.6-35B-A3B-FP8 translator/probe serve ====="
echo "ENDPOINT for decode clients: http://$(hostname):${PORT}/v1"
echo "model=${MODEL} port=${PORT} gpu=${GPU} max_len=${MAX_LEN}"

CMD=(
  "${VLLM}" serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype auto
  --port "${PORT}"
  --host 0.0.0.0
  --tensor-parallel-size 1
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --enable-prefix-caching
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --reasoning-parser "${REASONING_PARSER}"
  --trust-remote-code
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  CMD+=(--enforce-eager)
fi
"${CMD[@]}" &
echo $! > "${PID_FILE}"
echo "PID: $(cat "${PID_FILE}")"
wait
