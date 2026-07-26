#!/usr/bin/env bash
#SBATCH --job-name=qwen35_serve
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_serve_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_serve_%j.err
# normal QoS caps at 8 GPUs -> max 2 of these on general. For more replicas:
#   sbatch --partition=preempt --qos=preempt_qos --requeue serve_qwen35_122b.sbatch
# (preempt_qos allows 24 GPUs = 6 replicas; server is stateless, requeue is safe)

# Serve Qwen3.5-122B-A10B-FP8 (MoE, hybrid Gated-DeltaNet attention) as an
# OpenAI-compatible vLLM server, to be used as the local future sampler in
# consensus decoding (replaces the paid DeepSeek/GPT API calls).
#
# 4x L40S, tensor-parallel-size 4 (num_kv_heads=2, num_heads=32, experts=256
# all divide by 4; TP=5 would NOT work). FP8 weights ~116GB -> ~29GB/GPU.
#
# SCALE-OUT UNIT (L40S fallback): L40S is by far the most available GPU on babel
# (50 nodes) — if A100 pairs won't schedule, submit this N times instead; each
# job is one sampler replica/endpoint. Self-registers under ENDPOINT_DIR so
# run_qwen35_40k_array.sbatch auto-discovers healthy endpoints.

set -e

ENDPOINT_DIR="${ENDPOINT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/qwen35_endpoints}"
mkdir -p "${ENDPOINT_DIR}"

GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
VLLM="${GEMMA4_ENV}/bin/vllm"
MODEL="${MODEL:-/data/user_data/haolingp/models/Qwen3.5-122B-A10B-FP8}"
PORT="${PORT:-8300}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-16384}"
# ENFORCE_EAGER=1 skips torch.compile + CUDA-graph capture. Eager loads reliably
# but is ~3-5x SLOWER per token (~13 tok/s for this MoE on 4x L40S over PCIe).
# Default OFF (compiled) for usable throughput. The only compile hang seen was on
# babel-p5-20 and was node-specific (NCCL init, not compile) -> if a launch stalls,
# exclude that node and retry. Set ENFORCE_EAGER=1 to fall back to eager.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Reasoning parser: with enable_thinking=true, Qwen3.5 emits <reasoning>...</think>
# then the answer. --reasoning-parser qwen3 routes the chain-of-thought to
# message.reasoning_content and leaves message.content as the clean (JSON) answer,
# which the decoder parses. Empty -> no parser (use only with thinking disabled).
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export PATH="${GEMMA4_ENV}/bin:${PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- TP=4 NCCL hang mitigations ---
# Both the full-compile and eager launches hung in a userspace busy-spin right
# after pynccl init, before weight loading -> NCCL collective deadlock during
# the 4-way tensor-parallel setup. Disable P2P/NVLink shortcuts + custom
# all-reduce, which is the usual fix on PCIe-only / mixed-topology nodes.
# NCCL_DEBUG=INFO surfaces the exact channel/ring stall point in the log.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
DISABLE_CUSTOM_AR="${DISABLE_CUSTOM_AR:-1}"

echo "===== Qwen3.5-122B serve ====="
echo "ENDPOINT for decode clients: http://$(hostname):${PORT}/v1"
echo "http://$(hostname):${PORT}/v1" > "${ENDPOINT_DIR}/${SLURM_JOB_ID:-manual}.txt"
echo "node=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi -L | wc -l)}"
echo "model=${MODEL} port=${PORT} tp=${TP} max_len=${MAX_LEN} enforce_eager=${ENFORCE_EAGER}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CMD=(
  "${VLLM}" serve "${MODEL}"
  --served-model-name qwen35-sampler
  --dtype auto
  --port "${PORT}"
  --host 0.0.0.0
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_LEN}"
  --gpu-memory-utilization 0.90
  --max-logprobs 100
  --trust-remote-code
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  CMD+=(--enforce-eager)
fi
if [[ -n "${REASONING_PARSER}" ]]; then
  CMD+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ "${DISABLE_CUSTOM_AR}" == "1" ]]; then
  CMD+=(--disable-custom-all-reduce)
fi

"${CMD[@]}"
