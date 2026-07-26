#!/usr/bin/env bash
#SBATCH --job-name=qwen35_serve_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_serve_a100_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_serve_a100_%j.err

# Serve Qwen3.5-122B-A10B-FP8 on 2x A100_80GB (NVLink) instead of 4x L40S (PCIe).
# 122B FP8 weights ~116GB fit in 2x80GB=160GB (44GB left for KV/activations).
# NVLink + HBM should be ~3-5x faster per token than the L40S/PCIe build AND lets
# the COMPILED path (no --enforce-eager) work without the PCIe-only NCCL hang, for
# another ~3-5x. This is the throughput-preserving speedup for the THINKING sampler
# (thinking must stay on — the prompt asks for adversarial futures; see run_qwen35_common.sh).
#
# preempt: evictable, but the server is stateless — --requeue reloads weights in a
# few min, and decode clients resume via --skip-existing.
#
# SCALE-OUT UNIT: 2 free A100s are far easier to get than a whole node — submit
# this N times (each replica = one endpoint) instead of chasing an 8-GPU node.
# Each job self-registers under ENDPOINT_DIR; run_qwen35_40k_array.sbatch scans
# that dir and health-checks, so no hostname copy-pasting needed.

set -e

ENDPOINT_DIR="${ENDPOINT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/qwen35_endpoints}"
mkdir -p "${ENDPOINT_DIR}"

GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
VLLM="${GEMMA4_ENV}/bin/vllm"
MODEL="${MODEL:-/data/user_data/haolingp/models/Qwen3.5-122B-A10B-FP8}"
PORT="${PORT:-8300}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-16384}"
# Default COMPILED (eager OFF) — NVLink should not hit the PCIe NCCL hang. Set
# ENFORCE_EAGER=1 only if a launch stalls right after pynccl init.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export PATH="${GEMMA4_ENV}/bin:${PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# NCCL hang mitigations carried over from the L40S build (harmless on NVLink).
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"
DISABLE_CUSTOM_AR="${DISABLE_CUSTOM_AR:-0}"

echo "===== Qwen3.5-122B serve (A100 NVLink) ====="
echo "node=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi -L | wc -l)}"
echo "model=${MODEL} port=${PORT} tp=${TP} max_len=${MAX_LEN} enforce_eager=${ENFORCE_EAGER}"
echo "ENDPOINT for decode clients: http://$(hostname):${PORT}/v1"
echo "http://$(hostname):${PORT}/v1" > "${ENDPOINT_DIR}/${SLURM_JOB_ID:-manual}.txt"
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
