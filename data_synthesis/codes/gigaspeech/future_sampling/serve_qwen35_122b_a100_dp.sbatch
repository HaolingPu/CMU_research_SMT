#!/usr/bin/env bash
#SBATCH --job-name=qwen35_serve_dp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100_80GB:8
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_serve_dp_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_serve_dp_%j.err

# Full-node Qwen3.5-122B-A10B-FP8 sampler farm: 8x A100_80GB (NVLink) hosting
# FOUR TP=2 replicas. The 2xA100 TP=2 compiled path is validated (job 8812769
# booted CUDA graphs fine on babel-v5-16; it only died to preemption).
#
# Two modes:
#   SERVE_MODE=dp    (default) one endpoint, vLLM --data-parallel-size 4 does
#                    internal load balancing across the 4 replicas -> clients
#                    need a single SAMPLER_API_BASE per node.
#   SERVE_MODE=multi fallback if DP misbehaves: 4 independent vllm serve
#                    processes on ports PORT..PORT+3, each pinned to a GPU pair
#                    via CUDA_VISIBLE_DEVICES -> give clients all 4 endpoints.
#
# preempt-safe: server is stateless; --requeue reloads weights (~5 min) and
# decode clients resume via --skip-existing.
#
# NOTE: a whole 8-GPU A100 node is hard to get. The easier-to-schedule path is
# N independent 2-GPU jobs: serve_qwen35_122b_a100.sbatch (submit it several
# times). Both register endpoints under ENDPOINT_DIR so clients auto-discover.

set -e

# Self-registration: clients (run_qwen35_40k_array.sbatch) scan this dir and
# health-check each endpoint, so no hostname copy-pasting and stale files from
# dead jobs are ignored.
ENDPOINT_DIR="${ENDPOINT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/qwen35_endpoints}"
mkdir -p "${ENDPOINT_DIR}"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID:-manual}.txt"

GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
VLLM="${GEMMA4_ENV}/bin/vllm"
MODEL="${MODEL:-/data/user_data/haolingp/models/Qwen3.5-122B-A10B-FP8}"
PORT="${PORT:-8300}"
TP="${TP:-2}"
DP="${DP:-4}"
MAX_LEN="${MAX_LEN:-16384}"
SERVE_MODE="${SERVE_MODE:-dp}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export PATH="${GEMMA4_ENV}/bin:${PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# NVLink node: keep P2P/custom all-reduce ON (the PCIe hang mitigations are
# only needed on L40S nodes).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "===== Qwen3.5-122B sampler farm (8x A100, ${SERVE_MODE}) ====="
echo "node=$(hostname) job=${SLURM_JOB_ID}"
echo "model=${MODEL} port=${PORT} tp=${TP} dp=${DP} max_len=${MAX_LEN}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

COMMON_ARGS=(
  --served-model-name qwen35-sampler
  --dtype auto
  --host 0.0.0.0
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_LEN}"
  --gpu-memory-utilization 0.90
  --trust-remote-code
)
if [[ -n "${REASONING_PARSER}" ]]; then
  COMMON_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [[ "${SERVE_MODE}" == "multi" ]]; then
  : > "${ENDPOINT_FILE}"
  for i in $(seq 0 $(( DP - 1 ))); do
    echo "http://$(hostname):$(( PORT + i ))/v1" >> "${ENDPOINT_FILE}"
  done
  PIDS=()
  for i in $(seq 0 $(( DP - 1 ))); do
    GPUS="$(( i * TP ))"
    for j in $(seq 1 $(( TP - 1 ))); do
      GPUS="${GPUS},$(( i * TP + j ))"
    done
    REPLICA_PORT=$(( PORT + i ))
    echo "[replica ${i}] GPUs=${GPUS} port=${REPLICA_PORT}"
    echo "ENDPOINT for decode clients: http://$(hostname):${REPLICA_PORT}/v1"
    CUDA_VISIBLE_DEVICES="${GPUS}" \
      "${VLLM}" serve "${MODEL}" --port "${REPLICA_PORT}" "${COMMON_ARGS[@]}" \
      > "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_serve_dp_${SLURM_JOB_ID}_replica${i}.log" 2>&1 &
    PIDS+=($!)
  done
  # If any replica dies, kill the rest so SLURM requeues the whole node job.
  trap 'kill "${PIDS[@]}" 2>/dev/null' EXIT
  wait -n
  echo "[ERROR] a replica exited; failing the job so --requeue restarts the node"
  exit 1
else
  echo "http://$(hostname):${PORT}/v1" > "${ENDPOINT_FILE}"
  echo "ENDPOINT for decode clients: http://$(hostname):${PORT}/v1"
  "${VLLM}" serve "${MODEL}" --port "${PORT}" --data-parallel-size "${DP}" "${COMMON_ARGS[@]}"
fi
