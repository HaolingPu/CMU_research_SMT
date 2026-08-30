#!/usr/bin/env bash
#SBATCH --job-name=q38_gemma_fit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-o9-24,babel-q9-32,babel-t5-28
#SBATCH --output=/home/haolingp/slurm_runs/qwen38_gemma_fit_%j.out
#SBATCH --error=/home/haolingp/slurm_runs/qwen38_gemma_fit_%j.err

set -euo pipefail

SERVER="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/qwen38/serve_qwen38_gemma_colocated.sh"
RUN_DIR="/home/haolingp/slurm_runs/qwen38_gemma_fit_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

export GPU=0
export LOG_DIR="${RUN_DIR}"
export QWEN_PID_FILE="/tmp/qwen38_${SLURM_JOB_ID}.pid"
export GEMMA_PID_FILE="/tmp/gemma4_${SLURM_JOB_ID}.pid"

cleanup() {
  "${SERVER}" stop || true
}
trap cleanup EXIT

"${SERVER}" >"${RUN_DIR}/server.out" 2>"${RUN_DIR}/server.err" &
server_pid=$!

wait_endpoint() {
  local name=$1
  local port=$2
  for i in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[ERROR] server launcher exited" >&2
      tail -n 100 "${RUN_DIR}/server.err" >&2 || true
      tail -n 100 "${RUN_DIR}/qwen38.log" >&2 || true
      tail -n 100 "${RUN_DIR}/gemma4.log" >&2 || true
      return 1
    fi
    sleep 1
  done
  return 1
}

wait_endpoint qwen38 8310
wait_endpoint gemma4 8311

for spec in "qwen38-sampler:8310" "gemma4-sampler:8311"; do
  model=${spec%%:*}
  port=${spec##*:}
  echo "=== ${model} models ==="
  curl -fsS "http://127.0.0.1:${port}/v1/models"
  echo
  echo "=== ${model} completion ==="
  curl -fsS "http://127.0.0.1:${port}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model}\",\"prompt\":\"The bank\",\"max_tokens\":16,\"temperature\":1.0,\"top_p\":0.95,\"n\":2}"
  echo
done

echo "=== GPU processes after requests ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "COLOCATION_SMOKE_OK"

