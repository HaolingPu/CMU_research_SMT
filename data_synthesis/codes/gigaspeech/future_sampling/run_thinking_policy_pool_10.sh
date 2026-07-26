#!/usr/bin/env bash
# Throughput trial for thinking_policy:
#   - GPU 0: shared base model + simalign + controller process
#   - GPU 1-7: seven thinking-model vLLM servers
#
# Trial run (10 utterances on one node, 8 GPUs):
#   sbatch run_thinking_policy_pool_10.sh
#
# Common overrides:
#   MAX_ROWS=10 PARALLEL_UTTERANCES=10 sbatch run_thinking_policy_pool_10.sh
#   BASE_PORT=8101 PARALLEL_UTTERANCES=20 sbatch run_thinking_policy_pool_10.sh

#SBATCH --job-name=think_pool_10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/think_pool_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/think_pool_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_pool_7srv_10utt/slurm_logs
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_pool_7srv_10utt

if [[ ! -f /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv ]]; then
  echo "ERROR: MANIFEST not found: /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
  exit 1
fi
if [[ ! -f /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy.py ]]; then
  echo "ERROR: Script not found: /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy.py"
  exit 1
fi

export HF_HOME="/data/user_data/haolingp/hf_cache"

echo "===== START THINKING-POOL TRIAL ====="
echo "job_id=${SLURM_JOB_ID:-N/A} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

declare -a SERVE_PIDS=()
declare -a THINKING_BASES=()

cleanup() {
  echo "[Cleanup] stopping thinking servers ..."
  for pid in "${SERVE_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SERVE_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "[Step 1] Starting 7 thinking servers on GPU 1-7 ..."
for gpu in 1 2 3 4 5 6 7; do
  port=$((8001 + gpu - 1))
  THINKING_BASES+=("http://localhost:${port}/v1")
  log_file="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_pool_7srv_10utt.thinking_gpu${gpu}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "/data/user_data/haolingp/models/Qwen3-30B-A3B-Thinking-2507-FP8" \
    --served-model-name "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8" \
    --reasoning-parser qwen3 \
    --dtype auto \
    --port "${port}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    > "${log_file}" 2>&1 &

  pid=$!
  SERVE_PIDS+=("${pid}")
  echo "  GPU ${gpu} -> port ${port} pid=${pid} log=${log_file}"
done

echo "[Step 1] Waiting for all thinking servers to become ready ..."
for idx in "${!SERVE_PIDS[@]}"; do
  gpu=$((idx + 1))
  port=$((8001 + idx))
  pid="${SERVE_PIDS[$idx]}"
  ready=0
  for i in $(seq 1 300); do
    if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
      echo "  thinking server on GPU ${gpu} ready at port ${port} (took ~${i}s)"
      ready=1
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: thinking server on GPU ${gpu} died before readiness. Check /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_pool_7srv_10utt.thinking_gpu${gpu}.log"
      exit 1
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "ERROR: thinking server on GPU ${gpu} not ready after timeout"
    exit 1
  fi
done

THINKING_API_BASES_CSV="$(IFS=,; echo "${THINKING_BASES[*]}")"
echo "[Step 1] thinking_api_bases=${THINKING_API_BASES_CSV}"

echo "[Step 2] Running thinking_policy on GPU 0 ..."
SIMALIGN_MODEL="/data/user_data/haolingp/models/LaBSE" CUDA_VISIBLE_DEVICES=0 python "/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy.py" \
  --input-tsv "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv" \
  --output-root "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_pool_7srv_10utt" \
  --task-id 0 \
  --num-tasks 1 \
  --max-rows 10 \
  --base-model-path "/data/user_data/haolingp/models/Qwen3-4B-Base" \
  --thinking-api-bases "${THINKING_API_BASES_CSV}" \
  --thinking-model-name "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8" \
  --parallel-utterances 10 \
  --future-sampling-batch-size 4 \
  --future-sampling-batch-wait 0.05 \
  --num-futures 5 \
  --future-tokens 10 \
  --sample-temperature 1.0 \
  --thinking-temperature 0.1 \
  --thinking-max-tokens 16384 \
  --overwrite

echo "===== DONE THINKING-POOL TRIAL ====="
