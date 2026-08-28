#!/usr/bin/env bash
# ============================================================
# Gemini Flash + local 30B future-distribution gate on GigaSpeech
# No simalign, direct Pro routing, final completion on Pro.
# 8 GPUs in parallel via SLURM array, total 100 outputs.
# ============================================================
#SBATCH --job-name=gem_fd30b100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=220G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7%8
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem_fd30b100_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem_fd30b100_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS=8
MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
PY_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/gemini/llm_future_sampling_thinking_policy_gemini_future_distribution.py"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini_future_distribution_local30b_100"
SLURM_LOG_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs"

if (( TASK_ID < 4 )); then
  MAX_ROWS=13
else
  MAX_ROWS=12
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}"
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: python script not found: ${PY_SCRIPT}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${SLURM_LOG_DIR}"

export HF_HOME="/data/user_data/haolingp/hf_cache"

echo "===== START TASK ${TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID:-N/A} node=$(hostname) time=$(date)"
echo "output_root=${OUTPUT_ROOT}"
echo "task_id=${TASK_ID} num_tasks=${NUM_TASKS} max_rows=${MAX_ROWS}"
echo "thinking_model=gemini-3-flash-preview fallback_model=gemini-3.1-pro-preview"
echo "gate_model=/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
echo "routing=local_30b_distribution_gate -> direct_pro | final_completion=pro | simalign=disabled"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

CUDA_VISIBLE_DEVICES=0 python "${PY_SCRIPT}"   --input-tsv "${MANIFEST}"   --output-root "${OUTPUT_ROOT}"   --task-id "${TASK_ID}"   --num-tasks "${NUM_TASKS}"   --max-rows "${MAX_ROWS}"   --base-model-path "/data/user_data/haolingp/models/Qwen3-4B-Base"   --gate-model-path "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"   --gate-model-gpu-memory-utilization 0.55   --thinking-model-name "gemini-3-flash-preview"   --fallback-model-name "gemini-3.1-pro-preview"   --final-completion-model-name "gemini-3.1-pro-preview"   --thinking-reasoning-effort "high"   --fallback-reasoning-effort "low"   --parallel-utterances 4   --future-sampling-batch-size 4   --future-sampling-batch-wait 0.05   --num-futures 10   --future-tokens 12   --sample-temperature 1.0   --thinking-temperature 0.1   --thinking-max-tokens 4096   --fallback-max-tokens 4096   --probe-samples-per-future 3   --probe-max-futures 2   --probe-temperature 0.7   --probe-max-tokens 16   --probe-rollout-max-chars 4   --probe-distribution-chars 2   --probe-avg-entropy-threshold 0.75   --probe-js-threshold 0.20   --probe-agreement-threshold 0.50   --gpu-memory-utilization 0.25   --overwrite

echo "===== DONE TASK ${TASK_ID} ====="
