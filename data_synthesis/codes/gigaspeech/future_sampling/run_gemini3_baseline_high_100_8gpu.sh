#!/usr/bin/env bash
# ============================================================
# Gemini-3 thinking-policy on GigaSpeech
# 8 GPUs in parallel via SLURM array, total 100 outputs.
#
# Usage:
#   sbatch /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_gemini3_baseline_high_100_8gpu.sh
# ============================================================
#SBATCH --job-name=gem3_base100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=220G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7%8
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem3_base100_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem3_base100_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS=8

# Exact 100 outputs under modulo sharding:
# tasks 0-3 => 13 rows each, tasks 4-7 => 12 rows each.
if (( TASK_ID < 4 )); then
  MAX_ROWS=13
else
  MAX_ROWS=12
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  exit 1
fi
if [[ ! -f "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv" ]]; then
  echo "ERROR: manifest not found: /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
  exit 1
fi
if [[ ! -f "/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy_gemini.py" ]]; then
  echo "ERROR: python script not found: /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy_gemini.py"
  exit 1
fi

mkdir -p "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini3_baseline_high_100_8gpu"
mkdir -p "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export SIMALIGN_MODEL="/data/user_data/haolingp/models/LaBSE"

echo "===== START TASK ${TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID:-N/A} node=$(hostname) time=$(date)"
echo "output_root=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini3_baseline_high_100_8gpu"
echo "task_id=${TASK_ID} num_tasks=8 max_rows=${MAX_ROWS}"
echo "thinking_model=gemini-3-flash-preview reasoning=high prompt=base"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

CUDA_VISIBLE_DEVICES=0 python "/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy_gemini.py" \
  --input-tsv "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv" \
  --output-root "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini3_advanced_high_100" \
  --task-id "${TASK_ID}" \
  --num-tasks 8 \
  --max-rows "${MAX_ROWS}" \
  --base-model-path "/data/user_data/haolingp/models/Qwen3-4B-Base" \
  --thinking-model-name "gemini-3-flash-preview" \
  --thinking-prompt-version "advanced" \
  --gemini-include-thoughts \
  --thinking-reasoning-effort "high" \
  --parallel-utterances 8 \
  --future-sampling-batch-size 4 \
  --future-sampling-batch-wait 0.05 \
  --num-futures 10 \
  --future-tokens 12 \
  --sample-temperature 1.0 \
  --thinking-temperature 0.1 \
  --thinking-max-tokens 4096 \
  --align-device "cuda:0" \
  --gpu-memory-utilization 0.95 \
  --overwrite

echo "===== DONE TASK ${TASK_ID} ====="
