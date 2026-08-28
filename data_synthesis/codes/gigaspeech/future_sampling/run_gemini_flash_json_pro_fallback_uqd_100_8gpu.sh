#!/usr/bin/env bash
# ============================================================
# Gemini Flash JSON + direct Pro fallback on GigaSpeech
# No simalign, no second Flash, BLEU/LAAL written into output JSON.
# 8 GPUs in parallel via SLURM array, total 100 outputs.
#
# Usage:
#   sbatch /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_gemini_flash_json_pro_fallback_uqd_100_8gpu.sh
# ============================================================
#SBATCH --job-name=gem_flash_nosim100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=220G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7%8
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem_flash_nosim100_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem_flash_nosim100_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS=8
MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
PY_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/gemini/llm_future_sampling_thinking_policy_gemini_json_flash_pro_fallback.py"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini_flash_json_pro_nosimalign_nosecond_100"
SLURM_LOG_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs"

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
echo "thinking_model=gemini-3-flash-preview fallback_model=gemini-pro-latest reasoning=high"
echo "routing=flash_then_direct_pro | simalign=disabled | second_flash=disabled | metrics=bleu+laal"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

CUDA_VISIBLE_DEVICES=0 python "${PY_SCRIPT}"   --input-tsv "${MANIFEST}"   --output-root "${OUTPUT_ROOT}"   --task-id "${TASK_ID}"   --num-tasks "${NUM_TASKS}"   --max-rows "${MAX_ROWS}"   --base-model-path "/data/user_data/haolingp/models/Qwen3-4B-Base"   --thinking-model-name "gemini-3-flash-preview"   --fallback-model-name "gemini-pro-latest"   --thinking-reasoning-effort "high"   --fallback-reasoning-effort "low"   --parallel-utterances 8   --future-sampling-batch-size 4   --future-sampling-batch-wait 0.05   --num-futures 10   --future-tokens 12   --sample-temperature 1.0   --thinking-temperature 0.1   --thinking-max-tokens 4096   --fallback-max-tokens 4096   --gpu-memory-utilization 0.90   --overwrite

echo "===== DONE TASK ${TASK_ID} ====="
