#!/usr/bin/env bash
#SBATCH --job-name=giga_llm_cache
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/llm_full_translation_cache/slurm_logs/llm_cache_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/llm_full_translation_cache/slurm_logs/llm_cache_%A_%a.err

set -e
echo "===== START PIPELINE ====="

source ~/.bashrc
conda activate vllm

MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
CODE="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/build_llm_full_translation_cache.py"
OUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/llm_full_translation_cache/train_xl_case_robust_asr_filtered"

MODEL_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
NUM_TASKS="${NUM_TASKS:-8}"
TP="${TP:-1}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_ROWS="${MAX_ROWS:-}"

mkdir -p "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/llm_full_translation_cache/slurm_logs"
mkdir -p "${OUT_ROOT}"

echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

export HF_HOME="/data/user_data/haolingp/hf_cache"

if [[ -n "${MAX_ROWS}" ]]; then
  python "${CODE}" \
    --input-tsv "${MANIFEST}" \
    --output-root "${OUT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${NUM_TASKS}" \
    --tp "${TP}" \
    --batch-size "${BATCH_SIZE}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-rows "${MAX_ROWS}"
else
  python "${CODE}" \
    --input-tsv "${MANIFEST}" \
    --output-root "${OUT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${NUM_TASKS}" \
    --tp "${TP}" \
    --batch-size "${BATCH_SIZE}" \
    --max-tokens "${MAX_TOKENS}"
fi

echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
