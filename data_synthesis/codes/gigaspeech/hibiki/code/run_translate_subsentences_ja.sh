#!/usr/bin/env bash
#SBATCH --job-name=trans_subsent_ja
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=1-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/hibiki_logs/trans_subsent_ja_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/hibiki_logs/trans_subsent_ja_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START JA SHARD ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"

source ~/.bashrc
conda activate vllm

export HF_HOME="/data/user_data/haolingp/hf_cache"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/subsentence_ref_shards_ja"
MODEL_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
CODE="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/hibiki/code/translate_subsentences.py"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/hibiki_logs
mkdir -p "${OUTPUT_ROOT}"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

python "${CODE}" \
  --input-tsv "${INPUT_TSV}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --target-lang ja \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks 8 \
  --tp 1 \
  --gpu-memory-utilization 0.90 \
  --batch-size 512 \
  --max-model-len 4096

echo "===== DONE JA SHARD ${SLURM_ARRAY_TASK_ID} ====="
echo "After all array jobs finish, merge with:"
echo "python ${CODE} --input-tsv ${INPUT_TSV} --output-root ${OUTPUT_ROOT} --target-lang ja --merge"
