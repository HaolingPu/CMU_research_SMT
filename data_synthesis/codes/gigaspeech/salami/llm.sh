#!/usr/bin/env bash
#SBATCH --job-name=giga_llm_salami
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu


set -e
echo "===== START PIPELINE ====="

source ~/.bashrc

echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

conda activate vllm

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=2


python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/salami/llm_output_salami.py \
  --input-tsv /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
  --output-root /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_raw \
  --task-id ${SLURM_ARRAY_TASK_ID}  \
  --num-tasks 8 \
  --tp 1 \
  --batch-size 2048

echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="


# testing
# python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/salami/llm_output_salami.py \
#   --input-tsv /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
#   --output-root /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_raw \
#   --task-id 0  \
#   --num-tasks 1 \
#   --tp 1 \
#   --batch-size 8 \
#   --max-rows 20 \
#   --overwrite