#!/usr/bin/env bash
#SBATCH --job-name=giga_llm_train_xl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu


set -e
echo "===== START PIPELINE ====="

source ~/.bashrc
MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
CODE="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/llm_output_gigaspeech_trajectory.py"
OUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_raw"
NUM_TASKS="${NUM_TASKS:-8}"
TP="${TP:-1}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
mkdir -p "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs"
echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

conda activate vllm
python "${CODE}" \
  --input-tsv "${MANIFEST}" \
  --output-root "${OUT_ROOT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --tp "${TP}" \
  --batch-size "${BATCH_SIZE}"
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="