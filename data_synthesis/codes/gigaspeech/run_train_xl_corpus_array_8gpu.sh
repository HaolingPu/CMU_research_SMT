#!/usr/bin/env bash
#SBATCH --job-name=giga_corpus_train_xl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=1-12:00:00
#SBATCH --array=0-7
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/corpus_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/corpus_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START PIPELINE ====="

source ~/.bashrc
MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
CODE="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/export_mfa_corpus_gigaspeech.py"
OUT_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus"
NUM_TASKS="${NUM_TASKS:-8}"
mkdir -p "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs"

conda activate SMT
python "${CODE}" \
  --input-tsv "${MANIFEST}" \
  --output-dir "${OUT_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --overwrite