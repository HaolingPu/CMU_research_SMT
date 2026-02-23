#!/usr/bin/env bash
# ============================================================
# EAST  Stage 3 — MetricX prediction (8-GPU array job)
#
# Input  : ${BASE}/metricx_shards/input_00 … input_07
# Output : ${BASE}/metricx_shards/output_00.jsonl … output_07.jsonl
#
# Submit : sbatch --array=0-7 stage3_metricx.sh
#            (or let the scheduler expand the array automatically)
# Then   : sbatch stage4_final.sh   (after ALL 8 array tasks finish)
# ============================================================
#SBATCH --job-name=east_s3_metricx
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/stage3_metricx_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/stage3_metricx_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== EAST Stage 3 shard ${SLURM_ARRAY_TASK_ID} — START $(date) ====="

source ~/.bashrc
conda activate metricx

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

BASE=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east

SHARD_ID=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
INPUT=${BASE}/metricx_shards/input_${SHARD_ID}
OUTPUT=${BASE}/metricx_shards/output_${SHARD_ID}.jsonl

echo "  Input : ${INPUT}"
echo "  Output: ${OUTPUT}"

cd /data/user_data/haolingp/data_synthesis/codes/metricx

PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer            /data/user_data/haolingp/models/mt5-xl \
  --model_name_or_path   /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
  --max_input_length     1536 \
  --batch_size           1 \
  --input_file           ${INPUT} \
  --output_file          ${OUTPUT} \
  --qe

echo "===== EAST Stage 3 shard ${SLURM_ARRAY_TASK_ID} — DONE $(date) ====="
