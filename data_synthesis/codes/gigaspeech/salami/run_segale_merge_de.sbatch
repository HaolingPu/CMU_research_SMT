#!/usr/bin/env bash
#SBATCH --job-name=salami_seg_merge_de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_merge_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_merge_%j.err

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de
SHARDS_ROOT=${BASE}/segale_pipeline/shards
OUTPUT=${BASE}/segale_pipeline/aligned_all.jsonl

echo "===== SALAMI de SEGALE merge START $(date) ====="
python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/merge_aligned_shards.py \
  --shards-root ${SHARDS_ROOT} \
  --output      ${OUTPUT} \
  --num-shards  24
echo "===== SALAMI de SEGALE merge DONE $(date) ====="
