#!/usr/bin/env bash
# ============================================================
# SALAMI ja — Stage 2: convert streaming dataset to MetricX QE input
#                       and split into 8 shards.
# ============================================================
#SBATCH --job-name=salami_ja_s2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/stage2_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/stage2_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== SALAMI ja Stage 2 START $(date) ====="

source ~/.bashrc
conda activate metricx

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech

python ${CODE}/convert_metricx_gigaspeech.py \
  --stream_dir ${BASE}/streaming_salami_dataset \
  --output     ${BASE}/metricx_input.jsonl \
  --keep-source-case

echo "metricx_input lines: $(wc -l < ${BASE}/metricx_input.jsonl)"

rm -rf  ${BASE}/metricx_shards
mkdir -p ${BASE}/metricx_shards
split -d -n l/8 \
  ${BASE}/metricx_input.jsonl \
  ${BASE}/metricx_shards/input_

ls -lh ${BASE}/metricx_shards/

echo "===== SALAMI ja Stage 2 DONE $(date) ====="
echo "Next: sbatch --array=0-7 stage3_metricx_ja.sh"
