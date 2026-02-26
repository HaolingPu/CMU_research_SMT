#!/usr/bin/env bash
# ============================================================
# Salami  Stage 2 — Streaming dataset + MetricX input
#
# Input  : ${BASE}/llm_output_offline_merged_fixed
#          ${BASE}/good_train_xl_salami_mfa.jsonl
# Output : ${BASE}/streaming_salami_dataset
#          ${BASE}/metricx_shards/input_00 … input_07  (→ Stage 3)
#
# Submit : sbatch stage2_streaming.sh
# Then   : sbatch --array=0-7 stage3_metricx.sh
# ============================================================
#SBATCH --job-name=salami_s2_stream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/stage2_stream_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/stage2_stream_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== Salami Stage 2 — START $(date) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami
CODE=/data/user_data/haolingp/data_synthesis/codes/gigaspeech
TSV=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv

# ── Step 4: Build streaming trajectory (chunk = 960 ms) ──────
conda activate SMT

rm -rf ${BASE}/streaming_salami_dataset

python ${CODE}/multi_trajectory_gigaspeech.py \
  --llm-dir    ${BASE}/llm_output_offline_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --good-jsonl ${BASE}/good_train_xl_salami_mfa.jsonl \
  --output-dir ${BASE}/streaming_salami_dataset \
  --chunk-ms 960 \
  --overwrite

# ── Step 4-check: Quality check on streaming dataset ─────────
python ${CODE}/check_streaming_dataset.py \
  --stream_dir ${BASE}/streaming_salami_dataset \
  --tsv        ${TSV} \
  --report     ${BASE}/check_streaming_report.json \
  --zh_excess_threshold 2 \
  || true   # non-blocking

# ── Step 5: Convert to MetricX input ─────────────────────────
conda deactivate
conda activate metricx

python ${CODE}/convert_metricx_gigaspeech.py \
  --stream_dir ${BASE}/streaming_salami_dataset \
  --output     ${BASE}/metricx_input.jsonl

# ── Split into 8 shards ──────────────────────────────────────
rm -rf  ${BASE}/metricx_shards
mkdir -p ${BASE}/metricx_shards

split -d -n l/8 \
  ${BASE}/metricx_input.jsonl \
  ${BASE}/metricx_shards/input_

echo "Shards written:"
ls -lh ${BASE}/metricx_shards/

echo "===== Salami Stage 2 — DONE $(date) ====="
echo "Next: sbatch --array=0-7 stage3_metricx.sh"
