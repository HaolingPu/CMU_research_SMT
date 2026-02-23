#!/usr/bin/env bash
# ============================================================
# Salami  Stage 4 — Merge → filter → final dataset
#
# Input  : ${BASE}/metricx_shards/output_00.jsonl … output_07.jsonl
# Output : ${BASE}/metricx_output.jsonl
#          ${BASE}/metricx_filtered_t3.0.jsonl
#          ${BASE}/final_jsonl_salami
#          ${BASE}/check_final_report.json
#
# Submit : sbatch stage4_final.sh   (after ALL Stage 3 tasks complete)
# ============================================================
#SBATCH --job-name=salami_s4_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/stage4_final_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/stage4_final_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== Salami Stage 4 — START $(date) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami
CODE=/data/user_data/haolingp/data_synthesis/codes/gigaspeech
TSV=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv

# ── Sanity check: all 8 shards must exist ────────────────────
echo "Checking shard outputs..."
MISSING=0
for i in $(seq 0 7); do
  SHARD=$(printf "%02d" $i)
  if [[ ! -f ${BASE}/metricx_shards/output_${SHARD}.jsonl ]]; then
    echo "  ERROR: missing output_${SHARD}.jsonl"
    MISSING=$((MISSING + 1))
  fi
done
if [[ ${MISSING} -gt 0 ]]; then
  echo "ERROR: ${MISSING} shard(s) missing. Aborting."
  exit 1
fi
echo "  All 8 shards present."

# ── Step 5: Merge 8 shard outputs ────────────────────────────
conda activate metricx

cat ${BASE}/metricx_shards/output_*.jsonl \
  > ${BASE}/metricx_output.jsonl

echo "Merged MetricX output: $(wc -l < ${BASE}/metricx_output.jsonl) lines"

# ── Step 6: Filter by MetricX score ──────────────────────────
python ${CODE}/filter_metricx_gigaspeech.py \
  --input     ${BASE}/metricx_output.jsonl \
  --output    ${BASE}/metricx_filtered_t3.0.jsonl \
  --threshold 3.0

echo "Filtered: $(wc -l < ${BASE}/metricx_filtered_t3.0.jsonl) lines kept"

# ── Step 7: Build final dataset ──────────────────────────────
conda deactivate
conda activate SMT

rm -rf ${BASE}/final_jsonl_salami

python ${CODE}/final_output_gigaspeech.py \
  --metricx_jsonl ${BASE}/metricx_filtered_t3.0.jsonl \
  --stream_dir    ${BASE}/streaming_salami_dataset \
  --output_dir    ${BASE}/final_jsonl_salami

# ── Step 7-check: Quality check on final output ──────────────
python ${CODE}/check_salami_final.py \
  --tsv       ${TSV} \
  --manifest  ${BASE}/metricx_filtered_t3.0.jsonl \
  --final_dir ${BASE}/final_jsonl_salami \
  --report    ${BASE}/check_final_report.json \
  || true

echo "===== Salami Stage 4 — DONE $(date) ====="
echo "Final dataset: ${BASE}/final_jsonl_salami"
echo "Quality report: ${BASE}/check_final_report.json"
