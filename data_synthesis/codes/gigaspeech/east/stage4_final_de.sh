#!/usr/bin/env bash
# ============================================================
# EAST  de  -- Stage 4: Merge MetricX outputs -> filter -> final dataset
# ============================================================
#SBATCH --job-name=east_de_s4_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH --exclude=babel-p9-28,babel-s5-32
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/stage4_final_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/stage4_final_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== EAST de Stage 4 START $(date) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech

# Sanity check: all 8 shards must exist
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

# Step 5: Merge 8 shard outputs
conda activate metricx

cat ${BASE}/metricx_shards/output_*.jsonl > ${BASE}/metricx_output.jsonl
echo "Merged MetricX output: $(wc -l < ${BASE}/metricx_output.jsonl) lines"

# Step 6: Filter by MetricX score
python ${CODE}/filter_metricx_gigaspeech.py \
  --input     ${BASE}/metricx_output.jsonl \
  --output    ${BASE}/metricx_filtered_t3.0.jsonl \
  --threshold 3.0
echo "Filtered: $(wc -l < ${BASE}/metricx_filtered_t3.0.jsonl) lines kept"

# Step 7: Build final dataset
conda deactivate
conda activate SMT

rm -rf ${BASE}/final_jsonl_east

python ${CODE}/final_output_gigaspeech.py \
  --metricx_jsonl ${BASE}/metricx_filtered_t3.0.jsonl \
  --stream_dir    ${BASE}/streaming_EAST_dataset \
  --output_dir    ${BASE}/final_jsonl_east

echo "Final dataset count: $(find ${BASE}/final_jsonl_east -name '*.json' | wc -l)"

echo "===== EAST de Stage 4 DONE $(date) ====="
echo "Final dataset: ${BASE}/final_jsonl_east"
