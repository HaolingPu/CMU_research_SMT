#!/usr/bin/env bash
# ============================================================
# SALAMI de — Stage 4: merge MetricX outputs → filter → final
# ============================================================
#SBATCH --job-name=salami_de_s4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/stage4_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/stage4_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== SALAMI de Stage 4 START $(date) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech

MISSING=0
for i in $(seq 0 7); do
  S=$(printf "%02d" $i)
  if [[ ! -f ${BASE}/metricx_shards/output_${S}.jsonl ]]; then
    echo "  ERROR: missing output_${S}.jsonl"
    MISSING=$((MISSING + 1))
  fi
done
if [[ ${MISSING} -gt 0 ]]; then
  echo "ERROR: ${MISSING} shard(s) missing. Aborting."
  exit 1
fi
echo "All 8 shards present."

conda activate metricx
cat ${BASE}/metricx_shards/output_*.jsonl > ${BASE}/metricx_output.jsonl
echo "Merged: $(wc -l < ${BASE}/metricx_output.jsonl) lines"

python ${CODE}/filter_metricx_gigaspeech.py \
  --input     ${BASE}/metricx_output.jsonl \
  --output    ${BASE}/metricx_filtered_t3.0.jsonl \
  --threshold 3.0
echo "Filtered: $(wc -l < ${BASE}/metricx_filtered_t3.0.jsonl) lines kept"

conda deactivate
conda activate SMT
rm -rf ${BASE}/final_jsonl_salami
python ${CODE}/final_output_gigaspeech.py \
  --metricx_jsonl ${BASE}/metricx_filtered_t3.0.jsonl \
  --stream_dir    ${BASE}/streaming_salami_dataset \
  --output_dir    ${BASE}/final_jsonl_salami
echo "Final: $(find ${BASE}/final_jsonl_salami -name '*.jsonl' | wc -l) files"

echo "===== SALAMI de Stage 4 DONE $(date) ====="
echo "Final dataset: ${BASE}/final_jsonl_salami"
