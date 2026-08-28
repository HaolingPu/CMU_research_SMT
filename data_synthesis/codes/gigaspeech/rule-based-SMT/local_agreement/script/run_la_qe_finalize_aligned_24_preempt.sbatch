#!/usr/bin/env bash
#SBATCH --job-name=la_qe_fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=16G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_qe_fin_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_qe_fin_%j.err

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR required}"
QE_FILTERED_DIR="${QE_FILTERED_DIR:?QE_FILTERED_DIR required}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:?FINAL_OUT_DIR required}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
MIN_RATIO_REF="${MIN_RATIO_REF:-0.7}"
NUM_SHARDS="${NUM_SHARDS:-24}"

CODE="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
MERGED="${METRICX_RUN_DIR}/metricx_output.jsonl"
QE_REPORT="${METRICX_RUN_DIR}/filter_report_per_utt.jsonl"
LR_REPORT="${METRICX_RUN_DIR}/length_ratio_report.jsonl"

mkdir -p "${QE_FILTERED_DIR}" "${FINAL_OUT_DIR}" /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

echo "===== LA QE finalize START $(date) ====="
echo "METRICX_RUN_DIR=${METRICX_RUN_DIR}"
echo "QE_FILTERED_DIR=${QE_FILTERED_DIR}"
echo "FINAL_OUT_DIR=${FINAL_OUT_DIR}"
echo "QE_THRESHOLD=${QE_THRESHOLD}"
echo "MIN_RATIO_REF=${MIN_RATIO_REF}"
echo "MAX_RATIO_REF=${MAX_RATIO_REF}"
echo "NUM_SHARDS=${NUM_SHARDS}"

rm -f "${MERGED}"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_ID=$(printf "%02d" "$i")
  SHARD_FILE="${METRICX_RUN_DIR}/metricx_shards/output_${SHARD_ID}.jsonl"
  if [[ -f "${SHARD_FILE}" ]]; then
    cat "${SHARD_FILE}" >> "${MERGED}"
  else
    echo "[WARN] missing shard: ${SHARD_FILE}"
  fi
done
echo "Merged MetricX rows: $(wc -l < "${MERGED}")"

python "${CODE}/filter_consensus_by_metricx_qe_per_sentence.py" \
  --metricx-output "${MERGED}" \
  --output-dir "${QE_FILTERED_DIR}" \
  --threshold "${QE_THRESHOLD}" \
  --clean-output \
  --report "${QE_REPORT}"

python "${CODE}/filter_consensus_by_length_ratio.py" \
  --input-dir "${QE_FILTERED_DIR}" \
  --output-dir "${FINAL_OUT_DIR}" \
  --max-ratio-ref "${MAX_RATIO_REF}" \
  --min-ratio-ref "${MIN_RATIO_REF}" \
  --clean-output \
  --report "${LR_REPORT}"

echo "QE kept: $(find "${QE_FILTERED_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "Final kept: $(find "${FINAL_OUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "===== LA QE finalize DONE $(date) ====="
