#!/usr/bin/env bash
#SBATCH --job-name=segale_qe_fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=12G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_qe_fin_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_qe_fin_%j.err

# Merge metricx shards -> per-sentence filter (any sentence>thr drops case)
# -> length-ratio filter on the survivors.
#
# Required env vars:
#   METRICX_RUN_DIR        -- contains metricx_shards/output_NN.jsonl after predict
#   QE_FILTERED_DIR        -- output dir for QE-passed JSONs
#   FINAL_OUT_DIR          -- output dir for QE+length-ratio passed JSONs
# Optional:
#   QE_THRESHOLD            (default 3.0)
#   MAX_RATIO_REF           (default 1.5)
#   MIN_RATIO_REF           (default 0.7)

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
NUM_SHARDS=8

CODE=/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling
MERGED="${METRICX_RUN_DIR}/metricx_output.jsonl"
QE_REPORT="${METRICX_RUN_DIR}/filter_report_per_utt.jsonl"
LR_REPORT="${METRICX_RUN_DIR}/length_ratio_report.jsonl"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs
mkdir -p "${QE_FILTERED_DIR}" "${FINAL_OUT_DIR}"

echo "===== SEGALE QE finalize START $(date) ====="
echo "METRICX_RUN_DIR  : ${METRICX_RUN_DIR}"
echo "QE_FILTERED_DIR  : ${QE_FILTERED_DIR}"
echo "FINAL_OUT_DIR    : ${FINAL_OUT_DIR}"
echo "QE_THRESHOLD     : ${QE_THRESHOLD}"
echo "MAX_RATIO_REF    : ${MAX_RATIO_REF}"
echo "MIN_RATIO_REF    : ${MIN_RATIO_REF}"

echo "[Merge] combining ${NUM_SHARDS} metricx shard outputs..."
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
echo "[Merge] merged: $(wc -l < ${MERGED}) lines"

echo "[Filter QE] keep cases where every sentence QE <= ${QE_THRESHOLD}"
python "${CODE}/filter_consensus_by_metricx_qe_per_sentence.py" \
  --metricx-output "${MERGED}" \
  --output-dir     "${QE_FILTERED_DIR}" \
  --threshold      "${QE_THRESHOLD}" \
  --clean-output \
  --report         "${QE_REPORT}"
echo "[Filter QE] kept $(ls ${QE_FILTERED_DIR}/*.json 2>/dev/null | wc -l) JSONs"

echo "[Filter length-ratio] ${MIN_RATIO_REF} <= length_ratio_ref <= ${MAX_RATIO_REF}"
python "${CODE}/filter_consensus_by_length_ratio.py" \
  --input-dir      "${QE_FILTERED_DIR}" \
  --output-dir     "${FINAL_OUT_DIR}" \
  --max-ratio-ref  "${MAX_RATIO_REF}" \
  --min-ratio-ref  "${MIN_RATIO_REF}" \
  --clean-output \
  --report         "${LR_REPORT}"
echo "[Final] kept $(ls ${FINAL_OUT_DIR}/*.json 2>/dev/null | wc -l) JSONs"

echo "===== SEGALE QE finalize DONE $(date) ====="
