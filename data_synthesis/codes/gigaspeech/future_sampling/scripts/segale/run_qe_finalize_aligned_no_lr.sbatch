#!/usr/bin/env bash
#SBATCH --job-name=segale_qe_fin_nolr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=12G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_qe_fin_nolr_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_qe_fin_nolr_%j.err

# Merge metricx shards -> per-sentence QE filter (drop any case with sent>thr).
# No length-ratio filter (use this when the target IS the reference, so the
# length ratio is meaningless / always ~1).
#
# Required env vars:
#   METRICX_RUN_DIR   -- contains metricx_shards/output_NN.jsonl after predict
#   FINAL_OUT_DIR     -- output dir for QE-passed JSONs (final training data)
# Optional:
#   QE_THRESHOLD      (default 3.0)

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR required}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:?FINAL_OUT_DIR required}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS=8

CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling
MERGED="${METRICX_RUN_DIR}/metricx_output.jsonl"
QE_REPORT="${METRICX_RUN_DIR}/filter_report_per_utt.jsonl"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs
mkdir -p "${FINAL_OUT_DIR}"

echo "===== SEGALE QE finalize (no-LR) START $(date) ====="
echo "METRICX_RUN_DIR  : ${METRICX_RUN_DIR}"
echo "FINAL_OUT_DIR    : ${FINAL_OUT_DIR}"
echo "QE_THRESHOLD     : ${QE_THRESHOLD}"

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
  --output-dir     "${FINAL_OUT_DIR}" \
  --threshold      "${QE_THRESHOLD}" \
  --clean-output \
  --report         "${QE_REPORT}"
echo "[Final] kept $(ls ${FINAL_OUT_DIR}/*.json 2>/dev/null | wc -l) JSONs"

echo "===== SEGALE QE finalize (no-LR) DONE $(date) ====="
