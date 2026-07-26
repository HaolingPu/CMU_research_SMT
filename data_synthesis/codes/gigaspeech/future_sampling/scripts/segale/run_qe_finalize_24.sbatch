#!/usr/bin/env bash
#SBATCH --job-name=segale_qe_fin24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_qe_fin24_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_qe_fin24_%j.err

# Merge 24 metricx shards -> per-sentence MAX-QE filter.
# Filter semantics: keep an utterance iff max(per-sentence QE) <= QE_THRESHOLD.
# This is the existing filter_consensus_by_metricx_qe_per_sentence.py behavior.
#
# Required env vars:
#   METRICX_RUN_DIR  -- contains metricx_shards/output_NN.jsonl
#   QE_FILTERED_DIR  -- output dir for QE-passed JSONs
# Optional:
#   QE_THRESHOLD      (default 3.0  ← MAX threshold per the user spec)

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR required}"
QE_FILTERED_DIR="${QE_FILTERED_DIR:?QE_FILTERED_DIR required}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS=24

CODE=/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling
MERGED="${METRICX_RUN_DIR}/metricx_output.jsonl"
QE_REPORT="${METRICX_RUN_DIR}/filter_report_per_utt.jsonl"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs
mkdir -p "${QE_FILTERED_DIR}"

echo "===== SEGALE QE finalize (24-shard, MAX filter) START $(date) ====="
echo "METRICX_RUN_DIR  : ${METRICX_RUN_DIR}"
echo "QE_FILTERED_DIR  : ${QE_FILTERED_DIR}"
echo "QE_THRESHOLD     : ${QE_THRESHOLD}  (MAX semantics: drop if any sentence > threshold)"

echo "[Merge] combining ${NUM_SHARDS} metricx shard outputs..."
rm -f "${MERGED}"
missing=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_ID=$(printf "%02d" "$i")
  SHARD_FILE="${METRICX_RUN_DIR}/metricx_shards/output_${SHARD_ID}.jsonl"
  if [[ -f "${SHARD_FILE}" ]]; then
    cat "${SHARD_FILE}" >> "${MERGED}"
  else
    echo "[WARN] missing shard: ${SHARD_FILE}"
    missing=$((missing+1))
  fi
done
echo "[Merge] merged: $(wc -l < ${MERGED}) lines  (missing shards: ${missing})"

echo "[Filter QE-MAX] keep cases where MAX per-sentence QE <= ${QE_THRESHOLD}"
python "${CODE}/filter_consensus_by_metricx_qe_per_sentence.py" \
  --metricx-output "${MERGED}" \
  --output-dir     "${QE_FILTERED_DIR}" \
  --threshold      "${QE_THRESHOLD}" \
  --clean-output \
  --report         "${QE_REPORT}"

KEPT=$(ls "${QE_FILTERED_DIR}"/*.json 2>/dev/null | wc -l)
echo "[Filter QE-MAX] kept ${KEPT} JSONs"
echo "===== SEGALE QE finalize DONE $(date) ====="
