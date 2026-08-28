#!/usr/bin/env bash
# ============================================================
# Finalize MetricX QE (per-sentence) after shards complete.
# Merges shard outputs, runs per-utt AND-aggregation filter:
#   keep iff every sentence's QE <= threshold.
# Emits a per-utt report jsonl for threshold inspection.
# ============================================================
#SBATCH --job-name=cons_mfin_ps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=12G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_mfin_ps_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_mfin_ps_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

CONDA_BASE="/home/haolingp/miniconda3"
BASE_PYTHON="${CONDA_BASE}/bin/python"

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR not set}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR:?FILTERED_OUTPUT_DIR not set}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS="${NUM_SHARDS:-4}"

FILTER_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/filter_consensus_by_metricx_qe_per_sentence.py"
MERGED_OUTPUT="${METRICX_RUN_DIR}/metricx_output.jsonl"
REPORT_JSONL="${METRICX_RUN_DIR}/filter_report_per_utt.jsonl"

mkdir -p "${METRICX_RUN_DIR}"
mkdir -p "${FILTERED_OUTPUT_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

echo "===== CONSENSUS METRICX FINALIZE (per-sentence) START $(date) ====="
echo "MetricX run dir     : ${METRICX_RUN_DIR}"
echo "Filtered output dir : ${FILTERED_OUTPUT_DIR}"
echo "Threshold (<=)      : ${QE_THRESHOLD}"

# --- Merge shard outputs ---
echo "[Merge] Merging ${NUM_SHARDS} shard outputs ..."
rm -f "${MERGED_OUTPUT}"
for i in $(seq 0 $(( NUM_SHARDS - 1 ))); do
    SHARD_ID=$(printf "%02d" "${i}")
    SHARD_FILE="${METRICX_RUN_DIR}/metricx_shards/output_${SHARD_ID}.jsonl"
    if [[ -f "${SHARD_FILE}" ]]; then
        cat "${SHARD_FILE}" >> "${MERGED_OUTPUT}"
    else
        echo "[WARN] Missing shard: ${SHARD_FILE}"
    fi
done
echo "[Merge] Merged output: $(wc -l < "${MERGED_OUTPUT}") lines"

# --- Per-utt AND filter ---
echo "[Filter] Per-sentence QE AND threshold ${QE_THRESHOLD} -> ${FILTERED_OUTPUT_DIR}"
"${BASE_PYTHON}" "${FILTER_SCRIPT}" \
  --metricx-output "${MERGED_OUTPUT}" \
  --output-dir "${FILTERED_OUTPUT_DIR}" \
  --threshold "${QE_THRESHOLD}" \
  --report "${REPORT_JSONL}" \
  --clean-output

KEPT_COUNT=$(ls "${FILTERED_OUTPUT_DIR}"/*.json 2>/dev/null | wc -l)
echo "[Filter] Kept: ${KEPT_COUNT}"
echo "[Filter] Report: ${REPORT_JSONL}"

echo "===== CONSENSUS METRICX FINALIZE (per-sentence) DONE $(date) ====="
