#!/usr/bin/env bash
# ============================================================
# Finalize MetricX QE summary after 8 shards complete.
# ============================================================
#SBATCH --job-name=cons_mfin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=12G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_mfin_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_mfin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

CONDA_BASE="/home/haolingp/miniconda3"
BASE_PYTHON="${CONDA_BASE}/bin/python"

METRICX_RUN_DIR="${METRICX_RUN_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-100-qe-metricx}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR:-}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS="${NUM_SHARDS:-4}"

SUMMARY_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/summarize_metricx_qe.py"
FILTER_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/filter_consensus_by_metricx_qe.py"
SUMMARY_TXT="${METRICX_RUN_DIR}/summary.txt"
SUMMARY_JSON="${METRICX_RUN_DIR}/summary.json"
MERGED_OUTPUT="${METRICX_RUN_DIR}/metricx_output.jsonl"

mkdir -p "${METRICX_RUN_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

echo "===== CONSENSUS METRICX FINALIZE START $(date) ====="
echo "MetricX run dir : ${METRICX_RUN_DIR}"

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

# --- Summarize ---
"${BASE_PYTHON}" "${SUMMARY_SCRIPT}" \
  --input "${METRICX_RUN_DIR}" \
  --latency future_sampling \
  --threshold "${QE_THRESHOLD}" \
  --output-json "${SUMMARY_JSON}" | tee "${SUMMARY_TXT}"

# --- Filter QE <= threshold ---
if [[ -n "${FILTERED_OUTPUT_DIR}" ]]; then
    echo "[Filter] Filtering QE <= ${QE_THRESHOLD} → ${FILTERED_OUTPUT_DIR}"
    mkdir -p "${FILTERED_OUTPUT_DIR}"
    "${BASE_PYTHON}" "${FILTER_SCRIPT}" \
      --metricx-output "${MERGED_OUTPUT}" \
      --output-dir "${FILTERED_OUTPUT_DIR}" \
      --threshold "${QE_THRESHOLD}" \
      --clean-output
    echo "[Filter] Kept $(ls "${FILTERED_OUTPUT_DIR}"/*.json 2>/dev/null | wc -l) files"
else
    echo "[Filter] Skipping (FILTERED_OUTPUT_DIR not set)"
fi

echo "===== CONSENSUS METRICX FINALIZE DONE $(date) ====="
