#!/usr/bin/env bash
#SBATCH --job-name=minp_mfin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/minp_mfin_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/minp_mfin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

CONDA_BASE="/home/haolingp/miniconda3"
BASE_PYTHON="${CONDA_BASE}/bin/python"

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR is required}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?EXPERIMENT_DIR is required}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR:?FILTERED_OUTPUT_DIR is required}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS=8

SUMMARY_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/summarize_metricx_qe.py"
FILTER_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/filter_consensus_by_metricx_qe.py"
SUMMARY_TXT="${METRICX_RUN_DIR}/summary.txt"
SUMMARY_JSON="${METRICX_RUN_DIR}/summary.json"
MERGED_OUTPUT="${METRICX_RUN_DIR}/metricx_output.jsonl"

mkdir -p "${METRICX_RUN_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

echo "===== MINP METRICX FINALIZE START $(date) ====="
echo "MetricX run dir     : ${METRICX_RUN_DIR}"
echo "Experiment dir      : ${EXPERIMENT_DIR}"
echo "Filtered output dir : ${FILTERED_OUTPUT_DIR}"
echo "QE threshold        : ${QE_THRESHOLD}"
echo "Num shards          : ${NUM_SHARDS}"

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

"${BASE_PYTHON}" "${SUMMARY_SCRIPT}" \
  --input "${METRICX_RUN_DIR}" \
  --latency future_sampling \
  --threshold "${QE_THRESHOLD}" \
  --output-json "${SUMMARY_JSON}" | tee "${SUMMARY_TXT}"

echo "[Filter] Filtering QE <= ${QE_THRESHOLD} -> ${FILTERED_OUTPUT_DIR}"
mkdir -p "${FILTERED_OUTPUT_DIR}"
"${BASE_PYTHON}" "${FILTER_SCRIPT}" \
  --metricx-output "${MERGED_OUTPUT}" \
  --output-dir "${FILTERED_OUTPUT_DIR}" \
  --threshold "${QE_THRESHOLD}" \
  --clean-output
echo "[Filter] Kept $(ls "${FILTERED_OUTPUT_DIR}"/*.json 2>/dev/null | wc -l) files"

echo "===== MINP METRICX FINALIZE DONE $(date) ====="
