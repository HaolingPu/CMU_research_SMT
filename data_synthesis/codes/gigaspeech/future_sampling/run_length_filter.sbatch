#!/usr/bin/env bash
# ============================================================
# Length-ratio filter (CPU only). Runs after QE finalize.
# Reads QE-filtered per-utt JSONs and keeps only those within
# length_ratio bounds; copies survivors to ${LEN_FILTERED_DIR}.
# ============================================================
#SBATCH --job-name=cons_mlen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=8G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_mlen_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_mlen_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

CONDA_BASE="/home/haolingp/miniconda3"
BASE_PYTHON="${CONDA_BASE}/bin/python"

QE_FILTERED_DIR="${QE_FILTERED_DIR:?QE_FILTERED_DIR not set}"
LEN_FILTERED_DIR="${LEN_FILTERED_DIR:?LEN_FILTERED_DIR not set}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
MAX_RATIO_SRC="${MAX_RATIO_SRC:-2.5}"

FILTER_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/filter_consensus_by_length_ratio.py"
REPORT_JSONL="${LEN_FILTERED_DIR}/length_filter_report.jsonl"

mkdir -p "${LEN_FILTERED_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

echo "===== LENGTH FILTER START $(date) ====="
echo "QE-filtered input  : ${QE_FILTERED_DIR}"
echo "Length-filter out  : ${LEN_FILTERED_DIR}"
echo "max-ratio-ref (<=) : ${MAX_RATIO_REF}"
echo "max-ratio-src (<=) : ${MAX_RATIO_SRC}"

"${BASE_PYTHON}" "${FILTER_SCRIPT}" \
  --input-dir  "${QE_FILTERED_DIR}" \
  --output-dir "${LEN_FILTERED_DIR}" \
  --max-ratio-ref "${MAX_RATIO_REF}" \
  --max-ratio-src "${MAX_RATIO_SRC}" \
  --report "${REPORT_JSONL}" \
  --clean-output

KEPT=$(ls "${LEN_FILTERED_DIR}"/*.json 2>/dev/null | wc -l)
echo "Kept: ${KEPT}"
echo "Report: ${REPORT_JSONL}"
echo "===== LENGTH FILTER DONE $(date) ====="
