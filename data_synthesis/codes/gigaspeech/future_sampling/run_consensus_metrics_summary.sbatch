#!/usr/bin/env bash
# ============================================================
# Summarize BLEU / LAAL / MetricX QE for one consensus run.
# ============================================================
#SBATCH --job-name=cons_sum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=8G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_sum_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_sum_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

CONDA_BASE="/home/haolingp/miniconda3"
BASE_PYTHON="${CONDA_BASE}/bin/python"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-100-calib}"
METRICX_RUN_DIR="${METRICX_RUN_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-100-calib-metricx}"
SUMMARY_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/summarize_consensus_metrics.py"
SUMMARY_TXT="${SUMMARY_TXT:-${METRICX_RUN_DIR}/overall_metrics.txt}"

mkdir -p "$(dirname "${SUMMARY_TXT}")"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

echo "===== CONSENSUS METRICS SUMMARY START $(date) ====="
echo "Experiment dir : ${EXPERIMENT_DIR}"
echo "MetricX dir    : ${METRICX_RUN_DIR}"
echo "Summary txt    : ${SUMMARY_TXT}"

"${BASE_PYTHON}" "${SUMMARY_SCRIPT}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --metricx-dir "${METRICX_RUN_DIR}" | tee "${SUMMARY_TXT}"

echo "===== CONSENSUS METRICS SUMMARY DONE $(date) ====="
