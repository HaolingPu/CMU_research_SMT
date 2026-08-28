#!/usr/bin/env bash
#SBATCH --job-name=cons_metricx_aln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_metricx_aln_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_metricx_aln_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# One-shot MetricX QE on a pre-built per-sentence input + per-sentence filter.
# Required env vars:
#   METRICX_RUN_DIR    -- contains metricx_input.jsonl (already produced)
#   FILTERED_OUTPUT_DIR -- where to copy passed JSONs
# Optional:
#   QE_THRESHOLD       -- default 3.0
#   FILTER_SCRIPT      -- default per-sentence filter

set -e

source ~/.bashrc
conda activate metricx

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR is required}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR:?FILTERED_OUTPUT_DIR is required}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
FILTER_SCRIPT="${FILTER_SCRIPT:-/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/filter_consensus_by_metricx_qe_per_sentence.py}"

INPUT_JSONL="${METRICX_RUN_DIR}/metricx_input.jsonl"
OUTPUT_JSONL="${METRICX_RUN_DIR}/metricx_output.jsonl"
REPORT_JSONL="${METRICX_RUN_DIR}/filter_report_per_utt.jsonl"

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: missing or empty input: ${INPUT_JSONL}"
  exit 1
fi

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs
mkdir -p "${FILTERED_OUTPUT_DIR}"

echo "===== MetricX QE (aligned) START $(date) ====="
echo "INPUT  : ${INPUT_JSONL}"
echo "OUTPUT : ${OUTPUT_JSONL}"
echo "FILTER OUTPUT DIR : ${FILTERED_OUTPUT_DIR}"
echo "QE_THRESHOLD : ${QE_THRESHOLD}"

cd /home/haolingp/CMU_research_SMT/data_synthesis/codes/metricx
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer /data/user_data/haolingp/models/mt5-xl \
  --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file "${INPUT_JSONL}" \
  --output_file "${OUTPUT_JSONL}" \
  --qe

python "${FILTER_SCRIPT}" \
  --metricx-output "${OUTPUT_JSONL}" \
  --output-dir "${FILTERED_OUTPUT_DIR}" \
  --threshold "${QE_THRESHOLD}" \
  --clean-output \
  --report "${REPORT_JSONL}"

echo "===== MetricX QE (aligned) DONE $(date) ====="
