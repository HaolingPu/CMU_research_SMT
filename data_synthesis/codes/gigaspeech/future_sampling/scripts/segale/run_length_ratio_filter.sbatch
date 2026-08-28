#!/usr/bin/env bash
#SBATCH --job-name=segale_lr_fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=00:30:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_lr_fin_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_lr_fin_%j.err

# Length-ratio filter on top of QE-MAX output.
# Required env vars:
#   INPUT_DIR   -- dir of QE-MAX-passed JSONs (e.g. .../qe3-aligned-max)
#   OUTPUT_DIR  -- dir for length-ratio-passed JSONs (e.g. .../qe3-lr-aligned-max)
# Optional:
#   MAX_RATIO_REF (default 1.5)
#   MIN_RATIO_REF (default 0.7)

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

INPUT_DIR="${INPUT_DIR:?INPUT_DIR required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
MIN_RATIO_REF="${MIN_RATIO_REF:-0.7}"

CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling
REPORT_DIR="$(dirname "${OUTPUT_DIR}")/metricx-aligned"
mkdir -p "${OUTPUT_DIR}" "${REPORT_DIR}" \
         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs

echo "===== Length-Ratio Filter ====="
echo "Input dir   : ${INPUT_DIR}"
echo "Total JSONs : $(ls ${INPUT_DIR}/*.json 2>/dev/null | wc -l)"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Range       : ${MIN_RATIO_REF} <= length_ratio_ref <= ${MAX_RATIO_REF}"

python "${CODE}/filter_consensus_by_length_ratio.py" \
  --input-dir      "${INPUT_DIR}" \
  --output-dir     "${OUTPUT_DIR}" \
  --max-ratio-ref  "${MAX_RATIO_REF}" \
  --min-ratio-ref  "${MIN_RATIO_REF}" \
  --clean-output \
  --report         "${REPORT_DIR}/length_ratio_report.jsonl"

echo "[LR_FINAL] kept $(ls ${OUTPUT_DIR}/*.json 2>/dev/null | wc -l) JSONs"
