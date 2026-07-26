#!/usr/bin/env bash
# ============================================================
# MetricX QE for consensus decoding outputs (single GPU)
#
# Submit:
#   sbatch --export=ALL,EXPERIMENT_DIR=/abs/consensus_dir \
#     /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_consensus_single.sbatch
# ============================================================
#SBATCH --job-name=cons_metricx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/metricx-qe/slurm_logs/metricx_consensus_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/metricx-qe/slurm_logs/metricx_consensus_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate metricx

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

EXPERIMENT_DIR="${EXPERIMENT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/m10_empty_prefix_probe_fix_6929779}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/metricx-qe}"
EXPERIMENT_NAME="$(basename "${EXPERIMENT_DIR}")"
RUN_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}"

CONVERTER="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/convert_metricx_consensus.py"
SUMMARY="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/summarize_metricx_qe.py"
METRICX_ROOT="/data/user_data/haolingp/data_synthesis/codes/metricx"
TOKENIZER="/data/user_data/haolingp/models/mt5-xl"
MODEL="/data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6"

mkdir -p "${OUTPUT_BASE}/slurm_logs"
mkdir -p "${RUN_DIR}"

INPUT_JSONL="${RUN_DIR}/metricx_input.jsonl"
OUTPUT_JSONL="${RUN_DIR}/metricx_output.jsonl"
SUMMARY_TXT="${RUN_DIR}/summary.txt"
SUMMARY_JSON="${RUN_DIR}/summary.json"

echo "===== MetricX QE START $(date) ====="
echo "Experiment dir : ${EXPERIMENT_DIR}"
echo "Run dir        : ${RUN_DIR}"

python "${CONVERTER}" \
  --input-dir "${EXPERIMENT_DIR}" \
  --output "${INPUT_JSONL}"

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: empty MetricX input: ${INPUT_JSONL}"
  exit 1
fi

cd "${METRICX_ROOT}"
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer "${TOKENIZER}" \
  --model_name_or_path "${MODEL}" \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file "${INPUT_JSONL}" \
  --output_file "${OUTPUT_JSONL}" \
  --qe

python "${SUMMARY}" \
  --input "${RUN_DIR}" \
  --latency future_sampling \
  --output-json "${SUMMARY_JSON}" | tee "${SUMMARY_TXT}"

echo "===== MetricX QE DONE $(date) ====="
