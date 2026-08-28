#!/usr/bin/env bash
# ============================================================
# Postprocess a consensus decoding directory:
# 1) Convert to MetricX QE input
# 2) Run MetricX QE
# 3) Summarize QE
# 4) Optionally copy consensus JSONs with QE <= threshold to output dir
# ============================================================
#SBATCH --job-name=cons_qe3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=18:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_qe3_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/cons_qe3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

CONDA_BASE="/home/haolingp/miniconda3"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
  exit 1
fi

conda activate metricx
echo "python=$(command -v python)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

EXPERIMENT_DIR="${EXPERIMENT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-2k}"
METRICX_RUN_DIR="${METRICX_RUN_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-2k-metricx}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/consensus-2k-qe3}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
FILTER_BY_QE="${FILTER_BY_QE:-1}"

CONVERTER="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/convert_metricx_consensus.py"
SUMMARY="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/summarize_metricx_qe.py"
FILTER_JSONS="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/filter_consensus_by_metricx_qe.py"
METRICX_ROOT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/metricx"
TOKENIZER="/data/user_data/haolingp/models/mt5-xl"
MODEL="/data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6"

mkdir -p "$(dirname "${METRICX_RUN_DIR}")"
mkdir -p "${METRICX_RUN_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs
if [[ "${FILTER_BY_QE}" == "1" ]]; then
  mkdir -p "${FILTERED_OUTPUT_DIR}"
fi

INPUT_JSONL="${METRICX_RUN_DIR}/metricx_input.jsonl"
OUTPUT_JSONL="${METRICX_RUN_DIR}/metricx_output.jsonl"
SUMMARY_TXT="${METRICX_RUN_DIR}/summary.txt"
SUMMARY_JSON="${METRICX_RUN_DIR}/summary.json"

echo "===== CONSENSUS QE3 POSTPROCESS START $(date) ====="
echo "Experiment dir      : ${EXPERIMENT_DIR}"
echo "MetricX run dir     : ${METRICX_RUN_DIR}"
echo "QE threshold        : ${QE_THRESHOLD}"
echo "Filter by QE        : ${FILTER_BY_QE}"
if [[ "${FILTER_BY_QE}" == "1" ]]; then
  echo "Filtered output dir : ${FILTERED_OUTPUT_DIR}"
fi

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
  --input "${METRICX_RUN_DIR}" \
  --latency future_sampling \
  --threshold "${QE_THRESHOLD}" \
  --output-json "${SUMMARY_JSON}" | tee "${SUMMARY_TXT}"

if [[ "${FILTER_BY_QE}" == "1" ]]; then
  python "${FILTER_JSONS}" \
    --metricx-output "${OUTPUT_JSONL}" \
    --output-dir "${FILTERED_OUTPUT_DIR}" \
    --threshold "${QE_THRESHOLD}" \
    --clean-output
else
  echo "Skipping QE-based JSON filtering."
fi

echo "===== CONSENSUS QE3 POSTPROCESS DONE $(date) ====="
