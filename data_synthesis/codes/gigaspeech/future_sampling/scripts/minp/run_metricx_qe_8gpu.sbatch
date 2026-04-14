#!/usr/bin/env bash
#SBATCH --job-name=minp_mqe8
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/minp_mqe8_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/minp_mqe8_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== MINP MetricX shard ${SLURM_ARRAY_TASK_ID:-0} START $(date) ====="

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

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR is required}"

SHARD_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-0}")
INPUT="${METRICX_RUN_DIR}/metricx_shards/input_${SHARD_ID}"
OUTPUT="${METRICX_RUN_DIR}/metricx_shards/output_${SHARD_ID}.jsonl"

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing shard input: ${INPUT}"
  exit 1
fi

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

echo "METRICX_RUN_DIR=${METRICX_RUN_DIR}"
echo "INPUT=${INPUT}"
echo "OUTPUT=${OUTPUT}"

cd /data/user_data/haolingp/data_synthesis/codes/metricx
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer /data/user_data/haolingp/models/mt5-xl \
  --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --qe

echo "===== MINP MetricX shard ${SLURM_ARRAY_TASK_ID:-0} DONE $(date) ====="
