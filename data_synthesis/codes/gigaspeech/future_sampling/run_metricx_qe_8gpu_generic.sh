#!/usr/bin/env bash
# ============================================================
# Generic MetricX QE prediction (8-GPU array job)
# Usage:
#   sbatch --export=ALL,BASE_OUTPUT_DIR=/abs/output_dir         #     /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_8gpu_generic.sh
# ============================================================
#SBATCH --job-name=fs_metricx8
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/metricx_generic_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/metricx_generic_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== MetricX shard ${SLURM_ARRAY_TASK_ID:-0} START $(date) ====="

if [[ -z "${BASE_OUTPUT_DIR:-}" ]]; then
  echo "ERROR: BASE_OUTPUT_DIR is not set."
  echo "Submit with: sbatch --export=ALL,BASE_OUTPUT_DIR=/abs/output_dir $0"
  exit 1
fi

source ~/.bashrc
conda activate metricx

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

SHARD_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-0}")
INPUT="${BASE_OUTPUT_DIR}/metricx_shards/input_${SHARD_ID}"
OUTPUT="${BASE_OUTPUT_DIR}/metricx_shards/output_${SHARD_ID}.jsonl"

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing shard input: ${INPUT}"
  exit 1
fi

mkdir -p "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs"

echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "INPUT=${INPUT}"
echo "OUTPUT=${OUTPUT}"

cd /home/haolingp/CMU_research_SMT/data_synthesis/codes/metricx
PYTHONNOUSERSITE=1 python -m metricx24.predict           --tokenizer /data/user_data/haolingp/models/mt5-xl           --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6           --max_input_length 1536           --batch_size 1           --input_file "${INPUT}"           --output_file "${OUTPUT}"           --qe

echo "===== MetricX shard ${SLURM_ARRAY_TASK_ID:-0} DONE $(date) ====="
