#!/usr/bin/env bash
#SBATCH --job-name=metricx_qe_p24
#SBATCH --array=0-23%24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/metricx_qe_p24_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/metricx_qe_p24_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# MetricX QE predict — 24-way preempt array variant.
# Required env vars:
#   METRICX_RUN_DIR  -- contains metricx_shards/input_NN

set -e

echo "===== MetricX QE shard ${SLURM_ARRAY_TASK_ID:-0} (preempt 24-array) START $(date) ====="

CONDA_BASE="/home/haolingp/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metricx
echo "python=$(command -v python)"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR is required}"

SHARD_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-0}")
INPUT="${METRICX_RUN_DIR}/metricx_shards/input_${SHARD_ID}"
OUTPUT="${METRICX_RUN_DIR}/metricx_shards/output_${SHARD_ID}.jsonl"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing shard input: ${INPUT}"
  exit 1
fi

# Skip on requeue if already done.
if [[ -s "${OUTPUT}" ]]; then
  IN_LINES=$(wc -l < "${INPUT}")
  OUT_LINES=$(wc -l < "${OUTPUT}")
  if (( OUT_LINES >= IN_LINES )); then
    echo "[SKIP] shard ${SHARD_ID}: already complete (${OUT_LINES} >= ${IN_LINES})"
    exit 0
  fi
  echo "[RESUME] shard ${SHARD_ID}: partial output ${OUT_LINES}/${IN_LINES}; will overwrite"
fi

echo "METRICX_RUN_DIR=${METRICX_RUN_DIR}"
echo "INPUT =${INPUT}      ($(wc -l < ${INPUT}) lines)"
echo "OUTPUT=${OUTPUT}"

cd /home/haolingp/CMU_research_SMT/data_synthesis/codes/metricx
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer /data/user_data/haolingp/models/mt5-xl \
  --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --qe

echo "===== MetricX QE shard ${SLURM_ARRAY_TASK_ID:-0} DONE $(date) ====="
echo "wrote $(wc -l < ${OUTPUT}) lines"
