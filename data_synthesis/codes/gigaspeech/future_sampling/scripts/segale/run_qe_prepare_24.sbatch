#!/usr/bin/env bash
#SBATCH --job-name=segale_qe_prep24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_qe_prep24_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_qe_prep24_%j.err

# Convert merged aligned_all.jsonl + consensus_root -> metricx per-sentence input,
# then split into 24 shards for metricx predict.
#
# Required env vars:
#   ALIGNED_FILE      -- path to aligned_all.jsonl
#   CONSENSUS_ROOT    -- consensus generation root (J 40k root)
#   METRICX_RUN_DIR   -- output dir

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

ALIGNED_FILE="${ALIGNED_FILE:?ALIGNED_FILE required}"
CONSENSUS_ROOT="${CONSENSUS_ROOT:?CONSENSUS_ROOT required}"
METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR required}"
NUM_SHARDS=24

mkdir -p "${METRICX_RUN_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs

INPUT_JSONL="${METRICX_RUN_DIR}/metricx_input.jsonl"
SHARD_DIR="${METRICX_RUN_DIR}/metricx_shards"

echo "===== SEGALE QE prepare (24-shard) START $(date) ====="
echo "ALIGNED_FILE    : ${ALIGNED_FILE}"
echo "CONSENSUS_ROOT  : ${CONSENSUS_ROOT}"
echo "METRICX_RUN_DIR : ${METRICX_RUN_DIR}"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/convert_metricx_consensus_aligned.py \
  --aligned        "${ALIGNED_FILE}" \
  --consensus-root "${CONSENSUS_ROOT}" \
  --output         "${INPUT_JSONL}" \
  --require-stream-json

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: empty metricx_input: ${INPUT_JSONL}"
  exit 1
fi

rm -rf "${SHARD_DIR}"
mkdir -p "${SHARD_DIR}"
split -d -a 2 -n "l/${NUM_SHARDS}" "${INPUT_JSONL}" "${SHARD_DIR}/input_"

echo "Lines in metricx_input.jsonl : $(wc -l < ${INPUT_JSONL})"
echo "Shards written to            : ${SHARD_DIR}"
ls "${SHARD_DIR}" | head -5
echo "===== SEGALE QE prepare DONE $(date) ====="
