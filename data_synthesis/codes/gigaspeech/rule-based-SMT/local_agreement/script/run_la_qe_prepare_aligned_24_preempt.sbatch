#!/usr/bin/env bash
#SBATCH --job-name=la_qe_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=16G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_qe_prep_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_qe_prep_%j.err

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

ALIGNED_FILE="${ALIGNED_FILE:?ALIGNED_FILE required}"
CONSENSUS_ROOT="${CONSENSUS_ROOT:?CONSENSUS_ROOT required}"
METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR required}"
NUM_SHARDS="${NUM_SHARDS:-24}"

mkdir -p "${METRICX_RUN_DIR}" /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

INPUT_JSONL="${METRICX_RUN_DIR}/metricx_input.jsonl"
SHARD_DIR="${METRICX_RUN_DIR}/metricx_shards"

echo "===== LA aligned QE prepare START $(date) ====="
echo "ALIGNED_FILE=${ALIGNED_FILE}"
echo "CONSENSUS_ROOT=${CONSENSUS_ROOT}"
echo "METRICX_RUN_DIR=${METRICX_RUN_DIR}"
echo "NUM_SHARDS=${NUM_SHARDS}"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/convert_metricx_consensus_aligned.py \
  --aligned "${ALIGNED_FILE}" \
  --consensus-root "${CONSENSUS_ROOT}" \
  --output "${INPUT_JSONL}" \
  --require-stream-json

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: empty metricx_input: ${INPUT_JSONL}"
  exit 1
fi

rm -rf "${SHARD_DIR}"
mkdir -p "${SHARD_DIR}"
split -d -n "l/${NUM_SHARDS}" "${INPUT_JSONL}" "${SHARD_DIR}/input_"

echo "Lines in metricx_input.jsonl: $(wc -l < "${INPUT_JSONL}")"
echo "Shards written to: ${SHARD_DIR}"
echo "===== LA aligned QE prepare DONE $(date) ====="
