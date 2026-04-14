#!/usr/bin/env bash
#SBATCH --job-name=minp_mprep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/minp_mprep_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/minp_mprep_%j.err
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

EXPERIMENT_DIR="${EXPERIMENT_DIR:?EXPERIMENT_DIR is required}"
METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR is required}"
NUM_SHARDS=8

CONVERTER="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/convert_metricx_consensus.py"
INPUT_JSONL="${METRICX_RUN_DIR}/metricx_input.jsonl"
SHARD_DIR="${METRICX_RUN_DIR}/metricx_shards"

mkdir -p "${METRICX_RUN_DIR}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

echo "===== MINP METRICX PREP START $(date) ====="
echo "Experiment dir : ${EXPERIMENT_DIR}"
echo "MetricX run dir: ${METRICX_RUN_DIR}"
echo "Num shards     : ${NUM_SHARDS}"

python "${CONVERTER}" \
  --input-dir "${EXPERIMENT_DIR}" \
  --output "${INPUT_JSONL}"

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: empty MetricX input: ${INPUT_JSONL}"
  exit 1
fi

rm -rf "${SHARD_DIR}"
mkdir -p "${SHARD_DIR}"
split -d -n l/${NUM_SHARDS} "${INPUT_JSONL}" "${SHARD_DIR}/input_"

echo "Prepared MetricX input : ${INPUT_JSONL}"
echo "Prepared shard dir     : ${SHARD_DIR}"
echo "===== MINP METRICX PREP DONE $(date) ====="
