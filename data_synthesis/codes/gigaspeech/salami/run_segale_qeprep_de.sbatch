#!/usr/bin/env bash
#SBATCH --job-name=salami_seg_qeprep_de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_qeprep_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_qeprep_%j.err

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de
ROOT=${BASE}/segale_pipeline
ALIGNED=${ROOT}/aligned_all.jsonl
CONSENSUS_ROOT=${ROOT}/consensus_format
METRICX_RUN_DIR=${ROOT}/metricx_aligned
NUM_SHARDS=24

mkdir -p ${METRICX_RUN_DIR}
INPUT_JSONL=${METRICX_RUN_DIR}/metricx_input.jsonl
SHARD_DIR=${METRICX_RUN_DIR}/metricx_shards

echo "===== SALAMI de SEGALE QE prepare START $(date) ====="
echo "ALIGNED        : ${ALIGNED}"
echo "CONSENSUS_ROOT : ${CONSENSUS_ROOT}"
echo "METRICX_RUN_DIR: ${METRICX_RUN_DIR}"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/convert_metricx_consensus_aligned.py \
  --aligned        ${ALIGNED} \
  --consensus-root ${CONSENSUS_ROOT} \
  --output         ${INPUT_JSONL} \
  --require-stream-json

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: empty metricx_input: ${INPUT_JSONL}"
  exit 1
fi

rm -rf ${SHARD_DIR}
mkdir -p ${SHARD_DIR}
split -d -a 2 -n l/${NUM_SHARDS} ${INPUT_JSONL} ${SHARD_DIR}/input_

echo "Lines in metricx_input.jsonl : $(wc -l < ${INPUT_JSONL})"
echo "Shards written to            : ${SHARD_DIR}"
echo "===== SALAMI de SEGALE QE prepare DONE $(date) ====="
