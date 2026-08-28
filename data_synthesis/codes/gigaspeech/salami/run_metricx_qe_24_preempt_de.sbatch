#!/usr/bin/env bash
#SBATCH --job-name=salami_seg_qe_de
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-t9-16,babel-p5-28
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_qe_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_qe_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

METRICX_RUN_DIR=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/segale_pipeline/metricx_aligned

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

SHARD_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-0}")
INPUT="${METRICX_RUN_DIR}/metricx_shards/input_${SHARD_ID}"
OUTPUT="${METRICX_RUN_DIR}/metricx_shards/output_${SHARD_ID}.jsonl"

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing shard input: ${INPUT}"
  exit 1
fi

if [[ -s "${OUTPUT}" ]]; then
  HAVE=$(wc -l < "${OUTPUT}")
  WANT=$(wc -l < "${INPUT}")
  if (( HAVE >= WANT )); then
    echo "Shard already complete ($HAVE >= $WANT), skipping."
    exit 0
  fi
fi

echo "===== SALAMI de seg MetricX QE shard ${SHARD_ID} START $(date) ====="

cd /home/haolingp/CMU_research_SMT/data_synthesis/codes/metricx
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer /data/user_data/haolingp/models/mt5-xl \
  --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --qe

echo "===== SALAMI de seg MetricX QE shard ${SHARD_ID} DONE $(date) ====="
