#!/usr/bin/env bash
#SBATCH --job-name=teacher_metricx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_metricx_%A_%a.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/synth/slurm_logs/teacher_metricx_%A_%a.err

set -euo pipefail

BASE="${BASE:?BASE is required}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-24}"
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_QE_SHARDS}" ]]; then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside NUM_QE_SHARDS=${NUM_QE_SHARDS}" >&2
  exit 2
fi

SHARD_ID=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")
INPUT="${BASE}/metricx_shards/input_${SHARD_ID}"
OUTPUT="${BASE}/metricx_shards/output_${SHARD_ID}.jsonl"
if [[ ! -s "${INPUT}" ]]; then
  echo "Missing or empty MetricX shard: ${INPUT}" >&2
  exit 1
fi

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate metricx
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

cd /home/haolingp/CMU_research_SMT/data_synthesis/codes/metricx
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer /data/user_data/haolingp/models/mt5-xl \
  --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --qe

test -s "${OUTPUT}"
echo "MetricX shard ${SHARD_ID} complete at $(date --iso-8601=seconds)"
