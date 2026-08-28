#!/usr/bin/env bash
#SBATCH --job-name=la_seg_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=16G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_seg_prep_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_seg_prep_%j.err

set -e

CONSENSUS_ROOT="${CONSENSUS_ROOT:?CONSENSUS_ROOT required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
NUM_DOCS="${NUM_DOCS:-40000}"
NUM_SHARDS="${NUM_SHARDS:-24}"
SYS_ID="${SYS_ID:-la_rand40k}"

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/segale"
LA_SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement/script"

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale
export PYTHONNOUSERSITE=1

mkdir -p "${OUT_ROOT}" /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

echo "===== LA SEGALE prepare START $(date) ====="
echo "CONSENSUS_ROOT=${CONSENSUS_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "NUM_DOCS=${NUM_DOCS}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "SYS_ID=${SYS_ID}"

python "${LA_SCRIPT_DIR}/merge_la_outputs.py" \
  --input-root "${CONSENSUS_ROOT}" \
  --output "${OUT_ROOT}/all_generation.jsonl" \
  --num-docs "${NUM_DOCS}"

python "${SCRIPT_DIR}/prepare_segale_shards.py" \
  --consensus-root "${CONSENSUS_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --num-docs "${NUM_DOCS}" \
  --num-shards "${NUM_SHARDS}" \
  --sys-id "${SYS_ID}"

echo "===== LA SEGALE prepare DONE $(date) ====="
