#!/usr/bin/env bash
#SBATCH --job-name=segale_prep_p24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_prep_p24_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_prep_p24_%j.err

# Stage 1 — convert consensus JSONs into 24 SEGALE shards.
# Required env vars:
#   CONSENSUS_ROOT
#   OUT_ROOT
#   NUM_DOCS
#   SYS_ID

set -e

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale
export PYTHONNOUSERSITE=1

CONSENSUS_ROOT="${CONSENSUS_ROOT:?CONSENSUS_ROOT required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
NUM_DOCS="${NUM_DOCS:-40000}"
SYS_ID="${SYS_ID:-$(basename ${CONSENSUS_ROOT})}"
NUM_SHARDS=24

mkdir -p "${OUT_ROOT}" \
         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs

echo "===== SEGALE prepare (24-shard) START $(date) ====="
echo "CONSENSUS_ROOT : ${CONSENSUS_ROOT}"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "NUM_DOCS       : ${NUM_DOCS}"
echo "SYS_ID         : ${SYS_ID}"

TOTAL_FOUND=$(ls "${CONSENSUS_ROOT}"/task_*/per_utt/*.json 2>/dev/null | wc -l)
echo "found ${TOTAL_FOUND} consensus JSONs under task_*/per_utt/"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/prepare_segale_shards.py \
  --consensus-root "${CONSENSUS_ROOT}" \
  --out-root       "${OUT_ROOT}" \
  --num-docs       "${NUM_DOCS}" \
  --num-shards     "${NUM_SHARDS}" \
  --sys-id         "${SYS_ID}"

# Sanity: all 24 shards must have non-empty system.jsonl and ref.jsonl.
SHARDS_ROOT="${OUT_ROOT}/shards"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_ID=$(printf "%02d" "$i")
  S="${SHARDS_ROOT}/shard_${SHARD_ID}/system.jsonl"
  R="${SHARDS_ROOT}/shard_${SHARD_ID}/ref.jsonl"
  if [[ ! -s "$S" || ! -s "$R" ]]; then
    echo "ERROR: missing/empty shard ${SHARD_ID} (${S} or ${R})"
    exit 1
  fi
done
echo "===== SEGALE prepare DONE $(date) ====="
