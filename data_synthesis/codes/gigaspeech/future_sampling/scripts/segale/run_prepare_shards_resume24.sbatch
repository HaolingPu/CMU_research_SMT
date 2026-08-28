#!/usr/bin/env bash
#SBATCH --job-name=segale_prep_resume24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=01:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_prep_resume24_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_prep_resume24_%j.err

# Build 24 resume shards from MISSING_UTTS_FILE -> RESUME_SHARDS_ROOT.
# Required env vars:
#   CONSENSUS_ROOT       -- J 40k root
#   RESUME_OUT_ROOT      -- e.g. .../J_40k-segale-p24/resume
#   MISSING_UTTS_FILE    -- text file, one utt_id per line
#   SYS_ID               -- written into rows (e.g. J_40k)
# Optional:
#   NUM_SHARDS           (default 24)

set -e

CONSENSUS_ROOT="${CONSENSUS_ROOT:?CONSENSUS_ROOT required}"
RESUME_OUT_ROOT="${RESUME_OUT_ROOT:?RESUME_OUT_ROOT required}"
MISSING_UTTS_FILE="${MISSING_UTTS_FILE:?MISSING_UTTS_FILE required}"
SYS_ID="${SYS_ID:?SYS_ID required}"
NUM_SHARDS="${NUM_SHARDS:-24}"

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale
export PYTHONNOUSERSITE=1

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs

echo "===== SEGALE prepare (RESUME, 24-shard) START $(date) ====="
echo "CONSENSUS_ROOT     : ${CONSENSUS_ROOT}"
echo "RESUME_OUT_ROOT    : ${RESUME_OUT_ROOT}"
echo "MISSING_UTTS_FILE  : ${MISSING_UTTS_FILE}"
echo "MISSING utts count : $(wc -l < ${MISSING_UTTS_FILE})"
echo "SYS_ID             : ${SYS_ID}"
echo "NUM_SHARDS         : ${NUM_SHARDS}"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/prepare_segale_shards.py \
  --consensus-root "${CONSENSUS_ROOT}" \
  --out-root       "${RESUME_OUT_ROOT}" \
  --num-docs       100000 \
  --num-shards     "${NUM_SHARDS}" \
  --sys-id         "${SYS_ID}" \
  --include-utts   "${MISSING_UTTS_FILE}"

echo "===== Sanity: per-shard row counts ====="
for i in $(seq -w 0 $((NUM_SHARDS - 1))); do
  SDIR="${RESUME_OUT_ROOT}/shards/shard_${i}"
  SYS="${SDIR}/system.jsonl"
  REF="${SDIR}/ref.jsonl"
  ns=$(wc -l < "${SYS}" 2>/dev/null || echo 0)
  nr=$(wc -l < "${REF}" 2>/dev/null || echo 0)
  echo "shard_${i}: system=${ns}  ref=${nr}"
  if [[ "${ns}" -eq 0 ]]; then
    echo "ERROR: shard ${i} empty"
    exit 1
  fi
done

echo "===== SEGALE prepare (RESUME) DONE $(date) ====="
