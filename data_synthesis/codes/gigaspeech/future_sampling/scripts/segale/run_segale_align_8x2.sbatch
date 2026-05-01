#!/usr/bin/env bash
#SBATCH --job-name=segale_align
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_align_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_align_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Each array task processes TWO shards in parallel on TWO GPUs:
#   array_id k -> shards (k) and (k + STRIDE), default STRIDE=8
# So 8 array tasks * 2 shards/task = 16 shards total, fitting MaxJobsPU=10 limit.
#
# Required env vars:
#   SHARDS_ROOT      -- contains shard_00, shard_01, ...
# Optional:
#   STRIDE           -- second shard offset (default 8)
#   TASK_LANG, PROC_DEVICE, EMBEDDING_MODEL, MAX_SIZE
#   NUM_SHARDS_TOTAL -- expected total shards (default 16, for sanity check)

set -e

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT is required}"
STRIDE="${STRIDE:-8}"
TASK_LANG="${TASK_LANG:-zh}"
PROC_DEVICE="${PROC_DEVICE:-cuda}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/LaBSE}"
MAX_SIZE="${MAX_SIZE:-8}"
NUM_SHARDS_TOTAL="${NUM_SHARDS_TOTAL:-16}"

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_A=$((ARRAY_ID))
SHARD_B=$((ARRAY_ID + STRIDE))

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

echo "===== SEGALE align array_id=${ARRAY_ID} START $(date) ====="
echo "node             : ${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE_DEVICES (slurm)     : ${CUDA_VISIBLE_DEVICES:-?}"
echo "SHARD_A=${SHARD_A}  SHARD_B=${SHARD_B}  (STRIDE=${STRIDE}, total=${NUM_SHARDS_TOTAL})"

run_one() {
  local shard_idx="$1"
  local local_gpu="$2"

  local shard_id
  shard_id=$(printf "%02d" "${shard_idx}")
  local shard_dir="${SHARDS_ROOT}/shard_${shard_id}"
  local sysf="${shard_dir}/system.jsonl"
  local reff="${shard_dir}/ref.jsonl"
  local aligned="${shard_dir}/system/aligned_spacy_system.jsonl"
  local logp="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_align_${SLURM_ARRAY_JOB_ID:-A}_${ARRAY_ID}_shard${shard_id}.log"

  if [[ ! -f "${sysf}" || ! -f "${reff}" ]]; then
    echo "[shard ${shard_id}] ERROR: missing inputs in ${shard_dir}" | tee -a "${logp}"
    return 2
  fi
  if [[ -s "${aligned}" ]]; then
    echo "[shard ${shard_id}] already aligned ($(wc -l < ${aligned}) rows), skip" | tee -a "${logp}"
    return 0
  fi

  echo "[shard ${shard_id}] start (local_gpu=${local_gpu}) -> ${logp}"
  CUDA_VISIBLE_DEVICES="${local_gpu}" \
  segale-align \
    --system_file     "${sysf}" \
    --ref_file        "${reff}" \
    --segmenter       spacy \
    --task_lang       "${TASK_LANG}" \
    --proc_device     "${PROC_DEVICE}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --max_size        "${MAX_SIZE}" \
    >>"${logp}" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[shard ${shard_id}] FAILED rc=$rc"
    return $rc
  fi
  if [[ ! -s "${aligned}" ]]; then
    echo "[shard ${shard_id}] FAILED: empty/missing ${aligned}"
    return 1
  fi
  echo "[shard ${shard_id}] DONE rows=$(wc -l < ${aligned})"
}

# Launch two shards in parallel, one per local GPU index.
run_one "${SHARD_A}" 0 &
PID_A=$!
run_one "${SHARD_B}" 1 &
PID_B=$!

wait $PID_A
RC_A=$?
wait $PID_B
RC_B=$?

if [[ $RC_A -ne 0 || $RC_B -ne 0 ]]; then
  echo "===== array_id=${ARRAY_ID} FAILED  shard_${SHARD_A} rc=$RC_A  shard_${SHARD_B} rc=$RC_B"
  exit 1
fi

echo "===== SEGALE align array_id=${ARRAY_ID} DONE $(date) ====="
