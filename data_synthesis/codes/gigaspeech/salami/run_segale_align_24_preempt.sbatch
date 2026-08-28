#!/usr/bin/env bash
#SBATCH --job-name=salami_seg_align
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-t9-16,babel-p5-28
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/segale_align_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/segale_align_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT is required}"
TASK_LANG="${TASK_LANG:-ja}"
PROC_DEVICE="${PROC_DEVICE:-cuda}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/LaBSE}"
MAX_SIZE="${MAX_SIZE:-8}"

SHARD_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-0}")
SHARD_DIR="${SHARDS_ROOT}/shard_${SHARD_ID}"
SYSTEM_FILE="${SHARD_DIR}/system.jsonl"
REF_FILE="${SHARD_DIR}/ref.jsonl"

if [[ ! -f "${SYSTEM_FILE}" || ! -f "${REF_FILE}" ]]; then
  echo "ERROR: missing inputs in ${SHARD_DIR}"
  ls -la "${SHARD_DIR}" || true
  exit 1
fi

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

echo "===== SALAMI SEGALE align shard ${SHARD_ID} START $(date) ====="
echo "node=${SLURMD_NODENAME:-$(hostname)} gpu=${CUDA_VISIBLE_DEVICES:-?}"

# Idempotent skip if already aligned (preempt requeue safe)
ALIGNED="${SHARD_DIR}/system/aligned_spacy_system.jsonl"
if [[ -s "${ALIGNED}" ]]; then
  WANT=$(wc -l < "${SYSTEM_FILE}")
  HAVE=$(wc -l < "${ALIGNED}")
  if (( HAVE >= WANT )); then
    echo "Shard already aligned ($HAVE >= $WANT lines), skipping."
    exit 0
  fi
fi

segale-align \
  --system_file "${SYSTEM_FILE}" \
  --ref_file    "${REF_FILE}" \
  --segmenter   spacy \
  --task_lang   "${TASK_LANG}" \
  --proc_device "${PROC_DEVICE}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --max_size    "${MAX_SIZE}"

if [[ ! -s "${ALIGNED}" ]]; then
  echo "ERROR: expected aligned output missing or empty: ${ALIGNED}"
  exit 1
fi

echo "===== SALAMI SEGALE align shard ${SHARD_ID} DONE $(date) ====="
echo "aligned rows: $(wc -l < ${ALIGNED})"
