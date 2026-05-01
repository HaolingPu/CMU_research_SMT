#!/usr/bin/env bash
#SBATCH --job-name=segale_align
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_align_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_align_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Per-shard SEGALE alignment (no -v, no intermediate dumps).
# Required env vars:
#   SHARDS_ROOT      -- contains shard_00, shard_01, ...
# Optional:
#   TASK_LANG        -- default zh
#   PROC_DEVICE      -- default cuda
#   EMBEDDING_MODEL  -- default sentence-transformers/LaBSE
#   MAX_SIZE         -- default 8

set -e

SHARDS_ROOT="${SHARDS_ROOT:?SHARDS_ROOT is required}"
TASK_LANG="${TASK_LANG:-zh}"
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

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale

# Avoid pip user site / interfering paths; segale env should be self-contained.
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

echo "===== SEGALE align shard ${SHARD_ID} START $(date) ====="
echo "system_file     : ${SYSTEM_FILE}"
echo "ref_file        : ${REF_FILE}"
echo "embedding_model : ${EMBEDDING_MODEL}"
echo "task_lang       : ${TASK_LANG}"
echo "proc_device     : ${PROC_DEVICE}"
echo "max_size        : ${MAX_SIZE}"
echo "node            : ${SLURMD_NODENAME:-$(hostname)}  gpu=${CUDA_VISIBLE_DEVICES:-?}"

# IMPORTANT: no -v / -vv -- skip intermediate aps_results dumps.
segale-align \
  --system_file "${SYSTEM_FILE}" \
  --ref_file    "${REF_FILE}" \
  --segmenter   spacy \
  --task_lang   "${TASK_LANG}" \
  --proc_device "${PROC_DEVICE}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --max_size    "${MAX_SIZE}"

ALIGNED="${SHARD_DIR}/system/aligned_spacy_system.jsonl"
if [[ ! -s "${ALIGNED}" ]]; then
  echo "ERROR: expected aligned output missing or empty: ${ALIGNED}"
  ls -la "${SHARD_DIR}/system" || true
  exit 1
fi

echo "===== SEGALE align shard ${SHARD_ID} DONE $(date) ====="
echo "aligned rows: $(wc -l < ${ALIGNED})"
