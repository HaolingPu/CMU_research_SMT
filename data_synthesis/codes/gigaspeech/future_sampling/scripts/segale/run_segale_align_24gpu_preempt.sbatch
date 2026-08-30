#!/usr/bin/env bash
#SBATCH --job-name=segale_align_p24
#SBATCH --array=0-23%24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=16G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-t9-16,babel-t5-32
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_align_p24_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_align_p24_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Per-shard SEGALE alignment — 24-way preempt array variant.
# Required env vars:
#   SHARDS_ROOT      -- contains shard_00 .. shard_23
# Optional:
#   TASK_LANG        (default zh)
#   PROC_DEVICE      (default cuda)
#   EMBEDDING_MODEL  (default sentence-transformers/LaBSE)
#   MAX_SIZE         (default 8)

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
ALIGNED="${SHARD_DIR}/system/aligned_spacy_system.jsonl"
DONE_FLAG="${SHARD_DIR}/system/ALIGN_DONE.txt"

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs

# Sentinel-based skip: only skip if the prior run completed cleanly. A partial
# (preempted) output stays on disk; segale-align itself opens the aligned file
# in append mode and calls load_done_doc_ids() at startup to skip any doc_ids
# already written, so a preempted shard resumes from where it left off.
if [[ -f "${DONE_FLAG}" ]]; then
  echo "[SKIP] shard ${SHARD_ID} ALIGN_DONE present ($(wc -l < ${ALIGNED} 2>/dev/null) rows)"
  exit 0
fi
if [[ -f "${ALIGNED}" ]]; then
  echo "[RESUME] partial aligned found ($(wc -l < ${ALIGNED}) rows); segale-align will skip done doc_ids and continue"
fi

if [[ ! -f "${SYSTEM_FILE}" || ! -f "${REF_FILE}" ]]; then
  echo "ERROR: missing inputs in ${SHARD_DIR}"
  ls -la "${SHARD_DIR}" || true
  exit 1
fi

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

echo "===== SEGALE align shard ${SHARD_ID} (preempt 24-array) START $(date) ====="
echo "system_file     : ${SYSTEM_FILE}"
echo "ref_file        : ${REF_FILE}"
echo "embedding_model : ${EMBEDDING_MODEL}"
echo "task_lang       : ${TASK_LANG}"
echo "node            : ${SLURMD_NODENAME:-$(hostname)}  gpu=${CUDA_VISIBLE_DEVICES:-?}"

requeue_job() {
  echo "[REQUEUE] $1; requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}" || true
  exit 0
}

# GPU free-memory precheck. preempt occasionally packs us onto a physical GPU
# another tenant is already using (~40 GiB resident); loading LaBSE then OOMs
# in seconds with exit 1, and SLURM --requeue does NOT auto-requeue user-code
# exits. So we explicitly requeue here and exit 0 to bounce to another node.
MIN_FREE_MB="${MIN_FREE_MB:-8192}"
GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
              | awk 'NR==1{gsub(/[^0-9]/, "", $0); print}')
echo "gpu free        : ${GPU_FREE_MB:-?} MB  (require >= ${MIN_FREE_MB} MB)"
if [[ -z "${GPU_FREE_MB}" || "${GPU_FREE_MB}" -lt "${MIN_FREE_MB}" ]]; then
  requeue_job "insufficient GPU memory on ${SLURMD_NODENAME:-$(hostname)} (free=${GPU_FREE_MB:-?} MB)"
fi

GPU_ECC_ERRORS=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{gsub(/[^0-9]/, "", $0); print}')
echo "uncorrected ECC : ${GPU_ECC_ERRORS:-unknown}"
if [[ "${GPU_ECC_ERRORS:-}" =~ ^[0-9]+$ ]] && (( GPU_ECC_ERRORS > 0 )); then
  requeue_job "GPU has ${GPU_ECC_ERRORS} volatile uncorrectable ECC errors on ${SLURMD_NODENAME:-$(hostname)}"
fi

set +e
segale-align \
  --system_file "${SYSTEM_FILE}" \
  --ref_file    "${REF_FILE}" \
  --segmenter   spacy \
  --task_lang   "${TASK_LANG}" \
  --proc_device "${PROC_DEVICE}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --max_size    "${MAX_SIZE}"
ALIGN_RC=$?
set -e

if (( ALIGN_RC != 0 )); then
  GPU_ECC_ERRORS=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{gsub(/[^0-9]/, "", $0); print}')
  if [[ "${GPU_ECC_ERRORS:-}" =~ ^[0-9]+$ ]] && (( GPU_ECC_ERRORS > 0 )); then
    requeue_job "alignment exited ${ALIGN_RC} after GPU reported ${GPU_ECC_ERRORS} uncorrectable ECC errors"
  fi
  echo "ERROR: segale-align exited ${ALIGN_RC}" >&2
  exit "${ALIGN_RC}"
fi

if [[ ! -s "${ALIGNED}" ]]; then
  echo "ERROR: expected aligned output missing or empty: ${ALIGNED}"
  ls -la "${SHARD_DIR}/system" || true
  exit 1
fi

# Mark this shard as cleanly finished. Resume runs key off this sentinel.
touch "${DONE_FLAG}"

echo "===== SEGALE align shard ${SHARD_ID} DONE $(date) ====="
echo "aligned rows: $(wc -l < ${ALIGNED})"
