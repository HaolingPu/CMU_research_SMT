#!/usr/bin/env bash
# End-to-end SEGALE alignment for a consensus-decoding run.
# Sampling-policy agnostic: works for top-k, top-p, min-p, etc.
# Usage:
#   bash submit_segale.sh <CONSENSUS_ROOT> <OUT_ROOT> [NUM_DOCS] [SYS_ID]
#     CONSENSUS_ROOT  Dir containing job_*/task_*/<utt>.json
#     OUT_ROOT        Dir for SEGALE outputs (will be created)
#     NUM_DOCS        How many docs to align (default 50000)
#     SYS_ID          Logical system identifier; default = basename(CONSENSUS_ROOT)
#
# Examples:
#   # 50k subset of top_5_k4 generation
#   bash submit_segale.sh \
#     /.../consensus_decoding_en_zh_top_5_k4/job_7521064 \
#     /.../consensus_decoding_en_zh_top_5_k4/job_7521064-segale50k
#
#   # whole min-p run, custom sys_id
#   bash submit_segale.sh \
#     /.../consensus_decoding_en_zh_minp_1em3/job_XXXX \
#     /.../consensus_decoding_en_zh_minp_1em3/job_XXXX-segale 100000 minp_1em3
#
#   # whole top-p run
#   bash submit_segale.sh \
#     /.../consensus_decoding_en_zh_topp_0p9/job_XXXX \
#     /.../consensus_decoding_en_zh_topp_0p9/job_XXXX-segale 80000 topp_0p9

set -e

CONSENSUS_ROOT="${1:?Usage: $0 <CONSENSUS_ROOT> <OUT_ROOT> [NUM_DOCS] [SYS_ID]}"
OUT_ROOT="${2:?Usage: $0 <CONSENSUS_ROOT> <OUT_ROOT> [NUM_DOCS] [SYS_ID]}"
NUM_DOCS="${3:-50000}"
SYS_ID="${4:-$(basename "${CONSENSUS_ROOT}")}"
# 8 shards * 1 GPU each. Cluster cap: general partition = 8 GPUs concurrent + 8 array jobs concurrent.
NUM_SHARDS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "CONSENSUS_ROOT : ${CONSENSUS_ROOT}"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "NUM_DOCS       : ${NUM_DOCS}"
echo "SYS_ID         : ${SYS_ID}"

mkdir -p "${OUT_ROOT}"

echo
echo "[1/6] Preparing SEGALE shards (${NUM_DOCS} docs / ${NUM_SHARDS} shards)..."
python "${SCRIPT_DIR}/prepare_segale_shards.py" \
  --consensus-root "${CONSENSUS_ROOT}" \
  --out-root       "${OUT_ROOT}" \
  --num-docs       "${NUM_DOCS}" \
  --num-shards     "${NUM_SHARDS}" \
  --sys-id         "${SYS_ID}"

SHARDS_ROOT="${OUT_ROOT}/shards"
ALIGNED_MERGED="${OUT_ROOT}/aligned_all.jsonl"
METRICX_RUN_DIR="${OUT_ROOT}/metricx-aligned"
QE_FILTERED_DIR="${OUT_ROOT}/qe3-aligned"
FINAL_OUT_DIR="${OUT_ROOT}/qe3-lr-aligned"

echo
echo "[2/6] Submitting 8-GPU segale-align array (1 GPU per shard)..."
ALIGN_JOB=$(sbatch --parsable \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT}" \
  "${SCRIPT_DIR}/run_segale_align_8gpu.sbatch")
echo "[align array]    ${ALIGN_JOB}"

echo
echo "[3/6] Submitting merge step (afterok:${ALIGN_JOB})..."
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ALIGN_JOB}" \
  --time=00:20:00 \
  --partition=general --qos=normal \
  --gres=gpu:L40S:1 \
  --cpus-per-task=2 --mem=8G \
  --job-name=segale_merge \
  -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_merge_%j.out \
  -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segale_merge_%j.err \
  --wrap "source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && python ${SCRIPT_DIR}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards ${NUM_SHARDS}")
echo "[merge]          ${MERGE_JOB}"

echo
echo "[4/6] Submitting QE prepare (afterok:${MERGE_JOB})..."
QE_PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${MERGE_JOB}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_MERGED},CONSENSUS_ROOT=${CONSENSUS_ROOT},METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SCRIPT_DIR}/run_qe_prepare_aligned.sbatch")
echo "[qe prepare]     ${QE_PREP_JOB}"

echo
echo "[5/6] Submitting 8-GPU MetricX predict (afterok:${QE_PREP_JOB})..."
TOPK_DIR=/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk
QE_PREDICT_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREP_JOB}" \
  --exclude=babel-t9-16 \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${TOPK_DIR}/run_metricx_qe_8gpu.sbatch")
echo "[qe predict 8gpu] ${QE_PREDICT_JOB}"

echo
echo "[6/6] Submitting QE finalize + length-ratio filter (afterok:${QE_PREDICT_JOB})..."
QE_FIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREDICT_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},QE_FILTERED_DIR=${QE_FILTERED_DIR},FINAL_OUT_DIR=${FINAL_OUT_DIR}" \
  "${SCRIPT_DIR}/run_qe_finalize_aligned.sbatch")
echo "[qe finalize]    ${QE_FIN_JOB}"

echo
echo "Pipeline submitted:"
echo "  prepare       : done"
echo "  align(8x)     : ${ALIGN_JOB}"
echo "  merge         : ${MERGE_JOB}        (afterok:${ALIGN_JOB})"
echo "  qe prepare    : ${QE_PREP_JOB}      (afterok:${MERGE_JOB})"
echo "  qe predict(8x): ${QE_PREDICT_JOB}   (afterok:${QE_PREP_JOB})"
echo "  qe finalize   : ${QE_FIN_JOB}       (afterok:${QE_PREDICT_JOB})"
echo
echo "Outputs (after completion):"
echo "  Per-shard align    : ${SHARDS_ROOT}/shard_NN/system/aligned_spacy_system.jsonl"
echo "  Aligned merged     : ${ALIGNED_MERGED}"
echo "  MetricX run dir    : ${METRICX_RUN_DIR}"
echo "  After QE filter    : ${QE_FILTERED_DIR}"
echo "  After QE+LR filter : ${FINAL_OUT_DIR}   <-- final training data"
echo "  Manifest           : ${OUT_ROOT}/shard_manifest.json"
