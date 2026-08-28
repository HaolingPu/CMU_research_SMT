#!/usr/bin/env bash
# End-to-end SEGALE alignment + per-sentence MetricX QE filter (no length-ratio).
# Use this when the target IS the reference you generated yourself (length
# ratio is meaningless / always ~1).
#
# Usage:
#   bash submit_segale_no_lr.sh <CONSENSUS_ROOT> <OUT_ROOT> [NUM_DOCS] [SYS_ID]
#     CONSENSUS_ROOT  Dir containing job_*/task_*/<utt>.json (or task_*/<utt>.json)
#     OUT_ROOT        Dir for SEGALE outputs (will be created)
#     NUM_DOCS        How many docs to align (default 50000)
#     SYS_ID          Logical system identifier; default = basename(CONSENSUS_ROOT)
#
# Optional env:
#   QE_THRESHOLD    (default 3.0)
#
# Final filtered cases land in: ${OUT_ROOT}/qe3-aligned/

set -e

CONSENSUS_ROOT="${1:?Usage: $0 <CONSENSUS_ROOT> <OUT_ROOT> [NUM_DOCS] [SYS_ID]}"
OUT_ROOT="${2:?Usage: $0 <CONSENSUS_ROOT> <OUT_ROOT> [NUM_DOCS] [SYS_ID]}"
NUM_DOCS="${3:-50000}"
SYS_ID="${4:-$(basename "${CONSENSUS_ROOT}")}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPK_DIR=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk

echo "CONSENSUS_ROOT : ${CONSENSUS_ROOT}"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "NUM_DOCS       : ${NUM_DOCS}"
echo "SYS_ID         : ${SYS_ID}"
echo "QE_THRESHOLD   : ${QE_THRESHOLD}"
echo "(no length-ratio filter)"

mkdir -p "${OUT_ROOT}"

echo
echo "[1/5] Preparing SEGALE shards (${NUM_DOCS} docs / ${NUM_SHARDS} shards)..."
python "${SCRIPT_DIR}/prepare_segale_shards.py" \
  --consensus-root "${CONSENSUS_ROOT}" \
  --out-root       "${OUT_ROOT}" \
  --num-docs       "${NUM_DOCS}" \
  --num-shards     "${NUM_SHARDS}" \
  --sys-id         "${SYS_ID}"

SHARDS_ROOT="${OUT_ROOT}/shards"
ALIGNED_MERGED="${OUT_ROOT}/aligned_all.jsonl"
METRICX_RUN_DIR="${OUT_ROOT}/metricx-aligned"
FINAL_OUT_DIR="${OUT_ROOT}/qe3-aligned"
SLURM_LOGS=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs

mkdir -p "${SLURM_LOGS}"

echo
echo "[2/5] Submitting 8-GPU segale-align array (1 GPU per shard)..."
ALIGN_JOB=$(sbatch --parsable \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT}" \
  "${SCRIPT_DIR}/run_segale_align_8gpu.sbatch")
echo "[align array]    ${ALIGN_JOB}"

echo
echo "[3/5] Submitting merge step (afterok:${ALIGN_JOB})..."
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ALIGN_JOB}" \
  --time=00:20:00 \
  --partition=general --qos=normal \
  --gres=gpu:L40S:1 \
  --cpus-per-task=2 --mem=8G \
  --job-name=segale_merge \
  -o "${SLURM_LOGS}/segale_merge_%j.out" \
  -e "${SLURM_LOGS}/segale_merge_%j.err" \
  --wrap "source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && python ${SCRIPT_DIR}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards ${NUM_SHARDS}")
echo "[merge]          ${MERGE_JOB}"

echo
echo "[4/5] Submitting QE prepare (afterok:${MERGE_JOB})..."
QE_PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${MERGE_JOB}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_MERGED},CONSENSUS_ROOT=${CONSENSUS_ROOT},METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SCRIPT_DIR}/run_qe_prepare_aligned.sbatch")
echo "[qe prepare]     ${QE_PREP_JOB}"

echo
echo "[5/5] Submitting 8-GPU MetricX predict + QE-only finalize..."
QE_PREDICT_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREP_JOB}" \
  --exclude=babel-t9-16 \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${TOPK_DIR}/run_metricx_qe_8gpu.sbatch")
echo "[qe predict 8gpu] ${QE_PREDICT_JOB}"

QE_FIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREDICT_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},FINAL_OUT_DIR=${FINAL_OUT_DIR},QE_THRESHOLD=${QE_THRESHOLD}" \
  "${SCRIPT_DIR}/run_qe_finalize_aligned_no_lr.sbatch")
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
echo "  After QE filter    : ${FINAL_OUT_DIR}   <-- final training data"
echo "  Manifest           : ${OUT_ROOT}/shard_manifest.json"
