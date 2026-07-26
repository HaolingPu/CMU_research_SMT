#!/usr/bin/env bash
# End-to-end SEGALE align + MetricX QE-MAX filter on preempt 24-GPU array.
# Specialized for the J 40k production output layout (task_NN/per_utt/*.json).
#
# Stages (all chained with --dependency afterok; everything goes into the
# slurm queue, nothing runs locally):
#   1. prepare shards            (single CPU job)
#   2. SEGALE align              (array 0-23, preempt, 1 GPU each)
#   3. merge_aligned_shards.py   (single CPU job)
#   4. QE prepare                (single GPU job; tokenize + 24-split)
#   5. MetricX QE predict        (array 0-23, preempt, 1 GPU each)
#   6. QE-MAX filter (t=3)       (single CPU job)
#
# Defaults:
#   CONSENSUS_ROOT = /.../consensus_decoding_prod/J_40k
#   OUT_ROOT       = /.../consensus_decoding_prod/J_40k-segale-p24
#   NUM_DOCS       = 40000
#   NUM_SHARDS     = 24
#   QE_THRESHOLD   = 3.0  (MAX semantics — drop if any sentence > threshold)
#
# Optional env var:
#   DEPEND_ON_JOBS=jobid[:jobid:...]   chain Stage 1 afterok on these (e.g. the
#                                       running J 40k array IDs). All stages
#                                       then chain after, so you can fire the
#                                       whole pipeline before generation finishes.
#
# Usage:
#   # fire now, waiting for J 40k array jobs to finish first:
#   DEPEND_ON_JOBS=7942988:7943007 bash submit_J40k_post.sh
#
#   # or fire after J 40k is already done:
#   bash submit_J40k_post.sh [CONSENSUS_ROOT] [OUT_ROOT] [NUM_DOCS] [QE_THRESHOLD]

set -e

CONSENSUS_ROOT="${1:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k}"
OUT_ROOT="${2:-${CONSENSUS_ROOT}-segale-p24}"
NUM_DOCS="${3:-40000}"
QE_THRESHOLD="${4:-3.0}"
SYS_ID="$(basename "${CONSENSUS_ROOT}")"
NUM_SHARDS=24

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================================================="
echo "J 40k post-processing pipeline (preempt, 24-array)"
echo "==================================================="
echo "CONSENSUS_ROOT  : ${CONSENSUS_ROOT}"
echo "OUT_ROOT        : ${OUT_ROOT}"
echo "NUM_DOCS        : ${NUM_DOCS}"
echo "NUM_SHARDS      : ${NUM_SHARDS}"
echo "QE_THRESHOLD    : ${QE_THRESHOLD}  (MAX filter; drop case if any sentence > thr)"
echo "SYS_ID          : ${SYS_ID}"
echo "DEPEND_ON_JOBS  : ${DEPEND_ON_JOBS:-(none — assumes J 40k already done)}"

mkdir -p "${OUT_ROOT}" \
         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs

SHARDS_ROOT="${OUT_ROOT}/shards"
ALIGNED_MERGED="${OUT_ROOT}/aligned_all.jsonl"
METRICX_RUN_DIR="${OUT_ROOT}/metricx-aligned"
QE_FILTERED_DIR="${OUT_ROOT}/qe$(printf '%.0f' "${QE_THRESHOLD}")-aligned-max"

echo
echo "[1/6] Submitting Stage 1 prepare ($(if [[ -n "${DEPEND_ON_JOBS}" ]]; then echo "afterok:${DEPEND_ON_JOBS}"; else echo "no dep"; fi))..."
PREP_DEP_FLAG=""
if [[ -n "${DEPEND_ON_JOBS}" ]]; then
  PREP_DEP_FLAG="--dependency=afterok:${DEPEND_ON_JOBS}"
fi
PREP_JOB=$(sbatch --parsable \
  ${PREP_DEP_FLAG} \
  --export="ALL,CONSENSUS_ROOT=${CONSENSUS_ROOT},OUT_ROOT=${OUT_ROOT},NUM_DOCS=${NUM_DOCS},SYS_ID=${SYS_ID}" \
  "${SCRIPT_DIR}/run_prepare_shards_24.sbatch")
echo "[prepare]          ${PREP_JOB}"

echo
echo "[2/6] Submitting 24-GPU SEGALE align array (preempt, afterok:${PREP_JOB})..."
ALIGN_JOB=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB}" \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT}" \
  "${SCRIPT_DIR}/run_segale_align_24gpu_preempt.sbatch")
echo "[align array]      ${ALIGN_JOB}"

echo
echo "[3/6] Submitting merge step (afterok:${ALIGN_JOB})..."
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ALIGN_JOB}" \
  --time=00:30:00 \
  --partition=cpu --qos=cpu_qos \
  --cpus-per-task=2 --mem=8G \
  --job-name=segale_merge_p24 \
  -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_merge_p24_%j.out \
  -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/segale_merge_p24_%j.err \
  --wrap "source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && PYTHONNOUSERSITE=1 python ${SCRIPT_DIR}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards ${NUM_SHARDS}")
echo "[merge]            ${MERGE_JOB}"

echo
echo "[4/6] Submitting QE prepare (afterok:${MERGE_JOB})..."
QE_PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${MERGE_JOB}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_MERGED},CONSENSUS_ROOT=${CONSENSUS_ROOT},METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SCRIPT_DIR}/run_qe_prepare_24.sbatch")
echo "[qe prepare]       ${QE_PREP_JOB}"

echo
echo "[5/6] Submitting 24-GPU MetricX QE predict (afterok:${QE_PREP_JOB})..."
QE_PREDICT_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREP_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SCRIPT_DIR}/run_metricx_qe_24gpu_preempt.sbatch")
echo "[qe predict 24x]   ${QE_PREDICT_JOB}"

echo
echo "[6/6] Submitting QE-MAX finalize (afterok:${QE_PREDICT_JOB})..."
QE_FIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREDICT_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},QE_FILTERED_DIR=${QE_FILTERED_DIR},QE_THRESHOLD=${QE_THRESHOLD}" \
  "${SCRIPT_DIR}/run_qe_finalize_24.sbatch")
echo "[qe finalize]      ${QE_FIN_JOB}"

cat <<EOF

===================================================
Pipeline submitted. Dependency chain:
  prepare           : done (synchronous)
  align(24x preempt): ${ALIGN_JOB}
  merge             : ${MERGE_JOB}      (afterok:${ALIGN_JOB})
  qe prepare        : ${QE_PREP_JOB}    (afterok:${MERGE_JOB})
  qe predict(24x)   : ${QE_PREDICT_JOB} (afterok:${QE_PREP_JOB})
  qe finalize MAX=3 : ${QE_FIN_JOB}     (afterok:${QE_PREDICT_JOB})

Watch:  squeue -u haolingp -o "%.10i %.4P %.14j %.8T %.10M %R"
Logs:   /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/

Final filtered output:
  ${QE_FILTERED_DIR}
EOF
