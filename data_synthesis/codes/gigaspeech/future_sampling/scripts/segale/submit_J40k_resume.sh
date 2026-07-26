#!/usr/bin/env bash
# Resume J 40k SEGALE→QE pipeline for the 16203 utts that the first run missed
# (preempted shards were marked "done" by the old -s skip guard, causing partial
# alignment to be treated as complete). Strategy:
#
#   Stage 0: missing list is precomputed at $RESUME_ROOT/missing_utts.txt
#   Stage 1: prepare 24 resume shards under $RESUME_ROOT/shards/
#   Stage 2: SEGALE align array on those shards (24x preempt, sentinel-protected)
#   Stage 3: merge resume aligned + existing aligned_all.jsonl -> aligned_full.jsonl
#   Stage 4: qe_prepare (patched converter understands task_*/per_utt/*.json)
#   Stage 5: metricx QE predict (24x preempt)
#   Stage 6: finalize MAX filter (t=3) -> qe3-aligned-max/

set -e

OUT_ROOT="${OUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k-segale-p24}"
CONSENSUS_ROOT="${CONSENSUS_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k}"
RESUME_ROOT="${OUT_ROOT}/resume"
RESUME_SHARDS_ROOT="${RESUME_ROOT}/shards"
MISSING_UTTS_FILE="${RESUME_ROOT}/missing_utts.txt"
SYS_ID="$(basename "${CONSENSUS_ROOT}")"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS=24

if [[ ! -f "${MISSING_UTTS_FILE}" ]]; then
  echo "ERROR: missing_utts.txt not found at ${MISSING_UTTS_FILE}"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs"
mkdir -p "${LOGS}"

ALIGNED_OLD="${OUT_ROOT}/aligned_all.jsonl"
ALIGNED_RESUME="${RESUME_ROOT}/aligned_resume_all.jsonl"
ALIGNED_FULL="${OUT_ROOT}/aligned_full.jsonl"
METRICX_RUN_DIR="${OUT_ROOT}/metricx-aligned-full"
QE_FILTERED_DIR="${OUT_ROOT}/qe$(printf '%.0f' "${QE_THRESHOLD}")-aligned-max-full"

echo "==================================================="
echo "J 40k RESUME pipeline (16203 missing utts → 24 shards)"
echo "==================================================="
echo "OUT_ROOT          : ${OUT_ROOT}"
echo "RESUME_ROOT       : ${RESUME_ROOT}"
echo "MISSING utts      : $(wc -l < ${MISSING_UTTS_FILE})"
echo "ALIGNED_OLD       : ${ALIGNED_OLD} ($(wc -l < ${ALIGNED_OLD}) rows)"
echo "ALIGNED_FULL      : ${ALIGNED_FULL}"
echo "METRICX_RUN_DIR   : ${METRICX_RUN_DIR}"
echo "QE_FILTERED_DIR   : ${QE_FILTERED_DIR}"
echo "QE_THRESHOLD      : ${QE_THRESHOLD}  (MAX semantics)"

echo
echo "[1/6] Submitting RESUME prepare..."
PREP_JOB=$(sbatch --parsable \
  --export="ALL,CONSENSUS_ROOT=${CONSENSUS_ROOT},RESUME_OUT_ROOT=${RESUME_ROOT},MISSING_UTTS_FILE=${MISSING_UTTS_FILE},SYS_ID=${SYS_ID},NUM_SHARDS=${NUM_SHARDS}" \
  "${SCRIPT_DIR}/run_prepare_shards_resume24.sbatch")
echo "[prep]            ${PREP_JOB}"

echo
echo "[2/6] Submitting 24x SEGALE align array on RESUME shards (preempt)..."
ALIGN_JOB=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB}" \
  --export="ALL,SHARDS_ROOT=${RESUME_SHARDS_ROOT}" \
  "${SCRIPT_DIR}/run_segale_align_24gpu_preempt.sbatch")
echo "[align resume]    ${ALIGN_JOB}"

echo
echo "[3/6] Submitting merge (resume + existing aligned_all -> aligned_full.jsonl)..."
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ALIGN_JOB}" \
  --time=00:30:00 \
  --partition=cpu --qos=cpu_qos \
  --cpus-per-task=2 --mem=8G \
  --job-name=segale_merge_resume \
  -o "${LOGS}/segale_merge_resume_%j.out" \
  -e "${LOGS}/segale_merge_resume_%j.err" \
  --wrap "source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && PYTHONNOUSERSITE=1 python ${SCRIPT_DIR}/merge_aligned_shards.py --shards-root ${RESUME_SHARDS_ROOT} --output ${ALIGNED_RESUME} --num-shards ${NUM_SHARDS} && cat ${ALIGNED_OLD} ${ALIGNED_RESUME} > ${ALIGNED_FULL} && echo 'aligned_full.jsonl total:' \$(wc -l < ${ALIGNED_FULL})")
echo "[merge]           ${MERGE_JOB}"

echo
echo "[4/6] Submitting QE prepare (uses ALIGNED_FULL)..."
QE_PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${MERGE_JOB}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_FULL},CONSENSUS_ROOT=${CONSENSUS_ROOT},METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SCRIPT_DIR}/run_qe_prepare_24.sbatch")
echo "[qe prepare]      ${QE_PREP_JOB}"

echo
echo "[5/6] Submitting 24x MetricX QE predict (preempt)..."
QE_PREDICT_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREP_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SCRIPT_DIR}/run_metricx_qe_24gpu_preempt.sbatch")
echo "[qe predict 24x]  ${QE_PREDICT_JOB}"

echo
echo "[6/6] Submitting QE-MAX finalize (t=${QE_THRESHOLD})..."
QE_FIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREDICT_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},QE_FILTERED_DIR=${QE_FILTERED_DIR},QE_THRESHOLD=${QE_THRESHOLD}" \
  "${SCRIPT_DIR}/run_qe_finalize_24.sbatch")
echo "[qe finalize]     ${QE_FIN_JOB}"

cat <<EOF

===================================================
RESUME pipeline submitted:
  prep              : ${PREP_JOB}
  align(24x resume) : ${ALIGN_JOB}     (afterok:${PREP_JOB})
  merge full        : ${MERGE_JOB}     (afterok:${ALIGN_JOB})
  qe prepare        : ${QE_PREP_JOB}   (afterok:${MERGE_JOB})
  qe predict(24x)   : ${QE_PREDICT_JOB}(afterok:${QE_PREP_JOB})
  qe finalize MAX=${QE_THRESHOLD}: ${QE_FIN_JOB}    (afterok:${QE_PREDICT_JOB})

Final filtered output:
  ${QE_FILTERED_DIR}

Watch: squeue -u haolingp -o "%.10i %.4P %.20j %.8T %.10M %R"
EOF
