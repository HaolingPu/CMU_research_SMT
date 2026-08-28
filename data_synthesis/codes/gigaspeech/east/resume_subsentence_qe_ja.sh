#!/usr/bin/env bash
# ============================================================
# RESUME: re-submit ja sub-sentence QE chain from segale-align onward.
# Skips the inline build_consensus_format + prepare_segale_shards steps
# (their outputs already exist under .../segale_qe/{consensus_format,shards}).
#
# segale_align.py has been patched to write incrementally + skip-existing,
# so this can also be safely re-run if a single shard times out again.
# ============================================================
set -e

LANG_TGT=ja
QE_THRESHOLD="${QE_THRESHOLD:-5.0}"
NUM_SHARDS=24
EXCLUDE_NODES="babel-p9-28,babel-s5-32"

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_${LANG_TGT}
WORK=${BASE}/segale_qe
CONS_FMT=${WORK}/consensus_format
SHARDS_ROOT=${WORK}/shards
ALIGNED_MERGED=${WORK}/aligned_all.jsonl
METRICX_RUN_DIR=${WORK}/metricx
FINAL_OUT=${WORK}/final_jsonl_east
LOGDIR=${WORK}/slurm_logs

CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech
SCRIPT_DIR=${CODE}/east
SEGALE_DIR=${CODE}/future_sampling/scripts/segale
TOPK_DIR=${CODE}/future_sampling/scripts/topk

mkdir -p "${LOGDIR}"

# Sanity-check that the prerequisites exist.
if [[ ! -d "${CONS_FMT}" || ! -d "${SHARDS_ROOT}" ]]; then
  echo "ERROR: missing ${CONS_FMT} or ${SHARDS_ROOT} -- run the full orchestrator first."
  exit 1
fi
for sid in $(seq -f "%02g" 0 $((NUM_SHARDS-1))); do
  d="${SHARDS_ROOT}/shard_${sid}"
  if [[ ! -f "${d}/system.jsonl" || ! -f "${d}/ref.jsonl" ]]; then
    echo "ERROR: shard ${d} missing system.jsonl or ref.jsonl"
    exit 1
  fi
done
echo "Prerequisites OK; resuming from segale-align (${NUM_SHARDS} shards on preempt)."

# --- Step 3: segale-align 24-shard array on preempt -----------------------
# preempt_qos caps at 24 GPUs, so %24 keeps the array fully concurrent.
# --requeue: preemption is harmless because segale_align.py was patched
# to write incrementally + skip-existing on resume.
# Lean resource request (measured peak: ~1.5G RSS, <1 active CPU) so the
# preempt scheduler can place us into smaller free node fragments fast.
echo "[3/7] segale-align ${NUM_SHARDS}-shard array (preempt, lean: 12G/4cpu, --requeue, 2d)"
ALIGN_JID=$(sbatch --parsable \
  --partition=preempt \
  --qos=preempt_qos \
  --array=0-$((NUM_SHARDS-1))%${NUM_SHARDS} \
  --time=2-00:00:00 \
  --requeue \
  --cpus-per-task=4 \
  --mem=12G \
  --gres=gpu:L40S:1 \
  --exclude="${EXCLUDE_NODES}" \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT},TASK_LANG=${LANG_TGT}" \
  -o "${LOGDIR}/segale_align_%A_%a.out" \
  -e "${LOGDIR}/segale_align_%A_%a.err" \
  "${SEGALE_DIR}/run_segale_align_8gpu.sbatch")
echo "  align array: ${ALIGN_JID}  (array=0-$((NUM_SHARDS-1)))"

# --- Step 4: merge aligned shards -----------------------------------------
echo "[4/7] merge aligned shards"
MERGE_JID=$(sbatch --parsable \
  --dependency=afterok:${ALIGN_JID} \
  --time=00:20:00 \
  --partition=general --qos=normal \
  --gres=gpu:1 \
  --exclude="${EXCLUDE_NODES}" \
  --cpus-per-task=2 --mem=8G \
  --job-name=east_${LANG_TGT}_segmerge \
  -o "${LOGDIR}/segale_merge_%j.out" \
  -e "${LOGDIR}/segale_merge_%j.err" \
  --wrap "source ~/.bashrc && conda activate /data/user_data/haolingp/conda_envs/segale && python ${SEGALE_DIR}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards ${NUM_SHARDS}")
echo "  merge: ${MERGE_JID}"

# --- Step 5: QE prepare ---------------------------------------------------
echo "[5/7] QE prepare"
QE_PREP_JID=$(sbatch --parsable \
  --dependency=afterok:${MERGE_JID} \
  --gres=gpu:1 \
  --exclude="${EXCLUDE_NODES}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_MERGED},CONSENSUS_ROOT=${CONS_FMT},METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  -o "${LOGDIR}/qe_prep_%j.out" \
  -e "${LOGDIR}/qe_prep_%j.err" \
  "${SEGALE_DIR}/run_qe_prepare_aligned.sbatch")
echo "  qe prepare: ${QE_PREP_JID}"

# --- Step 6: MetricX QE 8-shard array -------------------------------------
echo "[6/7] MetricX QE 8-shard array"
QE_PRED_JID=$(sbatch --parsable \
  --dependency=afterok:${QE_PREP_JID} \
  --gres=gpu:1 \
  --exclude="${EXCLUDE_NODES}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  -o "${LOGDIR}/qe_predict_%A_%a.out" \
  -e "${LOGDIR}/qe_predict_%A_%a.err" \
  "${TOPK_DIR}/run_metricx_qe_8gpu.sbatch")
echo "  qe predict 8x: ${QE_PRED_JID}"

# --- Step 7: custom finalize ---------------------------------------------
echo "[7/7] custom finalize (utt-level filter @ thr=${QE_THRESHOLD})"
FIN_JID=$(sbatch --parsable \
  --dependency=afterok:${QE_PRED_JID} \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},CONS_FMT_ROOT=${CONS_FMT},STREAM_DIR=${BASE}/streaming_EAST_dataset,FINAL_OUT_DIR=${FINAL_OUT},QE_THRESHOLD=${QE_THRESHOLD}" \
  -o "${LOGDIR}/finalize_%j.out" \
  -e "${LOGDIR}/finalize_%j.err" \
  "${SCRIPT_DIR}/run_subsentence_finalize_ja.sbatch")
echo "  finalize: ${FIN_JID}"

cat <<EOF

============================================================
ja sub-QE chain RESUMED (segale-align onward, preempt + 24 shards).
  align(${NUM_SHARDS}x)    : ${ALIGN_JID}   (preempt, 2d, --requeue, incremental + resumable)
  merge         : ${MERGE_JID}    (afterok:${ALIGN_JID})
  qe prepare    : ${QE_PREP_JID}  (afterok:${MERGE_JID})
  qe predict(8x): ${QE_PRED_JID}  (afterok:${QE_PREP_JID})
  finalize      : ${FIN_JID}      (afterok:${QE_PRED_JID})
  thr           : ${QE_THRESHOLD}

Watch:
  squeue -u haolingp
  for s in \$(seq 0 $((NUM_SHARDS-1))); do
    f=${LOGDIR}/segale_align_${ALIGN_JID}_\${s}.out
    [ -f "\${f}" ] && echo "shard \${s}: \$(grep -c '^doc_id:' \"\${f}\") docs done"
  done
============================================================
EOF
