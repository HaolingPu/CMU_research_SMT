#!/usr/bin/env bash
# ============================================================
# EAST de  -- sub-sentence QE pipeline orchestrator (PREEMPT, 24-shard align).
#
# 1) repackage llm_output_merged into fake-consensus JSONs (per (utt,latency))
# 2) prepare segale shards (system.jsonl + ref.jsonl per shard) -- 24 shards
# 3) submit segale-align 24-shard array (preempt, 1 GPU each)
# 4) chain merge of aligned shards (preempt)
# 5) chain QE prepare (convert -> metricx_input.jsonl + 8 shards) (preempt)
# 6) chain MetricX QE 8-shard array (preempt)
# 7) chain custom finalize (utt-level filter + final_output) (preempt)
# ============================================================
set -e

LANG_TGT=de
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
ALIGN_SHARDS=24                  # segale align shards
QE_SHARDS=8                      # metricx qe shards (hardcoded in qe_prepare/qe_predict)
EXCLUDE_NODES="babel-p9-28,babel-s5-32,babel-m5-32,babel-n9-32,babel-o5-16,babel-o5-24"

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_${LANG_TGT}
LLM_MERGED=${BASE}/llm_output_merged
STREAM_DIR=${BASE}/streaming_EAST_dataset

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

mkdir -p "${WORK}" "${CONS_FMT}" "${LOGDIR}"

# --- Step 1: repackage llm_output_merged -> consensus-format JSONs --------
echo "[1/7] repackage llm_output_merged -> consensus-format"
source ~/.bashrc
conda activate SMT
python "${SCRIPT_DIR}/build_consensus_format_east.py" \
  --llm-merged-dir "${LLM_MERGED}" \
  --stream-dir     "${STREAM_DIR}" \
  --out-root       "${CONS_FMT}" \
  --target-lang    "${LANG_TGT}"
conda deactivate

CONS_NUM=$(find "${CONS_FMT}" -name '*.json' | wc -l)
echo "  consensus-format JSONs: ${CONS_NUM}"

# --- Step 2: prepare segale shards (24 shards) ----------------------------
echo "[2/7] prepare segale shards (NUM_DOCS=${CONS_NUM}, ${ALIGN_SHARDS} shards)"
conda activate /data/user_data/haolingp/conda_envs/segale
python "${SEGALE_DIR}/prepare_segale_shards.py" \
  --consensus-root "${CONS_FMT}" \
  --out-root       "${WORK}" \
  --num-docs       "${CONS_NUM}" \
  --num-shards     "${ALIGN_SHARDS}" \
  --sys-id         "east_${LANG_TGT}"
conda deactivate

# --- Step 3: segale-align 24-shard array (preempt) -------------------------
echo "[3/7] segale-align ${ALIGN_SHARDS}-shard array (preempt)"
ALIGN_JID=$(sbatch --parsable \
  --partition=preempt --qos=preempt_qos --requeue \
  --array=0-$((ALIGN_SHARDS - 1))%${ALIGN_SHARDS} \
  --gres=gpu:1 \
  --mem=32G \
  --exclude="${EXCLUDE_NODES}" \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT},TASK_LANG=${LANG_TGT}" \
  -o "${LOGDIR}/segale_align_%A_%a.out" \
  -e "${LOGDIR}/segale_align_%A_%a.err" \
  "${SEGALE_DIR}/run_segale_align_8gpu.sbatch")
echo "  align array: ${ALIGN_JID}"

# --- Step 4: merge aligned shards (preempt) -------------------------------
echo "[4/7] merge aligned shards"
MERGE_JID=$(sbatch --parsable \
  --dependency=afterok:${ALIGN_JID} \
  --partition=preempt --qos=preempt_qos --requeue \
  --time=00:20:00 \
  --gres=gpu:1 \
  --exclude="${EXCLUDE_NODES}" \
  --cpus-per-task=2 --mem=8G \
  --job-name=east_${LANG_TGT}_segmerge \
  -o "${LOGDIR}/segale_merge_%j.out" \
  -e "${LOGDIR}/segale_merge_%j.err" \
  --wrap "source ~/.bashrc && conda activate /data/user_data/haolingp/conda_envs/segale && python ${SEGALE_DIR}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards ${ALIGN_SHARDS}")
echo "  merge: ${MERGE_JID}"

# --- Step 5: QE prepare (preempt) -----------------------------------------
echo "[5/7] QE prepare"
QE_PREP_JID=$(sbatch --parsable \
  --dependency=afterok:${MERGE_JID} \
  --partition=preempt --qos=preempt_qos --requeue \
  --gres=gpu:1 \
  --mem=12G \
  --exclude="${EXCLUDE_NODES}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_MERGED},CONSENSUS_ROOT=${CONS_FMT},METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  -o "${LOGDIR}/qe_prep_%j.out" \
  -e "${LOGDIR}/qe_prep_%j.err" \
  "${SEGALE_DIR}/run_qe_prepare_aligned.sbatch")
echo "  qe prepare: ${QE_PREP_JID}"

# --- Step 6: MetricX QE 8-shard array (preempt) ---------------------------
echo "[6/7] MetricX QE ${QE_SHARDS}-shard array (preempt)"
QE_PRED_JID=$(sbatch --parsable \
  --dependency=afterok:${QE_PREP_JID} \
  --partition=preempt --qos=preempt_qos --requeue \
  --array=0-$((QE_SHARDS - 1))%${QE_SHARDS} \
  --gres=gpu:1 \
  --mem=32G \
  --exclude="${EXCLUDE_NODES}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  -o "${LOGDIR}/qe_predict_%A_%a.out" \
  -e "${LOGDIR}/qe_predict_%A_%a.err" \
  "${TOPK_DIR}/run_metricx_qe_8gpu.sbatch")
echo "  qe predict ${QE_SHARDS}x: ${QE_PRED_JID}"

# --- Step 7: custom finalize (sbatch's #SBATCH already preempt) -----------
echo "[7/7] custom finalize"
FIN_JID=$(sbatch --parsable \
  --dependency=afterok:${QE_PRED_JID} \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},CONS_FMT_ROOT=${CONS_FMT},STREAM_DIR=${STREAM_DIR},FINAL_OUT_DIR=${FINAL_OUT},QE_THRESHOLD=${QE_THRESHOLD}" \
  -o "${LOGDIR}/finalize_%j.out" \
  -e "${LOGDIR}/finalize_%j.err" \
  "${SCRIPT_DIR}/run_subsentence_finalize_de.sbatch")
echo "  finalize: ${FIN_JID}"

echo
echo "Pipeline submitted (preempt, ${ALIGN_SHARDS}-shard align, ${QE_SHARDS}-shard QE):"
echo "  prepare       : done (inline)"
echo "  align(${ALIGN_SHARDS}x)    : ${ALIGN_JID}"
echo "  merge         : ${MERGE_JID}        (afterok:${ALIGN_JID})"
echo "  qe prepare    : ${QE_PREP_JID}      (afterok:${MERGE_JID})"
echo "  qe predict(${QE_SHARDS}x): ${QE_PRED_JID}      (afterok:${QE_PREP_JID})"
echo "  finalize      : ${FIN_JID}          (afterok:${QE_PRED_JID})"
echo
echo "Outputs:"
echo "  aligned merged : ${ALIGNED_MERGED}"
echo "  metricx run    : ${METRICX_RUN_DIR}"
echo "  final dataset  : ${FINAL_OUT}"
echo "  logs           : ${LOGDIR}"
