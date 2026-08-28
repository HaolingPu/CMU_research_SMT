#!/usr/bin/env bash
# ============================================================
# EAST ja  -- sub-sentence QE pipeline orchestrator.
#
# 1) repackage llm_output_merged into fake-consensus JSONs (per (utt,latency))
# 2) prepare segale shards (system.jsonl + ref.jsonl per shard)
# 3) submit segale-align 8-shard array (on a fresh L40S env, 1 GPU each)
# 4) chain merge of aligned shards
# 5) chain QE prepare (convert -> metricx_input.jsonl + 8 shards)
# 6) chain MetricX QE 8-shard array
# 7) chain my custom finalize (utt-level filter + final_output_gigaspeech.py)
# ============================================================
set -e

LANG_TGT=ja
QE_THRESHOLD="${QE_THRESHOLD:-5.0}"
NUM_SHARDS=8
EXCLUDE_NODES="babel-p9-28,babel-s5-32"

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_${LANG_TGT}
LLM_MERGED=${BASE}/llm_output_merged
STREAM_DIR=${BASE}/streaming_EAST_dataset

WORK=${BASE}/segale_qe
CONS_FMT=${WORK}/consensus_format          # fake job_*/task_*/<doc>.json
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

# --- Step 2: prepare segale shards ----------------------------------------
echo "[2/7] prepare segale shards (NUM_DOCS=${CONS_NUM}, ${NUM_SHARDS} shards)"
conda activate /data/user_data/haolingp/conda_envs/segale
python "${SEGALE_DIR}/prepare_segale_shards.py" \
  --consensus-root "${CONS_FMT}" \
  --out-root       "${WORK}" \
  --num-docs       "${CONS_NUM}" \
  --num-shards     "${NUM_SHARDS}" \
  --sys-id         "east_${LANG_TGT}"
conda deactivate

# --- Step 3: segale-align 8-shard array -----------------------------------
echo "[3/7] segale-align 8-shard array"
ALIGN_JID=$(sbatch --parsable \
  --gres=gpu:1 \
  --exclude="${EXCLUDE_NODES}" \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT},TASK_LANG=${LANG_TGT}" \
  -o "${LOGDIR}/segale_align_%A_%a.out" \
  -e "${LOGDIR}/segale_align_%A_%a.err" \
  "${SEGALE_DIR}/run_segale_align_8gpu.sbatch")
echo "  align array: ${ALIGN_JID}"

# --- Step 4: merge aligned shards (depends on align) ----------------------
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

# --- Step 5: QE prepare (convert -> metricx_input + split) ----------------
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

# --- Step 6: MetricX QE 8-shard array (depends on prepare) ----------------
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

# --- Step 7: custom finalize (utt-level filter + final_output) ------------
echo "[7/7] custom finalize"
FIN_JID=$(sbatch --parsable \
  --dependency=afterok:${QE_PRED_JID} \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},CONS_FMT_ROOT=${CONS_FMT},STREAM_DIR=${STREAM_DIR},FINAL_OUT_DIR=${FINAL_OUT},QE_THRESHOLD=${QE_THRESHOLD}" \
  -o "${LOGDIR}/finalize_%j.out" \
  -e "${LOGDIR}/finalize_%j.err" \
  "${SCRIPT_DIR}/run_subsentence_finalize_ja.sbatch")
echo "  finalize: ${FIN_JID}"

echo
echo "Pipeline submitted:"
echo "  prepare       : done (inline)"
echo "  align(8x)     : ${ALIGN_JID}"
echo "  merge         : ${MERGE_JID}        (afterok:${ALIGN_JID})"
echo "  qe prepare    : ${QE_PREP_JID}      (afterok:${MERGE_JID})"
echo "  qe predict(8x): ${QE_PRED_JID}      (afterok:${QE_PREP_JID})"
echo "  finalize      : ${FIN_JID}          (afterok:${QE_PRED_JID})"
echo
echo "Outputs (after completion):"
echo "  aligned merged : ${ALIGNED_MERGED}"
echo "  metricx run    : ${METRICX_RUN_DIR}"
echo "  final dataset  : ${FINAL_OUT}"
echo "  logs           : ${LOGDIR}"
