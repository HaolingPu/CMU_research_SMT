#!/usr/bin/env bash
# End-to-end: run Prefix Alignment on 40k cases, then SEGALE subsentence
# alignment + per-sentence MetricX QE (<= 3.0) + length-ratio filter.
#
# Usage:
#   bash submit_pa_40k.sh
#
# Each PA shard self-hosts a vLLM server on its own GPU; no manual server setup.
#
# Optional env vars (with defaults):
#   MT_API_MODEL    qwen3-instruct
#   MODEL_PATH      /data/user_data/.../Qwen3-30B-A3B-Instruct-2507-FP8
#   TOKENIZER_PATH  same as MODEL_PATH
#   TOTAL_ROWS      40000
#   ROW_OFFSET      0
#   NUM_SHARDS      8
#   SYS_ID          pa_40k
#   QE_THRESHOLD    3.0
#   MAX_RATIO_REF   1.5
#   MIN_RATIO_REF   0.7
#
# Final filtered cases land in:
#   /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/
#       prefix_alignment/pa_40k/qe3-lr-aligned/

set -e

OUT_ROOT="${OUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/prefix_alignment/pa_40k}"
SYS_ID="${SYS_ID:-pa_40k}"
TOTAL_ROWS="${TOTAL_ROWS:-40000}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
MIN_RATIO_REF="${MIN_RATIO_REF:-0.7}"
MT_API_MODEL="${MT_API_MODEL:-qwen3-instruct}"
MODEL_PATH="${MODEL_PATH:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv}"

PA_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/prefix_alignment"
SEGALE_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/segale"
TOPK_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk"

CONSENSUS_ROOT="${OUT_ROOT}/raw"           # contains task_NN/<utt>.json after PA
SHARDS_ROOT="${OUT_ROOT}/shards"
ALIGNED_MERGED="${OUT_ROOT}/aligned_all.jsonl"
METRICX_RUN_DIR="${OUT_ROOT}/metricx-aligned"
QE_FILTERED_DIR="${OUT_ROOT}/qe3-aligned"
FINAL_OUT_DIR="${OUT_ROOT}/qe3-lr-aligned"
SLURM_LOGS=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/prefix_alignment/slurm_logs

mkdir -p "${OUT_ROOT}" "${CONSENSUS_ROOT}" "${SLURM_LOGS}"

echo "OUT_ROOT       : ${OUT_ROOT}"
echo "TOTAL_ROWS     : ${TOTAL_ROWS}  (offset ${ROW_OFFSET})"
echo "NUM_SHARDS     : ${NUM_SHARDS}"
echo "MODEL_PATH     : ${MODEL_PATH}"
echo "SYS_ID         : ${SYS_ID}"
echo

echo "[1/7] Submitting PA ${NUM_SHARDS}-shard array (each shard self-hosts vLLM)..."
PA_JOB=$(sbatch --parsable \
  --array=0-$((NUM_SHARDS - 1))%${NUM_SHARDS} \
  --export="ALL,OUT_ROOT=${OUT_ROOT},MT_API_MODEL=${MT_API_MODEL},MODEL_PATH=${MODEL_PATH},TOKENIZER_PATH=${TOKENIZER_PATH},INPUT_TSV=${INPUT_TSV},TOTAL_ROWS=${TOTAL_ROWS},NUM_SHARDS=${NUM_SHARDS},ROW_OFFSET=${ROW_OFFSET}" \
  "${PA_DIR}/script/run_pa_40k_array.sbatch")
echo "[pa array]      ${PA_JOB}"

echo
echo "[2/7] Submitting SEGALE shard prep (afterok:${PA_JOB})..."
PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${PA_JOB}" \
  --time=00:30:00 \
  --partition=general --qos=normal \
  --gres=gpu:L40S:1 \
  --cpus-per-task=2 --mem=8G \
  --job-name=pa_segprep \
  -o "${SLURM_LOGS}/pa_segprep_%j.out" \
  -e "${SLURM_LOGS}/pa_segprep_%j.err" \
  --wrap "python ${SEGALE_DIR}/prepare_segale_shards.py --consensus-root ${CONSENSUS_ROOT} --out-root ${OUT_ROOT} --num-docs ${TOTAL_ROWS} --num-shards 8 --sys-id ${SYS_ID}")
echo "[seg prep]      ${PREP_JOB}"

echo
echo "[3/7] Submitting 8-GPU SEGALE align array (afterok:${PREP_JOB})..."
ALIGN_JOB=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB}" \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT}" \
  "${SEGALE_DIR}/run_segale_align_8gpu.sbatch")
echo "[align array]   ${ALIGN_JOB}"

echo
echo "[4/7] Submitting align merge (afterok:${ALIGN_JOB})..."
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ALIGN_JOB}" \
  --time=00:20:00 \
  --partition=general --qos=normal \
  --gres=gpu:L40S:1 \
  --cpus-per-task=2 --mem=8G \
  --job-name=pa_segmerge \
  -o "${SLURM_LOGS}/pa_segmerge_%j.out" \
  -e "${SLURM_LOGS}/pa_segmerge_%j.err" \
  --wrap "source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && python ${SEGALE_DIR}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards 8")
echo "[merge]         ${MERGE_JOB}"

echo
echo "[5/7] Submitting QE prepare (afterok:${MERGE_JOB})..."
QE_PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${MERGE_JOB}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_MERGED},CONSENSUS_ROOT=${CONSENSUS_ROOT},METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SEGALE_DIR}/run_qe_prepare_aligned.sbatch")
echo "[qe prepare]    ${QE_PREP_JOB}"

echo
echo "[6/7] Submitting 8-GPU MetricX predict (afterok:${QE_PREP_JOB})..."
QE_PREDICT_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREP_JOB}" \
  --exclude=babel-t9-16 \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${TOPK_DIR}/run_metricx_qe_8gpu.sbatch")
echo "[qe predict 8x] ${QE_PREDICT_JOB}"

echo
echo "[7/7] Submitting QE finalize + length-ratio filter (afterok:${QE_PREDICT_JOB})..."
QE_FIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREDICT_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},QE_FILTERED_DIR=${QE_FILTERED_DIR},FINAL_OUT_DIR=${FINAL_OUT_DIR},QE_THRESHOLD=${QE_THRESHOLD},MAX_RATIO_REF=${MAX_RATIO_REF},MIN_RATIO_REF=${MIN_RATIO_REF}" \
  "${SEGALE_DIR}/run_qe_finalize_aligned.sbatch")
echo "[qe finalize]   ${QE_FIN_JOB}"

echo
echo "Pipeline submitted:"
echo "  pa(8x)        : ${PA_JOB}"
echo "  seg prep      : ${PREP_JOB}        (afterok:${PA_JOB})"
echo "  align(8x)     : ${ALIGN_JOB}       (afterok:${PREP_JOB})"
echo "  merge         : ${MERGE_JOB}       (afterok:${ALIGN_JOB})"
echo "  qe prepare    : ${QE_PREP_JOB}     (afterok:${MERGE_JOB})"
echo "  qe predict(8x): ${QE_PREDICT_JOB}  (afterok:${QE_PREP_JOB})"
echo "  qe finalize   : ${QE_FIN_JOB}      (afterok:${QE_PREDICT_JOB})"
echo
echo "Final filtered training data (after success):"
echo "  ${FINAL_OUT_DIR}"
