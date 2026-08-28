#!/usr/bin/env bash
set -e

# Submit dependent LA random-segment postprocessing:
# generation -> merge/SEGALE shard prep -> SEGALE align -> merge aligned
# -> MetricX QE -> QE filter -> length-ratio filter.
#
# Usage:
#   bash submit_la_rand40k_segale_qe_preempt.sh <GEN_JOB_ID>

GEN_JOB_ID="${1:?Usage: $0 <GEN_JOB_ID>}"

LA_SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement/script"
SEGALE_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/segale"
TOPK_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk"

CONSENSUS_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k"
OUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k_segale"
NUM_DOCS=40000
NUM_SHARDS=24
SYS_ID="la_rand40k"
QE_THRESHOLD=3.0
MAX_RATIO_REF=1.5
MIN_RATIO_REF=0.7

SHARDS_ROOT="${OUT_ROOT}/shards"
ALIGNED_MERGED="${OUT_ROOT}/aligned_all.jsonl"
METRICX_RUN_DIR="${OUT_ROOT}/metricx"
QE_FILTERED_DIR="${OUT_ROOT}/qe3"
FINAL_OUT_DIR="${OUT_ROOT}/qe3_lr"

mkdir -p "${OUT_ROOT}" /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

echo "GEN_JOB_ID=${GEN_JOB_ID}"
echo "CONSENSUS_ROOT=${CONSENSUS_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "NUM_DOCS=${NUM_DOCS}"
echo "NUM_SHARDS=${NUM_SHARDS}"

PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${GEN_JOB_ID}" \
  --export="ALL,CONSENSUS_ROOT=${CONSENSUS_ROOT},OUT_ROOT=${OUT_ROOT},NUM_DOCS=${NUM_DOCS},NUM_SHARDS=${NUM_SHARDS},SYS_ID=${SYS_ID}" \
  "${LA_SCRIPT_DIR}/run_la_segale_prepare_24_preempt.sbatch")

ALIGN_JOB=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB}" \
  --partition=preempt --qos=preempt_qos --requeue \
  --array=0-23%24 \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT}" \
  "${SEGALE_DIR}/run_segale_align_8gpu.sbatch")

MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ALIGN_JOB}" \
  --partition=preempt --qos=preempt_qos --requeue \
  --time=00:30:00 \
  --gres=gpu:L40S:1 \
  --cpus-per-task=4 --mem=12G \
  --job-name=la_seg_merge \
  -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_seg_merge_%j.out \
  -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_seg_merge_%j.err \
  --wrap "source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && python ${SEGALE_DIR}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards ${NUM_SHARDS}")

QE_PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${MERGE_JOB}" \
  --export="ALL,ALIGNED_FILE=${ALIGNED_MERGED},CONSENSUS_ROOT=${CONSENSUS_ROOT},METRICX_RUN_DIR=${METRICX_RUN_DIR},NUM_SHARDS=${NUM_SHARDS}" \
  "${LA_SCRIPT_DIR}/run_la_qe_prepare_aligned_24_preempt.sbatch")

QE_PREDICT_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREP_JOB}" \
  --partition=preempt --qos=preempt_qos --requeue \
  --array=0-23%24 \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${TOPK_DIR}/run_metricx_qe_8gpu.sbatch")

QE_FIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREDICT_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},QE_FILTERED_DIR=${QE_FILTERED_DIR},FINAL_OUT_DIR=${FINAL_OUT_DIR},QE_THRESHOLD=${QE_THRESHOLD},MAX_RATIO_REF=${MAX_RATIO_REF},MIN_RATIO_REF=${MIN_RATIO_REF},NUM_SHARDS=${NUM_SHARDS}" \
  "${LA_SCRIPT_DIR}/run_la_qe_finalize_aligned_24_preempt.sbatch")

echo "Pipeline submitted:"
echo "  generation     : ${GEN_JOB_ID}"
echo "  prepare        : ${PREP_JOB}       (afterok:${GEN_JOB_ID})"
echo "  align 24x      : ${ALIGN_JOB}      (afterok:${PREP_JOB})"
echo "  merge aligned  : ${MERGE_JOB}      (afterok:${ALIGN_JOB})"
echo "  qe prepare     : ${QE_PREP_JOB}    (afterok:${MERGE_JOB})"
echo "  qe predict 24x : ${QE_PREDICT_JOB} (afterok:${QE_PREP_JOB})"
echo "  qe+lr finalize : ${QE_FIN_JOB}     (afterok:${QE_PREDICT_JOB})"
echo
echo "Final output:"
echo "  ${FINAL_OUT_DIR}"
