#!/usr/bin/env bash
# SALAMI ja SEGALE QE finalize @ threshold 7.0095 (yields exactly 12,500 whole-doc).
# Reuses already-merged metricx_output.jsonl; rebuilds final_jsonl_salami tree.
#SBATCH --job-name=salami_ja_fin_t7p01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --time=1:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/seg_fin_t7p01_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/seg_fin_t7p01_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== SALAMI ja SEGALE finalize @ t=7.0095 START $(date) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech
ROOT=${BASE}/segale_pipeline

MERGED=${ROOT}/metricx_aligned/metricx_output.jsonl
CONS_ROOT=${ROOT}/consensus_format
STREAM_DIR=${BASE}/streaming_salami_dataset

THR=7.0095
TAG="t7p01"
FILTERED=${ROOT}/metricx_filtered_${TAG}.jsonl
REPORT=${ROOT}/finalize_report_${TAG}.jsonl
FINAL_DIR=${BASE}/final_jsonl_salami

echo "merged metricx : ${MERGED}  ($(wc -l < ${MERGED}) lines)"
echo "threshold      : ${THR}"
echo "filtered out   : ${FILTERED}"
echo "final dir      : ${FINAL_DIR}"

conda activate metricx
python ${CODE}/east/finalize_segale_qe_east.py \
  --metricx-output        ${MERGED} \
  --consensus-format-root ${CONS_ROOT} \
  --filtered-output       ${FILTERED} \
  --threshold             ${THR} \
  --report                ${REPORT}

KEPT=$(wc -l < ${FILTERED})
echo "[filter] kept docs: ${KEPT}"

conda deactivate
conda activate SMT
rm -rf ${FINAL_DIR}
python ${CODE}/final_output_gigaspeech.py \
  --metricx_jsonl ${FILTERED} \
  --stream_dir    ${STREAM_DIR} \
  --output_dir    ${FINAL_DIR}

NUM_UTTS=$(ls -1 ${FINAL_DIR} 2>/dev/null | wc -l)
NUM_FILES=$(find ${FINAL_DIR} -name 'offline_latency.jsonl' 2>/dev/null | wc -l)
TOTAL_LINES=$(find ${FINAL_DIR} -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
echo "[final] utt dirs            : ${NUM_UTTS}"
echo "[final] offline jsonl files : ${NUM_FILES}"
echo "[final] total kept rows     : ${TOTAL_LINES}"

echo "===== SALAMI ja SEGALE finalize @ t=7.0095 DONE $(date) ====="
