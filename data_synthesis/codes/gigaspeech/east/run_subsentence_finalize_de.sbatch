#!/usr/bin/env bash
# ============================================================
# EAST de  -- sub-sentence QE: finalize.
# Merges metricx shards, runs custom utt-level filter
# (kept iff every sub-sentence QE <= threshold), then builds
# the streaming final_jsonl_east via final_output_gigaspeech.py.
#
# Required env vars:
#   METRICX_RUN_DIR       contains metricx_shards/output_NN.jsonl
#   CONS_FMT_ROOT         consensus-format JSONs root (has east_stream_json field)
#   STREAM_DIR            streaming_EAST_dataset
#   FINAL_OUT_DIR         output dir for streaming final jsonl
# Optional:
#   QE_THRESHOLD          default 3.0
# ============================================================
#SBATCH --job-name=east_de_subqe_fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --exclude=babel-p9-28,babel-s5-32,babel-m5-32,babel-n9-32,babel-o5-16,babel-o5-24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== EAST de sub-sentence QE finalize START $(date) ====="

source ~/.bashrc

METRICX_RUN_DIR="${METRICX_RUN_DIR:?METRICX_RUN_DIR required}"
CONS_FMT_ROOT="${CONS_FMT_ROOT:?CONS_FMT_ROOT required}"
STREAM_DIR="${STREAM_DIR:?STREAM_DIR required}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:?FINAL_OUT_DIR required}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
NUM_SHARDS=8

CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech
MERGED="${METRICX_RUN_DIR}/metricx_output.jsonl"
FILTERED="${METRICX_RUN_DIR}/metricx_filtered_subqe_t${QE_THRESHOLD}.jsonl"
REPORT="${METRICX_RUN_DIR}/filter_report_subqe.jsonl"

echo "METRICX_RUN_DIR : ${METRICX_RUN_DIR}"
echo "CONS_FMT_ROOT   : ${CONS_FMT_ROOT}"
echo "STREAM_DIR      : ${STREAM_DIR}"
echo "FINAL_OUT_DIR   : ${FINAL_OUT_DIR}"
echo "QE_THRESHOLD    : ${QE_THRESHOLD}"

echo "[merge] ${NUM_SHARDS} metricx shards -> ${MERGED}"
rm -f "${MERGED}"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SH=$(printf "%02d" "$i")
  F="${METRICX_RUN_DIR}/metricx_shards/output_${SH}.jsonl"
  if [[ -f "${F}" ]]; then cat "${F}" >> "${MERGED}"
  else echo "[WARN] missing shard: ${F}"
  fi
done
echo "[merge] $(wc -l < ${MERGED}) lines"

conda activate metricx
echo "[filter] sub-sentence -> per (utt,latency) filter at thr=${QE_THRESHOLD}"
python "${CODE}/east/finalize_segale_qe_east.py" \
  --metricx-output         "${MERGED}" \
  --consensus-format-root  "${CONS_FMT_ROOT}" \
  --filtered-output        "${FILTERED}" \
  --threshold              "${QE_THRESHOLD}" \
  --report                 "${REPORT}"

KEPT=$(wc -l < "${FILTERED}")
echo "[filter] kept (utt,latency) rows: ${KEPT}"

if [[ "${KEPT}" == "0" ]]; then
  echo "WARNING: no rows kept; final dataset will be empty."
fi

conda deactivate
conda activate SMT

rm -rf "${FINAL_OUT_DIR}"
python "${CODE}/final_output_gigaspeech.py" \
  --metricx_jsonl "${FILTERED}" \
  --stream_dir    "${STREAM_DIR}" \
  --output_dir    "${FINAL_OUT_DIR}"

NUM_UTTS=$(ls -1 "${FINAL_OUT_DIR}" 2>/dev/null | wc -l)
NUM_FILES=$(find "${FINAL_OUT_DIR}" -name '*.jsonl' 2>/dev/null | wc -l)
TOTAL_LINES=$(find "${FINAL_OUT_DIR}" -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
echo "[final] utt dirs               : ${NUM_UTTS}"
echo "[final] .jsonl files           : ${NUM_FILES}"
echo "[final] total trajectory lines : ${TOTAL_LINES}"

echo "===== EAST de sub-sentence QE finalize DONE $(date) ====="
