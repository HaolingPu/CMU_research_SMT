#!/usr/bin/env bash
# ============================================================
# EAST ja  -- sub-sentence QE: finalize with per-latency thresholds.
# Re-uses the merged metricx_output.jsonl (already produced); skips
# the shard-merge step. Writes a new metricx_filtered file tagged
# with the (low,med,high) thresholds and rebuilds final_jsonl_east.
#
# Required env:
#   METRICX_RUN_DIR  CONS_FMT_ROOT  STREAM_DIR  FINAL_OUT_DIR
#   THR_LOW  THR_MED  THR_HIGH
# ============================================================
#SBATCH --job-name=east_ja_subqe_fin_perlat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2:00:00
#SBATCH --exclude=babel-p9-28,babel-s5-32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== EAST ja per-latency finalize START $(date) ====="
source ~/.bashrc

METRICX_RUN_DIR="${METRICX_RUN_DIR:?required}"
CONS_FMT_ROOT="${CONS_FMT_ROOT:?required}"
STREAM_DIR="${STREAM_DIR:?required}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:?required}"
THR_LOW="${THR_LOW:?required}"
THR_MED="${THR_MED:?required}"
THR_HIGH="${THR_HIGH:?required}"

CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech
MERGED="${METRICX_RUN_DIR}/metricx_output.jsonl"
TAG="L${THR_LOW}_M${THR_MED}_H${THR_HIGH}"
FILTERED="${METRICX_RUN_DIR}/metricx_filtered_subqe_${TAG}.jsonl"
REPORT="${METRICX_RUN_DIR}/filter_report_subqe_${TAG}.jsonl"

echo "thresholds: low=${THR_LOW} med=${THR_MED} high=${THR_HIGH}"
echo "merged    : ${MERGED}  ($(wc -l < ${MERGED}) lines)"

conda activate metricx
python "${CODE}/east/finalize_segale_qe_east.py" \
  --metricx-output         "${MERGED}" \
  --consensus-format-root  "${CONS_FMT_ROOT}" \
  --filtered-output        "${FILTERED}" \
  --threshold-low          "${THR_LOW}" \
  --threshold-medium       "${THR_MED}" \
  --threshold-high         "${THR_HIGH}" \
  --report                 "${REPORT}"

KEPT=$(wc -l < "${FILTERED}")
echo "[filter] kept (utt,latency) rows: ${KEPT}"

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

# per-latency kept count from filtered file
echo "[final] per-latency kept counts:"
python - <<PY
import json
from collections import Counter
c = Counter()
for line in open("${FILTERED}"):
    try: r = json.loads(line)
    except: continue
    c[r.get("metadata",{}).get("latency","?")] += 1
for k,v in sorted(c.items()): print(f"   {k:7s}: {v}")
PY

echo "===== EAST ja per-latency finalize DONE $(date) ====="
