#!/usr/bin/env bash
# SALAMI de SEGALE QE finalize.
# Step 1: merge metricx shards -> metricx_output.jsonl
# Step 2: report per-doc max distribution + auto-pick threshold to keep ~12,500 whole-docs
# Step 3: finalize with the chosen threshold + build final_jsonl_salami tree
#SBATCH --job-name=salami_seg_fin_de
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
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_fin_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de/slurm_logs/seg_fin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== SALAMI de SEGALE finalize START $(date) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_de
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech
ROOT=${BASE}/segale_pipeline

SHARDS=${ROOT}/metricx_aligned/metricx_shards
MERGED=${ROOT}/metricx_aligned/metricx_output.jsonl
CONS_ROOT=${ROOT}/consensus_format
STREAM_DIR=${BASE}/streaming_salami_dataset
FINAL_DIR=${BASE}/final_jsonl_salami

# Step 1: merge metricx shard outputs
echo "[merge] cat ${SHARDS}/output_*.jsonl -> ${MERGED}"
cat ${SHARDS}/output_*.jsonl > ${MERGED}
echo "[merge] $(wc -l < ${MERGED}) lines"

# Step 2: pick threshold yielding ~12,500 whole-docs (per-doc MAX score)
THR_TARGET=12500
THR=$(python - <<PYEOF
import json
from collections import defaultdict
m=defaultdict(lambda: float("-inf"))
with open("${MERGED}") as f:
    for line in f:
        r=json.loads(line)
        meta=r.get("metadata") or {}
        utt=meta.get("utt_id")
        s=r.get("prediction")
        if utt is None or s is None: continue
        s=float(s)
        if s>m[utt]: m[utt]=s
xs=sorted(m.values())
N=${THR_TARGET}
if N>len(xs): N=len(xs)
print(f"{xs[N-1]:.6f}")
PYEOF
)
TAG="t$(printf '%.4f' ${THR} | tr '.' 'p')"
FILTERED=${ROOT}/metricx_filtered_${TAG}.jsonl
REPORT=${ROOT}/finalize_report_${TAG}.jsonl

echo "[threshold] auto-picked THR=${THR} (target ${THR_TARGET} docs)  TAG=${TAG}"
echo "[filter] out: ${FILTERED}"
echo "[final ] dir: ${FINAL_DIR}"

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

echo "===== SALAMI de SEGALE finalize DONE $(date) ====="
