#!/usr/bin/env bash
#SBATCH --job-name=icl50_report
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
RUN_DIR="${RUN_DIR:?RUN_DIR required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
PYTHON=/data/user_data/haolingp/conda_envs/gemma4/bin/python
"${PYTHON}" "${RUN_DIR}/runtime/pilot_case_guard.py" report --manifest "${RUN_DIR}/run_manifest.json"
if [[ -f "${RUN_DIR}/review/READY.json" ]]; then
  echo "Verified review bundle already published"
  exit 0
fi
BUILD="${RUN_DIR}/review-attempt-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}"
"${PYTHON}" "${RUN_DIR}/runtime/build_review_bundle.py" \
  --decode-root "${OUTPUT_ROOT}" --pilot-arm suffix_icl_v3 \
  --input-tsv "${RUN_DIR}/selection/input_50.tsv" \
  --output-dir "${BUILD}/suffix_icl_v3" --limit 50 --require-complete --skip-audio \
  --run-name "Source-only boundary + v3 suffix ICL (same 50 cases)"
cp "${OUTPUT_ROOT}/pilot_summary.json" "${BUILD}/pilot_summary.json"
cp "${OUTPUT_ROOT}/case_comparison.csv" "${BUILD}/case_comparison.csv"
cp "${RUN_DIR}/selection/cases.json" "${BUILD}/cases.json"
cp "${RUN_DIR}/selection/input_50.tsv" "${BUILD}/input_50.tsv"
cp "${RUN_DIR}/run_manifest.json" "${BUILD}/READY.json"
test ! -e "${RUN_DIR}/review"
mv "${BUILD}" "${RUN_DIR}/review"
echo "[READY] ${RUN_DIR}/review (50 verified cases)"
