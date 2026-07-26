#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/submit_gemma4_true_minp_100_metrics.sh"

echo "[compare] submit true min-p 0.05"
OUT_005=$(MIN_P=0.05 RUN_TAG=true-minp-0p05-100-metrics bash "${SCRIPT}")
printf '%s\n' "${OUT_005}"
SUMMARY_JOB_005=$(printf '%s\n' "${OUT_005}" | sed -n 's/.*summary job id    : //p' | tail -n 1)

echo
echo "[compare] submit true min-p 0.1"
MIN_P=0.1 RUN_TAG=true-minp-0p1-100-metrics GEN_DEPENDENCY="${SUMMARY_JOB_005}" bash "${SCRIPT}"
