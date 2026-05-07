#!/usr/bin/env bash
# Sweep NUM_CONCURRENT_CASES at fixed workload to find the throughput sweet spot.
#
# Each run hits the already-running localhost vLLM servers (see
# scripts/synth/debug_consensus.sh). Servers must be up before running this.
#
# Defaults to the production workload knobs and Qwen instruct.
# To benchmark with Gemma instead, export INSTRUCT_API_MODEL and
# INSTRUCT_TOKENIZER_PATH before running.
#
# Examples:
#   # Qwen instruct (defaults)
#   bash scripts/synth/bench_concurrency.sh
#
#   # Gemma instruct
#   INSTRUCT_API_MODEL=gemma4-e4b-it \
#   INSTRUCT_TOKENIZER_PATH=/data/user_data/siqiouya/ckpts/pretrained/llm/gemma-4-E4B-it \
#     bash scripts/synth/bench_concurrency.sh
#
#   # Smaller / faster sweep
#   MAX_ROWS=32 CONCURRENCIES="8 16" bash scripts/synth/bench_concurrency.sh
#
# Per-run log: /tmp/bench_logs/${TAG}_c${cc}.log
# Per-run output dir: ${REPO_ROOT}/debug_output/${TAG}_c${cc}/
# Final summary: prints to stdout AND saved at /tmp/bench_logs/${TAG}_summary.txt
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEBUG_SH="${REPO_ROOT}/scripts/synth/debug_consensus.sh"

# === Workload knobs (override via env vars) ===
TAG="${TAG:-bench}"
MAX_ROWS="${MAX_ROWS:-96}"
MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS:-10}"
NUM_FUTURES="${NUM_FUTURES:-200}"
SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES:-100}"
FUTURE_TOKENS="${FUTURE_TOKENS:-10}"
CONCURRENCIES="${CONCURRENCIES:-4 8 16 32}"

LOG_DIR="${LOG_DIR:-/tmp/bench_logs}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/debug_output}"
mkdir -p "${LOG_DIR}"

# Auto-tag with the instruct model so qwen and gemma runs don't clobber each other.
INSTRUCT_TAG="${INSTRUCT_API_MODEL:-qwen3-instruct}"
TAG="${TAG}_${INSTRUCT_TAG}"
SUMMARY_FILE="${LOG_DIR}/${TAG}_summary.txt"

echo "Benchmark: TAG=${TAG} MAX_ROWS=${MAX_ROWS} CONCURRENCIES='${CONCURRENCIES}'" | tee "${SUMMARY_FILE}"
echo "Logs:    ${LOG_DIR}/${TAG}_c<N>.log" | tee -a "${SUMMARY_FILE}"
echo "Outputs: ${OUTPUT_BASE}/${TAG}_c<N>/" | tee -a "${SUMMARY_FILE}"
echo | tee -a "${SUMMARY_FILE}"

run_one() {
  local cc="$1"
  local log="${LOG_DIR}/${TAG}_c${cc}.log"
  local out="${OUTPUT_BASE}/${TAG}_c${cc}"
  rm -rf "${out}"
  echo "===== START c=${cc} at $(date +%H:%M:%S) =====" | tee -a "${SUMMARY_FILE}"
  MAX_ROWS="${MAX_ROWS}" \
  MAX_CONSENSUS_STEPS="${MAX_CONSENSUS_STEPS}" \
  NUM_FUTURES="${NUM_FUTURES}" \
  SECONDARY_NUM_FUTURES="${SECONDARY_NUM_FUTURES}" \
  FUTURE_TOKENS="${FUTURE_TOKENS}" \
  NUM_CONCURRENT_CASES="${cc}" \
  OUTPUT_DIR="${out}" \
  bash "${DEBUG_SH}" > "${log}" 2>&1
  echo "===== DONE  c=${cc} at $(date +%H:%M:%S) =====" | tee -a "${SUMMARY_FILE}"
  # Capture the BENCHMARK SUMMARY block to the summary file
  awk '/BENCHMARK SUMMARY/,/^=========/' "${log}" | tail -20 | tee -a "${SUMMARY_FILE}"
  echo | tee -a "${SUMMARY_FILE}"
}

for cc in ${CONCURRENCIES}; do
  run_one "${cc}"
done

echo "ALL DONE at $(date +%H:%M:%S)" | tee -a "${SUMMARY_FILE}"
echo "Summary saved at: ${SUMMARY_FILE}"
