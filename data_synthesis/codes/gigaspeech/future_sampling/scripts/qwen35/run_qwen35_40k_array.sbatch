#!/usr/bin/env bash
#SBATCH --job-name=qwen35_40k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --partition=array
#SBATCH --qos=array_qos
#SBATCH --time=6-00:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_40k_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_40k_%A_%a.err

# 40k-row consensus decoding on the OLD-ASR frozen TSV with the Qwen3.5-122B
# THINKING sampler served by a remote replica farm (serve_qwen35_122b_a100_dp.sbatch).
#
# 8 array tasks (array_qos: 8 GPU / 100 running jobs), 1x L40S each for the
# local instruct translator, 5000 rows per task, sharded round-robin across
# the sampler endpoints. Requeue + --skip-existing make this preemption-safe.
#
# Sampler endpoints: by default AUTO-DISCOVERED from ENDPOINT_DIR — every serve
# script (serve_qwen35_122b.sbatch on L40S, serve_qwen35_122b_a100*.sbatch on
# A100) self-registers there, and this script health-checks each entry and keeps
# only live ones. Works with any mix of A100/L40S replicas. To pin explicitly:
#   sbatch --export=ALL,LABEL=qwen35_40k_oldasr,SAMPLER_API_BASES="http://babel-v5-16:8300/v1 ..." \
#     run_qwen35_40k_array.sbatch
#
# Throughput knobs (defaults tuned for the farm; override via --export):
#   NUM_CONCURRENT_CASES=64  keep the sampler replicas saturated (8 was far
#                            below saturation: ~315 tok/s aggregate measured)
#   GPT_API_TIMEOUT=1800     thinking calls queue under heavy batch
#   ENABLE_VERBOSE=0         40k verbose logs would be pure disk noise

set -e

LABEL="${LABEL:?Set LABEL, e.g. qwen35_40k_oldasr}"
ENDPOINT_DIR="${ENDPOINT_DIR:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/qwen35_endpoints}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ -z "${SAMPLER_API_BASES:-}" ]]; then
  # Auto-discover: collect registered endpoints, keep only ones that answer.
  # Stale files from dead/preempted serve jobs are filtered out here.
  CANDIDATES=$(cat "${ENDPOINT_DIR}"/*.txt 2>/dev/null | sort -u || true)
  SAMPLER_API_BASES=""
  for ep in ${CANDIDATES}; do
    if curl -s --max-time 5 "${ep%/v1}/health" > /dev/null 2>&1 \
       || curl -s --max-time 5 "${ep}/models" > /dev/null 2>&1; then
      SAMPLER_API_BASES="${SAMPLER_API_BASES} ${ep}"
    else
      echo "[skip] dead endpoint ${ep}"
    fi
  done
  SAMPLER_API_BASES="${SAMPLER_API_BASES# }"
fi
if [[ -z "${SAMPLER_API_BASES}" ]]; then
  echo "[ERROR] no live sampler endpoints (checked ${ENDPOINT_DIR}) — start serve_qwen35_122b*.sbatch first"
  exit 1
fi

# Round-robin this task onto one sampler endpoint.
read -r -a ENDPOINTS <<< "${SAMPLER_API_BASES}"
N_EP="${#ENDPOINTS[@]}"
export SAMPLER_API_BASE="${ENDPOINTS[$(( TASK_ID % N_EP ))]}"
echo "task ${TASK_ID} -> sampler ${SAMPLER_API_BASE} (${N_EP} endpoints)"

export TOTAL_ROWS="${TOTAL_ROWS:-40000}"
export NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-64}"
export GPT_API_TIMEOUT="${GPT_API_TIMEOUT:-1800}"
export ENABLE_VERBOSE="${ENABLE_VERBOSE:-0}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/qwen35/${LABEL}}"
# Stable across requeues so --skip-existing resumes into the same run dir.
export OUTPUT_JOB_ID="${OUTPUT_JOB_ID:-${SLURM_ARRAY_JOB_ID}}"

bash /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/qwen35/run_qwen35_common.sh "${LABEL}"
