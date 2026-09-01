#!/usr/bin/env bash
# Submit the non-overlapping second slice (rows 20,004-39,999) on another cluster.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)}"

: "${INPUT_TSV:?Set INPUT_TSV to the frozen input TSV}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a shared output directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the vLLM environment's python}"
: "${QWEN38_MODEL:?Set QWEN38_MODEL to Qwen3.8-27B-FP8}"
: "${GEMMA_MODEL:?Set GEMMA_MODEL to gemma-4-E2B-it}"
: "${QWEN36_MODEL:?Set QWEN36_MODEL to Qwen3.6-35B-A3B-FP8}"

NUM_TASKS="${NUM_TASKS:-8}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-${NUM_TASKS}}"
ROW_OFFSET="${ROW_OFFSET:-20004}"
SLICE_ROWS="${SLICE_ROWS:-19996}"
OUTPUT_TASK_OFFSET="${OUTPUT_TASK_OFFSET:-100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPU_GRES="${GPU_GRES:-gpu:2}"

if (( NUM_TASKS < 1 || MAX_CONCURRENT_TASKS < 1 )); then
  echo "NUM_TASKS and MAX_CONCURRENT_TASKS must be positive" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/slurm_logs"
COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
MANIFEST="${OUTPUT_ROOT}/external_run_manifest.txt"
{
  printf 'submitted_at=%s\n' "$(date -Iseconds)"
  printf 'git_commit=%s\n' "${COMMIT}"
  printf 'input_tsv=%s\n' "${INPUT_TSV}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'row_offset=%s\n' "${ROW_OFFSET}"
  printf 'slice_rows=%s\n' "${SLICE_ROWS}"
  printf 'num_tasks=%s\n' "${NUM_TASKS}"
  printf 'max_concurrent_tasks=%s\n' "${MAX_CONCURRENT_TASKS}"
  printf 'qwen38_model=%s\n' "${QWEN38_MODEL}"
  printf 'gemma_model=%s\n' "${GEMMA_MODEL}"
  printf 'qwen36_model=%s\n' "${QWEN36_MODEL}"
} >"${MANIFEST}"

export REPO_ROOT INPUT_TSV OUTPUT_ROOT PYTHON_BIN
export QWEN38_MODEL GEMMA_MODEL QWEN36_MODEL
export NUM_TASKS ROW_OFFSET SLICE_ROWS OUTPUT_TASK_OFFSET
export NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-12}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

SBATCH_ARGS=(
  --parsable
  --job-name=ambiguity_external_half2
  --nodes=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --gres="${GPU_GRES}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
  --array="0-$((NUM_TASKS - 1))%${MAX_CONCURRENT_TASKS}"
  --requeue
  --output="${OUTPUT_ROOT}/slurm_logs/%A_%a.out"
  --error="${OUTPUT_ROOT}/slurm_logs/%A_%a.err"
  --export=ALL
)
[[ -z "${PARTITION:-}" ]] || SBATCH_ARGS+=(--partition="${PARTITION}")
[[ -z "${QOS:-}" ]] || SBATCH_ARGS+=(--qos="${QOS}")
[[ -z "${ACCOUNT:-}" ]] || SBATCH_ARGS+=(--account="${ACCOUNT}")
[[ -z "${GPU_CONSTRAINT:-}" ]] || SBATCH_ARGS+=(--constraint="${GPU_CONSTRAINT}")

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_external_ambiguity_2gpu.sbatch")
printf 'slurm_job_id=%s\n' "${JOB_ID}" | tee -a "${MANIFEST}"
echo "Submitted ${JOB_ID}; manifest: ${MANIFEST}"
