#!/usr/bin/env bash
# Resource flags are supplied by submit_external_second_half.sh so this file is
# portable across clusters with different partition and GPU naming conventions.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKER="${SCRIPT_DIR}/run_external_ambiguity_worker.sh"

if [[ -z "${CONTAINER_IMAGE:-}" ]]; then
  exec bash "${WORKER}"
fi

command -v apptainer >/dev/null 2>&1 || {
  echo "[ERROR] CONTAINER_IMAGE is set but apptainer is unavailable" >&2
  exit 2
}

# Keep every host path unchanged inside the container. This lets the same
# worker and manifest run in Conda or container mode without path rewriting.
BIND_PATHS=(
  "${REPO_ROOT}"
  "$(dirname -- "${INPUT_TSV}")"
  "${OUTPUT_ROOT}"
  "${QWEN38_MODEL}"
  "${GEMMA_MODEL}"
  "${QWEN36_MODEL}"
  "${HF_HOME}"
)

APPTAINER_ARGS=(exec --nv --cleanenv --pwd "${REPO_ROOT}")
declare -A SEEN_BINDS=()
for path in "${BIND_PATHS[@]}"; do
  [[ -e "${path}" ]] || {
    echo "[ERROR] bind source does not exist: ${path}" >&2
    exit 2
  }
  if [[ -z "${SEEN_BINDS[${path}]:-}" ]]; then
    APPTAINER_ARGS+=(--bind "${path}:${path}")
    SEEN_BINDS[${path}]=1
  fi
done

ENV_NAMES=(
  REPO_ROOT INPUT_TSV OUTPUT_ROOT QWEN38_MODEL GEMMA_MODEL QWEN36_MODEL
  NUM_TASKS ROW_OFFSET SLICE_ROWS OUTPUT_TASK_OFFSET NUM_CONCURRENT_CASES
  TARGETED_NUM_FUTURES MIN_VOTERS_RATIO FUTURE_SRC_WINDOW ID_COLUMN PORT_BASE
  HF_HOME HF_HUB_CACHE SLURM_ARRAY_TASK_ID SLURM_JOB_ID
)
for name in "${ENV_NAMES[@]}"; do
  [[ -z "${!name:-}" ]] || APPTAINER_ARGS+=(--env "${name}=${!name}")
done
APPTAINER_ARGS+=(
  --env "PYTHON_BIN=/opt/generation/bin/python"
  --env "VLLM_BIN=/opt/generation/bin/vllm"
  --env "GEMMA4_ENV=/opt/generation"
)

echo "container_image=${CONTAINER_IMAGE}"
exec apptainer "${APPTAINER_ARGS[@]}" "${CONTAINER_IMAGE}" bash "${WORKER}"
