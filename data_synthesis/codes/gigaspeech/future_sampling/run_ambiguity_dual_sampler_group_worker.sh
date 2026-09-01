#!/usr/bin/env bash
# One 3-GPU group: two Qwen3.8+Gemma sampler workers share one Qwen3.6 probe.

set -euo pipefail

GROUP_ID="${SLURM_ARRAY_TASK_ID:-0}"
REPO="/home/haolingp/CMU_research_SMT"
FS="${REPO}/data_synthesis/codes/gigaspeech/future_sampling"
ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${ENV}/bin/python"

SAMPLER_SERVER="${FS}/scripts/qwen38/serve_qwen38_gemma_colocated.sh"
TRANSLATOR_SERVER="${FS}/scripts/qwen36/serve_qwen36_35b.sh"
DECODER="${FS}/consensus_decoding_token_id_level_instruct.py"

QWEN38_MODEL="${QWEN38_MODEL:-/data/user_data/haolingp/models/Qwen3.8-27B-FP8}"
GEMMA_MODEL="${GEMMA_MODEL:-/data/user_data/haolingp/models/gemma-4-E2B-it}"
QWEN36_MODEL="${QWEN36_MODEL:-/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
TOTAL_ROWS="${TOTAL_ROWS:-40000}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_LOGICAL_WORKERS="${NUM_LOGICAL_WORKERS:-16}"
OUTPUT_TASK_OFFSET="${OUTPUT_TASK_OFFSET:-12}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-12}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
FUTURE_SRC_WINDOW="${FUTURE_SRC_WINDOW:-1}"

LOGICAL_A=$(( 2 * GROUP_ID ))
LOGICAL_B=$(( LOGICAL_A + 1 ))
ROWS_PER_WORKER=$(( (TOTAL_ROWS + NUM_LOGICAL_WORKERS - 1) / NUM_LOGICAL_WORKERS ))
PORT_BASE=$(( 9000 + 8 * GROUP_ID ))
QWEN_A_PORT=${PORT_BASE}
GEMMA_A_PORT=$(( PORT_BASE + 1 ))
QWEN_B_PORT=$(( PORT_BASE + 2 ))
GEMMA_B_PORT=$(( PORT_BASE + 3 ))
QWEN36_PORT=$(( PORT_BASE + 4 ))
RUN_LOG_DIR="${OUTPUT_ROOT}/serve_logs/dual_group_$(printf '%02d' "${GROUP_ID}")"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${RUN_LOG_DIR}"

pid_file() {
  printf '/tmp/amb_dual_%s_%s_%s.pid' "${SLURM_JOB_ID}" "${GROUP_ID}" "$1"
}

cleanup() {
  set +e
  GPU=0 QWEN_PORT="${QWEN_A_PORT}" GEMMA_PORT="${GEMMA_A_PORT}" \
    QWEN_PID_FILE="$(pid_file qwen_a)" GEMMA_PID_FILE="$(pid_file gemma_a)" \
    "${SAMPLER_SERVER}" stop
  GPU=1 QWEN_PORT="${QWEN_B_PORT}" GEMMA_PORT="${GEMMA_B_PORT}" \
    QWEN_PID_FILE="$(pid_file qwen_b)" GEMMA_PID_FILE="$(pid_file gemma_b)" \
    "${SAMPLER_SERVER}" stop
  GPU=2 PORT="${QWEN36_PORT}" PID_FILE="$(pid_file qwen36)" \
    "${TRANSLATOR_SERVER}" stop
}
trap cleanup EXIT INT TERM

wait_health() {
  local name=$1 port=$2 launcher_pid=$3
  for i in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    fi
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
      echo "[ERROR] ${name} launcher exited before health check passed" >&2
      return 1
    fi
    sleep 1
  done
  echo "[ERROR] ${name} timed out" >&2
  return 1
}

echo "===== dual-sampler ambiguity group ${GROUP_ID} ====="
echo "job=${SLURM_JOB_ID} node=$(hostname) logical_workers=${LOGICAL_A},${LOGICAL_B}"
echo "num_concurrent_cases=${NUM_CONCURRENT_CASES} rows_per_worker=${ROWS_PER_WORKER}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

GPU=0 QWEN_PORT="${QWEN_A_PORT}" GEMMA_PORT="${GEMMA_A_PORT}" \
QWEN_MODEL="${QWEN38_MODEL}" GEMMA_MODEL="${GEMMA_MODEL}" \
PARALLEL_MODEL_START=1 \
QWEN_PID_FILE="$(pid_file qwen_a)" GEMMA_PID_FILE="$(pid_file gemma_a)" \
LOG_DIR="${RUN_LOG_DIR}/sampler_a" \
  "${SAMPLER_SERVER}" >"${RUN_LOG_DIR}/sampler_a.out" 2>"${RUN_LOG_DIR}/sampler_a.err" &
sampler_a_pid=$!

GPU=1 QWEN_PORT="${QWEN_B_PORT}" GEMMA_PORT="${GEMMA_B_PORT}" \
QWEN_MODEL="${QWEN38_MODEL}" GEMMA_MODEL="${GEMMA_MODEL}" \
PARALLEL_MODEL_START=1 \
QWEN_PID_FILE="$(pid_file qwen_b)" GEMMA_PID_FILE="$(pid_file gemma_b)" \
LOG_DIR="${RUN_LOG_DIR}/sampler_b" \
  "${SAMPLER_SERVER}" >"${RUN_LOG_DIR}/sampler_b.out" 2>"${RUN_LOG_DIR}/sampler_b.err" &
sampler_b_pid=$!

GPU=2 PORT="${QWEN36_PORT}" MODEL="${QWEN36_MODEL}" \
SERVED_MODEL_NAME=qwen36-translator PID_FILE="$(pid_file qwen36)" \
MAX_LEN=4096 MAX_NUM_SEQS=64 GPU_MEM_UTIL=0.85 \
  "${TRANSLATOR_SERVER}" >"${RUN_LOG_DIR}/translator.out" 2>"${RUN_LOG_DIR}/translator.err" &
translator_pid=$!

wait_health qwen_a "${QWEN_A_PORT}" "${sampler_a_pid}"
wait_health gemma_a "${GEMMA_A_PORT}" "${sampler_a_pid}"
wait_health qwen_b "${QWEN_B_PORT}" "${sampler_b_pid}"
wait_health gemma_b "${GEMMA_B_PORT}" "${sampler_b_pid}"
wait_health qwen36 "${QWEN36_PORT}" "${translator_pid}"

run_decoder() {
  local logical_id=$1 qwen_port=$2 gemma_port=$3
  local start_row=$(( ROW_OFFSET + logical_id * ROWS_PER_WORKER ))
  local remaining=$(( TOTAL_ROWS - logical_id * ROWS_PER_WORKER ))
  local max_rows=${ROWS_PER_WORKER}
  local output_task=$(( OUTPUT_TASK_OFFSET + logical_id ))
  local task_dir="${OUTPUT_ROOT}/task_$(printf '%02d' "${output_task}")"
  if (( remaining <= 0 )); then
    return 0
  fi
  if (( max_rows > remaining )); then
    max_rows=${remaining}
  fi
  mkdir -p "${task_dir}/per_utt" "${task_dir}/verbose"
  echo "[Decoder ${logical_id}] rows=${start_row}+${max_rows} output=${task_dir}"
  "${PYTHON}" "${DECODER}" \
    --input-tsv "${INPUT_TSV}" \
    --instruct-tokenizer-path "${QWEN36_MODEL}" \
    --instruct-api-base "http://127.0.0.1:${QWEN36_PORT}/v1" \
    --instruct-api-model qwen36-translator \
    --use-targeted-instruct-sampling \
    --targeted-sampler-api-base "http://127.0.0.1:${gemma_port}/v1" \
    --targeted-sampler-api-model gemma4-sampler \
    --targeted-sampler-tokenizer-path "${GEMMA_MODEL}" \
    --targeted-sampler2-api-base "http://127.0.0.1:${qwen_port}/v1" \
    --targeted-sampler2-api-model qwen38-sampler \
    --targeted-sampler2-tokenizer-path "${QWEN38_MODEL}" \
    --targeted-num-futures "${TARGETED_NUM_FUTURES}" \
    --future-source-window-chunks "${FUTURE_SRC_WINDOW}" \
    --min-voters-ratio "${MIN_VOTERS_RATIO}" \
    --row-idx "${start_row}" \
    --max-rows "${max_rows}" \
    --num-concurrent-cases "${NUM_CONCURRENT_CASES}" \
    --skip-existing --skip-existing-root "${OUTPUT_ROOT}" \
    --output-jsonl "${task_dir}/per_utt/_.jsonl" \
    --verbose --compact-verbose --verbose-dir "${task_dir}/verbose" \
    >"${task_dir}/run.log" 2>&1
  printf 'group=%s logical_worker=%s rows=%s completed=%s\n' \
    "${GROUP_ID}" "${logical_id}" "${max_rows}" "$(date -Iseconds)" >"${task_dir}/DONE.dual.txt"
}

set +e
run_decoder "${LOGICAL_A}" "${QWEN_A_PORT}" "${GEMMA_A_PORT}" &
decoder_a_pid=$!
run_decoder "${LOGICAL_B}" "${QWEN_B_PORT}" "${GEMMA_B_PORT}" &
decoder_b_pid=$!
wait "${decoder_a_pid}"; decoder_a_status=$?
wait "${decoder_b_pid}"; decoder_b_status=$?
set -e

if (( decoder_a_status != 0 || decoder_b_status != 0 )); then
  echo "[ERROR] decoder statuses: A=${decoder_a_status} B=${decoder_b_status}" >&2
  exit 1
fi
echo "[DONE] dual-sampler group ${GROUP_ID}"
