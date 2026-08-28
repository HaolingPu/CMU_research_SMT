#!/usr/bin/env bash
# _run_anchor_40k_common.sh
# ---------------------------------------------------------------------------
# Shared worker body for the anchor-and-veto 40k production decode (variant A
# veto params from the anchor_smoke500 sweep: min-p 0.05, top-k 5, voters 1.0;
# see wiki 2026-07-consensus-register-forensics and analyze_anchor_smoke.py).
# Structure mirrors _run_J_40k_common.sh; exec'd by the thin SLURM array
# wrappers run_anchor_40k_general.sbatch / run_anchor_40k_preempt.sbatch.
#
# Input is the OLD-ASR frozen-reference TSV (the flagship top5-axis5 data,
# same as the smoke runs) — NOT the qwenasr TSV.
# ---------------------------------------------------------------------------
set -e

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

FS_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
EVAL_ENV="/data/user_data/haolingp/conda_envs/evaluation"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM_QWEN="${EVAL_ENV}/bin/vllm"
VLLM_GEMMA="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${FS_DIR}/serve_instruct_gpu0.sh"
SERVE_GEMMA="${FS_DIR}/serve_gemma4_it_gpu1.sh"
DECODER="${FS_DIR}/consensus_decoding_anchor.py"
INSTRUCT_TOK="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
GEMMA_TOK="/data/user_data/haolingp/models/gemma-4-E2B-it"

INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/anchor_40k}"

TOTAL_ROWS="${TOTAL_ROWS:-40000}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_TASKS="${NUM_TASKS:-16}"

ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))
START_ROW=$(( ROW_OFFSET + TASK_ID * ROWS_PER_TASK ))

TASK_DIR="${OUTPUT_ROOT}/task_$(printf '%02d' "${TASK_ID}")"
LOG_DIR="${OUTPUT_ROOT}/serve_logs"
DONE_FILE="${TASK_DIR}/DONE.txt"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${TASK_DIR}/per_utt" "${TASK_DIR}/verbose" "${LOG_DIR}" \
         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

if [[ -f "${DONE_FILE}" ]]; then
  echo "[SKIP] ${DONE_FILE} exists — task ${TASK_ID} already complete"
  exit 0
fi

# Unique ports per task; base 8700 so co-located J40k tasks (base 8500) and
# anchor smoke ports (8600-8631) never collide.
PORT_INSTRUCT=$(( 8700 + 2 * TASK_ID ))
PORT_GEMMA=$(( PORT_INSTRUCT + 1 ))
PID_INSTRUCT="/tmp/anchor40k_qwen_${SLURM_JOB_ID}.pid"
PID_GEMMA="/tmp/anchor40k_gemma_${SLURM_JOB_ID}.pid"

stop_servers() {
  set +e
  PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_qwen_${TASK_ID}.log" 2>&1
  PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" \
    bash "${SERVE_GEMMA}" stop > "${LOG_DIR}/stop_gemma_${TASK_ID}.log" 2>&1
}
trap stop_servers EXIT

wait_health() {
  name="$1"; url="$2"; timeout_s="${3:-900}"
  for i in $(seq 1 "${timeout_s}"); do
    if curl -s "${url}" > /dev/null 2>&1; then
      echo "[READY] ${name} after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] ${name} not ready: ${url}"
  return 1
}

echo "===== anchor 40k task ${TASK_ID} START $(date) ====="
echo "job=${SLURM_JOB_ID} node=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES}"
echo "rows: start=${START_ROW} count=${ROWS_PER_TASK}  ports: qwen=${PORT_INSTRUCT} gemma=${PORT_GEMMA}"
echo "input: ${INPUT_TSV}"
echo "out:   ${TASK_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

START_TS=$(date +%s)
echo "START=$(date)" > "${TASK_DIR}/timing.log"

PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" VLLM_BIN="${VLLM_QWEN}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_qwen_${TASK_ID}.log" 2>&1 &

GPU=1 PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" VLLM_BIN="${VLLM_GEMMA}" \
  READY_TIMEOUT=900 bash "${SERVE_GEMMA}" > "${LOG_DIR}/serve_gemma_${TASK_ID}.log" 2>&1 &

wait_health "qwen"  "http://127.0.0.1:${PORT_INSTRUCT}/health" 900
wait_health "gemma" "http://127.0.0.1:${PORT_GEMMA}/health"    900

echo "SERVERS_READY=$(date)" >> "${TASK_DIR}/timing.log"

"${PYTHON}" "${DECODER}" \
  --input-tsv "${INPUT_TSV}" \
  --instruct-tokenizer-path "${INSTRUCT_TOK}" \
  --instruct-api-base "http://127.0.0.1:${PORT_INSTRUCT}/v1" \
  --instruct-api-model "qwen3-instruct" \
  --use-targeted-instruct-sampling \
  --targeted-sampler-api-base "http://127.0.0.1:${PORT_GEMMA}/v1" \
  --targeted-sampler-api-model "gemma4-it" \
  --targeted-sampler-tokenizer-path "${GEMMA_TOK}" \
  --targeted-sampler2-api-base "http://127.0.0.1:${PORT_INSTRUCT}/v1" \
  --targeted-sampler2-api-model "qwen3-instruct" \
  --targeted-sampler2-tokenizer-path "${INSTRUCT_TOK}" \
  --targeted-num-futures 20 \
  --future-source-window-chunks "${FUTURE_SRC_WINDOW:-1}" \
  --anchor-max-tokens 24 \
  --veto-min-p 0.05 \
  --veto-top-k 5 \
  --veto-min-voters-ratio 1.0 \
  --row-idx "${START_ROW}" \
  --max-rows "${ROWS_PER_TASK}" \
  --num-concurrent-cases 16 \
  --skip-existing \
  --output-jsonl "${TASK_DIR}/per_utt/_.jsonl" \
  --verbose --verbose-dir "${TASK_DIR}/verbose" \
  2>&1 | tee -a "${TASK_DIR}/run.log"

CMD_STATUS="${PIPESTATUS[0]}"
echo "END=$(date)" >> "${TASK_DIR}/timing.log"
ELAPSED=$(( $(date +%s) - START_TS ))
echo "elapsed=${ELAPSED}s" >> "${TASK_DIR}/timing.log"

ACTUAL=$(ls -1 "${TASK_DIR}/per_utt"/*.json 2>/dev/null | wc -l)
echo "task ${TASK_ID}: wrote ${ACTUAL}/${ROWS_PER_TASK} jsons in ${ELAPSED}s"

if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] decoder exit code ${CMD_STATUS}"
  exit "${CMD_STATUS}"
fi

echo "task ${TASK_ID} done $(date)" > "${DONE_FILE}"
echo "[DONE] anchor 40k task ${TASK_ID}"
