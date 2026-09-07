#!/usr/bin/env bash
#SBATCH --job-name=amb_single_ab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=96G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --time=03:00:00
#SBATCH --exclude=babel-o9-24,babel-q9-32,babel-t5-28,babel-v9-28
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Decode ONE TSV row several times under different decoder variants with the
# production models loaded once (Qwen3.8 + Gemma samplers on GPU 0, Qwen3.6 probe
# on GPU 1). Each variant writes to ${OUTPUT_ROOT}/<label>/task_00/. Intended for
# A/B reading of a single case in the trajectory viewer; not a benchmark.
#
# Env:
#   ROW_IDX        TSV row index to decode (default 10 = AUD0000000003_1015)
#   ROW_IDXS       optional space-separated row indices; still sequential on 2 GPUs
#   OUTPUT_ROOT    required
#   VARIANTS       ';'-separated "label=extra decoder args" (default below)
#   SAMPLER_SEED   deterministic seed shared by all variants (default 1015)

set -euo pipefail

REPO="/home/haolingp/CMU_research_SMT"
FS="${REPO}/data_synthesis/codes/gigaspeech/future_sampling"
ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${ENV}/bin/python"
SAMPLER_SERVER="${FS}/scripts/qwen38/serve_qwen38_gemma_colocated.sh"
TRANSLATOR_SERVER="${FS}/scripts/qwen36/serve_qwen36_35b.sh"
DECODER="${DECODER:-${FS}/consensus_decoding_token_id_level_instruct.py}"
CASE_GUARD="${CASE_GUARD:-${FS}/experimental/pilot_case_guard.py}"
QWEN38_MODEL="${QWEN38_MODEL:-/data/user_data/haolingp/models/Qwen3.8-27B-FP8}"
GEMMA_MODEL="${GEMMA_MODEL:-/data/user_data/haolingp/models/gemma-4-E2B-it}"
QWEN36_MODEL="${QWEN36_MODEL:-/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
ROW_IDX="${ROW_IDX:-10}"
ROW_IDXS="${ROW_IDXS:-${ROW_IDX}}"
if [[ -n "${PILOT_CASE_COUNT:-}" ]]; then
  ROW_IDXS="$(seq "${SLURM_ARRAY_TASK_ID:-0}" "${PILOT_GROUPS:-4}" "$((PILOT_CASE_COUNT - 1))" | tr '\n' ' ')"
fi
SAMPLER_SEED="${SAMPLER_SEED:-1015}"
VARIANTS="${VARIANTS:-baseline=;joinfix=--future-join-mode sentence-aware;joinfix_sentend=--future-join-mode sentence-aware --sentence-end-completion;full=--future-join-mode sentence-aware --sentence-end-completion --future-source-window-mode until-closed}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
FUTURE_SRC_WINDOW="${FUTURE_SRC_WINDOW:-1}"

PORT_BASE=$(( 8700 + 4 * (SLURM_JOB_ID % 50) ))
QWEN38_PORT=${PORT_BASE}; GEMMA_PORT=$(( PORT_BASE + 1 )); QWEN36_PORT=$(( PORT_BASE + 2 ))
QWEN38_PID_FILE="/tmp/ab_q38_${SLURM_JOB_ID}.pid"; GEMMA_PID_FILE="/tmp/ab_gemma_${SLURM_JOB_ID}.pid"; QWEN36_PID_FILE="/tmp/ab_q36_${SLURM_JOB_ID}.pid"
LOG_DIR="${OUTPUT_ROOT}/serve_logs/${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
export HF_HOME="/data/user_data/haolingp/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"; export TOKENIZERS_PARALLELISM=false
mkdir -p "${LOG_DIR}"

cleanup() {
  set +e
  GPU=0 QWEN_PORT="${QWEN38_PORT}" GEMMA_PORT="${GEMMA_PORT}" QWEN_PID_FILE="${QWEN38_PID_FILE}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" "${SAMPLER_SERVER}" stop
  GPU=1 PORT="${QWEN36_PORT}" PID_FILE="${QWEN36_PID_FILE}" "${TRANSLATOR_SERVER}" stop
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

wait_health() {
  local name=$1 port=$2 pid=$3
  for i in $(seq 1 1200); do
    curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && { echo "[READY] ${name} after ${i}s"; return 0; }
    kill -0 "${pid}" 2>/dev/null || { echo "[ERROR] ${name} launcher exited" >&2; return 1; }
    sleep 1
  done
  echo "[ERROR] ${name} timed out" >&2; return 1
}

echo "===== small-case A/B job=${SLURM_JOB_ID} node=$(hostname) rows=${ROW_IDXS} seed=${SAMPLER_SEED} ====="
echo "commit=$(git -C "${REPO}" rev-parse HEAD)"
echo "decoder=${DECODER}"
sha256sum "${DECODER}"
GPU=0 QWEN_PORT="${QWEN38_PORT}" GEMMA_PORT="${GEMMA_PORT}" QWEN_MODEL="${QWEN38_MODEL}" GEMMA_MODEL="${GEMMA_MODEL}" \
  QWEN_PID_FILE="${QWEN38_PID_FILE}" GEMMA_PID_FILE="${GEMMA_PID_FILE}" LOG_DIR="${LOG_DIR}/samplers" \
  "${SAMPLER_SERVER}" >"${LOG_DIR}/samplers.out" 2>"${LOG_DIR}/samplers.err" &
sampler_pid=$!
GPU=1 PORT="${QWEN36_PORT}" MODEL="${QWEN36_MODEL}" SERVED_MODEL_NAME=qwen36-translator PID_FILE="${QWEN36_PID_FILE}" \
  MAX_LEN=4096 MAX_NUM_SEQS=64 GPU_MEM_UTIL=0.85 "${TRANSLATOR_SERVER}" >"${LOG_DIR}/translator.out" 2>"${LOG_DIR}/translator.err" &
translator_pid=$!
wait_health qwen38 "${QWEN38_PORT}" "${sampler_pid}"
wait_health gemma "${GEMMA_PORT}" "${sampler_pid}"
wait_health qwen36 "${QWEN36_PORT}" "${translator_pid}"

IFS=';' read -r -a VARIANT_LIST <<< "${VARIANTS}"
read -r -a ROW_LIST <<< "${ROW_IDXS}"
FAILED_ROWS=()
for ROW_IDX in "${ROW_LIST[@]}"; do
[[ "${ROW_IDX}" =~ ^[0-9]+$ ]] || { echo "Invalid row index: ${ROW_IDX}" >&2; exit 2; }
for spec in "${VARIANT_LIST[@]}"; do
  label="${spec%%=*}"; extra="${spec#*=}"
  if [[ ${#ROW_LIST[@]} -gt 1 ]]; then
    TASK_DIR="${OUTPUT_ROOT}/row_${ROW_IDX}/${label}/task_00"
  else
    TASK_DIR="${OUTPUT_ROOT}/${label}/task_00"
  fi
  [[ "${label}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid label: ${label}" >&2; exit 2; }
  guard_args=(--input-tsv "${INPUT_TSV}" --row-idx "${ROW_IDX}" --task-dir "${TASK_DIR}"
              "--extra-args=${extra}" --seed "${SAMPLER_SEED}" --window "${FUTURE_SRC_WINDOW}")
  if "${PYTHON}" "${CASE_GUARD}" validate "${guard_args[@]}"; then
    echo "[SKIP] completed row=${ROW_IDX} variant=${label}"
    continue
  fi
  "${PYTHON}" "${CASE_GUARD}" prepare "${guard_args[@]}"
  mkdir -p "${TASK_DIR}/per_utt" "${TASK_DIR}/verbose"
  echo "===== variant ${label}: extra args [${extra}] ====="
  start_ts=$(date +%s)
  # shellcheck disable=SC2086
  if ! "${PYTHON}" "${DECODER}" \
    --input-tsv "${INPUT_TSV}" \
    --instruct-tokenizer-path "${QWEN36_MODEL}" \
    --instruct-api-base "http://127.0.0.1:${QWEN36_PORT}/v1" \
    --instruct-api-model qwen36-translator \
    --use-targeted-instruct-sampling \
    --targeted-sampler-api-base "http://127.0.0.1:${GEMMA_PORT}/v1" \
    --targeted-sampler-api-model gemma4-sampler \
    --targeted-sampler-tokenizer-path "${GEMMA_MODEL}" \
    --targeted-sampler2-api-base "http://127.0.0.1:${QWEN38_PORT}/v1" \
    --targeted-sampler2-api-model qwen38-sampler \
    --targeted-sampler2-tokenizer-path "${QWEN38_MODEL}" \
    --targeted-num-futures "${TARGETED_NUM_FUTURES}" \
    --future-source-window-chunks "${FUTURE_SRC_WINDOW}" \
    --min-voters-ratio "${MIN_VOTERS_RATIO}" \
    --targeted-sampler-seed "${SAMPLER_SEED}" \
    --targeted-fail-on-api-error \
    --row-idx "${ROW_IDX}" --max-rows 1 --num-concurrent-cases 1 \
    --skip-existing \
    --output-jsonl "${TASK_DIR}/per_utt/_.jsonl" \
    --verbose --verbose-dir "${TASK_DIR}/verbose" \
    ${extra} 2>&1 | tee "${TASK_DIR}/run_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}.log"; then
    # Isolate the failure: record it, keep going with the remaining rows, and
    # exit non-zero at the end so dependent report jobs do not run on a partial set.
    echo "[FAILED] row=${ROW_IDX} variant=${label}; no completion marker written" >&2
    printf 'variant=%s row=%s job=%s restart=%s log=%s\n' "${label}" "${ROW_IDX}" "${SLURM_JOB_ID}" "${SLURM_RESTART_COUNT:-0}" "${TASK_DIR}/run_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}.log" >>"${TASK_DIR}/FAILED.txt"
    FAILED_ROWS+=("row=${ROW_IDX} variant=${label}")
    continue
  fi
  "${PYTHON}" "${CASE_GUARD}" validate "${guard_args[@]}"
  rm -f "${TASK_DIR}/FAILED.txt"
  printf 'variant=%s row=%s seed=%s extra=%s elapsed_seconds=%s\n' "${label}" "${ROW_IDX}" "${SAMPLER_SEED}" "${extra}" "$(( $(date +%s) - start_ts ))" >"${TASK_DIR}/DONE.txt"
done
done
if [[ ${#FAILED_ROWS[@]} -gt 0 ]]; then
  echo "===== ${#FAILED_ROWS[@]} case(s) FAILED: ${FAILED_ROWS[*]} =====" >&2
  exit 1
fi
echo "===== all variants done ====="
