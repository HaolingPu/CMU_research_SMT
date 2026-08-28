#!/usr/bin/env bash
#SBATCH --job-name=prespropose_smoke500
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/prespropose_smoke500_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/prespropose_smoke500_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Present-proposes / futures-verify smoke: 500 utterances, rows 0-499 of the
# flagship old-ASR frozen-reference TSV. Single track; paired analysis is
# against the J_40k per-utt jsons on disk for the same utt_ids.
# Gate INVERTED vs flagship: candidates = present top-6, futures verify by
# majority support (min-voters-ratio 0.75 -> 15/20).
# See wiki: experiments/2026-07-present-propose-gate.
set -e

FS_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
EVAL_ENV="/data/user_data/haolingp/conda_envs/evaluation"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM_QWEN="${EVAL_ENV}/bin/vllm"
VLLM_GEMMA="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${FS_DIR}/serve_instruct_gpu0.sh"
SERVE_GEMMA="${FS_DIR}/serve_gemma4_it_gpu1.sh"
DECODER="${FS_DIR}/experimental/consensus_decoding_present_propose.py"
INSTRUCT_TOK="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
GEMMA_TOK="/data/user_data/haolingp/models/gemma-4-E2B-it"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/present_propose_smoke500"
TASK_DIR="${OUTPUT_ROOT}/task_00"
LOG_DIR="${OUTPUT_ROOT}/serve_logs"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${TASK_DIR}/per_utt" "${TASK_DIR}/verbose" "${LOG_DIR}"

PORT_INSTRUCT=8630
PORT_GEMMA=8631
PID_INSTRUCT="/tmp/prespropose_smoke_qwen_${SLURM_JOB_ID}.pid"
PID_GEMMA="/tmp/prespropose_smoke_gemma_${SLURM_JOB_ID}.pid"

stop_servers() {
  set +e
  PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_qwen.log" 2>&1
  PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" \
    bash "${SERVE_GEMMA}" stop > "${LOG_DIR}/stop_gemma.log" 2>&1
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

echo "===== present-propose smoke500 START $(date) job=${SLURM_JOB_ID} node=$(hostname) ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" VLLM_BIN="${VLLM_QWEN}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_qwen.log" 2>&1 &

GPU=1 PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" VLLM_BIN="${VLLM_GEMMA}" \
  READY_TIMEOUT=900 bash "${SERVE_GEMMA}" > "${LOG_DIR}/serve_gemma.log" 2>&1 &

wait_health "qwen"  "http://127.0.0.1:${PORT_INSTRUCT}/health" 900
wait_health "gemma" "http://127.0.0.1:${PORT_GEMMA}/health"    900

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
  --future-source-window-chunks 1 \
  --candidate-top-k 6 \
  --min-voters-ratio 0.75 \
  --row-idx 0 \
  --max-rows 500 \
  --num-concurrent-cases 12 \
  --skip-existing \
  --output-jsonl "${TASK_DIR}/per_utt/_.jsonl" \
  --verbose --verbose-dir "${TASK_DIR}/verbose" \
  2>&1 | tee -a "${TASK_DIR}/run.log"

CMD_STATUS="${PIPESTATUS[0]}"
ACTUAL=$(ls -1 "${TASK_DIR}/per_utt"/*.json 2>/dev/null | wc -l)
echo "smoke: wrote ${ACTUAL}/500 jsons, decoder exit=${CMD_STATUS}"
exit "${CMD_STATUS}"
