#!/usr/bin/env bash
#SBATCH --job-name=J_retr5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --exclude=babel-t9-16,babel-o5-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/J_retr5_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/J_retr5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

FS_DIR="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
EVAL_ENV="/data/user_data/haolingp/conda_envs/evaluation"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM_QWEN="${EVAL_ENV}/bin/vllm"
VLLM_GEMMA="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${FS_DIR}/serve_instruct_gpu0.sh"
SERVE_GEMMA="${FS_DIR}/serve_gemma4_it_gpu1.sh"
DECODER="${FS_DIR}/consensus_decoding_retranslate.py"
INSTRUCT_TOK="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
GEMMA_TOK="/data/user_data/haolingp/models/gemma-4-E2B-it"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_smoke5_retranslate"
TASK_DIR="${OUTPUT_ROOT}/task_00"

PORT_INSTRUCT=8500
PORT_GEMMA=8501
PID_INSTRUCT="/tmp/J_retr5_qwen_${SLURM_JOB_ID}.pid"
PID_GEMMA="/tmp/J_retr5_gemma_${SLURM_JOB_ID}.pid"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${TASK_DIR}/per_utt" "${TASK_DIR}/verbose" "${OUTPUT_ROOT}/serve_logs" \
         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

stop_servers() {
  set +e
  PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" bash "${SERVE_INSTRUCT}" stop \
    > "${OUTPUT_ROOT}/serve_logs/stop_qwen.log" 2>&1
  PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" bash "${SERVE_GEMMA}" stop \
    > "${OUTPUT_ROOT}/serve_logs/stop_gemma.log" 2>&1
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

echo "===== J smoke5 retranslate START $(date) ====="
echo "node=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES}"
echo "out: ${TASK_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" VLLM_BIN="${VLLM_QWEN}" \
  bash "${SERVE_INSTRUCT}" > "${OUTPUT_ROOT}/serve_logs/serve_qwen.log" 2>&1 &
GPU=1 PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" VLLM_BIN="${VLLM_GEMMA}" \
  READY_TIMEOUT=900 bash "${SERVE_GEMMA}" > "${OUTPUT_ROOT}/serve_logs/serve_gemma.log" 2>&1 &

wait_health "qwen"  "http://127.0.0.1:${PORT_INSTRUCT}/health" 900
wait_health "gemma" "http://127.0.0.1:${PORT_GEMMA}/health"    900

echo "SERVERS_READY=$(date)"

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
  --row-idx 0 \
  --max-rows 5 \
  --num-concurrent-cases 5 \
  --soft-vote-top-k 20 \
  --soft-vote-min-p 0.1 \
  --soft-vote-threshold 0.8 \
  --min-voters-ratio 0.75 \
  --output-jsonl "${TASK_DIR}/per_utt/_.jsonl" \
  --verbose --verbose-dir "${TASK_DIR}/verbose" \
  2>&1 | tee "${TASK_DIR}/run.log"

echo "===== END $(date) ====="
echo "wrote: $(ls -1 ${TASK_DIR}/per_utt/*.json 2>/dev/null | wc -l) utts"
