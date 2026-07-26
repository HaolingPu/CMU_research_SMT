#!/usr/bin/env bash
#SBATCH --job-name=J_prod_200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/J_prod_200_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/J_prod_200_%j.err
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
DECODER="${FS_DIR}/consensus_decoding_token_id_level_instruct.py"
INSTRUCT_TOK="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
GEMMA_TOK="/data/user_data/haolingp/models/gemma-4-E2B-it"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_preempt_200"
LOG_DIR="${OUTPUT_ROOT}/serve_logs"

# Use unique ports based on $SLURM_JOB_ID to avoid colliding with other
# jobs on the same node (preempt is shared).
PORT_INSTRUCT=$(( 8500 + (SLURM_JOB_ID % 100) ))
PORT_GEMMA=$(( PORT_INSTRUCT + 1 ))
PID_INSTRUCT="/tmp/J200_qwen_${SLURM_JOB_ID}.pid"
PID_GEMMA="/tmp/J200_gemma_${SLURM_JOB_ID}.pid"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${OUTPUT_ROOT}/per_utt" "${OUTPUT_ROOT}/verbose" "${LOG_DIR}" \
         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

stop_servers() {
  set +e
  PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_qwen.log" 2>&1
  PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" \
    bash "${SERVE_GEMMA}" stop > "${LOG_DIR}/stop_gemma.log" 2>&1
}
trap stop_servers EXIT

wait_health() {
  name="$1"; url="$2"; timeout_s="${3:-600}"
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

echo "===== J production sanity 200-case (preempt) ====="
echo "job=${SLURM_JOB_ID} node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ports: qwen=${PORT_INSTRUCT}  gemma=${PORT_GEMMA}"
echo "out:   ${OUTPUT_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

START_TS=$(date +%s)
echo "START=$(date)" > "${OUTPUT_ROOT}/timing.log"

PORT="${PORT_INSTRUCT}" PID_FILE="${PID_INSTRUCT}" VLLM_BIN="${VLLM_QWEN}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_qwen.log" 2>&1 &

GPU=1 PORT="${PORT_GEMMA}" PID_FILE="${PID_GEMMA}" VLLM_BIN="${VLLM_GEMMA}" \
  READY_TIMEOUT=900 bash "${SERVE_GEMMA}" > "${LOG_DIR}/serve_gemma.log" 2>&1 &

wait_health "qwen"  "http://127.0.0.1:${PORT_INSTRUCT}/health" 900
wait_health "gemma" "http://127.0.0.1:${PORT_GEMMA}/health"    900

echo "SERVERS_READY=$(date)" >> "${OUTPUT_ROOT}/timing.log"

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
  --max-rows 200 \
  --num-concurrent-cases 16 \
  --output-jsonl "${OUTPUT_ROOT}/per_utt/_.jsonl" \
  --verbose --verbose-dir "${OUTPUT_ROOT}/verbose" \
  2>&1 | tee -a "${OUTPUT_ROOT}/run.log"

CMD_STATUS="${PIPESTATUS[0]}"
echo "END=$(date)" >> "${OUTPUT_ROOT}/timing.log"
ELAPSED=$(( $(date +%s) - START_TS ))
echo "elapsed=${ELAPSED}s" >> "${OUTPUT_ROOT}/timing.log"

ACTUAL=$(ls -1 "${OUTPUT_ROOT}/per_utt"/*.json 2>/dev/null | wc -l)
echo "wrote ${ACTUAL}/200 jsons in ${ELAPSED}s ($(awk "BEGIN{printf \"%.2f\",${ELAPSED}/${ACTUAL}}")s/case)"

if [[ "${CMD_STATUS}" != "0" ]]; then
  echo "[ERROR] decoder exit code ${CMD_STATUS}"
  exit "${CMD_STATUS}"
fi
echo "[DONE] J production 200-case"
