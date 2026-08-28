#!/usr/bin/env bash
#SBATCH --job-name=la_verbose_3cases
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=0-02:00:00
#SBATCH --exclude=babel-p9-28,babel-s5-32,babel-m5-32,babel-n9-32
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_verbose_3cases_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/la_verbose_3cases_%j.err

set -e

# Run LA-N=2 verbose step-by-step traces for 3 utterances at fixed seg=1, 2, 3.
# Output (trace_segN_<utt>.txt + result_segN_<utt>.json) goes next to
# la_verbose_driver.py in the LA codes directory.

LA_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement"
FS_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling"
GEMMA4_ENV="/data/user_data/haolingp/conda_envs/gemma4"
PYTHON="${GEMMA4_ENV}/bin/python"
VLLM="${GEMMA4_ENV}/bin/vllm"

SERVE_INSTRUCT="${FS_DIR}/serve_instruct_gpu0.sh"
DRIVER="${LA_DIR}/la_verbose_driver.py"
TOKENIZER="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"

PORT=8390
PID_FILE="/tmp/la_verbose_instruct_${SLURM_JOB_ID:-manual}.pid"
LOG_DIR="${LA_DIR}/output_test"
mkdir -p "${LOG_DIR}"

export HF_HOME="/data/user_data/haolingp/hf_cache"
export HF_HUB_CACHE="/data/user_data/haolingp/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/user_data/haolingp/hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"
export PATH="${GEMMA4_ENV}/bin:${PATH}"

stop_servers() {
  set +e
  PORT="${PORT}" PID_FILE="${PID_FILE}" \
    bash "${SERVE_INSTRUCT}" stop > "${LOG_DIR}/stop_instruct_verbose.log" 2>&1
}
trap stop_servers EXIT

wait_health() {
  url="$1"
  for i in $(seq 1 300); do
    if curl -s "${url}" > /dev/null 2>&1; then
      echo "[READY] instruct after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] instruct not ready: ${url}"
  return 1
}

echo "===== LA verbose 3-case run (job=${SLURM_JOB_ID:-manual} node=$(hostname)) ====="

PORT="${PORT}" PID_FILE="${PID_FILE}" VLLM_BIN="${VLLM}" \
  bash "${SERVE_INSTRUCT}" > "${LOG_DIR}/serve_instruct_verbose.log" 2>&1 &

wait_health "http://127.0.0.1:${PORT}/health"

# Three cases: same utterances as the JSON-replay we already have, one per seg.
"${PYTHON}" "${DRIVER}" \
  --input-tsv "${INPUT_TSV}" \
  --mt-api-base "http://127.0.0.1:${PORT}/v1" \
  --mt-api-model qwen3-instruct \
  --mt-tokenizer-path "${TOKENIZER}" \
  --la-n 2 --lcp-mode char \
  --max-new-tokens 1024 --target-lang Chinese \
  --out-dir "${LA_DIR}" \
  --utt-seg AUD0000000003_1011:1 \
  --utt-seg AUD0000000003_100:2 \
  --utt-seg AUD0000000003_0:3

echo "===== done ====="
