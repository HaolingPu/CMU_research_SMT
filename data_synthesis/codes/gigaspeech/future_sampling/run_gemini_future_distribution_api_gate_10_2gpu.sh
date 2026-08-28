#!/usr/bin/env bash
#SBATCH --job-name=gem_fd_api10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=220G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-1%2
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem_fd_api10_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs/gem_fd_api10_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm


TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS=2
MAX_ROWS=5
PORT="$((8100 + TASK_ID))"
MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
PY_SCRIPT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/gemini/llm_future_sampling_thinking_policy_gemini_future_distribution.py"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini_future_distribution_api_gate_10"
SLURM_LOG_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/slurm_logs"
SERVE_MODEL="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
SERVE_NAME="qwen3-instruct"
GATE_API_BASE="http://127.0.0.1:${PORT}/v1"
SERVE_LOG="${SLURM_LOG_DIR}/gate_serve_${SLURM_JOB_ID:-manual}_${TASK_ID}.log"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}"
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: python script not found: ${PY_SCRIPT}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${SLURM_LOG_DIR}"

export HF_HOME="/data/user_data/haolingp/hf_cache"

echo "===== START 10-CASE TASK ${TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID:-N/A} node=$(hostname) time=$(date)"
echo "output_root=${OUTPUT_ROOT}"
echo "task_id=${TASK_ID} num_tasks=${NUM_TASKS} max_rows=${MAX_ROWS}"
echo "thinking_model=gemini-3-flash-preview fallback_model=gemini-3.1-pro-preview"
echo "gate_server_model=${SERVE_MODEL} gate_server_name=${SERVE_NAME} port=${PORT}"
echo "gate_api_base=${GATE_API_BASE}"
echo "gate_serve_log=${SERVE_LOG}"
echo "routing=api_gate_logprobs -> direct_pro | final_completion=pro | simalign=disabled"
echo "probe_avg_entropy_threshold=0.65 probe_js_threshold=0.10"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

CUDA_VISIBLE_DEVICES=0 vllm serve "${SERVE_MODEL}" \
  --served-model-name "${SERVE_NAME}" \
  --dtype auto \
  --port "${PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 > "${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
trap 'kill ${SERVE_PID} >/dev/null 2>&1 || true' EXIT
echo "gate_serve_pid=${SERVE_PID}"

python - <<PY2
import time, urllib.request, sys
url = "${GATE_API_BASE}/models"
for _ in range(600):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                print("Gate server ready:", url)
                sys.exit(0)
    except Exception:
        time.sleep(1)
print("ERROR: gate server did not become ready:", url)
sys.exit(1)
PY2

python - <<PY3
import sys
import time
from openai import OpenAI

base_url = "${GATE_API_BASE}"
client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=30.0)
last_error = None
for attempt in range(1, 61):
    try:
        resp = client.completions.create(
            model="${SERVE_NAME}",
            prompt="Hello",
            max_tokens=3,
            temperature=0.0,
            logprobs=5,
            stop=["\n"],
        )
        choice = resp.choices[0]
        print(
            "Gate completions probe OK:",
            {
                "attempt": attempt,
                "text": getattr(choice, "text", ""),
                "finish_reason": getattr(choice, "finish_reason", None),
                "has_logprobs": getattr(choice, "logprobs", None) is not None,
            },
        )
        sys.exit(0)
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        time.sleep(1)

print("ERROR: gate completions probe failed:", last_error)
sys.exit(1)
PY3

CUDA_VISIBLE_DEVICES=1 python "${PY_SCRIPT}" \
  --input-tsv "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --task-id "${TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --max-rows "${MAX_ROWS}" \
  --base-model-path "/data/user_data/haolingp/models/Qwen3-4B-Base" \
  --thinking-model-name "gemini-3-flash-preview" \
  --fallback-model-name "gemini-3.1-pro-preview" \
  --final-completion-model-name "gemini-3.1-pro-preview" \
  --thinking-reasoning-effort "high" \
  --fallback-reasoning-effort "low" \
  --num-futures 10 \
  --future-tokens 12 \
  --sample-temperature 1.0 \
  --thinking-temperature 0.1 \
  --thinking-max-tokens 4096 \
  --fallback-max-tokens 4096 \
  --gate-api-base "${GATE_API_BASE}" \
  --gate-api-model-name "${SERVE_NAME}" \
  --gate-api-key EMPTY \
  --overwrite

echo "===== DONE 10-CASE TASK ${TASK_ID} ====="
