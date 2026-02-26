#!/usr/bin/env bash
# ============================================================
# Future Sampling (Final) -- Dual-model pipeline
#
# GPU layout (per node, 2 GPUs):
#   GPU 0: Base model  (Qwen3-30B-A3B-FP8) in-process llm.generate()
#   GPU 1: Instruct model (vllm serve on port 8100)
#
# Trial: NUM_TASKS=1 MAX_ROWS=5 sbatch --array=0 llm_final.sh
# Full:  sbatch llm_final.sh
# ============================================================
#SBATCH --job-name=giga_fut_final
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
CODE="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_final.py"
OUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/llm_batch_output"

BASE_MODEL="${BASE_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-FP8}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
INSTRUCT_PORT=8100
NUM_TASKS="${NUM_TASKS:-8}"
MAX_ROWS="${MAX_ROWS:-}"

NUM_CANDIDATES="${NUM_CANDIDATES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-30}"
SAMPLE_TEMP="${SAMPLE_TEMP:-0.8}"

EXTRA_ARGS=()
if [[ -n "${MAX_ROWS}" ]]; then
  EXTRA_ARGS+=(--max-rows "${MAX_ROWS}")
fi
[[ -n "${OVERWRITE:-}" ]] && [[ "${OVERWRITE}" != "0" ]] && EXTRA_ARGS+=(--overwrite)

mkdir -p "$(dirname "${OUT_ROOT}")"/slurm_logs
mkdir -p "${OUT_ROOT}"

echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ---- Step 1: Start instruct serve on GPU 1 ----
echo "[Step 1] Starting instruct serve on GPU 1 ..."
export HF_HOME="/data/user_data/haolingp/hf_cache"

CUDA_VISIBLE_DEVICES=1 vllm serve "${INSTRUCT_MODEL}" \
  --served-model-name qwen3-instruct \
  --dtype auto \
  --port "${INSTRUCT_PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 &

SERVE_PID=$!
echo "  serve PID: ${SERVE_PID}"

# Wait for server to be ready
for i in $(seq 1 300); do
  if curl -s "http://localhost:${INSTRUCT_PORT}/health" > /dev/null 2>&1; then
    echo "  Instruct server ready (took ~${i}s)"
    break
  fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "ERROR: Instruct serve process died."
    exit 1
  fi
  sleep 1
done

if ! curl -s "http://localhost:${INSTRUCT_PORT}/health" > /dev/null 2>&1; then
  echo "ERROR: Instruct server not ready after 300s"
  kill "${SERVE_PID}" 2>/dev/null
  exit 1
fi

# ---- Step 2: Run pipeline on GPU 0 ----
echo "[Step 2] Running pipeline on GPU 0 ..."

CUDA_VISIBLE_DEVICES=0 python "${CODE}" \
  --input-tsv "${MANIFEST}" \
  --output-root "${OUT_ROOT}" \
  --base-model-path "${BASE_MODEL}" \
  --instruct-api-base "http://localhost:${INSTRUCT_PORT}/v1" \
  --instruct-model-name "qwen3-instruct" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --tp 1 \
  --num-candidates "${NUM_CANDIDATES}" \
  --future-tokens "${FUTURE_TOKENS}" \
  --sample-temperature "${SAMPLE_TEMP}" \
  --verbose \
  "${EXTRA_ARGS[@]}"

# ---- Cleanup ----
echo "[Cleanup] Stopping instruct serve ..."
kill "${SERVE_PID}" 2>/dev/null
wait "${SERVE_PID}" 2>/dev/null || true

echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
