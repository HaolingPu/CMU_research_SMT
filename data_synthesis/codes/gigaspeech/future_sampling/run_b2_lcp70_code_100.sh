#!/usr/bin/env bash
# Baseline runner (100 cases): 4 tasks x 25 rows

#SBATCH --job-name=fut_b2_lcp70c
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
PYTHON_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_lcp70_code.py"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/llm_batch_output_b2_lcp70_code_simalign_100"

BASE_MODEL="/data/user_data/haolingp/models/Qwen3-4B-Base"
INSTRUCT_MODEL="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
SIMALIGN_MODEL="/data/user_data/haolingp/models/LaBSE"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS=4
MAX_ROWS=25
INSTRUCT_PORT=$((8100 + TASK_ID))

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: MANIFEST not found: $MANIFEST"
  exit 1
fi
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "ERROR: Script not found: $PYTHON_SCRIPT"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_ROOT")/slurm_logs"
mkdir -p "$(dirname "$OUTPUT_ROOT")"

echo "===== START TASK ${TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID:-N/A} node=$(hostname) time=$(date)"
export HF_HOME="/data/user_data/haolingp/hf_cache"

CUDA_VISIBLE_DEVICES=1 vllm serve "$INSTRUCT_MODEL" \
  --served-model-name qwen3-instruct \
  --port "$INSTRUCT_PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  > "${OUTPUT_ROOT}.task${TASK_ID}.serve.log" 2>&1 &
SERVE_PID=$!

for i in $(seq 1 120); do
  if curl -s "http://localhost:${INSTRUCT_PORT}/health" > /dev/null 2>&1; then
    echo "Instruct server ready"
    break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "ERROR: Instruct serve died. Check ${OUTPUT_ROOT}.task${TASK_ID}.serve.log"
    exit 1
  fi
  sleep 2
done

if ! curl -s "http://localhost:${INSTRUCT_PORT}/health" > /dev/null 2>&1; then
  echo "ERROR: Instruct server not ready in time"
  kill "$SERVE_PID" 2>/dev/null || true
  exit 1
fi


SIMALIGN_MODEL="$SIMALIGN_MODEL" CUDA_VISIBLE_DEVICES=0 python "$PYTHON_SCRIPT" \
  --input-tsv "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --task-id "$TASK_ID" \
  --num-tasks "$NUM_TASKS" \
  --max-rows "$MAX_ROWS" \
  --instruct-api-base "http://localhost:${INSTRUCT_PORT}/v1" \
  --parallel-utterances 16 \
  --future-sampling-batch-size 4 \
  --base-model-path "$BASE_MODEL" \
  --align-method simalign \
  --overwrite

kill "$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true

echo "===== DONE TASK ${TASK_ID} ====="
