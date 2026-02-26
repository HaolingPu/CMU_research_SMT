#!/usr/bin/env bash
# ============================================================
# Future Sampling -- LLM future source sampling + majority vote
#
# Input  : TSV with src_text_full, src_trajectory
# Output : ${OUT_ROOT}/{utt_id}.json (source_future_sampling, target_future_sampling, actions)
#
# Trial : NUM_TASKS=1 MAX_ROWS=50 sbatch --array=0 llm.sh
# Full  : sbatch llm.sh   (8 tasks, --array=0-7, 8 GPUs in parallel)
# ============================================================
#SBATCH --job-name=giga_fut
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
CODE="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling.py"
OUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling/llm_output"
MODEL_PATH="${MODEL_PATH:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
NUM_TASKS="${NUM_TASKS:-8}"
TP="${TP:-1}"
MAX_ROWS="${MAX_ROWS:-}"

# Future sampling hyper-parameters
NUM_CANDIDATES="${NUM_CANDIDATES:-10}"
FUTURE_TOKENS="${FUTURE_TOKENS:-30}"
SAMPLE_TEMP="${SAMPLE_TEMP:-0.8}"
TAU="${TAU:-0.7}"

# MAX_ROWS = per-task limit (Python --max-rows). So 8 tasks x 50 = 400 outputs.
# Optional: set GLOBAL_MAX_ROWS (e.g. 50) to cap total rows; per-task = ceil(GLOBAL_MAX_ROWS/NUM_TASKS).
EXTRA_ARGS=()
if [[ -n "${GLOBAL_MAX_ROWS:-}" ]]; then
  PER_TASK_ROWS=$(python3 -c "import math; print(math.ceil(int(${GLOBAL_MAX_ROWS})/${NUM_TASKS}))")
  EXTRA_ARGS+=(--max-rows "${PER_TASK_ROWS}")
elif [[ -n "${MAX_ROWS}" ]]; then
  EXTRA_ARGS+=(--max-rows "${MAX_ROWS}")
fi
[[ -n "${OVERWRITE:-}" ]] && [[ "${OVERWRITE}" != "0" ]] && EXTRA_ARGS+=(--overwrite)

mkdir -p "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling/slurm_logs"

echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

python "${CODE}" \
  --input-tsv "${MANIFEST}" \
  --output-root "${OUT_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --tp "${TP}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --future-tokens "${FUTURE_TOKENS}" \
  --sample-temperature "${SAMPLE_TEMP}" \
  --tau "${TAU}" \
  "${EXTRA_ARGS[@]}"

echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
