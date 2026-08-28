#!/usr/bin/env bash
#SBATCH --job-name=pa_40k
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/prefix_alignment/slurm_logs/pa_40k_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/prefix_alignment/slurm_logs/pa_40k_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Self-contained PA shard: starts a per-node vLLM on the same GPU as BERTScore
# (Qwen3-30B-A3B-FP8 ~33GB at 0.65 mem-util, BERTScore xlm-roberta-large ~3GB,
#  fits comfortably on L40S 48GB).
#
# Required env vars (set by submit_pa_40k.sh):
#   OUT_ROOT          base output dir
# Optional:
#   MT_API_MODEL      qwen3-instruct
#   MODEL_PATH        /data/user_data/.../Qwen3-30B-A3B-Instruct-2507-FP8
#   TOKENIZER_PATH    same as MODEL_PATH (vLLM serves both)
#   INPUT_TSV         (default: train_xl ... subsentence_ref.tsv)
#   TOTAL_ROWS        (default 40000)
#   NUM_SHARDS        (default 8 -- must match #SBATCH --array)
#   ROW_OFFSET        (default 0)
#   NUM_CONCURRENT    (default 8)
#   SCORER            (default bertscore)
#   TARGET_UNIT       (default char)
#   LCP_MODE          (default char)
#   VLLM_PORT         (default 8100)
#   VLLM_MEM_UTIL     (default 0.65 -- leave ~17GB for BERTScore + workspace)

set -e

OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
MT_API_MODEL="${MT_API_MODEL:-qwen3-instruct}"
MODEL_PATH="${MODEL_PATH:-/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
INPUT_TSV="${INPUT_TSV:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv}"
TOTAL_ROWS="${TOTAL_ROWS:-40000}"
NUM_SHARDS="${NUM_SHARDS:-8}"
ROW_OFFSET="${ROW_OFFSET:-0}"
NUM_CONCURRENT="${NUM_CONCURRENT:-32}"
SCORER="${SCORER:-bertscore}"
TARGET_UNIT="${TARGET_UNIT:-char}"
LCP_MODE="${LCP_MODE:-char}"
TARGET_LANG="${TARGET_LANG:-Chinese}"
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.8}"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SIZE=$(( (TOTAL_ROWS + NUM_SHARDS - 1) / NUM_SHARDS ))
ROW_IDX=$(( ROW_OFFSET + SHARD_ID * SHARD_SIZE ))
MAX_ROWS=${SHARD_SIZE}

SHARD_TAG=$(printf "task_%02d" "${SHARD_ID}")
OUT_DIR="${OUT_ROOT}/raw/${SHARD_TAG}"
OUT_JSONL="${OUT_ROOT}/raw/${SHARD_TAG}.jsonl"
SLURM_LOGS="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/prefix_alignment/slurm_logs"
VLLM_LOG="${SLURM_LOGS}/pa_40k_vllm_${SLURM_ARRAY_JOB_ID:-$$}_${SHARD_ID}.log"

mkdir -p "${OUT_DIR}" "${SLURM_LOGS}"

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/gemma4
export HF_HOME="/data/user_data/haolingp/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

PA_ROOT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/prefix_alignment"

echo "===== PA shard ${SHARD_ID} START $(date) ====="
echo "OUT_DIR        : ${OUT_DIR}"
echo "ROW_IDX..MAX   : ${ROW_IDX} .. $((ROW_IDX + MAX_ROWS - 1))"
echo "node           : ${SLURMD_NODENAME:-$(hostname)}  gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "MODEL_PATH     : ${MODEL_PATH}"
echo "VLLM_PORT      : ${VLLM_PORT}  (mem-util=${VLLM_MEM_UTIL})"

# --- Start local vLLM ---------------------------------------------------------
echo "[vllm] starting on localhost:${VLLM_PORT} ..."
vllm serve "${MODEL_PATH}" \
  --served-model-name "${MT_API_MODEL}" \
  --dtype auto \
  --port "${VLLM_PORT}" \
  --host 127.0.0.1 \
  --max-model-len 2048 \
  --gpu-memory-utilization "${VLLM_MEM_UTIL}" \
  --tensor-parallel-size 1 \
  --max-logprobs 100 \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "[vllm] pid=${VLLM_PID}, log=${VLLM_LOG}"

cleanup() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] killing pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 3
    kill -9 "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[vllm] polling /health (max 600s) ..."
READY=0
for i in $(seq 1 600); do
  if curl -fs "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "[vllm] ready after ~${i}s"
    READY=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] ERROR: process died during startup; tail of log:"
    tail -n 60 "${VLLM_LOG}"
    exit 1
  fi
  sleep 1
done
if [[ "${READY}" -ne 1 ]]; then
  echo "[vllm] ERROR: not ready in 600s; tail of log:"
  tail -n 60 "${VLLM_LOG}"
  exit 1
fi

# --- Run prefix_alignment ----------------------------------------------------
echo "[pa] launching prefix_alignment.py"
python "${PA_ROOT}/prefix_alignment.py" \
  --input-tsv "${INPUT_TSV}" \
  --id-key id \
  --src-trajectory-key src_trajectory \
  --target-key llm_reference_text \
  --mt-api-base "http://localhost:${VLLM_PORT}" \
  --mt-api-model "${MT_API_MODEL}" \
  --mt-tokenizer-path "${TOKENIZER_PATH}" \
  --target-lang "${TARGET_LANG}" \
  --target-unit "${TARGET_UNIT}" \
  --lcp-mode "${LCP_MODE}" \
  --scorer "${SCORER}" \
  --bertscore-device cuda:0 \
  --row-idx "${ROW_IDX}" \
  --max-rows "${MAX_ROWS}" \
  --num-concurrent-cases "${NUM_CONCURRENT}" \
  --overwrite \
  --output-jsonl "${OUT_JSONL}" \
  --output-dir "${OUT_DIR}"

echo "[pa] DONE; files written: $(ls ${OUT_DIR}/*.json 2>/dev/null | wc -l)"
echo "===== PA shard ${SHARD_ID} END $(date) ====="
