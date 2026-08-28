#!/usr/bin/env bash
#SBATCH --job-name=hibiki_final_ja_13k_preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=120G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-23
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/haolingp/data_synthesis/codes/gigaspeech/hibiki/output/hibiki_ja_13k_v2/slurm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/codes/gigaspeech/hibiki/output/hibiki_ja_13k_v2/slurm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

HIBIKI_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/hibiki"
CODE_DIR="$HIBIKI_DIR/code"
PYTHON_SCRIPT="$CODE_DIR/contextual_alignment_final.py"
MODEL_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref_subsentence_ref_ja.tsv"
OUTPUT_DIR="$HIBIKI_DIR/output/hibiki_ja_13k_v2"

MAX_ROWS=13000
NUM_TASKS=24
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TP=1
GPU_MEM_UTIL=0.90

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_TSV" ]]; then
  echo "Missing input TSV: $INPUT_TSV"
  exit 1
fi

echo "job_id=${SLURM_JOB_ID:-N/A} host=$(hostname) time=$(date)"
echo "script=$PYTHON_SCRIPT"
echo "model=$MODEL_PATH"
echo "input=$INPUT_TSV"
echo "target_lang=ja"
echo "partition=preempt"
echo "qos=preempt_qos"
echo "max_rows=$MAX_ROWS"
echo "task_id=$TASK_ID"
echo "num_tasks=$NUM_TASKS"
echo "output_dir=$OUTPUT_DIR"

python "$PYTHON_SCRIPT" \
  --input-tsv "$INPUT_TSV" \
  --target-lang ja \
  --tokenizer-path "$MODEL_PATH" \
  --base-model-path "$MODEL_PATH" \
  --max-rows "$MAX_ROWS" \
  --task-id "$TASK_ID" \
  --num-tasks "$NUM_TASKS" \
  --tp "$TP" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --output-dir "$OUTPUT_DIR"

echo "done time=$(date)"
echo "pretty_json_dir=$OUTPUT_DIR"
