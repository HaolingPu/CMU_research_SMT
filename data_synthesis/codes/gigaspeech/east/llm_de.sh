#!/usr/bin/env bash
# ============================================================
# EAST  de  -- Stage 0: LLM trajectory generation (en -> de)
#
# 24-way array job on PREEMPT.  Each task processes rows where
#   (idx % 24 == task_id), capped at --max-rows per task.
#   24 workers * 4167 rows = 100008 utts (~100k target).
#
# Resume mode: skips already-existing utts, so preemption-requeues
# pick up where they left off.
# ============================================================
#SBATCH --job-name=east_de_llm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-p9-28,babel-s5-32,babel-m5-32,babel-n9-32,babel-o5-16,babel-o5-24
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== de LLM stage START $(date) (task=${SLURM_ARRAY_TASK_ID}) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de
MANIFEST=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/llm_output_gigaspeech_trajectory.py

NUM_TASKS=24
TP=1
BATCH_SIZE=4096
MAX_ROWS=4167   # per worker; 24 workers * 4167 = 100008 utts (~100k)

mkdir -p "${BASE}/llm_output_raw" "${BASE}/slurm_logs"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

conda activate vllm

python "${CODE}" \
  --input-tsv     "${MANIFEST}" \
  --output-root   "${BASE}/llm_output_raw" \
  --target-lang   de \
  --task-id       "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks     "${NUM_TASKS}" \
  --tp            "${TP}" \
  --batch-size    "${BATCH_SIZE}" \
  --max-rows      "${MAX_ROWS}"

echo "===== de LLM stage DONE $(date) (task=${SLURM_ARRAY_TASK_ID}) ====="
