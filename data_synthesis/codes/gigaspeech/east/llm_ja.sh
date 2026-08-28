#!/usr/bin/env bash
# ============================================================
# EAST  ja  -- Stage 0: LLM trajectory generation (en -> ja)
#
# 8-way array job. Each task processes rows where (idx % 8 == task_id),
# capped at --max-rows per task (5000 -> 40k total utts).
#
# Chain:  sbatch --dependency=afterok:<this_jid> stage1_ja.sh
# ============================================================
#SBATCH --job-name=east_ja_llm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=1-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== ja LLM stage START $(date) (task=${SLURM_ARRAY_TASK_ID}) ====="

source ~/.bashrc

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja
MANIFEST=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/llm_output_gigaspeech_trajectory.py

NUM_TASKS=8
TP=1
BATCH_SIZE=4096
MAX_ROWS=12500  # per worker; 8 workers * 12500 = 100k total. Resume mode skips
                # already-existing utts (the 40k from prior run), so only the new
                # 60k get computed.

mkdir -p "${BASE}/llm_output_raw" "${BASE}/slurm_logs"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

conda activate vllm

python "${CODE}" \
  --input-tsv     "${MANIFEST}" \
  --output-root   "${BASE}/llm_output_raw" \
  --target-lang   ja \
  --task-id       "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks     "${NUM_TASKS}" \
  --tp            "${TP}" \
  --batch-size    "${BATCH_SIZE}" \
  --max-rows      "${MAX_ROWS}"

echo "===== ja LLM stage DONE $(date) (task=${SLURM_ARRAY_TASK_ID}) ====="
