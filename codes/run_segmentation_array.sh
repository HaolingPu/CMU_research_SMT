#!/usr/bin/env bash
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=128G
#SBATCH --partition=general
#SBATCH --time=2-00:00:00

#SBATCH --array=0-3             # DP=4 → 提交 4 个任务
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

##Optional but recommended:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

echo "===== START WORKER TASK $SLURM_ARRAY_TASK_ID ====="

source ~/.bashrc
conda activate vllm

cd /data/user_data/haolingp/codes/

# ---------------------------------------------
# 每个任务独立运行 Qwen3-30B 推理（TP=2）
# ---------------------------------------------
echo "Launching local vLLM engine (TP=2) for task $SLURM_ARRAY_TASK_ID"

python llm_output_vllm.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-tasks 4 \
    --tp 2

echo "===== TASK $SLURM_ARRAY_TASK_ID DONE ====="
