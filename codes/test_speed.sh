#!/usr/bin/env bash
#SBATCH --job-name=test_speed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --time=2:00:00

##SBATCH --array=0-3             # DP=4 → 提交 4 个任务
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

##Optional but recommended:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu
source ~/.bashrc
# 1. Activate conda environment
conda activate vllm
echo "[OK] Activated vllm"


start=$(date +%s)

python llm_output_vllm.py \
    --task-id 0 \
    --num-tasks 1 \
    --num-parquets 1 \
    --num-en 1 \
    --tp 1 \
    --batch-size 64

end=$(date +%s)
echo "Total time: $((end - start)) seconds"
