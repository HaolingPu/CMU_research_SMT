#!/usr/bin/env bash
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --time=2-00:00:00

#SBATCH --array=0-7          #提交 8 个任务
#SBATCH -o slurm_logs/%A_%a.out
#SBATCH -e slurm_logs/%A_%a.err

##Optional but recommended:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

echo "===== START WORKER TASK $SLURM_ARRAY_TASK_ID ====="
echo "=============================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "=============================================="

# 显示GPU信息
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=============================================="



source ~/.bashrc
conda activate vllm

cd /data/user_data/haolingp/codes/

# ---------------------------------------------
# 每个任务独立运行 Qwen3-30B 推理（TP=2）
# ---------------------------------------------
echo "Launching local vLLM engine (TP=1) for task $SLURM_ARRAY_TASK_ID"
    
python llm_output_Salami.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-tasks 8 \
    --tp 1 \
    --batch-size 64 \
    --num-en 1 \
    --num-parquets all \
    --num-samples all \

echo "===== TASK $SLURM_ARRAY_TASK_ID DONE ====="

EXIT_CODE=$?

# ============================================================
# 结束信息
# ============================================================
echo "=============================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Task $SLURM_ARRAY_TASK_ID completed successfully"
else
    echo "✗ Task $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE"
fi
echo "=============================================="

# 显示最终GPU状态
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

exit $EXIT_CODE