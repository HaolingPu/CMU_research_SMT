#!/usr/bin/env bash
# ============================================================
# SALAMI LLM raw generation — Japanese target.
#
# 24-shard preempt array; each shard processes up to MAX_ROWS / 24 rows
# of the GigaSpeech English ASR TSV through Qwen3-30B-FP8 with the
# Japanese SALAMI prompt.
#
# Output: ${BASE}/llm_output_raw/{utt_id}.json
#         (target_lang="Japanese" written into each record)
#
# Submit:
#   sbatch llm_ja.sh                  # full 24-shard preempt run
# ============================================================
#SBATCH --job-name=giga_llm_salami_ja
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --array=0-23%24
#SBATCH --time=1-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc
conda activate vllm

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=2

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja
mkdir -p ${BASE}/slurm_logs ${BASE}/llm_output_raw

# Cap total input rows; per-shard cap is MAX_ROWS / NUM_TASKS.
# 100k input / 24 shards ≈ 4167 rows per shard.
MAX_ROWS_TOTAL=100000
NUM_TASKS=24
PER_SHARD_CAP=$(( (MAX_ROWS_TOTAL + NUM_TASKS - 1) / NUM_TASKS ))
echo "MAX_ROWS_TOTAL=${MAX_ROWS_TOTAL}, NUM_TASKS=${NUM_TASKS}, PER_SHARD_CAP=${PER_SHARD_CAP}"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/salami/llm_output_salami.py \
  --input-tsv /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
  --output-root ${BASE}/llm_output_raw \
  --task-id ${SLURM_ARRAY_TASK_ID} \
  --num-tasks ${NUM_TASKS} \
  --tp 1 \
  --batch-size 2048 \
  --max-rows ${PER_SHARD_CAP} \
  --target-lang Japanese

echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
