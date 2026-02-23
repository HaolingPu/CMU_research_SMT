#!/usr/bin/env bash
#SBATCH --job-name=giga_dev_llm_8gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=220G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=07:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/dev_bench_8gpu/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/dev_bench_8gpu/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START PIPELINE ====="

source ~/.bashrc
MANIFEST="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/dev_case_robust_asr-filtered.tsv"
CODE="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/llm_output_gigaspeech_trajectory.py"
OUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/dev_bench_8gpu/llm_output_raw"
LOG_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/dev_bench_8gpu/slurm_logs"
NUM_TASKS="${NUM_TASKS:-8}"
TP="${TP:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OVERWRITE="${OVERWRITE:-1}"        # 1 => add --overwrite
MAX_ROWS="${MAX_ROWS:-}"           # optional, per worker
mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"
echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) time=$(date)"
echo "manifest=${MANIFEST}"
echo "num_tasks=${NUM_TASKS} tp=${TP} batch_size=${BATCH_SIZE}"
echo "output_root=${OUT_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv



conda activate vllm
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=2
CMD=(
  python -u "${CODE}"
  --input-tsv "${MANIFEST}"
  --output-root "${OUT_ROOT}"
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --num-tasks "${NUM_TASKS}"
  --tp "${TP}"
  --batch-size "${BATCH_SIZE}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ -n "${MAX_ROWS}" ]]; then
  CMD+=(--max-rows "${MAX_ROWS}")
fi
START_TS=$(date +%s)
"${CMD[@]}"
END_TS=$(date +%s)
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "elapsed_sec=$((END_TS-START_TS))"