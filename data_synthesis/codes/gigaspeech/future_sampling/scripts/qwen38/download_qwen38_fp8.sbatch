#!/usr/bin/env bash
#SBATCH --job-name=q38_download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/haolingp/slurm_runs/qwen38_download_%j.out
#SBATCH --error=/home/haolingp/slurm_runs/qwen38_download_%j.err

set -euo pipefail

PYTHON="${PYTHON:-/data/user_data/haolingp/conda_envs/evaluation/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}"
MODEL_DIR="${MODEL_DIR:-/data/user_data/haolingp/models/Qwen3.8-27B-FP8}"

export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

echo "Downloading ${MODEL_ID} to ${MODEL_DIR}"
"${PYTHON}" - "${MODEL_ID}" "${MODEL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, model_dir = sys.argv[1:]
snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    token=True if __import__("os").environ.get("HF_TOKEN") else None,
)
PY

echo "Downloaded files:"
du -sh "${MODEL_DIR}"
test -f "${MODEL_DIR}/config.json"

