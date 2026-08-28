#!/usr/bin/env bash
# Trial run for OpenAI-hosted thinking policy:
#   - GPU 0: shared base model + simalign + controller process
#   - Thinking model: OpenAI Responses API (for example gpt-5.4)
#
# Submit:
#   sbatch /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_thinking_policy_openai_10.sh
#
# This script loads OPENAI_API_KEY from ~/.bashrc.

#SBATCH --job-name=think_oai_10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=220G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/think_oai_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/slurm_logs/think_oai_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

source ~/.bashrc
conda activate vllm

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_openai_gpt54_10utt

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  echo "Expected to load it from ~/.bashrc, but it is missing."
  exit 1
fi

echo "OPENAI_API_KEY loaded from ~/.bashrc"

if [[ ! -f /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv ]]; then
  echo "ERROR: MANIFEST not found: /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
  exit 1
fi

if [[ ! -f /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy_openai.py ]]; then
  echo "ERROR: Script not found: /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy_openai.py"
  exit 1
fi

export HF_HOME="/data/user_data/haolingp/hf_cache"

echo "===== START OPENAI THINKING TRIAL ====="
echo "job_id=${SLURM_JOB_ID:-N/A} node=$(hostname) time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

SIMALIGN_MODEL="/data/user_data/haolingp/models/LaBSE" CUDA_VISIBLE_DEVICES=0 python "/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy_openai.py" \
  --input-tsv "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv" \
  --output-root "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/thinking_policy_openai_gpt54_10utt" \
  --task-id 0 \
  --num-tasks 1 \
  --max-rows 10 \
  --base-model-path "/data/user_data/haolingp/models/Qwen3-4B-Base" \
  --thinking-api-base "https://api.openai.com/v1" \
  --thinking-model-name "gpt-5.4" \
  --thinking-reasoning-effort "medium" \
  --thinking-reasoning-summary "auto" \
  --thinking-verbosity "low" \
  --parallel-utterances 10 \
  --future-sampling-batch-size 4 \
  --future-sampling-batch-wait 0.05 \
  --num-futures 5 \
  --future-tokens 10 \
  --sample-temperature 1.0 \
  --thinking-temperature 0.1 \
  --thinking-max-tokens 4096 \
  --overwrite

echo "===== DONE OPENAI THINKING TRIAL ====="
