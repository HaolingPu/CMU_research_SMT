#!/usr/bin/env bash
#SBATCH --job-name=deepseek_consensus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=08:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/deepseek_consensus_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/deepseek_consensus_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Run consensus decoding with the DeepSeek future sampler (OpenAI-compatible
# /chat/completions backend). Same instruction as the GPT sampler.
#
# 8-task array, 1 L40S GPU per task (only the instruct vLLM server needs a GPU;
# DeepSeek runs via API). 8 tasks -> 8 GPUs total, partitioned across the row range.
#
# Sanity / speed test (default):
#   sbatch --export=ALL,LABEL=deepseek_test10,TOTAL_ROWS=10 run_deepseek_general.sbatch
#
# Quality + speed at 100 rows, with reasoning enabled:
#   sbatch --export=ALL,LABEL=deepseek_test100,TOTAL_ROWS=100,CHAT_EXTRA_BODY='{"thinking": {"type": "enabled"}}' run_deepseek_general.sbatch
#
# Override anything else via --export=ALL,KEY=VALUE,...
#   CHAT_SAMPLER_MODEL, CHAT_EXTRA_BODY, NUM_FUTURES, FUTURE_TOKENS,
#   CANDIDATE_TOP_K, MIN_P, TOP_P, NUM_CONCURRENT_CASES, TARGET_LANG, etc.

LABEL="${LABEL:-deepseek_test10}"

bash /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/deepseek/run_deepseek_common.sh "${LABEL}"
