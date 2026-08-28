#!/usr/bin/env bash
#SBATCH --job-name=qwen35_consensus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_consensus_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwen35_consensus_%A_%a.err

# Consensus decoding with the LOCAL Qwen3.5-122B future sampler.
# The sampler runs as a SEPARATE 4x L40S server (serve_qwen35_122b.sbatch);
# this job only serves the 1-GPU instruct translator and calls that server.
#
# REQUIRED: pass the running sampler endpoint, e.g.
#   sbatch --export=ALL,LABEL=qwen35_test_1,TOTAL_ROWS=10,SAMPLER_API_BASE=http://babel-s5-32:8300/v1 run_qwen35_general.sbatch

set -e

LABEL="${LABEL:-qwen35_test_1}"

bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/qwen35/run_qwen35_common.sh "${LABEL}"
