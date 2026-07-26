#!/usr/bin/env bash
#SBATCH --job-name=fix_win3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/fix_win3_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/fix_win3_%j.err

set -e
IN_DIR="${IN_DIR:?}"
OUT_DIR="${OUT_DIR:?}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs
/data/user_data/haolingp/conda_envs/evaluation/bin/python \
  /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/fix_period_and_length_filter.py \
  --in-dir "${IN_DIR}" \
  --out-dir "${OUT_DIR}" \
  --min-ratio 0.7 \
  --max-ratio 1.5
