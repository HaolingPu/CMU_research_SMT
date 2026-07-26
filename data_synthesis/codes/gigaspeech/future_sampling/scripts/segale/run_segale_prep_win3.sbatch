#!/usr/bin/env bash
#SBATCH --job-name=segprep_win3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=02:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segprep_win3_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs/segprep_win3_%j.err

set -e
CONSENSUS_ROOT="${CONSENSUS_ROOT:?}"
OUT_ROOT="${OUT_ROOT:?}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/slurm_logs
/data/user_data/haolingp/conda_envs/evaluation/bin/python \
  /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/segale/prepare_segale_shards.py \
  --consensus-root "${CONSENSUS_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --num-docs 12500 \
  --num-shards 8 \
  --sys-id J_40k_qwenasr_sentsplit_win3
