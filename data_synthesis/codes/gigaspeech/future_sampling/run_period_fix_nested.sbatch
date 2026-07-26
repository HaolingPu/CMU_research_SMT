#!/usr/bin/env bash
#SBATCH --job-name=periodfix_w3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/periodfix_w3_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs/periodfix_w3_%j.err

# Step-1 period-fix on the decode trajectory (NEW dASR has the 。X artifact),
# producing a mirrored task_*/per_utt tree that the SEGALE+QE pipeline runs on.
set -e
IN_ROOT="${IN_ROOT:?IN_ROOT required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/slurm_logs
/home/haolingp/miniconda3/bin/python \
  /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/period_fix_traj_nested.py \
  --in-root "${IN_ROOT}" \
  --out-root "${OUT_ROOT}"
