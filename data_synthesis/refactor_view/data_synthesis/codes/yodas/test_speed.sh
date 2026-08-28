#!/usr/bin/env bash
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=general
#SBATCH --time=2-00:00:00

#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

##Optional but recommended:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

source ~/.bashrc


rm -rf /data/user_data/haolingp/outputs/mfa_output
rm -rf ~/.local/share/mfa


conda activate SMT


echo "(Already exists) mfa_textgrid_output"
mfa align \
  --clean \
  --final_clean \
  --num_jobs 64 \
  --temporary_directory /data/user_data/haolingp/.cache/mfa \
  --overwrite \
  --output_format long_textgrid \
  /data/group_data/li_lab/haolingp/yodas-granary/mfa_corpus/en000 \
  english_us_arpa \
  english_mfa \
  /data/user_data/haolingp/outputs/mfa_textgrids/en000