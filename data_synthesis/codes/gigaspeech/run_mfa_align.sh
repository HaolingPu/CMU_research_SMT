#!/usr/bin/env bash
#SBATCH --job-name=giga_mfa_align
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/mfa_align_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/mfa_align_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== MFA ALIGN START ====="

source ~/.bashrc
conda activate SMT

CORPUS_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus"
OUTPUT_DIR="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid"
TEMP_DIR="/data/user_data/haolingp/.cache/mfa_gigaspeech"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "$0")/../outputs/gigaspeech/slurm_logs" 2>/dev/null || true
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

# Clean previous MFA temp to avoid stale state
rm -rf "${TEMP_DIR}"
rm -rf ~/.local/share/mfa

mfa align \
  --clean \
  --final_clean \
  --single_speaker \
  --num_jobs 64 \
  --temporary_directory "${TEMP_DIR}" \
  --overwrite \
  --output_format long_textgrid \
  "${CORPUS_DIR}" \
  english_us_arpa \
  english_us_arpa \
  "${OUTPUT_DIR}"

echo "===== MFA ALIGN DONE ====="
echo "TextGrids written to: ${OUTPUT_DIR}"
