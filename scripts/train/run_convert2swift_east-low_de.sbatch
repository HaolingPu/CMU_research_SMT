#!/usr/bin/env bash
#SBATCH --job-name=conv2swift_east-low_de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_east-low_de_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_east-low_de_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
mkdir -p /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation
export PYTHONNOUSERSITE=1

echo "===== CONVERT2SWIFT EAST-low de START $(date) ====="
python /home/haolingp/CMU_research_SMT/scripts/train/convert2swift_east-low_de.py

OUT=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_de-EAST-low_origin.jsonl
echo
echo "[FINAL]"
echo "  manifest    : ${OUT}"
echo "  lines       : $(wc -l < ${OUT})"
echo "  audio clips : /data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_de_EAST-low/"
echo "===== CONVERT2SWIFT EAST-low de DONE $(date) ====="
