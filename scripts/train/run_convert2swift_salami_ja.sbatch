#!/usr/bin/env bash
#SBATCH --job-name=conv2swift_salami_ja
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --exclude=babel-t9-16,babel-n5-20
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_salami_ja_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_salami_ja_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
mkdir -p /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation
export PYTHONNOUSERSITE=1

echo "===== CONVERT2SWIFT SALAMI ja START $(date) ====="
python /home/haolingp/CMU_research_SMT/scripts/train/convert2swift_salami_ja.py

OUT=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_ja-Simul-MuST-C_origin.jsonl
echo
echo "[FINAL]"
echo "  manifest    : ${OUT}"
echo "  lines       : $(wc -l < ${OUT})"
echo "  audio clips : /data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_ja_Simul-MuST-C/"
echo "===== CONVERT2SWIFT SALAMI ja DONE $(date) ====="
