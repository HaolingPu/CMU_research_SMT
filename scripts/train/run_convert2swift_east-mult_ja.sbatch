#!/usr/bin/env bash
#SBATCH --job-name=conv2swift_east_ja
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_east_ja_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs/conv2swift_east_ja_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# ja EAST-latency2mult swift manifest builder.
# Reads sub-sentence-QE-filtered EAST trajectories at
#   /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/segale_qe/final_jsonl_east
# Writes audio chunks under
#   /data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_ja_EAST-latency2mult/
# Writes manifest at
#   /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_ja-EAST-latency2mult_origin.jsonl
# CPU-only; no GPU needed (just soundfile read/write).

set -e
mkdir -p /home/haolingp/CMU_research_SMT/scripts/train/slurm_logs

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation
export PYTHONNOUSERSITE=1

echo "===== CONVERT2SWIFT EAST ja START $(date) ====="
python /home/haolingp/CMU_research_SMT/scripts/train/convert2swift_east-mult_ja.py

OUT=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_ja-EAST-latency2mult_origin.jsonl
echo
echo "[FINAL]"
echo "  manifest    : ${OUT}"
echo "  lines       : $(wc -l < ${OUT})"
echo "  audio clips : /data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_ja_EAST-latency2mult/"
echo "===== CONVERT2SWIFT EAST ja DONE $(date) ====="
