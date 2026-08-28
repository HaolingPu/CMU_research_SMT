#!/bin/bash
#SBATCH --job-name=simultst-whisper
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user_data/haolingp/data_synthesis/simul_tst_common/slurm_whisper_%j.log

set -e

BUILD=/data/user_data/haolingp/data_synthesis/simul_tst_common
WAV_DIR=/data/group_data/li_lab/siqiouya/datasets/must-c/v2.0/en-zh/data/tst-COMMON/wav
OUT=$BUILD/asr_out
mkdir -p "$OUT"

for f in "$WAV_DIR"/*.wav; do
    base=$(basename "$f" .wav)
    if [ -s "$OUT/$base.txt" ]; then
        echo "skip $base (done)"
        continue
    fi
    echo "=== $base $(date)"
    "$BUILD/whisper_venv/bin/whisper" "$f" \
        --model medium.en \
        --model_dir "$BUILD/models" \
        --output_dir "$OUT"
done
echo "ALL DONE $(date)"
