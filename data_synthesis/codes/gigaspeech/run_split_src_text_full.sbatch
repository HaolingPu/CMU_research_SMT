#!/usr/bin/env bash
#SBATCH --job-name=qwenasr_sentsplit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=02:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwenasr_sentsplit_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/qwenasr_sentsplit_%j.err

set -e
source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/haolingp/conda_envs/segale
export PYTHONNOUSERSITE=1

mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs

IN="${IN:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_qwenasr-filtered.tsv}"
OUT="${OUT:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_qwenasr-filtered_sentsplit.tsv}"

echo "===== sentence-split START $(date) ====="
echo "IN  : ${IN}  ($(wc -l < "${IN}") rows)"
echo "OUT : ${OUT}"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/split_src_text_full_spacy.py \
  --input-tsv "${IN}" \
  --output-tsv "${OUT}" \
  --n-process 16 \
  --batch-size 256

echo "OUT rows: $(wc -l < "${OUT}")"
echo "===== sentence-split DONE $(date) ====="
