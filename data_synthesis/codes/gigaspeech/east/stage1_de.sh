#!/usr/bin/env bash
# ============================================================
# EAST  de  -- Stage 1: post_process + multi_trajectory
#
# Runs after llm_de.sh array finishes (no GPU strictly needed).
#   1) post_process_llm_output_gigaspeech.py  (merge single-word chunks)
#   2) multi_trajectory_gigaspeech.py         (align to src_trajectory; no MFA)
#
# Submit:
#   jid=$(sbatch --parsable east/llm_de.sh)
#   sbatch --dependency=afterok:$jid east/stage1_de.sh
# ============================================================
#SBATCH --job-name=east_de_s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --time=8:00:00
#SBATCH --exclude=babel-p9-28,babel-s5-32,babel-m5-32,babel-n9-32,babel-o5-16,babel-o5-24
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/stage1_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/stage1_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== de Stage 1 START $(date) ====="

source ~/.bashrc
conda activate SMT

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech

# --- Step 1: Post-process LLM output (merge single-word chunks) -----------
# auto-picks target join separator from JSON's target_lang (de -> space)
echo "--- post_process ---"
python "${CODE}/post_process_llm_output_gigaspeech.py" \
  --input-dir   "${BASE}/llm_output_raw" \
  --output-dir  "${BASE}/llm_output_merged" \
  --overwrite

# --- Step 2: Build streaming trajectory dataset --------------------------
# Uses manifest's src_trajectory (already 960ms grid) -- no MFA.
echo "--- multi_trajectory ---"
python "${CODE}/multi_trajectory_gigaspeech.py" \
  --llm-dir     "${BASE}/llm_output_merged" \
  --output-dir  "${BASE}/streaming_EAST_dataset" \
  --chunk-ms    960 \
  --overwrite

# --- Quick stats ---------------------------------------------------------
echo "--- counts ---"
RAW=$(find "${BASE}/llm_output_raw" -maxdepth 3 -name '*.json' | wc -l)
MRG=$(find "${BASE}/llm_output_merged" -maxdepth 3 -name '*.json' | wc -l)
STR=$(find "${BASE}/streaming_EAST_dataset" -maxdepth 3 -name '*.json' | wc -l)
echo "llm_output_raw         : ${RAW}"
echo "llm_output_merged      : ${MRG}"
echo "streaming_EAST_dataset : ${STR}"

echo "===== de Stage 1 DONE $(date) ====="
echo "Next (manual, in metricx env):"
echo "  conda activate metricx"
echo "  python ${CODE}/convert_metricx_gigaspeech.py --stream_dir ${BASE}/streaming_EAST_dataset --output ${BASE}/metricx_input.jsonl"
