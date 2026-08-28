#!/usr/bin/env bash
# ============================================================
# EAST  ja  -- Stage 1: post_process + multi_trajectory
#
# Runs after llm_ja.sh array finishes (no GPU needed).
#   1) post_process_llm_output_gigaspeech.py  (merge single-word chunks)
#   2) multi_trajectory_gigaspeech.py         (align to src_trajectory; no MFA)
#
# Submit:
#   jid=$(sbatch --parsable east/llm_ja.sh)
#   sbatch --dependency=afterok:$jid east/stage1_ja.sh
# ============================================================
#SBATCH --job-name=east_ja_s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=8:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/stage1_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/stage1_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== ja Stage 1 START $(date) ====="

source ~/.bashrc
conda activate SMT

BASE=/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech

# --- Step 1: Post-process LLM output (merge single-word chunks) -----------
# auto-picks target join separator from JSON's target_lang (ja -> no-space)
echo "--- post_process ---"
# resume mode: skip already-merged utts (cheap re-run after extending llm_output_raw)
python "${CODE}/post_process_llm_output_gigaspeech.py" \
  --input-dir   "${BASE}/llm_output_raw" \
  --output-dir  "${BASE}/llm_output_merged"

# --- Step 2: Build streaming trajectory dataset --------------------------
# Uses manifest's src_trajectory (already 960ms grid) -- no MFA.
echo "--- multi_trajectory ---"
# resume mode: skip already-built streaming utts; only new ones get processed
python "${CODE}/multi_trajectory_gigaspeech.py" \
  --llm-dir     "${BASE}/llm_output_merged" \
  --output-dir  "${BASE}/streaming_EAST_dataset" \
  --chunk-ms    960

# --- Quick stats ---------------------------------------------------------
echo "--- counts ---"
RAW=$(find "${BASE}/llm_output_raw" -maxdepth 3 -name '*.json' | wc -l)
MRG=$(find "${BASE}/llm_output_merged" -maxdepth 3 -name '*.json' | wc -l)
STR=$(find "${BASE}/streaming_EAST_dataset" -maxdepth 3 -name '*.json' | wc -l)
echo "llm_output_raw         : ${RAW}"
echo "llm_output_merged      : ${MRG}"
echo "streaming_EAST_dataset : ${STR}"

echo "===== ja Stage 1 DONE $(date) ====="
echo "Next (manual, in metricx env):"
echo "  conda activate metricx"
echo "  python ${CODE}/convert_metricx_gigaspeech.py --stream_dir ${BASE}/streaming_EAST_dataset --output ${BASE}/metricx_input.jsonl"
