#!/usr/bin/env bash
# ============================================================
# EAST  Stage 1 — LLM output post-processing
#
# Input  : ${BASE}/llm_output_raw          (raw LLM JSON files)
# Output : ${BASE}/llm_output_merged_fixed
#          ${BASE}/good_train_xl_east_mfa.jsonl  (→ feed to Stage 2)
#
# Submit : sbatch stage1_llm.sh
# Then   : sbatch stage2_streaming.sh   (after this job finishes)
# ============================================================
#SBATCH --job-name=east_s1_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/stage1_llm_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/stage1_llm_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== EAST Stage 1 — START $(date) ====="

source ~/.bashrc
conda activate SMT

BASE=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east
CODE=/data/user_data/haolingp/data_synthesis/codes/gigaspeech

mkdir -p ${BASE}/slurm_logs

# ── Step 0: Fix LLM raw (方案 A：只修不筛) ─────────────────────
# Restore EN punct from manifest, sync ZH punct; do NOT filter by zh/en punct.
# Only token-mismatch is dropped; punct mismatch is kept for data volume.
rm -rf ${BASE}/llm_output_raw_fixed

python ${CODE}/fix_llm_raw.py \
  --in_dir         ${BASE}/llm_output_raw \
  --out_dir        ${BASE}/llm_output_raw_fixed \
  --out_good_jsonl ${BASE}/good_train_xl_east_fixed.jsonl \
  --sync_zh_punct \
  --zh_punct_allow_insert

# ── Step 1: Post-process (merge one-word chunks, etc.) ───────
rm -rf ${BASE}/llm_output_merged_fixed

python ${CODE}/post_process_llm_output_gigaspeech.py \
  --input-dir  ${BASE}/llm_output_raw_fixed \
  --output-dir ${BASE}/llm_output_merged_fixed \
  --overwrite

# ── Step 2: Find bad JSONs (MFA alignment check) ─────────────
python ${CODE}/find_bad_json_gigaspeech.py \
  --llm-dir    ${BASE}/llm_output_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --corpus-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus \
  --good-jsonl ${BASE}/good_train_xl_east_mfa.jsonl \
  --bad-jsonl  ${BASE}/bad_train_xl_east_mfa.jsonl

echo "===== EAST Stage 1 — DONE $(date) ====="
echo "Next: sbatch stage2_streaming.sh"
