#!/usr/bin/env bash
# ============================================================
# SALAMI ja — Stage 1: adapt raw → post_process → streaming dataset
#
# Input  : ${BASE}/llm_output_raw            (salami raw JSON)
# Output : ${BASE}/llm_output_adapted        (offline.Source/Target + src_trajectory)
#          ${BASE}/llm_output_merged         (single-word chunks merged)
#          ${BASE}/streaming_salami_dataset  (960ms windowed)
# ============================================================
#SBATCH --job-name=salami_ja_s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=8:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/stage1_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/slurm_logs/stage1_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== SALAMI ja Stage 1 START $(date) ====="

source ~/.bashrc
conda activate SMT

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech
TSV=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv

mkdir -p ${BASE}/slurm_logs

# Step 1: repair/filter SALAMI raw source chunks against each utt input.
echo "--- fix salami raw source chunks ---"
rm -rf ${BASE}/llm_output_raw_fixed
python ${CODE}/fix_llm_raw.py \
  --in_dir          ${BASE}/llm_output_raw \
  --out_dir         ${BASE}/llm_output_raw_fixed \
  --out_good_jsonl  ${BASE}/good_salami_source_fixed.jsonl

# Step 2: adapt salami raw -> offline.{Source,Target} + src_trajectory
echo "--- adapt salami raw to east-compatible format ---"
rm -rf ${BASE}/llm_output_adapted
python ${CODE}/salami/adapt_salami_to_east_format.py \
  --raw-dir       ${BASE}/llm_output_raw_fixed \
  --manifest-tsv  ${TSV} \
  --output-dir    ${BASE}/llm_output_adapted \
  --overwrite

# Step 3: merge one-word Source chunks (re-uses east's post_process)
echo "--- post_process: merge one-word chunks ---"
rm -rf ${BASE}/llm_output_merged
python ${CODE}/post_process_llm_output_gigaspeech.py \
  --input-dir   ${BASE}/llm_output_adapted \
  --output-dir  ${BASE}/llm_output_merged \
  --overwrite

# Step 4: build streaming trajectory using src_trajectory (no MFA)
echo "--- multi_trajectory (960ms) ---"
rm -rf ${BASE}/streaming_salami_dataset
python ${CODE}/multi_trajectory_gigaspeech.py \
  --llm-dir     ${BASE}/llm_output_merged \
  --output-dir  ${BASE}/streaming_salami_dataset \
  --chunk-ms    960 \
  --overwrite

# counts
ADAPT=$(find ${BASE}/llm_output_adapted -maxdepth 1 -name '*.json' | wc -l)
MRG=$(find ${BASE}/llm_output_merged -maxdepth 1 -name '*.json' | wc -l)
STR=$(find ${BASE}/streaming_salami_dataset -maxdepth 2 -name '*.json' | wc -l)
echo "adapted: ${ADAPT}  merged: ${MRG}  streaming: ${STR}"

echo "===== SALAMI ja Stage 1 DONE $(date) ====="
echo "Next: sbatch stage2_metricx_input_ja.sh"
