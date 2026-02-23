#!/usr/bin/env bash
#SBATCH --job-name=giga_salami_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/pipeline_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/pipeline_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# conda init
set -e
echo "===== START PIPELINE ====="

source ~/.bashrc

conda activate SMT

BASE=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami

# 0) Fix LLM raw: restore punctuation from manifest, filter token mismatches,
#    sync Chinese punct to repaired English boundaries, filter missing zh punct
rm -rf  ${BASE}/llm_output_raw_fixed

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/fix_llm_raw.py \
  --in_dir         ${BASE}/llm_output_raw \
  --out_dir        ${BASE}/llm_output_raw_fixed \
  --out_good_jsonl ${BASE}/good_train_xl_salami_fixed.jsonl \
  --sync_zh_punct \
  --zh_punct_allow_insert \
  --filter_zh_punct \
  --zh_excess_threshold 2

# 1) salami -> offline (restructured)
rm -rf  ${BASE}/llm_output_offline_fixed

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/salami/map_salami_to_offline_gigaspeech.py \
  --input_dir  ${BASE}/llm_output_raw_fixed \
  --output_dir ${BASE}/llm_output_offline_fixed

# 2) Merge one-word chunks
rm -rf  ${BASE}/llm_output_offline_merged_fixed

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/post_process_llm_output_gigaspeech.py \
  --input-dir  ${BASE}/llm_output_offline_fixed \
  --output-dir ${BASE}/llm_output_offline_merged_fixed \
  --overwrite

# MFA corpus and MFA textgrid (DONE - reuse existing)

# 3) find bad json
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/find_bad_json_gigaspeech.py \
  --llm-dir    ${BASE}/llm_output_offline_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --corpus-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus \
  --good-jsonl ${BASE}/good_train_xl_salami_mfa.jsonl \
  --bad-jsonl  ${BASE}/bad_train_xl_salami_mfa.jsonl \
  --allow-one-word

# 4) creating streaming dataset
rm -rf  ${BASE}/streaming_salami_dataset

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/multi_trajectory_gigaspeech.py \
  --llm-dir    ${BASE}/llm_output_offline_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --good-jsonl ${BASE}/good_train_xl_salami_mfa.jsonl \
  --output-dir ${BASE}/streaming_salami_dataset \
  --chunk-ms 960 \
  --overwrite

# 4-check) Quality check on streaming dataset
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/check_streaming_dataset.py \
  --stream_dir ${BASE}/streaming_salami_dataset \
  --tsv        /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
  --report     ${BASE}/check_streaming_report.json \
  --zh_excess_threshold 2 \
  || true   # non-blocking: report only, never fail the pipeline

# 6) create Metricx_input
conda deactivate
conda activate metricx
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/convert_metricx_gigaspeech.py \
  --stream_dir ${BASE}/streaming_salami_dataset \
  --output     ${BASE}/metricx_input.jsonl

echo "Pipeline done, Now need to split the metricx input and predict!"

# 把 metricx input split 8 份
rm -rf  ${BASE}/metricx_shards
mkdir -p ${BASE}/metricx_shards
split -d -n l/8 \
  ${BASE}/metricx_input.jsonl \
  ${BASE}/metricx_shards/input_
