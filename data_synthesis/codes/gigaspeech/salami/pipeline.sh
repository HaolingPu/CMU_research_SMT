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

# 0) Fix LLM raw: restore punctuation from manifest, filter token mismatches
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/fix_llm_raw.py \
  --in_dir         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_raw \
  --out_dir        /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_raw_fixed \
  --out_good_jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/good_train_xl_salami_fixed.jsonl

# 1) salami -> offline (restructured)
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/salami/map_salami_to_offline_gigaspeech.py \
  --input_dir  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_raw_fixed \
  --output_dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_offline_fixed

# 2) Merge one-word chunks
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/post_process_llm_output_gigaspeech.py \
  --input-dir  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_offline_fixed \
  --output-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_offline_merged_fixed \
  --overwrite

# MFA corpus and MFA textgrid (DONE - reuse existing)

# 3) find bad json
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/find_bad_json_gigaspeech.py \
  --llm-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_offline_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --corpus-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus \
  --good-jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/good_train_xl_salami_mfa.jsonl \
  --bad-jsonl  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/bad_train_xl_salami_mfa.jsonl \
  --allow-one-word

# 4) creating streaming dataset
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/multi_trajectory_gigaspeech.py \
  --llm-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/llm_output_offline_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --good-jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/good_train_xl_salami_mfa.jsonl \
  --output-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/streaming_salami_dataset \
  --chunk-ms 960 \
  --overwrite

# 6) create Metricx_input
conda deactivate
conda activate metricx
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/convert_metricx_gigaspeech.py \
  --stream_dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/streaming_salami_dataset \
  --output /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_input.jsonl



echo "Pipeline done, Now need to split the metricx input and predict!"




# 把 metricx input split 8 份
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_shards
split -d -n l/8 \
  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_input.jsonl \
  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_shards/input_
