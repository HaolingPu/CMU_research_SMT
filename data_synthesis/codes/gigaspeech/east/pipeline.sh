#!/bin/bash
#SBATCH --job-name=giga_train_xl_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/pipeline_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/slurm_logs/pipeline_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu



# conda init (避免 ~/.bashrc 里 BASHRCSOURCED 报错)

set -e
echo "===== START PIPELINE ====="

source ~/.bashrc

conda activate SMT

# ===========================
# 0) Fix LLM raw: restore punctuation from manifest, filter token mismatches
#    Input good list: good_train_xl_strict.jsonl (228k, passed MFA check)
# ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/fix_llm_raw.py \
  --in_dir         /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_raw \
  --out_dir        /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_raw_fixed \
  --out_good_jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/good_train_xl_east_fixed.jsonl

# # ===========================
# # 1) Post-process LLM output (raw_fixed -> merged_fixed)
# # ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/post_process_llm_output_gigaspeech.py \
  --input-dir  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_raw_fixed \
  --output-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_merged_fixed \
  --overwrite

# # ===========================
# # 2) Strict find_bad
# # ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/find_bad_json_gigaspeech.py \
  --llm-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --corpus-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus \
  --good-jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/good_train_xl_east_mfa.jsonl \
  --bad-jsonl  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/bad_train_xl_east_mfa.jsonl

# ===========================
# 3) Build trajectory (960ms)
# ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/multi_trajectory_gigaspeech.py \
  --llm-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --good-jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/good_train_xl_east_mfa.jsonl \
  --output-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/streaming_EAST_dataset \
  --chunk-ms 960 \
  --overwrite

# ===========================
# 4) Convert to MetricX input
# ===========================
conda deactivate
conda activate metricx
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/convert_metricx_gigaspeech.py \
  --stream_dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/streaming_EAST_dataset \
  --output /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_input.jsonl

# split into 8
mkdir -p /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_shards

split -d -n l/8 \
  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_input.jsonl \
  /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_shards/input_

echo "Pipeline done up to metricx split. Run metricx predict jobs manually."


# ===========================
# 5) MetricX predict  (run as separate jobs)
# ===========================
# export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=1
# export PYARROW_IGNORE_TIMEZONE=1
# export MKL_SERVICE_FORCE_INTEL=1

# cd /data/user_data/haolingp/data_synthesis/codes/metricx
# PYTHONNOUSERSITE=1 python -m metricx24.predict \
#   --tokenizer /data/user_data/haolingp/models/mt5-xl \
#   --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
#   --max_input_length 1536 \
#   --batch_size 1 \
#   --input_file /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_input.jsonl \
#   --output_file /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_output.jsonl \
#   --qe || true


# 合并8个metricx output (run manually after all 8 predict jobs finish)
cat /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_shards/output_*.jsonl \
  > /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_output.jsonl


# ===========================
# 6) Filter MetricX
# ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/filter_metricx_gigaspeech.py \
  --input /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_output.jsonl \
  --output /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_filtered_t3.0.jsonl \
  --threshold 3.0

# ===========================
# 7) Final dataset
# ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/final_output_gigaspeech.py \
  --metricx_jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_filtered_t3.0.jsonl \
  --stream_dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/streaming_EAST_dataset \
  --output_dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/final_jsonl_dataset

# echo "===== PIPELINE DONE ====="
