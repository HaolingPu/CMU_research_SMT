#!/usr/bin/env bash
#SBATCH --job-name=giga_refineEAST_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/slurm_logs/pipeline_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/slurm_logs/pipeline_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START PIPELINE ====="

source ~/.bashrc

conda activate SMT

BASE=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east

# ===========================
# 0) Fix LLM raw: restore punctuation from manifest, filter token mismatches,
#    sync Chinese punct to repaired English boundaries, filter missing zh punct
# ===========================
rm -rf ${BASE}/llm_output_raw_fixed

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/fix_llm_raw.py \
  --in_dir         ${BASE}/llm_output_raw \
  --out_dir        ${BASE}/llm_output_raw_fixed \
  --out_good_jsonl ${BASE}/good_train_xl_refined_fixed.jsonl \
  --sync_zh_punct \
  --zh_punct_allow_insert \
  --filter_zh_punct \
  --zh_excess_threshold 2

# ===========================
# 1) Post-process LLM output (raw_fixed -> merged_fixed)
# ===========================
rm -rf ${BASE}/llm_output_merged_fixed

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/post_process_llm_output_gigaspeech.py \
  --input-dir  ${BASE}/llm_output_raw_fixed \
  --output-dir ${BASE}/llm_output_merged_fixed \
  --overwrite

# ===========================
# 2) Strict find_bad
# ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/find_bad_json_gigaspeech.py \
  --llm-dir    ${BASE}/llm_output_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --corpus-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_corpus \
  --good-jsonl ${BASE}/good_train_xl_refined_mfa.jsonl \
  --bad-jsonl  ${BASE}/bad_train_xl_refined_mfa.jsonl

# ===========================
# 3) Build trajectory (chunk=960ms)
# ===========================
rm -rf ${BASE}/streaming_refined_east_dataset

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/multi_trajectory_gigaspeech.py \
  --llm-dir    ${BASE}/llm_output_merged_fixed \
  --mfa-dir    /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/gigaspeech_mfa_textgrid \
  --good-jsonl ${BASE}/good_train_xl_refined_mfa.jsonl \
  --output-dir ${BASE}/streaming_refined_east_dataset \
  --chunk-ms 960 \
  --overwrite

# 3-check) Quality check on streaming dataset
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/check_streaming_dataset.py \
  --stream_dir ${BASE}/streaming_refined_east_dataset \
  --tsv        /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
  --report     ${BASE}/check_streaming_report.json \
  --zh_excess_threshold 2 \
  || true   # non-blocking

# ===========================
# 4) Convert to MetricX input
# ===========================
conda deactivate
conda activate metricx

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/convert_metricx_gigaspeech.py \
  --stream_dir ${BASE}/streaming_refined_east_dataset \
  --output     ${BASE}/metricx_input.jsonl

echo "Pipeline done, Now need to split the metricx input and predict!"

# 把 metricx input split 8 份
rm -rf  ${BASE}/metricx_shards
mkdir -p ${BASE}/metricx_shards

split -d -n l/8 \
  ${BASE}/metricx_input.jsonl \
  ${BASE}/metricx_shards/input_


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
#   --input_file ${BASE}/metricx_input.jsonl \
#   --output_file ${BASE}/metricx_output.jsonl \
#   --qe || true


# 合并8个metricx output (run manually after all 8 predict jobs finish)
cat ${BASE}/metricx_shards/output_*.jsonl \
  > ${BASE}/metricx_output.jsonl


# ===========================
# 6) Filter MetricX
# ===========================
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/filter_metricx_gigaspeech.py \
  --input     ${BASE}/metricx_output.jsonl \
  --output    ${BASE}/metricx_filtered_t3.0.jsonl \
  --threshold 3.0

# ===========================
# 7) Final dataset
# ===========================
rm -rf ${BASE}/final_jsonl_refined_EAST

python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/final_output_gigaspeech.py \
  --metricx_jsonl ${BASE}/metricx_filtered_t3.0.jsonl \
  --stream_dir    ${BASE}/streaming_refined_east_dataset \
  --output_dir    ${BASE}/final_jsonl_refined_EAST

# 7-check) Quality check on final output
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/check_salami_final.py \
  --tsv       /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
  --manifest  ${BASE}/metricx_filtered_t3.0.jsonl \
  --final_dir ${BASE}/final_jsonl_refined_EAST \
  --report    ${BASE}/check_final_report.json \
  || true   # non-blocking

echo "===== PIPELINE DONE ====="
