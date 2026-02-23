#!/usr/bin/env bash
#SBATCH --job-name=EAST_Refined
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=general
#SBATCH --time=2-00:00:00

#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

##Optional but recommended:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu


set -e
echo "===== START PIPELINE ====="

source ~/.bashrc
# 1. Activate conda environment
conda activate vllm
echo "[OK] Activated vllm"


# 2. Run llm_output.py
cd /data/user_data/haolingp/data_synthesis/codes/yodas/
# python /data/user_data/haolingp/data_synthesis/codes/yodas/llm_output.py --num-en 1 --num-parquets 1

echo "[OK] Segmentation completed."

# ⭐ 新增：后处理LLM输出，合并单词chunks if it is only one word in the chunk
python /data/user_data/haolingp/data_synthesis/codes/yodas/post_process_llm_output.py \
    --input_dir /data/user_data/haolingp/outputs/llm_output_OFFLINE \
    --output_dir /data/user_data/haolingp/outputs/llm_output_SALAMI_merged \
    --lang en000



# Now we have the llm segmentation
# 3. run MFA =====> audio + text
echo "(Already exists)Exporting MFA corpus for en000 (ALL parquets)..."
# python /data/user_data/haolingp/data_synthesis/codes/yodas/export_mfa_corpus.py \
#     --input-root /data/group_data/li_lab/siqiouya/datasets/yodas-granary/data \
#     --lang en000 \
#     --output-dir /data/user_data/haolingp/outputs/mfa_corpus \
#     --num-parquets all \
#     --num-samples all


# 4. audio + text ==> textgrid
# delete the cache first
rm -rf /data/user_data/haolingp/outputs/mfa_output
rm -rf ~/.local/share/mfa

conda deactivate
conda activate SMT

echo "(Already exists) mfa_textgrid_output"
# mfa align \
#   --clean \
#   --final_clean \
#   --single_speaker \
#   --num_jobs 64 \
#   --temporary_directory /data/user_data/haolingp/.cache/mfa \
#   --overwrite \
#   --output_format long_textgrid \
#   /data/user_data/haolingp/outputs/mfa_corpus/en000 \
#   english_us_arpa \
#   english_us_arpa \
#   /data/user_data/haolingp/outputs/textgrids/en000



# 5. find the good and bad json:
python /data/user_data/haolingp/data_synthesis/codes/yodas/find_bad_json.py \
  --llm-root /data/user_data/haolingp/outputs/llm_output_SALAMI_merged \
  --mfa-root /data/user_data/haolingp/outputs/textgrids \
  --corpus-root /data/user_data/haolingp/outputs/mfa_corpus \
  --lang en000 \
  --output-dir /data/user_data/haolingp/outputs \
  --good-name good_SALAMI_en000_all.jsonl \
  --bad-name bad_SALAMI_en000_all.jsonl


# 6. use good jsons to get the trajectory
echo "Running mult_trajectory.py ..."
# python /data/user_data/haolingp/data_synthesis/codes/yodas/multi_trajectory.py \
#   --llm-root /data/user_data/haolingp/outputs/llm_output_EAST_merged \
#   --mfa-root /data/user_data/haolingp/outputs/mfa_textgrid_output \
#   --good-root /data/user_data/haolingp/outputs \
#   --output-root /data/user_data/haolingp/outputs/streaming_dataset_EAST \
#   --langs en000
# echo "[OK] Generated streaming_dataset/"

python /data/user_data/haolingp/data_synthesis/codes/yodas/multi_trajectory.py \
  --llm-root /data/user_data/haolingp/outputs/llm_output_modified \
  --mfa-root /data/user_data/haolingp/outputs/textgrids \
  --good-root /data/user_data/haolingp/outputs \
  --output-root /data/user_data/haolingp/outputs/streaming_dataset \
  --langs en000
echo "[OK] Generated streaming_dataset/"

# 7. convert the stream into metrix form

conda deactivate
conda activate metricx

echo "Converting streaming dataset → MetricX input ..."
python /data/user_data/haolingp/data_synthesis/codes/yodas/convert_metricx.py \
    --stream_dir /data/user_data/haolingp/outputs/streaming_dataset  \
    --output /data/user_data/haolingp/outputs/metricx_input.jsonl

echo "[OK] metricx_input.jsonl generated"


###########################################
# 5. Run MetricX-24 QE Scoring
###########################################

TOKENIZER_DIR=/data/user_data/haolingp/models/mt5-xl
MODEL_DIR=/data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6
METRICX_OUTPUT=/data/user_data/haolingp/outputs/metricx_output.jsonl
METRICX_INPUT=/data/user_data/haolingp/outputs/metricx_input.jsonl

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

echo "Running MetricX QE ..."

cd /data/user_data/haolingp/data_synthesis/codes/metricx/
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer $TOKENIZER_DIR \
  --model_name_or_path $MODEL_DIR \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file  $METRICX_INPUT \
  --output_file $METRICX_OUTPUT \
  --qe || true

echo "[OK] MetricX scoring step finished (errors ignored)"

###########################################
# 6. Filter bad examples
###########################################
THRESH=3.0
FILTERED_OUTPUT=/data/user_data/haolingp/outputs/metricx_filtered_t${THRESH}_EAST.jsonl

echo "Filtering MetricX results with threshold = $THRESH ..."

python /data/user_data/haolingp/data_synthesis/codes/yodas/filter_metricx.py \
  --input  $METRICX_OUTPUT \
  --output $FILTERED_OUTPUT \
  --threshold $THRESH

echo "[OK] Filtered dataset saved → $FILTERED_OUTPUT"


###########################################
# 7. get the final dataset
###########################################

python /data/user_data/haolingp/data_synthesis/codes/yodas/final_output.py \
  --metricx_jsonl /data/user_data/haolingp/outputs/metricx_filtered_t5.0_EAST.jsonl \
  --stream_dir /data/user_data/haolingp/outputs/streaming_dataset_EAST \
  --output_dir /data/user_data/haolingp/outputs/final_jsonl_dataset_EAST \


# python /data/user_data/haolingp/data_synthesis/codes/yodas/final_output.py \
#   --metricx_jsonl /data/user_data/haolingp/outputs/metricx_filtered_t5.0_EAST.jsonl \
#   --stream_dir /data/user_data/haolingp/outputs/streaming_dataset_EAST \
#   --output_dir /data/user_data/haolingp/outputs/final_jsonl_dataset_EAST \
#   --only-lang en000 \
#   --only-pq 00000000 \
#   --max-lines 10000


echo "===== PIPELINE COMPLETE ====="
