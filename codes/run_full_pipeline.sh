#!/bin/bash
#SBATCH --job-name=streaming-pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00


set -e
echo "===== START PIPELINE ====="

# 1. Activate conda environment
conda activate vllm
echo "[OK] Activated vllm"

# 2. Run llm_output.py
cd /data/user_data/haolingp/codes/
python /data/user_data/haolingp/codes/llm_output.py --num-en 1 --num-parquets 1

# python /data/user_data/haolingp/codes/llm_output.py --num-en 1 --num-parquets 1 --num-samples 20


# Now we have the llm segmentation
# 3. run MFA =====> audio + text
python /data/user_data/haolingp/codes/export_mfa_corpus.py \
    --input-root /data/group_data/li_lab/siqiouya/datasets/yodas-granary/data \
    --lang en000 \
    --output-dir /data/user_data/haolingp/outputs/mfa_corpus \
    --num-parquets 1 \
    # --num-samples 20


# 4. audio + text ==> textgrid
# delete the cache first
rm -rf /data/user_data/haolingp/outputs/mfa_output
rm -rf ~/.local/share/mfa

conda deactivate
conda activate SMT

mfa align \
    /data/user_data/haolingp/outputs/mfa_corpus \
    english_us_arpa \
    english_us_arpa \
    /data/user_data/haolingp/outputs/mfa_textgrid_output \
    --clean

# 5. find the good and bad json:
python /data/user_data/haolingp/codes/find_bad_json.py

# 6. use good jsons to get the trajectory
echo "Running mult_trajectory.py ..."
python /data/user_data/haolingp/codes/multi_trajectory.py
echo "[OK] Generated streaming_dataset/"

# 7. convert the stream into metrix form

echo "Converting streaming dataset → MetricX input ..."
python /data/user_data/haolingp/codes/convert_metricx.py \
    --stream_dir /data/user_data/haolingp/outputs/streaming_dataset \
    --output /data/user_data/haolingp/outputs/metricx_input.jsonl

echo "[OK] metricx_input.jsonl generated"



###########################################
#8. Activate MetricX environment
###########################################
conda deactivate
conda activate metricx
echo "[OK] Activated metricx env"

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

cd /data/user_data/haolingp/codes/metricx/
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer $TOKENIZER_DIR \
  --model_name_or_path $MODEL_DIR \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file  $METRICX_INPUT \
  --output_file $METRICX_OUTPUT \
  --qe

echo "[OK] MetricX scoring finished"


###########################################
# 6. Filter bad examples
###########################################
THRESH=5.0
FILTERED_OUTPUT=/data/user_data/haolingp/outputs/metricx_filtered_t${THRESH}.jsonl

echo "Filtering MetricX results with threshold = $THRESH ..."

python /data/user_data/haolingp/codes/filter_metricx.py \
  --input  $METRICX_OUTPUT \
  --output $FILTERED_OUTPUT \
  --threshold $THRESH

echo "[OK] Filtered dataset saved → $FILTERED_OUTPUT"


###########################################
# 7. get the final dataset
###########################################

python /data/user_data/haolingp/codes/final_output.py \
  --metricx_jsonl /data/user_data/haolingp/outputs/metricx_filtered_t5.0.jsonl \
  --stream_dir /data/user_data/haolingp/outputs/streaming_dataset \
  --output_dir /data/user_data/haolingp/outputs/final_jsonl_dataset \



echo "===== PIPELINE COMPLETE ====="
