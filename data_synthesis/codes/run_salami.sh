#!/usr/bin/env bash
#SBATCH --job-name=SALAMI
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



echo “starting !”

set -e
echo "===== START PIPELINE ====="

source ~/.bashrc


# python /data/user_data/haolingp/codes/post_process_llm_output.py \
#     --input_dir /data/user_data/haolingp/outputs/llm_output_Salami_offline \
#     --output_dir /data/user_data/haolingp/outputs/llm_output_SALAMI_merged \
#     --lang en000



# # 5. find the good and bad json:
# python /data/user_data/haolingp/codes/find_bad_json.py \
#   --llm-root /data/user_data/haolingp/outputs/llm_output_SALAMI_merged \
#   --mfa-root /data/user_data/haolingp/outputs/textgrids \
#   --corpus-root /data/user_data/haolingp/outputs/mfa_corpus \
#   --lang en000 \
#   --output-dir /data/user_data/haolingp/outputs \
#   --good-name good_SALAMI_en000_all.jsonl \
#   --bad-name bad_SALAMI_en000_all.jsonl


# # 6. use good jsons to get the trajectory
# echo "Running mult_trajectory.py ..."
# python /data/user_data/haolingp/codes/multi_trajectory.py \
#   --llm-root /data/user_data/haolingp/outputs/llm_output_SALAMI_merged \
#   --mfa-root /data/user_data/haolingp/outputs/textgrids \
#   --good-root /data/user_data/haolingp/outputs/good_SALAMI_en000_all.jsonl \
#   --output-root /data/user_data/haolingp/outputs/streaming_dataset_SALAMI \
#   --langs en000
# echo "[OK] Generated streaming_dataset_SALAMI/"



conda deactivate
conda activate metricx

# echo "Converting streaming dataset → MetricX input ..."
# python /data/user_data/haolingp/codes/convert_metricx.py \
#     --stream_dir /data/user_data/haolingp/outputs/streaming_dataset_SALAMI  \
#     --output /data/user_data/haolingp/outputs/metricx_input_SALAMI.jsonl

# echo "[OK] metricx_input.jsonl generated"


TOKENIZER_DIR=/data/user_data/haolingp/models/mt5-xl
MODEL_DIR=/data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6
METRICX_OUTPUT=/data/user_data/haolingp/outputs/SALAMI/metricx_output_SALAMI.jsonl
METRICX_INPUT=/data/user_data/haolingp/outputs/SALAMI/metricx_input_SALAMI.jsonl

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
  --qe || true

echo "[OK] MetricX scoring step finished (errors ignored)"


###########################################
# 6. Filter bad examples
###########################################
THRESH=5.0
FILTERED_OUTPUT=/data/user_data/haolingp/outputs/SALAMI/metricx_filtered_t${THRESH}_SALAMI.jsonl

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
  --metricx_jsonl /data/user_data/haolingp/outputs/SALAMI/metricx_filtered_t5.0_SALAMI.jsonl \
  --stream_dir /data/user_data/haolingp/outputs/SALAMI/streaming_dataset_SALAMI \
  --output_dir /data/user_data/haolingp/outputs/SALAMI/final_jsonl_dataset_SALAMI \


echo "===== PIPELINE COMPLETE ====="
