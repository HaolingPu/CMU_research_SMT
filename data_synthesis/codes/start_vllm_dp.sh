#!/usr/bin/env bash
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=128G
#SBATCH --partition=general
#SBATCH --time=20:00:00

#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

##Optional but recommended:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

conda activate metricx

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
  --qe || true

echo "[OK] MetricX scoring step finished (errors ignored)"

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
