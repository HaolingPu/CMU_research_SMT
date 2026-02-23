#!/usr/bin/env bash
#SBATCH --job-name=metricx_salami
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/llm_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/slurm_logs/llm_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu


set -e
echo "===== START PIPELINE ====="

source ~/.bashrc
conda activate metricx

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1

SHARD_ID=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
IN=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_shards/input_${SHARD_ID}
OUT=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_shards/output_${SHARD_ID}.jsonl

cd /data/user_data/haolingp/data_synthesis/codes/metricx
PYTHONNOUSERSITE=1 python -m metricx24.predict \
  --tokenizer /data/user_data/haolingp/models/mt5-xl \
  --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
  --max_input_length 1536 \
  --batch_size 1 \
  --input_file  ${IN} \
  --output_file ${OUT} \
  --qe || true

echo "DONE, Now need to concatenate"


## 合并8个metricx output (run manually after all 8 jobs finish)
cat /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_shards/output_*.jsonl \
  > /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_output.jsonl



# # # 6) Filter MetricX (run after merge)
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/filter_metricx_gigaspeech.py \
  --input /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_output.jsonl \
  --output /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_filtered_t3.0.jsonl \
  --threshold 3.0

# # # 7) Final dataset (run after filter)
python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/final_output_gigaspeech.py \
  --metricx_jsonl /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_filtered_t3.0.jsonl \
  --stream_dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/streaming_salami_dataset \
  --output_dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/final_jsonl_salami

echo "===== PIPELINE DONE ====="
