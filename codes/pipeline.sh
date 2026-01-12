#!/usr/bin/env bash
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=128G
#SBATCH --partition=general
#SBATCH --time=20:00:00

#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

##Optional but recommended:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu


source ~/.bashrc
conda activate metricx

echo "Running on host: $(hostname)"
echo "PWD at start: $(pwd)"

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
