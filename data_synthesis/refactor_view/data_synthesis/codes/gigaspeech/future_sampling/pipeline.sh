#!/bin/bash
#SBATCH --job-name=giga_fs_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling/slurm_logs/pipeline_%A.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling/slurm_logs/pipeline_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== START FUTURE SAMPLING PIPELINE ====="

source ~/.bashrc

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
BASE="/data/user_data/haolingp/data_synthesis"
OUT="${BASE}/outputs/gigaspeech/train_xl_future_sampling"
CODE="${BASE}/codes/gigaspeech"

STREAM_DIR="${OUT}/llm_output"
METRICX_INPUT="${OUT}/metricx_input.jsonl"
METRICX_OUTPUT="${OUT}/metricx_output.jsonl"
THRESH=3.0
FILTERED="${OUT}/metricx_filtered_t${THRESH}.jsonl"
FINAL_DIR="${OUT}/final_jsonl_dataset"

# ----------------------------------------------------------------
# 1) Convert streaming output -> MetricX QE input
#    (future_sampling format is auto-detected by the updated converter)
# ----------------------------------------------------------------
conda activate metricx

echo "Converting streaming output -> MetricX input ..."
python "${CODE}/convert_metricx_gigaspeech.py" \
  --stream_dir "${STREAM_DIR}" \
  --output "${METRICX_INPUT}" \
  --keep-source-case

echo "[OK] MetricX input: $(wc -l < "${METRICX_INPUT}") lines"

# Split into 8 shards for parallel MetricX predict
mkdir -p "${OUT}/metricx_shards"
split -d -n l/8 "${METRICX_INPUT}" "${OUT}/metricx_shards/input_"

echo "Shards created. Submit run_metricx_predict.sh next."

# ----------------------------------------------------------------
# 2) MetricX predict (run as separate array job — see run_metricx_predict.sh)
#    After all 8 shards finish, merge:
# ----------------------------------------------------------------
# cat ${OUT}/metricx_shards/output_*.jsonl > ${METRICX_OUTPUT}

# ----------------------------------------------------------------
# 3) Filter by MetricX score
# ----------------------------------------------------------------
# python "${CODE}/filter_metricx_gigaspeech.py" \
#   --input "${METRICX_OUTPUT}" \
#   --output "${FILTERED}" \
#   --threshold "${THRESH}"

# ----------------------------------------------------------------
# 4) Build final dataset
# ----------------------------------------------------------------
# python "${CODE}/final_output_gigaspeech.py" \
#   --metricx_jsonl "${FILTERED}" \
#   --stream_dir "${STREAM_DIR}" \
#   --output_dir "${FINAL_DIR}"

echo "===== PIPELINE STAGE 1 DONE (metricx shards ready) ====="
