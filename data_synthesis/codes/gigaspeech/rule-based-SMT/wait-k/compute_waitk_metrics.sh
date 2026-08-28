#!/usr/bin/env bash
# Compute BLEU, LAAL, and MetricX QE for all 5 wait-k configs (k=3,6,9,12,15).
# Outputs a summary table matching the LA-2 format.
set -e

WAITK_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/wait-k"
CONVERTER="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/convert_metricx_consensus.py"
METRICX_CODE="/home/haolingp/CMU_research_SMT/data_synthesis/codes/metricx"
CONDA_BASE="/home/haolingp/miniconda3"
BASE_PY="${CONDA_BASE}/bin/python"

# 1. Stop vLLM server if running
if [[ -f /tmp/vllm_waitk_local.pid ]]; then
  echo "[1/4] Stopping vLLM server..."
  PORT=8100 PID_FILE=/tmp/vllm_waitk_local.pid \
    bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/serve_instruct_gpu0.sh stop || true
  sleep 5
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 2. Convert JSONs to MetricX input
echo "[2/4] Converting wait-k JSONs to MetricX input..."
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metricx
for K in 3 6 9 12 15; do
  SRC_DIR="${WAITK_DIR}/output/waitk${K}_100_stride1"
  OUT_JSONL="${WAITK_DIR}/output/waitk${K}_metricx_input.jsonl"
  python "${CONVERTER}" --input-dir "${SRC_DIR}" --output "${OUT_JSONL}"
done

# 3. Run MetricX QE for each k
echo "[3/4] Running MetricX QE predict..."
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYARROW_IGNORE_TIMEZONE=1
export MKL_SERVICE_FORCE_INTEL=1
export CUDA_VISIBLE_DEVICES=0

cd "${METRICX_CODE}"
for K in 3 6 9 12 15; do
  INPUT="${WAITK_DIR}/output/waitk${K}_metricx_input.jsonl"
  OUTPUT="${WAITK_DIR}/output/waitk${K}_metricx_output.jsonl"
  # Clear HF datasets cache to avoid stale caching
  rm -rf /data/user_data/haolingp/hf_cache/datasets/json/default-* 2>/dev/null || true
  echo "  -> k=${K}"
  PYTHONNOUSERSITE=1 python -m metricx24.predict \
    --tokenizer /data/user_data/haolingp/models/mt5-xl \
    --model_name_or_path /data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6 \
    --max_input_length 1536 \
    --batch_size 1 \
    --input_file "${INPUT}" \
    --output_file "${OUTPUT}" \
    --qe
done

# 4. Aggregate BLEU/LAAL from JSONs + QE from metricx output, print summary
echo "[4/4] Computing summary table..."
"${BASE_PY}" "${WAITK_DIR}/aggregate_waitk_metrics.py"
