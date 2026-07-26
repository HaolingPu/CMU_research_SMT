#!/usr/bin/env bash
set -e

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash $0 OUTPUT_DIR [NUM_SHARDS]"
  exit 1
fi

OUTPUT_DIR="$1"
NUM_SHARDS="${2:-8}"
CONVERTER="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/convert_metricx_gigaspeech.py"
METRICX_INPUT="${OUTPUT_DIR}/metricx_input.jsonl"
SHARD_DIR="${OUTPUT_DIR}/metricx_shards"
ARRAY_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/run_metricx_qe_8gpu_generic.sh"

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "ERROR: output dir not found: ${OUTPUT_DIR}"
  exit 1
fi
if [[ ! -f "${CONVERTER}" ]]; then
  echo "ERROR: converter not found: ${CONVERTER}"
  exit 1
fi
if [[ "${NUM_SHARDS}" != "8" ]]; then
  echo "WARNING: ${ARRAY_SCRIPT} is hard-coded to 8 array tasks; requested NUM_SHARDS=${NUM_SHARDS}."
fi

source ~/.bashrc
conda activate metricx

python "${CONVERTER}"           --stream_dir "${OUTPUT_DIR}"           --output "${METRICX_INPUT}"           --keep-source-case

if [[ ! -s "${METRICX_INPUT}" ]]; then
  echo "ERROR: MetricX input is empty: ${METRICX_INPUT}"
  exit 1
fi

rm -rf "${SHARD_DIR}"
mkdir -p "${SHARD_DIR}"
split -d -n l/${NUM_SHARDS} "${METRICX_INPUT}" "${SHARD_DIR}/input_"

echo "Prepared MetricX input: ${METRICX_INPUT}"
echo "Prepared MetricX shards: ${SHARD_DIR}"
echo "Next step:"
echo "  sbatch --export=ALL,BASE_OUTPUT_DIR=${OUTPUT_DIR} ${ARRAY_SCRIPT}"
