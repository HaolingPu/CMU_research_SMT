#!/usr/bin/env bash
set -e

# Submit the min-p=0.1 (1em1) pipeline: gen + QE prepare/predict/finalize.

MINP_DIR="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_0.1"
QE_THRESHOLD=3.0

# Stage 1: generation
gen_id=$(sbatch --parsable "${MINP_DIR}/run_minp_1em1_40k_general.sbatch")
experiment_dir="${OUTPUT_ROOT}/job_${gen_id}"
metricx_run_dir="${OUTPUT_ROOT}/all_40k-metricx"
filtered_output_dir="${OUTPUT_ROOT}/consensus_decoding_en_zh_minp_1em1"

prep_id=$(sbatch --parsable \
  --dependency="afterok:${gen_id}" \
  --export="ALL,EXPERIMENT_DIR=${experiment_dir},METRICX_RUN_DIR=${metricx_run_dir},NUM_SHARDS=8" \
  "${MINP_DIR}/run_metricx_qe_prepare.sbatch")

predict_id=$(sbatch --parsable \
  --dependency="afterok:${prep_id}" \
  --export="ALL,METRICX_RUN_DIR=${metricx_run_dir}" \
  "${MINP_DIR}/run_metricx_qe_8gpu.sbatch")

final_id=$(sbatch --parsable \
  --dependency="afterok:${predict_id}" \
  --export="ALL,METRICX_RUN_DIR=${metricx_run_dir},EXPERIMENT_DIR=${experiment_dir},FILTERED_OUTPUT_DIR=${filtered_output_dir},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=8" \
  "${MINP_DIR}/run_metricx_qe_finalize.sbatch")

echo "minp=0.1 (1em1): ${gen_id} -> ${prep_id} -> ${predict_id} -> ${final_id}"
