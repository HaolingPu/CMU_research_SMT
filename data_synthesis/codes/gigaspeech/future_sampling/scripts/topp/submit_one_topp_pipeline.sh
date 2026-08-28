#!/usr/bin/env bash
set -e

# Submit one top-p consensus pipeline (gen + QE prepare/predict/finalize).
#
# Usage: bash submit_one_topp_pipeline.sh <label> <val>
#   e.g. submit_one_topp_pipeline.sh 0p3 0.3

LABEL="${1:?Usage: bash submit_one_topp_pipeline.sh <label> <val>}"
VAL="${2:?Usage: bash submit_one_topp_pipeline.sh <label> <val>}"

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/topp"
MINP_QE_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"
OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topp/consensus_decoding_en_zh_topp_${VAL}"
QE_THRESHOLD=3.0

GEN_SBATCH="${SCRIPT_DIR}/run_topp_${LABEL}_40k_general.sbatch"

gen_id=$(sbatch --parsable "${GEN_SBATCH}")
experiment_dir="${OUTPUT_ROOT}/job_${gen_id}"
metricx_run_dir="${OUTPUT_ROOT}/all_40k-metricx"
filtered_output_dir="${OUTPUT_ROOT}/consensus_decoding_en_zh_topp_${LABEL}"

prep_id=$(sbatch --parsable \
  --dependency="afterok:${gen_id}" \
  --export="ALL,EXPERIMENT_DIR=${experiment_dir},METRICX_RUN_DIR=${metricx_run_dir},NUM_SHARDS=8" \
  "${MINP_QE_DIR}/run_metricx_qe_prepare.sbatch")

predict_id=$(sbatch --parsable \
  --dependency="afterok:${prep_id}" \
  --export="ALL,METRICX_RUN_DIR=${metricx_run_dir}" \
  "${MINP_QE_DIR}/run_metricx_qe_8gpu.sbatch")

final_id=$(sbatch --parsable \
  --dependency="afterok:${predict_id}" \
  --export="ALL,METRICX_RUN_DIR=${metricx_run_dir},EXPERIMENT_DIR=${experiment_dir},FILTERED_OUTPUT_DIR=${filtered_output_dir},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=8" \
  "${MINP_QE_DIR}/run_metricx_qe_finalize.sbatch")

echo "top-p=${VAL} (${LABEL}): ${gen_id} -> ${prep_id} -> ${predict_id} -> ${final_id}"
