#!/usr/bin/env bash
set -e

# Submit a single wait-k pipeline (gen + QE prepare + QE predict + QE finalize)
# for the given k. Used by the watchdog to stagger submissions within the
# MaxJobsPU=10 limit.
#
# Usage: bash submit_one_pipeline.sh <k>   e.g. submit_one_pipeline.sh 9

K="${1:?Usage: bash submit_one_pipeline.sh <k>}"
LABEL="k${K}"

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/wait-k/script"
MINP_QE_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"
OUTPUT_BASE="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/wait-k"
QE_THRESHOLD=3.0

GEN_SBATCH="${SCRIPT_DIR}/run_waitk_${LABEL}_50k_general.sbatch"
OUTPUT_ROOT="${OUTPUT_BASE}/waitk_${LABEL}_50k_general"

gen_id=$(sbatch --parsable "${GEN_SBATCH}")
experiment_dir="${OUTPUT_ROOT}/job_${gen_id}"
metricx_run_dir="${OUTPUT_ROOT}/all_50k-metricx"
filtered_output_dir="${OUTPUT_ROOT}/waitk_${LABEL}_qe3"

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

echo "k=${K}: ${gen_id} -> ${prep_id} -> ${predict_id} -> ${final_id}"
