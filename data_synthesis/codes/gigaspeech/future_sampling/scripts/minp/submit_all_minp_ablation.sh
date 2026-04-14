#!/usr/bin/env bash
set -e

SCRIPT_DIR="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp"

# Absolute min-p ablation: 1e-2, 1e-3, 1e-4, 1e-5
sbatch "${SCRIPT_DIR}/run_minp_1em2_40k_preempt.sbatch"
sbatch "${SCRIPT_DIR}/run_minp_1em3_40k_preempt.sbatch"
sbatch "${SCRIPT_DIR}/run_minp_1em4_40k_preempt.sbatch"
sbatch "${SCRIPT_DIR}/run_minp_1em5_40k_preempt.sbatch"
