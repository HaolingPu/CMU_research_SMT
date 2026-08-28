#!/usr/bin/env bash
set -e

SCRIPT_DIR="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/topp"

sbatch "${SCRIPT_DIR}/run_topp_0p5_40k_preempt.sbatch"
sbatch "${SCRIPT_DIR}/run_topp_0p7_40k_preempt.sbatch"
sbatch "${SCRIPT_DIR}/run_topp_0p9_40k_preempt.sbatch"
sbatch "${SCRIPT_DIR}/run_topp_0p95_40k_preempt.sbatch"
sbatch "${SCRIPT_DIR}/run_topp_0p99_40k_preempt.sbatch"
