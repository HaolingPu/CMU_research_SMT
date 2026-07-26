#!/usr/bin/env bash
#SBATCH --job-name=Jw3full
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-11
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/Jw3full_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/Jw3full_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# FULL canonical pipeline, generation step:
#   decode the first 40,000 rows of the NEW dASR (qwenasr sub-sentence-split) data,
#   window=3, same dualbase token-id-level consensus decoder as old/new-ctrl.
# Downstream (period-fix -> per-sentence QE3 -> length filter -> sample 12.5k ->
#   train -> infer -> eval) is chained separately afterok this array.
set -e

export INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_qwenasr_sentsplit_frozen_llm_reference.tsv"
export OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k_qwenasr_sentsplit_win3_FULL40k"
export FUTURE_SRC_WINDOW=3
export TOTAL_ROWS=40000
export ROW_OFFSET=0
export NUM_TASKS=12

exec bash /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/_run_J_40k_common.sh
