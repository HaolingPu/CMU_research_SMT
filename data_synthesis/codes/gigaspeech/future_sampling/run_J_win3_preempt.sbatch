#!/usr/bin/env bash
#SBATCH --job-name=Jwin3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-7
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/Jwin3_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/Jwin3_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# Window=3 re-decode of the 12,500 matched intersection IDs (qwenasr source).
# Identical to the J_40k pipeline EXCEPT --future-source-window-chunks 3 (vs 1).
set -e

export INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_INTERSECT_win3_src12500.tsv"
export OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k_qwenasr_sentsplit_win3"
export FUTURE_SRC_WINDOW=3
export TOTAL_ROWS=12500
export ROW_OFFSET=0
export NUM_TASKS=8

exec bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/_run_J_40k_common.sh
