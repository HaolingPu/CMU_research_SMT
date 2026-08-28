#!/usr/bin/env bash
#SBATCH --job-name=anchor40k_p
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-11
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/anchor40k_p_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/anchor40k_p_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
exec bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/_run_anchor_40k_common.sh
