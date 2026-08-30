#!/usr/bin/env bash
#SBATCH --job-name=amb40k_decode
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=96G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-11%12
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-o9-24,babel-q9-32,babel-t5-28
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/amb40k_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/amb40k_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -euo pipefail
exec bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_ambiguity_q38_gemma_q36_worker.sh
