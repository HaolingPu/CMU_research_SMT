#!/usr/bin/env bash
#SBATCH --job-name=amb40k_dual
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:3
#SBATCH --mem=96G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-7%8
#SBATCH --time=3-00:00:00
#SBATCH --time-min=01:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-o9-24,babel-q9-32,babel-t5-28
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -euo pipefail
exec bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_ambiguity_dual_sampler_group_worker.sh
