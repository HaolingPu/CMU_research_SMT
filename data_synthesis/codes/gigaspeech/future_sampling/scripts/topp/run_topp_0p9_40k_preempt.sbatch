#!/usr/bin/env bash
#SBATCH --job-name=topp_0p9_40k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=140G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-11
#SBATCH --time=1-10:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/topp_0p9_40k_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/topp_0p9_40k_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

bash /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topp/run_topp_common.sh 0.9 0p9
