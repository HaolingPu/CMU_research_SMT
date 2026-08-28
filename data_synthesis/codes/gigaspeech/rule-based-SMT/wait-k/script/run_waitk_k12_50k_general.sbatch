#!/usr/bin/env bash
#SBATCH --job-name=waitk_k12_50k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7%8
#SBATCH --time=1-10:00:00
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/waitk_k12_50k_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/waitk_k12_50k_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/wait-k/script/run_waitk_common.sh 12 k12
