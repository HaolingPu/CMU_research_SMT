#!/usr/bin/env bash
#SBATCH --job-name=topk_preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-11
#SBATCH --time=1-10:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/topk_preempt_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/topk_preempt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

TOPK="${TOPK:?Set TOPK via --export=ALL,TOPK=<int>}"
LABEL="${LABEL:-topk${TOPK}}"

bash /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk/run_topk_common.sh "${TOPK}" "${LABEL}"
