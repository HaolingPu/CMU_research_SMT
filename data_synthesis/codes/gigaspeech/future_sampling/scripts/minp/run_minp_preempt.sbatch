#!/usr/bin/env bash
#SBATCH --job-name=minp_preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-11
#SBATCH --time=1-10:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/minp_preempt_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/minp_preempt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

MINP="${MINP:?Set MINP via --export=ALL,MINP=<float> e.g. 0.01 ; LABEL is optional, default minp_<MINP>}"
LABEL="${LABEL:-minp_${MINP}}"

bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/scripts/minp/run_minp_common.sh "${MINP}" "${LABEL}"
