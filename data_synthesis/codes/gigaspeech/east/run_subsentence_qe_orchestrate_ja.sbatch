#!/usr/bin/env bash
# ============================================================
# Orchestrator wrapper that runs submit_subsentence_qe_ja.sh as a real
# slurm job, so it can be made --dependency=afterok:<stage1_jid>.
#
# submit_subsentence_qe_ja.sh runs build_consensus_format + prepare_segale_shards
# inline (CPU+1GPU env), then sbatches the rest of the sub-QE chain.
# ============================================================
#SBATCH --job-name=east_ja_subqe_orch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --exclude=babel-t9-16
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/subqe_orch_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/slurm_logs/subqe_orch_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== sub-QE orchestrator START $(date) ====="
source ~/.bashrc

# QE_THRESHOLD picked up by inner submit script (default now 5.0)
export QE_THRESHOLD="${QE_THRESHOLD:-5.0}"
echo "QE_THRESHOLD = ${QE_THRESHOLD}"

bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/east/submit_subsentence_qe_ja.sh

echo "===== sub-QE orchestrator DONE $(date) ====="
