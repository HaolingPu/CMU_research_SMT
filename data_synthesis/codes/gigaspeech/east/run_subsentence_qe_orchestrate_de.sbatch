#!/usr/bin/env bash
# ============================================================
# Orchestrator wrapper that runs submit_subsentence_qe_de.sh as a real
# slurm job, so it can be made --dependency=afterok:<stage1_jid>.
#
# submit_subsentence_qe_de.sh runs build_consensus_format + prepare_segale_shards
# inline (CPU+1GPU env), then sbatches the rest of the sub-QE chain.
# ============================================================
#SBATCH --job-name=east_de_subqe_orch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-p9-28,babel-s5-32,babel-m5-32,babel-n9-32,babel-o5-16,babel-o5-24
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/subqe_orch_%j.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/slurm_logs/subqe_orch_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e
echo "===== sub-QE orchestrator (de) START $(date) ====="
source ~/.bashrc

export QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
echo "QE_THRESHOLD = ${QE_THRESHOLD}"

bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/east/submit_subsentence_qe_de.sh

echo "===== sub-QE orchestrator (de) DONE $(date) ====="
