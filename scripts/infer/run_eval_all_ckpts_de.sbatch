#!/usr/bin/env bash
#SBATCH --job-name=eval_longform_de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --exclude=babel-t9-16,babel-o5-16,babel-p5-24
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/eval_longform_de_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/eval_longform_de_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

# Fast-fail if home NFS not mounted on this node.
if [ ! -f /home/haolingp/miniconda3/etc/profile.d/conda.sh ]; then
    echo "[FATAL] /home/haolingp/miniconda3 not visible on $(hostname). Aborting." >&2
    exit 1
fi

mkdir -p /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs

cd /home/haolingp/CMU_research_SMT/scripts/infer
echo "===== eval_all_ckpts_de.sh START $(date) ====="
bash /home/haolingp/CMU_research_SMT/scripts/infer/eval_all_ckpts_de.sh
echo "===== eval_all_ckpts_de.sh DONE $(date) ====="
