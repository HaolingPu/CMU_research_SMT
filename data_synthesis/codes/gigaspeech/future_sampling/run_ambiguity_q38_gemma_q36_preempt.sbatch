#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=96G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu
# Job name, array, --time-min, --exclude, log paths and the task environment come from scripts/submit_v3json_35k.sh.

set -euo pipefail
exec bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_ambiguity_q38_gemma_q36_worker.sh
