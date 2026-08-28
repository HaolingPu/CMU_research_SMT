#!/usr/bin/env bash
#SBATCH --job-name=Jw3cls
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-11
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH -o /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/Jw3cls_%A_%a.out
#SBATCH -e /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/slurm_logs/Jw3cls_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

# FULL canonical pipeline, generation step — CLAUSE-SPLIT variant (⑤b, matches old-asr clause granularity).
#   Same as run_J_win3_FULL40k_preempt.sbatch but the INPUT_TSV is the qwenasr blob
#   re-split on SENTENCE-FINAL punctuation only ([.!?;。！？；] with abbreviation/decimal
#   guards, no spaCy, no mid-clause coordinating-conjunction cuts), see
#   split_src_text_full_punct.py. Eliminates the boundary-leakage fragments + adjacent
#   duplication that the spaCy cc-split caused. window=3, same dualbase decoder.
#   Downstream (period-fix -> per-sentence QE3 -> length -> sample 12.5k -> train
#   -> infer -> eval) is chained afterok this array by run_CLAUSE_win3_FULL_chain.sh.
set -e

export INPUT_TSV="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_qwenasr_clausesplit_frozen_llm_reference.tsv"
export OUTPUT_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k_qwenasr_clausesplit_win3_FULL40k"
export FUTURE_SRC_WINDOW=3
export TOTAL_ROWS=40000
export ROW_OFFSET=0
export NUM_TASKS=12

exec bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/_run_J_40k_common.sh
