#!/usr/bin/env bash
#SBATCH --job-name=v3json_smoke16
#SBATCH --nodes=1 --cpus-per-task=12 --gres=gpu:L40S:2 --mem=96G
#SBATCH --partition=preempt --qos=preempt_qos --time=03:00:00 --time-min=01:00:00
#SBATCH --exclude=babel-o9-24,babel-q9-32,babel-t5-28,babel-v9-28,babel-p5-24,babel-o5-28,babel-p5-20
#SBATCH -o /home/haolingp/v3probe/cases_logs/smoke16_%j.out -e /home/haolingp/v3probe/cases_logs/smoke16_%j.err
# Production worker, one task, first 64 rows, 16 concurrent cases, v3-JSON + boundary flags.
# Purpose: exercise the JSON-schema sampler and the boundary flags under production concurrency
# before the 35k submission. Output goes to a pilot root, never the production root.
set -euo pipefail
export INPUT_TSV=/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv
export OUTPUT_ROOT=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_pilots/v3json-smoke16-$(date -u +%Y%m%dT%H%MZ)
export TOTAL_ROWS=64 NUM_TASKS=1 NUM_CONCURRENT_CASES=16 AMBIGUITY_TUNING_CONCURRENCY=16
export TARGETED_NUM_FUTURES=20 MIN_VOTERS_RATIO=1.0 FUTURE_SRC_WINDOW=1
export PROMPT_VERSION=future_set_v3_suffix_icl
export EXTRA_DECODER_ARGS="--targeted-sampler-context source-only --future-source-window-mode sentence-anchor --sentence-end-completion --sentence-end-boundary-mode conservative --sentence-end-punctuation match-source --targeted-prompt-version future_set_v3_suffix_icl"
export SLURM_ARRAY_TASK_ID=0
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
exec bash /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/run_ambiguity_q38_gemma_q36_worker.sh
