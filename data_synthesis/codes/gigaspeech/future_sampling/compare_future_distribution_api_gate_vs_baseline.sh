#!/usr/bin/env bash
set -e

OUT="${1:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini_future_distribution_api_gate_100}"
BASELINE="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/metricx/gemini3_advanced_high_100_metricx_output.jsonl"

python /home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/compare_metricx_qe.py   --baseline "${BASELINE}"   --candidate "${OUT}"   --latency future_sampling   --threshold 3.0   --top-k 15
