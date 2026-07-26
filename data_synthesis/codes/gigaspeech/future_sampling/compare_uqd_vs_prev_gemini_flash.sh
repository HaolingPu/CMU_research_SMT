#!/usr/bin/env bash
set -euo pipefail

CANDIDATE_DIR="${1:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/gemini_flash_json_pro_fallback_uqd_100}"
BASELINE_DIR="${2:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini/metricx/gemini3_advanced_high_100_metricx_output.jsonl}"
COMPARE_SCRIPT="/data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/compare_metricx_qe.py"

python "${COMPARE_SCRIPT}"           --baseline "${BASELINE_DIR}"           --candidate "${CANDIDATE_DIR}"           --latency future_sampling           --threshold 3.0           --top-k 15
