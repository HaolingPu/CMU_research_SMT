#!/usr/bin/env bash
# ============================================================
# 70k consensus-decoding run with top-p = 0.9 nucleus candidate selection.
# Same setup as the top_5_v2 run: 4 array tasks × 2 L40S, concurrent=48,
# max_consensus_steps=12, future_source_window_chunks=3, min_p=0.0.
# Output: /data/.../topp/consensus_decoding_en_zh_top_p_0.9_v2/
# Pipeline: gen -> per-sentence prepare -> MetricX(8 shards) -> QE<=3 AND -> length filter.
# ============================================================
set -e

MAIN_SUBMIT="/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling/submit_consensus_dualbase_60k_per_sentence_qe3_len.sh"

EXP_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topp/consensus_decoding_en_zh_top_p_0.9_v2" \
TOTAL_ROWS=70000 \
ROW_OFFSET=0 \
TOP_P=0.9 \
CANDIDATE_TOP_K=5 \
MAX_CONSENSUS_STEPS=12 \
FUTURE_SOURCE_WINDOW_CHUNKS=3 \
MIN_P=0.0 \
NUM_TASKS=4 \
NUM_CONCURRENT_CASES=48 \
NUM_FUTURES=20 \
SECONDARY_NUM_FUTURES=10 \
FUTURE_TOKENS=20 \
NUM_QE_SHARDS=8 \
QE_THRESHOLD=3.0 \
MAX_RATIO_REF=1.5 \
MAX_RATIO_SRC=2.5 \
GEN_TIME_LIMIT=2-00:00:00 \
  bash "${MAIN_SUBMIT}"
