#!/usr/bin/env bash
# 用 simalign 逐个跑 4 个 selection-mode。
# 设置：
#   - disable sentence path
#   - verbose
#   - disable majority_vote backoff
#   - max_rows=2
#   - parallel_utterances=2
# 可选第 3 个参数传单个 mode，只跑该 mode。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_TSV="${1:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
OUTPUT_ROOT="${2:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/out_simalign_test_three}"
MODE_FILTER="${3:-all}"

if [[ ! -f "$INPUT_TSV" ]]; then
  echo "ERROR: input TSV not found: $INPUT_TSV"
  echo "Usage: $0 [input.tsv] [output_root] [mode|all]"
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/models}"

for MODE in majority_vote lcp70_llm lcp70_code lcp_code; do
  if [[ "$MODE_FILTER" != "all" && "$MODE_FILTER" != "$MODE" ]]; then
    continue
  fi
  OUT="$OUTPUT_ROOT/v2_verbose_${MODE}"
  mkdir -p "$OUT"
  echo "========== mode=$MODE -> $OUT =========="
  python llm_future_sampling_majority_vote_v2.py \
    --input-tsv "$INPUT_TSV" \
    --output-root "$OUT" \
    --verbose \
    --selection-mode "$MODE" \
    --consensus-ratio "0.7" \
    --max-rows 2 \
    --parallel-utterances 2 \
    --disable-sentence-path \
    --majority-vote-disable-backoff \
    --overwrite
  echo ""
done

echo "Done. Verbose logs: $OUTPUT_ROOT/v2_verbose_*/verbose_*.log"
echo "Input: $INPUT_TSV"
echo "Output root: $OUTPUT_ROOT"
