#!/usr/bin/env bash
# 用 core_v2（simalign）跑一条 test-one，并打 verbose，方便看 word alignment 效果。
# Align 与 base 都在 GPU 0 上。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_TSV="${1:-}"
OUTPUT_ROOT="${2:-}"

if [[ -z "$INPUT_TSV" || -z "$OUTPUT_ROOT" ]]; then
  echo "Usage: $0 <input.tsv> <output_root>"
  echo "Example: $0 /path/to/MANIFEST.tsv /path/to/OUT"
  echo "Runs: CUDA_VISIBLE_DEVICES=0 python llm_future_sampling_majority_vote_v2.py --input-tsv <input> --output-root <output> --test-one --verbose"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/models}"

python llm_future_sampling_majority_vote_v2.py \
  --input-tsv "$INPUT_TSV" \
  --output-root "$OUTPUT_ROOT" \
  --test-one \
  --verbose

echo "Done. Check verbose log under output_root: verbose_<utt_id>.log"
