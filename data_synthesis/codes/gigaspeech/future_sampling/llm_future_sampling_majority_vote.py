#!/usr/bin/env python3
"""majority_vote entrypoint: semantic safe-prefix synthesis."""

import os
import sys


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    core = os.path.join(script_dir, "llm_future_sampling_core.py")
    argv = [
        sys.executable,
        core,
        *sys.argv[1:],
        "--selection-mode",
        "majority_vote",
        "--consensus-ratio",
        "0.7",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0 \
# SIMALIGN_MODEL=/data/user_data/haolingp/models/LaBSE \
# python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_majority_vote.py \
#   --input-tsv /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
#   --output-root /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/out_simalign_test_one/majority_vote \
#   --align-method simalign \
#   --test-one --verbose --overwrite \
#   --num-tasks 1 --parallel-utterances 1 \
#   --instruct-api-base http://localhost:8100/v1 \
#   --no-tee --utt-id AUD0000000003_0
