#!/usr/bin/env python3
"""Randomly sample n rows from a JSONL file."""

import argparse
import random

"""
DATA_DIR=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests
python sample_jsonl.py 12500 ${DATA_DIR}/train_xl_case_robust_asr-filtered_zh-Simul-MuST-C_fixed_v2.jsonl ${DATA_DIR}/train_s_zh-Simul-MuST-C-fixed_v2_origin.jsonl
python sample_jsonl.py 12500 ${DATA_DIR}/train_xl_case_robust_asr-filtered_zh-EAST-latency2mult.jsonl ${DATA_DIR}/train_s_zh-EAST-latency2mult_origin.jsonl
python sample_jsonl.py 12500 ${DATA_DIR}/train_xl_case_robust_asr-filtered_zh-refined-EAST-latency2mult.jsonl ${DATA_DIR}/train_s_zh-refined-EAST-latency2mult_origin.jsonl
"""

def main():
    parser = argparse.ArgumentParser(description="Randomly sample n rows from a JSONL file.")
    parser.add_argument("n", type=int, help="Number of rows to sample")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    if args.n > len(lines):
        print(f"Warning: requested {args.n} rows but file only has {len(lines)} rows. Writing all rows.")
        sampled = lines
    else:
        random.seed(args.seed)
        sampled = random.sample(lines, args.n)

    with open(args.output_file, "w") as f:
        f.writelines(sampled)

    print(f"Sampled {len(sampled)} rows from {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
    main()
