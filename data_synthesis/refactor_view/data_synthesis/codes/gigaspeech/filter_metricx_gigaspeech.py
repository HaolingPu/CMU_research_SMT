#!/usr/bin/env python3
"""
Filter MetricX prediction jsonl for GigaSpeech pipeline.
Keep entries with prediction <= threshold.
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict


def get_prediction(item: Dict[str, Any]) -> float:
    if "prediction" not in item:
        raise KeyError("missing prediction")
    return float(item["prediction"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter MetricX output by threshold.")
    parser.add_argument("--input", required=True, help="MetricX output jsonl path.")
    parser.add_argument("--output", required=True, help="Filtered output jsonl path.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Keep if prediction <= threshold.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0
    kept = 0
    removed = 0
    bad_line = 0

    kept_latency = Counter()
    removed_latency = Counter()

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
                pred = get_prediction(item)
                latency = (
                    item.get("metadata", {}).get("latency", "unknown")
                    if isinstance(item, dict)
                    else "unknown"
                )
            except Exception:
                bad_line += 1
                continue

            if pred <= args.threshold:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
                kept_latency[latency] += 1
            else:
                removed += 1
                removed_latency[latency] += 1

    print("========== MetricX Filter (GigaSpeech) ==========")
    print(f"Input      : {args.input}")
    print(f"Output     : {args.output}")
    print(f"Threshold  : {args.threshold}")
    print(f"Total lines: {total}")
    print(f"Kept       : {kept}")
    print(f"Removed    : {removed}")
    print(f"Bad lines  : {bad_line}")
    if kept_latency:
        print(f"Kept by latency   : {dict(kept_latency)}")
    if removed_latency:
        print(f"Removed by latency: {dict(removed_latency)}")
    print("===============================================")


if __name__ == "__main__":
    main()



# count 684361
# mean refined 5.637319132290635 mean baseline 4.263881392457108
# p50 refined 5.489077568054199 p50 baseline 4.04551362991333
# <3 refined 0.20398444680512187 <3 baseline 0.3223357263198809