#!/usr/bin/env python3
"""
Convert streaming trajectory JSON files to MetricX QE input.jsonl.

Input:  recursively read *.json under --stream_dir
Output: one jsonl with fields expected by metricx24.predict --qe
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

from tqdm import tqdm


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def join_zh_segments(segments: List[str]) -> str:
    text = "".join(str(x) for x in segments).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def collect_json_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def detect_levels(data: Dict) -> List[Tuple[str, str, str]]:
    """
    Return tuples: (latency_name, source_key, target_key)
    """
    levels: List[Tuple[str, str, str]] = []

    # EAST 3-latency
    if "source_low_latency" in data and "target_low_latency" in data:
        levels.append(("low", "source_low_latency", "target_low_latency"))
    if "source_medium_latency" in data and "target_medium_latency" in data:
        levels.append(("medium", "source_medium_latency", "target_medium_latency"))
    if "source_high_latency" in data and "target_high_latency" in data:
        levels.append(("high", "source_high_latency", "target_high_latency"))

    # offline
    if "source_offline" in data and "target_offline" in data:
        levels.append(("offline", "source_offline", "target_offline"))

    return levels


def convert_dataset(stream_dir: str, output_file: str, keep_source_case: bool = False) -> None:
    files = collect_json_files(stream_dir)
    print(f"Found {len(files)} trajectory JSON files.")

    kept = 0
    skipped = 0
    skipped_error_json = 0
    skipped_no_levels = 0
    skipped_bad_source = 0

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        for path in tqdm(files, desc="Converting"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                skipped += 1
                continue

            if not isinstance(data, dict):
                skipped += 1
                continue

            if "error" in data:
                skipped_error_json += 1
                skipped += 1
                continue

            utt_id = str(data.get("utt_id", "")).strip()
            if not utt_id:
                skipped += 1
                continue

            source_raw = str(data.get("original_text", data.get("input", ""))).strip()
            if not source_raw:
                skipped_bad_source += 1
                skipped += 1
                continue

            source_text = source_raw if keep_source_case else normalize_text(source_raw)

            levels = detect_levels(data)
            if not levels:
                skipped_no_levels += 1
                skipped += 1
                continue

            for latency, _, tgt_key in levels:
                target_segments = data.get(tgt_key, [])
                if not isinstance(target_segments, list) or len(target_segments) == 0:
                    continue

                hypothesis = join_zh_segments(target_segments)
                if not hypothesis:
                    continue

                item = {
                    "source": source_text,
                    "hypothesis": hypothesis,
                    "reference": "",
                    "metadata": {
                        "utt_id": utt_id,
                        "latency": latency,
                        "num_segments": len(target_segments),
                        "source_raw": source_raw,
                        "stream_json": path,
                    },
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1

    print("\n==============================")
    print("MetricX conversion done")
    print("==============================")
    print(f"Output file          : {output_file}")
    print(f"Kept examples        : {kept}")
    print(f"Skipped files total  : {skipped}")
    print(f"  - error json       : {skipped_error_json}")
    print(f"  - no latency keys  : {skipped_no_levels}")
    print(f"  - empty source     : {skipped_bad_source}")
    print("==============================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert streaming dataset to MetricX QE input.")
    parser.add_argument("--stream_dir", required=True, help="Root directory of streaming trajectory JSONs.")
    parser.add_argument("--output", required=True, help="Output MetricX input jsonl.")
    parser.add_argument(
        "--keep-source-case",
        action="store_true",
        help="Do not lowercase/normalize source text.",
    )
    args = parser.parse_args()
    convert_dataset(args.stream_dir, args.output, keep_source_case=args.keep_source_case)


if __name__ == "__main__":
    main()