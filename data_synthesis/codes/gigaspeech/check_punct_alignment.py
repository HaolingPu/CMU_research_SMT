#!/usr/bin/env python3
"""
Check punctuation alignment between English source and Chinese target
in llm_output_raw_fixed JSONs.

Works for both salami (segmented_pairs) and EAST (low_latency/... keys) formats.

Usage:
  python check_punct_alignment.py \
    --input_dir .../llm_output_raw_fixed \
    [--bad_jsonl ./bad_punct.jsonl] \
    [--sample N]            # only check first N files (for quick spot-check)
    [--threshold 1]         # max allowed |en_count - zh_count|, default 1
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Punct counting
# ---------------------------------------------------------------------------

EN_SENT_END = re.compile(r'[.!?]')
ZH_SENT_END = re.compile(r'[。！？]')

# Also track clause-level for reference
EN_CLAUSE = re.compile(r'[,;:]')
ZH_CLAUSE = re.compile(r'[，；：、]')


def count_punct(text: str, pattern: re.Pattern) -> int:
    return len(pattern.findall(text))


# ---------------------------------------------------------------------------
# Format detection & extraction
# ---------------------------------------------------------------------------

LATENCY_LEVELS = ["low_latency", "medium_latency", "high_latency", "offline"]


def extract_salami(data: dict) -> Optional[Tuple[str, str]]:
    """Returns (joined_en, joined_zh) from segmented_pairs, or None."""
    pairs = data.get("segmented_pairs")
    if not isinstance(pairs, list):
        return None
    eng_parts, zh_parts = [], []
    for p in pairs:
        if isinstance(p, list) and len(p) == 2:
            e, z = str(p[0]).strip(), str(p[1]).strip()
            if e or z:
                eng_parts.append(e)
                zh_parts.append(z)
    if not eng_parts:
        return None
    return " ".join(eng_parts), "".join(zh_parts)


def extract_east(data: dict) -> Optional[Tuple[str, str]]:
    """Returns (joined_en, joined_zh) from first available latency level, or None."""
    for level in LATENCY_LEVELS:
        obj = data.get(level)
        if not isinstance(obj, dict):
            continue
        eng = obj.get("English", [])
        zh = obj.get("Chinese", [])
        if isinstance(eng, list) and isinstance(zh, list) and eng:
            return " ".join(str(x).strip() for x in eng), "".join(str(x).strip() for x in zh)
    return None


def extract_texts(data: dict) -> Optional[Tuple[str, str]]:
    if "segmented_pairs" in data:
        return extract_salami(data)
    for level in LATENCY_LEVELS:
        if level in data:
            return extract_east(data)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--bad_jsonl", default=None, help="Write bad utterances to this JSONL.")
    p.add_argument("--sample", type=int, default=0,
                   help="Only check first N files (0 = all).")
    p.add_argument("--threshold", type=int, default=1,
                   help="Max allowed |en_sent_end - zh_sent_end| before flagging. Default=1.")
    return p.parse_args()


def list_json_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def main():
    args = parse_args()

    files = list_json_files(args.input_dir)
    if not files:
        print(f"No JSON files found under: {args.input_dir}")
        return
    if args.sample > 0:
        files = files[:args.sample]

    bad_fp = open(args.bad_jsonl, "w", encoding="utf-8") if args.bad_jsonl else None

    total = 0
    skipped_error = 0
    skipped_no_text = 0
    flagged = 0
    diff_counter: Counter = Counter()  # en_count - zh_count
    zh_zero_with_en_pos = 0  # EN has >=1 sent-end but ZH has 0

    for path in tqdm(files, desc="Checking"):
        total += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped_error += 1
            continue

        if not isinstance(data, dict) or "error" in data:
            skipped_error += 1
            continue

        result = extract_texts(data)
        if result is None:
            skipped_no_text += 1
            continue

        joined_en, joined_zh = result

        en_sent = count_punct(joined_en, EN_SENT_END)
        zh_sent = count_punct(joined_zh, ZH_SENT_END)
        diff = en_sent - zh_sent
        diff_counter[diff] += 1

        if en_sent >= 1 and zh_sent == 0:
            zh_zero_with_en_pos += 1

        if abs(diff) > args.threshold:
            flagged += 1
            if bad_fp:
                bad_fp.write(json.dumps({
                    "utt_id": data.get("utt_id", os.path.splitext(os.path.basename(path))[0]),
                    "file": path,
                    "en_sent_end": en_sent,
                    "zh_sent_end": zh_sent,
                    "diff": diff,
                    "en_text": joined_en[:200],
                    "zh_text": joined_zh[:200],
                }, ensure_ascii=False) + "\n")

    if bad_fp:
        bad_fp.close()

    print("\n========== Punctuation Alignment Check ==========")
    print(f"Input dir         : {args.input_dir}")
    print(f"Threshold         : |diff| > {args.threshold}")
    print(f"Total files       : {total}")
    print(f"Skipped (error)   : {skipped_error}")
    print(f"Skipped (no text) : {skipped_no_text}")
    checked = total - skipped_error - skipped_no_text
    print(f"Checked           : {checked}")
    print(f"Flagged (|diff|>{args.threshold}) : {flagged}  ({100*flagged/checked:.1f}% of checked)" if checked else "")
    print(f"ZH has 0 sent-end (EN>=1) : {zh_zero_with_en_pos}  ({100*zh_zero_with_en_pos/checked:.1f}%)" if checked else "")
    print("\nDiff distribution (EN_sent_end - ZH_sent_end):")
    for k in sorted(diff_counter.keys()):
        bar = "#" * min(50, diff_counter[k] * 50 // max(diff_counter.values()))
        print(f"  {k:+4d} : {diff_counter[k]:8d}  {bar}")
    if args.bad_jsonl and flagged:
        print(f"\nBad utterances written to: {args.bad_jsonl}")


if __name__ == "__main__":
    main()
