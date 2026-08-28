#!/usr/bin/env python3
"""
Check that final_jsonl_salami matches metricx_filtered_t3.0.jsonl (the manifest).

Checks:
  1. utt_id sets are identical (no missing / extra)
  2. For each matched pair:
     - joined target (non-empty segs) == manifest hypothesis
     - joined source (non-empty segs) == manifest source_raw  (optional, reported separately)

Reports all mismatches to stdout and writes a JSON summary.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

MANIFEST = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/metricx_filtered_t3.0.jsonl"
FINAL_DIR = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/final_jsonl_salami"
REPORT_OUT = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_salami/check_final_vs_manifest.json"


def join_segs(segs):
    return "".join(s for s in segs if s)


def load_manifest(path):
    """Returns dict: utt_id -> {hypothesis, source_raw, source_lower}"""
    print(f"Loading manifest: {path}")
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            meta = d.get("metadata", {})
            uid = meta.get("utt_id", "").strip()
            if not uid:
                continue
            index[uid] = {
                "hypothesis": d.get("hypothesis", ""),
                "source_raw": meta.get("source_raw", ""),
                "source_lower": d.get("source", ""),
            }
    print(f"  Loaded {len(index)} manifest entries")
    return index


def load_final_jsonl(root):
    """Returns dict: utt_id -> {source_joined, target_joined}"""
    print(f"Loading final jsonl from: {root}")
    index = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    uid = d.get("utt_id", "").strip()
                    if not uid:
                        continue
                    index[uid] = {
                        "source_joined": join_segs(d.get("source", [])),
                        "target_joined": join_segs(d.get("target", [])),
                        "source_raw_segs": d.get("source", []),
                        "target_raw_segs": d.get("target", []),
                    }
    print(f"  Loaded {len(index)} final jsonl entries")
    return index


def main():
    manifest = load_manifest(MANIFEST)
    final = load_final_jsonl(FINAL_DIR)

    manifest_ids = set(manifest.keys())
    final_ids = set(final.keys())

    missing_in_final = sorted(manifest_ids - final_ids)
    extra_in_final = sorted(final_ids - manifest_ids)

    print(f"\n=== utt_id coverage ===")
    print(f"  manifest:        {len(manifest_ids)}")
    print(f"  final jsonl:     {len(final_ids)}")
    print(f"  missing in final (in manifest but not final): {len(missing_in_final)}")
    print(f"  extra in final   (in final but not manifest): {len(extra_in_final)}")

    # Compare content for matching IDs
    common_ids = sorted(manifest_ids & final_ids)
    target_mismatches = []
    source_mismatches = []

    for uid in common_ids:
        m = manifest[uid]
        f = final[uid]

        # Target vs hypothesis
        if f["target_joined"] != m["hypothesis"]:
            target_mismatches.append({
                "utt_id": uid,
                "manifest_hypothesis": m["hypothesis"],
                "final_target_joined": f["target_joined"],
                "final_target_segs": f["target_raw_segs"],
            })

        # Source vs source_raw (case-insensitive because manifest 'source' is lowercased)
        # Compare against source_raw with punctuation stripped from both
        # We'll just report length and content differences
        src_joined = f["source_joined"]
        src_raw = m["source_raw"]
        if src_joined != src_raw:
            source_mismatches.append({
                "utt_id": uid,
                "manifest_source_raw": src_raw,
                "final_source_joined": src_joined,
            })

    print(f"\n=== Content comparison (among {len(common_ids)} common entries) ===")
    print(f"  target mismatches (joined target != hypothesis): {len(target_mismatches)}")
    print(f"  source mismatches (joined source != source_raw): {len(source_mismatches)}")

    # Analyze target mismatches by type
    punct_only_diffs = []
    other_diffs = []
    for mm in target_mismatches:
        hyp = mm["manifest_hypothesis"]
        tgt = mm["final_target_joined"]
        # Check if they're the same when stripping common Chinese punctuation
        import re
        punct_re = re.compile(r'[，。！？；：、,.!?;:]')
        if punct_re.sub('', hyp) == punct_re.sub('', tgt):
            punct_only_diffs.append(mm)
        else:
            other_diffs.append(mm)

    print(f"\n  Of target mismatches:")
    print(f"    punctuation-only differences: {len(punct_only_diffs)}")
    print(f"    other differences:            {len(other_diffs)}")

    # Show first 5 examples of each type
    if punct_only_diffs:
        print(f"\n--- First 5 punctuation-only target diffs ---")
        for ex in punct_only_diffs[:5]:
            print(f"  utt_id: {ex['utt_id']}")
            print(f"  manifest: {ex['manifest_hypothesis'][:120]}")
            print(f"  final:    {ex['final_target_joined'][:120]}")
            print()

    if other_diffs:
        print(f"\n--- First 5 other target diffs ---")
        for ex in other_diffs[:5]:
            print(f"  utt_id: {ex['utt_id']}")
            print(f"  manifest: {ex['manifest_hypothesis'][:120]}")
            print(f"  final:    {ex['final_target_joined'][:120]}")
            print()

    if missing_in_final[:5]:
        print(f"\n--- First 5 missing in final ---")
        for uid in missing_in_final[:5]:
            print(f"  {uid}")

    if extra_in_final[:5]:
        print(f"\n--- First 5 extra in final ---")
        for uid in extra_in_final[:5]:
            print(f"  {uid}")

    # Write report
    report = {
        "manifest_count": len(manifest_ids),
        "final_count": len(final_ids),
        "missing_in_final_count": len(missing_in_final),
        "extra_in_final_count": len(extra_in_final),
        "missing_in_final": missing_in_final[:100],
        "extra_in_final": extra_in_final[:100],
        "target_mismatch_count": len(target_mismatches),
        "target_punct_only_diff_count": len(punct_only_diffs),
        "target_other_diff_count": len(other_diffs),
        "source_mismatch_count": len(source_mismatches),
        "target_punct_only_examples": punct_only_diffs[:20],
        "target_other_examples": other_diffs[:20],
        "source_mismatch_examples": source_mismatches[:20],
    }
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report written to: {REPORT_OUT}")


if __name__ == "__main__":
    main()





# python codes/gigaspeech/check_salami_final.py \
#   --tsv       /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
#   --manifest  outputs/gigaspeech/train_xl_salami/metricx_filtered_t3.0.jsonl \
#   --final_dir outputs/gigaspeech/train_xl_salami/final_jsonl_salami_patched \
#   --report    outputs/gigaspeech/train_xl_salami/check_salami_final_report.json



# python codes/gigaspeech/check_salami_final.py \
#   --tsv       /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv \
#   --manifest  outputs/gigaspeech/train_xl_salami/metricx_filtered_t3.0.jsonl \
#   --final_dir /data/group_data/li_lab/haolingp/data_synthesis/gigaspeech/final_jsonl_salami \
#   --report    outputs/gigaspeech/train_xl_salami/check_salami_final_report.json
