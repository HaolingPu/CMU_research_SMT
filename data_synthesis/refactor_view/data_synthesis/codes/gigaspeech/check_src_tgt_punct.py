#!/usr/bin/env python3
"""
Check punctuation correspondence between source (English) and target (Chinese)
in final_jsonl_salami entries.

Checks:
  1. Source/target array length equality (alignment)
  2. Sentence-ending marks: English [.?!] vs Chinese [。？！]
  3. Comma/clause marks:    English [,;] vs Chinese [，；]
  4. Per-entry flag if Chinese has zero sentence-enders but English has >=2

Usage:
  python check_src_tgt_punct.py \
    --final_dir .../final_jsonl_salami_patched \
    --report    ./check_src_tgt_punct_report.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Punctuation sets
# ---------------------------------------------------------------------------
EN_SENT_END  = set('.?!')       # English sentence-ending
ZH_SENT_END  = set('。？！')    # Chinese sentence-ending
EN_CLAUSE    = set(',;')        # English clause
ZH_CLAUSE    = set('，；')      # Chinese clause
EN_COLON     = set(':')
ZH_COLON     = set('：')

def count_punct(text: str, charset: set) -> int:
    return sum(1 for ch in text if ch in charset)

def all_punct_counts(text: str) -> Counter:
    return Counter(ch for ch in text if not ch.isalnum() and not ch.isspace())

def join_src(segs):
    return " ".join(s for s in segs if s)

def join_tgt(segs):
    return "".join(s for s in segs if s)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir", required=True)
    ap.add_argument("--report",    default="check_src_tgt_punct_report.json")
    args = ap.parse_args()

    total = 0
    len_mismatch = []

    # sentence-end mismatch stats
    sent_end_zero_tgt   = []   # English has >=2 sent-enders, Chinese has 0
    sent_end_big_diff   = []   # |en_sent - zh_sent| >= 3
    # overall distributions
    en_sent_counts = Counter()
    zh_sent_counts = Counter()
    en_clause_counts = Counter()
    zh_clause_counts = Counter()
    diff_sent_end_dist = Counter()   # (en - zh) distribution

    for dirpath, _, files in os.walk(args.final_dir):
        for fn in sorted(files):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    uid = d.get("utt_id", "")
                    src_segs = d.get("source", [])
                    tgt_segs = d.get("target", [])
                    total += 1

                    # --- check 1: array length ---
                    if len(src_segs) != len(tgt_segs):
                        len_mismatch.append({
                            "utt_id": uid,
                            "src_len": len(src_segs),
                            "tgt_len": len(tgt_segs),
                        })
                        continue

                    src_text = join_src(src_segs)
                    tgt_text = join_tgt(tgt_segs)

                    en_sent  = count_punct(src_text, EN_SENT_END)
                    zh_sent  = count_punct(tgt_text, ZH_SENT_END)
                    en_clause = count_punct(src_text, EN_CLAUSE)
                    zh_clause = count_punct(tgt_text, ZH_CLAUSE)

                    en_sent_counts[en_sent] += 1
                    zh_sent_counts[zh_sent] += 1
                    en_clause_counts[en_clause] += 1
                    zh_clause_counts[zh_clause] += 1
                    diff_sent_end_dist[en_sent - zh_sent] += 1

                    # --- flag: English has >=2 sentence-enders but Chinese has 0 ---
                    if en_sent >= 2 and zh_sent == 0:
                        sent_end_zero_tgt.append({
                            "utt_id": uid,
                            "en_sent_end": en_sent,
                            "zh_sent_end": zh_sent,
                            "src": src_text[:120],
                            "tgt": tgt_text[:120],
                        })

                    # --- flag: big absolute difference ---
                    if abs(en_sent - zh_sent) >= 3:
                        sent_end_big_diff.append({
                            "utt_id": uid,
                            "en_sent_end": en_sent,
                            "zh_sent_end": zh_sent,
                            "diff": en_sent - zh_sent,
                            "src": src_text[:120],
                            "tgt": tgt_text[:120],
                        })

    # --- print summary ---
    print("\n" + "=" * 60)
    print("SOURCE / TARGET PUNCTUATION CHECK")
    print("=" * 60)
    print(f"\nTotal entries: {total}")

    print(f"\n--- Array length alignment ---")
    print(f"  Mismatched: {len(len_mismatch)}")

    print(f"\n--- Sentence-ending marks ---")
    print(f"  English [.?!] counts (top 10 values):")
    for cnt, freq in sorted(en_sent_counts.items())[:10]:
        print(f"    en_sent={cnt}: {freq} entries")
    print(f"  Chinese [。？！] counts (top 10 values):")
    for cnt, freq in sorted(zh_sent_counts.items())[:10]:
        print(f"    zh_sent={cnt}: {freq} entries")
    print(f"\n  Difference distribution (en - zh), top 15:")
    for diff, freq in sorted(diff_sent_end_dist.items(), key=lambda x: -x[1])[:15]:
        print(f"    diff={diff:+d}: {freq} entries")

    print(f"\n  Flagged: English >=2 sent-ends but Chinese = 0: {len(sent_end_zero_tgt)}")
    if sent_end_zero_tgt:
        for ex in sent_end_zero_tgt[:5]:
            print(f"    {ex['utt_id']}  en={ex['en_sent_end']} zh={ex['zh_sent_end']}")
            print(f"      src: {repr(ex['src'])}")
            print(f"      tgt: {repr(ex['tgt'])}")

    print(f"\n  Flagged: |en - zh| >= 3: {len(sent_end_big_diff)}")
    if sent_end_big_diff:
        for ex in sent_end_big_diff[:5]:
            print(f"    {ex['utt_id']}  en={ex['en_sent_end']} zh={ex['zh_sent_end']} diff={ex['diff']:+d}")
            print(f"      src: {repr(ex['src'])}")
            print(f"      tgt: {repr(ex['tgt'])}")

    print(f"\n--- Comma/clause marks ---")
    print(f"  English [,;] counts (top 8):")
    for cnt, freq in sorted(en_clause_counts.items())[:8]:
        print(f"    en_clause={cnt}: {freq} entries")
    print(f"  Chinese [，；] counts (top 8):")
    for cnt, freq in sorted(zh_clause_counts.items())[:8]:
        print(f"    zh_clause={cnt}: {freq} entries")

    # --- write report ---
    report = {
        "total": total,
        "len_mismatch_count": len(len_mismatch),
        "len_mismatch_examples": len_mismatch[:20],
        "sent_end_zero_tgt_count": len(sent_end_zero_tgt),
        "sent_end_zero_tgt_examples": sent_end_zero_tgt[:50],
        "sent_end_big_diff_count": len(sent_end_big_diff),
        "sent_end_big_diff_examples": sent_end_big_diff[:50],
        "diff_sent_end_distribution": {str(k): v for k, v in sorted(diff_sent_end_dist.items())},
        "en_sent_end_distribution": {str(k): v for k, v in sorted(en_sent_counts.items())},
        "zh_sent_end_distribution": {str(k): v for k, v in sorted(zh_sent_counts.items())},
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
