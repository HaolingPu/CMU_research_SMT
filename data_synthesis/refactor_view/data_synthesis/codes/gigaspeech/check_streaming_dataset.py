#!/usr/bin/env python3
"""
check_streaming_dataset.py

Quality check on the streaming trajectory dataset
(output of multi_trajectory_gigaspeech.py).

Handles both salami format (source_offline / target_offline)
and EAST format (source_low_latency, source_medium_latency, source_high_latency, ...).

Checks per utterance:
  1. SOURCE TEXT  : joined(source) == TSV src_text_full
                    (uses offline for salami, high_latency for EAST)
  2. SRC/TGT LEN  : source and target arrays have equal length
  3. SRC/TGT PUNCT: sentence-ending marks EN [.?!] vs ZH [。？！]
                      - zh_missing  : en_sent >= 2 and zh_sent == 0
                      - large_diff  : |en_sent - zh_sent| >= 3
                      - zh_excess   : zh_sent - en_sent >= zh_excess_threshold
  4. EMPTY SEGS   : flag if source has many consecutive empty segments

Usage:
  python check_streaming_dataset.py \
    --stream_dir  .../streaming_salami_dataset \
    --tsv         .../train_xl_case_robust_asr-filtered.tsv \
    --report      ./check_streaming_report.json \
    [--sample N]  [--zh_excess_threshold 2]
"""

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter
from typing import Optional, Tuple

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

EN_SENT_END = set('.?!')
ZH_SENT_END = set('。？！')
EN_CLAUSE   = set(',;')
ZH_CLAUSE   = set('，；')

# Preference order when picking the key for source-text comparison
# (we want the most complete/full-sentence version)
PREFERRED_SUFFIXES = ["offline", "high_latency", "medium_latency", "low_latency"]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def normalize_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())


def count_chars(text: str, charset: set) -> int:
    return sum(1 for c in text if c in charset)


def join_source(segs) -> str:
    return normalize_ws(" ".join(s for s in (segs or []) if s))


def join_target(segs) -> str:
    return "".join(s for s in (segs or []) if s)


def parse_src_text_full(raw_value) -> str:
    """Join src_text_full list into a single string."""
    if not raw_value:
        return ""
    raw = str(raw_value).strip()
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = raw
    if isinstance(parsed, list):
        sentences = [str(x).strip() for x in parsed if str(x).strip()]
    else:
        sentences = [str(parsed).strip()] if str(parsed).strip() else []
    return " ".join(sentences).strip()


def normalize_for_token_cmp(s: str) -> str:
    """Case-insensitive, punct-stripped token comparison."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ──────────────────────────────────────────────
# TSV loading
# ──────────────────────────────────────────────

def load_tsv(tsv_path: str) -> dict:
    """Returns {utt_id -> src_text_full_joined}."""
    index = {}
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            uid = row.get('id', '').strip()
            if uid:
                index[uid] = parse_src_text_full(row.get('src_text_full', ''))
    return index


# ──────────────────────────────────────────────
# Per-utterance check
# ──────────────────────────────────────────────

def detect_suffix(d: dict) -> Optional[str]:
    """Return the best source/target suffix to use for checks."""
    for suf in PREFERRED_SUFFIXES:
        if f"source_{suf}" in d:
            return suf
    return None


def check_one(
    d: dict,
    tsv_src: Optional[str],
    zh_excess_threshold: int,
) -> dict:
    utt_id = d.get("utt_id", "")
    result = {
        "utt_id":    utt_id,
        "issues":    [],
        "src_ok":    True,
        "punct_ok":  True,
        "en_sent":   0,
        "zh_sent":   0,
        "suf":       None,
    }

    suf = detect_suffix(d)
    if suf is None:
        result["issues"].append("no_source_key")
        result["src_ok"] = False
        return result

    result["suf"] = suf
    src_segs = d.get(f"source_{suf}", [])
    tgt_segs = d.get(f"target_{suf}", [])

    src_joined = join_source(src_segs)
    tgt_joined = join_target(tgt_segs)

    # ── Check 1: source text vs TSV ──
    if tsv_src is not None:
        tsv_norm = normalize_ws(tsv_src)
        if src_joined == tsv_norm:
            pass  # exact match
        elif src_joined.lower() == tsv_norm.lower():
            result["issues"].append("src_case_diff")
        elif normalize_for_token_cmp(src_joined) == normalize_for_token_cmp(tsv_norm):
            result["issues"].append("src_punct_diff_only")
        else:
            result["issues"].append("src_text_mismatch")
            result["src_ok"] = False
            result["src_joined"] = src_joined[:120]
            result["tsv_src"]   = tsv_norm[:120]
    elif not src_joined:
        result["issues"].append("src_empty")
        result["src_ok"] = False

    # ── Check 2: array length equality ──
    if len(src_segs) != len(tgt_segs):
        result["issues"].append(f"len_mismatch:src={len(src_segs)},tgt={len(tgt_segs)}")

    # ── Check 3: punct alignment ──
    en_sent = count_chars(src_joined, EN_SENT_END)
    zh_sent = count_chars(tgt_joined, ZH_SENT_END)
    result["en_sent"] = en_sent
    result["zh_sent"] = zh_sent

    if en_sent >= 2 and zh_sent == 0:
        result["issues"].append(f"zh_missing:en={en_sent},zh=0")
        result["punct_ok"] = False
        result["src_sample"] = src_joined[:100]
        result["tgt_sample"] = tgt_joined[:100]
    elif abs(en_sent - zh_sent) >= 3:
        result["issues"].append(f"punct_large_diff:en={en_sent},zh={zh_sent},d={en_sent-zh_sent:+d}")
        result["punct_ok"] = False
        result["src_sample"] = src_joined[:100]
        result["tgt_sample"] = tgt_joined[:100]
    elif zh_sent - en_sent >= zh_excess_threshold:
        result["issues"].append(f"zh_excess:zh={zh_sent},en={en_sent},excess={zh_sent-en_sent}")
        result["punct_ok"] = False
        result["src_sample"] = src_joined[:100]
        result["tgt_sample"] = tgt_joined[:100]

    # ── Check 4: suspicious empty source segments ──
    # In streaming format, leading empty segments are normal (t=0 window).
    # Flag only if the source is pathologically empty: ALL segments are empty,
    # OR if there's no non-empty segment at all (nothing was emitted).
    non_empty_src = sum(1 for s in src_segs if s)
    if non_empty_src == 0 and src_segs:
        result["issues"].append("src_all_empty")

    return result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def list_json_files(root: str):
    out = []
    for dirpath, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Quality check on streaming trajectory dataset."
    )
    ap.add_argument("--stream_dir",          required=True,
                    help="Streaming dataset directory.")
    ap.add_argument("--tsv",                 default=None,
                    help="TSV manifest for source text check (optional but recommended).")
    ap.add_argument("--report",              default="check_streaming_report.json",
                    help="Output report JSON path.")
    ap.add_argument("--sample",              type=int, default=0,
                    help="Only check first N files (0 = all).")
    ap.add_argument("--zh_excess_threshold", type=int, default=2,
                    help="Flag when zh_sent - en_sent >= this (default 2).")
    args = ap.parse_args()

    # Load TSV index
    tsv_index = {}
    if args.tsv:
        print(f"Loading TSV: {args.tsv}")
        tsv_index = load_tsv(args.tsv)
        print(f"  {len(tsv_index):,} entries")

    # List files
    files = list_json_files(args.stream_dir)
    if not files:
        print(f"No JSON files found under: {args.stream_dir}")
        return
    if args.sample > 0:
        files = files[:args.sample]
    print(f"Checking {len(files):,} files ...")

    # Counters
    total          = 0
    error_load     = 0
    no_tsv_entry   = 0
    src_exact      = 0
    src_case_diff  = 0
    src_punct_diff = 0
    src_mismatch   = 0
    src_empty      = 0
    len_mismatch   = 0
    punct_missing  = 0
    punct_excess   = 0
    punct_large    = 0
    empty_seg_warn = 0

    en_sent_dist   = Counter()
    zh_sent_dist   = Counter()
    diff_dist      = Counter()

    examples_src   = []
    examples_punct = []

    for path in files:
        total += 1
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            error_load += 1
            continue

        if not isinstance(d, dict) or "error" in d:
            error_load += 1
            continue

        utt_id  = d.get("utt_id", os.path.splitext(os.path.basename(path))[0])
        tsv_src = tsv_index.get(utt_id) if tsv_index else None
        if tsv_index and tsv_src is None:
            no_tsv_entry += 1

        r = check_one(d, tsv_src, args.zh_excess_threshold)

        # Tally source check
        issues_set = set(r["issues"])
        if "no_source_key" in issues_set or "src_empty" in issues_set:
            src_empty += 1
        elif "src_text_mismatch" in issues_set:
            src_mismatch += 1
            if len(examples_src) < 30:
                examples_src.append({
                    "utt_id": utt_id,
                    "src":    r.get("src_joined", "")[:120],
                    "tsv":    r.get("tsv_src", "")[:120],
                })
        elif "src_case_diff" in issues_set:
            src_case_diff += 1
        elif "src_punct_diff_only" in issues_set:
            src_punct_diff += 1
        else:
            src_exact += 1

        # Tally length
        if any(i.startswith("len_mismatch") for i in r["issues"]):
            len_mismatch += 1

        # Tally punct
        en_sent = r["en_sent"]
        zh_sent = r["zh_sent"]
        en_sent_dist[en_sent] += 1
        zh_sent_dist[zh_sent] += 1
        diff_dist[en_sent - zh_sent] += 1

        if any(i.startswith("zh_missing") for i in r["issues"]):
            punct_missing += 1
            if len(examples_punct) < 30:
                suf = r.get("suf", "offline")
                examples_punct.append({
                    "utt_id": utt_id,
                    "issue":  next(i for i in r["issues"] if i.startswith("zh_missing")),
                    "src":    r.get("src_sample", "")[:100],
                    "tgt":    r.get("tgt_sample", "")[:100],
                })
        if any(i.startswith("zh_excess") for i in r["issues"]):
            punct_excess += 1
        if any(i.startswith("punct_large_diff") for i in r["issues"]):
            punct_large += 1

        # Empty seg
        if any(i.startswith("src_all_empty") for i in r["issues"]):
            empty_seg_warn += 1

    # ──── Print summary ────
    checked = total - error_load
    punct_bad = punct_missing + punct_excess + punct_large

    print("\n" + "=" * 68)
    print("STREAMING DATASET QUALITY CHECK")
    print("=" * 68)
    print(f"  Stream dir          : {args.stream_dir}")
    print(f"  Total files         : {total:,}")
    print(f"  Load errors         : {error_load:,}")
    print(f"  Checked             : {checked:,}")

    print(f"\n--- Source text vs TSV src_text_full ---")
    if tsv_index:
        print(f"  No TSV entry        : {no_tsv_entry:,}")
        print(f"  Exact match         : {src_exact:,}")
        print(f"  Case-only diff      : {src_case_diff:,}")
        print(f"  Punct-only diff     : {src_punct_diff:,}")
        print(f"  Content mismatch    : {src_mismatch:,}  ({100*src_mismatch/max(checked,1):.1f}%)")
        print(f"  Empty/missing src   : {src_empty:,}")
    else:
        print(f"  (no TSV provided — skipped)")

    print(f"\n--- Source/target array length ---")
    print(f"  Length mismatch     : {len_mismatch:,}")

    print(f"\n--- Source/target punctuation alignment ---")
    print(f"  Any punct issue     : {punct_bad:,}  ({100*punct_bad/max(checked,1):.1f}%)")
    print(f"    ZH missing        : {punct_missing:,}  (en>=2 & zh=0)")
    print(f"    ZH excess         : {punct_excess:,}  (zh-en>={args.zh_excess_threshold})")
    print(f"    Large diff        : {punct_large:,}  (|en-zh|>=3)")

    print(f"\n  en_sent - zh_sent distribution (top 15 by frequency):")
    for k, v in sorted(diff_dist.items(), key=lambda x: -x[1])[:15]:
        bar = '#' * min(40, v * 40 // max(diff_dist.values(), default=1))
        print(f"    {k:+4d} : {v:8,}  {bar}")

    print(f"\n--- Other ---")
    print(f"  All-empty source (pathological) : {empty_seg_warn:,}")

    if examples_src:
        print(f"\n  Source mismatch examples (up to 5):")
        for ex in examples_src[:5]:
            print(f"    {ex['utt_id']}")
            print(f"      streaming: {repr(ex['src'])}")
            print(f"      TSV      : {repr(ex['tsv'])}")

    if examples_punct:
        print(f"\n  Punct issue examples (up to 5):")
        for ex in examples_punct[:5]:
            print(f"    {ex['utt_id']}  {ex['issue']}")
            print(f"      src: {repr(ex['src'])}")
            print(f"      tgt: {repr(ex['tgt'])}")

    print("=" * 68)

    # ──── Write report ────
    report = {
        "stream_dir":          args.stream_dir,
        "total":               total,
        "error_load":          error_load,
        "checked":             checked,
        "no_tsv_entry":        no_tsv_entry,
        "src_exact":           src_exact,
        "src_case_diff":       src_case_diff,
        "src_punct_diff_only": src_punct_diff,
        "src_mismatch":        src_mismatch,
        "src_empty":           src_empty,
        "len_mismatch":        len_mismatch,
        "punct_bad":           punct_bad,
        "punct_zh_missing":    punct_missing,
        "punct_zh_excess":     punct_excess,
        "punct_large_diff":    punct_large,
        "empty_seg_warn":      empty_seg_warn,
        "diff_distribution":   {str(k): v for k, v in sorted(diff_dist.items())},
        "en_sent_distribution":{str(k): v for k, v in sorted(en_sent_dist.items())},
        "zh_sent_distribution":{str(k): v for k, v in sorted(zh_sent_dist.items())},
        "src_mismatch_examples":  examples_src[:50],
        "punct_issue_examples":   examples_punct[:50],
    }
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report written to: {args.report}")


if __name__ == "__main__":
    main()
