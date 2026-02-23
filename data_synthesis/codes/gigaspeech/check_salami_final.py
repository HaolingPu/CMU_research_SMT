#!/usr/bin/env python3
"""
Check final_jsonl_salami against the TSV manifest (src_text_full) and
MetricX manifest (hypothesis).

Checks per entry:
  1. SOURCE TEXT:  " ".join(non-empty source segs) == TSV src_text_full joined
  2. SOURCE PUNCT: punctuation symbol counts match between joined source and TSV text
  3. TARGET TEXT:  "".join(non-empty target segs) == metricx hypothesis
  4. TARGET PUNCT: punctuation symbol counts match
  5. SRC/TGT ARRAY LENGTH: source and target segment arrays have equal length
  6. SRC/TGT SENT-END PUNCT: English [.?!] vs Chinese [。？！] count alignment
  7. SRC/TGT CLAUSE PUNCT:  English [,;]  vs Chinese [，；]  count alignment

Usage:
  python check_salami_final.py \
    --tsv       .../train_xl_case_robust_asr-filtered.tsv \
    --manifest  .../metricx_filtered_t3.0.jsonl \
    --final_dir .../final_jsonl_salami_patched \
    --report    ./check_salami_final_report.json
"""

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def join_src(segs):
    return " ".join(s for s in segs if s)

def join_tgt(segs):
    return "".join(s for s in segs if s)

def parse_src_text_full(raw_value) -> str:
    """Parse src_text_full list from TSV and join into a single string,
    matching exactly what llm_output_salami.py writes as the `input` field."""
    if not raw_value:
        return ""
    raw = str(raw_value).strip()
    if not raw:
        return ""
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = raw
    if isinstance(parsed, list):
        sentences = [str(x).strip() for x in parsed if str(x).strip()]
    else:
        sentences = [str(parsed).strip()] if str(parsed).strip() else []
    return " ".join(sentences).strip()

def punct_counts(text: str) -> Counter:
    """Count every non-alphanumeric, non-space character."""
    return Counter(ch for ch in text if not ch.isalnum() and not ch.isspace())

def diff_counters(c1: Counter, c2: Counter) -> dict:
    """Return {char: (in_final, in_manifest)} for chars that differ."""
    all_keys = set(c1) | set(c2)
    return {k: (c1[k], c2[k]) for k in sorted(all_keys) if c1[k] != c2[k]}


# ---------------------------------------------------------------------------
# Source / target punctuation alignment
# ---------------------------------------------------------------------------
EN_SENT_END = set('.?!')
ZH_SENT_END = set('。？！')
EN_CLAUSE   = set(',;')
ZH_CLAUSE   = set('，；')

def count_punct(text: str, charset: set) -> int:
    return sum(1 for ch in text if ch in charset)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tsv(path):
    """Returns {id -> src_text_full_joined}"""
    print(f"Loading TSV: {path}")
    index = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = row.get("id", "").strip()
            if not uid:
                continue
            index[uid] = parse_src_text_full(row.get("src_text_full", ""))
    print(f"  {len(index)} entries")
    return index


def load_manifest(path):
    """Returns {utt_id -> hypothesis}"""
    print(f"Loading MetricX manifest: {path}")
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uid = d["metadata"]["utt_id"]
            index[uid] = d.get("hypothesis", "")
    print(f"  {len(index)} entries")
    return index


def load_final(root):
    """Returns {utt_id -> entry_dict}"""
    print(f"Loading final jsonl: {root}")
    index = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    uid = d.get("utt_id", "")
                    if uid:
                        index[uid] = d
    print(f"  {len(index)} entries")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv",       required=True, help="train_xl_case_robust_asr-filtered.tsv")
    ap.add_argument("--manifest",  required=True, help="metricx_filtered_t3.0.jsonl (for hypothesis)")
    ap.add_argument("--final_dir", required=True, help="final_jsonl_salami directory")
    ap.add_argument("--report",    default="check_salami_final_report.json")
    args = ap.parse_args()

    tsv      = load_tsv(args.tsv)
    manifest = load_manifest(args.manifest)
    final    = load_final(args.final_dir)

    tsv_ids      = set(tsv)
    manifest_ids = set(manifest)
    final_ids    = set(final)

    # IDs present in final but not in TSV (shouldn't happen)
    extra_in_final   = sorted(final_ids - tsv_ids)
    # IDs in metricx manifest but not in final
    missing_in_final = sorted(manifest_ids - final_ids)

    # Common set for checking: must be in all three
    common = sorted(final_ids & tsv_ids & manifest_ids)
    total  = len(common)

    src_exact       = 0
    src_diff        = []
    src_punct_exact = 0
    src_punct_diff  = []
    tgt_exact       = 0
    tgt_diff        = []
    tgt_punct_exact = 0
    tgt_punct_diff  = []

    # src/tgt punctuation alignment accumulators
    st_len_mismatch      = []
    st_sent_end_zero_tgt = []   # English >=2 sent-enders but Chinese = 0
    st_sent_end_big_diff = []   # |en_sent - zh_sent| >= 3
    en_sent_counts       = Counter()
    zh_sent_counts       = Counter()
    en_clause_counts     = Counter()
    zh_clause_counts     = Counter()
    diff_sent_end_dist   = Counter()

    for uid in common:
        tsv_src  = tsv[uid]
        hyp      = manifest[uid]
        d        = final[uid]

        src_joined = join_src(d.get("source", []))
        tgt_joined = join_tgt(d.get("target", []))

        # --- check 1: source text ---
        if src_joined == tsv_src:
            src_exact += 1
        else:
            src_diff.append({
                "utt_id":   uid,
                "tsv_src":  tsv_src,
                "final_src": src_joined,
            })

        # --- check 2: source punct ---
        delta = diff_counters(punct_counts(src_joined), punct_counts(tsv_src))
        if not delta:
            src_punct_exact += 1
        else:
            src_punct_diff.append({"utt_id": uid, "diff": delta})

        # --- check 3: target text ---
        if tgt_joined == hyp:
            tgt_exact += 1
        else:
            tgt_diff.append({
                "utt_id":    uid,
                "hypothesis": hyp,
                "final_tgt": tgt_joined,
            })

        # --- check 4: target punct ---
        delta2 = diff_counters(punct_counts(tgt_joined), punct_counts(hyp))
        if not delta2:
            tgt_punct_exact += 1
        else:
            tgt_punct_diff.append({"utt_id": uid, "diff": delta2})

        # --- checks 5-7: source / target punctuation alignment ---
        src_segs = d.get("source", [])
        tgt_segs = d.get("target", [])
        if len(src_segs) != len(tgt_segs):
            st_len_mismatch.append({
                "utt_id": uid,
                "src_len": len(src_segs),
                "tgt_len": len(tgt_segs),
            })
        else:
            en_sent   = count_punct(src_joined, EN_SENT_END)
            zh_sent   = count_punct(tgt_joined, ZH_SENT_END)
            en_clause = count_punct(src_joined, EN_CLAUSE)
            zh_clause = count_punct(tgt_joined, ZH_CLAUSE)

            en_sent_counts[en_sent]   += 1
            zh_sent_counts[zh_sent]   += 1
            en_clause_counts[en_clause] += 1
            zh_clause_counts[zh_clause] += 1
            diff_sent_end_dist[en_sent - zh_sent] += 1

            if en_sent >= 2 and zh_sent == 0:
                st_sent_end_zero_tgt.append({
                    "utt_id": uid,
                    "en_sent_end": en_sent,
                    "zh_sent_end": zh_sent,
                    "src": src_joined[:120],
                    "tgt": tgt_joined[:120],
                })

            if abs(en_sent - zh_sent) >= 3:
                st_sent_end_big_diff.append({
                    "utt_id": uid,
                    "en_sent_end": en_sent,
                    "zh_sent_end": zh_sent,
                    "diff": en_sent - zh_sent,
                    "src": src_joined[:120],
                    "tgt": tgt_joined[:120],
                })

    # --- print ---
    print("\n" + "=" * 60)
    print("CHECK RESULTS")
    print("=" * 60)

    print(f"\n--- Coverage ---")
    print(f"  TSV entries:              {len(tsv_ids)}")
    print(f"  MetricX manifest:         {len(manifest_ids)}")
    print(f"  Final jsonl:              {len(final_ids)}")
    print(f"  Missing in final (vs manifest): {len(missing_in_final)}")
    print(f"  Extra in final (not in TSV):    {len(extra_in_final)}")
    print(f"  Common (all three):       {total}")

    print(f"\n--- SOURCE text: joined segs vs TSV src_text_full [{total}] ---")
    print(f"  exact match:    {src_exact}")
    print(f"  MISMATCH:       {len(src_diff)}")
    if src_diff:
        print(f"\n  First 5 mismatches:")
        for ex in src_diff[:5]:
            print(f"    {ex['utt_id']}")
            print(f"      tsv:   {repr(ex['tsv_src'][:100])}")
            print(f"      final: {repr(ex['final_src'][:100])}")

    print(f"\n--- SOURCE punct counts ---")
    print(f"  exact match:    {src_punct_exact}")
    print(f"  MISMATCH:       {len(src_punct_diff)}")
    if src_punct_diff:
        sym = Counter()
        for item in src_punct_diff:
            for s in item["diff"]: sym[s] += 1
        print(f"  Top differing symbols:")
        for s, c in sym.most_common(10):
            print(f"    {repr(s)}: {c} entries")
        print(f"  First 5 examples:")
        for ex in src_punct_diff[:5]:
            print(f"    {ex['utt_id']}: {ex['diff']}")

    print(f"\n--- TARGET text: joined segs vs hypothesis [{total}] ---")
    print(f"  exact match:    {tgt_exact}")
    print(f"  MISMATCH:       {len(tgt_diff)}")
    if tgt_diff:
        print(f"\n  First 5 mismatches:")
        for ex in tgt_diff[:5]:
            print(f"    {ex['utt_id']}")
            print(f"      hyp:   {repr(ex['hypothesis'][:100])}")
            print(f"      final: {repr(ex['final_tgt'][:100])}")

    print(f"\n--- TARGET punct counts ---")
    print(f"  exact match:    {tgt_punct_exact}")
    print(f"  MISMATCH:       {len(tgt_punct_diff)}")
    if tgt_punct_diff:
        sym = Counter()
        for item in tgt_punct_diff:
            for s in item["diff"]: sym[s] += 1
        print(f"  Top differing symbols:")
        for s, c in sym.most_common(10):
            print(f"    {repr(s)}: {c} entries")
        print(f"  First 5 examples:")
        for ex in tgt_punct_diff[:5]:
            print(f"    {ex['utt_id']}: {ex['diff']}")

    print(f"\n--- SOURCE / TARGET punctuation alignment [{total}] ---")
    print(f"  Array length mismatches:   {len(st_len_mismatch)}")

    print(f"\n  Sentence-ending marks (English [.?!] vs Chinese [。？！]):")
    print(f"    English counts (top 10):")
    for cnt, freq in sorted(en_sent_counts.items())[:10]:
        print(f"      en_sent={cnt}: {freq} entries")
    print(f"    Chinese counts (top 10):")
    for cnt, freq in sorted(zh_sent_counts.items())[:10]:
        print(f"      zh_sent={cnt}: {freq} entries")
    print(f"    Difference distribution (en - zh), top 15:")
    for diff, freq in sorted(diff_sent_end_dist.items(), key=lambda x: -x[1])[:15]:
        print(f"      diff={diff:+d}: {freq} entries")
    print(f"    Flagged: English >=2 sent-ends but Chinese = 0: {len(st_sent_end_zero_tgt)}")
    if st_sent_end_zero_tgt:
        for ex in st_sent_end_zero_tgt[:5]:
            print(f"      {ex['utt_id']}  en={ex['en_sent_end']} zh={ex['zh_sent_end']}")
            print(f"        src: {repr(ex['src'])}")
            print(f"        tgt: {repr(ex['tgt'])}")
    print(f"    Flagged: |en - zh| >= 3: {len(st_sent_end_big_diff)}")
    if st_sent_end_big_diff:
        for ex in st_sent_end_big_diff[:5]:
            print(f"      {ex['utt_id']}  en={ex['en_sent_end']} zh={ex['zh_sent_end']} diff={ex['diff']:+d}")
            print(f"        src: {repr(ex['src'])}")
            print(f"        tgt: {repr(ex['tgt'])}")

    print(f"\n  Comma/clause marks (English [,;] vs Chinese [，；]):")
    print(f"    English counts (top 8):")
    for cnt, freq in sorted(en_clause_counts.items())[:8]:
        print(f"      en_clause={cnt}: {freq} entries")
    print(f"    Chinese counts (top 8):")
    for cnt, freq in sorted(zh_clause_counts.items())[:8]:
        print(f"      zh_clause={cnt}: {freq} entries")

    print()

    # --- write report ---
    report = {
        "tsv_count":           len(tsv_ids),
        "manifest_count":      len(manifest_ids),
        "final_count":         len(final_ids),
        "missing_in_final":    missing_in_final[:200],
        "extra_in_final":      extra_in_final[:200],
        "common_count":        total,
        "source_exact":        src_exact,
        "source_mismatch_count": len(src_diff),
        "source_mismatch_examples": src_diff[:50],
        "source_punct_exact":  src_punct_exact,
        "source_punct_mismatch_count": len(src_punct_diff),
        "source_punct_mismatch_examples": src_punct_diff[:50],
        "target_exact":        tgt_exact,
        "target_mismatch_count": len(tgt_diff),
        "target_mismatch_examples": tgt_diff[:50],
        "target_punct_exact":  tgt_punct_exact,
        "target_punct_mismatch_count": len(tgt_punct_diff),
        "target_punct_mismatch_examples": tgt_punct_diff[:50],
        # src/tgt alignment
        "st_array_len_mismatch_count": len(st_len_mismatch),
        "st_array_len_mismatch_examples": st_len_mismatch[:20],
        "st_sent_end_zero_tgt_count": len(st_sent_end_zero_tgt),
        "st_sent_end_zero_tgt_examples": st_sent_end_zero_tgt[:50],
        "st_sent_end_big_diff_count": len(st_sent_end_big_diff),
        "st_sent_end_big_diff_examples": st_sent_end_big_diff[:50],
        "diff_sent_end_distribution": {str(k): v for k, v in sorted(diff_sent_end_dist.items())},
        "en_sent_end_distribution": {str(k): v for k, v in sorted(en_sent_counts.items())},
        "zh_sent_end_distribution": {str(k): v for k, v in sorted(zh_sent_counts.items())},
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
