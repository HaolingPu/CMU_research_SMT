#!/usr/bin/env python3
"""
gen_manifest_report.py

Generate a manifest_check_report.json for a given final JSONL dataset.
Handles the flat AUD_ID/<latency>_latency.jsonl directory structure
(used by EAST, refined_east, salami).

Compares concatenated source segments in the final output against
original_text in the streaming dataset JSONs.

Usage:
  python gen_manifest_report.py \
    --final_dir  /path/to/final_jsonl_dir \
    --stream_dir /path/to/streaming_dataset \
    --out_report /path/to/manifest_check_report.json \
    [--latency low]          # filter to one latency (e.g. low, offline); default=all
"""

import os
import re
import json
import argparse
from tqdm import tqdm


def normalize_text(s: str) -> str:
    """Strip punctuation (for token-level comparison)."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_punctuation_chars(s: str) -> str:
    return "".join(c for c in s if not c.isalnum() and not c.isspace() and c != "'")


def build_stream_index(stream_root: str) -> dict:
    index = {}
    for dirpath, _, files in os.walk(stream_root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            utt_id = fn[:-5]
            index[utt_id] = os.path.join(dirpath, fn)
    return index


def collect_jsonl_files(final_root: str, latency_filter=None) -> list:
    files = []
    for dirpath, _, filenames in os.walk(final_root):
        for fn in sorted(filenames):
            if not fn.endswith("_latency.jsonl"):
                continue
            lat = fn[: -len("_latency.jsonl")]
            if latency_filter and lat != latency_filter:
                continue
            files.append((os.path.join(dirpath, fn), lat))
    files.sort()
    return files


def punct_diff(manifest_norm: str, final_source: str):
    mf_punct = extract_punctuation_chars(manifest_norm)
    fn_punct = extract_punctuation_chars(final_source)
    mf_counts, fn_counts = {}, {}
    for c in mf_punct:
        mf_counts[c] = mf_counts.get(c, 0) + 1
    for c in fn_punct:
        fn_counts[c] = fn_counts.get(c, 0) + 1
    missing = extra = ""
    for c in sorted(set(mf_counts) | set(fn_counts)):
        m = mf_counts.get(c, 0)
        n = fn_counts.get(c, 0)
        if m > n:
            missing += c * (m - n)
        elif n > m:
            extra += c * (n - m)
    return missing, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir", required=True)
    ap.add_argument("--stream_dir", required=True)
    ap.add_argument("--out_report", required=True)
    ap.add_argument("--latency", default=None, help="e.g. low, offline (default: all)")
    args = ap.parse_args()

    print(f"Building streaming index from {args.stream_dir} ...")
    stream_index = build_stream_index(args.stream_dir)
    print(f"  Indexed {len(stream_index):,} streaming JSONs")

    jsonl_files = collect_jsonl_files(args.final_dir, args.latency)
    print(f"Found {len(jsonl_files)} JSONL files (latency filter: {args.latency or 'all'})")
    if not jsonl_files:
        print("No files found. Exiting.")
        return

    total = ok = bad = punct_only = token_diff_count = not_in_manifest = 0
    mismatches = []

    for jpath, lat in tqdm(jsonl_files, desc="Files"):
        with open(jpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                total += 1
                utt_id = item.get("utt_id", "")
                source_list = item.get("source", [])
                target_list = item.get("target", [])

                # Look up streaming JSON
                p = stream_index.get(utt_id)
                if not p:
                    not_in_manifest += 1
                    bad += 1
                    continue

                try:
                    with open(p, "r", encoding="utf-8") as sf:
                        seg = json.load(sf)
                except Exception:
                    bad += 1
                    continue

                manifest_text = seg.get("original_text", "")

                # Concatenate non-empty source segments
                non_empty = [s for s in source_list if isinstance(s, str) and s.strip()]
                final_source = normalize_whitespace(" ".join(non_empty))
                manifest_norm = normalize_whitespace(manifest_text)

                # Exact match (case-insensitive, whitespace-normalized)
                if final_source.lower() == manifest_norm.lower():
                    ok += 1
                    continue

                bad += 1

                # Classify mismatch
                reasons = []
                final_toks = normalize_text(final_source).split()
                manifest_toks = normalize_text(manifest_norm).split()

                if final_toks != manifest_toks:
                    token_diff_count += 1
                    reasons.append("token_mismatch")
                else:
                    punct_only += 1
                    reasons.append("tokens_ok_punct_differs")

                missing, extra = punct_diff(manifest_norm, final_source)
                if missing or extra:
                    reasons.append(f"punct_diff:missing={repr(missing)},extra={repr(extra)}")

                # Concatenate target
                non_empty_tgt = [t for t in target_list if isinstance(t, str) and t.strip()]
                translation = " ".join(non_empty_tgt)

                mismatches.append({
                    "id": utt_id,
                    "source": final_source,
                    "translation": translation,
                    "manifest_src": manifest_norm,
                    "reasons": reasons,
                    "missing_punct": missing,
                    "extra_punct": extra,
                })

    report = {
        "meta": {
            "final_root": args.final_dir,
            "latency": args.latency or "all",
            "total": total,
            "ok": ok,
            "bad": bad,
            "punct_only_diff": punct_only,
            "token_diff": token_diff_count,
            "not_in_manifest": not_in_manifest,
        },
        "mismatches": mismatches,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total entries    : {total:,}")
    if total:
        print(f"  OK (exact match) : {ok:,}  ({100*ok/total:.1f}%)")
        print(f"  Bad (any mismatch): {bad:,}  ({100*bad/total:.1f}%)")
        print(f"    ↳ punct-only   : {punct_only:,}")
        print(f"    ↳ token-diff   : {token_diff_count:,}")
        print(f"    ↳ not in manifest: {not_in_manifest:,}")
    print("=" * 60)
    print(f"\nReport → {args.out_report}")


if __name__ == "__main__":
    main()
