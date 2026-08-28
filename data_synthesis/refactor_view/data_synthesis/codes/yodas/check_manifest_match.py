#!/usr/bin/env python3
"""
check_manifest_match.py

Checks whether the concatenated source segments in the final output
match the original_text (manifest) — INCLUDING punctuation.

validate_final_output.py uses normalize_text() which strips punctuation,
so it won't catch punctuation loss. This script fills that gap.

For each entry in the final output JSONL:
  1. Concatenate source segments -> final_source
  2. Load original_text from the streaming dataset JSON -> manifest_text
  3. Compare both raw and punctuation-aware, report any mismatch

Usage:
  python check_manifest_match.py \
    --final_dir  /path/to/final_jsonl_dataset \
    --stream_dir /path/to/streaming_dataset \
    --out_report /path/to/report.json \
    [--only_lang en000] \
    [--only_pq 00000000] \
    [--max_lines 1000] \
    [--latency low]
"""

import os
import re
import json
import argparse
from typing import Optional
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Strip punctuation (same as pipeline). Used for token-level match."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_whitespace(s: str) -> str:
    """Collapse whitespace but keep all characters including punctuation."""
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_punctuation_chars(s: str) -> str:
    """Return only punctuation characters from a string."""
    return "".join(c for c in s if not c.isalnum() and not c.isspace() and c != "'")


# ---------------------------------------------------------------------------
# Build streaming dataset index: utt_id -> json path
# ---------------------------------------------------------------------------

def build_stream_index(stream_root: str, only_lang=None, only_pq=None):
    index = {}
    for dirpath, _, files in os.walk(stream_root):
        if only_lang and (os.sep + only_lang + os.sep) not in (dirpath + os.sep):
            continue
        if only_pq and (os.sep + only_pq + os.sep) not in (dirpath + os.sep):
            continue
        for fn in files:
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            utt_id = fn[:-5]  # strip .json
            index[utt_id] = p
    return index


# ---------------------------------------------------------------------------
# Collect final JSONL files
# ---------------------------------------------------------------------------

def collect_final_jsonl_files(final_root, only_lang=None, only_pq=None, latency_filter=None):
    files = []
    for lang in sorted(os.listdir(final_root)):
        if not os.path.isdir(os.path.join(final_root, lang)):
            continue
        if only_lang and lang != only_lang:
            continue
        for pq in sorted(os.listdir(os.path.join(final_root, lang))):
            pq_dir = os.path.join(final_root, lang, pq)
            if not os.path.isdir(pq_dir):
                continue
            if only_pq and pq != only_pq:
                continue
            for fn in sorted(os.listdir(pq_dir)):
                if not fn.endswith("_latency.jsonl"):
                    continue
                lat = fn.replace("_latency.jsonl", "")
                if latency_filter and lat != latency_filter:
                    continue
                files.append(os.path.join(pq_dir, fn))
    return files


# ---------------------------------------------------------------------------
# Compare one entry
# ---------------------------------------------------------------------------

def compare_entry(utt_id, source_list, stream_index):
    """
    Returns a dict describing the comparison result.
    Keys:
      ok            : bool
      reasons       : list[str]
      final_source  : str   (concatenated, whitespace normalized)
      manifest_text : str   (original_text from streaming JSON)
      missing_punct : str   (punctuation chars in manifest but not in final)
      extra_punct   : str   (punctuation chars in final but not in manifest)
    """
    result = {
        "utt_id": utt_id,
        "ok": True,
        "reasons": [],
        "final_source": "",
        "manifest_text": "",
        "missing_punct": "",
        "extra_punct": "",
    }

    # Look up streaming JSON
    p = stream_index.get(utt_id)
    if not p:
        result["ok"] = False
        result["reasons"].append("stream_json_not_found")
        return result

    try:
        with open(p, "r", encoding="utf-8") as f:
            seg = json.load(f)
    except Exception as e:
        result["ok"] = False
        result["reasons"].append(f"stream_json_load_error:{e}")
        return result

    manifest_text = seg.get("original_text", "")
    if not manifest_text:
        result["ok"] = False
        result["reasons"].append("missing_original_text")
        return result

    # Concatenate source segments (only non-empty ones)
    non_empty = [s for s in source_list if isinstance(s, str) and s.strip()]
    final_source = normalize_whitespace(" ".join(non_empty))
    manifest_norm = normalize_whitespace(manifest_text)

    result["final_source"] = final_source
    result["manifest_text"] = manifest_norm

    # ---- Check 1: exact match (whitespace-normalized, case-preserved) ----
    if final_source.lower() != manifest_norm.lower():
        result["ok"] = False
        result["reasons"].append("text_mismatch_with_punctuation")

        # Detailed: which punctuation is missing / extra?
        mf_punct = extract_punctuation_chars(manifest_norm)
        fn_punct = extract_punctuation_chars(final_source)

        missing = ""
        extra = ""
        # Count character by character (multiset diff)
        mf_counts = {}
        fn_counts = {}
        for c in mf_punct:
            mf_counts[c] = mf_counts.get(c, 0) + 1
        for c in fn_punct:
            fn_counts[c] = fn_counts.get(c, 0) + 1
        all_chars = set(mf_counts) | set(fn_counts)
        for c in sorted(all_chars):
            m = mf_counts.get(c, 0)
            n = fn_counts.get(c, 0)
            if m > n:
                missing += c * (m - n)
            elif n > m:
                extra += c * (n - m)

        result["missing_punct"] = missing  # in manifest but not in final
        result["extra_punct"]   = extra    # in final but not in manifest

        if missing or extra:
            result["reasons"].append(f"punct_diff: missing={repr(missing)} extra={repr(extra)}")

        # Also check if tokens match (to distinguish punct-only vs token-level diff)
        final_toks = normalize_text(final_source).split()
        manifest_toks = normalize_text(manifest_norm).split()
        if final_toks != manifest_toks:
            result["reasons"].append("token_mismatch_too")
        else:
            result["reasons"].append("tokens_ok_but_punct_differs")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Check manifest vs final output (including punctuation).")
    ap.add_argument("--final_dir",  required=True, help="Root of final_jsonl_dataset/")
    ap.add_argument("--stream_dir", required=True, help="Root of streaming_dataset/")
    ap.add_argument("--out_report", required=True, help="Output report JSON path")
    ap.add_argument("--only_lang",  default=None,  help="Filter to one language, e.g. en000")
    ap.add_argument("--only_pq",    default=None,  help="Filter to one parquet, e.g. 00000000")
    ap.add_argument("--latency",    default=None,  help="Filter to one latency: low/medium/high/offline")
    ap.add_argument("--max_lines",  type=int, default=None, help="Max lines per JSONL file (debug)")
    ap.add_argument("--show_bad",   type=int, default=20,   help="Print first N bad entries to stdout")
    args = ap.parse_args()

    print("=" * 60)
    print("Manifest vs Final Output Punctuation Check")
    print("=" * 60)

    # Build streaming index
    print(f"\n[1/3] Building streaming dataset index from: {args.stream_dir}")
    stream_index = build_stream_index(args.stream_dir, only_lang=args.only_lang, only_pq=args.only_pq)
    print(f"      Indexed {len(stream_index)} streaming JSONs")

    # Collect final JSONL files
    print(f"\n[2/3] Collecting final JSONL files from: {args.final_dir}")
    jsonl_files = collect_final_jsonl_files(
        args.final_dir,
        only_lang=args.only_lang,
        only_pq=args.only_pq,
        latency_filter=args.latency,
    )
    print(f"      Found {len(jsonl_files)} JSONL files")

    if not jsonl_files:
        print("No JSONL files found. Check --final_dir / --only_lang / --only_pq.")
        return

    # Process
    print(f"\n[3/3] Checking entries...\n")

    grand_total = 0
    grand_ok = 0
    grand_bad = 0
    grand_punct_only = 0  # tokens match but punctuation differs
    grand_token_diff = 0  # actual token-level mismatch

    report_files = []
    shown_bad = 0

    for jfile in tqdm(jsonl_files, desc="Files"):
        lat = os.path.basename(jfile).replace("_latency.jsonl", "")
        file_total = file_ok = file_bad = 0
        file_bad_items = []

        with open(jfile, "r", encoding="utf-8") as f:
            for line_i, line in enumerate(f):
                if args.max_lines is not None and line_i >= args.max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                grand_total += 1
                file_total += 1

                try:
                    item = json.loads(line)
                except Exception as e:
                    grand_bad += 1
                    file_bad += 1
                    file_bad_items.append({"line": line_i, "reasons": [f"parse_error:{e}"]})
                    continue

                utt_id = item.get("utt_id", "")
                source  = item.get("source", [])

                res = compare_entry(utt_id, source, stream_index)

                if res["ok"]:
                    grand_ok += 1
                    file_ok += 1
                else:
                    grand_bad += 1
                    file_bad += 1
                    reasons_str = " | ".join(res["reasons"])
                    if "tokens_ok_but_punct_differs" in reasons_str:
                        grand_punct_only += 1
                    if "token_mismatch_too" in reasons_str:
                        grand_token_diff += 1

                    bad_entry = {
                        "line": line_i,
                        "utt_id": utt_id,
                        "latency": lat,
                        "reasons": res["reasons"],
                        "missing_punct": res["missing_punct"],
                        "extra_punct": res["extra_punct"],
                        "final_source": res["final_source"][:200],
                        "manifest_text": res["manifest_text"][:200],
                    }
                    file_bad_items.append(bad_entry)

                    # Print first N bad entries
                    if shown_bad < args.show_bad:
                        shown_bad += 1
                        print(f"\n--- BAD #{shown_bad}: {utt_id} [{lat}] ---")
                        print(f"  Reasons   : {res['reasons']}")
                        if res["missing_punct"]:
                            print(f"  Missing punct (in manifest, not in final): {repr(res['missing_punct'])}")
                        if res["extra_punct"]:
                            print(f"  Extra punct  (in final, not in manifest):  {repr(res['extra_punct'])}")
                        print(f"  MANIFEST  : {res['manifest_text'][:120]}")
                        print(f"  FINAL SRC : {res['final_source'][:120]}")

        rel = os.path.relpath(jfile, args.final_dir)
        report_files.append({
            "file": rel,
            "latency": lat,
            "total": file_total,
            "ok": file_ok,
            "bad": file_bad,
            "bad_rate": round(file_bad / file_total, 4) if file_total else 0,
            "bad_items": file_bad_items,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total entries         : {grand_total}")
    print(f"  OK (exact match)      : {grand_ok}  ({100*grand_ok/grand_total:.1f}% )" if grand_total else "  (no entries)")
    print(f"  BAD (any mismatch)    : {grand_bad}")
    print(f"    ↳ punct-only diff   : {grand_punct_only}  (tokens match, punctuation differs)")
    print(f"    ↳ token-level diff  : {grand_token_diff}  (actual word mismatch)")
    print("=" * 60)

    # Write report
    os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)
    report = {
        "meta": {
            "final_dir":  args.final_dir,
            "stream_dir": args.stream_dir,
            "only_lang":  args.only_lang,
            "only_pq":    args.only_pq,
            "latency":    args.latency,
            "max_lines":  args.max_lines,
        },
        "summary": {
            "total":           grand_total,
            "ok":              grand_ok,
            "bad":             grand_bad,
            "punct_only_diff": grand_punct_only,
            "token_diff":      grand_token_diff,
            "bad_rate":        round(grand_bad / grand_total, 4) if grand_total else 0,
        },
        "files": report_files,
    }
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport written → {args.out_report}")


if __name__ == "__main__":
    main()