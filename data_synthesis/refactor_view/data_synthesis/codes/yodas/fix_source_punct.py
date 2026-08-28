#!/usr/bin/env python3
"""
fix_source_punct.py

Post-process final JSONL output by restoring punctuation from manifest:
  1. Exact match with manifest → keep as-is.
  2. Tokens match but punct differs → restore punct from manifest into source
     segments, then verify result exactly matches manifest → keep.
  3. Token mismatch or restoration fails → discard.

Maintains the same directory structure as the input (AUD_ID/latency_latency.jsonl).
Also writes a flat combined good JSONL.

Usage:
  python fix_source_punct.py \
    --final_dir   /path/to/final_jsonl \
    --stream_dir  /path/to/streaming_dataset \
    --out_dir     /path/to/final_jsonl_fixed \
    --out_flat    /path/to/good_fixed.jsonl \
    [--latency low]
"""

import os
import re
import json
import argparse
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text helpers (same as pipeline)
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# ---------------------------------------------------------------------------
# Punctuation restoration
# ---------------------------------------------------------------------------

def restore_punct_in_segments(source_list: list, manifest_text: str):
    """
    Given a list of source segments and the manifest original_text,
    try to restore punctuation from manifest into the segments.

    Returns (ok: bool, new_source_list: list)
      ok=True  → result exactly matches manifest (case-insensitive)
      ok=False → token mismatch or restoration failed; caller should discard
    """
    mf_norm = normalize_whitespace(manifest_text)

    # Non-empty segment indices
    seg_idx = [i for i, s in enumerate(source_list)
               if isinstance(s, str) and s.strip()]
    if not seg_idx:
        return False, source_list

    # Concatenated source
    full = normalize_whitespace(" ".join(source_list[i] for i in seg_idx))

    # Check token-level match
    src_toks = normalize_text(full).split()
    mf_toks  = normalize_text(mf_norm).split()
    if src_toks != mf_toks:
        return False, source_list  # token mismatch, discard

    # Already exact?
    if full.lower() == mf_norm.lower():
        return True, source_list

    # Find word spans in manifest (keep apostrophes as word chars)
    word_spans = [(m.start(), m.end())
                  for m in re.finditer(r"[a-zA-Z0-9']+", mf_norm)]

    if not word_spans:
        return False, source_list

    # Words per non-empty segment
    word_counts = [len(normalize_text(source_list[i]).split()) for i in seg_idx]

    if sum(word_counts) != len(word_spans):
        # Shouldn't happen if token check passed, but guard anyway
        return False, source_list

    new_list = list(source_list)
    word_cursor = 0
    mf_pos = 0  # current position in mf_norm (tracks what's been "consumed")

    for k, orig_i in enumerate(seg_idx):
        n = word_counts[k]
        if n == 0:
            new_list[orig_i] = ""
            continue

        last_w = word_cursor + n - 1
        end_of_last_word = word_spans[last_w][1]

        # Trailing punctuation: non-space chars immediately after last word
        if last_w + 1 < len(word_spans):
            next_word_start = word_spans[last_w + 1][0]
            between = mf_norm[end_of_last_word:next_word_start]
            m = re.match(r'[^\s]*', between)
            trailing = m.group() if m else ''
        else:
            # Last segment: include everything to end of manifest
            trailing = mf_norm[end_of_last_word:].rstrip()

        seg_end = end_of_last_word + len(trailing)

        # New segment = manifest text from mf_pos (includes any leading punct/space
        # that was "between" previous segment and this one) to seg_end
        new_seg = mf_norm[mf_pos:seg_end]
        new_list[orig_i] = new_seg

        # Advance mf_pos past trailing punct + whitespace
        mf_pos = seg_end
        while mf_pos < len(mf_norm) and mf_norm[mf_pos] in (' ', '\t', '\n'):
            mf_pos += 1

        word_cursor += n

    # Verify: concatenated new non-empty segs exactly matches manifest
    new_full = normalize_whitespace(" ".join(new_list[i] for i in seg_idx))
    if new_full.lower() != mf_norm.lower():
        return False, source_list  # restoration failed, discard

    return True, new_list


# ---------------------------------------------------------------------------
# Streaming index
# ---------------------------------------------------------------------------

def build_stream_index(stream_root: str) -> dict:
    index = {}
    for dirpath, _, files in os.walk(stream_root):
        for fn in files:
            if fn.endswith(".json"):
                index[fn[:-5]] = os.path.join(dirpath, fn)
    return index


# ---------------------------------------------------------------------------
# Collect JSONL files
# ---------------------------------------------------------------------------

def collect_jsonl_files(final_root: str, latency_filter=None):
    files = []
    for dirpath, _, filenames in os.walk(final_root):
        for fn in sorted(filenames):
            if not fn.endswith("_latency.jsonl"):
                continue
            lat = fn[: -len("_latency.jsonl")]
            if latency_filter and lat != latency_filter:
                continue
            files.append((os.path.join(dirpath, fn), lat,
                          os.path.relpath(dirpath, final_root)))
    files.sort()
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir",  required=True)
    ap.add_argument("--stream_dir", required=True)
    ap.add_argument("--out_dir",    required=True,
                    help="Output directory (mirrors input structure)")
    ap.add_argument("--out_flat",   default=None,
                    help="Optional flat JSONL with all kept entries")
    ap.add_argument("--latency",    default=None,
                    help="Filter to one latency (default: all)")
    args = ap.parse_args()

    print(f"Building streaming index from {args.stream_dir} ...")
    stream_index = build_stream_index(args.stream_dir)
    print(f"  Indexed {len(stream_index):,} streaming JSONs")

    jsonl_files = collect_jsonl_files(args.final_dir, args.latency)
    print(f"Found {len(jsonl_files)} JSONL files")
    if not jsonl_files:
        print("No files found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    flat_fh = open(args.out_flat, "w", encoding="utf-8") if args.out_flat else None

    # Counters
    total = kept_exact = kept_fixed = dropped_token = dropped_fix_fail = dropped_no_manifest = 0

    for jpath, lat, rel_dir in tqdm(jsonl_files, desc="Files"):
        out_subdir = os.path.join(args.out_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"{lat}_latency.jsonl")

        with open(jpath,     "r", encoding="utf-8") as fin, \
             open(out_path,  "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                total += 1
                utt_id = item.get("utt_id", "")
                source = item.get("source", [])

                # Look up manifest
                p = stream_index.get(utt_id)
                if not p:
                    dropped_no_manifest += 1
                    continue

                try:
                    with open(p, "r", encoding="utf-8") as sf:
                        seg = json.load(sf)
                except Exception:
                    dropped_no_manifest += 1
                    continue

                manifest_text = seg.get("original_text", "")
                if not manifest_text:
                    dropped_no_manifest += 1
                    continue

                # Concatenated source vs manifest
                non_empty = [s for s in source if isinstance(s, str) and s.strip()]
                final_source = normalize_whitespace(" ".join(non_empty))
                mf_norm      = normalize_whitespace(manifest_text)

                if final_source.lower() == mf_norm.lower():
                    # Already exact
                    kept_exact += 1
                    out_line = json.dumps(item, ensure_ascii=False)
                    fout.write(out_line + "\n")
                    if flat_fh:
                        flat_fh.write(out_line + "\n")
                    continue

                # Try to restore punctuation
                ok, new_source = restore_punct_in_segments(source, manifest_text)
                if not ok:
                    # Token mismatch → discard
                    dropped_token += 1
                    continue

                # Verify restored source
                new_non_empty = [s for s in new_source if isinstance(s, str) and s.strip()]
                new_full = normalize_whitespace(" ".join(new_non_empty))
                if new_full.lower() != mf_norm.lower():
                    dropped_fix_fail += 1
                    continue

                # Keep with restored source
                kept_fixed += 1
                item["source"] = new_source
                out_line = json.dumps(item, ensure_ascii=False)
                fout.write(out_line + "\n")
                if flat_fh:
                    flat_fh.write(out_line + "\n")

    if flat_fh:
        flat_fh.close()

    kept_total = kept_exact + kept_fixed
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total input          : {total:,}")
    print(f"  Kept exact match     : {kept_exact:,}")
    print(f"  Kept after punct fix : {kept_fixed:,}")
    print(f"  ─────────────────────")
    print(f"  Total kept           : {kept_total:,}  ({100*kept_total/total:.1f}%)")
    print(f"  Dropped token-diff   : {dropped_token:,}")
    print(f"  Dropped fix-fail     : {dropped_fix_fail:,}")
    print(f"  Dropped no-manifest  : {dropped_no_manifest:,}")
    print("=" * 60)
    print(f"\nOutput dir  → {args.out_dir}")
    if args.out_flat:
        print(f"Flat JSONL  → {args.out_flat}")


if __name__ == "__main__":
    main()
