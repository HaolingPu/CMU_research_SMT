#!/usr/bin/env python3
"""
Patch the existing final_jsonl_salami to fix source text issues:

  Problem 1 (488 entries): source segments have wrong case compared to
    manifest source_raw (e.g. "Irish" vs "irish"). Root cause: bug in
    fix_llm_raw.py restore_punct that returned original segments unchanged
    when lowercase matched. Fixed in fix_llm_raw.py for future runs.

  Problem 2 (46 entries): source segments are truncated (last few words
    missing). Root cause: multi_trajectory_gigaspeech.py dropped segments
    whose MFA alignment end=None. Fixed in multi_trajectory for future runs.
    For the current data these 46 entries are DROPPED (only 46/60908 = 0.07%).

Fix strategy for current data:
  - Case-only diff (488): re-distribute source_raw back into segments using
    the same word-boundary logic as restore_punct.
  - Truncated (46): drop the entry, log the utt_id.
  - All other entries: pass through unchanged.

Usage:
  python patch_final_jsonl_source.py \
    --manifest  .../metricx_filtered_t3.0.jsonl \
    --input_dir .../final_jsonl_salami \
    --output_dir .../final_jsonl_salami_patched

The output has the same directory/file structure as the input.
"""

import argparse
import json
import os
import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers (mirrors fix_llm_raw.py logic)
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def restore_source_segs(seg_list: List[str], source_raw: str) -> Tuple[bool, List[str]]:
    """
    Given a list of source segments (some may be empty) and the ground-truth
    source_raw text, re-distribute source_raw into the non-empty segments,
    preserving empty-string positions.

    Returns (ok, new_seg_list).
    ok=False means token mismatch (truncation) → caller should drop entry.
    ok=True  means new_seg_list is fully fixed.
    """
    mf = normalize_ws(source_raw)

    # Pull out only non-empty segments (preserve positions)
    non_empty_indices = [i for i, s in enumerate(seg_list) if s]
    non_empty_segs = [seg_list[i] for i in non_empty_indices]

    if not non_empty_segs:
        return False, seg_list

    full = normalize_ws(" ".join(non_empty_segs))

    # Token-level check (case-insensitive)
    src_toks = normalize_text(full).split()
    mf_toks  = normalize_text(mf).split()
    if src_toks != mf_toks:
        return False, seg_list  # truncation or real content mismatch → drop

    # Already exact?
    if full == mf:
        return True, seg_list

    # Rebuild each non-empty segment from mf using word spans
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"[a-zA-Z0-9']+", mf)]
    if not word_spans:
        return False, seg_list

    word_counts = [len(normalize_text(s).split()) for s in non_empty_segs]
    if sum(word_counts) != len(word_spans):
        return False, seg_list

    new_non_empty = []
    word_cursor = 0
    mf_pos = 0

    for k, seg in enumerate(non_empty_segs):
        n = word_counts[k]
        if n == 0:
            new_non_empty.append("")
            continue

        last_w = word_cursor + n - 1
        end_lw = word_spans[last_w][1]

        if last_w + 1 < len(word_spans):
            between = mf[end_lw: word_spans[last_w + 1][0]]
            m = re.match(r'[^\s]*', between)
            trailing = m.group() if m else ''
        else:
            trailing = mf[end_lw:].rstrip()

        seg_end = end_lw + len(trailing)
        new_non_empty.append(mf[mf_pos:seg_end])

        mf_pos = seg_end
        while mf_pos < len(mf) and mf[mf_pos] in (' ', '\t', '\n'):
            mf_pos += 1

        word_cursor += n

    # Verify
    new_full = normalize_ws(" ".join(new_non_empty))
    if new_full.lower() != mf.lower():
        return False, seg_list

    # Reconstruct full seg_list (put fixed non-empty back in original positions)
    result = list(seg_list)
    for idx, fixed_seg in zip(non_empty_indices, new_non_empty):
        result[idx] = fixed_seg

    return True, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str):
    print(f"Loading manifest: {manifest_path}")
    index = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uid = d["metadata"]["utt_id"]
            index[uid] = d["metadata"].get("source_raw", "")
    print(f"  {len(index)} entries loaded")
    return index


def patch_jsonl_file(src_path: str, dst_path: str, manifest: dict,
                     stats: dict, dropped_ids: list) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    lines_out = []
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uid = d.get("utt_id", "")
            source_raw = manifest.get(uid, "")

            if not source_raw:
                # No manifest entry → keep as-is
                lines_out.append(d)
                stats["no_manifest"] += 1
                continue

            src_segs = d.get("source", [])
            joined = " ".join(s for s in src_segs if s)

            if joined == source_raw:
                lines_out.append(d)
                stats["exact"] += 1
                continue

            ok, fixed_segs = restore_source_segs(src_segs, source_raw)
            if not ok:
                # Truncated → drop
                dropped_ids.append(uid)
                stats["dropped_truncated"] += 1
                continue

            # Verify fix
            fixed_joined = " ".join(s for s in fixed_segs if s)
            if fixed_joined == source_raw:
                d["source"] = fixed_segs
                lines_out.append(d)
                stats["fixed_case"] += 1
            else:
                # Unexpected: keep original but warn
                lines_out.append(d)
                stats["unfixed"] += 1

    with open(dst_path, "w", encoding="utf-8") as f:
        for d in lines_out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   required=True, help="metricx_filtered_t3.0.jsonl")
    ap.add_argument("--input_dir",  required=True, help="final_jsonl_salami directory")
    ap.add_argument("--output_dir", required=True, help="output directory")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)

    stats = {"exact": 0, "fixed_case": 0, "dropped_truncated": 0,
             "no_manifest": 0, "unfixed": 0}
    dropped_ids = []

    jsonl_files = []
    for dirpath, _, files in os.walk(args.input_dir):
        for fn in files:
            if fn.endswith(".jsonl"):
                jsonl_files.append(os.path.join(dirpath, fn))
    jsonl_files.sort()
    print(f"Found {len(jsonl_files)} jsonl files to patch")

    for src_path in jsonl_files:
        rel = os.path.relpath(src_path, args.input_dir)
        dst_path = os.path.join(args.output_dir, rel)
        patch_jsonl_file(src_path, dst_path, manifest, stats, dropped_ids)

    total = sum(stats.values())
    print("\n" + "=" * 55)
    print("PATCH SUMMARY")
    print("=" * 55)
    print(f"  Total entries processed : {total}")
    print(f"  Already exact           : {stats['exact']}")
    print(f"  Fixed (case restored)   : {stats['fixed_case']}")
    print(f"  Dropped (truncated src) : {stats['dropped_truncated']}")
    print(f"  No manifest entry       : {stats['no_manifest']}")
    print(f"  Unfixed (unexpected)    : {stats['unfixed']}")
    print(f"  Final kept entries      : {total - stats['dropped_truncated']}")
    print("=" * 55)

    if dropped_ids:
        log_path = os.path.join(args.output_dir, "dropped_truncated_ids.txt")
        with open(log_path, "w") as f:
            for uid in dropped_ids:
                f.write(uid + "\n")
        print(f"\nDropped utt_ids written to: {log_path}")

    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
