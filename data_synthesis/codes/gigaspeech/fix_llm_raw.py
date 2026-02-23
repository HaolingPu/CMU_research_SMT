#!/usr/bin/env python3
"""
fix_llm_raw.py

Fix LLM output raw JSONs BEFORE post_process/merge step:
  - Compare English source segments against the `input` (manifest) field.
  - Tokens match but punct differs → restore punct from manifest.
  - Token mismatch or restoration fails → discard the whole utterance.
  - Write fixed JSONs to output directory.

Supports two raw formats:
  [refined_east]  low_latency/medium_latency/high_latency keys, each with "English" list
  [salami]        segmented_pairs list of [English, Chinese] pairs

Usage:
  # Refined EAST
  python fix_llm_raw.py \
    --in_dir  .../llm_output_raw \
    --out_dir .../llm_output_raw_fixed \
    --good_jsonl .../good_train_xl_refined.jsonl \
    --out_good_jsonl .../good_fixed.jsonl

  # Salami
  python fix_llm_raw.py \
    --in_dir  .../llm_output_raw \
    --out_dir .../llm_output_raw_fixed \
    --good_jsonl .../good_train_xl_salami.jsonl \
    --out_good_jsonl .../good_fixed.jsonl
"""

import os
import re
import json
import argparse
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# ---------------------------------------------------------------------------
# Core: restore punct from manifest into a flat list of segments
# ---------------------------------------------------------------------------

def restore_punct(seg_list: list, manifest: str):
    """
    seg_list : list of non-empty English strings (flat, no padding)
    manifest : the original text (input field)

    Returns (ok: bool, new_seg_list: list)
      ok=True  → new_seg_list concatenation exactly matches manifest
      ok=False → token mismatch or restoration failed; discard utterance
    """
    mf = normalize_ws(manifest)
    full = normalize_ws(" ".join(seg_list))

    # Token-level check
    src_toks = normalize_text(full).split()
    mf_toks  = normalize_text(mf).split()
    if src_toks != mf_toks:
        return False, seg_list

    # Already exact?
    if full.lower() == mf.lower():
        return True, seg_list

    # Word spans in manifest
    word_spans = [(m.start(), m.end())
                  for m in re.finditer(r"[a-zA-Z0-9']+", mf)]
    if not word_spans:
        return False, seg_list

    # Words per segment
    word_counts = [len(normalize_text(s).split()) for s in seg_list]
    if sum(word_counts) != len(word_spans):
        return False, seg_list

    new_segs = []
    word_cursor = 0
    mf_pos = 0

    for k, seg in enumerate(seg_list):
        n = word_counts[k]
        if n == 0:
            new_segs.append("")
            continue

        last_w = word_cursor + n - 1
        end_lw = word_spans[last_w][1]

        # Trailing punct (non-space chars right after last word)
        if last_w + 1 < len(word_spans):
            between = mf[end_lw: word_spans[last_w + 1][0]]
            m = re.match(r'[^\s]*', between)
            trailing = m.group() if m else ''
        else:
            trailing = mf[end_lw:].rstrip()

        seg_end = end_lw + len(trailing)
        new_segs.append(mf[mf_pos:seg_end])

        mf_pos = seg_end
        while mf_pos < len(mf) and mf[mf_pos] in (' ', '\t', '\n'):
            mf_pos += 1

        word_cursor += n

    # Final verification
    new_full = normalize_ws(" ".join(new_segs))
    if new_full.lower() != mf.lower():
        return False, seg_list

    return True, new_segs


# ---------------------------------------------------------------------------
# Format detection and fix dispatch
# ---------------------------------------------------------------------------

EAST_LATENCIES = ["low_latency", "medium_latency", "high_latency"]


def detect_format(obj: dict) -> str:
    if "segmented_pairs" in obj:
        return "salami"
    for lat in EAST_LATENCIES:
        if lat in obj:
            return "east"
    return "unknown"


def fix_east(obj: dict, manifest: str):
    """
    Fix all latency levels in a refined_east / EAST raw JSON.
    Returns (ok, fixed_obj):
      ok=False if ANY latency has token mismatch → discard whole utterance.
    """
    fixed = dict(obj)
    for lat in EAST_LATENCIES:
        if lat not in obj:
            continue
        eng = obj[lat].get("English", [])
        if not eng:
            continue
        ok, new_eng = restore_punct(eng, manifest)
        if not ok:
            return False, obj   # one bad latency → discard whole utterance
        fixed[lat] = dict(obj[lat])
        fixed[lat]["English"] = new_eng
    return True, fixed


def fix_salami(obj: dict, manifest: str):
    """
    Fix segmented_pairs in a salami raw JSON.
    Returns (ok, fixed_obj).
    """
    pairs = obj.get("segmented_pairs", [])
    if not pairs:
        return False, obj

    eng_segs = [p[0] for p in pairs]
    ok, new_eng = restore_punct(eng_segs, manifest)
    if not ok:
        return False, obj

    fixed = dict(obj)
    fixed["segmented_pairs"] = [[new_eng[i], pairs[i][1]]
                                 for i in range(len(pairs))]
    return True, fixed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",         required=True,
                    help="Input llm_output_raw directory")
    ap.add_argument("--out_dir",        required=True,
                    help="Output directory for fixed JSONs")
    ap.add_argument("--good_jsonl",     default=None,
                    help="good_*.jsonl listing utt_ids to process (if omitted, process ALL jsons in in_dir)")
    ap.add_argument("--out_good_jsonl", default=None,
                    help="Output good.jsonl (only kept utt_ids)")
    args = ap.parse_args()

    # Load utt_ids: from good_jsonl if provided, else scan all .json in in_dir
    good_rows = []  # keep original rows for rewriting (only used when good_jsonl given)
    if args.good_jsonl:
        print(f"Loading good utt_ids from {args.good_jsonl} ...")
        good_ids = set()
        with open(args.good_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    uid = obj.get("file") or obj.get("utt_id") or obj.get("id", "")
                    if isinstance(uid, str) and uid:
                        good_ids.add(uid)
                        good_rows.append((uid, line))
                except Exception:
                    good_ids.add(line)
                    good_rows.append((line, line))
    else:
        print(f"No --good_jsonl provided; scanning all .json files in {args.in_dir} ...")
        good_ids = set()
        for fname in os.listdir(args.in_dir):
            if fname.endswith(".json"):
                good_ids.add(fname[:-5])  # strip .json
    print(f"  {len(good_ids):,} utt_ids to process")

    os.makedirs(args.out_dir, exist_ok=True)

    kept_exact = kept_fixed = dropped_token = dropped_no_file = 0
    kept_ids = []

    for utt_id in tqdm(sorted(good_ids), desc="Utterances"):
        src_path = os.path.join(args.in_dir, f"{utt_id}.json")
        if not os.path.exists(src_path):
            dropped_no_file += 1
            continue

        try:
            with open(src_path, encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            dropped_no_file += 1
            continue

        manifest = normalize_ws(obj.get("input", ""))
        if not manifest:
            dropped_no_file += 1
            continue

        fmt = detect_format(obj)

        if fmt == "east":
            ok, fixed_obj = fix_east(obj, manifest)
        elif fmt == "salami":
            ok, fixed_obj = fix_salami(obj, manifest)
        else:
            dropped_token += 1
            continue

        if not ok:
            dropped_token += 1
            continue

        # Check if any change was made (for stats)
        changed = (fixed_obj is not obj)

        out_path = os.path.join(args.out_dir, f"{utt_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixed_obj, f, ensure_ascii=False)

        if changed:
            kept_fixed += 1
        else:
            kept_exact += 1
        kept_ids.append(utt_id)

    # Write updated good.jsonl using the cached rows
    kept_set = set(kept_ids)
    if args.out_good_jsonl:
        print(f"\nWriting {args.out_good_jsonl} ...")
        if good_rows:
            with open(args.out_good_jsonl, "w", encoding="utf-8") as f:
                for uid, raw_line in good_rows:
                    if uid in kept_set:
                        f.write(raw_line + "\n")
        else:
            # No good_jsonl was provided; write a simple id-only jsonl
            with open(args.out_good_jsonl, "w", encoding="utf-8") as f:
                for uid in sorted(kept_ids):
                    f.write(json.dumps({"file": uid}) + "\n")

    total = kept_exact + kept_fixed + dropped_token + dropped_no_file
    kept  = kept_exact + kept_fixed
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total good utt_ids      : {len(good_ids):,}")
    print(f"  Kept exact match        : {kept_exact:,}")
    print(f"  Kept after punct fix    : {kept_fixed:,}")
    print(f"  ─────────────────────────")
    print(f"  Total kept              : {kept:,}  ({100*kept/len(good_ids):.1f}%)")
    print(f"  Dropped token-mismatch  : {dropped_token:,}")
    print(f"  Dropped missing file    : {dropped_no_file:,}")
    print("=" * 60)
    print(f"\nFixed JSONs → {args.out_dir}")
    print(f"Good list   → {args.out_good_jsonl}")


if __name__ == "__main__":
    main()
