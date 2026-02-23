#!/usr/bin/env python3
"""
check_gigaspeech_manifest.py

Compare final output source (concatenated) vs manifest src_text,
including punctuation. Output a JSON report with mismatches containing:
  - id (utt_id)
  - source (concatenated source from final output)
  - translation (concatenated target from final output)
  - manifest_src (src_text from manifest)
  - reasons (e.g. punct_only_diff, token_mismatch)
  - missing_punct / extra_punct

Only checks low_latency (source is the same across latencies).
"""

import os
import re
import csv
import json
import argparse
import sys


# ----------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Strip punctuation — same as pipeline normalize_text."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def punct_diff(s_manifest: str, s_final: str):
    """Return (missing_in_final, extra_in_final) as strings."""
    def punct_counts(s):
        counts = {}
        for c in s:
            if not c.isalnum() and not c.isspace() and c != "'":
                counts[c] = counts.get(c, 0) + 1
        return counts

    mc = punct_counts(s_manifest)
    fc = punct_counts(s_final)
    all_chars = set(mc) | set(fc)
    missing = ""
    extra = ""
    for c in sorted(all_chars):
        m = mc.get(c, 0)
        n = fc.get(c, 0)
        if m > n:
            missing += c * (m - n)
        elif n > m:
            extra += c * (n - m)
    return missing, extra


# ----------------------------------------------------------------
# Load manifest
# ----------------------------------------------------------------

def load_manifest(tsv_path: str) -> dict:
    """Returns dict: id -> src_text"""
    print(f"Loading manifest: {tsv_path}")
    manifest = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = row.get("id", "").strip()
            src = row.get("src_text", "").strip()
            if uid:
                manifest[uid] = src
    print(f"  Loaded {len(manifest)} entries from manifest\n")
    return manifest


# ----------------------------------------------------------------
# Process final output
# ----------------------------------------------------------------

def run_check(final_root: str, manifest: dict, output_json: str,
              latency: str = "low", max_entries: int = None):

    jsonl_pattern = f"{latency}_latency.jsonl"

    # Collect all matching JSONL files
    jsonl_files = []
    for dirpath, _, files in os.walk(final_root):
        for fn in files:
            if fn == jsonl_pattern:
                jsonl_files.append(os.path.join(dirpath, fn))
    jsonl_files.sort()
    print(f"Found {len(jsonl_files)} JSONL files matching '{jsonl_pattern}'")

    total = 0
    ok = 0
    mismatches = []
    missing_in_manifest = 0
    count = 0

    for fi, jfile in enumerate(jsonl_files):
        if fi % 500 == 0:
            print(f"  {fi}/{len(jsonl_files)} files, {total} entries so far...", flush=True)
        with open(jfile, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if max_entries is not None and count >= max_entries:
                    break
                count += 1
                total += 1

                try:
                    item = json.loads(line)
                except Exception:
                    continue

                utt_id = item.get("utt_id", "")
                source_list = item.get("source", [])
                target_list = item.get("target", [])

                # Concatenate non-empty segments
                src_concat = normalize_whitespace(
                    " ".join(s for s in source_list if isinstance(s, str) and s.strip())
                )
                tgt_concat = "".join(
                    t for t in target_list if isinstance(t, str)
                ).strip()

                manifest_src = manifest.get(utt_id)
                if manifest_src is None:
                    missing_in_manifest += 1
                    continue

                manifest_norm = normalize_whitespace(manifest_src)

                # Compare (case-insensitive, whitespace normalized)
                if src_concat.lower() == manifest_norm.lower():
                    ok += 1
                    continue

                # Mismatch — classify
                reasons = []
                final_toks = normalize_text(src_concat).split()
                manifest_toks = normalize_text(manifest_norm).split()

                if final_toks == manifest_toks:
                    reasons.append("tokens_ok_punct_differs")
                else:
                    reasons.append("token_mismatch")

                missing_p, extra_p = punct_diff(manifest_norm, src_concat)
                if missing_p or extra_p:
                    reasons.append(f"punct_diff:missing={repr(missing_p)},extra={repr(extra_p)}")

                mismatches.append({
                    "id":          utt_id,
                    "source":      src_concat,
                    "translation": tgt_concat,
                    "manifest_src": manifest_norm,
                    "reasons":     reasons,
                    "missing_punct": missing_p,
                    "extra_punct":   extra_p,
                })

    # Summary
    bad = len(mismatches)
    punct_only = sum(1 for m in mismatches if "tokens_ok_punct_differs" in m["reasons"])
    token_diff = sum(1 for m in mismatches if "token_mismatch" in m["reasons"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Entries checked      : {total}")
    print(f"  OK (exact match)     : {ok}   ({100*ok/total:.1f}%)" if total else "")
    print(f"  MISMATCH             : {bad}   ({100*bad/total:.1f}%)" if total else "")
    print(f"    ↳ punct-only diff  : {punct_only}")
    print(f"    ↳ token-level diff : {token_diff}")
    print(f"  Not in manifest      : {missing_in_manifest}")
    print("=" * 60)

    # Print first 5 examples
    for i, m in enumerate(mismatches[:5]):
        print(f"\n--- Example #{i+1}: {m['id']} ---")
        print(f"  Manifest : {m['manifest_src'][:120]}")
        print(f"  Final src: {m['source'][:120]}")
        print(f"  Missing punct: {repr(m['missing_punct'])}")
        print(f"  Extra punct  : {repr(m['extra_punct'])}")
        print(f"  Reasons  : {m['reasons']}")

    report = {
        "meta": {
            "final_root":  final_root,
            "latency":     latency,
            "total":       total,
            "ok":          ok,
            "bad":         bad,
            "punct_only_diff": punct_only,
            "token_diff":      token_diff,
            "not_in_manifest": missing_in_manifest,
        },
        "mismatches": mismatches,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport → {output_json}")


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir",  required=True,
                    help="Root of final_jsonl_dataset_EAST/")
    ap.add_argument("--manifest",   required=True,
                    help="Path to manifest TSV (train_xl_case_robust_asr-filtered.tsv)")
    ap.add_argument("--out_report", required=True,
                    help="Output JSON report path")
    ap.add_argument("--latency",    default="low",
                    choices=["low", "medium", "high", "offline"],
                    help="Which latency JSONL to check (default: low)")
    ap.add_argument("--max_entries", type=int, default=None,
                    help="Cap total entries for debug")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    run_check(
        final_root=args.final_dir,
        manifest=manifest,
        output_json=args.out_report,
        latency=args.latency,
        max_entries=args.max_entries,
    )