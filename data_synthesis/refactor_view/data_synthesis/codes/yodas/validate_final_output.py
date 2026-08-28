#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional

from tqdm import tqdm
import tgt


# -----------------------------
# Normalization (must match your corpus clean_text / normalize_text)
# -----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# TextGrid token extraction
# -----------------------------
SKIP_TOKENS = {"sil", "sp"}                      # ignored
BAD_TOKENS  = {"<unk>", "unk", "spn", "<eps>"}   # if appear -> bad immediately


def load_textgrid_word_tokens_and_time(
    textgrid_path: str
) -> Tuple[Optional[List[str]], Optional[float], List[str]]:
    """
    Returns:
      - tokens: list[str] or None
      - tg_seconds: float or None (TextGrid total seconds)
      - reasons: list[str] (empty if ok)
    """
    try:
        tg = tgt.read_textgrid(textgrid_path)
    except Exception as e:
        return None, None, [f"textgrid_parse_error:{type(e).__name__}:{str(e)}"]

    # total seconds: prefer TextGrid max_time, fallback to last interval end
    tg_seconds = None
    try:
        tg_seconds = float(getattr(tg, "max_time"))
    except Exception:
        tg_seconds = None

    try:
        tier = tg.get_tier_by_name("words")
    except Exception:
        return None, tg_seconds, ["missing_words_tier"]

    toks: List[str] = []
    last_end = 0.0

    for itv in tier.intervals:
        last_end = max(last_end, float(itv.end_time))
        raw = (itv.text or "").strip()
        if not raw:
            continue

        norm = normalize_text(raw)
        if not norm:
            continue

        for t in norm.split():
            if t in BAD_TOKENS:
                return None, tg_seconds, [f"bad_token_in_textgrid:{t}"]
            if t in SKIP_TOKENS:
                continue
            toks.append(t)

    if not toks:
        return None, tg_seconds if tg_seconds is not None else last_end, ["empty_textgrid_words"]

    if tg_seconds is None:
        tg_seconds = last_end

    return toks, tg_seconds, []


# -----------------------------
# Final-output source concat & length checks
# -----------------------------
def source_tokens_from_final(source_segments: List[str]) -> List[str]:
    """
    source_segments: list of per-second emitted source strings (can include "")
    We ignore empty strings, normalize, and tokenize.
    """
    joined = " ".join([s for s in source_segments if isinstance(s, str) and s.strip() != ""])
    return normalize_text(joined).split()


def expected_seconds_from_textgrid(tg_seconds: float, eps: float = 1e-6) -> int:
    """
    Your trajectory uses:
      max_sec = ceil(max_time - eps)
      seconds = max_sec
    """
    return int(math.ceil(tg_seconds - eps))


# -----------------------------
# Locate TextGrid path
# -----------------------------
def textgrid_path_for_utt(textgrid_root: str, utt_id: str) -> str:
    """
    Expected layout:
      textgrid_root/en000/00000000/utt_....TextGrid
    """
    parts = utt_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"bad_utt_id_format:{utt_id}")
    lang = parts[1]
    pq   = parts[2]
    return os.path.join(textgrid_root, lang, pq, f"{utt_id}.TextGrid")


def latency_from_filename(fn: str) -> Optional[str]:
    """
    low_latency.jsonl -> low
    medium_latency.jsonl -> medium
    high_latency.jsonl -> high
    offline_latency.jsonl -> offline
    """
    if not fn.endswith(".jsonl"):
        return None
    base = fn[:-len(".jsonl")]
    if not base.endswith("_latency"):
        return None
    lat = base[:-len("_latency")]
    lat = lat.lower().strip()
    if lat in {"low", "medium", "high", "offline"}:
        return lat
    return None


# -----------------------------
# Validate one latency jsonl file
# -----------------------------
def validate_final_jsonl(
    final_jsonl_path: str,
    textgrid_root: str,
    max_lines: Optional[int] = None,
) -> Dict[str, Any]:
    bad_items: List[Dict[str, Any]] = []
    total = 0
    ok = 0

    with open(final_jsonl_path, "r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            if max_lines is not None and line_i >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                item = json.loads(line)
            except Exception as e:
                bad_items.append({
                    "line": line_i,
                    "reasons": [f"jsonl_parse_error:{type(e).__name__}:{str(e)}"],
                })
                continue

            utt_id = item.get("utt_id")
            src = item.get("source")
            reasons = []

            if not isinstance(utt_id, str) or not utt_id:
                reasons.append("missing_or_bad_utt_id")
            if not isinstance(src, list):
                reasons.append("missing_or_bad_source_list")

            if reasons:
                bad_items.append({"utt_id": utt_id, "reasons": reasons})
                continue

            # TextGrid path
            try:
                tg_path = textgrid_path_for_utt(textgrid_root, utt_id)
            except Exception as e:
                bad_items.append({"utt_id": utt_id, "reasons": [str(e)]})
                continue

            if not os.path.exists(tg_path):
                bad_items.append({"utt_id": utt_id, "reasons": [f"missing_textgrid:{tg_path}"]})
                continue

            tg_toks, tg_seconds, tg_reasons = load_textgrid_word_tokens_and_time(tg_path)
            if tg_toks is None or tg_seconds is None or tg_reasons:
                bad_items.append({"utt_id": utt_id, "reasons": tg_reasons or ["textgrid_load_failed"]})
                continue

            # ---- Check 1: token sequence match ----
            final_toks = source_tokens_from_final(src)

            if final_toks != tg_toks:
                n = min(len(final_toks), len(tg_toks))
                mismatch_i = None
                for i in range(n):
                    if final_toks[i] != tg_toks[i]:
                        mismatch_i = i
                        break
                if mismatch_i is None and len(final_toks) != len(tg_toks):
                    mismatch_i = n

                bad_items.append({
                    "utt_id": utt_id,
                    "reasons": ["token_mismatch_final_source_vs_textgrid"],
                    "debug": {
                        "final_n": len(final_toks),
                        "tg_n": len(tg_toks),
                        "mismatch_i": mismatch_i,
                        "final_slice": final_toks[max(0, (mismatch_i or 0)-6):(mismatch_i or 0)+6],
                        "tg_slice": tg_toks[max(0, (mismatch_i or 0)-6):(mismatch_i or 0)+6],
                    }
                })
                continue

            # ---- Check 2: seconds length match ----
            final_seconds = len(src)
            expected_sec = expected_seconds_from_textgrid(tg_seconds)

            if final_seconds != expected_sec:
                bad_items.append({
                    "utt_id": utt_id,
                    "reasons": ["seconds_length_mismatch"],
                    "debug": {
                        "final_seconds": final_seconds,
                        "textgrid_seconds_float": tg_seconds,
                        "expected_seconds_ceil": expected_sec,
                    }
                })
                continue

            ok += 1

    return {
        "file": final_jsonl_path,
        "total": total,
        "ok": ok,
        "bad": total - ok,
        "bad_items": bad_items,
    }


# -----------------------------
# Collect files: compatible with SALAMI (offline only)
# -----------------------------
def collect_latency_jsonl_files(
    final_root: str,
    only_lang: Optional[str],
    first_pq: Optional[int],
    mode: str,  # "salami" | "east" | "auto"
) -> List[str]:
    """
    mode:
      - salami: only offline_latency.jsonl
      - east:   only low/medium/high_latency.jsonl
      - auto:   whatever *_latency.jsonl exists in dir
    """
    mode = mode.lower()
    if mode not in {"salami", "east", "auto"}:
        raise ValueError(f"bad --mode: {mode}")

    jsonl_files: List[str] = []

    for lang in sorted(os.listdir(final_root)):
        lang_dir = os.path.join(final_root, lang)
        if not os.path.isdir(lang_dir):
            continue
        if only_lang and lang != only_lang:
            continue

        pqs = sorted([
            pq for pq in os.listdir(lang_dir)
            if os.path.isdir(os.path.join(lang_dir, pq)) and pq.isdigit() and len(pq) == 8
        ])
        if first_pq is not None:
            pqs = pqs[:first_pq]

        for pq in pqs:
            pq_dir = os.path.join(lang_dir, pq)
            if mode == "salami":
                candidates = ["offline_latency.jsonl"]
            elif mode == "east":
                candidates = ["low_latency.jsonl", "medium_latency.jsonl", "high_latency.jsonl"]
            else:
                # auto: any *_latency.jsonl
                candidates = [fn for fn in os.listdir(pq_dir) if fn.endswith("_latency.jsonl")]

            for fn in sorted(candidates):
                p = os.path.join(pq_dir, fn)
                if os.path.exists(p):
                    jsonl_files.append(p)

    return sorted(jsonl_files)


def main():
    ap = argparse.ArgumentParser(
        description="Validate final trajectory jsonl vs TextGrid tokens + duration (compatible with SALAMI offline)."
    )
    ap.add_argument("--final_root", required=True,
                    help="Root of final output dir (contains lang/pq/*_latency.jsonl)")
    ap.add_argument("--textgrid_root", required=True,
                    help="Root of TextGrids: textgrid_root/en000/00000000/utt...TextGrid")
    ap.add_argument("--only_lang", default=None, help="Only process one lang, e.g. en000")
    ap.add_argument("--first_pq", type=int, default=10, help="Only process first N parquet dirs per lang (sorted).")
    ap.add_argument("--max_lines", type=int, default=None, help="Cap lines per jsonl file (debug).")
    ap.add_argument("--mode", type=str, default="auto",
                    help="salami|east|auto. salami checks offline_latency only.")
    ap.add_argument("--out_json", required=True, help="Write a single JSON report here.")
    args = ap.parse_args()

    jsonl_files = collect_latency_jsonl_files(
        final_root=args.final_root,
        only_lang=args.only_lang,
        first_pq=args.first_pq,
        mode=args.mode,
    )

    print(f"Found {len(jsonl_files)} latency jsonl files under {args.final_root} "
          f"(only_lang={args.only_lang}, first_pq={args.first_pq}, mode={args.mode})")

    report: Dict[str, Any] = {
        "meta": {
            "final_root": args.final_root,
            "textgrid_root": args.textgrid_root,
            "only_lang": args.only_lang,
            "first_pq": args.first_pq,
            "max_lines": args.max_lines,
            "mode": args.mode,
        },
        "files": [],
        "summary": {"total": 0, "ok": 0, "bad": 0},
    }

    grand_total = 0
    grand_ok = 0
    grand_bad = 0

    for p in tqdm(jsonl_files, desc="Validating files"):
        stats = validate_final_jsonl(
            final_jsonl_path=p,
            textgrid_root=args.textgrid_root,
            max_lines=args.max_lines,
        )

        rel = os.path.relpath(p, args.final_root)
        lat = latency_from_filename(os.path.basename(p))

        print(f"\n== {rel} ==")
        print(f"  latency={lat} total={stats['total']} ok={stats['ok']} bad={stats['bad']}")

        report["files"].append({
            "rel_path": rel,
            "abs_path": p,
            "latency": lat,
            "total": stats["total"],
            "ok": stats["ok"],
            "bad": stats["bad"],
            "bad_items": stats["bad_items"],
        })

        grand_total += stats["total"]
        grand_ok += stats["ok"]
        grand_bad += stats["bad"]

    report["summary"]["total"] = grand_total
    report["summary"]["ok"] = grand_ok
    report["summary"]["bad"] = grand_bad

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n================ SUMMARY ================")
    print(f"TOTAL: {grand_total}")
    print(f"OK   : {grand_ok}")
    print(f"BAD  : {grand_bad}")
    print(f"Report JSON: {args.out_json}")
    print("=========================================")


if __name__ == "__main__":
    main()


# python validate_final_output.py \
#   --final_root /data/user_data/haolingp/outputs/EAST/final_jsonl_dataset_EAST \
#   --textgrid_root /data/user_data/haolingp/outputs/textgrids \
#   --only_lang en000 \
#   --first_pq 10 \
#   --mode east \
#   --out_json /data/user_data/haolingp/outputs/EAST/validate_report_EAST.json
