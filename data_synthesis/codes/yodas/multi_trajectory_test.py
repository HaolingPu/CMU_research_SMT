#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
import math
import tgt


# ============================================================
# Normalize (match your .lab clean_text)
# ============================================================
def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Load TextGrid words
# ============================================================
def load_word_alignment(textgrid_path: str):
    tg = tgt.read_textgrid(textgrid_path)
    tier = tg.get_tier_by_name("words")
    out = []
    for itv in tier.intervals:
        if not (itv.text or "").strip():
            continue
        out.append({
            "word": (itv.text or "").strip(),
            "start": float(itv.start_time),
            "end": float(itv.end_time),
        })
    return out


# ============================================================
# 3 matching strategies
# ============================================================
def idx_naive(tokens, mfa_tokens):
    """Each token: scan from 0, pick FIRST occurrence."""
    matched = []
    for t in tokens:
        for i, w in enumerate(mfa_tokens):
            if w == t:
                matched.append(i)
                break
    return matched


def idx_mono_chunk_reset(tokens, mfa_tokens):
    """
    WRONG mono: within-chunk monotonic, but cursor resets to 0 every chunk.
    This is what your previous test script did.
    """
    matched = []
    cursor = 0
    for t in tokens:
        for i in range(cursor, len(mfa_tokens)):
            if mfa_tokens[i] == t:
                matched.append(i)
                cursor = i + 1
                break
    return matched


def build_aligned_mono_global(chunks, mfa_words):
    """
    TRUE mono: cursor persists across chunks.
    """
    mfa_tokens = [normalize_text(w["word"]) for w in mfa_words]
    aligned = []
    cursor = 0

    for chunk in chunks:
        tokens = normalize_text(chunk).split()
        matched = []

        for t in tokens:
            for i in range(cursor, len(mfa_tokens)):
                if mfa_tokens[i] == t:
                    matched.append(i)
                    cursor = i + 1
                    break

        if not matched:
            aligned.append({"chunk": chunk, "start": None, "end": None, "idx": []})
        else:
            aligned.append({
                "chunk": chunk,
                "start": mfa_words[matched[0]]["start"],
                "end": mfa_words[matched[-1]]["end"],
                "idx": matched
            })
    return aligned


def build_aligned_per_chunk(chunks, mfa_words, mode: str):
    mfa_tokens = [normalize_text(w["word"]) for w in mfa_words]
    aligned = []

    for chunk in chunks:
        tokens = normalize_text(chunk).split()

        if mode == "naive":
            matched = idx_naive(tokens, mfa_tokens)
        elif mode == "mono_chunk_reset":
            matched = idx_mono_chunk_reset(tokens, mfa_tokens)
        else:
            raise ValueError("mode must be naive or mono_chunk_reset")

        if not matched:
            aligned.append({"chunk": chunk, "start": None, "end": None, "idx": []})
        else:
            aligned.append({
                "chunk": chunk,
                "start": min(mfa_words[i]["start"] for i in matched),
                "end": max(mfa_words[i]["end"] for i in matched),
                "idx": matched
            })

    return aligned


# ============================================================
# Assign by second (same as your logic)
# ============================================================
def assign_chunks_by_second(aligned_chunks):
    valid = [x for x in aligned_chunks if x["end"] is not None]
    if not valid:
        return [{"second": 0, "emit": []}]

    max_time = max(x["end"] for x in valid)
    eps = 1e-6
    max_sec = int(math.ceil(max_time - eps))

    emitted = set()
    timeline = []
    for sec in range(max_sec):
        sec_end = sec + 1
        emit = []
        for i, item in enumerate(aligned_chunks):
            if i in emitted:
                continue
            if item["end"] is not None and item["end"] <= sec_end:
                emit.append(item["chunk"])
                emitted.add(i)
        timeline.append({"second": sec, "emit": emit})
    return timeline


# ============================================================
# Pretty print helpers
# ============================================================
def fmt_idx(idx_list, mfa_words, mfa_tokens, limit=8):
    if not idx_list:
        return "(no match)"
    s = []
    for j in idx_list[:limit]:
        w = mfa_words[j]
        s.append(f"{j}:{mfa_tokens[j]}[{w['start']:.2f},{w['end']:.2f}]")
    if len(idx_list) > limit:
        s.append(f"...(+{len(idx_list)-limit})")
    return " | ".join(s)


# ============================================================
# Debug runner
# ============================================================
def run(textgrid_path, llm_json_path, level="low_latency", max_words=60, max_chunks=30):
    seg = json.load(open(llm_json_path, "r", encoding="utf-8"))
    mfa_words = load_word_alignment(textgrid_path)
    mfa_tokens = [normalize_text(w["word"]) for w in mfa_words]

    if "offline" in seg:
        level = "offline"
    chunks = seg[level]["English"]

    print("\n================ DEBUG INFO ================")
    print(f"TextGrid : {textgrid_path}")
    print(f"LLM JSON : {llm_json_path}")
    print(f"Level    : {level}")
    print("============================================\n")

    print("---- TextGrid words (head) ----")
    for i in range(min(len(mfa_tokens), max_words)):
        w = mfa_words[i]
        print(f"{i:4d}  {mfa_tokens[i]:15s} [{w['start']:.2f},{w['end']:.2f}]")
    print("--------------------------------\n")

    # aligned
    aligned_naive = build_aligned_per_chunk(chunks, mfa_words, mode="naive")
    aligned_mcr   = build_aligned_per_chunk(chunks, mfa_words, mode="mono_chunk_reset")
    aligned_mg    = build_aligned_mono_global(chunks, mfa_words)

    print("======== CHUNK ALIGNMENT COMPARISON ========")
    for i, ch in enumerate(chunks[:max_chunks]):
        toks = normalize_text(ch).split()

        a = aligned_naive[i]
        b = aligned_mcr[i]
        c = aligned_mg[i]

        print(f"\nCHUNK {i:02d}: {ch}")
        print(f"  tokens: {toks}")

        print(f"  naive          end={a['end']}")
        print(f"    {fmt_idx(a['idx'], mfa_words, mfa_tokens)}")

        print(f"  mono_chunk_reset end={b['end']}")
        print(f"    {fmt_idx(b['idx'], mfa_words, mfa_tokens)}")

        print(f"  mono_global     end={c['end']}")
        print(f"    {fmt_idx(c['idx'], mfa_words, mfa_tokens)}")

        # highlight if end differs
        ends = (a["end"], b["end"], c["end"])
        if len(set([x for x in ends if x is not None])) > 1:
            print("  >>> END DIFF <<<")

    # trajectories
    traj_naive = assign_chunks_by_second(aligned_naive)
    traj_mcr   = assign_chunks_by_second(aligned_mcr)
    traj_mg    = assign_chunks_by_second(aligned_mg)

    print("\n================ TRAJECTORY (3-way) =================")
    max_len = max(len(traj_naive), len(traj_mcr), len(traj_mg))
    for sec in range(max_len):
        n = traj_naive[sec]["emit"] if sec < len(traj_naive) else []
        r = traj_mcr[sec]["emit"]   if sec < len(traj_mcr) else []
        g = traj_mg[sec]["emit"]    if sec < len(traj_mg) else []

        print(f"sec {sec:02d}")
        print(f"  naive         : {n}")
        print(f"  mono_chunk_rst: {r}")
        print(f"  mono_global   : {g}")

        if (n != g) or (r != g):
            print("  >>> DIFFER <<<")

    print("\n=====================================================")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Compare naive vs mono_chunk_reset vs mono_global")
    ap.add_argument("--textgrid", required=True, help="Path to *.TextGrid")
    ap.add_argument("--llm", required=True, help="Path to LLM output *.json")
    ap.add_argument("--level", default="low_latency",
                    choices=["offline", "low_latency", "medium_latency", "high_latency"])
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--max-chunks", type=int, default=30)
    args = ap.parse_args()

    run(args.textgrid, args.llm, level=args.level, max_words=args.max_words, max_chunks=args.max_chunks)



# (SMT) [haolingp@babel-n5-16 codes]$ python /data/user_data/haolingp/data_synthesis/codes/yodas/multi_trajectory_test.py   --textgrid /data/user_data/haolingp/outputs/mfa_textgrid_output/en000/00000000/utt_en000_00000000_0055.TextGrid   --llm /data/user_data/haolingp/outputs/llm_output_modified/en000/00000000/utt_en000_00000000_0055.json   --level low_latency   --max-words 120   --max-chunks 25