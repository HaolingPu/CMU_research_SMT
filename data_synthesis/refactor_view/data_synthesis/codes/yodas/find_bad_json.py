#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
find_bad_json.py

Quality filter for:
1) LLM JSON sanity (structure/length/type)
2) MFA TextGrid sanity (words tier exists, non-empty, no <unk>/spn/etc)
3) Strict corpus check: TextGrid tokens == .lab tokens (must match your clean_text)
4) NEW: Strict LLM-vs-TextGrid monotonic match (mono-global):
   - For each level, all tokens in LLM English chunks must appear in TextGrid words tier,
     in order. Any missing token => BAD.

Directory expectations:
- LLM JSON:
    <llm_root>/<lang>/<pq>/utt_....json
- TextGrid:
    <mfa_root>/<lang>/<pq>/utt_....TextGrid
- Corpus (.lab):
    <corpus_root>/<lang>/<pq>/utt_....lab

Outputs:
- good_*.jsonl: {"file": "<utt_base>", "pq": "<pq>"}
- bad_*.jsonl : {"file": "<utt_base>", "pq": "<pq>", "reasons":[...]}
"""

import os
import json
import re
import argparse
from tqdm import tqdm
import tgt


# ============================================================
# Normalization (MUST match your .lab creation clean_text)
# ============================================================
def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# File collection
# ============================================================
def collect_files_single_parquet(parquet_dir: str, ext: str):
    """
    Returns: {filename_base: full_path}
    """
    mapping = {}
    if not os.path.exists(parquet_dir):
        return mapping

    for fn in os.listdir(parquet_dir):
        if fn.endswith(ext):
            full = os.path.join(parquet_dir, fn)
            base = fn[:-len(ext)]
            mapping[base] = full
    return mapping


# ============================================================
# LLM JSON checks (strict)
# ============================================================
def check_llm_json_obj(data: dict):
    """
    Returns: (ok: bool, reasons: list[str])
    """
    reasons = []

    if "offline" in data:
        levels = ["offline"]
    else:
        levels = ["low_latency", "medium_latency", "high_latency"]

    for lv in levels:
        if lv not in data:
            reasons.append(f"missing_level:{lv}")
            continue

        block = data.get(lv, {})
        if "English" not in block or "Chinese" not in block:
            reasons.append(f"missing_lang_keys:{lv}")
            continue

        eng = block["English"]
        zh = block["Chinese"]

        if not isinstance(eng, list) or not isinstance(zh, list):
            reasons.append(f"not_list:{lv}")
            continue

        if len(eng) == 0 or len(zh) == 0:
            reasons.append(f"empty_chunks:{lv}")
            continue

        if len(eng) != len(zh):
            reasons.append(f"len_mismatch:{lv}:{len(eng)}!={len(zh)}")
            continue

        for i, x in enumerate(eng):
            if not isinstance(x, str):
                reasons.append(f"en_chunk_not_str:{lv}:{i}")
                break

        for i, x in enumerate(zh):
            if not isinstance(x, str):
                reasons.append(f"zh_chunk_not_str:{lv}:{i}")
                break

    return (len(reasons) == 0), reasons


def load_llm_json(full_path: str):
    """
    Returns: (seg: dict|None, reasons: list[str])
    """
    try:
        seg = json.load(open(full_path, "r", encoding="utf-8"))
        if not isinstance(seg, dict):
            return None, [f"llm_json_not_dict:{type(seg).__name__}"]
        return seg, []
    except Exception as e:
        return None, [f"llm_json_load_error:{type(e).__name__}:{str(e)}"]


# ============================================================
# TextGrid + corpus strict checks
# ============================================================
SKIP_TOKENS = {"sil", "sp"}                     # ignore these
BAD_TOKENS = {"<unk>", "unk", "spn", "<eps>"}   # if appear -> bad immediately


def load_lab_tokens(lab_path: str):
    with open(lab_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return normalize_text(text).split()


def load_textgrid_tokens(textgrid_path: str):
    """
    Returns: (tokens: list[str] | None, reasons: list[str])
    """
    try:
        tg = tgt.read_textgrid(textgrid_path)
    except Exception as e:
        return None, [f"textgrid_parse_error:{type(e).__name__}:{str(e)}"]

    try:
        tier = tg.get_tier_by_name("words")
    except Exception:
        return None, ["missing_words_tier"]

    toks = []
    for itv in tier.intervals:
        raw = (itv.text or "").strip()
        if not raw:
            continue

        norm = normalize_text(raw)
        if not norm:
            continue

        for t in norm.split():
            if t in BAD_TOKENS:
                return None, [f"bad_token_in_textgrid:{t}"]
            if t in SKIP_TOKENS:
                continue
            toks.append(t)

    if not toks:
        return None, ["empty_textgrid_words"]

    return toks, []


def check_textgrid(textgrid_path: str):
    toks, reasons = load_textgrid_tokens(textgrid_path)
    return (toks is not None and len(reasons) == 0), reasons


def strict_corpus_check(textgrid_path: str, lab_path: str):
    """
    Strict: tokens(TextGrid words tier, filtered) == tokens(lab)
    Returns: (ok: bool, reasons: list[str])
    """
    if not os.path.exists(lab_path):
        return False, ["missing_lab"]

    lab_toks = load_lab_tokens(lab_path)
    if not lab_toks:
        return False, ["empty_lab"]

    tg_toks, tg_reasons = load_textgrid_tokens(textgrid_path)
    if tg_toks is None:
        return False, tg_reasons

    if lab_toks != tg_toks:
        n = min(len(lab_toks), len(tg_toks))
        mismatch_i = None
        for i in range(n):
            if lab_toks[i] != tg_toks[i]:
                mismatch_i = i
                break
        if mismatch_i is None and len(lab_toks) != len(tg_toks):
            mismatch_i = n

        reasons = [
            "token_mismatch",
            f"lab_n={len(lab_toks)}",
            f"tg_n={len(tg_toks)}",
            f"mismatch_i={mismatch_i}",
            f"lab_slice={lab_toks[max(0, mismatch_i-5):mismatch_i+5]}",
            f"tg_slice={tg_toks[max(0, mismatch_i-5):mismatch_i+5]}",
        ]
        return False, reasons

    return True, []


# ============================================================
# NEW: strict LLM vs TextGrid (mono-global)
# ============================================================
def strict_llm_vs_textgrid(seg: dict, textgrid_path: str):
    """
    Strict mono-global check:
    For each level, all tokens in LLM English chunks must be found in TextGrid words
    in order (monotonic). Any missing token => bad.
    Returns: (ok: bool, reason: str)
    """
    tg_toks, tg_reasons = load_textgrid_tokens(textgrid_path)
    if tg_toks is None:
        return False, "LLM_TEXTGRID_TEXTGRID_BAD:" + ",".join(tg_reasons)

    if "offline" in seg:
        levels = ["offline"]
    else:
        levels = ["low_latency", "medium_latency", "high_latency"]

    for lv in levels:
        if lv not in seg:
            return False, f"LLM_TEXTGRID_MISSING_LEVEL:{lv}"
        if "English" not in seg[lv]:
            return False, f"LLM_TEXTGRID_MISSING_EN:{lv}"

        eng_chunks = seg[lv]["English"]
        cursor = 0

        for ci, chunk in enumerate(eng_chunks):
            toks = normalize_text(chunk).split()
            for ti, t in enumerate(toks):
                found = False
                for i in range(cursor, len(tg_toks)):
                    if tg_toks[i] == t:
                        cursor = i + 1
                        found = True
                        break
                if not found:
                    return False, f"LLM_TEXTGRID_MISMATCH:{lv}:chunk{ci}:token{ti}:{t}"

    return True, "OK"


# ============================================================
# Process all parquets for a language
# ============================================================
def process_language(llm_root: str, mfa_root: str, corpus_root: str,
                     save_good: str, save_bad: str, lang: str):
    print(f"\n===== Processing {lang} =====")

    lang_llm_dir = os.path.join(llm_root, lang)
    lang_mfa_dir = os.path.join(mfa_root, lang)
    lang_corpus_dir = os.path.join(corpus_root, lang)

    if not os.path.exists(lang_llm_dir):
        print(f"ERROR: LLM directory not found: {lang_llm_dir}")
        return
    if not os.path.exists(lang_mfa_dir):
        print(f"ERROR: MFA directory not found: {lang_mfa_dir}")
        return
    if not os.path.exists(lang_corpus_dir):
        print(f"ERROR: CORPUS directory not found: {lang_corpus_dir}")
        return

    parquet_dirs = sorted([
        d for d in os.listdir(lang_llm_dir)
        if os.path.isdir(os.path.join(lang_llm_dir, d)) and d.isdigit() and len(d) == 8
    ])
    print(f"Found {len(parquet_dirs)} parquet directories")

    # clear outputs
    open(save_good, "w").close()
    open(save_bad, "w").close()

    total_good = 0
    total_bad = 0

    for pq in parquet_dirs:
        llm_parquet_dir = os.path.join(lang_llm_dir, pq)
        mfa_parquet_dir = os.path.join(lang_mfa_dir, pq)
        corpus_parquet_dir = os.path.join(lang_corpus_dir, pq)

        llm_map = collect_files_single_parquet(llm_parquet_dir, ".json")
        mfa_map = collect_files_single_parquet(mfa_parquet_dir, ".TextGrid")

        # union keys: if either side exists, we evaluate
        all_keys = sorted(set(llm_map.keys()) | set(mfa_map.keys()))

        pq_good = 0
        pq_bad = 0

        for key in tqdm(all_keys, desc=f"{lang}/{pq}"):
            reasons = []
            ok = True

            # ---------- presence checks ----------
            if key not in llm_map:
                ok = False
                reasons.append("missing_llm_json")
                seg = None
            else:
                seg, load_reasons = load_llm_json(llm_map[key])
                if seg is None:
                    ok = False
                    reasons.extend(load_reasons)

            if key not in mfa_map:
                ok = False
                reasons.append("missing_textgrid")

            # ---------- content checks ----------
            if ok and seg is not None:
                llm_ok, llm_reasons = check_llm_json_obj(seg)
                if not llm_ok:
                    ok = False
                    reasons.extend(llm_reasons)

            if ok:
                tg_ok, tg_reasons = check_textgrid(mfa_map[key])
                if not tg_ok:
                    ok = False
                    reasons.extend(tg_reasons)

            # ---------- NEW: strict LLM vs TextGrid ----------
            if ok and seg is not None:
                align_ok, align_reason = strict_llm_vs_textgrid(seg, mfa_map[key])
                if not align_ok:
                    ok = False
                    reasons.append(align_reason)

            # ---------- strict corpus match (TextGrid vs .lab) ----------
            if ok:
                lab_path = os.path.join(corpus_parquet_dir, key + ".lab")
                corpus_ok, corpus_reasons = strict_corpus_check(mfa_map[key], lab_path)
                if not corpus_ok:
                    ok = False
                    reasons.extend(corpus_reasons)

            item = {"file": key, "pq": pq}
            if ok:
                with open(save_good, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                pq_good += 1
                total_good += 1
            else:
                item["reasons"] = reasons
                with open(save_bad, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                pq_bad += 1
                total_bad += 1

        print(f"  [{pq}] Good: {pq_good}, Bad: {pq_bad}")

    print(f"\n===== {lang} SUMMARY =====")
    print(f"Total Good: {total_good}")
    print(f"Total Bad : {total_bad}")
    print(f"Total     : {total_good + total_bad}")
    print("=" * 30)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Quality check LLM JSON + MFA TextGrid + strict corpus + strict LLM-vs-TextGrid monotonic match."
    )
    parser.add_argument("--llm-root", type=str, required=True,
                        help="Root dir for LLM outputs: <llm_root>/<lang>/<pq>/*.json")
    parser.add_argument("--mfa-root", type=str, required=True,
                        help="Root dir for MFA TextGrid: <mfa_root>/<lang>/<pq>/*.TextGrid")
    parser.add_argument("--corpus-root", type=str, required=True,
                        help="Root dir for MFA corpus (wav/lab): <corpus_root>/<lang>/<pq>/*.lab")
    parser.add_argument("--lang", type=str, default="en000")
    parser.add_argument("--output-dir", type=str, default="/data/user_data/haolingp/outputs")
    parser.add_argument("--good-name", type=str, default=None)
    parser.add_argument("--bad-name", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    good_path = os.path.join(
        args.output_dir,
        args.good_name if args.good_name else f"good_EAST_{args.lang}_all.jsonl",
    )
    bad_path = os.path.join(
        args.output_dir,
        args.bad_name if args.bad_name else f"bad_EAST_{args.lang}_all.jsonl",
    )

    process_language(
        llm_root=args.llm_root,
        mfa_root=args.mfa_root,
        corpus_root=args.corpus_root,
        save_good=good_path,
        save_bad=bad_path,
        lang=args.lang,
    )

    print("\n✅ All quality checks completed!")
    print("Results saved:")
    print(f"  - {good_path}")
    print(f"  - {bad_path}")


if __name__ == "__main__":
    main()
