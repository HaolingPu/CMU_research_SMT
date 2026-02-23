#!/usr/bin/env python3
"""
Strict quality filter for GigaSpeech pipeline:
1) LLM JSON sanity
2) MFA TextGrid sanity
3) TextGrid tokens == .lab tokens
4) LLM English chunks can be matched monotonically in TextGrid tokens
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import tgt
from tqdm import tqdm


SKIP_TOKENS = {"sil", "sp"}
BAD_TOKENS = {"<unk>", "unk", "spn", "<eps>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find bad GigaSpeech instances with strict checks.")
    parser.add_argument("--llm-dir", required=True, help="LLM JSON directory (recursive).")
    parser.add_argument("--mfa-dir", required=True, help="MFA TextGrid directory (recursive).")
    parser.add_argument("--corpus-dir", required=True, help="MFA corpus .lab directory (recursive).")
    parser.add_argument("--good-jsonl", required=True, help="Output good jsonl path.")
    parser.add_argument("--bad-jsonl", required=True, help="Output bad jsonl path.")
    parser.add_argument(
        "--allow-one-word",
        action="store_true",
        help="Allow one-word English chunks (default: disallow).",
    )
    return parser.parse_args()


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def list_files(root: str, suffix: str) -> List[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(suffix):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def build_index(root: str, suffix: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    idx: Dict[str, str] = {}
    dup: Dict[str, int] = {}
    for p in list_files(root, suffix):
        utt = os.path.basename(p)[: -len(suffix)]
        if utt in idx:
            dup[utt] = dup.get(utt, 1) + 1
        else:
            idx[utt] = p
    return idx, dup


def load_lab_tokens(lab_path: str) -> Tuple[Optional[List[str]], List[str]]:
    try:
        with open(lab_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        toks = normalize_text(text).split()
        if not toks:
            return None, ["empty_lab"]
        return toks, []
    except Exception as e:
        return None, [f"lab_load_error:{type(e).__name__}:{str(e)}"]


def load_textgrid_tokens(textgrid_path: str) -> Tuple[Optional[List[str]], List[str]]:
    try:
        tg = tgt.read_textgrid(textgrid_path)
    except Exception as e:
        return None, [f"textgrid_parse_error:{type(e).__name__}:{str(e)}"]

    try:
        words = tg.get_tier_by_name("words").intervals
    except Exception:
        return None, ["missing_words_tier"]

    toks: List[str] = []
    for w in words:
        raw = (w.text or "").strip()
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


def is_one_word(text: str) -> bool:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text))) <= 1


def check_llm_json_obj(obj: Dict[str, Any], allow_one_word: bool) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []

    if "offline" in obj:
        levels = ["offline"]
    else:
        levels = ["low_latency", "medium_latency", "high_latency"]

    for lv in levels:
        if lv not in obj:
            reasons.append(f"missing_level:{lv}")
            continue

        block = obj.get(lv, {})
        if not isinstance(block, dict):
            reasons.append(f"level_not_dict:{lv}")
            continue

        eng = block.get("English")
        zh = block.get("Chinese")
        if not isinstance(eng, list) or not isinstance(zh, list):
            reasons.append(f"not_list:{lv}")
            continue
        if len(eng) == 0 or len(zh) == 0:
            reasons.append(f"empty_chunks:{lv}")
            continue
        if len(eng) != len(zh):
            reasons.append(f"len_mismatch:{lv}:{len(eng)}!={len(zh)}")
            continue

        for i, (e, z) in enumerate(zip(eng, zh)):
            if not isinstance(e, str) or not e.strip():
                reasons.append(f"bad_en_chunk:{lv}:{i}")
                break
            if not isinstance(z, str) or not z.strip():
                reasons.append(f"bad_zh_chunk:{lv}:{i}")
                break
            if not allow_one_word and is_one_word(e):
                reasons.append(f"one_word_chunk:{lv}:{i}")
                break

    return len(reasons) == 0, reasons, levels


def strict_corpus_check(lab_toks: List[str], tg_toks: List[str]) -> Tuple[bool, List[str]]:
    if lab_toks == tg_toks:
        return True, []

    n = min(len(lab_toks), len(tg_toks))
    mismatch_i = None
    for i in range(n):
        if lab_toks[i] != tg_toks[i]:
            mismatch_i = i
            break
    if mismatch_i is None:
        mismatch_i = n

    return False, [
        "token_mismatch_lab_vs_textgrid",
        f"lab_n={len(lab_toks)}",
        f"tg_n={len(tg_toks)}",
        f"mismatch_i={mismatch_i}",
        f"lab_slice={lab_toks[max(0, mismatch_i-6):mismatch_i+6]}",
        f"tg_slice={tg_toks[max(0, mismatch_i-6):mismatch_i+6]}",
    ]


def strict_llm_vs_textgrid(obj: Dict[str, Any], levels: List[str], tg_toks: List[str]) -> Tuple[bool, List[str]]:
    for lv in levels:
        eng_chunks = obj.get(lv, {}).get("English", [])
        cursor = 0
        for ci, chunk in enumerate(eng_chunks):
            for ti, tok in enumerate(normalize_text(chunk).split()):
                found = False
                for i in range(cursor, len(tg_toks)):
                    if tg_toks[i] == tok:
                        cursor = i + 1
                        found = True
                        break
                if not found:
                    return False, [f"llm_textgrid_mismatch:{lv}:chunk{ci}:tok{ti}:{tok}"]
    return True, []


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.good_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.bad_jsonl) or ".", exist_ok=True)

    llm_files = list_files(args.llm_dir, ".json")
    if not llm_files:
        raise FileNotFoundError(f"No JSON files in {args.llm_dir}")

    tg_index, tg_dup = build_index(args.mfa_dir, ".TextGrid")
    lab_index, lab_dup = build_index(args.corpus_dir, ".lab")

    if tg_dup:
        print(f"Warning: duplicate TextGrid basenames: {len(tg_dup)}")
    if lab_dup:
        print(f"Warning: duplicate LAB basenames: {len(lab_dup)}")

    good = 0
    bad = 0
    reason_counter: Counter[str] = Counter()

    with open(args.good_jsonl, "w", encoding="utf-8") as good_f, open(
        args.bad_jsonl, "w", encoding="utf-8"
    ) as bad_f:
        for llm_path in tqdm(llm_files, desc="Checking"):
            base = os.path.splitext(os.path.basename(llm_path))[0]
            reasons: List[str] = []

            try:
                obj = json.load(open(llm_path, "r", encoding="utf-8"))
            except Exception as e:
                reasons.append(f"llm_json_load_error:{type(e).__name__}:{str(e)}")
                obj = None

            utt_id = base
            if isinstance(obj, dict) and isinstance(obj.get("utt_id"), str) and obj["utt_id"].strip():
                utt_id = obj["utt_id"].strip()

            if not isinstance(obj, dict):
                item = {"file": utt_id, "path": llm_path, "reasons": reasons}
                bad_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                bad += 1
                reason_counter.update(reasons)
                continue

            if "error" in obj:
                reasons.append(f"has_error_field:{obj.get('error')}")

            llm_ok, llm_reasons, levels = check_llm_json_obj(obj, allow_one_word=args.allow_one_word)
            if not llm_ok:
                reasons.extend(llm_reasons)

            tg_path = tg_index.get(utt_id)
            if tg_path is None:
                reasons.append("missing_textgrid")
            lab_path = lab_index.get(utt_id)
            if lab_path is None:
                reasons.append("missing_lab")

            tg_toks = None
            if tg_path is not None:
                tg_toks, tg_reasons = load_textgrid_tokens(tg_path)
                reasons.extend(tg_reasons)

            lab_toks = None
            if lab_path is not None:
                lab_toks, lab_reasons = load_lab_tokens(lab_path)
                reasons.extend(lab_reasons)

            if tg_toks is not None and lab_toks is not None:
                ok_corpus, corpus_reasons = strict_corpus_check(lab_toks, tg_toks)
                if not ok_corpus:
                    reasons.extend(corpus_reasons)

            if tg_toks is not None and llm_ok and "error" not in obj:
                ok_align, align_reasons = strict_llm_vs_textgrid(obj, levels, tg_toks)
                if not ok_align:
                    reasons.extend(align_reasons)

            if reasons:
                item = {
                    "file": utt_id,
                    "path": llm_path,
                    "textgrid": tg_path,
                    "lab": lab_path,
                    "reasons": reasons,
                }
                bad_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                bad += 1
                reason_counter.update(reasons)
            else:
                good_item = {
                    "file": utt_id,
                    "path": llm_path,
                    "textgrid": tg_path,
                    "lab": lab_path,
                }
                good_f.write(json.dumps(good_item, ensure_ascii=False) + "\n")
                good += 1

    print("\n========== Summary ==========")
    print(f"LLM json count : {len(llm_files)}")
    print(f"Good           : {good}")
    print(f"Bad            : {bad}")
    print(f"good_jsonl     : {args.good_jsonl}")
    print(f"bad_jsonl      : {args.bad_jsonl}")
    # if reason_counter:
    #     print("\nTop reasons:")
    #     for r, c in reason_counter.most_common(20):
    #         print(f"  {c:6d}  {r}")


if __name__ == "__main__":
    main()