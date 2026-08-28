#!/usr/bin/env python3
"""
fix_llm_raw.py

逐 segment 对比 English 尾部标点和对应 Chinese 尾部标点：

第一步：看当前中文 segment 的结尾标点类型是否已经匹配英文（同属 sent/clause/colon 类别）→ 匹配就跳过

第二步（move-only，优先）：如果不匹配，看下一个中文 segment 开头有没有同类标点 → 有的话把它从下一段开头移到当前段结尾（joined text 不变，只是标点边界换位）

第三步（insert，可选）：如果找不到能移的标点，且开了 --zh_punct_allow_insert，就按 EN2ZH_PUNCT 映射（.→。 ,→， ?→？ 等）直接插入一个中文标点

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
from collections import Counter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Chinese target punctuation diagnostics
# ---------------------------------------------------------------------------

EN_SENT_END = set('.?!')
ZH_SENT_END = set('。？！')
EN_CLAUSE   = set(',;')
ZH_CLAUSE   = set('，；')
EN_COLON    = set(':')
ZH_COLON    = set('：')

EN_TRAIL_PUNCT = EN_SENT_END | EN_CLAUSE | EN_COLON
ZH_TRAIL_PUNCT = ZH_SENT_END | ZH_CLAUSE | ZH_COLON
ZH_CLOSERS = set('”’"\')）】》」』〕】')

EN2ZH_PUNCT = {
    '.': '。',
    '?': '？',
    '!': '！',
    ',': '，',
    ';': '；',
    ':': '：',
}


def _count_chars(text: str, charset: set) -> int:
    return sum(1 for ch in text if ch in charset)


def check_zh_punct(pairs: list, manifest: str) -> dict:
    """
    Check whether the joined Chinese target has sentence-ending punctuation
    consistent with the English manifest.

    Returns a dict with keys:
      flagged      : bool   — True if this utterance has a punct issue
      reason       : str
      en_sent_end  : int    — count of [.?!] in manifest
      zh_sent_end  : int    — count of [。？！] in joined Chinese
      zh_joined    : str    — joined Chinese target (first 120 chars)
    """
    en_sent = _count_chars(manifest, EN_SENT_END)
    zh_segs = [p[1] for p in pairs if isinstance(p, list) and len(p) == 2]
    zh_joined = "".join(zh_segs)
    zh_sent = _count_chars(zh_joined, ZH_SENT_END)

    flagged = False
    reason = ""
    if en_sent >= 2 and zh_sent == 0:
        flagged = True
        reason = f"en_sent={en_sent} but zh_sent=0"
    elif abs(en_sent - zh_sent) >= 3:
        flagged = True
        reason = f"en_sent={en_sent} zh_sent={zh_sent} diff={en_sent-zh_sent:+d}"

    return {
        "flagged":     flagged,
        "reason":      reason,
        "en_sent_end": en_sent,
        "zh_sent_end": zh_sent,
        "zh_joined":   zh_joined[:120],
    }


def _punct_bucket(ch: str) -> str:
    if ch in EN_SENT_END or ch in ZH_SENT_END:
        return "sent"
    if ch in EN_CLAUSE or ch in ZH_CLAUSE:
        return "clause"
    if ch in EN_COLON or ch in ZH_COLON:
        return "colon"
    return ""


def _trailing_punct_signal(text: str, punct_set: set) -> str:
    """
    Returns the trailing punctuation char (if any) before optional closing quotes/brackets.
    """
    s = (text or "").rstrip()
    if not s:
        return ""
    i = len(s) - 1
    while i >= 0 and s[i] in ZH_CLOSERS:
        i -= 1
    if i >= 0 and s[i] in punct_set:
        return s[i]
    return ""


def _leading_punct_signal(text: str, punct_set: set) -> str:
    s = (text or "")
    i = 0
    while i < len(s) and s[i].isspace():
        i += 1
    if i < len(s) and s[i] in punct_set:
        return s[i]
    return ""


def _insert_before_zh_closers(text: str, punct: str) -> str:
    s = (text or "")
    if not s:
        return s
    rstripped = s.rstrip()
    suffix_ws = s[len(rstripped):]
    i = len(rstripped)
    while i > 0 and rstripped[i - 1] in ZH_CLOSERS:
        i -= 1
    return rstripped[:i] + punct + rstripped[i:] + suffix_ws


# ASCII punct that LLMs sometimes produce in Chinese text
_ASCII_TO_ZH_PUNCT = {',': '，', '.': '。', '?': '？', '!': '！', ';': '；', ':': '：'}


def _normalize_zh_trailing_ascii_punct(text: str) -> str:
    """
    If a Chinese segment ends with ASCII punctuation (before any closers),
    replace it with the corresponding full-width Chinese punctuation.

    Guards:
      - Segment must contain at least one CJK character (skip pure-English/numeric segs).
      - The character immediately before the ASCII punct must be a CJK character or a
        full-width ZH punct, so that English words inside quotes like 'he said "hello."'
        are NOT normalized (the '.' follows 'o', not a CJK char).
    """
    if not text:
        return text
    if not any('\u4e00' <= c <= '\u9fff' for c in text):
        return text
    s = text.rstrip()
    suffix_ws = text[len(s):]
    i = len(s) - 1
    while i >= 0 and s[i] in ZH_CLOSERS:
        i -= 1
    if i >= 0 and s[i] in _ASCII_TO_ZH_PUNCT:
        prev_is_cjk_or_zh_punct = (
            i > 0 and (
                '\u4e00' <= s[i - 1] <= '\u9fff'
                or s[i - 1] in ZH_TRAIL_PUNCT
            )
        )
        if prev_is_cjk_or_zh_punct:
            s = s[:i] + _ASCII_TO_ZH_PUNCT[s[i]] + s[i + 1:]
    return s + suffix_ws


def sync_zh_punct_with_en(eng_segs: list, zh_segs: list, allow_insert: bool = False):
    """
    Heuristic Chinese punctuation sync without reference.

    Strategy:
      0) normalize ASCII trailing punct in Chinese segments to full-width (e.g. "," -> "，")
      1) if zh segment already has any trailing ZH punct → leave it alone (avoid double-punct)
      2) move-only (preferred): if zh punctuation already exists at next segment head,
         move it back to current segment tail to mirror EN segmentation; preserves joined ZH text
      3) optional insert: if still missing, insert mapped Chinese punctuation by EN trailing mark

    Returns (new_zh_segs, changed)
    """
    if len(eng_segs) != len(zh_segs):
        return zh_segs, False

    # Step 0: normalize ASCII trailing punct in Chinese segments.
    # This fixes LLM outputs like "他是编辑这一事实," -> "他是编辑这一事实，"
    # and avoids double-punct bugs when we later try to add Chinese punct.
    out = [_normalize_zh_trailing_ascii_punct(z) if isinstance(z, str) else z
           for z in zh_segs]
    changed = out != list(zh_segs)

    for i, en in enumerate(eng_segs):
        en_sig = _trailing_punct_signal(en, EN_TRAIL_PUNCT)
        if not en_sig:
            continue
        en_bucket = _punct_bucket(en_sig)
        if not en_bucket:
            continue

        cur = out[i]
        if not isinstance(cur, str) or not cur:
            continue

        cur_tail = _trailing_punct_signal(cur, ZH_TRAIL_PUNCT)
        # If the segment already ends with any ZH punct (even wrong class), leave it alone.
        # Adding more punct on top of existing punct creates errors like "，。" or "，，".
        if cur_tail:
            continue

        # Prefer moving punctuation from the next segment's leading position.
        if i + 1 < len(out) and isinstance(out[i + 1], str) and out[i + 1]:
            nxt = out[i + 1]
            nxt_head = _leading_punct_signal(nxt, ZH_TRAIL_PUNCT)
            if nxt_head and _punct_bucket(nxt_head) == en_bucket:
                # Move one leading punct char from next -> current (preserves joined text)
                j = 0
                while j < len(nxt) and nxt[j].isspace():
                    j += 1
                out[i] = _insert_before_zh_closers(cur, nxt[j])
                out[i + 1] = nxt[:j] + nxt[j + 1:]
                changed = True
                continue

        # No local punctuation to move; optionally insert a mapped one.
        if allow_insert:
            mapped = EN2ZH_PUNCT.get(en_sig)
            if mapped:
                out[i] = _insert_before_zh_closers(cur, mapped)
                changed = True

    return out, changed


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Normalize for word counting — must stay consistent with word_spans regex."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_tokens_for_match(s: str) -> str:
    """Lenient normalization used ONLY for the token-equality check.

    Handles common LLM-vs-manifest surface discrepancies:
      - thousands-separator commas : 3,000  -> 3000
      - intra-word hyphens         : COVID-19 -> covid19

    NOTE: does NOT merge abbreviation dots (U.S. -> us) because that
    changes the word_spans count and would break restore_punct alignment.
    """
    s = (s or "").lower()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)          # 3,000 -> 3000
    s = re.sub(r"(?<=[a-z0-9])-+(?=[a-z0-9])", "", s)  # COVID-19 -> covid19
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

    # Token-level check (lenient: handles 3,000/3000, COVID-19/COVID19, etc.)
    src_toks = normalize_tokens_for_match(full).split()
    mf_toks  = normalize_tokens_for_match(mf).split()
    if src_toks != mf_toks:
        return False, seg_list

    # Already exact?
    if full == mf:
        return True, seg_list

    # Word spans in manifest
    word_spans = [(m.start(), m.end())
                  for m in re.finditer(r"[a-zA-Z0-9']+", mf)]  #在 manifest 上找每个词的 span
    if not word_spans:
        return False, seg_list

    # Words per segment
    word_counts = [len(normalize_text(s).split()) for s in seg_list]
    if sum(word_counts) != len(word_spans):  #所有 segment 的词数总和是否等于 manifest 的词数
        return False, seg_list

    new_segs = []
    word_cursor = 0
    mf_pos = 0

    for k, seg in enumerate(seg_list):
        n = word_counts[k]
        if n == 0:
            new_segs.append("")
            continue

        last_w = word_cursor + n - 1 #find last word ending position in mf
        end_lw = word_spans[last_w][1]

        # Trailing punct (non-space chars right after last word) important!
        if last_w + 1 < len(word_spans):
            between = mf[end_lw: word_spans[last_w + 1][0]]
            m = re.match(r'[^\s]*', between)
            trailing = m.group() if m else ''  
        else:
            trailing = mf[end_lw:].rstrip()

        # create new segment
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


def fix_east(obj: dict, manifest: str, partial: bool = False,
             sync_zh_punct: bool = False, zh_allow_insert: bool = False):
    """
    Fix all latency levels in a refined_east / EAST raw JSON.
    Returns (ok, fixed_obj, changed).

    partial=False (default): any latency mismatch → discard whole utterance.
    partial=True:  drop only the bad latency key; keep utterance if ≥1 latency ok.
    sync_zh_punct: heuristically sync Chinese punctuation to repaired English per latency.
    """
    fixed = dict(obj)
    changed = False
    any_ok = False
    for lat in EAST_LATENCIES:
        if lat not in obj:
            continue
        eng = obj[lat].get("English", [])
        if not eng:
            any_ok = True   # latency present but empty — not a mismatch
            continue
        ok, new_eng = restore_punct(eng, manifest)
        if not ok:
            if partial:
                del fixed[lat]  # drop only this latency
                continue
            else:
                return False, obj, False  # all-or-nothing
        fixed[lat] = dict(obj[lat])
        fixed[lat]["English"] = new_eng
        if new_eng != eng:
            changed = True

        if sync_zh_punct:
            zh = obj[lat].get("Chinese", [])
            new_zh, zh_chg = sync_zh_punct_with_en(new_eng, zh, allow_insert=zh_allow_insert)
            fixed[lat]["Chinese"] = new_zh
            if zh_chg:
                changed = True

        any_ok = True
    if not any_ok:
        return False, obj, False
    return True, fixed, changed


def fix_salami(obj: dict, manifest: str, sync_zh_punct: bool = False, zh_allow_insert: bool = False):
    """
    Fix segmented_pairs in a salami raw JSON.
    Returns (ok, fixed_obj, changed).
    """
    pairs = obj.get("segmented_pairs", [])
    if not pairs:
        return False, obj, False

    eng_segs = [p[0] for p in pairs]
    ok, new_eng = restore_punct(eng_segs, manifest)
    if not ok:
        return False, obj, False

    eng_changed = (new_eng != eng_segs)

    zh_segs = [p[1] for p in pairs]
    zh_changed = False
    if sync_zh_punct:
        zh_segs, zh_changed = sync_zh_punct_with_en(
            new_eng, zh_segs, allow_insert=zh_allow_insert
        )

    fixed = dict(obj)
    fixed["segmented_pairs"] = [[new_eng[i], zh_segs[i]]
                                 for i in range(len(pairs))]
    return True, fixed, (eng_changed or zh_changed)


# ---------------------------------------------------------------------------
# Chinese target punctuation helpers
# ---------------------------------------------------------------------------

def _get_zh_joined(fixed_obj: dict, fmt: str) -> str:
    """Return all Chinese segments concatenated, regardless of format."""
    if fmt == "salami":
        segs = [p[1] for p in fixed_obj.get("segmented_pairs", [])
                if isinstance(p, list) and len(p) == 2]
    else:  # east
        segs = []
        for lat in EAST_LATENCIES:
            if lat in fixed_obj:
                segs.extend(fixed_obj[lat].get("Chinese", []))
    return "".join(str(s) for s in segs)


def check_zh_punct_str(zh_joined: str, manifest: str,
                       zh_excess_threshold: int = 2) -> dict:
    """Same logic as check_zh_punct but takes pre-joined Chinese string.

    Flagging criteria:
      1. en_sent >= 2 and zh_sent == 0  → Chinese missing sentence structure
      2. abs(en_sent - zh_sent) >= 3    → large discrepancy in either direction
      3. zh_sent - en_sent >= zh_excess_threshold
                                        → Chinese has gratuitously many more
                                          sentence-enders than English
                                          (default threshold=2; allows +1 for
                                          legitimate cases like 难道...吗？)
    """
    en_sent = _count_chars(manifest, EN_SENT_END)
    zh_sent = _count_chars(zh_joined, ZH_SENT_END)

    flagged = False
    reason = ""
    if en_sent >= 2 and zh_sent == 0:
        flagged = True
        reason = f"en_sent={en_sent} but zh_sent=0"
    elif abs(en_sent - zh_sent) >= 3:
        flagged = True
        reason = f"en_sent={en_sent} zh_sent={zh_sent} diff={en_sent-zh_sent:+d}"
    elif zh_sent - en_sent >= zh_excess_threshold:
        flagged = True
        reason = f"zh_excess: zh_sent={zh_sent} en_sent={en_sent} excess={zh_sent-en_sent}"

    return {
        "flagged":     flagged,
        "reason":      reason,
        "en_sent_end": en_sent,
        "zh_sent_end": zh_sent,
        "zh_joined":   zh_joined[:120],
    }


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
    ap.add_argument("--sync_zh_punct", action="store_true",
                    help="Heuristically sync Chinese segment punctuation to repaired English segmentation (salami and east)")
    ap.add_argument("--zh_punct_allow_insert", action="store_true",
                    help="When syncing Chinese punct, allow heuristic insertion if no local punctuation can be moved (higher risk)")
    ap.add_argument("--filter_zh_punct", action="store_true",
                    help="Drop utterances where Chinese target is missing sentence-ending punctuation "
                         "(en>=2 sent-ends & zh=0, or |diff|>=3, or zh exceeds en by zh_excess_threshold). "
                         "Applies to both salami and east.")
    ap.add_argument("--zh_excess_threshold", type=int, default=2,
                    help="When --filter_zh_punct is on, also drop utterances where Chinese has this many "
                         "more sentence-enders than English (default=2, allowing +1 for e.g. 难道...吗？). "
                         "Set to 0 to disable the excess check.")
    ap.add_argument("--east_partial", action="store_true",
                    help="EAST: drop only bad-latency keys instead of the whole utterance on mismatch.")
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

    kept_exact = kept_fixed = dropped_token = dropped_no_file = dropped_zh_punct = 0
    kept_ids = []

    # Chinese target punct diagnostics
    zh_punct_flagged_count = 0
    zh_punct_examples = []

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
            ok, fixed_obj, changed = fix_east(
                obj, manifest,
                partial=args.east_partial,
                sync_zh_punct=args.sync_zh_punct,
                zh_allow_insert=args.zh_punct_allow_insert,
            )
        elif fmt == "salami":
            ok, fixed_obj, changed = fix_salami(
                obj, manifest,
                sync_zh_punct=args.sync_zh_punct,
                zh_allow_insert=args.zh_punct_allow_insert,
            )
        else:
            dropped_token += 1
            continue

        if not ok:
            dropped_token += 1
            continue

        # Chinese target punct check (both formats)
        zh_joined = _get_zh_joined(fixed_obj, fmt)
        excess_thr = args.zh_excess_threshold if args.filter_zh_punct else 999
        result = check_zh_punct_str(zh_joined, manifest, zh_excess_threshold=excess_thr)
        if result["flagged"]:
            zh_punct_flagged_count += 1
            if len(zh_punct_examples) < 10:
                zh_punct_examples.append({
                    "utt_id":      utt_id,
                    "fmt":         fmt,
                    "reason":      result["reason"],
                    "en_sent_end": result["en_sent_end"],
                    "zh_sent_end": result["zh_sent_end"],
                    "manifest":    manifest[:120],
                    "zh_joined":   result["zh_joined"],
                })
            if args.filter_zh_punct:
                dropped_zh_punct += 1
                continue

        out_path = os.path.join(args.out_dir, f"{utt_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixed_obj, f, ensure_ascii=False, indent=2)

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
    if args.filter_zh_punct:
        print(f"  Dropped zh-punct issue  : {dropped_zh_punct:,}")
    print("=" * 60)
    print(f"\nFixed JSONs → {args.out_dir}")
    print(f"Good list   → {args.out_good_jsonl}")

    # --- Chinese target punct report ---
    print("\n" + "=" * 60)
    print("CHINESE TARGET PUNCT DIAGNOSTICS")
    filter_note = " [filtered]" if args.filter_zh_punct else " [not filtered — add --filter_zh_punct to drop]"
    print(f"  Flagged (en>=2 sent-ends & zh=0, or |diff|>=3): {zh_punct_flagged_count:,}{filter_note}")
    if zh_punct_examples:
        print(f"\n  Examples (up to 10):")
        for ex in zh_punct_examples:
            print(f"\n  [{ex['utt_id']}] ({ex['fmt']})  {ex['reason']}")
            print(f"    EN manifest : {ex['manifest']}")
            print(f"    ZH joined   : {ex['zh_joined']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
