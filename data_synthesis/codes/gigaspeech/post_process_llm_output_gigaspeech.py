#!/usr/bin/env python3
"""
Post-process GigaSpeech LLM outputs:
- Merge one-word Source chunks with neighboring chunks.
- Keep Source/Target array lengths aligned.
- If sentence_spans exists, merge inside each sentence slice and update spans.

Target-side join separator depends on target_lang:
- de: space (German uses inter-word spaces)
- zh / ja / others: no separator
"""

import argparse
import copy
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


LATENCY_LEVELS = ["low_latency", "medium_latency", "high_latency", "offline"]

# Target languages whose written form uses inter-word spaces.
SPACE_JOIN_LANGS = {"de", "en", "fr", "es"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one-word Source chunks in GigaSpeech LLM output JSON files."
    )
    parser.add_argument("--input-dir", required=True, help="Input directory containing JSON files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed JSON files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output-dir. Default: skip existing files.",
    )
    return parser.parse_args()


def is_single_word_chunk(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text))
    return len(words) <= 1


def target_join_sep(target_lang: Optional[str]) -> str:
    if target_lang and target_lang.lower() in SPACE_JOIN_LANGS:
        return " "
    return ""


def merge_single_word_chunks(
    src_list: List[str],
    tgt_list: List[str],
    tgt_sep: str,
) -> Tuple[List[str], List[str], bool]:
    """
    Merge one-word Source chunks while preserving index alignment.
    Strategy:
    - If single-word chunk is not the first output chunk, merge into previous.
    - If it is the first output chunk and next exists, merge with next.

    The Target side uses `tgt_sep` for joining (space for de, empty for zh/ja).
    """
    if len(src_list) != len(tgt_list):
        raise ValueError(f"Length mismatch: Source={len(src_list)} Target={len(tgt_list)}")

    if not src_list:
        return src_list, tgt_list, False

    out_src: List[str] = []
    out_tgt: List[str] = []
    changed = False

    i = 0
    n = len(src_list)
    while i < n:
        s = str(src_list[i]).strip()
        t = str(tgt_list[i]).strip()

        if is_single_word_chunk(s):
            # Normal case: merge to previous chunk.
            if out_src:
                out_src[-1] = (out_src[-1] + " " + s).strip()
                out_tgt[-1] = (out_tgt[-1] + tgt_sep + t).strip()
                changed = True
                i += 1
                continue
            # Edge case: first chunk is single-word -> merge with next chunk.
            if i + 1 < n:
                s2 = str(src_list[i + 1]).strip()
                t2 = str(tgt_list[i + 1]).strip()
                out_src.append((s + " " + s2).strip())
                out_tgt.append((t + tgt_sep + t2).strip())
                changed = True
                i += 2
                continue

        out_src.append(s)
        out_tgt.append(t)
        i += 1

    return out_src, out_tgt, changed


def valid_spans(spans: Any, total_len: int) -> bool:
    if not isinstance(spans, list):
        return False
    prev_end = 0
    for item in spans:
        if not (isinstance(item, list) and len(item) == 2):
            return False
        start, end = item
        if not (isinstance(start, int) and isinstance(end, int)):
            return False
        if start < prev_end or end < start:
            return False
        prev_end = end
    return prev_end <= total_len


def process_level_with_spans(data: Dict[str, Any], level: str, tgt_sep: str) -> bool:
    level_obj = data.get(level)
    spans_obj = data.get("sentence_spans", {}).get(level)
    if not isinstance(level_obj, dict) or not valid_spans(spans_obj, len(level_obj.get("Source", []))):
        return False

    src_all = level_obj.get("Source", [])
    tgt_all = level_obj.get("Target", [])
    if not isinstance(src_all, list) or not isinstance(tgt_all, list):
        return False
    if len(src_all) != len(tgt_all):
        return False

    merged_src_all: List[str] = []
    merged_tgt_all: List[str] = []
    new_spans: List[List[int]] = []
    cursor = 0
    changed_any = False

    for start, end in spans_obj:
        sent_src = src_all[start:end]
        sent_tgt = tgt_all[start:end]
        merged_src, merged_tgt, changed = merge_single_word_chunks(sent_src, sent_tgt, tgt_sep)
        changed_any = changed_any or changed

        new_spans.append([cursor, cursor + len(merged_src)])
        merged_src_all.extend(merged_src)
        merged_tgt_all.extend(merged_tgt)
        cursor += len(merged_src)

    data[level]["Source"] = merged_src_all
    data[level]["Target"] = merged_tgt_all
    data["sentence_spans"][level] = new_spans
    return changed_any


def process_json_obj(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if "error" in data:
        return data, False

    updated = copy.deepcopy(data)
    changed_any = False

    target_lang = updated.get("target_lang") or updated.get("tgt_lang")
    tgt_sep = target_join_sep(target_lang)

    for level in LATENCY_LEVELS:
        if level not in updated:
            continue
        level_obj = updated[level]
        if not isinstance(level_obj, dict):
            continue
        src = level_obj.get("Source")
        tgt = level_obj.get("Target")
        if not isinstance(src, list) or not isinstance(tgt, list):
            continue

        # Preferred path for GigaSpeech output: merge per sentence and update spans.
        if (
            level != "offline"
            and isinstance(updated.get("sentence_spans"), dict)
            and level in updated["sentence_spans"]
        ):
            changed = process_level_with_spans(updated, level, tgt_sep)
            changed_any = changed_any or changed
            continue

        # Fallback for files without sentence_spans.
        merged_src, merged_tgt, changed = merge_single_word_chunks(src, tgt, tgt_sep)
        updated[level]["Source"] = merged_src
        updated[level]["Target"] = merged_tgt
        changed_any = changed_any or changed

    return updated, changed_any


def list_json_files(input_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.endswith(".json"):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def process_file(input_path: str, output_path: str) -> Tuple[bool, bool, Optional[str]]:
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            shutil.copy2(input_path, output_path)
            return True, False, "non-dict json, copied"

        updated, changed = process_json_obj(data)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)
        return True, changed, None
    except Exception as e:
        return False, False, str(e)


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    files = list_json_files(args.input_dir)
    if not files:
        print(f"No JSON files found under: {args.input_dir}")
        return

    total = 0
    changed_cnt = 0
    copied_cnt = 0
    skipped_existing = 0
    failed = 0

    for input_path in tqdm(files, desc="Post-process"):
        rel = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel)
        ensure_parent(output_path)

        if os.path.exists(output_path) and not args.overwrite:
            skipped_existing += 1
            continue

        ok, changed, note = process_file(input_path, output_path)
        total += 1
        if not ok:
            failed += 1
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "error": note,
                        "source_file": input_path,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            continue
        if changed:
            changed_cnt += 1
        elif note is not None:
            copied_cnt += 1

    print("\n========== Done ==========")
    print(f"Input dir         : {args.input_dir}")
    print(f"Output dir        : {args.output_dir}")
    print(f"JSON processed    : {total}")
    print(f"Changed           : {changed_cnt}")
    print(f"Copied/unchanged  : {copied_cnt}")
    print(f"Skipped existing  : {skipped_existing}")
    print(f"Failed            : {failed}")


if __name__ == "__main__":
    main()
