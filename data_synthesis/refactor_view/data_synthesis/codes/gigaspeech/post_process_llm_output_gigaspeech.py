#!/usr/bin/env python3
"""
Post-process GigaSpeech LLM outputs:
- Merge one-word English chunks with neighboring chunks.
- Keep English/Chinese array lengths aligned.
- If sentence_spans exists, merge inside each sentence slice and update spans.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one-word chunks in GigaSpeech LLM output JSON files."
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


def merge_single_word_chunks(
    eng_list: List[str],
    zh_list: List[str],
) -> Tuple[List[str], List[str], bool]:
    """
    Merge one-word chunks while preserving index alignment.
    Strategy:
    - If single-word chunk is not the first output chunk, merge into previous.
    - If it is the first output chunk and next exists, merge with next.
    """
    if len(eng_list) != len(zh_list):
        raise ValueError(f"Length mismatch: English={len(eng_list)} Chinese={len(zh_list)}")

    if not eng_list:
        return eng_list, zh_list, False

    out_eng: List[str] = []
    out_zh: List[str] = []
    changed = False

    i = 0
    n = len(eng_list)
    while i < n:
        e = str(eng_list[i]).strip()
        z = str(zh_list[i]).strip()

        if is_single_word_chunk(e):
            # Normal case: merge to previous chunk.
            if out_eng:
                out_eng[-1] = (out_eng[-1] + " " + e).strip()
                out_zh[-1] = (out_zh[-1] + z).strip()
                changed = True
                i += 1
                continue
            # Edge case: first chunk is single-word -> merge with next chunk.
            if i + 1 < n:
                e2 = str(eng_list[i + 1]).strip()
                z2 = str(zh_list[i + 1]).strip()
                out_eng.append((e + " " + e2).strip())
                out_zh.append((z + z2).strip())
                changed = True
                i += 2
                continue

        out_eng.append(e)
        out_zh.append(z)
        i += 1

    return out_eng, out_zh, changed


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


def process_level_with_spans(data: Dict[str, Any], level: str) -> bool:
    level_obj = data.get(level)
    spans_obj = data.get("sentence_spans", {}).get(level)
    if not isinstance(level_obj, dict) or not valid_spans(spans_obj, len(level_obj.get("English", []))):
        return False

    eng_all = level_obj.get("English", [])
    zh_all = level_obj.get("Chinese", [])
    if not isinstance(eng_all, list) or not isinstance(zh_all, list):
        return False
    if len(eng_all) != len(zh_all):
        return False

    merged_eng_all: List[str] = []
    merged_zh_all: List[str] = []
    new_spans: List[List[int]] = []
    cursor = 0
    changed_any = False

    for start, end in spans_obj:
        sent_eng = eng_all[start:end]
        sent_zh = zh_all[start:end]
        merged_eng, merged_zh, changed = merge_single_word_chunks(sent_eng, sent_zh)
        changed_any = changed_any or changed

        new_spans.append([cursor, cursor + len(merged_eng)])
        merged_eng_all.extend(merged_eng)
        merged_zh_all.extend(merged_zh)
        cursor += len(merged_eng)

    data[level]["English"] = merged_eng_all
    data[level]["Chinese"] = merged_zh_all
    data["sentence_spans"][level] = new_spans
    return changed_any


def process_json_obj(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if "error" in data:
        return data, False

    updated = copy.deepcopy(data)
    changed_any = False

    for level in LATENCY_LEVELS:
        if level not in updated:
            continue
        level_obj = updated[level]
        if not isinstance(level_obj, dict):
            continue
        eng = level_obj.get("English")
        zh = level_obj.get("Chinese")
        if not isinstance(eng, list) or not isinstance(zh, list):
            continue

        # Preferred path for GigaSpeech output: merge per sentence and update spans.
        if (
            level != "offline"
            and isinstance(updated.get("sentence_spans"), dict)
            and level in updated["sentence_spans"]
        ):
            changed = process_level_with_spans(updated, level)
            changed_any = changed_any or changed
            continue

        # Fallback for files without sentence_spans.
        merged_eng, merged_zh, changed = merge_single_word_chunks(eng, zh)
        updated[level]["English"] = merged_eng
        updated[level]["Chinese"] = merged_zh
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