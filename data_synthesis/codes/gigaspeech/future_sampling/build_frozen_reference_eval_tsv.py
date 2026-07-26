#!/usr/bin/env python3
"""Build a frozen evaluation TSV with a stable reference column.

The output TSV preserves the original manifest columns and appends:
  - llm_reference_text
  - reference_source
  - reference_chars

Later evaluation runs can use this TSV directly so every method is scored
against the exact same reference text.
"""

import argparse
import ast
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_TRANSLATION_CACHE_DIR = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "llm_full_translation_cache/train_xl_case_robust_asr_filtered"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a frozen eval TSV with stable references.")
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-tsv", required=True)
    p.add_argument("--translation-cache-dir", default=DEFAULT_TRANSLATION_CACHE_DIR)
    p.add_argument("--id-column", default="id")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--require-reference", action="store_true",
                   help="Fail if any row cannot be assigned a reference.")
    return p.parse_args()


def parse_list_column(raw: Any) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)] if str(parsed).strip() else []


def extract_reference_from_row(row: Dict[str, str]) -> Optional[str]:
    for key in ("llm_reference_text", "tgt_text_full", "tgt_text", "target_text", "translation", "ref_text", "reference"):
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        vals = parse_list_column(raw)
        if vals:
            text = "".join(str(v).strip() for v in vals if str(v).strip())
        else:
            text = str(raw).strip()
        if text:
            return text
    return None


def load_translation_cache(cache_dir: str) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not cache_dir or not os.path.isdir(cache_dir):
        return cache
    jsonl_files = sorted(glob.glob(os.path.join(cache_dir, "task_*.jsonl")))
    print(f"[Cache] Loading translation cache from {cache_dir} ({len(jsonl_files)} files) ...")
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                utt_id = str(entry.get("utt_id", "")).strip()
                ref = str(entry.get("llm_full_translation", "")).strip()
                if utt_id and ref:
                    cache[utt_id] = ref
    print(f"[Cache] Loaded {len(cache)} entries.")
    return cache


def main() -> None:
    args = parse_args()
    if os.path.exists(args.output_tsv) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_tsv}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_tsv)), exist_ok=True)
    cache = load_translation_cache(args.translation_cache_dir)

    with open(args.input_tsv, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in TSV: {args.input_tsv}")
        fieldnames = list(reader.fieldnames)
        for extra in ("llm_reference_text", "reference_source", "reference_chars"):
            if extra not in fieldnames:
                fieldnames.append(extra)

        total = written = missing = 0
        with open(args.output_tsv, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                total += 1
                utt_id = str(row.get(args.id_column, "")).strip()
                ref = cache.get(utt_id, "")
                source = "cache"
                if not ref:
                    ref = extract_reference_from_row(row) or ""
                    source = "manifest_reference" if ref else ""
                if not ref:
                    missing += 1
                    if args.require_reference:
                        raise ValueError(f"Missing reference for utt_id={utt_id or '<empty>'}")
                row["llm_reference_text"] = ref
                row["reference_source"] = source
                row["reference_chars"] = str(len(ref.replace(" ", ""))) if ref else "0"
                writer.writerow(row)
                written += 1

    print(
        f"[Done] wrote {written} rows to {args.output_tsv} | "
        f"missing_reference={missing} / {total}"
    )


if __name__ == "__main__":
    main()
