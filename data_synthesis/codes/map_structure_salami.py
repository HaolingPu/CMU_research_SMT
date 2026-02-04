#!/usr/bin/env python3
"""
Map Salami-format JSON -> Offline-format JSON.

Input per file (salami):
{
  "segmented_pairs": [[eng, zh], ...],
  "output": "...",
  "input": "...",
  "utt_id": "..."
}

Output per file (offline):
{
  "offline": {"English": [...], "Chinese": [...]},
  "input": "...",
  "utt_id": "...",
  "salami_output": "..."   # optional debug
}

Directory layout:
  input_dir/lang/00000000/utt_....json
  output_dir/lang/00000000/utt_....json
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm
from typing import Tuple, Optional


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def map_one_file(in_path: str, out_path: str, copy_errors: bool = True) -> Tuple[bool, str]:
    """
    Returns: (success, status_msg)
    """
    try:
        data = _read_json(in_path)

        # If upstream wrote an error json, keep it so downstream doesn't break.
        if isinstance(data, dict) and "error" in data:
            if copy_errors:
                _safe_mkdir(os.path.dirname(out_path))
                shutil.copy2(in_path, out_path)
            return True, "copied_error"

        pairs = data.get("segmented_pairs", None)
        if not isinstance(pairs, list):
            raise ValueError("missing/invalid segmented_pairs (expected list)")

        eng: list[str] = []
        zh: list[str] = []

        for item in pairs:
            # each should be [eng, zh]
            if not (isinstance(item, list) and len(item) == 2):
                continue
            e, z = item[0], item[1]
            if not (isinstance(e, str) and isinstance(z, str)):
                continue

            e = e.strip()
            z = z.strip()

            # skip empty segments
            if not e and not z:
                continue

            eng.append(e)
            zh.append(z)

        if len(eng) == 0:
            raise ValueError("parsed zero segments from segmented_pairs")
        if len(eng) != len(zh):
            raise ValueError(f"eng/zh length mismatch after parse: {len(eng)} vs {len(zh)}")

        out = {
            "offline": {
                "English": eng,
                "Chinese": zh,
            },
            "input": data.get("input", ""),
            "utt_id": data.get("utt_id", os.path.splitext(os.path.basename(in_path))[0]),
            # keep the final connected translation for debug/reference
            "salami_output": data.get("output", ""),
        }

        _write_json(out_path, out)
        return True, "mapped"

    except Exception as e:
        # Write a structured error so downstream can still proceed
        _write_json(
            out_path,
            {
                "error": str(e),
                "source_file": in_path,
            },
        )
        return False, f"failed: {e}"


def process_language(input_dir: str, output_dir: str, lang: str, only_parquet: Optional[str]) -> None:
    in_lang = os.path.join(input_dir, lang)
    out_lang = os.path.join(output_dir, lang)

    if not os.path.isdir(in_lang):
        raise FileNotFoundError(f"Input lang dir not found: {in_lang}")

    _safe_mkdir(out_lang)

    parquet_dirs = sorted(
        d for d in os.listdir(in_lang)
        if os.path.isdir(os.path.join(in_lang, d)) and d.isdigit() and len(d) == 8
    )

    if only_parquet is not None:
        parquet_dirs = [d for d in parquet_dirs if d == only_parquet]
        if len(parquet_dirs) == 0:
            raise ValueError(f"--only_parquet={only_parquet} not found under {in_lang}")

    total = 0
    ok = 0
    failed = 0

    for pq in parquet_dirs:
        in_pq = os.path.join(in_lang, pq)
        out_pq = os.path.join(out_lang, pq)
        _safe_mkdir(out_pq)

        files = sorted(f for f in os.listdir(in_pq) if f.endswith(".json"))
        print(f"📁 {lang}/{pq}: {len(files)} files")

        for fn in tqdm(files, desc=f"{lang}_{pq}"):
            in_path = os.path.join(in_pq, fn)
            out_path = os.path.join(out_pq, fn)

            total += 1
            success, _ = map_one_file(in_path, out_path, copy_errors=True)
            if success:
                ok += 1
            else:
                failed += 1

    print(f"\n✅ Done mapping {lang}: total={total} ok={ok} failed={failed}")
    print(f"Output saved to: {out_lang}")


def main():
    ap = argparse.ArgumentParser(description="Map salami outputs to offline segmentation format.")
    ap.add_argument("--input_dir", required=True, help="Root dir of llm_output_SALAMI")
    ap.add_argument("--output_dir", required=True, help="Root dir to write offline-format jsons")
    ap.add_argument("--lang", required=True, help="e.g. en000")
    ap.add_argument("--only_parquet", default=None, help="Debug: only run one parquet, e.g. 00000000")
    args = ap.parse_args()

    process_language(args.input_dir, args.output_dir, args.lang, args.only_parquet)


if __name__ == "__main__":
    main()
