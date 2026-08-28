#!/usr/bin/env python3
"""
Map Salami-format JSON -> Offline-format JSON (GigaSpeech, recursive dirs).

Input per file (salami):
{
  "segmented_pairs": [[eng, zh], ...],
  "output": "...",
  "input": "...",
  "utt_id": "..."
}
or (from our global-batch Salami runner):
{
  "salami_outputs": [{"segmented_pairs":[[...]], "output":"..."}, ...],
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
"""

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_json_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def collect_pairs_from_item(item: Dict[str, Any]) -> List[List[str]]:
    pairs = item.get("segmented_pairs", None)
    if not isinstance(pairs, list):
        return []
    out: List[List[str]] = []
    for p in pairs:
        if not (isinstance(p, list) and len(p) == 2):
            continue
        e, z = p[0], p[1]
        if not (isinstance(e, str) and isinstance(z, str)):
            continue
        e = e.strip()
        z = z.strip()
        if not e and not z:
            continue
        out.append([e, z])
    return out


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

        pairs: List[List[str]] = []
        salami_output_text = ""

        # Case A: single salami output
        if isinstance(data, dict) and "segmented_pairs" in data:
            pairs = collect_pairs_from_item(data)
            salami_output_text = data.get("output", "")

        # Case B: list of salami outputs per sentence
        elif isinstance(data, dict) and isinstance(data.get("salami_outputs"), list):
            outputs = data.get("salami_outputs", [])
            for item in outputs:
                if not isinstance(item, dict):
                    continue
                pairs.extend(collect_pairs_from_item(item))
            # best-effort: keep joined output if present
            salami_output_text = " ".join(
                [str(x.get("output", "")).strip() for x in outputs if isinstance(x, dict)]
            ).strip()

        if not pairs:
            raise ValueError("missing/invalid segmented_pairs")

        eng: List[str] = []
        zh: List[str] = []
        for e, z in pairs:
            eng.append(e)
            zh.append(z)

        if len(eng) == 0:
            raise ValueError("parsed zero segments from segmented_pairs")
        if len(eng) != len(zh):
            raise ValueError(f"eng/zh length mismatch after parse: {len(eng)} vs {len(zh)}")

        out = {
            "offline": {"English": eng, "Chinese": zh},
            "input": data.get("input", ""),
            "utt_id": data.get("utt_id", os.path.splitext(os.path.basename(in_path))[0]),
            "salami_output": salami_output_text,
        }

        _write_json(out_path, out)
        return True, "mapped"

    except Exception as e:
        _write_json(out_path, {"error": str(e), "source_file": in_path})
        return False, f"failed: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Map salami outputs to offline format (GigaSpeech).")
    ap.add_argument("--input_dir", required=True, help="Root dir of salami outputs (recursive).")
    ap.add_argument("--output_dir", required=True, help="Root dir for offline-format jsons.")
    args = ap.parse_args()

    files = list_json_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No JSON files under: {args.input_dir}")

    total = 0
    ok = 0
    failed = 0

    for in_path in tqdm(files, desc="Mapping"):
        rel = os.path.relpath(in_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel)
        total += 1
        success, _ = map_one_file(in_path, out_path, copy_errors=True)
        if success:
            ok += 1
        else:
            failed += 1

    print(f"\n✅ Done: total={total} ok={ok} failed={failed}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
